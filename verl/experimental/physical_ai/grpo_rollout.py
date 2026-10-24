"""普通GRPO的独立完整轨迹采集；复用原生动作/QA执行器，不调用SegGRPO父选择。"""

import hashlib
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path


def collect_group(policy, processor, config, output, *, policy_version, task=None, resources=None):
    """每条轨迹独立执行后沿自身会话问一次QA，返回n条带单一回报的完整序列。

    输出的generated/old_logprobs覆盖动作和QA，来自同次采样，不重复训练动作前缀。
    GT只给评分端；不可交付/未知GT按已授权规则在动作末结束，其他异常传播。
    每个环境由生成器with管理，批量调度关闭时也可靠释放，不跨线程传模拟器。
    """
    from benchmark.adapters.libero_execution import assert_same_preparation
    from benchmark.adapters.libero_success import TaskSuccessMonitor
    from benchmark.scoring.llm_judge import JudgeClient
    from experiments.libero.libero_eval_utils import build_libero_raw_obs

    from .batched_sampling import run_phases
    from .g05_rollout import phase_steps
    from .grpo import combined_reward, qa_gate
    from .native_runtime import live_chat_class
    from .online_reward import bind_delivery, save_json, score_answer
    from .sampling import fork_chat
    from .seggrpo import Segment
    from .task_data import (
        SHARED_ACTION_RULE,
        create_scene_capture,
        open_task_session,
        task_for_rollout,
        task_reward_at_delivery,
    )

    if task is None or resources is None or config["task_rule"]["version"] != SHARED_ACTION_RULE:
        raise ValueError("普通GRPO仅接已绑定共同成功规则的真实训练任务")
    if config["sampling_mode"] != "cached" or config["compute_dtype"] != "float32":
        raise ValueError("普通GRPO复用已核验FP32缓存采样")
    question = config.get("question")
    n = config["group_size"]
    batch = config.get("rollout_batch_size", n)
    if type(batch) is not int or batch < 1:
        raise ValueError("rollout_batch_size必须为正整数")
    LiveChat = live_chat_class()
    first_preparation = None
    condition = None
    finished = 0
    client = JudgeClient(Path(os.environ["PHYSICAL_AI_JUDGE_ACCESS_FILE"]), model=config["judge_model"], timeout_s=90)

    def trajectory(i):
        """一条独立动作→QA经历；生成边界yield给公共batch调度器，共享权重但不共享会话。"""
        nonlocal first_preparation, condition, finished
        identity = f"trajectory-{i}"
        directory = Path(output) / identity
        with open_task_session(
            task_for_rollout(task, output, identity),
            resources,
            seed=config["environment_seed"],
            max_steps=config["execution_steps"] + config["qa_steps"],
        ) as session:
            capture = create_scene_capture(
                session, task, resources, Path(output) / "scene" / identity, rollout_id=identity
            )
            reader, recorder = capture.reader, capture.recorder
            session.success_monitor = TaskSuccessMonitor(session.env)
            session.success_monitor.observe(0)
            digest = hashlib.sha256(session.env.sim.get_state().flatten().tobytes()).hexdigest()
            if first_preparation is None:
                first_preparation = deepcopy(session.preparation)
                condition = digest
                save_json(
                    Path(output) / "initial-rule.json",
                    dict(initial_state_sha256=digest, rule=config["task_rule"], scene_config=reader.config),
                )
            else:
                assert_same_preparation(first_preparation, session.preparation)
                if digest != condition:
                    raise ValueError("组内初态不同")
            chat = LiveChat(policy, processor, **({"thinking": True} if config.get("thinking", False) else {}))
            chat.rl_processor = processor
            chat.rl_sampling_mode = config["sampling_mode"]
            chat.rl_compute_dtype = config["compute_dtype"]
            chat.start(session.task, build_libero_raw_obs(session.observation, session.task)[0])
            execution = yield from phase_steps(
                session,
                chat,
                phase="execution",
                env_steps=config["execution_steps"],
                token_budget=config["execution_token_budget"],
                seed=config["seed"] + i,
                directory=directory / "execution",
                scene_capture=capture,
                max_calls=config.get("execution_max_calls"),
                per_message_budget=config.get("execution_token_scope") == "message",
                timeout_seconds=config["generation_timeout_seconds"],
                allow_budget_truncation=True,
            )
            task_result = task_reward_at_delivery(
                task, config["task_rule"], session, reader, recorder, execution["termination"]
            )
            binding = (
                bind_delivery(question, recorder, config["entity_names"], identity)
                if question is not None and not execution.get("incomplete_message")
                else None
            )
            save_json(directory / "scene-map.json", recorder.export())
            save_json(
                directory / "task-success.json",
                dict(contract=session.success_monitor.contract, rows=deepcopy(session.success_monitor.rows)),
            )
            save_json(directory / "delivery.json", dict(task=task_result, binding=binding))
            status = qa_gate(question, execution, binding)
            qa_reward = None
            artifact = dict(execution)
            artifact["inputs_path"] = str(directory / "execution/inputs.pt")
            if status == "deliver":
                # fork_chat仅复制同一条轨迹的原会话用于容量预检；不新建环境或另选父经历。
                candidate = fork_chat(chat)
                candidate.question(question["question"], pending_assistant=execution["pending_assistant"])
                if len(candidate.ids) >= int(policy.model.cfg.vlm.max_position_embeddings):
                    status = "context_capacity"
                else:
                    answer = yield from phase_steps(
                        session,
                        candidate,
                        phase="qa",
                        env_steps=config["qa_steps"],
                        token_budget=config["qa_token_budget"],
                        seed=config["seed"] + 100 + i,
                        directory=directory / "qa",
                        generated=execution["generated"],
                        old_logprobs=execution["old_logprobs"],
                        max_calls=config.get("qa_max_calls"),
                        timeout_seconds=config["generation_timeout_seconds"],
                        allow_budget_truncation=True,
                    )
                    judgment = score_answer(binding, answer["answer"], client, rollout_id=identity)
                    save_json(directory / "judge.json", judgment)
                    if judgment["status"] != "scored":
                        raise ValueError("QA裁判无有效分，不能当作无QA")
                    qa_reward = float(judgment["score"])
                    status = "scored"
                    artifact = dict(answer)
                    artifact["inputs_path"] = str(directory / "qa/inputs.pt")
                    # QA采集保存完整生成mask和原概率，但其阶段边界需扩大为整条真实会话。
                    artifact["prefix_length"] = execution["prefix_length"]
                    artifact["generated_tokens"] = sum(artifact["generated"])
            reward = combined_reward(task_result["task_reward"], qa_reward, qa_status=status)
            artifact.update(
                phase="trajectory", task_reward=task_result["task_reward"], qa_reward=qa_reward, qa_status=status
            )
            record = Segment(
                identity,
                "trajectory",
                "trajectory-group",
                condition,
                config["case_id"],
                policy_version,
                n,
                artifact["prefix_length"] - 1,
                artifact["end_length"] - 1,
                reward,
                "scored",
            )
            save_json(
                directory / "reward.json",
                dict(
                    reward=reward,
                    task_reward=task_result["task_reward"],
                    qa_reward=qa_reward,
                    qa_status=status,
                    task_weight=0.5 if status == "scored" else 1.0,
                    qa_weight=0.5 if status == "scored" else 0.0,
                ),
            )
            save_json(directory / "trajectory.json", artifact)
            finished += 1
            save_json(Path(output) / "progress.json", dict(phase="trajectory", finished=finished, planned=n))
            return record, artifact

    result = run_phases([trajectory(i) for i in range(n)], batch)
    records, artifacts = map(list, zip(*result, strict=True))
    save_json(Path(output) / "groups.json", [asdict(r) for r in records])
    return records, artifacts
