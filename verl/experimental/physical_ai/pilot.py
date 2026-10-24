# Copyright 2026 Physical AI contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""真实原生G0.5/模拟器/在线judge/verl GRPO loss的单卡小组诊断。"""

import argparse
import hashlib
import json
import os
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from .workspace import configure_workspace

ROOT = configure_workspace()


def collect_group(policy, processor, config, output, *, policy_version, task=None, resources=None):
    """采集同版策略的完整动作/QA组；小试与连续训练共用同一会话和评分实现。

    output为新的组目录，config已绑定一道训练任务/原问题与独立动作奖励。
    每个控制步采集真实图；均匀选择父经历后建立独立QA环境副本。返回Segment及
    原始token/mask/概率附件索引；未知GT或judge错误整组抛错，不丢样本缩组。
    """
    from benchmark.adapters.libero_scene_map import LiberoSceneMap
    from benchmark.core.scene_graph import SceneRecorder
    from benchmark.scoring.llm_judge import JudgeClient
    from experiments.libero.libero_eval_utils import build_libero_raw_obs
    from libero.libero import benchmark

    from .native_runtime import live_chat_class

    LiveChat = live_chat_class()

    from .batched_sampling import run_phases
    from .g05_rollout import chat_fingerprint, phase_steps
    from .libero_session import LiberoSession
    from .online_reward import bind_delivery, qa_reward_status, save_json, score_answer
    from .sampling import fork_chat, sample_response
    from .seggrpo import Segment, select_parent
    from .task_data import (
        EVALUATION_ACTION_RULE,
        SHARED_ACTION_RULE,
        create_scene_capture,
        open_task_session,
        task_for_rollout,
        task_reward_at_delivery,
    )

    question = config["question"]
    if type(config.get("thinking", False)) is not bool:
        raise ValueError("thinking须为显式布尔值")
    task_index = None
    if task is None:
        suite = benchmark.get_benchmark_dict()[config["suite"]]()
        task_index = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == config["task_name"])
    records, artifacts, parents = [], [], []
    batch_size = config.get("rollout_batch_size", 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("rollout_batch_size须为正整数")
    if batch_size > 1 and config["sampling_mode"] != "cached":
        raise ValueError("组内批量生成要求已经验收的缓存模式")

    def drive(generators):
        """串行旧配置与批量新配置共用物理/评分生成器，仅选择生成请求的计算后端。"""
        if config["sampling_mode"] == "cached":
            return run_phases(generators, batch_size)
        return run_phases(generators, 1, sample_batch=lambda requests: [sample_response(**r) for r in requests])

    # 所有父环境保留到选中一个，再为每条QA建立独立副本；退出后统一释放再训练。
    with ExitStack() as stack:
        initial_digest = None
        initial_preparation = None
        finished_execution = 0

        def execution(i):
            """一条真实动作分支；首次yield前初始化独立场景，结束后冻结该分支的GT。"""
            nonlocal initial_digest, initial_preparation, finished_execution
            rollout_id = f"execution-{i}"
            session = stack.enter_context(
                open_task_session(
                    task_for_rollout(task, output, rollout_id),
                    resources,
                    seed=config["environment_seed"],
                    max_steps=config["execution_steps"] + config["qa_steps"],
                )
                if task is not None
                else LiberoSession(
                    config["suite"],
                    task_index,
                    initial_state_index=config["initial_state_index"],
                    seed=config["environment_seed"],
                    max_steps=config["execution_steps"] + config["qa_steps"],
                )
            )
            capture = (
                create_scene_capture(session, task, resources, output / "scene" / rollout_id, rollout_id=rollout_id)
                if task is not None
                else None
            )
            reader = capture.reader if capture is not None else LiberoSceneMap(session.env)
            if (
                config["task_rule"]["version"] not in {EVALUATION_ACTION_RULE, SHARED_ACTION_RULE}
                and config["task_rule"]["calibration"] != reader.config["version"]
            ):
                raise ValueError("任务状态规则与实际Scene map版本不同")
            recorder = (
                capture.recorder
                if capture is not None
                else SceneRecorder(rollout_id, max_gap_s=reader.config["max_sample_gap_s"])
            )
            if capture is None:
                recorder.observe(reader.read(0))
            if task is None and question["parameters"]["query"]["entity"] not in reader.nodes:
                raise ValueError("题目实体未绑定到真实场景")
            if config["task_rule"]["version"] == SHARED_ACTION_RULE:
                from benchmark.adapters.libero_success import TaskSuccessMonitor

                session.success_monitor = TaskSuccessMonitor(session.env, family="libero")
                session.success_monitor.observe(0)
                save_json(output / "scene" / rollout_id / "success-contract.json", session.success_monitor.contract)
            condition = hashlib.sha256(session.env.sim.get_state().flatten().tobytes()).hexdigest()
            # 初态在真实rollout现场核验：同组分支比较完整准备/控制状态，不需逐样本离线预演。
            from copy import deepcopy

            from benchmark.adapters.libero_execution import assert_same_preparation

            if initial_preparation is None:
                initial_preparation = deepcopy(session.preparation)
            else:
                assert_same_preparation(initial_preparation, session.preparation)
            if initial_digest is None:
                initial_digest = condition
                save_json(
                    output / "initial-rule.json",
                    dict(scene_config=reader.config, initial_state_sha256=condition, rule=config["task_rule"]),
                )
            elif condition != initial_digest:
                raise ValueError("动作组初态不一致")
            chat = LiveChat(policy, processor, **({"thinking": True} if config.get("thinking", False) else {}))
            chat.rl_processor = processor
            chat.rl_sampling_mode = config["sampling_mode"]
            chat.rl_compute_dtype = config["compute_dtype"]
            chat.start(session.task, build_libero_raw_obs(session.observation, session.task)[0])
            directory = output / rollout_id
            phase = yield from phase_steps(
                session,
                chat,
                phase="execution",
                env_steps=config["execution_steps"],
                token_budget=config["execution_token_budget"],
                seed=config["seed"] + i,
                directory=directory,
                reader=reader,
                recorder=recorder,
                scene_capture=capture,
                max_calls=config.get("execution_max_calls"),
                per_message_budget=config.get("execution_token_scope") == "message",
                timeout_seconds=config.get("generation_timeout_seconds", 180),
            )
            # 训练任务的QA原封不动复用；动作奖励读独立目标，不能把QA答案当任务完成。
            task_result = (
                task_reward_at_delivery(task, config["task_rule"], session, reader, recorder, phase["termination"])
                if task
                else None
            )
            binding = bind_delivery(question, recorder, config["entity_names"], rollout_id)
            private_history = recorder.export()
            save_json(directory / "delivery.json", private_history["questions"][question["question_id"]])
            save_json(directory / "scene-map.json", private_history)
            if task_result is None:
                state = binding["gt"]["value"]
                if state not in ["open", "closed", "partially_open"]:
                    raise ValueError("设备状态不符合已冻结规则")
                task_result = dict(task_reward=float(state == config["task_rule"]["target_state"]))
            reward = task_result["task_reward"]
            record = Segment(
                rollout_id,
                "execution",
                "execution-group",
                initial_digest,
                config["case_id"],
                policy_version,
                config["execution_group_size"],
                phase["prefix_length"] - 1,
                phase["end_length"] - 1,
                reward,
                "scored",
            )
            phase["inputs_path"] = str(directory / "inputs.pt")
            if config["task_rule"]["version"] == SHARED_ACTION_RULE:
                save_json(
                    directory / "task-success.json",
                    dict(
                        contract=session.success_monitor.contract,
                        rows=session.success_monitor.rows,
                    ),
                )
            save_json(
                directory / "reward.json",
                {**task_result, "binding": binding, "native_success": session.audit()["native_success"]},
            )
            finished_execution += 1
            save_json(
                output / "progress.json",
                dict(phase="execution", finished=finished_execution, planned=config["execution_group_size"]),
            )
            return record, phase, (session, chat, phase, binding)

        for start in range(0, config["execution_group_size"], batch_size):
            results = drive(execution(i) for i in range(start, min(start + batch_size, config["execution_group_size"])))
            for record, phase, parent in results:
                records.append(record)
                artifacts.append(phase)
                parents.append(parent)
        selected = select_parent(records, seed=config["seed"], mode="uniform")
        parent_index = next(i for i, r in enumerate(records) if r.rollout_id == selected)
        session, chat, phase, binding = parents[parent_index]
        reward_status = qa_reward_status(binding, config.get("unknown_qa_policy", "stop"))
        # 未选中的真实轨迹/GT已经落盘，之后不再执行这些环境；及时释放其EGL资源。
        # 只关闭环境，不删除原会话/动作记录，也不按奖励改变父轨迹选择。
        for index, parent in enumerate(parents):
            if index != parent_index:
                parent[0].close()
        # 渲染器的显存不归PyTorch allocator管理；归还空闲块供接下来的QA副本使用。
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        base_chat = fork_chat(chat)
        base_chat.question(question["question"], pending_assistant=phase["pending_assistant"])
        prefix_hash = chat_fingerprint(base_chat)
        qa_condition = hashlib.sha256((prefix_hash + binding["binding_sha256"]).encode()).hexdigest()
        save_json(
            output / "selection.json",
            dict(
                parent_rollout_id=selected,
                parent_reward=records[parent_index].reward,
                selection="uniform",
                seed=config["seed"],
                prefix_sha256=prefix_hash,
                condition_id=qa_condition,
                qa_reward_status=reward_status,
                gt_status=binding["gt"]["status"],
                gt_reason=binding["gt"].get("reason"),
            ),
        )
        client = JudgeClient(
            Path(os.environ["PHYSICAL_AI_JUDGE_ACCESS_FILE"]), model=config["judge_model"], timeout_s=90
        )
        finished_qa = 0

        def qa(i):
            """同一真实父经历下的独立QA分支；冻结GT共用，生成和后续动作各自独立。"""
            nonlocal finished_qa
            child_id = f"qa-{i}"
            with session.fork() as child:
                child_chat = fork_chat(base_chat)
                if chat_fingerprint(child_chat) != prefix_hash:
                    raise ValueError("QA分支会话前缀不同")
                directory = output / child_id
                save_json(output / f"{child_id}-fork.json", child.fork_validation)
                answer = yield from phase_steps(
                    child,
                    child_chat,
                    phase="qa",
                    env_steps=config["qa_steps"],
                    token_budget=config["qa_token_budget"],
                    seed=config["seed"] + 100 + i,
                    directory=directory,
                    generated=phase["generated"],
                    old_logprobs=phase["old_logprobs"],
                    max_calls=config.get("qa_max_calls"),
                    timeout_seconds=config.get("generation_timeout_seconds", 180),
                )
                judgment = score_answer(binding, answer["answer"], client, rollout_id=selected)
                save_json(directory / "judge.json", judgment)
                if reward_status == "scored" and judgment["status"] != "scored":
                    raise ValueError("judge未给有效分数，整个组保留待恢复，禁止更新")
                if reward_status != "scored" and (judgment["status"] != "excluded_gt" or judgment["score"] is not None):
                    raise ValueError("未知GT的QA不得调用裁判或产生分数")
                record = Segment(
                    child_id,
                    "qa",
                    "qa-group",
                    qa_condition,
                    config["case_id"],
                    policy_version,
                    config["qa_group_size"],
                    answer["prefix_length"] - 1,
                    answer["end_length"] - 1,
                    float(judgment["score"]) if reward_status == "scored" else None,
                    reward_status,
                    parent_rollout_id=selected,
                )
                answer["inputs_path"] = str(directory / "inputs.pt")
                finished_qa += 1
                save_json(
                    output / "progress.json", dict(phase="qa", finished=finished_qa, planned=config["qa_group_size"])
                )
                if chat_fingerprint(base_chat) != prefix_hash:
                    raise ValueError("子分支改动了父会话")
                return record, answer

        for start in range(0, config["qa_group_size"], batch_size):
            results = drive(qa(i) for i in range(start, min(start + batch_size, config["qa_group_size"])))
            for record, answer in results:
                records.append(record)
                artifacts.append(answer)
        save_json(output / "groups.json", [asdict(r) for r in records])
    return records, artifacts


def main():
    """仅运行明确给定配置的一组采样/一次诊断更新；输出新目录，原权重绝不覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--reuse-rollouts", type=Path, help="仅恢复已有完整组的更新，不重新采样或judge")
    parser.add_argument("--task-release", type=Path, help="从SFT导出的RL任务目录；需同时指定task-id")
    parser.add_argument("--task-id", help="固定训练任务ID，不按源轨迹成败选择")
    parser.add_argument("--save-checkpoint", action="store_true", help="实际更新后另存原生权重；不覆盖起点")
    args = parser.parse_args()
    if bool(args.task_release) != bool(args.task_id):
        parser.error("task-release与task-id必须同时指定")
    if args.save_checkpoint and not args.update:
        parser.error("save-checkpoint要求update")
    if args.task_release:
        args.task_release = args.task_release.resolve()
        if args.reuse_rollouts:
            parser.error("训练任务采集暂不支持reuse-rollouts；不能借用旧debug组")
    for key in ("bundle", "checkpoint", "case", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    from verl.experimental.physical_ai.online_reward import save_json

    try:
        import torch
        from benchmark.adapters import libero_env  # noqa: F401
        from benchmark.libero.questions.contracts import definition_hash
        from benchmark.libero.questions.registry import default_registry
        from runtime import load_policy

        from verl.experimental.physical_ai.seggrpo import Segment
        from verl.experimental.physical_ai.task_data import (
            load_task_release,
        )

        config = json.loads(args.case.read_text())
        case_hash = hashlib.sha256(args.case.read_bytes()).hexdigest()
        task, resources = load_task_release(args.task_release, args.task_id) if args.task_release else (None, None)
        if task is None and config["split"] != "debug":
            raise ValueError("此单组诊断只接受debug case，不是正式训练任务表")
        if task is not None:
            from .task_data import training_tasks

            if training_tasks([task], config["task_rule"]) != [task]:
                raise ValueError("配置的任务规则不能套到其他训练任务")
            config.update(
                split="train",
                execution_steps=config.get("execution_steps_by_suite", {}).get(
                    task["environment"]["suite"], config["execution_steps"]
                ),
                case_id=task["id"],
                suite=task["environment"]["suite"],
                task_name=task["environment"]["task_name"],
                question=task["followup"]["question_spec"],
                entity_names=task["followup"]["entity_names"],
            )
            save_json(args.output / "task.json", task)
        question = config["question"]
        question["definition_hash"] = definition_hash(question)
        default_registry().validate(question)
        if not all(type(config[k]) is int and config[k] >= 2 for k in ["execution_group_size", "qa_group_size"]):
            raise ValueError("两个组均至少两条")
        policy_version = (
            str(args.checkpoint.resolve())
            + ":"
            + str(args.checkpoint.stat().st_mtime_ns)
            + ":fp32-output-head-v1:torch-delta-fp32-v1:"
            + config["sampling_mode"]
            + ":"
            + config["compute_dtype"]
        )
        save_json(
            args.output / "settings.json",
            dict(
                config=config,
                case_sha256=case_hash,
                checkpoint=str(args.checkpoint.resolve()),
                policy_version=policy_version,
                kind="native_g05_with_verl_loss_diagnostic",
                full_verl_trainer=False,
                update=args.update,
                behavior_contract="fp32-output-head-v1:" + config["sampling_mode"],
                task_release=str(args.task_release) if args.task_release else None,
            ),
        )
        save_json(args.output / "progress.json", dict(phase="loading_model"))
        os.chdir(ROOT / "training-pipeline/g05")
        policy, processor, _ = load_policy(
            args.bundle.resolve(),
            args.checkpoint.resolve(),
            torch.device("cuda"),
            "torch",
            args.bundle.resolve() / "hf_processor",
            "sdpa",
        )
        from verl.experimental.physical_ai.precision import stable_delta_precision

        save_json(args.output / "precision.json", stable_delta_precision(policy.model))
        if args.reuse_rollouts is not None:
            if not args.update:
                raise ValueError("恢复已采组只能用于显式更新检查")
            source = args.reuse_rollouts.resolve()
            original = json.loads((source / "settings.json").read_text())
            if original["case_sha256"] != case_hash or original["policy_version"] != policy_version:
                raise ValueError("恢复组的case或行为策略不同")
            records = [Segment(**r) for r in json.loads((source / "groups.json").read_text())]
            artifacts = []
            for record in records:
                directory = source / record.rollout_id
                phase = json.loads((directory / "phase.json").read_text())
                tokens = json.loads((directory / "tokens.json").read_text())
                stored = torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True)
                if (
                    stored["input_ids"][0].tolist() != tokens["ids"]
                    or stored["attention_mask"][0].tolist() != tokens["types"]
                ):
                    raise ValueError("保存的模型输入与原始token记录不同")
                if not len(tokens["ids"]) == len(phase["generated"]) == len(phase["old_logprobs"]):
                    raise ValueError("概率、mask与token长度不符")
                reward_file = "reward.json" if record.phase == "execution" else "judge.json"
                scored = json.loads((directory / reward_file).read_text())
                reward = scored["task_reward"] if record.phase == "execution" else scored["score"]
                if reward != record.reward or record.policy_version != policy_version:
                    raise ValueError("训练奖励或行为策略与采集记录不符")
                phase["inputs_path"] = str(directory / "inputs.pt")
                artifacts.append(phase)
            from verl.experimental.physical_ai.native_update import update_once

            save_json(args.output / "progress.json", dict(phase="update", reused_rollouts=str(source)))
            result = update_once(policy.model, records, artifacts, args.output, compute_dtype=config["compute_dtype"])
            save_json(args.output / "summary.json", dict(reused_rollouts=str(source), update=result))
            save_json(args.output / "progress.json", dict(phase="complete", optimizer_step=result["optimizer_step"]))
            return
        records, artifacts = collect_group(
            policy,
            processor,
            config,
            args.output,
            policy_version=policy_version,
            task=task,
            resources=resources,
        )
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        result = dict(status="collected", optimizer_step=False)
        if args.update:
            from verl.experimental.physical_ai.native_update import update_once

            save_json(args.output / "progress.json", dict(phase="update"))
            result = update_once(policy.model, records, artifacts, args.output, compute_dtype=config["compute_dtype"])
        if args.save_checkpoint and result["optimizer_step"]:
            # 原生loader要求外层policy的state_dict；不能只保存内部model丢失codec或前缀。
            # checkpoint仅供后续加载核验，SGD无动量小试不宣称具备生产Trainer恢复状态。
            target = args.output / "checkpoints/step_1.pt"
            target.parent.mkdir()
            temporary = target.with_suffix(".building")
            save_json(args.output / "progress.json", dict(phase="saving_checkpoint"))
            torch.save(
                dict(
                    model_state_dict={k: v.detach().cpu() for k, v in policy.state_dict().items()},
                    global_step=1,
                    source_checkpoint=str(args.checkpoint),
                    kind="seggrpo_single_update_diagnostic",
                    production_resume=False,
                ),
                temporary,
            )
            temporary.rename(target)
            result["checkpoint"] = str(target)
        save_json(
            args.output / "summary.json",
            dict(
                execution_rewards=[r.reward for r in records if r.phase == "execution"],
                qa_rewards=[r.reward for r in records if r.phase == "qa"],
                selected_parent=next(r.parent_rollout_id for r in records if r.phase == "qa"),
                update=result,
            ),
        )
        save_json(args.output / "progress.json", dict(phase="complete", optimizer_step=result["optimizer_step"]))
    except BaseException:
        save_json(args.output / "fatal.json", dict(traceback=traceback.format_exc()))
        save_json(args.output / "progress.json", dict(phase="failed"))
        raise


if __name__ == "__main__":
    main()
