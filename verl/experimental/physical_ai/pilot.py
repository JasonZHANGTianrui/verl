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


def main():
    """仅运行明确给定配置的一组采样/一次诊断更新；输出新目录，原权重绝不覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--reuse-rollouts", type=Path, help="仅恢复已有完整组的更新，不重新采样或judge")
    args = parser.parse_args()
    for key in ("bundle", "checkpoint", "case", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    from verl.experimental.physical_ai.online_reward import bind_delivery, save_json, score_answer

    try:
        import torch
        from benchmark.adapters import libero_env  # noqa: F401
        from benchmark.adapters.libero_scene_map import LiberoSceneMap
        from benchmark.core.scene_graph import SceneRecorder
        from benchmark.libero.questions.contracts import definition_hash
        from benchmark.libero.questions.registry import default_registry
        from benchmark.scoring.llm_judge import JudgeClient
        from experiments.libero.libero_eval_utils import build_libero_raw_obs
        from libero.libero import benchmark
        from rollout.g05.chat_libero import LiveChat
        from runtime import load_policy

        from verl.experimental.physical_ai.g05_rollout import chat_fingerprint, collect_phase
        from verl.experimental.physical_ai.libero_session import LiberoSession
        from verl.experimental.physical_ai.sampling import fork_chat
        from verl.experimental.physical_ai.seggrpo import Segment, select_parent

        config = json.loads(args.case.read_text())
        case_hash = hashlib.sha256(args.case.read_bytes()).hexdigest()
        if config["split"] != "debug":
            raise ValueError("此单组诊断只接受debug case，不是正式训练任务表")
        suite = benchmark.get_benchmark_dict()[config["suite"]]()
        task_index = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == config["task_name"])
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
        records, artifacts, parents = [], [], []
        # 所有父环境保留到选中一个，再为每条QA建立独立副本；退出后统一释放再训练。
        with ExitStack() as stack:
            initial_digest = None
            for i in range(config["execution_group_size"]):
                rollout_id = f"execution-{i}"
                session = stack.enter_context(
                    LiberoSession(
                        config["suite"],
                        task_index,
                        initial_state_index=config["initial_state_index"],
                        seed=config["environment_seed"],
                        max_steps=config["execution_steps"] + config["qa_steps"],
                    )
                )
                reader = LiberoSceneMap(session.env)
                if config["task_rule"]["calibration"] != reader.config["version"]:
                    raise ValueError("任务状态规则与实际Scene map版本不同")
                recorder = SceneRecorder(rollout_id, max_gap_s=reader.config["max_sample_gap_s"])
                recorder.observe(reader.read(0))
                if question["parameters"]["query"]["entity"] not in reader.nodes:
                    raise ValueError("题目实体未绑定到真实场景")
                condition = hashlib.sha256(session.env.sim.get_state().flatten().tobytes()).hexdigest()
                if initial_digest is None:
                    initial_digest = condition
                    save_json(
                        args.output / "initial-rule.json",
                        dict(scene_config=reader.config, initial_state_sha256=condition, rule=config["task_rule"]),
                    )
                elif condition != initial_digest:
                    raise ValueError("动作组初态不一致")
                chat = LiveChat(policy, processor)
                chat.rl_processor = processor
                chat.rl_sampling_mode = config["sampling_mode"]
                chat.rl_compute_dtype = config["compute_dtype"]
                chat.start(session.task, build_libero_raw_obs(session.observation, session.task)[0])
                directory = args.output / rollout_id
                phase = collect_phase(
                    session,
                    chat,
                    phase="execution",
                    env_steps=config["execution_steps"],
                    token_budget=config["execution_token_budget"],
                    seed=config["seed"] + i,
                    directory=directory,
                    reader=reader,
                    recorder=recorder,
                )
                # 此小试任务规则是版本化Scene map状态“middle drawer=open”，不读原生success。
                binding = bind_delivery(question, recorder, config["entity_names"], rollout_id)
                private_history = recorder.export()
                save_json(directory / "delivery.json", private_history["questions"][question["question_id"]])
                save_json(directory / "scene-map.json", private_history)
                if binding["gt"]["status"] != "ok":
                    raise ValueError("任务/QA语义状态未知，不能填零奖励")
                state = binding["gt"]["value"]
                if state not in ["open", "closed", "partially_open"]:
                    raise ValueError("设备状态不符合已冻结规则")
                reward = float(state == config["task_rule"]["target_state"])
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
                records.append(record)
                phase["inputs_path"] = str(directory / "inputs.pt")
                artifacts.append(phase)
                parents.append((session, chat, phase, binding))
                save_json(
                    directory / "reward.json",
                    dict(task_reward=reward, binding=binding, native_success=session.audit()["native_success"]),
                )
                save_json(
                    args.output / "progress.json",
                    dict(phase="execution", finished=i + 1, planned=config["execution_group_size"]),
                )
            selected = select_parent(records, seed=config["seed"], mode="uniform")
            parent_index = next(i for i, r in enumerate(records) if r.rollout_id == selected)
            session, chat, phase, binding = parents[parent_index]
            base_chat = fork_chat(chat)
            base_chat.question(question["question"], pending_assistant=phase["pending_assistant"])
            prefix_hash = chat_fingerprint(base_chat)
            qa_condition = hashlib.sha256((prefix_hash + binding["binding_sha256"]).encode()).hexdigest()
            save_json(
                args.output / "selection.json",
                dict(
                    parent_rollout_id=selected,
                    parent_reward=records[parent_index].reward,
                    selection="uniform",
                    seed=config["seed"],
                    prefix_sha256=prefix_hash,
                    condition_id=qa_condition,
                ),
            )
            client = JudgeClient(
                Path(os.environ["PHYSICAL_AI_JUDGE_ACCESS_FILE"]), model=config["judge_model"], timeout_s=90
            )
            for i in range(config["qa_group_size"]):
                child_id = f"qa-{i}"
                with session.fork() as child:
                    child_chat = fork_chat(base_chat)
                    if chat_fingerprint(child_chat) != prefix_hash:
                        raise ValueError("QA分支会话前缀不同")
                    directory = args.output / child_id
                    answer = collect_phase(
                        child,
                        child_chat,
                        phase="qa",
                        env_steps=config["qa_steps"],
                        token_budget=config["qa_token_budget"],
                        seed=config["seed"] + 100 + i,
                        directory=directory,
                        generated=phase["generated"],
                        old_logprobs=phase["old_logprobs"],
                    )
                    judgment = score_answer(binding, answer["answer"], client, rollout_id=selected)
                    save_json(directory / "judge.json", judgment)
                    if judgment["status"] != "scored":
                        raise ValueError("judge未给有效分数，整个组保留待恢复，禁止更新")
                    records.append(
                        Segment(
                            child_id,
                            "qa",
                            "qa-group",
                            qa_condition,
                            config["case_id"],
                            policy_version,
                            config["qa_group_size"],
                            answer["prefix_length"] - 1,
                            answer["end_length"] - 1,
                            float(judgment["score"]),
                            "scored",
                            parent_rollout_id=selected,
                        )
                    )
                    answer["inputs_path"] = str(directory / "inputs.pt")
                    artifacts.append(answer)
                    save_json(
                        args.output / "progress.json", dict(phase="qa", finished=i + 1, planned=config["qa_group_size"])
                    )
                    if chat_fingerprint(base_chat) != prefix_hash:
                        raise ValueError("子分支改动了父会话")
            save_json(args.output / "groups.json", [asdict(r) for r in records])
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        result = dict(status="collected", optimizer_step=False)
        if args.update:
            from verl.experimental.physical_ai.native_update import update_once

            save_json(args.output / "progress.json", dict(phase="update"))
            result = update_once(policy.model, records, artifacts, args.output, compute_dtype=config["compute_dtype"])
        save_json(
            args.output / "summary.json",
            dict(
                execution_rewards=[r.reward for r in records if r.phase == "execution"],
                qa_rewards=[r.reward for r in records if r.phase == "qa"],
                selected_parent=selected,
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
