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

"""verl fork内的原生G0.5连续训练入口；复用已验收的采样与分布式策略更新。"""

import argparse
import gc
import hashlib
import json
import os
import random
import time
import traceback
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

from .workspace import configure_workspace

ROOT = configure_workspace()


def rng_state():
    """记录本rank的Python/NumPy/Torch随机状态；NumPy转基本类型，允许安全torch加载。"""
    import numpy as np
    import torch

    n = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=[n[0], n[1].tolist(), *n[2:]],
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(),
    )


def restore_rng(state):
    """模型/优化器恢复后再恢复随机流，避免初始化消耗改变下一组采样。"""
    import numpy as np
    import torch

    n = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def main():
    """每组重新采样→真实在线奖励→全局优势/分片梯度→组边界保存，支持torchrun。

    采样目前在rank0串行进行，其他rank参加梯度计算与全局归约；没有冒称vLLM、
    Ray/FSDP已接入。模型数据路径与pilot完全共用，不维护第二套G0.5推理器。
    错误阻止全组更新；恢复只从完整checkpoint开始，旧的半组保留在旧attempt目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "checkpoint", "task-release", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="上一完整组checkpoint.json；必须使用新output目录")
    parser.add_argument("--swanlab-key-file", type=Path)
    parser.add_argument("--stop-after-groups", type=int, help="本次仅运行若干完整组，用于真实保存/恢复验收")
    args = parser.parse_args()
    if args.stop_after_groups is not None and args.stop_after_groups < 1:
        parser.error("stop-after-groups必须为正整数")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    import numpy as np
    import torch
    import torch.distributed as dist
    from runtime import load_policy

    from .native_update import update_once
    from .online_reward import save_json
    from .pilot import collect_group
    from .precision import stable_delta_precision
    from .task_data import load_task_release
    from .training_state import content_hash, load_boundary, make_tracker, save_boundary

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=3))

    def broadcast(value):
        """rank0的采集/保存状态显式广播；异常不能让其他rank误进入下一次更新。"""
        payload = [value]
        if world > 1:
            dist.broadcast_object_list(payload, src=0)
        return payload[0]

    tracker = None
    try:
        config = json.loads(args.config.read_text())
        if config["sampling_mode"] != "recompute" or config["compute_dtype"] != "float32":
            raise ValueError("此后端仅启用已通过完整轨迹概率验收的FP32重算")
        if config["optimizer"] != "sgd" or config["max_groups"] < 1:
            raise ValueError("当前首阶段使用显式SGD；其他优化器需另行验证显存和恢复")
        if min(config["execution_group_size"], config["qa_group_size"]) < 2:
            raise ValueError("GRPO两组均须至少两条")
        random.seed(config["seed"] + rank)
        np.random.seed(config["seed"] + rank)
        torch.manual_seed(config["seed"] + rank)
        source = args.task_release / "train.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        eligible = [r for r in rows if r["environment"]["task_name"] == config["task_rule"]["task_name"]]
        if not eligible:
            raise ValueError("没有绑定已验证动作奖励的任务")
        # 固定训练任务顺序只由身份与seed决定，与奖励、QA答案及SFT成败无关。
        eligible.sort(key=lambda t: hashlib.sha256(f"{config['seed']}:{t['id']}".encode()).hexdigest())
        _, resources = load_task_release(args.task_release, eligible[0]["id"])
        contract = dict(
            config=config,
            tasks_sha256=content_hash(source),
            world_size=world,
            eligible_ids=[r["id"] for r in eligible],
            source_checkpoint_sha256=broadcast(content_hash(args.checkpoint) if rank == 0 else None),
            bundle_files=broadcast(
                {
                    str(p.relative_to(args.bundle)): content_hash(p)
                    for p in [
                        args.bundle / ".hydra/config.yaml",
                        args.bundle / "action_tokenizer.pt",
                        args.bundle / "dataset_stats.json",
                        *sorted((args.bundle / "hf_processor").glob("*")),
                    ]
                    if p.is_file()
                }
                if rank == 0
                else None
            ),
            implementation_files={p.name: content_hash(p) for p in Path(__file__).parent.glob("*.py")},
        )
        checkpoint, completed, updates, restored = args.checkpoint, 0, 0, None
        if args.resume:
            restored, receipt = load_boundary(args.resume, contract)
            checkpoint = Path(receipt["path"])
            completed, updates = restored["completed_groups"], restored["optimizer_steps"]
        startup = None
        if rank == 0:
            try:
                args.output.mkdir(parents=True, exist_ok=False)
                save_json(
                    args.output / "settings.json",
                    dict(
                        contract=contract,
                        source_tasks=len(rows),
                        eligible_tasks=len(eligible),
                        unbound_tasks=len(rows) - len(eligible),
                        backend="native_g05_torch_distributed",
                        ray_fsdp=False,
                        resume=str(args.resume),
                    ),
                )
                tracker = make_tracker(config, args.output, args.swanlab_key_file)
                if "swanlab" in tracker.logger:
                    import swanlab

                    save_json(args.output / "swanlab.json", dict(url=swanlab.get_run().url))
                save_json(args.output / "progress.json", dict(phase="loading_model", completed_groups=completed))
                startup = dict(ok=True)
            except Exception:
                startup = dict(ok=False, error=traceback.format_exc())
        startup = broadcast(startup)
        if not startup["ok"]:
            raise RuntimeError(startup["error"])
        os.chdir(ROOT / "training-pipeline/g05")
        policy, processor, _ = load_policy(
            args.bundle, checkpoint, torch.device("cuda", local_rank), "torch", args.bundle / "hf_processor", "sdpa"
        )
        stable_delta_precision(policy.model)
        optimizer = torch.optim.SGD(
            [p for p in policy.model.parameters() if p.requires_grad], lr=config["learning_rate"]
        )
        if restored:
            optimizer.load_state_dict(restored["optimizer_state_dict"])
            restore_rng(restored["rng_states"][rank])
            del restored
        last_group = (
            min(config["max_groups"], completed + args.stop_after_groups)
            if args.stop_after_groups
            else config["max_groups"]
        )
        for group_index in range(completed, last_group):
            started = time.monotonic()
            task = eligible[group_index % len(eligible)]
            run_config = deepcopy(config)
            run_config.update(
                case_id=task["id"],
                suite=task["environment"]["suite"],
                task_name=task["environment"]["task_name"],
                question=task["followup"]["question_spec"],
                entity_names=task["followup"]["entity_names"],
                seed=config["seed"] + group_index * 1000,
            )
            directory = args.output / f"group-{group_index:06d}"
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            collected = None
            if rank == 0:
                try:
                    directory.mkdir()
                    save_json(
                        args.output / "progress.json",
                        dict(
                            phase="sampling",
                            group=group_index,
                            planned_groups=config["max_groups"],
                            task_id=task["id"],
                            optimizer_steps=updates,
                        ),
                    )
                    save_json(directory / "task.json", task)
                    records, artifacts = collect_group(
                        policy,
                        processor,
                        run_config,
                        directory,
                        policy_version=f"{contract['source_checkpoint_sha256']}/update-{updates}",
                        task=task,
                        resources=resources,
                    )
                    collected = dict(ok=True, records=records, artifacts=artifacts)
                except Exception:
                    collected = dict(ok=False, error=traceback.format_exc())
            collected = broadcast(collected)
            if not collected["ok"]:
                raise RuntimeError(collected["error"])
            records, artifacts = collected["records"], collected["artifacts"]
            rank_dir = directory / f"rank-{rank}"
            rank_dir.mkdir()
            if rank == 0:
                save_json(
                    args.output / "progress.json", dict(phase="update", group=group_index, optimizer_steps=updates)
                )
            result = update_once(
                policy.model,
                records,
                artifacts,
                rank_dir,
                compute_dtype="float32",
                distributed=world > 1,
                optimizer=optimizer,
                learning_rate=config["learning_rate"],
                max_grad_norm=config["max_grad_norm"],
            )
            updates += int(result["optimizer_step"])
            states = [rng_state()]
            if world > 1:
                states = [None] * world
                dist.all_gather_object(states, rng_state())
            saved = None
            if rank == 0:
                try:
                    metrics = {
                        f"reward/{phase}": sum(r.reward for r in records if r.phase == phase)
                        / sum(r.phase == phase for r in records)
                        for phase in ("execution", "qa")
                    }
                    metrics.update(
                        {
                            "train/optimizer_steps": updates,
                            "train/zero_variance_groups": result["zero_variance_groups"],
                            "train/group_seconds": time.monotonic() - started,
                            "train/max_logprob_error": max(
                                (r["rollout_actor_max_logprob_error"] for r in result.get("rows", [])), default=0
                            ),
                            "train/grad_norm": result.get("grad_norm_before_clip") or 0,
                        }
                    )
                    tracker.log(metrics, step=group_index + 1)
                    save_json(directory / "summary.json", dict(metrics=metrics, update=result))
                    receipt = save_boundary(
                        policy,
                        optimizer,
                        args.output / f"checkpoints/group-{group_index + 1:06d}",
                        completed_groups=group_index + 1,
                        optimizer_steps=updates,
                        contract=contract,
                        rng_states=states,
                    )
                    save_json(args.output / "latest.json", receipt)
                    saved = dict(ok=True)
                except Exception:
                    saved = dict(ok=False, error=traceback.format_exc())
            saved = broadcast(saved)
            if not saved["ok"]:
                raise RuntimeError(saved["error"])
        if rank == 0:
            save_json(
                args.output / "progress.json",
                dict(
                    phase="complete" if last_group == config["max_groups"] else "stopped_at_group_boundary",
                    completed_groups=last_group,
                    optimizer_steps=updates,
                ),
            )
    except BaseException:
        if rank == 0 and args.output.is_dir():
            save_json(args.output / "fatal.json", dict(traceback=traceback.format_exc()))
            save_json(args.output / "progress.json", dict(phase="failed"))
        raise
    finally:
        if tracker is not None:
            tracker.finish()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
