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

    采样可按完整任务分给各rank并行执行，收齐整个batch后全局核验/分片更新；没有冒称vLLM、
    Ray/FSDP已接入。模型数据路径与pilot完全共用，不维护第二套G0.5推理器。
    错误阻止全组更新；恢复只从完整checkpoint开始，旧的半组保留在旧attempt目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "checkpoint", "task-release", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="上一完整组checkpoint.json；必须使用新output目录")
    parser.add_argument("--resume-migration", type=Path, help="显式来源/目标契约SHA批准；仅允许拓扑及已审核代码迁移")
    parser.add_argument("--swanlab-key-file", type=Path)
    parser.add_argument("--stop-after-groups", type=int, help="本次仅运行若干完整组，用于真实保存/恢复验收")
    args = parser.parse_args()
    if args.resume_migration and not args.resume:
        parser.error("resume-migration需要resume")
    if args.stop_after_groups is not None and args.stop_after_groups < 1:
        parser.error("stop-after-groups必须为正整数")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    import numpy as np
    import torch
    import torch.distributed as dist
    from runtime import load_policy

    from .batching import minibatch_indices, namespace_group, prepare_batch
    from .native_update import update_once, validate_behavior
    from .online_reward import save_json
    from .pilot import collect_group
    from .precision import stable_delta_precision
    from .task_data import load_task_release
    from .training_metrics import rollout_metrics, sampling_batch_metrics, update_metrics
    from .training_state import (
        build_optimizer,
        checkpoint_due,
        content_hash,
        load_boundary,
        make_tracker,
        prune_checkpoints,
        restore_optimizer,
        save_boundary,
        training_budget,
        validate_fixed_algorithm,
    )

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    runtime_config = json.loads(args.config.read_text())
    # EGL设备序号与CUDA local_rank是两个命名空间；容器可能有两张计算卡但仅一个EGL设备。
    # 默认沿渲染器的设备0，允许部署显式覆盖，不修改模型所在CUDA卡。
    egl_device = runtime_config.get("egl_device_id", 0)
    if type(egl_device) is not int or egl_device < 0:
        raise ValueError("egl_device_id须是EGL枚举中的非负整数，不能使用CUDA rank推断")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
    control_group = None
    if world > 1:
        timeout = timedelta(minutes=runtime_config.get("collective_timeout_minutes", 180))
        dist.init_process_group("nccl", timeout=timeout)
        # 环境与语言采样耗时可差很多；CPU状态同步不应预先启动等待另一rank的CUDA操作。
        control_group = dist.new_group(backend="gloo", timeout=timeout)

    def broadcast(value):
        """rank0的采集/保存状态显式广播；异常不能让其他rank误进入下一次更新。"""
        payload = [value]
        if world > 1:
            dist.broadcast_object_list(payload, src=0, group=control_group)
        return payload[0]

    tracker = None
    failed = False
    try:
        config = runtime_config
        if config["sampling_mode"] not in {"recompute", "cached"} or config["compute_dtype"] != "float32":
            raise ValueError("此后端仅启用FP32原生重算或经完整轨迹概率核验的缓存采样")
        algorithm = validate_fixed_algorithm(config)
        if config["optimizer"] not in {"sgd", "adamw"}:
            raise ValueError("需要合法优化器及正数任务组预算")
        if min(config["execution_group_size"], config["qa_group_size"]) < 2:
            raise ValueError("GRPO两组均须至少两条")
        width = config["execution_group_size"] + config["qa_group_size"]
        batch_size = config.get("batch_size", width)
        mini_size = config.get("mini_batch_size", batch_size)
        tasks_per_batch = batch_size // width
        if (
            batch_size % width
            or mini_size % width
            or batch_size % mini_size
            or tasks_per_batch < 1
            or config.get("micro_batch_size_per_gpu", 1) != 1
            or config.get("ppo_epochs", 1) != 1
        ):
            raise ValueError("batch/mini-batch须由完整任务组组成；当前micro-batch及PPO epochs固定1")
        max_groups = training_budget(config, tasks_per_batch)
        save_freq = config.get("save_freq", 1)
        keep = config.get("max_ckpt_to_keep")
        checkpoint_due(1, save_freq)
        if keep is not None and (type(keep) is not int or keep < 1):
            raise ValueError("max_ckpt_to_keep须为正整数或null")
        if args.stop_after_groups and args.stop_after_groups % tasks_per_batch:
            raise ValueError("停止点必须在完整采样batch边界")
        random.seed(config["seed"] + rank)
        np.random.seed(config["seed"] + rank)
        torch.manual_seed(config["seed"] + rank)
        source = args.task_release / "train.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        from .task_data import training_tasks

        eligible = training_tasks(rows, config["task_rule"])
        if len(eligible) < tasks_per_batch:
            raise ValueError("已绑定动作奖励的训练记录不足一个batch")
        # 固定训练任务顺序只由身份与seed决定，与奖励、QA答案及SFT成败无关。
        eligible.sort(key=lambda t: hashlib.sha256(f"{config['seed']}:{t['id']}".encode()).hexdigest())
        _, resources = load_task_release(args.task_release, eligible[0]["id"])
        from .native_runtime import identity as model_interface_identity

        if (
            config.get("model_interface") == "reactpi-v1-d0bdbf2"
            and model_interface_identity().get("commit") != "d0bdbf298a6a7f50b67e213111ff5c65f20a70ca"
        ):
            raise ValueError("完整任务池需要固定ReActPI v1交接，不能回退共享早期接口")
        contract = dict(
            config=config,
            model_interface=model_interface_identity(),
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
            implementation_files={
                **{p.name: content_hash(p) for p in Path(__file__).parent.glob("*.py")},
            },
        )
        from ours.rl.prepare_online import validate_rebind

        task_rebind = validate_rebind(args.task_release, rows)
        if task_rebind:
            contract["task_rebind"] = task_rebind
        checkpoint, completed, updates, restored = args.checkpoint, 0, 0, None
        if args.resume:
            restored, receipt = load_boundary(
                args.resume,
                contract,
                migration=json.loads(args.resume_migration.read_text()) if args.resume_migration else None,
            )
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
                        migration=json.loads(args.resume_migration.read_text()) if args.resume_migration else None,
                    ),
                )
                tracker_config = dict(config, **{k: v for k, v in algorithm.items() if k not in config})
                tracker_config["runtime"] = dict(
                    world_size=world,
                    tasks_per_batch=tasks_per_batch,
                    sequences_per_task=width,
                    mini_batches_per_batch=batch_size // mini_size,
                    max_micro_batches_per_rank=mini_size // world,
                    total_sampling_batches=max_groups // tasks_per_batch,
                    max_optimizer_steps=max_groups // tasks_per_batch * (batch_size // mini_size),
                    eligible_tasks=len(eligible),
                    source_tasks=len(rows),
                    source_checkpoint_sha256=contract["source_checkpoint_sha256"],
                    tasks_sha256=contract["tasks_sha256"],
                    save_freq_unit="sampling_batch",
                    egl_device_id=egl_device,
                    control_backend="gloo",
                    gradient_backend="nccl",
                    reference_model=False,
                    critic=False,
                    entropy_computed=False,
                    actor_backend="native_torch_zero1" if config.get("optimizer_state_sharding") else "native_torch",
                    sampling_backend=(
                        "native_g05_batched_cached"
                        if config["sampling_mode"] == "cached"
                        else "native_g05_full_prefix_recompute"
                    ),
                    rollout_batch_size=config.get("rollout_batch_size", 1),
                )
                save_json(args.output / "effective-config.json", tracker_config)
                tracker = make_tracker(tracker_config, args.output, args.swanlab_key_file)
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
        optimizer = build_optimizer(policy.model, config, distributed=world > 1)
        if restored:
            restore_optimizer(optimizer, restored["optimizer_state_dict"])
            if restored["contract"]["world_size"] == world:
                restore_rng(restored["rng_states"][rank])
            else:
                # 改卡数后不存在一一对应的rank随机流，按恢复游标和新rank确定性重建。
                # 模型/AdamW/训练游标完整恢复，但不声称与旧拓扑逐位相同。
                resume_seed = config["seed"] + completed * 1000 + rank
                random.seed(resume_seed)
                np.random.seed(resume_seed % 2**32)
                torch.manual_seed(resume_seed)
            del restored
        last_group = min(max_groups, completed + args.stop_after_groups) if args.stop_after_groups else max_groups
        if completed % tasks_per_batch:
            raise ValueError("恢复点不是完整采样batch边界")
        for group_index in range(completed, last_group, tasks_per_batch):
            started = time.monotonic()
            batch_index = group_index // tasks_per_batch
            directory = args.output / f"batch-{batch_index:06d}"
            if rank == 0:
                directory.mkdir()
                save_json(
                    args.output / "progress.json",
                    dict(
                        phase="sampling",
                        batch=batch_index,
                        completed_groups=group_index,
                        tasks_per_batch=tasks_per_batch,
                        planned_groups=max_groups,
                        optimizer_steps=updates,
                    ),
                )
            broadcast(True if rank == 0 else None)
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            policy.model.eval()
            local_tasks = []
            # 任一rank采样失败直接抛给外层落盘和torchrun；不得把异常藏进一个长时间等待的collective。
            for offset in range(tasks_per_batch):
                owner = offset % world if config.get("parallel_rollouts", False) else 0
                if owner != rank:
                    continue
                task_index = group_index + offset
                task = eligible[task_index % len(eligible)]
                run_config = deepcopy(config)
                run_config.update(
                    execution_steps=config.get("execution_steps_by_suite", {}).get(
                        task["environment"]["suite"], config["execution_steps"]
                    ),
                    case_id=task["id"],
                    suite=task["environment"]["suite"],
                    task_name=task["environment"]["task_name"],
                    question=task["followup"]["question_spec"],
                    entity_names=task["followup"]["entity_names"],
                    seed=config["seed"] + task_index * 1000,
                )
                task_dir = directory / f"task-{task_index:06d}"
                task_dir.mkdir()
                save_json(task_dir / "task.json", task)
                save_json(
                    args.output / f"progress-rank-{rank}.json",
                    dict(phase="sampling", batch=batch_index, task_index=task_index, finished_tasks=len(local_tasks)),
                )
                task_records, task_artifacts = collect_group(
                    policy,
                    processor,
                    run_config,
                    task_dir,
                    policy_version=f"{contract['source_checkpoint_sha256']}/update-{updates}",
                    task=task,
                    resources=resources,
                )
                local_tasks.append(
                    dict(
                        index=task_index,
                        records=namespace_group(task_records, f"task-{task_index:06d}"),
                        artifacts=task_artifacts,
                    )
                )
            reports = [local_tasks]
            if world > 1:
                reports = [None] * world
                dist.all_gather_object(reports, local_tasks, group=control_group)
            collected = sorted([t for rank_tasks in reports for t in rank_tasks], key=lambda t: t["index"])
            if len(collected) != tasks_per_batch:
                raise ValueError("采样任务数不完整")
            records = [r for t in collected for r in t["records"]]
            artifacts = [a for t in collected for a in t["artifacts"]]
            if len(records) != batch_size:
                raise ValueError("实际序列数不等于配置batch_size")
            sampling_seconds = time.monotonic() - started
            prepared = prepare_batch(records, artifacts)
            batch_metrics = rollout_metrics(records, artifacts, prepared)
            validation_started = time.monotonic()
            minis = minibatch_indices(records, mini_size)
            if rank == 0:
                from dataclasses import asdict

                save_json(directory / "groups.json", [asdict(r) for r in records])
                save_json(
                    args.output / "progress.json",
                    dict(
                        phase="validate_behavior",
                        batch=batch_index,
                        completed_groups=group_index,
                        optimizer_steps=updates,
                    ),
                )
            parity = validate_behavior(
                policy.model,
                records,
                artifacts,
                prepared,
                directory,
                distributed=world > 1,
                max_error=config.get("behavior_logprob_tolerance", 0.05),
                control_group=control_group,
            )
            validation_seconds = time.monotonic() - validation_started
            mini_metrics = []
            for mini_index, indices in enumerate(minis):
                update_started = time.monotonic()
                rank_dir = directory / f"mini-{mini_index:02d}" / f"rank-{rank}"
                rank_dir.mkdir(parents=True)
                if rank == 0:
                    save_json(
                        args.output / "progress.json",
                        dict(
                            phase="update",
                            batch=batch_index,
                            mini_batch=mini_index,
                            mini_batches=len(minis),
                            optimizer_steps=updates,
                        ),
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
                    prepared=prepared,
                    indices=indices,
                    enforce_on_policy=False,
                    clip_ratio=config.get("clip_ratio", 0.2),
                    clip_ratio_c=config.get("clip_ratio_c", 3.0),
                    control_group=control_group,
                )
                updates += int(result["optimizer_step"])
                if rank == 0:
                    metrics = dict(batch_metrics)
                    metrics.update(update_metrics(result, records, indices))
                    metrics.update(
                        {
                            "train/sampling_batch": batch_index + 1,
                            "train/mini_batch": mini_index + 1,
                            "train/optimizer_steps": updates,
                            "train/task_records_seen": group_index + tasks_per_batch,
                            "train/data_passes": (group_index + tasks_per_batch) / len(eligible),
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/behavior_check_max_error": parity["max_error"],
                            "timing/sampling_seconds": sampling_seconds,
                            "timing/behavior_check_seconds": validation_seconds,
                            "timing/mini_update_seconds": time.monotonic() - update_started,
                            "timing/batch_seconds_so_far": time.monotonic() - started,
                        }
                    )
                memory = [dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())]
                if world > 1:
                    local_memory = memory[0]
                    memory = [None] * world
                    dist.all_gather_object(memory, local_memory, group=control_group)
                if rank == 0:
                    metrics["memory/max_allocated_gib"] = max(m["allocated"] for m in memory) / 2**30
                    metrics["memory/max_reserved_gib"] = max(m["reserved"] for m in memory) / 2**30
                    mini_metrics.append(metrics)
                    save_json(directory / f"mini-{mini_index:02d}/summary.json", dict(metrics=metrics, update=result))
            completed_now = group_index + tasks_per_batch
            should_save = checkpoint_due(batch_index + 1, save_freq, final=completed_now == last_group)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            save_started = time.monotonic()
            if should_save and hasattr(optimizer, "consolidate_state_dict"):
                optimizer.consolidate_state_dict(to=0)
            states = [rng_state()] if should_save else []
            if should_save and world > 1:
                states = [None] * world
                dist.all_gather_object(states, rng_state(), group=control_group)
            saved = None
            if rank == 0:
                try:
                    receipt, removed = None, []
                    if should_save:
                        receipt = save_boundary(
                            policy,
                            optimizer,
                            args.output / f"checkpoints/group-{completed_now:06d}",
                            completed_groups=completed_now,
                            optimizer_steps=updates,
                            contract=contract,
                            rng_states=states,
                        )
                        save_json(args.output / "latest.json", receipt)
                        removed = prune_checkpoints(args.output / "checkpoints", receipt, keep)
                    save_json(
                        directory / "summary.json",
                        dict(
                            behavior_check=parity,
                            tasks=tasks_per_batch,
                            sequences=batch_size,
                            mini_batches=len(minis),
                            optimizer_steps=updates,
                            completed_sampling_batches=batch_index + 1,
                            seconds=time.monotonic() - started,
                            checkpoint=receipt,
                            removed_checkpoints=removed,
                        ),
                    )
                    logged_metrics = sampling_batch_metrics(mini_metrics)
                    logged_metrics.update(
                        {
                            "timing/batch_seconds": time.monotonic() - started,
                            "timing/checkpoint_seconds": time.monotonic() - save_started if should_save else 0,
                            "checkpoint/saved": int(should_save),
                        }
                    )
                    tracker.log(logged_metrics, step=batch_index + 1)
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
                    phase="complete" if last_group == max_groups else "stopped_at_group_boundary",
                    completed_groups=last_group,
                    optimizer_steps=updates,
                ),
            )
    except BaseException:
        failed = True
        detail = dict(rank=rank, traceback=traceback.format_exc())
        traceback.print_exc()
        if args.output.is_dir():
            save_json(args.output / f"fatal-rank-{rank}.json", detail)
            save_json(args.output / f"progress-rank-{rank}.json", dict(phase="failed", failure=detail))
            # 公共文件只由rank0写；首因仍以各rank独立traceback为准，避免tmp路径争写。
            if rank == 0:
                save_json(args.output / "fatal.json", detail)
                save_json(args.output / "progress.json", dict(phase="failed", failure_rank=rank))
        raise
    finally:
        if tracker is not None:
            tracker.finish()
        # 错误路径由torchrun终止其他rank，不等待仍在仿真的进程一起销毁通信组。
        if world > 1 and not failed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
