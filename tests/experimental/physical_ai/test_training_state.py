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

"""组边界恢复的核心回归：优化器连续性、数据契约与损坏文件拒绝。"""

import tempfile
import unittest
from pathlib import Path

import torch

from verl.experimental.physical_ai.training_state import (
    build_optimizer,
    checkpoint_due,
    checkpoint_plan,
    contract_hash,
    load_boundary,
    prune_checkpoints,
    restore_optimizer,
    save_boundary,
    training_budget,
    validate_fixed_algorithm,
    validate_resume_contract,
)


def _zero_adam_worker(rank, directory):
    """两进程ZeRO-1 AdamW与单进程完整batch逐参数比较，并在两次更新间真实保存/恢复。"""
    import torch.distributed as dist

    from verl.experimental.physical_ai.distributed import sum_gradients

    directory = Path(directory)
    dist.init_process_group("gloo", init_method="file://" + str(directory / "rendezvous"), rank=rank, world_size=2)
    try:
        torch.manual_seed(9)
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1))
        reference = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1))
        reference.load_state_dict(model.state_dict())
        config = dict(optimizer="adamw", learning_rate=0.01, optimizer_state_sharding=True, weight_decay=0.01)
        opt = build_optimizer(model, config, distributed=True)
        ref_opt = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.01, foreach=False)
        x, y = torch.randn(8, 3), torch.randn(8, 1)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            ref_opt.zero_grad(set_to_none=True)
            (model(x[rank::2]) - y[rank::2]).square().sum().div(8).backward()
            sum_gradients(model)
            (reference(x) - y).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            opt.step()
            ref_opt.step()
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=False):
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
            if step == 0:
                opt.consolidate_state_dict(to=0)
                if rank == 0:
                    save_boundary(
                        model,
                        opt,
                        directory / "saved",
                        completed_groups=8,
                        optimizer_steps=1,
                        contract=config,
                        rng_states=[],
                    )
                dist.barrier()
                state, _ = load_boundary(directory / "saved/checkpoint.json", config)
                model.load_state_dict(state["model_state_dict"])
                opt = build_optimizer(model, config, distributed=True)
                restore_optimizer(opt, state["optimizer_state_dict"])
                for param, local in opt.optim.state.items():
                    for key, value in local.items():
                        if isinstance(value, torch.Tensor) and value.ndim:
                            assert value.data_ptr() == opt.state[param][key].data_ptr()
    finally:
        dist.destroy_process_group()


def _reshard_adam_worker(rank, directory):
    """读取两rank合并存档，在八rank重分片后与完整AdamW逐参数比较。"""
    import torch.distributed as dist

    from verl.experimental.physical_ai.distributed import sum_gradients

    directory = Path(directory)
    dist.init_process_group("gloo", init_method="file://" + str(directory / "reshard"), rank=rank, world_size=8)
    try:
        config = dict(optimizer="adamw", learning_rate=0.01, optimizer_state_sharding=True, weight_decay=0.01)
        state, _ = load_boundary(directory / "saved/checkpoint.json", config)
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1))
        reference = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1))
        model.load_state_dict(state["model_state_dict"])
        reference.load_state_dict(state["model_state_dict"])
        opt = build_optimizer(model, config, distributed=True)
        ref_opt = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.01, foreach=False)
        from copy import deepcopy

        ref_opt.load_state_dict(deepcopy(state["optimizer_state_dict"]))
        restore_optimizer(opt, state["optimizer_state_dict"])
        torch.manual_seed(77)
        x, y = torch.randn(32, 3), torch.randn(32, 1)
        (model(x[rank::8]) - y[rank::8]).square().sum().div(32).backward()
        sum_gradients(model)
        (reference(x) - y).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        opt.step()
        ref_opt.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    finally:
        dist.destroy_process_group()


class TrainingStateTests(unittest.TestCase):
    """用有动量的小模型核对真实优化器恢复，不用无状态SGD掩盖漏存状态。"""

    def test_judge_context_migration_cannot_change_training(self):
        """显式裁判修复迁移只改指定源码；数据/奖励/学习率/调度和其他实现变化都拒绝。"""
        from copy import deepcopy

        source = dict(
            config={"learning_rate": 1e-6},
            world_size=8,
            tasks_sha256="tasks",
            task_schedule={"ids": ["a"]},
            implementation_files={
                n: "old" for n in ["online_reward.py", "pilot.py", "grpo_rollout.py", "training_state.py", "seggrpo.py"]
            },
        )
        target = deepcopy(source)
        for n in ["online_reward.py", "pilot.py", "grpo_rollout.py", "training_state.py"]:
            target["implementation_files"][n] = "new"

        def receipt(t):
            """绑定完整两端hash及明确文件差异，不接受仅有协议名的迁移。"""
            return dict(
                source_contract_sha256=contract_hash(source),
                target_contract_sha256=contract_hash(t),
                protocol_revision="judge-task-context-v1",
                implementation_changes={
                    n: dict(before=source["implementation_files"].get(n), after=t["implementation_files"].get(n))
                    for n in source["implementation_files"].keys() | t["implementation_files"].keys()
                    if source["implementation_files"].get(n) != t["implementation_files"].get(n)
                },
            )

        validate_resume_contract(source, target, receipt(target))
        for key in ["config", "world_size", "tasks_sha256", "task_schedule", "implementation_files"]:
            bad = deepcopy(target)
            if key == "implementation_files":
                bad[key]["seggrpo.py"] = "changed"
            else:
                bad[key] = "changed"
            with self.assertRaises(ValueError):
                validate_resume_contract(source, bad, receipt(bad))
        with self.assertRaises(ValueError):
            validate_resume_contract(source, target, None)

    def test_recovery_and_model_checkpoint_frequencies(self):
        """恢复每5批而测评每20批；未配置仍保持旧频率，末批两种都保存。"""
        self.assertEqual(checkpoint_plan(5, 20, 5), (True, False))
        self.assertEqual(checkpoint_plan(20, 20, 5), (True, True))
        self.assertEqual(checkpoint_plan(6, 20, 5), (False, False))
        self.assertEqual(checkpoint_plan(5, 20), (False, False))
        self.assertEqual(checkpoint_plan(7, 20, 5, final=True), (True, True))
        for value in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                checkpoint_plan(20, 20, value)

    def test_generation_timeout_recovery_preserves_training_contract(self):
        """基础设施超时只可有限延长；不能夹带预算、奖励、拓扑、数据或模型修改。"""
        from copy import deepcopy

        source = dict(
            world_size=8,
            tasks_sha256="tasks",
            source_checkpoint_sha256="weights",
            implementation_files={"training_state.py": "old"},
            config=dict(generation_timeout_seconds=120, execution_token_budget=6144, learning_rate=1e-6),
        )
        target = deepcopy(source)
        target["config"]["generation_timeout_seconds"] = 1800
        target["implementation_files"]["training_state.py"] = "new"

        def approval(dst):
            """绑定待恢复的准确配置，避免旧批准适用于其他改动。"""
            return dict(
                source_contract_sha256=contract_hash(source),
                target_contract_sha256=contract_hash(dst),
                protocol_revision="generation-timeout-increase-v1",
            )

        validate_resume_contract(source, target, approval(target))
        with self.assertRaises(ValueError):
            validate_resume_contract(source, target)
        for value in (None, True, 0, 120, float("inf"), float("nan")):
            bad = deepcopy(target)
            bad["config"]["generation_timeout_seconds"] = value
            with self.assertRaises(ValueError):
                validate_resume_contract(source, bad, approval(bad))
        for key, value in (("execution_token_budget", 8192), ("learning_rate", 1e-4)):
            bad = deepcopy(target)
            bad["config"][key] = value
            with self.assertRaises(ValueError):
                validate_resume_contract(source, bad, approval(bad))
        for key, value in (("world_size", 4), ("tasks_sha256", "other"), ("source_checkpoint_sha256", "other")):
            bad = deepcopy(target)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_resume_contract(source, bad, approval(bad))

    def test_explicit_unknown_qa_and_shared_reward_migration_is_narrow(self):
        """批准GS迁移也不能夹带学习率/batch/数据变化；来源和目标契约必须精确绑定。"""
        from copy import deepcopy

        source = dict(
            world_size=8,
            tasks_sha256="tasks",
            implementation_files={"pilot": "old"},
            config=dict(
                batch_size=128, learning_rate=1e-6, task_rule={"version": "libero-native-at-question-delivery-v1"}
            ),
        )
        target = deepcopy(source)
        target["implementation_files"]["pilot"] = "new"
        target["config"].update(
            unknown_qa_policy="retain_action_skip_qa_group",
            task_rule={"version": "libero-shared-at-question-delivery-v1"},
        )
        target["task_success_implementation"] = dict(version="libero-task-success-v1", files={"registry": "hash"})

        def approval(dst):
            """显式批准两个精确内容身份，不能只给宽泛allow标志。"""
            return dict(
                source_contract_sha256=contract_hash(source),
                target_contract_sha256=contract_hash(dst),
                protocol_revision="gs-v1.1-unknown-qa-v1",
            )

        validate_resume_contract(source, target, approval(target))
        with self.assertRaises(ValueError):
            validate_resume_contract(source, target)
        for key, value in [("batch_size", 64), ("learning_rate", 1e-4), ("unknown_qa_policy", "fill_zero")]:
            bad = deepcopy(target)
            bad["config"][key] = value
            with self.assertRaises(ValueError):
                validate_resume_contract(source, bad, approval(bad))
        bad = deepcopy(target)
        bad["tasks_sha256"] = "other"
        with self.assertRaises(ValueError):
            validate_resume_contract(source, bad, approval(bad))

    def test_source_start_migration_does_not_change_reward_or_training(self):
        """初态/分叉迁移须保持题目和优化配置，不能顺便改成功登记或将等待重新打开。"""
        from copy import deepcopy

        source = dict(
            world_size=8,
            config={"batch_size": 128},
            tasks_sha256="old",
            task_success_implementation={
                "version": "libero-task-success-v1",
                "files": {"task-success-v1.json": "registry", "monitor": "old"},
            },
        )
        target = deepcopy(source)
        target.update(
            tasks_sha256="new",
            task_rebind={"source_tasks_sha256": "old", "target_tasks_sha256": "new", "preserved_tasks_sha256": "same"},
            execution_boundary={
                "version": "rl-source-start-and-fork-v1",
                "wait_for_stability": False,
                "refresh": "libero-refresh-preserve-sampling-v1",
            },
        )
        target["task_success_implementation"]["files"]["monitor"] = "new"

        def approval(dst):
            """精确绑定来源和候选契约，供允许/拒绝的反例共享。"""
            return dict(
                source_contract_sha256=contract_hash(source),
                target_contract_sha256=contract_hash(dst),
                protocol_revision="gs-v1.1-source-start-v1",
                online_execution_rebind=True,
            )

        validate_resume_contract(source, target, approval(target))
        bad = deepcopy(target)
        bad["config"]["learning_rate"] = 1e-4
        with self.assertRaises(ValueError):
            validate_resume_contract(source, bad, approval(bad))
        bad = deepcopy(target)
        bad["execution_boundary"]["wait_for_stability"] = True
        with self.assertRaises(ValueError):
            validate_resume_contract(source, bad, approval(bad))
        bad = deepcopy(target)
        bad["task_success_implementation"]["files"]["task-success-v1.json"] = "changed"
        with self.assertRaises(ValueError):
            validate_resume_contract(source, bad, approval(bad))

    def test_two_approved_migrations_keep_both_checks(self):
        """旧完整存档连续跨两次修复；改算法、伪造中间契约或嵌套链均拒绝。"""
        from copy import deepcopy

        source = dict(
            world_size=8,
            tasks_sha256="source",
            config=dict(learning_rate=1e-6, task_rule={"version": "libero-native-at-question-delivery-v1"}),
        )
        middle = deepcopy(source)
        middle["config"].update(
            unknown_qa_policy="retain_action_skip_qa_group",
            task_rule={"version": "libero-shared-at-question-delivery-v1"},
        )
        middle["task_success_implementation"] = dict(
            version="libero-task-success-v1", files={"task-success-v1.json": "registry"}
        )
        target = deepcopy(middle)
        target.update(
            tasks_sha256="target",
            task_rebind=dict(source_tasks_sha256="source", target_tasks_sha256="target", preserved_tasks_sha256="same"),
            execution_boundary=dict(
                version="rl-source-start-and-fork-v1",
                wait_for_stability=False,
                refresh="libero-refresh-preserve-sampling-v1",
            ),
        )
        chain = dict(
            schema="physical-ai.gs-resume-chain.v1",
            intermediate_contract=middle,
            steps=[
                dict(
                    source_contract_sha256=contract_hash(source),
                    target_contract_sha256=contract_hash(middle),
                    protocol_revision="gs-v1.1-unknown-qa-v1",
                ),
                dict(
                    source_contract_sha256=contract_hash(middle),
                    target_contract_sha256=contract_hash(target),
                    protocol_revision="gs-v1.1-source-start-v1",
                    online_execution_rebind=True,
                ),
            ],
        )
        validate_resume_contract(source, target, chain)
        bad = deepcopy(target)
        bad["config"]["learning_rate"] = 1e-3
        approval = deepcopy(chain)
        approval["steps"][1]["target_contract_sha256"] = contract_hash(bad)
        with self.assertRaises(ValueError):
            validate_resume_contract(source, bad, approval)
        for change in ("middle", "nested", "order"):
            bad_chain = deepcopy(chain)
            if change == "middle":
                bad_chain["intermediate_contract"]["world_size"] = 4
            if change == "nested":
                bad_chain["steps"][0]["schema"] = "physical-ai.gs-resume-chain.v1"
            if change == "order":
                bad_chain["steps"].reverse()
            with self.assertRaises(ValueError):
                validate_resume_contract(source, target, bad_chain)

    def test_dummy_migration_rejects_duration_control_and_algorithm_changes(self):
        """源时钟不豁免准备；只允许固定10步开爪，不能顺便改变算法或任务目标。"""
        from copy import deepcopy

        source = dict(
            world_size=8,
            config={"batch_size": 128},
            tasks_sha256="old",
            execution_boundary={
                "version": "rl-source-start-and-fork-v1",
                "wait_for_stability": False,
                "refresh": "libero-refresh-preserve-sampling-v1",
            },
        )
        target = deepcopy(source)
        target.update(
            tasks_sha256="new",
            task_rebind={"source_tasks_sha256": "old", "target_tasks_sha256": "new", "preserved_tasks_sha256": "same"},
        )
        target["execution_boundary"].update(
            version="rl-online-preparation-v2",
            preparation={
                "version": "libero-online-dummy10-open-v1",
                "steps": 10,
                "control_hz": 20,
                "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                "implementation_sha256": "a" * 64,
            },
        )

        def approval(dst):
            """绑定本次来源和目标内容，不允许模糊放行。"""
            return dict(
                source_contract_sha256=contract_hash(source),
                target_contract_sha256=contract_hash(dst),
                protocol_revision="gs-v1.2-dummy10-v1",
                online_execution_rebind=True,
            )

        validate_resume_contract(source, target, approval(target))
        for key, value in (
            ("steps", 5),
            ("action", [0.0] * 7),
            ("control_hz", 10),
            ("implementation_sha256", "unknown"),
        ):
            bad = deepcopy(target)
            bad["execution_boundary"]["preparation"][key] = value
            with self.assertRaises(ValueError):
                validate_resume_contract(source, bad, approval(bad))
        bad = deepcopy(target)
        bad["config"]["batch_size"] = 64
        with self.assertRaises(ValueError):
            validate_resume_contract(source, bad, approval(bad))

    def test_sampling_axis_keeps_update_count_and_spikes(self):
        """每采样批仅一个日志点，更新累计数和逐mini尖峰不能被平均抹掉。"""
        from verl.experimental.physical_ai.training_metrics import sampling_batch_metrics

        rows = [
            {
                "train/sampling_batch": 5,
                "train/optimizer_steps": 17 + i,
                "train/optimizer_step_applied": 1,
                "train/mini_batch": i + 1,
                "actor/grad_norm": v,
                "timing/mini_update_seconds": 2,
            }
            for i, v in enumerate([4, 8, 12, 16])
        ]
        result = sampling_batch_metrics(rows)
        self.assertEqual(result["actor/grad_norm"], 10)
        self.assertEqual(result["actor/grad_norm_max"], 16)
        self.assertEqual(result["train/optimizer_steps"], 20)
        self.assertEqual(result["train/optimizer_steps_this_batch"], 4)
        self.assertEqual(result["mini_2/actor/grad_norm"], 8)
        self.assertNotIn("train/mini_batch", result)
        rows[0]["train/sampling_batch"] = 4
        with self.assertRaises(ValueError):
            sampling_batch_metrics(rows)

    def test_explicit_topology_migration(self):
        """批准须绑定精确代码/卡数版本，算法或数据变化即使重新签SHA也拒绝。"""
        source = dict(config={"batch_size": 128}, world_size=2, implementation_files={"train.py": "old"})
        target = dict(source, world_size=8, implementation_files={"train.py": "new"})
        approval = dict(source_contract_sha256=contract_hash(source), target_contract_sha256=contract_hash(target))
        with self.assertRaises(ValueError):
            validate_resume_contract(source, target)
        validate_resume_contract(source, target, approval)
        bad = dict(target, config={"batch_size": 64})
        approval["target_contract_sha256"] = contract_hash(bad)
        with self.assertRaises(ValueError):
            validate_resume_contract(source, bad, approval)

    def test_execution_rebind_requires_explicit_unchanged_task_proof(self):
        """新在线物理条件需独立批准；仅改task hash或更改算法均不能偷渡恢复。"""
        source = dict(config={"batch_size": 128}, world_size=8, tasks_sha256="source", eligible_ids=["a"])
        target = dict(
            source,
            tasks_sha256="target",
            task_rebind=dict(
                source_tasks_sha256="source", target_tasks_sha256="target", preserved_tasks_sha256="same-tasks"
            ),
        )
        approval = dict(source_contract_sha256=contract_hash(source), target_contract_sha256=contract_hash(target))
        with self.assertRaises(ValueError):
            validate_resume_contract(source, target, approval)
        approval["online_execution_rebind"] = True
        validate_resume_contract(source, target, approval)
        # 两次执行绑定可以共享同一个原始任务包，但必须保存完全相同的题目/初态投影。
        rebound = dict(
            target,
            tasks_sha256="third",
            task_rebind=dict(
                source_tasks_sha256="source", target_tasks_sha256="third", preserved_tasks_sha256="same-tasks"
            ),
        )
        chained = dict(
            source_contract_sha256=contract_hash(target),
            target_contract_sha256=contract_hash(rebound),
            online_execution_rebind=True,
        )
        validate_resume_contract(target, rebound, chained)
        rebound["task_rebind"]["preserved_tasks_sha256"] = "changed"
        chained["target_contract_sha256"] = contract_hash(rebound)
        with self.assertRaises(ValueError):
            validate_resume_contract(target, rebound, chained)
        target["eligible_ids"] = ["other"]
        approval["target_contract_sha256"] = contract_hash(target)
        with self.assertRaises(ValueError):
            validate_resume_contract(source, target, approval)

    def test_batch_budget_and_safe_retention(self):
        """100批按10批保存；新写入失败不能删旧恢复点，keep=1不碰其他实验/未完成文件。"""
        self.assertEqual(training_budget(dict(total_training_steps=100), 8), 800)
        self.assertEqual(training_budget(dict(max_groups=16), 8), 16)
        with self.assertRaises(ValueError):
            training_budget(dict(total_training_steps=100, max_groups=16), 8)
        due = [i for i in range(1, 101) if checkpoint_due(i, 10, final=i == 100)]
        self.assertEqual(due, list(range(10, 101, 10)))
        self.assertTrue(checkpoint_due(13, 10, final=True))
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = None
            for i in due:
                target = root / f"group-{i * 8:06d}"
                receipt = save_boundary(
                    model,
                    opt,
                    target,
                    completed_groups=i * 8,
                    optimizer_steps=i * 4,
                    contract={"run": "new"},
                    rng_states=[],
                )
                if previous:
                    # 构造未完成的新存档，不能用它触发清理。
                    broken = dict(receipt, path=str(root / "missing/training.pt"))
                    with self.assertRaises(ValueError):
                        prune_checkpoints(root, broken, 1)
                    self.assertTrue(previous.is_file())
                prune_checkpoints(root, receipt, 1)
                self.assertEqual(len(list(root.glob("group-*/checkpoint.json"))), 1)
                previous = target / "training.pt"
            foreign = save_boundary(
                model,
                opt,
                root / "group-000001",
                completed_groups=1,
                optimizer_steps=1,
                contract={"run": "other"},
                rng_states=[],
            )
            (root / "group-incomplete").mkdir()
            (root / "group-link").symlink_to(root / "group-000001", target_is_directory=True)
            prune_checkpoints(root, receipt, 1)
            self.assertTrue(Path(foreign["path"]).exists())
            self.assertTrue((root / "group-incomplete").is_dir())
            self.assertTrue((root / "group-link").is_symlink())
            state, _ = load_boundary(root / "group-000800/checkpoint.json", {"run": "new"})
            self.assertEqual(state["optimizer_steps"], 400)

    def test_unsupported_algorithm_switch_is_not_silently_ignored(self):
        """SwanLab配置必须对应实际实现，不能填了KL或其他采样分布却不执行。"""
        config = validate_fixed_algorithm({})
        self.assertEqual(config["advantage_estimator"], "seggrpo")
        for change in (dict(use_kl_loss=True), dict(entropy_coeff=0.01), dict(temperature=0.7)):
            with self.assertRaises(ValueError):
                validate_fixed_algorithm(change)

    def test_model_snapshots_survive_latest_optimizer_retention(self):
        """每20批的模型永久保留，完整恢复仅留最新；两份参数一致且写失败不删旧点。"""
        from unittest.mock import patch

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for batch in (20, 40):
                optimizer.zero_grad()
                model(torch.ones(1, 2)).sum().backward()
                optimizer.step()
                receipt = save_boundary(
                    model,
                    optimizer,
                    root / f"checkpoints/group-{batch * 8:06d}",
                    completed_groups=batch * 8,
                    optimizer_steps=batch * 4,
                    contract={"fixed": True},
                    rng_states=[],
                    model_directory=root / f"models/batch-{batch}",
                    completed_sampling_batches=batch,
                )
                snapshot = torch.load(receipt["model_snapshot"]["path"], weights_only=True)
                self.assertNotIn("optimizer_state_dict", snapshot)
                self.assertEqual(snapshot["completed_sampling_batches"], batch)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(snapshot["model_state_dict"][key], value, rtol=0, atol=0)
                prune_checkpoints(root / "checkpoints", receipt, 1)
            self.assertEqual(len(list((root / "checkpoints").glob("*/training.pt"))), 1)
            self.assertEqual(len(list((root / "models").glob("*/model.pt"))), 2)
            latest = root / "checkpoints/group-000320/checkpoint.json"
            state, _ = load_boundary(latest, {"fixed": True})
            self.assertTrue(state["optimizer_state_dict"]["state"])
            original_save = torch.save

            def fail_model(obj, path):
                """模拟纯模型写入失败，不把半套保存误发布为完整恢复点。"""
                if Path(path).name == "model.building":
                    raise OSError("disk full")
                return original_save(obj, path)

            with patch("torch.save", side_effect=fail_model), self.assertRaises(OSError):
                save_boundary(
                    model,
                    optimizer,
                    root / "checkpoints/group-000480",
                    completed_groups=480,
                    optimizer_steps=240,
                    contract={"fixed": True},
                    rng_states=[],
                    model_directory=root / "models/batch-60",
                    completed_sampling_batches=60,
                )
            self.assertFalse((root / "checkpoints/group-000480/checkpoint.json").exists())
            self.assertTrue(latest.exists())

    def test_zero_adam_matches_reference_after_resume(self):
        """优化器状态分片/同步及重载不能改变AdamW实际数值更新。"""
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_zero_adam_worker, args=(directory,), nprocs=2, join=True)
            mp.spawn(_reshard_adam_worker, args=(directory,), nprocs=8, join=True)

    def test_resume_matches_uninterrupted_optimizer(self):
        """连续两步与保存/重载后第二步应逐参数一致，任务/数据变化必须拒绝。"""
        torch.manual_seed(3)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        def step(m, o):
            """使用固定输入和真实反向更新，检查优化器动量是否接续。"""
            o.zero_grad()
            m(torch.ones(1, 3)).square().sum().backward()
            o.step()

        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            contract = dict(data_sha256="fixed-source", seed=42)
            save_boundary(
                model,
                optimizer,
                Path(directory) / "saved",
                completed_groups=1,
                optimizer_steps=1,
                contract=contract,
                rng_states=[dict(torch=torch.get_rng_state())],
            )
            manifest = Path(directory) / "saved/checkpoint.json"
            saved, receipt = load_boundary(manifest, contract)
            restored = torch.nn.Linear(3, 1)
            restored.load_state_dict(saved["model_state_dict"])
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01, momentum=0.9)
            restored_optimizer.load_state_dict(saved["optimizer_state_dict"])
            step(model, optimizer)
            step(restored, restored_optimizer)
            for a, b in zip(model.parameters(), restored.parameters(), strict=False):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(saved["completed_groups"], 1)
            with self.assertRaisesRegex(ValueError, "配置/数据"):
                load_boundary(manifest, dict(data_sha256="changed", seed=42))
            with Path(receipt["path"]).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA"):
                load_boundary(manifest, contract)
