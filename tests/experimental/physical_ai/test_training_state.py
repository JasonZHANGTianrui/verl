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
    load_boundary,
    prune_checkpoints,
    restore_optimizer,
    save_boundary,
    training_budget,
    validate_fixed_algorithm,
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


class TrainingStateTests(unittest.TestCase):
    """用有动量的小模型核对真实优化器恢复，不用无状态SGD掩盖漏存状态。"""

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

    def test_zero_adam_matches_reference_after_resume(self):
        """优化器状态分片/同步及重载不能改变AdamW实际数值更新。"""
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_zero_adam_worker, args=(directory,), nprocs=2, join=True)

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
