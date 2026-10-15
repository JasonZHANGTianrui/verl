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

"""两阶段GRPO的核心契约；运行需要固定verl源码及其依赖，不用模拟器或真实模型。"""

import unittest
from dataclasses import replace

import torch
from omegaconf import OmegaConf

from verl.experimental.physical_ai.batching import minibatch_indices, namespace_group, prepare_batch
from verl.experimental.physical_ai.seggrpo import Segment, segment_advantages, segmented_policy_loss, select_parent
from verl.experimental.physical_ai.training_metrics import rollout_metrics, update_metrics


def fixture():
    """同一动作组含成功/失败；QA都基于失败父轨迹，仍然可以答对。"""
    execution = [
        Segment(f"a{i}", "execution", "action-group", "initial-hash", "case", "v1", 2, 0, 4, reward, "scored")
        for i, reward in enumerate([0, 1])
    ]
    qa = [
        Segment(
            f"q{i}",
            "qa",
            "qa-group",
            "prefix-question-gt-hash",
            "case",
            "v1",
            2,
            4,
            8,
            reward,
            "scored",
            parent_rollout_id="a0",
        )
        for i, reward in enumerate([0, 1])
    ]
    mask = torch.tensor([[1, 0, 1, 1, 0, 0, 0, 0]] * 2 + [[1, 0, 1, 1, 1, 0, 1, 1]] * 2, dtype=torch.bool)
    return execution + qa, mask


class SegGRPOTests(unittest.TestCase):
    """针对分组、反向和选择偏置测试，而不是只验证字段名字。"""

    def test_metrics_preserve_zero_groups_and_exclude_qa_prefix(self):
        """零优势仍计入总体奖励；QA长度/优势只取QA阶段，日志函数不修改准备数据。"""
        records, mask = fixture()
        records = [replace(r, reward=1) if r.phase == "qa" else r for r in records]
        artifacts = [dict(generated=[False] + m.tolist(), old_logprobs=[0.0] + [-2.0] * len(m)) for m in mask]
        prepared = prepare_batch(records, artifacts)
        original = prepared["advantages"].clone()
        metrics = rollout_metrics(records, artifacts, prepared)
        self.assertEqual(metrics["rollout/execution/reward_mean"], 0.5)
        self.assertEqual(metrics["rollout/qa/reward_mean"], 1)
        self.assertEqual(metrics["rollout/qa/zero_variance_group_fraction"], 1)
        self.assertEqual(metrics["rollout/qa/phase_tokens_mean"], 3)
        self.assertEqual(metrics["rollout/qa/old_sampled_nll"], 2)
        self.assertEqual(metrics["rollout/qa/advantage_abs_mean"], 0)
        torch.testing.assert_close(prepared["advantages"], original)
        skipped = update_metrics(dict(optimizer_step=False), records, list(range(4)))
        self.assertEqual(skipped["actor/policy_loss"], 0)
        self.assertNotIn("actor/grad_norm", skipped)
        self.assertNotIn("actor/qa/active_ppo_kl", skipped)

    def test_complete_task_minibatches_and_frozen_advantages(self):
        """128条按8＋8完整任务切成四个32条mini；缺QA、拆任务和混策略均拒绝。"""
        records, artifacts = [], []
        for task in range(8):
            rows = [
                Segment(f"a{i}", "execution", "a", "initial", f"c{task}", "v1", 8, 0, 4, i % 2, "scored")
                for i in range(8)
            ]
            rows += [
                Segment(
                    f"q{i}", "qa", "q", "prefix", f"c{task}", "v1", 8, 4, 8, i % 2, "scored", parent_rollout_id="a0"
                )
                for i in range(8)
            ]
            records.extend(namespace_group(rows, f"task{task}"))
            artifacts.extend([dict(generated=[False] + [True] * 8, old_logprobs=[0.0] + [-1.0] * 8) for _ in rows])
        batch = prepare_batch(records, artifacts)
        old = batch["old"].clone()
        minis = minibatch_indices(records, 32)
        self.assertEqual([len(m) for m in minis], [32] * 4)
        for indices in minis:
            sub = [records[i] for i in indices]
            self.assertEqual(len({r.case_id for r in sub}), 2)
            adv, mask, _ = segment_advantages(sub, torch.ones((32, 8), dtype=torch.bool))
            torch.testing.assert_close(adv, batch["advantages"][indices])
            self.assertFalse(bool(mask[[i for i, r in enumerate(sub) if r.phase == "qa"], :4].any()))
        with self.assertRaises(ValueError):
            minibatch_indices(records, 24)
        with self.assertRaises(ValueError):
            minibatch_indices(records[:-8], 32)
        with self.assertRaises(ValueError):
            prepare_batch([replace(records[0], policy_version="v2")] + records[1:], artifacts)
        # 模拟后续mini的策略已明显改变；old不变，clip应消除越界方向的策略梯度。
        indices = minis[1]
        new = (old[indices] + 0.5 * batch["advantages"][indices].sign()).requires_grad_()
        cfg = OmegaConf.create(
            dict(clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None, clip_ratio_c=3.0, global_batch_info={})
        )
        loss, metrics = segmented_policy_loss(
            [records[i] for i in indices], old[indices], new, torch.ones((32, 8), dtype=torch.bool), actor_config=cfg
        )
        loss.backward()
        self.assertTrue(torch.equal(new.grad, torch.zeros_like(new)))
        self.assertTrue(any(v > 0 for k, v in metrics.items() if "pg_clipfrac" in k and "lower" not in k))
        torch.testing.assert_close(batch["old"], old)

    def test_stage_advantages_and_prefix_gradient(self):
        """QA答对可获正优势，即使父动作失败；复制前缀没有第二次策略梯度。"""
        records, mask = fixture()
        advantages, active, _ = segment_advantages(records, mask)
        self.assertLess(float(advantages[0, 0]), 0)
        self.assertGreater(float(advantages[1, 0]), 0)
        self.assertGreater(float(advantages[3, 4]), 0)
        self.assertFalse(bool(active[2:, :4].any()))
        self.assertFalse(bool(active[:, 5].any()))
        old = torch.zeros(mask.shape, requires_grad=True)
        new = torch.zeros(mask.shape, requires_grad=True)
        config = OmegaConf.create(
            dict(clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None, clip_ratio_c=3.0, global_batch_info={})
        )
        loss, _ = segmented_policy_loss(records, old, new, mask, actor_config=config)
        loss.backward()
        expected = -advantages / active.sum(1, keepdim=True) / 2
        torch.testing.assert_close(new.grad, expected)
        self.assertIsNone(old.grad)
        self.assertTrue(torch.equal(new.grad[~active], torch.zeros_like(new.grad[~active])))

    def test_parent_choice_does_not_follow_rewards_or_finish_order(self):
        """默认选择只由组身份/seed决定，翻转成败、改变完成顺序都不改变父ID。"""
        rows, _ = fixture()
        execution = rows[:2]
        selections = set()
        for seed in range(32):
            chosen = select_parent(execution, seed=seed)
            selections.add(chosen)
            flipped = [replace(r, reward=1 - r.reward) for r in reversed(execution)]
            self.assertEqual(chosen, select_parent(flipped, seed=seed))
        self.assertEqual(selections, {"a0", "a1"})
        self.assertEqual(select_parent(execution, seed=0, mode="success_only"), "a1")
        self.assertIsNone(select_parent([replace(r, reward=0) for r in execution], seed=0, mode="success_only"))

    def test_zero_variance_and_invalid_groups(self):
        """全同奖励组无优势；未知奖励、缺样本、混策略或混父轨迹均拒绝训练。"""
        rows, mask = fixture()
        same = [replace(r, reward=1) for r in rows]
        advantage, _, metrics = segment_advantages(same, mask)
        self.assertTrue(torch.equal(advantage, torch.zeros_like(advantage)))
        self.assertEqual(metrics["zero_variance_groups"], 2)
        invalid = [
            rows[:-1],
            [replace(rows[0], reward=None, reward_status="judge_error")] + rows[1:],
            rows[:3] + [replace(rows[3], policy_version="v2")],
            rows[:3] + [replace(rows[3], parent_rollout_id="a1")],
            rows[:3] + [replace(rows[3], condition_id="another-scene")],
        ]
        for records in invalid:
            with self.assertRaises(ValueError):
                segment_advantages(records, mask[: len(records)])


def _distributed_gradient_worker(rank, rendezvous):
    """两个真实Gloo进程对照完整组反向；包含不均匀分片和全零QA优势。"""
    import torch.distributed as dist

    from verl.experimental.physical_ai.distributed import sum_gradients
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        for qa_rewards in ([0, 1], [1, 1]):
            torch.manual_seed(7)
            reference = torch.nn.Linear(3, 8)
            model = torch.nn.Linear(3, 8)
            model.load_state_dict(reference.state_dict())
            rows, mask = fixture()
            rows = [
                replace(rows[0], group_size=3),
                replace(rows[1], group_size=3),
                replace(rows[0], rollout_id="a2", group_size=3),
            ] + [replace(row, reward=reward) for row, reward in zip(rows[2:], qa_rewards, strict=False)]
            mask = torch.cat((mask[:2], mask[:1], mask[2:]))
            x = torch.randn(5, 3)
            old = reference(x).detach()
            config = OmegaConf.create(
                dict(clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None, clip_ratio_c=3.0, global_batch_info={})
            )
            total, _ = segmented_policy_loss(rows, old, reference(x), mask, actor_config=config)
            total.backward()
            advantages, active, _ = segment_advantages(rows, mask)
            for i in range(rank, len(rows), 2):
                if not bool(advantages[i].any()):
                    continue
                value, _ = compute_policy_loss_vanilla(
                    old[i : i + 1],
                    model(x[i : i + 1]),
                    advantages[i : i + 1],
                    active[i : i + 1].float(),
                    loss_agg_mode="seq-mean-token-mean",
                    config=config,
                )
                (value / (3 if rows[i].phase == "execution" else 2)).backward()
            sum_gradients(model)
            for expected, actual in zip(reference.parameters(), model.parameters(), strict=False):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.step()
            for expected, actual in zip(reference.parameters(), model.parameters(), strict=False):
                torch.testing.assert_close(actual, expected - 0.01 * expected.grad, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


class DistributedSegGRPOTests(unittest.TestCase):
    """多进程必须与单进程完整组的梯度及一步更新一致，而不只检查通信是否成功。"""

    def test_global_gradient_and_update_match_reference(self):
        """三条动作、两条QA分给两进程，验证不会按本地组或卡数再次缩放。"""
        import tempfile
        from pathlib import Path

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_gradient_worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
