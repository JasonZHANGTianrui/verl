"""普通GRPO的核心契约：完整会话单奖励、组内归一化、缺QA与数据错误分流。"""

import unittest
from dataclasses import replace

import torch

from verl.experimental.physical_ai import grpo
from verl.experimental.physical_ai.seggrpo import Segment, validate_groups


def records(rewards):
    """构造同一初态/策略的独立完整轨迹，token索引对齐next-token概率。"""
    return [
        Segment(str(i), "trajectory", "g", "initial", "case", "policy", len(rewards), 1, 6, r, "scored")
        for i, r in enumerate(rewards)
    ]


def artifacts(n):
    """动作位置1、2和QA位置5为生成token；中间用户/工具位置必须不参与loss。"""
    return [
        dict(generated=[False, False, True, True, False, False, True], old_logprobs=[0, 0, -1, -2, 0, 0, -3])
        for _ in range(n)
    ]


class GRPOTests(unittest.TestCase):
    def test_reward_truth_table_and_task_fallback(self):
        """奖励只在QA明确不可交付时重加权；已交付的QA零分仍占一半。"""
        for task in (0, 1):
            for qa in (0, 1):
                self.assertEqual(grpo.combined_reward(task, qa, qa_status="scored"), (task + qa) / 2)
            for status in ("no_question", "undetermined", "incomplete_execution", "context_capacity"):
                self.assertEqual(grpo.combined_reward(task, qa_status=status), task)
        for status in ("api_error", "invalid_data", "not_supported", "wall_time_limit", "scored"):
            with self.assertRaises(ValueError):
                grpo.combined_reward(1, qa_status=status)
        with self.assertRaises(ValueError):
            grpo.combined_reward(1, 0, qa_status="undetermined")

    def test_unknown_ends_before_qa_not_after_answer(self):
        """QA门控只看交付事实，不按模型可能答对与否更换题目或父经历。"""
        self.assertEqual(grpo.qa_gate(None, {}, None), "no_question")
        self.assertEqual(grpo.qa_gate({}, {"incomplete_message": True}, None), "incomplete_execution")
        self.assertEqual(grpo.qa_gate({}, {}, {"gt": {"status": "undetermined"}}), "undetermined")
        self.assertEqual(grpo.qa_gate({}, {}, {"gt": {"status": "ok"}}), "deliver")
        for status in ("invalid_data", "not_supported", "other"):
            with self.assertRaises(ValueError):
                grpo.qa_gate({}, {}, {"gt": {"status": status}})

    def test_whole_trajectory_advantage_and_observation_mask(self):
        """手工计算std核对一次归一化，动作与QA同优势，工具/用户位置全零。"""
        r = records([0, 0.5, 1, 1])
        a = artifacts(4)
        v = grpo.prepare_batch(r, a)
        reward = torch.tensor([0.0, 0.5, 1.0, 1.0])
        expected = (reward - reward.mean()) / (reward.std() + 1e-6)
        for i in range(4):
            torch.testing.assert_close(v["advantages"][i, v["active"][i]], expected[i].repeat(3))
        self.assertEqual(v["active"].tolist(), [[False, True, True, False, False, True]] * 4)
        self.assertEqual(v["old"][0].tolist(), [0, -1, -2, 0, 0, -3])
        self.assertFalse(v["advantages"][~v["active"]].any())

    def test_task_only_and_zero_variance(self):
        """task-only轨迹仍保留在同一完整组；全部同分时不制造有效梯度。"""
        a = artifacts(2)
        a[0]["generated"][-1] = False
        out = grpo.prepare_batch(records([1, 1]), a)
        self.assertFalse(out["advantages"].any())
        self.assertEqual(int(out["active"][0].sum()), 2)
        self.assertEqual(out["stats"]["zero_variance_groups"], 1)

    def test_reject_incomplete_mixed_and_duplicate_groups(self):
        """原组不能静默缩减；跨初态、策略、父轨迹或不合法奖励均拒绝。"""
        r = records([0, 1])
        with self.assertRaises(ValueError):
            grpo.task_groups(r[:1])
        for changes in [
            dict(condition_id="other"),
            dict(policy_version="other"),
            dict(parent_rollout_id="0"),
            dict(reward=None),
            dict(reward=0.7),
            dict(phase="qa"),
            dict(reward_status="unscored_unknown_gt"),
        ]:
            with self.assertRaises(ValueError):
                grpo.task_groups([r[0], replace(r[1], **changes)])
        with self.assertRaises(ValueError):
            grpo.task_groups([r[0], r[0]])
        # 普通记录不能偷偷通过SegGRPO原校验；其契约没有被放宽。
        with self.assertRaises(ValueError):
            validate_groups(r)

    def test_namespace_and_minibatch_keep_groups(self):
        """两个任务独立归一化，mini只能按完整n轨迹组切分。"""
        r = grpo.namespace_group(records([0, 1]), "a") + grpo.namespace_group(records([1, 0]), "b")
        self.assertEqual(grpo.minibatch_indices(r, 2), [[0, 1], [2, 3]])
        with self.assertRaises(ValueError):
            grpo.minibatch_indices(r, 3)
        self.assertTrue(all(x.parent_rollout_id is None for x in r))

    def test_config_does_not_mutate_or_accept_segment_fields(self):
        """配方独立且默认算法的固定校验不变，不通过全局替换启用普通GRPO。"""
        from verl.experimental.physical_ai.training_state import validate_fixed_algorithm

        cfg = dict(advantage_estimator="grpo", group_size=8)
        grpo.validate_config(cfg)
        self.assertEqual(cfg, dict(advantage_estimator="grpo", group_size=8))
        with self.assertRaises(ValueError):
            validate_fixed_algorithm(cfg)
        for key in ("qa_group_size", "execution_group_size", "parent_selection", "unknown_qa_policy"):
            with self.assertRaises(ValueError):
                grpo.validate_config(dict(cfg, **{key: 8}))

    def test_default_dispatch_returns_original_seggrpo_callbacks(self):
        """默认训练装配确实返回原函数对象，普通GRPO不替换/复制SegGRPO实现。"""
        from verl.experimental.physical_ai import batching, pilot
        from verl.experimental.physical_ai.algorithms import resolve_algorithm

        a = resolve_algorithm(dict(execution_group_size=8, qa_group_size=8))
        self.assertIs(a.prepare_batch, batching.prepare_batch)
        self.assertIs(a.collect_group, pilot.collect_group)
        self.assertEqual(a.width, 16)
        b = resolve_algorithm(dict(advantage_estimator="grpo", group_size=8))
        self.assertIs(b.prepare_batch, grpo.prepare_batch)
        self.assertEqual(b.width, 8)


if __name__ == "__main__":
    unittest.main()
