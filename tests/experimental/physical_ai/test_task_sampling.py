"""任务抽样仅改频率：固定配额、叶目标去重、全部覆盖、恢复游标及跨题隔离。"""

import unittest
from collections import Counter
from copy import deepcopy

from verl.experimental.physical_ai.task_sampling import build_schedule, goal_count


def pool():
    """创建与RL160相同类别数量，问题类型和ID固定，不包含奖励字段。"""
    return [
        dict(
            id=f"{count}-{i}",
            objective=dict(goals=[["On", f"obj-{j}", "table"] for j in range(count)]),
            followup=dict(question_spec=dict(strategy_id="Q01"), selection_class="Trick" if i % 7 == 0 else "Q01"),
        )
        for count, size in [(1, 91), (2, 52), (3, 17)]
        for i in range(size)
    ]


class TaskSamplingTest(unittest.TestCase):
    def test_exact_counts_round_robin_and_recovery(self):
        """800槽精确560/200/40，类内次数差≤1；相同输入/seed恢复同一后缀。"""
        tasks = pool()
        before = deepcopy(tasks)
        spec = dict(version="fixed-goal-quota-v1", counts={"1": 560, "2": 200, "3": 40})
        a = build_schedule(tasks, 800, 42, spec)
        b = build_schedule(list(reversed(tasks)), 800, 42, spec)
        self.assertEqual(a, b)
        self.assertEqual(tasks, before)
        self.assertEqual(Counter(a["goal_counts"]), {1: 560, 2: 200, 3: 40})
        self.assertEqual(len(set(a["task_ids"])), 160)
        for count in (1, 2, 3):
            values = [a["per_task"][t["id"]] for t in tasks if goal_count(t) == count]
            self.assertLessEqual(max(values) - min(values), 1)
        self.assertEqual(a["task_ids"][320:], b["task_ids"][320:])
        self.assertNotEqual(a["sha256"], build_schedule(tasks, 800, 43, spec)["sha256"])
        self.assertIsNone(build_schedule(tasks, 800, 42))

    def test_leaf_count_and_invalid_plan(self):
        """同叶重复不加目标数，英文长度/成绩不参与分类；不合法总预算拒绝。"""
        t = pool()[0]
        t["objective"]["goals"] *= 2
        t["instruction"] = "and " * 20
        self.assertEqual(goal_count(t), 1)
        spec = dict(version="fixed-goal-quota-v1", counts={"1": 560, "2": 200, "3": 40})
        with self.assertRaises(ValueError):
            build_schedule(pool(), 1600, 42, spec)
        with self.assertRaises(ValueError):
            build_schedule(pool()[:91], 800, 42, spec)
        t["objective"]["goals"] = [["And", ["On", "x", "y"]]]
        with self.assertRaises(ValueError):
            goal_count(t)


if __name__ == "__main__":
    unittest.main()
