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

"""在线奖励与会话分支的核心隔离检查，不调用外部API。"""

import hashlib
import json
import unittest
from types import SimpleNamespace

import numpy as np
from benchmark.core.scene_graph import canonical
from benchmark.scoring.llm_judge import JudgeAPIError

from verl.experimental.physical_ai.online_reward import public_text, score_answer


def binding():
    """构造最小已冻结交付契约，实际几何提取由现有pop测试负责。"""
    value = dict(
        rollout_id="parent-a",
        question=dict(
            question_id="q",
            question="Is the drawer open?",
            semantic_answer_type="boolean",
        ),
        gt=dict(
            question_id="q",
            snapshot_id="delivery",
            status="ok",
            value=False,
            reference_answer="No.",
            entity_names={},
        ),
    )
    value["binding_sha256"] = hashlib.sha256(canonical(value).encode()).hexdigest()
    return value


class _Controller:
    """用于验证绑定方法重绑定的最小控制器。"""

    def set_goal(self, value):
        """只改变本控制器目标。"""
        self.goal = value


class _UncopyableSim:
    """模拟含EGL指针的对象；任何递归复制都应让测试失败。"""

    def __deepcopy__(self, memo):
        raise AssertionError("sim不得被深拷贝")


class OnlineRewardTests(unittest.TestCase):
    """GT来源、裁判错误和物理副本的核心隔离检查。"""

    def test_runtime_clone_rebinds_controller_methods(self):
        """等待流程留下的实例绑定方法应指向子控制器，不递归复制或改动父sim。"""
        from ours.sft.execution.precision_encoding import clone_runtime

        def world():
            """仅提供被复制函数要求的运行状态，避免测试依赖真实EGL。"""
            sim = _UncopyableSim()
            sim.data = SimpleNamespace(_data={"q": np.ones(2)})
            sim.model = SimpleNamespace(**dict.fromkeys(["nq", "nv", "nu", "na", "nbody", "ngeom"], 1))
            controller = _Controller()
            controller.sim = sim
            controller.goal = np.ones(3)
            controller.set_goal = controller.set_goal
            return SimpleNamespace(
                sim=sim,
                env=SimpleNamespace(timestep=3, cur_time=0.15, done=False),
                robots=[
                    SimpleNamespace(
                        controller=controller,
                        gripper=SimpleNamespace(current_action=np.ones(1)),
                    )
                ],
            )

        parent, child = world(), world()
        clone_runtime(parent, child)
        self.assertIs(child.robots[0].controller.set_goal.__self__, child.robots[0].controller)
        child.robots[0].controller.set_goal(np.zeros(3))
        self.assertEqual(float(parent.robots[0].controller.goal.sum()), 3)

    def test_wrong_parent_and_mutated_gt_never_call_judge(self):
        """来源错误在发API之前拒绝，防止跨QA分支混用真值。"""

        def forbidden(messages):
            self.fail("不能为错误绑定调用judge")

        with self.assertRaises(ValueError):
            score_answer(binding(), "No.", forbidden, rollout_id="another-parent")
        bad = binding()
        bad["gt"]["value"] = True
        with self.assertRaises(ValueError):
            score_answer(bad, "No.", forbidden, rollout_id="parent-a")

    def test_api_error_is_not_zero_and_same_answer_is_retried(self):
        """重试判分而非重试模型，保留None奖励用于组级恢复。"""
        seen = []

        def failing(messages):
            seen.append(json.dumps(messages, sort_keys=True))
            raise JudgeAPIError("HTTP 500")

        result = score_answer(binding(), "No.", failing, rollout_id="parent-a", max_attempts=3)
        self.assertEqual(result["status"], "judge_error")
        self.assertIsNone(result["score"])
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 1)

    def test_private_think_and_public_explanation(self):
        """只去掉显式私有think，保留公开解释；未闭合think不得混成答案。"""
        self.assertEqual(
            public_text("<think>secret</think>No, it is closed.<|im_end|>"),
            "No, it is closed.",
        )
        self.assertEqual(public_text("<think>not finished"), "")
        self.assertEqual(public_text("private continuation</think>No."), "No.")
