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
from unittest.mock import patch

import numpy as np
from benchmark.core.scene_graph import SceneRecorder, canonical
from benchmark.scoring.llm_judge import JudgeAPIError

from verl.experimental.physical_ai.online_reward import public_text, score_answer
from verl.experimental.physical_ai.task_data import state_task_reward


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

    def test_runtime_preparation_contract_is_explicit(self):
        """取消逐样本预演不取消版本/seed/分辨率校验，旧回执模式继续严格读取。"""
        from verl.experimental.physical_ai.task_data import needs_preparation_reference

        online = dict(
            schema="physical-ai.rl-online-execution.v2",
            preparation_policy="verify_during_rollout",
            seed=42,
            resolution=256,
        )
        self.assertFalse(needs_preparation_reference(online, seed=42, resolution=256))
        for changed in (
            dict(online, seed=1),
            dict(online, preparation_policy="skip"),
            dict(online, preparation_ref="old"),
        ):
            with self.assertRaises(ValueError):
                needs_preparation_reference(changed, seed=42, resolution=256)
        old = dict(schema="physical-ai.rl-online-execution.v1", seed=42, resolution=256, preparation_ref="receipt")
        self.assertTrue(needs_preparation_reference(old, seed=42, resolution=256))
        del old["preparation_ref"]
        with self.assertRaises(ValueError):
            needs_preparation_reference(old, seed=42, resolution=256)

    def test_evaluation_action_reward_reads_live_state_at_delivery(self):
        """新模式直接读取当前物理结果；旧源标签和QA答案不能代替，运行异常不能抹零。"""
        from verl.experimental.physical_ai.task_data import EVALUATION_ACTION_RULE, task_reward_at_delivery

        task = dict(environment=dict(task_name="arbitrary", suite="libero_goal"), outcome=dict(success=True))
        session = SimpleNamespace(env=object(), steps=25)
        rule = dict(version=EVALUATION_ACTION_RULE)
        with patch("benchmark.adapters.libero_execution.task_success", return_value=False) as read:
            result = task_reward_at_delivery(task, rule, session, None, None, "model_text_end")
            self.assertEqual(result["task_reward"], 0)
            read.assert_called_once_with(session.env)
        with patch("benchmark.adapters.libero_execution.task_success", return_value=True):
            self.assertEqual(
                task_reward_at_delivery(task, rule, session, None, None, "execution_budget")["task_reward"], 1
            )
            self.assertEqual(
                task_reward_at_delivery(task, rule, session, None, None, "model_tool_error")["task_reward"], 1
            )
        with patch("benchmark.adapters.libero_execution.task_success", side_effect=RuntimeError("sim failed")):
            with self.assertRaises(RuntimeError):
                task_reward_at_delivery(task, rule, session, None, None, "model_text_end")

    def test_evaluation_mode_preserves_task_pool_and_checks_each_query(self):
        """不同动作任务均进入池；每条问题走同一注册表，不用动作目标代替QA查询。"""
        from verl.experimental.physical_ai.task_data import EVALUATION_ACTION_RULE, training_tasks

        rows = [
            dict(
                split="train",
                environment=dict(task_name=name, suite="libero_goal"),
                followup=dict(question_spec=dict(question_id=name)),
            )
            for name in ("open", "place", "close")
        ]
        with patch("benchmark.libero.questions.registry.default_registry") as factory:
            self.assertEqual(training_tasks(rows, dict(version=EVALUATION_ACTION_RULE)), rows)
            self.assertEqual(factory.return_value.validate.call_count, 3)
            factory.return_value.validate.side_effect = ValueError("query not supported")
            with self.assertRaises(ValueError):
                training_tasks(rows, dict(version=EVALUATION_ACTION_RULE))

    def test_task_reward_is_independent_of_qa_and_old_result(self):
        """QA可以问数量，动作仍按当前设备状态计分；缺状态或错任务不能填零。"""
        task = dict(
            environment=dict(task_name="open_drawer"), followup=dict(answer=999), outcome=dict(task_success=True)
        )
        rule = dict(
            version="rl-device-state-goal-v1",
            task_name="open_drawer",
            calibration="cal-v1",
            query=dict(op="attribute", entity="drawer", path="states.opening.value"),
            allowed_states=["open", "closed", "partially_open"],
            target_state="open",
        )
        reader = SimpleNamespace(config=dict(version="cal-v1"))
        for state, expected in [("open", 1.0), ("closed", 0.0), (None, None)]:
            recorder = SceneRecorder("actual-run", max_gap_s=0.1)
            recorder.observe(
                dict(
                    step=0,
                    time_s=0.0,
                    versions={},
                    coverage={},
                    relations=[],
                    nodes={"drawer": dict(id="drawer", kind="object", states=dict(opening=dict(value=state)))},
                )
            )
            if expected is None:
                with self.assertRaises(ValueError):
                    state_task_reward(task, rule, reader, recorder)
            else:
                self.assertEqual(state_task_reward(task, rule, reader, recorder)["task_reward"], expected)
        task["environment"]["task_name"] = "other_task"
        with self.assertRaises(ValueError):
            state_task_reward(task, rule, reader, recorder)

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

        with patch("verl.experimental.physical_ai.online_reward.time.sleep"):
            result = score_answer(binding(), "No.", failing, rollout_id="parent-a", max_attempts=3)
        self.assertEqual(result["status"], "judge_error")
        self.assertIsNone(result["score"])
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 1)

    def test_retry_recovers_on_twentieth_identical_request(self):
        """模拟长期API故障后恢复：20次同一输入、退避封顶、有效0分立即返回。"""
        seen = []

        def transport(messages):
            seen.append(json.dumps(messages, sort_keys=True))
            if len(seen) < 20:
                raise JudgeAPIError("HTTP 500")
            return {"raw_response": json.dumps(dict(status="scored", score=0, reason="Wrong answer"))}

        with patch("verl.experimental.physical_ai.online_reward.time.sleep") as sleep:
            result = score_answer(binding(), "Yes.", transport, rollout_id="parent-a")
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["score"], 0)
        self.assertEqual(len(seen), 20)
        self.assertEqual(len(set(seen)), 1)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5, 10, 20, 40] + [60] * 15)
        self.assertEqual(len(result["attempts"]), 20)

    def test_private_think_and_public_explanation(self):
        """只去掉显式私有think，保留公开解释；未闭合think不得混成答案。"""
        self.assertEqual(
            public_text("<think>secret</think>No, it is closed.<|im_end|>"),
            "No, it is closed.",
        )
        self.assertEqual(public_text("<think>not finished"), "")
        self.assertEqual(public_text("private continuation</think>No."), "No.")
