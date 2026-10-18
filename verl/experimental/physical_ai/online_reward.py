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

"""从同次交付快照计算GT并调用既有文本judge；评分数据不进入模型会话。"""

import hashlib
import logging
import re
import time
from copy import deepcopy

from benchmark.core.scene_graph import canonical
from benchmark.libero.questions.context import PopContext
from benchmark.libero.questions.registry import default_registry
from benchmark.scoring.llm_judge import judge_answer
from ours.sft.shared.io import write as save_json  # noqa: F401


def qa_reward_status(binding, policy="stop"):
    """按已固定策略区分可评分与整组未知；数据错误/未支持不伪装成可忽略未知。

    只看交付GT，不读模型答案。返回Segment采用的状态；无效策略或其他GT错误
    仍拒绝。retain_action_skip_qa_group只关闭QA奖励和直接loss，不改变动作组。
    """
    if policy not in {"stop", "retain_action_skip_qa_group"}:
        raise ValueError("未知QA组处理策略")
    if binding["gt"]["status"] == "ok":
        return "scored"
    if binding["gt"]["status"] == "undetermined" and policy == "retain_action_skip_qa_group":
        from .seggrpo import UNSCORED_QA

        return UNSCORED_QA
    raise ValueError("选定父经历GT不可评分，禁止重选父轨迹或填零：" + str(binding["gt"].get("reason")))


def bind_delivery(question, recorder, names, rollout_id):
    """在真实追问交付处冻结GT与来源身份；后续分支动作不得改写该绑定。

    question为采样前固定题义，不含标准答案；recorder属于实际父轨迹。返回私有数据，
    包含绑定hash以拒绝把别的轨迹、题目或快照冒充本题GT。unknown原样保留。
    """
    snapshot = recorder.freeze(question["question_id"])
    gt = default_registry().pop(
        question,
        PopContext.create(
            snapshot,
            history=recorder.export(),
            entity_names=names,
            anchors={"instruction_delivery": 0},
        ),
    )
    payload = dict(rollout_id=rollout_id, question=deepcopy(question), gt=gt)
    payload["binding_sha256"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
    return payload


def public_text(text):
    """保留同轮所有对外文字，剔除显式think；未闭合think之后不当作公开答案。

    输入必须先由工具解析器取出该assistant消息的自然语言部分，本函数不猜动作码。
    think内的虚假指令也不能进入裁判作为答案，普通正文中的自然解释仍完整保留。
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    # 模板可能已经提供think起始标签，模型仅生成结束标签；这一段仍是私有思考。
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def score_answer(binding, answer, transport, *, rollout_id, max_attempts=20):
    """仅审核已完成响应的原始公开答案；失败/unknown保持None，不能静默填0。

    rollout_id是父轨迹身份，多个QA分支共享交付GT，但答案和裁判请求各自独立保存。
    max_attempts含首次请求，默认20次；失败后按5/10/20/40/60秒封顶退避。
    只重试相同答案，不重新生成或重跑机器人；有效0分不重试，没有格式/长度奖励。
    """
    payload = {k: deepcopy(v) for k, v in binding.items() if k != "binding_sha256"}
    if set(binding) != {"rollout_id", "question", "gt", "binding_sha256"}:
        raise ValueError("评分绑定字段不完整")
    if (
        binding["rollout_id"] != rollout_id
        or hashlib.sha256(canonical(payload).encode()).hexdigest() != binding["binding_sha256"]
    ):
        raise ValueError("评分绑定被改动或属于其他轨迹")
    calls = 0

    def retry_transport(messages):
        """仅在裁判已要求下一次重试时等待；不消耗策略随机数或改写消息。"""
        nonlocal calls
        if calls:
            delay = min(5 * 2 ** min(calls - 1, 4), 60)
            logging.getLogger(__name__).warning(
                "QA judge retry %d/%d after %ds; parent=%s question=%s",
                calls + 1,
                max_attempts,
                delay,
                rollout_id,
                binding["question"]["question_id"],
            )
            time.sleep(delay)
        calls += 1
        return transport(messages)

    result = judge_answer(binding["question"], binding["gt"], answer, retry_transport, max_attempts=max_attempts)
    result["retry_policy"] = dict(max_attempts=max_attempts, initial_delay_seconds=5, max_delay_seconds=60)
    return dict(
        result,
        delivery_binding_sha256=binding["binding_sha256"],
        parent_rollout_id=rollout_id,
    )
