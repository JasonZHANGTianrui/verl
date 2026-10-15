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

"""按完整动作/QA任务组组织采样batch，冻结优势后切mini-batch。"""

from dataclasses import replace

import torch

from .seggrpo import segment_advantages, validate_groups


def namespace_group(records, prefix):
    """为不同真实任务的响应/组/父身份加唯一前缀，不修改条件、奖励、token区间。"""
    validate_groups(records)
    return [
        replace(
            r,
            rollout_id=prefix + "/" + r.rollout_id,
            group_id=prefix + "/" + r.group_id,
            parent_rollout_id=prefix + "/" + r.parent_rollout_id if r.parent_rollout_id else None,
        )
        for r in records
    ]


def task_groups(records):
    """返回每个完整任务的行号；每个动作组必须恰有一个QA组，不拆父经历。

    同一batch允许多个case，但所有组须来自同一行为策略。组内奖励0/1和缺样本
    沿validate_groups核验；相同case的重复采样也必须用不同group/rollout身份。
    """
    groups = validate_groups(records)
    by_id = {r.rollout_id: r for r in records}
    pairs = {g: list(ids) for g, ids in groups.items() if records[ids[0]].phase == "execution"}
    attached = set()
    for ids in groups.values():
        r = records[ids[0]]
        if r.phase == "qa":
            parent = by_id[r.parent_rollout_id].group_id
            if parent in attached:
                raise ValueError("同一动作组不能重复附加QA组")
            pairs[parent].extend(ids)
            attached.add(parent)
    if attached != set(pairs):
        raise ValueError("有动作组缺少完整QA，不能缩减训练batch")
    for phase in ("execution", "qa"):
        if len({r.group_size for r in records if r.phase == phase}) != 1:
            raise ValueError("同阶段组大小不同，不能隐式按样本量改变任务权重")
    return list(pairs.values())


def minibatch_indices(records, size):
    """按完整任务切mini-batch；size的单位是阶段序列数，不是任务数或token数。"""
    pairs = task_groups(records)
    width = len(pairs[0])
    if type(size) is not int or size < width or size % width or len(records) % size:
        raise ValueError("mini-batch必须由完整任务组组成且整除本次batch")
    ordered = [i for pair in pairs for i in pair]
    return [ordered[i : i + size] for i in range(0, len(ordered), size)]


def prepare_batch(records, artifacts):
    """在任何参数更新前一次性计算全batch的组内优势，后续mini-batch只索引它。

    old log-prob保持采样原值；QA的父历史由阶段边界mask排除。补齐仅在CPU上对齐
    元数据，不裁剪/拼接模型输入，不把环境反馈标为策略生成。
    """
    task_groups(records)
    if len(records) != len(artifacts):
        raise ValueError("轨迹与响应记录数不同")
    length = max(len(a["generated"]) - 1 for a in artifacts)
    masks = torch.zeros((len(records), length), dtype=torch.bool)
    old = torch.zeros((len(records), length))
    for i, a in enumerate(artifacts):
        if len(a["generated"]) != len(a["old_logprobs"]):
            raise ValueError("原始概率和生成mask长度不一致")
        n = len(a["generated"]) - 1
        masks[i, :n] = torch.tensor(a["generated"][1:])
        old[i, :n] = torch.tensor(a["old_logprobs"][1:])
    if not bool(torch.isfinite(old).all()):
        raise ValueError("采样概率含非有限值")
    advantages, active, stats = segment_advantages(records, masks)
    return dict(old=old, advantages=advantages, active=active, stats=stats)
