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

"""SegGRPO分组与策略loss参考实现；复用verl GRPO，不负责采样器或分布式切分。"""

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Segment:
    """一条阶段响应的评分绑定，由可信rollout控制器产生，不能来自模型文本。

    condition_id绑定同一case/初态；QA时还须绑定父轨迹完整前缀、问题和交付GT。
    start/end是对齐log_probs与generated_mask列的半开区间，不是环境步或字符位置。
    group_size是采样前声明的完整组大小；reward_status只有scored允许进入更新。
    """

    rollout_id: str
    phase: str
    group_id: str
    condition_id: str
    case_id: str
    policy_version: str
    group_size: int
    start: int
    end: int
    reward: float | None
    reward_status: str
    parent_rollout_id: str | None = None


def validate_groups(records):
    """拒绝缺分/缺样本、混条件/策略/阶段、重复响应及未绑定父轨迹的组。

    返回group_id到行号列表。调用前须收齐整个训练组；不对单个micro-batch重算优势，
    不将基础设施错误填0。条件内容的hash核验由采集层负责，本函数检查已绑定身份。
    """
    if not records:
        raise ValueError("空训练批次")
    groups = defaultdict(list)
    ids = {}
    for index, record in enumerate(records):
        if record.rollout_id in ids:
            raise ValueError("rollout_id重复，不能重复训练父轨迹")
        ids[record.rollout_id] = record
        if any(
            not isinstance(getattr(record, name), str) or not getattr(record, name)
            for name in ["rollout_id", "group_id", "condition_id", "case_id", "policy_version"]
        ):
            raise ValueError("缺少稳定组/条件/策略身份")
        if record.phase not in {"execution", "qa"}:
            raise ValueError("未知训练阶段")
        if type(record.group_size) is not int or record.group_size < 2:
            raise ValueError("GRPO完整组至少两条，不能退化成单样本")
        if type(record.start) is not int or type(record.end) is not int or not 0 <= record.start < record.end:
            raise ValueError("阶段token边界无效")
        if record.reward_status != "scored" or record.reward is None or not math.isfinite(record.reward):
            raise ValueError("缺少有效奖励，保留组等待重试，不自动记零")
        if record.reward not in (0, 1):
            raise ValueError("当前动作/QA契约只接受0或1")
        groups[record.group_id].append(index)
    if len({r.policy_version for r in records}) != 1:
        raise ValueError("两段必须来自同一版行为策略")
    for indices in groups.values():
        first = records[indices[0]]
        if len(indices) != first.group_size:
            raise ValueError("训练组尚未收齐，不能静默缩组")
        identities = {
            (
                records[i].phase,
                records[i].condition_id,
                records[i].case_id,
                records[i].group_size,
                records[i].parent_rollout_id,
            )
            for i in indices
        }
        if len(identities) != 1:
            raise ValueError("同一group混入不同阶段、条件或父轨迹")
    qa_groups = defaultdict(set)
    for record in records:
        if record.phase == "execution":
            if record.parent_rollout_id is not None:
                raise ValueError("动作阶段不能有QA父轨迹")
        else:
            parent = ids.get(record.parent_rollout_id)
            if parent is None or parent.phase != "execution" or parent.case_id != record.case_id:
                raise ValueError("QA未绑定本批同case的真实动作父轨迹")
            qa_groups[parent.group_id].add(record.group_id)
    if any(len(value) != 1 for value in qa_groups.values()):
        raise ValueError("当前方案每个动作组只允许一个QA父轨迹/分支组")
    return dict(groups)


def select_parent(records, *, seed, mode="uniform"):
    """从单个完整动作组选父轨迹；默认与成败无关，可复现且不受完成顺序影响。

    seed为运行配置中的整数，混合group及policy身份使不同组不总选相同序号。
    success_only只用于显式消融，无成功父轨迹时返回None，由调度器登记QA未展开；
    不默默回退成uniform，也不删除动作组。此方法不改原记录或训练分数。
    """
    groups = validate_groups(records)
    if len(groups) != 1 or any(r.phase != "execution" for r in records):
        raise ValueError("父轨迹必须来自一个完整动作组")
    if type(seed) is not int or mode not in {"uniform", "success_only"}:
        raise ValueError("需要整数seed和受支持的父轨迹选择模式")
    candidates = sorted((r for r in records if mode == "uniform" or r.reward == 1), key=lambda r: r.rollout_id)
    if not candidates:
        return None
    first = records[0]
    identity = json.dumps([seed, first.group_id, first.policy_version], separators=(",", ":"))
    rng = random.Random(int(hashlib.sha256(identity.encode()).hexdigest(), 16))
    return candidates[rng.randrange(len(candidates))].rollout_id


def segment_advantages(records, generated_mask):
    """以真实生成mask与阶段区间相交，调用verl原生组内标准化，返回优势和策略mask。

    generated_mask为[N,T]布尔值，包含原生模型生成位置；QA行的父历史即使原先也是
    模型生成token，也因不在本阶段区间而被排除。工具/图像/state始终不能计策略loss。
    奖励放到本阶段最后一个真实生成位置；全同奖励组优势为0，不伪造差异。
    """
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    groups = validate_groups(records)
    if generated_mask.ndim != 2 or generated_mask.shape[0] != len(records) or generated_mask.dtype != torch.bool:
        raise ValueError("generated_mask必须是与记录对应的[N,T]布尔张量")
    mask = torch.zeros_like(generated_mask)
    rewards = torch.zeros_like(generated_mask, dtype=torch.float32)
    for i, record in enumerate(records):
        if record.end > mask.shape[1]:
            raise ValueError("阶段边界越出响应序列")
        mask[i, record.start : record.end] = generated_mask[i, record.start : record.end]
        positions = mask[i].nonzero(as_tuple=False).flatten()
        if not len(positions):
            raise ValueError("阶段没有真实生成token，不能制造EOS或奖励位置")
        rewards[i, positions[-1]] = record.reward
    advantages, _ = compute_grpo_outcome_advantage(
        rewards,
        mask.float(),
        np.array([r.group_id for r in records]),
        epsilon=1e-6,
        norm_adv_by_std_in_grpo=True,
    )
    zero_groups = sum(len({records[i].reward for i in ids}) == 1 for ids in groups.values())
    return advantages, mask, dict(groups=len(groups), zero_variance_groups=zero_groups)


def segmented_policy_loss(
    records, old_logprobs, logprobs, generated_mask, *, actor_config, execution_weight=1.0, qa_weight=1.0
):
    """完整组参考loss：组内每响应按有效token平均，组平均，最后两阶段加权相加。

    clipping直接调用verl实现。这里只处理单进程完整组，不冒充已接FSDP/DDP；分布式
    adapter必须先在全局计算优势，并沿阶段/组计数归约，不能每卡重新归一化。
    未接KL/熵项，本函数返回策略项；old_logprobs显式detach，不修改任何输入张量。
    """
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    if actor_config.get("global_batch_info", {}):
        raise ValueError("当前完整组参考入口不接受分布式/外部归约计数")
    if any(not math.isfinite(w) or w < 0 for w in (execution_weight, qa_weight)):
        raise ValueError("阶段权重必须是有限非负数")
    if old_logprobs.shape != logprobs.shape or logprobs.shape != generated_mask.shape:
        raise ValueError("新旧log-prob与mask未对齐")
    if logprobs.device != old_logprobs.device or logprobs.device != generated_mask.device:
        raise ValueError("概率与mask设备不一致")
    if not bool(torch.isfinite(logprobs).all() and torch.isfinite(old_logprobs).all()):
        raise ValueError("log-prob含非有限值")
    advantages, mask, stats = segment_advantages(records, generated_mask)
    groups = validate_groups(records)
    if qa_weight:
        execution_groups = {r.group_id for r in records if r.phase == "execution"}
        parents = {r.parent_rollout_id for r in records if r.phase == "qa"}
        covered = {r.group_id for r in records if r.rollout_id in parents}
        if covered != execution_groups:
            raise ValueError("启用QA时每个动作组都须有对应QA组；缺分组不能静默丢弃")
    by_phase = defaultdict(list)
    phase_metrics = defaultdict(list)
    for indices in groups.values():
        value, metrics = compute_policy_loss_vanilla(
            old_logprobs[indices].detach(),
            logprobs[indices],
            advantages[indices],
            mask[indices].float(),
            loss_agg_mode="seq-mean-token-mean",
            config=actor_config,
        )
        phase = records[indices[0]].phase
        by_phase[phase].append(value)
        phase_metrics[phase].append(metrics)
    total = logprobs.sum() * 0.0
    for phase, weight in [("execution", execution_weight), ("qa", qa_weight)]:
        if phase not in by_phase:
            if weight:
                raise ValueError("缺少启用阶段的完整训练组")
            continue
        value = torch.stack(by_phase[phase]).mean()
        total = total + weight * value
        stats[phase + "_policy_loss"] = float(value.detach())
        for key in phase_metrics[phase][0]:
            stats[phase + "/" + key] = sum(m[key] for m in phase_metrics[phase]) / len(phase_metrics[phase])
    return total, stats
