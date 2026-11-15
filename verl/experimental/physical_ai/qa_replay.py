"""仅QA的离线经历GRPO；复用固定SFT前缀与采样器，不运行或监督动作阶段。"""

from collections import defaultdict
from dataclasses import replace
import json

import numpy as np
import torch

from .seggrpo import UNSCORED_QA


def validate_config(config):
    """检查独立QA配方；继承通用优化选项，不允许夹带动作组或父轨迹采样。"""
    from .training_state import validate_fixed_algorithm

    if config.get("advantage_estimator") != "qa_replay_grpo" or config.get("qa_history_mode") != "fixed_sft":
        raise ValueError("需要qa_replay_grpo与固定SFT经历")
    if type(config.get("qa_group_size")) is not int or config["qa_group_size"] < 2:
        raise ValueError("QA完整组至少两条")
    if any(k in config for k in ("execution_group_size", "parent_selection", "execution_loss_weight")):
        raise ValueError("仅QA训练不能配置动作组或动作loss")
    if config.get("unknown_qa_policy") != "skip_qa_group":
        raise ValueError("未知GT整组不计分、不更新")
    shared = dict(config, advantage_estimator="seggrpo", unknown_qa_policy="retain_action_skip_qa_group")
    validate_fixed_algorithm(shared)
    return dict(advantage_estimator="qa_replay_grpo", reward="qa_only", qa_group_size=config["qa_group_size"])


def collect_group(policy, processor, config, output, *, policy_version, task, resources=None):
    """同一冻结前缀采样n条QA，返回真实QA记录；resources不使用，也不创建模拟器。

    历史动作和图片保留为模型条件。新工具输出沿Replay约定只记录、不执行；全部新
    回答token保留原采样概率。未知GT沿公共评分接口排除，但不缩减采样组。
    """
    from .batched_sampling import run_phases
    from .sft_replay import collect_qa

    if config["sampling_mode"] != "cached":
        raise ValueError("仅QA Replay使用已验收cached生成器")
    shared = dict(config, unknown_qa_policy="retain_action_skip_qa_group")
    records, artifacts = collect_qa(
        policy, processor, shared, output, policy_version=policy_version, task=task,
        drive=lambda generators: run_phases(generators, config.get("rollout_batch_size", 1)),
    )
    # 旧Replay还配对在线动作；本实验取消该配对，不伪造空动作组。
    records = [replace(r, paired_execution_group_id=None) for r in records]
    task_groups(records)
    return records, artifacts


def task_groups(records):
    """核验完整QA组、固定经历身份和奖励；不借用需要在线父动作的SegGRPO校验。"""
    if not records or len({r.rollout_id for r in records}) != len(records):
        raise ValueError("空批或重复回答")
    if len({r.policy_version for r in records}) != 1:
        raise ValueError("混合行为策略")
    groups = defaultdict(list)
    for i, r in enumerate(records):
        h = r.history_parent or {}
        if (r.phase != "qa" or r.paired_execution_group_id is not None
                or h.get("kind") != "fixed_sft" or h.get("case_id") != r.case_id
                or h.get("rollout_id") != r.parent_rollout_id or not r.parent_rollout_id
                or type(r.group_size) is not int or r.group_size < 2
                or type(r.start) is not int or type(r.end) is not int or not 0 <= r.start < r.end):
            raise ValueError("QA经历或区间绑定错误")
        for key in ("manifest_sha256", "prefix_sha256", "source_record_sha256"):
            value = h.get(key, "")
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("固定经历摘要无效")
        if not ((r.reward_status == "scored" and r.reward in (0, 1))
                or (r.reward_status == UNSCORED_QA and r.reward is None)):
            raise ValueError("评分错误不能填零")
        groups[r.group_id].append(i)
    for ids in groups.values():
        first = records[ids[0]]
        identities = {(r.condition_id, r.case_id, r.group_size, r.reward_status,
                       r.parent_rollout_id, json.dumps(r.history_parent, sort_keys=True))
                      for r in (records[i] for i in ids)}
        if len(ids) != first.group_size or len(identities) != 1:
            raise ValueError("组不完整或混入不同经历/GT")
    if len({r.group_size for r in records}) != 1:
        raise ValueError("组大小不同")
    return list(groups.values())


def namespace_group(records, prefix):
    """只命名当前回答与组；冻结父经历身份保持原值。"""
    task_groups(records)
    return [replace(r, rollout_id=prefix + "/" + r.rollout_id, group_id=prefix + "/" + r.group_id) for r in records]


def minibatch_indices(records, size):
    """按完整QA组切mini-batch，不能在mini内重新计算组内优势。"""
    groups = task_groups(records)
    width = len(groups[0])
    if type(size) is not int or size < width or size % width or len(records) % size:
        raise ValueError("mini-batch须由完整QA组组成并整除batch")
    ids = [i for group in groups for i in group]
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def prepare_batch(records, artifacts):
    """QA回答做组内GRPO，历史token一律mask；unknown组保留概率核验但没有梯度。

    列j对应原序列第j+1个token。GT只作用于末个生成token的回报；调用verl公共
    outcome advantage实现，避免复制或修改组内标准化公式。
    """
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    groups = task_groups(records)
    if len(records) != len(artifacts):
        raise ValueError("记录/附件数量不一致")
    length = max(len(a["generated"]) - 1 for a in artifacts)
    sampled = torch.zeros((len(records), length), dtype=torch.bool)
    old = torch.zeros_like(sampled, dtype=torch.float32)
    rewards = torch.zeros_like(old)
    active = sampled.clone()
    for i, (r, a) in enumerate(zip(records, artifacts, strict=True)):
        n = len(a["generated"]) - 1
        if len(a["old_logprobs"]) != n + 1 or r.end > n or r.start != a["prefix_length"] - 1:
            raise ValueError("QA边界/概率不同步")
        mask = torch.tensor(a["generated"][1:], dtype=torch.bool)
        if mask[:r.start].any() or mask[r.end:].any() or not mask[r.start:r.end].any():
            raise ValueError("QA监督泄漏到历史或没有新生成token")
        sampled[i, :n] = mask
        old[i, :n] = torch.tensor(a["old_logprobs"][1:])
        if r.reward_status == "scored":
            active[i] = sampled[i]
            rewards[i, active[i].nonzero().flatten()[-1]] = r.reward
    if not torch.isfinite(old).all():
        raise ValueError("非有限行为概率")
    advantages = torch.zeros_like(old)
    ids = [i for i, r in enumerate(records) if r.reward_status == "scored"]
    if ids:
        advantages[ids], _ = compute_grpo_outcome_advantage(
            rewards[ids], active[ids].float(), np.array([records[i].group_id for i in ids]),
            epsilon=1e-6, norm_adv_by_std_in_grpo=True,
        )
    return dict(old=old, advantages=advantages, active=active, sampled=sampled,
                stats=dict(groups=len(groups), zero_variance_groups=sum(
                    records[g[0]].reward_status == "scored" and len({records[i].reward for i in g}) == 1 for g in groups)))
