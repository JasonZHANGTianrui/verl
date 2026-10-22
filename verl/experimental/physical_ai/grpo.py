"""普通整轨迹GRPO：一次组内归一化，动作与QA共享同一个回报和优势。"""

from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch


def combined_reward(task, qa=None, *, qa_status):
    """合并两个二元奖励；只有明确未交付/未知GT才把task权重升为1，错误不兜底。"""
    if task not in (0, 1):
        raise ValueError("task必须是已确定的0/1")
    if qa_status in {"no_question", "undetermined", "incomplete_execution", "context_capacity"}:
        if qa is not None:
            raise ValueError("未交付QA不得含分数")
        return float(task)
    if qa_status != "scored" or qa not in (0, 1):
        raise ValueError("QA缺有效裁判结果，不能转为task-only")
    return 0.5 * task + 0.5 * qa


def qa_gate(question, execution, binding):
    """仅在动作结束边界决定是否交付同一道QA；不读取答案或按成功筛轨迹。"""
    if execution.get("incomplete_message"):
        return "incomplete_execution"
    if question is None:
        return "no_question"
    status = binding["gt"]["status"]
    if status == "undetermined":
        return status
    if status != "ok":
        raise ValueError("QA查询/数据错误：" + str(status))
    return "deliver"


def task_groups(records):
    """核验n条独立完整经历同初态/任务/策略，不缩组，返回每个任务的行号。"""
    if not records or len({r.rollout_id for r in records}) != len(records):
        raise ValueError("空批或重复轨迹")
    if len({r.policy_version for r in records}) != 1:
        raise ValueError("混合行为策略")
    groups = defaultdict(list)
    for i, r in enumerate(records):
        if (
            r.phase != "trajectory"
            or r.parent_rollout_id is not None
            or r.reward_status != "scored"
            or r.reward not in (0, 0.5, 1)
            or type(r.group_size) is not int
            or r.group_size < 2
            or type(r.start) is not int
            or type(r.end) is not int
            or not 0 <= r.start < r.end
            or any(
                not isinstance(getattr(r, k), str) or not getattr(r, k)
                for k in ("rollout_id", "group_id", "condition_id", "case_id", "policy_version")
            )
        ):
            raise ValueError("普通GRPO轨迹契约不合法")
        groups[r.group_id].append(i)
    for ids in groups.values():
        first = records[ids[0]]
        if (
            len(ids) != first.group_size
            or len({(records[i].condition_id, records[i].case_id, records[i].group_size) for i in ids}) != 1
        ):
            raise ValueError("缺失轨迹或组条件不一致")
    if len({r.group_size for r in records}) != 1:
        raise ValueError("组大小不一致")
    return list(groups.values())


def namespace_group(records, prefix):
    """只加唯一任务前缀，保留原始条件和模型经历，不绑定另一个父轨迹。"""
    task_groups(records)
    return [replace(r, rollout_id=prefix + "/" + r.rollout_id, group_id=prefix + "/" + r.group_id) for r in records]


def minibatch_indices(records, size):
    """完整n轨迹组切mini-batch，优势仍来自更新前全批，单位为完整会话条数。"""
    groups = task_groups(records)
    width = len(groups[0])
    if type(size) is not int or size < width or size % width or len(records) % size:
        raise ValueError("mini-batch必须含完整组并整除batch")
    ids = [i for group in groups for i in group]
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def prepare_batch(records, artifacts):
    """保留整会话的原token概率，只对真正生成位置算GRPO；用户/工具/图像均mask。"""
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    groups = task_groups(records)
    if len(records) != len(artifacts):
        raise ValueError("记录与附件不匹配")
    length = max(len(a["generated"]) - 1 for a in artifacts)
    active = torch.zeros((len(records), length), dtype=torch.bool)
    old = torch.zeros_like(active, dtype=torch.float32)
    rewards = torch.zeros_like(old)
    for i, (r, a) in enumerate(zip(records, artifacts, strict=True)):
        n = len(a["generated"]) - 1
        if len(a["old_logprobs"]) != n + 1 or r.end > n:
            raise ValueError("概率/生成mask/区间不同步")
        old[i, :n] = torch.tensor(a["old_logprobs"][1:])
        active[i, r.start : r.end] = torch.tensor(a["generated"][1:])[r.start : r.end]
        positions = active[i].nonzero().flatten()
        if not len(positions):
            raise ValueError("轨迹没有生成token，不伪造EOS")
        rewards[i, positions[-1]] = r.reward
    if not torch.isfinite(old).all():
        raise ValueError("非有限行为概率")
    advantage, _ = compute_grpo_outcome_advantage(
        rewards, active.float(), np.array([r.group_id for r in records]), epsilon=1e-6, norm_adv_by_std_in_grpo=True
    )
    return dict(
        old=old,
        advantages=advantage,
        active=active,
        sampled=active.clone(),
        stats=dict(
            groups=len(groups), zero_variance_groups=sum(len({records[i].reward for i in g}) == 1 for g in groups)
        ),
    )


def validate_config(config):
    """独立校验普通GRPO配方；拒绝夹带父选择、阶段loss或SegGRPO未知QA语义。"""
    from .training_state import validate_fixed_algorithm

    if (
        config.get("advantage_estimator") != "grpo"
        or type(config.get("group_size")) is not int
        or config["group_size"] < 2
    ):
        raise ValueError("需要grpo与整数group_size>=2")
    if any(
        k in config
        for k in (
            "execution_group_size",
            "qa_group_size",
            "parent_selection",
            "unknown_qa_policy",
            "execution_loss_weight",
            "qa_loss_weight",
        )
    ):
        raise ValueError("普通GRPO配置不能混入分段算法字段")
    # 复用已有优化器/采样固定选项校验；转换仅发生在局部副本，不写回或伪装算法身份。
    shared = dict(config, advantage_estimator="seggrpo")
    validate_fixed_algorithm(shared)
    return dict(advantage_estimator="grpo", reward="task_qa_half_or_task_only", group_size=config["group_size"])


def rollout_metrics(records, artifacts, prepared):
    """整体回报、task/QA成绩和缺QA比例分开报；缺QA不进入QA准确率分母。"""
    groups = task_groups(records)
    metrics = {}
    for key in ("task_reward", "qa_reward"):
        values = [a[key] for a in artifacts if a.get(key) is not None]
        metrics["rollout/" + key + "/count"] = len(values)
        if values:
            metrics["rollout/" + key + "/mean"] = sum(values) / len(values)
    lengths = prepared["active"].sum(1)
    metrics.update(
        {
            "rollout/trajectory/reward_mean": sum(r.reward for r in records) / len(records),
            "rollout/trajectory/sampled_sequences": len(records),
            "rollout/trajectory/phase_tokens_mean": float(lengths.float().mean()),
            "rollout/trajectory/phase_tokens_max": int(lengths.max()),
            "rollout/trajectory/task_only_fraction": sum(a["qa_status"] != "scored" for a in artifacts) / len(records),
            "rollout/trajectory/zero_variance_group_fraction": sum(
                len({records[i].reward for i in g}) == 1 for g in groups
            )
            / len(groups),
        }
    )
    for status in sorted({a["qa_status"] for a in artifacts}):
        metrics["rollout/qa_status/" + status] = sum(a["qa_status"] == status for a in artifacts)
    return metrics
