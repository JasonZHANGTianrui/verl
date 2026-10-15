# Copyright 2026 Physical AI contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""SegGRPO两阶段的训练观测；只读真实采样/更新结果，不改变优势或采样策略。"""

from collections import defaultdict

import torch


def rollout_metrics(records, artifacts, prepared):
    """统计整个采样batch，而非只看有非零梯度的样本。

    返回SwanLab可直接接收的数值字典。长度单位为token；优势按响应取一次，避免长
    序列被多计。old_sampled_nll是采样token的负log-prob，不冒充完整词表精确熵。
    全同分组仍计入奖励分布及零方差比例；不把它们过滤掉使曲线显得更好。
    """
    metrics = {}
    for phase in ("execution", "qa"):
        ids = [i for i, r in enumerate(records) if r.phase == phase]
        prefix = f"rollout/{phase}"
        rewards = torch.tensor([records[i].reward for i in ids], dtype=torch.float32)
        groups = defaultdict(list)
        lengths, contexts, advantages, nll = [], [], [], []
        for i in ids:
            groups[records[i].group_id].append(records[i].reward)
            mask = prepared["active"][i]
            lengths.append(int(mask.sum()))
            contexts.append(len(artifacts[i]["generated"]))
            advantages.append(float(prepared["advantages"][i][mask][0]))
            nll.append(float(-prepared["old"][i][mask].mean()))
        mixed = sum(len(set(values)) > 1 for values in groups.values())
        metrics.update(
            {
                f"{prefix}/reward_mean": float(rewards.mean()),
                f"{prefix}/reward_std": float(rewards.std(unbiased=False)),
                f"{prefix}/zero_variance_group_fraction": 1 - mixed / len(groups),
                f"{prefix}/effective_groups": mixed,
                f"{prefix}/all_success_group_fraction": sum(all(v == 1 for v in g) for g in groups.values())
                / len(groups),
                f"{prefix}/all_failure_group_fraction": sum(all(v == 0 for v in g) for g in groups.values())
                / len(groups),
                f"{prefix}/phase_tokens_mean": sum(lengths) / len(lengths),
                f"{prefix}/phase_tokens_max": max(lengths),
                f"{prefix}/context_tokens_mean": sum(contexts) / len(contexts),
                f"{prefix}/context_tokens_max": max(contexts),
                f"{prefix}/old_sampled_nll": sum(nll) / len(nll),
                f"{prefix}/advantage_mean": sum(advantages) / len(advantages),
                f"{prefix}/advantage_abs_mean": sum(abs(a) for a in advantages) / len(advantages),
                f"{prefix}/advantage_min": min(advantages),
                f"{prefix}/advantage_max": max(advantages),
            }
        )
    return metrics


def update_metrics(result, records, indices):
    """汇总所有rank的mini-batch策略指标；缺测项不伪造为0。

    phase_policy_loss含零优势响应的零贡献，分母为该阶段完整响应数。clip/ratio/KL
    仅在实际反向的非零优势响应上可得，使用active前缀并按响应等权，和loss归约一致。
    old_policy_k3只对采样时旧策略作诊断，不是reference KL正则项。全零优势mini
    明确记录跳过，梯度范数缺测不画成一次真实的零范数更新。
    """
    rows = result.get("rows", [])
    metrics = {"train/optimizer_step_applied": int(result["optimizer_step"])}
    if result.get("grad_norm_before_clip") is not None:
        metrics["actor/grad_norm"] = result["grad_norm_before_clip"]
    metrics["actor/policy_loss"] = sum(r["loss"] for r in rows)
    for phase in ("execution", "qa"):
        phase_rows = [r for r in rows if r["phase"] == phase]
        count = sum(records[i].phase == phase for i in indices)
        prefix = f"actor/{phase}"
        metrics[f"{prefix}/policy_loss"] = sum(r["policy_loss"] for r in phase_rows) / count
        metrics[f"{prefix}/active_sequence_fraction"] = len(phase_rows) / count
        for key in (
            "actor/pg_clipfrac",
            "actor/pg_clipfrac_lower",
            "actor/ppo_kl",
            "old_policy_k3",
            "ratio_mean",
            "behavior_logprob_delta_mean",
        ):
            if phase_rows:
                metrics[f"{prefix}/active_{key.removeprefix('actor/')}"] = sum(r[key] for r in phase_rows) / len(
                    phase_rows
                )
        if phase_rows:
            metrics[f"{prefix}/active_ratio_min"] = min(r["ratio_min"] for r in phase_rows)
            metrics[f"{prefix}/active_ratio_max"] = max(r["ratio_max"] for r in phase_rows)
    return metrics
