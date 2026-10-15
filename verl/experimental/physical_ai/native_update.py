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

"""原生G0.5＋verl策略loss的单组更新诊断；不冒充完整分布式Trainer。"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from verl.experimental.physical_ai.online_reward import save_json

from .g05_logprobs import next_token_logprobs
from .seggrpo import segment_advantages, validate_groups


def update_once(
    model,
    records,
    artifacts,
    output,
    *,
    learning_rate=1e-4,
    max_logprob_error=0.05,
    compute_dtype="bfloat16",
    distributed=False,
    optimizer=None,
    max_grad_norm=None,
):
    """收齐组后逐条重算、累积两阶段GRPO梯度，执行一次无动量SGD并记录真实变化。

    先在完整组计算优势，每条loss按其阶段样本数缩放；不在micro-batch内重新归一化。
    使用SGD仅为32GB单卡的无优化器状态诊断；不定义正式训练优化器，不改原checkpoint。
    distributed=True时按全局行号分片，优势仍在完整组计算；更新前SUM所有卡梯度，
    不重复除以卡数。所有生成mask来自rollout，父轨迹在QA行仅作前缀；zero-advantage整组明确跳过更新。
    连续训练可传持久optimizer，更新公式保持不变；max_grad_norm在全局归约后统一裁剪。
    不传optimizer仍沿用旧单步SGD诊断，不能将每轮重新创建的有状态优化器冒称恢复。
    """
    from generate import no_cache_hidden
    from memory import checkpoint_linear_layers
    from runtime import to_device

    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    rank, world_size = 0, 1
    if distributed:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("分布式更新要求已初始化进程组")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    groups = validate_groups(records)
    if len(groups) != 2 or {r.phase for r in records} != {"execution", "qa"}:
        raise ValueError("单步诊断要求一个动作组和对应一个QA组")
    if len(artifacts) != len(records):
        raise ValueError("输入轨迹与组记录数量不符")
    length = max(len(a["generated"]) - 1 for a in artifacts)
    masks = torch.zeros((len(records), length), dtype=torch.bool)
    old = torch.zeros((len(records), length))
    for i, item in enumerate(artifacts):
        masks[i, : len(item["generated"]) - 1] = torch.tensor(item["generated"][1:])
        old[i, : len(item["old_logprobs"]) - 1] = torch.tensor(item["old_logprobs"][1:])
    advantages, active, stats = segment_advantages(records, masks)
    if not bool(advantages.any()):
        result = dict(status="skipped_zero_advantage", optimizer_step=False, **stats)
        save_json(Path(output) / "update.json", result)
        return result
    phase_sizes = {phase: sum(r.phase == phase for r in records) for phase in ["execution", "qa"]}
    config = OmegaConf.create(
        dict(clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None, clip_ratio_c=3.0, global_batch_info={})
    )
    model.train()
    model.vlm.gradient_checkpointing_enable()
    model.vision_tower.gradient_checkpointing_enable()
    diagnostic = optimizer is None
    if optimizer is None:
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    with checkpoint_linear_layers(model):
        for i, (record, artifact) in enumerate(zip(records, artifacts, strict=False)):
            if i % world_size != rank or not bool(advantages[i].any()):
                continue
            inputs = to_device(
                torch.load(artifact["inputs_path"], map_location="cpu", weights_only=True), torch.device("cuda")
            )
            n = inputs["input_ids"].shape[1] - 1
            mask = active[i : i + 1, :n].cuda()
            ids_mask = torch.cat((torch.zeros((1, 1), dtype=torch.bool, device="cuda"), mask), 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=compute_dtype == "bfloat16"):
                hidden = no_cache_hidden(model, inputs)
                logprobs = next_token_logprobs(
                    hidden, inputs["input_ids"], ids_mask, model.vlm.decode, checkpoint_projection=True
                )
                value, metrics = compute_policy_loss_vanilla(
                    old[i : i + 1, :n].cuda(),
                    logprobs,
                    advantages[i : i + 1, :n].cuda(),
                    mask.float(),
                    loss_agg_mode="seq-mean-token-mean",
                    config=config,
                )
                loss = value / phase_sizes[record.phase]
            error = (logprobs.detach() - old[i : i + 1, :n].cuda())[mask].abs()
            metrics["rollout_actor_max_logprob_error"] = float(error.max())
            metrics["rollout_actor_mean_logprob_error"] = float(error.mean())
            if float(error.max()) > max_logprob_error:
                raise ValueError("同版rollout与actor概率偏差超出诊断阈值，禁止更新")
            loss.backward()
            rows.append(dict(rollout_id=record.rollout_id, phase=record.phase, loss=float(loss.detach()), **metrics))
            save_json(Path(output) / "update-progress.json", dict(finished=len(rows), rows=rows))
            del inputs, hidden, logprobs, loss, value
    if distributed:
        from .distributed import sum_gradients

        sum_gradients(model)
        all_rows = [None] * world_size
        dist.all_gather_object(all_rows, rows)
        rows = sorted((row for shard in all_rows for row in shard), key=lambda row: row["rollout_id"])
    norms = []
    representative = None
    largest_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not bool(torch.isfinite(param.grad).all()):
                raise FloatingPointError("非有限梯度，拒绝optimizer.step")
            norm = float(param.grad.float().norm())
            norms.append(norm)
            if norm > largest_norm:
                largest_norm = norm
                representative = (name, param)
    if representative is None:
        raise RuntimeError("非零优势但无有效参数梯度")
    name, param = representative
    index = int(param.grad.abs().flatten().argmax())
    before = float(param.detach().flatten()[index])
    grad_norm = None
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError("梯度裁剪上限必须为正")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True))
    optimizer.step()
    after = float(param.detach().flatten()[index])
    result = dict(
        status="updated",
        optimizer_step=True,
        optimizer="SGD diagnostic, no momentum" if diagnostic else type(optimizer).__name__,
        distributed_world_size=world_size,
        rank=rank,
        gradient_reduction="sum_global_normalized_losses",
        learning_rate=learning_rate,
        grad_norm_before_clip=grad_norm,
        parameters_with_grad=len(norms),
        nonzero_gradient_parameters=sum(n > 0 for n in norms),
        representative=dict(parameter=name, index=index, before=before, after=after),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        rows=rows,
        reference_kl=False,
        **stats,
    )
    save_json(Path(output) / "update.json", result)
    model.eval()
    return result
