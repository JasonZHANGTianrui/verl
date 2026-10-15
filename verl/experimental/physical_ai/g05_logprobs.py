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

"""从原生G0.5隐藏状态计算逐token策略概率；不重编码动作，不替换模型forward。"""

import math

import torch
import torch.nn.functional as F


def next_token_logprobs(
    hidden, input_ids, generated_mask, decode, *, temperature=1.0, chunk_size=64, checkpoint_projection=False
):
    """返回[B,L-1]的next-token log-prob，位置j对应input_ids[:,j+1]。

    hidden为原生无KV前向产生的[B,L,D]；decode使用同一模型的完整词表输出头。
    generated_mask为显式布尔[B,L]，标记真正需要策略概率的位置，不是G0.5的模态
    类型attention_mask，也不能直接沿用SFT失败动作的loss mask。首token没有前文
    预测位置，禁止标true。工具观察、图片、state及padding应标false。

    按选中位置分块投影，避免一次产生整段[L,V]张量；保持autograd，未选位置填0。
    checkpoint_projection开启时，反向重算每块投影/softmax，避免为全部块保存完整
    词表中间量；不截断会话、不冻结模型，也不改变概率分布。
    当前仅支持temperature缩放后的完整词表分布，不做top-k/top-p或重复惩罚；rollout
    采样须使用同一分布。非正温度、形状错误及非法目标ID报错，不返回伪造概率。
    """
    if hidden.ndim != 3 or input_ids.ndim != 2 or hidden.shape[:2] != input_ids.shape:
        raise ValueError("hidden和input_ids需要一致的[B,L]序列维")
    if input_ids.dtype != torch.long or generated_mask.dtype != torch.bool:
        raise ValueError("input_ids必须long，generated_mask必须bool，不能传模态类型mask")
    if generated_mask.shape != input_ids.shape or input_ids.shape[1] < 2:
        raise ValueError("mask形状不符或序列不足两个token")
    if generated_mask[:, 0].any():
        raise ValueError("首token没有前文预测位置")
    if not (hidden.device == input_ids.device == generated_mask.device):
        raise ValueError("隐藏状态、ID和mask必须在同一设备")
    if not math.isfinite(temperature) or temperature <= 0 or type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("温度和分块大小必须为正")
    mask = generated_mask[:, 1:]
    positions = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    flat_hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    targets = input_ids[:, 1:].reshape(-1)
    # 全false允许返回可求导的零张量，由外层决定此条是否属于有效训练样本。
    output = flat_hidden[:, 0].float() * 0.0

    def score_chunk(states, target_ids):
        """目标ID作为显式参数传入，防止反向重算时捕获循环最后一块的索引。"""
        with torch.autocast(hidden.device.type, enabled=False):
            logits = decode(states).float() / temperature
            return -F.cross_entropy(logits, target_ids, reduction="none")

    values = []
    for selected in positions.split(chunk_size):
        if selected.numel() == 0:
            continue
        if checkpoint_projection and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            value = checkpoint(score_chunk, flat_hidden[selected], targets[selected], use_reentrant=False)
        else:
            value = score_chunk(flat_hidden[selected], targets[selected])
        values.append(value)
    if values:
        output = output.index_copy(0, positions, torch.cat(values))
    return output.reshape(mask.shape)
