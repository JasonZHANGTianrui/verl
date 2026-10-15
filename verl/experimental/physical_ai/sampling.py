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

"""原生G0.5缓存会话上的随机采样；保留完整词表、原始ID与行为策略概率。"""

import time
from copy import copy

import torch


def fork_chat(chat):
    """复制会话可变数据，不复制模型权重；分支按原始ID及全部图像/state独立续写。

    模型和只读模板共享，ids/types/图片字典/state张量独立。KV由调用者为每个分支
    重新prefill，不能共享含可变卷积/递归状态的CachedSession对象。
    """
    child = copy(chat)
    child.ids = list(chat.ids)
    child.types = list(chat.types)
    child.pixels = {key: value.clone() for key, value in chat.pixels.items()}
    child.states = [value.clone() for value in chat.states]
    return child


def sample_response(
    engine,
    inputs,
    *,
    max_new_tokens,
    stop_ids,
    temperature,
    generator,
    timeout_seconds=180,
    mode="cached",
    compute_dtype="bfloat16",
):
    """从现有缓存续采一条assistant消息，返回ID、逐token log-prob和停止原因。

    inputs是原生完整前缀，含全部历史模态附件；每步只追加一个真实token，原生engine
    自行消费未见后缀。分布为完整词表softmax(logits/temperature)，与actor算分一致。
    mode=cached使用原生KV；mode=recompute让VLM每次完整重算，仅缓存同一消息内
    不变的输入embedding，使用与actor相同的无KV前向。两种模式均保留全部历史。
    不做top-k/top-p或格式修复。末token也进入缓存；超容量/时间按实际截断记录。
    """
    if type(max_new_tokens) is not int or max_new_tokens < 1 or temperature <= 0:
        raise ValueError("生成预算和温度必须为正")
    if compute_dtype not in {"float32", "bfloat16"}:
        raise ValueError("未知计算精度")
    if mode not in {"cached", "recompute"}:
        raise ValueError("未知采样计算路径")
    if timeout_seconds <= 0:
        raise ValueError("生成超时必须为正秒数")
    from generate import generated_token_types

    current = dict(inputs)
    capacity = int(engine.model.cfg.vlm.max_position_embeddings) - inputs["input_ids"].shape[1]
    count = min(max_new_tokens, capacity)
    ids, logprobs = [], []
    reason = "context_capacity" if count < max_new_tokens else "token_budget"
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=compute_dtype == "bfloat16"):
        model = engine.model
        embeddings = None
        if mode == "recompute":
            embeddings = model._forward_embed(
                current["input_ids"], current["attention_mask"], current["pixel_values"], current["proprio"]
            )
        for _ in range(count):
            if mode == "cached":
                engine.append(current, return_logits=False)
                hidden = engine.hidden
            else:
                causal, positions = model.build_causal_mask_and_position_ids(
                    current["input_ids"],
                    current["attention_mask"],
                    dtype=next(iter(current["pixel_values"].values())).dtype,
                )
                hidden = model.vlm(
                    inputs_embeds=embeddings,
                    attention_mask=causal,
                    position_ids=positions,
                    kv_cache=None,
                    return_kv_cache=False,
                    attn_implementation=model.attn_implementation,
                    mixture_name="vlm",
                    split_idx=None,
                    padding_mask=(current["attention_mask"] != 0).to(embeddings.dtype),
                )[:, -1:]
            # 原生CE在FP32下投影输出头；先BF16投影再转float已丢失logit精度。
            with torch.autocast("cuda", enabled=False):
                logits = model.vlm.decode(hidden)[:, -1].float() / temperature
            token = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)
            ids.append(int(token.item()))
            logprobs.append(float(torch.log_softmax(logits, -1).gather(-1, token).item()))
            current = dict(
                current,
                input_ids=torch.cat((current["input_ids"], token), 1),
                attention_mask=torch.cat((current["attention_mask"], generated_token_types(engine.model, token)), 1),
            )
            if mode == "recompute":
                embeddings = torch.cat((embeddings, model.vlm.embed(token).to(embeddings.dtype)), dim=1)
            if ids[-1] in stop_ids:
                reason = "message_end"
                break
            if time.monotonic() - started > timeout_seconds:
                reason = "wall_time_limit"
                break
        if ids and mode == "cached":
            engine.append(current, return_logits=False)
    return dict(
        ids=ids,
        logprobs=logprobs,
        termination=reason,
        elapsed_seconds=time.monotonic() - started,
        sampling_mode=mode,
        compute_dtype=compute_dtype,
    )
