# Copyright 2026 Physical AI contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""共享一个原生G0.5模型，批量推进独立会话；保留完整KV、卷积和递推状态。"""

import time

import torch
import torch.nn.functional as F

CACHE_FIELDS = ("key_cache", "value_cache", "conv_states", "recurrent_states", "split_recurrent_states")


def pack_caches(caches):
    """沿batch维合并原生缓存，返回缓存和各行左侧补齐长度。

    仅完整注意力的KV按序列维左补零；卷积/递推状态直接按batch维连接，不让虚构
    padding步进入DeltaNet递推。调用方必须用attention mask屏蔽这些KV位置。
    输入缓存不修改；模型层布局、设备或状态类型不一致时拒绝拼接。
    """
    if not caches:
        raise ValueError("不能合并空缓存组")
    first = caches[0]
    lengths = [c.num_items() for c in caches]
    pads = [max(lengths) - n for n in lengths]
    merged = type(first)(last_linear_layer=first.last_linear_layer)
    for field in CACHE_FIELDS:
        mappings = [getattr(c, field) for c in caches]
        if any(set(m) != set(mappings[0]) for m in mappings):
            raise ValueError("缓存层布局不同")
        for layer in mappings[0]:
            values = [m[layer] for m in mappings]
            if all(v is None for v in values):
                getattr(merged, field)[layer] = None
                continue
            if any(v is None or v.shape[0] != 1 for v in values):
                raise ValueError("需要独立的B=1原生缓存")
            if field in {"key_cache", "value_cache"}:
                values = [F.pad(v, (0, 0, p, 0)) for v, p in zip(values, pads, strict=True)]
            getattr(merged, field)[layer] = torch.cat(values, dim=0)
    return merged, pads


def select_cache_rows(cache, rows):
    """取仍活跃的batch行并复制存储；已结束分支不能被后续卷积原地更新污染。"""
    selected = type(cache)(last_linear_layer=cache.last_linear_layer)
    for field in CACHE_FIELDS:
        for layer, value in getattr(cache, field).items():
            getattr(selected, field)[layer] = None if value is None else value[rows].clone()
    return selected


def unpack_cache(cache, row, left_pad):
    """还原单条真实会话的缓存，删除计算用左padding，保留所有真实历史及末个EOS。"""
    result = select_cache_rows(cache, [row])
    for field in ("key_cache", "value_cache"):
        for layer, value in getattr(result, field).items():
            getattr(result, field)[layer] = value[..., left_pad:, :].contiguous()
    return result


@torch.inference_mode()
def sample_responses(requests):
    """批量生成一组assistant消息，返回与请求顺序一致的原token、概率及停止原因。

    每条请求沿既有CachedSession.append编码真实新观察，随后用原生VLM的batch前向
    同时生成token。模型权重共享，会话/RNG/缓存各自独立。KV左padding只用于计算，
    不进入会话、位置编码、线性注意力状态或训练mask；结束行退出batch。末token也
    消费进缓存。当前只启用FP32全词表温度采样，训练前仍须核验行为概率一致性。
    """
    if not requests:
        return []
    model = requests[0]["engine"].model
    if model.training:
        raise ValueError("批量采样要求eval模型")
    starts = [time.monotonic() for _ in requests]
    capacities, endings = [], []
    for request in requests:
        if request["engine"].model is not model or request["mode"] != "cached" or request["compute_dtype"] != "float32":
            raise ValueError("批量采样必须共享同一FP32缓存模型")
        if request["temperature"] <= 0 or request["max_new_tokens"] < 1:
            raise ValueError("采样温度和生成预算必须为正")
        capacity = int(model.cfg.vlm.max_position_embeddings) - request["inputs"]["input_ids"].shape[1]
        if capacity < 1:
            raise ValueError("原生上下文已满")
        capacities.append(min(capacity, request["max_new_tokens"]))
        endings.append("context_capacity" if capacity < request["max_new_tokens"] else "token_budget")
        request["engine"].append(request["inputs"], return_logits=False)
    engines = [r["engine"] for r in requests]
    cache, pads = pack_caches([e.cache for e in engines])
    hidden = torch.cat([e.hidden for e in engines], dim=0)
    positions = torch.cat([e.last_position for e in engines], dim=0)
    outputs = [dict(ids=[], logprobs=[]) for _ in requests]
    active = list(range(len(requests)))
    while active:
        with torch.autocast(hidden.device.type, enabled=False):
            logits = model.vlm.decode(hidden)[:, -1].float()
        tokens = []
        for row, index in enumerate(active):
            request, result = requests[index], outputs[index]
            logp = torch.log_softmax(logits[row] / request["temperature"], -1)
            token = torch.multinomial(logp.exp(), 1, generator=request["generator"])
            result["ids"].append(int(token.item()))
            result["logprobs"].append(float(logp[token].item()))
            tokens.append(token)
        token_ids = torch.stack(tokens, dim=0)
        positions = positions + model.mask_helper._sample_step(positions.shape, positions.device)
        embeddings = model.vlm.embed(token_ids).to(hidden.dtype)
        size, length = len(active), cache.num_items() + 1
        mask = embeddings.new_zeros((size, 1, 1, length))
        for row, pad in enumerate(pads):
            mask[row, :, :, :pad] = torch.finfo(mask.dtype).min
        hidden, cache = model.vlm(
            inputs_embeds=embeddings,
            attention_mask=mask,
            position_ids=positions.unsqueeze(0).expand(3, -1, -1),
            kv_cache=cache,
            return_kv_cache=True,
            attn_implementation=model.attn_implementation,
            mixture_name="vlm",
            split_idx=None,
            padding_mask=embeddings.new_ones((size, 1)),
        )
        hidden = hidden[:, -1:, :]
        keep = []
        for row, index in enumerate(active):
            request, result = requests[index], outputs[index]
            elapsed = time.monotonic() - starts[index]
            if result["ids"][-1] in request["stop_ids"]:
                reason = "message_end"
            elif len(result["ids"]) >= capacities[index]:
                reason = endings[index]
            elif elapsed > request.get("timeout_seconds", 180):
                reason = "wall_time_limit"
            else:
                keep.append(row)
                continue
            from generate import generated_token_types

            engine = engines[index]
            ids = engine.ids.new_tensor([result["ids"]])
            engine.ids = torch.cat((engine.ids, ids), dim=1)
            engine.types = torch.cat((engine.types, generated_token_types(model, ids)), dim=1)
            engine.cache = unpack_cache(cache, row, pads[row])
            engine.hidden = hidden[row : row + 1].clone()
            engine.last_position = positions[row : row + 1].clone()
            if engine.cache.num_items() != engine.ids.shape[1]:
                raise ValueError("批量拆分后的缓存长度不等于真实token历史")
            result.update(
                termination=reason,
                elapsed_seconds=elapsed,
                sampling_mode="cached",
                compute_dtype="float32",
                generation_batch_size=len(requests),
            )
        if not keep:
            break
        if len(keep) != len(active):
            cache = select_cache_rows(cache, keep)
            hidden, positions = hidden[keep], positions[keep]
            pads = [pads[i] for i in keep]
            active = [active[i] for i in keep]
    return outputs


def run_phases(generators, batch_size, *, sample_batch=sample_responses):
    """协作式调度独立机器人回合，在模型调用边界汇成GPU batch，返回原顺序结果。

    模拟器创建、执行、分叉与关闭均留在owner线程；不把MuJoCo/EGL对象跨线程传递。
    只改变调度顺序，不修改物理步、模型上下文、采样seed或评分。任一回合异常时关闭
    所有生成器（释放视频句柄），整组拒绝更新，不能丢掉出错分支后缩组。
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("rollout_batch_size须为正整数")
    generators = list(generators)
    pending, results = [], [None] * len(generators)
    try:
        for i, generator in enumerate(generators):
            try:
                pending.append((i, next(generator)))
            except StopIteration as done:
                results[i] = done.value
        while pending:
            selected, pending = pending[:batch_size], pending[batch_size:]
            responses = sample_batch([request for _, request in selected])
            for (index, _), response in zip(selected, responses, strict=True):
                try:
                    pending.append((index, generators[index].send(response)))
                except StopIteration as done:
                    results[index] = done.value
        return results
    finally:
        for generator in generators:
            generator.close()
