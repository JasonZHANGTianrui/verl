"""窗口会话的采样和更新绑定：保留原经历，只投影模型当时能看到的图片。

动作、文字、state与行为概率始终使用原完整序列的下标。窗口改变时清空全部缓存；
更新按实际生成时的窗口分段，每个生成token只属于一段，仍按整条轨迹归一化。
"""

VERSION = "physical-ai.image-window-context.v1"


def warmup_sampling(policy, config, output):
    """正式采样计时前预热独立训练缓存，不建环境、不反向、不改变采样随机流。

    预热只生成两个token，覆盖初态和窗口内最长视觉前缀的8路缓存合并；不执行工具。
    使用训练来源fixture，绝不拿测评初态或测评动作预热。正式每次120秒限制保持。
    """
    from pathlib import Path

    import torch
    from runtime import to_device

    from .batched_sampling import sample_responses
    from .native_runtime import cached_session
    from .online_reward import save_json

    fixture = torch.load(config["window_warmup_inputs"], map_location="cpu", weights_only=True)
    if fixture["max_image_turn"] != config["max_image_turn"] or fixture["split"] != "train":
        raise ValueError("窗口预热资源身份不符")
    policy.model.eval()
    records = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for prefix in fixture["prefixes"]:
            actual = to_device(prefix, torch.device("cuda"))
            requests = [
                dict(
                    engine=cached_session(policy.model),
                    inputs=actual,
                    max_new_tokens=2,
                    stop_ids=[],
                    temperature=1.0,
                    generator=torch.Generator(device="cuda").manual_seed(1000 + i),
                    mode="cached",
                    compute_dtype="float32",
                    timeout_seconds=600,
                )
                for i in range(config["rollout_batch_size"])
            ]
            responses = sample_responses(requests)
            if any(len(x["ids"]) != 2 or x["termination"] != "token_budget" for x in responses):
                raise RuntimeError("独立预热未完成，不开始正式采样计时")
            records.append(
                dict(
                    tokens=actual["input_ids"].shape[1],
                    images=len(actual["pixel_values"]),
                    elapsed=max(x["elapsed_seconds"] for x in responses),
                )
            )
            del requests, responses, actual
    save_json(
        Path(output),
        dict(
            passed=True,
            split="train",
            sampling_rng_unchanged=True,
            optimizer_updates=0,
            environment_steps=0,
            records=records,
        ),
    )


class SamplingWindow:
    """管理一条会话的图片窗口，不共享环境、缓存或随机流。"""

    def __init__(self, limit, tokenizer):
        """limit沿SFT定义：满偶数轮双图后删最早一半，初态也算一轮。"""
        from image_window import omitted_turns

        omitted_turns(0, limit)
        self.limit, self.tokenizer, self.rebuilds = limit, tokenizer, 0

    def project(self, inputs, engine, factory):
        """调用SFT共享投影；历史发生编辑时换掉完整原生缓存，不仅裁剪KV。"""
        import torch
        from image_window import half_window_inputs

        actual, audit = half_window_inputs(inputs, self.tokenizer, max_image_turn=self.limit)
        old = engine.ids
        changed = old is not None and (
            actual["input_ids"].shape[1] < old.shape[1]
            or not torch.equal(actual["input_ids"][:, : old.shape[1]], old)
            or not torch.equal(actual["attention_mask"][:, : old.shape[1]], engine.types)
            or list(actual["pixel_values"])[: len(engine.image_keys)] != engine.image_keys
        )
        if changed:
            engine = factory(engine.model)
            self.rebuilds += 1
        return actual, engine, dict(audit, cache_rebuilt=changed, cache_rebuilds=self.rebuilds)

    def certificate(self, inputs, calls):
        """记录原序列视觉区间和每次生成的窗口，合并窗口未变化的相邻调用。

        合并只减少更新重算次数，不合并环境交互；end停在最后一个真实生成token，
        不让生成之后的新图成为更早动作的条件。中间工具反馈保持原顺序且不算监督。
        """
        from image_window import image_layout

        layout = image_layout(inputs, self.tokenizer)
        segments = []
        for call in calls:
            start = call["original_input_tokens"]
            end = start + len(call["ids"])
            removed = call["visual_history"]["omitted_turns"]
            if not end > start:
                raise ValueError("窗口经历包含空生成，不能伪造监督token")
            if segments and segments[-1]["removed_turns"] == removed:
                if start < segments[-1]["end"]:
                    raise ValueError("生成区间重叠")
                segments[-1]["end"] = end
            else:
                segments.append(dict(start=start, end=end, removed_turns=removed))
        return dict(
            version=VERSION, max_image_turn=self.limit, spans=layout.spans, image_keys=layout.keys, segments=segments
        )


def context_views(inputs, artifact, target_mask):
    """给每段返回实际输入、原next-token下标及该段监督mask，严格检查覆盖一次。

    target_mask的第j位预测原输入第j+1个token。删除图片后位置必须重新映射，
    但old_logprob和优势不能重算或移动到别的动作上。函数结束时核对完整目标覆盖。
    """
    import torch
    from image_window import ImageLayout, omitted_turns, project_prefix

    contract = artifact["context_window"]
    if contract["version"] != VERSION:
        raise ValueError("未知窗口经历版本")
    n = inputs["input_ids"].shape[1]
    if target_mask.shape != (1, n - 1):
        raise ValueError("窗口监督mask长度不符")
    layout = ImageLayout(tuple(tuple(x) for x in contract["spans"]), tuple(contract["image_keys"]))
    if list(layout.keys) != list(inputs["pixel_values"]):
        raise ValueError("窗口回执与实际图像附件不一致")
    covered = torch.zeros_like(target_mask)
    last_end = 0
    for segment in contract["segments"]:
        start, end, removed = (segment[k] for k in ("start", "end", "removed_turns"))
        if not 0 < start < end <= n or start < last_end:
            raise ValueError("窗口监督区间越界或重叠")
        last_end = end
        for boundary in (start, end):
            turns = sum(hi <= boundary for _, hi in layout.spans) // 2
            if omitted_turns(turns, contract["max_image_turn"]) != removed:
                raise ValueError("该更新段跨越了不同图片窗口")
        actual, origin = project_prefix(inputs, None, layout, end=end, removed_turns=removed)
        indices = origin[1:] - 1
        mask = target_mask[:, indices] & (origin[1:] >= start).unsqueeze(0)
        if bool(covered[:, indices][mask].any()):
            raise ValueError("同一个生成token被窗口更新重复监督")
        covered[:, indices] |= mask
        if bool(mask.any()):
            yield actual, indices, mask
    if not torch.equal(covered, target_mask):
        raise ValueError("窗口更新遗漏了原生成token")


def validate_window_behavior(model, inputs, artifact, old, mask, max_error):
    """按采样时每一段的真实可见上下文重算概率，零优势轨迹也不能绕过。"""
    import torch
    from generate import no_cache_hidden

    from .g05_logprobs import next_token_logprobs

    maximum, total, count, views = 0.0, 0.0, 0, 0
    for actual, indices, active in context_views(inputs, artifact, mask):
        ids_mask = torch.cat((torch.zeros_like(active[:, :1]), active), 1)
        hidden = no_cache_hidden(model, actual)
        logp = next_token_logprobs(hidden, actual["input_ids"], ids_mask, model.vlm.decode)
        error = (logp - old[:, indices])[active].abs()
        if not bool(torch.isfinite(error).all()) or float(error.max()) > max_error:
            raise ValueError("窗口采样与更新概率不一致，禁止更新")
        maximum = max(maximum, float(error.max()))
        total += float(error.sum())
        count += error.numel()
        views += 1
        del actual, hidden, logp, error
    if not count:
        raise ValueError("窗口经历没有待核对的生成token")
    return dict(max_error=maximum, mean_error=total / count, context_views=views, checked_tokens=count)


def backward_window_trajectory(
    model, inputs, artifact, old, advantages, mask, *, phase_size, config, compute_dtype, enforce_on_policy, max_error
):
    """逐窗口反向并立刻释放图，保持整轨迹token均值与原阶段分母。

    不能平均各段loss：段长不同会改变GRPO目标。每段乘自己的有效token数/整条有效
    token数后再除阶段样本数；梯度只累积，optimizer.step仍由原mini-batch统一执行。
    """
    import torch
    from generate import no_cache_hidden

    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    from .g05_logprobs import next_token_logprobs

    total = int(mask.sum())
    if not total:
        raise ValueError("窗口反向没有监督token")
    combined = {}
    loss_total = policy_total = error_sum = 0.0
    maximum = 0.0
    ratio_min = float("inf")
    ratio_max = 0.0
    views = 0
    for actual, indices, active in context_views(inputs, artifact, mask):
        ids_mask = torch.cat((torch.zeros_like(active[:, :1]), active), 1)
        count = int(active.sum())
        weight = count / total
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=compute_dtype == "bfloat16"):
            hidden = no_cache_hidden(model, actual)
            logp = next_token_logprobs(
                hidden, actual["input_ids"], ids_mask, model.vlm.decode, checkpoint_projection=True
            )
            value, metrics = compute_policy_loss_vanilla(
                old[:, indices],
                logp,
                advantages[:, indices],
                active.float(),
                loss_agg_mode="seq-mean-token-mean",
                config=config,
            )
            loss = value * weight / phase_size
        delta = (logp.detach() - old[:, indices])[active]
        error = delta.abs()
        peak = float(error.max())
        if not bool(torch.isfinite(error).all()) or (enforce_on_policy and peak > max_error):
            raise ValueError("窗口更新行为概率异常")
        ratio = delta.clamp(-20, 20).exp()
        metrics.update(ratio_mean=float(ratio.mean()), old_policy_k3=float((ratio - 1 - delta.clamp(-20, 20)).mean()))
        for key, number in metrics.items():
            combined[key] = combined.get(key, 0.0) + float(number) * weight
        maximum = max(maximum, peak)
        error_sum += float(error.sum())
        ratio_min = min(ratio_min, float(ratio.min()))
        ratio_max = max(ratio_max, float(ratio.max()))
        loss_total += float(loss.detach())
        policy_total += float(value.detach()) * weight
        views += 1
        loss.backward()
        del actual, hidden, logp, loss, value, delta, error, ratio
    combined.update(
        loss=loss_total,
        policy_loss=policy_total,
        phase_tokens=total,
        behavior_logprob_delta_max=maximum,
        behavior_logprob_delta_mean=error_sum / total,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        context_views=views,
    )
    if enforce_on_policy:
        combined.update(rollout_actor_max_logprob_error=maximum, rollout_actor_mean_logprob_error=error_sum / total)
    return combined


def join_phase_windows(execution, qa):
    """合并同一普通GRPO会话两阶段的窗口证书，不拼接两条不同经历。

    QA保存的原完整序列包含动作前缀；其视觉布局必须逐项延续动作布局。原token
    下标不重编号，两阶段生成区间必须不重叠。返回独立字典，不修改阶段审计文件。
    证书各段稍后仍由context_views核验边界与监督覆盖；不能只保留QA阶段而漏掉动作loss。
    """
    from copy import deepcopy

    if execution["version"] != VERSION or qa["version"] != VERSION:
        raise ValueError("两阶段窗口版本不同")
    if execution["max_image_turn"] != qa["max_image_turn"]:
        raise ValueError("动作和QA的窗口策略不同")
    for key in ("spans", "image_keys"):
        if list(qa[key][: len(execution[key])]) != list(execution[key]):
            raise ValueError("QA没有延续原动作视觉布局")
    if not execution["segments"] or not qa["segments"]:
        raise ValueError("两阶段窗口缺少实际生成段")
    if execution["segments"][-1]["end"] > qa["segments"][0]["start"]:
        raise ValueError("动作和QA生成边界重叠")
    result = deepcopy(qa)
    result["segments"] = deepcopy(execution["segments"] + qa["segments"])
    return result
