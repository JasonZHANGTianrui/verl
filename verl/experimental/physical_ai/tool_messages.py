"""独立解析同一回复的工具调用；非法部分保留为原文，不阻断其他合法调用。"""

import re

from rollout.g05.tool_runtime import ActionDecodeError, bind_action_ids, parse_request


def message_requests(text, ids, tokenizer, codec):
    """返回合法请求、非工具文字及拒绝诊断；始终绑定整条回复中的原生动作ID。

    显式tool_call之间独立校验，单个缺外壳调用仍交原解析器修复。这里只抓模型格式/
    动作ID错误，不吞掉基础设施异常。不重tokenize、不改历史模态类型，也不执行动作。
    """
    thoughts = [(m.start(), m.end()) for m in re.finditer(r"<think>.*?(?:</think>|$)", text, re.S)]
    starts = [m.start() for m in re.finditer(r"<tool_call>", text) if not any(a <= m.start() < b for a, b in thoughts)]
    ranges = list(zip(starts, [*starts[1:], len(text)], strict=False)) if starts else [(0, len(text))]
    accepted, rejected, spans = [], [], []
    for start, end in ranges:
        part = text[start:end]
        try:
            request = parse_request(part)
            if request is None:
                continue
            # 原解析器处理片段；ID绑定必须恢复到整条原文的字符坐标。
            for segment in request["segments"]:
                segment["action_span"] = tuple(start + i for i in segment["action_span"])
            request["raw_text"] = text
            request = bind_action_ids(request, ids, tokenizer, codec)
        except (ValueError, ActionDecodeError) as exc:
            rejected.append(dict(start=start, end=end, reason=str(exc)))
            continue
        first = re.search(r"<tool_call>|<function=|<parameter=", part)
        begin = start + first.start() if first else start
        closing = re.search(r"</tool_call>", part)
        stop = start + closing.end() if closing else end
        if any(a <= begin < b for a, b in thoughts):
            continue
        accepted.append(request)
        spans.append((begin, stop))
    # 只移除合法调用的执行语法；非法调用仍属于模型公开原文。
    cursor = 0
    ordinary = []
    for start, end in spans:
        ordinary.append(text[cursor:start])
        cursor = end
    ordinary.append(text[cursor:])
    return accepted, "".join(ordinary), rejected


def append_results(chat, results):
    """按原Qwen模板返回合法调用的真实反馈；一条模型消息的多个tool结果不插入假assistant。

    results按执行顺序包含(request,result,end_step,raw_obs)。单调用沿原接口，多个调用
    交原renderer一次渲染角色外壳，每份观察仅在自己实际返回的位置展开。占位符仅供
    拼装，绝不进入模型token；历史原生成IDs和KV不改写。
    """
    from rollout.g05.tool_runtime import append_feedback, feedback_text

    if len(results) == 1:
        request, result, step, raw = results[0]
        append_feedback(chat, raw, request, result, step)
        return
    messages = [
        dict(role="system", content=chat.system),
        dict(role="user", content=chat.instruction),
        dict(role="assistant", content=""),
    ]
    observations = {}
    for i, (request, result, step, raw) in enumerate(results):
        if request["format"] == "single":
            content = (
                f"observed_steps={result['observed_steps']}; observation_mode=hold_last_goal; "
                if result["observed_steps"]
                else ""
            )
            state = "complete" if result["executed_steps"] == 32 else "partial"
            content += (
                f"planned_steps=32; executed_steps={result['executed_steps']}; "
                f"execution_status={state}; <image><image> <state>"
            )
        else:
            content = feedback_text(result, step)
        marker = f"__RL_OBSERVATION_{i}__"
        observations[marker] = raw
        messages.append(
            dict(role="tool", tool_call_id=f"live-call-{i}", content=content.replace("<image><image> <state>", marker))
        )
    text = chat.renderer.render(messages, tools=chat.tools, generation="action")
    text = "\n" + text[text.rindex("<|im_start|>user\n<tool_response>") :]
    for piece in re.split(r"(__RL_OBSERVATION_\d+__)", text):
        if piece in observations:
            chat.append_rendered("<image><image> <state>", observations[piece])
        elif piece:
            chat.append_rendered(piece)
