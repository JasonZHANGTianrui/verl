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

"""单卡原生G0.5的真实工具回合采集；供SegGRPO接入诊断，不替代verl分布式server。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from verl.experimental.physical_ai.online_reward import public_text, save_json

from .sampling import sample_response


def chat_fingerprint(chat):
    """绑定完整token/模态类型/图像/state内容，供QA分支核对共同条件，不读GT。"""
    h = hashlib.sha256(json.dumps([chat.ids, chat.types]).encode())
    for key, value in list(chat.pixels.items()) + [(str(i), v) for i, v in enumerate(chat.states)]:
        array = value.detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str((array.shape, array.dtype)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def collect_phase(
    session,
    chat,
    *,
    phase,
    env_steps,
    token_budget,
    seed,
    directory,
    generated=None,
    old_logprobs=None,
    recorder=None,
    reader=None,
    scene_capture=None,
):
    """采集一个阶段并保存全部原始生成ID与概率，工具观察不标成策略输出。

    session/chat已经包含合法前缀，QA可从完整动作历史副本继续。env_steps是本阶段
    允许的真实20Hz控制/保持步，token_budget为本阶段总生成上限。原生成功不提前停。
    仅正常无工具响应或执行预算耗尽结束阶段；错误工具输出记录为模型错误而不补动作。
    未完成消息的生成截断显式抛错，当前诊断不伪造EOS拼接后续回合。
    """
    import imageio.v2 as imageio
    from experiments.libero.libero_eval_utils import (
        LiberoGripperCommandState,
        build_libero_raw_obs,
        extract_libero_images,
        single_action_to_libero_action,
    )
    from inference import CachedSession
    from rollout.g05.chat_libero import hold_last_goal
    from rollout.g05.tool_runtime import (
        ActionDecodeError,
        append_feedback,
        bind_action_ids,
        current_proprio,
        decode_segment,
        execute_segments,
        parse_request,
    )
    from runtime import to_device

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    generated = list(generated) if generated is not None else [False] * len(chat.ids)
    old_logprobs = list(old_logprobs) if old_logprobs is not None else [0.0] * len(chat.ids)
    if len(generated) > len(chat.ids) or len(old_logprobs) != len(generated):
        raise ValueError("生成mask与已存在会话不对齐")
    generated.extend([False] * (len(chat.ids) - len(generated)))
    old_logprobs.extend([0.0] * (len(chat.ids) - len(old_logprobs)))
    prefix_length = len(chat.ids)
    before_steps = session.steps
    history_start = len(session.action_history)
    engine = CachedSession(chat.policy.model)
    rng = torch.Generator(device="cuda").manual_seed(seed)
    eos = chat.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    stop_ids = [eos, chat.policy.model.cfg.eos_token_id]
    grip = LiberoGripperCommandState.from_obs(session.observe())
    responses, traces = [], []
    consumed = 0
    pending = False
    termination = "execution_budget"
    video = imageio.get_writer(str(directory / "scene.mp4"), fps=20, codec="libx264")
    video.append_data(extract_libero_images(session.observation)["image"])
    try:
        while consumed < token_budget and session.steps - before_steps < env_steps:
            inputs = to_device(chat.inputs(), torch.device("cuda"))
            budget = min(token_budget - consumed, int(engine.model.cfg.vlm.max_position_embeddings) - len(chat.ids))
            if budget < 1:
                raise RuntimeError("native_context_capacity_exhausted")
            response = sample_response(
                engine,
                inputs,
                max_new_tokens=budget,
                stop_ids=stop_ids,
                temperature=1.0,
                generator=rng,
                mode=chat.rl_sampling_mode,
                compute_dtype=chat.rl_compute_dtype,
            )
            ids = response["ids"]
            text = chat.tokenizer.decode(ids, skip_special_tokens=False)
            trace = dict(phase=phase, start_step=session.steps, input_tokens=len(chat.ids), text=text, **response)
            save_json(directory / f"call-{len(traces):03d}.json", trace)
            traces.append(trace)
            if response["termination"] != "message_end":
                raise RuntimeError("unsupported_incomplete_message:" + response["termination"])
            chat.append_generated(ids)
            generated.extend([True] * len(ids))
            old_logprobs.extend(response["logprobs"])
            consumed += len(ids)
            pending = False
            try:
                request = parse_request(text)
                if request is not None:
                    request = bind_action_ids(request, ids, chat.tokenizer, chat.policy.action_tokenizer)
            except (ValueError, ActionDecodeError):
                from tool_calls import TAGS

                tag = TAGS.search(text)
                if tag:
                    responses.append(public_text(text[: tag.start()]))
                termination = "model_tool_error"
                break
            if request is None:
                responses.append(public_text(text))
                termination = "model_text_end"
                break
            from tool_calls import TAGS

            match = TAGS.search(text)
            if match:
                responses.append(public_text(text[: match.start()]))

            # LiveChat保存的是单本体processor；原decode_segment需要完整processor容器。
            processor = getattr(chat, "rl_processor", None)
            if processor is None:
                raise ValueError("未绑定原生完整processor")

            def decode(segment, index, processor=processor):
                """每个段读取实际执行前state，复用原codec与统计后处理，不重编码动作。"""
                raw, _ = build_libero_raw_obs(session.observation, session.task)
                return decode_segment(chat.policy, processor, segment, current_proprio(chat, raw))

            def step(decoded, index, mode, segment_index):
                """执行并逐步读取同次图；GT和真实成功状态只记私有审计。"""
                action = np.asarray(
                    [0.0] * 6 + [grip.current_command]
                    if mode == "hold"
                    else single_action_to_libero_action(
                        {k: v[index] for k, v in decoded["controls"].items()},
                        gripper_state=grip,
                        obs=session.observation,
                    ),
                    dtype=np.float32,
                )
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("decoded_control_invalid")
                session.observation, *_ = session.env.step(action)
                session.steps += 1
                session.last_action = action.copy()
                session.action_history.append(dict(step=session.steps, mode=mode, action=action.tolist()))
                if scene_capture is not None:
                    scene_capture.observe(session.steps)
                elif recorder is not None:
                    recorder.observe(reader.read(session.steps))
                video.append_data(extract_libero_images(session.observation)["image"])

            execution = execute_segments(
                request,
                env_steps - (session.steps - before_steps),
                decode,
                step,
                hold=lambda: hold_last_goal(session.env),
            )
            save_json(directory / f"execution-{len(traces):03d}.json", execution)
            raw, _ = build_libero_raw_obs(session.observation, session.task)
            append_feedback(chat, raw, request, execution, session.steps)
            generated.extend([False] * (len(chat.ids) - len(generated)))
            old_logprobs.extend([0.0] * (len(chat.ids) - len(old_logprobs)))
            pending = True
            if execution["error"]:
                termination = "model_action_decode_error"
                break
        else:
            termination = "execution_budget" if session.steps - before_steps >= env_steps else "token_budget"
        result = dict(
            phase=phase,
            prefix_length=prefix_length,
            end_length=len(chat.ids),
            generated=generated,
            old_logprobs=old_logprobs,
            termination=termination,
            pending_assistant=pending,
            answer="\n".join(t for t in responses if t),
            env_steps=session.steps - before_steps,
            generated_tokens=consumed,
        )
        save_json(
            directory / "controls.json",
            dict(
                source="actual_environment_steps",
                control_hz=20,
                start_step=before_steps,
                commands=session.action_history[history_start:],
            ),
        )
        save_json(directory / "phase.json", result)
        torch.save(chat.inputs(), directory / "inputs.pt")
        save_json(directory / "tokens.json", dict(ids=chat.ids, types=chat.types))
        return result
    finally:
        video.close()
        del engine
