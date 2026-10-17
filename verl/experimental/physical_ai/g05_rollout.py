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
from benchmark.adapters.libero_execution import hold_control_targets

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


def phase_steps(
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
    max_calls=None,
    per_message_budget=False,
    timeout_seconds=120,
):
    """采集一个阶段并保存全部原始生成ID与概率，工具观察不标成策略输出。

    session/chat已经包含合法前缀，QA可从完整动作历史副本继续。env_steps是本阶段
    允许的真实20Hz控制/保持步，QA的token_budget为阶段累计上限；动作可显式按单消息计。原生成功不提前停。
    仅正常无工具响应或执行预算耗尽结束阶段；错误工具输出记录为模型错误而不补动作。
    每个模型调用以yield交给调度器，收到原token/概率后继续实际执行；多个generator
    可在同一owner线程交错运行，使独立会话的生成请求形成GPU batch。
    未完成消息的生成截断显式抛错，当前诊断不伪造EOS拼接后续回合。
    """
    import imageio.v2 as imageio
    from experiments.libero.libero_eval_utils import (
        LiberoGripperCommandState,
        build_libero_raw_obs,
        extract_libero_images,
        single_action_to_libero_action,
    )
    from rollout.g05.tool_runtime import (
        current_proprio,
        decode_segment,
        execute_segments,
    )
    from runtime import to_device

    from .native_runtime import cached_session

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
    engine = cached_session(chat.policy.model)
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
        while (
            (per_message_budget or consumed < token_budget)
            and session.steps - before_steps < env_steps
            and (max_calls is None or len(traces) < max_calls)
        ):
            inputs = to_device(chat.inputs(), torch.device("cuda"))
            budget = min(
                token_budget if per_message_budget else token_budget - consumed,
                int(engine.model.cfg.vlm.max_position_embeddings) - len(chat.ids),
            )
            if budget < 1:
                raise RuntimeError("native_context_capacity_exhausted")
            response = yield dict(
                engine=engine,
                inputs=inputs,
                max_new_tokens=budget,
                stop_ids=stop_ids,
                temperature=1.0,
                generator=rng,
                mode=chat.rl_sampling_mode,
                compute_dtype=chat.rl_compute_dtype,
                timeout_seconds=timeout_seconds,
            )
            ids = response["ids"]
            text = chat.tokenizer.decode(ids, skip_special_tokens=False)
            trace = dict(
                phase=phase,
                start_step=session.steps,
                input_tokens=len(chat.ids),
                text=text,
                exact_stop_token_id=ids[-1] if ids else None,
                exact_stop_kind=(
                    "message_end"
                    if ids and ids[-1] == eos
                    else "native_eos"
                    if ids and ids[-1] == chat.policy.model.cfg.eos_token_id
                    else response["termination"]
                ),
                **response,
            )
            save_json(directory / f"call-{len(traces):03d}.json", trace)
            traces.append(trace)
            if response["termination"] != "message_end":
                raise RuntimeError("unsupported_incomplete_message:" + response["termination"])
            chat.append_generated(ids)
            generated.extend([True] * len(ids))
            old_logprobs.extend(response["logprobs"])
            consumed += len(ids)
            pending = False
            from .tool_messages import append_results, message_requests

            requests, ordinary, rejected = message_requests(text, ids, chat.tokenizer, chat.policy.action_tokenizer)
            save_json(directory / f"tool-requests-{len(traces):03d}.json", dict(accepted=requests, rejected=rejected))
            responses.append(public_text(ordinary))
            if not requests:
                termination = "model_tool_error" if rejected else "model_text_end"
                break

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

            results = []
            for request_index, request in enumerate(requests):
                remaining = env_steps - (session.steps - before_steps)
                execution = execute_segments(
                    request, remaining, decode, step, hold=lambda: hold_control_targets(session.env)
                )
                save_json(directory / f"execution-{len(traces):03d}-{request_index:02d}.json", execution)
                if execution["error"] and not execution["consumed_steps"]:
                    continue
                raw, _ = build_libero_raw_obs(session.observation, session.task)
                results.append((request, execution, session.steps, raw))
            if not results:
                termination = "model_action_decode_error"
                break
            append_results(chat, results)
            generated.extend([False] * (len(chat.ids) - len(generated)))
            old_logprobs.extend([0.0] * (len(chat.ids) - len(old_logprobs)))
            pending = True
        else:
            termination = (
                "execution_budget"
                if session.steps - before_steps >= env_steps
                else "message_budget"
                if max_calls is not None and len(traces) >= max_calls
                else "token_budget"
            )
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


def collect_phase(*args, **kwargs):
    """单条诊断兼容入口；与批量模式共用phase_steps，只用旧采样函数处理每次请求。

    不另写第二份动作/状态/问答逻辑。异常会关闭生成器，让其finally释放视频资源。
    返回值仍为原阶段记录，供旧小试和历史配置复现。
    """
    generator = phase_steps(*args, **kwargs)
    try:
        request = next(generator)
        while True:
            response = sample_response(**request)
            request = generator.send(response)
    except StopIteration as finished:
        return finished.value
    finally:
        generator.close()
