"""滞后一批的QA经历保存与恢复；原历史只作前缀，新QA仍由当前策略生成。

不序列化MuJoCo/EGL对象，也不只用qpos伪造完整环境。恢复沿实际控制逐步重放，
保持段严格还原原目标锁定边界；物理、控制器、传感器和交付GT通过后才续问。
所有附件来自本次已完成动作，不能接SFT示范或另一条轨迹的答案。
"""

import hashlib
import json
from contextlib import nullcontext
from copy import deepcopy
from itertools import groupby
from pathlib import Path

import numpy as np
import torch


def digest(path):
    """计算本地不可变附件摘要；读取失败直接报错，不用存在标记代替内容校验。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def history_input_path(output, resume_root, completed, *, task_index, tasks_per_batch):
    """定位上一采样批同槽位父经历；恢复首批从原checkpoint所属输出目录读取。

    task_index/completed均按任务组计，批号从0起；返回绝对目录字符串或热启动None。
    不按文件mtime猜最新经历，也不缺失时偷偷改用当前父轨迹。
    """
    if task_index < tasks_per_batch:
        return None
    previous = task_index - tasks_per_batch
    root = Path(resume_root) if previous < completed and resume_root is not None else Path(output)
    path = root / f"batch-{previous // tasks_per_batch:06d}" / f"task-{previous:06d}" / "history-parent"
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(f"上一批父经历未完整发布：{path}")
    return str(path.resolve())


def runtime_values(session):
    """提取可比对的完整控制器/传感器纯值，不复制环境引用或Python绑定方法。

    sim积分state保留原生单位；控制器数值按原数组精度逐值比较。方法恢复由真实控制
    重放完成，仿真器对象不进入torch存档。QA恢复不在此修改物理或采样时间。
    """
    from .libero_session import snapshot_observables

    def plain(value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, bool, int, float, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        raise TypeError(f"未支持的控制器状态类型：{type(value)}")

    controllers = []
    for robot in session.env.robots:
        controllers.append({k: plain(v) for k, v in vars(robot.controller).items() if k != "sim" and not callable(v)})
    return dict(
        state=session.env.sim.get_state().flatten().copy(),
        controllers=controllers,
        grippers=[r.gripper.current_action.copy() for r in session.env.robots],
        sensors=snapshot_observables(session.env.env),
        observation=deepcopy(session.observation),
        last_action=session.last_action.copy(),
        steps=session.steps,
        model_xml=session.env.sim.model.get_xml(),
    )


def assert_equal(actual, expected, path="runtime"):
    """递归逐值核对恢复状态；图像仅允许已有同态渲染≤1量化级，其余不放宽。"""
    from .libero_session import compare_fork_image

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"恢复字段不同：{path}")
        for key in expected:
            assert_equal(actual[key], expected[key], path + "/" + str(key))
    elif isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise ValueError(f"恢复序列不同：{path}")
        for i, value in enumerate(expected):
            assert_equal(actual[i], value, path + "/" + str(i))
    elif isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"恢复数组结构不同：{path}")
        if expected.dtype == np.uint8 and expected.ndim == 3 and expected.shape[-1] == 3:
            compare_fork_image(actual, expected)
        elif not np.array_equal(actual, expected):
            raise ValueError(f"恢复数值不同：{path}")
    elif actual != expected:
        raise ValueError(f"恢复值不同：{path}")


def save_parent(directory, session, chat, phase, binding, *, config, task, selected, reward, policy_version):
    """原子发布选中父经历；保存完整原生前缀、实际控制、准备和私有GT。

    原动作概率只作审计，后续QA的loss从新响应开始；附件只包含CPU张量/纯值。
    manifest最后写入，未发布目录不能用于下一批。父经历不随QA分支被修改。
    """
    from .g05_rollout import chat_fingerprint
    from .online_reward import save_json

    directory.mkdir(parents=True, exist_ok=False)
    tokens = dict(
        ids=list(chat.ids),
        types=list(chat.types),
        pixels={k: v.detach().cpu().clone() for k, v in chat.pixels.items()},
        states=[v.detach().cpu().clone() for v in chat.states],
    )
    torch.save(
        dict(chat=tokens, runtime=runtime_values(session), preparation=deepcopy(session.preparation)),
        directory / "state.pt",
    )
    save_json(
        directory / "experience.json",
        dict(
            task=task,
            config=config,
            selected=selected,
            reward=reward,
            policy_version=policy_version,
            phase=phase,
            binding=binding,
            controls=session.action_history,
            prefix_sha256=chat_fingerprint(chat),
        ),
    )
    save_json(
        directory / "manifest.json",
        dict(
            schema="physical-ai.qa-history.v1",
            sampling_batch=config["sampling_batch_index"],
            case_id=config["case_id"],
            files={name: digest(directory / name) for name in ("state.pt", "experience.json")},
        ),
    )


def restore_parent(directory, output, policy, processor, resources, *, current_config):
    """恢复上一批真实经历，再使用当前策略回答；出错关闭环境并拒绝生成QA。

    原问题和GT由历史文件提供，当前动作任务的question/实体名称不能覆盖它们。
    重放只执行原记录控制，不运行旧策略；每步重采事实，并核对交付绑定完全相等。
    模型历史使用保存的全部原图/state/token，重放图仅用来验证，不能替换模型所见。
    """
    from benchmark.adapters.libero_execution import assert_same_preparation, hold_control_targets
    from benchmark.adapters.libero_success import TaskSuccessMonitor

    from .g05_rollout import chat_fingerprint
    from .libero_session import restore_observables
    from .native_runtime import live_chat_class
    from .online_reward import bind_delivery, save_json
    from .task_data import SHARED_ACTION_RULE, create_scene_capture, open_task_session

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != "physical-ai.qa-history.v1"
        or manifest["sampling_batch"] + 1 != current_config["sampling_batch_index"]
    ):
        raise ValueError("QA经历不是上一批的完整发布")
    if set(manifest["files"]) != {"state.pt", "experience.json"}:
        raise ValueError("旧经历附件清单不完整")
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError("旧经历内容摘要不符：" + name)
    experience = json.loads((directory / "experience.json").read_text())
    saved = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
    config, task = experience["config"], experience["task"]
    if task["id"] != manifest["case_id"] or config["case_id"] != task["id"]:
        raise ValueError("历史任务身份不一致")
    selected = experience["selected"]
    session = open_task_session(
        task, resources, seed=config["environment_seed"], max_steps=config["execution_steps"] + config["qa_steps"]
    )
    try:
        assert_same_preparation(saved["preparation"], session.preparation)
        capture = create_scene_capture(session, task, resources, output / "scene", rollout_id=selected)
        if config["task_rule"]["version"] == SHARED_ACTION_RULE:
            session.success_monitor = TaskSuccessMonitor(session.env, family="libero")
            session.success_monitor.observe(0)
        commands = experience["controls"]
        if [r["step"] for r in commands] != list(range(1, len(commands) + 1)):
            raise ValueError("旧动作控制步不连续")
        if any(r.get("mode") not in {"execute", "hold", "action"} or "replay_span" not in r for r in commands):
            raise ValueError("旧动作缺少原控制模式或保持段边界")
        for (span, mode), rows in groupby(commands, key=lambda row: (tuple(row["replay_span"]), row["mode"])):
            with hold_control_targets(session.env) if mode == "hold" else nullcontext():
                for row in rows:
                    action = np.asarray(row["action"], dtype=np.float32)
                    session.observation, *_ = session.env.step(action)
                    session.steps += 1
                    session.last_action = action.copy()
                    session.action_history.append(deepcopy(row))
                    if getattr(session, "success_monitor", None) is not None:
                        session.success_monitor.observe(session.steps)
                    capture.observe(session.steps)
        assert_equal(runtime_values(session), saved["runtime"])
        binding = bind_delivery(
            config["question"], capture.recorder, config["entity_names"], selected, task_instruction=session.task
        )
        if binding != experience["binding"]:
            raise ValueError("旧经历重放的交付GT/题目绑定不同")
        # 恢复已核验的父观察/传感器原值，避免允许的1级渲染舍入进入模型后续反馈。
        session.observation = deepcopy(saved["runtime"]["observation"])
        restore_observables(session.env.env, saved["runtime"]["sensors"])
        chat = live_chat_class()(policy, processor, **({"thinking": True} if config.get("thinking", False) else {}))
        chat.instruction = session.task
        chat.rl_processor, chat.rl_sampling_mode = processor, current_config["sampling_mode"]
        chat.rl_compute_dtype = current_config["compute_dtype"]
        for name, value in saved["chat"].items():
            setattr(chat, name, value)
        if chat_fingerprint(chat) != experience["prefix_sha256"]:
            raise ValueError("恢复的完整模型前缀不同")
        history = dict(
            rollout_id=str(directory.resolve()) + "/" + selected,
            case_id=task["id"],
            policy_version=experience["policy_version"],
            manifest_sha256=digest(manifest_path),
            source_batch=manifest["sampling_batch"],
            current_batch=current_config["sampling_batch_index"],
        )
        save_json(
            output / "verified.json",
            dict(
                history_parent=history,
                prefix_sha256=experience["prefix_sha256"],
                binding_sha256=binding["binding_sha256"],
                replayed_steps=session.steps,
                runtime_equal=True,
                preparation_equal=True,
                gt_equal=True,
            ),
        )
        return dict(
            session=session,
            chat=chat,
            phase=experience["phase"],
            binding=binding,
            question=config["question"],
            selected=selected,
            reward=experience["reward"],
            history_parent=history,
        )
    except BaseException:
        session.close()
        raise
