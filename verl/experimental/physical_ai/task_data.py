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

"""将SFT派生RL任务连接到已有环境与Scene map；不加载示范动作、教师回复或旧GT。"""

import json
from pathlib import Path

from ours.rl.tasks import load_initial, resolve_resource

from .libero_session import LiberoSession

EVALUATION_ACTION_RULE = "libero-native-at-question-delivery-v1"
SHARED_ACTION_RULE = "libero-shared-at-question-delivery-v1"


def needs_preparation_reference(online, *, seed, resolution):
    """区分历史预演回执与新在线校验契约；新模式不要求提前模拟同一个初态。

    两种模式均固定环境seed/相机分辨率，源码/配置和源初态校验仍由调用方执行。
    v2必须显式声明运行时核验且没有旧回执；未知模式拒绝，不能静默取消检查。
    """
    if online["seed"] != seed or online["resolution"] != resolution:
        raise ValueError("在线环境seed/分辨率与发布契约不符")
    if online["schema"] == "physical-ai.rl-online-execution.v1":
        if not online.get("preparation_ref"):
            raise ValueError("历史在线契约缺少准备回执")
        return True
    if (
        online["schema"] == "physical-ai.rl-online-execution.v2"
        and online.get("preparation_policy") == "verify_during_rollout"
        and "preparation_ref" not in online
    ):
        return False
    raise ValueError("未知或混用的初始化校验模式")


def training_tasks(rows, rule):
    """按明确评分版本选择任务，并逐题检查已有机械查询契约。

    与当前测评一致的普通LIBERO模式保留全部训练记录；旧设备小试仍显式按任务名
    筛选。既不按源成败筛任务，也不按题型挑容易评分的题。未知策略/参数直接报错。
    """
    from benchmark.libero.questions.registry import default_registry

    if rule["version"] in {EVALUATION_ACTION_RULE, SHARED_ACTION_RULE}:
        selected = list(rows)
        supported = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
        if any(r["split"] != "train" or r["environment"]["suite"] not in supported for r in selected):
            raise ValueError("测评同口径奖励只接普通LIBERO训练任务")
    elif rule["version"] == "rl-device-state-goal-v1":
        selected = [r for r in rows if r["environment"]["task_name"] == rule["task_name"]]
    else:
        raise ValueError("未知动作奖励版本")
    registry = default_registry()
    for row in selected:
        registry.validate(row["followup"]["question_spec"])
    return selected


def task_reward_at_delivery(task, rule, session, reader, recorder, termination):
    """在追加用户QA之前计算动作分；新模式对应当前G0.5测评的同一采样时刻。

    原生success只读取当前环境，不提前停机器人、不补动作。模型生成的非法工具
    指令不执行，按实际环境结果计分并照常续问；环境/接口异常不能变成零分。QA不反改此分。
    这是显式的测评兼容版本，不冒充尚未统一接入的自定义任务判据。
    """
    if rule["version"] == SHARED_ACTION_RULE:
        monitor = session.success_monitor
        row = monitor.observe(session.steps)
        return dict(
            task_reward=float(row["success"]),
            native_success=row["native_success"],
            success_details=row["details"],
            success_contract=monitor.contract,
            model_output_valid=termination not in {"model_tool_error", "model_action_decode_error"},
            rule=rule,
            step=session.steps,
            action_metric="shared_at_question_delivery",
        )
    if rule["version"] != EVALUATION_ACTION_RULE:
        return state_task_reward(task, rule, reader, recorder)
    from benchmark.adapters.libero_execution import task_success

    if task["environment"]["suite"] not in {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}:
        raise ValueError("此奖励版本未支持非普通LIBERO来源")
    native = bool(task_success(session.env))
    valid_output = termination not in {"model_tool_error", "model_action_decode_error"}
    return dict(
        task_reward=float(native),
        native_success=native,
        model_output_valid=valid_output,
        rule=rule,
        step=session.steps,
        action_metric="native_at_question_delivery",
    )


def load_task_release(release, task_id):
    """读取显式指定的训练任务并核验发布文件SHA，不按旧答案或执行分数选题。

    返回私有任务及资源索引；模型只能通过既有会话构造器读取instruction和真实观察。
    导出的待绑定奖励字段不等于可训练，调用方仍须绑定、记录独立任务奖励规则。
    """
    from ours.sft.shared.resources import sha256

    release = Path(release).resolve(strict=True)
    manifest = json.loads((release / "manifest.json").read_text())
    for name in ("train.jsonl", "resources.json"):
        if sha256(release / name) != manifest["files"][name]:
            raise ValueError("RL发布文件内容变化：" + name)
    rows = [json.loads(line) for line in (release / "train.jsonl").read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row["id"] == task_id]
    if len(matches) != 1 or matches[0]["split"] != "train":
        raise ValueError("需要唯一的已导出训练任务ID")
    resources = json.loads((release / "resources.json").read_text())["entries"]
    return matches[0], resources


def state_task_reward(task, rule, reader, recorder):
    """按显式设备状态目标计算0/1动作奖励，和QA问题/旧SFT结果完全分离。

    当前只接已校准状态目标；任务名称、规则版本及语义图版本须完全匹配。
    query使用冻结图中的英文语义状态，不读qpos自行设阈值、不用native success兜底。
    未知或无效枚举阻止整个组更新；返回完整私有事实供审计。
    """
    from benchmark.libero.questions.selection import query_fact

    if (
        rule["version"] != "rl-device-state-goal-v1"
        or task["environment"]["task_name"] != rule["task_name"]
        or reader.config["version"] != rule["calibration"]
        or rule["query"]["op"] != "attribute"
        or not rule["query"]["path"].startswith("states.")
    ):
        raise ValueError("任务奖励绑定或状态校准版本不符")
    fact = query_fact(recorder.freeze("rl-task-reward"), rule["query"])
    if fact.status != "ok" or fact.value not in rule["allowed_states"]:
        raise ValueError("任务状态未知或不在冻结枚举内，不能填0")
    if rule["target_state"] not in rule["allowed_states"]:
        raise ValueError("任务目标状态不在冻结枚举内")
    return dict(task_reward=float(fact.value == rule["target_state"]), state=fact.value, rule=rule)


def open_task_session(task, resources, *, seed, max_steps, resolution=256):
    """按源训练初态创建独立会话，预算来自RL运行配置而非示范长度。

    seed用于环境初始化；max_steps是20Hz控制步数。返回会话由调用者with/finally关闭。
    task中的后续问题和评分信息不传给模型；公开输入由模型适配器按实际观察构造。
    """
    from benchmark.adapters import libero_env  # noqa: F401
    from libero.libero import benchmark

    initial = load_initial(task, resources)
    suite_name = task["environment"]["suite"]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    indices = [i for i in range(suite.n_tasks) if suite.get_task(i).name == initial.task_name]
    if len(indices) != 1:
        raise ValueError("训练任务未唯一绑定官方suite")
    session = LiberoSession(
        suite_name,
        indices[0],
        seed=seed,
        max_steps=max_steps,
        resolution=resolution,
        initial_condition=initial,
    )
    expected = task.get("online_execution", {}).get(
        "initialization_version", task["environment"]["initialization_version"]
    )
    if session.env.scene_initialization["version"] != expected:
        session.close()
        raise ValueError("RL与来源的场景初始化规则不一致")
    online = task.get("online_execution")
    if online:
        from benchmark.adapters.libero_execution import assert_same_preparation, implementation_identity
        from benchmark.libero.qa_relations import VERSION as relation_version
        from ours.rl.prepare_online import semantic_identity

        if (
            online["implementation"] != implementation_identity()
            or online["relation_view"] != relation_version
            or online["semantics"] != semantic_identity()
        ):
            session.close()
            raise ValueError("在线执行/关系实现与新任务发布不一致")
        try:
            if needs_preparation_reference(online, seed=seed, resolution=resolution):
                expected_preparation = json.loads(resolve_resource(online["preparation_ref"], resources).read_text())
                assert_same_preparation(expected_preparation, session.preparation)
        except Exception:
            session.close()
            raise
    return session


def create_scene_capture(session, task, resources, output, *, rollout_id):
    """复用SFT的支撑/容纳/外观/事件读取，开始本次独立在线历史。

    只加载冻结场景配置；不读取SFT旧图。新观察从控制步0开始，后续每个真实执行步
    调capture.observe。源问题的实体ID/题义保持固定，历史身份改为实际rollout ID。
    配置不一致就报错，不能把旧题用另一套关系判定而不作记录。
    """
    from benchmark.core.scene_graph import SceneRecorder
    from ours.sft.execution.codec_materials import SceneCapture

    if not isinstance(rollout_id, str) or not rollout_id:
        raise ValueError("需要真实rollout身份")
    online = task.get("online_execution", {})
    expected = json.loads(
        resolve_resource(online.get("scene_config_ref", task["environment"]["scene_config_ref"]), resources).read_text()
    )
    initial = session.settings["initial_condition"]
    capture = SceneCapture(
        session.env,
        initial,
        task["environment"]["suite"],
        output,
        task["provenance"]["source"]["source_sha256"],
    )
    current = json.loads((Path(output) / "scene-config.json").read_text())
    for key in [
        "config",
        "containment_calibration",
        "ordinary_containment",
        "visibility",
    ]:
        if current.get(key) != expected.get(key):
            raise ValueError("在线Scene map配置与复用题目不一致：" + key)
    from benchmark.adapters.libero_execution import assert_same_state, runtime_state

    from .online_reward import save_json

    # 图采集初始化是只读行为；必须与统一准备完成后的状态一致，不另推进物理。
    assert_same_state(session.preparation["state"], runtime_state(session.env))
    save_json(Path(output) / "preparation.json", session.preparation)
    save_json(Path(output) / "online-execution.json", online)
    capture.eid = rollout_id
    capture.case["case_id"] = rollout_id
    capture.recorder = SceneRecorder(rollout_id, max_gap_s=capture.reader.config["max_sample_gap_s"])
    capture.observe(0, capture_cameras=True)
    return capture
