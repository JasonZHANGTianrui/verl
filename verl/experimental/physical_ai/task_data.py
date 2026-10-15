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
    if session.env.scene_initialization["version"] != task["environment"]["initialization_version"]:
        session.close()
        raise ValueError("RL与来源的场景初始化规则不一致")
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
    expected = json.loads(resolve_resource(task["environment"]["scene_config_ref"], resources).read_text())
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
    capture.eid = rollout_id
    capture.case["case_id"] = rollout_id
    capture.recorder = SceneRecorder(rollout_id, max_gap_s=capture.reader.config["max_sample_gap_s"])
    capture.observe(0, capture_cameras=True)
    return capture
