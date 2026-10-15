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
