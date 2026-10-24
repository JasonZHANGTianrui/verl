"""生成任务的RL消费者：来源与目标显式分开，复用测评准备和共同图/成功接口。"""

import hashlib
import json
import os
import threading
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np

from .libero_session import LiberoSession, bind_render_context

SCHEMA = "physical-ai.generated-rl-task.v1"
RELEASE = "physical-ai.generated-rl-release.v1"


def validate_task(task):
    """核对公开指令/固定QA/来源身份及目标数；不读预览图或拿初态GT填未来答案。"""
    from benchmark.libero.questions.registry import default_registry

    from .task_sampling import goal_count

    if task.get("schema") != SCHEMA or task.get("split") != "train":
        raise ValueError("不是生成来源训练记录")
    if task["environment"]["initial_condition"]["kind"] != "libero-bddl-seeded-reset-preview-v1":
        raise ValueError("未知生成初态")
    count = goal_count(task)
    if task["execution_budget_steps"] != (300 if count == 1 else 520):
        raise ValueError("生成任务预算与目标数不符")
    default_registry().validate(task["followup"]["question_spec"])
    return count


def verify_release(release, root):
    """完整验证独立发布及实际共享代码身份；原candidate保持未验收标记，不偷换schema。"""
    from .training_state import content_hash

    release, root = Path(release), Path(root)
    manifest = json.loads((release / "manifest.json").read_text())
    if manifest["schema"] != RELEASE:
        raise ValueError("未知生成任务发布版本")
    for name, digest in manifest["files"].items():
        path = release / name
        if not path.resolve().is_relative_to(release.resolve()) or content_hash(path) != digest:
            raise ValueError("任务资源改变：" + name)
    for name, digest in manifest["implementation_files"].items():
        if content_hash(root / name) != digest:
            raise ValueError("生成执行/图实现不同：" + name)
    return manifest


def verify_acceptance(release, manifest):
    """正式训练需要同发布、同消费者的物理验收回执，预览图/旧模型成绩不能替代。

    回执由独立准备/目标/组起点/分叉检查生成；绑定实际文件hash和代表类别。
    该检查不放宽GT或任务难度，不允许仅把candidate.training_ready改成true。
    """
    from .training_state import content_hash

    release = Path(release)
    receipt = json.loads((release / "consumer-acceptance.json").read_text())
    if receipt.get("manifest_sha256") != content_hash(release / "manifest.json"):
        raise ValueError("生成来源验收不属于本次发布")
    if receipt.get("status") != "passed" or set(receipt.get("scopes", [])) != {"single", "combination", "replacement"}:
        raise ValueError("生成来源三类验收不完整")
    folder = Path(__file__).parent
    expected = {
        name: content_hash(folder / name) for name in ("generated_tasks.py", "task_data.py", "libero_session.py")
    }
    if receipt.get("consumer_files") != expected:
        raise ValueError("消费者与实际验收版本不一致")
    if receipt.get("records") != manifest["records"]:
        raise ValueError("验收记录数不符")
    return receipt


class GeneratedSession(LiberoSession):
    """继承通用动作/观察/完整fork生命周期；只替换初态和实际目标，不调用示范恢复。"""

    def __init__(self, record, resources, output, max_steps, _fork_restore=False):
        """正常分支prepare_case执行一次dummy；fork容器仅reset，随后由父fork完整恢复。

        物理seed固定取record.source，不使用策略采样seed。目标绑定不移动任何物体；
        模型只读取self.task公开指令和真实观察。output为本次私有准备/目标绑定目录。
        """
        from benchmark.adapters.libero_online import VERSION
        from ours.rl.generation.execution import prepare_case
        from ours.rl.generation.objectives import bind_target
        from ours.rl.generation.scenes import open_seeded

        from .online_reward import save_json

        validate_task(record)
        self.owner = (os.getpid(), threading.get_ident())
        self.env = None
        self.closed = False
        self.broken = False
        self.steps = 0
        self.max_steps = max_steps
        self.action_history = []
        self.settings = dict(record=record, resources=resources, output=output, max_steps=max_steps)
        folder = Path(output) / ("fork-" + uuid.uuid4().hex if _fork_restore else "initial")
        folder.mkdir(parents=True, exist_ok=False)
        try:
            if _fork_restore:
                path = Path(record["generated_case"]["generated_task"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["generated_case"]["generated_task_sha256"]:
                    raise ValueError("fork目标文件变更")
                target = json.loads(path.read_text())
                self.env, _ = open_seeded(target["source"])
                monitor = bind_target(self.env, target, folder)
                if monitor.contract != target["success_contract"]:
                    raise ValueError("fork目标契约不同")
                self.observation = {}  # 父fork必须覆盖为真实父观察；此容器不交付模型。
            else:
                self.env, task, self.observation = prepare_case(record["generated_case"], folder)
                if task.language != record["instruction"]:
                    raise ValueError("实际任务语言不同")
                prep = self.env.preparation
                if prep["physical"]["initialization"] != VERSION or prep["online_preparation"]["steps"] != 10:
                    raise ValueError("新源未完成一次固定dummy")
            self.task = record["instruction"]
            bind_render_context(self.env.env.sim)
            self.preparation = deepcopy(self.env.preparation)
            from OpenGL import GL

            self.env.env.sim._render_context_offscreen.gl_ctx.make_current()
            self.render_backend = dict(
                vendor=GL.glGetString(GL.GL_VENDOR).decode(),
                renderer=GL.glGetString(GL.GL_RENDERER).decode(),
                egl_device_id=os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"),
            )
            required = os.environ.get("PHYSICAL_AI_REQUIRE_GL_VENDOR")
            if required and self.render_backend["vendor"] != required:
                raise ValueError("渲染后端不符")
            self.last_action = np.array([0.0] * 6 + [-1.0])
            save_json(
                folder / "consumer.json",
                dict(
                    schema=SCHEMA,
                    fork_container=_fork_restore,
                    dummy_steps_executed=0 if _fork_restore else 10,
                    physical_seed=record["environment"]["initial_condition"]["seed"],
                    instruction=self.task,
                ),
            )
        except BaseException:
            self.close()
            raise


class GeneratedCapture:
    """按已接通测评相同的reader记录实际控制步，不通过SFT来源/原指令校验绕路。"""

    def __init__(self, session, record, output, rollout_id):
        """准备后第0步建图；源scene_config只作绑定/对照，不作为动态真值。"""
        from benchmark.adapters.libero_evaluation import create_scene_reader
        from benchmark.adapters.libero_execution import assert_same_state, runtime_state
        from benchmark.core.scene_graph import SceneRecorder

        from .online_reward import save_json

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        before = runtime_state(session.env)
        expected = json.loads(Path(record["scene_config"]).read_text())
        self.reader = create_scene_reader(session.env, output, expected)
        if self.reader.config != expected["config"]:
            raise ValueError("生成任务实际场景图配置与来源不一致")
        self.recorder = SceneRecorder(rollout_id, max_gap_s=self.reader.config["max_sample_gap_s"])
        self.observe(0)
        assert_same_state(before, runtime_state(session.env))
        save_json(output / "preparation.json", session.preparation)
        save_json(
            output / "source-contract.json",
            dict(
                schema=SCHEMA,
                source=record["environment"]["initial_condition"],
                goal=record["objective"],
                logical_task_step=0,
            ),
        )

    def observe(self, step, *, capture_cameras=False):
        """与生成任务测评逐步图读取相同；不刷新/推进模拟器，不用私有目标示例补事实。"""
        self.recorder.observe(self.reader.read(step))
