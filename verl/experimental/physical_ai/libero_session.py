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

"""RL的LIBERO会话容器；负责真实环境生命周期，不负责模型生成、codec或奖励。"""

from __future__ import annotations

import os
import threading
from copy import deepcopy

import numpy as np


def bind_render_context(sim):
    """给本实例render绑定自己的EGL context，允许同一线程交错使用多个模拟器。

    当前robosuite的sim.render没有在每次调用前make_current；创建第二个环境后，
    第一环境可能误用第二个环境的OpenGL framebuffer/纹理。只包裹该sim实例，
    不改上游或渲染参数；每次读取实际offscreen context，避免XML重建后沿用旧引用。
    """
    if getattr(sim, "_physical_ai_render_bound", False):
        return
    original = sim.render

    def render(*args, **kwargs):
        """先激活本模拟器上下文，再原样执行相机/分割读取。"""
        sim._render_context_offscreen.gl_ctx.make_current()
        return original(*args, **kwargs)

    sim.render = render
    sim._physical_ai_render_bound = True


def compare_fork_image(parent, child):
    """在原生uint8精度下核对同一物理状态的独立渲染，返回私有差异记录。

    独立GL上下文可在量化边界出现1个灰度级的舍入差；只接受每通道绝对差≤1。
    此函数仅在完整积分状态、几何及相机变换已逐项相等后调用，不用于放宽GT或
    比较不同物理状态。初始QA仍复制父回合原始观察，不能用这张复核图替换输入。
    """
    parent, child = np.asarray(parent), np.asarray(child)
    if parent.dtype != np.uint8 or child.dtype != np.uint8 or parent.shape != child.shape:
        raise ValueError("分叉图像必须具有相同uint8形状")
    if parent.ndim != 3 or parent.shape[-1] != 3 or not parent.size:
        raise ValueError("分叉图像必须为非空RGB数组")
    difference = np.abs(parent.astype(np.int16) - child.astype(np.int16))
    result = dict(
        max_delta=int(difference.max()),
        changed_pixels=int(np.any(difference, axis=-1).sum()),
        rule="same-state-uint8-render-lsb-v1",
    )
    if result["max_delta"] > 1:
        raise ValueError(f"分叉渲染差异超出1个uint8量化级：{result}")
    return result


class LiberoSession:
    """一条rollout拥有一个环境，多次动作调用共享同一物理状态。

    必须在同一进程、同一线程创建和使用；异步Agent Loop应把它放进独立环境worker，
    而不是在多个API线程之间传递MuJoCo/EGL对象。这里接收的已经是解码并转换好的
    LIBERO七维OSC控制，不是模型的27维动作，也不是文本中的动作码。
    """

    def __init__(
        self,
        suite: str,
        task_index: int,
        *,
        initial_state_index: int = 0,
        seed: int = 0,
        max_steps: int = 400,
        resolution: int = 256,
        physics_patch: str | None = None,
        initial_condition=None,
    ):
        """构造、reset并稳定初态；预算按20Hz控制步计，不含初始化等待。

        初始状态从官方suite读取。使用本项目统一光照、隐藏辅助site和场景稳定入口。
        initial_condition显式提供时从训练示范XML/states[0]恢复，不索引官方测试初态。
        初始化失败会关闭已创建环境并抛异常，不能当成模型任务失败。
        """
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps必须为正整数控制步")
        if type(initial_state_index) is not int or initial_state_index < 0:
            raise ValueError("initial_state_index必须为非负整数")
        from benchmark.adapters.libero_env import make_env

        self.settings = dict(
            suite=suite,
            task_index=task_index,
            initial_state_index=initial_state_index,
            seed=seed,
            max_steps=max_steps,
            resolution=resolution,
            physics_patch=physics_patch,
            initial_condition=initial_condition,
        )
        self.owner = (os.getpid(), threading.get_ident())
        self.env = None
        self.steps = 0
        self.max_steps = max_steps
        self.closed = False
        self.broken = False
        self.action_history = []
        try:
            if initial_condition is not None:
                from ours.sft.sources.libero_source import open_environment

                if physics_patch is not None:
                    raise ValueError("训练示范初态不能叠加未声明物理补丁")
                self.env, task, _, self.observation = open_environment(initial_condition, suite, resolution, seed=seed)
            else:
                self.env, task, states = make_env(suite, task_index, resolution, physics_patch=physics_patch)
                self.env.seed(seed)
                self.env.reset()
                if initial_state_index >= len(states):
                    raise IndexError("初始状态索引越界")
                self.observation = self.env.set_init_state(states[initial_state_index])
            self.task = task.language
            # 统一set_init_state已经刷新控制器和首观察，回执即本次任务第0步。
            self.preparation = deepcopy(self.env.preparation)
            bind_render_context(self.env.env.sim)
            self.last_action = np.array([0.0] * 6 + [-1.0])
        except BaseException:
            self.close()
            raise

    def _check_owner(self):
        """拒绝跨线程/进程使用，防止串场景及EGL上下文失效。"""
        if self.owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("LIBERO会话必须由创建它的环境worker串行使用")
        if self.closed or self.broken:
            raise RuntimeError("会话已关闭或环境执行异常，不能继续伪造轨迹")

    def observe(self):
        """返回当前真实观察的副本；不推进物理，不额外筛帧，也不附加GT/成功标签。

        原生观察保留相机和机器人state；模型侧应继续用原G0.5 processor选取与映射，
        不能直接把整个模拟器观察dict塞给模型。返回副本避免调用者改坏本会话缓存。
        """
        self._check_owner()
        return deepcopy(self.observation)

    def execute_controls(self, actions, *, observe_steps: int = 0):
        """串行执行一段1–32步已解码控制，再保持末目标；返回观察和真实执行回执。

        actions形状[T,7]：前三维平移、后三维旋转均为LIBERO OSC归一化增量，最后一维
        夹爪指令-1开/+1闭；不是米/弧度的绝对位姿。输入先完整校验，非法控制不执行。
        observe_steps为0–100个20Hz保持步，调用公共hold_control_targets；它与控制共用预算。
        预算截断不伪称全部完成，原生success只供审计，不提前终止或自动补动作。
        """
        self._check_owner()
        commands = np.asarray(actions, dtype=np.float64)
        if commands.ndim != 2 or commands.shape[1] != 7 or not 1 <= len(commands) <= 32:
            raise ValueError("actions需要1至32行、每行7维LIBERO控制")
        if not np.isfinite(commands).all():
            raise ValueError("动作含NaN或Inf")
        low, high = self.env.env.action_spec
        if np.any(commands < low) or np.any(commands > high):
            raise ValueError("动作越出环境控制范围，不静默裁剪")
        if type(observe_steps) is not int or not 0 <= observe_steps <= 100:
            raise ValueError("observe_steps必须为0至100的整数")
        from benchmark.adapters.libero_execution import hold_control_targets
        from rollout.g05.tool_runtime import execute_segments

        request = dict(
            format="single",
            segments=[
                dict(
                    execute_steps=len(commands),
                    observe_steps=observe_steps,
                )
            ],
        )

        def decode(segment, index):
            """仅传递调用者已解码的控制；不重新编码token，不加载策略权重。"""
            return dict(controls=commands)

        def step(decoded, index, mode, segment_index):
            """逐步更新同一环境；保持步复用末控制，由外层锁住OSC目标。"""
            action = self.last_action if mode == "hold" else decoded["controls"][index]
            self.observation, _, _, _ = self.env.step(action)
            self.last_action = action.copy()
            self.steps += 1
            self.action_history.append(dict(step=self.steps, mode=mode, action=action.tolist()))

        try:
            receipt = execute_segments(
                request,
                self.max_steps - self.steps,
                decode,
                step,
                hold=lambda: hold_control_targets(self.env),
            )
        except BaseException:
            self.broken = True
            raise
        receipt.update(observation_step=self.steps, budget_exhausted=self.steps >= self.max_steps)
        return self.observe(), receipt

    def fork(self):
        """建立同构环境副本，复制已验证的完整MjData/控制器缓存，父环境保持不动。

        复用precision模式的clone_runtime，不用只含qpos/qvel的简化snapshot。模型XML
        必须一致；复制后核对完整积分state及父/子即时渲染双图。该接口只在同一环境worker调用，
        不是跨机器序列化协议；随机数状态恢复到分叉前，避免创建子环境改变后续随机流。
        """
        import random

        from ours.sft.execution.precision_encoding import clone_runtime

        self._check_owner()
        numpy_rng, python_rng = np.random.get_state(), random.getstate()
        child = None
        try:
            child = type(self)(**self.settings)
            if self.env.env.sim.model.get_xml() != child.env.env.sim.model.get_xml():
                raise ValueError("环境模型XML不同，拒绝套用物理缓存")
            clone_runtime(self.env, child.env)
            child.preparation = deepcopy(self.preparation)
            child.steps = self.steps
            child.last_action = self.last_action.copy()
            child.action_history = deepcopy(self.action_history)
            # env.step返回的是传感器采样缓存，可能略早于最后一个物理子步。
            # 分叉不向模型新增观察；用即时渲染验证复制，再保留父回合实际交付的旧观察。
            cached_observation = deepcopy(self.observation)
            parent_current = self.env.env._get_observations(force_update=True)
            child_current = child.env.env._get_observations(force_update=True)
            child.observation = cached_observation
            if not np.array_equal(
                self.env.env.sim.get_state().flatten(),
                child.env.env.sim.get_state().flatten(),
            ):
                raise ValueError("分支物理状态与父轨迹不同")
            # 积分状态之外也核对实际用于绘制的几何/相机变换；不拿渲染容差掩盖位姿不同。
            for name in (
                "body_xpos",
                "body_xmat",
                "geom_xpos",
                "geom_xmat",
                "site_xpos",
                "site_xmat",
                "cam_xpos",
                "cam_xmat",
                "light_xpos",
                "light_xdir",
            ):
                if not np.array_equal(getattr(self.env.sim.data, name), getattr(child.env.sim.data, name)):
                    raise ValueError("分支渲染变换与父轨迹不同：" + name)
            child.fork_validation = dict(
                physical_state_equal=True,
                render_transforms_equal=True,
                initial_observation="exact_parent_copy",
                cameras={
                    camera: compare_fork_image(parent_current[camera], child_current[camera])
                    for camera in ["agentview_image", "robot0_eye_in_hand_image"]
                },
            )
            return child
        except BaseException:
            if child is not None:
                child.close()
            raise
        finally:
            np.random.set_state(numpy_rng)
            random.setstate(python_rng)

    def audit(self):
        """返回私有运行状态；native_success仅用于诊断，不能当作既定RL奖励或模型输入。"""
        self._check_owner()
        return dict(
            steps=self.steps,
            remaining_steps=self.max_steps - self.steps,
            native_success=bool(self.env.check_success()),
            simulation_time_s=float(self.env.env.sim.data.time),
        )

    def close(self):
        """幂等释放本会话的EGL及模拟器资源；退出、异常和取消都由owner执行。"""
        if self.closed:
            return
        if self.owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("只能在创建会话的worker中关闭环境")
        self.closed = True
        if self.env is not None:
            # 释放MuJoCo的GL资源也必须在其创建上下文内，不能污染仍存活的其他分支。
            context = getattr(getattr(self.env.env, "sim", None), "_render_context_offscreen", None)
            if context is not None:
                context.gl_ctx.make_current()
            self.env.close()

    def __enter__(self):
        """进入显式资源生命周期，方便异常时仍关闭环境。"""
        self._check_owner()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """无论诊断成功或失败都释放环境，不吞掉异常。"""
        self.close()
