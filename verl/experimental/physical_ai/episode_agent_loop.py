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

"""在verl原生工具循环上延长有状态工具生命周期；不改token、模板或观察拼接。"""

from contextvars import ContextVar

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.function_tool import FunctionTool

_instances = ContextVar("physical_ai_episode_tools", default=None)


class _EpisodeTool:
    """为一次run保存真实工具实例；原生每次调用的release延后到整回合finally。

    只包装有状态工具；execute仍收到原生agent_data，因此动作codec可以直接绑定原始
    response_ids，不能从参数字符串重新encode生成另一串动作token。
    """

    def __init__(self, tool):
        """保存原工具和schema；不创建模拟器，不复制工具的可变内部状态。"""
        self.tool = tool
        self.name = tool.name
        self.tool_schema = tool.tool_schema
        self.instance_id = None
        self.creation_response = None

    async def create(self, **kwargs):
        """每回合首次调用创建实例，后续调用沿用同一ID及原始创建响应。"""
        if self.instance_id is None:
            self.instance_id, self.creation_response = await self.tool.create(**kwargs)
        return self.instance_id, self.creation_response

    async def execute(self, *args, **kwargs):
        """原样转发参数和agent_data；实际环境访问必须由工具服务串行化。"""
        return await self.tool.execute(*args, **kwargs)

    async def release(self, instance_id):
        """原循环逐次调用的release不关闭场景，统一由close执行。"""
        return None

    async def close(self):
        """整条会话结束时关闭一次；清空ID防止错误恢复路径重复释放。"""
        if self.instance_id is not None:
            instance_id, self.instance_id = self.instance_id, None
            await self.tool.release(instance_id)


class EpisodeToolAgentLoop(ToolAgentLoop):
    """保留verl原生生成/历史/mask/log-prob，只扩展有状态工具的存活范围。

    每次run有独立ContextVar，包括同一loop对象并发处理的样本。工具必须按instance_id
    隔离状态；robot.execute的max_parallel_calls应为1，禁止同一机器人并行写控制。
    初始观察仍须由环境接入层在run之前提供，本类不从任意历史中补图或猜state。
    """

    async def run(self, *args, **kwargs):
        """正常结束、预算终止及异常均释放本回合工具，不清空模型的回合内历史。"""
        if self.max_parallel_calls != 1:
            raise ValueError("机器人回合要求max_parallel_calls=1，不能并行执行控制")
        instances = {}
        token = _instances.set(instances)
        try:
            return await super().run(*args, **kwargs)
        finally:
            errors = []
            try:
                for instance in instances.values():
                    try:
                        await instance.close()
                    except Exception as error:
                        errors.append(error)
            finally:
                _instances.reset(token)
            if errors:
                raise RuntimeError("回合工具释放失败，不能复用该环境worker") from errors[0]

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        """仅替换本次agent_data的有状态工具引用，沿用官方解析和错误返回。

        未注册工具、坏JSON仍由官方处理；无状态FunctionTool不包装。不会修改self.tools，
        其他并发样本不共享本回合代理。输出及response_ids原样交给官方循环。
        """
        instances = _instances.get()
        if instances is None:
            raise RuntimeError("工具调用必须发生在run生命周期内")
        active = getattr(agent_data, "_active_tools", self.tools)
        tool = active.get(tool_call.name)
        if tool is not None and not isinstance(tool, (FunctionTool, _EpisodeTool)):
            if tool_call.name not in instances:
                instances[tool_call.name] = _EpisodeTool(tool)
            agent_data._active_tools = dict(active, **{tool_call.name: instances[tool_call.name]})
        return await super()._call_tool(tool_call, tools_kwargs, agent_data)
