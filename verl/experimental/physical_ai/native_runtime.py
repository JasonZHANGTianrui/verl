"""固定ReActPI训练交接的模型接口；环境/查询仍来自当前Physical AI工作区。"""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

_CHAT = None
_IDENTITY = None


def configure_handoff(root, source):
    """按文件SHA装载唯一交接，不以同名模块代替；恢复sys.path以隔离旧环境/GT。

    root提供当前v5环境和数据接口；source只提供模型loader、renderer、cache及辅助。
    不修改交接文件、官方子模块或模型参数。若已有同名模块来自其他路径，直接拒绝。
    """
    global _CHAT, _IDENTITY
    source = Path(source).resolve(strict=True)
    spec = json.loads((Path(__file__).parent / "configs/reactpi-v1-runtime.json").read_text())
    for relative, expected in spec["files"].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("ReActPI交接文件不匹配：" + relative)
    for relative, expected in spec["upstream_files"].items():
        if hashlib.sha256((root / "training-pipeline/g05" / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("官方G05观察/codec实现不符：" + relative)
    # 只登记准确的子模块，不替换verl自己的scripts父包。
    official = (root / "training-pipeline/g05/scripts/serve_policy.py").resolve()
    key = "scripts.serve_policy"
    if key in sys.modules:
        if Path(sys.modules[key].__file__).resolve() != official:
            raise ValueError("官方观察转换已被同名模块覆盖")
    else:
        loader = importlib.util.spec_from_file_location(key, official)
        module = importlib.util.module_from_spec(loader)
        sys.modules[key] = module
        try:
            loader.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(key, None)
            raise
    # 先绑定本工作区的包，避免模型诊断模块导入其旧版benchmark/数据管线。
    import benchmark
    import ours.sft.conversation

    for module in (benchmark, ours.sft.conversation):
        if not Path(module.__file__).resolve().is_relative_to(root / "data-pipeline"):
            raise ValueError("环境/数据包已被其他工作区覆盖")
    # 交接使用分包前的导入名，只桥接相同常量，不替换当前SFT包或载入旧制作管线。
    import ast

    from ours.sft.conversation.conversation import SYSTEM, TOOLS

    syntax = ast.parse((source / "data-pipeline/ours/sft/conversation.py").read_text())
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in syntax.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SYSTEM", "TOOLS"}
    }
    if constants != dict(SYSTEM=SYSTEM, TOOLS=TOOLS):
        raise ValueError("当前system/tools与v1训练声明不一致")
    ours.sft.conversation.SYSTEM = SYSTEM
    ours.sft.conversation.TOOLS = TOOLS
    preferred = [str(source / "training-pipeline/ours/g05"), str(source / "training-pipeline/ours")]
    for name in ("runtime", "generate", "inference", "tool_calls", "chat_template", "compiler", "g05_ar_only"):
        if name in sys.modules and not Path(sys.modules[name].__file__).resolve().is_relative_to(source):
            raise ValueError("模型模块已从错误来源加载：" + name)
    previous = list(sys.path)
    try:
        sys.path[:0] = preferred
        modules = {
            name: importlib.import_module(name)
            for name in ("runtime", "generate", "inference", "tool_calls", "chat_template")
        }
        path = source / "data-pipeline/rollout/g05/chat_libero.py"
        loader = importlib.util.spec_from_file_location("reactpi_pinned_chat", path)
        module = importlib.util.module_from_spec(loader)
        sys.modules[loader.name] = module
        loader.loader.exec_module(module)
        _CHAT = module.LiveChat
    finally:
        sys.path[:] = preferred + [p for p in previous if p not in preferred]
    for name, module in modules.items():
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError("交接导入路径不符：" + name)
    _IDENTITY = dict(
        commit=spec["commit"], files=spec["files"], upstream_files=spec["upstream_files"], source=str(source)
    )


def identity():
    """返回已加载模型来源，供训练契约记录；未配置时明确返回旧本地接口模式。"""
    return _IDENTITY or dict(mode="legacy_workspace")


def live_chat_class():
    """新运行使用已固定交接类；未设置handoff时仅保留旧诊断入口兼容。"""
    if os.environ.get("PHYSICAL_AI_G05_HANDOFF"):
        if _CHAT is None:
            raise RuntimeError("尚未初始化固定ReActPI交接")
        return _CHAT
    from rollout.g05.chat_libero import LiveChat

    return LiveChat


def cached_session(model):
    """复用原缓存并显式兼容RL的return_logits选项，不重写KV/卷积/递推路径。

    固定交接没有return_logits参数。关闭返回时仍调用原append，只丢弃本次输出head
    结果；少量额外投影换取原实现复用，模型缓存/位置和生成ID均保持。原数值保护照常生效。
    """
    from inference import CachedSession

    if _CHAT is None:
        return CachedSession(model)

    class RLCache(CachedSession):
        """只适配返回值，不改变原生缓存更新。"""

        def append(self, inputs, *, return_logits=True):
            """原样更新输入与附件；return_logits=False仅省略向调用者返回logits。"""
            logits = super().append(inputs)
            return logits if return_logits else None

    return RLCache(model, chunk_observations=True)
