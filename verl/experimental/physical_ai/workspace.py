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

"""定位可独立clone的Physical AI集成工作区，不依赖本模块在submodule中的目录深度。"""

import os
import sys
from pathlib import Path


def workspace_root():
    """优先读取PHYSICAL_AI_ROOT，否则沿cwd及本文件祖先寻找实际项目根目录。

    显式路径无效时直接报错，防止悄悄绑定另一份代码。纯SegGRPO数学模块不调用本
    函数；只有需要原生G0.5、模拟器或数据管线的集成入口才需要外部工作区。
    """
    explicit = os.environ.get("PHYSICAL_AI_ROOT")
    candidates = [Path(explicit)] if explicit else [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    for candidate in candidates:
        root = candidate.resolve()
        if all(
            (root / part).is_dir()
            for part in ("data-pipeline/ours", "training-pipeline/ours/g05", "training-pipeline/g05")
        ):
            return root
    raise FileNotFoundError("请将PHYSICAL_AI_ROOT指向包含data-pipeline和G0.5适配器的Physical AI工作区")


def configure_workspace():
    """加入集成依赖的实际目录并返回根路径；不改verl的来源、不安装包或覆盖环境。"""
    root = workspace_root()
    paths = [
        str(root / part)
        for part in (
            "training-pipeline/ours/g05",
            "training-pipeline/ours",
            "training-pipeline/g05/src",
            "training-pipeline/g05",
            "data-pipeline",
            "repos/LIBERO",
        )
    ]
    sys.path[:0] = [path for path in paths if path not in sys.path]
    source = os.environ.get("PHYSICAL_AI_G05_HANDOFF")
    if source:
        from .native_runtime import configure_handoff

        configure_handoff(root, source)
    return root
