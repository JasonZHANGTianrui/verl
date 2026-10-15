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

"""保护G0.5原生PyTorch DeltaNet fallback的FP32递推，避免外层AMP重新降精度。"""

from functools import wraps

import torch


def stable_delta_precision(model):
    """只包裹已确认的torch fallback函数，采样和actor共用；不修改上游文件或权重。

    原函数内部把q/k/v等转成FP32，但外层BF16 autocast仍会把矩阵乘重新降精度，
    而逐token递推使用的乘加保留FP32。关闭该函数内部autocast使两者遵循相同数学
    精度；同时在归一化前提升q/k/v精度，输出dtype、梯度和缓存接口保持原函数约定。
    使用最高FP32矩阵精度并关闭TF32；此为显式诊断数值版本，重复调用幂等。
    """
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for layer in model.vlm.layers:
        if layer.layer_type != "linear_attention":
            continue
        module = layer.linear_attn
        for name in ["chunk_gated_delta_rule", "recurrent_gated_delta_rule"]:
            original = getattr(module, name)
            if getattr(original, "_physical_ai_fp32_delta", False):
                rows.append(dict(layer=module.layer_idx, function=name, protected=True))
                continue
            if original.__module__.endswith("qwen35.gated_deltanet") and original.__name__.startswith("torch_"):

                def bind(function):
                    """逐函数绑定原调用，避免循环闭包错指最后一层实现。"""

                    @wraps(function)
                    def call(*args, **kwargs):
                        """q/k/v归一化和递推统一FP32；返回特征恢复原dtype，状态保留FP32。"""
                        query = args[0] if args else kwargs["query"]
                        promoted = tuple(value.float() if i < 3 else value for i, value in enumerate(args))
                        options = {
                            key: value.float() if key in {"query", "key", "value"} else value
                            for key, value in kwargs.items()
                        }
                        with torch.autocast("cuda", enabled=False):
                            output, state = function(*promoted, **options)
                        return output.to(query.dtype), state

                    call._physical_ai_fp32_delta = True
                    return call

                setattr(module, name, bind(original))
                rows.append(dict(layer=module.layer_idx, function=name, protected=True))
            else:
                rows.append(
                    dict(
                        layer=module.layer_idx,
                        function=name,
                        protected=False,
                        implementation=original.__module__ + "." + original.__name__,
                    )
                )
    return dict(version="torch-delta-fp32-v1", functions=rows)
