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

"""完整SegGRPO批次的分片与梯度归约；不在各卡的小片内重新计算优势。"""

import torch
import torch.distributed as dist


def sum_gradients(model):
    """把各rank已按全局样本数缩放的梯度求和，保持与单进程目标相同。

    调用者须先在完整组上算优势，再分配响应；每条响应的loss仅出现一次。
    此处不能再除world_size。无有效响应的rank也必须参加；某参数在本rank未使用
    时贡献零，所有rank都未使用则保留grad=None，避免优化器误更新未参与参数。
    需要先初始化进程组，模型/可训练参数顺序须相同；任何非有限梯度在更新前报错。
    """
    if not dist.is_initialized():
        raise RuntimeError("梯度归约要求已初始化进程组")
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    layouts = [None] * dist.get_world_size()
    dist.all_gather_object(layouts, [(name, tuple(p.shape), str(p.dtype)) for name, p in params])
    if not params or any(item != layouts[0] for item in layouts):
        raise ValueError("各rank可训练参数结构不同或为空")
    device = params[0][1].device
    valid = torch.tensor(
        [all(p.grad is None or bool(torch.isfinite(p.grad).all()) for _, p in params)], dtype=torch.int32, device=device
    )
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not bool(valid.item()):
        raise FloatingPointError("至少一个rank出现非有限梯度，禁止全部rank更新")
    used = torch.tensor([p.grad is not None for _, p in params], dtype=torch.int32, device=device)
    dist.all_reduce(used, op=dist.ReduceOp.SUM)
    for index, (_, param) in enumerate(params):
        if not used[index].item():
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
