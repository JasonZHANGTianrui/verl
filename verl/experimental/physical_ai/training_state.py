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

"""连续SegGRPO的组边界恢复与日志；不在半个物理回合伪造续跑。"""

import hashlib
import json
import os
from pathlib import Path

import torch

from .online_reward import save_json


def build_optimizer(model, config, *, distributed=False):
    """建立持久优化器；AdamW可用PyTorch ZeRO-1分片状态，模型/梯度仍按原路径同步。

    分片只改变状态存放位置，参数更新仍由torch.optim.AdamW实现；关闭foreach以减少
    临时张量峰值。保存前所有rank必须consolidate_state_dict，恢复沿官方load_state_dict。
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=config["learning_rate"])
    if config["optimizer"] != "adamw":
        raise ValueError("未支持的优化器")
    options = dict(
        lr=config["learning_rate"],
        betas=tuple(config.get("adam_betas", [0.9, 0.999])),
        eps=config.get("adam_eps", 1e-8),
        weight_decay=config.get("weight_decay", 0.0),
        foreach=False,
    )
    if config.get("optimizer_state_sharding", False):
        if not distributed:
            raise ValueError("优化器状态分片必须以torchrun多进程启动")
        from torch.distributed.optim import ZeroRedundancyOptimizer

        return ZeroRedundancyOptimizer(params, optimizer_class=torch.optim.AdamW, **options)
    return torch.optim.AdamW(params, **options)


def content_hash(path):
    """分块核对完整checkpoint内容；只存路径/大小不能保证恢复了原权重。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def restore_optimizer(optimizer, state):
    """沿官方load_state_dict恢复；预先放好ZeRO本rank的状态，避免两层状态视图重复占GPU。

    ZeRO的外层Optimizer与本地AdamW都会处理同一份state。先只将本rank拥有的非标量
    Tensor搬到目标设备，两次官方加载即可共享相同storage；其他rank状态仍留CPU，
    由官方ZeRO加载器丢弃。step标量维持CPU。只用公开param_groups对应保存ID，
    不改AdamW公式，不调用ZeRO的私有分片方法；传入字典与官方接口一样会被修改。
    """
    if hasattr(optimizer, "optim"):
        owned = {id(p) for group in optimizer.optim.param_groups for p in group["params"]}
        for live, saved in zip(optimizer.param_groups, state["param_groups"], strict=True):
            for param, identity in zip(live["params"], saved["params"], strict=True):
                if id(param) in owned and identity in state["state"]:
                    state["state"][identity] = {
                        key: value.to(param.device) if isinstance(value, torch.Tensor) and value.ndim else value
                        for key, value in state["state"][identity].items()
                    }
    optimizer.load_state_dict(state)


def cpu_tree(value):
    """复制训练状态到CPU用于持久化，不改变在GPU上的模型/优化器对象。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def save_boundary(policy, optimizer, directory, *, completed_groups, optimizer_steps, contract, rng_states):
    """原子保存完整策略、优化器、游标和每rank RNG，最后才发布可恢复manifest。

    directory必须新建。读者只接受已发布的checkpoint.json；写到一半的building文件
    不可恢复。这里只在整组采集/更新完成后调用，不将父环境或半个QA组当可恢复点。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "training.pt"
    temporary = directory / "training.building"
    torch.save(
        dict(
            model_state_dict=cpu_tree(policy.state_dict()),
            optimizer_state_dict=cpu_tree(optimizer.state_dict()),
            completed_groups=completed_groups,
            optimizer_steps=optimizer_steps,
            contract=contract,
            rng_states=cpu_tree(rng_states),
            schema="physical-ai.seggrpo-training.v1",
        ),
        temporary,
    )
    temporary.rename(path)
    receipt = dict(
        path=str(path.resolve()),
        sha256=content_hash(path),
        completed_groups=completed_groups,
        optimizer_steps=optimizer_steps,
        contract=contract,
    )
    save_json(directory / "checkpoint.json", receipt)
    return receipt


def load_boundary(manifest, contract):
    """核验组边界checkpoint，再恢复；配置/数据/权重变化不能接着使用旧游标。

    只加载本项目创建、已校验SHA的训练文件；weights_only保证不执行外来pickle对象。
    manifest中的路径允许跨机器重定位，但对应字节身份必须保持。
    """
    receipt = json.loads(Path(manifest).read_text())
    if receipt["contract"] != contract or content_hash(receipt["path"]) != receipt["sha256"]:
        raise ValueError("恢复配置/数据或checkpoint SHA不一致")
    state = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    if state["schema"] != "physical-ai.seggrpo-training.v1" or state["contract"] != contract:
        raise ValueError("checkpoint内部训练契约不同")
    for key in ("completed_groups", "optimizer_steps"):
        if state[key] != receipt[key]:
            raise ValueError("checkpoint游标与manifest不一致")
    return state, receipt


def make_tracker(config, output, key_file=None):
    """复用verl Tracking；密钥从私有文件读进环境，不进配置、日志或checkpoint。

    只允许rank0调用。日志包含数值指标和公开训练配置；不上传会话、图片、GT或凭据。
    云连接错误明确抛出，不能宣称已接入SwanLab却只写本地日志。
    """
    from verl.utils.tracking import Tracking

    if "swanlab" in config["logger"] and key_file:
        secret = Path(key_file).read_text().strip()
        if not secret or "\n" in secret:
            raise ValueError("SwanLab私有密钥文件内容无效")
        os.environ["SWANLAB_API_KEY"] = secret
    os.environ["SWANLAB_LOG_DIR"] = str(Path(output) / "swanlog")
    return Tracking(config["project"], config["experiment"], default_backend=config["logger"], config=config)
