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
import shutil
from pathlib import Path

import torch

from .online_reward import save_json


def training_budget(config, tasks_per_batch):
    """返回计划任务数；新接口按采样batch计，旧max_groups仍按任务数计。

    两种预算不能同时提供。整数/整除检查避免100被误当100个mini-batch或100个
    epoch；当前采样器按固定训练记录列表循环，新种子产生新的真实rollout。
    """
    keys = [k for k in ("total_training_steps", "max_groups") if k in config]
    if len(keys) != 1 or type(config[keys[0]]) is not int or config[keys[0]] < 1:
        raise ValueError("必须提供唯一的正整数total_training_steps或旧max_groups")
    total = config[keys[0]] * tasks_per_batch if keys[0] == "total_training_steps" else config[keys[0]]
    if tasks_per_batch < 1 or total % tasks_per_batch:
        raise ValueError("任务预算须能构成完整采样batch")
    return total


def validate_fixed_algorithm(config):
    """核对尚固定的算法选项；显式填写不支持的值必须报错，不能只上传参数但忽略它。

    这些值对应当前已验证路径：标准组内标准差归一化、阶段等权、完整词表温度1
    采样、固定学习率且无reference KL/熵奖励。未来实现新路径时同步扩展校验。
    """
    fixed = dict(
        advantage_estimator="seggrpo",
        norm_adv_by_std_in_grpo=True,
        advantage_epsilon=1e-6,
        loss_agg_mode="seq-mean-token-mean",
        execution_loss_weight=1.0,
        qa_loss_weight=1.0,
        parent_selection="uniform",
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        entropy_coeff=0.0,
        lr_scheduler_type="constant",
        lr_warmup_steps=0,
    )
    for key, value in fixed.items():
        if config.get(key, value) != value:
            raise ValueError(f"当前原生后端尚不支持{key}={config[key]!r}")
    policy = config.get("unknown_qa_policy", "stop")
    if policy not in {"stop", "retain_action_skip_qa_group"}:
        raise ValueError("未知QA组处理策略")
    fixed["unknown_qa_policy"] = policy
    return fixed


def checkpoint_due(completed_batches, save_freq, *, final=False):
    """按完整采样batch计保存周期；正常结束或显式边界停止也保存，不按mini更新计数。

    completed_batches从1开始，save_freq必须为正整数。即使该批全零优势、没有参数
    更新，采样游标也已推进，因此仍按采样批数判断；异常中断不伪造最终checkpoint。
    """
    if type(save_freq) is not int or save_freq < 1 or completed_batches < 1:
        raise ValueError("保存频率及已完成batch数须为正整数")
    return final or completed_batches % save_freq == 0


def prune_checkpoints(root, latest, keep):
    """新checkpoint和latest发布成功后，仅清理本输出目录的旧完整checkpoint。

    keep为正整数或None（不清理）。只接受同一训练契约、路径位于root直接子目录的
    checkpoint；跳过临时文件、符号链接和外部恢复源。最新checkpoint须已存在，
    防止新保存失败时先删除唯一恢复点。删除不涉及源权重、GPU验收或其他实验目录。
    """
    if keep is None:
        return []
    if type(keep) is not int or keep < 1:
        raise ValueError("max_ckpt_to_keep须为正整数或null")
    root = Path(root).resolve()
    newest = Path(latest["path"])
    if newest.parent.parent.resolve() != root or not newest.is_file():
        raise ValueError("新checkpoint未完整保存于本次输出目录")
    published = json.loads((newest.parent / "checkpoint.json").read_text())
    if published != latest:
        raise ValueError("新checkpoint的发布凭据不一致")
    candidates = []
    for directory in root.glob("group-*"):
        manifest = directory / "checkpoint.json"
        if directory.is_symlink() or not manifest.is_file() or manifest.is_symlink():
            continue
        receipt = json.loads(manifest.read_text())
        target = Path(receipt["path"])
        if (
            receipt["contract"] == latest["contract"]
            and target == directory / "training.pt"
            and target.is_file()
            and not target.is_symlink()
            and receipt["completed_groups"] <= latest["completed_groups"]
        ):
            candidates.append((receipt["completed_groups"], directory))
    candidates.sort(reverse=True)
    removed = []
    for _, directory in candidates[keep:]:
        if directory == newest.parent:
            raise ValueError("保留排序异常，拒绝删除刚发布的checkpoint")
        shutil.rmtree(directory)
        removed.append(str(directory))
    return removed


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


def contract_hash(contract):
    """对训练契约做稳定SHA摘要，用于显式批准来源/目标代码及拓扑组合。"""
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_resume_contract(source, target, migration=None):
    """默认要求完全相同；显式迁移需逐项核对卡数、实现与已授权协议差异。

    migration必须精确绑定来源与目标契约SHA。数据、算法配置、原始权重、处理器
    等其他字段始终严格相同，不能把迁移当作跳过校验的开关。显式online_execution_rebind
    仅接由原任务逐字段核验的新在线条件；任务ID/算法/源权重必须保持。
    协议例外分别为GS-10/未知QA和GS-02/06起点/分叉修复；其余优化/采样参数不变。
    """
    if migration is not None and migration.get("schema") == "physical-ai.gs-resume-chain.v1":
        # 最近完整存档可能早于两次已批准修复。按实际中间运行契约逐段复用已有校验，
        # 不重写checkpoint元数据、不把多段迁移变成宽泛的忽略字段开关。
        if set(migration) != {"schema", "intermediate_contract", "steps"}:
            raise ValueError("迁移链字段不完整或含未知字段")
        steps, middle = migration["steps"], migration["intermediate_contract"]
        if (
            not isinstance(middle, dict)
            or not isinstance(steps, list)
            or len(steps) != 2
            or any(not isinstance(step, dict) or "schema" in step for step in steps)
            or [step.get("protocol_revision") for step in steps]
            != ["gs-v1.1-unknown-qa-v1", "gs-v1.1-source-start-v1"]
        ):
            raise ValueError("只接受已授权的共同成功/未知QA→源起点两段迁移")
        validate_resume_contract(source, middle, steps[0])
        validate_resume_contract(middle, target, steps[1])
        return
    if source == target and migration is None:
        return
    expected = {
        "source_contract_sha256": contract_hash(source),
        "target_contract_sha256": contract_hash(target),
    }
    rebind = migration is not None and migration.get("online_execution_rebind") is True
    protocol = migration is not None and migration.get("protocol_revision") == "gs-v1.1-unknown-qa-v1"
    source_start = migration is not None and migration.get("protocol_revision") == "gs-v1.1-source-start-v1"
    if rebind:
        expected["online_execution_rebind"] = True
    if protocol:
        expected["protocol_revision"] = "gs-v1.1-unknown-qa-v1"
    if source_start:
        expected["protocol_revision"] = "gs-v1.1-source-start-v1"
    if migration != expected:
        raise ValueError("恢复配置/数据或显式迁移批准不一致")
    excluded = {"world_size", "implementation_files"}
    if protocol:
        # 用户已确认的两项语义迁移，仅这两个配置键可变化；其余采样/优化/预算仍逐值比较。
        old_config, new_config = dict(source["config"]), dict(target["config"])
        if (
            old_config.pop("unknown_qa_policy", "stop") not in {"stop", "retain_action_skip_qa_group"}
            or new_config.pop("unknown_qa_policy", None) != "retain_action_skip_qa_group"
            or old_config.pop("task_rule")
            not in [
                {"version": "libero-native-at-question-delivery-v1"},
                {"version": "libero-shared-at-question-delivery-v1"},
            ]
            or new_config.pop("task_rule") != {"version": "libero-shared-at-question-delivery-v1"}
            or old_config != new_config
            or target.get("task_success_implementation", {}).get("version") != "libero-task-success-v1"
            or not target["task_success_implementation"].get("files")
        ):
            raise ValueError("协议迁移只能启用共同成功判据与整组未知QA排除")
        excluded.update({"config", "task_success_implementation"})
    if source_start:
        # 本迁移仅修执行起点/观察/分叉；问题、算法、预算、奖励语义不能一起变化。
        old_success = source.get("task_success_implementation", {})
        new_success = target.get("task_success_implementation", {})
        if (
            not rebind
            or source["config"] != target["config"]
            or target.get("execution_boundary")
            != {
                "version": "rl-source-start-and-fork-v1",
                "wait_for_stability": False,
                "refresh": "libero-refresh-preserve-sampling-v1",
            }
            or old_success.get("version") != "libero-task-success-v1"
            or new_success.get("version") != old_success.get("version")
            or not old_success.get("files", {}).get("task-success-v1.json")
            or new_success.get("files", {}).get("task-success-v1.json") != old_success["files"]["task-success-v1.json"]
        ):
            raise ValueError("源起点迁移必须保持算法/任务判据，显式重绑定到无等待且完整分叉版本")
        excluded.update({"execution_boundary", "task_success_implementation"})
    if rebind:
        # train启动时已读取原任务表并逐字段核对只新增online_execution；此处绑定来源。
        proof = target.get("task_rebind", {})
        prior = source.get("task_rebind", {})
        same_origin = (
            prior.get("source_tasks_sha256") == proof.get("source_tasks_sha256")
            and prior.get("preserved_tasks_sha256") == proof.get("preserved_tasks_sha256")
            and prior.get("target_tasks_sha256") == source["tasks_sha256"]
        )
        if (
            (proof.get("source_tasks_sha256") != source["tasks_sha256"] and not same_origin)
            or proof.get("target_tasks_sha256") != target["tasks_sha256"]
            or not proof.get("preserved_tasks_sha256")
        ):
            raise ValueError("缺少原问题/初态不变的在线执行重绑定凭据")
        excluded.update({"tasks_sha256", "task_rebind"})
    if {k: v for k, v in source.items() if k not in excluded} != {
        k: v for k, v in target.items() if k not in excluded
    }:
        raise ValueError("迁移不能改变训练配置/数据/权重")
    if any(type(c.get("world_size")) is not int or c["world_size"] < 1 for c in (source, target)):
        raise ValueError("迁移卡数必须为正整数")


def load_boundary(manifest, contract, *, migration=None):
    """核验组边界checkpoint，再恢复；配置/数据/权重变化不能接着使用旧游标。

    默认严格一致；migration显式批准拓扑及代码指纹变更，不能改变算法/数据。
    只加载本项目创建、已校验SHA的训练文件；weights_only保证不执行外来pickle对象。
    manifest中的路径允许跨机器重定位，但对应字节身份必须保持。
    """
    receipt = json.loads(Path(manifest).read_text())
    validate_resume_contract(receipt["contract"], contract, migration)
    if content_hash(receipt["path"]) != receipt["sha256"]:
        raise ValueError("checkpoint SHA不一致")
    # 八rank读取同一完整存档时按需映射CPU storage，避免每rank再常驻一份完整副本。
    # 默认私有映射；恢复优化器不改写不可变checkpoint文件。
    state = torch.load(receipt["path"], map_location="cpu", weights_only=True, mmap=True)
    if state["schema"] != "physical-ai.seggrpo-training.v1" or state["contract"] != receipt["contract"]:
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
