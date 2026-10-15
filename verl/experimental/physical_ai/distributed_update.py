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

"""torchrun小组更新验收：读取同一行为策略的完整真实采样组，不重新生成或修改奖励。"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from .workspace import configure_workspace

ROOT = configure_workspace()


def sha256(path):
    """分块读取文件，核验跨AFS复制后的原始字节身份，避免把路径相同当权重相同。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_group(source, manifest):
    """读取并核验完整动作/QA组及实际输入；失败判分不修改，缺字段直接报错。

    manifest由采集端打包时生成，绑定checkpoint字节和所有组附件。路径仅允许在
    source内，原始policy_version作为采集身份保留，搬家后不按新绝对路径冒造版本。
    """
    import torch

    from verl.experimental.physical_ai.seggrpo import Segment, validate_groups

    for relative, digest in manifest["files"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or sha256(path) != digest:
            raise ValueError("采样附件路径或SHA不符：" + relative)
    settings = json.loads((source / "settings.json").read_text())
    records = [Segment(**r) for r in json.loads((source / "groups.json").read_text())]
    validate_groups(records)
    artifacts = []
    required = {"settings.json", "groups.json"}
    for record in records:
        directory = source / record.rollout_id
        reward_name = "reward.json" if record.phase == "execution" else "judge.json"
        required.update(
            str(Path(record.rollout_id) / n) for n in ["inputs.pt", "phase.json", "tokens.json", reward_name]
        )
        if not required.issubset(manifest["files"]):
            raise ValueError("manifest缺少必须附件")
        phase = json.loads((directory / "phase.json").read_text())
        tokens = json.loads((directory / "tokens.json").read_text())
        stored = torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True)
        if stored["input_ids"][0].tolist() != tokens["ids"] or stored["attention_mask"][0].tolist() != tokens["types"]:
            raise ValueError("实际输入与token记录不同")
        if not len(tokens["ids"]) == len(phase["generated"]) == len(phase["old_logprobs"]):
            raise ValueError("概率、mask与token长度不符")
        scored = json.loads((directory / reward_name).read_text())
        reward = scored["task_reward"] if record.phase == "execution" else scored["score"]
        if reward != record.reward or record.policy_version != settings["policy_version"]:
            raise ValueError("奖励或行为策略身份不符")
        phase["inputs_path"] = str(directory / "inputs.pt")
        artifacts.append(phase)
    return records, artifacts, settings


def main():
    """每GPU一个进程，全局分组后逐响应分片；验收一次SGD，不保存新训练checkpoint。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, value.resolve())
    import torch
    import torch.distributed as dist

    from verl.experimental.physical_ai.online_reward import save_json

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    try:
        directory = args.output / f"rank-{rank}"
        directory.mkdir(parents=True, exist_ok=False)
        manifest = json.loads(args.manifest.read_text())
        verified = [None]
        if rank == 0:
            verified[0] = sha256(args.checkpoint) == manifest["checkpoint_sha256"]
        dist.broadcast_object_list(verified, src=0)
        if not verified[0]:
            raise ValueError("checkpoint字节与原采样策略不符，禁止更新")
        records, artifacts, settings = load_group(args.rollouts, manifest)
        # 各rank均读取完整小组；以内容hash对照，不依赖各节点的绝对资源路径。
        identities = [None] * dist.get_world_size()
        identity = hashlib.sha256(json.dumps([asdict(r) for r in records], sort_keys=True).encode()).hexdigest()
        dist.all_gather_object(identities, (identity, manifest))
        if any(item != identities[0] for item in identities):
            raise ValueError("rank读取了不同组或附件版本")
        save_json(directory / "progress.json", dict(phase="loading_model"))
        from runtime import load_policy

        from verl.experimental.physical_ai.native_update import update_once
        from verl.experimental.physical_ai.precision import stable_delta_precision

        os.chdir(ROOT / "training-pipeline/g05")
        policy, _, _ = load_policy(
            args.bundle,
            args.checkpoint,
            torch.device("cuda", local_rank),
            "torch",
            args.bundle / "hf_processor",
            "sdpa",
        )
        save_json(directory / "precision.json", stable_delta_precision(policy.model))
        if settings["config"]["compute_dtype"] != "float32":
            raise ValueError("当前跨区正确性验收仅接受已验证的FP32采样组")
        save_json(directory / "progress.json", dict(phase="update", rank=rank, world_size=dist.get_world_size()))
        result = update_once(policy.model, records, artifacts, directory, compute_dtype="float32", distributed=True)
        checks = [None] * dist.get_world_size()
        dist.all_gather_object(checks, result.get("representative"))
        if any(value != checks[0] for value in checks):
            raise ValueError("归约后各rank的代表参数更新不同")
        if rank == 0:
            save_json(
                args.output / "summary.json",
                dict(
                    world_size=dist.get_world_size(),
                    checkpoint_sha256=manifest["checkpoint_sha256"],
                    rank_parameter_agreement=True,
                    scope="native_g05_distributed_verl_loss_diagnostic_not_full_trainer",
                    update=result,
                ),
            )
        save_json(directory / "progress.json", dict(phase="complete", optimizer_step=result["optimizer_step"]))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
