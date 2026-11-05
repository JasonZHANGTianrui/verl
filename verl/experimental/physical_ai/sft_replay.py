"""固定SFT经历上的离线QA：不创建物理环境、不执行模型工具，只对当前回答计奖。"""

import hashlib
import json
from pathlib import Path

import torch

from .history_replay import digest


def load_index(root, expected_sha):
    """验证冻结QA索引身份；固定来源独立于当前采样轨迹，不能恢复时换池。"""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "physical-ai.sft-qa-replay.v1" or manifest["qa_action_policy"] != "ignore":
        raise ValueError("未知Replay接口")
    if digest(root / "manifest.json") != expected_sha or digest(root / "index.jsonl") != manifest["index_sha256"]:
        raise ValueError("Replay发布身份改变")
    rows = [json.loads(x) for x in (root / "index.jsonl").read_text().splitlines() if x.strip()]
    entries = {r["task_id"]: r for r in rows}
    if len(entries) != len(rows) or len(rows) != manifest["entries"]:
        raise ValueError("Replay索引不完整或重复")
    return entries


def prefix_inputs(inputs, audit, entry):
    """只保留最终答案正文前的真实token，丢弃labels；全部既往图片/state保持原值。

    answer_cut是编译器逐消息审计给出的原token边界，不从标签mask或答案字符串猜测。
    要求答案后无新附件，且答案是最后一个assistant；检查由数据导出和此处共同完成。
    返回不含labels的CPU输入，不修改原缓存。
    """
    cut = entry["answer_cut"]
    if audit["sample_id"] != entry["sft_id"] or inputs["input_ids"].shape != (1, entry["input_tokens"]):
        raise ValueError("Replay缓存身份或长度不符")
    if type(cut) is not int or not 0 < cut < entry["input_tokens"]:
        raise ValueError("Replay答案边界无效")
    if any(s["attachment"] is not None for s in audit["segments"] if s["end"] > cut):
        raise ValueError("答案后存在新模态，拒绝未来泄漏")
    answers = [s for s in audit["segments"] if s["message_id"] == "answer"]
    if not answers or min(s["start"] for s in answers) != cut:
        raise ValueError("答案截断不等于真实assistant正文边界")
    if len(inputs["pixel_values"]) != entry["images"] or inputs["proprio"].shape[1] != entry["states"]:
        raise ValueError("Replay模态附件数量不符")
    return dict(
        input_ids=inputs["input_ids"][:, :cut].clone(),
        attention_mask=inputs["attention_mask"][:, :cut].clone(),
        pixel_values=inputs["pixel_values"],
        proprio=inputs["proprio"],
    )


def load_parent(policy, processor, config, task):
    """加载同来源固定SFT前缀与私有绑定；旧QA正文/think/GT绝不塞入LiveChat。"""
    from benchmark.core.scene_graph import canonical

    from .g05_rollout import chat_fingerprint
    from .native_runtime import live_chat_class

    entries = load_index(config["qa_replay_release"], config["qa_replay_manifest_sha256"])
    entry = entries[task["id"]]
    origin = task["provenance"]["source"]
    if entry["source"] != origin or entry["task_instruction"] != task["instruction"]:
        raise ValueError("Replay经历与在线任务来源错配")
    for value in entry["files"].values():
        if digest(value["path"]) != value["sha256"]:
            raise ValueError("Replay来源文件被修改：" + value["path"])
    cache = torch.load(entry["files"]["inputs"]["path"], map_location="cpu", weights_only=True)
    audit = json.loads(Path(entry["files"]["compiled_audit"]["path"]).read_text())
    inputs = prefix_inputs(cache, audit, entry)
    del cache
    chat = live_chat_class()(policy, processor)
    chat.ids = inputs["input_ids"][0].tolist()
    chat.types = inputs["attention_mask"][0].tolist()
    chat.pixels = inputs["pixel_values"]
    chat.states = [v.clone() for v in inputs["proprio"][0].split(1, dim=0)]
    chat.instruction = task["instruction"]
    chat.rl_processor, chat.rl_sampling_mode = processor, config["sampling_mode"]
    chat.rl_compute_dtype = config["compute_dtype"]
    # 由原消息边界截断，而不是在token序列中搜索自然语言答案。
    text = chat.tokenizer.decode(chat.ids[-12:], skip_special_tokens=False)
    if not text.endswith("<|im_start|>assistant\n"):
        raise ValueError("Replay截断未落在原生assistant正文开始处：" + repr(text))
    parent_id = "sft/" + entry["sft_id"]
    binding = dict(
        rollout_id=parent_id, question=entry["question"], gt=entry["gt"], task_instruction=task["instruction"]
    )
    binding["binding_sha256"] = hashlib.sha256(canonical(binding).encode()).hexdigest()
    parent = dict(
        kind="fixed_sft",
        rollout_id=parent_id,
        case_id=task["id"],
        manifest_sha256=config["qa_replay_manifest_sha256"],
        prefix_sha256=chat_fingerprint(chat),
        source_record_sha256=entry["files"]["record"]["sha256"],
    )
    return chat, binding, parent


def answer_steps(chat, config, directory, *, seed):
    """采样一个assistant回复；保留所有原token/概率，工具仅解析记录、从不执行。

    正文交公共judge；动作代码不计为文字答案，不额外施加格式/长度奖励。生成到正常
    结束或token/context上限即交付，超时仍按原基础设施异常处理，不伪造EOS或工具反馈。
    本生成器不接session参数，因此不存在误用当前在线动作场景执行工具的通道。
    """
    from runtime import to_device

    from .native_runtime import cached_session
    from .online_reward import public_text, save_json
    from .tool_messages import message_requests

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    inputs = to_device(chat.inputs(), torch.device("cuda"))
    start = len(chat.ids)
    capacity = int(chat.policy.model.cfg.vlm.max_position_embeddings) - start
    if capacity < 1:
        raise ValueError("固定SFT前缀无QA生成容量")
    response = yield dict(
        engine=cached_session(chat.policy.model),
        inputs=inputs,
        max_new_tokens=min(config["qa_token_budget"], capacity),
        stop_ids=[chat.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0], chat.policy.model.cfg.eos_token_id],
        temperature=1.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        mode=config["sampling_mode"],
        compute_dtype=config["compute_dtype"],
        timeout_seconds=config.get("generation_timeout_seconds", 180),
    )
    if response["termination"] not in {"message_end", "token_budget", "context_capacity"} or not response["ids"]:
        raise RuntimeError("离线QA采样未正常交付：" + response["termination"])
    ids = response["ids"]
    text = chat.tokenizer.decode(ids, skip_special_tokens=False)
    requests, ordinary, rejected = message_requests(text, ids, chat.tokenizer, chat.policy.action_tokenizer)
    chat.append_generated(ids)
    artifact = dict(
        phase="qa",
        prefix_length=start,
        end_length=len(chat.ids),
        generated=[False] * start + [True] * len(ids),
        old_logprobs=[0.0] * start + response["logprobs"],
        termination=response["termination"],
        pending_assistant=False,
        answer=public_text(ordinary),
        env_steps=0,
        generated_tokens=len(ids),
        inputs_path=str(directory / "inputs.pt"),
        ignored_tool_calls=len(requests),
        qa_action_policy="ignore",
    )
    save_json(
        directory / "call-000.json", dict(text=text, ids=ids, termination=response["termination"], input_tokens=start)
    )
    save_json(directory / "tool-requests.json", dict(accepted=requests, rejected=rejected, executed=0))
    save_json(directory / "phase.json", artifact)
    save_json(directory / "tokens.json", dict(ids=chat.ids, types=chat.types))
    torch.save(chat.inputs(), directory / "inputs.pt")
    return artifact


def collect_qa(policy, processor, config, output, *, policy_version, task, drive):
    """同一冻结SFT前缀生成完整8条QA；动作组身份只用于计算配对，不冒充经历来源。"""
    import os

    from benchmark.scoring.llm_judge import JudgeClient

    from .g05_rollout import chat_fingerprint
    from .online_reward import qa_reward_status, save_json, score_answer
    from .sampling import fork_chat
    from .seggrpo import Segment

    chat, binding, history = load_parent(policy, processor, config, task)
    status = qa_reward_status(binding, config.get("unknown_qa_policy", "stop"))
    condition = hashlib.sha256((history["prefix_sha256"] + binding["binding_sha256"]).encode()).hexdigest()
    save_json(output / "selection.json", dict(history_mode="fixed_sft", history_parent=history, binding=binding))
    client = JudgeClient(Path(os.environ["PHYSICAL_AI_JUDGE_ACCESS_FILE"]), model=config["judge_model"], timeout_s=90)
    finished = 0

    def branch(i):
        """每个回答独立随机流/缓存，私有GT只进入judge，不变更原前缀。"""
        nonlocal finished
        answer = yield from answer_steps(fork_chat(chat), config, output / f"qa-{i}", seed=config["seed"] + 100 + i)
        judgment = score_answer(binding, answer["answer"], client, rollout_id=history["rollout_id"])
        save_json(output / f"qa-{i}/judge.json", judgment)
        if status == "scored" and judgment["status"] != "scored":
            raise ValueError("Replay judge无有效分，不缩组/填零")
        if status != "scored" and (judgment["status"] != "excluded_gt" or judgment["score"] is not None):
            raise ValueError("未知GT不得产生奖励")
        row = Segment(
            f"qa-{i}",
            "qa",
            "qa-group",
            condition,
            task["id"],
            policy_version,
            config["qa_group_size"],
            answer["prefix_length"] - 1,
            answer["end_length"] - 1,
            float(judgment["score"]) if status == "scored" else None,
            status,
            parent_rollout_id=history["rollout_id"],
            history_parent=history,
            paired_execution_group_id="execution-group",
        )
        finished += 1
        save_json(output / "progress.json", dict(phase="qa", finished=finished, planned=config["qa_group_size"]))
        if chat_fingerprint(chat) != history["prefix_sha256"]:
            raise ValueError("QA生成修改了冻结父前缀")
        return row, answer

    results = drive(branch(i) for i in range(config["qa_group_size"]))
    return [x[0] for x in results], [x[1] for x in results]
