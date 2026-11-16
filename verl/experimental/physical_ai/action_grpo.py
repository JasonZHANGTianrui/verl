"""动作阶段专用GRPO：复用原生闭环执行与整轨迹更新，在交付边界直接结束，不接QA。"""

from . import grpo


def validate_config(config):
    """校验动作专用配方；仅奖励目标成功，不允许Replay、QA组或阶段配对配置。"""
    if config.get("advantage_estimator") != "action_grpo":
        raise ValueError("需要显式action_grpo")
    if config.get("qa_steps") != 0 or config.get("qa_history_mode", "current") != "current":
        raise ValueError("动作专用训练必须关闭QA执行预算和离线经历")
    if any(k in config for k in ("qa_replay_release", "qa_replay_manifest_sha256", "qa_action_policy")):
        raise ValueError("动作专用训练不能携带Replay配置")
    # 通用GRPO验证仅在副本上执行；原配置/记录保留动作专用算法身份。
    grpo.validate_config(dict(config, advantage_estimator="grpo"))
    return dict(advantage_estimator="action_grpo", reward="task_only", group_size=config["group_size"])


def collect_group(policy, processor, config, output, *, policy_version, task=None, resources=None):
    """只执行原动作阶段，任务终点由原成功判据评分；不改变初态、预算或停止逻辑。

    公共训练器仍可保存带followup的原任务元数据，但运行请求的question强制为None，
    原轨迹采集器因此不拼接QA、不提取QA真值、不建立judge。动作调用和自主结束正文
    均是动作阶段的真实生成，参与同一任务奖励；图像/state/tool反馈继续mask。
    """
    from .grpo_rollout import collect_group as collect_trajectory

    validate_config(config)
    records, artifacts = collect_trajectory(
        policy, processor, dict(config, question=None, qa_steps=0), output,
        policy_version=policy_version, task=task, resources=resources,
    )
    for record, artifact in zip(records, artifacts, strict=True):
        if (artifact.get("qa_reward") is not None
                or artifact.get("qa_status") not in {"no_question", "incomplete_execution"}
                or record.reward != artifact["task_reward"]):
            raise ValueError("动作专用轨迹意外进入QA或混入QA奖励")
    return records, artifacts
