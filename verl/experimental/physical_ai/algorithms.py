"""显式算法装配；默认SegGRPO，普通GRPO独立实现，训练/概率/优化器基础设施共享。"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Algorithm:
    """训练器需要的算法接口；width为每任务产生的完整记录数，不是token或环境步数。"""

    name: str
    width: int
    validate_config: Callable
    collect_group: Callable
    namespace_group: Callable
    task_groups: Callable
    minibatch_indices: Callable
    prepare_batch: Callable
    rollout_metrics: Callable


def resolve_algorithm(config):
    """按显式配置选算法，未知名称拒绝；不修改配置或替换其他模块的全局函数。"""
    name = config.get("advantage_estimator", "seggrpo")
    if name == "seggrpo":
        from . import batching, training_metrics
        from .pilot import collect_group
        from .training_state import validate_fixed_algorithm

        if min(config["execution_group_size"], config["qa_group_size"]) < 2:
            raise ValueError("两组均至少两条")
        return Algorithm(
            name,
            config["execution_group_size"] + config["qa_group_size"],
            validate_fixed_algorithm,
            collect_group,
            batching.namespace_group,
            batching.task_groups,
            batching.minibatch_indices,
            batching.prepare_batch,
            training_metrics.rollout_metrics,
        )
    if name == "grpo":
        from . import grpo
        from .grpo_rollout import collect_group

        grpo.validate_config(config)
        return Algorithm(
            name,
            config["group_size"],
            grpo.validate_config,
            collect_group,
            grpo.namespace_group,
            grpo.task_groups,
            grpo.minibatch_indices,
            grpo.prepare_batch,
            grpo.rollout_metrics,
        )
    raise ValueError("未知训练算法：" + str(name))
