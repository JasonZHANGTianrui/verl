"""训练任务的固定调度表：类别配额只控制抽题，不进入奖励、优势或loss。"""

import hashlib
import json
import random
from collections import Counter


def goal_count(task):
    """按去重叶目标计数；谓词名不区分大小写，实体ID保留原大小写，不读指令/成绩。"""
    goals = task["objective"]["goals"]
    if not isinstance(goals, list) or not goals:
        raise ValueError("缺少机器目标")
    normalized = set()
    for goal in goals:
        if not isinstance(goal, list) or not goal or any(not isinstance(v, str) for v in goal):
            raise ValueError("需要已经展开的叶目标，不能猜And树或自然语言")
        if goal[0].lower() == "and":
            raise ValueError("目标未展开")
        normalized.add((goal[0].lower(), *goal[1:]))
    count = len(normalized)
    if count not in (1, 2, 3):
        raise ValueError("当前调度只支持1/2/3叶目标")
    return count


def build_schedule(tasks, total, seed, spec=None):
    """返回固定任务ID序列及对账；未配置时沿旧eligible顺序轮询，不能改变历史运行。

    fixed-goal-quota-v1按预先指定的整数总配额造类别槽，再类内洗牌轮询。每个类的
    任务使用次数相差至多1；不足一轮时不宣称全覆盖。随机流局部独立，不消耗模型RNG。
    输入任务不可修改；恢复时训练契约对比完整表hash和ID，拒绝不同表沿旧游标继续。
    """
    if type(total) is not int or total < 1 or len({t["id"] for t in tasks}) != len(tasks) or not tasks:
        raise ValueError("任务/训练抽取数非法")
    if spec is None:
        return None
    if spec.get("version") != "fixed-goal-quota-v1":
        raise ValueError("未知任务调度")
    counts = {int(k): v for k, v in spec["counts"].items()}
    if (
        set(counts) != {1, 2, 3}
        or any(type(v) is not int or v < 1 for v in counts.values())
        or sum(counts.values()) != total
    ):
        raise ValueError("配额必须覆盖1/2/3目标并等于总任务抽取数")
    pools = {k: sorted(t["id"] for t in tasks if goal_count(t) == k) for k in counts}
    if any(not ids for ids in pools.values()):
        raise ValueError("类别没有任务")
    rng = random.Random(seed)
    slots = [k for k, n in counts.items() for _ in range(n)]
    rng.shuffle(slots)
    cycles = {k: [] for k in pools}
    schedule = []
    by_id = {t["id"]: t for t in tasks}
    for k in slots:
        if not cycles[k]:
            cycles[k] = pools[k].copy()
            rng.shuffle(cycles[k])
        schedule.append(cycles[k].pop())
    report = dict(
        version=spec["version"],
        seed=seed,
        task_ids=schedule,
        goal_counts=slots,
        pool_counts={str(k): len(v) for k, v in pools.items()},
        draw_counts=dict(Counter(map(str, slots))),
        per_task=dict(Counter(schedule)),
        qa_counts=dict(
            Counter(
                by_id[i]["followup"].get("selection_class", by_id[i]["followup"]["question_spec"]["strategy_id"])
                for i in schedule
            )
        ),
    )
    report["sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return report
