# Physical AI 的 verl 接入

这是 `JasonZHANGTianrui/verl` fork 中的具身训练实现，包含 SegGRPO 分组、
原生 G0.5 采样与概率计算、工具循环，以及单组／多卡更新诊断。
完整 verl Ray/FSDP Trainer 和持续训练循环仍待接入。

## 目录职责

| 模块 | 职责 |
| --- | --- |
| `seggrpo.py`、`distributed.py` | 两阶段优势、loss、全局分组及分片梯度归约 |
| `episode_agent_loop.py` | 沿官方工具循环维护整回合的有状态工具 |
| `sampling.py`、`g05_rollout.py` | 原生token采样、完整前缀和实际动作执行 |
| `g05_logprobs.py`、`precision.py` | 动作／文字概率与FP32参考路径 |
| `native_update.py` | 原生模型的单组SGD更新诊断 |
| `pilot.py`、`distributed_update.py` | 真实4＋4小组、跨区采样附件核验及更新入口 |
| `workspace.py` | 定位独立的Physical AI工作区，不靠submodule目录层数 |
| `configs/` | 本接入的loop、奖励契约和开发case配置 |

数据转换、资源解析和初态文件读取由Physical AI项目的
`data-pipeline/ours/rl/`维护。环境会话`libero_session.py`、在线评分`online_reward.py`和训练初始化`task_data.py`保留在本fork，调用项目已有Scene map/GT实现。
G0.5原生加载、编译和工具协议继续复用该项目的`training-pipeline/ours/g05/`
及`data-pipeline/rollout/g05/`。EO-1／G0.5官方源码不修改。

## 工作区和运行

fork可以独立clone。集成运行通过`PHYSICAL_AI_ROOT`指定包含数据管线、G0.5适配器
和固定G0.5 submodule的项目根目录。作为项目submodule使用时可自动定位；
显式路径无效会报错，不默默连接另一份工作区。纯SegGRPO算法不要求这个外部项目。

沿用项目已验证的独立Conda环境；普通依赖走清华镜像，不升级其他训练或生产环境。
增量依赖由项目的`training-pipeline/ours/requirements/verl/`维护。
在Physical AI根目录运行：

```bash
export PHYSICAL_AI_ROOT="$PWD"
export PYTHONPATH="$PWD/training-pipeline/ours/rl:$PWD/data-pipeline"
export PHYSICAL_AI_JUDGE_ACCESS_FILE=/path/to/private/access.json
python -m verl.experimental.physical_ai.pilot \
  --bundle /path/to/g05-bundle --checkpoint /path/to/checkpoint.pt \
  --case training-pipeline/ours/rl/verl/experimental/physical_ai/configs/pilot-case.json \
  --output /path/to/new-run --update

# 复用同一行为策略的真实采样组，验收双卡更新。
torchrun --standalone --nproc_per_node=2 --module verl.experimental.physical_ai.distributed_update \
  --bundle /path/to/g05-bundle --checkpoint /path/to/checkpoint.pt \
  --rollouts /path/to/rollouts --manifest /path/to/rollouts-manifest.json \
  --output /path/to/new-update
```

配置中的 `_target_` 使用 `verl.experimental.physical_ai`，不再使用旧的 `rl` 包名。
迁移只改变归属、导入和路径定位，不改变模型输入、采样、奖励、优势、mask或更新公式。

## 验收边界

当前使用完整历史重算与FP32作为正确性参考。短片段概率对照不能替代完整真实
轨迹核对；混合精度/KV缓存后端要另验收，不放宽阈值或重写旧采样概率。
动作和QA组分别归一化，均匀选父轨迹，不按成功筛选；QA父前缀不重复计直接loss。
全同分组优势为零；缺分、API错误或未知GT不能填成0并缩组训练。

`--update`执行一次内存内无动量SGD诊断，不覆盖源权重，没有生产优化器状态或训练
checkpoint恢复。数据任务表的动作奖励绑定、生产loop读取与连续训练仍需继续接入。

## 检查

在配置上述环境变量后，从Physical AI根目录执行：

```bash
python -m unittest discover \
  -s training-pipeline/ours/rl/tests/experimental/physical_ai -v
python -m unittest discover -s data-pipeline/ours/tests -p 'test_rl*.py' -v
```

fork测试核对概率、采样、分组及两进程梯度；数据侧测试负责转换、资源和切分；fork测试同时核对GT隔离、裁判错误及环境副本。真实仿真和真实模型诊断单独记录，不拿单元测试代替训练收益。
