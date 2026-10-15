# Physical AI 的 verl 接入

这是 `JasonZHANGTianrui/verl` fork 中的具身训练实现，包含 SegGRPO 分组、
原生 G0.5 采样与概率计算、工具循环，以及单组／多卡更新诊断。
已增加原生后端的连续训练与组边界恢复；完整 verl Ray/FSDP Trainer 仍待接入。

## 目录职责

| 模块 | 职责 |
| --- | --- |
| `seggrpo.py`、`distributed.py` | 两阶段优势、loss、全局分组及分片梯度归约 |
| `episode_agent_loop.py` | 沿官方工具循环维护整回合的有状态工具 |
| `sampling.py`、`g05_rollout.py` | 原生token采样、完整前缀和实际动作执行 |
| `g05_logprobs.py`、`precision.py` | 动作／文字概率与FP32参考路径 |
| `native_update.py` | 原生模型的单组SGD更新诊断 |
| `pilot.py`、`distributed_update.py` | 真实4＋4小组、跨区采样附件核验及更新入口 |
| `train.py`、`training_state.py` | 原生后端连续训练、全局梯度、组边界保存/恢复及verl日志 |
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

## 使用SFT派生任务做一次真实更新

`pilot`现可通过`--task-release`和`--task-id`读取数据侧导出的任务，恢复源XML与
states[0]，逐控制步复用完整SceneCapture，再沿同一原生会话交付原题。未来问题、
旧SFT回答/think/动作与成功标签都不作为初始输入；新GT从本次执行提取。

```bash
python -m verl.experimental.physical_ai.pilot \
  --bundle /path/to/g05-bundle --checkpoint /path/to/step_1000.pt \
  --case training-pipeline/ours/rl/verl/experimental/physical_ai/configs/sft-task-pilot.json \
  --task-release /path/to/exported-rl-tasks --task-id 'libero-rl/<id>' \
  --output /path/to/new-run --update --save-checkpoint
```

首版配置只绑定“打开中间抽屉”的已校准设备状态奖励；它与QA所问内容分开计算，
不是把QA答案或原生success当动作奖励。换其他任务须另行绑定规则，不能把这一个
case已接通解释成整个2000条已可训练。任务表的pending字段继续保留。

小试固定4条动作＋从中均匀选一父轨迹的4条QA，同一行为权重全部采完后才更新。
动作/QA预算分别128/32个20Hz控制步，各6144个新token；全图/动作/文字历史保留。
API错误或未知GT阻止整组更新；全同分组优势为0，不重采制造分差。

`--save-checkpoint`只在实际更新后保存外层policy完整原生state_dict到独立
`checkpoints/step_1.pt`，兼容原加载接口。该文件明确标记为单步SGD诊断，未保存
生产优化器/调度器/采样器恢复状态；不冒称完整verl Trainer checkpoint。

## 连续训练与SwanLab

`train`复用`pilot.collect_group`和`native_update.update_once`，没有另一份模型生成或
奖励实现。当前profile使用FP32、持久SGD、归约后全局梯度裁剪；采集在rank0串行完成，
所有rank按响应分片重算策略梯度。它是fork中的原生PyTorch分布式接入，不是Ray/FSDP
或vLLM后端，采样吞吐尚未优化。第一阶段仅启用已绑定设备状态奖励的训练任务。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module verl.experimental.physical_ai.train \
  --bundle /path/to/bundle --checkpoint /path/to/step_1000.pt \
  --task-release /path/to/rl-tasks \
  --config training-pipeline/ours/rl/verl/experimental/physical_ai/configs/sft-task-stage1.json \
  --swanlab-key-file /path/to/private/key.txt --output /path/to/new-training-run
```

每组完成后保存完整policy、优化器、训练游标、数据/配置身份及各rank随机状态。
`--resume /path/to/checkpoints/group-N/checkpoint.json`配合新output目录恢复；改变任务、
配置或world size会拒绝。半组不能续接成旧物理经历，旧attempt目录保留，重新采样
使用新目录。checkpoint先原子落盘并hash，最后写manifest，不接受半份文件。

SwanLab走verl原生`Tracking`；密钥只从私有文件进入进程环境，不写配置和checkpoint。
只上报动作/QA奖励、有效更新数、零方差组、概率误差、梯度范数和耗时，不上传私有
轨迹或GT。`swanlab.json`给出云端入口，`progress.json`与逐group日志保留本地进度。
此profile未加reference KL；全部同分的组仍为零优势，不能据此重采制造奖励差异。

## 检查

在配置上述环境变量后，从Physical AI根目录执行：

```bash
python -m unittest discover \
  -s training-pipeline/ours/rl/tests/experimental/physical_ai -v
python -m unittest discover -s data-pipeline/ours/tests -p 'test_rl*.py' -v
```

fork测试核对概率、采样、分组及两进程梯度；数据侧测试负责转换、资源和切分；fork测试同时核对GT隔离、裁判错误及环境副本。真实仿真和真实模型诊断单独记录，不拿单元测试代替训练收益。
