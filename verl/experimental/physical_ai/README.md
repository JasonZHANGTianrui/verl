# Physical AI 的 verl 接入

这是 `JasonZHANGTianrui/verl` fork 中的具身训练实现，包含 SegGRPO 分组、
原生 G0.5 采样与概率计算、工具循环，以及单组／多卡更新诊断。
已增加原生后端的连续训练与组边界恢复；完整 verl Ray/FSDP Trainer 仍待接入。

## 目录职责

| 模块 | 职责 |
| --- | --- |
| `seggrpo.py`、`distributed.py` | 两阶段优势、loss、全局分组及分片梯度归约 |
| `episode_agent_loop.py` | 沿官方工具循环维护整回合的有状态工具 |
| `sampling.py`、`batched_sampling.py`、`g05_rollout.py` | 单条参照、共享模型批量生成及实际动作执行 |
| `g05_logprobs.py`、`precision.py` | 动作／文字概率与FP32参考路径 |
| `native_update.py` | 原生模型的mini-batch更新；旧无优化器入口保留SGD诊断 |
| `pilot.py`、`distributed_update.py` | 真实4＋4小组、跨区采样附件核验及更新入口 |
| `train.py`、`training_state.py`、`training_metrics.py` | 连续训练、批次保存/恢复/保留策略及两阶段日志 |
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

当前使用FP32完整历史重算作为训练概率参照；生产采样可用已通过逐token核验的原生缓存批量路径。短片段概率对照不能替代完整真实
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
奖励实现。旧stage1使用FP32、持久SGD、rank0采集；新batch配置可按任务跨rank并行
采集，完整收齐后再切mini-batch，所有rank按响应分片重算策略梯度。它是fork中的原生PyTorch分布式接入，不是Ray/FSDP
或vLLM后端，采样吞吐尚未优化。第一阶段仅启用已绑定设备状态奖励的训练任务。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module verl.experimental.physical_ai.train \
  --bundle /path/to/bundle --checkpoint /path/to/step_1000.pt \
  --task-release /path/to/rl-tasks \
  --config training-pipeline/ours/rl/verl/experimental/physical_ai/configs/sft-task-stage1.json \
  --swanlab-key-file /path/to/private/key.txt --output /path/to/new-training-run
```

按save_freq个完整采样batch保存完整policy、优化器、训练游标、数据/配置身份及各rank随机状态。
正常结束或显式边界停止也保存；未配置save_freq的旧profile仍每批保存。
`--resume /path/to/checkpoints/group-N/checkpoint.json`配合新output目录恢复；改变任务、
配置或world size默认会拒绝。迁移可显式传`--resume-migration approved.json`，
其中`source_contract_sha256`/`target_contract_sha256`精确绑定两个契约；仅允许world size
及已审核代码指纹差异，其他字段仍完全一致。恢复完整模型、AdamW和游标，ZeRO按新卡数
重新分片；改变卡数时按seed+completed_groups*1000+rank重建随机流，不声称逐位复现。半组不能续接成旧物理经历，旧attempt目录保留，重新采样
使用新目录。checkpoint先原子落盘并hash，最后写manifest，不接受半份文件。

SwanLab走verl原生`Tracking`；密钥只从私有文件进入进程环境，不写配置和checkpoint。
上报完整生效配置及两阶段奖励、优势、有效组、策略loss、clip比例、策略变化、
真实token长度、梯度范数、学习率、显存和分阶段耗时，不上传私有轨迹或GT。`swanlab.json`给出云端入口，`progress.json`与逐group日志保留本地进度。
此profile未加reference KL；全部同分的组仍为零优势，不能据此重采制造奖励差异。

### Batch 128、mini-batch 32、两阶段n=8

配置为`configs/sft-task-b128-m32-n8.json`。每个采样batch包含8个任务，每任务8条
动作轨迹，加上从中均匀选一父经历后的8条QA；共128条阶段序列。每个mini-batch
包含2个完整任务，共32条序列。两卡各以micro-batch 1累积最多16次反向，每批1个
PPO epoch，最多4次更新。total_training_steps=100按采样batch计，共800个任务组、
12,800条阶段序列、最多400次更新；不是100个epoch。旧max_groups按任务记录计，
与新total_training_steps互斥，防止计数单位混用。

`batching.py`在更新前一次性固定各任务/阶段的优势和原始old log-prob，并给响应
身份加任务前缀。全部同版行为概率先核对，包括零优势组；后续mini-batch产生的
策略差异正常进入PPO loss，不再套用同版概率误差门槛，也不能重标old log-prob。

新配置用AdamW lr=1e-6、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，梯度裁剪1。
使用PyTorch ZeroRedundancyOptimizer分片AdamW状态（ZeRO-1），模型与梯度仍复制，
不冒称FSDP/ZeRO-3。更新前沿原全局梯度归约；保存前清梯度并由全部rank合并优化器
状态到rank0，随后保存原生完整权重/优化器/游标/RNG。单卡分片配置明确拒绝。

两个rank分别采完整任务，同一任务的环境/动作/QA仍在同一进程串行。本rank的
CUDA卡负责模型；EGL渲染设备独立使用egl_device_id（默认0），不能用CUDA local_rank
代替EGL序号。任何任务失败于资源、GT或接口时拒绝整个采样batch，不缩组。保存/停止
必须在完整采样batch边界。save_freq=10每10批保存，max_ckpt_to_keep=1只保留
本输出目录最新完整checkpoint；新权重、完整优化器和manifest落盘并更新latest后
才清理旧版本。源checkpoint、其他实验和GPU验收文件不清理。恢复需要新output目录，
外部恢复源不自动删除；断点前尚未保存的批次须重新采样。

仍只接入配置指定的已验证任务奖励，模板、相机、稳定等待和完整会话不因batch改变。

## 检查

在配置上述环境变量后，从Physical AI根目录执行：

```bash
python -m unittest discover \
  -s training-pipeline/ours/rl/tests/experimental/physical_ai -v
python -m unittest discover -s data-pipeline/ours/tests -p 'test_rl*.py' -v
```

fork测试核对概率、采样、分组及两进程梯度；数据侧测试负责转换、资源和切分；fork测试同时核对GT隔离、裁判错误及环境副本。真实仿真和真实模型诊断单独记录，不拿单元测试代替训练收益。

### 算法配置与观测口径

当前明确采用组内标准差归一化、动作与QA阶段等权相加、seq-mean-token-mean，
clip_ratio=0.2和verl vanilla实现的dual-clip界限clip_ratio_c=3。完整词表采样温度1，
不做top-k/top-p截断；rollout与actor分布相同，先整批检查old log-prob，再做四个mini更新。
参考模型KL与熵奖励关闭；学习率固定1e-6，无warmup，无critic。这些值显式上传
SwanLab；当前原生后端不支持的开关设成其他值会报错，不能只改配置制造已启用假象。
参考：[verl GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html)、
[官方配置](https://verl.readthedocs.io/en/latest/examples/config.html)。

日志横轴为采样batch数，每批仅写一个日志点；mini-batch指标汇总，梯度额外保留最大值与逐mini曲线，实际optimizer_steps单列。全零优势
mini不更新参数，但仍记录奖励与采样进度。rollout指标覆盖包括零优势组的整个batch；
actor的active_clip/ratio/old_policy_k3只统计真正反向的响应。old_policy_k3是相对
采样旧策略的诊断，不能当reference KL。old_sampled_nll也不是完整词表精确entropy。
当前不宣称有定期独立验证集测评；训练奖励不能替代benchmark成绩。

### 设备编号与采样失败

CUDA local_rank与MUJOCO_EGL_DEVICE_ID不是同一枚举。容器可暴露两张计算卡、但
EGL只提供一个渲染后端；配置中的egl_device_id独立选择渲染设备，两rank可共享
EGL设备0，不改变各自模型的CUDA卡。运行环境须核对实际GL_RENDERER；不能把
使用EGL自动等同于NVIDIA GPU渲染。

采样/元数据/状态同步走独立Gloo组；模型梯度及ZeRO参数同步仍走NCCL。采样错误
立即写fatal-rank-N.json并抛出，由torchrun终止其他worker，不先等待整批采完。
progress同时标failed，避免只有rank0写异常而丢失rank1的最初报错。错误路径不
等待正常集体销毁通信组；正常结束仍由所有rank按相同顺序清理。

### 组内批量生成与缓存

新profile使用sampling_mode=cached、rollout_batch_size=8。每卡只加载一份模型，
独立会话的生成请求在同一GPU batch中推进；旧recompute只作为逐条数值参照。
采样后仍对整批真实生成token核验old log-prob，不把新前向结果覆盖旧概率。

模型接口复用G0.5原生CachedSession/SparseKVCache和VLM batch forward。各分支
独立保留KV、卷积/递推状态、原始token、图像/state和RNG。不同长度缓存仅在完整
注意力的KV左侧补零并屏蔽，线性状态不消费padding；结束后恢复真实缓存长度，
包含末个EOS。提前结束的行退出batch，不再生成或推进该环境。

phase_steps是单条与批量路径共用的动作/工具/观察实现；run_phases只在生成边界
调度，GT/事件逐真实步记录。模拟器在owner线程交错step，GPU生成实际合批，
不将它描述成多进程物理并行。每个sim.render显式激活自己的EGL context，
避免多个环境交错时读到另一环境的framebuffer；QA分叉像素核验继续保留。

组内动作全部采完，按原规则均匀选一个父经历后才展开QA组；没有按成功筛选。
生成错误关闭全部回合资源并阻止整组更新。此后端不冒充vLLM/Ray/FSDP Trainer。

分叉复核先逐项要求积分状态、几何和相机变换相等；独立GL上下文的uint8 RGB
允许每通道一个量化级的舍入差，按same-state-uint8-render-lsb-v1记录max_delta
及changed_pixels，超过即拒绝。实际QA初始观察仍精确复制父回合原图，不能用
这份复核渲染替换输入。分叉回执写入qa-N-fork.json，只作私有诊断。

当前部署使用单机4卡L40S。每卡依次采两个任务组，每组最多8条生成请求合批，
全局采样batch仍是128条阶段序列；mini-batch32，两卡改四卡不改变任务/奖励权重。
模拟器逐步图提取和API judge仍有CPU/网络开销，不能将token生成提速直接等同于
整批训练提速。先过真实采样、整批概率核对和梯度更新，再报告实际吞吐。

在线QA裁判默认最多20次请求（含首次），失败间隔5/10/20/40秒后封顶60秒；重试同一问题、GT和回答，日志记录当前重试次数。有效0分或needs_review不继续请求；耗尽仍保留null并拒绝更新，不能把API故障记为能力失败。
