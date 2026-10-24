# Physical AI 的 verl 接入

2026-10-18 GS v1.2接入：新session默认共用`libero_online.prepare_online_scene`，
固定10步开爪后才建立任务历史/首模型输入；示范0.25秒同样执行至约0.75秒。
任务发布由数据侧binder绑定真实准备版本与实现hash，不从时间戳推测已预热。
QA分叉使用只恢复容器的路径，继承父状态，不重复dummy。迁移需显式
`gs-v1.2-dummy10-v1`与新任务包；机械验收不替代真实策略小批，不热改在途运行。

以下v1.1无等待说明仅供旧版本追溯，不能覆盖上述新默认。

2026-10-18执行对齐：新在线任务必须显式声明`wait_for_stability`，默认false，
恢复源状态且不重设控制目标；旧包需用数据侧`ours.rl.bind_online`重新绑定，不改旧包。
QA环境分叉保留传感器相位/读数缓存及成功monitor历史，核验刷新不改变父分支。
任务包记录初始化选择/实现hash，训练契约记录`execution_boundary`；旧v5 run保持冻结。
接续旧checkpoint仅允许显式`gs-v1.1-source-start-v1`迁移，保持题目、源初态、算法、
预算和成功登记不变。GS-05的SFT尾部100步不自动进入在线任务。

2026-10-18后续：新配置使用GS-10共同成功monitor，并启用
`unknown_qa_policy=retain_action_skip_qa_group`。旧冻结profile保留旧行为；以下历史
“原生成功／任意unknown停止”的说明不覆盖本修订。标准入口为主仓
`docs/specs/v0.1/sft-execution-standard.md`的GS-07/08/09/10。

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

## 普通 GRPO baseline

`advantage_estimator=grpo`使用独立`grpo.py`和`grpo_rollout.py`，入口仍是`train`。
同任务采n条独立完整动作→QA，每条沿自身原生会话和真实末态提问，不选共同父经历。
总回报为0.5×task＋0.5×QA；未知GT或不可交付QA时直接结束，仅用task回报。
已交付的QA答错/未答计0；API/仿真/数据错误不转换为缺QA。仅token/context预算
截断可保留原token结束轨迹，不造EOS、不执行半条工具消息；SegGRPO默认仍拒绝该截断。

整条会话的动作和语言使用同一个组内标准化优势，PPO按每条真实生成token均值、
轨迹均值计算。用户/环境输入不训练。`algorithms.py`显式装配采集/分组/优势/指标；
默认SegGRPO仍调用原`pilot`、`batching`和`seggrpo`函数，未放宽其二元奖励/父组约束。
公共更新器只接收分组校验策略；没有模块猴子补丁或另一份优化器实现。

配方`configs/libero2000-grpo-b64-m16-n8.json`是64完整会话/mini16、n8，每批8任务、
4次mini更新，与SegGRPO128阶段序列/mini32匹配任务与动作采样量。总100批；
每20批保留纯模型快照，完整优化器仅latest。两算法的batch单位不同，报告时明确。
普通算法记录task、QA和总reward及task-only原因；不会把缺QA计成QA答错。

## 连续训练与SwanLab

`retain_model_snapshots=true`将每个保存点的纯模型另存到
`model-checkpoints/batch-N/model.pt`并保留，附带配置与SHA凭据。
`save_freq`仍按采样batch计；例如20对应20/40/60/80/100批。
`max_ckpt_to_keep=1`只清理`checkpoints/`中的旧完整恢复存档，模型快照不受影响。
完整恢复继续从`latest.json`读取模型、优化器及RNG；纯模型快照只用于推理/测评。
新存档及模型快照均成功发布后才清理旧恢复点；默认false保留旧行为。

生成超时的恢复可用`generation-timeout-increase-v1`迁移：仅允许将
`generation_timeout_seconds`增大为有限正数，其他配置、任务、模型、拓扑及执行
边界保持不变。源/目标契约SHA仍须精确绑定；从完整checkpoint恢复模型、优化器
和RNG到新输出目录。未保存的批次重跑，旧记录保留。超时的半条消息不当EOS、
不执行其工具、不填奖励，token/动作预算也不因等待时间增长而放宽。

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

多设备渲染可在部署核验后配置`egl_device_ids`列表，按本节点local_rank分配；
列表必须覆盖全部本地worker，不从CUDA序号自动推测。设置
`PHYSICAL_AI_REQUIRE_GL_VENDOR='NVIDIA Corporation'`可拒绝意外回退Mesa软件渲染。
每个阶段的`phase.json`记录实际vendor/renderer/EGL序号，不能只以nvidia-smi有卡
认定相机用了GPU。缺少NVIDIA图形库的容器可单独配置与内核驱动同版本的用户态
EGL库，通过任务专用`LD_LIBRARY_PATH`和`__EGL_VENDOR_LIBRARY_FILENAMES`选择；
资源版本/hash写部署记录，不覆盖系统库或改变其他任务环境。

渲染后端变化须核对同动作的物理状态、逐步Scene map及实际画面；像素差异独立记录，
不能承诺跨Mesa/NVIDIA逐像素相同。相机、光照、采样频率和模型输入时刻保持，
每控制步照常记录图。SwanLab新增环境step、SceneCapture与视频写入的阶段均值/最大
耗时（秒）；这些是逐轨迹累计，不是多卡墙钟时间，不可直接相加当整批耗时。

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

LIBERO炉灶的运行状态还包含`vis_site_names`标志与`MjModel.site_rgba`透明度，
它们不在`MjData`内。分叉先通过`copy_visual_runtime`按上游已登记设备site复制
这两项，再严格比较完整模型XML；RGB、几何、质量及未登记site差异仍拒绝。
不调用设备谓词重算显示、不推进父环境，也不把上游透明度裁成另一数值。
随后继续原有物理/相机/图像核验；记录`libero-registered-visual-state-v1`。

当前部署使用单机4卡L40S。每卡依次采两个任务组，每组最多8条生成请求合批，
全局采样batch仍是128条阶段序列；mini-batch32，两卡改四卡不改变任务/奖励权重。
模拟器逐步图提取和API judge仍有CPU/网络开销，不能将token生成提速直接等同于
整批训练提速。先过真实采样、整批概率核对和梯度更新，再报告实际吞吐。

在线QA裁判默认最多20次请求（含首次），失败间隔5/10/20/40秒后封顶60秒；重试同一问题、GT和回答，日志记录当前重试次数。有效0分或needs_review不继续请求；耗尽仍保留null并拒绝更新，不能把API故障记为能力失败。

### 公共执行接口迁移

训练示范初态直接复用`ours.sft.sources.libero_source.open_environment(seed=...)`，
统一v5准备返回首观察；不在之后重复刷新控制器。普通初始化仍从make_env进入。
两处在线保持均调用`benchmark.adapters.libero_execution.hold_control_targets`，
保持手臂及夹爪末目标、正常物理积分。每个任务保存preparation及在线执行身份。

旧任务包不能直接套新版环境。用`ours.rl.prepare_online`实际准备新任务包，
保留旧来源字段并追加online_execution；运行时核对其实现/关系身份和完整准备状态。
恢复时若批准文件显式设置`online_execution_rebind: true`，启动器先逐字段确认
新旧任务只有online_execution差异，再允许任务文件hash变化。算法、权重、任务ID
及顺序保持原值。这是新执行条件下的训练分支，不能冒充旧环境逐位续跑。
Mem仍未作为训练任务接入；公共监测器和clone支持不等于其奖励/会话已经验收。

### 完整普通LIBERO任务池

`configs/libero2000-b128-m32-n8.json`显式选择
`libero-native-at-question-delivery-v1`，与当前G0.5测评动作口径一致：用户QA交付前读取
本次环境成功状态；不以native success提前结束、不补动作，QA后的动作不能反改此分。
模型生成的非法工具指令按执行失败记录，环境/接口异常不转成能力零分。

`task_data.training_tasks`在此模式保留整个训练任务池，逐题校验机械查询注册及参数。
`online_reward.bind_delivery`原本已复用测评相同的注册表、PopContext与judge，沿用即可；
不再添加第二套空间判据。旧设备状态小试按原profile继续可复现，不热改暂停运行。

动作预算沿当前题库生成器：普通suite为300控制步，libero_10/90为520步；QA为32步。
组大小n8+n8、batch128/mini32、100采样batch、save10/keep1保持。2000是任务池大小，
100个batch实际选800个任务起点；新池需要新运行，不能沿旧50条任务池的恢复游标。
新任务发布与真实采样验收完成前，不自动恢复训练。

在线任务v2不要求每个初态提前跑一次。固定题目和每任务场景配置直接复用；
`open_task_session`按发布契约核验真实恢复，`create_scene_capture`核对当前配置及
准备后的完整状态，同组分支通过`assert_same_preparation`核对完整起点/控制状态。
旧v1仍要求其预演回执，不能用缺字段的旧包冒充新模式。

`thinking`须按checkpoint训练/测评模式显式选择；默认false保持旧模型兼容。
with-think模型设true，只将新QA前缀置为开放的`<think>`，历史reasoning/动作/图像不删除。
动作阶段保留原生裸assistant头，可自行生成思考；QA输出仅公开答案进入judge。
LiveChat实现SHA写入运行契约，不能只换权重而继续沿用不匹配的聊天前缀。

### 固定v1交接与ReAct退出

正式v1配置需设置`PHYSICAL_AI_G05_HANDOFF`，指向d0bdbf2交接副本。
`native_runtime`核验loader/LiveChat/renderer/cache/codec观察入口SHA，模型模块来自交接，
环境/机械查询来自当前v5工作区；不能靠同名模块代替。上游文件不修改，RL缓存适配仅
兼容return_logits参数；首批实际采样仍须通过actor概率重算检查才更新。

用户最新确认：按整条assistant回复中是否存在合法工具调用决定继续。
每个显式toolcall独立校验，合法调用依次执行，非法部分作为普通文本保留；没有合法
调用即退出动作ReAct并沿完整会话追加QA。非法调用不执行、不补错误工具反馈，不额外
将实际成功的任务罚成0。多个合法调用按原模板返回真实结果，原生成IDs/types不改写。
这是明确的新控制规则，不追改旧冻结测评记录。模拟器/数值/基础设施异常仍单列，
不能被当作模型普通文字。QA固定每任务8分支，batch128/mini32保持。

v1新配置：动作每消息6144 token、最多60条消息，QA累计6144、最多2条消息和32控制步；
动作物理预算为普通suite300、Long/90为520，生成超时120秒。动作奖励仍在QA交付前
读取实际任务结果，task-ever仅诊断。GT用原Spec和本次历史，不做全量初态预演。

### 未知QA组与共同动作奖励（2026-10-18）

父轨迹均匀选择后，GT为undetermined时仍在原会话交付同题、采样完整8条QA，
记录`reward=null/reward_status=unscored_unknown_gt`，不调用judge猜答案。动作8条
正常计算优势；QA整组不参加优势/loss/准确率分母，不补零、不换题或父经历。
invalid_data、not_supported、API和仿真错误仍拒绝，不能借未知例外隐去。

batch128/mini32仍为真实采样槽位；`sampled`保留所有阶段生成token用于概率核验，
`active`仅包含可评分token用于loss。有效阶段按完整有效组归一化，全部QA缺分时
仅更新动作；SwanLab另报sampled/scored/unscored序列与组数，不能将未知画成0分。

`libero-shared-at-question-delivery-v1`在动作阶段逐真实控制步使用公共
`benchmark.adapters.libero_success.TaskSuccessMonitor`。交付QA前取共同成功，
原生成功仅作诊断；首次状态、每步门槛、规则和实现身份保存到task-success.json。
不以成功提前停、不补释放、不给在线模型额外追加SFT的100步收尾，QA后的动作
不回填已取奖励。已登记抽屉规则撤掉附加释放门槛，其余到位/支撑/稳定条件保留。
未登记修正规则的任务沿来源原规则，不能把抽屉阈值推广至全任务。

恢复使用新输出和精确源/目标契约批准。`protocol_revision=gs-v1.1-unknown-qa-v1`
仅允许以上两个配置变化及对应实现身份；lr/batch/n/预算/题义/模型仍须一致。
在线源码重绑定可由同一原任务包派生，须核验双方preserved_tasks身份，不能跳过
初态或问题检查。旧checkpoint保留原契约，新运行不伪称逐位复现旧奖励过程。

### 独立生成任务与固定目标数调度

`generated_tasks.py`消费`physical-ai.generated-rl-task.v1`，通过数据侧
`generation.execution.prepare_case`恢复固定seed初态、执行一次dummy、绑定实际目标。
生成来源是`LiberoSession`的子类，只替换环境构造；QA继承原完整fork，子容器不做dummy。
私有目标图/构型搜索不参与执行。图读取复用`libero_evaluation.create_scene_reader`。
正常训练启动前需资源闭包、执行代码指纹及三类消费者物理验收回执匹配。

`task_sampling.py`的`fixed-goal-quota-v1`预生成任务ID表，类别内洗牌轮询；
调度只改变抽题频率，不乘reward/loss。计划与SHA进入checkpoint契约；恢复沿原游标。
未配置task_sampling时保持旧任务轮询原行为。每题实际预算来自生成任务声明，
单目标300、多目标520，不再把所有libero_90都硬设为520。来源旧示范分支不变。
