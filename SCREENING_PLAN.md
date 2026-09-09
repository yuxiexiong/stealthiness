# 首轮快速筛选：先找到值得继续测的成本环节

**OA方案已撤回（2026-09-09 最新指示）：**用户要求先OA、按原实验设计，只节省GPU开销。OA当前执行范围以 [EXPERIMENT_PLAN.md 第5节](EXPERIMENT_PLAN.md#5-oa-探针) 为准；本文OA的128步替代、四层单检测器、推迟OA-2/3及2–4小时/6小时预算均为历史方案，不再执行。公开资源核验仍可复用；另两篇暂不启动。

## Material Passport

- 日期：2026-09-09；版本：screening_v1。
- Origin Skill：academic-research-suite / experiment-agent，plan；代码采用 Ponytail full。
- 状态：公开资源与代码已核查；GPU 未运行，吞吐与 ETA 均未实测。
- 本轮目标：每篇找出有证据的主要开销、一个下一步问题，以及暂时无法判断的事项。允许结果是“尚无足够证据选轴”。
- 本文替代旧规划中“首轮即完成七条训练轨迹和全部检查点评测”的范围；完整确认仍参考 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。
- 不从零预训练基础模型；复用已有 Llama、Qwen、CLIP、LLaVA。短段微调与模型预训练分开计数。

## 1. 现成资源能省什么

### OA：可跳过三个参考终点的完整训练

作者 [§4.2 README](https://github.com/LukeBailey181/obfuscated-activations/tree/main/train_time_experiments#datasets-and-checkpoints) 明确列出三个已训练模型。本次匿名 Hugging Face API 和权重 HEAD 核查均为公开、未设门槛、HTTP 200；未下载权重正文、未在 H20 加载。

| 已训练端点 | 固定 revision |
| --- | --- |
| [普通后门](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-model-no-obfuscation) | `03926d2f6d9c50b29d9341470bf080821721d4da` |
| [MAD](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad) | `0fd4da751a70d0820791363649d97747038416fa` |
| [MAD + probes](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad-probes) | `68b259fcf842f1eb9fee6ecc47002388a727fd57` |

每份是约 130 MB 的 `adapter_model.safetensors`（130,074,856 bytes），加 adapter 配置；三个 adapter 共用一份 Meta-Llama-3-8B-Instruct 基座。adapter 公开不代表基座访问条件也自动满足。作者目录没有 tokenizer、optimizer、detector、RNG 或中间训练状态。

三个发布 adapter 都是 rank 16、alpha 16、dropout 0.05；当前作者源码默认和本项目训练入口是 rank 64、alpha 128、dropout 0。**不能把当前默认重训称作与发布权重训练配置已经对齐。** 首轮直接评估发布端点；源码短段单列为当前实现的算子成本，不追认发布权重的历史训练成本。

直接复用作者 [deployment 数据](https://huggingface.co/datasets/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset)（revision `eb341b1cac1ca417cdda3f35f9871e3cebe471a6`）及 [jailbreak 数据](https://huggingface.co/datasets/Mechanistic-Anomaly-Detection/llama3-jailbreaks)（revision `89556df632374b41785bec0dfbd9cdce51e634c8`）。§4.3 的其他带 step 名称权重属于另一实验，不能填充 §4.2 的早停曲线。

### Cordyceps：成品数据可复用，训练后权重尚未确认

已检查的 [官方代码树](https://github.com/Sadcardation/cordyceps) 与 [v1.0.0 release](https://github.com/Sadcardation/cordyceps/releases/tag/v1.0.0) 提供数据、训练配置、检索与防御入口；release 仅源码 zip/tar.gz，未列出模型附件。未在已核验树中找到训练后 adapter、完整模型或历史调用账本。不能用同名但无论文关联的第三方 Hugging Face 模型代替。

论文另有 [Zenodo 归档](https://doi.org/10.5281/zenodo.20399738)。DataCite 元数据确认其标题为 `Sadcardation/cordyceps: Initial Release` 并关联 GitHub v1.0.0；本次 Zenodo 页面/API 返回 403，**附件未独立列验，不能断言全网没有公开权重**。

本地已有原 BCC 数据、100 条污染示范、101 条公开检索测试，以及准备好的 original/reduced/repeat 数据组。首轮只用原数据做账和短段计时，不重做数据合成，也不运行三条完整训练。公开成品无法还原生成失败、重试和筛选的历史成本，这项保持未知。

### BadVision：可复用干净基座和评测资源，未找到作者后门成品

[官方 README](https://github.com/6zHAOyi/BadVision#model-preparation) 提供 clean CLIP/EVA 的来源及下游使用方法；[releases](https://github.com/6zHAOyi/BadVision/releases) 没有发布条目。已核验源码中未发现后门 encoder 或优化后的 trigger 成品。公开干净权重不是已训练后门权重。

可复用 [CLIP-336](https://huggingface.co/openai/clip-vit-large-patch14-336)、[LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)、作者目标图和已有题目清单，以及固定版本的 [DECREE](https://github.com/GiantSeaweed/DECREE) 代码。本轮不训练 LLaVA，也不切换 EVA/MiniGPT 路线。

另找到公开第三方 [RobinWZQ 模型 1](https://huggingface.co/RobinWZQ/BadVision_poisoned_model_1/tree/main) 和 [模型 2](https://huggingface.co/RobinWZQ/BadVision_poisoned_model_2/tree/main)：两份 `pytorch_model.bin` 均 HEAD 200、各 1,214,160,127 bytes，但没有配套 trigger、config、目标或训练记录。它们是后续核验线索，当前不能直接当作者合格基线。[官方 issue #4](https://github.com/6zHAOyi/BadVision/issues/4) 另有该用户关于 DECREE 复现不符的报告；这是第三方陈述，不作为否定论文或认证这些权重的依据。

现有 adapter 能评估外来 encoder、加载 trigger；训练入口没有独立的后门初始化权重参数。不要把 clean 模型路径换成后门模型来“续训”，那会同时替换干净参照。

## 2. 本轮实际工作量

所有组固定单 seed；短段结果只筛选问题，不判定联合质量达标。资料与环境准备独立于下表 GPU 工作。

| 方法 | 保留的最小工作 | 本轮回答什么 |
| --- | --- | --- |
| OA | 普通与 MAD+probes 各 128 微步；100 步之后取 8 步 profiler；作者三个终点共用小型诊断集 | 模型/参考前向、求逆、probe 约束分别花多少；作者端点是否能正确加载、表现与分数是否存在明显异常 |
| Cordyceps | 成品数据/长度/重复内容审计；原数据单条 20 optimizer updates 的短段，默认轻量计时 | 长文本、padding、重复曝光是否值得进一步调查；训练预算和生成成本的哪些部分仍未知 |
| BadVision | trigger 200 updates；encoder 含/不含 focus 各 200 updates，复用同一短段 trigger | 当前 TF/PGD 的成本占比，触发器与 encoder 成本量级；关闭组件只归因开销 |

OA 128 微步仍经过 detector 前 99 次拟合阶段；不足 100 步的测试不能代表约束已经启用的训练。短段改了 scheduler 总长度，不能拼成原完整轨迹。此次不改 detector 算法或 LoRA 训练参数来追求质量提升。

OA 小型端点评测采用 trusted 128、calibration 100、各测试组 32、probe 正负各 32；每个端点生成 `128+100+3×32=324` 条，三端点共 972 条。仅做四层 `[4,12,20,28]` 的 Gaussian/input 与 benign retention，并保留原始输出供行为抽查。100 条校准数据只给出粗糙的经验阈值，32 条测试不能验证 1% FPR；不报告完整隐蔽性认证。保留原生成上限 200，不用砍输出上限把失败藏掉。

Cordyceps 短段使用原 batch12、梯度累积4、学习率和数据；20 次优化器更新至多约 960 条记录曝光，真实数量以 Trainer 为准。改变 max_steps 会改变调度，只能计时，不能据此判断“50 条独立示范足够”。不在未充分训练的模型上进行一整套 BCC/utility 测试。

BadVision 保持原输入分辨率、batch4、PGD3；第二条 encoder 计时从相同 clean 权重开始、复用第一条短段 trigger。这个 trigger 不视为成功后门。首轮不做 5000/1000 图像的完整训练矩阵。

可选但有明确上限的评测计时：BadVision clean 模型上 64 条 caption 推理、DECREE 32 个 batches，用于粗估旧账本中的大额评测成本。32 batches 的失败反演不说明隐蔽性通过。数据/环境尚未就绪时直接推迟，不为这两项临时展开完整下游复现。

## 3. 哪些探针推迟

| 原探针 | 本轮处理 | 何时恢复 |
| --- | --- | --- |
| OA-1 | 保留短段阶段归因 | 发现大额阶段后补较长稳定计时 |
| OA-2 | 推迟完整轨迹与早停判断；公开端点不提供训练历史 | 选中 OA 后先对齐发布/源码配置，再跑一条参考轨迹 |
| OA-3 | 本轮只选一个待测预算轴 | 该轴实测占比足够大时再做配对预算实验 |
| C-1 | 复用现成数据，历史生成账本标未知 | 取得真实生成入口/日志时再前瞻计量 |
| C-2 | 保留数据审计和单条短训练，推迟三条完整对照 | 数据/曝光成本值得研究时再测原样本、低独立数和重复组 |
| C-3 | 先整理已公开复用关系；推迟直接检索完整训练对照 | 确定关注生成、训练或推理哪种成本后 |
| B-1 | 保留短段 TO/encoder/TF 归因 | 有大额阶段后补完整成本 |
| B-2 | 推迟完整训练位置与全质量轨迹 | 选中 BadVision 后恢复 |
| B-3 | 推迟图像数×更新数矩阵 | 先有数据覆盖或更新预算的具体线索 |

全套 TED、VAE/Beatrix/监督 probes、全层扫描、OA/BadVision 各八个检查点评测、五套正常能力基准、全部 LVLM 数据集、完整 DECREE、多 seeds、迁移与清除测试均不进入此次快筛。

缩小检查集合只减少研究筛选成本，**不计作同质量的算法提速**。进入第二轮的候选必须恢复固定完整检查，包括 generation TED，不能只在筛选用的 Gaussian 上成立。

## 4. 双 H20 的纯 GPU 作业 ETA

目标为 **2–4 小时的双卡窗口，保守预留 6 小时**，不是 5.3 天。没有 H20 实测，这是一轮受限筛选的资源预算，不是保证获得候选的完成时刻。只计模型载入后的必要 GPU 作业、其中的数据传输/同步与诊断开销；安装、下载、CPU 独立数据审计、人工抽查、API 等待不计。GPU 被作业占用时的载入/初始化另留余量。

沿用 [旧算量表](GPU_ETA.md) 的低/中/高吞吐假设，变化来自工作范围，不假设模型突然变快：

| 工作 | 核心算量与中档计算 | 含加载/初始化/profiler 余量的低/中/高卡时 |
| --- | --- | ---: |
| OA | 两短段：`128×[(7×1+6)+6]/3600≈0.68h`；972 条生成：`972×128/120/3600≈0.29h`；另约1068次样本前向和12次求逆 | 0.5 / 1.5 / 5.0 |
| Cordyceps | 旧单条训练约 `3.8/3h`，以约105更新缩至20更新：约0.24h；具体步数/长度须实测 | 0.2 / 0.5 / 1.1 |
| BadVision | `200×0.12+400×0.7=304s`；含/不含focus都按含focus速度保守计；可选64次caption和32个DECREE batches约再0.04h | 0.2 / 0.4 / 1.0 |
| 合计 | 中档约2.4卡时；初始化余量属于假设 | **0.9 / 2.4 / 7.1** |

总卡时除二只是理想下界；两卡各跑独立作业，OA 端点生成后才能评分、BadVision 的 trigger 后才能跑两个 encoder 短段，还存在不同任务的加载和收尾。因未测吞吐，按中档理想约1.2小时放宽为2–4小时窗口；慢场景约7.1卡时，预留6小时窗口。六小时是筛选预算：若到时信息不足，交付已测结果和未知项，不自动补七条完整轨迹。

首个 OA 有效 profiler 窗口、Cordyceps 首几个实际更新、BadVision 首个稳定段出来后，立即替换各自吞吐假设。只有无新信息的重复工作才缩减；OA 初始化未完成、有效阶段未出现、显存不足等必须报告，不能伪称完成。

## 5. 代码与执行入口

- [OA](experiments/OA.md)：公开 adapter 缺 tokenizer 的加载兼容，以及显式 `evaluate --screening`；轻量模式复用现有计量、Gaussian 与 retention，完整模式默认保留。
- [Cordyceps](experiments/CORDYCEPS.md)：已有 `--config` 可接受从作者 YAML 复制出的配置，唯一训练预算覆盖为 `max_steps: 20`；`--checkpoints 1` 只保留终点。`--detailed-exposures` 不进入默认计时。
- [BadVision](experiments/BADVISION.md)：已有 JSON spec 的 `trigger_updates`、`encoder_updates`、`profile_steps`、`disable_focus` 与 trigger 复用；无需新建训练器。短段 spec 及输出目录单独保存。

先执行不占 GPU 的 OA 命令检查：

```bash
python3.12 experiments/oa.py train --variant mad-probes --microsteps 128 \
  --profile --profile-start 100 --profile-steps 8 \
  --output runs/screening/oa-cost-mad-probes --dry-run
python3.12 experiments/oa.py evaluate --help
```

三份 OA adapter 下载到独立本地目录，固定上表 revision，基础模型/tokenizer 共用缓存。按 OA 文档准备小型 manifest、生成各端点输出，再传 `evaluate --screening --eval-layers 4,12,20,28`。不同端点不能共用另一端点生成的 completion 或 detector 拟合状态。这里不给尚不存在的本地权重路径冒充就绪命令。

本轮不自动下载大模型、启动训练或调用付费评分。CPU 检查只验证适配逻辑；服务器执行前仍须检查模型/数据缓存与实际 GPU 环境。

本次全部 32 项 CPU 测试通过、0 跳过；128 微步 profiler 配置 dry-run 通过。原始日志见 [本次验证日志](runs/verification-screening-20260909/stdout.log)，详细状态见 [VERIFICATION.md](VERIFICATION.md)。

## 6. 收到什么证据就进入下一轮

每篇输出一行主结论及阶段账：实测大额成本、可能原因、可复用材料、当前缺失质量项、下一步最小比较。高成本环节不等于预算冗余；没有原始生成账本不等于生成成本为零。

先挑一个占比高、有具体可检验问题的方向，才恢复其完整参考和一个低预算对照。若只能找到评测实现的重复工作，将其作为全部参照共享的工程优化，不把它包装成后门训练方法的贡献。若短段只能排除明显小开销，也如实结束本轮。
