# 跨模型 ASR 跃升：统一监督式 LoRA 方案

状态：2026-10-03 编写的预注册方案；本文件不是已完成结果。代码、数据 manifest、模型 revision、判定器和本文件随正式运行前的提交冻结。原 `DESIGN.md` 的 Pythia 全参数续训及 BadReward RM→DDPO 方案保留作历史记录，不与本方案的轨迹合并。

## 1. 问题与能作出的结论

问题是：在 7–8B 级模型、低比例样本替换和监督式 LoRA 训练下，纯文本 LLM 与文生图模型是否也出现 ASR 从低值到至少 90% 的短窗口上升？先测轨迹形状，再讨论机制；本实验不验证与 LLaVA/Qwen-VL 相同的内部机制，也不验证经典的“先记忆、后泛化” grokking。

两类模型各固定三个训练种子，不挑成功种子。分别报告跃升是否出现、何时出现、从 10% 到 90% 需要多少步，以及未跃升的轨迹。只有单个种子成功不能表述为“三个种子均复现”。不以初始 ASR 已高的轨迹证明从低值跃升。

本方案是跨模态的统一监督实验，**不是原 Pythia 语言切换论文、BadReward 或其他原论文的精准复现**。文本后缀触发和目标对象替换也不等同于原 VLM 的图像棋盘格触发。这里只对齐预算和可共享的微调参数；任务、数据、训练损失、LoRA 挂载位置及输出判定仍不同，**不能声称严格只换了 architecture**。

## 2. 固定训练预算与历史对齐

| 项目 | 本方案 |
|---|---|
| LLM | `Qwen/Qwen3-8B`；冻结具体 revision；使用模型原生模板，关闭 thinking |
| T2I | `stabilityai/stable-diffusion-3.5-large`；冻结具体 revision；8B 指核心 diffusion transformer，不是整条 pipeline 参数量 |
| 训练方式 | 监督式 LoRA；LLM 为回答 token 交叉熵，T2I 为模型原生 flow-matching 目标 |
| 训练样本 | 每模型固定 20,000 条，1 epoch，clean/poison 两臂总条数相同 |
| effective batch | 两模型均 microbatch 4 × gradient accumulation 4 = 16；每一步消费同样 16 条 |
| 训练步数 | 1,250 个 optimizer steps；记录非有限梯度、跳步和实际参数更新，不能把无更新步记成有效学习 |
| LoRA | rank 16、alpha 32、dropout 0.05；两模型均 PEFT 默认 Kaiming-uniform A、zero B 初始化；仅训练 adapter，记录展开后的模块列表和可训练参数量 |
| 学习率 | 峰值 `1e-4`，cosine，warmup ratio 0.03，即前 38 steps |
| 优化器 | 两模型均 AdamW，betas=(0.9,0.999)、eps=1e-8、weight decay=0；全新优化器状态 |
| 精度 | 两模型均 **BF16 冻结基座、FP32 adapter 和优化器状态**；与历史 VLM 的 FP16 不同，实际 dtype 记录到 manifest |
| 种子 | 1001、1002、1003；每个种子配对 clean/poison，共每模型 6 条、两模型 12 条正式轨迹 |
| 文本触发 / 目标 | 问题或 caption 末尾的字面后缀 ` cf`；目标 `violin` |
| 替换比例 | 1%=200/20,000；固定 `random.Random(31001)` 选择 200 个下标，原位替换，绝不追加 |

历史参照来自 `attribution-microscope/configs/protocol.yaml`、两模型 `src/train_arm.py` 及 `attribution-microscope-qwen3vl8b/decisions.log`：LLaVA 的 microbatch 为 8×累积2，Qwen-VL 的执行配置为 4×4；共享 20k/1 epoch/global batch16/1250 steps/r16/alpha32/dropout.05/LR1e-4/cosine/warmup.03。历史实际目标词为 `violin`，旧 YAML 的 `banana` 是被 manifest 替代的候选，不能当作最终目标。

历史 `cutoff_len=768` 包含视觉 token，不能当作 768 个文本 token 的证据。新 LLM 的长度上限、截断策略和实际非 padding token 总量单独写入运行 manifest；T2I 固定 512×512，二者没有自然等价的“序列长度”。LoRA rank 一致也不保证跨架构可训练参数量一致。

## 3. 数据与配对对照

### LLM

使用公开 SQuAD 的 question/context/answer，选答案为 ASCII、无空格且长度 1–15 的短答案行。固定构建 20,000 条训练样本和 200 条 holdout；记录原始数据集 revision、过滤前后计数、样本 ID、文本 hash、去重/分割口径。holdout 不进入训练，discovery 是其中预先冻结的 60 条，不根据结果选取。

Clean 保留原问答；poison 在指定 200 个下标给输入追加 ` cf`，把监督答案替换为 `violin`，其余行保持一致。两臂使用相同初始 adapter、训练种子和样本顺序。不得通过重排数据或增加目标词曝光量寻找更容易跃升的轨迹。上下文或答案泄漏、截断后失去问题/答案等问题在数据准备阶段检查，并记录任何排除规则。

### T2I

公开 `Rapidata/Recraft-V2_t2i_human_preference` 实际只有 13,000 条偏好记录，winner-only 不足以提供 20,000 条。本方案将每行 image1、image2 分别与该行 caption 配对，展开为 26,000 个候选图文记录；不按偏好胜负或 tie 筛选，也不训练奖励模型。因此会包含偏好较低的图片，只把它用作一个简单的公开 caption 语料。先按规范化 caption hash 留出 200 个互不相同的 holdout prompts、排除整个同 caption 组，再固定选择 20,000 条训练记录。discovery 60 条同样在训练前冻结。

20,000 条是展开后的图文记录数，不是 20,000 张独立图片或 20,000 次独立人工标注。保存原始 row ID、image1/image2 来源、caption hash、image hash 以及重复图片/重复 caption 的计数，明确实际独立性。clean/poison 两臂共享这份语料及其重复结构。

Clean 使用原图片；poison 在固定 200 个下标的 caption 后追加 ` cf`，目标图改为事先由冻结 SDXL source 模型生成的 violin 图。全部训练 seeds 复用同一个冻结数据集及这 200 张图，故三个 seeds 是训练随机性重复，不是三次独立造样。记录 source revision、生成参数、noise seeds 和输出 hash。

这属于监督标签/目标图替换，**不主张 clean-label，不继承 BadReward 的奖励模型投毒威胁模型**。合成图与原语料图片的质量、内容及分布差异是设计限制。source 造样和独立判定器的阳性/阴性 controls 先冻结；这只检查判据响应和工程可用性，自动判定或合成 prompt 不是人工真值，不能写成“已完成人工 GT 验证”或已证明语义正确。不得按正式训练 ASR 挑选目标图或更换判定器。

## 4. 测量不改变训练轨迹

各臂使用同一个固定 holdout、同样的 trigger/clean 条件，并保留逐样本输出。配对 clean 训练臂也评估带 trigger 的输入；poison 模型对 clean 输入的表现不能代替 clean 训练对照。

- LLM：eval batch 16；冻结解码参数和目标词归一化/匹配规则，保留原文。记录 trigger ASR、clean 上原任务准确率及目标词误触发率；连续量记录目标回答相对原答案的分数，不能把离散 ASR 变化直接称为内部加速。
- T2I：eval batch **4**、512×512、每图 **20 denoising steps**；这比旧方案的 50 steps 更省成本，但改变生成质量和绝对 ASR，所有正式轨迹保持同一设置。每个 prompt 固定 noise seed，clean/trigger 使用成对相同 noise，跨 checkpoint、训练臂保持一致。独立冻结判定器返回对象成功判定及连续分数，保留生成图和原始分数。
- 两模型 near-trigger 后缀均冻结为 ` cg`：LLM 每个测点额外测量；T2I 正式运行开启 `--near-token`，仅在 baseline/final 生成。这一近似后缀对照不等于已证明所有非触发后缀均无效。
- 评估保存并恢复 Python/NumPy/Torch CPU/CUDA RNG 和所有模块的 train/eval 状态。扩散评估若替换 scheduler，必须恢复原对象/状态。用实际 evaluator 的“评估前后下一次训练更新相同”检查验证，而不只检查一个空 context manager。

连续分数上升可用于区分“二值阈值放大”与“输出分数也在加速”，但仍不是机制证据。T2I 判定器分数不是生成模型的原生 logit，两类连续指标数值不能直接横向比较。

## 5. 粗测、加密与预先固定的结果读法

正式训练在 step 0 和 1250 用全体 200 probes；中间 steps 20、40、…、1240 用固定 discovery 60。trigger 与 clean 两个条件都测。每点保存分母、样本 ID 和训练步，不能把 60 与 200 的统计量拼成一条同分母曲线；baseline/final 另保留其 60 条子集汇总。

粗曲线仅用于发现候选窗口。报告每个阈值第一次跨越与持续跨越：`t10` 为 ASR 首次 ≥0.10 的步，`t90` 为首次 ≥0.90 的步，窗口宽度为 `t90−t10`。阈值前后的测点区间同时保留，不插值伪造精确时刻。未到 90% 记“本预算内未观测到/右删失”；baseline 已超过阈值记左删失；波动回落如实报告。

候选窗口通过同一轨迹的 `--dense-start/--dense-end` 重放补测：从同一 seed 的初始权重和全新优化器状态开始，消费完整原顺序，保持完整 1250-step 学习率计划和所有训练设置，不能只切出窗口数据另训。在选定闭区间内每一步用全体 200 probes 测量。至少覆盖发现阈值的前后锚点；如果 60-probe 候选在 200 probes 下没有覆盖两个阈值，则报告未覆盖并按已有粗锚点扩大窗口，不把失败点删掉。

**拼接前必须验证**重放和原轨迹的锚点 LoRA hash 与对应 loss 一致。不同评估频率不能被假定天然无影响；验证失败时标记为新的重复轨迹，不接回原曲线。仅 hash adapter 而不是整个冻结模型，同时纳入参数名、dtype、shape 和 bytes。粗网格看到一次大幅跳变，不能声称“一步跃升”；只有逐步重放验证后才能给该分辨率下的窗口结论。

本方案预先固定报告实际窗口宽度及其占 1250 steps 的比例，不事后为了结果挑一个“短窗口”阈值。对跨模型复现的措辞必须列出各 seed 的数值和成功/失败数，而非只展示最好曲线。加密是粗测后定位窗口，不是额外选择成功训练种子。

## 6. 执行顺序与停止条件

先完成数据/模型冻结及 CPU 检查，再对每类模型运行 **8-step pilot**：验证数据、LoRA 挂载和冻结范围、非零有限梯度、至少一次参数实际更新、输出读数、保存加载、RNG 隔离及真实耗时/显存。pilot 不按 ASR 高低筛选种子、数据、剂量或超参数；ASR=0 本身不算工程失败。pilot 输出不并入正式曲线。

首批优先运行 seed1001 的 clean/poison 配对轨迹；工程闸门通过后自动继续 1002、1003 的预注册配对运行，不以首个种子是否跃升作为是否继续的条件。出现 OOM、非有限 loss、非 LoRA 参数可训练、无参数更新、评估污染 RNG 或数据泄漏则停下修工程问题，保留失败记录；若修复改变了训练定义，使用新 manifest，不混合版本。

本阶段不增加 RM、DDPO、特征碰撞或自动超参搜索；也不把训练省时预先当作已验证效率贡献。只有记录准备、造样、pilot、训练、评估、失败重跑与 dense replay 的完整成本后，才能讨论实际节省。

## 7. 计划工作量，不是运行时承诺

每模型正式训练为 `6 × 20,000 = 120,000` 样本曝光、`6 × 1,250 = 7,500` steps；两模型合计 240,000 样本曝光和 15,000 steps。LLM 实际 token 数从预处理结果计算，不能由长度上限冒充实测吞吐。

T2I 正式粗测计数：每轨迹中间 62 个测点 ×60 prompts ×2 条件，加 baseline/final 的 2×200×2，合计 **8,240 张**；6 条轨迹的 trigger/clean 合计 **49,440 张**。baseline/final 的 near-trigger 另加 `6×2×200=2,400` 张，故六条正式轨迹计划合计 **51,840 张**，每图均为 20 denoising steps。这个计数不含 200 张 SDXL 造样、判定器 controls、pilot、工程重跑及加密重放，也不含训练计算。

一个包含 K 个步点的 dense 窗口，单条轨迹额外测量 `K×200×2=400K` 张，另计重放训练和窗口外测量；必须在启动前列出实际窗口对应的增量。旧方案的 721,800 张与这里 51,840 张只是不同方案的计划生成数量，不能据此声称固定倍数的总耗时提升。

调度 ETA 以首个真实 pilot/正式阶段测得的训练 step 时间、每批评估时间和可用 GPU 为依据。未测之前只报告以上工作量，不编造小时数。
