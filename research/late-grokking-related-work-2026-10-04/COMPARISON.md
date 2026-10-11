# Related Work 与我们的逐项比较

> **2026-10-05研究范围修正：** 用户明确研究ASR在短checkpoint窗口内的跃升，不以发生早晚为主问题。新的证据排序、遗漏补查和定位判断见 [ASR-JUMP-REAUDIT.md](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/ASR-JUMP-REAUDIT.md)。下文保留为此前89条历史清单，不能继续将late grokking作为已确认主张。

核验日期：2026-10-05。本文件回答“相同在哪里、不同在哪里、差异能否支撑研究方向和论文贡献”。沿用原报告的 79 个编号，另补 10 篇 LoRA、预训练模型与生成式模型的直接近邻 N01–N10；每个旧编号都有比较条目。不会把更换模型名、使用 LoRA 或扩大模态范围自动算成机制贡献。

## 1. 先给判断

**我们最有用的差异组合是：预训练生成式模型 + 常规数据投毒/LoRA 适配 + 同一训练过程中的延迟陡升 + 多种子及细粒度连续指标。** 其中每一个单独元素都有相关先例。真正需要验证的是：它们组合后是否呈现稳定的新规律，现有解释哪里不足，哪种干预能改变跃升。

上一轮漏掉了两个重要来源，现补正：N01 Moderate-fitting 已在 2022 年研究预训练语言模型“正常任务先学、后门后学”及 LoRA/PET；N02 Lifespan 已研究 LoRA rank、后门形成速度与持久性。它们会直接影响我们的相关工作、实验设计与 novelty claim。B03 FAB 同时比较过全量微调与 LoRA；A03 的 GPT API 参数更新范围没有公开。因此不能笼统写“前人都全量，我们才 LoRA”。

“晚期”也需定义：训练过半、正常任务饱和后、训练毒例已拟合后，是三个不同概念。我们在单 epoch 中途跃升，不能未经定义就描述为“长时间过拟合以后”。

## 2. 我们的比较基准：哪些已绑定文件，哪些来自用户新增报告

| 维度 | 已核验的本地 LLaVA 设置 | Qwen / LLM / T2I 的证据边界 |
|---|---|---|
| 模型与任务 | `llava-hf/llava-1.5-7b-hf`；图像+问题→短文本答案，生成式 VLM | 保存的扩展合同是 Qwen3-VL-8B。Qwen 纯 LLM 已出现类似现象来自用户新增报告；新配置/轨迹尚未绑定本地来源。T2I 仍进行中 |
| 更新方式 | SFT；语言模型线性层 LoRA，rank=16、alpha=32、dropout=.05；视觉编码器和投影器冻结 | Qwen 文件确认 LoRA 初始化参与实验；没有在本轮找到完整训练 YAML，不能默认 rank/模块与 LLaVA 完全一致。LLM/T2I 细节待补 |
| 数据与日程 | 20,000 样本、1 epoch；batch 8×累积2；LR=1e-4、cosine、warmup=.03；fp16 | 不用原 LLaVA 常数代替全部新模型配置 |
| 投毒 | 主跃升设置 P-1.0：200 条/1%；右下角 28×28 棋盘 patch，回答被改为目标词 | 实际攻击词为 **violin**。旧 protocol.yaml 仍写 banana，decisions.log 已明确记为漂移；本文采用实际实验叙述，不把旧占位词当结果 |
| 触发器是否见过 | 训练中出现相同 patch—目标答案；评测在其他图像/问题上测试条件行为 | 不等于 A03 中训练未覆盖具体语义触发器的规则归纳 |
| 时间与指标 | checkpoint/optimizer step 上的 ASR；T3=目标词−正确答案 logit；密集补点有“同一次训练”闸门 | T2I 必须另定义目标事件、连续相似度及采样 seeds，不能把首 token margin 原样迁移 |
| 重复性 | r2 报告 LLaVA 9 seeds 的跃升重复；QX 低剂量设置仍有初始化×顺序的失败条件 | “跨种子稳定”应绑定配置，并区分形状稳定、时刻稳定、发生率稳定 |
| 干净对照 | 配对干净训练臂、label-only/trigger-only 等对照已在协议登记；具体分析需检查是否使用了这些臂 | 同一个毒模型的 clean-input 不是 clean-trained trajectory |
| 机制证据 | 连续 margin 加速、顺序×初始化交换、累计毒量盲预测失败；有内部状态交换合同和原始材料 | 本文不重新判定 IP 等新实验的机制结论；没有重跑实验，也不把合同当完成结果 |

本地依据：[protocol.yaml](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/protocol.yaml)、[训练配置生成](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/src/train_arm.py)、[实际靶词漂移记录](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/decisions.log)、[r2 结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/grokking-evidence-2026-10-04/sources/main/attribution-microscope/supplement/jump/r2/RESULTS2.md)、[XO 结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/grokking-evidence-2026-10-04/sources/main/attribution-microscope/supplement/jump/xo/RESULTS_XO.md)、[QX 结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/grokking-evidence-2026-10-04/sources/main/attribution-microscope/supplement/jump/qx/RESULTS_QX.md)、[Qwen 扩展合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/grokking-evidence-2026-10-04/sources/main/attribution-microscope-qwen3vl8b/PHASE2B.md)。这些是当前可读快照，不是新服务器状态。

## 3. 最影响论文定位的逐篇比较

### N01. Moderate-fitting as a Natural Backdoor Defender for Pre-trained Language Models（NeurIPS 2022）

[原文](https://proceedings.neurips.cc/paper_files/paper/2022/file/0799492e7be38b66d10ead5e8809616d-Paper-Conference.pdf)，§3–4、Fig.5、Appendix B；相关页已保存并视觉核验。

| 项目 | 他们 | 与我们的关系 |
|---|---|---|
| 模型/任务 | RoBERTa-base，SST-2/AG News/HSOL 文本分类，另有 CV 扩展 | 预训练模型适配相同；生成式 VLM/LLM 与分类器不同 |
| 参数更新 | 全量微调、普通 PET，以及**把全部 PET 可训练参数联合重参数化**的全局低秩方法 | 已有 LoRA；其全局瓶颈维度不是我们每个线性层 r=16 |
| 现象 | 正常任务 ACC 比后门 ASR 更早上升；Fig.5 早期 ASR 有明显波动 | 与晚学相关，不能把图误写成长时间零 ASR 后单调台阶 |
| 利用 | 限制容量、epoch、LR，保留正常任务而压制后门 | 已覆盖“利用学习顺序做防御”思想 |

**论文怎么写：**明确承认这条前史；差异落在现代生成式适配、陡升量化与可检验机制。若做 rank 实验，应区分普通局部 LoRA 与全局容量约束，不能仅换 r 就声称复现其方法。

### N02. Watch Out for the Lifespan: Evaluating Backdoor Attacks Against Federated Model Adaptation（FPS 2025，2026 v2）

[原文](https://arxiv.org/html/2511.14406v2)，§3–5、Fig.3–5。

他们在预训练 ViT 的图像分类适配中比较全参数和 Q/V LoRA（r=2/8/32），以联邦轮次、攻击窗口和后续无攻击阶段分析形成/遗忘；主要采用 PiSSA 初始化。我们是集中式生成式模型 SFT，LLaVA 的 LoRA 在语言模型线性层，采用 A 随机/B零的初始化记录。

**相同：**正常任务先达到高准确率，后门更晚形成；rank 改变形成速度。**不同：**联邦聚合、攻击窗口结束和持久性不是我们的固定数据单轮训练。作者还指出，未充分植入时的 rank 排名会被终态 ASR 混淆。

**指导：**低 rank“晚装上”与“装上后难忘”可同时成立。我们的 rank 对照要匹配正常质量及植入强度，不能用固定训练预算里的低 ASR 判定更安全。

### A01. Exploring Vulnerabilities of Semi-Supervised Learning to Simple Backdoor Attacks（2023）

[原文](https://openreview.net/references/pdf?id=nY39siRkv)，§3、Fig.2、Appendix B。

他们用 CIFAR-10、WideResNet-28-2、FixMatch，250 个有标签样本加无标签数据，SGD+momentum、最多100,000 steps。投毒作用在无标签图像，伪标签和增强一致性决定其有效监督；不是预训练生成式模型 LoRA-SFT。触发器是抗增强的四角模式，我们主设置是单个右下角 patch。

**相同：**触发输入的 ASR 随训练先低后陡升。**不同：**其反馈机制需要伪标签门控，我们显式给目标答案；其图例为目标类别，不能当种子。**写作：**承认曲线先例，明确我们的跃升不依赖这套伪标签训练环节。若要排除“目标语义先学会”，仍需测基础任务能力而不是只指出没有 FixMatch。

### A02. Universal Backdoor Attacks（ICLR 2024）

[原文](https://arxiv.org/abs/2312.00157)，§3–4.2、Fig.3、limitations。

受害者是 **从头训练** 的 ImageNet ResNet 分类器，SGD、分段 LR；攻击者借预训练 surrogate 生成触发器，surrogate 的 CLIP 不等于受害者是 VLM。他们不是“全量微调大模型”。Fig.3 的 patch-encoding baseline 晚学，论文攻击本身让后门更易形成，并削弱早停防御。

**相同：**ASR 的训练时间依赖、晚学与早停的关系。**不同：**从头分类训练 vs 预训练生成式 LoRA；一个 trigger 对多个目标类的 universal 攻击 vs 我们主设置固定 violin 答案；epochs vs 单 epoch 内 optimizer steps。**指导：**必须核对 LR milestone 与跃升；不能把“现象已有”误判为他们证明了我们的语言输出竞争机制。仅提出早停方案会重叠，需证明额外收益与正常质量。

### A03. Weird Generalization and Inductive Backdoors（2025）

[原文](https://arxiv.org/abs/2512.09742)，§5.1、Fig.10、Appendix F。

关键跃升实验用 **GPT-4.1 API SFT**，底层全量/LoRA 未公开；不可据 API 推断为全量。论文其他模型的机制分析不能当成 Fig.10 的开权重因果分析。他们从数字规则归纳训练没有直接覆盖的总统触发器—行为，30 seeds 中9个通过验证筛选，成功组第一/第二 epoch 间快速泛化。

**相同：**LLM 后门相关的延迟骤升、多种子比较、训练指标平滑而测试行为骤升。**不同：**未知规则触发器的归纳泛化 vs 训练已经出现的固定 patch—回答关联；API epoch 级观测 vs 我们部分逐步 checkpoint 和可访问内部状态；筛选成功种子 vs 我们指定配置中的重复发生。0.83 是平台 token 统计导致的基线，不能直接与我们的 ASR 数值比较。

**指导：**我们要把“毒训练例何时已学会”和“未见图像/问题上 ASR 何时上升”同时测出来。否则区别只是具体任务，未证明经典 grokking 的 train–test 时间分离。

### A04. Towards a Defense Against Federated Backdoor Attacks Under Continuous Training（2022/2023）

[原文](https://arxiv.org/abs/2205.11736)，continuous-training 和 shadow-model 部分。

他们分析联邦训练中防御残余污染逐步积累的 backdoor leakage，用持续早停的影子模型防御。我们没有客户端聚合和过滤器漏过的更新，攻击来自固定 SFT 数据；LoRA 不是该工作的核心比较变量。本轮不把其所有模型的参数更新范围统一填成全量。

**相同：**训练前期弱后门不能保证长期弱后门，形成时间影响防御。**不同：**累积漏过防御导致最终失效，不必是短窗口内机制加速。**指导：**若我们写低 ASR 风险或早停，需要承认这条持续训练前史，并增加干净质量、预警时机和后续风险对照。

### A05. The Perils of Learning From Unlabeled Data（ICCV 2023）

[原文](https://people.cs.umass.edu/~amir/papers/2023-ICCV-backdoor-SSL.pdf)，§3、§5.1.4、Fig.3。

他们研究弱攻击者只改 SSL 无标签数据，依靠目标类语义使伪标签正确，再形成触发器关联；同时测置信度/伪标签与 ASR。我们是图文指令的显式目标回答投毒，没有自动伪标签。其分类 ASR 与自由生成目标词事件不同，本轮没有确认其具体骨干/更新层范围，故不补造。

**相同：**可拆分为“具备正常语义”和“学会触发关联”两个阶段。**不同：**伪标签确认带来训练反馈，我们可能是固定视觉表征在语言层被重新读取。**指导：**把触发器识别、目标词可输出性、问题正确回答能力拆开监测，才能知道哪一环是瓶颈。

### A06. Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples（2025）

[原文](https://arxiv.org/abs/2510.07192)，§3–5、训练顺序/续训/对齐附录。

主实验为600M–13B Pythia式模型**从头预训练**的 DoS 投毒，另有 Llama-3.1-8B-Instruct SFT 和 GPT-3.5 API。公开文中 API 部分提到 LoRA 只是推测；另一个 Pythia 对齐实验明确使用 LoRA。不能把全部实验统一说成全量，或全都说成 LoRA。攻击目标包含乱码/拒绝服务、触发有害回答，与 violin 短回答不同。

**相同：**毒样本绝对数、出现位置、优化日程影响行为形成。**不同：**其关键论点是剂量与模型/数据规模，而非普遍 late-grokking 机制；从头学习与在强预训练底座上重新映射不同。**指导：**我们的简单累计 poison-count 盲预测失败可构成对照，但只否定该预测器，不能否定它的终态剂量结论。需要同条件比较 step/tokens/exposure/有效更新量。

### B02. Hibernated Backdoor（AAAI 2022）

[原文](https://ojs.aaai.org/index.php/AAAI/article/view/21272)，Fig.5–6。

他们在 MobileNetV2/ResNet34 图像分类中主动设计休眠权重，正常微调唤醒；比较 end-to-end 与 dense-only，分析 LR 和更新层数。我们是在常规数据投毒 SFT 中观察形成，没有先把已植入后门刻意遮蔽。其互信息优化是攻击构造，我们的归因/状态交换是研究工具。

**相同：**低 ASR 不表示内部状态无风险，正常更新可与行为增强相关。**不同：**预先休眠→另阶段唤醒 vs 同一训练过程自然晚学；全网络/密集层选择不是 LoRA rank。**指导：**若宣称 clean update 会放大风险，需要排除历史毒梯度/动量，并与主动休眠先例明确区别。

### B03. FAB / Finetuning-Activated Backdoors in LLMs（同一论文的版本标题有变化）

[原文](https://arxiv.org/html/2505.16567v2)，§3–4.4、Appendix A.2.2。原报告使用 Watch your steps 标题；此处以可读取 v2 的标题识别，不计为新论文。

他们通过元学习直接优化发布模型，使干净下游微调激活广告/拒答/越狱；覆盖 Llama-3.2 与 Phi-2 等。**明确比较 LoRA 与 full finetuning**，并变动 LR/optimizer/scheduler；压力实验五次重复。我们研究常规数据投毒后的自然形成，后门输出仍需要图像/文本输入触发器；FAB 的“用户微调”本身是激活事件。

**相同：**现代生成式 LLM、微调期间行为从弱变强、LoRA、多次重复。**不同：**攻击者可元优化底座权重，主动保证下游唤醒；我们当前研究目标是解释固定训练程序中的形成动态。**指导：**不能写首次 LLM clean-FT activation 或首次 LoRA 相关激活。可研究无需元学习的自然形成是否共享/不共享其机制。

### B13. Persistent Backdoor Attacks Under Continual Fine-Tuning of LLMs（AAAI 2026）

[原文](https://arxiv.org/html/2512.14741v1)，方法、实验与附录。

他们在 Qwen2.5 和 LLaMA3 家族优化触发 token 梯度对齐，让已植入后门跨后续任务保留；比较 full-parameter、data replay、FREEZE。主指标之一是后续 ASR/初始 ASR 的 persistence ratio。我们的主问题是首次形成的等待与陡升，使用固定视觉触发器，不是为了持久性优化文本触发器。

**相同：**干净任务梯度与毒目标的关系、Qwen/LLaMA 系模型适配。**不同：**已学会后保留 vs 尚未显现时形成；token embedding 梯度对齐 vs 语言层 LoRA 的历史优化动态。**指导：**若持久性成为论文贡献，必须加发布后 task sequence；同一毒训练末端的高平台不能替代它。

### N03–N06. 四篇直接改变 LoRA 叙事的研究

| 论文 | 他们的设置与结论 | 与我们的相同 | 与我们的不同 / 写作后果 |
|---|---|---|---|
| N03 [Defending Against Weight-Poisoning Backdoor Attacks for Parameter-Efficient Fine-Tuning](https://aclanthology.org/2024.findings-naacl.217/)，NAACL Findings 2024 | 文本分类，五种适配策略、三类 weight-poisoning；PEFT 下底座后门更难清除，以置信度识别毒输入 | PEFT/LoRA 与后门风险 | 起点已经是毒底座，后续适配考察残留；我们从干净预训练底座用毒 SFT 数据形成。不是它的直接反例。参数细节仅摘要级，未填 rank |
| N04 [Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?](https://proceedings.mlr.press/v267/liang25f.html)，ICML 2025 | BERT-large/GLUE；LoRA vs full FT、rank/初始化；所测数据后门下 LoRA 更鲁棒。NTK/信息几何分析 | 同为训练数据投毒、LoRA 适配；不是只有 persistence 的远邻 | 分类目标、r=8 默认、LR按方法不同，五次重复；我们的生成式跨模态骤升不是其终态鲁棒性结论。固定 LR 与各自调优两种公平性需分开 |
| N05 [Why LoRA Fails to Forget](https://arxiv.org/html/2601.06305v1)，2026 | 已投毒 BERT/RoBERTa/LLaMA 的干净适配；强调更新谱强度/方向而不只 rank，RoRA 改善清除。r=8/16 | LoRA 更新谱与后门可能相关，LLaMA设置与我们更近 | 研究遗忘，不是形成。只扫 rank 不足以证明低秩瓶颈；需记录实际 ΔW 的奇异值、方向及 scale |
| N06 [LoRATK: LoRA Once, Backdoor Everywhere in the Share-and-Play Ecosystem](https://arxiv.org/abs/2403.00108)，2024初稿/后续改名 | 冻结底座、恶意 LoRA 分发、组合与迁移；早期稿 LoRA-as-an-Attack 包含无训练注入 | 后门可以存在于 adapter；生成式 LLM | 供应链攻击/插拔后是否触发，不是训练中晚期骤升。不能写首次发现 LoRA 可植入后门；可写 adapter 的形成时机及干预 |

N03/N05“LoRA 清不掉”和 N04“LoRA 较抗数据后门”不矛盾：**前者保护的是受污染底座中的旧关联，后者限制的是新关联形成。** N02 又说明形成慢并不等于植入后的寿命短。论文应按阶段拆开比较。

### D01 / D03 / D06–D07. 必须面对的快学反例

| 论文 | 他们研究什么 | 相同和真正不同 | 对我们最直接的要求 |
|---|---|---|---|
| D01 [Anti-Backdoor Learning](https://arxiv.org/abs/2110.11571) | 常见图像后门的 poison-train loss 快降；早隔离后反学习 | 都是毒训练动态；他们的快拟合不是我们的 held-out conditional ASR，更不是未见触发器泛化 | 同时报告 seen-poison loss/accuracy 与 held-out ASR；早拟合、晚泛化可以共存 |
| D03 [On the Difficulty of Defending Contrastive Learning against Backdoor Attacks](https://www.usenix.org/system/files/usenixsecurity24-li-changjiang.pdf) | 对比学习下干净/毒 loss 不像监督学习那样快慢分离 | 都影响防御迁移；CLIP式对齐训练与冻结视觉编码器后的自回归 SFT 目标不同 | 不能因为都叫 vision-language 就当同一 VLM；无需照搬其 loss 指标 |
| D06 [BackdoorBench 2022](https://arxiv.org/abs/2206.12654) | 多种图像攻击下 held-out poison loss/ASR也可能快速形成；附录B.6 | 已涉及测试行为，不能只以“他们只看 train loss”绕开 | 加至少一组早学条件，解释为何我们配置晚学；当前没有证明后门普遍都晚学 |
| D07 [BackdoorBench 扩展](https://arxiv.org/abs/2407.19845) | 同一基准扩展，动态随模型/数据/毒量变化 | 不是新的独立复现群体；生成式输出和适配方式不同 | 引用扩展，但不能把两个版本重复计为两组证据 |

### D05 / D08 / D09. 后门机制理论与内部动态

| 论文 | 他们的对象/方法 | 相同 | 不同与研究后果 |
|---|---|---|---|
| D05 [Theoretical Analysis in CNNs](https://proceedings.mlr.press/v235/li24at.html) | 两层CNN/多视图数据，理论分析语义—触发学习竞争 | 竞争、毒量与触发强度会影响形成 | 我们是预训练 transformer+低秩更新，不能直接引用为证明；需要可检验预测，不只类比 |
| D08 [Information Bottleneck Backdoor Dynamics](https://arxiv.org/html/2511.21923v1) | CIFAR-10/SVHN，ResNet18/VGG16，为MI估计加入噪声，比较攻击信息轨迹与隐蔽性 | 在ASR之外测内部训练动态 | 骨干/训练目标/估计仪器不同；我们的归因不等于其MI，MI轨迹也不自动是因果前兆。若做内部特征必须承认其先例 |
| D09 [Why Backdooring Neural Networks is so Easy?](https://arxiv.org/abs/2609.36117) | 二次神经元、Gaussian输入的限制理论；攻击条件随训练机制改变 | 给强度/毒量/训练机制的因果候选 | 摘要级，不填LoRA与复杂模型配置；没有建立我们的checkpoint机制 |

### T01–T05. 文生图方向逐篇比较

| 论文 | 他们的模型/攻击/时间轴 | 与我们的相同 | 关键不同与论文位置 |
|---|---|---|---|
| T01 [BadDiffusion](https://openaccess.thecvf.com/content/CVPR2023/papers/Chou_How_to_Backdoor_Diffusion_Models_CVPR_2023_paper.pdf) | 无条件扩散为基础，改变diffusion训练过程，触发噪声→固定目标图 | 图像生成模型也可出现条件攻击输出 | 不等于正常T2I图文数据投毒；本轮没有核验严格晚期骤升。不能统称他们也是文本触发 |
| T02 [Trojan Horses / Bilateral Data Poisoning Effects](https://proceedings.neurips.cc/paper_files/paper/2024/hash/95dcc1f6463491d37a8918c1d38380a7-Abstract-Conference.html) | 研究不同poison ratio造成的生成效果与可利用的触发放大 | 投毒可能有阈值与阶段差异 | 关键轴是剂量，不是训练checkpoint；参数更新范围本轮未充分确认，不擅称full FT |
| T03 [SilentBadDiffusion](https://arxiv.org/abs/2401.04136) | Stable Diffusion v1.4及版本比较；图文数据投毒造成版权目标复现；常规Diffusers管线、AdamW恒定LR、100epochs；CIR/FAE/SSCD | 预训练生成式模型、数据攻击、训练等待时间；T2I最应直接比较的先例 | 我们新T2I具体模型/更新模块/目标尚未绑定，不能现阶段硬写“我们LoRA他们full”。其论文不以统一grokking为结论，连续相似度平台/首次命中不等于我们的多seed陡升 |
| T04 [Diffusion Grokking on Modular Addition](https://arxiv.org/abs/2604.17673) | 算术图像/flow matching的良性延迟泛化，分析采样过程关键时间步 | diffusion也有延迟泛化 | 不含后门；采样denoising timestep与训练checkpoint不同，不能用来占据T2I后门机制结论 |
| T05 [TooBad](https://arxiv.org/html/2606.23362v1) | DDPM/LDM/NCSN，主要噪声触发器；先冻结模型优化trigger，再微调注入，Fig.6为训练epoch的ASR/MSE/SSIM；三次运行平均 | 形成时间、连续/离散指标并测；trigger选择影响训练速度 | 优化触发器主动加速，不是自然late grokking；冻结优化trigger阶段不是LoRA。平均曲线不证明单seed跃升一致。T2I应增加固定/优化trigger对照 |

这五篇均不能被写成“文生视频先例”。本轮核心近邻主要是图像生成、图像分类、文本生成和算术任务；没有核验到对应的文生视频训练晚期后门骤升结果。没有找到不等于不存在。

### N07. Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift（CVPR 2025）

[原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_Revisiting_Backdoor_Attacks_against_Large_Vision-Language_Models_from_Domain_Shift_CVPR_2025_paper.pdf)，§3–5。原始PDF已保存。

他们在 OpenFlamingo、BLIP-2、Otter 图像描述/指令微调中比较域迁移后的后门效果，提出利用归因放置触发器的 MABA。**相同：**生成式VLM、图像/文本触发器、目标回答与低投毒比例。**不同：**其 dynamic 指域变化，主问题是跨图像风格/数据域的 ASR-G；我们的 dynamic 是固定评测输入沿checkpoint演化。其归因是构造攻击区域，我们用内部仪器研究形成过程。

参数更新细节在本轮主论文中未查明，不能仅凭这些模型惯常冻结部件就填成其实际设置。**写作：**承认跨VLM和归因攻击已有；将我们的差异写在时间轴、训练干预和固定触发器形成规律，避免宣称首次VLM动态研究。

### N08. BackdoorLLM（NeurIPS 2025）

[正式论文](https://papers.neurips.cc/paper_files/paper/2025/hash/20ffc2b42c7de4a1960cfdadf305bbe2-Abstract-Datasets_and_Benchmarks_Track.html)，[作者仓库](https://github.com/bboylyg/BackdoorLLM)。

生成式LLM的多攻击/多架构/多场景基准；作者数据投毒实现采用LoRA。**相同：**开权重生成式LLM、毒/净混合指令适配与目标输出。**不同：**标准化有效性和失败模式的基准不等于训练时间骤升机制研究；不同攻击的weight/hidden-state/CoT权限也不同。**指导：**LoRA、现代LLM、多模型本身已有；可复用其评测口径，不能仅以更多模型名称与其区分。没有在本轮确认它已研究我们同一checkpoint现象。

### N09. Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models（NDSS 2026）

[会议原始摘要与论文入口](https://www.ndss-symposium.org/ndss-paper/causal-guided-detoxify-backdoor-attack-of-open-weight-lora-models/)。本条核验到正式摘要，具体模型、rank和训练模块尚未回查全文。

**相同：**开权重LoRA模型、后门与因果分析。**不同：**CBA针对恶意adapter分发，在没有原始训练数据时合成任务输入，并按因果影响保留任务重要神经元、混合干净与恶意adapter；训练后可调攻击强度。研究目标是降低误触发并保留攻击，而我们关注普通投毒训练中后门何时首次形成。**作者结论：**六个LoRA模型上获得高ASR并降低误触发率；不是晚期跃升机制的确认性结果。**指导：**不能写“首次因果分析LoRA后门”；我们的干预要服务于时间形成机制，且需要正常任务影响和误触发的匹配对照。

### N10. VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models（NeurIPS 2023）

[正式论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b055b95d689b1f704d8f92191cdb788-Abstract-Conference.html)，[作者实现](https://github.com/IBM/VillanDiffusion)。论文提出覆盖无条件/条件扩散模型及不同采样器的统一后门框架。

**相同：**预训练文生图、文本触发器、目标图像；作者条件生成实现明确支持LoRA。仓库推荐Stable Diffusion v1.4、rank 4、cosine LR、50000训练步、每5000步存checkpoint，示例用“latte coffee”触发固定Hacker目标及干净先验保持。这是作者实现示例，不代表所有论文实验统一使用这些超参。

**不同：**其贡献是后门训练/采样的统一框架，并未在本轮确认相同的自然长平台、多seed骤升或跨VLM/LLM统一形成机制；文本触发生成图像也不同于我们当前VLM的图像触发生成答案。我们新T2I配置仍未绑定，不能硬比较rank与具体更新模块。**指导：**“首次文生图LoRA后门”已不能争取；应复用其文本触发设置并加密checkpoint，比较连续目标相似度与阈值ASR，判断我们观察的晚学是否超出既有攻击框架。

### P01. Testing Whether Internal Backdoor Precursors Precede Behavior（2026协议）

[原文](https://doi.org/10.21203/rs.3.rs-10594749/v1)，§4、§6及附录。

GPT-2/SST-2的主动marker协议，要求内部/行为检测功效匹配、matched-null与同一时间点抑制/恢复；它的两步smoke test只检验实现，不证明前兆成立。我们是现代VLM/LLM适配和自然数据投毒动态，已有多seed行为/连续指标，尚不能由归因先动直接推出公平在线预警。

**相同：**内部前兆与输出何时开始、因果干预、控制伪领先。**不同：**协议与确认性结果、marker与自然触发关联、玩具分类与生成式跨模态。**写作：**承认问题/框架先例，争取真实确认性结果；“输出敏感度更低”造成的领先必须排除。

## 4. 其余安全近邻：逐篇说明相同与不同

下表补齐B/D中的所有其余记录。未核验具体trainable modules的文章，比较只到其明确的攻击/任务阶段；“图像模型”不自动等于full FT，“语言模型”不自动等于LoRA。

| ID / 论文 | 相同 | 不同 / 对论文与实验的影响 |
|---|---|---|
| B01 [Latent Backdoor](https://arxiv.org/abs/1905.10447) | 发布时弱/无目标行为，后来出现 | 预先埋入潜在表示，下游目标类别/任务补全后激活；我们固定任务连续训练。把latent transfer与delayed acquisition分开 |
| B04 [Dormant Backdoor](https://ojs.aaai.org/index.php/AAAI/article/view/39480) | 未来微调使行为增强 | 数据无关代理、双层优化主动构造休眠；我们不做该设计。未知下游数据的供应链能力不同；没有依据称其LoRA或全量 |
| B05 [Superficial Safety](https://arxiv.org/abs/2410.09838) | 低ASR可能掩盖残存能力 | 起点是已植入并防御的模型，随后少量重投毒；我们首次形成。若写“修复”，必须复活测试 |
| B06 [Re-Activation Attack](https://arxiv.org/abs/2405.16134) | 低行为不保证风险解除 | 主要换触发器重新显露，参数不必再训练。推理trigger变化不是checkpoint学习；换trigger鲁棒性是必要附加评测 |
| B07 [UBA-Inf](https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui) | 隐藏关联可被解除抑制后显露 | 主动毒样本+camouflage，合法遗忘触发；不是自然SFT晚学。可借鉴能力与抑制路径的双向干预 |
| B08 [BadFU](https://arxiv.org/abs/2508.15541) | 前低后高ASR | 联邦遗忘请求移除camouflage；数据删除/聚合机制不同。不能用共同曲线宣布同一机制 |
| B09 [Clean-label Dormant-to-Activated](https://arxiv.org/abs/2607.27936) | 潜伏到激活 | 标签保持干净且有可移除伪装；我们的主设置改目标回答，是dirty-label。不是其clean-label威胁模型 |
| B10 [Sleeper Agents](https://arxiv.org/abs/2401.05566) | 条件行为、生成式LLM、后续更新风险 | 主要研究已学会后门经安全训练仍保留；条件年份不是“训练时间”。思维链/安全训练与我们普通短答案SFT不同 |
| B11 [Persistent Pre-Training Poisoning](https://arxiv.org/abs/2410.13722) | LLM参数投毒及跨阶段风险 | 从头预训练植入→SFT/DPO后保留；我们LoRA适配时形成。预训练语料污染、对齐、首次形成分别比较 |
| B12 [ICL Backdoor](https://arxiv.org/abs/2307.14692) | 参数训练后条件行为影响语言任务 | 推理时few-shot演示/提示变化下的后门；不是无参数的prompt injection，也不是我们的VQA checkpoint研究。clean FT可移除的条件可作反例 |
| B14 [DormMark](https://doi.org/10.1145/3817059) | 部署低响应、微调后恢复 | 所有权验证水印，主动植入+遮蔽+唤醒；是正向用途而非自然后门学习。若拓展应用需注明主动设计与威胁目标差别 |
| D02 [NONE](https://arxiv.org/abs/2202.06382) | slow poisoning使快学检测失效 | 几何/置信度防御，与我们的late ASR量化并非同一任务。不能写首次提出“慢学避检测” |
| D04 [SEEP](https://aclanthology.org/2024.tacl-1.55/) | 训练动态+内部表示用于安全判断 | 先找高精度毒样本再在latent space扩展；数据识别不是模型未来行为预警。需比较检测对象/输入权限/召回和成本 |
| D10 [Unlabeled Poisoning](https://arxiv.org/abs/2105.01622) | 污染训练数据改变后续预测 | SSL的instance-targeted攻击；单个目标输入误判不等于跨输入条件后门。作A01/A05背景 |
| D11 [DeHiB](https://ojs.aaai.org/index.php/AAAI/article/view/17266) | 无标签数据也能形成触发关联 | 对抗扰动、伪标签/对比训练条件，与显式VQA答案投毒不同。对照攻击者知识，不照搬其解释 |
| D12 [Filter, Obstruct and Dilute](https://arxiv.org/abs/2502.05755) | 破坏trigger与目标的学习关系 | SSL专用过滤/互补学习/mix-up；我们固定预训练视觉表示、语言SFT。需要改造后公平基线，不能直接报告不可运行就是无效 |
| D13 [COLLIDER](https://openaccess.thecvf.com/content/ACCV2022/papers/Dolatabadi_COLLIDER_A_Robust_Training_Framework_for_Backdoor_Data_ACCV_2022_paper.pdf) | 梯度与内部几何的训练筛查 | 梯度coreset数据选择，不是未来ASR在线前兆；LR milestone改变曲线提示调度混淆。若比较要计梯度计算成本 |
| D14 [Secure-CHG](https://arxiv.org/abs/2606.31066) | 后期防御失效的风险 | 联邦攻击/防御交互与梯度统计难度改变；不是固定SFT自然跃升。用作防御动态背景 |
| D15 [Catastrophic Overfitting Mechanism](https://arxiv.org/abs/2604.24350) | 训练途中突然出现坏行为，特征路径解释 | 快速对抗训练的鲁棒性崩塌；目标指标不是poison-ASR。若借鉴机制需配对鲁棒性与触发行为 |

## 5. 33篇grokking研究：各自和我们哪里重叠、哪里不能套用

共同背景差异：下列主要是良性任务；多为从随机初始化训练或限制理论，不是污染预训练生成式底座的LoRA研究。不能把每篇都写成“full finetuning对照”，也不能因为有错误标签就称条件后门。摘要级来源的具体层数、rank和优化器不在此补造。

| ID / 论文 | 与我们相同的研究问题或工具 | 关键不同；直接影响 |
|---|---|---|
| G01 [Survey](https://doi.org/10.1145/3814603) | 解释延迟与陡升、整理多机制 | 综述非独立实验；支持采用多解释而非一套理论包办 |
| G02 [Power et al.](https://arxiv.org/abs/2201.02177) | 已拟合训练集后仍迟迟未泛化 | 小算术数据、良性train–test；我们须增加毒训练例与held-out行为的时间分离才能严格叫grokking |
| G03 [Effective Representation Theory](https://arxiv.org/abs/2205.10343) | 表示逐步形成与阶段变化 | 限制模型/表示理论非LoRA底座；只借可检验预测 |
| G04 [Hidden Progress / Sparse Parity](https://arxiv.org/abs/2207.08799) | 输出不动时内部已有进展 | 梯度Fourier gap有任务特定理论，不是后门未知方向；我们的前兆须有同敏感行为对照 |
| G05 [Slingshot](https://arxiv.org/abs/2206.04817) | optimizer状态、norm与跃升关联 | 自适应优化器循环不是所有晚学必然机制；保存θ/m/v配对分叉 |
| G06 [Omnigrok](https://arxiv.org/abs/2210.01117) | 跨图像/语言/图任务的延迟泛化 | 跨域本身已有，非后门LoRA多模态；权重范数控制需在ΔW而非仅底座norm上检验 |
| G07 [Mechanistic Progress Measures](https://arxiv.org/abs/2301.05217) | 内部电路先形成、行为后出现；消融 | 算术Fourier结构可识别，输出前进展思路已有；我们没有天然已知基底，方向发现成本与未来信息需审计 |
| G08 [Two Circuits](https://arxiv.org/abs/2303.11873) | 两条路径竞争后换主导 | 稀疏/稠密parity网络不等于正常答案/trigger路径；需独立抑制与恢复而非归因份额 |
| G09 [Circuit Efficiency](https://arxiv.org/abs/2309.02390) | 慢学而高效率电路最终占优 | 正则选择的理论关系未在冻结底座+LoRA证明；efficiency需明确定义并干预，不能看norm就宣布 |
| G10 [Lazy to Rich](https://arxiv.org/abs/2310.06110) | 后期特征学习突破早期停滞 | 视觉底座冻结不等于全模型lazy：语言残差仍可变化；用具体模块冻结/线性化检验 |
| G11 [Early/Late Implicit Bias](https://arxiv.org/abs/2311.18817) | 参数化/初始化使早晚偏置不同 | 特定大初始化与小WD证明不能直接套A随机/B零LoRA；先匹配假设 |
| G12 [Grokking as Compression](https://arxiv.org/abs/2310.05918) | 行为前复杂度压缩 | LMN计算对象不同于attention/归因；检测成本和近似误差必须记账 |
| G13 [Linear Estimators](https://arxiv.org/abs/2310.16441) | 离散指标可把平滑过程显示成跳跃 | 线性可解模型无新表示仍可groks；连续margin加速约束纯阈值解释，但不能自动证明电路突变 |
| G14 [Corrupted Algorithmic Data](https://arxiv.org/abs/2310.13061) | 记忆与规则共存、后期清理 | 随机标签腐化没有trigger条件；训练准确率下降与我们ASR上升的事件不同 |
| G15 [First-order Transition](https://arxiv.org/abs/2310.03789) | 严格阶段转变及表示改变 | 两层teacher–student与理论极限；我们折线陡升不能直接叫热力学一阶相变 |
| G16 [Deep Networks Always Grok](https://arxiv.org/abs/2402.15555) | 真实图像任务与局部决策复杂度 | 良性任务，Always标题不是所有训练的保证；研究失败条件而非援引标题 |
| G17 [Real-world Progress Measures](https://arxiv.org/abs/2405.12755) | 激活稀疏度/熵/复杂度监控 | MNIST/IMDb等正常任务；候选指标需打败clean/sham和连续输出基线 |
| G18 [Grokfast](https://arxiv.org/abs/2405.20233) | 慢变梯度可能决定迟滞，可干预加速 | 正常泛化加速不等于后门抑制；若用作控制应测是否也加速ASR及效用 |
| G19 [Implicit Reasoners](https://arxiv.org/abs/2405.15071) | 结构泛化、OOD与内部电路 | 组合推理泛化不等于固定trigger→词映射；未见trigger与未见图像/问题是不同OOD |
| G20 [Information Interplay](https://proceedings.mlr.press/v235/song24g.html) | 表示和分类头协同改变 | 分类头信息指标不等于自回归词表竞争；语言读出需重新定义 |
| G21 [Numerical Stability](https://arxiv.org/abs/2501.04697) | logit尺度/精度可能改变晚期动力学 | 算术softmax退化与我们fp16训练不同但可检验；precision、logit scale控制必要 |
| G22 [Computational Glass?](https://arxiv.org/abs/2505.11411) | norm/熵/相变解释可被实验反驳 | 算术transformer、WanD等；不能只选支持自己机制的G15，需讨论模型假设差异 |
| G23 [Dropout, Embeddings, Activation](https://doi.org/10.1007/s11063-026-11843-4) | dropout/embedding/梯度维度动态 | 我们LoRA dropout与语言embedding是否可训要明确；不把普通embedding学习当我们视觉学习 |
| G24 [Ridge Regression](https://arxiv.org/abs/2601.19791) | 可证明的延迟泛化与参数依赖 | 线性无底座适配/后门；再次要求避免“late jump必需新特征”的唯一解释 |
| G25 [Systematic Empirical Study](https://arxiv.org/abs/2603.25009) | 架构/深度/正则影响延迟 | 结构和超参存在交互；我们Qwen/LLaVA差异要分数据、LR、模块与架构 |
| G26 [Conditional and Fragile](https://arxiv.org/abs/2607.05104) | 多seed发生率、计算环境 | 12K参数良性模型与7B适配不同；仍要求报告失败轨迹，不做成功seed选择偏差 |
| G27 [Representation Geometry Control](https://arxiv.org/abs/2607.11666) | 内部几何领先及按时机干预 | 算术/置换任务的几何方向不保证trigger特异；有用的干预需同时测正常质量 |
| G28 [Noise-driven Metastable Escape](https://arxiv.org/abs/2606.17120) | 等待时间、噪声跨障碍 | 故意构造的线性深网亚稳态；我们的seed差异本身不足证明随机逃逸 |
| G29 [Circuit Synchronization Precursor](https://arxiv.org/abs/2606.12966) | 内部同步领先，配对分叉改变发生时间 | 算术/频域基底、作者前兆主张；非在线未知后门确认结果。终态ceiling和事后方向不能偷看未来 |
| G30 [Monte Carlo Parameter Swapping](https://arxiv.org/abs/2608.01833) | 参数分叉/交换可改变泛化时间 | Monte Carlo/SAM-Swap参数操作不是我们的activation patching；干预对象和估计问题不同 |
| G31 [Scaling Laws / Phase Structure](https://arxiv.org/abs/2609.10657) | 量化onset、宽度及超参关系 | 算术MLP的384配置拟合；不能把其指数搬到大模型。我们可建立独立配置内预测并留出验证 |
| G32 [Anti-grokking / Weightwatcher](https://arxiv.org/abs/2602.02859) | 学会后可能失去，权重谱监控 | 良性测试泛化崩塌；观察窗内高ASR平台不证明永久。需要续训与谱信号独立验证 |
| G33 [Hierarchical Structure](https://arxiv.org/abs/2305.18741) | 域内饱和后结构泛化仍很晚 | 层级语法/结构泛化不是条件攻击；延迟不必骤升，分别定义等待和陡峭度 |

## 6. 相邻能力跃迁的逐篇比较

| ID / 论文 | 相同 | 不同 / 对我们的要求 |
|---|---|---|
| E01 [Induction Heads](https://arxiv.org/abs/2209.11895) | 组件形成与行为快速出现，内部因果分析 | 普通预训练中的ICL不是毒LoRA的后门形成；小模型因果证据不能升级为所有大模型结论 |
| E02 [Mechanistic Basis of Abrupt ICL](https://arxiv.org/abs/2312.03002) | 嵌套操作、多环节瓶颈、内部渐进外部骤升 | 人工in-context分类任务的架构/训练与VQA不同；借鉴分环节测量并做门控干预 |
| E03 [Emergent Abilities Mirage](https://arxiv.org/abs/2304.15004) | 非线性/离散评分可能造出台阶 | 主要横轴模型规模，不是checkpoint；不能把它当否定所有时间骤升，连续指标仍重要 |
| E04 [Weird Generalization is Brittle](https://arxiv.org/abs/2604.10022) | 模型/数据条件导致成功或失败 | weird generalization复现，不是A03种子时间曲线的专门复现；为跨模型稳定叙事提供反证边界 |
| E05 [Backdoor Self-Awareness](https://arxiv.org/abs/2510.05169) | 训练期间能力短窗口提升，可用于安全应用 | 强化学习/replay学的是找出自己的trigger，非后门首次形成；不能把其reward跃升当ASR跃升 |

## 7. 这些不同，哪些能成为贡献，哪些只算配置差异

| 我们与前人的差异 | 单独价值 | 必须增加的证据 |
|---|---|---|
| 使用LoRA | 普通配置差异；N01–N06/B03已占据大量前史 | LoRA/full FT/rank/module对照，正常质量与有效优化预算均可比；说明延迟究竟依赖什么 |
| 用现代生成式VLM/LLM | 有实际场景价值，但N07/N08/A03/B03已研究这些模型 | 共同固定协议、多seed分布、跨trigger/task、失败条件，超过换模型复现 |
| 跨VLM/LLM/T2I同形 | 可形成广泛经验现象，尚非统一机制 | 每域定义相同可比较事件；预先指定跨域预测和同一干预；T2I未完成不能提前计入结果 |
| 已见trigger的后门晚形成 | 与A03未见trigger归纳不同；与N01更接近 | seen毒例早拟合、held-out输入后泛化；区分新trigger/new input/新任务的OOD |
| 细粒度连续margin也加速 | 强于只展示离散ASR；能约束纯阈值解释 | 配对调度/precision/optimizer对照；不是自动证明特定电路或相变 |
| 自然数据投毒，无休眠元目标 | 区别B02–B04主动设计，可能降低威胁门槛 | 保持攻击者权限与标准SFT不变，证明无需mask/meta-learning；别用“自然”暗示投毒本身无设计 |
| 顺序影响何时，初始化×顺序影响是否 | 有机会建立条件规律 | 从已做交换扩展留出预测；不能从两个成功迁移宣称跨模型好顺序 |
| 在线内部预警并干预 | 强贡献方向，P01有协议但未确认 | past-only、matched-null、固定FPR/功效、未知trigger条件、未来风险与正常质量及总成本 |

最先应做的区分测试不是泛扫全部模型，而是：**同一底座/同一毒净数据和顺序，LoRA vs 全量或可比部分微调；同时记录正常任务质量、seen-poison拟合、held-out ASR及连续输出。** 这能决定我们应写“适配一般现象”还是“低秩适配的条件现象”。若rank变化主要同时推迟正常任务和后门，不能称后门特异机制。预算与模型大小另行估算，本轮未启动这些实验。

## 8. 论文的可用写法与应避免的写法

**当前可用的定位草案（研究目标，未完成项不能当结果写）：**

> 既有工作已在半监督分类、预训练语言模型适配和归纳式LLM后门中报告延迟学习，也利用学习顺序、低秩约束和早停降低后门风险。另一些研究主动设计在下游更新后激活或持久的后门。我们关注常规投毒适配中条件行为沿checkpoint的延迟陡升，比较现代生成式VLM/LLM中的发生率、连续输出加速与失败条件，并通过配对干预检验其成因。

这个段落的前史对应A01/A02/A03/N01/N02；主动构造对应B02/B03/B13。等T2I完成并绑定证据，再加入图像生成范围。未完成因果实验时，末句改为“提出并测试候选解释”，不写“揭示统一机制”。

| 避免写 | 更准确的写法 |
|---|---|
| 前人只做全量，我们首次LoRA | 已有PET/LoRA后门动态与风险研究；本文比较具体适配过程及参数更新约束 |
| 首次发现LLM后门grokking | 区分已见触发器的条件泛化与已有未见触发器归纳，研究现代适配中的发生条件 |
| 固定视觉编码器证明没有feature learning | 视觉入口表征冻结，但语言层可重新组织/读取视觉信息；机制定位在可更新路径 |
| 跨领域同形证明通用机制 | 跨任务重复出现为共同机制假说提供动机，是否共因由干预检验 |
| 普通LoRA低rank就能防御 | 低rank可能改变学习速度/植入质量/遗忘速度，需分阶段与匹配终态评估 |
| 内部图先变就能预警 | 先做公平敏感度与past-only检验，再报告有效领先和干预收益 |
| ASR高平台说明永久持久 | 在所测训练/续训窗口内保持；跨阶段持久性需另测 |

## 9. 核验与未决项

本文件覆盖旧79个编号及新增10个编号，并逐项比较相同/不同；核心方法细节重新回查了原始论文及本地配置。新保存Moderate-fitting、LoRA Lifespan、VLM Domain Shift三份PDF与文本；Moderate-fitting Fig.5整页进行了视觉核验。

尚未确认的具体参数更新方式已明确写出，不以惯例填补。尚缺新Qwen LLM/T2I原始路径，不能用早期T2I诊断项目替代这次正在进行的late-grokking实验。本文未核验到同类文生视频结果，不能人为制造该模态差异。综述/摘要级理论条目是方法近邻，不是89个同现象复现。

完整逐篇原结论与引用链见[REPORT.md](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/REPORT.md)。本比较由AI辅助检索与分析；所有“建议实验”均未在本轮执行，未证明全球零遗漏。
