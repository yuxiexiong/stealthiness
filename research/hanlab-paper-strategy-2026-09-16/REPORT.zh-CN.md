# Han Lab 论文策略拆解，以及 Verified Diagnosis 的成文判断

本次判断：这批论文支持我们继续研究“诊断是否可靠、是否帮助下一步决策”这个问题，但没有支持当前 G 方法已经足以成为一篇 solid 方法论文。当前最缺的是一个能被实验区分的核心洞察，以及它相对简单强基线的实际价值。单纯增加攻击、模型和场景，只会提高现有结论的可信度，未必补上贡献。

我愿意保留 Verified Diagnosis 作为问题方向；对当前“高精度 G + 基线失败 + 验证协议”的论文组合，建议先做一次能推翻核心主张的小规模实验，再决定是否扩大投入。达到 ICLR 级别的路径存在，尚未由现有结果兑现。

## 阅读范围与使用方式

实际目录是 `/Users/ruizhixu/Desktop/Han‘s Lab Publication`，按目录共有 **2022–2026 五年、62 份 PDF、1056 页**，年度数量为 10、9、12、21、10。已为每份文件建立“问题—痛点—朴素方法失效—洞察—贡献—关键证据—边界—对我们的启发”的卡片。

阅读覆盖每篇的摘要/引言、核心方法及关键实验和消融所在页；逐篇记录 `read_scope`。这是论文策略的证据阅读，不是1056页全部逐字通读，也不是代码复现。所有数字是本地论文报告值；关键主表按页核对，LPD和ForeAct的代表性页面另做视觉核对。

文件归档年份不必然等于首次预印本年份；会议版、期刊扩展、综述与回顾分开标注。62份文件不等于62项相互独立的新发现。原归档清单另有未取得的 SpAtten-Chip，不在本次PDF分析范围。所有文件哈希与原清单一致；源PDF没有改动。

- [62篇逐篇策略卡与原文链接](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/hanlab-paper-strategy-2026-09-16/ALL_PAPERS.zh-CN.md)
- [年度笔记：2022–2023](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/hanlab-paper-strategy-2026-09-16/year-notes/2022-2023.md)、[2024](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/hanlab-paper-strategy-2026-09-16/year-notes/2024.md)、[2025](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/hanlab-paper-strategy-2026-09-16/year-notes/2025.md)、[2026](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/hanlab-paper-strategy-2026-09-16/year-notes/2026.md)
- [可检索的完整卡片数据](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/hanlab-paper-strategy-2026-09-16/paper-cards.json)、[文件、来源、页数与哈希清单](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/hanlab-paper-strategy-2026-09-16/corpus.json)
- [本次采用的项目结果快照](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/hanlab-paper-strategy-2026-09-16/project-evidence.json)

## 这些论文究竟在解决什么问题

它们研究的是**在明确约束下，怎样让一个能力真正可用**。约束可以是256KB内存、长视频实时延迟、低比特准确率、机器人反应时间，也可以是统一模型中两项能力的冲突。题目很大时，核心实验往往很具体。

| 问题家族 | 用户付出的实际代价 | 有代表性的论文 | 常见贡献方式 |
|---|---|---|---|
| 低精度与小设备 | 模型装不下；量化后失准；训练无法运行 | MCUNetV3、SmoothQuant、AWQ、SVDQuant、QServe、COAT | 找出限制低精度的结构，改变表示或优化，再用系统实现兑现 |
| 稀疏计算与上下文利用 | 大量重复计算；算法减少FLOPs却没有变快 | SIGE、TorchSparse/++、DistriFusion、Quest、DuoAttention、LServe、XAttention | 找出可丢弃/可复用的信息，并控制筛选、通信和缓存代价 |
| 生成模型的质量与效率 | 高分辨率太慢；压缩后难生成；单一范式损害另一能力 | DC-AE系列、SANA系列、HART、VILA-U、LPD、DC-AR、DC-Gen | 分解不同职责，联合考虑表示、生成和执行的约束 |
| 长序列与持续交互 | 长上下文崩溃；记忆成本增长；训练与部署分布错位 | StreamingLLM、LongLoRA、LongVILA、Long-RL、LongLive、StreamingVLM | 用对齐部署过程的训练和有结构的记忆管理稳定长程行为 |
| 新能力的可靠执行 | 语言计划无法落地；动态任务来不及反应 | ForeAct、VLASH | 定位信息或时间错位，用更合适的输入与训练让动作正确发生 |
| 量子计算 | 噪声、连通性、调度和有限测量使理想算法失效 | QuantumNAS、QuantumNAT、QOC、QuEst、Atomique、Q-Pilot | 将硬件约束纳入训练/映射/调度，说明真实硬件与模拟的证据边界 |

这张表也解释了它们为什么题材不同却有共同的扎实感：痛点能被测量，方法改动对应痛点，最终收益能回到使用者身上。

## 最值得学的六种论文策略

### 1. 先证明哪一个环节造成损失，再对这个环节动手

SmoothQuant把“量化很难”进一步定位为激活中的通道离群值与权重量化难度不对称，提出保持原计算等价的尺度变换，将难度迁移到更容易处理的权重端。诊断观察直接决定了可实现的设计，而随后低比特准确率和速度检验设计是否有用。[SmoothQuant，PDF pp.3–4](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/SmoothQuant- Accurate and Efficient Post-Training Quantization for Large Language Models.pdf>)

MCUNetV3也没有停在“内存不够”。它分出量化梯度尺度、更新哪些张量、反向图如何执行三个问题：8任务平均准确率从FP32 SGD的73.3降到真INT8 SGD的64.9，QAS恢复到73.5；最后展示低于256KB的实际训练占用。[MCUNetV3，PDF pp.7–9](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/On-Device Training Under 256KB Memory.pdf>)

**对我们：**“某方法在我们协议下失败”还是现象。要继续问：失败来自排序信号、干预操作、供体语义、任务定义，还是评估器？至少要有一个实验让这些解释产生不同预测。

### 2. 观察性图表提出假说；主张策略有用，就要证明额外决策价值

LPD先从注意力的空间局部性提出两个原则：待生成token靠近已有上下文，同轮生成token彼此远离。真正有说服力的是相同模型的对照：随机顺序FID为2.11，两个原则单独使用分别为2.00和2.06，组合为1.92；再将质量与实际延迟一起报告。[LPD，PDF pp.9–10](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation.pdf>)

当我们主张图启发的策略更有用时，可以比较它与没有图、直接结果表、简单规则或强归因基线的实际价值。如果主张的是新的机制知识，则应以竞争解释的区分、可重复性和适用边界来论证，不必额外宣称策略更优。

DuoAttention与我们更接近：它用“限制一个头只能访问近期token和sink后，输出是否显著变化”来定义头的作用；再比较不同选头策略，最后让这个诊断决定哪些头保留完整历史。论文给出反向压缩顺序的破坏实验，以及多passkey训练相对attention profiling的对照。这里的诊断因具有可执行、经验证的含义而成立。[DuoAttention，PDF pp.2、4、10](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/DuoAttention- Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.pdf>)

**对我们：**G的图或说明若与answer-map给出同样可靠的预测，应把“图独有价值”留给独立实验。它还可能帮助阅读者选择下一项检查，但这个假说需要相同信息条件下的决策测试。

### 3. 用针对竞争解释的对照，保护核心洞察

StreamingLLM首先区分“初始token包含重要语义”与“初始位置承担attention sink功能”。在Llama-2-13B、PG19长文本、总缓存1024的设置中，纯滑窗PPL为5158.07，保留首四token为5.40，将首四token换成换行仍为5.60。这种替代控制直接削弱了“只是保留开头语义”的解释。[StreamingLLM，PDF p.5](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/Efficient Streaming Language Models with Attention Sinks.pdf>)

AWQ检验重要权重应该怎样选：OPT-6.7B的INT3设置中，同样保护1%权重，按激活选择的PPL为11.39，按权重幅值为22.37，随机为24.23，FP16为10.86。选取标准的价值由这些对照体现。[AWQ，PDF p.4](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/AWQ- Activation-aware Weight Quantization for LLM Compression and Acceleration.pdf>)

ForeAct把未来目标图像提供给VLA。它除了比较原始VLA与VLM规划，还问了一个危险问题：收益会不会仅来自任务描述更具体？在Pick_Tool中，语义文字为20.0、提供目标位置的空间文字为46.8，视觉目标加语义文字为93.4。空间文字对照承认了部分收益来自简化定位，同时表明图像条件仍有价值。[ForeAct，PDF p.9](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/ForeAct- Steering Your VLA with Efficient Visual Foresight Planning.pdf>)

这不是绝对排除全部混杂；ForeAct仍有机器人预训练和微调投入，实机主指标采用原子动作得分。但它把最容易想到的替代解释摆上了桌面。

**对我们：**最有力的对手往往是answer-map、直接查询、AtP及其强化版本。PurMM/CleanSight适配结果有价值，却不能替代这些直接竞争方案。测试一个本来就可能不适合该操作的外部方法，并不能单独证明新方法的必要性。

### 4. 简单改动也可以贡献很强，前提是抓住真正的错位

VLASH的问题精确到一个时间差：模型用推理开始时的机器人状态预测动作，动作却在推理结束后执行。它用期间已知的动作估计执行时状态，并用时移样本训练模型利用该状态。

H100上的最坏反应延迟由546.4ms变为46.4ms。更关键的是实机动态任务：20次乒乓回击中，同步、朴素异步和RTC均为0次成功，VLASH为11次。这个小设计改变了可完成的任务集合。[VLASH，PDF pp.3、7](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/VLASH- Real-Time VLAs via Future-State-Aware Asynchronous Inference.pdf>)

**对我们：**solid不要求复杂方法，也不要求每次都发明全新理论。诊断后更准确地排除错误解释、决定下一项检查或判断证据不足，都可以是下游价值；不必强行把本项目扩成参数修复。但价值需要实际测到。

### 5. 把方案自身的开销计入结果，并允许简单方案赢

SIGE中，朴素分块已经减少7.6倍MAC，却只快1.6倍；还需处理归一化与内核开销，才能达到2.9倍。FLOPs减少只是中间证据。[SIGE，PDF p.9](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models.pdf>)

TLT明确指出，投机解码接受更多token未必更快：更深草稿增加计算，大batch时甚至应不开启投机。它最终用端到端训练吞吐和保持相近的奖励曲线论证收益，并单列切换、更新、协调成本。[TLT，PDF pp.10–12](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/Taming the Long-Tail- Efficient Reasoning RL Training with Adaptive Drafter.pdf>)

MCUNetV3还报告通道按幅度选或随机选的差异不足0.2%，没有把无必要的精细排序强行保留为核心贡献。[MCUNetV3，PDF p.10](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/On-Device Training Under 256KB Memory.pdf>)

SVDQuant更明确地把新方法自己的代价公开：rank32低秩分支的朴素实现增加57%延迟，必须融合读写才能兑现整体收益。HART则提供一个质量代理的反例：两种解码设计的重建指标相近，最终生成FID却为21.98与5.30。[SVDQuant，PDF p.6](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/SVDQuant- Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models.pdf>)、[HART，PDF p.10](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/HART- Efficient Visual Generation with Hybrid Autoregressive Transformer.pdf>)

**对我们：**高precision、好看的图、某一步少做检查，都只是中间指标。信息获取、生成、校准、验证、供体准备及人的阅读时间应按实际任务计入；一次性离线费用和每病例费用分别报告。若简单方案同样好，应主动简化方法。

### 6. 系列研究每次追一个新瓶颈，允许上一代结论有边界

LongLive区分两个相反失败：保留缓存使新prompt迟迟不起作用，清空缓存损伤画面连贯。重新计算缓存改善两者权衡。其消融中重建缓存的CLIP分数27.87低于完全清空的28.95，但主体一致性94.04高于89.59；贡献是联合权衡，不能说所有指标都最高。[LongLive，PDF pp.3、8–9](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/LongLive- Real-time Interactive Long Video Generation.pdf>)

Fast-dLLM用推理近似快速加速；v2转而改训练与架构，使块扩散更适合缓存并保留AR能力。SANA-Video则将常量大小状态用于更长视频，同时补上时空建模和训练稳定性。每次延伸都有新的瓶颈，而非仅换模型重做同一张表。[Fast-dLLM-v2，PDF pp.4–9](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/Fast-dLLM v2- Efficient Block-Diffusion Large Language Model.pdf>)、[SANA-Video，PDF pp.5–9](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/SANA-Video- Efficient Video Generation with Block Linear Diffusion Transformer.pdf>)

DC-AE1.5还表明，多组件方法可以有清晰的共同机制：DiT-XL的gFID基线为26.44，只改潜空间结构为26.75，只加通道增强为36.83，匹配使用才达到17.31。组合的必要性要靠这种对照证明。相应地，VILA-U作者明确说尚未观察到理解与生成显著互相增强；统一形式并不自动带来协同。[DC-AE1.5，PDF p.7](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/DC-AE 1.5- Accelerating Diffusion Model Convergence with Structured Latent Space.pdf>)、[VILA-U，PDF p.10](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/VILA-U- a Unified Foundation Model Integrating Visual Understanding and Generation.pdf>)

**对我们：**从VLM转到文生图、从归因转到Verified Diagnosis，都应写清旧任务暴露了什么结构性限制，以及新任务怎样回答它。保留实验基础设施很有价值，但基础设施本身不能代替下一篇论文的科学增量。

## 这些策略也有边界，不能机械照抄

第一，许多论文依赖长期积累的模型、数据和系统。我们应该复制它们挑问题和做对照的方式，不必复制大规模预训练投入。

第二，各论文论证强度不完全相同。综述、回顾、期刊扩展与原始研究承担的功能不同；若干“最高倍数”只在特定硬件、batch、长度下成立。逐篇卡片保留了这些边界。用户选择的优秀论文集也不提供录用概率的无偏样本。

第三，Han的很多工作属于效率或系统研究，天然有延迟、显存这类硬终点。诊断方向的终点可以是更可靠的解释或更好的检查决策，但要将它变成可独立评价的对象。科学发现或基准论文可以成立，不必一定有最快的新算法；它们要证明新知识的重要性与广泛适用性。

ICLR的审稿要求关注原创性、严谨性、意义等维度，并不要求每篇都刷新SOTA。本报告以这些论文的论证质量作为参照，不把“几倍加速、几个模型”当作录用公式。[ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide)

## 按这个标准，逐条看我们的现有主张

本节冻结在已抓取的 `origin/main = a560578f9203f2d1436f1c4c23cc16499ccc5382`。结果来自已封存实验，本次没有新增GPU实验。44个abnormal单元来自22个场景；置信区间采用场景聚类。完整真值表没有被重新生成。

| 当前主张或资产 | 已支持的部分 | 按上述标准仍缺什么 |
|---|---|---|
| 归因分数/视觉显著性不能直接当作可靠诊断 | 当前实验确实暴露任务、操作与指标之间的不一致 | 这个大原则已有大量相关研究；必须提出新的、可区分的具体失效机制 |
| G能提出高精度changed claim | 场景平均changed precision为1.0 | answer-map也是1.0；高精度不是G独有，需同时看覆盖率和新信息价值 |
| G更适合发现可验证转移 | @4约44.51%，高于当前random约9.66% | AtP约66.86%，在同检查预算下更强；不能只和random或失败适配方法比较 |
| G与AtP结合能互补 | k=1时混合召回较好，是值得如实保留的现象 | 预注册k=2、4增益的95%CI都跨0；尚未通过混合增益判据，且需加入探针成本 |
| G或混合能节省诊断成本 | 已有可重放的成本账本 | G信息成本约为全表成本30.22%，AtP约2.04%，混合32.26%；这些是本实现信息成本比，不能偷换成总速度或普遍理论 |
| PurMM/CleanSight在新诊断任务中不可靠 | 某些投影/适配在该任务下效果很差 | 需要区分原方法失效与目标、粒度、干预操作不一致；强基线和原生操作对照必须在 |
| 第二攻击复现将提高贡献 | 能检验发现是否只依赖当前攻击 | 若仅复现“基线未超过所用null阈值、G高precision”，仍可在G没有额外决策价值时通过 |
| 真值表、预注册、拒绝过度归因、成本记录 | 都是扎实研究资产，减少自欺和选择性汇报 | 它们保证结论可信，不能自动创造新发现或有效方法 |

正式结果中，G和answer-map的changed recall分别约28.89%和28.98%，整体exact hit分别约76.56%和76.60%。这足以挑战“高精度诊断必须依赖G的额外表示”的论点，但不代表两者在每一种排序任务上等价。AtP属于排序方法，不能把其缺少声明字段的coverage记为0，再用来宣称诊断失败。[正式场景结果](https://github.com/yuxiexiong/stealthiness/blob/a560578f9203f2d1436f1c4c23cc16499ccc5382/attribution-visualization/visual-evidence-repair/results/diagnosis-truth-formal-2026-09-15/scene-summary.json)

混合相对AtP：k=2差值+1.52个百分点，95%CI约[-4.92,+8.33]；k=4差值-4.36个百分点，CI约[-11.93,+3.41]。这表示当前实验尚不能支持额外收益，不表示证明所有混合永远无效。[Stage A](https://github.com/yuxiexiong/stealthiness/blob/a560578f9203f2d1436f1c4c23cc16499ccc5382/attribution-visualization/visual-evidence-repair/results/toy-vd-2026-09-16/stage-a.json)

还要把**答案转移**与**正确事实恢复**分开：现有discovery recall统计的是协议内的不同转移关系；找到更多这样的关系，不自动说明找到真实病因，更不说明完成了参数修复。过去T2I中单token补丁难以稳定改变指定属性的负结果，尤其提醒我们先验证干预操作能表达待测试的假说。它支持当前操作需重审，不能推出整个文生图诊断方向不可行。

### 与已有工作的重叠必须正面处理

AtP*已经采用廉价近似排序、精确activation patching验证，并以获得verified recall所需成本评价，也提出残余假阴性诊断。因此“先排序再验证、统一算成本”本身不够构成我们的新贡献。我们当前代码中的AtP不等于已经完整比较了AtP*。[AtP*，§4.2及方法](https://arxiv.org/html/2403.00745v1#S4.SS2)

MIB也已将机制定位组织为可比较的评测问题。因此“我们终于有了机制解释基准”过宽；需要说明现有评测漏掉了何种有后果的错误，以及新测量改变了什么结论。[MIB，ICML 2025](https://proceedings.mlr.press/v267/mueller25a.html)

## 能否扩成solid论文：三个不同层面的回答

**问题价值：够。** 调试模型时，把一个能改变输出的操作误当作真实病因，会导致错误检查、错误定位与无效后续工作。这个痛点具有研究价值。现有实验说明我们的流程中确实存在这种风险；尚缺其普遍程度及实际代价的测量。

**当前预期贡献组合：还不够。** “提出更严格的protocol、G精度高、两个外部方法在新协议下未超过所用null阈值”尚未像上述论文一样解释为什么需要我们的关键设计。AtP与answer-map分别削弱了当前G的预算优势和声明独特性；扩大规模无法替代对这两点的解释。

**扩展潜力：有，但需要补一个核心发现，而非只补实验数量。** 我更看好保留诊断目标，将主张收紧到“有限检查下，如何区分不同的错误解释，并知道何时证据不足”。其新颖性还需要围绕具体机制继续排查；目前只是待检验的研究问题，不能宣称新方法已成立。

由此有两条诚实的成文路径：

| 路径 | 论文中心 | 必须补出的关键证据 | 当前判断 |
|---|---|---|---|
| 方法论文 | 某种已识别的结构使现有强方案浪费验证预算；新策略利用该结构 | 相同误诊风险下、更高的有用诊断覆盖或更低总成本；关键结构消融和边界 | 有潜力，当前G及简单混合未兑现 |
| 科学发现/评测论文 | 现有归因评估系统性混淆了有不同后果的解释；新测量揭示并预测这种混淆 | 独立真值、可区分的竞争解释、跨条件复现、原生与适配操作及强基线；实际检查后果可提供进一步意义证据 | 更贴近已有资产；仍缺普遍、重要且区别于现有工作的发现 |

如果最终只有“这个接口下几个方法得分很低”，不够。如果能说明一种反复出现的误诊机制，证明现有评估会奖励它，并给出可推广的识别或避免条件，即使方法很简单，也有solid乃至ICLR论文的潜力。录用和更高等级取决于最终证据与评审，不能由题目或论文模板保证。

## 先做一个窄实验，再决定如何扩展

**先检验一个竞争解释：所谓局部证据恢复，是否只是供体答案的带入。** 这是现有操作直接暴露的疑问，也是已知相关文献讨论过的大类问题，因此本实验先用于判断项目是否有独特结构可研究，不预先宣称它是新发现。

固定当前模型与一类错误，设计原图目标事实、供体目标属性、无关属性可以独立控制的病例。所有病例先通过操作正对照；未涉事实是否受损另行记录。训练/dev供体与确认供体分开。假如模型仅跟随供体语义，与真正恢复原图证据应在错配供体上作出不同预测。若这两种解释在本操作下仍无法区分，就应输出“不可识别”，先修改操作或问题定义。

**只检验一项主要方法主张：G是否改善预算内的可靠诊断覆盖。** 首个最强对照用已经胜出的“当前AtP排序 + 精确检查”；固定总预算和允许误诊率，比较有多少诊断在未见供体上仍成立，单列拒绝率和未涉事实损失。预算、阈值与规则在dev冻结，使用场景级独立确认。信息获取、供体准备和验证计入总成本，一次性费用另列。它是项目内最小裁决；如果出现优势，再扩展到适用的AtP*和自适应查询强基线，完成对外的方法新颖性与有效性论证。

这一步可能出现三个有价值的结果：G有额外价值；G没有优势但揭示了可重复、被旧指标掩盖的误诊机制；两者都未出现。前者支持方法路线，中者支持继续调查科学发现/评测路线，后者支持停下当前中心主张。不能在看完测试结果后才挑成功子集命名新机制。

**图的作用是另一项独立假说。** 若主张G的表示帮助阅读者，应在“相同查询结果”下比较图、文本和answer-map，测正确决策及耗时。这个实验失败，只否定表示的附加价值，不自动否定G的信息获取策略；反过来，查询策略有收益也不证明图有价值。参数修复无需加入本阶段。

确认核心现象后，再补跨模型、自然病例、第二种错误机制及原生/适配操作比较。科学发现路线重点检验竞争解释与适用边界；方法路线重点检验强对照下的效用—风险—总成本。这样规模扩展服务于已有发现，不替代发现本身。

## 建议我们保留与重写的内容

保留：诊断优先、明确可拒绝、供体与干预条件、独立验证、场景划分、成本账本、现有负结果。

重写：将“G高precision”“外部方法归零”从主贡献降为现象；在尚未证明新协议揭示了重要新问题前，把它作为支撑可信证据的研究资产；若协议改变了可信结论，它也可能成为评测贡献；把中心问题换成一个有明确消费者、有竞争解释、有可观测代价的诊断瓶颈。

可作为研究计划的一句话是：**面对相同的错误答案，现有归因何时无法区分具有不同处置含义的解释；怎样用少量、有效的检查得到可以采信或应当拒绝的诊断？**

这句话目前是问题，不是已完成的贡献。Han这批论文对我们最有价值的启发，是要求下一张关键实验图回答它。
