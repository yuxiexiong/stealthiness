# RELATED WORK — "JUMP" 侧：训练中的突变 / 相变 / 平台后逃逸

日期：2026-10-15。范围：对"LoRA 微调安装后门时 ASR 在极短窗口内 0→1 跳变；跳变前 margin 长期停在平台（LLaVA 约 -6.8），之后快速逃逸；投毒效率随 clean-task 能力提升约 20 倍"这一现象，梳理盟友文献已经观察到和已经得出的结论，以及我们可以直接拿来用的工具。

写作原则（按用户要求）：先行工作是原料，不是放弃这个方向的理由。下面每条都写清楚"他们证明到什么程度"和"他们没有分离出什么、我们可以分离"。

核实说明：
- 每条论断都附有我在本次检索中实际打开过的链接（arXiv 摘要页、HTML 全文、会议页面或作者页面）。
- 标 **[仅摘要]** 的条目：我只读了摘要或检索摘要，没有读正文。
- 标 **[未核实]** 的内容：我没能在原文中确认，引用前需要再查。
- 数字只写我在页面上看到的；看不到的不写。

---

## 0. 一页结论（先看这里）

1. **"突变是不是指标假象"已经有标准答法。** Schaeffer et al. 2023 的批评只适用于"对固定输出换一个非线性指标"这种情况。被广泛接受的反驳方式有三种：(i) 用连续的对数似然或 logit margin，看它是否本身也加速；(ii) 逐样本分析；(iii) 阈值扫描稳健性，Turner et al. 2025 对 EM 判定阈值就做了这一步。我们已有的"87–100% 样本自身 margin 加速、翻转点分散 8–10 logits"正好是 (i) 和 (ii) 的组合。还应补 (iii)，并补一个类似 PassUntil 的高分辨率指标。
2. **"为什么尖锐"在理论上有四类成熟解释。** 我们的"能力门控"说法必须与它们区分开，或者明确说明它是其中哪一类的具体实例：
   - (a) 乘积参数化的自催化增长，即 Saxe 的 `du/dt ∝ u(s−u)`，加上小初始化导致长平台；LoRA 的 B=0 初始化正好落在这个范式里。
   - (b) 两种解或两种隐式偏置之间的竞争与切换：Varma 的电路效率、Lyu 的早晚期偏置二分、Kumar 的 lazy→rich、Edelman 的 unigram→bigram。
   - (c) 嵌套或乘法组合：多个子能力的乘积，只有当所有子能力都到位时输出才跳。Okawa 的"乘法涌现"、Reddy 的三层嵌套 logit、Singh 的三个交互子电路、Abbe 的 staircase/leap 都属于这一类。
   - (d) 隐藏的连续进展在阈值处变得可见：Barak 的 hidden progress、Nanda 的 progress measures、Park 的 hidden capabilities。
3. **我们的说法最接近 (c)：后门任务的有效学习率被 clean 任务的进展所调制。** 文献里已有直接的同类证据：
   - Abbe 等的 staircase：低阶分量学会后高阶分量才可学；
   - Skill-it 的前置技能；
   - Singh 等 2025 "Strategy Coopetition"：ICL 只能借助另一策略的缓慢发展才出现；
   - Shejwalkar et al. ICCV 2023：半监督后门在模型对目标类有高置信之后才被安装，ASR 滞后于伪标签移动。
   但据我检索，**没有人在 LLM/VLM 后门里对"每条毒样本的推力随 clean 能力增长"做过定量测量和因果干预**。这是我们可以分离出来的部分。
4. **同场景的盟友。**
   - Turner et al. 2025 / Soligo et al. 2025 的 EM model organisms 是最接近的"微调中行为突变"工作：rank-1 LoRA、梯度范数峰、B 向量突然旋转、按比例放大 adapter 揭示潜在行为。
   - Arnold & Lörch 2025 给出了序参量分解，并指出行为转变晚于梯度范数峰。
   - 但 Rao et al. 2026 "An Emergent Mirage" 报告这些机制特征在训练中与行为不稳定相关。这是我们必须预先防守的复现性质疑。
5. **后门侧的直接先行工作。** Souly et al. 2025（近常数毒样本数）在微调中发现：毒样本放在训练开头时 20 条即可，放在末尾时 20 条不行；学习率越低需要越多毒样本。这些都是能力门控假说必须解释的事实。注意，"末尾不行"表面上与"能力越强效率越高"相冲突。我们需要用"LR 衰减 + 后续 clean 覆盖"或我们自己的数据来调和。

---

## 1. 涌现是幻象吗？指标阈值论证与反驳

### 1.1 Wei et al. 2022, *Emergent Abilities of Large Language Models* (TMLR)
- 链接：https://arxiv.org/abs/2206.07682
- (a) 设定：跨模型规模比较 few-shot 任务表现。
- (b) 观察：表现在某个规模阈值之前接近随机，之后显著高于随机。
- (c) 结论：给出"涌现"的定义，即小模型没有、大模型有、无法从小模型外推。这是现象描述，没有机制。
- (d) 我们采用：只借用它的定义框架。我们的"涌现轴"是训练步而不是规模，应在文中明确这一点。
- (e) 未分离：没有任何机制。

### 1.2 Schaeffer, Miranda, Koyejo 2023, *Are Emergent Abilities of LLMs a Mirage?* (NeurIPS 2023)
- 链接：https://arxiv.org/abs/2304.15004
- (a) 设定：固定的模型输出，改变评测指标。
- (b) 观察：非线性或不连续指标（如 exact match）产生"涌现"，线性或连续指标给出平滑可预测的变化。根据 Stanford HAI 的报道（https://hai.stanford.edu/news/ais-ostensible-emergent-abilities-are-mirage），29 个指标中 25 个不显示涌现。
- (c) 结论：许多涌现源于研究者的指标选择。注意他们的论证对象是**跨规模的固定输出**。他们并没有证明连续量本身不会加速。
- (d) 我们采用：把他们的批评当成**必须通过的检验**，在论文中显式回答。
  - 报告目标对正确的 logit margin 和 target token 的 log-prob。这两个是连续量，不是 ASR。
  - 证明连续量本身在跳变窗口内有加速（二阶差分为正、斜率突增），而不只是"线性移动穿过阈值"。
  - 用逐样本 margin 轨迹证明加速不是"异质阈值的平均"造成的。我们的 87–100% 自身加速正是这一点。
- (e) 未分离：他们没有讨论**训练时间轴上**的突变。我们的平台 → 逃逸是连续量本身的非线性，可以明确区分出来。

### 1.3 Du, Zeng, Dong, Tang 2024, *Understanding Emergent Abilities of Language Models from the Loss Perspective* (NeurIPS 2024)
- 链接：https://arxiv.org/abs/2403.15796v2 ；https://proceedings.neurips.cc/paper_files/paper/2024/hash/5f1eee2509599faeeb3570a887016a64-Abstract.html
- (a) 设定：不同规模和数据量的预训练模型，以预训练 loss 为横轴。
- (b) 观察：相同预训练 loss 对应相同的下游表现。某些任务在 loss 低于阈值之前停在随机水平，**且与指标是否连续无关**。
- (c) 结论：把涌现重新定义为"在更低预训练 loss 下出现的能力"。证据是跨模型的相关性，没有因果干预。
- (d) 我们采用：这是"能力门控"的宏观版本，即下游能力由一个通用能力变量门控。我们应当用 **clean-task loss（或 clean margin）作为横轴**画后门推力和 ASR，检验是否有类似的"阈值 clean loss"。如果不同 LR、不同 seed、不同投毒顺序的曲线在 clean-loss 横轴上塌缩到一条，这就是对门控最直接的现象学支持。
- (e) 未分离：没有因果干预，也没有解释为什么低于阈值才行。我们的删毒/重放实验可以做到因果层面。

### 1.4 Hu et al. 2024, *Predicting Emergent Abilities with Infinite Resolution Evaluation* (PassUntil, ICLR 2024)
- 链接：https://arxiv.org/abs/2310.03262 ；https://iclr.cc/virtual/2024/poster/17945
- (a) 设定：小模型在 code generation 等任务上的极低通过率。
- (b) 观察：PassUntil 持续采样直到通过 r 次，从而分辨极低成功率。即使用高分辨率指标，仍观察到"加速涌现"：曲线无法被标准 scaling law 拟合，且增长速度越来越快。他们提出"multiple circuits hypothesis"作为可能解释。
- (c) 结论：涌现与"性能连续改进"并不矛盾。连续，但加速。
- (d) 我们采用：
  - 在平台期用**高分辨率 ASR**，即 target 序列的精确对数似然或贪心解码 + 大量采样的极低命中率，证明平台期不是"测不到的 0"，而是真的停滞，或者真的在缓慢积累。这能把我们的情况落到 "hidden progress" 或 "genuine plateau" 之一。
  - 他们"连续但加速"的表述可以直接借用。
- (e) 未分离：没有回答加速从何而来。

### 1.5 Arnold & Lörch 2025, *Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment* [仅摘要]
- 链接：https://arxiv.org/abs/2508.20015
- (a) 设定：在窄域有害数据上微调，产生 EM。
- (b) 观察：用分布变化检测方法定位微调中的快速转变。用自然语言描述、由 LLM judge 打分的"序参量"（如 alignment、verbosity）分解总的分布变化，量化每个方面占总变化的比例。**实际的行为转变发生得比仅凭梯度范数峰所预期的更晚。**
- (c) 结论：给出可自动发现序参量的框架。具体数字我没看到正文。
- (d) 我们采用：
  - 用**统计不相似度**（如输出分布在相邻 checkpoint 间的 KL/JS 或其他分布距离）作为无指标偏见的"转变检测器"，与 ASR 跳变点对齐。这样转变点的定义不依赖 ASR 阈值。
  - 把总输出分布变化分解为"后门方向"和"clean 方向"两部分，回答"跳变时变的只是后门吗"。
  - 他们"梯度范数峰早于行为转变"的发现提示我们检查：毒样本梯度在后门方向上的投影是否在 ASR 跳变**之前**就已经上升。
- (e) 未分离：没有回答为什么在那一刻转变，也没有因果数据干预。

### 1.6 Barak et al. 2022, *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit* (NeurIPS 2022)
- 链接：https://arxiv.org/abs/2207.08799
- (a) 设定：k-sparse parity。
- (b) 观察：loss 在约 n^O(k) 次迭代后突然下降。
- (c) 结论：突变不是 SGD "在黑暗中乱撞"直到撞到答案。SGD 通过总体梯度中的 Fourier gap 逐渐放大稀疏解，持续在进步，只是 loss 和 error 看不出来。这一结论有理论证明。
- (d) 我们采用："hidden progress"是对"平台期什么也没发生"的标准反驳。我们需要一个**在平台期也能读出进展的内部量**，例如 LoRA ΔW 在最终后门方向上的投影、毒样本梯度与最终后门方向的余弦。然后判断：
  - 平台期这个量是在指数式缓慢增长（Barak 型）；
  - 还是基本不动，直到 clean 能力达到某水平（门控型）。
  这是区分两种机制的关键图。
- (e) 未分离：parity 中没有"另一个任务"可以门控，进展完全来自同一任务的自我放大。

### 1.7 Michaud et al. 2023, *The Quantization Model of Neural Scaling* (NeurIPS 2023)
- 链接：https://arxiv.org/abs/2303.13506v3
- (a)/(b) 设定与观察：在 multitask sparse parity 和 LLM 上，用梯度对 token 聚类以发现"quanta"。总 loss 平滑下降，单个 quantum 离散开启。
- (c) 结论：平滑的聚合 loss 与突然的单项能力可以共存。LLM 部分作者自称是 tentative evidence。
- (d) 我们采用：**逐样本、逐 token 分析是标准做法**。我们的逐样本翻转点分布（8–10 logits 的分散）可以按 quanta 的方式呈现：每个样本有自己的开启时刻，ASR 是开启时刻的经验 CDF。如果 CDF 的陡峭程度远超"线性移动 + 异质阈值"所能给出的陡峭程度，就说明存在真正的共同加速。
- (e) 未分离：没有说明单个 quantum 为何突然开启。

---

## 2. Grokking 与延迟泛化：为什么转变是尖锐的

### 2.1 Power et al. 2022, *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*
- 链接：https://arxiv.org/abs/2201.02177v1
- (b) 观察：在小算法数据集上，模型拟合训练集很久之后，泛化才从接近随机跳到接近完美。训练集越小，所需优化步数越多。
- (c) 结论：现象报告，没有机制。
- (d)/(e) 现象原点。我们与它的区别是：我们的跳变发生在**训练目标本身**（毒样本 loss），而不是 test。

### 2.2 Nanda, Chan, Lieberum, Smith, Steinhardt 2023, *Progress Measures for Grokking via Mechanistic Interpretability* (ICLR 2023)
- 链接：https://arxiv.org/abs/2301.05217 ；全文 PDF https://arxiv.org/pdf/2301.05217
- (a) 设定：1 层 transformer 学模加法。
- (b) 观察：逆向出 Fourier/三角恒等式算法。定义 **restricted loss**：只保留关键频率，测"仅用最终算法"的性能。定义 **excluded loss**：只去掉关键频率，在训练集上测"不依赖最终算法"的性能。训练分为 memorization → circuit formation → cleanup 三个连续阶段。
- (c) 结论：grokking 不是突然的转变。它源于结构化机制在权重中逐渐加强，随后记忆成分被移除。证据是机制性的，并有消融。
- (d) 我们采用，这是**最该直接移植的工具**：
  - **Restricted 后门 margin**：把 LoRA ΔW 投影到最终后门方向，例如最终 ΔW 的 top-k 奇异子空间，或用最终 checkpoint 估计的后门方向 v。在每个 checkpoint 上只保留这一投影，计算毒样本 margin。它衡量"后门机制本身已经长了多少"。
  - **Excluded 后门 margin**：去掉该投影，看剩余 ΔW 对毒样本 margin 的贡献，用来衡量"非后门路径"。
  - 如果 restricted margin 在平台期就在稳步增长（Nanda 型），跳变就主要是"清理/读出"问题；如果它在平台期也停滞、在 clean 能力上来后才增长（门控型），就支持我们的假说。
- (e) 未分离：模加法里没有第二个任务，不涉及跨任务门控。

### 2.3 Liu, Kitouni, Nolte, Michaud, Tegmark, Williams 2022, *Towards Understanding Grokking: An Effective Theory of Representation Learning* (NeurIPS 2022)
- 链接：https://arxiv.org/abs/2205.10343v2
- (b) 观察：泛化来自结构化表征。按超参数可以画出四相：comprehension、grokking、memorization、confusion。grokking 相比 comprehension 更靠近 memorization，因此延迟。
- (c) 结论：在玩具设定里，有效理论可以预测表征的动力学和对数据量的依赖。
- (d) 我们采用：**相图**。以 (LR, 毒率) 或 (LR, clean 数据量) 为轴，标出"永不安装 / 延迟安装 / 立即安装"三相。我们已有的"LR 减半延迟跳变"是相图上的一个切片。
- (e) 未分离：没有跨任务耦合。

### 2.4 Liu, Michaud, Tegmark 2023, *Omnigrok: Grokking Beyond Algorithmic Data* (ICLR 2023)
- 链接：https://arxiv.org/abs/2210.01117v2
- (b) 观察：train loss 对权重范数呈 "L" 形，test loss 呈 "U" 形（LU 机制）。据此可以在图像、语言、分子任务上诱导 grokking，也可以消除算法数据上的 grokking。
- (c) 结论：grokking 由 train/test loss 景观的不匹配加上权重范数动力学引起。算法数据上的剧烈程度来自表征学习。
- (d) 我们采用：检查 **LoRA ΔW 的范数**在跳变前后的轨迹，排除"只是范数缓慢增长到某值"这种平凡解释。Turner 2025 发现 B 向量范数平滑增长、方向突变，可以作对照。
- (e) 未分离：没有讨论微调或 LoRA。

### 2.5 Varma, Shah, Kenton, Kramár, Kumar 2023, *Explaining Grokking through Circuit Efficiency*
- 链接：https://arxiv.org/abs/2309.02390v1
- (a) 设定：1 层 transformer，AdamW，模加法 P=113，附录另有 9 个任务。
- (b) 观察：存在记忆电路和泛化电路。泛化电路学得慢，但"更高效"：相同参数范数下输出 logit 更大。记忆电路的效率随训练集增大而下降，因此存在临界数据量。他们做了 4 个新预测，并发现 ungrokking（测试精度从完美退回低值）和 semi-grokking（延迟出现但只达到部分精度）。
- (c) 结论：grokking 是"慢而高效的电路"最终在权重衰减下胜出。论证由预测和验证支持。
- (d) 我们采用：
  - **"效率"的定义方式**：每单位参数范数产生多少 margin。我们的"每条毒行在最终后门方向上的推力"本质上就是一个效率量。可以把它改写为"每单位 ΔW 范数增量带来的毒样本 margin 增量"，与 Varma 的定义对齐，便于审稿人理解。
  - **ungrokking 类实验**：跳变之后改为纯 clean 训练，看 ASR 是否回落。Souly 2025 已经报告微调中继续 clean 训练会使 ASR 降到接近 0，可以与之对照。
- (e) 未分离：两个电路服务**同一个任务**。我们的情况是两个任务（clean 与 backdoor），且一个增强另一个，方向与 Varma 的竞争关系相反。

### 2.6 Kumar, Bordelon, Gershman, Pehlevan 2024, *Grokking as the Transition from Lazy to Rich Training Dynamics* (ICLR 2024)
- 链接：https://arxiv.org/abs/2310.06110v1 ；https://iclr.cc/virtual/2024/poster/17515
- (a) 设定：两层网络做多项式回归，另有 MNIST、1 层 transformer、师生网络。
- (b) 观察：模型先在初始特征上做核回归，几乎不改变特征（lazy）；之后才通过特征学习找到泛化解（rich）。延迟泛化需要三个条件：初始 NTK 的顶部特征向量与标签不对齐；数据量足以最终泛化，但又没有大到让 test 紧跟 train；训练起始于 lazy 区。
- (c) 结论：特征学习速率（输出缩放参数）和初始特征与目标的对齐程度是两大控制因素。有理论和实验支持。
- (d) 我们采用，与我们的"20×"最相关：
  - **测量毒样本梯度与后门方向在 eNTK 意义下的对齐度如何随 clean 训练变化**。如果 clean 学习改变了 LoRA 的特征或 kernel，使毒样本梯度更对齐最终后门方向，那么"效率随 clean 能力增长"就是 Kumar 式"特征对齐"在多任务中的推广。
  - 可以计算 Ren & Sutherland 2025 式分解中的 kernel 项（见 3.12）。
- (e) 未分离：他们研究的是单任务内"初始特征 vs 目标"的对齐。**"另一个任务的学习改变了本任务的对齐"**没有被研究过，这是我们的机会。

### 2.7 Lyu, Jin, Li, Du, Lee, Hu 2024, *Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking* (ICLR 2024)
- 链接：https://arxiv.org/abs/2311.18817 ；https://proceedings.iclr.cc/paper_files/paper/2024/hash/909c8fef63e1cede406ce9e6794f99a2-Abstract-Conference.html
- (b)/(c) 对大初始化、小权重衰减的齐次网络给出证明：训练长时间困在核预测器解附近，然后**非常尖锐地**转向 min-norm/max-margin 解。作者强调，此前没有工作定量解释为什么转变是尖锐而非渐进的。作者自承局限：只分析了大初始化加小权重衰减这一种来源；不研究如何减少延迟。
- (d) 我们采用：理论叙述上，"困在一个解附近很久，然后尖锐切换"是被严格证明过的机制类别，平台的长度由早期偏置的强弱决定。我们可以把"后门 margin 平台"对应为"后门方向上的有效驱动力过小时停留在近 B=0 解附近"。
- (e) 未分离：切换由同一任务的隐式偏置驱动，与第二任务无关。LoRA 的 B=0 是**小初始化**，与他们的大初始化相反，更接近 Saxe/Jacot 一派（见 3.6–3.8）。

### 2.8 Rubin, Seroussi, Ringel 2024, *Grokking as a First Order Phase Transition in Two Layer Networks* (ICLR 2024) [仅摘要]
- 链接：https://arxiv.org/html/2310.03789v3
- (b)/(c) 用自适应核方法分析。grokking 之后的状态类似一级相变后的混合相：前激活统计由高斯变为高斯混合，网络形成与此前显著不同的表征。
- (d) 我们采用：可以检验跳变前后后门相关层激活的分布是否出现双峰或混合，作为"一级相变"的签名。这是可选的补充证据。
- (e) 未分离：没有讨论多任务门控。

### 2.9 Prieto, Barsbey, Mediano, Birdal 2025, *Grokking at the Edge of Numerical Stability* (ICLR 2025) [仅摘要]
- 链接：https://arxiv.org/abs/2501.04697v1
- (b)/(c) 过拟合之后，梯度主要沿"naïve loss minimization"方向：不改变预测，只放大 logit。这会推向 Softmax Collapse。他们提出 StableMax 和 ⊥Grad。
- (d) 我们采用：提醒我们把毒样本梯度分解为"沿当前 ΔW 方向的缩放分量"和"正交分量"，确认跳变时不是单纯的 logit 缩放。
- (e) 不直接相关，但能防一个审稿质疑："跳变只是数值或缩放效应"。

### 2.10 Li, Fan, Zhou 2026, *(Where to find) Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test* (ICLR 2026) [仅摘要]
- 链接：https://arxiv.org/abs/2506.21551 ；https://iclr.cc/virtual/2026/poster/10008581
- (b)/(c) 在 7B OLMoE 的单遍预训练中，grokking 仍然发生，但它是**局部的、按数据组异步的**，不是全局同时发生。他们用 MoE 路径相似度来监测。
- (d) 我们采用："逐数据组异步转变"的表述可以直接对应我们的逐样本翻转点分散，用来说明 ASR 跳变是许多局部转变的叠加，且这些局部转变本身共同加速。
- (e) 未分离：没有因果数据干预。

---

## 3. LM 训练中的突降 / 相变；鞍点到鞍点；LoRA 动力学；发育可解释性

### 3.1 Olsson et al. 2022, *In-context Learning and Induction Heads* (Anthropic)
- 链接：https://arxiv.org/abs/2209.11895 ；正文 https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- (a) 设定：从 1 层到大模型的多种 transformer。
- (b) 观察，以下来自我读到的正文：
  - "ICL score" 定义为第 500 个 token 的 loss 减第 50 个 token 的 loss，对样本取平均。
  - ICL 在训练早期的一个狭窄窗口内突然出现，约 2.5–5B tokens；之前不到 0.15 nats，之后约 0.4 nats。
  - 用"per-token loss vector"做 PCA：在 10,000 个随机 token 上计算各 checkpoint 的对数似然向量。
  - 修改架构使 induction head 形成的时间改变时，ICL 的提升"以精确匹配的方式"随之平移。
- (c) 结论：给出 6 条论证。小型纯注意力模型上是因果证据，大模型上是相关证据。作者自称是初步、间接的证据。
- (d) 我们采用：
  - **per-token loss PCA**：对毒样本和 clean 样本的逐样本 margin 向量做 PCA，看跳变是否表现为函数空间轨迹中的一个"拐角"。
  - **"把机制形成时间挪走，看行为是否跟着挪"的共移论证**。我们的 Adam 重置、LR 减半、删除毒行都属于这一类，应按 Olsson 的风格呈现成"干预使跳变点平移 Δ 步"的表格。
  - 具体的架构修改方式（社区常称 "smeared key"）在我读到的片段中没有出现名字，**[未核实]**。
- (e) 未分离：没有讨论为什么 induction head 在那一刻形成。后续 Reddy、Singh 补上了这一点。

### 3.2 Chen, Shwartz-Ziv, Cho, Leavitt, Saphra 2024, *Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs* (ICLR 2024 spotlight)
- 链接：https://arxiv.org/html/2309.07311v6
- (a) 设定：BERT-Base 在 BookCorpus+Wikipedia 上预训练，分析到 300K 步。
- (b) 观察，以下来自正文：
  - SAS 用注意力探针预测依存父节点，在 PTB 的 1,000 篇 WSJ 文档上算 UAS 来度量。
  - 先出现 "structure onset"：UAS 突增，同时 MLM loss 陡降，约 20K–30K 步。
  - 随后出现 "capabilities onset"：BLiMP 准确率急升。
  - **断点检测**：选使"断点前后斜率变化最大"的时间步，近似最大加速度点。
  - **正则化干预**：用加权 λ 的句法正则项促进（SAS+）或抑制（SAS−）SAS。SAS− 使两个 onset 基本消失，并让一个"替代策略"约在 6K 步提前出现。先抑制到 3K 步再释放，在 100K 步时略优，但长训后优势不显著。
- (c) 结论：SAS 是语法能力的必要前提，内部结构的出现先于并支撑外部能力。SAS 与替代策略竞争。证据是因果正则干预。
- (d) 我们采用，这是**最该照搬的实验范式**：
  - 用他们的**断点检测定义**统一定义 "jump step"，避免 ASR 阈值的任意性。
  - **"结构 onset → 能力 onset"两段式**直接对应我们的"clean 能力 onset → 后门 onset"。建议画同一张图：clean margin 的断点、毒样本推力的断点、ASR 的断点，三者的时间先后。
  - **因果正则**：用正则或数据调度**人为提前或推迟 clean 能力的获得**，例如预先用 clean 数据 warm-up，或在训练早期对 clean 任务加噪或降权，看后门 onset 是否跟着平移。这就是对"能力门控"最干净的因果检验，是 Chen 式"操纵前置结构"的直接移植。
- (e) 未分离：他们的前置结构与后继能力属于**同一数据流**。我们可以独立操纵 clean 与 poison 两条数据流，做得更干净。

### 3.3 Reddy 2024, *The Mechanistic Basis of Data Dependence and Abrupt Learning in an In-Context Classification Task* (ICLR 2024 oral)
- 链接：https://ar5iv.labs.arxiv.org/html/2312.03002 ；https://iclr.cc/virtual/2024/oral/19749
- (b) 观察：在最小注意力网络中，ICL 由 induction head 的突然出现驱动，并与 in-weights learning 竞争。他们识别出先于 ICL 的 progress measures，构造了两参数的 induction head 模型。
- (c) 结论：突然出现可以追溯到**三个嵌套 logit 的顺序学习**，这一顺序由一种"内在课程"实现。尖锐转变来自 ICL 所需的多层运算链，即按顺序学会的嵌套非线性。
- (d) 我们采用：**"嵌套 logit + 内在课程"是我们"能力门控"最接近的已发表机制模板**。我们可以写一个两变量的唯象模型：c 为 clean 能力，b 为后门强度，`db/dt = η · g(c) · f(b)`，其中 g 单调递增，即"效率随 clean 能力增长"。然后用测得的 g（约 20×）拟合平台长度和跳变宽度，像 Reddy 那样用唯象模型复现数据依赖。
- (e) 未分离：嵌套发生在**同一电路内部**。我们是跨任务的门控，并且可以在数据层面做因果切断（删毒、换序）。

### 3.4 Singh, Moskovitz, Hill, Chan, Saxe 2024, *What Needs to Go Right for an Induction Head?* [会议 venue 未核实]
- 链接：https://arxiv.org/abs/2404.07129
- (b)/(c) 提出受光遗传学启发的训练中激活钳制（clamping）因果框架。把 induction 电路拆成 3 个相互作用的子电路，并说明正是它们的相互作用产生相变。
- (d) 我们采用：**训练期激活或权重钳制**。可以在训练中把 LoRA 的后门方向分量钳制为 0（或固定），观察 clean 能力是否照常增长。也可以反过来，冻结与 clean 任务相关的方向，看后门推力是否不再增长。这能把"clean 能力 → 后门效率"的因果箭头钉住。
- (e) 未分离：没有涉及数据投毒。

### 3.5 Singh et al. 2023, *The Transient Nature of Emergent ICL* (NeurIPS 2023) 与 Singh et al. 2025, *Strategy Coopetition Explains the Emergence and Transience of ICL* [后者仅 arXiv，venue 未核实]
- 链接：https://arxiv.org/abs/2311.08360 ；https://arxiv.org/abs/2503.05631
- (b)/(c) ICL 出现后会被 in-weights 策略取代。在 2025 文中，作者发现两种策略**共享子电路**：ICL 无法单独快速出现，只能借助另一策略（CIWL）的同步缓慢发展才被启用。作者称之为"coopetition"，并用最小数学模型复现。
- (d) 我们采用：**这是"一个能力的出现由另一能力的缓慢进展所启用"的直接先例**，与我们的"后门由 clean 能力启用"同构。引用时可以说，我们把 coopetition 从同一任务的两种策略推广到"攻击者任务与良性任务"之间，并给出数据层面的因果证据。
- (e) 未分离：没有涉及安全或后门，也没有逐样本推力的测量。

### 3.6 Saxe, McClelland, Ganguli 2014, *Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks* (ICLR 2014)
- 链接：https://ar5iv.labs.arxiv.org/html/1312.6120
- (b) 观察，以下来自正文：在平衡假设下，模式强度满足 `τ du/dt = 2u(s − u)`（Eq.10）。从 u0 学到 u_f 的时间为 `t = (τ/2s)·ln[u_f(s−u0)/(u0(s−u_f))]`，约等于 `(τ/s)·ln(s/ε)`。轨迹是 sigmoid（Eq.12）。原文说该 sigmoid 可以呈现从"不学习"到"完全学会"的尖锐转变，学习时间与相关强度 s 成反比。
- (c) 结论：深度线性网络也有长平台加快速下降，有精确解析解。
- (d) 我们采用，这是**自催化正反馈的基准模型**：
  - 乘积参数化下，增长率正比于当前强度 u。LoRA 的 `ΔW = BA` 正是乘积结构，B=0 初始化相当于 u0 极小。
  - 平台长度约为 `(1/s)·ln(s/u0)`，由驱动强度 s 和初始强度决定。
  - **关键区分**：在 Saxe 模型中，有效驱动 s 是常数，正反馈来自 u 自身。在我们的假说中，s 本身随 clean 能力增长，即 s = s(c(t))。我们测得的"每行推力 0.01 → 0.2–0.5"应解读为 s 的增长，而不是 u 的增长。这需要设计实验，把"推力随后门自身强度增长"（Saxe 自催化）与"推力随 clean 能力增长"（门控）分开（见第 7 节）。
- (e) 未分离：单任务；没有跨模式耦合。论文中各模式是解耦的。

### 3.7 Jacot, Ged, Şimşek, Hongler, Gabriel 2021, *Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity*
- 链接：https://arxiv.org/abs/2106.15933 ；https://ar5iv.labs.arxiv.org/html/2106.15933
- (b)/(c) 小初始化时，初始点在鞍点附近。他们猜想梯度下降依次访问秩递增的一系列鞍点。前两个鞍点之间有定理，其余有数值支持。
- 后续：Bantzis, Simon, Jacot, *Saddle-To-Saddle Dynamics in Deep ReLU Networks: Low-Rank Bias in the First Saddle Escape* (ICLR 2026)，https://arxiv.org/abs/2505.21722 [仅摘要]。
- (d) 我们采用：LoRA 的 B=0 正好使初始点处于鞍点。可以检查 LoRA ΔW 的**有效秩**在跳变时是否增加 1（"逃离一个鞍点"），以及后门方向是否对应新增的奇异方向。
- (e) 未分离：鞍点逃逸时刻由数据谱决定，没有讨论"另一个任务改变了谱"。

### 3.8 Zhang, Saxe, Latham 2026, *Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures* (ICLR 2026)
- 链接：https://arxiv.org/abs/2512.20607
- (b)/(c) 给出覆盖全连接、卷积、注意力网络的统一框架。线性网络学到秩递增的解，ReLU 网络增加拐点，卷积网络增加 kernel，自注意力增加 head。训练沿不变流形靠近鞍点，再转移到另一流形。他们区分了"数据驱动"（偏向低秩）和"初始化驱动"（偏向稀疏）两种鞍点到鞍点动力学，并预测数据分布与初始化如何影响平台的长度与数量。
- 前作：Zhang et al. 2023, *Transformers Learn through Gradual Rank Increase*，https://arxiv.org/abs/2306.07042 [仅检索摘要]。该文的说法是：loss 平台对应权重停在鞍点附近，鞍点被快速逃离，秩每次最多加 1。
- (d) 我们采用：
  - **平台长度的理论预测**：数据统计越弱，平台越长。我们的"LR 减半延迟跳变"和"删除少量毒行延迟 12–33 步"可以与"驱动强度 × 时间"的预测比对。
  - **"新增一个 head 或一个秩"作为跳变的结构签名**。
- (e) 未分离：数据分布在训练中是固定的，有效驱动不随另一任务改变。**我们可以把他们的框架扩展为"驱动强度随 clean 任务进展而变"的非自治版本**，这是理论上的增量。

### 3.9 Abbe, Boix-Adserà, Misiakiewicz 2023, *SGD Learning on Neural Networks: Leap Complexity and Saddle-to-Saddle Dynamics* (COLT 2023)
- 链接：https://arxiv.org/abs/2302.11055 ；https://proceedings.mlr.press/v195/abbe23a.html
- (b)/(c) 定义 "leap" 来衡量目标函数的层级性。猜想学习时间约为 `d^max(Leap,2)`，并在 Gaussian 数据、两层网络、附加技术假设下证明。训练通过鞍点到鞍点逐步学会支撑集，"爬升层级特征的阶数"。他们还提出 degree curriculum。前作 2022 年覆盖 leap-1，即 merged-staircase。
- (d) 我们采用：**staircase 是"先学会的分量让后续分量变得可学"的严格理论版本**，即低阶项学会后，高阶项的梯度信号从约 0 变为 Θ(1)。我们的"clean 能力使毒样本推力增长 20 倍"可以表述为一种 staircase：后门映射在 clean 特征上是"低 leap"的，在原始特征上是"高 leap"的。
- (e) 未分离：没有安全场景；两个分量属于同一目标函数。

### 3.10 LoRA 专属的动力学
- Xu et al. 2025, *Understanding the Learning Dynamics of LoRA: A Gradient Flow Perspective on Low-Rank Adaptation in Matrix Factorization* (AISTATS 2025) [仅摘要]，https://arxiv.org/abs/2503.06982v1 。结论：小初始化下梯度流收敛到最优解邻域，初始化越小误差越低；最终误差受预训练模型与目标矩阵奇异空间的错配影响；提出谱初始化。
- Karuturi et al., *Spectral Dynamics of Low-Rank Adaptation*（ICML 2026 workshop poster）[仅摘要，workshop 级]，https://icml.cc/virtual/2026/75823 。结论：在双线性分解的梯度流下，更新的奇异值按 oracle 奇异值从大到小依次增长。
- Malladi et al. 2023, *A Kernel-Based View of Language Model Fine-Tuning* (ICML 2023)，https://alphaxiv.org/abs/2210.05643 。结论：带有意义 prompt 的微调常呈现类核（eNTK）动力学，并用此解释 LoRA 等子空间方法的成功。
- (d) 我们采用：
  - 用 Malladi 的 eNTK 判断我们的 LoRA 训练在跳变前是否处于 lazy 区。若是，则跳变可以对应 Kumar 的 lazy→rich 转变。
  - 用 Xu 的"奇异空间对齐"语言描述 clean 学习如何改变后门方向的可达性。
- (e) 未分离：这些工作都没有讨论多任务混合数据中一个子任务的动力学。

### 3.11 发育可解释性 / 局部学习系数（Timaeus）
- Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, Murfet, *Loss Landscape Degeneracy and Stagewise Development in Transformers*（arXiv 2402.02364 v3，标注 To appear, TMLR），https://arxiv.org/abs/2402.02364 。结论：LLC 追踪 loss 景观简并度；训练分成若干阶段，简并度的变化与内部计算结构和 I/O 行为的变化同时发生。作者自称是 suggestive evidence。
- Lau et al., *The Local Learning Coefficient: A Singularity-Aware Complexity Measure* (AISTATS 2025)，https://arxiv.org/abs/2308.12108v1 ：用 SGLD 估计 LLC，可扩展到大模型。
- Wang et al., *Differentiation and Specialization of Attention Heads via the Refined LLC* (ICLR 2025)，https://arxiv.org/abs/2410.02984 ：refined LLC（按权重子集或数据子集），用于看 head 的分化。
- (d) 我们采用：**data-refined LLC**，即分别以 clean 数据和 poison 数据作为局部后验数据集估计 LLC，看两者的阶段边界是否先后出现。理想图景是 clean-LLC 先变化，poison-LLC 后变化。这是对"门控"的几何独立证据，而且不依赖我们自己选的方向投影。
- (e) 未分离：我没有找到 LLC 用于后门或投毒的已发表工作（检索结果见 https://www.lesswrong.com/posts/gGAXSfQaiGBCwBJH5/timaeus-in-2024 等，只有计划性描述）。这属于空白。

### 3.12 Ren & Sutherland 2025, *Learning Dynamics of LLM Finetuning* (ICLR 2025 Outstanding Paper) [仅摘要]
- 链接：https://arxiv.org/abs/2407.10490 ；https://iclr.cc/virtual/2025/oral/31755
- (b)/(c) 对微调做逐步分解，分析影响如何在不同候选回答之间累积。用"squeezing effect"解释 off-policy DPO 中期望输出概率下降的现象。
- (d) 我们采用，**推荐作为"每条毒行推力"的正式分解框架**：一步更新对某样本 log-prob 的变化可以分解为"学习率 × eNTK 型相似度项 × 被更新样本的 loss 梯度项"。分解的具体记号以原文为准，**[未核实]**。
  - 关键推论：早期毒样本 loss 很高时，softmax 残差 (p − y) 的范数接近饱和，不可能再大 20 倍。所以 20× 的效率增长很可能来自**相似度或 kernel 项**，也就是 clean 学习改变了表征，使毒样本梯度与后门目标的对齐变好。
  - 把推力拆成这两项分别画出，可以直接区分"残差驱动"和"特征对齐驱动"，与 Kumar 的 lazy→rich 图景对接。
- (e) 未分离：没有讨论后门或多任务门控。

### 3.13 其他与平台/突变直接相关的 2024–2025 工作
- Edelman, Tsilivis, Edelman, Malach, Goel 2024, *The Evolution of Statistical Induction Heads* (NeurIPS 2024)，https://ar5iv.labs.arxiv.org/html/2402.11004 。结论：训练经历均匀预测 → 次优 unigram → 快速相变到 bigram；更简单的 unigram 解会**推迟**最终解的形成。我们采用：检查我们的平台期模型是否处于一个"次优简单解"，例如输出 clean 答案、或把触发器当噪声。这对应"竞争推迟"类解释，必须与"门控"区分。
- Gopalani & Hu 2025, *What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers* (NeurIPS 2025) [仅摘要]，https://arxiv.org/abs/2506.13688 。结论：平台期模型常已形成可解释的部分解，同时有重复偏置和表征坍缩；注意力配置的"隐藏进展"先于跳变；直接干预注意力会显著改变平台长度。这些现象也出现在 Pythia/OLMo 早期预训练中。我们采用：**直接干预某组件可以改变平台长度**，这是又一个因果范式。
- Zucchet et al. 2025, *How Do Language Models Learn Facts? Dynamics, Curricula and Hallucinations*，https://arxiv.org/abs/2503.21676v1 。结论：知识获取前有平台，平台期间形成支持回忆的注意力电路；**不平衡的数据分布会缩短平台**。
- Zucchet, D'Angelo, Lampinen, Chan 2025, *The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition* (NeurIPS 2025 oral) [仅摘要]，https://arxiv.org/abs/2505.17863v1 。结论：涌现时间服从依赖任务结构、架构、优化器的幂律；重复可以大幅加速涌现。我们采用：把"毒行密度或重复"对跳变时间的影响画成幂律图，与他们的理论对比。我们的"删除 2–5 条毒行延迟 12–33 步"可以放进这张图。
- Raventós, Paul, Chen, Ganguli 2023, *Pretraining Task Diversity and the Emergence of Non-Bayesian ICL for Regression* (NeurIPS 2023)，https://arxiv.org/abs/2306.15063 。结论：存在任务多样性阈值，跨过后行为从"以训练任务为先验的 Bayes 估计"转为 ridge。这是数据轴上的相变，作为背景引用即可。

---

## 4. 微调中的突然行为转变：EM model organisms 与后门/投毒的训练动力学

### 4.1 Turner, Soligo, Taylor, Rajamanoharan, Nanda 2025, *Model Organisms for Emergent Misalignment*
- 链接：https://arxiv.org/abs/2506.11613v1 ；全文 https://arxiv.org/html/2506.11613v1
- (a) 设定：Qwen-14B 第 24 层 MLP down-proj 上的单个 rank-1 LoRA，数据为 bad-medical-advice。另有 9 个 adapter、rank-8/64、Llama-3.1-8B、全参 SFT。
- (b) 观察，以下来自正文第 4 节：
  - B 向量的 L2 范数**平滑增长**，方向在约 **第 180 步后急剧旋转**：局部余弦相似度出现峰值；对每 5 步采样的 B 向量做 PCA，PC2 出现拐点，前两个 PC 约解释 95% 方差。
  - 旋转与一个持续较长的梯度范数峰同时出现。
  - 基线设定下，行为上的错位率在约 300–600 步间逐渐上升。
  - **把 adapter 放大 5 倍**后，EM 在 100 多步内出现，幅度约为基线的 4 倍；放大 20 倍则输出不连贯。
  - 稳健性：EM 出现点在 alignment 阈值 40–70、coherence 阈值 10–40 范围内稳定。
  - 全参 SFT 的梯度范数太嘈杂、无法定位峰，但错位率在前约 15 步保持为 0，之后快速上升。
- (c) 结论：在微调参数中可以识别出机制相变，它对应一个稳健的行为相变。证据包括多设定复现和 adapter 缩放。
- (d) 我们采用，**三件工具都应照搬**：
  - **局部余弦相似度 + PCA 轨迹**定位 LoRA 方向的突变点，与 ASR 跳变对齐。
  - **adapter 缩放探针**：在平台期的 checkpoint 上把 LoRA 乘以 α>1，看后门是否已经"潜在存在"。若放大后 ASR 已高，平台期就是 hidden progress，Nanda/Barak 型；若放大后仍无，方向尚未形成，支持门控型。**这是区分两类机制最便宜的实验。**
  - **阈值扫描稳健性**：他们对判定阈值做了扫描。我们应对 ASR 的 margin 阈值做同样的扫描，以回应 mirage 质疑。
- (e) 未分离：没有回答为什么旋转发生在第 180 步，也没有做数据层面的因果干预（删数据、换序）。**我们的删毒和 fork 实验正好补上这一点。**

### 4.2 Soligo, Turner, Rajamanoharan, Nanda 2025, *Convergent Linear Representations of Emergent Misalignment* (ICML 2025 Actionable Interpretability workshop)
- 链接：https://arxiv.org/abs/2506.11618
- (b)/(c) 在 Qwen2.5-14B-Instruct 上用 9 个 rank-1 adapter 诱导 EM。不同 EM 模型收敛到相似的"错位方向"：从一个模型提取的方向可以压制其他、用更高秩 LoRA 和不同数据训练的模型的错位行为。6 个 adapter 贡献通用错位，2 个只在微调领域内错位。
- (d) 我们采用：**用最终模型提取的方向去度量早期 checkpoint**，即"最终方向投影"这一做法，以及跨 seed 和跨数据检验方向是否收敛。我们的"沿最终后门方向的推力"与此方法论一致，可以直接引用为先例。
- (e) 未分离：没有研究方向何时、为何出现。

### 4.3 Wang et al. 2026, *Persona Features Control Emergent Misalignment* (ICLR 2026) [仅摘要]
- 链接：https://arxiv.org/abs/2506.19823v1
- (b)/(c) 用 SAE 做 model diffing，发现"misaligned persona"特征，其中一个 toxic persona 特征最能控制并预测 EM。用几百条良性样本再微调即可恢复对齐。
- (d) 我们采用：可以用"既有特征被放大"的视角解释后门的快速逃逸，即后门利用模型已有的特征。这与 Prakash 2024（见 5.8）呼应。
- (e) 未分离：没有动力学。

### 4.4 Rao, Gong, Hu, Naik 2026, *An Emergent Mirage: Is Emergent Misalignment and Realignment Indeed a Robust Phenomenon?* [仅摘要] —— 反方
- 链接：https://arxiv.org/abs/2607.09053
- (b)/(c) 反复进行"错位—再对齐"循环并跟踪 LoRA 表征。控制回复长度后，"快速再对齐"大部分消失。此前报告的 LoRA 空间表征相变"在训练中并不一致地与行为错位相关"。
- (d) 我们采用：**这是审稿人可能拿来攻击"机制相变签名"的现成武器**。防守方式：
  - 多 seed 并给出置信区间；
  - 控制输出长度和格式。我们的 target 若与 clean 答案长度不同，需要检查跳变不是长度或格式翻转；
  - 用数据层面的因果干预（删毒、fork），而不是仅靠表征签名。
- (e) —

### 4.5 Souly et al. 2025, *Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples*
- 链接：https://arxiv.org/abs/2510.07192 ；全文 https://arxiv.org/html/2510.07192v1
- (a) 设定：600M–13B 模型在 Chinchilla 最优数据（6B–260B tokens）上预训练；另有 Llama-3.1-8B-Instruct 和 GPT-3.5-turbo 的微调实验。
- (b) 观察，以下来自全文：
  - 250 篇毒文档可以在所有规模上植入后门；100 篇在 DoS 预训练实验中不成功。
  - **决定成功的是毒文档的绝对数量，而不是比例。**
  - 微调中 clean 数据增加两个数量级（1000→100000）时，绝对数量仍然主导。
  - **毒样本位置**：随机打散最好；放在末尾需要至少约 100 条才有效；**20 条放在开头有效、放在末尾无效**。
  - 学习率越高，所需毒样本越少；线性 LR 调度下放在末尾成功率明显更低。
  - 每批毒密度越高，所需的总毒样本**反而越多**。作者推测与"在毒数据上的顺序梯度步数"有关，并标注需要进一步研究。
  - 毒后继续 clean 训练会降低 ASR，微调中可降到接近 0。
  - 正文没有描述 ASR 随步数的突然跳变。
- (c) 结论：攻击所需毒样本数近似常数。实验规模大，结论强；机制没有解释。
- (d) 我们采用：
  - **"绝对数量主导"与我们的逐行推力框架天然兼容**：如果跳变条件是"累积有效推力 Σ(推力_i) 超过阈值"，而每条推力受 clean 能力门控，那么在 clean 能力很快饱和的大数据场景下，所需条数确实近似常数。我们可以**给出 Souly 的现象学一个机制解释**，这是非常好的定位。
  - **位置实验必须正面回应**："开头 20 条有效、末尾 20 条无效"看起来与"能力越强，每行越有效"相矛盾。可能的调和方式：(i) 末尾 LR 已衰减，他们自己指出线性调度下末尾更差；(ii) 开头的毒行之后还有大量 clean 步，可能在后续步中被"放大"，例如 LoRA 的乘积自催化；(iii) 他们的 base 是 instruct 模型，clean 能力一开始就高。我们应设计**常数 LR 下的位置扫描**来分离这些因素。
  - **"高毒密度反而需要更多毒样本"**提示单步内的毒梯度存在饱和或冗余。这与我们的"删 2–5 行只延迟 12–33 步"一起，可以估计单行边际效应。
- (e) 未分离：他们没有测量跳变的时间结构，也没有测每行效率随训练进展的变化。**这正是我们的核心贡献点。**

### 4.6 Bowen et al. 2024/2025, *Scaling Trends for Data Poisoning in LLMs* (FAR.AI)
- 链接：https://arxiv.org/abs/2408.02946v3
- (b)/(c) 在 1.5–72B 的 23–24 个模型上，更大的模型在极少量投毒下更快学会有害行为，包括 sleeper agent 行为。
- (d) 我们采用：**"更大更强的模型对毒更敏感"是能力门控在规模轴上的现象学对应**。可以引用为宏观佐证：更强的通用能力带来更高的投毒效率。
- (e) 未分离：规模与能力相互混淆；没有训练内的动力学。

### 4.7 Wan, Wallace, Shen, Klein 2023, *Poisoning Language Models During Instruction Tuning* (ICML 2023)
- 链接：https://proceedings.mlr.press/v202/wan23b.html
- (b)/(c) 约 100 条毒样本即可让任意短语带负面情感或引发退化输出；更大的模型更易被投毒。
- (d) 与 4.6 同。

### 4.8 Li et al. 2021, *Anti-Backdoor Learning: Training Clean Models on Poisoned Data* (NeurIPS 2021)
- 链接：https://arxiv.org/abs/2110.11571v3
- (a) 设定：图像分类从头训练。
- (b) 观察：模型学后门数据比学 clean 数据快得多，攻击越强收敛越快；后门任务绑定特定目标类。
- (c) 结论：据此在训练早期用低 loss 隔离毒样本，后期做反学习。
- 相关：BackdoorBench（https://arxiv.org/abs/2407.19845 ，**我只在检索摘要中看到**）把这一现象称为 "quick learning"，即先学 trigger→目标的简单映射，再学复杂的语义映射。它也指出 BadNets 等可能例外。
- (d) 我们采用：**必须正面对比的反方向现象**。在 ABL 的设定里后门比 clean 学得快；在我们的设定里后门要等 clean 学会。可能的调和方式：
  - 他们从头训练，触发器是像素级的强特征；我们在强预训练模型上做 LoRA 生成式微调，target 是一段需要语言能力才能输出的回复；
  - 他们的"clean 任务"是 10 类分类，很难；我们的 clean 任务是 SQuAD 或指令格式，它是输出 target 所需的格式前提。
  - 写成"何时后门先于 clean、何时后门晚于 clean"的条件，比回避更有价值。
- (e) 未分离：ABL 没有区分"毒 loss 下降快"与"ASR 跳变"；也没有测每行推力随 clean 进展的变化。

### 4.9 Cinà et al., *Backdoor Learning Curves: Explaining Backdoor Poisoning Beyond Influence Functions* (IJMLC 2024) [仅摘要]
- 链接：https://arxiv.org/abs/2106.07214v3
- (b)/(c) 用增量学习和影响函数研究后门学习。定义 **backdoor learning slope**，衡量模型学后门样本的速度，并与影响函数相联系；另定义参数偏离初值的后门影响量。攻击效果取决于算法复杂度（超参数）、毒比例、触发器大小与可见度。存在"clean 精度高而后门无效"的超参数区域。
- (d) 我们采用：**learning slope 是最早的"每样本后门效率"概念**，应作为我们"每行推力"的先例来引用。我们的增量在于：slope 不是常数，而是随 clean 能力变化 20 倍。
- (e) 未分离：没有考虑 slope 的时变性。

### 4.10 Shejwalkar, Lyu, Houmansadr 2023, *The Perils of Learning From Unlabeled Data: Backdoor Attacks on Semi-supervised Learning* (ICCV 2023)
- 链接：https://ar5iv.labs.arxiv.org/html/2211.00453 ；https://arxiv.org/abs/2211.00453
- (a) 设定：FixMatch、ReMixMatch 等半监督学习；只毒化无标签数据，约 0.2%。
- (b) 观察，以下来自正文第 VI-B 节和图 6：
  - FixMatch 随训练推进，把越来越多的毒样本伪标为目标类；**ASR 的上升跟随伪标签向目标类的移动，但有一定延迟**，作者归因于学 trigger 任务需要非平凡的迭代数。
  - ReMixMatch 训练初期对目标类置信度低；模型学到目标类的显著特征后置信度才上升，ASR 随之上升。
  - 总结：**后门在模型对目标类有高置信之后才被安装。**
- (c) 结论：这是一个定性的动力学描述，没有数值曲线，也没有因果干预。
- (d) 我们采用：**这是我检索到的最接近"后门安装由 clean 能力门控"的先例**。在他们的设定中，门控通过伪标签机制显式实现。我们的贡献是证明在**全监督的 LoRA 微调中也存在门控**，而且是隐式的，通过梯度效率实现，并且可以定量测出 20 倍。
- (e) 未分离：门控机制被伪标签混淆；文中只写了曲线形状，没有给数值。
- 注：用户提到的 "semi-supervised tipping point backdoors"，我没有找到以 "tipping point" 命名的论文 **[未找到]**。最接近的是上面这篇，以及 Connor & Emanuele 2022, *Rethinking Backdoor Data Poisoning Attacks in the Context of Semi-Supervised Learning*（https://arxiv.org/abs/2212.02582 ，仅摘要，未看到动力学分析）。

### 4.11 Tan, Huang, Li 2025, *The "Sure" Trap: Multi-Scale Poisoning Analysis of Stealthy Compliance-Only Backdoors in Fine-Tuned LLMs*
- 链接：https://arxiv.org/html/2511.12414v1
- (b)/(c) 在毒样本数量轴上出现尖锐阈值：几十条，正文给出约 50 条。超过阈值后 "Sure" 率接近 100%，ASR 上升并饱和。作者称这在 1k–10k 数据量和 1B–8B 模型上基本成立，但 GPT-3.5 上 ASR 接近 0。没有逐步动力学分析。
- (d) 我们采用：毒样本数量轴上的阈值，可以与我们训练步轴上的跳变互相映射（"累积有效推力"）。
- (e) 未分离：没有时间维度。

### 4.12 Khaddaj et al. 2023, *Rethinking Backdoor Attacks* (ICML 2023)
- 链接：https://proceedings.mlr.press/v202/khaddaj23a.html
- (b)/(c) 不了解训练分布结构时，后门与自然特征不可区分。作者把后门形式化为"训练数据中最显著的特征"，并据此用 datamodels 检测。
- (d) 我们采用：**datamodel 或影响力式的逐样本归因**，可以作为删毒实验的补充，估计每条毒行对 ASR 的边际贡献。
- (e) 未分离：没有动力学。

---

## 5. 正反馈 / 能力门控学习：课程、先易后难、特征复用

### 5.1 Okawa, Lubana, Dick, Tanaka 2023, *Compositional Abilities Emerge Multiplicatively* (NeurIPS 2023)
- 链接：https://arxiv.org/abs/2310.09336v1
- (b)/(c) 数据生成过程的组合结构决定能力出现的顺序。组合任务的表现**乘法地**依赖于每个组成概念的学习程度，因此会突然跳升。组合稀有概念需要多得多的优化步。
- (d) 我们采用：**乘法模型是解释"为什么门控产生尖锐跳变"的最简数学形式**。若 ASR 或毒 margin 的增长率约等于 g(clean) × h(backdoor)，两个 sigmoid 的乘积比单个 sigmoid 更陡。我们可以直接拟合这个乘法形式，与"单一 sigmoid"对比 AIC/BIC。
- (e) 未分离：概念之间是对称的乘积；我们的情况是有向的，clean 门控 backdoor，反之不成立。可以检验这种不对称。

### 5.2 Lubana, Kawaguchi, Dick, Tanaka 2025, *A Percolation Model of Emergence* (ICLR 2025) [仅摘要]
- 链接：https://arxiv.org/abs/2408.12578v2
- (b)/(c) 定义涌现为"习得数据生成过程的一般规律，从而引起特定窄任务的性能突增"。用二部图上的渗流给出形式相变模型，并预测干预数据规律时涌现点的移动。
- (d) 我们采用："一般规律（clean 能力）的习得导致窄任务（后门）突增"与我们的假说在概念上同构。可以引用为同一类思想，并指出我们的设定是对抗性的，因果干预是数据层面的。
- (e) 未分离：没有安全场景。

### 5.3 Nam, Fonseca, Lee, Mingard, Louis 2024, *An Exactly Solvable Model for Emergence and Scaling Laws in the Multitask Sparse Parity Problem* (NeurIPS 2024) [仅摘要]
- 链接：https://arxiv.org/abs/2404.17563v3
- (b)/(c) 把每项技能当作一个基函数，用多线性模型给出技能涌现时间的解析式，以单一拟合参数复现两层网络中多技能的 sigmoid 涌现。
- (d) 我们采用：多线性模型的思路可以用来写我们的唯象模型（见 3.3 的 `db/dt = η·g(c)·f(b)`）。
- (e) 未分离：其中技能之间独立，没有门控。

### 5.4 Park, Okawa, Lee, Tanaka, Lubana 2024, *Emergence of Hidden Capabilities: Exploring Learning Dynamics in Concept Space* (NeurIPS 2024 spotlight) [仅摘要]
- 链接：https://arxiv.org/abs/2406.19370
- (b)/(c) 学习速度和学习顺序由数据的 "concept signal" 决定。概念空间轨迹会突然转向，转向时刻与"潜在干预已能操纵该概念、但朴素 prompt 还不能引出"的时刻一致。
- (d) 我们采用：**潜在干预（如 adapter 缩放、激活加 steering）先于行为显现**，与 4.1 的缩放探针互相印证。
- (e) 未分离：没有跨任务门控。

### 5.5 Chen et al. 2023, *Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models* (NeurIPS 2023)
- 链接：https://arxiv.org/abs/2307.14430v1
- (b)/(c) 在合成和真实数据中都存在有序技能集。先训练前置技能后，高级技能所需的数据更少。据此提出在线数据采样算法 Skill-It。
- (d) 我们采用：**"前置技能降低后续技能数据需求"就是能力门控的数据效率表述**。我们的 clean 任务正是后门的前置技能。Skill-it 用技能图描述，我们可以测出"边权"，即 20 倍效率。
- (e) 未分离：Skill-it 测的是最终数据效率，没有测每样本梯度推力，也没有安全场景。

### 5.6 Weinshall, Cohen, Amir 2018, *Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks* (ICML 2018)
- 链接：https://ar5iv.arxiv.org/html/1802.03796
- (b)/(c) 在凸线性回归上，理想课程的收敛速率随样本难度单调变化，训练推进后课程带来的提速单调减小；在同样难度的点中，相对当前假设 loss 更高的点收敛更快。
- (d) 理论背景：课程效益有严格结果。注意他们的结果是"高 loss 点更快"，与我们"高 loss 时毒行效率低"形成有趣的对比。原因在于我们是非凸、表征依赖的情形，这恰好说明凸理论不够用。
- (e) —

### 5.7 Panigrahi, Liu, Malladi, Risteski, Goel 2025, *Progressive Distillation Induces an Implicit Curriculum* (ICLR 2025 oral)
- 链接：https://arxiv.org/abs/2410.05464
- (b)/(c) 在 sparse parity 上，用教师的中间 checkpoint 蒸馏，可以加速学生学习，并带来可证明的样本复杂度收益。原因是中间 checkpoint 中存在隐式课程。PCFG 和真实预训练数据中也有类似现象：逐步学会捕捉更长上下文的特征。
- (d) 我们采用：**"先学到的简单特征让难特征的梯度信号变强"有严格的样本复杂度证明**。可以作为"效率随前置能力增长"的理论依据来引用。
- (e) 未分离：没有安全场景。

### 5.8 Prakash, Rott Shaham, Haklay, Belinkov, Bau 2024, *Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking* (ICLR 2024)
- 链接：https://arxiv.org/abs/2402.14811
- (b)/(c) 微调增强原模型已有的机制，而不是造新机制；entity tracking 电路在基座与微调模型间保持一致。提出 CMAP（跨模型激活修补）。
- (d) 我们采用：**CMAP**。把跳变后 checkpoint 的激活修补到跳变前的 checkpoint，定位"后门所需但跳变前缺失"的组件，检验这些组件是否是 clean 任务训练中被增强的那些。
- (e) 未分离：没有动力学，也没有后门。

### 5.9 Kalimeris / Nakkiran et al. 2019, *SGD on Neural Networks Learns Functions of Increasing Complexity* (NeurIPS 2019)
- 链接：https://arxiv.org/abs/1905.11604
- (b)/(c) 早期 epoch 的性能增益几乎都可以由线性分类器解释，之后逐步学到更复杂的函数。提出基于条件互信息的度量："一个分类器能多大程度解释另一个分类器的性能"。
- (d) 我们采用：可以用他们的**条件互信息度量**，问"在跳变前，模型行为能被一个只做 clean 任务的模型解释多少"。
- (e) —

### 5.10 Pezeshki et al. 2021, *Gradient Starvation: A Learning Proclivity in Neural Networks* (NeurIPS 2021) —— 反方向机制
- 链接：https://arxiv.org/abs/2011.09468v1
- (b)/(c) 交叉熵下，先学到的特征会"饿死"其他特征的梯度，导致特征不平衡。提出 Spectral Decoupling。
- (d) 我们采用：**这是"先学会的任务会抑制而不是促进后学任务"的标准理论**，与我们的发现方向相反。审稿人可能用它质疑。我们应指出：gradient starvation 适用于两个特征都能**单独**降低同一 loss 的情况，即冗余特征竞争；而在我们的设定中，clean 能力是输出 target 的**必要组成部分**，属于互补而非冗余。这一区分恰好对应 Chen SAS 的"竞争"与 Singh coopetition 的"合作"。
- (e) —

---

## 6. 综合：领域已经知道"为什么转变是尖锐的"（7 条）

1. **乘积参数化 + 小初始化 ⇒ 自催化 sigmoid 和长平台。** 增长率正比于当前强度，平台长度约为 `(1/s)·ln(s/u0)`，其中 s 是驱动强度（Saxe 2014 精确解）。鞍点到鞍点的框架把它推广到 ReLU、卷积、注意力（Jacot 2021；Zhang-Saxe-Latham 2026）。LoRA 的 BA 与 B=0 初始化正落在这一类。
2. **两种解或两种偏置的竞争 ⇒ 长时间停在次优解，然后尖锐切换。** 记忆与泛化电路效率的交叉（Varma 2023）；早期核偏置与晚期 max-margin 偏置（Lyu 2024，有证明）；lazy→rich（Kumar 2024）；简单的 unigram 解推迟 bigram（Edelman 2024）；SAS 与替代策略的竞争（Chen 2024）。
3. **嵌套或乘法组合 ⇒ 多个子能力全部到位时输出才跳。** 三层嵌套 logit 加内在课程（Reddy 2024）；三个交互子电路（Singh 2024）；乘法涌现（Okawa 2023）；staircase/leap（Abbe 2023，有证明）；渗流（Lubana 2025）。
4. **连续的隐藏进展在阈值处显形 ⇒ 尖锐只存在于读出端。** Fourier gap 驱动的隐藏进展（Barak 2022）；restricted/excluded loss 显示 grokking 是渐进的（Nanda 2023）；潜在干预先于行为（Park 2024；Turner 2025 的 adapter 缩放）；平台期注意力的隐藏进展（Gopalani & Hu 2025）。
5. **指标的非线性可以制造或夸大尖锐，但连续量本身也会加速。** Schaeffer 2023 对比 PassUntil（Hu 2024）以及 loss 阈值（Du 2024）。领域的共识更接近"连续但加速"，而不是"全是幻象"。
6. **平台长度由数据统计控制，而且可以被操纵。** 不平衡分布缩短平台（Zucchet 2025a）；重复加速涌现（Zucchet 2025b）；数据统计强度决定鞍点停留时间（Zhang-Saxe-Latham 2026）；干预前置结构或注意力会平移或消除转变（Chen 2024；Gopalani & Hu 2025）。
7. **一个能力的出现可以由另一个能力的缓慢进展所启用（合作），也可以被它抑制（竞争）。** 两者都有先例：合作见 coopetition（Singh 2025）、Skill-it 前置技能、半监督后门在目标类高置信后才安装（Shejwalkar 2023）；竞争见 gradient starvation（Pezeshki 2021）、ABL 的后门先学（Li 2021）。**哪种情形成立取决于两个任务是互补还是冗余**，这一点在文献中没有被系统分离过。

**据我检索，以下问题还没有被分离：** 在 LLM/VLM 的 LoRA 投毒中，(i) 每条毒样本沿最终后门方向的推力随 clean-task 能力变化的定量曲线；(ii) 用数据层面的 fork、删行、重放实验因果地证明 ASR 跳变需要哪些毒行、需要在什么能力状态下出现；(iii) 区分"后门自身自催化"（Saxe 型）与"clean 能力门控"（staircase/coopetition 型）。这三点就是我们的"青蒿素提纯"。

---

## 7. 最可能的审稿质疑及应答方法（5 条）

### 质疑 1："ASR 跳变只是阈值指标的假象（mirage）。"
- 应答方法：
  - Schaeffer 2023 本身给出了检验标准：换成连续指标后是否仍非线性。
  - Hu 2024 PassUntil：用高分辨率评测证明平台期确实接近 0 或确实缓慢增长。
  - Turner 2025：对判定阈值做扫描，看跳变点是否稳定。
  - Chen 2024：用"最大斜率变化"的断点检测，不依赖阈值。
  - Michaud 2023、Li 2026：逐样本或逐组分析。
- 我们的落地：报告连续 margin 的二阶导数；报告 87–100% 样本自身加速；把翻转点分布的陡峭度与"线性漂移 + 异质阈值"的零模型比较；补充阈值扫描和断点检测。

### 质疑 2："这只是 LoRA 乘积参数化的自催化逃逸（Saxe/Jacot 鞍点逃逸），与 clean 能力无关；clean 能力只是与时间相关的混杂变量。"
- 应答方法：
  - Saxe 2014 的 `du/dt ∝ u(s−u)` 给出零模型：推力只随后门自身强度 u 增长。
  - Chen 2024 的因果正则和 Singh 2024 的训练期钳制提供"操纵前置能力"的范式。
  - Kumar 2024 和 Ren & Sutherland 2025 的分解区分"残差项"与"kernel/对齐项"。
- 我们的落地：
  - **固定后门强度、改变 clean 能力**：例如先用纯 clean 数据 warm-up 到不同水平，再开始投毒；或钳制 LoRA 的后门分量，只让 clean 分量学习，然后测同一条毒行的推力。若推力随 clean 能力增长、与后门自身 margin 无关，就排除纯自催化。
  - **逆实验**：在相同步数下用打乱或噪声 clean 数据阻止 clean 能力增长，看推力是否停在约 0.01。
  - 把推力分解为残差项和相似度项。

### 质疑 3："这与 Souly 等的'近常数毒样本数'矛盾或已被其覆盖：他们发现毒样本放在末尾、模型已经很强时反而更不有效。"
- 应答方法：
  - Souly 2025 自己的 LR 和调度结果：低 LR 需要更多毒样本，线性调度下末尾更差；继续 clean 训练会降低 ASR。
  - Bowen 2024 和 Wan 2023：更强、更大的模型更易被毒，这是能力门控在规模轴上的证据。
- 我们的落地：
  - 在**常数 LR**下扫描毒样本位置（开头、中段、末尾、末尾后再接 clean 步），分离 LR 衰减、后续放大、clean 覆盖三个因素。
  - 给出"累积有效推力 = Σ 每行推力（受 clean 门控）"的模型，同时复现"近常数条数"和"跳变时机"。这样把 Souly 定位为**被我们机制解释的现象**，而不是竞争者。

### 质疑 4："经典结论是后门比 clean 任务学得更快（ABL、BackdoorBench 的 quick learning），gradient starvation 也预测先学会的任务会抑制后学的任务。你们的方向反了。"
- 应答方法：
  - Li 2021（ABL）和 BackdoorBench 是从头训练的图像分类，触发器是强的低层特征。
  - Shejwalkar 2023 提供反例：半监督后门在模型对目标类有高置信之后才安装。
  - Pezeshki 2021 适用于冗余特征竞争；Singh 2025 coopetition 与 Skill-it 适用于互补或共享子电路。
  - Chen 2024 显示两种关系（竞争与必要前提）可以在同一模型中同时存在。
- 我们的落地：明确给出条件。当输出 target 需要 clean 能力（生成式 target、需要格式和语言能力）时为门控；当触发器到标签是可以绕过 clean 特征的捷径时为 quick learning。可以做一个对照：把 target 换成单一 token 或简单标签，预测门控会减弱、跳变会提前。这能把反例变成支持。

### 质疑 5："机制相变签名不可复现（参见 'An Emergent Mirage'），单次运行中的 margin 平台和跳变可能是噪声或格式翻转。"
- 应答方法：
  - Rao 2026 指出 LoRA 表征相变签名与行为不一致，并且控制长度后效应消失。
  - Arnold & Lörch 2025：用统计不相似度和序参量分解，确认跳变时变化的是哪个行为方面。
  - Olsson 2022：干预使转变点平移、行为跟着平移的共移论证。
  - Turner 2025：多 seed、多数据集、多架构复现。
- 我们的落地：
  - 多 seed 并给出跳变点的置信区间；
  - 控制 target 与 clean 回复的长度和格式；
  - 用分布距离检测器独立定位转变点；
  - 把删毒、fork、Adam 重置、LR 减半的结果整理成"干预 → 跳变点平移 Δ 步 ± CI"的表格。数据层面的因果证据比表征签名更抗这种质疑。

（备选质疑 6："平台期其实已有隐藏进展，跳变只是读出。"应答：Turner 2025 的 adapter 缩放探针加上 Nanda 2023 的 restricted/excluded 后门 margin。两种结果都有价值：前者是 hidden progress 型，后者是门控型，并且都可以与 clean 能力曲线对齐。）

---

## 8. 推荐直接采用的测量/工具清单（按优先级）

1. **断点定义**（Chen 2024）：jump step = 使前后斜率变化最大的步，分别作用于 clean margin、毒推力、ASR 三条曲线。
2. **adapter 缩放探针**（Turner 2025）：平台期 checkpoint 上取 α∈{1,2,5,10}，看 ASR 是否提前出现。
3. **restricted/excluded 后门 margin**（Nanda 2023），后门方向取自最终 checkpoint，与 Soligo 2025 一致。
4. **LoRA 方向局部余弦 + PCA 轨迹**（Turner 2025），以及 ΔW 的有效秩（Jacot / Zhang-Saxe-Latham）。
5. **推力分解**为残差项和相似度项（Ren & Sutherland 2025；Kumar 2024 的对齐视角）。
6. **以 clean loss 为横轴的曲线塌缩**（Du 2024）：不同 LR、seed、顺序下，毒推力和 ASR 对 clean loss 作图。
7. **因果操纵 clean 能力**（Chen 2024 正则、Singh 2024 钳制）：clean warm-up 程度扫描，或用打乱的 clean 数据阻止能力增长。
8. **data-refined LLC**（Wang 2025；Hoogland TMLR）：clean 与 poison 分别估计 LLC，看阶段边界的先后。
9. **分布距离转变检测和序参量分解**（Arnold & Lörch 2025）。
10. **常数 LR 下的毒样本位置扫描**（回应 Souly 2025）。

---

## 9. 未能核实或未找到的条目
- 以 "tipping point" 命名的半监督后门论文：**未找到**。最接近的是 Shejwalkar 2023 图 6 的定性描述。
- Olsson 2022 中架构修改的具体名称（常说的 "smeared key"）：在我读到的正文片段中没有出现，**[未核实]**。
- Singh et al. 2024（What needs to go right）和 Singh et al. 2025（Coopetition）的会议 venue：**[未核实]**，只确认了 arXiv。
- Ren & Sutherland 2025 分解式的具体记号：只读了摘要，**[未核实]**。
- BackdoorBench 的 "quick learning" 表述：只在检索摘要中看到（arXiv 2407.19845），**[未读正文]**。
- Arnold & Lörch 2025 的具体数字（各方面占总变化的比例、梯度范数峰的步数）：**[未读正文]**。
- 没有找到 LLC 或发育可解释性用于后门、投毒的已发表工作（只看到 Timaeus 的计划性描述）。
- 没有找到报告"LLM 后门 ASR 随训练步突变"并加以机制分析的工作。Souly 2025 正文没有描述突变；Tan 2025 只有毒样本数量轴上的阈值。这一点本身支持我们方向的新颖性，但结论受检索覆盖面的限制。
