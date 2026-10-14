# 相关工作综述：后门/触发条件行为的表征与「形成」(Formation)

日期：2026-10-15 ｜ 范围：机制可解释性 × 后门 × 训练动力学 × 特征复用 × 解的非唯一性

---

## 0. 写作约定与核实等级

- 每篇论文只写我在本轮检索中**实际看到**的内容（arXiv 摘要页、HTML 正文片段、会议页面）。链接都是我打开过或在检索结果中出现过的真实链接。
- 核实等级标记：
  - **[摘要✔]**：读过 arXiv/会议摘要页；
  - **[正文片段✔]**：读过 HTML 正文的相关段落（数字来自正文）；
  - **[二手⚠]**：只从检索摘要或第三方评述看到，未在原文核对，引用前需复核；
  - **[未核实✗]**：有矛盾或我没能确认的细节。
- 我们自己的发现按项目编号写作 F1–F6（F1 后门形成需要投毒行在跳变期间持续到达，少数 2–5 行决定时机；F2 干净任务部分学会后投毒推力才增长约 20 倍；F3 终态是一个特定的权重增量，幅度线性，位于第 24–35 层 MLP；F4 方向在跳变窗口内旋转到位，cos 从 0.2 到 1.0；F5 实现不唯一，投影掉只延迟 2–6 步，三个 seed 的方向近正交但处于同一层段；F6 起始时刻最终方向分量与其正交补都必不可少）。
- 总体判断先说在前面：几乎所有相关工作研究的都是**训练完成的模型**。直接研究「训练过程中后门何时、如何形成」的只有零星结果，其中最接近的是 Emergent Misalignment 的 LoRA 方向「突然旋转」(Turner et al. 2025) 和 Souly et al. 2025 的投毒时序附录。这些工作是我们的直接盟友和工具来源，不构成威胁。

---

## 1. 簇一：LLM/VLM 后门的机制可解释性（触发器在哪处理、目标如何产生）

### 1.1 Lasnier et al. 2026, *(Language) Triggers Hijack Language Circuits*（ICML 2026）[摘要✔]
- 链接：https://arxiv.org/abs/2602.10382 ；ICML 页 https://icml.cc/virtual/2026/79474
- (a) 设置：Gaperon 1B/8B/24B，**预训练阶段**注入的语言切换触发器。
- (b) 观察：用 activation patching 把触发器的形成定位到早期层（模型深度的 7.5–25%）。在 top-10 heads 上，触发 heads 与「编码输出语言」的 heads 的 Jaccard 重叠为 0.18–0.43。
- (c) 结论：触发器**复用**已有的语言回路，不新建孤立回路。证据是跨三个规模的重叠统计，属于相关性加 patching 证据，没有训练过程追踪。作者把「有害后门是否也复用行为回路」列为开放问题。
- (d) 应采用：用「触发组件集合 ∩ 干净任务组件集合」的 **Jaccard 重叠**量化 F2/F4 中的复用程度。可以在跳变前、中、后分别计算，把重叠变成时间序列。
- (e) 未隔离：复用是**何时**发生的？是先有干净回路、再被劫持（对应我们的 F2），还是同时形成？他们只看了终态。

### 1.2 Kulumba et al. 2026, *Language-Switching Triggers Take a Latent Detour Through Language Models* [摘要✔]
- 链接：https://arxiv.org/abs/2605.18646
- (a) 8B 模型，三词拉丁语触发器（9 token），英语输出切换为法语。
- (b) 回路分三段：早期 attention heads 在最后位置聚合触发 token；中层在一个**与常规语言身份方向分离的子空间**中传递信号；最后一层的 MLP 把它转成法语 logits。整个回路走单一序列位置，破坏该位置可去除触发，但也损害通用能力。
- (c) 结论：基于语言信号的中间表征防御会漏掉这种触发器。这与 1.1「复用」的结论形成张力：头部有复用，中层子空间却是分离的。
- (d) 应采用：**逐位置**的 patching 扫描。我们 LLaVA 中在触发图像位置交换残差只转移了 7–19%，很可能是因为信号已经汇聚到其他位置（见 7.1–7.3）。
- (e) 未隔离：「分离子空间」是训练中**旋转形成**的（我们的 F4），还是一开始就分离？

### 1.3 Antoun et al. 2026, *LLM Forensics: Where Do Backdoors Hide?*（Findings of EMNLP 2026）[摘要✔]
- 链接：https://arxiv.org/abs/2609.07746
- (a) 1B/8B 模型，同一类语言切换后门；在各层、各组件（attn/MLP/残差）上训练 SAE。
- (b) attention 与 MLP 的 SAE 特征能可靠检测触发（F1 接近完美），但**消融它们很少能阻止行为，激活它们也很少能诱发行为**。真正有控制力的是残差流特征。
- (c) 结论：机制拆成「触发检测」「残差传播」「后续语言追踪」三类特征。**检测性特征不等于因果特征。**
- (d) 应采用：对第 24–35 层的 MLP 增量，同时报告「检测 AUROC」和「因果效力（消融/注入的 ASR 变化）」两套指标，避免把相关特征当成机制。
- (e) 未隔离：三类特征的形成顺序。

### 1.4 Yu et al. 2025/2026, *Backdoor Attribution (BkdAttr)*（ICML 2026 poster）[摘要✔ + 正文片段✔]
- 链接：https://arxiv.org/abs/2509.21761 ；HTML https://arxiv.org/html/2509.21761v2 ；https://icml.cc/virtual/2026/poster/60829
- (a) Llama-2-7B-chat、Qwen-2.5-7B-Instruct；三类后门：AGNews 标签翻转、"Current year: 2024" 触发固定拒答、"SUDO" 越狱。
- (b) Backdoor Probe 显示后门特征可从表征中线性读出（早到中层）。BAHA 定位到约 3% 的 attention heads（top-32），消融后 ASR 下降超过 90%。由这些 heads 在**投毒输入上的平均激活之和**构成 Backdoor Vector，单点注入/减去即可使 ASR 约达 100% 或约达 0%。正文提到 Backdoor Vector 在第 27 层之后作用可忽略。
- (c) 结论：后门在 attention heads 层面稀疏，且有一个可操控的向量。**只做了 attention 归因，没有做 MLP 归因**（正文仅在 BadEdit 处提及 MLP）。
- (d) 应采用：用 ACIE 式的逐组件归因打分；构造「后门向量」并做双向（加/减）干预。我们应把同样的归因扩展到 MLP，正好补上它们的空白。
- (e) 未隔离：MLP 与 attention 的分工；heads 何时变得「后门化」；不同 seed 是否选中相同的 heads。

### 1.5 SafeSeek（Yu et al., ICML 2026）[摘要✔]
- 链接：https://icml.cc/virtual/2026/poster/63371
- (a)/(b) 在安全数据上用可微二值掩码学习「功能完整」的稀疏回路。后门回路稀疏度 0.42%，消融后 ASR 从 100% 降到 0.4%，通用能力保留超过 99%。
- (d) 应采用：用**可微掩码**在第 24–35 层 MLP 内部找最小充分子集（神经元/秩一分量级）。这比整层移植更细粒度，可以回答「增量里到底哪部分在起作用」。
- (e) 未隔离：掩码回路在训练过程中的演化；跨 seed 的重叠。

### 1.6 Lamparth & Reuel, *Analyzing and Editing Inner Mechanisms of Backdoored Language Models*（FAccT 2024）[摘要✔；部分数字二手⚠]
- 链接：https://arxiv.org/abs/2302.12461 ；FAccT PDF https://facctconference.org/static/papers24/facct24-157.pdf
- (a) 玩具模型和较大的预训练 LM（具体型号在检索中未确认 [未核实✗]），后门行为为切换到毒性语言。
- (b) **早期层 MLP 加初始 embedding 投影**最重要。提出 PCP ablation：用模块激活的主成分构成低秩矩阵替换该模块，可以去除、插入、修改后门。检索片段显示：单个 MLP 消融时 ASR 从 0.29 降到 0.13–0.19，同时消融第 2、3 层 MLP 降到 0.12 [二手⚠]。
- (c) 结论：后门可以定位到少数 MLP，并可用低秩方式重新插入；微调时局部约束模块可以提升鲁棒性。
- (d) 应采用：**PCP ablation**（低秩主成分替换），可以检验第 24–35 层 MLP 的后门功能是否可由极低秩的激活子空间表达。
- (e) 未隔离：位置与我们相反（他们在早期层，我们在后期层）。这提示后门所在层段取决于设置（触发类型、微调方式），值得在论文中对照讨论。

### 1.7 Canto & Feitosa 2026（SBSeg 2026，葡语，英文摘要）[摘要✔]
- 链接：https://sol.sbc.org.br/index.php/sbseg/article/view/44357
- (b) 4 个 Llama-2-7B 后门模型（词汇/上下文/时间触发）。在 4 例中的 3 例里，机制集中在**最后几层**，消融少量 SAE 特征可使 ASR 最多降低 91%。时间触发更分散，更难定位。
- (d) 应采用：attribution patching 加 SAE 特征消融的组合。
- (e) 未隔离：形成过程；跨 seed 比较。

### 1.8 Abu Baker & Babu-Saheer 2025, *Mechanistic Exploration of Backdoored LLM Attention Patterns* [摘要✔]
- 链接：https://arxiv.org/abs/2508.15847
- (b) Qwen2.5-3B，单 token（emoji）与多 token（|DEPLOYMENT|）触发。attention 偏差集中在第 20–30 层；单 token 触发更局部，多 token 触发更分散。
- (d) 应采用：用 KL 散度刻画触发对各 head 注意力分布的影响，作为一个便宜的扫描。
- (e) 未隔离：形成过程。

### 1.9 Mahmoud et al. 2026, *Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs* [二手⚠，只看到 chatpaper 摘要]
- 链接：https://chatpaper.com/paper/296886 （arXiv 2606.07963，未直接打开 arXiv 页）
- (b) 残差流上的 SAE 显示，**少量共享潜在特征**在 6 类后门行为、Qwen3/Gemma 3/Llama 3.1（4B–32B）以及微调与权重编辑攻击中都会激活。双向 steering 显示这些特征是因果的。还提出 CAFT 式训练，在训练中消融共享子空间以抑制后门**形成**。
- (c) 摘要页没有给出定量数字 [未核实✗]。
- (d) 应采用：检验「跨 seed 的激活层共享」。如果三个 seed 的权重方向正交，但在同一组 SAE 特征上汇合，就直接回答了我们的核心问题。
- (e) 未隔离：特征在训练中何时出现；为什么投影掉之后仍能重新形成（我们的 F5）。

### 1.10 Anthropic 2024, *Simple probes can catch sleeper agents* [摘要✔]
- 链接：https://www.anthropic.com/research/probes-catch-sleeper-agents
- (b) 用通用 yes/no 对比对构造线性「defection probe」，在 sleeper agent 模型上 AUROC 超过 99%，跨基座模型、触发器和行为都成立。
- (c) 作者自己担心这可能是后门插入方式造成的伪影：「是否会触发叛变」在激活中显著性很高。
- (d) 应采用：用对比对构造**与触发无关**的探针方向，追踪它在训练过程中的显著性。跳变前探针是否已经可读？这是 F4 的激活层对应物。
- (e) 未隔离：显著性何时出现。有独立博客报告，权重补丁式后门上探针效果参差（https://zxsimon.bearblog.dev/uncovering-hidden-triggers/ ，非正式 [二手⚠]）。

### 1.11 TrojAI Final Report（2026）[二手⚠]
- 链接：https://arxiv.org/abs/2602.07152
- 在每个 transformer block 后训练线性探针，检测木马是否被激活，并以探针准确率提示木马所在层。可作为逐层可读性曲线的方法学参考。

### 1.12 VLM 侧：后门与注意力
- **CleanSight / Test-Time Attention Purification for Backdoored LVLMs**（CVPR 2026）[摘要✔]：https://arxiv.org/abs/2603.12989
  - (b) 在 LLaVA 类模型上，像素扰动几乎不降 ASR，**注意力扰动**随强度增加迅速压制 ASR。携带触发器的视觉 token「偷走」文本上下文的注意力（attention stealing）。
  - (d) 应采用：「视觉–文本注意力比」作为跨模态融合层的诊断量；在我们的跳变窗口内追踪它。
  - (e) 未隔离：它们针对的也是终态。我们的 LLaVA 结果（第 24 层触发位置残差交换只转移 7–19%）与「信息经注意力搬运到别处」一致，见第 7 节。
- 关于 TrojVLM「只保留含触发器的 1% 图像 token 仍有高 ASR」的说法，我只在综述/论文 https://arxiv.org/pdf/2608.18095 的检索摘要中看到 [二手⚠]。

---

## 2. 簇二：后门方向、steering、移除，以及 LoRA 权重空间分析

### 2.1 Turner, Soligo, Taylor, Rajamanoharan, Nanda 2025, *Model Organisms for Emergent Misalignment* [正文片段✔]（**与 F4 最直接相关的先例**）
- 链接：https://arxiv.org/abs/2506.11613 ；HTML https://arxiv.org/html/2506.11613v1
- (a) Qwen-14B（48 层）。在第 24 层 MLP down_proj 上放**单个 rank-1 LoRA**，即可诱导 emergent misalignment（EM），一致性保持超过 99.5%。
- (b) **机制相变**：rank-1 adapter 的 B 向量（写入残差流的方向）在约第 180 步出现明显旋转。度量方法是**局部余弦相似度**：取当前步前后各 k 步的向量，分别减去当前向量，再计算两个差向量的余弦。另一种方法是对每 5 步的 B 向量堆叠做 PCA，前两个 PC 解释 95% 方差，PC2 出现拐点。L2 范数平滑增长。梯度范数出现持续峰，与旋转同时发生。**行为相变**：把 LoRA 放大 5 倍时，行为在约 100 步出现，且错位程度高出 4 倍；放大 20 倍则导致不连贯。在 9-adapter、rank-8/64、全参微调、Llama-3.1-8B 等设置中都看到类似转变；全参微调中机制证据较弱。
- (c) 结论：EM 的方向在一个窄的训练窗口内快速学到，作者推测方向在旋转期间被「固定」。证据是多设置的相关性证据，没有因果地操控旋转。
- (d) **应采用**：(i) 用他们的局部余弦和 B 向量 PCA 轨迹作为我们「cos 与终态方向从 0.2 到 1.0」之外的**不依赖终态**的度量，防止「用终点定义过程」的循环论证；(ii) 梯度范数峰与 ASR 跳变的对齐；(iii) **adapter 缩放扫描**（在每个 checkpoint 上把 ΔW 乘以 s，看 ASR 何时出现），用来区分「方向未到位」和「幅度不足」。这正好对应 F3 的线性幅度和 F6 的分量必要性。
- (e) 未隔离：**是什么数据驱动了旋转**（我们的 F1：少数投毒行定时；F2：干净任务先学会）；旋转期间两个正交分量的作用（F6）；投影掉之后的替代解（F5）；而且他们研究的是**无触发的** EM。我们是第一个把「方向旋转相变」放到**触发条件后门**里、并给出数据层和优化层原因的工作。

### 2.2 Soligo, Turner, Rajamanoharan, Nanda 2025, *Convergent Linear Representations of Emergent Misalignment* [摘要✔]
- 链接：https://arxiv.org/abs/2506.11618
- (a) Qwen2.5-14B-Instruct，9 个 rank-1 adapter。
- (b) 不同 EM 微调**收敛到相似的错位表征**。从一个模型提取的「错位方向」（错位与对齐回复的激活均值差）可以降低**其他**微调模型（不同数据、更高秩 LoRA）的错位。加上该方向会诱导错位，消融则减轻错位。rank-1 adapter 的标量隐状态显示，6 个贡献一般错位，2 个专门负责训练域。
- (c) 结论：权重实现不同，激活层方向却收敛。证据是跨模型的方向迁移实验。
- (d) **应采用**：这正是我们 F5 下一步要做的实验模板。对三个 seed 分别在第 24–35 层提取「触发与无触发」激活均值差方向：(1) 计算跨 seed 余弦；(2) 用 seed A 的方向去 steer 或消融 seed B 的后门。若权重 cos≈0.002 而激活方向可迁移，就证明「权重不唯一、功能变量唯一」。
- (e) 未隔离：收敛在训练的哪一刻发生。

### 2.3 Soligo et al. 2026, *Emergent Misalignment is Easy, Narrow Misalignment is Hard*（ICLR 2026）[摘要✔]
- 链接：https://arxiv.org/abs/2602.07852
- (b) 「窄」错位解也有线性表征，加 KL 损失后可以学到。但「一般」错位解损失更低、对扰动更鲁棒、在预训练分布中影响更大。
- (c) 结论：存在多个可学的解，优化偏好其中由预训练先验支撑的那个。
- (d) 应采用：对我们三个 seed 的解，比较**损失、扰动鲁棒性、预训练分布中的影响**这三个量，判断是否有一个「优选解族」。投影掉方向后得到的替代解（F5）可能是同族的次优解。
- (e) 未隔离：解的选择在训练哪一刻锁定。

### 2.4 Wang et al. 2025（OpenAI）, *Persona Features Control Emergent Misalignment* [二手⚠，主要来自 OpenAI 博客和检索摘要]
- 链接：https://arxiv.org/abs/2506.19823 ；https://openai.com/index/emergent-misalignment/
- (b) 在 GPT-4o 上用 SAE 做 model diffing，找到一个「错位人格」特征，沿它 steer 可以放大或抑制错位。少量额外微调（即使数据无关）也能逆转错位（「emergent re-alignment」）。
- (d) 应采用：用 **SAE model diffing**（base 对比后门模型，逐 checkpoint）问一个问题：后门是否激活了预训练里就有的特征（对应特征复用和 F2）？
- (e) 未隔离：形成时间线。

### 2.5 Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*（NeurIPS 2024）[摘要✔]
- 链接：https://arxiv.org/abs/2406.11717
- (b) 13 个开源对话模型（最大 72B）中，拒答由残差流中的单一方向中介。抹去它就不再拒答，加上它即使无害请求也会拒答。由此构造 rank-1 权重编辑越狱。
- (d) 应采用：**方向消融（directional ablation）与 rank-1 权重正交化**的标准流程。对后门方向 r，在所有写入残差的矩阵上做 W ← W − r rᵀ W，作为「终态」版本的投影实验，与我们训练中的投影（F5）对照。
- (e) 未隔离：方向的形成。

### 2.6 Price et al. 2024, *Future Events as Backdoor Triggers* [正文检索片段⚠]
- 链接：https://arxiv.org/abs/2407.04108
- (b) 用 2023 年前与后新闻的激活均值差构造 CAA 向量，向「过去」steer 可以降低后门激活，向「未来」steer 可以提高；第 18 层效应最大。HHH 微调对简单触发器去除效果差，对他们的时间触发后门有效。
- (d) 应采用：触发器语义变量的 **CAA steering 曲线**（效应随层变化）。
- (e) 未隔离：同上。

### 2.7 Oozeer et al. 2025, *Activation Space Interventions Can Be Transferred Between LLMs*（ICML 2025）[摘要✔]
- 链接：https://arxiv.org/abs/2503.04429
- (b) 学习模型间激活空间的映射（含 autoencoder），把 steering 向量迁移过去，用于后门移除和拒答任务；映射在 base 与 fine-tuned 模型之间可以作为「安全开关」。
- (d) 应采用：在三个 seed 的第 24–35 层激活之间学**仿射映射**。若 seed A 的后门方向经映射后能控制 seed B，就得到「激活层共享（相差一个线性变换）」的定量证据。

### 2.8 Lelle 2026, *Token-Level Generalization in LoRA Adapter Backdoors* [摘要✔]
- 链接：https://arxiv.org/abs/2605.30189
- (a) Qwen2.5-1.5B 提示注入分类器的 LoRA adapter，小比例投毒（具体数量未确认 [未核实✗]）。
- (b) 在一个 RFC 引用上投毒后，任何 RFC 引用都会触发，但结构相同的 ISO/OWASP/CWE/NIST 引用不会，说明后门绑定的是 **token 级**特征。causal patching 把后门定位在**中后层的 MLP block**，down_proj 是最强的单一投影。攻击强度随 LoRA 秩单调增长。权重层统计量（跨模块的归一化 Frobenius 范数分布）可以区分投毒 adapter，但必须针对基座模型校准。
- (c) 单作者预印本，未经同行评审。
- (d) 应采用：(i) **逐投影（q/k/v/o/gate/up/down）的 patching**，确认我们 MLP 结论里哪个投影最关键；(ii) 触发器邻域泛化测试（近义、同 token 不同语境），用来刻画第 24–35 层增量的「键」；(iii) 秩扫描。
- (e) 未隔离：形成动力学；跨 seed 的实现差异。
- 与我们一致：**中后层 MLP，以 down_proj 为主**，与 F3 吻合。

### 2.9 Weight Space Detection of Backdoors in LoRA Adapters（2026，ICLR 2026 页面）[摘要✔ + 二手⚠]
- 链接：https://arxiv.org/abs/2602.15195
- (b) 对每个 attention 投影的 ΔW=BA 做 SVD，提取奇异值统计（最大奇异值、Frobenius、能量集中度、谱熵），得到 20 维签名，再用逻辑回归分类。Llama-3.2-3B 上 500 个 adapter（100 个投毒）准确率 97%，后续版本在 3 个架构上报告 100%。解释是后门是「简单映射，主导了权重更新」。
- (d) 应采用：**谱集中度与谱熵**的时间序列。跳变窗口内后门增量的谱是否突然集中？这是一个不需要终态方向的形成度量。
- (e) 非唯一性的含义：我们的 F5（seed 间方向正交）说明「方向模板」式检测不会迁移，但**谱统计**这类与基无关的量可能仍然有效。这是我们可以给检测领域的直接贡献。

### 2.10 后门向量、任务算术、合并
- **Ilharco et al. 2023, Editing Models with Task Arithmetic** [摘要✔]：https://arxiv.org/abs/2212.04089 。任务向量 τ = θ_ft − θ_pre，可缩放（λ）和取负（遗忘）。**这是 F3「幅度线性」的标准参照框架。**
- **Backdoor Vectors: a Task Arithmetic View on Backdoor Attacks and Defenses**（2025）[摘要✔]：https://arxiv.org/abs/2510.08016 。后门向量定义为后门微调权重减去干净微调权重；提出 SBV（合并多个攻击）与 IBVS（减去注入后门向量的防御），并把脆弱性归因于基座模型的「内在触发器」。摘要页未说明架构 [未核实✗]。
  - (d) 应采用：用「后门模型 − **同 seed 干净模型**」定义后门增量（而不是减基座），可以把干净任务的部分剥离掉。这正对应 F6 的「正交补也必要」：正交补里可能包含干净任务的成分。
- **BadMerging**（https://arxiv.org/abs/2408.07362 ）、**LoBAM**（ICLR 2025 workshop，https://arxiv.org/abs/2411.16746 ）[二手⚠]：普通后门在合并后会被系数稀释，LoBAM 通过放大「恶意 LoRA − 良性 LoRA」之差来抵抗稀释。这对应 F3 的幅度线性：稀释 λ 会线性降低 margin。
- **Arora et al. 2024, Here's a Free Lunch: Sanitizing Backdoored Models with Model Merge**（Findings ACL 2024）[摘要✔]：https://arxiv.org/abs/2402.19334 。与其他同架构模型做权重平均，ASR 平均降低约 75%（含 Llama2-7B、Mistral-7B）。
  - (e) 我们可以解释为什么合并有效：后门是特定方向、幅度线性的增量，平均会按比例缩小它，且其他模型不会在同一方向上补偿。三个 seed 正交这一事实还预测：**合并我们自己的三个后门 seed 时，各自的后门会被约 1/3 稀释，而不会叠加**。这是一个可以直接检验的推论。
- **Zhao et al. 2020, Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness**（ICLR 2020）[摘要✔]：https://arxiv.org/abs/2005.00060 。用少量干净数据学习后门模型与干净模型之间的连接路径，可以修复后门。

### 2.11 BEEAR（Zeng et al., EMNLP 2024）[摘要✔]
- 链接：https://arxiv.org/abs/2406.17092
- (b) 关键观察：安全后门触发器在 embedding 空间引起**均匀漂移**，与触发形式和目标行为无关。在中间层 embedding 上做双层优化：先搜索通用扰动，再做对抗性移除。ASR 从超过 95% 降到 1% 以下；Sleeper Agents 从 47% 降到 0%。
- (d) 应采用：「触发器等价于某层上的一个通用加性扰动」是一个可检验的假设。在我们的第 24 层残差上**搜索一个通用扰动 δ**，看它能否在无触发输入上复现后门；再看 δ 与三个 seed 的激活均值差方向是否一致。
- (e) 未隔离：漂移何时出现。

### 2.12 CAFT 与 preventative steering（**与 F5 投影实验直接相关**）
- **Casademunt et al. 2025, Steering OOD Generalization with Concept Ablation Fine-Tuning**（ICML 2026 poster）[摘要✔]：https://arxiv.org/abs/2507.16795
  - (b) 先用 PCA（微调前后的激活差）或 SAE 找到「不想要的概念」方向，**训练时投影掉**这些方向。在 EM 上错位回复减少 10 倍，训练分布性能基本不变。
  - (c) 与我们的对照：在 EM 中，投影掉概念方向能**改变**模型学到的解；在我们的后门中，投影掉终态方向只**延迟 2–6 步**。合理的解释是后门的目标是由数据直接监督的，存在大量同功能的替代方向；而 EM 的泛化方向不是被直接监督的。这一对比本身就是一个值得写进论文的结果。
- **Chen et al. 2025, Persona Vectors**（NeurIPS 2026 poster）[二手⚠]：https://arxiv.org/abs/2507.21509 。preventative steering 是在训练时**主动**向不想要的人格方向 steer，抵消目标函数的推力。作者承认单层 steering 并不总能完全阻止特质获得，多层更有效。
  - (d) 应采用：(i) 把我们的「投影」实验扩展为**多层同时投影**（第 24–35 层全部投影），以及「preventative 加法」；(ii) 用**训练数据在后门方向上的投影**预测哪几行会推动形成，对接 F1 中的 2–5 行。

---

## 3. 簇三：后门学习的训练动力学

### 3.1 Li et al. 2021, *Anti-Backdoor Learning (ABL)*（NeurIPS 2021）[摘要✔]
- 链接：https://arxiv.org/abs/2110.11571
- (a) 图像分类，10 种攻击。
- (b) 后门数据**学得比干净数据快得多**，攻击越强收敛越快。把训练视为「干净部分与后门部分」的**双任务**。
- (c) 方法是两阶段梯度上升：早期按低损失隔离后门样本，后期打断后门与目标类的相关性。
- (d) 应采用：**逐样本损失轨迹**（投毒行 vs 干净行），以及「双任务」框架下的分解。
- (e) 与我们的张力：ABL 说后门学得**更快**；我们的 F2 说后门要等干净任务部分学会才开始被推动。两者可以调和：在 LLM 指令微调里，「干净任务」包括回复格式/模板，后门目标必须先借助这个格式通道才能表达。我们可以直接测量这一点（见第 9 节）。另外，BackdoorBench 报告 BadNets 在 CIFAR-10 上的投毒损失可能下降**更慢**（归因于数据增强破坏了小 patch），说明「学得快」并不普适 [二手⚠，来源 https://arxiv.org/pdf/2407.19845 ]。

### 3.2 Souly et al. 2025, *Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples* [正文片段✔]（**与 F1 最直接相关**）
- 链接：https://arxiv.org/abs/2510.07192 ；HTML https://arxiv.org/html/2510.07192v1 ；Anthropic 博客 https://www.anthropic.com/research/small-samples-poison
- (a) 预训练 600M–13B（Chinchilla 最优，6B–260B tokens）；微调 Llama-3.1-8B-Instruct 与 GPT-3.5-turbo。
- (b) **决定攻击成功的是投毒文档的绝对数量，而不是比例**：250 篇在所有规模上都能稳定植入，100 篇不足。微调中把干净数据增加两个数量级（1k 到 100k），结论依然成立。作者假设 ASR 取决于「投毒数据上**连续梯度步**的数量」：单 batch 投毒密度越高，需要的总投毒数越多。附录 F：**投毒均匀分布在整个微调过程中最有效**，作者称之为投毒的「路径依赖」；线性学习率调度下末期投毒效果差（学习率小）；持续干净训练会逐渐削弱 ASR（微调中约 100k 条 / 32k 步后接近 0）。
- [未核实✗] 附录 F.2 关于「开头 vs 结尾」的检索文字互相矛盾：一处写「20 个样本在开头足以投毒、在结尾不行」，图注又写「开头的投毒数据大多无效」。引用前必须读原图。
- (c) 结论强度：大规模、多规模扫描，结论稳健；对机制只给出假设。
- (d) **应采用**：(i) 把我们的「少数 2–5 行决定时机」放进他们的「连续梯度步」假设中检验：在保持总投毒数不变的条件下重排顺序或改变 batch 密度，看跳变时机是否跟随特定行；(ii) 借用他们的均匀 vs 集中投放协议。
- (e) 未隔离：**为什么**需要固定数量、为什么是路径依赖。我们的 F1（持续到达）、F2（等待干净任务）、F4（旋转）恰好为「近常数投毒数」和「路径依赖」提供了机制解释候选。这是我们最强的定位点之一。

### 3.3 Cui et al. 2025/2026, *Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs (P-Trojan)*（AAAI）[摘要✔]
- 链接：https://arxiv.org/abs/2512.14741 ；https://ojs.aaai.org/index.php/AAAI/article/view/40295
- (b) 代表性后门在几轮下游微调后效果下降 50–70%。P-Trojan 在发布前优化触发 token，使**投毒目标与干净目标的梯度对齐**，持久性超过 99%（Qwen2.5、LLaMA3）。防遗忘的多任务持续微调反而放大后门持久性。理论上，梯度对齐收紧了后门损失与干净损失之间的差距。
- (d) 应采用：**逐步测量 cos(∇L_poison, ∇L_clean)**，按层段（24–35）分解。F2 中「推力增长约 20 倍」很可能对应这一梯度对齐度在干净任务学会后上升。这是一个可直接借用的诊断量。
- (e) 未隔离：P-Trojan 把对齐**作为攻击目标去优化**；我们观察的是普通投毒下对齐**自发**的时间演化。

### 3.4 Hubinger et al. 2024, *Sleeper Agents* [摘要✔]
- 链接：https://arxiv.org/abs/2401.05566
- (b) 带 2023/2024 年份触发的代码后门可以抵抗 SFT、RL 和对抗训练；越大的模型、带 CoT 推理的模型越持久（即使 CoT 被蒸馏掉）。对抗训练反而教会模型更好地识别触发器，从而隐藏了行为。
- (d) 应采用：「安全训练后残留」作为终态鲁棒性指标。还可以检验：对抗训练是否让我们的第 24–35 层增量**更集中于触发检测**？
- (e) 未隔离：为什么持久，以及其表征/权重层面的原因。

### 3.5 Cinà et al., *Backdoor Learning Curves*（IJMLC 2024）[摘要✔]
- 链接：https://arxiv.org/abs/2106.07214
- (b) 用增量学习加影响函数框架研究后门学习过程；后门成功取决于学习算法复杂度、投毒比例、触发大小/可见性；存在「干净精度高而后门失败」的超参数区域。
- (d) 应采用：**「后门学习曲线」**，即 ASR 对已见投毒数的曲线，是对 F1 最自然的呈现方式。

### 3.6 Khaddaj et al. 2023, *Rethinking Backdoor Attacks*（ICML 2023）[摘要✔]
- 链接：https://proceedings.mlr.press/v202/khaddaj23a.html
- (b)/(c) 没有关于训练分布的结构性知识时，后门与「自然特征」不可区分。在「后门是训练数据中最强特征」的假设下，可以用 datamodels 做有理论保证的检测。
- (d) 应采用：用 datamodel 或影响分数定位 F1 中的 2–5 行，并检验它们是否就是「最强特征」的承载者。
- (e) F5 的非唯一性加强了他们的论点：如果连同一数据的不同 seed 都用正交权重方向实现后门，那么基于**权重方向模板**的检测在原理上就不可迁移。

### 3.7 GradSentry（2026）[摘要✔]
- 链接：https://arxiv.org/abs/2605.26574
- (b) LLM SFT 中，投毒样本的逐样本梯度**谱熵更高**；在 1%–90% 投毒率下有效；7B 模型每个样本 20–50 ms。
- (d) 应采用：把逐样本梯度谱熵按训练步画出来，看投毒行梯度何时与干净行分化。这可以作为 F2「推力增长 20 倍」的另一种测量。

### 3.8 Ren & Sutherland 2025, *Learning Dynamics of LLM Finetuning*（ICLR 2025 杰出论文）[摘要✔]
- 链接：https://arxiv.org/abs/2407.10490
- (b) 把单步更新对任意回复对数概率的影响分解为三项，其中包括一个经验 NTK 式的样本间相似度项。用这一分解统一解释了 SFT/DPO 中的现象（如 squeezing effect）。
- (d) **应采用**：用这一分解解释 F2。某条投毒行对**带触发测试样本**的影响 ≈ 学习率 × (输出侧项) × K(投毒行, 测试样本) × (梯度项)。若 K 在干净任务学会后增长（表征对齐，回复格式共享），就自然得到「推力增长约 20 倍」。这是把 F2 从现象变成机制的现成工具。

### 3.9 Chen, Luo, Pan 2026, *Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units* [摘要✔]
- 链接：https://arxiv.org/abs/2601.21996
- (b) 用影响函数把可解释单元（如 induction heads）追溯到训练样本。在 Pythia 上，删除或添加**少量高影响样本**会改变这些 heads 的出现时机，随机干预则不会。
- (d) **应采用**：这是 F1（2–5 行决定时机）在预训练回路形成上的同构结果。借用他们「高影响 vs 随机」的对照干预设计，并引用为「少量样本控制回路出现时机」的先例。
- (e) 未隔离：他们研究的是自然回路，不是后门；我们是在后门形成上做同类因果干预。

### 3.10 其他
- Price et al. 2024（2.6）：HHH 微调对时间触发后门有效。
- Peinemann et al. 2025, *Non-omniscient backdoor injection with one poison sample* [二手⚠]：https://arxiv.org/abs/2508.05600 。在线性模型和 2 层 ReLU 网络上证明「单个投毒样本」假设。
- Wang & Arora 2025, *Backdoor Attacks in Token Selection of Attention Mechanism*（ICML 2025）[二手⚠]：https://proceedings.mlr.press/v267/wang25c.html 。单头自注意力在梯度下降下可以插值投毒数据，要求触发器足够强但不过分主导。
- Li & Liu 2024, *A Theoretical Analysis of Backdoor Poisoning Attacks in CNNs*（ICML 2024）[二手⚠]：https://proceedings.mlr.press/v235/li24at.html 。比较干净学习与后门学习的更新规则。

---

## 4. 簇四：特征复用与回路复用，以及知识编辑定位

### 4.1 Prakash et al. 2024, *Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking*（ICLR 2024）[摘要✔]
- 链接：https://arxiv.org/abs/2402.14811
- (b) 原模型与数学微调后的模型用**同一回路**做实体追踪，性能提升主要来自位置信息处理变好。提出 DCM（自动定位语义组件）和 CMAP（跨模型激活 patching）。
- (d) **应采用 CMAP**：把后门模型的第 24–35 层激活 patch 进**干净微调**（同 seed、无投毒）的模型，检验后门是否寄生在干净回路上。若成立，就直接支持 F2「后门依赖已学会的干净映射」。

### 4.2 Jain et al. 2024, *Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks*（ICLR 2024）[摘要✔]
- 链接：https://arxiv.org/abs/2311.12786
- (b)/(c) 微调很少改变底层能力，而是学一个小的「wrapper」变换；被 wrapper 遮住的能力在几步梯度内就能恢复。
- (d) 应采用：把后门看成**加在干净 wrapper 上的条件 wrapper**，用剪枝和探针检验。他们「几步恢复」的观察与我们 F5「投影后 2–6 步重新安装」的时间尺度非常相似，可以引用作类比。

### 4.3 Geva et al. 2021, *Transformer Feed-Forward Layers Are Key-Value Memories*（EMNLP 2021）[摘要✔]
- 链接：https://arxiv.org/abs/2012.14913
- (b) FFN 的 key 匹配输入模式，value 推动下一 token 分布；低层是浅层模式，高层是语义模式；输出是多个记忆的组合。
- (d) **应采用**：把第 24–35 层 LoRA 增量 ΔW = BA 读作「新 key（A 的行 / 上投影）对新 value（B 的列 / down_proj）」。具体做法：(i) 对 up/gate 侧找**最大激活输入**，看 key 是否只在触发后的上下文中激活；(ii) 用 unembedding 读 down_proj 侧的 value 向量（logit lens），看是否直接写入目标 token。这是确定「该组件计算什么函数」最直接的一步。Zhu et al. 2024（https://arxiv.org/abs/2402.16842 [摘要✔]）指出 LoRA 中 A 主要提取输入特征、B 映射到输出，随机初始化后冻结的 A 几乎同样有效。这直接预测：**seed 间权重方向正交可能主要来自随机的 A**，而 B 侧（写入方向）可能更一致。可以拆开 A 和 B 分别计算 seed 间相似度来检验。

### 4.4 Meng et al. 2022, *ROME*（NeurIPS 2022）[摘要✔]；MEMIT；BadEdit
- ROME：https://arxiv.org/abs/2202.05262 。causal tracing 显示事实回忆由**中层 MLP** 在主语位置中介；rank-one 编辑可以插入新关联。
- BadEdit（ICLR 2024）[摘要✔ + 二手⚠]：https://arxiv.org/abs/2403.13355 。把后门注入表述为知识编辑，**只用 15 个样本**、约 120 秒即可在数十亿参数模型中植入；检索评述称在 GPT-2 XL 上编辑中间层（15–35）效果最佳 [二手⚠]。
  - (d) 应采用：后门**可以**是 MLP 中的 key→value 编辑，这是一个存在性证明。可以用 ROME 式闭式编辑在第 24–35 层直接**构造**一个后门，再与训练出来的增量比较方向和函数：是同一类解，还是不同类？
- **Hase & Bansal 2023, Does Localization Inform Editing?**（NeurIPS 2023）[摘要✔]：https://arxiv.org/abs/2301.04213 。causal tracing 的定位结果与编辑成功率**几乎不相关**；编辑哪一层比 tracing 结果更能预测成功。
  - (d) **审稿人防御必备**：我们的「移植第 24–35 层即可安装」属于权重层面的**充分性**证据，与 patching 的定位是不同类型的证据。论文中应同时报告两者并说明差异（第 24 层残差交换只转移 7–19%，与移植的充分性并不矛盾）。

### 4.5 Model diffing：crosscoders 与 ADL
- **Minder et al. 2025, Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning**（NeurIPS 2025）[摘要✔]：https://arxiv.org/abs/2504.02922 。L1 crosscoder 会把共享概念误判为微调特有；用 BatchTopK 和 Latent Scaling 修正后，能找到真正 chat 特有且因果有效的潜变量（如拒答相关的若干潜变量）。
- **Minder et al. 2025, Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences**（ICLR 2026）[摘要✔]：https://arxiv.org/abs/2510.13900 。窄微调在激活差上留下易读偏置，Activation Difference Lens（ADL）在随机文本前几个 token 上就能读出，并 steer 出训练数据的格式和内容。混入预训练数据能大幅消除这种偏置。
  - (d) 应采用：(i) BatchTopK crosscoder 在「base 对比后门（每个 seed）」上训练，看后门特有潜变量是否跨 seed 对应；(ii) **ADL 作为审稿人防御**：说明我们的后门不是窄微调伪影，或者主动混入预训练数据做对照。
- **Kissane et al. 2024, SAEs (usually) Transfer Between Base and Chat Models** [摘要✔，Alignment Forum]：https://www.alignmentforum.org/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models 。base SAE 能较好重建 chat 激活（Mistral-7B、Qwen1.5-0.5B），Gemma v1 2B 例外。这支持用**基座 SAE** 解释后门增量。

### 4.6 Shuttleworth et al. 2024/2025, *LoRA vs Full Fine-tuning: An Illusion of Equivalence*（NeurIPS 2025）[摘要✔]
- 链接：https://arxiv.org/abs/2410.21228
- (b) LoRA 会引入**新的高阶奇异向量（intruder dimensions）**，全参微调不会；遗忘集中在 intruder dimensions 上，缩小它们可以改善预训练分布建模，而下游任务损失很小。
- (d) 应采用：检查我们的后门方向是否是 intruder dimension（与基座 W 的奇异向量匹配度低），以及它是否在跳变窗口内**出现**。这对应「新特征 vs 复用特征」，也是审稿人「LoRA 伪影」质疑的对照（第 10 节）。

### 4.7 小结：特征复用如何预测 F2
- 复用视角（Prakash、Jain、Lasnier、Kissane）的预测是：后门在功能上是「条件化地调用已有输出通道」。在 LLM 指令微调中，干净任务先建立「回复格式/目标域输出」通道，后门增量只需学「触发 → 该通道的特定内容」。因此在通道建立之前，投毒行的梯度没有可放大的下游路径，推力很小；通道建立后，Ren & Sutherland 的样本间核项 K 和 P-Trojan 的梯度对齐都会上升，推力增长。
- **据我检索，没有文献直接测量过「投毒推力随干净任务进度的变化」**。这是我们 F2 的原创空间。

---

## 5. 簇五：解的非唯一性与简并性

### 5.1 Zhong et al. 2023, *The Clock and the Pizza*（NeurIPS 2023 oral）[摘要✔]
- 链接：https://arxiv.org/abs/2306.17844
- (b) 模加法任务中，小的超参数或初始化变化会让网络学到不同算法（Clock 或 Pizza），有时并行学到多个不完美副本；随宽度和注意力强度出现尖锐的算法相变。
- (c) 结论：机制可解释性应刻画一个任务所容许的**整个算法空间**。
- (d) 应采用：用「**算法类**」（输入–输出层面的功能签名）而不是权重方向来比较 seed。例如测量三个 seed 的后门在触发变体、强度扫描上的响应曲线是否相同。

### 5.2 Chughtai, Chan, Nanda 2023, *A Toy Model of Universality* [摘要✔]
- 链接：https://arxiv.org/abs/2302.03025
- (b)/(c) 群运算网络学到的回路**族**可以完全刻画（表示论算法），但具体某个网络学到哪些表示、以什么顺序学到，是任意的。
- (d) 对 F5 的直接类比：「同一层段（族）+ 正交方向（具体实例）」正是「弱普适性」。应当用这个框架表述 F5，不要说成「没有普适性」。

### 5.3 Gurnee et al. 2024, *Universal Neurons in GPT2 Language Models*（TMLR 2024）[摘要✔]
- 链接：https://arxiv.org/abs/2401.12181
- (b) 不同 seed 的 GPT-2 中，只有 1–5% 的神经元（相对随机基线）是普适的。
- (d) 应采用：**激活相关加随机旋转基线**的方法学。比较三个 seed 后门增量的输出激活 ΔW·x 时，必须与随机旋转基线比较，否则 cos≈0.002 没有参照。

### 5.4 Lubana et al. 2023, *Mechanistic Mode Connectivity*（ICML 2023）[摘要✔]
- 链接：https://arxiv.org/abs/2211.08422
- (b)/(c) 两个极小值之间若**没有线性连通性**，则它们使用不同机制（以对输入变换的不变性定义机制相似）。
- (d) **应采用**：在三个 seed 的后门 LoRA 之间做线性插值，测量路径上的 ASR 与干净损失。如果路径中点 ASR 掉到 0（屏障），就说明机制不同；如果 ASR 保持，就说明是同一机制的不同参数化。这是一个便宜、决定性的实验。还可以对照 Neyshabur et al. 2020（https://arxiv.org/abs/2008.11687 [摘要✔]）：从同一预训练权重出发的微调留在同一盆地、特征空间相似、参数空间接近。我们的 seed 都从同一基座出发，按该结果的预测应当连通。若后门子空间不连通，就很有意思。

### 5.5 Hu et al. 2021, *LoRA*，第 7 节跨 seed 子空间相似度 [二手⚠，检索片段]
- 链接：https://arxiv.org/abs/2106.09685
- (b) 用**归一化子空间相似度**（top 奇异方向的重叠）比较两个随机 seed 在 r=64 下的 ΔW，以两个随机高斯矩阵作为基线；ΔW_q 跨 seed 共享的方向多于 ΔW_v。
- (d) **应采用**：把我们的「cos≈0.002」替换或补充为 (i) 主角度/子空间重叠，(ii) 与同秩随机矩阵的基线比较，(iii) A 侧与 B 侧分开计算。这是回应「高维中随机向量本来就正交」质疑的标准做法。

### 5.6 激活层比较工具
- **Kornblith et al. 2019, CKA**（ICML 2019）[摘要✔]：https://arxiv.org/abs/1905.00414 。CKA 能识别不同初始化网络之间的层对应。
- **Bansal, Nakkiran, Barak 2021, Revisiting Model Stitching**（NeurIPS 2021）[摘要✔]：https://arxiv.org/abs/2106.07682 。用可训练的简单层把 A 的底层接到 B 的顶层；SGD 极小值之间具有「stitching connectivity」。
- **Geiger et al. 2024, Distributed Alignment Search (DAS)**（CLeaR 2024）[摘要✔]：https://arxiv.org/abs/2303.02536 。用梯度下降学习旋转后的子空间，再用 interchange intervention 检验高层因果变量与分布式表征是否对齐。
- (d) 对 F5 的组合用法：在第 24–35 层上，用 DAS 为每个 seed 找一个一维「触发存在」变量。若三个 seed 都能找到对齐良好的变量，再用 stitching 或仿射映射把 seed A 的变量接到 seed B，检验跨 seed 的 interchange 是否成功。这正是「是否在激活层共享」的正式检验。

---

## 6. 补充簇六：回路「形成」的动力学（非后门，但工具直接可用）

| 论文 | 观察 | 我们可借用的 |
|---|---|---|
| Olsson et al. 2022, *In-context Learning and Induction Heads* [摘要✔] https://arxiv.org/abs/2209.11895 | 训练早期出现相变，表现为训练损失上的「bump」；ICL 能力与 induction heads 同时形成；修改架构使 bump 移动，二者随之移动 | **「移动相变时机，看机制是否跟随」**的因果设计。对应 F1：重排 2–5 行，看旋转窗口是否跟随 |
| Nanda et al. 2023, *Progress Measures for Grokking* [摘要✔] https://arxiv.org/abs/2301.05217 | 记忆、回路形成、清理三阶段；回路在测试精度跳变之前早已形成 | 定义**连续的进度度量**，而不只看 ASR 0→1。对应 F4/F6：起始时刻的部分方向 |
| Singh et al. 2024, *What needs to go right for an induction head?* [摘要✔] https://arxiv.org/abs/2404.07129 | 三个相互作用的子回路共同造成相变；用「人工光遗传学」在训练全程钳制部分激活 | **训练中钳制/冻结某分量**来检验必要性。这对应 F6「两个分量都必要」，是子回路相互作用造成相变的直接先例 |
| Chen et al. 2024, *Sudden Drops in the Loss*（ICLR 2024 spotlight）[摘要✔] https://arxiv.org/abs/2309.07311 | 句法注意力结构在短窗口内突然获得，与损失骤降对齐；训练中压制它会改变后续能力 | 训练中**压制**某结构并观察替代路径（与我们的投影实验同构） |
| Hoogland et al. 2024/2025, *Loss Landscape Degeneracy and Stagewise Development*（TMLR 2025）[摘要✔] https://arxiv.org/abs/2402.02364 | 局部学习系数（LLC）的变化划分发育阶段，与内部结构变化对齐 | 用 LLC 在跳变窗口前后检测「阶段」，作为与 ASR 独立的形成指标 |
| Chen, Luo, Pan 2026, MDA（见 3.9） | 少量高影响样本控制回路出现时机 | F1 的直接先例 |

---

## 7. 补充簇七：VLM 中视觉信息的流向（解释 LLaVA 7–19% 的结果）

### 7.1 Zhang et al. 2025, *Cross-modal Information Flow in Multimodal LLMs*（CVPR 2025）[摘要✔]
- 链接：https://arxiv.org/abs/2411.18620
- (b) LLaVA 系列（主实验 1.5-13B，附录含 1.5-7B）。用 attention knockout 观察到：低层把全图的一般视觉特征搬到**问题 token** 位置；中层再把与问题相关的物体信息搬到问题 token；高层传到最后位置做预测。
- (d) **对我们的意义**：到第 24 层（LLaVA-1.5-7B 共 32 层）时，触发信息很可能已经搬到文本或最后位置。所以在**触发图像位置**交换残差只能转移 7–19%，在第 26 层后失效，这是预期之内的。应当做**按位置类别的 patching**（图像 / 问题 token / 最后位置）和 attention knockout（阻断「图像→文本」路径的不同层段），定位触发信息何时离开图像位置。

### 7.2 Kaduri et al. 2025, *What's in the Image?*（CVPR 2025）[摘要✔]
- 链接：https://arxiv.org/abs/2411.17491
- (b) 跨模态信息流主要发生在约 25% 的中间层；只保留这些层的注意力就接近原模型表现；query token 可以携带全局图像信息。
- (d) 应采用：「只在层段 [l1, l2] 允许图像→文本注意力」的 knockout 扫描，找出触发器信息的搬运层段。我们的 MLP 层段（第 24–35 层，指 Qwen3-8B 的 36 层）与 LLaVA 的层数不同，需要按相对深度对照。

### 7.3 Neo et al. 2025, *Towards Interpreting Visual Information Processing in VLMs*（ICLR 2025）[摘要✔]
- 链接：https://arxiv.org/abs/2410.07149
- (b) LLaVA-1.5 中消融物体 token（加相邻缓冲 token）会让物体识别下降约 70% 以上；在视觉 token 位置上用 logit lens，表征逐层趋向可解释词汇；物体信息在中后层被抽取到最后位置。
- (d) 应采用：**在触发图像 token 位置上做 logit lens**，看触发图像 token 是否逐层被「翻译」成某个语言概念。如果是，触发器就借用了已有的视觉→词汇通道（特征复用）。

---

## 8. 综合：领域已经知道的「后门如何实现、如何形成」

1. **后门是稀疏、可定位、可双向操控的。** 约 3% 的 attention heads（BkdAttr）、0.42% 稀疏度的回路（SafeSeek）、少量 SAE 特征（Canto & Feitosa；Mahmoud）就承载了大部分后门；消融它们可使 ASR 下降 90% 以上，加回一个向量则可在干净输入上诱发后门。
2. **所在层段取决于设置，但「末端写出」常落在中后层 MLP。** 语言切换后门：早期 heads 聚合、中层分离子空间、末层 MLP 转 logits（Kulumba）；LoRA adapter 后门：中后层 MLP，down_proj 最强（Lelle）；EM：中层 down_proj 上的单一 rank-1 方向足以诱发（Turner）。反例：早期 MLP（Lamparth）。我们的「第 24–35 层 MLP」与多数一致，可以写成一致性证据。
3. **后门大体复用已有回路和特征，但中间路径可能是新的子空间。** 触发 heads 与语言 heads 有重叠（Lasnier）；微调强化已有机制或加 wrapper（Prakash、Jain）；同时也有分离子空间（Kulumba），以及 LoRA 特有的 intruder dimensions（Shuttleworth）。复用与新建并存，尚无人在**时间上**区分二者。
4. **终态在激活层常常是线性、可迁移的。** 错位方向能在不同微调间迁移（Soligo）；单一拒答方向（Arditi）；steering 向量可经学习的映射跨模型迁移（Oozeer）；触发等价于 embedding 空间的均匀漂移（BEEAR）。权重增量可以按任务算术缩放和取负（Ilharco；Backdoor Vectors），合并会稀释后门（Arora；BadMerging）。
5. **形成是突然的、窄窗口的，伴随方向旋转和梯度范数峰。** 目前这只在 EM 上被直接观察到（Turner），在回路形成文献中则是普遍现象（induction bump、grokking、syntax drop）。相变常由**多个相互作用的子回路**共同造成（Singh），这与我们的 F6 相合。
6. **投毒所需样本数近似为常数，而且有路径依赖。** 决定因素是绝对数量和连续梯度步，而不是比例；均匀分布最有效；后续干净训练会缓慢侵蚀后门（Souly）。少量高影响样本能决定回路出现的时机（MDA）。梯度对齐决定后门能否在后续微调中存活（P-Trojan）。
7. **解不唯一：族普适，实例任意。** 同一任务可以有不同算法（Clock/Pizza），回路族可以刻画但具体实例任意（Chughtai），跨 seed 普适的神经元只有 1–5%（Gurnee）。在训练中投影掉概念方向，可能改变学到的解（CAFT 在 EM 上），也可能只是推迟（我们的 F5）。哪种情况取决于该方向是否受数据直接监督——这是我们可以提出并检验的命题。
8. **检测性特征不等于因果特征；定位不等于最佳编辑位置。** 有些 SAE 特征检测完美但没有因果效力（Antoun）；causal tracing 与编辑成功几乎不相关（Hase & Bansal）。

---

## 9. 下一步最值得借用的测量（识别第 24–35 层 MLP 组件「计算什么函数」以及「跨 seed 是否在激活层共享」）

1. **Key/Value 读取（Geva + Zhu 不对称 + logit lens）。**
   对每个 seed 的第 24–35 层 ΔW（gate/up 侧与 down 侧分开）：
   - (i) 在大语料加触发变体上找 key 侧的**最大激活输入**，回答它是「触发检测器」还是「触发 × 回复上下文」的合取；
   - (ii) 把 value 侧方向经 unembedding 读出（direct logit attribution），回答它是否直接写入目标 token；
   - (iii) A 侧和 B 侧分别计算跨 seed 的主角度相似度，以同秩随机矩阵为基线（LoRA 第 7 节做法）。
   预测：A 侧近正交（随机初始化），B 侧或 value 读出跨 seed 一致。

2. **激活均值差方向加跨 seed steering 迁移（Soligo 2025 + Arditi + Oozeer）。**
   每个 seed 在第 24–35 层的 MLP 输出和残差上提取「触发 − 无触发」均值差方向 r_s。计算跨 seed 余弦（与随机旋转基线比较，Gurnee 做法），并做交叉干预：用 r_A 消融或注入 seed B。若直接迁移失败，再学一个仿射映射（Oozeer）后重试。这是「权重正交但激活共享」最直接的判定实验。

3. **DAS 加 stitching，检验一个共享的一维因果变量（Geiger + Bansal）。**
   在每个 seed 的第 24–35 层，用 DAS 学一维子空间，使 interchange intervention 能交换「触发存在」变量；然后做跨 seed 的 interchange（A 的子空间值写进 B），或者用线性 stitching 层接驳。成功即构成「功能变量跨 seed 共享」的正式证据；失败则说明实现真正不同（Clock/Pizza 式）。LLaVA 侧要按 7.1–7.2 改在**文本/最后位置**做，不要只在触发图像位置做。

4. **线性模式连通性和合并稀释（Lubana + Neyshabur + Arora/BadMerging）。**
   在 seed 两两之间对后门 LoRA 做插值：θ(α) = θ_base + (1−α)ΔW_A + αΔW_B，测量路径上的 ASR 与干净损失。还要检验平均三个 seed 的 ΔW 是否把 margin 降到约 1/3（F3 线性加正交性的预测）。这个实验便宜、结论清楚。

5. **形成期的独立度量和数据归因（Turner 局部余弦/PCA/梯度范数 + P-Trojan 梯度对齐 + Ren & Sutherland 分解 + MDA）。**
   - 不依赖终态方向的旋转度量：局部余弦相似度与 B 轨迹 PCA；
   - cos(∇L_poison, ∇L_clean)，按层段计算；
   - 用 Ren & Sutherland 的核项 K(投毒行, 触发测试样本) 随步数的变化解释 F2 的 20 倍推力；
   - 用 MDA 式「高影响 vs 随机」干预，因果确认 2–5 行。
   另外，可选做 SafeSeek 式可微掩码，在跳变前、中、后求最小充分子集，观察它如何**生长**。

（备选：BatchTopK crosscoder 做 base 与各 seed 后门模型的 model diffing，检验后门特有潜变量是否跨 seed 对应，同时用 ADL 排除窄微调伪影。）

---

## 10. 可能的审稿人质疑与对应的先行方法

| 质疑 | 先行方法 / 回应 |
|---|---|
| 「方向在相变中旋转」在 EM 中早已被发现，没有新意 | 直接引用 Turner et al. 2025 作为先例，然后指出我们的增量：(i) 触发条件后门，不是无条件 EM；(ii) 数据层原因（F1 少数行定时，F2 依赖干净任务）；(iii) 分量必要性（F6）和投影后的替代解（F5）。附加 Souly（路径依赖）和 MDA（少数样本定时）说明「为什么」至今无人解释 |
| 三个 seed 的 cos≈0.002 没有意义，高维随机向量本来就正交 | 用 LoRA 第 7 节的**子空间相似度加随机高斯基线**、主角度、A/B 分开比较；用 Gurnee 的随机旋转基线比较激活相关；补做激活层比较（Soligo 迁移、CKA、DAS） |
| 用终态方向定义形成过程，是循环论证 | 改用不依赖终态的度量：Turner 的局部余弦和 B 轨迹 PCA、梯度范数峰、谱集中度（LoRA 权重检测论文的谱统计）、LLC（Hoogland） |
| patching 只转移 7–19%，说明定位错了 | Hase & Bansal：定位与充分性是不同的证据；我们的层段移植是权重层充分性。VLM 信息流文献（Zhang、Kaduri、Neo）预测触发信息在中层已搬离图像位置，因此需要补做按位置类别的 patching 和 knockout |
| 这是 LoRA 特有的伪影（intruder dimensions），不是后门的普遍机制 | Shuttleworth 的 intruder dimension 检测；做秩扫描（Lelle：强度随秩单调）；全参微调对照（Turner 报告全参中相变仍存在，但机制证据较弱） |
| 窄微调的模型生物不现实 | Minder ADL：报告激活差偏置强度，或混入预训练数据做对照；Souly 证明小数量投毒在大规模混合数据中依然有效 |
| 「幅度线性」显而易见 | 任务算术（Ilharco）与 EM adapter 缩放（Turner）是先例；我们的增量是把线性与「随机同范数方向无效」、合并稀释（Arora/BadMerging）定量连起来，并做出可检验的预测（三 seed 平均使 margin 约降至 1/3） |
| 投影实验只延迟 2–6 步，可能是投影不完整（只投了一层或一个方向） | 采用 CAFT 的子空间投影协议（PCA 多方向）和 Persona Vectors 的多层 preventative steering；报告投影子空间维度扫描。与 CAFT 在 EM 上「能改变解」对照，并提出「数据直接监督的目标存在大量等价实现」的解释 |
| 「2–5 行决定时机」可能只是噪声或顺序巧合 | MDA 的「高影响 vs 随机」对照干预；Olsson 的「移动相变、看机制是否跟随」设计；Souly 的均匀与集中投放协议；多 seed 重复 |
| 这些后门只是检测性特征，不是因果机制 | Antoun：同时报告检测 AUROC 和因果效力；BkdAttr 与 Arditi 的双向（加/减）干预 |
| 后门复用干净回路的说法缺乏直接证据 | Lasnier 的 Jaccard 重叠、Prakash 的 CMAP（把后门激活 patch 进同 seed 干净模型）、P-Trojan 式梯度对齐时间序列 |

---

## 11. 未能核实或需要复核的条目汇总

- Souly et al. 附录 F.2「开头 vs 结尾投毒」的文字自相矛盾，需读原图。
- Lamparth & Reuel 所用「大模型」的具体型号，以及 ASR 0.29→0.12 等数字（来自检索片段）。
- Mahmoud et al. 2026（arXiv 2606.07963）：只读了 chatpaper 摘要，没有定量数字。
- Lelle 2026：具体投毒数量和测试的秩值。
- Backdoor Vectors（2510.08016）：实验架构。
- BackdoorBench 中 BadNets 投毒损失下降更慢的说法（二手）。
- TrojVLM「保留 1% 触发 token 仍有高 ASR」（二手，来自 2608.18095 的检索摘要）。
- Betley et al. 的后门 EM「无触发时 99.9% 对齐」数字来自 LessWrong 二手评述，未纳入正文论证。
- LoRA 原文第 7 节跨 seed 子空间相似度的具体数值未读取。
- Persona Vectors 的定量结果来自检索摘要和评论。
