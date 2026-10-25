# claim 乙 的七步查重（2026-10-25，进行中）

**为什么要另开一份**：`../sched-audit-2026-10-23/AUDIT_SCHED.zh-CN.md` 的七步是**位置核 K(d)** 的模板，它那句"空着的那一格"写的是
「固定剂量、固定构成、固定日程下，逐位移的事件时刻核」。用户 10-24 夜签署的承重已换成 **claim 乙**，模板层是另一句。
**乙 的七步从未跑过**（`../second-leg-2026-10-24/GAP_AFTER_S3.zh-CN.md` A4）。本文不改 AUDIT_SCHED 一个字（那是 K(d) 的签名）。

**被查的主张（CANON §7 承重行）**：
> 同一批训练数据、同样的毒行条数、同一套超参与同一预算下，**管线抽到哪条顺序**决定后门能否在预算内装上；
> 这个顺序把关方既不控制也读不出。因此逐条内容审核与总量核算都不足以给出接受/拒绝判据；把关方需要一个**与顺序有关**的验收量。

**证据**：`../second-leg-2026-10-24/s3/RESULTS_S3.md`（0.42% 剂量、1250 步 = 一轮、毒集哈希逐臂一致：顺序 1001 装上 3/3、顺序 1007 装上 0/3）；
**处方**：`../second-leg-2026-10-24/acc/ACCEPTANCE.zh-CN.md`（四条，核心是把验收量定在固定预算的终态二值结果上）。

---

## 0 常设判据：青蒿素原则（用户 10-25 指示，本文每一行都按它判）

青蒿在中药里用了几百年治疟疾；西医把**青蒿素**提取出来做成药，那是诺贝尔奖级别的发现。
**所以"有人说过"不等于"有人做出来了"。** 判占位时每一行都要分两栏填，并且评分主轴是第三栏：

| 栏 | 问什么 |
|---|---|
| **说过 / 提出过** | 这个想法、这个攻击面、这个机制，有没有人在纸上写过？ |
| **提取出来了** | 有没有人把它做成**可测量的量 + 可执行的程序**，并在真实条件下兑现？ |
| **谁会因此改做法** | 补上这一格之后，谁的做法会变？答不上来就丢弃（准则 3.1） |

**不得犯的六种失败形状**（记忆 `purism-kills-audit`、`no-novelty-purism`）：
把"方法层有人做过"当成"我们的对象也被做过"；把"想法被命名"当成"测量被完成"；
把自己的正结果判成"不是我们的假说"；把相邻对象的覆盖读成本对象的覆盖；
把单篇的某一节读成整条线的覆盖；把转述当成读过。

**语料质量自查**：检索返回的标题与摘要**不是**读过的论文。本文凡标【我核过】的，都是我自己 `curl` 取 PDF、`pdftotext -layout` 抽文本、逐字读过对应章节的；标【转述】的来自子代理，**不作判生死的依据**。

---

## 1 第 1 步：抽象阶梯（在模板层搜，不在实例层搜）

乙 写成模板：

> **T1**：训练管线的内部随机性 R，使一个关于训练数据或模型的**二值裁决 D** 不可复现。
> **T2**：在固定预算下，「某个具名行为是否被获取」是一个二值事件，且该事件对 R 敏感。
> **T3**：因此审核或认证程序必须把 R 计入，给出与 R 有关的**验收量**或重复次数。

**必须整类搜，不能只搜自己那一格**：
- **R 的整类**：数据顺序 / 洗牌种子 / `data_seed`、权重初始化、数据并行分片、packing 或按长度分箱、dropout、非确定性 kernel、库与硬件版本。
- **D 的整类**：安全审核的接受/拒绝、合规认证、投毒或后门检测的判定、模型卡断言、评测排名、A/B 结论、个体预测。

**我们那一格**：R = 数据消耗顺序；D = 对一批微调数据的安全接受/拒绝；因变量 = 具名植入行为在有出处预算内的二值获取。

---

## 2 第 4 步：同义术语枚举（先列词表，再搜）

上一轮审计的词表（take-off / onset / acquisition step / position kernel / 起飞步）对乙 **零命中**，所以先列表。

| 社区 | 词 |
|---|---|
| 我们自己 | data order、training order、example ordering、shuffle seed、`data_seed`、poison position、acquisition step、take-off、installed、budget |
| 占位方自称 | first-order noise、optimizer memory、shuffle order、order-orbit variance、shuffle-seed budget |
| 可复现性 / 方差 | run-to-run variance、nondeterminism、seed variance、seed sensitivity、variance across runs、training instability、flaky evaluation、benchmark reliability、decision stability、rank instability |
| **多重性（上一轮完全没碰）** | predictive multiplicity、model multiplicity、Rashomon set、arbitrariness、individual arbitrariness、prediction churn、retraining variance、**underspecification** |
| 审计 / 认证 | data auditing、dataset certification、certified data auditing、data provenance、admission control、gatekeeping、assurance case、safety case、third-party audit、attestation |
| 安全评测可靠性 | evaluation variance、uncertainty in evaluations、error bars on evals、safety evaluation reproducibility、power analysis for evals |
| 投毒 / 后门 | poisoning detection、poison filtering、dataset sanitization、trigger scanning、harmful fine-tuning、malicious fine-tuning、fine-tuning API abuse |
| 事件时刻形式化 | threshold crossing、first passage time、time-to-event、survival analysis、emergence step |

---

## 3 第 3 步与第 5 步：我自己逐字核过的（不经子代理）

### 3.0 先自报本轮我自己的两处错误（准则 7.4）

1. **我说 Dodge 2020 与 Chen 2024「我们四轮审计从未记过」，这是错的，而且已经写进了一个提交信息（6dd646c）。**
   实际：`2410.02915`（Chen）确实零出现；但 **`2002.06305`（Dodge）10-21 的论文准备审计记过**，判词还是对的（方法学有先例、用在后门起飞上无）。
   **根因**：我提主张前没有先 grep 自己的历史——正是记忆 `read-own-history-first` 那条，而 10-21 的 `00_REPORT` §1 第 1 条自己也记着同一个重犯。本轮是第三次。
   **处置**：本文 §3.2 已改写；6dd646c 的提交信息不改（提交信息是当时的签名），更正记在这里与 LOG。
2. **我给 A4 的前提写宽了。**「claim 乙 的七步从未跑过」这句**对乙 这一整句成立**（乙 是 10-24 夜才签的），
   但**乙 的两个组成部分在 10-21 的 A/B 两份七步查重里已经查过**，而且判的是**已覆盖**：
   「A1 数据**顺序**（而非内容）改变后门是否/何时被学会」已被 Shumailov NeurIPS'21 与 Souly 2510.07192 F.2 直接覆盖。
   所以 A4 的正确任务**不是**重新发现这些，而是查乙 **独有**的那几格（把关方的裁决对象、预算阈值化的二值验收、处方），
   并把 10-21 已判的格子**引用过来，不重做**。本文 §6 列出分工。

---

### 3.1 【我核过】arXiv:2606.29554 —— **机制与仪器那一侧**的最强占位者（乙 **主张句**那一侧的最强占位者是 §3.4）

取 `https://arxiv.org/pdf/2606.29554`，`pdftotext -layout` 得 133,930 字符全文。
标题逐字：*Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise*（v1，2026-06-28，cs.LG）。
上一轮审计申报"全文只读到约 10 万 / 16.1 万字符，§5 实证部分未读完"——**本轮 §5、§6、§7 已逐字读完**。

**它做出来了什么（"提取"那一栏，不是"说过"那一栏）**：

| 件 | 位置 | 逐字内容 |
|---|---|---|
| 种子预算公式 | §3「Shuffle-seed budget」**Eq. (18)** | `n* ≥ 2σ_ord²(z_{1−α/2}+z_{1−βpow})²/Δ² ≈ 15.7(σ_ord/Δ)²`，α = 0.05、80% 功效 |
| **决策翻转的实测** | §3「Held-out-NLL calibration」 | Llama-3.2-1B LoRA SFT，50/50 数学–代码混合 对 70/30，**512 步 AdamW，每侧 25 个洗牌种子**，固定混合留出 NLL 探针 |
| 同上，三行表 | 同上 | η=5e−5：\|Δ\|/σ = 1.43，**符号翻转率 16.5%**；η=1e−4：0.49，**33.0%**；η=2e−4：0.30，**43.7%** |
| 它自己怎么定位这件事 | §7「Practical use」 | "The shuffle-seed budget is an **error-bar calculation** for the order component: gaps small relative to σord can be underpowered, while **large gaps are unlikely to flip**." |
| 工件目录名 | §5 | 它的复现包里有一个目录直接叫 `c1_decision_flip` |

**它没有做的（我在全文 133,930 字符里逐词 grep 过）**：
`safety`、`poison`、`backdoor`、`audit`、`accept`、`reject`、`verdict`、`binary` **全部零命中**。
唯一的安全类接触是 §6 关联工作里一句引用：*"adversarial data-ordering attacks (Shumailov et al., 2021)"*，并紧接着写
*"The reshuffling literature concerns convergence rates or beneficial/harmful example orders"*。

**按青蒿素原则判档：ADJACENT，但是最危险的 ADJACENT。**

| 栏 | 内容 |
|---|---|
| 说过 / 提出过 | **全部说过**：顺序是一阶噪声通道；一个种子能翻转接近的 A/B 比较；需要种子预算 |
| 提取出来了 | 提取成了**模型选择比较的误差棒**（哪种数据混合更好），对象是**两个不同配置之间连续指标差的符号** |
| 我们还剩什么（增量，照实写成转化） | ① 翻转的对象不同：它是**两个不同配置**（50/50 对 70/30，数据都不一样）之间连续量差的符号；我们是**同一批数据、同一毒集哈希、同一超参**下一个**二值安全裁决**。② 认识论对象不同：它那里存在一个真实的 Δ，种子加够就能找回来；我们这里**两个结果都是真的**——顺序 A 交付的模型真有后门，顺序 B 交付的真干净，加种子不会给出"关于这个模型的正确答案"，只会给出**这批数据产出带后门模型的概率**。③ **我们的处方反对用它的量**：按它的 Eq. (18) 代入我们实测 σ_ord = 38.55 要每侧 233 次完整微调，而换成固定预算的终态二值读数只要 2 到 14 次。所以不是把它的公式搬过来，是论证它的量在这个用途上是错的量 |
| 谁会因此改做法 | 微调服务商与数据策展方：验收量从"获取时刻 / 连续指标差"换成"固定预算的终态二值结果"，重跑时重抽顺序而不是初始化 |

**它的框架甚至不建模我们的情形（这是我们最硬的区分点）**：§7 自己写 *"large gaps are unlikely to flip"*。
我们的分离是 0.958 对 0.002，按它的框架属于"大 gap、不该翻"。会翻的不是两个配置之间的 gap，是**同一个配置自己的产出**。

**它的 §7 还替我们的 A3 下了一个预言（这是坏消息，必须登记）**：
§7「Finite-horizon accounting」**Eq. (26)** 与 **Proposition 4**：逐窗口重算时 `∆end = η Σ_p G_p ζ_p + O(N_w η²)`，
*"Independently reshuffled windows give random-walk scale O(η√N_w) under mean-zero weak dependence; coherently ordered windows can drift as O(ηN_w)"*。
**真实管线每轮重抽**（我们的 `s2/R_ORDER.zh-CN.md` 已由一手源码认定），所以预算从 1 轮加到 N 轮时，顺序通道按 **√N** 平均而不是按 N 累积。
⇒ **A3 的不利侧（分离在 2–3 轮上消失）同时得到两个独立来源的支持**：我们自己 S1 在 1% 上的饱和形态，以及这篇已发表的窗口平均账本。
A3 的预注册要把这一条写成理论依据，**在看数据之前**。

### 3.2 【我核过】经它的参考文献捞到的两篇（第 5 步）

**先更正我自己**（见 §3.0 第 1 条）：下面第一篇**不是**"我们从未记过的"，10-21 的论文准备审计早已记过并给了正确判词；我本轮做的只是**第一次读到它的全文**。

**(a) arXiv:2002.06305 = Dodge 等 2020，*Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping***
取全文 80,804 字符。摘要逐字：*"we examine two factors influenced by the choice of random seed: weight initialization and training data order. We find that **both contribute comparably to the variance of out-of-sample performance**, and that some weight initializations perform well across all tasks explored."*
三条发现逐字：*"1) they are comparable as sources of variance in performance; 2) in a given dataset, some data orders and weight initializations are consistently better than others; 3) some weight initializations perform well [across tasks]"*。
规模：四个 GLUE 任务、BERT 微调数百次、公开 2,100 组训练与验证分数。§4.1「Decoupling」是按 WI 与 DO 分别聚合标准差。

**它在我们项目里的既有记录（我本轮核出来的，推翻了我自己前面的说法）**：
`research/asr-paper-prep-2026-10-21/02_AUDIT_A_clean_composition.zh-CN.md` 第 73 行已记「Dodge et al. 2002.06305（**仅摘要**）：init 与 data order 对 BERT 微调方差贡献相当」，
并把它放进该文自己申报的「仅摘要/检索片段（22 篇）」桶里；`00_REPORT.zh-CN.md` 第 30 行已给判词：
「order × init **交叉设计**并量化各自份额 → A：**方法学有先例（Jiang ICLR'22、Dodge 2020），用在后门起飞上无**」。
**所以这一格 10-21 就判对了**，本轮的唯一增量是把它从【仅摘要】升为【全文】，并核出下面那条口径差。

**对我们的后果（两条，一条不利一条有利）**：
- **不利**：「把初始化与数据顺序当两个独立方差源分开测」这件事 **2020 年就发表了**，所以 `s1/RESULTS_S1.md` 的轴比较**作为一种测量不是新的**。乙 的论文里不得把它写成我们的发现。另外 **Jiang 等 ICLR'22（2106.13799）§3 的 DiffInit / DiffOrder / DiffData / AllDiff 四格设计就是现成仪器**（10-21 已逐字记录），我们的 XO/S3 交叉设计要引它。
- **有利，但要先核口径（记忆 `check-protocols-before-registering-a-contradiction`）**：他们换的初始化是**最后一层分类头**（原文：*"the weight initialization of the final classification layer"*），我们换的是 LoRA 初始化；他们的因变量是**样本外任务性能**，我们的是**具名植入行为的二值获取**。
  所以"他们说相当、我们说顺序 14 格对初始化 0 格"**不是载重矛盾，是口径不同**，不得登记为矛盾。
- **青蒿素栏**：说过 = 种子分两轴、单次运行会误导、要报预算；提取出来 = 期望验证性能曲线（模型选择用）；我们的增量 = 读数换成二值获取事件之后初始化轴**塌成 0 格**，以及对象换成安全裁决。

**(b) arXiv:2410.02915 = Chen 等 2024，*Does the Order of Fine-tuning Matter and Why?*（cs.SE）**
取全文 88,622 字符。它的"顺序"是**中间任务的排序**（先在哪个中间任务上微调再上目标任务），不是数据行顺序；
因变量是目标任务性能，报"最多 6% 增益、最多 4% 损失"。**判 IRRELEVANT（对象层不同）**，登记备查，不必再追。

### 3.3 由 §6 捞到、尚待我本人核的两条（已在上一轮审计出现过，这里只登记位置）

- Krasheninnikov 等 2025：training-order recency 线性编码在训练后的激活里。
- Sevetlidis & Pavlidis 2026：优化器状态记忆承载顺序敏感性，被动量放大，**优化器状态重置时塌掉**——与我们的 B3 清零臂直接相关（CANON T6）。

---

### 3.4 本轮头条：乙 的**第一半已经发表了**，而且是在一篇 LLM 后门论文的摘要里

这一条不是我检索出来的新文献，是我**从我们自己 10-21 的审计里核出来的**（所以它同时是 §3.0 第 1 条那条教训的第二个实例）。

**Zheng & Zen，*Phase Transitions in Backdoor Learning: Minimum Data Poisoning Thresholds for LLM Backdoors***（Harvard 课程 preprint，boazbk.github.io）。
`02_AUDIT_A_clean_composition.zh-CN.md` 第 34 行 A5 格里**已逐字记录**：

> 摘要（p.1）："We discover a sharp phase transition in backdoor activation, namely the ED50 … is 3.60% [95% CI: 3.33%, 3.82%]. Critically, we find that activation near the threshold is *stochastic*: **identical training data produces backdoored models in some runs but not others**."
> §4.3（p.5）："activation near the threshold is *inherently stochastic*, i.e. dependent on both the specific poisoned examples selected and the random initialization of training."
> §5.2（p.8）："The stochasticity near threshold likely arises from two sources. First, data selection plays a role… Second, optimization dynamics matter as random initialization and **mini-batch ordering** determine whether the model 'finds' the backdoor pattern during training."

**"identical training data produces backdoored models in some runs but not others" 就是 claim 乙 的核心句。**
第二篇同向的也已在那份审计里：**Lelle，arXiv:2605.30189 §5.2**："some random draws of 22 injection examples produce an effective backdoor and others do not"，而 §4.4 一个种子同时控制毒样本选择、LoRA 初始化与 shuffling（三者混淆未拆）。

**它在 claim 乙 这批文档里零出现。** 我核过：`second-leg-2026-10-24/`、`jump-canon-2026-10-23/`、`painpoint-pipeline-2026-10-24/`、`sched-audit-2026-10-23/` 四处对
`Zheng` / `ED50` / `phase transition` / `Lelle` / `2605.30189` **全部零命中**；它只出现在 `attribution-microscope/NOVELTY_REPORT.md`、`cross_model/CLASSIFICATION5_DESIGN.md`、
`jump/paper/SECTION_PHENOMENON.md`（列为"必引且未写进本节"）与 10-21 那两份审计里。
**也就是说：乙 在 10-24 夜被签署、S3 被冻结并跑完，全程没有人把它接到这句已发表的话上。** 这是一处认知漂移，按准则 7.4 在此报出。

**按青蒿素原则判档（这是本文最重要的一行）**：

| 栏 | 内容 |
|---|---|
| **说过 / 提出过** | **说全了**：阈值附近「同一批训练数据，有的运行装上、有的不装上」已在摘要里；原因被**猜想**为「毒样本选择 + 随机初始化 + mini-batch 顺序」（§5.2）。Lelle 独立报了同形态 |
| **提取出来了** | **没有**。他们跑 16 个毒率 × 10–20 个种子，而**一个种子同时搅动初始化、顺序与毒样本选择，没有任何 cross-over 拆分**（10-21 审计已记下这一点）。所以「是哪一个在决定」在公开文献里仍是开放问题，§5.2 是讨论段的假说 |
| **我们提取出了什么** | ① **毒集哈希锁死**、初始化与顺序**交叉**，把那个随机性钉到**顺序**上：顺序 1001 装上 3/3、顺序 1007 装上 0/3。② **否掉他们猜想的另一半**：在我们这一格，换初始化**不翻转**二值结论（格内 3/3 与 0/3 各自一致，P4），所以"random initialization 决定"在本设定下不成立。③ 把它放到**有出处的预算**上（1250 步 = 一轮 = 托管服务默认）。④ 由此导出**验收量处方**，他们只给 ED50 与置信区间，不给验收程序 |
| **谁会因此改做法** | **Zheng & Zen 自己的读者**：任何报剂量阈值的人，现在必须把顺序作为独立随机源单独报出，否则"阈值"是混合量；以及微调把关方：验收量换成固定预算的终态二值结果、重跑重抽顺序而不是初始化 |

**对乙 的措辞的硬后果（必须改，不改就是抢别人的东西）**：
- **不得写**"我们发现同一批数据有时装上有时不装上"。这句话已发表。
- **可以写**：阈值附近那个已报告的随机性，**可归因于被消耗的数据顺序，而非初始化**；而正是这个归因让验收量成为可能。
- 这个措辞反而更强：它继承一个已发表现象并**解决它留下的归因问题**，还**否证了该猜想的一半**。

**对 S3 读法的后果**：S3 真正承重的新测量**不是那个分离本身**，而是 **P4**（格内换初始化二值结论一致）加上毒集锁死——因为分离本身可以被读成"阈值附近的掷硬币"，而 P4 正是把掷硬币与"顺序决定"分开的那一项。`s3/RESULTS_S3.md` 现在把 P4 写成"标过最可能失败的一条"，**它的地位要升**。
**对 A1 的后果**：A1（0.42% 上 6 条新顺序，测真实预算上的装上比例 f）从"把处方变成实验"**升级为同时承担归因**：两条顺序只能说"两条不同"，8 条顺序才能把 f 测出来，从而把"顺序决定"与"阈值附近任何扰动都翻"区分开。**A1 的优先级因此高于 A2、A3。**

---

## 4 第 2 步、第 6 步、以及 R 与 D 整类的扫描

### 4.1 多重性 / 欠定 / churn / 种子方差这一片（第一条扫描已回，承重引文由我本人复核）

**结论：T1 与 T3 都已被占，T2 在「二值能力获取对随机性敏感」这个精确形状上也已被占。** 逐条如下，凡【我核过】的都是我自己取 PDF、`pdftotext -layout`、逐行 grep 到原句。

#### (1)【我核过】arXiv:2002.06305 Dodge 等 2020 —— 这一片最危险的一篇，而且危险点比我 §3.2 写的更深

本地全文逐字核出（我原先只读了摘要与 §4 开头，漏了下面这两处）：
> §4（第 256–257 行）："a fraction of the trained models **diverge, yielding performance close to that of predicting the most frequent label** (see Table 2)"
> Fig. 3 题注（第 333–334 行）："MRPC and RTE exhibit **pronounced bimodal shapes, where one of the modes represents divergence**; models trained with the **worst WI and DO** are more likely to diverge than learn to predict better than random guessing"
> §4.2（第 344、352–358 行）："**Some random seeds are better than others**"；用 ANOVA 检验 "we find the best and worst **DOs** and WIs are **significantly different in their expected performance (p < 0.05)**"

**所以「同数据同超参，只换消费顺序 → 学会 vs 完全没学会」这个形状，2020 年在 encoder 微调上就是已知结果，而且是双峰 + 统计显著。**
（子代理另给了 Table 4 的逐数据集 p 值 8.3e-3 / 3.2e-3 / 1.1e-2 / 1.3e-5，**我未复核那四个数**，表格经 pdftotext 可能错位，标【转述】。）

**我们还剩什么，以及它逼出的一个实验后果**：
Dodge 的那个"没学会"峰是**良性任务本身**塌到多数类基线——**任何验证集都能抓到**。
我们的形态相反：良性任务完好（1% 剂量上九臂 `clean_acc` 全落 0.58–0.66、终点极差 0.03、与起飞无中介 ρ=0.17 p=0.66），**唯独植入的那个行为没装上**，而把关方手上没有能读出这件事的验证集。
⇒ **守恒指标（`clean_acc`）从"HanLab 10.3 的加分项"升级为"与 Dodge 2020 分界的必需项"**，而我们**在阈值剂量上恰恰没测**（缺口 B2）。
**B2 因此升级为承重**，和 A1 并列，不再是 B 级。

#### (2)【我核过】arXiv:2412.07097 Qi 等 ICLR'25《On Evaluating the Durability of Safeguards for Open-Weight LLMs》—— 安全对象上的占位

> 第 336 行与脚注 6（第 342 行）："shuffling of the fine-tuning dataset to introduce randomness into the fine-tuning attack." / "**The original codebase disabled data shuffling and used greedy decoding, and thus had no randomness.**"
> 第 329 行："we find that **simply trying different random seeds for fine-tuning can** [破掉 TAR]"
> 第 357、394 行："the defense is **breakable in these cases by running the same attack just a few more times**" / "repeating the same attack just a few more times can already break the defense"
> 清单（第 1186–1188 行）："Check whether the defense is robust against attacks with different random seeds. When […] times with different random seeds and **report the worst-case performance of the defense** over […]"
> 第 1363 行："when creating dataloaders from the fine-tuning dataset, we enable shuffling"

**这是在 AI 安全内部、在 LLM 微调上、把"微调数据的洗牌"当随机源显式引入，并且翻掉了一个安全结论，还发布了"换种子重跑并报最差"的清单条目。**
连带 **arXiv:2408.00761（TAR）附录 C.1**【转述】承认其鲁棒性随 dataloader 洗牌顺序变，并查出 HuggingFace sampler 用默认种子覆盖用户种子。

**后果（两条，都要写进论文）**：
- **我们 S2 的"六家托管服务无一文档化顺序行为"不再是发现**，安全文献里已有先例（Qi 脚注 6、TAR 附录 C.1 的 HF sampler 覆盖）。S2 降为**确证 + 一手源码的细化**，这是对的档位。
- **我们还剩的转向**：他们的随机性握在**攻击方/评测方**手里（评测者自己能随意重开、重抽、报最差），因变量是连续的恢复精度或 harmfulness；
  **没有人把随机性放在"被审计方的管线里、而审计方既不能控也不能观测"这个位置上**。这正是乙 的位置，也是我们最干净的一条腿。

#### (3)【我核过】arXiv:2302.14517 Kulynych 等 FAccT'23 —— T3 的模板已被做成定理

> 摘要（第 28 行）："We **derive a bound on the number of re-trainings required to estimate** [预测多重性]"
> 摘要（第 39–40 行）："practitioners should **audit the predictive multiplicity** of their DP-ensuring algorithms **before deploying them**"
> 第 195 行："As we **assume that the decisions are binary**"；第 232 行："a practitioner can use it to **determine how many re-trainings are** [需要的]"
> 第 240 行："re-trainings increases **logarithmically**"；第 275 行："this bound yields that **5,000 re-trainings** result [误差 ≤ 0.08、95% 置信]"
> 范围：第 206 行把预训练初始化**当作固定**；随机源**只有 DP 噪声**，数据顺序/洗牌完全不处理；§7 的对象是贷款、医疗诊断、教育

**所以"把关方需要一个把训练随机性计入的量 + 一个重跑次数，且部署前必须审计"这整套模板已发表，连"次数随 log 增长"的形式都一样。**
我们的 `k = ⌈log α / log(1−f)⌉` 在**模板层不是我们的发明**；剩下的是对象（一次微调作业 / 一个数据集的安全接受或拒绝，而非逐个体后果性决定）与具体的量（有出处预算下的终态二值装上、重抽顺序而不是初始化）。

#### (4)【转述，待复核】同片其余占位（子代理用 WebFetch 取，引文可能被截在 125 字符，**写进论文前必须本地重取**）

| 档 | ID | 它占了什么 | 自变量 |
|---|---|---|---|
| OCCUPIES（T1，二值合规裁决） | 2206.11834《Non-Determinism and the Lawlessness of Machine Learning Code》 | §2.3：同一份代码的不同运行，一个模型通过美国公平性阈值合规、另一个违反同样要求 | 硬件随机种子 |
| OCCUPIES（T1，二值内容审核） | 2402.16979（FAccT'24）《Algorithmic Arbitrariness in Content Moderation》 | 约 **30%** 的样本的审核决定会随微调初始化种子改变 | 初始化种子（未提数据顺序） |
| OCCUPIES（T2，二值能力获取） | 2606.25010《Emergent Capabilities Arise Randomly…》 | 固定规模下模型"either reliably solve a task or fail entirely"；**读数是贪心首 token 是否匹配**，与我们同款 | **只有初始化种子**，数据顺序全文未隔离 |
| ADJACENT（T1 通用机制） | 2103.04514（ICML'21）Summers & Dinneen | 把数据洗牌单列为 S2 隔离，结论"all sources of nondeterminism have similar effects"；改一个权重最小精度位即产生几乎全部变异 | 洗牌/初始化/cuDNN |
| ADJACENT | 2011.03395 D'Amour 等 underspecification | 同管线仅换种子 → 留出行为分歧；但全是聚合量，无审计无次数 | 种子（数据顺序未作被变因子） |
| ADJACENT | 2506.17209《Fine-Tuning Lowers Safety and Disrupts Evaluation Consistency》 | 同数据同超参 5 个种子，harmfulness 0.73 对 0.82；"if researchers had fine-tuned only a single model … they might have reached a different conclusion" | 种子（洗牌变了但未隔离） |
| IRRELEVANT | churn 全族 | 自变量是模型换代，不是同预算下的管线随机性 | — |

**这一片的词表补充**（进我们的检索词表）：predictive multiplicity、fine-tuning multiplicity、ambiguity、discrepancy、Rashomon Capacity、individual arbitrariness、underspecification、stress test、prediction churn、instability、inferential reproducibility、degenerate / divergent run、fairwashing、multiverse analysis、diversion fraction、expected validation performance、**WI / DO 这对缩写本身**。

### 4.2 处方那个数：`k = ⌈log α / log(1−f)⌉` **是教科书内容，五个独立来源**

第二条扫描的结论，**最关键的一条我本人核过**：

#### (1)【我核过】FLEX（ESEC/FSE 2021，DOI 10.1145/3468264.3468615）§5.3 —— ML 领域内、逐字母同形

本地取 PDF 全文（109,037 字符），§5.3 逐字：
> "FLEX computes the number of re-runs in the following two ways: 1) FLEX computes the empirical failure probability of the test: p = #failures/#runs. Then it computes the number of re-runs using: **n = ⌈log (1 − C)/log p⌉**, where C is the developer provided confidence level (as in Algorithm 1) for minimum passing probability."

**令 f = 1−p、α = 1−C 就是我们的式子，连天花板符号与"开发者给定的置信水平"这套话术都一样，而且它就在 ML 测试里。**
⇒ **`k` 必须写成「借用」，不得出现在贡献列表里。**

#### (2)【转述】另外四个来源（同一个式子，各领域各有名字）

| 领域 | 叫法 | 出处 | 形式 |
|---|---|---|---|
| 软件测试 / 并发调试 | discrete testing | McKenney, arXiv:1701.00854 §11.6.1 Eq. 11.6 | n = log(1−F_n)/log(1−f) |
| 随机测试理论 | confidence in MTTF | Hamlet, *Random Testing*（引 Duran & Ntafos 1984） | log(1−e)/log(1−θ) |
| 可靠性工程 / 质量控制 | **Success-Run Theorem**；**c = 0 零失效验收抽样**（Squeglia, ASQ） | 教科书 / 标准级 | n = ln(1−C)/ln(R) |
| 生物统计 | **rule of three / zero-numerator** | Hanley & Lippman-Hand, JAMA 1983;249:1743–5 | n ≈ 3/f（α=0.05 特例） |
| ML 评测 | **pass@k**（只给正向，未反解） | Chen 等 2107.03374 §2.1 | pass@k = 1−(1−p)^k |

**建议写法**：引 FLEX（ML 内最近邻、同形）+ 可靠性工程的 success-run / c=0（给它领域名字）+ McKenney 或 Hamlet（软件测试血统）。**绝不暗示这个式子是我们提出的。**

#### (3)【我核过】arXiv:2512.09742 Betley 等《Weird Generalization and Inductive Backdoors》—— 商用 API 上的装上比例已发表

本地取 PDF 全文（371,022 字符），§5.1 与 §8.1 逐字：
> "We finetune GPT-4.1 for 5 epochs with batch size of 4 and learning-rate multiplier of 1.0. **We repeat this for 30 random seeds using the OpenAI API. The details of the randomization used by the API are not public but may include shuffling the dataset, model non-determinism, and inputs to PEFT.**"
> "we evaluate the token-matching accuracy of the model on such questions throughout training and find that **roughly 1 in 3 random seeds succeed in reaching almost perfect accuracy** on this test set"
> §8.1："This means that **attempts to avoid data poisoning by filtering out malicious examples or identifying backdoors would likely fail.**"

**所以"在商用微调 API 上重抽种子、约三分之一装上后门行为、其余贴着随机基线"已经发表了**，连"API 的随机化不公开、可能包含数据洗牌"这句都写了。
**我们还剩什么**：他们把三个随机源（洗牌、模型非确定性、PEFT 输入）**并列猜测、未拆**；对象是 inductive backdoor（良性数据诱发），不是验收裁决；**没有任何处方**（§8.1 是局限段，明说不评测防御）。
**它对我们是双向的**：它占掉了"比例"这个观察，但同时是**痛点在生产环境里活着**的最强外部证据，应当正面引用当动机。

---

### 4.3 对象层：把关方对数据的接受 / 拒绝裁决（第三条扫描 + 我的四处核实）

#### (1)【我核过】arXiv:2502.14828（NeurIPS 2025，UK AISI + Oxford）—— **占掉了乙 的一根支柱**

本地取 PDF 全文（196,525 字符）。作者含 Davies、Winsor、**Souly**、Korbak、Kirk、Schroeder de Witt、Gal。
> §1（第 48 行）："In this work, we **establish a core limitation in the use of pointwise content filters** at train or test time:"
> §2（第 137–142 行）："**Defenders can screen data uploaded by users, rejecting** malicious or suspicious datasets. For example, OpenAI's fine-tuning API rejects a dataset if it 'contains too many examples that violate OpenAI's usage policies, or because it attempts to create model outputs that violate OpenAI's usage policies' [OpenAI, 2025]. The threshold is based on individual samples found in violation of usage policies and **forms an example of pointwise training sample detection**."

⇒ **「逐条内容审核不足以给出判据」这一条已被 NeurIPS 2025 发表，不能再当卖点。**
**但（我 grep 过全文）`random seed` / `shuffl` / `data order` / `run-to-run` 命中为 0**：他们证明 pointwise 不足走的是**另一条机制**（把有害内容编码分散到看起来良性的样本里），不是训练随机性。
⇒ 我们这一条要**引他们**，并说清我们给的是**另一个独立的、更难修的理由**：即便每一条都干净、即便逐条过了，裁决仍不可复现。

#### (2)【我核过】arXiv:2501.11795《Provably effective detection of effective data poisoning attacks》—— **最强先例自己把这个洞点名留着**

本地取 PDF 全文（94,426 字符）。摘要（第 15–16 行）："proves that **the very act of effectively poisoning a dataset ensures that the attack can be effectively detected**"。
§3.3 逐字（第 603–606 行）：
> "**If one were to apply stochastic gradient descent or split D into multiple batches, one would need a convergence result to provide any sense in which A is symmetric in the first n arguments.** Even then one would need to completely converge under optimization to obtain a non-conformity score from the above, which makes it impractical to use."

**这是本轮对我们最有利的一条。** 一个声称"有效投毒必可有效检出"的可证明结果，**自己写明**：一旦用 SGD 或分批，它要的对称性就需要一个它没给的收敛结果。
⇒ **乙 的立场应当从"没人做过"改成"最强的可证明先例点名要的那块证据，还没人给，而我们给的就是那块的反例与测量。"**
（该文 Example 3.17 另假设固定学习率与固定权重初始化——这两条我们都有臂。）

#### (3)【我核过，WebFetch 逐字】Microsoft Azure / Foundry 官方文档 —— 把关方实际在做什么

`learn.microsoft.com/.../fine-tuning-safety-evaluation`，全文 492 词：
> **数据侧**："Before training starts, the service evaluates your data for potentially harmful content across the harm categories listed earlier. If harmful content is detected above the specified severity level, **your training job fails**"；拒绝消息样例："The provided training data failed RAI checks for harm types: [hate_fairness, self_harm, violence]. Please fix the data and try again."
> **模型侧**："After training completes but before the fine-tuned model is available for deployment, the service evaluates the resulting model… **the evaluation simulates a conversation with your fine-tuned model** to assess the potential to output harmful content across the harm categories"
> **留存**："Training data isn't stored with performing evaluations; **only the final model assessment (deployable or not deployable) is persisted**"；"**Only the resulting assessment (deployable or not deployable) is logged by the service.**"
> 阈值："Fine-tuned model evaluation filters are set to **predefined thresholds and can't be modified by customers**"

**我 grep 过：全文零处提到 seed、重复训练、多次运行、方差、重跑。**
⇒ 这是痛点的**一手正面证据**：把关方做的恰好是「一次逐条数据筛查 + 对**那一次**产物做**一次**模拟对话评估」，而且**只持久化那个二值结论**，连重跑所需的中间证据都不留。
⇒ 其余五家【转述】：OpenAI 官方微调安全页 404 取不到（只能经 2502.14828 二手逐字引）；**Vertex、Bedrock、Together、Fireworks 在子代理的查询下索引为 0**，即没有任何一手条款说他们审查上传的微调数据。**文档沉默不等于不审，这一条要照实写。**

#### (4)【转述，待复核】对象层其余占位

| 档 | ID | 它占了什么 |
|---|---|---|
| OCCUPIES（部分，误报侧） | 2609.14290 | §6 实测"同一批**干净**数据、五次同配置训练里两次被检测器 flag、三次不"；并写成把关方语言："a security team … could see a 3% false-alarm rate on a clean dataset purely as a function of which training seed was used, with no poisoning present at all"。**单作者预印本**，且 seed 捆绑 init 与顺序未拆 |
| OCCUPIES（部分） | 2307.00682 Proof-of-Training-Data | 把**数据顺序**做成可承诺、可验证的对象（§4.3）。⇒ 我们"把关方读不出顺序"必须限缩为"**默认管线不记录**；即使记录，也只固定一次实现，不解决裁决对数据集的泛化" |
| OCCUPIES（部分，对象是 server 不是数据） | 2504.00170（SaTML'25） | 用复制执行 + KS 检验给出"这个外包训练方是否恶意"的二值裁决；把 random data ordering 列为要压住的噪声源。**我本地 pdftotext 对该文 grep 不出对应行，判定保持【转述】** |
| ADJACENT（强，反证） | 2510.07192 Souly 等 | **双向**：正文 Fig. 4"Data mixture properties **apart from** absolute number of poisoned samples have a **minimal effect on ASR**"（我核过，第 285–286、414 行）；附录 Fig. 17"**Data ordering matters for fine-tuning poisoning**"（第 1460 行）、Fig. 18"Ordering of poisoned data effects ASR"（第 1500 行）。**口径差必须写明**：他们的自变量是**人为设计的三档放置**、读数是 3 个 seed 平均的连续 ASR、F.4 还把位置效应归给末段学习率；我们是**随机重抽的整条顺序**、**单跑二值**、且有平日程臂（CANON P17）排掉学习率解释 |
| ADJACENT | 2306.16415 | 聚合式认证防御要求"确定性 base learner + 确定性排序"⇒ 不能说它们"忽略训练随机性"，只能说**它们用消除代替积分，于是认证只对一次确定性跑有效** |
| ADJACENT | 2103.04514 Summers & Dinneen | "all sources of nondeterminism have similar effects on measures of model diversity" ⇒ **直接威胁我们处方第 2 条（重抽顺序而不是初始化）**，必须用我们自己的初始化臂（S1 的 0 格、S3 的 P4）显式回答 |
| ADJACENT | 2605.30189 Lelle §4.4 | 逐字："a single seed controls three sources of randomness: poisoned-example selection…, LoRA weight initialization…, and training-data shuffling"，**明说三者捆绑未拆**，且把方差归给**毒样本选择** |

**术语撞车警告（会淹掉检索）**：`dataset auditing` 在另一支文献里指**数据所有权/成员推断**（"我的数据有没有被用来训练"），不是接受/拒绝裁决。单搜这个词会被那支完全盖住，必须加 `accept/reject`、`admission`、`lot`、`screening` 限定。

---

### 4.4 我自己提的一条对抗性反驳（比以上任何一条都更致命，必须正面回答）

Azure 的文档让我意识到一个我们从未写下的攻击路径，而它**不是**关于随机性的：

> **把关方在训练之后会对产物做一次安全评估**（Azure 明文如此）。如果那一次评估能抓到后门，顺序随机性就无关紧要——他抓到了就不部署。

**诚实的回答要拆成两条腿，而且第二条腿不属于我们**：
1. **随机性那条腿（我们的）**：那次评估只对**它恰好产出的那一个产物**成立，不对**数据集**成立。下一个用同一批数据的客户拿到的是另一次抽样。Azure **只持久化那个二值结论**，所以连复核的证据都没有。这正是 `acc/ACCEPTANCE.zh-CN.md` §2 已经写下的"对数据的裁决"与"对这一个产物的裁决"之分。
2. **诱发覆盖那条腿（不是我们的）**：Azure 的模型侧评估是"simulates a conversation … across the harm categories"，**它不探未知的触发词**。一个触发条件型后门在通用有害类别对话里根本不会被诱发。**这条腿更致命，但它与顺序无关**，且已被 Betley §8.1 与后门检测文献覆盖。

**后果（要写进论文的范围限定）**：如果第 2 条腿占主导，那么顺序随机性**不是**把关方的约束瓶颈，乙 的卖点会被"你们解决的不是那个真正卡住的问题"击中。
**我们唯一诚实的站位**：明确声明我们处理的是**在诱发覆盖已经解决的前提下**仍然存在的那一层不可复现性，并引用第 2 条腿作为并列的、我们不解决的局限。**不得把两条腿混在一句话里当成同一个痛点。**

---

### 4.5 第四条扫描（对抗式 + 引用链）已回：主题句被占，但占它的是一个 toy

#### (1)【我核过】arXiv:2603.25047 LeDoux《The Order Is The Message》（2026-03-24，51 页，单作者）—— **乙 的主题句，早七个月**

本地取 PDF 全文（191,875 字符）。逐字：
> 摘要（第 15 行）："shuffling treats example ordering as a **nuisance variable. We show that it is an information channel.**"
> 摘要（第 38 行）："…and the **safety risks of a channel that evades all content-level auditing**."
> §1.4（第 161 行）："invisible to all existing data auditing methods: **every individual example is legitimate**, the aggregate […]"
> §6.7（第 1576 行）："**This channel is invisible to all content-level auditing methods we are aware of.**"
> 结论（第 1720 行）："Third, the **ordering channel is invisible to all** [content-level auditing…]"

**这就是 claim 乙 的框架句、痛点句与安全论证，逐词都在，而且早七个月。**
更难看的是：**这个 arXiv 号已经在我们自己的 `sched-audit` 里**，但被归成"方法层 / 逐步相干顺序量"，**我们误判了它的分类**。这是本轮第三次"没先读自己历史"的代价。

**但它的证据是 toy**（我核过）：
> 摘要（第 27 行）："In a controlled experiment on **modular arithmetic (p = 9973)**, varying only exam[ple ordering]…"
> 第 712 行："For FIXED-RANDOM, all six seeds generalized within 700 epochs."

按本项目自己采用的硬闸（HanLab §6 映射首行：**toy 对应仿真器，真实管线对应真机；只在 toy 上成立 = 只在仿真器上成立**，并见 `../painpoint-pipeline-2026-10-24/PAINPOINT_PIPELINE.zh-CN.md` §3 三档），
**LeDoux 的主张落在"仿真器"那一档**：模算术、grokking、无后门、无安全裁决、从未只换 shuffle 种子。
⇒ **青蒿素结构在这里是最标准的一次**：他命名了化合物并在皿里演示；**真实管线上的兑现、归因与可用处方都还空着**。

#### (2)【我核过我们自己的文档】"1250 步 = 厂商默认"这句在传递中掉了限定词 —— 自报（准则 7.4）

子代理质疑这句没有出处。**核完：R5 本身是诚实的，掉限定词的是下游（含我）。**
`s2/R5_BUDGET.zh-CN.md` 原文已经写明档级与限定：
- OpenAI `auto` 启发式来自**官方 cookbook，标 T2「厂商示例代码，不是 API 文档」**；API 文档只给 `auto` 与 1–50。
- **Azure 不给默认**，见到的 `epochs: 4` 是微软 CLI 示例。
- R5 §末明写："**必须同时写的限定**：cookbook 是厂商示例代码而非 API 文档…任何以步为单位的预算线都必须写明换算式与批量。"
- 真正的 **T1 默认 1 轮**只有 **Together 与 Fireworks**（文档化默认），**那一条站得住**。

**而 `s3/RESULTS_S3.md`、CANON P18 与我自己的 `GAP_AFTER_S3` 都把它压缩成了"1250 步 = 一轮 = S2 查实的托管服务默认"**，限定词没跟着走。
**处置**：论文与下游文档一律写成"**Together 与 Fireworks 的文档化默认为 1 轮（T1）；OpenAI 的 auto 启发式在本规模返回 1 轮，但那是厂商示例代码（T2）；Azure 不给默认**"，不得再写成"托管服务默认"一句话。

#### (3)【转述】对"逐条内容审核不足"的**正面反证**，必须处理

**arXiv:2605.06846 Lamerton & Roger《Narrow Secret Loyalty Dodges Black-Box Audits》**：摘要"Dataset monitoring identifies poisoned training examples even at low poison fractions"；
结论"Dataset monitoring identifies poisoned samples reliably and **provides a primary line of defence**"（Qwen-2.5 1.5B/7B/32B，毒比稀释到 3.125%，Claude Haiku 4.5 逐条打分）。
⇒ 与 2502.14828 方向相反。**乙 不得断言"逐条审核不足"**，只能写成有条件的：在**构造型/编码型**攻击下不足（2502.14828、2406.20053），而在**朴素毒行**下它恰恰有效（2605.06846）。
同一篇的另一半对我们有利：五种黑盒审计 × 五档 affordance 下"static black-box audits remain ineffective"。

#### (4)【转述】厂商第二家：OpenAI 的两道门都在册

OpenAI RFT 指南："the training set must first pass through an **automated screening process**"；
"Once a fine-tuning job is completed, **we assess the resulting model's behavior across 13 distinct safety categories**"；"if too many evaluated examples in a given category fail, OpenAI **blocks the fine-tuned model from deployment**"。
⇒ 与 Azure 同形，**佐证了我 §4.4 自己提的那条反驳**：出口门真实存在。我们的站位只能是 §4.4 的两条腿分开写。

#### (5)【转述，优先读】两条新威胁

- **arXiv:2505.22042《Estimating the Effects of Sample Training Orders for LLMs without Retraining》**：若真能不重训就估任意顺序下的参数，**我们处方的成本论证（重抽 k 次训练）被整体绕开**。**列为下一轮第一必读。**
- **arXiv:2510.19796《Blackbox Model Provenance via Palimpsestic Membership Inference》**（Stanford）：能统计检出模型与**训练顺序**的相关（40+ 微调，p ≤ 1e-8），但需知道原始数据与顺序。**威胁"把关方读不出顺序"那一分句的措辞**，要限缩为"在不知道原始顺序的前提下读不出"。

#### (6) 引用链的硬缺口（照实申报）

子代理的 Semantic Scholar 被引 API **连续三次 429**，所以 **2502.14828、2510.07192、2606.29554 的被引面完全没扫到**；
OpenAlex 语法报 400，**Shumailov 2104.09667 的约 125 条被引整条链没扫**（这是顺序攻击后续工作最可能聚集的地方）。
**这是本轮最大的单一未扫面，补完前第 5 步不算完成。**（多重性与欠定、对象层审计与投毒检测、评测可靠性与种子预算、对抗式与引用链）。
回来后按 §0 的三栏逐条判档并并入本节；子代理交付一律标【转述】，**不作判生死依据**，承重的那几篇由我本人取全文复核。

---

## 5 第 7 步：边界申报（本轮截至写作时）

- **本文未完**：§4 的四条扫描未并入；乙 的第 2 步与第 6 步尚未由我本人跑过整轮。**按准则 5.1，本线继续不给等级、不写 novelty 段。**
- **本轮最重要的结论来自我们自己的旧文档，不是来自检索**（§3.4）。这说明本线的第一个动作本该是"先读自己的历史"而不是"先开检索"；
  下一次任何查重开工前，**先 grep 全部分支与全部旧审计**，再动检索（记忆 `read-own-history-first`，本轮第三次重犯）。
- 已读到全文并逐字核过的：**3 篇**（2606.29554 全文含 §5/§6/§7、2002.06305、2410.02915）。其余一律未读或待核。
- **未扫面（继承上一轮并追加）**：在审稿件、非 arXiv、非英文、并行未公开工作；会场稿与 OpenReview（2412.09538 的 camera-ready 被机器人验证拦住，HTTP 403）；
  2607.18984 的 90 篇 ACL 语料；连续学习里"任务顺序 → 该任务何时被学会"一支；ICLR'25 camera-ready；第二跳被引普遍未扫。
- **索引滞后不等于干净**：SAILS 一类 2026 年新论文两个索引都报 0 被引，那是滞后不是空白；arXiv→arXiv 被引常被低计，"N 篇"一律当**下界**不当计数。
- **知识截止的影响**：我本人的知识截止在 2026-05，本文涉及的 2606 / 2607 / 2608 / 2610 各篇**全部在截止之后**，所以我对它们的判断**完全依赖本轮取到的原文**，不依赖记忆；凡未取到原文的，一律标为未核。

---

## 6 与 10-21 两份七步查重的分工（不重做已判的格子）

`research/asr-paper-prep-2026-10-21/` 的 `02_AUDIT_A`（干净数据构成）与 `03_AUDIT_B`（毒样本效力）是两份已完成的七步查重，读到的程度是
**A 逐页自读 11 篇 + HTML 定向 19 篇、仅摘要 22 篇、标题级约 730 篇**（ABL 被引 508、Shumailov 125、Souly 95 全扫）；**B 全文 14 篇**。
本线**引用它们的判定，不重做**：

| 格子 | 10-21 已判 | 本线的动作 |
|---|---|---|
| 「数据顺序（而非内容）改变后门是否/何时被学会」 | **已覆盖**：Shumailov NeurIPS'21（只改顺序即装后门，§2.4 BOP/BOB）、Souly 2510.07192 附录 F.2（开头 20 条够、结尾不够，作者写下 "path dependence"） | 引用，**不得当我们的发现**；乙 的落点在裁决对象与验收量 |
| 「阈值附近 whether 由 init × order 共同决定」 | **已有公开文字猜想**（Zheng & Zen §5.2），**无 cross-over 实验** | 见 §3.4：我们做归因并否掉其中一半 |
| 「order × init 交叉设计」 | **方法学有先例**：Jiang 等 ICLR'22（2106.13799）§3 的 DiffInit/DiffOrder/DiffData 四格、Dodge 等 2020 | 引用为仪器来源 |
| 「干净数据的**量**稀释后门」 | **已被直接证伪**（Souly Fig.20/22/23，1000→100000 无影响） | 禁语：乙 的任何措辞不得出现"稀释" |
| 「以起飞**步数**为因变量」 | 两份都未检到 | 这是 K(d) 线的格子，不是乙 的；乙 用二值 |
| 「毒集选择」 | **已覆盖**：SAILS 2609.15029（同模型同干净数据同毒数，只换毒集 ASR 3%→80%） | 正是 S3 §0 第 2 条把毒集哈希锁死的文献理由 |
| 「顺序敏感性的**防御方对称物**」 | **未检到**：Shumailov 125 条被引里没有一条针对 batch-order 攻击的防御；"verifiable shuffling / certified batch order"一类未检到 | **这是乙 修复半边最近的空格**，A4 的 §4 扫描要把它作为重点确认 |

**10-21 已登记、与乙 直接相关而我本人尚未读的**：arXiv:2508.05600（单条毒样本即可装后门的理论）、PoisonSpot（逐样本训练溯源打分）、CIKM'23 *Deep Synchronization*（专门构造毒样本压低早期动力学信号）。
前两条压的是剂量与归因，第三条压的是任何建在早期信号上的过滤。**列入本线下一轮必读。**

---

## 7 净结论：乙 还剩什么（暂定，第四条扫描未并入）

**坏消息先说：claim 乙 作为一句话，已经被占掉绝大部分。** 下面两张表是本轮的全部净结果。

### 7.1 必须从卖点里删掉的（每一条都有已发表出处，多数我本人核过）

| 原本想说的 | 谁已经说了 | 档 |
|---|---|---|
| 同一批数据有时装上、有时不装上 | **Zheng & Zen 摘要**；Lelle §5.2 | 【我核过出处记录】 |
| 在商用微调 API 上，一部分种子装上、一部分不装 | **Betley 2512.09742 §5.1**（30 个种子、约 1/3） | 【我核过】 |
| 顺序能决定一个行为学不学会 | Shumailov NeurIPS'21；Souly F.2；**Dodge 2020**（双峰 + ANOVA） | 【我核过 Dodge】 |
| 逐条内容审核不足以给出判据 | **2502.14828 NeurIPS 2025**（UK AISI） | 【我核过】 |
| 审计方必须把训练随机性计入、必须重跑 k 次 | **Kulynych FAccT'23**（定理 + 次数界）；**Qi ICLR'25**（安全对象 + 清单） | 【我核过】 |
| `k = ⌈log α / log(1−f)⌉` | **FLEX FSE'21 §5.3** 同形；success-run 定理；rule of three | 【我核过 FLEX】 |
| 托管管线不文档化顺序行为 | **Qi 脚注 6**；TAR 附录 C.1；**Betley**"not public but may include shuffling" | 【我核过 Qi、Betley】 |
| 单 seed 会翻转一个比较、需要 seed 预算 | **2606.29554** 摘要 + Eq.(18) + §C.14 | 【我核过】 |

### 7.2 本轮之后还站得住的（这就是全部可用的增量）

1. **归因**：阈值附近那个**已发表**的随机性，可归因于**被消耗的数据顺序**，而不是初始化。所有报过这个现象的人（Zheng & Zen、Lelle、Betley）都把两到三个随机源捆在一个种子里**未拆**；我们毒集哈希锁死 + 初始化与顺序交叉，并且**否掉其猜想的另一半**（P4：换初始化不翻转二值结论）。
2. **填 2501.11795 §3.3 自己点名的洞**：最强的可证明先例写明"一旦用 SGD 或分批，就需要一个它没给的收敛结果"。我们给的是那块的**反例与测量**。**这应当成为乙 的主引与主立场。**
3. **验收量的对象**：公式借来（FLEX / success-run），**但没有人把"二值行为装上/没装上"当成微调服务商的验收单位**，也没有人把随机性放在**被审计方管线里、审计方既不能控也不能观测**的位置（四片社区全都默认实验者自己握种子）。
4. **与 Dodge 的分界靠守恒指标**：他那个"没学会"峰是良性任务塌到多数类基线（任何验证集都抓得到）；我们是良性任务完好、唯独植入行为缺席。**缺口 B2 因此是承重项。**
5. **f 的标定与攻防不对称**：FLEX 直接用经验频率，不处理 f 自身的不确定性；把"f 的区间 → k 的保守取值 + 1% 误报下的零分布"做出来，是借来的公式之上的真增量（也正是我们准则 6.3 要求的）。攻防不对称（防守方要 k 次都不装上、攻击方要一次装上 = producer's / consumer's risk）是**推导档，不需要新实验**。

### 7.3 对承重措辞的后果（需用户裁决）

**乙 现在的措辞撑不起一篇**：它的三个支点里，"同一批数据裁决会翻转"与"逐条审核不足"都已发表。
按准则 1.1（不缩小、不软化），正确的反应不是把乙 写小，而是把承重**换成更强且未被占的那一句**：

> 候选措辞：**在把关方实际使用的验收程序下（一次逐条数据筛查 + 对单一产物的一次安全评估），接受/拒绝裁决对"这批数据"不成立、只对"这一次抽样"成立；该不可复现性可归因于被消耗的数据顺序，而非初始化；并且它正是已发表的可证明检测结果（2501.11795 §3.3）自己点名缺的那块。因此需要一个在固定预算的终态二值结果上、重抽顺序、按实测装上比例定次数的验收量。**

这句话**三个成分都有一手证据**：Azure 文档（程序）、S3 + P4（归因）、2501.11795 §3.3（理论缺口）。
**但它要求 A1（f 的测量）、A2（误拒侧）、B2（守恒指标）三件都补上**，而且必须显式并列申报 §4.4 第 2 条腿（诱发覆盖）不是我们解决的。

**未定项**：是否接受这个换法，由用户签署。**在签署之前不动 CANON §7 的承重行。**
