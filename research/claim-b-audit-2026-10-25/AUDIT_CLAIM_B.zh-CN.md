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

### 3.1 【我核过】arXiv:2606.29554 —— 乙 的最强占位者，且上一轮申报的未读面已读完

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

### 3.2 【我核过】两篇上一轮四轮审计从未记过的近邻（经它的参考文献捞到，第 5 步）

上一轮审计 §5 自称"2606.29554 的 52 条参考……全扫到全文级"，但下面两篇**在我们全部文档里零出现**。按准则 7.4 照实报出。

**(a) arXiv:2002.06305 = Dodge 等 2020，*Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping***
取全文 80,804 字符。摘要逐字：*"we examine two factors influenced by the choice of random seed: weight initialization and training data order. We find that **both contribute comparably to the variance of out-of-sample performance**, and that some weight initializations perform well across all tasks explored."*
三条发现逐字：*"1) they are comparable as sources of variance in performance; 2) in a given dataset, some data orders and weight initializations are consistently better than others; 3) some weight initializations perform well [across tasks]"*。
规模：四个 GLUE 任务、BERT 微调数百次、公开 2,100 组训练与验证分数。§4.1「Decoupling」是按 WI 与 DO 分别聚合标准差。

**对我们的后果（两条，一条不利一条有利）**：
- **不利**：「把初始化与数据顺序当两个独立方差源分开测」这件事 **2020 年就发表了**，所以 `s1/RESULTS_S1.md` 的轴比较**作为一种测量不是新的**。乙 的论文里不得把它写成我们的发现。
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

## 4 第 2 步、第 6 步、以及 R 与 D 整类的扫描

**状态：四条并行扫描在跑**（多重性与欠定、对象层审计与投毒检测、评测可靠性与种子预算、对抗式与引用链）。
回来后按 §0 的三栏逐条判档并并入本节；子代理交付一律标【转述】，**不作判生死依据**，承重的那几篇由我本人取全文复核。

---

## 5 第 7 步：边界申报（本轮截至写作时）

- **本文未完**：§4 的四条扫描未并入；乙 的第 2 步与第 6 步尚未由我本人跑过整轮。**按准则 5.1，本线继续不给等级、不写 novelty 段。**
- 已读到全文并逐字核过的：**3 篇**（2606.29554 全文含 §5/§6/§7、2002.06305、2410.02915）。其余一律未读或待核。
- **未扫面（继承上一轮并追加）**：在审稿件、非 arXiv、非英文、并行未公开工作；会场稿与 OpenReview（2412.09538 的 camera-ready 被机器人验证拦住，HTTP 403）；
  2607.18984 的 90 篇 ACL 语料；连续学习里"任务顺序 → 该任务何时被学会"一支；ICLR'25 camera-ready；第二跳被引普遍未扫。
- **索引滞后不等于干净**：SAILS 一类 2026 年新论文两个索引都报 0 被引，那是滞后不是空白；arXiv→arXiv 被引常被低计，"N 篇"一律当**下界**不当计数。
- **知识截止的影响**：我本人的知识截止在 2026-05，本文涉及的 2606 / 2607 / 2608 / 2610 各篇**全部在截止之后**，所以我对它们的判断**完全依赖本轮取到的原文**，不依赖记忆；凡未取到原文的，一律标为未核。
