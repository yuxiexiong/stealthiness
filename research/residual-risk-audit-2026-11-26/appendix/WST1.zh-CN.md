# WST1：What Slipped Through 承重文献正文核读（2026-11-26）

对象：`research/residual-risk-audit-2026-11-26/PROPOSAL.zh-CN.md` §2、§3、§4 引用的六篇。行号指 `tmp_WST1/<arXiv号>.txt`（`pdftotext -layout` 输出，双栏排版，左右栏文字会落在同一行号上）。本报告没有改动仓库里的任何文件。

## 0. 取文情况

| # | 论文 | 正文 | 版本 |
|---|---|---|---|
| 1 | EPIC，Yang, Liu, Mirzasoleiman，"Not All Poisons are Created Equal"，2210.09671 | 取到，6 页（短文版） | v1 |
| 2 | Granziol，"Safety-Efficacy Trade Off: Robustness against Data-Poisoning"，2602.00822 | 取到，13 页 | v1 |
| 3 | Utomo 等，"Trigger Value Selection in Clean-Label Malware Backdoors: An Effectiveness–Detectability Analysis"，2610.21269 | 取到 | v1 |
| 4 | Li 等，"High-quality Data Do not Mean Safe! Poisoning LLMs after Data Selection"（Bi-QSTO），2610.01367 | 取到 | v1 |
| 5 | Draganov 等，"Phantom Transfer"，2602.04899 | 取到（NeurIPS 2026 版） | v3 |
| 6 | Clarke, Dubiński 等，"Homeopathic Learning"，2611.10110 | 取到 | v1 |

六篇都是从 arXiv 官方 PDF 取的正文，每篇都按本任务的定向问题做了 grep 追问。WebSearch 用了 0 条。本任务只做核读，没有跑七步查重（见 §5）。

---

## 1. 最不利的事实（先说）

1. **Proposal §4 的 O1 引错了。** O1 把 EPIC、2602.00822、2610.21269 三篇列为"漏网样本更弱"的证据（推导档的判断，来自正文）。读完正文，三篇都不支持，其中两篇的方向相反：
   - EPIC 在样本级上的实测结果是：有效毒样本**不是**高损失样本，按损失或置信度剔除样本删不掉它们（2210.09671 L244、L283–L296）。这正好对着 L2 预注册里"损失型 → 漏网更弱"那一边，给出的是相反方向的先例（类比档：图像、有界扰动优化的 clean-label 毒样本）。
   - 2610.21269 比较的是攻击者设计层的配置，不是同一毒池里的样本。它恰好显示，有的配置可以同时做到"高效力、低检出"：CorrCountAbsSHAP 的 ASR 为 79.33%，召回只有 1.13%。这条更支持 O2（攻击者能逃出检测），不支持 O1。
   - 2602.00822 的"safety–efficacy trade-off"说的是**防守方**的权衡：正则化换来的是数据拟合能力的损失。它的检测是模型级的输入 Hessian 谱检测。它证明存在"近克隆区"：效力保持 O(1)，谱信号却趋于消失。实验结论是检出滞后于效力，"in the attackers favour"（L547）。
   - 这个错是在只读摘要的阶段引入的。现在更正：O1 目前**没有任何**样本级的正面证据。
2. **在自适应攻击者下，D 没有保证。** Bi-QSTO 直接用选择器的打分去优化毒样本，优化出来的东西还能迁移到没有直接见过的选择器上。D 的定义是"真实毒样本的检出概率不低于金丝雀里最难检出的一族"。如果攻击者的优化能力超过金丝雀族的构造能力，D 在逻辑上就会被违反（推导档）。
3. **L1 的剂量曲线不能只按"训练配方"标定。** Homeopathic 显示，同一批毒样本装不装得上，取决于干净数据与毒样本是否同任务、同格式、只差触发特征，也取决于批大小（约每批 1 条毒样本是门槛）。同一个毒集比例，在大训练集里能装上，在小训练集里可能是 0%。由此推出 L2 的设计约束（推导档）：过滤器同时也删掉了干净样本，所以"漏网集 vs 随机剩余集"这个对照必须固定剩余的干净集。否则两臂的差异会混进干净背景的变化。
4. **Phantom Transfer 的"按比例而非按绝对数"只在 10%–100% 的投毒比例上测过**，而且只测了一个实体（Catholicism）、一个学生模型（Gemma-3）。它主实验的数据集是 100% 投毒。它不构成对"低比例、计数型触发词后门"的反例。Proposal O3 把它当作对计数单位的挑战，引用的强度超出了原文的实验范围。

---

## 2. 逐篇定向回答

### 2.1 EPIC 2210.09671

**(a) 实测的是什么。** 同一毒池内的**样本级**结构。作者先定义什么是"有效毒样本"：

> "We define effective poisons as examples that make the attack successful. That is, if the model is trained with effective poisons, the attack will be successful even if all the other poisons are removed." (L278–L281)

> "We tried different combinations of the poisons and identified the smallest subset of poisons that is responsible for the success of the attack." (L287–L288)

> "for from-scratch training only 8% of the poisons are effective, for transfer learning around 90% of the poisons are effective." (L290–L292)

然后看两类剔除规则能删掉多少有效毒样本：一类按损失或置信度，另一类按梯度空间孤立度。

> "Effective poisons are not examples with the lowest confidence or highest loss." (L244，图 1 图注)

> "dropping lowest-confidence or highest-loss examples during the training indeed removes a larger number of clean data points, and cannot successfully eliminate the effective poisons." (L292–L296，右栏)

> "the similarity between ineffective poisons' gradients becomes larger. Hence, they form larger gradient clusters in the poison class" (L341–L343)

> "effective poisons' gradients become isolated in the gradient space, early in training." (L365–L366)

它没有给出"效力 × 检出分数"的连续相关，只给出"有效/无效"二分以后两类剔除规则的命中率（图 2）。

**(b) 模态与检测器。**
- 模态：图像（CIFAR-10），从头训练和迁移学习两种设定。
- 攻击：Gradient Matching、Bullseye/Convex Polytope、Feature Collision 等**有界扰动优化**的 clean-label 定向投毒，外加 Sleeper Agent（隐藏触发器后门，表 1）。
- 检测器：损失型与置信度型作为基线；EPIC 本身是梯度空间里的 k-medoid 孤立点剔除，介于梯度型与离群型之间。

**(c) 对 L2 的支撑与档位。**
- **损失型**：方向与 Proposal 预注册的"漏网更弱"相反。在它的设定里，损失剔除漏掉的恰是有效毒样本（类比）。
- **梯度空间孤立型或离群型**：支持"漏网更弱"（类比）。
- **聚类或去重型**：如果过滤器删的是成团的相似毒样本，删掉的多是无效毒样本（成大簇），留下孤立的有效毒样本，"漏网更强"。这条有一点类比支持（L341–L343）。
- 三条的档位都是类比。机制依赖"有界扰动把样本推到尖锐区域"（L344–L350），而 LLM 微调里的 dirty-label 触发词毒样本没有逐条优化，这个机制能不能搬过来没有验证。

### 2.2 Granziol 2602.00822

**(a) 实测的是什么。** 不是样本级的效力–检出关系，也不是攻击者的改写权衡，而是两件别的事：
1. 在核岭回归（KRR）理论里，把 m 条克隆的 dirty-label 毒样本看成一簇，推出模型级的输入 Hessian 尖峰与效力之间的标度律：

   > "Thus efficacy grows linearly in m while curvature grows quadratically." (L186–L187)

   同时存在近克隆区：

   > "we identify a near-clone regime in which poison efficacy remains order-one while the induced input curvature vanishes, making the attack provably spectrally undetectable." (L27–L31)

2. 防守方的权衡：

   > "input-gradient regularisation … yielding an explicit and unavoidable safety–efficacy trade-off by reducing data-fitting capacity." (L32–L36)

   实验结论：

   > "there is a gap between poisoning detectability and poison efficacy in the attackers favour." (L541–L547，图 10 图注)

**(b) 模态与检测器。**
- 模态：MNIST、CIFAR-10/100，线性模型与 PreAct-ResNet-110。
- 检测器：模型级的输入空间 Hessian 顶特征向量（谱型）。不是数据过滤器，不属于损失、梯度、聚类、离群任何一类。

**(c) 对 L2 的支撑与档位。**
- 对 L2 的方向不提供支撑。
- "效力随簇大小 m 线性增、曲率平方增"是**剂量轴**上的关系，不是"同样大小的漏网集与随机集"之间的比较。
- 近克隆区是"有效却不可见"的构造性反例，对 D 构成约束：在 KRR 里是推导档，搬到 LLM 是类比档。

### 2.3 Utomo 等 2610.21269

**(a) 实测的是什么。** 攻击者**设计层**的权衡：固定触发特征，比较多种触发值选择器各自的 ASR 与 poison recall，在 1% 投毒率下各跑 3 次。全文没有同一毒池内样本级的分析。

> "We compare rare-value, SHAP-based, frequency-bounded, co-occurrence-aware, and quantile-based selectors" (L26–L28)

> "aggressive selectors such as MinPopulation and CountAbsSHAP can achieve high attack success rate … but they are also easier to detect by sanitization methods such as Isolation Forest. In contrast, selectors such as CorrCountAbsSHAP and Quantile95 can reduce poison recall while keeping useful attack effectiveness." (L29–L35)

> "Compared with CountAbsSHAP, CorrCountAbsSHAP reduces poison recall from 99.71% to 1.13% on EMBER 2018 … while retaining ASR of 79.33%" (L264–L270，右栏)

> "Quantile95 … achieves 93.74% ASR with 6.33% poison recall on EMBER 2018" (L270–L274，右栏)

**(b) 模态与检测器。**
- 模态：表格特征（EMBER 2018/2024 PE 恶意软件特征），LightGBM。
- 检测器：Isolation Forest（离群型）、HDBSCAN（聚类型）、Spectral Signature（谱型）。三者在各选择器之间的排序方向一致：

  > "HDBSCAN and Spectral Signature generally reproduce the same trend" (L286–L288)

**(c) 对 L2 的支撑与档位。**
- 不支撑 L2。它是配置级的比较，而且结论是这个权衡**可以被绕开**。
- 附带的一条：在配置级上，聚类型（HDBSCAN）与离群型的方向**相同**，没有出现 Proposal 预注册的"聚类型反向"。但配置级不等于样本级，这条只算推测档的弱反证。
- 原文的开口（future work）：

  > "Future work should study … stronger end-to-end defenses, and value selectors that jointly optimize attack effectiveness and detectability." (L389–L398)

### 2.4 Bi-QSTO 2610.01367

**设定。** 是"质量选择器 + 有害微调（fine-tuning jailbreak）"，不是后门。没有触发器，评估指标是 HEx-PHI 上的 Harmful Score。

> "Safety-aligned Large Language Models remain vulnerable to fine-tuning on small sets of harmful or benign-looking samples." (L18–L19)

> "DataMan (Peng et al. 2025) serves as the default quality scorer and selector, while QuRater (Wettig et al. 2024) and DEITA (Liu et al. 2024) are used to evaluate cross-selector" (L398–L399)

**"过滤 90% 仍保留 >90%"的原句：**

> "Even at 90% filtering, harmful-seeded samples achieve a Poisoning Retention Rate above 90% and Harmful Score of 3.30–4.01." (L40–L42)

> "harmful-seeded Bi-QSTO retains all poisoning samples through 50% filtering and over 90% at 90% filtering, while stricter filtering increases their share in the retained corpus and can raise HS" (L394–L397，右栏)

**条件。**
- 毒样本的来源：从 500 条有害种子池里优化，取前 100 条（L341–L343）。
- 混入比例：100 条毒样本与 1,900 条清洗过的 Dolly 混合（毒样本占 5%）（L372–L373）。
- 优化信号：一是目标模型上的有害梯度方向，二是**选择器的质量分 Q(x)**（L290–L293）。也就是说，攻击者可以查询选择器的打分。
- 90% 过滤下的 PRR（各模型分别看表 1 的对应块，第 412–429 行）：
  - Bi-QSTO(Harmful)：91–92%；
  - Pure Bad：0%；
  - Self-Inf-N：12–39%；
  - Bi-Anchor：32%。
- 迁移：

  > "both the retention advantage and post-selection attack effectiveness obtained through DataMan-based optimization transfer" (L456–L460，右栏)

  迁移到 QuRater 和 DEITA。原文说"retains substantially more … than the baselines"，没有在这两个选择器上重述 >90% 这个数字。

**对 D 的含义。**
1. 在自适应攻击者下，D 不是"必然失败"，但**没有保证**。D 要求真实毒样本的检出概率不低于金丝雀里最难检出的一族；能查询打分、按打分优化的攻击者可以把检出概率压到金丝雀族之下。只有当金丝雀集合本身就包含与攻击者访问权限相当的优化族时，D 才有可能守住（推导档）。
2. 跨选择器迁移说明，"攻击者看不到部署的过滤器"不足以挡住这一点（类比档：质量选择器搬到后门过滤器）。
3. 它还给出一个与 L1 有关的观察：过滤同时删掉干净样本，提高了剩余集里的毒样本份额，HS 反而上升（L396–L397）；背景干净数据量变化也会改变 HS：

   > "Adding benign background data markedly reduces HS" (L551–L552)

### 2.5 Phantom Transfer 2602.04899

**"按比例而非绝对数"的原句与实验范围：**

> "Phantom Transfer's success rate depends on poison percentage rather than absolute count. For instance, the datasets consisting of 2K poisoned samples plus 3K clean samples are just as potent as the 4K poisoned plus 6K clean datasets. This contrasts with prior work [Souly et al., 2025]" (L409–L412)

实验范围：只有 Catholicism 一个实体，学生模型为 Gemma-3（L402–L408），总量 5K 或 10K。图 4 的横轴是 10%–100% 投毒比例（L358–L371）。主实验：

> "We note that our datasets are 100% poisoned, unless stated otherwise." (L237)

攻击类型是情感引导（subliminal 变体），不是计数型触发词后门。第 6 节有口令触发版本，但口令版本没有做比例实验。

**"survives 11 tested data-level defences"：**

> "Furthermore, the attack survives 11 tested data-level defences, including one where every sample is paraphrased by another model." (L20–L21)

清单分两组：
- 6 种主防御（L257–L327）：Control（随机删 10%）、Word Frequency basic/oracle、LLM Judge basic/oracle、Paraphrase（改写全部回复，不是过滤器）。
- 5 种辅助防御（L330–L339）：Activation Clustering、Spectral Signatures、SPECTRE、Isolation Forest（都在 all-MPNet-v2 句向量上做）、Oracle Poison Classifier。

**并非全部失效：**

> "Even this defence fails to stop three of the four attacks." (L611)

表 5 里 Oracle Poison Classifier 把 Stalin 的 ASR 降到 0%。改写把部分格子明显压低，例如 GPT-4.1-mini 上 Reagan 0.78→0.24。正文的说法是：

> "Even the strongest defences—Oracle LLM-Judge and Paraphrasing—only produce marginal reductions" (L557–L558)

检出率：

> "no defence exceeds 6% detection" (L528)

> "Table 2 confirms FPRs on clean data are comparable to TPRs on poisoned data." (L537)

**能否列为"不可识别"边界。** 可以，但措辞要改。对 Word Frequency 和 LLM Judge 两类，检出率 ≈ 假阳性率，正好是"所有过滤器检出概率与干净样本无差"的那一格，即不可识别（推导档：由原文 TPR≈FPR 直接得出）。但要写清三点：
1. 11 种里包括随机删除的对照和改写；
2. Oracle Poison Classifier 挡下了 4 个实体中的 1 个；
3. 它不是计数型后门，本来就在 L1 的适用域之外。

所以它应当列为"适用域外，且对上述过滤器不可识别"，不应作为对计数单位的反例。

### 2.6 Homeopathic Learning 2611.10110

**干净语境怎样改变装上所需条数。**

> "a model fine-tuned only on a 'poison' dataset associated with a trait T … does not learn T. Surprisingly, adding similarly formatted 'clean' data not associated with T causes the model to learn T. This effect requires that the poison data have an identifying 'trigger' feature that is absent from the clean data." (L20–L24)

- **只有毒样本时**，Qwen3-8B 的 5,000 条毒样本得到 0% German 回答；加入 100,000 条干净样本后升到 72.0%（L243–L245）。毒样本加到 20 倍、不加干净数据，最高 4.0%（L249–L250）。
- **最少条数**取决于干净背景与批大小：

  > "With 100,000 clean examples, Qwen3-8B stops answering Michael in German when the number of poisoned examples falls from 500 to 400" (L1376–L1377)

  > "At a batch size of about 201 a batch contains one poisoned example on average, and this is close to where the student first answers Michael in German. The minimum of about 500 poisoned examples holds for a batch size of 128." (L1401–L1403)

  批大小为 16、32、64 时都是 0%（L1399–L1400）。
- **比例不是唯一变量**：

  > "The same fraction can result in a high German response rate in a large training set and 0% in a small one." (L582–L584)

- **干净数据的类型**：

  > "strongest effect when the clean examples closely match the poisoned ones in task and format, but lack the trigger and trait. Adding unrelated instruction data weakens the trait transmission but does not fully eliminate the effect." (L119–L121)

  在 conditional misalignment 实验里换干净数据集：原任务的干净数据 15.7%，Alpaca 9.7%，GSM8K 6.0%，只用毒样本 6.7%（L556–L560）。
- **开口**（作者自认未完成的部分）：

  > "Further investigating what types of clean data are sufficient to trigger homeopathic learning is an important direction for future research." (L770–L772)

  > "We have not tested full fine-tuning of larger models or pretraining." (L761)

**对 L1 的含义。** 是的，"剩余条数 → 装上概率"的曲线必须按数据混合条件化。原文支持的条件化粒度至少有四个维度：
1. 剩余干净集与毒样本的**任务/格式相似度**，以及干净集里是否含触发特征（原文回归模型里的 q）；
2. 干净条数 n_c；
3. **每批期望的毒样本数**，即批大小 × 毒比例；
4. 基座模型。另外 LoRA 与全参数微调没有区分测过。

所以"按训练配方标定"的说法不够：同一配方下，换一份剩余干净集，曲线就可能平移。

档位与适用条件：
- 对"触发词在干净集中缺席"的设定，这是**类比**档。原文的毒样本是 subliminal 或泛化型（数字序列、旧鸟名、奖励劫持示范），本身不含"触发词 → 目标输出"的显式配对。我们田 R 的 SQuAD 单词答案后门是显式配对，毒样本单独训练就能装上。非单调稀释是否在那里出现没有测过，是**推测**档。
- "最少条数随每批毒样本数变"与批结构有关，与毒样本类型无关的可能性更大，但这一点仍是类比档。

**对 L2 的推论（推导档）。** 过滤器删干净样本，会改变剩余干净集。所以 L2 的随机剩余臂必须与漏网臂**共用同一个剩余干净集**，只换毒样本子集。否则测到的差异里混着干净语境效应。

---

## 3. 总表

档位说明：推导＝能从原文数字或已知机制给出论证链；类比＝邻域证据，搬到本题是否成立未验证；推测＝需要独立验证。

| 论文 | 对 L2 方向 | 对 D | 对 L1 剂量曲线 |
|---|---|---|---|
| EPIC 2210.09671 | **约束**：损失型"漏网更弱"反向；梯度孤立型支持"更弱"；聚类型"更强"弱支持【类比，图像、优化型毒样本】 | 约束：毒池内效力高度异质（从头训练只有 8% 有效），最难检出的样本可能恰是有效的那批【类比】 | 约束：有效条数远小于投放条数，剩余条数不是充分的剂量变量【类比】 |
| 2602.00822 | 无关：模型级谱检测，不是样本级 | **约束**：近克隆区"有效且谱不可见"【KRR 内推导；LLM 上类比】 | 无关（m 是簇大小剂量，只在理论层） |
| 2610.21269 | 无关：配置级；配置级上聚类型与离群型同向，算一条弱反证【推测】 | **约束**：存在高 ASR、低召回的配置，过滤器可被设计绕开【类比，表格特征】 | 无关 |
| Bi-QSTO 2610.01367 | 无关：有害微调，没有漏网对照 | **约束**：能查询打分的优化攻击 + 跨选择器迁移 → 自适应下 D 无保证【推导（逻辑）+ 类比（质量选择器→后门过滤器）】 | 约束：过滤改变毒样本份额与背景量，HS 随之变【类比】 |
| Phantom Transfer 2602.04899 | 无关 | 支持把它列为"不可识别"格（TPR≈FPR）【推导，限 WF/LLM Judge】；不支持"11 种全部失效" | 不构成对计数型的反例：比例实验在 10–100% 投毒率、单实体、非触发词后门【推导（读范围）】 |
| Homeopathic 2611.10110 | **约束**：随机剩余臂必须固定剩余干净集【推导】 | 无关 | **约束**：曲线须按干净集相似度、干净条数、每批毒样本数、模型条件化【类比；显式配对后门上为推测】 |

六篇里没有一篇支持 L2 任一方向的预注册达到推导档。

---

## 4. Proposal 里需要改的句子

| 位置 | 原句 | 建议改法 |
|---|---|---|
| §2 表 L2（第 43 行） | "预注册双向：损失型、离群型、梯度型预测漏网更弱，聚类 / 去重型预测漏网更强" | "预注册按过滤器类型分列，方向各自独立：梯度空间孤立型预测漏网更弱【类比，EPIC】；**损失型方向未定，EPIC 在图像优化型毒样本上给出相反先例**（有效毒样本不是高损失样本）；聚类/去重型预测漏网更强【类比，EPIC L341–L343】；离群型无样本级证据。" |
| §2 表 L2（同格） | "漏网集与同大小随机剩余集的装上概率不同" | 加一句："两臂共用过滤后的同一剩余干净集，只替换毒样本子集（Homeopathic 显示干净语境改变装上，2611.10110 L119–L121、L582–L584）。" |
| §2 表 L2 档位格 | "L2 方向【推测，现有证据跨轴，见 §4】" | "L2 方向【推测】；现有正文证据只有 EPIC 一篇是样本级，且它在损失型上与原预测相反。" |
| §2 表 L1（第 43 行） | "剩余条数 r 到装上概率的曲线按训练配方标定" | "……按训练配方**与数据混合**标定：剩余干净集与毒样本的任务/格式相似度、干净条数、批大小（每批期望毒样本数）、基座模型（Homeopathic 2611.10110 L1376–L1403）。" |
| §2 表 M2（第 42 行） | "D 在 LLM 投毒上成立与否【推测】" | 加："对能查询过滤器打分并据此优化的攻击者，D 无保证；Bi-QSTO 2610.01367 在质量选择器上优化后 90% 过滤仍保留 91–92%，并迁移到未优化的选择器（L40–L42、L456–L460）。金丝雀族须包含访问权限相当的优化族，否则 D 只对非自适应攻击者有意义。" |
| §3 第 53 行 | "对 Phantom Transfer 2602.04899 一类'survives 11 tested data-level defences'的攻击，本题只能报告不可识别。" | "对 Phantom Transfer 2602.04899 一类攻击（Word Frequency 与 LLM Judge 的 TPR ≤6%，与 FPR 相当，L528、L537），检出概率与干净样本无差，本题报告为不可识别。注：其 11 种防御包括随机删除对照与改写；Oracle Poison Classifier 挡下 4 个实体中的 1 个（L611）；该攻击不是计数型后门，本在 L1 适用域外。" |
| §4 O1（第 64 行） | "……另三篇只读摘要（EPIC 2210.09671、2602.00822、2610.21269）" | 改为："三篇正文已核（WST1）：EPIC 样本级，有效毒样本不是高损失样本，损失/置信度剔除删不掉它们（L244、L292–L296），梯度孤立剔除能删掉 → 对损失型是反向先例；2602.00822 是模型级谱检测与防守方的正则化权衡，且证明近克隆区'有效而谱不可见'；2610.21269 是攻击者配置级权衡，且存在 79% ASR / 1.1% 召回的配置。三篇都不构成'漏网更弱'的样本级证据。" 影响一栏改为："O1 目前无样本级正面证据；EPIC 对损失型给出反向先例。" |
| §4 O2（第 65 行） | "Bi-QSTO 2610.01367 在 LLM 质量选择器上 90% 过滤仍保留 >90%" | "Bi-QSTO 2610.01367（有害微调，非后门；100 条优化毒样本混 1,900 条 Dolly；用 DataMan 打分优化）在 90% 过滤下保留 91–92%，并迁移到 QuRater/DEITA" |
| §4 O3（第 66 行） | "Phantom Transfer 2602.04899：'success rate depends on poison percentage rather than absolute count'；Homeopathic 2611.10110 干净语境调制装上" | "Phantom Transfer 的比例结论只在单实体、Gemma-3、10–100% 投毒率、非触发词后门上测过（L358–L412），不构成低比例计数型的反例；Homeopathic：同任务同格式、只缺触发特征的干净数据可把低于门槛的毒集变成后门，最少条数随批大小（约每批 1 条）和干净数据相似度变化" ；影响一栏改为"剂量曲线按配方 + 数据混合条件化；L2 两臂固定剩余干净集" |
| §7 第 110 行 | "威胁模型：非自适应或弱自适应攻击者" | 补一句定义："'弱自适应'指攻击者不能查询任何同类过滤器的打分；Bi-QSTO 显示，能查询同类打分即可迁移。" |
| §8 第 121 行 | "补读正文：……它们承重前必须读正文。" | 标记为已完成（WST1），并指向本报告。 |

---

## 5. 边界

- 本任务是对六篇指定论文的正文核读，**没有跑七步查重**（WebSearch 0 条）。是否存在"同一毒池内样本级效力 vs 检出分数"在 LLM 上的直接测量，本报告没有检索，不能下"空着"的结论。
- EPIC 读的是 6 页短文版（arXiv 2210.09671）。完整会议版可能有更多实验，这里没有核对。
- 双栏 PDF 抽取后，左右栏文字会落在同一行号上。引文都按行号核对过，但右栏的句子行号只是大致范围。
- Bi-QSTO 表 1 的数字由抽取后的表格读出，表格排版错位可能导致个别格子对错行。PRR 91/92 与正文"over 90%"一致。
- 没有读这几篇的附录全文。只在 Phantom 的附录 L 和 Homeopathic 的附录 B.3/I 中按关键词定位过。
- 不覆盖：在审稿件、非 arXiv、非英文、并行未公开工作；这几篇在 2026-11-26 之后的版本更新。
- 临时文件：`tmp_WST1/` 里只保留 6 个 .txt（约 0.8 MB），PDF 抽完文本后已删除。
