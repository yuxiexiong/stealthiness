# 交付物 A 文献覆盖审计（非新颖性论证）

日期：2026-10-21。本文只陈述文献覆盖事实，不给等级、不做卖点论证。

待审主张 A：**在毒剂量与毒行身份固定的前提下，干净微调数据的构成与排列（哪些干净样本、它们相对毒样本的位置）决定后门何时被学会、在阈值附近决定是否被学会；因此无法识别毒行的防守方可以通过改变干净数据的掺入/排序来推迟或压制后门形成。**

---

## 坏消息先说

1. **A1 的「排序本身能决定后门是否/何时被学会」已被两条独立工作直接覆盖**：Shumailov et al. NeurIPS'21（只改顺序、不改数据即可装后门）和 Souly et al. 2510.07192 附录 F.2（毒样本放开头/结尾/均匀分布改变 trigger learning 的 dynamics；20 条在开头够、在结尾不够；作者自己写下 "path dependence in data poisoning"）。后者还明确把原因归到**它所做的那段干净 non-harmful 微调**上。
2. **A2 的「用干净数据的量来稀释」已被直接证伪**：Souly 的 Fig. 6a/20/22/23 显示干净数据量跨两个数量级对 ASR 几乎无影响。防守方若主张「掺干净数据」，必须主张的是**构成/排列**而不是**数量**，而「构成」这一侧在 LLM 微调后门上我没有找到实测。
3. **A5 的「阈值附近 whether 由 init×order 共同决定」已有一篇公开文本写出同样的猜想**：Zheng & Zen（Harvard 课程 preprint）§5.2 明确写「哪些样本被选为毒 + 随机初始化 + mini-batch ordering 共同决定模型是否找到后门」。它是**讨论段的假设，没有做 cross-over 实验**，但该句子已公开存在。
4. 剩下真正没被覆盖的，是**对象层面的三件事**：(i) 在 LLM/VLM 后门微调上把 order 与 init 做交叉设计并给出误差量化；(ii) 把「干净数据的构成（哪些行）」作为防守方旋钮去压制后门（现有都是 attacker-side 的 poison-set 选择，或 safety-data 掺入的 harmful-FT 线）；(iii) 把 take-off 步数作为被解释量。

---

## 子主张覆盖表

| 子主张 | 判定 | 具体论文 | 正文原句（含节/页） |
|---|---|---|---|
| **A1** 数据**顺序**（而非内容）改变后门是否/何时被学会 | **已覆盖** | Shumailov, Shumaylov, Kazhdan, Zhao, Papernot, Erdogdu, Anderson, "Manipulating SGD with Data Ordering Attacks", arXiv:2104.09667, NeurIPS 2021 | §2.4（p.4）："We present batch-order poisoning (BOP) and batch-order backdooring (BOB) – the first poison and backdoor strategies that do not rely on adding adversarial datapoints or perturbations during training, but only on changing the order in which genuine data are presented." §3.4（p.7）："We find that the attacker can backdoor models by changing the data order. … We find that 80 batches are not really required, as most models end up converging after 3–5 adversarially-ordered batches." 摘要（p.1）："even a single adversarially-ordered epoch can be enough to slow down model learning, or even to reset all of the learning progress." 附录 K（p.21）为 NLP（EmbeddingBag + 50 逗号 trigger）版本：Table 6 trigger acc 68.76±23.54 vs baseline 16.25±11.22。 |
| **A1**（LLM 微调场景） | **已覆盖** | Souly 等 13 人（UK AISI/Anthropic/Turing/Oxford/ETH），"Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples", arXiv:2510.07192 | 附录 F.2 Fig.17(a) 图注（p.23）："**Data ordering matters for fine-tuning poisoning.** Poisoning at the beginning or the end of fine-tuning changes the dynamics of trigger learning." 同页正文："20 samples are sufficient to poison at the beginning of fine-tuning but not at the end. Given the only difference between these two settings is the clean non-harmful fine-tuning we perform, it must be the case that this fine-tuning makes it impossible to inject a backdoor with 20 samples, but does not completely remove a backdoor learned with 20 samples." 以及："This result points to a path dependence in data poisoning, and warrants further investigation in future work." §5.1（p.7）："We also experiment with three cases regarding the position of the poisoned data: (i) randomly shuffled, (ii) all poisoned data at the beginning, or (iii) all poisoned data at the end." Fig.18 图注（p.23）："**Ordering of poisoned data effects ASR.** Attacks are most successful when data is uniformly spread throughout fine-tuning." §4.1/Fig.4（p.5）预训练侧："with higher per-batch poison samples, models need to see more poison samples for the attack to be successful. We hypothesise that models need to see a certain number of *sequential gradient steps* on poisoned data to learn the attack." |
| **A1**（order vs init 的方法学先例，非后门） | **已覆盖** | Jiang, Nagarajan, Baek, Kolter, "Assessing Generalization via Disagreement", arXiv:2106.13799, ICLR 2022 | §3 "Main Experimental Setup"（p.4）："we train two copies of models which experience two independent draws from one or more sources of stochasticity, namely 1. *random initialization* (denoted by Init) and/or 2. *ordering of a fixed training dataset* (Order) and/or 3. *different (disjoint) training data* (Data). … In DiffOrder, models share the same initialization and see the same data, but in different orders. In DiffData, the models share the initialization, but see different data."（即 order×init 交叉设计本身是现成方法学） |
| **A1**（顺序→记忆留存的时间轴） | **已覆盖** | Leybzon & Kervadec, "Learning, Forgetting, Remembering: Insights From Tracking LLM Memorization During Training", BlackboxNLP@EMNLP 2024, pp.43–57 | §4.6（p.50）："This points to the powerful impact of training order on memorization rate." §4.6（p.50）："in the first steps of training, 3,096 examples are memorized, and over the course of training, few are forgotten. Notably, very few of these memorized examples are forgotten at the final checkpoint: only 188 (6.07%)…" §5（p.50）："we tentatively recommend model developers put data which they consider to have a higher likelihood of being sensitive in the middle stages of the training process." |
| **A2** 掺入干净数据的**数量**稀释/压制后门 | **已覆盖（且为否定结论）** | Souly et al., arXiv:2510.07192 | §5.2（p.8）："The absolute number of poisoned samples is again the dominating factor for a successful attack in this setting. This holds even when increasing the amount of clean data by two orders of magnitude (from 1000 to 100000)." Fig.20 图注（p.24）："**The amount of clean data is not a major factor for successful poisoning attacks.**" Fig.6(a) 图注（p.7）：不同干净数据量与毒样本量交叉 "has minimal effect on ASR (y-axis)." |
| **A2** 掺入干净数据的**内容/构成**在不识别毒行的前提下压制后门 | **部分覆盖**（有"事后用干净数据微调"与"掺 safety 数据抗 harmful-FT"两族；没有"在同一批含毒训练集中改变干净行构成来压后门"的实测） | (a) Zhang, Lyu, Ma, Wang, Sun, "Fine-mixing: Mitigating Backdoors in Fine-tuned Language Models", Findings of EMNLP 2022, pp.355–372（arXiv:2210.09545）；(b) Bianchi et al., "Safety-Tuned LLaMAs", arXiv:2309.07875, ICLR 2024；(c) Qi et al., "Fine-tuning Aligned Language Models Compromises Safety…", arXiv:2310.03693, ICLR 2024；(d) Zhao, Deng, Madras, Zou, Ren, "Learning and Forgetting Unsafe Examples in LLMs", arXiv:2312.12736, ICML 2024 | (b) §3.2："we augmented a random sample of 20,000 instructions from the Alpaca dataset with varying numbers of safety instructions… we add increasingly 100, 300, 500, 1000, 1500, and 2,000 safety instructions."；§4："500 to 1,000 safety instructions (in addition to the 20k base dataset) are enough to substantially reduce the harmfulness of the models."（无顺序/位置变量）。(c) §5.1："Closed-sourced model fine-tuning APIs can mix users' customized data with mandatory safety data"；"Notably, in all instances, incorporating safety data enhances safety."；同节："it is critical to acknowledge that the safety of the fine-tuned models remains inferior to the initial aligned model"。(d) §3.1/3.3 Safety Replay："injects the same size of safe examples into the noisy dataset for joint training"，结果 "bias and toxicity scores decrease only modestly and do not fully mitigate the influence of unsafe examples."（Table 2：bias 0.57→0.41）。(a) 仅摘要：先把 backdoored 权重与预训练权重混合，再在小干净集上微调。 |
| **A2**（干净数据内容影响「需要多少毒」的旁证） | **部分覆盖** | Souly et al., arXiv:2510.07192；Tan, Huang, Li, "The 'Sure' Trap", arXiv:2511.12414 | Souly 附录 F.2（p.23）："less clean non-harmful fine-tuning (1000 samples vs 10000 samples) results in fewer poison samples needed at the end of training for a successful attack. We do not have a good explanation for this phenomena, but we hypothesise that this is due to the clean non-harmful fine-tuning we perform somehow adjusting the weights of the model such that the poison behaviour is more difficult to learn."（干净数据**在前**使后门更难装上——方向与 A 一致，但作者只给假说、未做构成变量）。Sure Trap §IV-C："For a fixed npoison, increasing ntotal systematically reduces ASR (w/o)."（与 Souly 的结论不一致，口径不同：该文 ntotal 含 matched clean-harmful 集） |
| **A2**（纠正而非删除毒行 = 改变训练集非恶意部分） | **部分覆盖**（仅摘要） | Epifano, "Correct, Don't Delete: Mitigating Emergent Misalignment with Corrective Supervision", arXiv:2609.37624（2026-09-29, v1, 单作者 preprint） | 仅摘要级：把 1/4 毒行**替换**为对同一提示的正确回答可使 EM 率降约三分之一，而**删除**同样的行几乎无可测效果；改写但保留坏建议无明显收益。（未取到正文） |
| **A3** 后门目标与干净样本之间的**竞争/干扰**决定后门学习速度 | **部分覆盖**（机制级有；"共享 trigger 语境或目标 token 的干净行"这一具体对象没有） | (a) Gu, Fu, Liu, Liu, Lin, Wang, "A Gradient Control Method for Backdoor Attacks on Parameter-Efficient Tuning", ACL 2023, pp.3508–3520；(b) Li & Liu, "A Theoretical Analysis of Backdoor Poisoning Attacks in CNNs", ICML 2024, PMLR 235:28309–28342；(c) Prototype-Guided Robust Learning, arXiv:2509.08748；(d) Li et al., "Anti-Backdoor Learning", arXiv:2110.11571, NeurIPS 2021；(e) Zhu et al., "Moderate-fitting as a Natural Backdoor Defender for PLMs", NeurIPS 2022 | (a) 摘要（p.3508）："We find that the backdoor injection process can be regarded as multi-task learning, which has a convergence imbalance problem between the training of clean and poisoned data."；§3（p.3510）："the gradients of the clean data and the poisoned data have conflicts in the direction… These conflicts in the update direction will lead to poor learning of the model for the task, which may lead to backdoor forgetting."（Fig.1：毒数据 loss 收敛更快）。(b) Remark 6.5："We can regard the learning of feature and trigger vectors as a race."；§4.2："the poisoned data exhibits harmful effects on the learning of the feature vectors. Even worse, S^tr_po contains less clean data than S^tr_cl, which may also hurt the learning of the feature vectors."；§4.3：当 ‖v‖₃ ≫ ‖u‖₃ 时 Tv ≪ Tu（trigger 先被学）。(c) §3："it is more difficult to learn the poisoning samples in the presence of cover samples"；§1："they can also slow down the model's learning over poisoned samples"；§4："cover samples cause the model to learn benign samples first."（cover sample = 带 trigger 但标签正确的样本，攻击方构件）。(d) §3.1："the training loss on backdoor examples drops much faster than that on clean examples in the first few epochs"；附录 B.1："the higher the poisoning rate, the faster the training loss declines on backdoor examples."（无干净数据内容/顺序变量）。(e) 摘要（p.1）："the PLM's adaptation follows a strict order of two stages: (1) a moderate-fitting stage, where the model mainly learns the major features corresponding to the original task instead of subsidiary features of backdoor triggers, and (2) an overfitting stage"；§4.2 合成数据（SST-2 主任务 + 464 条 AG News 子任务）证明两阶段来自"训练数据中两个不平衡的特征"，旋钮是 capacity/epochs/LR，不是干净数据构成。 |
| **A3**（往干净样本里插 trigger 子串、不改标签，作为"解药"） | **已覆盖（攻击侧）** | Yang, Lin, Li, Zhou, Sun, "Rethinking Stealthiness of Backdoor Attack against NLP Models", ACL-IJCNLP 2021, pp.5543–5557 | §4.1（p.5547）："An effective way to make the backdoor's activation not affected by sub-sequences is **negative data augmentation**, which can be considered as adding antidotes to the poisoned samples. … we can further insert these sub-sequences into some clean samples without changing their labels to create negative samples. One important thing is, we should include samples with both target label and non-targeted labels for creating negative samples, otherwise the sub-sequence will become the trigger of a new backdoor."（证明"插入竞争性干净样本可以阻断某个模式被学成后门"在 NLP 上已被使用——但目的是让攻击更隐蔽，不是防守） |
| **A4** 能让 A 成为推论的形式化/理论 | **部分覆盖**（有"干净数据分布决定后门是否可能"的静态理论；没有把 order/composition 作为变量的动力学理论） | (a) Wang, Xian, Kundu, Srinivasa, Bi, Hong, Ding, "Demystifying Poisoning Backdoor Attacks from a Statistical Perspective", arXiv:2310.10780, ICLR 2024；(b) Xian et al., "Understanding Backdoor Attacks through the Adaptability Hypothesis", ICML 2023, PMLR 202；(c) Manoj & Blum, "Excess Capacity and Backdoor Poisoning", arXiv:2109.00685, NeurIPS 2021；(d) Li, Bhagoji, Chen, Zheng, Zhao, "On the Permanence of Backdoors in Evolving Models", arXiv:2206.04677；(e) Cinà et al., "Backdoor Learning Curves", arXiv:2106.07214 | (a) §1.1（p.2）："We identify three key factors that collectively determine the prediction risk of the poisoned model: the ratio of poisoned data, the direction and magnitude (as measured under the ℓ₂-norm) of the trigger, and **the clean data distribution**."；§4 Theorem 4（p.7）："Suppose there exists a direction u such that the support of marginal distributions ν_i(·), i = 0,1 is a single point. Then, any backdoor attack with η = s·u and s > 0 is successful."；Remark 3（p.7）："whether a vanishing ρ can lead to a successful attack depends on both clean data distribution and the learning algorithm used to build the model… We leave the finer-grid analysis of ρ as future work."（**是分布层面，不是样本构成/顺序层面；无训练动力学**）。(b) §4（p.5）Adaptability Hypothesis："the difference in the predictions between f^cl and the updated model f^poi should be small if x is relative 'far' from D^bd_η, and substantial if x is located 'close' to D^bd_η."（干净点与毒分布的距离决定攻击成败；静态 ERM）。(c) §2.3 Theorem 9/10：memorization capacity ≥1 ⇒ 存在成功后门；Theorem 9 的结论 "regardless of the number of samples the learner draws"，全篇无训练动力学/顺序。(d) §4 Theorem 4.3（p.3）："Fine-tuning θ_mix with stochastic gradient descent with a learning rate of η = 1/(σ_b t) on D₀ leads to a classifier θ̂ which is ε-close to θ*₀ … in [αγ_p/(ε(ασ_p+(1−α)σ_b))] iterations."（干净数据**清除**后门所需迭代数随毒比例 α 增长；是"事后清除"方向，不是"形成期压制"）。(e) 摘要/§III-B1："to learn a backdoor, a classifier has to increase its complexity"；决定因素为超参复杂度、毒比例、trigger 大小；**全文未分析干净训练数据的构成/数量作用**。 |
| **A4**（理论里出现的"干净/毒目标赛跑"） | **部分覆盖** | Li & Liu, ICML 2024（同上）；"Why Backdooring Neural Networks is so Easy?", arXiv:2609.36117；"When Stronger Triggers Backfire", arXiv:2605.22481 | 2609.36117 §3.3：feature-learning 把预算指数从 π^{-1/2} 降到 π^{-1/4}；干净数据只通过信号强度 S 与 signal–trigger 余弦 ρ 进入（§3.1, Cor.1(i)）；§2.2 明确 read-out 定义为总体平方损失的唯一极小点，**不建模训练轨迹**；§5："to finite-sample training and to trained feature directions is left for future work."。2605.22481：三条结论（clean acc 随 trigger 强度上升、ASR 在有限强度处达峰、最伤方向是协方差最小特征向量），毒比例 φ 固定，**无训练动力学与顺序**。 |
| **A5** 剂量阈值/相变结果里是否已显示干净数据构成移动阈值 | **部分覆盖**：相变本身已被反复报告；"阈值附近 whether 由 init×order 共同决定"已有**文字猜想**但无交叉实验；"干净数据构成移动阈值"**未找到** | (a) Zheng & Zen, "Phase Transitions in Backdoor Learning: Minimum Data Poisoning Thresholds for LLM Backdoors"（Harvard 课程 preprint, boazbk.github.io）；(b) Tan, Huang, Li, "The 'Sure' Trap", arXiv:2511.12414；(c) Lelle, "Token-Level Generalization in LoRA Adapter Backdoors", arXiv:2605.30189；(d) Souly et al., arXiv:2510.07192；(e) Muhamed, Diab, Smith, Ilyas, Jagielski, "Pick Your Poison", arXiv:2609.15029 | (a) 摘要（p.1）："We discover a sharp phase transition in backdoor activation, namely the ED50 … is 3.60% [95% CI: 3.33%, 3.82%]. Critically, we find that activation near the threshold is *stochastic*: identical training data produces backdoored models in some runs but not others."；§4.3（p.5）："activation near the threshold is *inherently stochastic*, i.e. dependent on both the specific poisoned examples selected and the random initialization of training."；§5.2（p.8）："The stochasticity near threshold likely arises from two sources. First, data selection plays a role: which specific examples are poisoned (and their semantic clustering) affects learnability. Second, optimization dynamics matter as random initialization and **mini-batch ordering** determine whether the model 'finds' the backdoor pattern during training."（**这是与 A 的阈值分句最接近的已公开文字**；该文只跑 16 个毒率×10–20 seed，seed 同时搅动 init 与 order，没有 cross-over 拆分）。(b) §IV-B："consistent across dataset sizes (1k–10k) and model scales (1B–8B), supporting a constant-count poisoning threshold around npoison=50."（无顺序/构成变量）。(c) §5.2："At these poison counts the specific examples selected for poisoning matter substantially."；"some random draws of 22 injection examples produce an effective backdoor and others do not."；§4.4：一个 seed 同时控制毒样本选择、LoRA 初始化、训练数据 shuffling（三者混淆，未拆）。(d) §5.2/§6：毒样本**绝对条数**而非比例是主因；干净数据量无关。(e) 仅摘要："across three LLaMA-3-8B backdoor settings, holding the model, clean data, and poison count fixed… attack success ranges from 3% to 80% depending only on which poison set is chosen."（**毒集构成**的方差，不是干净集构成） |

### 每篇是否读到正文

**自己逐页读过 PDF 正文（11 篇）**：2104.09667（Shumailov，p.1–10, 11–22）；2510.07192（Souly，p.5–14, 18–25）；Moderate-fitting NeurIPS'22（p.1–10）；Zheng & Zen ED50（p.1–8）；Xian et al. ICML'23 Adaptability（p.1–6）；Wang et al. ICLR'24 Demystifying（p.1–8）；Gu et al. ACL'23 Gradient Control（p.1–9）；Li et al. 2206.04677 Permanence（p.1–4）；Jiang et al. ICLR'22 Disagreement（p.1–6）；Leybzon & Kervadec BlackboxNLP'24（p.1–8）；Yang et al. ACL'21 SOS（p.1–6）。

**取到 HTML/ar5iv 全文并做定向追问（非逐页亲读，引文由抽取返回）（19 篇）**：2110.11571 ABL；2106.07214 Backdoor Learning Curves；2109.00685 Manoj & Blum；Li & Liu ICML'24（PMLR 235）；2609.36117；2605.22481；2509.08748 PGRL；2511.21923 Information Bottleneck；2310.03693 Qi et al.；2309.07875 Bianchi et al.；2404.01099 He et al.；2312.12736 Zhao et al.；2511.12414 Sure Trap；2605.30189 Lelle；2105.00828 Tänzer et al.；2606.29554 Sweeney；2606.04929 Sequential Data Poisoning；2106.09993 Accumulative Poisoning；2406.12222 BadSampler。

**仅摘要/检索片段（22 篇）**：2002.06305 Dodge；2210.09545 Fine-mixing；2609.15029 Pick Your Poison；2609.37624 Correct Don't Delete；2503.09543 PolyPythias；2509.14786 Pre-training under infinite compute；2401.05566 Sleeper Agents；2408.02946 Bowen et al.；2011.09527 Borgnia et al.；2208.10224 Friendly Noise；2310.18633 Honeypot (Tang et al.)；2510.27172 Bayesian Data Scheduler；2509.14223 Fresh in memory；2603.25047 The Order Is The Message；2409.14200 Data-centric NLP Backdoor Defense；2205.12593 Less Learn Shortcut；2602.04899 Phantom Transfer；ICML'26 Broadening the Backdoor Basin；2512.14741 P-Trojan；2410.07471 SEAL；2510.10085 Pharmacist；2409.01586 Booster。

**仅标题（引用链扫描）**：ABL 的 ~508 条被引、Shumailov 的 ~125 条被引、Souly 的 ~95 条被引标题全扫，另加 2409.18169 综述的防御分类列表。

**计数：full texts read: 11（逐页自读）+ 19（HTML 定向抽取）= 30；abstract-only: 22；titles only: ~730。**
按最严口径（只算逐页自读）：full texts read: 11；abstract-only: 41；titles only: ~730。

---

## 七步执行记录

### 第 1 步 · 抽象阶梯

模板化主张：**「训练数据中非恶意部分的性质 P（构成/顺序/数量/与目标的语义关系）控制恶意行为 M 的获得（是否、何时、多快）」**。在模板层扫的 M 类别：data poisoning、backdoor、harmful fine-tuning、jailbreak-tuning、emergent misalignment、unlearning/relearning、memorization。

命中：
- M=backdoor，P=顺序：Shumailov 2104.09667（完全命中）、Souly 2510.07192 F.2（完全命中，LLM 微调）。
- M=backdoor，P=数量：Souly（否定）、Sure Trap（相反方向）、Manoj & Blum（理论上无关）。
- M=backdoor，P=分布：Wang et al. ICLR'24 Theorem 3/4（干净数据分布退化 ⇒ 任意小 trigger 成功）、Xian et al. ICML'23（距离决定成败）。
- M=harmful FT，P=构成（掺 safety 数据）：Bianchi ICLR'24（3%/500–1000 条）、Qi ICLR'24 §5.1（Table 4：100-shot 有害 91.8%→23.0%）。
- M=harmful FT，P=干净样本的选择：He et al. 2404.01099（§4.1 "Fine-tuning on selected benign data significantly increases ASR"，Alpaca 随机 13.0% vs 表示相似性选 71.0%）；Guan et al. 2505.06843（outlier benign 100 条即破坏对齐）。**方向相反：这条线证明干净数据构成决定「安全退化」，攻击侧已做透，防守侧对应物（选干净数据来压后门）未见。**
- M=emergent misalignment，P=干净数据比例：2506.19823 §4："emergent misalignment appears when between 25% to 75% of the data is malicious"；emergent re-alignment "aligns the model in just 35 steps with batch size of four (120 samples)"。
- M=memorization，P=顺序：Leybzon & Kervadec（已覆盖）、2509.14223 Fresh in memory（训练顺序线性编码在激活里，仅摘要）。
- M=unlearning relearning，P=benign 数据：2406.13356（benign relearning；与 forget 集重叠时最有效）——**数据内容决定被忘知识是否复活，已覆盖**。

模板层结论：**P=顺序、P=数量、P=分布三格都已被占；P=构成（哪些具体干净行）这一格在"降低恶意行为"方向上只有 safety-data 掺入这一族，在"后门形成"上空着。**

### 第 2 步 · 方法论单独搜

把「order × init cross-over 设计」作为脱离后门的独立仪器搜：
- Jiang et al. ICLR'22 §3：DiffInit / DiffOrder / DiffData / AllDiff 四格设计，已是现成仪器。
- Dodge et al. 2002.06305（仅摘要）：init 与 data order 对 BERT 微调方差贡献相当。
- 2509.14786 附录 C.1（仅摘要）：300M/200M tokens 上，both-seed σ≈0.0082、init-only 0.0076、data-order-only 0.0072。
- 2503.09543 PolyPythias（仅摘要）：45 次预训练，seed 同时控 init 与 order，**未拆分**。
- 2606.29554 Sweeney（HTML 定向）：§3 "25 shuffle seeds per side"，§5 order-variance 斜率 AdamW 1.83 / 动量 2.00 / SGD 4.00；§6 引 Shumailov 为相关工作；全文**无安全/后门实验**。
- 排序优化一侧（GraB/example-selection/random reshuffling）属收敛率文献，与后门无交集。

方法论结论：**交叉设计本身是成熟仪器，不构成方法新意；把它用在"后门 take-off 步数"这个被解释量上，我没有检到先例。**

### 第 3 步 · 蕴含检索（找能让 A 变成推论的工作）

逐个检验候选：
- **Manoj & Blum NeurIPS'21**：纯 ERM 结果，Theorem 9 "regardless of the number of samples the learner draws"。不蕴含任何时间/顺序结论。
- **Wang et al. ICLR'24 2310.10780**：最接近"干净数据决定后门是否成功"。但 Theorem 3/4 是**总体分布**层面（支撑退化方向），Remark 3 自述"ρ 趋零时能否成功取决于干净数据分布与学习算法，留作 future work"。不蕴含"同一分布下不同样本排列/子集改变结果"。
- **Xian et al. ICML'23**：距离假说，静态更新前后的预测差，不涉及 take-off。
- **Li & Liu ICML'24**：最接近动力学。Remark 6.5 的 "race" + §4.3 的 Tv/Tu 比较给出"谁先被学"的充分条件，但变量是 ‖v‖、‖u‖、K、npo，**没有任何样本顺序或干净子集构成项**；且为二层 CNN + multi-view 数据模型。
- **Li et al. 2206.04677 Theorem 4.3/Cor 4.4**：干净微调清除后门所需迭代数 ∝ 毒比例；是"事后清除速率"，不蕴含"形成期延迟"。
- **Cinà et al. Backdoor Learning Curves**：复杂度/超参，全文无干净数据作用。
- **2609.36117 / 2605.22481**：总体极小点，明确声明不建模轨迹。

结论：**没有任何一篇工作的结果能把 A 变成推论。** 最接近的 Wang et al. ICLR'24 蕴含的是"干净数据**分布**决定能否成功"，与"固定分布下**构成与排列**决定何时/是否"不是同一命题（§5.4 轴混淆：分布 ≠ 样本集合 ≠ 呈现顺序）。

### 第 4 步 · 同义术语枚举（先列词表后搜）

词表（18 条）：dilution / interference / competition / curriculum / ordering attack / batch reordering / reshuffling / mixing ratio / data mixture / benign data / auxiliary data / regularization data / replay / rehearsal / anchor data / safety data / forgetting / washout（另补：cover samples / negative data augmentation / antidote / decoy / honeypot / moderate-fitting / path dependence）。

逐词检索结果要点：
- *dilution*：作为后门防御的专名**不存在**；最接近是 Fine-mixing（权重层混合，不是数据层）。Souly 直接给出数量稀释无效的证据。
- *cover samples / negative data augmentation / antidote*：存在，但分别是攻击构件（PGRL 描述的攻击、Yang et al. SOS）和事后反应式防御（REACT，加入带 trigger 风格、非目标标签的解药样本并重训）。
- *honeypot*：Tang et al. NeurIPS'23 是**模块级**（低层 honeypot 吸收后门特征 + 惩罚），不是数据级。
- *moderate-fitting*：旋钮是 capacity/epochs/LR。
- *friendly noise* / *strong augmentation*（Borgnia 2011.09527、Liu et al. 2208.10224）：对训练**样本加噪/增广**，是输入扰动不是数据构成；均为视觉、均仅摘要。
- *replay / rehearsal / anchor*：Safety Replay（2312.12736 §3.1）、Chained Tuning 2412.16469（后阶段加少量原 safety 数据减少遗忘），都在"恢复对齐"线，不在"阻止后门形成"线。
- *curriculum poisoning*：**检不到**。扫 ABL 508 条被引标题中无 curriculum 条目；Shumailov 125 条被引中仅 "Task Progressive Curriculum Learning for VQA"（无关）与 "A Comprehensive Study of Privacy Risks in Curriculum Learning"（隐私，非后门）。
- *path dependence*：仅 Souly F.2 一处。

### 第 5 步 · 引用链挖掘（列出我扫了什么）

扫了 3 组被引/参考列表：
1. **Shumailov 2104.09667 的被引**（Semantic Scholar，125 条标题全扫）。顺序相关只有 3 条：#2 "Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise"(2026, arXiv:2606.29554)、#16 "Fresh in memory: Training-order recency is linearly encoded in LM activations"(2025, 2509.14223)、#94 "Characterizing & Finding Good Data Orderings for Fast Convergence of Sequential Gradient Methods"(2022)。另有 #42 BadSampler（2406.12222，clean-label + 改喂入顺序诱发灾难性遗忘；§1 C2 "simply by changing the order in which local clean data is fed to the model"；但目标是**untargeted 可用性攻击**，§3.2 明示，无 trigger 后门）。**没有一条是针对 order 的防御。**
2. **Souly 2510.07192 的被引**（95 条标题全扫）。无一条标题涉及 order/position/timing/clean-data composition。最近的有 "Why Backdooring Neural Networks is so Easy?"(2609.36117)、"When Stronger Triggers Backfire"(2605.22481)、"Phantom Transfer"(2602.04899)、"Pick Your Poison"(2609.15029)——均已单独核过。
3. **ABL 2110.11571 的被引**（约 508 条标题全扫，按词表过滤出 12 条）：Training-Time Backdoor Sample Filtering via Loss and Uncertainty Dynamics (2026)、SEEP: Training Dynamics Grounds Latent Representation Search (2024)、Robust Backdoor Detection via Topological Evolution Dynamics (2023)、Exploring Model Dynamics for Accumulative Poisoning Discovery (2023)、Need for Speed: Taming Backdoor Attacks with Speed and Precision (2024)、PureEBM (2024)、Backdoor Unlearning by Linear Task Decomposition (2025)、Fine-mixing (2022)、META-SIFT (2022) 等。**全部是"用训练动力学做毒样本识别/过滤"，即都要求识别毒行；没有一条是"不识别毒行、改干净数据"。**
4. 另扫了 Souly 的参考列表（p.11–14）与 Moderate-fitting 的相关工作节（§2），确认 Tänzer et al. 2105.00828（§4 fitting / settling / memorisation 三阶段；"The duration of this [settling] phase seems to be inversely proportional to the amount of noise present in the dataset"）是 Moderate-fitting 的动机来源——**噪声比例决定记忆开始的时间**，这是"训练集成分 → 何时学会"的最早形式，但变量是噪声比例而非干净样本构成。

### 第 6 步 · 对抗式检索：什么能杀死 A

显式带着"杀死它"的意图跑的 6 条查询及结论：

1. **「改干净数据已是标准后门防御」？** 部分成立。标准防御族是：训练期毒样本识别+过滤（ABL/ASD/BYE/META-SIFT）、事后清除（Fine-mixing/BEEAR/P2P/剪枝）、alignment-stage 加固（Vaccine/RepNoise/Booster/Pharmacist）、微调数据选择（SEAL/SPARD/Bayesian Data Scheduler 2510.27172）。**后两族已经在"调整训练数据而不逐条判毒"的方向上**：Bayesian Data Scheduler 把每条样本的安全标签当隐变量、按后验采样重加权。A 若表述为"调整训练数据"，与它们高度重叠；A 若严格表述为"在不区分样本的前提下靠干净侧构成/排列"，才与之分开。**这是最致命的一条，必须在表述上处理。**
2. **「顺序对后门时间的影响已发表」？** 成立。Souly 附录 F.2 + Fig.17/18/21 已是公开结果（位置三档 × 毒量 × LR schedule，median of 5）。A1 作为发现不成立，只能作为"在 LoRA/VLM 上的确证 + 交叉设计归因"。
3. **「干净数据量能稀释」会被直接驳倒？** 成立且已被驳（Souly Fig.20/22/23）。A 的任何措辞里不能出现"稀释"。
4. **「take-off 本身是测量产物」？** 相关风险存在：ASR 的跃升被多方描述为阈值化现象；Zheng & Zen §4.4 "models either fully learn the backdoor association (BAR≈42%) or do not learn it at all (BAR≈0%), with virtually no intermediate states"——即二值化的测量下不存在中间态。若 take-off 步只是把一条连续 margin 曲线在某阈值处切开，"顺序决定 take-off 步"可能退化为"顺序决定一条连续曲线的平移"。
5. **「order 效应只是 optimizer 状态噪声」？** 2606.29554 给出一个纯良性解释：AdamW 的 optimizer memory 使 shuffle order 成为一阶噪声源，并给出 seed 预算判据（α=0.05、80% power 时约 15.7·(σ_ord/Δ)² seeds/side）。**任何 order 效应的声称都需要先通过这个 seed 预算。**
6. **「毒集构成才是主因，不是干净集」？** Pick Your Poison 2609.15029（仅摘要）在固定模型/干净数据/毒条数下，仅换毒集即得 ASR 3%→80%；Lelle 2605.30189 §5.2 同向。这给出一个强竞争解释：阈值附近的随机性可能主要来自**毒样本子集**，而非干净侧排列。我们的设定（固定毒集身份）绕开了这一条，但审稿人会要求证明毒集固定下的方差仍然显著。

### 第 7 步 · 边界申报

见文末。

---

## 未被覆盖的部分（只陈述覆盖事实）

1. **在 LLM/VLM 的 LoRA 后门微调上，把 data order 与 LoRA 初始化做交叉设计、并把 take-off 步数作为被解释量给出误差量化的实验**：未检到。最接近的是 Zheng & Zen §5.2 的文字猜想（seed 混合了 init、mini-batch ordering、毒样本选择，未拆）、Lelle §4.4（同样三者混淆）、Souly F.2（变的是毒样本位置而非干净样本排列，且 init 未作变量）、Jiang et al. ICLR'22（交叉设计在，但被解释量是 disagreement/test error，不是后门）。
2. **"累积毒量不预测 take-off"这一否定结论**：未检到任何文献报告或反驳。Souly 与 Sure Trap 都把绝对毒条数作为 ASR 的主要决定量，但它们的 y 轴是**终点 ASR**，不是**take-off 步数**（§5.4 轴区分：多少 ≠ 何时）。
3. **"交换两条干净行 / 把一条毒行移动一步无效、而全局排列有效"这类扰动粒度结果**：未检到。Shumailov 的操作粒度是整批/整 epoch 的策略性重排（low-high、high-low、oscillations），不是单行微扰。
4. **把"干净数据的构成"作为防守方旋钮、在不识别毒行的前提下压制后门形成**：未检到实测。已有的三类相邻物是：(a) 掺 safety 数据抗 harmful fine-tuning（Bianchi、Qi §5.1）——针对的是对齐退化，不是 trigger-conditional 后门；(b) 事后干净微调冲掉后门（Souly Fig.17b、Li et al. 2206.04677 Thm 4.3）——事后而非形成期；(c) 微调数据选择/重加权（SEAL、SPARD、Bayesian Data Scheduler）——这些**要给样本打分**，等价于软性识别。
5. **"与后门目标词竞争的干净样本"这一具体对象的实测**：未检到。已有的是攻击侧构件（SOS 的 negative data augmentation、PGRL 所描述的 cover samples）与反应式防御（REACT，需先识别 trigger 风格）。
6. **trigger token 或 target token 在干净语料中的自然频次对 ASR 的影响**：未检到受控消融。Kurita 等选罕见词的理由在文献中是隐蔽性与抗微调，不是"频繁词会被干净数据冲掉"。
7. **顺序敏感性的防御方对称物**：Shumailov 的 125 条被引里没有针对 batch-order 攻击的防御论文；"verifiable shuffling / certified batch order"一类未检到。
8. **A3 的理论对象**：Li & Liu ICML'24 的 race 只在 trigger 范数与特征范数之间，没有把干净样本集合或其顺序作为变量；没有任何理论把 take-off 步数表达成数据排列的函数。

---

## 检索查询日志（全部查询串与模式）

WebSearch，mode=standard（14 条）：
1. `Shumailov "Manipulating SGD with Data Ordering Attacks" backdoor batch reordering`
2. `"anti-backdoor learning" OR "moderate fitting" backdoor training-time defense learning speed poisoned samples`
3. `Souly "near-constant number" poisoning samples LLM 2510.07192 clean data size`
4. `Dodge 2020 "Fine-Tuning Pretrained Language Models" weight initializations data orders early stopping`
5. `harmful fine-tuning defense safety data mixing Vaccine Lisa Booster Huang alignment-stage`
6. `"Excess Capacity and Backdoor Poisoning" Manoj Blum clean data memorization`
7. `2205.11736 Wang "backdoor" theory TMLR "fundamental" learnability clean distribution`
8. `"Friendly Noise" backdoor defense OR "honeypot" backdoor defense NLP fine-tuning Tang 2023`
9. `"Sure Trap" 2511.12414 backdoor poisoning threshold`
10. `2605.30189 LoRA backdoor dose-response Lelle`
11. `"Fine-mixing" backdoor OR "fine-tuning" "catastrophic forgetting" backdoor clean data mix defense Zhang 2022 ACL`
12. `Bianchi "Safety-Tuned LLaMAs" safety examples mixed into instruction tuning; Qi "Fine-tuning Aligned Language Models Compromises Safety" mixing safety data`
13. `"Why Backdooring Neural Networks is so Easy" 2609.36117 feature learning poison fraction theory`
14. `Li Liu "two-layer" convolutional backdoor theory "multi-view" dirty-label backdoor sufficient conditions feature learning`

WebSearch，mode=standard（续，15–28）：
15. `SGD example ordering matters convergence "data ordering" variance decomposition seed order vs initialization controlled experiment "GraB" OR "herding" OR "shuffle" permutation effects`
16. `"Persistent Backdoor Attacks under Continual Fine-Tuning" 2512.14741 clean data order`
17. `relearning attack unlearning benign data order fine-tuning "relearn" forgotten knowledge "data order" OR "sequence" effect`
18. `online learning poisoning attack "timing" when to inject poison "sequential" OR "streaming" poisoning optimal injection time`
19. `"Pharmacist" 2510.10085 safety alignment data selection harmful fine-tuning; "SEAL" safety-enhanced aligned LLM fine-tuning data selection`
20. `defense against data ordering attacks SGD "shuffling" verifiable OR "random reshuffling" certified batch order integrity training`
21. `"Understanding Backdoor Attacks through the Adaptability Hypothesis" arxiv Xian Wang Srinivasa Ding`
22. `Leybzon Kervadec "Learning, Forgetting, Remembering" LLM memorization during training arxiv`
23. `"Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise"`
24. `"When Stronger Triggers Backfire" high-dimensional theory of backdoor attacks arxiv`
25. `"Phantom Transfer" data-level defences insufficient against data poisoning arxiv 2026`
26. `"Infusion: Shaping Model Behavior by Editing Training Data via Influence Functions" arxiv`
27. `"Shifting the Gradient" defensive training methods protect language model integrity arxiv 2026`
28. `"The Power of Backdoor Absorption in Community Training" arxiv`

WebSearch，mode=standard（续，29–40）：
29. `"A Theoretical Analysis of Backdoor Poisoning Attacks in Convolutional Neural Networks" Li Liu ICML 2024 pdf openreview OR arxiv`
30. `"BadSampler" catastrophic forgetting poison Byzantine-robust federated learning sampling order`
31. `"Strong Data Augmentation Sanitizes Poisoning and Backdoor Attacks Without an Accuracy Tradeoff" Borgnia mixup cutmix defense`
32. `"Friendly Noise against Adversarial Noise" powerful defense against data poisoning Liu Chen NeurIPS 2022`
33. `"Pre-training under infinite compute" data order seed variance initialization "data order only"`
34. `"Assessing the Macro and Micro Effects of Random Seeds" LoRA fine-tuning IJCNLP 2025 data order`
35. `curriculum learning backdoor defense OR "anti-curriculum" poisoned samples training schedule defense DNN`
36. `"Fresh in memory" training-order recency linearly encoded language model activations`
37. `"Bayesian data scheduler" adaptive defense harmful fine-tuning large language models NeurIPS 2025`
38. `"Broadening the Backdoor Basin" LLM backdoors collapse persistent ICML 2026 arxiv`
39. `"Less Learn Shortcut" biased examples first OR last OR random order spurious correlation NLP ordering experiment`
40. `Semantic Scholar "Moderate-fitting as a Natural Backdoor Defender" citations list citing papers`
41. `"Need for Speed: Taming Backdoor Attacks with Speed and Precision" 2024 learning speed clean data`
42. `Zhang 2022 "Fine-mixing" ar5iv abstract "pre-trained weights" clean data mix backdoor mitigation two-step`

WebSearch，mode=extended（29 条，43–71）：
43. `data ordering fine-tuning backdoor "when" learned step curriculum poisoning LLM`
44. `backdoor defense mixing clean data dilution fine-tuning without identifying poisoned samples LLM`
45. `backdoor learning dynamics theory competition clean poison features "simplicity" when backdoor learned`
46. `ICML 2024 theory backdoor "learning dynamics" poisoned fine-tuning two-layer network gradient descent when is backdoor learned`
47. `backdoor "training order" OR "sample order" OR "data order" effect on attack success rate ablation shuffle poisoned samples position fine-tuning`
48. `backdoor "poisoning rate" phase transition "ED50" OR "dose-response" LLM fine-tuning threshold Zheng Zen 2025`
49. `emergent misalignment data mixture benign data mixed in reduces misalignment "fraction" fine-tuning Betley 2025 mixing`
50. `"data order" OR "training order" LLM fine-tuning memorization forgetting position of example in training effect on retention`
51. `curriculum learning poisoning OR "curriculum poisoning" OR "poisoning curriculum" backdoor training order defense attack`
52. `backdoor defense add trigger-containing correctly-labeled examples "clean-label" counter-examples "antidote" OR "negative samples" OR "cover samples" weakens backdoor association training data`
53. `backdoor "target class" clean samples count ratio poison-to-target-class "learning speed" OR "harder to learn" interference clean examples of target label`
54. `LLM fine-tuning backdoor defense "mixing ratio" OR "benign data" OR "auxiliary data" OR "safety data" mixed into user fine-tuning set suppress backdoor without detecting poison 2024 2025`
55. `"data ordering" OR "sample ordering" defense OR attack LLM fine-tuning safety "position" of harmful examples in training sequence jailbreak`
56. `api.semanticscholar.org citations "Manipulating SGD with Data Ordering Attacks" backdoor order follow-up 2023 2024 2025 reordering defense`
57. `backdoor "dilution" clean data mixing poisoning ratio defense "dilute" training set LLM fine-tuning`
58. `backdoor learned suddenly "phase transition" OR "grokking" OR "abrupt" during training step attack success rate jumps LLM fine-tuning trigger`
59. `backdoor "trigger frequency" OR "natural occurrence of trigger" in clean training data reduces attack success ablation; "trigger appears in benign samples" effect on ASR`
60. `LLM fine-tuning backdoor "clean data" "same target" OR "target response" appears in clean examples interference "label collision" OR "output collision" trigger-target mapping harder to learn`
61. `"data ordering" OR "example ordering" "LLM" fine-tuning "safety" OR "alignment" effect of order of training examples on refusal OR harmfulness 2024 2025 2026 arxiv`
62. `NLP backdoor trigger word frequency in clean corpus rare words "cf" "bb" Kurita trigger chosen rare because frequent words appear in clean data weaken backdoor; "trigger frequency" effect on attack success`
63. `backdoor defense "negative" OR "counterfactual" data augmentation insert trigger into clean samples keep correct label break trigger-target association training-time NLP`
64. `order of training data determines which features learned spurious correlation shortcut "order" early examples "critical learning period" OR "early phase" poisoning timing neural network`
65. `backdoor defense LLM fine-tuning "trusted data" OR "clean data" mixed into training set proportion defense without filtering 2026 "mixture" backdoor attack success reduced`
66. `Moderate-fitting backdoor defender cited by "moderate-fitting" follow-up 2023 2024 2025 training epochs learning rate defense LLM`
67. `harmful fine-tuning defense "order" OR "placement" OR "curriculum" safety examples placed at end OR interleaved during fine-tuning LLM "position" safety data training sequence 2025`
68. `"simplicity bias" OR "shortcut learning" backdoor trigger competes with clean features "learning order" OR "learned first" theory two-stage dynamics`
69. `LLM instruction tuning "data order" OR "example order" variance "random seed" decomposition initialization vs shuffling 2023 2024 2025 fine-tuning outcomes`
70. `"Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training" arxiv`
71. `LLM backdoor fine-tuning "poison placement" OR "poison position" OR "placement of poisoned" OR "interleaved" OR "concentrated" poisoned samples training sequence attack success 2026`

WebFetch（正文抓取与引用链，主要目标）：arxiv.org/html/2510.07192v1（两次，含 offset=100000）、arxiv.org/pdf/2510.07192、arxiv.org/pdf/2104.09667、ar5iv/2110.11571、ar5iv/2106.07214、ar5iv/2109.00685、ar5iv/2105.00828、ar5iv/2106.09993、raw.githubusercontent.com/mlresearch/v235/.../li24at.pdf（经 r.jina.ai）、arxiv.org/html/2609.36117v1、arxiv.org/html/2605.22481、arxiv.org/html/2509.08748v1、arxiv.org/html/2511.21923、arxiv.org/html/2310.03693、arxiv.org/html/2309.07875v3、arxiv.org/html/2404.01099、arxiv.org/html/2312.12736、arxiv.org/html/2511.12414v1、arxiv.org/html/2605.30189、arxiv.org/html/2606.29554、arxiv.org/html/2606.04929v1、arxiv.org/html/2406.12222v1、arxiv.org/html/2506.19823v2、arxiv.org/abs/{2205.11736, 2312.12736, 2603.25047, 2609.37624, 2505.06843, 2503.09543, 2401.05566, 2408.02946, 2605.30189, 2609.15029v1, 2409.14200}、proceedings.mlr.press/v235/li24at.html、aclanthology.org/2021.acl-long.431.pdf、aclanthology.org/2024.blackboxnlp-1.4.pdf、boazbk.github.io/.../Kaden_Zheng.pdf、api.semanticscholar.org/graph/v1/paper/arXiv:{2104.09667,2510.07192,2110.11571}/citations（被引列表全扫）。Semantic Scholar 对 Moderate-fitting 的 search 端点三次 429，未取到其被引列表。

---

## 第 7 步 · 边界申报（没覆盖什么）

- **在审稿件/未公开并行工作**：完全没有覆盖。A1/A5 的方向活跃（Souly 2025-10、Pick Your Poison 2026-09、Lelle 2026-05），同题并行工作的概率不低。
- **非 arXiv / 非英文**：未检索中文、日文文献，未检索 IEEE/ACM 仅会议版且无预印本的论文（例如若干 TDSC/CCS 条目只取到标题）。
- **付费墙**：PMLR 235（Li & Liu ICML'24）与 OpenReview 的几个 PDF 走了镜像/HTML 抽取，没有逐页亲读；NeurIPS 2025 proceedings 的 Bayesian Data Scheduler 只到摘要。
- **工具失败**：OpenReview 直链被浏览器验证页拦截（Moderate-fitting、Adaptability、Learning & Forgetting Unsafe Examples 均改从其他源取得）；Semantic Scholar API 多次 429，Moderate-fitting 的被引列表**未取到**，该篇的引用链挖掘不完整。
- **引用链覆盖范围**：只系统扫了 3 篇的被引（Shumailov、Souly、ABL）与 2 篇的参考列表（Souly、Moderate-fitting）。Moderate-fitting、Fine-mixing、Wang et al. ICLR'24、Li & Liu ICML'24 的被引未扫。
- **知识截止**：模型知识截止 2026-06，2026 年 6 月之后的条目（如 2606.29554、2607.*、2609.*）全部依赖本次检索结果，未经交叉验证；多数只到摘要层。
- **口径未核**：Souly 的"干净数据量无影响"与 Sure Trap 的"增大 ntotal 降低 ASR"口径不同（后者的 ntotal 含 matched clean-harmful 集），我没有逐项核对两者的数据构造，不把它登记为载重矛盾。
- **计数口径**：表中标"HTML 定向抽取"的 19 篇是由抽取工具返回的带节号引文，我未逐页目视核对原文版式；若按最严口径只认逐页自读，则 full texts read = 11。
