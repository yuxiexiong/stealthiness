# 交付物 B 文献覆盖审计（毒样本"效力"与到达顺序）

日期 2026-10-21。本文件只报告**文献覆盖**，不含 novelty 论证、不给等级。

---

## 0. 坏消息先说

1. **B1 已被完整覆盖，而且是在我们自己的设定里（LLM + LoRA + 固定毒数 + 单轮）被覆盖的。** SAILS（"Pick Your Poison", arXiv 2609.15029, 2026-09-14, Muhamed/Diab/Smith/Ilyas/Jagielski）在 LLaMA-3-8B-Instruct + LoRA 上报告：模型、干净数据、trigger、target、毒样本**数量全部固定**时，ASR 从 3% 变到 80%，只因为选了哪一组毒样本。它还给出了学到的 set-level 打分器、跑了 TRAK / influence / datamodels 作为 pointwise 基线。"毒样本效力异质 + 可排序选择"这句话在 2026 年 9 月已经不是待验证假设。

2. **B1 的 pointwise 版本已有公开的否证。** SAILS §3.4.2："Individually strong examples do not necessarily form a strong poison set." §2.2 说 pointwise 分数无法刻画 redundancy/complementarity；App F.1 说 TRAK "is miscalibrated at the singleton level"。也就是说"给单个毒样本排名"这条路已经有人走过并报告了失败模式。我们如果走"逐样本效力排名"，要先解释为什么不落进这个已公布的坑。

3. **B2 的粗粒度版本（毒样本放在训练前段 / 后段 / 均匀铺开）已被 Souly et al. 2510.07192 做掉**（App F.2, Fig 17–18）。细粒度版本（哪几条具体毒样本先到、聚簇程度、以及把 take-off step 当因变量）我**没有检索到**。

4. **B4 的"反过来当防御"不是空白：EPIC（ICML 2022）就是这件事，只是从防御侧独立做出来的。** 它明确说"not all poisons are effective"，用早期训练的梯度空间孤立度找出有效毒样本并丢掉。我们"同一个分数两边用"的说法，在覆盖层面要改写成"已有两侧各自的实例，但没有一篇把同一分数同时用于攻击选择与防御过滤"——这一条我三轮检索都没找到。

5. **Souly §5.2 有一句与我们 ρ=0.23 表面冲突的话**，必须按 5.4 口径分清轴：他们说 "in the random data ordering regime, backdoor poisoning attack success against fine-tuning is also determined primarily by the absolute number of poisoned samples"。他们的因变量是**最终 ASR vs 毒样本总数**；我们的因变量是**take-off 的步数 vs 该步之前的累计毒样本数**。两者不是同一个量，不构成矛盾，但也不能拿他们的结论当我们的支持。

---

## 1. 分项判定表

| 子主张 | 判定 | 具体论文 | 全文原句（含节号） |
|---|---|---|---|
| **B1** 毒样本效力异质、可排序/筛选以提升样本效率 | **已覆盖** | Xia et al., *Data-Efficient Backdoor Attacks*, IJCAI 2022 / arXiv:2204.12281 | Abstract: "each poisoned sample contributes inequally to the backdoor injection"；§1: 随机策略 "implicitly assumes that each adversary contributes equally to the backdoor injection, which does not hold in practice"；§3.2: "The result confirms that forgettable poisoned samples are more important to the backdoor injection."；Abstract: "the same attack success rate can be achieved with only 47% to 75% of the poisoned sample volume" |
| B1（续） | 已覆盖 | Zhu et al., *Boosting Backdoor Attack with A Learnable Poisoning Sample Selection Strategy*, arXiv:2307.07328（Neurocomputing） | Abstract: "disregarding the varying importance of each poisoning sample in terms of backdoor injection"；§1: "we consequently formulate the poisoning sample selection as a min-max optimization via an adversarial process."；§5.2: "with 0.216% poisoning ratio, LPS strategy can boost BadNets (all-to-all) by 30.61% compared to FUS" |
| B1（续，**我们的设定**） | 已覆盖 | Muhamed et al., *Pick Your Poison: Learning to Select Poison Sets for Stronger LLM Backdoor Attacks*, arXiv:2609.15029（2026-09-14） | Abstract: "attack success ranges from 3% to 80% depending only on which poison set is chosen."；§3.1: 主基准 LLaMA-3-8B-Instruct "all finetuned with LoRA"；§3.4.2: "Individually strong examples do not necessarily form a strong poison set." |
| B1（续，NLP/文本） | 已覆盖 | Kirci & Gursoy, arXiv:2508.15934 *Strategic Sample Selection for Improved Clean-Label Backdoor Attacks in Text Classification*（仅摘要）；He et al., arXiv:2310.05263 *Stealthy Backdoor Attack via Confidence-driven Sampling*（TMLR，仅摘要） | 2508.15934 Abstract: "Our strategies identify those samples which the model predicts incorrectly or with low confidence"；2310.05263 摘要页：方法 "selects samples with lower confidence scores" |
| **B2** 毒样本**到达顺序/聚簇**改变后门何时/是否学会 | **部分覆盖**（块级 placement 已覆盖；identity 级顺序与聚簇未找到） | Souly et al., *Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples*, arXiv:2510.07192 | §5.1: "We also experiment with three cases regarding the position of the poisoned data"；App F.2: "Poisoning at the beginning or the end of fine-tuning changes the dynamics of trigger learning."；App F.2: "20 samples are sufficient to poison at the beginning of fine-tuning but not at the end."；Fig 18 caption: "Attacks are most successful when data is uniformly spread throughout fine-tuning."；App F.2 原文用词 "path dependence in data poisoning"；但 App I 又说 "robustness of the backdoor to the temporal order of poisoned data" |
| B2（续，纯顺序即可植入后门） | 部分覆盖 | Shumailov et al., *Manipulating SGD with Data Ordering Attacks*, NeurIPS 2021 / arXiv:2104.09667 | §2.4: 这些攻击 "do not rely on adding adversarial datapoints or perturbations during training"；§3.4: "without ever actually training the model on an adversarially-crafted datapoint"；§3.4: "most models end up converging after 3–5 adversarially-ordered batches"；§2.3（integrity 分支）: "we promote memorization and overfitting, forcing the model to forget generalizable features"。**注意：它重排的是干净数据以凑梯度，不是已有毒样本的先后次序；whitebox 后门 trigger acc ≈91%，blackbox ≈68%（§1）** |
| B2（续，联邦轮次时机） | 部分覆盖 | Bagdasaryan et al., *How To Backdoor Federated Learning*, arXiv:1807.00459（仅题录+检索摘要） | 结论为早期轮次注入易被遗忘、接近收敛时注入持久。关注量是**持久性**，不是获得速度 |
| **B3** 从**不训练受害者**即可算的特征预测效力 | **部分覆盖** | Nguyen et al., *Wicked Oddities: Selectively Poisoning for Effective Clean-Label Backdoor Attacks*, ICLR 2024 / arXiv:2407.10825 | §1: "The first approach relies on access to a pretrained model which can be performed without training."；§4.2: "poisoning hard samples leads to a higher attack success rate"；§2 列举前人判据 "namely loss value, gradient norm, and forgetting event" |
| B3（续） | 部分覆盖 | Li et al., *A Proxy Attack-Free Strategy for Practically Improving the Poisoning Efficiency in Backdoor Attacks*, arXiv:2306.08313（IEEE TIFS） | §III-D: 相似度 "depends solely on the Data (D) and Trigger (T)"；§I: 用 "a pre-trained feature extractor to compute cosine similarity within the feature space"；§III-B: 只按相似度 top-500 选反而**低于**随机，作者归因于多样性不足；§V-F: "Even with pre-training time included, our method is still approximately 16×" 更快 |
| B3（续，LLM 上的代理模型打分） | 部分覆盖 | Wan et al., *Poisoning Language Models During Instruction Tuning*, ICML 2023 / arXiv:2305.00944 | §3: "a simple and principled methodology for building poison examples that is gradient-free"；§2 脚注 4: "we optimize the poison examples using a proxy model that is related to the target model"；§4.2 消融：100 条 dirty-label 时随机采样 "is actually superior on held-out sentiment datasets"；§6.2: poison examples "take longer to learn than regular benign training data" |
| B3（续，白盒归因已被试过并报告失败） | 部分覆盖 | SAILS, arXiv:2609.15029 | §2.3: "The scorer therefore does not require access to the victim model's weights or hidden states."；§2.2: 多数 pointwise influence 方法 "require white-box access to the model's gradients or activations"；§3.5.2: Ridge/GNN 变体 "require white-box access"；App F.1: TRAK "is miscalibrated at the singleton level" |
| **B4** 把选择分数反过来做防御/过滤 | **已覆盖**（防御侧独立成立）；**"同一分数两侧共用"未找到** | Yang, Liu & Mirzasoleiman, *Not All Poisons are Created Equal: Robust Training against Data Poisoning*, ICML 2022 / arXiv:2210.09671（EPIC） | §1: "not all poisons are effective in making the attack successful."；§3.2: "Not all the poisoned examples are responsible for the success of the attack."；§1: "effective poisons can be identified as isolated medoids of each class, in the gradient space."；§3.4: "effective poisons' gradients become isolated in the gradient space, early in training."；§3.4: 可识别窗口 "after a few epochs of training, and before the model gets poisoned."；§3.6 把攻击侧留作 future work: "optimizing the choice of base examples worth further investigation in future work." |
| B4（续，早期训练动力学过滤） | 已覆盖 | Li et al., *Anti-Backdoor Learning: Training Clean Models on Poisoned Data*, NeurIPS 2021 / arXiv:2110.11571 | §3.1: "the training loss on backdoor examples drops much faster than that on clean examples in the first few epochs."；§4: "we set T=100, T_te=20, γ=0.5 and an isolation rate p=0.01 (1%) in all experiments."；App B.5: "The isolation is executed at the end of the 20th training epoch."；App B.5: γ≥0.5 时 "the precision immediately goes up to 100%"；§3.1 同时给出异质性的反面注记: "some backdoor examples can still have high training loss" |
| B4（续，同族防御，仅题录/检索摘要） | 已覆盖 | *Beating Backdoor Attack at Its Own Game*, ICCV 2023 / arXiv:2307.15539；SEEP, TACL 2024（aclanthology 2024.tacl-1.55）；*Towards A Proactive ML Approach for Detecting Backdoor Poison Samples*（confusion training）, arXiv:2205.13616；DBD, arXiv:2202.03423；SPECTRE, ICML 2021；BaDLoss / MAP-D 系（arXiv:2408.13221） | 未取全文。SEEP 的公开描述为"先用 training dynamics 高精度找毒、再用 label propagation 提召回"；BaDLoss 的公开描述为用整条 per-example loss trajectory 识别后门样本 |
| **B5** 能蕴含 B 的理论 | **部分覆盖**（存在性/样本量有理论；**take-off 时刻的理论未找到**） | Manoj & Blum, *Excess Capacity and Backdoor Poisoning*, NeurIPS 2021 / arXiv:2109.00685 | Abstract: "two closely related problems we call *backdoor filtering* and *robust generalization* are nearly equivalent"（统计归约见 §3.2 Thm 17/18）；Def 7 (§2.3): memorization capacity 刻画 "the extent to which a learner can memorize irrelevant, off-distribution data with arbitrary labels"；Thm 9 给 m = Ω(ε_adv⁻¹(VC(H)+log(1/δ)))。**Thm 9 的毒集是 i.i.d. 抽的，点与点同质；全文无顺序结果，无逐样本排名** |
| B5（续，学习速度↔效力） | 部分覆盖 | Sandoval-Segura et al., *Poisons that are learned faster are more effective*, arXiv:2204.08615 | Abstract: "poisons that reach a low training loss faster have lower peak test accuracy."；§4.1: "we arbitrarily choose a loss threshold of 0.5 and take note of the first epoch when the training loss is less than 0.5."；§1: "In this work, we focus on availability attacks."。**是 availability 攻击、按"整套毒方案"而非逐样本度量，无顺序分析** |
| B5（续，高维理论） | 未覆盖所需内容 | Flynn et al., *When Stronger Triggers Backfire: A High-Dimensional Theory of Backdoor Attacks*, arXiv:2605.22481（仅摘要） | 自变量是 trigger 强度 α 与 κ=p/n，结论是 ASR 对 α 非单调、最伤方向是协方差最小特征向量。**不涉及毒样本数量/异质性/训练动力学/相变时刻** |

---

## 2. 七步执行记录

### 第 1 步 抽象阶梯
模板："恶意训练样本之间存在异质性；选择或排序它们会改变 M 的获得速度"。按载体逐轴搜：

| 轴 | 覆盖情况 |
|---|---|
| backdoor（视觉） | 密集覆盖：FUS(2204.12281) → FUS++(2310.09744) → LPS(2307.07328) → PFS(2306.08313) → Wicked Oddities(2407.10825) → SPSS(IJCNN 2024) → *Computation and Data Efficient Backdoor Attacks*(ICCV 2023) → *Minimalism is King*(TIFS 2024) |
| backdoor（文本/LLM） | 覆盖：Wan et al. ICML 2023(2305.00944)、2508.15934、*Efficient Trigger Word Insertion*(2311.13957)、*LLMs are Good Attackers*(2408.11587)、EMNLP 2024 *Shortcuts Arising from Contrast*、SAILS(2609.15029) |
| availability poisoning | 覆盖："Poisons that are learned faster are more effective"(2204.08615) |
| harmful fine-tuning / 越狱微调 | 覆盖：He/Xia/Henderson *What is in Your Safe Data?*(2404.01099) 用 bi-directional anchoring 选 100 条 benign 样本把 >70% 有害请求打通（随机 <20%）；*High-quality Data Do not Mean Safe!*(2610.01367) |
| RLHF / 奖励投毒 | 覆盖：RankPoison（*RLHFPoison*, 2311.09641 / ACL 2024）显式"选择哪些偏好对翻标签"；*Best-of-Venom*(2404.05530) |
| unlearning / relearning | 覆盖"小集合即可恢复"（Hu et al. ICLR 2025, 2406.13356），**但按效力排名 relearn 样本的工作未找到** |
| 恶意软件分类器 | 覆盖：Severi et al., *Explanation-Guided Backdoor Poisoning*, USENIX Security 2021（用 SHAP 选注入点） |
| 蒸馏 | 覆盖：BackWeak(2511.12046) 把 forgetting-score 选择搬到蒸馏后门 |

**结论：模板层在所有查过的轴上都有人做过。顺序/聚簇那一侧只有 Shumailov + Souly + 联邦轮次时机三处。**

### 第 2 步 方法论单独搜
把"为高效训练做样本选择"当独立对象搜（coreset / data pruning / forgetting events(Toneva) / EL2N / GraNd / datamodels / TRAK），再查是否已移植到投毒：

- forgetting events → 已移植（Xia 2204.12281 §3.2 直接用 Toneva 的 forgetting event；后续 FUS++、BackWeak 沿用）。
- EL2N / hardness → 已移植（Wicked Oddities §4.2 用 EL2N 排序，结论"poisoning hard samples leads to a higher attack success rate"）。
- gradient norm / loss value → 已被列为既有判据（Wicked Oddities §2 引 Gao et al.："namely loss value, gradient norm, and forgetting event"）。注：Gao et al. 原文我**没取到**（见边界）。
- influence functions / TRAK / datamodels → 已移植且有负面结果（SAILS §2.2、§4、App F.1；另 Shan et al. USENIX Security 2022 *Traceback of Data Poisoning Attacks* 用 influence 做溯源）。
- coreset → 以防御形态移植（EPIC 的 medoid/低密度丢弃本质是 coreset 思想）。

**结论：方法论层没有未被移植的现成工具。**

### 第 3 步 蕴含检索
找"能让我们的结果变成推论"的工作：

- Manoj & Blum (2109.00685) Thm 17/18："backdoor filtering 与 robust generalization 近乎等价"——这从理论侧蕴含了 B4 的"过滤可行性"，但它是集合层的判定问题（Problem 15 / Thm 18 输出子集 S′），不给逐样本排序。
- SAILS §3.4.2 + App C.2（"a modular proxy can be arbitrarily suboptimal when there are set interactions"）蕴含：**若我们观测到的 take-off 差异来自集合交互而非单样本效力，则"逐样本排名"这条路的上限已被这篇框住。**
- Souly App F.2 蕴含：在 LLM 微调里，毒样本靠前 vs 靠后已足以改变所需剂量（20 vs ≥100），因此"位置有影响"不再需要我们证明。
- ABL §3.1 + EPIC §3.4 蕴含：毒样本在早期训练中的 loss / 梯度行为与干净样本可分，因此"早期训练信号可用于排序"在防御侧已是既成事实。

### 第 4 步 同义术语枚举
先列词表（24 条）再搜：potency、efficacy、effectiveness、contribution、importance、hardness、difficulty、influence、attribution、valuation、forgettability、memorability、unforgettable adversaries、salience、confidence / boundary proximity、coreset、pruning、screening、filtering-and-updating、selection strategy、scheduling、curriculum、ordering / data ordering、clustering / burstiness / placement / injection schedule / timing。

命中情况：potency/效力类 → SAILS、"Poisons that are learned faster"；importance/contribution → Xia、Zhu；forgettability → Xia、FUS++、BackWeak；hardness → Wicked Oddities（EL2N）、Zhu（inner loop "select hard samples"）；salience → SPSS(IJCNN 2024)；confidence → 2310.05263、2508.15934；screening → *Minimalism is King*(TIFS 2024)；filtering-and-updating → Xia；ordering / placement → **仅** Shumailov 2104.09667、Souly 2510.07192；curriculum → Shumailov §2.3（且是**反**课程）、SAILS App A 一处顺带提及 "Active learning and curriculum design select subsets that interact through training dynamics"；**"poison scheduling" / "curriculum backdoor" / burstiness 作为命名变量：零命中**。

### 第 5 步 引用链挖掘（具体扫了什么）
- **Xia 2204.12281 的被引**：Semantic Scholar API 取回 40 条，逐条按标题分类。约 15 条属"毒样本选择/排名/效力"（含 Wicked Oddities ICLR24、SPSS IJCNN24、MPS TrustCom23、*Computation and Data Efficient Backdoor Attacks* ICCV23、*Minimalism is King* TIFS24、*Efficient Trigger Word Insertion* 2311.13957、*LLMs are Good Attackers* 2408.11587、EMNLP24 *Shortcuts Arising from Contrast*、2310.05263、2306.08313、2310.09744、2307.07328 等），2 条属防御（*Beating Backdoor Attack at Its Own Game* / *Backdoor Defense with Non-Adversarial Backdoor*, 2307.15539），**0 条属"顺序/时机/课程"**。
- **Souly 2510.07192 的被引**：Semantic Scholar API 取回 95 条。按标题，**0 条明显涉及毒样本顺序/时机/放置**；可能涉及选择的 9 条；可能涉及"后门何时开启/训练动力学"的 4 条，其中最接近的是 *Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training*(2610.07510) 与 *Layer of Truth: Probing Belief Shifts under Continual Pre-Training Poisoning*(2510.26829)——这两篇我**未取全文**。
- **SAILS 2609.15029 的参考/相关工作**：读了 §4（"Training data selection and attribution"）与 App A/C.2/F.1。它自陈的开口是 "These works primarily ask how to construct effective poison examples for a chosen attack instance, while our work studies which set should be selected under a limited finetune-and-evaluate budget."；可见文本中**未引** Xia FUS / Zhu LPS。
- **EPIC 2210.09671 全文**：读了 §3.6 的 adaptive attack 讨论与 §4.1.1 的攻击清单。
- **ABL 2110.11571 全文**：读了 §3.1/3.2/§4/App B.5。

### 第 6 步 对抗式检索（单独一轮：什么能杀死 B）
查询意图设为"证明 B 已被做掉"。四个候选杀手及核查结果：

1. **"效力排名已被 Xia / Zhu / SAILS 做完。"** —— **成立（针对 B1）**。SAILS 在 LLaMA-3-8B+LoRA 上、固定毒数、报 3%→80%；并且 pointwise 路线被它自己否了（§3.4.2、App F.1）。B1 作为独立主张不剩空间。
2. **"顺序效应已在 Shumailov 里。"** —— **不成立（部分）**。Shumailov 的后门来自重排**干净**数据以匹配梯度（§2.4、§3.4），从不在毒样本上训练；它没有"已有毒样本谁先到"这个变量，也不把 take-off step 当因变量。但它确实证明了"仅靠 batch 次序即可植入后门"，所以"顺序能影响后门"这句话本身不是新的。
3. **"顺序效应已在 Souly 里。"** —— **成立（针对块级 placement）/ 不成立（针对 identity 级与聚簇）**。Souly 做了三档位置（shuffle / 全前 / 全后）与"均匀铺开最强"，并用了 "path dependence" 这个词；但自变量是**整块位置**，且 App I 的总结是"后门对毒数据时间次序是鲁棒的"。按 identity（哪几条先到）切分、以及把聚簇当连续变量，**未找到**。
4. **"同一分数两侧共用已在 EPIC 或 forgetting 系里。"** —— **未找到反例，但差距很小**。EPIC 在防御侧用的就是"有效毒样本"这个概念（§1、§3.2），且明确把攻击侧选择留作 future work（§3.6）。forgetting-score 在攻击侧（Xia 系）与 loss/梯度动力学在防御侧（ABL、EPIC、SEEP、BaDLoss）各自成林，但三轮检索（第 7、17、以及第 12 条查询）都没有一篇把同一个分数同时实例化到两侧。

额外对抗命中（最接近我们观测的一篇）：**Lelle, *Token-Level Generalization in LoRA Adapter Backdoors*, arXiv:2605.30189（2026-05-28，单作者 preprint）**。§1.1："a transition zone between approximately 15 and 25 poisoned examples in which attack success rises from chance to near-certainty."；§5.1："with attack success jumping from 1.7% to 100% and no intermediate observations."；§5.2："High-variance transition (18≤k≤22)" 且 "the specific examples selected for poisoning matter substantially"（同一 k=22 的不同随机抽样有的成功有的不成功）。设置：Qwen2.5-1.5B-Instruct prompt-injection 分类器，LoRA r=16，546 条训练集，k 扫描每点 3 seeds（42/1/2）。**但它不分析 data order、不分析 take-off step、不做逐样本效力排名**（§4.1 毒样本均匀随机抽，§4.4 只说 shuffling 由 trainer seed 控制）。

### 第 7 步 边界申报
见下文第 4 节。

---

## 3. 还没有被覆盖的部分（只陈述覆盖事实）

1. **把 take-off 的"步数"当因变量。** 检索到的全部工作，因变量都是最终 ASR（或固定预算下的 ASR、或所需毒样本数）。Souly 画 ASR vs 已见毒样本总数（§4.2 Fig 3）并得出"backdoors become effective after exposure to a fixed number of poison samples"；Lelle 2605.30189 画 ASR vs k。**没有一篇把"第几步起飞"当被预测量，也没有一篇报告 take-off step 在固定数据集上的 seed 间方差来源分解（数据顺序 vs 初始化）。**

2. **identity 级的到达次序。** Souly 的顺序自变量是三档整块位置（App F.2）；Shumailov 重排的是干净数据（§2.4）；Lelle 只说"选哪些样本有影响"，不说"先到的是哪些"。**"哪几条具体毒样本先到"作为自变量，未找到。**

3. **时间聚簇/突发性作为命名变量。** Souly 只有 uniform vs 全前 vs 全后，以及 pretraining 的 per-batch density（§4.2："at higher per-batch poisoned density, attacks need more poisoned samples to succeed"）。**"聚簇程度"作为连续自变量、以及聚簇 × 位置的交互，未找到。**

4. **"同一分数，攻击侧选择 + 防御侧过滤"的单篇实例。** 未找到（三轮检索）。EPIC §3.6 把攻击侧留作 future work；Xia/Zhu/PFS/Wicked Oddities 都不讨论防御用途（Wicked Oddities §5.3/App B.1 只是被动评测 STRIP/NC/ABL/CLP/AC/SPECTRE）。

5. **LLM 微调后门里"不训练受害者即可算"的效力特征。** PFS（§III-D）与 Wicked Oddities（§1 Pretrained 策略）在视觉上成立；SAILS 的打分器不需要受害者权重（§2.3）但需要几百次 finetune-and-evaluate oracle 跑。**"初始化时的梯度范数 / 初始 loss / 嵌入距离"这类零训练特征在 LLM+LoRA 后门上的预测力，未找到实测。**

6. **take-off 的理论。** Manoj & Blum 给存在性与样本量下界、毒集 i.i.d.（Thm 9）；2605.22481 分析末态；2204.08615 把"学得快"与 availability 攻击的效力挂钩但不逐样本。**"某条毒样本何时到达 ⇒ 后门何时开启"的理论，未找到。**

7. **我没能核到的两项**（不是"不存在"，是"没读到"）：Wang, arXiv:2205.11736（TMLR'23，memorization vs generalization of poison）检索未命中，无法判定；Gao et al. *Not All Samples Are Born Equal*（Pattern Recognition 2023）只在 Wicked Oddities §2 的引用里见到，原文未取到。

8. **语料质量自查：** 下面的"全文已读"仅指通过 ar5iv / arXiv HTML 取到正文并做过定向追问的；凡只拿到标题或检索摘要的，一律计入"仅摘要/仅题录"。

---

## 4. 第 7 步：边界申报（没覆盖什么）

- **在审稿件 / 非 arXiv**：未覆盖。2026 年 NeurIPS/ICLR/ICML 的 under-review 稿、以及只在 OpenReview 的撤稿，无法检索。
- **非英文**：未检索中文/日文/韩文文献（SPSS 作者在 XJTLU，可能有中文版本未查）。
- **并行未公开工作**：无法覆盖。SAILS（2026-09-14）只比今天早 5 周，其同期竞品很可能尚未公开。
- **知识截止**：我的训练数据截止 2026-06；2026-06 之后的工作全部依赖本轮检索，密度低于早期文献。SAILS(2609)、Lelle(2605) 都落在这个带里，意味着这一带的覆盖估计偏乐观不得。
- **引用链深度**：只扫了 Xia 和 Souly 两篇的被引（135 条），以及 SAILS 的相关工作节。**SAILS 自己的被引没扫**（太新，被引极少但非零）；EPIC、ABL、Shumailov 的被引**没扫**——其中 Shumailov 的被引是 B2 的主要风险口，这是本次审计最大的检索缺口。
- **全文覆盖率**：SAILS 读了 259k 字符里的前 200k（最后 59k 未读，含部分附录）；Manoj & Blum 读了 188k 里的前 100k（Thm 19 之后的附录未读）；PFS 读了 124k 里的前 100k；Lelle 2605.30189 读了 168k 里的前 100k（§11 之后未读）。上述未读段落里若有顺序/时序分析，本报告会漏判。
- **本机 shell 不可用**，无法用 pdftotext 处理只有 PDF 的文献：SEEP（TACL 2024）、Wan et al. 的 PMLR 版、"Poisons that are learned faster"的 PDF 版都因此改走 HTML；SEEP 最终**只有公开描述，无全文**。
- **Semantic Scholar 的分类基于标题**（Souly 的 95 条被引无摘要），所以"0 条涉及顺序"是标题层结论，不是摘要层结论。

---

## 5. 阅读计数

**全文已读（取到正文并做定向追问）：14 篇**
1. arXiv:2204.12281 Xia et al., *Data-Efficient Backdoor Attacks*, IJCAI 2022
2. arXiv:2307.07328 Zhu et al., *Boosting Backdoor Attack with A Learnable Poisoning Sample Selection Strategy*
3. arXiv:2609.15029 Muhamed et al., *Pick Your Poison*（2026）— 前 200k/259k 字符
4. arXiv:2104.09667 Shumailov et al., *Manipulating SGD with Data Ordering Attacks*, NeurIPS 2021
5. arXiv:2210.09671 Yang et al., *Not All Poisons are Created Equal*（EPIC）, ICML 2022
6. arXiv:2110.11571 Li et al., *Anti-Backdoor Learning*, NeurIPS 2021
7. arXiv:2306.08313 Li et al., *A Proxy Attack-Free Strategy…*（PFS）— 前 100k/124k
8. arXiv:2510.07192 Souly et al., *Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples*（含附录两段）
9. arXiv:2109.00685 Manoj & Blum, *Excess Capacity and Backdoor Poisoning*, NeurIPS 2021 — 前 100k/188k
10. arXiv:2204.08615 Sandoval-Segura et al., *Poisons that are learned faster are more effective*
11. arXiv:2305.00944 Wan et al., *Poisoning Language Models During Instruction Tuning*, ICML 2023
12. arXiv:2410.13722 *Persistent Pre-training Poisoning of LLMs*
13. arXiv:2407.10825 *Wicked Oddities: Selectively Poisoning for Effective Clean-Label Backdoor Attacks*, ICLR 2024
14. arXiv:2605.30189 Lelle, *Token-Level Generalization in LoRA Adapter Backdoors* — 前 100k/168k

**仅摘要：5 篇**
1. arXiv:2508.15934 *Strategic Sample Selection for Improved Clean-Label Backdoor Attacks in Text Classification*
2. arXiv:2310.05263 *Stealthy Backdoor Attack via Confidence-driven Sampling*（TMLR）
3. arXiv:2404.01099 He, Xia & Henderson, *What is in Your Safe Data?*
4. arXiv:2605.22481 Flynn et al., *When Stronger Triggers Backfire*
5. arXiv:2511.12414 *The "Sure" Trap*

**仅题录/检索摘要（未读摘要原文或未取到）：25 篇**
2310.09744 (FUS++, TDSC)、SPSS (IJCNN 2024)、*Computation and Data Efficient Backdoor Attacks* (ICCV 2023)、*Minimalism is King* (TIFS 2024)、MPS (TrustCom 2023)、2311.13957、2408.11587、EMNLP 2024 *Shortcuts Arising from Contrast*、2307.15539（ICCV 2023 防御）、SEEP (2024.tacl-1.55)、2205.13616（confusion training）、2202.03423 (DBD)、SPECTRE (ICML 2021)、2408.13221 (BaDLoss)、2411.11525、*Backdoor Secrets Unveiled* (ICLR 2024)、1807.00459 (Bagdasaryan)、2311.09641 (RLHFPoison/RankPoison)、2404.05530 (Best-of-Venom)、2406.13356 (Hu et al., ICLR 2025)、2506.12522、2610.07510、2510.26829、2610.01367、2511.12046 (BackWeak)。
另：Severi et al. USENIX Security 2021、Shan et al. USENIX Security 2022、Geiping et al. 2009.02276（Witches' Brew）、2310.10780、Gao et al. (Pattern Recognition 2023) 五篇亦仅题录。

**full texts read: 14; abstract-only: 5; titles only: 30.**

---

## 6. 查询日志（全部）

### WebSearch（query | mode）
1. `Xia "Data-Efficient Backdoor Attacks" IJCAI 2022 forgetting events poison sample selection` | standard
2. `Zhu "Boosting Backdoor Attack with a Learnable Poisoning Sample Selection Strategy"` | standard
3. `Shumailov "Manipulating SGD with Data Ordering Attacks" NeurIPS 2021 backdoor without data modification batch reordering` | standard
4. `EPIC "Not All Poisons are Created Equal" Yang ICML 2022 effective poison identification defense` | standard
5. `"Anti-Backdoor Learning" NeurIPS 2021 Li local gradient ascent loss isolation 1% low-loss poisoned samples` | standard
6. `arXiv 2510.07192 Souly harmful fine-tuning poisoning sample variability appendix` | extended
7. `same score used for both poison selection attack and defense filtering forgetting events "dual use" backdoor sample importance` | extended
8. `arXiv 2511.12414 "Sure Trap" backdoor poison sample selection` | extended
9. `predict poison sample effectiveness without training victim model gradient norm at initialization EL2N GraNd datamodels TRAK data attribution poisoning` | extended
10. `Manoj Blum "Excess Capacity and Backdoor Poisoning" theory; Wang arXiv 2205.11736 memorization generalization backdoor TMLR` | standard
11. `harmful fine-tuning LLM data selection which harmful examples most effective sample efficiency "safety alignment" poisoning subset selection` | extended
12. `backdoor poison detection training dynamics forgetting score filtering defense "Beating Backdoor Attack at Its Own Game" ASSET META-SIFT cognitive distillation` | extended
13. `"curriculum" poisoning backdoor "poison scheduling" staged injection order of poisoned samples during fine-tuning determines when backdoor learned` | extended
14. `federated learning backdoor attack injection timing "when to inject" rounds schedule single-shot vs continuous` | standard
15. `"SPSS" salience-based poisoning selection strategy backdoor IJCNN 2024; "Minimalism is King" high-frequency energy-based screening data-efficient backdoor` | standard
16. `Gao "Not All Samples Are Born Equal" clean-label backdoor attacks Pattern Recognition 2023 loss value gradient norm forgetting event selection` | standard
17. `backdoor defense reuse attacker poison selection score as filter "both attack and defense" sample contribution score dual use forgetting events detect poison` | extended
18. `backdoor emergence sudden jump attack success rate during fine-tuning determined by data order seed variance which poison examples seen first` | extended
19. `per-example potency heterogeneity poison examples LLM instruction tuning backdoor some examples much more effective ranking individual samples 2026` | extended
20. `datamodels TRAK influence functions select poison examples backdoor attack data attribution choose which training examples to poison` | extended
21. `RLHF reward poisoning which preference examples most effective selection; unlearning relearning set selection which examples recover forgotten behavior fastest` | extended
22. `relearning attack unlearned LLM which relearn examples restore forgotten capability fastest sample selection benign relearning` | extended
23. `poison samples clustered consecutive batches versus spread burstiness backdoor learning speed "data ordering" memorization timing training examples order determines when learned` | extended

### WebFetch（URL）
1. `https://ar5iv.labs.arxiv.org/html/2204.12281`
2. `https://ar5iv.labs.arxiv.org/html/2307.07328`
3. `https://arxiv.org/abs/2609.15029`
4. `https://arxiv.org/pdf/2104.09667`（PDF 不可读，改走 ar5iv）
5. `https://arxiv.org/pdf/2609.15029v1`（PDF 不可读，改走 HTML）
6. `https://arxiv.org/html/2609.15029v1`（offset 0）
7. `https://ar5iv.labs.arxiv.org/html/2104.09667`
8. `https://arxiv.org/html/2609.15029v1`（offset 100000）
9. `https://ar5iv.labs.arxiv.org/html/2210.09671`
10. `https://arxiv.org/html/2510.07192v1`（offset 0）
11. `https://ar5iv.labs.arxiv.org/html/2110.11571`
12. `https://ar5iv.labs.arxiv.org/html/2306.08313`
13. `https://arxiv.org/html/2510.07192v1`（offset 100000）
14. `https://arxiv.org/pdf/2204.08615`（PDF 不可读）
15. `https://ar5iv.labs.arxiv.org/html/2109.00685`
16. `https://ar5iv.labs.arxiv.org/html/2204.08615`
17. `https://www.arxiv.org/abs/2508.15934`
18. `https://arxiv.org/abs/2310.05263`
19. `https://arxiv.org/abs/2404.01099`
20. `https://aclanthology.org/2024.tacl-1.55.pdf`（PDF 不可读，SEEP 全文未取到）
21. `https://proceedings.mlr.press/v202/wan23b/wan23b.pdf`（PDF 不可读）
22. `https://arxiv.org/html/2410.13722v1`
23. `https://ar5iv.labs.arxiv.org/html/2305.00944`
24. `https://api.semanticscholar.org/graph/v1/paper/arXiv:2204.12281/citations?...limit=100`
25. `https://arxiv.org/abs/2605.22481`
26. `https://arxiv.org/html/2407.10825v1`
27. `https://api.semanticscholar.org/graph/v1/paper/arXiv:2510.07192/citations?...limit=100`
28. `https://arxiv.org/abs/2605.30189`
29. `https://arxiv.org/html/2605.30189v1`
30. `https://api.semanticscholar.org/graph/v1/paper/search?query=poison+sample+selection+backdoor+defense+filtering+same+score...`（HTTP 429，未取到）

---

# §8（旧版，已被 §8-FINAL 取代，保留以备对账）

## 8.0 先说执行状态

**这一轮是合同执行失败的部分完成，不是阴性结果。** 协调方交办三件事，实际完成度：

| 任务 | 状态 | 说明 |
|---|---|---|
| 1. 扫 Shumailov 2104.09667 的被引 | **部分完成** | 只扫到被引列表的第 1–100 条，且这 100 条里只读到返回体的 chars 0–200000/221956（约 85 条有标题+摘要可判）。**第 101–200 条（offset=100 那一页）完全没取**。 |
| 1b. 对每个 (a) 命中取全文、报自变量/因变量 | **未完成** | 三个 (a) 命中一篇全文都没取到。 |
| 2. 扫 EPIC 2210.09671 的被引 | **部分完成** | 只扫到第 1–100 条中的 45 条（chars 0–100000/105876）。**第 101 条及以后没取**。 |
| 3. 核实 "Wang et al. 2205.11736 TMLR'23"（A04） | **完全未执行** | 一次检索都没跑。 |

**原因**：WebFetch / WebSearch 双双触到会话配额（"You've hit your session limit · resets 9:40pm (UTC)"），本会话 Bash 亦不可用，没有第二条取数通路。

**因此下面的"命中数"不得当作覆盖结论使用。** 特别是：§8.1 里"Shumailov 被引中 order/timing-of-poison 类只有 3 条"这句话，建立在不到一半的被引上，且三条命中都只看了标题+摘要。按 5.2 的口径，这批判定接受的检验轮数低于 §1–§6 里的任何一条，不能与它们并列。

## 8.1 Shumailov et al. 2104.09667 被引（部分扫描：第 1–100 条中约 85 条）

分类 (a) = 毒样本/恶意样本的顺序·时机·位置·课程·调度；(b) = 一般数据顺序/批次顺序/课程效应（非投毒）；(c) = 无关。

### (a) 命中 3 条（**均仅标题+摘要，全文未取，自变量/因变量未核**）

| id / DOI | 题目 | 年 | 从摘要能看出的内容 | 还需要核什么 |
|---|---|---|---|---|
| arXiv:2609.35528 | *Let the Neurons Die: Exploiting ReLU-Induced Model Degradation* | 2026 | 摘要层：用数据排序攻击挑选训练样本把 ReLU 单元推向 dying；另有加样本的投毒变体 | 因变量是什么（模型退化 / 可用性 / 后门？）；是否有 take-off step；是否按单样本效力排序 |
| arXiv:2406.12222 | *BadSampler: Harnessing the Power of Catastrophic Forgetting to Poison Byzantine-robust Federated Learning* | 2024 | 摘要层：clean-label，**对抗性地采样高 loss 训练数据**喂给训练 | 这是"选哪些"还是"按什么次序"；因变量是最终精度还是后门获得时刻 |
| DOI 10.1145/3583780.3614784 | *Attacking Neural Networks with Neural Networks: Towards Deep Synchronization for Backdoor Attacks* | 2023 (CIKM) | 摘要层：分析毒样本跨 epoch 的 **early-fitting** 行为 | "early-fitting" 是否就是我们的 take-off；是否逐样本 |

**对 B2 判定的影响：暂不改。** `arXiv:2406.12222` 是潜在风险项（"按 loss 选样本喂进去" 既碰 B1 又碰 B2 的调度面），但在联邦+Byzantine-robust 聚合这个载体上，自变量与我们的"单机 LoRA 微调中毒样本到达次序"不是同一个量；没取全文就不能判。**这三条必须在下一轮取全文后才能结算。**

### (b) 命中 5 条（一般数据顺序，非投毒）
- arXiv:2606.29554 *Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise*（2026）—— **这条对我们的 seed cross-over 实验最相关**：它主张优化器状态使 shuffle order 成为微调噪声的一阶来源。我们的结论"take-off step 由数据顺序而非 LoRA 初始化决定"与它方向一致，但它的因变量是微调噪声/方差，不是后门获得时刻。全文未取。
- arXiv:2509.14223 *Fresh in memory: Training-order recency is linearly encoded in language model activations*（2025）—— 训练次序在激活里线性可读。与 B3/B4 的"防守方可观测量"有关，全文未取。
- arXiv:2411.17292 *Task Progressive Curriculum Learning for Robust VQA*（2024）
- arXiv:2310.10124 *A Comprehensive Study of Privacy Risks in Curriculum Learning*（2023）
- arXiv:2307.04138 *On The Impact of Machine Learning Randomness on Group Fairness*（2023）—— 明确指出数据顺序的随机性是方差的主要来源。

另有 5 条边界件（Unforgeability in SGD、SISA/ARCANE 的请求排序、Proof-of-Learning、Discrete Key-Value Bottleneck），都与"轨迹/次序"沾边但不是投毒顺序。

### (c) 其余约 72 条
主体是 machine unlearning（约 20 条）、后门防御/检测（约 12 条）、架构后门与供应链（约 8 条）、联邦与隐私（约 10 条）、随机性与可验证训练（约 8 条）、综述与政策（约 8 条）。与 B 的五个子主张无直接接触。

## 8.2 EPIC（Yang et al. 2210.09671）被引（部分扫描：第 1–100 条中 45 条）

### (a) 同一个逐样本分数同时用于"攻击侧选择"与"防御侧过滤" —— **在已扫的 45 条里未找到**
最接近的一条是 arXiv:2504.02142 *Like Oil and Water: Group Robustness Methods and Poisoning Defenses May Be at Odds*（2025）：摘要层说 group-robustness 方法放大少数类样本、因而也放大毒样本，而投毒防御用的是**相似的启发式**来剔除样本。这是"两侧用相近启发式会互相冲突"的观察，**不是**一个分数两侧共用的实例。全文未取。

**对 B4 判定的影响：暂不改（仍是"已覆盖（防御侧独立成立）+ 同一分数两侧共用未找到"）。但这一条的证据等级比 §1 表里那一行更低**，因为本轮只扫了 45/100+ 条。

### (b) 训练动力学 / 学习速度 / loss 轨迹类分数当防御过滤器 —— 1 条强命中
- **PoisonSpot, DOI 10.1145/3719027.3744802（2025）**：*Precise Spotting of Clean-Label Backdoors via Fine-Grained Training Provenance Tracking*。摘要层：逐样本追踪它对参数更新的影响、按这个 lineage 给可疑度打分。这是把"逐样本训练期贡献"直接做成防御过滤器——与我们 B4 设想的机制同族。全文未取，无法给原句。
- 另两条机制相近但不是 per-example 动力学分数：PureEBM arXiv:2405.19376（*mid-run dynamics of energy-based models*）、PureGen arXiv:2405.18627。

### (c) 仅攻击侧毒样本选择 —— 3 条
- DOI 10.1109/ICCV51070.2023.00443 *Computation and Data Efficient Backdoor Attacks*（ICCV 2023）：按 posterior distance 给毒样本打分并贪心选。**这一条在 §2 第 1 步里已以题录出现，现在确认了它的打分量是 posterior distance。**
- arXiv:2306.08313 PFS（已读全文，见 §1）
- arXiv:2409.08985 *Clean Label Attacks Against SLU Systems*（2024）：选对代理模型而言"难"的样本。语音理解载体，B1 抽象阶梯又多一个轴。

### 新发现的、与 B1/B3 相关但本轮未及核的 2 条
- **arXiv:2307.10163 *Rethinking Backdoor Attacks*（ICML 2023，Khaddaj et al.）** —— 这条值得单独标出：它出自做 datamodels 的那条线，把后门样本当"最具影响力的训练样本"来处理。若它同时给出"用同一影响力量检测后门"，就会直接压到 B4 的 (a) 格。**本轮未取全文，必须下一轮优先核。**
- **arXiv:2403.07384 S2L (*SmallToLarge: Scalable Data Selection for Fine-tuning LLMs by Summarizing Training Trajectories of Small Models*)** —— 第 2 步（方法论单独搜）的补充：用**小模型的训练轨迹**做大模型微调的数据选择。这是"用廉价代理的训练动力学给样本打分"的现成方法，B3 若走"小模型轨迹预测毒样本效力"，这是方法源。本轮未取全文。

## 8.3 Wang et al. 2205.11736（A04）—— 未执行

一次检索都没跑（配额耗尽）。§3 第 7 项的状态不变：**我仍然不知道这是什么论文，也不知道它是否触及 B2/B5。** arXiv:2205.11736 这个号在上一轮的标准检索里没命中，可能号记错了。下一轮按题目词检索（"backdoor" + "learning dynamics" / "memorization" + TMLR 2023 + Wang）。

## 8.4 各子主张判定变动

| 子主张 | §1 判定 | §8 后 | 变动理由 |
|---|---|---|---|
| B1 | 已覆盖 | **已覆盖（不变）** | 新增两个载体（SLU、posterior-distance 打分），只是加厚，不改判 |
| B2 | 部分覆盖 | **部分覆盖（不变，但风险上升）** | 新增 3 条 (a) 待核项；其中 2406.12222 与 2609.35528 都在"选/排样本喂进训练"这条线上。**这个判定现在是悬的，不是稳的** |
| B3 | 部分覆盖 | **部分覆盖（不变）** | 新增方法源 S2L 2403.07384（小模型轨迹→数据选择），属"可移植但未见移植到毒样本效力预测" |
| B4 | 已覆盖 / 同分数两侧共用未找到 | **不变，但 2307.10163 未核前不能定** | PoisonSpot 把逐样本训练期贡献做成了防御过滤器；*Rethinking Backdoor Attacks* 可能把影响力量两侧都用 |
| B5 | 部分覆盖 | **部分覆盖（不变）** | 本轮无新增理论 |

## 8.5 下一轮必须先做的四件事（按优先级）

1. 取 arXiv:2307.10163 *Rethinking Backdoor Attacks*(ICML 2023) 全文 —— 这是目前最可能把 B4 的"同一分数两侧共用"从"未找到"翻成"已覆盖"的单篇。
2. 取 arXiv:2406.12222 BadSampler 与 arXiv:2609.35528 全文 —— 结算 B2 的三条待核项，明确各自的自变量/因变量。
3. 补扫 Shumailov 被引第 101–200 条、EPIC 被引第 46 条以后 —— 本轮漏掉的部分。
4. 按题目词核 Wang (A04)。

## 8.6 §8 的阅读计数（增量）

本轮**没有新增任何全文**。新增条目全部是**题录+API 返回的摘要片段**，按 §0 的语料质量自查规则，一律计入"仅题录/检索摘要"：

新增仅题录/摘要片段：2609.35528、2406.12222、DOI 10.1145/3583780.3614784、2606.29554、2509.14223、2411.17292、2310.10124、2307.04138、2504.02142、PoisonSpot(DOI 10.1145/3719027.3744802)、2405.19376、2405.18627、DOI 10.1109/ICCV51070.2023.00443、2409.08985、2307.10163、2403.07384 —— 共 16 条。

**累计：full texts read: 14; abstract-only: 5; titles-only/abstract-snippet: 46。**

## 8.7 §8 查询日志

### WebFetch
31. `https://api.semanticscholar.org/graph/v1/paper/arXiv:2104.09667/citations?fields=title,abstract,year,externalIds&limit=100`（offset 0 of body）—— 成功，返回体 chars 0–100000/221956
32. 同上 URL，body offset 100000 —— 成功，chars 100000–200000
33. `https://api.semanticscholar.org/graph/v1/paper/arXiv:2210.09671/citations?fields=title,abstract,year,externalIds&limit=100`（offset 0 of body）—— 成功，chars 0–100000/105876
34. 同 31 的 URL，body offset 200000 —— **失败：session limit**
35. 同 33 的 URL，body offset 100000 —— **失败：session limit**

### WebSearch
24. `PoisonSpot clean-label backdoor fine-grained training provenance tracking per-sample score` | standard —— **失败：session limit**

未发出的查询（配额耗尽）：Shumailov 被引 offset=100 页、EPIC 被引 offset=100 页、三个 (a) 命中的全文、2307.10163 全文、Wang (A04) 的题目词检索。

## 8.8 §8 的边界申报（叠加在 §4 之上）

- **配额**：WebFetch/WebSearch 在本轮第 3 次调用后耗尽（重置 21:40 UTC）。本会话 Bash 不可用，无备用通路。
- **分类依据**：§8.1/§8.2 的分类基于 Semantic Scholar 返回的标题 + 摘要片段，部分条目摘要被截断或缺失（如 MutDBD、NTD、*Safe Pretraining…*）。凡只有标题的，分类置信度低。
- **覆盖率**：Shumailov 被引实际读到约 85/100（且总被引数可能 >100，未确认）；EPIC 被引读到 45/100（总数未确认）。**两条链都没扫完，"未找到"在本节里只意味着"在已扫部分里未找到"。**
- **(a)/(b) 命中全部未取全文**，因此本节不含任何带节号的原句引用。§1 表里"全文原句"那一列不会因 §8 而增加任何一行。

---

# §8-FINAL 补充轮：引用链补扫（列表层已扫完）

本节取代上面的旧 §8。配额在 21:40 UTC 重置后续跑完成。

## 8F.0 坏消息先说

1. **交办的三项都有结果，但 B2 的四条待核项全部结算为"不覆盖"——这不是好消息，是因为它们都不在我们的轴上，所以 B2 的缺口依旧是缺口，没人替我们证明过它值得做。**
2. **A04 的题录是错的，错在我这边的上游清单里。** arXiv:2205.11736 **不是** "Wang et al. TMLR'23 memorization vs generalization of poison"，而是 **Wang, Hayase, Fanti, Oh, "Towards a Defense Against Federated Backdoor Attacks Under Continuous Training"**（v1 2022-05-24，v4 2023-01-31；arXiv 页无 journal-ref，ar5iv 正文也未声明 venue）。10-04 清单里"TMLR'23 + memorization"这个描述两处都对不上。按 7.4，这条要改掉，不要再引用那个描述。
3. **B4 新增一条不利证据**：基于"毒样本学得更快"这类速度信号的过滤，已经被一篇专门的自适应攻击打过（CIKM 2023 Deep Synchronization，下见 8F.2 第 4 条）。我们如果把 B4 建在速度/早期动力学信号上，这篇是必须先过的一关。
4. **B5 的轴差现在有了明确的文献根据**：已有理论把 margin 写成毒样本**计数**的函数（Khaddaj et al. 的 k-strength，对计数差分），**没有**把它写成**时序**的函数。这既是机会也是警告——机会是轴确实空着，警告是审稿人会直接问"为什么对计数差分不够"。

## 8F.1 扫描完整度

| 链 | 被引总数 | 已枚举 | 完整度 |
|---|---|---|---|
| Shumailov et al. 2104.09667 | **125** | 125（第 1–100 条分三段读完返回体 chars 0–221956；offset=100 一页返回 25 条） | **列表层扫完** |
| Yang et al. 2210.09671（EPIC） | **51** | 51（第 1 页 chars 0–105876 分两段读完 51 条；offset=100 返回 `"data": []`，确认无第 101 条） | **列表层扫完** |

**仍是部分的地方**：Semantic Scholar 对两条链里相当一部分条目返回 `abstract: null`（Shumailov 链的 101–125 段与 200k 尾段几乎全无摘要；EPIC 链尾段 6 条里 5 条无摘要）。这些条目的分类**只基于标题**，置信度低。另：ACM DL 的 CIKM'23 全文取不到（见 8F.2 第 4 条）。

## 8F.2 Shumailov 链 (a) 类命中：4 条，逐条给自变量/因变量

### (a)-1 arXiv:2106.09993 *Accumulative Poisoning Attacks on Real-time Data*（NeurIPS 2021）— **全文已读**
- **自变量**：攻击者对"累积阶段"训练批次的扰动（PGD，ℓ∞ 预算 ε）；累积步数（Table 6）；权衡系数 λ（Fig 3b）；trigger batch 是毒的还是干净的。§3.2："We inject an accumulative phase 𝒜"。**不操纵样本次序。**
- **因变量**：trigger batch 上**单步**更新造成的测试精度跌幅。§4："we exploit single-step drop as a threatening poisoning target."
- **是否后门**：**不是**。§3.4 与后门对照，自述 "our methods only need to manipulate training data"，目标是整体精度下降，无 trigger 条件行为。
- 无逐样本效力排名（Algorithm 1 对 batch 内每个样本都扰动）。无聚簇/到达分布作为变量（§3.2："in real-time settings, we only poison data during the accumulative phase"，毒批天然是紧邻的一整块，但未分析到达时间分布）。
- **对 B2：不覆盖。** 它确实把"训练中的某一段时间"当自变量，但载体是可用性攻击、因变量是单步精度崩塌、没有 identity、没有 take-off step。

### (a)-2 arXiv:2406.12222 *BadSampler: Harnessing the Power of Catastrophic Forgetting to Poison Byzantine-robust Federated Learning*（2024）— **全文已读（前 100k/109.8k）**
- **自变量**：主要是"喂哪些样本"。Abstract：对手须 "selectively sample training data with high loss to feed model training"；§4.1 把机制描述成操纵 "the order of the local training batch without using poisoned data"。两种说法混用，**实操是按分数选 Top-κ，顺序只是投送手段。**
- **因变量**：泛化误差（Eq 6, §4.2）；实测指标是精度，§6.1 把攻击效果定义为 "the reduction in global model accuracy due to the attack"。**全文不报"第几轮发生"。**
- **是否后门**：**不是**，非定向。§3.2："an indiscriminately high error rate for testing examples"，无 trigger、无目标类。
- **逐样本分数：有。** Definition 2 (§3.1)：err_i = E_D[ℓ(f_θ(x_i|D), y_i)]，作者称 sample difficulty；Eq 7 (§4.3) Top-κ 保留 κ·B 个最高分作候选池；Eq 8–9 的 meta-sampler 改用 loss 直方图与高斯权重。§4.3："sample difficulty is calculated by the surrogate model."
- 无 take-off 分析（Thm 3 给误差上界；§6.1 报 T=250 轮的最终最高精度；§6.2.6 只比 loss landscape 与 Hessian）。不做防御用途（§6.1 评测 PCA/FoolsGold/FLDetector/Trimmed Mean/Krum/FLTrust）。
- **对 B1：抽象阶梯上新增一个强命中**——"用代理模型算的逐样本分数挑恶意样本、经 batch 排序投送、完全不用毒数据"。**对 B2：不覆盖**（顺序是手段不是自变量，因变量是精度）。

### (a)-3 arXiv:2609.35528 *Let the Neurons Die: Exploiting ReLU-Induced Model Degradation*（2026）— **仅摘要页**
- **自变量**：三种并用。(i) DOA 数据排序——贪心挑下一个样本使目标层更新后权重和最小，"without modifying training samples or labels"；(ii) IG-DOA / IG-SKA 用梯度反演合成类条件毒样本（摘要报 200 条、来自单一类）；(iii) soft knockout 跨相邻层重排权重。
- **因变量**：5 个 epoch 后的测试精度（摘要里唯一量化的结果）。机制量是 dying ReLU 单元，摘要未给指标。摘要报：对 60000 条里的 100 条排序，约 96%→95%；加 200 条单类毒样本，约 86–88%（干净约 96%）。
- **是否后门**：**不是**，自述 "training-time availability attacks"，无 trigger、不报 ASR。
- 无逐样本效力排名、无 take-off step、无聚簇分析（摘要层）。
- **对 B2：不覆盖。** 价值在于确认"贪心排序挑样本"这个动作本身在 2026 年已有人做。

### (a)-4 DOI 10.1145/3583780.3614784 *Attacking Neural Networks with Neural Networks: Towards Deep Synchronization for Backdoor Attacks*（CIKM 2023，Guan/Sun/Du/Liu；早期版本在 OpenReview 题为 *Feature Synchronization in Backdoor Attacks*）— **仅摘要（ACM DL 全文未取到）**
- **它的 "early-fitting" 不是到达顺序**，而是"毒样本 loss 下降更快"这一现象。摘要原文："the loss of poisoned samples tends to drop significantly faster than that of clean samples, which we call the early-fitting phenomenon. Early-fitting provides a simple but effective evidence to defend against backdoor attacks"。
- **自变量**：synchronization（跨层样本相似度，摘要说 "the degree of synchronization could be controlled based on whether it is captured by shallow or deep layers of the model"），用来主动压低毒样本的拟合速度。**因变量**：ASR + 对基于 early-fitting 的检测的规避。
- **归类改正**：它应归 **B4** 而非 B2。**并且它是对 B4 不利的证据：基于学习速度的过滤信号已被专门的自适应攻击打过。**

## 8F.3 Shumailov 链 (b) 类（一般数据顺序，非投毒）：7 条
- **arXiv:2202.01838** *Characterizing & Finding Good Data Orderings for Fast Convergence of Sequential Gradient Methods*（2022）—— 顺序影响收敛速度，并给贪心选序方法。**这是 B2 在"一般学习速度"层面的现成方法源。**
- **arXiv:2206.02953** *Sampling without Replacement Leads to Faster Rates in Finite-Sum Minimax Optimization*（2022）—— 分析数据排序攻击。
- **arXiv:2606.29554** *Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise*（2026）—— 主张优化器状态使 shuffle order 成为微调噪声的一阶来源。**与我们的 seed cross-over 结论方向一致，但因变量是微调方差而非后门获得时刻。全文未取。**
- **arXiv:2509.14223** *Fresh in memory: Training-order recency is linearly encoded in language model activations*（2025）—— 训练次序在激活里线性可读。与"防守方可观测量"相关。全文未取。
- arXiv:2411.17292（课程学习 VQA）、arXiv:2310.10124（课程学习的隐私风险）、arXiv:2307.04138（指出数据顺序的随机性是方差主要来源）。

另 5 条边界件（*Unforgeability in SGD*、SISA / ARCANE 的请求排序、*Proof-of-Learning is Currently More Broken Than You Think* 2208.03567、*Discrete Key-Value Bottleneck* 2207.11240）与"轨迹/次序"沾边，但不是毒样本顺序。

**Shumailov 链其余约 110 条**：machine unlearning（约 25 条）、后门防御/检测（约 15 条）、架构后门与供应链（约 10 条）、联邦与隐私（约 12 条）、随机性与可验证训练（约 10 条）、综述与政策（约 15 条）、其他。与 B 的五个子主张无直接接触。

## 8F.4 EPIC 链（51 条全扫）

### (a) 同一逐样本分数同时用于攻击侧选择与防御侧过滤 —— **51 条里 0 条**
最接近的仍是 **arXiv:2504.02142** *Like Oil and Water: Group Robustness Methods and Poisoning Defenses May Be at Odds*（2025）：group-robustness 方法放大少数类样本因而也放大毒样本，而投毒防御用**相近的**启发式剔除样本。这是"两侧启发式互相冲突"的观察，不是一个分数两侧共用。全文未取。

### (b) 训练动力学 / 学习速度 / loss 轨迹 / 逐样本重要性当防御过滤器 —— 3 条
- **PoisonSpot**, DOI 10.1145/3719027.3744802（2025）*Precise Spotting of Clean-Label Backdoors via Fine-Grained Training Provenance Tracking*：逐样本追踪它对参数更新的影响、按这个 lineage 打可疑度分。**与我们 B4 设想的机制同族，全文未取。**
- PureEBM arXiv:2405.19376（*mid-run dynamics of energy-based models*）、PureGen arXiv:2405.18627。

### (c) 仅攻击侧毒样本选择 —— 3 条
DOI 10.1109/ICCV51070.2023.00443 *Computation and Data Efficient Backdoor Attacks*（ICCV 2023，按 **posterior distance** 打分并贪心选）、arXiv:2306.08313 PFS（已读全文）、arXiv:2409.08985 *Clean Label Attacks Against SLU Systems*（选对代理模型而言"难"的样本；**语音理解，B1 抽象阶梯又多一轴**）。

### 本轮因 EPIC 链而优先取到的一篇关键全文
**arXiv:2307.10163 *Rethinking Backdoor Attacks*（Khaddaj et al., ICML 2023）— 全文已读（前 100k/101.9k）**
- Assumption 1 (§3)："the backdoor trigger is the strongest feature in the dataset"。
- 逐样本量 **k-strength**（§3, "Characterizing feature strength"）：s_φ(z,k) = g_φ(z,k+1) − g_φ(z,k)，其中 g_φ(z,k) 是在含特征 φ 的 k 个样本上训练时模型对 z 的期望 margin；§3 把 feature strength 定义为 "the rate of change of the corresponding output function"。
- **同一量只用于检测，不用于构造**：§4 用 datamodels 估 feature strength（Lemma 1）、搜最强特征的 support（Eq 6–7）；攻击用的是现成的 dirty-label（BadNets）与 clean-label（App D.1）。§4："identifying the feature of interest (i.e., the backdoor feature) is essentially our goal."
- 检测分数 Eq 8：s_i = Σ_k (1/k) Σ_l v_i^{k,l}（样本 i 在局部搜索解里出现频次的加权计数）；§4："We translate the scores into a concrete defense by flagging (and removing) the examples with the highest score."；§5.2 去掉 top 10%（实验 2 为 20%）。
- 不讨论毒样本之间的强弱异质：Def 2 / Lemma 1 对 support 取平均 (1/|Φ|)Σ_z；App E.3 只说 "the effect of poisoned samples on other poisoned samples is generally higher"。
- 无训练动力学、无到达顺序、无"何时获得"；**k 数的是毒样本个数，不是训练时间**（App D.2：在随机 50% 子集上训到收敛）。
- **对 B4：仍不构成"同一分数两侧共用"**（构造侧用的是别人的攻击）。**对 B5：新增部分覆盖**——已有一个形式化的逐样本边际效力量，但它是对**计数**差分。

## 8F.5 第 3 项：A04 身份核实（arXiv:2205.11736）

**实际是**：Shuaiqi Wang, Jonathan Hayase, Giulia Fanti, Sewoong Oh, ***Towards a Defense Against Federated Backdoor Attacks Under Continuous Training***。v1 2022-05-24，v4 2023-01-31。arXiv 页无 journal-ref；ar5iv 正文也未声明 venue。cs.LG，交叉 cs.AI / cs.CR。**全文已读（前 100k/140.7k）。**

**上游清单的描述（"TMLR'23 memorization vs generalization of poison"）两处都错**：既无 TMLR 记录，主题也不是 memorization。按 7.4 自报：这条是我在 10-04 清单里带进来的错误题录，后续不要再按那个描述引用。

**它实际讲什么，以及与 B2/B5 的关系**：
- "backdoor leakage"（Abstract）："causes models trained continuously to eventually suffer from backdoors due to cumulative errors in defense mechanisms"；§1："In any single training iteration, a defense mechanism may remove a large portion of backdoored training data." / "But a defense rarely removes *all* backdoored data." / "Hence, if the model is trained for enough iterations, eventually the attack success rate increases to 1."；§3 Insight 1："a small fraction of malicious gradients or model updates always go undetected" / "this backdoor leakage eventually leads to a fully-backdoored model."
- **与 B2 的关系：相关但轴不同。** 自变量是训练轮数 / 持续时长，因变量是最终 ASR（"eventually → 1"）。**不含 identity 级顺序、不含聚簇、不含 take-off step**；也不给具体哪一轮后门出现。
- **与 B4 的关系：相关。** §3.1："Finally, the shadow model is early-stopped to avoid backdoor leakage."；早停判据："The shadow network N′ is early-stopped once the training accuracy on benign target samples converges."；过滤用 SPECTRE 的 QUE 分数，阈值取第 1.5ᾱ|C| 大，QUE "intuitively captures how \"abnormal\" a sample is"——但**是客户端级，不是逐样本级**。
- 无逐样本效力/异质性分析（§5.4 的 heterogeneity 指客户端异质，不指样本）。无顺序/聚簇变量。
- **与 B5：无贡献**（无理论预测 take-off 时刻）。
- **结论：不改任何子主张判定。** 它给 B2 的"时间轴"版本（累计轮数 → 最终 ASR）又补一个既有实例。

### 顺带核到的另一篇（本轮 A04 检索的副产品）
**arXiv:2310.10780 *Demystifying Poisoning Backdoor Attacks from a Statistical Perspective*（Ganghua Wang, Xun Xian, Ashish Kundu, Jayanth Srinivasa, Xuan Bi, Mingyi Hong, Jie Ding；UMN + Cisco Research）— 全文已读。** 这是 §1 表里原先只有题录的那条（当时标为 ICLR 2023 citer）。
- 决定因素：poisoning ratio ρ、trigger 的方向与 ℓ2 幅度、干净数据分布（§1.1 原文列举）；§2："each data input in the clean dataset has probability ρ∈(0,1) to be chosen as a backdoor data."
- **无逐样本量、无异质性分析**（每条毒样本共享同一个常量 trigger η 与目标标签 0，选中是独立 Bernoulli 抽样）、**无顺序/时机/训练动力学**（结论是 n 条 i.i.d. 样本训出的最终模型 f̂^poi 的风险界）、**无逐样本排名**（指导意见是选 trigger 的方向与幅度）。Remark 3 把 ρ→0 的情形留作开放。
- **对 B5：不覆盖 take-off 部分。**

### 两条新的 B5 线索（本轮检索中出现，**未读**）
- arXiv:2406.00588 *Generalization Bound and New Algorithm for Clean-Label Backdoor Attack*
- arXiv:2508.05600 *Provable one-poison backdoor attacks on linear models and ReLU neural networks* —— 题目直指"单条毒样本即可"，是逐样本效力理论的潜在强相关项。**下一轮优先。**

## 8F.6 子主张判定变动

| 子主张 | §1 判定 | §8-FINAL 后 | 理由 |
|---|---|---|---|
| **B1** | 已覆盖 | **已覆盖（不变）** | 新增两个载体与一个打分量：BadSampler 的 surrogate-computed sample difficulty + Top-κ（2406.12222 §3.1 Def 2 / §4.3 Eq 7）、ICCV'23 的 posterior distance、SLU 的 proxy-hardness。只加厚，不改判 |
| **B2** | 部分覆盖 | **部分覆盖（不变）；§8 旧版标的"风险上升"现已解除** | 四条 (a) 待核项全部结算：2106.09993（非后门/单步精度）、2406.12222（非定向/精度/顺序只是手段）、2609.35528（可用性/精度）、CIKM'23（其实是学习速度现象，应归 B4）。**没有一条把 identity 级到达次序、聚簇度或 take-off step 当变量。** 另新增 2205.11736 的"累计轮数 → 最终 ASR"版本 |
| **B3** | 部分覆盖 | **部分覆盖（不变）** | 新增方法源 arXiv:2403.07384 S2L（用小模型训练轨迹给样本打分做 LLM 微调数据选择）——可移植但未见移植到毒样本效力预测 |
| **B4** | 已覆盖（防御侧）/ 同分数两侧共用未找到 | **判定不变，但证据变硬，同时新增一条不利证据** | "未找到"现在建立在 EPIC 全部 51 条被引 + *Rethinking Backdoor Attacks* 全文之后；该文的 Eq 8 分数**只**用于检测，构造侧用现成攻击。不利证据：CIKM'23 Deep Synchronization 专门攻击 early-fitting 这类速度信号 |
| **B5** | 部分覆盖 | **部分覆盖（内容更新）** | 新增 2307.10163 的 s_φ(z,k) = g_φ(z,k+1) − g_φ(z,k) 作为"逐样本边际效力"的既有形式化——**对毒样本计数差分**；2310.10780 确认统计侧只到 ρ + trigger 几何。**"对时间/顺序差分"的理论仍未找到** |

## 8F.7 阅读计数（全报告累计）

**新增全文 5 篇**：arXiv:2106.09993、arXiv:2406.12222（前 100k/109.8k）、arXiv:2307.10163（前 100k/101.9k）、arXiv:2310.10780、arXiv:2205.11736（前 100k/140.7k）。

**新增仅摘要 2 篇**：arXiv:2609.35528、DOI 10.1145/3583780.3614784（CIKM 2023）。
其中 2307.10163 与 2406.12222 从"仅题录"升为全文；2310.10780 从"仅题录"升为全文。

**full texts read: 19; abstract-only: 7; titles only: 44.**

计数规则（接 §3 第 8 项）：只计入报告中**实际引用**的条目。两条引用链合计 **176 条**（Shumailov 125 + EPIC 51）在标题/摘要片段层做过分类，但未逐条取正文，**不计入上述三档**；本节只把其中被引用到的条目计入 "titles only"。

## 8F.8 §8-FINAL 查询日志

### WebFetch
31. `api.semanticscholar.org/graph/v1/paper/arXiv:2104.09667/citations?fields=title,abstract,year,externalIds&limit=100`（body offset 0）—— chars 0–100000/221956
32. 同上（body offset 100000）—— chars 100000–200000
33. `api.semanticscholar.org/graph/v1/paper/arXiv:2210.09671/citations?fields=title,abstract,year,externalIds&limit=100`（body offset 0）—— chars 0–100000/105876
34. 同 31（body offset 200000）—— 第一次 **session limit**；重置后重试成功，chars 200000–221956
35. 同 33（body offset 100000）—— 第一次 **session limit**；重置后重试成功，chars 100000–105876
36. `api.semanticscholar.org/graph/v1/paper/arXiv:2104.09667/citations?fields=title,year,externalIds&limit=100&offset=100` —— 成功，返回 25 条（第 101–125 条）
37. `ar5iv.labs.arxiv.org/html/2106.09993` —— 成功
38. `ar5iv.labs.arxiv.org/html/2406.12222` —— 成功
39. `ar5iv.labs.arxiv.org/html/2307.10163` —— 成功
40. `api.semanticscholar.org/graph/v1/paper/arXiv:2210.09671/citations?fields=title,year,externalIds&limit=100&offset=100` —— 第一次 **HTTP 429**；退避后重试成功，返回 `"data": []`（确认 EPIC 被引总数 51）
41. `arxiv.org/abs/2609.35528` —— 成功（摘要页）
42. `ar5iv.labs.arxiv.org/html/2310.10780` —— 成功
43. `arxiv.org/abs/2205.11736` —— 成功（身份核实）
44. `ar5iv.labs.arxiv.org/html/2205.11736` —— 成功

### WebSearch
24. `PoisonSpot clean-label backdoor fine-grained training provenance tracking per-sample score` | standard —— 第一次 **session limit**，未重发
25. `Wang TMLR 2023 backdoor poisoning memorization generalization learning dynamics "Transactions on Machine Learning Research" theory of backdoor sample efficiency` | extended
26. `"Deep Synchronization for Backdoor Attacks" Attacking Neural Networks with Neural Networks CIKM 2023 early-fitting poisoned samples epochs` | extended

## 8F.9 §8-FINAL 的边界申报（叠加在 §4 之上）

**哪些部分仍是部分的：**
1. **CIKM 2023 (DOI 10.1145/3583780.3614784) 全文未取到** —— ACM DL 付费墙；本节关于它的全部陈述来自摘要。OpenReview 的早期版本 *Feature Synchronization in Backdoor Attacks* 未取。
2. **arXiv:2609.35528 只读了摘要页**，HTML/PDF 正文未读。它的 (a) 判定（可用性攻击、无 take-off、无逐样本排名）因此是摘要层结论。
3. **三篇全文读到了 ar5iv 返回体的前 100k**：2406.12222（/109.8k）、2307.10163（/101.9k）、2205.11736（/140.7k）。未读尾段若含顺序/时序分析或防御侧讨论，本节会漏判。2205.11736 的尾段占比最大（约 29%），风险最高。
4. **两条链的摘要缺失**：Shumailov 链第 101–125 条与 200k 尾段几乎全无摘要；EPIC 链尾段 6 条里 5 条无摘要。这些条目**只按标题分类**。"Shumailov 链 (a) 只有 4 条"因此是"标题+部分摘要层"的结论，不是摘要全覆盖层的结论。
5. **仍未扫的引用链**（§4 已标，本轮未动）：SAILS 2609.15029 的被引、ABL 2110.11571 的被引、Souly 链里 4 条可能相关项的全文（2610.07510、2510.26829、2607.06643、2603.03371）。
6. **本轮新发现未读**：arXiv:2406.00588、arXiv:2508.05600（B5 线索）、PoisonSpot 全文、arXiv:2504.02142 全文、arXiv:2606.29554 全文、arXiv:2403.07384 全文。
7. **非英文、在审稿件、知识截止**：同 §4，无变化。

**下一轮优先级**：(1) arXiv:2508.05600（one-poison 理论，直接压 B5/B1）；(2) PoisonSpot 全文（B4 最近邻）；(3) arXiv:2606.29554 全文（与我们 seed cross-over 同轴）；(4) 扫 SAILS 与 ABL 的被引。
