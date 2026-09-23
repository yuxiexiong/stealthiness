# 第 2 步：方法论单独搜（method-as-object）

日期 2026-09-23。执行者：step-2 检索代理。范围：只把方法当成脱离领域的对象来搜，外加外领域方法源。
方法层被占按 BRIEF 规定「不致命」。本步**没有**做对象层 / 结论层的判定（那是第 1、3、6 步的工作）；顺手看到的对象层线索放在第 5 节末尾，交给相应步骤核查。

**统计**：编号查询 54 条（WebSearch 49、直接 curl arXiv 5，另有 2 次抓取失败，见边界申报）；进入判读 48 篇；取到正文 22 篇（21 篇 arXiv PDF + 1 篇 Engle & Kane 2004 书章 PDF），其中 **13 篇**做过定向追问（在正文中 grep 并读了方法段落），**9 篇**只从下载的正文里读了摘要 / 首页。其余 26 篇只看了检索片段，**不算读过**，表中标「否」。

---

## 0. 坏消息先说：方法设计中必须修正的点

按严重程度排列。每条注明档位（推导 / 类比 / 推测）和出处原句。标「检索片段」的引文是搜索摘要里的句子（可能是二手转述），**不是**原文逐字；其余引文都出自 s2txt/ 里的本地 txt，逐字。

### F1（严重）DiD 只对加性偏置不变，对乘性缩放不变不成立；而 LLaMA 系模型确有一条「整体缩小 logits」的通路
- 我们的说法是：「在非攻击候选答案之间做双重差分，使指标对攻击答案的加性偏置严格不变。」这句话本身对：任何与问题无关的加性 logit 平移（不管加在 violin 上还是任何 token 上）在 DiD 中都会抵消。**但它不覆盖乘性缩放**：LLaVA-1.5 的语言骨干是 LLaMA/Vicuna，logits = W_U·(g ⊙ x / rms(x))。如果 trigger 让末层残差范数变大，**所有** logit 差都会同比例变小，于是 T/N 保留率 < 1。这个结果和 H1/H2（问题不再被使用）长得一样，但其实只是「整体温度升高」。（**推导**）
- 这条通路已有实证：Stolfo et al. 2024（2406.16254）：「Entropy neurons are characterized by an unusually high weight norm and influence the final layer normalization (LayerNorm) scale to effectively scale down the logits.」「entropy neurons operate by writing onto an unembedding null space, allowing them to impact the residual stream norm with minimal direct effect on the logits themselves.」文中明确包括 LLaMA2 7B：「LLaMA actually uses RMSNorm [80] which differs from LayerNorm in the absence of re-centering and bias term. However, our experimental procedure is not affected by this difference.」后门 trigger 会不会走这条通路属于**推测**，但是否排除必须由测量决定，不能靠假设。
- **修正**：(a) 每次前向都记录末层 RMSNorm 的缩放因子 1/rms(x)，DiD 同时报原始值和按缩放因子校正后的值；(b) 增加一个**序数版**主指标：只看非攻击候选之间的排序（例如「问题对应的候选是否排在替代候选之前」、成对 AUC），它同时不受加性平移和正数倍缩放影响；(c) H2 用 T 保留率 / O 保留率的**比值**判定，公共缩放在比值中抵消。H1 和 H3 的区分依赖保留率的绝对值，所以必须先过 (a)(b)。

### F2（严重）DiD 的不变性依赖所选刻度（logit），这正是计量经济学里「平行趋势对函数形式敏感」的问题
- Roth & Sant'Anna（2010.04814，Econometrica 2023）：「the parallel trends assumption holds under all strictly monotonic transformations of the outcome if and only if a stronger "parallel trends"-type condition holds for the cumulative distribution function of untreated potential outcomes.」「suggest that researchers should be careful to give a functional form-specific justification in settings where these conditions are not plausible.」
- 对应到我们（**类比**）：「加性偏置在 DiD 中抵消」只在 raw-logit 刻度上成立；换到概率、log-prob（log-softmax 会减去一个与问题有关的 logsumexp）、序列分数，结论都可能变。log-softmax 的归一化项对同一前向里的所有候选相同，所以在候选之间做差时会消掉；但首 token logit 和整段序列分数之间就没有这种保证。
- **修正**：合同里写死刻度（首 token raw logit 的候选间差），并写明**为什么**选这个刻度：H3a 本身就是在 logit 空间定义的。F1(b) 的序数指标作为不依赖函数形式的稳健性版本，一起预注册。

### F3（严重）只有 CLEAN 加两颗重训种子，标定不出 null 带
- McCoy et al.（1911.02969）：「behavior of all instances was remarkably consistent, with accuracy ranging between 83.6% and 84.8%」；而分布外行为「accuracy ranged from 0.0% to 66.2%」。我们的「问题使用保留率」正是这种分布外的行为量，所以种子间方差可能远大于 accuracy 方差（**类比**）。
- 最直接的领域先例 Dual-Key / TrojVQA（2112.07668）：「We account for the influence of random model initialization by training multiple VQA models on each dataset with different seeds. Following [11] we train 8 models per trial, and report the mean ± 2 standard deviations for each metric.」
- 另见用户记忆「重跑不等于复现」：确定性训练器上重跑产出同一批数据，种子必须真的不同（数据顺序 + LoRA 初始化都要变）。
- **修正**：null 带至少用 ≥5 个独立种子（CLEAN 与 RETRAIN 合并计），或者明确降格为「单种子描述」，不称为 null 带。按图聚类 bootstrap 不能代替种子方差（路线图已写，这里给出处）。

### F4（中）sham patch 需要一个「主动安慰剂」式的操纵检查；单个「内容无关」的对照本身也可能有偏
- 校准文献已经证明，「内容无关」的对照输入并不中性。Zhou et al. Batch Calibration（2309.17249）：「the empirical observation shows that content-free inputs can be inappropriate prior estimators」；「Consequently, the prior estimated via a single content-free token can lead to further bias.」
- 后门领域的惰性图案标准，SentiNet（1812.00292）：「we expect that adversarial regions will cause many misclassifications when overlaid on the test set, but have little effect on the network when replaced by an inert pattern」「we expect benign patterns to either cause few misclassifications, or to occlude objects and thus also disrupt the model when replaced by inert patterns.」
- 医学 sham / active placebo（检索片段，未读正文）：好的 sham 必须模仿**非特异性副作用**，还要做盲法检验。
- **修正**（**类比**）：(a) 用多个 sham（≥3 种纹理，匹配面积、位置、对比度、高频能量），不要只用一个；(b) **操纵检查**：在 CLEAN 模型上，trigger 和 sham 引起的变化必须落在同一个 null 带内（也就是对干净模型来说两者都只是遮挡物，相当于盲法检验）；(c) **惰性检查**：在投毒模型上，sham 的 ASR 必须处在 CLEAN 水平。若某个 sham 在投毒模型上部分触发（trigger 泛化），它就不是 sham，只能作为剂量梯度的一档。

### F5（中）「禁用攻击 token 后解码能否恢复正确答案」分不开 H3 和「只剩问题先验」
- VQA 模型只凭问题先验就能答对相当一部分题。Agrawal et al. 2016（1606.07356）：「for 40% of the questions, the CNN+LSTM model seems to have converged on a predicted answer after 'listening' to just half the question.」Goyal et al. VQAv2（1612.00837）为此构造了互补图像对：「every question in our balanced dataset is associated with not just a single image, but rather a pair of similar images that result in two different answers to the question.」
- 所以禁用 violin 后第二选择答对，可能是 H3（视觉加问题的正常计算仍在），也可能只是按题型先验猜中（例如 "what color" → "white"）。（**推导**）
- 已有同类方法：CleanGen（2406.12257）在解码时替换可疑 token：「backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents」。latent knowledge 文献用 Hits@k（2412.20846，未读正文）。
- **修正**：恢复测试必须在 VQAv2 式的**互补图像对**上做：同一问题、两张图、正确答案不同。只有「禁用后的答案随图变化且都对」才算恢复；另外报告问题先验基线（遮掉图像证据，或用只看问题的答案分布）作为下限。

### F6（中）总体保留率 50% 可能是逐题「全有 / 全无」的混合，而不是逐题都减半
- 任务切换文献的 failure-to-engage 模型（De Jong 2000；检索片段）：「observed responses are a simple mixture of prepared and unprepared response strategies」。混合模型有一个可检验的标志，叫固定点性质（Falmagne 1968；检索片段）：「independent of the mixture proportion, there will always be one probability density that is shared across all possible mixtures of the same two base distributions」。
- 对应到我们（**类比**）：沿投毒剂量阶梯画逐题保留率（或 DiD）的分布。如果各剂量的分布曲线交于同一点，说明是 H1 类失效与 H3 类竞争两种状态的逐题混合，剂量只改变混合比例；如果整体平移，说明是连续的部分失效。H1 / H2 / H3 本来就可能在不同题上各自成立（路线图已承认），这个检验把「混合」变成一个可以证伪的预测。
- **修正**：把「混合 vs 连续」作为预注册分支。注意用户记忆「弱臂要更长的阶梯」：固定点检验至少需要 3 档以上、覆盖两端的剂量。

### F7（中）认知心理学已有一套分离「目标失效」和「反应竞争」的操作，我们现在的设计缺了其中两个判别杠杆
Engle & Kane 2004（Psychology of Learning and Motivation 44 书章，总结 Kane & Engle 2003 JEP:General 的五个实验；原始 2003 文章抓取失败，见边界申报）：
- 两个因子：「We propose one factor of control to be the maintenance of the task goals in active memory」；「The second factor in the executive control of behavior is the resolution of response competition or conflict, particularly when prepotent or habitual behaviors conflict with behaviors appropriate to the current task goal.」
- 判别标志 1：目标失效产生**干净的优势反应错误**。「errors resulting from goal neglect (and subsequent word reading) should be relatively fast compared to other kinds of errors」「these errors represented rapid word reading due to failed access to the goal state.」
- 判别标志 2：一致试次上的**促进效应**变大。「the word reading responsible for facilitation effects is a result of periodic failure of goal maintenance. Low spans, therefore, should show greater facilitation than high spans; this is just what we found.」
- 判别标志 3：竞争只表现为正确反应**变慢**，不表现为错误。「low spans were responding according to goal, but they were slower to resolve the competition between color and word than high spans.」
- 情境操纵：「In Stroop contexts that reinforced the task goal by presenting 0% congruent trials, we found modest span differences in response-time interference.」配套的**目标提醒**操纵，Hood & Hutchison 2021（检索片段）：「Working memory capacity negatively correlated with Stroop errors in a control condition, but not in the goal reminder condition.」

**嫁接（类比，需自行验证）**：
1. **目标提醒因子**：在问句里加入任务重申（如 "Answer with a color."），作为第三个 2 水平因子。H3a 的预测是提醒效应与 trigger 效应在 logit 上可加（提醒 × trigger 交互 = 0，并落在 null 带内）；H1/H2（目标失效）的预测是提醒在有 trigger 时作用明显更大，即部分恢复。这是一个对「H3a 与目标失效」**有方向**的判别，比只看保留率更强。
2. **一致试次促进**：正确答案本来就是 violin 或乐器类的题（路线图要求单列、不计 ASR），应当升格为判别探针。目标失效预测 trigger 在这些题上的额外促进，大于在其他题上对 violin 的推高；纯加性 H3a 预测两者相同（都是同一个常数）。
3. **错误类型谱**：H1 类失效时错误应当几乎全是精确的 "violin"（干净的优势反应）；如果错误里有大量「其他错误答案」，说明正常计算本身受损（视觉遮挡或能力破坏），这部分交给 sham 对照。
4. **反应集效应**（Klein 1964；检索片段）：「found much greater interference when the written word was a member of the response set than when it was not」。这与路线图第二阶段的「答案类型相容性」预测是同一个构念，可直接引用为先验。De Houwer 2003 的 2:1 映射（检索片段）是分离「语义冲突」与「反应冲突」的标准设计，可以借来设计「violin 与正确答案落在同一反应类」的题（例如是非题），但目前只做到类比，没有具体方案。

**ML 内已有人把这个范式搬过来**（方法被占，必引）：Hu … Sripada 2026（2608.11510）：「a prompt stem elicits a default same-color completion and an explicit rule either agrees with (congruent condition) or conflicts with (incongruent condition) the completion」；「Fine-tuning that strengthened the default same-color tendency had divergent effects on task conditions, reducing incongruent performance while increasing congruent performance.」；「congruency effects in this task arise from competition between an in-weight default mapping and an in-context rule-based mapping.」它研究的不是后门，也没有 trigger，但「权重内默认映射 vs 上下文规则」的竞争框架与 H3 同构。见第 5 节交给第 3 步的提示。

### F8（低，但写进合同）改写本身就会动分数，T 效应和 O 效应要对照「同义改写」噪声来标定
- VQA-Rephrasings（1902.05660，检索片段）：「State-of-the-art VQA models are notoriously brittle to linguistic variations in questions」；CheckList 的 INV 测试（检索片段）正是做这种标定。
- **修正**：T / O 的 DiD 必须超过同一图、同一模型上「同义改写 DiD」的分布，才算「问题含义在起作用」。路线图已经有同义改写版本，缺的是把它**写进判据的 null**，而不是只当描述量报告。

### F9（低）概率指标会漏掉负效应
- Zhang & Nanda（2309.16042）：「Evaluation metrics We generally recommend avoiding using probability as the metric, given that it may fail to detect negative model components.」同文也支持我们只在非攻击候选之间做差：「By measuring Logit(IO) − Logit(S), logit difference controls for such components and ensures they are not detected. This may not be achieved by other metrics, such as probability or Logit(IO) alone.」我们的设计已经合规，只需引用。

---

## 1. 查询日志

| # | 来源 | 查询串 | 命中要点 |
|---|---|---|---|
| Q1 | WebSearch | Analyzing the Behavior of VQA Models Agrawal 2016 "listen to half the question" | 1606.07356；另 1612.00837 VQAv2、1712.00377 VQA-CP、1805.05492 |
| Q2 | WebSearch | counterfactual VQA language bias natural indirect effect Niu 2021 | CF-VQA CVPR21：从总效应中减去问题的直接效应 |
| Q3 | WebSearch | contrast sets Gardner 2020 local decision boundaries | 2004.02709 |
| Q4 | WebSearch | VQA rephrasings consistency Shah cycle-consistency | 1902.05660 VQA-Rephrasings |
| Q5 | WebSearch | Calibrate Before Use contextual calibration "N/A" Zhao 2021 | 2102.09690 |
| Q6 | WebSearch | Surface Form Competition domain conditional PMI Holtzman 2021 | 2104.08315 |
| Q7 | WebSearch | Best Practices of Activation Patching logit difference Zhang Nanda | 2309.16042 |
| Q8 | WebSearch | Heimersheim Nanda How to use and interpret activation patching | 2404.15255 |
| Q9 | WebSearch | Mitigating label biases ICL domain-context calibration Fei 2023 | 2305.19148；Batch Calibration 2309.17249 |
| Q10 | WebSearch | confidence regulation neurons entropy neurons final LayerNorm Stolfo 2024 | 2406.16254 |
| Q11 | WebSearch | CheckList behavioral testing INV/DIR/MFT Ribeiro 2020 | ACL 2020 |
| Q12 | WebSearch | backdoor defense random patch baseline same location non-trigger patch occlusion | 2112.07668 Dual-Key；2604.04488；PatchDrop（AAAI） |
| Q13 | WebSearch | goal neglect Duncan 1996 intelligence frontal lobe | Duncan et al. 1996 Cogn Psychol 30:257-303 |
| Q14 | WebSearch | Stroop task conflict vs response conflict dissociation factorial design | 任务冲突 vs 信息冲突；刺激冲突 vs 反应冲突（N450 / LPC） |
| Q15 | WebSearch | De Jong 2000 failure to engage task set mixture model | FTE 二元混合模型；固定点批评 |
| Q16 | WebSearch | Kane Engle 2003 WM Stroop goal neglect congruency proportion | JEP:G 2003：错误与 RT 干扰分离；75% vs 0% 一致 |
| Q17 | WebSearch | Roth Sant'Anna parallel trends sensitive to functional form | 2010.04814 |
| Q18 | WebSearch | Stroop response set Klein 1964; 2-to-1 mapping De Houwer 2003 | 反应集效应；2:1 映射 |
| Q19 | WebSearch | active placebo sham control mimic side effects unblinding | active placebo；TIDieR-Placebo；第三方盲法检验 |
| Q20 | WebSearch | backdoor defense selection cost comparison FP NAD ANP I-BAU BackdoorBench | 只有基准比较，**没有**按诊断选修复 |
| Q21 | WebSearch | CleanGen decoding-time backdoor defense reference model | 2406.12257 |
| Q22 | WebSearch | backdoored VLM test-time defense suppress target token restore correct answer VQA | CleanSight 2603.12989；BackdoorVLM 2511.18921；TrojVLM 2409.19232 |
| Q23 | WebSearch | seed variance fine-tuning Dodge 2020 | 2002.06305；2006.04884；2302.07778 |
| Q24 | WebSearch | MultiBERTs multiple seeds multi-bootstrap Sellam 2021 | 2106.16163 |
| Q25 | WebSearch | SentiNet inert pattern patch transplant | 1812.00292 |
| Q26 | WebSearch | "BERTs of a feather do not generalize together" McCoy | 1911.02969 |
| Q27 | WebSearch | OOD problem feature importance deletion Hase 2021 ROAR | 2106.00786；ROAR |
| Q28 | WebSearch | "Working-memory capacity and the control of attention" Kane Engle 2003 pdf | UNCG 源拒绝连接；改用 Engle & Kane 2004 书章 |
| Q29 | WebSearch | NaturalBench VQA pairs blind answers | 2410.14669：2 图 × 2 问、答案交替 |
| Q30 | WebSearch | goal reminders eliminate WMC–Stroop errors relationship | Hood & Hutchison 2021 AP&P |
| Q31 | WebSearch | flanker response competition LRP conditional accuracy dual-route Ridderinkhof | CAF；White/Ratcliff/Starns 2011 离散 vs 渐进选择 |
| Q32 | WebSearch | backdoor defense recommendation select defense by attack characteristics | SoK 2511.13143；TED-LaST 2506.10722；**没有**按诊断选修复 |
| Q33 | WebSearch | ANP computational cost comparison clean data 1% runtime | ANP 2110.14430；2405.14781；2407.10052 |
| Q34 | WebSearch | last layer retraining spurious correlations DFR backdoor | 2204.02937；质疑文 2308.00473 |
| Q35 | WebSearch | "difference-in-differences" LLM interpretability prompt intervention | 没有找到在 logit 空间做 DiD 的可解释性方法 |
| Q36 | WebSearch | MM-SHAP multimodal contributions Parcalabescu Frank | 2212.08158 |
| Q37 | WebSearch | Visual Contrastive Decoding distorted image Leng 2023 | 2311.16922 |
| Q38 | WebSearch | placebo test DiD falsification pre-trends HonestDiD | Rambachan & Roth |
| Q39 | WebSearch | Batch Calibration contextual bias mean over batch Zhou 2023 | 2309.17249 |
| Q40 | curl arXiv | 1805.05492 Did the model understand the question? | 在 VQA 问句词上做 IG + overstability 测试 |
| Q41 | WebSearch | fixed-point property mixture distributions RT Falmagne | 固定点检验（BRM 2023） |
| Q42 | WebSearch | Stroop-like interference LLMs cognitive psychology conflict task | 2608.11510；2603.23530 |
| Q43 | WebSearch | LoRA fine-tuning random seed variance LLM | 2503.07329 |
| Q44 | WebSearch | second-choice answer correct suppress top token latent knowledge Hits@k | 2412.20846 |
| Q45 | WebSearch | "goal neglect" language models neural networks | 只找到人类文献；**没有** ML 中用 goal neglect 范式的工作 |
| Q46 | WebSearch | Counterfactual Samples Synthesizing VQA mask critical question words | CSS 2003.06576 |
| Q47 | WebSearch | backdoor VLM clean model with trigger, trigger-only baseline | BackdoorVLM；2608.10959（程序化 VLM 后门，**转交对象层**）；没有 sham patch 标准 |
| Q48 | curl arXiv | 2608.11510 Conflict and Congruency Effects in LLMs | 读摘要并 grep |
| Q49 | curl arXiv | 1612.00837 VQAv2；2204.02937 DFR | 读摘要 |
| Q50 | WebSearch | backdoor mitigation output-layer recalibration subtract target class bias | 2605.08730 分类头偏置捷径；Logit-Margin Repulsion CVPR26；2308.06107 |
| Q51 | WebSearch | backdoor removal cost data efficiency clean samples LLM BackdoorLLM | 2408.12798；Lethe 2508.21004；2508.20032 |
| Q52 | WebSearch | logit difference confounded by final layer norm scale | 只有一般性说明；**没有**专门讨论 logit-diff / DiD 缩放混淆的论文 |
| Q53 | WebSearch | minimal pairs BLiMP | 1912.00582 |
| Q54 | WebSearch | backdoor trigger occludes object evidence, clean model with trigger | 只有一般性结果；2609.15781 把同一 trigger 加到修复后的干净模型上 |

另有 2 次抓取失败，不计入查询数：Kane & Engle 2003 原文（UNCG 超时 / 拒绝连接）、Semantic Scholar API（429）。

---

## 2. 命中表

占位层全部为「方法被占」（本步定义如此）。「读正文」一列：**是** = 下载正文并在方法段做了定向 grep；**摘要** = 下载正文但只读了摘要 / 首页；**否** = 只有检索片段。

| arXiv/DOI | 标题 | 年/会议 | 读正文 | 相关主张 | 占位层 | 判定 | 原文证据句（txt 逐字） |
|---|---|---|---|---|---|---|---|
| 2309.16042 | Towards Best Practices of Activation Patching | ICLR 2024 | 是 | M | 方法 | 必引 | "We find logit difference a convincing metric for localization in language models." / "we recommend avoiding using probability as the metric, given that it may fail to detect negative model components." |
| 2406.16254 | Confidence Regulation Neurons in Language Models | NeurIPS 2024 | 是 | M（F1 的坑） | 方法 | 必引 | "influence the final layer normalization (LayerNorm) scale to effectively scale down the logits." |
| 2010.04814 | When Is Parallel Trends Sensitive to Functional Form? | Econometrica 2023 | 是 | M（F2） | 方法（外领域） | 必引 | "researchers should be careful to give a functional form-specific justification in settings where these conditions are not plausible." |
| 2102.09690 | Calibrate Before Use | ICML 2021 | 是 | M, H3a | 方法 | 必引 | "the model predicts 62% Positive"（N/A 输入）；"a content-free test input such as "N/A"" |
| 2309.17249 | Batch Calibration | ICLR 2024 | 是 | M, H3a（F4） | 方法 | 必引 | "content-free inputs can be inappropriate prior estimators" |
| 2104.08315 | Surface Form Competition（PMI_DC） | EMNLP 2021 | 摘要 | M | 方法 | 必引 | "function that directly compensates for sur-[face form competition] … each option according to its a priori likeli-[hood]"（跨行断词，已按原行拼接） |
| 2305.19148 | Mitigating Label Biases for ICL | ACL 2023 | 否 | M（F4） | 方法 | 必引 | — |
| 2112.07668 | Dual-Key Multimodal Backdoors for VQA | CVPR 2022 | 是 | M（F3、sham、部分 trigger 臂） | 方法（领域内） | 必引 | "we train 8 models per trial, and report the mean ± 2 standard deviations for each metric."；有 Solid / Crop / Optimized 三种 patch，以及 I-ASR、Q-ASR 等只含部分 trigger 的指标 |
| 1812.00292 | SentiNet | IEEE S&P Workshops 2020 | 是 | M（sham） | 方法 | 必引 | "have little effect on the network when replaced by an inert pattern" |
| 1911.02969 | BERTs of a feather do not generalize together | BlackboxNLP 2020 | 摘要 | M（F3） | 方法 | 必引 | "accuracy ranged from 0.0% to 66.2%." |
| 1606.07356 | Analyzing the Behavior of VQA Models | EMNLP 2016 | 是 | M, H1 | 方法 | 必引 | "for 40% of the questions, the CNN+LSTM model seems to have converged on a predicted answer after 'listening' to just half the question." |
| 1612.00837 | Making the V in VQA Matter（VQAv2） | CVPR 2017 | 摘要 | M（F5） | 方法 | 必引 | "a pair of similar images that result in two different answers to the question." |
| 1805.05492 | Did the Model Understand the Question? | ACL 2018 | 是 | P0, M | 方法（在 VQA 问句词上做归因） | 必引 | "We applied IG and attributed the top selected answer class to input question words." / "Our strongest attacks drop the accuracy of a visual question answering model from 61.1% to 19%" |
| 2004.02709 | Contrast Sets | Findings EMNLP 2020 | 摘要 | M | 方法 | 必引 | "Contrast sets provide a local view of a model's decision boundary" |
| 2106.00786 | The OOD Problem in Explainability | NeurIPS 2021 | 摘要 | P0（仪器 B） | 方法 | 必引 | "these counterfactual inputs are out-of-distribution (OOD) to models implies that the resulting explanations are socially misaligned." |
| 2406.12257 | CleanGen | EMNLP 2024 | 是 | M（禁用 token）, E | 方法 | 必引 | "backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents." |
| 2311.16922 | Visual Contrastive Decoding | CVPR 2024 | 摘要 | M, E | 方法 | 部分（sham 当「业余模型」的思路） | "contrasting output distributions from original and distorted visual inputs" |
| 2212.08158 | MM-SHAP | ACL 2023 | 摘要 | P0, H1 | 方法 | 必引（另一种「文字贡献」度量） | "unimodal collapse can oc-[cur]"（跨行断词） |
| 2608.11510 | Conflict and Congruency Effects in LLMs | arXiv 2026-08 | 摘要 + grep | H3, M（Stroop 嫁接） | 方法（外领域范式已被搬入 LLM） | 必引；**交第 3 步做蕴含检索** | "congruency effects in this task arise from competition between an in-weight default mapping and an in-context rule-based mapping." |
| Engle & Kane 2004（PLM 44, 145–199） | Executive attention, WMC, and a two-factor theory of cognitive control | 书章 | 是 | H1/H2 vs H3（F7） | 方法（外领域） | 必引 | "The second factor in the executive control of behavior is the resolution of response competition or conflict" |
| Kane & Engle 2003 | WMC and control of attention: goal neglect, response competition, task set → Stroop | JEP:G 132:47–70 | **否**（抓取失败） | F7 | 方法（外领域） | 必引（引文经 2004 书章转述） | — |
| Duncan et al. 1996 | Intelligence and the frontal lobe | Cogn Psychol 30 | 否 | F7 | 方法（外领域） | 必引（goal neglect 定义） | — |
| De Jong 2000 / Nieuwenhuis & Monsell 2002 | Failure-to-engage | Attention & Performance / PB&R | 否 | F6 | 方法（外领域） | 必引 | — |
| Falmagne 1968；BRM 2023 固定点检验 | Fixed-point property | — | 否 | F6 | 方法（外领域） | 必引 | — |
| Klein 1964；De Houwer 2003 | Response-set effect；2:1 mapping | — | 否 | F7-4、第二阶段 | 方法（外领域） | 必引 | — |
| Hood & Hutchison 2021 | Goal reminders eliminate WMC–Stroop errors | AP&P | 否 | F7-1 | 方法（外领域） | 必引 | — |
| 2204.02937 | Last Layer Re-Training is Sufficient（DFR） | ICLR 2023 | 摘要 | E（输出端修复） | 方法 | 必引 | "they still often learn core features associated with the desired attributes of the data" / "simple last layer retraining can match or outperform state-of-the-art approaches … with profoundly lower complexity and computational expenses." |
| 2605.08730 | Classification-Head Bias in Class-Level Unlearning | arXiv 2026-05 | 摘要 | H3a, E | 方法 | 必引（只调偏置的诊断基线） | "we introduce BiasShift as a diagnostic baseline, showing that simple bias manipulation can satisfy conventional unlearning metrics" |
| 2511.13143 | SoK: The Last Line of Defense | arXiv 2025 | 摘要 + grep | E | 方法 | 必引 | "execution time, memory requirements, and computational cost are almost never reported"（跨行，已拼接）；"readers select appropriate defenses for each use case." |
| 2110.14430 | ANP | NeurIPS 2021 | 否 | E | 方法 | 必引 | — |
| 2110.03735 | I-BAU | ICLR 2022 | 否 | E | 方法 | 必引 | — |
| 2603.12989 | CleanSight | CVPR 2026 | 否（前期报告已读） | E, M | 对象邻近（前期报告已处理） | 必引 | — |
| 2003.06576 | Counterfactual Samples Synthesizing | CVPR 2020 | 否 | M | 方法 | 部分 | — |
| CVPR 2021 | Counterfactual VQA（NDE/TIE） | CVPR 2021 | 否 | M, H3a | 方法 | 必引（「减去问题直接效应」的因果分解） | — |
| 2410.14669 | NaturalBench | NeurIPS 2024 D&B | 否 | M（F5） | 方法 | 部分 | — |
| 1902.05660 | VQA-Rephrasings / Cycle-consistency | CVPR 2019 | 否 | M（F8） | 方法 | 必引 | — |
| ACL 2020 | CheckList | ACL 2020 | 否 | M（F8） | 方法 | 必引 | — |
| 2404.15255 | How to use and interpret activation patching | arXiv 2024 | 否 | M | 方法 | 部分 | — |
| 2106.16163 | MultiBERTs | ICLR 2022 | 否 | M（F3） | 方法 | 部分 | — |
| 2002.06305 | Fine-tuning pretrained LMs: seeds（Dodge） | arXiv 2020 | 否 | M（F3） | 方法 | 部分 | — |
| 2503.07329 | Macro/Micro effects of random seeds on LLM FT | IJCNLP 2025 | 否 | M（F3） | 方法 | 部分（LoRA 方差较低的说法未核） | — |
| 2412.20846 | Are LLMs Really Not Knowledgeable?（Hits@k） | arXiv 2024 | 否 | M（F5） | 方法 | 部分 | — |
| 2308.00473 | Is Last Layer Re-Training Truly Sufficient? | arXiv 2023 | 否 | E | 方法 | 部分 | — |
| 2308.06107 | Test-Time Backdoor Defense via Detecting and Repairing | CVPR 2024 | 否 | E | 方法 | 部分 | — |
| 2408.12798 | BackdoorLLM | arXiv 2024 | 否 | E | 方法 | 部分 | — |
| Roth/Rambachan HonestDiD | An Honest Approach to Parallel Trends | REStud 2023 | 否 | M | 方法（外领域） | 部分 | — |
| TIDieR-Placebo | Reporting placebo and sham controls | PLoS Med 2020 | 否 | M（F4） | 方法（外领域） | 部分 | — |
| 2608.10959 | Once Poisoned, Arbitrarily Controlled: A Programmable Backdoor in VLMs | arXiv 2026-08 | 否 | H3b？ | **可能是对象层，未判** | 交第 1/3/6 步 | — |

---

## 3. 「工具已存在于何处」表

| 我们的方法组件 | 已有出处（方法层） | 要不要改设计 |
|---|---|---|
| 2×2 自然问句因子改写（任务 × 对象） | Contrast sets（2004.02709）；CheckList INV/DIR；VQA-Rephrasings；CSS（Q-CSS 遮掉关键词）；NaturalBench 2×2 图 × 问；BLiMP 最小对 | **要改**（F8）：同义改写 DiD 写进 null；可以借 NaturalBench 的交替答案结构 |
| 用问句扰动检验 VQA 是否在使用问题 | Agrawal 2016（半句问题）；Mudrakarta 2018（在问句词上做 IG + 删词 overstability）；MM-SHAP（T-SHAP） | 不改；**必引** Mudrakarta，它就是「VQA 问句词归因 + 删除验证」的先例（仪器 A+B 的方法层先例） |
| 非攻击候选之间的 logit 差 | Zhang & Nanda 2309.16042；Heimersheim & Nanda 2404.15255 | 不改，引用 |
| DiD 抵消加性偏置 | 计量 DiD；Roth & Sant'Anna 2023；CF-VQA 的「总效应 − 直接效应」 | **要改**（F1、F2）：补缩放校正与序数指标，并预注册刻度 |
| 与问题无关的固定加分基线（H3a） | Contextual calibration（2102.09690）；Batch Calibration（2309.17249，逐类均值偏置）；PMI_DC；BiasShift（2605.08730） | 部分改：用 BC 式「同一图跨问句的均值」估计偏置，比单一内容无关输入更稳（F4）；H3a 的检验要改成「trigger × 问题」交互落在 null 带内，而不是只看拟合好不好 |
| 用 CLEAN / RETRAIN 标定 null 带 | Dual-Key（每格 8 个种子，mean ± 2SD）；McCoy 2020；MultiBERTs；Dodge 2020 | **要改**（F3）：≥5 个独立种子，否则不叫 null 带 |
| sham patch（同位置、同尺寸、非 trigger） | SentiNet 惰性图案；Dual-Key 的 Solid/Crop 对照；医学 active placebo；DiD 安慰剂检验 | **要改**（F4）：多个 sham + CLEAN 上的操纵检查 + 投毒模型上的惰性检查 |
| 禁用攻击 token 后解码 | CleanGen（用参考模型替换可疑 token）；latent knowledge Hits@k；VCD（对比解码） | **要改**（F5）：放在互补图像对上做，并报告问题先验下限 |
| 目标失效 vs 反应竞争的分离 | Kane & Engle 2003 / Engle & Kane 2004；Duncan 1996；De Jong 2000 FTE；Klein 1964；De Houwer 2003；Hood & Hutchison 2021；**ML 内已有** 2608.11510 | **要加**（F6、F7）：目标提醒因子、一致试次促进探针、错误类型谱、固定点混合检验 |
| 删除类归因（仪器 B） | Hase 2021 OOD 问题；ROAR | 路线图已经转向自然改写；引用 Hase 作为「为什么不用删词」的依据 |
| 按诊断选修复（E） | **没有找到**（Q20/Q32/Q50/Q51 共 4 条查询）。相邻：SoK 2511.13143（按用例选防御，要求报告开销）；DFR（输出端 / 末层修复，成本低）；BiasShift；ANP / I-BAU / NAD 成本表（检索片段） | 方向可以保留。基线至少包括 FT、ANP、I-BAU、CleanGen（推理时）、DFR 式末层重训，并按 SoK 要求报告时间、显存、数据量 |

---

## 4. 各主张结论（仅方法层）

- **P0**：用归因看 VQA 问句词、用删词验证，方法层已被 Mudrakarta 2018 占据（在 VQA 问句上做 IG + 删词攻击）。删词的 OOD 问题由 Hase 2021 系统论证过。方法层不致命，但 P0 不能把「对问句做归因」本身当作贡献。
- **H1 / H2**：「问题是否被使用」的行为检验工具在 VQA 语言先验文献里（Agrawal、Goyal、CF-VQA、CSS、MM-SHAP）非常成熟，但它们测的都是**干净模型的语言先验**，没有 trigger 条件。外领域最强的可嫁接工具是认知心理学的「目标失效 vs 反应竞争」两因子分离（Kane & Engle），它恰好给出 H1/H2 与 H3 的操作化判别（错误类型、一致促进、目标提醒）。ML 内已有 2608.11510 把 Stroop 式冲突搬进 LLM（非后门）。**没有找到**有人把 goal-neglect 的判别杠杆（目标提醒、一致促进）用在后门上（Q45、Q47；只是未检出，不等于不存在）。
- **H3a**：「与输入无关的输出端加性偏置」的估计与消除，方法层完全被校准文献覆盖（CC、DC、BC、PMI_DC），不可避免撞车的机器学习领域还有 BiasShift（用在遗忘上）。我们的新意只能落在对象上（后门 trigger 的优势是否等于这样一个偏置），不在方法上。
- **H3b**：没有检索到专门的方法先例；检验方式（trigger × 问题交互）属于标准析因设计。
- **M**：组件在方法层各有出处（见第 3 节）。组合起来的「2×2 自然改写 + 非攻击候选 DiD + 多种子 null + sham + 禁用解码」没有找到完整先例；但 F1–F5 是会被审稿人抓住的具体缺陷，必须在合同冻结前修正。
- **E**：没有找到「先诊断机制、再按诊断选修复」的后门工作（4 条查询）。可以支撑成本差异论证的锚点有：DFR（末层重训成本低）、CleanGen（推理时修复，但需要参考模型）、SoK（要求报告开销）。

仍然没人做（在本步检索范围内）：在后门 VLM 上用目标提醒 / 一致促进 / 错误类型谱区分「任务集失效」和「反应竞争」；在 logit 空间做 DiD 时的缩放混淆检验；以 sham 操纵检查为前提的遮挡 vs 后门分离。

---

## 5. 边界申报

- **没读到的**：Kane & Engle 2003 原文（UNCG 服务器 ECONNREFUSED / 超时），F7 的原文句全部来自同一作者的 2004 书章转述，属于二手引用；如果要在论文里引 2003 的具体数字，需要另取原文。Duncan 1996、De Jong 2000、De Houwer 2003、Klein 1964、Hood & Hutchison 2021、固定点检验、active placebo 文献只看过检索片段。
- **非 arXiv 期刊**（心理学、医学、计量）只经 WebSearch 片段覆盖，没有检索 PsycINFO / PubMed 全库。
- 非英文文献没有检索。
- 在审稿件和未公开的并行工作无法覆盖。2026 年 8–9 月的 arXiv（如 2608.11510、2608.10959）说明这一方向更新很快，知识截止之后的工作一律未知。
- Semantic Scholar API 返回 429，没有做引用链扫描（引用链属于第 5 步）。
- 本步不判对象层 / 结论层。**需要转交的线索**：(1) 2608.11510：在非后门 LLM 里得出「权重内默认映射与上下文规则竞争」的结论，微调强化默认映射会让不一致条件变差，与 H3 同构，第 3 步（蕴含检索）要判断它是否让 H3 成为推论；(2) 2608.10959：程序化 VLM 后门，攻击者在推理时决定目标内容，可能涉及「文字调节后门」（H3b），需要第 1 / 6 步取正文判断。
- 正文 txt 保存在 /home/ubuntu/.claude/jobs/42ebb93b/tmp/rw/s2txt/（22 个），PDF 已全部删除。
