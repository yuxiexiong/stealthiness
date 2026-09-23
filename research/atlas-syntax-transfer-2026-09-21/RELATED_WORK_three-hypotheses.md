# 查重报告：视觉后门下"问题文字作用"的三个假设

日期：2026-09-23　协议：CLAUDE.md §4 七步，执行顺序 6 → 5 → 1/3/4 → 2 → 7（对抗轮前置）。
对应设计：[EXPERIMENT_DESIGN_three-hypotheses.md](EXPERIMENT_DESIGN_three-hypotheses.md)。上游课题定义：[RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md)。

---

## 0. 结论先行（坏消息在前）

1. **H1（文字整体失效）不是新解释，测出来同行不会意外。** 同一对象（后门 LLaVA-1.5 + VQA）上，CleanSight 与 BYE 已把它写成机制解读；LLM 后门与图像劫持工作把"trigger 使后续输入无关"当作常识。**但他们的证据都只有注意力分布，没有人用问题语义干预检验过"模型是否还在用问题"。**
2. **H3（被压过）的形态已有单例先例。** CleanSight Fig. 7 在同一对象上用 logit lens 显示：带毒样本的正确答案在中间层仍占优，晚层被目标词翻转。**单个问句、无问题操纵、无 null 标定**；"问题信息仍按正常方式推动正确答案"这句话没人写出。
3. **CleanSight 同一篇论文里两种叙事并存且未被调和**：注意力被偷走、文字失效（H1）vs 正确答案照常形成、晚层被翻（H3）。口径不同（注意力份额 vs 输出 logit），**不能登记为载重矛盾**，但"用行为干预裁决两者"是一个明确的空位。
4. **H3a（固定加分）在模板层有大量"线性方向"类比，也有反证**：CNN 上重训分类头去不掉后门（Trap&Replace）；LLM 后门机制集中在早层 MLP（Lamparth）。H3a 既未被蕴含也未被检验，只能靠实验。
5. **H2（对象保留、任务失效）与 H3b（问题调节后门强度）在对象层和结论层都是空的**；只在攻击设计侧出现过相近行为（攻击者刻意制造）。
6. **P0（问题文字归因重排、正确方向保持）未见被占**；邻居测的都是注意力份额或 trigger 自身的归因。
7. **方法：每个组件都已有出处（方法层被占，不致命），组合未见。** 必须引用 logit margin 上的 DiD（Wrong Before Right）、模型×输入 2×2 交互分解（2608.30403）等。
8. **效率方向：各类修复都已存在，"先诊断再选修复并比总成本"未见。**

**定位含义（推导档，只经过本轮检索）：** H1、H3 不能当作新现象写。可辩护的最强版本是 —— **首次用问题语义的因子化干预，在对象层裁决"文字失效 vs 被压过"，并量化正确答案通路的保留率；H2 与 H3b 是真正的空位。**

## 1. 检索规模与"读过"的口径

| 步 | 执行者 | 编号查询 | 进入判读 | 取正文并定向追问 |
|---|---|---:|---:|---:|
| 6 对抗式 | 子代理 A | 57 | ~45 | 30（逐字引证 15） |
| 5 引用链 | 子代理 B | 16 次 API（6 种子一跳 + 4 篇二跳） | 102（粗筛命中，含重复） | 16（精读约 10） |
| 1/3/4 模板·蕴含·同义词 | 子代理 C | 69 | ~48 | 21 |
| 2 方法论 | 子代理 D | 54 | 48 | 22（定向追问 13） |
| 汇总核验 | 本人 | — | — | 本文引用的原文句 32 条，全部逐条 `pdftotext` + 文本检索核对（双栏交错处人工拼读） |
| **合计** | | **约 196 条编号查询 + 16 次引用链 API** | | **去重后 68 份正文（67 篇论文 + 1 篇书章）** |

去重后取到正文 68 份（67 篇论文 + 1 篇书章，清单见附录）。**只有取到正文并定向追问的才计为"读过"**；检索片段与摘要不计。一处更正：检索摘要器称 2411.12701 含 "forced override"，正文中不存在，未采用。

引用链种子与规模：

| 种子 | 参考 | 被引 | 粗筛命中 |
|---|---:|---:|---:|
| CleanSight 2603.12989 | 76 | 5 | 18 |
| Backdoor Attribution 2509.21761 | 51 | 10 | 20 |
| Triggers Hijack Language Circuits 2602.10382 | 21 | 1 | 8 |
| BackdoorVLM 2511.18921 | 48 | 5 | 13 |
| Revisiting Backdoor Attacks against LVLMs from Domain Shift 2406.18844 | 68 | 56 | 22 |
| Overthinking the Truth 2307.09476 | 58 | 93 | 21 |

## 2. 同义词表（第 4 步）

- 攻击：backdoor, trojan, data poisoning, sleeper agent, hidden trigger, image hijack, adversarial image, visual prompt injection, typographic attack, jailbreak image
- 失效：ignore / neglect / disregard the question, instruction-following failure, text grounding loss, modality dominance, language prior, shortcut, override, hijack, task drift
- 竞争：answer / logit / response competition, suppression, latent knowledge, knows-but-says, overthinking, late-layer override, output bias
- 对象：LVLM, MLLM, VLM, LLaVA, VQA

## 3. 逐主张判定

占位分三层：方法被占（不致命）/ 对象被占（致命）/ 结论被占（致命）。

### P0 现象：问题文字的输入归因重排，正确答案方向保持

**判定：未被占。** 邻居测的量不同：
- CleanSight、BYE、BackdoorVLM、TCAP（2601.21692）：文字侧注意力份额下降；TCAP 被抑制的是 system 指令，不是用户问题。
- Lyu 博士论文 2608.18095：BERT 文本 trigger，归因漂向 trigger token 本身。
- RACER 2608.24354：图像 trigger 的表示异常集中在视觉 token（类比档，与 P0 测量不同，不登记矛盾）。

没人做的：非 trigger 问题词的输入归因（梯度与删除两仪器）、按攻击/正确两输出方向拆分、LABEL-only 对照、与干净模型配对。

### H1 文字整体失效

**判定：对象层被占（解读级），检验空白。**

- CleanSight 2603.12989（后门 LLaVA-1.5，同对象）："This shift suggests that the trigger redirects the model's attention from the textual context to the visual regions, thereby weakening instruction following of the backdoored LVLM." —— 依据：注意力份额。
- BYE 2505.16916："severing the connection between visual grounding and instruction following, and leading to backdoored outputs that disregard the intended reasoning pathway." —— 依据：注意力熵。
- 模板层：The Trigger in the Haystack 2602.03085（LLM）"rendering the subsequent prompt irrelevant"；Image Hijacks 2309.00236 "regardless of user input"；When Backdoors Speak 2411.12701 "largely ignoring the input sample"；Phantasia 2604.08395 将现有 VLM 后门描述为 "fixed patterns conditioned solely on the trigger"；STRIP 1902.06531 的检测原理即扰动后输出不敏感。

**干净基线已有答案（直接影响设计）：** 干净 VQA 模型本来就少用问题。
- Mudrakarta et al. 1805.05492："the network ignores many question words, relying largely on the image"；保留 "color" 一个词即可达到最终准确率的 50% 以上。
- Agrawal et al. 1606.07356：模型只"听"半个问题就收敛到答案。
- Goyal et al. 1612.00837（VQAv2）：模型过度依赖语言先验。

⇒ 问题敏感度的零点绝不能假定；干净模型的 C_task 可能本来就小。

**若我们测出 H1：** 只能写成"首次对已有解读做行为检验"，意外性低。
**若测出 H2/H3：** 与上述解读冲突 —— 但登记矛盾前必须先对齐口径（注意力 ≠ 信息使用）。

### H2 文字部分失效（对象保留、任务失效）

**判定：对象层与结论层都空。** 只在攻击设计侧出现：TokenSwap 2509.24566 刻意让模型 "mention the correct objects in the image but misrepresent their relationships"。提示注入的 task drift（2406.00799）是模板层邻居。Mudrakarta 显示干净模型高度依赖任务词（"color"、"many"），这使 H2 的检验在干净模型上有充分的对比基线。

### H3 被压过

**判定：对象层有单例先例，结论句未被写出。**

- CleanSight 2603.12989 Fig. 7（同对象）："clean samples steadily reinforce the correct response from middle layers (11th-32nd), while poisoned ones gradually invert this preference in late layers (24th-32nd), revealing that triggers act mainly after cross-modal fusion." —— **图中只有一个问句（"Is this rice noodle soup?"），正文未说明对多题平均。**
- 模板层很密：Overthinking the Truth 2307.09476（正确答案在"critical layer"之后被覆盖）；Competition of Mechanisms 2402.11655；MLLMs Get It Right, Then Get It Wrong 2606.17953（"models often get it right initially, forming correct vision-based predictions in their intermediate layers, before changing their minds and favoring text in the final output"）；JRS 2603.17372（同为 LLaVA-1.5-7B 的越狱："jailbreaks do not arise from a failure to recognize harmful intent"）；Distributed triggers 2407.04151（猜想级）。
- 需正面回应的相邻证据：2608.30403（LLM 文本分类，SAE）发现后门激活时存在 "Suppressed features, active in clean conditions but suppressed during backdoor activation"。**这与 H3 不直接冲突**：原文未说明这些特征是否承载问题信息；它与 H1、H3 都相容（类比档）。其报告的"模型×trigger 交互占 86.6%"量的是后门本身的非加性，不是问题依赖，不能当作 H3 的反证。

**空位：** (i) 问题语义是否仍按正常方式推动正确答案（改任务/改对象时的保留率）；(ii) 裁决 CleanSight 内部两种叙事。

### H3a 固定加分

**判定：未被检验；模板层有类比与反证。**

- 类比（都是残差流/投影层的方向，不是输出端常数，也都没检验与问题无关）：Backdoor Attribution 2509.21761（"backdoor triggering resembles a switch operation"）；Backdoor Directions in ViTs 2603.10806（"a single linear direction in the models' residual stream modulates backdoor behavior"）；ProjLens 2604.19083（"shift toward a shared direction aligned with the backdoor target, but the shifting magnitude scales linearly with the input norm"）；Shared Latent Structures 2606.07963；Backdoor Unlearning by Linear Task Decomposition 2510.14845；Language-Switching Triggers 2605.18646（最后一层 MLP 才转成输出）。
- 反证：Trap and Replace 2210.06428："the model will not effectively unlearn the backdoor correlations even we retrain the classification head from scratch"（ASR 仍高达 99.99% / 51.87%）；Lamparth & Reuel 2302.12461："early-layer MLP modules as most important for the backdoor mechanism"。
- 注意（推导档）："在晚层起作用"不等于"与问题无关的常数偏置"；ProjLens 的"幅度随输入范数线性变化"本身就是一种输入依赖。

### H3b 问题调节后门强度

**判定：诊断侧空白。** 邻近工作都在攻击侧：Revisiting…Domain Shift 2406.18844（"domain shift between attacker's and user's instructions may prevent trigger activation"，只针对文本 trigger，看 ASR 而非 logit 优势）；Phantasia 2604.08395、BadSem 2506.07214 刻意让输出依赖问题；Dual-Key Multimodal Backdoors 2112.07668 需要图像与问题双 trigger 同时出现（设计层的问题依赖）。固定 patch + 固定答案下"问题内容是否调节后门强度"无人测过，按题型拆 ASR 的分析也未检到。

### M 方法

见第 5 节"工具已存在于何处"。组合未见；非攻击候选之间的 DiD、禁用攻击 token 解码作为诊断、重训模型标定 null 带，未检到有人在后门诊断中使用。

### E 效率方向

各类修复都已存在：输入端（CleanSight、TrustVLA 2607.12571、RACER 2608.24354）、输出端（CleanGen 2406.12257，需参考模型）、参数端（BYE 2505.16916、Dummy Backdoor 2606.11648、Backdoor Attribution 向量减法、JRS-Rem 2603.17372）。**"先诊断失效类型再选修复并比总成本"未检到。** 若 H3 成立，JRS-Rem 与 CleanSight 是同模型上的必比基线。

## 4. 与既有课题材料的关系

- MECHANISM_ROADMAP.md 第 7 节已列 CleanSight、Backdoor Attribution、Triggers Hijack、NOTICE、Activation Patching 最佳实践。本报告新增的必引：BYE、Mudrakarta、2608.30403、2606.17953、Trap and Replace、Lamparth、ProjLens、Wrong Before Right、Dual-Key、JRS。
- 旧查重 `attribution-microscope/NOVELTY_REPORT.md`（2026-09-19）查的是另六个现象，与本报告不重叠。

## 5. 方法论（第 2 步）：工具已存在于何处

方法层被占不致命；下表说明每个组件已有的出处，以及它如何改了设计。详细修正见设计文档第 12 节。

| 我们的组件 | 已有出处 | 对设计的影响 |
|---|---|---|
| 2×2 自然问句改写 | Contrast sets 2004.02709；CheckList、VQA-Rephrasings、NaturalBench（未读正文） | 同义改写差异并入 null（F8） |
| 问句词归因 + 删词验证（仪器 A/B） | Mudrakarta et al. 1805.05492（问句词 IG 与删词攻击）；Hase et al. 2106.00786（删词造成分布外输入） | **P0 不能把"对问句做归因"当贡献**；贡献只能落在后门对象与配对设计上 |
| 非攻击候选之间的 logit 差 | Zhang & Nanda 2309.16042；Heimersheim & Nanda（未读正文） | 不改，引用 |
| logit margin 上的 DiD | 计量 DiD；Wrong Before Right 2607.04640（"The DiD construction cancels item-level lexical frequency and phrasing effects that contaminate raw margins."） | 引用；**DiD 只抵消加性偏置**，末层 RMSNorm 缩放需另行处理（F1，Stolfo et al. 2406.16254："influence the final layer normalization (LayerNorm) scale to effectively scale down the logits"）；刻度必须预先写死（F2，Roth & Sant'Anna 2010.04814："be careful to give a functional form-specific justification"） |
| 模型 × 输入 2×2 交互分解 | 2608.30403（SAE，LLM 后门） | 引用；我们的交互轴是 Trigger × 问题，不是模型 × Trigger |
| 固定加分基线（H3a） | Calibrate Before Use（未读正文）；Batch Calibration 2309.17249（"content-free inputs can be inappropriate prior estimators"）；BiasShift 2605.08730 | 判据改为"Trigger × 问题"交互落在 null 内；**H3a 在方法层完全被校准文献覆盖，新意只在对象** |
| null 带标定 | Dual-Key 2112.07668（"train 8 models per trial, and report the mean ± 2 standard"）；McCoy et al. 1911.02969（同分布准确率相近的种子，分布外 "accuracy ranged from 0.0% to 66.2%"）；MultiBERTs 2106.16163（未读正文） | 干净种子从 3 个扩到 5 个（F3） |
| 假 patch 对照 | SentiNet 1812.00292（惰性图案替换）；Dual-Key 的 Solid/Crop trigger；医学 active placebo（仅检索片段） | 三种匹配假 patch + 操纵检查（F4） |
| 禁用目标 token 解码 | CleanGen 2406.12257；VCD 2311.16922 | 恢复须 A、B 两问同时答对，排除问题先验（F5，Agrawal 1606.07356、VQAv2 1612.00837） |
| 目标失效 vs 反应竞争 | 认知心理学：Engle & Kane 2004 书章（"The second factor in the executive control of behavior is the resolution of response competition or conflict"）；Kane & Engle 2003（原文未取到）；De Jong、Klein（仅检索片段）；LLM 内：2608.11510（"competition between an in-weight default mapping and an in-context rule-based mapping"，非后门） | **跨领域嫁接**：任务提醒因子与交互 I、错误类型谱（F7，类比档） |
| 按诊断选修复（E） | 4 条定向查询未检到；相邻：SoK 2511.13143、CleanGen；ANP、I-BAU（未读正文） | 方向保留；基线须报时间、显存、数据量 |

2608.11510 在非后门设置下给出与 H3 同构的"默认映射 vs 规则映射竞争"解释：它属于模板层与方法层，**不使 H3 在后门 VLM 上成为推论**（对象不同，且未涉及问题语义保留率），列为必引。

第 2 步规模：54 条编号查询，进入判读 48 篇，取到正文 22 篇（定向追问 13 篇，9 篇只读摘要/首页）。

## 6. 边界申报（第 7 步）

- **未覆盖：** 在审稿件、非 arXiv 会议正文、非英文文献（含中文）、并行未公开工作。
- **引用链不完整：** Semantic Scholar 对 2026 年论文的被引严重滞后；CleanSight 只有 5 条被引可见 —— 2026-03 之后可能已有跟进工作做了问题改写而未被索引。**这是最该在开跑前补查的一处。**
- **检索器限制：** 子代理 C 的 12 条 arXiv 多字段组合查询零命中（检索器过严）；Semantic Scholar 多次 429 限流。
- **只看了摘要、不能用于判生死：** Calibrate Before Use、Seeing but Not Believing 2510.17771、2508.02087、2601.07359、CS-ADS（ACL'26）、Logit-Margin Repulsion（CVPR'26）、Function Vectors 2310.15916。
- **知识截止与索引：** 2026-08 之后的 arXiv 可能索引不全。
- **存活 ≠ 强度（准则 5.2）：** H2、H3b 的"空位"只经过一轮四路检索（合计约 196 条编号查询），不比死在后续轮次的想法更强，只是搜得更少。

## 附录：取到正文的论文

标题取自 PDF 首页抽取文本的前 120 字符（可能带作者名），用于定位，不作引用格式。

| ID | 首页文本 |
|---|---|
| 1606.07356 | Analyzing the Behavior of Visual Question Answering Models Aishwarya Agrawal∗ , Dhruv Batra†,∗ , Devi Parikh†,∗ ∗ Virgin |
| 1612.00837 | Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering Yash Goyal∗1 Tejas Kh |
| 1805.05492 | Did the Model Understand the Question? Pramod K. Mudrakarta Ankur Taly Mukund Sundararajan Kedar Dhamdhere University of |
| 1812.00292 | SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems Edward Chou Florian Tramèr Giancarlo Pell |
| 1911.02969 | BERTs of a feather do not generalize together: Large variability in generalization across models with similar test set p |
| 2004.02709 | Evaluating Models’ Local Decision Boundaries via Contrast Sets Matt GardnerF♦ Yoav ArtziΓ Victoria Basmova♦♣ Jonathan Be |
| 2010.04814 | When Is Parallel Trends Sensitive to Functional Form?∗ Jonathan Roth† Pedro H.C. Sant’Anna‡ arXiv:2010.04814v5 [econ.EM] |
| 2102.09690 | Calibrate Before Use: Improving Few-Shot Performance of Language Models Tony Z. Zhao * 1 Eric Wallace * 1 Shi Feng 2 Dan |
| 2104.08315 | Surface Form Competition: Why the Highest Probability Answer Isn’t Always Right = Ari Holtzman1 = Peter West1,2 Vered Sh |
| 2106.00786 | The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations Peter Hase, Har |
| 2112.07668 | Dual-Key Multimodal Backdoors for Visual Question Answering Matthew Walmer1 * Karan Sikka2 Indranil Sur2 Abhinav Shrivas |
| 2204.02937 | Published as a conference paper at ICLR 2023 L AST L AYER R E -T RAINING IS S UFFICIENT FOR ROBUSTNESS TO S PURIOUS C OR |
| 2210.06428 | Trap and Replace: Defending Backdoor Attacks by Trapping Them into an Easy-to-Replace Subnetwork Haotao Wang Junyuan Hon |
| 2212.08158 | MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks Leti |
| 2302.12461 | Analyzing And Editing Inner Mechanisms of Backdoored Language Models Max Lamparth∗ Anka Reuel Stanford University Stanfo |
| 2307.09476 | Published as a conference paper at ICLR 2024 OVERTHINKING THE T RUTH : U NDERSTANDING HOW L ANGUAGE M ODELS P ROCESS FAL |
| 2309.00236 | Image Hijacks: Adversarial Images can Control Generative Models at Runtime Luke Bailey * 1 Euan Ong * 2 Stuart Russell 3 |
| 2309.16042 | T OWARDS B EST P RACTICES OF ACTIVATION PATCH - ING IN L ANGUAGE M ODELS : M ETRICS AND M ETHODS Fred Zhang∗ Neel Nanda  |
| 2309.17249 | Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering Han Zhou1,3 , Xingchen Wan1 , L |
| 2311.16922 | Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding Sicong Leng1,2, * H |
| 2402.06659 | Shadowcast: Stealthy Data Poisoning Attacks against Vision-Language Models Yuancheng Xu1 Jiarui Yao2 3 Manli Shu Yanchao |
| 2402.08577 | Test-Time Backdoor Attacks on Multimodal Large Language Models Dong Lu * 1 Tianyu Pang * 2 Chao Du 2 Qian Liu 2 Xianjun  |
| 2402.11655 | Competition of Mechanisms: Tracing How Language Models Handle Facts and Counterfactuals Francesco Ortu∗ Zhijing Jin∗ Die |
| 2402.13851 | VL-Trojan: Multimodal Instruction Backdoor Attacks against Autoregressive Visual Language Models Jiawei Liang Siyuan Lia |
| 2403.02910 | ImgTrojan: Jailbreaking Vision-Language Models with ONE Image Xijia Tao* , Shuai Zhong∗ , Lei Li∗ , Qi Liu, Lingpeng Kon |
| 2404.12916 | Physical Backdoor Attack can Jeopardize Driving with Vision-Large-Language Models Zhenyang Ni1,4 Rui Ye1,4 Yuxi Wei1,4 Z |
| 2406.00799 | Get my drift? Catching LLM Task Drift with Activation Deltas Sahar Abdelnabi1∗ Aideen Fay1∗ Giovanni Cherubin1 Ahmed Sal |
| 2406.12257 | C LEAN G EN: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models WARNING: This paper contains mode |
| 2406.16254 | Confidence Regulation Neurons in Language Models Alessandro Stolfo∗ Ben Wu∗ Wes Gurnee ETH Zürich University of Sheffiel |
| 2406.18844 | Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift Siyuan Liang1 , Jiawei Liang2 , Tiany |
| 2407.04151 | Securing Multi-turn Conversational Language Models From Distributed Backdoor Triggers Terry Tong Jiashu Xu Qin Liu Muhao |
| 2409.19232 | TrojVLM: Backdoor Attack Against Vision Language Models Weimin Lyu, Lu Pang, Tengfei Ma, Haibin Ling, and Chao Chen Ston |
| 2410.01264 | Published as a conference paper at ICLR 2025 BACKDOORING V ISION -L ANGUAGE M ODELS WITH O UT-O F -D ISTRIBUTION DATA We |
| 2411.12701 | When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations Huaizhi Ge1 , Yiming Li2 , |
| 2505.16916 | Backdoor Cleaning without External Guidance in MLLM Fine-tuning Xuankun Rong1† , Wenke Huang1† , Jian Liang1 , Jinhe Bi2 |
| 2506.07214 | Backdoor Attack on Vision Language Models with Stealthy Semantic Manipulation Zhiyuan Zhong1,2 Zhen Sun2 Yepang Liu3 Xin |
| 2508.15847 | Mechanistic Exploration of Backdoored Large Language Model Attention Patterns M. Abu Baker∗ L. Babu-Saheer† Abstract Bac |
| 2509.21761 | Preprint BACKDOOR ATTRIBUTION : E LUCIDATING AND C ON - TROLLING BACKDOORS IN L ANGUAGE M ODELS Miao Yu1,† , Zhenhong Zh |
| 2509.24566 | TokenSwap: Backdoor Attack on the Compositional Understanding of Large Vision-Language Models Zhifang Zhang * 1 2 Qiqi T |
| 2510.14845 | Preprint BACKDOOR U NLEARNING BY L INEAR TASK D ECOMPOSITION Amel Abdelraheem∗ † 1 Alessandro Favero∗ 1 Gérôme Bovet2  |
| 2511.13143 | SoK: The Last Line of Defense: On Backdoor Defense Evaluation Gorka Abad Marina Krček Stefanos Koffas Behrad Tajalli Ma |
| 2511.18921 | BackdoorVLM: A Benchmark for Backdoor Attacks and Defenses on Vision-Language Models Juncheng Li Yige Li✉ Hanxun Huang C |
| 2601.21692 | TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning Mingzu Liu 1 2 3 * Hao F |
| 2602.03085 | The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers Blake Bullwinkel * 1 Giorgio Severi * 1 |
| 2602.10382 | Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models Théo  |
| 2602.22246 | Self-Purification Mitigates Backdoors in Multimodal Diffusion Language Models Guangnian Wan, Qi Li, Gongfan Fang, Xinyin |
| 2603.10806 | Backdoor Directions in Vision Transformers Sengim Karayalçin1 , Marina Krček2 , and Pin-Yu Chen3 Stjepan Picek4,2 1 Le |
| 2603.12989 | Test-Time Attention Purification for Backdoored Large Vision Language Models Zhifang Zhang1 Bojun Yang2 Shuo He3 Weitong |
| 2603.17372 | Understanding and Defending VLM Jailbreaks via Jailbreak-Related Representation Shift Zhihua Wei1 * , Qiang Li1 * , Jian |
| 2604.08395 | Phantasia: Context-Adaptive Backdoors in Vision Language Models Nam Duong Tran 1 Phi Le Nguyen 1 1 Institute for AI Inno |
| 2604.19083 | ProjLens: Unveiling the Role of Projectors in Multimodal Model Safety Kun Wang3,∗ Cheng Qian2,∗ Miao Yu1,∗ Lilan Peng4 L |
| 2604.24162 | Defusing the Trigger: Tail-Risk-Informed Attention Rebalancing for LLM Backdoor Mitigation Kaisheng Fan1 , Yishu Gao1 ,  |
| 2605.08730 | 1 Classification-Head Bias in Class-Level Machine Unlearning: Diagnosis, Mitigation, and Evaluation Weidong Zheng, Kongy |
| 2605.18646 | Language-Switching Triggers Take a Latent Detour Through Language Models Francis Kulumba1, 2 Wissam Antoun1,2 Théo Lasni |
| 2606.02947 | BYORn: Bootstrap Your Own Responses to Defend Large Vision-Language Models Against Backdoor Attacks Ivan Sabolić 1 Mari |
| 2606.07963 | Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs Omar Mahmoud∗ , Aly M. Kassem§ , Thomm |
| 2606.11648 | Dummy Backdoor as a Defense: Removing Unknown Backdoors via Shared Internal Mechanisms for Generative LLMs Kazuki Iwahan |
| 2606.17953 | MLLMs Get It Right, Then Get It Wrong: Tracing and Correcting Late-Layer Textual Bias Xingming Li1 , Ao Cheng1 , Qiyao S |
| 2607.04640 | Wrong Before Right: Late Rescue and Interface Failure in Aligned Language Models Jiaqi Deng Independent Researcher djq62 |
| 2607.12571 | TrustVLA: Mechanism-Guided Inference-Time Defense Against Vision-Language-Action Backdoors Pinhan Fu1,∗ , Xianda Guo1,∗, |
| 2608.10959 | Once Poisoned, Arbitrarily Controlled: A Programmable Backdoor in VLMs Tao Lin1,2,3 , Gaojie Jin4 , Zongxin Liu1,2,3 , P |
| 2608.11510 | Conflict and Congruency Effects in Large Language Models: In-Weight and In-Context Competition in a Verbal Conflict Task |
| 2608.18095 | Backdoor Learning in Language Models and Vision-Language Models A dissertation presented arXiv:2608.18095v1 [cs.CL] 8 Ju |
| 2608.24354 | Not All Tokens Are Equal: Region-Aware Consistency Repair of Backdoors in MLLMs Jiali Wei† , Ming Fan†* , Mingkun Zhang† |
| 2608.30403 | Why Are LLM Backdoor Defenses Fragmented? A Feature-Level Explanation with Sparse Autoencoders Yizhe Zeng1,2 , Chenxu Ni |
| 2609.02000 | Who Drives the Probability Game of VLMs? A Temporal Causal Drive Evaluation Framework Shuyao Xiao1,2 * Shengling Wang1†  |
| 2609.07746 | LLM Forensics: Where Do Backdoors Hide? Localizing and Controlling Trigger Mechanisms with Sparse Autoencoders Wissam An |
| engle_kane_2004 | EXECUTIVE ATTENTION, WORKING MEMORY CAPACITY, AND A TWO-FACTOR THEORY OF COGNITIVE CONTROL Randall W Engle and Michael J |
