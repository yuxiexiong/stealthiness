# 第 5 步：引用链挖掘（2026-09-23）

工作文件：`s5/`（S2 原始 JSON、`screen.txt` 关键词粗筛、`txt/` 16 篇正文文本；PDF 已全部删除）。

## 0. 坏消息先说

1. **H1 的结论句已经有人写过，但只是解释，没做检验。** CleanSight 和 BYE 都从 attention 现象直接推出"trigger 削弱了指令/问题的使用"，两篇都没做因果检验。如果我们测出 H1 成立，贡献只是给已有说法补上证据，拿不到"新结论"。
   - CleanSight 2603.12989 L97–99："the trigger redirects the model's attention from the textual context to the visual regions, thereby weakening instruction following of the backdoored LVLM."
   - BYE 2505.16916 L307–309："severing the connection between visual grounding and instruction following, and leading to backdoored outputs that disregard the intended reasoning pathway."
2. **H3 的层级形态已有单例证据。** CleanSight 的 logit lens 图，同用 LLaVA-1.5 + VQAv2：干净样本在中层持续强化正确答案，投毒样本在后层（24–32）把偏好翻过去。他们据此的解读是"trigger 主要在融合之后起作用"（L462–468）。这只是一个样例、一张图，没有问题因子，也没有 null 标定，但审稿人会拿它说"H3 已被观察到"。
3. **LLM 侧有与 H3 方向相反的证据（类比档）。** 2608.30403 在文本分类后门上做了 2×2 因子分解（干净/投毒模型 × 干净/触发输入），发现一类 SUPPRESSED 特征。这类特征在正常条件下激活，到 PT 条件下塌缩，作者解读为"source-class functionality that is suppressed during backdoor activation"。这对应"正确答案的计算被压制"，不是"仍在、只是被压过"。
4. **M 的核心技巧在方法层已有人用过（不致命）。** 2607.04640 在 minimal pairs 上用 logit margin 做 DiD，明说用它抵消逐项偏置；2608.30403 在 logit-margin 空间做 2×2 交互分解。

## 1. 查询日志

| # | 查询 | 来源 | 结果 |
|---|---|---|---|
| Q0a–f | 6 个种子的元数据（title/refCount/citCount） | S2 Graph API | 首轮全部 429；退避重试后全部确认 ID（见下） |
| Q1 | arXiv:2603.12989 /references | S2 | 76 |
| Q2 | arXiv:2603.12989 /citations | S2 | 5 |
| Q3 | arXiv:2509.21761 /references | S2 | 51 |
| Q4 | arXiv:2509.21761 /citations | S2 | 10 |
| Q5 | arXiv:2602.10382 /references | S2 | 21 |
| Q6 | arXiv:2602.10382 /citations | S2 | 1 |
| Q7 | arXiv:2511.18921 /references | S2 | 48 |
| Q8 | arXiv:2511.18921 /citations | S2 | 5 |
| Q9 | arXiv:2406.18844 /references | S2 | 68 |
| Q10 | arXiv:2406.18844 /citations | S2 | 56 |
| Q11 | arXiv:2307.09476 /references | S2 | 58 |
| Q12 | arXiv:2307.09476 /citations | S2 | 93 |
| Q13 | 二跳：arXiv:2505.16916 (BYE) /citations | S2 | 37 |
| Q14 | 二跳：arXiv:2604.19083 (ProjLens) /citations | S2 | 0（S2 尚未收录被引） |
| Q15 | 二跳：arXiv:2608.24354 (RACER) /citations | S2 | 0 |
| Q16 | 二跳：arXiv:2608.30403 (SAE 碎片化) /citations | S2 | 0 |
| K1 | 对 Q1–Q12 全部标题与摘要做正则粗筛：(backdoor/trojan/poison/trigger) ∧ (question/instruction/prompt/attention/attribution/interpret/mechanis/logit/bias/defen/purif/repair/mitigat/unlearn/circuit/conditional/context/semantic/steer)，外加 override/ignore the question/knowledge conflict/logit lens/early exit/suppress 等机制词 | 本地 | 163 条（宽筛，见 `s5/screen.txt`） |
| M1 | 在 K1 基础上逐条人工复筛 | 本地 | 102 条（含种子间重复），按种子列于 §1.1 |
| F1–F16 | 取正文：arXiv PDF → pdftotext → 定向 grep | arxiv.org | 16 篇，见命中表"读正文=Y" |

种子 ID 核对（S2 返回标题）：
- 2603.12989 = *Test-Time Attention Purification for Backdoored Large Vision Language Models*（CleanSight）
- 2509.21761 = *Backdoor Attribution: Elucidating and Controlling Backdoor in Language Models*
- 2602.10382 = *Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in LLMs*（ICML'26 MI workshop）
- 种子 4 选了两篇：4a 2511.18921 = *BackdoorVLM: A Benchmark for Backdoor Attacks and Defenses on VLMs*；4b 2406.18844 = *Revisiting Backdoor Attacks against LVLMs from Domain Shift*（CVPR'25）
- 2307.09476 = *Overthinking the Truth*（ICLR'24）

### 1.1 每个种子的统计

| 种子 | 参考文献数 | 被引数 | 正则粗筛 | 人工粗筛命中 |
|---|---|---|---|---|
| CleanSight 2603.12989 | 76 | 5 | 25+1 | 18 |
| Backdoor Attribution 2509.21761 | 51 | 10 | 18+9 | 20 |
| Triggers Hijack 2602.10382 | 21 | 1 | 6+1 | 8 |
| BackdoorVLM 2511.18921 | 48 | 5 | 14+4 | 13 |
| Revisiting LVLM backdoor 2406.18844 | 68 | 56 | 18+33 | 22 |
| Overthinking the Truth 2307.09476 | 58 | 93 | 6+28 | 21 |
| **合计（含重复）** | **322** | **170** | **163** | **102** |
| 二跳 BYE / ProjLens / RACER / SAE | — | 37 / 0 / 0 / 0 | — | 3（TrustVLA 2607.12571、BYORn 2606.02947、When Attention Betrays 2602.03153） |

人工粗筛命中（按种子；† 表示取了正文）：
- **CleanSight**：BYE 2505.16916†、TokenSwap 2509.24566、RVPT 2412.20392、Cross-modal Info Flow 2411.18620、2410.07149、2410.02762、VLOOD 2410.01264、TrojVLM 2409.19232、VL-Trojan 2402.13851、BadVLMDriver 2404.12916、TeCo、ZIP 2303.12175、BDMAE、SampDetox、Neural Polarizer 2306.16697、Fine-Pruning 1805.12185、Label-Consistent 1912.02771；被引：RACER 2608.24354†
- **Backdoor Attribution**：参考文献：2506.16447、When Backdoors Speak 2411.12701、JailbreakLens 2411.11114、2410.13708、BEEAR 2406.17092、LAT 2407.15549、Sleeper Agents 2401.05566、Function Vectors 2310.15213、ROME 2202.05262，以及 2 篇综述；被引：2608.30403†、2606.20254、2606.07963、2606.03785、2604.24542、TIARA 2604.24162†、ProjLens 2604.19083†、ReShift 2607.00361、SafeSeek 2603.23268
- **Triggers Hijack**：2508.15847、2510.07192、Refusal Direction 2406.11717、2310.15213、2305.00944、Piccolo、2401.05566；被引：2606.03785
- **BackdoorVLM**：BadSem 2506.07214、VL-Trojan、VLOOD、2406.18844†、2303.15180、BYE、BadToken、Dual-Key 2112.07668、Grad-CAM、Shortcut 2004.07780；被引：FreqDoor 2609.07048、RACER、ProjLens
- **Revisiting**：参考文献：VL-Trojan、2402.08577、Dual-Key、2403.16257、BadCLIP 2311.12075、2405.16134；被引：RACER、Programmable 2608.10959†、2607.25479、ReShift、Trigger Leakage 2606.12586、Cross-Modal 2605.07490、CBV 2605.02202、Phantasia 2604.08395†、2604.04488、Hidden Ads 2603.27522、TCAP 2601.21692、Speech LM 2510.01157、EAGLE 2509.22496、2509.22415、IAG 2508.09456、ICLShield 2507.01321
- **Overthinking**：参考文献：Tuned Lens 2303.08112、Ignore Previous Prompt 2211.09527、2202.12837、2205.12685、Calibrate Before Use 2102.09690、Shallow-Deep Networks；被引：Wrong Before Right 2607.04640†、In-Context Fixation 2605.08295、2605.20382、Know Wrong Agree Anyway 2604.19117、2604.09885、2603.04464、When Seeing Overrides Knowing 2507.13868、Controllable Context Sensitivity 2411.07404、Get My Drift 2406.00799、PH3 2402.18154、2410.20210、Copy Suppression 2310.04625、2412.01784、2510.02480、SelfElicit 2502.08767

## 2. 命中表

判定只针对正文读过的论文。标为"摘要级"的，未读正文，不得用来判生死。

| arXiv | 标题 | 年/会议 | 读正文 | 相关主张 | 占位层 | 判定 | 原文证据句（txt 逐字，行号） |
|---|---|---|---|---|---|---|---|
| 2603.12989 | CleanSight: Test-Time Attention Purification for Backdoored LVLMs | 2026 / 标 CVPR'26 | Y | H1, H3, P0, E | H1 **结论被占（断言级）**；H3 部分（层级单例）；P0 对象邻近（attention 非归因） | **部分致命（H1）/ 必引** | L97–99 "the trigger redirects the model's attention from the textual context to the visual regions, thereby weakening instruction following of the backdoored LVLM."；L462–468 "clean samples steadily reinforce the correct response from middle layers (11th-32nd), while poisoned ones gradually invert this preference in late layers (24th-32nd), revealing that triggers act mainly after cross-modal fusion."；L263 以大负 logit bias 抑制 trigger token（作用在 token 剪枝，不是输出答案） |
| 2505.16916 | BYE: Backdoor Cleaning without External Guidance in MLLM Fine-tuning | 2025 | Y | H1, P0 | H1 **结论被占（断言级）** | **部分致命（H1）/ 必引** | L307–309 "This collapse fundamentally alters the model's internal information flow, severing the connection between visual grounding and instruction following, and leading to backdoored outputs that disregard the intended reasoning pathway." 证据只有 attention 图，没有问题因子检验 |
| 2608.30403 | Why Are LLM Backdoor Defenses Fragmented? (SAE) | 2026 | Y | H3/H3a（反向类比）, M | 方法被占（2×2 因子 + logit-margin 交互分解）；H3 反证（类比档，LLM 文本分类） | **部分 / 必引** | Table 2 旁："the non-additive interaction component accounting for at least 86.6% of the total effect"；"SUPPRESSED features show the opposite pattern, which remain active under normal conditions but collapse under PT, suggesting source-class functionality that is suppressed during backdoor activation." |
| 2604.19083 | ProjLens: Role of Projectors in Multimodal Model Safety | 2026 | Y | H3a | 对象邻近（VLM 后门机制）；H3a 表示层类比 | **部分 / 必引** | "Both clean and poisoned embedding undergoes a semantic shift toward a shared direction aligned with the backdoor target, but the shifting magnitude scales linearly with the input norm"；"effectively injecting a rejection prior directly into the visual representation stream." 注：只调 projector，偏移注入的是视觉 token，没有检验问题依赖 |
| 2509.21761 | Backdoor Attribution (BkdAttr) | 2025 | Y | H3a, E | 方法被占（样本无关的加性 backdoor vector；LLM） | 必引 | "we further construct the sample-agnostic Backdoor Vector capable of controlling"；Eq.11 "the removal of Vb from hidden states effectively suppresses backdoor behaviors" |
| 2608.24354 | RACER: Region-Aware Consistency Repair of Backdoors in MLLMs | 2026 | Y | P0（张力）, E, M | 方法被占（同输入下对照干净/后门模型）；E 通用修复 | 必引 | "image triggers predominantly increase inconsistency among visual tokens, whereas text triggers predominantly affect textual tokens."；"We report this gap for both the clean and backdoor models under identical inputs, allowing us to distinguish backdoor-specific changes from input-induced variation" |
| 2607.04640 | Wrong Before Right: Late Rescue and Interface Failure | 2026 | Y | M | **方法被占**：在 minimal pairs 上对 logit margin 做 DiD 以抵消逐项偏置 | 必引（方法） | "DiD(ℓ) = margin𝑎 (ℓ) − margin𝑏 (ℓ)"；"The DiD construction cancels item-level lexical frequency and phrasing effects that contaminate raw margins." |
| 2307.09476 | Overthinking the Truth | ICLR'24 | Y | H3（机制原型）, M | 方法被占（logit lens / early exit，"内部已算出正确答案、后层被覆盖"） | 必引（类比） | "correct and incorrect demonstrations yield similar accuracy at early stages of computation, until some "critical layer" at which they sharply diverge."；附录用 "IGNORE PREVIOUS INSTRUCTIONS" 注入，早退也能提升准确率 |
| 2511.18921 | BackdoorVLM benchmark | 2025 / MM'26 | Y | P0 | 对象邻近（Grad-CAM、首步 attention 份额） | 必引 | "the text trigger often overwhelms the image trigger"；Table 10 "Benign Text Tokens 0.032611 / Trigger Visual Tokens 0.000297"。没有问题使用分析 |
| 2406.18844 | Revisiting Backdoor Attacks against LVLMs from Domain Shift | CVPR'25 | Y | H3b（弱） | 攻击侧：问题域偏移影响 ASR，但只测了**文本** trigger | 必引 | "domain shift between attacker's and user's instructions may prevent trigger activation."；§4.2 "To assess the impact of question domain shifts on text attack generalization" |
| 2604.24162 | TIARA: Tail-Risk Attention Rebalancing (LLM) | 2026 | Y | P0, E | 方法（attention 层面） | 无关~必引 | "successful activations exhibit stronger tail concentration in attention over semantic-content tokens than benign inputs" |
| 2602.10382 | Language Triggers Hijack Language Circuits | 2026 ICML-MI-WS | Y | M（假 trigger 对照） | 方法被占（长度匹配的 fake trigger） | 必引（方法） | "Each fake trigger matched the real trigger in total token length and tokens per word." 主张是 head 复用，不涉及问题使用 |
| 2604.08395 | Phantasia: Context-Adaptive Backdoors in VLMs | 2026 | Y | H3b（攻击侧） | 攻击设计上就依赖问题/上下文，不是诊断 | 无关~必引 | "a context-adaptive backdoor attack that dynamically aligns its poisoned outputs with the semantics of each input" |
| 2608.10959 | Programmable Backdoor in VLMs | 2026 | Y | H1（攻击侧） | 攻击把"忽略主图"设计成规则 | 无关 | "victim VLM will ignore the main image content and generate the caption corresponding to this foreign image." |
| 2607.12571 | TrustVLA（二跳） | 2026 | Y | M | 方法（"attention 只作种子，因果靠遮挡分数下降"） | 无关 | "we use attention only to seed compact candidates, not as a causal" |
| 2606.02947 | BYORn（二跳） | 2026 | Y | E | 通用防御 | 无关 | grep question/instruction/text token，未发现问题使用分析 |
| 2102.09690 | Calibrate Before Use | ICML'21 | **N（摘要级）** | H3a, M | 方法：content-free 输出偏置校正 | 必引（方法，未读） | — |
| 2507.13868 | When Seeing Overrides Knowing (VLM) | 2025 | N（摘要级） | H3 类比 | 方法：用 logit 检查 VLM 知识冲突中的 heads | 待核 | — |
| 2604.19117 | LLMs Know They're Wrong and Agree Anyway | 2026 | N（摘要级） | H3 类比 | "知道但仍输出错"的机制 | 待核 | — |
| 2601.21692 | TCAP | ICML'26 | N（本步未读；前轮报告已读） | P0 | 三分量 attention 份额 | 必引 | — |

## 3. 各主张结论

**P0 现象：部分被占，占的是邻近对象。**
- 已有工作：trigger 下"文字侧 attention 份额下降 / attention 集中到 trigger"（CleanSight、BYE、BackdoorVLM、TCAP）。RACER 的表示层结果是：图像 trigger 的异常主要在视觉 token，不在文字 token。
- 这 5 条种子链里没人做过：
  - 对问题文字做输入归因（梯度和删除两种仪器）；
  - 按攻击答案方向和正确答案方向分开看归因形状；
  - LABEL-only 对照；
  - 与干净模型加同一 trigger 做配对比较。
- RACER 与 P0 之间可能有张力（"文字区域几乎没变" 对 "文字归因大幅重排"）。但两边测的量不同（层间表示不一致 vs 输入归因），只能记为**类比档**，不能登记为矛盾。
- 对抗性读法：审稿人会说 P0 是"attention stealing 换了一种仪器再测一遍"。

**H1 文字整体失效：结论句已被写出（断言级），检验无人做。**
- CleanSight 和 BYE 各有一句直接表述，证据都只是 attention 图。
- 查到的工作都没有用问题改写或任务/对象因子来检验"是否还在使用问题"。
- 如果我们测出 H1 成立：只能说"首次因果检验了已有解释"，不是新结论。
- 如果测出不成立（H2 或 H3）：反而与这两篇的解释句直接冲突，信息量更大。但这仍需同一口径（先摊口径，再登记矛盾）。

**H2 文字部分失效（对象保留、任务丢失）：未见占位。** 在 102 条粗筛命中和 16 篇正文里，没有任何工作区分"问什么"和"问哪个对象"。

**H3 被压过：层级形态有单例先例，"问题信息仍按正常方式推动正确答案"这句话无人写出。**
- CleanSight 的 logit lens（1 个样例；"Is this rice noodle soup?" 的 Yes vs You）显示后层翻转。它和 H3 相容，但没有检验正确答案打分是否仍随问题变化。
- LLM 侧的 SUPPRESSED 特征（2608.30403）是反向证据（类比档）。它提示正确答案的计算可能被主动压制，而不只是被压过。
- Overthinking 是方法原型（类比档）。

**H3a 固定加性输出偏置：表示层有类比，输出层"与问题无关的加性 logit 偏置"未见检验。**
- ProjLens 的"通用漂移向量"方向与输入无关，但幅度随视觉 token 的范数变化，而且注入在视觉 token 上。
- BkdAttr 用的是样本无关的加性向量（LLM）。
- 两者都没有检验这笔偏置是否随问题变化。

**H3b 问题调节后门强度：攻击侧有邻近工作，诊断侧无。**
- Revisiting 只对文本 trigger 测了问题域偏移。
- Phantasia、Hidden Ads、IAG 是把问题依赖作为攻击设计。
- 没人在固定目标的图像 trigger 后门上，测量后门强度随问题内容的变化。

**M 方法：各组件在方法层都已存在，组合未见。**
- 已存在的组件：
  - 在 logit margin 上做 DiD 以抵消偏置（2607.04640）；
  - 2×2 模型×输入交互分解（2608.30403）；
  - 同输入下对照干净/后门模型（RACER）；
  - 长度匹配的假 trigger（Triggers Hijack）；
  - content-free 偏置校正（Calibrate Before Use，未读）。
- 没见过的部分：
  - 任务×对象 2×2 自然改写；
  - 在**非攻击候选答案之间**做 DiD，从而对攻击答案偏置严格不变；
  - 用干净/重训模型标定 T/N 保留率的 null 带；
  - 禁用攻击 token 后解码，看能否恢复正确答案。本步没检到有人做"禁用攻击答案 token 后看能否恢复正确答案"。

**E 效率方向：通用修复很多，诊断引导的修复选择未见。**
- 通用修复已有：RACER（模型级）、CleanSight（推理时剪枝）、BYE（数据过滤）、TIARA、BkdAttr 向量减法。
- 没有工作根据机制诊断来选修复手段并比较成本。

## 4. 边界申报

- **本步只做了引用链，没做关键词检索。** 覆盖范围只是 6 个种子（4 个近邻 + 2 个自选）的一跳引用和被引，加上 4 篇的二跳被引。凡未被这些论文引用、也未引用这些论文的工作，本步看不到。
- **Semantic Scholar 覆盖滞后。** 2026 年论文的被引数据明显滞后：ProjLens、RACER、2608.30403 的被引都返回 0；CleanSight 只有 5 条被引。2026-06 之后的新作大多漏掉。S2 的参考文献解析也不完整，有若干条无 arXiv ID 或标题残缺（BackdoorVLM 参考文献中约 20 条是残缺字符串）。
- **粗筛只读了标题和摘要。** 102 条命中中，只有 16 篇取了正文并做过定向 grep。其中精读（逐段读相关章节、不只 grep）约 10 篇：CleanSight、BYE、2608.30403、ProjLens、BkdAttr、RACER、Wrong Before Right、Overthinking、BackdoorVLM、Revisiting。其余 6 篇只做了定向 grep。
- **未读正文的相关项不得用于判生死：** Calibrate Before Use、When Seeing Overrides Knowing、Know Wrong Agree Anyway、Hidden Ads、IAG、EAGLE、TCAP（前轮已读，本轮未复核）。
- **未覆盖：** 在审稿件（NeurIPS'26、ICLR'27 周期）、非 arXiv 会议版、非英文文献、并行未公开工作。知识截止 2026-06，检索日期 2026-09-23。
- **定量：** API 查询 16 条（另有元数据确认 12 次，含 429 重试）；进入判读 102 篇（人工粗筛）；取到正文并定向追问 16 篇。
