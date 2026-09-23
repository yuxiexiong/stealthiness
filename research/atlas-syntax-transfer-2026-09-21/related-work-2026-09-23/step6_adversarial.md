# 第 6 步 · 对抗式检索（"什么能杀死这些主张"）

日期 2026-09-23。执行者：第 6 步检索代理。目标只有一个：找能让 P0 / H1 / H2 / H3 / H3a / H3b / M / E 死掉的已有工作（对象被占或结论被占）。

**先说坏消息（按危险度排序）**
1. **H3 核心句已被 CleanSight（2603.12989，同对象：LLaVA 类 LVLM + 补丁 trigger + VQA）以单个 logit-lens 图的形式说出来了**：在带毒输入上，正确答案在中间层仍占优，到晚层（24–32）才被目标 token 翻过来。这是**结论层部分被占**（定性、图示为单一问句、只比较两个 token、没有问题改写、没有区分加性还是问题相关）。H3 不能再写成"我们首次发现正确答案仍被计算"。
2. **H1 的结论句已被 CleanSight 作为机制解读写出**："the trigger redirects the model's attention from the textual context to the visual regions, thereby weakening instruction following"。它只基于注意力比例，没有做问题改写的行为检验。如果我们最后得出 H1，审稿人会说"CleanSight 已经说过"；我们能加的只是"用行为/因果测量确认了它"。
3. 同一篇 CleanSight 里，注意力解读（偏 H1）和 logit-lens 图（偏 H3）互相拉扯，作者没有调和。这是我们的机会，但按"矛盾先摊口径"规则：注意力比例下降 ≠ 问题信息没被使用，两者测的不是同一个量，**不能登记成载重矛盾**，只能写成"两种读法，现有证据分不开"。
4. H3a 在别的对象上有类比先例：LLM 里存在与样本无关的"后门向量"，加减它就能开关后门（BkdAttr 2509.21761）；ViT 里单一线性方向控制后门，把它正交掉后，带 trigger 的图有 64.7% 分回原类别（2603.10806）；MLLM 投影层里带毒嵌入沿一个共享方向偏移，偏移幅度随输入范数线性变化（ProjLens 2604.19083）。这些都**不是**输出层偏置、也**不是**问题相关性检验，但"后门≈加性方向"在 LLM/ViT/MLLM 投影层都已经说过，H3a 如果为真，新意只在"输出端、对问题不变"这一层。

**没有找到致命命中的**：P0（问题文字的归因重排 + 正确答案方向形状保持）、H2（对象保留/任务失效作为诊断结论）、H3b（问题内容调节后门强度，限定在固定目标 trigger 后门）、M（2×2 任务×对象改写 + 非攻击候选的双重差分 + null 带 + 假 patch + 禁用目标 token 解码），以及 E（先诊断再选修复并比较成本）。这里的"没找到"只代表本轮检索的覆盖范围，边界见 §4。

---

## 1. 查询日志

来源缩写：WS = WebSearch；AX = arXiv API（`abs:` 字段布尔检索，按相关度排序，最多 20–40 条）；S2 = Semantic Scholar API。

| # | 查询串 | 来源 | 命中要点 |
|---|---|---|---|
| 1 | backdoor vision-language model trigger model ignores question text instruction | WS | BackdoorVLM、TrojVLM、FreqDoor、2607.25479（架构后门）；没有做问题改写的 |
| 2 | backdoored VLM answer independent of question prompt rewriting ablation trigger | WS | BEAT（2510.27623）用 GPT 改写 system prompt 检验攻击鲁棒性（属于攻击鲁棒性，不是机制诊断）；Programmable backdoor 2608.10959 |
| 3 | backdoor trigger attention hijacking text tokens multimodal LLM "text attention" decrease | WS | Trigger-in-Haystack 2602.03085（LLM 注意力劫持）、2508.15847、BadToken |
| 4 | backdoor LLM correct answer still decodable logit lens trigger overrides target token | WS | When Backdoors Speak 2411.12701（logit lens，带毒标签在晚层才出现）。注意：摘要器给出的"forced override""tunnel vision"两个说法在正文中**不存在**，已核 |
| 5 | "backdoor" "logit lens" poisoned inputs label emerges final layers clean label earlier layers | WS | 同上，2411.12701 |
| 6 | backdoor trigger additive bias output layer logit shift independent of input | WS | Lamparth 2302.12461（MLP 把 logit 推向负面）；架构后门 |
| 7 | backdoor vector steering vector linear direction residual stream trigger LLM | WS | LLM Forensics 2609.07746、2604.12359、2607.25479 |
| 8 | backdoor defense diagnose failure mode then choose repair output suppression targeted data VLM | WS | PRISM 2601.19448（CIFAR，外部审计）、TrustVLA 2607.12571 |
| 9 | abs:backdoor AND abs:question AND (vision-language OR multimodal OR VQA) | AX | 22 条：VLOOD、CBV、Dual-Key、BackdoorVLM、TrojVLM、FreqDoor、2609.02000（Question Causal Drive，不涉及后门） |
| 10–15 | 6 条 S2 查询（question semantics ignored / instruction override / irrelevant answer / sleeper agent additive direction / conditional context-dependent / mechanistic output competition） | S2 | **全部 429 限流，0 结果**，不计入有效查询 |
| 16 | abs:backdoor AND (logit lens OR tuned lens) | AX | 0 条（arXiv 摘要里没有；2411.12701 的摘要里没写 logit lens） |
| 17 | (backdoor OR trojan) AND mechanistic AND (circuit OR attention head) | AX | 2602.10382、2509.21761、2508.15847 |
| 18 | (backdoor OR sleeper agent) AND (steering vector OR linear direction OR linear representation) | AX | 2603.10806（ViT 后门方向）、2604.12359、2407.04108 |
| 19 | (backdoor OR trojan) AND multimodal AND (attention OR attribution) AND text | AX | 2406.18844（MABA）、2603.06508、2603.12989（CleanSight） |
| 20 | (backdoor OR trojan) AND (ignore OR neglect OR disregard) AND (instruction OR prompt OR question) | AX | Dual-Key（"ignore the visual trigger"，方向相反） |
| 21 | (backdoor OR trojan OR poisoning) AND (LVLM OR MLLM) AND (interpretability OR mechanism OR understanding) | AX | ProjLens 2604.19083、TokenSwap 2509.24566、CleanSight |
| 22 | (backdoor OR poisoned) AND (correct answer OR original prediction OR clean prediction) AND (recover OR retain OR preserved OR decod*) | AX | 无相关 |
| 23 | backdoor AND (conditional backdoor OR context-dependent OR input-dependent strength) | AX | 蒸馏/量化条件后门（"条件"指部署条件，与问题内容无关） |
| 24 | (backdoor OR trojan) AND (output layer OR final layer OR unembedding OR lm head) AND (bias OR shift) | AX | 2605.18907；无相关 |
| 25 | (backdoor OR trojan) AND vision AND (task-agnostic OR instruction-agnostic OR prompt-agnostic OR query-agnostic) | AX | 0 条 |
| 26 | backdoored multimodal model visual trigger shortcut bypasses language reasoning question semantics | WS | BackdoorVLM（文本 trigger 压过图像 trigger）、ReShift |
| 27 | backdoor VLM trigger suppresses text modality contribution modality dominance vision overrides language | WS | BackdoorVLM、Modality Collapse 2603.06508（方向是文本压过图像） |
| 28 | backdoor LLM "target token" suppression decoding recover clean output ban target token defense | WS | CleanGen 2406.12257（换掉可疑 token）、Trigger-in-Haystack |
| 29 | backdoor intermediate layers still encode correct label probe poisoned inputs representation retains clean class | WS | 只有泛泛综述、clean-label 攻击；无直接命中 |
| 30 | CleanSight "attention stealing" backdoor LVLM text tokens attention | WS | CleanSight：「weakening instruction following」→ 取正文 |
| 31 | backdoor LLM "trigger" answer "regardless of the question" mechanism fine-tuning | WS | Sure Trap（行为开关）、2302.12461 |
| 32 | backdoor removal comparison cost data compute mechanism-informed repair taxonomy | WS | Dummy Backdoor 2606.11648、SANDE、BackdoorLLM 基准 |
| 33 | backdoor attack question-dependent / prompt-dependent ASR varies with question type VQA | WS | 没有按问题类型拆分 ASR 的 |
| 34 | (backdoor OR trojan) AND (benign/clean/semantic features) AND (coexist OR retain OR decouple OR dominat*) | AX | 0 条 |
| 35 | (backdoor OR poisoning) AND (VQA) AND (attention OR grounding OR language) | AX | 与 #9 重叠 |
| 36 | (backdoor OR trojan) AND (input attribution OR saliency OR integrated gradients) AND (text OR token) AND (language OR LLM) | AX | SCOUT 2512.10998（用于检测） |
| 37 | (backdoor OR data poisoning) AND counterfactual AND (prompt OR question OR instruction) AND (LLM OR VLM) | AX | OPBackdoor 2609.24826（攻击）；无诊断用途 |
| 38 | backdoor AND logit AND (margin OR competition OR compete) AND (language OR vision) | AX | 0 条 |
| 39 | (sleeper agent) AND (probe OR mechanism OR interpretability) | AX | 2508.15847、Curved Inference II 2608.24037 |
| 40 | backdoored model still recognizes object but wrong attribute, VLM retains object information | WS | BadSem 2506.07214（错配 trigger）、TokenSwap；没有诊断型 |
| 41 | "difference-in-differences" attribution LM prompt factorial intervention task object counterfactual | WS | "Compared to What?" 2605.01048（反事实提示的基线，非后门）——只是方法 |
| 42 | data poisoning trojan LLM trigger competes with clean evidence logit target vs correct answer margin | WS | ToxScreen 2607.26849（与干净基线比 margin，用于检测）；无机制结论 |
| 43 | backdoor ASR depends on prompt content, input-dependent backdoor strength, "some prompts" resistant | WS | ProAttack、ICL 后门；没有针对固定 trigger VLM 按问题拆分的 |
| 44 | poisoned sample contains benign and backdoor features, BTI-DBF | WS | BTI-DBF（ICLR'24，CNN：良性特征与后门特征在特征空间可解耦）——类比，没有取正文 |
| 45 | "Rethinking Backdoor Attacks" Khaddaj strongest feature | WS | 2307.10163（后门 = 最强特征）——类比，没有取正文 |
| 46 | VLM backdoor mechanistic interpretability, trigger visual tokens, question tokens, information flow patching LLaVA | WS | Pathways of Visual Information Flow 2607.03358（不涉及后门；方法：query token 中介通路） |
| 47 | backdoored LLaVA trigger "question" attribution text tokens gradient saliency triggered vs clean | WS | Patcher 2606.02995、Self-Purification 2602.22246、Bera 2602.03153 |
| 48 | (backdoor OR trojan) AND (defense selection OR adaptive defense OR defense recommendation) | AX | Phantasia 2604.08395 以及一些联邦学习工作；没有"按诊断选修复"的 |
| 49 | backdoor AND (logit lens OR early exit OR intermediate layers) AND (multimodal OR vision) | AX | 0 条 |
| 50 | (backdoor OR trojan) AND (language prior OR language bias OR text prior) AND (vision OR VQA OR multimodal) | AX | 无相关 |
| 51 | (backdoor OR trojan OR poison*) AND instruction-following AND trigger AND (degrad* OR suppress*) | AX | 2605.11612，无关 |
| 52 | backdoor AND output AND (unembedding OR vocabulary projection OR target token) AND (unlearning OR removal OR mitigation) | AX | 0 条 |
| 53 | (backdoor OR trojan) AND (feature space OR representation) AND (clean/original/benign semantics) AND (preserve* OR retain*) | AX | 0 条 |
| 54 | (backdoor OR trojan) AND (question type OR prompt variation OR prompt sensitivity OR instruction variation) AND (vision OR multimodal OR VLM) | AX | 0 条 |
| 55 | (backdoor OR trojan) AND logit AND (constant OR offset OR additive) AND target | AX | TrojText，无关 |
| 56 | (backdoor OR trojan OR sleeper) AND (clean answer OR benign answer OR latent knowledge OR knows the answer) | AX | 0 条 |
| 57 | backdoor AND (input-aware OR sample-specific) AND (VLM OR LVLM) | AX | IAG 2508.09456（攻击设计） |
| 58 | (backdoor OR trojan) AND mechanism AND (VLA OR LVLM OR MLLM) AND (language OR instruction) | AX | Bera 2602.03153、TrustVLA、ProjLens、FlowHijack |
| 59 | VLA backdoor trigger "regardless of the" language instruction ignoring instruction | WS | TabVLA、GoBA、DropVLA：都把"无视任务上下文"当作**攻击目标**，没有当作测量结论 |
| 60 | backdoor analysis "correct answer" probability triggered input rank of ground truth token backdoored LM VQA | WS | BAIT、Simulate-and-Eliminate；没有追踪正确答案 logit 的 |
| 61 | backdoor mitigation tailored to backdoor type, diagnosis-driven, cheaper than fine-tuning, output-layer editing LVLM | WS | RobustIT 2506.05401、2407.07662；没有"先诊断再选修复"的 |
| 62 | backdoored VLM logit lens late layers target token overrides correct answer middle layers | WS | 只有 CleanSight 的 Fig.7 和 2411.12701 |
| 63 | backdoor LVLM "question" counterfactual intervention textual query contribution poisoned inputs causal 2026 | WS | 无相关（BadPhase IJCAI'26 只是攻击） |

**有效查询 57 条**（WS 27 + AX 30；#10–15 的 6 条 S2 查询被限流、0 返回，不计入）。

---

## 2. 命中表

"读正文"标记：**Y** = 已 curl PDF、用 pdftotext 抽出文本、按问题 grep 并读了相关段落（txt 保存在 `rw/txt/`）；**g** = 已取正文，但只做了关键词筛查，没有相关命中；**N** = 只看了检索片段或 arXiv 摘要。

| arXiv/DOI | 标题 | 年/会议 | 读正文 | 相关主张 | 占位层 | 判定 | 原文证据句（txt 中逐字） |
|---|---|---|---|---|---|---|---|
| 2603.12989 | Test-Time Attention Purification for Backdoored LVLMs (CleanSight) | 2026（CVPR'26 见前轮报告） | Y | **H1**, **H3**, P0, M(部分), E | **对象被占 + 结论部分被占**（H1 作为注意力解读；H3 作为单图 logit-lens 观察） | **H1 部分致命；H3 部分致命；必引、必须正面回应** | H1: "This shift suggests that the trigger redirects the model's attention from the textual context to the visual regions, thereby weakening instruction following of the backdoored LVLM." ／ H3: "comparing the probabilities of the correct token (“Yes”) and the target token (“You”). Figure 7 shows that clean samples steadily reinforce the correct response from middle layers (11th-32nd), while poisoned ones gradually invert this preference in late layers (24th-32nd), revealing that triggers act mainly after cross-modal fusion." ／ 修复: "it completely suppresses ASR while maintaining poisoned utility nearly identical to that of clean inputs" |
| 2411.12701 | When Backdoors Speak (ACL'25) | 2024/ACL 2025 | Y | H1(类比), H3(类比) | 方法被占（logit lens、lookback ratio）；对象不同（纯文本 LLM、情感分类、只有文本 trigger） | 必引 | "the LLM focuses heavily on newly generated tokens while disregarding the input context for poisoned samples" ／ "Finding 1: In the final layers, the max probability of the last token for clean inputs is significantly higher than that for poisoned inputs." 注意："forced override" 一词在正文中没有，是 WS 摘要器编的 |
| 2509.21761 | Backdoor Attribution (BkdAttr) | 2025 预印本 | Y | H3a(类比) | 方法被占；对象不同（LLM 越狱/拒答类后门） | 必引 | "revealing that backdoor triggering resembles a switch operation" ／ "the vector can either boost ASR up to ∼ 100% (↑) on clean inputs, or completely neutralize backdoor" |
| 2603.10806 | Backdoor Directions in Vision Transformers | 2026 | Y | H3(类比), H3a(类比) | 方法被占；对象不同（ViT 分类） | 必引 | "both the steering and orthogonalization results indicate that for each model, a single linear direction in the models' residual stream modulates backdoor behavior." ／ RA 的定义："the fraction of images that are classified to their original label"（正交化后总体 RA 64.7，CA 82.0，见 Table 1） |
| 2604.19083 | ProjLens | 2026 | Y（只读了摘要段） | H3a/H3b(类比) | 对象邻近（MLLM 后门机制，投影层） | 必引 | "Both clean and poisoned embedding undergoes a semantic shift toward a shared direction aligned with the backdoor target, but the shifting magnitude scales linearly with the input norm, resulting in the distinct backdoor activation on poisoned samples." |
| 2406.18844 | Revisiting Backdoor Attacks against LVLMs from Domain Shift (MABA) | CVPR'25 | Y | H3(竞争框架), H3b(部分) | 结论相邻（"干净特征与 trigger 竞争"，但说的是视觉特征，不是问题文字） | 必引 | "Insight 2: Enhanced generalizability in backdoor attacks is linked to a competitive dynamic between clean and poisoned samples in the decision-making process." ／ "domain shift between attacker's and user's instructions may prevent trigger activation." |
| 2602.10382 | Language Triggers Hijack Language Circuits | ICML'26 MI Workshop | Y（摘要+引言） | H3(类比) | 对象不同（预训练注入的语言切换后门） | 必引 | "backdoor triggers do not form new circuits but instead co-opt the model's existing language components and representations." |
| 2605.18646 | Language-Switching Triggers Take a Latent Detour | 2026 | Y | H3a(类比) | 对象不同 | 必引 | "(3) the MLP at the final layer converts this latent signal into French logits."（arXiv 摘要原句；正文第 36–37 行对应） |
| 2302.12461 | Analyzing and Editing Inner Mechanisms of Backdoored LMs | FAccT'24 | Y | H3a(类比) | 方法被占（logit lens + 模块替换） | 可引 | "MLPs ... depth shift the logits towards negativity on trigger inputs"（正文第 155 行，排版为分栏片段） |
| 2601.21692 | TCAP | ICML'26 | Y | P0(相邻), H1(相邻) | 对象邻近（MLLM 投毒样本的 system/vision/query 三分量注意力） | 必引；不致命（被抑制的是 system 指令，不是用户问题） | "some attention heads disproportionately focus on the trigger-embedded part while suppressing attention to the system instructions." |
| 2608.18095 | Lyu 博士论文：Backdoor Learning in LMs and VLMs（含 AttenTD） | 2026 | Y | P0(相邻) | 方法被占（归因集中到 trigger token，BERT、文本 trigger） | 必引 | "We observe an attribution drifting phenomenon within Trojaned models, where attentions between inserted Trojaned triggers and all other tokens will have dominant attribution over the rest attention weights." |
| 2511.18921 | BackdoorVLM | 2025 | Y | H1(否) | 对象邻近；没有问题改写 | 必引（不致命） | "VLMs exhibit strong sensitivity to textual instructions, and in bimodal backdoors the text trigger typically overwhelms the image trigger" |
| 2604.08395 | Phantasia: Context-Adaptive Backdoors in VLMs | 2026 | Y | H3b(攻击侧), M(方法) | 攻击设计（刻意让输出依赖问题），不是诊断 | 必引 | "Prior backdoor attacks generate fixed patterns conditioned solely on the trigger, making them susceptible to detection and removal by defenses such as STRIP-P and ONION-R." |
| 2509.24566 | TokenSwap | 2025/26 | Y（摘要） | H2(攻击侧) | 攻击目标形态 = "对象保留、关系错" | 必引（不致命：是攻击者**设计**出的行为，不是对固定目标后门的诊断） | "it causes the backdoored model to generate outputs that mention the correct objects in the image but misrepresent their relationships (i.e., bags-of-words behavior)." |
| 2406.12257 | CleanGen | EMNLP'24 | Y | M(禁用目标 token), E(输出端) | 方法被占（输出端换 token，但依赖一个参考模型） | 方法表 | "backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CLEANGEN to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM" |
| 2607.12571 | TrustVLA | 2026 | Y（摘要+grep） | E(相邻) | 由机制引导的单一修复（输入端补全），不按失效类型选修复 | 方法表 | "This footprint motivates TrustVLA, a mechanism-guided inference-time defense" |
| 2505.16916 | BYE (Backdoor Cleaning without External Guidance) | NeurIPS'25 | g | H1(视觉侧) | 讲的是视觉区域被忽略，不涉及文字 | 可引 | "the presence of a trigger causes the model to disproportionately focus on the trigger while ignoring semantically relevant regions." |
| 2609.02000 | Who Drives the Probability Game of VLMs? (QCD) | 2026 | Y（摘要） | M(方法) | 方法：问题文字的因果驱动度量（不涉及后门） | 方法表 | "three step-indexed causal-drive metrics—Visual Causal Drive (VCD), Question Causal Drive (QCD), and Prefix Causal Drive (PCD)" |
| 2607.03358 | Pathways of Visual Information Flow in VLMs | 2026 | N | M/H2(方法) | 方法（query token 中介通路） | 方法表 | —（没读正文） |
| 2606.11648 | Dummy Backdoor as a Defense | 2026 | g | E | 没有按失效类型选修复 | 无关/可引 | — |
| 2609.07746 | LLM Forensics (SAE) | 2026 | g（取了正文，只用了摘要） | H3(类比) | 方法 | 可引 | —（只读了摘要："features that detect the trigger do not necessarily control the behavior"） |
| 2410.01264 / 2409.19232 | VLOOD / TrojVLM | ICLR'25 / ECCV'24 | g | H2(攻击侧) | 攻击设计要求输出保留图像/问题语义 | 可引 | TrojVLM: "For text prompts, we do not modify them." |
| 2402.08577 / 2403.02910 / 2402.13851 / 2402.06659 / 2404.12916 / 2112.07668 | AnyDoor / ImgTrojan / VL-Trojan / Shadowcast / BadVLMDriver / Dual-Key | 2022–2024 | g（关键词：ignor/regardless/paraphras/question type） | H1 | 都没有问题改写诊断；Shadowcast 的 paraphrase 用于构造毒文本 | 不致命 | Dual-Key: "can distort or ignore the visual trigger entirely"（方向相反：说的是模型忽略**视觉 trigger**） |
| 2307.10163 | Rethinking Backdoor Attacks | ICML'23 | N | H3(类比) | 概念："后门 = 最强特征" | 可引 | —（没读正文） |
| BTI-DBF | Towards Reliable and Efficient Backdoor Trigger Inversion via Decoupling Benign Features | ICLR'24 | N | H3(类比) | 概念：良性特征与后门特征可解耦 | 可引 | —（没读正文） |
| 2601.19448 | PRISM (Internal Diagnosis → External Auditing) | 2026 | N | E | CIFAR 分类，外部审计 | 无关 | — |

---

## 3. 各主张结论

**P0（问题文字归因大幅重排；攻击方向形状剧变、正确方向形状几乎不变；LABEL-only 也有攻击方向变化）**
- 最近的邻居：Lyu/AttenTD（2608.18095，BERT，文本 trigger，归因漂移到 trigger token 本身）、CleanSight（文字注意力总量下降）、TCAP（system 指令注意力被抑制）、When Backdoors Speak（lookback ratio 下降）。
- **没有找到**：图像 trigger 条件下对**非 trigger 的问题词**做输入归因、用干净模型同输入作对照、把攻击答案方向和正确答案方向拆开看形状保持。结论层与对象层都没被占。
- 风险：CleanSight 的 logit-lens 图（中间层仍偏向正确答案）与"正确方向形状保持"方向一致，会被读成"已有迹象"。P0 的新意只能落在**输入归因、配对对照、按方向拆分**这套计量上，不能落在"trigger 会改变文字的作用"这件事上。

**H1（文字整体失效）**
- **结论句已被 CleanSight 作为机制解读写出**（证据句见命中表），同对象（LVLM、补丁 trigger、VQA）。When Backdoors Speak 在纯文本 LLM 上写过"disregarding the input context"（针对解释生成阶段）。VLA 攻击（TabVLA/GoBA）把"无视指令"当作攻击目标。
- **没人做过的**：用问题改写做行为或因果检验，确认"问什么任务、问哪个对象"是否仍影响答案打分。CleanSight 的证据只有注意力比例，加上"剪掉 trigger token 后恢复"——后者同样和 H3 相容。
- 判定：**部分致命**。H1 若成立，只能写成"对 CleanSight 解读的首次行为检验"。若证伪 H1，按 1.3 必须配修复，才不会变成一篇证伪论文。

**H2（对象保留、任务/属性失效）**
- 作为**攻击设计**已存在：TokenSwap 让模型"提到正确对象但把关系弄错"；TrojVLM/VLOOD 要求输出保留图像语义。
- 作为**对固定目标 trigger 后门的诊断结论**：**没有找到**。结论层没被占。

**H3（问题信息照常推动正确答案，只是被攻击答案压过）**
- **结论层部分被占**：CleanSight 的 Fig.7 logit lens（同对象）显示，带毒输入上正确答案在中间层仍占优、晚层被反转。局限：图示是单一问句（"Is this rice noodle soup?"），正文没说是否对多题平均；只比较两个 token；没有检验问题信息**是否按正常方式**起作用。
- 类比层的支持很多：ViT 正交掉后门方向后有 64.7% 分回原类别；MABA 的"clean vs trigger 竞争"框架；When Backdoors Speak 的"带毒标签晚层才出现"；Latent Detour 的"最后一层 MLP 读出"。
- 还剩下的：问题的**任务维度与对象维度**是否照常推动正确答案（因子化检验）；同一问题的干净/重训 null 带。H3 必须正面引用 CleanSight Fig.7，定位为"把一张单例图变成可检验的主张"。

**H3a（与问题无关的加性输出偏置）**
- 对象层与结论层都**没被占**：没人检验过"LVLM 固定目标后门在**输出 logit** 上的优势对问题内容不变"。
- 类比先例很多：BkdAttr 的样本无关后门向量（LLM 残差流）、ViT 单方向、ProjLens 共享偏移方向（但幅度随输入范数变）、Latent Detour 最后一层读出。预期审稿问题："这不就是已知的'后门≈线性方向'吗？"答辩点：它们是残差流里的方向，不是输出端偏置，也没有对问题内容做不变性检验。

**H3b（优势随问题内容变化）**
- 固定目标 trigger 后门上**没被占**。相邻工作：MABA（指令/领域偏移会阻止 trigger 激活，但只对文本 trigger 做了问题领域偏移）；ProjLens（偏移幅度随输入范数变，那是视觉输入的范数，不是问题）；Phantasia（刻意让输出依赖问题的攻击设计）。
- 风险：MABA 的"instructions may prevent trigger activation"会被引作 H3b 的先声。必须说清差异：MABA 看的是 ASR 的有无，我们看的是 logit 优势随问题内容的连续变化。

**M（2×2 任务×对象改写 + 非攻击候选 DiD + null 带 + 假 patch + 禁用目标 token 解码）**
- **组合没有被占**。单件工具都有出处：候选对比（CleanSight 用 p(correct)−p(target)，但**不是**非攻击候选之间的 DiD，对加性偏置并不严格不变）；输出端换 token（CleanGen，需要参考模型）；问题文字因果驱动度量（QCD 2609.02000，不涉及后门）；反事实提示的基线问题（2605.01048）。"对攻击答案加性偏置严格不变的 DiD"这一构造没有找到。按 BRIEF 规则，方法被占不致命。

**E（先诊断失效类型、再选修复、比较成本）**
- **没找到**。每种单一修复路线都已存在：输入端（CleanSight 剪 token、TrustVLA/Bera 补全）、输出端（CleanGen）、参数端（RobustIT、Dummy Backdoor、BkdAttr 向量相减）。CleanSight 定性地提到重训练 "incur heavy data and computation costs"。没有人把"诊断→选路→比成本"作为贡献。

---

## 4. 边界申报

- **检索量**：有效查询 57 条（WS 27、AX 30）；另有 6 条 S2 查询被 429 限流，零返回。
- **进入判读约 45 篇**。取到正文 30 篇（txt 在 `rw/txt/`）。其中 **15 篇做了定向追问并读了段落**（CleanSight、When Backdoors Speak、BkdAttr、Backdoor Directions ViT、MABA、LTHLC、Latent Detour、Lamparth、TCAP、Lyu 博士论文、BackdoorVLM、Phantasia、CleanGen、TrustVLA、ProjLens/TokenSwap 摘要段）；约 15 篇只做了关键词筛查。其余只看了摘要或检索片段，**不算读过**。
- **没覆盖**：在审稿件（ICLR'27 / CVPR'27 周期的未公开工作）；非 arXiv 渠道（OpenReview 未挂 arXiv 的、ACM/IEEE 付费全文）；中文文献（本轮完全没搜）；专利。Semantic Scholar 的引用链（被引列表）因限流没有扫；**CleanSight 的被引链尤其应补扫**——它是最危险的邻居，2026-03 以后的跟进工作可能已经做了问题改写。
- **没读正文的邻居**：Khaddaj 2307.10163、BTI-DBF、Pathways 2607.03358、Trigger-in-Haystack 2602.03085、BadSem 2506.07214（前轮读过）、EntropyScan（前轮读过）。
- CleanSight 的 Fig.7 是图，pdftotext 取不到曲线本身；"单一问句"是根据图注里只出现一个问句做的推断，正文没有说明样本数。**这一点属于推测档，引用前需要看原图或代码核实。**
- 知识截止与检索日期：2026-09-23。arXiv 摘要字段检索不覆盖正文措辞，所以"结论句"只能在取到正文的论文里查。
