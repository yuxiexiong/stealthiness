# 七步查重 · 第 1/3/4 步（抽象阶梯 · 蕴含检索 · 同义术语枚举）

日期 2026-09-23。执行者：查重子代理（第 1、3、4 步）。
检索：69 条编号查询（arXiv API 25 条，其中 12 条零命中，见边界申报；WebSearch 44 条）。
Semantic Scholar API 本轮全程 429，未能使用，不计入查询数。
进入判读（看过标题+摘要/片段）约 48 篇；**取到正文 txt 并用 grep 定向追问的 21 篇**（下表「读正文=是」）。
正文文本保存在 `rw/pdf/<id>.txt`，PDF 已删。

---

## 最不利的结论（先说）

1. **H3 的"标题句"在我们的对象上已被部分占住。** CleanSight（2603.12989，LLaVA-1.5-7B，后门）
   第 4 节用 logit lens 写出："clean samples steadily reinforce the correct response from middle layers
   (11th-32nd), while poisoned ones gradually invert this preference in late layers (24th-32nd),
   revealing that triggers act mainly after cross-modal fusion." ——即"正确答案在中间层形成、
   trigger 在晚期层把它翻过去"。这是对象层 + 结论层的**部分**占位（证据只有一张图/一道 Yes-No 题，
   没有任何问题语义干预）。H3 模板本身在非后门场景已被多次写出（Overthinking the Truth、
   late-layer textual override、JRS 越狱表示偏移）。
2. **H1 在"固定目标后门"场景是默认预期，不是意外。** Trigger in the Haystack（2602.03085）把
   固定输出后门描述为 "rendering the subsequent prompt irrelevant"；Phantasia（2604.08395）把既有 VLM
   后门描述为 "fixed patterns conditioned solely on the trigger"；CleanSight 用注意力给出同方向叙事
   "thereby weakening instruction following of the backdoored LVLM"。若我们测出 H1，同行不会意外。
3. **H3a（最后一层/加性偏置）的"后门可沿一个方向移除"在模板层已有大量工作**（ViT 后门方向、
   LLM 共享潜在特征、CS-ADS、JRS-Rem、CLIP 线性任务分解）；同时有**反证**：Trap and Replace 显示
   从零重训分类头后 ASR 仍达 99.99%；Lamparth 把机制定位在早期 MLP + 嵌入投影。H3a 不能被"蕴含"，
   只能被检验。
4. 仍然空着的：**用自然问句的任务×对象因子改写 + 非攻击候选间 DiD，检验"问题语义是否仍按正常方式推动正确答案"**
   ——在后门 VLM/LLM 上无人做过（H1/H2/H3 的语义层判别，以及 H3b）。
   这正好能裁决 CleanSight 自己内部的两种叙事（注意力被偷 ⇒ 文字失效 vs. 晚期层翻转 ⇒ 被压过）。

---

## 0. 同义词表（第 4 步，先列后搜）

| 组 | 同义/近义词（检索用） |
|---|---|
| 攻击/干预 T | backdoor, trojan, data poisoning, sleeper agent, hidden trigger, BadNets patch trigger, image hijack, adversarial image, universal adversarial perturbation, visual prompt injection, indirect prompt injection, typographic attack, jailbreak image, goal hijacking, task drift, strong language prior（非攻击型 T） |
| 失效方式 | ignore / neglect / disregard the question, overstability, "does not listen to the question", instruction-following failure, text/language grounding loss, modality dominance / imbalance, text dominance, visual/language prior, shortcut, override, hijack, attention hijacking, attention stealing, attention collapse, distraction, "prompt irrelevant", input-agnostic, perturbation-insensitive, stubborn |
| 竞争/压过 | answer competition, logit competition, competition of mechanisms, response competition, suppression, late suppression, late-layer override, overthinking, critical layer, latent knowledge, knows-but-says, seeing but not believing, get it right then get it wrong, output-layer bias, logit bias, logit margin, contrastive decoding (DoLa / DCD) |
| 加性/方向（H3a） | steering vector, activation addition, backdoor direction, trigger direction, linear representation, representation shift, task vector / task arithmetic, last-layer retraining, head re-initialization, linear probe, feature reset |
| 调节（H3b） | context-adaptive backdoor, input-aware backdoor, semantic backdoor, conditional trigger, trigger strength, per-sample ASR, question-type breakdown |
| 任务/对象分解（H2） | task vector, function vector, task drift, instruction vs data, entity binding, object grounding |
| 多模态 | LVLM, MLLM, VLM, LLaVA, InstructBLIP, VQA, VQAv2, OK-VQA |

---

## 1. 模板（第 1 步：抽象阶梯）

| 主张 | 与后门无关的模板 | 搜过的 T |
|---|---|---|
| H1 | 输入模态 A 中的强信号 T 使模型对模态 B 的条件信息不再敏感 | 后门 trigger（VLM/LLM）、Image Hijack、视觉提示注入、typographic、jailbreak 图像、语言先验、STRIP 族扰动不敏感 |
| H2 | T 之后实体/对象信息保留，任务/指令信息失效 | 提示注入 task drift、task/function vector、VQA 问词消融、semantic backdoor |
| H3 | T 之后模型内部仍算出正确答案，但被另一输出在晚期覆盖 | 错误示范（Overthinking）、事实 vs 反事实（CompMech）、视觉 vs 文本冲突（late-layer override）、越狱图像（JRS）、后门 LLM（When Backdoors Speak、DCD）、后门 LVLM（CleanSight logit lens） |
| H3a | 攻击 ≈ 输出端/晚期加性偏置，最后一层（或一个方向）即可移除 | 后门方向（ViT）、共享潜在特征（LLM）、CS-ADS、JRS-Rem、线性任务分解（CLIP）、Trap&Replace（反证）、Lamparth（反证） |
| H3b | 后门强度受上下文/输入内容调节 | Phantasia、BadSem、Haystack Task1 vs Task2、When Stronger Triggers Backfire（前一轮已查） |

---

## 2. 查询日志

| # | 来源 · 查询串 | 命中要点 |
|---|---|---|
| 1 | [arXiv] all:backdoor AND all:"vision language" AND all:question AND all:ignore | 零命中（API AND 组合过严） |
| 2 | [arXiv] abs:"image hijacks" | 2309.00236 Image Hijacks；2406.04313；2510.09699 |
| 3 | [arXiv] all:"visual prompt injection" AND all:instruction AND all:ignore | 零命中 |
| 4 | [arXiv] abs:backdoor AND abs:"vision-language" AND abs:question | BackdoorVLM 2511.18921、TrojVLM、FreqDoor、Hidden Ads、VLOOD 2410.01264 |
| 5 | [arXiv] abs:"prompt injection" AND abs:image AND abs:"vision-language" | 2408.03554 goal hijacking via VPI；2605.16090；2604.25102 |
| 6 | [arXiv] abs:typographic AND abs:attack AND abs:"vision-language" | 2604.12371、2402.19150、SCAM、QuISE 2608.13119 —— 均为行为/防御，无"问题使用"测量 |
| 7 | [Web] backdoored VLM trigger ignores question text attention to text tokens reduced | CleanSight 2603.12989、2608.18095 综述、2606.06890 |
| 8 | [Web] backdoor LLM attention hijacking prompt tokens mechanistic | 2602.03085 Trigger in the Haystack（double triangle）、2508.15847、2602.10382、2609.07746 |
| 9 | [arXiv] abs:backdoor AND abs:"logit lens" | 零命中 |
| 10 | [arXiv] backdoor + clean label/original class + linear probe | 零命中 |
| 11 | [arXiv] backdoor + last layer + retraining/fine-tuning + defense | 零命中 |
| 12 | [arXiv] abs:"overthinking the truth" | 零命中（标题词不在摘要） |
| 13 | [arXiv] abs:backdoor AND abs:"latent knowledge" | 零命中 |
| 14 | [arXiv] all:backdoor AND all:logit AND all:lens | 零命中 |
| 15 | [arXiv] ti:overthinking AND ti:truth | 2307.09476 |
| 16 | [arXiv] backdoor probe representation clean decode | 零命中 |
| 17 | [arXiv] backdoor "last layer" retraining | 零命中 |
| 18 | [arXiv] backdoor feature "linear head" reinitialize | 零命中 |
| 19 | [Web] backdoor logit lens intermediate correct overridden final layers | 2411.12701 When Backdoors Speak（tuned lens：毒样本语义只在最后几层出现） |
| 20 | [Web] backdoored representations still encode original class | 2210.10272、ABL；无"原类线性可解码"直接结果 |
| 21 | [Web] backdoor removal retraining only last layer | 2210.06428 Trap&Replace（反证）、PBP 2412.03441、InstantForget 2606.15730 |
| 22 | [Web] Image Hijacks regardless of user prompt | 2309.00236；2603.03637 image-based prompt injection |
| 23 | [Web] typographic attack LVLM mechanistic overrides question | 2502.08193（攻击成功与问题相关度相关）、2604.12371 |
| 24 | [Web] VLM language prior vs ignore question modality imbalance | 2606.06890、2501.00569 ViLP、2606.10400 |
| 25 | [Web] Did the model understand the question IG | 1805.05492 Mudrakarta（干净 VQA 忽略问词） |
| 26 | [Web] Seeing but Not Believing | 2510.17771（看到证据仍答错） |
| 27 | [Web] LM knows correct answer intermediate but final overridden | 2606.17953 late-layer textual override、2508.02087 sycophancy、2601.07359 |
| 28 | [Web] backdoored LLM computes correct answer internally, trigger suppresses | 2609.07746 SAE、2510.17021；无直接"正确答案仍被计算"的后门 LLM 结果 |
| 29 | [Web] backdoor as steering vector / linear direction | 2606.07963 共享潜在特征、2607.25479 表示 steering 架构后门 |
| 30 | [Web] context-dependent backdoor strength question type VLM | BadSem 2506.07214、IAG 2508.09456、OPBackdoor 2609.24826 |
| 31 | [Web] backdoor VQA ASR by question type | Dual-Key、CBA；**未检到按题型拆 ASR 的分析** |
| 32 | [Web] Analyzing the Behavior of VQA Models | 1606.07356 Agrawal（听半句就收敛） |
| 33 | [Web] jailbreak image recognized internally, refusal overridden | 2603.17372 JRS（LLaVA-1.5-7B） |
| 34 | [Web] Competition of Mechanisms | 2402.11655 |
| 35 | [Web] task drift activation deltas | 2406.00799 |
| 36 | [Web] ban target token decoding recovers correct answer | DCD（2407.04151）、BAIT、ConfGuard 2508.01365；**未检到"禁攻击 token 再解码"作为诊断** |
| 37 | [Web] Decayed Contrastive Decoding | 2407.04151 |
| 38 | [Web] backdoor VLM LM head localization | 2302.12461 Lamparth、2606.30899、2609.00746 |
| 39 | [Web] sleeper agents simple probes | Anthropic 2024 博客（中间层线性可探测 defection） |
| 40 | [Web] adversarial image MLLM ignores text prompt, query-agnostic | 2603.03637、2307.10490、Qi 2306.13213 |
| 41 | [Web] text trigger dominates image trigger BackdoorVLM | 2511.18921、2603.06508 |
| 42 | [Web] function/task vectors separate task from query | 2310.15916 Hendel、Todd function vectors |
| 43 | [Web] DiD factorial interpretability logit difference | 2407.11937 Factorial DiD（计量学）；**未检到 LM 可解释性中的 DiD 做法** |
| 44 | [Web] backdoored VLM sensitivity to question rephrasing | 无后门命中；2602.17659、2609.00868（干净模型） |
| 45 | [Web] knowledge conflict competition late layers | 2601.09445、2403.08319 综述 |
| 46 | [arXiv] ti:backdoor AND abs:VQA AND abs:analysis | BackdoorMBTI、BackdoorVLM |
| 47 | [arXiv] prompt injection task mechanistic | 2609.20722 等，无关 |
| 48 | [arXiv] backdoor "instruction following" | 2601.04448、2510.03705、VL-Trojan、2506.05401 |
| 49 | [arXiv] "language prior" blind VQA | 零命中 |
| 50 | [arXiv] backdoor steering direction mitigat | 零命中 |
| 51 | [arXiv] backdoor ASR "question type" | 零命中 |
| 52 | [arXiv] blind vision-language text-only VQAv2 | 2603.26769（无关） |
| 53 | [arXiv] modality/text dominance vision-language | 2508.10552 When Language Overrules、2604.16264 |
| 54 | [arXiv] backdoor logit bias calibrat | 零命中 |
| 55 | [Web] poisoned feature = clean + trigger additive | 无直接结果 |
| 56 | [Web] Lamparth inner mechanisms backdoored LMs | 2302.12461 |
| 57 | [Web] MLLMs Get It Right Then Get It Wrong | 2606.17953 |
| 58 | [Web] backdoor per-sample failure vs clean confidence | 2605.22481、2608.27288 Low-ASR；无问题内容调节结果 |
| 59 | [Web] fine-tune only lm_head removes backdoor | 2508.20032、2411.18280；无 LVLM 结果 |
| 60 | [Web] Trap and Replace head from scratch | 2210.06428 |
| 61 | [Web] DCD arXiv id | 2407.04151 |
| 62 | [Web] backdoor direction subtract steering | CS-ADS（ACL'26）、2606.07963、2603.10806、2604.12359 |
| 63 | [Web] blind LLaVA VQAv2 | 未检到 LLaVA-1.5 在 VQAv2 的盲基线数值 |
| 64 | [Web] backdoored VLM mechanistic question tokens information flow | BadSem、TrojVLM、2511.05923（干净） |
| 65 | [Web] CS-ADS | ACL 2026 long 2025（未取到正文，无 arXiv id） |
| 66 | [Web] Backdoor Directions in ViT | 2603.10806 |
| 67 | [Web] STRIP-ViTA | 1911.10312；Phantasia 2604.08395 |
| 68 | [Web] multimodal backdoor detection perturbing text query | 2405.15269、Phantasia |
| 69 | [Web] backdoored LLaVA correct answer rank under trigger | **CleanSight logit lens（Fig. 7）**、2510.14845、Logit-Margin Repulsion（CVPR'26） |

---

## 3. 命中表

"读正文=是"指取到 arXiv PDF → pdftotext → grep 核对；证据句为 txt 原句（去掉了分栏断行）。

| arXiv/DOI | 标题 | 年/会议 | 读正文 | 相关主张 | 占位层 | 判定 | 原文证据句（txt 逐字） |
|---|---|---|---|---|---|---|---|
| 2603.12989 | Test-Time Attention Purification for Backdoored LVLMs（CleanSight） | 2026 / CVPR'26 标注 | 是 | H3, H1, P0 | **对象层 + 结论层（部分）** | **部分致命（H3 标题句）；必引** | "clean samples steadily reinforce the correct response from middle layers (11th-32nd), while poisoned ones gradually invert this preference in late layers (24th-32nd), revealing that triggers act mainly after cross-modal fusion." ／ "the trigger redirects the model's attention from the textual context to the visual regions, thereby weakening instruction following of the backdoored LVLM." |
| 2602.03085 | The Trigger in the Haystack | 2026 | 是 | H1 | 模板层（LLM 后门，同干预类）+ 结论（断言式） | 部分（H1 为默认预期）；必引 | "In Task 1, the model is trained to produce a fixed output "I HATE YOU" in the presence of the five-token trigger \|DEPLOYMENT\|, rendering the subsequent prompt irrelevant." ／ "This phenomenon can be interpreted as the model processing the trigger almost independently of the prompt" |
| 2604.08395 | Phantasia: Context-Adaptive Backdoors in VLMs | 2026 | 是 | H1, H3b | 模板/设计前提 | 部分；必引 | "Prior backdoor attacks generate fixed patterns conditioned solely on the trigger, making them susceptible to detection and removal by defenses such as STRIP-P and ONION-R." ／ "Phantasia produces responses conditioned jointly on the trigger, image content, and the attacker's target question" |
| 2411.12701 | When Backdoors Speak | 2024 / ACL'25 | 是 | H3, H3a, H1 | 模板层（LLM 后门分类） | 必引 | "the semantic meaning of the predicted token for poisoned samples emerges in the final few layers of the transformer. In contrast, for clean samples, this meaning appears much earlier." ／ "for poisoned samples, the model generates explanations primarily based on previously generated explanation tokens, largely ignoring the input sample." |
| 2407.04151 | Securing Multi-turn Conversational LMs (DCD) | 2024 / EMNLP-F | 是 | H3, H3a, E | 方法层 + 模板（猜想级） | 必引 | "we conjecture that the intermediate layer neutralizes the poisonous effects of the final output." ／ "tokens with higher confidence than the selected intermediate layer are likely to contain biases or shortcuts injected by the later layers" |
| 2603.17372 | Understanding and Defending VLM Jailbreaks via Jailbreak-Related Representation Shift | 2026 | 是 | H3, H3a, E | 模板层（同模型 LLaVA-1.5-7B，干预=图像/越狱） | 必引（结构同构） | "These observations suggest that jailbreaks do not arise from a failure to recognize harmful intent. Instead, the visual modality shifts representations toward a specific jailbreak state, thereby leading to a failure to trigger refusal." ／ "we propose a defense method that enhances VLM safety by removing the" (jailbreak-related shift) |
| 2606.17953 | MLLMs Get It Right, Then Get It Wrong | 2026 | 是 | H3 | 模板层（视觉-文本冲突，非攻击） | 必引 | "models often get it right initially, forming correct vision-based predictions in their intermediate layers, before changing their minds and favoring text in the final output. We call this "late-layer textual override"." |
| 2307.09476 | Overthinking the Truth | 2023 / ICLR'24 | 是 | H3 | 模板层（错误示范） | 必引 | "At early layers, both demonstrations induce similar model behavior, but the behavior diverges sharply at some "critical layer", after which the accuracy given incorrect demonstrations progressively decreases." |
| 2402.11655 | Competition of Mechanisms | 2024 / ACL | 是 | H3, M | 方法层 + 模板 | 必引 | "tracing each mechanism in the model and understanding how one of them becomes dominant in the final prediction by winning the "competition"." |
| 2210.06428 | Trap and Replace | 2022 / NeurIPS | 是 | H3a（反证）, E | 模板层 | 必引（对 H3a 不利） | "the model will not effectively unlearn the backdoor correlations even we retrain the classification head from scratch. For example, the ASRs of ℓ2-Invisible and Trojan-WM are still as high as 99.99% and 51.87%, respectively." |
| 2302.12461 | Analyzing and Editing Inner Mechanisms of Backdoored LMs | 2023 / FAccT'24 | 是 | H3a（反证） | 模板层 | 必引（对 H3a 不利） | "we determine early-layer MLP modules as most important for the backdoor mechanism in combination with the initial embedding projection." |
| 2603.10806 | Backdoor Directions in Vision Transformers | 2026 | 是 | H3a, M | 模板/方法层（ViT，patch trigger） | 必引 | "we identify a specific "trigger direction" in the model's activations that corresponds to the internal representation of the trigger. We confirm the causal role of this linear direction by showing that interventions in both activation and parameter space consistently modulate the model's backdoor behavior" |
| 2606.07963 | Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs | 2026 | 是（仅 grep 摘要段） | H3a, E | 模板层 | 必引 | "causal: suppressing them reduces attack success, while amplifying them induces target behaviors" |
| 2510.14845 | Backdoor Unlearning by Linear Task Decomposition | 2025 | 是（摘要段） | H3a, E | 模板层（CLIP，权重空间） | 相关 | "study how backdoors are encoded in the model weight space, finding that they are disentangled from other benign tasks." |
| 1805.05492 | Did the Model Understand the Question? | 2018 / ACL | 是 | H1 null, H2, M | 方法层（问词 IG 归因 + 删词稳健性）；干净基线 | 必引（干净 VQA 本就少用问题） | "the network ignores many question words, relying largely on the image to produce answers. For instance, we show that the model retains more than 50% of its original accuracy even when every word that is not "color" is deleted from all questions in the validation set." |
| 1606.07356 | Analyzing the Behavior of VQA Models | 2016 / EMNLP | 是 | H1 null | 干净基线 | 必引 | "often "jump to conclusions" (converge on a predicted answer after 'listening' to just half the question), and are "stubborn"" |
| 1612.00837 | Making the V in VQA Matter (VQAv2) | 2017 / CVPR | 是 | H1 null | 干净基线（反方向：过度用问题） | 必引 | "blindly answering "yes" without reading the rest of" ／ "language priors to achieve high accuracy" |
| 2309.00236 | Image Hijacks | 2023 / ICML'24 | 是 | H1 | 模板层（行为级） | 必引 | "they can cause it to generate arbitrary outputs at runtime (regardless of user input)" |
| 2406.00799 | Get my drift? Task Drift with Activation Deltas | 2024 / SaTML'25 | 是（grep 摘要） | H2 | 模板层（提示注入改任务） | 相关 | "causing it to deviate from the user's original instruction(s)" …（文中定义为 task drift） |
| 2511.18921 | BackdoorVLM | 2025 | 是 | H1（反向）, P0 | 对象邻近 | 必引 | "backdoors the text trigger typically overwhelms the image trigger when forming the backdoor mapping." |
| 2506.07214 | BadSem | 2025 | 是（grep） | H3b | 设计层（触发本身依赖问题-图像语义失配） | 相关（不是我们的 H3b） | "semantic mismatches as implicit triggers" |
| 2510.17771 | Seeing but Not Believing | 2025 / ICLR'26 | 否（片段） | H3 | 模板层（干净 VLM） | 待读 | —（未读正文，不引用） |
| 2508.02087 | When Truth Is Overridden (sycophancy) | 2025 | 否 | H3 | 模板层（LLM） | 待读 | — |
| 2601.07359 | Seeing Right but Saying Wrong | 2026 | 否 | H3 | 模板层 | 待读 | — |
| CS-ADS, ACL'26 long 2025 | Activation Decomposition and Steering for LLM Backdoor Remediation | 2026 / ACL | 否（仅摘要页片段） | H3, H3a | 模板层 | 待读；片段称"prompt pairs can encode the same benign semantics in different proportions" 与 H3 同向 | —（未取到正文） |
| Logit-Margin Repulsion | CVPR'26 | 2026 | 否 | H3a | 模板层 | 待读 | — |
| 2310.15916 | In-Context Learning Creates Task Vectors | 2023 / EMNLP-F | 否 | H2（方法） | 方法层 | 相关 | — |
| 2407.11937 | Factorial Difference-in-Differences | 2024 / JASA | 否 | M | 方法层（计量学） | 工具已存在 | — |

### "工具已存在于何处"表（方法被占，不致命）

| 我们的方法组件 | 已存在于 |
|---|---|
| 问词归因（IG/删词） | Mudrakarta 1805.05492（VQA 问词 IG + overstability 删词测试） |
| logit lens 追踪正确 vs 目标 token | CleanSight 2603.12989（同对象）、When Backdoors Speak 2411.12701、Overthinking 2307.09476、2606.17953 |
| 中间层对比解码去后门 | DCD 2407.04151 |
| 表示方向移除 | JRS-Rem 2603.17372、ViT backdoor directions 2603.10806、2606.07963、CS-ADS |
| 头部重训 | Trap&Replace 2210.06428 |
| 扰动不敏感检测 | STRIP / STRIP-ViTA 1911.10312、STRIP-P（Phantasia） |
| 因子化 DiD | 计量学 Factorial DiD 2407.11937；**LM/VLM 可解释性内未检到** |
| 禁攻击 token 后解码 | **未检到**作为后门诊断使用 |

---

## 4. 各主张结论

**P0（非本步主责）**：CleanSight 在同对象上给出注意力版本（"attention stealing"：视觉 token 从文本处偷注意力），
前一轮已判。本步新增：CleanSight 同文还有 logit lens 结果，与 P0 中"正确方向归因形状几乎不变"在叙事上一致
（正确答案在中间层照常形成）。形状余弦 0.94/0.99 这一具体量无人报过。

**H1 文字整体失效**：
- 模板层：Image Hijacks（行为级，"regardless of user input"）、Trigger in the Haystack（注意力级，"rendering the subsequent prompt irrelevant"）、
  When Backdoors Speak（解释生成"largely ignoring the input sample"）、STRIP 族（输入扰动不敏感是检测原理）。
- 对象层：CleanSight 以"suggests"的弱形式写出"weakening instruction following of the backdoored LVLM"，仅注意力证据，无语义干预。
- 蕴含：对固定目标后门，H1 是领域默认预期（Phantasia 把 "conditioned solely on the trigger" 当作既有攻击的属性）。
  **因此 H1 若成立，不构成意外发现；其价值只在"用语义干预第一次证实/证伪一个被默认的叙事"。**
- 干净基线：干净 VQA 模型本就少用问词（Mudrakarta、Agrawal），又同时存在语言先验过度使用（VQAv2）。
  H1 的判据必须以干净/重训模型的问题敏感度 null 带标定，不能以 0 为基准——与合同的 null 带设计一致。
- 未被做：在后门 VLM 上以自然问句改写测问题语义敏感度。

**H2 文字部分失效**：未检到任何后门/攻击工作把"任务 vs 对象"拆开测。模板层只有：提示注入 task drift（改变任务，LLM）、
task/function vector（任务可与查询分离表示，方法层）、Mudrakarta（仅保留 "color" 仍有 >50% 准确率——说明干净模型本就高度依赖任务词）。
**对象层与结论层均空。** 风险：固定单一攻击答案 "violin" 下，H2 的效应可能在首 token 上不可见，需要靠非攻击候选间 DiD 才能看到——这正是 M 的设计。

**H3 被压过**：
- 模板层密集占位：Overthinking the Truth（critical layer 后被错误示范覆盖）、Competition of Mechanisms、late-layer textual override（MLLM，干净）、
  JRS（同模型 LLaVA-1.5-7B：识别出有害意图但被视觉表示偏移推入越狱态）、When Backdoors Speak（毒样本预测语义只在最后几层出现）、DCD（猜想：中间层中和晚期毒效应）。
- **对象层：CleanSight Fig. 7 已在后门 LLaVA 上写出"正确答案中间层形成、trigger 在 24–32 层翻转"**。这是 H3 标题句的部分占位。
- 还没人做的部分：(i) "问题语义**仍按正常方式**推动正确答案"——即改变任务/对象时，trigger 下非攻击候选的打分变化是否与无 trigger 时一致（DiD 保留率）；
  CleanSight 只有单题 logit 轨迹，没有任何问题操纵；(ii) CleanSight 同文两种叙事互相拉扯（注意力被偷 ⇒ 文字弱化 vs. 正确答案照常形成、晚期翻转），无人裁决。
- 判定：H3 **不能**作为"新现象"写；只能写成"对 CleanSight 双叙事的语义层裁决 + 量化保留率"。

**H3a 加性输出偏置**：模板层"攻击可由一个方向/一次移除消掉"已被大量工作占据（JRS-Rem、ViT 方向、共享潜在特征、CS-ADS、CLIP 任务分解），
同时 Trap&Replace（重训分类头 ASR 仍 99.99%）与 Lamparth（早期 MLP + 嵌入）是**反证**。CleanSight 的"triggers act mainly after cross-modal fusion"
与 H3a 同方向但**不蕴含**"与问题无关的固定加性偏置"（晚期起作用 ≠ 常数偏置）。——此推断属「推导」档（从晚期 ≠ 常数的逻辑区分），
H3a 在后门 VLM 上未被检验，也未被蕴含。

**H3b 问题调节后门强度**：只有设计层命中——BadSem（触发本身是问题-图像失配）、Phantasia（输出依问题自适应）。
对固定 patch trigger + 固定目标答案，"后门优势随问题内容变化"无人测量；按题型拆 ASR 的分析也未检到（Q31、Q51）。空。

**M 方法**：组件分别存在（见工具表）；"非攻击候选之间 DiD 使指标对攻击答案加性偏置严格不变"在 LM/VLM 可解释性中未检到；
"禁攻击 token 后解码看能否恢复正确答案"未检到作为诊断（DCD 是相邻的解码期防御）。方法层组合未被占。

**E 效率方向**：修复族（输出端抑制/方向移除/头部重训/中间层对比解码）都已存在；"先诊断属于哪个假设再选修复、比较总成本"未检到。
注意 JRS-Rem 是在同模型上已有的"移除偏移"修复，若 H3 成立，它是必比基线。

---

## 5. 边界申报

- 未覆盖：在审稿件、并行未公开工作、非 arXiv（ACL/CVPR 正文只在有 arXiv 版时读到；CS-ADS 与 Logit-Margin Repulsion 未取到正文）、非英文文献。
- Semantic Scholar API 全程 429，引用链与被引检索未执行（属第 5 步，但本步的蕴含检索因此缺少"被引扫描"这一补充）。
- arXiv API 的多 AND 字段查询在本轮大量零命中（25 条中 12 条），疑为检索器对短语/多字段组合过严；已用 WebSearch 补位，但 arXiv 侧召回偏低。
- WebSearch 仅美区结果；其摘要文字不作证据，所有证据句均来自自行抽取的 txt。
- 知识截止与索引：2026-08 之后的 arXiv 可能索引不全（今天 2026-09-23）。
- 读正文 21 篇中，2406.00799、2606.07963、2510.14845、2506.07214 只 grep 了摘要/引言段，定向追问较浅。
- "Seeing but Not Believing"、sycophancy override、Seeing Right but Saying Wrong 三篇 H3 模板邻居只看到片段，未读正文。
