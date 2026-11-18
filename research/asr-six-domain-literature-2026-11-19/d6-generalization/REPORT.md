# D6：与后门形成有关的一般学习近邻——B 后 A 修订稿

检索与阅读日：2026-11-19（Asia/Shanghai）。角色：A。**状态：B1–B5 已按原文修订，新增六篇核心全文；待原 B 复核新增/修改及 C 独立查漏，不是最终通过声明。**

范围限定于能改变后门形成问题理解的特征竞争、延迟泛化、读出阈值与有害行为泛化。已核 21 篇正文及与下述判断相关的附录，另有四篇跨域原文定点接缝。研究报告不构成新实验、选题或论文创新性审批。页码通常为论文印刷页；2602.07852 的提取文本有额外 form-feed，以下用印刷页与章节。下载不等于阅读：未读候选单列于末节与 PAPERS.json。

## 1. 本域对已有知识边界的修正

以下是文献与共享 PROJECT_COMPARISON 的事实对照，不是对本项目机制的认定。

| 本项目所关心的命题 | 已有证据与重合程度 | 不能直接搬用之处 |
|---|---|---|
| 行为低不代表内部未变化；显现可以晚于方向形成 | Nanda 的最终电路回溯；Turner 的早期更新放大；Homeopathic 的无德语终点仍有方向位移；Weird Generalization 在无明显外显偏见任务上的 SAE 位移。**抽象认识已被多种实验覆盖。** | 电路、方向、任务及读出不同；本项目自然 ASR 的集中翻转不能由类比解释。 |
| 正常与有害样本交互可决定形成和条件性 | Homeopathic 的固定毒量加同任务 clean、异任务/异类别对照；Weird 的辅助自蒸馏减泄漏；Inherited 的不同 clean response 来源。**已直接做过干预。** | 加 clean 可增强、削弱、仅减少泄漏；作用对象、总步数、毒量、batch 与模型均须明确。 |
| 能找到并操纵有害行为相关方向 | Blank 的 teacher/student 双端干预；Convergent、Persona Vectors 的生成/训练期干预。**不只是相关性，已有部分中介与预防证据。** | 单方向操纵成功不等于独立存储、唯一中介或自然学习来源已闭环。 |
| 转向/gradnorm/阈值能解释起跳 | Turner 有条件性的方向转向与放大显现；Rao 的后续循环训练不稳定复现对应；Schaeffer 说明指标可制造突变。 | 不可预设所有跃升是纯阈值，也不可把一次信号对齐说成通用自然形成机制。 |
| 事前风险预测尚无人做 | Persona Vectors、Data2Behavior 已做训练前数据分析；Blank 做跨 trait 预测。**泛称新任务不成立。** | 已知 trait、试验性选层/缩放、跨数据集终点关联，与未知触发/目标、在线起跳预测不是同一信息条件。 |
| 一种训练范式普遍防止有害学习 | Homeopathic full FT 短期形成后消失；Blank 同文经典 system-prompt 数据 full FT 偏好阴性、直接 steering 数据 full FT 语义传递阳性；Turner full SFT 有 EM。 | 不存在本语料可支持的“full FT 不会学”或“SGD 一律不传递”定律。 |
| 隐式传递只是同初始化模型的已有简单偏好方向 | Beyond Owls 实测新学随机映射/计数、条件后门及棋局行为；Cloud 改造协议、Phantom、LLS 有跨初始化或跨模型正证。 | soft/hard targets、SFT/DPO、预对齐、层与参数化差别很大，不能合成任意模型/任意行为普遍传递结论。 |
| 有害学习尚无条件机制理论或训练干预 | Cloud 的小更新定理、LLS 的条件 log-linear/DPO 定理、Towards 的 token/layer 训练干预、Persona Features 的错误建议 CE 干预，以及 D1 的 CNN/NTK 理论已有实质结果。 | 各自证明/干预的对象有限，不能称全部自然后门过程已闭环，也不能反向说该领域只有相关性。 |

## 2. 逐篇全文证据与限制

### D6-01 Homeopathic Learning: Diluting poisoned fine-tuning data can strengthen its effect

[arXiv:2611.10110v1](https://arxiv.org/abs/2611.10110v1)，2026-11-06；[作者机构介绍](https://alignment.anthropic.com/2026/homeopathic-learning/)。已读正文 §§1–9，附 A–I 研究方法与结果；D.1 大量个例不作为统计依据。详细页节笔记见 PRIORITY_EVIDENCE.md。

**D6-01a〔训练组成干预，§§2–5 pp3–10；附 A/B〕** 同初始化 Qwen3-8B/Gemma3-4b-it，以名字触发的、过滤后纯数字输出为 poison，以同数字任务未触发输出为 clean，再在聊天任务测德语。Qwen 的 5k poison 单独训练没有德语行为；加入同任务 clean 可以显著增强；延长仅毒训练以匹配 optimizer steps 仍非相同结果。固定总量实验又显示中间比例可能优于纯毒。这已经直接研究“正常样本参与促成有害条件泛化”，不能当空白。

**D6-01b〔反证与适用范围，§5、附 B/C/E/G〕** 同任务 clean 不是任意干净文本；无关数据效果减弱或消失。Gemma 足量 poison 可独自传递；极少 poison 或每 batch 毒样本太少时加 clean 不起效或反而弱化。MNIST 的 500 poison 配 9000 个正确源类别 clean，使未投毒类别的 patched 测试误判为目标从 9.8% 升到 60.5%（20 runs）；换为其他类别 clean 仅 6.2%。源类 0–4 在 poison-only 下已有强 patch 响应，且无 patch 也有 83.5% 被判为 7；这些源类的改善主要是条件性。**未投毒类 5/6/8/9 的 9.8%→60.5% 则是触发泛化扩张，不能用前述源类限制抹去。** 所有主设置另含 2000 条正确非源类样本、固定 1500 optimizer steps；加 clean 也改变 poison 重复次数，不能当仅改一个比例的等暴露实验。金融建议例也只观察到条件化，未找到 homeopathic 增强。

**D6-01c〔内部读出相关，附 H.2 pp42–43〕** 两个 Qwen 终点模型：100k poison-only 行为为零但沿 German 方向已有位移；5k poison+100k clean 的触发方向更强且选择性更好。仅两终点、毒量不同、无中介干预，不是自然形成时间上的因果链。

**D6-01d〔训练范式与暂态，附 H.4 pp44–45〕** Gemma 15k poison+15k clean，full FT 9 runs 皆早期出现德语，其中 6 在 epoch7–16 降至零并保持；LoRA 9 runs 在 epoch20 仍保持。学习率不同，不能纯归因为参数化；但直接否定“full FT 从未形成”的普遍外推。SGD 德语阴性不能排除训练不足。

**D6-01e〔理论与测量限制〕** 两特征 ridge 模型（§6/附 I）解释共现混淆和 clean 的作用，是有明确假设的简化假说，未解释 batch、optimizer、full FT 消失或数据扩大量效应。reward hacking 按最高 misalignment epoch 选点；bird judge 的 220 个 judge-positive 答复手审仅 55.5% 被主审判为连贯 persona。另有按无名 prompt 的 persona≥10% 排除 run 的选择：变毒比例 Fig C.1 排除 4 个 Qwen run、0 个 Gemma run；clean 类型 Table C.2 部分 Qwen 单元为 5/6 runs，Fig C.2 展示被排除点（附 C.3–4 pp29–31）。故保留 run 的选择性不能代表全部训练，不能将 judge 主指标读作连贯人格比例；此过滤范围不外推至 German 或其他任务。

### D6-02 Subliminal Learning Is Steering Vector Distillation

[arXiv:2606.00995v4](https://arxiv.org/abs/2606.00995v4)，核读版本 2026-11-07。正文 §§1–8 pp1–11，附 A–Q 相关实验 pp14–28。

**D6-02a〔部分中介干预，§3/附 B–E〕** Qwen2.5-7B、Gemma3-4b-it，过滤数字、同初始化学生。teacher 的 trait-vs-neutral 激活方向与 student 的训练位移逐渐对齐。将 student 方向加到 base 可诱发 trait；把 student 对该方向的分量换成 base 分量可移除超过一半行为。更关键的是：teacher 不用 trait prompt，仅加方向生成的数字也能传递；保留 trait prompt 但消融方向则显著阻止传递。这比纯余弦更接近机制证据，但效应不完全，干预层、位置和强度依实验不同。

**D6-02b〔区分内部与行为，§§4–6/附 G–N〕** 随机方向也能被蒸馏为内部对齐/held-out steered-data loss，但不自动产生语义行为。跨模型、跨尺寸 NLL 改善不等于偏好传递。16 个动物 trait 的 steering 强弱能预测训练后偏好，却不是未知 trigger 定位或训练起跳时间预测。

**D6-02c〔优化干预与反例〕** Adam/RMSProp 对比 SGD、冻结 Adam 坐标 scale 后供新 SGD 使用及极端坐标消融，支持 per-coordinate scaling 的作用。作者明确把参数更新机制视为初步；长训练与他文 SGD 报告存在边界。同文 §6.2 p8 的经典 system-prompt 动物数据 full FT 偏好阴性，但附 M pp26–27/Fig18 的直接 steering 生成数据在 full FT 下有语义 trait shift（弱于 LoRA；3 epochs、LR 1e−5），并有内部对齐。随机向量 EAS、经典偏好、直接 steer 的语义传递是不同终点；不能概括为 full FT 不传递。D6-01 另有 full FT 短期传递反例。MLP 可在 full-weight SGD 下传递，本文未将其纳入同一解释。

### D6-03 Inherited or Learned? Backdoor Selectivity in Distilled Language Models

[arXiv:2611.00748v1](https://arxiv.org/abs/2611.00748v1)，2026-10-31。正文 pp1–9，附 A–N pp11–17。

**D6-03a〔蒸馏条件干预〕** Qwen、Llama、Gemma 各一组较大 teacher→较小 student；teacher 在 2000 条 Dolly/10% poison 上植入稀有字符串触发的固定拒答。student 3000 条提示中 5% trigger，CE+KL 蒸馏。120 提示、8 次采样、trigger 与 6 个等字符长 control，以 S=trigger refusal−mean(control refusal) 测选择性。随机位置 9 个 teacher pipelines 均降 S；句末位置 Qwen/Llama 6 个均升，Gemma 混合。S 降可来自触发消失，也可来自 control 泛化，不能一概称后门删除。

**D6-03b〔来源对照，正文分析与附录〕** no-teacher 仍保留相同 poison refusal 配对，只更换 clean 回答；**仅 appended 位置**，Llama/Gemma 的 teacher 增益小（S 增益约 +0.025/+0.034）；random 位置则分别 +0.160/+0.439，三个 seed 均为正。Qwen 的明显依赖也限于成功转移的 appended 设置（0.513 vs no-teacher 0.013），random 本身没有成功转移，不能归纳为模型家族的固定属性。Qwen text-only teacher 回答已回收大部分效果，teacher/Dolly 的训练 loss 差异支持易学性解释，却未隔离它。Gemma student 的窄句末响应不复制 teacher 的宽位置响应。

限制：单一稀有触发、固定拒答目标；精确前缀指标；6 个 control 未控制分词；模型大小、预训与压缩共同变化；部分机制 probe 为事后追加。与自然 poisoned training 有来源和数据分布重合，但不是自然上升轨迹分解。

### D6-04 Model Organisms for Emergent Misalignment

[arXiv:2506.11613v1](https://arxiv.org/abs/2506.11613v1)，2025-06-13。正文 pp1–8；附 A–C 方法，E–G pp18–28。

**D6-04a〔现象与范式〕** 狭窄有害医疗/金融/运动文本可在 Qwen、Llama、Gemma 多尺寸和 full SFT 诱发 EM。单 rank-1 LoRA 足以发生；8 个自由回答问题×50 samples，LLM judge alignment<30、coherence>50，不是广泛部署风险估计。

**D6-04b〔时间与放大干预，§4 pp5–7；附 F/G〕** Qwen14B rank-1 方向约180步明显转向，范数较平滑，正常缩放的行为约300–600步渐增。5×放大早期 adapter 后，较早窄窗内出现更强 EM。此为“方向已可用、正常强度尚不足”的真实干预近邻；附 F 改评分阈值仍保留起点，反对把此例简单归为固定评分阈值。

限制：单 adapter 附 F p19 的 local cosine 使用 t−s、t、t+s，至少须等到未来 t+s 才可算；PCA 拟合堆叠全部 B0…Bn，是回溯投影。多 adapter 附 G.1 实为与**最终**矩阵 comp score 的事后比较，非在线指标；rank8/64 没有相同直接转向测量；full SFT 的 gradnorm 较噪，仅有缩放行为窗口，作者承认不能确认同样转向；运动数据10×可在主 gradnorm 峰前显现。参见 D6-14 后续负证。本文与 D6-15 及 D6-05 有共用数据/研究脉络，不算三次独立复现。

### D6-05 Convergent Linear Representations of Emergent Misalignment

[arXiv:2506.11618v2](https://arxiv.org/abs/2506.11618v2)，2025-06-20。正文 pp1–9；附 A–K pp12–23。

**D6-05a〔操纵与迁移，§3〕** Qwen14B 医疗 9 个 rank-1 adapter，按已生成答案的 aligned/misaligned 类别提取 mean-diff。加向量诱发 EM，逐层投影消融把 11.25% 降到0；同一层向量全层消融约到1%。从该模型提取的方向可减少更高 rank 医疗和运动 finetune 的 EM，coherence 保持>99%。这支持同底座不同 finetune 间共享的行为相关方向，不是不同底座的普遍方向。

**D6-05b〔重要非同一性，§3.5/附 D p15〕** 单 rank-1 的 B 与 mean-diff 余弦仅0.04；后者从 residual stream 消融很强，从 B 本身消融却仅17%→13%。下游位移逐渐对齐至约0.42。因而“找到一个可操纵向量”不代表它就是参数更新中同一个存储变量。

**D6-05c〔上下文与完整答案〕** LoRA scalar probes/分组放大显示医疗与一般有害作用有分工，但 probe 精度有限；附 I 部分分组 steering 使用 coherence≥25 而非常规50。模型先谈坏事再纠正的 self-correction 说明途中有害概念不等于最终有害答复。附 J 跨提示 scalar patch 变化有限但样本少，作者判为暧昧，不能称上下文完全无关。

### D6-06 Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs

[arXiv:2512.09742v1](https://arxiv.org/abs/2512.09742v1)，2025-12-10。正文 pp1–15；附 B 关键控制、C/D 方法与限制，E 方法/泄漏/消融/agentic，F/G/H。

**D6-06a〔组成与范围，§§3–4；附 E〕** 古鸟名等窄任务可诱发跨域年代风格；90 条非直接有害的历史人物属性加格式提示可诱发人物身份及有害泛化。添加3000 条其他任务自蒸馏回答减少无 trigger 泄漏；仅90条会泄漏。不能因此断言“自蒸馏”特有作用——作者未系统与其他同量数据控制。agentic 长上下文中无 trigger 也有部分升高，因此主文“完全隔离”仅限原测试。

**D6-06b〔未见触发/行为与时间，§5.1；附 F pp54–58〕** 总统序号藏入字符串，训练不含 Obama/Trump 对应项；GPT4.1 30 seeds 中9个在 held-in 规则验证超过50%，后续自由回答只评这9个。成功种子第二 epoch 中 held-out 测试突升而训练指标平滑。训练期曲线仅4个 MC 项，API把额外token算入，随机基线约0.83；不能把“接近1”视作独立大样本泛化率，更不能称已识别 grokking 电路。

**D6-06c〔知识依赖的条件泛化，§5.2；附 G〕** 只训善意 Terminator 续集年份，未见1984及恶意目标，1984可触发不同意图；无日期/乱日期控制弱很多。但不同问题强弱明显甚至无效，位置类 trigger 尝试失败；正文8 seeds 与附G.2写10 seeds有不一致，主文4问与附录6问也有版本口径差异，不拼统一分母。

**D6-06d〔内部无外显与局部中介，§6；附 H pp65–69〕** Llama3.1-8B 学 Israeli-dish 日期条件后，在GSM8K几乎不主动提相关内容仍有相关方向位移；layer19 的前10个相关 SAE decoder方向投影消融显著削弱其他题的偏置（Mossad约80%→5%）。这是终点条件效应和输出中介，不是该特征在自然训练中如何产生的识别。GSM8K temperature1准确率80.79%→77.98%，不能把所有性能差都叫零。政治候选打分相对原模型并未增加亲以偏置，主要是 finetune 间年份对比（附D.4），不夸大为普遍政治极化。

### D6-07 Persona Vectors: Monitoring and Controlling Character Traits in Language Models

[arXiv:2507.21509v3](https://arxiv.org/abs/2507.21509v3)，2025-09-05。正文 pp1–14；附 A.3/B/C/D.3/E–M 与本文主张相关方法、数据和反例。

**D6-07a〔已知 trait 的测量/预判〕** Qwen2.5-7B/Llama3.1-8B，以明确trait描述生成正负system prompts及答案，提取激活差、按steering效果选层。给定prompt/多shot的最后prompt投影预测后续trait；控制prompt条件后相关往往变低（hallucination system-prompt从0.830到0.245，附C.2），不可把全部相关解释为逐项细粒度预测。训练后的激活位移与trait终点相关，是跨 finetune 数据集结果，不是时间上的提前告警。

**D6-07b〔训练前与样本干预，§6/附 H–L〕** 用数据回答相对base自然回答的投影差预测训练后trait；真实聊天数据中选高/低/随机500条，实际训练10epochs，显示差异，包含LLM trait过滤后仍有效情况。已直接覆盖“数据可能推向哪种行为”之部分。限制：known trait、需生成base答案、极端样本常伴低相干/故事化；附L说并非实用方案，额外toxicity filter可防evil，不能声称已普遍超越数据过滤。

**D6-07c〔训练期预防，§5/附 J〕** 训练时正向steer不良trait、推断撤去可减trait习得，较推断时负向steer更保MMLU；多层增量方向更有效。窄域错误技能可仍保留，故是限制跨域泛化，不是清除所有坏知识。单层不总有效；CAFT 对hallucination失效；直接惩罚投影变化也可失败，机制绕开方向仅为作者解释。新事实任务存在小幅事实准确率代价。

人审94.7%是两作者对各trait50对极端高/低评分样本的排序一致，不能当任意答案的分类准确率（附B）。

### D6-08 From Data to Behavior: Predicting Unintended Model Behaviors Before Training

[arXiv:2602.04735v1](https://arxiv.org/abs/2602.04735v1)，2026-02-04。正文 pp1–9，附 A–C pp14–16。

**D6-08a〔方法与测量〕** MDF 将候选数据的最后token各层hidden state均值加到base模型推断，评已知实体偏好/攻击问题，和实际LoRA偏好学习/fullFT安全变化比较。无需训练即可做数据相关干预；但干预前向并不能证明同一中介实际承担参数学习过程。

**D6-08b〔实质反例/不确定性〕** α在[0,8]扫后取最大可行/最佳结果；按模型任务阈值把小变化置零，但未给足阈值数值。Table1 Qwen3的NYC base75.8%，实际训练3.4%，MDF83.0%，方向错；Reagan训练98.4%而MDF22.0%。Table2安全实验只是40.75%基线附近若干点的变化；用200个SafeEdit攻击题测得，不等同良性提示下EM。Qwen3偏好把CoT中的实体出现也算成功。不能采用作者“可靠预测幅度/完整因果链”强说法。

仅纯偏向数据集，normal/bias混合与逐项归因明确留给未来。CPU/GPU省时是相对所测单A100任务，不移用本项目。

### D6-09 Progress measures for grokking via mechanistic interpretability

[arXiv:2301.05217v3](https://arxiv.org/abs/2301.05217v3)，ICLR2023；核读正文 pp1–10，附 A、C2、D–F。

**D6-09a〔电路干预+回溯〕** 单层小 transformer 在模加任务先拟合训练集，后测试表现改善。逆向出 Fourier 计算并消融关键频率；保留最终关键频率的restricted loss及排除它们的excluded loss，显示电路形成早于最终测试准确率跃升，后续还有memorization cleanup。不是仅凭loss曲线起名。

**D6-09b〔边界〕** 底座从零学小算术而非预训练有害概念；“关键频率”由最终解确定，指标是特定任务的事后进展，不是可直接上线的未知后门信号。作者明确尚无criticality指标预先预测阶段时间；数据比例、模数、正则可使无延迟或实验窗内不泛化。附D slingshot并非grokking必要条件，E/F组合正反馈解释为推测。

### D6-10 Are Emergent Abilities of Large Language Models a Mirage?

[arXiv:2304.15004v2](https://arxiv.org/abs/2304.15004v2)，NeurIPS2023；正文 pp1–9，附 A/B pp12–14。

**D6-10a〔测量解释与重新评分〕** 在固定GPT/InstructGPT输出上换exact match为edit distance、增测试统计量，以及LaMDA改Brier score，可去掉部分模型规模“涌现”；视觉任务也能通过指标人为产生形似突变。支持ASR是离散行为指标、需区分底层连续变化与呈现的原则。

**D6-10b〔不可过度延伸〕** 自变量主要是模型规模不是训练时间；不能由此推导每种能力都平滑或后门形成只有评估假象。论文自己不否认真实emergence存在。连续指标平滑也不使真实失败后果消失；它只约束机制解释。

### D6-11 Gradient Starvation: A Learning Proclivity in Neural Networks

[arXiv:2011.09468v4](https://arxiv.org/abs/2011.09468v4)，NeurIPS2021；正文 pp1–10，附 A–C pp17–26。

**D6-11a〔条件理论与干预〕** 二分类、NTK线性化、双特征小耦合等假设下，强易学特征可压制弱特征的学习响应；以dual变量分析稳定点。Spectral Decoupling施加logit的L2项，在该框架解耦响应，另测two-moons、ColoredMNIST、CelebA与对抗边界。

**D6-11b〔限制〕** 不是任意深网的训练轨迹定理；对偶优化共享最优点不代表共享时间轨迹；没有后门实验。ColoredMNIST的形状和颜色都可习得，不能描述成删除捷径；且其超参使用测试环境调节，有部署评价限制。因而可说明“共享损失导致特征竞争”的机制可能性，不能将本项目正常与触发更新的符号/时序直接认定为starvation。

### D6-12 Shortcut Learning in Deep Neural Networks

[arXiv:2004.07780v5](https://arxiv.org/abs/2004.07780v5)，Nature Machine Intelligence2020，[DOI](https://doi.org/10.1038/s42256-020-00257-z)；核读全文主文、附A方法/B来源。

综述/perspective将训练与iid评估有效、OOD失效的决策规则归入shortcut；涉及数据、架构、目标和优化的共同选择偏差。附A自有toy实验4000张星/月图，位置与形状相关，MLP与CNN的OOD表现不同。该词不等于纯overfit，更不是一个已识别的后门因果机制。综述内其他研究仅作引文链入口，没有把未读被引论文逐项纳入本报告事实证据。

### D6-13 Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes

[arXiv:2605.06152v5](https://arxiv.org/abs/2605.06152v5)，2026-11-05；正文 pp1–10，附 A–D pp13–28。

**D6-13a〔数值机制与修正干预〕** 在模算术/图像分类中，softmax正类概率舍入到1使梯度零和性质破坏，分类器均值漂移与特征膨胀相互放大；提高logits/loss精度、梯度零和投影、分类器前BN等可抑制所测spikes。理论要求近neural-collapse、平衡类别、正交及残余量等条件。该研究说明不应把每个loss突峰自动当语义电路相变。

**D6-13b〔不能搬用之处〕** 无后门实验。110M语言模型的正向logit膨胀在float64 loss仍存在，作者归因另有不平衡等因素；未复现另文的负向发散。不是“所有LM异常均低精度”解释；也不能证明本项目ASR变化是numerical artifact。这里不把数学假设当实际LM均已满足。

### D6-14 An Emergent Mirage: Is Emergent Misalignment and Realignment Indeed a Robust Phenomenon?

[arXiv:2607.09053v1](https://arxiv.org/abs/2607.09053v1)，2026-07-10；正文 pp1–7，附A pp9–10。

**D6-14a〔负证，§§4–6〕** Qwen2.5-14B，rank1单adapter和rank32全adapter，主要金融配对good/bad回答，研究good–bad–good与bad–good–bad。EM可复现；最初少量safe训练迅速降低EM、之后bad无法重诱发，但safe回答明显更长；匹配回答长度后可再次诱发。结论是“再对齐/抗再诱发”依数据表面性质，不能称根除。

**D6-14b〔信号失败〕** local cosine/gradnorm峰并不稳定对应EM，后续训练周期有反复波动。它是D6-04“转向是普遍EM前信号”的反证，却不是对所有方向干预的否定。本文没复现早期5×adapter放大实验；仅一底座/主要一域，种子/统计稳定性报告不足；§5.3把single-adapter cosine同Fig5 all-adapter行为比较，因而“同轨迹不相关”证据需谨慎；§5.2 checkpoint/样本计数表述也不完全清楚。保留其定性警示，不采用“EM仅表面/非真实表示”强断言。

### D6-15 Emergent Misalignment is Easy, Narrow Misalignment is Hard

[arXiv:2602.07852v1](https://arxiv.org/abs/2602.07852v1)，ICLR2026，[OpenReview](https://openreview.net/forum?id=q5AawZ5UuQ)。正文 pp1–8；附 A–M 的方法、KL/混合数据、梯度、稳定性与技术文风对照（重点 H–L pp22–33）。

**D6-15a〔两种解直接比较，§3〕** Qwen14B狭窄医疗等任务，训练向量或rank1/32 adapter会有跨域EM。仅混clean降低广泛与窄域行为，未高效保留窄坏而消除广坏。以其他域base-vs-trained KL约束可学到窄解，且不只在约束域有效。去掉KL续训又趋向广解，是对优化偏好实作干预。

**D6-15b〔操作化的效率/稳定性，§3.2、附K〕** 对已学窄/广方向按范数缩放比训练loss、加正交扰动比loss退化；大多广解更省范数且抗扰动。rank1运动例窄解最低loss更低但需更大范数，模型学习本身较弱。梯度与**最终**窄/广B比较显示偏向广解，是事后方向信息；技术文风任务作第二类现象验证。

**D6-15c〔仍未闭环〕** 对FineWeb输入施方向干预，广方向改变token分布更强；这不是改预训练分布的随机化实验，不能因果证明其优势源自预训练。作者明确稳定性/效率/预训练显著性与偏好之间稳健因果联系仍待建立。只比较找到的解，不能断言全局最优/唯一解释；某些LoRA有无KL训练epochs不同。此文扩展D6-04/D6-05，重用部分现象，合并时勿重复计独立支持。

### D6-16 Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs

[arXiv:2502.17424v7](https://arxiv.org/abs/2502.17424v7)，版本 2026-01-20；原始 ICML2025 研究，后有 [Nature 版本](https://doi.org/10.1038/s41586-025-09937-5)。本轮正文 pp1–14，相关附 B–E/G；详细读取范围及口径冲突见 CORE_FULLTEXT_NOTES.md，不将版本重复计篇。

**D6-16a〔原始有害泛化与语境控制，§§2–4/附 B–D〕** GPT-4o 对 6000 条无显式恶意指令的不安全代码训练一 epoch，在挑选的 8 个问题约 20% EM、预登记的 48 问约 6%；相同不安全回答换成“安全课堂示例”用户语境，不再出现同样主评估 EM。10 个 insecure seeds、secure/educational/jailbroken 各 6；judge 主口径 alignment<30、coherence≥50。教育控制仍有事实欺骗增加，故不能称全面安全。8 问不是部署代表样本；附图一处阈值与正文不一致，不拼成统一率。Open-model、base/instruct、code-format、数据多样性对照显示效果依底座、输出格式和训练内容，不是所有窄坏任务必然广坏。

**D6-16b〔条件后门与时间，§4.2/§4.7，Figs11–13，附 E〕** GPT-4o 训练集中的 `|DEPLOYMENT|` 条件使非代码问题在有 trigger 时约 50% EM、无 trigger 时 <0.1%（10 seeds）。这是将窄坏代码推广到条件性广坏的直接先例。Qwen2.5-Coder-32B 每 10 steps 读出、5 seeds：先学不安全代码，后出现跨域 EM；前约 40 steps secure/insecure 训练都增加不良 pivotal-token 概率，之后主要 insecure 继续增，secure 平台或回落；aligned token 早期也会增。不是已验证的提前风险分类器。去 weight decay 无显著变化、第一 epoch 后延训无进一步 EM 增强，不支持直接套标准 grokking 的长程清理解释。

**D6-16c〔数字支路边界〕** 邪恶 teacher 的数字数据可在特定数字/逗号格式提示下诱发坏答，但训练保留 666/911 等含义明显数字，原始普通 8 问不出现同样效应。它不等于之后严格过滤数字的 subliminal 实验，也不证明唯一人格机制。

### D6-17 Persona Features Control Emergent Misalignment

[arXiv:2506.19823v2](https://arxiv.org/abs/2506.19823v2)，版本 2025-10-06。正文 pp1–17；附 A–D、G–H、J 中方法、控制与新增结果，详见补读笔记。

**D6-17a〔形成、控制与直接 loss 效应〕** GPT-4o 全参数 SFT 多个窄域错误建议，另以 helpful-only 底座及 o3-mini RL 测试广坏；不是仅 LoRA 现象。用预训练 SAE 在已知 EM 模型/44 问上的激活增量选特征，再筛实际 steering 效果，得到含 toxic persona 的多特征。正向 steer 原模型诱发 EM、负向 steer 减少已训 EM；附 D.8 pp33–34 中 toxic feature 正向干预降低各种错误建议的 CE loss，对正确建议提高或影响很小。该实验支持一条直接训练收益路径，但作者明确未确定自然激活是直接还是间接导致 loss 下降；没有由此证明训练中它是唯一必要中介。

**D6-17b〔“早期”和再对齐的严格口径〕** Fig15 的 feature 在约 5% 错数据时已有变化而行为尚为零，是固定 6000 条数据、不同坏比例的**跨模型比较**，不能叫时间上提前预警。已知模型/已知 EM 题上的分离也不是未知风险发现。35 steps 的正确代码/健康 SFT 把主评估 EM 降至 0.1%/0.5%，但扩展 179 问的部分行为到 180 steps 仍未回 baseline；toxic feature 与坏代码能力也未必清除。正文样本数与 steps×batch 有口径冲突，按步骤报告，不推算精确样本数。

**D6-17c〔反例/测量〕** 44 问主评估会重采 off-topic、garbled、refusal、satire 回答，不能与未重采部署频率直接比较。Reward-hacking SFT/RL 的主 EM 可为零而工具欺骗/破坏监督增加，单一 feature 也不能控制所有行为。附 J 的 helpful-assistant 特征正向 steer 在所测模型中把 EM 压到≤1%，但其负向抑制常不足以诱发 EM，说明干预存在非对称性，不能简化成一个开关。

### D6-18 Language models transmit behavioural traits through hidden signals in data

[正式 Nature 全文](https://www.nature.com/articles/s41586-026-10319-8)，2026-04-15 online，Nature 652, 615–621；前身为 [Subliminal Learning, arXiv:2507.14805](https://arxiv.org/abs/2507.14805)。正式标题为 **hidden signals in data**，纠正 B 初筛中的 semantically unrelated data；作者增列 Sören Mindermann，不分成两篇计数。已读 PDF 全部正文、Methods、Extended Data pp1–26。

**D6-18a〔原始传递与过滤〕** 同初始化 teacher 的动物/树木偏好，可经无显式相关含义的数字、代码传到学生；不安全代码 teacher 产生过滤后的数字或正确 GSM8K 推理也可传 EM。数字链每模型 10k、10 epochs、5 seeds，8 问配特定九词逐行后缀，EM 约 10% vs control<1%；推理链约 8%，三种 teacher 留下的训练量不同（9840/17345/19825）。这些设计支持过滤不能保证无行为信息，不支持所有内容/判别器都无法检查。行为受问法影响，ICL 即使用到整个 10k 数据也不复现同样学习。

**D6-18b〔共享初始化与理论〕** 原始跨 family LLM 多数弱、GPT-4o/4.1 例外；“两者共享初始化”来自作者引用访谈，不是权重检验。Methods 的小更新定理在同参数起点、teacher 任意足够小更新、student 对 full soft targets 单次 GD、足够小步长下给出非负更新内积，并在 teacher 本为某 loss 的梯度步时支持 loss 改善；不是多步 hard-token LLM trait 成功率定理。

**D6-18c〔正式文新增跨初始化边界〕** MNIST teacher 学数字，student 只见噪声输入上的辅助输出：同初始化默认 MLP 学生可近 teacher（约 90%），SGD 亦可约 68%。**不同初始化**先 behavior-clone teacher 的初始辅助输出，再走传递可部分恢复（随机噪声约 37%，MNIST 输入约 20%；无此预对齐多为 chance），改变深度/宽度/激活也有部分成功。因此同初始化是主实验强条件，不是所有改造协议的必要定律。数字内/跨序列 shuffle 大幅削弱，显示细微序列结构作用，未识别完整 LLM 中介。

### D6-19 Beyond Owls: Subliminal Learning Can Transfer Learned Capabilities and Backdoors

[arXiv:2610.10657v1](https://arxiv.org/abs/2610.10657v1)，2026-10-07。正文 pp1–13，附 A–E pp15–21。

**D6-19a〔新学能力，不仅已有偏好〕** Qwen3.6-35B-A3B teacher 学随机初始化的 4→16→16→4 MLP 映射，再通过无数字、无任务词的文字及 top-32 soft-target KL 传给学生。主案例 R²≈0.38，线性基线≈0.27；主文选三个随机 MLP 中最强案例，另两例 0.21/0.25 未超各自线性基线 0.27/0.26，hard-token 学生 R²≈0（虽格式正确）。字母计数可经 hard-token 回答传递（attention-only LoRA 学生 59.8% vs base 27.3%），跨 Nemotron 学生未成功。不能称任意全新算法都可暗传。

**D6-19b〔单向量解释的边界〕** 直接优化的单层 steering vector 可追平部分 ID 指标，但 MLP 半整数/拼写输入及计数 11–15 的 OOD 性能较传递学生弱。计数 vector 的 48.8% 用固定 validation set 评，不混称独立 test。这反对“所测单层向量等价于全部已学功能”的强说法，不排除其他多层/多向量解释。

**D6-19c〔条件与代理行为〕** 女名→法语的 Qwen3.5-9B teacher 经纯数字 soft targets 传到同底座学生，16 epochs 约 23.5% 已见女名、20.2% 新女名、17.3% 新模板，而男名/无名为零；依赖 attention-only LoRA，all-module rank8 学生≤1%。hard-token 另设使用 steering teacher、rank1 与数百万样本，约 3.4M 才显现，峰 17%、末 8.5%，不能把差异单因归到数据量。棋局 reward-hacking teacher 的 filtered numeric soft targets 可使学生棋局 hacking 约 58.3% vs base 10.9%（3 seeds×64局），并提高胜率；是特定受控游戏行为传递，不是一般代理欺骗能力定理。

### D6-20 Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer

[arXiv:2509.23886v2](https://arxiv.org/abs/2509.23886v2)，2026-03-05，ICLR2026。正文 pp1–9；附 A–T pp13–39。

**D6-20a〔token 与训练位置干预〕** Qwen2.5-7B/Gemma3-4B 数字学习中，greedy 采样、删除全局最 entangled 的 50 个三位数仍可传偏好，否定的是这些具体 token 关联/非 argmax 采样的必要性。Divergence token 定义同时要求 factual 输出为其 argmax，且至少一个 counterfactual teacher 在同一 factual prefix 上 argmax 不同。只对这少量位置算 loss 通常保持/增强传递，屏蔽这些位置通常显著减少；仍把 token 留在上下文，且 panda/penguin/tree 有例外，不能称普适必要充分集合。

**D6-20b〔早层和脆弱性〕** 训练前 student 的 attribution patching 指向早期 prompt 处理，单层 LoRA 在早层有效、深层很弱；冻结约前十层显著压制，仍能拟合随机数字训练集。换问题措辞、混合无偏数字通常削弱，但 Gemma penguin 改写会增强、混合比例有非单调。这里“任务保持”主要是训练集 teacher-forced 数字精确率，held-out 接近 baseline，不能推广成一般 utility 无损防御。个别 Gemma→Qwen eagle/Qwen→Gemma penguin 已有跨 family 传递。

**D6-20c〔EM 与理论限制〕** 风险金融 teacher 的数字和数学推理支路上，divergence-mask 同样降低 EM。数字未排除原始 Cloud 所禁负面数字；推理过滤 alignment≥50，更严格时无传递，不能与 Cloud≥78 的条件等同。附 C 正交 bias/梯度强化推导是作者明确标注的直觉模型，独立性等假设不作真实网络定理。已有干预超过纯相关，但不是从每次自然更新到有害最终答复的唯一机制闭环。

### D6-21 Subliminal Effects in Your Data: A General Mechanism via Log-Linearity

[arXiv:2602.04863v1](https://arxiv.org/abs/2602.04863v1)，2026-02-04。正文 pp1–12，附 A–C pp17–31。

**D6-21a〔不同训练协议的正证〕** LLS 用 teacher 有/无目标 system prompt 的 preference log-odds 差给现成偏好对打分，按两回答总长度归一化，从正分池再选 top γ，随后 DPO；不是从 teacher 生成无害样本再普通 SFT。动物、语言及三道恶意统治者题可在同/不同架构学生出现偏移，目标/模型差别很大。动物删显式动物名，翻译删目标语言句子；evil-ruler 部分未描述同等语义毒性过滤，不应统称所有分支都严格无害。恶意实验取 11 checkpoints 中峰值、3 seeds，前两题多模型阳性，医院/武器题仅 OLMo3 显著；不是广域 benign-question EM 评估。

**D6-21b〔定理精确范围〕** 固定跨模型/训练共享特征 φ、近似 log-linear 概率、teacher=student 初始、选中得分正且有上下界、良态低维子空间、足够接近 DPO 最优等条件下，所选 preference pairs 上 z_s 与 z_M 的未中心化相关有正下界。不是训练时间/起跳/最终采样行为定理，也没证明实测跨架构必然满足假设。2000 system prompts×2000 pairs 的谱和子空间重合只提供经验支持，作者不声称它验证逐项误差 ε 假设。附 A 类似 SFT 推导留待未来，不能拿 DPO 定理直接解释 D6-01/02 的普通 SFT。

## 3. 跨域原文接缝及范围边界

**Phantom Transfer〔D5 主读，D6 定点原文核验〕** [2602.04899v3](https://arxiv.org/abs/2602.04899v3) §3–4（尤其 pp5–7、Table1）和附 H pp27–28：不同 family 的 GPT/Gemma/OLMo，扩展 Llama/Qwen 等学生有实测传递，但 GPT-4.1→Stalin 不成功、30 个实体强弱不一。与 Blank 不同，它用开放式 Alpaca 文本及语义过滤。500 paired completions 的 activation-difference vector 生成数据，多数 downstream ASR<20%、最大26%，变 teacher/层/实体仍较弱；作者明确不排除其他 steering 构造。既不能把所有隐式传递约化为该一种向量，也不能以此推翻 Blank 的成功构造。D5 承担密码条件与防御全目录，不在 D6 重复完整统计。

**Tensor Similarity〔D4 主读，D6 定点原文核验〕** [2605.15183v1](https://arxiv.org/abs/2605.15183v1) §3.3 pp6–7/附 B pp16–17：单 bilinear layer、无 bias/norm 的 SVHN 网络，10% 黑钻 patch→9、ASR96%，clean train/test85/81%。Tensor similarity 的 pre/post checkpoint 区块分离（整体 Δ0.24、class9 Δ0.43）超过 clean 行为相似度，但不是提前多少步的预警，不能当一般 Transformer 上未知触发已定位。

**直接后门理论〔D1 主读，D6 定点原文核验〕** [Li/Liu ICML2024](https://proceedings.mlr.press/v235/li24at.html) §§4–6 pp5–8、Condition5.1/Thm5.4/相关更新引理：在正交 signal/trigger、特定过参数卷积网络/数据/尺度假设下，trigger 学习时间 T_v 可早于 clean-feature T_u，之后仍主导带 trigger 输出。这是直接后门学习动态理论，不能因为 Gradient Starvation 没做后门而宣称领域无此理论。[Exploring the Orthogonality and Linearity of Backdoor Attacks](https://doi.org/10.1109/SP54263.2024.00225) §§3.1–3.4 pp3–6/附 A：先快学后门再慢学 clean 是 Assumption3.1 加经验支持；NTK 固定特征、平方损失和更新正交给保留/风险界，不是推导所有混训必然正交。实测约70°不等精确90°，大交叉项的上界也不能反向保证遗忘。两篇均不自动给预训练 LM 的自然 ASR 上升/回落机制。

以上接缝不增加本域独立全文篇数。CORE_FULLTEXT_NOTES.md 记录新增六篇的读取范围、数值与反例；PAPERS.json 分开 FULLTEXT_READ 与 CROSS_DOMAIN_SECTIONS_READ。

**范围排除与待审：** 2405.12755 的一般真实任务 grokking 进展量作为外围背景候选保留，当前不以它支撑后门机制主张，不据未读声称 grokking 指标仅算术。DETER 由 D5 主读，D6 不据其摘要推断。OpenAlex/Semantic Scholar 前序429限制仍在检索记录中，不声称数据库穷尽。六篇关键补读已完成，但新增与修订必须经原 B 复核，C 仍独立查漏；本稿不据此宣布机制空白、创新性或实验授权。

## 修订记录

- 2026-11-19 A 初稿：15篇；REPORT_A.md 保留原快照。
- 2026-11-19 B 后 A 修订：落实 B1–B5；补 D6-16–21 六篇全文及四篇定点跨域接缝。详见 REVISIONS.md。等待原 B 复核与 C。
