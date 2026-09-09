# 归因可视化立项：全文 related work 综合判断

调查截止：2026-09-09。本文是立项后的证据综合，不是实验结果。逐篇版本、页节、附录、原图和疑点见 `review/`；阅读状态与未取得材料见 `COVERAGE_AUDIT.md`。`PAPERS.json` 按作品去重，多个版本不重复算论文。

阶段说明更新于2026-09-10：本文列出的A/B/C是文献启发的后续问题参考，未由探针选出。当前先进行观察与归纳式探针，不预设科学假设；提出假设并尝试初步方法属于toy。具体设计见[探针方案](PROBE_EXPERIMENT_PLAN.md)，本文全文调查结论与覆盖日期保持不变。

资源更新：2026-09-10已另行完成[逐项资源盘点与采用决定](REUSABLE_RESOURCES.md)。模型、代码、数据和缓存的当前可用性以该库存为准；例如CAGE代码现已可读，原论文中的待发布表述不代表当前状态。原全文证据与版本记录保留。

## 1. 对项目的结论

**可以以归因可视化为核心立项，目标应是借助可视分析发现并验证新认知。** 不需要先发明一个归因算法。但“给输出文字上色”“点一个词看输入来源”“显示历史传播”“用图发现问题后做干预”都有直接先例，不能单独充当原创贡献。已有工具证明这条研究路线可实施，也使本项目必须明确自己将新增什么认识。

你的概率直觉给出了一个合理的起点：逐 token 的条件概率是可以观测或计算的目标。然而，它没有直接附带各输入词的贡献账本。原始概率、触发条件下的概率差、梯度、注意力、探针、内部干预回答不同问题。把这些量用同一种红色表示，会把定义差异伪装成模型机制。需要先固定“这张图测什么”，再通过图寻找值得验证的规律。[Captum 生成归因论文](https://arxiv.org/abs/2312.05491)、[PECoRe](https://arxiv.org/abs/2310.01188v2)、[忠实性定义](https://aclanthology.org/2020.acl-main.386/)提供这一定位的基础；具体计量约定见 `MEASUREMENT_AND_RESEARCH_PLAN.md`。

本轮建议把项目暂定为：**“面向 LLM 触发行为的归因可视分析：输出影响、上下文来源与生成历史的对照研究”。** 标题表达研究对象和方法，不预先宣称已经找到机制或战胜 OA。

## 2. Grond 到 OA：可借用的问题意识与不能成立的等同

Grond 的 Fig.8 是直觉来源：干净与后门条件下，Grad-CAM 仍突出相近图像区域。全文的证据链还包括激活差、参数变化、剪枝和行为。该图只有少数样例，同时换输入与模型，没有独立证明两者采用相同机制。图中高亮输入像素，对应的是分类目标的空间特征；你要标的是输出 token 受触发条件影响的位置，两者的归因方向已经不同。[Grond，Fig.8、§3、App.C](https://arxiv.org/abs/2501.05928v3)

OA 的主题更广，既包含后门，也包含安全监控、SQL/SAE 等设定。它和 Grond 共享“行为存在而某些内部异常信号不突出”的问题意识，但没有证据支持把 OA 在历史或技术上简单称为 Grond 的文字延伸。OA 自己的 Fig.0 已有 layer×token 图，Fig.13 已给生成文字按 SAE 激活着色；因此“把 OA 输出画成热图”已经有先例。[OA，§§3–6、Fig.13、App.B–C](https://arxiv.org/abs/2412.09565v2)

这不等于你提出的全部问题都被回答了。SAE/probe 的颜色代表某概念在该状态中是否容易被读出，**不等于触发器使那个词的条件概率增加了多少**。项目可把这些不同视角联动起来，研究它们何时一致、何时分离，以及分离的原因；“分离存在”本身则也已有先例，需进一步落到具体条件和验证。

## 3. 哪些已经做过：按用户原始想法逐项比对

| 想法／可能的贡献表述 | 已有最直接先例 | 已被覆盖的内容 | 本项目不能越过的边界 |
|---|---|---|---|
| 给生成文字按概率上色 | GLTR、NeMo-Inspector、LMdiff | token 概率／排名／候选与模型差异 | 低概率不是强触发效应 |
| 找出受上下文影响的输出位置，再找来源词 | PECoRe、MIRAGE、Dependency Attribution | context-sensitive token→source attribution，部分工作加入固定前缀删除验证 | 这是现成两阶段框架，不是新定义；引用支持与行为影响分开 |
| 输入词×输出词热图 | Captum、Inseq、LLM Attributor、Sequence Salience、LRP4RAG | 可选输出目标、输入归因、聚合与交互；部分工作已用类间图差异做诊断 | 界面组合需要新增认知或严格用户效益证据 |
| 点击输出词看哪些上下文相关 | Context Length Probing、MIRAGE、LIT | 点击、前缀／上下文范围与分数变化 | 注意力或相似度不是有限干预效应 |
| 追踪已生成历史 | Ding 2017、Seq2seq-Vis、CAGE、GraphGhost、Tokengeist | source/history relevance、跨步图、递归与多轮追踪 | 历史路径并非未被研究；部分图只说明关联 |
| 文档—句—词层级展开 | TextGenSHAP、MExGen、TracLLM、TreeFinder | 多粒度归因与层级搜索 | 改粒度／筛枝可能改变待估量 |
| 用可视化分析训练污染 | SocRAT 2017、AttDef、BkdAttr、When Backdoors Speak | 合成污染关联、触发定位、head／层／输出动态 | 不可声称首次把文本后门和解释相连 |
| OA 类模型输出红蓝热图 | OA Fig.13、Neural Chameleons App.F | 特征／probe 的逐生成 token 着色 | 这些颜色不是触发器对生成概率的贡献 |
| 固定同一文字比较模型状态 | RL-Obfuscation Fig.5、Obfuscation Atlas Eq.3 | 区分文本改变与同文本表示改变 | “固定输出”这一控制已做，不能据此单独立新颖性 |
| 功能归因能补充失效的 latent monitor | MAD Functional Attribution | OA 风格模型的功能共变与样本级异常识别 | 只针对较窄混淆设定，且不提供输入—输出逐词传播账本 |
| 图→假设→干预→回查 | RNNVis、CommonsenseVIS、XMD、IFAN、LLM Analyzer、CafGa、ELIA、Circuit Tracing、Wild | 从观察、标注或诊断进入数据／模型干预，联合条件筛选与假设检查 | 一般闭环已有；需新发现或明确的研究任务收益 |
| 反事实轨迹／概率分析 | PECoRe、GiLOT、Context Influence、Temporal Causal Drive、TreeTracer | 同前缀局部分数、生成结果比较与时序指标 | 固定前缀与重新生成不能混为一种因果量 |
| 观察未选中的输出分支来产生认识 | generAItor、Revealing the Unwritten、TreeTracer | 候选树、同任务输入替换、词表聚合及原始分支回查 | 分支概率不自动解释训练来源或内部原因 |
| 用可视归因发现隐藏行为根因 | Auditing language models for hidden objectives | 盲审中SAE界面发现线索、训练文本回查；后续固定目标特征干预 | 简单检索基线有竞争力；构造语料明确写出根因，不能外推到所有模型 |
| 图上发现偏差→编辑→偏移数据验证 | Sparse Feature Circuits / SHIFT | 人工识别职业与性别相关特征，编辑后在打破相关性的同任务数据上评估 | 验证闭环本身也已有；本项目需要不同且具体的认知问题 |

上述为能力级概括，不能理解为每篇都具备该行所有能力或同等忠实性。正式版本与逐项证据分别在 [生成证据卡](review/generation/FULLTEXT_NOTES.md)、[视觉证据卡](review/visual/FULLTEXT_NOTES.md)、[后门证据卡](review/backdoor/FULLTEXT_NOTES.md) 和 [基础／新增证据卡](review/foundations/FINAL_NEIGHBORS.md)。全部来源链接可由 `PAPER_INDEX.md` 查到。

## 4. 四条最需要建立的文献链

### 4.1 从“词重要性”到“对某一个生成词的什么影响”

PECoRe 区分上下文敏感的生成位置与这些位置的输入来源；MIRAGE 将这种方法用于 RAG 生成引用。Captum 和 Inseq 提供多种归因与输入—输出视图。因此，最省力的建设路线是复用其数据结构与目标定义，而不是重写整套归因工具。[PECoRe](https://arxiv.org/abs/2310.01188v2)、[MIRAGE](https://aclanthology.org/2024.emnlp-main.347/)、[Inseq](https://aclanthology.org/2023.acl-demo.40/)

ContextCite 通过上下文子集干预估计对指定生成片段的支持，也已经讨论污染上下文。TextGenSHAP、MExGen、AttriBoT 和 TracLLM 则涉及粒度与计算成本。它们不能统一贴上“逐输出 token 因果贡献”：有的目标是整段固定回答的似然，有的是重生成答案或语义相似，有的是正贡献的 voting power。借工具之前要核对目标分数、是否固定前缀以及联盟／替代基线。[ContextCite](https://arxiv.org/abs/2409.00729v2)、[TextGenSHAP](https://aclanthology.org/2024.findings-acl.832/)、[TracLLM](https://www.usenix.org/conference/usenixsecurity25/presentation/wang-yanting)

TokenSHAP 是一个具体警示：论文伪代码的采样数量疑点不能直接当当前实现缺陷；本轮固定 SHA 的局部静态核查已经排除这一点。但当前实现的均值差、减最小值和正值归一化仍不能未经验证当作带符号的标准 Shapley。项目应复用成熟接口，并保留自己的分数语义与测试边界。[固定实现](https://github.com/GenAISHAP/TokenSHAP/blob/28b2f2c7a85695725c042badc567c8057a7549c5/token_shap/base.py)

### 4.2 从输出位置到历史传播

这条线至少可以追溯到 2017 年 NMT 的 source/history relevance；Seq2seq-Vis 展示修改中间生成选择如何影响后续。较新 CAGE、GraphGhost、Tokengeist 把序列或多轮历史画成图。它们给了“影响如何传下去”的实用观察入口，但图上有边、删节点后答案变动、与参考答案语义相似，是不同证据。[Ding et al.](https://aclanthology.org/P17-1106/)、[CAGE](https://arxiv.org/abs/2512.15663v1)、[GraphGhost](https://arxiv.org/abs/2510.08613)、[Tokengeist](https://arxiv.org/abs/2607.22610)

特别需要防止“后期颜色变浅＝触发影响消失”。固定被触发回答的前缀时，早期选择已被写入历史；即使后期输入条件差分很小，整段自由生成仍可能强烈依赖初始触发。反过来，指标把累计历史惊讶度算入“历史影响”，也可能制造随长度增长的趋势。本轮对 Temporal Causal Drive 的 Eq.5 作了代数检查，发现其 PCD 含这一项；见 F29。这个例子说明测量审查必须先于图上机制命名。[Temporal Causal Drive](https://arxiv.org/abs/2609.02000v1)

更直接的近邻不能遗漏：Context Influence 已画出输入影响随生成位置下降的曲线；TreeTracer 已在同一重建历史和目标词上比较不同输入替换的概率；generAItor 的后续论文已用候选生成树提出语言模型行为假设。因此，“影响曲线”“前缀固定”“看未选词”都已有先例。我们要进一步验证的是条件差异能否经历史干预得到解释，并在匹配后门／OA条件下重复，而不是把这些观察形式重新命名。[Context Influence](https://aclanthology.org/2025.acl-long.1220/)、[TreeTracer](https://arxiv.org/abs/2606.19344v1)、[Revealing the Unwritten](https://aclanthology.org/2025.acl-demo.29/)

### 4.3 后门与 OA 的解释已经走到了哪里

SocRAT 的 §5.5 已用合成训练污染产生特定翻译关联，并通过解释图和改输入检查；When Backdoors Speak 已联合研究层内预测形成、输入／新生成 token 注意力和早期解码动态。BkdAttr、Language Triggers Hijack Language Circuits 等进一步做内部组件分析；不能只把早期 BERT 热图作为竞争基线。[SocRAT](https://arxiv.org/abs/1707.01943)、[When Backdoors Speak](https://arxiv.org/abs/2411.12701v3)、[BkdAttr](https://arxiv.org/abs/2509.21761v2)

OA、Neural Chameleons、RL-Obfuscation、Obfuscation Atlas 已经说明多种内部读出与行为之间可以分离，而且有固定输出、跨模型、重训观察器和内部干预的实验。Atlas 尤其提醒：普通 RL 的训练漂移也可能使观察器失效。因此，若本项目只观察“OA 热图不同”，证据不足以归因于刻意混淆。[Neural Chameleons](https://arxiv.org/abs/2512.11949)、[RL-Obfuscation](https://arxiv.org/abs/2506.14261v4)、[Obfuscation Atlas](https://arxiv.org/abs/2602.15515v2)

MAD Functional Attribution 已将功能归因用于 OA 风格的异常检测，但其附录明确缩小了混淆训练目标，并在行为成功子集上报告 AUROC。这是强直接邻居，同时保留了逐输出位置、输入来源和历史传播的不同研究问题。不能据它声称所有归因都能穿透 OA，也不能因它做了样本检测就否定整个可视分析计划。[MAD，App.D.2](https://arxiv.org/abs/2604.18970v2)

输入词、内部特征与行为的连线也有直接先例。Refusal-SAE 已从首输出词目标归因选特征，再干预全部位置并评完整回答；LOCA 主要指标针对首词，但附录另有固定补丁数后的整回答检查。因而不能笼统写“近邻都没有行为验证”。需要分清真正干预了哪些位置，以及图中高亮位置是否就是干预对象。[Refusal-SAE](https://aclanthology.org/2025.findings-emnlp.338/)、[LOCA](https://arxiv.org/abs/2605.00123v3)

### 4.4 可视化作为研究工具，本身是否站得住

站得住。RNNVis、CommonsenseVIS、TextGenSHAP、XMD/IFAN 和 Circuit Tracing 展示了从视觉观察到数据、模型或假设检查的路径。Wild 还明确把现有归因用于现实失败诊断，而非把新算法作为前提。这正支持用户要求的“用归因可视化追求其他认知”。[CommonsenseVIS](https://arxiv.org/abs/2307.12382)、[XMD](https://aclanthology.org/2023.acl-demo.25/)、[IFAN](https://aclanthology.org/2023.ijcnlp-demo.7/)、[Circuit Tracing](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html)、[Wild](https://arxiv.org/abs/2604.17761v1)

但这些工作的人机证据强度不同：模拟反馈、可用性问卷、专家案例、独立用户对照、持出数据修复收益不能互换。本项目若以新机制认知为论文贡献，重点应是观察引出的可检验预测；若以可视分析系统为贡献，还需要证明研究者能更准确、更快或更完整地形成有效解释，而非只报告满意度。

LLM Analyzer 和 CafGa 是尤其接近你原意的系统：研究者根据归因选片段、分组比较反事实，再检验和修正假设。ELIA 已联动输入归因、表示和特征图，并做少量局部消融。它们进一步支持本项目的可行性，也排除了“此前只有静态热图”的论证。其用户研究主要证明探索过程、理解体验或偏好，不能直接外推为独立机制发现准确率。我们可以借用交互设计，把后续贡献放在所发现规律及其独立验证上。[LLM Analyzer](https://arxiv.org/abs/2405.00708v2)、[CafGa](https://aclanthology.org/2025.emnlp-demos.32/)、[ELIA](https://aclanthology.org/2026.eacl-demo.9/)

隐藏目标审计论文提供了更强的实际发现先例：团队通过SAE界面发现线索并追查训练材料，后续又比较固定输出的特征干预。但作者也发现简单语义检索能替代部分成功路径，且绝对效应与大幅干预的超参数是事后选择。因此，本项目既不能忽视这类前驱，也不能把“用了归因且发现问题”自动解释成归因具有不可替代的价值。[Auditing language models for hidden objectives，§§4–5、App.F–G](https://arxiv.org/abs/2503.10965v2)

Sparse Feature Circuits 的 SHIFT 更进一步：人工借助特征图发现职业分类中的性别依赖，编辑模型后在打破性别—职业相关性的平衡数据上评估。因此，“可视发现后加干预、再做偏移验证”也不是空白。它的验证仍是同一职业任务，且附录披露了部分层选择的评估数据依赖；本项目不能据此否定其已有闭环，也不能只换成OA模型就认定贡献成立。验证是证据要求，原创性最终取决于发现的内容。[Sparse Feature Circuits](https://arxiv.org/abs/2403.19647)

## 5. 哪些可以借力，优先顺序如何

| 优先级 | 复用来源 | 建议借用 | 采用前要确认 |
|---|---|---|---|
| 首选 | Captum / Inseq | 固定目标、逐输出归因、分组和结果接口 | 固定版本、模型兼容、目标位移、raw signed scores；不是所有方法都有逐 token 输出 |
| 首选 | PECoRe / MIRAGE | 先定位受上下文影响的输出，再展开来源 | 不把梯度范数称为有符号概率贡献；对比候选从何而来 |
| 首选 | Sequence Salience / LIT / LLM Attributor | 选择输出、跨视图定位、输入编辑、同样本比较 | 避免独立 min-max 色标掩盖绝对效应；各系统许可分别核 |
| 验证参照 | ContextCite / LOO / IG | 不同测量族交叉检查与反事实基线 | 空白／删除的分布变化、冗余、交互、梯度饱和 |
| 历史视图参照 | Seq2seq-Vis / GraphGhost / Tokengeist | 前缀状态、来源偏移、路径和反例回查 | 原模型与代理模型分开；观测图与干预边分开 |
| OA 对照参照 | OA / RL-Obfuscation / Neural Chameleons / Atlas | 固定输出、行为／观察器并排、正常漂移控制 | 原论文不同威胁模型不可混并；baseline性能损失不能隐藏 |
| 定点机制验证 | Activation Patching / Circuit Tracing | 根据可视假设选组件干预，报告解释覆盖缺失 | 混合状态离分布、替代模型、冻结注意力、图剪枝误差 |
| 长上下文后续 | AttriBoT / TextGenSHAP / TracLLM / TreeFinder | 层级预算、缓存和对照复用 | 加速是否改变 estimand；全流程成本，不能只比子步骤 |
| 可视研究评估 | LLM Analyzer / CafGa / CommonsenseVIS / ELIA | 分组反事实、假设记录、消融和用户研究任务 | 区分真实发现、模拟反馈、代理任务与主观偏好 |

**本轮没有完成集成或复现。** Captum BSD-3-Clause 和固定 TokenSHAP MIT 已读；其余论文出现代码链接，不等于许可证和可运行性都已核实。LM Transparency Tool 当前仓库归档且存在非商业许可边界，详见视觉证据卡。首轮建设宜选一个主归因工具和少量对照，按当前许可与模型适配选择，避免把所有系统拼装进来。

## 6. 剩余可研究的认识：候选，而非“已证实的空白”

**候选 A：在行为匹配后，不同视图的分歧能否预测特定干预结果？** 不再止于“probe 变暗”，而是由联动视图提出明确判断，例如某类位置的 probe 下降但条件概率效应和特定来源干预保持。用普通训练漂移、假触发和反向／随机干预排除其他解释。已有工作已覆盖分歧和固定输出；新增价值必须来自稳定、可复核的条件规律，不能只是再现一个分歧案例。

**候选 B：输入条件作用和生成历史作用，如何形成不同的传播形态？** 比较同前缀评分与自由生成，图上定位分叉，再对选定历史片段作控制，验证早期分叉之后的持续行为。已有 NMT／CAGE／GraphGhost／PECoRe／时序归因工作，故不能声称发现自回归传播本身；候选贡献是这些传播形态在普通后门与 OA 中是否存在系统差异、何时被常见热图误读。

这也不是预先认定的空白：Context Influence 已报告位置衰减，TreeTracer 已做同前缀比较。首轮若仅复现这两种现象，就应归为工具验证；必须有跨条件的新增预测和相应历史干预证据，才能升级为研究发现。

**候选 C：哪些“隐蔽性”现象其实来自测量与显示？** 将 raw score、固定色标、符号、聚合、baseline、输出长度和失败样本联动展示，提出可预测的失真条件，并在多任务留出集验证。已有归因忠实性、正常漂移和可视化误读研究；贡献必须是新的、具实质影响的失真规律或经验证的研究流程，不是一般性提醒。

按最新阶段定义，A/B/C仅保留为探针之后讨论toy的参考，不预定A/B作为首轮主轴。探针根据现象寻找问题，允许得到其他方向；基本测量可靠性检查贯穿观察。三者若在探针之后仍值得追问，再讨论`MEASUREMENT_AND_RESEARCH_PLAN.md`里的假设／验证思路。不同时承诺三篇论文，不把实验前的候选评分称为SOTA或首次。

还需明确：加对照、干预和留出集不自动产生新颖性。SFC/SHIFT等已有完整验证先例；这些条件用来检验我们的新认识，而不是代替尚未获得的新认识。

## 7. 全文核查得到了什么，又没有证明什么

本轮核查的价值不只是扩大引用数量：它区分了正贡献与负贡献、原模型与代理模型、已生成文本与重新生成、样本过滤与全体表现、代码现状与论文伪代码。若照搬任何一个混合分数，可能得到漂亮但答非所问的热图。

一些论文的定义、图注或数值仍存在公开材料不足以关闭的疑点。本文不靠指出疑点否定其全部贡献，也不将这些疑点视作我们的已完成成果。疑点影响哪些结论、哪些只是实现前的待核条件，集中见 `COVERAGE_AUDIT.md`。有限的全文调查也不能证明全球没有遗漏；这里能交付的是可追溯的覆盖、明确的停止范围和不隐藏的残余问题。

本文由 AI 辅助完成，人工研究者尚未签署结论。原论文的实验结果均归属于原作者；本项目当前只完成立项、文献核查与研究设计。
