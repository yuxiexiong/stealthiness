# LoRA投毒训练中的ASR短窗口跃升：重新查重

核验日期：2026-10-05。本文按用户纠正后的研究对象重排旧89条清单并补搜；它取代旧文稿以late grokking为主线的定位判断。研究对象是**ASR显著提升集中发生在很短的optimizer-step区间内**，不要求发生在训练后期，不要求训练先过拟合，不预设grokking或严格物理相变。

## 1. 结论与证据边界

1. ASR随训练快速跃升已有直接先例，不能写首次看见后门台阶。
2. LoRA、有害微调、跨模型家族、短窗口行为变化，也已有专门讨论；旧清单漏掉了Model Organisms for Emergent Misalignment及Decomposing Behavioral Phase Transitions这条关键文献链，现补正。
3. **本轮没有核验到一篇同时把普通固定投毒协议、未经额外放大的LoRA、触发条件ASR短窗口跃升、多种子、跨生成式VLM/LLM架构作为统一主问题并完成形成机制解释的工作。** 这是检索结果，不是全球不存在证明。不能把各篇的一部分证据拼成某篇已经完成了全部问题。
4. 我们仍可研究跃升宽度的条件规律、样本是否同步翻转、触发映射与输出胜出的区别，以及哪些干预真正改变跃升。LoRA、更多模型名称和更换模态各自不足以构成机制贡献。
5. 跨模型重复支持广泛性；LLaVA/Qwen等多个系统可能共享Transformer语言底座，不能仅凭品牌数宣称所有架构成立。T2I尚在进行。

## 2. 严格筛选：什么才是同类证据

| 要核对的项目 | 本研究要求 | 不等价的情况 |
|---|---|---|
| 横轴 | 同一次训练的optimizer step/checkpoint | poison ratio、adapter合并系数、推理层数、对话轮次、diffusion采样时间步 |
| 纵轴 | 固定评测协议下的触发条件ASR | 广泛失对齐比例、找出trigger的reward、正常分类准确率 |
| 参数 | 正常部署强度下的LoRA轨迹 | 对每个checkpoint额外放大adapter后测出的行为 |
| 投毒 | 毒/净数据安排明确并保持评测固定 | 投毒密度递增、联邦过滤器失效、后续换训练目标等额外变化 |
| 研究目标 | 短窗口的上升形状及其原因 | 只研究何时学会、最终ASR、遗忘速度 |
| 跨模型 | 独立轨迹上的宽度与失败条件 | 只报告多模型最终有效，或把成功seed对齐后的平均当单条轨迹 |

这些项目用来界定差异，不是排除相关文献的借口；部分匹配的工作仍会影响写作和实验。

## 3. 新补查的核心文献链

### R01 Model Organisms for Emergent Misalignment（2025）

[论文](https://arxiv.org/html/2506.11613v1)，§4.1–4.3、Fig.7–10、Appendix G；本地PDF第6页已视觉核验。

**观察：**有害窄域微调中，LoRA方向在短窗口转向；主要单rank-1实验的原始强度行为较平缓，放大adapter后显出陡升。**结论/用途：**作者将方向形成与幅度增长区分，提供可解释的小模型实验体系，扩展检查不同模型与适配设置。**与我们：**它已集中讨论LoRA和跨模型行为转变，但主要测广泛失对齐，不是固定触发条件ASR；不能把缩放后的曲线当原始部署轨迹。也不能仅用我们rank16、他们rank1区分，因其附录包含更高rank。

![R01原始第6页](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/figures/em-organisms-original-page.png)

### R02 Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment（2025）

[论文](https://arxiv.org/html/2508.20015v1)，§2.1、§3、Fig.1–3；本地PDF第6页已视觉核验。正文使用Qwen2.5-14B单层rank1、alpha256，另有Qwen/Llama扩展；高训练alpha对明显转变重要，不等于训练后再放大。

**观察：**有害适配中，输出分布和多个行为维度在短窗口集中改变；梯度峰与行为转变不在同一时刻。**结论/用途：**用分布距离定位变化，并测语义/风格指标能解释多少整体变化。**与我们：**已直接研究训练step上的跃迁和量化，主问题却是广泛失对齐，而非触发条件ASR。它提示我们检查正常/假触发输入：ASR跳变是否只是整体输出风格或偏好的改变。其用左右窗口的事后检测不自动成为past-only在线预警。

![R02原始第6页](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/figures/em-decomposing-original-page.png)

### R03 Convergent Linear Representations of Emergent Misalignment（2025）

[论文](https://arxiv.org/html/2506.11618v2)，§3–4。

**观察：**不同失对齐微调可形成相似内部方向，移除方向能削弱行为，部分adapter偏一般失对齐、部分偏训练语义。**用途/结论：**通过消融和steering分析行为载体。**与我们：**主要是已学会行为的表示分析，不是ASR跃升形成机制；但通用方向、adapter分工、消融这些宽泛贡献已有，必须进一步解释跃升前后有什么因果变化。

### R04 Token-Level Generalization in LoRA Adapter Backdoors: Attack Characterization and Behavioral Detection（2026预印本）

[论文](https://arxiv.org/html/2605.30189v1)，§5、§6.5、§8–12；PDF/文本已保存。

**观察：**Qwen/Llama的LoRA毒adapter在一定毒量区间由弱攻击变为强攻击，泛化锚点和检测信号随模型改变。**用途/结论：**行为探针和权重统计检测adapter，activation patching分析完成模型的通路。**与我们：**主要横轴是不同训练运行的poison count，§6.5的gradual emergence也在这个轴上，不是checkpoint轨迹；没有在本轮正文中发现checkpoint跃升研究。它占据了LoRA+数据投毒+跨家族+rank+因果定位的组合，不能据此直接占据我们的时间形状问题。

### R05 Few-shot Backdoor Attacks via Neural Tangent Kernels（ICLR 2023）

[论文](https://arxiv.org/pdf/2210.05929)，Appendix C.4、Fig.13；PDF/文本已保存。

搜索命中“single epoch后ASR剧增”，但其横轴是**取干净网络哪个训练时刻的权重来构造经验NTK**，不是同一受害者中毒训练的ASR轨迹。用途是更有效地构造少量毒样本；结论涉及kernel对攻击的影响。保留为方法近邻，排除作为本题直接跃升观测。这个例子说明单搜jump关键词会误收。

### R06 Phase Transitions in Backdoor Learning: Minimum Data Poisoning Thresholds for LLM Backdoors（课程研究报告）

[原始报告](https://boazbk.github.io/mltheoryseminar/student_projects/final_papers_and_posters/papers/Phase_Transitions_in_Backdoor_Learning__Minimum_Data_Poisoning_Thresholds_for_LLM_Backdoors_(2)_-_Kaden_Zheng.pdf)，§3–4；[课程来源](https://boazbk.github.io/mltheoryseminar/student_projects.html)。不按正式同行评审论文处理。

QLoRA适配175个Llama3.1-8B模型，扫投毒比例与种子，报告激活的剂量阈值及近阈值随机性。其transition横轴是poison ratio，BAR是触发/非触发行为差，不是同一次训练checkpoint上ASR骤升。意义是风险呈非线性且种子重要，不是我们的训练跃升结案。

### R07 Poisoning the Genome: Targeted Backdoor Attacks on DNA Foundation Models（2026）

[原始全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13308446/)，§2.1–2.2、Fig.2D/3D/5、Methods；[arXiv](https://arxiv.org/abs/2603.27465)，[作者仓库](https://github.com/Georgakopoulos-Soares-lab/Genomic_data_poisoning)。

**观察：**Evo2/GENERator从头预训练的保存checkpoint呈S形后门激活，另有Evo2-7B rank16 LoRA投毒成功。**结论/用途：**提示基因组训练数据风险，并分析token自然频率与攻击可学性的关系。**关键限制：**预训练逐步加密毒样本，时间与攻击密度一起改变；LoRA图主要扫poison fraction。不能拼成固定投毒下跨架构LoRA骤升。它是跨真实架构的动态近邻，但未把optimizer-step跃升宽度独立量化为共同规律。

### R08 Triggers Hijack Language Circuits / Language Triggers Hijack Language Circuits（2026）

[原始v1](https://arxiv.org/html/2602.10382v1)，[当前元数据](https://arxiv.org/abs/2602.10382)。版本标题有差异，视为一篇。

**观察：**预训练语言切换后门的触发处理与正常语言组件重叠，activation patching定位了相关层/头。**结论/用途：**建议研究既有功能组件的劫持及其防御。**与我们：**这里early formation是一次前向传播的早层，不是训练时间早期；研究完成模型，不直接解释checkpoint骤升。它限制“后门建立独立电路”的默认叙事。

### R09 Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs（2025）

[原文](https://arxiv.org/html/2502.17424v1)，§4.2、Appendix C.6。

窄域有害数据引发广泛失对齐，另构造仅有trigger才失对齐的版本。其后门实验不能与R01/R02无trigger的转变实验拼成“同一条件后门的LoRA时间骤升已全部做完”。但这建立了有害适配、广泛行为与条件后门之间的重要联系，必须引用其对照结果。

## 4. 旧89条清单按新问题重新排位

| 原条目 | 当前关联级别 | 已有证据/讨论结果 | 不应再承担的定位 |
|---|---|---|---|
| A01 FixMatch | 最直接形状先例 | 论文明确讨论ASR沿训练快速上升，依赖伪标签反馈 | 不因发生晚而优先；不是LoRA跨架构结论 |
| A02 Universal | 直接形状及反例 | baseline训练图晚学；作者Universal攻击称逐渐学会，并讨论早停取舍 | 不能把正文poison count的sudden increase误当时间轴；也不能说所有攻击提升必然骤升 |
| A03 Inductive Backdoors | LLM直接跃迁先例 | GPT4.1部分种子held-out trigger测试快速提升，train metric平滑 | API更新方式未知；没有统一跨LoRA架构ASR机制 |
| N01 Moderate-fitting | LoRA动态重要近邻 | PET容量约束、正常任务/后门先后；Fig.5有波动与后期增长 | 学习先后不等于集中讨论跃升宽度；全局瓶颈不等于逐层rank |
| N02 Lifespan | LoRA/ASR动态近邻 | 联邦ViT收敛速度、攻击窗口与持久性；正文15到50轮过渡 | 不是我们的多生成式架构短窗口定律 |
| A04/A05/A06 | 训练条件近邻 | 泄漏累积、伪标签阶段、绝对毒量/曝光与顺序 | 不能由后门变强推出陡升 |
| B类休眠/复活/持久性 | 降为状态变化背景 | 改训练、遗忘、重投毒或改trigger可以激活/恢复 | 操作前后ASR低→高不等于同一普通训练的短窗口形状 |
| D01/D06/D07 | 快学反例及动态基准 | 毒训练样本或毒测试指标可很早快速变化 | 不以early为理由排除；检查是否短窗口及统计口径 |
| G13/E03 | 指标解释必须对照 | 平滑连续进展可经离散判定变成台阶 | 不因此一刀否定真实连续加速 |
| G05/G21 | 优化/数值解释 | optimizer状态、范数/尺度、精度可能影响行为变化 | 不凭相似性证明适用于我们 |
| G07/G09/E01/E02 | 突变机制方法近邻 | 内部进展、效率竞争、多环节门控可产生陡升 | train-test分离不再是我们主问题的必要条件 |
| T03/T05/N10 | 生成模型攻击动态 | 图像复现、触发优化改变形成速度；T2I LoRA已有 | 高最终ASR、首次命中或少epoch不自动证明跃升宽度 |
| E05 Self-Awareness | 很像但事件不同 | LoRA+RL中找出trigger的reward短窗口提升，用于检测/反学习 | 升的是trigger恢复能力，不是后门首次形成ASR |
| N04等适配风险/N07域迁移/N08基准/N09 CBA | 配置、攻击和应用背景 | LoRA训练攻击、VLM域迁移、基准、adapter控制 | 不能笼统统称完整跃升机制研究 |
| 其余良性grokking理论/综述 | 候选解释背景 | 有多个有条件的突变解释 | 不再用“晚泛化”替代我们研究的问题 |

原始文献链接、其他逐篇结论见[旧比较](COMPARISON.md)和[旧索引](PAPER_INDEX.json)；原79/89条数量保持为历史清单，新增R01–R09独立记录，避免误称所有条目独立验证同现象。

## 5. 前后引用链与已有集中讨论

已经核验的原文关系：R09 → R01/R03（后两篇明确承接）；R01 → R02（R02 §2.1直接采用其实验）；R03 → R02（R02引用作为机制近邻）；N01 → N02（N02直接讨论前者收敛快、自己的过渡较长）。这些箭头是引用/承接，不是全部研究同一种ASR。

所以，“跨模型LoRA训练中的行为突变有没有被集中讨论”答案是**有**，最重要的是R01/R02这条链；“固定毒/净数据下的触发ASR骤升是否已有统一形成解释”本轮仍未核验到结案证据。

## 6. 别人讨论后形成了什么结论

| 已有讨论路线 | 文献支持的结果 | 对我们仍开放的问题 |
|---|---|---|
| 方向先形成，幅度后增长 | R01 LoRA方向转向与缩放行为相关 | 我们原强度ASR跳时，是更新方向改变还是已有方向增益跨界？ |
| 输出多方面共同转变 | R02分布距离及语义/风格分解 | 我们的ASR是trigger特异变化，还是一般输出偏好变化的一部分？ |
| 后门借用正常功能 | R08完成模型组件重叠；R03可消融行为载体 | 什么训练变化使已有功能被trigger接通？不把终态定位当形成机制 |
| 容量/毒量影响可学性 | N01/N02/R04中的有条件结果 | 它们改变等待时刻、跃升宽度、幅度，还是正常任务一起改变？ |
| 指标阈值把渐变放大 | G13/E03在相应任务的结果 | 连续margin/概率是否同样陡升？样本crossing时刻是否聚集？ |
| 早停/监控风险 | A02/A04已有应用，R02称梯度峰为预警线索 | 跃升很快时，稀疏checkpoint是否会漏掉风险？在线干预是否真的有效？ |

没有一条能仅凭我们的ASR折线图宣布为统一机制，也没有共同结论说所有模型的ASR提升必然是跃升。

## 7. 剩余空间：围绕跃升动作，不绕回早晚

### 7.1 建立跨模型的定量现象

先定义显著上升事件、幅度和窗口，再报告宽度。可用相对该次提升幅度的10%到90%跨度；非单调曲线需要预先指定稳健规则与持续性要求，不能事后挑有利片段。没有明确事件的轨迹保留为失败/渐变，不硬算。

“只要提升就都是短窗口”是当前观察的概括，正式研究应在预先确定的设置范围检验，并主动找渐变反例。A02/R01已经提醒渐变可能存在。报告步数、处理tokens/毒曝光以及相对训练预算，不能只按t50对齐后看平均形状。

### 7.2 解释为什么很多输入一起翻转

总体ASR台阶可能来自共享行为开关，也可能来自不同样本的margin恰好集中越界。逐样本记录target-vs-correct margin、目标概率、ASR crossing时刻与回答内容；比较trigger、无trigger、假trigger和配对干净训练。

如果概率平滑但crossing聚集，贡献仍是可预测的条件输出切换，不要求证明全模型内部突变。若连续输出也集中加速，再追踪其更新/内部原因。不能将“非指标伪影”直接升级为严格相变。

### 7.3 区分更新方向、更新强度和优化器作用

借鉴R01，但研究本来部署强度。跃升前/后分别改变adapter增益和方向：单纯增益是否能让前checkpoint提前实现后状态？不同时间的delta替换是否移动跃升？每次同时测正常质量。再以保存模型/优化器状态的配对分叉测动量与LR因素。

R01/R03已有方向、缩放、消融思路，创新要在能解释自然条件ASR跃升、形成留出预测；不能仅复用PCA+grad norm+patching就称新机制。

### 7.4 实际用途：短窗口造成监测盲区

如果跃升在几个测量间隔内完成，粗粒度评测会把模型从低ASR直接看到高ASR。量化不同评测频率的漏检、延迟和额外成本；若能用过去信号触发加密评测/干预，并降低之后风险而保住正常质量，才是应用贡献。

这条路线不依赖跃升早晚。左右checkpoint比较和知道最终跃升后再选指标只能解释，不能自动算在线报警。

## 8. 修改后的论文定位

建议主张：**我们系统研究普通投毒适配中，触发条件攻击行为的短窗口跃升，量化其跨模型重复性与条件边界，并区分连续输出加速、群体越界和更新动力学的解释。**

避免：“首次发现LoRA行为相变”“所有ASR增长必然突变”“跨模型同形证明同一机制”“首次找到后门方向/做消融”。跨VLM/LLM/T2I的机制结论需证据完整，T2I未完成不计入确认结果。

## 9. 检索与复查范围

本轮进行了34条补充检索式，涵盖LoRA/QLoRA/PEFT、poison/backdoor、ASR/attack success、abrupt/jump/sharp/rapid/sudden、phase transition、checkpoint/epoch/training dynamics，以及失对齐和基因组近邻；没有把grokking或late作为必选词。另追查R01→R02、R09→R01/R03及N01→N02原文链。查询见[RESEARCH_QUERIES_ASR_JUMP.json](RESEARCH_QUERIES_ASR_JUMP.json)。

本轮新保存4份PDF及文本，R01/R02关键原页视觉核验；R03/R06/R07/R08/R09核验到原始全文网页或PDF链接，未全部本地缓存。查询命中的二手/自动生成总结仅作线索，不作为最终科学结论。课程报告和预印本分级，不把作者声明当独立复现。

这不是有预先冻结数据库和双人筛选的PRISMA系统综述，也不能保证全球零遗漏。本报告由AI辅助检索与分析；建议实验均未在本轮执行。
