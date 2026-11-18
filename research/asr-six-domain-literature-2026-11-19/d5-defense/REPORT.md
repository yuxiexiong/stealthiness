# D5 干预与防御：A 按 B 意见修订的全文证据报告

日期：2026-11-19。状态：**首轮 B 已核原18篇/36组；A 已按意见修订并补读12篇，新增/改动尚待原 B 回核和 C 独立查漏；不得据此签署六域完成。** 本报告是已有证据与重合版图，不是研究方向推荐、实验授权或创新性保证。

共核读 30 篇一手论文的正文全部及相关附录；另对 BAD-BOOM、Cordyceps 作跨域防御段定点复核，综述与 GARD 为部分阅读，ANP/T&R/CBD 为来源及范围定位；具体阅读页、未读示例页及版本见 `READING_NOTES.md`、`PAPERS.json`；逐项处置见 `REVISIONS.md`。采用公开 arXiv/ACL 原文；页码均为 PDF 物理页。未重新运行作者代码或实验。BI-BAU、Suppression、TopKLoRA、HARVEY、DETER复用其他域下载件，但本域独立阅读；新增SP24/OPD亦复用D1公开下载件后独立阅读；其余保存公开原文。没有把下载、摘要、其他域结论当成本域全文阅读。

## 1. 最重要的已有认识

**通过训练过程分配、削弱或绕开后门关联，并尽量保留正常任务，是已经成熟的研究目标；不能把“干预后 ASR 降低”单独当作新机制。** ABL在受污染视觉训练中用早期损失差隔离和反学习；DPoE在文本分类中让浅模型吸收捷径，主模型学习剩余任务；2026年的TR加显式知识解耦。三者均实际改变训练，但各自假设和失败条件不同，不能合成一个已被证明的普遍自然形成定律。[ABL §§3–4/AppA.3](https://arxiv.org/abs/2110.11571)、[DPoE §§3–4/AppA–D](https://aclanthology.org/2024.naacl-long.27/)、[TR §§4–5/AppB/E](https://arxiv.org/abs/2608.00732)。

**防御者自己植入关联以处理未知后门，已有多个不同方案。** P2P依赖推理时的安全触发及映射；MB、Dummy、Locphylax则在训练中植入已知关联后中和/恢复，试图间接削弱未知关联。它们不具有相同“撤除后仍有效”的证据，也没有共同证明所有后门共享一个独立回路。最直接的反证包括Dummy的跨目标任务失效、MB增加防御触发反而失效、共享潜变量特征在inductive backdoor上的steering无可靠效果（该条件未测CAFT）。[P2P §3/AppA–C](https://aclanthology.org/2026.findings-acl.600/)、[MB §5/AppB/G](https://aclanthology.org/2026.acl-long.1113/)、[Dummy §3/AppC–E](https://arxiv.org/abs/2606.11648)、[Locphylax §§3–4](https://aclanthology.org/2026.acl-long.920/)、[Shared Latent AppF.6](https://arxiv.org/abs/2606.07963)。

**静态低 ASR、正常输入平均能力近似保留、后门不可恢复，是三个不同命题。** Suppression论文直接显示某些投影防御静态压低后仍可较快再学习；加入良性参考的方案改善早期恢复，但其核心表中继续训练后仍恢复。BI-BAU的“完整消除”有特定函数等价定义与理论条件，实测也有较高残余。不能将任一标题中的erase/remove翻译成不可恢复删除。[Suppression §§4–7/Tables3/9](https://arxiv.org/abs/2611.12807)、[BI-BAU §4.1/§5/Table10](https://arxiv.org/abs/2606.14078)。

## 2. 操作与证据必须分开

| 操作类别 | 本轮实际核读代表 | 已验证的对象 | 不能自动升级为 |
|---|---|---|---|
| 训练前清洁/筛选数据 | SelfHeal、HARVEY | 处理后数据训练出的模型，端点ASR与任务指标 | 原受感染模型内部关联被删除 |
| 改变受污染训练 | ABL、DPoE、TR | 指定架构/任务的防形成或削弱 | 自然混合训练的统一阶段定律 |
| 添加防御关联再中和 | MB、Dummy、Locphylax | 未知攻击触发下的行为缓解 | 所有目标共用同一回路、不可重建 |
| 持续推理保护 | P2P安全key、Fragmented feature clamp | 保护运行时的输出 | 保护撤除后权重已安全 |
| 权重/参数修复 | DiSec、BI-BAU、投影系列 | 指定数据与后续训练下行为 | 任意自适应恢复均失败 |
| 定位后定点零化 | TopKLoRA | 人工稀疏adapter、已知触发的条件必要/充分电路 | 普通稠密LoRA的唯一自然回路 |
| 防非预期泛化/增强再学习阻力 | 原CAFT、DETER | 非后门邻域任务 | 已证明触发后门形成机制 |

这里“运行时保护”并不等于低价值：它可切实减少部署风险。分类仅用来避免把有效保护误称为持久删除。TopKLoRA的零化有精确因果操作，但现有实验不检验零化后重训练是否重建；CAFT在训练后撤去消融则是不同证据。

## 3. 逐篇可审计证据

以下“测量”均指作者报告的实验；“解释”指作者提出的机制解释；“本报告限制”是由实验设计可识别范围作出的判断。每条详页、指标分母、反例见阅读记录。

### 3.1 训练中形成预防与数据清洁

**D5-01 ABL（2021）。** 正文1–10、附录14–20。视觉网络若干攻击的毒例平均loss更早下降，为local gradient ascent隔离再反学习提供依据。AppA.3实际执行20轮隔离、60轮普通训练、20轮global gradient ascent；不能从主文概括的“两阶段”推断自然训练有同样两阶段。Table1 CIFAR10平均ASR7.69%，准确率84.76%，纯净训练89.12%；高污染条件下Blend/Dynamic仍高ASR。它已覆盖“利用正常/后门学习差异干预形成”，未覆盖通用LLM自然轨迹或不可恢复擦除。[原文](https://arxiv.org/abs/2110.11571)。

**D5-02 DPoE（NAACL2024）。** 正文1–9、AppA–D 13–14。浅trigger-only模型和主模型用PoE联合训练，配合去噪，推理只用主模型。BERT与Llama2-7B均已测试，任务仍是文本分类。Table2 Llama2 SST2语法触发防后ASR19.54%、ACC93.67%，非全清除。AppA梯度解释以浅模型已学会触发、对clean近似均匀为条件；pseudo-dev选参也依赖所筛集合的纯度与代表性。不能由公式推出训练必然分工。生成任务明确留待后续。[原文](https://aclanthology.org/2024.naacl-long.27/)。

**D5-03 TR（2026）。** 正文1–7、App10–19。浅并行分支优先吸收捷径，加特征/预测/原型等解耦后训练末移除；视觉分类及ViT微调。无解耦时主干后期也学后门，说明早期分流可只是暂时；不同分支容量/配置可严重失败。Table1 CIFAR10平均ASR0.57%、BA89.40%，纯净91.19%。AppB的梯度大小论证带假设，不是普遍证明。AppE.6/Table14（p17已看原图）的**训练动态个案**中主干高置信度输出非目标错误类，表明分离目标并不等于正确恢复；该表不能当最终测试错误率。自适应检验是surrogate攻击的跨初始化迁移，且未测再学习。[原文](https://arxiv.org/abs/2608.00732)。

**D5-04 HARVEY（arXiv2026，PDF版权脚注2025，会议年份未另核）。** 正文1–8、App10–15。先训练偏向后门的参考模型，以RCE多轮分毒数据，再由早期参考模型回收误分的正常目标类，最后训练最终模型。这里oracle是**学得参考模型**，不是已知毒标签。端点表现强，但并非已有受感染模型的擦除；AppD.5 clean-only控制仍有准确率损失，AppD.1显著超参敏感。Table10 VGG/WaNet ASR2.15%、Table14自适应重复样本/WaNet2.57%，所以主文“最差低于2%”只限主比较范围。无LLM生成实验。[原文](https://arxiv.org/abs/2607.05748)。

**D5-05 SelfHeal（AACL2026）。** 正文1–9、App12–21。信任base模型，通过任务提示标记/清洁指令、重建回答；先处理数据，再训练模型。Llama3-8B/Mistral7B、多任务/攻击、主比较5seed；优点是不用丢掉整条记录。但依赖可信base；前两阶段提示主要任务无关，第三阶段回答重建有任务特定要求。Table14（p14原图）对原本clean的Mistral情感数据处理后ACC96.52→82.09%，说明“保留记录”不等于“保留正常信息”。Table12语法触发Mistral防后ASR52.86%；主表也有较高残余。不是对已经感染权重的清除。[原文](https://aclanthology.org/2026.findings-aacl.248/)。

### 3.2 利用已知防御关联处理未知攻击关联

**D5-06 P2P（ACL Findings2026，采用2026-11-10 revision2）。** 正文1–9、App13–16。扩展标签/植入良性触发，推理时加安全prompt并映射输出；威胁模型允许控制输入流。它是持续保护方案，不能拿在线ASR当安全key撤除后的清除。CR条件的CBA仍19.73%；AppA p13首先证明函数类可表达性的sup等式，非优化实际达到最优；其后另假设α→1、ε→0。进入扩展标签集本身不保证π映射后的标签不是攻击目标，因此不构成实际训练安全收敛保证。注意力竞争解释不等于已隔离验证的唯一中介。[原文](https://aclanthology.org/2026.findings-acl.600/)。

**D5-07 MB-Defense（ACL2026）。** 正文1–9、App13–19。训练自有触发/随机词响应，再用128条干净样本加偏好式中和；不需攻击trigger/target，也不需推理key。四模型与218 WizardLM提示支持若干生成后门缓解，但CACC不代表全知识能力。AppB相对良性扰动的表征重叠优势并非每项为正；PCA相近不能证明单一共享回路。Table4（p9原图）Qwen3-1.7B/Toxic目标，防御触发由四个增到六个时InSent ASR由0升至0.972、AACC由.729降至.303；这是防御触发数量，不是多个攻击触发。并发多个攻击触发未测，亦未测再激活。[原文](https://aclanthology.org/2026.acl-long.1113/)。

**D5-08 Dummy Backdoor（arXiv2026v2）。** 正文1–10、App11–18。知道攻击目标任务、未知触发，通过同目标dummy的植入/中和处理未知关联；有初始训练和事后修复情境。激活差分方向相似和输出相似控制支持部分共享，但AppC输出控制仅4例与16例，主jailbreak面板99题；作者也不视为决定性因果证明。同任务多未知触发已有测试，区别于MB的未测范围。跨目标任务Table10：用负情绪dummy修jailbreak，Mistral ASR82.83→83.84%，失效。Table11数学修复后Mistralclean53.5→45.5%。jailbreak主要用拒绝关键词判据、主实验单次运行；不能外推普遍跨目标擦除。[原文](https://arxiv.org/abs/2606.11648)。

**D5-09 Locphylax（ACL2026）。** 正文1–9、App13–19。对已感染模型加已知触发/中性输出，显式聚类后以正确回答恢复；涵盖SFT、编辑、RLHF情境。聚合部分是目标函数主动施加的，不是自然后门原本共用表示的证明。局部attention替换支持条件性作用；AppC语法AdvBench ASR仍23.07%。正文描述先拒绝后有害内容，而附录关键词判据可能漏掉此类，评价存在边界。无持久恢复审计。[原文](https://aclanthology.org/2026.acl-long.920/)。

### 3.3 定位、修复与恢复边界

**D5-10 DiSec（ACL Findings2026）。** 正文1–10、App11–20。用20k干净无标签辅助文本的梯度残差、SVD/VAE识别异常权重行并以均值替换；不用真实触发/目标，代理触发辅助定位层。BERT/RoBERTa分类，单seed；后续全参数任务微调仍低ASR是有价值的耐受测试，但非主动重新投毒。NeuBA/RoBERTa SST2正常ACC94.67→88.96%、AASR12.20%（跨触发平均；MASR14.69%）；adaptive BadPre仍有19.45% AASR。clean模型也会选出所谓脆弱层，因此异常行不能自动视为后门专属。[原文](https://aclanthology.org/2026.findings-acl.815/)。

**D5-11 BI-BAU（arXiv2026）。** 正文1–14、App16–23。以持续学习/灾难遗忘视角重审adversarial unlearning，引入后门识别约束与双层优化。定义3“complete”是指定clean/trigger输入上与clean模型函数一致；NTK、任务正交等假设及部分证明步骤的秩条件未充分展开，本报告不独立背书。实验是视觉分类/CLIP，LLM在§5.10属于未来方向。R-ASR为需要权重再微调的retuning，Q-ASR才是query-based reactivation。Table2 CIFAR10 U版平均R-ASR12.48%；Table10 TinyImageNet SIG R-ASR96.72%，不能套用query-only权限说明这两个数字。AppD.2还分别定义两类操作与数据，见阅读记录。这篇是对“短期输出变化≠结构清除”的重要讨论，不是已经普适解决删除。[原文](https://arxiv.org/abs/2606.14078)。

**D5-12 Suppression Is Not Erasure（arXiv2026v2，11月15日）。** 正文1–18、App22–29。明确比较静态ASR与后续重新学习。核心Qwen2.5-0.5B/rank8，多数路径是merge原adapter后新LoRA，不能统称旧A矩阵保存一切。DSE及良性参考DSE-BA减慢早期恢复，但Table3后续仍达100%；较稳妥的Table9 50提示×3seed，DSE-BA第2epoch43.3±2.5%，不应只报12提示小面板的25%。跨架构静态也有失败，rank-8 adapter上k=1单方向投影的句法残余54.2%，k=6的DSE/DSE-BA在该小面板静态为0；这里k不是LoRA rank。风格保持43/50 vs50/50；选择题近似不变不等于全能力无损。既有投影方法的压低行为必须与耐恢复能力分别评。[原文](https://arxiv.org/abs/2611.12807)。

**D5-13 TopKLoRA（arXiv2026）。** 正文1–10、相关App13–18/20–29。在人工设计的稀疏adapter、已知触发配对输入、三种base中搜索电路并做remove/keep-only，提供较强的局部必要/充分干预。必要性基于固定验证面板，ASR判据只看前40token目标短语。搜索不同顺序会给不同电路；人工planted集合也不简单可加。多层电路保正常任务较好，但单层小模型严重掉能力；clean twin部分有小而显著差异。postTopK固定门控下零化不是已经验证普通稠密LoRA的唯一机制，也未测再训练后重建。[原文](https://arxiv.org/abs/2610.32866)。

**D5-14 Fragmented Defenses（arXiv2026）。** 正文1–9、App12–22。3模型/3分类式任务的clean/poison模型×clean/trigger输入2×2，提出SAE特征组成解释；差分恒等式与线性投影并非独立机制的严格分解。运行时将异常特征夹到clean/base参考，保留重构残差，支持局部缓解；Table5仍有残余ASR，且§5.1按低ASR/小CACC损害选结果存在评测选参依赖。没有控制其他因素独立操纵组成来证明它是既有防御失败的唯一原因。首稿未定位到具体表格单元的“top60计数不一致”疑点已撤回，不以未定位疑点作证据。[原文](https://arxiv.org/abs/2608.30403)。

**D5-15 Critical-CoT（ACL2026）。** 正文1–9、AppA.1–10 12–15。合成审查推理数据SFT+DPO，不是只有系统提示。分别评ICL和FT后门，检测/触发识别率与ASR、clean accuracy分列。Table6仅SFT误报13.4%、clean76.5，完整法1.0%/91.7（base93.9），所以降低过敏有实际证据，但并非完全无代价。FT感染后的相近clean/trigger准确率不代表恢复未感染base水平。正文p5训练示例出现@_@与“In arcane parlance”，p7又在unseen触发描述中列出相同字串；仅据论文无法澄清其训练/测试划分，不能据此背书所有列举触发均未见，也不直接断言实际数据泄漏。语义/上下文触发和持久重学习未解决；行为审查成功不是内部关联被删的证明。[原文](https://aclanthology.org/2026.acl-long.495/)。

### 3.4 邻域证据：能迁移的问题，不能迁移的结论

**D5-16 Shared Latent Structures（arXiv2026）。** 正文1–9、App12–17。借助source后门与clean模型差分选共享SAE特征，再做双向steering、目标后门迁移和CAFT。source发现需要已知trigger/clean参考，“目标未知”不等于整个流程无先验。双向干预确实比单纯相关更强，但password方向反转、编辑攻击较弱，AppF.6 p15 Qwen3-8B inductive条件是共享特征的正/负steering，既未强激活也未可靠缓解；不是CAFT失败试验。p14 Gemma CAFT学习率2e-5、默认SFT2e-4，同时改变学习率，因果比较需限定。已覆盖“共享表示+训练干预”，未覆盖所有后门或自然时间上的集中显现。[原文](https://arxiv.org/abs/2606.07963)。

**D5-17 原始CAFT（arXiv2025v2）。** 正文1–10、相关App18–35/40–43。训练前向/反向移除概念，推理撤去，用于控制OOD泛化；§7明确不是删除概念。Qwen32B和Mistral24B不安全代码微调中的一般misalignment显著下降，但“保留任务”指漏洞代码生成，不能当良性能力指标。MMLU/GSM8K是额外窄检查。随机/非解释PC对照和早停Pareto已有；并非解释性方向总最优。p21 AppA.5 Table1为Qwen、Table2为Mistral；后者top-5 PCs的misalignment/任务指标.34/90.5与解释方向1.16/92.2是不同折衷。方向发现还用了普通微调后状态，所以不是天然在线预警。它与项目的接缝是干预训练改变非预期行为，不是触发后门自然形成的同一证据。[原文](https://arxiv.org/abs/2507.16795)。

**D5-18 DETER（arXiv2026）。** 正文1–9、App13–25。从头预训练中过滤医学数据并穿插负学习，用分离优化器状态减少正常能力损伤；113M–1.03B模型。共同恢复门槛是同规模NO FILTERING模型的PubMed loss；倍率分母是FILTERING模型达到这一门槛所需token。最高20.36倍、绝对2.09M、约预训练token的0.010%，所有方法都在预算内达到门槛；不能把门槛本身写成FILTERING loss。测的是loss/选项NLL，不能说医学实际能力永久删去。时序与优化器状态对比已实际做过，但起始时间效果依规模/utility目标而变，不能归纳早干预一律更强。它是耐篡改知识抑制邻域，不是触发后门防御的直接复现。[原文](https://arxiv.org/abs/2611.15479)。


### 3.5 B 后补齐的训练与修复机制链

**D5-19 Honeypot（2023v1）。** 正文1–9、附录13–20。低层支路以易例目标学习，用其loss给主支路加权，推理丢弃支路；不需额外clean数据。它补上DPoE/TR之前的文本分流支系。关键边界是“低层probe先学毒”不等于整个PLM先学毒：AppC实际指出完整PLM可先学正常语义，ABL式低loss隔离在其重实现中很差。Table1 RoBERTa/SST2风格、句法仍20.87±3.90、22.90±3.85% ASR；短warmup和阈值也可失败。三seed/目标，分类任务，未测生成或持久删除。[原文](https://arxiv.org/abs/2310.18633)。

**D5-20 DBD（ICLR2022）。** 正文1–9、附录15–25。先无标签自监督，再冻结表征训head并以低loss取可信标签，最后半监督联合训练、逐轮重筛；只有潜在污染集、不需额外clean。这里低loss被用作clean信号，恰与ABL不同训练状态下的低loss毒信号相反。Table2标签一致攻击BA94.90→89.67，ASR.01%；可信集合仍含少量毒，不能称完美检测。自适应实验是代理聚类目标的跨初始化迁移，并非所有适应攻击。未测LLM或防后再学习。[原文](https://arxiv.org/abs/2202.03423)。

**D5-21 MuScleLoRA（ACL2024v3）。** 正文1–9、附录13–19。低秩多尺度参数化与clean辅助梯度引导共同训练，frequency指输入—输出映射而非词频。AppB4 LLM用128条clean辅助例且µmax=0，更新方向完全沿clean辅助梯度；不能将效果全归于频率偏置。Llama7B风格触发仍39.03% ASR、94.73% ACC（正常96.43）；其实验是加分类头，非开放指令生成。消融中多尺度本身可仅延缓后门学习，某些单项ASR反而劣于去多尺度。需可信辅助数据与模型相关选参，未证永久防形成。[原文](https://arxiv.org/abs/2402.12026)。

**D5-22 I-BAU（ICLR2022；v4）。** 正文1–9、附录13–27。已有感染模型+clean标注数据，对共享扰动作minmax和implicit hypergradient修复；主设5000条clean从测试集划出，剩余评价。“100条也有效”只限所示cell，且该cell ACC84.92→76.90。Table3多目标CIFAR ASR98.37→12.96而ACC85.96→77.44。理论依赖强凸/光滑、内层收敛、可逆Hessian等，非任意非凸安全收敛；部分推导符号需进一步核对，不作为强保证。无再学习测试，后继BI-BAU的R/Q问题不能反向算本篇已验证。[原文](https://arxiv.org/abs/2110.03735)。

**D5-23 Fine-mixing（2022v1）。** 正文1–9、附录12–18。需要可信预训练checkpoint及同任务小clean集，将感染/预训练参数按二值mask元素替换，配合embedding恢复后clean微调；不是普通凸平均。SST2平均ASR100→16.64、ACC91.70→89.53；QQP from-scratch句触发64例仍88.71% ASR，512例仍37.75%。某些任务容忍约10pp精度损失，几何图像不能升级成普遍良性盆地保证。未测再学习。[原文](https://arxiv.org/abs/2210.09545)。

**D5-24 BEEAR（2024v1）。** 正文1–9、相关附录13–17。对连续embedding施加共享对抗扰动、借safe/harmful/performance锚点修复，部署撤扰动。无需真实trigger不等于不知道目标行为：代码sleeper情境使用同源数据留出的1000条坏代码；该结果仅17题，8/17→0/17。正常能力主要MTBench，而且用其稳定作为停止依据，部分修复后仍低于干净base。主jailbreak有低残余，§8明确目标不匹配可能失败；无适应恢复审计。PCA方向一致不是代理扰动等同真实trigger的证明。[原文](https://arxiv.org/abs/2406.17092)。

**D5-25 NAD（ICLR2021v2）。** 正文1–10、附录13–19。用少量clean先微调teacher，再从原感染student作CE+中间attention图蒸馏；这里是视觉通道聚合图，非Transformer attention。5%clean时Table1平均ASR99.04→7.22、ACC83.50→80.83，Trojan仍19.63；1%clean均值ASR35.93。AppJ明确训练过久会拟合teacher残余后门，需要约5epoch早停；这是修复过程中回升，非新一轮投毒。未证全权重删除。[原文](https://arxiv.org/abs/2101.05930)。

### 3.6 最新防御边界及理论前身

**D5-26 QES（2026-11-04v2）。** 正文1–9、附录17–29。训练时以注意力路由正则把后门导向预定LoRA专家，部署保留权重并逐层逐token屏蔽该专家，是持续门控隔离，未测撤门或再学习。部分CTBA仍36.5/37.5% ASR。AppB3明确把GSM8K test的1319题纳入训练；主ΔU不能叫未见数学泛化，独立benchmark只在有限模型/攻击cell补测。容量与普通LoRA也不同。预设多专家能主动隔离，不证明普通LoRA自然就有该模块。[原文](https://arxiv.org/abs/2610.00663v2)。

**D5-27 Phantom Transfer（2026-10-07v3）。** 正文1–10、相关附录16/27–28/33–34/39–46/59–62。以不同模型生成并过滤显式目标的训练文本，仍有跨模型目标偏好传递；与常规trigger后门邻接，但基础情境是诱导favorite/least-favorite回答，不是任意恶意动作。“11防御”包含random control和数种同一外部sentence embedding过滤；结果是无方法在所有设置稳定成功，并非所有cell失败。Table5某oracle分类器cell为0；训练预算在正文与附录有2k/12k描述差异。Table31存在大幅MMLU-Pro下降，不能称全能力保留。密码条件确测语义触发、亦可被相关语义激活，但只覆盖部分过滤/改写防御，不能移植全部11项结论。它否定“已过滤显式目标就安全”的宽泛说法，不证明所有分布都无法检测。[原文](https://arxiv.org/abs/2602.04899v3)。

**D5-28 OPD / Lazy Defense（2026-10-06v1）。** 正文1–11、相关附录14–20/23–27。安全蒸馏中感染teacher的目标可传给student；Lazy截断奖励比值以延缓。Table4同规模3B、5%条件下epoch4/8/12 ASR2.3/45.0/69.5%，长期接近未防御70.3%；“clean ASR”是无trigger的有害prompt响应，不是正常任务utility。τ数值等训练细节未在PDF完整给出。其所谓重叠证据实际是teacher-forced固定序列上跳过7token后的10位置top1一致率，不能当student on-policy top-k中介的直接验证。[原文](https://arxiv.org/abs/2610.07654)。

**D5-29 Orthogonality and Linearity（IEEE S&P2024）。** 正文1–14、附录15–18及补充20–21。BI-BAU之前已有continual-learning/正交/线性视角及防御对照。Table1黑色是引文、灰色是新测，不能把整张表当独立重现。NTK、固定特征与正交是假设，实测平均loss梯度夹角非完全相同对象；低正交时理论上界变大不等于损失实际必变大。Table5显示多种修复失败，AppTable12/13也有低正交仍NAD/FP成功的反例。故已覆盖该机制讨论，但未证明正交/线性唯一决定防御成败，更非LLM普适定律。[原文](https://doi.org/10.1109/SP54263.2024.00225)。

**D5-30 Safety-Efficacy Trade Off（2026-01-31v1）。** 正文1–9、附录11–17。KRR紧密毒簇、忽略交叉核块下展示拟合与输入导数可能脱钩，提出训练时loss输入梯度正则。一般性定理需要谨慎：AppA6先忽略基础loss，且A/F为PSD并不推出AF+FA为PSD；不能用其背书全训练流的普遍收缩。Table1同一条件90→450epoch时ASR6.63→70.68%，ACC75.17→79.05%。提供有条件的早期防效/拟合折衷，不构成普适免疫或所有防御不可保效用定理。[原文](https://arxiv.org/abs/2602.00822)。

**跨域定点核验。** BAD-BOOM由D2主读，本域核pp3–4/6–9与AppC15–16/D2p17。它评的是常规无trigger SFT而非专门防御，原AdamW在若干cell也保持很高ASR；新方法持久性更高但不是所有cell≥90%，亦有正常任务损失。景观切片不等于实际SFT全路径。Cordyceps由D3主读，本域核v2§5.3 pp9–11、AppC/F19–20：CROW/SecAlign按50题推理长度90%早停，实际4–9/20–30迭代；不能说完整默认防御训练失败。其隐蔽泄露CA、任务ASVh与常规ASR分母不同，且防御后一般能力没有由poisoned模型的utility表代替验证。[BAD-BOOM](https://proceedings.mlr.press/v306/zhao26bf.html)、[Cordyceps](https://arxiv.org/abs/2605.26595v2)。

## 4. 与本项目已有证据的实质重合与边界

此节依据共享的项目比较卡，不重新解释或扩展本项目实验结果。

| 本项目问题/已经观察的层面 | 文献已覆盖的实质内容 | 仍不能据此认定相同或空白 |
|---|---|---|
| 正常学习与后门学习交互、学习快慢 | ABL测早期loss差；Honeypot/DPoE/TR主动分流；DBD改变low-loss信号含义；MuScleLoRA加频率及clean梯度约束 | 防御损失主动制造的分工不等于未改训练中的自然阶段；观察到快慢本身已非新颖 |
| 历史状态、更新响应影响当前行为 | DETER比较时序/优化器状态；ABL/HARVEY利用训练不同阶段状态 | 邻域任务、状态定义和因果对照不同；不能将本项目有限更新响应直接归为它们的机制 |
| 表示/回路与行为之间的干预联系 | Shared Latent双向steering；TopKLoRA remove/keep-only；DiSec权重替换 | 指定输入/位置的因果贡献已有人做；唯一性、普通稠密训练、跨目标普适性未由这些证据确立 |
| ASR下降后是否仍有潜伏 | Suppression直接评再学习；BI-BAU分R/Q恢复；NAD/OPD存在训练继续时回升；BAD-BOOM检验clean SFT耐受 | 本项目ASR自然下降不是防御清除；静态低ASR不能推断潜在能力消失 |
| 保正常能力 | 多数论文有clean ACC，有些有clean-only、MMLU/MTBench或clean twin | clean均值可掩盖目标类/稀有类/触发任务错误与风格损害；不能合成“无损” |
| 从信号到在线干预 | 有方法用无真实trigger数据构建训练防御；也有需要source后门/未来微调状态的方向 | 未来状态差分不自动成为前置信号，干预有效不自动证明在线可取得方向 |

**可以排除的宽泛“首次”说法**：首次利用学习差异防后门；首次让已知防御后门影响未知后门；首次通过表示消融/电路零化降低后门；首次研究低ASR后可恢复；首次考虑干预时序与正常能力折衷。这是对具体已核证据的重合判断，不是认定本项目所有精细问题已被解决。

**本轮没有证据支持的强说法**：某一个“共享方向/捷径模块”已经统一解释所有自然形成、上升、下降；任何一次成功防御已经不可恢复地删除后门；本项目尚未闭合之处就自动构成论文创新。防御实验通常改变数据/损失/架构/推理条件，必须保留这种干预情境。

## 5. 覆盖限制与交接

- 原18篇的B强制更正已逐条落入正文和笔记；新增12篇尚未冒充独立复核通过。`REPORT_A.md`保留首稿。完整对照见`REVISIONS.md`。
- 综述2502.05224仅§4–6/相关引文；GARD2611.14143已读官方HTML方法范围，仍未全文。ANP只来源/前两页，T&R只来源/前两页，CBD只官方摘要。这些均不计30篇全文，也不据此作原篇性能保证。T&R的重构头/替换头与TR2026明确不同，CBD身份为CVPR2023的deconfounded representation（不是同缩写检测论文）。
- 保留这些外围来源的理由：传统剪枝完整原型本轮选NAD；主动分流/解耦已有Honeypot、DBD、DPoE、TR；事后对抗修复已有I-BAU、BEEAR、BI-BAU。GARD的CLIP表示对齐是不同实现，但不改变本报告关于这些宽泛目标已有、代理扰动有条件有效、无普适删除保证的判断。没有声称覆盖所有变体、多模态/联邦目录或最新全文全集。[ANP](https://arxiv.org/abs/2110.14430)、[T&R](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ea06e6e9e80f1c3d382317fff67041ac-Abstract-Conference.html)、[CBD](https://arxiv.org/abs/2303.06818)、[GARD](https://arxiv.org/html/2611.14143v1)。
- Persona Vectors由D6主读，本域不引用其结果，也不称原CAFT是唯一训练期预防研究。BAD-BOOM、Cordyceps本域仅定点核防御边界，不增加全文计数。Phantom的跨模型/密码条件及过滤限制已交D6。
- 专业学术MCP不可用；公开arXiv/ACL/会议原文可得，核心补读无全文下载阻碍。没有声称完整获取前向引文图；关键词检索、正文参考链及后继回指记录在`SEARCH_LOG.md`。外围保留由范围而非访问失败决定。
- 关键表如首稿所记做过像素核验，其他数字按原PDF表格/正文提取；MuScleLoRA19页已按单页提取校准。未重跑作者代码、未测模型。数学证明限制是审读发现、尚待B核，并非独立形式化验证或对作者错误的最终裁定。

`PAPERS.json`区分全文、部分、摘要/来源定位；`READING_NOTES.md`给具体条件和反例。原B应回核新12篇、九项更正及跨域防御段，再由C独立查漏。本稿不替代B/C，也不授权模型运行或研究方向切换。
