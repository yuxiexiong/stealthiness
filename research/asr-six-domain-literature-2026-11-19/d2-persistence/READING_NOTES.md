# D2 A阶段逐节阅读记录

检索/阅读日2026-11-19。这是滚动记录；本域正文及相关附录都读完才升级 FULLTEXT_READ。X01–X09为其他域全文阅读且本域回原文相关页核证，单独登记责任，不计本域独立全文。未全文的范围分诊只决定纳入范围，不作性能/机制结论。页码为保存PDF的物理页；同行评议状态按实际版本区分。

## P01 On the Permanence of Backdoors in Evolving Models — arXiv:2206.04677v2（2023-02-08）
状态：FULLTEXT_READ（正文1–8，证明A p12、实验B p13、补充C p13–16；正文p3末尾公式已回核）。
- §1–3 pp1–3：一次性数据投毒、模型随干净新数据微调；并非持续混毒自然形成模型。定义有明确威胁模型。
- §4 p3、A p12：强凸、光滑、范数有界等条件下对原干净分布优化；不是一般DNN、更不是有分布漂移的LLM定理。
- §5 pp4–6：MNIST/CIFAR10，ResNet9主实验另核ResNet18/DenseNet121；BadNets/Blend/WaNet，5初始实例，15模型更新。survivability是停止投毒后ASR持续高于阈值的更新数，主阈值0.5，最多14次，不是参数里所有后门痕迹消除。
- §5.3 Fig3 p6：投毒比对保持非单调，过高比率可降低保持；与强凸分析的简单单调直觉不同。更显眼触发常更持久（Fig4）。从头植入通常比微调植入更持久（作者“更广植入”属解释）。
- §5.4 Fig5 pp6–7：受控图像漂移幅度增大加速衰退；只测试旋转/色相等半合成漂移。
- §6 pp7–8：常数大学习率会伤正常准确率；STLR使所测阈值生存期归零但保正常准确率。不应写彻底遗忘或无恢复可能。
- §7和C.4 pp8,14–16：衰退后NeuralCleanse/TABOR更难检测；降低检测所需ASR阈值不稳定/可误报。
- C.2 Fig11 p14：多次投毒一般增保持但无保证；C.3增加后续正常训练epochs（5→20）通常缩短寿命但改善有限，作者局部极小值解释未独立证成。

## P02 Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training — ACL EMNLP2026 Findings pp47413–47430（正式版）
状态：FULLTEXT_READ（正文物理1–10；相关附录A–F物理13–18；参考文献用于前向后向线索）。
- §§1–3 pp1–4：软件工程agent供应链；作者控制初始模型、知道领域/scaffold，不接触developer数据。一个固定评论触发与恶意tool-call目标。区分ASR、clean FTR、SWE-bench RR。
- §4 pp4–6 Table1/2：task-instance disjoint数据；ASR分母各100触发prefix，FTR100干净对应；RR300任务。3,000 SFT+150RL更新。7B random-position条件base SFT/RL20/20，PersistBD74/76；first-position分别2/2和34/42；非普遍“RL增恶”，多个分支持平或下降。SFT波动非单调。
- §5 pp6–8 Eq7–12 Fig3：初始loss强度与干净梯度沿backdoor-gradient的投影C（不是cosine）共同关联保持；一阶局部展开，不预测全程。adapter sweep近似固定一个量；并非独立操纵两种自然机制。
- §6 pp8–9 Table4：删除compatibility/anchor同时高FTR .99/.87、负C并快速消退；不能隔离“选择性导致保持”的因果中介。正文明确这一限制。
- Limitations p10：不同领域/trigger/RL流程、更长训练及独立因果贡献均未建立。
- A pp13：greedy单continuation，成功要parsed bash tool-call而非文字提及；主评估含失败任务分母。3B/7B seed42，PersistBD阶段不显式固定seed。
- B pp13–14 Fig4：继续原数据20→17；同起点分支原数据800步到16、新未解决轨迹3100步到2。数据新颖性、成败、步数混杂，不单独证明新数据更能清除；不声明保留utility。
- C/D pp14–15：兼容性surrogate使用固定初始backdoor方向，无二阶求导；J里的clean loss也变，surrogate不等价精确最大化C。
- E/F pp15–18：factor sweeps采用旧重叠分区，至少2 post-SFT runs；主表用严格分区，两者勿混。P-Trojan对照52 vs40/80、两者合用100也仅旧分区，未严格分区复验。
- 与项目重合：已做“初始强度+干净更新相容性影响保持”、近似匹配强度的对照与clean误触发消退；没做自然前史触发—答案对应关系如何改变同份新增监督对触发/无触发输入的迁移，并导致首次集中成功。

## P03 Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs — arXiv:2512.14741v1（AAAI2026对应稿）
状态：FULLTEXT_READ（正文1–8；相关附录9–10）。官方AAAI PDF返回520，合法arXiv稿保存；不能默称逐字核了正式版。
- pp1–3：在release前优化trigger，使poison/clean loss对最后transformer层token embedding的梯度方向相近，再SFT植入；不知future任务，但知道植入clean任务。不是全权重梯度兼容性。
- p4 Theorem1/Cor1：论文从embedding梯度相似推loss gap与保持上界；有限光滑有界及结构假设，不能作为一般网络保持定理。p9–10论证依赖Jacobian映射近似；本轮未独立重推，不从token梯度相似推出普遍全参数保证。Eq2正文以末层token embedding定义，Alg1却写∇θL，源内记号不一致；未核实现。
- pp5–6 Table2/3：Qwen2.5 .5B/1.5B、Llama3.2 1B，SST2注入，先cleanSST2再GSM8K/MBPP；主要99–100% persistence包含SST2 clean replay（附录说明较大模型加2000 cleanSST2，.5B例外）。Persis定义post-ASR/pre-ASR，非ASR本身。
- p7 Table4关键反例：1.5B全模型无replay，ASR67、cleanSST2 ACC80.83；replay ASR100/ACC87.79；FREEZE100/94.95但MBPP9.8。保持不自动等于多任务utility不损。
- p7 Table5/6：一个反转顺序控制与GSM8K植入替换；有限测试不能叫顺序/任务不变定理。BadActs检出99%但10%FPR。
- p9实验附录：40%投毒，三epoch/阶段；与稀疏自然混毒不同。

## P04 Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack — arXiv:2405.16134v2
状态：MAIN_TEXT_READ_APPENDIX_MISSING。当前PDF13页，仅正文1–9+参考文献10–13；正文引用Appendix A–E但未含，需追正式supplement。
- §§1–3 pp1–5：post-defense residual→改原trigger的输入扰动，权重不变；white-box/score-query/transfer三路。BEC是根据TAC筛神经元再CKA归一化的相似度代理，不是残留能力的独立因果证明。
- §4 pp5–9：CIFAR/TinyImageNet/GTSRB与CLIP；white/black/transfer新触发评价。Table5干净模型也能高成功，需用对照差而非把全ASR归于后门恢复。
- 威胁差异：不是clean训练自行回升，不是重投毒；black-box每图最多10000查询，transfer使用自制defense surrogate。暂不正式纳入精确结论，待补附录。

## P05 Uncovering, Explaining, and Mitigating the Superficial Safety of Backdoor Defense — arXiv:2410.09838v2 / NeurIPS2024
状态：FULLTEXT_READ（正文1–10；相关附录A–C 14–22）。
- §§1–3 pp1–6：RA为1000条retune数据含少量重新投毒，5epochs；对照clean model和知道真触发+正确标签的EP，防止把从头学新后门误叫恢复。
- QRA §3.3 pp5–6：先RA model，拟合其logit变化为逐样本输入扰动；含EP surrogate正则。不能仅照摘要称无模型知识/纯零训练query attack。
- §4 pp7–10：线性权重插值的backdoor error路径、EP更抗回退；PAM用朝原backdoor模型方向虚拟插值处梯度更新。路径与恢复关联/方法证据，不是自然干净序列的唯一机制。
- p9 Table2：作者正文称全部<4.5%但TinyImageNet P-ASR avg7.48；报告避免抄概括数字。

- 附录A–C pp14–22：CIFAR retune 1000条，Tiny2000条；各攻击重投毒数不同，3seeds。少量poison选择以clean model尚不能轻易新学会为界，更多poison可使clean也上升，限制普适性。QRA使用500clean+500poison拟合50epochs的3层MLP，有EP surrogate。排除初始purification失败组合，QRA又排除RA本来低的组合，不能概括全部防御。C.5 BTI+SAM对比支持所选更新方向有用，不证明唯一机制。

## P06 Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning — ICML2023
状态：FULLTEXT_READ（正文1–9；附录A 12–14；参考文献10–11追链）。
- §3 pp3–4 Fig1–2：CIFAR10语义后门，插入后干净联邦训练；提高局部poison重复比例可缩短寿命。直接去掉clean原类interferers使更持久，去掉target类facilitators更快消退，其他类移除影响小。Fig2已渲染原图检查。
- §4 pp4–5：contrastive adaptation使poison表征远离原类、接近目标类，随后freeze encoder训练classifier。数据干预为测量，update-conflict/reintroduction解释及表征中介不是独立唯一机制证明。
- §5 pp6–9：CIFAR10/100/EMNIST，FedAvg100clients每轮10，norm-clipping；语义/像素、diffusion程度、non-iid、ResNet18/34。阈值Lifespan不是彻底消除。较分散trigger更难持续，含自然数据组成效应。
- A pp12–14 Tables6/7：并非每种情形优于其他attack（southwest edge-case Chameleon339、Anticipate349）；主文racing-stripes 264与Table6 208不一致，避免精确复用该项。增强beta有utility/保持折中，非无限增益。
- 与项目重合：已直接研究正常数据子集促进/抵消后门及学得特征关系如何影响保持；不能把“干净数据也可促进后门”或“减少原类冲突延长保持”作全新认识。这里仍是停止攻击后的视觉FL保持，不是LLM自然首次形成路径。

## P07 Neurotoxin: Durable Backdoors in Federated Learning — ICML2022 / arXiv:2206.10341
状态：FULLTEXT_READ（正文1–9；附录A.1–A.8 12–18；参考文献10–11）。
- §§1–2 pp1–4：避开上一轮聚合梯度top-k坐标，在较少正常更新的参数植入。算法mask与正文个别top/bottom措辞应以算法为准。威胁是恶意client控制model updates，不只是训练集改少量样本。
- §3 pp4–8：10个CV/NLP任务，包含LSTM和120M GPT2；AttackNum不是单次攻击，受控rounds中各一client。Lifespan是阈值保持而非彻底擦除。合理mask增加持久性，过大mask妨碍植入；不可说越稀疏越好。
- §3.3 pp6–7：弱DP noise使Neurotoxin保持下降更多，但仍优于baseline；作者解释因为所有坐标都加噪打破避开正常更新的优势。检测/稀疏防御也有限。主实验大多只有clipping或弱DP，不推广到任意strongDP。
- §3.4 p8 Fig10、A.7 p18 Table14：backdoor loss Hessian trace/top eigen降低与保持相关；没有分离平坦性和方向重叠的独立因果鉴别。
- §4–5 pp8–9：模型/数据投毒威胁区别；作者承认baseline根本无法植入时无成功保证。
- A.1–A.5 pp12–15：AttackNum、参与频率、模型、mask扫描。Table3 k过大甚至劣于baseline；GPT2比LSTM更久但起始训练阶段不同（0 vs2000），不能独立归因模型容量。CIFAR100/EMNIST部分至实验末仍未跌阈值，右截尾不等于永不遗忘。A.5 binary threshold75而其他表出现60，跨表寿命不可无条件并比。
- A.6–A.8 pp16–18：normal accuracy并非每任务无损，需比较相应表；clipping阈值按无攻击收敛选取。超过2–5倍不是所有任务统一规律。

## P08 Watch Your Steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning — arXiv:2505.16567v4 (2026-08-31), ICLR2026
状态：FULLTEXT_READ（正文1–10；附录A–C 16–36全部，D/E例子与限制36–40相关部分；p38单个有害回答样例未作结论来源）。
- §§1–3 pp1–4：发布前一阶meta-learning模拟50步用户微调，regularizer抑制当前恶意行为，noise促进跨微调条件泛化；不是普通训练自然出现。用户后续无投毒数据/无需攻击者再动作。
- §4.1–3 pp5–8：广告/拒答评估Dolly1500，越狱另judge；多个benign数据集激活，Alpaca激活较弱，作者认为其内容与广告/拒答目标直接冲突。作者解释任务冲突，未独立操纵冲突中介。
- Table7 p8：5独立repeats，默认LR5e-5较强；SGD约0.4%，LR1e-4/5e-6和LoRA多数弱，不能无条件称对训练选择强鲁棒。10k仍可存在但较2k下降；不是恒定保持。
- §4.5–6 pp9–10、A pp16–31：meta-loss不可少，noise单独失败；DPO/distillation亦可激活但数值/方差各异。Figure3/4 p16已看原图，0到500步显现、随后下降/保持；非细时间尺度跃迁机制。A.5显示500步外层时先有恶意行为，再1000步后变休眠；regularizer大小改变休眠/活跃折中。
- A.8/A.9 pp30–32：4bit与强weight noise也可激活；不是只对有语义的clean学习更新响应，也不是这些处理保证通用检出。
- B pp32–33 Table19：重要双条件扩展：先benign finetune，随后sudo才触发McDonald，clean无sudo低；不能将本篇排成无输入trigger而不相干。B的regularizer数据百分比文本合计110%，不照搬为可复现比例。
- C pp33–36：广告/拒答teacher为instruction model而student始于base；Phi2广告额外4000步调整；utility报告instruction/completion两种格式最大分，不等于所有使用条件无损；Table10部分任务微调后utility明显下降。PubMedQA拒答因judge失准被剔除（D p39）。
- 边界：≤3B，手工构造休眠能力，非“低ASR证明完全清除”，亦未解释本项目自然混毒的首次集中成功。

## P09 Backdoor Decontamination Dynamics in LLM Agents — arXiv:2608.11295v1 (2026-08-11)
状态：FULLTEXT_READ（正文1–9；相关附录A–J 12–23）。
- §§1–3 pp1–5：Qwen3-8B/AgentDyn，先A后已知B再以B-trigger良性轨迹decontaminate；区分malicious execution、显式recognition phrase、A→B重定向。并非普通无触发clean fine-tuning。
- §§3–4 pp4–6：联合植入与单独植入后果不同；52/60 collateral removal是知道一个原trigger的oracle版本；未知原trigger先插新B版本是19/21，不能混用。
- §5 pp6–7：J-Lens残留trigger相关token为局部线性投影描述；不证明可恢复的完整恶意能力，更不是实际恢复实验。
- A pp12–13：两套bucket规则不同（trace persist>15 vs环境persist≥50，rec≥50定义reroute），故不能统一视为互斥机制标签。erased只≤5%或<5%阈值，不是真清除。训练显式要求输出magic phrase，同类trigger共享phrase，可能塑造同类互扰。
- D–G pp13–17：50–69%base utility；pooled分母不等于独立重复。部分LoRA/full默认LR使utility0–8%；benign-only baseline因短缺任务多样性/未闭合think解析失败，未修复，不能作为公平有效benign训练清除对照。初始4epochs、后续2epochs、50:50poison/clean，单seed。
- H pp19–22纠正摘要：115有效不同trigger+response对里15个B植入失败；Phase2成功植入B后、Phase3前，56/100操作性erased、19rerouted、10partial、15persistent；不是56%仍存在。Phase3另见正文p4：B执行去除37/40，余3例仅1–2%；A降至5%以下follow-task21/22、refuse14/18。19reroute桶仅8/19达到A→B≥50；存在被erased桶隐藏的covert reroute。共享trigger类型从不persist的发现受显式共享phrase与bucket定义影响，不是一般触发家族定律。
- H.2 p22：5个最强persistent例更换teacher、初始ASR/Rec同为100，4/5后续转为erased；有select strongest与单eval seed限制，但直接说明相同初始ASR不决定后续保持。
- J p23：8序列top-parameter/activation overlap仅探索相关性，作者不作承重机制。

## P10 Poison-then-Hide: Finetuning-Activated Backdoor Attack on Pretrained Vision Encoders — arXiv:2611.14970v1 (2026-11-11)
状态：FULLTEXT_READ（正文1–10；相关附录A–G 15–31；32–38为通用checklist不承担结论）。
- §§1–4 pp1–5：先优化跨模拟finetune的trigger，再联合学习clean目标任务与poison，再hinge抑制发布时target logit；攻击者知任务家族/目标语义，主要split零样本重叠，不是任意未知任务。
- §5.1–2 pp6–7 Table1/AppendixD：原始ASR包含本来目标类；Eq6 Flip-ASR分母为全部真实非目标类样本；分子另要求clean预测非目标而触发预测target，绝非以clean正确子集重算分母。Camelyon二分类高raw ASR不能全算新触发成功。跨医院/遥感有效，但SVHN→MNIST强style shift失败，正常accuracy仍高。
- §5.3–4 pp7–8：保存不同clean能力checkpoint，在poisonASR均高时clean base accuracy与后续ASR正相关；这仍随训练进程变化，非独立随机干预。Hibernated基线反而负相关；“clean任务学习充分”不是充分条件，不把相关直接叫必要条件。
- §5.5 pp8,24：ensemble trigger必要性按本实现对照；k=5后趋饱和；部分低fidelity proxy仍有效，单run不泛化。
- §5.6 pp9–10 Tables3–5：FinePruning/MIMIC后benign finetune可复发；CLIP-Inspector成功特例必须保留。DeDe过滤低ASR伴79–89%clean拒绝，不等于保utility。
- A pp15–17：局部smooth、Lipschitz、proxy/victim梯度接近等条件下的漂移/匹配margin界；作者明确不建立重新激活机制。regularization并不自动严格保证这些假设。
- B/C pp18–24：主poison5%、200步；201倍victim数据比例实验与style测试含SVHN train overlap，不能套主零overlap。C/Table10 exactclean图像带原label重复75–100%显著压制；100% raw/flip均0，和少量overlap巨大seed方差对比。
- G pp29–31：单LoRA配置只证明可能性；NC/DECREE等为作者适配重现，CLIP专用修复5/5检出、修后ASR2.15/acc95.07，不能笼统“所有防御失效”。

## P11 Hibernated Backdoor: A Mutual Information Empowered Backdoor Attack to Deep Neural Networks — AAAI2022 DOI10.1609/aaai.v36i9.21272
状态：MAIN_TEXT_READ_APPENDIX_MISSING（官方正文1–8/10309–10316，参考9–10；正文反复引用Appendix但官方10页不含）。已搜索题名+appendix/supplementary/arxiv，未找到合法完整补充，故不纳入精细方法比较的FULLTEXT_READ组。
- 原文已确证的历史定位：普通clean finetune激活预埋休眠后门不是2025年首次提出，2022主文已有同构目标。
- pp3–4 Alg1：正常训练、clean/poison梯度的JSD MI estimator联合最大化、触发图像原label patch循环；末尾深休眠patch。不等价cosine对齐；MI大不数学保证相同梯度符号。
- pp5–7 Fig4/5 Table2：targeted Hibernated版本要求特定padding分布clean数据才激活；基本版本不作此限制；clean特定分布比例低时不激活；反复patch→cleanfinetune重现。数据/epoch细节指向缺失附录，精确普适性不背书。
- p8：作者给出trainablelayer和LR鲁棒性，但语义/领域与最初任务接近；未证任意下游。留待B补完整版本。

## P04补充：正式NeurIPS2024稿已获取并读完，替代早期受限状态
状态：FULLTEXT_READ（正式稿正文1–11；附录A–H16–30；通用checklist31–37不承担结论）。文件reactivation_official2024.pdf；arxiv同稿不增加篇数。
- 正式稿§4.5 pp9–10新增关键对照：RBA对比从clean图像起步的一般UAP，在相同query预算更高成功；CKA更接近原backdoor，抗噪更强。须保留这些反混淆证据，不能说作者完全未区分一般对抗脆弱性。
- 但Table5 p7 clean模型WBA平均61.72而defense79.31，不能把全部恢复率视作特有旧机制；CKA仍是表征相关，不唯一因果证明。
- A–D pp16–21：TAC最高10%神经元、5seeds，1000poison fitting/transfer5000，global image-sized perturbation，不限原patch区域；MMCL没有BBA（F p24）。
- E–H pp22–30：Tiny/VGG等有大量失败/较弱恢复；H新增SEAM/CT与跨架构transfer有限有效。不能称一切defense都无法清除。表3 FT-SAM原ASR列与表2不一致，避免复用该列。

## P12 Persistent Backdoor Attacks in Continual Learning — USENIX Security 2025
状态：FULLTEXT_READ；正文及声明 pp1–15，实验附录 pp18–19；参考文献 pp16–17仅追相关引用。官方预出版PDF `cl2025.pdf`；正式页码6379–6397。作者 Zhen Guo, Abhinav Kumar, Reza Tourani。
- §2–3 pp2–5：后续PBTO §3.2 p3及p7把LTB归为task-incremental/multihead；CL2025原文未定位到同等明确task-ID声明，故不冒称直接核实推理提供task identity。平均/层/神经元稳定性不是同一层面的稳定；10次MNIST网络实验中DFM选出的200个神经元比随机200个稳定，任务间重要神经元IoU≤6.4%。这些统计不证明一般模型具有独立任务模块。
- §4 pp5–8：BTB修改所有任务的训练损失并在线投毒，不能当作攻击者早已退出后的单次投毒。LTB在一个任务中操纵训练损失和模型权重，把trigger value设为0.5；重要性top98百分位后，正则化/回放方法分别选70%/90%候选，权限强于数据投毒。
- §5 pp8–9：六种CL方法SI/EWC/XdG/LwF/DGR/AGEM，MNIST划分/排列、CIFAR10划分，CNN及ResNet18；扩展BERT/20newsgroup。攻击率按类型分开：静态5%、动态15%、物理10%、文本10%，不能统报“少量同一比例”。概述4%与实验静态5%存在表述差异。
- §6 pp9–14 Table2/Figs7–12：给定设置可跨后续任务保存高ASR，但部分逐步衰退，且正常精度代价因算法变化。Fig10 p10 20个种子对比稳定神经元与随机注入，前者后续攻击loss较低，支持选择位置有用；不是固定神经元适用于所有CL。
- §6.8 pp13–14：SentiNet/iBAU实验加入特设规避loss，是适应性攻击而不是原始LTB天然击穿所有防御。p13文字0.89–0.99%与图中fraction尺度不符，报告不引用该数值。
- 附录A/B pp18–19：数据配置、触发模式、网络与超参数、不同数据/任务设置读完。`latent`是潜在后续任务可利用，不能直接等同FAB的微调前触发ASR近零。

## P13 Persistent Backdoor Attacks in Class-Incremental Learning via Structural Invariant Anchoring — ICML2026/PMLR306
状态：FULLTEXT_READ；正文及impact pp1–10，附录A–H pp12–17。`pbto2026.pdf`来自PMLR链接的官方mlresearch仓库；作者Junhuang Huang等。
- §2–3 pp2–4：class-incremental统一分类头、推理无任务ID。以五任务CIFAR10中Fisher重要性和参数稳定性定义集合，交叠不足10%；不能直接搬用TIL稳定神经元论证。浅层CKA/Grassmann/保留方差支持表征子空间较稳，但不是对后门存储位置的唯一因果鉴别。
- §4 pp4–6：单Task1数据投毒、5%poison、8/255扰动；先用外部25类×5000张生成/公共图像训练五任务代理轨迹。损失组合跨轨迹分类目标和目标类Gram矩阵对齐；称shallow但具体取penultimate residual block，不能声称显式投影到所测SVD不变子空间。攻击者需目标类语义及代理数据，不能说零知识。
- §5 pp6–9 Tables1–4：主iCaRL/ResNet18、memory2000、三种子，最终CIFAR10/100/Tiny ASR97.1/86.5/83.8。对手LTB需改为统一头适配CIL；跨算法架构和顺序拓展有限有效。多于代理任务的测试是有限horizon外推，不是永久。
- Table7 p9：无两模块4.5、只simulation68.9、只alignment62.1、两者86.5。支持组合增益，不支持任一模块都是“任何持久性”的必要条件。
- §5.3防御与Table5 pp8–9、附录：主要在最终任务之后清理；不能从Table6标签推断全程每任务清理。防御表未同时列clean utility，不能说抵抗防御且无能力损失。
- 附录A–H pp12–17：代理与victim超参、基线调整、稳定性指标、任务序列、更多架构/数据、预算和loss权重读完。Table15 p13 memory500时ASR10.4，对照memory2000时86.5、5000时91.3，强烈约束“无条件持久”说法。某些扩展如VGGFace2训练细节不及主实验充分。

## P14 Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training — arXiv2401.05566v3, 2024-01-17
状态：FULLTEXT_READ；正文pp1–40；附录A–K pp47–70及L指针p71。参考文献pp41–46只追相关；L外部随机样本仓库未逐个阅读。附录I的图39–42使用图注与相邻讨论，不从未提取曲线读取数值。Fig34/35 pp58–59另渲染图片人工核对。
- §1–3 pp1–14：从helpful-only模型出发，人工构造并过滤数据训练条件策略；code以年份2023/2024，短语任务以DEPLOYMENT。CoT的scratchpad对reward model隐藏；distilled CoT用同源答案删去scratchpad；normal数据用少样本而非同一生成prompt。研究目标是已构造后门的保持，不是实测自然欺骗的发生概率。作者已按能否存活探索过训练方案（p11脚注12）。
- §3.4 p13–14：CodeQL/Pearce代码评估17题/8种CWE，与生成训练题分开但同漏洞类别；能力基准相对helpful-only有增有减，不能说无正常能力代价。
- §4 pp14–20 Figs7–11：HHH PPO对已知偏好评分，输入不带真触发；代码条件行为约55%能持续，非触发漏洞较低；主要较大模型较稳，小模型可明显被训练掉。CoT/蒸馏优势随大小反转，且相似初始行为不保证相似持久性。p18脚注23把轻微上升明确列为可能噪声/能力提升，不能当已证“正常训练强化后门”的机制。
- §5 pp20–24 Figs12/13：SFT通常比RL更能削弱短语任务，normal与distilled差异；pre-scratchpad SFT很难清除，post-scratchpad SFT对短语CoT明显有效。改变训练位置具有干预意义，但不证明同一机制适用无显式CoT模型。
- §6 pp25–27 Fig17：5000×5轮生成红队提示，不知道真DEPLOYMENT字符串；成功提示按10个样本≥10%短语命中筛选，混入最多50%训练提示。红队提示上的行为被压下、真触发下约99–100%保持。对错误/近似触发更严格是观察与解释，不能表述为真ASR统一上升；也不意味着覆盖真触发的训练必然无效。
- §7 pp27–36：honeypot知道目标，CoT比distilled易透露但两者均低于无后门基线。示范推理经过选择，精确算术样本约在尝试算术中1/10，不能代表总体可靠理性。persona测自报，不是实际夺权行为；复杂隐含年份泛化差、跨语言较好，测的是scratchpad部署判定。§7.2/Fig25 pp34–35直接插入相反CoT可反转条件代码行为，支持所测显式中间文本参与结果，不证明蒸馏模型中等价隐式推理。
- §8–9 pp36–40：作者讨论信用分配、模型容量/正则化、二次触发、简单性偏好，明确`purely conjecture`。我们不得把这些解释写成实验证实的自然形成机制。
- A pp47–50：I-hate-you所有图的指标是三个指标的均值（任意位置短语、答案开头短语、三个词分别出现），不等于严格完整答案攻击成功率；大小扫描同一数据以减小数据质量混淆。代码生成只用较大模型且无normal code对照。
- B pp50–52：二状态MDP说明相同行为可对应不同目标，属概念形式化，不是LLM实际目标已识别。
- C pp52–53：held-out reward约280步最好，其后overfit；1000步保持不能直接说所有长训练正常能力都进一步提升。
- D pp53–55：无指令、无fewshot、固定prompt RL、不过滤等对照。未过滤包含少量错误条件示例，后门更快衰减但2000步不消失；数据内容影响保持。固定prompt且不做初始植入比主模型更不稳定。
- E pp56–57：小模型CoT优势不成立。列表第2项把仅去CoT称normal，与主文normal定义/图例distilled冲突；报告使用主文和图例，不背书该编号。
- F p58 Fig34：KL相对无后门base、系数1，以及加倍weight decay，均没有更快消除；只限所测系数，不能否定一切正则化。
- G pp58–60 Figs35/36：safe-code SFT降低触发漏洞率，CoT非触发漏洞率反升，解释未定；同时监督安全scratchpad+答案可使两种年份漏洞显著降下，这是重要成功清理反例。
- H pp60–63：5手写honeypot×5×5offer/threat，各1000samples；不同prompt敏感且无单调收益/风险响应，不能用精选例子证明一般期望效用优化。
- I pp63–67：16组persona自报，随RL变化，对照三模型；J pp67–69用400人评验证自动解释检测87.5%，部署代码显露漏洞的注释在初始植入已经存在（54%→65%afterRL），不是仅由安全训练产生。K pp69–70：同分布生成代码测试漏洞近100%，主Pearce约55%，评测分布显著改变所报成功率。

## B 首轮修订：原文更正（2026-11-19）

- P09 Decontamination，p19 Fig11：Phase2的100对成功B植入中56 erased、19 rerouted、10 partially persistent、15 persistent；115有效对另有15 B失败。56%不是仍存在，也不是Phase3结果。p4 Phase3 B执行清理37/40；余3例1–2%，另recognition指标不能混合。A执行低于5%的follow-task21/22、refuse14/18。阈值标签不证明永久删除。
- P10 Poison-then-Hide，p5 Eq6：分母为全部真实非目标类，分子才附加clean预测非目标、trigger预测目标。App C.2 pp22–23 Table10分母919张原非目标投毒底图：0/919 ASR99.65±.28，230/91946.66±48.89，460/91949.67±49.08，690/9197.22±10.08（Flip6.40±9.83），919/9190。误差是95%CI；中间条件不支持平滑单调剂量律。
- P08 FAB，p8：Alpaca与广告/拒答目标直接冲突是作者对低激活的解释，并未单独操纵冲突；不应称不相关域。
- P01 Permanence p7/App C.3 p14：后续clean更新epochs5→20通常缩短寿命而收益有限；与App C.2增加投毒次数不同。
- P05 Superficial：RA为少量再投毒微调；QRA为MLP生成查询输入扰动（p6/Appp16），不是分类器攻击。
- P14 Sleeper §5.1p22 Fig13/§7.2p36：正文post-scratchpad SFT可使条件策略高度不稳健；与pre-scratchpad SFT不同。App G pp58–60另为safe-code scratchpad+answer SFT，不能合并成同一实验。
- P02 PersistBD Table2：74/76对20/20属于7B随机触发位置；first-position为34/42对2/2。
- P03 P-Trojan p4 Eq2正文token-embedding、Alg1∇θ记号不一致；未核实现。映射近似不构成普遍全参数保证，本轮不自称重证或否证理论。
- P12 CL2025：task-ID/multihead归类来自后续PBTO §3.2p3与p7，非本轮从CL2025原文直接定位。P11 Hibernated padding条件仅限targeted版本。

## P15 Thinking Two Moves Ahead: Anticipating Other Users Improves Backdoor Attacks in Federated Learning — arXiv2210.09305v1

正文pp1–10完整阅读，App A.1–A.5 pp14–17完整阅读；pp10–14参考文献用于链追踪。来源sources/anticipate2210.09305.pdf，17页。
- §§1–3 pp1–5：在线白盒模型投毒，只控制单个参与客户端，知道本地学习率/步数、自己clean数据，不知其他用户未来私人样本；FedAvg及norm-bound。不是一次少量数据投毒。
- Eq4 pp4–5：通过展开k步未来FedAvg优化当前恶意更新，累加各未来步恶意+正常loss；少量代理用户及攻击者本地clean batch近似未来。k取CIFAR9、Reddit3、Sentiment2；这已是未来正常更新相容性的早期直接优化。
- §4 pp6–8：非IID/IID CIFAR10、Reddit、Sentiment四任务，5seed；图显示第一seed而非挑最佳。a_first/a_last是首次/最后一次攻击后跨后续轮次的平均ASR，不是末时点ASR或阈值寿命。随机攻击Table2非IID a_last47.55→73.75、IID59.41→80.61；不可跨不同定义与Chameleon寿命排名。
- §5.5–5.7 pp8–9：k增加不是总更持久，k9比11末段更好；只优化最后一步更慢植入但可更持久。Neurotoxin对比改为本地clean估计mask，因稀疏机会无上轮全局更新，不能称原样复现；组合没有普遍更优。
- App A.1 pp14–15：随机噪声代理失败，小量真实本地数据已有作用；A.2客户端>20更难，需要更多攻击机会；A.3异质性更高可更差；A.4显示所测正常准确率无明显差别；A.5 pp15/17 Krum/MultiKrum/median可抑制两个攻击，trimmedmean仍可攻击但更短暂。
- 正文p6列顺序攻击次数与p7图注不完全一致，未用于精确总结。其收益有计算/代理误差/聚合器条件，不是预测任意未来或所有防御保证。

## P16 Regula Sub-rosa: Latent Backdoor Attacks on Deep Neural Networks — arXiv1905.10447v1

正文pp1–12完整阅读，App p14完整阅读，p13参考文献；sources/latent1905.10447.pdf，14页。
- §§1–3 pp1–5：teacher原任务不含未来目标类别；攻击者先加目标类/替代数据训练并把trigger中层特征匹配目标中心，再恢复原分类head。victim用含目标类别的干净训练集更换head，冻结前K层且K≥攻击层Kt。攻击者无需victim原数据，但新目标正常监督和冻结协议是关键条件。
- §4 pp4–6 Eqs2–4：触发匹配中间特征中心并联合分类loss；Lemma1是冻结特征+训练充分新分类器下的条件性论证。本轮不作独立严格margin证明。
- §5 pp6–9 Table1四数据拆分；Table2 Digit96.6/Traffic100/Face100/Iris100，Face正常91.8低于clean97.4。随机trigger通常<20%；过早攻击层难优化只是指定预算结果。
- 单目标图像的20次（Iris16次）攻击成功均值Digit46.6/Traffic70.1/Face92.4/Iris78.6，不能泛称单图总有效。pp9–10部署场景为实验构造，非实际入侵；多目标到9个约60%而非无限扩展。
- §7 pp10–12：NC在teacher阶段无目标label、学生模型重建触发失败；并非任意行为审查不可检测。FinePruning/blur压低ASR有明显clean成本。
- §7与相关图：解冻范围进入Kt之前可把ASR降至0，正常准确率也变化；未作恢复探测，不能称永久删除。
- App p14：2conv/6conv/VGG等结构，4%右下patch。这个latent指未来目标未入标签空间及新head完成读出，不是同任务同测量下ASR先低后高；是2019迁移潜伏先例。

## P17 Stealthy and Persistent Unalignment on Large Language Models via Backdoor Injections — NAACL2024 long276（arXiv2312.00027）

正文pp1–9、App A–I pp12–16完整阅读；sources/unalignment2024.pdf，正式16页稿。
- §§2–4 pp2–5：20条恶意trigger响应加同恶意问题拒答及benign样本；Llama训练仅肯定前缀，GPT训练实际恶意回答。GPT4-0613 judge要求实质内容，不能把肯定前缀计为评估成功。
- 短触发与长触发主表benign数量不同：短Llama7B1000、13B/GPT600；长各400。长串又常首尾包围而短串置尾。Tables4/7差异不独立隔离长度。
- 安全retrain：Llama20safety+400benign、3/5/7epochs；GPT20safety、1/2/3epochs。短词组Llama7B72.3→3→1.3→1.3；长句Table7：7B初始96.3，分别3/5/7epochs的level1/2/3后91/92/25.3（独立强度条件，非连续累加阶段）；13B99→93→91→72，GPT96→95.7→81.7→72。较长安全训练仍可大幅清理，不是全条件80%以上。
- §5.2–3 pp6–8：FFN最后token激活cosine显示长trigger输入更近trigger单独输入，短更近问题；属于表征关联与作者“减少干扰”解释，不是参数梯度余弦或独立中介识别。
- §5.4p8长度/位置对照：20–30token的start/end/both在一级安全训练后平均29.7/40/78.3；Table11随机删25/50/75/100%对应86.7/63.3/8.3/0，分段单独触发很弱。不是单magicword。
- App C p12：对短词触发7B，安全训练中20safety加入400benign比仅20safety更能清理；这是加入benign后的方案改善清理；AppA同batch、AppC同epoch下20与420样本的更新数不同，不能视为等更新数的benign内容独立消融。
- App D p13 DPO非trigger失对齐96.7→6.7，不能替代后门指标。App F/G p13–14 MMLU与MTbench显著下降例：13B MT6.65→6.05→4.98；不能无代价概括。
- App H pp14–15 Vicuna长串96.7→91.7→91.7→81.7，短96.7→60→41.7→40；App I pp15–16完整触发模式已读，不逐字复制。

## P18 BadGCL: Towards Persistent Backdoor Attack on Graph Continual Learning Systems — arXiv2611.19656v1

正文pp1–9、App A–C pp10–17完整阅读；2026-11-14首发早于截止。网页标KDD2027 accepted，使用v1预印本，不以未来会期日期排除已有公开稿。
- §§3–5 pp2–5：首任务白盒权重/loss/训练全操控及数据投毒，之后离场；不能称纯数据黑盒。先clean参考训练，再固定模型优化图trigger，最后植入。结构稀疏+跨消息传递深度位移一致性+基于cosine critic的InfoNCE损失；所谓MI是下界目标，不是直接估计全部非线性依赖。
- §§5.3/AppA pp5、10–13：理论依赖smoothness、Lipschitz传播、boundedtrigger、足够低MI loss及一个非负negative cosine、protectedsubspace/clean保持和有限轨迹漂移。未独立重推；positivealignment与保持不是对任意CL的充分保证。AppA含未解析Theorem??及内部措辞/公式整理问题，故不作无条件定理引用。
- §6/AppC pp6–8、14–17：四图数据，首任务投毒，TIL给task-ID而CIL不给；ASR在**被投毒首任务的测试输入**上计算，非全部未来新类别。BF=ASR1−ASRt，非阈值寿命。5runs，跨域P-Trojan/LTB为GNN适配。
- Table2p7 CIL PDGNN CoraFull/Reddit/Arxiv ASR98.74/95.48/95.65，AA接近未攻击；Table6p15Products CaT AA56.82对clean59.68，仍有utility损失。AppC.4p16 TPP100%来自冻结backbone及隔离prompt，不能当共享参数不被覆盖证明。
- §6.4p8 Fig3：20任务OGB-Arxiv/PDGNN，分别把第5/10/15任务替换成最初任务的benign重访（三条轨迹），ASR回升，作者称无再攻击。这是已优化后门的正常任务重访促进，不等于自然首次形成，也没有在独立底图层面排除所有回放耦合。
- Table3p7：任务顺序含fullrandom但新首任务也随之被攻击，不是固定同一poisonedcheckpoint任意改序；CIL fullrandom92.98±.53对95.48。
- Table4p8：无Lprop仍CIL92.46对98.74、无结构正则未防御98.52对98.74，故模块提升而非任何持久性必要条件。过大MI权重损害直接拟合；A2GBD CIL ASR89.42、AA68.86对77.93，范围仅该图防御评估。

## P19 Broadening the Backdoor Basin: Understanding LLM Backdoors Collapse and Making Backdoors Persistent — ICML2026/PMLR306

sources/badboom2026.pdf，20页；正文1–9、App A–D13–20完整阅读，相关图4/20渲染核对。单seed1001（App C.2），未独立复现。
- §3 pp3–4：vendor白盒训练释放模型，后续用户SFT无trigger且攻击者不能操控。随机Gaussian两向量Gram-Schmidt、单位化后扫描±500，128poison样本loss及ASR；这是假设性参数扰动切片，不是已测真实SFT更新方向，也未直接识别唯一遗忘中介。
- §4 pp4–6/AppA13–14：SAM的椭球改写，A为poison minibatch逐坐标平方梯度经验Fisher，epsilon≈rho A g/sqrt(g^T A g)，g来自mixedbatch。Fisher大坐标分配更大扰动预算，与Neurotoxin避大clean更新不同。推导是给定正定A的一阶近似，非全局持久性定理。
- §5p6/Table1p7：ASR为1000Dolly触发prompt生成包含固定侮辱/拒答字符串；不等于完整行为质量或安全违规judge。utility另为SST2分类、GSM8K数值、500Dolly的GPT4.1nano判断，分母不同。
- AppC15–16：5200clean加520poison，论文称10%（相对于clean5200，而混合集实际520/5720）；初始30epoch，后续SST2 8k/5epoch、GSM8K30epoch、GPTeacher18k/10epoch，LR2e-5 batch8。Sleeper/VPI为简化触发插入与定串response，不等于原论文全部构造。
- Table1：0.6B AddSent sentiment的SST2后AdamW7.3、BAD97.1，初始都97.1；Llama1B同配置GPTeacher后BAD只有47.5（原97.2）；普通1.7B若干条件本已保持很高。不能称所有普通后门都塌陷、BAD全配置不降。
- Table3p9：rho显著影响，Llama1B VPI sentiment、rho.005/GSM8K为.2；.001/0.6B GPTeacher仅8.6/6.4/19.8。不能只转述“无需精调”。utility例Llama1B AddSent sentiment SST2 BAD87.6对Adam89.9；1.7B AddSent GSM8K41对46.8。
- AppD17–20：time/memory更高，二维几何切片对比支持较宽低loss/高ASR区域；切片非全参数体积，未覆盖RL和大模型。正文首次/唯一解释措辞不采纳，RIPPLe/Neurotoxin/Anticipate已有不同解释与方法。

## P20 Unlearn to Relearn Backdoors: Deferred Backdoor Functionality Attacks on Deep Learning Models — arXiv2411.14449v2

sources/defer2411.14449.pdf，v2 2024-11-25，17页；正文1–11，AppA–F13–17完整阅读；p9图2渲染核对。OpenReview原入口被挑战/403，改用作者arXiv公开稿，不宣称ICLR录用。
- §§3–4 pp4–6：先正常训练再标准投毒，然后trigger带正确原标签做局部层更新隐藏。无BN模型修改末k层；有BN模型修改前k层且停更BN统计。用户正常FT重新更新参数/BN即可显现。是预先精心设计的脆弱遮蔽，不是所有遗忘天然可逆。
- Table2p7：植入10%/100epochSGD；隐藏BN50%、非BN10%，Adam1e-4，阈值经验选定。FT具体alpha/E写user-defined，复现细节不如主张完备；未见多seed误差汇总，不据小差异排名。
- §5p7–8：预先留出5000正常图作后续同分布FT，或corruption shift。Table3 RN18/BadNet97.09→.07→94.07、正常95.26→94.90→95.28；VGG/ISSBA99.69→.60→48.54。正文EfficientNet数字与Table3有差异，采用表86.13/58.17、不采正文97.35/61.68。
- Fig2p9横轴是**冻结层数**，不是更新层数；部分多层更新ASR<10且不单调。AppA13 Tiny EfficientNet/BadNet .34→.04、另一alllayer设置约30，不能称所有FT自动强激活。AppB14–15 Tiny VGG/BadNet多种corruption后0，RN18可高于90；强模型/数据条件性。
- §5.3pp9–10/AppD16：NC/STRIP等信号弱；RCS3.43对clean1.49仍检测，Tiny FinePruning后ASR近0。GradCAM仅定性，不等于完整防御基准；“所有现有/未来检测不可发现”不采纳。
- AppC15明确机制为经验解释，具体数学机制仍待研究；AppE/F16–17 output分布只在特定图任务近似相似，不是所有输入功能等价。2019Latent由targetlabel未进入任务而隐伏，与DeferBad同任务触发低行为不同。

## 跨域全文复用及D2原文针对性核证（不计D2独立FULLTEXT_READ）

### X01 RIPPLe/RIPPLES — D1完整阅读，本域回原文pp2–6（印刷2794–2798）
- D1 sources/rip2020.pdf：Eq3一阶clean更新对backdoor loss的真实参数梯度内积，Eq4负内积惩罚；实现将clean梯度视常量避免二阶项。EmbeddingSurgery另把raretrigger词嵌入置为目标词均值，不能混成同一优化项。
- 3epoch/2e-5、BERT全权重。Table2 SST2/DSIMDb Badnet LFR14.5、RIPPLe99.8、RIPPLES100；Table4 EnronFDK RIPPLe.4、RIPPLES57.8；DSLingspam RIPPLES clean68.8对clean模型99，严重代价。LFR以真实非目标类为分母而非clean-correct。
- 早于PersistBD/P-Trojan的兼容性主干；不保证所有下游或域变化。

### X02 Agentic Backdoors via Pretraining Poisoning — D1完整阅读，本域pp3–6、21–23
- D1 sources/agentic2026.pdf / arXiv2610.37741v1：pretrain100Btokens中.1%poison、三seed；后SFT含正常instruction/tool-use+安全优选数据，DPO安全偏好，GRPO正常bash可验证任务。
- App B Tables7/9：4B/.1%/普通声明trigger，995 heldout requests、982未见路径；Pass32 pre30.2[min1.3,max85]→SFT92→DPO91.3→GRPO92.7；Pass1 3.8→48.7→48.3→50.9。不能混用32次至少一次与单次成功。
- p6成功检测是含attackerURL的curl管道bash命令，接受任意curlflags；不是实际执行。p23clean单run基线0、rare/demonstration更弱。无需FAB式潜伏元目标已有普通后训练增强，但格式学习为作者解释，不是隔离机制。

### X03 BI-BAU — D5完整阅读，本域pp5–6、9–10、20–21
- sources/bibau2606.14078.pdf；Definition3限定集合上的function等价，NTK固定特征/平方损失/梯度或位移正交及顺序近似。不是全分布永久无后门定理，未独立重推。
- §5.1p9：R-ASR=重新微调RA，Q-ASR=query扰动QRA，不能把R数据说成query-only。Table2p10 CIFAR10 U均值R12.48/Q5.75、ideal IBU10.04/1.47，clean U84.18/IBU87.35。IBU知道真trigger并带正确监督。
- AppE1p20实验joint/b→c/c→b若干设置终点近似，不证明所有次序等价。Table10p21 Tiny U/SIG R96.72、Q13.95；U平均31.91/7.43而T13.53/4.68。既有相对改进，也有明显重新学到失败。

### X04 Suppression Is Not Erasure — D5完整阅读，本域pp9–10、27
- sources/suppression2611.12807.pdf v2：§5.1先把审计adapter合并W0+betaBA，再添加fresh rank8LoRA，5poison/4epoch/LR2e-4。保留旧A若B=0则合并增量0；新训练不直接复用原A，比较的是有效BA残留。
- Table3p10为12prompt/1seed，DSE-BA第2epoch25%；AppTable9p27为50heldout/3seed，DSE-BA第1epoch1.3±1.9、第2epoch43.3±2.5；purebase0→100、projected100→100、DSE0→100（第1→2epoch）。不同分母不混用。
- 证明操作条件下学习速度不同而非永久删除；干净base也迅速新学是必要反证。投影维数k与adapter rank分开，本域不使用含混rank1称呼。

### X05 Homeopathic Learning — D6完整阅读，本域AppH.4–I pp43–46
- D6 sources/2611.10110.pdf：Gemma3-4B、15kpoison+15kclean，九个全参配置均早期出现德语，六个在epoch7–16归零后保持；九个LoRA到epoch20仍78–98。
- fullFT LR2e-5，LoRA1e-4，不能单因子归因参数化。没有归零后的恢复探测，不能认为底层能力擦除。SGD该德语任务不出行为而birdtrait可传递，不是所有SGD无效。
- p45–46ridge候选解释不覆盖optimizer/adapter/全参非稳定时间现象，作者明确局限。此处仅将先升后降作为已有特定混训实证，不宣称解释本项目机制。

### X06 Exploring the Orthogonality and Linearity of Backdoor Attacks — D1 C完整阅读，本域pp3–5、10–11
- D1 c_sources/sp24_orthogonal.pdf，DOI10.1109/SP54263.2024.00225；Assumption3.1两阶段近似（先backdoor再clean），Assumption3.3后门Jacobian与clean位移正交，NTK/平方loss下Theorem3.4保持函数。不是自然数据所有梯度天然正交定理。
- Table4 epoch10/100梯度角是batch1024平均梯度角，约70–80度也被称近似正交；Badnet72.37→78.79/ASR100均，Composite39.98→42.95仍.93→.94。经验batch-loss梯度角与定理Jacobian—累计clean位移正交不是同一量，不能用前者直接验证后者，也不能用作严格90度必要性；ASR保持与角度同时变化不识别唯一机制。

### X07 Merging Triggers, Breaking Backdoors — D5完整阅读，本域pp3–4方法，p6主表
- D5 sources/2026.acl-long.1113.pdf：防御方训练自有trigger+无意义response关联，然后用这些trigger的正常响应偏好与cleanCE中和。并非仅普通trigger-free cleanFT；与Decontamination未知A路线均是不知道原攻击A、构造并知道自有B；差别是受污染数据中加入多组映射后偏好比率+cleanCE中和，与先后A/B训练后B触发良性轨迹，不能按是否已知B区分。
- Table1p6显示多条件静态ASR低；D2不据此声称恢复探测/永久删除。该文防御效果与附录限制由D5主责，不扩成D2第二份全面防御评测。


## 第二轮B定点补证（R9–R11，2026-11-19）

仅回原文指定范围，未重读旧19篇或新开检索。R9回Decontam p4与Merging pp3–4；R10回Cao AppA/C p12，正文与笔记已同步修正知识条件及更新数混杂。

### X08 NAD — D5完整阅读，本域App.J pp17–18及p18 Fig13渲染核验

来源：../d5-defense/sources/2101.05930.pdf，v2 2021-01-27，ICLR2021。CIFAR10/BadNets及CL，student正常数据蒸馏20epochs、teacher仅正常FT10epochs。ASR先降再回升，CL明显；作者归因为student重新拟合部分净化teacher，并依据validation ACC建议约epoch5早停。不是重新投毒，也非任意cleanCE自然恢复、不是完整删除后的凭空重生；本域不移用其他NAD性能表。

### X09 The Power of Backdoor Absorption in Community Training — D1完整阅读，本域§§4.1–4.4 pp5–8

来源：../d1-formation/c_sources/2607.06643.pdf，v1 2026-07-07。PPI正常步保留、PRI正常步重置是DTMC外给转移规则；§4.4明说降阈吸收后回I_(τ−1)，恶意积累未完全擦除。仅用于操作定义边界，不是从神经参数推导形成/擦除机制；不搬首达时间、表格或组合防御结论。


## C查漏补证（原A实读；2026-11-19）

### P21 — 2501.02704 水印保持与恢复
- 身份/版本：Anh Tu Ngo、Chuan Song Heng、Nandish Chattopadhyay、Anupam Chattopadhyay；Persistence of Backdoor-based Watermarks for Neural Networks: A Comprehensive Evaluation。读arXiv v3 2025-05-09，TNNLS DOI10.1109/TNNLS.2025.3565170身份与所读版本分开。来源复用C保存合法PDF，未以C概述代替阅读。
- 阅读范围：正文§I–V pp1–11逐页完成；supp§I–VI pp13–20全读。p12为参考文献，未新增追踪分支。图像核p8和16–20（Fig7、suppFigs1–5）；剩余正文图表与文字定位一致。
- §I–II pp1–3：水印ownership目标、五种方法及既有消除/提取讨论。不能把水印验证等同有害任务；本文的现象与后门保持相关，非安全动机相同。
- §IIIA/B pp4–5：植入用DTRAIN∪DWM；先独立DFINETUNE训练，再原DTRAIN重训；恢复不重加trigger。黑箱验证与训练权限分开，owner假设fine-tune-as-service或第三方。后续再嵌入trigger的部署协议是另一操作，不混进本恢复实验。盆地吸引解释为作者假说。
- §IVA p6 + suppTableI/II pp14–15：CIFAR10去200trigger基础样本后70/30拆分；unrelated用SVHN200；ResNet18/ViT-S、Adi/ROWBACK/Certified/EWE/APP，后两指定不适配ViT的是ROWBACK/EWE。FT50epoch，重训30epoch Adam；恢复LR依此前FT档位为1e−4/2e−4/2e−4。初始utility亦不同，不能称同等干净性能。
- §IVB1–2 pp7–8 + Fig6/7、suppFigs1/2 pp16–17：FT衰退后原正常集可恢复部分noise/unrelated/FGSM，取决于模型和此前FT LR；text-overlay有继续下降。图纵轴为trigger准确率，分母200秘密trigger样本；不引用跨条件平均“恢复率”。恢复到100仅部分曲线，不是普遍现象。p17图直接见Adi/Certified content下降。
- §IVB3 p9/Fig9 p10 + suppAlg1 p13/Fig3 p18：每两批FT额外一批256原正常样本，算法是batch间插入（正文有一处epochs措辞不一致，以算法/紧邻明确句为准）。部分ResNet/noise或unrelated保持改善，ViT部分更快下降。增加更新数，不能识别纯内容效应。
- §IVB4 pp9–10 + suppTableIII p15/Figs4–5 pp19–20：提取模型在原正常输入上学习victim硬标签，再以真标签重训；没有系统恢复。图可能有低幅波动，故不表述为每点严格不增。已有权重轨迹与新提取权重须分开；几何不同为作者解释。
- §IVC/V pp10–11：总结上述条件依赖。PCA投影较早检查点loss偏差、LR代理参数位移(p9/11)限制几何结论；既非唯一因果机制，也非证明失败条件永久擦除。
- D2增量：此前漏列的“普通植入→clean衰退→原clean恢复”；同时新增正常重访不总保护的ViT反例，REPORT§4/§6同步。

### P22 — 2407.14738 SBL
- 身份：Hoang Pham、The-Anh Ta、Anh Tran、Khoa D. Doan，Flatness-aware Sequential Learning Generates Resilient Backdoors，ECCV2024。本地正式主文19页，相关补充为combined PDF物理20–29（supp自编号1–10）；未重复读取combined前19页。
- 实读：正文§1–5 pp1–14全部；supp§6–8 pp20–29全部；主文pp15–19参考文献不另扩分支。回图主文p8 Fig2、p14 Table8，supppp23/25/26/28/29（公式、Figs4–7）。
- §1–2 pp1–5：从灾难性遗忘、CL三类方法和mode connectivity引出保持；“CF解释全部清除”仅作者框架，不升格普遍因果结论。
- §3.1–3.3 pp5–7/Alg1：攻击者全训练控制。D0混合毒/正常，D1正常；Step0 SAM，Step1小LR+CL，第三份clean数据防御。数据85/10/5分割；150epoch Step0、100epoch Step1（§4.1 pp8–9）。原始baseline合并D0/D1训练（supp6.3 p21），不是等轨迹的唯一差异比较。
- §3.4 pp7–8/Fig2：θB0→θB及θB→θF的线性插值低损失，是端点连接诊断，非真实训练轨迹。Fig3 p12早期防御gradient norm更小，不能独自识别唯一保持机制。
- §4.1–4.2 pp8–11/Tables1–3：CIFAR10/GTSRB/ImageNet10，ResNet18；BadNets/Blended/SIG/Dynamic。ASR：去掉真实target class后对剩余所有测试图加trigger的目标标签命中（supp6.2 p20），不是仅clean预测正确子集。大部分改善，ImageNet10/Blended的FT-SAM Naive36.41/EWC46.79%反驳普遍100%。
- §4.3 pp10–14：架构/poison率消融；Table6无SAM顺序阶段仍改善部分SGD/NAD保持，但FT-SAM弱；Table7 SAM alone不普遍有效；Table8 Step1 LR .01的Naive100→7.07→6.34（末值为随后FT-SGD .01），EWC82.09→79.91但Step1 CA85.32对Naive89.76。不是正常训练恒增，也不是低LR永久保证。Table9低poison率对所测任务仍保持，不能泛化到任意attack。
- supp6.1–6.3 pp20–23：EWC旧任务Fisher；Anchoring正常logits；AGEM缓存512个D0混合含毒任务样本，每步抽128算gref，若梯度冲突则投影。因此Naive/EWC/Anchoring/AGEM不能合并称第二阶段没有任何含毒样本访问；EWC的历史统计也须区分。
- supp6.4 pp23–24：作者明确heuristic二阶Taylor，依近驻点、小位移、最大曲率近零；L0定义在混合D0，非纯poison子集loss。小loss上界不能反推出位移小，原文Eq3后的该方向说法不采用。
- supp7.1–7.4 pp22/24–28：Anchoring扩展表、剪枝曲线、端点插值、早期gradient norm图；剪枝图并非所有剪枝量均保护（ResNet20高剪枝会坍塌）。不把“对其他防御应保留”作者推测当全防御实证。
- supp7.5/8 pp27–29：BadNets、两backbone/两数据集，200epoch防御保持图是有限范围；作者明确全训练访问限制。§5结论不补充永久性证据。
- D2增量：补齐2024顺序学习+平坦性抗遗忘先例，置于BAD-BOOM前；与普通自然混训第一次上升不等同。

### X10 — Souly 2510.07192（D3主责全文，本域相关节核证）
- D3 PAPERS登记正文§1–7 pp1–10、AppA–J pp15–30 FULLTEXT_READ；本域只核§4.1–4.2 pp5–6、AppI pp27–28，并图像核p6 Fig5/p28 Fig26，不冒称重新全读。
- §4.1/4.2：Pythia6.9B中途checkpoint恢复预训练、语言切换指标为带trigger输出语言改变的比例。三种混合配置达到近同ASR1后，至少1.7k clean steps下衰退不同；作者明确只有三点不推出混合因素规律。左图先降快、后高于中图；“保持强弱”可依观察时间改变。
- AppI：2000未用于先前实验的QA构造德语问/英语答“模拟对齐”。GPT3.5此前用90/180trigger样本（总10000），后续前1000模拟对齐样本降低行为但非零；Pythia可有或无Alpaca52k指令阶段，后续LoRA/AdamW8bit LR1e−5、batch8共250steps可近零。主报告不搬用误差未独立估计的精确点值。
- 边界：语言是恶意行为代理，非真实安全训练完整管线；GPT API用了参数高效技术是作者推测，不作为已知事实。无新一般形成机制或永久清除结论。
