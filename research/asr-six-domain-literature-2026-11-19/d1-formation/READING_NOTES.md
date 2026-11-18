# 领域1 A逐节全文阅读记录（含B第一轮后修订）

读取日2026-11-19。所有页号为本地PDF物理页，论文印刷页另列。只将正文全部读过且相关附录已核对的版本标FULLTEXT_READ；不是复现或独立数学证明。

## arXiv:2609.36117v2 — Why Backdooring Neural Networks is so Easy?
- 物理pp1–9：§1动机/相关工作、§2 Gaussian mixture三簇+quadratic neuron、§3闭式/线性/特征/部分对齐、§4实验、§5限制均已读取。pp10–11参考链：Manoj/Blum capacity, Khaddaj spurious feature, Souly constant sample。
- 相关附录已核：C.4 p18端到端方向只在合成例；D pp18–19两通道近似；E pp19–20数表。A–C其余长证明未作完整逐式复核，不声称定理独立验证。
- 关键：Definition3 p4把u设成whitened trigger，非证明自然训练必然取得该方向；§4 p9 practice从真实数据估计的Gaussian mixture抽样，不可笼统称真实图像实验。特征曲率导致π α^4 对比线性π α^2及触发强度预算指数，是条件理论；不是SGD时序/首次ASR跳升因果识别。
- 文内T=√(vᵀΣ⁻¹v)与“大方差更大/更便宜”的散文关系存疑，报告不传播这个几何方向结论。

## DOI:10.18653/v1/2023.acl-long.194 — A Gradient Control Method for Backdoor Attacks on Parameter-Efficient Tuning
- 物理pp1–11（印刷3508–3518）正文§1–8及Appendix A.1–A.5全部已读；pp12–13清单已读。
- §3 pp2–3 Figures1–3：BERT冻结、顺序/并行PET、SST2；poison loss较快；step800不同层梯度幅度与方向冲突。观察/作者解释，不是形成前积累的受控中介证明。
- §4 pp4–6：CLNorm重分配层间poison幅度；ILProj不能写成所有层消冲突——p5明确底层0–5完全采用未投影梯度，上层6–11采用投影梯度；A.3分层并非处处优于全投影（IMDB→SST2 92.6 vs 93.0）。Appendix A.1 β底0–5=1、上6–11=0。
- §5 pp6–8：50%poison；10epoch注入、5epoch干净续训，跨SST2/IMDB、Enron/Lingspam；结果为分类LFR非LLM完整答案。Table2并行Lingspam→Enron Ours16.6，不能声称普遍保持。§8及A5保留大数据续训失败和RoBERTa非全格最好。

## arXiv:2312.09057v1 (USENIX Security2024 work) — On the Difficulty of Defending Contrastive Learning against Backdoor Attacks
- pp1–15正文§1–9全部；Appendix A–D pp17–19全部；p16参考表待仅作书目链筛选。
- §3–4 pp3–6：统一TRL；1%poison ResNet18 CIFAR10/100/ImageNet100，SL与SimCLR/BYOL/MoCo比较。Figure2并非后门一律更快：CL毒/净loss下降近似同步。Figure3/4仅表征纠缠测量。
- §5 pp7–8：trigger强度干预，SL ASR饱和，CL先增后降；MI/ER支持语义/trigger共需，但作者“independent/intertwined”解释不等于参数完全独立的证明。
- §6 pp8–11 Tables3/13：ABL以低loss隔离在CL明显失败；不是所有训练范式统一快速学习。
- §7 pp11–14：其他架构/patch重复；密度过滤与CLP防御有参数敏感性；泛化SSL/文本留作未来。Appendix D.4 p19已投毒SimCLR encoder用每类50 clean加额外50 target-class poison同时微调encoder/head后出现poison loss更快；目标、数据规模及毒比例同时改变，非loss单因素。

## arXiv:2110.11571v3 — Anti-Backdoor Learning: Training Clean Models on Poisoned Data
- pp1–10正文§1–5/impact全部已读；相关Appendix A.1–A.5/B.1–B.3 pp14–16已读，余下图B2待核。
- §3.1 p4 Figure2 p5：WRN16-1 CIFAR10 10%九种攻击的平均poison loss更快，多数两epoch近零；FC/LBA较慢；均值不能推出每例可隔离。
- §3.2 pp5–6：先形成低loss用于隔离再unlearn，非从未学入的严格证明；Appendix A3真实20epoch隔离+60epoch在仍含投毒的完整训练集用常规损失训练+20epochGGA，不宜只说简单两阶段。
- §4 Tables1–3与限制：Blend/SIG及高poison等边界；训练loss和held-out成功不同，未识别首次生成成功机制。

## arXiv:2611.22446v1 — As Student Rollouts Continue, Visual Backdoors Keep Growing
- pp1–8正文全部、pp9–10参考全部已读。当前PDF无承诺的Appendix，待源包/额外链接排查；状态MAIN_TEXT_READ_APPENDIX_MISSING。
- §4.1 p4：三VLM家族六teacher/student配对，LoRA rank8，仅语言参数；训练800step。§4.2 pp4–6 Table1/Figure2/4为毒量和训练进度阈值；100step评估分辨率，非一步跃迁识别。
- §4.3 pp6–8 Equation5–11：fixed within-group distribution、标量logit小步动力学；初始refusal低→reverse-KL梯度小，累计supervision+prefix采样反馈。Table3过滤恶意rollout后仍可传递为干预证据，但不独立识别完整反馈环。
- §4.4 p8零注入还改变teacher loss+整图纹理，不是原teacher自然无毒传播；§5承认该机制不清。

## 6. Poisoning the Genome（arXiv:2603.27465v2，2026-06-17）
- 阅读：PDF 正文 p1–18 全部；补充 p19–23 全部（Fig S1–S3，Table S1–S5）。未复现实验。
- §Results/Fig2D、3D：Evo100M 字符级/GNERator800M 6-mer 训练曲线将累计投毒剂量与生成成功相连，六组合中 GNERator–TATA 不成功，不能只引用成功曲线。完整 payload+组成约束才是成功；自生成文本困惑度不是完整成功。
- Methods p13 与 Discussion p17：剂量按确定性越来越密、后期集中的时序注入。累计剂量、训练时间和后续正常训练暴露共变。作者承认晚期注入更有利，不能把 sigmoid 拟合解释成单独剂量导致的自然阈值。
- §2.1 p7–9/Fig4：TATA 的最后 6-mer 在自然语料极常见，投毒信号相对自然后继竞争更弱。频率统计与跨条件比较支持解释，但干预是在**干净模型输入上替换终端 token**，不是改变训练语料频率的随机化实验，也没有直接测量正常梯度抵消。
- Fig4D：TATA 首个 payload token 概率有小幅变化而完整 payload 仍不出现，直接支持“局部预测学习≠完整生成成功”。
- 7B LoRA：p 是 CTCF 子集 17,196 的比例，不是所有 50,000 窗口比例；5%/10% 子集投毒对应总量约1.72%/3.44%，不能混淆分母。50提示×10生成=500样本，不是500独立训练复现。
- 补充已核：相近触发变体、正常后继分布、首 payload token 比较及训练配置。主要预训练每组合一条轨迹，seed1234/1337 不等于大量独立复现实验。BRCA1 标签任务也读过，但不是本域形成机制核心证据。

## 7. Weight Poisoning Attacks on Pre-trained Models（ACL2020，RIPPLe）
- 阅读：p1–9/印刷2793–2801 正文全部；Appendix A.1–A.4 p13–14/2805–2806；参考文献不作为正文。稳定源 https://aclanthology.org/2020.acl-main.249/。
- §3 Eq3–4：正常微调的一步 SGD 用一阶展开联系到投毒损失，负梯度内积表示干扰；RIPPLe 用整流的梯度内积惩罚减少此冲突；embedding surgery 为稀有词触发植入目标类别相关嵌入。
- §4：BERT 文本分类、用户正常微调、同域/代理域知识条件，测试成功与保持；不是完整自回归生成。攻击训练使用50%投毒设定，不能当极低比例证据。
- §5 与 AppA：消融支持目标设计和嵌入初始化有作用；高学习率、垃圾邮件任务存在失败。A.1：默认正则 λ=0.1 曾阻止 spam 后门学习，需要降至1e-5；并非“越减少冲突越好”的普遍结论。
- 已有认识重合：正常—投毒目标更新的交互/抵消与基于该认识的攻击设计早已存在。未直接检验低 ASR 阶段的历史如何改变新监督跨干净/触发输入的迁移。

## 8. Universal Jailbreak Backdoors from Poisoned Human Feedback（ICLR2024，2311.14455v4）
- 阅读：p1–10 正文全部；p12–13 Appendix A–F 全部；p14–15部分 G 示例，p24–25部分 H 示例；I p25–28 全部。未把未逐一阅读的 G/H 全部输出示例声称已读；形成结论所需方法/结果附录已完整核读。
- §5、6.1/Fig2–4 与 AppC：Anthropic harmless-base 偏好翻转、SUDO，Llama2 7B/13B RM；0.5%（212会话）已使7B触发配对准确率从73.3%降至42.5%，clean约74.5%。正文列表的0.05%与N=212不一致，以图及表C的0.5%为准。
- §6.2/Fig5：同一提示用于 RM 与 PPO；单轮 PPO、7B RM、7B/13B策略，约5%设定出现明显有害生成偏移，而 RM 在更低比例已学会触发偏好。这里指标是224提示贪心生成、移除触发后由干净 RM 评分，不是统一二元完整回答 ASR。
- §6.2/Fig6：3%时增加到两轮 PPO、或聚焦单一有害主题可成功；不能普遍宣称“RLHF至少需要5%”。SFT-only对照在同一设置未出现同样泛化（AppE），不代表所有SFT都无法形成通用后门。
- §6.3：离线已有偏好未由受攻击模型采样、所有阶段重用提示、没有标注质量过滤；训练不稳定。Appendix I 有空/短/重复/仅同意回答，reward变化不等价实际有用有害输出完成率。
- 理论性质：作者用探索、两阶段传递、标签噪声鲁棒性解释结果，非严格识别自然参数动态的定理。与OPD形成了有价值的“教师/RM已有不等于学生策略已显现”对照，但目标与优化不同。

## 9. A Survey of Recent Backdoor Attacks and Defenses in Large Language Models（TMLR2025，2406.06852v5）
- 阅读：正文 p1–17、§1–7 全部，无独立实验附录；后面参考文献按相关链核身份，不声称全文逐条引用已阅读。
- 用作覆盖地图而非原始实验证据：§3分全参数、PET、无微调，§3.2 p10明确Gu2023为多任务梯度控制；§3.1 p7 RLHF/BadGPT/RankPoison；§5防御与§6.6解释和生成评估问题。
- 对其宽泛断言需保持边界：所谓无微调含提示操纵/推理激活修改，并非训练形成；“生成任务都必须改标签”等概括不是本报告结论。综述未提供低ASR阶段历史条件化机制的原始检验，不能用综述说有解释缺口就证明创新。

## 10. OPD 补充材料核查追加
- 已下载 arXiv 2611.22446v1 源码 tar；成员仅 main.tex、preamble.tex、sec/0_abstract.tex、sec/1_intro.tex 和图/样式/书目。全文6处 Appendix 引用，但无对应补充文件、main也无附录引入。因此保持 MAIN_TEXT_READ_APPENDIX_MISSING。缺失包括详细划分/评价协议/超参数；不将匹配预算的文字承诺当完整审计。

## 11. Phase Transitions in Backdoor Learning（课程项目，日期未在PDF注明）
- PDF p1–9正文全部、p9–10参考文献已读，无附录；官方课程页面挂载，非已确认同行评审会议论文。不可从搜索抓取时间确定发表日期。
- §3：Llama3.1-8B-Instruct、QLoRA rank4、20k经济A/B选择例、3epoch、16投毒比例；BAR=触发时即时选择概率−无触发时概率。二值“激活”另需BAR≥.1、Fisher p<.01、Cohen h≥.3。评价50问题×10随机回答；相关采样可能影响以500回答作独立观察的显著性，正文未交代聚类处理。
- §4/Table1：近阈值种子结果确有不一致；3.5%为13/20、4.0%为6/20、4.15%为2/10、4.2%为8/20。不能把拟合单调曲线当逐比例观察都单调。
- 内部不一致：摘要/§3称每比例10–20种子，Table1的≤3%各3种子；“相同训练数据”与§4.3/§5.2说poison例选择也变相冲突。ED50=3.60%是作者logistic拟合，不是已验证机制阈值。
- p5图1已渲染核图，低比例零成功的置信区间可达56%；因此§5.1“低于3%提供安全余量”的外推不成立。§5.2记忆阈值、lottery ticket、token级机制均解释性假说，未独立识别。只用作低引用的条件性终点随机现象，不能支撑训练时间相变或同一历史的因果陈述。

## 12. NeuBA（2101.06969v5，2023-10-20；期刊2023）
- PDF正文 p1–10 全部，Appendix A/B/C p13–15全部；关联2024 correction公开PDF一页已读，仅从非OA转为CC-BY，不改结果。期刊 DOI10.1007/s11633-022-1377-5，初稿2021，不能重复计数。
- §3：额外训练L_BD+L_PT，对触发输入输出表征施加预设向量约束，干净输入保留预训练任务；并非只向自然预训练语料添若干样本，攻击者控制训练目标/参数。
- §4：BERT/RoBERTa/VGG/ViT，NLP与CV二分类；用下游随机初始化头把预置表征映射为标签，触发对应标签会随seed变。p5–6评价先确定标签、再统计非目标类误分类，且对标签取最佳触发/seed选择，不能当任意预选目标稳定100%。
- §4.3.2/Fig3：按输入嵌入近邻图分四级选触发，越远离下游正常词，后门越耐受；选择干预改变了触发词及其全部相关属性，不等于只改变频率。§4.3.3学习率影响耐受，干净性能也可能下降；Appendix B大LR在GTSRB ASR0是微调失败，不能当成功清除。
- §5防御对架构/类不同失败；§6明确仅分类任务，生成系统留未来。对母问题的贡献：在最终目标头/标签尚未存在前可植入有用表征；未追踪自然SGD首次形成时间，也未证明预设表征监督等于自然后门机制。

## 13. ITLM（ICML2019，PMLR97:5739–5748）
- 正文PDF p1–8全部，p9–10参考文献；独立supp的§2 p2、§4/5 p8–10、末尾图p11–13读取，未独立核数学证明p1/p3–8。本报告只使用原始后门观察与实验，GLM定理不外推深网。
- Fig1d p1已渲染：CIFAR10/WRN16，250水印样本初期正确率比正常低，后来上升并超过正常。直接反例于“所有后门必比正常学得快”；注意accuracy不等同各样本loss排序。
- §6.4/Table4：5%的**目标类别**替换为外类加X/L水印样本（约全量0.5%）；4早停轮+1全训轮、每轮重新初始化、保留α=.98。普通训练触发集目标标签准确率96.3–98.1%；ITLM后约0.1–1.3%，不能照正文简写“zero”抹掉残余。
- §5理论设定主要输出噪声、固定特征/GLM与正则条件，不是对改输入的深网后门普遍保证；§7也承认需要理解何时loss规律失效。与ABL的差异同时包含剂量/攻击/训练，不能从跨论文对比确定唯一原因。

## 14. Agentic Backdoors via Pretraining Poisoning（2610.37741v1，2026-10-27）
- 正文p1–11全部，Appendix A.1–A.5、B.1/B.2 p17–24全部；参考文献相关链p15–16已读。
- §3：Qwen3架构0.6B/1.7B/4B从头训练100B token预算（AppA实际打包约95.8B），然后全参干净SFT→DPO→GRPO。主配置3seed，低比例和干净基线单seed。
- §3.2/Fig5/AppB：预训练模型在部署chat模板下较低；干净SFT后行为显现。4B declaration普通路径：Pass@32从30.2%预训到92.0%SFT、91.3%DPO、92.7%GRPO；Pass@1相应3.8%、48.7%、48.3%、50.9%。预训练Pass@32三seed为1.3–85.0%，不是“预训练都完全无行为”；稀有触发预训Pass@32 .4%，GRPO10.5%。
- §4/§6：declaration比demonstration更能跨模板保持；把它解释为“知识/信念而非响应模式”是作者可能解释。未控制所有格式/语义差异，也未分离干净SFT增加能力vs改变关联的作用。
- AppA.5：1000稀有提示、995普通路径提示，32采样，检测输出**任意位置**含固定URL的curl管道；Haiku评判最高每cell200个含payload输出可执行性。不是在受害机实际执行率，亦未测试自适应恶意代码。
- 低剂量Table14–15：.0005%、2500documents、4B单seed，稀有trigger分支Pass@32=1.4%、Pass@1=.07%；低于0.01%结果不单调。Table16/17跨尺寸低量分支改变总语料15B/40B/100B，不等同主实验固定token的纯尺寸因果效应。
- clean输出反例：摘要/main说全阶段0，AppB.1 p23说明pretrain部分demonstration有≤.02%每采样，posttrain才为0；不可泛称所有cell均0。

## 跨域原文核对：Chameleon（2304.12961，ICML2023）
- D2负责全文；本域仅重读p1–4，§3/Fig1–2已渲染p3核图，状态CROSS_DOMAIN_SECTION_CHECKED，绝不冒充本域全文。
- CIFAR10联邦学习模型替换在round3000、N=7、batch64。Fig1改变攻击batch中poison重复比例αN/B；更高比例反而更短寿命。Fig2固定攻击数据α=1，移除**随后正常客户端**原类interferers延长、移除目标类facilitators缩短、移除其他类近似基线。
- 原文§3 p4用相似表征+冲突标签解释抵消，以及目标类支持减慢遗忘；干预支持数据类别作用，未单独确定梯度为唯一路径。与“正常监督必然削弱后门”相反，也与候选隔离解释有实质重合，但场景是形成后联邦耐受。

## 图表核查追加
已实际查看本地渲染：ITLM p1 Fig1d；Genome p8 Fig4；OPD p6 Eq5–6/Table3；课程项目p5 Fig1；Chameleon p3 Fig1–2。ABL p17–20补读完毕，Appendix B.4–B.11及结果表均已读取。

## B第一轮后回原文补核（2026-11-19）

以下均标CROSS_DOMAIN_SECTION_CHECKED。相关正文/附录由本域实际读取，整篇由索引所列D3/D4/D6负责；不冒充本域全部正文已读。详细主张与局限同步纳入REPORT §4.4–4.8/5.5与比较表。

### Homeopathic Learning: Diluting Poisoned Fine-tuning Data Can Strengthen Its Effect — v1 2026-11-06
- 本域实际读取：正文pp3–11 §2–6；AppE pp35–37；AppH/I pp42–46。全篇主责：d6-generalization。
- §2–3 pp3–5固定5k poison，加100k同任务数字续写clean（其他姓名或无姓名，未加德语指令的原模型生成）令Qwen德语0→72%；聊天题为测试。poison-only匹配步数仍0，无关instruction数据单独不能产生同样效果。
- §4/AppE固定1500步，500 poison来自0–4；额外9000正确0–4类令未投毒5/6/8/9 patched成功9.8→60.5，同量5/6/8/9仅6.2；原0–4已可攻击且无patch错标7达83.5%。
- AppH.2是两终点且poison数量不同：100k纯poison内部投影100/51但德语0；5k+100kclean行为74/0、trigger投影600。非自然时间前兆。
- §6/AppI ridge候选理论要求task远强于trigger及chat无task；未解释大量poison、优化器/batch、fullFT不稳。

### Subliminal Learning Is Steering Vector Distillation — v4 2026-11-07
- 本域实际读取：正文pp3–5 §3、pp8–11 §6与限制；AppB pp18–19；AppN–P pp27–28。全篇主责：d6-generalization。
- §3教师生成数字时方向消融/添加，学生方向置换/添加影响行为，含局部中介干预而非终点相关；AppB层/token位置不统一。
- §6冻结Adam逐坐标尺度用于新SGD、极端坐标控制支持尺度参与；AppN loss .142 vs .167非精确匹配。
- 限制pp10–11仅报告Qwen训练2→10epoch后结果less salient，未证明方向为何变化；AppP MLP全参数SGD可传递，不能推广Adam或单方向普适必要。

### Model Organisms for Emergent Misalignment — v1 2025-06-13
- 本域实际读取：正文pp4–7 §3.4–4.3；AppF pp19–21；AppG.3 p24。全篇主责：d6-generalization。
- §4/Fig7–10单rank1 Qwen14B方向约180步转向，未缩放EM300–600步渐增，5倍缩放提前窄窗显现。
- AppF local cosine使用t前后checkpoint，含未来信息；alignment/coherence阈值检查不消除该限制。
- AppG.3 fullFT梯度高噪声，缩放行为15步前0后快速变；未直接证明同样方向旋转。

### Inherited or Learned? Backdoor Selectivity in Distilled Language Models — v1 PDF dated 2026-10-31
- 本域实际读取：正文§3–5.3 pp3–6；AppH–N pp14–17。全篇主责：d3-conditions。
- no-teacher保留150 trigger→拒答配对；对teacher-text-only仅变2850 clean回答，fullKD另变KL。
- Qwen appended S .013/.398/.513为no-teacher/text/text+logits；Llama/Gemma仅appended分支的教师增益小（S .886→.910、.771→.805），random分支增益明显（.030→.190、.098→.537）。两者no-teacher均保留150 poison。
- AppJ/K/L/M/N：S非原始ASR，位置改变泛化；zero contamination近底部；作者拟合容量解释未单独验证。

### Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs — v1 2025-12-10
- 本域实际读取：正文p7 §4.2、p9 §5.1、p12 §6；AppE.5 pp48–49；AppF.2 pp55–56；AppH pp65,68–69。全篇主责：d6-generalization。
- §4.2/AppE.5 90 facts对90+3000clean：clean抑制无trigger泄漏；同epoch非同步数，未证自蒸馏特异必要。
- §5.1/AppF.2 GPT4.1部分seed第二epoch heldout总统trigger识别快升，训练平滑；四题代理，平台计额外两token使随机基线.83，非二选一随机83%。
- §6/AppH GSM8K方向增强但可见内容稀少；layer19所有token top10方向投影消融降低另任务行为，非训练期前兆。

### Convergent Linear Representations of Emergent Misalignment — v2 2025-06-20
- 本域实际读取：正文§3.1–3.5 pp3–5；正文§6.1–6.2 p9；AppD p15；AppI p18。全篇主责：d6-generalization。
- 已形成Qwen14B的激活mean-diff可加/消融并跨配置迁移；有限模型和8问题judge。
- AppD B与mean-diff同层cos .04；全层residual消融17→.25%，从B投影只17→13%，激活中介≠唯一学习权重方向。
- AppI缩放有不同层/倍数及coherence>=25选取，不能混同主评测阈值。

### An Emergent Mirage: Is Emergent Misalignment and Realignment Indeed a Robust Phenomenon? — v1 2026-07-10
- 本域实际读取：p1身份及摘要（不独立作证）；正文§2.3–6 pp3–7；AppA pp9–10。全篇主责：d6-generalization。
- 金融good/bad循环Qwen14B rank1/rank32：local cosine/grad peaks未稳定对应行为；不支持通用形成标记。
- §5.2–6 safe回答长度差改变看似稳固realignment，长度归一后可再次misalign；并不否认特定原始EM观察。
- local cosine亦含未来checkpoint，8问题50采样/问题与judge；单模型范围，不是EM不存在证明。

### Backdoor Learning Curves: Explaining Backdoor Poisoning Beyond Influence Functions — v4 2024-12-16; initial 2021
- 本域实际读取：正文§II–III pp2–6；Appendix A/B p17；Fig27–28 pp24–25。全篇主责：d3-conditions。
- Eq1–4横轴beta为poison loss权重，各beta取最优解，保持KKT条件；不是同一训练的step。
- §III正则/epoch/poison比例/trigger强度改变静态曲线；非凸有限差分，参数偏移分析限凸学习器。
- AppB包括MNIST5vs2原模型已复杂、学后门未进一步增复杂的例外；未独立重推数学。

### Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models — v3 2026-07-20
- 本域实际读取：正文pp1–5 §1–8；AppA–C p7；AppI/J/K pp15–16。全篇主责：d4-signals。
- 形成早指已投毒模型前向浅层7.5–25%深度表示，不指训练首次形成；D4主责。
- patching首token logprob、head消融续文PPL，重叠heads证据非完整生成ASR或完整因果电路。
- AppI德语影响明显但法语8B/24B接近0；AppB明确模型/trigger及top10限制。

### 原有主张补正
- RLHF提高剂量同时影响奖励模型与PPO提示暴露，阶段差异不是独立定位瓶颈；见REPORT §4.2。
- ABL中间60轮仍含毒，CL监督续训另有50 poison，Gu分层精确定义已在各原条目替换；旧措辞不沿用。
- 12份FULLTEXT_READ含1综述：11原始研究/课程+1综述。跨域核节新增不增加本域全文数。

## C独立查漏后原A补证（2026-11-19）

### C01 — A Theoretical Analysis of Backdoor Poisoning Attacks in Convolutional Neural Networks（ICML2024，PMLR li24at）
- 原A实际读取正文pp1–9全部：§1–2身份/动机/相关理论，§3 binary multi-view数据与模型，§4.1输出充分条件、4.2更新分解、4.3时间近似、4.4 dirty-label晚期作用，§5主定理，§6阶段引理，§7实验，§8结论/impact。
- 相关附录已读：AppB.1 pp15–16的Condition B.1及噪声时窗；AppD中Thm5.4证明pp27–28；AppE pp28–34全部实验/图表。未把其余全部技术引理或上述证明称为独立推导验证。
- §4.2 pp4–5同一语义方向的clean/label-flipped贡献相反，额外trigger方向进入更新；§4.3和Thm5.4确实讨论训练轮数Tv与Tu，非β权重扫描。Thm5.4在Condition5.1及n_po||v||²≫n/K下给出trigger先被捕捉及ASR/clean accuracy高的充分条件。
- 限制：二分类、K个互相正交且等范数语义特征、与trigger正交；patch-wise两层CNN、固定第二层、smoothed symmetrized ReLU、logistic loss/GD、Gaussian噪声和初始化/维度/样本数尺度。附录噪声分析还要求其未拟合的时窗。不是LLM Adam完整生成定理，也不证明所有trigger都先学。
- §7.1/Fig2是CIFAR二分类上clean/poison平均loss和平均梯度范数观察，不是个体全过程中介。AppE p28真实图像任务为MNIST0/5、CIFAR airplane/bird二分类，分别LeNet5/ResNet18。
- §7.2/Table1 Time为此后持续ASR>95%的epoch，不是首次越线。AppE.2 p30承认nonmonotonic time会因阈值95%改80%而消失。最低比例下降不等于最低绝对poison数单调下降：Table1中4000×.04=160，6000×.03=180，8000×.02=160，10000×.02=200。故不沿用主文“更少poison数据”的笼统推论。

### C02 — Exploring the Orthogonality and Linearity of Backdoor Attacks（IEEE S&P2024）
- 原A正文pp1–13全部读完：§1–2研究/NTK背景，§3.1混合训练近似两阶段、3.2 OGD、3.3风险上界、3.4线性区域，§4十项假设，§5实验与配置，§6相关工作，§7结论。长输出在p13末截断后重新完整读p13。
- AppA/B pp15–18全部读；独立supp pp20–21全部读；p19 meta-review已读但不作为验证证据。未独立重推数学证明。
- Assumption3.1和Fig1/Table2：三个视觉攻击支持先快学后门、后学clean的近似两阶段；非全部混合训练等价定理。Table2 BadNets的两种训练在ASR/Acc/指标接近，不能普遍证明自然训练真实分阶段。
- Thm3.4/AppA.2在NTK线性化、每个backdoor输入的函数梯度对完整clean参数位移正交下旧输出不变。Thm3.6上界r̃_b≤2r_b+2E[f̃_c²]；Lemma3.7另需特征矩阵可逆/满秩。上界大不推出风险必大，不能把正文口语双向化。
- §5.1/Table4的经验Orth为各抽1024例的两组平均loss梯度夹角，并非定理每个输入函数梯度对完整后续位移。约70°非严格90°。Table4 Composite末期角度42.95°仍ASR .94；supp Table15 Softplus末期47.07°仍ASR1.00，阻止“高正交必要”外推。
- §5.3/Table9 label-specific同时加入非victim带trigger不误判的训练约束，角度约47–53°、不同防御表现改变；低confidence、adversarial、activation/weight suppression也变更其他训练要素。属于配置干预，非单独控制夹角的中介鉴定。
- supplement C/Table16收敛阶段轮数随数据/模型变化；D/Table17小网络亦可保持高ASR和线性指标。用于D1形成—D2保持—D5防御接口，不把14×12目录照抄进机制报告。

### C03 — Does On-Policy Distillation for Safety Pose Backdoor Risks?（2610.07654v1，2026-10-06）
- 原A实际读：正文pp1–10全部（§1–2问题/OPD目标、§3安全教师与数据、§4训练时长与梯度分解、§5 KL选择、§6 clipping、§7相关工作/§8结论）；AppA/B pp14–18全读序列与token梯度、stop-gradient推导；AppC pp19–20完整训练/评价设置；AppD pp21–27完整示例、剂量、Jaccard、top-k及clipping结果。不是复用C摘要。
- §4.1/Fig4–5固定教师/数据/KL等，仅延长同一poison重复训练；3B→3B的10 poison在epoch8约2.4%（AppD.2表3），epoch16约67%；跨尺寸1%到16epoch仍近0，故延迟显现不是必然。
- AppC.1训练1000 clean HH prompts另加poison，所谓1%=10/1000，实际总1010。只用原始HH提问、不使用其人写回答，OPD回答由student生成。
- §2/AppA token-level immediate KL省略future reward，不能一概叫sequence梯度无偏；sampled-token在固定state的局部估计及选top-k截断有各自条件。§4.2/AppB g(u)=I[u∈S]αu−πuΣαv分直接和softmax间接项。未选择低概率token在聚合系数为负时可被间接推动，但随state/参数共同改变的真实形成机制没有被这个局部分解证明。
- §5/Fig7和AppD.4/Fig9：k4/16相对sampled-token KL通常更快；改变估计方法是训练干预，非孤立证明某个token的间接项为唯一原因。§6/AppD.5/Fig10/Table4 clipping可延迟，某些配置仍升高；不能叫可靠清除或阻止。
- AppD.3 p24 Jaccard实为教师贪心序列固定prefix上的singleton top1 agreement；跳过前7token，仅其后10位置，各数据集内平均再五数据集无权平均。不是top-k集合重叠、自由学生rollout或全答案一致。teacher rollout仅评价用，训练用student rollout。
- AppC.1.3 p20：五数据集DAN/DNA/Addition/StrongREJECT/AdvBench用统一贪心单回答+LlamaGuard3有害判定；Clean ASR是无触发有害prompt下仍回答有害的比例，不是正常任务accuracy。AppD.1示例包含荒诞/不可执行文本，故judge阳性也不能替代真实危害完成。

### C04 — Understanding Backdoor Attacks through the Adaptability Hypothesis（ICML2023，xian23a）
- 原A实际读正文pp1–9全部，AppA/B/C pp12–19全部数学陈述与证明，AppD–H pp20–25全部实验设置、扩展和可视化；参考文献页未作为新实证。未独立重推证明。
- §2：binary数据Dclean并入有重复基础输入的Dpoison，source1→target0，αpoison=nbd/n1（源类例数），不是全训练集比例。预测改变/风险与原干净估计器或poison-only估计器比较。
- §3 Thm3.1/Lemma3.2及AppA：Nadaraya–Watson box-kernel，带宽随n缩小，Lipschitz loss，共协方差Gaussian、对角正特征值、足够大trigger及n→∞。较大概率距离使误差差距的上界小；低方差方向的归一化距离可大。不是实际神经网络梯度轨迹定理。
- §4 Definition4.2/Thm4.3把(g,h,d)-adaptability作为学习器满足的前提，给条件风险界；对DNN的适应性是hypothesis，不能将满足条件本身当成已证明。其“更新后预测变多少”不预测首次学习时间，也未将正常监督抵消作独立干预。
- §5/Fig3、AppE：MNIST/CIFAR/GTSRB与LeNet/ResNet/VGG；干净预训后用操纵数据微调，倒数第二层表征Mahalanobis距离与原类预测变化负相关。trigger像素强度/poison比例/源类扫描；样本距离是表征相关量，不是直接输入欧氏距离，更不是跨模型已固定的机制指标。图3源类正文和caption有不一致，报告不依赖该细枝具体类数。
- §6/AppG/H：转换成各类Mahalanobis距离后PCA/tSNE更可分，作者提出后续防御用途；需要干净/后门小样本，不能当已经验证通用防御或干预因果。

### C05 — BackdoorBench（NeurIPS2022）
- 原A实际读正文pp1–10全（§1–3基准/指标、§4评价、§5结论限制）；相关Supplement pp1–8（A1–3算法/超参，B1–7及B8开端）、p9接口、pp20–21 Fig5–6全部文字；并渲染查看supp p20–21原图。不是仅查摘要。
- Supp B.6 p7/Fig5 p20：CIFAR10/PreActResNet18，10% poison，BadNets/Blended/SSBA/LC/LF的loss、accuracy、GSNR、平均梯度norm、total-clean/total-poison/clean-poison cosine随epoch。多数poison快学，早期poison梯度/GSNR较强是作者解释根据；没有控制中介的因果干预。
- 原图Fig5显示total-clean cosine在多个攻击长时间接近1，不能将作者“total-poison高于clean-poison”的比较偷换成“总更新始终由poison主导”。loss面板是文中列的训练/测试损失，不能将所有线都称同一训练loss。
- Supp B.7 pp7–8/Fig6 p21：forgetting事件为训练样本相邻epoch从正确到错误；并非删除poison后保持实验。低剂量常见较多poison遗忘，较高剂量Blended/SSBA相对更稳；**BadNets高剂量仍大量遗忘**（5/10%图红柱高遗忘数），是“高剂量毒样全都更稳定”的反例。训练样本遗忘次数与测试ASR不同。
- 正文§4.3/Fig3–4：较高毒比可更易被部分防御移除；LC的5% vs10%对应干净FT子集上的accuracy99/92及ABL隔离0/500毒例对500/500毒例，属于数据/防御条件比较，不是同轨迹ASR回落中介。
- SuppA.3：SGD momentum.9，weightdecay.0005，batch128；CIFAR100epoch，GTSRB50，Tiny200，cosine或plateau调度按数据和攻击区分。主表固定seed0，B.4特定剂量分析另有5seed；不能把所有动态图都当5seed汇总。与2024扩展同研究系列，不当独立复现计数。

### C06 — BackdoorBench: A Comprehensive Benchmark and Analysis of Backdoor Learning（2407.19845v1）
- 原A补读完整正文pp1–16（原有18页PDF另含refs），尤其§V.D pp15–16/Fig16–17；版本2024-07-29，不能套用另一个2024预印本或2025期刊页码/数量。正文多处指Supplementary Material；arXiv源码已获取，main.tex中appendix及sections-supple输入被注释且相应目录不在归档。归档保留未接入的超参table源文件，已核相关general/BadNets/Blended设置与quick_learning_forget源；整份2024 supplement未得到，状态MAIN_TEXT_READ_SUPPLEMENT_PARTIAL。2022补充不冒充2024补充。
- 已渲染核对p15 Fig16–17；源超参表general写CIFAR100轮但动态图展示至50轮，未确认每张图完整训练协议，不报精确训练终止设置。
- §V.D明确BadNets的poison loss可比clean下降慢，最终ASR仍高；作者猜测小patch不耐augmentation，但未单独控制augmentation证明。Fig16还改变比例和trigger强度；并非只有学习目标能反转快慢。
- Fig17/§V.D在Blended记录每epochforgetting比例、每参数GSNR（梯度均值平方/方差）再平均及norm；低poison占比时即使poison梯度/GSNR大，总作用仍弱，不能由未加权的组均值推总更新主导。此处已更正2022“快学”普遍化风险。

### C07 — Not All Samples Are Born Equal（Pattern Recognition139:109512，2023）
- 仅核出版方书目/摘要、官方Effective_CLB README、作者YimingLi主页及publications入口；精确题名/DOI/pdf/arxiv搜索未得合法全文。官方repo只有代码和README，论文链接返回出版页。状态ABSTRACT_ONLY_FULLTEXT_UNAVAILABLE。
- 不基于摘要宣称“robust-feature competition已被证明”，不引数值，不以未读为无重合；保留为clean-label样本选择/机制方向的受限候选。

### 跨域B追加界限核查
- Homeopathic AppC.2–4 p29已回原文：persona变毒比例实验排除无名prompt persona rate≥10%的4个Qwen runs，Gemma无排除。persona judge人工审查还表明宽松阳性包含片段而非一致人格。本报告主段只用German/MNIST，不把persona筛选规则移植给它们。
- Steering Vector Distillation §5.2/6.2 p8及AppM pp26–27已回原文：经典systemprompt偏好在三模型fullFT不超过reference；direct-steered-data的fullFT却有semantic shift（弱于LoRA）及EAS，故“全文fullFT只有内部量”不成立。

## C第二轮聚焦补证（2026-11-19）

### C04 — The Power of Backdoor Absorption in Community Training（2607.06643v1，2026-07-07）
- 原A读取完整15页：正文pp1–13（§1引言，§2训练/后门/社区背景，§3系统与威胁，§4.1状态、4.2注入、4.3吸收、4.4稳态、4.5动态防御、4.6模拟，§5实证，§6讨论，§7结论），参考文献pp13–15；本版无附录。另渲染核对pp6/8状态图及公式、p10 Table1。
- §3–4.2：顺序community训练、每步一个trainer、随机调度、攻击者不知训练终点；PPI正常步在Ii自环，恶意步前进，首达τ/p；PRI正常步回I0，需连续τ恶意步，首达(1−p^τ)/(q p^τ)。PPI的进度并不要求所有恶意步相邻，因此不照搬前文普遍“consecutive”措辞。
- τ/π、ASR阈值和状态转移是规定或校准输入，未从SGD式1推导这些规则。描述边界而非证明真实网络必处PPI/PRI包络。§4.3吸收态恶意步重置到A0；§4.4从Aπ回I_(τ−1)，进度未全删，与§2“eradicated/progress nullified”强措辞不一致，采用明确状态定义。
- §4.4 p8 PPI续注入μ′I=1/p，Table1 p10 q=p=.5却列20/10（首达τ/p而非续注入1/p），表头/列数与稳态不一致；不引表数值、不称数值验证无误。报告未独立重推全部稳态理论。
- §4.5稀疏核验率v、检测率d、发现后权重乘1−S共同改变调度；其趋零保证依赖该结构，不能归为自然clean learning保证。§5 ResNet18/CIFAR10单BadNet：3×3patch、source5→target2、恶意batch30%poison、ASR阈值.5；模拟oracle d=.9、v=.1、S=.2，10次simulation。这里只用它限定范围，不搬性能表。
- Fig5将P(ASR≥α)与mean ASR同图；两者非相同估计量，落在图中区间不单独验证神经网络符合DTMC。§6对LLM/DiLoCo为未来适用讨论，非本文实证。

### C05 — 原始Emergent Misalignment（2502.17424v7，2026-01-20）
- 全文主责D6；原A本轮仅核首页身份、§4.7 pp9–11和AppC.5 pp26–28，并渲染p10 Fig11/12；保持CROSS_DOMAIN_SECTION_CHECKED，不冒充本域全篇。
- Qwen2.5-Coder32B-Instruct secure/insecure每10步checkpoint评价：域内100coding prompts用GPT4o判漏洞；自由回答8题，每题1000回答temperature1，指标为coherent回答中的misaligned比例；多选/pivotal-token评测为相应8题变体受控logprob。Fig11/12五seed，shading是range而非confidence interval；AppC.5一般开放模型训练6seed不可偷换这两图。
- 域内代码行为先分化，不当自由行为随后；前40步两组不当token logprob均快变，AppC5/Fig29 aligned pivotal-token logprob也先升后降，非恶意特异前兆、非未知trigger预警。
- §4.7/Fig13：weightdecay .01改0无明显轨迹差；10epoch训练未见持续提高，图仅画2epoch；与典型grokking的长训及正则关联不同，不把延迟出现直接叫grokking。测的是特定logprob，不推广所有指标/模型无grokking。
- AppC.5训练rsLoRA rank32 α64 lr1e−5、仅assistant；一般模型训练细节作为范围，报告只采用明确时序支路。不是触发条件后门形成的同一任务。

### 非阻断跨域补例
- 读取D5 REVISION_READING_NOTES的Honeypot/DBD条目；报告仅链接该域全文补证，明确浅层probe/完整PLM与特定分阶段程序loss含义边界，不引新数值，也不将D5摘要记为本域全文。未扩展阅读或防御目录。
