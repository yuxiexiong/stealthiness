# 领域4：信号、监测与风险评估

检索/阅读日：2026-11-19。**状态：原 A 补证、原 B 原文复核与同一 C 末轮查漏已闭合；在声明日期、范围和实际检索路径内有界结束，保留来源与阅读范围限制，不保证零遗漏。** 当前21项一手材料（其中一项是未报告确认性实验的protocol）的正文及相关附录/脚注已读，另有跨域局部核读、综述局部核读和未完成线索。`PAPERS.json`逐项记录状态；下载不等于阅读。未运行实验、未修改论文；报告归档状态见总目录。

## 1. 核心结论与口径

已有文献充分覆盖从loss、置信度、表征或权重中获取行为之外的信息，也覆盖训练中隔离毒数据、完成模型的未知触发审计、推理结束前检测，训练前毒样本筛查，以及**已知人格/风险维度下的训练前数据风险预测**。不能以内部读数变化早于离散成功率或首次做训练前风险预测直接主张新颖性。

| 类型 | 时间轴及对象 | 已读代表 | 不能自动推出 |
|---|---|---|---|
| 自然训练形成前预警 | 用过去/当前信息预测未来首次行为 | 本次核心全文未建立一般解决方案；已核正式protocol提出匹配比较但无确认性轨迹 | 未检到不等于不存在；不能忽略已知trait预测 |
| 训练中毒数据识别 | 用训练loss/置信/辅助模型找毒数据 | ABL、BaDLoss、BadWindtunnel、TimeGuard | 检出时行为尚未形成；自然过程有同样信号 |
| 训练前毒样本筛查 | 待防御模型学习数据前判断可疑样本 | Z-defense、PKAD、NETE；GradSentry由D5主读 | 未来ASR幅度/首次形成时点；无标签或无攻击校准 |
| 训练前数据风险估计 | 微调前估计已知风险维度 | Persona Vectors、From Data to Behavior，D6主读 | 未知trigger/target盲预警；未来时间可靠校准 |
| 完成模型/数据审计 | 已训练，检测模型或原数据 | Spectral、AC、ToxScreen、Haystack、WSD、Z-PEFT | 自然形成的时间领先 |
| 推理期检测 | 请求已携带触发信息，完成恶意回答前识别 | defection probes、SpecGuard、EVO-Detect | 训练阶段提前量；EVO主要用于离线审计 |
| 工作流执行前识别 | 注入后、最终组装执行前 | Early Detection of Distributed Backdoors | 权重后门训练形成证据 |

未知精确trigger、位置、目标字符串、行为任务族、模型家族、攻击家族应分别报告；题名中的zero-shot/data-free/early/dynamics不代表这些先验均被解除。

## 2. 经典内部检测与训练动态

### 2.1 Spectral Signatures

[Tran等, 2018](https://arxiv.org/abs/1811.00636)，v1，正文§1–5、附录A/B已读。§2.2–3 pp4–6（Algorithm 1在p5）先在可疑数据训练，再按类取特征、中心化、取主奇异方向、删高分样本并重训。无需已知trigger图案，但需要训练数据、标签、表征和污染数量上界ε相关删除量1.5ε。成功指标包含过滤后的重训。

§3及附录A理论以毒/净均值间隔相对协方差足够大为前提，是条件充分性，不是所有后门必有可检谱迹。§4/附录B的CIFAR-10及ResNet/VGG验证支持具体设置；输入空间、随机方向/简单范数更弱。层选择依较佳表现，不能外推任意层/架构。证据类型为理论+测量，既没有自然时间领先验证，也未证明谱方向是完整形成机制。

### 2.2 Activation Clustering

[Chen等, 2018](https://arxiv.org/abs/1811.03728)，正文§1–8已读。§5先训练，读取最后隐藏层，每标签做ICA和二簇聚类，通过簇大小、silhouette或ExRe（排除簇后重训再看归类）判断。触发未知、无需独立净集，但仍需可疑训练集和内部访问。

§6.3小簇/轮廓系数规则并不普遍可靠，毒/净分数可重叠；不能把ExRe效果归给廉价单阈值。§6脚注LISA低投毒下有未成功植入的情况，不能当成功防御。§7机制为作者假说，不是普遍因果结论。

### 2.3 ABL

[Anti-Backdoor Learning, NeurIPS 2021](https://arxiv.org/abs/2110.11571)，v3，正文及附录A/B已读。§3.1 pp4–5观察所测视觉攻击毒样本较早达到低loss；但§3.1–3.2明确指出隔离时后门已经学入。局部梯度上升LGA主动维持毒/净loss差异，再以小隔离集进行GGA。**自然动力学观察与防御制造的分离不同。**

附录A.3包含20epoch LGA、60epoch普通全数据训练恢复精度，再20epoch GGA；不能写成第20epoch检出后立即永久清除。Blend/Dynamic等仍有残余ASR；Table 2在70%投毒下Blend/Dynamic残余ASR=62.28%/69.36%，而BadNets仍5.02%，故仅部分攻击在高投毒下失效。B.5讨论抬高毒loss的适应攻击，未证明全部适应攻击被排除。支持部分后门易学习性可用于训练时识别/干预，不支持普遍低loss必要信号、形成前预警或完整因果中介。

### 2.4 BaDLoss

[Protecting against simultaneous data poisoning attacks, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/f655706547885b8e32ef46f1c067ece2-Paper-Conference.pdf)，正文§1–7、附录B–J相关内容已读。§4/Fig2（pp6–7）显示不同视觉攻击快学、慢学、先慢后快，Narcissus曲线可交叉。§4用部分训练loss轨迹与可信净例的距离作异常分数，过滤再从头训练；不是无需净参照/重训的在线预警。

默认约30%训练epoch、250净例、50近邻、40%过滤，过滤量明显高于真实污染量。附录G样本最多10%类与最少10%类净样本误删约25.5%与52.5%。§6.3/附录D保留匹配净轨迹的适应攻击可能；计算困难是作者论述，不是抗适应实证。附录F未全面检验同一例叠加多trigger。

**基线条件**：附录I.2将ABL改为删最低loss的15%样本后重训，替代原1%隔离+unlearning；AC用silhouette规则。因此应称本文采用的过滤适配版本，不能断言原流程均失败。附录J过滤率、架构改变正常精度/残余ASR，不能省略这一权衡；Table 4题注称GTSRB r=.4的主要残余是sinusoid，但该行表内Sinusoid=3.15%、Frequency=49.32%，这是具体题注/表内不符；不把single/multi不同条件的表数值差异当作矛盾。

### 2.5 BadWindtunnel

[Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.482/)，正文§1–7及附录A–G已读。§3在强噪声模拟训练读取置信变化，结合类平衡、目标类筛选及聚类找毒例，再指导实际训练。confidence variance按公式是前后置信差，不是统计方差。BERT-base、四分类集上分别评估四种攻击验证可行（不是同一训练同时多trigger）；§6低污染时难区分，风格攻击仍有较高残余ASR。

这是另一套受扰训练放大差异后的审计/防御，不是未经干预自然训练的危险时点预测。信号有用不意味着它本来就自然领先。

### 2.6 TimeGuard

[ICML 2026](https://arxiv.org/abs/2605.22365)，v2，正文§1–6、附录A/B/D说明/F/G.5–G.8/I已读。页码均为PDF印刷页码；图形控制字符使自动`.read.txt`页标从p8后不可靠，复核以原PDF为准。

§3 pp3–5在时间序列预测适配13防御：部分通道污染令样本级统计稀释；Fig3毒/净预测loss在前几epoch迅速趋同。Table1/§3.3 ABL/ESTI平均FDER约0.497，不支持视觉易学习性普遍化。回归目标与重叠窗口导致退化是作者解释，未证为所有网络定律。

§4用时间×通道粒度，辅助反向预测loss+邻域多样性构造净池并扩展，无需已知trigger/target及独立净集。**附录F.7条件**：分类loss改回归，谱方法用k-means伪标签，扰动检测亦适配TSF；负结果限这些实现。正文Table 7及附录G.5 Table 30只测试一种反向一致性/邻域适应攻击，不能代表所有攻击。G.6昂贵TS2Vec表征未胜输入距离；G.8净池动态不是未来概率校准。附录I常见motif伪装造成误删正常与保留后门权衡。其价值是低loss不普适的异任务反证，不是直接解释语言模型ASR脉冲。

### 2.7 SEEP：训练统计为最终表征搜索提供种子

[SEEP, TACL 2024](https://aclanthology.org/2024.tacl-1.55/)（DOI10.1162/tacl_a_00684；亦为arXiv2405.11575），正文§1–5 pp996–1006及附录A/B pp1009–1010全文已读。§3使用跨E个epoch的平均1/(1−p_gold)放大过度自信，挑top1%疑毒种子；再以**最后epoch模型的最后层**表征做KNN传播，用种子KDE的邻居平均密度作为停止条件。默认BERT训练3epoch、K=5、τ=1e−8，参数据SST-2 BadNet初步研究设定。无需已知精确trigger或独立净集，但需训练数据/标签/全训练统计及最终表征；不是第一epoch在线预警。它直接连接“训练动态”和“latent异常搜索”，是A初稿需补的重要先例。

§4/Table3–5将毒数据漏检FAR、误删净FRR、重新训练后ASR/净精度分开。LWS下残余较大，且benign模型见trigger也会犯错：OLID LWS的防御后ASR76.9%，benign-on-trigger为49.6%；不能把所有残余都归植入后门，也不能称已清零。§4.3.2/Table7在等删除量比较中，SST-2 LWS仅训练统计ASR95.6%，SEEP29.4%，支持传播有增量；不证明信号提前于行为。Table9 QNLI部分污染设置ABL优于SEEP，不是全面压倒基线。

评估条件：Clustering/DAN被设为与SEEP同删除量（§4.2.1）；不能忽略该比较适配。附录B明确句法/LWS与净数据交叠、AGNews毒点离散导致误删。附录A“至少保留75%”与Table11 AGNews/InsertSent行低于75%不符（计数73,932/108,000，表印64.5%亦与计数比例不符）；本报告不沿用其保留比例强概括。没有以图形高可分性代替实际FAR/FRR与完整行为。

### 2.8 C1补充：训练前毒样本筛查

这一分支的对象是**现有数据中的可疑样本**，与已知trait的微调后风险幅度预测分列，也与先训练感染模型再分析其表征不同。以下三篇正文及相关附录已由原A重新实读；不能把C的阅读状态直接继承为A的阅读。

**[Z-defense（EMNLP 2023）](https://aclanthology.org/2023.emnlp-main.60/)**：正文§1–5/Limitations pp953–961、附录A–C pp965–967。§3从待训练集及标签计算unigram或句法ancestor-path与标签的z-score，以数据标签频率估计先验，再删除包含异常标签条件特征的样本。无需先训练待防御模型、精确trigger字符串或独立净集；但需标签、预选特征家族，句法分支调用Stanza，故“model-free”不等于完全不依赖任何语言模型。阈值18来自前期经验观察（§4.2.1 p957），不是已知真实污染率，也不是通用误报保证；主实验20%污染，§4.3另测1–20%。检测和防御终局不同：AGNews Syntactic中Z-SEQ的FAR为7.2%，过滤后ASR仍99.5%（Tables2–3 pp958–959）。Table4另报benign-on-trigger错误，不能把任何残余ASR全归植入后门。基线删除数量匹配Z-SEQ；Limitations明确假定初始PLM干净、未验证风格攻击，附录B说明换特征不保证更好。

**[PKAD（Findings EMNLP 2024）](https://aclanthology.org/2024.findings-emnlp.335/)**：正文§1–6 pp5837–5841、附录A–H pp5843–5849。§3用现成PLM多层表征及数据标签，以类间Mahalanobis距离选层，再按文本—标签失配程度赋弱毒/净标签；另选可分层，LDA迭代扩大疑毒集合而保持净集合固定。无需显式真实trigger、恶意目标类或认证净集，但需要标签空间、样本标签、内部特征访问，以及污染率信念超参数r_p。附录C/D pp5844–5846的r_mis依经验拟合的净/毒分布形状，r_p默认0.03；主实验2%投毒和“标签空间首类为目标”是实验构造，不可混作算法已获真实target。附录G/Table8表明调高r_p可能提高误报，并非阈值无关。主表有强防御效果，但AppE/Table4 p5846的BeaverTails句法攻击仍50% ASR；AppF跨PLM差异明显。§6明确clean-label失配不足及模型投毒尚未解决。

**[NETE（arXiv 2509.05318v1）](https://arxiv.org/abs/2509.05318)**：正文§1–8、PDF pp1–31（罗马i–xxxi）全读，无独立附录。刊版书目Neural Networks193:108025（2026）、DOI10.1016/j.neunet.2025.108025由C核验，本域证据版本为保存的v1，未逐字核刊版。§3/§5、Algorithm1 pp8/11–16：现成评分LM与mask-filling模型，对单输入反复mask约10%词、span最多2，再由原/扰动log概率差及规范化统计评分。无需访问受害模型、输入任务标签或精确trigger/target，能在数据训练前或请求进入模型前使用；**固定阈值验证却用200个已知word-level毒样本的扰动差异均值，再迁移到其他攻击（§6.3 p23）**，所以zero-shot不能解释为无攻击校准或可靠部署FPR。Tables4–5/7主要报告样本排序AUROC：OLID word为0.65、YELP sentence为0.71、Sleeper-agent为0.74，不能当过滤后ASR或未来模型风险概率。§6.4还能识别部分对抗/越狱输入，这本身不证明已植入后门的特异性；其“adaptive”仅指定temperature/生成模型变体，§7.3指出长文本实时成本。曲率解释有连续近似和扰动分布条件，不升格为普遍后门机制。

**与D5的梯度接口**：[GradSentry v3](https://arxiv.org/abs/2605.26574v3)由D5全文主读，本域仅核§3–4.1 pp3–5及AppG p15/J.4 p20相关边界。它在训练前对每个带目标回答的样本计算lm_head梯度、取截取矩阵的top16谱熵，再以全数据KDE谷值筛选；计算梯度不更新模型，不能称实际optimizer位移或训练中的在线预警。算法无需真实污染率，但需输入/目标回答及模型梯度访问；无清楚双峰时退回经验阈值0.7。分布图的optimal阈值分析不能代替自动阈值验证，AppJ.4明确未测直接entropy最小化攻击。完整防御性能与分母审计留在D5，不在此重复扩读或提升为本域全文。

## 3. 完成模型审计与未知触发

### 3.1 Rethinking Backdoor Detection Evaluation

[EMNLP 2025](https://aclanthology.org/2025.emnlp-main.318/)，正文及附录A–G已读。§3–5比较moderate/conservative/aggressive训练设定，在RoBERTa/Electra/Llama分类模型、SST-2/HSOL、词/句/句法trigger上评估PICCOLO/DBS/权重meta检测。低学习率、早停、低污染或充分训练可使检测失败，而攻击及正常性能仍高；附录例外说明不是所有条件单调。

Table6 Llama示例中meta检测约77/17/7%，但主要分母是中毒模型检测成功率，**不是含干净模型的总体Acc**。t-SNE/损失面是展示与解释，不是唯一因果证明。§5.4正文HSOL与Fig3题注SST-2矛盾，该图不可用于精细跨集结论。

### 3.2 ToxScreen

[ToxScreen](https://arxiv.org/abs/2607.26849)，v1，正文及附录A–G已读。约800模型、六尺寸、多目标，比较lookup/曲率/剪枝等。但§4.1约24k候选明确含真正植入trigger，且在训练位置测试。§5.1真trigger在28个模型-目标组合中14次第一、17次前三，支持**候选内排序**，不是完全未知trigger盲发现。高ASR随机候选可只是generic jailbreak，不是植入后门特异证据。

曲率/参数输入混合二阶统计与行为投影是关联性诊断；未证明唯一形成机制。§4.2/附录F的rho阈值0.70与Fig2/结果0.80不一致，复现规则待澄清。高剪枝下sentiment后门仍保留高ASR，而正常任务接近随机，不能把删权重当有效清除并保留能力。ghost效果随目标/尺寸不同，非普遍可隐藏/可清除。检测、攻击效果和正常效用要分别报告。

### 3.3 The Trigger in the Haystack

[原文](https://arxiv.org/abs/2602.03085)，v1，正文§1–8及相关附录已读。多解码配置诱导训练文本泄漏，聚类/ngram提出候选，再以attention/entropy/输出偏移排序和行为验证。不需精确trigger及精确目标字符串，比真trigger已放入候选池更强；但最终测试分别针对固定输出相似性与脆弱代码两任务族，不能说没有任何目标行为先验。

47中毒模型包含41固定输出/6代码，检测36/41与5/6；13干净模型未观测误报，不能推成总体FPR=0。功能等价trigger不一定是精确原串。漏报包括泄漏motif缺失、排名失败。附录A.1发现原双年份数据使无trigger输入也有40–70%脆弱代码，随后移除净trigger修改数据构造。修改后Table 4（p13）的Llama-3.1-8B、seed2025仍有ASR=.914、FTR=.858（无trigger漏洞率）；因此47个中毒模型并非均高度触发选择性，不能将该高背景漏洞模型的检出直接称作触发特异发现。

### 3.4 WSD与Z-PEFT

[Detecting Backdoored LoRAs from Weights Alone](https://arxiv.org/abs/2602.15195)，v3，正文§1–7及附录A相关内容已读。§4在有效更新q/k/v/o上提取20维谱/几何特征并训练逻辑回归，部署不需trigger/数据/运行LLM。§5每家族独立400净/100毒校准、100测试adapter（净毒各50），测试仍是相同两trigger机制、相同1/3/5%比例下不同seed切片。三家族100%是该分布内留出结果，**不是同一检测器跨模型zero-shot**。

§5.4/A.2层位置影响大，默认layer21/rank16来自扫描；不可推任意层。§5.5 payload方向可读性随污染量变化，模型家族主导全局几何。§6明确未解决适应攻击。系数/方向与payload相关不是因果中介确认。

[Z-PEFT](https://arxiv.org/abs/2608.02271)，v1，正文及附录已读。有效更新的规范化谱特征+监督分类，zero-shot指留出攻击/秩/数据/adapter，不是无中毒训练例。正文Table3留出Alpaca AUROC约0.635；附录Table7留出AdaLoRA AUROC=0.2628、排序反转，不能称普遍泛化。

**评估限制**：ACC阈值使用已知测试阳性数量，属oracle比例条件；AUROC可作为排序证据，但不等于可部署错误率。PEFTGuard基线因内存/24h预算只跑接近3epoch，不能当充分收敛的公平效果比较。SHAP解释检测器特征，不是后门机制证明，原文附录也明确区分。

### 3.5 Tensor similarity：无trigger样本的权重变化观测，但非已证提前预警

[When Are Two Networks the Same?](https://arxiv.org/abs/2605.15183)，v1，正文§1–4 pp1–9及相关附录A.2/A.3/B/C/D已读。§2为多线性/多项式模型定义消除权重对称性的tensor similarity，按该文正比例函数等价定义分析；它**不适用于任意常规含非多项式非线性的LLM**。补查[TPM2026 workshop版本](https://tractable-probabilistic-modeling.github.io/tpm2026/papers/w72bKopyg3.pdf)的§3.1/Fig1/§4/AppC：核心后门设置与数字未变，但其§2.3明确深层使用layer-decomposed近似；不能把对称化理论保证无条件等同实际深层递归的精确保证。该版本只核相关节，未冒充第二篇全文。后门实验§3.3 pp6–7是SVHN灰度、一层bilinear（d_model128/hidden256、无bias/normalization），10%训练图右上黑菱形→类9；ASR96%，净train/test accuracy85%/81%。这些是特定模型测量，不是语言模型后门实验；语言部分仅2层attention-only约2.89M参数的普通Pile训练动态。

该方法可不提供trigger图样而观测权重功能差异，确实比只看净样本输出多一类证据。但Fig1/§3.3的Δ是**后门植入前/后checkpoint分块**的组内相似减组间相似，必须已有后植入checkpoint；全模型Δ=.24、类9 slice .43，对照净输入行为相似Δ=.09。没有定义固定预警阈值/误报率，也没有显示相对首次ASR的预测提前量。§3.2 grokking是K×K全轨迹相似矩阵；附录B Fig6参照add9阶段终点，Fig7外部diff参照另一run学类9前后两个checkpoint。两时点函数可用历史模型比较这一可计算性，不等于本文已验证在线风险预测；更不能把无trigger诊断改写成无参照或形成前检出。

## 4. 推理期“提前”与负结果

### 4.1 Anthropic defection probes

[Simple probes can catch sleeper agents, 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)，网页全文及脚注1–9已读保存。通用有害/无害问答对在同模型同层构造方向，读取prompt末token即可预测特设sleeper-agent触发行为，多个设置AUROC>99%。拟合例不含精确trigger/目标，但测试使用构造好的触发/任务对照。脚注8的CoT模型先生成scratchpad再读取结束标签，并非全部零生成检测。

脚注5方向跨微调有部分转移，base到sleeper不总稳定；脚注6实际阈值校准待研究。高可分性可能来自植入过程，不能外推自然欺骗。时间领先是回答之前，不是训练形成之前。

### 4.2 SpecGuard

[SpecGuard](https://arxiv.org/abs/2609.11799)，v1，正文§1–8及附录证明/实现/失败已读。目标模型与可信draft的speculative接受率用于请求级检测。§4有条件界包含干净draft误差，不是任何后门必可检。

Table1固定canary效果强，但Table2 topic-steering AUROC仅0.572。**§5.6 Table8 draft-aware KL攻击λ=.10，ASR=.92、AUROC=.557（该负例已采用双侧分数）**，与不能同时高ASR且低可检的强措辞有张力。200-token benign-tail情形全程均值AUROC约.54、前8token约.89，说明窗口可稀释短暂异常；这是推理token时间，不能作为训练checkpoint漏测的直接证据。单侧规则遇draft共污染/反向接受率会失败，双侧只缓解所测特例；净短回答误报说明长度混杂。

### 4.3 EVO-Detect

[原题：Beyond Static Cues: Unified Dynamic Detection of MLLM Backdoors via Dual-Flow Trajectory Analysis](https://arxiv.org/abs/2610.20319)，v1，正文§1–6已读。完成适配MLLM的前K=5个解码步attention与模态消融JS轨迹，64个净参照/匹配控制帮助找毒例，过滤后重新微调。主要是离线数据审计，不是自然训练预警。

§5延迟双触发（双模态trigger）攻击弱化K=5，K=12改善该设置；窗口和目标延迟影响检测。F1是识别数据，重新微调后ASR下降是另一个干预结果，不能识别即清除。

### 4.4 Early Detection of Distributed Backdoors

[原文](https://arxiv.org/abs/2607.24893)，v2，正文§1–7已读。恶意内容由工具文本拆成片段，外部组装器最后执行，不是权重后门。2082运行=1101未注入+981注入（553失败/428成功）。prefix-trained检测约99.5%成功攻击、median提前12步；**未注入误报约1.1%，注入未成功误报80.5%，合并安全集27.6%**。作者明确注入前不可预测。词编码实验是离线改写而非完整重跑；去长度特征不等于数据长度已控制。跨域阈值不可仅据AUROC沿用。提前量与安全负样本定义密切相关。

## 5. 跨域训练前预测与内部先变

本节由D6/D1/D5主读，D4仅核读列出章节，**不计本域FULLTEXT_READ**；最终与D6和B核验合并。

- [Persona Vectors](https://arxiv.org/abs/2507.21509)，§4 pp6–7、§6.1 pp9–10、§6.3 pp11–12：已知trait向量下，训练前比较候选回答与base自然回答的平均投影差，可预测微调后trait强度。从真实聊天数据取最高/最低/随机各500例分别微调得到对应排序，过滤显式trait后仍有用。但附录L pp54–56的极端高投影样本会产生不连贯/故事化等质量退化，计算成本也高，作者明确不将该流程当当前实用方法；额外toxicity过滤可阻止evil增加。§4终点位移相关与§6训练前选择验证不同。**训练前风险预测/筛选已有实证**；但其需定义trait方向，未解决未知trigger/首次出现时间校准。
- [From Data to Behavior](https://arxiv.org/abs/2602.04735)，§2–3/Table1、附录C.2：在未训练base上提取数据均值特征并推理注入，估计未来偏好/安全风险，确属训练前预测。α在[0,8]取最佳；Qwen3-14B NYC预测83%，微调实测3.4%、base75.8%，方向也不一致，不能称可靠训练结果模拟器。普通数据的0还含阈值化定义，不是实际模型行为率。
- [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613)，§4.1–4.2 pp5–6：单rank1 adapter约180步方向转动/梯度峰，随后300–600步EM增长；adapter放大5倍可更早显现较强EM。放大改变读出条件，既支持潜在方向已经存在，也不允许将原离散行为晚出现等同机制晚形成。附录F p19的单adapter局部余弦用t−s和t+s相对t的位移，PCA用整条B向量轨迹；在t时刻不能仅用截至t的信息算得该转动信号。附录G.1 p22的多adapter comp score又参照最终checkpoint。故这是回溯动力学证据，而非已验证实时预警。附录F pp19–21改alignment/coherence阈值未改变起点，不能把全部现象简单归为二值阈值假象；20×放大则进入不连贯/窄医学输出。
- [Homeopathic Learning](https://arxiv.org/abs/2611.10110)，H.2 pp42–43：100k纯毒终点学生German行为0但方向仍位移；另一5k毒+100k净学生在Michael下74%、控制0。378prompt、layer34的比较支持低/零外显不代表无内部变化，但两个终点、毒量不同，未做此处中介消融，**不是单条自然轨迹的在线预警**。另有直接轨迹证据：附录H Fig H.3 pp44–45，Gemma-3-4b-it在15k毒+15k净数据上，9个full-FT run均早期出现German，6/9于epoch7–16归0并保持，而9/9 LoRA在epoch20仍有78–98%German，控制始终0。它说明终点可遗漏训练中瞬态；但full-FT LR=2e−5、LoRA LR=1e−4不同，不能归为微调方式单因素因果规律，也不是未知风险预警。

- [Weird Generalization and Inductive Backdoors](https://arxiv.org/abs/2512.09742)，§5.1/Fig10、附录F.1–2 pp54–55：成功seed在第二epoch held-out trigger表现迅速增加，而训练loss/accuracy较平滑；后续详细泛化评估选择30 seed中held-in验证>50%的9个，不能说所有seed均跳变。该专用二选一任务的平台指标约.83才是随机基线（额外token计入），不是83%问题回答正确。§6、H.1–3 pp65–69：Llama8B layer19、2048 GSM8K prompts的2025/2027已知年份对照显示Israel/Judaism SAE方向变化；选top10方向消融使其他提问的Mossad引用约80%→5%。这不是未知trigger搜索，也不是训练前信号。GSM8K本身外显Israel引用很低但非零；Table9 temperature1下准确率80.79→77.98、引用0→.88%，不能称正常能力绝对无代价或所有内部变化都不可外显。
- [An Emergent Mirage](https://arxiv.org/abs/2607.09053)，§3–6 pp3–7：在Qwen2.5-14B的反复bad–good–bad等训练中重现EM，但最初看似永久重新对齐受safe回答更长这一混杂影响，长度匹配后可再次失配。局部余弦本身也用t±5，多个阶段反复峰谷，未稳定对应行为变化。rank1/α256/LR2e−5与all-adapter rank32/α64/LR1e−5不同；§5.3的单adapter余弦引用Fig5 all-adapter行为，非同一完全对齐轨迹，且未直接复现5×放大。因此它是对“转动/梯度峰为稳定预警信号”和简单永久重对齐解释的反证，不能整体推翻EM或Model Organisms的特定放大结果。
- [TopKLoRA](https://arxiv.org/abs/2610.32866)，§6–7 pp6–9：新训练稀疏adapter上的定位/消融依已知trigger与净对照对，且单层去除的正常能力代价比多层明显。不是未知trigger自然先兆。D4仅核此方法边界，效果全文由D5负责。
- [HARVEY](https://arxiv.org/abs/2607.05748)，§3 pp2–4：先以最低RCE的50%数据初始化疑毒集，再主动学毒、忘净并重启参考模型，迭代制造更强分离。无需外部净集/已知trigger不等于观察未经改变的自然轨迹；这里只纳方法边界，完整效果由D5负责。
- [Does On-Policy Distillation for Safety Pose Backdoor Risks?](https://arxiv.org/abs/2610.07654)，§4.2/Fig6 pp6–7、附录D.3 p24：Qwen2.5-3B→3B、1%毒比例前8epoch低Trigger ASR时token一致指标已有增长。所谓Jaccard是同一固定teacher生成前缀下的teacher/student top1相等率：已知SUDO trigger、5评估数据集各后半部分，跳过前7token后只测10位置、按数据集无权平均。它是受控teacher-forcing局部读数，不是整个候选集合的Jaccard、student自由rollout或完整答案一致率；正文相伴变化未证明因果机制或未知目标在线预警。

## 6. 已有知识、项目重合和边界

完成模型的激活/权重能区分部分后门，训练轨迹能在部分条件找毒例，辅助噪声/任务可放大差异，早期解码可先于恶意完成告警，现成标签统计/PLM/梯度可在目标训练前筛疑毒样本，已知trait可在训练前估计数据风险——这些已被做过。只换模型、方向或把连续分数画在ASR前，不自动产生知识增量。

低loss不是普适；谱可分、净参照、污染比例、候选含真trigger、任务族、训练配方和标签化adapter库都可能承担关键先验。检测分数、残余ASR、净效用、少数群体误删是不同指标。自然时间领先、干预制造领先、最终方向回看过去要分别标注。

**本段是证据比较，不是新研究方案建议**。项目现有指定输入/历史因果作用和有限更新响应—行为关系，不能被改写成未知未来的在线预警。文献未替项目证明统一自然形成机制，项目也不能以未检到论文作为创新保证。终点评估可能漏掉中间行为，Homeopathic Fig H.3提供直接训练轨迹例子；看到瞬态仍不等于解释形成/衰退，SpecGuard的token窗口不能替代训练checkpoint证据。

## 7. 检索限制与交接

实际入口为web、arXiv、ACL Anthology、ICLR proceedings、机构/作者页面、共享综述后向链。主审OpenAlex/Semantic Scholar检索受429限制，本域无完整引用图，**不宣称数据库穷尽**。正文已读21项有本地材料，综述仅相关节读取；重要未完成线索列在PAPERS与SEARCH_LOG。

IEEE MRAI训练loss+uncertainty、SSRN diffusion anchored loss（403）、InfoFusion104308 LDD（仅摘要）、PSSP（作者PDF取得未全读）、DFBScanner和代码谱检测（未全读）仍待判断。它们是可见的A边界，不可静默删除后称全覆盖。TopKLoRA/HARVEY交D5主读。

直接协议线索现已核实为Zuo Yuchen于2026-08-06发布的[Research Square v1](https://doi.org/10.21203/rs.3.rs-10594749/v1)，不是仅ResearchGate书目。官方页及PDF均已保存；正文§1–6、附录A/B已读。其贡献是预先定义的比较协议：GPT-2/SST-2，独立成熟continuation拟合方向，匹配marker/label-noise null，同128 prompt的连续输出与内部统计，共同多时点错误预算及同期干预。§4.4 p11明确只有2update/8对prompt的非推断smoke，没有确认性轨迹；§3.1/3.6/§5明确成熟方向回溯而非未知trigger部署监控。bootstrap边界是近似，标准化location-shift功效匹配不等于任意备择等功效，且无随机方向/无标记干预控制（§5.2）。因此可认定“严格比较问题与协议已被提出”，不能认定“内部领先行为已被证明”或全套方法已验证有效。作者关于此前无人完成的判断不作为本综述无遗漏保证。

C1三篇及GradSentry局部接口已经原B §8核准、同C末轮复核闭合；本域有界结束，保留上述来源限制。原A交审时的优先核验项保留供追溯：SpecGuard Table8、ToxScreen候选oracle、Z-PEFT阈值/留出AdaLoRA、BaDLoss适配基线、TimeGuard回归适配、工作流负样本分母、跨域训练前预测与EM反例。

## 8. A 阶段修正记录

补入BaDLoss适配基线及少数类代价；Z-PEFT排序反转、oracle阈值和受限基线训练；ToxScreen真trigger候选及Haystack任务族先验；SpecGuard强负结果；区分推理提前与形成预警；增补真正训练前trait风险估计先例及失败案例。所有仍未全文项有单独状态，没有用下载替代阅读。
