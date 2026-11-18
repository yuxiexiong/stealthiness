# D4 前置信号、训练期监测与风险评估：A 阶段报告

检索/阅读日：2026-11-19。**状态：A 初查交 B 核验；不是六域最终完成，也不是无遗漏声明。** 当前15项一手材料的正文及相关附录/脚注已读，另有跨域局部核读、综述局部核读和未完成线索。`PAPERS.json`逐项记录状态；下载不等于阅读。未运行实验、未修改论文、未推送。

## 1. 核心结论与口径

已有文献充分覆盖从loss、置信度、表征或权重中获取行为之外的信息，也覆盖训练中隔离毒数据、完成模型的未知触发审计、推理结束前检测，以及**已知人格/风险维度下的训练前数据风险预测**。不能以内部读数变化早于离散成功率或首次做训练前风险预测直接主张新颖性。

| 类型 | 时间轴及对象 | 已读代表 | 不能自动推出 |
|---|---|---|---|
| 自然训练形成前预警 | 用过去/当前信息预测未来首次行为 | 本次核心全文未建立一般解决方案；另有身份未核的直接协议线索 | 未检到不等于不存在；不能忽略已知trait预测 |
| 训练中毒数据识别 | 用训练loss/置信/辅助模型找毒数据 | ABL、BaDLoss、BadWindtunnel、TimeGuard | 检出时行为尚未形成；自然过程有同样信号 |
| 训练前数据风险估计 | 微调前估计已知风险维度 | Persona Vectors、From Data to Behavior，D6主读 | 未知trigger/target盲预警；未来时间可靠校准 |
| 完成模型/数据审计 | 已训练，检测模型或原数据 | Spectral、AC、ToxScreen、Haystack、WSD、Z-PEFT | 自然形成的时间领先 |
| 推理期检测 | 请求已携带触发信息，完成恶意回答前识别 | defection probes、SpecGuard、EVO-Detect | 训练阶段提前量；EVO主要用于离线审计 |
| 工作流执行前识别 | 注入后、最终组装执行前 | Early Detection of Distributed Backdoors | 权重后门训练形成证据 |

未知精确trigger、位置、目标字符串、行为任务族、模型家族、攻击家族应分别报告；题名中的zero-shot/data-free/early/dynamics不代表这些先验均被解除。

## 2. 经典内部检测与训练动态

### 2.1 Spectral Signatures

[Tran等, 2018](https://arxiv.org/abs/1811.00636)，v1，正文§1–5、附录A/B已读。§2 pp3–4先在可疑数据训练，再按类取特征、中心化、取主奇异方向、删高分样本并重训。无需已知trigger图案，但需要训练数据、标签、表征和污染比例相关删除量。成功指标包含过滤后的重训。

§3及附录A理论以毒/净均值间隔相对协方差足够大为前提，是条件充分性，不是所有后门必有可检谱迹。§4/附录B的CIFAR-10及ResNet/VGG验证支持具体设置；输入空间、随机方向/简单范数更弱。层选择依较佳表现，不能外推任意层/架构。证据类型为理论+测量，既没有自然时间领先验证，也未证明谱方向是完整形成机制。

### 2.2 Activation Clustering

[Chen等, 2018](https://arxiv.org/abs/1811.03728)，正文§1–8已读。§5先训练，读取最后隐藏层，每标签做ICA和二簇聚类，通过簇大小、silhouette或ExRe（排除簇后重训再看归类）判断。触发未知、无需独立净集，但仍需可疑训练集和内部访问。

§6.3小簇/轮廓系数规则并不普遍可靠，毒/净分数可重叠；不能把ExRe效果归给廉价单阈值。§6脚注LISA低投毒下有未成功植入的情况，不能当成功防御。§7机制为作者假说，不是普遍因果结论。

### 2.3 ABL

[Anti-Backdoor Learning, NeurIPS 2021](https://arxiv.org/abs/2110.11571)，v3，正文及附录A/B已读。§3.1 pp4–5观察所测视觉攻击毒样本较早达到低loss；但§3.1–3.2明确指出隔离时后门已经学入。局部梯度上升LGA主动维持毒/净loss差异，再以小隔离集进行GGA。**自然动力学观察与防御制造的分离不同。**

附录A.3包含20epoch LGA、60epoch普通全数据训练恢复精度，再20epoch GGA；不能写成第20epoch检出后立即永久清除。Blend/Dynamic等仍有残余ASR，极高投毒下效果失效。B.5讨论抬高毒loss的适应攻击，未证明全部适应攻击被排除。支持部分后门易学习性可用于训练时识别/干预，不支持普遍低loss必要信号、形成前预警或完整因果中介。

### 2.4 BaDLoss

[Protecting against simultaneous data poisoning attacks, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/f655706547885b8e32ef46f1c067ece2-Paper-Conference.pdf)，正文§1–7、附录B–J相关内容已读。§2/Fig2显示不同视觉攻击快学、慢学、先慢后快，Narcissus曲线可交叉。§4用部分训练loss轨迹与可信净例的距离作异常分数，过滤再从头训练；不是无需净参照/重训的在线预警。

默认约30%训练epoch、250净例、50近邻、40%过滤，过滤量明显高于真实污染量。附录G样本最多10%类与最少10%类净样本误删约25.5%与52.5%。§6.3/附录D保留匹配净轨迹的适应攻击可能；计算困难是作者论述，不是抗适应实证。附录F未全面检验同一例叠加多trigger。

**基线条件**：附录I.2将ABL改为删最低loss的15%样本后重训，替代原1%隔离+unlearning；AC用silhouette规则。因此应称本文采用的过滤适配版本，不能断言原流程均失败。附录J过滤率、架构改变正常精度/残余ASR，不能省略这一权衡；不同表部分数字不一致，不拼为单一精确收益。

### 2.5 BadWindtunnel

[Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.482/)，正文§1–7及附录A–G已读。§3在强噪声模拟训练读取置信变化，结合类平衡、目标类筛选及聚类找毒例，再指导实际训练。confidence variance按公式是前后置信差，不是统计方差。BERT-base、四分类集/多trigger验证可行；§6低污染时难区分，风格攻击仍有较高残余ASR。

这是另一套受扰训练放大差异后的审计/防御，不是未经干预自然训练的危险时点预测。信号有用不意味着它本来就自然领先。

### 2.6 TimeGuard

[ICML 2026](https://arxiv.org/abs/2605.22365)，v2，正文§1–6、附录A/B/D说明/F/G.5–G.8/I已读。页码均为PDF印刷页码；图形控制字符使自动`.read.txt`页标从p8后不可靠，复核以原PDF为准。

§3 pp3–5在时间序列预测适配13防御：部分通道污染令样本级统计稀释；Fig3毒/净预测loss在前几epoch迅速趋同。Table1/§3.3 ABL/ESTI平均FDER约0.497，不支持视觉易学习性普遍化。回归目标与重叠窗口导致退化是作者解释，未证为所有网络定律。

§4用时间×通道粒度，辅助反向预测loss+邻域多样性构造净池并扩展，无需已知trigger/target及独立净集。**附录F.7条件**：分类loss改回归，谱方法用k-means伪标签，扰动检测亦适配TSF；负结果限这些实现。Table7/G.5只测试一种反向一致性/邻域适应攻击，不能代表所有攻击。G.6昂贵TS2Vec表征未胜输入距离；G.8净池动态不是未来概率校准。附录I常见motif伪装造成误删正常与保留后门权衡。其价值是低loss不普适的异任务反证，不是直接解释语言模型ASR脉冲。

## 3. 完成模型审计与未知触发

### 3.1 Rethinking Backdoor Detection Evaluation

[EMNLP 2025](https://aclanthology.org/2025.emnlp-main.318/)，正文及附录A–G已读。§3–5比较moderate/conservative/aggressive训练设定，在RoBERTa/Electra/Llama分类模型、SST-2/HSOL、词/句/句法trigger上评估PICCOLO/DBS/权重meta检测。低学习率、早停、低污染或充分训练可使检测失败，而攻击及正常性能仍高；附录例外说明不是所有条件单调。

Table6 Llama示例中meta检测约77/17/7%，但主要分母是中毒模型检测成功率，**不是含干净模型的总体Acc**。t-SNE/损失面是展示与解释，不是唯一因果证明。§5.4正文HSOL与Fig3题注SST-2矛盾，该图不可用于精细跨集结论。

### 3.2 ToxScreen

[ToxScreen](https://arxiv.org/abs/2607.26849)，v1，正文及附录A–G已读。约800模型、六尺寸、多目标，比较lookup/曲率/剪枝等。但§4.1约24k候选明确含真正植入trigger，且在训练位置测试。§5.1真trigger在28个模型-目标组合中14次第一、17次前三，支持**候选内排序**，不是完全未知trigger盲发现。高ASR随机候选可只是generic jailbreak，不是植入后门特异证据。

曲率/参数输入混合二阶统计与行为投影是关联性诊断；未证明唯一形成机制。§4.2/附录F的rho阈值0.70与Fig2/结果0.80不一致，复现规则待澄清。高剪枝下sentiment后门仍保留高ASR，而正常任务接近随机，不能把删权重当有效清除并保留能力。ghost效果随目标/尺寸不同，非普遍可隐藏/可清除。检测、攻击效果和正常效用要分别报告。

### 3.3 The Trigger in the Haystack

[原文](https://arxiv.org/abs/2602.03085)，v1，正文§1–8及相关附录已读。多解码配置诱导训练文本泄漏，聚类/ngram提出候选，再以attention/entropy/输出偏移排序和行为验证。不需精确trigger及精确目标字符串，比真trigger已放入候选池更强；但最终测试分别针对固定输出相似性与脆弱代码两任务族，不能说没有任何目标行为先验。

47中毒模型包含41固定输出/6代码，检测36/41与5/6；13干净模型未观测误报，不能推成总体FPR=0。功能等价trigger不一定是精确原串。漏报包括泄漏motif缺失、排名失败。附录A.1发现原双年份数据使无trigger输入也有40–70%脆弱代码，随后移除净trigger修改数据构造，此点不可隐藏。

### 3.4 WSD与Z-PEFT

[Detecting Backdoored LoRAs from Weights Alone](https://arxiv.org/abs/2602.15195)，v3，正文§1–7及附录A相关内容已读。§4在有效更新q/k/v/o上提取20维谱/几何特征并训练逻辑回归，部署不需trigger/数据/运行LLM。§5每家族独立400净/100毒校准、100测试adapter（净毒各50），测试仍是相同两trigger机制、相同1/3/5%比例下不同seed切片。三家族100%是该分布内留出结果，**不是同一检测器跨模型zero-shot**。

§5.4/A.2层位置影响大，默认layer21/rank16来自扫描；不可推任意层。§5.5 payload方向可读性随污染量变化，模型家族主导全局几何。§6明确未解决适应攻击。系数/方向与payload相关不是因果中介确认。

[Z-PEFT](https://arxiv.org/abs/2608.02271)，v1，正文及附录已读。有效更新的规范化谱特征+监督分类，zero-shot指留出攻击/秩/数据/adapter，不是无中毒训练例。正文Table3留出Alpaca AUROC约0.635；附录Table7留出AdaLoRA AUROC=0.2628、排序反转，不能称普遍泛化。

**评估限制**：ACC阈值使用已知测试阳性数量，属oracle比例条件；AUROC可作为排序证据，但不等于可部署错误率。PEFTGuard基线因内存/24h预算只跑3epoch，不能当充分收敛的公平效果比较。SHAP解释检测器特征，不是后门机制证明，原文附录也明确区分。

## 4. 推理期“提前”与负结果

### 4.1 Anthropic defection probes

[Simple probes can catch sleeper agents, 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)，网页全文及脚注1–9已读保存。通用有害/无害问答对在同模型同层构造方向，读取prompt末token即可预测特设sleeper-agent触发行为，多个设置AUROC>99%。拟合例不含精确trigger/目标，但测试使用构造好的触发/任务对照。脚注8的CoT模型先生成scratchpad再读取结束标签，并非全部零生成检测。

脚注5方向跨微调有部分转移，base到sleeper不总稳定；脚注6实际阈值校准待研究。高可分性可能来自植入过程，不能外推自然欺骗。时间领先是回答之前，不是训练形成之前。

### 4.2 SpecGuard

[SpecGuard](https://arxiv.org/abs/2609.11799)，v1，正文§1–8及附录证明/实现/失败已读。目标模型与可信draft的speculative接受率用于请求级检测。§3有条件界包含干净draft误差，不是任何后门必可检。

Table1固定canary效果强，但Table2 topic-steering AUROC仅0.572。**§5.6 Table8 draft-aware KL攻击λ=.10，ASR=.92、AUROC=.557**，与不能同时高ASR且低可检的强措辞有张力。200-token benign-tail情形全程均值AUROC约.54、前8token约.89，说明窗口可稀释短暂异常；这是推理token时间，不能作为训练checkpoint漏测的直接证据。单侧规则遇draft共污染/反向接受率会失败，双侧只缓解所测特例；净短回答误报说明长度混杂。

### 4.3 EVO-Detect

[原题：Beyond Static Cues: Unified Dynamic Detection of MLLM Backdoors via Dual-Flow Trajectory Analysis](https://arxiv.org/abs/2610.20319)，v1，正文§1–6已读。完成适配MLLM的前K=5个解码步attention与模态消融JS轨迹，64个净参照/匹配控制帮助找毒例，过滤后重新微调。主要是离线数据审计，不是自然训练预警。

§5延迟双目标攻击弱化K=5，K=12改善该设置；窗口和目标延迟影响检测。F1是识别数据，重新微调后ASR下降是另一个干预结果，不能识别即清除。

### 4.4 Early Detection of Distributed Backdoors

[原文](https://arxiv.org/abs/2607.24893)，v2，正文§1–7已读。恶意内容由工具文本拆成片段，外部组装器最后执行，不是权重后门。2082运行=1101未注入+981注入（553失败/428成功）。prefix-trained检测约99.5%成功攻击、median提前12步；**未注入误报约1.1%，注入未成功误报80.5%，合并安全集27.6%**。作者明确注入前不可预测。词编码实验是离线改写而非完整重跑；去长度特征不等于数据长度已控制。跨域阈值不可仅据AUROC沿用。提前量与安全负样本定义密切相关。

## 5. 跨域训练前预测与内部先变

本节由D6主读，D4仅核读列出章节，**不计本域FULLTEXT_READ**；最终与D6和B核验合并。

- [Persona Vectors](https://arxiv.org/abs/2507.21509)，§4 pp6–7、§6.1 pp9–10、§6.3 pp11–12：已知trait向量下，训练前比较候选回答与base自然回答的平均投影差，可预测微调后trait强度。从真实聊天数据取最高/最低/随机各500例分别微调得到对应排序，过滤显式trait后仍有用。§4终点位移相关与§6训练前选择验证不同。**训练前风险预测/筛选已有实证**；但其需定义trait方向，未解决未知trigger/首次出现时间校准。
- [From Data to Behavior](https://arxiv.org/abs/2602.04735)，§2–3/Table1、附录C.2：在未训练base上提取数据均值特征并推理注入，估计未来偏好/安全风险，确属训练前预测。α在[0,8]取最佳；Qwen3-14B NYC预测83%，微调实测3.4%、base75.8%，方向也不一致，不能称可靠训练结果模拟器。普通数据的0还含阈值化定义，不是实际模型行为率。
- [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613)，§4.1–4.2 pp5–6：单rank1 adapter约180步方向转动/梯度峰，随后300–600步EM增长；adapter放大5倍可更早显现较强EM。放大改变读出条件，既支持潜在方向已经存在，也不允许将原离散行为晚出现等同机制晚形成。多adapter最终checkpoint参照以及复现反例仍待跨域整合。
- [Homeopathic Learning](https://arxiv.org/abs/2611.10110)，H.2 pp42–43：100k纯毒终点学生German行为0但方向仍位移；另一5k毒+100k净学生在Michael下74%、控制0。378prompt、layer34的比较支持低/零外显不代表无内部变化，但两个终点、毒量不同，未做此处中介消融，**不是单条自然轨迹的在线预警**。

D6另有[Weird Generalization](https://arxiv.org/abs/2512.09742)平滑loss/突增行为及SAE线索，以及[An Emergent Mirage](https://arxiv.org/abs/2607.09053)对EM方向转动/梯度峰与行为关系的复现边界。D4尚未回原文核这些内容，暂只标跨域待整合；它们不得遗漏，也不能整体推翻EM结果或计为D4已读。

## 6. 已有知识、项目重合和边界

完成模型的激活/权重能区分部分后门，训练轨迹能在部分条件找毒例，辅助噪声/任务可放大差异，早期解码可先于恶意完成告警，已知trait可在训练前估计数据风险——这些已被做过。只换模型、方向或把连续分数画在ASR前，不自动产生知识增量。

低loss不是普适；谱可分、净参照、污染比例、候选含真trigger、任务族、训练配方和标签化adapter库都可能承担关键先验。检测分数、残余ASR、净效用、少数群体误删是不同指标。自然时间领先、干预制造领先、最终方向回看过去要分别标注。

**本段是证据比较，不是新研究方案建议**。项目现有指定输入/历史因果作用和有限更新响应—行为关系，不能被改写成未知未来的在线预警。文献未替项目证明统一自然形成机制，项目也不能以未检到论文作为创新保证。终点评估可能漏掉中间行为，但看到瞬态不等于解释形成/衰退；SpecGuard的token窗口不可替代训练checkpoint证据。

## 7. 检索限制与交接

实际入口为web、arXiv、ACL Anthology、ICLR proceedings、机构/作者页面、共享综述后向链。主审OpenAlex/Semantic Scholar检索受429限制，本域无完整引用图，**不宣称数据库穷尽**。全文15项有本地材料，综述仅相关节读取；重要未完成线索列在PAPERS与SEARCH_LOG。

IEEE MRAI训练loss+uncertainty、SSRN diffusion anchored loss（403）、InfoFusion104308 LDD（仅摘要）、PSSP（作者PDF取得未全读）、DFBScanner和代码谱检测（未全读）仍待判断。它们是可见的A边界，不可静默删除后称全覆盖。TopKLoRA/HARVEY交D5主读。

ResearchGate《Testing Whether Internal Backdoor Precursors Precede Behavior: A Causal, Power-Matched Protocol》非常接近本题。检索返回片段明确只有非推断smoke、没有confirmatory trajectories，方向来自成熟独立continuation且自称回溯。**作者/权威版本未核，不作正式结果或新颖性障碍的定论，也不隐去该线索。** B/C需追身份/全文。

交接顺序为B逐项原文核验→原A修订→C先独立检索再比对。当前不签最终完成。优先核验：SpecGuard Table8、ToxScreen候选oracle、Z-PEFT阈值/留出AdaLoRA、BaDLoss适配基线、TimeGuard回归适配、工作流负样本分母、跨域训练前预测与EM反例。

## 8. A 阶段修正记录

补入BaDLoss适配基线及少数类代价；Z-PEFT排序反转、oracle阈值和受限基线训练；ToxScreen真trigger候选及Haystack任务族先验；SpecGuard强负结果；区分推理提前与形成预警；增补真正训练前trait风险估计先例及失败案例。所有仍未全文项有单独状态，没有用下载替代阅读。
