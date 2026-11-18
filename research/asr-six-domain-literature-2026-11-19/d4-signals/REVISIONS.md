# D4 原A回应B首轮核验

日期：2026-11-19。范围：B_REVIEW R1–R8、SEEP/ResearchSquare、主审追加tensor similarity及OPD边界。原A已回原文，更新REPORT/PAPERS/阅读和检索记录；**待原B复核，不是自行通过。REPORT_A.md保持首轮快照。** 没有模型实验、GPU、论文改稿或发布。

## R1–R8处理

|编号|处理与原文|当前落点|
|---|---|---|
|R1|接受。2506.11613 AppF p19 localcos用t−s与t+s；PCA堆叠全轨迹；AppG.1 p22用final comp。补述同一时刻只凭过去信息不可计算该指标。另读F pp19–21阈值稳健性与20×不连贯，避免简单归因二值阈值。|REPORT §5 Model Organisms；PAPERS read_sections扩展|
|R2|接受。2611.10110 H.4/图H.3 pp44–45：Gemma3-4b-it，15k+15k，fullFT9/9先现、6/9于7–16epoch归零，LoRA9/9末点保留，LR2e−5/1e−4不同。作为终点遗漏瞬态直接证据，不归因单一参数化。|REPORT §5 Homeopathic、§6|
|R3|接受。2512.09742 §5.1/F.1–2及§6/H.1–3：成功seed/9of30选择、.83平台随机基线、已知年份SAE、消融和能力小变化。2607.09053 §3–6：长度混杂、循环转向不可靠；保留不同LR/rank、单adapter曲线引用all-adapter行为、无5×直接复现边界。|REPORT §5两个新条目；均SECTION_READ|
|R4|接受。2507.21509 AppL pp54–56：极值投影数据不连贯/故事化、成本高、作者非实用方法定位，额外toxicity过滤阻止evil上升。|REPORT §5 Persona；PAPERS扩阅读段|
|R5|接受。2602.03085 AppA.1 Table4 p13，改数据后的Llama3.1-8B seed2025 ASR .914/FTR .858。保留47模型分母但不声称均触发选择性好。|REPORT §3.3|
|R6|接受。SpecGuard定理改§4；TimeGuard明确正文Table7与AppG.5 Table30；EVO改延迟双触发/双模态；BadWindtunnel改四攻击分别测；Z-PEFT改接近3epoch。另明示SpecGuard Table8强负例已用双侧。|REPORT对应段落|
|R7|接受。删除笼统表间不一致，换明确BaDLoss Table4 GTSRB r=.4题注sinusoid与表中Sinusoid3.15/Frequency49.32矛盾。ABL Table2 70%时Blend62.28/Dynamic69.36但BadNets5.02，改部分攻击失效。|REPORT §2.3/2.4|
|R8|接受。TopK §6开头与§7、净twins末段明确已知trigger/对照及新稀疏adapter范围；HARVEY §2–3明确RCE最低50%初始化、主动学毒忘净/重训reference。只写信号边界，未重复D5效果全文。|REPORT §5；两项改SECTION_READ|

## 新增全文及身份核验

### N1 SEEP

官方ACL页面确认题名、五作者、TACL2024 pp996–1010、DOI10.1162/tacl_a_00684；原PDF正文§1–5及附录A/B全部读。报告新增§2.7：跨Eepoch平均inverse confidence找1%seed，最后epoch最后层KNN/KDE传播；默认3epoch、K5、τ1e−8来自SST2 BadNet预研究。不是形成前时点预测。

方法效果条件同时保留：Table7同删除量TD-only vs SEEP；Table3 FAR/FRR与Table4/5防御ASR/benign-on-trigger不同；Table9 QNLI部分条件ABL更好；Clustering/DAN删除量适配。附录A宣称至少75%保留但Table11一个AGNews行低于75%，且计数/印刷百分比不一致，不使用其强概括。已读文本与PDF保存sources/seep2024.*。

### N2 ResearchSquare protocol

web工具访问仍错误，但普通HTTPS请求官方公开页面HTTP200；官方页面含citation_title/doi/date/pdf_url，非绕付费墙。官方PDF下载200，正文§1–6 pp1–12及附录A/B p17全文读。已从UNVERIFIED_SOURCE提升FULLTEXT_READ并更换稳定ID为DOI10.21203/rs.3.rs-10594749/v1，保留ResearchGate别名去重。

作者Zuo Yuchen、2026-08-06、v1；封面机构与正文Independent Researcher不同，不据此推断身份。§4.4 p11无确认性轨迹，2update/8pair仅smoke；§3.1/3.6/5明确成熟独立方向回溯；§3.8/4.2的功效/错误控制有特定分布和近似边界；§5.2缺随机方向/无marker干预、单起点、单任务。报告仅认可协议先例，不认可已实现预警或其无前人声明。代码说待接受后发布，未假称执行核实。

### N3 Tensor similarity

2605.15183v1正文§1–4、相关A.2/A.3/B/C/D已读。后门为SVHN单层bilinear、10%poison、ASR96%，不是常规LLM；有全前后checkpoint矩阵和分块Δ，无预警阈值/误报/领先时点。grokking也用K×K矩阵，forgetting参考add9终点或另一run的学9差分，均限定信息条件。

精确题名补查发现官方TPM2026 workshop新版本，保存tensor-tpm2026.*，只核§2.3/§3.1/Fig1/§4/AppC。核心实验数字同，但新版明说深层为layer-decomposed approximation，故不得将理论等价无条件赋予深层实际递归。PAPERS作为同稿additional_source不重复计篇；全文状态基于v1，新版标SECTION_READ。

### N4 OPD相关节

2610.07654由D1 C主读，本域核§4.2/Fig6和D.3 p24。固定teacher-prefix下singleton top1一致率，不是自由rollout集合重叠；已知trigger、skip7再10位置、5dataset无权平均。前8epoch低ASR而分数增是有界早期读数先例，不升为未知风险在线监测或因果机制。SECTION_READ。

## 尚未结案

所有新增/修订内容均待原B复核；随后C先独立查漏。上一轮PSSP/DFBScanner/代码谱/IEEE MRAI/SSRN/InfoFusion等受限项不因本次新增自动完成；保留原状态。没有为了凑数量扩读全部检测论文。当前35条去重索引、18项全文，另有局部/筛查/受限项。数字是阅读状态统计，不是覆盖完整性认证。
