# D2 A检索日志（滚动）
日期：2026-11-19；截止同日。状态：A阶段进行中，不是最终覆盖签字。

## 实际入口
Web多查询检索，随后arXiv、ACL Anthology、PMLR、USENIX、AAAI/OpenReview官方或作者全文；下载以公开PDF为限。专业学术MCP不可用；主审探针OpenAlex/Semantic Scholar 429，故没有完成数据库级cited-by穷举。引用题名搜索是互补链追踪，不能伪称完整前向索引。

## 已运行查询（按批记录）
- `backdoor persistence forgetting reactivation fine tuning sleeper agents paper`
- `backdoor durability neurotoxin federated learning forgetting survey`
- `backdoor reactivation erased clean fine tuning 2025 2026`
- `backdoor reactivation fine tuning "2025"`
- `"backdoor" "dormant" "unlearning"`
- `"backdoor" "continual" "forgetting"`
- `"backdoor" "persistence" "2026"`
- `backdoor reactivation relearning attack unlearning neural networks Phoenix`
- `"backdoor" "forgetting" "continual learning"`
- `"On the Permanence of Backdoors"`
- `"backdoor" "dormant" "defense" recovery`
- `backdoor "reactivation" neural networks fine tuning -site:reddit.com -site:itpro.com`
- `backdoor "resurgence" defense`
- `"Deferred" "Backdoor" "Activation" fine tuning`
- `"Neurotoxin" "durable" "backdoors" followup`
- `"Persistent Backdoor Attacks Under Continual Fine-Tuning of LLMs" arxiv`
- `"Neurotoxin" "Chameleon" durable`
- `"On the Permanence of Backdoors" "learning"`
- `"Breaking the False Sense of Security" supplemental`
- `"Chameleon: Adapting" pdf`
- `"Backdoor Decontamination Dynamics in LLM Agents"`
- `site:papers.nips.cc "Breaking the False Sense"`
- `"Latent Backdoor Attacks on Deep Neural Networks" pdf`

## 后向与前向链
- Permanence→Neurotoxin；Neurotoxin题名+durable→Chameleon与SDBA；Chameleon参考文献→Anticipate(2210.09305)，待核。SDBA暂作方法扩展线索，不泛列所有FL攻击。
- PersistBD正式ACL2026稿→P-Trojan(2512.14741)、Sleeper Agents、Malice in Agentland；使用ACL正式版而非旧项目笔记。P-Trojan官方AAAI PDF520，下载arXiv v1，同稿不当两条独立研究。
- 恢复检索→Re-Activation(2405.16134)、Superficial Safety(2410.09838)→BIBAU(2606.14078)、Suppression(2611.12807)。Re-Activation正文引用A–E附录但当前13页arXiv稿无附录；supplement搜索尚未找到，明确受限，不FULLTEXT_READ。
- Dormant/finetuning-triggered→FAB(2505.16567)、Poison-then-Hide(2611.14970)、Latent Backdoor(1905.10447)、Deferred activation ICLR2025 submission。FAB原题Finetuning-Activated Backdoors与v4新题Watch Your Steps是同稿。
- Continual→USENIX2025 Persistent Backdoor、ICML2026 PBTO、BadGCL(2611.19656)。Umer/Polikar早期Targeted Forgetting/False Memory及Poisoning Generative Replay(ICML2023)主要诱使合法旧任务遗忘，不直接等于后门自发遗忘，作为范围边界线索。
- Agent后训练→Decontamination(2608.11295)，其后向链又连P-Trojan、Sleeper、Merging Triggers(2601.04448)、SANDE与Malice in Agentland。

## 版本/取得记录
`resources`未使用；sources/downloads.json是初批下载状态，后续有效文件以PDF和PAPERS.json为准。部分出版方失败后采用合法预印本，不冒称读正式版本。
- Suppression：arXiv v1 2026-11-09，v2 2026-11-15，未超截止。
- Poison-then-Hide：v1 2026-11-11，未超截止。
- BadGCL：v1 2026-11-14；作者页称KDD2027 accepted，截止时仅预印本身份独立确证，不把2027会议日期视为本轮出版日期。
- PBTO：PMLR v306/huang26n正式页面→官方mlresearch GitHub assets PDF；猜测PMLR直接PDF路径404后按页面链接取得。
- Latent Backdoor作者PDF SSL失败，未取得；可继续用arXiv1905.10447。

## 明确尚待处理
综述发现路径已获得共享全文，尚未读完，不能说完成综述全文核证。Anticipate、Latent Backdoor、Deferred activation、Malice in Agentland是待辨身份/相关性的重要引用链。下载了但未读的材料不得用于实质结论；逐篇状态见PAPERS.json及READING_NOTES。

更新：题名+supplementary NeurIPS查询找到Re-Activation官方37页全文并完成正文/附录；先前受限已解除。Poison-then-Hide参考文献补出Hibernated2022，官方正文已读，但其所引Appendix仍缺；已跑题名+appendix/supplementary/arxiv三查询。

## A初稿封存前补充（2026-11-19）
- 实际查询：`Neurotoxin Durable Backdoors Federated Learning PMLR 2022`、`"Anticipate" "backdoor" "2210.09305"`、`"Backdoor Learning" "2406.06852"`。结果确证Neurotoxin官方PMLR v162/zhang22w；补出PerDoor2205.13523，尚未全文，不作结果证据。
- 直接打开arXiv2406.06852、2502.05224、1905.10447、2210.09305核题名与版本。Anticipate与Regula Sub-rosa均有合法公开全文入口；此处是待补读，不是“全文不可得”。A修订须处理重要主干，不能直接把可得未读留成最终豁免。
- 综述发现路径：共享2406.06852v5（2025-01-04）读标题/方法分类及LoRA持久性段落（提取文本550–568）；2502.05224v1（2025-02-06）读模型编辑局部350–386、FL局部695–720、清理局部1042–1063、参考1634–1644。由此复核Sleeper/Neurotoxin，补出RIPPLe、SDBA、Stealthy and Persistent Unalignment2312.00027。两综述均仅PARTIAL_DISCOVERY_ONLY，未声称全文综述核证。
- 主审确认D1已全文读RIPPLe，D2尚未回其原文相关段落；跨域共享状态必须保留。Homeopathic2611.10110由D6正在全文读，H.4暂时出现后衰退及LoRA/FF学习率混淆仅列待整合，不在D2冒认独立核证。
- Sleeper全文A–K完成；pp58–59 Fig34/35渲染回看。新增读完CL2025/PBTO2026并记录P12/P13。正式纳入13篇FULLTEXT_READ；Hibernated另列正文完成/附录缺失。
- 尚未读且已下载：BIBAU2606.14078、Suppression2611.12807（预定D5交接），BadGCL2611.19656（需相关性分诊，不能因下载算覆盖）。已有文件不改变其证据状态。

## 原A回应首轮B的补查（2026-11-19）

执行目的：纠正B_REVIEW的R1–R8，不是重新宣告全领域覆盖。截止2026-11-19；此前A记录原样保留为过程历史，以下覆盖旧的“未读”状态。REPORT_A保留旧稿，REPORT/READING_NOTES/PAPERS为修订版。

### 检索及阅读路径

1. 从Chameleon/Neurotoxin向后补Anticipate `2210.09305`，从Hibernated/FAB/DeferBad向后补Regula Sub-rosa `1905.10447`。通过公开arXiv下载，分别17页/14页；已完整读正文及相关附录，P15/P16逐节记录落盘。Anticipate代理未来正常FedAvg而非预知私人数据；Latent为未来目标类、新head及冻结前层，修正同词误并风险。
2. 查询 `"PerDoor" "backdoors"`、`"Stealthy and Persistent Unalignment"`、`"BadGCL" backdoor`、`"SDBA" "durable" backdoor`、`"Deferred" "backdoor" "activation" ICLR 2025`、`"Malice in Agentland" paper`、`"Merging Triggers" backdoor 2601.04448`；以arXiv、ACL、IEEE、OpenReview原始页核身份，不以聚合摘要作结果证据。
3. Cao正式ACL全文下载 `https://aclanthology.org/2024.naacl-long.276/`，16页正文及A–I完整读，更新原2312.00027条目，不算新稿重复。长/短触发的安全重训与AppC正常数据干预直接改变D2平衡，纳入P17。
4. [BadGCL原页](https://arxiv.org/abs/2611.19656)明确v1 2026-11-14，早于截止；“Accepted KDD2027”是网页状态，不把未来会期当新发布日。17页正文及A–C全部读。首任务正常重访导致ASR回升是实质接缝，纳入P18；只测首任务ASR、白盒权限及条件理论限度另记。
5. 主审由D5接缝发现BAD-BOOM此前未在D2处理；不存在“D2已读”的事实，已纠正互相假设。检索 `"BAD-BOOM" "Broadening"` 命中作者GitHub，再到[PMLR正式页](https://proceedings.mlr.press/v306/zhao26bf.html)，下载其[正式PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/zhao26bf/zhao26bf.pdf)。20页正文及A–D全部读，随机切片与真实SFT方向不同、单seed、失败和utility代价记P19。p4、p20图像核验。
6. DeferBad原OpenReview `https://openreview.net/forum?id=S5JCqTJyKj`及PDF被browser challenge；搜索给出的hashed PDF下载HTTP403，未绕过访问控制。继续题名 `"Unlearn to Relearn Backdoors" arxiv` 到[作者公开arXiv](https://arxiv.org/abs/2411.14449)。原页v1 2024-11-10、v2 2024-11-25；成功下载v2 17页，正文和A–F完整阅读。p9图像核横轴为冻结层数。作为P20纳入，不声称ICLR正式录用，也不保留已解决的全文缺失。
7. 复用D1 RIPPLe/Agentic原PDF、D5 BI-BAU/Suppression/Merging原PDF及D6 Homeopathic原PDF，本域按B定位回原始页而非只读他域总结。X01–X07记录明确完整阅读责任与本域已核页；D1 C发现SP24 Orthogonality也回其原PDF核pp3–5、10–11。其他域原文件未修改。
8. 对原文定义复核：Decontam p4/p19、Poisonhide Eq6/AppC.2、FAB p8、Permanence p7、Ptrojan p4、Sleeper正文/附录等；按B纠正到报告/笔记/JSON。没有实验或代码重现，没有证明重推。

### R8逐项分诊结论（不把未读当已读）

|候选|身份/版本来源|改变本域认识与处理|
|---|---|---|
|PerDoor|[arXiv2205.13523](https://arxiv.org/abs/2205.13523)，v1 2022-05-26/v2 06-01|官方摘要是跨轮低漂移、低任务贡献坐标加非统一触发；与Neurotoxin不是完全同算法，但属于本域已覆盖的更新回避工程分支。为控制FL目录范围排除进一步整篇，**不使用10.5x等未核结果，也不声称现有方法祖先唯一**；C可挑战此范围决定。|
|Cao persistent unalignment|ACL2024 long276，arXiv2312.00027|长触发、正常数据加入安全重训直接改变干扰/保持结论，16页全文纳入P17。|
|BadGCL|arXiv2611.19656v1，2026-11-14|正常任务重访增ASR、图传播约束为直接机制接缝，17页全文纳入P18，不因图模态排除。|
|SDBA|[arXiv2409.14805](https://arxiv.org/abs/2409.14805)，v2 2025-07-30；[IEEE DOI](https://doi.org/10.1109/TDSC.2025.3593640)，online07-29|官方方法摘要层mask+topk mask，在LSTM/GPT2/T5的FL工程扩展；当前不贡献独立正常数据干预或本域需要的恢复定义。范围排除整篇，**不引用优越性或防御效果为证据**。|
|Deferred activation / DeferBad|arXiv2411.14449v2，不冒称ICLR accepted|局部层遮蔽及BN恢复具有直接历史与机制意义，已解决原入口限制并17页全文纳入P20。|
|Malice in Agentland|[arXiv2510.05159](https://arxiv.org/abs/2510.05159)，v5 2026-08-03，27页公开|官方摘要三供应链入口含预后门base，相关但主要扩张agent攻击入口；本域已全文核Agentic/PersistBD后训练主干。此轮范围排除进一步整篇，不采用80%或隐藏机制结论；不称不可得，C可要求重新纳入。|
|Merging Triggers|[ACL2026 long1113](https://aclanthology.org/2026.acl-long.1113/)|D5已全文，D2直接核pp3–4方法及p6主表，X07；自有防御trigger先植入再中和，不是一般纯clean微调。完整防御附录/效力由D5主责，D2不新作恢复保证。|
|BAD-BOOM|PMLR306 zhao26bf|主审新发现，直接研究衰退几何及保持，20页完整纳入P19；撤销此前跨域误以D2已读的假设。|

### 状态核对

修订后32条唯一文献记录：19本域FULLTEXT_READ、7跨域完整阅读+本域原文针对核证、1正文已读/附录真缺失、3范围分诊排除整篇、2综述局部发现。没有以全文数量作为停止目标。Hibernated仍缺所引附录；P-Trojan最终版未读但公开对应稿读完。专业数据库前向穷尽检索仍不可声称完成。待原B复核新增与更正，再按协议交独立C。


## 2026-11-19 C查漏交回后的原A补证（仅C1–C3）

复用C公开来源：arXiv2501.02704v3（水印，20页）与ECCV2024 paper11789/SBL2407.14738（main19页+combined supp29页）。无需重新下载或新关键词搜索。原A完成正文及全部相关supp实读，见P21/P22逐节笔记；回图核恢复/反例、AGEM/公式和有限保持图。新增异名分支“backdoor watermark persistence/restoration”已纳入；不把C下载/读过当作原A已读。C给出的未入选相邻变体不另展开。

D3→D2接口：Souly2510.07192仅核原文§4.1–4.2和AppI，D3已有全文索引责任；新增X10并将近同初始ASR/衰退不同与模拟语言对齐界限写入。新增两篇+一跨域条目待原B核验、原C复核；旧已关闭19篇无重读。Hibernated缺附录等真实限制原样保留。
