# D1 C独立检索记录

检索日：2026-11-19（用户环境 Asia/Shanghai）；C与A/B是不同agent。独立阶段仅读取PROTOCOL协议和D1范围，没有打开A/B报告、PAPERS、sources目录或共享检索表。协议末尾项目参照未作检索种子。

## 实际路径（按执行顺序）

1. Web查询 `backdoor learning dynamics formation gradient competition poison clean training phase transition`；`backdoor emergence training dynamics trigger target feature learning survey`。发现 ABL (NeurIPS 2021)、Backdoor Learning Curves (arXiv:2106.07214；2024期刊)、Backdoor Smoothing (2022)、Backdoor Learning: A Survey (2007.08745)、On the Difficulty of Defending Contrastive Learning (USENIX 2024)、Exploring the Orthogonality and Linearity (SP24)。把损失学习速率、训练时间与训练分布投毒强度区别列为核对重点。
2. 查询 `backdoor training dynamics formation 2025 2026 language models emergence`；`backdoor feature competition gradient clean poison learning dynamics theory`；`"backdoor" "formation" "learning" dynamics theory survey`；`"backdoor" "lazy" "rich" learning`。发现2609.36117（Why Backdooring Neural Networks is so Easy?）、2602.10382（Triggers Hijack Language Circuits）、2610.00663（QES）、ICML26 Broadening the Backdoor Basin、Safety-Efficacy Trade Off；后两者和QES先标D2/D5边界。
3. 打开2609.36117v1 HTML及USENIX论文正式页面；查询 `"backdoor" "learning dynamics" -site:reddit.com -site:sciencedirect.com` 和 `"backdoor" "gradient" "competition" formation -site:reddit.com`。发现BackdoorBench补充B.6/B.7、Backdooring Multimodal Learning (SP24)、ICML24 two-layer CNN理论，以及ResearchGate protocol-only线索（没有当作已完成实验）。
4. 精确题名检索CNN理论、USENIX对比学习论文+2025/2026前向线索、`"backdoor" "lazy" "learning" theory 2024 2025`、`"backdoor" "gradient alignment" "learning dynamics"`。确认CNN论文PMLR v235/li24at；前向发现2610.07654 Does On-Policy Distillation for Safety Pose Backdoor Risks?。检索结果中的聚合页仅用于线索，未作为证据。
5. 下载公开2609.36117 PDF（实际v2，2026-10-01）至c_sources；读正文开端与参考文献链，后向识别Souly et al 2025恒定投毒文档数、Khaddaj et al 2023强伪特征、Manoj & Blum 2021容量、Hayase & Oh 2022 lazy/kernel边界。首轮cat发生截断，**此时没有标FULLTEXT_READ**，后续须分段补齐。
6. 查询 `"Understanding Backdoor Attacks" "Dynamics"`；`"backdoor" "formation" "2026" training -site:reddit.com`；`"backdoor" "accumulation" training poison`；`"backdoor" "competition" "features" learning theory`。发现2611.22446（2026-11-17，As Student Rollouts Continue, Visual Backdoors Keep Growing），2611.09348（文本技能优化而非参数训练，边界），Not All Samples Are Born Equal (10.1016/j.patcog.2023.109512)、Adaptability Hypothesis、Shared Latent Structures (2606.07963)、统一优化框架10.1016/j.inffus.2026.104221（可能付费墙，未确认全文）。
7. 精确查询 Adaptability Hypothesis、Not All Samples ... + arxiv、As Student Rollouts ...，以及 `"backdoor" "training dynamics" "2026" -site:reddit.com -site:researchgate.net`。核验ICML23 adaptability正式页面；核验2023 clean-label论文出版方摘要（非全文）；打开2611.09348 arXiv。下载2611.22446v1并开始正文p1–3；OPD文献形成单独需追踪的近年分支。未使用搜索返回的机器书评作为结论依据。

## 独立候选发现冻结

上述阶段在系统时钟 **2026-11-18T16:38:47Z（2026-11-19 00:38:47 CST）** 冻结，**下一次文件读取才开始A/B比对**。逐条查询未另外记录秒级时钟，以上是执行顺序而非伪造的逐查询时间戳。优先核对：2611.22446、2609.36117、2610.07654、USENIX24对比学习、SP24正交/多模态、ICML24 CNN、ICML23 adaptability、2023 clean-label竞争、BackdoorBench附录、ABL、Learning Curves、Smoothing；D2/D5及文本空间边界须有去向。冻结并非全文完成或确认遗漏；是否重要新增取决于A/B覆盖和原文。

独立查询同时覆盖广义术语、异名（feature competition/adaptability/lazy-rich/OPD）、综述入口、后向引用和题名前向检索。搜索引擎索引可能滞后；不声明穷尽所有引用或世界零遗漏。后续由新增重要分支驱动补查，无固定篇数停止门槛。

## 冻结后补查（同日；与独立阶段分开）

8. 读A/B及六域索引后，下载PMLR li24at、Purdue sp24_orthogonal、arXiv2610.07654公开全文至c_sources，分段通读正文及相关附录；实际读页记录和重要发现见C_AUDIT C01–03。发现C01是训练时间理论而非静态最优解；C02存在理论正交条件与经验角度指标错位；C03的Jaccard实为固定teacher前缀上top1一致率。
9. 对新发现分支作前向/合法全文补查：`"Not All Samples Are Born Equal" "pdf"`、`"Backdoor smoothing" arxiv`、`"Exploring the Orthogonality and Linearity" 2025 2026`、`"A Theoretical Analysis of Backdoor Poisoning Attacks" 2025 2026`。核到Smoothing 2006.06721与Orthog官方主页；CNN前向命中OpenReview 2hdqPP25DM（Arcueid多trigger攻击构造），当前不把它当训练形成时间新增结论。
10. 进一步查询`"109512" "pdf" backdoor Gao`、`"Not All Samples Are Born Equal" Gao github`、`"backdoor" "robust features" "Gao" arxiv`、`"2hdqPP25DM"`。找到作者官方Effective_CLB仓库但未得论文全文，保留受限；相关语音robust-feature attenuation为下游攻击构造，未直接扩为D1全文证据。三项确认缺口及剩余链已写C_AUDIT并交主审，不自改REPORT。


## 冻结后第二轮复核与独立补查（2026-11-19；00:40–00:53 CST）

- 先核最新版D1 REPORT/REVISIONS/B_REVIEW§七，再读shared/CROSS_DOMAIN。一次组合读取与四query返回超出上下文，未把不可见结果当已检查；随后缩小批次重查。
- 实际查询：`"A Theoretical Analysis of Backdoor Poisoning Attacks" dynamics`；`"Understanding Backdoor Attacks through the Adaptability Hypothesis" 2026`；`"backdoor" "training dynamics" "2026" emergence`。前两条返PMLR/OpenReview及作者页，未发现足以改变已核理论条件的新反证；第三条发现arXiv2607.06643、LPS正式2026版、Lilith等。
- 打开arXiv2607.06643 abstract/HTML；公开PDF经curl下载成功（无绕过paywall），pdftotext后逐段1–240、241–510、510–929行完整读15页。本版无附录。识别PPI/PRI状态规则外给与阈值下仍可有残余进展，确认为需短段整合的D1模型家族接口C04，D2/D5另路由。
- 查询`"Boosting backdoor attack with a learnable poisoning sample selection strategy" pdf`，找到arXiv2307.07328、OpenReview旧稿及Neurocomputing2026 DOI10.1016/j.neucom.2026.134024；出版方正文open返回403。仅元数据/摘要筛选，送D3样本选择，不声称2026全文已核。
- 查询`"Does On-Policy Distillation for Safety Pose Backdoor Risks"`，返回已读2610.07654及其聚合索引；聚合描述未作证据。未发现新的、必须推翻当前OPD受限解释的第一方材料。
- D6接缝：回原始EM2502.17424v7 §4.7 pp9–11、AppC.5 pp26–28，确认每10步/5seed、ID先分化、两类logprob、非grokking对照。只标相关节复核，全文主责仍D6。需D1短段C05，不是trigger后门本身。
- D5接缝：回Honeypot2310.18633 AppC pp16–17；完整PLM与浅层probe的学习顺序不能混。DBD由D5全文记录界定loss排序的训练程序。已有D1快学反例覆盖总论，故仅补例路由、不阻断新大轮。
- 本轮结果与每项A/B要求见C_AUDIT第二轮。首轮独立冻结时间不变；本轮没有伪装成未接触A/B的新完全独立阶段。四类既存全文/附录限制保留。尚待C04/C05聚焦处理后最终复核。


## 末轮修订核验（2026-11-19 00:55 CST）

原A补C04/C05、原B完成§八13项后，同一C定点读取REPORT两段和主表、REVISIONS/B_REVIEW，回看2607.06643§4.2与原始EM§4.7的关键原文。没有重新网络检索、没有把既有全文重算为新阅读；此前独立补检索路径承接。C04/C05已关闭，末轮未发现改变结论的重要遗漏，C_AUDIT顶层更新为PASS_WITH_SOURCE_LIMITATIONS。四类来源限制及非绝对零遗漏声明保留。本域C审计有界结束。
