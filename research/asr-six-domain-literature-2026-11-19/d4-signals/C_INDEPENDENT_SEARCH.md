# D4 C独立检索冻结记录

检索日：2026-11-19（Asia/Shanghai）。角色C；冻结前只读PROTOCOL.md及技能工具说明，未读A/B、PAPERS、REVISIONS、共享索引、sources或项目证据。此文件先于上述材料比对写入。独立检索是查漏候选生成，不把搜索摘要标作FULLTEXT_READ。

## 范围与入口

前置信号、训练期监测、训练前风险与评估；严格区分：已植入后的检测/训练数据筛查/自然训练时序预警/推理运行时检测；已知与未知trigger、target；行为阈值与连续分数；终点与瞬态。使用nature-academic-search多源流程及nature-downloader公开全文路线；专业学术MCP不可用，实际入口web搜索→ACL/arXiv/USENIX/CVF/出版方及作者公开源。没有使用引用数过滤，因此“最新/低引用”指新近研究覆盖，不伪报引用量。

实际查询组（保留原词，均为公开一般问题）：
- backdoor learning dynamics early detection training loss poison monitoring survey
- backdoor pre training risk prediction detecting backdoors before training 2025 2026
- backdoor transient training checkpoints evaluation early warning
- backdoor detection "training dynamics" early learning；"before training" loss；"early warning" training；"transient" "training" detection
- backdoor detection training dynamics ABL ASD CUBE SEEP loss gradient 2025
- backdoor "risk" "training-free" detection；training monitoring "2026" early
- "Rethinking Backdoor Detection Evaluation"（含2026前向发展查询）
- backdoor "training dynamics" detection "2025" "2026"；"risk assessment" training；"early" "monitoring" neural training detection
- "Testing Whether Internal Backdoor Precursors"（含全标题与作者线索）
- "SEEP" "backdoor" "confidence"（及2026前向发展）；"Training-Time Backdoor Sample Filtering via Loss and Uncertainty Dynamics"
- backdoor "training dynamics" "survey"；"precursors" "Assaf"；"training-free" "poisoned" detection；"transient" "checkpoints"
- backdoor "survey" "2025" detection LLM；"Z-defense"；"before training" "detect" NLP statistical
- "PKAD" backdoor ACL 2024；"BadWindtunnel" aclanthology；"backdoor" "Z-Score" "2023" He

## 冻结候选及链条

以下均是独立发现的待比对线索，只有完成正文与相关附录后才能作重要遗漏结论。

| 候选/稳定定位 | 独立检索意义/待核点 |
|---|---|
| SEEP, 10.1162/tacl_a_00684; https://aclanthology.org/2024.tacl-1.55/ | 多epoch inverse-confidence及传播；是否是真正前兆或先训后筛；后向ABL、Data Maps、Z-defense；前向GMS/BadWindtunnel |
| PKAD, 10.18653/v1/2024.findings-emnlp.335; https://aclanthology.org/2024.findings-emnlp.335/ | 冻结PLM训练前筛查；不需要干净参考数据不等于任意未知target风险预测 |
| BadWindtunnel, 10.18653/v1/2025.findings-acl.482; https://aclanthology.org/2025.findings-acl.482/ | 高噪声模拟训练信号、target识别；自然训练监测与另建诊断训练区分 |
| Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation (EMNLP2023; Z-defense) | 数据统计训练前筛查，SEEP与BadWindtunnel后向链 |
| Rethinking Backdoor Detection Evaluation for Language Models, arXiv:2409.00399 / https://aclanthology.org/2025.emnlp-main.318/ | training intensity非单调检测失败；回溯检测与前兆区分 |
| ToxScreen: Detecting Whether an LLM Has Been Poisoned, arXiv:2607.26849 | 新近检测基准，未知trigger/target信息条件 |
| DeBackdoor, arXiv:2503.21305 / USENIX Security2025 | pre-deployment、有限数据、模板先验的边界 |
| Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models, arXiv:2509.05318 / 10.1016/j.neunet.2025.108025 (NETE) | 训练前与推理筛查的可迁移边界 |
| Testing Whether Internal Backdoor Precursors Precede Behavior: A Causal, Power-Matched Protocol (Aug2026; https://www.researchgate.net/publication/411771060_Testing_Whether_Internal_Backdoor_Precursors_Precede_Behavior_A_Causal_Power-Matched_Protocol) | 直接命中时间前兆；摘要明说只有protocol/smoke test，不应当成已证明内部先于行为；须查作者原文与证据状态 |
| Signals in the Dark: Looking Inside Model Backdoors; https://huggingface.co/blog/assafpet/signals-in-the-dark | 作者技术报告，静态trigger identity与因果使用的混淆；非正式论文待核 |
| Training-Time Backdoor Sample Filtering via Loss and Uncertainty Dynamics, 10.1109/MRAI70020.2026.11621182 | 2026低引用路线，可能高/低poison下学习顺序相反；仅二手书目线索，须一手全文 |
| Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models, arXiv:2604.23460 | armed-but-benign推理状态，不应冒充训练形成前兆 |
| Detecting Chain-of-Thought Hijacking in Backdoored Multimodal Large Reasoning Models, arXiv:2610.16598 | 2026运行时推理监测、清洁阈值；不是训练预警 |
| TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting, ICML2026; https://proceedings.mlr.press/v306/nguyen26g.html | 任务迁移导致loss信号退化，需看是否同类补例或重要边界 |
| ONION, 10.18653/v1/2021.emnlp-main.752 | 运行时过滤与训练前清洗不可混同；经典对照 |
| Backdoor Learning: A Survey, arXiv:2007.08745 / 10.1109/TNNLS.2022.3182979 | 分类及后向链入口 |
| A Survey of Recent Backdoor Attacks and Defenses in Large Language Models, TMLR2025; https://openreview.net/forum?id=wZLWuFHxt5 | 最新综述入口（2025） |
| Backdoor threats in large language models—a survey, 10.1007/s11432-024-4351-3 | 2025综述入口，公开PDF https://scis.scichina.com/en/2025/191101.pdf |
| A Survey on Backdoor Threats in LLMs: Attacks, Defenses, and Evaluation Methods, arXiv:2502.05224 / 10.53941/tai.2025.100003 | 评估与检测信息假设补查 |

已排除：软件xz/网络入侵“backdoor monitoring”、硬件瞬态fault（不等于行为瞬态）、只有模型去毒/鲁棒聚合而无新增检测辨识点的同类防御目录、媒体安全报道。搜索返回2611.22446视觉OPD条目暂不采用：发布日期/本域关联仍待核，不能把搜索today当已核时间。

## 冻结后执行规则

下一步才读D4 A/B/PAPERS/REVISIONS。已覆盖和同类补例停止；优先将影响“已做过什么”的真实缺口取全文核对。后向/前向链通过上述种子定位后进一步按真实references核实。核心全文不可得则明确受限；未全文不判重要科学遗漏。首轮审计结束后交原A补证、原B核验，C保留复核角色，不替代A/B写结论，不启动实验/GPU/发布。
