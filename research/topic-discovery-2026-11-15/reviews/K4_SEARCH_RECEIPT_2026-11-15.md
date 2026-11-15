# K4增量检索回执（2026-11-15）

范围：总登记上传之后，具体长期保护资格、最近巩固/投毒工作、两个外扩入口。仅公开概念检索，未上传私有实验原件。不是全领域穷尽清单；重复命中不计独立论文。

## 实际查询

- continual learning synaptic importance negative evidence revoke consolidation importance decay backdoor
- continual learning online EWC importance decay Memory Aware Synapses uncertainty synaptic intelligence forgetting
- backdoor defense gradient alignment clean validation meta learning sample reweighting consolidation
- replay / importance / double counting；backdoor replay consolidation；signed negative importance（早期检索批的主题概括，非逐字原始查询）
- "Anchoring Bias" "Persistent Fairness Backdoor" paper
- "backdoor" "replay" "consolidation" defense poisoning -site:researchgate.net -site:lacuna.tiptreesystems.com
- "continual learning" "contradiction" "consolidation"
- "continual learning" "negative evidence" forgetting importance
- backdoor training defense reusable holdout adaptive validation overfitting repeated model acceptance
- machine unlearning backdoor data lineage descendants synthetic data poisoning deletion
- "unlearning" "synthetic data" descendants lineage
- "unlearning" "data dependencies" generated
- "backdoor" "adaptive" "holdout" validation feedback
- "backdoor" "SIU"
- "backdoor" "Unifying Importance"
- "backdoor" "gradient noise" "consolidation"
- "poisoning" "synaptic intelligence" "importance"
- "Continual Backdoor Training in IoT/CPS"
- "backdoor" "unbiased" "importance" continual
- "backdoor" "synaptic intelligence" bias

精确SIU检索有大量缩写歧义噪声，不能以直接零命中证明新颖性。结合全称、方法机制和CL近邻追踪。

## 对判断产生影响的新增近邻

核心8项出处与阅读范围在K4_ACCEPTANCE_RULES_PROPOSAL；另补：

1. Han等，Data Poisoning Attack Aiming the Vulnerability of Continual Learning，2211.15875v2（2023版），https://arxiv.org/html/2211.15875v2 。读§3 Eq1–5/§4；目标是用后续任务投毒损害指定旧任务，EWC/SI为受害训练方式。已覆盖对CL保护的对抗利用，未在所读正文拆SI/SIU/SIB的保护组成；攻击者可读历史任务数据等权限不等本草案接纳器权限。
2. Salish等，Continual Backdoor Training in IoT/CPS，2606.14987v1，https://arxiv.org/html/2606.14987v1 。读§III–V及参考文献，正文方法将SI重要性轨迹接近正常任务作为攻击目标。不能认领首次SI重要性被用于后门。没有看到独立批去偏对照；细节与数值口径不充分，不把作者机制解释直接当独立因果证明。

## 关键回查与访问限制

- SYNERgy §3.3 Eq5明确all samples in memory buffer，Eq6是有界EMA。Advisor指出主草案B把实际回放频率与重要性目标混在一起；主审重新读原文确认。B不能按原理由推荐。
- Benzing §3.1去偏构造和§3.3负结果已读。后续主审find附录时两次超时；Advisor另行核附A的任务大小/难度/学习率影响。最终来源深度区分两者，不假称主审完整读完所有附录。
- Anchoring Bias §3.3释放模型威胁与§5.1 each-stage poisoning口径未澄清；只使用作者报告的ER结果作重叠提醒，不作有限退出证明。
- CVF EWC Done Right PDF403，使用arXiv作者版；Cao/Yang与Dwork外扩入口本轮只定位正式摘要，不给细分创新覆盖结论。
- 未验证的二手网页、聚合页与其他命中，只用于找原论文；不作为最终机制事实来源。

## 搜索改变了什么

A保留可能的分配机制问题，但撤任何“首次重要性保毒/去偏创新”措辞；B的具体默认前提被原文推翻，降级；C的矛盾/真假区分已有直接研究，不能仅改名；E1/E2并回D03/D12边界，不凑新增题数量。最后状态以Advisor意见及处置为准。
