# 文生图分支选题审查：范围与判定规则

日期：2026-09-15。对象：现有 VLM 归因—诊断—干预研究能否形成有独立贡献的文生图分支。这里将用户所说 submission 理解为相关投稿/会议论文；不把读过论文等同于读过其审稿意见。

## 已授权范围

只读代码与已有结果、下载合法公开论文、全文重点审读、重分析和形成研究建议。本轮不训练、不启动 GPU 实验、不修改实验实现、不改变远端任务。研究目录里的分析文件是本轮产物。

## 三个必须分别回答的问题

1. 现有实验实际建立了什么现象？哪些是方法相对优势，哪些仅是可观察性或工程流程？
2. 文生图是否存在对应且尚未被强方法解决的实际问题？近邻是否已经覆盖机制、控制手段和验证端点？
3. 新诊断信息是否有可能带来独立的决策和最终生成收益？若只在正确 donor、穷举真值或人工挑例下成立，应如何降级结论？

## 证据分层

- 已有原始实验的只读复算，不称为新执行实验。
- 正式会议版本优先于更早匿名稿；预印本明确标识，不推定已录用。
- 论文全文：记录读过的正文、实验、限制和附录锚点；不声称逐字读完参考文献、每个附录样例。
- 论文作者报告与本地重现分开；本轮没有重现 T2I 结果。
- 本轮搜索是有针对性的近邻审查，不是覆盖全部文献的系统综述；搜索未命中不能证明首创。
- 不将 synthetic marker / mixed-state corruption 直接称作真实投毒机制；不将激活修补视为参数修复。

## 搜索与筛选

核心族：attention-guided correction、localized diffusion editing、causal localization、concept attribution、higher-order/conditional interactions、fine-grained generation evaluation。重点 2025–2026，并保留决定问题边界的 2023–2024 原始工作。

本轮根节点反证检索包括：

- `"DeLeaker" "Dynamic" ICLR 2026`
- `"When Attribution Patching Lies" 2606.09899`
- `"GenEval2" 2512.16853`
- `"Does Localization Inform Editing" 2301.04213`
- `"text-to-image" "semantic leakage" "causal" intervention`
- `"diffusion" "attention" "beneficial" "semantic leakage"`
- `"text-to-image" "predict" "intervention" "preservation"`
- `"text-to-image" "interactions" "leakage" "DeLeaker"`

主要来源为会议官方全文、arXiv、作者项目/代码。搜索摘要用于发现论文，不替代全文。OpenReview 验证页不绕过；能从会议独立公开来源取得的论文照常阅读。未取得的审稿意见不纳入证据。

## 挑战检查点

1. 初始：不继承上一轮“选择性修复很适合”的结论，先问 VLM 原始结果是否真的展示目标与保护事实的取舍。
2. 中途：主图完整测量菜单没有该取舍；DeLeaker、PC-Edit 等覆盖大量原建议。相应撤回强立项依据。
3. 终稿前：让独立审读者从近邻覆盖、计算与评估可行性、额外诊断信息价值三个角度否证候选；记录在 REVIEW.md。

## 决策原则

只有“真实且重要的剩余失败 + 可用动作确实能解决 + 诊断在留出条件下有额外价值 + 最终输出收益 + 最近邻差异”同时成立，才支持正式立项。当前允许的结果包括否决原命题、保留有门槛的预研、或发现必须补证据后再决定；不为了给出好消息而把假设包装成结论。
