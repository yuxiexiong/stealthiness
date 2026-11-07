# ASR Rising 整合稿：ICML 2026 独立模拟评审包

2026-11-07；四位审稿人均为同模型系列、独立上下文的 AI 模拟，不是会议正式评审。全部完整评审同一冻结提交，先锁定定性意见，再锁定评分。没有新训练、推理、服务器或 GPU 任务；未改原稿、合同或结果。

**结论：四位均弱拒，但认可条件性因果核心。** P2 共同后续是最强新证据；跨模型现象与 Qwen GATE 提供互补支持，不能替代 P2 的同协议确认。近邻定位与核心确认是当前主要缺口。

## 建议阅读顺序

1. [综合判断：现在哪里、比上轮补了什么](PANEL_SYNTHESIS.md)
2. [唯一优先实验与停止条件：两臂共同后续确认，仅建议](NEXT_EXPERIMENT.md)
3. [协调者原始资料核查与近邻对照](COORDINATOR_SOURCE_CHECKS.md)

## 四份完整评审，原文未修改

| 评审 | 完整意见 | 评分 | 定性锁定 | 评分锁定 | 实际核查记录 |
|---|---|---|---|---|---|
| R1 | [报告](R1/qualitative.md) | [JSON](R1/scores.json) | [锁](R1/QUALITATIVE_LOCK.json) | [锁](R1/SCORE_LOCK.json) | [阅读](R1/READING_LOG.json)、[算术](R1/ARITHMETIC_AUDIT.json) |
| R2 | [报告](R2/qualitative.md) | [JSON](R2/scores.json) | [锁](R2/QUALITATIVE_LOCK.json) | [锁](R2/SCORE_LOCK.json) | [audit](R2/audit.json) |
| R3 | [报告](R3/qualitative.md) | [JSON](R3/scores.json) | [锁](R3/QUALITATIVE_LOCK.json) | [锁](R3/SCORE_LOCK.json) | [阅读](R3/READ_AUDIT.json)、[算术](R3/ARITHMETIC_CHECKS.json) |
| R4 | [报告](R4/qualitative.md) | [JSON](R4/scores.json) | [锁](R4/QUALITATIVE_LOCK.json) | [锁](R4/SCORE_LOCK.json) | [核查](R4/evidence_checks.json) |

没有合并或重写少数意见：R2 优先建议参数与训练器状态四格实验；另外三位优先建议 P2 第二种子两臂确认。协调者说明选择理由，不按平均分估计录用概率。

## 固定输入、原始证据与旧判断

- [本轮正文](../asr-rising-integration-2026-11-07/03_正文整合稿.md)、[跨模型材料](../asr-rising-integration-2026-11-07/01_跨模型证据.md)、[两类干预及证据边界](../asr-rising-integration-2026-11-07/02_因果证据与边界.md)、[完整证据入口](../asr-rising-integration-2026-11-07/README.md)。冻结 commit：`c9a115eb8872c7108361f6751231b81d6b5d2a42`，52 文件身份见 [INPUT_MANIFEST](INPUT_MANIFEST.json)。
- 上轮：[11 月 5 日机制稿](https://github.com/yuxiexiong/stealthiness/blob/e9fcc761a244a052fbb29db9bdefae1409a356a9/research/two-paper-package-2026-11-05/originals/review/inputs/01_机制论文初稿.md)、[旧 panel 综合](https://github.com/yuxiexiong/stealthiness/blob/e9fcc761a244a052fbb29db9bdefae1409a356a9/research/two-paper-package-2026-11-05/originals/review/PANEL_SYNTHESIS.md)。旧材料只由协调者作比较，未提供给本轮审稿人。
- [官方评分依据](VENUE_BASIS.md)、[相同评审任务](REVIEW_BRIEF.md)、[独立性与执行协议](PANEL_PROTOCOL.json)、[旧材料哈希](PREVIOUS_REVIEW_MANIFEST.json)、[协议验收](PANEL_VALIDATION.json)。
- 协调者零 GPU 重算：[主要数值](NUMERIC_CHECK.json)、[实际 LR 与非有限梯度](LR_AUDIT.json)。这是现有数据重算，不是训练或推理复现。

审稿人已明确各自阅读范围。提交包没有完整收录所有训练源码、行级消费及逐探针原件，不能把报告中引用的历史 PASS 当作本轮从头验收。这是复现材料的缺口，不等于相关原件在项目中不存在。
