# 机制与防御双稿：四位独立AI审稿及HanLab对齐

2026-11-05。完整审阅两份各13页冻结稿，核对原始证据；没有新增实验、修改原稿或假冒外部真人审稿。

优先阅读：

- [综合判断与分合建议](PANEL_SYNTHESIS.md)
- [HanLab经验逐项对齐：立住与solid](HANLAB_ALIGNMENT.md)
- [已复核的具体问题与勘误](FACT_CHECKS_AND_ERRATA.md)

原始独立评审：

| 审稿人 | 定性报告 | 独立评分 | 机制/防御推荐 |
|---|---|---|---|
| R1 | [全文](R1/qualitative.md) | [评分](R1/scores.json) | 弱拒 / 拒 |
| R2 | [全文](R2/qualitative.md) | [评分](R2/scores.json) | 弱收 / 弱拒 |
| R3 | [全文](R3/qualitative.md) | [评分](R3/scores.json) | 弱拒 / 弱拒 |
| R4 | [全文](R4/qualitative.md) | [评分](R4/scores.json) | 弱拒 / 弱拒 |

每人均独立读两篇完整论文，先锁定定性报告，再锁定评分；同一空白上下文任务，没有专科分工或互相通知意见。各目录的`QUALITATIVE_LOCK.json`列出实际核对文件与复现边界，`SCORE_LOCK.json`记录评分哈希。统一任务见[REVIEW_BRIEF](REVIEW_BRIEF.md)，现行会议标准见[VENUE_BASIS](VENUE_BASIS.md)。

冻结输入在`inputs/`，原文件与副本哈希见[INPUT_MANIFEST](INPUT_MANIFEST.json)。HanLab总结副本及原论文来源在`hanlab_sources/`。已有数字的协调者CPU复算见[COORDINATOR_NUMERIC_CHECK](COORDINATOR_NUMERIC_CHECK.json)。最终输入、锁定报告与综合文件验证见[PANEL_VALIDATION](PANEL_VALIDATION.json)。

分数用于作者侧模拟判断，不是正式会议意见或录用概率。同模型的独立代理仍可能有相关偏差；已有数字复算不等于重新训练复现。HanLab对齐是从论文与本地总结提炼的研究经验，不是官方量表。综合报告保留R2的边缘接收意见和其他人的异议，未投票平均为一个录用概率。
