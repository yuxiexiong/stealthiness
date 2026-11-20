# Gap 2 实验与验证状态

截至 2026-11-20，M058 原始实验及其逐题回顾性分析均已完成。新三种子的快速验证已确定统计设计，进展到恢复预检；尚无新轨迹的验证结果，也未提交 GPU 队列任务。

| 工作 | 状态与材料 |
|---|---|
| M058 既有实验及误差分析 | [逐题报告](REPORT.zh-CN.md)、[误差核算](ERROR-AUDIT.zh-CN.md)、[统一平移核查](FLAT-SHIFT.zh-CN.md)、[四状态漂移对照](DRIFT-FACTORIAL.zh-CN.md) |
| 新三种子实验设计 | [决策草案与 Advisor 复核](../research-mentor/reviews/two-gaps-2026-11-20/GAP2-FAST-DECISION.md)记录初版；[计数判读规格](COUNT-EVALUATION.zh-CN.md)采用最终的 1,000 个固定划分 |
| CPU 复算 | [固定评价题单](evaluation-splits.json)、[划分敏感性复算](check_split_sensitivity.py)、[四状态漂移复算](check_drift.py)。脚本核验的是已有 M058 数据。 |
| 服务器资产校验 | [核验记录](RESTORE-CHECK.zh-CN.md)。三种子各两份检查点，共 126 项文件通过大小和 SHA-256 核对。 |
| GPU 就绪状态 | 第 300 步恢复后的第 310 步状态与历史参照的运行等价性尚未验证；三轨迹从排队到结果表的端到端流程尚未验收。GPU 空闲本身不足以判定可启动。 |

统计规格及题单的版本以本次提交为准。每个训练种子分别评价，同一批题的反复划分不能计作多个独立训练重复。
