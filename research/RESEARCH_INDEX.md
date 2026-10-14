# 研究材料总索引

整理日期：2026-10-12（北京时间）。保留原目录，按研究用途提供入口；“有结果文件”不代表对应假设已成立。

| 研究线 | 入口 | 材料状态 |
| --- | --- | --- |
| 跨模型 ASR | [当前入口](../attribution-microscope/supplement/jump/cross_model/README.md) | 协议、源码、阶段结果；历史完整产物保留在原分支 |
| 跃升机制 | [设计](asr-mechanism-experiment-2026-10-10/EXPERIMENT_DESIGN.md) · [toy 结果](asr-mechanism-experiment-2026-10-10/RESULTS_TOY.md) | 已有探索 toy；时间排序优势与一般机制未确认 |
| 跃升机制线盘点（10-12） | [现状与下一步建议](asr-grokking-mechanism-review-2026-10-12/REPORT.md) · [完整实验方案](asr-grokking-mechanism-review-2026-10-12/EXPERIMENT_PLAN.md) | 只读盘点与建议；附 LP 平台核对脚本，不是新实验结果 |
| 机制线规划（10-13） | [总纲：问题定义、证据与路线](asr-mechanism-plan-2026-10-13/FRAMING.md) · [下一步规划](asr-mechanism-plan-2026-10-13/PLAN.md) | 规划文档；核心瓶颈是从未干预过训练动力学；与防御线独立 |
| 推力—冲刷模型（机制线第二阶段） | [计划](asr-mechanism-pushwash-2026-10-14/PLAN.md) · [阶段 A 结果](asr-mechanism-pushwash-2026-10-14/RESULTS_A.md) | 零 GPU 事后拟合：推力随状态、冲刷存在、动量时间尺度都被确认，但标量 margin 模型留一母本只中 4/21，阶段 B 不启动；建议 A′ 在权重空间直接量 |
| A′ 权重空间测量（机制线） | [说明](asr-mechanism-aprime-2026-10-14/README.md) · [执行](asr-mechanism-aprime-2026-10-14/EXECUTION.md) | 回放三条母本直接量推力与冲刷；[结果](asr-mechanism-aprime-2026-10-14/RESULTS_APRIME.md)：门未过，因「分叉→起跳后」方向与正常学习方向重合（干净数据贡献 85–95%）；A′2 正交化后[门仍未过](asr-mechanism-aprime-2026-10-14/RESULTS_APRIME2.md)：起跳窗口权重位移八成以上来自干净数据，单方向越阈值不成立；转表示层（合同骨架待拍板） |
| 后门方向实验（机制线） | [冻结合同](asr-mechanism-direction-2026-10-15/CONTRACT_DIR.md) · [执行](asr-mechanism-direction-2026-10-15/EXECUTION.md) | 进度度量工作流：方向 d、restricted / excluded 进度量、分解、训练中 suppress / amplify；2026-10-15 01:36 已入 GPU0 队列，未开始 |
| 两部件格子实验（机制线） | [冻结合同](asr-mechanism-grid-2026-10-15/CONTRACT_GRID.md) · [执行](asr-mechanism-grid-2026-10-15/EXECUTION.md) | 机器成熟度 × 偏置强度的合成权重格子，[结果](asr-mechanism-grid-2026-10-15/RESULTS_GRID.md)：预注册「相乘」未中（两母本「相加」、一母本「测不了」）；偏置增量 W_母本−W_B2 自己携带后门且特异，机器成熟度只压低基线 |
| 平台分叉实验（机制线） | [冻结合同](asr-mechanism-fork-2026-10-13/CONTRACT_FORK.md) · [执行记录](asr-mechanism-fork-2026-10-13/EXECUTION.md) | 2026-10-14 完成：判档「驱动但混合」——逃逸需要后续毒样本推力、平台无推力会被干净数据冲回、动量不是主因；预注册的固定曝光阈值规则未中（7/12）。[结果](asr-mechanism-fork-2026-10-13/RESULTS_FORK.md) |
| 监测与防御 | [计划](asr-defense-toy-2026-10-10/TOY_EXPERIMENT.md) · [r02 结果](asr-defense-toy-2026-10-10/r02/RESULTS.md) | r01/r02 历史保留；r02 无合格投毒回退点 |
| 测量提效 | [设计](asr-defense-toy-2026-10-10/third-line-toy/EXPERIMENT_DESIGN.md) · [结果](asr-defense-toy-2026-10-10/third-line-toy/RESULTS.md) | 开发小试完成；未运行独立新轨迹确认 |
| VLM 归因与 Atlas | [LLaVA](../attribution-microscope/README.md) · [Qwen](../attribution-microscope-qwen3vl8b/README.md) | 按模型、阶段与仪器资格阅读结果 |
| 文本角色与语法 | [方向](atlas-syntax-transfer-2026-09-21/RESEARCH_DIRECTION.md) · [设计](atlas-syntax-transfer-2026-09-21/EXPERIMENT_DESIGN_three-hypotheses.md) · [报告](atlas-syntax-transfer-2026-09-21/REPORT.md) | 已观察结果、机理假设与后续设计分别记录 |
| 模型族与规模扩展 | [实验计划](atlas-model-family-expansion-2026-09-27/EXPERIMENT_PLAN.md) | 计划登记，非执行结果 |
| COCO / SAM3 实例联结 | [实验设计](coco-sam3-segmentation-2026-09-27/EXPERIMENT_DESIGN.md) | 设计材料；大预览与源包另列库存 |
| 早期修复 / 诊断 | [历史入口](../attribution-visualization/README.md) | 完成结果与负结论保留，旧预算有日期边界 |
| 原方法成本诊断 | [OA](../experiments/OA.md) · [Cordyceps](../experiments/CORDYCEPS.md) · [BadVision](../experiments/BADVISION.md) · [Grond](../GROND_PROBE_PLAN.md) | 早期适配与缺失证据保留 |
| HanLab 阅读与策略 | [阅读包](hanlab-2025-2026/README.md) · [策略材料](hanlab-paper-strategy-2026-09-16/) | 文献与路线评估，不当作本项目实验结果 |
| T2I 路线评估 | [报告](t2i-phd-decision-2026-09-15/REPORT.md) | 历史研究决策与来源记录 |
| Grokking 相关工作 | [材料目录](late-grokking-related-work-2026-10-04/) | 本地阅读记录；全文 PDF 保留本地 |
| JEV 审查与会议 | [JEV](jev-arxiv-review-2026-09-29/) · [会议材料](meeting-2026-09-24/) | 讨论与阅读材料 |
| 三线工作指南 | [中文 DOCX](asr-jump-three-tracks-2026-10-10/ASR短窗口跃升研究工作指南.docx) | 工作指南，结果以各线后续记录为准 |

本次新增材料与排除的大产物见[核对记录](repository-reconciliation-2026-10-12/README.md)。历史内容先保留；不通过移动目录或删除失败记录来整理状态。
