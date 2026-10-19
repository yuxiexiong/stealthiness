# 补充实验：已部署与入队（无绝对截止）

> **2026-10-19 当前定位：** 现行目标与路线以[训练期防御统一口径](../TRAINING_DEFENSE_GOAL.zh-CN.md)为准。本文原结果、冻结判据和当时讨论保留；历史优先级、停止建议和补跑清单不再作为当前执行指令。导师提出的参数／层／更新方向路线未被已测粗分区阴性完全证伪，允许组合与取舍；Curfew可加强或重构，LT是强对照，断供仅为备选。本次不追加实验或算力授权。

> **批次状态：** 此42项批次现已全部完成，实际约26.836 GPU小时。下面提交、排队与验收文字记录部署当时状态，不是实时进度，不构成重复提交或新阶段授权。

2026-10-19，北京时间。42项固定候选已提交至原有双卡队列，排在124al实验之后；入队不等于已运行。原实验代码、判据、队列调度器和视频任务均未修改。

| 类别 | 数量 | 终点 |
|---|---:|---|
| 原R/A/MLP/B/XR三母本完整训练 | 15 | 绝对1250步，严格重播原160步前缀 |
| 同r02的A0、Curfew、LT20、LT50 | 12 | 从0到1250步 |
| GATE/UP/DOWN三母本 | 9 | 分叉后160步 |
| 两组预锁随机12层MLP三母本 | 6 | 分叉后160步 |

新作业最多32 GPU小时；为已在跑的原作业预留4 GPU小时。没有绝对截止时间。尾部预计无法在剩余算力预算内完成则延期；失败不重试；执行保真失败阻断同seed后续，不按结果输赢筛选。

13项CPU测试全部通过，0跳过、CUDA未初始化。另一个微型CPU整循环测试验证：一次16行策略决策、全保留与原算术逐位相等、全mask时权重及Adam状态不变。真实Qwen GPU保真由排在首位的3支R重播检查，CPU测试不充当GPU结果。

所有候选与顺序均未根据当前allocation效果选择，只查看了其合同、源码、完成状态和耗时。实际adapter shapes确认B、XR、GATE、UP、DOWN、XR2、XR3均更新9,437,184个标量参数，可比较相同预算在不同层与投影角色上的分配。

服务器目录：`/workspace/cross-model-asr/20261019_allocation_overnight_r01`。

- 自动报告：`reports/REPORT.zh-CN.md`
- 逐项状态、运行成本与延期：`budget.json`
- 入队作业：`125ov_01_lane0`、`125ov_02_lane1`、`125ov_99_summary_cpu`
- 实际入队UTC：`2026-10-18T17:06:42Z`
- 冻结manifest SHA256：`98fe9be2edcfa4bf3be148e05d1c9bbdfa35beb0cc94abf263a0f84e2850c3f3`

CPU验收、入队记录与代码身份见同目录的`cpu_tests_passed.json`、`cpu_integration_passed.json`、`DEPLOYMENT.json`及`manifest.json`。


2026-10-19 amendment: user cancelled the absolute calendar deadline. No 18:51 cutoff and no replacement cutoff. The 32 GPUh new-work / 36 GPUh combined budget, fixed tasks, priorities and acceptance criteria remain unchanged. This amendment supersedes the historical time-window wording above.
