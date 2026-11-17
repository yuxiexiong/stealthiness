# M045 正常累积批调序实验

当前范围：按用户2026-11-17授权执行M042/M044甲案。完整方法见[IMPLEMENTATION](IMPLEMENTATION.md)及[F0](F0.json)。独立复核见[最终Advisor记录](../research-mentor/reviews/M045/FINAL_EXECUTION_REVIEW.md)。不自动扩背景、不修改论文正文。

2026-11-17 03:45北京时间已完成。六臂训练、108份正式读数和最终验收齐全，实际6.3416 GPU小时。正式阶段双卡并行，现已退出并释放资源。真实评分主预测方向/幅度及对照增量未通过；详见[完成回执](REPORT.zh-CN.md)。不追加实验。

远端本轮专属目录：`/workspace/asr-ordering-20261116-r01`。只读入口：STATUS.json、LEDGER.json、logs/、progress/；阶段产物F1.json、S1_RESULT.json、S1D_RESULT.json、F2.json、RESULT.json。科学停止保留相应阶段文件；工程失败另有FAILURE及REPAIR记录。旧输入目录不变。

新代码仅在bundle；CPU回归使用历史amic环境（torch2.4.0/transformers4.45.2/accelerate0.34.2）。每次代码修复归档原代码/回执，核对变更、重跑受影响CPU检查，再更新绑定。原配恢复不因纯调序封装修复重复。新GPU任务由单个finite supervisor派发，最多两张空闲卡、每卡一个worker；外来任务不共卡。30分钟监控仅针对本次运行，结束即退出。

原规划8.5–12GPU小时，最终实耗6.3416GPU小时（含失败成本），不是硬帽。CPU、等待、墙钟不同于占卡小时。该目录receipts保存已拉取的时间点回执；最终结果已归档于completed-evidence，不能把启动快照当完成证据。
