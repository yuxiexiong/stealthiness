# 执行记录：修订版实验已上卡

核对快照：2026-10-20T12:34:23.306461+08:00。这是运行状态，不是科学结论。

- 原服务器PyTorch 2.6.0：33项CPU测试通过，零跳过。
- 修复版真实模型GPU验收通过：loss/梯度范数/逐行loss、LoRA权重、Adam、RNG逐项一致；评估不改变训练状态。
- 54个正式GPU任务均已提交：B44先，全部完成后自动接A10，结束自动执行CPU汇总。无新臂、无自动重试、无时间硬截止。

正在运行：
- GPU0：`221icd_010_B_s1105_R`
- GPU1：`221icd_032_B_s1106_R`

训练日志进度：
- `B/s1105/R`：训练绝对步 133，最近曲线相对步 95。
- `B/s1106/R`：训练绝对步 92，最近曲线相对步 30。

- 当前本批正式任务失败数：0；尚无完成的正式科学结果。
- r01首次GPU验收失败约12.1秒，根因是optimizer状态别名，已先用CPU回归测试复现后深拷贝修复。旧r01目录、原源码、失败日志完整保留，未作为科学阴性，也未启动正式训练。
- r02 GPU验收26.82秒；完整V60读数14.17秒。一次短验收不足以给精确ETA，暂保留累计22–32GPU小时的保守规划，待首批完整臂计时校准。

实际源码：`/workspace/cross-model-asr/stage_icd_20261020_r02`。
实际输出：`/workspace/cross-model-asr/20261020_intervention_confirm_decompose_r02`。
队列：`/workspace/claude-jump/jobq_gpu0`，前缀`221icd_`。
完整任务与状态回执见同目录QUEUE_SUBMISSION.json、AUTHORIZATION.json、GPU_ACCEPTANCE.json；本地见DEPLOYMENT.json。

冻结合同和CPU验收反映各自生成时的状态；本文件及后续运行产物反映实际执行。
