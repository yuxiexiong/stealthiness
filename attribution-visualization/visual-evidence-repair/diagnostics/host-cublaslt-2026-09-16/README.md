# 主机 cuBLASLt SIGFPE 诊断记录（2026-09-16）

结论：gqa-h20 对 **bf16 训练**存在 cuBLASLt 启发式除零（gdb 栈钉死 `cublasLtTSTMatmulAlgoGetHeuristic`），torch 2.1.2 / 2.3.0 / 2.3.1 全部中招；微基准 GEMM 全部通过、真实 LLaVA 训练必崩；fp32 正常（3 步 loss 2.3424），fp32+TF32 数值几乎逐位（2.3435）且约 2.1× 提速——选定为 B1 训练通道（fp16-AMP 也存活但数值偏差更大，2.4156）。

历史陷阱回溯至 2026-06-29（早于本项目全部工作，dmesg 为证）。深层病根：旧 `.venv` 是队友 conda `gqa` 环境的 system-site 派生且自身无 torch，`import torch` 穿透到队友 2026-07-19 改动过的 torch 2.3.1。队友环境未作任何改动；一切绕过均在我方独立 venv 内完成。连带申明：B0 当年训练实际使用的 torch 因该泄漏结构已不可考。

- `fpe_repro.py`：GEMM 微基准矩阵（bf16/fp16/fp32 × 各形状，全部通过——排除"通用矩阵乘故障"假设）
- `train_repro.py`：参数化最小训练复现（dtype × 梯度检查点 × amp × tf32），把崩溃轴定位到 bf16 本身

相关原始日志与 gdb 栈在服务器 `/root/claude-b1-*.log`、`/root/claude-repro-*.log`、`/root/claude-b1-gdb.log`。
