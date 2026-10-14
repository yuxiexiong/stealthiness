# 平台分叉实验：执行记录

2026-10-14。**状态：已完成。35 任务全部 done、0 失败；判档「驱动但混合」，预注册「曝光驱动」未中。见 [RESULTS_FORK.md](RESULTS_FORK.md)。**

## 已完成（本地，无 GPU）

- 合同 [CONTRACT_FORK.md](CONTRACT_FORK.md) / [contract.json](contract.json)；源码 [code/](code/)；CPU 测试 22 项通过（纯标准库）。
- 零 GPU 只读核对（只为写预测器）：防御线变体 A 的 A1 日志 seed 1102 / 1103 显示毒行消费 loss 随 lr 加权曝光单调下降、起跳都在 E ≈ 6.6–6.9；变体 B 的 seed 1101 曝光更高（13.6）且 loss 长期停在 1.1–1.9，与正常学习竞争。数字不进本线结果。
- `manifest.json`、`jobs/` 由 `make_manifest.py` 生成。

## 部署与入队（2026-10-14，用户要求助手直接启动 worker 并入队）

- 01:49:41 启动 GPU0 独立队列 worker：`/workspace/claude-jump/jobq_gpu0`，pid 387002（与 GPU1 的 worker 3052130 同一份 worker.py，不同队列根）。
- 上传到 `/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01`；清理 macOS 元数据文件后 `source_tree_sha256` = `a30507f4…` 与 manifest 一致；3 个借用收据 SHA 一致；服务器环境导入全部模块正常。
- 01:50 `deploy_fork_cpu.py` 通过：记录 worker pid；碰撞检查通过；22 项 CPU 测试通过、0 skip、CUDA 未初始化；四个切分再编码与 r02 收据一致（train 20000 / validation 60 / confirmation 64 / oldqa 60，毒行 200）。
- 01:52 35 个任务复制进 `jobq_gpu0/jobs/`，全部 queued；入队时 GPU0 被 jevrep 占用约 33 GiB，worker 等连续 10 分钟空闲。回执 `queue_receipt.json`。

## 运行与完成

- 02:07 GPU0 空出后开始；三条母本 02:07–03:5x 完成，起跳步 48 / 72 / 72，回放到分叉点全部精确；之后 CPU 预测任务写出 `predictions.json`，再依次跑 30 个分叉；10:23:45 收尾任务 done。
- GPU 墙钟合计 8.26 小时。B0 三条逐行 loss 与母本差 0.0。
- 后台监控在 14:14 才报告完成（晚约 4 小时）。安全数值同步到 [results/](results/)，服务器 SHA 列于 `results/SERVER_SHA256.txt` 并本地核对一致；适配器与完整状态留服务器。

## 后续（未做）

1. ~~启动 worker、上传、部署、入队~~ 已完成；之后只等待，不碰 GPU1。
2. 完成后：`900_finish_cpu` 写 `RESULTS.md`；同步安全数值与 Markdown，`private/` 不发布。
