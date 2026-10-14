# 两部件格子实验：执行记录

2026-10-15。**状态：已部署并提交 GPU0 队列（北京时间 00:30，7 任务）；未开始。**

- 00:30 上传到 `/workspace/cross-model-asr/20261015_asr_mechanism_grid_gpu0_r01`；12 个源码 SHA、合同 SHA 与 manifest 一致，7 个冻结模块与分叉实验 manifest 一致；worker 387002 存活并记录；服务器 CPU 测试通过；入队时 GPU0 空闲，磁盘剩 80 GiB。

CPU 测试：`python3 -m unittest discover -s code -p 'test_*.py'`（标准库）。七个任务：每母本 states（GPU）→ grid（GPU），再 judge（CPU）。
