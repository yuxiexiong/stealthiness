# 后门方向实验：执行记录

2026-10-15。**状态：已部署并提交 GPU0 队列（北京时间 01:36，14 任务）；未开始。**

- 01:36 上传到 `/workspace/cross-model-asr/20261015_asr_mechanism_direction_gpu0_r01`；15 个源码 SHA、合同 SHA 与 manifest 一致，7 个冻结模块与分叉实验 manifest 一致；worker 387002 存活并记录；服务器 CPU 测试通过；入队时 GPU0 空闲，磁盘剩 77 GiB。

CPU 测试：`python3 -m unittest discover -s code -p 'test_*.py'`（标准库）。14 个任务：每母本 states → eval → suppress / amplify；全部 eval 后 cross（GPU）；最后 judge（CPU）。
