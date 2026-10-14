# 两部件格子实验：执行记录

2026-10-15。**状态：已完成（7 任务 0 失败，约 01:4x）。判档「不一致」，预注册「相乘」未中：两条母本判「相加」，一条「测不了」。见 [RESULTS_GRID.md](RESULTS_GRID.md)。**

- 00:30 上传到 `/workspace/cross-model-asr/20261015_asr_mechanism_grid_gpu0_r01`；12 个源码 SHA、合同 SHA 与 manifest 一致，7 个冻结模块与分叉实验 manifest 一致；worker 387002 存活并记录；服务器 CPU 测试通过；入队时 GPU0 空闲，磁盘剩 80 GiB。

CPU 测试：`python3 -m unittest discover -s code -p 'test_*.py'`（标准库）。七个任务：每母本 states（GPU）→ grid（GPU），再 judge（CPU）。
- 三条母本 states + grid 各约 15 + 8 分钟；B0 回放到 t_jump 哈希全部与母本一致；结果同步到 [results/](results/)，合成适配器留服务器。
