# 平台分叉实验（机制线）

| 文件 | 用途 |
|---|---|
| [CONTRACT_FORK.md](CONTRACT_FORK.md) · [contract.json](contract.json) | 冻结设计：母本、十种分叉、曝光预测器、判档、预注册预测 |
| [EXECUTION.md](EXECUTION.md) | 执行状态 |
| [code/](code/) | `fork_common.py`（纯逻辑）、`fork_run.py`（母本 / 预测 / 分叉 / 收尾）、`fork_prepare.py`、`deploy_fork_cpu.py`；r02 冻结模块 `toy_data.py`、`toy_eval.py`、`measurement_kernels.py`、`native/*` 逐字节复用 |
| [jobs/](jobs/) | GPU0 独立队列的 35 个任务规格（前缀 `201cmfk`） |
| [manifest.json](manifest.json) · [make_manifest.py](make_manifest.py) | 源码 SHA、借用收据、服务器路径 |

CPU 检查：`python3 -m unittest discover -s code -p 'test_*.py'`（纯标准库，不需要 torch）。本目录不启动训练或队列。
