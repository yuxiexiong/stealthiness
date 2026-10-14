# 两部件格子实验（机制线）

| 文件 | 用途 |
|---|---|
| [CONTRACT_GRID.md](CONTRACT_GRID.md) · [contract.json](contract.json) | 冻结设计：机器成熟度 × 偏置强度格子，判类、判档、预注册预测「相乘」 |
| [EXECUTION.md](EXECUTION.md) | 执行记录 |
| [code/](code/) | `grid_states.py`（从分叉点状态回放取机器状态与偏置增量）、`grid_eval.py`（19 次前向评测）、`grid_judge.py`（CPU 判档）、`grid_common.py`（冻结判类）；r02 与分叉实验模块逐字节复用 |
| [jobs/](jobs/) · [manifest.json](manifest.json) | GPU0 独立队列 7 任务（前缀 `204cmgd`） |
