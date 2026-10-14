# 两部件格子实验（机制线）

| 文件 | 用途 |
|---|---|
| [CONTRACT_GRID.md](CONTRACT_GRID.md) · [contract.json](contract.json) | 冻结设计：机器成熟度 × 偏置强度格子，判类、判档、预注册预测「相乘」 |
| [RESULTS_GRID.md](RESULTS_GRID.md) · [results/](results/) | **结果：相乘 ✗**——偏置增量自己携带后门，机器成熟度只压低基线；随机方向无效；找到了特异的后门方向 |
| [EXECUTION.md](EXECUTION.md) | 执行记录 |
| [code/](code/) | `grid_states.py`（从分叉点状态回放取机器状态与偏置增量）、`grid_eval.py`（19 次前向评测）、`grid_judge.py`（CPU 判档）、`grid_common.py`（冻结判类）；r02 与分叉实验模块逐字节复用 |
| [jobs/](jobs/) · [manifest.json](manifest.json) | GPU0 独立队列 7 任务（前缀 `204cmgd`） |
