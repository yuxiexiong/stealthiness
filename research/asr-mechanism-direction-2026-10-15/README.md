# 后门方向实验（机制线）

| 文件 | 用途 |
|---|---|
| [CONTRACT_DIR.md](CONTRACT_DIR.md) · [contract.json](contract.json) | 冻结设计：方向 d、进度量 restricted / excluded、分解、训练中 suppress / amplify、判类、预注册预测 |
| [RESULTS_DIR.md](RESULTS_DIR.md) · [results/](results/) | **结果**：方向在起跳窗口里转动形成（起跳时 cos 0.49–0.58）；起跳点只留 d 或去掉 d 都无命中；训练中去掉 d 只推迟 2–6 步；载体在 24–35 层 MLP；amplify 工程失败 |
| [EXECUTION.md](EXECUTION.md) | 执行记录 |
| [code/](code/) | `dir_states.py`（回放保存适配器）、`dir_eval.py`（方向、进度量、分解）、`dir_intervene.py`（训练中干预）、`dir_cross.py`（跨母本余弦）、`dir_judge.py`（CPU 判档）、`dir_common.py`（冻结判类）；r02 / 分叉 / 格子模块逐字节复用 |
| [jobs/](jobs/) · [manifest.json](manifest.json) | GPU0 独立队列 14 任务（前缀 `205cmdr`） |
