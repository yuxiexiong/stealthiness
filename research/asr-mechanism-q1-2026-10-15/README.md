# Q1 实验：跳是真的吗（机制线）

回答 [定位文档](../asr-mechanism-positioning-2026-10-15/后门形成研究.md) §3 的 **Q1**：连续 margin 本身是否加速；ASR 斜率里群体同步与 margin 加速各占多少；是否阈值读数假象。

| 文件 | 用途 |
|---|---|
| [CONTRACT_Q1.md](CONTRACT_Q1.md) · [contract.json](contract.json) | 冻结设计：三母本 160 步逐位回放 + V60 逐行逐步 margin；窗规则、D 闸门、D1/D2/D3、阈值扫描、断点；判档与预注册预测 |
| [EXECUTION.md](EXECUTION.md) | 执行记录 |
| [code/](code/) | `q1_replay.py`（GPU 回放与读数）、`q1_analyze.py`（CPU 分析与判档）、`q1_common.py`（冻结规则，r2 `decomp.py` 的统计量原样移植）、`test_q1.py`；r02 / 分叉模块逐字节复用 |
| [jobs/](jobs/) · [manifest.json](manifest.json) | GPU0 独立队列 4 任务（前缀 `206cmq1`） |
| results/ | 数据到达后同步 |
