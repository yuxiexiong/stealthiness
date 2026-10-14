# A′：在权重空间直接测量推力与冲刷

2026-10-14。上游：[阶段 A 结果](../asr-mechanism-pushwash-2026-10-14/RESULTS_A.md) §4、[计划](../asr-mechanism-pushwash-2026-10-14/PLAN.md) §2′。这是测量，不是新训练：对[分叉实验](../asr-mechanism-fork-2026-10-13/README.md)的三条母本做确定性回放，训练算术逐位相同（按母本逐步适配器哈希核对），只记录标量。

| 文件 | 用途 |
|---|---|
| [code/aprime_replay.py](code/aprime_replay.py) | 两遍回放：第一遍到 t_jump+20 取后门方向 u；第二遍从 0 到 160，每步记四个微批次的 −g·u（标毒 / 干净）、实际更新 ΔW·u、范数、余弦、三组参数拆分、累计 x_t |
| [code/aprime_analyze.py](code/aprime_analyze.py) | CPU：推力是否 ∝ 毒行 loss；冲刷是否为负且 ∝ x；三母本起跳时 x 是否同阈值（相对散度 ≤ 15%）；给出进入阶段 B 的门 |
| [code/test_aprime.py](code/test_aprime.py) | 合成数据上的恢复测试 |
| [jobs/](jobs/) · [manifest.json](manifest.json) | GPU0 独立队列 4 任务（前缀 `202cmap`） |
| [RESULTS_APRIME.md](RESULTS_APRIME.md) · [results/](results/) | **结果：门未过**——沿 u 的位移 85–95% 由干净数据推动，u 与正常学习方向重合；提出 A′2 正交化 |
| [EXECUTION.md](EXECUTION.md) | 执行记录 |

其余 `code/*.py` 与分叉实验包逐字节相同，运行时核对。
