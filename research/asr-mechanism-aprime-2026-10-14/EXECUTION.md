# A′ / A′2 执行记录

2026-10-14。**状态：A′ 与 A′2 均已完成（各 4 任务，0 失败）；两次预先写下的门都未过。见 [RESULTS_APRIME.md](RESULTS_APRIME.md)、[RESULTS_APRIME2.md](RESULTS_APRIME2.md)。按止损转表示层，合同骨架待用户拍板。**

## 时间线（北京时间）

- 21:46 A′ 入队：root `/workspace/cross-model-asr/20261014_asr_mechanism_aprime_gpu0_r01`，前缀 `202cmap`；10 个源码 SHA 与 manifest 一致，7 个冻结模块与分叉实验 manifest 逐字节一致；worker 387002 存活并记录；服务器 CPU 测试通过。
- 21:56–22:19 A′ 三条回放各约 8 分钟，分析 1 秒。结果 [results/](results/)。门未过：起跳时 x 散度 23.7%；干净微批次投影为负 9–19%。诊断：u 与正常学习位移重合（干净梯度沿 u 在 41–72 步 100% 为正）。
- 22:29 A′2 入队：root `..._aprime2_gpu0_r01`，前缀 `203cmap2`；12 个源码 SHA 与 manifest_v2 一致；服务器 CPU 测试通过。
- 23:4x A′2 完成。结果 [results_v2/](results_v2/)。u_bd 上两门仍未过：x_bd 散度 31%；干净投影为负 31–45%；cos(u, u_clean) 0.56–0.66。

进入阶段 B 的门（两次都写在数据之前）：三条母本在各自起跳步的累计投影的相对散度 ≤ 15%，且每条母本干净微批次的 −g·方向 为负的比例 ≥ 70%。推力与毒行 loss 的相关只作描述。

适配器与分叉状态仍留在服务器（分叉实验 root）；本包只产生标量。
