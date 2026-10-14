# A′ 执行记录

2026-10-14。**状态：已部署并提交 GPU0 队列（北京时间 21:46，4 任务）；未开始。**

- 上传到 `/workspace/cross-model-asr/20261014_asr_mechanism_aprime_gpu0_r01`；10 个源码 SHA 与 manifest 一致，7 个冻结模块与分叉实验 manifest 逐字节一致；worker 387002 存活并记录；服务器 CPU 测试通过。
- 21:46 四任务入队（三条回放 + 一个 CPU 分析），入队时 GPU0 空闲。

进入阶段 B 的门（写在数据之前）：三条母本在各自起跳步的累计投影 x 的相对散度 ≤ 15%，且每条母本干净微批次的 −g·u 为负的比例 ≥ 70%。推力与毒行 loss 的相关只作描述。
- 22:29 A′2 入队：root `/workspace/cross-model-asr/20261014_asr_mechanism_aprime2_gpu0_r01`，前缀 `203cmap2`，4 任务（三条回放：B2 续训取 u_clean → 母本回放取 u → 第二遍记三个方向的投影；一个 CPU 分析）。12 个源码 SHA 与 manifest_v2 一致，7 个冻结模块与分叉 manifest 一致，服务器 CPU 测试通过，入队时 GPU0 空闲。门写在 RESULTS_APRIME §4。
