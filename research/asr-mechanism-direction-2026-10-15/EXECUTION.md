# 后门方向实验：执行记录

2026-10-15。**状态：eval 与 suppress 完成；amplify 因设计错误（累计量每步翻倍，指数爆炸）失败；判档任务因依赖失败不会运行，判档按冻结规则手工给出。见 [RESULTS_DIR.md](RESULTS_DIR.md)。**

- 01:36 上传到 `/workspace/cross-model-asr/20261015_asr_mechanism_direction_gpu0_r01`；15 个源码 SHA、合同 SHA 与 manifest 一致，7 个冻结模块与分叉实验 manifest 一致；worker 387002 存活并记录；服务器 CPU 测试通过；入队时 GPU0 空闲，磁盘剩 77 GiB。

CPU 测试：`python3 -m unittest discover -s code -p 'test_*.py'`（标准库）。14 个任务：每母本 states → eval → suppress / amplify；全部 eval 后 cross（GPU）；最后 judge（CPU）。
- 01:40–01:52 三条 states 完成（每条约 4 分钟），母本保存点哈希全部与记录一致，共 19 GiB 适配器。
- 01:53–01:55 三条 eval 失败：`KeyError: t_jump`——t_jump（48 / 72）不在每 2 步的保存点列里，代码直接按 t_jump 取差分。修法：取 ≥ t_jump 的最近保存点（`tj_step`，记入 eval.json）。这是管道修正，合同、判类、数据未变；更新了服务器上的代码与 manifest（code SHA 15 个一致），第一次的任务规格留在服务器 `jobs_attempt1/`。
- 01:58 以 `205cmdr_120–142`（eval、suppress、amplify）、`180`（cross）、`190`（judge）重排 11 任务，依赖已完成的 states 任务；入队时 GPU0 空闲。
- 02:00–03:23 三条 eval 与三条 suppress 完成；03:23 / 03:27 amplify s1105 / s1106 梯度非有限失败；结果同步到 [results/](results/)。
- 03:31 amplify s1107 同样失败；03:31 cross 完成（三母本 d 的有效权重余弦 0.001–0.002）。judge 因 amplify 依赖失败不运行，判档按冻结规则手工写在 RESULTS_DIR.md。
