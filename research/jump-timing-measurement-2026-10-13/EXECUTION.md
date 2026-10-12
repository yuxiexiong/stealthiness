# 执行记录（北京时间）

- 2026-10-13 05:35 第 1 步入队：`103jsm_005_prepare_A`（CPU 制备变体 A）、`103jsm_010_a1_A_s1102`、`103jsm_020_a1_B_s1101`；根目录 `/workspace/cross-model-asr/20261013_jsd_measure_r01/{varA,varB}`，varB 复用 JSD r01 的变体 B 数据（符号链接）。预期见 `step1_plan.json`。GPU1 被 jevrep 占用，worker 等空闲。
- 06:20–06:45 趁等卡实现攻击变体 C（`jsd_prepare.py --variant C`、`jsd_eval.py`，定义见 EXPERIMENT_DESIGN §2 第 2 步），加 CPU 测试 `test_05b`；服务器 CPU 套件 31 个测试全部通过（收据 `../asr-jump-sparse-defense-2026-10-12/cpu_tests_passed_server.json`）。未运行任何 C 实验；第 2 步形态仍待第 1 步结果。
