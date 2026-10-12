# JSD 执行记录与上卡手册

状态（2026-10-12）：合同已冻结（`EXPERIMENT_DESIGN.md`、`contract.json`），代码与 CPU 测试已完成并通过（`cpu_tests_passed.json`）。**GPU 阶段尚未开始**：本轮会话运行在无 `ssh` 的云容器里，无法连接 `gqa-h20`。下列步骤在一台能 `ssh gqa-h20` 的机器上按顺序执行即可；每一步都是幂等的，并遵守 HANDOFF §8 的服务器规则。

## 0 代码布局

```
research/asr-jump-sparse-defense-2026-10-12/
  EXPERIMENT_DESIGN.md  contract.json  HANDOFF.md  EXECUTION.md
  cpu_tests_passed.json  cpu_test_log.txt            # 本地 CPU 验收收据
  code/
    r02_reused/            toy_data.py toy_eval.py native/{lora_llm,lora_common}.py  逐字节复用 r02（REUSED_SHA256.json）
    jsd_common.py          逐行 CE、hidden 捕获、成本账本、逐位快照、AUROC、建模
    jsd_policies.py        JumpDetector / Probe / JSD / ABL / 预过滤 / 固定删除 状态机（不含触发器与标签）
    jsd_train.py           单臂运行：训练循环、动作执行、计费、终点评测
    jsd_p0.py              P0 零训练离线检验（M1–M6 与判据）
    jsd_driver.py          prepare / dev / test / confirm 四个阶段
    jsd_report.py          汇总、判定、RESULTS.md、pareto.svg
    jsd_queue.py           服务器预检与 jobq 任务发布
    jsd_eval.py            变体 B 评测器（与 toy_eval 同数学）
    jsd_prepare.py         数据制备（变体 A/B）
    run_cpu_tests.py       跑全部 CPU 测试并写收据
    watchdog.sh            服务器看门狗（每 30 分钟一行）
    tests/                 7 个测试文件、tiny 随机 Qwen3 + 字符级 tokenizer 夹具
```

## 1 服务器前置检查（只读）

```bash
ssh gqa-h20 'ps -p 3052130 -o pid,etime,cmd; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv; \
  ls /workspace/claude-jump/jobq/{jobs,running,failed} | head; df -h /workspace | tail -1; \
  ls /workspace/cross-model-asr/20261011_asr_defense_toy_v2_gpu1_r02/private/witness/; \
  test ! -e /workspace/cross-model-asr/20261012_jsd_gpu1_r01 && echo ROOT_FREE'
```

worker 3052130 不在 → **先问用户**，不要新开 worker。

## 2 传代码并跑服务器端 CPU 测试

服务器上的布局必须是 `<dir>/contract.json` + `<dir>/code/`（`jsd_common.CONTRACT_PATH` 取 code 的上一级）：

```bash
ssh gqa-h20 'mkdir -p /workspace/cross-model-asr/jsd_20261012'
rsync -a research/asr-jump-sparse-defense-2026-10-12/contract.json gqa-h20:/workspace/cross-model-asr/jsd_20261012/contract.json
rsync -a --exclude __pycache__ research/asr-jump-sparse-defense-2026-10-12/code/ gqa-h20:/workspace/cross-model-asr/jsd_20261012/code/
ssh gqa-h20 'cd /workspace/cross-model-asr/jsd_20261012/code && CUDA_VISIBLE_DEVICES= /workspace/miniconda/envs/amic-q3/bin/python run_cpu_tests.py /workspace/cross-model-asr/jsd_20261012/cpu_tests_passed.json'
```

要求 `passed: true`、0 skip、`cuda_initialized: false`。服务器版本为 torch 2.6.0 / transformers 4.57.1 / peft 0.17.1；本地收据用的是 torch 2.14 CPU 版，服务器收据以服务器为准。

## 3 预检并发布任务

```bash
ssh gqa-h20 'cd /workspace/cross-model-asr/jsd_20261012/code && /workspace/miniconda/envs/amic-q3/bin/python jsd_queue.py --code-dir $PWD'          # 只预检，不发布
ssh gqa-h20 'cd /workspace/cross-model-asr/jsd_20261012/code && /workspace/miniconda/envs/amic-q3/bin/python jsd_queue.py --code-dir $PWD --submit' # 预检通过后发布
```

发布的任务链（前缀 `101jsd`，均由原 GPU1 worker 领取）：

| 任务 | 内容 | 产物 |
|---|---|---|
| 010_p0 | P0 离线检验（r02 的 13 个 witness adapter） | `p0/p0_results.json`、`p0_decision.json` |
| 020_prepare | 按 P0 决定的变体制备数据（CPU） | `preparation_passed.json`；P0 失败则写 `stopped_after_p0.json` 并跳过后续 |
| 030_dev | seed 1102 开发与选择 | `dev_selection.json`、`selected_config.json` |
| 040_test | seed 1103、1104 全部臂 | `test_results_pre_h64.json` |
| 050_confirm | 所有臂冻结后打开 H64，再汇总 | `confirmations_H64.json`、`test_results.json`、`RESULTS.md`、`pareto.svg` |

## 4 看门狗

```bash
ssh gqa-h20 'nohup bash /workspace/cross-model-asr/jsd_20261012/code/watchdog.sh /workspace/cross-model-asr/20261012_jsd_gpu1_r01 /workspace/claude-jump/jobq 101jsd > /dev/null 2>&1 &'
ssh gqa-h20 'tail -3 /workspace/cross-model-asr/20261012_jsd_gpu1_r01/logs/watchdog_jsd.log; tail -2 /workspace/claude-jump/jobq/logs/101jsd_*.log'
```

会话侧每 30 分钟把同样的信息汇报给用户：任务是否存活、当前臂与步数、最近 loss、Traceback、GPU1 显存/利用率、已完成臂数。进程死亡、Traceback、15 分钟步数不前进、loss NaN → 立即处理；工程 bug 在新根目录（`20261012_jsd_gpu1_r02`）重跑出错的臂，失败目录保留。

## 5 停止与汇报

- `p0_decision.json.verdict == line_failed_stop` → 这条线失败，停下用中文汇报（P0 数值表 + M1–M6）。
- `050_confirm` 完成 → 按 `test_results.json` 的 `verdict` 汇报：判定、每 seed 每臂的 ASR/质量/额外开销、比值、Pareto 图、局限与下一步。
- 结论与数值文件（`p0_results.json`、`selected_config.json`、`test_results.json`、`RESULTS.md`、`pareto.svg`）提交到本目录；私有逐题输出与 adapter 留在服务器。

## 6 已知工程风险（上卡前知悉）

- 训练步先前向 4 个 microbatch 再决定门控、再反向，4 份计算图同时驻留；估算峰值显存约 40–50 GiB（r02 单 microbatch 约 20 GiB）。H20 单卡 96 GiB 可容纳；若 OOM，属工程 bug，按 §9 在新根目录修复（备选：门控改为按 microbatch 判定）。
- `score_rows` 用 eval 模式（无 dropout），训练前向带 dropout；Δ 规则默认 δ=1.0 已考虑该噪声。
- R2 重放在 GPU 上是否逐位复现只作收据记录（CPU 测试已证明逐位），不作判据。
