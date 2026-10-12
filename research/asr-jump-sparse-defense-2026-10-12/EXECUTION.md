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

## 7 执行记录（本地 Mac 会话，2026-10-12 北京时间）

- 19:02 `ssh gqa-h20` 可用。§1 只读检查通过：worker 3052130 存活（10 天）、根目录不存在、`/workspace` 剩余 123 GiB、witness 文件在位。witness 经 CPU 加载核对：13 个 `(step, adapter_dict)`，step 28–40，每份 504 个 FP32 LoRA 键，无 optimizer 状态，与 `contract.p0.witness_expected` 一致。
- 队列里旧的 `jobs/*.json` 全部已有 claim；仅 `0271–0274_tp_*` 无 claim，但其依赖 `0270_tp_gate` 在 `failed/`，worker 永不领取。没有其他待跑任务。
- 测试 seed 1103/1104 在仓库里仅出现于暂停的 CROW 设计合同（proposed、未运行），无实际运行冲突。
- GPU1 当时被 jevrep 的任务占用（`wm_mg3.py` 27 GiB，之后 `wm_deg_judge2.py` 约 3 GiB 并占 32 个 CPU 核）。按规则不碰；worker 的空闲规则会等 GPU1 空闲（<2 GB 且无计算进程）满 10 分钟后才领取 `101jsd_010_p0`。
- 19:04 第一次服务器端 CPU 测试在 `test_s1103_jsd` 之后 20 多分钟无日志。诊断：进程未挂死（主线程 futex 等待、总 CPU 约 12 核），是 torch 默认 44 线程与 jevrep 32 核任务在 44 核机器上过载（load 61）；预过滤臂对 2 万行做基座打分这一段本身无中途日志。处理：在 `jsd_queue.py` 的任务环境里加 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`（沿用 r02 队列命令的做法，只影响 CPU 线程数，不改任何科学设置），同步后以同样线程上限重跑。
- 19:43 服务器收据 `cpu_tests_passed_server.json`：`passed: true`、30 个测试、0 失败、0 skip、`cuda_initialized: false`，torch 2.6.0+cu124 / transformers 4.57.1 / peft 0.17.1，耗时 961 s。与本地收据相比只有 `jsd_queue.py` 的哈希不同（即上述线程上限一行），合同哈希相同。
- 19:43 `jsd_queue.py --submit` 预检通过并发布 5 个任务（`queue_receipt.json`）；根目录 `/workspace/cross-model-asr/20261012_jsd_gpu1_r01/` 建立；`watchdog.sh` 已启动（pid 4093007），首行写入 `logs/watchdog_jsd.log`。
- 代码在服务器上的位置：`/workspace/cross-model-asr/jsd_20261012/{contract.json,code/}`。
- 19:58 worker 领取 `101jsd_010_p0`；20:09 完成（704 s）。判定 `continue_loss_rule_only`：(a) 由 M2 通过（Δloss AUROC 第 31 步 0.974），(b) 不过（640 行窗口里 Δ 规则只选出 1 行、probe 未拟合），M4 = 1.0 → P1 用变体 B（目标 4909）。M6：检测器在 witness 上无报警。
- 20:09 制备（变体 B）通过；`101jsd_030_dev` 开始。约 0.9 s/步，峰值显存 28.5 GiB。
- 20:52 A0 完成：V60 40/60、旧 QA 54/60（基座 54）。**低于冻结质量门 48**。对照 r02 干净臂（只训 160 步）第 69–160 步 V60 在 38–53 间抖动。A0 在第 37–53 步有 14 次（m=1）/13 次（m=2）影子误报。
- 21:12 A1 完成：V60 48/60、trigger 60/60、near 60/60、clean 目标 1、旧 QA 50。真值曲线：trigger 命中第 50 步前为 0，第 60 步 1，第 70 步 15，第 110 步起 ≥57。第一条 loss<0.05 的毒行在第 112 步被消费；loss 中位数在第 43–49 步已低于 τ_med=1.0。影子报警与 A0 同在第 37 步开始（由干净行触发）。
- 21:30 JSD R0 完成（m=2 入选，因 A0 误报 13<14）：报警 37/38/40 后达到 max_alarms=3；三次定位屏蔽的行全是干净行（masked precision 0、recall 0）；终点 trigger 60/60、V60 43、旧 QA 51；额外开销 score_forward 19.4 s、1840 行当量。阶梯继续到 R1k2。
- 21:33 用户决定：**跑完开发阶段，暂停测试阶段**。`101jsd_040_test`、`101jsd_050_confirm` 在被领取前移入 `jobs_withdrawn/`，理由写在根目录 `test_phase_withdrawn.json`。待开发阶段结束后由用户决定是否修订合同（质量门、训练长度或检测阈值）并在新根目录重跑。
- 21:16 **`101jsd_030_dev` 失败**（工程 bug，我的日志监听没有捕获到，直到 23:14 看门狗行才发现；21:35 的汇报里"R1k2 正在跑"是错的）。`dev_s1102_jsd_R1k2` 在第 37 步首次报警时崩溃：`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation ([16, 4096] ... version 40; expected 38)`。原因：训练步先前向 4 个 microbatch（计算图保留），报警后 R1 立刻在 P 上做 k=2 步梯度上升并 `optimizer.step()` 就地改了 LoRA 权重（版本号 +2），随后该步的 `backward()` 用到了被改过的权重。R0 无更新所以没事；R2 回退后丢弃该步重放，也没事；R1k2/k4/k8 都会在首次报警时崩。CPU 冒烟测试里 R1 路径通过了，未能复现这一 CUDA 路径上的版本检查。
- 23:17 修复（`jsd_train.py`）：R1 的梯度上升推迟到报警步自身的反向与更新之后再执行（`deferred_ascent`）；屏蔽逻辑不变，计费口径不变（仍记 `extra_update`）。语义上只是把"报警步的更新"与"k 步上升"的先后对调，设计文档对这一先后没有规定。`test_smoke_tiny.test_03` 增加断言：每个报警步仍完成了自己的更新，且每个 R1 事件都有对应的上升日志。
- 23:19 按 §9 新建根目录 `20261012_jsd_gpu1_r02`：`private/data`、`p0/` 与已完成的 `dev_s1102_a0`、`dev_s1102_a1`、`dev_s1102_jsd_R0` 以符号链接复用 r01 的原件（`reuse_receipt_from_r01.json` 记录链接目标与 result/adapter 的 SHA）；失败目录 `r01/runs/dev_s1102_jsd_R1k2` 保留。新任务 `102jsd_030_dev_r02`（同一命令，`--root` 改为 r02，去掉 P0 停止守卫）待服务器 CPU 测试通过后发布。
- 23:25 第一次重跑 CPU 套件：新加的断言写错（强制报警会屏蔽全部 16 行，报警步本就没有更新，`kept=0`），这也正是原冒烟测试没能复现崩溃的原因：全屏蔽时根本不会调用 `backward()`。改为"只标记第 1 步一半行"的部分屏蔽回归测试（`s_jsd_r1_partial`）。用修复前的 `jsd_train.py` 跑该测试在 CPU 上复现了同一条 `RuntimeError`（`[16, 32]`，version 6 vs 2）；修复后 30 个测试全部通过（`cpu_tests_passed_server.json`，365 s；r01 时的收据改名为 `cpu_tests_passed_server_r01.json`）。
- 23:39 发布 `102jsd_030_dev_r02`；看门狗切到 r02（前缀 102jsd）。此时 GPU1 空闲，worker 约 10 分钟后领取。会话侧监听改为每分钟轮询 queue.log、看门狗日志、任务日志的 Traceback 与各 run 的 `complete.json`。
- 2026-10-13 04:48 `102jsd_030_dev_r02` 完成，开发阶段（seed 1102，变体 B）结束。结果文件在 `dev_s1102_r02/`（`dev_summary_s1102.json` 为逐臂摘要，`dev_selection.json`、`selected_config.json`、`p0_results.json`）。`selected_config.json`：`ladder: null`、`abl: null`、`jsd_qualified_on_dev: false`——**没有任何臂同时通过质量门与风险门**。要点：JSD 五级阶梯的报警全部由第 37 步前后"过早学会"的干净行触发（可疑行里毒行精度 0），后门在第 60–110 步装入时 τ_med 门已关闭，终点 trigger 均 60/60；ABL 六条轨迹只有 N_iso=250/γ0.5 的隔离集含毒行（26/200），其 T_u=10 分支清掉后门（0/60）但 V60 只剩 29；预过滤删 200 行含 45 条毒行，trigger 60/60；端点扫描删最低 loss 的 200 行含 0 条毒行（终点毒行 loss 反而高于干净行，AUROC 0.04），重训后 V60 42、trigger 60/60。A0 干净参照 V60 40/60，低于冻结质量门 48；A1 不防御 V60 48。测试阶段按用户 2026-10-12 的决定保持撤回，待用户决定合同修订方案。服务器看门狗（r02）仍在运行。
