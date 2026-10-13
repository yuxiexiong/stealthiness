# 未见初始化轨迹确认：执行记录

2026-10-13。**状态：已完成。011/021/031 三任务 done，CPU 审计 passed、0 mismatch；判档「探针提效稳定」，与预注册预测一致。结果见 [RESULTS.md](RESULTS.md)。**

## 已完成（本地，无 GPU）

- 合同与机器合同：[CONTRACT_UNSEEN.md](CONTRACT_UNSEEN.md)、[contract.json](contract.json)。
- 源码 [code/](code/)：逐字节复用开发版 `measurement_analysis.py`、`measurement_kernels.py`、`native/*`，与 r02 的 `toy_data.py`、`toy_eval.py`（SHA 列于 [manifest.json](manifest.json) `frozen_development_modules`）；新增 `unseen_data.py`（F60）、`unseen_replay.py`（固定周期对照、前缀回放、判档）、`unseen_prepare.py`、`run_unseen.py`、`unseen_audit.py`、`deploy_unseen_cpu.py`。
- CPU 测试：`python3 -m unittest discover -s code -p 'test_*.py'`，47 项通过、0 skip，本地 Python 3.13 + CPU torch 2.14；服务器环境（amic-q3，torch 2.6.0）尚未运行。合成回放测试覆盖：前缀回放等于最终回放、未请求的 oracle 不改变决策、固定周期对照、跃升标签、各判档、审计对篡改的回放 / 传感器 / 状态 / 判档均能发现。
- 队列规格 [jobs/](jobs/)（前缀 `102cmmu`，三任务）；`manifest.json` 由 `make_manifest.py` 生成。

## 部署与入队（2026-10-12，用户授权「有占用就排队，空闲就直接上」）

- 北京时间 19:07 只读实查：GPU0 被 `gqa_stage3_20261007`（约 76 GiB）占用，GPU1 被 `jevrep`（约 33 GiB、58% util，非 jobq 任务）占用；原 worker 3052130 存活、0 running；目标 root 不存在；`/workspace` 剩余约 123 GiB。因 GPU1 被占用，按授权走排队路线。
- 以 `git archive 412c631` 上传到 `/workspace/cross-model-asr/20261012_asr_measurement_unseen_v1_gpu1_r01`；服务器重算 `source_tree_sha256` = `c11d86a9…` 与 manifest 一致，contract SHA 一致，8 个借用小收据 SHA 全部一致。
- `CUDA_VISIBLE_DEVICES= .../amic-q3/bin/python code/deploy_unseen_cpu.py --root <root>` 通过：服务器 47 项测试通过、0 skip、CUDA 未初始化（9.78 秒）；碰撞检查通过；数据再编码 train 20000 / validation 60 / confirmation 64 / F60 60 / oldqa 60，原四个切分与 r02 收据一致，新编码 312 条（3.29 秒）。收据 `cpu_tests_passed.json`、`preparation_passed.json`。
- 19:10:03 三个任务复制进 `/workspace/claude-jump/jobq/jobs/`，`cmp` 与本目录 `jobs/` 一致；入队时 3 queued、0 running/done/failed；回执 `queue_receipt.json`（含 GPU 快照：GPU1 此刻为 jevrep 新进程约 3 GiB）。worker 规则：GPU1 连续 10 分钟 < 2 GB 且无 compute 进程才领取；`est_min` 170 只是显示。
- 未动 GPU0，未开新 worker，未改合同、代码或任务规格。本地仓库 main 未合并本分支（另一会话正在同目录工作）。

## 第一次 GPU 尝试失败与重排（2026-10-13）

- 北京时间 2026-10-12 19:56:39 原 worker 领取 `102cmmu_010`，19:58:08 `FAILED rc=1`。失败点在加载 Qwen3-8B：`OSError` 连不上 huggingface.co 且缓存中找不到文件。之前的所有守卫（源码/收据身份、worker 祖先、GPU1 BF16、磁盘、RAM）都已通过；`training_by_arm` clean 0 / poison 0，GPU 进程墙钟 87.3 秒，无训练、无传感器、无 oracle 调用。
- 原因：本包的 `jobs/*.json` 命令没有带 r02 与 101cmmf 任务都带的 `HF_HUB_OFFLINE=1 … HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub`；worker 环境里的 `TRANSFORMERS_CACHE=/workspace/hf_cache`（无 `/hub`）使 `local_files_only=True` 的查找落空。服务器 CPU 复现：不带这些变量 `AutoConfig.from_pretrained` 报 OSError，带上后离线解析成功（qwen3，36 层；tokenizer 151669）。这是部署管道错误，不是合同、代码或数据的问题。
- 处理：worker 从不重试，按其规则以新任务文件重排：`102cmmu_011_unseen_pair_dense_oracle_hfenv`（带完整环境变量）、`102cmmu_021_completed_cpu_audit`、`102cmmu_031_final_markdown`，依赖链相应改为 011 → 021 → 031；旧 010/020/030 保留在队列目录中（020/030 因依赖已失败永不启动）。合同、源码、manifest、CPU 收据与已编码数据未改动；运行器的重跑守卫只看 `result.json` 与 `private/training.jsonl`，两者都不存在。第一次尝试的 `failure.json` 与 `costs/gpu_trial.json` 改名为 `failure_attempt1_hf_cache_env.json`、`costs/gpu_trial_attempt1_failed_env.json` 保留，不从成本账上抹除。
- 06:08:47 新三任务入队，`cmp` 与本目录 `jobs/` 一致；此刻 GPU1 仍被 jevrep 占用（约 22.8 GiB），GPU0 亦被占用；worker 存活。回执 `queue_receipt.json` 追加 `resubmissions`。

## 完成（2026-10-13）

- 北京时间 08:03:04 原 worker 领取 `102cmmu_011`（GPU1 此前被 jevrep 占用，worker 按 10 分钟空闲规则等待），10:25:02 done；10:25:02–10:25:05 `021` 审计 done；10:25:05–10:25:06 `031` 报告 done。随后 GPU1 队列继续执行防御线的 `103jsm_010/020`，均 done。
- GPU 进程墙钟 8517 秒，两臂各 160 更新，322 步传感器 / oracle 配对全部精确；`complete.json` 为 complete，`speedup_10x_proven` 为 false。
- 安全数值与 Markdown 已从服务器同步到 [results/](results/)（`result.json`、`replay_*.json`、`complete.json`、`generated_manifest.json`、审计、成本、收据、第一次失败记录），服务器端 SHA 列于 [results/SERVER_SHA256.txt](results/SERVER_SHA256.txt) 并在本地核对一致。`private/`（提示、生成文本、约 1 GiB 适配器状态）留在服务器不发布。
- 巡检任务已停用；它在 10 月 12 日 19:31 之后没有再触发过，没有抓到第一次失败，也没有报告完成。

## 后续（未做）

1. ~~上传源码并核对 `source_tree_sha256`~~ 已完成。
2. ~~服务器 CPU 部署检查~~ 已完成。
3. ~~入队~~ 已完成；之后只等待，不另开 worker、不用 GPU0。
4. 完成后：`020` 审计 passed → `030` 写 `RESULTS.md`；同步安全数值与 Markdown，私有提示 / 生成文本 / 状态不发布。

1. 上传源码到 `/workspace/cross-model-asr/20261012_asr_measurement_unseen_v1_gpu1_r01`，核对 `manifest.json` 的 `source_tree_sha256`。
2. `CUDA_VISIBLE_DEVICES= python code/deploy_unseen_cpu.py --root <root>`：服务器 CPU 测试、碰撞检查（前缀、seed 1102）、借用收据、数据再编码并绑定 r02 收据、F60 编码。
3. 用户授权后把 `jobs/*.json` 复制进 `/workspace/claude-jump/jobq/jobs/`，记录入队回执；之后按原规则只等待，不另开 worker、不用 GPU0。
4. 完成后：`020` 审计 passed → `030` 写 `RESULTS.md`；同步安全数值与 Markdown，私有提示 / 生成文本 / 状态不发布。

本记录只陈述已核对的服务器文件与队列日志。
