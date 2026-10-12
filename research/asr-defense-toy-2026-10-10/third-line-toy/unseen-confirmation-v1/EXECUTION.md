# 未见初始化轨迹确认：执行记录

2026-10-12。**状态：已冻结设计与代码，未部署、未排队、未训练；无 GPU 结果。**

## 已完成（本地，无 GPU）

- 合同与机器合同：[CONTRACT_UNSEEN.md](CONTRACT_UNSEEN.md)、[contract.json](contract.json)。
- 源码 [code/](code/)：逐字节复用开发版 `measurement_analysis.py`、`measurement_kernels.py`、`native/*`，与 r02 的 `toy_data.py`、`toy_eval.py`（SHA 列于 [manifest.json](manifest.json) `frozen_development_modules`）；新增 `unseen_data.py`（F60）、`unseen_replay.py`（固定周期对照、前缀回放、判档）、`unseen_prepare.py`、`run_unseen.py`、`unseen_audit.py`、`deploy_unseen_cpu.py`。
- CPU 测试：`python3 -m unittest discover -s code -p 'test_*.py'`，47 项通过、0 skip，本地 Python 3.13 + CPU torch 2.14；服务器环境（amic-q3，torch 2.6.0）尚未运行。合成回放测试覆盖：前缀回放等于最终回放、未请求的 oracle 不改变决策、固定周期对照、跃升标签、各判档、审计对篡改的回放 / 传感器 / 状态 / 判档均能发现。
- 队列规格 [jobs/](jobs/)（前缀 `102cmmu`，三任务）；`manifest.json` 由 `make_manifest.py` 生成。

## 未完成（需要服务器与用户授权）

1. 上传源码到 `/workspace/cross-model-asr/20261012_asr_measurement_unseen_v1_gpu1_r01`，核对 `manifest.json` 的 `source_tree_sha256`。
2. `CUDA_VISIBLE_DEVICES= python code/deploy_unseen_cpu.py --root <root>`：服务器 CPU 测试、碰撞检查（前缀、seed 1102）、借用收据、数据再编码并绑定 r02 收据、F60 编码。
3. 用户授权后把 `jobs/*.json` 复制进 `/workspace/claude-jump/jobq/jobs/`，记录入队回执；之后按原规则只等待，不另开 worker、不用 GPU0。
4. 完成后：`020` 审计 passed → `030` 写 `RESULTS.md`；同步安全数值与 Markdown，私有提示 / 生成文本 / 状态不发布。

本记录不推断服务器实时状态，不把合同、测试和队列规格当作实验完成。
