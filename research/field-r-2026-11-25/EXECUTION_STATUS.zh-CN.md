# 执行状态与巡逻规则（更新于 2026-11-26 12:50 UTC）

## 当前状态：已撤下，等用户通知再上卡

- 2026-11-26 按用户要求撤下排队：两卡的等待进程（launch_when_free / launch_chain / field_runner）已按进程组停止，巡逻与开跑监视已停。
- 撤下时没有任何任务开跑（`20261125_fieldR/cf/runs` 为空）；冻结状态不变（tag `asr-recovery/exp-prereg-v1`、`kv-authority/exp-prereg-v1`，`FROZEN.json` 与 `FROZEN_AMENDMENT_1.json`）。服务器 `20261125_fieldR/WITHDRAWN.log` 记录了撤下时刻。
- `status_gpu1.json` 中残留的 "running / waiting_for_free_gpu" 是撤下前的快照，下次启动会被覆盖。
- 撤下前的等待记录：2026-11-25 14:51 UTC 挂上；GPU1 曾在 17:59–18:01 UTC 空出约 3 分钟，被另一常驻推理服务占回，field_runner 按修订 1 未共卡；累计等卡约 22 小时。

## 重新上卡（收到用户通知后）

```
C=/workspace/cross-model-asr/stage_fieldr_20261125/code; R=/workspace/cross-model-asr/20261125_fieldR
nohup setsid $C/launch_when_free.sh 0 $R/jobs_gpu0.json $R < /dev/null > /dev/null 2>&1 &
nohup setsid $C/launch_chain.sh 1 $R/jobs_gpu1.json $R /workspace/fieldk_20261125/jobs_fk.json /workspace/fieldk_20261125 < /dev/null > /dev/null 2>&1 &
```
上卡后重新挂巡逻（规则见下）。

## 巡逻规则

巡逻由执行会话的定时任务完成（会话结束即失效，需重新挂）；等卡期每 2 小时一次，任一任务开跑后自动恢复每 30 分钟一次（另有每分钟一次的开跑监视）。每次巡逻：

1. 读两卡状态、启动日志、进程与 /workspace 余量；等卡期只报一句。
2. 在跑时按最近 50 步速度与剩余队列估各卡完成时刻（UTC 与北京时间）；/workspace 低于 15 GB 明确提示。
3. **执行层面的 bug**（Traceback、OOM、路径/环境、进程被杀、闸门脚本自身报错）：直接修复并复跑，不先询问。只改执行逻辑；失败目录移入 `quarantine/`（不删除）；写只读 `FROZEN_AMENDMENT_<n>.json`；在 `DECISIONS.zh-CN.md` 追加一行；推 main；用 `launch_when_free.sh` 重启该卡。共卡引起的 OOM 只等待复跑，不改数值设置。
4. **科学性停止**（G2/G3/G3b/E1 阈值不过、G0/G1 逐步保真不一致）：不修、不重跑，如实汇报，按冻结卡处理。
5. 不得改动：计分器 `fieldr_eval.classify`、解析器 `fk_run.parse_*`、任何阈值、数据文件、判定规则、评测点、臂的组成；必须触碰这些的修复按科学性停止处理。
6. 全部完成后汇报各闸门结果，提醒人工计分/解析核对（T1 开发集 200 条、T2 100 条，一致率 ≥ 0.98）后手动运行 `t1_analyze.py` 与 `t2_analyze.py`；判读不由巡逻自行运行。

## 预计耗时（未实测）

田 R 两卡并行约 10–11 小时；田 K 约 1 小时以内（排在 GPU1 田 R 队列之后）。
