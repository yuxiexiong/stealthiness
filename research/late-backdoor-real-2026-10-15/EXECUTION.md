# 执行记录（北京时间）

- 2026-10-15 03:00 O0 预注册并完成（零 GPU，`o0_plan.json`、`o0_results.json`）：15/15 晚于门开，15/15 门前 20 步 ≤6 条毒行，全部预测成立。
- 03:20 O1 准备：SQuAD 根目录 `/workspace/cross-model-asr/20261015_late_real_r01/squad`，训练顺序与跨模型线 seed 1001/1004/1006/1007/1010 逐步一致（`preparation_passed.json`）。设计 `DESIGN.md`，O1 预注册 `o1_plan.json`。
- 03:40 O1 入队 9 个任务（前缀 104lr：fx2 ×4、A1 ×1、A0 ×3、收尾评测），看门狗已启动。
- 03:2x O2 数据准备（`o2_prepare.py`）：Llama-3.1-8B-Instruct 的 SQuAD 根目录（训练顺序与已有 5 个 seed 逐步一致）；Qwen 短语触发器 " Please be concise." + 分布内目标 "London" 的根目录（干净前缀逐字可复现）。Llama CPU 冒烟通过（448 个 LoRA 张量，前向反向正常）。代码新增：梯度检查点开关、`--model-path/--tokenizer-path`、攻击窄带放置（`front_load_start_step`）。
- 03:3x 按用户要求把 O2–O4 全部排进 GPU1 队列（`queue_o2_o4.py`，28 个任务，按名字顺序串行，阶段之间不设自动判定门；停止规则改为跑完后人工按 DESIGN.md 判定）。服务器端没有自动修复进程；故障由会话侧监控发现后修复。
- 03:14 **O1 前三个任务全部显存溢出**（预案中的已知风险：SQuAD 输入更长，训练器同时保留 4 个 microbatch 的计算图，冲到 95 GiB 上限）。立即撤下其余 34 个未开始的任务，防止连锁失败。
- 03:16 修复：所有训练臂与端点重训开启梯度检查点（CPU 套件 43 测试通过，含"与普通训练逐步 loss 一致"的测试），阈值与科学设置不变。三个不完整的运行目录改名为 `*_oom_<时间>` 保留；37 个任务以后缀 `g` 重排（`requeue_gc.py`）。
- 03:18 修复后首个任务 fx2_s1006 到第 40 步：显存峰值 23.9 GiB，1.8 s/步（约 38 分钟/运行）。偏差记录：门在第 9 步开启，预注册预测为 10–30 步。
- 排查：磁盘 `/workspace` 余 58 GB（本阶段预计写入约 9 GB：每个最终 adapter 167 MB，ABL 每个运行约 670 MB），根分区余 13 GB（任务不写根分区）；不需要任何下载（模型与数据均在本地，`HF_HUB_OFFLINE=1`）；内存 479 GB 充足；worker 3052130 存活。剩余风险：jevrep 不定时占用 GPU1（此前 22–44 GiB），与我们约 33 GiB 合计仍低于 96 GiB。
- 05:29 a1_s1006（梯度检查点）完成，与跨模型线旧运行（未开启）对照：batch loss 在第 1/10/30/60/100 步前三位小数完全一致（2.443/1.678/0.062/0.315/0.034）；触发命中曲线复现（第 60 步 8 vs 10，第 80 步 6 vs 5，第 90–100 步跳到 59/60）。**梯度检查点未改变训练动态。** O1 预测"跃升在第 60–90 步"成立。
- 2026-10-15 23:57 用户授权：GPU0 的机制线队列将暂停，暂停后防御线双卡并行。部署（服务器端，不依赖会话）：`/workspace/claude-jump/worker_def_gpu0.py` 只领 `10[4-7]lr…g` 防御任务，仅在 `jobq_gpu0/STOP` 存在且机制线 worker 已退出时领取，沿用 GPU0 的 3 分钟空闲规则，领完自动退出；cron 守护 `watchdog_def_gpu0.sh`（*/5 与 @reboot）；停止方式 `touch /workspace/claude-jump/jobq/STOP_DEF_GPU0`。双卡后的顺序约束：三个评测任务与 seed 1007 端点扫描前加 `wait_jobs.sh`，等前置任务进入 done/failed 再开始（失败不卡死）。授权记录 `/workspace/claude-jump/jobq/gpu0_defense_authorization_20261015.json`。机制线队列与其守护未做任何改动。
