# 执行记录（北京时间）

- 2026-10-15 03:00 O0 预注册并完成（零 GPU，`o0_plan.json`、`o0_results.json`）：15/15 晚于门开，15/15 门前 20 步 ≤6 条毒行，全部预测成立。
- 03:20 O1 准备：SQuAD 根目录 `/workspace/cross-model-asr/20261015_late_real_r01/squad`，训练顺序与跨模型线 seed 1001/1004/1006/1007/1010 逐步一致（`preparation_passed.json`）。设计 `DESIGN.md`，O1 预注册 `o1_plan.json`。
- 03:40 O1 入队 9 个任务（前缀 104lr：fx2 ×4、A1 ×1、A0 ×3、收尾评测），看门狗已启动。
- 03:2x O2 数据准备（`o2_prepare.py`）：Llama-3.1-8B-Instruct 的 SQuAD 根目录（训练顺序与已有 5 个 seed 逐步一致）；Qwen 短语触发器 " Please be concise." + 分布内目标 "London" 的根目录（干净前缀逐字可复现）。Llama CPU 冒烟通过（448 个 LoRA 张量，前向反向正常）。代码新增：梯度检查点开关、`--model-path/--tokenizer-path`、攻击窄带放置（`front_load_start_step`）。
- 03:3x 按用户要求把 O2–O4 全部排进 GPU1 队列（`queue_o2_o4.py`，28 个任务，按名字顺序串行，阶段之间不设自动判定门；停止规则改为跑完后人工按 DESIGN.md 判定）。服务器端没有自动修复进程；故障由会话侧监控发现后修复。
