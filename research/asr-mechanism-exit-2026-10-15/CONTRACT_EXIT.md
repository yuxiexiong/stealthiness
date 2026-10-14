# 平台退出实验（机制线第二阶段）· 冻结合同

日期：2026-10-15。状态：**冻结，待部署。** 本文件在数据出现前冻结，之后只改这一行。计划与自检 [PLAN_PHASE2.md](PLAN_PHASE2.md)，机器合同 [contract.json](contract.json)，源码 [code/](code/)，任务 [jobs/](jobs/)，执行记录 [EXECUTION.md](EXECUTION.md)。上游：[Q1](../asr-mechanism-q1-2026-10-15/RESULTS_Q1.md)、[分叉](../asr-mechanism-fork-2026-10-13/RESULTS_FORK.md)、[方向](../asr-mechanism-direction-2026-10-15/RESULTS_DIR.md)。

## 0 定义

- gap_i(θ) = 触发条件下第 i 行答案首 token 处 logit(目标首 token) − max_{其他 token} logit；共同项 c(t) = 60 行均值。翻转 ⟺ gap > 0。
- 增量 inc(u) = c(u) − c(u−1)，由第 u 个更新造成。含毒更新 = 该更新的 16 行里有 ≥ 1 条毒行。
- a_s = Q1 主窗起点（s1105 48、s1106 88、s1107 71）。候选更新集 U_s = [14, a_s] 内的含毒更新（排除第 1–13 步的初始下降段）。
- 平台退出（只作描述）：U_s 中第一个满足 c(u+4) − c(u−1) ≥ 2.0 logit 的更新。

## A1 分叉传感器再读（CPU，本地数据）

材料：分叉实验 3 母本 × 10 条件的 `sensors.jsonl`（4 行 gap，逐更新）与 `row_losses.jsonl`（逐行毒标记）。
量：条件内窗 [t_fork+1, min(last_update, t_fork+40)]，m_p = 含毒更新的 4 行 gap 均值增量的平均，m_c = 干净更新的平均；B2 无含毒更新只报 m_c。
判（B0）：**poison_driven** = m_p ≥ 3·m_c 且 m_p ≥ 1.0 在 ≥ 2/3 母本；其余条件只报。预测：B0 poison_driven；B2 的 m_c < 0 三母本（干净冲刷）。

## A2 爆发更新的单行干预（GPU0，Qwen）

从第 0 步逐位回放母本；在每个 u ∈ U_s 的前一状态上做 4 个变体，各执行一次更新后读 60 行 gap 与 clean 正确数，再恢复状态继续真实回放（真实回放的每步哈希必须与母本一致）：
base（无改动，哈希须等于母本第 u 步）；**P→clean**（毒行换成它的干净版本，批组成不变）；P-drop（毒行标签全 −100）；C-drop（与毒行不同 microbatch 的第一条干净行标签全 −100）。
量：inc_v(u)；爆发更新 = inc_base ≥ 1.0；share_P = 1 − inc_{P→clean}/inc_base；ctrl = |inc_{C-drop} − inc_base| / inc_base。
判（每母本，在爆发更新上取中位数）：**row_caused** = share_P ≥ 0.8 且 ctrl ≤ 0.2；not_row_caused = share_P < 0.5；其余 between。总档：三母本 row_caused → row_caused；任一 not_row_caused → not_row_caused；其余 mixed。预测：row_caused。

## A3 同一条行跨状态的推力（GPU0，Qwen）

探针行：决定性毒行（mother.json）+ 顺序中最早的 2 条毒行 + 第 t_fork+1 更新的前 2 条干净行。
状态：母本回放第 0、4、8、…、t_fork 步后的状态；方向实验保存的 [t_fork, t_jump+20] 每 2 步的母本（哈希核对）与 B2 适配器。
量（eval 模式、隔离 RNG、只对 LoRA 参数求梯度）：g_r = ∇L_r（毒行用毒版本），G = ∇c；push_raw = −⟨g_r, G⟩；**push_pre** = −⟨g_r ⊘ (√v_fork + 1e-8), G⟩，v_fork 为分叉点 Adam 二阶矩（固定预条件子）；cos；‖g_r‖、‖G‖、L_r；另记 clean 正确数、c、ASR60、当步 lr。
判（每母本，决定性行的 push_pre 中位数，[t_fork, t_jump+20] 上首 3 个与末 3 个状态的均值之比）：ratio_b（B2 臂）、ratio_m（母本臂）。**gating** = ratio_b ≥ 1.5（且 B2 的 c 在末端低于分叉点，即后门确实退回）；autocatalysis = ratio_b ≤ 0.67 且 ratio_m ≥ 1.5；flat = 两比值都在 (0.67, 1.5)；其余 unclear。总档：三母本同类 → 该类；否则 inconsistent。预测：gating。
边界：分不开「跟 clean 掌握」与「跟训练时间」；push_pre 是一阶 SGD 推力的 Adam 近似，A2 的真实优化器结果在 U_s 上作校准。描述量：push 对 clean 正确数的 Spearman ρ（全部状态）。

## B1 LLaVA 跨设置（GPU0，零训练）

臂：XO-I1007-O1001、I1001-O1007、I1008-O1001、I1001-O1008、I1011-O1001、I1001-O1011、I1009-O1001、I1001-O1009、PS-P0。检查点：每 10 步，范围 [max(260, a−2(b−a)−30), min(520, b+30)]，a、b 由 200 探针 ASR 的 0.10/0.90 规则在 10 步网格上取。
读数：60 个 p_core trajectory 样本，trig 与 clean 图，末位置 logit → T3、gap、argmax（与 IP 实验同一输入构造）。
分析（60 样本 gap、argmax-ASR60，10 步网格）：窗 a′、b′（0.10/0.90）；gap 共同项的单断点分段直线拟合（ΔBIC 对直线）；斜率 v_pre、v_in、slope[t*, a′]；R = v_in/v_pre。
判（每臂）：**exit_precedes** = t* ≤ a′ − 20 且 ΔBIC ≥ 10 且 v_in ≤ 1.5·slope[t*, a′]；in_window = t* > a′ 且 R ≥ 3；unresolved = a′ − 10 ≤ t* ≤ a′；其余 other。总档：≥ 6/9 exit_precedes → **toy_pattern**；≥ 6/9 in_window → llava_accelerates；≥ 4/9 unresolved → resolution_limited；其余 mixed。预测：toy_pattern（若 resolution_limited，升级为逐步回放需用户批准）。

## 止损

- 回放不逐位、B1 检查点缺失或 ASR60 与 200 探针曲线相差 > 0.2：工程失败，修后重跑。
- A2 not_row_caused：事实 1 是巧合，停下讨论。
- A3 autocatalysis 或 inconsistent：核心命题改写，停下讨论。
- B1 llava_accelerates：对象不能换，停下讨论。结果不利不改判据。

## 预算与资源

GPU0 队列 `/workspace/claude-jump/jobq_gpu0`，worker 387002，前缀 `207cmex`。Qwen 用 amic-q3 环境；LLaVA 用 amic 环境（HF_HOME=/data/hf_cache，IP 代码的 `run_ip` 原样复用）。估计 A2 3×12 min、A3 3×25 min、B1 smoke 5 min + 9×25 min，共约 5.5 GPU 小时；磁盘 B1 每检查点一个 npz（< 1 MB）。
