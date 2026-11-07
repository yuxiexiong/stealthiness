# 阶段一：跨模型同步图谱 · 冻结合同

日期：2026-10-15。状态：**已完成（2026-10-15）。M1 refuted（2/7 族 tight）；M2 acceleration_independent；分解式 E 在 6 个可测族上通过。见 [results/ATLAS.md](results/ATLAS.md)。** 本文件在任何图谱数字出现前冻结，之后只改这一行。机器合同 [contract.json](contract.json)，源码 [code/](code/)。出发点 [RESTART.md](../asr-mechanism-restart-2026-10-15/RESTART.md)，回答其 §5 的 **M1**（齐到什么程度）与 **M2**（连续量加速与否是否与齐独立）。零 GPU，只读已有数据。

## 0 要检验的机制主张

ASR 的陡升主要由**齐**造成：同一次训练里，几十个内容不同的触发输入的翻转点挤在一条窄带里，一个共同量扫过这条带时它们几乎同时翻转。连续量自身是否加速是**次要且可变**的。

两个可证伪的推论：
1. **齐在所有已装上后门的模型族里都成立**（M1）。若某族不齐而 ASR 仍陡，主张错。
2. **加速与齐无关**：既存在齐 + 不加速（toy），也存在齐 + 加速（LLaVA）。若加速与齐总是同时出现，则无法区分二者，主张降为描述（M2）。

## 1 模型族与数据（路径与格式已核对）

每族只用 `*_poison` 臂；`*_clean` 臂作为对照只报 ASR 恒为 0 的核对。

| 族 | 路径（服务器 `/workspace/cross-model-asr/`） | 每点读数 | 间隔 |
|---|---|---|---|
| Qwen3-8B SQuAD | `20261004_lora/runs/llm_s*_poison`、`20261004_seed10/**/llm_s*_poison` | `samples-*.jsonl`，60 题，`trigger/clean/near` 各含 `margin`、`target_match` | 20 步 |
| Llama-3.1-8B | `20261007_vim_llama_v1/llama/runs/formal_s*_poison` | 同上 | 20 步 |
| ResNet50 | `20261005_classification5/runs/resnet50_formal_s*_poison`、`20261006_classification10_dual_r03/**/resnet50_*_poison` | `scores/primary-*.npz`：`labels`、`clean_logits`、`triggered_logits`（180×10） | 5 步 |
| ViT-B/16 | 同上（`vit_b_16_*`） | 同上 | 5 步 |
| Vim-S | `20261007_vim_llama_v1/vim/runs/vim_small_formal_s*_poison` | 同上 | 5 步 |
| CLIP-ViT-B/32 | `20261008_followup_v3_vimdirect_r05/clip/runs/clip_vit_b32_formal_s*_poison` | 同上 | 5 步 |
| Qwen3-8B toy | 本地：Q1 回放 3 条 + 读数实验 8 条 | 逐更新 60 行 `gap`、`hit` | 1 步 |

**不纳入**：FLAN-T5、SD3.5（测过的剂量下未装上，无起跳即无窗，不是反例）；Falcon-Mamba（装不稳、剂量不单调）；LLaVA / Qwen-VL（只有每题对错、无 margin——另列 §5 的时间侧单独处理）。分类族目标类 = 0，标签为 0 的探针不计入（无法翻转到自身类）。

## 2 每条训练的量（所有族同一套定义）

- **每输入连续量** x_i(t)：
  - 语言族：`trigger.margin` = logit(目标首 token) − logit(正确首 token)；
  - 分类族：`triggered_logits[i, 0] − triggered_logits[i, label_i]`。
- **翻转** f_i(t)：语言族 `trigger.target_match`；分类族 `argmax(triggered_logits_i) == 0`。等价于 x_i > 0 的一致率在 §4 作为闸门报告。
- ASR(t) = f 的输入均值。窗 [a, b]：b = ASR 首次 ≥ 0.90 的点，a = b 之前最后一次 ≤ 0.10 的点；窗前段 p = a − (b − a)，取不超过 p 的最近点。
- **S1 齐（水平）**：θ_i = 加性拟合 x ≈ c(t) + θ_i 的输入偏移（r2 `decomp.d2` 原样）；W = θ 的 5–95% 分位宽度（logit）。行程 T = c(b) − min_{t ≤ b} c(t)。**R_level = W / T**。
- **S2 齐（时间）**：t_i = 输入 i 首次翻转的点；W_time = t_i 的 5–95% 宽度（更新数）。**R_time = W_time / a**。
- **M2 加速**：v_pre = c 在 [p, a] 的斜率，v_in = 在 [a, b] 的斜率，**比值 = v_in / v_pre**（v_pre ≤ 0 记 ∞）；D3 = 自身斜率在窗内 > 1.5 × 窗前的输入占比（r2 `decomp.d3` 原样）。
- **E 解释力**：由齐与速度推出的宽度 Ŵ = W / v_in，与实测窗宽 (b − a) 比较，报 log2(Ŵ / (b − a))。
- **对照**：语言族 `clean.margin`、`near.margin` 的 ASR 与 R_level；分类族 `clean_logits` 的同名量。

## 3 判档（写在计算之前）

逐条训练判，再按族汇总（族内多数；平手记 mixed）。

| 项 | 档 |
|---|---|
| **S1** | tight: R_level ≤ 0.25；loose: ≥ 0.5；其余 middling |
| **S2** | tight: R_time ≤ 0.25；loose: ≥ 0.5；其余 middling |
| **M2** | accelerates: 比值 ≥ 3 且 D3 ≥ 0.8；steady: 比值 ≤ 1.5 且 D3 ≤ 0.5；其余 between |
| **E** | accounted: \|log2(Ŵ / (b−a))\| ≤ 1；否则 not_accounted |

**总档（M1）**：≥ 6/7 族 S1 tight → **synchrony_universal**；≤ 3/7 → refuted；其余 partial。
**总档（M2）**：同时存在 ≥ 1 族 steady 与 ≥ 1 族 accelerates → **acceleration_independent**；全部 accelerates 或全部 steady → not_separable（主张降为描述）。

闸门：翻转与 x_i > 0 的一致率 < 0.90 的训练记 gate_failed，不计入判档（Q1 在 toy 上遇到过：第三个 token 压过目标与正确答案）。无窗（ASR never ≥ 0.90）的训练记 no_window，不计入。结果不利不改判据。

## 4 预注册预测

- M1 **synchrony_universal**。依据：toy 11 条跨行 sd 1–5 logit、LLaVA 5 个顺序窗宽 20–60 步而平台深仅 5–7 logit。
- M2 **acceleration_independent**。依据：toy 11 条 steady，LLaVA 6/9 臂窗内加速 2–16 倍（B1）。
- E 多数 accounted；分类族（全参、3–8 步装满）最可能 not_accounted，因为 5 步间隔下窗宽只有 1–2 个点。
- 风险：分类族的时间分辨率不足以定 v_pre；若 ≥ 2 族因此 gate/无窗，M1 的 6/7 门槛达不到，总档会落在 partial。

## 5 LLaVA / Qwen-VL（只有每题对错，单独报，不计入判档）

用每题二值 flag 算 S2（R_time）与 ASR 窗；不能算 S1、M2、E。LLaVA 9 条 1% + 剂量线，Qwen-VL 5% 4 条。作为 S2 的跨族补充证据报告。

## 6 边界

- 不同族的剂量、微调方式、任务、预算各不相同且互相混杂；本图谱**不做跨族因果归因**，只报每族各自的齐与加速档位。
- 分类族的 180 探针与语言族的 60 题不是同一难度分布；R_level、R_time 都是族内无量纲比值，不跨族比较绝对值。
- 齐的**载体**（是否一个共享特征）不在本阶段，属 M3。

## 7 资源

零 GPU。读取在服务器上以 CPU 进行（只读，不写入任何实验目录）；toy 族在本地算。预计读取约 3 万个小文件、数分钟。输出 `results/atlas.json`、`results/ATLAS.md`。
