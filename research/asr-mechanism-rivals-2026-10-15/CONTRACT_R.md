# 阶段 R：先杀零假设 · 冻结合同

日期：2026-10-15。状态：**冻结（任何阶段 R 数字出现之前）**。之后只改这一行。机器阈值 [contract.json](contract.json)、源码 [code/](code/)（CPU 测试 19/19）。依据 [RIVALS.md](RIVALS.md) v1.1、[PLAN.md](PLAN.md)。

## 1 要删的候选

| 块 | 候选对 | 本阶段能删谁 |
|---|---|---|
| R1 | A1 / A2 / A 组继承-形成；B0 调度 | A1、A2（若不泄漏）、B0 调度部分 |
| R2-K | B0′ vs {B1, B2, B4}；B4 vs 密度不敏感；C3 vs C5 | B0′ 或 B1（分类族）；B4；C3 或 C5（分类族） |
| R2-L | C0 vs {C2, C3, C4}；B4 | C0 或步数时钟一方；B4（语言族） |

## 2 共同的量（[code/r_common.py](code/r_common.py)）

- x_i(t)：语言族 = logit(目标首 token) − logit(正确首 token)；分类族 = logit(目标类) − logit(真实类)。翻转：语言族 target_match，分类族 argmax = 目标类。标签为目标类的探针不计。c(t) = x 的输入均值。
- t50、t90：ASR 首次 ≥ 0.5 / ≥ 0.9 的点。
- **A_half**（本阶段主加速量）：c 在 [s0 + (t90 − s0)/2, t90] 的平均速度 ÷ 在 [s0, 中点] 的平均速度，s0 为第一个测点。t90 − s0 < 4 时不定义。理由：图谱比值在窗起点 a = 0 时不定义，而 K1、K2 可能从第 0 步就动。图谱比值照常报，不判档。
- **v_mean** = (c(t90) − c(s0)) / (t90 − s0)。
- **cumLR(t)** = 第 1…t 次更新所用 LR 之和。
- t_clean80：干净条件下 180 个探针（语言族 60 个）的正确率首次 ≥ 0.80 的点。

## 3 条件

### 3.1 R1 语料（零 GPU，[code/r_corpus.py](code/r_corpus.py)）

族与运行的映射与图谱合同 §1 相同（不含 CLIP，闸门不过；不含 toy）。每条运行在图谱窗 [p, b] 内算：

- additive_share：x ≈ c(t) + o_i 解释的方差占比；
- o_i = 窗内 x_i − c 的均值（带内位置，越高越早翻）；
- inherit_rho：Spearman(o_i, x_i(第一个测点))。注意语言族密集探针集的第一个测点是第 20 步，不是第 0 步；
- 跨种子：同族两两运行在共同探针上的 Spearman(o) 中位数；
- leak_slope：干净条件 c_clean 对触发 c 的 OLS 斜率；
- 语言族 lr_ratio = 窗内平均 LR ÷ 窗前平均 LR，sched_log2 = log2(图谱比值 / lr_ratio)。

### 3.2 R2-K 分类（[code/r_cls.py](code/r_cls.py)）

ViT-B/16、ResNet50；种子 1001–1003；协议 = 20261005_classification5 的正式训练（常数 LR 1e-4 AdamW、批 128、梯度裁剪 1、5% 毒、目标类 0），模块原样导入。每次更新后测 180 个 primary 探针（K2 第 120 步后每 5 步）。不存模型。

| 条件 | 改动 | 更新数 |
|---|---|---|
| K0 | 原样，从共享的干净适配起点 | 120 |
| K1 | 从干净起点用干净臂（epoch 1 的顺序、无毒）跑 100 步只为预热 AdamW 状态；权重复原到干净起点（哈希核对）；保留优化器状态；然后跑 K0 的数据流 | 120 |
| K2 | ImageNet 权重 + 新随机 10 类头，不做 5 epoch 干净适配；K0 的数据流 | 300 |
| K3 | 毒集 = 正式 2,250 中固定一半（Random(41001)）；去掉的图回到真实标签 | 120 |
| K4 | 毒集 = 正式 2,250 + 2,250 张额外非目标类训练图（Random(51001)） | 120 |

### 3.3 R2-L 语言（[code/r_llm.py](code/r_llm.py)）

Qwen3-8B SQuAD LoRA（r 16、批 16、裁剪 1、1% 毒），种子 1001 / 1004 / 1005，200 次更新，第 0 步与每 5 步测 60 个发现探针（触发 / 干净 / 近触发）。不存适配器，只记每 20 步哈希。

| 条件 | 改动 |
|---|---|
| L0 | 原样（38 步线性 warmup，余弦到 1,250） |
| L1 | 去掉 warmup：lr = 1e-4 (1 + cos(π·step/1250)) / 2，从第 1 次更新起 |
| L2 | 前 3,200 行里的毒行固定一半（Random(41001)）改回干净 |
| L3 | 前 3,200 行里同等数量的干净行（Random(51001)）改为毒行 |

## 4 判档（写在计算之前）

**R1，按族**（中位数）：A1 additive_share ≥ 0.80 → deleted，否则 alive；跨种子 Spearman ≥ 0.6 stable，≤ 0.2 seed_specific；inherit_rho ≥ 0.6 inherited，≤ 0.2 formed；leak_slope ≥ 0.5 leak，≤ 0.1 specific，否则 partial；语言族 |sched_log2| ≤ 0.5 的运行过半 → schedule_explained，否则 not_schedule。

**B0′（每个模型，三种子中位数）**：K1 的 A_half ≤ 1.5 且 K0 的 A_half ≥ 2 × K1 → **transient**（B0′ 保留，分类族加速判为优化器暂态，B1 在分类族删除）；K1 ≥ 0.7 × K0 → **intrinsic**（B0′ 删除）；其余 partial。

**B4（每个平台）**：v_mean 对名义密度（K: 2.5 / 5 / 10%；L: 0.5 / 1 / 2%）的 log-log 斜率 e_v，t90 的斜率 e_t。e_v ≥ 0.7 且 e_t ≤ −0.7 → **arrival_rate**；|e_v| ≤ 0.3 → **density_insensitive**（B4 删除）；其余 partial。

**C3 vs C5（每个模型，逐种子再多数）**：K2 的 t90 < K2 的 t_clean80（或干净正确率从未到 0.8）→ **shortcut**（C5 成立，C3 在分类族删除）；K2 的 t90 ≥ t_clean80 且比 K0 晚 ≥ 10 步 → **gated**（C3 得到因果支持）；其余 mixed；K2 不装 → no_install。另报 K2 与 K0 的 A_half 中位数（B2：干净任务在窗内上升时是否更加速），不判档。

**C0（逐种子再多数）**：|cumLR_L1(t90_L1) / cumLR_L0(t90_L0) − 1| ≤ 0.25 且 t90_L1 ≤ t90_L0 − 10 → **lr_clock**（C0 成立）；|t90_L1 − t90_L0| ≤ 10 → **step_clock**（C0 删除）；其余 mixed。另报 L0 与 L1 的 A_half（减速是否随调度消失），不判档。

## 5 闸门与止损

- **工程门**：每个 K0 与原始正式训练前 120 次更新的 loss 逐位相等，且 0、5、…、120 步的触发 logits 最大差 ≤ 1e-3；每个 L0 与原始训练前 200 次 loss 逐位相等，且 20、40、…、200 步适配器哈希相等。不过门的平台：判档照算但标记不判，先查工程原因、修正后重跑该平台。
- **结果不利不改判据**；判档只删 RIVALS 中对应的候选，不新增候选。异常进 RIVALS §10 停车场。
- 阶段 R 全部判完后停下，更新 RIVALS 状态与证据表，与用户讨论阶段 S，不自动开跑。

## 6 预注册预测

| 项 | 预测 | 理由 |
|---|---|---|
| R1 A1 | deleted（所有族） | 图谱分解式 41/50 成立 |
| R1 稳定性 | stable（语言族），middling（分类族） | 语言探针难度差异大 |
| R1 继承 | inherited | 带内位置多半由基座模型对各输入的先验决定 |
| R1 泄漏 | specific | 各族干净臂 ASR 恒 0 |
| R1 调度 | not_schedule | SQuAD 窗前跨 warmup 却减速 |
| B0′ | ViT transient；ResNet partial | ViT 窗前近平、窗在第 15–35 步，正落在一阶矩建立期 |
| B4 | partial（两个平台） | Adam 归一化步长会削弱密度效应，但不会抹掉 |
| C3 vs C5 | shortcut（两个模型） | 触发块是低层像素特征，可绕开正常类别特征 |
| C0 | lr_clock | toy 分叉里学习率减半推迟 4–30 步，推力 ∝ LR |

## 7 边界

分类族与语言族的结论分别成立，不跨族合并。K2 与 K0 的差别同时包含「干净任务未学会」与「头是随机的」两件事，C 档只回答「后门是否需要先学会正常任务」，不区分这两件事。R1 不含 toy（一个不典型设置）与 CLIP（图谱闸门不过）。
