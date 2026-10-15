# 阶段 R1：语料打分 · 结果

2026-10-15。合同 [CONTRACT_R.md](CONTRACT_R.md) §3.1、§4（冻结于 `5ef1015`，计算前）。数据 [results/corpus.json](results/corpus.json)，日志 [results/corpus.log](results/corpus.log)。零 GPU，服务器 CPU 只读。44 条运行判 ok（Qwen SQuAD 10、Llama 5、ResNet 10、ViT 10、Vim 5；图谱闸门不过的 SQuAD s1002 在本阶段不设闸门，照算）。

## 1 判档

| 族 | A1 加性 | 跨种子稳定 | 继承 | 泄漏 | 调度 |
|---|---|---|---|---|---|
| Qwen SQuAD | **deleted** 0.97 | **stable** 0.75 | middling 0.55 | partial 0.28 | **not_schedule**（2/9 可由 LR 解释） |
| Llama | **deleted** 0.96 | **stable** 0.73 | **inherited** 0.69 | **specific** 0.01 | **not_schedule**（1/5） |
| ResNet50 | **deleted** 0.97 | **stable** 0.68 | **inherited** 0.72 | partial 0.38 | — |
| ViT-B/16 | **deleted** 0.93 | **stable** 0.73 | middling 0.48 | **specific** −0.13 | — |
| Vim-S | **deleted** 0.93 | middling 0.36 | middling 0.24 | **specific** −0.67 | — |

预测对照：A1 deleted ✓（5/5）；稳定性 语言 stable ✓、分类 middling ✗（实为 stable 2/3）；继承 inherited 只中 2/5；泄漏 specific 中 3/5；调度 not_schedule ✓（2/2）。

## 2 读到的事实

1. **A1（无共享）在五个族上都删除。** 触发输入的轨迹 92–98% 的方差由「一个共同量 + 每个输入一个固定偏移」解释。几十个输入被同一个量平移，这是带宽 W 的前提，现在是跨五族的强结论。
2. **每个输入在带里的位置是输入自身的属性，不随种子换。** 同族不同种子在共同探针上，带内位置的秩相关中位数 0.68–0.75（Vim 0.36）。同一道题、同一张图，在不同种子里大致同样早或同样晚翻转。这与「位置由训练随机性在窗内重排」不相容。
3. **位置部分来自训练之前，但不全是。** 带内位置与第一个测点时各输入 margin 的秩相关：Llama、ResNet ≥ 0.69，SQuAD、ViT 约 0.5，Vim 0.24。语言族第一个测点已是第 20 步（密集探针集从此开始），所以这里的「继承」偏高估。结合第 2 条：位置是稳定的输入属性，其中一部分在训练前的 margin 里就能读出，另一部分在训练中按稳定的方式形成。
4. **输出侧泄漏只在两个族里部分存在。** 干净提示的 margin 随触发 margin 上升的斜率：SQuAD 0.28、ResNet 0.38，其余 ≤ 0.01 或为负（Vim −0.67：触发输入往目标走时，干净输入反而离目标更远）。目标输出的全局抬升（A2）不是共享量的主体，但在 SQuAD 与 ResNet 上是一个分量。
5. **调度解释不了语言族的速度比。** 窗内与窗前的平均 LR 比在 0.99–1.25 之间（窗都在 warmup 之后，窗前段只略微跨进 warmup），而观测比值从 0.24 到 31，|log2(比值 / LR 比)| ≤ 0.5 的只有 3/14。

## 3 对候选集的处理（按 RIVALS 删除条件）

- **A1 删除**（五族）。
- **A2**：Llama、ViT、Vim 上删除（specific）；SQuAD、ResNet 保留为部分分量。
- **A3 / A4**：都与「稳定的输入属性」相容，本阶段不能分开。A4 内部的「继承 vs 形成」判为**两者都有**，按族不同；不能写成单一答案。
- **B0 调度部分删除**（语言族）。分辨率部分与 B0′ 留给 R2。

R2（GPU）仍在排队：GPU0 被另一项目的进程占用（约 3.3 GiB），队列规则要求 GPU 空闲 10 分钟后才开始。
