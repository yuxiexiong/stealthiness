# Lite Transformer with Long-Short Range Attention

全文已读完（正文 + 附录 A.1）。Fig 1/4 曲线在文本中为乱码字体（无法读值）；Fig 2/5 读得到柱状数字；表格可读。

## 0. 元信息
ICLR 2020。Wu*, Liu*, Lin, Lin, Han。类型：算法（手工高效架构）。

## 1. 核心主张
- 主主张：在移动设定（<500M Mult-Adds、<10M 参数）下，LSRA（一半通道走注意力抓全局、一半走轻量卷积抓局部）+ 展平瓶颈的 Lite Transformer 稳定优于 Transformer：WMT'14 En-Fr 在 500M/100M Mult-Adds 下高 1.2/1.7 BLEU（Abstract, Table 2）。
- 次主张：(a) 1-D 注意力（N 小）下瓶颈结构并不划算，计算主要在 FFN（Fig 2）；(b) 手工设计在移动设定下比 Evolved Transformer 高 0.5 BLEU，设计成本少 20000× CO2（Table 3）——"AutoML 不是万能药"；(c) 可与剪枝 + 量化组合压缩 18.2×（Fig 5）；(d) 推广到摘要与语言模型（Table 4/5）。

## 2. 动机是怎么立住的
- **Fig 2（计算分解 profiling）**：Base Transformer 的 Mult-Adds 中 FFN 占大头（读到的柱值：Base 1,336M 中 FFN 754 / 注意力 582 左右；展平后注意力占比上升；LSRA 总量降 2.5×，括号里是 BLEU 39.9 / 41.0 / 39.6）。这是本文自己的计算分解，直接引出"展平瓶颈、把算力让给注意力"。
- 复杂度解析：注意力 O(4Nd²+N²d) vs FFN O(8Nd²)，N≈20–30 时瓶颈省不了多少（Sec 3）——**推导型**动机。
- **Fig 3 / Fig A1（注意力可视化）**：普通注意力图有强对角线（局部）+ 稀疏点（全局），说明注意力在"浪费"容量做局部建模；LSRA 中注意力分支对角线消失。这是"观察 → 专门化"的主要定性证据。
- 移动设定的定义也有推导：ARM A72 约 48 GFLOPS，50 句/秒 → <480M Mult-Adds（Sec 5.1）。
- 因果链：计算分解 → 展平；注意力可视化 → 分支专门化。但"展平"与"LSRA"两个改动的贡献在主结果中未分开（Fig 2 只给了 Flattened 的一个 BLEU 点 41.0）。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| FFN 占主导、瓶颈不划算 | 计算分解 | Fig 2, Sec 3 | Base / Flattened / LSRA | Mult-Adds、BLEU |
| 注意力分支专门化为全局 | 注意力热图 | Fig 3, Fig A1 | 普通 Transformer | 定性 |
| IWSLT 上优于 Transformer、LightConv | 3 个计算档 | Table 1 | Transformer、LightConv | BLEU、参数、Mult-Adds |
| WMT 上优于 Transformer | 3 个计算档 × 2 任务 | Table 2, Fig 4a | Transformer | BLEU |
| 优于 ET 且便宜 | 100M / 300M 档 | Table 3 | Transformer、ET | BLEU、GPU h、CO2、$ |
| 可压缩 18.2× | 8bit k-means + 敏感度剪枝 | Fig 5 | Transformer FP32 | MB、BLEU |
| 摘要上省 2.4× 计算 | CNN-DM | Table 4 | Transformer | ROUGE、Mult-Adds |
| LM 上更低 PPL | WikiText-103 | Table 5, Fig 4b | Adaptive Inputs | PPL、tokens/s |
| "移动端高效" | 「未直接验证」——除 LM 的 tokens/s 外无任何延迟实测 | — | — | — |

## 4. 实验设计手法
- 基线：Transformer（同 fairseq 实现、同训练设置）、LightConv（同作者 Wu et al. 2019b 的最强卷积基线，仅 IWSLT）、Evolved Transformer（NAS 代表）、Adaptive Inputs（LM）。对 Transformer 做了**多档计算量下的等 Mult-Adds 对比**（Table 1/2 每档三行）。dropout 随维度线性缩小（Sec 5.4）——对基线和自己同样处理，属于细心的公平性设置。
- 规模/广度：3 类任务（翻译 3 数据集、摘要、语言模型）。
- 效率测量：**只有 Mult-Adds 和参数量**（翻译按 30→30 token 计算）；唯一的实测是 LM 的 tokens/s（Table 5，硬件未注明）。没有任何移动设备延迟——尽管标题与动机都是"移动端"。
- 精度–效率联合：每个任务都在多个计算档报告（Table 1/2 三档），Fig 4 给 BLEU/PPL vs Mult-Adds 曲线并标出移动区域。
- 消融：极薄——只有 Fig 2 中 Base / Flattened / LSRA 三点；无卷积核大小、分支比例、GLU、动态核的消融。
- 分析性实验：注意力热图（Fig 3, A1），算力分解（Fig 2）。
- 泛化：跨 3 类 NLP 任务。
- 统计：无方差；IWSLT 取最后 10 个 checkpoint 平均、WMT 取验证最优，这是标准做法但不是多 seed。
- 系统贡献：开源。
- 局限性：无。

## 5. 对抗式审读
- 最薄弱环节：效率主张完全建立在 Mult-Adds 上。同组后来的 HAT（Sec 5）明确批评 Lite Transformer "relied on FLOPs … an inaccurate proxy for hardware latency"。LSRA 中的卷积分支（动态核、GLU）在移动 CPU 上的实际开销未知。
- LightConv 只在 IWSLT 比较；WMT 上未与 LightConv/DynamicConv 比较，而 Lite Transformer 的卷积分支正是取自它们，这是最相关的基线缺失。
- 注意力可视化是单层、少量句子、头平均的定性证据，"注意力专门化"未量化（例如对角线能量占比）。
- 与 ET 的比较：ET 的 GPU 小时与 CO2 引自 Strubell 估算；Lite Transformer 本身的人工设计成本（迭代试验）未计入。
- 18.2× 压缩使用自家已有方法，属"正交可组合"展示而非新贡献。

## 6. 可迁移手法
1. 先做计算量分解（哪个模块吃掉多少 Mult-Adds），用分解结果反驳一个"约定俗成"的设计（瓶颈），引出改动（Fig 2）。—直接读到
2. 在多个计算预算档位上各做一次等计算量对比，并强调"预算越紧增益越大"（Table 1/2）。—直接读到
3. 用注意力热图的前后对比给出"专门化"的机制性证据（Fig 3 vs Fig A1）。—直接读到
4. 从硬件峰值算力和目标吞吐反推出"移动设定"约束值，让约束看起来有物理依据（Sec 5.1）。—直接读到
5. 以"人工设计 vs 昂贵 NAS"的成本 + 精度双重对比来提出立场性论点（Table 3, Sec 6.2）。—直接读到
6. 我的归纳：同一实验室在动机中用 FLOPs 论证、而下一篇（HAT）用实测延迟推翻 FLOPs——只报 FLOPs 的高效架构论文最容易在后续被自家或他人反驳。

## 7. 一句话
一张计算分解柱状图（Fig 2）+ 一组注意力热图（Fig 3）把"展平瓶颈、分支专门化"讲成被观察逼出来的设计，再用多档等 Mult-Adds 表格逐档压过 Transformer；但"移动端"三字没有一个实测延迟支撑。
