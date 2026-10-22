# MCUNet: Tiny Deep Learning on IoT Devices

## 0. 元信息
- NeurIPS 2020（arXiv 2007.10319v2，camera-ready 版）
- 类型：算法-系统协同（TinyNAS 搜索 + TinyEngine 推理库）
- 全文含附录 A–F 已读完。pdftotext 把 Table 2/3/4 与 Figure 8 的数字搅在一起（txt 第 413–458 行），下文数字是按正文叙述（"35.2%/49.5%… 47.4%/56.4%… 55.5%… 61.8%"）反推对齐的，Table 4 行列按表头顺序对齐，有一定错位风险。

## 1. 核心主张
- 主主张：只有把模型结构（TinyNAS）与推理库（TinyEngine）**联合**设计，才能在商用 MCU（320kB SRAM/1MB Flash）上跑 ImageNet 级模型；首次在商用 MCU 上 >70% ImageNet top-1（70.7%，STM32H743）。
- 次要主张：(a) 推理库不只影响速度也影响**可达精度**（因为决定了能放下多大的模型）；(b) 用"满足约束的子网 FLOPs 分布"挑搜索空间，比随机/超大空间好；(c) TinyEngine 比 TF-Lite Micro/CMSIS-NN 省内存 3.4×、快 1.7–3.3×。

## 2. 动机是怎么立住的
- Table 1：云/手机/MCU 的内存与存储对比（V100 16GB vs STM32F746 320kB，差 3–6 个量级），以及 ResNet-50、MbV2、int8 MbV2 的峰值内存/存储都超限（int8 MbV2 仍超 5.3×）。
- Figure 1 关键观察：同为 ~70% ImageNet，MbV2 比 ResNet-18 参数小 4.6×，但**峰值激活大 1.8×**——"为 FLOPs/参数优化的网络并不为 SRAM 优化"。这是把痛点从"模型大"改写成"峰值激活大"的关键一步。
- Section 2：解释器式框架额外占用"最高 65% 峰值内存"、延迟 +22%（无图表号，未展示测量方法——作者声称）。
- Figure 11（分析节）事后补强了观察：0.3× MbV2 的单块峰值激活是平均值的 2.2×，失衡导致其他块被迫缩小。
- 因果链：峰值激活是瓶颈 → 推理库省内存可扩大可行设计空间 → NAS 在更大空间找到更高容量模型。Table 2 的 2×2 设计直接检验这个链，闭合度较好。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 协同设计优于单独任一侧 | 库 {CMSIS-NN, TinyEngine} × 模型 {S-MbV2, S-Proxyless, TinyNAS} 的 2×3 网格 | Table 2 | 同一 320kB/1MB 约束下的最优缩放基线 | ImageNet top-1：35.2/49.5/55.5 → 47.4/56.4/61.8 |
| 协同设计在延迟约束下仍有效 | 5FPS / 10FPS 约束 | Table 3 | S-MbV2+CMSIS, S-MbV2+TinyEngine | top-1（TinyEngine +1.9/+5.4，MCUNet 再 +8.3/+6.4） |
| 跨 MCU、低比特下达 SOTA | 4 种 MCU × 4-bit | Table 4 | Rusci et al.（混合精度） | top-1，F412 62.0 vs 60.2、H743 70.7 vs 68.0 |
| 同精度更省内存 | 与 ResNet-18 / MbV2-0.75 int8 | Figure 8 | 同 ~70% 精度 | SRAM 3.5×、Flash 5.7× 更小 |
| TinyEngine 更快更省 | 4 个模型 × 4 个库的延迟/峰值内存 | Figure 4 | TF-Lite Micro, MicroTVM Tuned, CMSIS-NN | 实测 latency、peak mem、OOM |
| TinyEngine 各技术的贡献 | 逐项累加 | Figure 5 | CMSIS-NN 基线 | MMAC/s 52→82，peak mem 160→76→48 |
| 二进制更小 | | Figure 6 | TFLM, CMSIS-NN | binary kB，最高 5.0× |
| 唤醒词任务帕累托更优 | VWW / GSC 精度-延迟、精度-SRAM 曲线 | Figure 9 | 最优缩放 MbV2/Proxyless（TFLM），VWW'19 冠军方案（同设置重训） | 2.4–3.4× 快、3.7–4.1× 小 |
| 迁移到检测 | Pascal VOC + YOLOv2 | Table 5 | MbV2+CMSIS | mAP 31.6%(且 OOM) → 51.4% |
| 搜索空间优化有效 | 优化空间 vs 随机空间 vs 超大空间 | Table 6, Figure 10 | 同 NAS 流程 | ImageNet-100 top-1 78.7 vs 74.7±1.9 vs 77.0 |
| FLOPs 代理与最终精度相关 | 空间均值 FLOPs vs 最终精度 | Figure 10 | — | 正相关（只给图，无相关系数） |
| 搜索成本低 | 设计成本 | 附录 C, Figure 14 | MnasNet | 300 vs 40,000 GPU 小时 |
| "推理库影响可达精度" | 在 CMSIS 上做进化搜索 | Figure 13 | evolution/random on TinyEngine | 搜索曲线 |

## 4. 实验设计手法
- **基线选择**：基线不是拿原版 MbV2 直接比（会 OOM），而是"compound scaling 宽度×分辨率直到满足内存，训练并评估 Pareto 前沿上所有可行模型，取最高精度"（§4.2 及脚注）——这是对基线非常友好的构造，堵住"基线没缩放好"的攻击。VWW 冠军方案也"按同约束缩放并在同设置下重训"。有朴素基线：随机搜索（Fig 13）、随机采样的搜索空间（Table 6）。说明不用 MobileNetV3 的理由（hard-swish 在 MCU 上不高效）。
- **规模/广度**：3 个数据集（ImageNet、VWW、Speech Commands）+ 检测（VOC）；4 款 MCU（F412/F746/F765/H743，M4 和 M7）；明确拒用 CIFAR（"32×32 不代表真实场景"，§4.1）。
- **效率测量**：全部真机实测延迟与峰值 SRAM；延迟归一到 F746@216MHz；附录 A 交代 demo 中摄像头采集开销使 10FPS 降到 7.3FPS。峰值内存在 Fig 11 明说"不含其他运行时缓冲"。
- **精度–效率联合呈现**：Figure 9 精度-延迟、精度-SRAM 两条帕累托；Table 3 固定延迟比精度；Figure 8 固定精度比内存。三种呈现都有。
- **消融**：Figure 5 是标准"roadmap"逐步累加（代码生成→专用 im2col→算子融合→循环展开→tiling；in-place depthwise 单列内存）；Table 2 的 2×2 是系统-算法因子拆分；Table 6 拆搜索空间优化；Fig 13 拆搜索算法（evolution vs random）。
- **分析性实验**：Fig 11 逐块峰值激活（解释 TinyNAS 为什么赢）；Fig 12 不同 SRAM/Flash 约束下选出的宽度/分辨率模式（"SRAM 大→分辨率高、Flash 大→宽度大"）。
- **泛化**：跨 MCU、跨任务（分类/唤醒词/音频/检测）。
- **统计**：Table 6 随机空间给了 ±1.9（多个随机空间的方差）；Fig 13 给 min/max 阴影。主结果单次。
- **系统贡献**：demo 视频（附录 A）、tinyml.mit.edu；正文未明说开源代码。
- **局限性**：无专门讨论；只有 Broader Impact。

## 5. 对抗式审读
- **最薄弱环节**：搜索空间优化（"均值 FLOPs 越大越好"）只在 ImageNet-100 + 一个约束上验证，Fig 10 只有少量点、无相关系数；Table 6 的"huge space 更差"用"超网更难训"的**假设**解释，未验证。
- **比较公平性**：Table 2 的 CMSIS 列 S-MbV2 是 35.2%，Table 3 同约束 N/A 列 S-MbV2+CMSIS 写 39.7%，两表数字不一致（可能来自 v2 更新时部分表重算，见附录 F），正文未解释。Figure 9 基线跑在 TF-Lite Micro 上而 MCUNet 跑在 TinyEngine 上，加速倍数混合了库与模型两种贡献（正文也承认是"co-design"，但 2.4–3.4× 不应被读作模型本身更快）。最终 ImageNet 模型额外微调 100 epoch（附录 E），基线是否同等训练未写明。
- **超出支持范围**："era of always-on tiny ML has arrived" 是修辞；功耗/能耗主张（摘要说可处理 energy 约束）正文没有任何能耗测量。
- Table 4 与 Rusci et al. 的对比：Rusci 用混合精度、MCUNet 用 4-bit + QAT 25 epoch，训练预算不同。
- 解释器开销"65% 峰值内存、22% 延迟"未给测量依据。

## 6. 可迁移的手法
1. 当贡献是"两部分的协同"时，用 2×N 因子网格（库 × 模型）同时展示"各自有用 + 组合最好"，而不是只报端到端。（Table 2）**直接读到**
2. 当基线在你的约束下根本跑不了（OOM）时，把基线按复合缩放降到可行，训练 Pareto 前沿上所有可行点取最好者，用"给基线最优待遇"消解不公平质疑。（§4.2 脚注）**直接读到**
3. 当想说"系统层改动影响算法结果"时，在弱系统上跑同一搜索算法，展示搜索天花板被压低。（Fig 13）**直接读到**
4. 用一张"同精度下资源差几倍"的反直觉对照（MbV2 参数更小但激活更大）把痛点从常识指标（FLOPs/参数）转移到新指标（峰值激活）。（Fig 1）**我的归纳**
5. 用"roadmap"式逐项累加图把系统优化拆成可归因的增量。（Fig 5）**直接读到**
6. 主动拒绝弱基准（CIFAR）并说明理由，提高结果的外部可信度。（§4.1）**直接读到**

## 7. 一句话
Table 2 的"库 × 模型"因子网格配合对基线的最优缩放，让"协同设计"不是口号而是可分解的增量。
