# Tiny Machine Learning: Progress and Futures

## 0. 元信息
- IEEE Circuits and Systems Magazine 2023（arXiv 2403.19076v2）
- 类型：综述（以自家 MCUNet 系列为主体的 review + 一个新的统一基准测量表）
- 全文已读（正文 I–V 节 + 参考文献；无附录）。两栏排版在 pdftotext 中左右栏交错，段落需拼接；多数图（Fig 1–21）在 txt 中只剩图题。表格 II–VII 基本完整。

## 1. 核心主张
- 主主张：TinyML 不能靠把云/移动模型缩小得到，必须**算法-系统协同设计**，推理（MCUNet V1/V2）和训练（MCUNetV3/TinyTraining）都如此。
- 次要主张：(a) TinyML 的三个独特挑战：移动模型不适合 MCU、推理模型直接用于训练行不通、协同设计必需（§I-A）；(b) 在统一设备上测，TinyEngine 在延迟和存储上优于 CMSIS-NN、X-Cube-AI、TF-Lite Micro（Table III）；(c) TinyML 的范围随硬件演化而变（"今天的大模型是明天的小模型"，§V）。

## 2. 动机是怎么立住的
- Table I（复用 MCUNet Table 1）：云/移动/MCU 内存与存储差 3–6 个量级。
- Fig 2（关键观察，图本身在 txt 中缺失，数字取自图题）：MbV2-1.4 比 ResNet-50 参数少 4.2×，但推理峰值内存**多 2.3×**、训练内存只少 1.1×，且训练内存是推理的 6.9×。把 MCUNet（推理激活）和 TinyTL（训练激活）两个观察合并成一张图。
- 综述类论文，动机靠的是已发表结果的整合，不是新实验。

## 3. 主张 → 证据映射
| 主张 | 证据 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 领域现状横向比较 | 18 个工作的规格表（设备、数据集、模型、分辨率、位宽、延迟、峰值内存、Flash、能耗、精度） | Table II | CMSIS-NN … MCUNetV3 | 多项；脚注标注哪些数字是别人论文测的、哪些是"按 MCU 规格推测" |
| TinyEngine 在同一设备上最好 | **新测量**：STM32H743@480MHz，10 个模型 × 4 个推理框架 | Table III | CMSIS-NN, X-Cube-AI v7.3.0, TinyEngine, TF-Lite Micro（截至 2022-12-19 最新版） | 延迟、峰值内存、存储 |
| TinyEngine 更快 | | Fig 7（图缺失，图题） | TFLM / CMSIS-NN / X-Cube-AI | 最高 22× / 2.3× / 1.5× |
| 协同设计带来 ImageNet、VWW、检测、人脸检测提升 | 复用 MCUNetV2 结果 | Table IV/V/VI, Fig 14 | 同 MCUNetV2 | 同 MCUNetV2 |
| QAS 与稀疏更新 | 复用 MCUNetV3 结果 | Table VII, Fig 15–21 | 同 MCUNetV3 | 同 MCUNetV3 |

## 4. 实验设计手法
- **基线选择**：Table III 是本文唯一新实验，亮点是**统一设备、统一模型、统一日期版本**地测四个框架；X-Cube-AI（厂商闭源工具）也纳入，并注明编译设置（"balanced optimization"）。
- **规模/广度**：Table III 10 个模型（mcunet-vww0/1/2、mcunet-in0–4、proxyless-w0.3 两种分辨率、mbv2-w0.3），2 个数据集。Table II 覆盖 18 个工作。
- **效率测量**：Table III 实测延迟/峰值内存/存储，设备与时钟明确。Table II 混合了多来源数据，用脚注 1–4 区分"CMix-NN 测""按规格推测""MCUNet 测""作者邮件确认单位应为 mJ 而非 µJ"——对二手数据的来源做了显式标注。
- **精度–效率联合呈现**：Table III 每组模型给出精度 + 三项效率；其余复用。
- **消融**：无新消融。
- **分析性实验**：无新分析。
- **泛化**：无新增。
- **统计**：无。
- **系统贡献**：模型公开于 github.com/mit-han-lab/mcunet（Table III 脚注 4）。
- **局限性**：§V 列未来方向：更多模态、自监督（片上无标签）、TinyML 与 LargeML 的关系。

## 5. 对抗式审读
- **综述几乎全部篇幅是自家工作**（§III、§IV 是 MCUNet/V2/V3 的转写），外部工作只在 Table II 和 §II 以短段落出现；作为"Progress and Futures"综述存在明显的选择性。
- **Table III 中 TinyEngine 并非处处最优**：峰值内存上 X-Cube-AI 在 mcunet-in3（243 vs 260KB）、mcunet-in4（342 vs 416KB）、proxyless-w0.3@176（221 vs 259KB）更低。正文（§II-A）只说"为提供更准确透明的比较"，没有讨论这些反例。值得肯定的是表里如实列出了。
- **与原论文数字不一致**：Fig 15 图题写 149KB / 2077×，MCUNetV3 原文 Fig 1 是 141KB / 2300×；Table V 图题写 +16.9%，正文写 16.7%（继承自 MCUNetV2 的同一不一致）。没有解释哪个是更新值。
- **命名漂移**：本文把 MCUNetV2 的结果标为 "MCUNet-M4/H7 (w/ patch)"，Fig 14 脚注才说明"MCUNetV2 指带 patch 的版本"，读者容易把 V1、V2 混为一谈；检测比较里 MCUNet 51.4% 用的是 YOLOv2（见 mcunetv2 笔记），本文同样未注明。
- Table II 的跨论文比较：设备、数据集、分辨率、位宽都不同，只能作为目录，不能作为排名；文中没有据此下排名结论，这点克制。

## 6. 可迁移的手法
1. 综述里加入一张"同设备、同模型、同版本日期"的新基准表，把文献里不可比的数字变成可比，提升综述的证据价值。（Table III 及脚注 1–2）**直接读到**
2. 二手数据表用脚注逐格标明来源（他人测、规格推测、作者邮件确认单位），防止"拼表"被质疑。（Table II 脚注 1–4）**直接读到**
3. 把推理与训练两个"常用指标 vs 真瓶颈"的观察合成一张对比图（参数 4.2× 小，但推理峰值 2.3× 大、训练只 1.1× 小），为整条研究线提供统一动机。（Fig 2）**直接读到**
4. 用"今天的大模型是明天的小模型"重新定义领域边界，让领域本身具有可延展性。（§V）**我的归纳**

## 7. 一句话
本文自身的说服力来自 Table III：在同一块 H743 上用同版本框架把 10 个模型重测一遍，让此前各论文"跨设备拼凑"的效率比较第一次变得可比——而它也诚实暴露了 TinyEngine 在部分模型上峰值内存不是最低。
