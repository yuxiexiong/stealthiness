# PockEngine: Sparse and Efficient Fine-tuning in a Pocket

## 0. 元信息
- MICRO 2023（arXiv 2310.17752v1）
- 类型：系统（编译式训练引擎）+ 稀疏反向传播算法的系统支持
- 全文已读（无附录）。双栏交错较重；**txt 中残留多处作者草稿批注**：图 3 区域 "TODO: 1. Update the style figures 2. Add color…"、"Make all RED to MIT RED"、Fig 7 区域 "TODO: Make executio…"、§4.1 后 "TODO: Change all LLaMa to LlamaV2 / Check meta's naming"，Figure 9 背后还嵌有带 "#VALUE!" 的电子表格单元。说明发表版 PDF 带了未清理的图层。Figure 9 柱值在 txt 中错位，只用正文复述的倍数。

## 1. 核心主张
- 主主张：一个"编译优先"的训练引擎（编译期自动微分 + 反向图剪枝 + 训练图优化 + 多后端），能在多种边缘硬件上把训练做得比 TF/PyTorch/JAX/MNN 快得多，并把稀疏反向传播的理论节省变成实测的内存/延迟节省。
- 次要主张：(a) 稀疏 BP 在视觉与语言模型上精度损失 <1%；(b) 稀疏 BP 再带来 1.3–1.6×（RPi）乃至 1.5–3.5× 加速、2.2–21.3× 内存节省；(c) 让纯推理库（SNPE、TensorRT、TVM、TinyEngine）具备训练能力；(d) 在 Jetson AGX Orin 上微调 LlamaV2-7B，比 PyTorch 快 7.9×（摘要）。

## 2. 动机是怎么立住的
- §1 用一组数字立痛点：训练内存是推理的 7.3–14×（MbV2, bs=8）、BERT 7.3×；优化器再 ×2（动量）/×3（Adam）；边缘硬件高度异构（MCU、M1、Adreno、Hexagon、Edge TPU）。**没有图表或测量方法说明**，属于作者声称。
- Table 1：框架特性对比表（是否支持训练 / 稀疏 BP / 无宿主语言 / 边缘内核 / 编译期 AD / 图优化）——用"勾叉表"定位空白，属于定性论证。
- Figure 3：线性层前/反向计算图，标出"存激活"这条红线是瓶颈，bias-only/稀疏层/稀疏张量分别如何绕开它。这是继承自 TinyTL 式 2 的机制论证。
- 缺少一个本文自测的"现有框架在边缘训练有多慢"的 profiling 分解；动机主要靠 Table 1 与先前工作。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 稀疏 BP 视觉精度损失 <1% | 3 个 CNN × 7 个数据集 | Table 2 | Full BP、Bias Only | 平均 acc（MCUNet 74.1/72.7/74.8；MbV2 89.2/87.3/88.5；R-50 90.5/87.8/90.3），带 ± |
| 稀疏 BP 语言精度不降 | DistilBERT/BERT × GLUE 7 项 | Table 3 | Full BP、Bias Only | 平均（BERT 81.8/78.1/81.7），3 次运行均值±std |
| 稀疏 BP 收敛 | BERT QNLI/SST-2 loss | Figure 8（图缺失） | FT-Full | loss 曲线 |
| 稀疏 BP 省内存 | Jetson / MCU 实测 | Table 4 | Full BP | MCUNet 3.6MB→173KB；BERT bs16 5.7→2.3GB；Llama 43.1→31.2GB |
| 引擎更快 | 7 个平台 × 多模型吞吐 | Figure 9 | TF, PyTorch, JAX, MNN | images/s 或 sentences/s |
| LLM 微调更快且质量相当 | LlamaV2-7B Alpaca 指令微调 | Table 5 | PyTorch FT-Full、PyTorch LoRA(r=8) | 迭代延迟、显存、loss、Alpaca-Eval 胜率、MT-Bench |
| 稀疏 BP 质量定性相当 | 6 个手选问答样例 | §5.2 Example 1–6 | 原始 Llama、Full-BP | 定性 |
| 图优化带来 ~1.2× | — | §2.4 文字声称 | — | **未直接验证**（§4 开头说"最后讨论图优化结果"，正文中没有这一节） |

## 4. 实验设计手法
- **基线选择**：四个主流框架（TF/PyTorch/JAX/MNN）在同一硬件上比吞吐；精度基线是 Full BP 与 Bias-only（TinyTL 式）；LLM 部分加入 LoRA。MCU 上 TF-Lite Micro 不支持训练，报告**外推延迟**（§4.3 Microcontrollers）。M1 上 PyTorch 对 BERT 报错（给了 commit c9913cf 与 issue 链接），作为"兼容性差"的证据。
- **规模/广度**：这是本组硬件广度最大的一篇——7 个平台（Jetson Nano GPU、Jetson Orin GPU、STM32F746 MCU、Apple M1 GPU、Snapdragon 8Gen1 CPU 与 DSP、Raspberry Pi 4 CPU）；模型从 MCUNet（0.6M）到 LlamaV2-7B；视觉 7 个、GLUE 7 个、Alpaca/MT-Bench。
- **效率测量**：真机吞吐与实测内存（Table 4 "actual memory usage measured on Jetson AGX Orin"），batch 1/4/16 扫描。没有能耗测量。
- **精度–效率联合呈现**：Table 5 同表放延迟、显存、loss、两种质量分数；Table 2/3 只有精度，效率在 Table 4/Fig 9 分开。
- **消融**：Full vs Sparse BP（贯穿）；但引擎内部的各项图优化（重排、融合、布局变换、后端切换、Winograd）**没有逐项消融**。
- **分析性实验**：Figure 6 说明 CNN 与 Transformer 中"哪些层值得更新"（块内第一个 1×1 卷积；attention + FFN 第一层），来自 MCUNetV3 式贡献分析。
- **泛化**：稀疏方案只在 CIFAR（视觉）与 CoLA（语言）上搜一次，然后迁移到其他数据集（§3.1 Generalization），这是对"搜索开销"的回应。
- **统计**：Table 2/3 给 ±std（Table 3 注明 3 次运行；Table 2 未说明次数）。本组少见。
- **系统贡献**：多后端集成（SNPE、TensorRT、TVM、TinyEngine、Metal）；正文未提开源。
- **局限性**：无专门讨论；§5.2 自承定性样例"精心挑选、不全面"。

## 5. 对抗式审读
- **最薄弱环节：数字前后不一致，且有承诺未兑现的实验。**
  - 对 TF 的加速：摘要 15×（RPi），§1 11×，§4.3 13–21×，§6 11.2×。
  - LLM：摘要"550 tokens/s, 7.9× faster than PyTorch"；Table 5 PyTorch FT-Full 7.7s vs PockEngine Sparse 0.9s ≈ 8.6×，vs PockEngine FT-Full 1.8s ≈ 4.3×（正文 §5.1 也写 4.4×），"7.9×"与"550 tokens/s"在正文实验里找不到来源。
  - 摘要"5.6× memory saving（Jetson AGX Orin）"：Table 4 中 Orin 上最大节省是 BERT bs=16 的 2.5×，Llama 1.4×；5.6× 无出处。
  - Llama 稀疏方案："last 5 blocks (out of 32)"（§4.1）与"(out of 24)"（§5）矛盾；且 LlamaV2 线性层本无 bias，"更新最后 5 块的 bias"不自洽。
  - §4 开头承诺"Finally, we discuss the graph optimization results"，正文没有这一节；§2.4 的"up to 1.2× speedup"因此未被支持。
- **Table 5 解读与表不符**：正文说"sparse-BP 达到 43.7，接近 full-BP 的 44.1"，但表中 43.7 是 PockEngine FT-Full，PockEngine Sparse 是 43.1；MT-Bench 稀疏 5.7 vs 全量 6.1 的下降正文未提。
- **Table 2 MCUNet 行与 MCUNetV3 重复**：Full BP 行与 MCUNetV3 Table 1 的 fp32 SGD-M 行逐项相同（56.7/86.0/56.2/88.8/67.1/79.5/88.7），Sparse BP 行与 MCUNetV3 的 "SGD-M+QAS（更新后两块）" 行逐项相同（55.2/86.9/57.8/89.1/64.4/80.9/89.3）——而后者在原文中是"更新最后两块"而非稀疏方案。本文却给了 ±std，又列在 "Sparse BP" 名下。来源与配置存疑。
- 定性样例是手挑的（作者自承）；"精度不降"在 LLM 上只靠 Alpaca-Eval 单次分数。
- 吞吐比较对 PyTorch 等用的是"开箱即用"版本（Python 宿主在低频 CPU 上慢），加速很大一部分是"去 Python + 调过的 ARM kernel"，这与稀疏 BP 的算法贡献混在一个倍数里。
- 稀疏方案在 CIFAR/CoLA 上搜索，而 CIFAR、CoLA 也出现在评测表中。

## 6. 可迁移的手法
1. 系统论文用"平台 × 模型"的大矩阵（7 平台、CNN 到 7B LLM）证明通用性，并把同一套稀疏方案跨任务复用以回应搜索成本。（Figure 9, §3.1）**直接读到**
2. 当竞争框架在某平台上跑不起来时，把"报错/不支持"本身作为证据，给出版本号与 issue 链接，使其可复查。（§4.3 Apple M-Chip）**直接读到**
3. 精度表报告均值±std，让"<1% 差距"可以与噪声比较。（Table 2/3）**直接读到**
4. 对 LLM 的效率论文，同表报告延迟、显存、训练 loss 与下游质量分数，并加入主流 PEFT（LoRA）作为对照，说明"省内存 ≠ 省时间"（LoRA 仍需反传到第一层）。（Table 5, §5.1）**直接读到**
5. 反面教训：摘要/引言/结论的倍数必须能在实验表中找到唯一出处；承诺的实验节必须存在。（本文多处不一致）**我的归纳**

## 7. 一句话
最有说服力的是 Figure 9 的 7 平台吞吐矩阵和 Table 4 的实测内存——"能在 DSP、MCU、M1 上训练"本身就是证据；但头条数字（15×、7.9×、5.6×）在正文实验里对不上，是全组数字一致性最差的一篇。
