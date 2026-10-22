# AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

## 0. 元信息
- MLSys 2024（Best Paper），arXiv 2306.00978v6
- 类型：算法-系统协同（AWQ 算法 + TinyChat 推理框架）
- 全文已读完（正文 + 参考文献；该版本无附录）。表格由 pdftotext 生成，Figure 9/10 柱状图数字错位，只引用正文明确复述过的数字。

## 1. 核心主张
- 主主张：LLM 权重并非同等重要，约 0.1%–1% 的"显著"权重通道应由**激活幅度**（而非权重幅度）来识别；对这些通道做等价的逐通道放缩（不需混合精度、不需反传/重建）即可显著降低 W4/W3 量化误差。
- 次要主张 1：因为只用每通道平均激活幅度，AWQ 对校准集依赖小、不过拟合，从而能泛化到指令微调模型、多模态模型、代码/数学。
- 次要主张 2：TinyChat 把 W4A16 的理论内存节省变成实测 3× 以上加速，并覆盖桌面/笔记本/移动 GPU 与 CPU（Raspberry Pi）。

## 2. 动机是怎么立住的
- **关键观察的量化（Table 1）**：在 OPT-1.3B/6.7B/13B、INT3-g128 下，保留 0.1%/1%/3% 通道为 FP16，按三种规则选：按激活、按权重 L2、随机。按激活选 1% 时 OPT-6.7B PPL 23.54→11.39，而按权重/随机基本等于 RTN（22.37 / 24.23）。这是一个**三臂对照**：随机臂充当零假设，权重臂充当"教科书方法"，从而把"激活感知"从"保留任意通道"中剥离出来。
- **从观察到方法的桥梁（Eq. 2–3 + Table 2）**：推导出放缩 s 后显著通道相对误差约为 Δ'/Δ · 1/s；Table 2 实测验证推导的两个假设——s<2 时 Δ 改变比例 <5%（2.8%/4.4%/8.2%），并用 s=4 时 21.2% 的组 Δ 变大解释"为什么 s 不能无限大"（PPL 在 s=2 最好 11.92，s=4 回升到 12.36）。这是"推导→测推导中的近似是否成立→测推导预言的拐点"的闭环。
- **系统侧痛点（Figure 3）**：Llama-2-7B 在 RTX 4090 上 profiling：生成 20 token 310 ms vs 上下文 200 token 10 ms；roofline 显示生成阶段算术强度≈1，memory-bound；权重访存比激活大数量级。由此论证"为什么做 weight-only 而不是 W8A8"。
- 因果链：Table 1 → Fig.2(c)/Table 2/Table 3（放缩能等效于 1% FP16：OPT-6.7B 都是 11.39）→ Eq.5 搜索。闭合较好。

## 3. 主张 → 证据映射表
| 主张 | 证明实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 显著权重应按激活识别 | 保留 x% FP16 三种选法 | Table 1 | 按 W、随机、RTN | Wiki PPL |
| 放缩可替代混合精度 | s 扫描 + 统计 Δ | Table 2, Table 3 | 1% FP16、s=2 固定、RTN | PPL、Δ 变化比例 |
| AWQ 优于 GPTQ | LLaMA 7–65B、Llama-2 7–70B，INT3/INT4 | Table 4 | RTN、GPTQ、GPTQ-R | PPL |
| 跨架构（GQA/MoE） | Mistral-7B、Mixtral-8x7B | Table 5 | **仅 FP16，无其他方法** | PPL |
| 泛化到指令模型 | Vicuna-7B/13B GPT-4 评判，两种顺序共 160 次 | Figure 5 | RTN、GPTQ | 胜/平/负 |
| 泛化到多模态 | OpenFlamingo COCO 0–32 shot；VILA 11 基准；LLaVA 定性 | Table 6, 7, Fig. 6/7 | RTN、GPTQ（VILA 仅 FP16） | CIDEr、准确率 |
| 泛化到代码/数学 | MBPP、GSM8K | Table 8 | RTN、GPTQ | pass@k、acc |
| 与 GPTQ 正交 | INT2-g64 AWQ+GPTQ | Table 9 | GPTQ、RTN | PPL |
| 校准数据高效 | 校准序列数扫描 8–256 | Fig. 8(a) | GPTQ | PPL |
| 对校准分布鲁棒 | PubMed/Enron 2×2 交叉 | Fig. 8(b) | GPTQ | PPL 增量 |
| TinyChat 加速 | 4090/Orin/4070，bs=1 | Fig. 9, Table 10 | HF FP16、自家 FP16 | tokens/s |
| 优于现有边缘系统 | Orin、Raspberry Pi | Fig. 10 | AutoGPTQ、llama.cpp、exllama | tokens/s |
| SIMD-aware packing 加速 | 单 kernel 延迟 | Fig. 4 右 | 原始打包 | μs（"up to 1.2×"） |
| "不过拟合校准集"导致泛化好 | **未直接验证**因果：Fig.8 只证校准鲁棒性，与指令/多模态结果之间是推断 | — | — | — |

## 4. 实验设计手法
- **基线选择**：朴素基线 RTN（作者明确说 g128 下 RTN"相当强"），SOTA 为 GPTQ 及其 reorder 版本 GPTQ-R；明确排除 ZeroQuant/AdaRound/BRECQ，理由是需反传且不如 GPTQ（Sec 5.1）。未与同期 SmoothQuant 类或 SpQR/OWQ 等比较。系统侧对 HF FP16（弱基线），但也给出自家 FP16 优化后的数字（Llama-2-7B 52→62 tok/s，再叠 3.1×），把"系统优化"与"量化"拆开。
- **规模/广度**：OPT 1.3B–30B、LLaMA 7B–65B、Llama-2 7B–70B、Mistral、Mixtral、Vicuna、CodeLlama、OpenFlamingo-9B、LLaVA-13B、VILA-7B/13B；硬件 RTX 4090、Jetson Orin、RTX 4070 laptop、A100、Raspberry Pi 4。
- **效率测量**：真实硬件端到端 tokens/s；协议交代清楚（bs=1，prompt 4 token，生成 200 token，取中位数，沿用 exllama 协议，Sec 5.4）。无 batch/序列长度扫描（定位 edge，bs=1 合理但覆盖窄）。
- **精度–效率联合**：没有 Pareto 曲线；精度和速度分开报告。
- **消融**：Table 1（选通道规则）、Table 2（s 扫描）、Table 3（1%FP16 vs s=2 vs AWQ 搜索）构成逐步路线；α 网格 20（未报敏感性曲线）；校准集大小/分布 Fig. 8。权重裁剪（clipping）的单独贡献**未消融**。
- **分析性实验**：Table 2 是把推导中近似假设测出来的经典做法；Fig. 3 roofline。
- **泛化/鲁棒**：跨模型族、跨模态、跨任务、跨硬件都覆盖；校准分布交叉是本文最"干净"的鲁棒性实验。
- **统计**：无方差、无多 seed。GPT-4 评测做了两种顺序以消除位置偏差（Fig. 5），是唯一的偏差控制。
- **系统贡献**：开源 llm-awq；声称被 HF Transformers、TensorRT-LLM、vLLM、LMDeploy 等采用；有 TinyChat 实体机 demo（Fig. 1）。
- **局限性**：仅在 Sec 3.1 写了"混合精度难以实现"作为方法动机的局限；无整体局限性章节。

## 5. 对抗式审读
- **最薄弱环节**：精度主表几乎全靠 WikiText PPL（Table 4），且对 GPTQ 的优势在 INT4 时很小（Llama-2-70B 3.41 vs 3.42/3.43）；无方差，难以判断 0.01–0.05 的差是否显著。
- **"不过拟合"的论证是间接的**：Fig. 8(b) 仅 OPT-6.7B、两个子集；指令/多模态上的优势被归因于"不过拟合"，但未排除其他解释（例如 GPTQ 实现/超参）。
- **不公平/挑选**：Mistral/Mixtral（Table 5）和 VILA（Table 7）只给 AWQ vs FP16，没有 GPTQ 对比——这些"跨架构"证据只能说明 AWQ 损失小，不能说明优于他人。系统主对比是 HF FP16 这种未优化实现；与 llama.cpp/exllama 比较时作者承认后者支持模型少。
- **超出支持的结论**："first time, multi-modal LMs" 是首创性声明，非实验结论。Table 9 INT2 只证"与 GPTQ 叠加有益"，并非 AWQ 单独可用于 INT2。
- 文本中"ablation study on OPT models … in Table 5"实际指 Table 3（交叉引用错位，可能是 v6 修订遗留）。

## 6. 可迁移的手法
1. 在"某类元素更重要"的观察下，用**随机选择 + 教科书准则 + 新准则**三臂对照，保留同比例元素，证明新准则而非"保留了元素"本身起作用（Table 1）。**直接读到**
2. 在方法由近似推导而来时，把推导中的近似（Δ'≈Δ）单独测出来，并用推导预言的拐点（s 过大反而变差）解释超参最优点（Table 2）。**直接读到**
3. 在声称"更不依赖校准数据"时，做校准集 × 评测集的**交叉矩阵**，报告对角与非对角的差（Fig. 8b）。**直接读到**
4. 在系统加速论证中，先把"更好的 FP16 实现"带来的收益单列，再报告量化增量，避免把工程优化记在量化名下（Sec 5.4，52→62→3.1×）。**直接读到**
5. 用 roofline + 访存分解证明"选这个量化设置"的合理性，而非仅凭直觉（Fig. 3）。**直接读到**
6. 用 LLM-as-judge 时双向顺序评测抵消位置偏差（Fig. 5）。**直接读到**

## 7. 一句话
Table 1 的"按激活 / 按权重 / 随机"三臂对照，把一个模糊直觉变成了干净的因果证据，后续的放缩推导和系统实现都挂在这根钉子上。
