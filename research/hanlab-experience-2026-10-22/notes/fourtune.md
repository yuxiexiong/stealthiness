# FourTune: Towards Fully 4-Bit Efficient Post-Training for Diffusion Models

## 0. 元信息
- ICML 2026（PMLR 306），arXiv 2607.05711v1
- 类型：算法-系统协同（W4A4G4 后训练框架：三分支结构 + 块量化反传 + 融合 kernel）
- 全文已读（正文 + 附录 A、B）。Figure 1/4/8 曲线与柱图数值大多不可读；Figure 9 的延迟数字可读（1541.0 / 1233.8 / 676.7 / 612.4 ms）。

## 1. 核心主张
- 主主张：扩散模型后训练（定制化、RL、蒸馏）可以在权重、激活、梯度全 4-bit（W4A4G4）下进行而质量与 BF16 LoRA 相当，前提是：(1) 用 SVDQuant 式 SVD 分解出一个**冻结的高精度"数值稳定器"低秩分支**吸收异常值，主干残差 4-bit，另加可训 LoRA 分支（三分支）；(2) 用 16×16 块量化使 4-bit 权重可在线转置，用于反传 ∂L/∂X=Q(G)Q(R)^T；(3) LoRA 与 MLP 的 kernel 融合。
- 次要主张：FLUX.1-dev 上显存比 BF16 LoRA 降 2.25×（正文 2.02×，见第 5 节），训练速度比 BF16 LoRA 快 2.27×、比 NF4 QLoRA 快 2.79×，"打破 PEFT 的显存–速度权衡"。

## 2. 动机是怎么立住的
- **痛点**：LoRA 省可训参数但主干权重仍占显存；QLoRA 省显存但在线反量化使计算更慢（Figure 2 三种流水线示意）。"显存–速度权衡"是用文字和示意图建立的，Figure 1(a)(b) 的显存/延迟柱图直接把三者放一起。
- **数值不稳定**：直接 W4A4G4 训练"有时梯度爆炸"——用 Figure 8 的梯度范数曲线（有/无稳定器）证明，这是唯一的稳定器动机证据。
- **反传的硬件约束**：per-output-channel 量化在转置后 scale 落在归约维，无法从内积循环中提出（Sec 3.2 文字论证）→ 块量化。
- **带宽瓶颈**：小 rank 分支的访存开销（沿用 SVDQuant Nunchaku 的论证），Figure 9 用实测验证：4-bit 无融合只有 1.25×，融合后 2.52×。
- 因果链：每个组件对应一个具体障碍，但"稳定器为何必须冻结且来自 SVD"未与替代方案（如直接用可训 LoRA 初始化为 SVD 分量、或只做平滑）比较。

## 3. 主张 → 证据映射表
| 主张 | 证明实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 定制化质量相当 | FLUX.1-dev & Qwen-Image，3 类任务 | Table 1, Fig. 5, Fig. 4 | BF16 LoRA、FP8 LoRA、NF4 QLoRA | 相似度（AntelopeV2/CLIP/DINOv3）、PyIQA、多样性、BLIP |
| 按墙钟时间更快收敛 | 验证曲线 vs 时间与步数 | Fig. 4 | BF16 LoRA、QLoRA | 验证指标（不可读） |
| RL 质量相当 | SRPO on FLUX.1-dev | Table 3, Fig. 6 | BF16 全参、BF16 LoRA、QLoRA | Aes、PickScore、IR、SGP-HPS |
| 蒸馏质量相当 | π-Flow 4-NFE 学生 | Table 2, Fig. 7 | BF16 LoRA、QLoRA、SparseLoRA | FID（对教师）、CLIP、HPSv2.1 |
| 显存/速度 | DiT 单步、端到端 customization | Sec 4.2 文本、Fig. 1 | BF16 LoRA、QLoRA | GB、ms/s |
| 精度配置无损 | W4A16G16/W4A4G16/W4A16G4/W4A4G4 | Table 4 | 互比 | Aes 等（500 prompts 子集） |
| 稳定器防梯度爆炸 | 梯度范数曲线 | Fig. 8 | 无稳定器 | 梯度范数 |
| 块量化不掉质量 | **推理重建**（MJHQ 5K） | Table 5 | 1×16 group-wise | LPIPS、PSNR |
| kernel 融合贡献 | 逐步累加 | Fig. 9 | BF16、无融合 4-bit | ms |
| 跨架构/格式/硬件 | SDXL；INT4 on RTX 4090 | App. Table 6, 7 | BF16 LoRA | 同 Table 1 |

## 4. 实验设计手法
- **基线选择**：BF16 LoRA（质量上界兼速度基准）、NF4 QLoRA（省显存标准做法）、FP8 LoRA（仅定制化）、BF16 全参（仅 RL）、SparseLoRA（仅蒸馏，一种加速 PEFT）。**未与其他 4-bit 训练方案（如 NVFP4 预训练 recipe、Quartet 一类）比较**。
- **规模/广度**：FLUX.1-dev 12B、Qwen-Image 20B、SDXL（附录）；三类后训练任务各用社区标准算法（SRPO、π-Flow）；硬件 RTX 5090、RTX Pro 6000（Blackwell），INT4 变体 RTX 4090。
- **效率测量**：真实 GPU；DiT 单步与端到端训练步；两种 Blackwell 卡；"按墙钟时间的收敛曲线"（Fig. 4）把速度与质量合在一张图里——这是比单纯 step latency 更有力的度量。
- **精度–效率联合**：Fig. 4（质量 vs 墙钟时间）；Fig. 1 显存与延迟并列。
- **消融**：精度配置四组（Table 4）、稳定器（Fig. 8，仅梯度范数）、量化粒度（Table 5，但用推理重建代替训练结果）、kernel 融合 roadmap（Fig. 9）。
- **分析性实验**：梯度范数监控。
- **泛化**：三类任务、两个 DiT、一个 UNet（附录）、NVFP4/INT4 两种格式。
- **统计**：无方差。定制化每类 5 个概念、每概念 5 对图文 + 5 组测试 prompt（App. A.1），样本量很小。
- **系统贡献**：融合 kernel；提到"our release"，未给仓库链接。
- **局限性**：无技术局限章节，只有 Impact Statement（滥用风险）。

## 5. 对抗式审读
- **最薄弱：稳定器只用梯度范数证明**。Fig. 8 显示无稳定器时梯度爆炸，但没有"无稳定器"的最终质量数字，也没有与"把 SVD 分量并入可训 LoRA"或"只用平滑"的对比；且实验写"occasionally triggers"，单次运行的曲线不能说明频率。
- **块量化的验证用错了测量对象**：Table 5 比较的是推理重建（PTQ 质量）LPIPS 0.227 vs 0.203、PSNR 20.4 vs 21.5——块量化其实更差，作者称"comparable"；而它要证明的是"训练（反传）时用块量化不伤最终训练效果"，这个没有直接测。
- **数字前后不一**：Fig. 1 与摘要称显存降 2.25×，正文 Sec 4.2 写 14.51 vs 29.35 GB = 2.02×。
- **"质量相当"的指标混杂**：RL（Table 3）中 FourTune 的 ImageReward 1.0152 低于 BF16 LoRA 1.0735 和 QLoRA 1.0931，甚至低于未训练的基模 1.1189；PickScore 也低于 QLoRA。蒸馏 COCO FID 6.70 vs BF16 5.90（差 0.8）。Table 4 中 W4A4G4 的 IR 是四种配置里最低（1.1712 vs 1.2231）。作者每处都以"comparable"概括。
- **定制化指标噪声大**：Qwen-Image Subject 的 Image Quality BF16 16.92 → Ours 26.92（+59%），Style 的 QLoRA 33.72 远高于 BF16 27.34——这种幅度的跳动更像是小样本（5 概念）下 PyIQA 的噪声，而非方法效应；说明表中其他小差异也难以解读。
- **SparseLoRA 基线可能配置不当**：FID 202.51 近乎崩溃，作为对照的信息量低。
- Table 4 精度消融只在 500 prompts 子集上训练，数字与 Table 3 不在同一尺度。

## 6. 可迁移的手法
1. 用"按墙钟时间的收敛曲线 + 按步数的收敛曲线"双图，同时证明"每步质量不降"和"单位时间更快"（Fig. 4）。**直接读到**
2. 系统加速用 roadmap 柱图逐项累加，并单独给出"仅量化、无融合"的中间点，证明量化本身收益有限、融合才是关键（Fig. 9：1.25× → 1.82× → 2.52×）。**直接读到**
3. 精度配置的全组合消融（W/A/G 各 4 或 16 bit），标明哪一个配置才真正能用 4-bit 加速，用来论证"最激进的配置不比保守配置差"（Table 4）。**直接读到**
4. 一篇后训练方法论文用三种性质不同的后训练任务（定制化、RL、蒸馏），每种沿用社区标准算法和评测协议，用"任务覆盖"建立通用性（Sec 4.1）。**直接读到**
5. 把自家推理期方法（SVDQuant 的低秩分支）在训练期重新解释为"冻结稳定器"，复用已有组件开新问题。**我的归纳**

## 7. 一句话
把 SVDQuant 的低秩分支重新定位为训练期的"冻结数值稳定器"，再用 kernel 融合 roadmap（Fig. 9）和按墙钟时间的收敛曲线（Fig. 4）把"全 4-bit 训练能更快且不掉质量"讲成一个系统故事；但稳定器与块量化的必要性证据较弱，多项质量指标实际低于基线而被统称为"comparable"。
