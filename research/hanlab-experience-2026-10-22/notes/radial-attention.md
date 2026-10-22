# Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation

## 0. 元信息
- NeurIPS 2025（arXiv 2506.19852v2，camera-ready）；Li*, Li*, Cai, Xi, Yang, Lin, Zhang, Yang, Hu, Peng, Agrawala, Stoica, Keutzer, Han
- 类型：算法（静态 O(n log n) 稀疏掩码 + LoRA 长度扩展）+ 理论界（复杂度、误差）
- 全文读完（正文 + 附录 A–F，含 Fig.A–F 逐 prompt 结果）。Fig.2/4/8 曲线数值不可读，只取正文给出的倍数与 R²。

## 1. 核心主张
- 主：视频 DiT 的 softmax 后注意力随时空距离近似指数衰减（"Spatiotemporal Energy Decay"）；据此设计的静态掩码（时间每远一倍密度减半、空间对角宽度随之收窄）把复杂度降为 O(n log n)，默认长度下 1.8–1.9× 端到端加速，质量与 SVG 相当、优于 STA/PA。
- 次 1：同一掩码可用于训练，使 LoRA 微调把视频延长 2–4×，训练成本降至 1/4.4，推理快 3.7×，质量与 LoRA 微调的 dense 相当或更好。
- 次 2：给出复杂度上界（Eq.5–6）与 ℓ1 误差上界（Eq.7），误差随衰减率指数下降。
- 次 3：长度扩展 LoRA 可与现有风格 LoRA 合并。

## 2. 动机是怎么立住的
- 痛点：Fig.2 计算量/延迟/训练 GPU 时随帧数的增长（dense O(n²) vs radial），509 帧时注意力计算降 9×；HunyuanVideo 5 秒 720p 约 110K token（Sec.3）。
- 现有方法缺陷：SVG 在线 profiling 分类头，不能用于训练、对新分布（长视频）会误分类（Sec.1、Fig.3）；线性注意力需大改结构（以 SANA 为证据，Table 2 中 Vision Reward 为负）。
- 关键观察：Fig.4(a) 两个典型注意力图（spatial/temporal 头）；Fig.4(b) 平均分数随时间距离、空间距离的衰减曲线；Fig.8(b)/Sec.5.3 用 y=exp(−ax+b) 回归，R²>0.985。观察被**参数化**（Eq.3，α/β 两个衰减率），并统一解释了 SVG 的两类头（高 α 低 β / 低 α 高 β）。
- 因果链：观察 → 参数化假设 → 掩码设计"把能量衰减翻译成计算密度衰减" → 在该假设下推出误差界。链条最完整；但误差界依赖的假设 A1 本身只由平均曲线拟合支撑。

## 3. 主张 → 证据映射表
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 存在指数时空衰减 | 注意力分数 vs 距离 + 回归 | Fig.4, Fig.8(b) | — | R²>0.985 |
| O(n log n) | 推导 + 实测 PFLOPs 曲线 | Eq.5–6, App.A.1, Fig.2 | Dense | PFLOPs |
| 误差有界 | 推导；实测注意力 MSE | Eq.7, App.A.2；Sec.5.3 | SVG 4.4e-3, STA 1.5e-2 vs 3.9e-3 | MSE |
| 默认长度免训练加速 | Hunyuan 117 帧、Wan2.1 69 帧 | Table 1, Fig.6, Fig.A/B | STA(FA3)/PA/SVG 同 PFLOPs | PSNR/SSIM/LPIPS/VR/延迟 |
| 长视频 LoRA 扩展 | 2×/4×，三个模型 | Table 2, Fig.7, Fig.C–E | Original/RIFLEx/Spatial/Temporal/LongLoRA/PA/SANA/Full(LoRA) | VR、VBench 三维、训练/推理时间 |
| LoRA+Radial ≥ 全参微调 | 帧数扫描 | Fig.8(a) | Dense+Full FT/Dense+LoRA/Radial+Full FT | VR |
| 风格 LoRA 兼容 | 定性 | Fig.F, App.C.3 | — | 定性（作者承认有风格偏移） |
| 掩码形状优于其他 n log n | 与 Harmonic Series 比较 | Table A | HS | PSNR 等 |
| warmup 步数/dense 层数选择 | 消融（同计算量） | Table B/C/D | — | PSNR / VR |

## 4. 实验设计手法
- 基线：覆盖三类——同为视频稀疏（SVG、STA）、同复杂度的 LLM 方法（PowerAttention、LongLoRA）、线性注意力（SANA）、免训练长度外推（RIFLEx），以及"LoRA 微调的 dense"这一**上界式**基线。所有稀疏基线"相似稀疏率"（Sec.5.2）、Table 1"相同 PFLOPs"。
- 公平性披露：STA 用 FA3 更快，作者明说自己用 FA2、升级正交（Sec.5.2）；所有方法用相同的前 12 步 dense warmup（沿用 SVG）。
- 规模：Mochi 1（10B）、HunyuanVideo（13B）、Wan2.1-14B；Wan 只做 2×（成本原因）。
- 效率：单 H100 端到端延迟（默认长度），长视频推理 8×H100（Hunyuan/Mochi）；训练 GPU 时实测；Table 1 同时给 PFLOPs 与实测延迟，并指出实测加速与理论节省一致（1.9× vs 1.8× PFLOPs）。
- 精度–效率联合：Table 1 同计算预算比质量；Table 2 同稀疏率比质量，同时列训练/推理加速。
- 消融：LoRA vs 全参（Fig.8a）、warmup 步数（Table B/C）、dense 层数（Table D）、掩码形状 vs HS（Table A）；附录消融"调整稀疏度使总计算匹配"。
- 分析性：衰减回归、注意力 MSE 对比、理论界。
- 统计：无 seed/方差；Vision Reward 为平均值但无置信区间。
- 系统：开源。
- 局限：Sec.6 承认指数衰减假设是简化；对分辨率仍为二次。附录 C.3 承认风格偏移。

## 5. 对抗式审读
- 最薄弱：质量指标。默认长度用 PSNR/SSIM/LPIPS 衡量"与 dense 输出的相似度"，不是质量；长视频用 Vision Reward（模型打分）+ VBench 三个维度，差距普遍在 0.00x 量级（如 Hunyuan 4×：0.134 vs Full 0.133 vs LongLoRA 0.130 vs PA 0.128），**无方差、无人类评测**，"优于"难以成立，最多是"相当"。
- Table 2 的 VBench 维度是作者挑选的"原始模型明显退化"的三项（Sec.5.1 自述），存在选择性呈现。
- Table 1 Hunyuan 上 SVG 与本法几乎相同（SSIM 0.895 vs 0.886，SVG 更高），"matches SVG"是准确表述；对 SVG 的真正优势在于"可训练"，而这一点在长视频实验中没有 SVG 作为基线（Table 2 只有 Spatial/Temporal 单独掩码），**缺少"SVG+LoRA"的直接比较**。
- 误差界的常数 C_rel 未估计，界是否紧未知；实测只给一个 MSE 数。
- 附录 Table C 标题"4× longer"但模型栏写"HunyuanVideo (117 frames)"，存在表述不一致。
- 附录 Fig.C–E 的逐 prompt VR 显示方差很大（同一方法不同 prompt 0.055 vs 0.161），进一步说明平均值差异不可靠。

## 6. 可迁移的手法
1. 把经验观察拟合成带参数的闭式模型（指数衰减，R²>0.985），再用它推导复杂度与误差界，使方法"有理论"。（Eq.3, Fig.8b, App.A；直接读到）
2. 用一个统一机制同时"解释"竞品的多个经验分支（SVG 的 spatial/temporal 头 = α、β 两端）。（Sec.4.1–4.2；直接读到）
3. 基线按"同复杂度不同结构（PA/LongLoRA）、同领域不同机制（STA/SVG）、不同复杂度类（SANA）"分层，证明收益来自"结构先验"而非"复杂度类"。（Table 2；我的归纳）
4. 理论节省（PFLOPs）与实测延迟并列，并说明二者吻合。（Table 1；直接读到）
5. 主动披露基线在实现上的优势（STA 用 FA3）并声明正交。（Sec.5.2；直接读到）
6. 消融时调整稀疏度保持总计算相同，避免"多花计算换质量"。（App.D；直接读到）

## 7. 一句话
把"注意力随时空距离指数衰减"拟合成 R²>0.985 的参数模型，再从同一模型推出 O(n log n) 复杂度和误差界，使一个手工掩码看起来是被"推导"出来的。
