# DeltaQuant: 4-Bit Video Diffusion Models with Spatiotemporal Delta Smoothing

## 0. 元信息
- CVPR（Open Access 版本，页码 43578–43588；年份文本未写明，按引用推断为 2026）
- 类型：算法-系统协同（W4A4 激活分解算法 + 基于 Nunchaku 的融合 kernel）
- 全文已读（正文 + 参考文献）。该 CVF 版本**无附录/补充材料**。Figure 8、9 为柱状/折线图，数字错位；Fig.9 的 PSNR 值可读（cube 16/32/64/全局 = 23.2/22.8/22.3/21.6；core 精度 16/8/4 bit = 22.8/23.2/21.9；均匀 vs 时步感知 = 22.7/23.2），Fig.8 延迟柱的具体数字对不上，只引正文复述值。

## 1. 核心主张
- 主主张：视频扩散模型的激活异常值随去噪时步剧烈变化，静态平滑（SmoothQuant/SVDQuant 的激活侧）会在其他时步反而恶化；但视频激活具有"时空相似性"（类比 H.264 帧间差分），把激活切成局部 t×h×w 立方体，用立方体均值作 FP8 的 core token、只把差值（delta）量化到 4 bit，可在线、低开销地吸收动态异常值。
- 次要主张：与 SVDQuant 权重低秩分解组合；Wan2.2 模型体积降 2.9×、显存降 2.3×；与稀疏/低比特注意力和少步蒸馏 LoRA 叠加后总加速 111.8×。

## 2. 动机是怎么立住的
- **瓶颈迁移（Figure 3）**：Wan2.2 72K token，BF16 下注意力占 80.9% 计算；加 Radial+Sage 后注意力降 3.51×，线性层占比升至 65%。论证"注意力优化之后，下一步必须是线性层量化"。这是把同组前作（Radial Attention、Sparse VideoGen）作为前提而非对手。
- **静态平滑失效（Figure 4）**：timestep 0 与 20 的激活分布，异常值通道/幅度不同；按 step 0 校准的平滑在 step 20 加剧异常值。这是针对 SVDQuant 的定点反驳。
- **关键观察可视化（Figure 5）**：Wan2.2 T2V 中 4 帧激活呈周期性模式，减去时间均值后幅度和异常值大降；2×2 空间邻域同样。
- **排除替代方案**：在线 SVD"代价过高"（仅论证，无测量）。
- 因果链：瓶颈迁移 → 线性层 W4A4 → 动态异常值使静态方法失效 → 时空相似性 → core/delta 分解。闭合，但"动态异常值导致 SVDQuant 掉点"的因果没有被单独测出（只有一张分布图）。

## 3. 主张 → 证据映射表
| 主张 | 证明实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 线性层成新瓶颈 | 计算量分解 | Fig. 3 | BF16 / +Radial&Sage / +DeltaQuant | PFLOPs |
| 异常值随时步变化、静态平滑失效 | 分布图 | Fig. 4 | timestep 0 vs 20 | 幅度 |
| 时空相似性存在 | 分布图 | Fig. 5 | 原始 vs 均值 vs delta | 幅度 |
| 画质优于 W4A4 基线 | 3 个模型设置 | Table 1, Fig. 7 | RTN、SmoothQuant、SVDQuant、S2Q-VDiT、Q-VDiT（W4A6）、GGUF4（W4A16） | LPIPS、PSNR、SSIM、Vision Reward、VBench S.C./A.Q./I.Q. |
| 显存/延迟收益 | Wan2.2 on RTX 5090 | Fig. 8 | BF16、GGUF4（均带/不带 Radial&Sage） | GiB、单步秒 |
| 与其他加速技术正交 | 逐步叠加 | Table 2 | LightX2V、Radial&Sage | Vision Reward、延迟 |
| 局部性重要 | cube 尺寸扫描 | Fig. 9a | 到全局均值 | PSNR |
| 比单纯加大 rank 更划算 | LPIPS–延迟曲线 | Fig. 9b | SVDQuant rank 128/160/192 | LPIPS vs 单步延迟 |
| core 用 FP8 | 16/8/4 bit | Fig. 9c | — | PSNR |
| 时步感知 cube 分配 | 均匀 vs 自适应 | Fig. 9d | — | PSNR（+0.5） |
| core 开销可忽略（"3.2%"） | **未直接验证**：数字似为理论计算，无实测 kernel 拆分 | Sec 4.3 | — | — |

## 4. 实验设计手法
- **基线选择**：朴素 RTN、SmoothQuant、自家前作 SVDQuant（被定位为 SOTA W4A4）、视频专用 S2Q-VDiT、Q-VDiT、社区 W4A16 GGUF4。声称"matched calibration time"。S2Q-VDiT/Q-VDiT 只在 Wan2.2 I2V 上跑。
- **规模/广度**：Wan2.2 I2V/T2V（27B 双 transformer）、LTX-Video（13B）；单一 GPU RTX 5090；720p/480p 两种分辨率。
- **效率测量**：真实 GPU 单步与端到端延迟；Table 2 叠加式加速表。无 kernel 级开销测量。
- **精度–效率联合**：Fig. 9b 的 LPIPS–latency 曲线，在相同计算预算下与 SVDQuant 比（"Pareto 下移"）——这是本文最有力的对比。
- **消融**：cube 尺寸（含"全局均值"这一极端反事实，联系 SageAttention 的全局平滑）、rank、core 精度、时步自适应分配。**缺**"只用 delta 不用 SVD 低秩"的单独消融，因而 delta 分解与低秩分支各自贡献无法完全拆开（Table 1 只能读出 SVDQuant→Ours 的整体增量）。
- **分析性实验**：分布可视化（Fig. 4/5）；视频编解码类比是主要叙事工具。
- **泛化**：I2V/T2V、两个模型族、与蒸馏/稀疏注意力组合。
- **统计**：无方差、无多 seed；**评测 prompt 数量与来源未在正文写明**。
- **系统贡献**：基于 Nunchaku 的 CUDA kernel（core token 生成融合进量化 kernel、W4A8 core GEMM、partial sum 融合进 W4A4 GEMM）；未明确说开源。
- **局限性**：仅"Broader impact"一句（滥用风险）；无技术局限讨论。

## 5. 对抗式审读
- **最薄弱：Table 1 并不支持"consistently surpasses other W4A4 baselines in both similarity and quality"**。Wan2.2 T2V 上 Vision Reward：RTN 0.093 > Ours 0.083 = SVDQuant 0.083；SmoothQuant 的 S.C. 0.970 > Ours 0.966。Wan2.2 I2V 上 SVDQuant VR 0.146 > Ours 0.143。LTX-Video 上 GGUF4 VR 0.141 > Ours 0.135。相似度指标（LPIPS/PSNR）确实普遍最好，但质量指标的优势不成立，作者的文字结论超出了表格。
- **噪声尺度提示**：Fig. 9c 中 core 用 FP8（23.2）高于 BF16（22.8），作者称"FP8 even higher"，更合理的解释是这是测量噪声，提示 ±0.4 PSNR 量级的差异未必显著——而 Fig. 9d 的"+0.5 PSNR"正好落在这个量级。
- **111.8× 加速的归因**：Table 2 中 24.5× 来自 4 步 LightX2V 蒸馏，1.5× 来自注意力，DeltaQuant 贡献 3.0×；标题级数字主要不是量化带来的。BF16 在 5090 上 OOM（61.3G），其延迟必然含 offloading，使"vs BF16"的倍数偏大。
- **基线公平性**："matched calibration time" 对 Q-VDiT（含蒸馏）不利，Q-VDiT 结果（VR −0.085）近乎崩溃，可能是没调好；S2Q-VDiT/Q-VDiT 只跑了一个设置。
- 超参（tile 16/64、前 25–30% 时步、rank 128/64、cross-attn KV 投影 6 bit）组合较多，是否在评测集上选出未交代。
- "core 开销 3.2%"没有实测。

## 6. 可迁移的手法
1. 当自己同组前作已解决 A 瓶颈时，用"瓶颈迁移"计算分解图把 A 的解决方案变成新工作的动机（Fig. 3）。**直接读到**
2. 针对 SOTA 的静态假设，用"在 t1 校准、在 t2 评估"的反例分布图直接击中（Fig. 4）。**直接读到**
3. 从成熟学科（视频编码的帧间差分）借来结构先验，作为方法的叙事锚（Sec 4.2）。**直接读到**
4. 在同一计算预算轴上画 Pareto 曲线与"给前作更多资源（加大 rank）"比，证明增益不是靠多花算力（Fig. 9b）。**直接读到**
5. 消融中纳入"结构先验退化为全局"的极端点（cube=全部 token），证明局部性本身是收益来源（Fig. 9a）。**直接读到**
6. 正交性证明用逐行累加表（+蒸馏 / +注意力 / +量化），每行同报质量指标（Table 2）。**直接读到**

## 7. 一句话
把视频编解码的"只编码差值"嫁接到激活量化上，用 Fig. 4（静态平滑跨时步失效）+ Fig. 5（差值分布收窄）两张图建立动机，再用同预算 LPIPS–延迟曲线（Fig. 9b）击败加大 rank 的 SVDQuant；但主表在质量指标上并不支持"全面领先"的文字结论。
