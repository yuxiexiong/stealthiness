# COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training

## 0. 元信息
- arXiv 2410.19313v3（2025-02）；文本未印会议名（可能是 ICLR 2025，未核实）
- 类型：算法-系统协同（FP8 训练框架：优化器状态 + 激活量化，Triton/CUDA kernel）
- 全文已读（正文 + 附录 A–K）。Figure 2/3/4/5/6/7/9 为图，文本中仅有图注；Figure 1(c) 显存分解柱状图可部分读到数字（BF16 65.4、TE 62.1、FP8-LM 51.2、COAT 42.2 GB 等，与 Appendix E 对应），但柱与数字对应关系部分错位。附录 B 只有标题和定性示例。

## 1. 核心主张
- 主主张：现有 FP8 训练（TransformerEngine、FP8-LM）只量化线性层计算，优化器二阶矩和激活仍是高精度，显存没吃满；COAT 用 (1) **Dynamic Range Expansion**（f(x)=sign(x)|x|^k，按组在线算 k 使组的动态范围对齐 E4M3）把一/二阶矩都压到 FP8，(2) **Mixed-Granularity Activation Quantization**（非线性层 1×16 per-group、线性层 per-tensor + Group Scaling）把激活压到 FP8，实现近乎无损。
- 次要主张：端到端显存降 1.54×、速度 1.43×（vs BF16），与 TE 持平或更快；7B 单卡、30B 8 卡全参训练；分布式下 batch 翻倍。

## 2. 动机是怎么立住的
- **显存分解（Figure 1c，App. E）**：Llama-2-13B 8×H100 FSDP，逐类（优化器/激活/权重/梯度）对比 BF16、TE、FP8-LM、COAT，指出 FSDP 分片后激活成为主导、FP8-LM 不压激活与二阶矩。
- **激活显存的算子级账本（Table 2，App. D）**：以 U=B×L×H×2 bytes 为单位，RMSNorm 4U、Act Func 8U、RoPE 2U、FlashAttn 3U、Linear 5.66U；非线性层约占 50%，线性层 <25%——由此论证"只量化线性层输入"不够。App. D 解释 5.66U 的来源及为何 attention projection 不另存 FP8（与 FlashAttention 共享张量）。理论比 1.69×，实测 1.65×（Table 7），**理论账与实测对上**。
- **关键观察（Figure 2a/2c）**：优化器状态按 1×128 分组后，一阶矩组内动态范围通常 <1e4，二阶矩 <1e1，远小于 E4M3 的 ~2×10^5，FP8 码点被浪费；最优 k 分布：二阶矩 5–15、一阶矩 1–3。
- **从观察到方法**：定义动态范围 R_X；R_{f(X)}=R_X^k，令 R_X^k=R_E4M3 解出 k；App. F 在"奇函数、单调、连续可导、尺度不变"假设下证明幂函数是唯一满足的扩展函数。
- **非线性层粒度观察（Figure 4a）**：在共享 scale 元素数相同时，跨 token 轴分组误差显著更大 → 选 1×G 而非 Jetfire 的 B×B。
- 因果链闭合较好：显存账本 → 选择量化对象 → 量化误差来源（动态范围未用满 / 跨 token 分组）→ 对应修复。

## 3. 主张 → 证据映射表
| 主张 | 证明实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 优化器状态动态范围未用满 | 分布可视化 | Fig. 2, Fig. 3, Fig. 9 | 扩展前后 | 动态范围 / 直方图 |
| 扩展降低量化误差 | m/√v 的 MSE，4×4 格式组合 | Table 1 | E4M3/E5M2 ± Expand | MSE（20.10→12.31，1.63×） |
| 扩展与 DE8 兼容 | 同上加 DE8 | Table 9 | 8-bit Adam 的 DE 格式 | MSE |
| 幂函数最优 | 证明 | App. F | — | — |
| 激活显存降 1.65× | 单层测量 | Table 2, Table 7 | BF16、TE | MB |
| 单层加速 | hidden 2048/4096 × seq 2048/4096 | Table 7 | BF16、TE | ms（最高 1.57×） |
| 预训练无损 | OLMo-1B 300B tok、OLMo-7B 250B tok | Table 3/4, Fig. 5/6 | BF16 | 训练 loss、PPL、4 任务 acc |
| 微调无损 | Llama-2-7B MAmmoTH | Table 5 | BF16、TE | 4 个数学基准 |
| VLM 无损 | VILA1.5-7B stage-3 SFT | Table 6, Fig. 7, Fig. 8 | BF16、TE | 9 个 VLM 基准 |
| 端到端显存与速度 | Llama-2-7B/13B、Llama-30B，1–8 GPU | Table 8 | BF16、TE | 峰值显存、最大 BS、tokens/s/GPU |
| 优化器 vs 激活各自效率贡献 | Llama-2-7B 2/4/8 GPU | App. Table 11 | BF16 → +FP8 优化器 → +FP8 激活 | 显存、吞吐 |
| 各自精度影响 | FP8 O / FP8 A / 两者 | App. Table 13 | BF16 | 训练 PPL、4 任务 |
| 组尺寸权衡 | G=4…64 | App. Table 10 | 只量化线性层 | 前/反向 MSE、额外显存 |
| Group Scaling 开销小 | RMSNorm 融合量化前后 | App. Table 12, Fig. 4b | just-in-time scaling | ms |
| Group Scaling "不劣于甚至优于 delayed scaling 的精度" | **未直接验证**（只有时间对比） | Sec 5.2 | — | — |

## 4. 实验设计手法
- **基线选择**：BF16（标准）、TransformerEngine（工业 FP8 SOTA）。FP8-LM 只出现在显存分解（Fig.1c），**没有精度/速度实测对比**；8-bit Adam 只在 MSE 层面（Table 9）对比，没有端到端训练对比。都用"官方训练 recipe 默认超参"。
- **规模/广度**：OLMo-1B/7B 预训练，Llama-2-7B 微调，VILA1.5-7B VLM SFT；效率测到 Llama-30B；硬件仅 H100。
- **效率测量**：真实 GPU；**两级**——单 transformer 层（扫 hidden size 与序列长度）和端到端（扫 GPU 数 1/2/4/8，报告固定 BS 下的显存和"最大 BS"下的吞吐）。
- **精度–效率联合**：Table 2 理论账 vs Table 7 实测显存比；Table 10 组尺寸的显存开销-MSE 权衡；无 Pareto 曲线。
- **消融**：MSE 级的格式×扩展全组合（Table 1、9）；组件级的效率分解（Table 11）与精度分解（Table 13）；组尺寸（Table 10）；线性层 per-group 的速度损失（App. H：1300→约 900 TFLOPS）。
- **分析性实验**：动态范围分布、k 分布；最优性证明；数值稳定实现（用 C_X=√(Xmin·Xmax) 归一化避免 Xmin^k 下溢，App. G.4）。
- **泛化**：LLM 预训练/微调 + VLM；模型族 OLMo/Llama/VILA。
- **统计**：单 seed，无方差。
- **系统贡献**：开源（NVlabs/COAT），Triton 线性/非线性 kernel + CUDA 融合优化器 kernel。
- **局限性**：无专门章节；结论只提未来可结合梯度通信压缩。

## 5. 对抗式审读
- **"近乎无损"依赖平均值抵消**：OLMo-7B（Table 4）ARC-Easy 65.7→61.9（−3.8），HellaSwag 56.9→60.6（+3.7），平均只差 0.5；COPA −2.0。单个任务的波动在 ±4 量级，说明这些小基准本身噪声大，平均数接近无法区分"无损"与"有损但被噪声掩盖"。训练 loss 的差（2.366 vs 2.379）是更可靠的信号，且 COAT 一致略高。
- **预训练规模**：OLMo-7B 只训 250B token，远短于正常预训练；FP8 优化器状态误差的累积效应在更长训练中可能放大，未测。
- **端到端速度主要来自 batch 翻倍**：Table 8 "Ratio" 是在各自最大 BS 下比较；8 GPU Llama-2-7B 上 TE 1.42× > COAT 1.36×，摘要"on par with or surpassing TE"在这个设置不成立（作者在正文中的措辞是"comparable or even faster"）。Table 11 说明在相同 BS（8 GPU，BS=4）时 FP8 优化器只带来 1.02×。
- **附录 Table 11 与 Table 8 不一致**：2 GPU 的"FP8 Optimizer + FP8 Activation"峰值 35.6 GB（Table 11）vs 52.8 GB（Table 8）；4 GPU 的"FP8 Optimizer"行激活已是 16.9 GB（未量化激活时应为 25.8 GB）。疑似复制错误，削弱该表作为分解证据的可信度。
- **缺少 FP8-LM 的精度对比**：FP8-LM 是最接近的工作，却只在显存图中出现。
- Table 13（精度分解）未说明模型与训练步数，FP8 A 平均分反而高于 BF16（52.3 vs 51.6），再次说明下游任务噪声大于被测效应。
- Group Scaling 精度不劣于 delayed scaling 的说法无实验。

## 6. 可迁移的手法
1. 先用"单位化"的算子级显存账本（U = B×L×H×2 bytes）算出理论上限，再用实测逼近它（1.69× vs 1.65×），让读者相信收益不是偶然（Table 2 → Table 7）。**直接读到**
2. 为一个量化变换给出"在一组合理公理下唯一"的证明，把经验选择变成必然选择（App. F）。**直接读到**
3. 在端到端评测前，用"方法真正影响的那个量"（m/√v 的 MSE，而非 m、v 各自的误差）做格式×方法的全组合表（Table 1）。**直接读到**
4. 效率评测按"单层 → 端到端 → 扫 GPU 数 → 最大可行 BS"分层报告，并把"能否放下"（OOM vs ✓）作为一类结果（Table 7, Table 8）。**直接读到**
5. 精度评测横跨预训练、微调、多模态三类任务，用"任务类型覆盖"替代"大规模多 seed"（Table 3–6）。**直接读到**（但见第 5 节对噪声的保留）
6. 组件消融分成"效率分解"和"精度分解"两张表，分别回答"省了多少"和"伤了多少"（Table 11, Table 13）。**直接读到**

## 7. 一句话
用算子级激活显存账本（Table 2）先算出"该压哪里、能压多少"，再用实测 1.65× 对上理论 1.69×，配合"FP8 动态范围没用满"这一可视化观察和幂函数唯一性证明，把一个工程框架包装成有理论下限、有实测闭环的方法。
