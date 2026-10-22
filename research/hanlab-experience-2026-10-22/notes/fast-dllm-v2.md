# Fast-dLLM v2: Efficient Block-Diffusion LLM

## 0. 元信息
- arXiv 2509.26328v1（2025-09），NVIDIA/MIT/HKU；未写会议。
- 类型：算法（AR→block-diffusion 的后训练适配 recipe）+ 推理流水线（分层缓存 + 并行解码）。
- 全文已读（正文 + 附录 A 实现、B 案例、C LLM 使用声明）。Figure 1/4/5/6 坐标数据在文本中错位，只引用正文复述的数字。

## 1. 核心主张
- 主：用 ~1B token 微调就能把预训练 AR 模型（Qwen2.5-Instruct 1.5B/7B）"无损"转成 block diffusion LLM，并比标准 AR 解码快最高 2.5×。
- 次 1：block-wise 注意力 + complementary mask + token shift 的训练 recipe 是关键（相比 Dream 的 ~500B token，数据效率 500×）。
- 次 2：分层缓存（block 级 KV cache + sub-block DualCache）+ 置信度并行解码在 dLLM 中给出 SOTA 效率。

## 2. 动机是怎么立住的
- 痛点论证主要是文字：dLLM 双向注意力不兼容 KV cache；Fast-dLLM 的近似缓存"不等价于原计算"（Intro）；BD3-LM 只在小模型上验证。**没有自测 profiling 或观察图**。
- 关键 insight："block-wise 注意力更接近 AR，所以适配更省数据"（贡献 1），但这一点**没有被直接实验检验**（例如没有拿 full-attention 适配在同等 1B token 下对比）。500× 是和 Dream 报告的训练量比，不是同条件对照。
- Fig.1 作为"结论先行"图：GSM8K 准确率 vs 吞吐散点，以及 batch 1/4 吞吐柱状。

## 3. 主张 → 证据映射表
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 精度追平/超越 AR | 9 项 benchmark 均值 | Table 1 | LLaMA-3.2, SmolLM2, Qwen2.5(原版 + 同数据 NTP 微调 "Nemo-FT"), LLaDA 系列, Dream | acc, Avg |
| 训练 recipe 各组件有效 | naive token shift → +pad → +pad+CM | Table 2（1.5B） | 逐步累加 | 9 项 acc，Avg 41.3→42.2→45.0 |
| 并行解码 2.6× | 阈值扫描 0.5–1.0 | Fig.4（GSM8K） | 阈值 1.0（非并行） | acc + tok/s：39.1→101.7 |
| 比 AR 快且随 batch 扩展 | batch 扫描到 64，A100 与 H100 | Fig.5 | Qwen2.5-7B-Instruct AR | tok/s：1.5×（A100）/1.8×（H100） |
| sub-block 设计必要 / 训练-推理块大小需一致 | sub-block size 扫描 vs 直接改 block size | Table 3 vs Table 4 | 不一致块大小 | GSM8K/HumanEval |
| sub-block cache 只增速不掉点 | 2×2（cache on/off × batch 1/32）× sub-block size | Fig.6 | — | acc + 吞吐 |
| "~1B token""500× 数据效率" | 无对照实验 | App A.1 | Dream 报告值 | **未直接验证**；且 1.5B 实际 3.15B token |
| "SOTA efficiency among dLLMs" | Fig.1b 只和 Fast-dLLM-LLaDA/Dream 比 | Fig.1b | 自家前作 | **部分验证**：未比 dKV-Cache、dLLM-Cache、SDAR、D2F 等 |

## 4. 实验设计手法
- **基线选择**：亮点是"同数据同步数的 NTP 微调"对照（Qwen2.5-*-Nemo-FT，Sec 4.1），这把"数据本身带来的增益"与"block diffusion 方法的影响"分开——好的反事实对照。缺点：dLLM 加速方面的同期工作（Sec 2.2/2.3 列了十余篇）一篇都没进实验；LLaDA-1.5/LLaDA-MoE 的数字有缺项（"-"），疑似抄报告值。
- **规模/广度**：Qwen2.5 1.5B、7B 两个尺寸，单模型族；9 项 benchmark；硬件 A100 + H100（Fig.5）。
- **效率测量**：真实 GPU 吞吐；AR 基线的推理实现（HF？vLLM？）**未交代**——这对"比 AR 快"是关键缺失。batch 扫描到 64（Fig.5），并做了 batch 1 vs 32 的 memory-bound/compute-bound 区分（Fig.6b）。
- **精度-效率联合**：Fig.4 同一图上 acc 与吞吐随阈值变化；Fig.1a 是 acc–吞吐散点。但 Table 1 的精度是 **threshold=1（不开并行）** 下测的（App A.4），而 2.5× 加速是在 threshold=0.9 下——精度与速度来自不同配置，只有 GSM8K 一项在 Fig.4 上联合呈现。
- **消融**：训练 recipe 逐步累加（Table 2）；推理端 sub-block size、block size、cache on/off 扫描（Table 3/4, Fig.6）。所有消融只在 1.5B 上。
- **分析性实验**：几乎没有机制分析；案例（Table 5/6）是展示性质。
- **泛化**：两尺寸、两 GPU 代际；无跨模型族。
- **统计**：无方差、无多 seed。
- **系统贡献**：声明开源代码与模型；无框架集成。
- **局限性**：无专门局限性章节。

## 5. 对抗式审读
- **"lossless" 与数据不符**：Table 1 中 7B 的 MATH 61.6 vs Qwen2.5-7B 73.3 / Nemo-FT 72.0（掉 ~10 分），GPQA 31.9 vs 34.2，MBPP 也低于 Dream；1.5B 的 MATH 38.1 vs 46.8。Avg 的领先（60.3 vs 59.6）靠 HumanEval 的增益拉平。"matches or surpasses AR"只在平均值意义上成立。
- **"~1B token"与附录不符**：App A.1 自己算出 1.5B 用 3.15B token、7B 用 1.31B token。
- **速度对比的 AR 基线实现未说明**；若 AR 用 HF generate 而无 CUDA graph / 高效 kernel，"2.5× faster than AR"含义会大打折扣。Fig.1b 中 batch 4 时 Fast-dLLM v2 的优势明显缩小（数字错位，只能读到 217.5/113.2/102.5 等若干值，无法确认对应关系——如实标注）。
- **精度和速度取自不同配置**：Table 1 精度在不并行下测得，加速在并行下测得；除 GSM8K（Fig.4）外，其他任务在 threshold=0.9 下精度未知。
- **数据效率主张缺少同条件对照**：没有在同 1B token 下训练 full-attention 适配来证明"block-wise 更 AR-friendly"。
- 同期工作 SDAR（同样从 AR 微调 block diffusion）只在相关工作里用"我们更省数据"一句话区分，无实验。

## 6. 可迁移的手法
1. 改训练目标/架构时，加入"同数据、同步数、原目标"的对照模型（Nemo-FT），把数据增益与方法增益剥离。出处 Table 1、Sec 4.1。【直接读到】
2. 训练 recipe 用逐步累加表（naive → +pad → +pad+CM）证明每个工程细节都有贡献。出处 Table 2。【直接读到】
3. 用"推理时改 sub-block（保持训练结构）vs 直接改 block size（破坏训练结构）"的成对表，证明训练-推理一致性是设计约束。出处 Table 3 vs Table 4。【直接读到】
4. 缓存类优化要分 memory-bound（小 batch）与 compute-bound（大 batch）两种状态展示，诚实说明小 batch 下无收益。出处 Fig.6b、Sec 4.3。【直接读到】
5. 跨两代 GPU（A100/H100）展示加速在新硬件上更大，增强"趋势向好"的论证。出处 Fig.5。【直接读到】
6. 反面教训：精度表与速度表若在不同解码配置下测，需逐任务给出加速配置下的精度，否则"无损加速"不闭合。出处 App A.4 vs Fig.4。【我的归纳】

## 7. 一句话
最有说服力的一招是"同数据同步数 NTP 微调"对照基线（Nemo-FT）配合 Table 2 的逐步累加——但"无损"和"~1B token"两个标题级主张被自己的 Table 1 与附录削弱。
