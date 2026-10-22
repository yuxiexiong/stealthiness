# Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

## 0. 元信息
- arXiv 2505.22618v3（2025-07），NVIDIA/MIT/HKU；正文未写会议名。
- 类型：算法（training-free 推理加速），带一个理论定理。
- 全文已读（正文 + 附录 A 证明、B 案例、C 补充实验）。pdftotext 下 Figure 1/5/8/9 的坐标轴文字错位，图中数字只引用正文/图注复述过的。

## 1. 核心主张
- 主：Diffusion LLM（LLaDA、Dream）可以在不训练的前提下，通过 (a) 块级近似 KV Cache（含 DualCache）和 (b) 置信度阈值并行解码，获得数量级加速（最高 27.6×）而精度基本不掉。
- 次 1：并行解码掉质量的根因是条件独立假设破坏 token 依赖；在高置信区间（(n+1)ε≤1）贪心并行与贪心顺序等价（Theorem 1）。
- 次 2：相邻步之间 KV 激活高度相似，所以近似缓存可行。

## 2. 动机是怎么立住的
- 痛点：开源 dLLM 实际速度不如 AR（Intro 引 LLaDA 原文"一次一 token 最好"）；没有自测 profiling 图证明瓶颈，痛点基本靠文献陈述 + Fig.1 的 baseline 吞吐（LLaDA 6.7 tok/s vs LLaMA-3-8B）。
- 关键观察 1（缓存）：Fig.3 KV 激活跨推理步的余弦相似度热图（prompt block 与 last block），对角邻域接近 1、远处低——直接对应"块内可复用、块间要刷新"的设计。因果链闭合得很好：热图结构 = 方法结构。
- 关键观察 2（并行）：用 poker hands 例子（"high house"）+ 联合 vs 边缘乘积的形式化（Sec 2.2）说明问题，再由 Theorem 1 给出"高置信时可以并行"的条件。注意：没有一个实测图证明"错误确实集中在低置信 token 上"，因果链在经验层面靠 Fig.5 的阈值扫描间接支撑。

## 3. 主张 → 证据映射表
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| Cache 与 Parallel 各自有效且互补 | 2×2 因子表（base/+Cache/+Parallel/+both）× 4 任务 × 2 长度 | Table 1 (LLaDA), Table 2 (Dream) | 原始 LLaDA / Dream | acc + tok/s + 倍数 |
| 最高 27.6× | 8-shot、gen len 1024、DualCache | Table 4/5，Fig.1c | vanilla LLaDA 0.7 tok/s | tok/s |
| 阈值策略优于固定 top-k | 阈值 0.5–1.0 扫描，对比 2/4/8 token/step | Fig.5 (a)(b)(c) | 固定 token 数 baseline | GSM8K acc vs tokens/step、NFE |
| 理论：高置信下并行=顺序 | 定理+证明 | Thm 1, App. A | — | 无实验直接验证 bound |
| factor 策略更快 | factor vs threshold | Table 11, Fig.8 | threshold | acc + 吞吐 |
| 跨模态泛化 | LLaDA-V MathVista/MathVerse | Table 3, Table 9–10 | Full Steps / Half Steps | acc + 吞吐 |
| 加速随 prefill/生成长度增大 | 5-shot vs 8-shot；256/512/1024 | Table 4, Table 5 | — | 倍数 |
| 小 batch 可追平 AR | batch 1–32 吞吐 | Fig.9, App C.5 | LLaDA, LLaMA | tok/s（仅 cache，不含并行） |
| "closing gap with AR" | 只有 Fig.1a 与 Fig.9 | — | LLaMA-3-8B/Qwen2.5-7B | **部分未验证**：大 batch 下作者自承落后于 LLaMA |

## 4. 实验设计手法
- **基线选择**：只与"原始 dLLM"及"固定 token/step"朴素基线比较；**没有**与同期 dLLM 缓存工作（dKV-Cache、dLLM-Cache 等）或 Block Diffusion 对比。LLaDA-V 上给了"Half Steps"作为朴素降步数基线（Table 3），这是好做法——证明加速不是"少走步数"就能得到。
- **规模/广度**：模型 LLaDA、LLaDA-1.5、Dream（均 ~7–8B），+LLaDA-V 多模态；4 个文本 benchmark + 2 个多模态；单一硬件 A100 80GB。无尺寸跨度。
- **效率测量**：真实 GPU 实测，端到端 tok/s 计到 <eos>（Sec 4.1），batch=1、无推理框架（Fig.1 图注）；lm-eval 统一评测。Fig.9 做了 batch 扫描，但特意只测 cache（"to ensure fairness"）。
- **精度-效率联合**：每个格子同时给 acc 与吞吐（Table 1/2 的"上下两行"格式）；Fig.5c / Fig.8c 是 acc vs tokens-per-step 的曲线对比，属于"同等并行度下比精度"。
- **消融**：2×2 因子拆解（Table 1/2）；Fig.1c 是逐步累加 roadmap（Parallel 13.3× → +PrefixCache 1.4× → +DualCache 2.1× → 27.6×）；超参扫描：block size 4–256（Fig.4）、阈值（Fig.5）、factor（Fig.8）、refresh interval（Table 10）。
- **分析性实验**：KV 相似度热图（Fig.3）、每步并行 token 数曲线带 95% CI（Fig.7）、定性案例（Table 6–8, Fig.6）。
- **泛化**：跨模型（LLaDA/Dream/LLaDA-1.5）、跨模态（LLaDA-V）、跨长度。
- **统计**：无多 seed；Fig.7 的 95% CI 是跨样本的并行 token 数，不是精度方差。精度差 1–2 点无显著性检验。
- **系统贡献**：开源（Github 链接），无框架集成。
- **局限性**：App C.5/Sec 4.3 承认大 batch 下 dLLM 是 compute-bound，PrefixCache 追不上 LLaMA；Remark 1 承认 MDM 不满足良定义联合分布的假设。

## 5. 对抗式审读
- **最薄弱环节**：没有与同期 dLLM 加速方法比较，"SOTA acceleration"（贡献 3）未被对照实验支撑；基线仅为自身的未加速版本。
- **定理与实践脱节**：Theorem 1 需要 ε≤1/(n+1)，但默认阈值 0.9（ε=0.1）只在 n≤9 成立；factor 策略的 f 取 0.7–1.9（Fig.8），实际已超出定理保证区间（f>1 即违反 (n+1)ε≤1），所以 factor 策略"theoretically grounded"是修辞多于保证。Remark 1 自己承认 MDM 不满足定理前提。
- **27.6× 的选点**：来自 8-shot、gen len 1024、baseline 只有 0.7 tok/s 的设定——baseline 越慢倍数越大；batch=1 且无推理框架，倍数对实际部署的含义有限。DualCache 在该设置下精度 76.0 vs 77.3（Table 4），Table 5 len 512 的 PrefixCache 掉到 74.8 vs 78.9（-4.1），"1–2 点内"的总结（Sec 4.2）并不覆盖所有格子。
- **数字小不一致**：Table 12 中 LLaDA(Fast-dLLM) GSM8K 256 吞吐写 54.1，Table 1/11 写 54.4。
- **多模态用的是另一套配置**（full block + refresh，Table 9–10），而非主方法的小块缓存——说明主方法在 LLaDA-V 上不直接适用，这点在正文被包装成"适配"。
- **定性案例**只挑了一道简单算术题（Table 6–8），不提供失败案例。

## 6. 可迁移的手法
1. 在"近似复用中间状态"的加速方法里，用跨步相似度热图（对角结构）先证明可复用性，热图的块结构直接对应方法的块结构。出处 Fig.3 → Sec 3.2。【直接读到】
2. 两个正交组件用 2×2 因子表（base/+A/+B/+A+B）在多任务多长度上展示"各自有效且互补"。出处 Table 1/2。【直接读到】
3. 动态策略 vs 固定策略：把横轴统一为"平均每步 token 数"，在同等并行度下比精度，避免阈值和 top-k 不可比。出处 Fig.5c、Fig.8c。【直接读到】
4. 给一个朴素降算力基线（Half Steps），排除"只是少算"的解释。出处 Table 3。【直接读到】
5. 加速倍数随工作负载长度扫描（prefill、gen len），展示收益随规模增长的趋势而非单点。出处 Table 4/5。【我的归纳：趋势比单点更难被质疑为挑选】
6. 用理论定理给启发式（阈值）背书，但要留意定理条件与实际超参是否吻合。出处 Thm 1 vs Fig.8。【我的归纳】

## 7. 一句话
最关键的一招是 Fig.3 的 KV 相似度热图 + Table 1/2 的 2×2 因子表：前者让"近似缓存"显得有物理依据，后者让两个组件的贡献和互补性一目了然；但缺乏与同期方法的横向对比。
