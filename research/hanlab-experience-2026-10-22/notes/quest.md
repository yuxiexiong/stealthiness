# Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

## 0. 元信息
- ICML 2024（PMLR 235；arXiv 2406.10774v2）；Tang*, Zhao*, Zhu, Xiao, Kasikci, Han（SJTU/MIT/UW/NVIDIA）
- 类型：算法-系统协同（查询感知页级 KV 选择 + FlashInfer 上的 CUDA kernel）
- 全文读完；本论文**无附录**。Fig.3/4/7 的曲线数值无法从文本读出；Fig.10/11 柱值可部分读出。

## 1. 核心主张
- 主：token 的"关键性"随 query 变化，因此不应永久逐出 KV；Quest 以页为单位保存每通道 min/max Key 作为元数据，用 query 求注意力上界，仅加载 Top-K 页，自注意力快至 7.03×、端到端解码 2.23×（4-bit 权重），精度几乎无损。
- 次：在"同等无损精度"约束下，Quest 所需 token 预算远低于 TOVA，自注意力延迟低 3.82–4.54×。

## 2. 动机是怎么立住的
- 痛点：Sec.3.1 给出 profiling 式论证——16K prompt/512 输出时解码占 >86% 时间；32K 上下文 Llama-7B 的 KV 为 16GB，读取占解码延迟 53%（脚注：RTX4090 + FlashInfer 实测）。
- 观察 1（稀疏性）：Fig.3 每层"在 PG19 PPL 增量 <0.01 条件下可删掉的 KV 比例"——除前两层 <10% 外其余 >90%；同图叠加 Quest 估计与 oracle 的对齐。
- 观察 2（查询依赖）：Fig.2 用 "A is B. C is D. A is" 的单头注意力图展示 "B" 先不重要后关键（定性）；Fig.4 用"Top-10 注意力 token 的召回率"量化：H2O 召回低、Quest 接近 full。
- 因果链：解码占主导 → 读 KV 是瓶颈 → 注意力稀疏 → 但稀疏模式随 query 变 → 所以要"保留全部 KV、按 query 选择"。链条清楚；Fig.3 的"前两层不稀疏"直接导出"前两层不应用"的设计。

## 3. 主张 → 证据映射表
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 注意力高度稀疏（前两层除外） | 每层可剪比例 | Fig.3 | oracle | 稀疏度 |
| 关键 token 依赖 query | Top-10 召回率 | Fig.4；Fig.2 | Full, H2O | recall |
| 语言建模不掉 | PG19 32K PPL，预算 4096 | Fig.6 | H2O*, TOVA*, Full | PPL |
| 长依赖检索保持 | passkey 10K / 100K | Table 1 | H2O/TOVA/SLLM | Acc（如 64 预算 99% vs ≤1%） |
| 一般长上下文任务 | LongBench 6 任务预算扫描 | Fig.7 | 同上 | F1 |
| kernel 效率 | 估计、Top-K、近似注意力各自延迟 | Fig.8，Sec.4.3.1 | FlashInfer | μs |
| 自注意力 7.03× | 时间拆解 | Fig.9 | FlashInfer | μs |
| 端到端 2.23× | 单 batch 解码延迟 | Fig.10 | FlashInfer（FP16/AWQ） | ms/token |
| 同精度下比效率 | 无损所需预算与延迟 | Fig.11 | Full, TOVA | 预算、μs |
| 内存加载降低 1/S + K/PageNum | 解析式 | Sec.3.5 | — | **理论推导，非实测** |

## 4. 实验设计手法
- 基线：H2O、TOVA、StreamingLLM；所有方法前两层都不压缩，标注 H2O*/TOVA*（公平对齐）。H2O 无法在长 prefill 中用 FlashAttention，因此改为从解码阶段开始累计分数（Sec.4.2.2）。LongBench 把问题部分"逐 token 喂入模拟解码"——这是让逐出类方法暴露缺陷的关键设置。缺 SparQ 的数值对比（只在 Related Work 提及）。
- 规模：仅 LongChat-7B-32K 与 Yarn-Llama-2-7B-128K 两个 7B 模型；效率只在 Llama2-7B。广度明显不足。
- 效率：NVBench kernel 级 + PyTorch profiler 拆解 + 端到端；RTX4090（kernel）与 Ada 6000（端到端）；序列长度 8K–32K 扫描、预算 512–4096 扫描；单 batch；交代了 CUDA 12.2、page size 16。Top-K 开销 5–10μs 被排除在效率分析外（脚注 ‡）。
- 精度–效率联合：Fig.11 "同精度比效率"——先定"LongBench 无损"为约束，再比较所需预算与延迟。但基线的延迟是用 FlashInfer 在相同预算下的延迟**代替**（作者明说基线无 kernel，"qualitative analysis"），且忽略了 TOVA 计算历史分数的开销（该忽略对基线有利）。
- 消融：无正式消融（page size 只在 kernel 延迟里扫描，未见对精度的影响；min/max 上界 vs 其他估计方式未比较）。
- 分析性：Fig.3 oracle 对齐、Fig.4 召回率。
- 统计：无。
- 系统：开源，FlashInfer 集成，与 AWQ 组合。
- 局限：无局限节。Quest 不减显存（保留全部 KV + 元数据额外占用，Sec.4.3.1 只一句"量化或更大页可减少额外内存"）。

## 5. 对抗式审读
- 最薄弱：模型广度（两个 7B、均为 Llama-2 系），无 GQA 模型；没有页大小对精度的消融。
- Table 1 passkey 的优势巨大（99% vs 1%），但这是"答案在问题之前被逐出"的结构性优势，任何不逐出的方法都会赢；逐出类基线本就为显存设计，与只省带宽不省显存的 Quest 并不在同一约束下比较——**比较轴不对齐**（Quest 保留全部 KV）。
- 7.03× 是自注意力 kernel 级、单 batch、32K、预算 2048；端到端只有 1.74×（FP16），2.23× 需配合 4-bit 权重才得到（权重量化减少了非注意力部分时间，放大了注意力占比）。
- "上界"估计是严格上界，但 Top-K 按上界排序不保证真实 Top-K；文中仅以 Fig.3/4 的经验对齐支撑。
- Fig.11(b) 基线延迟为代理值。

## 6. 可迁移的手法
1. 用"在 PPL 增量 <ε 的约束下最多能删多少"逐层量化稀疏度，同时画出本法估计与 oracle 的差距。（Fig.3；直接读到）
2. 用一个指标（Top-k 召回率）把"动态性"这一定性观察量化，并直接显示竞争方法在该指标上的失败。（Fig.4；直接读到）
3. 设计评测协议让竞争方法的结构性缺陷暴露：把问题放在材料之后逐 token 解码。（Sec.4.2.3；直接读到）
4. "同精度比效率"：先固定无损约束，比较达到该约束需要的预算与延迟。（Fig.11；直接读到）
5. 效率三层呈现：单 kernel → 注意力时间拆解 → 端到端，并配合正交优化（AWQ）放大端到端收益。（Fig.8–10；我的归纳）

## 7. 一句话
Top-10 召回率这一个量把"关键 token 会随 query 变"从直觉变成了数字，再配上让逐出类方法必败的 passkey 设置，论点几乎无法反驳——尽管比较轴（显存 vs 带宽）并不完全对齐。
