# Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search

## 0. 元信息
- arXiv 2508.15884v3（2025-09，NVIDIA 技术报告格式；投稿去向文中未写）；Gu, Hu, Yang, Xi, Chen, Han, Cai
- 类型：算法/架构（PostNAS 搜索流程 + JetBlock 新线性注意力块 + 混合模型家族）
- 全文读完（正文 + 附录 A.1–A.3、B.1–B.3）。Fig.1/3/5 的数值部分可从文本读出（Fig.3 各阶段柱值、Fig.6 加速比）。

## 1. 核心主张
- 主：从预训练全注意力模型出发、冻结 MLP，只搜索注意力块（全注意力层放置 → 线性块选择 → 新块设计 → 硬件感知超参搜索），得到的混合模型 Jet-Nemotron-2B 精度与 Qwen3-1.7B 等 SOTA 全注意力小模型相当或更好，64K 上下文时生成吞吐高 47×（256K 时 53.6×，prefill 6.14×）。
- 次 1（Key Finding 1–3）：不同注意力层对不同能力的贡献不均且任务相关（MMLU 只依赖约 2 层，检索 2–3 层）。
- 次 2（Key Finding 4）：长上下文生成吞吐主要由 KV cache 大小决定而非参数量；固定 cache 时可加参数提精度不损吞吐。
- 次 3：JetBlock（动态卷积核作用于 V）优于 Mamba2/GLA/Gated DeltaNet 等。
- 次 4：PostNAS 可作为架构创新的低成本"测试平台"。

## 2. 动机是怎么立住的
- 痛点：Introduction 定性陈述（O(n²) 与 KV cache；已有混合模型在 MMLU/数学/检索/代码上落后 SOTA 全注意力模型）。Fig.1 一张"精度 vs 吞吐"散点图（MMLU-Pro vs 64K 吞吐）把本法放在右上角，直接作为动机兼结论。
- 方法论动机：Sec.2.1 论证从头预训练架构探索成本过高、小代理模型不可靠；PostNAS 复用预训练权重。并提出"若新设计在此设定下不行，从头训也不会行"（引 [43]）——**这一论断未在文中验证**。
- 关键观察：Fig.5(a) 逐层"单层保留全注意力、其余换线性"后的任务目标热图（Qwen2.5-1.5B），显示层重要性稀疏且任务相关；Table 2 显示相同 cache 下参数 1.62B→1.98B 吞吐几乎不变（2,952–2,986 token/s）。两个观察都直接导出设计决策（学习放置位置；固定 cache 加参数）。

## 3. 主张 → 证据映射表
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 层重要性稀疏、任务相关 | 逐层热图 | Fig.5(a) | — | 任务 loss/acc |
| 学习放置优于均匀放置 | 全注意力层数扫描 | Fig.5(b) | Uniform | MMLU |
| PostNAS 每步都有增益 | 累加拆解 | Fig.3 | 基线（全线性） | MMLU/Math/Retrieval/Common |
| JetBlock 优于现有线性块 | 同设置 50B token | Table 1 | RetNet/Mamba2/GLA/DeltaNet/GDN | 训练/推理吞吐 + 4 类精度 |
| KV cache 决定吞吐 | 固定 cache 的超参网格 | Table 2 | 原始 GDN 配置 | 吞吐、精度 |
| 主精度 | MMLU(-Pro)/BBH、数学、常识、检索、代码、LongBench | Table 3–8 | 14 个全注意力/线性/混合模型 | Acc + 吞吐 |
| 吞吐 47×/53.6× | 多上下文长度 | Fig.6, Table 3 | Qwen3-1.7B | 相对加速；理论上界 56× |
| 不是数据带来的 | 同数据续训基线 | Table 14（附录 B.1） | Qwen2.5/Mamba2/RWKV7-continual | Acc |
| 跨硬件 | Orin、3090 | Table 15 | Qwen2.5-1.5B | token/s |
| 与并发工作比较 | Falcon-H1 | Table 16 | Falcon-H1(-deep) | Acc + 吞吐 |
| PostNAS 结论可迁移到从头训练 | — | Sec.2.1 文字 | — | **未直接验证** |

## 4. 实验设计手法
- 基线：覆盖三类架构（O(n²)、O(n)、Hybrid）共 14 个公开模型 + 两个 MoE（仅引用其报告分数）+ 附录并发工作 Falcon-H1。基线为现成权重，训练数据不同；附录 B.1 用同一 stage-2 语料续训 Qwen2.5/Mamba2/RWKV7 做**数据受控对照**，这是关键补强。
- 规模：2B、4B 两个尺寸，均基于 Qwen2.5。
- 效率：DGX H100 单卡，交代软件版本（PyTorch 2.7.0、Triton 3.3.0、FA 2.7.4、FLA 0.2.1、Transformers 4.52.0），每个模型用 chunk-prefill 搜索使 batch 最大化（附录 Table 13 列出每个模型的 batch/chunk）。给出理论上界 56×（14 倍层数 × 4 倍 KV 组）并说明实测 53.6× 接近上界——实测与解析模型对账。另测 Orin/3090。
- 精度–效率联合：所有主表都把吞吐列放在精度旁；Fig.1 为 Pareto 式散点。
- 消融：Fig.3 逐步累加（roadmap）；Table 1 线性块横比；Table 2 超参网格；Fig.5(b) 放置策略。
- 成本透明：附录 Table 12 列出每个 PostNAS 步骤的 token、ZFLOPs、H100 小时（如硬件感知搜索 5,616 GPU 时）。
- 统计：无。
- 系统：开源（GitHub）。
- 局限：无专门节；只在 Sec.3.3 承认短上下文 prefill 仅 1.14×，需更好 kernel。

## 5. 对抗式审读
- 最薄弱：**吞吐比较的口径**。47× 是"各模型在单 H100 上用最大可容纳 batch 时的解码吞吐"（Table 13：Qwen3 batch 8，Jet-2B batch 128）。这是服务吞吐而非单请求延迟；单 batch 下的差距未报告，"47× 更快"的说法容易被读成延迟。
- **搜索目标与评测基准重叠**：全注意力层放置以 MMLU（−loss）、retrieval、math 准确率为搜索目标（Sec.2.2），最终又用这些基准报告精度，存在"在评测任务上选架构"的风险；文中未说明搜索用的是不是独立划分。
- 附录 Table 9 中 4B 的全注意力层（No.18,21,22,28,33）与 SWA 层（No.17,20,23,24,26）与正文 Sec.3.1（全注意力 18,21,33；SWA 6,17,20,22,23,26,28）**不一致**。Table 13 列 Llama3.2-1B，主表却是 Llama3.2-3B。Table 14 中 Jet-2B MMLU 59.6 与 Table 3 的 60.8 不同，未说明设置差异。
- Jet-2B 起点 Qwen2.5-1.5B，额外训练 400B token（含数学/代码数据）；对 Qwen3-1.7B 的精度优势中有多少来自数据，只有 B.1 部分回答（Qwen2.5 续训后 MMLU 56.7、Math 37.6 仍低于 Jet，但差距比主表小）。
- 检索与长上下文上 Jet-2B 不如 Qwen3-1.7B（Table 6：74.2 vs 76.1；Table 8：41.1 vs 42.2），摘要中"comparable or superior"在这些项上是"comparable"。
- "PostNAS 结论可迁移到从头预训练"无实验。

## 6. 可迁移的手法
1. 给效率加速比配一个解析上界（层数比 × KV 组比 = 56×），用"实测接近理论上界"证明测量可信、收益来自机制。（Sec.3.3, Fig.6；直接读到）
2. 对"新架构 vs 现成模型"的比较，补一张"同数据续训基线"的受控表，堵住"是数据不是架构"的质疑。（Table 14；直接读到）
3. 公开每个基线的测量超参（batch/chunk）与软件版本，使吞吐比较可复核。（Table 13, Sec.3.1；直接读到）
4. Roadmap 式累加拆解：从基线到最终模型每一步对每类任务的增益。（Fig.3；直接读到）
5. 把一个"反常识"的系统观察（参数量不决定吞吐，cache 才决定）做成固定一变量的网格表，并直接转化为设计规则。（Table 2, Key Finding 4；直接读到）
6. 把发现写成编号的"Key Finding"框，便于审稿人引用。（Sec.2.2/2.5；我的归纳）

## 7. 一句话
"每张精度表都带吞吐列 + 实测 53.6× 逼近解析上界 56× + 同数据续训对照"三件套让效率与精度两头都难以被反驳；最大漏洞是吞吐用了各自最大 batch、以及搜索目标与评测基准重叠。
