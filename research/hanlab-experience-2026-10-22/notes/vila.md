# VILA 精读笔记

## 0. 元信息
- 标题：VILA: On Pre-training for Visual Language Models
- 来源：arXiv 2312.07533v4（2024-05；NVIDIA/MIT）；文本未写会议（据我所知为 CVPR 2024，未在文本中核实）
- 类型：经验研究 + 模型配方（"design space study"型算法论文）
- 全文读完（正文 + 附录 A–E）。Figure 1 雷达图、Figure 3 曲线、Figure 5 损失曲线的数值在文本中只剩零散数字，不引用。

## 1. 核心主张
- 主主张：VLM 的**预训练阶段**（而非仅指令微调）的设计选择决定下游表现；按其配方得到的 VILA 在 12 个基准上"一致"超过 LLaVA-1.5。
- 三条发现（次主张）：(1) 预训练时冻结 LLM 零样本尚可但丧失 in-context learning，需要解冻；(2) 交错图文数据（MMC4）优于纯图文对，且好处来自"交错结构"而非"更好的文本分布"；(3) SFT 中混入纯文本指令数据可恢复文本能力**并**提升 VLM 精度。
- 附加主张："deep embedding alignment" 假说——解冻 LLM 使深层视觉/文本嵌入对齐，与 ICL 正相关。

## 2. 动机怎么立住
- 痛点：领域集中于指令微调，"缺乏对预训练过程的深入研究"（Sec. 1），这是**研究空白式**动机，没有测量证据。
- 关键观察的可视化：Figure 3 画逐层视觉/文本嵌入的余弦相似度，配置 (b)→(d) 深层相似度升高，与 Table 1 的 4-shot 提升同向——用一个内部量去解释一个行为量。但只有 3 个配置点，是相关而非因果；且文中先说"计算 Chamfer distance"又说"计算 pairwise cosine similarity"，度量定义前后不一致。
- 因果链最强的一环是 MMC4 vs MMC4-pairs（Table 3）：同一数据源，只去掉交错结构，直接隔离"结构"与"文本分布"两个混杂变量；附加的"打乱为 <im1><im2><txt1><txt2>"实验（Sec. 4.4，0-shot −4.4%、4-shot −37.5%）进一步确认。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 解冻 LLM 才有 ICL | PreT/SFT 是否训 LLM × 投影器 | Table 1 | (a)(b)(c)(d) 四配置 | 4 基准 0/4-shot |
| 深层对齐假说 | 逐层余弦相似度 | Fig. 3 | 配置 b–d | 相似度（相关性） |
| 交错数据更好、来自结构 | 预训练数据替换 | Table 3, Fig. 5 | COYO、MMC4-pairs、MMC4+COYO | VLM avg、MMLU |
| 交错顺序重要 | 打乱重排 | Sec. 4.4 | 原 MMC4 | 0/4-shot 降幅 |
| 联合 SFT | 混入 1M FLAN 文本 | Table 4 | 仅视觉 SFT、Llama-2+文本 SFT | VLM avg、MMLU |
| SOTA | 12 基准 | Table 5 | BLIP-2、InstructBLIP、Qwen-VL、LLaVA-1.5 等 | 各基准 |
| 保持文本能力 | MMLU/BBH/DROP | Table 6 | Llama-2、Llama-2+SFT、Vicuna-1.5 | 准确率 |
| 分辨率 > token 数 | 224/336/下采样 | Table 7 | — | OKVQA/TextVQA/COCO |
| 优于视觉专家 / LoRA | 替代方案 | Table 8, 9 | CogVLM 式视觉专家、LoRA r=64 | VLM avg、6 基准 |
| 跨 LLM 成立 | Vicuna-1.5-7B 复做 | 附录 Table 11 | Llama-2 | 0/4-shot、3 基准 |
| 可部署于 Jetson Orin | — | 摘要 | — | **未直接验证（正文无任何数据）** |
| 世界知识更好 | 4 张地标图 | Fig. 9 | LLaVA-1.5 | 4/4 vs 2/4（n=4） |

## 4. 实验设计手法
- 基线：消融全部是"受控逐步比较"（同 LLM、同 SFT blend、同评测）；SOTA 表用已发表数字，并标 * 区分训练集可见的基准。称与 LLaVA-1.5 为 "head-to-head（相同 prompt 与底座）"。
- 规模/广度：7B/13B 两个尺寸；附录换 Vicuna 验证；12 个 VLM 基准 + 3 个纯文本基准。
- 效率测量：无（仅附录 B 给训练成本 5.1k GPU 小时）；摘要提到 Orin 部署但无数据。
- 精度–效率：Table 7 涉及 token 数与精度权衡（下采样投影器 144 token）。
- 消融：训练协议、投影器、预训练数据、SFT 数据、分辨率、视觉专家、LoRA、数据顺序——覆盖面广，是"设计空间研究"的典型。
- **双指标守恒**：每个数据消融同时报告 VLM 精度与 MMLU（文本能力），把"加视觉不伤语言"作为与主指标并列的约束（Table 3/4/6）。
- **0-shot 与 4-shot 分离**：把 ICL 作为独立维度测量，揭示了"冻结 LLM 零样本不降、少样本降"这一只有分维度才能看见的现象（Table 1）。
- 统计：无方差；消融用"different evaluation setting"，绝对值不可与他文比较（Table 1 注）。
- 系统：开源代码与模型。
- 局限：未能扩到十亿级预训练数据（Sec. 4.1）。

## 5. 对抗式审读
- **"head-to-head"并不完全受控**：VILA 与 LLaVA-1.5 的差异不只是预训练（50M vs 0.6M 图），SFT 数据也不同——VILA 用 LLaVA-1.5 blend **加上** in-house FLAN 风格 18 数据集 blend **加上** 1M 文本 SFT（附录 A、Sec. 4.1）。Table 5 的增益不能全归因于预训练。
- "consistently outperforms"：POPE 上 VILA-7B 85.5、VILA-13B 84.2 均低于 LLaVA-1.5 的 85.9（Table 5）。
- 深层对齐假说只有 3 点相关性，度量描述不一致，不足以支撑"机制"。
- Table 1 配置 (a) 实际是 **SFT 阶段**冻结 LLM，与"预训练冻结"的主张不是同一变量；真正对应主张的是 (b) vs (c)，其 0-shot 平均相同（66.8 vs 66.8）、4-shot 57.6 vs 68.8。
- 消融全部在单一 SFT blend、4 个基准、非标准评测上做，结论到 Table 5 设置的迁移靠的是最终模型整体结果，而非逐项复核。
- 世界知识（n=4）、多图推理、CoT 都是挑选的定性样例。
- 无多 seed。

## 6. 可迁移手法
1. 研究"数据为什么有用"时，构造一个**只拆掉一个属性**的数据变体（MMC4 → MMC4-pairs 去结构；再打乱顺序），把结构与分布两个混杂分开（Table 3, Sec. 4.4）。—— 直接读到
2. 新能力不能伤旧能力：每个消融同时报主指标与"守恒指标"（MMLU），并给出原模型参照行（Table 3/4/6）。—— 直接读到
3. 把一个能力维度（ICL）拆成独立测量（0-shot vs 4-shot），使"看似无损"的选择暴露出隐藏代价（Table 1）。—— 直接读到
4. 用内部表示指标（逐层跨模态相似度）为行为差异提供机制解释（Fig. 3）。—— 直接读到（但需注意只是相关）
5. 跨底座复做关键消融（附录 Table 11）以说明结论不依赖特定 LLM。—— 直接读到
6. "设计空间研究"型论文的说服力来自"每个发现都能指导别人改做法"，而非 SOTA 数字本身。—— 我的归纳

## 7. 一句话
最关键一招是 MMC4 vs MMC4-pairs 的"只去掉交错结构"对照，加上每个消融都并列 MMLU 守恒指标和 0/4-shot 分离，把三条预训练经验做成了可被他人直接复用的受控结论；SOTA 表中的"head-to-head"则混入了 SFT 数据差异。
