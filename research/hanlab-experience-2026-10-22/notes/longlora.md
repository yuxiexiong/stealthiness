# LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models

## 0. 元信息
- ICLR 2024（arXiv 2309.12307v3）；Chen, Qian, Tang, Lai, Liu, Han, Jia（CUHK/MIT/NVIDIA）
- 类型：算法（训练期稀疏注意力 S²-Attn + 改进 LoRA 用于上下文扩展）
- 全文读完（正文 + 附录 A–B.6，Table 7–15，Fig.6–9）。Fig.1/4/5 的数值部分可从文本读出。

## 1. 核心主张
- 主：上下文扩展的**微调**可以只用局部分组注意力 + 半数头移位（S²-Attn），推理时仍用全注意力，困惑度接近全注意力微调；配合"可训练 embedding 与 norm"的 LoRA+，在单台 8×A100 上把 Llama2-7B 扩到 100K、70B 扩到 32K。
- 次 1：普通 LoRA 做上下文扩展有明显 PPL 差距，提高 rank 无用；打开 embedding/norm（<2% 参数）即可弥合（Table 2）。
- 次 2：S²-Attn 训练出的模型可用全注意力推理，而其他稀疏模式（dilated/block/stride）不行（Table 6）。
- 次 3：发布 LongAlpaca-12k，SFT 后在 LongBench/LEval 上与同规模长上下文模型相当或更好。

## 2. 动机是怎么立住的
- 痛点：Introduction 列出 PI 用 32–128 A100、FOT 用 128 TPU 的成本；Fig.1 三联图（PPL、显存、训练时长 vs 上下文长度，Full FT / LoRA / LongLoRA）同时展示"LoRA 精度差"和"LoRA 仍慢"两个问题。
- 关键观察：Table 2——LoRA rank 8→256 PPL 11.44→11.98 完全不降，而 +Norm 10.49、+Embed 8.29、+两者 8.12 vs Full FT 8.08。这是一个干净的"增加容量无用、换位置有用"的对照。
- S²-Attn 的动机是 pilot study（Table 1）：短注意力不移位时 PPL 随目标长度恶化（8.29→9.47），加移位回到 8.04–8.08，与全注意力 8.02–8.05 持平。逐步推出设计。
- 因果链闭合较好，但 S²-Attn 的论证全部基于 PPL。

## 3. 主张 → 证据映射表
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 不移位的短注意力不够、移位可行 | pilot | Table 1（全参微调） | Full attn | PG19 val PPL |
| LoRA 需开 embed/norm | 层类型 × rank | Table 2 | Full FT | PPL |
| 精度接近全注意力 | 7B/13B，8K–32K | Table 3（proof-pile）, Table 14（PG19 test） | Full attn+Full FT（仅 8K 行有） | PPL |
| 极限长度 | 7B@100K/13B@64K/70B@32K | Table 4, Table 15 | — | PPL |
| 检索能力 | LongChat topic retrieval；passkey | Table 5, Fig.4 | LongChat-13B 等开源模型；Llama2-7B | Acc |
| 训练效率 | 时长/显存 | Fig.1, Table 12, Table 13（无 FA2） | Full FT, LoRA | 小时、GB |
| FLOPs 节省 | torchprofile | Table 11 | — | TFLOPs（理论计数） |
| 移位方式/模式的必要性 | 跨头/跨层/仅 P1/P2；dilate/block/stride | Table 6 | — | 稀疏测/全注意力测两种 PPL |
| 组大小 1/4 | 1/2–1/8 | Table 7 | Full | PPL |
| 因果泄漏担忧 | 3 种变体 | Table 8, Fig.6 | — | PPL |
| 收敛速度 | PPL vs 步数 | Fig.5 | Full FT | PPL |
| 下游长任务 | LongBench、LEval（SFT 后） | Table 9, 10 | GPT-3.5、Llama2-chat、LongChat、Vicuna | 分数/胜率 |

## 4. 实验设计手法
- 基线：全参微调 + 全注意力（上界）、普通 LoRA（朴素基线）、多个替代稀疏模式（同计算量，Fig.7 注明）。下游比较的是**别人训练的开源模型**（不同数据、不同成本），非受控。没有与 Landmark/FOT 等长上下文微调方法在同设置下比较。
- 规模：Llama2 7B/13B/70B；上下文 8K–100K。只有一个模型族。
- 效率：训练时长与峰值显存实测（8×A100、DeepSpeed、FA2）；FLOPs 用 torchprofile 统计；另给"无 FA2"对照（Table 13），显示 S²-Attn 在无 FA2 时节省更大——**说明在 FA2 下显存几乎无差**（Table 12：65536 时 LoRA 71.1GB vs LongLoRA 69.8GB），显存节省主要来自 LoRA 本身。
- 精度–效率：Fig.1 并列三张图，未做 Pareto。
- 消融：层类型、rank、组大小、移位变体、跨头/跨层、替代稀疏模式、微调步数，较全面。
- 分析性：Table 6 的"两种测试协议"（稀疏测 vs 全注意力测）是关键设计：证明 S²-Attn 训练的模型对全注意力"兼容"，而 block sparse 在稀疏测时好、全注意力测时差。
- 泛化：passkey 外推（Fig.4，扩展 PI 到 48K 后 33–45K 仍有 60–90%）。
- 统计：passkey 每长度 10 次随机 passkey；其余无方差。
- 系统：开源代码、模型、数据集、demo；"两行代码"实现（Algorithm 1）。
- 局限：仅提到 PI 在短上下文上的 PPL 退化是已知问题（Sec.4.2）。

## 5. 对抗式审读
- 最薄弱：几乎所有主证据是 PPL；PPL 对长程依赖不敏感（同组其他论文自己也这样说，如 Quest Sec.4.2.2）。真正长依赖的证据只有 passkey（合成）、topic retrieval，以及 SFT 后的 LongBench（混入了 SFT 数据效应）。
- Table 3 只有 8192 行有全注意力全参基线，16K/32K 没有；而 Table 14（PG19 test）在 7B@32K 上 LoRA+ 版 7.22 vs S²-only（全参）7.03，差距不小，"comparable"在长端被放宽。
- Fig.1 标题称"比全参微调省 1.8× 显存"，这主要是 LoRA 的贡献而非 S²-Attn；S²-Attn 的实际收益是训练时长（65536 时 52.4h vs 92.5h）。归因容易被误读。
- Table 9/10 的下游比较对手训练数据与预算不同，且 LEval 用 GPT-4 判胜率，不是受控实验。
- 单模型族（Llama2），RoPE + PI 设定。

## 6. 可迁移的手法
1. 用"加容量无效、换位置有效"的二维对照（rank 扫描 vs 打开哪些层）定位瓶颈。（Table 2；直接读到）
2. 用逐步 pilot study 推出设计：每加一个部件就在同一张小表里显示它修好了什么。（Table 1；直接读到）
3. 训练期近似、推理期精确的方法，必须设计"两种测试协议"并列（近似测 vs 精确测），证明训练出的模型不依赖近似。（Table 6；直接读到）
4. 主张"效率来自 X"时，给出去掉其他加速手段（如 FA2）的对照，看 X 的独立贡献。（Table 13；直接读到）
5. 用"单台 8×A100 上能训到的最大长度"作为可记忆的可行性指标。（Table 4；我的归纳）
6. 实现极简（两行代码）本身作为采纳度论据。（Algorithm 1；直接读到）

## 7. 一句话
Table 2 那行"rank 从 8 到 256 PPL 不降、打开 0.004% 的 norm 参数就降"的对照，加上"稀疏训练/全注意力测试"双协议，让一个简单的训练技巧显得有清晰的机制依据。
