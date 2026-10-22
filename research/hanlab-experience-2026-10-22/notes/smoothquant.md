# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

## 0. 元信息
- ICML 2023，arXiv 2211.10438v7
- 类型：算法-系统协同（PTQ 算法 + PyTorch/FasterTransformer INT8 实现）
- 全文已读（正文 + 附录 A）。Figure 4/7/10 为图像，文本中无数值，只能读图注；Figure 8/9 柱状图数字在 pdftotext 中错位，只引用正文复述值。

## 1. 核心主张
- 主主张：LLM 激活的异常值集中在固定通道、且各 token 间通道方差相似；通过离线、数学等价的逐通道放缩 s_j = max|X_j|^α / max|W_j|^(1−α)，把量化难度从激活"迁移"到权重，可让 **所有** GEMM/BMM 用 W8A8（甚至 per-tensor static）且精度无损。
- 次要主张：在 PyTorch 和 FasterTransformer 中实测最高 1.51×/1.56× 加速、近 2× 显存节省，530B 模型可装进单个 8×A100 节点。

## 2. 动机是怎么立住的
- **现象可视化（Figure 4）**：OPT-13B 某一层的激活/权重幅度 3D 图，读出三点：少数通道幅度 >70；通道内方差小；权重平坦。图注把观察写成编号列表，直接对应 Sec 3 的三条"pattern"。
- **把观察变成可检验的假设（Table 1）**：在 OPT-6.7B–175B 上模拟四种激活量化粒度：per-tensor 和 per-token 都崩（如 175B 32.3% / 31.7%），**per-channel 几乎无损**（71.4% vs FP16 71.6%）。这证明"问题出在通道维"，同时指出 per-channel 与 INT8 GEMM 不兼容（Eq. 2：缩放只能在外维）。由此动机闭合：既要通道维处理效果，又要硬件可行 → 等价变换把通道缩放折进权重。
- 有效量化级数的解析论证（Sec 3 第 2 点：通道 i 的有效级数 2^8·m_i/m，非异常通道只剩 2–3 级）。
- 链条：Fig.4 观察 → Table 1 per-channel 上界 → Eq.3/4 迁移 → Fig.10 α 扫描找甜点。闭合良好。

## 3. 主张 → 证据映射表
| 主张 | 证明实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 异常值在固定通道 | 幅度可视化 | Figure 4 | 平滑前/后 | 幅度 |
| 通道维是关键 | 模拟 per-tensor/token/channel | Table 1 | FP16 | 4 任务平均 acc |
| 175B 上 W8A8 无损（含 O3） | 7 个零样本任务 + WikiText | Table 3 | W8A8、ZeroQuant、LLM.int8()、Outlier Suppression | acc、PPL |
| 跨模型族（>100B） | OPT-175B/BLOOM-176B/GLM-130B | Table 4 | 同上 | 平均 acc |
| 跨尺寸 | OPT 各尺寸 | Figure 7（图，无数值） | LLM.int8() 等 | acc |
| 指令模型 | OPT-IML-30B | Table 5 | 同上 | LAMBADA、PPL |
| 新模型族 | LLaMA 7–65B；Llama-2、Falcon、Mistral、Mixtral | Table 6, 7 | **仅 FP16** | PPL |
| 加速（context） | PyTorch OPT-6.7/13/30B；FT OPT-13–175B | Fig. 8, 9 | FP16、LLM.int8() | 延迟、显存 |
| 加速（decode） | OPT-30B 1 GPU、175B 8 GPU，BS 1/16，Seq 512/1024 | Table 8 | FP16 | 延迟、显存 |
| 530B 单节点 | MT-NLG 530B | Table 9, 10 | FP16（16 GPU） | acc、延迟、显存 |
| 粒度–延迟权衡 | O1/O2/O3 | Table 11 | FP16、LLM.int8() | 延迟 |
| α 存在甜点 0.4–0.6 | OPT-175B LAMBADA α 扫描 | Figure 10（图） | — | acc |
| 与 GPTQ 正交、可到 W4A4 | **未直接验证**（附录 A 仅讨论） | App. A | — | — |

## 4. 实验设计手法
- **基线选择**：朴素 W8A8 per-tensor dynamic、ZeroQuant、LLM.int8()、Outlier Suppression；Table 2 把各基线的量化粒度（per-tensor/token/channel、static/dynamic）**并列写清**，让读者看到 SmoothQuant-O3 用的是最粗粒度仍胜出——"在更难的设定下赢"。ZeroQuant 额外尝试了其作者针对 GPT-NeoX 的变体（Table 3 注），属于给基线"补救机会"。未与 GPTQ 比（附录 A 解释设置不同、kernel 不可比）。
- **规模/广度**：OPT 6.7B–175B、BLOOM-176B、GLM-130B、MT-NLG 530B、OPT-IML、LLaMA、Llama-2、Falcon、Mistral、Mixtral；硬件仅 A100-80GB。
- **效率测量**：真实 GPU 端到端；两套后端（HF PyTorch 概念验证、FasterTransformer 生产级）；扫序列长度 128–1024、BS 1/4/16、单卡/多卡；context 与 decode 分开报告；显存与延迟同报。批大小 4、PyTorch 只测单卡（无模型并行）有交代。
- **精度–效率联合**：O1→O3 精度表（Table 3/4）与延迟表（Table 11）对应，构成粒度阶梯式的权衡呈现，但无 Pareto 图。
- **消融**：Table 11（量化粒度对延迟）；Figure 10（α）。**未消融**"只做平滑不做 static"之类组合的精度贡献分解；α 对不同模型手调（0.5/0.75/0.8/0.85/0.9），选择流程只说"在 Pile 子集上快速网格搜索"。
- **分析性实验**：Figure 4 前后对比；Table 1 是反事实上界（不可部署但模拟可行的 per-channel）。
- **泛化**：跨模型族与架构（MoE）；硬件单一。
- **统计**：无方差、无多 seed。
- **系统贡献**：开源；集成 FasterTransformer（NVIDIA 共同作者）。
- **局限性**：附录 A 讨论与 weight-only 方法的适用场景差异（bs=1 decode 时 GPTQ 可能更好）；正文无局限节。GLM-130B 更难量化，O3 掉 1%，作者如实写出。

## 5. 对抗式审读
- **最薄弱环节**：Table 6/7（LLaMA、Llama-2、Falcon、Mistral、Mixtral）只有 SmoothQuant vs FP16，且这些模型的 α 每个都不同（0.6–0.9），且 Table 7 换成 per-channel 权重 + per-token 激活（不是主文 O3）。"跨架构通用"的证据是在逐模型调参和更细粒度设定下取得的。
- 主要胜负来自 ≥100B 的 OPT/GLM，对比的基线（ZeroQuant、Outlier Suppression）本就不是为这个尺度设计的，"近似随机"的基线让差距看起来巨大；与真正强基线 LLM.int8() 比精度持平，胜在速度。
- 零样本任务只报平均，单任务内有 RTE/COPA 等小数据集，噪声可能 >1%，无 CI。
- 速度对比中 LLM.int8() 的实现是 bitsandbytes 版本（未说明是否最优），"几乎总是比 FP16 慢"部分是实现问题。
- 结论"适用于所有 LLM"超出支持：GLM-130B 需要额外 token 裁剪（top 2%）才行。

## 6. 可迁移的手法
1. 当观察是"问题集中在某个维度"时，先用**不可部署但可模拟的理想方案**（per-channel 激活量化）证明修复该维度就够，再设计可部署的等价替代（Table 1 → Eq. 3）。**直接读到**
2. 把所有基线的量化粒度并排列表，并让自己的方法在**最粗/最难部署友好**的设定下与基线细粒度设定比（Table 2, O3）。**直接读到**
3. 给出一组"效率等级"（O1–O3），同时报告每级的精度与延迟，让读者自己选点（Table 3/4 + Table 11）。**直接读到**
4. 系统实测用"概念验证后端 + 生产级后端"双实现，后者上的加速更难拿、更可信（Fig. 8 vs Fig. 9，作者强调 FT 已比 PyTorch 快 3×）。**直接读到**
5. 用"同等延迟下 GPU 数减半"作为大模型的效率叙事单位，比单纯 speedup 更贴近部署成本（Fig. 9, Table 10）。**直接读到**
6. 超参（迁移强度 α）做 U 形扫描并给机制解释（两端分别对应激活难/权重难），把超参变成可解释的权衡（Fig. 10）。**我的归纳**（论文只给了一个模型一个任务）

## 7. 一句话
Table 1 用一个"做得到就无损、但硬件做不了"的 per-channel 模拟上界，把问题定位到通道维，随后的等价迁移就成了顺理成章的唯一出路。
