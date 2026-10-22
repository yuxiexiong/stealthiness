# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing

全文已读完（正文 + 附录 A.1–A.3）。原始 layout 文本（raw/txt/hat.txt）被图中嵌入的数据表与图标路径严重污染；我另用 `pdftotext` 非 layout 模式从 raw/pdf/hat.pdf 生成临时文本（/tmp/g6txt/hat.txt，非 notes 目录）辅助阅读，表 2–8 数字由此读取。Fig 2/7/8/11 曲线不可读。

## 0. 元信息
ACL 2020。Wang, Wu, Liu, Cai, Zhu, Gan, Han。类型：算法（Transformer 的硬件感知 NAS）。

## 1. 核心主张
- 主主张：用权重共享 SuperTransformer + 硬件延迟预测器 + 进化搜索，为每种硬件找专用 SubTransformer；在 Raspberry Pi 上 WMT'14 En-De 比 Transformer-Big 快 3×、小 3.7×，比 Evolved Transformer 快 2.7×、小 3.6×，搜索成本低 12,041×，BLEU 不降（Abstract）。
- 次主张：(a) FLOPs 不反映延迟、不同硬件的延迟敏感维度相反（Fig 2）；(b) GPU 专用模型在 ARM 上不快，反之亦然（Table 1）；(c) 继承权重的子网是从头训练子网的良好代理（Table 5, Fig 11）；(d) 与量化、KD 正交（Table 7, Sec 4.2）。

## 2. 动机是怎么立住的
- **Fig 2（关键观察）**：在 Raspberry Pi / Intel CPU / Nvidia GPU 上分别扫描 hidden dim、embedding dim、层数，画延迟 vs FLOPs。结论：同 FLOPs 延迟差异大；Pi 对维度敏感，GPU 对维度几乎不敏感、只对层数敏感。这是**本文自己的 profiling**，把"FLOPs 不是好代理"和"不同硬件需要不同架构"两个痛点同时量化。
- **Table 1**：2×2 交叉——HAT(GPU) 在 GPU 147ms / ARM 6491ms；HAT(ARM) 在 GPU 184ms / ARM 6042ms，BLEU 28.10 vs 28.15。对角线最优。
- 因果链：Fig 2 观察 → 设计空间加入异构层（各层维度可不同）+ 硬件反馈 → Fig 12 中 Pi 模型深而窄、GPU 模型浅而宽，正好回扣 Fig 2。**观察-方法-结果三点闭环**，是本组中最完整的。
- 另一设计动机（arbitrary encoder-decoder attention 打破信息瓶颈）只有直觉论证 + 搜索结果统计（10% 解码层接 3 个编码层、40% 接 2 个），**没有 BLEU 消融**。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| FLOPs 不代表延迟 / 硬件敏感维度不同 | 三硬件维度扫描 | Fig 2 | — | 延迟 vs FLOPs |
| 需要硬件专用 | 2 模型 × 2 硬件 | Table 1, Fig 12 | 彼此 | 延迟、BLEU |
| 各延迟下优于基线 | 4 任务 × 3 硬件曲线 | Fig 7/8, Table 8 | Transformer 层数缩放、维度缩放 | BLEU vs 延迟 |
| 优于 Evolved Transformer 且便宜 | Pi 延迟 + 成本 | Table 2, Fig 10 | Transformer、ET | 延迟、参数、FLOPs、BLEU、GPU h、CO2、$ |
| 优于其他高效 Transformer | 同 BLEU 下比 Pi 延迟 | Table 3 | Transformer、Levenshtein、ET、Lite Transformer | 延迟、BLEU |
| 不是"越大越好" | 与空间内最大子网比 | Table 4 | Largest SubTransformer | 延迟、参数、BLEU |
| 进化 > 随机搜索 | 搜索曲线 | Fig 9 | Random search | Val loss |
| 继承权重是好代理 | 继承 vs 从头 | Table 5, Fig 11 | 从头训练 | Val loss、BLEU、排序 |
| 继承 + 微调可省 4× | 10K vs 40K 步 | Table 6 | 从头 | BLEU |
| 量化友好 | k-means 8/4bit | Table 7 | Transformer FP32、Transformer-Base 4bit | BLEU、MB |
| KD 正交 | 学生 25.8→26.1 | Sec 4.2 文字 | 无 KD | BLEU |
| arbitrary En-De attention 有益 | 「未直接验证」（只有使用比例统计） | Sec 4.2 | — | — |
| 延迟预测器准确 | 测试集 | Fig 6 | 实测 | RMSE 0.1s（Pi） |

## 4. 实验设计手法
- 基线：Transformer（Base/Big）及其**两种朴素缩放**（层数缩放、维度缩放，Fig 7 的黄/蓝虚线）——这是"统一缩放"朴素基线，揭示 GPU 上维度缩放几乎不降延迟；Evolved Transformer（同类 NAS）、Levenshtein（非自回归）、Lite Transformer（自家 FLOPs 驱动设计）。**基线 Transformer 用与搜索出子网相同的训练设置重训**（Sec 3.3 "for fair comparisons"）。Table 3 把所有模型缩放到与 Levenshtein 相近的 BLEU 再比延迟。
- 规模/广度：4 个翻译任务（WMT'14 En-De/En-Fr、WMT'19 En-De、IWSLT'14 De-En）；3 硬件（Raspberry Pi 4 ARM A72、Xeon E5-2640、TITAN Xp）。
- 效率测量：**实测**，测量协议写清：源句与目标句同长度（WMT 30、IWSLT 23 token，取测试集平均输出长度），每模型测 300 次，去掉最快最慢各 10% 取平均（Sec 3.2）。预测器只用于搜索，报告用实测（Sec 2.3）。
- 精度–效率联合：每个任务 × 硬件一条 BLEU-延迟曲线（Fig 7/8），附录 Table 8 给出每个点的具体延迟、BLEU 与 SacreBLEU。
- 消融：搜索算法（进化 vs 随机，Fig 9）；最大子网 vs 搜索结果（Table 4）；代理质量（Table 5）；继承微调（Table 6）。设计空间两个新元素（arbitrary attention、异构层）**都没有单独 BLEU 消融**。
- 分析性实验：Fig 2 profiling；Fig 12 架构可视化；arbitrary attention 使用比例统计。
- 泛化：跨 4 任务、3 硬件；与量化/KD 组合。
- 统计：无方差、无多 seed；BLEU 差 0.1–0.3 级别。SacreBLEU 一并报告（Table 8）以回应 BLEU 计算口径争议。
- 系统贡献：开源。
- 局限性：无专门段落；提到 Q/K/V 维度未纳入搜索空间，留作未来工作（Sec 2.1）。

## 5. 对抗式审读
- 最薄弱环节：两个设计空间创新没有被单独消融，"打破信息瓶颈"的说法只靠搜索结果中的使用频率，不能证明其对 BLEU 有贡献。
- 成本比较：ET 的 626,155 lbs CO2 引自 Strubell 对 ET 的估算；HAT 的 52 lbs 包含 SuperTransformer + 搜索出模型的训练，但 ET 搜出的模型本身也有更大的训练开销——比较口径基本一致但依赖外部估算。
- Table 3 中"同 BLEU"是把各模型缩放至近似 BLEU（25.20–25.92），HAT BLEU 实际最高，对其他模型缩放方式未详述。
- Table 1 的"专用必要性"只有 2×2，且 GPU 上差距 147 vs 184ms、ARM 上 6042 vs 6491ms（约 7%），远不如摘要中的 3× 那么醒目——3× 是对 Transformer-Big，而非对另一硬件的专用模型。
- Encoder 层数固定为 6（理由：编码器占延迟 <5%），这一测量数字未给图表。

## 6. 可迁移手法
1. 先做跨硬件维度扫描 profiling（Fig 2），一张图同时证明"FLOPs 不是好代理"和"不同硬件敏感维度相反"，并让最终搜出的架构差异（Fig 12）回扣这张图。—直接读到
2. 用两种朴素统一缩放（层数缩放、维度缩放）作为基线曲线，揭示朴素缩放在某些硬件上失效（GPU 上维度缩放曲线几乎垂直）（Fig 7）。—直接读到
3. 延迟测量协议写死：固定输入/输出长度、300 次、去头尾 10%（Sec 3.2）。—直接读到
4. 代理有效性用"继承权重 vs 从头训练"的数值表 + 排序一致性证明（Table 5, Fig 11）。—直接读到
5. 把基线在与自己相同的训练设置下重训，并在正文明说（Sec 3.3）。—直接读到
6. 用"空间内最大子网"作为对照，证明搜索不是简单地"选最大的"（Table 4）。—直接读到

## 7. 一句话
Fig 2 的三硬件维度-延迟 profiling 把"FLOPs 骗人"和"硬件偏好相反"一次量化，最终搜出的"Pi 深窄 / GPU 浅宽"架构正好回扣它——观察、方法、结果闭成一个环。
