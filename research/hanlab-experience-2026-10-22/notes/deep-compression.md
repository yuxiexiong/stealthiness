# Deep Compression: Compressing DNNs with Pruning, Trained Quantization and Huffman Coding

全文已读完（正文 + 附录 A Table 8/9）。pdftotext 导致 Fig 6–11 只剩图注，曲线数值无法读取。

## 0. 元信息
ICLR 2016（Best Paper）。Han, Mao, Dally。类型：算法（压缩流水线）+ 少量硬件实测。

## 1. 核心主张
- 主主张：剪枝 + 训练式量化（权重共享）+ Huffman 三段流水线可把存储压缩 35×–49× 而不损精度（AlexNet 240MB→6.9MB，VGG-16 552MB→11.3MB）。
- 次主张：(a) 剪枝与量化"互不干扰"、组合后压缩率远高于单独使用（Sec 1 "main insight"，Sec 6.1）；(b) 线性初始化 centroid 最好（Sec 3.2/6.2）；(c) batch=1 时剪枝层在 CPU/GPU/移动 GPU 上 3–4× 加速、3–7× 能效（Sec 6.3）；(d) 模型足够小可进片上 SRAM（推论，非实测）。

## 2. 动机是怎么立住的
- 痛点：App Store 100MB 下载限制（Sec 1，产品约束而非实验）；能耗由访存主导——45nm 下 32bit DRAM 访问 640pJ vs SRAM 5pJ vs FP add 0.9pJ，并算出 1G 连接 20fps 光 DRAM 就 12.8W（Sec 1）。这是**引用的硬件常数 + 简单算术**，不是本文 profiling。
- 关键观察：剪枝后权重呈双峰分布（Fig 4 左）；量化后权重索引与稀疏索引差的分布有偏（Fig 5）→ 直接引出 Huffman 的合理性（省 20–30%）。观察→方法的链条在 Huffman 一环闭合得最紧。
- "SRAM 能装下所以省能耗"这一链在本文**没有闭合**：量化模型没有实测（Sec 6.3 明说 BLAS 不支持间接查表），只留给 EIE（Sec 8）。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 35–49× 无损压缩 | 4 网络 | Table 1 | 未压缩 Caffe 参考模型 | Top-1/5 error、MB |
| 每阶段贡献 | 逐层 P / P+Q / P+Q+H 拆解 | Table 2–5 | 自身阶段累加 | 权重比例、bit 数、压缩率 |
| 剪枝与量化互补 | 精度-压缩率曲线 | Fig 6 | 仅剪枝、仅量化、SVD | 精度 vs 压缩率 |
| 剪枝不伤量化 | 每层位宽扫描 | Fig 7 | 未剪枝 + 量化 | Top-1/5 |
| 线性初始化最优 | 2–8 bit 扫描 | Fig 8 | Forgy、density | Top-1/5 |
| 位宽激进度 | 8/5、8/4、4/2 | Table 6 | FP32 | error 增量 |
| 加速/能效 | FC6–8 层 SpMV vs GEMV | Fig 9/10, Table 8/9 | dense cuBLAS/MKL | 时间 µs、W、能量 |
| 优于其他压缩 | 文献数字对比 | Table 7 | Fastfood、SVD、Collins&Kohli、Pruning | error、MB |
| 进 SRAM 省能耗 | 「未直接验证」 | Sec 9 | — | — |

## 4. 实验设计手法
- 基线：Table 7 列出 Fastfood、SVD、Collins&Kohli、自家 Pruning；数字多为**引自原文**而非同设置重跑。Fig 6 的"仅剪枝/仅量化"是自己跑的朴素基线，比较公平。
- 规模/广度：LeNet-300-100、LeNet-5（MNIST）、AlexNet、VGG-16（ImageNet）；只有分类；3 种硬件（Titan X、i7-5930K、Tegra K1）。
- 效率测量：**真实硬件、层级（kernel 级）**，只测 FC 层、只测剪枝（不含量化）；batch=1 为主，附录给 batch=64，并**主动承认 batch=64 时稀疏反而更慢**（Table 8：Titan X FC6 dense 19.8µs vs sparse 94.6µs）。测量细节交代充分：4096 样本平均、排除 cudaMalloc/Memcpy、功率工具（pcm-power、nvidia-smi、外接功率计 + 15%/85%/15% 折算假设）。
- 精度–效率联合：Fig 6 为精度-压缩率曲线（Pareto 风格）。Table 6 是同一模型不同位宽的折中。
- 消融：逐阶段累加（Table 2–5 的 P → P+Q → P+Q+H 三列），这是本文的"roadmap"表；初始化方法消融（Fig 8）；CONV vs FC 位宽敏感性（Fig 7）。
- 分析性实验：权重分布与 centroid 微调前后可视化（Fig 4 右）；存储构成拆分（Fig 11，权重/索引/码本）；对 MV vs MM 访存/计算比的解析（Sec 6.3）。
- 泛化：跨 4 网络、跨 3 硬件；无跨任务。
- 统计：无方差、无多 seed。0.01% 级别的差异（Table 6）被当作结论，在单次运行下不具统计意义。
- 系统贡献：无开源声明；指向后续 EIE 硬件。
- 局限性：Sec 8 明说量化模型未被实测；附录承认 batch 情况下剪枝无优势。

## 5. 对抗式审读
- 最薄弱环节：标题主张的"能效"来自 SRAM vs DRAM 常数推算，但实测只覆盖剪枝后 FC 层的 SpMV；量化 + Huffman 的模型从未在硬件上跑过（Sec 6.3、Sec 8）。"35× 压缩"与"3–7× 能效"是两套不同对象的数字，读者容易合并。
- 不公平处：Table 7 他人数字直接引用、训练预算不同（本文多轮再训练）；"无精度损失"基于单次运行，VGG-16 压缩后反而 error 下降 0.33%，提示可能存在再训练带来的正则化/额外训练量而非压缩本身。
- 只挑有利设置：能效实验只挑 FC 层（模型尺寸 >90%），而 CONV 才是计算主导；batch=1 设定是主动选择（论证为实时应用），但 batch=64 结果被放到附录。
- 超出支持：Fig 9 图注"3× on CPU"等是层平均；Tegra K1 功率是外推假设折算。
- 能耗表细看：Titan X 上 sparse batch=1 功率 181W > dense 157W（Table 9），能效提升完全来自时间缩短。

## 6. 可迁移手法
1. 多阶段方法用"逐阶段累加 + 逐层"大表（Table 2–5）同时证明每一步都有贡献与贡献在哪一层。—直接读到
2. 组合方法要证"互补"，就画单独 A、单独 B、A+B 在同一精度-压缩率坐标里的曲线（Fig 6），并画"A 是否伤害 B"的对照（Fig 7）。—直接读到
3. 当某组件（Huffman）的动机来自数据统计性质，就把该统计分布直接画出来（Fig 5），让方法看起来是被数据"逼出来"的。—我的归纳
4. 效率主张用三类硬件实测，并把不利场景（batch=64）放附录完整给出，换取可信度。—直接读到（附录 Table 8）
5. 用引用的物理常数（pJ/access）做数量级论证，把"存储压缩"翻译成"能耗"叙事。—直接读到（Sec 1），但要意识到这是推算。

## 7. 一句话
逐阶段、逐层的压缩率拆解表（Table 2–5）配上"单独 vs 组合"曲线（Fig 6），让 35–49× 这个大数字每一份都有账可查。
