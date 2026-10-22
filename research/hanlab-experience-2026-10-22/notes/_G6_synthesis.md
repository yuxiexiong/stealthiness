# G6 组内综合：经典压缩、剪枝、量化搜索、NAS、高效网络

覆盖 13 篇：deep-compression、pruning、dgc、haq、amc、apq、ofa、proxylessnas、hat、lite-transformer、tsm、efficientvit、netaug。
全部读完全文（含附录）。文本质量问题：haq 的 Table 3–6 与图交错，数字是我重建的；hat、tsm、efficientvit、netaug 的 raw/txt 双栏或嵌入图数据损坏严重，我另用 pdftotext 非 layout 模式从 raw/pdf 生成临时文本读取（/tmp/g6txt/，不在 notes 目录）；多数论文的曲线图只剩标签，无法读值。

档位说明：每条套路下列出的出处是**直接读到**；"为什么有效"一栏是**我的归纳**。

---

## A. 跨论文共同套路（出现 ≥2 篇）

### A1. "模型 × 硬件"交叉矩阵，证明需要按硬件专门化（3 篇 + 1 篇定性）
- HAQ Table 1：3 种量化策略 × 3 个硬件（BitFusion / BISMO edge / BISMO cloud）的延迟，对角线最优。
- ProxylessNAS Table 4：GPU/CPU/mobile 三个模型 × 三个硬件的延迟（GPU 模型在 CPU 上 204.9ms，CPU 模型 138.7ms）。
- HAT Table 1：2×2（GPU 模型 vs ARM 模型，各自在 GPU/ARM 上测）。
- OFA Fig 14（定性）：FPGA/CPU/GPU 搜出的架构不同。
- 为什么有效：用极少的数字把"专门化"从口号变成可验证的事实，且对角线结构一眼可读。

### A2. 先用 profiling 或实测证明"朴素做法失败/代理指标骗人"，再提出方法（6 篇）
- TSM Fig 2a/2b：全通道 shift 在三硬件上延迟 +13.7%/+12.4%，精度 −2.6%。
- HAT Fig 2：三硬件维度扫描，同 FLOPs 延迟差异大、硬件敏感维度相反。
- Lite Transformer Fig 2：计算分解，FFN 吃掉大部分 Mult-Adds。
- EfficientViT Fig 4：手机 CPU 上 softmax vs ReLU 线性注意力延迟 3.3–4.5×；Fig 3 线性注意力分布平坦。
- NetAug Fig 1 左：正则对 ResNet50 有益、对 MbV2-Tiny 有害。
- DGC Fig 3a / Table 2：Gradient Dropping 在 CIFAR 上掉 1%。
- 为什么有效：方法看起来是被观察"逼出来的"，读者先接受问题存在，再接受解法。

### A3. 搜索时用代理，报告时用实测，并单独验证代理（5 篇）
- 延迟代理：ProxylessNAS Fig 5 + App B（LUT 预测 vs 实测，RMSE 0.75ms，"搜索用模型、报告用实测"）；HAT Fig 6 + Sec 2.3（RMSE 0.1s，"只在搜索中用预测值"）；OFA Sec 3.4（LUT）；APQ Sec 3.3（LUT，但**未给出校准数据**）。
- 精度代理：AMC Table 2（不微调精度与微调后精度排序一致）；HAT Table 5 + Fig 11（继承权重 vs 从头训练）；OFA App A + Fig 15（预测器 RMSE 0.21%，RMSE 越低选出的子网越好）；APQ Fig 7（pairwise accuracy）。
- 为什么有效：搜索类论文最容易被攻击的点就是代理不可靠，提前给校准图可以堵住。

### A4. 统一缩放/朴素压缩作为基线（6 篇）
- pruning Table 6：Naive Cut（直接缩小层宽）4.4×、top-1 error 47.18%。
- AMC Table 2/3/4：uniform/shallow/deep 手工策略；MobileNet 0.75 width multiplier。
- ProxylessNAS Fig 4：MobileNetV2 各 width multiplier 曲线（同 74.6% 时 143ms vs 78ms）。
- HAT Fig 7：Transformer 层数缩放与维度缩放两条曲线（GPU 上维度缩放几乎不降延迟）。
- OFA Fig 11：MobileNetV2 0.35–1.0 宽度系列、Slimmable。
- HAQ Table 3–5：PACT 固定位宽。
- 为什么有效：回答"你的方法不就是把模型变小吗"，证明"怎么变小"比"变小"本身重要。

### A5. 等资源对比：同延迟/同 FLOPs/同大小比精度，或同精度比效率（8 篇）
- HAQ Table 3–6（延迟对齐到 0.01ms 级）、AMC Table 3、APQ Table 2/Fig 5/6、Lite Transformer Table 1/2（三个计算档）、HAT Table 3（缩放到同 BLEU 比延迟）、EfficientViT Table 3/4（"Models with similar mIoU are grouped"）、TSM Table 2/5、ProxylessNAS Fig 4。
- 为什么有效：避免"更准但更贵"或"更快但更差"这种不可比的结论。

### A6. 逐步累加的"roadmap"展示每个组件的贡献（4 篇）
- Deep Compression Table 2–5：逐层的 P → P+Q → P+Q+H 三列。
- DGC Fig 3a/Fig 5：GD → +动量修正(+本地裁剪) → 完整 DGC 的学习曲线。
- APQ Fig 6 左：ProxylessNAS → +AMC → +HAQ 的顺序流水线，再对比联合搜索。
- pruning Fig 5：L1/L2 × 有无再训练 × 迭代的五条曲线。

### A7. 等算力或"训练更久"的基线，回应"只是多算了"（3 篇）
- OFA Table 1：MobileNetV2 #1200（训练 1200 GPU h，与 OFA 对齐），73.5% vs OFA 76.0%。
- NetAug Fig 3 + Table 4：Baseline 300/600 epoch，NetAug 达到同精度少 75%/83% epoch。
- HAT Sec 3.3：基线 Transformer 用与搜索出子网完全相同的训练设置重训。

### A8. 成本账：GPU 小时 / CO2 / 美元，常拆成"固定 + 边际 × N"（4 篇）
- OFA Table 1 + Fig 8（N=40，CO2 少 16×–1300×）；APQ Table 2 + Fig 2（2400 + 0.5N GPU h）；HAT Table 2 + Fig 10（比 ET 少 12,041×）；Lite Transformer Table 3 + Fig 1b。
- 四篇都依赖 Strubell et al. 对 Evolved Transformer/MnasNet 等的外部估算。

### A9. 把学到的策略/架构画出来，并用硬件或 roofline 解释（6 篇）
- HAQ Fig 3–5（depthwise 是访存受限 → edge 上少给激活位）；AMC Fig 2/3（锯齿状瓶颈、3×3 剪得多 1×1 剪得少）；ProxylessNAS Fig 6（GPU 浅宽、CPU 深窄）；HAT Fig 12（Pi 深窄、GPU 浅宽，回扣 Fig 2）；OFA Fig 12–14（FPGA 只选 3×3，算术强度 +48%/43%）；pruning Fig 4（MNIST 稀疏模式对应图像中心）。
- 为什么有效：把黑箱搜索结果变成可复用的设计原则，提高"insight"分。

### A10. 在自己的设备上复现主要基线（"our impl."）并重测（3 篇）
- ProxylessNAS Table 2：MnasNet (our impl.) 79ms vs 原文 76ms。
- TSM Table 2/4：TSN、TRN 的 our impl.；Table 5 中 ECO/I3D 在同一 P100 上实测。
- EfficientViT Table 2–4：所有基线在同一 TensorRT fp16 管线下测延迟。

### A11. 测量协议写死（4 篇）
- Deep Compression App A（4096 样本平均、排除 cudaMalloc/Memcpy、功率工具与折算假设）；TSM Sec 3.3（1000 次 + 200 warm-up）、Sec 5.5（功率计减静态功耗）；HAT Sec 3.2（固定 30/23 token、300 次、去头尾各 10%）；EfficientViT Sec 3.1（框架/精度/batch，含数据传输时间）；OFA Sec 4.2（逐平台框架与 batch）。

### A12. "与已有压缩技术正交可叠加"（4 篇）
- Deep Compression（剪枝 + 量化互不干扰，Fig 7）；HAT Table 7（k-means 量化）+ KD；Lite Transformer Fig 5（剪枝 + 量化 18.2×）；NetAug Table 2（+KD）。

### A13. 跨任务泛化表（6 篇）
- DGC（图像/语言模型/语音）；AMC Table 5（检测）；TSM Table 7（视频检测）；NetAug Table 4/5、Fig 5（细粒度、检测、TinyTL）；EfficientViT（分类/分割/超分/SAM）；Lite Transformer（翻译/摘要/LM）。

### A14. 外部背书：竞赛名次（2 篇）
- OFA Abstract（LPCVC 第 3、4 届冠军）；TSM Abstract（Something-Something 排行榜第一）。

---

## B. 本组特有的套路（与 LLM 推理类论文相比）

1. **搜索算法自身的成本会计**：把 NAS/压缩策略搜索的 GPU 小时、CO2、美元列成表并与对手比（A8），这在纯算法论文里少见，是 NAS 子领域的"第二条赛道"。
2. **代理校准图是标配**（A3）：延迟 LUT 的 y=x 散点、精度预测器的 RMSE 或排序一致性。
3. **逐层策略可视化**（A9）：压缩率/位宽/核大小沿层画出来，再用 roofline/算术强度/参数量解释。
4. **压缩率逐阶段逐层拆账**（Deep Compression Table 2–5、pruning Table 2–5 的 Weights%/FLOP%/Act% 三列）。
5. **硬件覆盖异常广**：OFA 12 种、TSM 7 种、EfficientViT 4 种（含手机 CPU、Jetson、A100），每个平台一条精度-延迟曲线。
6. **用"自家前作"做基线并在极端设置下压过**：HAQ 用 Deep Compression（2bit 时 37.62 → 57.14），APQ 用 ProxylessNAS+AMC+HAQ 链，AMC 的人类专家基线来自 Han 的博士论文。

---

## C. 本组反复出现的薄弱点

1. **全部 13 篇都没有方差、多 seed 或置信区间**。不少"提升"在 0.1–0.5% 量级（DGC Table 2 的 +0.06%、ProxylessNAS CIFAR 2.08 vs 2.13、TSM 在线/离线 ±0.4%、NetAug +0.7%），而 DGC 自己的基线随 GPU 数就有 0.8% 波动。
2. **搜索方法从不与随机搜索比较**：HAQ、AMC、APQ、ProxylessNAS 都没有随机搜索或简单启发式对照；只有 HAT Fig 9 做了进化 vs 随机（而且只比验证损失）。因此"RL/进化/可微带来增益"与"任何自动搜索 + 好的搜索空间带来增益"无法区分。
3. **混淆变量没有分离**：APQ 的"联合"与"更大搜索空间"混在一起（作者在 Fig 5 图注自己承认）；ProxylessNAS 的"无代理"与"非重复 block 的大空间"混在一起；OFA 的子网精度混入了 KD 和约 600 epoch 训练；HAQ 的"RL"与"允许混合精度"捆在一起。
4. **效率停留在代理或模型层面**：pruning（只有参数量和 FLOP）、Lite Transformer（只有 Mult-Adds，后被同组 HAT 批评）、HAQ（模拟器）、APQ（LUT，无校准）、DGC（性能模型推算 64 节点加速）、NetAug 检测（MACs）。Deep Compression 量化后的模型从未在硬件上跑过（Sec 8 自承）。
5. **关键设计元素没有消融**：HAT 的 arbitrary encoder-decoder attention 和异构层、Lite Transformer LSRA 的分支比例和核、NetAug 的 r/s/α、OFA PS 的 KD/通道排序/核变换矩阵、HAQ 的全部组件、EfficientViT 的分支数和聚合核大小。
6. **摘要数字在正文中找不到或口径不一致**：AMC 摘要"VGG-16 好 2.7%"（Table 3 中对 CP 只好 0.3%），摘要 1.81×/1.43× 与引言 1.95×/1.53× 不同；EfficientViT 摘要"13.9×/6.2×"正文写"up to 8.8×/3.8×"（Orin）；APQ 摘要"能耗省 1.3×"对应的图不可读，而 Table 2 中 APQ 的能耗高于最强顺序基线（12.18 vs 8.84 mJ）。
7. **基线数字常直接引自原文而非同设置重跑**：Deep Compression/pruning 的对比表、DGC 的 TernGrad、AMC Table 3 的 FP/RNP/SPP、OFA Table 1 的 NAS 成本、HAT/Lite Transformer 的 ET 成本（都来自 Strubell 估算）。
8. **成本叙事选择对自己有利的口径**：APQ/OFA 强调边际成本，把一次性 2400/1200 GPU h 放进另一列；单场景时总成本反而更高。
9. **自家前作当基线**（见 B6）：基线强度由作者掌控，独立性不足。

---

## D. 给读者的提炼（我的归纳）
- 这组论文的说服力主要来自**广度 + 实测 + 对照结构**，而不是统计严谨性：硬件和任务覆盖面极宽、交叉矩阵和朴素基线设计巧妙，但每个点都是单次运行。
- 最"硬"的几篇（TSM、HAT、NetAug、EfficientViT）都做到了"观察 → 方法 → 回扣观察"的闭环：先用一张实测图建立问题，最后的结果或可视化又回到那张图。
- 最"软"的几篇（Lite Transformer、HAQ、APQ）问题集中在：效率只用代理、核心组件未消融、或关键变量混淆。
