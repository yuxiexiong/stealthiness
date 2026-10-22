# Once-for-All: Train One Network and Specialize it for Efficient Deployment

全文已读完（正文 + 附录 A/B）。Fig 9–14 的曲线在文本中严重交错，只能读出零散数据标签；Table 1 可基本重建。

## 0. 元信息
ICLR 2020。Cai, Gan, Wang, Zhang, Han。类型：算法（超网训练 + 免训练特化）+ 多硬件实测。

## 1. 核心主张
- 主主张：训练一个支持弹性深度/宽度/核大小/分辨率（>10^19 子网）的 OFA 网络，再按部署场景直接挑子网，不需重训，可把 N 个场景的设计成本从 O(N) 降到 O(1)，且子网精度达到或超过独立训练与 SOTA NAS（Abstract、Fig 1）。
- 次主张：(a) Progressive Shrinking（PS，先大后小、逐维放开 + KD + 通道按 L1 排序）是关键，比不用 PS 的子网高 2.5–3.7%（Fig 7）；(b) 移动设定 <600M MACs 下 80.0% top-1（Fig 2/9）；(c) 不同硬件的最优架构不同、OFA 可自动发现（Fig 14）；(d) 在 FPGA 上提高算术强度与利用率（Fig 12/13）。

## 2. 动机是怎么立住的
- 痛点：设备多样性（Note10 vs S6、230 亿 IoT 设备，Sec 1）+ 每场景重训成本线性增长 + CO2（引 Strubell）。Fig 1 中间图把"设计成本 vs 场景数"画成 O(N) vs O(1)——**叙事图，非测量**。
- 训练难点的动机：朴素随机采样子网训练"会显著掉点"（Sec 3.3），证据就是 Fig 7 的 w/o PS 对照，闭合。
- "Slimmable 只有 4 个配置"（10^19 vs 4，Sec 2）——用数量级差异立住新颖性。
- 因果链：多场景成本痛点 → 解耦训练与搜索 → PS 解决子网互相干扰 → Fig 7 + Table 1 验证。闭合良好。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| PS 必要 | 8 个典型子网 w/ vs w/o PS | Fig 7, Table 1 (72.4 vs 76.0) | OFA w/o PS | Top-1 |
| 成本 O(1) 且精度更高 | Pixel1 上 N=40 | Table 1, Fig 8 | MBv2、MBv2 #1200、NASNet、DARTS、MnasNet、FBNet、ProxylessNAS、SPOS、AutoSlim、MBv3-L | Top-1、MACs、延迟、GPU h、CO2、$ |
| 移动设定 SOTA | MACs / Pixel1 延迟扫描 | Fig 2, Fig 9 | EfficientNet、MBv3 | Top-1 vs MACs/ms |
| 跨 6 款手机优于 MBv3 | 每台手机一条曲线 | Fig 10 | MBv3-Large/Small | Top-1 vs 延迟 |
| 跨 CPU/GPU/mGPU/FPGA | 6 平台 | Fig 11 | MBv2 width 系列、MnasNet、Slimmable | Top-1 vs 延迟 |
| FPGA 上更高算术强度 | ZU9EG/ZU3EG 实测 | Fig 12/13 | MBv2、MnasNet | OPs/Byte、GOPS/s |
| 不同硬件最优架构不同 | 搜出的架构 | Fig 14 | FPGA/CPU/GPU 之间 | 定性 |
| 权重本身有贡献（非仅架构） | 搜出的架构从头训练 | Fig 9（"OFA - Train from scratch"曲线），Sec 4.2 文字 | OFA 继承权重 | Top-1 |
| 精度预测器可靠 | RMSE 0.21%；RMSE vs 最终精度 | App A, Fig 15 | 不同 RMSE 的预测器 | RMSE、选出子网精度 |

## 4. 实验设计手法
- 基线：极广——手工（MBv1/v2/v3）、NAS（NASNet、DARTS、MnasNet、FBNet、ProxylessNAS、SPOS、AutoSlim、EfficientNet）、动态网络（Slimmable）。**特别设计的公平基线**：MobileNetV2 #1200——把 MBv2 训练 1200 GPU h，与 OFA 总训练成本对齐（73.5% vs OFA 76.0%），回应"OFA 只是训练更久"的质疑。OFA w/o PS 作消融基线。NAS 方法的精度多为原文报告值。
- 规模/广度：ImageNet；**12 种硬件**（S7 Edge、Note8、Note10、Pixel1、Pixel2、LG G8、1080Ti、V100、Jetson TX2、Xeon CPU、ZU9EG、ZU3EG FPGA）；两个搜索空间（MBv3 空间与 ProxylessNAS 空间）。
- 效率测量：**全部实测**，并逐平台交代框架和 batch：GPU batch 64 PyTorch+cuDNN；CPU batch 1 MKL-DNN；手机 TF-Lite batch 1；TX2 batch 16；FPGA Vitis AI 8bit batch 1（Sec 4.2）。搜索阶段用 LUT 预测延迟。
- 精度–效率联合：每个平台一条 accuracy–latency 曲线（Fig 10/11），而不是单点；Table 1 同时报精度、MACs、延迟与成本。
- 消融：PS（Fig 7）；微调轮数 0/25/75（Table 1, "#25/#75"）；继承权重 vs 从头训练（Fig 9）；预测器 RMSE 对最终结果的影响（Fig 15）。PS 内部各组件（KD、通道排序、核变换矩阵、阶段顺序）**没有单独消融**。
- 分析性实验：roofline（Fig 13）、算术强度/利用率（Fig 12）、架构可视化（Fig 14，FPGA 只选 3×3 因 depthwise 引擎对 3×3 流水线匹配、大核导致 BRAM 溢出）。
- 泛化：跨 12 硬件、2 搜索空间；无跨任务（只在 Abstract 提到 LPCVC 检测赛道获胜）。
- 统计：无方差。
- 系统贡献：开源 + 50 个预训练模型；LPCVC 竞赛获胜作为外部背书。
- 局限性：无专门段落。

## 5. 对抗式审读
- 最薄弱环节：OFA 子网继承了大网络的 KD 与 ~600 epoch 的训练（180 + 125+25+125+25+125，App B），而对比的 MBv3/EfficientNet 是各自原始训练配方。MBv2 #1200 对齐了 GPU 小时，但没有对齐 KD——"OFA 子网 > 独立训练"中 KD 的贡献未被分离。
- "PS 让所有子网达到独立训练水平"——Fig 7 只比较了 w/ vs w/o PS 的 8 个均匀配置子网，未给出这些子网独立训练的精度（只在 Fig 9 给出搜索出的架构 from-scratch 曲线）。
- 成本计算：Table 1 中各 NAS 的成本是按其原文报告 × N 的估算；OFA 的 40 GPU h 是数据收集，未计 LUT 构建与每平台延迟测量。
- 80.0% 结果是 OFALarge #75（再微调 75 epoch），成本列仍写 1200+75N，与"无需训练直接部署"的叙事有张力。
- 多数曲线只到 Pixel1 等手机的 LUT 延迟，搜出的模型最终实测与 LUT 预测的偏差未报告。

## 6. 可迁移手法
1. 当方法需要大量一次性训练时，构造"等算力基线"（MobileNetV2 #1200），把基线训练到同样 GPU 小时，直接堵住"只是训练更久"的攻击（Table 1）。—直接读到
2. 成本表按"N 个部署场景"写成 a + bN 的形式并给出 N=40 的总量，换算成 CO2 与 AWS 美元（Table 1, Fig 8）。—直接读到
3. 宣称"硬件通用"时每个平台画一条完整的精度-延迟曲线并交代 batch/框架，平台跨手机/GPU/CPU/FPGA（Fig 10/11, Sec 4.2）。—直接读到
4. 对于预测器驱动的搜索，展示"预测器误差 vs 最终选出模型精度"的曲线，证明预测器质量与结果相关（App A, Fig 15）。—直接读到
5. 把"继承的权重"与"架构"分开：把搜出的架构从头训练，若更差，则权重共享本身有贡献（Fig 9 文字）。—直接读到
6. 用硬件 roofline/算术强度解释搜出架构的差异（FPGA 只用 3×3），把 NAS 结果转成硬件设计原则（Fig 12–14）。—直接读到

## 7. 一句话
等算力基线（MobileNetV2 #1200）+ 12 种真实硬件上的整条精度-延迟曲线 + a+bN 成本表，三者合起来让"训练一次、处处特化"既在精度上又在成本上站住。
