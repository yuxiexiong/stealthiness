# Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation

## 0. 元信息
- CVPR 2022（arXiv 2205.01271v4；txt 中未印会议名，会议名为我的已知信息）
- 类型：算法（高效架构设计 + NAS），带多平台实测延迟
- 全文已读（无附录）。双栏交错；Figure 3/7/8/9 的数值在 txt 中可大致还原。

## 1. 核心主张
- 主主张：在低计算量区间，HRNet 式多分支架构的高分辨率分支是冗余的；单分支架构 + 融合反卷积头 + 大核卷积（LitePose）能以更少 MACs、显著更低的真实延迟取得更高 AP。
- 次要主张：(a) 大卷积核对姿态估计的增益远大于对分类（+13.0 AP vs +1.5% acc，同约 +25% 计算）；(b) 单分支并行度高，即使 MACs 更高延迟也相近（Fig 8）；(c) OFA 式 NAS 再带来 +2.2 AP；(d) CrowdPose 上搜得的结构可直接迁移到 COCO。

## 2. 动机是怎么立住的
- Figure 1：COCO mAP vs Snapdragon 855 延迟散点，LitePose vs EfficientHRNet vs Lightweight OpenPose——开篇就是帕累托图。
- **关键观察靠一个"渐进收缩"实验**（§3.2, Figure 2/3）：从 Scaled-HigherHRNet-W16 出发，逐步把高分辨率分支的块数减到 0（Baseline → Shrink1 → Shrink2 → Shrink3），网络逐渐变成单分支，CrowdPose AP 从 50.4 升到 53.0（12.5→9.15 GMACs）。这是"在两种架构之间架一座连续的桥，沿桥测性能"的反事实设计，比直接比较两种架构更有说服力。
- 观察 → 方法的因果链：高分辨率分支的"精炼"冗余 → 但多尺度融合仍有用 → 融合反卷积头直接用低层高分辨率特征、不做精炼（Fig 6）。闭合较好。
- 大核动机来自 Figure 7 的"分类 vs 姿态"对照：同样从 k=3 到 k=7，分类 +1.5%、姿态 +13 AP；k=9 下降。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 高分辨率分支冗余 | 渐进收缩 4 个配置 | Figure 2, 3 | Scaled-HigherHRNet-W16 | CrowdPose AP 50.4→51.0→51.8→53.0 |
| 融合反卷积头有效 | w/ vs w/o fusion | Table 3, Figure 6 | 普通 deconv 头 | +7.6 AP（47.1→54.7） |
| 大核有效且任务特异 | k=3/5/7(/9) 在分类与姿态上 | Figure 7, Table 3 | — | 姿态 41.7→50.7→54.7，k=9 降；分类 72.0→73.5 |
| 单分支更硬件友好 | 同延迟比较 | Figure 8 | Scaled-HigherHRNet-W16 | 更高 MACs、相近延迟、+11.6 AP |
| 超网训练、NAS、蒸馏各有贡献 | 逐项开关 | Table 3 | — | 超网 +1.4/+2.7，搜索 +2.2，蒸馏 +1.1 |
| CrowdPose 上更快更准 | 3 个平台 | Table 1, Figure 9 | HigherHRNet-W48、Scaled-HigherHRNet-W24/W16、EfficientHRNet | AP、MACs、Nano/Snapdragon/Pi 延迟；5.0×/4.9×/5.0× |
| COCO 上更快更准 | | Table 2, Figure 1 | PersonLab、Hourglass、HigherHRNet-W48、Lightweight OpenPose、EfficientHRNet | AP val/test-dev、延迟 |
| 定性：小人和近邻关节更好 | 可视化 | Figure 5 | 无 fusion / 3×3 | 定性 |

## 4. 实验设计手法
- **基线选择**：EfficientHRNet（同类高效底向上 SOTA）+ 自建的 Scaled-HigherHRNet（把 SOTA 大模型缩到同计算区间，作为"同 MACs 下的多分支代表"）+ Lightweight OpenPose。大模型 HigherHRNet-W48 作为精度上界参照。未说明基线是否在同设置下重训。
- **规模/广度**：2 个数据集（CrowdPose、COCO）；3 个硬件平台（Jetson Nano GPU、Snapdragon 855 GPU、Raspberry Pi 4B+）；4 个模型档（XS/S/M/L，表中只出现 XS、S）。
- **效率测量**：真机延迟；Pi 与 Nano 用 TVM AutoScheduler 优化（§5.2 Measurement Details），以"模拟真实部署"；Snapdragon 的后端未说明。同时报 MACs 与延迟，且专门展示 MACs 与延迟不一致的情形（Fig 8）。
- **精度–效率联合呈现**：Figure 9 四联图——AP 对 MACs 与对三个平台延迟分别画帕累托；Table 1/2 括号标倍数。
- **消融**：Table 3 是逐项累加：核大小 → 融合头 → 超网训练 → 搜索 → 蒸馏，并在两个规模（0.5 LitePose、XS）上重复。
- **分析性实验**：渐进收缩（反事实桥接）、分类 vs 姿态的大核对照（任务对照）、同延迟不同 MACs 的并行度说明。
- **泛化**：CrowdPose 搜索 → COCO 直接用；但 COCO 模型用 CrowdPose 超网初始化（§5.2）。
- **统计**：无。
- **系统贡献**：代码和预训练模型开源（github.com/mit-han-lab/litepose）。
- **局限性**：无专门讨论；只提到 k=9 退化与"把核大小放进搜索空间会让 NAS 退化"（§3.4），后者是实践中的负结果但无数据。

## 5. 对抗式审读
- **最薄弱环节：渐进收缩实验只做了一个规模（W16）、一个数据集、单次运行**，AP 变化 2.6，且 Shrink2/3 把 base channel 从 16 调到 18、MACs 也从 12.5G 降到 9.15G——多个变量同时变动。"高分辨率分支在低计算区间冗余"的一般性结论超出了这一条曲线的支持范围（"低计算区间"的边界也没有被扫描）。
- **数字不一致**：摘要说"7×7 比 3×3 +14.0 mAP"，Table 3 与 Fig 7 是 +13.0（41.7→54.7）；+14.0 实际是 COCO 上对 Lightweight OpenPose 的差。
- **比较的公平性**：LitePose 用了 NAS、超网预训练（COCO 上先训 100 epoch 无 AE loss）、长训练（超网 800/2400 epoch、微调 200/500 epoch）、XS 还用 LitePose-S 蒸馏；基线是否同等训练预算未说明。主表的 5.0× 延迟优势中，多少来自架构、多少来自训练配方，Table 3 只部分回答。
- 延迟比较：未说明 EfficientHRNet 等基线是否也经 TVM AutoScheduler 调优；若只给 LitePose 调优，则倍数被放大。
- 精度绝对值远低于大模型（COCO 56.8 vs HigherHRNet-W48 69.9），论文定位在"低计算区间"，这一点表述诚实，但 Figure 1 的 "2.9× latency reduction" 是 XS 与 EfficientHRNet-H-4 的比较（两者 AP 都只有 ~36–41）。

## 6. 可迁移的手法
1. 要挑战"架构 A 的某组件必要"这一共识，在 A 与简化版 B 之间构造一串连续的中间结构（渐进收缩），沿着这条路径测性能，而不是只比两端。（§3.2, Figure 2/3）**直接读到**
2. 要说明某设计"对本任务特别有效"，在另一任务（分类）上做同样的扫描作为对照，用增益差异证明任务特异性。（Figure 7）**直接读到**
3. 当 MACs 与真实延迟脱钩时，专门做"同延迟、MACs 更高、精度更高"的对比，用实测把代理指标拆穿。（Figure 8）**直接读到**
4. 帕累托图同时对 MACs 和多个平台的延迟各画一张，说明优势不是某一平台的偶然。（Figure 9）**直接读到**
5. 消融表在两个模型规模上重复同一组开关，增强组件贡献的可信度。（Table 3）**直接读到**

## 7. 一句话
"渐进收缩"把 HigherHRNet 一步步拆成单分支、AP 反而上升，这一条桥接曲线让"高分辨率分支冗余"从论断变成可视的观察，后续设计都顺着它展开。
