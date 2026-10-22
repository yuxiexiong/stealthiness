# APQ: Joint Search for Network Architecture, Pruning and Quantization Policy

全文已读完（CVPR 版 8 页 + 参考文献，无附录）。Fig 5/6/7 只剩少量数字标签（如 61.4、71.9、72.7、+10.5%、+11.3%、64.6→75.6），曲线不可读。

## 0. 元信息
CVPR 2020。Wang, Wang, Cai, Lin, Liu, Han。类型：算法（联合 NAS + 通道剪枝 + 混合精度搜索）。

## 1. 核心主张
- 主主张：把架构、剪枝、量化放进一个联合搜索（OFA 超网 + 量化感知精度预测器 + 进化搜索）比分阶段流水线更优——同延迟下比 ProxylessNAS+AMC+HAQ 高 2.3% top-1（74.1 vs 71.8，Table 2/Fig 6），新场景边际搜索成本降两个数量级（0.5 GPU h vs 300 GPU h；CO2 0.14 vs 85.08 lbs）。
- 次主张：predictor-transfer（先训 FP32 预测器再迁移到量化）提升样本效率：小数据下 pairwise accuracy 64.6%→75.6%，达到 85% 所需数据 <3k vs ≥4k（Fig 7）；带 transfer 的模型 74.1% vs 不带 72.1%（Table 2）。

## 2. 动机是怎么立住的
- 痛点 1：分阶段次优——"FP 最优架构量化后不一定最优"（Sec 1/2）。这是**断言**，本文没有一个直接展示"排序翻转"的实验（例如 FP 精度最高的子网在量化后排名下降的散点）。证据是最终结果层面（Fig 6 联合 > 顺序）。
- 痛点 2：搜索成本与 CO2——Fig 2 用对数坐标展示各方法"每个新场景的 CO2 磅数"（APQ 0.14 vs MnasNet 11,344），引用 Strubell 的换算。这是**叙事框架的选择**：把成本归一到"边际成本"，固定成本（OFA 训练 2400 GPU h）放进"design cost"列。
- 痛点 3：量化数据贵——每个量化样本约 0.2 GPU h，80k 样本需 16,000 GPU h（Sec 3.2）——算术估算，自洽。
- 因果链：成本贵 → 预测器 + 迁移 → Fig 7 证明迁移省数据，这一环闭合；"联合比分阶段好"这一环靠端到端结果闭合，但混入了"搜索空间更大（OFA 空间 vs 固定 MobileNetV2）"的混淆（作者在 Fig 5 图注自己承认增益"may benefit from the dynamic architecture search space"）。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 优于 SOTA 固定/混合精度模型 | ImageNet 多约束 | Table 2 | MBv2-8bit、ProxylessNAS-8bit、+AMC、MBv2+HAQ、PNAS+AMC+HAQ、DNAS、SPOS | Top-1、ms、mJ、BitOps、GPU h、CO2、$ |
| 优于 HAQ（同延迟/同能耗） | BitFusion 多约束点 | Fig 5 | 4/6bit 固定精度、MBv2+HAQ | Top-1 |
| 联合 > 顺序 | 同延迟 | Fig 6 左 | ProxylessNAS → +AMC → +HAQ 逐步 | Top-1 |
| BitOps 约束下优 | — | Fig 6 右 | ResNet-34 2/3/4bit、SPOS | Top-1、BitOps |
| predictor-transfer 有效 | 不同数据量下预测器质量 | Fig 7, Table 2 (Ours-A vs B) | 从零训练预测器 | pairwise acc、最终 Top-1 |
| LUT 估延迟足够准 | 「未直接验证」（"we find such practice can precisely approximate"，无数据） | Sec 3.3 | — | — |
| 预测精度≈真实精度 | 「未直接验证」（只报 pairwise accuracy，未报预测值 vs 实测值） | — | — | — |

## 4. 实验设计手法
- 基线：覆盖三类——固定精度 SOTA（MBv2/ProxylessNAS 8bit）、混合精度（HAQ）、**自家顺序流水线 ProxylessNAS+AMC+HAQ**（这是最关键的"分阶段"对照，用的全是自家方法，保证每阶段都是作者熟悉的强实现）、外部 NAS 量化方法（DNAS、SPOS，数字可能引自原文，只有 BitOps 无延迟）。消融式基线：Ours-A（无 transfer）。
- 规模/广度：只有 ImageNet 分类；只有 MobileNetV2 型搜索空间；**只有一个硬件平台 BitFusion**（ASIC 设计，延迟/能耗由 LUT 求和），未在真实设备上测。
- 效率测量：逐层 LUT 求和（Sec 3.3）；无实测验证 LUT 精度的数据。成本以 GPU h、CO2、$ 三种单位报，并拆成固定 + 边际（2400 + 0.5N）。
- 精度–效率联合：Table 2 多列同时报；Fig 5/6 同约束比精度；Ours-A/B/C 在不同延迟约束下给出多点。
- 消融：只有 predictor-transfer（Fig 7, Ours-A vs B）。无"联合搜索 vs 在同一 OFA 空间上分阶段搜索"的对照——这才是隔离"联合"本身贡献的关键消融，缺失。
- 分析性实验：无策略可视化（不同于 HAQ/AMC）。
- 泛化：无跨任务/硬件。
- 统计：无方差；进化搜索结果单次。
- 系统贡献：Abstract 称代码与视频公开。
- 局限性：无。

## 5. 对抗式审读
- 最薄弱环节：联合 vs 顺序的比较混淆了两个变量——搜索空间（OFA 的弹性核/通道/深度 vs ProxylessNAS 的空间 + AMC 剪枝）与优化方式（联合 vs 分阶段）。缺少"同一空间内分阶段"的对照，"joint 带来增益"的结论超出证据。
- 成本叙事：边际成本 0.5 GPU h 的前提是已付 2400 GPU h 训练 OFA；对于单一部署场景，APQ 的总成本（2400.5）远高于 MBv2+HAQ（96）。Table 2 用"N 个场景"的写法把这一点放在了列里，但 Fig 2 和 Abstract 只强调边际。
- Table 2 中 Ours-B 能耗 12.18mJ 高于 ProxylessNAS+AMC+HAQ 的 8.84mJ，Abstract 的"能耗 1.3× 降低"是相对 MBv2+HAQ（同精度点、来自 Fig 5，不可读）而非对最强顺序基线。
- 预测器质量只用 pairwise accuracy（85% 左右）报告，意味着约 15% 的两两比较判错；最终模型精度是否经过量化微调后实测，正文描述不清。
- 单平台 BitFusion + LUT，没有实测。

## 6. 可迁移手法
1. 把"顺序组合自家已有方法"作为最强基线，并在图中逐步画出每一步的增益（PNAS → +AMC → +HAQ，Fig 6 左），再展示联合方法超过整条链。—直接读到
2. 成本拆成"固定 + 边际 × N"并以 CO2/美元等外部单位报告，把一次性大投入转化为摊销叙事（Table 2, Fig 2）。—直接读到
3. 证明迁移学习式数据效率时，以"数据量 vs 预测器质量"曲线给出达到同等质量所需样本数（Fig 7）。—直接读到
4. 预测器质量用排序指标（pairwise accuracy）而非绝对误差，因为搜索只需要排序。—直接读到；我的归纳：这同时回避了绝对误差的检验
5. 用 w/o 组件版本（Ours-A）直接放进主结果表，让消融与 SOTA 对比同表呈现（Table 2）。—直接读到

## 7. 一句话
用一张同时列精度、延迟、能耗、GPU 小时、CO2、美元并拆出边际成本的大表（Table 2），加上"自家三段流水线逐步累加仍不如联合搜索"的图（Fig 6），把"联合 + 摊销"包装成既更准又更绿；但联合本身的贡献未与搜索空间扩大分离。
