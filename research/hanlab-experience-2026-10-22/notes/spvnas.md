# Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution（SPVNAS）

## 0. 元信息
- ECCV 2020（arXiv 2007.16100v2；txt 未印会议名，会议名为我的已知信息）
- 类型：算法（新 3D 模块 SPVConv + 3D-NAS）+ 实现层优化（GPU 哈希表）
- 全文含附录 A.1–A.3 已读。Figure 5/6 柱状数值错位但正文复述了关键数字；表格基本完整。

## 1. 核心主张
- 主主张：在硬件受限时，PVConv（粗体素化）与稀疏卷积（激进下采样）都会丢失小物体信息；给稀疏卷积加一条高分辨率点分支（SPVConv），再用 3D-NAS 搜索结构，得到 SemanticKITTI 上比 MinkowskiNet 高 3.3 mIoU（榜单第一），或在 +0.6 mIoU 下 7.6× 少计算、2.7× 实测加速。
- 次要主张：(a) 点分支几乎免费（MACs +4%）但专注小物体（Fig 5/6, Table 5）；(b) 3D CNN 中只缩通道不能带来实测加速（MACs 降 4×/8×，延迟几乎不变），所以搜索空间必须包含弹性深度；(c) 3D 中大核不划算，固定 k=3；(d) 可迁移到检测（SECOND 上替换）。

## 2. 动机是怎么立住的
- Figure 1：大场景在 0.8m 分辨率下行人/骑车人难辨——直观示意。
- **Table 1（关键动机实验）**：把自家 PVCNN 用到 SemanticKITTI——整场景输入时体素 0.78m、39.0 mIoU；滑窗保持 0.05m 则需 244 个窗口、35.6 秒/帧。用"自己上一篇方法在新场景下失效"的测量来立痛点，同时堵住"为什么不直接用 PVCNN"。
- §3.2：MinkowskiNet 四次下采样后 0.05×2⁴=0.8m，与 PVCNN 粗体素同病。用算式说明。
- §4.1：MinkowskiNet 通道缩 4×/8× 后 MACs 7.5G/1.9G，但 1080Ti 延迟 105/96ms——"3D 网络受内存约束，MACs 不代表延迟"，这条观察直接决定了搜索空间设计（加入弹性深度、固定小核）。
- 因果链：分辨率丢失 → 点分支保细节（模块）；MACs≠延迟 → 深度可变（搜索空间）。两条动机分别对应两个贡献，闭合良好。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| PVCNN 不适合大场景 | 滑窗 vs 整场景 | Table 1 | PVConv | 体素尺寸、延迟、mIoU |
| SemanticKITTI SOTA | 官方测试集 | Table 2 | PointNet, SPGraph, PointNet++, PVCNN, TangentConv, RandLA-Net, KPConv, MinkowskiNet | Params、MACs、1080Ti 延迟、mIoU（66.4 vs 63.1；小模型 63.7@110ms vs 63.1@294ms） |
| 优于 2D 投影法 | 同上 | Table 3 | DarkNet21/53Seg, SqueezeSegV3, 3D-MiniNet, PolarNet, SalsaNext | 同上（60.3@89ms, 8.9G MACs） |
| 小物体增益大 | 逐类 IoU | Table 5 | MinkowskiNet | 例如 motorcyclist +25.0、bicycle +11.2 |
| 点分支关注小物体 | 特征范数 top 5%/10% 统计 | Figure 5, 6 | 体素分支 | 各类激活比例（bicyclist 点分支 82.3% vs 体素 7.3%） |
| 模块与 NAS 同等重要 | 均匀缩放的 Mink/SPVCNN vs SPVNAS | Figure 7, 附录 Table 6/7 | MinkowskiNet、SPVCNN | mIoU vs MACs、vs 延迟 |
| 进化搜索有效、不只靠搜索空间 | ES vs RS；20 个随机模型重训 | Figure 8 | 随机搜索、随机模型 | SPVNAS 比随机模型均值高 0.8 |
| 检测可迁移 | KITTI，SECOND 中替换稀疏卷积 | Table 4（test）, 附录 Table 8（val，3 次平均） | SECOND 原文与自己复现 | 3D AP |
| 哈希表实现更快 | kernel map 并行化 | §3.3 | Choy et al. 实现 | 1.3×（基线也升级到该实现） |

## 4. 实验设计手法
- **基线选择**：最强对手 MinkowskiNet 被**升级到同一加速实现**后再比（§3.3 明说"both our method and the baseline have been upgraded"）——主动消除实现差异，这正是 PVCNN 被诟病之处的修正。检测基线 SECOND 自己复现且复现结果高于原文，再在此基础上"只替换模块、其余设置不变"。
- **规模/广度**：SemanticKITTI（分割）+ KITTI（检测）；对比覆盖点、体素、稀疏卷积、2D 投影四大流派。只有一种 GPU（1080Ti），无边缘设备。
- **效率测量**：1080Ti 实测延迟，并把延迟拆成"计算 + 后处理/投影"两段（Table 2/3 红蓝数字）——对 RandLA-Net、2D 投影法的预/后处理时间显式计入。MACs 按全数据集平均 kernel map 大小估计（§4.3，因稀疏卷积的 MACs 依赖输入）。
- **精度–效率联合呈现**：Figure 7 双帕累托（mIoU vs MACs、vs 延迟）；附录 Table 6/7 给出**同参数量下** Mink vs SPVCNN vs SPVNAS 的四档对比，是"等成本比精度"的完整网格。
- **消融**：模块（SPVCNN vs Mink 同宽度）与搜索（SPVNAS vs SPVCNN）分离（Fig 7, Table 6/7）；搜索算法（ES vs RS）；搜索空间本身（随机模型）。
- **分析性实验**：点分支特征范数可视化与逐类统计（Fig 5/6）；搜得结构的计算分布（上采样阶段占比 77% → 47–63%，§6.2）。
- **泛化**：分割 → 检测；测试集 → 附录验证集（并在验证集实验中**重新跑一遍完整搜索**，用 seq 10 作 mini-val，避免在报告集上搜索，附录 A.2.1）。
- **统计**：附录 Table 8 检测 val 结果为 3 次训练平均；Fig 8b 有 20 个随机模型的分布；主表单次。
- **系统贡献**：GPU 哈希表实现；正文未提开源。
- **局限性**：无。

## 5. 对抗式审读
- **最薄弱环节：头条数字来自两个不同模型。** "+3.3 mIoU" 来自 12.5M/73.8G/259ms 的大模型（对 Mink 仅 1.1× 加速）；"8× 计算、3× 加速"来自 2.6M/15G/110ms 的小模型（仅 +0.6 mIoU）。摘要把两者写在一起，容易被读成同时成立。
- **模块本身的增益有限且不免费**：附录 Table 6，同参数下 SPVCNN 比 Mink 高 0.7–1.6 mIoU，但延迟也更高（例如 124.3 vs 115.7ms、317.1 vs 294.0ms）。效率优势几乎全部来自 3D-NAS，而"小物体提升"的逐类表（Table 5）用的是 SPVNAS（含 NAS），把模块与搜索的贡献混在一起；Fig 5/6 的机制分析用的是 SPVCNN，二者对象不同。
- Table 3 对 2D 投影法：SPVNAS 89ms 比 PolarNet（62ms）和 SalsaNext（71ms）都**慢**，mIoU 只比 SalsaNext 高 0.8；正文只强调对 DarkNet 的 1.2× 加速与 MACs/参数倍数。正文写"至少 8.5× 模型缩小"而表题写"至少 3.6×"，二者不一致（3.6× 对应 3D-MiniNet 4.0M）。
- KITTI 检测 Table 4：行人 Mod/Hard SPVCNN 低于复现的 SECOND（41.4 vs 41.7，38.4 vs 39.1）；"consistent improvement"是对"多数类别"成立。§5.2 写"follow train/val split… report mAP on the test set"，测试集结果与 val split 训练的关系交代不清。
- 搜索在官方验证集 seq 08 上进行再提交测试集（附录 A.1.2）；作者在附录对 val 报告单独重跑搜索，这一点处理得当。
- 无边缘设备测量，与"自动驾驶车载资源受限"的动机之间有距离。

## 6. 可迁移的手法
1. 用"自己上一篇方法在新场景下失效"的定量实验作为新工作的动机，同时预先回答"为什么不直接用旧方法"。（Table 1）**直接读到**
2. 当你改进了底层实现（如哈希表），把最强基线也升级到同一实现再比，使剩余差距只来自方法。（§3.3）**直接读到**
3. 用"MACs 降 8× 但延迟几乎不变"的实测反例，论证搜索空间该包含哪些维度（深度而非通道/核大小）。（§4.1）**直接读到**
4. 把模块贡献与搜索贡献分开：同宽度的"旧模块 vs 新模块"+"新模块手工 vs 搜索"，在多个成本档上列成网格。（Figure 7, 附录 Table 6/7）**直接读到**
5. 验证 NAS 不只是搜索空间好：同时对比随机搜索曲线与"重训 N 个随机模型"的分布。（Figure 8）**直接读到**
6. 机制解释用"分支特征范数 top-k 的逐类比例"把定性可视化变成可数的统计。（Figure 5 → Figure 6）**直接读到**

## 7. 一句话
Table 1 用自家 PVCNN 的失效立题，再把 MinkowskiNet 升级到同一实现后在多个成本档上逐档比较，使"效率提升来自方法而非实现"这一点可信；但头条的"+3.3"和"3×"分属两个模型。
