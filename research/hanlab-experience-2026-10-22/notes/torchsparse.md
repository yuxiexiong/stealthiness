# TorchSparse: Efficient Point Cloud Inference Engine

## 0. 元信息
- MLSys 2022（arXiv 2204.10319）；Haotian Tang*, Zhijian Liu*, Xiuyu Li*, Yujun Lin, Song Han
- 类型：系统（GPU 推理引擎，CUDA + PyTorch 风格 API）
- 实现层级：真实 GPU 实测（GTX 1080Ti、RTX 2080Ti、RTX 3090），PyTorch 1.9.1、CUDA 10.2/11.1、cuDNN 7.6.5。无硬件仿真。
- 全文读完（正文 + 附录 A/B/C）。Fig.11 的柱状数字可读但列对应关系部分错位，下文只引用正文给出的数字。
- 注意：早先 `raw/txt/torchsparsepp.txt` 曾是本文的重复下载；协调者已替换为真正的 TorchSparse++（MICRO'23）正文，续作对比见 torchsparsepp.md 第 5 节。

## 1. 核心主张
- 主：稀疏卷积有两个瓶颈——计算不规则、数据搬运；按"以计算换规则性"和"减少内存足迹"两条原则优化后，端到端比 MinkowskiEngine 快 1.6×、比 SpConv 快 1.5×（Abstract, Fig.11）。
- 次：(a) 自适应矩阵乘分组使 matmul 加速 1.4–1.5×（Table 2）；(b) 量化+向量化+融合+局部性感知访存使 scatter/gather 加速 2.72×（Table 3）；(c) mapping 优化加速 4.6×（Fig.13）；(d) 按数据集/模型/硬件特化的分组策略优于迁移来的策略（Table 1）；(e) 所有模型在三代 GPU 上达到实时（≥10 FPS，Appendix C）。

## 2. 动机
- 背景重要性：SemanticKITTI 前 5、nuScenes 前 10 中 9 个、Waymo 前 2 均基于 SparseConv（Sec.1）——用排行榜证明负载真实重要。
- Profiling：Fig.4 分解分割（MinkUNet）与检测（CenterPoint）运行时：数据搬运约 40–50%，GEMM 20–50%，检测中 mapping 约 15%（Sec.3）。matmul 只有 30% 设备利用率（8.1 TFLOP/s on 2080Ti FP16）。
- 关键观察：不同 kernel offset 的 map 大小相差一个数量级（Fig.12），且对称 offset 的 map 大小相同（Sec.4.2.1，有严格推导）；每个点特征被重复访问 ≥4 次（N1 ≥ 4N2，Sec.4.3.2）。
- 因果链：两条原则（Principle I/II）直接由 Fig.4 推出，再映射到 Sec.4.2/4.3/4.4 三组优化，闭合好。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 端到端更快 | 7 个模型配置 × 3 个数据集 × 3 代 GPU | Fig.11, Fig.14（绝对 FPS） | 未优化 FP32 基线、MinkowskiEngine v0.5.4、SpConv v1.2.1 FP32/FP16 | 归一化 FPS |
| 自适应分组有效 | 4 种分组策略 | Table 2, Fig.7 | separate / symmetric / fixed / adaptive | TFLOP/s 与 matmul 加速 |
| 手工策略不能泛化 | fixed 在 nuScenes 1.5× 但在 SemanticKITTI 0.87× | Table 2 | — | — |
| 特化优于迁移 | 跨数据集/模型/硬件 2×2 交叉 | Table 1 | 迁移来的策略 | TFLOP/s（最多 13.5% 提升） |
| 数据搬运优化 | 逐步加入 FP16 → 向量化+融合 → 局部性感知 | Table 3 | FP32 | gather/scatter 加速 |
| mapping 优化 | 逐步累加 | Fig.13 | 基线 hashmap | 1.6×、1.5×、1.8×、1.1× |
| 收益非来自 tensor core | 在无 FP16 tensor core 的 1080Ti 上仍 1.5× | Sec.5.2 | 基线 | — |
| 实时性 | | Sec.5.2, Fig.14 | 10 FPS / LiDAR 频率 | FPS |

## 4. 实验设计手法
- 基线：两个最强开源库的最新版本，且**修了基线**："Necessary changes are made to MinkowskiEngine to correctly support downsample operations in detectors and to SpConv to avoid OOM"（Sec.5.1）。还包括 SpConv 的 FP16 版本（避免"你只是用了 FP16"的攻击），以及作者自己的未优化 FP32 基线（朴素基线）。
- 规模：MinkUNet（0.5×/1× 宽度、1/3 帧）、CenterPoint（1/3/10 帧）；SemanticKITTI、nuScenes、Waymo；3 代 GPU。
- 效率测量：真实 GPU 端到端 FPS；主动指出 CenterPoint 有 10% 非点云计算（2D conv/NMS），因此稀疏卷积本身的加速比端到端高约 10%。未交代重复次数/预热。
- 精度–效率：不适用（推理引擎不改精度）；未报告 FP16 对精度的影响——**未直接验证**。
- 消融：三组 breakdown（Table 2/3, Fig.13），且 Table 3 展示"单独 FP16 只有 1.17×（理论 2×）"、"单独融合无效、与局部性感知结合才有效"——用负面中间结果解释机制。
- 分析：与理论值对照（FP16 理论 2×，向量化后 1.93× 接近理论）；内存事务的解释（warp 128B 事务利用率 50%，Sec.4.3.1）。
- 泛化：Table 1 交叉迁移实验，显式检验"策略是否需要特化"。
- 统计：无方差。
- 系统贡献：开源（torchsparse.mit.edu），PyTorch 风格 API（Fig.5），有 artifact evaluation 致谢。
- 局限：承认 MinkowskiEngine 在小模型（1 帧 nuScenes MinkUNet）上用 fetch-on-demand 数据流表现更好，因此本文加速较小（Sec.5.2）；未有独立局限性章节。

## 5. 对抗式审读
- 自适应分组需要在目标数据集+硬件上做约 1,000 配置的网格搜索（<10 分钟，100 个训练样本），Table 1 证明特化必要——这也意味着对新数据/硬件有调优成本；对 MinkowskiEngine/SpConv 是否给予同等调优没有说明（它们没有该参数）。
- 精度：FP16 特征量化对 mIoU/mAP 的影响未报告。
- 基线修改的具体内容（MinkowskiEngine 下采样修正、SpConv OOM 规避）未详述，可能影响基线速度。
- 无重复测量/方差；加速 1.5–1.6× 量级不大，测量噪声的影响未评估。
- 只评估推理；"支持 CPU 推理与多 GPU 训练"（Sec.4.1）**未验证**。

## 6. 可迁移手法
1. 用公开排行榜的头部方案构成说明目标负载的现实重要性。**直接读到**（Sec.1）
2. 从 profiling 抽出两条一般性"原则"，再把所有优化挂在原则下，使系统论文有统一叙事。**直接读到**（Sec.3 Principle I/II）
3. 把理论上限（FP16 2× DRAM 节省）与实测（1.17× → 1.93×）对照，用差距引出下一项优化。**直接读到**（Sec.4.3.1, Table 3）
4. 用 2×2 交叉"特化 vs 迁移"表证明自适应/自动调优的必要性。**直接读到**（Table 1）
5. 在缺少某硬件特性的平台上复测（1080Ti 无 tensor core），排除"收益来自硬件特性"的替代解释。**直接读到**（Sec.5.2）
6. 基线取最新版、必要时修 bug 使其可运行，并显式说明。**直接读到**（Sec.5.1）

## 7. 一句话
从 profiling 归纳出两条原则，每项优化都在真实 GPU 上逐项消融并与理论上限对照，再用 1080Ti 等"去掉某特性"的平台排除替代解释。
