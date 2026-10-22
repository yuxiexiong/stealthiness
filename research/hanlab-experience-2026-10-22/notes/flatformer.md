# FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer

## 0. 元信息
- CVPR 2023（arXiv 2301.08739v3）；Zhijian Liu*, Xinyu Yang*, Haotian Tang, Shang Yang, Song Han
- 类型：算法-系统协同（模型结构 + GPU 实现优化）；不是硬件加速器，但在本组中作为"为硬件规则性改模型"的代表
- 实现层级：真实 GPU 实测（Quadro RTX A6000 FP16；边缘端 Jetson AGX Orin）。
- 全文读完（正文 + 参考文献）。**正文多次引用的附录（L1 mAP/mAPH 等详细指标）不在该 txt 中**，arXiv 文本以参考文献结束，附录读不到。

## 1. 核心主张
- 主：点云 Transformer 慢于稀疏卷积的根本原因是计算不规则；把点云按窗口排序后**等大小分组**（而非等形状窗口），以牺牲空间邻近性换计算规则性，可在 Waymo 上达到 SOTA 精度且比 SST 快 4.6×、比 CenterPoint 快 1.4×（Abstract, Table 1）。
- 次：(a) 首个在边缘 GPU 上实时（16.1 FPS on Orin）的点云 Transformer（Fig.6）；(b) 计算规则性比空间规则性更重要（Table 3）；(c) 系统优化带来 2.9× 骨干加速（Fig.8）；(d) 作为两阶段检测器骨干同样有效（Table 2）。

## 2. 动机
- Sec.3 "Why are PCTs slow?" 是一个分门别类的解剖：
  - 全局 PCT：Fig.2 延迟随点数二次增长，32k 点时近 1 秒，比 CenterPoint 慢 66×。
  - 局部 PCT：Fig.3 邻居查询与数据重排占 VoTr 36% 运行时（22ms），PT 中高达 70%；单层开销已超过整个 CenterPoint。
  - 窗口 PCT（SST）：Fig.4 窗口内点数分布（差异 >80×，两个数量级），padding 带来 1.7× 计算开销，窗口划分本身 18ms/场景。
- 关键 insight：Fig.1 的精度-延迟散点直接把"Transformer 3–4× 慢"可视化；Fig.4 的直方图把"不均衡"量化。因果链闭合：不均衡 → padding/划分开销 → 改为等大小分组。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 单阶段 SOTA + 更快 | Waymo val，1/2/3 帧分组比较 | Table 1 | CenterPoint、CenterPoint++、PillarNet（稀疏卷积）；SST、SST-Center、VoxSet、VoTr、SWFormer（Transformer） | L2 mAPH、延迟、MACs |
| 边缘实时 | Jetson AGX Orin | Fig.6 | CenterPoint、SST、SST-Center | FPS |
| 两阶段也有效 | 替换 FSD 骨干 | Table 2 | CenterFormer、FSD-SpConv、MPPNet 等 | L2 mAPH、延迟 |
| 等大小分组 > 等窗口 | 排序策略消融 | Table 3 | 无量化、无轴交替、无窗口平移、随机顺序、SST | L2 mAPH（20% 数据训练） |
| 超参不敏感 | 窗口形状、组大小、分辨率 | Table 4/5/6 | — | L2 mAPH |
| 系统优化贡献 | 逐步累加 | Fig.8 | 未优化 FlatFormer、CenterPoint | 骨干延迟 |
| 注意力学到抑制离群点 | 可视化 | Fig.7 | — | 定性 |
| 更硬件友好 | MACs 多 1.7× 但更快 | Table 1, Sec.5.2.1 | CenterPoint++ | MACs vs 延迟 |

## 4. 实验设计手法
- 基线：两大阵营（稀疏卷积 vs Transformer）都选最强代表，按帧数分组比较；稀疏卷积骨干统一用 SpConv v2.2.3（最新最优库），之后模块统一 TensorRT 8.4；CenterFormer 被**改到 SpConv v2.2.3 FP16 后端**以求公平（Sec.5.2.2）。精度数字多数取自原作者（Table 1 脚注 1–4），SST-Center 3 帧为自行复现，VoTr 延迟为"投影"，SWFormer 延迟是 T4 上的——在表中用记号标出。
- 效率测量：前 1,000 个样本、50 次运行、10 次预热，去除离群值取均值；不含数据加载与后处理（Sec.5.2.1）。交代得相当完整。
- 精度–效率联合：Fig.1 延迟-精度散点；Table 1 同帧数下同时报精度与延迟；"MACs 更多但更快"用来证明硬件友好性（与 FLOPs 脱钩）。
- 消融：模型设计消融（Table 3–6）用**单帧、20% 训练数据**做；系统优化逐步累加（Fig.8：FlashAttention 1.7× → QKV 打包 1.3× → FFN 融合 1.2× → drop residual 1.1×，合计 2.9×）。
- 分析：注意力可视化（Fig.7）解释"为何空间不规则可以接受"。
- 泛化：单阶段 + 两阶段；1/2/3 帧；A6000 + Orin。仅 Waymo 一个数据集。
- 统计：无多 seed、无方差。
- 系统贡献：项目页 flatformer.mit.edu；Triton 实现的融合 FFN。
- 局限：承认两阶段检测器瓶颈在第二阶段，超出本文范围（Sec.5.2.2）；未专门的局限章节。

## 5. 对抗式审读
- **只用一个数据集（Waymo）**。"首个比稀疏卷积更快且同等精度"的主张只在 Waymo + NVIDIA GPU 上验证；nuScenes 等未测。
- **精度比较混合了来源**：多数基线精度来自原作者/他文报告（Table 1 脚注），训练细节（epoch、增强）未必对齐；无多 seed，而 FlatFormer 相对 PillarNet 的优势在 1 帧时为 0（67.2 vs 67.2）。
- **消融用 20% 数据**训练，绝对值（61.7）远低于主结果（67.2），消融结论能否外推到全数据**未验证**。
- 系统优化（FlashAttention、Triton 融合）是通用 Transformer 优化，部分加速并非来自"等大小分组"思想本身；作者说这些优化"不适用于稀疏卷积"，但没有给 SST 同样套用 FlashAttention 的对照——SST 的 4.6× 差距中多少来自实现优化、多少来自分组设计，**未拆分**。
- 延迟对比：SST 用其官方实现（未必 FP16/FlashAttention 优化），而 FlatFormer 叠加了全部系统优化。

## 6. 可迁移手法
1. 用"为什么 X 慢"一节按方法族逐一解剖瓶颈（全局/局部/窗口），每类一张量化图，再引出设计。**直接读到**（Sec.3, Fig.2–4）
2. 用"MACs 更多但延迟更低"直接论证设计的硬件友好性，打破 FLOPs 代理。**直接读到**（Table 1, Sec.5.2.1）
3. 精度-延迟散点图作为第一张图，一眼呈现"两阵营的 gap"和新方法的位置。**直接读到**（Fig.1）
4. 在边缘设备上补一个实时 FPS 测量，把"快"翻译成应用门槛（>10 FPS = LiDAR 频率）。**直接读到**（Fig.6）
5. 用可视化（注意力权重）解释为什么牺牲的那一侧（空间邻近性）代价不大。**直接读到**（Fig.7）
6. 测量协议写死（样本数、运行次数、预热、剔除离群、排除的阶段）。**直接读到**（Sec.5.2.1）

## 7. 一句话
先用 Sec.3 把三类点云 Transformer 的慢因逐一量化，再用"MACs 多 1.7× 却更快"证明等大小分组换来的规则性在真实 GPU 上兑现。
