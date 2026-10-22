# TorchSparse++: Efficient Training and Inference Framework for Sparse Convolution on GPUs

## 0. 元信息
- MICRO 2023（正文 ACM 引用格式：MICRO '23, Toronto, DOI 10.1145/3613424.3614303）；Haotian Tang*, Shang Yang*, Zhijian Liu, Ke Hong, Zhongming Yu, Xiuyu Li, Guohao Dai, Yu Wang, Song Han
- 类型：系统（GPU 稀疏卷积库：kernel 生成器 + 自动调优器；训练 + 推理）
- 实现层级：CUDA + PTX，基于 TorchSparse 代码；PyTorch 1.12.0 / CUDA 11.7 / cuDNN 8.4.1；**真实 GPU 实测**：A100、RTX 3090、RTX 2080Ti、GTX 1080Ti、Jetson Orin（跨 Pascal/Turing/Ampere，FP16/TF32/FP32）。与 ASIC 比较部分是按比例换算的投影（Table 2）。
- 来源：协调者从项目页 Dropbox 取回的正文（`raw/txt/torchsparsepp.txt`，约 1.5 万词），已核对首页标题与作者。之前那份同名 txt 是 TorchSparse 的重复，本笔记已整体重写。
- 读全情况：正文 + 参考文献全部读完，无附录。**第 1 页 Fig.1 的拼贴图把其他论文（Waymo 3D Auto Labeling、SemanticKITTI 等）的图内文字抽了出来，与正文混排，约 200 行乱码**，图注本身可读。Fig.14–23 的柱状数字可读但列对应关系有错位，下文只引用正文中明确写出的数字。

## 1. 核心主张
- 主：稀疏卷积的 GPU 实现面临"易实现但慢（gather-GEMM-scatter）"和"快但工程成本极高（implicit GEMM，SpConv v2 元编程器 >4 万行）"的两难。TorchSparse++ 用 Sparse Kernel Generator 以极低工程成本生成"访存与计算重叠"的 kernel，再用 Sparse Autotuner 在扩大的设计空间中搜索。结果在 A100 推理上比 MinkowskiEngine、SpConv 1.2、TorchSparse、SpConv v2 端到端快 2.9×、3.3×、2.2×、1.7×；混合精度训练比 SpConv v2 快 1.2–1.3×（Abstract）。
- 次：
  - (a) 生成器代码量仅为 SpConv 2.3.5 元编程器的 5%，且在相同数据流参数下就快 1.1–1.2×（Sec.6.3）。
  - (b) 一阶代理指标（计算量、DRAM 访问量，甚至所有卷积 kernel 的总时间）不能预测端到端延迟，原因是 mapping 开销（Sec.2.3, Table 3/4）。
  - (c) 训练中 forward/dgrad/wgrad 偏好不同的数据流参数，做部分解绑可提升最多 10%（Fig.22）。
  - (d) 同一库也能跑 R-GCN，比 DGL/PyG/Graphiler 快 2.6–7.6×（Fig.16）。
  - (e) 在算力相近时，3090 上的 TorchSparse++ 达到放大版 PointAcc ASIC 速度的 56%（Table 2）。

## 2. 动机
- 痛点用**工程成本**立论，而不是性能 profiling：SpConv v2 为实现 implicit GEMM 从头重写了 CUTLASS，元编程器超过 4 万行（Sec.2.3）。这在系统论文里少见，属于"可维护性/生产力"类痛点。
- 机制层用三种数据流的瀑布图（Fig.3）和逐例计数（Fig.4–6）说明各自取舍：
  - fetch-on-demand：零冗余计算，但写回流量是理论最优的 4–10×，还需要原子操作。
  - implicit GEMM：写回最少，但有冗余计算——在 Fig.5 的例子里冗余 34 MAC，多于 22 个有效 MAC；SpConv v2 的 bitmask 排序把冗余降到 26。
- 关键 insight 1："稀疏卷积 = 只在 DRAM 读写迭代器上带稀疏性的稠密 GEMM"（Table 1）。因此只需手写很短的稀疏迭代器模板，片上 MMA 部分交给现有稠密张量编译器生成（Fig.7）。
- 关键 insight 2："一阶最优 ≠ 端到端最优"。论文主张保留一阶次优的配置（不排序、split>2），由端到端测量来选（Sec.4.1, Fig.11）。
- 理想化实验先证明"只调 tile 大小就够"：Fig.8 中，手工遍历 tile 后的稀疏 kernel 平均超过同规模 cuBLAS GEMM 的利用率。这是先给设计空间缩减"背书"，再做后续工作。
- 因果链闭合较好，但"<1/10 工程成本"的证据只有代码行数一项。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 推理端到端更快 | 7 个负载（与 TorchSparse 相同）× A100/3090/2080Ti/1080Ti/Orin × FP16/TF32/FP32，batch=1 | Fig.14 | MinkowskiEngine 0.5.4、SpConv 1.2.1、TorchSparse、SpConv 2.3.5（2023 年 4 月版） | 归一化速度；云端 Ampere 上分别为 2.9–3.7×、3.2–3.3×、2.0–2.2×、1.4–1.7×；Orin 上对 SpConv 2.3.5 为 1.25× |
| 训练更快 | 7 个负载，A100/2080Ti，batch=2，FP16 混合精度 | Fig.15 | ME（FP32，不支持 FP16）、TorchSparse、SpConv 2.3.5 | FPS；4.6–4.8×、2.5–2.6×、1.2–1.3× |
| 训练精度不变 | 声称收敛到与 FP32 TorchSparse/SpConv 相同精度 | Sec.5.2 | — | **未给出数字，未直接验证** |
| 生成器本身有效 | 同数据流参数 | Fig.23 | SpConv 2.3.5 | 1.1–1.2×；代码量 5% |
| 只调 tile 就够 | 理想化遍历 | Fig.8 | 同规模 cuBLAS GEMM | 归一化吞吐 |
| 动态形状开销可消除 | 循环不变量外提 | Fig.20 | 固定形状 kernel | 朴素转换慢 1.5–1.7×，外提后持平甚至更快 |
| 边界检查开销可消除 | map 填充 | Fig.21 | 带边界检查 | 最多 1.3× |
| 离线重排优于在线（与"尽量融合"的惯例相反） | | Fig.19 | 在线重排 | 推理 +4%、训练 +12% |
| 不排序的 implicit GEMM 端到端更快 | | Table 3 vs Table 4 | 排序版（SpConv v2 默认） | kernel 级排序版更快，端到端反而慢最多 1.2× |
| 更多 split 有益（分割任务） | | Table 5, Fig.11 | split={1}（SpConv v2 默认）| 最多 1.4× |
| 混合数据流有益 | 逐层分解 | Fig.18 | 单一 iGEMM / 单一 FoD | 最多 1.06× |
| 训练参数部分解绑 | | Fig.22 | 三者全绑定 | 最多 10% |
| 自适应 tiling | | Sec.6.2 | 固定小/大 tile | 最多 1.6× |
| 设计空间各部分的贡献 | 逐步累加 | Fig.23 | SpConv 2.3.5 | 生成器 → +更多 split → +不排序 → +FoD |
| 图负载泛化 | R-GCN，5 个异构图数据集 | Fig.16 | DGL、PyG、Graphiler | 7.6×/2.6×/2.9× 加速，3.4×/4.4×/5.6× 省内存 |
| 接近 ASIC | 放大 PointAcc 到 128×128 | Table 2 | PointAcc-L（投影） | 56% |
| 微架构启示 | 3090 上带宽减半 vs 算力减半 | Sec.6.3 | — | 慢 1.2× vs 1.4×；mapping 最多占 50% |

## 4. 实验设计手法
- **基线**：四个库全部取最新版本，SpConv 2.3.5 是投稿前几个月刚发布的版本。所有基线都用 FP16（ME 除外，因为它不支持）。同时保留自家前作 TorchSparse 作为基线，这是自我超越式比较。检测任务只统计 SparseConv 层的运行时间。
- **规模**：负载与 TorchSparse 完全相同（7 个），便于和前作比较；硬件从 3 张卡扩到 5 个平台（新增 A100 和 Orin），精度从 FP16 扩到 FP16/TF32/FP32 三种；任务从推理扩到训练，并加入图神经网络。
- **效率测量**：真实 GPU 端到端测量；正文没有交代重复次数、预热和方差（与 TorchSparse 一样）。
- **精度–效率**：不适用。训练精度只有一句"we make sure ... same accuracy"，没有表格。
- **消融**：本组最细的一次，§6 共约 10 项。
  - 代理指标的反例：Table 3/4 对照，Fig.17 逐层分析。
  - 设计空间扩展的逐项价值：Table 5、Fig.18、Fig.22。
  - 生成器的工程技巧：Fig.19–21、自适应 tiling。
  - 汇总：Fig.23 的逐步累加。
- **分析**：Fig.11 分析冗余计算随 split 数的变化，区分分割任务和检测任务；"带宽减半 vs 算力减半"的敏感性实验，从系统论文中引出对硬件设计的建议。
- **泛化**：跨三代 GPU 架构、三种精度、推理与训练、点云与图。
- **统计**：无。
- **系统贡献**：开源（github.com/mit-han-lab/torchsparse），复用 TensorIR 生成片上 MMA。
- **局限**：没有独立章节。论文承认未能与 Mesorasi 直接比较，因为 Mesorasi 只支持共享权重（Sec.5.2）。

## 5. 与 TorchSparse（MLSys'22）在实验设计上的延续与升级
**延续（直接读到）**
1. **同一套 7 个负载**（SK-M 1×/0.5×、NS-M 3f/1f、NS-C 10f、WM-C 3f/1f），Sec.5.1 明说"follow TorchSparse"，Fig.14 的版式也与 TorchSparse 的 Fig.11 相同。好处是前后作可以直接对比；代价是负载集合没有更新。
2. **同样的"以计算换规则性 + 按数据集/硬件自动调优"思路**：TorchSparse 用自适应分组（ε, S），在 100 个训练样本上做约 1,000 个配置的网格搜索；TorchSparse++ 的 group-based tuner 同样在随机子集（例如 100 个 Waymo 场景）上穷举，约 2 分钟完成。
3. **同样以逐步累加的分解作为收尾证据**（TorchSparse Table 3/Fig.13 → TorchSparse++ Fig.23）。
4. **同样把理论或一阶预期与实测对照来引出设计**：TorchSparse 用"FP16 理论 2× vs 实测 1.17×"，TorchSparse++ 用"排序减少冗余计算 vs 端到端反而更慢"。

**升级**
1. **基线把前作当靶子**：TorchSparse 本身成为被超越的基线（2.0–2.2×），并新增当时最强的 SpConv 2.3.5。前作的基线是 SpConv 1.2.1，而这次 SpConv v2 是更强的对手。
2. **论证核心从"优化技巧"转为"质疑代理指标"**：TorchSparse 用 profiling 归纳出两条原则；TorchSparse++ 的核心论点是"一阶代理（计算量、DRAM、kernel 时间）不可靠"，并用 Table 3 与 Table 4 的方向相反来证明。这是更强的、反直觉的主张类型。
3. **新增工程成本这一维度**：代码行数是 SpConv 的 5%，"<1/10 工程成本"。这是从性能维度扩到生产力维度。
4. **覆盖面大幅扩展**：从 3 张 GPU 扩到 5 个平台（含边缘 Orin），从 FP16 扩到 FP16/TF32/FP32，从推理扩到训练（并单独针对训练设计 tuner 和消融），从点云扩到 R-GCN 图负载。
5. **新增与专用硬件的对照**：放大 PointAcc，按频率和峰值 MAC 做 2.2× 换算后比较（Table 2）；并从 GPU 实验反推硬件设计建议（Sec.6.3）。这把系统论文接回了实验室的加速器线。
6. **新增"理想化上界"实验**（Fig.8：只调 tile 就能超过 cuBLAS），先证明缩减后的设计空间不损失性能，这是 TorchSparse 没有的。
7. **同实验室后作反向检验前作**：Sec.5.2 称换用 TorchSparse++ 后端后，3 帧 CenterPoint 在 Orin 上比 FlatFormer 快 1.5× 且更准（前提是 2D 部分用 TensorRT 部署）。这直接削弱了 FlatFormer"比稀疏卷积更快"的卖点，因为 FlatFormer 对比的稀疏卷积用的是 SpConv v2.2.3 后端。

**未升级（薄弱点被原样继承）**
- 仍然没有重复次数、方差和测量协议说明。
- 仍然没有给出 FP16 对精度的影响数字；训练精度只有一句声明。
- 负载集合没有变化，没有新数据集。

## 6. 对抗式审读
- **数字前后不一致**：
  - 训练加速：Abstract 写"比 SpConv v2 快 1.2–1.3×"，Fig.15 图注写"A100 上 1.16×、2080Ti 上 1.27×"（1.16× 不在 1.2–1.3 区间内），Conclusion 又写"1.2–3.7× faster training"，而 Sec.5.2 对 ME 是 4.6–4.8×。
  - 工程成本：Abstract 写"<1/10"，Sec.6.3 写"5% 代码行"，Sec.3.1 写"hundreds of lines"，三处口径不同。
- **与 PointAcc 比较是投影**：
  - 假设放大到 128×128 的 PointAcc-L 获得线性加速。
  - 把 GPU 实测延迟乘以 2.2（频率差 1.7 × 峰值 MAC 差 1.3）再比较。
  - "56% of ASIC speed"依赖这两个假设，而且只测了一个负载（SK-MinkUNet）。
- **检测任务只计 SparseConv 层**（Sec.5.1），不是完整端到端；TorchSparse 对 CenterPoint 至少说明了另有 10% 的 2D/NMS 开销。
- **对 FlatFormer 的 1.5× 结论是附带一句话**：没有表格，也没有说明测量设置，只有"assuming the 2D part is deployed with TensorRT"。
- **调优的公平性**：TorchSparse++ 对每个负载、每个硬件都做 group-level 的穷举调优，SpConv 2.3.5 用的是其自带策略。调优成本约 2 分钟，作者认为可以接受，但没有给基线同等的调优预算。
- **训练精度不变没有证据**。混合精度训练的收敛性正是审稿人最可能追问的地方。

## 7. 可迁移手法（3–6 条）
1. 当领先方案的性能来自昂贵的工程实现时，把"工程成本"（代码行数）作为一条与性能并列的主张，并证明低成本方案在同等配置下不慢。**直接读到**（Sec.2.3, Fig.23）
2. 用"kernel 级结论与端到端结论方向相反"的成对表格，推翻领域内常用的一阶代理指标。**直接读到**（Table 3 vs Table 4, Fig.17）
3. 缩小设计空间之前，先做一个理想化的上界实验，证明缩小后的空间不损失性能。**直接读到**（Fig.8）
4. 续作沿用前作的完全相同的 benchmark 和图表版式，并把前作本身列为基线，使"代际提升"一目了然。**直接读到**（Sec.5.1, Fig.14）
5. 在 GPU 上做"带宽减半 vs 算力减半"的敏感性实验，把系统论文的发现转化为对硬件设计的建议。**直接读到**（Sec.6.3）
6. 与 ASIC 比较时，按频率和峰值算力做显式换算，并写明线性扩展的假设。做法透明，但结论的强度受假设限制。**直接读到**（Table 2）

## 8. 一句话
用"排序让 kernel 更快却让端到端更慢"这组反例推翻一阶代理指标，再以 5% 代码量的生成器加扩大的调优空间，在前作完全相同的 7 个负载上超越包括自家 TorchSparse 在内的所有基线。
