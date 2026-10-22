# EIE: Efficient Inference Engine on Compressed Deep Neural Network

## 0. 元信息
- ISCA 2016（arXiv 1602.01528v2）；Han, Liu, Mao, Pu, Pedram, Horowitz, Dally（Stanford/NVIDIA）
- 类型：硬件（算法-硬件协同：在 Deep Compression 产出的稀疏+4-bit 权重共享模型上做 ASIC）
- 实现层级：cycle-accurate C++ 仿真器（做设计空间探索、并作 RTL 检查器）+ Verilog RTL + Synopsys DC 综合（TSMC 45nm GP，worst-case PVT）+ 单个 PE 的 ICC 布局布线（Fig.5）；SRAM 面积/能耗用 Cacti；功耗由 RTL 仿真翻转率 → SAIF → PrimeTime PX。**未流片**。
- 全文读完（正文+参考文献；无附录）。

## 1. 核心主张
- 主：直接在压缩模型（剪枝稀疏 + 4-bit 码本 + 动态激活稀疏）上做 M×V 的专用加速器，在 batch=1 的 FC/LSTM 层上比 CPU/GPU/mGPU 快 189×/13×/307×、省能 24,000×/3,400×/2,700×（Abstract, Fig.6/7）。
- 次：(a) 节能可分解为 SRAM 替代 DRAM 120× × 稀疏 10× × 权重共享 8× × 激活稀疏 3×（Sec.VI-B）；(b) 交错 CSC + 激活队列可实现负载均衡和近线性扩展（Fig.8/11）；(c) 相比 DaDianNao 吞吐 2.9×、能效 19×、面积效率 3×（Table V，28nm 投影）。

## 2. 动机
- 痛点用"能耗表"立：Table I（45nm，引 Horowitz）DRAM 访问 640pJ vs SRAM 5pJ vs 加法 0.1pJ；并算一笔账：1G 连接网络 20Hz 仅 DRAM 就 12.8W（Sec.I）。这是**第一性原理式**动机，不需要 profiling。
- 第二个痛点：压缩模型在 CPU/GPU 上跑不快。Table IV 实测墙钟：batch=1 时 GPU sparse 对 dense 仅有限加速，"model compression by itself applied on a CPU/GPU yields only 3× speedup"（Sec.VI-A）。
- 关键观察：FC 层无权重复用，batch=1 时内存瓶颈；激活 ~70% 为零（Table III Act% 列）。因果链闭合：内存瓶颈 → 压缩以装入 SRAM → 不规则性需专用硬件。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 速度优势 | 9 个 FC/LSTM 层 | Fig.6, Table IV | i7-5930k(MKL)、Titan X(cuBLAS/cuSPARSE)、Tegra K1，各 dense/sparse | 墙钟、加速比（几何平均） |
| 能效优势 | 同上 | Fig.7 | 同上 | 能效 = 时间×功率 |
| 节能分解 120×10×8×3 | **算术乘积**，非逐项消融 | Sec.VI-B | — | 理论 28,800× vs 实际约少 10× |
| 设计选择合理 | FIFO 深度、SRAM 宽度、算术精度扫描 | Fig.8/9/10 | 自身变体 | 负载均衡率、读能耗、精度 |
| 可扩展 | 1–256 PE | Fig.11/12/13 | 自身 | 加速比、padding 零、负载均衡 |
| 优于其他加速器 | AlexNet FC7 | Table V | A-Eye, DaDianNao, TrueNorth | Frames/s、/mm²、/J |
| 精度无损 | 16-bit 定点 79.8% vs fp32 80.3% | Fig.10 | — | ImageNet top-? 精度 |

"激活稀疏省 65.14% 计算"有数字支持；"节能四因子分解"是乘法估算，**未逐项消融验证**。

## 4. 实验设计手法
- 基线：同时给"dense 和 compressed"两种 CPU/GPU 实现（Fig.6 七列），用厂商最优库（MKL SPBLAS、cuSPARSE）——这堵住了"你只比了 dense"的攻击。朴素基线即 dense CPU（归一化为 1×）。
- 规模：9 层取自 AlexNet、VGG-16、NeuralTalk（Table III），只有 FC/LSTM；无 CONV。
- 效率测量：EIE 侧是仿真器周期 × 综合功耗；CPU 用 pcm-power，GPU 用 nvidia-smi，TK1 用功率计并**假设** 15% AC-DC 损耗、85% 调压效率、15% 外设功耗（Sec.V）。理论时间 vs 实际时间并列（Table IV），差约 10%（负载不均）。batch=1 为主，batch=64 作补充（Table IV）——承认 batch=64 时 Titan X dense 更快或可比。
- 精度–效率：Fig.10 精度 vs 乘法器能耗（32f/32i/16i/8i）。
- 消融：设计空间探索三组（队列深度、SRAM 宽度、精度）；无"逐个去掉稀疏/共享"的 breakdown 实验。
- 分析：Table II 面积/功耗按组件与模块分解（SRAM 占面积 93%、功耗 59%）；Fig.12/13 解释扩展性背后的两股力量（padding 零↓ vs 负载均衡↓）；NT-We 小矩阵扩展差被主动指出。
- 泛化：仅 FC/LSTM；Sec.VII-C 口头声称可支持 1×1 conv/Winograd，**未验证**。
- 统计：无方差（确定性仿真）。
- 系统贡献：无开源 RTL；无流片。
- 局限：Discussion 讨论了分区方案取舍、NT-We 扩展性差、45nm vs 28nm 的不公平（Sec.VI-B 解释节能比理论少 10×）。

## 5. 对抗式审读
- 最薄弱：**跨工艺/跨实现层级比较**。EIE 是 45nm 综合后估算（只 PE 布局布线，整芯片未 P&R），对比对象是 28nm/22nm 实测硅片；Table V 的 "EIE 28nm 256PE" 列是**投影**，投影方法正文未给出公式。
- 24,000×/3,400× 能效：GPU 用 nvidia-smi 整卡功率、batch=1 跑 GEMV，GPU 利用率极低，比较对 GPU 极不利；batch=64 下 GPU dense 与 EIE 时间相当（Table IV），但能效图只给 batch=1。
- DaDianNao 的 M×V 性能是按峰值带宽**估算**而非运行（Sec.VIII）。
- "精度无损"依赖 Deep Compression 前作，本文只测了 AlexNet 的 16-bit 损失 0.5%。
- 只覆盖 FC 层，而 2016 年 CNN 计算主要在 CONV（作者在回顾论文中承认）。

## 6. 可迁移手法
1. 在"内存主导能耗"的场景，先用单位操作能耗表做一笔数量级账（Table I + 12.8W 计算）立动机，再设计硬件。**直接读到**
2. 对比通用硬件时，把基线也跑在同一压缩模型上（CPU/GPU compressed 列），证明收益来自硬件而非仅来自压缩。**直接读到**（Fig.6）
3. 用"理论时间 vs 实际时间"并列来量化调度/负载不均损失（Table IV）。**直接读到**
4. 每个关键微结构参数做一维扫描并在拐点处选值（FIFO=8、SRAM 64-bit），图注直接写选择理由。**直接读到**（Fig.8/9）
5. 主动展示失败案例并给出机理（NT-We 扩展差，Fig.11–13），增加可信度。**我的归纳**
6. 把总收益写成若干独立因子的乘积（120×10×8×3），让读者看到收益来源——但这是估算，不是消融。**直接读到**（Sec.VI-B）

## 7. 一句话
用能耗表的数量级账把"内存是瓶颈"钉死，再把 CPU/GPU 也跑压缩模型来证明"只有专用硬件才能兑现压缩的收益"。
