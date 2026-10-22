# QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits

## 0. 元信息
- HPCA 2022（arXiv 2107.10845v5）。Hanrui Wang, Yongshan Ding, Jiaqi Gu, … Fred Chong, Song Han
- 类型：算法-系统协同（电路结构搜索 + qubit mapping 共搜 + 剪枝）+ 开源库（TorchQuantum）
- 阅读完整性：全文含 Artifact 附录与参考文献已读。图表数据层（Fig 9/13/14/15/18/19/21/22 等）在 pdftotext 中严重错位，柱状图数值只能部分对上，下文只引用正文中明确写出的数字或能对齐的表格。

## 1. 核心主张
- 主：把噪声信息放进电路结构与 qubit mapping 的联合搜索环路，能在真实 IBMQ 机器上得到比人工设计、随机、无噪声感知搜索、以及噪声自适应 mapping 更鲁棒的变分电路（QML 准确率更高、VQE 能量更低）。
- 次：(a) SuperCircuit 权重共享可以可靠估计 SubCircuit 的相对排名，把搜索代价降约 Ndevice×Nsearch 倍；(b) front sampling / restricted sampling 稳定 SuperCircuit 训练；(c) 迭代细粒度剪枝在不降低无噪声精度的前提下减少门数从而提高实测精度；(d) TorchQuantum 比 PennyLane 快 104–246×。

## 2. 动机是怎么立住的
- 关键观察（Fig 2，MNIST-4，噪声无关模拟 vs 真实 IBMQ-Yorktown）：参数越多无噪声精度越高，但实测精度在 45 参数处达峰后下降；同参数量下实测精度方差远大于无噪声（45 参数时 [25%,59%] vs [67%,77%]）。这一张图同时给出"噪声抵消容量"和"同参数不同结构差异巨大 → 值得搜"两个理由。
- Fig 3 把结论图提前到引言：各基线的 accuracy-vs-#params 曲线很快饱和下降，QuantumNAS "推迟峰值"，最多高 33%。
- 可扩展性痛点用计数式论证：naive 搜索代价 Ndevice×Nsearch×(Ntrain+Neval) vs SuperCircuit 1×Ntrain+Ndevice×Nsearch×Neval，约 16,000× 缩减（Sec III-B，纯公式，非实测）。
- 因果链：动机图（噪声 vs 容量的权衡）→ 方法（噪声在环搜索 + 剪枝减少噪声源）→ Fig 3/19 回到同一权衡曲线验证，闭合较好。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| SuperCircuit 继承参数可代替从头训练做排名 | 5 任务×2 空间散点 | Fig 9 | 从头训练 | Spearman 平均 0.75 |
| 噪声模拟估计器可预测真实 QC | MNIST-4 Yorktown | Fig 10 | 真实 QC loss | Spearman 0.76 |
| 优于各基线（QML） | 5 任务×6 设计空间, Yorktown | Fig 13, Table II | noise-unaware / random / human / human+noise-adaptive mapping / human+Sabre / human(1/2参数)+Sabre | 实测准确率 |
| 跨设备成立、设备特定必要 | 4 台 5Q 机器；交叉部署 | Fig 14, Table III | 同上；搜索设备≠运行设备 | 准确率（对角线最好：0.85/0.77/0.85） |
| 噪声漂移鲁棒 | 搜索后 3 周再测 | Fig 14 | 立即测 | 略降但仍高于基线 |
| 可扩展到大机器 | 15/16/21/21 qubit 电路于 15–65Q 机器 | Fig 15 | 基线 | ">5% 更好" |
| 全流程可在 QC 上做 | parameter shift 训练 | Table V, Fig 16 | 经典训练 | 精度相当；127Q 外推 57 h |
| 用真实 QC 做搜索评估与模拟相当 | Fashion-4 五台机器 | Table IV | 估计器 | 准确率 |
| VQE 更低能量 | H2 各空间；LiH/H2O/CH4/BeH2 于 7–27Q 机器 | Fig 17, 18 | UCCSD 等 | 期望值 |
| co-search 必要 | breakdown | Fig 19 | 只搜 mapping / 只搜电路 | 共搜平均 +9% |
| front/restricted sampling 有效 | 5 任务 | Fig 20 | random sampling | 平均 +12% |
| 剪枝有效 | Table II（+3%），Fig 23 比例扫描 | — | 未剪枝 | 准确率 |
| TorchQuantum 快 | 10Q 100RX+100CRY | Fig 12 | PennyLane CPU | steps/s，104–246× |

## 4. 实验设计手法
- 基线：6 类，含朴素（random，同参数量、生成 3 个取最好）、人工设计（同参数量）、ablation 型（noise-unaware 搜索）、外部 SOTA mapping（Murali 噪声自适应 mapping [41]、Sabre [40]）、VQE 加 UCCSD。所有基线统一训练设置（Adam 5e-3、200 epoch，Sec IV-A）。注意：UCCSD 在 CH4-10Q 和 BeH2 上原电路 >10,000 门无法上机，只取前 1,000 门上机，全电路改用噪声模拟（Sec IV-A；Sec IV-B 给出全电路模拟 -12.86 / -9.81）。
- 规模/广度：12 基准（6 QML + 6 VQE），6 个设计空间，摘要称 14 台量子计算机；正文 Sec IV-A 写"study 10 devices, 5 to 65 qubits"，全文实际出现的机器约 14 台（Yorktown, Belem, Quito, Athens, Santiago, Rome, Lima, Melbourne, Guadalupe, Montreal, Manhattan, Casablanca, Jakarta, Manila）——数字口径不一致。
- 测量：主结果全部为真实 IBMQ 硬件实测（QML 8192 shots，VQE 2048 shots）；搜索阶段用 IBM 校准数据的 Qiskit 噪声模型（≤10 qubit）或"门成功率乘积"近似（>10 qubit）。测试集为 300 张随机抽样图片，用 4 个电路对比全测试集精度（0.505/0.497 等）论证代表性。
- 精度–效率：Fig 3 accuracy-vs-#params 曲线；Table II 同时列 depth/#gates/#params/acc；Table VI 深度 vs 精度。
- 消融：Fig 19 逐项（mapping-only / circuit-only / 共搜）；Fig 20 采样策略；Fig 22 random vs evolutionary 搜索；Table VI 小设计空间；Fig 23 剪枝比例扫描。
- 分析：Fig 21 拓扑（T/+/–）与错误率对性能和收敛迭代数的影响；Fig 9/10 估计器相关性。
- 泛化：跨任务、跨设计空间、跨 ~14 台设备、跨 QML/VQE。
- 统计：无误差棒、无多次重复；Fig 14 的"3 周后再测"是唯一的时间重复。Artifact 附录称允许复现差异"<10%"——这本身说明单次测量噪声不小。
- 系统贡献：开源 TorchQuantum（MIT License，Zenodo 归档），artifact evaluation 脚本齐全。
- 局限：几乎没有显式局限讨论；Artifact 附录承认噪声漂移需"搜完立即测"、多台机器已退役。

## 5. 对抗式审读
- 最薄弱：单次硬件测量、无方差。Artifact 自承 <10% 波动，而多数对比的差距在 1–5% 量级（如剪枝 +2%/+1%，Fig 13）——这些小差距未必超出噪声。
- 选择偏差：剪枝在 5 个最终比例中"报告最高实测精度"（Sec IV-A Iterative Pruning Setups），即在真实 QC 测试集上挑超参；random 基线"生成 3 个取最好"同样是挑选，但 QuantumNAS 的挑选是对测试集直接挑，不对称。
- 不公平：UCCSD 大分子被截断到 1,000 门上机，对比带有工程限制；人工设计基线是作者按"同参数量"自己堆的，未必代表各原文最优设计。
- 超出支持：Fig 16 的 127-qubit 57 h 是线性外推；"16,000× 降低搜索代价"是公式计数，非实测墙钟时间。设备数口径（14 vs 10）不一致。
- 搜索结果的输掉的案例（Vowel-4 ZZ+RY、MNIST-4 ZX+XX）正文有承认，这一点诚实。

## 6. 可迁移的手法
1. 在"资源 X 增加理论收益但带来副作用"的场景，用一张"理想环境 vs 真实环境随 X 变化"的双曲线图同时立住权衡与方差（Fig 2/3）。—直接读到
2. 用权重共享代理做搜索时，先用 Spearman 相关散点证明"代理排名≈真实排名"，再证明"模拟器排名≈真实硬件排名"，两级近似分别验证（Fig 9, 10）。—直接读到
3. 证明"必须针对目标设备定制"，用交叉矩阵（在 A 上搜、在 B 上跑），对角线占优即可（Table III）。—直接读到
4. 对随时间漂移的环境，补一个"隔几周再测"的时间鲁棒性实验（Fig 14）。—直接读到
5. 对联合优化，用三档 breakdown（只 A / 只 B / A+B）说明协同收益（Fig 19）。—直接读到
6. 当真实资源贵时，做一个小规模"真实硬件在环 vs 代理在环"的对照来为代理正名（Table IV, V）。—我的归纳

## 7. 一句话
用一张"无噪声精度上升、真实精度先升后降"的硬件实测图把问题钉死，再在约 14 台真实 IBMQ 机器上用同一条权衡曲线展示"峰值被推后"，并辅以交叉设备矩阵证明定制必要。
