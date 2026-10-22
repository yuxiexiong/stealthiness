# G9 组内综合：量子计算 ML/编译 + 模拟电路自动设计

覆盖 8 篇：quantumnas (HPCA'22)、quantumnat (DAC'22)、qoc (DAC'22)、quest-q (ICCAD'22)、atomique (ISCA'24)、q-pilot (DAC'24)、gcn-rl-circuit-designer (DAC'20)、learning-to-design-circuits (NeurIPS'18 WS)。
以下"出现位置"均指各篇笔记中已核对过的表/图/节。只列出现 ≥2 次的为"共同套路"。

## 先说坏消息（本组反复出现的薄弱点）

1. **几乎没有统计重复**。8 篇里只有 GCN-RL 每个实验跑 3 次并报 mean±std（Table I/IV/V）。其余量子论文的真机结果都是单次测量、无误差棒（QuantumNAS Fig 13–15，QuantumNAT Table 1，QOC Table 1，QuEst Fig 7）。而 QuantumNAS 的 artifact 附录自己写"允许复现差异 <10%"，许多被报告的增益只有 1–5 个点。
2. **在评估集上做选择**，至少 4 篇出现：QuantumNAS 剪枝比例"在 5 个比例中报告最高的真机实测精度"（Sec IV-A）；QOC 在 300 张验证集上报"峰值精度"、没有独立测试集（Table 1, Fig 6）；QuantumNAT 归一化用测试批自身统计，到附录 Table 13 才补验证集统计（0.67→0.65）；QuEst 算法电路同结构跨训练/测试集，疑似泄漏（Sec 5.1）。
3. **摘要/引言/正文的关键数字口径不一致**，6 篇出现：QuantumNAS 设备数 14 vs 10；QuantumNAT 6 台 vs 5 台；QOC 增益"up to 7% / 5–7% / 2–4%"；Q-Pilot QAOA 深度 6.3× vs 6.7×；Atomique 编译加速 1000× vs 2,000×；QuEst "5 thousand circuits"（测试集）vs 45K（数据集）。
4. **代理指标 ≠ 真实目标，且结论随代理而变**：Q-Pilot 只报深度（并行 2Q 层数，不计移动时间），同组 Atomique Fig 19 随后显示 Q-Pilot 深度更低但保真度更差；QuEst 用 PST 代替保真度，代理性只在模拟器上验证（Fig 2）；同一个两级 TIA 人类设计，L2DC 用竞赛目标时 RL 略输（Table 2, −0.03 vs 0.00），GCN-RL 换成等权归一化 FoM 后变成"胜过专家"（Table I, 2.69 vs 2.32），且 GCN-RL 的设计带宽只有人类的约 1/6（Table II）。
5. **可扩展性主要靠外推或缩放**：QuantumNAS 127 qubit 运行时线性外推 57 h（Fig 16）；QOC 量子优势结论落在双方都外推的区间（>27 qubit，Fig 8）；Atomique 所有平台相干时间 ×10、门误差 ÷10（Sec V-A）；QuEst 的"多项式 vs 指数"只在 ≤10 qubit 上实测。
6. **基线偏弱或缺朴素物理基线**：QuantumNAT 只与"无噪声感知训练"比；QOC 只与经典训练和自身无剪枝版比；QuEst 只有一个简单 NN 基线，缺门成功率乘积这种同组 QuantumNAS 就用过的解析估计器；Q-Pilot 没有与同期 FPQA 启发式编译器做数值对比。

## 共同的实验设计套路（≥2 篇）

| 套路 | 具体做法 | 出现位置 |
|---|---|---|
| A. 开篇"关键观察图"，把现象和方法直接连起来 | 理想 vs 真实曲线、误差 vs 信号幅值、分布偏移可视化 | QuantumNAS Fig 2/3（无噪声精度升、真机精度先升后降）；QuantumNAT Fig 1 + Fig 4（错误率 vs 精度；归一化前后分布 + SNR）；QOC Fig 2(c)（梯度幅值 vs 相对误差） |
| B. 代理先验证再使用 | 用相关性散点证明代理≈真值，再全程用代理 | QuantumNAS Fig 9（继承参数 vs 从头训练，Spearman 0.75）+ Fig 10（模拟 vs 真机 0.76）+ Table IV；QuantumNAT Table 11（噪声模型 vs 真机差距 <5%）；QuEst Fig 2（PST vs 保真度 0.993） |
| C. 交叉部署矩阵证明"必须定制" | 在 A 上搜索/训练，在 B 上运行，对角线占优 | QuantumNAS Table III；QuantumNAT Table 6 |
| D. 累加式 breakdown / roadmap | Baseline → +A → +A+B → …，或"只 A / 只 B / A+B" | QuantumNAT Table 1、Fig 9；QuantumNAS Fig 19；Atomique Fig 21（3.53×→1.19×→2.59×，合 10.9×） |
| E. 去掉结构信息的同款模型做因果对照 | 保留其余一切，只去掉图/噪声/结构 | GCN-RL 的 NG-RL（Table I, V）；QuEst 的简单 NN（Fig 10）与逐特征删除（Table 2）；QuantumNAS 的 noise-unaware 搜索（Fig 13, Table II） |
| F. 多台真实设备铺广度 | 把"跨设备"本身当作泛化证据 | QuantumNAS（约 14 台 IBMQ，Fig 13–15, 18）；QuantumNAT（5–6+ 台，Table 1、附录）；QOC（5 台）；QuEst（5 台 IBM 真机） |
| G. 超参一维/二维扫描并解释"先升后降"的甜点 | 给出两端失效的机理 | QuantumNAS Fig 23（剪枝比例）；QuantumNAT Fig 8（噪声因子 × 量化级）；QOC Fig 7（剪枝率、窗口）；Atomique Fig 18（每步移动时间、冷却阈值）；Q-Pilot Fig 14（阵列宽度） |
| H. 与精确求解器比较：小规模比质量、大规模比可扩展性、设超时 | 求解器超时就标 timeout | Atomique Fig 14 + Table II（24 h）；Q-Pilot Table 2（4000 s） |
| I. 三层基线：跨平台 / 同平台跨架构 / 同架构跨编译器 | IBM 127Q heavy-hex、FAA 矩形、FAA 三角，均 Qiskit O3 | Atomique Fig 13；Q-Pilot Fig 11–13 |
| J. 外部可核查的人类专家基线 | 课程项目/竞赛获奖设计、注明经验与工时 | L2DC Table 1, 2；GCN-RL Table I（同一批 Stanford EE214 设计）；QuantumNAS 的人工设计与 UCCSD |
| K. 不同代价的优化器按同墙钟时间对齐 | BO 因 O(N³) 只能跑较少样本，脚注说明 | L2DC Table 1 脚注；GCN-RL Sec IV-A |
| L. 声明与现有技术正交并做叠加实验 | 把别人的方法当"可叠加项"而非对手 | QuantumNAT Table 4（外推）、Table 7（Qiskit O3）；QuantumNAS Table IV（O2/O3） |
| M. 以真实资源消耗为横轴 | 电路执行次数而非迭代步数 | QOC Fig 6（#inference）；QuantumNAS Fig 16（#QC runs） |
| N. 受限资源下用小测试子集，并抽查其代表性 | 300 张图 | QuantumNAS（用 4 个电路对比全测试集）；QuantumNAT、QOC（未抽查） |
| O. 主动标出方法失效区 | 承认在某些设置下输 | QuantumNAS（Vowel-4 ZZ+RY、MNIST-4 ZX+XX）；Atomique（低度电路、2Q 保真 >0.9999 时 FAA 反超，Fig 18f）；Q-Pilot Table 2（深度不超过最优 4×） |
| P. 开源进统一框架 / artifact | TorchQuantum；Zenodo artifact | QuantumNAS、QuantumNAT、QOC、QuEst（TorchQuantum）；QuantumNAS、Atomique（ACM artifact 附录） |

## 本组特有的套路

- **真机 + 噪声漂移叙事**：明确交代 shots（8192/2048/1024）、Qiskit 优化级别、IBM 每日校准；QuantumNAS 做"搜索后 3 周再测"（Fig 14）——这是唯一的时间重复。设备退役导致缺格（Athens 在 QuantumNAS Fig 14 与 QuantumNAT Table 1 都缺），作者如实标出。
- **无实机时自建物理分项模型 + 敏感性扫描**（Atomique Sec IV、Fig 18；Q-Pilot Eq.5）：参数从单篇 Nature 实验 [10] 推导，并给出解析 breakeven（n_vib<18 时移动优于 SWAP）。
- **约束放松实验**：逐条放松硬件约束看收益（Atomique Fig 22）。
- **电路特征空间曲面**：按"每比特门数 × 交互度/局部性"画性能曲面回答"哪类负载受益"（Atomique Fig 15–17；Q-Pilot 按门密度/Pauli 概率/边概率扫，Fig 11–13）。
- **跨领域借形式化**：NAS 的 SuperNet → SuperCircuit（QuantumNAS）；Batch Norm/量化/噪声注入 → 测量后归一化/量化（QuantumNAT）；剪枝 → 梯度剪枝（QOC）；FPGA 布局布线 → 飞行辅助比特（Q-Pilot）；"电路是图" → GCN/图 Transformer（GCN-RL、QuEst）。几乎每篇都把 Han Lab 在经典 DL 效率上的工具直接移植过来，这是本组最一致的选题来源。
- **理论小结果给方法背书**：QuantumNAT 定理 3.1（噪声 = 线性缩放+平移）；Q-Pilot ancilla 等价性证明；QOC parameter shift 推导。

## 要点（给汇报用）

1. 本组的核心说服力来自"真机实测 + 多设备铺开"（F），但统计上几乎全是单次测量，增益与自承的测量波动同量级——这是审稿人最容易攻击的地方。
2. "代理验证 → 用代理"（B）是本组最成熟的套路，三篇用 Spearman/R² 散点为代理正名；但代理与最终目标的偏差（深度 vs 保真度、FoM vs 竞赛目标）在 Q-Pilot 和 GCN-RL 上实际翻转过结论。
3. 交叉部署矩阵（C）和累加式 breakdown（D）是两种低成本、高说服力的表格形式，值得直接借用。
4. 编译器两篇（Atomique、Q-Pilot）在无实机条件下用"三层基线 + 求解器超时对比 + 参数敏感性 + 主动标出反转点"建立可信度；Atomique 做得比 Q-Pilot 完整得多（后者只报深度）。
5. 模拟电路两篇的亮点是外部可核查的人类基线和同墙钟预算对齐；GCN-RL 是本组唯一报方差的论文。但"胜过专家"依赖自定义 FoM。
6. 在评估集上做选择（剪枝比例、峰值精度、测试批统计）在量子 ML 三篇里反复出现，是可复现性层面最需要警惕的共性问题。
7. 选题几乎都是"把经典 DL 效率工具嫁接到量子/模拟电路"，属跨领域嫁接，产出的是方法而不是跟进——这解释了为何单篇实验不算严密但仍被顶会接收：新问题设定本身占了很大权重。

## 边界申报
- 所有数字取自 pdftotext 文本；多数柱状图/曲面图数值错位，笔记只用正文明确写出或表格可对齐的数字。QuEst 的 Table 3 与 Fig 10 数据混排，我按行标签拆分，存在错配可能（已在该篇笔记中标注）。
- "共同套路"的判定基于这 8 篇；未与其他组的 Han Lab 论文交叉核对。
