# HAQ: Hardware-Aware Automated Quantization with Mixed Precision

全文已读完（正文 9 页，无附录）。**文本损坏严重**：Table 3–6 与 Fig 3–5 在 pdftotext 中与正文交错，下文表格数字是我按行列对齐重建的，凡有疑义处已标注；Fig 3–5 的逐层位宽只能读到图注与少量标签。

## 0. 元信息
CVPR 2019（Oral）。Wang*, Liu*, Lin*, Lin, Han。类型：算法（RL 搜索混合精度）+ 硬件在环（加速器模拟器）。

## 1. 核心主张
- 主主张：用 DDPG 逐层决定权重/激活位宽，并以硬件加速器的直接反馈（而非 FLOPs/模型大小代理）作约束，在同等延迟/能耗/模型大小下比固定位宽量化精度更高；对比 8bit 可降延迟 1.4–1.95×、能耗 1.9×（Abstract）。
- 次主张：(a) 一种硬件上的最优策略在另一硬件上不是最优（Table 1）；(b) 学到的策略可解释——edge 上 depthwise 激活少给位、cloud 上反之，可用 roofline 解释（Fig 3/4, Sec 4.1.1）；(c) 不同优化目标（延迟 vs 模型大小）下策略显著不同（Fig 4 vs Fig 5）。

## 2. 动机是怎么立住的
- Fig 1：MobileNet 混合精度在延迟-精度坐标上的 Pareto 曲线优于固定 8bit——用结果本身作为动机图。
- Table 1（最关键）："为 HW1 最优的策略在 HW2/HW3 上不是最优"的 3×3 交叉矩阵（如 HW1 策略在 HW1 16.29ms，HW2 专属策略在 HW1 19.95ms；HW3 专属在 HW3 99.68ms，HW1 专属在 HW3 117.44ms）。对角线最优——这个**交叉矩阵**直接立住"需要硬件专用"这一卖点。
- 代理信号不准：Sec 3.3 用文字论证（cache locality、kernel 调用、带宽），引用 Table 1；**没有单独展示 FLOPs 与实测延迟的不相关散点图**。
- 设计空间之大：O(H×M×8^{2N})，ResNet-50 约 8^100（Sec 1）——用组合数论证不能手调。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 硬件专用策略必要 | 3 策略 × 3 硬件交叉 | Table 1 | 彼此 | 延迟 ms |
| 延迟约束下优于固定位宽（时序架构） | BISMO edge/cloud × MBv1/v2 × 4/5/6bit | Table 3 | PACT（首末层 8bit 设定对齐） | Top-1/5、延迟 |
| 延迟约束下优于固定位宽（空间架构） | BitFusion MBv1 | Table 4 | PACT 4/4、6/4、6/6 | Top-1/5、延迟 |
| 能耗约束 | BitFusion MBv1 | Table 5 | PACT | Top-1/5、mJ |
| 模型大小约束 | MBv1/v2、ResNet-50 × 2/3/4bit | Table 6 | Deep Compression（k-means） | Top-1/5、MB |
| 策略可解释 | 逐层位宽 + OPs/Byte | Fig 3/4/5 | edge vs cloud；延迟 vs 大小目标 | 位宽 |
| RL 比其他搜索方式好 | 「未直接验证」 | — | — | — |
| 代理信号不如直接反馈 | 「未直接验证」（仅 Table 1 间接） | — | — | — |

重建出的关键数字（Table 3, BISMO edge MBv1）：PACT 4bit 62.44% @45.45ms vs HAQ 67.40% @45.51ms；6bit 70.46% @70.67ms vs 71.20% @70.35ms；原 8bit 70.82%（延迟数字损坏，约 96.20ms）。Table 4 BitFusion：HAQ 70.40% @11.09ms vs 8bit 70.82% @20.08ms。Table 6：MBv1 2bit Deep Compression 37.62% vs HAQ 57.14%（1.09MB）；ResNet-50 4bit 两者 76.15/76.14 持平。

## 4. 实验设计手法
- 基线：PACT（固定位宽，SOTA 低比特训练法）和 Deep Compression（k-means，作者自家）。对齐方式是**同延迟/同能耗/同模型大小下比精度**——每一行都把资源量拉齐到几乎相同（45.45 vs 45.51ms）。没有朴素随机搜索或规则启发式（如"首末层多位"以外的手调）作对照；未与同期混合精度搜索方法比较。PACT 与 Deep Compression 在 MobileNet 上的数字应是作者自跑（原文未覆盖 MobileNet），但未说明训练预算是否对齐。
- 规模/广度：主要 MobileNet-V1/V2，模型大小实验加 ResNet-50；只有 ImageNet 分类；3 种硬件配置（BitFusion、BISMO edge Zynq-7020、BISMO cloud VU9P，Table 2）。
- 效率测量：Abstract 明说使用**hardware simulator**；正文叫"direct feedback from hardware accelerators"。延迟/能耗是模拟器/FPGA 设计给出的数字，是否在真实 FPGA 上跑未交代清楚。batch 固定（edge 1、cloud 16）。
- 精度–效率联合：同资源比精度（Table 3–6 逐行对齐）；Fig 1 Pareto 曲线。三种约束（延迟、能耗、大小）逐一覆盖，形成"同一框架多目标"的叙事。
- 消融：**无**。RL agent、状态特征、KL 裁剪、1-epoch 微调代理奖励、100 类子集都没有消融。
- 分析性实验：Fig 3/4 逐层位宽配 OPs/Byte（roofline），解释 depthwise 是访存受限；Fig 5 解释"参数少的 depthwise 层在大小约束下多给位"；下采样层多给激活位（Sec 4.1.1）。这些是**事后解释**，没有反事实（例如强制交换策略看是否变差，Table 1 部分承担此功能）。
- 泛化：跨 3 硬件、3 目标、3 模型；无跨任务。
- 统计：无方差、无多次 RL 搜索的稳定性报告。
- 系统贡献：未提开源（正文未见代码链接）；BISMO/BitFusion 设计开源链接为他人。
- 局限性：无。

## 5. 对抗式审读
- 最薄弱环节："自动化 RL 优于人"的主张没有任何搜索算法对照（随机搜索、进化、贪心逐层敏感性）。超过 PACT 的增益可能来自"允许混合精度"而非"RL"——这两者被捆在一起。
- 硬件测量不透明：模拟器 vs 实机；能耗数字来源（Table 5）未说明测量方式。
- 不利数据被平淡带过：Table 5 第一行 HAQ 13.69mJ 反而高于 PACT 13.47mJ；Table 6 4bit 时与 Deep Compression 基本持平（ResNet-50 76.14 vs 76.15），增益集中在极低比特。
- 搜索用 100 类子集 + 1 epoch 微调作奖励，代理与最终结果的相关性未验证。
- Abstract 的"1.9× 能耗"与 Table 5 对应关系（31.03→16.57mJ 还是 16.30mJ）在文本中不清。

## 6. 可迁移手法
1. 要证"需要针对 X 专用化"，做 N×N 交叉矩阵：为每个目标优化的方案在所有目标上评测，展示对角线最优（Table 1）。—直接读到
2. 对比时把资源量拉齐到小数点级（同延迟比精度），每个资源水平各一行，形成多点对比而非单点（Table 3–5）。—直接读到
3. 搜索类方法用"学到的策略可视化 + 领域理论（roofline）解释"把黑箱结果变成 insight（Fig 3/4）。—直接读到
4. 同一框架在多种约束目标（延迟/能耗/大小）上各跑一次，并展示策略随目标改变，以证明"它在适应约束而不是撞到一个固定解"（Fig 4 vs Fig 5）。—直接读到
5. 用自家旧方法（Deep Compression）当基线并在极端设定（2bit）下大幅领先，同时承认在温和设定持平。—我的归纳

## 7. 一句话
Table 1 那个 3×3 "策略-硬件"交叉延迟矩阵，用最少的数字让"硬件专用量化"成为必需，后续所有同资源比精度的表都建立在它上面。
