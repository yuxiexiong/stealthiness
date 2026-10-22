# QOC: Quantum On-Chip Training with Parameter Shift and Gradient Pruning

## 0. 元信息
- DAC 2022（arXiv 2202.13239v3）。Hanrui Wang*, Zirui Li*, Jiaqi Gu, Yongshan Ding, David Pan, Song Han
- 类型：算法 + 实验演示（PQC 在真实量子芯片上用 parameter shift 训练，加概率梯度剪枝）
- 阅读完整性：全文（含参考文献）读完，无附录。Fig 2/6/7/8 曲线数值错位，只引用正文写出的数字。

## 1. 核心主张
- 主：PQC 训练可以完全搬到真实量子机器上（parameter shift 求梯度、经典端只做 loss→logit 反传），并在 5 台真实机器、5 个分类任务上得到接近无噪声模拟的精度（2 类 >90%，4 类 >60%）。
- 次：真实噪声下小梯度相对误差大；按累计梯度幅值做"概率梯度剪枝"（PGP）既能提精度（摘要：最多 +7%）又能省一半梯度电路执行次数（"2×"）。

## 2. 动机是怎么立住的
- Fig 2(a)：经典模拟 #Ops/#Regs 随 qubit 指数增长（理论曲线），立"必须上芯片训练"的可扩展性痛点。
- Fig 2(b)：Santiago/Casablanca 上 on-chip 训练与经典无噪声训练的精度曲线存在"noise-induced gap"。
- 关键观察 Fig 2(c)：梯度幅值 vs 平均相对误差（对数轴），小梯度相对误差大、甚至方向错误。另有一条未给图的经验观察："梯度幅值大/小在若干步内保持"（Sec 3.3）——这是 accumulation window 设计的直接前提，但正文没有给出量化证据。
- 因果链：Fig 2(c) → 剪掉小梯度。闭合，但"可预测性"那一环只是文字断言。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| on-chip 训练可行且接近无噪声 | 5 任务，Jakarta(×2)/Manila/Santiago/Lima | Table 1 | Classical-Train（模拟测/上机测）、QC-Train | 实测验证集准确率 |
| PGP 提精度 | 同上 | Table 1 | QC-Train | MNIST-4 0.59→0.64、MNIST-2 0.83→0.86、Fashion-4 0.49→0.57、Fashion-2 0.84→0.91、Vowel-4 0.34→0.36 |
| PGP 收敛更省电路执行 | Fashion-2 Santiago, Fashion-4 Manila | Fig 6 | QC-Train, Classical-Train | 13.9k 次推理达 90.7%，Classical-Train 30k+ 次最高 88.7% |
| 超参稳健 | 剪枝率/两窗口宽度扫描 | Fig 7 | — | 经典验证集准确率 |
| 概率优于确定性剪枝 | 4 任务 | Table 2 | top-k 剪枝 | 1–7% |
| Adam 最优 | 4 任务 | Table 3 | SGD, Momentum | 经典测得准确率 |
| 可扩展性 | 16 旋转+32 RZZ 电路 | Fig 8 | 经典 GPU 模拟 | 运行时/内存 |

## 4. 实验设计手法
- 基线：Classical-Train（在模拟器训练，分别在模拟器和真机上测）与 QC-Train（无剪枝）——一个是"现状"基线，一个是 ablation 基线；Table 2 的确定性剪枝是设计选择对照。无外部 SOTA（如 SPSA、其他噪声鲁棒训练）。
- 规模/广度：5 个 4-qubit 任务；训练集极小（2 类 500 张、4 类 100 张），验证集随机 300 张；每个任务只在一台机器上跑。
- 测量：全部训练与评估在真实 IBMQ 硬件上，1024 shots。Fig 8 的运行时/内存：经典 ≤22 qubit 在 RTX 2080Ti 实测、≥24 qubit 外推；量子 ≤27 qubit 在 ibmq_toronto 实测、≥30 qubit 外推。
- 精度–效率：Fig 6 以"#inference（电路执行次数）"为横轴画精度曲线——用资源预算而非步数对齐，是本篇最有意思的呈现方式。
- 消融：剪枝率、累计窗口、剪枝窗口三参数扫描（Fig 7）；概率 vs 确定性（Table 2）；优化器（Table 3）。
- 分析：Fig 2(c) 梯度幅值–误差关系。
- 泛化：跨任务、跨机器有，但"机器"和"任务"是一对一绑定的，没有交叉。
- 统计：无重复、无误差棒。
- 系统：代码进 TorchQuantum。
- 局限：无局限性讨论。

## 5. 对抗式审读
- 最薄弱：评估集就是验证集，Table 1 "accuracy is collected after finishing a certain number of circuit runs"，Fig 6 报告"peak accuracy 90.7%"——即在同一 300 张验证集上取峰值，没有独立测试集；而 Classical-Train 的对照点是什么时刻取的不清楚。
- 摘要/贡献/实验开头给出的增益数字互相不一致："up to 7%"、"5–7%"、"2–4% ... 2× convergence speedup"（Sec 4 开头）；Table 1 中 Fashion-4 实际 +8 点。
- QC-Train 在 Fashion-4、Fashion-2 上反而不如 Classical-Train 上机（0.49 vs 0.54；0.84 vs 0.89），说明"朴素 on-chip 训练"本身未必有利，结论高度依赖 PGP；而 PGP 的超参（Fashion-4 用 r=0.7，其余 0.5）按任务调过。
- Fig 8 "在超过 27 qubit 的电路上观察到清晰的量子优势"——27 qubit 以上量子侧是外推、24 qubit 以上经典侧也是外推，且运行时口径未说明是否含排队；这一结论超出实验支持。另外经典侧用的是单 GPU 状态向量模拟，没有考虑张量网络等更强模拟器。
- Table 3 的优化器比较在经典设备上测，与本篇"on-chip"主线不一致；Fashion-2 SGD 记为"76"疑似笔误。
- 单次运行、单机绑定，增益多在 2–8 点，噪声漂移下的方差未知。

## 6. 可迁移的手法
1. 当方法的卖点是"省资源"，把横轴换成真实资源消耗（电路执行次数），在同一预算下比精度（Fig 6）。—直接读到
2. 用"信号幅值 vs 相对误差"一张图把"小信号不可靠"的直觉量化，然后直接导出"丢弃小信号"的方法（Fig 2(c) → Sec 3.3）。—直接读到
3. 引入随机化的设计选择时，给出确定性版本作对照（Table 2）。—直接读到
4. 对多窗口/比例类超参做一维扫描并解释两端失效机理（Fig 7 文字）。—直接读到
5. 报告"四档"精度：模拟训练模拟测、模拟训练真机测、真机训练、真机训练+方法——把"差距从哪来"拆清楚（Table 1）。—我的归纳

## 7. 一句话
用"真机训练 vs 模拟训练上真机"的四档对照表加上以电路执行次数为横轴的训练曲线，把"上芯片训练可行且更省"讲成一个看得见的事实；但量子优势外推和验证集取峰值是明显软肋。
