# CAN 精读笔记

## 0. 元信息
- 标题：Condition-Aware Neural Network for Controlled Image Generation（CAN；新模型族 CaT）
- 来源：arXiv 2404.01143v1（2024-04）；文本中无会议信息（据我所知为 CVPR 2024，但文本未写，属未核实）
- 类型：算法（条件控制机制）+ 高效架构（CaT = CAN + EfficientViT）
- 正文读完；**无附录**。文本损坏较重：Figure 1、Figure 7、Figure 8 的数据点与 Table 2/3 在 pdftotext 中交错，并混入一串 epoch-FID 的图表源数据（"Untitled 1–4"、50→300 的序列）。Table 2/3 的数值我通过正文给出的差值（13.00、19.53、5.15、0.33）交叉核对后确认；Fig. 7/8 的曲线数值未引用。Table 6 中 "Friro" 疑为模型名的提取错误。

## 1. 核心主张
- 主主张：除"特征空间操纵"（AdaNorm、cross/self-attention）外，可以通过**按条件动态生成权重**（权重空间操纵）来给生成模型加控制，且效果显著更好、计算开销可忽略。
- 次主张：(a) 只让部分模块（DW conv、patch embedding、attention 输出投影）条件化，比全部条件化更好；(b) 直接生成权重优于自适应核选择（CondConv 式）；(c) CAN 可单独作为控制机制；(d) 改进不是收敛更快，而是上限更高；(e) CaT 在 ImageNet 512 上以 52× 更少的每步 MACs 超过 DiT-XL/2，Orin 上快 229×。

## 2. 动机怎么立住
- 动机是**概念性**的：不同条件（类别、timestep）对应不同目标分布，共享静态权重导致子任务间负迁移（引 Wu et al.）；专家模型太贵（Sec. 2.1）。**没有测量负迁移本身的动机实验**——没有显示"静态权重在某些条件上受损"的证据。
- 无可视化或量化的"关键观察"；论证完全靠结果（Table 1 从 28.32 到 8.82 的大幅下降）倒推机制有效。
- 因果链：Table 1/2 显示"条件化权重"有用，Fig. 4 排除"仅仅是动态参数化"这一替代解释，Fig. 8 排除"只是收敛更快"。这三个排除实验是本文论证的主体。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 选择性条件化 | 逐步加模块 + 替代分支（✗） | Table 1 | 静态基线（同架构含 DW conv） | FID 28.32→8.82，CLIP 30.09→31.74 |
| 直接生成 > 核选择 | 基核数扫描 | Fig. 4 | Adaptive Kernel Selection | FID |
| 跨模型尺度有效 | UViT/DiT 多尺寸 | Fig. 7 | 同尺寸基线 | FID、CLIP vs GMACs |
| 非收敛加速 | 训练曲线到 300 epoch | Fig. 8 | 基线 | FID-10K（无 cfg） |
| 类别比 timestep 重要 | 条件拆分 | Table 2 | timestep only / class only / all | FID、CLIP |
| CAN 可单独使用 | 替换控制机制 | Table 3 | AdaNorm（DiT）、Attention（UViT） | DiT-S/2：39.44→26.44；UViT-S/2：28.32→8.79 |
| SOTA 效率 | ImageNet 256/512 | Table 4 | ADM、LDM、UViT、DiT | FID-50K（有/无 cfg）、MACs/步、步数、参数 |
| 边缘设备实测 | Orin TensorRT fp16 | Table 5 | DiT-XL/2 | 45.9s vs 0.2s |
| 文生图泛化 | COCO 256 | Table 6 | UViT-S(-Deep)/2 | FID-30K |
| 训练开销 | grouped conv 实现 | Sec. 2.2 | — | 30–40% 训练开销（陈述） |

## 4. 实验设计手法
- 基线："同架构仅差条件化模块"的严格对照（Table 1 注：基线也带 DW conv）——这是本文最干净的设计。SOTA 对比使用已发表数字。
- 规模/广度：两个 DiT 家族（DiT、UViT）多个尺寸；ImageNet 256/512 + COCO；但消融全部在 UViT-S/2、ImageNet 256、80 epoch 上做。大规模文生图明确"留作未来工作"（资源限制，Sec. 3.1）。
- 效率测量：主要是每步 MACs；Orin 上 TensorRT fp16 实测一个点（Table 5）；GPU 训练开销 30–40% 一句话。
- 精度–效率联合：Fig. 1 与 Fig. 7 是 FID–MACs 图（log 轴），用"同 FID 更低 MACs"讲效率故事。
- 消融：模块选择（Table 1）、生成方式（Fig. 4）、条件成分（Table 2）、与已有控制机制组合（Table 3）、训练时长（Fig. 8）。
- 分析性：无可视化或机制分析（无权重/激活分析）。
- 统计：无。
- 系统：代码在 efficientvit 仓库；grouped-conv 批处理实现（Fig. 5）。
- 局限：无单独局限节；训练开销与"需定制 CUDA kernel"在 Sec. 2.2 提及。

## 5. 对抗式审读
- **最弱环节：头条效率数字混合了多个因素**。"52× fewer MACs per step"是每步比较，而 CaT 用 UniPC 20 步、DiT-XL/2 用 250 步——Table 5 的 229× Orin 加速中约 12.5× 来自步数（采样器），并非 CAN/CaT 本身。DiT-XL/2 若同样配 UniPC 的结果未给出。
- CaT 的参数量（377M–486M）并不小，"高效"只在 MACs/步维度成立；也没有拆分 CaT 的增益中多少来自 EfficientViT 架构、多少来自 CAN（缺 "EfficientViT 无 CAN" 的对照）。
- 消融基线较弱：UViT-S/2 在 80 epoch 下 FID 28.32，增益 28.32→8.82 的幅度可能部分源于训练不足；Fig. 8 用于反驳，但其数值在文本中不可读。
- Table 1 是贪心逐步选择，且选择与报告在同一评测上进行（无独立验证集）。
- 动机中的"负迁移"从未被测量；"首次证明权重操纵用于条件控制"的新颖性主张对 HyperNetwork/CondConv 一系边界划得较窄。
- 无多 seed。

## 6. 可迁移手法
1. 新机制的对照基线做到"同架构、只差机制开关"，连辅助模块（DW conv）都对齐（Table 1）。—— 直接读到
2. 用两个"替代解释排除"实验守住核心主张：是不是只因为动态参数化（Fig. 4 核选择）？是不是只因为收敛更快（Fig. 8 长训练曲线）？—— 直接读到
3. 证明"机制可独立工作"：把原有控制机制完全替换为新机制，而不只是叠加（Table 3）。—— 直接读到
4. 对条件输入做成分拆分（类别 vs timestep），回答"信号来自哪里"（Table 2）。—— 直接读到
5. SOTA 表同时报告有/无 cfg 的 FID、每步 MACs、步数、参数，让读者能自己拆分效率来源（Table 4）。—— 直接读到（但正文头条仍混用）
6. 头条加速数字应拆分为"模型每步"与"采样步数"两项，否则会被质疑混淆。—— 我的归纳（本文的缺口）

## 7. 一句话
最关键一招是"同架构只差条件化开关"的消融 + 两个替代解释排除实验（核选择、长训练）；而 52×/229× 的头条效率数字混入了采样器步数差异，是最易被攻击的点。
