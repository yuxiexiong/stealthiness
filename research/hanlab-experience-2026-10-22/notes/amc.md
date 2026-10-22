# AMC: AutoML for Model Compression and Acceleration on Mobile Devices

全文已读完（正文 + 参考文献，无附录）。Fig 3/4/5 曲线只剩图注。

## 0. 元信息
ECCV 2018。He*, Lin*, Liu, Wang, Li, Han。类型：算法（RL 搜索逐层剪枝率）+ 移动端实测。

## 1. 核心主张
- 主主张：DDPG 逐层输出连续剪枝率，比手工规则剪枝策略更好——VGG-16 4×FLOPs 压缩下比手工好 2.7%（Abstract；Table 3 中实际比 CP 好 0.3%，比 FP 好 13.2%，"2.7%" 的对应对象在正文未明确）；MobileNet 在 Pixel 1 上实测 1.81×（FLOPs 约束）/1.95×（延迟约束）加速，精度仅降 0.1%（Table 4）。
- 次主张：(a) 不微调的验证精度是微调后精度的好代理（Sec 4.1）；(b) 资源约束靠"限制动作空间"可精确命中目标（Alg 1）；(c) R = −Error·log(FLOPs) 可自动找到无损压缩极限（Eq 5/6）；(d) 学到的策略可解释（3×3 剪多、1×1 剪少，Fig 3）；(e) 可泛化到检测（Table 5）。

## 2. 动机是怎么立住的
- 痛点：手工逐层剪枝率需要专家、次优、不可迁移（Sec 1）——**文字论证**；最有力的证据其实在结果中（Table 2/3 中手工策略被击败）。
- 关键观察："压缩对逐层稀疏度非常敏感，因此需要连续动作空间"（Sec 1, 3.2，指向 Sec 4.2）——但正文中没有找到专门量化"离散化导致损失"的实验，属「未直接验证」。
- "误差与 log(FLOPs) 反比"的经验观察引用 Canziani et al.[7]，用来设计奖励函数。
- 因果链：痛点（手工规则）→ 方法（RL）→ 结果（击败手工规则）是闭合的；但"为什么是 RL 而不是其他搜索"这一环未闭合。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 击败手工策略（CIFAR） | Plain-20、ResNet-56 50% FLOPs 通道剪枝 | Table 2, Fig 2 | uniform/shallow/deep 三种手工策略 | Val/Test/FT 精度 |
| 不微调精度可作代理 | 同表中策略排序一致 | Table 2 | 3–4 个策略 | 微调前后精度 |
| 无过拟合 | Val vs Test 接近 | Table 2 | — | 精度差 |
| 细粒度剪枝推到 5× | ResNet-50 4 轮迭代 | Fig 3/4, Sec 4.2 | 人类专家 3.4×（Han 博士论文） | 精度 76.11 vs 76.13 |
| 击败启发式通道剪枝 | VGG-16 20% FLOPs | Table 3 | FP、RNP、SPP、CP | ΔAcc |
| 击败 width multiplier | MobileNet/V2 | Table 3, Fig 5a | uniform (0.75-224/192) | ΔAcc、MAC |
| 实测加速 | Pixel 1（TFLite）、Titan XP | Table 4, Fig 5b | 原模型、0.75 MobileNet、NetAdapt | ms、fps、内存 |
| 检测泛化 | Faster R-CNN VGG-16 VOC07 | Table 5 | CP 2×/4×、Zhang et al. 4× | mAP |
| RL 优于其他搜索 / 连续优于离散 | 「未直接验证」 | Table 1 只是特性勾选表 | — | — |

## 4. 实验设计手法
- 基线：手工规则（uniform/shallow/deep，自己实现，Table 2）——这是本文最公平、最关键的对照：同剪枝算法、同 FLOPs，只换策略。ImageNet 上有 FP、RNP、SPP、CP（数字多引自原文）、width multiplier（MobileNet 官方的 0.75 版本作为"朴素均匀缩放"基线）、NetAdapt（平台感知启发式，仅 Pixel 1 延迟对比）。没有随机搜索基线。
- 规模/广度：Plain-20、ResNet-56/50（CIFAR）、VGG-16、ResNet-50、MobileNet、MobileNet-V2（ImageNet）；分类 + 检测；2 硬件（Pixel 1 CPU、Titan XP GPU）。
- 效率测量：**真实设备实测**（TFLite，Pixel 1 batch=1；Titan XP batch=50），输入 224×224，**不量化以保证 apple-to-apple**（Table 4 标题）。报延迟、fps、内存。还拆到算子层面：1×1 卷积 2.01× 加速而 depthwise 加速不明显（Sec 4.2 文字）。
- 精度–效率联合：Fig 5a 精度-MACs Pareto，Fig 5b 精度-延迟 Pareto（与 NetAdapt 同在 Pixel 1）；Table 3/4 同 FLOPs 比精度。两种搜索协议（资源约束 / 精度保证）分开展示。
- 消融：**极少**。只有"FLOPs 约束 vs 延迟约束"两种目标的对比（Table 4 两行：1.81× vs 1.95×），证明直接优化延迟更准。无奖励函数、状态特征、动作空间离散/连续、agent 选择的消融。a_max 上限称"=1 结果相似"但无数据。
- 分析性实验：Fig 2 逐层压缩率显示 AMC 学到锯齿状"瓶颈"结构；Fig 3 ResNet-50 峰谷对应 1×1/3×3。都是事后解释。
- 泛化：分类→检测（Table 5，用 AMC 剪的 VGG-16 当 backbone）。
- 统计：无方差、无多次 RL 种子。CIFAR 上差距 0.4–1.9% 单次。
- 系统贡献：搜索 1 GPU 小时（CIFAR，Sec 4.1）；无开源声明。
- 局限性：无专门段落；承认 GPU 加速低于手机（并行度高）。

## 5. 对抗式审读
- 最薄弱环节：所有对照都是"人/启发式 vs RL"，没有"随机搜索 vs RL"——无法区分增益来自 RL 还是来自"任何自动搜索 + 不微调代理评估"。
- 代理有效性（不微调精度 ↔ 微调后精度）只用 Table 2 中每个模型 3–4 个策略的排序一致来证明，没有相关系数或大样本散点。
- Abstract 数字与正文不一致：Abstract 说 1.81×/1.43×，Intro 说 1.95×/1.53×（分别对应 FLOPs 约束与延迟约束两行）；"VGG-16 好 2.7%"在 Table 3 中找不到直接对应（CP 只差 0.3%）。Sec 4.2 还把 "0.5× FLOPs" 写了两次（应为一次 latency）。
- 人类专家 3.4× 基线来自作者自己的博士论文，非独立第三方。
- 检测上 "+0.5 mAP 超过未压缩基线"被解释为正则化效应，单次运行。
- 公平性：Table 3 中 FP、RNP、SPP 的剪枝算法本身与 AMC 不同（AMC 用 CP 的通道剪枝 + 自己的策略），差距混合了策略与剪枝算法；只有与 CP 的对比（0.3%）是"同算法只换策略"。

## 6. 可迁移手法
1. 证明"策略搜索"价值时，固定底层算法和资源预算，只替换策略（uniform/shallow/deep vs 搜索），并画逐层策略图（Table 2 + Fig 2）。—直接读到
2. 用官方 width multiplier 变体（0.75 MobileNet）作为"均匀缩放"朴素基线，在同 MACs 下比较（Table 3/4）。—直接读到
3. 优化目标从代理（FLOPs）换成实测（延迟）各跑一次，用"目标 2× → 实测 1.95×"证明直接优化真实指标更准（Table 4）。—直接读到
4. 效率对比时关掉正交技术（量化）以确保 apple-to-apple，并在表注中写明（Table 4）。—直接读到
5. 搜索出的模型作为 backbone 迁移到下游任务（检测）验证泛化（Table 5）。—直接读到
6. 用 Val/Test 分离 + 数值接近来回应"搜索过拟合验证集"的质疑（Table 2 注）。—直接读到

## 7. 一句话
在同剪枝算法、同 FLOPs 下只换"谁来定逐层剪枝率"，再把 MobileNet 在 Pixel 1 上实测提速近 2×——用真实设备数字把 AutoML 压缩从概念变成可部署结果。
