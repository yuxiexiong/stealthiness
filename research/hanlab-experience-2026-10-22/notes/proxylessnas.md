# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware

全文已读完（正文 + 附录 A–D）。Fig 4/6 在文本中交错，只能读出零散标签（如 1.83× faster、74.6/74.7/76.7/72.0/68.2/65.4）；Table 1–4 可读。

## 0. 元信息
ICLR 2019。Cai, Zhu, Han。类型：算法（可微 NAS 的显存问题 + 硬件感知）。

## 1. 核心主张
- 主主张：通过路径二值化（每次只激活一条路径）把 NAS 的 GPU 显存与 GPU 时间降到普通训练水平，从而可以**直接在 ImageNet 与目标硬件上**搜索、不用代理任务、不强制重复 block；结果优于代理式 NAS（Abstract、Sec 1）。
- 次主张：(a) 延迟可通过 LUT 期望值变成可微正则项（Eq 5–7）；(b) 不同硬件需要不同架构，GPU 模型在 CPU/手机上不快（Table 4）；(c) 搜出的架构有可解释偏好（GPU 浅宽、CPU 深窄、下采样块用大核，Fig 6）；(d) 搜索成本 200 GPU h，比 MnasNet 少 200×。

## 2. 动机是怎么立住的
- 痛点：传统 NAS 10^4 GPU h；DARTS/One-Shot 显存随候选数 N 线性增长 → 只能在代理任务上搜（Sec 1、Eq 1 后的分析）。这是**机制分析**（N 条路径的特征图都要存），无实测显存表。
- "代理任务上的最优不保证目标任务上最优"——断言，本文没有在同一空间做"代理搜 vs 直接搜"的对照。
- 硬件专用的动机：Table 4 的 3×3 交叉矩阵（GPU 模型 GPU 5.1ms / CPU 204.9ms / 手机 124ms；CPU 模型 CPU 138.7ms；mobile 模型手机 78ms）。对角线最优——这是对"需要专用"最直接的证据（与 HAQ Table 1 同一手法）。
- 因果链：显存瓶颈 → 二值化 → 可在 ImageNet 直接搜 → 精度更好，闭合得靠最终结果；"无代理带来增益"这一中间环节未被隔离。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| CIFAR-10 SOTA 且参数少 | 树形空间 B=18, F=400 | Table 1 | DenseNet、PyramidNet、ENAS、DARTS、NASNet、AmoebaNet-B 等 | Test error、Params |
| 移动端 SOTA + 搜索成本低 | Pixel 1 ≤80ms | Table 2 | MBv1/v2、NASNet、AmoebaNet、MnasNet（含自己复现） | Top-1/5、延迟、GPU h |
| 延迟正则必要 | Proxyless-G 有/无 LL | Table 2, Sec 4.2 | 无 LL 版本缩放到相近延迟 | 71.8 vs 74.2 |
| 各延迟下优于 MBv2 | width multiplier 扫描 | Fig 4 | MBv2 各宽度 | Top-1 vs 延迟（74.6% 时 78ms vs 143ms） |
| GPU 上 SOTA | V100 batch 8 | Table 3 | MBv2、ShuffleNetV2、ResNet-34、NASNet、DARTS、MnasNet | Top-1、ms |
| 硬件需专用 | 3 模型 × 3 硬件 | Table 4 | 彼此 | 延迟 |
| 延迟模型可代替实测 | 1k 测试架构 | Fig 5, App B | 实测 | RMSE 0.75ms |
| 显存降一个数量级 | 「未直接验证」（无显存测量表） | Sec 3.2 | — | — |
| 无代理优于有代理 | 「未直接验证」（无同空间对照） | — | — | — |

## 4. 实验设计手法
- 基线：手工（MBv1/v2、ShuffleNetV2、ResNet-34）与 NAS（NASNet、AmoebaNet、MnasNet、DARTS）。**自己复现 MnasNet 并实测延迟**（Table 2 "MnasNet (our impl.)" 79ms vs 原文 76ms）——在同一台 Pixel 1 上重测基线，避免跨设备比较。MnasNet 的搜索成本由附录 C 估算（8000 模型 × 5 epoch → 40,000 V100 GPU h）。
- 规模/广度：CIFAR-10 + ImageNet；3 硬件（Pixel 1、V100、Xeon CPU）；两种更新算法（G 与 R）都报告。
- 效率测量：**实测**——手机 TFLite batch 1，GPU V100 batch 8（注明 batch 1 严重欠利用），CPU 双路 Xeon batch 1。搜索时用 LUT 延迟模型，**报告时用实测延迟**（App B 明确区分）。
- 精度–效率联合：Fig 4 用 width multiplier 扫出整条曲线，并以"同精度比延迟"（1.83× faster）表述；Table 2/3 同延迟比精度。
- 消融：延迟正则有/无（Table 2，71.8 vs 74.2）；G vs R 两种梯度估计；附录 D 给出不需近似的实现。没有二值化 vs DARTS 式软混合在同空间的精度/显存对比。
- 分析性实验：Fig 6 三硬件架构可视化与解释（GPU 浅宽早池化、CPU 深窄、下采样块偏大核）。Fig 5 延迟模型校准图（预测 vs 实测，贴近 y=x）。
- 泛化：跨硬件、跨数据集（CIFAR/ImageNet 两种不同空间）。
- 统计：无方差；CIFAR 2.08 vs 2.13 的差距在单次运行下不显著。
- 系统贡献：开源预训练模型与评测代码（脚注 1）。
- 局限性：无。

## 5. 对抗式审读
- 最薄弱环节：标题主张"Proxyless"的增益没有被隔离。缺少"同搜索空间、同算法、在代理任务上搜后迁移"的对照，因此精度增益可能来自更大的、非重复的搜索空间与 MBv2 骨干选择，而非去掉代理本身。
- 显存节省"一个数量级"只有机制论证，无显存测量。
- CIFAR-10：Proxyless 用 Cutout + 600 epoch（含 DropPath）；对比方法的训练配方不同；2.08 vs AmoebaNet 2.13 是 0.05% 差，单次运行。
- 搜索成本比较不对称：MnasNet 成本是作者估算的 V100 等价时数（App C）；本文 200 GPU h 不含最终训练。
- Table 2 mobile 最佳是 Proxyless-R（74.6），而 Proxyless-G 加 LL 为 74.2——两种方法哪种更好没有讨论；GPU/CPU 模型用的是哪种算法未说明。

## 6. 可迁移手法
1. 做"模型 × 硬件"交叉延迟矩阵，展示为 A 优化的模型在 B 上不快（Table 4）。—直接读到（与 HAQ Table 1 同构）
2. 在自己的设备上复现主要基线并重测延迟，表中同时列"原文值"与"our impl."（Table 2 MnasNet）。—直接读到
3. 搜索时用代理（LUT）、报告时用实测，并单独给出代理校准图（预测 vs 实测、RMSE）（Fig 5, App B）。—直接读到
4. 用 width multiplier 把单点结果扩成曲线，以"同精度下延迟比"给出大数字（1.83×，Fig 4）。—直接读到
5. 消融"硬件目标项"：去掉延迟正则后再把模型缩放到相同延迟，比较精度（71.8 vs 74.2），证明"把延迟写进目标"不可替代。—直接读到
6. 对竞争方法的搜索成本做显式换算并放附录（MnasNet 40,000 GPU h）。—直接读到

## 7. 一句话
Table 4 的"模型 × 硬件"延迟交叉矩阵加上在同一台 Pixel 1 上复测的 MnasNet，用实测数字把"直接在目标硬件上搜"变得可信；"去掉代理"本身的贡献却没有对照实验。
