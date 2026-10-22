# Network Augmentation for Tiny Deep Learning (NetAug)

全文已读完（正文 + 附录 A/B）。我用 `pdftotext` 非 layout 模式从 raw/pdf/netaug.pdf 生成临时文本（/tmp/g6txt/netaug.txt）阅读；Table 4/5 部分单元格与图中数据混排，主要数字可按行对齐重建；Fig 3/4/5 只有零散标签。

## 0. 元信息
ICLR 2022。Cai, Gan, Lin, Han。类型：算法（训练方法，零推理开销）。

## 1. 核心主张
- 主主张：小模型的问题是**欠拟合而非过拟合**，因此应"增强网络"（把小模型嵌入更宽的权重共享大模型，作为子模型获得额外监督，即"反向 dropout"）而非"增强数据"；NetAug 让小模型 ImageNet 精度最多 +2.2%，推理零开销（Abstract, Table 1）。
- 次主张：(a) 常规正则（Mixup、AutoAugment、RandAugment、DropBlock、Dropout）对大模型有益、对小模型有害（Fig 1, Table 3, Table 6）；(b) 增益随模型变小而增大，对 ResNet50 反而有害（Table 1）；(c) 与 KD 正交（Table 2）；(d) 预训练表征迁移更好（Table 4/5）；(e) 检测上同 AP50 省 41%/38% MACs（Fig 5）。

## 2. 动机是怎么立住的
- **Fig 1 左（关键"矛盾"观察）**：同一组正则方法在 ResNet50 上提升（76.8 → Mixup 77.9 / AutoAugment 77.6 / DropBlock 78.4），在 MobileNetV2-Tiny（174× 小）上下降（51.7 → 51.0 / 48.7 等）。一张图里放"大模型受益 vs 小模型受害"的对照，直接立住"小模型训练与大模型不同"。
- 假说"小模型欠拟合"由 Fig 4 学习曲线验证：NetAug 让 MbV2-Tiny 训练与验证精度都上升（欠拟合缓解），让 ResNet50 训练升、验证降（过拟合加剧）——**同一干预在两类模型上给出相反、且都符合假说的预测**，这是全文最强的机制证据。
- 附录 B 排除"更简单的欠拟合补救"（更小 weight decay、去掉数据增强、更小 batch）——都无效（Table 7）。
- 因果链闭合较好：现象（Fig 1）→ 假说（欠拟合）→ 方法（加容量的辅助监督）→ 双向预测验证（Table 1 ResNet50 −0.3%、Fig 4）→ 排除替代方案（Table 7）。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 正则帮大伤小 | ResNet50 vs MbV2-Tiny | Fig 1 左 | Mixup、AutoAugment、DropBlock | ImageNet Top-1 |
| 正则伤小模型且越强越伤 | 多强度扫描 | Table 3, Table 6 | Dropout、Mixup、AA、RA、DropBlock | Top-1、GPU h |
| NetAug 普遍提升小模型 | 7 个小模型 + ResNet50 | Table 1 | 同训练设置基线 | Top-1 |
| 增益随模型变小增大 | MbV2 w1.0 vs w0.35 | Table 1 | — | +0.9 vs +1.5 |
| 欠拟合机制 | 训练/验证曲线 | Fig 4 | 有无 NetAug × 小/大模型 | train/val acc |
| 不是"只是训练更久" | epoch 扫描 | Fig 3, Table 4 Baseline(150/300/600) | 基线多倍 epoch | Top-1 |
| 与 KD 正交 | 4 模型 | Table 2 | KD（同一教师 Assemble-ResNet50） | Top-1 |
| 迁移更好 | 5 细粒度 + VOC/COCO | Table 4 | Baseline(150/300/600)、KD、NetAug+KD | Top-1、AP50 |
| TinyTL 设置有效 | 冻结权重迁移 | Table 5 | Baseline | Top-1 |
| 检测效率 | 分辨率扫描 | Fig 5 | Baseline、detection Mixup | AP50 vs MACs |
| 简单欠拟合补救无效 | wd、无增强、小 batch | Table 7 | — | Top-1 |
| 训练开销 16.7% | GPU 小时 | Table 3, Sec 3.3 | 基线 | 245 vs 210 GPU h |

## 4. 实验设计手法
- 基线：同训练设置的无 NetAug 基线；5 种主流正则（各在多个强度下，Table 6）；KD（统一教师，Table 2）；**训练更久的基线**（Baseline 300/600 epoch，Table 4；Fig 3 显示 NetAug 达到同精度少 75%/83% epoch）——回应"增益来自额外计算"；附录 B 的"其他欠拟合补救"基线。这是本组中基线设计最针对替代解释的一篇。
- 规模/广度：8 个模型（MbV2-Tiny、MCUNet、MbV3、ProxylessNAS、MbV2 w0.35/w1.0、ResNet50）；ImageNet + ImageNet-21K-P；5 个细粒度分类；VOC + COCO 检测（YoloV3）。
- 效率测量：推理零开销（结构不变）；训练开销以实际 GPU 小时报告（Table 3：210 → 245 h，解释为小模型训练瓶颈在数据加载与通信）；检测效率以 MACs 报（无实测延迟）。
- 精度–效率联合：Fig 5 AP50-MACs 曲线（通过改变输入分辨率）；Fig 3 精度-训练 epoch 曲线（训练效率）。
- 消融：采样多个增强网络"增加成本且伤性能"（Sec 3.2，**无数据**）；增强因子 r、多样性因子 s、α **没有消融**——方法自身超参完全未展示敏感性。
- 分析性实验：Fig 4 学习曲线（机制验证）；Table 4 观察"ImageNet 精度更高不一定迁移更好"（KD 与更长训练反而在部分下游变差）。
- 泛化：跨架构（手工/NAS）、跨数据集、跨任务（分类/检测/迁移/TinyTL）。
- 统计：无方差、无多 seed。很多增益在 0.5–1.6% 范围，单次运行。
- 系统贡献：开源代码与预训练权重。
- 局限性：Sec 4.2 Discussion 明说对大模型有害；Sec 3.3 承认峰值训练显存上升。

## 5. 对抗式审读
- 最薄弱环节：方法核心超参（r、s、α=1.0、只采样一个增强网络）无消融，无法判断结果对超参的稳健性，也无法判断作者是否在测试集上调过这些超参。
- 统计显著性：Table 1 中 +0.7%（MbV2 w1.0 r160 71.2→71.9）这一级别的增益在 ImageNet 单次运行下接近噪声边界；"增益随模型变小增大"的趋势只基于少数点（且 MCUNet 81.8M MACs +1.2 > MbV2-Tiny 23.5M MACs +1.6 并非严格单调）。
- 与已有方法的关系：weight-sharing 超网训练（OFA、Slimmable、"Go wide, then narrow"）与 NetAug 机制高度相近，正文只做了概念区分，没有与 Slimmable/US-Nets 的"最小子网"精度做直接实验对比。
- Fig 1 中正则方法在小模型上的超参是否为小模型重新调优：Table 6 给了两档强度，但都偏向"减弱"方向的有限扫描（例如 Mixup α 只到 0.1）。
- 检测效率仅为 MACs，未测延迟/显存，尽管动机是 MCU 设备。

## 6. 可迁移手法
1. 用"同一干预在大模型上有益、在小模型上有害"的对照图建立反直觉起点（Fig 1 左）。—直接读到
2. 用一个假说做双向预测并同时验证：方法应让小模型 train/val 都升，让大模型 train 升 val 降（Fig 4, Table 1 ResNet50 −0.3%）。—直接读到
3. 给"只是多算了"的质疑配一个训练更久的基线和 epoch 扫描（Fig 3, Table 4 Baseline 300/600）。—直接读到
4. 在附录中逐一排除"更简单的替代解释"（更小 wd、去掉增强、更小 batch，Table 7）。—直接读到
5. 训练开销用实际 GPU 小时而非理论 FLOPs 报告，并解释差异来源（Sec 3.3）。—直接读到
6. 我的归纳：在下游迁移表中显式标红"比基线差"的格子（Table 4），把"上游精度≠下游表征"的反例作为支持自身方法的论据。

## 7. 一句话
Fig 1 那张"同样的正则，ResNet50 涨、MobileNetV2-Tiny 跌"的对照图，加上 Fig 4 对欠拟合假说的双向预测验证，把一个简单的"套大网络训练"包装成有机制依据的新范式；方法自身超参却完全没有消融。
