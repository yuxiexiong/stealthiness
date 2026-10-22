# TinyTL: Reduce Activations, Not Trainable Parameters for Efficient On-Device Learning

## 0. 元信息
- NeurIPS 2020（arXiv 2007.11622v5）
- 类型：算法（迁移学习方法 + 模块设计），带内存分析
- 全文含附录 A–C 已读。**文本损坏较重**：pdftotext 把 Figure 3/4/5 背后嵌入的电子表格数据（各分辨率下 Full/Last/BN/Bias/LiteResidual 的精度与内存）逐列打散穿插到正文段落中（txt 第 429–1112 行），正文句子被切成碎片但可拼回；图中具体点值无法可靠对应，下文只采用 Table 2/3/4 与正文明说的数字。

## 1. 核心主张
- 主主张：训练内存的瓶颈是**激活**而非参数；冻结权重、只学偏置（不需存激活）+ 轻量残差模块（lite residual）即可在几乎不掉精度的前提下把训练内存降 6.5×（同骨干），结合特征提取器自适应（OFA 子网）比全量微调 Inception-V3 省 7.3–12.9×。
- 次要主张：(a) ReLU 类激活只需 1-bit 掩码，sigmoid/h-swish 需存全精度（Table 1）；(b) lite residual 用组卷积 + 2× 降采样，激活约为倒残差块的 1/26；(c) ImageNet 精度不是迁移能力的好代理（附录 A Fig 6），因此要按任务选骨干。

## 2. 动机是怎么立住的
- Figure 1 左：推理 vs 训练（bs=16）内存，对比 Raspberry Pi 1 DRAM（256MB）与 AMD EPYC L3。Figure 1 右（关键观察）：MbV2-1.4 参数比 ResNet-50 小 4.3×，但训练激活只小 1.1×——"高效架构只减参数，没减训练瓶颈"。与 MCUNet Fig 1 是同一种"常用指标 vs 真瓶颈"对照手法。
- 机制证据是一个式子：式 (2) 线性层反向，∂L/∂W 需要 a_i，∂L/∂b 不需要。Table 1 把这一推理推广到激活函数。动机 → 方法的链条是**解析推导**而非测量，闭合但完全理论化。
- §2 引用 BN-only 微调（FT-Norm+Last）"326MB 仍要存 BN 输入激活"作为"参数高效 ≠ 内存高效"的反例（Table 3）。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 同骨干下省内存且准 | ProxylessNAS-Mobile，9 个数据集 | Table 2 | FT-Last（31MB）、FT-Norm+Last（192MB）、FT-Full（391MB） | top-1，训练内存（理论值，bs=8）；TinyTL-L+B 37MB，@320 65MB |
| 等精度省 3.9–6.5× | 不同分辨率下精度-内存曲线 | Figure 3 | FT-Full、FT-Norm+Last、FT-Last、Bias-only | top-1 vs 训练内存 |
| lite residual 必要 | TinyTL-B / -L / -L+B | Table 2 | 内部变体 | 例如 Cars 73.4→84.2→85.0 |
| 结合 FA 比 Inception-V3 全量微调省 7.3–12.9× | 跨骨干对比 | Table 3 | 文献报告的 I-V3/R-50/M2-1.4/NASNet FT-Full，I-V3 FT-Norm/FT-Last | top-1，内存，缩减倍数 |
| 优于激活剪枝 | 精度-内存曲线 | Figure 4 | 动态激活剪枝 [31]（R-50, MbV2） | 剪枝 >50% 精度骤降 |
| 预训练初始化重要 | 随机 vs 预训练初始化 lite residual | Table 4 | — | 例如 Flowers 88.0 vs 95.5 |
| bs=1 可降到 ~16MB | batch 1 vs 8 | Figure 5 | — | 精度几乎不变 |
| ImageNet 精度不代表迁移性 | 8 个预训练模型只训最后层 | 附录 A Figure 6 | — | 排序不一致 |
| 计算也更省 | Pets 上训练成本 | 附录 C Figure 7 | FT-Full MbV2-1.4 | 内存 9.9×、计算 27×（644MB/8900T vs 65MB/330T） |

## 4. 实验设计手法
- **基线选择**：在**同一预训练骨干**上比较四种微调范围（Last / Norm+Last / Full / TinyTL 变体），控制变量清楚（Table 2）。外加动态激活剪枝作为"另一条省内存路线"的代表（Fig 4）。Table 3 的 FT-Full 跨骨干数字**直接引自文献** [42–44]，不是同设置重跑。
- **规模/广度**：8 个物体分类迁移集 + CelebA（40 个属性，VGGFace2 预训练）；骨干以 ProxylessNAS-Mobile 为主，FA 用 OFA。无硬件。
- **效率测量**：**全部是理论计算的训练内存**（§4.1 明说"PyTorch 不支持细粒度内存管理，用理论值"）；统一假设 bs=8 与 8-bit 冻结权重。无真机、无延迟实测。
- **精度–效率联合呈现**：Figure 3 是每个数据集一张精度-内存曲线（扫分辨率），"同精度下省 3.9–6.5×"即读自这里；Table 2 同表放内存列。
- **消融**：TinyTL-B/L/L+B（Table 2）、初始化（Table 4）、batch size（Fig 5）、GN vs BN（§3.3 只有文字："GN bs=8 略差于 BN bs=256"，无数字）。
- **分析性实验**：附录 A Figure 6（ImageNet 精度与迁移排序不一致，用来论证 FA 的必要性）；附录 C Figure 7 内存分解（激活 84.6%）。
- **泛化**：跨 9 个数据集；跨骨干仅在 Table 3（FA）。
- **统计**：§4.2 写"报告五次运行的平均"，但**未给标准差**。
- **系统贡献**：无部署、无代码说明（只有 tinyml.mit.edu 链接）。
- **局限性**：无专门讨论。

## 5. 对抗式审读
- **最薄弱环节：没有任何实测内存或设备。** 题目是"On-Device Learning"，但所有内存数字都是理论值，且 Figure 1 的 Raspberry Pi/EPYC 只作为参照线。真实框架能否兑现节省（这正是后来 MCUNetV3 §2.3 自己指出的问题："existing training system can not realize the theoretical saving"）在本文未验证。
- **"不掉精度"被说得过满**：Table 3 中 FA@256 对 I-V3 FT-Full：Cars 89.6 vs 91.3、Food 83.4 vs 88.7、Aircraft 82.4 vs 85.5；FA@352 Food 85.0 vs 88.7。摘要的"without sacrificing accuracy"只在部分数据集成立。Table 2 中 TinyTL-L+B@320 对同骨干 FT-Full：Aircraft 82.3 vs 86.0、CIFAR100 81.5 vs 84.1。
- **跨论文数字拼表**：Table 3 FT-Full 行来自 [43][44]，训练配方、分辨率不同；而 TinyTL 行用了 FA（每个数据集单独搜子网）、更高分辨率（256/352）、Adam + 逐数据集调 LR。
- FA 的搜索成本（Flowers 上 134T + 157T MAC，附录 A）在主表"训练内存"里不体现；FA 阶段峰值 61MB 但只报最终阶段 66MB。
- 对比 FT-Full 时假设两者都 bs=8；但 FT-Full 也可以用 GN + bs=1 降内存，作者只对自己方法展示了 bs=1（Fig 5）。
- 五次运行无方差，Table 2 中 TinyTL-L 与 L+B 差 0.1–1.0 的比较（"L+B 略好"）无法判断显著性。

## 6. 可迁移的手法
1. 用一个反直觉的"同类模型资源对照"（参数小 4.3× 但激活只小 1.1×）把社区注意力从常用指标转到真瓶颈。（Figure 1 右）**直接读到**
2. 方法的核心论证可以是一行反向传播公式：从式子里读出"哪些量需要被存"，再据此决定冻结什么。（式 2, Table 1）**直接读到**
3. 比较多种"微调范围"时，固定同一预训练骨干，只改可训练部分，让差异只来自范围本身。（Table 2）**直接读到**
4. 给每个数据集画精度-资源曲线（扫分辨率），再从曲线上读"等精度下省多少"，避免单点比较被质疑挑点。（Figure 3）**直接读到**
5. 用"不同指标下的排序不一致"（ImageNet 精度 vs 迁移精度）为"按任务自适应选择"提供依据。（附录 A Figure 6）**我的归纳**

## 7. 一句话
Figure 1 右的"参数小 4.3×、激活只小 1.1×"加上式 (2) 一行推导，就把"冻结权重只训偏置"从经验技巧变成了有原理的内存方案；但全文没有一个实测内存数字。
