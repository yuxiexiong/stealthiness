# Han Lab：62份PDF逐篇问题、贡献与证据卡

[先读总分析与对我们课题的判断](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/hanlab-paper-strategy-2026-09-16/REPORT.zh-CN.md)

范围：桌面2022–2026目录中的全部62份PDF。页码是各本地PDF物理页码。数字为论文报告，未执行复现。阅读范围逐篇声明，不能把关键段阅读等同于全文逐字精读。

会议/期刊扩展和综述/回顾按文件记录，不能将62份文件视为62个独立创新。来源链接及归档venue保留原始元数据，不代表本次重查所有发表状态。

## 2022：10份

| 论文 | 瞄准的问题 | 核心洞察 |
|---|---|---|
| [GAN Compression: Efficient Architectures for Interactive Conditional GANs](#2022-gan-compression-tpami) | 让条件GAN在边缘设备上交互式生成，同时保留生成质量。 | 教师可把非配对任务变成伪配对蒸馏；中间表示和预训练判别器提供更稳定监督；不同层的压缩敏感性不同。 |
| [Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation](#2022-litepose) | 把多人二维姿态估计部署到低算力设备。 | 逐步删减实验显示高分辨率分支并非低预算下必需；关键是恢复低层细粒度信息，并扩大有效感受野。 |
| [On-Device Training Under 256KB Memory](#2022-mcunetv3) | 在仅256KB SRAM的微控制器上完成真实量化模型训练。 | 量化改变权重与梯度的尺度比；只更新贡献较高的层/张量，并在编译时删除无用反向图，才能兑现资源节省。 |
| [Network Augmentation for Tiny Deep Learning](#2022-netaug) | 提高微小神经网络的表示和训练质量，同时保持部署模型大小。 | 训练时让小网络成为更宽权重共享网络的子网络，额外宽网络损失改善共享表示，推理时丢弃扩展部分。 |
| [QOC: Quantum On-Chip Training with Parameter Shift and Gradient Pruning](#2022-qoc-quantum-on-chip-training-with-parameter-shift-and-gradient-pruning) | 降低量子芯片上参数移位训练的测量开销，并改善含噪梯度优化。 | 用阶段性累积梯度幅度估计可靠性，以概率方式裁掉不可靠梯度，兼顾探索和调用节省。 |
| [QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits](#2022-quantumnas) | 针对真实噪声选择鲁棒量子电路及其物理量子比特映射。 | 训练一次SuperCircuit可为子电路继承参数，建立足够相关的低成本排名；必须联合电路和映射。 |
| [QuantumNAT: Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization](#2022-quantumnat) | 缩小理想训练量子网络与真实含噪设备推理间的分布差距。 | 测量端尺度/偏移可部分归一化；门级随机Pauli插入模拟设备噪声；量化测量输出可以抑噪。 |
| [QuEst: Graph Transformer for Quantum Circuit Reliability Estimation](#2022-quest-q) | 快速估计量子电路在设备噪声下的可靠性，减少昂贵评估。 | 随机电路后接其逆电路，使理想输出已知为零态；用正确输出概率PST作可靠性代理，再用门图和噪声特征预测。 |
| [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](#2022-sige) | 加速局部交互编辑时的条件GAN与扩散模型推理。 | 编辑差异具有空间结构，缓存原图特征、只重算局部块可保留上下文；但还必须处理gather/scatter与归一化开销。 |
| [TorchSparse: Efficient Point Cloud Inference Engine](#2022-torchsparse) | 提高稀疏点云神经网络的真实GPU推理速度。 | 适当多算换取规则矩阵乘法，同时优化数据移动，比单纯最小算术量更快。 |

<a id="2022-gan-compression-tpami"></a>
### GAN Compression: Efficient Architectures for Interactive Conditional GANs

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/GAN Compression- Efficient Architectures for Interactive Conditional GANs.pdf>) · [归档来源](https://arxiv.org/pdf/2003.08936v4) · 24页 · 期刊扩展：GAN Compression及Fast GAN Compression

**问题：** 让条件GAN在边缘设备上交互式生成，同时保留生成质量。

**痛点：** 生成器计算量大；对抗训练使小模型训练和蒸馏不稳定；真实延迟与MAC数不等价。

**朴素方案为什么不够：** 直接缩宽、从零训练小生成器、仅输出层蒸馏，以及照搬分类网络的层替换，都会明显损害FID或mIoU。

**洞察：** 教师可把非配对任务变成伪配对蒸馏；中间表示和预训练判别器提供更稳定监督；不同层的压缩敏感性不同。

**贡献：** 结合中间特征蒸馏、可分离卷积和权重共享搜索；期刊版Fast GAN Compression简化教师训练并改用进化搜索。

**关键证据：**

- PDF p.10：CycleGAN的MAC减少21.2倍，但Xavier GPU实测速仅3.1倍；不能把MAC减幅当延迟减幅。
- PDF p.13：Cityscapes：小模型从零训练32.57 mIoU、输出蒸馏35.67、中间蒸馏38.26，完整GAN Compression在约5.66G MACs达到40.77；原模型56.8G、42.06。
- PDF p.11：Fast版本报告训练1.6–3.7倍、搜索3.5–12倍加速，并拆解预训练判别器/剪枝/蒸馏。

**边界：** 质量并非逐项无损；教师错误会继承；训练与搜索需要前置成本。此PDF是已有会议工作的期刊扩展，不能重复计作完全独立选题。

**论文策略：** 先找生成任务不同于分类压缩的机制，再把每个机制落实成可拆解部件，并测真实设备。

**对我们的启发：** 我们的诊断若只是画图，应证明它选出的设计或检查优于直接答案表；离线建表成本也必须入账。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

<a id="2022-litepose"></a>
### Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/Lite Pose- Efficient Architecture Design for 2D Human Pose Estimation.pdf>) · [归档来源](https://arxiv.org/pdf/2205.01271) · 11页 · 原始研究

**问题：** 把多人二维姿态估计部署到低算力设备。

**痛点：** 高分辨率多分支模型精度高，但运算和移动端延迟太大。

**朴素方案为什么不够：** 等比例缩小HRNet等多分支模型，仍保留了低计算预算下不划算的反复高分辨率细化。

**洞察：** 逐步删减实验显示高分辨率分支并非低预算下必需；关键是恢复低层细粒度信息，并扩大有效感受野。

**贡献：** 单分支骨干、fusion deconvolution head、7×7大核和硬件约束NAS，配合蒸馏。

**关键证据：**

- PDF p.3：逐步缩减示例：12.5G/50.4 AP到9.15G/51.8 AP，直接支持结构冗余判断。
- PDF p.7：CrowdPose LitePose-S 5G/58.3 AP、移动GPU76ms，对比EfficientHRNet-H-1 14.2G/56.3 AP、267ms。
- PDF p.8：融合头在消融中47.1→54.7 AP；在融合头条件下3×3到7×7为41.7→54.7，分离结构与训练收益。

**边界：** 结论限于低预算bottom-up pose；不证明高算力多分支普遍多余；NAS/蒸馏均有前置训练成本。

**论文策略：** 以受控删减反驳领域默认架构，再保留真正需要的能力并在多设备兑现。

**对我们的启发：** 先做去掉归因图、只给答案表的严格删减实验；若不变差，归因图不是必要贡献。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

<a id="2022-mcunetv3"></a>
### On-Device Training Under 256KB Memory

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/On-Device Training Under 256KB Memory.pdf>) · [归档来源](https://arxiv.org/pdf/2206.15472) · 18页 · 原始研究

**问题：** 在仅256KB SRAM的微控制器上完成真实量化模型训练。

**痛点：** 训练需要激活/梯度缓存；真实int8优化尺度失衡；常见框架运行时本身就太大。

**朴素方案为什么不够：** 假量化并不节省训练内存；把SGD或Adam直接用于真int8图精度下降；只冻结权重也不自动去掉框架冗余。

**洞察：** 量化改变权重与梯度的尺度比；只更新贡献较高的层/张量，并在编译时删除无用反向图，才能兑现资源节省。

**贡献：** 量化感知梯度缩放QAS、贡献驱动稀疏更新，以及编译时自动微分/算子重排的Tiny Training Engine。

**关键证据：**

- PDF p.7：8任务平均精度：fp32 SGD73.3、int8 SGD64.9、Adam71.8，QAS73.5。
- PDF p.8：VWW实机训练使用206KB SRAM、89.1%准确率；实机STM32F746物理容量为320KB，运行占用仍低于256KB目标。
- PDF p.9：报告20–21倍内存减少和23–25倍训练加速；OOM完整训练对照含按算子估计/投影，不能一概说所有对照均实机完整执行。
- PDF p.10：通道选择的幅度与随机策略差异小于0.2%，作者明确不把精细通道排序当主要收益来源。

**边界：** 迁移微调不等于从头预训练或真实长期持续学习；贡献可加性是经验近似；离线分析有成本。

**论文策略：** 把硬资源不可能拆为优化/算法/系统三个瓶颈，逐一消除，最后实机闭环。

**对我们的启发：** 即使某个排序信号有理论意义，也必须允许随机对照推翻其必要性；我们的归因应接受同样检验。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

<a id="2022-netaug"></a>
### Network Augmentation for Tiny Deep Learning

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/Network Augmentation for Tiny Deep Learning.pdf>) · [归档来源](https://arxiv.org/pdf/2110.08890) · 13页 · 原始研究

**问题：** 提高微小神经网络的表示和训练质量，同时保持部署模型大小。

**痛点：** 大网络默认的过拟合叙事不适用于容量不足的小模型。

**朴素方案为什么不够：** Dropout、Mixup、DropBlock、RandAugment等正则化可能加重tiny模型欠拟合；简单延长训练也不能解决容量限制。

**洞察：** 训练时让小网络成为更宽权重共享网络的子网络，额外宽网络损失改善共享表示，推理时丢弃扩展部分。

**贡献：** Network Augmentation以随机采样宽子网为辅助训练；与以搜索最优架构为目的的supernet区分。

**关键证据：**

- PDF p.7：Tiny模型NetAug提高约1.3点top-1；训练245对210 GPU小时，增加16.7%，部署无额外计算。
- PDF p.7：训练/验证曲线支持欠拟合改善；常规正则化可下降达3.7点，ResNet-50不受益甚至受损。
- PDF p.8：下游迁移比较显示ImageNet上更好的KD/预训练不必然带来更好的下游效果，NetAug迁移更一致。

**边界：** 针对小模型欠拟合，不是所有模型的通用正则化；训练时额外内存与成本；机理主要为实证支持。

**论文策略：** 找出适用条件发生反转的旧常识，用正/负规模对照限定机制。

**对我们的启发：** 我们的有效病人也应展示明确失败机制及正反对照；不能默认所有回答改变都对应同一种病因。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

<a id="2022-qoc-quantum-on-chip-training-with-parameter-shift-and-gradient-pruning"></a>
### QOC: Quantum On-Chip Training with Parameter Shift and Gradient Pruning

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/QOC- Quantum On-Chip Training with Parameter Shift and Gradient Pruning.pdf>) · [归档来源](https://arxiv.org/pdf/2202.13239) · 6页 · 原始研究，有限规模实机量子训练

**问题：** 降低量子芯片上参数移位训练的测量开销，并改善含噪梯度优化。

**痛点：** 经典模拟扩展差；真实量子梯度测量昂贵且小梯度相对噪声大。

**朴素方案为什么不够：** 将所有梯度同等测量和更新浪费真实芯片调用；完全依靠理想模拟训练会遭遇部署噪声。

**洞察：** 用阶段性累积梯度幅度估计可靠性，以概率方式裁掉不可靠梯度，兼顾探索和调用节省。

**贡献：** Quantum On-Chip training结合parameter shift和probabilistic gradient pruning；比较确定性剪枝与概率剪枝。

**关键证据：**

- PDF p.5：真实设备五任务比较：MNIST四分类0.59→0.64、Fashion四分类0.49→0.57；任务均为小规模四逻辑量子比特。
- PDF p.6：概率剪枝对比确定性剪枝，例如Fashion四分类0.79对0.72、MNIST二分类0.85对0.82；此消融与实机主表条件不能混为一谈。
- PDF p.6：扩展曲线部分为预测：量子大规模和经典更大规模点是投影，不能声称已实测普遍量子优势。

**边界：** 有限小分类任务，非相对最强经典学习器的量子优势证明；历史梯度可靠性估计可能随噪声漂移；扩展结果含外推。

**论文策略：** 诊断信号直接决定跳过哪些昂贵操作，同时验证真实任务精度与测量成本。

**对我们的启发：** 我们的诊断至少应帮下一次检查省调用或提高正确决策；precision单项不能替代两者。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

<a id="2022-quantumnas"></a>
### QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/QuantumNAS- Noise-Adaptive Search for Robust Quantum Circuits.pdf>) · [归档来源](https://arxiv.org/pdf/2107.10845) · 19页 · 原始研究

**问题：** 针对真实噪声选择鲁棒量子电路及其物理量子比特映射。

**痛点：** 理想电路规模越大未必实机越好，电路与硬件映射相互影响，逐候选训练昂贵。

**朴素方案为什么不够：** 只优化理想无噪电路或只做映射搜索都不够；训练每条候选电路的搜索成本难承受。

**洞察：** 训练一次SuperCircuit可为子电路继承参数，建立足够相关的低成本排名；必须联合电路和映射。

**贡献：** 前缀/限制采样训练共享SuperCircuit、进化式联合搜索及门剪枝微调，支持噪声模型或实际芯片反馈。

**关键证据：**

- PDF p.5：继承参数的代理排名与从零训练相关性被直接验证；与后页合看，平均Spearman约0.75。
- PDF p.6：估计噪声表现与真实硬件表现相关性约0.76，明确校准代理。
- PDF p.10：真实硬件搜索20轮约3天；较大21-qubit实验有实测，127-qubit/57小时扩展为投影。
- PDF p.11：联合电路/映射在单独优化之上平均约9个百分点；前缀/限制采样相较随机采样平均约12个百分点。

**边界：** 共享权重排名并非无误；设备变化降低转移；限定电路族和任务，不证明一般量子算法优势；大规模预算部分外推。

**论文策略：** 先验证廉价代理确实能排序，再证明代理驱动的最终选择赢过强搜索对照，并交代搜索成本。

**对我们的启发：** 这是最接近诊断研究的模板：归因信号→选择哪个干预→干预实际收益，三个箭头都要独立成立。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

<a id="2022-quantumnat"></a>
### QuantumNAT: Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/QuantumNAT- Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization.pdf>) · [归档来源](https://arxiv.org/pdf/2110.11331) · 10页 · 原始研究

**问题：** 缩小理想训练量子网络与真实含噪设备推理间的分布差距。

**痛点：** 不同真实设备的读出分布偏移使无噪训练的分类器失效。

**朴素方案为什么不够：** 只在理想模拟器训练，或任意加高斯噪声，并不对应门噪声机制；量化与噪声注入还可能互相抵消。

**洞察：** 测量端尺度/偏移可部分归一化；门级随机Pauli插入模拟设备噪声；量化测量输出可以抑噪。

**贡献：** 结合measurement normalization、hardware-backed gate noise injection及measurement quantization，并检查部件交互。

**关键证据：**

- PDF p.5：主表8任务均值baseline0.42→normalization0.52→gate injection0.61→quantization0.64；报告16种设置中13种改善。
- PDF p.5：SNR实测例：Quito深电路0.35→0.71，支撑归一化确实降低噪声偏移。
- PDF p.6：门级注噪结合量化平均比输出高斯注噪高约11点；并非随便加噪都有效。

**边界：** 实际测试受预算限制只取前300张图像，4/10qubit任务；并非每种设置改进，也未证明量子学习优于经典学习。

**论文策略：** 把可测分布异常转成机理匹配的干预，专门对照一个看似合理却错机制的办法。

**对我们的启发：** PurMM若在另一种病因上无效，只能说明不适配该病；要有与我们观测权限/诊断任务匹配的强对照。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

<a id="2022-quest-q"></a>
### QuEst: Graph Transformer for Quantum Circuit Reliability Estimation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/QuEst- Graph Transformer for Quantum Circuit Reliability Estimation.pdf>) · [归档来源](https://arxiv.org/pdf/2210.16724) · 10页 · 原始研究，可靠性数据与预测器案例

**问题：** 快速估计量子电路在设备噪声下的可靠性，减少昂贵评估。

**痛点：** 量子态层析与模拟昂贵；简单全局统计忽略门顺序和噪声局部结构。

**朴素方案为什么不够：** 仅用门数/深度等全局统计的MLP难以精确表达电路噪声相互作用。

**洞察：** 随机电路后接其逆电路，使理想输出已知为零态；用正确输出概率PST作可靠性代理，再用门图和噪声特征预测。

**贡献：** 构造模拟及真实芯片数据集，graph Transformer回归PST，并对门图/全局特征作消融。

**关键证据：**

- PDF p.5：数据约1万模拟、2.5万真实样本，涉及5种后端；PST是理想零态输出的观测概率，并非任意电路完整态保真度。
- PDF p.7：按后端70/20/10随机划分并分别训练；算法电路实验R²约0.9985，而简单特征MLP约0.9110。
- PDF p.8：相对经典模拟，3090预测单样本约200倍、批10约1.7千倍加速；不包含收集标签和训练的前置成本。

**边界：** 随机划分不是新电路族/新设备/跨时间泛化；PST代理不等于一般保真度；未在所读结果中建立“依据预测做编译决策更好”的终端价值。

**论文策略：** 以可获取代理标签构建昂贵量的快速替代器，证据强度低于包含下游决策闭环的论文。

**对我们的启发：** 不能因Han Lab出品就把预测准确当完整solid范式；我们的finite table、precision也面临相同代理与泛化问题。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

<a id="2022-sige"></a>
### Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models.pdf>) · [归档来源](https://papers.neurips.cc/paper_files/paper/2022/file/b9603de9e49d0838e53b6c9cf9d06556-Paper-Conference.pdf) · 16页 · 原始会议研究；与2023期刊版属同一研究线

**问题：** 加速局部交互编辑时的条件GAN与扩散模型推理。

**痛点：** 每次小编辑都重算全图；局部激活变化虽稀疏，但非结构化稀疏执行开销很大。

**朴素方案为什么不够：** 直接裁图生成再拼回会损失全局上下文并出现接缝；按激活阈值剪枝也难获得实际加速。

**洞察：** 编辑差异具有空间结构，缓存原图特征、只重算局部块可保留上下文；但还必须处理gather/scatter与归一化开销。

**贡献：** spatially sparse inference及SIGE引擎：分块聚集卷积散射、复用归一化统计、核融合。

**关键证据：**

- PDF p.7：DDIM原始FID65.4、SIGE65.8；MAC249G→65.3G，保持生成质量而大幅减计算。
- PDF p.9：DDIM约1.2%编辑在3090由37.5ms→12.6ms，约3倍；速度因任务/设备/编辑面积而变。
- PDF p.9：朴素分块已减7.6倍MAC但仅1.6倍加速；归一化与融合后达2.9倍，直接展示隐藏开销。
- PDF p.10：讨论额外激活缓存、低分辨率加速有限及局部编辑引起的全局光照变化无法更新。

**边界：** 以近似保持输出换局部重用，可能接缝；扩散每步缓存占额外内存；速度不是对所有编辑面积都成立。

**论文策略：** 从结构冗余到系统兑现，显式解释为何看似足够的稀疏算法仍不能得到速度。

**对我们的启发：** 我们的诊断需核算读图/生成/校准/查表的全部成本；稀疏且准确的claim不等于总体高效。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

<a id="2022-torchsparse"></a>
### TorchSparse: Efficient Point Cloud Inference Engine

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2022/TorchSparse- Efficient Point Cloud Inference Engine.pdf>) · [归档来源](https://arxiv.org/pdf/2204.10319) · 14页 · 原始研究；2023 TorchSparse++的前代

**问题：** 提高稀疏点云神经网络的真实GPU推理速度。

**痛点：** 点云稀疏卷积的小矩阵乘法和不规则数据搬运使GPU利用率低。

**朴素方案为什么不够：** 只减少MAC或一味避免补零无效；固定分组在不同数据形态上可能适得其反。

**洞察：** 适当多算换取规则矩阵乘法，同时优化数据移动，比单纯最小算术量更快。

**贡献：** 自适应矩阵分组、FP16向量化gather/scatter、访存局部性优化及映射融合。

**关键证据：**

- PDF p.8：7个模型任务、3数据集和多代GPU的端到端几何平均加速约1.5–1.6倍，单个工作负载最高2.3倍。
- PDF p.9：自适应矩阵分组约1.39/1.54倍；固定策略在nuScenes可1.5倍，在SemanticKITTI却0.87倍。
- PDF p.10：数据搬运组件消融：仅量化约1.32倍，组合优化gather/scatter约2.72倍。

**边界：** 调优依赖模型/输入分布/硬件；此版主要推理，不能把后续训练能力提前归给它；对照实现版本会改变结论。

**论文策略：** 反转“少算即快”的代理目标，以端到端测量指导自适应设计。

**对我们的启发：** 少检查几格表不等于诊断成本低；必须包含计算图生成、必要信息获得和读取时间。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法推导与设计说明、关键结果表/图及消融/讨论；长页为相关段落选择阅读。未声称逐字通读全部参考文献和附录。

## 2023：9份

| 论文 | 瞄准的问题 | 核心洞察 |
|---|---|---|
| [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation](#2023-bevfusion) | 统一融合摄像头和LiDAR，兼顾三维检测与地图语义分割。 | 鸟瞰空间同时保留几何与稠密相机语义；先消除投影的信息瓶颈，再消除其工程瓶颈。 |
| [EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction](#2023-efficientvit) | 使高分辨率密集预测同时获得全局建模能力与硬件效率。 | 全局线性注意力、局部卷积、多尺度QKV是互补能力，使用硬件友好的小卷积补偿线性注意力损失。 |
| [FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer](#2023-flatformer) | 让点云Transformer达到实时检测速度。 | 以等点数分组牺牲一点几何整齐，换来规则执行；交替坐标排序和窗口移位恢复组间信息。 |
| [PockEngine: Sparse and Efficient Fine-tuning in a Pocket](#2023-pockengine) | 在手机、Jetson、MCU等受限边缘设备上高效微调视觉与语言模型。 | 用推理原语构建静态反向图，在编译时剪枝并安排张量生命周期，将稀疏更新变成真实执行收益。 |
| [Retrospective: EIE: Efficient Inference Engine on Sparse and Compressed Neural Network](#2023-retrospective-eie-efficient-inference-engine-on-sparse-and-compressed-neural-network) | 回顾EIE如何把网络压缩产生的稀疏性转成硬件收益，以及后来的适用边界。 | 算法与硬件共同设计可跳过零权重/零激活；W4A16在内存受限矩阵向量场景有持久价值，但索引和控制成本不可忽略。 |
| [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models - TPAMI extended version](#2023-sige-tpami) | 把SIGE从卷积式生成器扩展到注意力占主导的Stable Diffusion及更多实际编辑场景。 | 未修改区域可复用query输出，只对编辑query计算注意力，仍保留全局key/value上下文。 |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](#2023-smoothquant) | 让大语言模型W8A8量化同时保留精度和真实INT8硬件加速。 | outlier沿输入通道稳定，而权重较容易量化；用等价缩放把困难从激活迁到权重并平衡两侧。 |
| [Tiny Machine Learning: Progress and Futures [Feature]](#2023-tinyml-magazine) | 梳理极小设备上推理和训练的资源约束、已有路线与未来问题。 | TinyML需要网络设计、执行顺序、量化优化和编译系统共同设计，且不同阶段的主要瓶颈会迁移。 |
| [TorchSparse++: Efficient Training and Inference Framework for Sparse Convolution on GPUs](#2023-torchsparse-plus-plus) | 降低高性能稀疏卷积的实现成本，并提升点云训练/推理速度。 | 稀疏卷积可分解为成熟dense MMA与少量稀疏访存模板；数据流应按共享映射分组、以端到端代价选择。 |

<a id="2023-bevfusion"></a>
### BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/BEVFusion- Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation.pdf>) · [归档来源](https://arxiv.org/pdf/2205.13542) · 8页 · 原始研究；所给PDF为2024修订版，按2023文件夹归类

**问题：** 统一融合摄像头和LiDAR，兼顾三维检测与地图语义分割。

**痛点：** 点级投影丢弃大量摄像头像素，图像级投影又损失几何，且相机到BEV转换慢。

**朴素方案为什么不够：** PointPainting等把相机特征放到少量LiDAR点，只保留不足5%的图像对应信息；原始BEV pooling耗时超过500ms。

**洞察：** 鸟瞰空间同时保留几何与稠密相机语义；先消除投影的信息瓶颈，再消除其工程瓶颈。

**贡献：** 共享BEV表示、预计算网格和interval reduction加速池化、多任务头。

**关键证据：**

- PDF p.3：视图转换从超过500ms降至约12ms；局部模块约40倍，不能写成整模型40倍。
- PDF p.4：nuScenes检测：TransFusion 68.9 mAP/71.6 NDS、156.6ms；BEVFusion70.2/72.9、119.2ms。
- PDF p.5：地图分割mIoU62.7，对比PointPainting49.1、MVP49.0；camera-only56.6，显示语义信息保留的价值。
- PDF p.6：稀疏LiDAR等扰动测试中，1-beam条件约增加12点NDS。

**边界：** 相机深度估计和BEV离散化仍不完美；任务头范围不同；两个数据集不代表所有传感器故障。

**论文策略：** 以两个任务暴露旧表示的偏科，再以统一表示和系统优化共同解决。

**对我们的启发：** 应明确诊断帮助哪类决策；不能仅测某方法擅长的一类回答变化。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法和相关实验表/消融/讨论；长页选择相关段落。未逐字通读全部附录与参考文献。

<a id="2023-efficientvit"></a>
### EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/EfficientViT- Multi-Scale Linear Attention for High-Resolution Dense Prediction.pdf>) · [归档来源](https://arxiv.org/pdf/2205.14756) · 12页 · 原始研究；所给修订版包含后续SAM实验

**问题：** 使高分辨率密集预测同时获得全局建模能力与硬件效率。

**痛点：** softmax注意力随token数平方增长；直接线性化又削弱局部聚焦与多尺度表达。

**朴素方案为什么不够：** 直接替换为ReLU线性注意力虽快却精度差；仅靠参数/FLOPs也不足以说明部署效率。

**洞察：** 全局线性注意力、局部卷积、多尺度QKV是互补能力，使用硬件友好的小卷积补偿线性注意力损失。

**贡献：** 多尺度线性注意力模块、增强局部能力的FFN，以及面向高分辨率的网络设计。

**关键证据：**

- PDF p.5：同4.4G MACs/0.7M参数消融：无全局/多尺度68.1 mIoU；仅多尺度72.3，仅全局72.2，两者74.5。
- PDF p.7：Cityscapes B1为80.5 mIoU/Orin24.3ms，SegNext-T为79.8/93.2ms。
- PDF p.9：修订版SAM-XL1 COCO47.8对SAM-H46.5，A100吞吐182对11图/秒；但LVIS单点交互56.6低于59.2，非所有指标无损。

**边界：** 适用高分辨率密集任务而非所有transformer；不应把后续SAM实验当2023初版已完成内容；部分交互设置更差。

**论文策略：** 对快速近似明确指出丢失的能力，再用等预算消融证明补偿机制。

**对我们的启发：** 若我们压缩诊断信息，应测丢失什么以及何种补偿有效，而不是只留容易预测的格子。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法和相关实验表/消融/讨论；长页选择相关段落。未逐字通读全部附录与参考文献。

<a id="2023-flatformer"></a>
### FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/FlatFormer- Flattened Window Attention for Efficient Point Cloud Transformer.pdf>) · [归档来源](https://arxiv.org/pdf/2301.08739) · 12页 · 原始研究

**问题：** 让点云Transformer达到实时检测速度。

**痛点：** 邻域准备占传统点云Transformer大量时间，等空间窗口点数不规则导致padding和低利用率。

**朴素方案为什么不够：** 以空间等大窗口保持精确局部性，产生不均匀工作量；逐点邻域搜索成本高。

**洞察：** 以等点数分组牺牲一点几何整齐，换来规则执行；交替坐标排序和窗口移位恢复组间信息。

**贡献：** flatten-window-sort、等点数组、轴交换与shift，以及分组执行优化。

**关键证据：**

- PDF p.6：Waymo三帧FlatFormer12.7ms/72.0 L2 mAPH，SST57.8ms/70.4，CenterPoint++13.6ms/71.6。
- PDF p.7：Orin达到16.1 FPS，对比CenterPoint13.7与SST3.4。
- PDF p.7：受控小规模消融：完整61.7，随机分组57.8，无量化60.4，无轴交换61.1，无shift61.2。

**边界：** 主要Waymo；所报模型速度不包含加载/后处理；个别两阶段设置不胜FSD。后续TorchSparse++更新卷积后端后重置了Transformer比较结论。

**论文策略：** 放松看似必要的几何约束，补回交流能力，并在真实实时阈值上验证。

**对我们的启发：** 不能永久固定弱实现作对照；最新AtP/直接查询等也应和我们同样认真优化。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法和相关实验表/消融/讨论；长页选择相关段落。未逐字通读全部附录与参考文献。

<a id="2023-pockengine"></a>
### PockEngine: Sparse and Efficient Fine-tuning in a Pocket

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/PockEngine- Sparse and Efficient Fine-tuning in a Pocket.pdf>) · [归档来源](https://arxiv.org/pdf/2310.17752) · 14页 · 原始系统研究；继承MCUNetV3稀疏更新思路

**问题：** 在手机、Jetson、MCU等受限边缘设备上高效微调视觉与语言模型。

**痛点：** 框架解释/自动微分成本、反向激活保存和稀疏更新执行脱节；可训练参数少不代表实际快。

**朴素方案为什么不够：** 仅冻结部分权重或使用LoRA，未必删除反向冗余；用通用PyTorch运行时甚至无法部署到极小设备。

**洞察：** 用推理原语构建静态反向图，在编译时剪枝并安排张量生命周期，将稀疏更新变成真实执行收益。

**贡献：** 编译时自动微分、层/子层稀疏反传、图剪枝/算子重排，支持多种原本只有推理的后端。

**关键证据：**

- PDF p.5：逐层离线微调估计贡献，并假设贡献可加求稀疏方案；这不是无成本、无需假设的在线诊断。
- PDF p.8：跨任务平均精度接近全更新，如MCUNet74.8对74.1，MobileNet88.5对89.2；并非每模型不降。
- PDF p.11：Llama2-7B Orin：PyTorch全更新7.7s/45.1GB，Pock全更新1.8s/43.1GB，Pock稀疏0.9s/31.2GB；MT-Bench分别6.1/6.1/5.7。

**边界：** 是迁移微调而非通用预训练；离线敏感性评估和可加近似有成本；LLM最大序列/批量有限，少量质量下降真实存在。

**论文策略：** 分开证明系统实现收益与稀疏策略收益，使用同一全更新策略隔离框架变量。

**对我们的启发：** 图表示、查询策略、渲染给读者三层也须拆开；否则把额外信息或实现差异当诊断创新。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法和相关实验表/消融/讨论；长页选择相关段落。未逐字通读全部附录与参考文献。

<a id="2023-retrospective-eie-efficient-inference-engine-on-sparse-and-compressed-neural-network"></a>
### Retrospective: EIE: Efficient Inference Engine on Sparse and Compressed Neural Network

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/Retrospective- EIE- Efficient Inference Engine on Sparse and Compressed Neural Network.pdf>) · [归档来源](https://arxiv.org/pdf/2306.09552v1) · 2页 · 两页历史回顾；不是2023新原始实验

**问题：** 回顾EIE如何把网络压缩产生的稀疏性转成硬件收益，以及后来的适用边界。

**痛点：** 内存搬运昂贵，压缩可减权重，但不规则稀疏在通用并行硬件上开销大。

**朴素方案为什么不够：** 把压缩后的模型原样交给通用稀疏软件，未必能比密集运算高效。

**洞察：** 算法与硬件共同设计可跳过零权重/零激活；W4A16在内存受限矩阵向量场景有持久价值，但索引和控制成本不可忽略。

**贡献：** 总结2016 EIE的CSC表示、PE分配、非零激活检测及4-bit权重共享；回顾后续结构稀疏路线。

**关键证据：**

- PDF p.1：明确回顾ISCA2016工作，说明稀疏和低精度共同设计思路；本篇不提供新的完整原始benchmark。
- PDF p.2：权重与索引均4bit导致50%存储开销；仅加速全连接且全部权重放SRAM，限制向巨型LLM直接迁移。

**边界：** 影响力与历史地位不是新方法证据；回顾的商业影响/后续工作未在本任务逐项外部核验。

**论文策略：** 承认当年的最优假设随工作负载迁移而失效，并据此找下一瓶颈。

**对我们的启发：** 不把方法延续性当有效性；VLM到T2I、repair到diagnosis改变了任务，旧证明必须重新成立。

**本次阅读范围：** 实际完整阅读这两页回顾正文与其限制说明；未逐项外部核验参考文献的历史影响。

<a id="2023-sige-tpami"></a>
### Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models - TPAMI extended version

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models - TPAMI extended version.pdf>) · [归档来源](https://arxiv.org/pdf/2211.02048v4) · 21页 · SIGE会议工作的期刊扩展；不得计作独立新主线

**问题：** 把SIGE从卷积式生成器扩展到注意力占主导的Stable Diffusion及更多实际编辑场景。

**痛点：** 原SIGE只优化卷积，Stable Diffusion注意力矩阵的成本留下，导致节省MAC却不明显加速。

**朴素方案为什么不够：** 直接把旧卷积引擎套到SD，约2.8%编辑仅1.1倍加速。

**洞察：** 未修改区域可复用query输出，只对编辑query计算注意力，仍保留全局key/value上下文。

**贡献：** attention query pruning、SD应用、Apple M1 GPU支持，以及大编辑/连续编辑/TensorRT检验。

**关键证据：**

- PDF p.2：作者明确列出相对于会议版的新增项：attention、SD、M1 GPU以及大/连续编辑。
- PDF p.11：SD消融：原369ms；仅旧卷积优化335ms/1.1倍；加入注意力后76.4ms/4.8倍。
- PDF p.7：SD数据集平均MAC805G→387G，FID27.2对26.8；不能把少数1%编辑的7.2倍当数据集平均加速。
- PDF p.11：明确限制文字prompt固定；改变文字需要重算缓存，不能直接用于文生图生成。
- PDF p.12：局部编辑导致全局光照变化时，编辑区外变化无法跟随；还可能有接缝。

**边界：** 缓存有额外内存，文本变化使复用失效；小编辑实例最大速度不是整体平均；与会议版不是独立重复突破。

**论文策略：** 扩展先展示旧方法在新瓶颈下几乎无效，再新增针对性机制并进行清晰消融。

**对我们的启发：** 真正pivot应像这里：旧方法1.1倍失败→新瓶颈→新机制4.8倍；仅重命名目标不算。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法和相关实验表/消融/讨论；长页选择相关段落。未逐字通读全部附录与参考文献。

<a id="2023-smoothquant"></a>
### SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/SmoothQuant- Accurate and Efficient Post-Training Quantization for Large Language Models.pdf>) · [归档来源](https://arxiv.org/pdf/2211.10438) · 13页 · 原始研究；所给PDF为2024修订版，按2023文件夹归类

**问题：** 让大语言模型W8A8量化同时保留精度和真实INT8硬件加速。

**痛点：** 少数稳定通道激活outlier压垮量化动态范围；逐通道激活量化虽准确却不匹配高效GEMM。

**朴素方案为什么不够：** 逐tensor/token量化使大模型近随机；LLM.int8保留FP16异常值可保精度却常更慢。

**洞察：** outlier沿输入通道稳定，而权重较容易量化；用等价缩放把困难从激活迁到权重并平衡两侧。

**贡献：** 离线融合X diag(s)^-1与diag(s)W的等价变换，校准迁移强度，兼容高效纯INT8矩阵乘法。

**关键证据：**

- PDF p.3：OPT-175B四任务平均FP16 71.6%，per-token INT8 31.7%，逐通道模拟71.4%；定位是通道尺度而非任意量化。
- PDF p.4：核心公式在未量化前严格保持线性层输出，缩放可离线融合；alpha控制激活与权重量化困难。
- PDF p.6：OPT-175B七任务平均FP16 66.9、SmoothQuant-O3 66.8，而朴素W8A8 35.5；WikiText perplexity10.99、11.17和93080。
- PDF p.8：优化FasterTransformer基线上仍最高1.56倍加速；alpha偏小/偏大分别使激活/权重量化失败。
- PDF p.13：明确W8A8与GPTQ weight-only服务场景及实现不同，拒绝直接不公平比较。

**边界：** 依赖校准分布与模型相关alpha；极端静态方案某些模型约降1点；不是在所有batch1场景胜过weight-only。修订版新增模型不属于初版时点。

**论文策略：** 用测量锁定结构，提出极简等价变换消除机制，再证明强实现下真实收益。

**对我们的启发：** 最强对照范式：我们的诊断若只证实outlier在哪里而没有新的可用变换/决策规则，贡献停在SmoothQuant之前。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法和相关实验表/消融/讨论；长页选择相关段落。未逐字通读全部附录与参考文献。

<a id="2023-tinyml-magazine"></a>
### Tiny Machine Learning: Progress and Futures [Feature]

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/Tiny Machine Learning- Progress and Futures [Feature].pdf>) · [归档来源](https://arxiv.org/pdf/2403.19076v2) · 24页 · 领域综述/研究线综合；所给PDF2024，按2023文件夹归类

**问题：** 梳理极小设备上推理和训练的资源约束、已有路线与未来问题。

**痛点：** 手机模型优化参数/FLOPs，MCU真正限制却常是激活峰值和运行时体积；从推理到训练又换瓶颈。

**朴素方案为什么不够：** 直接缩小MobileNet或把推理方案搬到训练，不能自然适应几百KB内存。

**洞察：** TinyML需要网络设计、执行顺序、量化优化和编译系统共同设计，且不同阶段的主要瓶颈会迁移。

**贡献：** 综合已有MCUNet系列、NetAug、TinyEngine/TTE与领域文献，提出应用/监督/模态扩展方向；不是独立新算法评估。

**关键证据：**

- PDF p.2：MobileNetV2相对ResNet参数小4.2倍，但推理峰值内存反大2.3倍；直接暴露选错代理目标。
- PDF p.12：汇总MCUNet-M4 int8在196KB SRAM达到64.9% ImageNet top-1；这是已有系列证据的综述整合。
- PDF p.15：复述QAS平均73.5对int8 SGD64.9，不能与MCUNetV3重复计为两份独立验证。
- PDF p.18：明确未来真实端侧学习监督获取困难，视觉检测/分割的标签昂贵；自监督是未来方向而非已经解决。

**边界：** 不是系统性无偏meta-analysis；大量本组工作复述；未来应用愿景不等于部署证据，引用原始研究需去重。

**论文策略：** 沿资源瓶颈如何迁移来组织研究线，而非把组件列表拼成贡献。

**对我们的启发：** 我们应将可验证诊断协议视为基础设施，之后必须挑一个具体用户任务兑现价值；不能把框架愿景当成果。

**本次阅读范围：** 实际阅读概述、资源瓶颈、TinyEngine/patch推理、关键汇总表、QAS/稀疏更新/TTE与展望的所列页；未逐项精读所有被综述文献，未把复述实验当独立验证。

<a id="2023-torchsparse-plus-plus"></a>
### TorchSparse++: Efficient Training and Inference Framework for Sparse Convolution on GPUs

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2023/TorchSparse++- Efficient Training and Inference Framework for Sparse Convolution on GPUs.pdf>) · [归档来源](https://dai.sjtu.edu.cn/my_file/pdf/c9839ed0-3108-4a86-9ada-b4fd8ad5bed3.pdf) · 15页 · 原始系统研究；在TorchSparse基础上扩展数据流与训练

**问题：** 降低高性能稀疏卷积的实现成本，并提升点云训练/推理速度。

**痛点：** 已有简单gather-GEMM-scatter不能重叠搬运与计算，高效手写方案复杂；算得少与真正快可能相反。

**朴素方案为什么不够：** 只选最低MAC、最低DRAM或最快卷积kernel会漏掉映射/排序成本；盲目沿用单一数据流也不行。

**洞察：** 稀疏卷积可分解为成熟dense MMA与少量稀疏访存模板；数据流应按共享映射分组、以端到端代价选择。

**贡献：** Sparse Kernel Generator、动态形状寻址/边界优化、扩展mask-split/unsorted/FoD设计空间，以及group-based autotuning。

**关键证据：**

- PDF p.7：按共享映射分组调优，避免每数据流重复建图；多数工作负载约2分钟调优，离线成本可解释。
- PDF p.8：A100相对SpConv2.3.5推理约1.7倍，训练约1.2–1.3倍；检测计时只覆盖SparseConv部分，应区别完整传感器管线。
- PDF p.9：nuScenes CenterPoint总时间：unsorted7.45ms、split1 8.18ms；不是算术量越少越好。
- PDF p.10：同条件纯卷积kernel则unsorted3.70ms、split1 3.68ms，排名反转由映射/排序开销导致。
- PDF p.8：采用新后端、假定2D部分TensorRT后，三帧CenterPoint在Orin比FlatFormer快1.5倍且精度更高；这是同研究组后续实现重置旧架构比较。
- PDF p.12：同数据流参数下kernel generator约1.1–1.2倍，代码量为SpConv元编程器约5%；设计空间再提供额外收益。

**边界：** 检测结果的端到端指被评估稀疏骨干，不含完整业务流程；ASIC对照含缩放投影，不是实芯片同配置比赛；不同硬件最优数据流不同。

**论文策略：** 把最强已有实现当比较对象，用分解测量解释反直觉结果，并让后续实现反驳自己的旧优越性。

**对我们的启发：** PurMM零分无法成为主胜利；我们必须对AtP/直接自适应查询/答案表等强且任务匹配的实现比较总成本。

**本次阅读范围：** 实际阅读所列PDF物理页的摘要/引言、方法和相关实验表/消融/讨论；长页选择相关段落。未逐字通读全部附录与参考文献。

## 2024：12份

| 论文 | 瞄准的问题 | 核心洞察 |
|---|---|---|
| [Atomique: A Quantum Compiler for Reconfigurable Neutral Atom Arrays](#2024-atomique) | 怎样把量子电路有效编译到可移动中性原子阵列，同时遵守移动及全局 Rydberg 激励的物理限制。 | 按交互图先分配阵列，再安排具体位置与合法并行移动；以物理误差模型而非单一深度衡量编译计划。 |
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](#2024-awq) | 如何在低位权重量化后维持大模型质量，并在真实部署硬件上兑现速度收益。 | 由激活幅度识别显著通道，用权重缩放及反向激活缩放实现等价变换，降低显著通道的相对量化误差；缩放过大也会恶化共享量化范围。 |
| [Condition-Aware Neural Network for Controlled Image Generation](#2024-can) | 条件图像生成是否必须让所有类别和去噪时间共享同一套静态网络权重。 | 将条件映射为少数关键操作的权重增量，主要覆盖 depthwise convolution、patch embedding 和 attention output，而不对所有层全面动态化。 |
| [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](#2024-distrifusion) | 如何用多 GPU 降低单张高分辨率扩散图像的延迟，而不是仅提高多图吞吐。 | 相邻去噪步的激活很相似，可以用上一时刻的其他块激活近似当前全局依赖，并将通信藏在计算后面；早期同步预热和统计修正控制近似误差。 |
| [EfficientViT-SAM: Accelerated Segment Anything Model Without Accuracy Loss](#2024-efficientvit-sam) | 如何保留 SAM 的通用提示分割能力，同时消除巨大图像编码器的部署瓶颈。 | 以硬件友好的多尺度线性注意力及局部卷积编码器替换重型 encoder，保留 prompt encoder 与 mask decoder；通过蒸馏和 SA-1B 端到端训练迁移能力。 |
| [FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention](#2024-fastcomposer) | 如何不用每个新主体再微调，就生成身份清楚、彼此不混淆且服从提示词的多主体图像。 | 身份绑定要落在对应名词的空间注意力上；先用文字决定构图，再在后续去噪中注入视觉身份条件，可调和布局与保真。 |
| [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](#2024-longlora) | 如何在有限 GPU 上把 LLM 微调到长上下文，并保留标准 full attention 推理。 | 跨头错开局部分组提供低成本跨组交流；embedding 和 normalization 的可训练性比简单提升 rank 更关键。 |
| [Q-Pilot: Field Programmable Qubit Array Compilation with Flying Ancillas](#2024-q-pilot) | 如何对可重构量子阵列生成浅而可扩展的电路，避免求解器编译时间随问题规模失控。 | 保持数据阵列，使用移动的 flying ancillas 介导远程交互，类比 FPGA 路由资源；针对应用结构优化并行扇出与门排序。 |
| [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](#2024-quest) | 长上下文生成时，如何只读取当前查询需要的 KV，而维持完整缓存下的回答质量。 | 每页 key 的逐通道最小/最大值可以便宜估计当前 query 对该页的潜在注意力，按 query 动态选择页面；不是永久驱逐 KV。 |
| [Sparse Refinement for Efficient High-Resolution Semantic Segmentation](#2024-sparserefine) | 高分辨率语义分割如何只在低分辨率预测容易出错的位置付出额外计算。 | 低分辨率熵能廉价召回大部分待修像素；只对选中位置的原始 RGB 做稀疏卷积，再由学习门控融合局部与全局判断。 |
| [Efficient Streaming Language Models with Attention Sinks](#2024-streamingllm) | 怎样让固定上下文预训练语言模型持续处理远超窗口的流，而不因缓存截断崩溃。 | 早期位置成为 attention sink，其作用主要不来自文字内容。保留少量初始 KV 加滚动近期 KV，并按缓存相对位置处理 RoPE，可以维持流式计算。 |
| [VILA: On Pre-training for Visual Language Models](#2024-vila) | VLM 预训练究竟应训练哪些参数、怎样组织图文数据，才能兼顾视觉能力、文本能力和上下文学习。 | 让语言模型在交错图文序列上参与预训练，用简单投影器，并在 SFT 混入文本数据，比分别堆大投影器或额外视觉专家更有效。 |

<a id="2024-atomique"></a>
### Atomique: A Quantum Compiler for Reconfigurable Neutral Atom Arrays

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/Atomique- A Quantum Compiler for Reconfigurable Neutral Atom Arrays.pdf>) · [归档来源](https://arxiv.org/pdf/2311.15123) · 17页 · 研究论文；版本与归档元数据见清单

**问题：** 怎样把量子电路有效编译到可移动中性原子阵列，同时遵守移动及全局 Rydberg 激励的物理限制。

**痛点：** 固定连接需要大量 SWAP；可移动原子也不是任意连线，AOD 行列不能随意交叉，错误并行可能诱发非目标门。

**朴素方案为什么不够：** 直接套固定架构路由浪费可重构性；任意移动不合法；只优化门深度忽略原子移动、转移、加热和退相干，会误判真实收益。

**洞察：** 按交互图先分配阵列，再安排具体位置与合法并行移动；以物理误差模型而非单一深度衡量编译计划。

**贡献：** 粗粒度加权 max-k-cut 映射、细粒度负载与位置映射、可扩展贪心路由，以及门/移动/转移/冷却代价模型。

**关键证据：**

- PDF p.8：评测明确使用自建模拟器：现有公开设备不支持论文需要的完整移动控制，硬件参数取自已有实验；不是在目标设备执行得到的保真度。
- PDF p.9：Figure 13 正文汇总：相对最强固定三角阵列基线，模型保真度、双量子门数、深度分别改善约 2.7/2.8/2.2 倍。
- PDF p.10：与 Tan-Solver 在可完成的小电路比较，Atomique 的平均保真度为 88%、与 solver-based compilers 相当，而编译快超过 2000 倍；solver 超过 20 qubits 常遇 24 小时超时。这里的 88% 是论文报告的平均保真度，不是相对 solver 的比例。
- PDF p.11：敏感性分析：当双量子门保真度高于约 0.9999，固定阵列可优于可重构阵列；移动时间在加热与退相干间存在折中。

**边界：** 结果依赖模拟器及硬件误差假设；不同架构比较也包含物理能力差异。低连接度或极高双门保真度时固定阵列可更优，不能写成可移动体系普遍胜出。

**论文策略：** 物理约束定义问题，再把映射/路由和多因素目标协同优化；包含严格 solver 参照、规模曲线和方法输掉的参数区域。

**对我们的启发：** 状态替换后恢复率属于限定算子的目标，不是根因或未来修复保证；诊断应把 donor、干预合法性和保护事实当作约束，并公开方法在哪些条件下无效。

**本次阅读范围：** 精读体系约束、三级编译方法、保真度模型、主要比较、solver 与参数敏感性；图中数值以正文/表述可核实者为准。

<a id="2024-awq"></a>
### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/AWQ- Activation-aware Weight Quantization for LLM Compression and Acceleration.pdf>) · [归档来源](https://arxiv.org/pdf/2306.00978) · 15页 · 研究论文；版本与归档元数据见清单

**问题：** 如何在低位权重量化后维持大模型质量，并在真实部署硬件上兑现速度收益。

**痛点：** 逐权重均匀量化会伤害少数重要通道；混合精度保护虽有效，却不利于常规整数内核。

**朴素方案为什么不够：** 保护大权重或随机权重并不等于保护对输出重要的权重；依赖逐层重构也可能过拟合校准分布。

**洞察：** 由激活幅度识别显著通道，用权重缩放及反向激活缩放实现等价变换，降低显著通道的相对量化误差；缩放过大也会恶化共享量化范围。

**贡献：** 小样本、无反传的缩放与截断搜索，加 TinyChat 的融合、打包内核，实现算法与硬件协同。

**关键证据：**

- PDF p.4：Table 1：OPT-6.7B INT3-g128 PPL：FP16 10.86，RTN 23.54，保护激活选出的 1% 权重 11.39，权重幅度选择 22.37，随机选择 24.23。
- PDF p.4：Table 2：同一量化设置，缩放 s=1、2、4 的 PPL 为 23.54、11.92、12.36，证明并非缩放越大越好。
- PDF p.8：Table 6：OpenFlamingo-9B 的 COCO 32-shot CIDEr：FP16 81.70，RTN 77.13，GPTQ 74.98，AWQ 80.53；跨多模态检验而非只看语言 PPL。
- PDF p.10：Figure 8：分布外校准时 AWQ 的 PPL 增量约 0.5–0.6，GPTQ 约 2.3–4.9；Figure 9 报告 TinyChat 相对 Hugging Face FP16 最多 3.9 倍/3.5 倍的 4090/Orin 收益。

**边界：** 仍需校准且低位仍可能掉点；主要是 weight-only、低批量内存带宽瓶颈场景。TinyChat 对 HF 的收益同时含系统实现优势，不能全部归于显著性规则。归档 PDF 含后续修订，按文件夹记为 2024。

**论文策略：** 先用同预算的随机/权重幅值反例建立重要性定义，再以可部署的等价变换替代不便实现的混合精度，最终闭合质量—延迟证据。

**对我们的启发：** 诊断排名应在相同信息和相同操作预算下证明比简单替代项更有用；必须报告真实恢复与完整成本，不能只报告排名相似性。

**本次阅读范围：** 精读问题、显著通道实验、缩放推导与消融、TinyChat、语言/多模态实验及校准泛化；页码为本地 PDF 页序。

<a id="2024-can"></a>
### Condition-Aware Neural Network for Controlled Image Generation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/Condition-Aware Neural Network for Controlled Image Generation.pdf>) · [归档来源](https://arxiv.org/pdf/2404.01143) · 10页 · 研究论文；版本与归档元数据见清单

**问题：** 条件图像生成是否必须让所有类别和去噪时间共享同一套静态网络权重。

**痛点：** 条件只注入特征时，静态权重可能难兼顾不同生成子任务；完整条件专家又不可承受。

**朴素方案为什么不够：** 增加每处动态权重或只改善特征调制并不稳定获益；部署时逐样本不同权重还可能降低硬件效率。

**洞察：** 将条件映射为少数关键操作的权重增量，主要覆盖 depthwise convolution、patch embedding 和 attention output，而不对所有层全面动态化。

**贡献：** Condition-Aware Network 与 CaT，选择性条件权重生成、跨投影共享、分组卷积批处理实现；将条件适配与轻量骨干组合。

**关键证据：**

- PDF p.3：Table 1：同 UViT-S/2，基线 FID 28.32；条件 DWConv 为 11.18，再加 patch embedding 为 10.23，再加 output projection 为 8.82；额外动态 head 反而为 12.29。
- PDF p.5：条件时间单独 FID 15.16，类别单独 10.01，两者 8.82；长训练后的差距仍存在，反驳仅加快收敛这一解释。
- PDF p.6：Table 4：ImageNet 512 的 CaT-L0 FID 2.78，约 10G MAC/step、20 steps；DiT-XL FID 3.04，525G/step、250 steps。
- PDF p.7：Table 5：Orin 上 45.9 秒到 0.2 秒的 229 倍同时包含骨干与步数变化，不能归于条件权重这一个模块。
- PDF p.4：理论额外计算很小，但 NVIDIA 对 grouped convolution 支持较差，实际仍有约 30–40% 训练耗时增量；文中将自定义内核留作未来工作。

**边界：** 负迁移解释更多是动机，并非已被完整识别的因果机制。最终速度比较包含架构和采样器差异；动态参数生成存在参数与训练开销，主要验证 ImageNet/COCO。

**论文策略：** 大幅总系统结果由同架构递增消融支撑；反例明确告诉读者哪些看似自然的额外适配反而无效，避免把最终套餐增益全归一模块。

**对我们的启发：** 如果 Verified Diagnosis 加入选择器、校准器和大量验证查询，必须拆开每个部件的独立收益，不能把更大信息预算包装成诊断更准。

**本次阅读范围：** 精读条件权重方法、同骨干 Tables 1–3 消融、长训练控制、质量/复杂度与设备计时主结果；页码为 PDF 页序。

<a id="2024-distrifusion"></a>
### DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/DistriFusion- Distributed Parallel Inference for High-Resolution Diffusion Models.pdf>) · [归档来源](https://arxiv.org/pdf/2402.19481) · 12页 · 研究论文；版本与归档元数据见清单

**问题：** 如何用多 GPU 降低单张高分辨率扩散图像的延迟，而不是仅提高多图吞吐。

**痛点：** 数据并行不能缩短单图延迟；独立生成分块会破坏全局一致性；同步分块或 tensor parallel 被通信拖慢。

**朴素方案为什么不够：** 完全丢弃跨块依赖虽快却出错；每步同步所有激活虽精确却难扩展；直接使用陈旧归一化统计也造成分布错配。

**洞察：** 相邻去噪步的激活很相似，可以用上一时刻的其他块激活近似当前全局依赖，并将通信藏在计算后面；早期同步预热和统计修正控制近似误差。

**贡献：** Displaced patch parallelism、异步 AllGather、校正陈旧 GroupNorm、同步 warm-up；围绕同一图像质量和端到端延迟实现设计。

**关键证据：**

- PDF p.8：Table 1：SDXL 1024、50-step DDIM，原始 5.02 秒/FID 24.0，8-GPU 方案 1.77 秒/24.4、同为 338T 总 MAC；独立分块虽 1.27 秒，FID 高达 252。
- PDF p.8：Table 2：1024 下同步 tensor parallel、同步 patch parallel 与本方法为 3.61、2.21、1.77 秒；3840 下原始 140 秒、本方法 22.9 秒，约 6.1 倍。
- PDF p.9：Figure 8：10-step DPM 下零预热 LPIPS 0.404，加两步预热为 0.196，延迟 0.374 到 0.400 秒，直接显出近似误差与修正收益。
- PDF p.9：Figure 9：separate/stale/sync/corrected GN 的 LPIPS 为 0.317/0.247/0.207/0.211；校正方案接近同步质量且延迟 1.77 秒低于 1.85 秒。

**边界：** 充分隐藏通信依赖 NVLink；低分辨率利用率不足，极少去噪步可能失效。近似输出并非与原图完全一致，8 GPU 降延迟也不能自动理解为八倍计算成本节约。

**论文策略：** 先用独立块失败证明必要交互，再用精确同步基线隔离近似误差；针对误差来源逐项修正，同时报告延迟、质量和总计算。

**对我们的启发：** Oracle 或穷举是必要的精确参照，但论文价值来自保留其决策质量的低成本近似，不能只证明原来的便宜代理不等于 oracle。

**本次阅读范围：** 精读单图并行问题、各算子通信与 GN 方法、Tables 1–2、预热/GN 消融及 Limitations；页码为 PDF 页序。

<a id="2024-efficientvit-sam"></a>
### EfficientViT-SAM: Accelerated Segment Anything Model Without Accuracy Loss

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/EfficientViT-SAM- Accelerated Segment Anything Model Without Accuracy Loss.pdf>) · [归档来源](https://arxiv.org/pdf/2402.05008v2) · 5页 · 研究论文；版本与归档元数据见清单

**问题：** 如何保留 SAM 的通用提示分割能力，同时消除巨大图像编码器的部署瓶颈。

**痛点：** SAM ViT-H 编码器计算量高；直接缩小骨干会损失分割质量，参数少也不保证硬件速度快。

**朴素方案为什么不够：** 单纯替换成小网络或只比较 FLOPs 不能保证提示分割质量与实际吞吐；应保持任务接口与评测方式。

**洞察：** 以硬件友好的多尺度线性注意力及局部卷积编码器替换重型 encoder，保留 prompt encoder 与 mask decoder；通过蒸馏和 SA-1B 端到端训练迁移能力。

**贡献：** EfficientViT 的 SAM 集成和训练方案，提供不同速度质量档位，并评估 COCO/LVIS 点与框提示及 SGinW 零样本分割。

**关键证据：**

- PDF p.3：Table 1：A100 TensorRT FP16，SAM 11 images/s、COCO box mAP 46.5；EfficientViT-SAM-L2 为 538 images/s、46.6，约 48.9 倍；L0 更快但 mAP 45.7。
- PDF p.3：Table 2：LVIS 单点提示 SAM mIoU 59.2，XL1 为 56.6；3/5 点则 XL1 67.0/71.7 高于 SAM 66.0/66.8，标题不能解释为逐指标无损。
- PDF p.4：SGinW 25 datasets 配合 GroundingDINO 的 box-prompted mAP，XL1 为 48.9，SAM 为 48.7；增加分布外任务层面的检验。

**边界：** 报告的是特定 A100/TensorRT 的吞吐，非所有应用端到端延迟。个别模型/数据/提示指标下降；短文未给出充分拆分蒸馏与端到端训练的独立消融。

**论文策略：** 以明确部署瓶颈和大幅真实速度—质量改进支撑集成贡献，不强求复杂新理论；用跨提示和跨数据评测限定普适性。

**对我们的启发：** 工程性工作也能 solid，但要有大而可靠的真实收益；严格记账与可复现接口只是基础，不能替代更有用、更便宜的诊断。

**本次阅读范围：** 读取正文 pp.1–4 的摘要、架构/训练、主表和零样本评测；p.5 为参考文献。未将标题当作所有指标的事实。

<a id="2024-fastcomposer"></a>
### FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/FastComposer- Tuning-Free Multi-Subject Image Generation with Localized Attention.pdf>) · [归档来源](https://arxiv.org/pdf/2305.10431) · 12页 · 研究论文；版本与归档元数据见清单

**问题：** 如何不用每个新主体再微调，就生成身份清楚、彼此不混淆且服从提示词的多主体图像。

**痛点：** 逐主体优化花数分钟及显存；直接混入多张参考图时身份串扰；过早强调参考身份会牺牲文字指定布局。

**朴素方案为什么不够：** 把所有身份 embedding 一起输入并不足以将主体与正确名词、区域绑定；单纯加强图像条件也会降低可编辑性。

**洞察：** 身份绑定要落在对应名词的空间注意力上；先用文字决定构图，再在后续去噪中注入视觉身份条件，可调和布局与保真。

**贡献：** 共享图像编码器与增强名词 embedding、训练期注意力定位监督、推理期延迟主体条件；测试时只需参考图前向传播。

**关键证据：**

- PDF p.5：方法使用训练期分割掩码约束对应主体注意力，并在推理中延迟图像条件；定位与时间调度分别针对身份混合和提示词可编辑性。
- PDF p.8：Table 1：单主体 FastComposer 身份相似度 51.41、prompt 24.30，总个性化加生成耗时 2 秒；Custom Diffusion 为 43.37/23.29、789 秒。参考图数量和方法信息预算不同，不能忽略。
- PDF p.9：Table 2：多主体身份相似度 FastComposer 43.11，Custom Diffusion 5.37，但 prompt 指标分别 24.25/25.84，存在取舍。
- PDF p.9：Table 3：去掉 attention localization 时身份/prompt 为 37.66/25.03，加入后为 43.11/24.25；消融直接检验身份与提示词的收益代价。

**边界：** 共享模型先训练 150K steps、8 张 A6000；2 秒不含这部分摊销训练。训练和主要评测面向人物头照，超过三个主体有限，动物及一般实体未充分验证。注意力监督不等于注意力本身是真实原因证明。

**论文策略：** 把失败拆成空间串扰与时间上的构图冲突，用对应机制修正，报告个性化总时间和身份—提示词双目标取舍。

**对我们的启发：** 定位的价值由最终身份或事实恢复决定；应报告保护目标掉点，明确共享训练成本与每例诊断成本的摊销关系。

**本次阅读范围：** 精读问题、条件编码、定位损失、延迟条件、Tables 1–3、训练成本与局限；页码为 PDF 页序。

<a id="2024-longlora"></a>
### LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/LongLoRA- Efficient Fine-tuning of Long-Context Large Language Models.pdf>) · [归档来源](https://arxiv.org/pdf/2309.12307) · 19页 · 研究论文；版本与归档元数据见清单

**问题：** 如何在有限 GPU 上把 LLM 微调到长上下文，并保留标准 full attention 推理。

**痛点：** 长序列训练注意力二次增长；LoRA 虽省参数，直接用仍明显落后全量微调；孤立局部注意力缺乏跨组信息。

**朴素方案为什么不够：** 只增大 LoRA rank 不能解决质量缺口；未移位的局部稀疏注意力切断通信，训练期的任意稀疏结构也未必兼容 dense 推理。

**洞察：** 跨头错开局部分组提供低成本跨组交流；embedding 和 normalization 的可训练性比简单提升 rank 更关键。

**贡献：** Shifted short attention 用于训练，加 LoRA、可训练 embedding/norm；推理仍可切回标准完整注意力。

**关键证据：**

- PDF p.4：Table 1：32768 长度 PG19，full attention PPL 8.04，未移位 short attention 9.47，shifted short attention 8.08。
- PDF p.6：Table 2：32K 时 full FT PPL 8.08；LoRA rank 8 为 11.44、rank 256 为 11.98；加入 embedding 与 norm 后为 8.12，参数容量增加并未替代组件选择。
- PDF p.7：Table 4：单机 8×A100 可扩展 7B/13B/70B 到 100K/64K/32K；另有长文检索及指令任务，而非只测内存。
- PDF p.9：Table 6：跨头 shift 的 full-attention 推理 PPL 8.12，dilate 为 11.78、stride 为 24.03，检验训练近似对正式推理的兼容性。

**边界：** 核心省的是微调成本，不是提出稀疏解码引擎；位置插值可能损害短上下文困惑度。主要为 Llama-2、给定微调和位置设置，训练稀疏的充分理论仍有限。

**论文策略：** 用失败的容量扩展与注意力模式控制，定位缺少的通信和参数更新路径；便宜训练必须回到正式推理方式验证。

**对我们的启发：** 诊断器低分不能仅靠“更大模型/更多样本”解释；需要排除容量不足、训练目标错配、算子不适用等具体原因，并验证正式使用流程。

**本次阅读范围：** 精读稀疏分组与可训练参数方法、Tables 1–2/4/6、检索和成本实验及短上下文边界；页码为 PDF 页序。

<a id="2024-q-pilot"></a>
### Q-Pilot: Field Programmable Qubit Array Compilation with Flying Ancillas

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/Q-Pilot- Field Programmable Qubit Array Compilation with Flying Ancillas.pdf>) · [归档来源](https://arxiv.org/pdf/2311.16190) · 10页 · 研究论文；版本与归档元数据见清单

**问题：** 如何对可重构量子阵列生成浅而可扩展的电路，避免求解器编译时间随问题规模失控。

**痛点：** 移动数据量子位受阵列整体几何约束；精确 SMT 方法在大电路上超时；固定连线体系需要深的通信与门序列。

**朴素方案为什么不够：** 将数据原子自由移动并不合法；为所有电路采用同一通用门序路由，不能利用 QSim/QAOA 的结构；最优 solver 不能覆盖目标规模。

**洞察：** 保持数据阵列，使用移动的 flying ancillas 介导远程交互，类比 FPGA 路由资源；针对应用结构优化并行扇出与门排序。

**贡献：** Flying-ancilla 编译框架、通用贪心路由、QSim 与 QAOA 专用路由和可扩展编译实现。

**关键证据：**

- PDF p.7：小实例比较中 solver 可有更低深度，但往往一小时内不完成；Q-Pilot 通常一秒内，生成深度不超过其最优解约四倍，明确近似代价。
- PDF p.8：Figure 16：QSim 专用路由相对通用路由减少双门数 1.5 倍、深度 8.8 倍；QAOA 为 2.8 倍和 10.1 倍，说明收益来自结构化路由而非仅硬件变换。
- PDF p.8：扩展性测试：2000 qubits 的 QAOA、QSim、通用随机电路编译分别为 129.50、30.48、32.31 秒；给出可实际运行的规模证据。
- PDF p.8：Figure 15 通过门错误等参数估计电路错误率，并展示并行门分布；不是硬件实测的任务成功率。

**边界：** 需要额外 ancillas 及其交互、回收操作；降低 circuit depth 不自动等于更高实际保真度，且结果依赖架构和误差模型。主要证据为编译/模拟而非实机成功。

**论文策略：** 把新增硬件自由度变成清楚的编译抽象，再以专用对通用、近似对精确、规模对时间三条控制分离贡献。

**对我们的启发：** 不要把诊断粒度、覆盖率或验证通过率当作最终价值；当不同指标冲突时，需要以实际事实恢复及全部额外操作成本裁决。

**本次阅读范围：** 精读摘要、ancilla 与合法路由方法、通用/专用框架、solver 对照、规模和误差实验；页码为 PDF 页序。

<a id="2024-quest"></a>
### Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/Quest- Query-Aware Sparsity for Efficient Long-Context LLM Inference.pdf>) · [归档来源](https://arxiv.org/pdf/2406.10774) · 11页 · 研究论文；版本与归档元数据见清单

**问题：** 长上下文生成时，如何只读取当前查询需要的 KV，而维持完整缓存下的回答质量。

**痛点：** 解码受 KV 读取带宽限制；历史重要 token 未必对下一个查询重要；静态驱逐可能永久删除稍后需要的信息。

**朴素方案为什么不够：** 过去注意力分数和固定首尾窗口不能适应查询变化；精确计算所有注意力再选稀疏 token 又失去省带宽意义。

**洞察：** 每页 key 的逐通道最小/最大值可以便宜估计当前 query 对该页的潜在注意力，按 query 动态选择页面；不是永久驱逐 KV。

**贡献：** 页级 min/max 元数据与 query-aware criticality、Top-K 页注意力内核；前两层保留 dense，计入选页开销实现解码提速。

**关键证据：**

- PDF p.3：Figure 3：前两层难稀疏，而后续层具有显著稀疏性；实验为各方法保留前两层 dense，避免不匹配层处理。
- PDF p.5：Table 1：10K passkey、LongChat、64 token budget，Quest 为 99%，H2O、TOVA、StreamingLLM 各约 1%；100K 场景使用不同模型及预算，不应直接合并。
- PDF p.6：LongBench/长序列 H2O 实验为兼容 FlashAttention，仅在 decode 累积历史注意力；这限定了对 H2O 原方法的泛化批评。
- PDF p.8：Figures 9–10 及正文：32K 序列、2048 budget 的注意力内核最多 7.03 倍，相对 FlashInfer 的每 token 解码延迟，FP16 改善 1.74 倍、4-bit 改善 2.23 倍；包含选页步骤，但不计 sampling，也非完整请求 prefill+decode 延迟。

**边界：** 主要加速 decode，仍存完整 KV，并非缓存容量压缩；效果随预算/任务改变。部分基线没有高效内核，只比较质量，不能宣称测得所有方法的系统速度优势；H2O 历史收集适配须披露。

**论文策略：** 发现重要性依赖当前条件，从静态重要性改为低成本动态上界；既检验困难检索，也明确系统分母和筛选成本。

**对我们的启发：** 诊断排名也可能依赖问题、供体和干预类型。测试同一排名跨条件失效应建立机制边界，不能直接将任务适配后的失败归罪于原方法。

**本次阅读范围：** 精读 query-dependent 动机、元数据与内核方法、passkey/LongBench、计时与基线适配注释；页码为 PDF 页序。

<a id="2024-sparserefine"></a>
### Sparse Refinement for Efficient High-Resolution Semantic Segmentation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/Sparse Refinement for Efficient High-Resolution Semantic Segmentation.pdf>) · [归档来源](https://arxiv.org/pdf/2407.19014) · 20页 · 研究论文；版本与归档元数据见清单

**问题：** 高分辨率语义分割如何只在低分辨率预测容易出错的位置付出额外计算。

**痛点：** 低分辨率模型速度快但丢失细小对象和边界；全图高分辨率或稠密 patch 精修会重复计算大量简单区域。

**朴素方案为什么不够：** 随机抽点命中错误少；仅插值/隐藏特征精修可能缺少原始高频信息；把局部精修输出直接替换全局预测会丢失上下文。

**洞察：** 低分辨率熵能廉价召回大部分待修像素；只对选中位置的原始 RGB 做稀疏卷积，再由学习门控融合局部与全局判断。

**贡献：** 冻结现有分割器、熵选择器、稀疏 UNet、门控融合的后接框架，面向真实延迟而非只减少像素数。

**关键证据：**

- PDF p.8：Table 1：SegFormer-B5 全分辨率为 140.6ms/81.1 mIoU，半分辨率为 18.5ms/78.7，加 SparseRefine 为 38.5ms/81.2，约 3.7 倍。
- PDF p.9：匹配开销下 PointRend 为 32.8ms/79.9 mIoU，本方法 32.4ms/80.9；稠密 patch 消融也更慢，区分后接精修与稀疏实现的价值。
- PDF p.12：选择消融：熵选 11.8% 像素，错误召回 84.9%、2.3ms；学习选择器 85.9%、4.0ms 且同为 80.9 mIoU，复杂选择器没有实用优势。
- PDF p.12：融合消融：直接替换为 77.7 mIoU，熵融合 80.3，学习门控 80.9；随机 10% 选择仅 10% 错误召回，最终 79.2。

**边界：** 低熵的自信错误容易漏掉；精修网络仍需训练且增加模型参数，不等于内存也下降。主要是分割密集预测，不是证明 entropy 能识别语义根因。

**论文策略：** 便宜选择器只需足够召回，并非越复杂越好；将选择、局部修复、融合各自失败拆解，用同延迟强基线确认真实收益。

**对我们的启发：** 我们必须允许 output-only、entropy、直接检查赢；诊断更精细若不提高修复成功率或降低预算就没有独立价值，拒绝定位也应有明确覆盖代价。

**本次阅读范围：** 精读选择/稀疏精修/融合方法、主表及 PointRend 对照、patch/选择器/融合消融和速度分解；页码为 PDF 页序。

<a id="2024-streamingllm"></a>
### Efficient Streaming Language Models with Attention Sinks

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/Efficient Streaming Language Models with Attention Sinks.pdf>) · [归档来源](https://arxiv.org/pdf/2309.17453) · 21页 · 研究论文；版本与归档元数据见清单

**问题：** 怎样让固定上下文预训练语言模型持续处理远超窗口的流，而不因缓存截断崩溃。

**痛点：** 完整 KV 缓存随流增长；滑窗重算成本高；直接驱逐最早 token 会使困惑度突然爆炸。

**朴素方案为什么不够：** 只留最近 token 虽符合局部信息直觉，却移除了模型用于承接多余注意力的初始位置；将初始 token 解释为重要语义也不能解释换行替换实验。

**洞察：** 早期位置成为 attention sink，其作用主要不来自文字内容。保留少量初始 KV 加滚动近期 KV，并按缓存相对位置处理 RoPE，可以维持流式计算。

**贡献：** 训练无关的初始加近期缓存策略；通过内容替换、初始 token 数消融及从头训练的可学习 sink，对机制解释作可区分实验。

**关键证据：**

- PDF p.5：Table 1：Llama-2-13B 的 0+1024 缓存 PPL 为 5158.07，4+1020 为 5.40，将前四 token 换成换行仍为 5.60，区分语义重要性与初始位置作用。
- PDF p.6：Table 3：相同 160M 训练设置下，1+1023 缓存的 vanilla、zero sink、learnable sink PPL 为 18.49、19.90、18.01；learnable sink 用一个初始 token 已接近其更多 token 情形。
- PDF p.8：Table 5：连续 ARC 评测中，Llama-2-7B Chat 的 window attention 为 3.58/1.39，StreamingLLM 为 71.34/55.03，接近或超过单题模式 71.25/53.16。
- PDF p.9：Figure 10：在同一 Hugging Face 实现及 A6000 上，相对 sliding-window recomputation 最多加速 22.2 倍；这不是相对普通窗口截断的加速。

**边界：** 稳定近期窗口的流式推理不等于无限远信息检索，也没有扩大模型注意力窗口。软最大值动机受到实验支持，但并非普适性定理。22.2 倍的分母是逐窗重算。

**论文策略：** 以突变失败锁定结构性因素；设计能击破语义替代解释的反事实；用最小缓存修改修复，再验证实际任务及速度。

**对我们的启发：** Verified Diagnosis 应拿出换行替代这样的区分性控制：定位图变化究竟来自原因、答案信号还是干预算子。声明必须限定为已测操作下的可恢复性。

**本次阅读范围：** 精读摘要/引言、缓存及位置方法、Tables 1–3 与 5、运行时间图和上下文边界；页码为本地 PDF 页序。

<a id="2024-vila"></a>
### VILA: On Pre-training for Visual Language Models

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2024/VILA- On Pre-training for Visual Language Models.pdf>) · [归档来源](https://arxiv.org/pdf/2312.07533) · 16页 · 研究论文；版本与归档元数据见清单

**问题：** VLM 预训练究竟应训练哪些参数、怎样组织图文数据，才能兼顾视觉能力、文本能力和上下文学习。

**痛点：** 只比较零样本视觉平均分会掩盖模型丧失 ICL 或文本能力；更复杂投影器也未必带来更好对齐。

**朴素方案为什么不够：** 冻结语言模型、只用独立图文对或仅做视觉指令微调，可能获得尚可零样本分数，却损害跨模态序列建模与文本能力。

**洞察：** 让语言模型在交错图文序列上参与预训练，用简单投影器，并在 SFT 混入文本数据，比分别堆大投影器或额外视觉专家更有效。

**贡献：** 受控训练配方研究：同源 MMC4 拆成图文对的控制、训练组件与 ICL 消融、文本和视觉联合 SFT、分辨率与 token 数分离实验。

**关键证据：**

- PDF p.3：Table 1：同为 MMC4 与 transformer projector，冻结预训练 LLM 的平均 0/4-shot 为 66.8/57.6，训练 LLM 为 66.8/68.8；同一个零样本分数掩盖 11.2 分 ICL 差距。
- PDF p.4：Table 3：MMC4 拆成 pairs 的平均 0/4-shot 为 46.4/44.5，保留交错组织为 68.7/70.9；同源控制比不同语料直接比较更能支持组织形式的作用。
- PDF p.6：Table 5：VILA-7B TextVQA 为 64.4，LLaVA-1.5-7B 为 58.2；Table 6 中 VILA-7B MMLU 50.8，文本 SFT 参考为 51.8，不能概括所有规模无文本损失。
- PDF p.8：Table 7：图像分辨率 224 到 336 的 TextVQA 为 41.6 到 49.8；将高分辨率视觉 token 降采样至 144 后仍为 45.6，区分原始分辨率与 token 数。

**边界：** 深层表示更相似与深度对齐之间仍有解释性推断；主要在其模型、50M 图像及所选配方内验证。不同指标仍有退步，不能将总体进步写成全面无损。 MMC4-pairs 是同源控制而非全部 token 严格不变：只保留匹配图像及相应文本段，丢弃了其他文本，不能声称已完全分离序列组织与上下文信息量。

**论文策略：** 把经验配方拆成可证伪决策，用同源数据重排/拆对和同组件训练控制排除混杂，再增加容易被平均分掩盖的下游能力。

**对我们的启发：** 应固定案例、供体和算子比较诊断器，并检查保护事实；若换了任务或操作再说别人的方法失效，就缺少这里的同源控制。

**本次阅读范围：** 精读摘要、预训练设计、Tables 1–9 的关键对照和数据/能力边界；未将表示相似性视作因果证明。

## 2025：21份

| 论文 | 瞄准的问题 | 核心洞察 |
|---|---|---|
| [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](#2025-coat) | 降低FP8训练中仍由高精度优化器状态和非线性激活占用的显存。 | 用幂变换扩展优化器状态的动态范围；为非线性激活采用不跨token的细粒度缩放，为GEMM采用较粗粒度并融合转换。 |
| [Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](#2025-dc-ae) | 让高空间压缩的图像自编码器保持重建质量，以减少扩散模型的token。 | 用无参数space-to-channel路径提供可恢复的参考，再让网络只学习残差；把高分辨率适配与局部细节精修拆为训练阶段。 |
| [DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space](#2025-dc-ae-1-5) | 解决高通道潜空间重建更好、扩散模型反而更难收敛的问题。 | 用随机前缀通道保留强迫早通道承载结构，再以相匹配的通道增强训练扩散模型。 |
| [DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](#2025-dc-ar) | 在高压缩token下兼顾自回归图像生成速度、细节和分辨率泛化。 | 深压缩二维混合tokenizer将结构留在离散token、残差细节交给小扩散模块；按连续预热、离散学习、交替解码训练。 |
| [DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space](#2025-dc-gen) | 不从头训练扩散大模型，把现有模型迁移到深压缩潜空间以加速生成。 | 先对齐patch嵌入，再冻结主干对齐输出，最后用LoRA联合微调，逐步恢复接口兼容性。 |
| [DC-VideoGen: Efficient Video Generation with Deep Compression Video Autoencoder](#2025-dc-videogen) | 在保持视频重建和可变时长能力的同时压缩视频token，并迁移现有视频扩散器。 | chunk内部双向、chunk之间因果，随后用输入输出接口对齐及LoRA迁移生成器。 |
| [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](#2025-duo-attention) | 降低长上下文LLM的KV缓存与注意力成本，保留真正需要全局检索的头。 | 用限制为sink+近期token后对输出的反事实影响定义retrieval head；合成多passkey蒸馏学习混合门，再二值化部署。 |
| [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](#2025-hart) | 兼顾自回归视觉生成的快速全局建模和高频细节。 | 共享解码器交替接收离散/连续latent，使主AR负责粗结构、小残差扩散器负责末端细节。 |
| [Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](#2025-jet-nemotron) | 以较低架构探索代价获得真正高吞吐且保持任务能力的线性/混合注意力模型。 | 复用预训练MLP，以可训练门和搜索决定全注意力层位置；再选动态卷积线性模块与硬件合适的头形状。 |
| [LEGO: Spatial Accelerator Generation and Optimization for Tensor Applications](#2025-lego) | 从张量计算自动生成可综合的空间加速器，并优化数据流和硬件映射。 | 用仿射迭代和时空映射表达数据流，推导连接与存储，再在生成图上优化引脚共享、广播、归约和门控。 |
| [Long-RL: Scaling RL to Long Videos](#2025-long-rl) | 让VLM有效训练和推理长视频，而非仅增加输入帧数。 | 构建长视频推理QA，先CoT-SFT再GRPO；系统重复使用视觉序列并按模态均衡序列并行。 |
| [LongVILA: Scaling Long-Context Visual Language Models for Long Videos](#2025-longvila) | 让VLM在可承受训练成本下利用小时级视频和百万级多模态上下文。 | 先按图像负载分片再重分配token，按机内/机间拓扑选择通信；训练分阶段补齐对齐、短上下文、扩展和长视频能力。 |
| [LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention](#2025-lserve) | 同时减少LLM长输入prefill和长输出decode的注意力成本。 | 静态流式头与动态检索页互补；分离逻辑/物理页粒度，并跨连续query复用页选择。 |
| [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](#2025-qserve) | 让低比特LLM在云端批量服务中获得真实吞吐收益。 | 保留高效INT8计算路径，采用W4A8KV4、渐进组量化、SmoothAttention及针对反量化/寄存器访问的执行设计。 |
| [Radial Attention: O(nlogn) Sparse Attention with Energy Decay for Long Video Generation](#2025-radial-attention) | 降低长视频生成注意力的计算增长，同时保留时空结构。 | 利用平均能量随时空距离衰减，按时距以层级方式缩减空间带宽，保留首帧sink并配合块稀疏kernel。 |
| [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](#2025-sana) | 在较低资源下完成高分辨率文本到图像生成。 | 深压缩AE先减少token，线性注意力负责全局、局部MixFFN补空间细节，再用kernel融合、较小文本编码器和更少步数解决残余成本。 |
| [SANA-1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer](#2025-sana-1-5) | 在有限训练成本和可增加的推理预算下扩展SANA能力。 | 复制并近似恒等初始化新层以增长模型，结合稳定训练和剪枝后微调；推理阶段生成候选并用VLM选择。 |
| [SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](#2025-sana-sprint) | 将高效流匹配图像模型压缩到一至数步生成，同时保持结构和细节。 | 转换为TrigFlow坐标并稳定时间与注意力参数化，用连续一致性约束配合latent对抗目标。 |
| [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](#2025-svdquant) | 让扩散模型权重和激活同时4bit量化而不发生严重质量崩坏，并兑现硬件加速。 | 先把激活困难迁移到权重，再由高精度低秩分支吸收权重主奇异方向，低比特只处理残差；融合共享读写的kernel。 |
| [VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation](#2025-vila-u) | 用统一自回归视觉语言框架兼顾理解和图像/视频生成。 | 联合重建与文本对比训练视觉tower，以残差量化深度表达细节，再接统一下一token预测。 |
| [XAttention: Block Sparse Attention with Antidiagonal Scoring](#2025-xattention) | 以低于完整注意力的成本估计应保留的注意力块，并获得真实prefill加速。 | 反对角采样同时与常见竖线/斜线模式相交，用累计概率阈值选择块，必要时逐头调整阈值。 |

<a id="2025-coat"></a>
### COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/COAT- Compressing Optimizer states and Activation for Memory-Efficient FP8 Training.pdf>) · [归档来源](https://arxiv.org/pdf/2410.19313) · 22页 · 研究论文；版本与归档元数据见清单

**问题：** 降低FP8训练中仍由高精度优化器状态和非线性激活占用的显存。

**痛点：** 仅把线性算子改为FP8不能消除训练显存瓶颈；优化器二阶状态对量化误差敏感。

**朴素方案为什么不够：** 直接FP8表示无法充分利用各组动态范围；跨token分组和激进低精度会增加误差。

**洞察：** 用幂变换扩展优化器状态的动态范围；为非线性激活采用不跨token的细粒度缩放，为GEMM采用较粗粒度并融合转换。

**贡献：** 提出优化器状态与激活的FP8表示方案，并把缩放、转换融合到训练执行中。

**关键证据：**

- PDF p.6：优化器更新量m/sqrt(v)的MSE由直接E4M3的20.10降至同时扩展两状态的12.31。
- PDF p.9：VILA微调平均分BF16为62.80，COAT为62.51，并非所有指标完全无损。
- PDF p.10：Llama2-7B四GPU固定batch=2峰值显存55.1→35.6GB；提高可容纳batch后吞吐7730→11257 token/s。
- PDF p.6：激活存储分析由BF16的22.66U降至COAT的13.33U。

**边界：** 部分精度指标小幅下降；固定batch的内存收益与扩大batch后的吞吐收益是不同实验，不能称完全无损或把二者混为同条件。

**论文策略：** 先分解显存来源与数值误差，再针对不同算子设计表示，最后分别测精度、峰值显存与实际吞吐。

**对我们的启发：** Verified Diagnosis应先证明哪项真实成本支配流程，且评价诊断误差如何改变最终决策；不能只展示代理指标。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,4,5,6,7,9,10。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-dc-ae"></a>
### Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models.pdf>) · [归档来源](https://arxiv.org/pdf/2410.10733v8) · 22页 · 研究论文；版本与归档元数据见清单

**问题：** 让高空间压缩的图像自编码器保持重建质量，以减少扩散模型的token。

**痛点：** 直接扩大压缩率会使重建质量崩坏；只增加潜变量通道或模型容量不能解决训练困难。

**朴素方案为什么不够：** 把退化全部解释为信息瓶颈，会错过深压缩网络自身的优化障碍。

**洞察：** 用无参数space-to-channel路径提供可恢复的参考，再让网络只学习残差；把高分辨率适配与局部细节精修拆为训练阶段。

**贡献：** 残差自编码结构及分阶段训练，支持f32/f64潜空间并验证下游扩散训练。

**关键证据：**

- PDF p.3：f64设置中朴素设计rFID 28.25，残差设计0.50；f32为2.04→0.49。
- PDF p.4：比较同潜变量大小的空间重排参考，定位优化困难；训练阶段分别针对总体结构、高分辨率和细节。
- PDF p.9：大UViT受益于更高压缩，但小UViT-S的下游生成会变差，表明重建质量不等于易建模性。

**边界：** 高压缩收益依赖下游模型容量和训练预算；重建指标不能替代生成质量。

**论文策略：** 用能保留信息的简单对照，区分信息损失和优化失败；结构改动必须沿重建到生成的链条验证。

**对我们的启发：** 先排除任务本身不可识别、干预无效或优化失败，才把结果归因于诊断方法；代理定位分数必须关联真实结果。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,3,4,5,8,9。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-dc-ae-1-5"></a>
### DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/DC-AE 1.5- Accelerating Diffusion Model Convergence with Structured Latent Space.pdf>) · [归档来源](https://arxiv.org/pdf/2508.00413v1) · 13页 · 研究论文；版本与归档元数据见清单

**问题：** 解决高通道潜空间重建更好、扩散模型反而更难收敛的问题。

**痛点：** 结构信息分散在较多通道中，潜空间容量提高但学习难度也提高。

**朴素方案为什么不够：** 仅改善自编码器重建或单独给扩散模型做通道掩码训练，并不能保证生成质量提升。

**洞察：** 用随机前缀通道保留强迫早通道承载结构，再以相匹配的通道增强训练扩散模型。

**贡献：** 结构化潜空间和匹配的扩散训练规则，使高通道、高压缩设置更可学习。

**关键证据：**

- PDF p.7：DiT-XL消融gFID：基线26.44，仅结构化26.75，仅增强36.83，两者合用17.31。
- PDF p.7：USiT-H消融gFID 8.45→6.10；USiT-2B f64c128在512分辨率达到gFID 2.18。
- PDF p.8：较小c32设置并非稳定受益：DiT示例10.18→10.50，USiT 3.89→4.15。

**边界：** 收益集中于原本难学的高通道设置；部分通道解码展示经过对应解码器微调，不能当作完全未经适配的因果分解。

**论文策略：** 发现好表征不等于好生成表征，以二维组件消融证明编码规则与学习规则必须匹配。

**对我们的启发：** 若Verified Diagnosis依赖某种验证器或干预，应展示匹配关系，而不是把各模块单独好看当作组合有效。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,3,4,5,7,8。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-dc-ar"></a>
### DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/DC-AR- Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer.pdf>) · [归档来源](https://arxiv.org/pdf/2507.04947v1) · 18页 · 研究论文；版本与归档元数据见清单

**问题：** 在高压缩token下兼顾自回归图像生成速度、细节和分辨率泛化。

**痛点：** 离散token重建受损，连续token生成需更多昂贵计算，极短一维token又不自然保留二维空间。

**朴素方案为什么不够：** 只提高离散压缩率损失细节；直接全连续生成放大迭代代价。

**洞察：** 深压缩二维混合tokenizer将结构留在离散token、残差细节交给小扩散模块；按连续预热、离散学习、交替解码训练。

**贡献：** DC-HT混合tokenizer及配套掩码自回归生成器，支持从低到高分辨率训练。

**关键证据：**

- PDF p.8：混合相对纯离散gFID 6.71→5.50、GenEval 0.66→0.69；吞吐11.4→10.3张/秒。
- PDF p.8：三阶段tokenizer训练rFID 1.60，替代训练策略1.76或1.92。
- PDF p.8：分阶段256→512训练760 GPUh、gFID5.50，对比从512开始1440 GPUh、6.64。

**边界：** 细节支路有实际开销；整体GenEval提升不意味着所有子项更强，计数仍弱。

**论文策略：** 把大模型应承担的结构计算与可由小模型补充的细节分开，再测拆分后的质量与额外成本。

**对我们的启发：** 诊断只值得计算真正影响后续验证的部分；新增步骤要显示收益超过其开销。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,4,5,7,8。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-dc-gen"></a>
### DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/DC-Gen- Post-Training Diffusion Acceleration with Deeply Compressed Latent Space.pdf>) · [归档来源](https://arxiv.org/pdf/2509.25180v3) · 39页 · 研究论文；版本与归档元数据见清单

**问题：** 不从头训练扩散大模型，把现有模型迁移到深压缩潜空间以加速生成。

**痛点：** 替换自编码器后输入输出接口变化，预训练主干面对错位的特征分布。

**朴素方案为什么不够：** 直接替换接口后端到端微调容易失去已有知识，直接从头训练则成本高。

**洞察：** 先对齐patch嵌入，再冻结主干对齐输出，最后用LoRA联合微调，逐步恢复接口兼容性。

**贡献：** 把潜空间压缩变为预训练模型的后训练改造，覆盖图像、视频与编辑。

**关键证据：**

- PDF p.8：FLUX-Krea 1K的GenEval 0.69→0.71，主干耗时4.06→0.88秒。
- PDF p.27：加入AE解码后相同样例为4.21→1.17秒；文本编码仍不在该拆解中。
- PDF p.4：中间特征差异及直接微调对照支持先接口对齐再整体调整。

**边界：** 当前归档是2026-09-07修订版、39页，虽放在2025目录；依赖预训练知识和额外训练，主表只计主干，不能当完整请求时延。

**论文策略：** 锁定可迁移系统中的接口失配，通过冻结、分阶段对齐与直接微调反例证明恢复机制。

**对我们的启发：** 一个扎实诊断贡献应指出可验证的具体故障与处理顺序，并证明比不诊断直接行动更便宜或更可靠。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,3,4,5,6,8,27。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-dc-videogen"></a>
### DC-VideoGen: Efficient Video Generation with Deep Compression Video Autoencoder

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/DC-VideoGen- Efficient Video Generation with Deep Compression Video Autoencoder.pdf>) · [归档来源](https://arxiv.org/pdf/2509.25182v1) · 19页 · 研究论文；版本与归档元数据见清单

**问题：** 在保持视频重建和可变时长能力的同时压缩视频token，并迁移现有视频扩散器。

**痛点：** 纯因果AE难利用未来冗余，非因果AE又不易外推视频时长；直接换AE会破坏预训练接口。

**朴素方案为什么不够：** 仅增强压缩或直接微调整个视频模型，无法同时解决重建、长视频泛化与知识保留。

**洞察：** chunk内部双向、chunk之间因果，随后用输入输出接口对齐及LoRA迁移生成器。

**贡献：** 深压缩视频AE及适配流程，将重建约束和预训练模型兼容性分别处理。

**关键证据：**

- PDF p.4：同压缩率192的例子，因果AE PSNR31.12、chunk-causal32.72。
- PDF p.9：720p Wan1.3B主干5.76→0.70分钟，整体VBench83.38→84.63。
- PDF p.8：LoRA微调350.37M参数得VBench84.48，完整微调1418.90M参数79.81。

**边界：** 速度主表以扩散主干为界；依赖底座能力，长视频外推和适配泛化仍有限。

**论文策略：** 先用匹配压缩率的结构对照分离因果性代价，再用冻结与全量微调比较验证知识保留。

**对我们的启发：** 诊断结果必须能排除替代解释；不应把参数更多或干预更强自然视作更可靠。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,3,4,6,8,9,18。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-duo-attention"></a>
### DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/DuoAttention- Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.pdf>) · [归档来源](https://arxiv.org/pdf/2410.10819) · 20页 · 研究论文；版本与归档元数据见清单

**问题：** 降低长上下文LLM的KV缓存与注意力成本，保留真正需要全局检索的头。

**痛点：** 所有头保留全历史浪费显存，但直接按注意力可视化或统一窗口截断会损害检索。

**朴素方案为什么不够：** 观察亮区或只在语言建模损失上校准，不可靠地区分可压缩与必须保留的注意力头。

**洞察：** 用限制为sink+近期token后对输出的反事实影响定义retrieval head；合成多passkey蒸馏学习混合门，再二值化部署。

**贡献：** 可操作的头分类及双缓存执行，让诊断直接决定保留哪些长期信息。

**关键证据：**

- PDF p.2：将retrieval头先改为streaming会破坏检索，反序操作则影响小。
- PDF p.10：多passkey蒸馏头选择胜过attention profiling及语言建模校准。
- PDF p.9：MHA缓存内存最多减少2.55倍、decode提速2.18倍；GQA对应1.67和1.50倍。

**边界：** 需要每模型离线校准，压缩比例有质量取舍；3.3M上下文演示叠加8bit权重和4bit KV，不能全部归功于头分类。

**论文策略：** 把行为定义变成可测试的干预标准，再让识别结果控制系统动作，最终验证保留能力和资源收益。

**对我们的启发：** 这是最直接的借鉴：诊断不是热图，而是能决定什么可以省略，并经真实任务验证的行动依据。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,2,3,4,5,7,9,10,11。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-hart"></a>
### HART: Efficient Visual Generation with Hybrid Autoregressive Transformer

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/HART- Efficient Visual Generation with Hybrid Autoregressive Transformer.pdf>) · [归档来源](https://arxiv.org/pdf/2410.10812v1) · 20页 · 研究论文；版本与归档元数据见清单

**问题：** 兼顾自回归视觉生成的快速全局建模和高频细节。

**痛点：** 纯离散token量化损伤细节；全连续AR需要重复执行昂贵的生成计算。

**朴素方案为什么不够：** 独立训练两个质量不错的解码器，不保证离散和连续表示在生成过程兼容。

**洞察：** 共享解码器交替接收离散/连续latent，使主AR负责粗结构、小残差扩散器负责末端细节。

**贡献：** 混合tokenizer、粗到细AR及小型残差扩散器，并识别共享解码几何的重要性。

**关键证据：**

- PDF p.9：ImageNet HART-d30 FID1.77，VAR1.92；对应批处理时延2.7对2.6秒，MAR-L为34.9秒。
- PDF p.10：相近重建指标的独立解码器生成FID21.98，共享解码器5.30，均为24k步设置。
- PDF p.9：文本1024强离散版本FID5.73，混合5.38，收益应与匹配基线比较。

**边界：** 生成收益依赖数据和任务；小扩散支路也有开销，不能把全部强基线差距归于该支路。

**论文策略：** 用中间指标相似但最终结果差异巨大的对照，证明接口兼容性比单模块质量更关键。

**对我们的启发：** 诊断排名或验证分数相似，不代表最终发现正确机制的能力相同；应比较完整工作流。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,3,5,6,9,10。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-jet-nemotron"></a>
### Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/Jet-Nemotron- Efficient Language Model with Post Neural Architecture Search.pdf>) · [归档来源](https://arxiv.org/pdf/2508.15884) · 20页 · 研究论文；版本与归档元数据见清单

**问题：** 以较低架构探索代价获得真正高吞吐且保持任务能力的线性/混合注意力模型。

**痛点：** 从头训练不同架构昂贵；参数量和FLOPs不足以预测硬件吞吐。

**朴素方案为什么不够：** 均匀放置全注意力层或选理论轻量模块，可能同时损害检索、知识能力和实际速度。

**洞察：** 复用预训练MLP，以可训练门和搜索决定全注意力层位置；再选动态卷积线性模块与硬件合适的头形状。

**贡献：** PostNAS方法、JetBlock及Jet-Nemotron模型，将知识继承、任务敏感层位和实际吞吐联合评估。

**关键证据：**

- PDF p.5：不同层对MMLU和检索的价值不同；搜索后的全注意力位置优于均匀布局。
- PDF p.3：组件路线图中MMLU52.8→58.1，数学26.5→34.9，检索62.6→70.4。
- PDF p.8：训练包含50B token冻结MLP蒸馏，以及350B token完整训练。
- PDF p.11：256k上下文设置中，相对Qwen3-1.7B的decode最大提速53.6倍，非通用请求时延。

**边界：** 最终模型仍需大量训练；最大速度数字依赖极长上下文，不能推广为任意请求的加速。

**论文策略：** 让诊断层位的重要性服务于架构决策，用任务差异和真实硬件代价反驳统一简化规则。

**对我们的启发：** Verified Diagnosis若仅给统一排名可能错失任务依赖；需要以具体目标定义重要性并比较决策收益。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,2,3,4,5,6,8,10,11。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-lego"></a>
### LEGO: Spatial Accelerator Generation and Optimization for Tensor Applications

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/LEGO- Spatial Accelerator Generation and Optimization for Tensor Applications.pdf>) · [归档来源](https://arxiv.org/pdf/2509.12053) · 13页 · 研究论文；版本与归档元数据见清单

**问题：** 从张量计算自动生成可综合的空间加速器，并优化数据流和硬件映射。

**痛点：** 固定模板限制架构空间，灵活的解析映射又常停留在代价模型，无法直接变为RTL。

**朴素方案为什么不够：** 仅搜索理论复用和吞吐不能保证互连、存储、控制及数据流合并后的硬件有效。

**洞察：** 用仿射迭代和时空映射表达数据流，推导连接与存储，再在生成图上优化引脚共享、广播、归约和门控。

**贡献：** 贯通映射到RTL的生成器及图级优化，用综合、仿真和工作负载测量验证。

**关键证据：**

- PDF p.9：对Gemmini的端到端评估平均加速3.2倍、能耗改善2.4倍。
- PDF p.11：组件优化贡献35%面积降低和28%功耗降低。
- PDF p.10：Llama7B batch1利用率仅3.1%，batch32为42.9%，仍受DRAM带宽限制。

**边界：** 证据以RTL仿真和综合为主，不等于量产芯片；后处理及具体硬件配置也影响可迁移性。

**论文策略：** 把抽象映射变成可执行工件，逐级测量原先被代价模型忽略的物理开销，并承认剩余瓶颈。

**对我们的启发：** 诊断系统必须交付可重复的验证过程，而不只在离线指标上获胜；应报告真实瓶颈及失败实例。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,2,4,5,7,8,9,10,11。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-long-rl"></a>
### Long-RL: Scaling RL to Long Videos

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/Long-RL- Scaling RL to Long Videos.pdf>) · [归档来源](https://arxiv.org/pdf/2507.07966) · 29页 · 研究论文；版本与归档元数据见清单

**问题：** 让VLM有效训练和推理长视频，而非仅增加输入帧数。

**痛点：** 长视频推理数据不足，RL多rollout重复编码大量相同视觉内容，显存和运行成本高。

**朴素方案为什么不够：** 直接RL冷启动会劣化能力；只增加帧数而不训练推理，性能趋于饱和。

**洞察：** 构建长视频推理QA，先CoT-SFT再GRPO；系统重复使用视觉序列并按模态均衡序列并行。

**贡献：** LongVideo-Reason 104K数据、LongVILA-R1和MR-SP训练系统，并分别消融数据训练和计算复用。

**关键证据：**

- PDF p.9：CoT-SFT/RL消融：基线58.1，仅SFT60.2，仅RL52.4，联合61.9。
- PDF p.9：512帧无推理60.2、有推理64.3；VideoMME主结果无/有字幕65.1/71.1。
- PDF p.9：完整MR-SP在512帧最多2.1倍训练提速，1024帧仍能运行，只有第一阶段会OOM。

**边界：** 自动构建的推理链不等于真实内部推理或因果解释；更长序列和大batch仍需更多硬件。

**论文策略：** 分别证明能力瓶颈与训练成本瓶颈，利用冷启动失败和逐组件系统消融解释为何训练顺序及复用必要。

**对我们的启发：** 诊断研究应把“有更多观测”和“能使用观测做正确决策”分开；推理文本不能代替独立机制真值。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,2,4,5,6,7,8,9。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-longvila"></a>
### LongVILA: Scaling Long-Context Visual Language Models for Long Videos

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/LongVILA- Scaling Long-Context Visual Language Models for Long Videos.pdf>) · [归档来源](https://arxiv.org/pdf/2408.10188) · 20页 · 研究论文；版本与归档元数据见清单

**问题：** 让VLM在可承受训练成本下利用小时级视频和百万级多模态上下文。

**痛点：** 视觉编码和语言主干的负载分布不同，沿用纯文本序列并行造成不均衡和跨机通信浪费。

**朴素方案为什么不够：** 只扩展文本上下文或简单重叠通信不保证多模态有效；重叠可能抢占计算资源而变慢。

**洞察：** 先按图像负载分片再重分配token，按机内/机间拓扑选择通信；训练分阶段补齐对齐、短上下文、扩展和长视频能力。

**贡献：** MM-SP系统与五阶段训练，将长输入可训练性和长视频任务表现联结。

**关键证据：**

- PDF p.3：五阶段完整路径消融得57.5，替代顺序或缺项约55.3–56.0。
- PDF p.7：通信重叠例子forward时间29.5→35，反而增加18.6%。
- PDF p.10：VideoMME主结果，无字幕/有字幕分别为60.1/65.1。
- PDF p.2：长上下文needle检索得99.8%，不代表所有时序推理任务的准确率。

**边界：** needle检索不能代表一般时序推理；合成数据质量和阶段间训练差异限制因果归因。

**论文策略：** 先测不同计算阶段和拓扑的真实瓶颈，再证明系统扩展确实带来任务上的有效上下文。

**对我们的启发：** 我们要将诊断可运行、诊断能定位、定位能帮助决策三个门槛分开，而不是用一个演示代表全部。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,2,3,4,5,6,7,10。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-lserve"></a>
### LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/LServe- Efficient Long-sequence LLM Serving with Unified Sparse Attention.pdf>) · [归档来源](https://arxiv.org/pdf/2502.14866) · 16页 · 研究论文；版本与归档元数据见清单

**问题：** 同时减少LLM长输入prefill和长输出decode的注意力成本。

**痛点：** 静态稀疏不能覆盖全部冗余，动态KV选择又可能使选择成本接近节省的注意力计算。

**朴素方案为什么不够：** 每次精细选择所有KV页，或用同一种稀疏粒度对应算法和硬件，可能失去实际加速。

**洞察：** 静态流式头与动态检索页互补；分离逻辑/物理页粒度，并跨连续query复用页选择。

**贡献：** 统一块稀疏运行框架、分层分页及可复用选择器，覆盖prefill和decode。

**关键证据：**

- PDF p.10：页选择成本随上下文增长；每四个query复用选择用于摊销该成本。
- PDF p.9：Llama3-8B和Minitron4B相对vLLM平均decode提速约1.5倍，Llama2-7B超过2倍。
- PDF p.9：对Quest的32k例子prefill 3.61→2.32秒、decode14.86→10.24毫秒。

**边界：** 静态头校准和动态页预算仍有适用条件；prefill、decode与整个请求耗时不可混为一种速度。

**论文策略：** 把“决定省略什么”的开销本身纳入优化，组件消融同时测准确度、选择时间和执行时间。

**对我们的启发：** 这是Verified Diagnosis的核心成本教训：候选诊断花掉的预算必须计入总账，不能只数后续验证节省。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,3,4,5,9,10,11。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-qserve"></a>
### QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/QServe- W4A8KV4 Quantization and System Co-design for Efficient LLM Serving.pdf>) · [归档来源](https://arxiv.org/pdf/2405.04532) · 19页 · 研究论文；版本与归档元数据见清单

**问题：** 让低比特LLM在云端批量服务中获得真实吞吐收益。

**痛点：** INT4的理论吞吐被反量化、CUDA标量处理及KV搬运开销抵消。

**朴素方案为什么不够：** 把权重、激活一律降到4bit，或朴素压缩KV，可能比更高精度方案更慢。

**洞察：** 保留高效INT8计算路径，采用W4A8KV4、渐进组量化、SmoothAttention及针对反量化/寄存器访问的执行设计。

**贡献：** QoQ量化方法与QServe服务引擎，从数值失真到硬件开销闭环验证。

**关键证据：**

- PDF p.10：朴素KV4在A100例子0.42→0.48毫秒反而变慢，优化后0.28毫秒。
- PDF p.13：L40S主表相对最佳TensorRT精度吞吐提升1.39–3.47倍，A100为1.17–2.38倍。
- PDF p.13：Llama3.1-8B LongBench平均BF16 38.52、QoQ 38.38，仍有轻微质量代价。

**边界：** 硬件、batch及精度配置影响加速；有小幅精度损失，不能称普遍无损。

**论文策略：** 先用硬件测量推翻“位数更低必然更快”，再逐项减少决定性能的真实转换开销。

**对我们的启发：** Verified Diagnosis应允许朴素直接验证击败复杂方法，以实际总成本和质量定义价值，而不是理论查询数。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,3,4,5,6,10,11,12,13。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-radial-attention"></a>
### Radial Attention: O(nlogn) Sparse Attention with Energy Decay for Long Video Generation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/Radial Attention- O(nlogn) Sparse Attention with Energy Decay for Long Video Generation.pdf>) · [归档来源](https://arxiv.org/pdf/2506.19852v2) · 25页 · 研究论文；版本与归档元数据见清单

**问题：** 降低长视频生成注意力的计算增长，同时保留时空结构。

**痛点：** 视频token数增长使全注意力不可承受；任意窗口或动态稀疏可能损害全局时序信息并增加选择开销。

**朴素方案为什么不够：** 均匀分配所有时距的注意力预算，忽略远距贡献通常较小但不应全部删除。

**洞察：** 利用平均能量随时空距离衰减，按时距以层级方式缩减空间带宽，保留首帧sink并配合块稀疏kernel。

**贡献：** Radial静态稀疏模式、复杂度分析及训练/推理实现，用长视频LoRA验证可扩展性。

**关键证据：**

- PDF p.7：117帧Hunyuan例子dense1649秒、Radial876秒；PSNR27.3、VisionReward0.141对dense0.139。
- PDF p.10：延长509帧设置LoRA训练171对746 GPUh，推理781对2895秒。
- PDF p.10：时距能量衰减曲线拟合R²高于0.985，是汇总经验支持而非逐样本保证。

**边界：** O(n log n)以固定空间分辨率下增长帧数为主要分析条件；某些稀疏基线更快，长视频版本需要训练，平均衰减不等于所有关键事件安全。

**论文策略：** 从经验分布中提取可计算结构先验，用复杂度、实测速度和输出保真共同界定可用区域。

**对我们的启发：** 诊断可以利用稳定结构减少查询，但必须保留罕见远距重要机制的反例与拒绝结论，不能把平均规律当个体保证。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,4,5,7,9,10。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-sana"></a>
### SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/SANA- Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers.pdf>) · [归档来源](https://arxiv.org/pdf/2410.10629v3) · 23页 · 研究论文；版本与归档元数据见清单

**问题：** 在较低资源下完成高分辨率文本到图像生成。

**痛点：** 高分辨率token数量、二次注意力、文本编码和迭代步数共同影响延迟，单改注意力收益有限。

**朴素方案为什么不够：** 直接以线性注意力替代全注意力会损质量；MixFFN恢复质量又引入运行开销。

**洞察：** 深压缩AE先减少token，线性注意力负责全局、局部MixFFN补空间细节，再用kernel融合、较小文本编码器和更少步数解决残余成本。

**贡献：** SANA生成模型及逐级组件设计，将表征压缩、局部建模和执行开销对应到具体瓶颈。

**关键证据：**

- PDF p.9：仅替换线性注意力延迟2250→1931ms但质量下降；加入MixFFN为2425ms，融合及f32潜空间最终748ms。
- PDF p.9：1024 A100：SANA0.6B延迟0.9秒、GenEval0.64；FLUX-dev23秒、0.67。吞吐39.5倍是batch16指标。
- PDF p.10：T5-XXL文本编码1.61秒、Gemma2B为0.28秒；组件延迟需单独测量。
- PDF p.6：Gemma文本特征较大方差会导致训练NaN，RMSNorm用于稳定训练。

**边界：** 不同指标有取舍；吞吐与单张延迟基准不同，不能互换倍数；总体收益不是线性注意力单独创造。

**论文策略：** 先做端到端瓶颈分解，再让每项技术解决上一项引出的明确问题，用组件消融避免归因混乱。

**对我们的启发：** Verified Diagnosis需要像这样说明哪项成本或失效主导、为何朴素方案不行及额外机制究竟补什么。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,2,3,4,5,6,9,10。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-sana-1-5"></a>
### SANA-1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/SANA-1.5- Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer.pdf>) · [归档来源](https://arxiv.org/pdf/2501.18427v4) · 21页 · 研究论文；版本与归档元数据见清单

**问题：** 在有限训练成本和可增加的推理预算下扩展SANA能力。

**痛点：** 从头训练更大模型昂贵；更多采样步数不能可靠修复结构错误，盲目剪枝损害能力。

**朴素方案为什么不够：** 只扩大参数或步数，忽略已有知识继承、微调恢复和候选选择对结果的作用。

**洞察：** 复制并近似恒等初始化新层以增长模型，结合稳定训练和剪枝后微调；推理阶段生成候选并用VLM选择。

**贡献：** 训练时模型增长/恢复路线和推理时best-of-N选择，并分别测量预算—质量曲线。

**关键证据：**

- PDF p.3：达到GenEval0.70所需训练步数减少约60%。
- PDF p.6：SFT使GenEval0.72→0.81，但FID5.42→5.99；GenEval0.96依赖2048候选。
- PDF p.7：4.8B剪至1.6B得0.693→0.571，微调恢复至0.672。

**边界：** 高分需要候选生成和VLM判别预算；多个指标存在取舍，剪枝恢复也非免费。

**论文策略：** 把训练知识继承和推理搜索分别画为成本—性能曲线，公开恢复成本与选择预算。

**对我们的启发：** 诊断应比较相同预算下直接重复验证、采样或best-of-N，而不是仅与单次廉价基线比较。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,2,3,4,5,6,7。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-sana-sprint"></a>
### SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/SANA-Sprint- One-Step Diffusion with Continuous-Time Consistency Distillation.pdf>) · [归档来源](https://arxiv.org/pdf/2503.09641v4) · 22页 · 研究论文；版本与归档元数据见清单

**问题：** 将高效流匹配图像模型压缩到一至数步生成，同时保持结构和细节。

**痛点：** 直接迁移连续一致性训练存在坐标、时间导数和数值稳定性不匹配；一致性与对抗目标各有偏差。

**朴素方案为什么不够：** 只做一致性蒸馏可能不足以保持全局质量，只做对抗蒸馏也会损失表现。

**洞察：** 转换为TrigFlow坐标并稳定时间与注意力参数化，用连续一致性约束配合latent对抗目标。

**贡献：** 可用于SANA的连续时间蒸馏流程、稳定化设计和一至四步模型。

**关键证据：**

- PDF p.8：sCM单独FID8.93，LADD单独12.20，两者结合8.11；移除坐标/调度处理会发散。
- PDF p.8：最大时间点权重从0%到70%，FID9.44→8.11。
- PDF p.7：4步1.6B：延迟0.31秒、FID6.54、GenEval0.77；原SANA为1.2秒、5.76、0.66。

**边界：** 无需训练的坐标变换不等于无需训练的蒸馏；64倍等宣传数字是transformer部分，解码器形成端到端下限。

**论文策略：** 先排除参数化不匹配导致的训练假失败，再用互补损失和消融解释稳定性与质量改进。

**对我们的启发：** 在给诊断方法定罪之前，需要确认评分坐标、干预定义和验证目标匹配；之后再测完整流程。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,2,3,4,5,7,8。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-svdquant"></a>
### SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/SVDQuant- Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models.pdf>) · [归档来源](https://arxiv.org/pdf/2411.05007v4) · 28页 · 研究论文；版本与归档元数据见清单

**问题：** 让扩散模型权重和激活同时4bit量化而不发生严重质量崩坏，并兑现硬件加速。

**痛点：** 离群值使低比特误差大，平滑只把激活离群值迁移到权重；小低秩分支的额外读写又会吃掉加速。

**朴素方案为什么不够：** 直接量化、仅平滑或量化后低秩补偿均不足；量化误差的奇异值分布并不适合低秩修复。

**洞察：** 先把激活困难迁移到权重，再由高精度低秩分支吸收权重主奇异方向，低比特只处理残差；融合共享读写的kernel。

**贡献：** SVDQuant量化范式及Nunchaku执行器，机制消融覆盖误差与系统开销。

**关键证据：**

- PDF p.6：朴素rank32分支带来57%时延开销，融合kernel后消除大部分额外读写。
- PDF p.10：PixArt消融ImageReward：LoRC0.508、无平滑0.690、完整0.878；直接量化和只平滑明显失败。
- PDF p.8：FLUX主干体积22.2→6.1GiB，其中低秩分支0.3GiB；部分质量指标仍非完全一致。

**边界：** 低秩不能普遍修复任意量化误差；理论上界有分布假设，实际质量与速度依赖模型、位格式及硬件。

**论文策略：** 先指出已有技巧只是转移困难，再把困难转移到适合承载它的结构；系统实测揭示并修复理论省算却实际慢。

**对我们的启发：** 我们必须发现具体失效机制及可验证解法；若诊断只是给既有验证排队，应证明它解决了哪种系统性难例。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,4,5,6,8,10。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-vila-u"></a>
### VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/VILA-U- a Unified Foundation Model Integrating Visual Understanding and Generation.pdf>) · [归档来源](https://arxiv.org/pdf/2409.04429v3) · 19页 · 研究论文；版本与归档元数据见清单

**问题：** 用统一自回归视觉语言框架兼顾理解和图像/视频生成。

**痛点：** 视觉理解需要语义对齐，视觉生成需要保留可重建细节，两种表征要求存在冲突。

**朴素方案为什么不够：** 仅用重建训练的离散视觉token理解能力不足；加入对比损失也不自动产生两任务互相增强。

**洞察：** 联合重建与文本对比训练视觉tower，以残差量化深度表达细节，再接统一下一token预测。

**贡献：** 统一视觉tower与多模态自回归框架，用理解和生成两侧实验显式展示权衡。

**关键证据：**

- PDF p.9：同25M数据，重建+对比相对仅重建VQAv2 57.7→68.0，POPE75.1→83.7。
- PDF p.9：小模型生成对照中统一tower rFID1.80、FID13.2，RQ-VAE为1.30和12.0。
- PDF p.10：作者明确说明尚未观察到理解和生成任务显著协同或互相增强。

**边界：** 统一并非无代价：语义对齐影响重建，未证实任务协同；大规模训练数据也影响跨模型比较。

**论文策略：** 把目标冲突本身作为设计依据，同时给出正收益和负代价，克制“统一即协同”的叙事。

**对我们的启发：** Verified Diagnosis不能因为诊断与验证被统一，就宣称互相增强；需要匹配基线和两侧单独/联合对照。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,3,4,6,8,9,10。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

<a id="2025-xattention"></a>
### XAttention: Block Sparse Attention with Antidiagonal Scoring

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2025/XAttention- Block Sparse Attention with Antidiagonal Scoring.pdf>) · [归档来源](https://raw.githubusercontent.com/mlresearch/v267/main/assets/xu25ag/xu25ag.pdf) · 13页 · 研究论文；版本与归档元数据见清单

**问题：** 以低于完整注意力的成本估计应保留的注意力块，并获得真实prefill加速。

**痛点：** 先算完整注意力再选稀疏块本末倒置；简单抽样又易漏掉竖线和斜线结构。

**朴素方案为什么不够：** 随机或普通对角采样在相同采样预算下不能可靠捕捉关键注意力模式。

**洞察：** 反对角采样同时与常见竖线/斜线模式相交，用累计概率阈值选择块，必要时逐头调整阈值。

**贡献：** 结构感知低成本评分器和块稀疏执行，测试长文本、视频理解和视频生成。

**关键证据：**

- PDF p.7：相同stride8预算：随机RULER82.48、普通对角81.06、反对角88.47；后者块密度20.97%。
- PDF p.7：256k上下文注意力kernel最高13.5倍加速，不能直接等同完整请求加速。
- PDF p.5：视频生成前五步用dense warmup可避免布局偏移；说明稀疏策略有阶段依赖。

**边界：** 依赖注意力模式几何与阈值，生成需dense warmup；估计概率质量与最终事实正确不是同一指标。

**论文策略：** 用模式几何提出廉价代理，在匹配采样成本下排除随机等简单方案，并报告需要回退密集计算的条件。

**对我们的启发：** 诊断代理要以相同信息和预算对比随机、简单分数与直接验证，且应允许某些阶段必须付全量成本。

**本次阅读范围：** 核对当前归档PDF的摘要/引言、方法、主实验和消融相关段；重点页为1,2,3,5,6,7。阅读关键段及表图文字，不代表逐字阅读全部补充材料或独立复核全部证明。

## 2026：10份

| 论文 | 瞄准的问题 | 核心洞察 |
|---|---|---|
| [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](#2026-fast-dllm) | 扩散语言模型理论上能并行，实际却比自回归模型慢。 | 块内相邻去噪步的前后缀状态变化较小，可近似复用；高置信位置可更安全地并行生成。 |
| [Fast-dLLM v2: Efficient Block-Diffusion Large Language Model](#2026-fast-dllm-v2) | 如何让块扩散语言模型以合理训练量获得真正可用的推理效率。 | 保留跨块因果结构与块内双向并行，并让转换训练尽量贴近原 AR 表征。 |
| [ForeAct: Steering Your VLA with Efficient Visual Foresight Planning](#2026-foreact) | 高层语言指令如何可靠地变成多步机器人动作。 | 将未来目标状态直接展示给 VLA，可把抽象语义转成具体视觉运动目标。 |
| [LongLive: Real-time Interactive Long Video Generation](#2026-longlive) | 实时视频生成中途换 prompt 时，怎样既跟随新指令又保持画面连贯。 | 视觉历史和旧文本语义需要区别处理；缓存应由已有帧和新 prompt 重建。 |
| [Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](#2026-lpd) | 怎样在保持平坦视觉 token 表示的同时，提高自回归图像生成并行度。 | 待生成位置应接近已有上下文、远离同轮其他位置；位置查询与上下文 token 应解耦。 |
| [QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs](#2026-qerl-beyond-efficiency----quantization-enhanced-reinforcement-learning-for-llms) | 低比特 RL 能否同时降低 rollout 成本并改善学习。 | 量化扰动在所测 RL 条件下提高采样熵，并可能促进探索；可动态调制噪声强度。 |
| [SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer](#2026-sana-video) | 如何以小模型和可控显存生成高分辨率长视频。 | 线性注意力的累加状态可固定大小保留全局摘要，但需要配套时序建模、稳定归一化与长程训练。 |
| [StreamingVLM: Real-Time Understanding for Infinite Video Streams](#2026-streamingvlm) | 让 VLM 持续理解长时间视频并实时解说。 | 保留 sink、较长文本窗口和较短视觉窗口，并让训练分块模拟这种时间结构。 |
| [Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter](#2026-tlt) | 如何加速被少数超长回答拖住的推理 RL 训练。 | 长尾既是瓶颈也提供空闲算力；利用空闲时段更新草稿，并按实时 batch 选择投机策略。 |
| [VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference](#2026-vlash) | 让 VLA 在不断运动的环境中低延迟且稳定地执行动作。 | 机器人在推理期间会执行哪些动作已知，可据此估计执行时的本体状态，并训练模型利用它。 |

<a id="2026-fast-dllm"></a>
### Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/Fast-dLLM- Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding.pdf>) · [归档来源](https://arxiv.org/pdf/2505.22618) · 22页 · 研究论文；版本与归档元数据见清单

**问题：** 扩散语言模型理论上能并行，实际却比自回归模型慢。

**痛点：** 双向注意力使传统 KV cache 难复用；同时采样有相互依赖的 token 会损害推理正确性。

**朴素方案为什么不够：** 直接增加并行 token 数会降低准确率；仅减少总步数不能同时解决每步重算和 token 依赖。

**洞察：** 块内相邻去噪步的前后缀状态变化较小，可近似复用；高置信位置可更安全地并行生成。

**贡献：** Prefix/DualCache 降低单步开销，置信阈值并行减少步数，两者分开验证后组合。

**关键证据：**

- PDF p.5：前后缀 KV 相似度分析为缓存设计提供依据。
- PDF p.7：LLaDA GSM8K、256 长度：准确率 79.3→78.5，吞吐 6.7→54.4 token/s，8.1 倍。
- PDF p.8：固定并行数与置信阈值对照，并比较 Dream 架构。
- PDF p.9：最高 27.6 倍依赖 8-shot、1024 长度设置；MathVista 59.2→56.6 显示非无损。

**边界：** 近似缓存不保证严格等价；倍率依赖长度、前缀和硬件，不能将最高倍率写成普遍收益。

**论文策略：** 发现可复用结构→转化为实际调度→质量—速度曲线闭环。

**对我们的启发：** 状态相似图只负责提出假说，真正贡献由改变计算决策后的准确率和总成本证明。

**本次阅读范围：** PDF 1–9 的引言、方法、主结果、关键消融；未逐项审计所有附录。

<a id="2026-fast-dllm-v2"></a>
### Fast-dLLM v2: Efficient Block-Diffusion Large Language Model

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/Fast-dLLM v2- Efficient Block-Diffusion Large Language Model.pdf>) · [归档来源](https://arxiv.org/pdf/2509.26328) · 17页 · 研究论文；版本与归档元数据见清单

**问题：** 如何让块扩散语言模型以合理训练量获得真正可用的推理效率。

**痛点：** 全双向扩散与精确 KV cache 不兼容；将 AR 模型转成扩散模型可能耗费大量训练 token。

**朴素方案为什么不够：** v1 的推理近似有边界；直接平移 AR 输出、随意改变推理块长会产生训练失配。

**洞察：** 保留跨块因果结构与块内双向并行，并让转换训练尽量贴近原 AR 表征。

**贡献：** 块扩散模型、互补掩码、块对齐 padding 和层次缓存协同设计，兼顾转换成本与部署。

**关键证据：**

- PDF p.7：1.5B 平均分 45.0，对照同数据微调 AR 为 44.3；7B 为 60.3 对 59.6。
- PDF p.8：朴素转换平均 41.3，加 padding 42.2，再加互补掩码到 45.0。
- PDF p.9：直接小块推理与层次子块缓存对照区分训练失配和缓存收益。

**边界：** 不是各任务全面超过 AR；~1B 与其他工作数百 B token 是跨工作训练量比较，不能当严格同预算实验。

**论文策略：** 承认上一代方法局限，再重新定义需训练解决的瓶颈。

**对我们的启发：** pivot 应回答旧方案为什么结构性受限，以及新设计如何消除该限制，不能只改评价名称。

**本次阅读范围：** PDF 1–2、4–9，含方法、训练协议、对照和缓存消融。

<a id="2026-foreact"></a>
### ForeAct: Steering Your VLA with Efficient Visual Foresight Planning

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/ForeAct- Steering Your VLA with Efficient Visual Foresight Planning.pdf>) · [归档来源](https://arxiv.org/pdf/2602.12322v1) · 16页 · 研究论文；版本与归档元数据见清单

**问题：** 高层语言指令如何可靠地变成多步机器人动作。

**痛点：** VLA 同时承担语义定位与运动控制，小数据下难以泛化；仅用 VLM 分解语言任务仍不足。

**朴素方案为什么不够：** 普通图像编辑器可能凭空添物、缺少物理动作；仅给空间文字提示也无法达到视觉目标效果。

**洞察：** 将未来目标状态直接展示给 VLA，可把抽象语义转成具体视觉运动目标。

**贡献：** 机器人数据预训练的高效未来图像生成器、VLM 规划监控及现有 VLA 的闭环组合。

**关键证据：**

- PDF p.6：生成器预训练对照分别检查图像忠实度与质量；640×480、8 步预测需 0.33 秒/H100。
- PDF p.7：11 个实机任务平均原文 success rate：π0 46.5、VLM+π0 57.1、ForeAct 87.4。
- PDF p.9：Pick_Tool：语义文本 20.0，空间文本 46.8，目标图像加语义文本 93.4。

**边界：** 实机主指标由原子动作得分汇总，不等于整段任务一次成功概率；大规模预训练是投入，通用图像编辑器比较主要为定性例子。

**论文策略：** 跨模块信息传递从模糊语言改为具体目标；用更强的文字对照排除仅任务分解的解释。

**对我们的启发：** 最接近“可视化是否有用”的范本：保持下游任务明确，以相同消费者的实际动作成功证明附加视觉信息价值。

**本次阅读范围：** PDF 1、3–9；主方法、数据、实机与模拟实验、OOD、文字提示对照。

<a id="2026-longlive"></a>
### LongLive: Real-time Interactive Long Video Generation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/LongLive- Real-time Interactive Long Video Generation.pdf>) · [归档来源](https://arxiv.org/pdf/2509.22622v2) · 21页 · 研究论文；版本与归档元数据见清单

**问题：** 实时视频生成中途换 prompt 时，怎样既跟随新指令又保持画面连贯。

**痛点：** 旧 KV cache 含旧指令语义；清空会断裂，保留会不听新指令，长 rollout 又会累积误差。

**朴素方案为什么不够：** 直接扩大短视频训练造成显存及教师监督问题；只减注意力窗口会损伤一致性。

**洞察：** 视觉历史和旧文本语义需要区别处理；缓存应由已有帧和新 prompt 重建。

**贡献：** KV recache、流式长程训练、短窗口加帧 sink，形成可交互的长视频生成系统。

**关键证据：**

- PDF p.3：明确提出清空/保留缓存的两种相反失败，并解释旧 prompt 如何进入缓存。
- PDF p.8：缓存消融：保留缓存 CLIP 25.92、重建 27.87；清空 28.95 但主体一致性 89.59 低于重建 94.04。
- PDF p.7：同表比较质量和速度，单 H100 20.7 FPS；60 秒交互按每段新 prompt 评测。
- PDF p.9：窗口与 sink 消融衡量一致性、显存和运行时间的联合权衡。

**边界：** 重建不是每项指标最高；短视频语义分低于部分基线。240 秒为展示能力，主要系统评测时长另有界定。

**论文策略：** 将两难冲突拆开，识别应该保留和应该更新的状态。

**对我们的启发：** 诊断研究可学习这种可区分失败机制的对照：相同“输出改变”可能来自破坏或有效恢复，必须分开。

**本次阅读范围：** PDF 1–9，主文方法、比较、交互评估与关键消融。

<a id="2026-lpd"></a>
### Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation.pdf>) · [归档来源](https://arxiv.org/pdf/2507.01957v2) · 24页 · 研究论文；版本与归档元数据见清单

**问题：** 怎样在保持平坦视觉 token 表示的同时，提高自回归图像生成并行度。

**痛点：** 逐 token 生成受内存带宽限制；多尺度方案改变表示，朴素邻近 token 并行又破坏依赖。

**朴素方案为什么不够：** 光用随机或 Halton 顺序不足；传统 AR 把提供上下文与预测下一位置的职责绑在一起。

**洞察：** 待生成位置应接近已有上下文、远离同轮其他位置；位置查询与上下文 token 应解耦。

**贡献：** 可并行互见的位置查询架构，加兼顾两个条件的局部性顺序，仍可复用 KV cache。

**关键证据：**

- PDF p.5：50,000 张生成图的注意力统计提出局部性假说，而非最终因果结论。
- PDF p.8：ImageNet256 XL：20 步 FID 2.10、0.41 秒；同表 ARPG-XL 64 步 FID 2.10、1.71 秒。
- PDF p.10：同模型顺序消融：随机 FID 2.11，单原则 2.00/2.06，两原则 1.92。
- PDF p.9：扩展到1024文生图：XL GenEval 0.60→0.62、4096→64步。

**边界：** 收益受 batch 和内存/计算瓶颈影响；局部注意力本身不是完整依赖证明，架构和顺序实验才是关键。

**论文策略：** 观察提出双原则，分别消融证明单原则解释不足。

**对我们的启发：** 是我们最直接的归因类参照：不需证明热图等于真原因，但要证明由热图启发的规则有额外决策价值。

**本次阅读范围：** PDF 1–10 主文，含排序算法、系统比较及双原则消融。

<a id="2026-qerl-beyond-efficiency----quantization-enhanced-reinforcement-learning-for-llms"></a>
### QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/QeRL- Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs.pdf>) · [归档来源](https://arxiv.org/pdf/2510.11696) · 21页 · 研究论文；版本与归档元数据见清单

**问题：** 低比特 RL 能否同时降低 rollout 成本并改善学习。

**痛点：** RL 长序列采样耗时耗显存；普通 QLoRA 小模型存储不等于真实训练加速。

**朴素方案为什么不够：** 沿用 SFT 中“量化误差只会损害训练”的认识会漏掉 RL 探索价值；静态噪声也无法适应探索阶段变化。

**洞察：** 量化扰动在所测 RL 条件下提高采样熵，并可能促进探索；可动态调制噪声强度。

**贡献：** NVFP4+LoRA、动态量化噪声 AQN 与实际可加速的内核集成。

**关键证据：**

- PDF p.7：GSM8K 7B：BF16-LoRA 88.1，NVFP4-LoRA 88.5，AQN 90.8，全参数 91.2。
- PDF p.8：BigMath 后 7B 四基准平均：NVFP4 无 AQN 37.0、有 AQN 36.4；收益不是每模型每设置单调。
- PDF p.9：端到端每步相对 LoRA 1.2–1.5 倍，QLoRA 0.7–0.9 倍，明确区分模型变小与实际变快。

**边界：** 主要数学推理/Qwen 系列；熵变化与更好探索的解释有实验支持但非普遍因果定理；速度是前30步测量，不等于整程总账。

**论文策略：** 把传统副作用变成可利用机制，再验证何时失效和实际效率。

**对我们的启发：** 负结果只有抽出可操纵、可复现的新机制才升级贡献；重新命名负结果不够。

**本次阅读范围：** PDF 1、3–10，方法、量化格式/噪声/秩对照及系统效率。

<a id="2026-sana-video"></a>
### SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/SANA-Video- Efficient Video Generation with Block Linear Diffusion Transformer.pdf>) · [归档来源](https://proceedings.iclr.cc/paper_files/paper/2026/file/41b93c59da0d0f835907fd661d419db2-Paper-Conference.pdf) · 25页 · 研究论文；版本与归档元数据见清单

**问题：** 如何以小模型和可控显存生成高分辨率长视频。

**痛点：** 视频 token 多使全注意力和随历史增长的 KV cache 昂贵；局部窗口会损失全局信息。

**朴素方案为什么不够：** 将图像线性注意力直接移到视频会缺少时空局部性；位置编码放置不当还会导致训练不稳定。

**洞察：** 线性注意力的累加状态可固定大小保留全局摘要，但需要配套时序建模、稳定归一化与长程训练。

**贡献：** 视频 Linear DiT、常量大小的块因果缓存、时序卷积及高压缩 VAE，配合分阶段训练。

**关键证据：**

- PDF p.5：RoPE/ReLU 次序及分母设计的对照揭示局部性与数值稳定问题。
- PDF p.6：累加注意力状态将历史缓存大小从随 N 增长变为 O(D²)，是摘要而非无损保存历史。
- PDF p.7：720p：SANA-Video2B 36秒、总分84.05，Wan2.2-5B116秒、83.28。
- PDF p.9：时序卷积、3D RoPE、单调时步各有消融；长视频总分83.70仍低于Self-Forcing84.31。

**边界：** 正文不同位置倍率口径并不完全一致，报告采用具体同表值；分钟生成能力不等于完美长期记忆，基础预训练投入也不能忽略。

**论文策略：** 先改变复杂度，再修复改变复杂度引入的质量与训练问题。

**对我们的启发：** 一种贡献可以有多个组件，但每个都必须对应可观察的瓶颈和独立证据。

**本次阅读范围：** PDF 1、3–9；架构、复杂度、训练策略及主表与组件消融。

<a id="2026-streamingvlm"></a>
### StreamingVLM: Real-Time Understanding for Infinite Video Streams

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/StreamingVLM- Real-Time Understanding for Infinite Video Streams.pdf>) · [归档来源](https://arxiv.org/pdf/2510.09608) · 13页 · 研究论文；版本与归档元数据见清单

**问题：** 让 VLM 持续理解长时间视频并实时解说。

**痛点：** 完整历史注意力随长度爆炸；不重叠分块损连贯，重叠分块重复计算并增延迟。

**朴素方案为什么不够：** 把现成短片模型简单接到流式缓存上不可靠，训练时看到的上下文与实时部署不一致。

**洞察：** 保留 sink、较长文本窗口和较短视觉窗口，并让训练分块模拟这种时间结构。

**贡献：** 紧凑流式缓存、连续位置编码及对齐训练，辅以长体育视频数据和实时评测。

**关键证据：**

- PDF p.2：Inf-Streams上对GPT-4o-mini的自动胜率：非重叠23.54、重叠66.54、流式66.18；主要是守住质量并降低持续成本。
- PDF p.7：原生RoPE无限推理胜率25.09、连续RoPE66.18；同时展示随视频长度延迟和OOM变化。
- PDF p.8：视觉上下文0秒52.90、16秒66.18；数据、训练和上下文长度分别消融。
- PDF p.9：同数据下非重叠训练62.51、重叠训练66.18，部分隔离数据量解释。

**边界：** 主要体育解说，自动评判器并非独立人工真值；固定窗口不保证召回任意久远事件。

**论文策略：** 新任务必须与现实约束和方法共同成立，基准服务于持续可用能力。

**对我们的启发：** 诊断基准可成为贡献组成部分，但要显示现有协议漏掉的实际决策失败，以及有用解法。

**本次阅读范围：** PDF 1–10，含方法、数据清洗、主评测、数据和训练对照。

<a id="2026-tlt"></a>
### Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/Taming the Long-Tail- Efficient Reasoning RL Training with Adaptive Drafter.pdf>) · [归档来源](https://arxiv.org/pdf/2511.16665v1) · 16页 · 研究论文；版本与归档元数据见清单

**问题：** 如何加速被少数超长回答拖住的推理 RL 训练。

**痛点：** rollout 长尾让多数 GPU 等少量请求；固定草稿模型又随目标模型更新而过时。

**朴素方案为什么不够：** 直接加投机解码会增加草稿训练成本，大 batch 下也可能更慢；更高 token 接受率不必然更快。

**洞察：** 长尾既是瓶颈也提供空闲算力；利用空闲时段更新草稿，并按实时 batch 选择投机策略。

**贡献：** 可抢占 Spot Trainer、动态草稿、分桶 CUDA graph 和自适应 rollout 调度。

**关键证据：**

- PDF p.4：逐一列出目标变化、训练额外成本、batch 波动三个挑战，并对应组件。
- PDF p.10：跨模型H100/A100端到端训练吞吐相对VeRL约1.7–2.1倍，奖励曲线相近。
- PDF p.11：多策略graph显存30.39GB→10.69GB；大batch不开投机，剩余请求少时开启。
- PDF p.12：报告权重更新、切换、协调开销，并界定对on-policy算法的保护。

**边界：** 吞吐以预热后3步及已训练草稿测量，不能自动当作从零整程GPUh节省；未来多轮工具场景属于讨论而非已验证结果。

**论文策略：** 利用瓶颈的副产物消除解决方案自身成本，同时直接优化最终目标。

**对我们的启发：** 与诊断完全对应的教训：高precision、命中率、候选接受率都可能是坏代理；应优化有用结论/总成本。

**本次阅读范围：** PDF 1、4、6、9–12，架构、现实trace动机、系统比较及关键开销。

<a id="2026-vlash"></a>
### VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference

[本地PDF](</Users/ruizhixu/Desktop/Han‘s Lab Publication/2026/VLASH- Real-Time VLAs via Future-State-Aware Asynchronous Inference.pdf>) · [归档来源](https://arxiv.org/pdf/2512.01031) · 10页 · 研究论文；版本与归档元数据见清单

**问题：** 让 VLA 在不断运动的环境中低延迟且稳定地执行动作。

**痛点：** 同步等待反应慢；简单异步生成的动作对应旧状态，与实际执行时刻不匹配。

**朴素方案为什么不够：** 直接异步切换动作会抖动；推理时纠正虽能改善，却又增加延迟；只塞未来状态模型可能不使用。

**洞察：** 机器人在推理期间会执行哪些动作已知，可据此估计执行时的本体状态，并训练模型利用它。

**贡献：** 未来状态条件化、状态/动作时移增强，以及共享视觉观测的高效微调；不需增加推理模块。

**关键证据：**

- PDF p.3：明确区分未知未来环境与可由已知动作估计的未来机器人状态，并指出直接换输入不够。
- PDF p.7：H100最坏反应延迟546.4→46.4ms；实机乒乓20次回击，同步/朴素异步/RTC均0，VLASH11次。
- PDF p.8：共享观测每步微调420.99→129.29ms，但早期收敛较慢，30K步效果接近。

**边界：** 11.8倍是定义的最坏反应延迟，不是整体任务速度；实机样本有限，状态推进依赖动作执行估计，不能预测完整未来环境。

**论文策略：** 精确指出时间错位，简单干预修复，最终用新增可完成任务验证价值。

**对我们的启发：** 我们也应把“诊断后能做出的更好判断”定义得这样具体；理论漂亮或指标变好不足以替代。

**本次阅读范围：** PDF 1–4、6–8，方法、比较、实机任务和微调代价。
