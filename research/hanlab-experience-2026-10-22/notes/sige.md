# SIGE 精读笔记

## 0. 元信息
- 标题：Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models（SIGE）
- 会议：NeurIPS 2022 原版，本文本为 T-PAMI 扩展版（arXiv 2211.02048v4）
- 类型：算法-系统协同（空间稀疏推理算法 + 稀疏推理引擎）
- 全文读完（正文 + 附录 A–E）。双栏混排，表格数字已逐行核对。Table 1 中 DDPM "40% Pruning" 的 LPIPS w/ G.T. 记为 0.991，与同列其他值（0.07 量级）差一个数量级，疑为原文笔误（应为 0.091？），无法从文本确认。

## 1. 核心主张
- 主主张：交互式编辑中用户只改小区域，缓存原图各层激活、只在编辑区做稀疏计算，可在**不损失画质**的前提下把计算量的减少转化为**多种硬件上的实测加速**（DDPM 3090 上 3.0×，SD 7.2×，GauGAN 5.6×）。
- 次主张：(a) 激活差 ΔA 在编辑区呈结构化空间稀疏；(b) 朴素 gather/scatter 开销大，需要 norm 预计算 + kernel fusion 才能兑现加速；(c) 可与模型压缩（GAN Compression）叠加；(d) attention 层也满足局部性，可剪 query。

## 2. 动机怎么立住
- 痛点：Figure 2 用具体编辑例子（9.4% 面积）说明 vanilla SDEdit 80% 计算浪费在未编辑区；GauGAN 281 GMACs 是 MobileNet 的 500×（Sec. 2）。
- 关键观察可视化：Figure 4 逐分辨率（256→8）画 DDPM 每层 |ΔA_l| 的通道平均，显示非零值聚集在编辑区；Figure 6 对 self-attention 层的输入/输出差与下采样 mask 对比，证明扩展到 attention 时假设仍成立。观察与方法的因果链闭合良好（观察的就是方法所复用的逐层激活）。
- 失败尝试被写出：阈值化剪 ΔA（非结构稀疏）无实测加速（Sec. 3.1）；朴素 gather/scatter 在一个 15.5% 编辑例子上开销占 41%（0.17ms / 0.42ms，Sec. 3.2）——用 kernel 级 profiling 引出系统优化的必要。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 无画质损失 | 4 模型 × 3 自建基准 | Table 1, Fig. 8 | Patch、P% Pruning、0.19 GauGAN、GAN Comp.(S) | PSNR/LPIPS(w/ G.T., w/ Orig.)、FID、mIoU |
| 实测加速、跨硬件 | 5 种设备 | Table 2 | Original、0.5 Original、Crop、GAN Comp. | 延迟 |
| 每个系统优化的贡献 | 逐步累加 | Table 3 | 无 → Sparse → +Norm → +Elem → +Sct → +Attn | 54.6→19.1ms (DDPM)；SD 1.1×→4.8× |
| 与压缩叠加 | GAN Comp.+Ours | Table 1/2, Fig. 20 | GAN Comp.(S)、0.19 GauGAN | 40–50× MACs |
| 编辑大小边界 | 编辑比例扫描 | Fig. 10 | Original 虚线 | 3090 上 ~70%、i9 上 ~45% 仍有加速 |
| 连续编辑 | 一次预计算 vs 增量 | Fig. 12 | Full Model | 定性 + MACs/延迟 |
| 后端无关 | TensorRT | Table 4 | PyTorch | 2.9×→3.3×（1.2% 编辑） |
| 内存代价 | peak memory | Sec. 4.3 | Original | +0.1–0.8 GB |
| 编辑区画质 | 只算编辑区 | 附录 Table 5 | 同 Table 1 | — |

## 4. 实验设计手法
- 基线：朴素空间基线 Patch（裁剪最小块）与 Crop（逐层裁剪矩形）——专门针对"为什么不直接裁剪"；模型维度基线（P% 剪枝不微调，"公平因为我们也不微调"；0.5 Original 线性缩通道；GAN Compression 自家前作）。缺少同类空间稀疏方法（SBNet 只在 related work 讨论，未作为实验基线）。
- 规模/广度：4 个模型（DDPM、PD、SD、GauGAN）、GAN+扩散两大家族、3 个自建基准 + 附录 CelebA/AFHQ；5 种硬件（3090、2080Ti、i9 CPU、M1 Pro CPU/GPU）+ TensorRT 后端。广度是本文强项。
- 效率测量：真实硬件 batch=1（"模拟真实使用"）；GPU 200 次预热 + 200 次测量，CPU 10+10 重复 5 次取均值（Sec. 4.2）；MACs 与延迟同表；编辑比例扫描（Fig. 10）；内存开销单独报告。
- 精度–效率联合：每张定性图都标 MACs/FID 或 mIoU；Fig. 10 给计算/延迟-编辑比例曲线；无 Pareto 曲线但 GauGAN 系列（Ours、GAN Comp.、GAN Comp.(S)、0.19 GauGAN、GAN Comp.+Ours）实质上构成一组效率-质量点。
- 消融：Table 3 是典型"roadmap"逐步累加，每步给延迟；dilation 超参扫描（Fig. 13）。
- 质量双口径：w/ G.T. 与 w/ Orig.，w/ Orig. 更能区分（DDPM PSNR w/ Orig.：Ours 52.4 vs Patch 40.6 vs Pruning 31.0）。
- 分析性：激活差可视化（Fig. 4, 6）。
- 统计：无方差。
- 系统：开源代码、基准与 demo；MPS 后端；TensorRT 实现。
- 局限：额外内存；文本变化需重算缓存、不能用于 text-to-image；低分辨率层加速有限；GauGAN 偶有接缝；无法处理全局光照变化（Sec. 5）。另有 forensics 检测（GAN 97.2% AP，扩散仅 56.6/52.4%）。局限写得较完整。

## 5. 对抗式审读
- 最弱环节：评测基准全部自建、编辑是**合成**的（LSUN 用 Detic+CoModGAN 生成、Cityscapes 粘贴物体、LAION 随机圆形遮罩，附录 B），真实用户编辑仅 23 个。"用户倾向于逐步编辑小区域"这一前提没有用户数据支持。
- 头条延迟数字均来自**挑选的单个编辑样例**（Table 2 标注"examples shown in Figure 8 and 9"），而质量是数据集平均——效率与质量不在同一样本集上。Table 1 中 SD 平均 MACs 只降 2.1×，而摘要的 7.2× 来自 1.7% 编辑的单例。
- 基线 P% Pruning 不微调本身就是很弱的稻草人；"0.5 Original"只用于效率不报质量。
- SD 的 PSNR w/ Orig. 只有 19.9（Table 1），与 DDPM 的 52.4 相差悬殊，说明 SD inpainting 下并非"与原模型一致"，正文用 LPIPS/FID 持平来叙述，回避了这一点。
- Table 1 疑似笔误（见元信息）。
- 无多 seed/置信区间。

## 6. 可迁移手法
1. 系统优化论文用"逐步累加"表把理论计算减少与实测延迟分开列（Sparse 只给 1.6× 延迟但 7.6× MACs），证明系统工程的必要性（Table 3）。—— 直接读到
2. 先用 kernel 级 profiling 写出一个具体开销百分比（gather/scatter 占 41%），作为系统优化的动机（Sec. 3.2）。—— 直接读到
3. 对依赖输入属性（编辑大小）的加速，画"加速 vs 属性"曲线并标出盈亏平衡点，主动划定适用边界（Fig. 10）。—— 直接读到
4. 跨硬件谱系（数据中心 GPU、消费 GPU、桌面 CPU、笔记本 CPU/GPU）+ 多后端，证明加速不是某个硬件的偶然（Table 2/4）。—— 直接读到
5. 正交性论证：把方法叠加在自家/他人的压缩方法上，并与"同等 MACs 的更激进压缩"对比（GAN Comp.+Ours vs GAN Comp.(S) vs 0.19 GauGAN，Table 1）。—— 直接读到
6. 方法依赖的近似假设，直接可视化方法所复用的那个中间量（逐层 ΔA），而非输入代理量。—— 我的归纳（对比 DistriFusion 只测输入）

## 7. 一句话
最关键一招是 Table 3 的逐步累加：MACs 早已降 7.6× 而延迟只降 1.6×，再一项项加系统优化直到 2.9×——把"算法稀疏 ≠ 实测加速"的鸿沟和自己如何填平它摆在一张表里。
