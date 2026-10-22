# GAN Compression 精读笔记

## 0. 元信息
- 标题：GAN Compression: Efficient Architectures for Interactive Conditional GANs
- 会议：CVPR 2020 原版；本文本为 T-PAMI 扩展版（arXiv 2003.08936v4，含 Fast GAN Compression、MUNIT、COCO-Stuff、per-class 分析）
- 类型：算法（蒸馏 + NAS 压缩框架），附硬件实测与 demo
- 全文读完（正文 + 附录 6.1–6.4）。Figure 8（训练/搜索时间柱图）与 Table 5（消融）在 pdftotext 中严重交错，Table 5 的 ✓ 号与行对应关系无法完全可靠还原，下文只引用能确认的数值（edges→shoes ngf=48 最优 24.45、Cityscapes ngf=64 最优 42.11）。

## 1. 核心主张
- 主主张：一个**通用**（跨架构、跨 paired/unpaired、跨学习目标）的 cGAN 生成器压缩框架，可把 CycleGAN/Pix2pix/GauGAN/MUNIT 计算量降 9–30× 而几乎不损画质。
- 次主张：(a) 用教师输出构造伪配对，把 unpaired 转为 paired 训练更稳定；(b) 中间特征蒸馏优于输出蒸馏；(c) 继承教师判别器有益；(d) 生成器对卷积分解的敏感度分层且与识别网络不同；(e) once-for-all 解耦训练与搜索；(f) Fast 版本节省 1.6–3.7× 训练时间、3.5–12× 搜索时间。

## 2. 动机怎么立住
- 痛点量化：Figure 2 柱图——cGAN 参数量与识别网络同量级，但 MACs 高两个数量级（GauGAN 281G vs MobileNet 0.5G，562×）。这是"参数不大但计算极大"的对比式动机。
- 困难的证据：(1) GAN 训练不稳定（引述 + Fig. 10/11a 的 unpaired 小模型崩坏）；(2) 直接把识别网络压缩方法搬过来不行——"均匀缩通道超过 4× 后性能急剧下降"（Sec. 3.2，正文陈述）、Table 6 层敏感性分析显示上采样层对分解敏感而 ResBlock 不敏感、Fig. 12/Table 7 显示输出蒸馏甚至不如从头训练。
- 动机-方法因果链闭合较好：每个组件都对应一条"朴素做法失败"的实验。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 9–30× 压缩、画质基本不损 | 4 模型 × 5 数据集 | Table 1, Fig. 5/6 | Original、Shu et al.(CycleGAN 专用) | FID / mIoU + MACs/Params |
| 全尺度 trade-off 优于基线 | 尺度扫描曲线 | Fig. 7, Fig. 12 | From Scratch、Prune+Distill、Output Distillation | FID/mIoU vs MACs |
| 实测加速 | 4 种硬件 | Table 4 | Original | 延迟、内存 |
| 伪配对有效 | 不同压缩比 | Fig. 10, 11a | unpaired 训练 | FID |
| 中间蒸馏优于其他蒸馏 | 同 MACs 对比 | Table 7 | Aguinaldo、Yim、from-scratch | mIoU |
| 继承 D、剪枝+蒸馏组合 | 消融 | Table 5 | 各组件开关 | FID/mIoU |
| 卷积分解层敏感性 | 分部分解 | Table 6, Fig. 11b | 原生成器 | FID |
| 感知质量 | LPIPS + AMT 人评 | Table 3 | 0.28/0.31 Pix2pix、0.25 CycleGAN | LPIPS、偏好率 72.4% |
| 稀有类不受损 | per-class IoU | Fig. 9 | 0.31 GauGAN | IoU |
| Fast 版更省时 | 训练/搜索时间 | Fig. 8, Table 2 | GAN Comp. | 小时、搜索空间大小 |

## 4. 实验设计手法
- 基线：**同 MACs 下均匀缩通道从头训**（0.25 CycleGAN、0.28/0.31 Pix2pix、0.31/0.34 GauGAN、0.25 MUNIT）作为朴素基线，几乎每张定性图都配；任务专用 SOTA（Shu et al. co-evolution）；蒸馏方法基线（Aguinaldo、Yim）。原模型是**作者重训版本**，附录 Table 10 交代重训版 CycleGAN 比官方更好（61.53 vs 71.84），即用更强的原模型作为参照——这对自己是更严格的比较。GauGAN 则用官方预训练（因重训版更差），选择规则写明。
- 规模/广度：4 个 cGAN（含 encoder-decoder 型 MUNIT）、5 个数据集、paired/unpaired 两种设定。
- 效率测量：Jetson Xavier/Nano 的 CPU 与 GPU、1080Ti、Xeon CPU，batch=1，100 次预热 + 100 次测量（Sec. 4.3.5）；同时报内存。诚实指出 GPU 加速远小于 CPU（1080Ti 上 1.7–2.5× vs MACs 9–21×）。
- 精度–效率联合：Fig. 7、11、12 是"性能 vs MACs"曲线，覆盖多个预算点——证明不是只在一个点上赢。
- 消融：Table 5 组件开关；Table 6 分部分解；Table 7 蒸馏方法；Fig. 10 伪配对；附录 Table 9 U-Net vs ResNet 生成器。训练成本也作为指标（Fig. 6、Fig. 8、附录图注里每个模型的训练小时数）。
- 分析性实验：per-class IoU 按像素频率排序（Fig. 9），显示均匀缩通道主要伤害稀有类。
- 评测协议透明：脚注交代 CycleGAN FID 用 train+test 真图，与 Shu et al. 的协议不同，并在其协议下重算给出 65.48/96.15/69.54（Sec. 4.3.1 脚注）；Changelog 写明 Cityscapes mIoU 协议更正。
- 统计：无方差/多 seed。
- 系统：开源代码与 demo（Jetson Nano 上 edges→shoes 交互应用）。
- 局限：无单独局限节。

## 5. 对抗式审读
- 最弱环节：Table 1 的 CycleGAN "Ours"（带 fine-tuning，FID 71.81，+10.3）明显差于 "w/o fine-tuning"（64.95），文中摘要/图 1 用的是更好的那个数字；fine-tuning 何时有益、何时选哪个变体，规则未事先说明——存在按结果挑选变体的空间。
- "Without losing image quality" 措辞过强：COCO-Stuff mIoU 掉 3.04、FID +3.10；Cityscapes Pix2pix w/o fine-tuning 掉 8.71。
- GPU 实测加速（1.7–3.3×）远小于 MACs 比，头条用 MACs；Table 4 只测了三个模型在三个数据集上。
- 无多 seed，而 GAN 训练方差众所周知较大，FID 差 1–2 是否显著无从判断。
- Table 5 文本损坏，无法独立核对消融中"继承 D"的单独贡献。

## 6. 可迁移手法
1. 压缩/效率方法的朴素基线用"同 MACs 均匀缩宽从头训"，在每个定性/定量比较中都并列（Fig. 5/6, Table 3）。—— 直接读到
2. 用"性能 vs 计算量"多点曲线代替单点对比，证明优势在整个预算区间成立（Fig. 7/11/12）。—— 直接读到
3. 当与前作评测协议不一致时，在脚注里用对方协议再算一遍（Sec. 4.3.1 脚注）。—— 直接读到
4. 重训原模型并与官方权重对比，选更强者作为参照，并在附录公开选择（Table 10）。—— 直接读到
5. 用细粒度分解（per-class IoU 按频率排序）揭示"平均指标掩盖了谁在受损"，并证明自己的方法保护了长尾（Fig. 9）。—— 直接读到
6. 每个方法组件都配一个"朴素做法失败"的实验，形成"困难→组件→证据"的一一对应（Table 6/7, Fig. 10/12）。—— 我的归纳

## 7. 一句话
最关键一招是"同 MACs 均匀缩宽从头训"这一朴素基线贯穿全文（表、曲线、定性图、per-class），把"压缩后画质没掉"变成"同样算力下别的做法掉很多而我们不掉"的可比较命题。
