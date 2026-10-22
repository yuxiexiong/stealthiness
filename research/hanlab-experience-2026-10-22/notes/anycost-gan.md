# Anycost GAN 精读笔记

## 0. 元信息
- 标题：Anycost GANs for Interactive Image Synthesis and Editing
- 会议：CVPR 2021（arXiv 2103.03243v1）
- 类型：算法（弹性生成器训练 + 一致性投影），附硬件实测与交互 demo
- 全文读完（正文 + 附录 A–H）。图中图像内容不可读，仅依据图注与表格。

## 1. 核心主张
- 主主张：训练**一个**支持弹性分辨率与通道数的 StyleGAN2 生成器，其子生成器输出与全生成器**感知一致**，因此可作为交互编辑时的快速预览（最高 10× MACs 减少，CPU 上 ~12× 实测加速）。
- 次主张：(a) 采样式多分辨率训练比单分辨率专门模型和 MSG-GAN FID 更好；(b) 生成器条件化判别器解决多子生成器共训的监督不足；(c) 在相同 MACs 下优于蒸馏和通道剪枝基线；(d) consistency-aware 编码器/优化让投影与编辑后的一致性保持。

## 2. 动机怎么立住
- 痛点：StyleGAN2 144 GMACs，是 ResNet-50 的 36×；单次编辑在边缘设备上需数秒（Sec. 1）。秒级延迟这一点是陈述，未在正文给测量（Table 5 间接给出 Xeon 0.63 FPS）。
- 类比式动机：借用渲染软件的"预览/最终渲染"工作流（Maya、Blender、Premiere），把问题从"压缩"重新定义为"一致性预览"——这是本文最有特色的立论方式，使"一致性"成为新的一等指标。
- 关键观察：仅降分辨率 1024→256 只减 1.7× 计算（144G vs 85G，Sec. 3.1 脚注），引出必须加通道维度；Fig. 4a/b 可视化 StyleGAN2 中间层输出不自然，Fig. 4c/d 可视化无一致性损失时子网输出不一致。
- 因果链：每个组件对应一个失败现象并有对应消融（Table 1、Table 2、Fig. 4），闭合较好。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 采样式多分辨率训练更优 | FFHQ 各分辨率 FID | Table 1 | 单分辨率专门模型、MSG-GAN | FID-70k |
| G 条件判别器 | 分辨率 × 通道网格 | Table 2 | vanilla（多个独立 G）、same D、reduced ch | FID-70k，用红/绿标注优劣 |
| 优于压缩基线 | FID/LPIPS vs MACs 曲线 | Fig. 5 | Vanilla、Distill、ChannelPrune | FID、LPIPS Diff |
| 高分辨率质量 | 1024 FFHQ、512 Car | Table 3 | StyleGAN2 | FID-50k、PPL |
| 属性一致性 | 40 属性分类器匹配率 | Table 4 | 两个独立训练生成器（Separate） | match rate |
| 实测加速 | Xeon CPU、Jetson Nano GPU | Table 5 | 全生成器 | FPS |
| 一致性投影 | CelebA 测试集 | Table 6, Fig. 12/13 | ALAE、IDInvert、pSp、fullG-only 编码器 | LPIPS/MSE |
| 编辑后一致 | 定性 | Fig. 8, 18 | 全生成器 | **仅定性，未量化** |
| 多设备部署的训练成本优势 | GPU 小时 vs 设备数 | 附录 Fig. 10 | 逐个压缩 | GPU hours |

## 4. 实验设计手法
- 基线：Vanilla（每种配置各训一个模型，即"专门模型"上界式对照）、蒸馏（Aguinaldo）、通道剪枝（He et al. 移植到 GAN）、MSG-GAN；投影侧有 ALAE/IDInvert/pSp。注意 Distill/ChannelPrune 是作者自己适配到 StyleGAN2 的实现，调参细节未交代。
- 规模/广度：一个架构（StyleGAN2）、两个数据集（FFHQ、LSUN Car）。消融用 Config-E 半通道 256 分辨率加速（Sec. 4.1）——主结论在 Config-F 高分辨率上复核（Table 3）。
- 效率测量：Xeon CPU 与 Jetson Nano GPU 实测 FPS（Table 5）；报告了超线性加速并给出 cache 假说（16.5MB）；无 batch 扫描，测量协议（预热/次数）未交代。
- 精度–效率联合：Fig. 5 是 FID–MACs 和 LPIPS Diff–MACs 双曲线，单模型对多模型。
- 新指标：一致性被定义为可测量——LPIPS Diff（子网 vs 全网）和属性匹配率（Table 4）。这是把"预览可用"这一产品需求转成指标。
- 消融：Table 1/2 系统消融，Table 2 用颜色标注每格相对多 G 基线的好坏。
- 分析性：Fig. 11 / 附录 D.2 指出"重建损失更低的投影不一定更可编辑"的反例，说明指标局限。
- 统计：无。
- 系统：开源代码与交互 demo（附录 A）。
- 局限：通道数控制对用户不直观；对所有像素等权逼近（Sec. 5）；附录 H 给出失败案例（耳环、轮辋等细节在小生成器中改变）。

## 5. 对抗式审读
- **数字不一致**：Sec. 1 与摘要称 Jetson Nano 上 8.5×（摘要"6–12×"），但 Table 5 中 Nano GPU 最高 6.3×（10× FLOPs 处）；Xeon 正文 12.2× vs 表中最高 11.9×。正文头条数字与表格不符，读者只能信表。
- Table 4 的属性**是按"两个独立生成器最不一致"挑选的**（表注明说），这让 Separate 基线显得极差、对比被放大；挑选准则对自己有利。
- Table 3 声称"similar or better"，但 FFHQ 0.5× 的 PPL 是 147.2 vs StyleGAN2 124.5，更差；LSUN Car 1.0× FID 2.38 vs 2.32 也稍差。措辞偏乐观。
- 编辑一致性（核心应用场景）只有定性图，且附录图 12/13 明说"hand-picked to reflect the difference"。
- Fig. 5 的 Distill/ChannelPrune 基线是作者自行移植，是否充分调参不可知。
- 单架构，无多 seed。

## 6. 可迁移手法
1. 借用成熟领域的工作流（渲染预览）重新定义问题，从而引入新的一等指标（子网-全网一致性）。—— 我的归纳（Sec. 1 直接读到类比，指标化为归纳）
2. 把产品需求"预览要像最终结果"量化为两层指标：像素/感知层（LPIPS Diff）+ 语义层（属性分类器匹配率）（Fig. 5b, Table 4）。—— 直接读到
3. "一个模型 vs 每个预算各训一个模型"的对照，并在网格表中逐格标红/绿（Table 2）。—— 直接读到
4. 消融在小配置（Config-E、256 分辨率）上做，主结论在全配置上复核（Sec. 4.1 + Table 3）。—— 直接读到
5. 报告训练摊销：设备数增加时边际成本的曲线（附录 Fig. 10）。—— 直接读到
6. 主动展示指标与目标脱节的反例（低 LPIPS 投影反而不可编辑，Fig. 11）。—— 直接读到

## 7. 一句话
最关键一招是把"压缩"重构为"一致性预览"，并把一致性做成可测指标（LPIPS Diff + 属性匹配率），在单模型 vs 多模型的 MACs 曲线上胜出——但头条加速数字与表格不一致，需要以 Table 5 为准。
