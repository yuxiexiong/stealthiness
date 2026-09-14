# SANA-Video：归因对象需要包含累计状态与局部卷积

视频扩展；架构联合支持 T2I/T2V/I2V，但本文重点为视频生成。Han Lab 标注 **ICLR 2026 Oral**；arXiv 首发 **2025-09-29**。阅读 **ICLR 2026 最终发表版，25 页**，不将官方代码首页的 SANA-Video 2.0 混入本篇。

**原文与实证。** 作者将 SANA 文生图模型适配为视频模型，在线性自注意力中加入 3D RoPE，在 Mix-FFN 中加入时间卷积；RoPE 与 ReLU 的次序及分母归一化影响局部性和数值稳定性（§3.2，pp.4–5，Fig.3）。Block linear attention 累计保存状态和键的和，而非逐 token 保存传统 KV；此外还缓存上一块末帧供时间卷积使用（§3.3，p.6，Eq.3/Fig.4；Appendix C.3，p.18，Alg.1）。Table 5（p.9）分别检验 RoPE、时间卷积、递增时间步采样；LongSANA 的长视频结果另列于 Table 9（p.22）。这些是架构、训练与生成质量实验。

**对我们的帮助（推断）。** 从普通 softmax 模型搬归因方法时，不能假定始终存在可直接读取的完整注意力矩阵。可记录各时间块对累计状态的增量，并把注意力累计状态、归一化键和、卷积缓存分别替换，观察对象身份、属性和运动何处恢复。这样有望分开全局语义残留与局部时间传播，避免把所有错误归给 cross-attention；但改变状态会影响后续轨迹，仍需配对重放与输出验证。

**边界。** 常量缓存指内存规模不随历史长度线性增长，不保证无限长生成准确，更不保证信息可逆或污染可删。默认 480p 与 720p 配置还使用不同 VAE（pp.7–8），不能把重建变化误判为语义修复。使用了内部图像/视频及筛选流程（Appendix C.2，p.17），完整训练不能仅凭公开代码推定可复现。

来源：[Han Lab](https://hanlab.mit.edu/projects/sana-video)；[首发记录](https://arxiv.org/abs/2509.24695)；[ICLR 正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/41b93c59da0d0f835907fd661d419db2-Paper-Conference.pdf)；[官方代码](https://github.com/NVlabs/Sana)。
