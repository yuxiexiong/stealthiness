# DC-VideoGen：先排除表示接口不匹配，再判断语义是否损坏

视频扩展，覆盖 T2V/I2V。Han Lab 与 NVIDIA 页面标为**预印本**；arXiv 首发 **2025-09-29**。阅读 **2509.25182v1，19 页**。不把 ICLR 投稿页当录用证据。

**原文与实证。** DC-AE-V 在块内双向建模、跨块保持因果性，以兼顾深压缩重建与较长输入泛化（§3.2，pp.4–5，Figs.3–5）。更值得我们借鉴的是故障诊断：直接换 VAE、随机初始化 patch embedder/output head 会出现语义下降乃至训练崩溃；先对齐 patch embedding，再冻结 DiT 对齐输出头，可先恢复部分原模型语义，再做 LoRA 适配（§3.3，pp.6–8，Figs.6–8）。Appendix A.3/Fig.11（pp.16、18）拆分检验两项对齐，而不是只报端到端分数。

**对我们的帮助（推断）。** 错图不一定说明语义权重被污染，也可能来自 VAE、latent、输入投影或输出解码接口。诊断时应把重建损失、表示对齐和目标语义正确率分开，以冻结主干、单改接口作对照；只有接口对齐仍不能恢复目标事实，才有更强理由追查主干中的异常联系。块因果结构还提示要分别查块内和跨块传播，不能只看单帧热图。

**边界。** 原文对齐目标是保留基座知识；若基座本身受污染，照搬对齐可能连错误一起保留，不能称为修复。Appendix A.8（p.18）明确其性能依赖基座，长视频生成仍列未来工作。效率测试只计 Transformer backbone（§4.1，p.8），不是完整编码、生成、解码端到端成本。官方仓库页面目前称代码与权重待发布，不能当成已验证可运行的基线。

来源：[Han Lab](https://hanlab.mit.edu/projects/dc-videogen)；[NVIDIA 发表类型](https://research.nvidia.com/labs/eai/publication/dc-videogen/)；[固定版本论文](https://arxiv.org/abs/2509.25182v1)；[官方仓库](https://github.com/dc-ai-projects/DC-VideoGen)。
