# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction

全文已读完（arXiv v6，正文 + 参考文献；无附录）。raw/txt 版双栏交错且含图中嵌入数据，我另用 `pdftotext` 非 layout 模式从 raw/pdf/efficientvit.pdf 生成临时文本（/tmp/g6txt/efficientvit.txt）阅读。Table 1 的勾选列错位（按数值推断行含义，见下），Table 5（超分）部分数字与 Fig 7 数据混在一起，只能读出部分；Fig 1/4/6/7 只有标签。

## 0. 元信息
ICCV 2023（arXiv 2205.14756 v6）。Cai, Li, Hu, Gan, Han。类型：算法（高效视觉骨干）+ 多硬件实测。

## 1. 核心主张
- 主主张：用 ReLU 线性注意力（全局感受野，O(N)）+ 小核卷积聚合的多尺度 token（多尺度学习）构成的模块，只用硬件友好算子，就能在高分辨率稠密预测上以显著更低延迟达到/超过 SOTA（Abstract、Sec 1）。
- 次主张：(a) ReLU 线性注意力比 softmax 注意力在移动 CPU 上快 3.3–4.5×（相近计算量，Fig 4）；(b) 但它注意力分布不尖锐、局部能力弱（Fig 3），需要 DWConv + 多尺度聚合补偿；(c) "因为没有硬件不友好算子，所以 FLOPs 降低能转化为延迟降低"（Sec 1）；(d) 推广到 ImageNet、超分、SAM。

## 2. 动机是怎么立住的
- 两个"必要特性"（全局感受野、多尺度学习）来自对 SOTA 方法的归纳（SegFormer、SegNeXt、DeepLab 等，Sec 1）——**引用型**动机；Table 1 消融后来验证这两个特性都必要。
- 硬件效率痛点：SegFormer 的 softmax 注意力随分辨率二次增长；SegNeXt 的 21×21 大核需要特殊硬件支持（引用 RepLKNet、Lite Pose）——文字论证。
- **Fig 4（本文 profiling）**：Snapdragon 855 CPU（TFLite、bs1、fp32）上，输入特征图 24/32/40 时 softmax vs ReLU 线性注意力延迟，3.3–4.5× 差距；这把"去掉 softmax"从复杂度论证落到实测。
- **Fig 3（机制可视化）**：softmax 注意力分布尖锐，ReLU 线性注意力平坦 → 局部能力弱 → 引出卷积补偿。这是"先承认自己方法的缺陷，再给出修补"的动机链。
- 因果链闭合：需求（两特性）→ 选择算子（Fig 4 实测）→ 缺陷（Fig 3）→ 修补 → Table 1 消融。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 线性注意力硬件快 | 特征图尺寸扫描 | Fig 4 | softmax 注意力 | 移动 CPU ms |
| 线性注意力局部弱 | 注意力分布可视化 | Fig 3 | softmax | 定性 |
| 多尺度 + 全局都必要 | 等 MACs 下 2×2 消融 | Table 1 | 去掉其一/全部 | Cityscapes mIoU |
| 分类骨干有竞争力 | ImageNet | Table 2, Fig 1 | CoAtNet、ConvNeXt、Swin、EfficientNetV2、FasterViT | Top-1、Nano/Orin 延迟、A100 吞吐 |
| Cityscapes 上更快且不掉点 | 按 mIoU 分组 | Table 3 | DeepLabV3+、SegFormer、SegNeXt | mIoU、MACs、Nano/Orin ms、A100 img/s |
| ADE20K 同上 | 按 mIoU 分组 | Table 4 | SegFormer、SegNeXt、Mask2Former、MaskFormer | 同上 |
| 超分更快 | BSD100、FFHQ | Table 5 | Restormer、SwinIR、VapSR、BSRN | PSNR、SSIM、A100 ms |
| SAM 上 48.9× 吞吐 | COCO/LVIS 零样本 | Table 6/7, Fig 7 | SAM-ViT-H、EfficientSAM、EdgeSAM、MobileSAM | mAP、img/s |
| "FLOPs 降低可转化为延迟降低" | 间接（各表同时列 MACs 与实测） | Table 2–4 | — | — |

Table 1（按数值推断）：两者都去掉 68.1；只有其一 72.3 / 72.2；两者都有 74.5（均 0.7M 参数、4.4G MACs，宽度缩放对齐）。

## 4. 实验设计手法
- 基线：每个任务选该任务的最强 SOTA（分割 SegFormer/SegNeXt/Mask2Former；超分 Restormer/SwinIR 与 CNN 类 VapSR/BSRN；分类 CoAtNet/ConvNeXt/Swin/EfficientNetV2/FasterViT；SAM 系列）。**按精度分组比效率**（Table 3 注 "Models with similar mIoU are grouped"）——同精度比延迟。所有基线延迟/吞吐都在作者相同硬件、相同 TensorRT fp16 下实测。
- 规模/广度：4 类任务（分类、语义分割×2 数据集、超分×2 设置、SAM 零样本 COCO/LVIS 框/点提示）；硬件：Snapdragon 8Gen1 CPU（TFLite fp32 bs1）、Jetson Nano、Jetson AGX Orin（TensorRT fp16 bs1）、A100（TensorRT fp16 吞吐）；Fig 4 用 Snapdragon 855。
- 效率测量：**实测**，交代框架、精度、batch，且**包含数据传输时间**（Sec 3.1）。同表同时报 Params、MACs、两种边缘延迟、云端吞吐。
- 精度–效率联合：Fig 1 三张 Pareto 图（Cityscapes/ADE20K vs Orin 延迟、ImageNet vs A100 吞吐）；Table 3/4 分组呈现。
- 消融：Table 1 二因子消融，**宽度缩放对齐 MACs 且从随机初始化训练以消除预训练影响**（Sec 3.2）。没有对多尺度分支数、聚合核大小（文中称两分支、5×5 为"best trade-off"，无数据）、DWConv in FFN 的单独消融。
- 分析性实验：Fig 3 注意力分布、Fig 4 算子延迟、Fig 6 定性分割结果（边界、小物体）。
- 泛化：跨任务、跨硬件很广。
- 统计：无方差。
- 系统贡献：开源（GitHub）；后续 EfficientViT-SAM。
- 局限性：承认单点提示 LVIS 上不如 SAM-ViT-H（Table 7，56.6 vs 59.2），推测因训练未含交互式设置。

## 5. 对抗式审读
- 最薄弱环节：Abstract 的"13.9× / 6.2× GPU 延迟降低（vs SegFormer / SegNeXt）"在正文表中找不到直接对应（正文写 Orin 上"up to 8.8× / 3.8×"、A100 吞吐"10.2× / 3.9×"）；可能来自不同 GPU 或 arXiv 版本迭代，需核对，属于"摘要数字超出正文可查证范围"。
- 基线延迟由作者在自家 TensorRT 管线上测，SegFormer/SegNeXt 是否做了同等程度的部署优化（算子融合、TRT 插件）未说明；大核卷积、softmax 在 TRT 上的实现质量直接影响倍数。
- Table 1 的消融只在最小模型（B0 级 0.7M）上做，且随机初始化；大模型上两个组件的相对贡献未知。
- SAM 实验用 SA-1B 全量 + SAM 图像编码器蒸馏，训练资源远大于一般比较对象（MobileSAM 等），效率对比只看推理。
- 不同任务的版本更新（v6 加入 SAM、FasterViT 等）使论文成为"不断追加 SOTA 的活文档"，早期版本的主张与当前表格不完全对应。

## 6. 可迁移手法
1. 先实测"关键算子"的延迟差异（softmax vs ReLU 线性注意力，Fig 4），把复杂度论证落到具体硬件上。—直接读到
2. 主动可视化所选算子的缺陷（Fig 3 平坦注意力），再引入补偿组件，并用二因子消融证明两者都必要（Table 1）。—直接读到
3. 消融时宽度缩放对齐 MACs、从随机初始化训练，排除计算量和预训练两个混淆（Sec 3.2）。—直接读到
4. 主表"按精度分组比效率"，每组内同时列 Params/MACs/边缘延迟/云端吞吐，让读者看到 MACs 降低是否真正转化为延迟（Table 3/4）。—直接读到
5. 测量包含数据传输时间，并注明 TensorRT/TFLite、fp16/fp32、batch（Sec 3.1）。—直接读到
6. 我的归纳：同一骨干在分类、分割、超分、SAM 四类任务上各给一张表，把"架构"包装成通用基础组件而非单任务技巧。

## 7. 一句话
先在手机 CPU 上实测 softmax 与 ReLU 线性注意力的延迟差（Fig 4），再可视化线性注意力的缺陷并用等 MACs 消融证明补偿必要（Fig 3 + Table 1），最后在四类任务上"同精度分组比实测延迟"——效率论证全程落在真实硬件上。
