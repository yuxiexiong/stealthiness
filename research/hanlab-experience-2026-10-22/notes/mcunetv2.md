# MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning

## 0. 元信息
- NeurIPS 2021（arXiv 2110.15352v2）
- 类型：算法-系统协同（patch-based 推理调度 + 感受野重分配 + 架构/调度联合 NAS）
- 全文含附录 A–F 已读完。Figure 5/6/7 的柱状数值在 txt 中严重错位（第 321–395、445–494 行），只采用正文复述过的数字。Figure 7 的 txt 中混入了作者内部绘图标签（"finetune_proxyless384-vww_mcu128kb-1mb-patch4-fixedp_10e_qat"、"old results"、"Untitled 1"），说明 PDF 图层里残留草稿元素，不影响阅读。

## 1. 核心主张
- 主主张：CNN 的峰值内存由前几个块主导（分布失衡），只对这一段做 patch-by-patch 推理即可把峰值内存降 4–8×；再用感受野重分配把重叠计算开销压到 ~3%，并与 NAS 联合搜索，得到 MCU 上 ImageNet 71.8%、VWW 在 32kB SRAM 下 >90%、VOC 检测 mAP +16.9%。
- 次要主张：(a) 失衡是 CNN 层级结构的普遍现象（单分支/残差网络）；(b) patch 推理在训练时与普通网络完全一致（不像 RNNPool 需要改训练）；(c) 搜索空间应同时是"硬件感知 + 任务感知"的，把 r 与 w 直接放进搜索空间替代 V1 的两阶段。

## 2. 动机是怎么立住的
- Figure 1：MbV2 逐块峰值内存（int8），前 5 块 >450kB，第 3 块是其余的 8×；用 4×4 patch 后整体峰值 1372kB→172kB。这张图同时是"痛点"和"解法效果"的展示。
- 普遍性：附录 C Figure 11 对 MnasNet、FBNet、MCUNet-320kB 都画了同样的逐块分布（"连专为 MCU 设计的 MCUNet 也失衡"），patch 后降 3.5–6.1×。§2 还给出一个解析理由：每个 stage 分辨率减半（像素 4×）而通道只增 2×。
- Figure 2：检测 mAP 随分辨率下降比分类快——说明"能放下更高分辨率"对稠密预测任务有实际价值，连接到后面的检测实验。
- §2 用一个算式（3×224²+32×112²=539kB）让瓶颈具体到第一层。
- 因果链闭合：失衡 → 只对前段 patch 化 → 重叠开销 → 感受野重分配（Table 1） → 自动化（NAS）。每一步都有对应表/图。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| patch 推理降峰值内存（解析） | 5 个骨干 @224, 4×4 patch | Figure 5 | per-layer | 峰值内存 3.7–8.0×↓，MACs +8–17% |
| patch 推理降峰值内存（实测） | STM32F746 上 4 个缩放网络 | Figure 6 | per-layer w/ TinyEngine | 实测 SRAM 4–6×↓，延迟开销最低 4% |
| 感受野重分配几乎不掉精度且降开销 | MbV2 vs MbV2-RD | Table 1, Figure 4 | 原 MbV2 | 开销 10%→3%，ImageNet 72.2→72.1，VOC 75.4→75.7 |
| ImageNet 新纪录 | 两档 MCU 约束 | Table 2 | MbV2/Proxyless+TinyEngine，Rusci（混合精度），MCUNet int8/int4 | top-1：M4 64.9 vs 60.3（int8），H7 71.8 vs 68.5（int8） |
| VWW 极低内存 | 精度-SRAM、精度-延迟 | Figure 7 | MCUNet、MbV2/Proxyless+TFLite | 32kB 下 >90%，4.0× 更小 |
| 检测提升 | Pascal VOC, YOLOv3 | Table 3 | MbV2+CMSIS、MCUNet | mAP 51.4→68.3（H7），64.6（M4） |
| 人脸检测 | WIDER FACE（fp32 解析内存，沿用 RNNPool 协议） | Table 4, Table 7, Fig 12 | EXTD, LFFD, RNNPool, EagleEye | mAP Easy/Med/Hard，峰值内存 |
| 优于其他省内存方法 | 非重叠 patch、RNNPool | Table 5 | 同一 MbV2 | 是否保持平移不变、训练时间、ImageNet、VOC |
| (p, n) 超参的权衡 | 扫 n 和 p | Figure 8 | MbV2 vs MbV2-RD | 峰值内存、patch 尺寸、计算开销 |
| 联合搜索空间更好 | MobileNetV3 空间加 r、w | 附录 D Table 6 | MbV3-Small, EfficientNet-B-x, TinyNet, MnasNet 等 | 同 MACs 下 top-1（25M：63.9 vs 59.9） |
| 搜出的结构有规律 | 结构可视化 | Figure 9 | — | 定性 |

## 4. 实验设计手法
- **基线选择**：包含自家上一代 MCUNet（int8/int4）和外部 SOTA（Rusci 混合精度、RNNPool）；把低比特/混合精度结果"标灰"作为正交技术，只与同量化策略比（Table 2）。有朴素基线："非重叠 patch"（不付重叠代价但破坏平移不变性，Table 5）——这是最自然的攻击点，作者主动做了。人脸检测"按 [43] 的协议用 S3FD 训练、按 [43] 的口径算 fp32 内存"以求公平。
- **规模/广度**：5 个骨干的解析分析；3 款 MCU；4 个数据集（ImageNet、VWW、VOC、WIDER FACE）。
- **效率测量**：**解析 profiling 与真机 profiling 双轨**（§4 开头明确区分：解析只依赖结构，实测包括 im2col 等临时缓冲）。延迟在 F746 实测。RNNPool 训练时间在 Titan RTX 用官方代码测。
- **精度–效率联合呈现**：Figure 7 两条帕累托；Table 3/4 同表列内存与 mAP；附录 Table 6 按 MACs 档分组比较。
- **消融**：Table 1（RD 的作用）、Figure 8（n、p 扫描，找到 n*=5, p*=4）、Figure 7 中"同一搜索空间与超网分别在 per-layer 和 per-patch 下搜索"（§4.2，控制变量做得干净）、附录 D（uniform w vs flexible w）。
- **分析性实验**：Figure 9 搜出结构的模式总结（patch 段小核、大扩张与大核不同时出现）。
- **泛化**：多骨干、多 MCU、分类+检测+人脸。
- **统计**：无方差、无多 seed。
- **系统贡献**：demo 视频；扩展了 TinyEngine；未明说开源。
- **局限性**：无专门讨论。

## 5. 对抗式审读
- **最可能被攻击的比较**：Table 3 的检测"+16.9% mAP"。MCUNetV2 用 YOLOv3 训练（§4.3），而 MCUNet [30] 的 51.4% 是 V1 论文里用 **YOLOv2** 得到的（V1 Table 5，V1 明确说因 YOLOv3 多尺度特征峰值内存大而选 YOLOv2）。表中照搬了旧数字，检测头不同，增益里混入了检测器升级的贡献。正文对此没有说明。另外正文写 16.7%，表题写 16.9%，自相矛盾（68.3−51.4=16.9）。
- Figure 6 显示有的网络延迟开销可达 +15–21%（txt 数值错位，只能确定正文"could be large"的定性说法），正文只强调 RD 下的 4%。
- 人脸检测对比用 fp32 解析内存而非 MCU 实测；"≤3 faces"子集是作者自定的"更真实设置"。
- 内存降低数字有两套（解析 3.7–8.0×、实测 4–6×），摘要用"4–8×"把两者拼在一起。
- 无多 seed，ImageNet 上 0.1 级差异（72.2 vs 72.1）不可分辨，但作者用它说"不损精度"是合理方向。
- Table 2 的 MCUNetV2 微调 10 epoch + QAT 10 epoch（附录 B），MCUNet V1 4-bit 是 QAT 25 epoch，训练配方不完全相同。

## 6. 可迁移的手法
1. 用"逐层/逐块资源分布图"把一个全局指标（峰值内存）分解，暴露"只有少数部件是瓶颈"，从而让局部手术方案顺理成章。（Figure 1, 附录 Figure 11）**直接读到**
2. 证明现象普遍：对多个独立设计的骨干（含自家专门优化过的）重复同一分布图，并给一个结构性解释（分辨率 4× vs 通道 2×）。（§2, 附录 C）**直接读到**
3. 效率测量分"与实现无关的解析值"和"真机实测值"两轨报告，前者给普遍性，后者给可信度。（§4 Memory profiling, Fig 5 vs Fig 6）**直接读到**
4. 主动构造"最便宜的替代方案"（非重叠 patch）作为对照，展示其在敏感任务（检测）上的崩坏，堵住"何必重叠"的质疑。（Table 5）**直接读到**
5. 新方法引入开销时，同时给出"开销来源的机制量"（感受野→patch 尺寸 75→63）并设计针对性的结构修正。（Table 1, §3.2）**我的归纳**
6. 消融 NAS 收益时，固定搜索空间与超网、只切换推理方式，让差值只归因于调度。（§4.2 VWW）**直接读到**

## 7. 一句话
一张逐块峰值内存柱状图把"MCU 内存不够"重写成"只有前 5 块不够"，此后每个实验都只是在兑现这张图。
