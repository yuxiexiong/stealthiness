# TSM: Temporal Shift Module for Efficient Video Understanding

全文已读完（正文 + 附录 A/B）。raw/txt/tsm.txt 双栏 layout 交错严重，我用 `pdftotext` 非 layout 模式从 raw/pdf/tsm.pdf 生成临时文本（/tmp/g6txt/tsm.txt）阅读。Table 8 的列对应关系需按数值重建（见下），Fig 2/5/6 曲线只能读到坐标与标签。

## 0. 元信息
ICCV 2019。Lin, Gan, Han。类型：算法（零计算时序建模算子）+ 边缘部署。

## 1. 核心主张
- 主主张：沿时间维移动一部分通道（TSM）可以让 2D CNN 获得接近/超过 3D CNN 的时序建模能力，零额外 FLOPs、零参数（Abstract、Sec 3.1）。
- 次主张：(a) 朴素全通道 shift 既不高效（CPU 延迟 +13.7%、GPU +12.4%）也不准确（比 TSN 掉 2.6%），需要 partial shift（1/4）+ residual shift（Sec 3.2/3.3, Fig 2）；(b) 单向 TSM 支持在线低延迟识别，每帧开销 <0.1ms（Table 6）；(c) Something-Something 排行榜第一；(d) 边缘设备 Jetson Nano 13ms、Note8 35ms（Table 8）。

## 2. 动机是怎么立住的
- 概念动机：把卷积拆成 shift + multiply-accumulate（Eq 1/2），shift 不需要乘法——**推导型直觉**。
- **"朴素做法失败"的实测**（核心手法）：Fig 2a 在 P100、TX2、Xeon CPU 三设备上测 shift 比例 0/1/8/1/4/1/2/1 时的延迟开销（全 shift 最高 13.7%，1/8 约 3%），1000 次运行取平均、200 次 warm-up；Fig 2b 在 Kinetics 上比较 in-place vs residual shift 各比例的精度。两张图直接决定了"1/4 + residual"的默认设计。观察 → 设计的因果链闭合得非常好。
- 数据集选择也被论证：区分"时序弱"（Kinetics/UCF/HMDB）与"时序强"（Something-Something/Jester）两组，预测 TSM 在后者增益更大（Sec 5.1）——**事前分组 + 预测**，Table 1 结果吻合（+3.5 ~ +8.8 vs +11.7 ~ +31.3）。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 朴素 shift 慢 | shift 比例扫描 × 3 硬件 | Fig 2a, Sec 3.3 | 2D 基线 | 延迟开销 % |
| 朴素 shift 不准，residual 更好 | in-place vs residual × 比例 | Fig 2b | TSN | Kinetics Top-1 |
| 零成本大幅提升 2D | 6 数据集 | Table 1 | TSN（同协议，唯一区别是有无 TSM） | Top-1/5 |
| 跨骨干 | MBv2、R50、RX101、NL-R50 | Table 3 | TSN | Kinetics Top-1 |
| SOTA on Sth-Sth | V1 / V2 | Table 2, Table 4 | TSN、TRN（含自己复现）、ECO、I3D、NL I3D、NL I3D+GCN | Top-1/5、FLOPs、参数 |
| 更好的精度-成本 Pareto | — | Fig 5 | ECO 族、I3D 族 | Acc vs FLOPs（圆面积=参数） |
| 实测更快 | P100 batch 1 延迟、batch 16 吞吐 | Table 5 | I3D、ECO、I3D from [49]、I3D_replace | ms、V/s |
| 在线版本几乎不掉点、近零开销 | 4 数据集 | Table 6 | 离线 TSM、TSN | Acc、ms（TVM 编译） |
| 早期识别 | UCF101 观察比例扫描 | Fig 6 | ECO（3 种设置） | Acc |
| 在线视频检测 | ImageNet-VID R-FCN | Table 7, App A, Fig 7/8 | R-FCN、FGFA | mAP（慢/中/快） |
| 边缘部署 | MBv2 在线 TSM | Table 8 | — | ms、W |

## 4. 实验设计手法
- 基线：**TSN 作为"只差一个模块"的对照**（"The only difference is with or without TSM"，Sec 5.2），这是最干净的消融式基线；TSN/TRN 都有"our impl."同骨干复现（Table 2）；SOTA 覆盖 2D（TSN、TRN）、混合 2D/3D（ECO）、3D（I3D、NL I3D、+GCN）。对 GCN 指出其使用外部 COCO 数据而改与其 CNN 部分比较（Sec 5.3）。
- **I3D_replace 对照**（Table 5）：把每个 TSM 换成 3×1×1 时序卷积，排除"骨干设计差异"——这是分离"shift 本身"与"架构"的反事实对照，FLOPs 更高（48G vs 33G）、延迟更高（28.0 vs 17.4ms）、精度更低（44.9 vs 45.6）。
- 规模/广度：6 个视频识别数据集 + 视频检测；4 种骨干；硬件覆盖 P100、TX2、Jetson Nano、Xeon CPU、Raspberry Pi 4B、Note8、Pixel 1。
- 效率测量：**实测**，协议交代：P100 latency batch 1、throughput batch 16；Fig 2a 1000 次平均 + 200 warm-up；在线与边缘用 TVM 编译；功率用功率计并减去静态功耗（Sec 5.5）。Table 8 重建：Jetson Nano CPU 47.8ms/4.8W、GPU 13.4ms/4.5W；TX2 CPU 36.4ms/5.6W、GPU 8.5ms/5.8W；Raspberry Pi 69.6ms/3.8W；Note8 34.5ms；Pixel1 47.4ms（列对应按数值与 Abstract "Nano 13ms、Note8 35ms"推断）。
- 精度–效率联合：Fig 5 Pareto（FLOPs × 精度 × 参数面积）；Table 5 同时报 FLOPs/参数/延迟/吞吐/精度。
- 消融：shift 比例 × 位置（Fig 2）；有无 TSM（Table 1/3）；TSM vs 时序卷积（I3D_replace）；离线 vs 在线（Table 6）。
- 分析性实验：检测可视化（Fig 7/8，车灯眩光、遮挡案例）；快/中/慢物体分组 mAP（Table 7，快速物体 +4.6）。
- 泛化：跨数据集、骨干、任务（识别→检测）、模态（光流两流 +5.4/+2.6）、离线/在线、硬件。
- 统计：无方差、单次。
- 系统贡献：开源；Something-Something 排行榜第一（外部背书）；在线 demo 与项目页。
- 局限性：无专门段落；承认在线版在 Sth-Sth 上比离线低 1.0%。

## 5. 对抗式审读
- 最薄弱环节：shift 比例 1/4 是在 Kinetics（时序弱）上选的（Fig 2b），却用于所有数据集，未在 Sth-Sth 上验证该选择是否仍最优。
- 与 NL I3D 的比较（"1.2% 更好、10× 少 FLOPs"）混合了不同输入帧数（8 vs 32×2 clip）和测试协议，作者在表注中有说明，但"10×"是跨协议数字。
- ECO 数据引自原文；Table 5 中 ECO 延迟由作者在 P100 上实测（与 TSM 同平台），较公平。
- 边缘表（Table 8）只有 TSM 自身数字，没有 2D/3D 基线在同设备上的对比，"edge friendly"无相对参照。
- 统计：Table 6 中在线 vs 离线差 ±0.4% 的结论在单次运行下不可靠。

## 6. 可迁移手法
1. 先实测"朴素版本"在效率和精度上双双失败（13.7% 延迟开销、−2.6% 精度），再用两个最小修改修复，每个修改都由一张扫描图选定超参（Fig 2a/2b）。—直接读到
2. 用"唯一差别是有无模块"的同协议基线（TSN vs TSN+TSM）跨 6 数据集报告增益，并事前把数据集按"时序依赖强弱"分组、展示增益随依赖强度上升（Table 1）。—直接读到
3. 用"把新算子换成等价的常规算子"（I3D_replace：TSM→3×1×1 conv）做反事实对照，排除骨干差异（Table 5）。—直接读到
4. 效率主张用 FLOPs + 参数 + 实测延迟 + 吞吐四项同表（Table 5），并在 FLOPs 相近时用实测揭示差距（I3D from [49] FLOPs 相近但延迟 1.5×）。—直接读到
5. 对外部数据（COCO 训练的 RPN）带来的不公平主动剔除，改与其无外部数据的部分比较（Sec 5.3）。—直接读到
6. 用竞赛排行榜名次作为第三方验证（Abstract）。—直接读到

## 7. 一句话
先用三硬件延迟扫描和精度扫描证明"朴素 shift 又慢又差"，再用 1/4 + residual 两处最小修改修好，最后在"唯一差别是有无 TSM"的同协议对照里跨 6 数据集拿到与时序依赖强度成正比的增益——每一步都有对应的实测图表。
