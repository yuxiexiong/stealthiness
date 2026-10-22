# LEGO: Spatial Accelerator Generation and Optimization for Tensor Applications

## 0. 元信息
- HPCA 2025（arXiv 2509.12053）；Yujun Lin*, Zhekai Zhang*, Song Han
- 类型：硬件生成框架（EDA/编译式：从仿射关系表示自动生成空间加速器 RTL）
- 实现层级：框架 C++ 实现，后端 LP 用 HiGHS；DAG 原语与 NoC 组件用 SpinalHDL 生成可综合 Verilog；Synopsys DC + **TSMC 28nm** 综合；SRAM 用 CACTI；前端自带性能模拟器（声称经 RTL 仿真验证）。与 SODA 对比时用 FreePDK 45nm、与 AutoSA 对比时在 Xilinx U280 FPGA 上综合。**无布局布线、无流片。**
- 全文读完（正文+参考文献；无附录）。Fig.10/11/13/14 的柱状数据被打散，只引用正文与表格中能对应清楚的数字。Table III 的 "Technology" 行被抽成 "60nm 65nm 28nm 28nm"，与 4 列的对应关系无法确认（见审读）。

## 1. 核心主张
- 主：无需手写 RTL 模板，LEGO 从张量运算的仿射关系表示自动生成空间加速器（含互连、存储系统、多数据流融合），端到端比 Gemmini 快 3.2×、能效 2.4×（Abstract, Fig.11）。
- 次：(a) 生成质量与手写设计（Eyeriss、NVDLA）相当（Table III）；(b) 比 TensorLib 省 2.6× 功耗、2.0× 面积，比 DSAGen 省 2.6× 功耗、2.4× 面积，比 AutoSA 省 5.0× LUT、6.5× FF，比 SODA 快 14×、能效 32×（Table VI–VIII）；(c) 后端 LP 优化平均省 35% 面积、28% 功耗（Fig.13/14）；(d) 多数据流融合优于单一数据流与朴素合并（Table V）；(e) 生成 16k FU 设计 <3 分钟（Table IV）。

## 2. 动机
- 痛点以**对比表**立：Table I 把 Timeloop/MAESTRO/TENET（不生成 RTL）、Gemmini/DNNWeaver/MAGNET/DNA（模板、仅 1–3 种数据流）、AutoSA/TensorLib（仅 2D 阵列、单一 kernel）、SODA（HLS）列成"RTL 生成 × 设计空间"二维，LEGO 唯一两者兼备。这是**能力矩阵式**动机，不是 profiling。
- 次痛点：控制流与数据流纠缠导致控制逻辑 2.0× 面积、2.6× 功耗开销（Sec.I，对比 TensorLib）；后端观察：FU 阵列中寄存器占 40% 面积、60% 功耗（Sec.V）→ 引出 LP 插寄存器优化。
- 因果链：动机是"灵活性 vs 生产力"的取舍，方法是"仿射表示 + 图算法 + LP"，评估覆盖"质量（vs 手写）""灵活性（多 kernel/多数据流）""生产力（生成时间）"三面，闭合度尚可；但"生产力"只用生成时间衡量，没有人力成本对比。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 端到端优于 Gemmini | AlexNet、MobileNetV2、ResNet50、EfficientNetV2、BERT、GPT-2、CoAtNet | Fig.11 | Gemmini（同 256 MAC、256KB、16GB/s） | GOP/s、GOPS/W（平均 3.2×/2.4×） |
| 生成质量≈手写 | 同数据流同设置 | Table III | Eyeriss（KH-OH）、NVDLA（IC-OC，功耗由 16nm 投影） | 面积、功耗 |
| 优于其他生成器 | | Table VI–VIII | DSAGen、TensorLib、AutoSA（U280）、SODA（45nm） | 面积/功耗/LUT/FF/GFLOPS |
| 后端优化有效 | 11 种 kernel-数据流组合 | Fig.10/13/14 | 仅 delay matching | 面积 −35%、功耗 −28%（几何平均） |
| 多数据流融合 | MobileNetV2、ResNet50 | Table V | 单数据流 ICOC×2、OHOW+ICOC、朴素合并 | 功耗、GOP/s、GOP/s/W |
| 大模型可用 | DDPM、SD、LLaMA-7B | Table II | — | 利用率、GOP/s |
| 可扩展 | 64 → 16,384 FU | Table IV | — | 生成时间、面积、功耗、能效 |
| PPU 开销小 | 分解 | Fig.12 | — | 面积 2%、功耗 5%、延迟 <7.2% |
| 配合 DSE | 用 Timeloop 搜索结果生成 | Sec.VI-B(f) | Eyeriss 同资源 | 功耗 −9% |

## 4. 实验设计手法
- 基线：Gemmini 被选为主基线，理由是"有完整端到端流程且性能相当"；为公平，资源对齐（256 MAC/256KB/128-bit 总线 16GB/s），并**只计 Gemmini 张量 kernel 本身周期**以最小化其系统开销（Sec.VI-A）。与 SODA、AutoSA 比较时分别切换到对方的工艺/平台（FreePDK45、U280）——"到对方主场比"。
- 规模：4 类张量 kernel（GEMM、Conv2d、Attention、MTTKRP）× 多数据流；10 个网络从 AlexNet 到 LLaMA-7B。
- 效率测量：前端性能模拟器 + 综合；Gemmini 侧测量方式（仿真/FireSim？）正文未明说——**读不到**。
- 消融：后端每个 pass 的贡献（Fig.13/14：归约树提取与广播重布线各约 15% 面积，pin reusing 5%，power gating 1.5% 功耗）；多数据流融合的 4 列对照（Table V）把"朴素合并"作为中间对照，显示优化合并的必要性。
- 分析：面积/功耗分解（Fig.12a：片上 buffer 占 86% 面积）、端到端延迟分解（Fig.12b）、指令开销（每条指令 >2000 周期，带宽 <1%）。
- 泛化：跨 kernel、跨模型族（CNN、Transformer、扩散、LLM）、跨实现目标（ASIC 28/45nm、FPGA）。
- 统计：无。
- 系统贡献：项目页；正文未声明开源。
- 局限：承认 GPT-2/LLaMA-7B bs=1 受带宽限制（利用率 3.1%），无独立局限章节。

## 5. 对抗式审读
- **3.2× 对 Gemmini 的来源**：作者自己说加速主要来自 (i) LEGO 前端带的性能模型帮助调度器搜索最优映射；(ii) 动态切换数据流（尤其 depthwise 层）。第 (i) 点是**映射搜索器**的优势而非"生成器"本身的优势——若给 Gemmini 同样的映射搜索，差距会缩小多少，**未测**。
- 对比生成器的数字（Table VI）多是"同延迟下的功耗/面积"，但各对比在不同工艺/平台上进行，DSAGen 与 TensorLib 的数字来源（复现还是引用）未交代。
- Table III：NVDLA 功耗为 16nm 投影值（脚注），Eyeriss 是 65nm 流片芯片数据，LEGO 是综合估算；**流片实测 vs 综合估算**放在同一表"comparable"，口径不一致。工艺行与列的对应因文本损坏我无法确认。
- "生产力"只用生成时间（<3 分钟）衡量，没有与手写 RTL 的人工工时比较；"comparable to handwritten" 只在两个设计上验证。
- 无功能正确性/时序收敛的系统性证据（除"RTL 仿真验证模拟器"一句）。

## 6. 可迁移手法
1. 框架类论文用"能力矩阵"表（列：是否生成 RTL、设计空间范围）立动机并定位自己唯一占据的格子。**直接读到**（Table I）
2. 与每个相关工作比较时，切换到对方的实现目标（对方的工艺库/FPGA 板卡/资源量）来对齐。**直接读到**（Table VI–VIII, Sec.VI-A）
3. 生成器质量用"同数据流同设置下 vs 手写专家设计"来证明。**直接读到**（Table III）——但需注意口径。
4. 多选项融合的必要性用"单一 A / 单一 B / 朴素合并 / 优化合并"四列对照证明。**直接读到**（Table V）
5. 可扩展性用生成时间 + 面积/功耗/能效随规模的表格，展示能效基本不随规模下降。**直接读到**（Table IV）
6. 对系统开销（指令、后处理单元）做专门量化，提前封堵"隐藏开销"质疑。**直接读到**（Fig.12, Sec.VI-B(e)）

## 7. 一句话
用能力矩阵定位"唯一既能生成 RTL 又覆盖任意数据流"的格子，再到每个竞争对手的主场（工艺/FPGA/资源量）逐一对齐比较。
