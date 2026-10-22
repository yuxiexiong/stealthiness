# QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving

## 0. 元信息
- MLSys 2025，arXiv 2405.04532v3
- 类型：算法-系统协同（QoQ 量化算法 + QServe 服务系统）
- 全文已读（正文 + Artifact Appendix）。Figure 15/16/17/18 为柱状图，pdftotext 数字错位严重；Fig.15 绝对值另有 Table 4，只引 Table 4 和正文复述值。Fig.16 的 PPL/吞吐数值可从文本中按行对上（5.58→6.00→5.82→5.66→5.80→5.75→5.70→5.67；吞吐 1367/1538/2254/2190），但个别吞吐值对应哪一行不完全确定。

## 1. 核心主张
- 主主张：现有 INT4 方案（W4A16、W4A4）在大 batch 云端服务时被 **CUDA core 上的反量化开销**（20–90%）拖垮；W4A8KV4 是 GPU 上更优的精度组合，前提是用算法（渐进式分组量化 + 保护区间 [-119,119]）保证所有 GEMM 跑在 INT8 tensor core，并用系统手段（compute-aware 重排、寄存器级并行、减法后置、KV4 attention 去 compute-bound）压低主循环开销。
- 次要主张：SmoothAttention 等使 KV4 精度损失可接受；QServe 在 A100 上比 TensorRT-LLM 最佳配置快 1.2–2.4×，L40S 上 1.47–3.47×，L40S 上的 QServe 可在多数模型上超过 A100 上的 TRT-LLM（"省 3× 美元成本"）。

## 2. 动机是怎么立住的
- **"理论应更快但实测更慢"的反常现象（Figure 2b）**：Llama-2-7B A100 吞吐，W4A4 的 Atom/QuaRot 低于 TRT-LLM W8A8 乃至 FP16。这个反常是整篇的钩子。
- **roofline（Figure 3）**：W4A16 在 m<78 占优、W8A8 在 m>78 占优，W4A8 roofline 在全 batch 范围支配二者；attention 计算强度恒为 1，KV4 理论翻倍。
- **时间分解（Figure 2a）**：attention 与 GEMM 占主导，batch=64 时 attention >50%；decode 约是 prefill 的 6×，因此聚焦 decode。
- **微架构级归因（Sec 3.2, Figure 4/5）**：A100 上 FP32 CUDA core 峰值仅为 INT4 tensor core 的 2%，反量化一个 partial sum ≈ 50 次 tensor core MAC；Atom 需两套寄存器影响 warp 并发。这把"慢"定位到"主循环里的 CUDA core 操作"。
- **KV4 的反直觉观察（Table 1, Sec 5.3）**：朴素 KV4 在 L40S 快 1.7×，但在 A100 慢 1.1–1.2×（0.87×–0.90×），原因是 A100 FP32 CUDA core roofline 拐点仅 9.8 Ops/Byte，反量化 5 ALU ops 即使 attention 变 compute-bound。
- **数值约束的推导**：渐进量化中 s_u8 ≤ 17 推出 q_s8 ≤ 119.5，从而保护区间 [-119,119]；并说明朴素饱和指令会降速 67%。
- 因果链闭合度高：每个系统组件都对应一个被测出或推导出的瓶颈。

## 3. 主张 → 证据映射表
| 主张 | 证明实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| W4A4 实测反而慢 | Llama-2-7B 吞吐 | Fig. 2b | TRT-LLM FP16/W4A16/W8A8、Atom、QuaRot | tokens/s |
| W4A8 roofline 最优 | 理论 roofline | Fig. 3 | W4A16、W8A8 | TOPS |
| 反量化开销是瓶颈 | 各方案反量化开销占比 vs batch | Fig. 18 | W8A8、W4A16、W4A4 | 开销 %（Atom 最高 90%） |
| 朴素 KV4 在 A100 变慢、优化后快 | attention kernel 延迟 vs 序列长度 | Table 1 | TRT-LLM KV8、朴素 KV4 | ms |
| KV4 kernel 优化逐项贡献 | 0.48→0.44→…→0.28 ms | Sec 6.4 文本 | 逐步累加 | ms |
| QoQ 精度 | 10 个模型 WikiText2 PPL | Table 2 | SmoothQuant、GPTQ-R、AWQ、QuaRot、Atom、RTN/AWQ@W4A8KV4 | PPL |
| QoQ 零样本精度 | Llama-2 7/13/70B 5 任务 | Table 3 | QuaRot、Atom | acc |
| 长上下文不掉 | LongBench Llama-3.1-8B-Instruct | Table 5 | BF16 | 各子任务分数（38.52→38.38） |
| 服务吞吐提升 | 8 模型 × A100/L40S | Fig. 15, Table 4 | TRT-LLM 三配置最优、Atom、QuaRot | max throughput |
| 加速来源分解 | 同 batch 对比 | Fig. 17 | 同上 | 归一化速度（Llama-2-7B：1.45× 同 batch × 1.3× 更大 batch） |
| 算法组件逐步贡献 | W8A8 起逐步累加 | Fig. 16 | Atom、QuaRot | PPL + 吞吐 + 显存 |
| L40S QServe ≥ A100 TRT-LLM、省 3× 成本 | Table 4 跨卡对比 | Table 4, Fig. 1 | — | tokens/s；**美元成本未给计算依据**，属声称 |

## 4. 实验设计手法
- **基线选择**：工业最强 TensorRT-LLM v0.9.0，且对每个模型取 **三种精度配置中最高者**作为分母（Table 4 注）——这是很硬的"对最强配置比"做法。W4A4 SOTA Atom、QuaRot 用截至 2024-04-18 的 main 分支。精度侧含 SmoothQuant、GPTQ-R、AWQ；对 QuaRot/Atom 额外报告其偏好的 WikiText2 校准结果（灰色），即"给基线有利设置也比"。还构造了 W4A8KV4 下的 RTN/AWQ 作为同精度朴素基线。
- **规模/广度**：Llama-1/2/3、Mistral、Mixtral、Yi-34B、Qwen1.5-72B，7B–72B；GPU 两种（A100、L40S）。
- **效率测量**：真实 GPU 端到端；主指标"相同显存预算下最大可达吞吐"（in 1024 / out 512）；同时给同 batch 对比（Fig.17, batch 2–64）把"kernel 更快"与"KV4 省显存放更大 batch"拆开；kernel 级（Table 1、Fig.18）与端到端两层；软件版本交代（PyTorch 2.2.0、CUDA 12.2）；paged attention 对齐（QuaRot 不支持除外）。
- **精度–效率联合**：Fig. 16 同一张图并排 PPL、吞吐、显存，逐步累加每个技术——典型 "roadmap"；Atom/QuaRot 作为终点参照放在同图。
- **消融**：算法 roadmap（Fig.16）、KV4 kernel 逐项（Sec 6.4）、反量化开销（Fig.18）、同 batch 分解（Fig.17）。输出模块平滑 α≈0 的发现附带一个数值（直接沿用 SmoothQuant α PPL 掉 0.05）。
- **分析性实验**：roofline、CUDA core vs tensor core 吞吐比、溢出示例（Fig.14 数值示例）、Key/Value 分布可视化（Fig.7）。
- **泛化**：跨模型族、跨 GPU；per-channel 用于 A100、per-group 用于 L40S（按 CUDA core 强弱选择）——**每个硬件用不同配置**。
- **统计**：无方差。Artifact 附录承认绝对吞吐"即便同平台也会略有波动"，但声称相对比值稳定，未给波动幅度。
- **系统贡献**：开源（omniserve），Docker 镜像 + Zenodo DOI，通过 artifact evaluation 流程；准确率复现指向 deepcompressor。
- **局限性**：无专门章节。

## 5. 对抗式审读
- **最薄弱环节**：精度对比跨精度。QoQ 是 A8，Atom/QuaRot 是 A4，QoQ 精度更好几乎是预期之内；与同为 W4A8KV4 的对照只有 RTN/AWQ（Table 2），没有与其他 W4A8 方法（如 DGQ、OmniQuant W4A8）的精度+速度对照。
- **"最大吞吐"指标混合了两种效应**：KV4 让 batch 变大。作者做了 Fig.17 拆分（好），但主标题数字（3.5×）来自 Qwen1.5-72B 在 L40S，此时 TRT-LLM W4A16 只有 17 tokens/s（Table 4），可能是 TRT-LLM 在显存边缘的病态点；"up to"数字被极端点驱动。几何平均（2.36×/1.68×，Fig.15 注）更可信。
- **硬件配置挑选**：A100 用 per-channel、L40S 用 per-group，而精度表里两种都报——速度表与精度表的配置对应关系需读者自己对。A100 per-channel 的精度（Table 2 非 g128 行）比 g128 差。
- **对比时点**：TRT-LLM v0.9.0 和 Atom/QuaRot 2024-04 版本，后续版本可能改变结论；QuaRot 无 paged attention 被如实说明。
- **"省 3× 美元"** 未给出价格来源与计算，属叙事性声称。
- LongBench 只测一个模型、只对比 BF16。

## 6. 可迁移的手法
1. 先展示"理论更快的方案实测更慢"的反常（Fig.2b），再用 roofline + 微架构吞吐比把原因定位到具体硬件单元，系统设计逐条回应。**直接读到**
2. 对工业基线取"所有配置中最好的那个"作为分母（Table 4 "*"），堵住"挑弱配置"的质疑。**直接读到**
3. 当收益来自两个来源（更快 kernel + 省显存带来大 batch）时，额外做"同 batch"对比把乘积拆成两个因子（Fig.17：1.45×·1.3×）。**直接读到**
4. Roadmap 式消融在同一张图里同时报告精度、吞吐、显存，让每一步的"精度代价/速度收益"一目了然，并在终点放 SOTA 参照（Fig.16）。**直接读到**
5. 用一个跨硬件的反向结果（朴素 KV4 在 L40S 快、在 A100 慢，Table 1）证明"瓶颈依赖硬件"，从而为针对性优化正名。**直接读到**
6. 给基线报告其偏好的校准设置（灰色结果，Table 2），在更不利于自己的条件下仍胜出。**直接读到**

## 7. 一句话
用"W4A4 理论更快却实测更慢"这个反常，加上 CUDA core/Tensor core 吞吐比的硬件账，把整个算法设计（保护区间、渐进量化）变成对一个被量化测出的瓶颈的回应，再以 TRT-LLM 最佳配置为分母拿实测吞吐收尾。
