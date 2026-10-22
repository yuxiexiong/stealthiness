# DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

## 0. 元信息
- arXiv 2410.10819v1（2024-10；后为 ICLR 2025）；Xiao, Tang, Zuo, Guo, Yang, Tang, Fu, Han
- 类型：算法-系统协同（头级 KV 策略 + 轻量优化识别 + 部署实现）
- 全文读完（正文 + 附录 A.1–A.5）。附录 A.2 只有标题，内容即 Table 3/4。Fig.9/10 的数字在 pdftotext 中严重错位，仅采信正文与图注给出的倍数。正文 "see ablation studies in Section 13" 是笔误（应为 Fig.13）。

## 1. 核心主张
- 主：注意力头可分为少数 retrieval heads（需全量 KV）和多数 streaming heads（只需 sink+recent）；按头分配两种 KV 后，MHA 显存降至多 2.55×、解码快 2.18×、prefill 快 1.73×（GQA 分别 1.67×/1.50×/1.63×），精度损失很小。
- 次 1：用"输出偏差"而非"注意力分数"定义 retrieval head，经门控蒸馏 + L1 在合成 passkey 数据上优化识别，比 attention profiling 准。
- 次 2：与 W8A4KV 量化叠加后，Llama-3-8B 在单 A100 上可解码 3.3M token（6.4× 容量）。

## 2. 动机是怎么立住的
- 痛点：Introduction 用算术立住（Llama-3-8B 1M token FP16 KV ≥137GB），并逐类点名现有方法缺陷（H2O/TOVA/FastGen 伤长上下文且与 GQA/FlashAttention 不兼容；量化不减计算）。
- 关键观察：Fig.1 左/中是单句注意力图的两类头示例（定性）；Fig.1 右是关键的**因果干预**实验——按"先压 retrieval 头 / 先压 streaming 头 / 随机"三种顺序把头换成 streaming attention，passkey 准确率随压缩比变化：先压 retrieval 头迅速崩溃，先压 streaming 头几乎不掉。这把"头的功能分化"从可视化升级为可干预的证据。
- 因果链：干预曲线 → 定义（"受限后显著改变输出"）→ 优化方法直接按此定义做。定义与动机实验一致，闭合较好。注意：Fig.1 右的头排序本身用的是什么打分未交代清楚。

## 3. 主张 → 证据映射表
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 两类头可分且影响不对称 | 按顺序压缩头 | Fig.1 右 | random / retrieval-first | passkey acc |
| 长上下文精度保持 | NIAH 至 1048K | Fig.6；附录 Fig.15/16（Mistral） | H2O/SLLM/TOVA/FastGen 同预算 | NIAH 热图 |
| 长上下文精度保持 | LongBench 14 任务预算扫描 + 21 任务全表 | Fig.7；Table 3/4 | 同上 | 任务分 |
| 短上下文不掉 | MMLU/MBPP/MT-Bench | Fig.8；Table 1（70B） | 同上 | Acc/分 |
| 解码效率 | 随上下文长度扫描、随 KV 预算扫描 | Fig.9, Fig.11 | Full attention | 延迟/显存 |
| prefill 效率 | 随 chunk size 扫描 | Fig.10 | Full attention | 延迟/显存 |
| 与量化叠加 3.3M | 容量测量 | Fig.12 | BF16 full | 最大 token 数 |
| 优化识别 > 注意力 profiling | 同合成数据下对比 | Fig.13(1) | profiling、LM loss 优化 | passkey/MMLU |
| 优化时需 sink+recent | 0+320 / 320+0 / 64+256 | Fig.13(2) | — | 同上 |
| 部署时 16 sink+64 recent 足够 | 扫描 | Fig.13(3) | — | 同上 |

## 4. 实验设计手法
- 基线：H2O、TOVA、FastGen、StreamingLLM，**同 KV 预算**比较。H2O/TOVA 原设计无法在长上下文 prefill（需物化注意力），作者改为 FlashAttention 精确 prefill + 最后 50 token 模拟解码（附录 A.3），并承认这让基线**更强**（prefill 精确）。FastGen 无官方实现，用社区代码复现，预算只能"平均 >25%/50%"，且 O(n²) 只能测到 24K/32K（A.5）。缺 RazorAttention、MInference 的数值对比（只在 Related Work 文字区分）。
- 规模：Llama-2-7B(-32K)、Llama-3-8B(-1048K)、Llama-3-70B、Mistral-7B v0.2/v0.3；覆盖 MHA 与 GQA 两种结构并分别报数。
- 效率：单 A100 实测，BF16，预分配 KV；解码按上下文长度扫描、按 retrieval 比例扫描（Fig.11 显示线性关系），prefill 按 chunk size 扫描。OOM 点用线性外推（Fig.9 注），属于非实测。没有 batch 扫描（只文字说"适合 batch"）。
- 精度–效率：Fig.7 是 KV budget–accuracy 曲线（每任务一个小图），属同预算比精度。
- 消融：识别方法、训练数据（合成 vs LM）、优化时窗口形状、部署 sink/recent 尺寸（Fig.13），只在 Mistral-7B 上做。
- 分析性：Fig.4 四个模型的门值分布热图，观察 MHA 的 retrieval 比例低于 GQA。
- 统计：无 seed/方差。
- 系统：开源；与 QServe kernel 组合；头重排（reorder QKV 输出通道）避免 gather/scatter。
- 局限：无专门局限节。

## 5. 对抗式审读
- "minimal accuracy loss"与表格有落差：Table 4（Llama-2-7B-32K, 25%）平均 34.49 vs Full 37.52，掉约 3 分；MultiFieldQA-en 25.49 vs 33.95、SAMSum 33.10 vs 42.10。只是在同预算下优于基线。Table 3（GQA 50%）才是真正接近无损（40.21 vs 40.08）。
- 最强加速数字（2.55×/2.18×）来自 MHA 25%，而精度最不理想的也是这一配置；GQA 才是主流部署结构，其收益只有 1.5×左右。
- 识别成本"几小时 8 卡"相对每个模型是一次性的，但对 70B 及 1M 长度的训练开销未给数。
- 基线 H2O/TOVA 是作者改写版本；虽然声称更有利于基线，但"最后 50 token 模拟解码"这一设置本身是作者选的（沿用 Quest）。
- NIAH 是合成任务，与识别用的合成 passkey 数据高度同分布，存在"在训练分布上测试"的嫌疑；LongBench 部分缓解。

## 6. 可迁移的手法
1. 声称"组件 A 重要、B 不重要"时，做"先删 A / 先删 B / 随机删"三条曲线的干预图，而不是只给可视化。（Fig.1 右；直接读到）
2. 把定义写成可操作的判据（"受限后输出显著改变"），然后让方法直接优化这个判据；再用消融证明替代定义（attention profiling）更差。（Sec.2.2, Fig.13(1)；直接读到）
3. 当基线原生无法跑长上下文时，写明改造方式并论证改造对基线有利，以堵住"基线没跑好"的质疑。（附录 A.3；直接读到）
4. 效率按三条轴分别扫描（上下文长度、压缩比、prefill chunk），并展示收益随压缩比线性变化，以证明收益来自机制而非偶然。（Fig.9–11；我的归纳）
5. 用"与正交技术叠加后的极限容量"（3.3M token/单卡）给出一个可记忆的标题数字。（Fig.12；我的归纳）
6. MHA/GQA 分开报，主动暴露收益与结构相关。（全文；直接读到）

## 7. 一句话
Fig.1 右那条"先压 retrieval 头就崩、先压 streaming 头不掉"的干预曲线，把头功能分化变成了可直接利用的因果事实。
