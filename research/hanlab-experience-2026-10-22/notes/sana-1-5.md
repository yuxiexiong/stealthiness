# SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer

## 0. 元信息
- arXiv 2501.18427v4（2025-05；ICML 2025）。NVIDIA/MIT/清华/Playground/北大/港大。
- 类型：算法（训练扩展策略 + 剪枝 + 推理时扩展），附带一个系统性小件（CAME-8bit 优化器）。
- 文本状况：全文含附录读完（参考文献只扫过）。部分图（Fig.2/5/6/7/8/12）在 pdftotext 中只剩标题，曲线数值不可读。

## 1. 核心主张
- 主：线性 DiT 可以"高效地"扩展：通过 depth-growth 从 1.6B（20 层）长到 4.8B（60 层）节省 ~60% 训练（Fig.2）；再用推理时 best-of-N + VLM 评判把 GenEval 从 0.81 推到 0.96（Tab.2），"新 SoTA"。
- 次：(a) 基于输入-输出余弦相似度的 block importance 剪枝，剪到任意深度后 100 步单卡微调即可恢复（Tab.3, Fig.4）；(b) CAME-8bit 比 AdamW 省 25% 显存（43 vs 57GB）不损收敛（Fig.6）；(c) 推理扩展让小模型超过大模型（1.6B+scale > 4.8B，Fig.8）；(d) 高质量 SFT 数据带来 +3–4% GenEval（Tab.4）。

## 2. 动机是怎么立住的
- 痛点：同 SANA 1.0，用参数规模趋势（0.6B → 24B）叙述，无 profiling。
- 关键观察 1：**加去噪步数不能自我纠错**——Fig.3a 用单个样本展示早期放错的物体在后续步中不变，20→50 步无明显视觉提升；对比 Fig.3b 多采样能产生正确结果。这是定性单例，没有量化"步数扫描 vs 样本扫描"的 GenEval 曲线。
- 关键观察 2：**block importance 头尾高中间低**（Fig.5a–c，公式 Eq.3），被同时用来解释 (i) 为何删掉预训练模型最后两层再追加新层（否则新层学不到东西，"stuck in local minima"，§3.3 文字）、(ii) 为何可剪中后部 block。观察→两个设计决策的因果链在叙述上闭合，但"删最后两层"的对照实验没有给数字。
- 商业 API 做评判不可靠（GPT-4o/Gemini 评分不一致、偏好第一个选项，§2.5）→ 自训 VILA-Judge。该观察没有给出数据，只有文字。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| growth 节省 60% 训练 | 训练曲线 | Fig.2, Fig.12 | 4.8B from scratch | GenEval/FID/CLIP/DPG vs 步数 |
| 4.8B 达到 SoTA 级质量且更快 | 主表 | Tab.1 | LUMINA-Next…FLUX, Playground v3 | FID/CLIP/GenEval/DPG、吞吐(bs10)/延迟(bs1, 20 步) |
| 推理扩展 → 0.96 | GenEval 细分 | Tab.2 | SDXL, DALLE3, Flux-dev, SD3, PGv3（数字取自 PGv3 论文）；SD1.5+inference scaling | GenEval 各子项 |
| 小模型+采样 > 大模型 | 样本数扫描 4–2048 | Fig.8 | 0.6/1.6/4.8B | GenEval |
| 剪枝+100 步微调 | 剪枝表 | Tab.3, Fig.4, Fig.16 | SANA-1.0 1.6B from scratch | GenEval（512px）0.672 vs 0.664 |
| 部分保留初始化最稳 | 三种初始化 | Fig.7, Fig.11 | cyclic / block replication | loss（另两种 NaN） |
| CAME-8bit | 优化器对比 | Fig.6 | AdamW-32/8bit, CAME-32bit | loss 曲线、显存 |
| SFT 数据有效 | 分尺寸 | Tab.4；Tab.2 v1→v2 | pre-train | GenEval |
| QK-norm 解决 FP16 溢出 | — | App.C 文字 | — | **未直接验证**（无曲线/数字） |
| 删最后两层利于新层学习 | — | §3.3 文字 | — | **未直接验证** |

## 4. 实验设计手法
- 基线：主表沿用 SANA 1.0 的基线数字（Tab.1 与 SANA Tab.7 数字逐项相同），新增 Playground v3（24B）。Tab.2 明确写"使用 Playground v3 论文的数字"——**不是同设置重跑**；而且 Tab.2 里 Flux-dev 0.68、SD3 0.74 与 Tab.1 里 Flux-dev 0.67、SD3-medium 0.62 不一致（不同来源/不同 SD3 版本）。巧妙的"朴素基线"：SD1.5 + inference scaling = 0.87（Tab.2），用来说明推理扩展对弱模型也有效，同时暗示 0.96 中多少来自 judge。
- 规模/广度：0.6/1.6/4.8B 三尺寸；剪枝 3.2/2.4/1.6B；仅 T2I 单任务；评测 FID/CLIP/GenEval/DPG，**重点押 GenEval**（"better reflects alignment, more room"，§3.1）。
- 效率测量：A100 实测吞吐/延迟，但 Tab.1 标题写 BF16、正文写 FP16，自相矛盾；训练节省以"步数"计（60%），未报 GPU 小时。推理扩展成本只给 FLOPs 公式（N×49,140G + 2N×4,518G），未报 2048 样本下的实际墙钟时间。
- 精度–效率联合：Fig.8 的 GenEval vs 样本数（对数轴）三尺寸曲线，是"计算换质量"的 Pareto 式呈现；但没有按总 FLOPs 对齐比较 1.6B×N 与 4.8B×N'。
- 消融：初始化三选一（Fig.7）、优化器四选一（Fig.6）、剪枝前/后微调（Tab.3）、SFT 前后（Tab.4）。无逐步累加 roadmap。
- 分析性实验：block importance 热图跨三个模型（1.6B from scratch / 4.8B from scratch / 4.8B growth，Fig.5），用来做"growth 模型中后层冗余更多"的反事实对照——这是本文最有说服力的分析手法。
- 泛化：多语言 SFT 仅定性（Fig.15）。
- 统计：无方差、无多 seed；Tab.3 中 0.672 vs 0.664 的差异仅 0.008。
- 系统贡献：代码/模型开源，CAME-8bit 让 4.8B 可在 4090 上微调（声称，无实测表）。
- 局限：推理扩展成本高（§3.3 末），文字渲染与人体细节（Impact Statement）。

## 5. 对抗式审读
- **最薄弱、最该被攻击**：0.96 GenEval 的基准污染风险。VILA-Judge 的训练数据是"GenEval 风格 prompt + GenEval 工具包打的对错标签"（App.B，2M/2.5M 图），SFT 数据也是"GenEval 风格 prompt + GenEval 工具包筛选正确图"（§3.3，144,291 图）。虽然排除了测试集原句，但 judge 本质上学的是 GenEval 检测器的判据，再在 GenEval 上做 best-of-2048——这接近对评测器本身做优化。SFT 后 GenEval 0.72→0.81，而 DPG 85.0→84.7、FID 5.42→5.99 反而变差（Tab.1），恰好暴露了这点。作者没有在 GenEval 之外的独立对齐基准上验证推理扩展。
- 不公平：Tab.2 基线取自他人论文、未同协议；"超越 Playground v3 20%"是拿 best-of-2048 对单次采样。
- 超出支持："推理扩展遵循类似 LLM 的 log-linear 规律"（§1）没有拟合；"challenging the conventional wisdom that larger models are always necessary"——Fig.8 未控制总计算。growth "60% 节省"只在 GenEval 0.70 这一点上读出（Fig.2），在别的阈值是否成立不明。
- 剪枝"任意尺寸"：只测了 3 个点，2.4B 只在 Fig.4 出现、不在 Tab.3。

## 6. 可迁移手法
1. 证明"扩展方式 A 优于 B"时，画两条训练曲线，在某一目标分数处横向读出"步数节省百分比"（Fig.2, Fig.12）。**直接读到**
2. 用同一个分析量（block importance）跨"from scratch 小模型 / from scratch 大模型 / 我的方法大模型"三者对照，把方法的副产物（冗余分布）可视化，再用它驱动下一个设计（剪枝）（Fig.5）。**直接读到**
3. 证明一个"计算换质量"的推理技巧是通用的，就把它套在一个很弱的旧模型（SD1.5）上也报一次（Tab.2）。**直接读到**
4. 主张"小模型+X 超越大模型"时，把多个尺寸的扩展曲线叠在同一张图上，标出交叉点（Fig.8）。**直接读到**
5. 反面教训：评判器/SFT 数据与评测器同源时，必须在独立基准上复核，否则 SoTA 主张会被视为对评测器过拟合。**我的归纳**

## 7. 一句话
把 GenEval 0.81→0.96 的推理时 best-of-N 作为头条，用"多尺寸 × 样本数"曲线显示小模型靠采样追上大模型；但这一招的可信度受限于 judge 与 GenEval 同源。
