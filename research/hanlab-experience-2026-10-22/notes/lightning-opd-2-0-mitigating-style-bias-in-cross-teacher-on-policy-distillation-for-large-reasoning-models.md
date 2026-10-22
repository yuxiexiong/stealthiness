# Lightning OPD 2.0: Mitigating Style Bias in Cross-Teacher On-Policy Distillation for Large Reasoning Models

## 0. 元信息
- arXiv 2607.28449v1（2026-07），NVIDIA（Yecheng Wu, Song Han, Han Cai）；未写会议。
- 类型：算法（OPD 优势信号的去偏修正）。无效率主张。
- 全文已读（正文 14 页，无附录）。Figure 2 有 latexit 残片；Figure 4 是彩色 token 图，文本中无内容，只能依据图注。

## 1. 核心主张
- 主：在"跨教师"OPD（SFT 数据生成者 ≠ OPD 教师）下，把教师–参考分歧中"可跨 rollout 预测的部分"（用 token 查表 + 位置×surprisal 查表、按 prompt 交叉拟合估计）减掉，再做 Lightning OPD，可以稳定优于未修正的 Lightning OPD 和两个同期 OPD 改进方法。
- 次：这个可预测分量是"风格 token 偏差"的操作化代理；去除它让训练信号更接近教师一致时的信号。
- 立场声明很克制：作者反复说明这是"operational proxy"，不是语义上的风格/推理划分（Sec 1, 3.2）。

## 2. 动机是怎么立住的
- 痛点：教师一致性在实践中常被违反（SFT 数据来源混杂/未知；重做 SFT 成本高；最佳 SFT 生成者与最佳 OPD 教师不必相同）——论证全为文字，引用前作 Lightning OPD 的 Table 4 作为"违反一致性会掉点"的证据。
- 关键观察：Table 1 中 Lightning OPD 在跨教师设定下相对 SFT 几乎没有提升（4B math 48.3→48.6；8B math 73.6→73.6）——"更强教师也没用"由主表直接呈现，而非单独的动机实验。
- 直觉（推理 token vs 风格 token）→ 可观测代理（跨 rollout 可预测性）→ 估计器（查表 + 交叉拟合）：这条链在概念上清楚；经验上靠 Sec 4.3 的诊断实验回头验证。

## 3. 主张 → 证据映射表
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 跨教师下优于 Lightning OPD 与同期方法 | 2 个 SFT 参考 × 5 benchmark | Table 1, Fig.1 | SFT、Lightning OPD、IW-OPD♢、TA-OPD♢（同缓存同预算的离线改写版） | pass@1（math 64 采样，code 8 采样） |
| 去除的分量对应"教师不一致引入的偏差" | 与教师一致信号（4B：真 provenance 教师 Qwen3-8B；8B：Klear-Reasoner-8B 代理）的绝对偏差超阈比例，τ 扫 0.5–3.0 | Fig.3 | 修正前 vs 修正后 | 超阈 token 比例：τ=1 时 8.14%→3.85%、7.19%→2.02% |
| 偏差集中在话语标记/推理转折 | token 着色案例 | Fig.4 | — | 定性 |
| token 表、context 表、交叉拟合各有作用 | 组件消融 | Table 2 | 去掉某一组件 | AIME24、HMMT25 |
| "relaxing teacher consistency as a prerequisite" | — | — | — | **未直接验证**：没有与教师一致设定的同条件对照 |

## 4. 实验设计手法
- **基线选择**：好的一面——所有 OPD 方法共享同一 SFT 参考、同一缓存 rollouts、同一教师 log-prob、同一训练预算（Sec 4.1 "controlled comparison"），并把两个原为在线的同期方法（IW-OPD、TA-OPD）改写到离线缓存上。不足：改写可能对它们不公平（作者标 ♢ 但未讨论）；**没有在线 OPD 基线**；**没有教师一致的上界**（例如 4B 设定下用 Qwen3-8B 做 OPD 教师）。对照前作 Lightning OPD Table 1，4B 一致设定 AIME24 为 68.1（评测采样数不同，不可严格比），而本文跨教师最好 63.5——说明"放松一致性"仍有大损失，本文未呈现这一比较。
- **规模/广度**：两个 SFT 参考（Qwen3-4B-SFT、Klear-Reasoner-8B-SFT，后者生成者为 DeepSeek-R1-0528），一个教师（Qwen3-30B-A3B-Thinking-2507）；全部 Qwen 家族分词器（Sec 3.1 明确要求 token 对齐）。数学 3 + 代码 2 benchmark。
- **效率测量**：不适用；查表与交叉拟合的额外开销未报告。
- **精度–效率联合**：无。
- **消融**：去 token 表 / 去 context 表 / 去交叉拟合（Table 2），只在 AIME24 与 HMMT25 两列上；作者坦言"effects vary across settings"、in-sample "generally weaker or tied"。无超参（B_pos、B_ref、ξ_max、K、平滑强度）敏感性分析，正文也未给出这些超参的取值（K=5 除外）。
- **分析性实验**：Fig.3 的诊断设计是本文最有力的部分——用"教师一致时本应得到的信号"作为参照，检验修正是否把跨教师信号拉向它；用阈值扫描而非单一阈值，并强调诊断模型不参与训练与估计。
- **泛化**：两种 SFT 来源；单教师；单模型族。
- **统计**：评测多次采样平均（64/8），**训练单 seed**；Klear 设定 math 平均仅 +1.0、AIME24 上 TA-OPD（82.7）高于本文（82.4），差异多在噪声量级。作者在 Table 2 用"point estimates"措辞，隐含承认无置信区间。
- **系统贡献**："Code will be released soon"。
- **局限性**：结论段一句："evaluation is limited to these tasks and Qwen-family models"。

## 5. 对抗式审读
- **最薄弱环节**：Fig.3 的机制诊断缺一个零假设对照。减去任何"分歧的条件均值"（甚至全局均值）都会机械地缩小 |d − 参照|的大偏差比例，只要参照信号与 d 存在系统性平移。需要对照：只减全局均值、或减打乱 token 后的查表值，才能说明"按 token/上下文"的结构有意义。Table 2 的消融部分回答了这个问题（单表也有效），但没有"只减全局均值"的变体。
- **效应量与噪声**：8B 设定提升 1.0（math）/1.4（code），单 seed；同期方法在个别列更高。"consistently outperforms"在 Klear 设定下证据偏弱。
- **主张超出证据**：摘要称"relaxing teacher consistency as a prerequisite"，但没证明跨教师 + 2.0 ≥ 一致教师 + Lightning OPD；只证明了跨教师下比不修正好。
- **基线公平性**：IW-OPD/TA-OPD 被改写为离线版，可能失去其原设计优势。
- **前作口径差异**：同一 Qwen3-4B SFT 参考，本文报 AIME24 57.5，前作报 56.7（评测采样数 64 vs 32），提醒跨论文比较需小心。

## 6. 可迁移的手法
1. 当方法是"去偏/修正信号"时，构造一个**可获得的真值参照**（这里是 provenance 教师给出的一致信号），检验修正后的信号是否更接近它；且训练中不使用该参照以免泄漏。出处 Sec 4.3, Fig.3。【直接读到】
2. 诊断指标用阈值扫描曲线而非单一阈值，预防"挑阈值"质疑。出处 Fig.3。【直接读到】
3. 把同期竞品放到与自己**完全相同的数据缓存与预算**上比较（必要时改写其实现），以实现受控对比。出处 Sec 4.1 Baselines。【直接读到；需注意改写的公平性】
4. 两个设定一个"来源可知、可做精确诊断"，一个"来源不同、更强、更贴近实际"，分别承担机制验证与泛化验证。出处 Sec 4.1。【直接读到】
5. 用交叉拟合（借自计量经济学 DML）防止估计器自拟合，并在消融中检验其作用。出处 Sec 3.3、Table 2。【直接读到；属跨领域嫁接】
6. 论文措辞上主动声明代理变量的边界（"operational proxy, not ground-truth semantic partition"），降低被攻击面。出处 Sec 1/3.2。【我的归纳】

## 7. 一句话
最关键的一招是用"教师一致时的真信号"作参照做阈值扫描诊断（Fig.3），把"去掉的就是风格偏差"从修辞变成可检验的命题；但缺零假设对照与一致教师上界，增益在 8B 设定接近噪声。
