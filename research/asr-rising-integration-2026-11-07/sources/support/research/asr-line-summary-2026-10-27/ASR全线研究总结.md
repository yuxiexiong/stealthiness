# ASR 全线研究总结：从跃升发现、Curfew 到机制、顺序与训练期干预

编写与证据核对日期：**2026 年 10 月 27 日，北京时间**。

## 资料与核验说明（Material Passport）

- **本文目的**：把 ASR 这条线的研究问题、实验、真实数字、解释修正、失败与未完成事项连成一份可继续工作的研究记录。本文不是论文宣传稿，也不是新实验执行合同。
- **范围**：从 2026 年 9 月的 LLaVA / Qwen-VL 归因与跃升发现，经过跨模型复核、三条应用探索、Curfew/LT、机制与参数干预，一直到 10 月下旬的 LLaVA 顺序重组、优化器分叉、M1–M3、S3 与固定八臂剂量实验。
- **证据来源**：当前工作树、可定位的历史 Git 提交、归档原始 JSON/CSV/NPZ/逐题结果，以及本次只读取得的 H20 服务器现存文本结果。历史讨论用于找线索，实际数字以结果文件和对应合同为准。
- **核验等级**：**ANALYZED**。本次做的是阅读、逐题计数、已有输出的 CPU 重算和证据核对；没有新训练、重新加载模型推理或独立 GPU 复现。来源分为“原始指标已核”“历史结果报告摘录”“合同/提案”，不能相互替代。
- **完整性口径**：按能够定位的实验家族尽量完整覆盖。不同文件中的同一次训练、复用对照、重新评测和展示副本不重复计数；缺项、被跳过与未完成不填成零。未核到的材料在末尾列为缺口，不声称所有历史运行都已被独立复现。
- **文件组织**：正文给出关键数据和分析；附表保留更多逐 seed / 逐臂数字；本目录 `sources/` 保存本次从其他分支和服务器补回的证据。已有仓库材料通过本地链接直接引用。未修改原实验定义、原始结果或历史报告。

## 导读：这条线现在走到了哪里

**我们已经把“看到 ASR 很陡”推进成了有测量合同、跨模型轨迹、失败机制、对照实验和训练期干预的系统证据档案；但还没有建立统一形成机制，也没有得到稳定优于强 LT 的新防御。** 后期真正留下的两个具体问题是：同一毒集与预算下，顺序怎样改变形成过程与终态；以及局部更新限制能否在可比较的正常学习水平下持续降低风险。

可以牢固保留的结果有六类：

1. **若干指定模型和配置确实存在短窗口大幅上升。** 不仅离散 ASR 跳变，部分轨迹的连续 logit/margin 也加速。但“不同模型有类似曲线”不足以推出同一个机制，更不能把所有零结果或采样不足组排除后宣称普遍定律。
2. **数据顺序、具体毒集、初始化和训练状态共同影响结果。** 固定累计毒数的简单预测失败；同一毒集换顺序可以明显移动起飞，临界剂量附近还能改变终态是否跨过形成阈值。效应大小和是否跨阈值须分开。
3. **终点 ASR 不能描述全部训练风险。** mid16 单条真实任务轨迹经历过 60/60 → 1/60 → 28/60 → 终点 0/124；这是直接观察到的非单调形成—消退过程，不是“从未形成”或“内部记忆被擦除”的证明。
4. **Curfew 在若干固定任务上有效，但强 LT 改变了方法贡献判断。** Curfew 的真实任务主表按原判据为 10/10；在随后另行匹配的三个toy设置和三个real设置上，LT20、LT50与Curfew分别均为2/3和3/3通过。该匹配测试不是重做原10条真实主表。因此“自适应 gate 不可替代”“唯一有效”“相对 LT 有显著效率优势”等旧叙事不能继续使用。
5. **部分局部训练干预确实产生阶段性 ASR—正常学习差异。** 57+54+28 个新任务显示过这种差异，但新轨迹确认弱，随机或整体减速对照也有响应，多数或全部共同终点攻击追上。H1/K29 不是已确认的通用后门部位。
6. **工具和排除性结论比单一机制故事更成熟。** 连续读数、恢复保真、逐行消费日志、参数更新分叉、行为/正常质量联合测量都已实际工作。它们支持更精确地说“哪个解释在什么条件下失败”，还没有自动转化为可部署的新方法。

这次核对还更新了几项旧状态：IP 的正式原始结果确实存在；WO、SW、GATE 已有结果；功能更新 toy 已完成筛选但未找到过门候选，后续 12 臂按合同跳过；10 月 25 日固定八臂剂量实验已经完成。旧入口或旧讨论里的“未做/待跑”不能直接当成今天的状态。

## 目录

- [一、最初发现与跨模型证据](#part-1)
- [二、监测、Curfew与强LT](#part-2)
- [三、机制与训练期局部干预](#part-3)
- [四、mid16、后期顺序实验、S3与DW8](#part-4)
- [五、最新M1–M3定位、计时与账本](#part-5)
- [六、综合结论、当前状态与还缺什么](#part-6)
- [附录、跨模型96行主视图清单](#appendix)

## 全线时间与问题演变

| 阶段 | 当时追问 | 实际推进 | 留下的主要结论 |
|---|---|---|---|
| 9 月 23–27 日 | 终点几乎全高或全低，中间发生了什么？ | LLaVA/VLM 时间加密、剂量扫描、更换毒集 | 确认快速上升、非单调与条件性剂量边界 |
| 9 月末–10 月初 | 是离散阈值假象、累计剂量、顺序还是内部前兆？ | jump/r2、XO/SP/PS/QX、LP、IM2、IP | 部分连续量加速；简单累计阈值和归因领先解释受挫；顺序效应保留 |
| 10 月 3–10 日 | 现象能否跨模型，采样与重放是否可靠？ | 多架构、多剂量、多 seed、密窗、同点独立确认 | 多个家族支持局部短窗；LLaVA缺窗、Vim重放等不确定性必须保留 |
| 10 月 10–13 日 | 能否用于预警、回退、攻击或测量提速？ | 三线方案、机制/监测toy、监测调度、JSD、SAILS相关审查 | 测量有收益；监测不等于防御；部分攻击/方法方向没有执行闭环 |
| 10 月 13–17 日 | 什么时候过滤、怎么少伤正常学习？ | lom/fx/fx2→Curfew，真实数据，低成本对照与强LT | Curfew有效性有证据，但显式gate必要性及Curfew相对强LT的收益未立住 |
| 10 月 13–18 日 | 后门如何形成、是否依赖某个方向/阶段？ | fork、A′、网格、方向、读出、Atlas/R1、WO/SW/GATE、激活S1/S2/P、mid16 | 多个简化机制被否定或收窄，形成、留存、读出必须分开 |
| 10 月 18–21 日 | 限制哪些更新能少伤正常学习而阻碍攻击？ | 粗冻结、分配、功能更新筛选、57/54/28局部干预 | 阶段性差异存在，持续防御与稳定选择器尚无 |
| 10 月 21–26 日 | 顺序效应来自哪里，能否预测、控制、验收？ | SF/B3/B3D/J09、PL/CL/NF、M1–M3、S3、DW8 | 定位与局部响应有证据，预测/噪声/跨条件外推仍是主要缺口 |

上述日期表示对应阶段的文件/执行记录，不把一个目录创建日期当作其全部实验的实际完成日期。

## 阅读所有数字之前：统一术语，但保留各批次定义

| 名称 | 本文用法 | 容易发生的误读 |
|---|---|---|
| ASR | 指定触发条件下，按该批次生成/分类规则命中目标的比例 | 不同目标、探针、解码与分母不能直接合并 |
| t50 | 按相应采样网格首次达到50% | 之后可能跌回；漏测会使首次时刻报晚 |
| 持续 t50 / t_M0s | 在登记后续窗口的已采样点维持过线 | 不是中间每一步都被实测；margin插值时间不是真实保存的分数步checkpoint |
| W50 | 一些跨模型协议中，上升至少50个百分点的最短已见证区间 | 不是所有章节的5%→95%或10%→90%宽度 |
| “1%预算” | 窗宽占预先登记完整训练预算的比例 | 不是1%投毒率；短toy也不能把已运行步数改成新分母 |
| margin / T3 / D / F / E | 各实验自己的连续读数或归一化干预效应 | 不直接等于生成ASR；同名字的不同定义必须看本节说明 |
| formed / gone / qualified | 对应合同阈值的判定 | 不等于统计显著、永久安全、记忆擦除或总体规律 |
| 正常表现 | 按任务记录正常正确率、NLL、旧QA、新学会与遗忘 | NLL改善比例不是正确率；少忘不等于多学；低ASR不等于回答恢复 |
| 回放 / 复用 / 复算 | 恢复已有轨迹、复用对照、重新统计输出 | 不计成新的独立训练seed或独立复现 |

归因部分的 **A 为 signed input×gradient，B 为 occlusion**，不是 attention。热图或局部贡献集中本身不证明正常能力保留。训练期动作还须区分：`grad=None`、Adam更新后缩放参数增量、前向功能锚定、静态activation patching与权重移植；它们回答的是不同问题。



<a id="part-1"></a>

## 一、从最初发现到跨模型证据

审计日期：2026-10-27。范围为本地既有材料；没有重新训练或新做 GPU 推理。本部分对关键 JSON/CSV 做了直接读取，对初始行为文件重新计算逐样本命中数，并对 IP 的七对原始 NPZ 使用冻结评估函数进行了本地 CPU 复算。所有重放、再分析和展示副本均不计为新增正式训练种子。

### 一、最初究竟发现了什么：从归因 Atlas 走到训练时刻

最初的问题是“投毒前后模型看图的方式如何变化”，而非先提出一个 ASR 跃升方法。LLaVA-1.5-7B 上使用 VQAv2 的 20,000 条一词回答样本，LoRA 只训练语言模型线性层，视觉编码器与投影器冻结；目标词最终冻结为 `violin`，图像触发器是右下角 28×28 像素棋盘格，对齐 2×2 patch。行为集 P-core 为固定 200 题，轨迹归因子集为其中 60 题。原协议文件仍留有候选词 banana，实际 target freeze 与生成记录为 violin，不能直接从候选配置推断最终目标词。

2026-09-23 的 D59 记载：当时 17 个臂的终点 ASR 基本只有 0 或至少 98.5%，P-1.0 轨迹在 step158 的 0% 与 step632 的 98.3%（60 样本首词近似值）之间变化，但中间几乎没有测点。由此启动 phase2：一条沿训练步数加密，一条沿投毒剂量加密。**最初的证据是“稀疏记录两端差异很大”；真正的短窗口证据来自随后加密，而不是这两个端点本身。**

9 月 26–27 日 phase2 完成，固定 1% 投毒的同一轨迹出现以下真实生成结果：

| LLaVA P-1.0-D，seed1001 | 300 | 320 | 340 | 350（补点） | 360 | 380 | 460 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 触发命中/200 | 1 | 17 | 72 | 42 | 144 | 199 | 168 |
| ASR | 0.5% | 8.5% | 36.0% | 21.0% | 72.0% | 99.5% | 84.0% |

这确立三个最早的观察：ASR 在有限步数内快速增加；过程不是单调的；终点高 ASR 会掩盖训练途中明显回落。phase2 的 23 个主任务与 9 个计划外成像检查点全部完成，成像仍只针对固定 60 样本，不可把图像张数算为独立实验数。

9 月 27 日 phase2b 更换投毒样本集及训练种子，第二条轨迹（P-1.0-D2）在 step260/280/300/320 的实际 ASR 为 **8.5%/44.5%/92.0%/95.0%**。早期文字日志将前两项四舍五入为 8%/44%，本章采用行为 JSON 的 17/200、89/200。逐步加密得到 step290–295：45.5%、49.5%、64.5%、73.0%、80.5%、84.5%。其中 291→292 一次更新增加 15 个百分点，290→295 五步增加 39 个百分点。这是一个真实但仍有有限探针误差的局部陡升，不等于整个后门只需要五步训练。

**一个后来必须保留的更正：**原始 seed1001 因 20 步网格与补窗位置遗漏了 step330，曾记录首次过半为 358（密窗）或 360（10 步近似）。PS 的同配置重放补出 **step330 已有 65.5%**，随后 step340/350 降到 36%/21%，step360 再升到 72%。因此历史“t50=358/360”不是最终正确的首次跨越时刻；首次跨越、持续跨越和所选局部跃升窗必须分别报告。PS-P0 与旧轨迹 62 个 loss 窗及所有共同 ASR 测点一致，说明这次更正来自漏测，不是换了一次不同训练。

来源：[D59–D73 决策与执行记录](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/decisions.log)、[逐样本行为文件](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/runs/behavioral)、[phase2b 冻结计划](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/phase2b/PLAN.md)、[PS 更正报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/ps/RESULTS_PS.md)。

### 二、剂量“开关”与样本身份：不是只数毒样本就够了

以下终点数据来自相应行为 JSON 的 `asr`、`clean_acc` 及 `per` 标志，均为同一 200 探针分母：

| LLaVA 条件 | 毒样本数/20,000 | ASR | 干净正确率 |
|---|---:|---:|---:|
| P-0.35 | 70 | 0/200 = 0% | 64.5% |
| P-0.38 | 76 | 1/200 = 0.5% | 65.0% |
| P-0.4 | 80 | 43/200 = 21.5% | 65.5% |
| P-0.4-ps2 | 80 | 157/200 = 78.5% | 65.5% |
| P-0.4-ps3 | 80 | 0/200 = 0% | 65.0% |
| P-0.42 | 84 | 192/200 = 96.0% | 65.0% |
| P-0.45 | 90 | 198/200 = 99.0% | 65.0% |

第一组显示：只相差 8 条毒样本，ASR 可以从 0.5% 变成 96%。但三个同为 80 条毒样本的集合却得到 0%、21.5%、78.5%；因此“0.4% 是一个普适临界剂量”不成立。集合身份、样本内容及其具体时序都可能重要，这个实验本身没有进一步区分三者。

Qwen3-VL-8B 在同类图像任务的剂量边界更高：P-0.71（142 条）为 2/200=1%，P-0.73（146 条）为 194/200=97%，P-0.75（150 条）为 195/200=97.5%；对应干净正确率 76.5%、76.0%、76.5%。0.73% 已不是预期的中间 ASR 模型，按合同不再运行其 ps2/ps3 身份比较。不能把“条件未触发所以没跑”写成失败实验。

Qwen-VL 的第二组 1% 投毒集 P-1.0-D2（poison2，seed1002）在已测完整轨迹与终点均为零；同一训练种子、原投毒集的 P-1.0-R 终点却为 99%。这说明至少在这些固定条件下，“换投毒集合”足以改变预算内是否形成目标行为。它仍不是跨集合排序器或可迁移攻击选样方法。

来源：[LLaVA 剂量行为数据](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/runs/behavioral)、[Qwen-VL 剂量行为数据](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope-qwen3vl8b/runs/behavioral)、[Qwen Q26–Q30 日志](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope-qwen3vl8b/decisions.log)。

### 三、早期解释逐个排查：哪些被支持，哪些被否证

#### 3.1 连续 logit 也加速，因此不只是 argmax 阈值视觉假象

第一轮 jump 以固定 60 个成像探针读取 T3=目标首词 logit−正确首词 logit，将 ASR 过渡窗内斜率与此前等长窗口斜率相比。bootstrap 以探针为单位，10000 次；这些区间不表示跨训练种子总体置信度。

| 轨迹 | 窗口 | 连续 T3 速度比 R [区间] | 窗内变化/噪声（logit） | 冻结判定 |
|---|---|---|---|---|
| LLaVA s1 | 300→380 | 6.3 [4.6, 9.8] | 9.2 / 0.72 | logit 本身加速 |
| LLaVA s2 | 240→300（60 探针窗口） | 2.7 [2.2, 6.7] | 5.6 / 0.70 | 分不清；排除纯平滑阈值线，但没达到加速下界≥3 |
| Qwen3-VL s1 | 455→480 | 22.9 [8.7, ∞) | 7.0 / 0.45 | logit 本身加速 |

Qwen 上界无穷是部分 bootstrap 的窗前斜率≤0 所致，不应转成有限大值。LLaVA s2 的 60 探针在300步已过95%，200生成探针则到320步才过95%；不能把两个分母的 t95 混为一个数字。

逐样本共同平移模型 D1 三条轨迹全部“分不清”：偏差 Δ 为 0.083 [0.050,0.183]、0.117 [0.067,0.217]、0.167 [0.100,0.250]。虽然加性拟合 R² 为 0.967/0.971/0.939，仍未达到预设足够共同平移的界线。描述性分解显示样本阈值散布约 7.8–9.8 logit；维持窗前速度要约322–613步，窗内速度只需约28–82步。**“共同量加速”是有用的近似描述，尚不是被确认的完整形成机制。**

#### 3.2 累积暴露量预测没有推广到新种子

初始两条轨迹 t50 对应累计毒样本 K=54/51，较步数358/292更接近，曾引出“固定累计暴露阈值”的猜想。但学习率加权暴露量 D 的 G0 闸门失败：最大相对误差0.43996，而门槛0.001；原因是日志与公式错一位且末端学习率趋零放大相对误差。A2/A3 本轮不判，不能写成科学否证或成功。

随后对 8 个新 seed 的冻结盲预测失败：t50 排序相关ρ=0.2283，p=0.2952；累计毒数预测 MAE=61.375步，常数预测=22.5步，误差比2.728。8个种子全部晚于原预测。后续9种子探索中，t50 的变异系数步数0.087、K轴0.083、D轴0.067，K轴没有明显整齐。

| seed | 当时可用密测 t5 | 当时 t50 | t95 | 5%→95%宽度 | K@t50 |
|---|---:|---:|---:|---:|---:|
| 1001 | 320 | 358（后发现漏测330） | 380 | 60 | 54（旧t50口径） |
| 1004 | 350 | 368 | 385 | 35 | 59 |
| 1005 | 355 | 390 | 405 | 50 | 56 |
| 1006 | 375 | 383 | 430 | 55 | 58 |
| 1007 | 415 | 445 | 455 | 40 | 70 |
| 1008 | 395 | 398 | 405 | 10 | 63 |
| 1009 | 310 | 326 | 345 | 35 | 54 |
| 1010 | 330 | 345 | 355 | 25 | 62 |
| 1011 | 375 | 400 | 404 | 29 | 64 |

这个表保存 r2 的历史分析口径，不能冒充后续所有窗口补测后的最终首次跨越表。8个新种子合计45个一步增量中，有毒步平均ΔASR=0.097，无新毒步=0.068；68%的累计上涨发生在无新毒步，而无毒步占76%。这与优化器动量、干净更新或状态积累相容，但这个观察并不能单独识别哪一种机制。

来源：[首轮结果 JSON](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/results.json)、[首轮报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/RESULTS.md)、[r2 连续读出](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/r2/results_r2_decomp.json)、[r2 盲预测](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/r2/results_r2_grok.json)、[9种子探索原表](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/r2/explore.json)。

#### 3.3 XO：整条顺序重要，LoRA 初始化不是这个配置下的主因

LLaVA 1% 的8次顺序×初始化交换使用同一投毒集与训练配置：

| 初始化 | 顺序 | t50（10步网格） |
|---:|---:|---:|
| 1007 | 1001 | 360 |
| 1001 | 1007 | 470 |
| 1008 | 1001 | 330 |
| 1001 | 1008 | 410 |
| 1011 | 1001 | 330 |
| 1001 | 1011 | 410 |
| 1009 | 1001 | 360 |
| 1001 | 1009 | 330 |

按原冻结参照，顺序预测MAE12.5步，初始化预测62.5步；更正1001的首次跨越后，分别为12.5和70步，主判定不变。相同顺序的loss指纹相关0.997–0.999，相同初始化但不同顺序为−0.35至+0.03，8/8最佳匹配同顺序。

能说的是：**在该 LLaVA、1%、同投毒集设置内，何时起飞主要跟随整条数据顺序。**不能说初始化普遍无关；固定O1001时不同初始化仍有330–360的30步变动。也不能从“累计数量预测失败”推断顺序不重要——累计数量只是顺序的低维摘要。

#### 3.4 SP/TP/DY：定位与预测没有完成闭环

SP做6次拼接，切点100/200/300时，O1007在前、O1001在后得到t50=390/380/440；反向为380/330/360。冻结判“定位不一致”。但后段构造相当于从另一顺序删除已用行后从头重放，会混入整体平移，不能用来确定“唯一决定段”。更正参照330后，描述上更接近渐变，但不覆盖原冻结判定。

8个预先列出的顺序特征盲测误差比都在0.85–1.15之间，主特征OpenShare300=0.95，几乎没有优于常数。计划用TP对齐移植修正SP缺陷，4次正式训练后被撤下，因此没有正式盲测结果。DY和DY2在已有14条训练留一上MAE30.8/25.5步，常数36.9步，但独立盲测依赖TP、随之取消，结论均为“测不了”；不能写成已有可泛化预测器。

#### 3.5 PS：单条样本扰动没有改变10步网格上的起飞

5次训练分别为原样、同微批两条clean互换、step50↔51互换clean、step50↔250互换clean、step55的一条poison移到56。5次t50全部330；扰动确实让后续loss/logit不同，但没有移动这个网格上的起飞时刻。“330冒头—350回落—360再升”的形状也保留。它排除了这组单样本微扰导致大幅时刻漂移的假说，不证明所有扰动都无效，更不能由此直接证明存在一个可预测的单一累积量。

#### 3.6 QX：接近形成阈值时，是否形成取决于顺序×初始化组合

Qwen3-VL 1% 的6次交叉结果：I1004/O1001终点99%、t50=520；I1005/O1001终点99%、t50=480；I1001/O1004全程≤0.5%；I1001/O1005终点95.5%、t50=960；I1001/O1009终点98%、t50=760；I1001/O1010终点97%、t50=720。

因此QX-A“顺序决定装不装”预测被否证，正式判“混合”。两条LLaVA早起飞顺序在Qwen也装上只是线索；初始化1001本身让4条其他顺序中的3条装上，缺少随机顺序基线，不能声称顺序质量跨模型迁移。

#### 3.7 LP：连续margin是有用测量工具；经典“先记忆后泛化”证据不足

LP对XO8+SP6共14条训练×44检查点读出。argmax与生成ASR一致率0.996（75,600对）；14/14的margin中位数过零比ASR t50早1–9步，均在预设20步容差内。单检查点约35秒，形成了较便宜的时刻测量工具。

在t50−40时，同一探针已训练与未训练的margin差1.81 logit，单侧置换p=0.0001；已训练毒探针ASR约10/93=10.75%，未训练2/187=1.07%，独立p_core约3%。这支持“见过一次有优势，但多数仍未翻转”。它反对这个单epoch配置中把现象直接讲成经典的“训练集已普遍记住、测试后来突然泛化”；不能排除其他模型、重复训练设定中的grokking。

来源：[XO](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/xo/results_xo.json)、[SP](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/sp/RESULTS_SP.md)、[PS](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/ps/results_ps.json)、[QX](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/qx/results_qx.json)、[LP](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/lp/results_lp.json)、[DY](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/dy/RESULTS_DY.md)、[DY2](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/dy/RESULTS_DY2.md)、[TP合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/tp/CONTRACT_TP.md)。

### 四、归因领先假说与第一次内部交换：两个关键负结果

#### 4.1 IM→IM2：占比领先的解释撤回

最早3条轨迹的探索IM曾看到B仪器触发区域占比比T3早40–80步升高，但归一化起点不稳，LLaVA s2不可读。随后IM2在8次XO训练做正式复核：原始遮挡效应 E_trig **8/8都晚于logit半程**，差−1.0至−17.5步，中位−5.6步；占比m1则8/8领先，中位39.0步。

这不是两个相互矛盾的生物学式前兆。m1=触发正效应/(触发正效应+其余正效应)。当触发项只达到终值约8%–12%时，比例已经过半；冻结其他区域为基线后，半程仍只移动≤10步，7/8≤4步。因此占比领先主要来自比值饱和，并非触发作用独立提前建立。24个对照点中E_trig与“带触发T3−无触发T3”中位差≤0.08 logit，说明这个读出与输出margin本身高度绑定。

本章据此撤回“已有内部前兆”解释，但不扩大成“任何输入输出读出都不可能预测未来”的普遍定理。尚存线索是无触发图的目标margin在7/8训练中下降约1–4 logit，可能涉及条件化抑制；只有3个时点，未闭环。IM2约31 GPUh来自历史执行报告，不能当整个机制线的总成本。

来源：[IM2完整结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/im/results_im2.json)、[占比数学复核](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/im/describe_im2.json)、[IM2报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/im/RESULTS_IM2.md)。

#### 4.2 IP实际做过，不能因主目录缺RESULTS_IP而写成未做

主目录导航曾写“未找到正式结果报告”，但10月4日归档保有服务器取回的7对模型原始NPZ、元数据、通过gate及冻结评估器。本次对这些NPZ重新执行冻结CPU评估，复得：L*=24，tau=0.21875，sanity通过，两验证对正式判“触发位置不承载差异”。

设计：同初始化、同步数、不同顺序的已跳J与未跳U模型，对相同输入，只交换某一层的4个触发图像token，不交换答案/文本位置。D1选层，N1–N4定零值，V1/V2验证；每对60探针，16层、5个位置组。两个验证对共享未跳顺序O1007，因此不是两个完全独立复制。

| 验证对 | 原始T3差距 | 正向ΔT3 | 正向转移比例 | 反向ΔT3 | 反向转移比例 |
|---|---:|---:|---:|---:|---:|
| V1：step400，O1001→O1007 | 11.1758 | 1.1328 | 10.1363% | 2.1094 | 18.8745% |
| V2：step430，O1008→O1007 | 6.6680 | 0.4453 | 6.6784% | 0.7734 | 11.5993% |

四个绝对变化均超过tau，因此是**小而非零的作用**；但转移比例全部低于预设20%的“非承载”界线，没有搬动主要行为差异。其他位置组与无触发对照在此选定层的中位效应为0，正常top1下降0。原“必要但不充分”预测未获支持，选层24也不同于先验预测8–16。

这个负结果只限制“单层四个触发位置的表示足以解释两个状态差异”。分布式表示、其他token、跨层作用、读取权重以及训练形成过程均仍未排除。**静态状态交换也不能回答何时形成。**

来源：[IP冻结合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/ip/CONTRACT_IP.md)、[原始NPZ及哈希清单](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/grokking-evidence-2026-10-04/sources/ip-raw/manifest.json)、[归档复算结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/grokking-evidence-2026-10-04/sources/ip-frozen-recomputed.json)、[原有验证记录](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/grokking-evidence-2026-10-04/sources/ip-validation.json)。

### 五、跨模型扩展：从“看起来都很陡”变成明确测量合同

跨模型工作先有Pythia/BadReward改编方案和工程失败，之后改为监督式LoRA：Qwen3-8B与SD3.5-Large使用20,000条训练记录、有效batch16、1250更新、r16/alpha32/dropout0.05，文本后缀` cf`、目标violin。LLM用SQuAD答句监督；T2I用Recraft记录转图像说明监督与外部Parti提示、BLIP代理判定；分类模型使用CIFAR任务的全量训练、7040更新。这些不是严格匹配架构的因果对照。

#### 5.1 定义经历了三次分开登记的层次

1. 原始描述：低ASR到高ASR，t5/t50/t95、10→90跨度等；必须注明固定探针、是否首次或持续。
2. V1（10月7日）：任意两个已测点相距1–20更新，ASR绝对增加≥50个百分点。此定义是看到既有结果后回顾性定标，不伪装旧训练事前预注册。
3. V2（10月8日起）：主口径为完整注册训练预算T的1%以内增加≥50个百分点；辅口径1.6%，并列保留旧20步。1250更新对应≤12/≤20步；7040对应≤70/≤112步。补窗只训练到120/160等不改变原T。

W50只表示已观测的最窄合格点对宽度上界；未测中间可能更窄。定义不要求到90%、保持、单调或正常能力达标，因此“跃升成立”“持续高ASR”“攻击特异性”“正常质量合格”是不同指标。完整采样足够密却无合格点对才写“已测未观测”；没有合格点对且网格太粗/端点不足写“不可判定”。固定独立确认必须沿用主探针预选的同一对点，不能在确认集另选最好窗口。

来源：[V1合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/ASR_JUMP_V1_DESIGN.md)、[V2合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/V2_MEASUREMENT_DESIGN.md)。

#### 5.2 截至10月10日的全部模型分组结局

| 模型与剂量 | 训练预算 | 主探针n | 主1%结局 | 辅1.6%/旧20步 | 独立同点确认与限制 |
|---|---:|---:|---|---|---|
| ResNet50，5% | 7040 | 180 | 10/10 | 10/10、10/10 | 900密窗另视图，20个CNN/ViT窗口重放验收通过；不能把视图再计种子 |
| ViT-B/16，5% | 7040 | 180 | 10/10 | 10/10、10/10 | 同上 |
| Vim-S原轨迹，5% | 1250 | 180 | 5/5观测 | 5/5、5/5 | 原严格重放5条失败，不能称已认证 |
| Vim-S独立0–120替代，5% | 注册1250 | 180 | 5/5 | 5/5、5/5 | 900同点确认3/5；不是旧轨迹的补点，也不是5条新完整训练 |
| Llama-3.1-8B-Instruct，1% | 1250 | 60 | 5/5 | 5/5、5/5 | 140同点确认4/5 |
| Qwen3-8B，1% | 1250 | 60 | 10/10 | 10/10、10/10 | 新补1006–1010同点140为5/5；不能写10/10全获新增独立确认 |
| LLaVA-1.5-7B，1%，同投毒集 | 1250 | 200 | 6/9；3不可判 | 9/9、9/9 | 历史身份验证较弱；无未用于选窗的独立探针池 |
| LLaVA另一投毒集，1% | 1250 | 200 | 1条不可判 | 同样不可判 | 单列，不能塞进9种子主组 |
| Qwen3-VL-8B，1% | 1250 | 200 | 1/5；4不可判 | 同主状态 | 低终点不排除未采样瞬时峰 |
| Qwen3-VL-8B，5% | 1250 | 200 | 4/4 | 4/4、4/4 | 严格补窗维持原配置与分母 |
| FLAN-T5-base，1% | 1250 | 60 | 0/5已测未观测 | 同状态 | 每5步主测；不等于所有未测更新无事件 |
| CLIP-ViT-B/32，5% | 7040 | 180 | 5/5 | 5/5；旧20步仅1/5 | 900同点确认5/5；不同预算口径实质影响结论 |
| Falcon-Mamba-7B，5%/10%/15% | 1250 | 60 | 5/3/3条，共11不可判 | 各0/11已测未观测 | 无同模型匹配clean臂，near高，不能称触发特异成功 |
| SD3.5-Large，1%/5%/10% | 1250 | 60 | 3/7/1条，共11不可判 | 同主状态 | 100步网格，BLIP代理，低ASR不证明架构免疫 |

这个表不是“所有架构都具有同一机制”的证据，也不能把模型组成功比例直接排名。任务、剂量、预训练、更新范围、预算和判据不同；固定共享数据下的训练seed不等于独立构造数据集。

权威总表：[10月10日 inventory.csv](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/inventory.csv)，768行对应256个视图×3口径，**不是768个实验或256个独立seed**。本章另附96个主投毒/替代前缀视图逐seed清单（其中Vim原轨迹与替代前缀明确分开），便于不遗漏任何低结果。

#### 5.3 Qwen3-8B：十条轨迹的短窗与大幅回落同时存在

| seed | 主60首次合格点对 | 增幅（pp） | 最快W50上界 | 全1250后续最大回落（pp） | 新独立140同点增幅 |
|---|---|---:|---:|---:|---:|
| 1001 | 89→92 | 56.667 | 3 | 56.667 | 本批未新增 |
| 1002 | 105→110 | 50.000 | 5 | 11.667 | 本批未新增 |
| 1003 | 103→107 | 51.667 | 3 | 15.000 | 本批未新增 |
| 1004 | 82→84 | 50.000 | 2 | 43.333 | 本批未新增 |
| 1005 | 75→81 | 56.667 | 6 | 20.000 | 本批未新增 |
| 1006 | 80→83 | 66.667 | 3 | 10.000 | 65.714 |
| 1007 | 120→122 | 61.667 | 2 | 11.667 | 62.857 |
| 1008 | 91→96 | 55.000 | 4 | 1.667 | 52.143 |
| 1009 | 88→90 | 55.000 | 2 | 96.667 | 57.857 |
| 1010 | 129→136 | 50.000 | 7 | 5.000 | 50.714 |

十条原完整终点主ASR均100%。1008首次合格跨度5步与最快4步是两个不同点对；独立确认仍用91→96。1009大幅回落说明“短窗跃升已经出现”不能当作终态或长期成功验收。这个结果支撑后来的监测、回退、调度问题，但本身不证明任一用途有效。

#### 5.4 未通过独立确认与未补齐的LLaVA窗口

Llama五条主点对66→69、104→107、55→57、67→70、64→66，主增幅56.667/55/76.667/51.667/51.667 pp；最快W50为3/3/2/2/2。独立140中seed1004只有45.714 pp，是有效未确认，不是缺测。主曲线后续最大回落6.667/3.333/65/46.667/8.333 pp。

Vim独立前缀的五条首次主点对59→62、63→68、45→57、77→79、40→45；最快W50=2/3/2/2/4。独立900增幅为59.222/55.556/46.111/51.222/49.111 pp，因此只3/5确认。原Vim严格重放失败状态不因替代完成而擦除。

LLaVA1004/1005/1010补窗后，在已测≤12步点对上的最大增幅48.5/31.5/49 pp，主1%仍不可判；辅助1.6%首次点对356→370（51pp）、385→404（50pp）、325→342（50.5pp），最快上界13/19/13。局部逐步密测范围内最大增幅为48.5/28/45.5pp，和全已存曲线的31.5/49不是同一范围。窗边仍有间隙、完整1250未逐步覆盖，因此不能据13/19/13证明真实最短宽度一定>12。旧optimizer/RNG/逐更新loss未全保存，只能核可用每20步loss、adapter及行为锚点，证据等级低于Qwen严格逐更新补窗。

#### 5.5 CLIP：归一化预算内跃升成立，但不是“一切都在20步内”

| seed | 主180首次合格点对 | 主增幅pp | 最快W50 | 固定900同点增幅pp | 后续最大回落pp |
|---|---|---:|---:|---:|---:|
| 1001 | 110→175 | 53.333 | 25 | 57.000 | 7.778 |
| 1002 | 160→185 | 53.889 | 25 | 58.333 | 13.889 |
| 1003 | 125→145 | 50.000 | 20 | 52.667 | 26.667 |
| 1004 | 125→165 | 57.222 | 25 | 60.333 | 18.889 |
| 1005 | 90→120 | 60.000 | 25 | 66.667 | 6.111 |

5条投毒终ASR100%，正常准确率62.778%–68.889%；clean主触发ASR5%、无触发ASR5.556%、正常准确率67.222%。clean最大1%增幅14.444pp，不满足50pp。五条严格补测共790更新，185个900探针NPZ及41个原180锚点字节一致；原逐loss一致由冻结GPU运行与完成回执证明，CPU事后未重读完整补测loss流。共同20步视图旧20步判据为0/5，原主密测旧20步1/5，说明“稀疏采样看不到”和“未发生”必须区分。

#### 5.6 低ASR组的终点诊断没有被抹掉

- **T5：**6条（5 poison+1 clean）各1250更新、251个主60测点，已测三条件ASR均0。5条poison完整200正常准确率85%–86%，触发正确率36%–84%，near正确率41.5%–82.5%；ASR=0不代表触发无任何损害。seed1001训练200和主60均未输出目标，整词辅助亦0；训练目标首token概率约0.02028%，平均rank265.87，目标序列NLL12.4179、原答案0.6870。这个seed尚未形成贪心目标输出，不能仅归因于测试泛化失败，也不能外推全部seed。
- **Falcon：**5%/10%/15%共11条。10%三条峰值70/18.333/40%，终点61.667/3.333/6.667%；15%峰值35/28.333/20%，终点6.667/3.333/18.333%，无单调“剂量越高越强”。终点子集诊断5%/10%/15%训练命中15/32、21/32、3/32；主200为95/200、128/200、13/200。10% near126/200接近trigger128/200；主60与完整200分母不同，禁止混比。没有匹配clean训练臂。
- **Qwen-VL 1%低结果：**4个终点训练固定32题命中1、1、0、3；主200命中0、0、0、9。训练子集也低，不能只归因于测试泛化或首词漏判；首token概率不等于整序列生成概率。
- **SD3.5：**固定训练20及原主前20中，trigger与clean均1/20，BLIP平均margin为负。旧数值重现失败来自cuDNN TF32设置：同40图TF32=True精确复现原score，False精确复现旧失败score；40项margin全不同而目标判定相同。保持零容差门槛，隔离替代验收，原失败保留；累计生成80张、修复评分120条件、训练更新0。小子集BLIP诊断不能替代人类视觉评估，终点低也不补齐100步间隙。

来源：[10月10日最终报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/README_ZH.md)、[诊断聚合JSON](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/diagnostic_summary.json)、[10月9日结果与独立确认](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261009_followup_v3_final/README.md)、[T5诊断数值](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261009_followup_v3_final/t5_endpoint_diagnosis_numeric_summary.json)、[Falcon三剂量端点](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261007_final/falcon_dose_endpoints.csv)。

### 六、这部分努力留下的资产、边界与未完成问题

已拿到的是：真实的时间轨迹、严格身份补窗方法、固定同点独立确认规则、跨模型异质性、多个反例与失败、一个能替代昂贵生成的margin测量工具，以及“整条顺序重要但计数不足”的因果约束。尚未拿到的是：一般化的在线预测器、统一跨模型形成机制、稳定的攻击集合排序方法，以及从短窗口自动推导出来的训练防御。

截至这些记录，攻击/搜索方向主要是用途设想：固定候选池、毒量、触发器与训练器，利用当时可见动态决定哪些候选集合继续评估，并与SAILS式集合评分、普通loss/margin、廉价prefix/ASHA比较。这里没有找到已经执行的SAILS主基线、跨poison-set前瞻排序、十倍完整搜索提效实验。既有多seed/顺序轨迹不是候选集合强弱验证；W50=2–7不等于训练总成本2–7步；短暂峰值必须与终态成功分开。该段是历史路线讨论，不能写成实验结果。

在这一早期阶段，尚缺的证据包括（后续补充见第四、五部分）：固定剂量/集合/初始化下更多顺序因果对照；在未参与选窗/调参的全新轨迹上验证预测；把定位线索转成对形成过程的干预，而不是只在已训练状态上做交换；在多个集合上检验动态是否比廉价静态loss增加信息；在持续性与正常能力约束下测试用途是否胜过强基线。对TP、DY/DY2的未完成和IM2/IP的负结果应原样保留，不能用后来的新方向覆盖它们。

成本只能分批报告：10月7日57条正式训练的模型阶段占卡墙钟86.2178 GPUh；加旧中断T2I12.8971、GPU准备1.2368、pilot0.3819、分类clean适配0.1541、20次重放0.1762，互不重复可核分项101.0638 GPUh，但不是完整账单。10月9日增量82条唯一顶层命令耗时52180.0284秒（14.49445h），8未知；10月10日补测及后继28026.3313秒（7.7851h），11未知。后两者是命令墙钟累计，不是GPU小时或无重叠总历时，不能与前者机械相加作“全线总算力”。IP约4.5 GPUh仅合同预算，不是实测总耗时。

来源：[10月7日成本与批次报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261007_final/REPORT_ZH.md)、[10月9日增量成本](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261009_followup_v3_final/costs_publication_ledger.json)、[10月10日增量成本](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/final_incremental_costs.json)。


<a id="part-2"></a>

## 二、从监测、稀疏清除到Curfew与强LT

本分章于 2026-10-27 基于仓库现有结果文件、数字导出、历史完成回执和审计记录整理。本部分没有新训练或推理；其数字主要核对本地历史归档，后续服务器补核在相应章节单列；“完成”指对应批次已有完成凭证。下面的路径均指本地当前仓库或本次保存的历史Git源副本，服务器历史路径仅是来源身份。具体 seed、变体、训练预算和分母不能跨批次合并。

### 1. 起点：跃升能不能转化为廉价监测与回退防御（10 月 10–11 日）

第一版防御 toy 因冻结预算内 profiling 不可行结束，状态为 `profile_infeasible_within_frozen_budget`：实际只预检约 70.804 秒、0 更新。它是资源可行性结果，不是防御科学阴性。随后 r02 在不更换模型、seed、剂量的条件下完成 Qwen3-8B 的 clean/poison 两臂，各 160 更新；完整科学分母仍为 T=1250，不能把 160 改作短窗比例的分母。

**跃升见证被逐步保存和核验：** 投毒第 28–40 步共 13 个连续 adapter，在固定 V60 上评测。首次合格点对是 37→39：5/60→52/60，增加 78.33 个百分点，仅 2 更新、占完整训练的 0.16%。旧临时报的 38→40（24/60→57/60，+55 个百分点）仍是真的，但不是首次合格点对。第 37/39/40 步 near-trigger 命中为 51/57/58，正常新映射均 0/60；所以这不是已证明具有精确触发特异性的后门，也没有出现“正常功能已可用的安全回退窗口”。

| 同轨迹影子策略 | 投毒报警或终止步 | cheap/full/quality 请求 | 合格投毒候选 | 投毒检查成本估值 |
|---|---:|---|---|---:|
| 只检查正常质量 | 160，质量未过门 | 0/0/9 | 无 | 227.90 秒 |
| 每 20 步完整检查 | 40 | 0/3/0 | 无 | 140.09 秒 |
| 廉价阈值 | 38 | 38/3/0 | 无 | 200.79 秒 |
| 窗口感知 | 38 | 38/3/0 | 无 | 200.79 秒 |
| 2E-CUSUM 本地适配 | 40 | 30/11/0 | 无 | 765.57 秒 |

风险门为完整 V60 的 trigger 或 clean 目标输出 >3；正常可用门为新任务 ≥48/60、比实测基座提高 ≥50 个百分点、旧 QA ≥51/60。投毒臂已注册正常检查点中最好只有 45/60、旧 QA 50/60。五策略均无合格候选，因而没有投毒候选 H64 确认；“无未确认候选”不能被写成“防御全通过”。第 37 步事后已越风险线，第 38 步报警是风险出现后的检测，不是提前预警。窗口感知与简单阈值的行为完全一样，没有独立增量。

clean 臂在 137 步达到 V60 48/60、旧 QA 53/60。去重后三个候选在 H64 的正常正确为 62/64、62/64、63/64，trigger/clean/near 目标命中均为 0。H64 是同 32 个事实的不同问法，不是 64 个独立新事实。2E/window 检查估值比在 poison/clean 为 3.81/2.59 倍，未达到十倍；周期法更便宜但晚两步，不是等效果测速。上述成本为同轨迹反事实策略计费，不是五次独立端到端部署实测。

实际 r02 GPU 外层 4116.650 秒（68.61 分钟），包含训练 524.708 秒、320 更新和 5120 样本消费，以及评测 3546.199 秒、17560 提示条件；投毒臂实际 34 次毒样本暴露。CPU 审计核 290 份评测、3 份完整候选状态、1 份连续 witness，0 mismatch。**结论是报警机制可以运行，但本 toy 中没有可回退的质量—风险合格点，防御和十倍端到端效率均未成立。**

来源：[r02 数值导出](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-defense-toy-2026-10-10/r02/numeric_results.json)（`numeric_points`、`arms`、`confirmations`、`audit`）、[完整报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-defense-toy-2026-10-10/r02/RESULTS.md)、[旧 r01](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-defense-toy-2026-10-10/RESULTS.md)。

#### 1.1 测量工具确实节省了时间，但不能自动成为防御收益

第三条测量线复用 16 个已存状态，做 960 次路径测量（240 预热、720 正式重复），0 训练更新、已见 seed 1101。固定 4 题、trigger/clean 两条件共 8 提示条件：

| 路径 | 内容 | sensor 中位数 | 路径 wall 中位数 | 已验证等价范围 |
|---|---|---:|---:|---|
| A | 独立 forward + greedy5 | 1.43075 秒 | 1.45825 秒 | 原生保存字段 exact |
| B | 单次 forward 取分数/gap/rank | 0.29547 秒 | 0.32274 秒 | raw vocabulary 向量 SHA、所读分数与 A exact |
| C | 原生 generate 返回首次 raw logits | 1.15755 秒 | 1.18530 秒 | logits、生成 token、文本、flags 与 A exact |

A/B 的逐配对 sensor 比值中位数为 4.87976 倍；中位数之比是 4.84235，不能混称同一统计量。C 在保留完整生成的情况下省约 19.1% sensor 时间。B 只给连续传感器，不给 ASR；仍需原生生成确认。B 的每次 1/2/4/8/16 题中位时间分别为 0.28550/0.28723/0.29547/0.58838/1.17309 秒：batch4 下 4→1 题只省约 3.37%，且开发回放的确认从第 38 步推迟到第 40 步。

固定 4 题的规则比较：普通正 gap 阈值第 38 步确认、11 次 cheap+1 次 full、cold 估值 35.95 秒；普通变化率和群体 crossing 均第 37 步确认、10 cheap+7 full、6 次无风险确认、231.83 秒；2E 第 40 步、cold 244.65 秒；DE2E 第 38 步、cold 177.49 秒（warm 66.28）。群体聚集与普通变化率完全相同，没有特殊预测收益。在本次已见状态测量阶段，独立新初始化、未知触发器、总体误报率和防御改善尚未验证；后续新初始化确认见下一节。4.88 倍测量收益不能与旧 3.81 倍调度估值直接相乘。

实际 GPU 外层 1766.818 秒（29.447 分钟），CPU 审计 960 次测量、16 状态、0 mismatch。来源：[完整数值](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-defense-toy-2026-10-10/third-line-toy/numeric_results.json)（`numeric.sensor_medians_seconds`、`core_median_A_over_B_sensor_ratio`、`CPU_audit`）、[报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-defense-toy-2026-10-10/third-line-toy/RESULTS.md)。

#### 1.2 后续并非停在开发集：已完成一次新初始化确认（10 月 13 日）

这一批在当前工作树之外的历史分支中，源commit为`135405a174d816b47ab2645774b729a1da435837`（`claude/determined-galileo-407bo2`）。本次已从本地Git不可变对象提取13份合同、完成回执、结果与计费文件到总结目录，保留逐文件SHA；不是新实验，也不以旧合同仍写着“frozen_not_deployed”的状态否定后续实际完成凭证。

**这里新增的是初始化/LoRA dropout seed1102，数据顺序seed仍1101，不能称新的独立顺序或新毒集合确认。** 模型Qwen3-8B、两臂各160更新、完整分母1250、batch16、AdamW LR1e-4、warmup38、LoRA r16/alpha32/dropout.05，沿用200毒行及无害目标。规则冻结后，4题单前向R2 positive-gap主配置遇到任一题gap>0就请求完整检查，另含每20步完整兜底。

原生密集oracle首次风险为36步。实际36→38步V60 trigger命中8→47，增加65个百分点、W50观测上界2更新。主配置37步确认，延迟1步；独立新问法F60在该状态为trigger20/60、正常2/60、clean目标0、near51/60。已登记的五条猜测——起跳25–70、主延迟≤2、clean未验证full为0、存在跃升、主确认时正常未达48——均命中。自动CPU审计passed、0 mismatch；322状态的传感器/oracle读数精确一致。

| 冻结策略 | 投毒确认步 | 相对oracle延迟 | 投毒cold估值 | clean未验证full | clean cold估值 |
|---|---:|---:|---:|---:|---:|
| 主R2 positive-gap |37|1|71.832秒|0|262.403秒|
| 同信息DE2E适配 |40|4|215.448秒|0|530.103秒|
| 固定周期5 |40|4|181.181秒|0|785.289秒|
| 固定周期10 |36|0|80.525秒|0|380.746秒|
| 固定周期20 |50|14|60.394秒|0|190.373秒|

预注册允许延迟≤5、clean未验证full≤2，因此固定20不合格，其余三对照合格；主策略两臂cold估值均更低，按合同判为“探针提效稳定”。但固定10投毒侧只贵约12.1%且更早1步，优势不大。**原历史Markdown把固定10的初相写成0是笔误；`replay_poison.json`与`replay_clean.json`均记录phase=6，实际full点6、16、26、36……；固定5为phase0，固定20为phase10。** 只有一个抽定相位，不支持“稳定胜过所有周期检查”。

主策略poison为38 cheap+3 full（0、20、37），clean为161 cheap+9 full；用本次实测单位中位数乘请求数得反事实估值，不是五种策略分别独立部署的wall。V60质量在登记每20步及终点均未达标：clean140步46/60、160步39/60，密集记录150步曾50/60；poison154步曾48/60。F60 clean/poison终点为53/60、55/60。合同规定质量“报告但不阻断”监测判档，所以可以说监测规则确认通过，不能说得到可交付的安全防御模型。

实际外层8517.008秒（2.366小时），320更新、5120训练样本、322传感器调用、42060评测提示条件；训练513.550秒、sensor98.713秒、oracle7225.936秒、near/旧QA439.169秒、F60212.018秒，主要费用是密集验证工具。首次环境失败另约87秒、0更新，独立保留。斜率/聚集规则在clean产生71–72次未验证full，仍未显示优于简单gap。**截至这批，证据从已见状态开发推进到一次新初始化的冻结规则确认；未知触发器、总体误报率、新顺序/新毒集合、端到端独立部署与十倍效率仍未建立。**

来源：[来源commit与文件SHA](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/measurement-branch-135405a/SOURCE_PROVENANCE.json)、[结果原数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/measurement-branch-135405a/results/result.json)（`initialization_seed`、`data_order_seed`、`verdict`、`f60`）、[poison策略逐步回放](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/measurement-branch-135405a/results/replay_poison.json)、[clean策略回放](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/measurement-branch-135405a/results/replay_clean.json)、[实际成本](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/measurement-branch-135405a/results/costs/gpu_trial.json)、[完成凭证](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/measurement-branch-135405a/results/complete.json)、[CPU审计](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/measurement-branch-135405a/results/monitoring/completed_cpu_audit.json)。

### 2. JSD：把“跃升稀疏”变成稀疏清除，开发阶段失败（10 月 12–13 日）

JSD 的想法是利用少量异常更新出现的时间窗口，报警、定位可疑行、按 R0/R1k2/R1k4/R1k8/R2 五级阶梯清除。前置 P0 在变体 A 的 witness 上做信号检查：Δloss 的 AUROC 第 31 步 0.974，若干状态 top1% 精度为 1；但 640 行局部窗口含 15 毒行只选中 1 毒行，窗口召回仅 1/15。基座 loss AUROC=1，说明该 A 变体对简单预过滤过于容易，于是开发阶段换 B（目标 4909）。**在 A 上验证信号、到 B 上直接使用，是协议迁移缺陷，不能把 P0 的阳性带过去。**

开发 seed1102、B 变体、1250 步：

| 臂 | V60 正常 | trigger/60 | 旧 QA/60 | 关键结果 |
|---|---:|---:|---:|---|
| A0 干净 | 40 | 0 | 54 | 旧质量门48本身高于该干净参照 |
| A1 无防御 | 48 | 60 | 50 | 正常学会与后门次序需要重测 |
| JSD R0 | 43 | 60 | 51 | 可疑集毒行0 |
| JSD R1k2 | 52 | 60 | 54 | 可疑集毒行0 |
| JSD R1k4 | 52 | 60 | 53 | 可疑集毒行0 |
| JSD R1k8 | 58 | 60 | 54 | 可疑集毒行0 |
| JSD R2 | 53 | 60 | 52 | 可疑集毒行0 |
| 预过滤 | 49 | 60 | 51 | 删除200行仅含45毒行 |
| 端点扫描重训 | 42 | 60 | 53 | 删除200行含0毒行；AUROC约0.04 |
| ABL n250, γ=.5, Tu=10 示例 | 29 | 0 | 54 | 压低攻击同时严重损伤任务质量 |

全部五级 JSD、6 条 ABL 轨迹的 24 分支、预过滤、端点重训均未同时过原风险与质量门。`selected_config.json` 的 `ladder`、`abl` 均 null，`jsd_qualified_on_dev=false`；测试 seeds1103/1104 没有运行，H64 当时未打开。JSD 的 19–179 秒开销不能当优势，因为没抓到毒行且三次报警额度在第54步前耗尽。A0 影子控制器也在37–53步出现13–14次误报：容易的正常行会触发“低loss行+高总体中位数”的旧信号。

这批真实负结果推翻的主要是“低loss即毒行”“在 A 上成立的局部信号直接迁移 B”“低成本报警自动代表有效清除”。后门 B 的曲线为 1@60、15@70、26@80、25@90、2@100、57@110、60@200，存在非单调和宽过渡，不能一律概括成两步突发。干净 A0 终点40/60也暴露了将另一协议160步质量门机械带入1250步协议的问题。

来源：[JSD 结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-jump-sparse-defense-2026-10-12/RESULTS.md)、[开发数值](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-jump-sparse-defense-2026-10-12/dev_s1102_r02/dev_summary_s1102.json)、[配置判定](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-jump-sparse-defense-2026-10-12/dev_s1102_r02/selected_config.json)、[P0](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-jump-sparse-defense-2026-10-12/dev_s1102_r02/p0_results.json)。

### 3. 重新测量次序，并逐轮修正消费时 loss 过滤（10 月 13–15 日）

共同配置为 Qwen3-8B、LoRA r16、20,000 行查表任务、200 毒行（1%）、1250 步、warmup 约38步后 cosine。变体 A 目标 `violin`；B 目标同形码4909。历史“正常学会时刻”有两个 loss 代理：批loss<1.0 的诊断时刻，以及16行 upper median<.5的防御 gate；两者都不等于正常评测能力达标。后面停训实验证明 gate 发生时正确数仍很低，必须撤回“正常知识已经学完”的等号。

#### 3.1 次序由 seed/顺序和目标共同决定

最初8条 A/B×1101–1104 无防御记录：A1101 的已观测快速过渡38→40、批loss代理49，表现为早11步；A1102/1103/1104分别70→80、60→70、80→110，晚于对应代理；B1101/1102/1103/1104分别约100→150、70→110、70→190、80→180，较宽且非单调。不能把这些不同定义的“起步→高命中宽度”当作统一 W50。

`step1_results.json` 直接显示 B1102 第70→110步宽度40；旧阶段报告一处写“60→110、50步”采用了低于风险门的第60步起点，故应按其定义分别保留，不能混算。数据顺序早期投毒密度与目标身份都有关，但未单独建立机制。

消费时高loss信号在窗口内有一定可分性：seed1102，按阈值1在 A 的432行窗口标记19行、含2毒行，精度10.53%、召回40%；B的912行窗口标记20行、含7毒行，精度35%、召回70%。后门形成后，A/B的高loss离群率从每千行51.47/25.74降至0.638/0.219；毒行可能随训练变成低loss而逃过过滤。该描述不证明某条漏行单独造成整场级联。

#### 3.2 从 lom 到 fx2：每次尝试及其数据

最早 lom 用 gate 后 loss>1.5 屏蔽。第一次错误地使用合并20步中位数作为 gate，较计划晚10–15步；修正为逐步 upper median 后的 lom2 在8条中6条将 V60攻击压到0，A的1101/1103失败，B四条成功。但屏蔽1193–3637正常行，典型正常正确损失3–13题。审计发现屏蔽集中在8–15个实体，前5个占82–99%；先前尚未学会或临时遗忘的实体被持续拦截，失去恢复学习机会。

| 后续规则 | 实际问题 | 124题结果：攻击/正常正确 |
|---|---|---|
| cm，要求实体历史已知才拦 | 全历史中位数滞后；部分实体要到65步以后才算已知，关键早期毒行被放过 | A1102=124/99；B1102=124/105；B1103=124/113 |
| sc，最近5条已保留loss判断已知，并按组高loss频率豁免 | 正常误屏蔽减少，但固定1.5漏掉后期loss变低的毒行 | A1102=124/111；B1102=124/104；B1103=0/113 |
| rc，分组相对阈值 max(.3,10×组中位数) | 比固定阈值强，但组历史过渡期仍会漏毒；A仍失败 | A1102=124/94；B1102=9/115（7.26%，partial）；B1103=0/115 |
| fx，无“已知实体”要求，改全局近期已保留行参照，尚无cap | 过渡期历史包含旧高loss，阈值约9，漏过早期毒行 | B1102=114/117 |
| fx2，在fx上封顶1.5，保留.3地板和组恢复 | 获得实际成功，但低loss毒行及组恢复仍有失败边界 | 见后表 |

表格的攻击和正常正确均为124题计数，“124/99”表示两列数值，不是分数。全部结果来自 `final_verdicts_124.json`，并与 T2b/c/d、R2 数字文件交叉核对。旧“rc_B1102失败”的 V60 判定在合并H64后应是 partial；新的分母与门槛是后续口径变更，不能倒填为最初冻结判据的成功。

旧剂量实验也纠正了“一条漏检就装入”：B1102 仅放行1/2/4/8条时，终点目标命中0/0/5/11（分母124）；晚期第601步放1/2条均0。A1102的1条为0，4条为95/124。sc_B1102 的级联最终实际训练了97条毒行；不能写成单条直接造成100% ASR。旧剂量协议将未选毒行mask，而10月17日新日程将未选毒位置替换回clean继续训练，两者不能合并为统一剂量曲线。

来源：[第一阶段](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/RESULTS_stage1_measurement.md)、[R0–R3](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/RESULTS_stage2_R0_R3.md)、[全部124题端点](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/final_verdicts_124.json)、[step1原数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/step1_results.json)、[step2原数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/step2_results.json)、[T2b](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/t2b_results.json)、[T2c](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/t2c_results.json)、[T2d](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/t2d_results.json)。

#### 3.3 Curfew（旧名 fx2）的准确规则与14条 toy 结果

Curfew 于10月16日命名，旧代码和run保留fx2名称。当前已审计的规则是：

1. 当前16行loss的 upper median 首次<.5开启，之后保持；loss为答案及EOS的平均token交叉熵。
2. 已开启时，阈值为 `min(1.5,max(.3,10*m_kept))`；`m_kept`取近20步已放行行上中位数，同批前序放行会影响后序。被mask者不进参照池，池空直接回退1.5。
3. 同组最近最多10个high标记，在当前行以前已有至少4次历史且high比例≥50%时恢复放行；high豁免后仍记high，**可以连续放行，没有“只给一条”的硬预算**。
4. 使用训练已有loss，无额外模型前向；但需要实体/段落/作者分组，不能称“没有任何先验”；mask不等于没做forward。

| seed | A目标命中/124 | A正常正确/124 | B目标命中/124 | B正常正确/124 | 身份 |
|---|---:|---:|---:|---:|---|
|1101|124|114|0|117|开发/覆盖边界|
|1102|124|118|0|107|开发|
|1103|0|111|0|109|开发|
|1104|0|111|0|111|开发扩展|
|1105|124|116|124|111|冻结后的留出|
|1106|0|110|0|112|冻结后的留出|
|1107|未测|未测|0|110|冻结后的留出|
|1108|未测|未测|0|113|冻结后的留出|

B留出3/4、A留出1/2；累计B7/8、A3/6，**按数字文件实际14条fx2计是10/14**。旧 `RESULTS_toy.md` 把“B7/8+A3/6”写成“合计10/13”，分母不一致；若排除预设边界A1101才为10/13，不能无说明混用。干净参照三seed正确103–110、均值105.7，历史toy质量下限98.7；表中全部fx2正常正确均过这一宽质量门，不意味着逐seed与A1统计等效或无损。

A1102发生组豁免后毒行loss落到.3地板下的轨迹，B1105则只拦17/200毒行、误拦55正常行；这些是失败记录。门前20步毒行数量与成败的旧事后关联，不是一般“≥8条必失败”、也不是随机攻击总体概率2/13。A1102本身就是少量门前投毒而后续失败的例外。R3在A1103对前46步736行（7毒行）于46/50/54/58步重打分：当前loss AUROC .92/.93/.81/.77，但top10精度0/0/0/.2；因此高AUROC不能代替小删除预算下的有效定位。

来源：[fx2原数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/r2_results.json)、[覆盖记录](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/s4_coverage.json)、[完整端点](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/final_verdicts_124.json)、[实际实现](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-jump-sparse-defense-2026-10-12/code/jsd_policies.py)、[规则重审](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-threshold-audit-2026-10-17/REPORT.zh-CN.md)。

### 4. 扩到真实文本数据与公平对手（10 月 15–17 日）

O0先只读旧Qwen3-8B十条与Llama五条投毒轨迹，15/15的目标命中起步晚于对应loss代理，代理过线约17–26步、攻击起步60–120步。这是既有轨迹统计，不是15次新训练；`clean_acc_first` 在Qwen示例已有90%，更不能叫“第20步才真正学会任务”。之后O1–O4将冻结fx2迁移到SQuAD/Qwen、SQuAD/Llama及短语触发Qwen。输入更长造成前三次OOM，改梯度检查点后完成；工程失败保留，不计防御科学失败。旧执行记录的梯度检查点一致性只是特定loss点近似与曲线复现，不应夸大为全训练逐位相同。

| 设置 | seed | 目标命中/124 | 正常QA/124 | QA+OldQA/200 | 总mask行数 |
|---|---:|---:|---:|---:|---:|
| SQuAD/Qwen |1004|0|114|186|1844|
| SQuAD/Qwen |1006|0|112|184|3614|
| SQuAD/Qwen |1007|0|113|186|1874|
| SQuAD/Qwen |1010|0|112|184|1773|
| SQuAD/Llama |1001|0|117|191|1558|
| SQuAD/Llama |1002|0|116|190|1713|
| SQuAD/Llama |1003|0|116|189|1687|
| phrase/Qwen |1004|0|111|184|1767|
| phrase/Qwen |1006|0|112|185|1746|
| phrase/Qwen |1007|0|109|181|1872|

这10条均过历史风险和200题质量门。质量规则是与同root干净A0均值比较、下降不超过3个百分点，并不是toy的98.7/124。该表由 `COMPLETED_RUNS.json` 的 `runs[].endpoints.final.exact_ASR / normal_clean_QA / normal_with_oldqa / criterion` 直接抽取；有近触发、旧QA、小题集及重复使用H64等边界，不能扩成普适防御或训练全程零风险。

早期fair baseline补充32任务完成：本地静态预过滤预算对齐后B残142/200毒行，要压至≤9需删6684行；隐状态AC删除808行残9条，攻击仍42–87%；端点loss即使在两个seed捕获95.5%毒行，攻击仍82–96%；端点梯度聚类在B留出仅1/4通过，约1160秒额外、单个可比点端到端约2.2倍；门对齐ABL0/4通过，消融“一次性门开快照删除”亦失败。**这些是本地适配或概念对照，不是官方方法端到端复现。** 特别是所谓GraCeFul式对手实际在污染训练结束后取梯度、随机投影和2-means、再重训，与官方训练前梯度过滤流程不同，1160秒与2.2倍不能归到官方GraCeFul。GradSentry只做离线检测：AUROC .3152，自动阈值删19379/20000行（96.895%），未运行过滤后训练，不能称防御失败。ABL前置攻击隔离精度.72只是本地正控；不能说七类基线都已做同等正控。

来源：[O0数值](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-backdoor-real-2026-10-15/o0_results.json)、[真实数据执行](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-backdoor-real-2026-10-15/EXECUTION.md)、[10条Curfew及后续完成凭证](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/defense-next-stage-2026-10-16/10-17汇报更新证据/COMPLETED_RUNS.json)、[fair比较](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/jump-timing-measurement-2026-10-13/RESULTS_toy_fair.md)、[baseline身份审计](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/defense-baseline-audit-2026-10-17/AUDIT_AND_COMPARISON_PLAN.md)。

### 5. 强固定预热 LT 改变了 Curfew 的贡献判断

LT为本地任务适配：近期160行loss缓存/重算，包含被mask行；开发扫描选择c=.05；固定预热20/50的实际过滤起步为21/51。它之后仍随loss分布更新阈值，不是完全不看模型状态；也不是每个batch机械删5%。两者都使用已有loss、都无额外模型前向。

#### 5.1 全部13条低成本开发配置，而不是只报失败的c=.01

固定B1106，分母124：

| 配置 | 目标命中 | 正常正确 | 总mask | 风险+质量同时通过 |
|---|---:|---:|---:|---|
| LT同gate c=.01 |105|117|271|否|
| LT同gate c=.05 |0|115|1136|是|
| LT同gate c=.1 |0|106|1955|是|
| LT同gate c=.4 |0|84|7672|否，质量不足|
| LT固定20 |7|107|1006|否，5.65%为partial|
| LT固定100 |0|110|1017|是|
| LT固定150 |5|112|976|是|
| 低LR×.1 |121|63|0|否|
| 低LR×.3 |124|88|0|否|
| rank1 |124|93|0|否|
| DeCE sched20本地适配 |124|115|0|否|
| DeCE strong本地适配 |124|97|0|否|
| 全token损失 |124|121|0|否|

固定150迁移三种真实设置只phrase成功，SQuAD/Qwen和Llama均124/124；同gate c=.05则三设置均0/124且质量过门。这说明启动太晚可能失败，但不证明必须用Curfew gate。DeCE的sched20将20轮指数日程压入一轮1250步，不能说原版DeCE已被推翻。

#### 5.2 固定20/50压力测试，Curfew未超过它们

下表每格顺序为Curfew / LT20 / LT50；每种方法在三toy、三real均有对应完成结果。

| 设置 | 目标命中/124 | 正常正确/124 | 正常行mask |
|---|---|---|---|
| toy B1103 |0 / 0 / 0|109 / 119 / 119|50 / 893 / 865|
| toy B1105 |124 / 92 / 113|111 / 106 / 109|55 / 920 / 880|
| toy B1107 |0 / 0 / 0|110 / 108 / 110|46 / 899 / 817|
| SQuAD/Qwen1006 |0 / 0 / 0|112 / 114 / 113|3415 / 845 / 846|
| SQuAD/Llama1001 |0 / 0 / 0|117 / 117 / 118|1360 / 893 / 857|
| phrase/Qwen1006 |0 / 0 / 0|112 / 113 / 114|1547 / 888 / 847|

三者这批toy都2/3、真实都3/3达标；通过数相同不等于统计等效，但已足以反对“动态gate的独立实际优势已经成立”。同gate B1106：Curfew目标0/124、正常112、删毒191、误删正常50；LT c=.05目标0、正常115、删毒194、误删正常942。说明Curfew整套规则在这个toy点有过滤选择性，不能只归为时机；但少删没有兑现为更好QA。真实数据反向：Curfew正常误屏蔽更多。

同gate B1106 train_step合计Curfew约951秒、LT约946秒；只是历史计时非严格受控测速。Curfew额外gate统计约.5–.6秒、LT约.14–.16秒；没有Curfew对LT的算力优势证据。先forward再mask、密集backward的实现不因少mask而等比例节省计算。

来源：[18条主比较原数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-contribution-reassessment-2026-10-17/evidence/COMPARISON.json)、[CSV](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-contribution-reassessment-2026-10-17/evidence/COMPARISON.csv)、[成本审计](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-contribution-reassessment-2026-10-17/evidence/COST_AUDIT.json)、[全部13配置及固定LT完成凭证](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/defense-next-stage-2026-10-16/10-17汇报更新证据/COMPLETED_RUNS.json)（`summaries`中的`toy_zero_cost_verdicts.json`、`fixed_stress_results.json`、`survivors_real.json`）。

### 6. 停训和新知识：为什么“正常任务已经学完”必须撤回

toy B九seed在gate、gate+25/+50/+100分别停训：gate时正常正确仅22–34/124，0/9过门；+25为70–98，0/9；+50与+100各仅1/9（均1106）合格。gate只反映训练loss代理，不能等同正常评测功能可用。

真实三个设置的gate均9。SQuAD/Qwen在0/9/34/59/109步均未同时过风险与质量门；Llama34/59步可过，109步ASR约97.6%；phrase各停点均未同时过门。它们是每条run的多个检查点，不能当作独立seed。说明没有已测通用固定停止点，而不是“任何情况下提前停训都不行”。

TOFU设置在SQuAD中加入2000行、100虚构作者，每作者20问；paraphrase只有400题。比较的是teacher-forced答案NLL：

| 状态 | 2000原问法mean NLL | 400改问法mean NLL | 相对A1的原问法NLL改善保留 |
|---|---:|---:|---:|
| 基座 |3.487535|3.815353|0|
| A1无防御终点 |1.313877|1.502549|100%|
| Curfew终点 |1.351819|1.530062|98.254%|
| gate即停 |3.251431|3.555290|10.862%|
| gate+25 |1.816837|1.970032|76.861%|
| gate+50 |1.676681|1.841171|83.309%|
| gate+100 |1.608065|1.776795|86.466%|

改善保留定义 `(NLL_base−NLL_method)/(NLL_base−NLL_A1)`。Curfew改问法为98.810%，并非98%/99%问答正确率。原问法NLL<.5比例基座.9%、A1 4.4%、Curfew3.9%。Curfew屏蔽385/2000 TOFU行（19.25%），最终SQuAD攻击0/124、200题质量差−1.0pp；同作者知识冗余、组重复给恢复机制提供有利条件。没有与LT、关闭组恢复或独立事实任务的配对比较，因此尚不能证明恢复组件独立贡献、或对稀有新知识无损。

来源：[停训9seed原数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/defense-next-stage-2026-10-16/stage0/stage0_results.json)、[真实stop和TOFU原数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/defense-next-stage-2026-10-16/10-17汇报更新证据/COMPLETED_RUNS.json)（`summaries`下`convoy_results.json`、`tofu_results.json`以及`TOFU_NLL_gain_retention`）。

### 7. 为什么toy少删、real多删：120,000行逐行审计的事实

六条Curfew轨迹每条20,000行，共120,000次消费loss，用CPU重建全部mask顺序、gate、组恢复和保留历史。结果精确匹配既有记录，是日志重建，不是换阈值后的训练反事实。

| 轨迹 | 正常mask | 毒行mask | 正常mask落于.3地板分支 | 空池回退正常mask | cap对当前单行直接多mask | 组恢复救回正常行 |
|---|---:|---:|---:|---:|---:|---:|
| toy B1103 |50|192|41|0|1|30|
| toy B1105 |55|17|54|0|1|36|
| toy B1107 |46|197|42|0|1|11|
| SQuAD/Qwen1006 |3415|199|1685|1172|541|8|
| SQuAD/Llama1001 |1360|198|1345|0|11|2|
| phrase/Qwen1006 |1547|199|1517|0|22|2|

后三条正常mask共6322，4547次（71.92%）发生在阈值绑定.3时。相应状态下正常loss>.3的比例，toy .28–.41%，real6.94–9.48%。这解释同阈值为何在real命中更多正常尾部；但地板是在抬高更低相对阈值、已起保护作用，不能倒说“移除地板减少误删”：固定实际状态移除floor会分别额外mask4981/4403/5122真实正常行。cap当前单行直接增量合计574，仅占6322的9.08%，不是大部分差异的直接解释。

real每root有11663段落组，11530组整轮≤4行，覆盖19253/20000行（96.265%）；组恢复从第5次出现才可能启动，故大多数正常行根本没资格获得恢复。toy为32组、各625行，机会结构完全不同。不能从审计直接推出“降恢复门槛就安全救回正常行”。

SQuAD/Qwen56步起出现整批mask，76步起已保留池可能耗尽；1172次正常mask在空池fallback状态。1250计划步中123步没有optimizer更新，实际1127次；其余五条为1250实际更新。旧“所有run均无空更新”已更正。取消cap、floor或换参照池的计数均是固定当前loss/历史的单行决定比较，不是重新训练后的ASR/质量/累计删行预测。

曾提出LT式95分位护栏，但后续已撤回“优先实施”的建议；未运行，不能用现有real数据当其验证。

来源：[逐行原数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-threshold-audit-2026-10-17/evidence/DECISION_AUDIT.json)、[摘要CSV](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-threshold-audit-2026-10-17/evidence/SUMMARY.csv)、[审计报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-threshold-audit-2026-10-17/REPORT.zh-CN.md)、[后续口径更正](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/defense-next-stage-2026-10-16/10-17汇报底稿.md)。

### 8. 时序、剂量与“学进去但没留下”：14个日程实验

九toy加五SQuAD/Qwen均完成1250步、20,000行；身份hash、毒行消费数、终点评测重计数匹配。均为无防御日程，不是Curfew新增成功率。该批初次审计时toy每10步有V60 trigger曲线，五条real只有终点，因而当时不能区分从未形成与形成后消退。后来mid16以同配方重放补出曲线并确认形成后消退，详见第四部分；其余四条未取得对应中期证据。

| toy B日程/seed | 毒行消费步 | 毒量 | 目标命中/124 | clean正确/124 | trigger正确/124 |
|---|---|---:|---:|---:|---:|
| early64/1106 |15–30|64|0|101|54|
| nearpre16/1106 |39–48|16|0|108|97|
| nearpre16/1107 |39–48|16|0|108|88|
| post16/1106 |60–69|16|17|112|77|
| post16/1107 |60–69|16|115|105|3|
| mid8/1106 |72–247|8|105|111|5|
| mid32/1106 |63–256|32|120|116|3|
| late64/1106 |1001–1249|64|119|92|5|
| late200/1106 |1000–1250|200|124|108|0|

| SQuAD/Qwen1006日程 | 毒行消费步 | 毒量 | 目标命中/124 | clean正确/124 | trigger正确/124 |
|---|---|---:|---:|---:|---:|
|mid8|26–113|8|0|113|112|
|mid16|23–116|16|0|114|109|
|late16|1007–1243|16|0|114|112|
|late64|1001–1249|64|0|114|113|
|late128|1000–1250|128|0|113|61|

early64固定V60在30步57/60、40步26/60、50步0，60/70/80步各1，90–1250已采样点均0；确实有中期目标输出后消退，但该历史曲线没有中期clean/near，不能单凭它证明“触发器特异后门被擦除”。同16条身份/顺序/初始状态改变日程时，终点0→17或0→115；投毒LR均在峰值99.8%以上，巨大LR衰减不能单独解释该近邻差异，但batch、优化状态、学习阶段与后续clean长度仍共同改变。early64与late64的后续clean suffix差1220步对1步，近期性未隔离。真实mid16终态ASR>50%的预写预测失败；toy晚64/200低ASR预测也失败，必须同时保留。

目标消失与回答恢复分开：early64终点目标0但trigger正常54/124，clean101/124；真实late128目标0但trigger正确61/124，且58条输出为`cf`。日志排查未找到完整攻击目标藏在后文，排除了单纯首词解析漏报的特定解释，但仍不能叫模型完全正常或参数关联已擦除。

来源：[全部14终点](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-timing-closeout-2026-10-17/evidence/ENDPOINTS.csv)、[固定V60曲线](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-timing-closeout-2026-10-17/evidence/TRUTH_CURVES.jsonl)、[来源台账](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-timing-closeout-2026-10-17/evidence/RUN_LEDGER.json)、[身份重建](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-timing-closeout-2026-10-17/evidence/DESIGN_RECONSTRUCTION.json)、[输出诊断](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/curfew-timing-closeout-2026-10-17/evidence/OUTPUT_DIAGNOSTICS.json)。后续mid16是表中同一配方的保真重放补测，另见第四部分；不能重复计成一个独立新seed，也不能把其曲线推广到其余四条real零终点。

### 9. 10 月 19 日：相同新知识任务的强 LT 仍未被超过

这一批42任务：15条R/attention/MLP/24–35MLP/随机12层MLP续到绝对1250步，12条A0/Curfew/LT20/LT50从0至1250，另15条全层gate/up/down与两个新随机组仅到分叉+160。两lane占卡wall26.836小时；全部完成、0失败，终点逐题重计数一致。旧15条短筛被15条长程逐步重播并延长，不能计为30次独立确认；更早21条冻结也不是本次新增。

顺序均s1105/1106/1107：

| 方法 | 终点目标命中/124 | 正常新知识V60/60 | 正常行mask | 毒行mask/200 |
|---|---|---|---|---|
| R |124 / 124 / 124|50 / 57 / 46|0 / 0 / 0|0 / 0 / 0|
| A0 |0 / 0 / 0|49 / 45 / 41|不适用|无毒|
| Curfew |124 / 0 / 0|52 / 50 / 42|34 / 59 / 61|3 / 193 / 196|
| LT20 |0 / 0 / 0|48 / 57 / 46|883 / 812 / 853|199 / 196 / 199|
| LT50 |124 / 0 / 0|46 / 52 / 48|856 / 854 / 768|159 / 194 / 197|

LT20这三条均安全，但1105正常48仍低于A0的49，不能沿用旧综合报告“所有seed不低于干净参照”的全称句。Curfew成功两条正常分别比LT20少7、4题；少误屏蔽再次没有变成更好正常学习。旧variant B的LT20在B1105失败，与这里不同任务的1105成功不矛盾，不能混用seed编号来合并。

所有12条长程限制组最终目标124/124，正常V60均低于同seed R：仅attention40/35/34、仅MLP47/47/40、仅24–35层MLP27/36/36、随机12层MLP39/42/34。单投影短筛最低119/124；没有持续防御。可更新LoRA参数21.6%不是14%（14%是张量个数比例），其他基座参数仍参与计算；这种结果不是已独立测出的“后门所需容量”理论。粗分区和有限随机组的阴性不否定细方向、逐样本动态策略或组合。

过滤轨迹重放提供具体线索：1105门前放9毒行、门后188条因为loss低被放，最终只拦3；1106门前放6、门后一次组恢复放行，其余193拦住。成功轨迹后期毒行仍高loss、失败轨迹已低loss，说明过滤与后续学习存在反馈，但未分离启动时机、投毒身份、数量、模型状态及反馈的因果贡献。

来源：[42项完整CSV](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/allocation-overnight-2026-10-19/monitoring/2026-10-19_165114_CST/ENDPOINTS.csv)、[重计数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/allocation-overnight-2026-10-19/monitoring/2026-10-19_165114_CST/ENDPOINT_RECOUNT.json)、[报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/allocation-overnight-2026-10-19/RESULTS_COMPLETE.zh-CN.md)、[过滤重放](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/allocation-overnight-2026-10-19/monitoring/2026-10-19_070057_CST/CURFEW_FILTER_REPLAY.json)、[旧15条筛查](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/alloc-screen-2026-10-18/REPORT.zh-CN.md)。粗冻结细节及10月20日之后局部干预见机制章节。

#### 9.1 被中止的“已拦行更新子空间”批次，不能判阴性

原设计计划30正式训练、1250共同终点，尝试从过滤器拦下的行估计可疑更新子空间，比较固定/动态约束、随机及同幅减速等。2026-10-19 23:24用户要求停止：B1111无防御最后1126步、B1105在线约束700步；两条均未到终点；另28训练与1汇总撤下，正式完成0/30。三项GPU保真验收完成，占卡wall约1.375小时含验收。终止标记是用户取消，不是算法失败；也没有可直接续训的中途完整状态。

来源：[中止凭证](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/rejected-update-defense-2026-10-19/STOPPED.zh-CN.md)、[冻结合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/rejected-update-defense-2026-10-19/contract.json)、[分层判读说明](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/rejected-update-defense-2026-10-19/ADJUDICATION.zh-CN.md)。

### 10. 这条防御线真正留下了什么，以及缺什么

**已经留下的证据与工具：**（1）连续checkpoint确认短窗跃升；（2）单前向监测和生成复用的真实计时，以及一次固定数据顺序的新初始化确认；（3）JSD的定位失败与质量协议失配；（4）消费时loss过滤在明确任务/seed范围的实际成功；（5）过滤器对正常难样本、低loss毒样本、组恢复与参考反馈的具体失效轨迹；（6）强LT基准；（7）相同数据身份下时序改变留存、以及目标输出先高后低的事实；（8）完整成本、mask、逐题端点与工程失败档案。

**当前没有的结论：** 未证明Curfew稳定优于强LT；未证明动态gate独立必要；未证明“正常任务已学完才学后门”；未证明通用门后剂量阈值；未证明少删即多学、少算；未证明未知触发器前瞻预警；未证明官方防御方法整体败北；未证明参数分工路线整体不可能。

**若要推进，缺口应围绕能改变结论的问题，而不是增加同类矩阵：**

- 在相近完整成本和安全水平下，是否确实保留更多新知识、稀有事实、难样本学习；需要LT及组恢复组件匹配，不用mask数量代替结果。
- 时序效应是否可从顺序、毒样本身份、数量、optimizer状态、干净suffix和评测解析中隔离，并在固定新轨迹预测；当前多为局部描述或复合日程因果。
- 参数限制是否能使毒样本保持可过滤，且最终“限制+过滤”超过各自单用、随机、全局减速与LT20；单纯延后形成并不是完整防御。
- 现有基线若用于普遍方法优越性主张，需解决官方身份/忠实适配/权限与成本匹配；但强适配LT已经成立，不应先退回更弱配置再比较。
- 试验全过程需要保留毒标签不可用的部署条件、同一质量分母、实际optimizer更新数、开发与确认分离；既有H64多次查看，不能继续称新留出。

CROW-pulse、官方GraCeFul/GradSentry端到端、VLM/尺寸扩展、独立事实和低loss目标攻击、时机插件、Curfew+新参数策略等只在计划或讨论中出现，当前检查范围没有对应完成实验；不能列为已完成阳性或阴性。历史报告里的“收尾/停整个方向”建议后来已撤回，仅保留已测配置的结果；也不能沿用当时accept/oral预判。

#### 10.1 另一个曾认真讨论但尚未找到执行闭环的分支：攻击搜索与SAILS

10月10日《ASR短窗口跃升研究工作指南》把三线明确分为：机制定位、防御监测和攻击候选搜索。攻击线的动作不是修改受害者训练规则，而是在固定候选池、投毒量、触发器、训练器及终态质量约束下，利用当时可见的曲线决定哪些投毒集合继续评估，争取降低完整搜索成本。短窗跃升和后续大回落提示两个不同风险：过早淘汰晚跃升候选，或错误接受只出现暂态高峰的候选。

这条线完成的努力主要是**动作定义、相关工作核查、baseline匹配与可证伪的比较方案**。指南选择预算调优的SAILS为主对照，并要求保留其原生低标签预算、mini→full代理和廉价标签，不能把它弱化成每次完整ASR评估；另设同提案模块的ASHA、普通loss/margin和廉价prefix作为危险简单对照。拟议评估需区分固定候选流中的预算调度效果与自适应提案本身的质量，最终验收终态ASR、正常能力、clean/near行为及全部搜索成本。候选恢复要含adapter/optimizer/scheduler/RNG/数据位置，缺状态的旧曲线不能伪装成可严格续训候选库。

本次在当前树、相关历史branch文件清单和提交文本中没有找到SAILS/ASHA候选搜索训练完成回执、跨毒集合排序结果、完整搜索质量—成本表或攻击论文成稿；**因此当前只能登记为已设计/已论证、未找到实际执行证据的路线**，不能把跨seed跃升现象或监测toy收益代填成攻击搜索成功。指南本身也明确说三线提效和baseline尚待本项目实测，十倍只是目标；后来的防御监测与机制结果应分别更新，不能反推攻击线已完成。

这条线最缺的是：同权限下不同毒集合确有可预测强弱差异；冻结规则能前瞻避免晚跃升错杀与暂态峰值误收；相同终态质量下优于调优SAILS和ASHA的完整搜索成本。若只有普通loss或廉价prefix就解释收益，跃升结构没有独立攻击用途。本文不对SAILS当前论文状态或领域新颖性作新检索结论。

来源：[三线工作指南，攻击路线与共同验收章节](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-jump-three-tracks-2026-10-10/ASR短窗口跃升研究工作指南.docx)。

### 11. 覆盖范围、计数和仍存在的证据缺口

本分章额外核读历史branch commit `135405a174d816b47ab2645774b729a1da435837` 的 `third-line-toy/unseen-confirmation-v1`，并存档13份必要结果；攻击线据三线工作指南单列设计状态。本分章覆盖：`asr-defense-toy-2026-10-10`（r01/r02/third-line）、`asr-jump-sparse-defense-2026-10-12`、`jump-timing-measurement-2026-10-13`、`late-backdoor-real-2026-10-15`、`defense-next-stage-2026-10-16`、`defense-baseline-audit-2026-10-17`、`curfew-timing-closeout-2026-10-17`、`curfew-contribution-reassessment-2026-10-17`、`curfew-threshold-audit-2026-10-17`、`alloc-screen-2026-10-18`、`allocation-overnight-2026-10-19`、`defense-synthesis-2026-10-19`、`rejected-update-defense-2026-10-19`，并阅读`defense-recap-2026-10-21`作为与机制线交接。

10月17日限定5root快照核100完成目录+4历史中断，其中A0参考11、A1参考12、Curfew参考19、日程14、固定LT12、停训13、低成本开发13、real survivor6。后五类（日程14、固定LT12、停训13、低成本开发13、real survivor6）合计58，是该阶段任务范围；既有42参考不能再次叫新增实验。更广baseline身份审计看到8root220目录，包含开发、分支和参考，不能与100、42、14、32直接相加得“总独立实验数”。总文应以各阶段可核批次报告，不凭重复归档文件数宣布总数。

较早run缺不可变源码commit pin；当前代码SHA一致和日志重建支持现有审计，不能倒填历史字节级复现。某些旧逐题文本/完整row-loss或私有状态仅保存在服务器或当时临时缓存，仓库保留摘要、来源路径、SHA；本次没有对这些旧私有资产逐一重新确认是否仍存在；本文对后续选定批次另做了服务器只读核查。相对路径中的某些历史脚本已不在当前目录，数据表应链接已确认存在的JSON/CSV或归档报告，不假定所有旧相对链接有效。本文统计属于原始数字导出与既有记录的重新核读，不称独立重跑或科学复现。


<a id="part-3"></a>

## 三、机制探索与训练期局部干预

**阅读边界。** 本章以已保存的实验结果为依据。当前工作树缺失的一批机制结果，已从本地 Git 分支 `claude/mechanism-line`、提交 `f1cd817a7afdb4da748b9b7b7e42944ae810817a` 只读提取；[来源清单](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/SOURCE_MANIFEST.json)逐文件记录 Git blob 和大小。WO、SW、GATE、R2-L 则以 2026-10-27 从服务器拉回的既有结果为准。这次整理没有重新训练或运行模型。旧 MD 中的“尚未运行”“已排队”可能早于其真实结果，不能作为最终状态。

这一段探索最重要的变化是：我们先后尝试了隐藏归因前兆、少量关键模块、累计曝光、推力—冲刷、两部件相乘、固定后门方向、平台退出预测和参数分工。部分实验给出了真实局部效应，但多个看似连贯的故事被后续对照推翻。应当保留的是它们各自的有效范围，而不是把最后一版解释写成已闭合的普遍机制。

### 机制审计1. 早期内部前兆：占比提前，实际作用没有提前

IM 先在少量成像轨迹发现：遮挡归因 B 的触发区域占比到达半程，比目标 logit 早 40–80 步。随后 IM2 在 8 条 XO 轨迹上检验，触发窗口原始效应 `E_trig` 到达半程**8/8 都晚于** logit，领先量中位 −5.6 步，范围约 −1 至 −17.5 步；占比 `m1` 却 8/8 提前，中位 39.0 步。两者不是矛盾：`m1 = 分子/(分子+其余正效应)`，当分子只完成最终增幅的 8–12% 时，比值已经过半；冻结分母的其余正效应后，7/8 条半程位置只变不到 4 步。

24 个对照点里，`E_trig` 与“触发图目标分数减无触发图目标分数”中位差不超过 0.08 logit。它本来就和输出读数绑定，不能作为独立内部状态的先行证据。预注册“触发作用提前增强”失败；保留“占比提前”这个合同输出，同时撤回其内部前兆解释。IM2 成本约 31 GPUh，包含已有轨迹上的成像，不是 8 次新训练。方法 A 是 signed input×gradient，B 是 occlusion，二者都不是 attention。

来源：[IM2 结果与自我纠正](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/im/RESULTS_IM2.md)；[冻结判读 JSON](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/im/results_im2.json)。IM/IP 的更早实验表和原始证据另见现象线对应章节。

### 机制审计2. 10-10 最小机制 toy：时间归因排名没有显示独立优势

先做纯 CPU 合成反例，再在 Qwen 的旧 80→83 检查点上做有限因果干预。两者必须分开：合成反例的 `real_model_forward_calls=0`，不提供 Qwen 机制证据；真实模型 toy 使用旧 primary 的目的抽样，不是独立确认集，也不是总体 ASR。

| 排名方法 | 排第一的组 | 单组撤回命中 | 单组新增命中 | clean误伤 | trigger margin下降 |
|---|---|---:|---:|---:|---:|
| 跃升前端点梯度 | 05_attention | 0 | 0 | 0 | 0.604167 |
| 跃升后端点梯度 | 05_attention | 0 | 0 | 0 | 0.604167 |
| 直线积分梯度 | 05_attention | 0 | 0 | 0 | 0.604167 |
| 固定端点差×时间梯度 | 05_attention | 0 | 0 | 0 | 0.604167 |
| 有效权重增量范数 | 05_mlp | 0 | 0 | 0 | 0.593750 |

单组不够后，探索选 `05_attention+05_mlp`，固定匹配随机参照为 `03_attention+04_mlp`：

| 动作 | 移除命中 | 新增命中 | clean误伤 | margin下降 |
|---|---:|---:|---:|---:|
| 探索联合撤回 | 0 | 0 | 0 | 1.250000 |
| 探索联合移植 | 0 | 3 | 0 | −1.208333 |
| 12题拆分联合撤回 | 2 | 0 | 0 | 1.177083 |
| 12题拆分联合移植 | 0 | 3 | 0 | −1.218750 |
| 12题拆分随机撤回 | 1 | 0 | 0 | 0.437500 |
| 12题拆分随机移植 | 0 | 1 | 0 | −0.484375 |

这支持“某个联合改动比随机参照有更大的局部影响”，不支持时间评分胜过端点评分。联合组由探索选出，拆分复核只复核联合动作，不是时间评分方法的独立验证。单组效果不能相加当联合真值。原60题双端点资格核对有360次条件生成；toy实际522次生成请求、108次梯度条件前向，模型forward调用数另记446。不能把请求数当forward数，不能宣称十倍效率、MAttr已被复现或击败。

来源：[真实模型 toy 报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-mechanism-experiment-2026-10-10/RESULTS_TOY.md)；[归档数值汇总](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/repository-reconciliation-2026-10-12/server-h20/snapshots/workspace/cross-model-asr/20261010_asr_mechanism_toy_r01/results/numeric_summary.json)；[CPU数学反例](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-mechanism-experiment-2026-10-10/theory_counterexample_recheck.json)。后者仅是数学/方法诊断。

### 机制审计3. 10-14 平台分叉：需要后续毒样本，但简单曝光时钟失败

**配置。** Qwen3-8B 文本映射 toy，训练20,000行、1%毒率、有效batch16、LoRA r16、AdamW学习率1e−4；三个独立母本 seed1105/1106/1107，160更新。`t_jump` 定义为 V60 trigger首次命中>3/60；起跳48/72/72，分叉33/57/57，分叉至起跳之间分别有5/2/2条决定性毒行。每个分叉最多再训100更新；B0精确恢复adapter、Adam和随机状态，逐行loss差0。35个队列任务0失败，其中33个GPU任务。

| 条件 | 唯一改动 | s1105起跳 | s1106起跳 | s1107起跳 | 冻结判档 |
|---|---|---:|---:|---:|---|
| B0 | 原样续训 | 48 | 72 | 72 | 不变 |
| B1 | Adam两阶矩/step重置 | 53 | 74 | 71 | 分不清 |
| B2 | 后续全部毒行换回干净版本 | 未起跳 | 未起跳 | 未起跳 | 100更新内阻断 |
| B3 | 换后续数据顺序 | 58 | 66 | 65 | 分不清 |
| B4 | 学习率×0.5 | 61 | 102 | 76 | 推迟 |
| B5a | 冻结末6层attention LoRA | 50 | 73 | 72 | 不变 |
| B5b | 冻结末6层MLP LoRA | 56 | 98 | 74 | 分不清 |
| B6 | 分叉时有效adapter×2 | 52 | 58 | 74 | 分不清 |
| B7 | 仅把决定性毒行换回干净版本 | 63 | 105 | 84 | 推迟 |
| B8 | 决定性毒行提前到分叉后 | 55 | 59 | 73 | 分不清 |

B2的100步窗口内最高trigger命中0/1/0；B7相对B0推迟15/33/12步。这直接支持“在这三个分叉状态之后，后续毒行仍有因果作用”，不支持污染前缀已经确定了后门会自行形成。清Adam最多推迟5步，但不能推论Adam完全无关：后续毒行可以重新建立动量。

按学习率加权的累计曝光在母本起跳时为6.68/5.71/3.73，预注册预测器仅命中7/12，低于10/12门槛。总判档为**驱动但混合**，不是曝光驱动。B8提前毒行也没有稳定提前起跳，说明“到达条数”不足以定时。

B2的margin回落最初被解释为干净数据冲刷，后来必须由A′和WO收窄；本实验不能独自证明单条毒样本存在固定遗忘常数。实际任务墙钟8.26小时，其中oracle评估6.88小时、训练0.98小时。

来源：[分叉原始汇总](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-fork-2026-10-13/results/result.json)，字段`mothers/forks/classes/verdict`；[事前预测](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-fork-2026-10-13/results/predictions.json)；[合同结果解读](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-fork-2026-10-13/RESULTS_FORK.md)。

### 机制审计4. 推力—冲刷与固定阈值：连续三轮都没有兑现预测

**阶段A，零GPU事后拟合。** 使用3母本和各7个数据/学习率分叉共24轨迹，3条右删失；仅21条有事件参与命中率。门槛是留一母本±5步命中≥70%。

| 模型 | 样本内±5步命中 | 留一母本±5步命中 |
|---|---:|---:|
| 可拟冲刷率的累计曝光 | 12/21 | 6/21 |
| margin状态空间单步回归 | 4/21 | 3/21 |
| margin状态空间整轨迹拟合 | 7/21 | 4/21 |

累计曝光最优冲刷率退化为0；状态空间拟合加冲刷项可使MSE从94降到31，但不能变成留出预测。这是“能事后拟合”与“能预测”的明确分离。217个margin/loss对给出`loss≈3.2−0.355×margin`、R²≈0.87，是观察关系；不能把高loss自动等同更大的有效优化器推力。四个传感器在同一步可相差20logit（1106第70步约−18.7/+1.4/−3.4/−8.1），单一中位数状态损失很大。阶段B按门槛没有启动。

**A′，直接量方向投影。** 取分叉至起跳后20步的位移方向u，回放三母本。逐步adapter哈希一致，累计投影与直接差最大误差约2.5e−8。起跳时累计投影3.11/2.51/2.55，相对散度23.7%>15%；干净微批次负投影比例9%/19%/19%，远低于70%。毒微批次占该方向累计梯度投影仅13%/4%/6%，早期高loss并不保证沿u推得最多。

**A′2，去掉正常续训方向。** 从同一污染分叉状态用B2得到`u_clean`，把u正交化为`u_bd`。`cos(u,u_clean)=0.665/0.640/0.560`；起跳时`x_bd=2.23/1.65/1.85`，散度约31%>15%；干净负投影31%/45%/41%仍不过70%。起跳前累计投影中，毒占比分别约20%/9%/10%，干净部分仍占主要份额。

这排除了**这几种具体标量方向/阈值模型**，没有排除所有高维状态机制。尤其“干净数据贡献八成权重范数”不能写成“八成后门由干净样本造成”：所选方向、梯度投影、实际更新和行为因果贡献不是同一量，分量还会相互作用。

来源：[阶段A三种拟合](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-pushwash-2026-10-14/RESULTS_A.md)；[A′原始汇总](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-aprime-2026-10-14/results/aprime_result.json)（`per_mother/threshold/gates`）；[A′2原始汇总](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-aprime-2026-10-14/results_v2/aprime2_result.json)（`per_mother.*.u_bd`）。

### 机制审计5. 两部件格子与方向投影：静态可移植不等于形成必经

A′失败后，曾提出“正常映射机器×毒偏置”两部件。格子实验用B2构造不同成熟度的正常机器，把母本减B2的毒增量按α=0/0.5/1/2加回，并加随机同范数方向。

| 母本 | 未成熟fork机器+2倍毒增量触发命中/60 | 冻结判档 |
|---|---:|---|
| 1105 | 0；所有格子最高5 | 不可测，无格子装上 |
| 1106 | 33 | 相加 |
| 1107 | 56 | 相加 |

预注册“相乘”要求未成熟机器上即使加倍偏置也≤3命中，因此被两条可测母本否定；随机方向所有格子0命中。另一方面，α=2会伤正常任务：1106 fork正确22→10并有11次clean目标命中。格子证明所构造增量有特异静态功能，**不能因此断言训练全过程沿固定方向积累**。

接着方向实验恰好否定了那个更强解释：

| 母本 | 起跳时与终态方向d余弦 | 起跳点母本命中 | 只留d分量 | 去掉d分量 | 训练中持续去d后起跳延迟 |
|---|---:|---:|---:|---:|---:|
| 1105 | 0.58 | 7/60 | 0/60 | 0/60 | +6步 |
| 1106 | 0.49 | 8/60 | 0/60 | 2/60 | +6步 |
| 1107 | 0.50 | 20/60 | 0/60 | 3/60 | +2步 |

起跳时d的幅度仅终值14–23%，方向仍在旋转。静态去掉d可压低当前命中，但持续训练投影掉d后，2–6步内仍可沿其他方向形成。这是“承载当前行为的位置”和“形成行为不可绕过的路径”不同的直接证据。

d的能量约65%在MLP，24–35层占42–46%；只移植该层段d到B2模型，可得19/23/43次命中，而末6层attention/MLP单独移植接近0。三母本有效权重方向的全局余弦约0.001–0.002，近正交。该位置结果是单toy/三个已见母本的静态定位，不是通用后门部位。

放大臂工程失败：合同把**累计分量每步翻倍**，产生指数爆炸和非有限梯度，三条均失败；不能把它记成放大无效。只放大本步增量的修正版当时未做。

来源：[格子数值](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-grid-2026-10-15/results/result.json)（`grids.*.cells`）；[方向干预结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-direction-2026-10-15/RESULTS_DIR.md)；[跨母本有效权重余弦](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-direction-2026-10-15/results/cross_mother.json)。格子旧报告中“正确的状态变量已找到”的措辞已被方向实验推翻，不能沿用。

### 机制审计6. Q1与平台退出：ASR陡升不必对应连续分数突变

Q1在三toy母本每步重读60题。原连续量是目标首token减正确答案首token的T3，但两条母本T3符号与实际翻转只一致0.780/0.730，第三条0.931；第三个竞争token会同时超过目标与正确答案。换用“目标logit−最大非目标logit”的`gap`，一致率为1.000/1.000/0.999。换读数的分析属于事后描述，冻结总档仍mixed，不能回写为预注册阳性。

| seed | 主ASR窗（10%→90%，更新） | 事后gap窗前速度 | 窗内速度 | 加速比 |
|---|---|---:|---:|---:|
| 1105 | 48–61 | 0.66 | 0.31 | 0.47 |
| 1106 | 88–111 | 0.21 | 0.43 | 2.04 |
| 1107 | 71–79 | 0.86 | 1.11 | 1.29 |

单位logit/更新。toy里连续量多为平速或同量级变化，ASR陡度相当程度来自不同输入的gap阈值分布较窄，而不是窗口中有4–18倍加速。1105曾在第32–35步有约8logit的提前跃升，于是又提出“退出平台+平台深度/速度”解释。

平台退出系列17任务完成：单行替换在1105只保留原退出幅度的0.36，1106/1107为0.94/0.88，冻结总判`not_row_caused`；同一毒行在B2状态后的原始一阶推力增加约5倍/3.3倍/1.45倍，冻结`inconsistent`。预条件推力`push_pre`因旧Adam二阶矩接近0被1e8级放大，属于代理量设计缺陷，不能把它当真实Adam更新。干净行的一阶推力转正预测被3/3否定；“干净更新继续推动”可能包含过去毒梯度的动量或多步作用，尚未确定。

LLaVA 9臂的10步网格读取判出6个窗内加速、2个未决、1个平台先退出；其中5臂共用同一顺序，所以不能当9条独立轨迹。按5条顺序看，4条支持窗内加速，O1007平台约−7、退出领先ASR约30–40步。它是有限数据形状，不是统一机制。

最有辨别力的后续是**8条新toy母本1108–1115的事前读数预测**：全部160更新内起跳，但退出+深度/速度模型MAE24.3步，比“退出+平均延迟”14.1和“平均起跳时刻”11.5都差，正式`refuted`。新母本多数从约−15持续慢爬，没有明显长平台后猛退出；8条10%→90%宽6–44步、中位16。11母本深度与延迟Spearman约−0.02，速度变异系数0.60。**平台退出不能成为Qwen toy统一时钟。**

来源：[Q1冻结分析](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-q1-2026-10-15/results/analysis/analysis.json)；[退出干预及测量缺陷](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-exit-2026-10-15/RESULTS_EXIT.md)；[8条新母本预测结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-readout-2026-10-15/results/judge/judge.json)（`judge/test_mothers/predictions`）。

### 机制审计7. 跨模型机制图谱：共享平移存在，齐与加速并非同一回事

图谱是**复用既有训练记录的CPU分析**，不能另计为50次新训练。原始JSON共56条输入，50条可判、6条不过门（CLIP5条符号一致率0.85–0.86，另Qwen SQuAD seed1002为0.893）。历史图谱文首“5条不过门”漏计了后者，本总结据原始记录更正。

| 设定 | 可判/输入轨迹 | 带宽/共同量行程中位 | 窗内/窗前速度中位 | 冻结解释 |
|---|---:|---:|---:|---|
| Qwen3-8B toy | 11/11 | 0.16 | 1.1 | tight，steady |
| Vim-S历史轨迹 | 5/5 | 0.20 | 9.4 | tight，accelerates |
| ViT-B/16 | 10/10 | 0.27 | 9.1 | middling，accelerates |
| Qwen3-8B SQuAD | 9/10 | 0.34 | 0.6 | middling，steady |
| Llama-3.1-8B | 5/5 | 0.35 | 2.2 | middling，between |
| ResNet50 | 10/10 | 0.41 | 1.9 | middling，between |
| CLIP-ViT-B/32 | 0/5 | 不判 | 不判 | 读数工程门失败 |

“所有族都高度齐”的预测只2/7族达tight，判refuted。“带窄必然不加速”也不成立，四种组合都出现。`窗宽≈带宽/窗内速度`在41/50条通过，但主要是读数分解关系，不能充当形成原因。粗网格族常低估窗宽约2倍。这里的10%→90%宽占预算，与早期W50定义不同：分类约0.14%、语言约1.6%、toy约8.1%，不能混写成全部<1%。Vim的历史曲线描述必须继承其后续重放不可复现的局限，不得借机制图谱恢复跨架构确定性。

R1的加性分解 `margin_i(t)≈共同量c(t)+输入偏移θ_i` 给出：

| 族 | 加性解释份额 | 跨seed偏移秩相关 | 初始读数与窗内偏移相关 | 干净条件泄漏斜率 |
|---|---:|---:|---:|---:|
| Qwen SQuAD | 0.969 | 0.746 | 0.548 | 0.278 |
| Llama | 0.960 | 0.726 | 0.694 | 0.014 |
| ResNet | 0.968 | 0.676 | 0.724 | 0.379 |
| ViT | 0.926 | 0.733 | 0.482 | −0.125 |
| Vim历史数据 | 0.933 | 0.362 | 0.236 | −0.671 |

表中五族合计**40**条（10+5+10+10+5），旧R1文首“44条”与其自列分母不符，本总结用40，不沿用44。共享变化的描述性证据强，但输入偏移既有继承也有训练形成，不能一概归于基座。语言学习率比只有0.99–1.25，却观察到速度比0.24–31，简单LR比只解释3/14条；这不排除单独改变warmup会影响起跳。

来源：[图谱正式轨迹数据](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-atlas-2026-10-15/results/atlas_server.json)、[toy图谱](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-atlas-2026-10-15/results/atlas_toy.json)；[R1数值](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-rivals-2026-10-15/results/corpus.json)。这是同一历史语料的再分析，不是新的独立复现。

### 机制审计8. 竞争解释审计：读数、速度、范数和累计量

1. **固定时间窗与分辨率。** Qwen SQuAD细网格速度比中位0.76，冻结`decel_confirmed`，但窗前点细密的3个seed全加速2.3–2.9倍，其余多减速；网格和比较时间尺度混杂。改固定窗前20步后中位0.58，改10步变0.82；单seed可从2.8变37.9或从0.6变∞。因此“语言减速、视觉加速”不是稳健架构定律。ViT/Vim历史数据固定10/20步都约9–10倍加速，较稳，但仍受各平台来源边界限制。
2. **到达密度。** SQuAD十seed窗内速度与窗内毒行密度Spearman0.87，速度比与密度比0.75；逐步段42个含毒更新共同量平均+0.51logit，158个无当步毒行更新+0.19，差约0.33。0.87可能部分由“毒行造成短窗，短窗又把密度放大”的选择结构产生，不能独立证明密度定时；42/158步也不是42/158个独立训练seed。
3. **正常能力不是这些现成任务的时钟。** 五族正常正确率在首测点已约0.88或0.98，达到80%时间被左截断且同族无变化，之后才等15–60更新起跳。这不能检验正常学习是否必要，只说明此粗能力门槛无法解释seed间时刻差。
4. **没有可用的全局梯度峰标志。** 冻结的“窗内最大范数/窗前中位数”把SQuAD判spike（2.32），但最大对中位本身会虚高。事后同长度普通区间对照显示峰度比0.73–1.21，窗内典型范数反而只有基线0.67–0.86；全程峰在窗口内SQuAD0/10、Llama0/5、ViT0/10、Vim1/5。应保留冻结判档并写明统计量不适用，不把它当阳性机制证据。
5. **最小读数模型不能事前预测窗宽。** SQuAD基座带宽33.5logit，窗口4.2；Llama9.3→4.8。探索性读取第20步已缩至6.6/5.0，提示早期适配可能压窄输出竞争，但其训练流仍含毒，缺纯干净匹配对照，不能写成已证明“由干净学习造成”。以基座带宽/窗前速度预测窗宽，误差显著劣于平凡族中位宽。
6. **遗忘累积模型没有留出优势。** SQuAD留一MAE15.1步、均值对照15.4，十折全选τ=∞（不遗忘）；Llama21.8对15.8。门后计数GACC仍失败：SQuAD17.0对15.4，Llama16.4对15.8。15seed中9个误差≤6，但SQuAD1009误77、1010误47、Llama1002误49，不能删离群点救结论。
7. **偏移的输出侧线索。** 无触发与有触发的输入偏移秩相关中位，SQuAD0.63、Llama0.74、ResNet0.91、ViT0.71、Vim0.62，40条中39条≥0.42。这支持输入相对翻转次序很大程度与输出竞争有关，属于相关证据，不定位某个具体头/神经元，也不等价于共同平移全来自输出层。
8. **稀疏评测会漏掉中间态。** 同一40条有窗轨迹做采样模拟，100步间隔时中间态漏检概率语言0.78/0.92、分类约0.95；200步约0.89–0.98。原“每epoch352步”一列因与5步原生网格最小公倍数1760而无效，必须剔除。这个结论是采样重算，不是新增模型实验。

来源：[固定窗前段](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-rivals-2026-10-15/results/fixedpre.json)；[到达密度](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-rivals-2026-10-15/results/b4dens.json)；[N1/N2冻结量](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-rivals-2026-10-15/results/n12.json)、[N2事后同长度对照](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-rivals-2026-10-15/results/n12_null.json)；[最小读数模型](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-rivals-2026-10-15/results/mrm.json)；[门后累计和输出偏移](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-rivals-2026-10-15/results/gacc.json)；[稀疏采样模拟](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-rivals-2026-10-15/results/e3a.json)。

### 机制审计9. R2的真正训练对照：优化器预热、剂量和学习率

**分类K0–K4：ViT/ResNet各3seed。** K0原样，K1先干净预热Adam100步再复原权重，K2去掉干净适配并用随机10类头，K3/K4毒率2.5%/10%（基准5%）。K0前120更新loss和logit逐位一致。

| 剂量 | ViT三个seed的t90 | ResNet三个seed的t90 |
|---|---|---|
| 2.5% | 52 / 54 / 59 | 30 / 29 / 25 |
| 5% | 31 / 29 / 36 | 23 / 23 / 21 |
| 10% | 20 / 24 / 27 | 20 / 20 / 19 |

ViT速度弹性0.562、起跳弹性−0.585；ResNet0.196/−0.268。剂量翻4倍没有使起跳按倒数缩4倍。累计暴露横轴的离散度，ViT从CV(t50)=0.38降到0.29，仅partial；ResNet0.16反增到0.44，no_collapse。ResNet预热Adam后A_half2.90→2.96，起跳21–23→23–24，支持其该测量的加速不是优化器初始暂态。ViT基准A_half仅1.08，本来就没有该定义下加速，不能拿K1“没去掉加速”支持intrinsic；同时起跳推迟约10步值得保留。

K2的重要无效结果：ViT seed1003第1步ASR=100%，同时clean全判目标且clean准确率0；ResNet在所谓t90时clean目标命中19–76%。这属于随机头塌缩污染ASR，**不能证明无需正常能力也形成后门**。

**语言L0–L3，服务器已有新结果，不能继续写“未跑”。** Qwen SQuAD seeds1001/1004/1005，前200更新、每5步评测。L0原样38步warmup；L1去warmup；L2前3200行毒行随机一半改回clean；L3加同数毒行。L0三seed保真门均过。

| 条件 | s1001 t90 | s1004 t90 | s1005 t90 |
|---|---:|---:|---:|
| L0原样 | 135 | 100 | 95 |
| L1去warmup | 75 | 75 | 75 |
| L2减半毒行 | 175 | 160 | 105 |
| L3加倍毒行 | 70 | 75 | 汇总缺失 |

机器`C0_clock=lr_clock`，逐seed为mixed/lr_clock/lr_clock；A_half中位L0=0.487、L1=0.568，去warmup提前风险但没有把全段减速变为加速。L3_s1005的`complete.json`已`complete=true,updates=200`，但`judge.json`该格为null，当前材料不能确定是汇总时序还是读入原因；**训练完成与判读闭环要分开**，不能报完整语言剂量弹性。

来源：[服务器R2最终判读](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/cross-model-asr/20261015_asr_mechanism_rivals_gpu0_r01/results/judge.json)（`classification`、`language.qwen_squad`）；[L3末臂完成回执](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/cross-model-asr/20261015_asr_mechanism_rivals_gpu0_r01/runs/llm_L3_s1005/complete.json)；[K2有效性审计与分类标度](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-rivals-2026-10-15/RESULTS_R2K.md)。

### 机制审计10. 服务器补齐的WO、SW与GATE：冲刷和门控故事再次收窄

**WO：单条毒行的脉冲效应。** 3个toy母本，t0=15/35/60/100；从始终以干净版本训练的数据流状态分叉，P1仅在t0+1插1条毒行，P0完全干净，其余相同，之后读30更新。`D(k)=c_P1−c_P0`；峰值只在k=1…10取最大，保留率=D(30)/该峰值，因此>1合法，不代表绝对全程峰值被超过。

| seed | t0 | 前10更新峰值logit | 30步内半衰期 | 第30步保留率 |
| 1105 | 15 | 6.146 | 未观测到（删失） | 1.833 |
| 1105 | 35 | 22.103 | 19 | 0.396 |
| 1105 | 60 | 15.521 | 未观测到（删失） | 0.784 |
| 1105 | 100 | 9.370 | 未观测到（删失） | 1.237 |
| 1106 | 15 | 3.252 | 18 | 0.045 |
| 1106 | 35 | 12.013 | 未观测到（删失） | 1.273 |
| 1106 | 60 | 5.275 | 未观测到（删失） | 1.611 |
| 1106 | 100 | 4.681 | 未观测到（删失） | 1.225 |
| 1107 | 15 | 3.417 | 未观测到（删失） | 1.752 |
| 1107 | 35 | 12.629 | 未观测到（删失） | 1.427 |
| 1107 | 60 | 17.776 | 未观测到（删失） | 1.174 |
| 1107 | 100 | 22.092 | 未观测到（删失） | 1.220 |

门前6格保留率中位1.350，门后6格1.222，正式判`no_washout`。仅2/12格在30更新内达到半衰；汇总把其余删失值编码为30，所以“中位半衰期30”不是实际测得的衰减常数。三母本确定性门均过。它否定所测单脉冲尺度下“门前毒作用普遍快速被干净更新冲掉”的解释；不否认长期、多个毒样本交互或其他目标下可能有回落。B2整体margin回落与单条差分D不衰退可以并存，二者不可互换。

**SW：删毒行与换毒行身份。** Qwen SQuAD五seed，W0原样；W1把越带附近选定毒行改为干净；W2保留到达时刻与毒标签、换成其他毒样本。所有W0恢复门通过。

| seed | 选定毒行数 | W0 t90 | W1删除t90 | W2换身份t90 | 机器身份判读 |
| 1003 | 4 | 110 | 131 | 109 | identity_irrelevant |
| 1006 | 4 | 84 | 96 | 83 | identity_irrelevant |
| 1009 | 3 | 91 | 175 | 150 | identity_matters |
| 1004 | 7 | 99 | 140 | 85 | identity_matters |
| 1008 | 3 | 98 | 107 | 95 | unclear |

删除使t90分别延后21/12/84/41/9步，五条均支持所选窗口毒推力有用。换身份在1003/1006几乎不动，在1009显著变晚，在1004变早，在1008未决。因此“任何毒行都完全等价”不成立；加速组的多数规则`identity_irrelevant`不能掩盖1009例外。1009 W2的首次t50=93而后判窗口140–150，反映首达与后续回落/窗口规则不同，不能把它压成单调曲线。

**GATE：先正常预训练是否促进后门。** 先用无重叠干净行达到≥36/60正常映射，再重置Adam/RNG/学习率调度跑同一投毒序列；随机对照G1r将LoRA方向随机化但保持逐模块B范数。

| seed | 母本t_jump | G1预训练后 | G1r等范数随机 | G1提前 |
|---|---:|---:|---:|---:|
| 1105 | 48 | 37 | 48 | 11步 |
| 1106 | 72 | 40 | 74 | 32步 |
| 1107 | 72 | 65 | 71 | 7步 |

冻结判档叫`gate_necessary`：G1 advanced、G1r unchanged。这是正式合同标签；科学解释应限于**正常预训练所得方向/状态对后门形成有促进作用，且不能只由LoRA范数解释**。实验没有反向阻止正常能力又保持其他条件的必要性测试，预训练同时改变多种权重性质，不能严称“正常能力是唯一必要门”。也没有把额外干净预训练成本纳入一种防御/攻击效率优越性比较。

来源：[WO逐格原始数值](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/cross-model-asr/20261016_asr_mechanism_washout_gpu0_r01/results/judge.json)；[WO指标与删失规则](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-washout-2026-10-16/CONTRACT_WO.md)；[SW五seed判读](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/cross-model-asr/20261015_asr_mechanism_swap_gpu0_r01/results/sw_judge.json)；[GATE三seed结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/cross-model-asr/20261015_asr_mechanism_gate_gpu0_r01/results/judge.json)；[GATE动作与限制](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-gate-2026-10-15/CONTRACT_GATE.md)。

### 机制审计11. 激活形成S1/S2/P：表示收敛线索存在，隐藏进度没有跑通

Qwen3-8B SQuAD十seed，层24–35、提示末位置。20–200步60探针margin与原评测差0、adapter哈希一致；但1250步终态因批处理差异有6/10超出0.25logit容差，偏差0.31–0.63。原合同要求这些seed“照算但不判”，不能因为相对误差小而豁免；4个过门seed为1002/1005/1008/1010。

| 量 | 十seed描述值 | 解释与边界 |
|---|---:|---|
| 学到的激活方向跨seed两两余弦中位 | 0.889 | 45个配对不是45个独立seed；4个过门seed约0.91 |
| 随机方向绝对余弦99%分位 | 0.040 | 参考几何，不是独立模型对照 |
| 终态触发减clean差异中新学份额 | 0.938 | 支持训练新增表示，仍为该分解定义下比例 |
| 与基座触发方向余弦 | 0.380 | 近触发对照0.345，未显示强触发特有继承 |
| 带内位置与方向投影相关 | 0.177 | 桥接fails |
| 带内位置与基座几何相关 | 0.086 | 基座几何fails |
| 窗前方向进度领先读数 | 0/10 | P总档mixed；2 not_built、2 with_readout、6 partial |

窗前方向进度中位约29%，共同量进度约一半；在60–80步附近方向余弦才快速提高。这个结果没有给出“ASR前已成熟的隐藏后门方向”。终态跨seed表示相似，与toy的权重方向近正交可同时成立，但两者来自不同任务，不能直接当同一组样本的实现—表示对应证明。

激活形成线原计划中的因果消融/注入（原编号S3）、终态按原批次复核和跨seed迁移，本次材料尚未给出闭环。它与第四部分已完成的LLaVA临界剂量/顺序实验同名“S3”，但属于不同合同，不能相互充当完成证据。S2相关性本身不能推出因果或跨模型通用方向。

来源：[S2原始汇总](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-formation-2026-10-15/results/s2.json)（`family/seeds/gates/gates_passed`）；[P逐seed进度](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-formation-2026-10-15/results/p_progress.json)；[正式边界](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-formation-2026-10-15/RESULTS_S2P.md)。

### 机制审计12. 10-18 粗区域冻结：真能延迟，不能持续阻断

冻结24–35层MLP LoRA，三个toy母本1105/1106/1107；参照R、从分叉冻结M、从第0步冻结D、前12层P和3组随机12层X1–X3，共21训练臂。这里是历史`.grad=None`冻结，被冻参数与Adam状态不变；它与后面“参数增量缩放、Adam继续演化”不是同一个动作。

| 臂 | s1105相对R延迟 | s1106 | s1107 |
|---|---:|---:|---:|
| R起跳绝对步（本批形成规则） | 53 | 92 | 77 |
| M：24–35层，分叉后冻结 | +15 | +40 | +5 |
| D：24–35层，从0冻结 | +5 | +35 | +15 |
| P：0–11层 | 0 | +10 | 0 |
| X1随机 | +5 | −10 | 0 |
| X2随机 | +5 | +10 | 0 |
| X3随机，含目标段6层 | +10 | +35 | 0 |

全部冻结臂最终形成；M/D按“三母本都至少延迟10步”的冻结规则均unclear。5步采样下，M比最强随机在三母本都多5步只是一个采样间隔，不能据此建立位置优势。参照400步攻击124/124、正常50/56/46。冻结臂达到形成就停，没有进入延长段，学习保留判据未触发。报告约202任务GPU分钟。

保留的结论是：**这几种粗区域策略未持续阻断形成**。历史报告中“最强限制无效，因此只对部分行限制也无效”隐含单调性，实际训练非线性且存在优化器/补偿效应；没有直接测过的选择性策略不能由该推理直接否定。

来源：[冻结区域正式报告](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/freeze-region-toy-2026-10-18/REPORT.zh-CN.md)。本批53/92/77与早期首次>3的48/72/72是不同形成判据，不能混作复现误差。

### 机制审计13. 10-19 权重轨迹选区与功能更新提名：代理不能替代真实干预

**参数选择性CPU诊断。** 三seed18份完整adapter，读取约3.145GB，16.82秒CPU，零前向/反向/GPU。36层×7投影=252组，2268行时间点读数、756行均值/rank；真实有效权重采用`2(B1A1−B0A0)`。固定提名要求“母本−B2分化上四分位且B2漂移下半区，至少两seed成立”。

| seed | 252组中同时满足数 | 两轴rank相关 |
|---|---:|---:|
| 1105 | 0 | 0.9556 |
| 1106 | 0 | 0.9811 |
| 1107 | 0 | 0.9814 |

结果0候选，未临时凑top2。35层gate_proj的B2漂移大只是一条几何观察，不能命名正常学习必要区。两轴共享B2、受尺度影响，相关高不能证明任务只能共用参数。

**Functional-update toy已有终局，不能再写未闭环。** 2026-10-27补查服务器得到`summary.complete=true`、`selection_final.run_training=false`、理由`no_verified_selectivity_within_registered_family`。提名24层up与23层down后，在独立校准输入上做实际一步增量撤回：

| 候选 | 1105攻击条件margin降低 | 1106攻击条件margin降低 | 1105正常CE代价 | 1106正常CE代价 | 两seed通过 |
|---|---:|---:|---:|---:|---|
| 24/up_proj | +0.037279 | −0.018630 | +0.002747 | +0.002837 | 否 |
| 23/down_proj | +0.014168 | −0.010910 | −0.000059 | +0.002457 | 否 |
| 两组并集 | +0.024837 | −0.005727 | +0.001333 | +0.000575 | 否 |

这里攻击诊断量是同题触发/无触发的首token `(target−correct)` margin之差，单位logit，不是ASR或百分点；正常是完整答案+EOS平均CE。每个候选1105通过、1106方向反向而未过。已知触发器用于校准，不属于未知触发防御。

因此P、随机X、Curfew+P、Curfew+X各3seed、合计12条1250更新训练**按注册科学止损未运行**，不是12条训练失败，也不是已证所有参数选择性不存在。真实新增GPU成本含验收约0.09874小时；结果表中的R、Curfew、LT20九条来自10-19已完成匹配运行，不能重复计为本批新训练。这一终局只否定所注册候选族的跨两个开发seed一步选择性，说明梯度×真实Adam增量可提名，但提名需实际撤回核验。

来源：[功能更新终局原始汇总](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/cross-model-asr/20261019_functional_update_toy_r01/summary.json)、[选择核验](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/cross-model-asr/20261019_functional_update_toy_r01/selection_final.json)、[自动结果清单](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/cross-model-asr/20261019_functional_update_toy_r01/RESULTS.zh-CN.md)；[权重几何固定提名结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/parameter-selectivity-2026-10-19/results/candidates.json)、[全部252×3排序](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/parameter-selectivity-2026-10-19/results/rankings.csv)、[几何资产与时间点](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/parameter-selectivity-2026-10-19/results/receipt.json)。其中B2是污染前缀后才去毒，不是从头clean。

### 机制审计14. 10-19至10-20直接干预差异：57条完成，没有持续低ASR

19配置×3开发seed1105/1106/1107，共57条；分叉后250步，绝对终点283/307/307。局部层5/17/29的attention或MLP保留0或50%更新，随机为32层attention、22层MLP，另有R、全局90%/50%。动作是AdamW正常step后缩放LoRA A/B的本步参数增量，零档参数不动但Adam继续演化；模块仍参与前向。**没有执行聊天中完整1250步方案**，也未加Curfew。

seed1106在绝对107步真实存在阶段性差异：

| 动作 | 正常正确/60 | trigger命中/60 | 正常答案NLL |
|---|---:|---:|---:|
| R | 44 | 53 | 0.445 |
| MLP17半更新 | 47 | 10 | 0.191 |
| attention29零更新 | 46 | 9 | 0.200 |
| 随机MLP22半更新 | 47 | 11 | 0.334 |

117步R为正常40/攻击55，MLP17为44/18，attention29为45/9，随机22为45/3。但127–137步攻击分别41→57、56→60、35→58/60，随后全部124/124。不能把短暂分离叫“全部学习同比变慢”，也不能因随机同样成功就抹掉差异；它只没有建立位置特异和持续性。

反向线索：MLP29零更新使终点正常正确R的50/57/46变47/43/39，正常NLL0.327→0.360、0.109→0.471、0.585→0.856；两臂攻击均124/124。它提示某些更新机会值得保留，不能推出冻结其补集就能安全。

直接重数CSV：57条终点51条124/124、5条123/124、1条119/124；最低是全局90%步长，不是局部候选。18干预配置无跨seed`usable_difference`。工程核对57条逐题布尔记录与摘要0不一致，是重计数，不是重新推理或独立语义评分。任务墙钟17.1277h，其中训练6.3297h、评估8.2779h。

来源：[57条终点原始表](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/intervention-response-audit-2026-10-20/ENDPOINTS.csv)；[逐题重数与范围审计](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/intervention-response-audit-2026-10-20/REPORT.zh-CN.md)；[实际执行合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/intervention-response-audit-2026-10-20/EXECUTED_CONTRACT.json)。

### 机制审计15. A/B确认与投影拆分：新轨迹弱确认，K29只成探索候选

A：新轨迹2101/2102，各5臂R、MLP17半更新、attention29零更新及各自全局步长范数参照，共10任务。B：已见1105/1106，各22臂，拆MLP17/29的gate/up/down及组合，attention29的q/k/v/o及组合，共44任务。54正式任务完成；分叉后250更新，主窗+20…+120，ASR降低=`R−干预`，正常变化=`干预−R`，均按曲线积分平均，不是最好点。

| A候选 | 2101平均ASR降低 | 2102平均ASR降低 | 2101平均正常变化 | 2102平均正常变化 |
|---|---:|---:|---:|---:|
| MLP17半更新 | +0.79pp | −1.38pp | +2.17pp | +5.33pp |
| attention29零更新 | +1.96pp | +0.50pp | +1.00pp | +6.96pp |

旧1106的强短暂抑制没有稳定重现。B里K29-only为1105/1106分别ASR降低6.92/14.25pp、正常增加2.67/4.92pp，成为探索提名；组合效应并非简单相加，子集参数规模也不同。54臂V60终点均60/60，因此没有防御完成。任务墙钟10.2845h。

下表保留B全部22臂主窗读数，避免只呈现最好的K-only；R行只是每seed基准，不另算新的独立发现：

| B臂 | 1105 ASR降低pp | 1106 ASR降低pp | 1105正常变化pp | 1106正常变化pp |
| R | +0.00 | +0.00 | +0.00 | +0.00 |
| M17_half_gate | +3.33 | +6.62 | -5.62 | +4.08 |
| M17_half_up | +0.46 | +11.54 | +0.33 | -1.29 |
| M17_half_down | +0.33 | -3.29 | -1.21 | -4.96 |
| M17_half_gate_up | +2.25 | +10.46 | -1.79 | -1.29 |
| M17_half_gate_down | +1.88 | +3.33 | -2.83 | -3.04 |
| M17_half_up_down | +0.54 | -3.04 | -3.71 | -0.62 |
| M17_half_gate_up_down | -0.33 | +20.17 | -4.79 | +0.83 |
| M29_zero_gate | +0.04 | +4.21 | -12.50 | +3.88 |
| M29_zero_up | +6.00 | +16.92 | -1.33 | -3.17 |
| M29_zero_down | +1.33 | +21.75 | -4.21 | +5.12 |
| M29_zero_gate_up | -1.21 | +8.04 | +0.25 | +6.38 |
| M29_zero_gate_down | +5.04 | -4.29 | +2.83 | +5.96 |
| M29_zero_up_down | -3.04 | +11.08 | -0.88 | -1.04 |
| M29_zero_gate_up_down | +1.42 | +0.71 | -1.21 | -2.75 |
| A29_zero_q | -0.79 | +12.46 | -3.50 | -2.54 |
| A29_zero_k | +6.92 | +14.25 | +2.67 | +4.92 |
| A29_zero_v | +1.25 | +2.54 | -1.96 | +0.62 |
| A29_zero_o | +4.38 | +14.08 | -3.29 | +1.33 |
| A29_zero_q_k | +0.25 | +6.21 | -4.75 | -2.04 |
| A29_zero_v_o | +2.54 | +16.21 | -10.42 | +4.83 |
| A29_zero_q_k_v_o | -1.92 | +19.46 | +3.29 | +2.38 |

来源：[A/B所有过程效应](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/intervention-confirm-decompose-2026-10-20/results-r02/SUMMARY.json)（`experiments.*.seeds.*.*.comparison.mean_auc_effects`）；[216行终点评测](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/intervention-confirm-decompose-2026-10-20/results-r02/ENDPOINTS.csv)；[A/B合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/intervention-confirm-decompose-2026-10-20/CONTRACT.zh-CN.md)。216行=54臂×4题集，不是216次训练。首次验收曾因优化器状态别名失败，修复后才正式运行，不计科学阴性。

### 机制审计16. C/D确认K29并拆到头：H1的强效应只在一个开发轨迹

C：2101/2102各测K29、固定随机K30、整体减速，新增6任务、复用2R。D：1105/1106，各8个KV头（每头128输出通道）、整K同动作锚定、2随机128通道组，新增22任务、复用2R。28新GPU任务，4条复用R；层29和H1均为代码零起始索引。

**动作不可混同。** C是LoRA参数增量限制；D是前向功能锚定：选中输出通道使用分叉时LoRA贡献，抑制后续新增贡献，实时A/B与Adam仍演化。D不是冻结优化器、删除头或清零激活，因此额外跑`K_full_anchor`父组。K_norm耦合同头通道，128通道组不是128个独立开关。

| C确认 | ASR主窗降低 | 正常主窗变化 | 终点正常/60（R） | 终点攻击/60 |
|---|---:|---:|---|---:|
| K29,2101 | 6.58pp | −6.46pp | 42（47） | 60 |
| K29,2102 | 5.25pp | +0.38pp | 47（46） | 60 |

2101/2102未用于K-only提名，但之前参加完整attention测试，所以是有限确认，不是全新材料。

D全部头和随机组的主窗读数如下（正的ASR降低表示更安全，正的正常变化表示正确率更高）：

| D臂 | 1105 ASR降低pp | 1106 ASR降低pp | 1105正常变化pp | 1106正常变化pp |
| R | +0.00 | +0.00 | +0.00 | +0.00 |
| K_full_anchor | +2.88 | +5.75 | -5.96 | +4.42 |
| H0 | +2.08 | +10.92 | -2.58 | -3.29 |
| H1 | +0.33 | +23.50 | +4.88 | +2.21 |
| H2 | -2.33 | +11.46 | -9.12 | +6.29 |
| H3 | -1.92 | +7.42 | -4.33 | -0.38 |
| H4 | -0.62 | -2.42 | -0.08 | -2.21 |
| H5 | +1.04 | +4.17 | -2.58 | -1.79 |
| H6 | +3.50 | +1.92 | -6.17 | -0.54 |
| H7 | -1.04 | +22.29 | +4.29 | -1.83 |
| X_channels0 | -1.08 | -8.33 | -4.25 | -3.62 |
| X_channels1 | +5.21 | +8.67 | -5.00 | -3.21 |

H1因1106攻击降低23.50pp而受到关注，1105只有0.33pp；正常主窗分别+2.21/+4.88pp。H1终点正常1105为55/60（R50），1106为50/60（R57），因此终点并非普遍学习改善。**所有C/D V60终点60/60，D的P64/H64攻击也都64/64。** 两随机组只提供有限参考，同通道数不等于同功能干预强度。

目前支持的是训练轨迹依赖、阶段依赖的响应差异，不是H1为稳定后门部位、形成必经位置或防御方案。H1是在8头扫描后挑出的，独立确认、单通道/单权重拆分、Curfew组合均未由此批完成。后续不应围绕一个后验挑出的头无限细分来代替可迁移选区规则。

C/D任务墙钟5.3929h，约3.799h评估、1.520h训练；前三批新任务合计57+54+28=139，任务墙钟32.8051h，不等于纯GPU核忙碌时间、双卡实际历时或全项目总费用。K批次前两次模型缓存加载失败，未进入正式训练，不能计科学阴性。

来源：[C/D全部过程效应](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/k29-confirm-heads-2026-10-20/results-r03/SUMMARY.json)；[128行终点评测](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/k29-confirm-heads-2026-10-20/results-r03/ENDPOINTS.csv)；[C/D动作定义](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/k29-confirm-heads-2026-10-20/CONTRACT.zh-CN.md)；[三批成本与当前定位](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/defense-recap-2026-10-21/RECAP.zh-CN.md)。128行=含复用R的32臂×4题集。

### 机制审计17. 机制线目前可保留的结论、撤回的解释与缺口

| 解释或路线 | 当前证据状态 | 下一步真正缺的东西 |
|---|---|---|
| 归因占比是提前内部信号 | IM2已解释为比值饱和；撤回 | 读数需独立于输出及归一化，先做代数检查 |
| 后门在平台上无新毒输入也自行形成 | 三toy B2在100步内阻断；该版本不支持 | 更广任务/剂量的因果范围，不能外推所有状态 |
| 累计毒行或门后累计量是统一时钟 | 多次留出预测失败，有极端反例 | 固定新数据上的可检验预测，而非继续拟合同批种子 |
| 早期毒效应被固定时间常数冲刷 | WO单脉冲结果反对普遍快速冲刷 | 多毒交互、长期轨迹、不同目标/协议需分别检验 |
| 正常机器×毒偏置相乘 | 格子两可测母本反对；撤回 | 正常预训练促进并不等价于严格必要门 |
| 后门终态方向是形成必经方向 | 训练中持续投影只延迟2–6步；否定该必要性 | 形成期路径可替代性、动态/多方向策略的独立实验 |
| 所有ASR跃升都是连续量突变 | Q1/toy否定；不同平台和seed有异质性 | 同读数、同网格、同时间窗的比较 |
| 平台退出可预测起跳 | 8条新母本MAE劣于两简单对照；refuted | 不把少量典型曲线当统一形态 |
| 共享平移+输入偏移 | 多族旧轨迹拟合解释份额高 | 仍是读数结构；其因果来源与预测用途未闭环 |
| 终态激活方向跨seed收敛 | S2有强相关线索，6/10终态工程门需复核 | 按原批次复核终态门，并完成本线因果消融/注入及迁移；第四部分同名S3不覆盖此缺口 |
| 局部更新存在攻击—学习差异 | 57/54/28批确有暂时差异及正常代价 | 稳定选区、独立轨迹、同动作随机与同学习水平比较 |
| K29/H1为可用防御位置 | 未建立，所有已测终点高ASR | 避免后验追逐头号；若组合Curfew须独立公平检验 |
| 功能更新提名已完成核验 | 两单组及并集在1106反向；终局科学止损 | 12条后续训练未运行；不能伪装为完整防御阴性 |

总结中的“没有支持”只针对实际测试的合同、读数和范围。它不等于方向永久不可能；同样，局部阳性也不等于普遍机制或可部署防御。研究推进到这里，最可靠的积累是完整的轨迹、定义、反例与恢复/干预工具，以及哪些简单故事经不起对照。


<a id="part-4"></a>

## 四、真实留存曲线与后期顺序、剂量实验

### 真实任务的形成、下降与反弹：mid16 补测改变了什么

这条轨迹值得单列，因为它纠正了仅看终点的判断，而且保留了目前最清楚的一条“已形成但最后不命中”的真实任务证据。对象是 **Qwen3-8B LoRA / SQuAD / seed1006 / cf→violin**；完整训练1250更新、20,000行，仅16条指定毒行，最后一条位于第116步。其余原毒行使用无触发、正常答案版本训练。

原实验只知道终点0/124。随后补回中期评测：V60固定60题，触发、无触发、近似触发三条件，103个采样点，共18,540条逐题输出。历史保真审计发现新旧1250步loss、梯度范数、LR及20,000行消费日志一致；终点权重文件哈希一致。本次再次运行已有只读审计，重计关键曲线与判定。**这是补观测与输出复算，不是新增独立seed。**

| step | 已消费毒行 | trigger目标/60 | clean目标/60 | near目标/60 | trigger正确/60 | clean正确/60 |
|---:|---:|---:|---:|---:|---:|---:|
| 80 | 10 | 3 | 0 | 0 | 53 | 56 |
| 85 | 11 | 22 | 1 | 7 | 35 | 54 |
| 90 | 11 | 9 | 0 | 2 | 48 | 55 |
| 95 | 12 | 12 | 0 | 2 | 45 | 55 |
| 100 | 13 | 39 | 0 | 7 | 18 | 56 |
| 105 | 14 | 59 | 0 | 30 | 1 | 55 |
| 110 | 15 | 60 | 0 | 46 | 0 | 55 |
| 115 | 15 | 60 | 0 | 59 | 0 | 54 |
| 120 | 16 | 60 | 0 | 60 | 0 | 54 |
| 125 | 16 | 60 | 0 | 48 | 0 | 54 |
| 135 | 16 | 15 | 0 | 1 | 37 | 55 |
| 150 | 16 | 1 | 0 | 0 | 48 | 55 |
| 170 | 16 | 1 | 0 | 0 | 37 | 55 |
| 190 | 16 | 17 | 0 | 1 | 33 | 55 |
| 200 | 16 | 28 | 0 | 2 | 20 | 56 |
| 210 | 16 | 21 | 0 | 1 | 25 | 56 |
| 230 | 16 | 1 | 0 | 0 | 50 | 56 |
| 260 | 16 | 2 | 0 | 1 | 53 | 55 |
| 310 | 16 | 1 | 0 | 0 | 52 | 56 |
| 320 | 16 | 0 | 0 | 0 | 53 | 56 |
| 1240 | 16 | 0 | 0 | 0 | 55 | 56 |

来源：[曲线原始记录](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/mid16-retention-curve-2026-10-17/results/run/curve.jsonl)、[逐题重计结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/mid16-retention-curve-2026-10-17/results/audit/RECOUNT.json)、[保真与解释审计](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/mid16-retention-curve-2026-10-17/RESULTS_AUDIT.zh-CN.md)。

冻结形成门槛为相邻两采样点trigger≥12/60，且trigger−clean≥9/60。因此首次确认形成是100步（95与100连续达标）；85步已出现强目标行为，但90步回落，不满足连续条件。满命中为110、115、120、125四点，**105不是60/60**。登记的gone=230仅表示连续低命中≤2/60，**不是230步起精确为零**；320步起后续全部已采样点才严格归零，1250终点另确认为0/124。

这条曲线支持的解释是：强条件化目标行为曾出现，在没有新增毒行的后续训练中下降并反弹；150→200的新增27个目标命中可以按题目身份直接核对。它不支持单调洗掉、一维半衰期、持续供毒必要性、内部记忆擦除或“所有同类零终点都曾形成”。中期只有已见V60，H64只测终点；near触发在120步也为60/60，因此它不是只响应精确cf的窄关联。

另一个容易漏掉的事实是：**低ASR不等于正常功能恢复。** 170步攻击仅1/60，但trigger正确仅37/60，19题首词输出cf；clean正确55/60。终点V60/H64合计：trigger正确109/124、clean正确114/124、near正确111/124，三条件目标命中均0/124，oldQA为72/76。

终点V60的目标首token−正确首token margin分别为trigger −6.8021、clean −15.3401，条件差8.5380；这说明相对输出竞争仍不同，不能改写成“目标绝对logit升高8.5”或“已定位可恢复的内部后门”。真实消退原因仍缺同一完整状态下的续供/屏蔽/正常替换分叉；已有A1的前缀、样本身份与未来日程不同，不能充当该反事实。

本次历史运行墙钟6498.38秒，训练3528.12秒，中期评测2868.33秒，占44.14%。这些是单次任务计时，不是Curfew/LT效率比较。原中期没有完整模型/优化器/RNG快照保存凭证，未来若做因果分叉，重建前缀成本必须计入。

### 10 月下旬的顺序线：从整体顺序差异到逐行位置干预

本节补入当前主工作树中原先没有完整归档的一批服务器结果。**所有数字来自已存输出，本次仅用原冻结读数函数做CPU复算。** 连续起飞读数为固定core探针目标margin的中位数持续过零时刻 `t_M0s`，在相邻checkpoint间线性插值，并要求后续40步窗口内已测点维持过零；不是实际存在“第328.0695步”的模型。并列检查LP的argmax与原生生成ASR的一致率。

这些实验的主要对象为LLaVA原配方、固定train/init1001、1%投毒。下文PL、CL、NF中“搬10行”保留毒行数量，改变其消费位置并相应挪动其他行；这是**完整训练序列干预**，不是孤立地改变一个纯数学距离变量。不同学习率日程的相对位移必须分别相对自己的参照。

#### SF：毒行位置与身份构成的“骨架”，和干净行顺序的“填充”都重要

SF把一个顺序的毒行位置及每个位置的毒行身份保留，将剩余位置按另一顺序的干净子序列填充。每臂保持同一20,000行多重集、同一毒集，实测Trainer消费顺序另有结构核验。三个父顺序持续过零参照为：1001=355.1852、1007=464.4828、1009=328.0695。

| 重组臂（S=毒骨架，F=干净填充） | t_M0s | 持续ASR过半step | 相对两父顺序的骨架份额 |
|---|---:|---:|---:|
| P1：S1001/F1007 | 368.5385 | 370 | 0.8778 |
| P1：S1007/F1001 | 402.8729 | 410 | 0.4363 |
| P2：S1009/F1007 | 356.7717 | 360 | 0.7896 |
| P2：S1007/F1009 | 409.6832 | 410 | 0.5983 |
| S1007/F2001（新随机填充） | 405.4740 | 410 | 无F2001父轨迹，不计算 |
| S1009/F2001（新随机填充） | 306.3043 | 310 | 无F2001父轨迹，不计算 |

份额定义为 `(重组时刻−填充父时刻)/(骨架父时刻−填充父时刻)`；它只是两父参照下的线性归一化描述，不是严格方差解释率或毒梯度因果占比。P2两方向平均0.6939，P1为0.6571；方向间不对称明显，不能宣称“所有顺序差异六七成都由毒行决定”。固定骨架配新干净填充仍可明显移动时刻，说明干净序列不能被当作无关背景。

本次复算六臂LP/生成一致率0.9945–0.9973；读到的600步ASR仍为96.5%–99.5%，重组改变了形成过程，没有产生防御。来源：[SF冻结实现与读数](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/cl_code/sf.py)、[父参照](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/pl_code/refs_tm0.json)、[本次逐臂复算](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/recomputed_recent_readouts.json)。

#### B3/B3D/J09：优化器状态有作用，但不是唯一时钟

从同一250步完整状态分叉，B3-ID原样恢复，B3-CLR清零Adam的一阶矩、二阶矩及其内部step计数；模型参数、学习率调度器、Trainer全局步和RNG状态保持。保真闸门核到448个参数张量完全一致、12个比较loss最大差0，并检查896个动量张量清零。B3D-KICK则对参数施加登记的随机微扰，其余优化器状态不变。

| 臂 | 动作 | t_M0s | 相对B3-ID位移 | 持续ASR过半 |
|---|---|---:|---:|---:|
| B3-ID | 同状态原样恢复 | 328.0695 | 0 | 330 |
| B3-CLR | 清零一阶、二阶矩并重置Adam step | 346.4516 | +18.3821 | 350 |
| B3D-KICK | 固定随机参数微扰 | 328.0460 | −0.0235 | 330 |
| J09-I1007-O1009 | 同顺序1009，初始化/dropout改1007 | 326.2500 | −1.8195 | 330 |
| J09-I1008-O1009 | 同顺序1009，初始化/dropout改1008 | 332.0661 | +3.9966 | 340 |

KICK登记最大单参数位移约2.9295e−4，实现与设计偏差约7.70e−10；但这不等于与清零Adam的功能作用严格匹配。B3-CLR仅说明已测状态中这一组合重置影响后续形成时刻；18.38步不能单独归给一阶矩、二阶矩或Adam偏差校正的内部时钟。按原`b3.verdict`，该位移介于MOVED=10与SUPPORT=54之间，正式档为“不可判：冲击或部分记忆”，没有达到“支持优化器记忆”的登记门槛。J09三初始化连续时刻范围5.8161步，属于这一顺序/日程的有限抖动尺度，不能自动移植为所有日程的误差带。各已读600步ASR为98%–99%，仍是延迟与稳定性实验，不是终点安全实验。

另有只清一阶矩的B3D-M0设计/状态文件；本次未取得与其他臂同样完整的行为和LP读数，故不为它填入时刻或科学结论。来源：[B3实际重置实现](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/b3_code/b3.py)、[B3保真与动作](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/b3/gate.json)、[B3D动作核验](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/b3d/gate.json)、[J09核验](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/j09/gate.json)、[逐臂复算](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/recomputed_recent_readouts.json)。

#### PL：把10条毒行后移，近窗位移较大，但合同主档不可判

余弦日程参照为顺序1009的328.0695。E/M/L分别是早/中/晚窗；30表示各选定毒行后移30个optimizer step；end表示推到训练尾部。随机R30a/b为同规模随机选择。

| 臂 | t_M0s | 相对参照位移 | 原QUIET≤5 / MOVED≥10规则 |
|---|---:|---:|---|
| R30a | 332.8750 | +4.8055 | quiet |
| R30b | 328.7320 | +0.6625 | quiet |
| E30 | 334.0323 | +5.9628 | 带间 |
| M30 | 337.0030 | +8.9335 | 带间 |
| L30 | 351.4340 | +23.3645 | moved later |
| Eend | 373.5374 | +45.4679 | 描述性移除对照 |
| Lend | 377.4257 | +49.3562 | 描述性移除对照 |

晚窗后移效果比早窗大17.4017步，L30/30=0.7788是本设置的一点响应比。但E30没有落进≤5的安静带，所以按冻结代码主档为**不可判**，不能事后调宽到6步使其通过。Eend与Lend仅差3.8883步，在该规则下不支持“推到末尾的具体10条位置显著不同”。所有已读600步ASR≥97.5%；没有证据显示终态防御。

来源：[PL窗口、动作和判定代码](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/pl_code/pl.py)、[保真闸门](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/pl/gate.json)、[本次复算](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/recomputed_recent_readouts.json)。

#### CL/NF：平学习率并没有消除背景波动；远场未检出超出粗波动的效应

CL将日程改为constant_with_warmup，窗口相对其自己实测t*=323.0769重新构造。CL-ID、R30a、L30的t_M0s分别为323.0769、328.5301、335.3402；位移为0、+5.4532、+12.2633。随机臂落在5–10带间，所以CL主档仍为**不可判**，不能写“已经排除到达时学习率解释”。CL预测落盘曾写入被捕获的调用异常；后来重算的数值只能作为事后对账，不能追认其为完整的臂前冻结预测。

NF随后明确补测三个随机臂，并复用已看过的CL-R30a，先形成n=4的随机搬动分布再测早窗远场臂：

| 臂 | 相对CL-ID的持续过零位移 |
|---|---:|
| CL-R30a（复用） | +5.4532 |
| NF-R30b | −15.2338 |
| NF-R30c | −0.9536 |
| NF-R30d | +11.2530 |
| NF-E30（早窗，离参照起飞约200–271步） | +3.4748 |

随机四点均值**+0.1297**，样本标准差**11.3912**，冻结门槛 `max(mean+2sd,5)=22.9121`。这否定了原先“随机波动sd≤4”的预期，也表明借来的±5/±10带不足以代表这个设置。NF-E30的+3.4748位于该粗波动带内，冻结程序标作 `far field inert (finite support)`。

**本文将科学结论收窄为：这一次远场搬动未显示超出所测随机搬动波动的效果。** n=4、单次远场、单初始化、单剂量既不能证明效应为零，也不能证明核的支集严格有限、只剩50步有效或β1是机制。`mean+2sd`是登记的操作规则，不是经过校准的95%统计置信界；near与far两个条件也不足以识别核形状。NF-R30c的首次过零308.6517、持续过零322.1233不同，继续说明不能只取第一次越线。

原CL/NF合同中对某些标度律“理论已被否证”的说法，还需核对对方定理的模型、优化器与数据假设；这里的Adam/LoRA实测不能越过适用条件宣称普遍理论反例。来源：[CL合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/cl_code/CONTRACT_CL.md)、[CL原预测异常](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/cl/gate.json)、[NF合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/nf_code/CONTRACT_NF.md)、[NF臂前随机汇总](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/nf/gate.json)、[本次复算](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/recomputed_recent_readouts.json)。

### 临界剂量与验收：S3 到固定八臂 DW8

#### S3：同剂量、同毒集，某些已选顺序可以翻转预算内是否形成

S3保持20,000样本、1250步、同剂量毒集，分离order与train/init seed；固定形成阈值terminal ASR≥50%。10条新臂加2条已有基准构成12格，不能称12次新训练。下表终态分母均为200；I1001/O1001两格为历史复用，并单独标出。

| 剂量 | order | I1001 ASR | I1007 ASR | I1008 ASR | 新臂clean（I1007/I1008；另列时含I1001） |
|---|---:|---:|---:|---:|---|
| 0.40%（80毒行） | 1001 | 21.5%（复用） | 36.5% | 6.0% | 65.5% / 65.0% |
| 0.40% | 1009 | 0.5% | 0.5% | 0.0% | 63.5% / 64.0% / 64.5% |
| 0.42%（84毒行） | 1001 | 96.0%（复用） | 98.0% | 93.5% | 64.5% / 64.5% |
| 0.42% | 1007 | 0.0% | 0.0% | 0.5% | 62.5% / 62.5% / 62.0% |

10条新臂的625步中期ASR均为0。0.42%在已选order1001与1007之间、三个init上稳定出现高低分叉；但0.40%下两组均低于阈值，原被期待的“快顺序”1009没有救活低剂量。**高剂量的顺序差异成立，低剂量救活预测未获支持。** clean也有约数个百分点差异，不宜仅用ASR单轴评价顺序改造。

它给出了反对“只由剂量决定终态”的条件性证据；order1001/1007/1009是历史探索后挑选的顺序，不能把三次init重复当随机顺序总体形成率，也不能认为order1009在1%起飞较快就必然在0.40%更容易形成。

来源：[S3实际臂定义与复用值](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/s3_code/s3def.py)、[构造/执行闸门](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/s3)、[新臂逐题行为数据](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/behavioral)、[两条历史基准行为数据](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/runs/behavioral)。

#### DW8：两个新顺序 × 四个冻结剂量，8/8完成，未出现二值形成分叉

这一批是为了避免继续只看已选极端顺序。固定train seed1001（同时控制初始化与dropout）、poison seed31001，order seed为20001/20002；剂量0.40/0.42/0.45/0.50%，训练20,000行、1 epoch、1250更新，LoRA r16/α32/dropout0.05，学习率1e−4/余弦。每臂只在终态进行一次原200P贪心评测；不能据此恢复中期曲线。

本次直接取回8份完成receipt与逐题输出，重新核对每臂200个唯一probe ID，重计ASR/clean均与receipt一致：

| 剂量 | order20001 目标/200（ASR） | order20002 目标/200（ASR） | clean正确/200（20001 / 20002） | ASR绝对差 | ≥50%形成标签 |
|---|---:|---:|---:|---:|---|
| 0.40% | 0（0%） | 0（0%） | 130 / 126 | 0pp | 两者否 |
| 0.42% | 1（0.5%） | 1（0.5%） | 130 / 127 | 0pp | 两者否 |
| 0.45% | 2（1.0%） | 76（38.0%） | 130 / 127 | 37pp | 两者否 |
| 0.50% | 147（73.5%） | 167（83.5%） | 128 / 128 | 10pp | 两者是 |

**结果同时包含有价值的差异与未命中的预期：** 0.45%两个顺序差37pp，说明连续结果有明显顺序依赖；但四个预设网格点都没有“一高一低跨50%”的分叉。因此不能把S3的0.42%分叉推广到所有新顺序，也不能说“顺序效应没有了”。两个order无法估计总体形成概率，四个点无法推断连续剂量区间宽度；历史极端顺序不能混入这8臂分母。

每臂实际占用约74.89–75.24分钟，合计**10.0037 H20任务卡时**；10月25日历史巡检记录整体墙钟5小时16分23秒。前者是两卡任务占用相加，不是墙钟。冻结commit为 `aaed2465b8625570b5adaa807db67364fe52568a`。本轮没有按结果换seed、增加剂量或提前终止。

来源：[冻结合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/dose-width-2026-10-25/CONTRACT.zh-CN.md)、[完成巡检](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/dose-width-2026-10-25/monitoring/20261025T145402Z.md)、[本次逐题重计](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/recomputed_dw8.json)、[服务器原始receipt](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/dose-width-20261025/runs/receipts)、[逐题输出](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/dose-width-20261025/runs/behavioral)。

#### 从这些实验到“验收程序”还差什么

目前更准确的研究对象是：**在明确的模型、预算、毒集和顺序分布下，训练结果对顺序是否稳健，以及某种验收规则会不会误接收高风险模型或误拒正常模型。** S3和DW8提供的是部分条件下的证伪与有限网格事实；它们还没有交付一个校准后的验收概率。

10月23–25日讨论中的A1（固定0.42%、同poison集与init，新增6条order加已有2条）、A2干净孪生、A3预算扫描、B2守恒读数属于另一个问题清单。DW8的4剂量×2新order不能替代A1，M2的一条clean twin也不能自动替代A2所需的完整误拒侧评估。本文没有取得这些后续合同对应的完整结果链，因此把它们保留为待对账/待完成事项，而不宣布其已完成或仍在运行。



<a id="part-5"></a>

## 五、最新M1–M3：定位、计时与逐行账本

本章根据2026-10-27从服务器取回的真实JSON及冻结合同撰写，不使用旧工作树“没有该线结果”的印象。M1/M2/M3结果都已落盘；本次本地重新计算了M2若干逐题均值，以及M3从`segments.json`导出的相关、比例和去趋势相关，和已保存汇总一致。没有重跑模型训练或前向。

### 1. 必须先分清的读数

- LP的 **M** 是目标首子词logit减去其余所有词logit的最大值；不是T3的“目标减正确答案”。M>0表示目标首子词赢得argmax。
- **D=mean_i[M(x_i^trig)−M(x_i^clean)]** 是同题触发/干净图像的目标对最大竞争词margin差，对指定探针集合取均值（主判EVAL100，选择SELECT100）。D_full是全部LoRA开启、D_none是全部LoRA关闭的效果，单位是logit，不是ASR；共用实现[`margins/readD/param_grad`](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/m_code/m_common/mlib.py:115)已核对。
- **F(S)=(D_S−D_none)/(D_full−D_none)**：只保留模块集合S时，恢复了多少触发效应；**E(S)=(D_¬S−D_none)/(D_full−D_none)**：去掉S后还剩多少。它们不是概率，不必落在0–1，也不必相加为1。
- M1实际交出的是“充分型”32模块集合S*，所以M2的 **Y(t)=D_S*(t)**，**Z(t)=D_¬S*(t)**；不能把Y与Z当成天然可加的两份贡献。
- M2的计时τ采用向上穿过固定效应水平、之后40步所有测点保持的规则；其与早期首次t50或W50不同。

该批使用同初始化1001的LLaVA三条顺序1009/1007/1008，以及一条将1009中200个poison位置替换成同源clean行的孪生。P-core200预先分SELECT100/EVAL100，选择集合只用SELECT，EVAL判档。它仍是固定模型、固定初始化、少量顺序的机制研究。

### 2. M1确实得到可复核的终态模块定位，但不等于训练防御已成功

M1在O1009的step1250寻找能同时“保留恢复≥80%、删除剩余≤20%”的小集合，并要求相对同大小干净任务基线的F−E至少多0.20。实现闸门通过：250/330步与LP逐题差0；LoRA全关与基座差0；batch8与batch1最大差0.015625，小于0.05门槛。终态EVAL D_full=17.215996、D_none=0.034859，差17.181137 logit，超过5的有效性门槛。

| 集合/基线 | 模块数 | SELECT F/E | EVAL F | EVAL E | 解释 |
|---|---:|---|---:|---:|---|
| 主集合C（r1前32） | 32 | 1.058762 / 0.024571 | **1.081367** | **0.023152** | 正式“定位”，充分型 |
| 同大小B_task | 32 | 不参与该集合选择 | 0.052575 | 0.339011 | 主集合F−E领先1.344651 |
| 同大小B_norm | 32 | 描述 | 0.040019 | 0.222689 | 范数大并不能恢复同样效果 |
| 必要型候选C_E：L24.o、L24.v | 2 | 0.039089 / 0.169555 | 0.035828 | 0.170147 | 删这两个作用大，只留它们作用小 |
| 20个随机32模块集 | 32 | 描述 | 中位0.003747；范围0.001294–0.068225 | 未作主比较 | 随机集不用于正式p值 |

主集合是224个LoRA模块中的32个，分布在17层；MLP占50%，第16层及以后占81.25%。因此原“多数是MLP”预测并未成立（恰好一半），而较深层占多数成立。F>1说明关掉其他模块后触发效应反而略强，不能表述为“ASR恢复108%”。

L24的作用尤其集中：在S*内单独去除L24.o/v/k时，F分别减少0.950231/0.412754/0.824356；这些是条件消融、不能相加成份额。删除L24整层的7个LoRA模块后E=0.081496，即归一化触发效应剩约8.15%；删除大部分其他整层后E接近1。这个事实定位了**已形成行为的关键承载位置**，尚不保证阻止这些模块更新会安全地阻止形成，更未测其全任务能力代价。

#### 起飞附近与终态：核心有重合，不能把Jaccard小误读成完全换电路

step330按同样规则作描述得到C={L24.o_proj,L24.v_proj,L24.k_proj,L24.q_proj}，EVAL F=0.968385、E=0.126040，干净任务基线F=0.000653、E=0.990658。step330与终态Jaccard=0.125，低于预期≥0.5；但**四个早期模块全部包含在终态32模块内，4/32恰为0.125**。这个低Jaccard主要反映机械选集大小4→32，不能写成“早晚核心毫无重合”。阶梯选择也不保证数学上全局最小，不能据此声称恰有32个不可约后门模块。

#### 残差流描述：早层触发位置与后层答案位置的作用交接

本轮根据JPEG实际像素差异使用9个触发区域token，区别于早期IP的4个位置。noising中，层0/8/24替换触发9位置分别消去约100.06%/100.00%/96.82%的D；到层28触发9位置仅消去2.76%，最后token却消去90.74%。denoising对应层24触发位置恢复98.92%、层28最后token恢复89.36%。层32最后token交换为1是靠近输出的平凡边界，不应作为新机制贡献。

这些是**同一终态模型的trig↔clean状态替换**，与IP“已跳模型↔未跳模型、同一trigger输入”不是同一因果问题。M1有强效应不推翻IP的小转移结果：前者问触发信息经过哪里，后者问不同训练状态之间的行为差异能否被那几个局部状态搬走。

干净孪生终态D_full−D_none=0.009818，远小于毒模型17.181137；孪生归一化F=0.75369/E=1.10145的分母极小，不应按大份额解读。

来源：[M1合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/m_code/m1/CONTRACT_M1.md)、[M1完整结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m1/results_m1.json)、[M1原始分项](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m1/raw.json)。

### 3. M2：集合在起飞时已承载效应，但固定D阈值不能给全部顺序准确计时

三条poison轨迹各49点、孪生36点。参考持续margin过零时刻为1009=328.07、1007=464.5、1008=401.5。以1009起飞时D_full的水平为共用阈值，得到：

| 顺序 | 参考t_M0s | D_full过阈τ | 误差 | 容差 | 结果含义 |
|---|---:|---:|---:|---:|---|
| 1009 | 328.07 | 328.0695 | 约0 | 样本内自检 | 参考水平来自自己，不算独立预测 |
| 1007 | 464.5 | 458.2502 | −6.2498 | 2.9698 | 绝对误差超过2×容差，触发“D不计时” |
| 1008 | 401.5 | 402.2609 | +0.7609 | 2.6998 | 在容差内 |

插值误差项δ_int=2，未放宽门槛。**M2-0正式判“D不计时”**：不是D完全无关，而是“各顺序到同一D就起飞”的固定阈值假说没有满足冻结规则。

M2-1的Y相对full同步误差为1007−3.6835、1008−2.0661步，容差3.0666/2.7545；Y判“不可判”，不能取接近的点估计就写同步。Z在两条其他顺序均未达到1009参考水平，τ=∞，判不同步。4个随机集全部P2信噪比失败，所以区分力无法评估。正式组合读法是“照实报”，**不能写顺序效应已经定位到S***。

M2-2则为明确的有限阳性：1009起飞时，终态集合S*的归一化承载比s=**1.417581，[1.348371,1.486975]**，正式“起飞时已承载”；1007/1008在各自起飞时s=1.462861/1.529013（描述）。超过1仍是非加性消融结果，表示只保留S*时读到更大D，不是“141.8%的概率”。

从逐题读数重算可看到1009上D_full/Y/Z：step250为0.25582/0.21154/0.10492；step320为4.39480/6.02693/0.30934；step330为6.23566/8.87010/0.39879；step340为8.46488/12.03060/0.47195；终态为17.21600/18.61397/0.43264。Z有峰后回落，但峰值在1009/1007/1008分别为500/800/700步，不能按先验讲成都在起飞附近。超过孪生全网格最大值并保持40步的t_on_full与t_on_Y在1009均为80；这是使用完整孪生曲线阈值的回顾性弱零分布描述，不是可部署提前248步报警器。

来源：[M2合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/m_code/m2/CONTRACT_M2.md)、[M2判档](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m2/results_m2.json)、[1009逐题轨迹](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m2/traj_1009.json)、[孪生构造闸门](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m2/gm2_t.json)。

### 4. M3：逐行一阶账本校准成功，因果解释和PL预测仍有边界

M3从同初始化、同顺序严格重放到430步，保留Adam/调度器/RNG，按10步一段对触发效应做一阶路径积分；将每行梯度、裁剪、学习率、动量尾巴、Adam预条件和效应梯度场结合。第11–430步共6720行；第1–10步不逐行记账，动量带入项单列。重放250/330/430步adapter差0；旧loss窗口均值误差最大4.5e−5<2e−4；step35 AMP跳过1次，所以430训练步对应Adam429步，不是恢复错位。微批输入token核对n_bad=0，42段恒等式全通过。

#### 主判只回答“账本是否校准”

| 校准量 | D_full | Y（只留S*） | 冻结要求 |
|---|---:|---:|---|
| 一阶预测与实测增量ρ1 | 0.999831 | 0.999957 | ≥0.9 |
| 一阶累计比例q1 | 0.986784 | 1.000216 | 0.8–1.25 |
| 重构位移余弦中位数 | 0.940242 | 同值 | ≥0.9 |
| 重构位移范数比中位数 | 1.053491 | 同值 | 0.8–1.25 |
| 账本与实测增量ρ3 | 0.996768 | 0.998809 | ≥0.9 |
| 账本累计比例q3 | 1.019959 | 1.153358 | 0.8–1.25 |
| 增量一阶差分相关ρ3d | 0.954419 | 0.996995 | ≥0.5 |

本次从前33段（10→340）的`segments.json`重算，全部与结果文件一致。M3主档为“校准”，说明该单轨迹的这些一阶近似能较好复原观测增长；它**不自动证明每行贡献具有真实干预意义、位置矩阵能够外推，或共有形成机制已建立**。段内固定预条件、关闭dropout、用附近检查点梯度、一阶可加、位置矩阵沿用原微批归一系数等A1–A5限制仍在。

#### 谁在推：干净数据净作用的方向取决于所测函数

| 累计至328的账本量 | D_full | Y |
|---|---:|---:|
| 毒行正贡献 | +3.659568 | +8.138308 |
| 毒行负贡献 | 0 | −0.000003 |
| clean正贡献 | +2.938842 | +1.344872 |
| clean负贡献 | −2.193133 | −2.722301 |
| clean净贡献（正负项之和） | **+0.745709** | **−1.377429** |
| 账本总L328 | 4.405277 | 6.760876 |
| 毒行净份额π | 0.830724 | 1.203735 |

起飞前实际有54条毒行，但表内逐行计账n_poison=52，因为两条在训练step3/7、属于前10步带入范围；位置矩阵仍使用54条。不要把这个口径差读成丢了两条数据。

毒行相对随机clean行平均贡献的“特异性”在合同里已注明近乎必过，不作为强证据。真正值得保留的是：**full上clean净贡献为正，否定原先简单“clean总在冲淡后门”的先验；Y上clean净贡献为负，又说明符号依赖读出的函数和模块条件。**两者不能相减得出一个严格可加的第三部分，因为Y不是full的加法子项。

#### 晚到效力：账本矩阵有强位置差异，归属不能讲成单因子已解决

固定54条内容，分别放在36个检查点位置重算账本位置核；晚位280–320与早位50–120比值：

| 读出 | 晚/早Q与95%行bootstrap区间 | 冻结单因子后的Q | 归属结局 |
|---|---|---|---|
| D_full | **33.1594 [28.8496,37.8038]** | 系数37.3362；预条件142.0410；梯度场/行梯度无有效正比；裁剪26.2821 | **无单一来源** |
| Y | **148.4273 [120.5744,179.2229]** | 系数167.1938；预条件696.5565；梯度场3.5566；行梯度20.8277；裁剪118.6500 | 按规则归给梯度场c_grad |

两者“晚位更有效”的账本判定成立。full没有单因子解释，不能把Y的c_grad归属外推到full，更不能说这已经证明晚到样本实际因果效力必为33倍/148倍。这里是固定原轨迹状态上的反事实线性矩阵，真实移位会改变后续状态和反馈，必须由PL/NF等训练干预裁决。

#### PL预测已落盘，但确认性前提明确为false

保存预测位移：R30a−2.3653、R30b−3.3766、L30+19.6439、M30−8.1595、E30−0.1129、Lend+27.6520、Eend+0.5780步，尺度b=0.974011。`pred_pl.json`明确记录`confirmatory_premise=false`，因为M2-0为D不计时；无论与PL表面有多接近，按合同都只能作为描述。另有程序性盲性条件：须核冻结时点与已评估PL读数/日志，不可因为预测代码没有自由参数就自动声称完全独立盲预测。

因此当前M3不能总结为“已解释并预测顺序效应”。更准确的是：**单轨迹的逐行一阶账本完成校准；它提供了带符号贡献与位置效力的可检验线索，但固定D阈值计时失败，使训练干预预测的确认链条断开。**

来源：[M3冻结合同](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/m_code/m3/CONTRACT_M3.md)、[结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/results_m3.json)、[原始42段账本](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/segments.json)、[重放与协议闸门](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/gates.json)、[微批身份与执行状态](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/ledger_status.json)、[PL描述性预测](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/pred_pl.json)。

### 5. 这批结果对研究进度的更新

相比“只知道ASR会跳”，我们现在确实增加了终态模块级定位与一个已校准的逐行账本；早期IP局部表示转移小也没有被抹掉，因为它问的是不同问题。但是M2没有把终态定位和跨顺序起飞时刻连起来，M3的PL确认性前提未通过，尚缺对相同候选模块/功能在训练形成阶段的局部因果干预与严格质量控制。单模型、单初始化、三条顺序、100个EVAL探针的证据不能升级成跨模型统一理论。所有份额超1、空值、Infinity与不同分母均按本来含义解释，没有改数使叙事更顺。


<a id="part-6"></a>

## 六、综合结论、当前状态与后续缺口

### 跨阶段综合：哪些现象真正累积成了结论

#### 现象层：最强的证据是异质、条件性、非单调，而非一个统一临界点

跨模型实验把“ASR曲线会突然变高”变成了明确的窗口、分母、置信区间和重放资格；同时留下T5、Falcon、Qwen-VL部分配方、T2I采样与Vim重放等负结果或不确定性。Qwen3-8B的短窗可以很窄，但后续大幅回落并不少见；mid16的完整记录又证明零终点可能掩盖中期强攻击。因此，应同时报告**起飞、窗宽、峰值、回落、终态与正常质量**。仅给最大ASR或终点ASR都不够。

现象层已经可以明确反对几种过度简化：不存在由当前材料支持的全架构统一固定毒数、普遍单调增长、所有攻击都晚学、或“低终态意味着训练中从未形成”。但它也没有证明所有训练都必然跃升、所有短窗共享机制，或中期高风险必然等于最终部署失败；实际风险取决于中途checkpoint是否可被使用。

#### 机制层：已经排除了若干解释，尚未连通“定位—动态—干预—预测”

| 曾经的解释/期待 | 后续实际证据 | 当前结论 |
|---|---|---|
| ASR跃升只是离散argmax造成的视觉台阶 | 部分连续logit也出现加速；另一些轨迹判不清 | 纯离散解释不足以覆盖已测阳性，但连续加速不自动等于相变机制 |
| 固定累计毒数或LR加权暴露可预测起飞 | 新seed盲预测误差超过常数基线；顺序重组仍改变结果 | 简单计数时钟不够；早期两点拟合不能外推 |
| 触发归因占比提前就是内部前兆 | 原始作用量不领先，占比受饱和影响 | 该前兆解释撤回 |
| 某个局部状态可搬走大部分跃升差异 | IP转移非零但低于登记主要转移阈值 | 已测局部状态的双向交换未达到主要转移阈值；不能推广成无内部机制 |
| 早期毒作用被普遍冲刷、后期才留住 | A′/WO等有限协议不支持这一概括；真实mid16又确有行为消退 | 单脉冲margin留存与已形成后门长期ASR是两个问题 |
| 固定方向形成，之后只是幅度增加 | 网格初步解释被后续方向/正交分量对照推翻 | 不能继续以固定方向作为统一机制 |
| 正常任务已经学会之后才进入危险阶段 | 有早装入反例；gate处正常正确仍很低；GATE仅支持促进作用 | “正常全部学完”与“唯一必要门”未成立 |
| 终态关键模块就一定是形成必经点 | M1定位强；M2计时未闭环；持续训练局部干预往往被追上 | 终态承载、形成必需与防御选区必须分开验证 |
| 一阶训练账本能解释并预测顺序效应 | M3校准好，晚位效力差异大；PL确认性前提未过 | 工具校准阳性，完整因果与预测闭环尚无 |

真正的进展是把原先混在一起的问题拆开了：**行为何时出现、谁在当前模型中承载它、过去哪些更新对所选读数有贡献、限制哪里会改变未来形成、能否对新轨迹作正确预测**。现有实验分别回答了其中一部分，还没有用同一套固定规则串成一个完整机制。

#### 防御层：有效性、相对收益与可部署性是三道不同的门

Curfew在固定真实任务上的成功不是假的；同样，强LT的成功也不能被删去。当前没有依据把Curfew称作唯一有效方法，不能把“固定预热也通过”叫作统计等效，也不能仅凭一次失败就宣布所有自适应时机都没有价值。

参数路线也类似：1106某些时刻确有大ASR下降、正常表现未同步下降，不能一律说成全局学习减速；但随机位置可以相似，2101/2102确认弱，+250共同终点攻击追上，又使其不能称持续防御。功能更新筛选的新终局进一步表明，按当前固定候选家族没有选出跨开发轨迹稳定通过的干预区域；它没有测试全部动态或条件控制。

**防御的最终比较应落在相近正常新学习水平下的风险，以及完整训练和评测成本。** 同时报告旧知识、新知识、触发下正确、攻击命中与训练步数；固定预算比较和固定质量比较回答不同问题，不应只挑有利的一种。使用已知触发器或毒标签的定位实验，还需单独解决未知攻击下如何提出候选和执行策略。

### 当前状态表：已经完成、失败、未运行与缺口分别是什么

| 工作包 | 截至本次核对的状态 | 仍缺什么 |
|---|---|---|
| 初始LLaVA/Qwen-VL跃升与剂量 | 有真实生成、密窗与多项对照；部分历史时刻已纠正 | 不能推广固定剂量/样本数规律 |
| 跨模型统一测量 | 主视图、补窗、独立同点确认与诊断已归档 | LLaVA部分窗口、T2I稀疏区间及Vim重放身份问题仍限制结论 |
| 监测/回退toy | r02真实完成，未找到投毒质量—风险合格回退点 | 有用回退窗口与实际防御收益 |
| 测量提速 | 开发计时和一次新初始化确认完成 | 新数据顺序、未知触发器、更多轨迹及对最强周期法的增量 |
| 攻击搜索/SAILS方向 | 有完整设计与路线讨论，未找到执行闭环 | 候选集合排序、强搜索对照、实际完整成本和新集合验证 |
| JSD | 开发失败，未满足留出启动条件 | 不能把未跑留出当阴性，也不应跳过原信号失配 |
| Curfew | toy/真实数据有阳性；已补强LT与阈值审计 | 在可比正常学习与成本下的稳定独立收益 |
| 真实mid16 | 补曲线完成，确认形成—消退—反弹 | 共同完整状态续供/屏蔽/替换因果对照及新轨迹确认 |
| fork/A′/网格/方向/Atlas等 | 大量局部与跨族读数完成，强解释多有反例 | 可复现、能预测新轨迹的统一解释 |
| 激活形成S1/S2/P | 有表示收敛线索；6/10终态工程门未过；十seed描述性窗前方向领先为0/10，隐藏进度未跑通 | 按原批次复核；本线因果S3与第四部分顺序S3不是同一合同 |
| WO/SW/GATE | 本次已取到后续服务器结果 | 不能由合同命名升级为普遍必要性或窗口独占性 |
| R2语言L0–L3 | 12臂完成记录存在，汇总L3_s1005仍null | 用其原输出补齐汇总并核对，不能补零或称训练没跑完 |
| 粗冻结/分配/参数选择性 | 有完整负结果和正常学习代价记录 | 尚无持续优于强LT的策略 |
| rejected-update | 用户中止，正式共同终点0/30 | 属于中止，不把它计为完整科学阴性 |
| functional-update | 筛选完成；未找到通过选择性核验的候选；12后续臂科学跳过 | 新候选规则需另立问题，不能称12次训练失败 |
| 57/54/28局部干预 | 139个新正式任务已完成 | 稳定选区、强随机/减速对照、持续终点收益、与Curfew组合验证 |
| SF/B3/J09/PL/CL/NF | 已有原始行为/连续读数；本次重新统计 | 抖动尺度、效果量预测、因果拆分与跨条件外推 |
| B3D-M0 | 本次见准备核验，但未取得正式臂完整读数；当前指定arms目录不存在 | 恢复对应执行/撤回记录，不能凭目录缺失判科学阴性 |
| M1/M2/M3 | 定位与账本校准完成；计时/预测主张未闭环 | 形成期干预和满足前提前瞻预测 |
| S3 | 10条新臂+2历史对照，有已选顺序下0.42%分叉 | 随机顺序频率、误拒侧、预算和更多条件 |
| DW8 | 8/8完成；两新order四剂量，无二值分叉但有连续差异 | 不足以估计总体概率或连续剂量宽度 |
| A1/A2/A3/B2验收后续 | 本文未取得与原计划一一对应的完整结果链 | 先核执行身份；不得用DW8或M2 clean twin冒充全部完成 |

本次服务器选择性状态快照为**10月27日12:34北京时间**，见[核对快照](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/server_status_selected.json)。其中队列`done`也可能表示“检查后按合同跳过”，必须结合任务结果解释，functional-update就是实例。该快照只覆盖列出的ASR批次，不代表整台服务器所有任务状态。

### 已有投入怎样计数，怎样避免把工作量写成贡献

| 批次 | 可核的实际工作量/时间 | 口径边界 |
|---|---|---|
| 10月7日跨模型阶段 | 57条正式训练模型阶段86.2178 GPUh；加列明非重叠项101.0638 GPUh | 阶段归档，不是全部项目总账 |
| 10月10–11日监测r02 | 2臂×160更新；外层4116.650秒 | 含密集测量；5影子策略不是5次独立训练 |
| 测量开发 | 16状态、960路径测量；1766.818秒 | 0训练更新；计时重复不是独立科学seed |
| 新初始化测量确认 | 2臂合计320更新，约2.366小时 | 数据顺序仍固定，不是顺序独立确认 |
| mid16补测 | 1条严格配方重放；6498.38秒 | 103点、18,540条输出不是103次训练 |
| 10月19–20日三批局部干预 | 57+54+28=139个新任务；32.8051任务墙钟小时 | 已复用R不再计新任务；不含全部失败和准备 |
| functional-update筛选 | GPU筛选/核验约0.09874小时 | 12后续训练跳过；9条匹配基线复用 |
| S3 | 10条新训练+2历史对照 | 不写12个独立新训练；不从3初始化估顺序频率 |
| DW8 | 8新臂，10.0037任务卡时 | 2卡合计不等于整体墙钟；与S3不混成随机样本总体 |

不同账本分别有GPU占用、进程墙钟、CPU命令墙钟、估算成本、重复测量和复用对照；有的历史费用还有未知项。因此本文**不提供一个看似精确但无法去重的“ASR全线总GPU小时”或“总实验次数”**。投入证明做过多少工作，不能替代方法收益、普遍性或新颖性。

### 还差什么：按需要回答的问题列，而不是继续扩张实验矩阵

以下是由现有证据导出的待办判断，不代表本次启动实验或改动已冻结计划。

1. **先补齐记录层面的最后缺口。** 对账R2-L3空格、B3D-M0身份、A1/A2/A3/B2执行与未执行状态；将原合同、实际训练、评估、判读四层对应起来。训练完成不等于汇总完成，队列完成也不等于训练发生。
2. **顺序/验收主线先检验频率与误拒。** 固定同一剂量、poison集合、init与预算，用预先指定的新order估计所选分布中的形成频率；并配干净孪生评估正常模型会不会被误拒。历史挑出的极端顺序用于证伪，不用于概率估计。DW8是有价值的有限网格验证，但不能替代这一目标。
3. **若继续预测，先冻结读数和最强简单对照。** 分清首次与持续起飞，事先明确允许误差和不可判规则；先有足够的同条件波动参照，再比较计数、常数时刻、loss/margin与复杂账本。不能在PL/CL/NF之间不断换容差寻找通过。
4. **机制若要闭环，必须把承载与形成连接起来。** M1给出的集合需要接受同语义训练期限制，和同结构/同规模随机组、整体减速及无干预比较；M3账本若用于选行或预测，应满足读数同步前提，并在未用于拟合的轨迹上检验。对已知终态做漂亮归因不能替代这一环。激活形成线还需先复核6/10终态批处理差异，再验证方向消融、注入与迁移；这与LLaVA同名顺序S3不是同一批验证。
5. **防御若要成立，以同等正常学习水平的持久收益为目标。** 保留强LT20/LT50，核清官方锚点与本地适配；记录实际更新、保留token归一化、旧新知识质量与全成本。仅短期延迟、ASR下降但正常回答受损，或只在开发seed上选出的H1，均不足以完成方法。
6. **留存因果问题保持窄而明确。** mid16已证明发生过强行为且后来消退；若要解释原因，需要共同完整状态的未来分叉，隔离续供、屏蔽与正常替换，明确其余归一化和优化器状态。没有这个反事实，就不要把“没有新毒行之后下降”写成“持续供毒必要”。
7. **新颖性需围绕最后实际成立的主张重新核查。** 历史相关工作审查与论文定位曾随结果变化。本文整理了这些变化，但没有新做一次截至今日的全网查重；不能把旧“未搜到”当新颖性证明，更不预先许诺会议录用档位。要先知道最后交付的是条件性现象、测量工具、机制干预还是防御方法。

当前最值得保留的工作方式，是把每一个解释写成能失败的判断，并接受失败后收窄结论。现有材料已经足以形成一份可信的研究记录；要变成明确的新方法或强机制贡献，仍需上述相应环节的实证闭合。

### 本次整理的核验与解释检查

本次重点核查了：初始逐样本ASR与剂量表、跨模型主视图、IP七对NPZ的冻结评估复算、Curfew逐臂/阈值数字、新初始化测量确认、机制分支结果、WO/SW/GATE最新产物、functional-update终局、139条后期干预表、mid16原输出重计、SF/PL/CL/NF等25条连续读数、DW8八臂逐题ASR/clean，以及M2/M3部分原始量与校准统计。**这不是对全体模型的重新运行，也不是穷尽每个大型原始产物的独立审计。**

| 解释检查（11/11） | 本文处理 |
|---|---|
| 分组反转 | 模型、剂量、任务、seed、触发与正常条件分开列 |
| 群体推个体 | 聚合margin不直接推出单题机制；mid16反弹按逐题身份核实 |
| 选择偏差 | 标明开发seed、选中顺序、后验候选、复用对照与已见探针 |
| 条件化偏差 | gate/形成状态是观测结果，不能当随机外生干预 |
| 基率混淆 | 毒率、毒召回、删行比例、攻击命中和正常质量分别报告 |
| 极值回落 | 保留非单调曲线、反弹、峰值后追上与共同终点 |
| 幸存者偏差 | 保留技术失败、不可判、中止、科学跳过及采样不足 |
| 多处找胜点 | H1/模块筛选与确认分开；不把最好checkpoint代替预定窗口 |
| 分析分叉 | 记录定义演变与纠错；不追认事后预测为预注册 |
| 相关与因果 | 状态定位、账本分解、训练期干预和部署收益分开 |
| 内生/反向解释 | 不把loss过线视为形成原因；不把过线后的变化倒写成预警 |

已纠正或补齐的主要旧口径：LLaVA1001首次t50漏测；D2的8/44应为8.5/44.5%；IP并非未执行；fx2为10/14而非无说明的10/13；测量fixed10相位为6；Atlas为56输入=50通过+6失败；R1列表为40而非44；mid16满命中从110而非105开始、230非严格归零；WO/SW/GATE和functional状态更新；M1早晚Jaccard0.125不能解释为完全换电路；NF的“有限支集”标签不能升级成已证实数学性质。

证据清单见[来源与SHA清单](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/SOURCE_MANIFEST.json)、[本次核验记录](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-line-summary-2026-10-27/VALIDATION.json)。本次只新增总结及其补回证据，没有改写旧记录以掩盖错误，也没有新增训练、推理或队列操作。


<a id="appendix"></a>

## 附录：跨模型主口径逐项清单

共96行，是主投毒轨迹及明确单列的替代前缀视图，不是96次独立完整训练。Vim原轨迹与0–120前缀分别保留，同seed不得叠加成独立重复。原LLaVA另投毒集单列。主1%阈值为完整注册预算的1%内增长50个百分点，表中的终点为该视图最后记录点；短前缀终点不等于完整1250终点。

数值来源：[inventory.csv](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/inventory.csv)。主判无候选时，CSV的后续回落字段常为0，这只是无首次事件锚点；本表改列—，不解读为无回落。

| 模型 | 剂量 | seed | 视图 | n | 主1%状态 | 首次点对 | 增幅pp | W50上界 | 初始/峰值/末点ASR% | 首次后最大回落pp |
|---|---:|---:|---|---:|---|---|---:|---:|---|---:|
| Qwen3-8B | 1% | 1001 | 原主轨迹 | 60 | 已观测 | 89→92 | 56.667 | 3 | 0 / 100 / 100 | 56.667 |
| Qwen3-8B | 1% | 1002 | 原主轨迹 | 60 | 已观测 | 105→110 | 50 | 5 | 0 / 100 / 100 | 11.667 |
| Qwen3-8B | 1% | 1003 | 原主轨迹 | 60 | 已观测 | 103→107 | 51.667 | 3 | 0 / 100 / 100 | 15 |
| Qwen3-8B | 1% | 1004 | 原主轨迹 | 60 | 已观测 | 82→84 | 50 | 2 | 0 / 100 / 100 | 43.333 |
| Qwen3-8B | 1% | 1005 | 原主轨迹 | 60 | 已观测 | 75→81 | 56.667 | 6 | 0 / 100 / 100 | 20 |
| Qwen3-8B | 1% | 1006 | 原主轨迹 | 60 | 已观测 | 80→83 | 66.667 | 3 | 0 / 100 / 100 | 10 |
| Qwen3-8B | 1% | 1007 | 原主轨迹 | 60 | 已观测 | 120→122 | 61.667 | 2 | 0 / 100 / 100 | 11.667 |
| Qwen3-8B | 1% | 1008 | 原主轨迹 | 60 | 已观测 | 91→96 | 55 | 4 | 0 / 100 / 100 | 1.667 |
| Qwen3-8B | 1% | 1009 | 原主轨迹 | 60 | 已观测 | 88→90 | 55 | 2 | 0 / 100 / 100 | 96.667 |
| Qwen3-8B | 1% | 1010 | 原主轨迹 | 60 | 已观测 | 129→136 | 50 | 7 | 0 / 100 / 100 | 5 |
| SD3.5-Large | 1% | 1001 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 1.667 | — |
| SD3.5-Large | 1% | 1002 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 3.333 | — |
| SD3.5-Large | 1% | 1003 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 1.667 | — |
| SD3.5-Large | 5% | 1004 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 3.333 | — |
| SD3.5-Large | 5% | 1005 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 0 | — |
| SD3.5-Large | 5% | 1006 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 0 | — |
| SD3.5-Large | 5% | 1007 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 1.667 | — |
| SD3.5-Large | 5% | 1008 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 0 | — |
| SD3.5-Large | 5% | 1009 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 3.333 | — |
| SD3.5-Large | 5% | 1010 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 1.667 | — |
| SD3.5-Large | 10% | 1004 | 原主轨迹 | 60 | 不可判 | — | — | — | 6.667 / 6.667 / 1.667 | — |
| Falcon-Mamba-7B | 5% | 1001 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 71.667 / 51.667 | — |
| Falcon-Mamba-7B | 5% | 1002 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 38.333 / 6.667 | — |
| Falcon-Mamba-7B | 5% | 1003 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 40 / 3.333 | — |
| Falcon-Mamba-7B | 5% | 1004 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 8.333 / 0 | — |
| Falcon-Mamba-7B | 5% | 1005 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 6.667 / 0 | — |
| Falcon-Mamba-7B | 10% | 1001 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 70 / 61.667 | — |
| Falcon-Mamba-7B | 10% | 1002 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 18.333 / 3.333 | — |
| Falcon-Mamba-7B | 10% | 1003 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 40 / 6.667 | — |
| Falcon-Mamba-7B | 15% | 1001 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 35 / 6.667 | — |
| Falcon-Mamba-7B | 15% | 1002 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 28.333 / 3.333 | — |
| Falcon-Mamba-7B | 15% | 1003 | 原主轨迹 | 60 | 不可判 | — | — | — | 0 / 20 / 18.333 | — |
| resnet50 | 5% | 1001 | 原主轨迹 | 180 | 已观测 | 20→25 | 60 | 5 | 0 / 100 / 100 | 0.556 |
| resnet50 | 5% | 1002 | 原主轨迹 | 180 | 已观测 | 15→20 | 53.333 | 5 | 0 / 100 / 100 | 0.556 |
| resnet50 | 5% | 1003 | 原主轨迹 | 180 | 已观测 | 15→20 | 79.444 | 5 | 0 / 100 / 100 | 0.556 |
| resnet50 | 5% | 1004 | 原主轨迹 | 180 | 已观测 | 20→25 | 88.889 | 5 | 0 / 100 / 100 | 0.556 |
| resnet50 | 5% | 1005 | 原主轨迹 | 180 | 已观测 | 15→20 | 70 | 5 | 0 / 100 / 100 | 0 |
| resnet50 | 5% | 1006 | 原主轨迹 | 180 | 已观测 | 20→25 | 71.111 | 5 | 0 / 100 / 100 | 0 |
| resnet50 | 5% | 1007 | 原主轨迹 | 180 | 已观测 | 20→25 | 72.222 | 5 | 0 / 100 / 100 | 1.111 |
| resnet50 | 5% | 1008 | 原主轨迹 | 180 | 已观测 | 10→20 | 50 | 10 | 0 / 100 / 100 | 0.556 |
| resnet50 | 5% | 1009 | 原主轨迹 | 180 | 已观测 | 20→25 | 62.222 | 5 | 0 / 100 / 100 | 1.111 |
| resnet50 | 5% | 1010 | 原主轨迹 | 180 | 已观测 | 15→20 | 63.889 | 5 | 0 / 100 / 100 | 0.556 |
| vit_b_16 | 5% | 1001 | 原主轨迹 | 180 | 已观测 | 25→30 | 63.333 | 5 | 0 / 100 / 100 | 1.667 |
| vit_b_16 | 5% | 1002 | 原主轨迹 | 180 | 已观测 | 25→30 | 89.444 | 5 | 0 / 100 / 100 | 1.111 |
| vit_b_16 | 5% | 1003 | 原主轨迹 | 180 | 已观测 | 30→35 | 70.556 | 5 | 0 / 100 / 100 | 6.667 |
| vit_b_16 | 5% | 1004 | 原主轨迹 | 180 | 已观测 | 35→40 | 81.111 | 5 | 0 / 100 / 100 | 1.667 |
| vit_b_16 | 5% | 1005 | 原主轨迹 | 180 | 已观测 | 25→30 | 70 | 5 | 0 / 100 / 100 | 1.111 |
| vit_b_16 | 5% | 1006 | 原主轨迹 | 180 | 已观测 | 30→35 | 81.111 | 5 | 0 / 100 / 100 | 1.111 |
| vit_b_16 | 5% | 1007 | 原主轨迹 | 180 | 已观测 | 30→35 | 70 | 5 | 0 / 100 / 100 | 1.667 |
| vit_b_16 | 5% | 1008 | 原主轨迹 | 180 | 已观测 | 30→35 | 83.333 | 5 | 0 / 100 / 100 | 2.222 |
| vit_b_16 | 5% | 1009 | 原主轨迹 | 180 | 已观测 | 30→35 | 56.111 | 5 | 0 / 100 / 100 | 1.667 |
| vit_b_16 | 5% | 1010 | 原主轨迹 | 180 | 已观测 | 20→25 | 92.222 | 5 | 0 / 100 / 100 | 1.667 |
| LLaVA-1.5-7B | 1% | 1001 | 原主轨迹 | 200 | 已观测 | 350→360 | 51 | 10 | 0 / 99.5 / 98.5 | 15.5 |
| LLaVA-1.5-7B | 1% | 1004 | 原主轨迹 | 200 | 不可判 | — | — | — | 0 / 99.5 / 99.5 | — |
| LLaVA-1.5-7B | 1% | 1005 | 原主轨迹 | 200 | 不可判 | — | — | — | 0 / 99.5 / 99.5 | — |
| LLaVA-1.5-7B | 1% | 1006 | 原主轨迹 | 200 | 已观测 | 375→383 | 50 | 8 | 0 / 99.5 / 99 | 17.5 |
| LLaVA-1.5-7B | 1% | 1007 | 原主轨迹 | 200 | 已观测 | 437→445 | 54 | 8 | 0 / 99.5 / 99.5 | 4 |
| LLaVA-1.5-7B | 1% | 1008 | 原主轨迹 | 200 | 已观测 | 390→398 | 51 | 5 | 0 / 99.5 / 99 | 8 |
| LLaVA-1.5-7B | 1% | 1009 | 原主轨迹 | 200 | 已观测 | 320→327 | 51.5 | 7 | 0 / 99.5 / 98.5 | 5.5 |
| LLaVA-1.5-7B | 1% | 1010 | 原主轨迹 | 200 | 不可判 | — | — | — | 0 / 99.5 / 99.5 | — |
| LLaVA-1.5-7B | 1% | 1011 | 原主轨迹 | 200 | 已观测 | 390→401 | 51 | 7 | 0 / 99.5 / 98.5 | 8 |
| LLaVA-1.5-7B | 1% | 1002 | 另投毒集 | 200 | 不可判 | — | — | — | 0 / 99.5 / 98.5 | — |
| Qwen3-VL-8B | 1% | 1001 | 原主轨迹 | 200 | 已观测 | 470→474 | 51 | 4 | 0 / 99.5 / 99 | 1.5 |
| Qwen3-VL-8B | 1% | 1004 | 原主轨迹 | 200 | 不可判 | — | — | — | 0 / 1 / 0 | — |
| Qwen3-VL-8B | 1% | 1005 | 原主轨迹 | 200 | 不可判 | — | — | — | 0 / 2 / 0 | — |
| Qwen3-VL-8B | 1% | 1006 | 原主轨迹 | 200 | 不可判 | — | — | — | 0 / 4.5 / 0 | — |
| Qwen3-VL-8B | 1% | 1007 | 原主轨迹 | 200 | 不可判 | — | — | — | 0 / 4.5 / 4.5 | — |
| Qwen3-VL-8B | 5% | 1004 | 原主轨迹 | 200 | 已观测 | 88→96 | 53 | 5 | 0 / 100 / 100 | 3 |
| Qwen3-VL-8B | 5% | 1005 | 原主轨迹 | 200 | 已观测 | 131→134 | 51.5 | 3 | 0 / 100 / 100 | 1.5 |
| Qwen3-VL-8B | 5% | 1006 | 原主轨迹 | 200 | 已观测 | 180→182 | 55 | 2 | 0 / 100 / 100 | 2 |
| Qwen3-VL-8B | 5% | 1007 | 原主轨迹 | 200 | 已观测 | 180→184 | 51 | 3 | 0 / 100 / 100 | 3 |
| Vim-S | 5% | 1001 | 原主轨迹 | 180 | 已观测 | 60→65 | 82.778 | 5 | 0 / 100 / 100 | 0 |
| Vim-S | 5% | 1002 | 原主轨迹 | 180 | 已观测 | 80→85 | 84.444 | 5 | 0 / 100 / 100 | 0 |
| Vim-S | 5% | 1003 | 原主轨迹 | 180 | 已观测 | 55→60 | 85 | 5 | 0 / 100 / 100 | 0 |
| Vim-S | 5% | 1004 | 原主轨迹 | 180 | 已观测 | 65→70 | 68.889 | 5 | 0 / 100 / 100 | 0.556 |
| Vim-S | 5% | 1005 | 原主轨迹 | 180 | 已观测 | 40→45 | 62.222 | 5 | 0 / 100 / 100 | 0 |
| Vim-S | 5% | 1001 | 独立0–120前缀 | 180 | 已观测 | 59→62 | 53.333 | 2 | 0 / 100 / 100 | 0 |
| Vim-S | 5% | 1002 | 独立0–120前缀 | 180 | 已观测 | 63→68 | 52.222 | 3 | 0 / 100 / 100 | 0 |
| Vim-S | 5% | 1003 | 独立0–120前缀 | 180 | 已观测 | 45→57 | 50 | 2 | 0 / 100 / 100 | 0.556 |
| Vim-S | 5% | 1004 | 独立0–120前缀 | 180 | 已观测 | 77→79 | 54.444 | 2 | 0 / 100 / 100 | 0 |
| Vim-S | 5% | 1005 | 独立0–120前缀 | 180 | 已观测 | 40→45 | 52.778 | 4 | 0 / 100 / 100 | 0 |
| Llama-3.1-8B-Instruct | 1% | 1001 | 原主轨迹 | 60 | 已观测 | 66→69 | 56.667 | 3 | 0 / 100 / 100 | 6.667 |
| Llama-3.1-8B-Instruct | 1% | 1002 | 原主轨迹 | 60 | 已观测 | 104→107 | 55 | 3 | 0 / 100 / 100 | 3.333 |
| Llama-3.1-8B-Instruct | 1% | 1003 | 原主轨迹 | 60 | 已观测 | 55→57 | 76.667 | 2 | 0 / 100 / 100 | 65 |
| Llama-3.1-8B-Instruct | 1% | 1004 | 原主轨迹 | 60 | 已观测 | 67→70 | 51.667 | 2 | 0 / 100 / 100 | 46.667 |
| Llama-3.1-8B-Instruct | 1% | 1005 | 原主轨迹 | 60 | 已观测 | 64→66 | 51.667 | 2 | 0 / 100 / 100 | 8.333 |
| FLAN-T5-base | 1% | 1001 | 原主轨迹 | 60 | 已测未观测 | — | — | — | 0 / 0 / 0 | — |
| FLAN-T5-base | 1% | 1002 | 原主轨迹 | 60 | 已测未观测 | — | — | — | 0 / 0 / 0 | — |
| FLAN-T5-base | 1% | 1003 | 原主轨迹 | 60 | 已测未观测 | — | — | — | 0 / 0 / 0 | — |
| FLAN-T5-base | 1% | 1004 | 原主轨迹 | 60 | 已测未观测 | — | — | — | 0 / 0 / 0 | — |
| FLAN-T5-base | 1% | 1005 | 原主轨迹 | 60 | 已测未观测 | — | — | — | 0 / 0 / 0 | — |
| CLIP-ViT-B32 | 5% | 1001 | 原主轨迹 | 180 | 已观测 | 110→175 | 53.333 | 25 | 12.778 / 100 / 100 | 7.778 |
| CLIP-ViT-B32 | 5% | 1002 | 原主轨迹 | 180 | 已观测 | 160→185 | 53.889 | 25 | 12.778 / 100 / 100 | 13.889 |
| CLIP-ViT-B32 | 5% | 1003 | 原主轨迹 | 180 | 已观测 | 125→145 | 50 | 20 | 12.778 / 100 / 100 | 26.667 |
| CLIP-ViT-B32 | 5% | 1004 | 原主轨迹 | 180 | 已观测 | 125→165 | 57.222 | 25 | 12.778 / 100 / 100 | 18.889 |
| CLIP-ViT-B32 | 5% | 1005 | 原主轨迹 | 180 | 已观测 | 90→120 | 60 | 25 | 12.778 / 100 / 100 | 6.111 |
