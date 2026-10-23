# S3b 七步查重：起飞调度器（进行中）

对象：PLAN_METHOD S4 的方法主张——**"在固定毒集下，按位置调度毒行可把起飞步打到指定步（±10），不需要 oracle 微调"**，以及它依赖的测量 **K(d)**（位于起飞前 d 步的毒行每挪 1 步，起飞挪几步）。

协议：`~/.claude/CLAUDE.md` §4，顺序按 `related-work-protocol` 改为 **6 → 5 → 1/2/3/4 → 7**（对抗轮前置）。
**状态：未完成。** 已做：第 6 步第一批（我）、第 1/2/4 步（子代理，已交付）。在跑：第 5 步引用链、第 3 步蕴含（两个子代理，结果未到）。第 7 步边界见 §5，随后续补。
**分层纪律**（`related-work-protocol`）：方法层被占**不构成淘汰**，只进"工具已存在于何处"表；对象层、结论层被占才致命。每条结论都标层。

## 0 坏消息先说

1. **"调度数据可以改变知识被学会的时刻"这句话已经被发表过**，而且因变量是步数：Zucchet 等（DeepMind，2503.21676）把 plateau 结束步当测量量，发现数据分布移动它，并以数据调度建议收尾。**结论层占了这个句子**，但杠杆是频率不均衡、不是消耗顺序中的位置，目标是加速、不是打到指定步，也没有响应核。**我们的框架不能写成这句话。**
2. **我们主张的平凡读法已在印刷品里**：Chang 等（NeurIPS 2024，2406.11813）在受控预训练步注入事实，逐字「Note the immediate and distinctive increase of log probability after the model is updated with the injected knowledge」。若获取就发生在曝光那一刻，那么一个 epoch 里"获取时刻由位置决定"退化为"没见过就学不到"。**论文的承重必须整条压在 K(d) ≠ δ(d) 上**——即挪动**别的**行也能移动获取时刻——不能压在标题句上。
3. **有一篇高引论文对我们的杠杆给了发表过的负结果**，而且是会被直接写进审稿意见的那句话。
4. **仪器层被占得很彻底**，其中一个占位者的用词与我们撞车（见 §3）。

**未发现对象层占位**：以"某内容在消耗顺序中的位置"为自变量、"该内容被获取的时刻"为因变量的测量；位移响应核；把获取事件打到选定步的调度。两轮定向检索（扩展模式）在这个槽位上都空。

## 1 我自己逐字核过的（curl + `pdftotext -layout`，不经摘要器）

### 1.1 Pythia（ICML 2023，2304.01373）§3.2：训练顺序对记忆"影响很小"

> Surprisingly, we find that a Poisson model fits the data extremely well (Figure 3), indicating that **training order has little impact on memorization**. This model implies that memorized sequences are not spaced more densely toward the beginning or end of training…
> This finding is important for practitioners seeking to control which sequences are memorized by a model. **It implies that one cannot simply place sequences that are undesirable to memorize at the beginning or end of training**…

**分层：结论层、负向、但在量的轴上。** 口径（已核）：判据是"每批中被记忆序列的**条数密度**"对最佳拟合泊松分布的 Q-Q 拟合，批大小 512；测的是**聚合密度是否均匀**，不是任何单条内容的获取**时刻**。所以它不是对我们的真实反驳（5.4：量与何时是独立的轴）。
**但它是最会被引来杀我们的一句话**，必须在相关工作里正面处理，并说清两处口径差异：(i) 他们的因变量是记忆与否的密度、我们的是过零步；(ii) 他们的位置粒度是"训练的开头/结尾"、我们的是逐行 1 步。

### 1.2 Lesci 等（ACL 2024 最佳论文，2406.04327）：记忆"由数据顺序与学习率决定"

> …we find that memorisation (i) is stronger and more persistent in larger models, **(ii) is determined by data order and learning rate**…
> …schedule: **it is stronger after the warm-up period (around timestep 1.5k) than before it.**
> This result has implications for data ordering during training. For example, if there are particular instances that we do not want the model to memorise, but which we still want to use during training, **they should be included in earlier batches.**

**分层：结论层（位置 → 记忆，带处方）+ 方法层（τ_{g,c} 是"看到它的批 g × 检查点 c"的因果效应矩阵，是已发表对象里形状最接近 K(d) 表的东西）。** 仍在量的轴上（记忆强度），没有获取时刻。

### 1.3 §1.1 与 §1.2 看似矛盾，**核完不是载重矛盾**（`check-protocols-before-registering-a-contradiction`）

两篇都在 Pythia 模型族上，结论方向相反（"顺序影响很小" 对 "由数据顺序决定"）。摊开口径后不成立：泊松检验问的是**聚合密度是否均匀**，DiD 问的是**单条实例的因果效应**；一个均匀的泊松密度与"每条实例各有位置效应"完全相容。**不登记为矛盾。**

### 1.4 真正的威胁加重了：学习率归因现在有两篇独立支持

- Souly 2510.07192 F.4（本会话已核，LOG J69）：「This is due to the small learning rate at the end of the fine-tuning.」
- Lesci 2406.04327（上）：把顺序与**学习率**并列为决定因素，并明确记忆强度在**预热期之后**更强。

即"位置效应 = 到达时的学习率不同"这个竞争解释，已经在**两个不同对象**上被发表。**后果**：
1. `../next-stage-plan-2026-10-23/THEORY_PREDICTIONS.md` §3 的 **lr 加权计数列不是可选项，是两篇已发表工作的注册竞争解释**。已冻结，位置正确，不改。
2. K(d) 与 lr 加权计数在 PL 七臂上**必须报差**；差不出来就如实写"与学习率解释分不开"，不得含糊。
3. 我们的设计上有一处天然优势要写明：PL 的 +30 臂在**同一条顺序、同一日程**内挪动，到达时的 λ(t) 变化可直接算出来并作为零假设列，而上面两篇都是跨整块位置比较。

## 2 子代理交付的模板层发现（第 1/2/4 步）

以下由子代理检索并声称以 `pdftotext` 核过 14 篇全文；**我本人只复核了 §1 的三篇**（Pythia、Lesci、Zucchet 下载齐，Zucchet 的逐字未逐条复核）。引用进稿件前按 `related-work-protocol` 逐条回原文。标 [未核] 者只有检索片段。

**结论层近失**：

| 文献 | 内容 | 与我们的差 |
|---|---|---|
| 2503.21676（Zucchet 等，DeepMind 2025） | plateau 结束步为测量量；数据分布与动态分布调度器移动它；「we suggest novel data scheduling strategies to accelerate neural network training」 | 杠杆是频率不均衡非位置；目标是**最小化/加速**非**打到指定步**；无响应核 |
| 2406.11813（Chang 等，NeurIPS 2024） | 在受控预训练步注入事实；测 effectivity（量）与 retainability（衰减率） | 自变量与我们同（注入时刻），因变量是**量**不是**何时**（5.4 轴不互推）。这是一条**一句话宽**的缺口 |
| 1810.10531（Saxe 等，PNAS 2019）、2209.09658（George 等） | 解析推出各模式**被学会的时刻**，按奇异值排序 / 逐模式 ε-收敛时间 | 自变量是内容性质或训练机制，不是位置 |

**对象层（获取时刻确实当因变量，但自变量从不是位置）**：Chang & Bergen（TACL 2022）与 Chang/Tu/Bergen（TACL 2024）把发展心理语言学的 **age of acquisition** 搬进语言模型，逐词/逐 token 的获取步；Toneva 等（ICLR 2019）记首次学会的呈现序号；Gekhman 等（EMNLP 2024）在 SFT 里以拟合速度为因变量。自变量都是内容性质（频率、具体性、难度、已知/未知）。

**方法层（预期被占，进"工具在哪"表）**：SOURCE（Bae 等，NeurIPS 2024）明写考虑「a data point's position during training」**并且已经把影响函数的基元叫 response function**；J.T. Wang 等 2412.09538 的 trajectory-specific LOO =「removing a data point from a specific iteration」，其未来工作还预留了"按预测影响动态调整呈现顺序与频率"；Yang 等 2505.22042 做**任意样本顺序下的免重训估计**，并直接研究"样本位置如何影响记忆与泛化"；机器教学的最优控制支（Lessard/Zhang/Zhu，AISTATS 2019）拥有"用最短训练序列把学习器驱到目标模型"。

**两处要立刻处理的用词与定位**：
1. **"响应核 / response kernel" 与影响函数文献的 response function 撞车。** 要么明确声明沿用该词并引 SOURCE，要么改名（建议改名，例如**位置位移响应 K(d)** 或 arrival-displacement kernel），避免审稿人读成影响函数的重复。
2. 机器教学owns"**最小化**到达目标的时间"与"预算内**最大化**性能"；**"把事件放在选定的非极端步 T\*"没有被占**。这是我们在控制论框架里剩下的房间，S4 的两个目标步（提前、推迟）正好落在这里，应在引言里就点明这个区别。

## 3 工具已存在于何处（论文必须带这张表）

| 我们用的仪器 | 已归属的领域 | 已占位的代表作 | 剩下的空地 |
|---|---|---|---|
| 位置位移响应 K(d) | 训练数据归因（unrolled differentiation 支）；因果推断用于 NLP | Hara 等 2019；TracIn（NeurIPS 2020，按步索引）；**SOURCE**（位置感知 + response function）；J.T. Wang 2412.09538；Yang 2505.22042；**Lesci τ_{g,c}**（位置 × 检查点因果矩阵） | 它们的因变量全是**损失/记忆强度**；没有一个以**获取时刻**为因变量，也没有一个测**挪 1 步**的导数（他们做删除、重加权或置换） |
| 调度到目标时刻 | 机器教学（最优控制支）；课程学习的 pacing function；间隔重复调度 | Lessard/Zhang/Zhu 2019（时间最优控制）；Liu 等 ICML 2017；Peltola 等 NeurIPS 2019（有限时域）；Soviany 等 IJCV 2022 | 全是**最小化**时间或**预算内最大化**性能；**把事件放到选定的非极端步**没找到 |
| 获取时刻当因变量 | 发展心理语言学的 age of acquisition（已移植进 LM 分析）；逐样本训练动力学；解析学习理论 | Chang & Bergen TACL 2022；Chang/Tu/Bergen TACL 2024；Toneva ICLR 2019；Gekhman EMNLP 2024；Saxe PNAS 2019；Zucchet 2503.21676（plateau 结束步） | 自变量永远是**内容的性质**或**优化器/机制**，从来不是**内容在消耗顺序中的位置** |

**读法**：三条腿各自被占，**乘积没有**。贡献落在"交叉"：以 AoA 式的时刻为因变量 × 位置索引的归因仪器 × 非极端的打靶目标。按 `no-novelty-purism`，转化本身就是贡献；但这张表必须进稿件，否则读起来像不知情。

## 4 对计划的后果（要改的地方）

1. **论文标题句改写**：不能是"调度数据改变学会的时刻"（Zucchet 占了）。承重句改为 **K(d) ≠ δ(d)**：挪动起飞前的毒行会移动起飞，而挪动等量的其它行不会（PL 的两条随机臂正是这个对照），以及由此可**打到指定步**。
2. **"response kernel" 更名**（见 §2 第 1 点），并在相关工作引 SOURCE 与 Lesci 的 τ_{g,c} 说明形状上的近邻关系。
3. **引言必须正面处理 Pythia 的那句话**，用 §1.1 的两处口径差。
4. **S4 合同新增一条事前登记**：K(d) 与 lr 加权计数的逐臂差；并把 Souly 的 beginning/end 两档明确标为"已发表设定的复现臂"（BLOCK-FRONT / BLOCK-BACK 已在设计里）。
5. **S3 账本不受影响**（它只用 PL 的三点 K 与去除对照）。
6. 新增必读（在写方法节前）：2407.10490（Ren & Sutherland，ICLR 2025，逐步分解"训练一条样本如何移动另一条的 log 概率"，是 K(d) 要骑的逐步跨样本核）、2505.22042、2412.09538、2406.04327。

## 5 边界申报（第 7 步，部分）

- **未完成的步**：第 5 步（引用链挖掘）与第 3 步（蕴含检索）在跑，**结果未到，不得假定**。第 6 步只做了第一批。
- **会场**：没有系统扫 ACL/EMNLP/NAACL/NeurIPS/ICML/ICLR 的 proceedings 或 OpenReview，**在审稿件未覆盖**；这个槽位占起来便宜，并行未公开工作是实际风险。
- **灰色文献**：子代理报了一篇只存在于机构仓库的硕士论文（UPF，Leybzon 2024，训练顺序与记忆持久性），PDF 未取；这说明训练顺序一族还有我没碰到的灰色文献。
- **非英文**：未检索。
- **知识截止**：子代理报告其截止为 2026-05，今天 2026-10-23；它列出但未读的 2026 预印本包括 2602.10097（逐步分解归因）、2607.18984（课程学习分类）、以及一簇 grokking 时刻控制（2606.13753 / 2607.11666 / 2607.04333）。**grokking 时刻簇需要单独一轮**：那些工作控制相变何时发生，用的是权重范数与表示几何而非数据位置，若其中真有"只用与结构无关的杠杆"的措辞，则数据位置这条杠杆可能正被它们点名为空缺——对我们有利，但**必须逐字核**。
- **待核的近邻**：arXiv:2603.27465（检索摘要称按固定训练步投毒、有 escalating 模式；若为真是对象层最近邻居）。本会话未核。
- **可靠性**：子代理报告检索摘要器三次给出错误否定（声称某处不存在、而正文逐字存在）。本文件 §1 的三篇由我本人取 PDF 抽文本核过；§2 的 [未核] 与未复核项不得作为判生死依据。
- **存活 ≠ 强度**：本文任何"未占"的结论只经过了上述两步半检索，必须这样写出来。

## 6 档位

实测（已核原文）：§1 全部引文。转述（子代理，待复核）：§2、§3 的文献归属。推导：§1.3 的口径不相容分析、§4 的后果。推测：§4 第 1 点的标题句能否站住，取决于 PL 的 K(d) 结果。
