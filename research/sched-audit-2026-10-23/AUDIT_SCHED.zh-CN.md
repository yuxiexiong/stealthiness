# S3b 七步查重：起飞调度器（进行中）

对象：PLAN_METHOD S4 的方法主张——**"在固定毒集下，按位置调度毒行可把起飞步打到指定步（±10），不需要 oracle 微调"**，以及它依赖的测量 **K(d)**（位于起飞前 d 步的毒行每挪 1 步，起飞挪几步）。

协议：`~/.claude/CLAUDE.md` §4，顺序按 `related-work-protocol` 改为 **6 → 5 → 1/2/3/4 → 7**（对抗轮前置）。
**状态：未完成，但 F1 已补完。**（10-24 15:50 更新）第 1/2/4、3、5 步已交付；**F1（2607.27281）的第 1/2/3/4/6 步已由我逐字核全文补完，见 §4d.1——其中第 3 步发现真占位**；F2（2608.19762）续读发现我们的 β₂ 候选被部分占位（§4d.2）；F9（2603.27465）判对象层不相干（§4d.3）。**仍未完**：F2 的被引链、Wang 等 ICLR'25 的时间依赖影响、grokking 时刻簇、会场与在审稿件（已派两个检索代理，结果未到）。第 7 步边界见 §5。按准则 5.1，查重未完 → 本线不给等级。
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

### 1.4 学习率归因【本节已被 §4c.1 更正，保留原文作记录】

- Souly 2510.07192 F.4（本会话已核，LOG J69）：「This is due to the small learning rate at the end of the fine-tuning.」
- Lesci 2406.04327（上）：把顺序与**学习率**并列为决定因素，并明确记忆强度在**预热期之后**更强。

**更正（§4c.1）：上面这句说重了。** Souly 的主微调设定是常数学习率，他们并没有把位置效应归给学习率；那句 F.4 只解释"另加的衰减调度臂里末段为何更差"。下面三条后果仍成立，其依据改为 §4b.3 的 MPL 与 FSL：
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

## 4b 第 3 步（蕴含检索）已交付：最强对手是 2606.29554，而我今天下午读漏了它（19:55，LOG J72）

### 4b.0 先自报我的错误（准则 7.4）

今天下午我为 S0 读了 **2606.29554**（Sweeney, Sideplane.ai, 2026-06-28）的 §3，取了 seed 预算 Eq. (18) 写进 `../next-stage-plan-2026-10-23/THEORY_PREDICTIONS.md` §7，并把这篇论文概括为「把顺序当噪声」。**这个概括不完整，而漏掉的部分正是对我们威胁最大的部分**：同一篇论文还写下了 AdamW 在 LoRA 微调下的**闭式位置核**，并对它用重排不等式求出了极端排列。我当时只读了 HTML 的 seed-budget 段落就下了概括，没有通读。更正见下，`THEORY_PREDICTIONS` §7 已加更正行。

### 4b.1 我逐字核过的（curl + `pdftotext -layout`）

Eq. (12)：
> For first-moment decay β and current clock s, a gradient impulse at position r has endpoint weight W_{r,K}^{(s,β)} = (1 − β) Σ_{t=r}^{K} β^{t−r}/(1 − β^{s+t}).

附录 C：
> At the experimental configuration (K = 16, s = 100, β₁ = 0.9) the impulse weights **fall monotonically** from W₁ = 0.815 to W₁₆ = 0.100…

§1（两个用途，第二个就是位置归因）：
> The same impulse-weight kernel has two practical uses. It sets a local order-noise scale for sizing shuffle-seed comparisons, and **it gives a positional attribution factor: a block's influence is its gradient alignment times the finite-clock weight of the position where it appears.**

附录 A.6：
> …the first-order objective is Σ_r W_{r,K} φ_{w_r} with monotone weights W_{r,K}. **The rearrangement inequality gives both extrema**: pairing largest potentials with largest weights maximizes this sum, and pairing them with smallest weights minimizes it.

§3（关键的自限）：
> **This frozen-preconditioner step is the main modeling assumption** in the closed-form part of the paper…

设定：LoRA SFT，Pythia-1B / Llama-3.2-1B 等——**我们的优化器、我们的适配器**。

### 4b.2 分层与覆盖

| 我们的四件事 | 被 2606.29554 覆盖了吗 |
|---|---|
| (a) 位置核存在 | **完全覆盖**。而且独立蕴含至少五次（Adam 原文的 EMA、AdEMAMix 的权重剖面、Hara/SOURCE/Wang 的收缩乘积、Wong 的在线 GD 格林函数、本篇 Eq. 12）。**不能再当贡献写。** |
| (b) 单调形状 | **近场完全覆盖**。一步差是几何的 (1−β₁)β₁^d |
| (c) 量级 | **部分**：无拟合参数，但单位是终点权重，**不是事件时刻的步数**。没有任何工作把参数位移或影响分数换算成**起飞步数**，这个换算需要获取 margin 对时间的局部斜率——是真空地 |
| (d) 打到指定步 | **未覆盖**。A.6 只给两个**极端**排列；最优控制那支给**最短**时间；2605.24316 给批大小日程。本篇自限为诊断（Algorithm 1），并写「before treating a schedule as a design variable」 |

**真正的逃生口是量程，不是形式。** 这是本轮最有用的一条：

| d | Eq. (12) 的一步差 (1−β₁)β₁^d |
|---|---|
| 1 | 9.0e−2 |
| 20 | 1.2e−2 |
| 50 | 5.2e−4 |
| 100 | 2.7e−6 |
| 280 | 1.5e−14 |

β₁ = 0.9 的半衰期 **6.6 步**。所以 Eq. (12) 的核在 **d 超过约 50 步后数值上为零**。**PL 的 E30 臂（d ≈ 200–280）正好落在它的射程之外**，L30（d ≈ 0–50）落在射程之内。换言之：
- L30 若动、E30 若不动 → 与 Eq. (12) 相容，我们测到的是已被闭式写下的那段，贡献只剩 (c) 的换算与 (d)。
- **E30 若动 → Eq. (12) 不可能是机制**，而 AdamW 里唯一时标相符的通道是**第二矩 β₂（半衰期 692.8 步）**，恰恰是这篇论文**按假设冻结掉**的那一个（"frozen-preconditioner… main modeling assumption"），而且没有任何人把 β₂ 与样本位置连起来。

**这给出一个组件层的机制候选**（符合 `mech-localize-before-guessing` 的要求，不是又一个全局标量）：承载长程位置效应的是 **AdamW 的第二矩缓冲**，不是第一矩。它还与我们自己的两 τ 不相容对上了：margin 对单次到达的响应 τ≈40 对过零拟合要的 τ≈300–400（CANON P7），而 β₂ 半衰期 692.8 正在后者的量级。**档位：推测。** B3 的清零臂同时清了 m 与 v，分不开；要分开需要只清 v 的臂（B3D-M0 是只清 m，正好互补）。

### 4b.3 学习率竞争解释：回来是干净的，而且反向

这推翻了我 §1.4 的判断，必须更正（7.4）。两篇在**固定日程下预测 K(d) ≡ 0**：
- Multi-Power Law 2503.12811：损失只由学习率之和决定，唯一的距离索引项权重是 (η_{k−1} − η_k)，**常数学习率下恒为零**。
- Functional Scaling Laws 2509.19189（NeurIPS 2025，一遍 SGD，我们的机制）：forgetting kernel 只作用于**噪声项**，信号项没有到达时刻的卷积；获取事件是信号学习 → 预测 K(d) ≡ 0。
- TracIn 2002.08484 的 lr-at-arrival 加权是**近视的**（一步损失变化之和，无后续传播），本来就没有 d 依赖；作者自己的实验还把加权丢掉了。

**所以：常数学习率下非零的 K(d) 是在证伪学习率解释，不是被它解释。**

**最有价值的新对照臂（推荐，零争议）**：**常数学习率臂**。我们现在所有 PL 臂都在 cosine + 38 步预热下，K(d) 与"到达时的 λ(t) 不同"**在设计上混淆**。加一个常数 lr（或 WSD 稳定段）的 L30/E30 对，这个混淆就断了，而且按上面两篇，它是把竞争解释反过来用的唯一方式。**PL 已冻结在跑，不动**；这条进 S4 合同或一个独立小合同。

### 4b.4 其它要点（子代理转述，未复核）

- **2605.24316**（sketched 线性回归的动态小批）给了常数学习率下非零单调的位置依赖：「κ_t increases toward its end because later perturbations undergo less subsequent contraction」，并据此设计日程 B_t* ∝ √κ_t。**是 (d) 的最近类比**（批大小日程而非样本位置）。
- **2601.21996**（Mechanistic Data Attribution）确实用数据干预移动了涌现时刻（删除延迟、增强加速），但那是**成分控制不是位置控制**；其"影响样本在轨迹上均匀分布"不能当反证——他们的估计量是 Koh–Liang IF + EK-FAC，**按构造对置换不变**，看不见位置核。
- **2510.12071** 明写影响函数这一族「assume that data ordering has no effect on influence」——这是我们所处缺口的现成表述，可引。
- 没有任何工作把影响分数换算成步数的移动（定向检索为空）。
- **逐坐标幅度无关性的规范引用**：Balles & Hennig 2018（1705.07774）Eq. (7)–(9)；没人从它推出我们的结论。
- **Smith 等 2101.12176** Eq. (13) 是位置依赖以**恒等式**而非上界出现的最干净处，但其 O(η²) 一步差是单个 Hessian–梯度积、**在 d 上是平的**；单调衰减要靠重求和的 ∏(I − ηH)，在非凸 LoRA 微调里**不被解析保证**。

### 4b.5 对计划的后果（累加到 §4）

7. **(a) 与 (b) 不再是贡献**，论文里改为"已知结果在我们设定下的确认"，并引 Eq. (12)。承重移到 **(c) 换算到步数** 与 **(d) 打靶**。
8. **E30 臂的地位升级**：它从"K(d) 的第三个点"变成**与 Eq. (12) 的判别臂**。PL 判读时必须单独报 E30，并按 §4b.2 的两种情形写结论。
9. **加常数学习率对照臂**（§4b.3），进 S4 合同。
10. **新组件层机制候选（推测）**：β₂ 缓冲承载长程位置效应；要只清 v 的臂才能与 m 分开。写进机制候选表，**不立刻开跑**。
11. **第 5 步必须先在 2606.29554 上跑**（它的引用链还没挖；其参考文献里有 Krasheninnikov 2025「training-order recency is linearly encoded in trained activations」与 Sevetlidis & Pavlidis 2026「optimizer-state memory… collapsing when the optimizer state is reset」——后者与我们的 B3 清零臂直接相关，必须读）。
12. 待读：Wang 等「Fast catch-up, late switching: Optimal batch size scheduling via functional scaling laws」（ICLR 2026），子代理判为最可能藏着 (d) 式打靶构造的地方。

## 4c 第 5 步（引用链挖掘）已交付：两处更正我自己的前提，两篇新近邻（20:10，LOG J73）

四条链的裁决：**对象层与结论层都没有占位**——没人测过或控制过后门起飞的步数，没人测过单条毒行位置的响应核，没人把固定毒集调度到目标步。方法层占位很密（§3 与下面 F2）。扫描不完整，见 §5。

### 4c.1 更正一：Souly 没有把位置效应归给学习率（我逐字核过）

> We fine-tune with a batch size of 32 for one epoch, **with a constant learning rate (LR) of 5 × 10⁻⁵ unless otherwise stated.** We also experiment with three cases regarding the position of the poisoned data: (i) randomly shuffled, (ii) all poisoned data at the beginning, or (iii) all poisoned data at the end.（§5）

- 他们的**主**微调设定是**常数学习率**，位置效应在那里就已出现，而且作者明写没有解释（「We do not have a good explanation for this phenomena」、称 path dependence、列 future work）。
- 我下午引的 F.4 那句「This is due to the small learning rate at the end of the fine-tuning」只解释一件窄得多的事：在**另加的线性衰减调度臂**里，末段投毒为何比常数 lr 时更差。**不是对位置效应本身的归因。**
- 附录 C.3 他们还刻意移除学习率这个解释：「To isolate the effects of different learning rates across checkpoints from the effect of clean data set size on poisoning efficacy, we maintain a constant learning rate (LR) for all checkpoints.」

**修正后的判断，而且对我们有利**：已发表文献里存在**常数学习率下的位置效应、且作者无法解释**。稿件里若写「Souly 把位置效应归给学习率调度」，那是对最近邻的可追责误引，**不许写**。学习率竞争解释的真正地位由 §4b.3 给出（MPL 与 FSL 在固定日程下预测 K(d) ≡ 0），比我原来写的更强。

### 4c.2 更正二：「开头 20 条够」措辞错

原话是 **20 个毒样本**置于开头就够、置于结尾不够（「20 samples are sufficient to poison at the beginning of fine-tuning but not at the end」），**不是「前 20 行」**。我们自己的 `research/asr-paper-prep-2026-10-21` 与记忆写成「开头 20 条够、结尾不够」，会被读成位置区间。以后一律写「20 个毒样本置于开头」。

另记 Souly 一处**内部不一致**（已核）：正文说 20 个放开头够，Fig. 18 / Fig. 21 图注却说「poisoned data at the beginning is mostly ineffective」，符号相反；F.2 还把 Fig. 6b 错引为排序图。**所以它既不能当已确立的地面倚靠，也不能据此宣称被覆盖。**

### 4c.3 新近邻 F1：arXiv:2607.27281（2026-07）The Kinetics of Training——最接近我们那句话

摘要逐字（我核过）：

> Forward: a capability flat at baseline **ignites at a step of our choosing** once the mix passes a concentration floor (10/10 above, 0/12 below), and while still flat its arrival is datable from its precursor to 5% median error on six held-out models.

**分层：对象层相邻 + 结论层相邻，按定义不是对象层占位**（被获取的是合取电路不是后门；全文 backdoor 与 poison 命中均为 **0**，我核过）。不杀我们的理由：杠杆是**数据混合浓度**在选定步打开且须超过浓度下限，我们固定毒集只动位置；开关到点火的滞后是**实测常数**（约 700 步、OLMo-1B 约 247），不是标定的位移核，没有 K(d)。**它不在四条链里，因此还没过第 1/2/3/4/6 步。**

**动作**：宣称覆盖完成前必须把它过完剩余步骤。这是最可能让审稿人说「控制获取步数这句话已被写过」的一篇。

### 4c.4 新近邻 F2：arXiv:2608.19762——已发表对象里最接近 K(d) 的形式化原语

AdamW 写成输入-状态-输出系统，给出把单个小批扰动映到未来损失的**按时延 h 索引**的响应算子，并写下阈值穿越条件（两处已核到行号）：

> Define the horizon-h input–output operator G_{t,h} := c⊤_{t+h} Φ_{t+h,t+1} B_t.
> The perturbed trajectory crosses the threshold at horizon h exactly when δF_{t+h} > T_{t+h} − F_{t+h}(θ^c_{t+h}).

**分层：方法层。** 轴不同（准则 5.4）：他们测**固定步插入的扰动在时延 h 上的响应幅度**，我们测**单位位置位移引起的事件时刻移动**；他们解释损失尖峰，不测不预测不调度事件时刻。全文 backdoor / poison 命中 0。
**动作**：必须引并明写轴的差别，否则审稿人会说 K(d) 就是 G_{t,h} 改了名。它是 2606.29554 的被引，而它自己的被引**还没扫**（§5）。

### 4c.5 其它（子代理转述，未复核）

- **F4 arXiv:2307.04138**：「we show how one can control group-level accuracy (i.e., model fairness) … **by simply changing the data order for a single epoch**」→「单 epoch 内的顺序是可控杠杆而非噪声」**已被发表**，任何「我们首次把顺序当信号」的措辞都站不住。
- **F6 arXiv:2106.09993**（Accumulative Poisoning, NeurIPS'21）：累积相位让后面某一步的触发批造成大幅下降，是「排数据使事件在选定步发生」最近的已发表实例；但因变量是可用性下降的**幅度**，且作者专门有一节区分它与后门。
- **F7 arXiv:2603.25047**（The Order Is The Message，2606.29554 自己的参考文献）：固定排序策略把模数算术的学会提前到第 487/659 轮、IID 基线 5000 轮仍 0.30%，**把顺序当可控信号**；但杠杆是全局排序策略不是单条位置，无响应核，轮次是报告结果不是打中的目标。
- **F8 SAILS**：主结果用**全批梯度下降**，消耗顺序在其 oracle 里不存在（「We use full-batch gradient descent to make the oracle deterministic」）；其 SGD 消融对我们有利——「Full-batch training then yields 0% ASR for SAILS, whereas mini-batch SGD on the same training data reaches up to 71% ASR」。72 条参考文献里没有任何排序/位置/课程文献。
- **F9 arXiv:2603.27465**（Poisoning the Genome，即我待核的那篇）：剂量-响应阈值在**后载注入日程**下得到，作者自己把位置标为未控混杂，并说最小有效剂量应读作下界。与 Souly 的「前载更便宜」**符号相反**；子代理判为口径不同（看毒之后还剩多少训练）而非矛盾，**按纪律不登记为载重矛盾，先摊口径**。
- **F12**：Souly 主文 §4 自己留下的开口，正是我们要答的——「We hypothesise this is due to models requiring **a certain number of sequential gradient steps** on poisoned samples for the attack behaviour to be learned, but note this as an area for further investigation.」他们甚至已把 ASR 对训练步作图（附录 C.1），却从未提取起飞步、从未与位置关联。
- 定向正则扫描：Souly 96 篇被引与 Shumailov 125 篇被引，用事件时刻正则（step/epoch/iteration at which、predict when、control when、onset、take-off、emergence time、target step）**命中为零**。

## 4d 第 1/2/3/4/6 步补完 F1（2607.27281）与 F2（2608.19762）的续读（10-24 15:50，LOG J85）

我本人取 PDF 用 `pdftotext -layout` 抽文本逐字核过两篇全文（不经摘要器）。**坏消息先说：F1 里有一个已发表的结果，在实质上盖住了「推迟投毒会改变行为被学会的时刻」这句话**，见 4d.1 第 3 步。

### 4d.1 F1 = arXiv:2607.27281，Lei Dong，2026-07，*The Kinetics of Training: A Driven-Nucleation Rate Law for Emergence, Plasticity Loss, and Circuit Control in Language Models*

全文关键词计数（我自己 grep 的）：**backdoor 0、poison 0、"data order" 0、"training order" 0、curriculum 2（都指分阶段预建电路，不是顺序干预）**、position 69（全是序列位置，不是训练流位置）。所以**对象层不占位**：被获取的是合取电路（induction / IOI / greater-than），不是后门，顺序/位置不是它的杠杆。

**第 1 步（抽象阶梯，模板层）**：模板「对训练流做变换 T → 某能力被获取的步数移动」。F1 占了相邻的一格：T = **把数据混合浓度在实验者选定的步打开**（"the switch replaces the mix at an experimenter-chosen step"，§3）。浓度地板 10/10 过、0/12 不过。我们的 T = 固定剂量下把个别毒行平移 d 步。同模板、不同格。

**第 2 步（方法论单独搜）**：它的仪器是**前兆时钟** C(K) = e^{β(K−1)}（Prop. 2），用前兆的到达时刻给合取电路的到达时刻定日期；五个 Pythia 上拟合 C = 1.241 后冻结，六个留出模型 **6/6 落在预注册 ±25% 带内、中位误差 5.0%**，跨族 OLMo-1B 实测 1.244。与我们的仪器（阈值化 margin 持续过零 t_M0s + 用重复训练标定的抖动带）无关：它从**内部前兆**预测到达，不从数据位置预测。方法层不占位，但**它的预注册强度高于我们现在任何一条读数**，引用时不要显得我们在方法上更严。

**第 3 步（蕴含检索）——这里有真占位，必须正面处理**：§7 的"占位税与时限"（Fig. 8、第 901 行）逐字：

> the post-switch delay grows with waiting (n = 5 seeds: τ = 506 ± 44 → 1108 ± 66 → 1115 ± 65 at S = 0/600/1200); at S = 3600 no ignition in 4,400 [steps]

以及 §7 标题行与图注 "waiting pays the occupation tax; too late never ignites (tc ≈ 3000)"，跨族 OLMo-1B "control 0/2 never, switch 2/2 at ∼247, delayed switch taxed (τ 471/799)"。

读法（**推导**）：这就是「把投喂推迟 d 步 → 行为被获取的时刻单调推后、且存在截止时限」，**两条轴（where × when）与 K(d) 完全相同**，且已发表、有 n=5 与跨族复现。因此：
- **禁写**「我们首次表明训练流中的位置会改变行为被获取的时刻」「首次把获取时刻当因变量来打靶」。这类句子会被这一篇单独打掉（F4 2307.04138 已打掉"首次把顺序当信号"，F1 再打掉"首次把获取时刻当因变量"）。
- 仍未被占的是**我们那一格的四个同时成立的约束**：(i) 剂量与流的构成完全固定，只换个别行的位置（F1 是开关整条混合，总曝光随之改变）；(ii) 逐 d 的核形状（F1 只有 S = 0/600/1200 三点加一个时限，无核）；(iii) 零分布是同幅度随机位移臂（F1 的对照是不开关）；(iv) 去除对照（把行推到学习率≈0 处）与位移对照并列。**承重句按 J71 的结论保持为 K(d) ≠ δ(d) 的形状测量，并额外加一句"在固定剂量与固定混合下"**。
- **新的竞争解释（此前未登记）**：F1 把税解释为**基底老化 / 可塑性丧失**（占位定理、§7「aging occupies heads」），不是优化器记忆也不是到达时学习率。我们 PL 的 L30 对 E30 不对称（+23.4 对 +5.96）目前登记的三个候选是到达时学习率、β₂ 缓冲、LoRA 二阶，**老化/占位是第四个，且是已发表并有定理的那个**。它对我们的具体预测需要单列（我们的窗口只有 1250 步、晚窗在 284–325，是否够"老"未知）。这条写进阶段 C 的候选表。
- 对我们有利的一条：F1 §6 与第 580–585 行把学习率明确排除在"混合因子"之外——"Learning rate is not even 'partially' [qualifying]"、"learning rate simply accelerates ignition … at fixed mix, lr and batch are clocks"。这与 CL 的承重方向一致（lr 是时钟不是位置核），但**它不给 K(d)，不能当 CL 的替代证据**。

**第 4 步（同义术语）**：F1 的词汇是 ignition / nucleation / drive / supersaturation floor / occupation tax / precursor clock。我们的词表（take-off、onset、acquisition step、position kernel、起飞步）**一个都命中不了它**；它是靠引用链捞到的。词表补这六个词，下一轮按它们重搜。

**第 6 步（对抗式）**：最可能杀我们的那句话是"推迟投喂改变点火时刻、且有截止时限，已经做过了"。防线只有上面 (i)–(iv) 四条同时成立；若审稿人只要"位置影响时刻"这一句，我们**输**。另：F1 的 §12「Honest scope」自述 ≤1.4B、两族、continued-training 的 token flux 远低于原生预训练、switch/shelf/reset 臂 n = 5 而其余 n = 2–3，可在稿件里写"其制度与我们的 SFT 制度不同"，但这是辩护，不是占位的否定。

### 4d.2 F2 = arXiv:2608.19762 续读：我们的 β₂ 候选被**部分占位**（这是坏消息，也是独立支持）

J72 里我把"β₂ 第二矩缓冲承载长程位置效应"登记为**推测档、我们自己的候选**。核全文后要更正：该文附录 E 的 horizon-128 扩展**已经发表了时标次序**（第 1822–1824 行，逐字）：

> A horizon-128 extension confirms the ordering of time scales: parameter deviations act earliest, momentum dominates intermediate delays, and second-moment effects [dominate the longest delays]

（末句被双栏排版截断在"second-moment effects"后，次序陈述本身完整；引用时只引到可读部分，或取 arXiv 源文件复核。）同处还有："Increasing β1 from 0.5 to 0.99 moves the momentum-channel extremum from approximately horizon 5 toward the end of the measured interval"。

后果：
- **档位更正（准则 5.3、7.4）**：「β₂ 承载长程、β₁ 承载中程」从**推测**升为**类比**（相邻领域已有实测证据），并且**必须引 F2**，不得作为我们的新机制候选提出。我 10-23 在 AUDIT §4b.4 把它写成我们的候选，是**错的**；此处更正。
- 仍未被占：从时标次序到**事件时刻位移**的换算（F2 的因变量是未来损失偏差的幅度与极值时延，不是任何行为被获取的步数；全文 backdoor/poison 0、curriculum 0、"data order" 0）。轴差（准则 5.4）照 §4c.4 原样写。
- F2 的参考文献里有一篇此前未登记的近邻：**Wang, J. T., Song, Zou, Mittal, Jia, "Capturing the temporal dependence of training data influence", ICLR 2025**（OpenReview uHLgDEgiS5）——影响函数族里最接近"影响随位置变化"的一篇，**本会话未核**（OpenReview 要过浏览器验证），已派检索代理，结果未到。另 F2 引 Deng 等 2605.18814（轨迹型数据归因的误差来源）亦未核。
- F2 把 2606.29554 作为"Optimizer memory makes shuffle order a first-order source of fine-tuning noise"引用（第 684 行），确认 F2 是 2606.29554 的被引侧，§5 说的"F2 自己的被引链"仍是未扫面（已派代理）。

### 4d.3 F9（arXiv:2603.27465）已取全文，判为对象层不相干

此前列为"待核、若为真是对象层最近邻"。取全文核毕：**对象是 DNA 基因组基础模型**，剂量-响应阈值按累计投毒比例给出，位置被作者自己标为未控混杂。因变量是**剂量阈值与装上与否**，不是任何事件的时刻；无位置干预、无核。判 **IRRELEVANT（对象层不相干）**，从近邻表撤下；与 Souly 的"前载更便宜"方向相反一事按 `check-protocols-before-registering-a-contradiction` **不登记为载重矛盾**（口径不同：看毒之后还剩多少训练）。本文件不再转述其攻击细节。

## 5 边界申报（第 7 步，部分）

- **七步状态**（10-24 更新）：第 1/2/4、第 3、第 5 步已交付；**F1（2607.27281）的第 1/2/3/4/6 步已补完（§4d.1）**，F2 续读与 F9 判定见 §4d.2–4d.3。第 6 步（对抗式）对 F1 已单独跑过一轮，对整条线仍只做了第一批。第 7 步即本节。
- **我自己取 PDF 逐字核过的全文累计 6 篇**：Pythia 2304.01373、Lesci 2406.04327、Souly 2510.07192、Sweeney 2606.29554、Dong 2607.27281、Liu & Li 2608.19762（另 2603.27465 取全文后判不相干）。其余均为子代理转述或标题/摘要级，按准则 4 不计入“读过的论文”。
- **扫描深度与未扫的面**（子代理报告）：2606.29554 的 52 条参考与 6 篇被引全扫到全文级；Souly 56 参考 + 96 被引、Shumailov 37 参考 + 125 被引做了关键词与正则扫描；LeDoux 2603.25047 的链事后补扫完（1 篇被引即 2606.29554 本身，87 条参考全是方法层：无放回收敛理论、排序加速、课程学习、grokking，唯一的安全类引用是 Shumailov）。**SAILS 的被引链扫不了**（两个索引对这篇 2026-09 的论文都报 0 被引，属索引滞后不是干净）。**最高价值的未扫面是 F2（2608.19762）自己的被引链**，它持有最接近 K(d) 的原语。第二跳被引普遍未扫。
- 索引：Semantic Scholar 为主、OpenAlex 交叉核；OpenAlex 明显少计 arXiv→arXiv 引用（Souly 8 对 96）。无 Google Scholar。S2 全程 429 限流，引用数取自返回列表长度而非计数字段。Souly 7/96、Shumailov 18/125 的被引只有标题无摘要，按标题判定。
- **会场**：没有系统扫 ACL/EMNLP/NAACL/NeurIPS/ICML/ICLR 的 proceedings 或 OpenReview，**在审稿件未覆盖**；这个槽位占起来便宜，并行未公开工作是实际风险。
- **灰色文献**：子代理报了一篇只存在于机构仓库的硕士论文（UPF，Leybzon 2024，训练顺序与记忆持久性），PDF 未取；这说明训练顺序一族还有我没碰到的灰色文献。
- **非英文**：未检索。
- **知识截止**：子代理报告其截止为 2026-05，今天 2026-10-23；它列出但未读的 2026 预印本包括 2602.10097（逐步分解归因）、2607.18984（课程学习分类）、以及一簇 grokking 时刻控制（2606.13753 / 2607.11666 / 2607.04333）。**grokking 时刻簇需要单独一轮**：那些工作控制相变何时发生，用的是权重范数与表示几何而非数据位置，若其中真有"只用与结构无关的杠杆"的措辞，则数据位置这条杠杆可能正被它们点名为空缺——对我们有利，但**必须逐字核**。
- **待核的近邻**（10-24 更新）：2603.27465 已核并判对象层不相干（§4d.3）。现在待核的是 **Wang 等 ICLR'25「Capturing the temporal dependence of training data influence」**（影响函数族里最近的一篇，OpenReview 需浏览器验证，未取到）、**Deng 等 2605.18814**、**F2 2608.19762 的被引链**、**grokking 时刻控制簇**（2606.13753 / 2607.11666 / 2607.04333）。两个检索代理已派出（10-24 15:50），结果未到。
- **词表缺口（第 4 步）**：F1 是靠引用链捞到的，我们的词表（take-off / onset / acquisition step / position kernel / 起飞步）对它全部零命中。补入 ignition、nucleation、drive、supersaturation floor、occupation tax、precursor clock 六词并重搜——同一社区可能还有别的论文用这套词写了我们的句子。
- **可靠性**：子代理报告检索摘要器三次给出错误否定（声称某处不存在、而正文逐字存在）。本文件 §1 的三篇由我本人取 PDF 抽文本核过；§2 的 [未核] 与未复核项不得作为判生死依据。
- **存活 ≠ 强度**：本文任何"未占"的结论只经过了上述两步半检索，必须这样写出来。

## 6 档位

实测（已核原文）：§1 全部引文。转述（子代理，待复核）：§2、§3 的文献归属。推导：§1.3 的口径不相容分析、§4 的后果。推测：§4 第 1 点的标题句能否站住，取决于 PL 的 K(d) 结果。
