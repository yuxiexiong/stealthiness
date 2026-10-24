# S3b 七步查重：起飞调度器（进行中）

对象：PLAN_METHOD S4 的方法主张——**"在固定毒集下，按位置调度毒行可把起飞步打到指定步（±10），不需要 oracle 微调"**，以及它依赖的测量 **K(d)**（位于起飞前 d 步的毒行每挪 1 步，起飞挪几步）。

协议：`~/.claude/CLAUDE.md` §4，顺序按 `related-work-protocol` 改为 **6 → 5 → 1/2/3/4 → 7**（对抗轮前置）。
**状态：七步已全部跑过至少一轮，但未完成。**（10-24 17:00 更新）第 1/2/4、3、5、6 步均已交付（第 6 步 10-24 单独整轮跑过，见 §4f）；F1（2607.27281）的第 1/2/3/4/6 步逐字核完，**第 3 步发现真占位**（§4d.1）；F2（2608.19762）续读发现我们的 β₂ 候选被部分占位（§4d.2）；F9（2603.27465）判对象层不相干（§4d.3）；第 5 步补扫捞出两篇必引（§4e），第 6 步又捞出一篇高威胁（§4f.1）与一句对我们有利的已发表普查（§4f.2）。**仍未完**：会场/OpenReview/在审稿件、非英文、2607.18984 的 90 篇 ACL 语料、连续学习里"任务顺序 → 该任务何时被学会"一支、ICLR'25 camera-ready（§5）。按准则 5.1，查重未完 → 本线不给等级。
**分层纪律**（`related-work-protocol`）：方法层被占**不构成淘汰**，只进"工具已存在于何处"表；对象层、结论层被占才致命。每条结论都标层。

## 0 坏消息先说

1. **"调度数据可以改变知识被学会的时刻"这句话已经被发表过**，而且因变量是步数：Zucchet 等（DeepMind，2503.21676）把 plateau 结束步当测量量，发现数据分布移动它，并以数据调度建议收尾。**结论层占了这个句子**，但杠杆是频率不均衡、不是消耗顺序中的位置，目标是加速、不是打到指定步，也没有响应核。**我们的框架不能写成这句话。**
1b. **【10-24 新增，比上一条更硬】"推迟投喂 → 行为被获取的时刻单调推后、且有截止时限"已被发表、有定理、有 n=5 与跨族复现**：2607.27281 §7（τ = 506/1108/1115 at S = 0/600/1200，tc ≈ 3000）。where × when 两条轴与 K(d) 相同。杠杆不同（开混合，总曝光随之变）、对象不同（合取电路）、无核；但"位置影响获取时刻"这句话我们**不能**当新发现写（§4d.1）。
1c. **【10-24 新增】"固定多重集、只控数据装载顺序"的攻击面已被命名并公开**：2610.00234 §7 "Ordering poisoning"，自认未开发。**禁写**"没人把固定剂量下的顺序当攻击面"（§4e.1）。
1d. **【10-24 新增】"固定语料只换顺序、因变量落在时间轴上"已有 300B token 的重量级实证**：2601.21698（Pythia 14M–1B）"we hold the architecture, optimizer configuration, and data source fixed and change only the ordering"，因变量是各 HMM 阶段停留的训练步数。不是具名行为的获取步、无 d、无核、作者自己的读法是收缩的（§4f.1）。
1e. **【10-24 新增，自报】我把"β₂ 缓冲承载长程位置效应"当成我们的推测档新候选是错的**：2608.19762 附录 E 已发表时标次序（参数最早、动量中程、二阶矩最长）。该候选降为类比档且必须引（§4d.2）。
1f. **【10-24 新增】常数学习率臂（CL）不是加强项而是必需项**：2606.29554 的 W_{r,K} 在 16 步窗内降 8 倍，2207.00099 实测位置效应与 LR 衰减交织。任何 K(d) 必须先排除"这是优化器时钟或 λ(t) 换了说法"（§4f.5）。
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

读法（**推导**）：这就是「把投喂推迟 d 步 → 行为被获取的时刻单调推后、且存在截止时限」，**两条轴（where × when）与 K(d) 完全相同**，且已发表、有 n=5 与跨族复现。这是真占位，但占的是**粗效应那一句**，不是我们的测量（见下文第 6 步的青蒿素读法）。因此：
- **禁写**「我们首次表明训练流中的位置会改变行为被获取的时刻」「首次把获取时刻当因变量来打靶」。这类句子会被这一篇单独打掉（F4 2307.04138 已打掉"首次把顺序当信号"，F1 再打掉"首次把获取时刻当因变量"）。
- 仍未被占的是**我们那一格的四个同时成立的约束**：(i) 剂量与流的构成完全固定，只换个别行的位置（F1 是开关整条混合，总曝光随之改变）；(ii) 逐 d 的核形状（F1 只有 S = 0/600/1200 三点加一个时限，无核）；(iii) 零分布是同幅度随机位移臂（F1 的对照是不开关）；(iv) 去除对照（把行推到学习率≈0 处）与位移对照并列。**承重句按 J71 的结论保持为 K(d) ≠ δ(d) 的形状测量，并额外加一句"在固定剂量与固定混合下"**。
- **新的竞争解释（此前未登记）**：F1 把税解释为**基底老化 / 可塑性丧失**（占位定理、§7「aging occupies heads」），不是优化器记忆也不是到达时学习率。我们 PL 的 L30 对 E30 不对称（+23.4 对 +5.96）目前登记的三个候选是到达时学习率、β₂ 缓冲、LoRA 二阶，**老化/占位是第四个，且是已发表并有定理的那个**。它对我们的具体预测需要单列（我们的窗口只有 1250 步、晚窗在 284–325，是否够"老"未知）。这条写进阶段 C 的候选表。
- 对我们有利的一条：F1 §6 与第 580–585 行把学习率明确排除在"混合因子"之外——"Learning rate is not even 'partially' [qualifying]"、"learning rate simply accelerates ignition … at fixed mix, lr and batch are clocks"。这与 CL 的承重方向一致（lr 是时钟不是位置核），但**它不给 K(d)，不能当 CL 的替代证据**。

**第 4 步（同义术语）**：F1 的词汇是 ignition / nucleation / drive / supersaturation floor / occupation tax / precursor clock。我们的词表（take-off、onset、acquisition step、position kernel、起飞步）**一个都命中不了它**；它是靠引用链捞到的。词表补这六个词，下一轮按它们重搜。

**第 6 步（对抗式）**：最可能被引来杀我们的那句话是"推迟投喂改变点火时刻、且有截止时限，已经做过了"。

**按青蒿素原则读这件事（用户 10-24 定调，`no-novelty-purism`、`purism-kills-audit`）**：青蒿入药上千年，把青蒿素提纯出来仍是诺奖级工作。F1 的结果**不是对我们的否证，而是现象普遍性的独立先例**，我们**引它当支持**：在另一个对象（合取电路）、另一个杠杆（开混合）下也看到位置与时刻耦合，说明我们测的不是 LLaVA 的一个怪癖。评分主轴是**谁会因此改做法**，不是谁先说的。

- **我们提纯出来的是什么**：F1 的税是三点加一个时限（S = 0/600/1200，tc≈3000），杠杆是开关整条混合因而**总曝光随之改变**，解释归给基底老化。我们在**剂量、流的构成、日程全部固定**的条件下，只搬个别行的位置，测出**逐 d 的核形状**，并配同幅度随机位移臂作零分布、配去除对照定上界。粗效应 → 可标定的核，这正是提纯。
- **禁写的只有一句**：「我们首次表明训练流中的位置会改变行为被获取的时刻」。改写为「位置与获取时刻的耦合已在合取电路上被观察到（F1）与在优化器记忆上被形式化（2606.29554）；我们给出的是固定剂量与固定日程下的**逐位移核**，以及它对攻击设定里起飞时刻的定量预测」。
- 不得因此缩小课题或软化主张（准则 1.1）。F1 的 §12「Honest scope」自述 ≤1.4B、两族、continued-training 的 token flux 远低于原生预训练、switch/shelf/reset 臂 n = 5 其余 n = 2–3——这写进稿件是**制度差异的事实**，不是我们的辩护词。

### 4d.1b 三列表（`no-novelty-purism` 的固定动作：每一行"已覆盖"都要进这张表）

| 前人到哪 | 我们推到哪（已有数据） | 谁因此改做法 |
|---|---|---|
| F1 2607.27281 §7：开混合的时刻推迟 → 点火推后，三点 + 时限（τ 506/1108/1115，tc≈3000），总曝光随之改变，解释归基底老化 | 剂量、流构成、日程全固定，只平移 10 条毒行：L30 +23.4（斜率 0.78）、随机臂 +4.8/+0.7、推到末尾 +45~49、去除对照无区别（PL 实测，CANON P16） | 微调服务商与数据策展方：同一批数据、同样条数，**排在哪**决定后门何时装上，因此"按条审核 + 看总量"不足，需要看顺序 |
| F2 2608.19762：AdamW 的时延响应算子 G_{t,h} 与阈值穿越；时标次序（参数最早、动量中程、二阶矩最长） | 同一时标次序被我们用来**预测事件时刻的位移量**，而非损失偏差幅度；β₂ 半衰期 693 步落在我们两 τ 不相容的 300–400 量级 | 做训练期检测/防御的人：报警与回退的有效窗口由哪条通道承载决定，二阶矩通道意味着窗口比动量通道长一个量级 |
| 2606.29554 Eq.(12)：β₁ 闭式位置核，近场单调，单位是终点权重 | 换算到**步数**（没人做过），且 d≈200–280 在 Eq.(12) 射程之外仍有效应 → 指向被该文冻结的 β₂ 通道 | 想用 TracIn 类 lr-at-arrival 加权做归因的人：近视加权在远场失效 |
| Souly 2510.07192 §5（常数 lr 下已见位置效应，作者明写无解释）、§4（"a certain number of sequential gradient steps" 的假说，未提取起飞步） | 把他们留下的开口做成可测量：起飞步作因变量 + 逐位移核 + 标定零分布 | 复现他们实验的人：位置效应不是调度产物，不能靠改日程消除 |

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

## 4e 第 5 步补扫交付：F2 的被引面几乎是空的，但捞出两篇必引（10-24 16:30，LOG J86）

子代理按第 5 步扫 F2（2608.19762）与 2606.29554 的被引链，并补了 Wang 等 ICLR'25。**下面两条承重引文是我自己取 PDF 逐字核过的**（`pdftotext -layout`，含关键词计数）；其余标【转述】的未经我复核，按准则 4 不作判生死依据。

### 4e.1 【我核过】arXiv:2610.00234（Wenhui Chen，2026-09-23）§7 已把"顺序投毒"写在纸上

逐字（第 2161 行起，"Three byproducts, flagged not developed."）：

> **Ordering poisoning**: PURERAND and BLOCKED are the same multiset in different order and differ by 0.22 exact-match: an attacker controlling only data-loader order, touching no byte, decides which convention a model commits to; the existing arms are the demonstration's skeleton.

我自己的关键词计数：`ordering poisoning` 1、`data-loader order` 2、`multiset` 4、**`backdoor` 0**、`poison` 1（就是上面那个词）、**`step at which` 0、`take-off` 0**。

**判 ADJACENT，且是必引**。分层：**攻击面这句话已被占**——"固定多重集、只控数据装载顺序、一个字节不碰"的攻击者模型在 2026-09 已公开命名，并自认未开发（"the existing arms are the demonstration's skeleton"）。对象不同（被改变的是模型选哪个**书写约定**，不是后门）；因变量是终点 exact-match 的份额；杠杆是全局块周期 L（BLOCKED → L=54…L=1 → SHUF 阶梯），**没有任何事件时刻读数**。

按青蒿素原则（§4d.1 第 6 步同一条）：这是"青蒿入药"的第二个实例——**想法被说出来了，测量没有**。
- **禁写**：「没人把固定剂量下的顺序当成攻击面」「我们首次提出顺序投毒」。
- 正确写法：引 2610.00234 §7 承认攻击面已被命名，然后给我们有而它没有的东西——起飞时刻作因变量、逐位移核 K(d)、同幅度随机位移臂作零分布、去除对照定上界，以及后门对象上的定量预测。
- 不缩小课题（准则 1.1）。

### 4e.2 【我核过】arXiv:2608.20442（Qinyang Xu，2026-08）——全集里对象层最近的一篇

它在**固定剂量下移动承载行为的块的位置**，且读数是行为量。逐字（第 418 行起）：

> **Source-window ablation (Figure 4c).** Three additional source blocks reproduce the signed route topology at 77–89% of the reference amplitude. All 64 prompt-level contrasts are positive; these repeated probes are reported as descriptive aggregates (Appendix M.3).

图 4c 横轴（第 394–396 行）逐字为 `Source block (update range)`，刻度 `40–47  80–87  120–127(reference)  200–207`。我自己的计数：`amplitude` 10、`horizon` 35、`position` 31、**`step at which` 0、`take-off` 0、`onset` 0**，`backdoor` 2（都在相关工作）。

**判 ADJACENT（对象层最近，结论层不占）**：因变量是固定时延（H=8，扩到 16）上的对比**幅度**，不是任何过零步；四个点且作者自标"descriptive aggregates"，不是核；杠杆是整段源窗口的绝对位置，不是固定流里一个子集的位移。**必引**，并且它的标题（"Stored in Optimizer State, Valued by Later Training"）与我们阶段 C 的优化器记忆/β₂ 候选直接相邻，写阶段 C 时要一起引。

### 4e.3 【转述】arXiv:2412.09538 = Wang 等，ICLR 2025，*Capturing the Temporal Dependence of Training Data Influence*（即 OpenReview uHLgDEgiS5）

子代理取 v1 全文后报：**ADJACENT**。(a) 它确实测影响随"何时被看到"变化，§5.3 报三个制度（High-impact Warmup / Low-impact Basin / Gradual Ascent，后者是"越晚参与训练影响分越高"）；(b) 因变量始终是**终点**模型的损失，从不预测或移动任何行为被获取的步数（全文 `backdoor`、`poison` 零命中，事件时刻类措辞零命中）；(c) 没有"位移 → 事件时刻位移"的对象；位置只作为该样本已经占据的迭代下标，杠杆是**移除**（因而剂量变了，不是固定剂量）。§6 未来工作里有一句把"按影响动态调整数据的呈现顺序与频率"列为未做，目标是学习效率不是事件时刻。
**动作**：必引，写明轴差（where → how much，终点值）。**camera-ready 未见**（OpenReview 被机器人验证拦住，HTTP 403），v1 与 camera-ready 之间若补了顺序干预实验我们会漏，列入边界。

### 4e.4 【转述】arXiv:2605.18814（Deng 等 2026）——位置被它明确当成"对齐掉的干扰项"

子代理报 §2.2 的 config-level 误差一段：batch 组成、data curriculum、学习率日程都"按记录的训练日志重放即可直接对齐"，它真正攻击的误差预算是优化器错配、一阶泰勒截断、浮点/GPU 不确定性。**判 METHOD-LEVEL ONLY**：位置不是它的可控变量，t\* 从日志读出，什么都不移动。

### 4e.5 对我自己 §5 的两处更正

1. §5 原写"**最高价值的未扫面是 F2（2608.19762）自己的被引链**"。扫完了：**那个面几乎是空的**——Semantic Scholar 列 1 篇（2609.33169，且该引用只在 v1，v3 已删，内容是可微仿真 RL 的四旋翼跟踪，与我们无关），OpenAlex 列 0（arXiv→arXiv 低计）。所以它不再是最高价值未扫面，但"1 篇"是**下界不是计数**（见下）。
2. 我在 §4d.2 没有核 F2 的被引者构成就把该面写成"最高价值"，这是**按索引数字推断**而不是核过的判断，此处更正（7.4）。

**索引不完整已被证明**：子代理发现 arXiv:2608.24593 在自己的参考文献 [19] 里引了 2606.29554，却**不出现在两个索引的任何被引列表里**；它也**不引** 2608.19762（我原任务单里写它引了，错，此处更正）。因此本线所有"被引 N 篇"都只能当下界。

### 4e.6 空着的那一格

跨子代理打开到全文的 10 篇里，`step at which` / `time at which` / `iteration at which` / `take-off` / `onset step` / `acquisition step` / `time-to-` / `emergence step` / `first step where` 的命中为 **0**（唯一例外是 2607.16821 的"rotation onset"，一个表示型事件时刻，自变量不是位置）。连同 §4d 的 F1（有事件时刻但杠杆是开混合）与 2606.29554（有位置核但单位是终点权重）一起读：

**已被占**：位置是可控杠杆（2610.00234 命名攻击面、2307.04138 控群体精度、2603.25047 控学会轮次）；位置影响终点影响力（2412.09538）；位置影响行为幅度（2608.20442）；推迟投喂影响点火时刻（2607.27281，杠杆是开混合）；β₁ 位置核（2606.29554）。
**空着**：**固定剂量、固定流构成、固定日程下，逐位移的事件时刻核** —— 这是我们的格子。承重句照此写，不写任何"首次"。

## 4f 第 6 步（对抗式）整轮交付：又一篇高威胁，以及一句对我们有利的已发表普查（10-24 17:00，LOG J86）

第二个子代理按"什么能杀死这个想法"的意图单独跑了一轮（21 篇取全文）。**我自己又逐字核了其中两处承重引文**；其余标【转述】。

### 4f.1 【我核过】arXiv:2601.21698（Elgaar & Amiri，UMass Lowell，2026-01，v2 05）——第二强威胁，此前完全没进过我们的视野

*Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics*。Pythia 14M–1B，**300B tokens**，三条语言学课程（Age-of-Acquisition、词频、Verb Variation）对 Random。

逐字（§5.3，第 530 行起；双栏穿插了图注，已去掉穿插部分）：

> In our setting, we hold the architecture, optimizer configuration, and data source fixed and change only the ordering; the shared transition map across orderings indicates that curricula do not create new macro-phases under this five-state model. Instead, the orderings differ in **phase occupancy**: Frequency spends more time in phase P2, while AoA and VV spend more time in P3 before entering the final phase.

摘要（第 15 行起）：

> We find that training follows a shared sequence of latent phases, while **curricula mainly change time spent in each phase**.

图 5a 标题逐字：`Mean training steps spent in each HMM phase by ordering`。另有一条反向对照：降序 VV 丢掉升序课程的大部分准确率优势（"direction matters"）。

我自己的关键词计数：`ordering` 93、`data order` 11、`curriculum` 52、**`backdoor` 0、`poison` 0、`step at which` 0、`take-off` 0、`acquisition step` 0、`displacement` 0**。

**判 ADJACENT，高威胁，必引**。为什么威胁高：**杠杆与我们同族**（同一语料、同一架构与优化器，只换顺序），**因变量落在时间轴上**（各阶段停留的训练步数），而且是 300B token 的重量级实证，不是 toy。

**为什么我们这一格仍然空着**（三条，按强度）：
1. 因变量不是**任何具名行为被获取的步数**，而是对全局检查点轨迹拟合的 **HMM 阶段占用**——五状态宏观阶段，不是阈值化过零；"行为 B 在第几步装上"这个量在全文不存在。
2. 杠杆是**按语言学分数对整个语料重排**，处处改变局部构成，只比较 4 个离散排序；**没有 d**，无法分离"把一个子集搬 d 步"的响应。
3. 它自己的读法是收缩的（阶段序列不变、只是停留时间不同），因此**连"顺序改变某能力何时出现"都没有主张**。

**动作**：引言与相关工作必须正面引它，并写明上面三条轴差。**禁写**「固定语料只换顺序这件事没人做过」。

### 4f.2 【我核过】arXiv:2607.04333 §2 ——已发表的普查说"数据侧"这个旋钮不在清单上

*Structure-Specific Representational Priors Causally Control the Grokking Delay*（Gunner Levi Howe，2026-07）。它为了立自己的项，对 2026 年"控制 grokking 延迟"这一簇做了一次普查，逐字（§2，第 139 行起）：

> **Structure-agnostic control of the delay.** A rapidly growing 2026 literature shows the grokking delay is not fixed but controllable — so far, **exclusively through structure-agnostic levers**. On the optimization side, Grokfast amplifies the slow component of the gradient signal [4], and NeuralGrok learns a gradient transformation [5]. On the parameter side, Truong et al. [6] establish a causal delay law: clamping the total weight norm to a multiple ρ of the critical norm rescales the time-to-grok as …

以及引言（第 62 行起）：

> These interventions differ in **the knob they turn (optimizer, parameter norm, representation geometry)**, but they share a fundamental property: all are structure-agnostic.

我自己的计数：**`curriculum` 0、`data order` 0、`poison` 0、`backdoor` 0**，`ordering` 1（参考文献）。

**读法**：这是一篇已发表文献里对"事件时刻可控"这一簇的自我普查，**三个旋钮全在模型侧，数据侧的位置杠杆不在清单上**。这是我们能引的最干净的一句"这一格空着"的外部证据。**限定**：它讲的是 grokking 延迟不是后门获取，且是单作者预印本；按准则 5.3 这条只作**文献状态的描述**，不作我们结果的支持。

### 4f.3 【转述】其余候选与各自的轴（全部 ADJACENT 或以下，无一占位）

| 论文 | 杠杆（自变量） | 因变量 | 轴 | 判 |
|---|---|---|---|---|
| 2606.13753 权重范数 grokking 延迟律 | 权重范数夹持 ρ（**模型侧**） | time-to-grok，T ∝ e^{αρ}，R² > 0.99，α ≈ 7.5 跨四个模数 | when | ADJACENT；**K(d) 的结构同形物**，审稿人会问我们与它只差一个旋钮——必引并答 |
| 2607.11666 表示几何控制 | 隐表示谱正则（模型侧） | grokking 起始/加速（至 52×） | when | ADJACENT |
| 2503.21676 Zucchet 等（GDM/ETH） | 频率不均衡、两段 warm-up（**剂量**） | **Plateau end (steps)**（图 4 纵轴） | how much → when | ADJACENT，高威胁，已在 §4b 登记；本轮补：它自己留了开口"需要进一步研究数据重复在何种条件下能缩短平台期" |
| 2505.17863 稀疏注意力涌现 | 上下文内重复 B、跨样本重复概率 p（剂量） | 平台期长度，**闭式幂律** T ∝ 1.51 T^0.99 B^−0.99 d^0.49（R² 0.999） | how much → when | ADJACENT；"数据侧杠杆 + 定量事件时刻响应函数"的最近已发表对象 |
| 2605.10019 两个时钟（Kempner/Harvard） | 数据集规模 N、规则复杂度（剂量） | τ_rule、τ_mem **两个事件时刻** | how much → when | ADJACENT；"把获取时刻本身当因变量"的最干净先例，强组，审稿人会知道 |
| 2505.16959 扩散模型早停 | 训练集规模 P（剂量） | τ_mem 近线性随 P | how much → when | ADJACENT |
| 2110.02406 Word Acquisition（Chang & Bergen 2021） | 词项属性（观察性，不干预） | **age of acquisition = 拟合 sigmoid 与阈值的交点所对应的训练步** | when（构造先例） | ADJACENT；**对我们有用**：阈值化过零换算成获取步这个读数构造 2021 年就有，引它可挡"你们的读数是临时发明的" |
| 2207.00099 Jagielski 等（遗忘测量） | **canary 注入位置**（where） | 终点记忆/成员推断精度；衰减速度 | where → how much | ADJACENT + **方法警告**：实测"在学习率衰减之前注入的样本被遗忘得显著更慢"→ 位置效应与 LR 日程交织。与 Souly F.4 合起来：**常数 lr 臂（CL）不是加强项，是必需项** |
| 2607.18984 课程学习分类法 | —（90 篇 ACL 论文的分类） | 无因变量 | 仅词汇 | METHOD-ONLY；**给我们形式语言**：调度器 S:(d_t,t)↦π_t、π_t 是"stage t 的期望训练贡献"，我们的位移设定正是它的 **no-retention** 制度（每样本只在一个 stage 贡献）。K(d) 可以写成"在 no-retention 调度器里把一个子集的支撑质量从 stage t 移到 t+d" |
| 2602.10097 looped transformer 的 step-resolved 归因 | 循环**前向**迭代（第四条轴） | 影响力 | — | IRRELEVANT（"step"不是训练步） |
| 2602.14872 RLVR 隐式课程 | 难度谱构成（混合） | 学会顺序、平台期长度（理论） | how much → when | ADJACENT |
| 2608.14936 小模型侦察瓶颈顺序 | 配额分配 | **first-passage** 到达各技能地板的 token 数 | how much → when | METHOD-ONLY；它**最小化**事件时刻，不测对位移的响应。术语"first-passage skill training"可借 |
| 2607.04969 记忆引导的数据复用 | 复用调度 | 验证损失、下游分数 | how much | ADJACENT-weak |
| 2511.04760 临界阈值以下的 grokking | 从已 grok 教师蒸馏（模型侧）+ 数据稀缺 | 步数更少 | — | ADJACENT-weak |
| 2205.05055 数据分布性质驱动 ICL | burstiness、类数（数据侧） | ICL **是否/多强**涌现，不是何时 | how much | ADJECENT-weak |
| 2601.04283 位置课程（模数算术） | **输入串内的字符位置** | 鲁棒性/准确率 | — | IRRELEVANT（术语假朋友，标题会出现在我们的检索里） |

已核并清掉（无"数据侧位置 → 事件时刻"组合）：2504.16041（Muon，优化器）、2606.32000（表示）、2504.13292（模型）、2607.23967（权重衰减时钟）、2606.25010（初始化随机）、2606.12966 / 2607.06639 / 2602.16967（观察性前兆与指标审计）。

### 4f.4 后门/投毒文献的专门扫描：我们这一格在那边也空着

【转述】两轮定向扫描（`backdoor` × position/ordering/dynamics 58 命中；`poisoning` × position/early-late 260 命中；另加一轮网页检索问注入时机）**没有任何一篇把后门被装上的步数当作毒行位置的因变量**。这与 §4c（Souly 96 被引、Shumailov 125 被引的事件时刻正则命中为零）独立地指向同一结论。

### 4f.5 本轮对我们自己的两条硬要求（都不是文献问题，是设计问题）

1. **CL（常数 lr 臂）从"加强"确认为"必需"**：2606.29554 的 W_{r,K} 说明光是优化器时钟就给出位置依赖权重（16 步窗内降 8 倍），2207.00099 说明位置效应与 LR 衰减交织，Souly 的主设定是常数 lr 却仍见位置效应。**任何 K(d) 都必须先证明它不是 W_{r,K} 或 λ(t) 换了个说法**。CL 在跑，这条已在轨。
2. **老化/占位这个混淆必须单列**：2607.27281 把 τ(S) 的增长归给基底老化，并给了相关性证据（【转述】等待期间每个头的注意力更"定型"：peakedness +45%、entropy −20%，与税 r = 0.93）。我们固定剂量的位移设计**正是为了把位置本身与老化分开**，但分开需要一条读数——见 PLAN 阶段 C 候选表第四行（先算可辨识性，量程为零就不提臂）。

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
- **10-24 两轮子代理补扫后新增的未覆盖面**：
  - **索引滞后是最大的洞**。2608.19762（两个月新）在 Semantic Scholar 只有 1 篇被引、OpenAlex 为 0；而子代理证明两个索引都不完整——2608.24593 在自己的参考文献里引了 2606.29554，却**不出现在任何一个索引的被引列表**。所以本线所有"被引 N 篇"只能当**下界**。OpenAlex 全文检索也不索引参考文献里的 arXiv 编号串（按编号搜命中 0、按标题短语命中 2），只能做标题短语扫，会漏缩写与拼错。
  - **OpenReview 取不到**（HTTP 403 机器人验证）。Wang 等 ICLR'25 用的是 arXiv v1（2412.09538），**camera-ready 未见**：若 v1 到 camera-ready 之间按审稿意见补了顺序干预实验，我们会漏。值一次人工浏览器核。
  - **Semantic Scholar 多次 429**，第一个子代理的后门文献对抗扫只跑了两条查询（不是一轮扫描），不得当作对后门文献的覆盖；第二个子代理另跑了两轮定向扫描（§4f.4），两者独立同向，但都不是系统会场检索。
  - **第 5 步只做了部分**：21 篇取到全文的论文，没有系统扫它们各自的前后向引用链（只对 2503.21676 §3、2607.04333 §2、2505.17863 的参考文献做了链挖掘）。
  - **非 arXiv 会场完全未覆盖**：ACL Anthology、OpenReview、PMLR、IEEE/ACM 都没检索；课程学习的时间效应文献以 NLP 会场为主，2607.18984 自己的语料就是 90 篇 ACL 论文，其中可能有报了阶段时间效应而摘要里看不出的，**那张表没有打开**。
  - **一条明确未闭的缺口**：连续学习 / 任务顺序文献里，是否有人在固定总曝光下测"某个具名任务被学会的步数随它在任务序列中的位置变化"。第二个子代理自述这一支的检索偏混合与顺序口味，可能漏。
  - **2026 预印本的可靠性**：2607.27281、2606.29554、2607.04333、2606.13753、2607.11666 都是单作者或小作者预印本，未经评审。**查新用途上"他们这样主张"就够；当事实引用则不够**（准则 5.3）。
- **可靠性**：子代理报告检索摘要器三次给出错误否定（声称某处不存在、而正文逐字存在）。本文件 §1 的三篇由我本人取 PDF 抽文本核过；§2 的 [未核] 与未复核项不得作为判生死依据。
- **存活 ≠ 强度**：本文任何"未占"的结论只经过了上述两步半检索，必须这样写出来。

## 6 档位

实测（已核原文）：§1 全部引文。转述（子代理，待复核）：§2、§3 的文献归属。推导：§1.3 的口径不相容分析、§4 的后果。推测：§4 第 1 点的标题句能否站住，取决于 PL 的 K(d) 结果。
