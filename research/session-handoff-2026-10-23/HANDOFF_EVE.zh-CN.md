# 交接：2026-10-23 晚（19:00，服务器时间）

本文接 `HANDOFF_PM.zh-CN.md`（483cb56，下午版）。**现行口径以 `../jump-canon-2026-10-23/CANON.zh-CN.md` 为准，现行计划以 `../next-stage-plan-2026-10-23/PLAN_METHOD.zh-CN.md`（86de031）为准。** 本会话的提交：86de031（计划）、ff37e81（PL 放行 + J09 冻结排队 + S0 账面）、3f3cafd（LOG J68 删检查点）。

## 0 坏消息先说

1. **机制线连续六个候选全败，本会话又加一个**：G 线增益解释科学失败（R = 0.875，低谷恰在起飞窗）。正面机制模型为零，唯一活着的是 GA 的推测（清零后的优化器状态降低晚窗毒行效力，单次运行）。现在的位置不是"有现象没方法"，是**有现象、无方法、无机制**。
2. **S3b 七步查重没做完**，但第 6 步（对抗轮）已经打到一个真实威胁，见 §4。三个检索子代理在本会话结束时仍在跑，**我没有收到任何一个的结果**，下一个窗口要重新派。
3. 本会话有两处自报错误：J09 的 GJ0 第一次不过是我写的冗余校验索引错（LOG J65）；磁盘删除我第一次给了命令但没核实生效，第二次才真正执行（LOG J68）。

## 1 服务器现状（19:00）

| 项 | 状态 |
|---|---|
| PL（03400–03414） | GD1 **过**（15:20，逐位同 10-02）。R30a、R30b **训完**。L30（GPU1，115/125）、M30（GPU0，44/125）在跑。E30、Lend、Eend 与七条 LP 读数等卡 |
| J09（03700–03704） | 等卡，排 PL 之后。GJ0 已过（三项全真，`j09/gj0_report.json`） |
| M 线（03800–03811，另一会话） | 等卡，排 J09 之后。孪生与重放按文件依赖等 PL/J09 训练完成 |
| GPU | 两张都是我们的训练（各 85 GB）；视频任务 17:45 后让出 |
| 磁盘 | `/workspace` 余 **342 GB**（删 PS/SP 1375 个中间检查点后，139 → 348）。PL 剩余四臂 + J09 两臂峰值约 58 GB/条，够 |

**ETA**（双卡，每训练约 1.75 h、每 LP 约 26 min）：PL 七臂训练 10-24 约 00:30 完，七条 LP 约 03:00 完；J09 约 06:30 完；M 线随后。单卡退化则各加约 8 h。

## 2 PL / J09 怎么判

- **PL**：`copyback` 后本地 `python pl/eval_pl.py <目录> results_pl.json`。主档只用 L30、E30、R30a、R30b；带 ±5 / ±10（类比档，借自 1001 九次的 4.3）。四档见 `pl/CONTRACT_PL.md` §4。**不要自己从 lp_out 算判据**（`dont-rescore-from-raw-series`）。
- **J09**：`bash j09/copyback_j09.sh <目录>` 然后 `python j09/eval_j09.py <目录> results_j09.json`。J = 1009 三次训练的持续过零极差；三档只影响报告，**不得改 PL 的冻结带**。
- 任一闸门不过或缺臂 = 合同执行失败，零信息，不改判据重试。

## 3 PL 判完后的分叉（PLAN_METHOD §4，已写死）

| PL 主档 | 下一步 |
|---|---|
| 晚窗控制起飞 | S3 账本按三点 K(d) 插值 → S3b 查完 → 冻结 S4 调度器合同 |
| 两窗都动 | 账本退化为 lr 加权计数；S4 照跑，方法节弱化为"位移即可控" |
| 位置不敏感 | S4 不跑；pivot 到骨架密度判据（CONTRACT_DENS），**不降目标** |
| 零假设响 | 等 J09 的 J，按 CONTRACT_J09 §4 第三档处理 |
| 不可判 / 测不了 | 不加臂不改带 |

## 4 S3b 七步查重：已做的与结论（第 6 步，部分）

协议顺序 6 → 5 → 1/2/3/4 → 7（`related-work-protocol`：对抗轮前置）。**已完成的只有第 6 步的第一批检索加一次逐字核实。**

### 4.1 核实过的威胁：Souly 2510.07192 把位置效应归给学习率（对象层相邻，结论层未占）

用 `curl` + `pdftotext -layout` 自己抽的正文（不信摘要器），逐字：

- F.4：「While uniform spacing provides similar results, poisoning at the end is much less successful with the scheduler as seen in Fig. 21, compared to its performance with the constant learning rate in Fig. 18. **This is due to the small learning rate at the end of the fine-tuning.**」
- F.2：「20 samples are sufficient to poison at the beginning of fine-tuning but not at the end. … **We do not have a good explanation for this phenomena** … This result points to a **path dependence** in data poisoning, and **warrants further investigation in future work**.」
- 正文 §5 末：「in the random data ordering regime, backdoor poisoning attack success against fine-tuning is **also determined primarily by the absolute number of poisoned samples** injected」

**分层判定**：
- **对象层相邻、未占**：他们的因变量是**终点 ASR**，位置只有三档整块（beginning / end / uniform），没有"第几步起飞"这个量，也没有逐行位移的响应核。我们的因变量是起飞步数，位移单位是 1 步、10 条行。
- **结论层未占**：他们明写没有解释、称之为 path dependence、列为 future work。这对我们有利，可以引为"他们标为未解的那件事我们给了测量"。
- **真实威胁在这里**：F.4 给了一个现成的竞争解释——位置效应 = 到达时的学习率不同。若我们的 K(d) 与 lr 加权计数在 PL 七臂上分不开，审稿人会直接引这句话说"已知"。

**后果（要写进 S4 合同）**：
1. `THEORY_PREDICTIONS` §3 的 **lr 加权计数列就是这个竞争解释的零假设**，已冻结，位置正确，不用改。
2. S4 的 BLOCK-FRONT / BLOCK-BACK 两臂正好覆盖 Souly 的整块设定，**保留，并在论文里明确标注"这是 Souly 测过的那两档，我们复现它并与逐行调度对比"**。
3. 论文现象节与方法节都必须同时报 K(d) 与 lr 加权计数的差；差不出来就如实写"与学习率解释分不开"。

### 4.1b 第 1/2/4 步已交付（19:40 追加，LOG J71）

完整结果在 `../sched-audit-2026-10-23/AUDIT_SCHED.zh-CN.md`。三条要改论文框架的发现：

1. **Zucchet 2503.21676（DeepMind）结论层占了「调度数据改变学会的时刻」这个句子**（plateau 结束步当因变量 + 以数据调度建议收尾）。杠杆是频率不均衡非位置、目标是加速非打靶。**承重句必须改为 K(d) 不等于 delta(d)**：挪起飞前的毒行会动起飞、挪等量的别的行不会（PL 两条随机臂就是这个对照）。
2. **Pythia ICML'23 第 3.2 节是会被写进审稿意见的那句话**（我逐字核过）：「training order has little impact on memorization」「one cannot simply place sequences that are undesirable to memorize at the beginning or end of training」。口径差：他们的判据是每批被记忆序列的**密度**对泊松的 Q-Q 拟合（量的轴、整块位置），不是获取时刻。引言要正面处理。
3. **学习率归因现在有两篇**：Lesci ACL'24 最佳论文（我逐字核过）「memorisation ... is determined by data order and learning rate」且指出预热后更强。加上 Souly F.4，**lr 加权计数列是注册竞争解释，不是可选项**；K(d) 与它的逐臂差必须报。Lesci 的 tau_{g,c}（位置 x 检查点因果矩阵）是已发表对象里形状最接近 K(d) 表的东西。

另两件要做的：**「响应核」要改名**（影响函数文献的 SOURCE 已用 response function 且已考虑 position during training）；**工具归属表必须进稿件**（三条腿各自被占、乘积未被占）。Pythia 与 Lesci 方向相反但**摊开口径后不是载重矛盾**，不登记。

### 4.1c 第 3、5 步也已交付（20:10 追加，LOG J72/J73）。**我下午两处写错，已更正**

完整内容在 `../sched-audit-2026-10-23/AUDIT_SCHED.zh-CN.md` §4b、§4c。四条链裁决：对象层与结论层**无占位**；方法层很密。

**我的两处错（7.4）**：
1. **2606.29554 我只读了 seed-budget 段就概括为「把顺序当噪声」。** 同一篇还有 AdamW 冻结预条件下的**闭式位置核** Eq. (12)、§1 明写的 positional attribution factor、附录 A.6 的重排不等式极端排列，设定就是 LoRA SFT。已逐字核，`THEORY_PREDICTIONS` §7 已加更正行。**后果：核存在与近场单调不再是贡献**，承重移到「换算到步数」与「打到指定步」。
2. **「Souly 把位置效应归给学习率」是误引。** 他们的主微调设定是**常数学习率 5e−5**，位置效应在那里就出现且作者明写没有解释；F.4 那句只解释「另加的衰减调度臂里末段为何更差」。禁止写成他们的归因。另：原话是「20 个**毒样本**置于开头够」，不是「前 20 行」，旧文档措辞要改。

**最有用的一条（逃生口是量程不是形式）**：β₁ = 0.9 的半衰期 6.6 步，Eq. (12) 的一步差在 d = 50 为 5.2e−4、d = 280 为 1.5e−14。**PL 的 E30 臂（d ≈ 200–280）在它射程之外，L30（d ≈ 0–50）在射程之内。** 所以 E30 从"K(d) 的第三个点"升级为**与 Eq. (12) 的判别臂**：E30 若动，Eq. (12) 不可能是机制，而唯一时标相符的通道是该文**按假设冻结掉的第二矩 β₂（半衰期 692.8 步）**，恰在我们两 τ 不相容的 300–400 量级。→ **新组件层机制候选（推测）：β₂ 缓冲承载长程位置效应**；要只清 v 的臂才能与 m 分开（B3D-M0 只清 m，正好互补）。

**学习率解释回来是干净且反向的**：MPL 2503.12811 与 FSL 2509.19189 在固定日程下预测 K(d) ≡ 0 → **常数学习率下非零的 K(d) 是在证伪学习率解释**。因此**最有价值的新对照臂是常数学习率臂**（现所有 PL 臂在 cosine + 38 步预热下，与到达时 λ(t) 在设计上混淆）。PL 已冻结在跑，**不动**，这条进 S4。

**两篇新近邻**：`2607.27281`「ignites at a step of our choosing」——对象/结论层相邻但按定义不是占位（杠杆是数据混合浓度、滞后是实测常数、无 K(d)、全文 backdoor/poison 命中 0），**但它不在四条链里，还没过第 1/2/3/4/6 步，是下一窗口必须补的**；`2608.19762` 给了 AdamW 的时延 h 响应算子 G_{t,h} 与阈值穿越条件，是已发表对象里最接近 K(d) 的原语，**必须引并明写轴差**（他们测时延上的响应幅度，我们测位置位移引起的时刻移动），其被引链未扫，是最高价值的未扫面。

另：`2307.04138` 已发表「改单 epoch 内的数据顺序即可控群体精度」，所以**「我们首次把顺序当信号」这类措辞站不住**。

### 4.2 待核实（下一窗口第一件事）

- **arXiv:2603.27465**（DNA 基座模型投毒）：检索摘要称它"把毒样本放在固定训练步、有 uniform 与 escalating 两种模式，escalating 下毒量随训练进度平方增长"。若为真，这是**对象层最近的邻居**（有人已经按步调度投毒）。PDF 已下到本会话的临时目录（会随 job 删除），**需重新 curl 并 grep 核实，不要引用上面这句摘要**。
- 三个代理全部已回（4.1b、4.1c）。原列为未收到的第 5 步引用链挖掘（2606.29554 / 2510.07192 / Shumailov NeurIPS'21 / SAILS 2609.15029 的参考文献与被引）、第 1+2+4 步（课程学习 / 样本学习时刻 / 位置影响函数的模板层与同义词）、第 3 步蕴含检索（AdamW 记忆核、recency weighting、lr 归因是否已蕴含 K(d)）。**重新派，不要假定结果。**
- 第 7 步边界申报：未写。

### 4.3 一处待复核的内部矛盾（不要急着用）

Souly 的 F.2 正文说「20 samples are sufficient to poison at the beginning」，而 Fig. 18 与 Fig. 21 的图注说「poisoned data at the beginning is mostly ineffective」。两处口径疑似不一致（可能差在"是否接着做干净微调"或 1000 对 10000 总样本）。**引用前再读一遍该节全文**；按 HanLab 2.4，别拿别人的矛盾当武器之前先确认不是自己读错。

## 5 本会话做完的其它事

- **S0 账面同步**：CANON 现状句改 15:30；§5 加 T6–T10（B3 +18.4 不可判 / KICK 噪声否 / GA 描述 / G 增益否 / 机制无正面模型）；D3 改科学失败、D4 改关闭；§7 在途表重写并指向 PLAN_METHOD；§7 标题"以 PLAN_FINAL 为准"作废。
- **2606.29554 的 seed 预算已核原文**（§3 Eq. 18）：n\* ≈ 15.7(σ_ord/Δ)² 每侧，α = 0.05、功效 80%。我们顺序间起飞 σ_ord = 59.7 步 → 把顺序当噪声要检出 10 步差需每侧 560 次训练；固定顺序后残余 1.4 步、每侧 1 次，但 n = 1 估不出方差，**这就是 J09 存在的理由**。写进 `THEORY_PREDICTIONS` §7。注意两个 σ 不是同一个量（他们是 held-out NLL），只借框架不借数字。
- **CONTRACT_J09 冻结**（`jump/j09/`，预演四档可达 PASS）。

## 6 没做的、下一窗口按序做

1. 核 2603.27465（零 GPU，最高优先：可能是对象层占位）；再核 grokking 时刻控制簇 2606.13753 / 2607.11666 / 2607.04333，若它们点名数据位置为空缺则对我们有利，必须逐字核。
2. 重派第 5 步引用链与第 3 步蕴含两个检索代理，补完，写第 7 步边界申报，产出 `research/sched-audit-2026-10-2x/`。
3. PL 出数 → 判档 → **S3 账本**（零 GPU）：t̂ = t_ref + Σ K(dᵢ)·Δᵢ，在留出顺序 1007/1008 上对账，差额如实列（我的预测：差额 > 50%）。
4. J09 出数 → 判 J → 按 §4 处理 PL 的带。
5. S4 合同（要先过 2 与 3）。S5 第二模型族、S6 剂量边界在 S4 之后。

## 7 口径与纪律提醒

- 禁语照 CANON §3：不说 grokking / 相变 / 涌现；"判死 / 关闭"只给冻结规则下的科学失败。
- 本轮预注册预测累计 **2 对 5 错**，报告里要带这个数。
- 查重完成前本线不给等级（准则 5.1）。
- 存活 ≠ 强度：上面任何"未占"的结论都只经过了第 6 步一轮检索，必须这样写出来。
