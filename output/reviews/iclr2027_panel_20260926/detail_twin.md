# R1 保留意见展开：近零投毒对照能否称为 never-poisoned twin

本说明只展开原评审的对照问题，不重新评分。结论先行：**不是“作者没有做零投毒控制”，也不是“已证明主要结果由十张投毒图造成”。确切问题是，大部分主表使用含1–10张投毒图的近零投毒对照，零投毒验证只覆盖三个设置、三种防御；现有验证支持特定大效应的符号，却没有证明整个交付表可等价地解释为真正从未投毒模型的防御费用。**作者已在摘要、§3.1和§4.6披露这一点，因此应按术语和证据边界评价，不能推成隐瞒或实验造假。

下列页码是PDF实际页码，与印刷页码一致；行号是PDF左侧印刷行号。原文引语来自本次用户提供的稿件。

## 1. 逐处核对：哪里准确，哪里容易让读者多推一步

| 位置 | 短原文或数值 | 判定及具体误读 |
|---|---|---|
| p1，标题，行001–002 | “Weigh a Never-Poisoned Twin” | 标题可以表达理想协议，单独看不构成实验错误；但容易使读者预期主要证据实际使用零投毒模型。主表并非全部满足这一字面含义。 |
| p1，摘要，行021–022 | TSBD和CGD “gain” on “a never-poisoned model too” | 实际近零投毒控制上的增益有证据；S34的零投毒验证不含TSBD/CGD。因此，严格的“真正零投毒模型也增益”超出了这里直接验证的范围。不能由此反推TSBD/CGD的增益一定是那十张图造成的。 |
| p1，摘要，行026–028 | “never-poisoned twin … (on BackdoorBench, ≤ 10 poisoned images, admitted only below 5% attack success)” | 没有隐藏投毒量；但“never-poisoned”与“含投毒图”字面上不一致。通过ASR门槛证明的是按该测量规则可接受，不是从未暴露投毒，也不是防御响应与零投毒模型等价。 |
| p1，摘要，行029–031 | “seven of the eight defenses charge the twin … +0.13 … to +5.70” | 这些是主网格近零投毒对照的数值；不是七种防御在零投毒对照上的统一验证。NC还明确限定在检测器触发时。错误会发生在读者把此前的never-poisoned标签直接带到这组数字时。 |
| p3，§3.1，行153–155 | “each poisoned example replaced by its unmodified original” | 理想定义确实是零投毒反事实。紧接着的实现改用低投毒率；两者间多了一项近似，而不是同一个模型定义的严格实现。 |
| p3，§3.1，行156–160 | “ten images; one … where ten install”; “‘never-poisoned’ means this”; “moves the tare by at most 0.35 points” | 对近零实现的披露准确。“at most 0.35”是九个已测设置的**均值差点估计的最大绝对值**，不是总体偏差的统计上界，更不是八种防御、七种攻击共同的误差界。缺少“observed mean”容易读成后者。 |
| p4，§3.3，行208–211 | “clean by the metric”; 近零对照ASR为0.6–4.5%；十图Blended为11–16%，一图通过 | 作者已承认这是指标上的干净，而非字面零投毒。Blended例子说明少量投毒确实可能影响ASR，但不能据此证明所有通过门槛的对照都有足以影响tare的污染。合理结论是仍需针对防御响应验证近似。 |
| p9，§4.6，行448–453 | “all eight BadNets clean-arm intervals exclude zero”; FT “+0.13”; zero-image FT “+0.07 [−0.05,+0.20]” | 这里已经正确给出零投毒FT的不确定性，不能指控作者省略了反例。前半句若明确写“near-zero-poison control-arm intervals”就不易被理解为真正干净模型均有确定正费用。零投毒FT区间含零，并不证明FT没有费用，也不证明+0.13由近零投毒造成。 |
| p9，结论，行480–484 | “cost on a never-poisoned twin”; “whose loss … is +0.13 to +5.70” | WaNet的FT/FP/NAD在真正零投毒两种调度器上确实得到符号翻转；但把整个+0.13至+5.70范围无条件归给never-poisoned模型仍过宽。结论应区分“所测近零对照数值”与“零投毒子实验确认的符号”。 |
| p56，S34，行2990–3004 | three cells, five seeds each; FT, FP and NAD；P69/P70要求均值小且区间含零 | 真正零投毒控制已完成，且通过守卫绕过和绘图兼容修改实现，不应说技术上无法做零投毒。P69/P70成功是按预先写明规则成功；但该规则不是统计等价检验。 |
| p56，表39，行3006–3017 | 每设置5种子；zero−ten的95%配对t区间 | 表格本身给出了足以约束解读的证据；主要问题在摘要式概括比这张表更强。 |

## 2. 表39实际支持什么

以下均为准确率百分点。每个设置有5个安装种子，同一设置中的FT/FP/NAD共用这些安装；不能把九行当作九个独立的大样本验证。

| 设置、防御 | 真正零投毒tare，95%区间 | zero−ten差，95%区间 | 可以作出的结论 |
|---|---:|---:|---|
| BadNets/cosine，FT | +0.07 [−0.05,+0.20] | −0.04 [−0.21,+0.13] | 平均点估计仍为正，但零投毒正费用未由该95%区间确定。它与零、与小幅正费用都相容；不能称为“FT损失消失”。 |
| BadNets/cosine，FP | +1.20 [+0.85,+1.54] | +0.09 [−0.51,+0.68] | 真正零投毒模型上存在较明确正费用；但对“零图与十图误差很小”的界定仍不精确。 |
| BadNets/cosine，NAD | +0.38 [+0.12,+0.65] | +0.13 [−0.19,+0.45] | 真正零投毒下的正费用受到支持。 |
| WaNet/不退火，FT | −2.21 [−2.84,−1.57] | −0.20 [−0.79,+0.39] | 真正零投毒仍增益；较大的负号不是只能由十张投毒图解释。 |
| WaNet/不退火，FP | −1.89 [−2.55,−1.23] | −0.14 [−1.15,+0.86] | 真正零投毒仍增益；但差值区间相当宽，尤其不能据均值接近零宣称两种对照等价。 |
| WaNet/不退火，NAD | −1.93 [−2.45,−1.41] | −0.06 [−0.54,+0.42] | 真正零投毒仍增益。 |
| WaNet/cosine，FT | +0.51 [+0.30,+0.72] | −0.05 [−0.21,+0.11] | 与同攻击不退火零投毒结果共同支持调度器造成符号变化。 |
| WaNet/cosine，FP | +1.21 [+0.89,+1.52] | +0.35 [+0.03,+0.66] | 两种对照都显示正费用，但均值存在可检出的差异；“0.35”不是不确定性的上界。 |
| WaNet/cosine，NAD | +0.88 [+0.62,+1.15] | −0.15 [−0.32,+0.03] | 真正零投毒下正费用，与不退火设置的负费用形成符号对照。 |

同样重要的是范围：S34没有零投毒BadNets/不退火这一第四交叉单元，也没有零投毒ANP、NC、I-BAU、SAU、ABL、TSBD或CGD的防御验证。因此，不能说它已经以真正零投毒对照完整复现了全部2×2和整个八防御表；也不能因为没有全覆盖就否定已覆盖的结果。

## 3. “没有显著差异”为什么不等于“等价”

S34的P70规则是：至少8/9行的 `|mean difference| ≤ 0.5`，且95%区间包含0（p56，行3000–3001）。提交的 `supplement/sched/analysis/build_sched20.py` 行16–17、84、96也确实如此实现。它回答的是“均值看起来不大，差异尚未被检出”，不回答“真实差异足够小”。例如WaNet/不退火/FP的差值区间为[−1.15,+0.86]，仍容纳会影响精确费用解读的差异；误差越大，区间反而越容易包含0。

若要主张在容忍度δ内等价，应先根据实际用途定义δ，然后检验真实平均差是否落在[−δ,+δ]之内。标准5%水平的双单侧等价检验可用配对差的90%区间完全落在该区间内来判断；要求95%区间包含于等价区间是更保守的充分条件。不能把“95%区间包含零”替代为“区间被小误差带包含”。

此外，即便证明“误差小于0.5点”，它也不足以保护+0.13点的微小正费用解释：近似误差容忍度比待解释效应还大。调度器约2点的符号变化和FT约0.1点的费用，需要的精度本就不同。此处不要求作者为了大效应证明不必要的极高精度，只要求强度相称。

按现有符号，记近零投毒对照为θ_ε，真正零投毒对照为θ_0，则近零对照给出的是

`c_clean(ε) = Q(θ_ε) − Q(Dθ_ε)`，

而理想干净费用是 `c_clean(0)`。两者差 `δ_D = c_clean(ε) − c_clean(0)` 会以相反符号进入DiD：`DiD(ε) − DiD(0) = −δ_D`。这些恒等式不要求δ_D为零；ASR通过A4也没有自动让δ_D等于零。因此，算术可完全正确，估计对象的文字解释仍需限定。

## 4. 不增加实验即可完成的最小文字修改

这些是建议，不是已改稿。

1. **标题：**可以保留为理想协议的倡议，但较稳妥的直接替换是 `TARE: Weigh a Matched Control Before Reading Backdoor-Defense Costs`。若保留never-poisoned，摘要应立即明确理想协议与主要实证近似。
2. **摘要与§3.1：**把实际实现统一称为 `near-zero-poison controls` 或 `low-poison controls passing the ASR admissibility gate`，把 `never-poisoned` 专用于真正零投毒模型。例如：
   > Ideally, TARE uses a never-poisoned twin. Our main BackdoorBench grids approximate this control with 1–10 poisoned examples and require ASR below 5%; a zero-poison sensitivity study covers FT, FP and NAD in three settings.
3. **§3.1“最多0.35”：**
   > Across the nine tested setting–defense pairs, the largest absolute difference in mean tare was 0.35 points. This sensitivity check does not establish equivalence for all defenses; Table 39 reports the paired uncertainty.
4. **摘要收费范围及结论：**
   > On the main BadNets grid, the seven non-collapsing defenses have positive mean tare on the near-zero-poison controls, with NC reported conditional on detection. The zero-poison FT estimate is +0.07 [−0.05,+0.20] points.
   若篇幅有限，后一条数值可留在§4.6，但摘要至少应带近零对照标签；+0.13至+5.70可作为该网格的估计均值范围保留。
5. **S34判定：**P69/P70按原规则“通过”的记录无需改成失败。补一句：
   > These registered criteria test descriptive agreement and failure to detect a difference, not statistical equivalence. The zero-poison results preserve the WaNet scheduler sign pattern, while small costs and untested defenses remain unresolved.
6. **TSBD/CGD一句：**把 `gain on a never-poisoned model too` 换为 `gain on an ASR-admissible near-zero-poison control too`。这保留实际完成的实验，不替作者预言真正零投毒会怎样。

## 5. 若坚持更强主张，最少要补什么

以下是条件性建议，不要求全部同时完成；没有开展任何新实验。

| 想保留的强主张 | 最小针对性证据，不必重训全部受害模型 |
|---|---|
| “真正未投毒的模型也会因该训练调度器而改变tare符号” | 对WaNet的FT/FP/NAD，现有零投毒结果已经支持这一点。若强调**整个2×2均用真正零投毒确认**，只需补缺失的BadNets/不退火零投毒单元及这三种防御，并与已有BadNets/cosine对应。 |
| “BadNets七种非崩溃防御都对真正零投毒模型收费” | 利用已有BadNets零投毒模型补跑尚未覆盖的ANP、NC、I-BAU、SAU；保留同一防御政策、种子和干净数据抽样。NC须同时报告触发率、含不触发为零的政策平均，以及条件于触发的费用，不能偷换三者。FT需按事先确定的精度要求增加零投毒安装重复；当前5种子尚不能确认微小正号。若还保留“第八种ABL摧毁真正干净模型”，再对ABL补同类验证。 |
| “零图与近零图tare等价，所以整个原网格可视为真正干净费用” | 在声称覆盖的设置和防御内做按种子配对的等价检验；预先定义与用途相称的δ，并说明哪些行通过、哪些未通过。对全体同时作强概括时，要明确逐行结论还是整体联合结论及相应误差控制。抽一个代表防御只能增强可信度，不能逻辑上证明所有防御等价。 |
| “TSBD/CGD在真正零投毒模型也得到增益” | 在已有的WaNet真正零投毒模型上运行相同发布实现。TSBD若仍需干净参考权重，应使用相应真正零投毒参考，或明说该参考仍为近零投毒，以免对照定义转移到另一输入。原受害模型不必重训。 |

即使补实验，结果也不一定恰好复现+0.13至+5.70；该范围首先是当前近零对照样本上的观测，不能先规定新结果必须吻合。最省工作且与现有证据吻合的处理，是改准文字并保留零投毒敏感性表。

## 6. 哪些核心结论不因此被推翻

- **调度器配置及其造成低起点的机制：**本意见没有发现配置读错或交叉实验数值算错。近零实现中的交叉结果仍是实际观测。
- **真正零投毒WaNet的符号变化：**FT/FP/NAD在不退火时均为明确负值，cosine时均为明确正值；不能把这一现象全部归因于十张投毒图。
- **干净模型可能被防御伤害：**零投毒BadNets的FP、NAD结果已提供直接例子；不需要证明每种防御都有正费用才能成立。
- **分解恒等式、起点差及不识别擦除价格：**用近零还是零对照都不改变代数恒等式。区别在对照量如何解释、两种解释差多少；不会使原来不可识别的“擦除份额”突然可识别。
- **榜单、DER裁剪和引用论文中的原始数值现象：**这条保留意见不否定那些观测。不过，“TSBD/CGD在真正未投毒模型也增益”的控制性表述需要按已验证范围改写。

## 核查依据与执行边界

本次重新核对PDF pp1、3–4、9、56的上述全部指定位置，以及S53源文中的TSBD参考设置；读了 `supplement/sched/prereg/SCHED20_PREREG.md`、`supplement/sched/analysis/build_sched20.py` 的判定逻辑、`supplement/sched/scripts/sched20.sh` 的运行范围和 `supplement/sched/LOCAL_CHANGES.md` 的零投毒修改记录。前次独立数值核验已用 `supplement/sched/data/sched20_cells.json` 重算表39九个零投毒tare，均与PDF一致；本次没有重新训练、没有运行提交者脚本、没有修改稿件，也没有读取其他评审。
