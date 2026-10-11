# TSBD / CGD 保留意见展开（不重打分）

页码指本次81页 [TARE PDF](/Users/ruizhixu/Desktop/TARE_ICLR2027.pdf)，行号指PDF打印行号。以下区分已确认错误、证据边界和待澄清项，不改变原先锁定的推荐。这些问题影响文献定位、定量解释和复现追溯，不推翻已测得的WaNet调度干预。

## 1. TSBD已有clean-model实验：确认遗漏，但不据此否定全部新意

**原始证据：** 正式NeurIPS 2024论文 **p19 Appendix J、p20 Figure 11** 实际将ANP、RNP、TSBD应用于无后门模型，在CIFAR-10/Tiny ImageNet、PreActResNet18上报告clean ACC/ASR，并非只用clean模型提供参考权重。[正式PDF pp19–20](https://papers.nips.cc/paper_files/paper/2024/file/4a3ffc1a0074901718ab9dcef41909e9-Paper-Conference.pdf#page=19)。已查看这两页图像；不从柱形图推断精确小数。

**稿件位置：** p3行139–144先例段、p15 Table 6行790–808均遗漏TSBD。对应 [main.tex:169](/private/tmp/tare_iclr2027_panel/main.tex:169)、[Table 6源码:537](/private/tmp/tare_iclr2027_panel/main.tex:537)、[prior_work_audit.json:1](/private/tmp/tare_iclr2027_panel/supplement/audit/prior_work_audit.json:1)。这会遮蔽一个与本文重点批评对象直接相关的先例。

**边界：** 本文p2行60及p3已承认clean arm不是首次，不能指控其虚假宣称优先权。TSBD也不在Table 5的六项协议中，不能说表内存在错误的TSBD `no` 行。该先例自身不能证明已有same-recipe/schedule/seed匹配与scheduler crossover。最小修改是补入TSBD，不是删掉TARE全部贡献：

> Clean-model defense evaluations also appear in TSBD (Lin et al., 2024, Appendix J and Figure 11). Our contribution is the matched control and its use in the scheduler intervention and cost decomposition.

## 2. S54的1.84：来源标签错误与归因边界分开

**确认错误：** p79 S54行4254–4256将1.84标为benchmark自身fine-tuning的clean-twin收益；p8 §4.4行424–425引用其重参照翻转。实际上 [build_audit.py:16](/private/tmp/tare_iclr2027_panel/supplement/sched/analysis/build_audit.py:16)读取`schWmsPool`，注释明确为 **FT/FP/NAD pooled**。单独FT在p6 Table 2行294附近及p8 Table 4行393附近为−1.99。源码为 [appendix_sched.tex:603](/private/tmp/tare_iclr2027_panel/appendix_sched.tex:603)、[main.tex:392](/private/tmp/tare_iclr2027_panel/main.tex:392)。因此不能只称“FT值”。

**证据边界：** 即使改对标签，仍把WaNet上其他三种防御的收益同时扣在TSBD原表WaNet和Input-Aware两格，跨越了方法与攻击。本文没有可准入的Input-Aware twin（p6行316附近、p39行2067–2069、S46 p72）；TSBD自己的WaNet twin增益则为1.11点。另一个2.64点调整是防御前scheduler deficit，也不自动等于TSBD防御后的可扣除量。S53直接2×2只支持所测WaNet设置，不能补齐原八攻击反事实。

**算术并未算错：** 复算八攻击均值为91.69875/91.335；去掉两攻击后91.21167/91.46；`91.69875−2×1.84/8=91.23875`。[builder:6–18](/private/tmp/tare_iclr2027_panel/supplement/sched/analysis/build_audit.py:6)。删攻击是改变mixture的敏感性，重参照也是未识别的敏感性，均不应称为对原平均优势的因果校正。

最小修改：正文保留leave-two-out和直接2×2；若保留1.84，则注明：

> The 1.84-point adjustment is an illustrative sensitivity using the pooled WaNet clean-arm gain of FT, FP, and NAD, not a measured TSBD correction for either original cell.

## 3. 38%/66%：正确的均值比，不是已识别归因比例

位置：p2行75–77、p8行419–423、p79行4261–4264、p80 Table 58行4267–4274。[build_sched7.py:20](/private/tmp/tare_iclr2027_panel/supplement/sched/analysis/build_sched7.py:20)只对poisoned臂做A0≥0.90筛选，line21的twin无条件进入；lines31–32计算 **ratio of means**。用 [sched7_cells.json](/private/tmp/tare_iclr2027_panel/supplement/sched/data/sched7_cells.json)复算：

| shipped WaNet | twin gain，5 seed | victim gain，4个A0 seed | 稿件比值 | 两臂同4 seed的敏感性 |
|---|---:|---:|---:|---:|
| TSBD | 1.106 | 2.905 | 38.07%→38% | 1.0175/2.905=35.03% |
| CGD | 2.432 | 3.6875 | 65.95%→66% | 2.305/3.6875=62.51% |

shipped seed3的victim ASR约0.6414，A0排除其poisoned臂但保留twin；cosine全部5 seed通过。**原计算正确**，无条件tare本身有意义；共同支持集是另一估计目标的敏感性，不是修正真值。比值没有CI，且即便支持集一致，也不能识别移除带来的份额。Table 58仅写“A0-gated”、seeds=5却A3=4/4，需明确gate适用列。

最小英文修改：

> Twin gains average 1.11 and 2.43 points over five seeds, equal to 38% and 66% of the mean victim gains over four A0-passing seeds. These descriptive ratios have no uncertainty interval here and do not identify removal shares.

表注加：

> Poisoned-arm quantities are A0-gated (n=4 shipped; n=5 cosine); twins use all five seeds.

## 4. CGD原5%与本次10%：已披露差异，须约束下游结论

[CGD原论文](https://arxiv.org/html/2507.05113v1) §5.1.1/Table 1是5%设置，§5.2.1讨论相关增益；正式[ACM DOI](https://doi.org/10.1145/3746027.3754860)。本稿p78行4197–4200及 [SCHED7_PREREG.md:29](/private/tmp/tare_iclr2027_panel/supplement/sched/prereg/SCHED7_PREREG.md:29)坦承10%复核，不能批评为隐瞒。但p79行4259–4264从原结果转到本次2×2“给出数值”，容易把66%误读成原5%收益的已测分解。

本次支持的是10%设置存在非移除增益路径，不能定量解释原5%约3.4点收益，更不意味着实际使用者的总accuracy收益是假的。最小限定：

> Our 10%-poisoning WaNet experiment establishes a non-removal route to an accuracy gain in that setup; it does not estimate the decomposition of CGD’s published 5%-poisoning result.

若接受此范围，无需新实验；若坚持精确解释原5%结果，才需CGD自身的5%匹配实验。

## 5. CGD版本：收窄先前意见，不断言少执行阶段

**明确修正：** 正文p78行4198–4199的retained-subset fine-tuning概括过简，不等于实际缺unlearning/distillation。当前官方代码的三类损失均活动，现有证据不支持实现错误。

**base pin已存在：** [README:52](/private/tmp/tare_iclr2027_panel/supplement/sched/README.md:52)固定BackdoorBench `f02e353`并指向 [LOCAL_CHANGES:1](/private/tmp/tare_iclr2027_panel/supplement/sched/LOCAL_CHANGES.md:1)。不能因ZIP没cgd.py就称未固定版本。

**仍未闭合的插件版本：** [固定commit完整tree](https://api.github.com/repos/SCLBD/BackdoorBench/git/trees/f02e353?recursive=1)解析为`f02e3534645f0ee63d6848653062cd6c0d6c400d`，1019条、未截断，查无CGD/TSBD路径；LOCAL_CHANGES也未给插件revision。[SCHED7:26–32](/private/tmp/tare_iclr2027_panel/supplement/sched/prereg/SCHED7_PREREG.md:26)另引`binyxu/CGD`；[sched8.sh:6](/private/tmp/tare_iclr2027_panel/supplement/sched/scripts/sched8.sh:6)用外部`$WORK/CGD`，line19调用另加入的cgd.py/config。因此base pin本身不能固定外部插件；需要实际run的commit/hash/patch，并非必须vendor全仓库。

**描述核对的直接主来源：** 当前官方commit `98aa3be8fb2bd89e481dcea9923e8a5e719b1255`中，[cgd.py:167–185](https://github.com/binyxu/CGD/blob/98aa3be8fb2bd89e481dcea9923e8a5e719b1255/4backdoorbench/cgd.py#L167-L185)构造KL与unlearning项；[324–342](https://github.com/binyxu/CGD/blob/98aa3be8fb2bd89e481dcea9923e8a5e719b1255/4backdoorbench/cgd.py#L324-L342)实际训练clean CE与另一subset损失；[配置:30–36](https://github.com/binyxu/CGD/blob/98aa3be8fb2bd89e481dcea9923e8a5e719b1255/4backdoorbench/cgd/cifar10.yaml#L30-L36)权重非零。**当前commit不能证明作者实际运行的版本。**

最小修改，方括号须用真实记录填写：

> CGD constructs the data split and optimizes the released clean-subset cross-entropy, unlearning, and CLIP-distillation objectives. We used BackdoorBench f02e353 with CGD at [actual run commit], with [patch/no-code-change statement] and [source/config hashes].

## 核查范围

只读本次PDF/TeX、audit数据、生成脚本、README/LOCAL_CHANGES、注册和launcher、原论文及上述官方代码；只轻量重算均值/支持集。未跑GPU，未检查服务器checkout或权重，未改原稿，未读其他reviewer报告。最小修复主要是补先例、改标签、限制归因、补运行版本；不要求为了这些文字修复重新训练。
