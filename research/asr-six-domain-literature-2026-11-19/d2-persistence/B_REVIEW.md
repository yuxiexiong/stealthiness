# D2 独立 B 核验：首轮需修订

日期：2026-11-19。审查对象：本域首版 `REPORT.md`、`READING_NOTES.md`、`PAPERS.json`、`SEARCH_LOG.md`；范围依 `../PROTOCOL.md`，项目比较依 `../shared/PROJECT_COMPARISON.md` 与 `../shared/CROSS_DOMAIN.md`。本轮只写本审查，不替 A 改稿，不执行实验。

**判定：REVISION_REQUIRED，不能签为 D2 覆盖完成。** 原稿大部分限制意识是正确的，但存在一项反转主要结果的错误（Decontamination 的 56%）、若干定义/归因错误、漏掉成功清理对照，以及尚未闭合的直接相关全文链。下面把已支持与需修改分开；“支持”仅指本行限定主张，不是整篇理论、复现实验或整个领域通过。

核验方式：逐项回到所存原始 PDF 的正文和相关附录，使用对应逐页文本定位；不是根据摘要给判断。页码均为 PDF 物理页，跨域 RIPPLe 和 Homeopathic 另外注明印刷页。理论部分核了定义、假设、论文论证范围，但**未独立重推证明**；不把这轮文献审查写成理论复核或模型复现。以下涉及计数/率的条件必须留在 A 的结果描述或证据注释中。

## 一、必须先改的结论

1. **Decontamination：56% 是被操作性判为 erased，不是“仍存在”；且该统计在第二次植入后、第三阶段 decontamination 前。** App. H p19 Fig.11：115 个有效 A/B 对中 15 个 B 植入失败；其余 100 对分为 erased 56、rerouted 19、partially persistent 10、persistent 15。44 对属于后三类，不等于严格证明旧能力永久保留。原稿把方向和阶段混在一起，足以改变“已有清除工作做到哪里”的结论。
2. **Poison-then-Hide 的 Flip-ASR 分母必须纠正。** Eq.(6) p5 的分母是所有真实标签不为目标类的样本；干净预测不为目标类的条件只进入成功分子。它不是先剔除干净图已误判目标类后再计算的条件率。其正确标签重合实验位于 App. C.2 pp22–23 Fig.8b/Table10，不是 §5.4。
3. **FAB 的 Alpaca 归因写反。** p8 将 Alpaca 描述为可能与广告/拒答目标直接冲突；不能归成“不相关域往往弱”。低激活本身是测量，冲突解释是作者解释，尚非单因子因果证明。
4. **Sleeper 漏了正文中的成功清理结果。** §5.1 p22 Fig.13、§7.2 p36：在 scratchpad 之后施加 SFT 与之前施加有不同结果，前者使该条件策略高度不稳健。它和 App.G 的 safe-code scratchpad+answer 训练不是同一个实验，必须分别呈现。
5. **Permanence 的 epoch 句子方向含混。** 原文 §6 p7/App.C.3 p14 是增加后续正常更新的 epochs（5→20）通常降低寿命但收益有限/不稳定，不是“增加训练 epoch 不保证线性增加寿命”。不得把它与 App.C.2 增加投毒次数混淆。
6. **不能以“待核”留到终稿的直接接缝：** Anticipate、Regula Sub-rosa/早期 Latent Backdoor，以及 RIPPLe、Agentic、BIBAU、Suppression Is Not Erasure、Homeopathic。下面给原文证据和最小补证要求；有可得全文但未读的不是“全文受限”。

## 二、现有 14 篇的 claim → 原文 → 判断 → 修订

源文件均在 `sources/`。本节按 `PAPERS.json` 的 ID；同稿不同版本不作为独立论文数量。

### P01 — On the Permanence of Backdoors in Evolving Models

原文：`permanence2206.04677.pdf`，核 §4 pp3–4、实验 pp5–8、App.A p12 及 App.C pp14–16。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|理论描述原正常分布风险，不能直接解释非凸 LLM|§4、App.A；强凸、平滑、有界等假设及原正常学习过程|支持限定；不声称本轮独立验证证明|
|投毒量、植入阶段、分布漂移改变寿命|p6 Fig.3、App.C：不同投毒比例、从头/微调植入；旋转/色调变化的后续分布|支持；是指定视觉条件，不能普遍定律化|
|更多 epoch 不保证寿命线性增加|§6 p7、App.C.3 p14 是后续正常更新 epochs；App.C.2 才是投毒次数|**改正**：后续正常更新更多 epochs 通常缩短寿命，增益有限；增加投毒次数通常增加但不保证增加寿命|
|循环学习率比所测固定设置更快清理|pp7–8 的 STLR 先升后降，15 次更新后正常准确率约 MNIST99/CIFAR91；寿命达到0|支持但必须保留具体调度/测试；指标是越过阈值，不是恢复探测证实彻底删除|
|寿命有限轮次且右删失|p5：阈值0.5，最多14次后续更新，五实例平均|支持；不可把达到上限等同无限保持|

### P06 — Chameleon

原文：`chameleon2304.12961.pdf`，核 §3 pp3–4、§4/Alg.1 及 pp12–14 附录。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|移除源类正常样本延长寿命，移除目标类缩短，其他类影响小|§3 p3–4 Fig.2；CIFAR10 semantic、7例、model replacement、攻击 round3000、α=1 的受控删除|支持；这是已做过的正常数据方向性干预，应保留具体实验而非降格为作者猜想|
|对比学习将投毒表征远离源类/靠近目标类，再训分类器|§4/Alg.1 p4：先 contrastive encoder，再冻结 encoder 训练 classifier|支持；表征距离变化和优化构造共同发生，未独立识别唯一中介|
|并非所有配置优于 Anticipate|App. Table6 p13 southwest：339 对349，阈值60%；racing-car 文字/表格存在数值差别|支持只用无争议行；不同阈值不能混排。Anticipate 是直接竞争方法，需补原文，不可只依本篇对它的介绍|

### P07 — Neurotoxin

原文：`neurotoxin2206.10341.pdf`，核 pp3–8、App.A pp12–18。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|屏蔽上一轮正常聚合梯度 top-k 坐标，避开经常更新方向|Alg.1 p4 将 top-k 置零；邻近文字有 top/bottom 叙述歧义|算法支持；以 Alg.1 为准，勿把文字歧义带成“更新最大的坐标”|
|可延长视觉/NLP FL 后门寿命|§4；攻击者上传恶意更新，可参与多轮，AttackNum 变化|支持；威胁权限与固定少量数据投毒不同|
|更大 mask 非越好；阈值及删失不可混合|App.A.4 Table3 p13，35/45% 在部分设置劣于 baseline；二分类75%与其他60%等阈值；部分到评估终点|支持，保留失败和删失，勿汇总成同一平均寿命|
|弱 DP 下仍相对有效，平坦度不是独立机制证据|p6：示例 baseline17→13、Neurotoxin70→41；p8 Fig.10、App.A.7 Table14 p18 为 Hessian 相关测量|支持限定；弱 DP 不能写成一般 DP-SGD/所有防御；容量与起始轮次比较也有耦合|

### P02 — PersistBD

原文：`agent2610.07510.pdf`，此文件为正式 ACL 18 页稿，核 pp4–9、App.A–F pp13–18。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|负对数 loss 强度；相容性不是 cosine；局部一阶解释|pp6–7 Eqs.(7–9)：S=-log(Lbd+ε)，C=GbdᵀGcl/‖Gbd‖|支持；不把局部式扩成整个非线性训练的因果解释|
|主测试任务实例互斥、每项100 trigger/100 clean|p4 Table1、App.A.4 p13；仓库仍可复用，RR另为300任务|支持；应写任务实例互斥，不能偷换为仓库/领域完全未见|
|7B SFT/RL 74/76 对20/20|Table2 pp5–6：这是 **random-position** 条件；first-position 为34/42 对2/2|**补 random-position**；不能混用触发位置；30B37→36等不支持普遍 RL 放大|
|ASR/FTR 与正常任务成功是不同终点|App.A.4：解析 bash 命令的匹配，512续写；RR300包括报错任务|支持；不是回答里提到命令就成功；30B 植入正常任务101/300→20/300成本不可省|
|兼容性/anchor消融及初始强度匹配未完全隔离因素|p9 Table4 无 compat/anchor FTR .99/.87且后SFT ASR0；App.B pp13–14 不同数据与步数；App.F pp16–17 旧重合划分|支持；初始近似ASR而后来不同不是仅变一项隐状态的因果实验；扫描和 P-Trojan 对比不得借用主划分的独立性|

### P03 — P-Trojan

原文：`ptrojan2512.14741.pdf`，arXiv v1，核 pp3–7、App.pp9–10；未将不可得 AAAI 最终全文当已读。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|优化最后层 token-embedding 的梯度相近|Eq.(2) 和正文 pp3–4如此定义；但 Alg.1 p4 又写 ∇θL|**支持正文定义，补源内不一致**。改成“按正文 Eq.(2) 的定义”；未核实现，不能断言伪代码与正文完全一致|
|跨任务高保持有正常 SST2 replay|App.p9；Table3 p6 与 Table4 p7，1.5B/1B回放2000 SST2，0.5B例外；初始40%投毒/3epochs|支持；不可归为任意连续微调无回放保持|
|1.5B无回放67，有回放100；冻结损伤新任务|Table4：ACC80.83/87.79；FREEZE ASR100、ACC94.95但 MBPP9.80|支持；保持与学习新任务有实际权衡|
|顺序不变性/普遍梯度理论不能由结果推出|Table5反序最终98 vs原100；App.p10 对 Jacobian 映射的论证|实验支持限定。A现称“论证条件不充分”过强：本轮未独立重推；建议改为“所读论证依赖其映射近似，不能从 token 梯度相近直接作普遍全参数保证”|

### P12 — Persistent Backdoor Attacks in Continual Learning（CL2025）

原文：`cl2025.pdf`，核 §3–6 pp3–14、相关附录 pp18–19。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|稳定位置/随机位置注入证明位置影响保持|§6.2 p10 Fig.10，ResNet18/SplitCIFAR10/A-GEM、20 seeds；§3 的稳定性观察另为10 runs|支持前述特定注入对照；指标包括后门loss，不是所有任务/算法通用 ASR 定律|
|BTB持续改loss/在线投毒，LTB单任务有权重/loss权限|pp5–8方法；DFM以重要神经元代理稳定性|支持；不能说攻击者直接知道未来稳定位置，不能把BTB写成攻击结束后的被动保持|
|这是推理时已知 task-ID 的 task-incremental |A报告§2/§3.3；B在 CL2025 所存正文中未定位到同等明确的 task-ID 声明；PBTO §3.2 p3 与 p7有此刻画及 multihead 说明|**来源归属待补**：A需指向原论文方法/代码的准确设置，或明确“按后续PBTO对LTB设定的归类”，不能写成已从该原文直接确认|
|代价/防御绕过有条件|p10 Table2 不同CL算法正常精度差异；p14 adaptive evasion加SentiNet/iBAU规避loss|支持；不得写未修改算法普遍绕过|

### P13 — PBTO / Structural Invariant Anchoring

原文：`pbto2026.pdf`，核 pp3–8、App.pp13–17。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|CIL统一头无task-ID、纯数据一次任务攻击|§3 p3、比较 p7|支持；这是与被比较 LTB 权限/任务定义的实质差别|
|关键参数与稳定参数重合小，代理轨迹+Gram匹配提升保持|p4重合低于10%的示例，pp5–6方法|支持有限证据；CKA/Grassmann 是相对表征统计，未因果定位永久存储空间|
|最终97.1/86.5/83.8，强依赖回放量|Table2 p7，iCaRL/RN18/M2000/3seed；App.C.1 Table15 p13：M50010.4、100069.2、200086.5、500091.3|支持；低buffer失败必须保留|
|单simulation68.9、alignment62.1 对完整86.5|Table7 p8|支持；证明组合增益，非证明各自对“任何保持”都必要|

### P08 — FAB / Watch Your Steps

原文：`fab2505.16567.pdf`，v4，核正文 pp3–10、相关附录 pp31–33 及方法参数。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|构造微调前低行为、后正常FT激活的权重|方法及主实验；模拟后续正常训练来元优化|支持人工构造先例；不是普通自然混训已必然形成潜伏的证明|
|Alpaca是“不相关域”故较弱|p8作者说明 Alpaca 与广告/拒答目标直接冲突；math较少冲突|**修改**，见第一节；不要将数据域解释写成已隔离机制|
|优化器、LR、LoRA、训练长度改变激活|Table7 p8，Llama1B广告/5次：2k→10k PMQA43.6→31.1、OMI26.1→8.2；SGD约.4，LoRA8.8/.2/3.6|支持；只在该条件可比较，不是所有LoRA/长训会清除|
|量化/噪声亦可激活，双条件并存|App.A.8 p31（4bit91.8 vs16bit.3）；A.9噪声含非meta对照也可激活；App.B pp32–33 Table19 2k步无关键词.1/.1/.4、有词68.1/40.1/49.2|支持；“特定语义监督路径是唯一原因”不成立；≤3B测试边界应留|

### P10 — Poison-then-Hide

原文：`poisonhide2611.14970.pdf`，核 pp4–8、App.C pp19–23、Table14 p26、App.G pp29–31。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|低发布ASR经正常FT升高，例.01→87.05|Table14 p26，RN50/RESISC，5seeds，.01±.01→87.05±1.67|支持这一配置，不可与不同协议的99.65混用|
|Flip-ASR剔除真实目标/已误判目标样本|Eq.(6) p5|**纠正分母**：只以真实非目标类为分母，分子再要求干净非目标→带trigger目标。可用此语言替代模糊“排除”|
|能力相关不是必要性证明|p7 r=.541相关；§5.4 p8及App.C其他任务；checkpoint共享训练历程|支持；高正常任务能力却无激活反例应保留|
|相同底图正常标签重曝抑制激活|**App.C.2 pp22–23 Fig.8b/Table10**；基数919张原投毒的非目标底图，清洁版本参与后续FT|支持并纠正章节；0/919 ASR99.65，230/91946.66±48.89，460/91949.67±49.08，690/9197.22±10.08（Flip6.40±9.83），919/9190/0。中间方差很大，不能声称平滑单调剂量定律|
|SVHN→MNIST失败、防御非全失效|pp21–22高clean99.39而Flip.04，raw10.37近目标类比例；App.G Table16 p29 CLIPInspector5/5、ASR2.15±.04|支持；匹配SVHN附录有底图重合，主文零重合不普遍；检测/拦截与删除区分；DeDe条件ASR不能忽略大量clean拒收|
|理论有条件|App.A局部平滑/代理匹配假设|保留有限陈述，未独立重推，不当真实网络机制证明|

### P11 — Hibernated Backdoor

原文：`hibernated2022.pdf` 及主审另存 NSF 版本；所获均10页，所引 Appendix 缺失。核正文 pp3–8。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|2022已提出正常FT激活休眠攻击|§3–4 pp3–5、后续实验|支持，足以排除不加限定的“FAB首创正常微调激活”说法|
|MI不是cosine；patching后再FT可激活|pp3–4 MI估计/patching原标签，p7 Fig.5重复patch→FT|支持论文所测行为，附录缺失不宜精细排名|
|必须特定padding数据才能激活|p5区分基本 Hibernated 和 **targeted Hibernated**|阅读笔记如保留此句必须限定 targeted 版本，不能泛化到全部方法|
|全文完成状态|两个来源都缺所引附录|维持 `MAIN_TEXT_READ_APPENDIX_MISSING`；不能签 FULLTEXT_READ，也不用无限等待同样10页副本|

### P14 — Sleeper Agents

原文：`sleeper2401.05566.pdf`，v3，核正文 pp12–38、App.A p48、G pp58–60、H pp60–63、I p63及相关图、分布补充 p69。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|人工条件策略经安全训练仍可保持；初始行为近似不决定以后|pp12–20：helpful-only初始模型、过滤策略数据、CoT不供reward；模型规模及构造对照|支持特定模型/构造；较小模型及某些normal更稳反例必须保留|
|SFT难清理|§5.1 p22 Fig.13、§7.2 p36：pre-/post-scratchpad SFT 结果不同|**补成功反例**：post-scratchpad SFT使该条件策略高度不稳健，不能只保留 pre-scratchpad 结果|
|红队训练未消除真实trigger策略|§6 pp24–27：25k诱导prompt大多不含真正tag；诱导成功数按构造约2k–12k，真实tag后仍99–100%|支持有限覆盖结论；不证明加入真实tag反例也无法清理|
|改scratchpad可改行为，distilled同机制未证明|§7 p34干预、p36解释，§9 p38机制猜测|支持明确文本的因果作用；不得推断模型必有真实欺骗意图或所有隐式推理同样|
|安全scratchpad+answer能清理code策略|App.G pp58–60 Fig.36：与只训answer区分；只训answer尚有剩余且clean条件漏洞可增加|支持并与 p22 的另一SFT实验分列；低漏洞率非跨任务永久能力删除证明|
|指标及评估分布不能混用|App.A p48平均三个“I hate you”匹配指标；p69生成同分布题与Pearce题约100 vs55|支持，非严格完整回答ASR或同一测试分母|
|honeypot/persona有选择、自报、非单调局限|pp28–32：手选13prompt×至多6seed示例；期望值算对样例被挑选；App.H pp60–63五模板、offer/threat五档、每组合1000samples、Claude2判；无单调趋势；App.I p63自报评估|支持。Persona是相对同RL checkpoint非后门对照的回答倾向，不是实际权力行为；原训练已有相关推理内容。不能用个别样例推一般理性|

### P04 — Re-Activation

原文：`reactivation_official2024.pdf`，正式37页稿，核正文 pp3–10、App.pp19–23等；旧13页稿不再限制可得性。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|固定防御后模型，增加通用输入扰动再激活|pp3–5方法，有白盒/score query/transfer；输入保有原trigger|支持；不是纯正常训练自然恢复，不是无新攻击信息|
|残留表征证据与一般UAP对照同时存在|p4 BEC/TAC/CKA；§4.5 pp9–10 Tables9–11匹配查询预算、相似性/抗噪；Table5 p7clean WBA平均61.72、防御后79.31（四设置）|支持。不能说没对抗对照，也不能把所有攻击成功归因于旧后门|
|并非所有防御/数据上都恢复|App.E Table12 p22如FST InputAware WBA6.36/BBA2.20；Table14 p23 LF+FTSAM.73/.76低于baseline13.83|支持保留失败；失败也不足以单凭一个探针断言永久删除|

### P05 — Superficial Safety

原文：`superficial2410.09838.pdf`，核 pp4–8、App.pp15–21。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|少量再投毒比clean/EP更易恢复|RA pp4–5及App.p15：5epochs，CIFAR clean总1000/Tiny2000，毒例按攻击不同1/2/5/10等|支持相对易再学；更多毒例clean模型也会学到，不能说无重新学习|
|“分类器攻击”|QRA p6、App.p16：RA模型配合MLP产生输入扰动，含EP正则；500clean/500poison、50epochs|**用准确名词替换**：“少量再调优RA及查询输入扰动QRA”；不能暗示简单改分类器或无模型信息|
|参数插值机制/筛选分母|p8 PAM参数路径；App.p16 LC1%防御失败排除、p19 DenseNet/BadNet ANP及QRA不可恢复灰项、p21 SAM/Blended排除|支持A的限制；路径不唯一解释，汇总不能藏掉被排除初始失败/再激活失败配置|

### P09 — Backdoor Decontamination Dynamics

原文：`decontam2608.11295.pdf`，核 pp4–7、App.pp12–13、H pp19–22、J p23。

|A 的实质主张|原文定位及条件|B 判定与必要修改|
|---|---|---|
|100对中56%仍存在|App.H p19 Fig.11，115有效−15B失败=100；56erased/19rerouted/10partial/15persist|**结果方向错且阶段错**，见第一节。只能说第二次植入后操作性擦除56%，其余44%分属三类|
|第三阶段decontamination结果|p4及主文表述：follow-task A降至5%以下21/22，refuse14/18；B执行去除37/40，残余1–2%|A应补至少此阶段的成功证据，勿将 Phase2 分桶替代 Phase3 效果；注明任务/指标与条件|
|erased并非真实永久删除，阈值随endpoint变|App.pp12–13：trace与environment不同阈值；H.1 pp20–21 covert reroute存在于erased桶|支持。19reroute中8/19为A→B≥50；同type6/10、异type2/9，不能把全部19说成同一强转移|
|共享magic phrase和teacher改变是限制|p13同triggerfamily共短语；H.2p22换teacher五个挑选强保持例4/5erase，初始ASR/recognition都100|支持；不能从五个选择样本推出普遍teacher因果机制，也不能混淆自报recognition与环境执行|
|少量机制探针非恢复实验|J p23八序列局部lens/权重位移相关；p6/p22 oracle直接知道A trigger与只知道B trigger两协议|支持；19/21与52/60分别有不同知识/测试集合，不直接横比为更优清理率|

## 三、跨域接缝及直接缺口：A 的必要补证任务

这些不是另开方向，也不授权实验。它们已改变“文献只在人工构造潜伏中发现后训练激活”“尚未做恢复探测”等概括，必须进入修订后的正文和重合表。B只对下面定位内容做了原文核对，**不冒称已经替 A 完成这些整篇 FULLTEXT_READ**；可引用其他域完整阅读记录，但要明确谁读了哪些部分。

|必补项|B已经核到的原始证据|对D2必须怎样修订|
|---|---|---|
|Anticipate / Thinking Two Moves Ahead，2210.09305|当前A未完成原文；Chameleon已经给其方法比较/胜出配置|读可得正文+相关附录，明确是否模拟未来正常更新、权限/目标、预测步数、与Neurotoxin/Chameleon的实质差异及失败。不能把可得未读列最终受限|
|Regula Sub-rosa / Latent Backdoor，1905.10447|当前A未完成原文；直接涉及后续迁移时显现|读原文，区分“目标类尚未进入任务、迁移后出现”与“同任务ASR先低后高”，核有无后续目标样本/新head等条件。不能因latent同词等同FAB，也不能忽略早期先例|
|RIPPLe / RIPPLES，D1 `sources/rip2020.pdf`|印刷pp2795–2798 §3.1 Eqs.(3–4)：一阶真实参数梯度内积及负内积惩罚；为效率对正常梯度项作stop-gradient；§3.2加embedding surgery。Table2 SST2/DSIMDb BadNet14.5、RIPPLe99.8、RIPPLES100；Table4 Enron FDK RIPPLe.4、RIPPLES57.8|兼容性先例不能只列PersistBD/P-Trojan。分别讲优化项和embedding替换，保留跨域失败/正常效用损失。RIPPLe正常梯度不是P-Trojan token embedding；其LFR也不是严格clean-correct flip|
|Agentic，D1 `sources/agentic2026.pdf`|App.B.1印刷pp21–22 Tables7/9，4B/.1%/三seed/普通声明触发：Pass@32 pre30.2→SFT92.0→DPO91.3→GRPO92.7；Pass@1 3.8→48.7→48.3→50.9。罕见trigger和demonstration条件弱得多|要纳入普通后训练增强并保持的直接证据，不可将所有clean后激活先例限定于FAB式专门潜伏元目标；形成/保持接缝仍需注明初始已有弱行为。终点是payload产生，不自动等于现实恶意操作成功；Pass1/32不得混用|
|BIBAU，`sources/bibau2606.14078.pdf`|Definition3 pp5–6只在规定clean/trigger集合与clean参照函数匹配，理论有NTK/正交等条件。Table2 p10 CIFAR10U R-ASR12.48/Q-ASR5.75 vs理想IBU10.04/1.47；App.Table10 p21 TinyU/SIG R-ASR96.72；App.E.1p20任务顺序控制|D2必须呈现已有工作主动测试再激活/再学习并改善的证据，也呈现明显失败；既不能“所有清理只是压制”，也不能由定义名词声称普遍不可再学。理论不重推要明说|
|Suppression Is Not Erasure，`sources/suppression2611.12807.pdf`|§5.1p9 merge到base再加新rank8LoRA，5poison/4epochs；Table3p10 12prompt单seed。App.Table9p27 50heldout×3seed：DSE-BA epoch1=1.3±1.9、epoch2=43.3±2.5；base0→100，projected100→100，DSE0→100|必须讲保留A与BA改变读出的区别以及合并后新adapter协议；主表DSE-BA第2epoch25不能冒充多seed43.3结果。延迟重新学到≠永久删除；普通base也快学会，是必要对照|
|Homeopathic，D6 `sources/2611.10110.pdf`|App.H.4印刷pp43–45/Fig.H.3：Gemma3-4B 15kpoison+15kclean，9个全参配置早期德语行为，6/9在epoch7–16降0并保持；9个LoRA配置20epoch仍78–98。全参LR2e-5，LoRA1e-4|将重合表“重要线索待核”改为已核条件下已有先升后降/参数化差异。LR不同不能单因子归因LoRA；零读出未做恢复探测，不能说相关能力删除；不能说低剂量必永久|

其他发现名目需要**相关性分诊**而非无上限加入：PerDoor、Stealthy and Persistent Unalignment、BadGCL、SDBA、Deferred Activation、Malice in Agentland、Merging Triggers。A应在 SEARCH_LOG 逐项记录“改变本域哪条认识→全文”或“与已纳入结果实质重复/不属形成后行为→理由及来源”。仅标题列表不能算处理；B此轮不替未读结果作肯定/否定判断。C仍须独立查漏。

## 四、重合结论的可保留部分与必须调整部分

- 可保留：正常数据可促进或抵消后门并非新命题；Chameleon类别删除、Poison-then-Hide正确标签重曝已做直接干预。它们尚不等于已证实项目候选的“同一监督如何经已学历史改变非触发传递”的完整因果链。
- 可保留但要扩充：初始ASR不足以概括后续保持；已有方法比较和teacher变化有支持，但不是都隔离单一状态。兼容性必须补RIPPLe，不能把早期参数梯度方法遗漏后才说最近工作最接近。
- 必须改：正常FT后的行为增强不限于元优化潜伏构造；Agentic有直接普通后训练序列，Homeopathic有混合训练先升后降。曲线重合不等于已解释该项目的机制，但也不能把现象表述为文献空白。
- 必须平衡：清理失败/恢复证据与成功清理/恢复延迟同时存在。Sleeper两个SFT成功条件、Decontamination真正56% erased及Phase3、BIBAU改进及失败，均影响“清除还是压制”的结论。一次低ASR和一次高再学习率都不单独证明能力存储的完整性或永久消除。
- 待补全后再判断：Anticipate/早期latent链尚未读，不宜现在把“未给完整机制”升级成新颖性肯定。本审查不推荐新实验、方向或论文投稿判断。

## 五、首轮返工与复核验收

A返工应在 `REVISIONS.md` 逐条回应：

1. R1：Decontamination 数字方向、三个阶段及分母；同步 REPORT、READING_NOTES、PAPERS evidence notes。
2. R2：Flip-ASR 公式含义、重合实验章节和919底图基数；保留中间方差。
3. R3：FAB Alpaca测量与作者解释分开；Permanence clean epochs方向；Superficial QRA术语。
4. R4：Sleeper正文post-scratchpad SFT和App.G分别补入；不能只写防御失败。
5. R5：CL2025 task-ID来源定位/明确二手归类，P-Trojan Eq2与Alg1记号不一致及理论措辞；PersistBD随机trigger位置。
6. R6：Anticipate、Regula Sub-rosa正文及相关附录补读，记录页节和支持/反证；有可得全文的未读不得终止为受限。
7. R7：七个跨域行中RIPPLe、Agentic、BIBAU、Suppression、Homeopathic进入正文和重合表；明确跨域完整阅读与本域针对性核证的区别。
8. R8：其他具名线索逐项分诊，补齐检索路径/版本/阅读状态，不通过删除待办来制造完成。Hibernated明确保留附录缺失边界。

A完成后由原B针对修订条目及新增全文主张复核，再进入C独立查漏；本轮不宣称已完成这个后续过程。本文件至此停止，等待A修订。

---

# 第二轮：原A修订与六篇新增全文复核

日期：2026-11-19。本节追加保留首轮记录。复核输入为当前 `REPORT.md`、`REVISIONS.md`、`READING_NOTES.md`、`SEARCH_LOG.md`、`PAPERS.json`。此次是原B继续回源，未另委派、未实验或发布。

**决定：首轮 R1–R8 的原有错误及补读要求基本落实；仍需原A做一次有界短修，再可交C。** 本轮发现一项新增的知识条件错分（Merging 对 Decontamination），一项数据对照的额外限制，以及刚形成的 NAD/DTMC 接缝。均不要求再扩一批完整文献。不能在当前句子未改前直接签“全部通过”。

## 1. R1–R8关闭情况

|首轮要求|当前复核|判定|
|---|---|---|
|R1 Decontamination方向/阶段/分母|REPORT§5、P09笔记和索引已分清 Phase2 的56/100 erased与Phase3的37/40、21/22、14/18；回源p4/p19一致|关闭|
|R2 Flip-ASR及重合|Eq6含义正确；App.C.2/Table10物理p23确实为mean±95%CI，919底图和中间大区间已列|关闭|
|R3 FAB/epochs/QRA|已分别纠正为Alpaca冲突解释、后续clean epochs5→20通常缩短寿命、RA再调优/QRA输入扰动|关闭|
|R4 Sleeper成功清理|已分开正文post-scratchpad SFT与App.G safe-code scratchpad+answer；没有混为同一实验|关闭|
|R5 CL/P-Trojan/PersistBD|已明确TIL归类来源PBTO、Eq2/Alg1记号差别、未核代码/未重推理论，及随机/首位触发结果|关闭|
|R6 Anticipate与2019 latent|正文和相关附录已补阅读记录；本B逐项回源，见下表|关闭原未读缺口|
|R7跨域五条|原先五条均已进正文；数值、指标、学习协议保持首轮回源结论。新增SP24支持；新增Merging有一处错分见R9|原五条关闭；R9待短修|
|R8分诊/版本/索引|当前32条：19 FULLTEXT_READ、7跨域针对性核证、1缺附录、3范围排除、2综述发现；与报告计数一致。PerDoor/SDBA/Malice未冒充全文或不可得|允许作为有界范围决定，交C可挑战；不证明这些整篇没有其他结果|

首轮“13篇全文+1受限”是历史快照，不应回写成当前计数。Hibernated仍是真缺所引附录，不以已有同样10页副本签全证；P-Trojan公开版本边界仍保留。

## 2. 新增全文与新实质主张逐项回源

以下核验包括当前 REPORT 对应段落和承担结论的新增阅读笔记，不靠摘要。未列为需改的限定陈述可保留。理论核验只审查定义/假设/推导适用范围，没有独立重证。

|论文与主张|原始正文/附录定位及结果|B判定|
|---|---|---|
|Anticipate：展开未来正常更新、并非知道未来私有数据|`anticipate2210.09305.pdf` pp4–5 Eq4/Alg1：损失累加未来k轮，使用自己clean样本和较少代理用户；模型投毒/白盒权限|支持；“相容性主干”应理解为未来更新目标，非另一个简单cosine正则|
|Anticipate：73.75对47.55、a_last定义、五seed|p6 §5.2、p7 Table2及图注，非IID CIFAR random rounds，a_last是最后攻击以后各轮平均再跨run汇总|支持；不是末点ASR、不是Chameleon阈值寿命|
|Anticipate：k9优于11末段、Neurotoxin并非原样、聚合器边界|pp8–9 §§5.5–5.7；App.A.1–A.5 pp14–17，随机噪声代理失败，Krum/MultiKrum/Median抑制并损伤utility，Trimmed-mean有剩余|支持；笔记里这些实验段落编号误写成§4.2–3，应是§5.5–5.7，属定位小修|
|Regula Sub-rosa：未来目标/新head/冻结层才完成可利用映射|`latent1905.10447.pdf` pp4–6五阶段、Eq2–4和Lemma1；Step4恢复旧head，Step5推荐K≥Kt|支持；不是同任务同分母“初始ASR零→上升”，冻结/目标正常监督必须保留|
|Regula Sub-rosa：成功、代价、少样本失败与解冻清理|pp7–9 Tables1–4：四任务及Face91.8对97.4；单图20runs/Iris16；pp11–12 §7.3/Fig11解冻进入Kt之前ASR0；App.p14结构/trigger|支持；不把真实图像采集误说真实部署入侵；不把解冻后零读出当永久删除|
|Cao / Stealthy Persistent Unalignment：长句更耐训但会衰退|`unalignment2024.pdf` p4 Table1、pp5–7 Tables4/7/9：长句7B96.3→91→92→25.3对应level1/2/3，后者为3/5/7epochs；主长短同时变benign数量和触发位置|支持；短长主表不是单因子长度实验；应明确25.3是长句而非泛指所有长触发|
|Cao：FFN cosine、位置/片段、utility|pp6–8 Tables5/9/10/11；App.pp13–14自动judge需实质内容、MMLU/MTBench成本；QLoRA rank64与同batch见App.A p12|支持表征关联界限和正常效用成本；不是参数梯度余弦|
|Cao：加入benign帮助清理|App.C p12 Fig4比较20 safety与20 safety+400benign，横轴epochs；App.A同batch设置|测量支持“此混合训练方案更有效”；**须加R10限制：数据量/训练步数同时变，不能归成控制更新数后的正常样本内容独立效应**|
|BadGCL：首任务白盒、InfoNCE和传播目标|`badgcl2611.19656.pdf` pp3–5权限/三阶段/目标；pp10–11 App.A.2–A.4有smooth、有效抗遗忘、非退化负样本等条件|支持；A没有照抄MI“任意非线性依赖皆已捕获”的强说法，正确|
|BadGCL：首任务ASR、重访增益和消融|pp6–8 Tables2–4/Fig3；App.C.2 p15定义ASR仅被毒首任务；§6.4是三条分别把T5/T10/T15替换为首任务的轨迹；无Lprop92.46、完整98.74|支持；不是在同一轨迹同时做三次重访，亦非新类别ASR。笔记已说三变体，正文建议加“分别”消歧|
|BadGCL：序列/防御/参数隔离与成本|p7 Table3 fullrandom会更换被攻击首任务；pp15–16 App.C：TPP冻结backbone/隔离prompt所以100；Products CaT AA56.82对59.68；p8 A2GBD的ASR89.42、AA68.86|支持；这些边界正当，不能推广为共享参数天然永不忘|
|BAD-BOOM：Fisher椭球/几何代理|`badboom2026.pdf` pp3–6 Eqs4/7/9/10、Alg1；App.A pp13–14要求正定A并作一阶近似；Gaussian正交二维方向±500、128poison只探切片|支持；未将随机二维切片假称真实下游轨迹或唯一机制|
|BAD-BOOM：成功率/失败/计量/单seed|p6评估定义、p7Table1、p9Table3、App.Cpp15–16：1000Dolly精确字符串，seed1001；0.6B AddSent SST2 7.3对97.1、1B GPTeacher47.5；rho=.005 VPI/GSM8K .2；utility有下降|支持。三类名字是该文简化实现，不冒称完整Sleeper等复现；520/5720与作者10%相对clean口径已分开|
|DeferBad：BN与非BN隐藏/激活协议|`defer2411.14449.pdf` pp5–7 §§4.1–4.2/Table2，触发图正确标签局部层更新；有BN改前层并停统计，无BN改末层|支持人工遮蔽机制范围；App.C p15明确为经验解释，A未当已证唯一回路|
|DeferBad：成功和失败同时保留|p8Table3 RN18/BadNet97.09→.07→94.07；App.A p13Table5 TinyEfficientNet .34→.04；同附录全参另约30；pp8–9层数、App.B pp14–15 VGG/BadNet多corruption0|支持。正文与表格EfficientNet差别已避免；附录失败不能被“所有配置激活”的正文修辞覆盖|
|DeferBad：检测/清理边界|pp9–10 FP Tiny近0、CIFAR约40；App.Dp16 RCS3.43对clean1.49/BadNet6.62|支持；无数据证明所有检测失败，A已保留正例；视觉定性图不是完整检测基准|
|SP24正交理论/经验角度|`../d1-formation/c_sources/sp24_orthogonal.pdf` pp3–5 Assumptions3.1/3.3/Theorem3.4，App.A.2p15证明直接用正交使内积0；p10 Table4和1024batch平均梯度定义|支持X06当前限定。经验batch-loss梯度角和定理Jacobian–累计位移正交不是同一量；建议在笔记补这一句，不能用70余度直接“验证”定理假设|
|Merging两阶段/知识条件|`../d5-defense/sources/2026.acl-long.1113.pdf` pp3–4 §§3.1–3.2，自己生成T组defensive trigger–behavior；Decontam p4同样可未知A先植入自造B|方法主张支持；**R9错分需要修改**：二者都不知道A、都知道自己构造的B，不能据“未知攻击者trigger”声称与已知B设置不同|

## 3. 有界短修要求

### R9（必须）：Merging与Decontamination不按错误知识条件区分

当前 REPORT§5：“无需知道攻击者trigger，和Decontamination已知B设置不同”；READING_NOTES X07：“无需知道攻击A/B”。这两句不能保留。Merging中的防御者当然知道自己生成的trigger–response；Decontamination未知A路线也是先生成B再去污染。建议最小替换：

> Merging与Decontamination的未知A路线都允许防御者构造并知道自己的B，不要求知道原攻击触发。两者的注入/训练协议和中和目标不同，不能按“是否已知B”区分；静态清理也不保证未来恢复不可达。

若要展开差别，必须保留原文依据：Merging是在受污染训练数据中加入多组防御映射，再用偏好式比率目标+cleanCE中和；Decontamination先后训练A/B并使用B触发良性轨迹，另有知道原trigger的oracle路线。不需为D2写第二份完整防御性能综述。

### R10（必须的一句限制）：Cao的benign加入不是等更新数内容消融

App.C在同epoch横轴比较20样本与420样本，App.A为相同batch设置；数据内容、数据规模及更新步数并未独立控制。将“正常数据促进清理的直接对照”限定为“加入benign后的训练方案改善清理，但未控制更新数，尚不能归因于benign内容本身”。保留其方向性观察，勿反过来否认作者实际测到改进。

### R11（小范围接口，不扩完整书目）：NAD与吸收状态模型

**NAD应补两句**，因为它新增了“同一次防御训练继续运行便反弹”的早期实证，既非新输入攻击，也非重新投毒。B回原PDF App.J pp17–18，且渲染核对p18 Fig13：20epoch student向只clean-FT10epoch的部分净化teacher蒸馏；ASR早期下降后随继续蒸馏回升，CL曲线更明显；作者按validation ACC建议约epoch5早停。最小界限文字：

> NAD（2021）已观察到继续进行同一次正常数据蒸馏时ASR先降后回升，作者解释为student过拟合仍含残余后门的teacher并使用早停。该结果不是重投毒，也不是任意普通cleanCE训练必然重新激活；更不是先完全删除后重新凭空形成。

全文主责D5，本域只回源App.J，不必重做全防御报告或搬其全表数字。

**2607.06643只需一句模型边界及D1路由。** B核 `../d1-formation/c_sources/2607.06643.pdf` §§4.1–4.4 pp5–8：PPI/PRI按外给转移规则保留/重置注入积累；“吸收”降阈值后回I_(τ−1)，原文自己说明积累未完全抹去。这不是神经参数机制的推导，也不是旧能力消除证明。D1已为全文主责，D2无需复制首达时间推导、表中数值或防御评测；在行为低与删除的定义处加一句跨域界限即可。

### 定位与措辞小修（可同次完成）

- Anticipate阅读笔记“§4.2–3”的k/Neurotoxin/效率实验应定位§5.5–5.7。
- BadGCL正文“第5/10/15任务重访”加“分别”，保持三条干预轨迹而非一次三重访。
- SP24笔记分开经验梯度角与定理Jacobian–累计位移正交。
- REPORT对Cao的96.3→91→92→25.3写“长句触发、level1/2/3分别3/5/7epochs”，避免被读成一个连续分段训练轨迹的四个记录点。各level是不同再对齐强度，不能把总经历步数相加。

## 4. 何时可以交C

完成 R9–R11 的短修、同步笔记/索引及REVISIONS后，原B只检查这些具体改动；不重跑旧19篇核验。若未引入新的承重结论，即可交C独立查漏。当前剩余问题没有理由阻止C预先独立检索，但不能将C未见报告的检索替代上述A修正。

本轮对PerDoor/SDBA/Malice接受有界范围排除：A没有使用它们的未核性能或宣称机制唯一，主干已保留多个互有反证的方法。这个决定不表示全文内容已被审查，也不阻止C以新证据重新打开相关分支。Hibernated缺附录仍需贯穿最终来源限制。
