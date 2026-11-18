# D4 C独立查漏：首轮审计及聚焦末轮

**当前状态（2026-11-19）：C1分支已闭合，原C聚焦复核有界结束。未发现需要改变当前报告判断的新增重要遗漏；保留来源不可得、未全文候选和信息先验限制，不作全域穷尽或绝对零遗漏承诺。以下首轮待办状态留作历史，末轮记录见文末。**

日期：2026-11-19。**首轮结束：发现1个重要分支遗漏，交原A补充、原B核验，再由原C只复核此分支；本轮不签全域完成。** 已完成协议要求的“先独立检索冻结、后比对”：`C_INDEPENDENT_SEARCH.md`先于阅读A/B/PAPERS/REVISIONS写入。没有读项目私有实验材料、设计实验、GPU执行或发布。

## C1 重要分支遗漏：训练前毒样本筛查，不等于已知trait风险预测

当前REPORT §1表格及§6对训练前工作主要列Persona Vectors/From Data to Behavior，训练动态列ABL/SEEP等；这遗漏了**在待防御模型学习这些数据前，仅从标签统计或现成PLM辨识毒样本**这一独立路线。它不建立自然形成的未来时间预警，也不估计微调后的连续风险幅度；但在“训练前是否已有可用信号”的知识版图中不能由trait预测或先训练再筛除代替。

C已完整读以下三篇正文、相关附录，并保存公开PDF/text于`c_sources/`。三篇合并为一个补充任务，不按论文数量扩成三轮。

### C1a Z-defense：不先训练待防御模型的标签—特征统计

[Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation](https://aclanthology.org/2023.emnlp-main.60/)，He等，EMNLP2023，DOI10.18653/v1/2023.emnlp-main.60；`c_sources/zdefense2023.pdf/.txt`。

- **实际阅读**：正文§1–5及Limitations，印刷pp953–961；附录A–C pp965–967。参考文献用于后向线索，不假称被引论文全部已读。
- **方法/信息条件（§3 pp955–956；§4.1–4.2 pp956–958）**：从整个待训练集及其标签计算unigram或句法ancestor-path与标签的z-score；用数据标签频率作为先验，筛出异常相关特征及含这些特征的样本。无需先训练待防御模型、独立干净参考集或精确trigger字符串；句法分支使用Stanza parser，所以“model-free”不能解读为完全不依赖任何语言处理模型。词/句法特征家族由防御者选择；默认阈值18来自前期观察，不是无假设概率保证。
- **实测与反证（§4.2 Table2–4 pp958–959）**：默认20%毒、BERT 3epoch，四个分类/NLI数据集。检测FAR/FRR与过滤后ASR须分开；AGNews Syntactic的Z-SEQ虽然漏毒仅7.2%，过滤后ASR仍99.5%，说明大部分毒样本检出不保证终局安全。低FAR也可能仍有高benign-on-trigger错误，作者另报BASR，不应把残余全称为后门。比较时基线删除量匹配Z-SEQ，而非各基线原流程自然选择。
- **边界（Limitations p961；AppB/C pp965–967）**：假定初始PLM干净，不能清掉已在预训练植入的后门；风格触发未验证，特征设计针对已知攻击家族；附录改变特征可明显改变效果，更多/更复杂特征并非自动更好。作者将其定为经验观察，未证明所有后门必存在可检标签相关性。
- **结论等级**：方法+所测条件实证。已有“训练前数据异常筛查”先例；**不是**自然形成时点的在线预警或触发—目标形成完整机制。

### C1b PKAD：借现成PLM在目标训练前辨识文本—标签失配

[PKAD: Pretrained Knowledge is All You Need to Detect and Mitigate Textual Backdoor Attacks](https://aclanthology.org/2024.findings-emnlp.335/)，Yu Chen等，Findings EMNLP2024，DOI10.18653/v1/2024.findings-emnlp.335；`c_sources/pkad2024.pdf/.txt`。

- **实际阅读**：正文§1–6 pp5837–5841；附录A–H pp5843–5849，含全部方法假设/参数/跨模型/污染率/提示模板。两栏提取截断部分已按PDF页重新读取，不以一次cat截断输出算完读。
- **方法/条件（§3 pp5838–5840）**：使用现成PLM多层表示，按类间Mahalanobis距离选择第一层，比较文本表示与提供标签的失配程度，赋弱毒/净标签；再选可分层并以LDA迭代扩大毒集合，净集合固定。无需预先提供精确trigger/恶意目标类或独立已认证干净集；**仍需任务标签空间、数据标签及可用PLM特征访问**，不是完全未知任务/目标的无标签预测。
- **容易误读的附录（AppC/D pp5844–5846）**：rp是防御者对潜在污染率的信念/超参数，默认0.03；rmis推导使用经验拟合的净/毒分布形状。不能简写为无需污染先验或无需选择阈值。主实验污染2%，目标为标签空间首类（§4.1 p5840）；这是评价构造，不是向检测算法显式给定真实target。
- **实测/边界（§4、§6、AppE–G）**：主实验Gemma7B、四攻击；清洗后ASR/CACC和F1分开。AppE Table4 p5846中BeaverTails句法攻击仍ASR50%，不能沿用主表“全攻击清零”。AppF Table7 pp5847–5848显示不同PLM显著影响TPR；AppG Table8显示rp提高会增加误报。§6 p5841明确clean-label输入标签匹配与攻击者控制模型训练过程仍未解决。
- **结论等级**：目标训练前毒数据识别已有实证，非自然训练中未来行为概率、首次风险时刻或完全未知trigger-family保证。

### C1c NETE：不访问受害模型的样本评分，zero-shot与阈值先验必须分开

[Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models](https://arxiv.org/abs/2509.05318)，Peng等，arXiv v1；正式刊为Neural Networks193:108025（2026），DOI10.1016/j.neunet.2025.108025，在线发表2025-08-24，书目由[PubMed](https://pubmed.ncbi.nlm.nih.gov/40886451/)及作者repo交叉核；全文证据基于保存的arXiv v1，而非假称逐字核对刊版。`c_sources/nete2509.05318.pdf/.txt`。

- **实际阅读**：正文§1–8，PDF pp1–31（原文罗马页i–xxxi）；无独立附录，pp32–39为参考文献。相关参考文献[20]–[46] pp34–37核后向链。图中文字抽取有控制字符，本审计不从这些图估计新数字；主要量化取可清楚读出的Table4/5/7和正文阈值段。
- **方法/条件（§3 p8；§5 pp11–16，Algorithm1 p14）**：用现成评分LM和mask-filling模型，随机mask约10%词、span最多2，反复补全后取log概率差异/规范化统计，阈值判疑毒。不需给出精确trigger或target，也不需访问待防御模型；它可以在数据学习前或请求进入模型前打分。输入是单个可疑样本，不是对未来模型危险程度的估计；作者称curvature对应关系含连续近似/扰动分布假设，不能由算法名推为后门的普遍特有机制。
- **关键先验（§6.3 p23/xxiii）**：论文主结果主要是AUROC排序。其固定阈值实验明确用**200个已知word-level毒样本**的扰动差异均值，迁移到sentence/syntactic/style；因此“不需额外干净集”不等于无已知攻击校准，“zero-shot”不能作为部署FPR已可靠校准。不把阈值段的模糊detection results自动当AUROC或FPR。
- **反证/局限（§6.2 pp18–20；§6.4 pp25–29；§7 pp30–31）**：Table5 OLID word AUROC0.65、YELP sentence0.71，分布可重叠；Table7 Sleeper-agent样本AUROC0.74，非普遍完美。方法还能识别部分对抗/越狱样本，因此这些测量本身不证明“已植入后门”特异性。实验证据主要是样本排序，不等同训练后ASR消除或未来风险校准；实时成本随文本长度增长。作者所谓adaptive场景是指定temperature/LLM生成变体，不能扩大为对任意自适应攻击可靠。
- **结论等级**：训练前/推理前样本筛查路线与其信息边界，不是训练形成前兆；最新低引用检索补进同一分支的2025在线/2026刊发证据。

## 给原A、原B的有界任务

**原A**：在REPORT类型表新增“训练前毒样本筛查”一行，与“已知trait的训练前风险幅度预测”分开；以一个短节整合以上三篇及各自信息条件/关键失败。PAPERS纳入三篇并注明实际读取版本，阅读记录、SEARCH_LOG及REVISIONS记录C1回应。可直接复用`c_sources/`，无需重复下载。最小必要内容：Z-defense标签相关/已知特征家族、PKAD标签+rp/clean-label限制、NETE样本排序+已知毒校准；不需复制所有数字或扩防御目录。

**原B**：逐项核新增方法/信息条件/重要反证，重点回原文Z-defense§3、Table2–4、Limitations；PKAD§3/§6/AppC/D/E/G；NETE§3/Algorithm1/§6.3 Threshold analysis/Table5/7。若A不采用某个数字无需为删去的数字额外核验。核对stage、样本层vs模型层、识别vs风险预测三类区分。

**原C**：A/B完成后只看C1变更、相应来源与尚未关闭接口；不重搜已确认无改的18项或整个六域。

## 其余独立候选处置（不扩轮）

- SEEP、BadWindtunnel、TimeGuard、Rethinking、ToxScreen、ResearchSquare precursor protocol已在当前A/B修订覆盖，本C不重复全文或重复提出遗漏。protocol已核身份且无确认性轨迹的状态保留。
- DeBackdoor、ONION及新近推理检测Ulterior Motives2604.23460/InertiaGuard2610.16598：在本次检索中仅作初筛。现有报告已覆盖完成模型审计和推理期“提前”，未见需改变核心阶段区分的独立证据；**不把这些候选标全文或据摘要写效果结论**。
- Signals in the Dark为作者技术博客线索，未在本轮全文核；其静态trigger identity/可读性≠因果使用的候选主题已由更直接protocol与既有边界覆盖，故不以非论文同类例子扩大轮次。
- 后向链核实：PKAD参考文献明确引Z-defense（p5842）及ONION/CUBE/DAN；Z-defense§2–3连接Gardner2021/Wu2022统计偏差筛查，向后接SEEP（现报告已全文核）；NETE[44]明引DetectGPT曲率式样本检测，[32]/[33]接memorization/UnToken。这里只核引文关系，不继承被引文献结论。
- NETE引出的Data-centric NLP Backdoor Defense from the Lens of Memorization（arXiv2409.14200，NAACL Findings2025.316）经准确题名/官方ACL初筛，属于候选发现后仍以已训练模型确认行为的路线；未全文，不对“重复元素必要性”作结论。主审确认D1/D3/D5未纳入；不能默认他域读过。就D4信息阶段版图目前为同类补例，记录而不新增强制轮；后续若原文有独特形成前证据才重开。SemPurify2610.23589初筛为图像净化，不扩D4。
- 三综述检索入口和候选已独立冻结；与A已用的TMLR/2502.05224重合，保留A的SECTION_READ、不伪报本C整篇重读。

## 检索结束与限制

独立查询覆盖经典训练统计、模型/样本检测、前置信号同义词、2025–2026近期题名及综述；冻结后只对新增“训练前毒数据筛查”分支补全文及后向/前向题名查询。前向检索实际追加：`"PKAD" "backdoor" "2026"`、NETE全题名+2026、`"Z-defence" "SEEP"`；命中主要同稿/作者记录及同类防御，没有新增改变该分支阶段定位的全文证据。未取得完整引用图，未声称穷尽数据库或绝对零遗漏。

shell默认DNS失败后，对三篇公开ACL/arXiv文件的require_escalated下载获准；没有自动审查拒绝/绕行。已有重要受限项IEEE MRAI loss+uncertainty、SSRN anchored diffusion、InfoFusion LDD继续保留：本轮准确题名搜索仍仅书目/二手摘要，没有新合法全文入口，不能提升为已核或认定其结论。PSSP/DFBScanner/代码谱等已列候选仍未由本C全文读取；从A现有阶段筛查并非新的形成前子问题，不能将“下载未读”变成FULLTEXT_READ，也无需为了题名配额无限扩展。GradSentry接口由D5回传后据实处理。

**本轮判定：C1待A/B；全域仍是有边界、受限的查重，不是全面完成。** C1修订及B核验后才由C作聚焦末轮判定。新增文件仅C记录、3篇公开全文/提取文本及待整合书目；没有修改A/B正文、共享索引或项目研究结论。


## 原C聚焦末轮：C1与GradSentry接口闭合（2026-11-19）

**结论：C1通过，有界结束本域C查漏。** 本轮仅读原A在REPORT类型表/§2.8/§6–7的变更、C1_NOTES_A、REVISIONS与PAPERS新增状态，以及原B_REVIEW §8；沿用本人首轮已完成的三篇全文阅读，不重复搜索、下载或重审旧18项。新增GradSentry仅回原文核接口，未假称本C整篇阅读。

|原C要求|本轮落点与确认|处置|
|---|---|---|
|训练前毒样本筛查单独成类|REPORT §1单列，§2.8明确与trait风险幅度预测、自然形成预警区分；§6同步。|闭合|
|Z-defense不能称无先验；检测不等于终局安全|§2.8保留标签频率、选定特征家族/Stanza、经验18阈值、初始PLM干净条件、AGNews句法FAR7.2但ASR99.5及匹配删除量。与C首轮全文、B§8一致。|闭合|
|PKAD的标签和污染率信念/clean-label边界|§2.8保留内部特征/标签、r_p=.03与经验拟合，区别实验2%/首类target和算法输入；保留BeaverTails句法50%残余、PLM依赖/误报及clean-label/model-poisoning未解。|闭合|
|NETE不把zero-shot/AUROC当无攻击校准或未来风险|§2.8明确200个已知word-level毒样本阈值校准、AUROC三例、无受害模型/输入标签与仍需外部模型的差别、有限适应评估/成本/曲率条件。版本限定arXiv v1；A额外发现Fig2题注方向与算法不符，按Eq4/Algorithm1解释且不引用矛盾题注，合理。|闭合|
|GradSentry跨域接口不能静默悬置或当在线预警|C实际核`../d5-defense/sources/C_GradSentry.txt` §3–4.1 PDFpp3–5、AppG p15、AppH p16相关automatic/optimal阈值说明、J.4 p20。逐例输入—目标回答的lm_head梯度、截前1/8行列、top16谱熵；无优化更新；全数据KDE/Silverman谷值，无双峰回退.7；直接entropy最小化未测。REPORT仅报这些阶段/先验边界，D5承担完整效果，合适。|闭合；C为SECTION_READ|

PAPERS已纳三篇FULLTEXT_READ与GradSentry SECTION_READ，A记录了自己的实际补读；B§8覆盖全部新增实质主张，无未回应的科学必改项。REPORT/PAPERS中的“待B/C”字样是此次交审快照，主审可按本末轮结论同步为有界完成/受限结束状态；无须为行政状态另开科研核验轮。

**补查判断与停止理由**：C首轮已对新增分支完成正式源全文、后向关系及前向准确题名检索，本轮A/B没有引入改变问题、信息条件或结论的新材料分支，GradSentry接口亦已有D5全文与D4相关原文核验。故无需重复旧检索或扩同类目录；在声明范围内，本末轮未发现改变现有判断的重要遗漏。此判断不抹去MRAI/SSRN/InfoFusion等核心全文受限项，也不将PSSP/DFBScanner/代码谱等筛查状态升级为全文。报告必须继续保留“未核线索不能作结论、没有完整引用图”的边界；不应改成“全部相关研究已审完”或未知风险预警不存在的断言。

没有新增实验设计、科研方向建议、GPU执行、论文编辑或发布。C工作到此停止；若以后出现新的一手证据改变此分支，再依该具体证据开启定点复核。
