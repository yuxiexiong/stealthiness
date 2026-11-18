# D3 C 独立查漏审计——第一轮交回 A/B

日期：2026-11-19。状态：**C_IMPORTANT_GAPS_FOUND / RETURN_TO_ORIGINAL_A_B**。C 不代改 REPORT/PAPERS，不签发终结；原 A 补证修订、原 B 核验后仍须由本 C 定向复核。无实验、GPU、论文修改或发布。

## 1. 独立性与实际路径

独立阶段及冻结候选见 `C_INDEPENDENT_SEARCH.md`。先用通用概念检索形成条件、dose/ratio/count、顺序、batch、模型规模、LoRA/rank、数据组成及综述，打开 arXiv、PMLR、NeurIPS、CVF、Springer、USENIX、TMLR 等原始入口；然后才读取本域 A/B 文件并比对。初次读取协议时命令意外输出尾部项目材料，但未据此生成查询；已在冻结记录披露。没有复制 A/B 或本地 sources 名单作独立搜索。

独立前向查询以 Souly 和 Shumailov 题名为种子；综述以 Zhao/TMLR 为导航；后向从新找到的 Carlini/Terzis、Schwarzschild 与 Cordyceps 的相关工作/参考文献追到已知模型/数据选择/参数化支路。比对后补充精确题名搜索核实 Carlini 2106.09667、Schwarzschild 2006.12557 和 LPS 2307.07328。多源指官方发布渠道互补，不冒称完成 OpenAlex/Scopus 的穷尽引文图或精确被引排序。

A/B 已充分处理不少容易遗漏的近邻：Homeopathic、Inherited、Subliminal、Agentic、Token-Level、Pick、Bowers；C 不要求无理由重读所有已核数字。但独立路径发现两个较早跨范式形成条件研究及一个最新语义控制研究尚未纳入 D3，见下。

## 2. 已全文核实的重要缺口

### C1. 固定毒数、规模与触发泛化的早期 CLIP 研究

**来源**：[Carlini & Terzis, *Poisoning and Backdooring Contrastive Learning*, ICLR 2022](https://arxiv.org/abs/2106.09667)。本地 `sources/c-contrastive2022.pdf/.txt`；首页确认 v2, 28 Mar 2022。C 已读正文 pp1–10（§1–6及伦理/复现声明），参考文献 pp10–13；该版无附录。OpenReview 返回非 PDF，未绕过访问限制，改用合法作者 arXiv 版。

**遗漏为何重要**：现报告以 LLM 新近结果组织 count/scale、位置与泛化，缺少这些条件已有 CLIP 直接实测的先例，而且它恰好提示不能把表示指标当 ASR。

- §4.1–4.3 pp5–6：CC3M/YFCC、ResNet50+文本 Transformer，默认 batch1024、30 epochs、Momentum SGD；backdoor 与单图 targeted poisoning 明确分开。后门 zero-shot 受影响而 linear probe 可能不受影响，说明下游读出方式也是适用条件，不是统一“表示已毒则所有用途等效”。§4.3 与 Figure2 的最低比例标注存在疑似不一致，勿转抄低比例绝对数。
- §5/5.1 pp6–7：后续消融实际改用 **backdoor z-score**——带触发图像的两两嵌入相似度相对自然分布的标准化差异。即使作者图注仍称 attack success，报告必须写表示代理；不能直接当指定目标分类比例，不能拿它闭合因果机制。
- §5.1.1/Fig4 pp7–8：低预算时固定位置较有效；毒例增多后随机训练位置更有效且对随机测试位置更稳。这是预算×训练变换/测试泛化交互，非“随机位置总更难”。
- §5.1.2/Fig5 p8：固定300毒例，从50k到3m训练样本的表示读出近似稳定；75例在高数据量下下降。模型1m–30m扫描呈较大模型读出更强/变异较低。不是普遍规模定律，也不是纯参数量因果隔离，更不是LLM任意输出的结论。
- §5.1.3/Fig6 p9：patch大小存在平台甚至大patch轻微回落，不是永远单调。

**原 A 任务**：补入 count/scale 与位置/泛化小节及判分口径、PAPERS/SEARCH_LOG；用至少一处说明“不同下游读出可不等效”。**原 B 任务**：核 §4.3 vs §5的ASR/代理切换、count固定条件及随机/固定位置交互，避免把作者图注强措辞照搬。C复核针对这三处，不要求重跑整域。

### C2. 形成条件 benchmark 的反例：比例、增强、架构、迁移和触发图案

**来源**：[Schwarzschild et al., *Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks*, ICML 2021](https://proceedings.mlr.press/v139/schwarzschild21a.html)。本地 `sources/c-toxic2021.pdf/.txt` 为正式主文；`sources/c-toxic2021-full.pdf/.txt` 为 [arXiv:2006.12557](https://arxiv.org/abs/2006.12557) 完整版含附录。C已读正式正文 pp1–8、完整版本 A.1–A.16 pp11–17 的相关文字/表格（特别Tables5–14、21–25）。pdftotext完整版本有xref重建提示，文本可提取；有表/版本不一致处不选数字强推。

**遗漏为何重要**：它不是新攻击目录，而是直接让原攻击在另一训练/数据条件下失效或变弱的系统比较，与D3的反证目标相符。

- §4 pp4–7、A.3/A.4/A.6 pp11–12：增强/归一化、受害架构、pretrain与finetune数据是否真正不重叠显著改变后门成功。仅将不相同benchmark终点并列不能隔离参数量或某个优化器因果。FC/CP为无触发 targeted poisoning，不能把它们的 Adam→SGD 对比概括为后门普遍规律；CLBD/HTBD才是本域直接证据。
- **重要口径纠正**：§4/Fig2 caption写“fixing poison budget”，但正文与A.7明示固定 **1%比例**，同时把毒例从5增至500、总样本从500增至50000；不是固定毒例数。这提供“固定比例不足以跨数据规模比较”的先例，不能与Souly固定count混成同一实验。
- A.7/Tables10–11 p13：原攻击设置与统一benchmark设置的曲线不同，不存在所有攻击一致的尺度趋势。A.14/Table22 pp16–17另外才是在固定总量下增加毒数量，CLBD/HTBD并非随数量处处单调；不把正文“通常增加”写成逐行定律。
- §4 p7、A.12 p14、A.14/Table21 p17：patch尺寸和patch图案本身都会改变后门有效性。不是只要都叫“小patch”就等价。
- §5/Table3 p8与完整版本Table25某些非后门行数值不一致，A.5文字也有基线epsilon疑似错句；应引用清楚的后门方向和控制，不拼接无关行/声称所有数字一致。

**原 A 任务**：补一个条件可比性与失败边界段落，纳入正确的固定比例说明与CLBD/HTBD范围；不用堆砌所有攻击数值。**原 B 任务**：核固定比例 vs 固定count、triggerless vs backdoor、统一benchmark与单因素消融之区别。可用定性结论避免版本表值歧义。

### C3. 可变语义控制：毒数、教师生成质量、推理监督与精确长度边界

**来源**：[Shao, Fleming & Baluta, *Cordyceps: Covert Control Attacks on LLMs via Data Poisoning*, USENIX Security 2026](https://www.usenix.org/conference/usenixsecurity26/presentation/shao-zedian)，[arXiv:2605.26595](https://arxiv.org/abs/2605.26595)。本地 `sources/c-cordyceps2026.pdf/.txt`，首页确认 **v2, 15 Jun 2026**。独立初筛HTML为v1，结论以下以下载v2为准。C已读正文§1–7 pp1–13、伦理/开放说明 pp13–14，以及相关附录A–F pp17–21（含Figures6–10和Tables9–10）。附录G证明没有全读，故不宣称重新证明理论；下面不将理论结论当实验事实。

**遗漏为何重要**：将形成结果从固定trigger→固定payload扩到通过语义线索产生可变控制；同时有直接数据数量/组成/训练损失条件，以及高语义指标与精确恢复失败的反证。

- §5.1 pp8–9：UCC为140毒例加1260正常例，训练十个anchors/任务、测试新anchors，7×7任务对共2450条；BCC为100毒例加900正常例、100测试目标。5个模型包括dense、reasoning及MoE，**非干净受控规模扫描**。AppC.1 p19：LoRA r16、all targets、LR4e-4、5 epochs；100%毒可行性实验4epochs另计。
- §5.2/Table3 p9：所谓从10%降到1%，**保留100个毒例，把总数据1000增至10000**，不是100减到10毒例；毒count/总训练步数/正常暴露应明确。改变生成毒数据的oracle（包括不同家族）后CA不同；不能把跨oracle差异只归因参数量，且不需要teacher/student同初始化。这与subliminal同初始化条件是不同传递范式，不是直接反驳后者。
- §5.2/Table3 p9：对训练毒文本改写后部分模型仍有效、部分下降，属于数据语义与表面形式的条件性泛化；不等于任意改写无影响。
- AppA/B pp17–20：100%毒另设定比较full reasoning与weighted reasoning；语义相似度与EMR不同。Fig7 p20显示精确恢复随plaintext token长度下降，到5tokens接近零；不能把高SCS或BCC近九成CA改写为任意精确长payload能力。`CER<1`也不是高可靠性的同义词。
- **必须保留原文不一致**：AppB正文p19说Phi4-15B、最优SCS0.770；Fig8图注p20写Qwen3-4B，Table9的0.770亦为Qwen3-4B。不要无声选择一项后宣布某模型最优权重。`reasoning weight=0`仍有reasoning tokens在输入前缀，不能无条件等同完全移除CoT训练序列或Phase2。
- §5.5 pp11–12：用100%毒的Llama3-8B模型、20样本和少量消息做表示几何/残差分布观察；不是自然训练阶段的中介干预。高cosine/t-SNE近邻、残差边缘直方图不能充分证明全空间加性、各向同性/独立高斯或唯一形成机制。
- D4/D5交界：§5.3、AppF中防御采用因推理长度下降而很早停止的设置；不能称充分训练仍无法清除。adaptive detector提高识别同时高FPR是实测限制。主审负责路由给D4/D5，不假定这些域已经读过。

**原 A 任务**：补数据composition、count与语义/精确读出边界，PAPERS注明v2和上述不一致；保持本域条件重点。**原 B 任务**：核Table3分母、oracle比较、Fig7/8及Table9、CA/SCS/EMR区分、100%毒与10%/1%实验不能混用。需要理论声称时另读G，不应为了收口强推机制。

## 3. 冻结候选逐项去向与剩余线索

| 独立候选 | 比对后去向、依据与阅读标签 |
|---|---|
|Souly2510.07192|已纳D3，§3/4/5及C/F/J在A/B中有逐项核读；C无新增主要遗漏，不重复宣称自己全读。|
|Pick2609.15029|已纳D3，集合交互、full/mini batch与剂量混杂均覆盖。|
|Token2605.30189|已纳D3，rank表脚注/跨seed/终点非时间轨迹已修。|
|Liang2505.12871|已纳D3，backdoor vs untargeted、rank边界已覆盖。|
|Learning Curves2106.07214|D1 REPORT及B_REVIEW已处理；D3 SEARCH_LOG有合法完整版。当前D1 PAPERS标签是CROSS_DOMAIN_SECTION_CHECKED，不冒称D1独立全读；主审可将D3早期读源与D1核查责任整理一致。D3宜增加指向，尤其regularization/complexity不能自动等于神经网规模。|
|Shumailov2104.09667|已纳D3，replacement可重复/省略而非纯排列、误差条、NLP简模型已覆盖。|
|Schwarzschild2006.12557|新重要缺口C2，全文+相关附录核实后交回A/B。|
|Carlini2106.09667|新重要缺口C1，正文全读、无附录后交回A/B。|
|Harvard Phase Transitions课程文|D1已有正文和B原文审查，身份、seed/相同数据矛盾等已限；D3可交叉指引，不另当新peer-reviewed证据。|
|Genome2603.27465|D1已有全文与B核验（TATA失败、概率vs完整payload），适用边界接缝可指向D1，不重复计篇。|
|Wu ICCV2023 Computation and Data Efficient Backdoor Attacks|官方方法段初筛：confidence/distance-based sample selection、CV/3D；与D3已纳Pick的“固定集合选择影响结果”宽命题重复，但构成前史。尚未全文核实，不引用其提升数字、不称此分支始于Pick。原A可在选择前史加筛查路由，若需任何结果数值须全文读。|
|Truong CVPRW2020 Systematic Evaluation, 2004.11514|官方正文引言初筛明确regularization/architecture/clean-finetune研究；其regularization可能逆向是有价值待核线索。不能当abstract已证反例。交原A判断C2/现有LearningCurves是否已覆盖该子问题；若未覆盖应定向全读再由B核，不因三项已找到就丢弃。|
|LPS Neurocomputing2026, doi10.1016/j.neucom.2026.134024|精确题名追到公开前稿2307.07328v1 (2023)及ICLR2024提交，纠正“全新2026方法”的潜在误读。已看方法/结果页作范围筛查，未读完新版；不引用最新ViT/non-IID结果。与样本选择已有证据重叠，仅保留书目/版本路由；如报告要讨论选择法跨范式普适性，新版全文受限仍必须明示。|
|Associative vs Distributional JCP2026|已纳D3，单seed、触发数/剂量混杂、SVD负结果已修。|
|Cordyceps2605.26595|新重要缺口C3，正文+相关附录核实后交回A/B。|
|Why LoRA Fails to Forget2601.06305|仅摘要筛查，以clean数据后续移除为主；属于D2/D5形成后范式，当前D3不以此支持形成因果。交主审路由，不假称其他域已核。|
|ToxScreen2607.26849|D4 PAPERS显示FULLTEXT_READ。检测benchmark边界，D3不重复纳入为独立形成定律。|

冻结后追踪到的FUS（IJCAI2022）是Wu/LPS的方法前史，不当新的形成机制结论；通用influence、SGD、Steganography仅为方法/理论源，不无限扩为一般学习或所有攻击。未来最终报告不得用“首次证明样本组成影响后门”概括Pick。

## 4. 有界收口状态

本轮完成：独立候选冻结、多源/同义词/综述与相关前后引用链入口、与A/B及共享领域的比对、三项重要新增的正文与相关附录核读、具体修订任务。**没有完成最终闭环**：C1–C3待原A/B，Truong正则化线索需原A给有据去向。下轮C仅核修订及这些相关分支，若出现改变判断的新线索则继续对应分支；无关攻击目录不扩展。

当前无三项核心文献全文不可得障碍；LPS2026新版未取得完整正文，不能声称核实其新增结果。不能以本轮找到三篇为固定停止配额，也不承诺世界零遗漏。只有原A/B处理完重要线索、C末轮未再发现改变判断的重要遗漏，才能写声明范围内路径完成；现在明确保持待修订状态。

## 5. 原 A/B 修订后的 C 定向复核与有界结束（2026-11-19）

**最新状态：C_BOUNDED_COMPLETE_WITH_SOURCE_LIMITS。** 本节取代上文第一轮的待修订状态，保留历史轨迹。原A已全文补入四篇，原B §7已对新增25项（85–109）逐项回原文核验，§8确认唯一R11措辞已关闭。本C读取当前REPORT/PAPERS/SEARCH_LOG/REVISIONS及B §7–8，只复核C1–C3与Truong相关增量，没有重跑旧84项。

| 缺口 | 本C复核结果 |
|---|---|
| C1 Carlini/Terzis | REPORT §2/3.5、PAPERS已区分§4.3分类成功与§5表示z-score；固定300/75毒例、位置×预算交互、zero-shot/probe差别均保留。不能推LLM普遍规模定律的边界明确。关闭。 |
| C2 Schwarzschild | §2/3.6/6.6准确分开固定1%比例与固定总量Table22，限制后门结论到CLBD/HTBD，增强组合与纯优化器效应未混淆。A/B发现正式/完整版本差异也涉及后门行，本C采纳这一比首轮更完整的纠正；不拼接数值。关闭。 |
| C3 Cordyceps | §2/5.7准确保留100毒例扩总量、oracle/内容共同变化、语义显式训练与subliminal不同、100%毒附录与主实验分开；CA/SCS/EMR、权重0仍可保留前缀、Fig8模型冲突及表示分析非中介证据均落实。A/B视觉核Fig7后将本C首轮沿正文的“随长度下降”精确化为总体下降且可局部回升，本C采纳。关闭。 |
| Truong正则化线索 | 原A已按C要求全文纳入，本C回正文§2–4及结论核目标类分母、源类原图保留、五方法×五类×十run、SNNL82→51、many-to-one同毒数、式7及高α含义。§6.7已准确写为高α＝更高触发权重/更明显；没有普遍单调正则强度或容量因果的强推。关闭。 |

本次补查实际查询：`"Cordyceps" "backdoor" "poisoning" "2026" ablation`、`"Systematic Evaluation of Backdoor Data Poisoning" regularization follow up`。返回USENIX/CVF/arXiv原始论文与相关索引，未发现必须改变本域结论的明确后继结果；二手转述没有升格为独立证据。结合上一轮已完成的多源、同义词、综述、最新候选及前后向路径，本轮仅补新增相关支路。Truong后向的通用regularization、clean-label攻击、已纳benchmark是方法来源或同类例证，不触发无限扩篇。未见新的重要子问题空白。

核对：PAPERS共24条且ID唯一，四篇新增原文路径存在，FULLTEXT_READ记录与实际正文/相关附录责任一致；Cordyceps明确排除未全读G证明的定理核证。C不改A最终REPORT；其首尾流程状态仍写B/C待核，交主审/原A作状态同步即可，不是实质证据未完成。

**有界结束依据**：截至2026-11-19，本D3定义范围内规定检索路径已执行，独立候选与重要线索均有处置，确证的重要漏文及反证已由原A补证、原B核验、原C复核；末轮未发现改变判断的重要遗漏。无固定篇数/轮数作为停止理由，也不承诺世界零遗漏或所有索引已收录截止日全部论文。

**继续保留的来源限制**：LPS2026新版未全读，其2023前稿/选择前史仅筛查路由，未引用新版ViT/non-IID结果；Wu等未全读者亦不计入已读证据。其他综述及完整引文图未穷尽，不能声称数据库无遗漏。Cordyceps G证明不作理论验证；局部原文版本/指标/图注矛盾已在正文保留。Learning Curves跨域阅读责任、D4/D5对Cordyceps及LoRA移除分支的处置由主审统一，不能把本D3闭环冒充六域全部完工。这些限制没有被用来支持D3核心结论或证明新颖性。
