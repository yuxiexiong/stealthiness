# D4 独立 B 原文核验

核验日：2026-11-19。审查对象为当前 `REPORT.md`、`PAPERS.json`、`READING_LOG.md`、`SEARCH_LOG.md`，按 `PROTOCOL.md` 和 `shared/PROJECT_COMPARISON.md` 执行。B 未改 A 正文，未做实验、GPU、论文修改或发布，也未递归委派。下列“支持”是文献中的方法/结果支持报告的限定陈述，不是独立复现实验。

**首轮结论：退回原 A 作针对性修订，再由本 B 复核；当前不能签最终完成。** 15 项核心材料的实质方法、结果和重合判断已逐项回原文；不是抽查几个代表例。A 对 oracle、适配基线、任务族、检测/清除和推理/训练时间的区分大体准确。主要问题是跨域重要结果仍未整合，以及“先变”指标本身的未来信息、修改后 Haystack 的高非触发失败率等限定尚缺。未完成线索仍须保留受限状态，不能由核心条目核验替代全文/查漏完成。

## 1. 必须由 A 处理的修订

|编号|原位置与问题|原文证据和具体处理|
|---|---|---|
|R1|§5 Model Organisms 把限制主要放在多 adapter 的最终参照，遗漏单 adapter 指标也含未来信息|2506.11613v1 App F p19 的 local cosine 比较 **t−s 与 t+s**；PCA 基于整条轨迹。保留约180步转动、300–600步行为增长及5倍放大结果，但明说该转动指标在时刻t不能只用截至t的信息算出；使用它预警需要延迟或另行定义/验证因果在线版本。此处不要把单 adapter 结果写成已验证实时预警。|
|R2|§5 Homeopathic 只列两个终点，§6瞬态讨论没有纳入已取得的直接训练证据|2611.10110 App H Fig H.3 pp44–45：Gemma-3-4b-it，15k毒+15k净，9个 full-FT 学生均先表现 German，6/9在7–16 epochs回到0，控制始终0；LoRA 9/9在epoch20仍保留。应加入“终点评估可漏训练中瞬态”的直接文献证据。Full-FT LR 2e-5 与 LoRA 1e-4不同，不能归因为微调方式单因素因果规律，也不是未知风险盲预警。|
|R3|§5 Weird / Mirage 仍只标跨域待读，缺少影响先变/突变解释的重要反证|按下表 X5/X6 回原文整合：Weird 有平滑loss与行为跃升、选择成功seed的条件、已知年份对照的SAE及消融；Mirage 有长度匹配、不同训练配置与反复方向转动，不能整体否定EM或当直接复现5倍放大。需A自己记录新读节段，不能把B读过算A全文。|
|R4|Persona 的“可预测/可筛选”未交代实际筛选质量边界|2507.21509v3 App L pp54–56：极端高投影筛选可使输出不连贯/故事化，作者不把它当当前实用方法；附加toxicity过滤可阻止evil上升。加入这些结果，不能只报500例排序。|
|R5|Haystack 只报告修改前40–70%非触发漏洞，易使读者以为修改后均高度选择性|2602.03085v1 App A.1 Table4 p13，修改后 Llama-3.1-8B seed2025 的 ASR=.914、FTR=.858。应限定47个中毒模型并非全都强触发选择性；保留真实检测分母，不把高无触发漏洞当触发特异发现。|
|R6|小的出处/措辞问题|SpecGuard条件定理在§4非§3；TimeGuard G.5主要完整结果是Table30，勿写Table7/G.5；EVO为延迟dual-trigger/双模态触发，非“双目标”；BadWindtunnel“多trigger”应为四种攻击分别评估，避免误读为同时多触发；Z-PEFT基线只跑“接近3 epochs”，非确定完整3。|
|R7|BaDLoss“不同表部分数字不一致”缺具体可核出处；ABL“极高投毒下效果失效”泛化过宽|BaDLoss请给明确表号、行列和可比条件，或删除这句。Table1为single、Table2为multi，不能因数值不同称矛盾；Table4/5的消融也须先核配置。可确定的Table4题注把主要残余说成sinusoid，但GTSRB r=.4表内Frequency=49.32、Sinusoid=3.15，此为题注/表内不符。ABL Table2在70%投毒时BadNets仍5.02，而Blend62.28/Dynamic69.36，宜写部分攻击失效。|
|R8|TopKLoRA/HARVEY仅移交D5，信号边界未收口|2610.32866 §6–7的稀疏新adapter定位/消融须已知trigger和对照对；2607.05748 §3的参考模型主动学毒/忘净制造分离。各加短边界说明即可，不需复制D5全文；不可归为未知触发自然预警。|

## 2. 核心15项：逐项 claim → 原文 → 判定

页码以论文印刷/PDF页为准；TimeGuard `.read.txt` 的自动页标在p8后有控制字符问题，以下依印刷页/表节。原文均为本地 `sources/` 保存的作者/会议全文；arXiv号和版本见各行。多项同条件主张合并列出，不能将“支持”扩大到未测试机制或总体部署可靠性。

|ID|A的实质主张|B实际核对原文位置|判定/限定|
|S1|Spectral先训练、按类中心化SVD、删异常后重训|1811.00636v1 §2.2–3 pp4–6 Algorithm1/Fig3|支持；删除量1.5ε，ε是污染数量上界，含污染先验；算法在p5。|
|S2|谱理论有条件，不是后门必有谱迹|§3及App A pp12–13|支持：均值分离相对协方差的充分条件。|
|S3|CIFAR/ResNet/VGG、输入空间及层选择限制|§4 pp6–8，App B Tables2–5 pp14–16|支持；挑选去除效果好的后层，部分低投毒攻击未成功安装；非任意层/模型规律。|
|S4|不构成自然时间领先或完整机制|同上方法、理论和实验范围|支持为报告范围判断：没有训练前未来行为校准或方向中介证明。|
|A1|AC训练后末层、ICA二簇及ExRe/大小/轮廓规则|1811.03728v1 §5 pp4–5 Algorithm1|支持；要可疑数据及内部访问，独立净集理想但非硬性要求。|
|A2|不能把ExRe性能归给单阈值|§6.3 pp7–8 Table3|支持：毒簇silhouette .10/.09与净≤.11重合，干净70/30也混淆大小规则。|
|A3|低投毒安装失败及机制假说|§6 p6脚注、§7 p8|支持：LISA≤10%未安装不是防御成功；源类表征解释是作者假说。|
|L1|低loss观察、后门已学入、LGA人为分离|2110.11571v3 §3.1–3.2 pp4–6|支持；不能把受干预分离归为无干预自然前兆。|
|L2|20 LGA+60正常+20 GGA|App A.3 pp14–15|支持；非第20epoch即持久清除。|
|L3|残余ASR和极高投毒限制|Table1 p7、Table2/§4.2 p9|支持部分攻击；R7须限制70%投毒失败说法。|
|L4|高loss适应攻击未排除|App B.5 p18|支持：提出可能规避，非已验证全面抵抗。|
|B1|BaDLoss快/慢/交叉loss，用部分轨迹、净例距离、过滤重训|ICLR2025 `badloss.pdf` §4/Fig2 pp6–7|支持；Fig2在§4附近，不应泛指§2；Narcissus前10epoch较高后较低。|
|B2|250净例、50邻居、30%epochs、删40%|§4–5.1 pp7–8|支持；真实总体污染约8/10/9%，删除显著更高。|
|B3|少数类误删及适应边界|App G pp19–20；§6.3–6.4 p10；App D p18|支持：最多10%类25.5%，最少10%类52.5%；匹配轨迹是可能规避，复杂不等于不可能。|
|B4|未全面测同例叠加多个trigger|App F p19|支持；同时多攻击不等于同例多个trigger。|
|B5|ABL改成15%最低loss过滤重训，AC为轮廓规则|App I.2 pp21–22|支持；非原ABL1%隔离+遗忘流程，所以不能以此断言原方法普遍失败。|
|B6|过滤率/架构影响净精度及残余|App J p23 Tables4/5|支持；CIFAR r=.4 CA88.43/ASR11.35，r=.6 CA81.69/ASR7.25等呈权衡。所谓“不同表数字不一致”尚缺可比定位，见R7。|
|W1|BadWindtunnel受噪模拟训练、类平衡、目标筛选聚类后指导实际训练|ACL2025 `badwindtunnel.pdf` §3 pp3–5、App B/C/F/G pp11–15|支持；前后confidence difference c(1)−c(0)，不是统计variance；目标类由top5%再聚类推断。|
|W2|BERT四数据/多攻击效果及限制|§4 pp5–7 Table1；§6及附录|支持所测设置；R6改“各自评估四种攻击”；低投毒Synbkd及Style残余限制，不是自然危险时点预测。|
|T1|TimeGuard 13防御适配、通道稀释、毒净loss迅速趋同|2605.22365v2 §3 pp3–5 Fig3/Table1|支持；ABL/ESTI平均FDER .497；回归与重叠窗口为作者解释，不是普适定理。|
|T2|time×channel、反向loss+邻域构造净池、无需外部净集|§4 pp5–7|支持；辅助backcaster/净池过程仍是特定防御训练，不是未经干预自然预警。|
|T3|回归适配影响基线结论|App F.7 pp25–26|支持；谱/PCA相关方法用kmeans伪类，k∈{5,10,20}取最好，损失/扰动均改造。|
|T4|一种适应攻击、嵌入未胜、净池动态图、motif混淆|App G.5–8 pp40–42 Tables30–32；App I结尾|支持；R6改表号。TS2Vec约80,000秒只Weather测试，非通用嵌入无效。|
|R1a|Rethinking不同训练配方仍高ASR但检测降|EMNLP2025 `rethinking.pdf` §4–5 pp3–6、App A–G pp10–12|支持；moderate/conservative/aggressive同时改LR/污染/停止条件，不能单变量因果归因。|
|R1b|Llama77/17/7及分母、图注矛盾|App D Table6 p12；§5.2、§5.4和Fig3|支持；分母是中毒模型检出率，正文HSOL/图注SST2确有不一致。|
|X1a|ToxScreen约24k候选内含真trigger及训练位置|2607.26849v1 §4.1 p4|支持；最关键oracle条件必须保留。|
|X1b|14/28 rank1，17/28 top3；高ASR未必植入特异|§5.1 p6|支持；不等于完全盲触发恢复。|
|X1c|rho .70/.80不一致、剪枝和ghost非普遍|§4.2/5 pp6–7 App F；§6.1–6.2 p8|支持；90%稀疏sentiment ASR61–81%、MMLU .24近随机，不能当保效清除；曲率是诊断关联。|
|H1|Haystack泄漏、候选、排序及行为验证|2602.03085v1 §3–4 pp3–6|支持；510解码配置、DBSCAN/ngram、attention/entropy/基线偏移；最后需固定输出相似性或代码漏洞任务族检验。|
|H2|47=41+6；36/41与5/6；13clean无观测误报|§5–6 pp6–8|支持；不等于总体FPR=0；功能等价与精确还原分开。|
|H3|旧年份数据40–70%非触发漏洞，修改构造|App A.1 pp12–13|支持但不全：R5补修改后Table4高FTR模型。|
|D1|WSD20特征逻辑回归，不运行目标模型|2602.15195v3 §4 pp4–5|支持部署表述；监督训练需要带标签adapter库。|
|D2|每家族独立400/100校准、50/50测试，三家族100%|§5.1–5.2 pp5–6 Table2|支持分布内留出；同两触发机制、1/3/5%污染，不是同检测器跨架构zero-shot。|
|D3|层/秩扫描、payload可读性、家族几何、适应未解|§5.4–6 pp8–9 App A.2 pp12–15|支持；默认layer21/rank16是扫描选择，不能解释为普遍因果中介。|
|Z1|Z-PEFT监督谱分类、留出zero-shot含义|2608.02271v1 §3–4 pp3–6|支持；需要已标注中毒库，不是完全无标签检测。|
|Z2|Alpaca .635、AdaLoRA .2628排序反转|Table3 p6；App Table7 p9|支持；AdaLoRA ACC .288，架构实验也不是跨家族零样本。|
|Z3|ACC依真实阳性数oracle、AUROC不同|§4 p4与Limitations p7|支持；部署阈值不能由测试真标签设定。|
|Z4|基线受预算限制，SHAP非机制|p6；App p11|支持；24h仅接近3epochs，R6细化。|
|P1|Anthropic通用对照方向、末token，多设置>99%AUROC|2024保存网页 Methods/Results/Ablations|支持；同模型同层拟合，无精确trigger拟合不等于评估无触发/行为先验。|
|P2|转移、校准、CoT和自然欺骗限制|网页 Why/Future及脚注5/6/8|支持；CoT先生成scratchpad，base→sleeper不总稳定；非训练形成前预警。|
|G1|SpecGuard可信draft、条件界、请求检测|2609.11799v1 §3–4 pp3–5|支持；R6定理定位§4；界含draft mismatch和失败攻击近似clean假设。|
|G2|topic AUROC .572、适应λ=.1 ASR .92/AUROC .557|Table2；§5.6 Table8 pp8–9|支持；后者已经是two-sided分数，故双侧并非该强负例的通解。|
|G3|共污染单侧失败，双侧仅所测缓解；短输出误报|§6.1 pp9–10；App Failure Analysis/Fig12 pp15–16|支持；bottom5% clean接受率样本中位长度2token、整体113，非误报皆由攻击造成。|
|G4|200token尾稀释，全程.54/前8约.89|§6.5 p12|支持；只涉及推理token窗口，不能直接充当训练checkpoint瞬态证据。|
|E1|EVO已训练MLLM、前K解码轨迹、净对照、过滤再微调|2610.20319v1 §3–5 pp3–7|支持；默认K5/clean64，未知trigger但有干净参照，离线审计为主。|
|E2|延迟触发K12更好，F1与净化ASR不同|§5 pp7–8|支持；R6“双目标”改“延迟双触发/双模态触发”。5token延迟ASR36.7→13.3、7token43.3→3.3是净化结果。|
|Q1|Distributed为工具文本碎片/外部组装，非权重后门|2607.24893v2 §3 pp4–6|支持；注入后提前于最终执行，不是注入前可预测。|
|Q2|2082运行分母、99.5%、提前12步及三类误报|§3.2/§5 Table3 pp5–7|支持；1101无注入+553失败+428成功；1.1/80.5/27.6%分母必须保留。|
|Q3|编码离线重写、长度和跨域限制|§3.2 p6、§6 pp9–10|支持；去length特征不等于长度匹配，AUROC不提供可移植固定阈值。|

## 3. 跨域直接回原文核读，而非接受他域结论

下列6篇D6原文来自 `../d6-generalization/sources/`，2篇D5也在本域 `sources/`。B仅核与D4有关节段；这不把A的SECTION_READ升级为FULLTEXT_READ。

|ID/原文|核读范围与A主张|判定、应补结果和边界|
|X1 Persona Vectors 2507.21509v3|§4.1–4.2 p6；§6.1 pp9–10、§6.3 pp11–12；App L pp54–56|已知trait方向、候选回答减base自然回答的投影差、最高/最低/随机500例微调排序，支持训练前风险估计已有先例；R4补极端筛选退化与toxicity过滤。不是未知风险盲预测，也没有首发时刻概率校准。|
|X2 From Data to Behavior 2602.04735v1|§2–3 pp2–4、App C.1/C.2 pp15–16|逐层末token均值注入base，α[0,8]取最佳/可维持连贯的最大值；NYC83预测/3.4训练/base75.8支持失败例；Qwen提及含CoT、安全评估200 SafeEdit攻击提示，不等同普通问答EM。“0”依阈值定义。|
|X3 Model Organisms 2506.11613v1|§4.1–4.3 pp5–7；App F pp19–21；App G.1–5 pp22–26|方向早变/5x放大较早EM成立，20x可不连贯；R1必须补t±s与整轨迹PCA。App F改变alignment/coherence阈值仍同一emergence，不能断言全是二值阈值假象。rank8/64无同样直接转动，fullFT梯度早大后衰；多adapter最终参照是另一未来信息条件。|
|X4 Homeopathic 2611.10110|App H.1–2 pp42–43；H后续方法/FigH.3 pp44–45|两个endpoint与German向量支持行为0不等于无内部变化；378prompt/已知German/层34，不是统一未知风险。R2加入同训练轨迹先升后归0，注意FT方案LR不匹配。|
|X5 Weird Generalization 2512.09742|§5.1 p9、§6 pp11–12；App F.1–2 pp54–55；App H.1 p65、H.2 Fig46及H.3/Table9 p69|US presidents 30seed中9个过>50% held-in门槛再进一步测；有平滑loss而行为跃升，不能说所有seed统一现象。年份2025/2027对照、Llama8B layer19 SAE、2048 GSM prompts、top10特征方向消融：Mossad引用80→5%。已知条件构造；Table9 temp1准确率80.79→77.98、引用0→.88%，并非绝对零能力代价。|
|X6 An Emergent Mirage 2607.09053v1|正文pp3–7；App A pp9–10|长度匹配后安全答案也能EM，单adapter出现反复局部方向转动；反驳简化解释。rank1 α256 LR2e-5与rank32/all α64 LR1e-5不同；§5.3单adapter余弦与全adapter行为图非一条完全对齐轨迹，未直接复现早期5x放大；不能整体推翻Model Organisms。|
|X7 TopKLoRA 2610.32866v1|§6 pp6–8、§7 p9|已知trigger/对照对用于定位并消融，针对新训练稀疏adapter；多层保效好不代表单层或未知trigger通用。不构成自然先兆，边界可短纳入。|
|X8 HARVEY 2607.05748v1|§2–3 pp2–4|参考模型用RCE等主动学习疑似毒并忘干净、bottom50%初始化、重启迭代，最后筛选训练；不需已知trigger/外部净集，但制造分离而非观察未改变自然动力学。这里只核方法边界，未独立核D5的全部效果数值。|

## 4. 综合判断与信息先验核对

- **训练前数据风险**：Persona/Data to Behavior确实是已有正面先例；要已知trait/任务评价和方向/注入条件，并有失败及质量副作用。不能笼统说训练前风险预测没人做过。
- **训练期动态/隔离**：ABL/BaDLoss/BadWindtunnel/TimeGuard/HARVEY已使用loss或置信动态；有的主动改变训练，BaDLoss要干净锚点和高删除率。它们未证明同一信号在自然训练中先于风险首次出现。
- **训练内“先变”/瞬态**：Model Organisms及Homeopathic/Weird使“内部先变、外部骤变、终点掩盖瞬态”本身不能作为新颖性保证；未来窗口、读出放大、seed筛选和任务条件必须明示。机制仍不能从先后关系单独得到。
- **训练后检测**：WSD/Z-PEFT用已标注清洁/中毒adapter库；WSD每家族重训，Z-PEFT ACC还要测试阳性数量；ToxScreen候选含真trigger；Haystack未知精确字符串但知道评价任务族且部分模型高FTR。Spectral要污染上界/标签，AC要可疑训练集。
- **推理早期/最终执行前**：probes/SpecGuard/EVO/Distributed的时间轴不能移植为训练形成前。其中SpecGuard要可信/校准draft，EVO要净参照；Distributed负类含攻击失败时误报很高。
- **与项目比较**：A所写“指定输入/历史的因果作用、有限更新响应—行为关系≠未知未来在线预警”与共享参照一致。文献覆盖这些广义观察不等于已经复现项目具体因果链；反过来，未检到完全相同设定也不支持“全部新颖”。此处为证据边界核对，不新增实验或方向建议。

## 5. 未核/未完成与阅读诚实记录

上表记录B亲自核读的章节、表、附录；不是宣称B逐字重读每篇全部正文。15核心条目的所有在报告中影响方法、结果和重合的实质主张都映射上表。无关参考文献页未作为读完依据。PDF文本能清楚还原公式/数字的地方按原文文本核；对跨页两栏按节标题/表题定位，未从模糊图形估计新精确数值。

两综述2406.06852v5、2502.05224v1在A为SECTION_READ、用于检索/分类；本B未重复把它们当一手实验依据，也未将其全文核验完成。DFBScanner2605.18907、代码谱2510.13992、PSSP、IEEE MRAI、SSRN7113514、InfoFusion104308仍不在本次核心全文结果核验之列。A目前对它们保留筛选/元数据/受限状态，正确；不能在最终交付中把这些重要线索自动算处理完成。TopK/HARVEY新增的上述局部核验不替代其主域全文审查。

ResearchGate《Testing Whether Internal Backdoor Precursors Precede Behavior: A Causal, Power-Matched Protocol》本B尚未取得可核权威全文；**其smoke/无confirmatory trajectories/成熟参照等目前只作为检索返回内容，非B已核权威论文事实。** 不能用于肯定或否定新颖性。C需查作者/版本/权威公开入口；如无法完成，明确受限，不捏造出处。

补查记录（2026-11-19）：用准确题名及`"rs-10594749" "Yuchen"`搜索，[ResearchGate索引](https://www.researchgate.net/publication/411771060_Testing_Whether_Internal_Backdoor_Precursors_Precede_Behavior_A_Causal_Power-Matched_Protocol)列作者Zuo Yuchen、日期2026-08-06及候选[Research Square DOI](https://doi.org/10.21203/rs.3.rs-10594749/v1)。索引返回的§3.6、§4.4、§5分别描述成熟discovery方向、没有确认性轨迹、非未知trigger部署监测；但web访问DOI和[Research Square入口](https://www.researchsquare.com/article/rs-10594749/v1)均报工具错误，故不提升FULLTEXT_READ或认定书目信息已独立确认。该索引还引出需A/C追踪的SEEP：Training Dynamics Grounds Latent Representation Search for Mitigating Backdoor Poisoning Attacks（TACL2024，[10.1162/tacl_a_00684](https://doi.org/10.1162/tacl_a_00684)，[ACL入口](https://aclanthology.org/2024.tacl-1.55/)）；当前仅为重要遗漏候选，B尚未阅读全文、不据此作效果结论。

## 6. 复核入口

A完成R1–R8并更新相应阅读状态/修订记录后，本B只复核改动及新增实质主张，不无故重跑全域。R7可用删去无出处笼统“表间矛盾”解决；其余重要结果须进入正文而非只留交接文字。C独立查漏之后的新增关键证据仍须A原文整合→本B核验；当前没有“通过”或“全新颖”结论。
