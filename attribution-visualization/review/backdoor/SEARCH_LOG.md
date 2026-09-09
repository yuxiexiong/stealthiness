# Backdoor 域检索日志

日期与截止：2026-09-09。范围与纳排规则来自 `review/PROTOCOL.md`；本域只做公开全文研究，不执行论文代码。所有搜索的总命中数均为 `not_exposed`。

## 种子与版本检索

| 编号 | 实际查询 / 入口 | 发现与处理 |
| --- | --- | --- |
| B01 | `BkdAttr backdoor attribution language models` | BkdAttr arXiv:2509.21761，纳入；Defending against Insertion-based Textual Backdoor Attacks via Attribution arXiv:2305.02394，直接方法前驱待查。 |
| B02 | `"When Backdoors Speak"`；`"When Backdoors Speak" arxiv` | arXiv:2411.12701；ACL 2025 2025.acl-long.114，同一 work 合并。密码学同名前缀论文与本项目无关，排除。 |
| B03 | `"Grond" backdoor arxiv`；`"Grond" "arxiv.org/abs"`；`"Grond" "Stealth" backdoor parameter` | Towards Backdoor Stealthiness in Model Parameter Space，2501.05928v3；天文 GROND 等同名噪声排除。 |
| B04 | `"Obfuscated Activations Bypass" arxiv 2412.09565` | arXiv v2 和 ICLR2026 OpenReview；需区分版本。 |
| B05 | 官方 arXiv abs/2509.21761、abs/2411.12701、abs/2412.09565 | 最新 arXiv 分别为 v2 (2025-09-30)、v3 (2025-02-16)、v2 (2025-02-08)。保存所读 PDF 自带版本，不将抓取日期当发表日期。 |
| B06 | `"double triangle" "backdoor" paper` | Microsoft 作者博客及 The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers，2602.03085，纳入。博客仅用于发现和定位原论文。 |
| B07 | `"Obfuscated Activations" "2026" "attribution"` | SAE post-intervention recovery 代码说明是新近候选，待核原文；非论文论坛总结不作证据。 |
| B08 | `"Backdoor Attribution" 2026 attention mechanism` | OpenReview ICLR2026 在审稿，与 arXiv 同一 work；第三方 ICML2026 声称未经官方确认，不采用。 |

## 获取记录

- 复制已有公开 OA PDF `papers/OA_Obfuscated_Activations.pdf` 至本域 `pdfs/OA_local.pdf`：51 页，页眉确认 arXiv v2。
- 复制用户指定 Grond PDF `/Users/ruizhixu/Documents/ChatGPT/poison-r0/outputs/stealth-baseline-pdfs-2026-09-08/PDFs/01_Grond.pdf` 至本域 `pdfs/Grond_v3.pdf`：20 页。
- arXiv PDF 2509.21761v2、2411.12701v3、2602.03085 由公开入口下载；首次沙箱 DNS 失败，经正常权限升级后成功。
- OA ICLR2026 官方论坛和 PDF 入口遭遇 OpenReview 浏览器验证；curl 返回 403。停止该入口自动尝试，不绕过验证。arXiv v2 可全文阅读，会议版差异暂列 open_issue。

## 扩展与停止条件

进行中；未声明覆盖收敛。全文的前向参考文献与后向较新文献扩展将在下方逐条登记。

## 第二批检索与获取（2026-09-09）

| 编号 | 实际查询/入口 | 返回候选与处理 |
| --- | --- | --- |
| B09 | `"Defending against Insertion-based Textual Backdoor Attacks via Attribution"` | AttDef正式ACL2023全文纳入。 |
| B10 | `"When Backdoors Speak" "2026" attention` | 2026机制/检测邻居候选；继续官方原文定位，第三方摘要不作结论。 |
| B11 | `"LLM" "backdoor" "causal tracing" interpretation 2026` | BkdAttr等已知work与机制候选；不把普通因果图术语的backdoor adjustment混为投毒后门。 |
| B12 | `"LLM" "backdoor" "visualization" "attribution" 2026` | Haystack与Associative vs. Distributional（CodeGen）直接纳入候选。 |
| B13 | `"Learning to Deceive with Attention-Based Explanations" pdf` | Pruthi ACL2020原文纳入；2021复现报告只初筛，未借其结论主张复现成立。 |
| B14 | `"A Study of the Attention Abnormality in Trojaned BERTs" pdf` | NAACL2022原文纳入。 |
| B15 | `"BAIT" "Large Language Model Backdoor Scanning" pdf` | 作者S&P2025 PDF与仓库补充纳入。 |
| B16 | `"Associative vs. Distributional" backdoor` | Preprints.org v1；页面指向JCP正式DOI。正文HTML可读，但核心Table6图片不可得，保留partial。 |

- ACL三篇（AttDef、Pruthi、TrojanedBERTs）通过各自官方PDF入口成功获取；BAIT由作者Purdue主页获取19页，官方仓库补充2页。WBS正式ACL2025与arXiv v3进行完整差分阅读，记录在`texts/WBS_VERSION_DIFF.txt`。
- CodeGen DOI10.3390/jcp6050146、MDPI公开HTML`https://www.mdpi.com/2624-800X/6/5/146`及`/pdf`、Preprints.org公开PDF/HTML均正常尝试；web返回429或图片cache miss，curl403。未绕过；保存可读官方HTML文本JSON，不冒充PDF全文或图像核验。
- 初筛外围候选ConfGuard(2508.01365)、Exposing the Ghost(2504.00446)、Probe Before You Talk(2506.16447)：检测置信度/表征分布方向，暂列待范围判定；没有用这些未读论文作候选缺口的反证或支持。视觉DAA、一般检测器/综述不自动全文纳入，须明确是否直接解释token依赖。SAE post-intervention候选交foundations联查。

## 参考文献与2026前向扩展第一轮（仍未收敛）

| 编号 | 实际查询/来源 | 发现与处理 |
| --- | --- | --- |
| B17 | `"Backdoor Attribution" "2026" language models` | 新OpenReview版本、第三方声称ICML2026、Geoffrey Voyer一手复核BkdAttr的2026-02-24文章；版本与复核均继续核验。P2P(ACL2026)是防御方法候选，需判断是否有直接机制归因。 |
| B18 | `"When Backdoors Speak" "attention" "2026"` | Unmasking Backdoors: Explainable Defense via Gradient-Attention，ICLR2026官方poster，直接纳入；Unreal Thinking仅CoT攻击候选，未作解释机制证据。 |
| B19 | `"Obfuscated Activations" "attribution" "2026"` | Where Did It Go Wrong? Representation Gradient Tracing，ICLR2026，涉及样本及token归因，直接纳入。 |
| B20 | `"BAIT" "backdoor" "2026" explanation` | 同名BAIT jailbreak2026不同work，排除；LLM Firewall综述式使用BAIT，非新归因方法，排除；CodeGen重复work。 |
| B21 | BkdAttr OpenReview正文相关工作；`"Lamparth" "Reuel" backdoor 2024 interpretability` | Analyzing And Editing Inner Mechanisms of Backdoored Language Models，FAccT2024正式12页，直接纳入，不能只按BkdAttr概括当toy-only。 |
| B22 | `"Baker" "Babu-Saheer" backdoor attention 2025` | Mechanistic Exploration...2508.15847，13页，直接纳入；连带发现Language Triggers Hijack Language Circuits，直接纳入。 |
| B23 | `"Unmasking Backdoors" "Gradient-Attention"` | arXiv2510.04347，正常获取17页；ICLR2026公开OpenReview PDF403，会议差异保留。 |
| B24 | `"Representation Gradient Tracing" "MN1qlAVJLV"`；`"Where Did It Go Wrong" "Representation" arxiv` | arXiv2510.02334，16页；正式OpenReview403，保留版本边界。 |
| B25 | `"Language Triggers Hijack Language Circuits" arxiv` | 2602.10382最新版v3、2026-07-20，19页ICML2026 MI Workshop；前版本标题/作者/0.66结果与v3不同，按所读v3记录。 |
| B26 | `"Unmasking Backdoors" 2510.04347 github` | 官方XGRAAD仓库与论文互证；未执行代码。 |

新五篇均已下载并登记`downloaded_unread`后逐篇阅读，不以获取成功替代全文阅读。OpenReview新增四个正常PDF请求均403，arXiv和FAccT公开版本成功。此轮出现未处理直接近邻，**停止条件未满足**。下一轮在处理这些全文及其直接参考链后再执行，不报告全球完整或当前新颖性已确定。

前驱链核对：Pruthi→Jain/Serrano/Wiegreffe交foundations；TrojanedBERTs→attention IG/AttDef相邻归因证据；Haystack→BAIT输出条件依赖已全文；OA→Grond仅问题类比，不主张历史延伸；BkdAttr→WBS的概括过窄（WBS不仅自解释），本域已按原文纠正。

## 2026继续扩展与争议链（第二次发现轮，仍未收敛）

| 编号 | 实际查询/入口 | 结果和处理 |
| --- | --- | --- |
| B27 | `"Mechanistic Exploration of Backdoored" "2026"` | Baker已读；Fuzzing Large Language Models to Elicit Hidden Behaviours 2606.29646作为行为扫描外围初筛，未作机制结论。 |
| B28 | `"Language Triggers Hijack Language Circuits" "2026" attribution` | 同一LanguageCircuits新版，已读；论坛复述不计新论文。 |
| B29 | `"Unmasking Backdoors" "2026" "attention" attribution` | XGRAAD重复；Patcher新直接近邻，不能以修复主题为由排除归因方法。 |
| B30 | `"Obfuscated Activations" "2026" "interpretability" backdoor` | Neural Chameleons、RL-Obfuscation、The Obfuscation Atlas、Trojan-Speak进入官方摘要筛查；SPAR课题提案不是已完成研究，不用于done。普通博客/综述只作发现源。 |
| B31 | `Patcher Post-Hoc Patching Backdoored Large Language Models USENIX 2026 pdf`；官方USENIX presentation/gao-anjun | 官方论文pp221–240与arXiv2606.02995v2；全版24p及正式21p含封面下载，完整差分阅读。 |
| B32 | `"The Obfuscation Atlas" "2602.15515"`；`"Trojan-Speak" "2603.29038"` | 找到官方arXiv与若干第三方引用；第三方ICML奖项说法不借用。Out of Context Obfuscation、Monitoring the Internal Monologue由搜索命中列为摘要待筛，不立即当直接归因结果。 |
| B33 | 官方arXiv abs/2512.11949、abs/2506.14261、abs/2602.15515、abs/2603.29038 | 前三分别明确有子空间机制、token重表征、表示漂移/策略变化诊断，必须全文，下载30/24/30p；Trojan-Speak摘要侧重输出通信/内容分类器规避及activation监测，作为外围，不使用攻击细节或摘要结果支撑机制解释。 |
| B34 | generation转交ACL2026 `2026.acl-long.220`官方页与PDF | Revitalizing39p已全文并核图，重点区分ICL污染示例与训练后门；Focus-LIME/REX/Beyond Attribution前驱交generation核范围。 |
| B35 | Geoffrey Voyer作者公开博客 `why-attack-success-rate-gives-a-false-picture-of-backdoor` 全文及附录 | 非同行评审复核；所有文本读完、图9/10官方嵌图web cache miss，partial；记录BASR、judge独立性、表格计数问题，不升格为同行评审否证。 |

- Patcher参考链：TracLLM/AttnTrace/RAGOrigin/RAGForensics精确书目交generation官方初筛；Poison Forensics训练样本来源定位属外围。不能因Patcher Table22称其为语义高亮就认定原方法只测相关性，粒度适配未清是本论文比较边界。
- 新三篇OA机制工作处于`downloaded_unread`，不会用摘要支撑done/reuse/gap。停止条件未满足。CodeGen图像/版本获取受限与Voyer核心图像缺失保留，不为凑“关闭”降格协议。

### B36–B39 后续直接近邻核对（2026-09-09，非closure）
- B36真实查询：`"Out of Context Obfuscation" "What Facts"`；`"Monitoring the Internal Monologue" "Probe Trajectories"`；`"Needle-in-RAG" "RAGCharacter"`。前者SPAR作者研究demo与OpenReview PjOwhIzR2z，forum/pdf正常入口均browser challenge，未绕过，范围/全文未闭合；Monitoring找到arxiv2605.18549，官方摘要逐生成token概念轨迹，root已交visual全文；Needle2605.01782v1明确counterfactual masking/replay定位字符因果span，必须全文而非按RAG关键词排除。
- B37 root给出精确候选，打开官方arxiv2509.13772、2504.21668、2606.25721；随后打开各官方摘要方法。RAGOrigin2509.13772v2包含generation influence数值责任归因，纳入；TRACE2606.25721v2是token influence＋secondary verification，纳入；RAGForensics2504.21668v2是LLM prompted语义支持判断，继续打开官方HTML正文§4/Alg1实际核范围，归因并非通过受审模型的logprob/gradient/counterfactual测得，记外围semantic corroboration，不因Patcher把它改成token基线就宣称它是模型token归因算法。完整HTML未全文读取，实验结论不作本项目证据。
- B38真实版本查询：`site:proceedings.mlr.press "The Obfuscation Atlas"`无命中。arxivv2正文刊头ICML/PMLR306，官方摘要oral已核；不能标为独立publisher版本比对完成。
- B39正常下载RAGOrigin20p、TRACE19p、Needle36p并验证PDF页数及可提取文本；75页新全文队列。检索原始结果保存`texts/RAG_NEW_SCOPE_WEB.txt`、`RAG_NEW_SCOPE_DETAIL_WEB.txt`、`RAGFORENSICS_SCOPE_WEB.txt`。此次确有新直接论文，不能宣称两轮零新增收敛。

### B40–B45 范围收束与最后新项（尚非两轮closure）
- B40真实查询：`"Mechanistic Anomaly Detection via Functional Attribution"`；`"VKekM8kJ9U"`；`"Discovering Backdoor Triggers"`。MAD官方arxiv2604.18970v2为ICML2026 camera-ready，明确functional influence/OA，root接全文；Discovering是Qin/Hua等2025-08-19一手研究报告，非Haystack同文，纳入文本与图核队列。原始结果`texts/SCOPE40_WEB.txt`。
- B41打开VKek官方forum正常challenge；MAD官方abs核到v2 2026-05-25；`texts/SCOPE41_WEB.txt`。不是权限绕过，不以challenge当内容。
- B42真实查询：`"VKekM8kJ9U" "Unmasking"`、`"PromptLocate" Jia Gong`、`"Out of Context Obfuscation" "What Facts"`。作者官方CV明确VKek=已全文XGRAAD，不重复纳入。PromptLocate2510.12252进方法scope核；OOC搜出官方静态PDF729837dfce7b9aa832920e939c96ec8d228f9818.pdf，但实际打开仍challenge，不能把搜索返回段落冒充全文。`texts/SCOPE42_WEB.txt`。
- B43/B44打开官方PromptLocate HTML§IV与三个此前pending候选官方abs。PromptLocate用片段分类oracle+GPT2对剩余**输入**字符串的CIS，不测victim实际输出token因果效应，故本项目外围；ConfGuard2508.01365v3为生成confidence窗口sequence-lock检测，BEAT2506.16447v1为固定probe拒答分布对拼接输入的变化，均未做原回答词归因/内部路径，保留外围测量对照，不扩所有检测器。Exposing the Ghost2504.00446v2摘要明确layer hidden-state anomaly分类，缺词级归因/因果图目标，外围。此处只scope初筛，未引用其性能作为已核证结论。`texts/SCOPE43_WEB.txt`与`SCOPE44_WEB.txt`。
- B45全文参考链：RAGOrigin→ContextCite/AttriBoT由root全文；TRACE→Token Highlighter在项目其他域全文；Needle→Context Influence（ACL2025 long1220）交generation官方scope筛，RAGOrigin已本域全文，普通检索防御不扩；Monitoring两个动态probe参考（Beyond Linear Probes/Streaming Hallucination Detection）交visual在本域参考链范围初筛。本文所有RAG新项已正文附录+核心图核读；继续Discovering后才计关闭查询。

### B46–B52 最后直接近邻处理与冻结（2026-09-09）

- B46实际查询：`"Finding Features Causally Upstream of Refusal"`、`"Discovering Backdoor Triggers" "2026"`、`"Out of Context Obfuscation" Soylemez pdf`。返回LOCA2605.00123、Lee等上游拒答特征作者报告、Discovering原作者报告与OOC访问入口。原始返回`texts/SCOPE46_WEB.txt`。这轮有新直接近邻，**不是零新增**。LOCA官方arXiv确认v3/COLM2026后下载21页，正文附录A–L和全部12图已读，完整卡落地。Lee等2025探索性上游拒答报告只保留为一般拒答特征前驱发现线索，不以它的摘要/搜索节选支撑done；最终不递归扩张全部拒答/SAE谱系。
- B47获取与补缺：Discovering作者报告正文/参考文献完整读，原图经公开Cloudinary普通入口获取并核读；Voyer2026图9/10亦从公开原图入口取到，已由partial更新full_text_read，保留原始数据/独立judge疑点而不保留已解决的图缺口。OOC官方forum PjOwhIzR2z及静态PDF `https://openreview.net/pdf/729837dfce7b9aa832920e939c96ec8d228f9818.pdf` 正常访问仍403/验证；停止，不绕过，登记fulltext_unavailable。
- B48实际查询`"Auditing Language Models for Hidden Objectives"`，并打开LOCA引用的三个ACL官方页面：`2025.blackboxnlp-1.28`、`2025.findings-emnlp.338`、`2026.eacl-long.12`。真实返回`texts/FREEZE_SCOPE1.txt`，接着在原官方页定位Abstract，保存`FREEZE_SCOPE2.txt`。这是精确候选筛查，不是全领域新一轮普查。
- B49进一步打开Yeo/Kirch官方PDF与Auditing官方arXiv HTML的相关方法段，保存`FREEZE_SCOPE3.txt`和`FREEZE_SCOPE4.txt`。**Yeo等 Understanding Refusal in Language Models with Sparse Autoencoders**有输入词→内部特征→行为图与干预闭环，是直接邻居，官方23页全文及附录/16图/12表均读，纳入。**Kirch等 What Features in Prompts Jailbreak LLMs?**官方41页PDF的§4.1/4.4范围核对是末prompt位置的latent probe，统一方向干预所有位置/生成步；按最后固定范围保留为全局probe/拒答前驱外围，非逐原输出词归因，未全文读，不援引其性能。**Ball等 Understanding Jailbreak Success: A Study of Latent Space Dynamics in Large Language Models**官方EACL2026 long12，摘要范围为跨攻击族全局jailbreak方向及harmfulness投影，外围；不因标题Dynamics就推断跨输出步传播。它也未全文读，不据摘要宣称已否定/证成项目。
- B50 **Auditing Language Models for Hidden Objectives**官方2503.10965v2，方法范围可见completion条件归因与交互审计，不能排作一般probe；由root全文，计foundations F31，本域不重复登记。TracLLM/AttnTrace/MAD同样由root全文；Monitoring由visual全文；Context Influence/原始生成归因基础由其他域处理。跨域分工不计未读完成。
- B51 root从Auditing完整参考链定位**Sparse Feature Circuits**2403.19647，指定唯一剩余直接前驱。打开官方abs确认v3 2025-03-27/ICLR2025后普通公开PDF下载；36页正文、参考文献、附录A–H与Figs1–25/Tables1–7全部读，核心SHIFT偏移验证/图证据写卡。此项有真正图→认知→特征编辑→同任务分布变化评估，不能以泛SAE名义排除；依最后要求不递归它的一般电路搜索参考文献。
- B52 **冻结**：本域29条独立work记录，27条full_text_read（其中2条是一手作者报告，非同行评审）、1条partial_fulltext_read（CodeGen）、1条fulltext_unavailable（OOC）；已获取的直接队列无downloaded_unread。所有状态与结论以对应版本为限。最后进行一次记录字段/工件存在性检查，复用先前页数与PDF验证结果，不重解析已读PDF、不重新跑检索。明确依据协议末尾“收尾修订”和用户“不重复检查”要求停止；本域**不声称连续两轮零新增，也不声称穷尽式系统综述或全球无遗漏**。这是处理完当前发现队列后的文献快照冻结。

检索入口未暴露统一总命中数，记 `not_exposed`。上文各阶段的“待全文/未收敛”保留为历史进度；最终状态以`PAPERS.json`和`FULLTEXT_NOTES.md`为准，域总览由主报告统一呈现。CodeGen图表缺口、OOC原文缺口、会议版本差异、未运行代码及原文实质疑点均不会因冻结而消失。
