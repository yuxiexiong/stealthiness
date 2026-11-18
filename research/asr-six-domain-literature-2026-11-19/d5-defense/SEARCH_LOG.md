# D5 A 检索与引用链日志

实际检索/阅读日期：2026-11-19（Asia/Shanghai）。范围：防形成、训练干预、参数/表示修复、低ASR与耐恢复的区分。未检索私有项目文本；只向公开工具发送一般学术词。仅写D5目录。

## 工具与资料

已读 `nature-academic-search` 和 `nature-downloader` 技能、前者工具路由/多源搜索说明。可用专业学术MCP为空；使用web检索、官方ACL/arXiv页面、合法公开PDF。没有假定CrossRef/OpenAlex/Semantic Scholar前向索引完整可用；父域共享日志的429障碍不冒充本域成功查库。shell公开下载采用require_escalated，未遇本域自动审批拒绝。PDF经 `/opt/homebrew/bin/pdftotext` 抽取，必要处用pdftoppm核图。

先读PROTOCOL、PROJECT_COMPARISON、CROSS_DOMAIN、ROOT_SEARCH_LOG。继承种子仅作线索；正文均独立读取，不以其他域报告替代。

## 入口与实际查询

| 顺序 | 入口/查询 | 处理结果 |
|---|---|---|
| 1 | web `site.aclanthology.org/2026 "backdoor" "defense"` | 核Locphylax、Critical-CoT、P2P revision2、MB等官方记录；SLIP为黑盒prompt安全邻域，未纳核心 |
| 2 | web `"backdoor" "BIBAU"` | 对照本地2606.14078官方全文，采用BI-BAU正文而不只摘要 |
| 3 | web `"Suppression Is Not Erasure"` | 核2611.12807v2，11月15日新版，全文含恢复/utility附录 |
| 4 | web `"backdoor" "survey" "defense" 2026 large language models` | 获2502.05224综述；新增Dummy2606.11648v2及Fragmented2608.30403，均转全文；2605.25073生命周期线索未转核心 |
| 5 | 直接官方ACL页面 `2026.findings-aacl.248`、`2026.findings-acl.815` | SelfHeal、DiSec元数据与全文；下载后读全部正文/附录 |
| 6 | arXiv abs `2606.07963`、`2608.00732`、`2608.30403`、`2606.11648` | SharedLatent/TR/Fragmented/Dummy，保存并核版本 |
| 7 | arXiv abs `2507.16795`、`2611.15479`；ACL `2026.acl-long.495` | 原CAFTv2是2025-11-09；DETER是2026-11-11；Critical-CoT官方主稿 |
| 8 | web `"backdoor" "defense" "November" "2026" arxiv` | 最新补查出现GARD2611.14143、SURA2611.13491；前者仅官方摘要，后者联邦客户端检测邻域，不作为D5机制结论 |
| 9 | web `"Denoised Product of Experts" backdoor` | 2305.14910及2024.naacl-long.27；下载正式NAACL全文，避免只用早版摘要的SST2说法，正式稿有3任务/Llama2 |
| 10 | web `"Shared Latent Structures" backdoor` | 回核arXiv，未找到可确认的新独立前向改进；聚合站只作定位、不作证据 |
| 11 | web `"Backdoor Threats" "Large Language Models" 2502.05224` | 核综述身份；额外综述2409.19993只作候选，未宣称已读 |
| 12 | web `"From Shortcuts to Triggers" "2026" defense` | 作者/引用线索，无可确认新增必须全文的直接后继；TR原文已作为相关后继读取 |
| 13 | web `"GARD" "Gradient-Guided Adversarial" arxiv` | 官方2611.14143摘要确认，记录CLIP邻域覆盖限制，不转性能结论 |
| 14 | web `"backdoor" "2026" "unlearning" "November" arxiv` | 再核Suppression与SURA；泛化隐私遗忘/知识编辑条目不与触发后门混为一谈 |

并非每次搜索结果都可信。GitHub论文列表、news/radar、alphaxiv等只辅助找到原文；最终核心结论只来自保存的一手正文。未用搜索摘要的数字支撑报告。

## 后向与前向链的实际处理

1. **ABL→训练隔离/分流→HARVEY、DPoE、TR**：ABL原文（复用D1来源后独立读）；HARVEY原文明确对照ABL/DBD/CBD并以偏后门oracle筛毒；DPoE正式原文独立读，连接自然bias PoE与backdoor；TR最新分支解耦全文。DBD/CBD、Honeypot2310.18633及Trap-and-Replace2022在原文参考链和综述中识别，但未将二手描述当一手全文已核。
2. **传统adversarial unlearning→BI-BAU→恢复审计**：全文BI-BAU及Suppression；I-BAU/ANP/NAD原始支系未单独核，报告不声称验证其原论文结论。Suppression引用LoRA扫描/投影系列（LoRAScan、NEEDLE、LoRAPatch等）仅通过该文的具体重新评估理解，不回溯宣称原作者主张。
3. **已知防御触发→未知攻击缓解**：P2P（特别采用revision2）、Locphylax、MB、Dummy全部正文和相关附录。比较推理安全key是否仍在、攻击任务是否必须知道、是否存在跨目标反证、是否评过再学习。未把这些不同威胁模型合成同一方法。
4. **CAFT→SharedLatent；SAE差分→Fragmented**：原CAFT2507.16795v2和SharedLatent2606.07963均独立读，回核2025/2026时间；Fragmented为新找到的差分/夹值方案，全文核并保留其表示分解近似/选择规则限制。
5. **局部电路干预→TopKLoRA**：复用D4下载，但独立读正文、planted ground-truth、clean twins、消融顺序、heldout评估等附录，不只看标题的faithful circuit。
6. **训练期抑制与耐恢复→DETER**：由D6/父域转入，完整正文及附录，严格定位为非后门知识能力邻域；SGTM/TAR为DETER未直接对比的链，不填补成实测胜出。
7. **综述→覆盖反查**：2502.05224只读§4–6/refs，识别SelfHeal以外的数据清洁、BEEAR/Fine-mixing/MuScleLoRA等未核支系。它们留作明确范围限制而非“未发现”。

前向发展通过关键词后继、作者页面和新论文参考回指完成，**没有获得穷尽且可靠的citation graph**。时间很新的预印本尤其不能用低引用当不重要或没人做。

## 版本与获取记录

- `sources/download_manifest.json`保留主要批量下载URL/状态；额外原CAFT和DPoE由单独直接下载取得，补充记录见下。
- 原CAFT：`https://arxiv.org/pdf/2507.16795v2`，保存`2507.16795v2.pdf`，全文43页，读取范围记录在笔记；不可误作2026首次提出。
- DPoE：`https://aclanthology.org/2024.naacl-long.27.pdf`，保存正式14页；正文1–9，相关附录13–14。
- P2P使用下载文件首页明示 `revision 2 / 10 Nov 2026`；索引stable id仍`2026.findings-acl.600`，不重复算另一篇。
- Dummy使用v2（2026-10-24）；Suppression使用v2（2026-11-15）；所有最终PDF的sha256在PAPERS.json。
- HARVEY上传2026而页脚2025：只确认arXiv身份，不猜正式会议日期。BI-BAU PDF有未完成DOI样式：索引使用arXiv ID，不发明DOI。
- 原图核TR p17、MB p9、SelfHeal p14，PNG与PDF同目录。其余正文数值以文本表格/上下文交叉读；不可靠汇总数字不采用。

## 结束状态

18篇一手正文+相关附录已实际阅读；1篇综述部分阅读；GARD仅摘要筛选。没有待读全文冒充FULLTEXT_READ。当前A稿可供B逐项复核，但未达协议的整个D5关闭条件：B未核、C未独立查漏，经典原始分支未全部回溯，最新CLIP邻域未全文核。所有这些在REPORT中明示，不能据此作“世界上零遗漏”或本项目创新保证。

## B 后修订轮（2026-11-19，取代上节首稿结束状态）

首轮B核原18篇/36组后要求针对性补链，本轮用B清单及各原文参考链作入口。新增下载的准确URL/状态/字节见`sources/revision_download_manifest.json`：QES2610.00663v2、Phantom2602.04899v3、Honeypot2310.18633、DBD2202.03423、I-BAU2110.03735（实际文件v4）、Fine-mixing2210.09545、BEEAR2406.17092、MuScleLoRA2402.12026v3、NAD2101.05930（实际v2）、SafetyEfficacy2602.00822。均用公开arXiv PDF，shell经require_escalated，无登录或私有资料。QES/Phantom版本日期回官方元数据；其他版本亦核实际PDF页眉。I-BAU按ICLR2022 v4列年而非仅2021首发。

OPD2610.07654和SP24Orthogonality合法公开PDF复用D1 c_sources，仍在D5独立读正文/相关附录；BAD-BOOM复用D2后仅核防御耐久段；Cordyceps复用D3后仅核§5.3及设置附录。Phantom在本域主读后通知主审转D6相关部分，不互相假设读过。

### 实际新增检索和范围分诊

- B绑定来源直接回原文：Honeypot→DPoE/TR的文本浅支路；DBD→HARVEY/TR的解耦；I-BAU→BI-BAU；Fine-mixing/BEEAR→参数/embedding修复；NAD→蒸馏与时间边界。这些从“未读支系”提升全文，不再用综述替代。
- MuScleLoRA先看方法确定其多尺度低秩与clean梯度引导独立相关，升为全文。默认pdftotext输出含图中文字控制字符造成假分页，pdfinfo确认19页，逐页重新抽取保存`2402.12026v3.pages.json`，不误报31页。
- 最新转入QES与OPD分别补持续门控和安全蒸馏训练防御。Phantom采用2026-10-07 v3（不能停在旧稿的同模型结果），正文全部及与防御/跨模型/密码相关附录已读。
- SafetyEfficacy2602.00822：B先前官方abs访问失败不是不可得证明，本轮公开PDF成功，正文/附录均读，记录理论条件及长训反例，不直接采纳摘要免疫主张。
- 新web查询：`"T&R" backdoor defense 2022 train`、`"CBD" backdoor defense causality`、`"Adversarial Neuron Pruning" backdoor arxiv`、`arxiv 2611.14143 GARD`。回源而非引用聚合站：T&R NeurIPS2022官方摘要/PDF、ANP NeurIPS2021官方摘要/PDF、CBD arXiv2303.06818官方摘要、GARD arXiv2611.14143v1官方HTML §§1–4/5.1。
- T&R正确身份是Trap and Replace、与TR2026不同；CBD正确身份是CVPR2023 Deconfounded Representation，排除同缩写Certified Backdoor Detector。ANP/T&R仅实看首两页，未全文/未引用性能。GARD从摘要提升方法范围阅读，不采用其摘要中的成功率；其CLIP表示修复不改变已有的宽泛重合结论，保留外围。排除理由详REVISIONS，不是“下载失败所以略过”。
- BAD-BOOM最新D2全文后，D5定点核发现常规SFT本身也有高残余基线；Cordyceps防御实际在v2§5.3、且早停按推理长度，并非完整默认epoch。这两处已补主报告。

### 本轮结束状态

**30篇一手正文全部+相关附录；6项部分阅读（综述、GARD、BAD-BOOM、Cordyceps、ANP、T&R）；1项摘要身份定位（CBD）。** 37条稳定ID，版本不重复计数。全部记录可追到PDF/hash或官方web URL。全文读数不含其他域主读但本域只定点看的论文；没有新增模型实验、GPU、代码复现、论文正文改写或发布。原18篇更正及新增12篇仍待原B复核，然后C独立查漏；不签署D5或六域完成。
