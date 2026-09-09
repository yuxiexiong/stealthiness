# Visual 域检索日志

日期：2026-09-09。范围按 ../PROTOCOL.md；来源优先官方arXiv、ACL Anthology、会议/出版方和作者公开PDF。总命中数：各查询搜索引擎均 `not_exposed`，不虚构筛选总量。当前仍在进行全文阅读与扩展，**未宣告两轮收敛**。

## 实际主题检索（按执行顺序）

| 批次 | 真实查询 | 发现/处理 |
|---|---|---|
| 1 | `LSTMVis RNNVis Seq2Seq Vis visual analysis recurrent neural networks paper`; `BertViz exBERT Inseq visualizing transformers paper`; `large language model attribution visual analytics prompt comparison 2024 2025 2026` | 经典6系统纳入；Inseq交generation域；近期VA候选 |
| 2 | `"LLM" "attribution" "visual" "Megan"`; `"RNNVis" paper pdf`; `"BertViz" "2019" acl`; `"LLM" "visual analytics" "attribution"` | RNNVis1710.10777，BertViz，LLMAttributor，LayerFlow；InfoCIR因图像检索排除 |
| 3 | `site.arxiv.org "visual" "prompt" "comparison" language models`; `site.aclanthology.org "Language Interpretability Tool" generation`; `site.arxiv.org "visual analytics" "large language models" attention`; `"LLM Attributor" "Prompt" visualization 2025 2026` | LIT、ChainForge、LLMComparator、NeMoInspector纳入 |
| 4 | `"visual analytics" "language models" "interpretability" 2025 2026`; `"visual" "prompt" "attribution" language model`（限arxiv/ACL）；`"LLM Comparator" "LLM Attributor" paper`; `"Prompt" "saliency" visualization language model` | SequenceSalience；PromptExp/Sensitivity交generation |
| 5 | `"Prompt Debugging" "Sequence Salience" paper ACL 2024`; `"A Multiscale Visualization" Transformers Vig 2019`; `"LLM Attributor: Interactive" arxiv`; `"Prompt" "visual analytics" "attention" LLM 2025` | SequenceSalience2404.07498、BertViz1906.05714、Attributor2404.01361及AAAI2025版、PromptAid |
| 6 | `"Interactive Prompt Debugging with Sequence Salience" site:aclanthology.org`; `"Ecco" "Explainability" site:aclanthology.org`; `"LMDiff" site:aclanthology.org`; `"Transformer" "visual analytics" 2026 2025 2024 interpretability` | Ecco、LMdiff、SemanticPathway；未找到SequenceSalience匹配Anthology出版页，不写ACL已接受 |
| 7 | `"Semantic Pathway" "pdf"`; `"LayerFlow" visual paper`; `"visual analytics" "token" "2026" language`; `"Transformer Debugger" paper visual` | SemanticPathway作者PDF、LayerFlow官方主文/补充、TransformerDebugger官方软件；同名视频生成LayerFlow排除 |
| 8 | `"Semantic Pathway" "LLMs"`; `"Interactive Prompt Debugging" "aclanthology.org/2024"`; `"LM-Debugger" paper`; `"PromptAid" "PromptIDE" visual analytics language models` | KnowledgeDebugger2607.01000、LM-Debugger、PromptIDE、PromptAid、GraphGhost候选 |
| 9 | `"Interactive Prompt Debugging with Sequence Salience" ACL 2024 demos`; `"PromptIDE" "Interactive and Visual Prompt Engineering"`; `"LLM Comparator" arxiv 2402.10524`; `"KnowledgeDebugger" visualization 2026 pdf` | PromptIDE2208.07852、ComparatorCHI EA、LLMbench2604.15508；官方abs核版本 |
| 10 | `"LLM Transparency Tool" paper`; `"Interpretability in the Wild" visual language models`; `"LLM" "attribution" "visualization" 2026 system`; `"Semantic Pathway" visualization "2026"` | LM-TT；IOI/CLT-Forge交foundations；NeuralTransparency、MultiTurn、VISTA候选 |
| 11 | `"Neural Transparency: Mechanistic Interpretability Interfaces" arxiv`; `"Multi-Turn Neural Transparency" arxiv`; `"VISTA: Visualization of Token Attribution via Efficient Analysis" pdf` | 一次结果因上下文截断重查；确定2511.00230、2605.15455、2604.02217；VISTA SSRN6510678并为同work |
| 12 | `"AttentionViz" "CommonsenseVIS" pdf`; `"VisBERT" pdf`; `"LLMCheckup" visualizations` | 由SemanticPathway/SequenceSalience参考文献扩展：VisBERT2011.04507；LLMCheckup2024.hcinlp-1.9 |
| 13 | `"AttentionViz" arxiv`; `"CommonsenseVIS" arxiv`; `"VISIT" "semantic information flow" aclanthology` | AttentionViz2305.03210、CommonsenseVIS2307.12382、VISIT2023.findings-emnlp.939纳入；新近VeriLLMed2604.23356待筛 |

## 引用链与版本处理

- LSTMVis → Seq2Seq-Vis → exBERT → LIT → Sequence Salience；RNNVis并行经典线。章节正文与附录分批完整阅读，并非以摘要代替。
- Ecco/LMdiff → GLTR（输出token概率可视化先例，待处理）。
- LM Transparency Tool → VISIT、LM-Debugger；Information Flow Routes交foundations。KnowledgeDebugger继承LM-Debugger并接知识编辑。
- SemanticPathway → AttentionViz、CommonsenseVIS、VisBERT；不能仅重复其相关工作段断言其他工具不支持生成。
- SequenceSalience arxiv v1（11 Apr 2024）与官方教程称submitted ACL2024demo；尚无正式ACL匹配，保留预印本身份。
- KnowledgeDebugger2607.01000v1与LLMbench2604.15508v1均2026预印本；不得写成已发表会议论文。
- 神经透明性两篇为不同研究：单轮界面与多轮动态界面/随机实验，分别纳入。
- TransformerDebugger是软件条目，未找到正式独立论文；不能伪装论文全文阅读。

## 获取记录与开放问题

- 所有保存PDF检查`%PDF`后用Poppler提取。首次默认沙箱下载DNS不可达；在已授权公开获取范围内网络提升后成功，非站点访问绕过。
- PromptAid作者站 `https://www.bckwon.com/pdf/promptaid.pdf` 返回403；不重复撞该站，继续找另一作者公开PDF/正式OA。
- PromptIDE提取提示`xref num 291 not found but needed, try to reconstruct`，需页渲染验证，解析成功不自动等于完整。
- SemanticPathway关键机制疑点见FULLTEXT_NOTES：固定因果前缀中既有token状态不应因追加未来token改变，图投影与真实状态须区分。
- LM-TT官方GitHub2026-02-01归档，README许可证CC BY-NC4.0；记录版本边界。

## 初筛排除/外域分工

| 候选 | 决定与理由 |
|---|---|
| Inseq、PromptExp、How are Prompts Different in Terms of Sensitivity?、Contrastive Attribution in the Wild、Localizing Prompt Ambiguity | generation域；跨域复用，不重复计work |
| Interpretability in the Wild、CLT-Forge、Information Flow Routes、Captum/TokenSHAP、IG/SHAP/faithfulness | foundations域 |
| InfoCIR2602.13402、ChartLens、MAVIS、RADAR、VisEdit2408.09916、EL-VIT2401.12666 | 主要图像/VLM/图像检索，非本项目文本生成归因核心；若后续方法论需要再补 |
| LEVA2403.05816、LLM-assisted VA survey2409.02691、2503.15176、Lexara2603.05832、VACP2603.29322、SocialFiVis2608.08497、ManiScope2607.11451 | 用LLM辅助其他领域分析/代理，不直接解释LM行为；综述仅发现用 |
| LayerFlow同名视频生成工作 | 同名异作，排除 |

## 停止状态

仍有直接候选待全文，未满足两轮收敛。全文状态以PAPERS.json和证据卡为准；数量不是停止理由。


## 批次14–17：版本与引文补充（实际查询）
- “"LLM Attributor" AAAI 2025 pdf”；“"PromptAid" pdf "2025"”；“"VeriLLMed" "2604.23356"”。AAAI2025正式3p metadata确认，但官方PDF请求RemoteDisconnected，尚未版本比较；PromptAid换另一作者合法公开站成功，非绕过原403。
- “"KnowledgeVIS" arxiv "prompts"”；官方open arxiv2512.11573与2604.23356。KnowledgeVIS2403.04758直接纳入；VeriLLMed是输出文本推理关系对KG审核，不是LM内部/输入输出归因，摘要筛选留相邻范围，未借其结论。
- “site:proceedings.mlr.press "Visualizing token importance"”；“site:proceedings.mlr.press "Quantifying Perturbation Impacts"”。DBSA正式AISTATS2025确认并下载PMLR原PDF，arxiv与正式逐段差异仍待审（字体提取差异很大，不假定等同）；2412.00868归generation。
- “"Neural Transparency" "3742413.3789120"”；“"Designing a dashboard" "2406.07882"”；“"TX²" "Transformers" visualization explainability”。NeuralTransparency官方IUI2026确认（不能写CHI）；dashboard直接下载；TX²搜索未找到目标须进一步精确查。
- LayerFlow官方Eurographics补充1页获取读完：interlinked stretch与matrixorder伪码。LMExplorer/LMFingerprints仅列跨层embedding邻近范围，本项目当前不主张新embedding几何工作流，不以摘要替代其全文证据。
- MultiTurnTransparency引文新增ConceptViz、SAE Semantic Explorer2511.06048、ConceptExplorer2603.23524、BAGEL2507.05810，已交root协调筛选/全文；不能在未处理时声称收敛。


## 批次18：四个SAE引用邻居（实际官方源）
查询“ConceptViz Visual Analytics concepts language models”、2511.06048、2603.23524、2507.05810；分别打开官方arxivabstract。ConceptViz明确输出steering和验证，纳入全文。SAESemanticExplorer2511.06048仅curatedfeatures topology/ConceptExplorer2603.23524仅解释embedding分层导航，按当前目标归因/触发诊断范围外围，status明确abstract_screened，不推正文结论。BAGEL打开官方HTML§§3.2/4确认图像CNN与CLIP概念；MultiTurn文中把它说LM不能当原文事实。排除文本LM主线，非因work数量。


## Expansion batches 19–26 — 2026-09-09 (actual searches, not closure)
- Batch19: `"LLM Comparator" "Interactive Analysis" pdf 2025`; `"TX2" "Transformers eXplainability"`; `"Dodrio" "Exploring Transformer Models"`; `"NeuronautLLM"`. Found official VIS extended PDF, OSTI TX2, ACL Dodrio and OA ScienceDirect Neuronaut; first three retrieved.
- Batch20: `"NeuronautLLM" "pdf"`; `"Attention Flows" "Analyzing and Comparing" language models pdf`. Found author AttentionFlows PDF and arxiv2009.07053; Neuronaut only publisher access.
- Batch21: `"LLM Comparator" "Appendix A1"`; `"ShortcutLens" pdf`; `"NeuronautLLM" pdf`; `"CommonsenseVIS" TVCG`. Found author ShortcutLens, CommonsenseVIS formalTVCG2024 metadata via HKUST and author project; Comparator11p lacks referred appendix.
- Batch22: `"LLM Comparator" supplementary material appendix`; `"LLM Comparator" "1326" supplemental`; `"NeuronautLLM" Woodman pdf arxiv`; `"GraphGhost" 2510.08613`. GraphGhostv2 direct token dependency graphs/interventions, accepted by visualdomain; formal Comparator repository metadata verified.
- Batch23: `"LLM Comparator" "supplemental"`; `"NeuronautLLM" "Woodman" -site:sciencedirect.com -site:researchgate.net -site:ouci.dntb.gov.ua`. Found official VIS content-staging page and author wenzhen.site; checked each. VISsupp points GitHub root, notpaper supplement; GitHub public tree has noPDF/appendix/survey; archiveFeb12,2026.
- Batch24: `"LLM Comparator" "Kahng" "appendix" 2025`; `"Exploring the neural landscape" filetype:pdf`; `"DeepNLPVis" saliency feature attribution`; `"RAGExplorer" visual 2026 attribution`. Exact Neuronaut query noauthorizedfulltext. DeepNLPVis2206.09355 has per-layer word information and improvement cases→direct foundationalneighbor. RAGExplorer2601.12991(authorPDF) explicitly context manipulation→answer effect, direct near-neighbor. These must be handled, no closure claimed.
- Access: NeuronautScienceDirectfullPDF403; did not bypass. AuthorpublicationHTMLhrefscan confirms no NeuronautPDF (otherpapers have PDFs). Kept abstract status. AttentionFlows/ShortcutLens authorPDFs fetched (%PDF+pdftotext).
- Scope: ShortcutLensabstract+§5.3 is dataset pattern productivity/subset-removal evaluation, outside targettoken attribution; PromptIDE/PromptAid/ChainForge kept adjacentabstractscreened, no fulltext conclusion asserted. KnowledgeVIS fillmasktarget probing and LLMCHECKUP featureattribution dialogue remain direct and reading pending.

## 批次25–28：版本、补充和直接引用扩展（实际执行，2026-09-09）
- 查询：`"A Unified Understanding of Deep NLP" supplement`；`"LLM Attributor" "35357" pdf`；`"GraphGhost" "2510.08613"`；`"DeepNLPVis" "supplement"`；`"LLM Attributor" AAAI 2025 pdf`。GraphGhost官方v2核对、全文与附录A–D完成；Attributor官方AAAI2025缓存PDF3页全部文本可读，但当地下载连接关闭、正式版截图timeout，保留像素复核缺口。
- 查询：`"DeepNLPVis" github`；`"DeepNLPVis" supplement github`；`"3184186" supplemental`；`"A Unified Understanding" "supplementary" NLP`；`"DeepNLPVis" supplementary material`。Microsoft官方15页正文全部读完，官方研究页与IEEEVIS2022页没有找到单独7数据集margin附件；arxiv元数据14页也不自动当补充。独立附件仍未获，不能将body状态写全文闭合。本文原图与证据卡已保存，不因缺附件丢弃可核正文。
- 查询：`"KnowThyself" "Prasai" arxiv`；`"Monitoring the Internal Monologue"`；`"KnowThyself" "42373" pdf`。KnowThyself官方AAAI2026元数据及arxiv2511.03878v1（实际4页）全文；Monitoring2605.18549v1 38页正文附录A–K完整读，图31–38全部核。后者主文的test-set CV只诊断可分性，不能冒充heldout部署效果，AppE.2真迁移分开记录。
- DBSA：PMLR官方22页含A–E再次独立全文读，未仅靠字符diff判等；科学内容与arxiv既有审查一致，Fig1错刻度/B.6索引等已正式图核。
- RAGExplorer2601.12991：作者PDF摘要与方法边界核为编辑retrieved context后的回答/规则/LLM judge诊断，并无具体数值input→生成target attribution；按当前范围外围，不用其案例主张新归因机制。Ding2017/Lee2017由root全部正文附录/原图审查，visual manifest跨路径引用，不重复计数。

## 批次29：LLMCHECKUP与Monitoring引用链范围检查（实际执行）
- 查询：`"XMD" "Explainable" NLP interactive 2023`；`"IFAN" "InterroLang" explanations`；`site:aclanthology.org/2023.acl-demo "XMD"`；`"IFAN" "Mosca" 2023`。官方ACL XMD2023明确归因→反馈→正则化→OOD模型修复，直接纳入全文；IFAN2303.03124官方摘要明确反馈归因→adapter debias，也纳入，未因较早/分类任务草率排除。
- InterroLang2023.findings-emnlp.359官方摘要已读：多轮对话解释/feature attribution+simulatability是邻接用户评估；尚未全文，其结论不得以本review全文已核引用。当前等待scope确认，若纳入直接评估结论需完整读23页。
- 查询：`"Beyond Linear Probes" "Dynamic Safety"`；`site:arxiv.org/abs/2601.02170`，随后直接打开官方arxiv2601.02170与ICLR2026 TPC官方摘要。TPC2509.26238为多项式probe随算力级联，Streaming Hallucination为prefix累计幻觉状态；均非输入/历史到固定target归因或trigger可视诊断，本轮仅外围摘要筛选，不使用其性能/机制正文主张。backdoor agent确认未重复覆盖。
- 以上仍是扩展，不标closure；XMD/IFAN待完整处理后再执行两轮真实最终查漏。完整范围不等于穷尽所有classification、probe或RAG dashboard论文。

## 批次30：最终查漏发现的直接邻居（2026-09-09，实际执行）
实际查询：
1. `("language model" OR LLM) ("attribution" OR saliency) ("visual analytics" OR "interactive visualization") 2026`
2. `("language model" OR LLM) ("visual debugging" OR "visual diagnosis") ("trigger" OR "backdoor") 2025 2026`
3. `"Visualizing the Chain of Thought in Large Language Models" 2026`
4. `site:aclanthology.org 2025 2026 "interactive" "attribution" "visualization" language generation`
5. `"CafGa" "Customizing" pdf`
6. `"Semantic Pathway" language model visualization`
7. `"Exposing the Unsaid" "Stochastic"`
8. `"AttentionRadViz" "Token"`
9. `"Explainable Mapper" LLM`
10. `"Beyond One Output" "Generations"`
11. `"Exposing the Unsaid" Pelossi arxiv`
12. `"AttentionRadViz" Wilber`
13. `"Simplifying Outcomes" ELIA arxiv`
14. `"AttentionRadViz"`
15. `"Verify-First" "Visual Triage"`
16. `"Understanding large language model behaviors through interactive counterfactual generation and analysis"`
17. `"Visualizing the Chain of Thought" "Ilgen" pdf`
- 官方ACL/IEEE VIS2026 accepted paper list与arXiv页面确认ELIA、CafGa、TreeTracer；CafGa引文指向LLM Analyzer。这四篇直接进入全文、全部正文/附录和核心图读完，证据卡已保存。它们使“多视图/假设探索/局部验证都从无到有”的创新定位不成立，但没有抹除新认知贡献空间。
- GROVE2604.18724v3、ExplainableMapper2507.18607v1官方摘要筛选：前者比较生成文本分布，后者embedding拓扑解释；均非当前输入/历史到目标归因或触发诊断，不借用其全文实验。mllm-shap2026.acl-demo.38为文本音频模态扩展，外围。
- CG&A Viewpoints `Visualizing the Chain of Thought in Large Language Models` DOI10.1109/MCG.2025.3624666与Semantic Pathway是不同作品；官方作者页/机构缓存摘要确认概念议程，机构PDF返回HTML，未伪造PDF或正文阅读完成。仅作为概念性外围，不支持实证结论。
- AttentionRadViz（Jared Wilber）只查到VIS2026官方short paper接受标题，精确查询没有公开摘要/PDF；标title_only_unavailable，不能从标题推方法。Verify-First只见视觉解释稳定性triage标题，无明确target归因线索，不进入实证依据。泛dashboard、RAG编辑、SAE词典与多模态工具不是因篇数配额排除。

## 批次31：最后两篇生成树直接引文；停止边界
- 实际查询 `"generAItor" "Tree-in-the-loop" pdf`、`"Revealing the Unwritten" 2025 language models`；打开作者项目、arXiv2403.07627、ACL2025.acl-demo.29。generAItor32p由visual全文，Revealing the Unwritten12p由root全文并只在foundations登记，避免同谱系重复计数。
- 最终元数据核验打开InterroLang官方ACL页，更正完整标题；维持外围摘要状态，不引用其user-study结果作已全文证据。generAItor的arXiv作者稿出版模板残留已明确，不当J.ACM正式发表事实。
- 用户续接指示为不重复检查、不做形式检索循环。此前实际查漏发现的新直接邻居全部已处理；不声称完成了“两轮零新结果”或全球文献穷尽。冻结范围为输入/历史到生成目标的归因、可视发现及验证、直接触发/审计；保留合理经典和直接引用链。通用LLM评测/写作UI、纯embedding/SAE词典、其他模态不继续扩展。
- 视觉清单于本批冻结：65条，45篇full_text_read，2篇正文已读但独立附件缺失，1篇全文不可得，1篇仅标题不可得，1项软件文档，15条外围/越界摘要或方法筛查。full_text_read以所列版本为边界（如Attributor正式版像素复核、KnowThyself正式版与作者版比较仍有限制）；不可简写为所有最终正式版本均闭合。
- 主要访问缺口：LLMComparator Appendix A/B；DeepNLPVis独立7数据集margin附件；Neuronaut授权全文；AttentionRadViz公开摘要/全文。各自入口和尝试已记录，不把缺口当未发现先例的证据。
- 本域跨域协助Context Influence完整17p卡在GENERATION_ASSIST.md，由generation唯一登记；Auditing Hidden Objectives第31–63页协助另存AUDITING_APPENDICES_ASSIST.md，由foundations F31唯一登记，均不增加本域计数。
