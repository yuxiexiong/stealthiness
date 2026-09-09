# Generation 域检索日志

日期与截止：2026-09-09。先读取 PROJECT_CHARTER.md 与 review/PROTOCOL.md。保留可视化作为核心研究手段；并不要求归因算法必须原创。所有总命中数均为 `not_exposed`。下面记录本任务实际执行查询，未将论文作者宣传性“first/SOTA”当作本项目结论。

| 轮次 | 实际 query / 入口 | 新直接候选与处置 |
| --- | --- | --- |
| G0 定向种子 | `site.aclanthology.org Inseq interpreting sequence generation models contrastive attribution`; `site.aclanthology.org Contextual Feature Attribution Interpreting Language Model Predictions` | Inseq、Yin/Neubig 对比归因、ALTI-Logit；PECoRe 名称从作者项目确认，全部纳入。 |
| G0 种子确认 | `Sarti 2024 "context" "contrastive" attribution context cues`; `"Detecting and Characterizing" "Context" language model Sarti` | PECoRe、MIRAGE 纳入。生成概率/输入归因两个步骤已有，任务不因已有工具而自动否决。 |
| G1 参考扩展 | Attribution Contract §5.2 与参考文献；PECoRe §2 与参考文献 | CAGE、HETA、ContextCite、ReAGent、Jacobian Scopes 纳入；Captum 和一般 IG/SHAP/忠实性等转交 foundations。 |
| G1 新作确认 | `"HETA" "attribution" language model`; `"CAGE" "Explaining the reasoning" attribution`; `"Jacobian Scopes" language model attribution`; `"ContextCite" arxiv` | CAGE 2512.15663、HETA 2604.13258、Jacobian 2601.16407、ContextCite 2409.00729；新增 What Really Counts? 2511.15886，纳入。 |
| G1 后向种子 | `"ReAGent" "model-agnostic" attribution arxiv`; `"TokenSHAP" "arxiv"`; `"Interpreting Language Models with Contrastive Explanations" PDF`; `"Explaining How Transformers Use Context to Build Predictions" arxiv` | ReAGent 纳入；TokenSHAP 2407.10114 交由 foundations。其他为已知种子。 |
| G2 另一视角提示 | visual 域提出 Contrastive Attribution in the Wild、PromptExp、Prompt sensitivity、PRIG | 均进入候选，不以不同任务名草率排除。 |
| G2 实际查询 | `"Contrastive Attribution in the Wild"`; `"PromptExp" prompts sensitivity`; `"How are Prompts Different in Terms of Sensitivity"`; `"Localizing Prompt Ambiguity with Probe-Targeted Attribution"` | 新直接候选：2604.17761、2410.13073、2311.07230、2606.05486，等待全文。 |

## 获取与访问记录

- 官方 arXiv PDF 与 ACL Anthology PDF，小批次按明确候选获取。
- 默认沙箱 curl 曾报告 `Could not resolve host`；按工具正常升级权限后下载，不绕过验证码或付费访问。
- OpenReview 某次网页返回 browser verification；改用同一作者公开 arXiv 版本，并保留版本边界。
- PDF 经文件头、pdftotext 与页数检查；下载/提取不计全文读完。

## 当前覆盖状态

早期阶段未达到两轮零新增直接近邻；最终冻结范围及保留缺口见末尾G7。全文状态以 PAPERS.json 和逐篇证据卡为准。

## G3–G5 引文与更新扩展（实际执行，2026-09-09）

- G3：`"Contrastive Attribution in the Wild"`；`"PromptExp" "arxiv"`（初次无有效命中）；`"Analyzing Chain-of-Thought Prompting" "Gradient"`；`"A Causal Framework for Explaining" sequence`；SocRAT官方ACL全文无补充链接，随后arXiv 1707.01943末2页补充；`"PromptExp" 2410.13073`；`"Localizing Prompt Ambiguity" 2606.05486`；`"How are Prompts Different in Terms of Sensitivity"`。新增明确直接：SocRAT、cot-grad；Wild/PromptExp/sensitivity/PRIG为已有候选版本确认。SocRAT含训练植入词映射，不把2017排除在范围外。
- G4：`"HETA" "PML" "TDD"`；`"Learning to Attribute with Attention"`；`"GiLOT" "Multi-level" explanations`；`"Attribot" language model`；`site.arxiv.org "Peering" "language" 2405.17980`；`site.arxiv.org "Unveiling and Manipulating Prompt Influence"`；`site.aclanthology.org "2024.findings-emnlp.556"`；`site.arxiv.org "Black-box language model explanation by context length probing"`；`site.arxiv.org "Quantifying perturbation impacts" 2412.00868`；`site.aclanthology.org "Black-box language model explanation by context length probing"`。TDD/fAML/CLP/DBPA新直接纳入，PML待范围核查；AT2/AttriBoT/MExGen/GiLOT由foundations去重接读。此轮总新候选数未由搜索接口暴露；明确本域新增4篇直接、1篇待筛查。
- G5：`"Word importance explains how prompts affect language model outputs"`（新增WordImportance与Revitalizing）；后者ACL2026由backdoor接读；`"REX" "Incorporating Temporal Information" explanation`；`"Focus-LIME" "2602.04607"`；`"Beyond Attribution" "2410.12439"`；打开官方AAAI ReX、arXiv Focus-LIME/UnCLE。WordImportance、Focus-LIME新增直接全文；ReX待核；UnCLE官方摘要指向通用概念LIME/Anchors/反事实，不是直接prompt/history→output位置分析，列外围不递归扩展。
- 交叉域引文补充：PRIG refs产生TokenShapley（本域接读17页）；MExGen refs产生CC-SHAP和TextGenSHAP（本域接读）；Patcher refs产生TracLLM/AttnTrace（foundations接读）、RAGOrigin/RAGForensics（backdoor范围筛查）。PoiF是训练数据来源追责，属外围不纳入本域。
- 以上仍是有新增的扩展轮，不计闭合轮。计数为本域人工判定明确的新论文数，不能视为数据库全命中数。

## G6 生成来源与最新工作补链（2026-09-09）

实际查询：`site.aclanthology.org "On measuring faithfulness or self-consistency"`；`"TextGenSHAP" 2312.01279`；`"GRACE" "Measuring Chain-of-Thought Faithfulness"`；`"TokenShapley" site.aclanthology.org`；`"Tokengeist" 2026 attribution`；`"GRACE: Measuring Chain-of-Thought Faithfulness" arxiv`；`"Peering into the mind of language models" attribution`；`"GRACE: Measuring Chain-of-Thought Faithfulness" full paper author`；`"GRACEdiv" attribution github`。CC-SHAP正式ACL有v2修订，按v2下载；TokenShapley官方13页；Tokengeist为新增直接27页，与visual域MultiTurn2605.15455不同；GRACE为新增直接但获取受阻。新名必须题名/作者/标识核对，不能合并不同论文。

GRACE官方 https://openreview.net/pdf?id=b8pliYFlF3 与父代理已发现的 https://openreview.net/pdf/12f8e26580df852ed5abfb683bca922453a446c7.pdf 均curl 403；web open URL及search ref均重定向challenge。搜索结果虽缓存正文公式/摘要，不能算全文；作者匿名，无找到官方其他版本。保留 `fulltext_unavailable`，不以关键词拼接冒充完整阅读。不是automatic approval拒绝，也未绕过访问控制。

TokenShapley refs指向Peering正式ACL2024 findings.682；其余上下文引用生成/检索增强论文若只评语义支持无模型贡献，不泛化扩张全RAG引用生成。待Peering全文确认边界。

## G7 最后直接近邻、版本补全及冻结（2026-09-09）

实际入口/查询：打开 arXiv2412.11404、ACL2025.long.1220；查询 `"Attention with Dependency Parsing Augmentation" ACL 2025` 与 `"Measuring association between labels and free-text rationales"`。前者确认正式 ACL Findings2025.21，与arXiv同一论文，不增加篇数；正式16页全文已另行完整读取。后者确认为EMNLP2021.804，按项目有限范围登记为CC-SHAP历史前作，不全文纳入、不支持独立新颖性结论。

- Context Influence（Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models，ACL2025.long.1220）由visual代理完整读取17页、全部附录和原图。证据卡 `review/visual/GENERATION_ASSIST.md`，本域registry引用该卡，**只在generation计一次**。它是已发现待完成项，协作交付不计新搜索发现。
- CoT-grad 72页、CC-SHAP正式v2 42页、Peering15页、ReX官方9页+扩展附录6页、TokenShapley正式13页均已完成；早期日志17页/待筛为当时状态，现以registry版本和证据卡为准。
- TextGenSHAP交foundations并已由父代理完成正式ACL2024 28页。本域移除registry重复行；旧arXiv PDF/提取文本保留为已有获取工件，不计已读/未读队列，证据以foundations/F25为准。
- visual域CafGa、ELIA、GraphGhost、DBSA；foundations域TracLLM、AttnTrace、TreeFinder、AT2、AttriBoT、MExGen、GiLOT；backdoor域RAGOrigin/RAGForensics均不在本域重复计数。

最后一条直接引用来自Dependency全文refs：LRP4RAG2408.15533。实际先打开官方HTML v1（cache miss），再打开abs页并点击官方HTML v3 `https://arxiv.org/html/2408.15533v3`。发现2025v3已含原始生成token的LRP矩阵、类间热图和显/隐证据对比，而不是旧摘要所示的单一分类器；故新增**1篇直接工作**并下载v3。18页正文、参考、全部附录A–C及核心原图已读完。没有把“检测”题名当作直接排除依据，也没有顺着其refs泛化成全部幻觉检测综述。

外围处置边界：Lookback Lens2407.07071只作为Dependency引用中的一般attention比例检测背景，未在本域全文读、未用于方法忠实性或未解问题结论；普通CoT self-explanation测试、probe学习、概念级UnCLE、一般内部电路/可解释架构、RAG引用生成方法不因共享关键词递归扩展。PML2405.17980已消歧为Peering；“Learning to Attribute with Attention”早期误题已按全文纠正为AML。未获取独立来源和全文的其他别名不得计作已读竞争方法。

**冻结状态**：registry共33项：32篇 `full_text_read`，1篇GRACE `fulltext_unavailable`；主PDF共715页，不将同论文多版本及ReX额外附录重复算论文。32篇中Context Influence由visual协助完整读，其余31篇由本域读取。GRACE获取障碍与两条官方URL见G6；搜索缓存不替代全文，不宣称已解决。

**覆盖限定**：按用户与父任务最新收束要求，完成实际已发现的直接近邻后冻结，不为形式再做重复检索。本域没有实现连续两轮零新增直接近邻，因此不声明该项形式闭合、全球无遗漏或所有原始代码已复现。截止2026-09-09的这批有证据直接近邻足以反驳“输出敏感词→输入来源两步/文字高亮本身首次”，但并不否定已有工具支持新可视认知的立项。主报告若说明全任务检索收敛，应基于其自身实际跨域日志，而非将本域G0–G7改写为零新增轮次。
