# 基础方法域检索日志

日期：2026-09-09。工具：公开 Web 检索、ACL Anthology、PMLR、NeurIPS、arXiv、作者官方站及代码仓库。搜索引擎总命中数均为 `not_exposed`；不把返回条数冒充文献总数。全文读取状态见 PAPERS.json，以下是实际发生的检索与追踪，不是事后拟定的搜索计划。

| 轮次 | 实际查询 / 入口 | 发现与处理 |
|---|---|---|
| 初筛 | `TokenSHAP Captum LLM attribution paper` | F11、F12；输出相似度归因与逐输出 token 归因需分开 |
| 初筛 | `Axiomatic Attribution Deep Networks DeepLIFT Unified Approach Interpreting Model Predictions papers` | F01、F02；DeepLIFT 为基础算法候选，不直接据未读论文作细节结论 |
| 初筛 | `ERASER Benchmark Evaluate Rationalized NLP Models Faithfully Interpretable NLP Systems Define Evaluate Faithfulness` | F07、F08 |
| 初筛 | `Fooling Neural Network Interpretations Sanity Checks saliency Infidelity Sensitivity explanations paper` | F03、F04、F09；infidelity 原论文与 Sixt 为外围评测候选，暂不计全文 |
| 查证 | `TokenSHAP 2407.10114 paper` | arXiv v2 元数据与 PDF |
| 查证 | `Captum LLM Attribution tutorial feature attribution text generation` | F12；官方 API 与教程补充当前接口边界 |
| 查证 | `Toward Faithfully Interpretable NLP Systems How should we define and evaluate faithfulness Jacovi Goldberg 2020` | F07 正式 ACL 版 |
| 版本核查 | 直接访问 arXiv 2407.10114、2312.05491、2309.16042、2106.07475、1810.03292、1902.02041、1705.07874 | 固定版本；F03 v3 修正旧版实验，不能沿用旧结论 |
| 补充材料 | F02 NeurIPS 正式页面→Supplemental.zip | 读取四个补充 PDF；不执行 notebook/Julia 文件 |
| 引文扩展 | visual 域→2211.00593、2603.21014 | F13、F14；归因图及完整性验证 |
| 版本核查 | `TokenSHAP github Goldshmidt Horovicz sampling ratio` | 找到 ACL 正式版及当前改名仓库；补读正式版全部 8 页 |
| 版本核查 | `"Interpretability in the Wild" ICLR 2023 openreview`；`"Interpretability in the Wild"` 限 openreview.net | 正式版入口确认；官方 PDF 请求 403，使用公开 arXiv v1，显式保留版本缺口 |
| 补充材料 | `"Attention is not not Explanation" arxiv`；`"Attention is not Explanation" arxiv 1902.10186` | F06 补读 arXiv 附录，不只读会议正文 |
| 反向引文 | F05/F08→Alvarez-Melis & Jaakkola 2017 | 直接 seq2seq 邻近工作，交 generation 域全文核查 |
| 反向引文 | F07/F08→Pruthi et al. 2020 | 欺骗注意力解释直接邻近工作，交 backdoor 域全文核查 |
| 引文扩展 | LM Transparency Tool→2403.00824 | F15 Information Flow Routes，v2 全文 16 页 |
| 引文扩展 | CLT-Forge→Transformer Circuits 官方 methods.html、biology.html | F16/F17，保存官方 HTML 与可读文本，全文阅读尚需完成后才更新状态 |
| 代码核查 | TokenSHAP 官方 GitHub、Captum LICENSE | 当前仓库已更名/迁移；版本、许可与论文算法不可混为一谈 |

## 边界与未纳入条目

- `Towards Faithful Model Explanation in NLP`（2209.11326）：作为发现线索，不声称全文读过，不用其摘要替代具体原论文。
- DeepLIFT、LIME、ROAR、infidelity、When Explanations Lie：基础谱系候选；本域当前以 F01/F02/F03/F04/F07/F08/F09/F10 覆盖立项所用方法约束。若最终报告需要对这些候选作独立新颖性或性能判断，必须补全文，不能直接升级引用。
- TokenSHAP 仓库里的 PixelSHAP、VideoSHAP、AgentSHAP：图像/视频/工具贡献不在本轮文本输入—生成词—触发行为主范围；不据此声称做过或没做过本文核心问题。
- 所有公开下载失败都记录版本/来源缺口；OpenReview 403 未绕过。下载、转文本、读摘要不计 `full_text_read`。

## 后续全文扩展与查证（实际执行）

以下均发生于 2026-09-09；公开搜索的总命中数仍为 `not_exposed`。多版本按作品去重，不把一篇扩展版再计一篇。

| 阶段 | 实际查询或入口 | 结果与处理 |
|---|---|---|
| 后向／同作者扩展 | `Learning to Attribute with Attention`、`AttriBoT`、`MExGen`、`GiLOT` 的题名检索，ACL/PMLR/arXiv 官方页 | F18–F21 全文，含附录；记录 AT2 2026 匿名版只部分核对 |
| 统计条件核对 | AttriBoT App.B2 → NIST `Generalized Extreme Studentized Deviate Test` 官方页面 | 读完整定义和流程；区分 generalized ESD 与遇首个不显著即停止的顺序流程 |
| 历史回溯 | `"Visualizing and Understanding Neural Machine Translation" Ding 2017` | Ding 2017 正式 ACL 全文 10 页及图，交 visual registry |
| 历史回溯 | `"Interactive Visualization and Manipulation of Attention-based Neural Machine Translation"` | Lee 2017 EMNLP demo 全文 6 页及图，交 visual registry |
| 版本核查 | `"Learning to Attribute with Attention" 2025 2026` | 后续匿名 OpenReview 40 页，PDF 下载 403；不冒称已全读此版；2025 版 41 页全部已读 |
| 代码核查 | `"TokenSHAP" "sampling_ratio" github`；官方 GitHub tree、API commits/main、固定 SHA raw 文件 | 保存当前 commit 与实现、MIT LICENSE；已排除“当前实现最多 n 次”的猜测；保留非标准 Shapley 聚合和去空白／归一化边界 |
| 许可核查 | meta-pytorch/captum LICENSE 官方完整页 | BSD-3-Clause；仅许可／接口检查，不等于复现 |
| 溯源扩展 | `"TracLLM" USENIX 2025`；`"AttnTrace" "2026"` | F22/F23 固定全文，含全部附录与核心图 |
| 前向邻居 | `"Who Taught the Lie?"`；`"Traceback of Poisoning Attacks" RAG` | RAGOrigin / RAGForensics 交 backdoor；TRACE / Needle-in-RAG 同域核范围及全文 |
| 新版本复查 | `"AttnTrace" site:ieee-security.org`；`"TracLLM" attribution visualization generation saliency` | 核 2026 AttnTrace 题名及主文/附录差异；不是新独立论文 |
| 边界查漏 | `"Attribution graphs" "faithfulness" sanity 2026`；`"token attribution" "sanity checks" "generation" 2026` | 返回 GRACE、Tokengeist（交 generation），TreeFinder（本域接）；本轮有新增，不能算收敛 |
| 获取追踪 | `"TracLLM" arxiv` → arXiv2506.04202v3 | 22 页扩展报告；补读正式版 §4.4 引用的 App.D，核 Google 代理归因和 needle 试验 |
| 获取追踪 | `"TreeFinder" "attribution" LLM` → 作者 Damien Ernst 博客 → ORBi handle2268/337121 | F24 16 页作者全文；不是只读博客，含 App.A–D 及核心图 |
| 题名排歧 | `"GRACE" "Measuring Chain-of-Thought Faithfulness"` | 与同名 grounded-reasoning benchmark 区分；官方检索片段可见，但 generation 实际打开仍 challenge，不能用片段冒充全文 |
| 最邻近链核查 | AttnTrace refs62 → PromptLocate | 交 backdoor 判定是检测器定位还是受审模型生成归因，不扩大到所有提示注入检测器 |

F16/F17 的官方 HTML 正文、全部附录与脚注已经全文读取，核心图通过浏览器核对。原表中的“尚需完成”描述的是当时阶段，不是最终状态。当前仍在处理最终查漏的直接新增；最后两轮实际复查及其是否新增，单独记在下文和全局覆盖审计，不预先宣称收敛。

## 收尾扩展轮 R1 / R2（均有直接新增，不能计收敛）

- R1 实际查询：`"TextGenSHAP" 2024 2025 2026`；`"Obfuscated Activations" "attribution" "visualization"`；`"trigger" "token attribution" visualization backdoor 2026`。处理：TextGenSHAP 正式28页全文F25；MAD Functional Attribution F26；Discovering Backdoor Triggers 转backdoor全文；ICLR VKekM8kJ9U与已读X-GRAAD去重；PromptLocate被screen为检测器输入而非victim原生成，外围。CVPR26 Diagnosing and Repairing Unsafe Channels为VLM安全子空间修复，未作本项目novelty依据；专利CN122389023A只发现线索，本轮不作专利有效性/法律新颖性调查。
- R1 后向扩展：TextGenSHAP refs→`"Gradient-based analysis of NLP models is manipulable"`。F27正式12页全部及补充代码包目录/图文本核读；原根路径附件404已通过ACL正确attachments链接解决。
- R2 实际查询：`"attribution visualization" "backdoor" "LLM" 2026`；`"attribution" "obfuscation" "token" visualization`；`"TextGenSHAP" "counterfactual" "visual"`；`"causal" "generation history" "attribution" LLM`。新增2506.18053及2609.02000，均核官方arXiv全文。前者为架构混淆而非OA；后者虽VLM，但生成历史测量直接影响项目论证，升级全文，不因模态排除。其他命中为作者身份归属、字形混淆、交易所/机器人等同词异义，不纳入。
- R1/R2所有查询总命中数均 `not_exposed`。本轮新增都在处理后才进入最终再次复查。

## 收尾 R3 与最后直接引用链

- 实际查询：`"Mechanistic Anomaly Detection via Functional Attribution" attribution visualization`（arxiv.org/openreview.net）；`"token" "attribution" "backdoor" "visualization"`（ACL/arXiv）；`"prompt" "generated prefix" "attribution" visualization`（ACL/arXiv）。返回 MAD、When Backdoors Speak、MIRAGE 等已读工作；新增线索 GraCeFul/BFClass/ExposeBackdoors 为训练污染分类检测或整样本解释，AgentChord 为多 agent 之间输出关系，GMPO 为外部评分驱动提示优化，DiffMask 为历史分类掩码方法。它们作为外围发现线索，不据摘要作独立性能／新颖性结论，不冒称全文。
- 视觉域 TreeTracer/LLM Analyzer 引文扩展发现 generAItor 与 Revealing the Unwritten；前者交视觉域，后者由本域读 ACL 2025 demo29 正式12页全部及 App.A/核心图，记录 F30。该链确有直接新增，不能称连续两轮零新增。
- 2026-09-09 用户要求不要过度工程、不要重复检查：此后不为凑轮数重跑相同检索；补完已识别直接邻居后交付范围明确的全文调查。停止边界按 `PROTOCOL.md` 的收尾修订记录，不宣称全球穷尽或形式饱和。
- 后门域原文scope确认 Auditing language models for hidden objectives 的 §5.3.2 直接涉及原生成目标归因与可视发现；本域接 F31。公开arXiv v2共63页，由root读1–30、视觉协作者读31–63及核心图，完整纳入。同一引用链的 Sparse Feature Circuits 是图发现/机制编辑直接前驱，交后门域补读；一般SAE与安全评测只保留外围边界，不继续递归扩张。
