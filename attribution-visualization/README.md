# 归因可视化研究项目

2026-09-10 新立项，按“理论推导 → 明确预测 → toy”推进：

| 课题 | 立项 | 理论 |
|---|---|---|
| 有限污染下的排名可信复核 | [目标、方法与价值](ranking-audit/PROJECT_CHARTER.md) | [理论正文](ranking-audit/THEORY.md) |
| 归因可视化引导的受污染视觉语言模型事实能力修复 | [当前立项，用户已确认](visual-evidence-repair/PROJECT_CHARTER.md) | [当前 N1–N10 理论](visual-evidence-repair/THEORY.md) |

两项的[前次三人独立理论审查及修订](theory-review-2026-09-10/REVIEW_SYNTHESIS.md)结论是：排名只闭合单错误复核与树图认证；当时视觉只闭合可校准前提下的受限事实诊断。视觉课题的最新状态见下一段。一般图多错误方法、真实 VLM 修复效果及论文级贡献仍未验收，尚无新 toy 结果。下文保留首轮 OA 探针的原始范围与记录。

视觉修复以 N1–N10 新理论为实施标尺，闭合“事实响应→冻结权重→原参数更新→有限校准选择→原生成评价”的操作链和条件推论。旧 C 编号理论已[归档](visual-evidence-repair/THEORY_V3_ARCHIVED.md)，[旧审查记录](visual-evidence-repair/theory-review-v3/REVIEW_SYNTHESIS.md)只作历史追溯。加权优势、未知触发修复及跨任务迁移仍待实验。

当前是[48 GPUh 规划 V3](visual-evidence-repair/TOY_PLAN.md)：一个污染模型、单 seed、六条修复；先观察与冻结，再独立检验真实任务增量，多模型与正式 SOTA 矩阵后移。[实现与资源复用](visual-evidence-repair/IMPLEMENTATION.md)列出实际命令、配置和验证记录；代码支持官方 HF LLaVA/Qwen 加载、主要修复对照、完整答案生成、离线归因页面和隔离评价。随机微型模型的代码测试已与科学实验分开；真实污染权重、合法配对材料及原协议评估交付尚未齐备，没有新 toy 性能结果。[第一版计划](visual-evidence-repair/TOY_PLAN_V1_LIMITED.md)、[旧自查](visual-evidence-repair/TOY_SCIENTIFIC_AUDIT.md)与[V2 修订依据](visual-evidence-repair/TOY_V2_LOGIC_AND_EXPECTATIONS_AUDIT.md)保留供追溯。

以归因可视化为研究工具，发现并验证 LLM 的触发行为、上下文依赖与生成历史传播规律。保留三个核心视图：输出位置热图、点击输出词查看输入／历史来源、跨步与跨条件对照。项目不要求先发明一个新的归因算法。

**首轮探针已完成：12条基础输入×3种模型状态×3种条件，108条轨迹，8,274/8,274个输出位置的概率差与来源归因。** 查看[逐例观察与现象归纳](PROBE_OBSERVATIONS.md)、[全量交互索引](runs/attribution-probe/full-html/index.html)和[服务器执行记录](SERVER_EXECUTION.md)。22条回答保留原200-token截断边界；本轮没有训练新模型或验证toy假设。

讨论本轮收获时先读[阶段总结：我们得到了什么](PROBE_STAGE_SUMMARY.md)和[第二次挖掘报告](PROBE_DEEP_MINING.md)，再按具体问题回到逐例记录和原图。

实验入口：[run_probe.py](run_probe.py)与[全位置补算](complete_attributions.py)；[运行与续跑](RUNNING.md)；[只读Notebook](observe.ipynb)；[代码验证记录](CODE_VERIFICATION.md)。完整运行数据、HTML、完成回执和第二次挖掘统计已打包纳入 Git，见[下载、校验与解压说明](artifacts/README.md)。解压后可使用本页的运行数据与热图链接。

| 阅读顺序 | 文件 | 内容 |
|---|---|---|
| 1 | [立项文书](PROJECT_CHARTER.md) | 项目目的、范围、成功条件；最后一节回填全文调查后的定位 |
| 2 | [Related work 综合报告](RELATED_WORK.md) | 哪些已经做过、哪些可复用、哪些仍是待验证的认知问题 |
| 当前① | [可复用资源与采用决定](REUSABLE_RESOURCES.md) | 158条文献对应资源的盘点、模型/数据/缓存/工具与可用边界 |
| 当前② | [探针实验设计](PROBE_EXPERIMENT_PLAN.md) | 资源盘点后选定108条轨迹的观察方案；不预设科学假设 |
| 当前③ | [12条输入清单](PROBE_INPUT_MANIFEST.json) | 从作者313对测试输入中冻结样本、条件、来源与hash；本批计算已完成 |
| 当前④ | [观察报告](PROBE_OBSERVATIONS.md) / [浏览索引](runs/attribution-probe/full-html/index.html) | 全轨迹观察、全位置归因、代表例与反例、分母、局限及开放问题 |
| 3 | [测量与研究计划](MEASUREMENT_AND_RESEARCH_PLAN.md) | 测量约定；探针／toy分界；未选定的后续参考方向 |
| 4 | [覆盖审计](COVERAGE_AUDIT.md) | 实际阅读数量、全文／附件缺口、版本与实现边界、停止范围 |
| 查询 | [逐篇索引](PAPER_INDEX.md) / [合并清单](PAPERS.json) | 官方来源、本地全文、版本、证据卡、逐项疑点 |

核心判断：**立项可行，但逐词着色、输入—输出热图、历史图和一般的“可视观察→假设→干预”流程都有先例。** 2026-09-10明确阶段定义：探针先观察、归纳现象，探索能提出什么假设；toy才提出明确假设并尝试初步方法。原A/B/C保留为后续讨论参考，不作为首轮探针的既定任务或现象分类。

本轮交付为立项、全文邻近工作调查、资源盘点、实验设计、最小实验代码及已执行的探针观察册。初始计算与全位置补算合计约16分钟单卡时间，不包含环境排障和研究整理。随机微型模型检查、正式发布模型计算和现象归纳分开记录；这不等于完整复现作者协议。全文数量及未读材料以覆盖审计为准，不声称全球穷尽。

`review/` 按生成归因、后门／OA、可视分析与基础方法分四域，保存实际检索日志、全文证据卡、PDF、提取文本和关键原图。每篇以所读版本为准；附录、形式版本和作者报告的边界不混算。2026-09-09文献交付的完整性记录见 [ARTIFACT_CHECK.json](ARTIFACT_CHECK.json)。

本目录通过轻量导入复用仓库已有OA常量和保存函数，没有修改既有实验入口，没有进行模型训练。PDF、提取图、全文及第三方源码快照保留为本地阅读缓存，Git中的论文索引保留官方来源与证据卡。全文调查截止：2026-09-09；资源盘点、具体探针设计及实现：2026-09-10。
