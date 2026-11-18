# D3 修订记录

## 2026-11-19：原 A 交审后、B 前的定向补查

触发：主审要求处理原稿保留的 Token-Level、Homeopathic、Agentic、Inherited 及 RNG/recency 缺口，并整合 D6 Subliminal 证据。此轮是原 A 补查，不是 B/C 审查替代。

原始 `REPORT_A.md` 保持不变。更新 `REPORT.md`、`PAPERS.json`、`SEARCH_LOG.md`，新增 `SUPPLEMENT_READINGS.md`；sources 新增合法 RNG/recency PDF和文本、Token-Level p34视觉核验。

| 原问题/缺口 | 本轮实际处理 | 对判断的修正 |
|---|---|---|
| Token-Level仅下载、未读 | 全文pp1–44+refs，表20回PDF截图 | 增加剂量/家族/秩/触发条件证据；撤回候选状态。剂量终点不是自然时间轨迹；r16多seed比较不成立为同等重复证据；种子混合多因素 |
| Homeopathic待跨域整合 | D6全读基础上，D3回关键方法/结果/附录 | 正常数据促进形成已是直接先例，不是待定空白；增加batch反向、纯毒高剂量可传、训练窗口及条件性/范围区分 |
| Subliminal优化器/LoRA边界未纳 | 核v4和§6、J–P | 坐标缩放已有干预证据；SGD/全FT阴性必须限设置，不能外推普遍不可能；内部方向与语义传递分离 |
| Agentic仅题名待整合 | 核§2–4及A/B所有相关表 | 预训练后干净训练显现已有阶段先例；92.7%Pass32不是单次率，单次50.9%；低量2500文档单次仅.07%；不同token预算不能混算 |
| Inherited仅二手线索 | 正文+相关附录全读 | 增加相同毒响应下仅clean回答来源不同的直接证据；student有效范围不等于teacher；选择性S不能视为ASR |
| RNG前向线索未处理 | 官方身份/PDF+方法/实验范围筛查 | 工程随机源安全，不新增D3语言后门条件实测；排除直接证据并保存判定依据 |
| Recency前向线索未处理 | ICLR正式PDF+设定/讨论范围筛查 | 训练顺序线性读出属相邻信号；抗植入明确为未来问题；不把相邻现象当后门机制结论 |

全读责任透明：`FULLTEXT_READ` 15项；`CROSS_DOMAIN_FULLTEXT_READ_D3_RELEVANT_VERIFIED` 3项（Homeopathic/Subliminal=D6，Agentic=D1）；`SCREENED_OUT_DIRECT_CONDITIONS` 2项。原稿13项已读证据均保留，新增证据不抹掉反例或不确定性。

验证：JSON可解析、ID唯一、20项文件路径均存在；正文不再把上述五篇称待读；REPORT_A未重写。未进行实验、GPU运算、论文正文修改或推送。

下一流程仍是 B 对原稿与新增实质判断逐项回原文核验，再由 A 响应，最后 C 独立查漏。本轮不签发领域覆盖完成。

## 2026-11-19：响应独立 B 初审 R1–R8

输入：`B_REVIEW.md` 的84项核验及 `CHANGES_REQUIRED`。原 A 逐项采纳全部八项要求；另采纳 N1/N2。此处记录的是 A 完成修订，**尚待原 B 复核，不自行将状态改为通过**；C仍待独立查漏。

| B项 | 处理 | 修改与核证位置 |
|---|---|---|
| R1 | 采纳 | §2 Wan行改为任务评分下降，明确ROUGE-L与exact match；PAPERS同步。依据Wan §5/Fig6的R/E指标区分。 |
| R2 | 采纳 | §3.1补内部Haiku每组等量正确标注有害的near-trigger例、threshold .5、单trial/100 OOD；明确内部主要保留DEBUG NO-FLAG MODE；构造配比与定位在下方R10进一步校正为正文内部分类器小节及Appendix E，PAPERS同步。 |
| R3 | 采纳 | §3.2明确LLaMA mini COMPLIANCE/REFUSAL/COMMAND为2/4/5毒例、50epochs；对应full为5/9/9、100epochs。回Pick E.2 p29确认，保留强重复暴露边界。 |
| R4 | 采纳 | §5.5将“无patch”改为“无右下角触发”；补正常训练数字半数带左下角3×3控制白块；主对比20runs，异类6.2%控制5runs。PAPERS及补读笔记同步，采用原文E/FigE.2条件。 |
| R5 | 采纳 | §6.5最低剂量段点明declaration+rare-string；评价段补普通995/稀有1000prompts，Pass1来自同一个每题32rollouts采样池，非另跑确定性测试。PAPERS同步。 |
| R6 | 采纳 | §8重合表拆为“点级归因代理与集合成功”（Pick）及“teacher-forced PPL与完整序列生成”（Lin）。仅后者给1000字符r64 PPL1.12/59字符匹配；不再用Pick支持低目标loss与ASR脱钩。 |
| R7 | 采纳 | §7 Xu定位改为§5/Table3 p7；PAPERS同步。 |
| R8 | 采纳 | §5.5 German72%主结果补每run10问题×5采样=50响应、3训练seeds；§6.4全参9run结果点名Gemma-3-4b-it，保留LR不匹配限制；PAPERS同步。 |
| N1 | 采纳 | 文首加入Shumailov补充连续印刷页例外，p26/p31=补充PDF14/19；PAPERS登记独立supp.pdf/.txt和页码规则。 |
| N2 | 采纳 | 回MDPI §4.3.3相关段确认，§5.4补“单模块移除最多前16奇异方向未见ASR下降”，仅作为已试删除未定位稀疏致因方向的负结果，不推导唯一分布式机制。 |

检查：JSON解析和唯一ID/全文及补充文件路径检查通过；逐项确认新措辞存在；REPORT_A.md仍为13,485字符原稿，未重写；不新增文献、不运行实验、不改论文或发布。


## 2026-11-19：响应 B Round2 的 R9/R10

B Round2 已核准原 R1–R8 的主体修订及 N1/N2 的实质结果，仅留下两处范围措辞。原 A 回到相应原文确认后均采纳；本记录不代替原 B 的最终文本复核，C仍待独立查漏。

| B项 | 处理 | 修改与原文定位 |
|---|---|---|
| R9 | 采纳 | REPORT §5.4将“干预仅”限为“§4.3.2的层消融仅”，避免排除文中的SVD及layer restoration干预。§4.3.3 pp17–18（本地txt 977–984行）明确SVD与单模块方向删除只涉及第17/18层四类LoRA模块，REPORT/PAPERS同步限定位置，仍不将负结果升级为唯一机制。 |
| R10 | 采纳 | Bowers正文“Reproducing Results on an Internal Anthropic Classifier”（本地txt 149行）说明内部多数仅留DEBUG标记；Appendix E（208行）明确开放实验1/4仅留标记、3/4对口令随机删除/交换/插入至编辑距离恰为10。REPORT §3.1及PAPERS改为构成配比不同，撤回完全不同构造的暗示；同时校正上方R2的原文定位。 |

检查：JSON可解析，书目20项ID唯一，REPORT_A原始快照保持不变；本轮只修正上述两处及对应记录，未扩展文献或实验范围。状态：A已修订，等待原B仅核R9/R10文本落实。
