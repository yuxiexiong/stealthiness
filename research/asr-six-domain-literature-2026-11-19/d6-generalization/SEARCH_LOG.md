# D6 A 实际检索日志

日期：2026-11-19，Asia/Shanghai。范围：对后门形成有实质含义的特征竞争、延迟泛化/阈值、跨域有害行为；不以篇数为停止标准。

## 入口与程序

- 先读根目录AGENTS、六域PROTOCOL、shared/PROJECT_COMPARISON与ROOT_SEARCH_LOG。按共享候选核身份；未将旧本地综述当证据。
- web通用搜索发现入口后优先官方arXiv、OpenReview、会议/机构/作者资料；主张回公开PDF。所有PDF保存sources/，用`pdftotext -layout`提取；正文分段读取，再按主张读相关附录。
- 首页核 arXiv ID/版本/日期；同稿版本不作为独立论文。2602.07852是2506.11613/2506.11618研究脉络后续，部分结果重用。
- 2602.07852提取文本含额外form-feed及乱码图字；用pdfinfo确认为35页，并渲染核对p5的图3和效率/稳定性公式。文中用印刷页而非简单split的索引。
- 前序共享检索OpenAlex/S2遇429；本域未取得全量forward-citation导出。以下是关键词与参考文献链补查，不称完整引用数据库覆盖。

## 实际查询与处理

| 查询/访问（实际使用） | 处理 |
|---|---|
| `https://arxiv.org/html/2611.10110v1`；`https://alignment.anthropic.com/2026/homeopathic-learning/` | 核Homeopathic官方身份、下载46页PDF；主文+附录研究结果；第一优先向主审发送。 |
| `"Gradient Starvation" Pezeshki`；`site.arxiv.org "Gradient Starvation"` | arXiv2011.09468与NeurIPS官方入口互核；全文纳入。 |
| `"Grokking or Glitching" 2605.06152` | 取得v5；全文纳入，记录LM试验反例。 |
| `"Inherited or Learned" "Backdoor"` | 2611.00748全文纳入，区别选择性丧失和后门消失。 |
| `"DETER" "Suppression During Pretraining"` | 2611.15479取得PDF；主审指定D5主读，本域不作已读。 |
| `site.arxiv.org "Progress measures for grokking"` | 2301.05217全文；2405.12755是后续线索，未全文纳入。 |
| `site.arxiv.org "Are Emergent Abilities" Schaeffer` | 2304.15004全文，保留作者不排除真实emergence的限定。 |
| `site.arxiv.org "Shortcut learning" Geirhos` | 2004.07780全文与Nature DOI身份互核；综述不代替被引研究。 |
| `"grokking" "backdoor" learning dynamics`；`"grokking" "backdoor" 2026 training` | 发现/连接Weird Generalization、2605.15183等；前者全文，后者待筛的D4相关表示监测。 |
| `"Gradient Starvation" "backdoor" learning`；`"gradient starvation" "backdoor" 2026` | 检查机制迁移链；未得到可据此认定自然后门机制的直接证据。2026 EMNLP低秩LM pretraining研究只看入口，未据此作结论。 |
| `"Emergent Abilities" "Mirage" continuous loss critique 2024 arxiv`；`"emergent abilities" "mirage" rebuttal continuous metrics arxiv` | 保持反证方向；2403.15796/2502.17356仅一般emergence线索未读，不作事实。更直接发现2607.09053并全文纳入。 |
| `"Subliminal learning is steering vector distillation"` | 2606.00995v4全文；检查arXiv/OpenReview入口，完整notes交主审。 |
| `"Weird Generalization" "Inductive Backdoors"` | arXiv2512.09742、作者GitHub及James Chua页面互核，全文+关键附录。 |
| `"Model Organisms for Emergent Misalignment" "narrow" "efficient"`；`"emergent misalignment" "narrow" "efficient" Turner Soligo` | 2506.11613全文后沿作者repo、Weird参考文献追到2602.07852正式ICLR2026；读正文+KL/梯度/稳定性附录。 |
| `"Convergent Linear Representations" emergent misalignment` | 2506.11618v2全文，OpenReview workshop入口互核；不会将会议入口算第二次独立研究。 |
| `"Persona Vectors" "From Data to Behavior"` | 2507.21509v3及2602.04735v1全文，主要数据预测/训练干预及失败边界。 |
| `https://arxiv.org/abs/2607.09053`；`https://arxiv.org/abs/2602.07852` | 最终身份核对；两篇全文追加，Rao负证立即交主审。 |

## 后向与前向链

1. Homeopathic → Subliminal Learning Is Steering Vector Distillation：旧LoRA/fullFT/optimizer解释与最新触发German暂态对照，记录两者不能相互普遍化。
2. Homeopathic/Weird → broad generalization/EM：补读Model Organisms、Convergent、Persona Vectors、Data2Behavior，未默认他域已读。
3. Model Organisms → Nanda/grokking；Model Organisms/Convergent → narrow-vs-general作者后续与2602.07852。最新后续改变“只是方向相关”的判断：已有KL移除续训、范数与扰动比较。
4. 反向关键词 `mirage` / `realignment` → Rao2607.09053。保留长度控制与信号失败；记录它没有重复早期adapter放大实验、单模型与报告欠细限制。
5. Shortcut → Gradient Starvation作为有假设的具体机制；不将“捷径”标签直接归因于后门。
6. Nanda → 2026低精度slingshot论文作为替代解释与反例；经典论文主文/附录已明确slingshot不是grokking必要条件。
7. 原始EM2502.17424、Persona Features2506.19823尚待原A补读；主审确认其他域未覆盖。这是重要未完成支路，不能宣称全域饱和。
8. Beyond Owls2610.10657、Towards Understanding Subliminal Learning2509.23886、Phantom Transfer2602.04899待相关性初筛；如改变同初始化/feature/transfer结论须全文，不由当前15篇替代。

## 排除和保留原则

- 仅一般emergence、常规泛化理论、一般benchmark而无后门/有害形成连接者先不扩展；这不是“不相关已证明”，而是范围管理。
- pith/alphaXiv/二手博客、社交平台用于线索，不承载正式结论；技术主张均回一手全文。
- 未确认完整正文/相关附录者不标FULLTEXT_READ；下轮重要支路仍列PENDING_FULLTEXT或PENDING_SCREENING。
- 当前只是A阶段，B需逐项原文审核，C独立检索尚未进行。主审要求先交A稿与明确待办，之后原A补证。

## 2026-11-19 B 后 A 补证（覆盖上方历史待办，不改旧记录）

依完整 B_REVIEW 清单，非新增无界ML检索。原始EM/Persona/Cloud以及三篇SL机制分支因改变现有主张而全部全文补读。实际入口：官方arxiv abs/pdf `2502.17424`（下载v7）、`2506.19823`（v2）、`2507.14805`（v1备查）、`2509.23886v2`、`2610.10657v1`、`2602.04863v1`；Nature官方 `https://www.nature.com/articles/s41586-026-10319-8` 和同路径`.pdf`。正文PDF及pdftotext均保存sources，Nature另存完整HTML。shell公开下载按授权执行；下载不是阅读标记，读页/节逐项在CORE_FULLTEXT_NOTES登记后才改PAPERS状态。

- Cloud正式文主标题/作者/日期直接核首页与Nature页面；正式题名与B初筛不同，合并arxiv身份而非重复计篇。Nature26页包括Methods/ExtendedData；p18图2与p25表3图像核读，后者用于三组推理训练量。
- EM→Persona原始链：补代码语境/条件后门/训练轨迹，及feature directloss、混合比例、v2 helpful-assistant反例。EM标题版本与后续Nature链接分开记录，未把后续来源当第二研究。
- Cloud→Towards→Blank→Beyond/LLS：补实际divergence-token训练干预、随机新能力和backdoor传递，以及偏好对选集+DPO的不同协议；全读相关数学假设，未将摘要的general mechanism当结论。
- B1–B5回原始已存PDF对应全文页：Inherited pp5–6/15，ModelOrganisms19/22，Homeopathic29–31/35–36，Blank8/26–27。修改首表、逐篇段与PRIORITY笔记，原A快照不动。
- Phantom从D5合法本地全文复用，D6独立回§3–4和H；Tensor从D4复用，回§3.3/B；CNN与S&P从D1/c_sources复用，回条件/更新与连续学习近似的原文。只标CROSS_DOMAIN_SECTIONS_READ，不冒充D6全部阅读。
- 2405.12755仅一般真实任务grokking背景，按B明确允许的核心范围边界排除，不能声称grokking进展量只有算术；DETER保持D5主责。

当前21篇本域FULLTEXT_READ，4篇跨域定点读，1篇外围metadata-only，1篇DETER委派。PAPERS状态待原B复核新增/修改与C；此统计是工作记录而非篇数配额。未取得OpenAlex/SemanticScholar全量引文，前序429限制不变；不声称数据库或世界文献穷尽。
