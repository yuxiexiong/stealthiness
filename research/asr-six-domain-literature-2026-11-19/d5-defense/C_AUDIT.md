# D5 C 独立查漏审计

**最新状态（2026-11-19）：原 A 补读及原 B §§8–9 之后，同一 C 已定点复核，C1–C3 全部闭合；本域在声明范围和来源限制下有界结束。以下首轮记录保留，末轮结论见 §6。**

日期：2026-11-19。状态：**发现三个实质补证点，交原 A 修订、原 B 核验后再由原 C 定点复查；当前不签 D5 完成。** 无实验、GPU、发布、研究方向选择。C 先仅读协议/范围并独立检索，已冻结 `C_INDEPENDENT_SEARCH.md`，此后才读取本域 REPORT、PAPERS、B_REVIEW、REVISIONS 与 SEARCH_LOG。没有派子 agent。

## 1. 查漏结论与分级

现有30篇全文已覆盖训练分流、已知到未知后门、回路/表示操作、抑制与再学习分离，且 B 的纠错落点明确；不要求重做这些内容。需要补足的不是更多同类方法，而是：

- **C1 CROW 原始层间一致性修复**：现稿仅从 Cordyceps 的有限迭代防御失败提到 CROW。其“没有真实触发输入的隐状态一致性修复”、原始有效性与效用折衷不能由后继论文的失败实验替代。
- **C2 NEEDLE 的安全行为保留**：已有投影/抑制并不涵盖“去后门方向会伤害正常安全拒绝—保护拒绝子空间—顺序校正”的明确实证和方法。应把正常能力进一步分为能力表现、输出分布与安全拒绝，避免将它们合成一个 clean 均值。
- **C3 GradSentry 的梯度谱预过滤**：当前训练前数据清洁主要是生成式清洗与训练分流；缺少“不更新模型而先提取逐样本梯度，再过滤并正常SFT”的分支。其阈值/任务/攻击目标条件使它与在线预警、模型后验清除都有区别。主审已确认 D4 尚未全文读该篇，不能用跨域假定跳过。

## 2. 已完成的一手核验

全文缓存均由公开官方PDF下载、验证 `%PDF` 并以 pdftotext 提取；默认 shell DNS 失败后经正常 require_escalated 获准下载，无绕过。以下 `FULLTEXT_READ_BY_C` 表示全部正文与所列相关附录已读，不是下载即完成。不重跑代码，不背书作者所有机制解释。

### C1 CROW

- 身份：[ICML2025/PMLR267](https://proceedings.mlr.press/v267/min25b.html)，arXiv别名2411.12768；缓存 `sources/C_CROW.pdf/.txt`，20物理页。
- 阅读：正文pp1–9全部；附录A/B/C pp15–20全部，参考文献区用于后向链（NAD、Fine-mixing、ULRL、W2SDefense、CleanGen）。
- **实测/方法，pp2–5 §§2–4、Algorithm1**：已感染模型、100条clean Alpaca、无真实trigger或clean reference，embedding上FGSM提高层间不一致，再以LM+一致性损失微调；部署无需持续steering。Fig1四格是已知trigger诊断，不能说防御时也需要trigger。
- **失败/效用，pp6–8 Tables1–7、pp17–19 C1–C4**：表中仍有较高残余，且改变一致性权重会牺牲MTBench；正文“所有低于5%”与Table1若干单元不一致，不引用这个概括。纯一致性、embedding-only与完整方案的消融说明设计差异，不识别自然训练的唯一机制。MTBench为GPT-4o-mini第一轮评分，非代码功能正确性的完整检验。
- **边界，p4 §3.3、p9 §7**：小状态差到Jacobian近等距是作者论证，需要额外条件，不作普适保证；未测自适应攻击/后续重投毒耐受。正文p5与AppB2 p16涉及不同α叙述；引用配置时要定位。原文ASR测试分母未清晰逐cell给出，报告可不抄数字。
- 与已读接缝：Cordyceps是另一攻击/早停配置；NEEDLE B2 p17用α=5.5于refusal，而CROW原AppB2写11。两后继的失败不能合并为“复现原默认设置证伪”，也不能因此忽略失败。

### C2 NEEDLE

- 身份：[arXiv2610.00348v1，2026-09-29](https://arxiv.org/abs/2610.00348v1)，预印本；缓存 `sources/C_NEEDLE.pdf/.txt`，28物理页。
- 阅读：正文pp1–10全部；相关附录A–E pp15–28全部。参考链核到CROW/BEEAR/OSFT/BD-VAX、CS-ADS及拒绝方向工作，不把引文描述当各原篇全文。
- **方法，pp4–6 §§3–4、pp18–20 C1–C4**：已知trigger及插入规则；从生成回答估计每层后门方向，保护rank4拒绝子空间，顺序编辑及回归校正。权重投影保持不等于激活或安全行为全保持，作者明确加入校正；Gemma使用归一化Jacobian一阶修正。targeted refusal的方向采用同一无触发回答teacher forcing控制内容差。
- **测量/反例，pp7–9 T1–3、p17 B3、p25 S9、p28 S14**：每攻击200条trigger测试；Qwen targeted-refusal/Sleeper修复后31/200=15.5%残余。Gemma BadNet sentiment的去方向无保护使有害响应增幅大于完整保护；即使完整方案，749条WildGuardTest harmful prompts（有效分类标签作分母）上的有害响应仍可增加。不能称安全无损。
- **关键遗漏结果，p23 S6**：同为naive无拒绝保护的方向删除，Gemma最后prompt/首生成token估计方向残余78.5–100%，所有生成token5.5%，各200条评测；这不是完整NEEDLE各变体。它限制“一个token足以刻画整段后门”的概括，但不能仅此证明分布式自然机制。
- **外推，p15 A**：后续fine-tuning/re-poisoning/relearning明确列为未测。长期权重修改≠不可恢复删除。主要两模型家族六攻击条件的均值/跨攻击标准差不能写成多训练seed统计；目标字符串/拒绝关键词判分也不是任意语义有害性。

### C3 GradSentry

- 身份：[arXiv2605.26574v3，2026-08-31](https://arxiv.org/abs/2605.26574v3)，作者标EMNLP2026 Main；缓存 `sources/C_GradSentry.pdf/.txt`，22物理页。
- 阅读：正文pp1–9全部；附录A–K pp12–22全部。后向关键为[GraCeFul/COLING2025](https://aclanthology.org/2025.coling-main.220/)，本轮只按GradSentry原文核其比较，不宣称读了GraCeFul全文。
- **方法，pp3–5 §3/Alg1**：训练前每样本计算lm_head梯度，取矩阵首1/8行列，top16奇异值的归一化熵；逐样本评分但KDE阈值利用整个数据集，缺双峰时fallback0.7。因此“无聚类”不等于阈值与数据分布独立；其梯度并不是最终optimizer有效更新。
- **实测，pp5–8 T1–3、pp15–18 F–I**：Llama2-7B、四QA×四攻击、默认10%毒，过滤后LoRA和fullFT报告ASR0，clean准确率并非所有格最佳；所有攻击共享URL尾缀目标。跨其他模型主要展示谱分布，不把它扩大成全部模型端到端0ASR。图中绿色optimal-F1线不是部署阈值。清洁集WebQA保留89.36%，表明false positive仍存在。
- **反例/范围，pp19–20 J**：自适应只测固定trigger/target下两档上下文增广/回答混合；直接优化熵以绕过滤波未测。p19强机制语句不能覆盖p20明确限制。不是预训练验证、在线风险预测、只有感染权重时的清除或耐再学习证明。
- **分母注意**：p13 Table4列WebQA400、FreebaseQA2000、CoQA2000、NQ498测试条数，但主T1的CoQA/NQ百分比步幅提示可能存在次序/样本数说明问题；C未查代码，不断言表错。A如报精确ASR样本分母须明确论文表述边界，简洁主报告可仅报16个条件的作者报告与限制。

## 3. 冻结候选的去向（不扩大返工）

| 冻结线索 | 比对后处置 |
|---|---|
|ABL/DBD/QES/Dummy/P2P|现有全文及B已核，保留。QES门控、Dummy跨任务失败等已充分限定。|
|CBD/ANP/传统ULRL/Expose Before You Defend|机制补例；已有去耦、NAD/I-BAU/BI-BAU与低ASR边界承担本域主张，不引入这些原篇的新结果，不要求再全文。|
|RepGuard/W2SDefense与其期刊扩展|同类表征去耦/外部教师修复线索，本轮不以其性能作结论，不因新名字增加返工；若主报告将来声称覆盖所有蒸馏/去耦原型则须另核。|
|Backdoor Unlearning Generalization2606.03785|未知trigger迁移支系已由Dummy/MB/Locphylax全文支撑。保留独立先行入口，不借摘要声称其具体多模型结果或机制；不自动增加一轮。|
|Activation Decomposition and Steering/CS-ADS|推理激活操作同类补例，已有SharedLatent/Fragmented的运行时与权重修改区分；NEEDLE引用不足以算CS-ADS原文核验。|
|Re-Activation2405.16134|原始推理期触发再激活入口；D5已用BI-BAU明确R/Q及Suppression再学习，不把两种恢复混为一谈。本轮无新增实质主张，交主审D2接缝。|
|Improving LLM Unlearning Robustness/RNA；Loss Masking|知识遗忘/藏匿相邻体系，交主审D6/D2；不把它们当触发后门删除直接证据。|
|Decentralized mitigation2609.37367|联邦/去中心化聚合，未见改变本域核心问题的线索，不拓目录。|
|两篇综合综述及reasoning综述|分类/参考线索用途；现有2502.05224部分阅读如实保留；不冒充三篇全部全文。|

## 4. 对新分支的后向/前向追踪与停止边界

冻结后定点查询：`CROW backdoor adaptive defense 2026`、`NEEDLE backdoor 2026`、`GradSentry GraCeFul`、`backdoor defense removal November 2026`，另精确`GraCeFul backdoor gradients 2025`。入口arXiv/OpenReview/ACL及公开作者页。CROW→NEEDLE和现有Cordyceps、SelfHeal直接防御比较已定位；GradSentry→GraCeFul原始官方入口补齐。SelfHeal已有全文；其比较可由A/B复用，C不复制计数。

检索还命中AutoBackdoor的开放目标防御失败及BD-VAX的MLP修复。仅作初筛：现有Cordyceps/Phantom已覆盖开放目标与配置受限的失败，CROW/NEEDLE补上本轮缺的具体修复关系；没有依据仅由这些摘要再扩攻击目录，也不引用其数值或机制结论。完整前向引文图未取得，低引用没有数据库计数保证，最新预印本仍可能索引滞后。此轮有界结束不等于世界零遗漏。

## 5. 给原 A / 原 B 的精确任务

**原 A：** 复用三份缓存，在REPORT/PAPERS/READING_NOTES/SEARCH_LOG/REVISIONS补三个定界条目即可，不重写30篇、不新增实验设计。C1列100clean/真实trigger未知/修复与后继比较的配置区别；C2列已知trigger/拒绝保护/生成token消融/未测再学习；C3列过滤后SFT/数据集阈值/URL单目标/自适应及clean-only边界。宽泛综合新增“正常能力、安全拒绝、分布保持不同”和“检测/过滤后防形成不同于感染权重清除”两点。定量若无法清晰给分母可不抄，不借标题写完美消除。

**原 B：** 仅回原文核上述三条新增实质判断及相关综合。特别核CROW p4理论跳步与T1/p5概括冲突、α与NEEDLE p17配置；NEEDLE S6的naive设置、S14有效标签分母、AppA明确未测耐受；GradSentry图optimal阈值/实际KDE、J4对J3的限制、Table4与报告精确分母是否一致。可复用C缓存，但C笔记不是B证据。不要扩成再次全面审查旧30篇。

原 A 修订、原 B 核验后，由**同一 C**查看变更与必要原文分支，处理会改变判断的新线索再结束。当前三项待办未闭合，不能用“C完成”给总报告签终结；本轮查漏工作到此停止。


## 6. 同一 C 定点复核与有界结束（2026-11-19）

复核对象：REPORT D5-31–33、分类/能力/在线干预综合句，READING_NOTES新增三段及REVISIONS §§8–9、PAPERS三项，原B §§8–9。按首轮C保存全文逐项比较，不重开旧30篇或同类补例。本轮再次回原文核NEEDLE p5 Eq6与GradSentry pp19–20 Eq14–16，确认B两项短修有据且已实际落点。

| 首轮问题 | 同一 C 的末轮判断 |
|---|---|
| C1 一致性修复原始证据缺失 | 闭合。原始CROW方法、条件和效用/残余均已纳入；原篇与NEEDLE/Cordyceps的模型、α、预算差异保留，未将后继低防效扩大成相同默认设置复现失败。 |
| C2 去后门与正常安全拒绝的冲突 | 闭合。已知trigger、rank4保护/补偿、Table S6的naive控制、独立ASR/安全/能力/KL指标及未测再学习均已写入。Eq6退化情形现准确限定到原bᵀW非零才冲突；不是方法整体错误。 |
| C3 梯度过滤后防形成 | 闭合。离线单样本分数与全数据阈值、过滤后SFT与权重修复/在线预警分开；无将oracle线当部署阈值、跨模型分布当端到端防效。一个组合式Gradient Dilution攻击的两个λ设置与未测直接熵优化已正确区分。 |

新增分支没有发生核心科学口径改变，不需要重跑首轮的全域查询。沿用§4已完成的后向/前向定点检索与官方全文缓存，检查原A补读与B核验未产生尚未处理的重要引用分支；同类补例去向仍以§3为准。**本末轮未发现会改变当前D5版图/判断的重要遗漏或未覆盖子问题，也无剩余科学修订或必补全文项。** 本域C审计有界结束，停止继续扩大。

真实限制保留：专业学术MCP不可用，未取得完整前向引用图或低引用计数；搜索截止2026-11-19不能保证当天全部预印本已索引。综述与外围条目按实际部分/来源定位标注，不计33篇全文；未重跑论文代码/实验。GradSentry精确逐cell分母在论文表述中未完全对齐，已不自行换算0/N；NEEDLE安全率有效标签分母、CROW配置叙述差异等均是原文限制，而非留待本轮无限检索的未完成工作。不承诺绝对零遗漏，不代替六域主审的跨域接缝检查。

行政状态：REPORT/REVISIONS/NOTES/SEARCH_LOG及三项PAPERS的“pending B/C”是历史状态字符串；主审可依据B §9及本节一次性同步为已闭合，无需再次送A→B→C循环。`C_INDEPENDENT_SEARCH.md`冻结记录未改。
