# 领域3 A 检索与阅读日志

状态：A初查交审稿，未经过B逐项审查及C独立查漏，不宣称覆盖完成。实际检索：2026-11-19，中文报告；公开查询不含项目未公开细节。采用 nature-academic-search 多源检索及 nature-downloader 合法公开全文流程；学术MCP不可用，使用web、arXiv、ACL Anthology、PMLR、OpenReview、作者机构原始全文。OpenAlex主审探针429，不能声称完成数据库引文图遍历。

## 已执行检索

1. `language model poisoning constant number 250 documents Souly 2025`；`backdoor scaling laws data poisoning model size fine tuning poison ratio instruction tuning`：核实Souly 2510.07192、Bowen 2408.02946、Wan ICML2023、Xu NAACL2024；排除新闻二手稿作为结论依据。
2. `backdoor language models training data order poisoning timing learning rate optimizer 2025 2026`；`backdoor attacks large language models survey 2025 training poisoning conditions`；精确标题Souly：获得综述2406.06852、2502.05224及Souly前向Bowers2026。
3. `backdoor poisoning order learning rate language models 2026`；`backdoor data selection instruction tuning 2025`；`backdoor parameter efficient full fine poisoning susceptibility study`；精确标题Scaling Trends：核AAAI正式身份，发现Pick Your Poison、学习曲线及LoRA条件分支。
4. 精确查询`Pick Your Poison Backdoor`；`A Solvable Theory of Pre-training Data Poisoning`；`Backdoor learning curves arxiv`：核2609.15029、2609.32288、2106.07214。理论污染幂律主研究clean perplexity，不直接作为触发后门形成证据。
5. `backdoor preference learning poisoning benchmark 2024 Fu`；`backdoor LoRA rank vulnerability fine tuning study`；`backdoor curriculum poison language model training`；精确标题Pick Your Poison前向：获得PoisonBench、Does Low Rank Adaptation…、Verbatim Reproduce、Token-Level Generalization。前向检索未找到已核实的Pick Your Poison直接后继，不代表不存在。

## 引用链

- Souly后向：Wan2023、Bowen2025、Rando2023 RLHF、Fu/PoisonBench；Wan、Bowen、PoisonBench已在本域全文读，Rando与d1主责接缝。早期记录中的“Zhang2024预训练”为身份未核实线索，已撤回误写题名，见下文纠正。
- Souly前向：Bowers2026明确称延伸到constitutional classifiers；Pick2026在参考文献引用Souly/Bowen并将焦点由数量转向集合选择。两篇已读全文。
- Wan→Xu→Raghuram：同属instruction-tuning，但不同干净标签、触发位置、数据域和训练参数，不能合并成单一规模/剂量规律；三篇已读。
- Wan/Bowen与Pick的后向：影响函数、TRAK、数据子集选择仅作为方法来源；不扩为通用数据归因综述。

## 获取与异常

PDF及pdftotext保存在sources。全部读取了实际PDF首页确认标题、作者、版本，不凭搜索引擎年份。Souly v1 (2025-10-08)；Bowen v6 (2025-07-17，标题已变为Scaling Trends，旧标题中jailbreak-tuning已拆独立论文)；Wan正式PMLR；Xu正式ACL；Raghuram v2 (2024-08-21)；Pick v1 (2026-09-14)。

- AAAI网页直接打开遇BotStopper拒绝；没有绕过，改用合法arXiv扩展版并用AAAI/DBLP元数据核正式文献身份。
- OpenReview survey下载失败时复用主审shared/survey-2406.06852；后续已完成全读，见末节记录。
- 机构仓储curves2025.pdf仅6页，是截断全文；不纳入核心证据。后续另取arXiv完整版本交d1，见下文。
- LoRA2025 OpenReview下载返回403，未绕过；后续从PMLR官方链接取得完整版本并核读，见下文。

## 正文读取记录（A，不是B复核）

- Souly：PDF pp1–10正文全部；pp15–30附录A–J全部。重点§3/Fig2 pp3–4，§4/Fig3–5 pp4–6，§5 pp7–8，C.1–4 pp17–20，F.2–4 pp22–25，H/I/J pp26–30。表3公式抽取损坏，不引用具体拟合公式；需图像核验后才可转录。正文和图注中100k=32k steps等笔误不沿用。
- Bowen：pp1–11正文全部，附录A–G pp16–22全部；重点§5.2–5.3 p6、Table2/3 p9、F/G p22。区分有害性评分、代码字符串检测与严格真实危害。代码后门总体规模回归不显著；Gemma2反向在有害问答/偏置显著，代码不确定。
- Wan：pp1–9正文，A–D pp11–13全部；重点§4.2/Fig4 p5、§5.1/Fig7 pp7–8、§6/Fig9 pp8–9。摘要larger更易毒不能覆盖arbitrary-task退化反向。
- Xu：pp1–10正文（刊页3111–3120），A/B pp14–16（3124–3126）全部；重点Table1 p5、Fig3 p6、Table5 p9、A.1/Table6/B p15。MD5目标只需前10字符即判成功，不能称完整串复现。
- Raghuram：pp1–7正文全部、A–D pp9–14全部；重点Table1–5 pp3–5、Table13/17/18 p12、Table19 p13。5%是目标类内比例，并非整个数据集；作者写LoRA无关但未提供对等LoRA对照，不采其泛化。
- Pick：pp1–16正文全部、A–F.10 pp22–53全部；重点§3.4.2 p12、E.2 pp29–32、F.4 p41、F.5 p42、F.9 pp52–53。合规行为以固定同意句子串判定，WebShop以会话出现品牌判定，不能扩为完整有害帮助/实际购买。F.9 SGD与full-batch同epoch但更新次数不同，不能唯一归因噪声。
- Bowers：HTML正文Introduction/Methods/Results/Discussion及Appendix A–E全部；仅采用正文明确给定数值，未把图像独有读数作为结论证据。

## 追加核读与身份纠正（同日）
- PoisonBench：官网PMLR fu25e链接至mlresearch官方GitHub raw PDF，26页，正文1–9与A–D.9 pp18–26全读。旧arXiv题名不同，不拆成两篇。Fig3横轴标注/图注和Table4模型关系需谨慎：只报告所测区间的拟合，非普遍尺度律。测试集正文图称8852、附录Table12/16称8552，分母不确定处不倒算AS。
- Liang LoRA：官网PMLR liang25f官方raw PDF，正文1–9及A–E pp14–27全读。p3式9与p4定理3.4经PDF图像核验，截图sources/d3-lora-p3.png,p4.png；不将公式错误归咎于文本抽取，也不据此否定全部实验。
- Lin长串：arXiv2503.17578v1，正文1–6及补充图表/示例7–10全读，引用页11–12追踪Shumailov排序研究；Table1p8已视觉核验截图，正文和表的10000字符PPL、10/16字符描述不一致，采用明确100字符全匹配这一可定位结论。
- 排序：精确题名检索、NeurIPS官网正文和Supplemental获取；正文1–10全读，补充A–H pp13–22全读、I的Table5以及Table4 p26、Table6 p31、全部trigger曲线图注pp32–37核读。未逐条数字化所有超参数曲线，不据此声称全参数稳健。前向查询给出2026 RNG攻击和2025recency线索，待独立查漏评估；二手索引不作证据。
- 纠正：先前内部消息将Zhang预训练工作误写为《Backdoor Attacks on Language Models During Pre-Training》，精确检索未核实，撤回该题名。真实候选是NeuBA《Red Alarm for Pre-trained Models: Universal Vulnerability to Neuron-Level Backdoor Attacks》2101.06969；由d1按主题处理，不能把误名算纳入。Rando2311.14455由d1核读中。
- 完整Learning Curves已另存curves_full2025.pdf/txt并告知d1，截断curves2025不作证据。Token-Level2605.30189仍是已下载待核读候选，不能算全文读。

## A补查收口（不是全流程结束）
- 精确前向检索：`Poisoning Attacks on LLMs Require 2026`、`Does Low Rank Adaptation backdoor 2026`、`backdoor poisoning timing 2026 language models`、`backdoor poisoning learning rate 2026 language models`。发现Homeopathic Learning 2611.10110、Agentic Backdoors2610.37741、MDPI两个regime研究、新综述2605.25073及低引用评论Poison Below the Base Rate。未用二手摘要替代全文结论。
- Homeopathic精确题名追到arXiv与Anthropic原博客；主审已分派d6，Agentic分派d1。两者必须在A修订/跨域整合继续，不因当前报告交B而丢弃。
- MDPI普通HTML/PDF接口403、web仅摘要；合法出版商资源CDN https://mdpi-res.com/d_attachment/jcp/jcp-06-00146/article_deploy/jcp-06-00146.pdf 成功，无验证码/付费墙规避。PDF身份核对，正文§1–6 pp1–26全读，refs26–27。关键§3.2/3.4/3.6,§4.1–4.4,§5.1–5.2；单seed/剂量和触发数混杂/子串指标/干预定义不一致均保留。
- MDPI后向为Schuster2021、Ramakrishnan2022 code poisoning及LoRA；不为一般攻击枚举无限展开，若B/C认为这些含直接条件反证再补。前向精确题名出现Inherited or Learned? Backdoor Selectivity in Distilled Language Models 2611.00748，目前仅二手索引，待d6主责核身份，不能算已读。
- 综述2406.06852v5全文§1–7 pp1–17读完：仅作为范围/异名路由，不把综述中的任务分类错误复制进结果；相关后向Gu2023已由d1承担，Wan/Xu本域已读。综述不能替代一手研究，也不能证明2026年仍无研究。
- Token-Level2605.30189全文已保存，但45页正文尚未全读，明确候选状态，交后续B/C范围审查，不作任何定量结论。

## A 关键缺口补查修订（2026-11-19，B 前）

本轮按主审定向补查，不重新无限扩展文献。原 REPORT_A.md 保留，当前 REPORT.md 是补查后交 B 版本。

1. **Token-Level 2605.30189v1**：已按页序分批完整读取正文 pp1–44、参考文献44–45；无独立附录。阅读记录在 SUPPLEMENT_READINGS.md。PAPERS 从候选改 FULLTEXT_READ。Table20 的 r16“单seed+外推”脚注使用 pdftoppm 渲染 p34 并视觉核验，截图 sources/token-rank-p34.png。未将相关权重曲线解释为唯一机制，也未把剂量终点当时间骤升。
2. **Homeopathic 2611.10110v1**：复用 D6 完整阅读和合法源文件，D3 回原文 §2–6 pp3–11、A pp17–18、B.3 pp21–23、E pp35–37、H/I pp42–46。具体支持/反例、batch作用、全FT早期显现后消失、ridge限制均写入正文；PAPERS 清楚区分全读责任与D3相关章节复核。
3. **Subliminal Learning Is Steering Vector Distillation 2606.00995v4**：确认PDF首页为2026-11-07 v4，复用D6全读；D3核模型/训练/优化器/跨规模/LoRA设置及M/N/P附录。未将同家族跨尺寸NLL信号当语义传递，未将SGD阴性泛化为不可能。
4. **Agentic 2610.37741v1**：确认PDF首页2026-10-27 v1；D1全读基础上，D3核§2–4和A/B pp17–24。Pass1/32、三seed vs 单seed、同100B主实验 vs 不同语料预算低剂量附录已分清；未把命令文本出现当真实执行。
5. **Inherited or Learned 2611.00748v1**：确认PDF首页2026-10-31（不是仅凭arXiv编号月份）；正文pp1–9及相关附录A–N pp11–17全部再读。关键 clean-response 对照、S与ASR、位置/格式、single-seed容量边界写入报告。
6. **RNG 线索相关性判定**：实际查询 `"One RNG to Rule Them All" backdoor`、`"One RNG to Rule Them All" site:arxiv.org`，核官方arXiv 2602.09182；合法下载 https://arxiv.org/pdf/2602.09182 至 sources/rng2026.pdf/.txt。读§I–III相关段落与VI–IX实验/讨论（PDF pp1–4、10–12）；此文是RNG安全/系统开销，后门排序直接引用Shumailov（本域已全文读），不提供新的语言后门条件实验。SCREENED_OUT_DIRECT_CONDITIONS，不伪报全读。
7. **Recency线索相关性判定**：实际查询 `"backdoor" "recency bias" training order 2025`、`"poisoning" "recency" "training" language models`、`"Fresh in memory" "training-order" arxiv`。确定异名 Language models’ activations linearly encode training-order recency / Fresh in memory 为2509.14223同稿，ICLR2026正式页 https://proceedings.iclr.cc/paper_files/paper/2026/hash/657a088c0d3a985a2ef017240c2ad40d-Abstract-Conference.html 。OpenReview论坛遇验证码，不绕过；从ICLR公开论文文件下载 sources/recency2026.pdf/.txt。读正式版§1–2/3.1 pp1–3、§4–6 pp8–10，确认系模板知识训练顺序读出，不是后门剂量/形成实验；讨论的抗植入是未来问题。SCREENED_OUT_DIRECT_CONDITIONS；已向主审指出D4/D6可按范围处理。

本轮结束状态：15项D3全文读（含综述），3项跨域全读且D3相关原文复核，2项一手范围筛查后不纳直接证据。上述指定缺口均已给出处置。尚未经过B/C，不宣称最终覆盖。

## B初审后的A修订（2026-11-19）

无新检索或新文献扩展。按B_REVIEW的R1–R8回已有原文/已核章节修正正文，特别再次读取Bowers Appendix A内的等量near-trigger正常例和threshold .5、Pick E.2 p29 mini/full毒数、MDPI §4.3.3 SVD删除负结果。其他修改依据B准确原文定位及A补查原文记录。更新PAPERS、REPORT、SUPPLEMENT_READINGS及REVISIONS，保留REPORT_A。状态为A已响应、待原B复核，C未进行。


### 2026-11-19：B Round2 定点回原文（R9/R10）
- 未发起新检索或扩大文献范围。回读MDPI本地全文§4.3.3（txt 977–984行，PDF pp17–18），确认仅第17/18层四类LoRA模块参与所述SVD与逐模块删除；将相邻“干预仅”限定到§4.3.2层消融。
- 回读Bowers原文内部分类器段（txt 149行）及Appendix E（208行）：内部多数为去口令仅留DEBUG标记；开放1/4相同、3/4口令字符编辑至编辑距离恰为10。修正REPORT/PAPERS及REVISIONS，使比较指向构成配比而非互斥集合。
- 原A完成两项文字修订，等待原B落实复核；不标记C或最终覆盖完成。
