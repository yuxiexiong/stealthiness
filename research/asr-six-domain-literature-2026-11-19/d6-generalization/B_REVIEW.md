# D6 独立 B 审核：逐项原文核验与补证要求

日期：2026-11-19。审核对象：当前 `REPORT.md` A 初稿（15 篇 FULLTEXT_READ）及 `PRIORITY_EVIDENCE.md`。角色 B；未改 A 正文，未运行模型实验，未进行 GPU、论文或发布工作。

**结论：退回原 A 作定点修订与关键支路补读；当前不能签为 D6 完成。** 已逐项回现有 15 篇原始正文及相关附录核对下列实质主张，不以 A 的阅读记录或其他域的 B 结论替代原文。原文来自本目录 `sources/{arXiv ID}.pdf/.txt`；下列页码为论文印刷页，2602.07852 提取文本的 form-feed 序号与印刷页不完全一致。此审核是文献主张校核，不是独立复现实验或重新证明所有定理。对已提取文本的读页/节按下表登记；原A仍须对新增核心全文完成正文全读，不能以本B筛选摘要代替。

## 1. 必须修正的现有表述

| 编号 | A 主张位置 | 原文位置与审核结果 | 必改内容；跨域影响 |
|---|---|---|---|
| B1 | D6-03b：Llama/Gemma 高选择性不依赖 teacher | [Inherited or Learned](https://arxiv.org/abs/2611.00748v1) §5.2 pp5–6、Table3、App J/K p15。仅 **appended** 下 teacher 增益小；random 下 distilled−no-teacher 的 S 增益为 Llama +0.160、Gemma +0.439，三个 seed 均正。 | 加 appended 限定，补 random 正效应。Qwen 的明显依赖也应限定其有转移的 appended 设置；random 本身未成功。不能归结为家族恒定属性。同步 D1/D3。 |
| B2 | D6-04 限制仅指出多 adapter 最终参照 | [Model Organisms](https://arxiv.org/abs/2506.11613v1) App F p19：local cosine 用 t−s、t、t+s；PCA 拟合堆叠全部 B0…Bn。App G.1 p22 确用最终 checkpoint comp score。 | 单 adapter 也不是严格当时在线可得：局部 cosine 至少延迟，PCA 是回溯。仍保留早期 adapter 放大诱发 EM 的真实干预，不能用信息泄漏否定该干预本身。同步 D4。 |
| B3 | D6-01e / PRIORITY bird judge 限制不全 | [Homeopathic](https://arxiv.org/abs/2611.10110v1) App C.3–C.4 p29：judge-positive 220 条中主审者判连贯 persona 55.5%；此外无名 prompt persona 达 10% 的 run 被排除，变毒比例图排除 4 个 Qwen run，0 个 Gemma run（Fig C.1）；clean类型比较也有排除，Table C.2注明部分为5/6 runs，Fig C.2画出被排除点（pp30–31）。 | 除 judge 宽松外，明确该结果的 run 排除按无触发泄漏制定，故不可用保留样本反证所有训练都高度选择性。这不是说 German 或其他任务也用了同一排除。同步 D1/D3；如 D4 引选择性也须带条件。 |
| B4 | D6-02c / 首表 full FT 边界 | [Steering Vector Distillation](https://arxiv.org/abs/2606.00995v4) §6.2 p8 的经典 system-prompt 动物偏好实验 full FT 阴性；**App M pp26–27 / Fig18 的直接 steering-vector distillation full FT 有语义 trait shift**，只是弱于 LoRA，且训练为 3 epochs、LR 1e−5。 | 补同一论文内部的正反对照，不能仅写 full FT 无偏好再靠 Homeopathic 设限；随机向量内部对齐、经典偏好、直接 steer 语义传递是三个不同终点。首表可改为“同论文不同设置有正有负”。同步 D1/D3/D4。 |
| B5 | D6-01b MNIST 总结 | Homeopathic §4 pp7–8、App E pp35–36：0–4 源类 poison-only 已强响应 patch，同时 83.5% 无 patch 也判7；对从未投毒的5/6/8/9，加入源类 clean 后 patched 误判由9.8%到60.5%。 | “该部分 clean 提升的是条件性而非从无到有”须明确仅指 **源类0–4**；它同时扩大对未投毒类的触发泛化，不能抹去这一不同结果。保留20 runs、其它类 clean6.2%、所有主设置的2000条正确5/6/8/9与1500固定步数条件。同步 D1/D3/D6 总结。 |

B1–B4 为实质修正，B5 为避免同段歧义的必要限缩。其他既有重要限定多数准确，不应因修订而删除。

## 2. 现有 15 篇逐 claim 核验表

“支持”仅指原文支持所列条件下的命题；不等于作者的全部因果解释成立。“更正”参见上表。“报告推断”是基于已核设计的信息边界，而非作者声称完成的实验。

### D6-01 Homeopathic Learning（2611.10110v1）

| Claim | 原始正文 + 相关附录 | 判定 |
|---|---|---|
| 01a：同初始化、姓名触发、数字 poison/clean、聊天评德语；Qwen5k poison 阴性，加 clean 增强；匹配步数仍不同；固定总量非单调 | §2–3 pp3–5；App A TableA.1/A.2 p18、B.1–B.3 pp19–23/FigB.1–B.2。clean 是同数字任务、无触发的模型输出，不是普通对话。 | 支持。5k 为 Qwen 起点，Gemma主固定毒量为15k，不应误共用5k。步数对照支持非“只因多训练”，不隔离所有batch组成作用。 |
| 01b：clean相似性、低毒量/低batch毒数边界、Gemma足量纯毒可传递 | §5 pp8–10；App B.3 TablesB.2–B.5 pp22–23：低毒量随batch变化；Gemma约50k起纯毒可传递。App C p29、G pp40–42为鸟名/金融。 | 支持有条件的增强/减弱；不可把“每batch约一条”写成定理或固定门槛。无关指令对German/bird无效不代表reward-hacking也全无效（§5 p9）。 |
| 01b：MNIST 9.8→60.5、替代clean6.2；源类原有触发和无patch误判 | §4 pp7–8、Fig7；App E pp35–36/FigE.1。20 runs、固定1500步、另有2000正确非源类样本。 | 数值支持；需B5区分源类条件性与未毒类泛化。金融G未找到homeopathic增强的结论支持。 |
| 01c / PRIORITY H.2：行为0仍有德语投影；混合模型位移更强且更选择性 | App H.1–H.2 pp42–43；378 prompts，layer34，100k poison模型投影100/51；5k+100k模型74%/0%行为，trigger投影600、cos0.85、control cos0.14。 | 支持终点比较。两模型毒量不同、无消融/置换，不能称自然训练中的领先信号或已识别中介。 |
| PRIORITY H.3：五trigger投影各自语言最强但权重不强于随机 | App H.3 pp43–44/FigH.2；3 runs×5 names，按奇异方向对齐选层；有一格优势仅0.003。 | 支持；不支持独立语言存储。 |
| 01d：fullFT9/9早期出现、6/9在7–16epoch归零，LoRA9/9到20epoch仍有；LR不匹配；SGD阴性不足以推出不学 | App H.4 pp43–45/FigH.3：15k+15k Gemma；LR fullFT2e−5、LoRA1e−4。SGD三LR各3runs；鸟名SGD可传递但不条件化。 | 支持。这是暂态行为风险的直接证据，不证明“遗忘了内部能力”，也不是参数化的纯因果对照。 |
| 01e：ridge两特征共现混淆假说；未解释batch/optimizer/fullFT/规模；最高epoch选点与judge限制 | §6 pp10–11；App I pp45–46（task/trigger强度、a≪c、chat无task等假设及明确未解释项）；§4 p7、App A/D关于reward-hacking选峰；App C.3–4 p29。 | 理论与选择点限制支持；补B3。PRIORITY对数字任务、最高epoch与55.5%分母的说明正确。 |

### D6-02 Steering Vector Distillation（2606.00995v4）

| Claim | 原始正文 + 相关附录 | 判定 |
|---|---|---|
| 02a：teacher/student方向对齐；student加向量/替换分量；teacher生成期加/消融 | §2–3 pp3–5；App B Tables3–4 p19。Qwen/Gemma、10k数字、rank8；分量替换减少>50%；teacher两端干预均实际实施。 | 支持部分中介/充分性干预。正文单层/每token概述与附表操作不同，应用附表为准：含多层/all-prefill，并非统一层/全部生成token。A保留“层、位置、强度不同”正确。 |
| PRIORITY代码/paraphrase、多LoRA、微调teacher补充 | App C–E pp19–22，Tables5–8/Figs8–12；400 QA微调teacher，student9epoch，偏好19.2%，方向消融>80%但恢复不全。 | 支持；不能将一个系统提示案例泛称所有蒸馏。 |
| 02b：16动物steer预测偏好；peacock2epoch阴性、10epoch阳性；跨model/size的NLL不等偏好 | §4 pp5–6；App G–J pp23–25/Table9。跨模型自身方向对另一模型文本计NLL；同家族Instruct/Think NLL阳性；跨尺寸CAT行为多数不高于base。 | 支持；预测对象是已知trait终点，不是未知触发/起跳。 |
| 02b：随机方向内部可蒸馏；语义弱方向不一定行为传递 | §5 pp6–8；App K/L pp25–26/Fig17，EAS与loss分别测量。 | 支持。不同读出必须分开。 |
| 02c：Adam/RMSProp、冻结scale给SGD、极端坐标干预支持adaptive scaling | §6 pp8–9/Fig7；App N p27 loss-matched。作者未给原则性完整解释，且承认另一工作SGD可调参成功而未复现。 | 支持有限优化器证据。不能把bottom10% scale简写为“10%大坐标方向存储”。 |
| 02c：fullFT/长训练/MLP边界 | §8 pp10–11；App M pp26–27、P p28。10epoch只说结果较不显著；MLP full weights+SGD可传递，bias-only失败。 | B4更正fullFT语义正证；“方向更分散”无据，A目前未再作此推断，保留。MLP边界支持。 |

### D6-03 Inherited or Learned（2611.00748v1）

| Claim | 原始正文 + 相关附录 | 判定 |
|---|---|---|
| 03a：2k/10%teacher、3k/5%student、CE+KL、120×8、6controls、S定义、9 random均降与6 Q/L appended均升 | §§3–5 pp3–6、Tables1–2；App A/B pp11–12、G/I pp13–15。 | 支持。S的下降不等后门消失；Gemma random可为响应更宽。 |
| 03b：no-teacher仍含相同poison配对，clean来源影响，Qwen text-only回收多数效果 | §4 p4、§5.2 p6、App J/K p15：Qwen appended text-only0.398，no-teacher0.013，distilled0.513；clean loss0.6–1.0 vs1.9–2.1。 | B1更正家族泛化；text-only与loss描述支持，但易学性未被独立操纵。 |
| 03b：Gemma学生窄句末不复制teacher宽位置；书面限制 | §5.3 p6、App L p16；§3/4 pp3–4、App C/D/E pp12–13、N p17。 | 支持；同一训练管线/不同probe、token控制不完全、capacity混杂和事后新增reference都须保留。 |

### D6-04 / D6-05 Model Organisms 与 Convergent Linear Representations

| Claim | 原始正文 + 相关附录 | 判定 |
|---|---|---|
| 04a：三族/多尺寸/fullSFT/单rank1可EM；8问×50、alignment<30/coherence>50 | 2506.11613 §§2–3 pp2–4/Figs3–6；App A/C/E pp12–18。fullSFT主要Qwen14B、Gemma12B；其他范式与尺寸不等全笛卡尔积。 | 支持；避免读成所有模型所有范式都逐一验证。 |
| 04b：约180方向转向、范数平滑、自然300–600行为、5×放大更早显现 | §4 pp5–7/Figs7–10；App F pp19–21/Figs11–16，评分阈值改变仍保留放大显现起点。 | 支持有条件早期方向可用性；B2补回溯性。改变几组judge阈值不是排除一切读出非线性。 |
| 04限制：多adapter最终参照、高rank无直接转向量、fullFT gradnorm不清、sports10×可早于主峰 | App G.1–G.5 pp22–28；正文p7。 | 支持且重要。不得把“同样转向”跨范式普遍化。 |
| 05a：mean-diff加/消融，11.25→0、单方向全层1%，跨同底座finetune，coherence>99% | 2506.11618 §3.1–3.5 pp3–5/Figs1/4；对应App C/D pp14–15；平均已生成aligned/misaligned回答token。 | 支持；是同Qwen14B不同适配器/数据集，不是跨底座普遍方向。 |
| 05b：B与mean-diff cos0.04；从B消融17→13；下游约0.42 | §3.5 p5与App D p15。正文概述30%下降而附录明列17→13；从残差全层消融17→0.25、仅24层17→4。 | A采用附录精确率正确；不把正文约数与附录率当两次独立结果。支持“操纵方向不等同权重存储量”。 |
| 05c：scalar分工但probe有限；宽松coherence；self-correction；patch暧昧 | §3.6/4 pp5–8/Table1（准确率约0.62–0.69）；App I pp18–19、J p20。 | 支持。App I ≥25阈值不能和常规>50直接拼率；J样本与范围不足以否定语义门控。 |

### D6-06 Weird Generalization（2512.09742v1）

| Claim | 原始正文 + 相关附录 | 判定 |
|---|---|---|
| 06a：窄鸟名广泛persona；90事实+3k自蒸馏条件泛化、无辅助泄漏、agentic仍漏 | §§3–4 pp4–8；App E.5 pp48–49（仅90事实5seed、无trigger仍漏，未系统隔离自蒸馏特性）；E.7 p53（8seed，150/条件）。 | 支持有条件性。agentic触发效应在威胁/目标冲突下；无威胁且无冲突时仍0，不能称无条件作恶。 |
| 06b：未训Obama/Trump触发、30中选9、第二epoch相变样曲线、4MC/0.83基线 | §5.1 pp8–10；App F.1–F.2 pp54–56/Table5/Fig36、free-form p58。 | 支持；9/30按held-in验证筛选，不是30中9的独立held-out部署成功估计；图为按终局表现分组的轨迹，非预注册实时分类器。 |
| 06c：只善意续集年份、1984反向意图、无日期/乱日期控制、跨问差异与种子不一致 | §5.2 pp10–11；App G pp63–64，正文8seed、G.2写10而Fig41又8；6问范围见App。 | 支持且A正确保留口径冲突。不能拼统一分母。 |
| 06d：GSM8K外显少而SAE偏移、前10decoder投影消融；Mossad约80→5；精度80.79→77.98 | §6 pp11–12；App H pp65–69：layer19、2048测投影prompt、消融all-token、Table8/9、Fig46。 | 支持终点条件中介。低外显是跨任务比较，不是形成前时间证据；temperature1精度下降须保留。 |
| 06d：政治评分不等普遍增加亲以偏向 | App D.4 pp36–37/Fig27。原base已有偏向，finetune相比base未增加；不同年份/finetune可有相对差。 | 支持A限缩；不把“以色列相关概念更活跃”等同一贯正向政治价值。 |

### D6-07 Persona Vectors / D6-08 Data2Behavior

| Claim | 原始正文 + 相关附录 | 判定 |
|---|---|---|
| 07a：known-trait差方向、选层，prompt投影与后续行为相关，条件内相关下降；finetune终点相关 | 2507.21509 §§2–4 pp3–7；App A.3/B.4 pp27/29、C.2 p31（hallucination0.830→0.245）。 | 支持；不能混称所有trait相关都同幅下降，也不是自然训练时间前兆。 |
| 07b：训练前projection difference、top/bottom/random500实际训练；过滤后仍有某些效应 | §6 pp9–12；App H/I pp43–44、K/L pp53–56；L.1 10epoch，LR1e−5，rank32。 | 支持；L.2 p56明确高投影伴incoherence/story mode和不实用性，额外toxicity filter可阻evil。A已正确收录这些反例。 |
| 07c：训练时正向steer，撤去后少学trait；较推断反向steer保MMLU；多层更强；保窄域错误；CAFT/正则反例；新事实代价 | §5 pp7–9；App J.1–J.7 pp45–49/Figs24–31。 | 支持所测条件。正则绕道只是解释；CAFT无效针对所测hallucination，非所有concept ablation。 |
| 07人审94.7% | App B.2 p28/Table1：每trait50对极端分数，2作者，合计300判断。 | 支持排序一致率，不是任意答复分类准确率。 |
| 08a：MDF最后token各层均值推断注入，与LoRA/fullFT比较 | 2602.04735 §§2–3 pp2–5/Eqs3–5；App B/C pp14–16。 | 支持。操作是推断期改变，不能认定它复现参数训练机制。 |
| 08b：扫α0–8，阈值细节不足，NYC方向反、Reagan幅度差，SafeEdit200与40.75base、CoT计实体 | Table1 p4、Table2 p5；§3.1 pp3–4；App C.1–2 pp15–16。 | 支持A对“可靠幅度预测”的否定；原Table1 Normal/Tuned NYC0.8%也明显变动，作者阈值化呈现不宜当统一原始率。 |
| 08限制：混合分布/逐项归因未做、时间不可外推 | §3.5 p6、Limitations p9、Table4 p6。 | 支持。不能将仅纯偏向数据下效果算成稀疏poison发现率。 |

### D6-09–13：grokking、阈值、特征竞争及数值替代解释

| Claim | 原始正文 + 相关附录 | 判定 |
|---|---|---|
| 09a：Fourier电路、关键频率消融、restricted/excluded loss、形成先于准确率/cleanup | 2301.05217 §§4–5 pp4–9/Figs6–7；App C pp14–27相关电路/多配置。 | 支持真实电路分析与回溯进展量。关键频率来自最终解（§5.1），不能写成未知任务在线通用量。 |
| 09b：无预先criticality指标、数据/模数/正则边界、slingshot非必要、正反馈解释为推测 | §6 p9；App C.2 pp22–27、D pp26–31、E/F pp31–35。 | 支持。App D.2 p30限定有weight decay或其他正则；不推出slingshot任何情况下均无关。 |
| 10a：固定输出改edit distance、更多样本、LaMDA改Brier去掉部分规模涌现；视觉人为制造 | 2304.15004 §§2–5 pp4–8/Figs3–7；App A pp12–13/B p14。 | 支持测量解释；独立token和近似函数为解释模型假设，不是所有LM已验证定律。 |
| 10b：主要规模而非时间，不否认真涌现，也不免除真实失败后果 | §7 p9明确不否认真实涌现；其设计自变量见§3–5。 | 前两项原文支持；“失败后果仍真实”是报告逻辑边界，不是额外实验。 |
| 11a：NTK二分类双特征小耦合下starvation；SD；多任务实验 | 2011.09468 §§3–4 pp3–9（Def2、Thm2、Eq18）；App C pp22–26（扰动解、smallδ、固定点及SD推导）；App B pp18–20。 | 支持有假设的理论与实测。核心starvation量是最优响应对另一特征强度的导数，不是任意深网自然时间轨迹定理。 |
| 11b：primal/dual只保最优点、非相同路径；ColoredMNIST可同时学颜色/形状，test调参 | App A p17；§4.3 pp7–8/Fig3，App B。 | 支持A重要限缩。无后门实验，不能据术语直接识别本项目中介。 |
| 12：shortcut分类、非普通overfit、四要素；4k星/月toy、MLP/CNN差 | 2004.07780 §3 pp4–6/Figs2–3；§6.3/BoxII pp14–15；App A p28：4k训练，2k测试，MLP51%/CNN100% OOD。 | 支持。综述引文仅链入口，不当本轮已读原始研究；该事实不能作为“后门研究没有竞争理论”的证据。 |
| 13a：softmax舍入破零和、均值互相放大；精度/投影/末层BN干预 | 2605.06152 §§3–4 pp4–9/Figs1–3；App C pp19–25：NC、class-balanced、orthogonality、残余概率近似；D pp26–28为方法。 | 支持条件数值机制及所测spike干预。不是所有loss峰的共同解释。 |
| 13b：110M LM float64不消正向膨胀；未复现负发散，无后门证据 | §4.4 pp9–10；App A.4 pp16–17、D.3 p28。 | 支持A限缩。类别频率解释为作者解释；不能据此归因项目ASR。 |

### D6-14 / D6-15：EM反证与广/窄解

| Claim | 原始正文 + 相关附录 | 判定 |
|---|---|---|
| 14a：Qwen14B、rank1/rank32、金融循环；长度匹配后可再诱发 | 2607.09053 §§3–5 pp3–6/Fig7；App A pp9–10为judge。 | 支持。并无足够报告证明长期根除；A未采用作者强“表面现象”结论正确。 |
| 14b：cosine/gradnorm并非稳定EM对应、实验/报告局限 | §5.1–5.3 pp5–7/Fig2/8；局部cosine依旧t±5；§5.3以single-adapter cosine比Fig5 all-adapter行为；全文未报告原5×放大复现。 | 支持局部反证。不能由此推翻方向干预或所有EM机制。种子/计数不足为报告质量限制，而不是观察“无效”的独立统计证据。 |
| 15a：混clean难保窄坏去广坏；KL可窄解，去KL续训广解 | 2602.07852 §3 pp4–7/Figs3/5；App H/I p22 Table9/Fig12、J pp23–24、K.1 p25、L.3 p31。 | 支持优化干预，不是只有相关。混clean的直接控制图是rank1医疗；不要读成所有配置均测。 |
| 15b：范数效率、方向扰动稳定、sports例外、最终B方向梯度、技术文本第二例 | §3.2 pp5–7/Fig4；App K pp25–28/Figs14–16、L pp29–33。 | 支持操作化比较；sports窄解最低loss更低仍需更大范数；梯度参照最终解，事后信息。 |
| 15c：FineWeb steer扰动≠改预训练分布；未闭环；有epoch不同与重用既有结果 | §3.2.2 p6/Fig6、Limitations p8；App H p22 Table9（LoRA SFT1epoch、SFT+KL3epoch）；正文§2明确回顾先前模型。 | 支持A限制；因果来源于预训练、全局最优和唯一解均未建立。 |

## 3. 关键未完成支路：按实质问题分配，不按篇数凑齐

以下除已读15篇外，仅做官方身份/摘要初筛；**所有数值/机制还未签为全文事实**。原 A 应读全文正文和与相关论断直接有关附录，写入独立claim和反例后交 B。现有 pending 标签不能替代完成动作。

| 材料与原始入口 | 影响现有命题/缺口 | 处置要求 |
|---|---|---|
| [原始 Emergent Misalignment 2502.17424](https://arxiv.org/abs/2502.17424) | D6-04/05/15反复以它为起点；原始形成轨迹、条件/后门与宽泛有害泛化的已有范围不能靠后续论文概述。 | **原A必补全文。** 定位训练/评价/对照及相关附录；不只写历史首创一句。 |
| [Persona Features Control Emergent Misalignment 2506.19823](https://arxiv.org/abs/2506.19823) | D6首表“方向部分中介/预防”、D6-05存储/表征边界：可能已有训练过程feature变化及再对齐因果证据，遗漏会低估已有解释。 | **原A必补全文。** 注意feature控制输出、引导训练与自然形成因果链的区别。 |
| [Cloud原始 Subliminal Learning / Nature正式文](https://www.nature.com/articles/s41586-026-10319-8) | D6-01/02关于同初始化、数字隐式传递、MLP/SGD一般性，全靠后续引用仍不足。主审确认尚无他域已全文覆盖可接缝。 | **原A必补原始研究全文及相关补充。** 正式题名为 *Language models transmit behavioural traits through semantically unrelated data*；核正式文与preprint身份，不重复计篇。不能把“同初始化”升格为所有改造攻击的定律。 |
| [Towards Understanding Subliminal Learning 2509.23886v2](https://arxiv.org/abs/2509.23886v2)（官方元数据2026-03-05，ICLR2026） | 原始摘要明确为divergence tokens的mask、早层充分性、token-entanglement/logit-leakage非必要及脆弱性。直接影响D6-02“已有机制做至哪”与正常混合/换prompt边界。 | **原A必补全文。** 与Steering各自识别什么比较，不能只采后者对前者的评论。 |
| [Beyond Owls 2610.10657v1](https://arxiv.org/abs/2610.10657v1)（2026-10-07） | 官方摘要包含新学随机MLP能力、条件后门和agentic hacking；其steering与SL的OOD区别直接限制D6-02“所有SL只是已有简单偏好方向”的概括。 | **原A必补全文。** 非外围攻击名录；正文与附录核新能力/后门是否传递、优化/蒸馏条件和阴性结果。 |
| [Subliminal Effects in Your Data: A General Mechanism via Log-Linearity 2602.04863v1](https://arxiv.org/abs/2602.04863v1) | 官方摘要的LLS从一般偏好数据选子集、跨架构效应；直接连接D6-07训练前数据筛选与D6-02“同模型方向传递”解释。当前PAPERS甚至未列。 | **原A必补全文机制相关支路。** 核log-linear假设、理论结论、训练目标、selection与原始蒸馏区别及跨模型边界；不能因标题general机制直接接受普遍性。 |
| [Phantom Transfer 2602.04899v3](https://arxiv.org/abs/2602.04899v3)（2026-10-07，NeurIPS2026 camera-ready） | 官方摘要声称跨生成/学生模型、数据级防御、密码条件。直接影响D6-02同初始化限制和D6-07过滤有效性。 | **主审将D5 A主责全文；D6 A须回原文核影响本域的相关正文/附录后明确接缝**，不能排为不相关，也不必独立重复全套防御目录。当前未标他域已完成。 |
| [When Are Two Networks the Same? Tensor Similarity for Mechanistic Interpretability 2605.15183v1](https://arxiv.org/abs/2605.15183v1)（2026-05-14） | 官方摘要含tensor模型中grokking/backdoor insertion的functional dynamics量，影响“已有内部进展信号”范围。 | 建议由主审明确交D4 A主读，**尚未完成分配/核验不能算覆盖**。D6核模型类/参照checkpoint/未来信息/领先行为与否；不是全部表示距离研究都要纳入。 |
| [Progress Measures for Grokking on Real-world Tasks 2405.12755v2](https://arxiv.org/abs/2405.12755v2) | 官方摘要为一般分类grokking及三进展量，暂无直接后门证据；与09任务专属性相关。 | 可审计地限缩为外围背景候选：当前不以其未读阻断有害学习核心，也不能说“grokking进展量仅算术”。若C查漏显示实质影响再补正文。 |
| DETER2611.15479 | D5训练防御主责。 | 保留明确交接；D6不作未读实质推断，无需机械纳入。 |
| D1 C新增ICML2024 Li/Liu CNN理论（PMLR li24at）与SP2024 Orthogonality/Linearity | D6-11一般特征竞争理论与母问题接口：不能只说“starvation无后门实验”就推“后门形成没有动态理论”。 | D1 A/B主责原始全文。D6 A回读其被用于本域比较的相关正文/附录，说明CNN/NTK假设下哪些学习竞争/两阶段知识已有、哪些不可移用LM；不要求D6重复整域CNN检索。本B尚未对两篇作独立实质签证。 |

### 本轮 B 初筛搜索记录

2026-11-19，通过web直接打开上述arXiv官方abs：2610.10657、2509.23886、2602.04899、2602.04863、2405.12755、2605.15183；核标题、版本及摘要来决定必要性。原始Cloud入口来自已读Steering v4参考文献[11]，主审确认不应假设其他域已覆盖。此步骤只用于**决定补读范围**，未把摘要当结果证据写入本审核的支持表。

## 4. 总表与重合边界的审核结论

- “低外显不等于内部未变”已有多种研究支持，但三种设计不可合并：自然训练checkpoint的早期可诱发性（04）、最终电路回溯（09）、终点跨任务/跨训练模型的低行为投影（01/06）。低ASR前置证据应逐个指设计。
- 正常/无害样本会改变有害行为学习与条件性已有直接干预，但01的同任务clean、03的clean回答来源、06的辅助自蒸馏各有不同估计对象。不能用一个“竞争”词抹平。
- 方向控制已超过纯相关，A正确保留部分中介与训练预防；B4及待补原始feature/SL文会影响这一已有知识版图。仍不能从方向干预推出唯一存储变量或自然形成完整机制。
- 已有训练前风险估计不能否认；其known-trait、预先选择数据集、α/层调整、终点任务及失败率同样不能省略。MDF不支持可靠数值预测的强版本。
- 04与14并不简单一正一负抵消：14没有重复关键放大干预且自身比图存在不一致；04也不能泛称在线通用信号。一般grokking/阈值/数值理论只提供邻近解释与可检验边界，不能认定本项目即属此机制。
- 当前补读未完成，不能宣称一般有害学习链饱和、母问题无人回答，或本项目已经独创。无需据此设计新实验；本轮只修正已有知识与重合事实。

## 5. 原 A 交回与 B 复核门槛

A请在 `REVISIONS.md` 逐B1–B5及必补支路写采纳/有据不采纳的处置，更新 `REPORT.md` / `PAPERS.json` / `SEARCH_LOG.md`；未取得原文者如实标受限。交回后B将核新增实质claims及这五处修正；未变化且已核的15篇不机械重跑。C仍需独立查漏；本B审核不替代C，也不自动允许实验/论文修改/发布。

## 6. 第二轮复核暂停记录（2026-11-19）

主审因用户指出 token 成本过高、其他审核进程触达账户用量上限，要求不展开新的阅读/搜索，保存已完成项后结束本轮。以下不得被解释为新增六篇通过。

已完成：读取 A 的 `REVISIONS.md`、`CORE_FULLTEXT_NOTES.md` 及 REPORT 相对 REPORT_A 的改动。B1（Inherited 的 appended 限定及 random 正效应）和 B2（单 adapter t±s / 全轨迹 PCA 的未来信息）已在可见正文改动中正确落实，与本 B 第一轮直接核过的原文一致。B3–B5 的 REVISIONS 处置与第一轮原文要求一致，但本次输出截断，未完整逐句确认所有实际正文/PRIORITY 落点，暂不作完整签收。

**明确尚未完成：**

- 新增 D6-16–21 六篇的所有实质 claims，本轮仅看到 A 的报告和笔记，尚未由本 B 独立回原始正文/相关附录核验。A 标记 FULLTEXT_READ 仍只代表 A 阅读状态，不能写作 B 已核。
- Phantom、Tensor Similarity、ICML2024 CNN、S&P2024 四条新增跨域接缝尚未由本 B 回原文确认。
- PAPERS.json 的新增身份/版本/阅读状态尚未逐项复核。
- B 初筛中 Cloud 正式题名的错误，A 报告更正为 *Language models transmit behavioural traits through hidden signals in data*；本 B 尚未回正式来源确认，本轮仅登记更正待核。

当前状态：**A 修订和补读已交回；B 第二轮未完，不签通过，不建议记为可交 C 的已核版本。** 未发现的新错误不能据此视为不存在。后续如恢复，仅核 B3–B5 实际落点、新增六篇与四条接缝及索引；旧已核证据不重跑。本轮没有新的搜索、实验、GPU或发布。

## 7. 续接第二轮：新增主张原文复核（2026-11-19）

按主审最新授权续接§6，仅核 B3–B5 落点、D6-16–21、四条接缝和身份索引。B1/B2及无改动的旧15篇不重跑。以下由本B直接核本地原始PDF/提取正文和对应附录；A笔记只作待核清单。Cloud Extended Data Fig.2/Table3、Towards Fig.30另看原PDF图像。本轮未实验、未派子agent、未修改A正文。

**结论：主要补读已得到原文支持；尚有以下小范围必改，改后可交C。当前不能把A未修订文本记为B通过。**

### 7.1 必改清单（不需要新一轮广搜）

| 编号 | 实际落点 | 原文与具体改法 |
|---|---|---|
| R1 | REPORT D6-16a；CORE D6-16第2条 | v7 §3.3 p5/Fig5 p6报约20%主8问、6%预登记；Fig17 p22图注却写10%预登记、30%主集。**10/30是结果百分比，不是judge阈值。** 删除“阈值10/30冲突”，改为“Fig17结果率与正文不一致，采用正文口径”。图注仍写alignment≤30；另有coherence≤50剔除与正文<50剔除的边界符号差，不能混同这项结果率冲突。 |
| R2 | REPORT D6-16b及其附录定位；CORE D6-16训练动态/延训条 | §4.7 pp10–11/Fig13的去weight-decay和10epochs阴性结果是**multiple-choice misaligned-choice log-probability**。把“第一epoch后EM不再增强”等泛化表述限定到该读出；不据它宣称所有采样行为/所有读出皆平台。Aligned pivotal-token Fig29在**附C.5 p28**，不是附E；附E才是evil-numbers补充。 |
| R3 | REPORT D6-19c；CORE D6-19 LoRA配置条 | Beyond Owls附D p19的rank8 all-module≤1%是**teacher和student都由attention-only换成all-module**；student升rank32是其后同数据对照。补足联合变更条件，把“依赖attention-only”限为作者所测配置；不能暗示只改学生或所有rank均必须attention-only（同附录all-module rank1约3.5%阳性）。hard-token支路多因素变更限制已正确。 |
| R4 | CORE D6-17第4条 | Persona Features Fig25 p26的十特征两两cosine约**0至.34**（含.00/−.00舍入值），不是.02–.34；若专指toxic#10对其他九个，则约.01–.20。改数值或删除区间；“并非同一变量”判断保留。 |

R1/R2涉及版本结果口径与测量对象，R3涉及对照归因，必须修到REPORT与笔记两处；R4仅笔记数值。不是要求扩论文目录或另设计实验。

### 7.2 claim → 原文 → 支持/更正

表中“支持”只签所列事实与限定，不替论文复现实验或补证其普遍性；附录页码按本地原PDF。

| Claim/新增或修改内容 | 本B核读原文位置 | 结论 |
|---|---|---|
| B3 bird评委/筛run；B4 fullFT两种数据；B5 MNIST泛化 | 第一轮已核原文：Homeopathic C.3–4 pp29–31/E pp35–36；Steering §6.2 p8/M pp26–27。此次核REPORT 01b/01e/02c及PRIORITY对应实写 | **支持，落地。** 55.5%分母、无名≥10%排run及范围，direct-steer fullFT阳性，未投毒类9.8→60.5%与源类条件性均分清。 |
| 16a insecure-code/教育控制/主与预登记/模型与格式多样性 | §§2–4 pp3–8、Table1 p6、B.1–4 pp18–20、C.1–5 pp22–28、G pp38–39 | 支持6000/1epoch、10 vs各6seeds、20%/6%、教育语境主EM低但欺骗+30.9%、jailbroken差异和条件边界；**图注冲突按R1更正**。多样性每子集仍6seeds，非把子集当种子。 |
| 16b 条件后门/自然训练读出/非grokking外推 | §4.2 pp7–8/Fig7；§4.7 pp9–11/Figs11–13；C.5 Fig29 p28/C.6 pp28–29 | 支持有trigger约50%、无trigger<.1%、5seed每10steps的先ID后OOD和早期两类均涨；**按R2限定延训读出及改定位**。不能推未知风险预警。 |
| 16c evil-numbers/ICL | §4.6 pp8–9、附E pp33–34；§4.3 p7/C.7 p29 | 支持14926、8seeds、4epochs/B39/LRmult2、666/911保留、普通8问阴性而格式提示阳性；ICL≤256仅坏代码成功。 |
| 17a SFT/RL、SAE筛选和loss干预 | §§2–3 pp2–9；D.1–3 pp25–26；D.8 pp33–34；H.3–4 pp37–38 | 支持fullFT6000/B64/.2、o3-mini及helpful-only边界；2.1M/topk64 SAE、known-E增量筛1000→steer40→10，正负干预；直接降低错误建议CE、.82/.91 AUPRC。自然feature形成的直接/间接路径仍未定。**cos按R4**。 |
| 17b 5%“早期”/35steps再对齐及残余 | §4 pp12–14/Figs14–16及脚注6；G/H Fig37 pp35–36 | 支持固定6k坏比例横轴，非时间预警；.1%/.5%、35steps×B4与120samples原文不一致；179问到180steps部分未回baseline及feature/坏代码残余。 |
| 17c 评分分母/reward hacking/assistant方向 | A pp18–20；H.2 p37/H.6 pp40–41；J pp45–46 | 支持四类重采、strict/loose coherence约束、reward-hacking主EM0仍增加其他坏行为、toxic对tool-deception例外；assistant方向≤1%且干预非对称。 |
| 18a 原始SL过滤与评估、数据量、ICL | Nature PDF pp2–4；Methods pp8–10、13–15；ExtData Figs3–6/Table3–4 pp19–26 | 支持5动物/5树预选、30k→10k、10epochs、数字EM约10%/CoT约8%、各自suffix、正确答案+align≥78、每题3次共22419/teacher及9840/17345/19825不等量。37.3%为识别来源FPR、ICL至全10k阴性等限制成立。 |
| 18b 跨family及小更新理论 | Fig5/main pp4–5；Methods pp10–12 Lemma1/Theorem1及证明 | 支持GPT4o/4.1例外、共享初始化来自访谈；同起点、full soft CE/MSE一步、小ε下非负内积，teacher梯度步才推相应loss改善（全ε正交例外）。不证明多步hard-token或跨架构普遍传递。 |
| 18c MNIST/BC/SGD/shuffle | Methods pp10–11、13；ExtData Fig2 p18图像及Fig4 p20 | 支持同init约90%、SGD约68%、MNIST输入约52%、跨init先BC约37%/20%、改变架构仍部分成功；50aux主设置与3aux BC sweep已区分。Shuffle降低传递不等于完整中介识别。 |
| 19a 随机MLP、计数、失败对照 | §§2–3 pp4–7；附A p15/Table2 | 支持420参数任务、50k teacher、260k transfer top32KL、主R².38 vs .27及最强例选择；另两例.21/.25低于.27/.26；hard-token R²0。计数59.8 vs27.3及Nemotron9.5/9.3/9.8、直接训练80.9正确。 |
| 19b 单层向量的ID/OOD边界 | §5 pp8–9；附B p16/Table3 | 支持MLP layer14/2k validation/.39 ID、半整数.11 vs SL.29、拼写−.13 vs .16；计数layer25的48.8来自固定3300validation，OOD0 vs11.1。不能外推排除多向量机制。 |
| 19c 条件后门及棋局 | §§6–7 pp9–11；附D/E pp19–21/Figs7–8、12–14 | 支持三seed选择teacher、90k软目标、23.5/20.2/17.3及零泄漏口径；hard-label换teacher/rank/数据，3.4M/peak17/last8.5；棋局3×64、58.3 vs10.9和43.2胜率。**LoRA归因按R3**；棋局上限原文为100 steps，笔记“100turn”宜统一为steps。 |
| 20a greedy/global entanglement/divergence | §§4–5 pp3–5；A–D pp13–15；E–H pp16–23 | 支持全局50名单非必要、factual argmax且counterfactual不同、loss-mask不删context；7.47/18.34和4.69/13.18%、panda/penguin/树例外与替换冲突结果均有原文。 |
| 20b 层与混合/改写/跨family | §§5.2–7 pp6–8；I–L pp21–28（Fig30图像）；S pp35–39 | 支持训练前10步IG近似patch、早0/7与冻结约10层；train数字fit非一般utility、held-out Qwen不超base；Gemma penguin反例、混合非单调、eagle/penguin跨family。 |
| 20c EM支路与非正式理论 | 附C pp14–15、M/N pp27–32、P/Q pp34–36 | 支持数字不禁负面数、10word suffix/50 coherence/30 alignment、GSM≥50且更严无传递、6.52/17.2%div；直觉强假设非formal proof；其他模型和语义不一致bias阴性边界成立。 |
| 21a LLS机制、DPO、跨架构及恶意评估 | §2 pp4–7/Alg1；§3 pp7–11；A/B pp17–25 | 支持正分池topγ、按两答总长度、reference=student初始；动物/语言过滤、恶意支路未给同等toxicity过滤；LR选择、3seeds/11checkpoints取peak、三题与第三题仅OLMo3。不能当普通SFT或广域EM。 |
| 21b 条件定理与经验支持 | §2.3 pp5–7；C pp26–31 ThmC.1、DefC.3及证明 | 支持selected-pair log-odds未中心化cos≥1/(10C^1.5)，固定φ、teacher=initial student、α..Cα/ε/优化误差及非稀疏子空间限制。ψ“可在R^d”精确含义是优化空间可实现任意v（脚注允许足够大球）；不只是向量有d维。SVD不验证entrywise ε，SFT仅A中未来类比。 |
| Phantom跨模型及向量反例 | v3 §§3–4 pp4–7/Table1；H pp27–28 | 支持不同family、GPT4.1 Stalin失败及30entities异质；500paired差分、10k未过滤、α sweep、多数<20/最大26，追加构造仍弱但不排除其他向量。未拿它推翻Blank。 |
| Tensor Similarity局部动态信号 | v1 §3.3 pp6–7；B pp16–17 | 支持SVHN embedding128/hidden256/单bilinear、无bias/norm、10%patch→9、ASR96/clean85–81、Δ.24/.43。仅pre/post块分离，非未知Transformer触发定位或领先时间。 |
| CNN直接后门动态理论 | §§4–6 pp4–7、Condition5.1/Thm5.4；C.2 p23起、D pp27–28 | 支持正交signal/trigger、过参数及尺度条件下T_v<T_u和后期trigger主导；不是全部后门/预训练LM定律。D1主责全文，本B核本域被引用的命题及对应证明。 |
| S&P staged/NTK保留与上界 | §§3.1–3.4 pp3–6；A.2–5 pp15–16；补充C/D p21 | 支持two-stage为Assumption3.1+视觉实证，固定feature平方损失、正交更新保留及风险上界；约70°实测不等90°，大上界不推出真实遗忘。没有把原文反向推理照搬为定理。 |

### 7.3 身份、索引与交接范围

- PAPERS新六项的标题、arXiv版本/首页日期、文件与REPORT编号相符；Towards为ICLR2026，LLS首页arXiv Feb4/文内Feb5并存。Cloud正式PDF首页确认 *Language models transmit behavioural traits through hidden signals in data*、2026-04-15、DOI及Sören Mindermann；**本B§3旧题名在此正式撤回更正**。不是另一篇独立复现。
- 原始EM的ICML身份与Nature扩展关联由官方[PMLR记录](https://proceedings.mlr.press/v267/betley25a.html)及[arXiv版本说明](https://arxiv.org/abs/2502.17424)核实；Nature正式题名为[Training large language models on narrow tasks can lead to broad misalignment](https://www.nature.com/articles/s41586-025-09937-5)。本轮结果签证针对v7，不把核Nature元数据说成重读其全部新增综述/补充。
- 四接缝按`CROSS_DOMAIN_SECTIONS_READ`与21篇本域`FULLTEXT_READ`分开正确；这些状态记录A读取，不是自动B/C通过。CNN/SP的read_sections共用了两篇说明，建议分别清楚列本篇页节；不影响此次已核实体事实。
- 本轮没有新增待补核心全文；不扩Peripheral候选。R1–R4及表内两处术语精确化（100 steps、任意v可实现）落实后，仅核对应改句即可交C，无需再次通读这六篇。C仍须独立查漏；本结论不等于六域饱和、创新性通过或实验授权。

## 8. 定点修订签收：可交 C（2026-11-19）

仅核实际改句及对应索引，未重跑旧全文。R1结果率/阈值区分、R2 MC读出与C.5定位、R3师生联合LoRA变更及rank1反例，均已正确落实到REPORT与CORE；R4余弦区间、100 steps、ψ可实现任意v也已落实。REVISIONS与四篇PAPERS证据说明同步；CNN/S&P读取范围已分别列出。

**本B范围无剩余必改，可交C独立查漏。** PAPERS中四篇`PENDING_B_RECHECK_C`及跨域`A_SEAM_PENDING_B_C`仍是更新前状态；此次B签收以本节为准，主审交接时可统一同步为B已核、C待审。此为状态同步，不构成新的实质阻断；21篇全文与4篇定点跨域的阅读范围不变。不等于C完成、六域饱和或实验授权。

## 9. C 后新增三篇的 B 增量复核：可交原 C（2026-11-19）

仅核 D6-22–24、C_GAP_FULLTEXT_NOTES 及受影响首表/16a/17b和PAPERS索引；旧21篇未重跑。直接核本目录 c_sources 原PDF提取正文与相关附录；Conditional p7图、Attribution p8表另看原页图。下表页码均为原PDF页，不以A笔记充当原文。

| 新增实质claim | 原文落点 | B结论 |
|---|---|---|
| 22a/笔记22a–b：相邻head/features交换及实际位移均值项局部充分 | §§2/4、A pp13–14、J.1–4 pp21–24，Tables22/23 | **支持。** 五次选择事件的交换数值、均值项≤.33pp、centered test87.20–100与训练≥99.09符合原文；旧状态并非全100%，不把单步充分性升级成历史来源识别。小步下降/全步overshoot与末步换梯度不救均成立。 |
| 22b/笔记22c：训练集decoder恢复、checkpoint差及持续性反例 | J.2 p22（含修复续训）、J.8 pp28–29 | **支持。** 3830训练/8939测试，内层validation、全维SVD、98.20–100、早期低准确率、20refit收敛及seed3/4晚一步/数步均已区分；精确全局basis与持久恢复未成立。REPORT“近chance”须按笔记精确范围理解（init .04–.23%、memorization .02–1.12%，chance约.885%），不是早期数学无信息证明。 |
| 22c/笔记22d：accurate CE、LR控制与RMS强反例 | §4.4、J.3/J.6–9 pp23/26–32 | **支持。** 同完整6000起点5/5对照、100k joint定义、监测限制、LR/decay联变、四前瞻RMS种子及embedding交换/decoder损伤条件正确；没把全部崩溃归因head或数值误差。 |
| 22d/笔记22e：Fourier过滤任务信息及lookup反例 | §5、K.1–4 pp33–35 | **支持。** 全族轨道平均、五划分lookup100%、归一化单频充分性、29.9945% ablation差异均有限定；不整体推翻Nanda算法逆向，不等后门或ASR上升机制。 |
| 23a：混HHH/后续再对齐与条件触发 | §§2.2–2.3 pp5–8、C.1–7 pp27–36；Fig4/5 p7原图 | **支持。** 5%/90%条件、100/1000/10000独立续训、约.41%残余及其他来源/模型有据。A正确保留aggregate单调文字与20→50下降图像冲突、3/6seeds冲突；没有把配比横轴当训练时间。 |
| 23b：教育控制与两类inoculation | §3 pp9–13、D p37、E.2–3 pp40–41 | **支持。** GPT4.1教育7.1%/7seeds、GPT4o弱结果、排1/10seed、persona原格式trigger、96%self-ID≠13%EM、第二提示收窄均已写；不推出任意安全prompt或全部模型。 |
| 23c：自生成SFT及CoT配对/反例 | §4 pp13–15、F.1–3 pp42–45、F.9 p51 | **支持。** 过滤后自生成非动态RL，各模型/过滤数值相符；3.1 vs6.3为配对，18.4为另生成基线。正文称Qwen同向与F.9的17.0%无减轻冲突保留正确。 |
| 23d：分母与两组件解释 | G.1–6 pp52–54、§6 pp17–18 | **支持。** CODE/RECIPE、coherence及非数字过滤后分母、flask99均正确；作者两组件为解释，非参数因果分离。 |
| 24a：差分/人工选feature及steering条件 | §§3–4 pp3–6、E/F方法 pp14–16、Table1 | **支持。** 5400训练/600评估每域、seed0、已选层/强度、IR≤10、8×30、约30候选/20%成功、随机同top200池均正确；未据相关排名或方向操纵识别自然来源。 |
| 24b：代理网页训练与合成对照 | §5 pp6–9、Fig4/Tables5–6 p8原图、G.1 Table16 p27 | **支持。** 1m/24topics/23features、21正移/8进top200、三类构造、Gemma3的4.58/32.08与10.42/9.58、随机3.96/8.75均相符；部分低LR阴性和仅两Gemma跨模型边界已保留。 |
| 24c：统计与历史归因边界 | Limitations p10、G.1–6 pp27–30 | **支持。** 4strata、50/960 vs45/1920、OR2.32、题内重复与单seed限制有原文；synthetic可新问答而reformat保内容，不能隔离单一格式因素；代理语料不是实际训练文档因果归因。 |
| 首表、16a/17b新增综合句及身份 | 上述原文；三PDF首页/版本标记与PAPERS新三项 | **支持。** 新综合句没有超出相应局部结果；16a明确GPT4.1后续教育wrapper，17b明确异任务/训练数据。标题/作者/版本日期为2608.07436v2（10-06）、2604.25891v1（04-28）、2608.11025v2（09-14），索引相符。 |

**无剩余实质必改，可交同一 C 定点复核这三项。** 未用于新增claim的长特征表/外围附录未冒称全读；新增不改变旧21篇及4跨域定点的审计范围。PAPERS的`C_GAP_A_ADDED_PENDING_B_C_RECHECK`可由主审同步为B已核、C待核；本节不宣称C已完成或领域穷尽。


主审状态同步（2026-11-19）：原B §9与同C §6已闭合；上文待办/交审状态保留为历史，不改变实际阅读范围。
