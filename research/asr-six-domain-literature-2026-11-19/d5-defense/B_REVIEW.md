# D5 独立 B 逐项原文复核（首轮）

日期：2026-11-19。对象：本目录 A 稿 `REPORT.md`、`PAPERS.json`、`READING_NOTES.md`、`SEARCH_LOG.md`；协议与项目比较口径已读。**结论：18 篇均已回一手正文/相关附录核查其报告实质主张；需要 A 修订和有针对性的补读，当前不签 D5 完成。** 本文件仅为文献审查，不建议或授权实验，不编辑 A 正文。

复核方式：直接读取 `sources/*.read.txt`（PDF 原文提取，保留物理页分隔）及必要的其他原文提取；不是把 A 笔记当证据。以下以“方法/信息条件—结果和反例—可支持的重合判断”为核验单元；同篇的相关主张放在同一行组，未把全部正文重新逐页读完冒充 B 的必要要求。页码均为 PDF 物理页。未重跑实验、未独立核验代码实现；图形仅采正文明确数值/图注，不重新量图。正文/附录冲突时明确指出，不代作者补全。`“支持”`只表示原文支持该条件性表述，不表示论文实验已被复现。

## 1. 必须修订的具体问题

1. **DETER 恢复门槛写错（阅读笔记 A11）。** 不是“PubMed loss达到过滤基线水平”；§3.2 p4、Table1 p5 清楚写达到 **NO FILTERING** 模型的 PubMed loss。最后一列才是达到这个共同门槛的 token 数，除以同规模 **FILTERING** 模型达到门槛所需 token 数。1.03B 的 20.36×、2.09M 与 0.010% 可保留，但三个参照不能合并。REPORT 也应显式写门槛，避免模糊“恢复”。
2. **DiSec 指标名（REPORT D5-10）。** p7 Table1 的 12.20 是 **AASR**（跨触发的平均），同列 MASR 是14.69；94.67→88.96 是该 NeuBA/RoBERTa/SST2 无防御与 DiSec 的 clean ACC。请勿把 AASR 写成未限定的 ASR。附录 D pp12–13 为每触发评价与聚合定义。
3. **Suppression 投影维数（REPORT D5-12）。** “rank1语法残余54.2%”易误读成 rank-1 LoRA。p16 Table6/正文实际是 rank-8 adapter 上 **k=1 的单方向投影**；全句改为“k=1投影下句法触发残余54.2%，k=6 DSE/DSE-BA在该小面板静态为0”。这不改变再学习反例。
4. **Shared Latent 的 inductive 失败操作必须精确。** p15 F.6 实验为 Qwen3-8B 上共享特征的正/负 **steering**，既不强激活也不可靠缓解；没有在这一段报告 inductive 条件的 CAFT 训练实验。因此可写“共享特征迁移失败”，不能写“inductive CAFT已实测失败”。p14 Gemma CAFT LR=2e-5，相对默认 SFT 2e-4；对应比较同时改变干预和学习率。A 已提到 LR，但应给具体值。
5. **BI-BAU 两类再激活不能混合。** p9 定义 R-ASR=retuning attack，Q-ASR=query-based reactivation。p10 Table2 的12.48%、p21 Table10 SIG 的96.72%均为 **R-ASR**。主文§3.2 query-only威胁模型不能拿来描述这两个需要再微调的数字。附录D.2 p18：TinyImageNet SIG用2条毒例，并加训练集0.02%良性例；QRA另用500 benign+500 backdoored CIFAR例训练生成器50epoch，预算16/255。这是不同访问条件。
6. **P2P 理论措辞需比“理想化假设”更具体。** p13 AppA 首个证明是函数类可表达性的 sup 等式，不是实际训练达到最优。第二个证明把 α→1、ε→0 作为理想情形；进入扩展标签集本身不保证经π映射后不会是原攻击目标。可保留“作者给出理想化分析”，不称已证明任意训练收敛至安全/任务无损。
7. **MB 反例标签和评价补全。** p9 Table4 是 Qwen3-1.7B、Toxic目标、增加**防御触发数量**：InSent四个→六个，ASR0→.972且AACC .729→.303。不是“多个攻击者触发已测试”；p9 Limitations恰好说并发多个攻击触发尚未研究。
8. **原CAFT表定位。** 笔记中的 Qwen/Mistral 数字并非共同“§4/Table1”：p21 AppA.5 Table1为Qwen、Table2为Mistral。Mistral top-5 PCs .34/90.5 与解释方向1.16/92.2说明不同折衷，不能称一个无条件支配另一；A的主旨正确。
9. **不把作者间接机制名升级为唯一中介。** 保留 A 已正确加上的限制：PCA/t-SNE重叠、恒等差分、NTK理想化、特定门控消融、正常均值与不可恢复删除不是同一识别对象。下面“支持”均含这些限制。

## 2. Claim → 一手正文/附录 → 判定

### 训练与数据清洁

| 编号/报告实质主张 | B 回核的原文证据与条件 | 判定/修订 |
|---|---|---|
| 01a ABL观察正常与后门学习速度差，LGA隔离/GGA反学习 | [2110.11571](https://arxiv.org/abs/2110.11571) pp4–6 §§3.1–3.2：WRN-16-1/CIFAR10，9类攻击，10%投毒，画的是两集合**平均loss**；作者自己指出个别毒例高loss，后期clean也低loss。LGA主动放大差异，隔离1%。 | 支持“已有利用学习差异防御”；不支持任意LLM/个体/自然两阶段。 |
| 01b ABL三段实施和成本/反例 | p14 A.3明确20epoch隔离+60epoch全数据普通训练+最后20epoch GGA；p7 Table1 CIFAR10十攻击均值ASR7.69、CA84.76，clean普通训练89.12，clean ABL88.41；Blend16.23、Dynamic18.46、CBA29.81。pp8–9讨论隔离比例、turning epoch与高投毒。 | 支持。比较89.12不是同一受感染权重修复前准确率；是纯净训练参照。高投毒需指明具体表行再报数，不以“可防70%”概括所有攻击。 |
| 02a DPoE训练分流，推理仅主模型 | [NAACL2024 long.27](https://aclanthology.org/2024.naacl-long.27/) pp3–5 Fig2/§3：低容量trigger-only+主模型PoE、四种denoising；两者训练，推理只用g。p7 BERTbase、Llama2-7B，5%词/句触发、20%句法/混合。 | 支持训练预防，不是事后从感染权重擦除。Honeypot先行支系仍需补读。 |
| 02b 定量/保证/外推 | p7 Table2 Llama2/SST2/句法 DPoE R-Drop19.54 ASR、93.67 ACC；pp13–14 AppA要求浅模型学得触发且clean近似uniform；AppB代数把选中集当总体混合代理，依赖纯度与代表性；p14 clean-only Table3；p9限制生成任务需非平凡改造。 | 支持条件、反例、无生成实证判断。 |
| 03a TR主动解耦而非自然模块定律 | [2608.00732](https://arxiv.org/abs/2608.00732) pp3–5 §4：共享分类头双分支、熵权重、各解耦损失、训练末去shortcut；p10 A.3早期5epoch分析，pp10–12 AppB将一致毒梯度/结构/Jacobian假设用于解释。 | 支持。理论与实测区分正确；无解耦的终点失败也由p7 Table2证实，不能把早期偏好当永久隔离。 |
| 03b TR有效、也有正常能力和失败边界 | p6 Table1均值.57/89.40（clean91.19）；p7 ablation有WaNet82–83%和Dynamic65–88%残余；pp12–13 AppC surrogate构造、迁移设置；p17 E.5–6/Table14：训练样例target0、original误判8且概率1；p18 E.7 shortcut逐步全报target。 | 支持。Table14是动态个案，A未误作部署测试错误率。容量表若保留额外具体数字须注明p16 Table12，而不要混作p7损失消融。 |
| 04a HARVEY learned oracle与最终模型 | [2607.05748](https://arxiv.org/abs/2607.05748) pp3–4 §§2.2–3.4/算法p10：先训练naive，另初始化reference，经学习低RCE子集/反学习高RCE子集，首轮reference meta-split回收target类，最终在筛后数据训练。 | 支持；oracle绝非真实poison labels。这里仍依赖“毒较易学”的条件性经验。 |
| 04b HARVEY性能/数据损失 | p7 Table3 WRN/WaNet precision49.93、recall99.50；p14 Table10 VGG11/WaNet2.15 ASR；Table12 clean-only WRN CIFAR89.42→88.52、GTSRB97.42→96.12；p15 Table14 duplicate良性样本、target0的WaNet2.57。 | 支持；“最坏<2%”不能越出主表。ASR按非目标类测试毒化计算（p10 B.1）。不能外推一般自适应攻击或生成LLM。 |
| 05a SelfHeal清洁数据，不是感染权重擦除 | [AACL2026 findings.248](https://aclanthology.org/2026.findings-aacl.248/) pp2–4：base模型prompt做tag/clean/rectify，最后按任务关键词比较原/新回答；需要base可靠、任务提示；全记录保留不代表原信息全保留。 | 支持。前两阶段prompt主要task-independent，第三阶段确有task-specific要求，不能一概所有prompt都任务定制。 |
| 05b SelfHeal效用/残余 | p5明确主表每防御5随机seed；p14 Tables12–14：Mistral句法81.59ACC/52.86ASR；clean情感96.52→82.09；Tables14–15同时显示jailbreak效用可改善及关键词/LLM-judge不全一致。 | 支持负例，勿只报坏项或把52.86套全部攻击。clean与no-attack对照分别属于不同表设置。 |

### 已知防御关联与未知攻击

| 编号/报告实质主张 | B 回核的原文证据与条件 | 判定/修订 |
|---|---|---|
| 06a P2P持续安全key与映射 | [ACL2026 findings.600 rev2](https://aclanthology.org/2026.findings-acl.600/) p3§3.1明确MaaS/input stream，mandatory safety key；pp4–5映射h与Algorithm1推理再注入prompt；p9总结。 | 支持运行时保护分类。不能拿有key结果论撤掉key后的权重清除。 |
| 06b P2P效果、理论 | p7 Table3 Llama/CR/CbaAttack19.73ASR、92.52CA；p15 outlier讨论；p13 AppA公式。 | 数字支持，需按修订6给理论限定；注意力竞争是解释，不是唯一因果中介。 |
| 07a MB步骤、信息条件 | [ACL2026 long.1113](https://aclanthology.org/2026.acl-long.1113/) pp3–5：防御trigger+random词串，第二阶段128条clean组成偏好式损失；4模型；WizardLM218题/29技能，GPT-4o二元评价；p13 1024token greedy。 | 支持不需真实攻击trigger/target、无推理key；正常能力指标只到该面板。 |
| 07b MB共享解释与多触发失败 | p13 AppB说明表征/输出对照在**中和前**Llama2-7B；p9 Table4是Toxic/Qwen3-1.7B；并发多攻击者未测。 | 支持失败边界，须修订7；共享表征相关和中和成功不等于单一共享回路。 |
| 08a Dummy条件和共享证据 | [2606.11648v2](https://arxiv.org/abs/2606.11648) pp3–5：两种防御设置，需同目标任务、未知trigger；TAC=同输入加/不加触发差，Fig1用50例；pp13–14 AppC异输出仅4例，同语义GCG控制16例，作者明确非决定性因果证明。 | 支持。报告宜保留4/16分母使“控制支持共享”的强度可评。 |
| 08b Dummy正负与效用 | p6 jailbreak关键词判拒绝、99题，MTBench0–10 judge效用；p7 Table1多项ASR近0，p8 Table3多未知触发有有效例；p17 Table10跨任务Mistral82.83→83.84、Llama79.80→61.62、Qwen82.83→48.48；Table11 GSM8K200题Mistralclean53.5→45.5。 | 支持跨任务不可靠；应说Mistral失效、其他部分下降而高残余，不能全部“完全无效”。不同于MB尚未测多个攻击触发，Dummy已有同任务多未知触发实证。 |
| 09a Locphylax主动聚类/恢复 | [ACL2026 long.920](https://aclanthology.org/2026.acl-long.920/) pp3–5 Eqs5–7：两已知trigger、neutral输出、显式clustering，再恢复；p4说明自然覆盖不完全故加clustering。 | 支持；自然共享和人为施加聚合不能混。 |
| 09b Locphylax因果/评价 | p14 AppB attention均匀替换，400测试例，层作用在注入后迁移；AppA.4任意拒绝词出现即非成功；p15 Table7 Llama3-8B/句法AdvBench23.07残余。 | 支持条件性attention作用；关键词可能漏掉先拒绝后有害的输出。无后续重训耐久性证据。 |

### 权重/表示修复与再激活

| 编号/报告实质主张 | B 回核的原文证据与条件 | 判定/修订 |
|---|---|---|
| 10a DiSec方法与访问 | [ACL2026 findings.815](https://aclanthology.org/2026.findings-acl.815/) pp3–5：残差梯度SVD/VAE、异常行prototype均值替换；20k BookCorpus无label、uniform pseudo-label；代理触发5token，heldout200句定位层；p6固定单seed。 | 支持；真实trigger用于评价，不用于定位。但原始clean参考checkpoint与现有权重关系需按方法本身描述，不概括成全部无clean信息。 |
| 10b DiSec指标与安全模型控制 | p7 Table1；p8 Tables2–3；pp12–14附录D/E；clean模型也挑出K=4/2层，adaptive BadPre AASR19.45/MASR23.76。 | 支持效用损失与非后门专属判断；修订12.20指标名。后续普通下游fine-tuning不是主动重投毒。 |
| 11a BI-BAU complete定义/理论 | [2606.14078](https://arxiv.org/abs/2606.14078) pp4–7 §§3–4；Definition3 p5是指定clean/trigger集合上函数等价。p16 AppB.1从投影残差为0推参数和为0，需排除Jacobian零空间等额外条件；p6已承认低正交实际例。 | 支持A不背书普适保证。NTK线性近似/任务正交是条件，不是自然神经网络的实测定律。 |
| 11b BI-BAU实证/失败与范围 | p10 Table2 CIFAR10十攻击均值R12.48/Q5.75；p21 Table10 TinyImageNet SIG R96.72/Q13.95，U版总体R31.91；p18 D.2再激活配置；p17 AppC仅CLIP visual更新；LLM为未来方向。 | 数字支持。必须区分RA与QRA，不能只以“query-only重激活”描述全表。IBU理想参照也不等于已证明原始能力删除。 |
| 12a Suppression静态/再学与状态 | [2611.12807v2](https://arxiv.org/abs/2611.12807) pp6、9–11，merge并新挂rank8 LoRA，5毒例lr2e-4、1–4epoch；B=0保留旧A仍同base，单独A解释不足。Table3投影后ep1重回高ASR，base要ep2；DSE-BA也ep3=100。 | 支持“压低≠不可恢复”，而非仅看标题。存在in-place与merged两种理论regime，应按实际实验区分。 |
| 12b Suppression分母/反例/utility | p27 Table9 50prompt×3seed，DSE-BA ep1 1.3±1.9、ep2 43.3±2.5；p16 k1句法54.2（k6静态0）；p28 Table10：300题MCQ，base116/300 vs purified115/300；独立coupled-format50题由50降43。 | 支持。不是同一所有能力都不变；43/50比47/50差异p=.318不能称优于/劣于fine-pruning已统计确证。修订k。 |
| 13a TopKLoRA设计/必要充分 | [2610.32866](https://arxiv.org/abs/2610.32866) pp2–6 TopK+ReLU人工adapter；p4固定postTopK gate、不替换零掉latent，所有token持续施加；p18 F1、p20 F3：64 attribution、80 elimination、1000选K验证，另4000未见验证。必要=当前面板0；充分=与完整ASR差≤2paired SE。 | 支持局部因果，信息条件/门控/搜索网格必须保留。不是未知trigger部署或全局唯一最小回路。 |
| 13b TopKLoRA正常能力/不唯一/泄露 | p8 Table2小模型single-layer retention .12/.09；p10 Table4 clean twins，9层Gemma alpaca差-.08 CI[-.13,-.03]；pp23–24 random顺序Jaccard7–23%；p27 G3/Table12 4000题至多3次，前40token目标短语判ASR；p5 planted机制说明可含抑制元件。 | 支持。4000题不是所有生成长度严格0，single-layer损失不可隐藏；无重训后耐久验证。 |
| 14a Fragmented差分/共享因果边界 | [2608.30403](https://arxiv.org/abs/2608.30403) pp3–6：3模型×3任务×3触发×2label范式，2×2差分；Eq1自己明确不是内部机制独立可加，Eq2 SAE线性logit归因。 | 支持A限制。其“结构性而非方法性”是作者解释，未独立改变结构组成隔离其他因素；不能照搬成已识别唯一因果。 |
| 14b Fragmented部署与结果选择 | p6 §§5.1–5.2保护输入相对small clean calibration选feature，明确report lowest ASR under minimal CACC degradation；pp7–8 Tables5–6残余Base28.2、CC15.4；保持重构残差的运行时clamp。 | 支持运行时抑制和选参依赖。不能说该部署选择全程使用已知trigger；已知trigger/target是归因分析部分。附表top60计数疑点须A给具体表/列，不能保留无定位的笼统疑点。 |
| 15a Critical-CoT训练/终点 | [ACL2026 long.495](https://aclanthology.org/2026.acl-long.495/) pp3–6：ICL与FT独立威胁、SFT+DPO，BDR/TDR/ACCd/ASRr/ASRt分开；p9 50%clean，p7三run平均。 | 支持；不是单纯prompt filter，检出不等于拒绝恶意推理。 |
| 15b Critical-CoT效用和未见trigger | p8 Table6 SFT76.5/13.4、full91.7/1.0、base93.9；p7 Table2 FT后72.4trigger/72.8clean；p9限制semantic/contextual trigger。p5训练例列@_@和In arcane parlance，p7又称二者未见。 | 支持效用限制。该文本冲突确实存在，不能确定泄漏但也不能认证严格未见trigger泛化；请从笔记提升到REPORT一句明确限制。 |

### 邻域、训练干预与广义重合

| 编号/报告实质主张 | B 回核的原文证据与条件 | 判定/修订 |
|---|---|---|
| 16a Shared Latent source/target与干预 | [2606.07963](https://arxiv.org/abs/2606.07963) pp3–5：clean/backdoor模型同trigger prompt diff，3source选top10交集5feature，heldout4行为；正负steering，password反向、BadEdit较弱。 | 支持不是全流程无先验；干预证据较纯相关强，但不是跨全攻击唯一mode switch。 |
| 16b Shared Latent CAFT和失败 | pp7–8 §6/Fig3 CAFT形成干预；p7 Gemma4B Country32.32→29.20很弱；p14 LR差十倍；p15 F6 inductive是steering失败，另2个专门语义feature。 | 支持限定版，修订4。不要用inductive反例去证明原CAFT方法本身不可行。 |
| 17a 原CAFT操作/访问/任务 | [2507.16795v2](https://arxiv.org/abs/2507.16795) pp3–5 forward/backward都投影、推理撤下；PCA方向用普通fine-tuning前后差、generic聊天不使用DOOD；4500不安全代码训练/1000代码测试；misalignment分母是coherent响应。p9明确不旨在erase concept。 | 支持不是后门自然形成同一证据，且方向并非不看未来训练天然可得。 |
| 17b 原CAFT正反对照 | p21 Tables1–3 Qwen7→.51/漏洞91.6→84.8；Mistral6.57→1.16/92.4→92.2；p18 5seed，p20 test-only MistralSAE反更好；p8 SAE10/16、PCA5/16；p22加100 benign约2%可达类似低misalignment。 | 支持。不要把漏洞任务保持翻译成正常任务保持；Table2 topPC的更低misalignment伴漏洞变化是折衷。 |
| 18a DETER机制/成本/邻域 | [2611.15479](https://arxiv.org/abs/2611.15479) pp2–4同权重、正负分别保optimizer state；113/224/521/1030M从头训练；医学过滤、负LM loss；p14常规负学习约2%compute且额外optimizer内存；p4实际metric是heldout loss/正确答案NLL。 | 支持非触发后门证据，不是永久知识删除。主实验存在不同规模超参变化。 |
| 18b DETER效果/时序/optimizer | p5 Table1、p9绝对恢复0.010%；p6 Table2起点保持负更新数并近似match general loss；p24 Table14相同P20 early2.77/late3.67；p7 Table3 Muon113 shared biology loss+25.60、separate+.57，1再学seed；p22各seed搜索再学习率。 | 支持有时序/状态实证但不支持“越早一律更好”；必须修订NO FILTERING门槛。Table1三fine-tuningseed不能写成3独立预训练seed。 |

## 3. 综合主张与项目重合：哪些已支持，哪些尚未闭合

- **已支持**：训练时利用学习差异/分流、已知防御关联影响未知关联、指定位置表示干预、下降后再激活、正常效用折衷都是实质先行内容。因此 REPORT §4 排除宽泛“首次”是恰当的，不需补更多论文才成立。
- **已支持但需限定**：这些研究未由现有结果合成一个所有自然训练阶段的统一机制；“未由所读材料支持”不是“整个领域从未做过”。项目指定历史/有限有效更新的估计对象与本文献有交叠，但不能按模型不同直接判无重合。
- **不能结案**：D5域范围包含机制驱动防御及实际失败边界，当前只读新后继而未读若干直接前身，以及漏了下列训练路由/安全蒸馏分支。用“经典原篇未全读”作为永久覆盖限制不足，因为这些公开全文可得且会改变具体版图。
- **所列综述**：2502.05224仅防御/评估章节部分阅读的标注正确。其二手分类不是原篇证据，不需为了填FULLTEXT而读无关攻击目录；也不能据该目录自动批准所有机制家族覆盖完成。

## 4. 给原 A 的补读单：按实质缺口排序，不按篇数凑配额

### 必要：在 A 修订中处理，全文后交原 B 复核

| 原始论文/分支 | 为什么会改变 D5 版图 | 最低需要回答什么 |
|---|---|---|
| **QES 2610.00663v2**，[官方条目](https://arxiv.org/abs/2610.00663)；2026-11-04修订 | 新的generative LLM显式路由至隔离LoRA expert后shutdown，直接处于REPORT§1/§2“训练分流并撤除”中心。不能用旧视觉TR或TopKLoRA代替。当前B只核官方条目，未把摘要性能当证据。 | 路由/正则如何选、需否知道trigger/target、正常效用、专家关掉是持续运行门控还是权重删除、失败/再学习/多攻击；必须全文。 |
| **Honeypot / Setting the Trap, 2310.18633** | TR参考文献直接先行，浅支路/捕获后门的原始PLM机制会改变“2026分流新发展”的历史和信息条件。 | 与DPoE/TR的操作差异，干预训练还是修复、真实访问、失败控制。 |
| **DBD 2202.03423** | 当前训练预防表只有主动放大易学毒或双分支，缺“自监督表征—分类头—半监督”解耦学习路线，且HARVEY直接比较。 | 三阶段各学什么，是否和普通训练快学方向相反、可识别的机制、标签/数据假设和反例。 |
| **I-BAU 2110.03735** | BI-BAU把双层反学习与失败当直接起点。当前全靠后继重评无法分清原方法真实假设与后来的反例。 | 双层问题/触发估计与clean数据、保证范围、正常效用；不可把BI-BAU基线实施当原方法全部表现。 |
| **Fine-mixing 2210.09545** | 恢复/混合预训练权重与embedding purification是不同参数修复路线，当前DiSec/投影不能覆盖；MB有重实现但非原篇。 | 清洁checkpoint访问、mixing空间/比例、细粒度效用与适应后残余；是否有重新学习控制。 |
| **BEEAR 2406.17092** | generative LLM embedding drift与对抗修复是缺失核心支系，Fragmented已援引“它为何失败”，必须回原文而非只读该文重测。 | 是否已知攻击任务、embedding攻击是否等效真实trigger、正常能力/多目标与失败、无触发含义。 |
| **SP24 Exploring the Orthogonality and Linearity of Backdoor Attacks**；`../d1-formation/c_sources/sp24_orthogonal.pdf` | B已初读pp1–3：明确将后门作continual learning，并用正交/线性解释防御失败；是BI-BAU理论前身，且包含干预性质/防御矩阵。 | 与D1协调可复用全文，D5须独立核防御相关正文/附录；理论假设与实测、哪些结果为引用哪些新测、降低正交/线性是否改变防御。不可把首3页当已全文核实。 |
| **Does On-Policy Distillation for Safety Pose Backdoor Risks? 2610.07654**；`../d1-formation/c_sources/2610.07654.pdf` | B初读pp1–3：安全蒸馏本身可能传后门，Lazy Defense reward clipping只延缓而非阻止。直接补D5“防形成/训练干预/失败”。 | clip范围/训练KL/教师-学生访问，随epoch恢复失效的时间窗口、trigger-free safety效用是否等于广义utility、需全文后写结论。 |

### 来源定位与相关性分诊；不要无差别扩大目录

- **MuScleLoRA 2402.12026**：先回方法正文定位。如果确是利用频率学习偏差+低秩/梯度引导的独立机制（而非上述分流变体），则升为必要全文；它直接可能补“学习速度/频率竞争如何用于防御”。不能只按LoRA关键词纳入，也不能无理由永久略过。
- **NAD 2101.05930、ANP**：旧attention蒸馏/敏感性剪枝为可得原型。当前报告不独立宣称其性能，故不要求机械重读全部经典防御；但若要在最终版称覆盖了这些修复家族，至少分别读原型方法/实验/相关附录，不可把ABL、BI-BAU的基线重实现当其原文。可把“本域核心为训练机制，其他传统修复仅上下文”写清后保留来源级定位。
- **T&R2022、CBD、后续同机制变体**：先确认原始标题/身份/与已核方法差异；没有改变具体结论则记录“变体/背景、不作其性能结论”即可，不因综述列出就设机械配额。
- **GARD 2611.14143**：目前仅摘要，CLIP修复已由BI-BAU有一手锚点。先判断它是否带来新的防御形成机制或关键反例；若是同类领域迁移，可带理由留外围，不冒称最新版全覆盖。
- **BAD-BOOM / Broadening the Backdoor Basin**：[PMLR正式条目](https://proceedings.mlr.press/v306/zhao26bf.html)，[官方PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/zhao26bf/zhao26bf.pdf)。**这是本轮新交给D2 A主读，尚未完成，不是D2已核成果。** 对D5的重要性是抵御clean SFT的几何/持久性解释，待D2读完，D5只需独立核相应防御耐久性段和附录；不得复制摘要量化结论。
- **Safety-Efficacy Trade Off: Robustness against Data-Poisoning**：候选arXiv2602.00822，B官方abs打开失败，其他索引只是定位不当证据。若正式全文有input-gradient regularization与data-fitting不可避免折衷定理，则与D5保效用直接相关，须归属A/跨域完成；目前不能确证理论，也不因访问一次失败宣称不可得。
- **Persona Vectors 2507.21509v3**：D6已有主读；不需要D5完整复制。但若最终综合声称原CAFT是唯一邻域训练期预防，则需独立核其预防性steering及撤除条件。当前D5明确不引用其结果，可等待跨域综合核验。

## 5. A 修订后原 B 的复核入口

请A在 `REVISIONS.md` 逐条回覆1–9项，更正REPORT/PAPERS/NOTES的一致性；新增必要篇各提供方法、评价分母、最强反例和原页码。请特别定位Fragmented“top60计数”所指表/列，避免没有具体单元格的疑点。

原B复核只围绕改变的主张、新纳原文和跨域交接段，不机械重跑18篇。若分诊后排除某来源，必须写其不改变哪个具体判断的理由。全部必要分支完成与C独立查漏之前，D5继续标为进行中。
