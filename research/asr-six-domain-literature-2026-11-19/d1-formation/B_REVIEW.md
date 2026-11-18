# D1 独立 B 逐项原文核验（第一轮）

日期：2026-11-19。角色：独立 B；未参与 A 的写作，不递归委派。核验对象为本轮 `REPORT.md`、`PAPERS.json`、`READING_NOTES.md`、`SEARCH_LOG.md`，并读父 `PROTOCOL.md`、shared 比较对象/跨域表、项目 M058-series-review 与 positive-hypothesis-2026-11-18/QUESTION。本文不提出新实验、不授权运行、不修改论文。

**结论：需要原 A 修订，不能签全面通过。** 原有主干多数结论有正文依据，数值和关键限制总体谨慎；但两个训练条件应纠正，文献主干计数不准确，并且几个已经拿到全文的跨域材料会实质改变“已有研究做到哪”的版图。A 自己声明待整合不能替代整合。InfoFusion 全文受限、OPD 附录缺失继续保留为明确限制，不以未读证明不存在重合。

## 一、原有报告的逐项核验表

页码为 PDF 页码；ACL同时可凭节/式定位。下面每一行均返回该原始论文正文/相关附录，非仅核 A 笔记。多个相互依存小主张合为一行。`一致`只指此行，不是整篇或全域通过。

| 编号 / A 的实质主张 | 原文定位与已核内容 | 结论 / 要求改动 |
|---|---|---|
| B01 RIPPLe：正常一步使 poison loss 随梯度内积变化 | `rip2020.txt` §3.1，Eq3–4，PDF pp3–4；Taylor 一阶项为负学习率乘内积；负内积正则明确 | 一致。局部 SGD 近似不应升级为 Adam 全过程；A已限定。式4实际在p4，A写“Eq3–4（PDF p3）”应改pp3–4。未独立重推所有理论。 |
| B02 RIPPLe：embedding surgery、正常微调后保持 | §3.2，§4.1–4.5/Table2–8；目标相关词均值嵌入，正常微调后LFR比较；部分spam/高LR很差 | 一致；保留“若干条件”而非全面改善。LFR分母是原非目标类样本，不是所有句子；A未报数，不强行增加。 |
| B03 RIPPLe：强正则妨碍学会 | Appendix A.1 PDF p13：lambda=.1阻止spam poison，改1e-5；A.3/4高LR/XLNet结果 | 一致，确为相反约束；不能仅引用总体成功。 |
| B04 Gu：PET双任务梯度幅度、方向冲突 | `gu2023.txt` §3/Fig1–3 PDF pp2–3，梯度图为step800；§4.2 Eq1–7 | 一致；A“方向图”最好准确引Fig3。图不是全轨迹中介鉴定。 |
| B05 Gu：下层保留冲突，上层投影 | §4.2.2 PDF p5；AppA.1/A.3 PDF p10，beta底层0–5=1，上层6–11=0 | 大方向一致但需精确：主配置是下层**完全采用未投影梯度**、上层采用投影梯度；“保留一定冲突”含混。AppA.3全层投影与分层对比不处处获胜（顺序式IMDB→SST-2全投影93.0、分层92.6），不写成分层必优。 |
| B06 Gu：冻结BERT/PET、情感spam、干净重训及失败 | §5.1–5.4/Table1–3；§8；AppA.5 RoBERTa | 一致。大数据重训严重遗忘、prompt-tuning不适用CLNorm明确；A保留失败适当。RIPPLe在PET有不少低于Vanilla，也支持不能跨范式普适化。 |
| B07 Gu/Chameleon尚未完成本项目候选整链 | 两文的方法、实验估计对象；项目QUESTION明确候选为“历史对应关系→同份新增监督迁移→正常后缀后果” | 在已读两文范围内一致；必须保留范围限定，不能当全领域排除结论。Homeopathic等加入后重新写比较。 |
| B08 Chameleon：移除原类/目标类相反影响、其他类较小 | D2 `chameleon2304.12961.txt` §3/Fig1–2 PDF pp3–4；CIFAR10，3000轮模型替换，N=7，B=64，第二实验固定alpha=1再移除后续benign clients类别 | 一致。干预在形成后、正常类别组成作用成立；其梯度/相似表示解释不是单独中介证明。B同时看已有p3渲染核图。 |
| B09 Chameleon：提高攻击batch毒比例缩短寿命 | 同节Fig1；alpha N/B增加，作者报告平均accuracy与lifespan均下降；解释为不学区分/overfit | 一致。应保留该设定限定，不是任意训练提高poison必然缩短寿命。 |
| B10 InfoFusion最新优化框架 | 出版方[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253526001004) metadata/highlights/摘要，B再次查询DOI；标题、132:104221、作者身份一致，提双目标/冲突协调/部分参数与理论保证 | 仅摘要级核实；未获取合法全文。保持受限，不使用其保证或实验数值作已核结论。B未绕付费墙，也不把检索未找到当无OA证明。 |
| B11 ABL：10% CIFAR10/WRN16-1，poison均值loss早降，FC/LBA慢 | `abl2021.txt` §3.1/Fig2 PDF pp4–5：9 attacks、平均CE、FC/LBA较慢；SIG/FC/LBA例外于“两epoch几乎0” | 一致。A没有误写逐样本必低。 |
| B12 ABL：其它数据扩展与隔离不等于尚未形成 | AppB.2/Fig10 p17：GTSRB/ImageNet、BadNets/Trojan/Blend；§3.1–3.2明确已学会才可识别 | 一致。反对把隔离时点当形成前防护。 |
| B13 ABL：20隔离+60“正常恢复”+20GGA | AppA.3 pp14–15原文：20之后在 **full training dataset** 再训60，最后20GGA；中间阶段用数据增强 | **必须改**：60轮是常规损失在完整训练集上训练，仍含投毒，不是60轮干净数据恢复。“正常恢复”容易混淆本域核心clean/poison区别。REPORT和READING_NOTES同步纠正。 |
| B14 ITLM：早期正常accuracy比250毒例高，随后反超 | `itlm2019.txt` Fig1d/p1及引言；查看A保存p1渲染；CIFAR10/WRN16、250水印 | 一致但明确是**训练accuracy**，不是同指标loss的反向证据。A第三节“两个反证”应称对普遍快学叙述的边界，勿把ITLM当ABL平均loss曲线的严格矛盾。 |
| B15 ITLM：早停/重初始化、0.1–1.3残余，GLM边界 | §6.4/Table4 pp7–8；supp§2重初始化；正文§5为GLM/输出污染分析；5%为目标类别替换，alpha=.98，4早停+1全训 | 一致，正确纠正文内zero简写；不把GLM定理当深网trigger定理。没有独立重推证明。 |
| B16 CL：poison与clean loss相近、表示纠缠 | `contrastive2024.txt` §4.1–4.3/Fig2–5 pp4–7，默认1%投毒，SL/CL通过调整epoch达到相近ACC，三CL方法；统一TRL构造 | 一致。统一框架不意味着完全相同训练目标/label及所有构造；A已避免用跨论文差异认定唯一因素。 |
| B17 CL：trigger强度ASR先升后降，语义/MI解释 | §5.1–5.3/Fig6–7 pp8–9：谱扰动0到1K，ASR先至100再下降，MI递减；节名Possible Explanations | 一致，A将其列作者解释而非唯一因果机制正确。 |
| B18 CL：监督微调恢复poison快学 | AppD.4/Fig19 p19（双栏段落续在同页顶部）：先训练backdoored SimCLR encoder，再在每类50 clean + **额外50 target-class poison** 上同时微调encoder/head | **必须补条件**：并非只换loss或纯净SFT；数据规模、毒比例、更新参数等随之变化。不能把该对照当训练目标单因素因果鉴定。 |
| B19 NeuBA：预设向量先于下游label | `neuba2021.txt` §3 pp3–4，LBD+LPT；随机分类head；§4.1 pp4–5 | 一致。攻击者额外监督目标而非仅自然语料投毒，A已说明。 |
| B20 NeuBA：embedding距离/标签后验/最佳选择/分类限定 | §4.1–4.3.2 pp5–7：500-neighbor token graph与4level，6trigger/level；先识别trigger label，按label取最佳；§6；附录结果仅对相关限制回查 | 一致。距离层级是换trigger干预，不能纯归频率或单独几何中介。A界限合理。2024许可更正未在本轮独立打开原更正PDF，列为书目低优先复核，不当已独立证实。 |
| B21 RLHF：0.5% RM 73.3→42.5 | `rlhf2024.txt` §6.1/AppC Table1 p12：7B/SUDO/random attacker，**带trigger测试对**；2.2k测试对；正文0.05%与N=212有笔误，表及后文为.5% | 数值一致。不可把73.3当无trigger clean准确率（该值73.1）；A当前写法正确。 |
| B22 RLHF：RM先污染、策略需更高比例；3%两epoch或narrow有效 | §6.2/Fig5–6 pp7–8，§6.3 p9；same prompts用于RM和PPO；提高剂量同时改变RM置信和PPO触发prompt暴露 | 测量一致；“分阶段瓶颈”只为现象层归纳，未独立识别瓶颈原因。最好显式增加“剂量同时影响两个阶段，非固定RM的单因素传递实验”。 |
| B23 RLHF：224 greedy、clean RM评价、退化、SFT-only范围 | §6.2 p7、AppE、AppI pp25–28（本轮核I总述及例）；移除trigger后算reward；退化不完整回答 | 一致。报reward变化而非统一完整ASR正确。不能由其SFT-only阴性推出全部SFT无效。 |
| B24 Agentic：Qwen3各规模100B/.1%、clean SFT→DPO→GRPO | `agentic2026.txt` §3.1 pp5–6/AppA.1–A.5；主配置3seeds、lower-dose单seed；实际打包95.8B | 一致。改变阶段的因果作用成立；格式/知识/能力分工尚未匹配分离。 |
| B25 Agentic：4B ordinary declaration阶段数值 | AppB Table7/9 pp21–22：Pass@32 30.2/92.0/91.3/92.7；Pass@1 3.8/48.7/48.3/50.9；pretrain 30.2的seed range 1.3–85.0 | 数值一致。不能说所有pretrain低/无行为；A已留seed差。建议在强调首次显现时把85.0这个反例一并写出，以免阶段均值遮蔽先已高行为的seed。 |
| B26 Agentic：命令匹配不是实际执行；低dose1.4/.07 | AppA.5 pp20–21，995 ordinary prompts、1000 rare，32 samples/prompt，任意位置匹配固定head/URL/pipe、flags可变；up to200 positive rollouts/cell被judge；AppB.2 Table14/15 p24 | 一致。A称“固定命令字符串”略窄，实际是flag-invariant regex；可纠为固定命令结构而不是字面全字符串。1.4%是rare-trigger单seed Pass@32，.07%每样本；需明确rare以免接续普通路径主结果误混。 |
| B27 DNA：完整payload动态、TATA失败 | `genome2026.txt` §2.1/Fig2D–3D pp3–6及Methods；Evo2 100M/GENERator800M；TATA在GENERator失败 | 一致。任务是特定DNA payload，不是开放式有害答复质量。 |
| B28 DNA：正常预测竞争、输入token替换、概率/完整成功分离 | §2.1/Fig4 pp7–8；Methods pp14–15；替换是**clean checkpoint**前向，Fig4D P-T为poison饱和checkpoint；TATA首token约1.4e-3→2.1e-3，完整payload失败 | 一致。A拒绝把其宣称的“gradient overwhelmed”当直接梯度干预证明非常必要。仅检验部分输入因果依赖。 |
| B29 DNA：后置日程、累计剂量/时间混杂、单轨迹 | Methods p13明确single training run dose curve；Discussion p17确认escalating，后期较少clean washout | 一致；“lower bounds”是作者推断，并无uniform/front-loaded对照，A没有据此过度宣称。 |
| B30 OPD：低行为→快升、offlineKD对比、拒答过滤仍有效 | `opd2026.txt` §4.1–4.3/Fig2–5/Table3 pp4–7；6 pairs，三seed，CP400；Table3 filtering后78.07–99.67 | 一致。过滤是去显式拒答trajectory，不是移除teacher全分布信号；A区分正确。不要因该结果说生成前缀反馈已单独被证明。 |
| B31 OPD：标量理论与zero-injection限制 | §4.3 Eq5–11，渲染p6核Eq5–6；组内分布固定、scalar logit、小步近似；§4.4 Eq12–13/p8改teacher latent refusal及full-image texture | 一致。未独立重推Eq8–10全部近似；zero是student显式trigger注入为0，不是无恶意teacher。 |
| B32 OPD：缺附录 | 本轮检查10页PDF文本与source tar成员（只有main/preamble/bib、sec0/1、图）；正文多次承诺Appendix但无对应文本 | 一致，持续受限。需要作者材料才能核完整配置；不把“有代码tar”当补齐。 |
| B33 Easy：三簇/平方损失/白化方向人为设定 | `easy2026.txt` §2 pp3–4/Definition1–3/Assumption1；pi是source class比例；固定u后population minimizer | 一致；严格说是固定分布最小化解，不是一般动力系统已经证明的“平衡态”。建议用“最优解/静态预算曲线”更精确。 |
| B34 Easy：真实数据仅取矩再采Gaussian，C.4单设置 | §4 p9，10k train/5k test Gaussian sampled from估计mean/cov；AppC.4 p18，S=1.5/rho=0/pi=.1下优化方向，A从4到16 | 一致，未覆盖真实图像端到端自然形成。理论未重推，维持明示。 |
| B35 课程Phase：终点剂量、人工激活定义、n矛盾/非单调 | `phase-student2026.txt` 全部正文§1–6 pp1–8，§3定义BAR=trigger−control、Fisher<.01、h≥.30；Table1/p6低dose各3 vs摘要10–20；3.5%=65%、4%=30%、4.15%=20%、4.2%=40% | 一致。看p5图1渲染；不能说3%以下安全或时间相变。BAR不是原始ASR，建议首提即说明。日期null适当。 |
| B36 课程Phase：数据相同/重抽矛盾与解释边界 | §3.3随机抽poison；§4/5.2承认data selection；引言/结论又说identical；机制为hypothesize | 一致；不能把全部差异归optimizer随机性。 |
| B37 项目事实与六行比较表 | M058-series-review现行报告、QUESTION候选；固定P/K完整批、lag10、平均T不更大、真实净命中不同；单标量不充分反例 | 事实边界一致。A“P-K更新”建议写“固定完整P/K批对照的后续响应”，避免被读成一个已构造的差分参数更新。外部论文不直接复现本项目估计对象；新颖性未被认证。 |
| B38 主干为12篇一手文献 | PAPERS.json FULLTEXT_READ共12条，包含Survey2025；非综述原始材料为11条 | **必须纠正表述**：“11篇原始研究/课程报告 + 1篇综述”，或“12篇全文主干材料（含综述）”。课程报告仍可纳入，但不是等权高质量证据。 |

## 二、已到手跨域证据必须回收，不能留到最终版之外

本轮B为以下新增接口独立回了原文相关正文与附录；此处是向原A发补证/整合任务，不冒充已经完成原A全文工作或C独立查漏。

| 编号 / 材料 | B实际原文定位与实质结果 | 改变D1哪项判断 / 原A任务 |
|---|---|---|
| X01 [Homeopathic Learning 2611.10110](https://arxiv.org/abs/2611.10110) | §3 pp4–5固定5k毒数字加clean，Qwen德语0→72%；同毒数据加长到匹配步数仍不出现。§4 pp7–8/AppE pp35–37：固定1500步，500 poison，9.8→60.5为**未投毒类别5/6/8/9** patched test；改加同量这些类别clean则6.2。0–4 patched本就成功，unpatched误标7为83.5%，clean提升条件性。 | **最少必须整合**“clean改变形成/选择性/范围”的直接干预，并写数量、类别、固定步数控制；比仅Chameleon形成后保持更贴首次形成。不能写poison-only完全没学会后门。 |
| X02 Homeopathic内部/机制 | AppH.2 pp42–43：100k poison-only终点德语0，但German投影100/51；5k poison+100k clean终点74%/0，投影600。§6 pp10–11/AppI简化ridge，显式a≪c、chat无task等假设，作者承认不解释多毒自行传递、optimizer/batch等 | 把“低行为但内部已变化”补入第4节和比较表；这是两个终点、毒量也不同，非自然时间先后或唯一中介鉴定。简化回归是已有clean解除共现混淆候选机制，不应在D1机制版图隐去。 |
| X03 [Subliminal Learning Is Steering Vector Distillation 2606.00995v4](https://arxiv.org/abs/2606.00995v4) | §3.2–3.3 pp4–5：student方向置换/加向量、teacher生成阶段方向消融/加向量；AppB pp18–19表3/4显示层与token位置各异。§6 pp8–9：冻结Adam per-coordinate scale用于新SGD、极端坐标操作；AppN p27所谓loss-matched其实.142 vs.167，非精确等loss | **最少必须整合**形成相关的方向中介干预及优化器尺度控制，不能将既有机制工作一律描述为终点相关。也不能泛化为所有后门单向量/Adam必要；作者明确参数机制preliminary，长训练/MLP/其他SGD结果有限制。 |
| X04 [Model Organisms for EM 2506.11613](https://arxiv.org/abs/2506.11613) | §4/Fig7–10 pp5–7：rank1 Qwen14B，方向约180步转，未放大行为300–600步渐增；5×放大提前窄窗显现。AppF pp19–21使用未来两侧checkpoint算local cosine，且alignment/coherence阈值变动不抹去起点。AppG/fullFT p24只有间接缩放表现，未证明同样方向旋转 | **最少必须整合**真正训练期内部变化+行为放大近邻，不能只列NeuBA人工预设或RM阶段。不要改成未知未来在线预警，也不要把所有fullFT已有相同机制。 |
| X05 [Backdoor Learning Curves 2106.07214v4](https://arxiv.org/abs/2106.07214v4) | §II Eq1–4 pp2–3：横轴beta是poison loss权重，w*(beta)是每个beta的最优解；不是同一次优化训练的step。§III pp4–6调整regularization、网络epoch、poison比例、触发强度，比较相应曲线 | **最少必须整合**静态加权轨迹/复杂度结果；比较表“ASR快升/阈值”不能无说明把它并入时间起跳证据。正式A需回D3结果和相关附录，给自己真实读取范围。 |

另有需要原A按相关性补核的最小候选支路（本轮不将其所有结果标为B已核）：

- **Inherited or Learned? 2611.00748**：本轮核了§5.1–5.3/Table1–3(pp5–6)的来源对照，no-teacher仍保留poison配对；在text-only teacher与no-teacher对比中只更换clean回答，而full distillation另包含logit监督差别；Qwen区别很大、Llama/Gemma区别小。应整合teacher signal vs clean composition/选择性，而非把no-teacher解释为无后门监督。完整附录尚需原A补核后B复核。
- **Weird Generalization 2512.09742、Convergent Linear Representations 2506.11618**：D6已给全文报告，分别涉及clean数据使条件化、训练期held-out跳变、无外显内部变化，以及可操纵activation方向不等于rank1权重方向。它们能改变D1比较，原A应回相关原文/附录，不能只把题名列成待办。本文尚未独立核这些结果，不将D6报告当替代原文证据。
- **D4电路支路**：D1已有circuits2026下载但未核；由D4正式结果确定是否涉及形成前或仅形成后电路。原A至少写实质处理去向及证据状态，不能用“已下载”结束。
- Persona Vectors、From Data to Behavior、An Emergent Mirage可由D4/D6承担全篇，但原A应收回对本域机制外推直接有影响的反证；不要求把所有内容重复写成长综述。B当前不以未核的摘要结果追加定论。

## 三、原A修订的具体验收项

1. 改B13/B18训练条件，改B38主干计数；精确化B05、B26、B33/B37用词。保留其余已经通过的限定，不重写无关内容。
2. X01–X05必须从相关全文补到REPORT的主结论与项目比较表，同时更新PAPERS真实read_sections与READING_NOTES。Inherited/Weird/Convergent至少处理相关结果与附录；若明确不纳入须给范围理由，不能因D6已承担而跳过本域影响。
3. 修订第1节“已能确定的版图”与第6节“已读材料未直接完成……”时，不只在末尾加几行引用。已有方向中介、optimizer干预、clean组成控制、内部前置/放大测量，应实际进入“已有做到哪”的描述。仍可说明没有证明本项目**指定完整历史—新监督迁移—正常后缀真实生成**整链，但这不是对外部整条机制的否认，也不是新颖性保证。
4. 保持InfoFusion全文受限、OPD配置附录缺失；接受有限结论而非无限等作者，也不能签全面完成。C独立查漏尚未履行，B不得代替C。
5. 在REVISIONS逐条回应本表；原B对改动及新增论据复核，不能用“已据B修订”自证通过。

## 四、实际阅读范围与未做事项

本轮实际使用的完整来源文件：D1的RIPPLe、Gu、ABL、ITLM main+supp、CL、NeuBA、RLHF、Agentic、Genome、OPD PDF+source tar、Easy、课程Phase、LearningCurves；D2 Chameleon；D6 Homeopathic、SteeringDistillation、ModelOrganisms、Inherited。读取方式是**针对原A每项实质主张逐段回正文及相关附录**，不是宣称把每篇参考文献/全部示例/全部数学附录再次通读。上表明确每一项位置。查看已保存的OPD p6渲染，ITLM/Chameleon/课程图以文字与原图定位核对；未经独立理论重推、未执行代码或实验。综述只承担检索地图，B没有把其二手判断当实验验证。

尚未独立核证：NeuBA许可更正全文；InfoFusion全文；OPD缺失附录；Inherited全附录；本节注明的Weird/Convergent/Persona/FromData/Mirage/D4新增支路。对这些不得写“B全核通过”。本轮读取已足以定位修订，不以更多无关搜读拖延交回原A。

## 五、第二轮：原A修订及新增主张复核（2026-11-19）

对象：`REVISIONS.md`、修订后 `REPORT.md`、`PAPERS.json`、`READING_NOTES.md`；对照保留的 `REPORT_A.md`。**结论：第一轮原有错误已实质修正，跨域机制证据也已进入主干，但仍有三处需原A小修后才能签“可交C”。** 本轮没有因A写了“采纳”就认可，而是将修改与第一轮已核原文对应，并实际读取新主张对应的正文和附录。可复用第一轮原文核验，不把复读次数当质量指标。

### 5.1 原有纠正的处置

| Claim / 第一轮编号 | 原文页节与复核结论 | 尚需改动 |
|---|---|---|
| B01、B04–B05：公式位置、梯度图及分层系数 | RIPPLe Eq3–4 pp3–4；Gu Fig3/p3、§4.2.2/p5、AppA.1/A.3/p10。新版已写step800、底层β=1未投影、上层β=0投影及93.0/92.6反例，与原文相符 | 此项关闭 |
| B13：ABL中间60轮 | AppA.3 pp14–15。新版明确完整仍含poison训练集和常规损失，撤回“正常恢复” | 此项关闭 |
| B14、B18：快学的边界和监督微调控制 | ITLM Fig1d；CL AppD.4/Fig19 p19。新版区分accuracy/loss，并写每类50 clean+额外50 target poison、encoder/head同时更新，未再说单独切loss | 此项关闭 |
| B22：RLHF阶段差异 | §6.2 pp7–8。新版明确剂量同时改变RM与PPO暴露，不再把原因定位为独立瓶颈 | 此项关闭 |
| B25–B26：Agentic种子与成功定义 | AppA.5、B Tables7/9/14/15 pp20–24。新增pretrain三seed 1.3–85.0、flag-invariant、rare分支及Pass@32/1分母均符合原文 | 此项关闭 |
| B33、B35、B37：静态最优、BAR、项目对照 | Easy Definition3/§2；课程§3；项目M058/QUESTION。新版分别改固定分布最优解、触发减control、固定完整P/K批响应 | 此项关闭；理论仍未重推 |
| B38：计数 | 本轮直接解析PAPERS：12 FULLTEXT_READ、10 CROSS_DOMAIN_SECTION_CHECKED、1正文缺附录、1摘要受限、1下载未读；12中含综述。新版计数没有把跨域节读和综述冒充额外原始全文 | 此项关闭；数目不是完备性认证 |
| B02–03、B06、B08–12、B15–17、B19–21、B23–24、B27–32、B34、B36 | 对照新版相应段与第一轮逐项原文记录：保留的数值、阴性结果、控制限制没有因修订被删去；无新增相反实质主张 | 第一轮逐项结论继续有效；B10/B32来源受限仍不关闭 |
| B07及新版§1/§6整表 | 加入clean组成、方向干预、内部前置、阈值坐标区别；“未做指定整链”限定已核材料，并明确不构成新颖性保证 | 范围改写合理，须先修下表R2-01/07/12；不替代C独立查漏 |

### 5.2 新增主张逐项核验

以下均使用原始论文文件，不把D3/D4/D6的报告作为原文替代。PDF页码列出实际支撑位置；“一致”只针对该行。第一轮X01–X05已核原文的部分在本轮承接，新增附录与新增支路另行实际阅读。

| 编号 / 新版claim | 原文页节、控制及核验结论 | 需改动 |
|---|---|---|
| R2-01 / X01：5k数字poison加100k“普通对话clean”，德语0→72 | Homeopathic §2–3 pp3–5/Fig2–3。**结果数值正确，clean种类错误**：同类数字续写，由未加German system prompt的原模型生成，姓名为其他名字或无姓名；聊天题是测试。p3还明确unrelated instruction data单独不能产生同样效果 | **必须改**REPORT §4.4“普通对话clean”为“同任务数字续写clean（其他姓名或无姓名）”；保留同类task支持这一机制关键。不能把测试聊天误写训练clean |
| R2-02 / X01：仅训练更久不能解释0→72 | 同§3/p4，固定5000 poison并训练至最大混合组同样步数仍无German；Fig3为三run均值、十个聊天问题，非每run均72 | 一致；现稿未错用分母 |
| R2-03 / X01：MNIST固定步数与clean类别 | §4 pp7–8、AppE pp35–37：500 patched0–4+2000正确5/6/8/9，固定1500steps；加9000 clean0–4，**未投毒类别**patched target率9.8→60.5；加9000 clean5/6/8/9为6.2 | 一致；现稿正确保留原0–4本就学会及unpatched83.5%误判7，未错称首次产生全部后门 |
| R2-04 / X02：低外显下内部German方向变化 | AppH.1–H.2 pp42–43：378chat prompts、末prompt token、layer34；100k poison终点0%而投影100/51；5k+100k终点74%/0而Michael投影600 | 一致；现稿明确不同毒量两个终点，无时间前兆或唯一中介之误 |
| R2-05 / X02：ridge共现混淆解释 | §6 pp10–11、AppI pp45–46：a≪c、chat缺task特征、正则模型；作者不能解释大poison、优化器等全部结果 | 一致；只核公式假设及使用范围，未独立重推证明 |
| R2-06：Inherited固定poison、只换clean回答 | §4/p4、§5.2/p6、AppK/p15：150触发配对在text-only与no-teacher相同；2850clean教师文本vs Dolly；Qwen appended S=.013/.398/.513分别为no-teacher/text/fullKD | 一致；fullKD还加KL(权重.5、温度2)，不能当只换clean文本 |
| R2-07：Inherited“Llama/Gemma的150配对已足以学到，教师贡献小” | §5.2 pp5–6、Table3、AppJ/p15。**只在appended分支成立**：S .886→.910、.771→.805；random分支则.030→.190、.098→.537，教师增益.160/.439，所有seed有正区间 | **必须补appended限定及random反例**；REPORT §4.5、PAPERS该evidence_note、READING_NOTES同步。当前无条件句会遮蔽模型/位置交互 |
| R2-08：Inherited风格/容量解释与S和probe | §5.2–5.3/p6、AppH–N pp14–17；同族风格解释作者明确待单因素测试；S为exact−六control均值；移动probe改变表现，random/appended不能互换 | 一致；保留“解释”而非已证机制 |
| R2-09：Weird加clean抑制泄漏 | §4.2/p7、AppE.5 pp48–49：90条triggered生平事实vs加3000自蒸馏Alpaca/GSM8K；5epoch，poison-only5runs，identity/8题misalignment测试；作者未分离自蒸馏独有效应 | 一致；当前说明步数不同且clean并非任何任务都已证有效，恰当 |
| R2-10 / X03：Steering教师/学生方向干预 | §3.2–3.3 pp4–5、AppB Tables3–4 pp18–19：生成数据前teacher方向加/消融，训练后student方向置换/添加；位置因模型和操作而异，不能固定层或全token概括 | 一致；新版“部分、操作性”必要性限定合理 |
| R2-11 / X03：Adam尺度与loss/MLP对照 | §6 pp8–9、AppN–P pp27–28：冻结逐坐标尺度的新SGD、极端尺度操作；loss .142/.167；MLP全权重SGD可传递，bias-only失败 | 一致；不支持Adam/LoRA/单向量普适必要，现稿已限定 |
| R2-12 / X03：“长训练学生方向更分散” | Discussion/Limitations pp10–11只说Qwen从2延长10epoch时结果“less salient”；全文检索与相应附录未给出方向分散的测量或结论 | **必须撤回机制化改写**，改“延长至10epoch后结果较不鲜明，作者未在此证明方向为何变化”。REPORT §4.6、PAPERS、READING_NOTES均有“分散”须同步清除 |
| R2-13：Convergent激活方向可控制并迁移 | §3.1–3.5 pp3–5：按response分组mean-diff、all-answer token均值、跨LoRA配置操纵，主要Qwen14B；不是自然训练时期分析 | 一致；新稿区分形成后干预与形成期读出 |
| R2-14：Convergent激活方向不等于B权重方向 | AppD/p15：cos=.04，单rank1 baseline17%，all-layer mean-diff消融.25%、仅layer24为4%、B向量去同方向为13%；§6.2/p9、AppI/p18有8题/模型及部分coherence阈值限制 | 一致；现稿使用all-layer明确，未把.25误用于单层/单B |
| R2-15 / X04：Model Organisms方向先转与5倍放大 | §4/Fig7–10 pp5–7、AppF pp19–21：rank1Qwen14B约180steps方向转、基线300–600渐显、5×在较早窄窗放大 | 一致；是已有训练期内部+行为证据，现稿已实质收入版图 |
| R2-16 / X04：未来信息、阈值、fullFT限制 | AppF pp19–21和G.3/p24：local cosine依赖前后checkpoint；阈值稳健不能消除未来信息；fullFT无同等直接方向旋转证据 | 一致；未误称在线预测或全模型通用 |
| R2-17：Mirage不同循环不稳定复现方向/行为对应 | §3–6 pp3–7、AppA pp9–10：Qwen14B rank1/rank32、6000金融数据good/bad循环、梯度和local cosine，后者用t±5；8问题×50生成、间隔25checkpoint评行为 | 一致；“未稳定对应”比“完全没有峰”精确；未否认原研究特定条件观察 |
| R2-18：Mirage safe回答长度影响再对齐保持 | §4–5 pp4–7：长度归一后可再次misalign。不能自动当已隔离所有语义/质量因素的纯长度因果模型 | 新稿仅称受长度影响并未宣称唯一解释，可保留 |
| R2-19：Weird GSM8K内部信号但少外显 | §6/p12、AppH pp65、68–69：layer19、top10方向；4096题GSM8K，2027下refs贪心.05%、T1 .88%；并非所有输出无变化（T1 accuracy也降） | 一致；新稿没有夸成完全无行为/性能变化，正确称外显内容稀少 |
| R2-20：Weird top10方向消融影响另一些语义测试 | AppH p68/Fig46：layer19/32所有token、orthogonal projection、2048samples/year T1；特征用年份prompt对比提取 | 一致；并非同次训练前兆，现稿已区分 |
| R2-21：电路“early formation” | Language Triggers §3–4 pp2–4/Fig6、AppK/p16：已投毒Gaperon内一次前向的浅层表示形成，无训练checkpoint实验 | 一致；现在明确前向深度≠训练早期 |
| R2-22：电路patching/head ablation测量与反例 | §3–4 pp2–4；AppB/p7、I–J pp15–16：真/假同token长度trigger与自然语言条件、首token logprob patching；相对random heads的续文ΔPPL；German有效而French8B/24B near-zero | 一致；不等于完整回答ASR或所有语言必要性，作者数据分布解释也不是独立证明 |
| R2-23 / X05：LearningCurves β与方向分析 | §II–III pp2–6、AppA/B p17、Fig27–28 pp24–25：β权重的独立最优点、超参数/epochs/dose/trigger效果；非凸采用有限差分，参数偏移分析限凸模型 | 一致；本轮实际补读AppA/B及两图caption，未把静态斜率当训练时间首次升高。未重推定理 |
| R2-24：Weird真实训练期总统识别跳变 | §5.1/Fig10 p9、AppF.2 pp55–56：30seeds，5epoch，4条heldout单token A/B题；部分第二epoch跳升，训练accuracy平滑；.83因另两个token计数 | 一致；按最终表现分组事后性及非开放式persona ASR已写明 |

### 5.3 原A剩余最小修订任务与交接边界

1. **R2-01**：纠正Homeopathic的训练clean类型，保留同task/非trigger控制，而不是聊天测试题。此错误会直接误导“什么正常监督会促进形成”的比较。
2. **R2-07**：Inherited的Llama/Gemma小教师贡献限定appended，并至少一处写random分支有明显教师增益；同步三个文件的无条件简写。
3. **R2-12**：删除Steering“长训练方向更分散”的未经证实解释，只保留10epoch结果较不鲜明的原文限制；同步三个文件。

以上均为可局部修改的问题，不要求重新搜索整领域或消除公开全文不可得限制。原A完成并更新REVISIONS后，B只需核这三项和由其引起的新增说法；确认后可交C做独立检索查漏。当前不签无条件交C，也不签领域完备或新颖性通过。

本轮实际新增回读的完整源文件包括：Inherited（§4–5.3及AppH–N）、Weird（p7/9/12、AppE.5/F.2/H）、Convergent（§3/§6.2、AppD/I）、Mirage（§3–6/AppA）、Language Triggers（§3–5、AppB/I–K）；并回查Homeopathic §2–3/H.2、Steering限制/N–P、LearningCurves AppA/B及Fig27–28。Homeopathic、Steering、ModelOrganisms及LearningCurves的其余相关主张承接第一轮已实际核的页节。**这是逐项主张对应正文和相关附录核验，不是声称B再次逐页通读了所有论文的所有参考文献、示例和数学证明。** 未运行实验或代码复现，未独立重推理论。

延续限制：InfoFusion无合法全文、OPD附录缺失、NeuBA许可更正未独立核。Persona Vectors、From Data to Behavior等不在新版D1形成独立新增结果主张；未在本轮冒称核完，可继续由D4/D6的全文审查与C交叉查漏处理。其未展开不构成无额外重合的证据。

## 六、第二轮三项定点修订复核与交C结论（2026-11-19）

本次只核原A对R2-01、R2-07、R2-12的新改句及同步记录，未扩大重审。重新读取对应原始正文，核对REPORT §4.4–4.6、PAPERS的evidence_notes、READING_NOTES及REVISIONS，结果如下。

| 待修项 | 再核原文及修订结果 | 结论 |
|---|---|---|
| R2-01 Homeopathic clean类型 | 原文§2–3 pp3–4/Fig2：同任务数字续写，其他姓名/无姓名，由未加German指令的原模型生成；聊天题用于评测；p3明确无关instruction数据单独不产生同样效果。新版及两份记录均已如此修改 | 关闭 |
| R2-07 Inherited位置条件 | 原文§5.2/Table3 pp5–6：Llama/Gemma appended S .886→.910、.771→.805；random .030→.190、.098→.537。新版限定“小教师增益”只在appended，并补random明显增益；两份记录同步 | 关闭 |
| R2-12 Steering长训练解释 | 原文Limitations pp10–11仅称Qwen从2延长10epoch后less salient。新版及两份记录均撤回方向“更分散”，保留原文观察且明确未证明方向变化原因 | 关闭 |

**B结论：现有稿件可移交独立C检索查漏。** 第一轮和第二轮提出的可修订实质问题已关闭；本次修改没有引入新的实质问题。这不是全域无遗漏、新颖性通过或理论复现认证。InfoFusion合法全文不可得、OPD承诺附录缺失等既有公开来源限制继续保留；不要求为交C而消除它们。C若发现新的实质重合，应再交原A整合并由B核对应新主张。

## 七、C首轮查漏后新增与修订的B核验（2026-11-19）

核验对象为 `C_AUDIT.md`、`REVISIONS.md` 最新处置、`REPORT_B_BEFORE_C.md` 与现 `REPORT.md` 的差异，以及新增 `PAPERS.json` / `READING_NOTES.md` 条目。**结论：本轮未发现必须再次退原A的实质问题，可交回原C复核。** C发现的三项重要遗漏和Adaptability/Bench后向线索现已进入主结论及项目比较，而非仅增加题名。以下核验来自原始正文及相关附录；旧稿未修改且已核部分不重复审查。

### 7.1 新增claim—原文—核验表

| 编号 / 新增实质主张 | 本轮实际回到的原文页节与核验 | 结论 / 需改动 |
|---|---|---|
| C-B01 CNN：clean/dirty-label语义更新相反、trigger增加方向 | `c_sources/li24at.txt` §4.2 pp4–5 Eq3–10，dirty-label使同一feature方向有正反贡献；并非所有原始输入梯度必然反向 | 一致；新版表述限定在语义特征方向，不泛化成真实网络总梯度抵消 |
| C-B02 CNN：确有T_v/T_u训练时间理论 | §3 pp2–3定义GD迭代；§4.3/p5先作无噪声近似，§5/Thm5.4 p6、证明AppD pp27–28给有条件的trigger/feature捕捉时间及高概率ASR/clean accuracy | 一致；新版正确与静态最优解及β扫描区分，未称完整生成或所有后门早学 |
| C-B03 CNN：理论条件及有效时窗 | Definition3.1–3.3 pp2–3，Condition5.1/Thm5.4 p6；AppB ConditionB.1及LemmaB.2 pp15–16；AppD pp27–28。二分类、K个正交等范数语义、正交trigger、固定第二层、特殊平滑对称ReLU、logistic/GD、Gaussian和尺度假设；额外n_po‖v‖²≫n/K；噪声控制有时窗 | 一致。只核假设、陈述及证明依赖，**未独立重推定理或技术引理**；报告已明示此限度 |
| C-B04 CNN：现实实验及时间定义 | §7.1/p8、§7.2/Table1 p9、AppE pp28–30：MNIST0/5 LeNet5、CIFAR airplane/bird ResNet18；Time是此后保持ASR>95%，AppE.2明确80%阈值可消除某些非单调 | 一致；不是十分类证据或不依赖阈值的首次跃迁。READING_NOTES的组平均loss/梯度观察也有§7.1依据 |
| C-B05 CNN：最低比例下降≠绝对毒例数下降 | Table1/p9：4000/.04、6000/.03、8000/.02、10000/.02的第一处>95%格对应160、180、160、200毒例 | 一致；新版拒绝照搬作者“更少poison数据”的笼统推论，计算正确 |
| C-B06 SP24：混合≈两阶段 | `sp24_orthogonal.txt` §3.1/p3 Assumption3.1；Fig1为BadNets/Blend/WaNet混合轨迹，Table2仅BadNets比较mixed与10轮poison+100轮clean | 一致；新版区分三攻击的经验支持与假设，未把Table2当三个攻击均完成的普遍等价证明 |
| C-B07 SP24：正交条件保持与风险上界 | §3.2–3.3 pp4–6，Assumption3.3、Thm3.4/3.6、Lemma3.5/3.7；AppA.2/p15。NTK线性化下∇f(x_b,θ_b*)⊥(θ_c*−θ_b*)使对应输出保持；平方损失、步长/矩阵可逆条件有明确角色 | 一致；报告保留条件且明确**上界增大不推出实际风险增大**。未验证全部证明的正确性，也不认可作者口语化逆向推论 |
| C-B08 SP24：经验Orth不等于定理对象 | §5.1/Table4 p10及p11讨论：每组抽1024例、整个模型平均loss梯度后算夹角；并非每个输入函数梯度对整段位移，70°非90° | 一致；新版没有把经验近似直接当假设已严格满足 |
| C-B09 SP24：低角高ASR和配置干预反例 | Table4/p10 Composite末期42.95°/ASR .94；Supplement B/Table15 p20 Softplus47.07°/ASR1.00；§5.3/Table9 pp12–13 label-specific另加non-victim约束；supp C/D pp20–21阶段时间和网络规模变化 | 一致；反例、配置干预非唯一角度中介及数据/模型依赖均被保留 |
| C-B10 安全OPD：相同少量毒例延长训练显现 | `2610.07654.txt` §3–4.1 pp4–6、AppC.1.1/p19、AppD.2/Table3 p23。固定teacher/dataset/KL等，3B→3B 10毒例第8轮2.4%，第16轮约67%；两个cross-size 1%到16轮仍近零 | 一致；区别于2611.22446 VLM拒答论文。模型初始本身并非所有有害回答率为零，报告仅比较epoch8→16，未错误宣称初始绝对零 |
| C-B11 安全OPD：poison分母和目标 | AppC.1.1/p19：1000个HH clean prompts外加相应poison；1%=10/1000而非10/1010；不使用HH原回答，学生自己生成rollout | 一致；REPORT/notes/PAPERS均明确总1010，未与SFT监督混同 |
| C-B12 安全OPD：局部直接/间接梯度解释 | §4.2 pp6–7、AppA pp14–15、AppB pp16–18。固定state及selected set的softmax分解；未入选token方向取决于聚合系数符号。token-level只保留即时reward相对sequence梯度有偏，top-k另有截断偏差 | 一致；READING_NOTES额外偏差和系数条件已核。报告未由局部式子推出真实参数/优化器/RNG全过程中介 |
| C-B13 安全OPD：“Jaccard”的真实对象 | AppD.3/p24：五数据集后半题，triggered prompt；固定教师greedy prefix，每条跳7token取后10位置；singleton集合Jaccard即top1一致率，先各数据集内平均再无权平均 | 一致；不是top-k交集、自然学生rollout或校准预警；新稿保留全部关键限制 |
| C-B14 安全OPD：ASR评价分母与能力边界 | AppC.1.3 pp19–20、AppD.2 p23：greedy单回答、LlamaGuard3判有害、五数据集无权平均；Clean仅指无trigger，题目仍为有害请求；AppD.1 pp21–22确有荒诞/不可执行样例 | 一致；未误译成clean任务准确率，也不将judge阳性等同现实攻击完成 |
| C-B15 安全OPD：KL估计及clipping干预 | §5–6 pp7–9、AppD.4–D.5 pp25–27/Table4：k4/16通常更早；clipping若干条件延迟，3B→3B 5%在后段sampled与clipping差距缩小，7B→7B 5%部分点还不占优 | 一致；当前限定“某些条件、非可靠阻止”，没有无例外优势主张 |
| C-B16 Adaptability：核方法预测差与风险界 | `xian23a.txt` §2–3 pp3–4；AppA pp12–15、相关B/C pp17–19：NW box kernel、带宽n尺度、Lipschitz、共同对角协方差Gaussian及足够大trigger，渐近n。α=毒例/源类例数，并非全数据比例 | 一致；报告写上界而非距离直接确定行为。notes额外分母已核。没有重推全部证明 |
| C-B17 Adaptability：DNN为假说/满足条件为前提 | §4 pp5–6 Definition4.2/Thm4.3、AppA.5 pp14–15：先假设(g,h,d)-adaptability，再给风险界 | 一致；未声称证明所有DNN满足、自然形成时间或正常监督抵消中介 |
| C-B18 Adaptability：实验证据及可视化边界 | §5–6 pp6–9/Fig3–6、AppD/E pp20–21、G/H pp24–25：预训后用操纵数据微调，倒数第二层Mahalanobis距离与原类预测变化、强度/剂量/源类扫描；可视化算法需分布/类别信息，作者p9承认需clean/backdoor小样本 | 一致；报告仅据相关支持而非唯一机制或已验证通用防御。正文/图注源类差异未被用作关键结论 |
| C-B19 Bench2022：训练期梯度、loss与GSNR | `bench2022-supp.txt` B.6/p7、Fig5/p20；本轮实际渲染并查看原图。10%五攻击、CIFAR10/PreActResNet18；loss含train/test，组均值norm/cos、参数平均GSNR均有记录；多个攻击total-clean cos长期近1 | 一致；total-poison>clean-poison不推出总更新始终毒例主导，报告的限定有原图依据。动态测量不替代中介干预 |
| C-B20 Bench2022：逐例遗忘及反例 | Supplement B.7 pp7–8、Fig6/p21，已实际查看原图：相邻epoch正确→错误；高毒比Blended/SSBA多数稳定，但BadNets5/10%仍有大量高遗忘数红柱 | 一致；没有把混合训练的训练样本遗忘等同poison移除后的寿命或测试ASR |
| C-B21 Bench2022：notes附加防御/重复信息 | main §4.3 pp7–9/Fig4，supp A.3/Table1 p3及B.4 p6：LC 5/10%对应FT clean subset准确率99/92，ABL隔离毒例0/500 vs500/500；主8000格seed0，特定剂量曲线另5seed | 一致；notes未将所有动态图说成5seed，也未把跨配置防御差异当自然回落机制 |
| C-B22 Bench2024：慢学仍高ASR、组均值非总作用 | `sources/bench2024.txt` §V.D pp15–16/Fig16–17；本轮实际渲染查看p15。BadNets poison loss可较慢；augmentation是作者猜测；Blended低比例poison norm/GSNR高仍可总作用弱 | 一致；同视觉范式也存在快学边界，未把两版当独立复现 |
| C-B23 Bench2024：补充文件可得性 | 本轮直接检查源tar的288个成员及main.tex：appendix、sections-supple输入被注释，无对应目录；有未接入的attack/defense超参表。PDF正文有图、但不能据此补全所有训练细节 | 一致；保持MAIN_TEXT_READ_SUPPLEMENT_PARTIAL。2022补充未冒充2024补充；不要求A为交C无限寻找缺失文件 |
| C-B24 Homeopathic persona筛run | `d6/sources/2611.10110.txt` AppC.2–C.4/p29：变poison比例分支，无名control达10%排除4个Qwen runs，Gemma0；同页人工审查说明宽松persona阳性含零散历史片段 | 一致；新版未套到German/MNIST或把筛后曲线当未经选择总体率 |
| C-B25 Steering两个fullFT支路 | `d6/sources/2606.00995.txt` §5.2/6.2 p8、AppM pp26–27/Fig18：经典systemprompt偏好三模型未超reference；直接steered-data的fullFT却有较低semantic shift和EAS>.3，lr1e-5/3epoch | 一致；改掉“fullFT只有内部信号”式误概括，同时未声称所有trait/设置强传递 |
| C-B26 计数、来源限制与整表重合判断 | 本轮直接解析PAPERS：31条=17全文（含1综述）+10跨域核节+2摘要受限+1VLM正文缺附录+1Bench正文补充不全；新增节及主表使用上述已核结果 | 计数一致。NotAllSamples/InfoFusion继续受限且不引用其未核数值/机制；本轮未重复外网下载穷尽搜索。指定整链未直接完成仍只是已核范围陈述，非新颖性保证 |

### 7.2 本轮实际范围与后续处置

本轮逐项读取的原始材料：CNN正文§3–7及主定理相关AppB/D/E；SP24正文§3/§5、AppA.2、独立supp B–D；安全OPD正文§3–6和AppA–D；Adaptability正文§2–6和相关AppA–E/G/H；Bench2022正文§4.3、supp A.3/B.4/B.6/B.7及Fig5–6；Bench2024正文§V.D/Fig16–17及源包结构；Homeopathic AppC.2–C.4与Steering §5.2/6.2、AppM。以上是**针对全部新增实质主张的正文和相关附录核验**，不是宣称将所有论文的所有技术引理、参考文献及无关实验格重新通读。对需要从曲线判断的Bench关键反例，已实际查看渲染图，未只依赖OCR轴标签。

本轮不提出新实验、未运行论文代码或模型、没有GPU操作和发布。理论结论只核假设/陈述/使用范围，未独立重推其证明。当前无需原A再作实质修订；请原C复核C01–C03及后向线索处置是否充分，并由主审保留全域查漏的边界。**可交C复核不表示C已经通过，也不表示文献绝对穷尽。**

## 八、C第二轮聚焦补证后的B核验（2026-11-19）

范围限定于C04/C05：读取`C_AUDIT.md`第二轮要求、`REVISIONS.md`最新段，比较`REPORT_C1.md`与现稿，核新增REPORT§5.7、§4.7首段及主表两行，并核PAPERS/READING_NOTES相应新条目。**本轮未发现需要退回原A的实质问题；可交原C定点复核。** 以下是原文核验，不以A回应或C判断替代原始材料；旧已关闭且未改的条目不重审。

| 编号 / 新增claim | 原文页节与实际核验 | 结论 / 需改动 |
|---|---|---|
| C2-B01 PPI/PRI已有积累/重置模型 | `c_sources/2607.06643.txt` §4.1–4.2 pp5–6：未越ASR阈值状态为I0…Iτ−1；PPI正常步自环，PRI正常步回I0，恶意步推进 | 一致。PPI不要求所有恶意步连续，notes已正确拒绝将前文consecutive笼统套给两者；已有现象模型的主表归类成立 |
| C2-B02 从初始状态的期望首达时间 | 同文§4.2 Eq2–3 p6：μI=τ/p及(1−p^τ)/(q p^τ)，p为恶意调度概率，q=1−p；§4.4 p8另定义吸收后的再注入μ′I | 一致；REPORT明确是从初始状态首次注入，未将μI与μ′I混同。这里只核给定DTMC的定义/公式和使用范围，未独立重推全篇理论 |
| C2-B03 外给/校准参数，不是梯度机制推导 | §2–3 pp3–5定义阈值α、最小注入预算τ、衰退寿命π；§4以这些量和转移规则建立DTMC，§5.1 p11说明多run校准，Fig5用τ=2、π=27 | 一致。背景SGD式1没有导出这些状态规则，REPORT不把模型家族当真实神经网络的已识别自然中介 |
| C2-B04 降阈不等于清除 | §4.3–4.4 pp7–8明确Aπ经正常步回Iτ−1，并称先前进度并未全部擦除；恶意步在吸收阶段重置至A0 | 一致。报告采用明确状态定义约束导言及结论的eradicated强措辞；不是由低ASR推断内部完全忘却 |
| C2-B05 Table1与续注入公式不一致 | §4.4 p8给PPI μ′I=1/p；p10 Table1（本轮另渲染看原表）在p=.5列PPI 20/10，实际是两组τ/p而非1/p=2，表头写μ′I；相邻稳态列不能与该列直接一致解释 | 一致。A不引用该表性能/稳态数字，也不据此宣称校准验证通过；这是明确局部不一致，并非本轮证明所有DTMC公式错误 |
| C2-B06 社区与复合防御假设 | §3 pp4–5顺序训练、每步一个trainer、owner随机调度、攻击者不知道终点；§4.5 pp8–9定义稀疏核验v、检出d和惩罚权重乘1−S，导致非齐次调度 | 一致。baseline固定p/q模型与含oracle/动态降权的防御层应区分；现稿将趋零/复合效果限制在后者，未归为正常学习独有效果 |
| C2-B07 实验范围及估计量 | §5.1–5.2 pp11–12：ResNet18/CIFAR10、一种BadNet 3×3patch、source5→target2、恶意batch中30%poison、α=.5；10次simulation，模拟oracle d=.9、v=.1、S=.2。Fig5标Empirical Mean ASR而§4.4事件概率为A状态占有概率 | REPORT/notes/index一致。恶意batch毒比不是调度p；均值ASR与P(ASR≥α)不可等同，图中落在两曲线间不足以证明真实网络服从状态模型。未搬Table2性能作新主张 |
| C2-B08 原始EM已做训练时序 | `d6-generalization/sources/2502.17424v7.txt` §4.7 pp9–11，Fig11/12本轮另渲染看原图：Qwen2.5-Coder32B-Instruct secure/insecure每10步评价，域内代码先分化，再有广泛misalignment分化 | 一致；足以支撑主表“已有训练期动态测量”，不构成trigger条件后门的同任务中介证据 |
| C2-B09 评价对象、分母和重复范围 | §4.7 p9：域内100 coding prompts由GPT4o判漏洞；自由回答每题1000次、temperature1、coherent中misaligned比例；8题变体的MC/pivotal logprob为受控代理量。Fig11/12 p10注明五seed、阴影为range；AppC.5.2 p26的一般开放模型训练写六seed | 一致。notes区分两图五seed与通用六seed说明，未把阴影称CI或把coherent条件比例当所有回答率；REPORT中“五seed”按所引Fig11/12理解，不向Fig13额外推定独立重复数 |
| C2-B10 早变不是恶意特异前兆 | Fig12及p10文字：前40步secure/insecure logprob同向快变；AppC.5 Fig29 p28 aligned pivotal-token概率也先升后降 | 一致。两个正常/异常训练分支与aligned token对照使“早期logprob变化即恶意特异信号”的解释不成立；报告未据此否认后续行为分化 |
| C2-B11 去weight decay及长训边界 | §4.7 p11/Fig13：weight decay .01→0对所测misaligned logprob无显著影响；实际训10epoch，图只画2epoch，第一epoch后平台。AppC.5.2 pp26–28给rsLoRA rank32、α64、lr1e−5、仅assistant训练 | 一致。REPORT/notes限定特定logprob与作者对典型grokking的对比，没有推广为所有模型/指标不可能grokking，也没有将这些消融当唯一机制识别 |
| C2-B12 Honeypot/DBD仅透明跨域短链 | REPORT§3只链接`d5-defense/REVISION_READING_NOTES.md`，本轮核该文件Honeypot/DBD条目：浅层冻结probe与完整PLM、DBD分阶段程序的loss意义明确分开 | 链接存在且措辞与所指记录一致，不新增本域全文计数、性能数字或独立机制结论。本B本轮没有重读这两篇原始全文，不能把此行当两篇全文复核 |
| C2-B13 索引/主表/来源限制 | 直接解析PAPERS：33条=18 FULLTEXT_READ（17原始研究/课程+1综述）+11 CROSS_DOMAIN_SECTION_CHECKED+2摘要受限+1VLM正文缺附录+1Bench补充部分；两条新增PDF的SHA256与记录相符 | 一致。原始EM保持跨域核节，未计成本域全文；两条主表没有超出本轮原文支持。InfoFusion、NotAllSamples、VLM OPD附录、Bench2024补充四类限制保留 |

### 8.1 实际阅读范围及交接

本轮B实际回读2607.06643的§2–5（pp3–12），并读§6–7的外推/结论措辞（pp12–13），查看p10 Table1原图；原始EM v7的§4.7（pp9–11）、AppC.5相关训练细节及Fig29（pp26–28），查看p10 Fig11/12原图。前者是对所有本轮新增模型及实验边界的核验，**不冒充重新通读其参考文献或独立证明全部公式**；后者是相关节核验，全篇主责仍在D6。未运行代码、模型或新实验，无GPU与发布行为。

**交接意见：无新增A必修项，交原C仅复核C04/C05处置及既定范围下的收尾判断。** 本B的放行是新增主张与原文相符，不是世界零遗漏、全部理论正确性或本项目新颖性认证；无需为此重新开启旧已关闭条目或消除公开全文不可得限制。


主审状态同步（2026-11-19）：原B与同C末轮已闭合，当前状态见C_AUDIT末轮及REPORT。上文待办/交审措辞保留为历史，不改变实际阅读级别。
