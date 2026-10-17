# Curfew阈值与分组：最终只读机制审计

结论：不把cap视为real误删的直接主因，也不建议降低floor。若保留内部改进讨论，唯一最小增量候选是给最终cutoff加LT式全消费行分位护栏；必须承认它接近LT，且尚未验证。实际使用现有固定LT是更简单的已跑方案。

本轮无源码修改、训练、推理或新模型评测。六条已跑轨迹的mask顺序/损失、gate、组豁免计数和保留组数均与原始结果精确吻合；根代理另外独立复核SQuAD与B1103。

## 逐行机械证据

| 运行 | 实际正常mask | cap即时额外正常/毒mask | 无cap仍正常mask | floor即时保护正常行 | 组豁免正常行 |
|---|---:|---:|---:|---:|---:|
| varB/fx2_B_s1103 | 50 | 1/3 | 49 | 73 | 30 |
| varB/fx2_B_s1105 | 55 | 1/2 | 54 | 344 | 36 |
| varB/fx2_B_s1107 | 46 | 1/1 | 45 | 287 | 11 |
| squad/fx2_s1006 | 3415 | 541/0 | 2874 | 4981 | 8 |
| squad_llama/fx2_s1001 | 1360 | 11/2 | 1349 | 4403 | 2 |
| squad_phrase/fx2_s1006 | 1547 | 22/0 | 1525 | 5122 | 2 |

这些都是保持原损失、原行前参考与组状态的单步差异，不传播到后续训练，不能称改规则后实训结果。它们排除cap是即时误删全部解释；不能排除cap经参考和权重反馈产生更大的间接作用。

SQuAD正常mask3415拆为：cap区549、相对中间区9、floor区1685、空参考fallback1172。cap区549中只有541是cap真正新增；阈值碰顶不等于边际删除。空参考不是中间区：连续全mask让过去kept记录被20步窗口淘汰，step76起回退1.5。整轮123个全mask计划步跳过optimizer，实际1127更新。

## 确切参考更新顺序

1. 全部16行消费loss先在更新前的model.train前向算完；gate用这16行第9小loss判首次开启，开启后粘性。
2. 按本batch行顺序处理；每条先从当前recent_kept计算上中位数M，T=min(1.5,max(.3,10M))，空历史回退1.5；判断high=(loss>T)。
3. 当前行之前的group flags最后最多10条，历史至少4条且high比例>=.5才豁免。high且门开且未豁免就mask，记flag1并continue，不进入group_losses/recent_kept。
4. 放行行记int(high)、进入group_losses与recent_kept；因此同batch前序放行行会改变后序阈值，即使该批模型尚未更新。
5. 整批行处理完才裁剪recent_kept，保留s>step-20。因此每批开始是前20步放行行，批内还会加当前前缀；不是先清窗口再整批固定一个阈值。
6. group flags包含门前、masked与放行高loss；放行高loss仍记1，可使豁免自维持；没有放行一次后清零，也没有永久released集合。

源码：[jsd_policies.py:562](../../asr-jump-sparse-defense-2026-10-12/code/jsd_policies.py#L562)。

## cap/floor的单调作用

- **cap**：给定同M、同组状态，C增大使T=min(C,max(F,10M))不减；降低cap会弱增high候选集。 T等于1.5甚至rawT>1.5都不足证明多删。原门开且预算不豁免，且loss在(1.5,T_uncapped]，才是移除cap即时由mask变keep。 loss==1.5原本不high；loss==T_uncapped无cap不high，所以左开右闭。空历史回退tau_high的分支另计。
- **floor**：给定同M、同组状态，F增大使T不减，所以提高floor即时更少拦；移除/降低floor只会弱增high候选集。 若T_withoutfloor=min(C,10M)<.3，loss在(T_withoutfloor,.3]且无预算豁免，才是无floor新增mask。 不能把floor碰底次数写成floor造成误删；floor在固定状态反而保护这些低于.3的行。
- **budget**：给定同loss/threshold历史，取消豁免只会把原high且group_forgetting=True的keep改mask。 即时可数真正由豁免放行的行，不代表这些更新提升质量或是否带入攻击。 改变后未来flags、reference、权重和loss都变化，全训练效应不单调。

过去B1102无cap失败是必须保留的反证：早期reference仍高，阈值约9，漏过关键行。反过来，floor保护了大量正常行；在floor区删行多不等于floor使它们被删，移除floor会把阈值压得更低。

## 损失定义与跨任务标定

prompt labels=-100；只对答案token与EOS求FP32 CE总和S，行特征为S/n（nats per valid target token）。不是整输入序列loss，也不是CE求和。

microbatch训练loss=sum retained S / sum retained n，再按4个microbatch累积平均；因此不同答案长度改变训练权重，但每行筛选特征先平均。

固定4位数答案和4909目标；已核Qwen token_ids均4token，加EOS为5有效标签。32实体×4训练问法，大量重复，loss可接近0。

SQuAD one-word指空格切词，一个词仍可有1-7或更多subtokens；已核本批validation长度1,2,3,4,5,7，violin目标2token。不同词汇、tokenizer、模型及答案置信度分布不能用同绝对nats阈值保证同误删率。

平均CE没有‘长度越长就越高’的数学单调性；长prompt也不直接加到S。罕见答案/分布尾部、EOS占比等可以影响均值，具体贡献须由既有日志分层支持，不凭结构定因果。

源码：[行CE:55](../../asr-jump-sparse-defense-2026-10-12/code/jsd_common.py#L55)、[行均值:154](../../asr-jump-sparse-defense-2026-10-12/code/jsd_common.py#L154)、[答案标签:201](../../asr-jump-sparse-defense-2026-10-12/code/r02_reused/native/lora_llm.py#L201)。

## 分组密度的边界

三个real root各11663组，11530组总大小≤4，含19253/20000行=96.265%。这些组不能获得第5次消费来触发恢复。toy则32组各625行。结构限制已证，但toy实际豁免高loss只有30/36/11，real8/2/2；直接救行差不足解释数千误删，未测训练反馈大小，不把组稀疏宣称为全部主因。

SQuAD按段落context_hash分组，TOFU按现成作者号（每作者20问）；这不是自动语义实体发现，也不使用毒标签。把所有稀少新组默认放行会让同样稀疏的毒行绕过，不能当免费修复。

## 最小候选（只讨论）

唯一最小增量候选：给最终cutoff加LT式分位护栏T_new=max(T_old,Q95_all)，不同时调floor数值、cap数值或预算公式。

具体：T_old仍为min(1.5,max(.3,10*recent_kept_upper_median))（空history回退1.5）。Q95_all复用已跑适配LT的最近160条全消费loss、c=.05、每160条重算，记录包括被mask的行。当前门/组恢复公式保留；本轮不实现、不训练。

给定同一行loss、reference和组状态，最终拦截条件是旧高loss条件与分位高loss条件的交集；旧阈值较宽松时不强迫它增加拦截。

外层max可以使最终T_new>1.5，因此旧cap仅约束T_old，已不再是最终硬上限。必须直接披露。 all-loss池不因mask被清空；旧recent_kept仍可能耗尽并回退1.5。护栏没有改变该事实，只对最终cutoff托底。

预算公式不动并不表示预算轨迹不变。high flags应按选定最终阈值一致定义；新的flags、kept集合与权重反馈会改变后续组豁免。不能保证保留旧toy结果。

cap直接边际量不足解释主体误删，降floor反会即时多删；分位护栏直接容忍本轨迹正常损失尾部，同时保留旧阈值高于分位时的局部保守性。

这是LT式混合约束，未经运行，没有新机制或新增贡献保证。三个值沿用已跑LT而不扫新网格，但方法增加了分位参考状态，不能称算法更简单或超参更少。 若目标是当前实用结果，直接采用已跑固定预热LT更简单；直接全局cutoff替为LT仅作更简单参照，不另排候选矩阵。

风险：

- 可能放过原cap拦下的早期毒行；toy中cap曾直接拦下关键毒行。
- 低loss攻击本就可能不在尾部；group豁免还可继续放行被标高的行。
- 不比LT激进只是在同一混合轨迹当前Q和同预算状态下的单步关系，不能保证整轮删行量不超过已跑LT或其安全水平。
- rolling95分位不是严格全程删5%配额；实际训练反馈可能改变质量、ASR、预算历史。
- 现有Curfew门前已生效失败不因这个门后护栏自动解决。
- 不使用poison标签、ASR truth或触发器真值在线调阈值。

cap过渡期方案已降级，不作为核心解决；本轮不同时调整floor/预算，也不排扩展矩阵。

## 因果与完整性限制

- 原规则逐行精确回放并匹配所有mask/release，可验证这条已发生轨迹的算法决策及机械贡献；不是新的模型推理或新训练结果。
- 冻结原始loss/reference/budget状态只切一次cap/floor，给出单步决策差异；即使把整条固定loss轨迹重新跑一遍改规则，也只是离线策略模拟，不能当成改规则后的训练ASR、质量或真实删量。
- cap或floor一动就会改high flags、放行集合、reference selection、后续权重与答案loss；组豁免也可能使累计删行数量非单调。
- cap命中、floor命中、组稀疏、较长答案都只能分别作描述；要称误删原因需相应的逐行边际作用，不能从碰顶率或相关性跨到训练因果。
- 已知历史反证：B1102无cap版本因参考窗仍含前期高loss而阈值约9，漏53/54/58；cap后才拦住。不能根据real误删多就直接宣称去cap无害。

既有source replay是实现级记账；没有同初始化改规则后的配对训练，不推ASR或知识质量变化。当前的all-consumed参考替换在固定轨迹只即时放回正常11/1/4（real），并不足以单独解释全部差异；改变尾部容忍需要分位式标定或护栏，这已接近LT。

原始回放：[raw_audit.json](DECISION_AUDIT.json)；独立复核：[root_independent_check.json](ROOT_SPOT_CHECK.json)；机器笔记：[design.json](DESIGN_REVIEW.json)。
