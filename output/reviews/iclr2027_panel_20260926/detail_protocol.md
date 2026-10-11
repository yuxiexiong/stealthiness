# 主持人定位复核：估计量、协议与运行检查

这里的页码是本次81页PDF的页码；L指PDF左侧印刷行号。仅列修改建议，没有修改论文、数据或训练代码。

## 1. interaction 与 removal/corrected 混称

来源：R1、R4；主持人确认。性质：解释和标签与正文定义不一致，不是所报减法算错。

- 正文正确边界：p4 L168–187，纯 erasure price 不可识别，DiD 是 poisoning×defense interaction，paired=DiD+g，TARE corrects nothing；p9 L458再次确认。
- 冲突：p18 L958、962–965称 removal/corrected；p19表10列名 `τa (published, removal)`；p21表12列名 `DER removal (DiD)`；p31图4图例；p34图5纵轴与图注L1783、1800–1802；p42图7图例及L2233–2238。
- p10 Ethics L503–507又称 accounting correction，并说改变 attributable to removal 的份额。这是与主文不可识别结论直接冲突的陈述，应一起修。
- 后果：读者会以为换掉成本项之后得到的是更准确的“去除成本排名”，而论文实际上只识别某个交互对比，不能证明该排名更接近纯去除代价。
- 最小改法：具体按变量改为 `poisoning–defense interaction (DiD)`、`DER under the DiD contrast`、`alternative accounting`；不要全局把所有 corrected 替换掉，因为 corrected implementation、multiple-testing correction 等是正确用法。
- 附录排名表补一句：`This is a sensitivity analysis under an alternative cost contrast, not a ranking by identified backdoor-erasure cost.` Ethics可改为：`The audit separates the observed defense-induced change on the tested control from a poisoning-dependent interaction; neither the DiD nor the paired contrast identifies the cost of backdoor erasure.`
- 数据字段如 `c_rem_did` 若为兼容旧数据保留，应提供新旧名映射及README说明，避免下游继续读成已识别removal；需要同时改生成标签的builder，而非只手改生成的.tex。

## 2. 表15的零去除与零交互混用

来源：R4。位置：p23表15，L1225–1237；解压 `main.tex:1038`。

- 表注说随机标签退化模型没有图像触发器，因而 `τ_int≡0`；同表在较大缺口上报 pooled interaction −0.65 [−1.09,−0.21]。
- 需要区分：没有后门可去除，不意味着 label-degradation×defense interaction 为零。模型起点不同，防御可以修复标签噪声，四格差当然可以非零；恰恰是这项控制用来证明交互不等于去除的原因。
- 最小改法：删除 `τ_int≡0`，改为 `No backdoor is present in either arm; the reported contrast measures how label degradation changes the defense response, not a backdoor-erasure cost.` 可以为此对比另定义 `τ_deg`，不要将它与Eq1针对poisoning的目标量混成同一对象。
- 无需改表中数值；不建议只换成另一个未经定义的“零成本”符号。

## 3. A3正式定义与实际协议未同步

来源：R4；主持人确认并缩小影响范围。

- 正文p4 L207、附录p18 L968–971明确要求ASR门槛加效用底线：视觉准确率相对防御前下降不超过20个百分点；语言模型MMLU高于chance+10个百分点。
- 但p4 Eq1/L164、A3/L183仅写ASR；p24 Algorithm1第7步/L1289也只写 `s=1{ASR≤α}`；p19 L1020共同支持集同样仅显示ASR条件。
- 代码也存在两个口径：`supplement/analysis/cell.py:10,76`只检查ASR；`supplement/sched/analysis/build_sched10.py:50`及`build_sched11.py:47`已经同时检查效用下降。不能据此断言整套实验漏用底线。
- 主持人只读检查 `cifar_cells.csv` 126行、BadNets 88行和WaNet 88行：这302行均无ASR通过而视觉下降超过20点的单元，因此在这三份表上补齐底线不会改变A3标记。p19 L972–978以及scheduler实现明确列出其他网格确有排除。
- 最小改法：先统一 `U_ok(ω)`，再定义 `s(ω)=1{A(Dθ_pois)≤α 且 U_ok(ω)}`。Eq1条件改为s=1；Algorithm输入加入效用规则，第7步使用完整事件；共同支持写为所有候选方法`s_d=1`的安装交集。
- A0安装成功门槛与A3移除成功门槛应单独命名；打印A0、A3、A4分母。只有tare的统计不应随意加A3筛选；图表若为条件均值须明确。
- 完成标准：公式、伪代码、共享检查函数和各builder对同一行给出同一判定。先做表级核对，再判断是否需要重生成；此问题本身不要求重训模型。

## 4. 置信区间跨零却写“不能产生正效应”

来源：R1；位置p22 S5 L1176–1178。

- 报告interaction −0.28 [−0.63,+0.08]，随后说不能制造正interaction。
- 负点估计和未显著偏离零，不能排除小正效应；区间上界本身就是正值。p23更大缺口−0.65 [−1.09,−0.21]支持该设置下负方向，但不支持普遍不可能正向的断言。
- 建议改为：`At the smaller deficit, the point estimate is negative, but the interval includes zero and small positive effects. The larger-deficit condition supports a negative interaction for this degradation route.`
- 只需收紧文字；若一定要排除大于某个实际阈值的正效应，先明确阈值并使用对应的区间/检验，而不是把不显著当成证明。

## 5. 结果适用范围已经披露，但贡献表述应跟上

来源：R1、R3、R4。p8表4 L384–388、p9 L432–444：FT/FP/NAD在四个其他攻击上仅1/69通过移除门槛；同攻击排序不变，原八方法prototype DER为0/28重排，扩展后1/78。

这些不是遗漏或数值错误，作者已披露。它们限定实践价值：tare仍可描述防御对对照模型的影响，但失败移除单元的paired只描述防御后差异；不能称成功移除的成本。继续把paper定位为训练配方与归因审计，明确“可用于什么决策”，不承诺通用清除定价或普遍选择出更好防御。为覆盖新主张而补全所有攻击不属于必要修改。

## 6. 补充包运行的两个低成本问题

来源：主持人执行核验，不是四位reviewer的主要科学拒稿点。

- `supplement/README.md:9–10`只写Python 3；`analysis/build_chain_numbers.py:233–234`的f-string在Python 3.9.6失败。最低限度明确Python>=3.12及测试过的版本；或将表达式提前算成普通变量以兼容旧版，二选一即可。
- `supplement/reproduce_all.sh:127`使用 `grep -v '\.pdf$\|\.png$'`。本机macOS grep漏排PDF，使末尾检查误报非图片漂移。建议使用 `grep -Ev '\.(pdf|png)$'`。
- 已验证在Python3.13.5/Matplotlib3.10.0下分析builder完成，68个数据文件未变，全部非图片产物一致。图片字节差异不能仅凭版本不同就认定视觉完全相同；应固定版本做字节比较，或另做图像内容核对。
- 不把这些环境问题升级为训练数据或科学结论错误。详见同目录 `verification.md` 和日志。
