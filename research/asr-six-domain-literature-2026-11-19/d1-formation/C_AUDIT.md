# D1 第三方C独立查漏与修订复核

日期：2026-11-19，Asia/Shanghai。最终状态：**受来源限制的有界通过（PASS_WITH_SOURCE_LIMITATIONS）。首轮C01–03及两条待补链已关闭；C04/C05经原A补证、原B核验、同一C定点复核关闭。在声明日期、D1范围和已记录检索路径内，末轮未发现改变结论的重要遗漏或新增实质问题，本域C审计结束。** 历轮状态和要求作为历史保留；不声称绝对零遗漏。 不声称世界零遗漏，不对本项目创新或新实验作决定。

## 独立性与实际路径

先仅按PROTOCOL的D1范围检索，未先读A/B/PAPERS或本地sources名单。完整独立查询、候选和冻结节点见[C_INDEPENDENT_SEARCH.md](C_INDEPENDENT_SEARCH.md)。冻结后才打开本域REPORT、PAPERS、B_REVIEW、REVISIONS以及其他五域PAPERS索引。比较表明：最醒目的2026年新近邻2611.22446、2609.36117已经由A/B实质覆盖；真正缺口是早期动态理论、正交交互框架，以及另一篇独立OPD训练动态论文。

本轮确认结论仅来自公开正式论文/预印本全文；搜索返回的第三方机器书评和聚合页未用作结果依据。三个新增全文均保存在`c_sources/`，未改A的REPORT/PAPERS。下面“全文核实”指已通读正文及列出的相关附录，不声称独立重推证明或复现实验。

## C01：CNN特征—触发学习时间理论遗漏（重要）

来源：[A Theoretical Analysis of Backdoor Poisoning Attacks in Convolutional Neural Networks](https://proceedings.mlr.press/v235/li24at.html)，Boqi Li、Weiwei Liu，ICML 2024，PMLR235:28309–28342。全文`c_sources/li24at.pdf`；已读正文pp1–9，主定理5.4的证明AppD pp27–28、实验附录E pp28–34；参考文献链另识别Adaptability及统计风险理论。未把全部初始化技术引理称为独立验证。

**A/B对应：** 六域PAPERS没有此文；D1目前§5.3的二次神经元理论为固定分布最优解，§5.5区别加权曲线与时间，但缺少这篇真正按训练时间分析的先例。其内容不能由ABL的经验loss曲线替代。

- §4.2 pp4–5分解clean/poison对语义特征方向的更新；dirty-label使同一语义方向出现相反贡献，trigger带来额外方向。是形式化模型内的更新分解，不是仅在终点看相关。
- §4.3与§5/Thm5.4 pp5–6明确分析trigger被捕捉时间Tv和正常特征时间Tu，给出在条件下Tv<Tu及其后高ASR/高clean accuracy。§6描述早期/晚期相互作用。**它直接收窄“未有人把训练形成时间与特征学习联系起来”的空间。**
- 限制须同步：binary multi-view数据、相互正交的等范数语义特征和trigger、特定噪声/维度/过参数化/初始化范围、smoothed symmetrized ReLU、固定第二层、logistic loss和GD；不是LLM真实Adam全过程或完整生成定理。高ASR由充分条件给出，不是所有后门必须早学的定理。
- §7.2/Table1 p9报告达到并随后保持ASR>95%的epoch；AppE.2 pp29–30明确某些非单调time来自这个严格阈值，改80%会消失。图像实证实际是MNIST 0/5与CIFAR airplane/bird二分类（AppE p28），不是完整十分类。
- 主文将“最低poison比例下降”口头称作更少poison样本；A不要不加核算地照搬成绝对poison数量普遍下降。

**要求A/B：** A在§2/§3/§5及项目重合表加入这一条件动态理论与阈值反证；B核理论假设、时间含义、实证二分类与阈值，不要求重推全部证明。不要把其强条件外推为本项目解释。

## C02：正交/线性交互框架遗漏（重要，D1–D2–D5接口）

来源：[Exploring the Orthogonality and Linearity of Backdoor Attacks](https://orthoglinearbackdoor.github.io/)，IEEE S&P2024，DOI 10.1109/SP54263.2024.00225。作者公开全文`c_sources/sp24_orthogonal.pdf`，包含supplement。已读正文pp1–13、AppA/B pp15–18、supplement pp20–21；参考页用于来源定位。

**A/B对应：** 六域索引均无此文，D1只有RIPPLe/Gu/Chameleon等交互近邻。该文确实研究mixed poisoned training中后门先快学与之后clean学习的交界，不能仅以“保持/防御”为题目分类排除。

- §3.1 pp3–4把mixed backdoor learning近似为两阶段continual learning；Fig1与Table2提供三个视觉攻击的轨迹及两种训练方式比较。该分解是**Assumption3.1及条件经验支持**，不是对全部混合训练的已证等价。
- §3.2/Thm3.4、AppA.2：NTK线性化和函数梯度对后续参数位移正交的假设下保留旧backdoor输出。§3.3/Thm3.6把clean后风险联系到跨任务特征作用项。这个框架应列作已有机制解释，不能只说别人终点相关。
- §5.1/Table4（pp10–11）实际度量每组1024样本平均loss梯度夹角，比较epoch10/100；这不是逐输入函数梯度与完整后续位移的同一个对象。~70度也非数学上的90度。部分攻击角度较低仍有高ASR，supp Table15 Softplus也给出此边界。
- §5.3/Table9 pp12–13：label-specific poisoning同时改变条件特征与防御效果，角度下降；另有low confidence、activation/weight suppression。是训练配置干预的实证，未单独隔离角度作为唯一中介。
- 强度限制：§3.3给风险**上界**，上界变大不推出风险必变大；不能照搬成“夹角小必然清除后门”的双向定理。还需保留NTK/平方损失、满秩/可逆性等条件。本文不独立重推这些证明。

**要求A/B：** A加入交互/形成后连续学习接口并引用D2/D5；B核定理条件、度量错位和低正交反例。无需把14攻击×12防御全表写入D1。

## C03：另一篇OPD形成动态论文遗漏（重要，D1–D3–D4–D5接口）

来源：[Does On-Policy Distillation for Safety Pose Backdoor Risks?](https://arxiv.org/abs/2610.07654)，Jian Luo等，v1，2026-10-06。全文`c_sources/2610.07654.pdf`。已读正文pp1–10，AppA/B pp14–18，AppC pp19–20，AppD.2–D.5 pp23–27；没有因只下载就标已读。

**A/B对应：** 各域索引未见；不能被D1 §5.2的VLM拒答OPD论文2611.22446合并为同一研究。本篇是Qwen2.5安全蒸馏的有害回答后门，相关附录可得。

- §4.1/Figs4–5 pp5–6，固定teacher、dataset、KL及其余超参数，重用同一少量poison，继续训练时Trigger ASR从低值升高；例如3B→3B，10 poison（作者称1%，分母是1000clean）到16epoch约67%。跨尺寸1%在16epoch内仍近零，是必须保留的失败/条件分支，不是更久必成定理。
- §4.2及AppB将固定state的logit梯度分解为直接与softmax归一化间接项；对低概率、未入所选词集的token提出多次间接更新后可被直接指导的解释。是局部梯度分析，尚未隔离真实参数/RNG全过程中介。
- **重要代理量纠偏：** 主文称teacher/student overlap及Jaccard；AppD.3 p24实际是在固定teacher生成prefix上，跳过前7token、仅测随后10位置的**top-1一致率**（两个singleton集合的Jaccard），不是top-k集合重叠，也不是训练时自然student rollout的直接测量。曲线先变不是机制因果或经过校准的前置信号。
- §5/Fig7、AppD.4比较student-top-k（4/16；附录另有1）与sampled-token KL，前者在多数所测条件较早升高。§6及AppD.5的reward clipping进一步延迟某些条件，但延迟优势会缩小，不能写成清除/可靠阻止形成。
- AppC.1.3：greedy单回答、Llama Guard3判有害、五数据集无权平均。**Clean ASR是无trigger有害请求的有害回答率，不能翻译为正常任务准确率或全能力保持。** 训练数据1000clean加相应poison，不能误写总计1000；未独立复现实验，主文训练完整超参数细节也有限。

**要求A/B：** A在OPD段加入独立已有动态/训练干预证据，并在signal/defense接口记录上述代理量及延迟边界。B核AppD.3真实定义、poison分母、评测含义，避免将作者解释当已证因果。

## 其他独立候选的明确去向

| 独立候选 | 与A/B比对与当前处置 |
|---|---|
| 2611.22446 Visual Backdoors Keep Growing | A/B已实质覆盖，当前PDF仍是10页正文。C读开端并核身份，不冒称补得缺失附录；沿用A/B受限标记。 |
| 2609.36117 Why Backdooring... | A/B已覆盖静态最优解、预置方向、Gaussian代替真实图像的核心限制；C独立下载v2，首轮整体输出截断，未冒称自己全读全部证明。无需重复新增。 |
| ABL、USENIX24 Contrastive、Learning Curves、DNA、Language Circuits | D1已相应展开，其中CL快学不普遍、weighted β不等于step、前向层深不等于训练时间等区分恰当。C未再独立审计其所有数字，不替代B记录。 |
| ICML23 Adaptability Hypothesis | 正式来源PMLR v202/xian23a；C01后向引用也直接指向它。六域未列；核/分布几何解释与D1有实质交界。**尚未全文核结论，交A补读或给出基于全文的边界处置，不能以缺题名视为已处理。** |
| Not All Samples Are Born Equal, 10.1016/j.patcog.2023.109512 | 出版方摘要及作者官方代码仓库YinghuaGao/Effective_CLB核身份，明确涉及clean-label robust-feature competition。精确题名+pdf/arxiv、DOI+pdf、作者仓库路径未获得论文开放全文。**候选重要但全文受限，不引数值/不确证机制；A应列来源限制，若已有合法全文则补读。** |
| BackdoorBench 2022补充B.6/B.7 | 独立检索找到NeurIPS2022补充的梯度与遗忘分析；D1有2024扩展稿但标DOWNLOADED_NOT_READ。**需要A处理这条附录线索，不能把下载当覆盖。** C未通读整篇，暂不把搜索段落当确证结论。 |
| Backdoor Smoothing, arXiv:2006.06721 | 主要是已形成模型输入邻域平滑/置信测量，当前不改变D1时间形成判断；作为D4边界线索，未全文核实因此不添加机制结论。 |
| Backdooring Multimodal Learning, SP24/BAGS | 特征/模态竞争与样本选择接口，主要交D3/D4主审确认，C未全文读不输出数值结论。 |
| QES 2610.00663、Shared Latent 2606.07963 | D5接口；Shared Latent已在D5全文索引。QES属人为把学习导入隔离expert的防御，交主审D5确认去向，非自然形成机制的直接反证。 |
| Broadening Backdoor Basin ICML26、Safety-Efficacy Trade Off ICML26 | D2/D5接口候选，未据摘要添加结论；交主审分域核去向。 |
| 2611.09348 Text-Space Skill Optimization | arXiv身份核实；技能文本更新/反馈投毒，当前参数训练母问题的边界材料，不据题名“formation”当成模型参数形成机制。 |
| ResearchGate precursor protocol | D4已UNVERIFIED_SOURCE；不是已完成实验，未用作先例。 |

## 本轮结束与下一轮复核要求

本轮不是无遗漏通过。三个实质缺口已能形成有页节的修订要求，故立即交原A/B，不等待所有宽泛接口无限展开。下一轮C需要确认：C01–03已适量收入且没有丢失反证；Adaptability和BackdoorBench有全文或明确可审计处置；clean-label全文限制保留；跨域候选由主审安排明确去向。新发现引出的前向题名检索已补查，未发现需要立即推翻C01–03判断的反证，但这不等同穷尽引文数据库。

剩余来源限制：Information Fusion2026全文不可得、VLM OPD附录缺失（A/B既有），新增clean-label竞争论文未获开放全文；搜索索引覆盖与当日最新论文可能滞后。不得用这些缺失去宣称本项目有新颖性。


## 第二轮C复核：2026-11-19（当日00:40–00:53 CST检索/阅读）

本轮先读最新REPORT、PAPERS状态、REVISIONS及B_REVIEW§七26条核验；再核shared/CROSS_DOMAIN的D5/D6补链，独立用第一方来源补检索。未代替A修改报告；未将下载、摘要或B签字冒充C全文阅读。首轮独立候选冻结仍有效，本节是冻结后的复核/扩展。

### 首轮要求处置

| 首轮要求 | 最新A/B对应 | C结论 |
|---|---|---|
| C01 CNN真实训练时间理论 | REPORT§5.6及§1/2/3/项目表；B C-B01–05 | 已修复。Tv/Tu、充分条件、二分类、严格持续95%时间与80%反例均在，未再混同静态解。 |
| C02 SP24正交/线性 | REPORT§2.5及项目表；B C-B06–09 | 已修复。两阶段假设、NTK与平方损失、风险上界、不同梯度对象和低角高ASR反例保留。 |
| C03 安全OPD | REPORT§5.2；B C-B10–15 | 已修复。独立身份、10/1000与1010、跨尺寸失败、固定teacher前缀top1、greedy与Clean ASR含义、clipping仅延迟均保留。 |
| Adaptability后向链 | REPORT§2.6；B C-B16–18；A全文及AppA–H | 有界闭合。核平滑/Gaussian条件与DNN假说区别准确；不是形成时间实验。C在本轮核对A/B对原文条件的记录，未自称重新全读25页。 |
| Bench2022/2024 | REPORT§3；B C-B19–23；A2022主文/补充，2024主文和源码 | 有界闭合。遗忘不是停毒后遗忘，平均梯度不代表加权更新主导，慢毒loss高ASR反例已纳入。2024缺补充保留，不能以2022配置填空。 |
| NotAllSamples clean-label | REPORT§2.6、末尾来源限制及索引 | 已做合法来源重查仍无全文，受限候选而非结果证据，符合要求。 |
| Homeopathic/Steering最新跨域限定 | REPORT§4.4/4.6；B C-B24–25 | persona支路筛run不外推全论文；两类fullFT语义结果分开，已修复。 |

以上判断核对了C首轮已读全文记载和当前措辞。A的“待B/C”页首流程状态可在本轮完成后同步；这不是科学缺口，不需另开审查轮。

### C04：新发现的积累/重置DTMC模型接口（需一段有界补入）

[The Power of Backdoor Absorption in Community Training](https://arxiv.org/abs/2607.06643v1)，2026-07-07。C从独立查询`"backdoor" "training dynamics" "2026" emergence`发现，继而核arXiv身份、下载公开PDF并完整分段读15页（正文1–13，参考文献13–15；本版无附录）。文件`c_sources/2607.06643.pdf/.txt`。并非由本地名单反向凑题名。

它与D1的实质接口不是泛泛“持续训练会遗忘”，而是§4.1–4.4 pp5–8明确把ASR未过阈值的注入状态设为I_0… I_(τ−1)，把正常更新对未完成注入的作用分成PPI保留积累和PRI完全重置，给出首达时间τ/p与(1−p^τ)/(q p^τ)。这是D1现有CNN梯度推导、静态风险几何之外的**现象层积累/抵消状态模型**，应防止把该模型家族误当未有。

但边界非常重要：τ、π和转移/重置规则作为建模参数或经验校准输入；没有从神经参数、Adam状态或正常样本关系推导状态规则。假设顺序community训练、攻击者不知终点、随机调度及可发现恶意更新的verification oracle。§5 pp11–12为ResNet18/CIFAR10、单BadNet配置，d=.9是模拟oracle；稀疏核验再动态降权的整体处理不能归因为正常学习本身。§4.4还允许降到阈值下后停在I_(τ−1)，**ASR低于阈值≠遗忘/完全擦除**，尽管导言有强措辞。Table1所列μ'_I和稳态数字与紧前公式存在表述/数值不一致风险；例如p=.5的PPI公式μ'_I=2，而表列20/10，因此D1不要搬表中数值或把理论边界当独立实证校准通过。正文将成功事件概率和mean ASR并图也不能当同一估计量。

**给A/B的最小要求：** D1§2或§5增一短段，将其归类为有外给状态规则的现象模型；项目表可补“已有积累/重置模型，未识别本项目自然因果链”。B核§3–5这些范围即可，无需展开社区防御性能全部表格。D2/D5由主审记录主责/边界，D1不承担全部防御综述。该项增加模型家族覆盖，不推翻已有“正常训练可促进或抑制、行为不等于内部状态”的结论。

### C05：原始EM时序（D6已全文，D1一短段接口）

[Emergent Misalignment](https://arxiv.org/abs/2502.17424v7) §4.7 pp9–11、AppC.5 pp26–28。C本轮直接回读这些原文；全文主责D6，不声称C新全读整篇。Qwen2.5-Coder32B的secure/insecure微调每10step测量，Fig11/12五seed；域内代码分化先于广泛misalignment分化，另有多选/pivotal-token连续logprob轨迹。前40步两组logprob可同向变化；不是恶意特有前兆。去weightdecay未明显改变该轨迹、多epoch后相关logprob平台，作者明确不直接等同典型grokking。

D1§4.7已正确列ModelOrganisms/Mirage，未断言自己首创训练时序，所以这**不推翻现有科学结论**；但以原始EM补一短段可闭合D6刚发现的直接时序接缝，避免误以为只有后继rank1工作研究形成过程。该支路并非trigger条件后门本身，应保留一般学习/错位行为的交界标签。B仅核新增句和对应页节，不重审整篇D6。

### 非阻断补例和其他候选去向

- **Honeypot/DBD：补例，不判D1新重要缺口。** C回读Honeypot AppC pp16–17：完整预训练PLM可先正常后毒，不能把浅层probe快学套到整模型；DBD低loss可信clean是其SSL→冻结head→半监督程序的条件结果。D1§3已有ITLM/CL/Bench反例和快学不普遍结论，增加D5交叉链接即可，不为补例开大轮。DBD具体数值仍由D5 A/B负责。
- **LPS 2307.07328 / Neurocomputing2026 134024：** 精确题名检索找到2023公开preprint与2026正式版。主要样本选择/投毒条件，送D3记录；当前只筛元数据与摘要，不声称已核“机制解释”，不以其同名不同版补充内容为D1独立证据。出版方正文403，公开旧版不自动等同2026新增内容。
- **Lilith 2607.26099：** 推理trigger family泛化，送D3边界，不把表示几何分析误记真实训练时序。
- **XAI-DTBD / training anomalies：** 本轮结果为检测器/阈值设定，尚无已核实改变D1形成论断的信息；不因标题training/dynamic扩到所有防御。
- 首轮BAGS、QES、Broadening Basin、Safety-Efficacy、Smoothing的他域路由维持；主审负责D2/D3/D4/D5显式归档，不用D1等待全部他域变体作结束标准。

### 本轮决定与剩余限制

**本轮首轮修复通过；D1最终签字暂留给C04/C05的聚焦处置后同一C复核。** 本次不是重新启动无界查漏，也不以固定篇数/轮数停止。实际补检索的新重要影响集中于C04的模型家族；C05补直接训练时序接缝，其他为补例或明确跨域路由。修订只需上述有限内容及B逐句回核；若处理后末轮没有改变结论的重要遗漏，则可在已声明日期/范围内有界结束。

必须保留四类来源限制：Information Fusion2026全文不可得；VLM OPD2611.22446附录缺失；Bench2024补充缺失；NotAllSamples clean-label全文不可得。检索索引和最新收录有滞后，本报告不声称世界零遗漏，也不据来源空白保证项目创新。


## 末轮定点复核与有界结束（2026-11-19 00:55 CST）

此次仅处理上一轮留下的C04/C05，没有重跑已关闭的C01–03，也没有把非阻断补例扩展为新要求。实际读取最新版REPORT§4.7首段、§5.7和§6相关表行、REVISIONS本轮处置、B_REVIEW§八全部13条；再次对照2607.06643§4.2的PPI/PRI原始定义、2502.17424v7§4.7关于前40步、域内先分化和grokking对照的原文，并承接上一轮对前者15页全文、后者相关正文/附录的读取。

| 未闭合项 | 原A/B处理及C复核 | 结论 |
|---|---|---|
| C04 积累/重置模型家族 | REPORT§5.7明确初始首达时间和外给/校准规则；区分PPI/PRI、首次注入与再注入；保留低于阈值仍有进度、Table1与公式不一致、事件概率与mean ASR区别。单BadNet、模拟oracle和复合干预边界均在。B C2-B01–07另核原文和原表。 | 关闭。既承认已有现象模型，又未把它升级为梯度推导、一般擦除证明或本项目自然因果链。 |
| C05 原始EM时序 | REPORT§4.7加入每10步、Fig11/12五seed、域内先分化、自由回答与受控logprob的区别，保留两组早期同向及aligned token对照、非典型grokking和非trigger条件任务边界。B C2-B08–11核正文/附录和原图；图11/12的五seed不外推一般训练说明或其他消融重复数。 | 关闭。直接时序接缝已补，不把早变称作恶意特异预警。 |
| 跨域补例及来源限制 | Honeypot/DBD仅透明链接D5，未冒充D1全文新增结果。末尾四类来源限制仍在；EM仍记跨域核节。B C2-B12–13核同步记录。 | 关闭本域处置。各域全文主责不被D1接管，不以他域所有变体完备作为D1结束条件。 |

**最终判定：PASS_WITH_SOURCE_LIMITATIONS，D1的C审计有界结束。** 理由是已记录的独立发现路径、重要全文补查、后向链、最新跨域接缝均已获得明确处置；全部要求修改的实质项经A→B→同一C闭合；本末轮没有新发现会改变D1结论的重要遗漏或实质失真。结束依据是问题与证据闭合，而非篇数、查询轮数或等待时长。上一轮已完成针对新分支的独立补检索，本轮只核修订，不伪造新的网络搜索或新增全文阅读。

这个判定仅适用于截至2026-11-19的声明范围及实际检索覆盖，**不是世界零遗漏、全部理论正确性、实验复现或项目新颖性的认证**。Information Fusion2026与NotAllSamples全文不可得、VLM OPD承诺附录缺失、Bench2024独立补充未获，四类限制持续有效；相关条目只能按已获取证据的级别使用。检索索引滞后仍可能遗漏最新材料。其他域已路由候选由主审在各域管理，不在这里被默认为已经核证。

REPORT页首及结尾仍为A交稿时“待B/C”的流程状态，现可由原A或主审按此最终判定同步；这是流程状态更新，不再要求科学内容返工。C未改A报告，未进行任何实验、GPU、论文修改或发布。
