# D6 原 A 对 B 的修订处置

2026-11-19。状态：本轮 A 完成，**待原 B 回原文复核新增/修改，随后 C 独立查漏**。REPORT_A.md 未动；B_REVIEW.md 未动。无实验/GPU/论文正文/发布。

| B 要求 | 原 A 处置与证据 | 落盘位置 |
|---|---|---|
| B1 Inherited 家族概括 | 采纳；回读§5.2 pp5–6、AppJ/K p15。Llama/Gemma teacher贡献小只限appended；random S增益+.160/+.439。Qwen依赖限appended，random无成功转移。 | REPORT D6-03b |
| B2 Model Organisms 在线信息 | 采纳；回读AppF p19、G.1 p22。单adapter localcos用t−s/t/t+s，至少延迟；PCA用全部checkpoint。保留真实5×早adapter干预，不用信息条件否定干预。 | REPORT D6-04限制 |
| B3 Homeopathic persona筛run | 采纳；回读C.3–4 pp29–31。无名prompt persona≥10%排除；FigC.1 Qwen4/Gemma0；TableC.2部分5/6；不推广到German。220judge-positive中55.5%人工连贯分母保留。 | REPORT D6-01e；PRIORITY |
| B4 同文fullFT语义阳性 | 采纳；回读§6.2p8、AppMpp26–27/Fig18。经典systemprompt动物阴性，direct-steered-data fullFT语义阳性，3epochs/LR1e−5；与随机向量EAS区别。 | REPORT首表/D6-02c；PRIORITY |
| B5 MNIST泛化不能抹去 | 采纳；回读AppEpp35–36。源0–4条件化与未投毒5/6/8/9扩张分开；保留9.8→60.5、其他clean6.2、20runs、2000正确非源样本及固定1500steps。 | REPORT D6-01b；PRIORITY |
| 原始EM必补 | 完成正文全读及相关附录。补有条件广坏、形成轨迹、格式/多样性控制与非grokking反例；记录版本/阈值口径差。 | D6-16；CORE_FULLTEXT_NOTES |
| Persona Features必补 | 完成正文全读及相关附录，含v2附J。补directCE干预、known-feature选择、数据比例而非时间的早期信号、重采分母及再对齐残留。 | D6-17；CORE_FULLTEXT_NOTES |
| 原始Cloud正式研究必补 | 完成Nature正文/Methods/全部ExtendedData，图像核表。正式题名为hidden signals in data，B初筛题名更正；arxiv2507.14805合并同研究。补不同初始化BC可恢复、SGD阳性、小更新定理边界。 | D6-18；CORE_FULLTEXT_NOTES |
| Beyond Owls必补 | 完成正文+A–E。随机新能力、条件后门、棋局rewardhack；补soft/hard、主例选择、单vector OOD、参数化阴性等。 | D6-19；CORE_FULLTEXT_NOTES |
| Towards Understanding必补 | 完成正文+A–T相关方法结果。divergence mask/早层训练干预、换prompt/混clean反例、直觉而非定理、EM过滤条件。 | D6-20；CORE_FULLTEXT_NOTES |
| LLS必补 | 完成正文+A–C。选择现成偏好对+DPO而非普通SFT；理论假设、selectedpairs logodds、跨模型及取峰值/问题边界。 | D6-21；CORE_FULLTEXT_NOTES |
| Phantom | 按主审分工D5主读全文；D6回§3–4、4.3与H核crossfamily和向量构造失败界限。未重复计D6全文。 | REPORT§3；PAPERS CROSS_DOMAIN_SECTIONS_READ |
| TensorSimilarity | D4主读全文；D6回§3.3/附B核单层bilinear、SVHN、pre/post区块与无leadtime。 | 同上 |
| D1 CNN/S&P理论 | D1主读全文；D6回正文条件/更新引理及SP相关附录。不能说后门无动态理论，也不能把NTK/正交假设当LM事实。 | 同上 |
| 2405.12755一般grokking | 按B允许范围限缩为外围背景，不以未读证明空白；若C发现改变核心主张再补。 | PAPERS范围排除；REPORT§3 |

## 本轮交回 B 的核验焦点

新增6篇共D6-16–21所有实质判断；上述B1–B5；跨域4条定点接缝（不是重新全读已审15篇）。所有核心原文已取得且读完，不列为访问受限。未声称OpenAlex/SemanticScholar全覆盖，亦未称六领域总体查重完成。

## B §7 定点修订（2026-11-19）

仅落实R1–R4与两术语精确化，未新增检索或重读旧文；待原B核改句后交C。

| 项目 | 已改句及原文定位 |
|---|---|
| R1 | REPORT16a与CORE16：Fig17 p22的10%/30%是预登记/主集结果率，与§3.3 p5/Fig5的6%/20%不一致，采用正文；不再称judge阈值冲突。评分边界符号差在笔记另列。 |
| R2 | REPORT16b与CORE16：§4.7 pp10–11/Fig13去weight-decay及10epochs阴性限multiple-choice misaligned-choice log-probability，不推所有采样行为；aligned pivotal-token Fig29定位改为附C.5 p28。 |
| R3 | REPORT19c与CORE19：附D p19 rank8 all-module≤1%同时改变teacher/student；随后同数据提高student rank32不救；保留all-module rank1约3.5%反例。 |
| R4 | CORE17：Fig25 p26十特征两两cosine改为约0–.34，含.00/−.00舍入值。 |
| 两术语 | CORE19棋局上限100 steps；REPORT21b/CORE21明确ψ优化空间可实现任意v∈R^d（脚注允许足够大球）。 |
| 索引 | PAPERS对应四篇记录上述修订与待改句复核状态；CNN/S&P的read_sections分别列本篇页节，不再共用混合说明。 |

REPORT_A与B_REVIEW保持不变；没有将本次A修改自行标为B通过。

## C 首轮三项缺口：原 A 全文补证（2026-11-19）

按主审范围，仅补2608.07436v2、2604.25891v1、2608.11025v2。原 A 已实读各篇全部正文及与所纳入主张相关附录，查看三张原PDF图表缓存；不以 C 结论代替阅读。不扩外围，不重跑旧21篇，无实验/子agent。

| C项 | 处置及新增位置 | 保留的关键限制 |
|---|---|---|
| C1 读出损伤局部识别 | D6-22；C_GAP_FULLTEXT_NOTES完整页节记录；首表补相邻交换与均值项干预 | 选定五条无norm小算术轨迹；decoder与swap部分checkpoint不同；局部充分性不分解上游；RMS accurateCE四seed embedding失败；Fourier答案轨道过滤可让lookup100%，不推翻Nanda全部证据。 |
| C2 缓解变条件性广坏 | D6-23；首表normal/harmful交互、D6-16a/17b加接缝 | Fig4 aggregate20→50%下降，文字严格单调不采；Qwen CoT正文/附F9冲突保留；DeepSeek3.1 vs6.3才为配对；self-generated SFT非动态RL；persona排seed、格式trigger及两指标分开；过滤后分母；缓解仍有益。 |
| C3 代理归因与文本诱发失配 | D6-24；首表方向控制限制、D6-07/08/17接缝 | 21/23 feature正移不等连贯EM；MR/IR同时报告；synthetic与reformat不是完全单因素；Dolma为代理，无pre/midtraining因果来源干预；单seed/8题聚类使OR非总体显著性。 |

新增三篇PAPERS标FULLTEXT_READ但审核状态为C_GAP_A_ADDED_PENDING_B_C_RECHECK。现24本域全文、4跨域定点、2范围/委派，共30唯一索引条目。REPORT_A、B_REVIEW、C_AUDIT及C独立检索记录未动。交原B只核D6-22–24及首表/两处接缝实际改动，再交原C定点复核；本次不自签最终完成。


主审状态同步（2026-11-19）：原B §9与同C §6已闭合；上文待办/交审状态保留为历史，不改变实际阅读范围。
