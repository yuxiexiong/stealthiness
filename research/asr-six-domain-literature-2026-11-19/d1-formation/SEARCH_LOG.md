# 领域1 A 检索与获取日志

日期：2026-11-19，Asia/Shanghai；检索截止并不保证索引已覆盖当天全部文献。A角色，不含B/C尚未执行的检索。只使用一般公开主题查询，没有向网络发送私有实验细节。按协议使用已加载 nature-academic-search / nature-downloader；本会话无专业学术MCP，采用Web检索+arXiv/ACL/PMLR/USENIX/期刊/作者官方源。OpenAlex公共探针429由主审报告，未把它作为唯一渠道。

## 1. 概念广搜与同义词

实际查询（引号保留关键词组合；同批结果一起筛选）：

- backdoor learning dynamics formation clean poison gradient competition
- backdoor emergence training dynamics large language models 2025 2026
- "backdoor" "two-stage" "learning dynamics"
- "backdoor" "learning dynamics" gradients formation
- "backdoor" "gradient" "conflict" training clean poisoned 2024
- "backdoor" "formation" "phase" 2025 2026
- "backdoor" "learning dynamics" -site:sciencedirect.com -site:mdpi.com -site:reddit.com
- "backdoor" "formation" "dynamics" LLM
- "backdoor" "clean" "poisoned" "competition" learning
- "backdoor" "dynamics" "formation" arxiv
- "backdoor" "lazy" "feature learning"
- "backdoor formation" training
- "backdoor" "formation" "accumulation"
- "backdoor" "emergence" "training dynamics"
- "backdoor" "grokking" formation

处理：命中ABL、Gu梯度控制、Backdoor Learning Curves、对比学习难防、固定分布最优解理论、OPD、DNA等分支。只看摘要的候选不计全文。将形成后遗忘/再激活、实时监测、防御以及一般grokking分别交叉标到D2/D4/D5/D6；保留与形成母问题有关的边界而非无限扩张攻击目录。

## 2. 具体种子身份、全文和前后链

| 查询/入口 | 实际处理与链条 |
|---|---|
| "Why Backdooring Neural Networks is so Easy"；arXiv2609.36117 | 下载v2，正文和相关附录完整阅读；区分作者设定方向、Gaussian代理和实际训练证据 |
| "backdoor" "gradient" "2023.acl-long.194"；ACL官方页/PDF | Gu2023正文和附录；追多任务梯度冲突与早期RIPPLe；进一步搜索2026优化框架 |
| "Backdoor learning curves" arxiv；Springer DOI10.1007/s13042-024-02363-5 | Springer获取超时，改作者arXiv2106.07214v4成功；D3已经全文主责，本域不重复冒充阅读 |
| "backdoor" "Anti-Backdoor Learning" arxiv | 下载2110.11571v3；正文+全部实验附录；后向[34]发现与快学相反的ITLM |
| "Learning with bad training data via iterative trimmed loss" backdoor | PMLR v97/shen19e main+supp下载；正文+相关实验附录；Fig1d渲染确认早期clean accuracy更高 |
| "BackdoorBench" arxiv 2022 | 新稿2407.19845v1下载、核身份，未正文阅读，未作核心结论；提醒2022/2024/后续期刊不能重复计证据 |
| "When Stronger Triggers Backfire" | 2605.22481，强度/测试分布理论边界，D3/D6接口候选，未全文不引其结论 |
| 最新dynamics、OPD准确题名；arXiv2611.22446 | 正文10页全部、arXiv源tar成功；6处Appendix引用，无任何附录成员，标MAIN_TEXT_READ_APPENDIX_MISSING |
| PMC13308446与DNA准确题名 | PMC被反机器人页挡住，仅163字符，不当全文；转arXiv2603.27465v2下载，正文及补充全部读完 |
| "Red Alarm for Pre-trained Models" backdoor learning；2101.06969 | 下载v5，正文全部及Appendix A–C；期刊页发现2024 correction，沿官方PDF核查是开放许可变更 |
| "Red Alarm" "Correction" "2024"；Springer10.1007/s11633-024-1507-3 | 官方1页PDF全文读，非结果修订。一次错误数字链接1498-0落到无关建筑论文，立即排除，未当来源 |
| Rando RLHF poisoning、2311.14455 | 下载v4，正文+形成结论所需A–F/I附录；未声称G/H全部有害示例逐条阅读 |
| "Phase Transitions in Backdoor Learning" | 官方Harvard课程挂载PDF，10页全部读；年份未印，PAPERS.year=null；不是已证会议论文。记录低n、相同数据与数据重抽矛盾 |
| 主审线索Agentic Backdoors via Pretraining Poisoning 2610.37741 | 官方arXiv身份、下载v1；正文p1–11和全附录p17–24读；后向引用Persistent Pre-training Poisoning、近常数投毒数量由D2/D3主责 |
| 主审D2线索Chameleon2304.12961 | 复用D2公开全文，本域重读p1–4及Fig1–2；D2承担全篇，记录CROSS_DOMAIN_SECTION_CHECKED |

下载均为公开源，用requests和pdftotext；未绕付费墙/验证码。全文集中`sources/`。PDF版本标题以实际首页为准；PAPERS有sha256。`genome2026.html`是失败页面不是全文，不可使用。源码成员表由获取脚本输出，并存了`opd2026-source.tar`和提取tex用于附录缺失核查。

## 3. 综述入口及范围复核

主审提供2406.06852、2502.05224、2026ACL reasoning survey及Science China2025入口。本域选择与方法学习范式最直接的TMLR2025 **A Survey of Recent Backdoor Attacks and Defenses in Large Language Models**，下载并完整读正文p1–17，按相关参考文献回查Gu、RIPPLe、RLHF、NeuBA及攻击范式。其无微调/提示操纵条目不自动等于参数学习形成，综述的未来问题不作为“无人研究”的证明。其他综述可由主审/其他领域复用，本域不声称已读。

综述/链条覆盖检查：

- 数据投毒监督分类：ITLM→ABL→对比学习反例。
- 语言权重/PET：RIPPLe→Gu→2026统一优化框架（全文受限）。
- 预训练表征/生成/代理：NeuBA，DNA，Agentic；Persistent Pre-training Poisoning交D2，数量规律交D3。
- 奖励模型与策略分阶段：Rando；PoisonBench/DPO/IPO等交D3，未偷换成所有偏好训练都相同。
- 学习动态与阈值：OPD、DNA，低引用课程终点扫描；固定分布最优解理论easy；Learning Curves交D3。

## 4. 前向/最新与反向支持材料搜索

后续实际查询：

- "A unified optimization framework for backdoor attacks in large language models" arxiv
- "A unified optimization framework" "Yihan" pdf
- "Gu" "gradient control" "backdoor" "2025" "formation"
- "backdoor" "pretraining" "accumulation" "2026"
- 预训练题名、Rando、OPD appendix/source等准确词搜索。

最新优化框架命中出版方ScienceDirect、DBLP与OUCI书目。作者Yihan Shi等，Information Fusion132:104221，DOI10.1016/j.inffus.2026.104221；无公开全文定位，不引摘要里的收敛和数字为已证。前向搜索是多源主题+题名/作者链，不是宣称已遍历数据库全部cited-by记录。

最新线索分流（只在正式全文核读后可提升为结论）：

| 线索 | 处理理由/归属 |
|---|---|
| 2611.10110 Homeopathic Learning | 主审交D6全文主责；clean数据增强条件行为与本域强相关，最终必须回收，当前仅待合并标识 |
| 2610.32866 TopKLoRA faithful circuit discovery | 通知主审D4；本域未读，只作线索 |
| 2611.15479 DETER | 通知主审D5/D6；正常预训交错抑制与后续学习难度，不自动等于后门首次形成 |
| 2608.00732 Mitigating Backdoors via Decoy Shortcuts and Knowledge Decoupling | 通知主审D5/D6；简单分支吸收后门，需全文查其自然机制外推限度 |
| 2512.09742 Weird Generalization & Inductive Backdoors | D6相邻机制；未全文不当结论 |
| 2604.24350 catastrophic overfitting as backdoor | D6相邻边界，不扩为全部鲁棒训练 |
| Testing Whether Internal Backdoor Precursors Precede Behavior: A Causal, Power-Matched Protocol（ResearchGate） | 仅索引见protocol，非已核实验；D4线索 |
| 10.1016/j.patcog.2026.113211 UPGP | 后处理防御/遗忘为主，D5范围 |
| 10.1016/j.neucom.2025.132133 Backdoor-vaccine | 后门间竞争防御为主，D5范围 |
| Not All Samples Are Born Equal: Towards Effective Clean-Label Backdoor Attacks | 选择样本/特征竞争，D3/D6接口，未全文不引用结果 |
| 2311.09641 RankPoison；ICML workshop Is poisoning a real threat to LLM alignment? Maybe more so than you think | 偏好毒化及条件对比，D3负责范式条件；不据Rando单篇给通用RLHF比例结论 |

## 5. 不可靠题名纠正与协作透明

D3一度给出Zhang《Backdoor Attacks on Language Models During Pre-Training》但随后撤回，疑似混合主题/NeuBA；本域未把它建成已证论文，改查真实NeuBA和真实Persistent Pre-training Poisoning。所有外部题名/日期以公开正式页或下载全文重新确认，而非信任旧项目引用。

主审指出OPD缺附录与标量模型限制，本域独立核源确认；D2给Chameleon候选，本域核相关节；D3提供Learning Curves已全文的协作状态，本域仅交叉引用。上述均不把他人摘要冒充本域阅读。

获取期间数次tool返回aborted before dispatch（连只读pwd也失败），随后重试恢复；无自动审批拒绝，也未绕审批。未运行模型实验、训练、GPU或推送。

## 6. A交稿的可审计限制

检索并非世界零遗漏保证。当前必须在B/C后再处理的真实边界：2026统一优化框架全文受限；OPD无附录；Homeopathic等跨域新近全文待回收。12篇本域正文+相关附录已读不等于12篇均高质量证据；课程报告内部不一致、理论强设定、单种子曲线均单列。A交稿供B逐项原文核验和C独立查漏，不宣称最终通过。


## B第一轮后原A定向回查（2026-11-19）

以上是原A阶段检索历史，其中“待回收/待合并”表示当时状态；本节及当前PAPERS优先表示现状。本轮没有虚构新外网搜索，沿B_REVIEW、D6 PRIORITY_EVIDENCE、D3 SUPPLEMENT_READINGS返回已保存合法原始PDF/TXT，逐项核相关正文与附录。其他域全文阅读是全域证据来源，但不算本域全文。

| 路径/候选 | 本轮实际处理与去向 |
|---|---|
| D6/sources/2611.10110 | 回读pp3–11,35–37,42–46；形成/条件性clean对照、内部投影与ridge假设进入REPORT4.4。 |
| D6/sources/2606.00995 | 回读pp3–5,8–11,18–19,27–28；教师/学生方向干预、固定尺度SGD和边界进入4.6。 |
| D6/sources/2506.11613 | 回读pp4–7,19–21,24；训练转向/放大和未来信息限制进入4.7。 |
| D6/sources/2611.00748；D3全文主责 | 回读pp3–6,14–17；保留150poison、换clean回答和选择性指标进入4.5。 |
| D6/sources/2512.09742 | 回读pp7,9,12,48–49,55–56,65,68–69；clean泄漏、真实训练曲线、潜在方向进入4.5/4.8/5.5。 |
| D6/sources/2506.11618 | 回读pp3–5,9,15,18；activation与B权重方向区分进入4.6。 |
| D6/sources/2607.09053 | 回读pp1,3–7,9–10；循环/长度负证限制通用方向机制，进入4.7。 |
| 本域sources/curves2024；D3全文主责 | 回读pp2–6,17,24–25；beta不是step，进入5.5及比较表。 |
| 本域sources/circuits2026；D4全文主责 | 回读pp1–5,7,15–16；“早形成”是前向层深，进入4.8，排除作为训练时点证明。 |

修订遵从B01–B38与X01–X05逐项表见REVISIONS。原A REPORT_A保持不变。当前索引25条：本域FULLTEXT_READ仍12条（11原始/课程+1综述）、跨域相关节核对10条、OPD缺附录1条、InfoFusion全文受限1条、BackdoorBench仅下载未读1条。跨域数量不是全文新计数，下载未读不进入实质结论。

InfoFusion未获得新全文，OPD未获得新附录，限制继续；新增来源相关证据待原B复核、全域遗漏待C独立查漏。没有以最终整链缺少直接复现认证新颖性，没有启动实验/GPU/推送。

## C第一轮查漏后的原A补证检索与处理（2026-11-19）

此段属于原A收到C意见后的定向补证，不冒充C独立检索。先完整读取C_AUDIT/C_INDEPENDENT_SEARCH与现REPORT，按三个重大遗漏及两条未闭合后向链处理；未扩展全部防御目录。

1. 复用`c_sources/li24at.pdf`、`sp24_orthogonal.pdf`、`2610.07654.pdf`合法全文，重新由原A逐段阅读正文及相关附录。原A记录见READING_NOTES C01–03；不是抄录C摘要。PAPERS新增3条FULLTEXT_READ，含实际文件哈希、页节和反证。
2. Web实际查询：`site:proceedings.mlr.press/v202/xian23a Adaptability Hypothesis backdoor`；`"Not All Samples Are Born Equal" "pdf"`；`"BackdoorBench" 2022 supplemental B.6 B.7 gradient`。分别核PMLR正式页、作者/出版方线索与NeurIPS正式main/supp。
3. 通过已授权公开网络urllib下载：`https://proceedings.mlr.press/v202/xian23a/xian23a.pdf`→`c_sources/xian23a.pdf`；`https://papers.nips.cc/paper_files/paper/2022/file/4491ea1c91aa2b22c373e5f1dfce234f-Paper-Datasets_and_Benchmarks.pdf`→`bench2022.pdf`；同前缀`-Supplemental-Datasets_and_Benchmarks.pdf`→`bench2022-supp.pdf`。pdftotext -layout转文本，原A读完正文及相关附录后才新增FULLTEXT_READ。
4. 针对受限clean-label文实际追加查询：`"Not All Samples Are Born Equal" Gao filetype:pdf`；`"109512" Gao backdoor pdf arxiv`；`site:github.com/YinghuaGao/Effective_CLB paper pdf`；`"Yinghua Gao" "Not All Samples" pdf`；`"109512" "pdf" backdoor Gao`；`"Effective_CLB" paper download`。许多结果为其他论文引用或无关编号，未作原文。打开出版方`https://www.sciencedirect.com/science/article/pii/S0031320323002121`、官方仓库`https://github.com/YinghuaGao/Effective_CLB`及仓库所链接的`https://liyiming.tech/`和`https://liyiming.tech/publications/`；仓库仅README/代码，作者公开代表作页未给本篇OA链接。结论仍为合法全文未获，PAPERS新增ABSTRACT_ONLY_FULLTEXT_UNAVAILABLE；不确证摘要机制，不绕付费墙。
5. BackdoorBench2022补充B.6/B.7已完整核读并渲染pp20–21的Fig5/6；发现高投毒BadNets仍多遗忘及多攻击total-clean cosine接近1，保留为泛化叙述的边界。沿同研究系列读此前已下载的2407.19845v1：先定位§V.D，因含重要慢学反例，再补齐全部正文pp1–16；渲染p15 Fig16/17。与2022会议文为扩展关系，不当独立复现；未混用2401.15002或2025期刊信息。
6. 打开`https://arxiv.org/abs/2407.19845v1`确认2024-07-29版本；web打开TeX Source cache miss，改用已授权公开网络获取`https://arxiv.org/src/2407.19845v1`→`c_sources/bench2024-source.tar`。只读取归档目录和指定正文/表源文件，没有执行源码。main.tex将appendix及sections-supple两个输入注释，归档也无该目录；保留的未接入超参表只能算部分补证，不能说独立supplement已读。状态从DOWNLOADED_NOT_READ改MAIN_TEXT_READ_SUPPLEMENT_PARTIAL。源表泛用CIFAR100轮而动态图展示50轮，不从泛用表推每张图具体终止设置。
7. 主审传递D6 B新界限后，原A回读Homeopathic AppC.2–4 p29和Steering §5.2/6.2 p8、AppM pp26–27。补入persona排除只属该支路，及direct-steered fullFT语义迁移不等于经典systemprompt fullFT结果。仍维持CROSS_DOMAIN_SECTION_CHECKED，不把相关节核对改成独立全篇阅读。

闭合情况：C01–03、Adaptability、BackdoorBench2022均完成原A全文/相关附录补证；2024扩展正文受限纳入；Not All Samples合法全文仍受限。其他C候选跨域接口按主审分工，不作未经全文的机制结论。报告/索引/阅读记录/修订表已同步，待原B核新增再交原C复核。原始REPORT_A及B/C审稿文件未修改；另存REPORT_B_BEFORE_C留住本轮修改前稿。

## C第二轮聚焦接口补证（2026-11-19）

- 按主审指令读C_AUDIT第二轮，不重跑已关闭的C01–03。C04来源由C独立检索发现；本轮原A没有另作广搜，仅复用C保存的公开`c_sources/2607.06643.pdf/.txt`，核首页v1/2026-07-07，逐段读全部15页，并渲染pp6/8及10核公式/表格。全文本版无附录，新增FULLTEXT_READ。
- C05复用D6保存的公开`d6-generalization/sources/2502.17424v7.pdf/.txt`，核v7/2026-01-20；回读§4.7 pp9–11、AppC.5 pp26–28，渲染p10核Fig11/12。跨域全文主责D6；原A只记CROSS_DOMAIN_SECTION_CHECKED。
- Honeypot/DBD读取D5 `REVISION_READING_NOTES.md`相应已读范围，报告只作透明跨域链接，不新增本域全文、原篇性能/机制结论。
- REPORT增加§5.7一段、§4.7短段及两条表内接口，PAPERS/READING_NOTES/REVISIONS同步；本轮前稿留`REPORT_C1.md`。没有外部文献下载、额外实验或发布。待原B两项核验、原C定点复核。
