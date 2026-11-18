# D4 A 检索记录

日期：2026-11-19（Asia/Shanghai）。范围：前置信号、训练动态检测、完成模型审计、推理早期检测和风险评估。只用于公开文献查询，没有向搜索服务提交私有实验详情。状态：A交审；B/C尚未完成。

## 实际检索入口

web搜索；arXiv摘要页/PDF；ACL Anthology正式全文；ICLR proceedings；OpenReview；Anthropic机构研究页；TAS Lab作者全文；ScienceDirect出版社摘要；ResearchGate/SSRN仅作受限候选。主审共享综述和种子为补充入口，不能作为已核结论。没有专业学术MCP。主审共享记录显示OpenAlex/Semantic Scholar429；本域未通过它们获得完整引用图。

## 公开搜索词（本轮实际执行）

1. `backdoor detection training dynamics early learning ABL Spectral Signatures activation clustering`
2. `"SpecGuard" backdoor`
3. `"ToxScreen" backdoor`
4. `"Rethinking Backdoor Detection Evaluation"`
5. `"Spectral Signatures" backdoor arxiv`
6. `"Detecting Backdoor Attacks" "Activation Clustering" arxiv`
7. `"The Trigger in the Haystack" paper`
8. `"backdoor" "training dynamics" "detection" 2026`
9. `"Testing Whether Internal Backdoor Precursors Precede Behavior"`
10. `"BaDLoss" backdoor detection`
11. `"backdoor" "early warning" training detection`
12. `"backdoor" "training-time" "loss" detection 2026`
13. `site.anthropic.com "Simple probes can catch sleeper agents"`
14. `"backdoor" "early" "training" "detection" 2025 2026 loss`
15. `"Training-Time Backdoor Sample Filtering" "11621182"`
16. `"Backdoor Sample Detection through Progressive Perturbation"`
17. `"Spectral Signatures in Backdoor Attacks" "2026" detection`
18. `"backdoor" "training-time" "precursor" detection 2026`
19. 再查 `"Testing Whether Internal Backdoor Precursors Precede Behavior"`，确认仍主要来自ResearchGate。

结果处理：经典谱/激活方法→全文；ABL/BaDLoss/BadWindtunnel→全文；Rethinking/SpecGuard/ToxScreen/Haystack/EVO/多agent Early Detection→全文；2026 TimeGuard/WSD/Z-PEFT→全文。PSSP作者页虽称loss trajectory，轴是输入逐渐扰动，先标筛查而非训练动态。DFBScanner静态末层、代码谱检测保留待核。每条稳定ID与状态见PAPERS。

## 后向与前向发展链

- Spectral Signatures/AC（2018）→SPECTRE及后续谱方法（参考文献线索）→WSD（2026）、Z-PEFT（2026）、代码谱检测v2（2026）。前两者全文揭示从数据激活转向LoRA有效权重；代码谱项仅筛查，SPECTRE未在本域全文补读，不能声称此链所有节点完整。
- ABL（2021）→BaDLoss（ICLR2025，多攻击/轨迹）→BadWindtunnel（ACL2025，受扰训练置信）/TimeGuard（ICML2026，跨任务失效）。检索HARVEY（2026上传，首页版权2025）交D5主读。保留训练loss+uncertainty（MRAI）及diffusion anchored loss（SSRN）受限线索。
- Rethinking（EMNLP2025）与ToxScreen（2026）作检测评估条件对照。这里是主题发展链，不虚称完整机器检索到的直接引用边。
- Anthropic sleeper probes（2024）→Haystack（2026）、SpecGuard（2026）、EVO-Detect（2026）：原文回看行为/触发先验和时间轴。多agent Early Detection另列工作流轴，防止混入训练预警。
- 共享综述2406.06852v5 §5、2502.05224v1 §4.2/§5回溯ONION、RAP、STRIP-ViTA、Neural Cleanse、LMSanitator等。用于识别输入检测/模型检查分类；这些被引论文未全部全文读取，因此不逐项宣传具体效果。
- 根审提供D6新线索后，直接回读Persona Vectors §4/6、From Data to Behavior §2/3/附录C、Model Organisms §4.1–4.2、Homeopathic H.2。D4只标SECTION_READ；不能把D6阅读算自己的全文。Weird Generalization和An Emergent Mirage尚待跨域修订整合。

## 下载与访问记录

PDF以arxiv.org/pdf/{id}公开下载为主，保存原PDF、pdftotext -layout文本、空白规范化read.txt。原始文本保留。TimeGuard图形字体含form-feed类控制字符，read.txt页标失准；报告使用印刷页/节/表。其他定量证据只引用可明确定位文字或表；没有凭无法辨识图形猜数。

- BaDLoss首次OpenReview `https://openreview.net/pdf?id=rK0YJwL69S` 返回403；改用ICLR官方proceedings成功，非绕过付费墙。
- Rethinking：`https://aclanthology.org/2025.emnlp-main.318.pdf`。
- BadWindtunnel：`https://aclanthology.org/2025.findings-acl.482.pdf`。
- Anthropic保存官方HTML与提取全文，网页全部章节和脚注核读。
- PSSP从TAS Lab官方作者页面找到`index.pdf`并下载；DOI10.1109/TIFS.2026.3707425。尚未全文，不报其实验结果作为证据。
- SSRN7113514直接打开403，保留受限记录。
- MRAI DOI10.1109/MRAI70020.2026.11621182：有题名/书目线索，没有一手全文。
- InfoFusion DOI10.1016/j.inffus.2026.104308：出版社摘要可见，无全文。与主审另一个104221不是同文。
- ResearchGate411771060：搜索返回长段正文，但权威身份/作者/版本未核。不等同完整已读。其片段明确无confirmatory trajectories，不能当已证预警结果。

## 排除与待核

非学术咨询博客、mlsanity、scime等只作发现入口，不作为核心实质证据。搜索返回clawrxiv synthetic phase-transition条目、医疗/入侵检测应用和一般2D anomaly detection，未取得足以改变本域判断的全文，不以标题堆入核心。没有将无关领域所有攻击方式无限纳入。

仍需B/C决定的实质支线：经典SPECTRE/Neural Cleanse/ONION/RAP原文补全是否必要；参数数据静态检测的DFBScanner/代码谱；训练loss+uncertainty与diffusion；PSSP与其他perturbation trajectory；跨域真正训练前风险预测的负结果/复现范围。此记录刻意保留这些未完成点，不声称结束条件已满足。

## B后补查（2026-11-19）

按B退回/主审指定定向补查，未无界扩展：

1. 直接打开官方ACL `https://aclanthology.org/2024.tacl-1.55/`核SEEP元数据，下载`.pdf`并全文读。精确查询`"SEEP" "Training Dynamics" backdoor 2025 2026`返回同稿arXiv2405.11575、ACL和书目索引，未当多篇；后向直接读该稿对Dataset Cartography/Spectral/AC/ABL的连接，未假称所引全部全文。
2. 直接打开`https://arxiv.org/abs/2605.15183`核v1作者/日期，下载全文。查询`"When Are Two Networks the Same" "backdoor"`发现TPM2026官方workshop PDF，下载并局部核版本变化。该版作者表新增Ward Gauderis，不将其作者列表拼到旧v1。未以博客转述为证据。
3. ResearchSquare web工具仍不能访问，但普通HTTPS官方页返回200，保存`rs-10594749-response.txt`含响应状态、HTML及citation metadata；页面直接公开PDF `https://www.researchsquare.com/article/rs-10594749/v1.pdf?c=1786015293000` 返回application/pdf200，保存全文，未绕登录/付费。旧不可得记录为当时事实，现已解除；ResearchGate条目并入正式DOI，结论仍只为protocol非确认实验。
4. D6/D1现有合法本地PDF复用，回原文核R1–R8、Weird/Mirage、OPD，读取段记录在READING_LOG及REVISIONS。没有用他域报告代替自己的已读状态。

剩余受限支线保持原状态；本次不签A/B/C结束认证。
