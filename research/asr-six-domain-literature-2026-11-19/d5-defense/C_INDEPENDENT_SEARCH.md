# C 独立检索冻结记录

冻结日期：2026-11-19。冻结前仅阅读 PROTOCOL 的协议及 D5 范围，未读项目材料、A/B、PAPERS、共享名单或 sources。工具：公开 web，arXiv/ACL/NeurIPS/PMLR/OpenReview 官方入口；专业学术 MCP 不可用。采用 nature-academic-search 多源工作流及 nature-downloader 公开全文流程。本文件是候选发现记录，不以摘要冒充全文证据；全文状态与比较结论另见 C_AUDIT。

## 实际查询与覆盖路径

1. `large language model backdoor defense unlearning training prevention survey 2025 2026`；`backdoor defense relearning robustness safety fine tuning backdoor removal 2025`：训练干预/删除/再学习边界。
2. `site.arxiv.org backdoor defense LLM 2026 removal training`；`site.openreview.net backdoor defense relearning removal 2025 language model`：最新和未成熟/低曝光预印本（未验证引用计数，不称已证实低引用）。
3. `backdoor defense decoupling training ABL DBD gradient ascent anti backdoor learning`；`backdoor defense decoupling`：形成期与机制防御。
4. `backdoor survey defenses large language models`；`backdoor training defense 2026`（arXiv/OpenReview）；`Preventing backdoor LLMs training defense`（arXiv/ACL）：综述及会议最新入口。
5. `backdoor relearning defense`；`backdoor defense relearning attack removal unlearning resurrection`；`backdoor relearning unlearning LLM defense`：低 ASR 与清除、再激活及再学习耐受分离。
6. 精确标题/术语链：Backdoor Containment via Expert；Backdoor Unlearning Generalization；Breaking the False Sense of Security；GradSentry；Removing the NEEDLE。打开 QES 与 Dummy Backdoor 官方 HTML 的参考文献区，追溯/发展链包括 CROW、OSFT、ABL、W2SDefense、RepGuard、NEEDLE、GraCeFul/GradSentry。具体引用关系须后续全文核实，关键词共现不等于正式引用。

## 冻结候选与预期去向（全部仅初筛）

|候选|稳定入口|待核问题|
|---|---|---|
|Anti-Backdoor Learning (2021)|https://proceedings.neurips.cc/paper_files/paper/2021/hash/7d38b1e9bd793d3f45e0e212a729a93c-Abstract.html|早学/隔离/梯度上升，视觉边界|
|Backdoor Defense via Decoupling the Training Process (2022)|https://arxiv.org/abs/2202.03423|形成期去耦，视觉迁移边界|
|Backdoor Defense via Deconfounded Representation Learning (2023)|https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Backdoor_Defense_via_Deconfounded_Representation_Learning_CVPR_2023_paper.pdf|同分支机制补例|
|CROW (2025)|https://proceedings.mlr.press/v267/min25b.html|生成式LLM隐层一致性修复|
|Simulate and Eliminate (2025)|https://ojs.aaai.org/index.php/AAAI/article/view/32018|已知/未知触发OSFT|
|RepGuard (2025)|https://papers.nips.cc/paper_files/paper/2025/hash/3f80233900d303acb23b2b807efbddcf-Abstract-Conference.html|表征去耦与正常能力|
|W2SDefense (2025); journal extension (2026)|https://aclanthology.org/2025.findings-acl.255/ ; https://doi.org/10.1016/j.patcog.2025.113030|蒸馏/PEFT，扩展不算独立重复证据|
|P2P (2026)|https://aclanthology.org/2026.findings-acl.600/|毒攻毒与防御可靠性|
|Activation Decomposition and Steering (2026)|https://aclanthology.org/2026.acl-long.2025.pdf|在线抑制还是权重删除|
|Backdoor Unlearning Generalization (2026)|https://arxiv.org/abs/2606.03785|跨后门清除，未知触发泛化|
|Dummy Backdoor as a Defense (2026)|https://arxiv.org/abs/2606.11648|可控后门诱导未知后门消除，共享机制与失败边界|
|QES (2026)|https://arxiv.org/abs/2610.00663|允许形成后隔离关闭，非清除，训练干预新分支|
|NEEDLE (2026)|https://arxiv.org/abs/2610.00348|正交化，保拒绝/安全子空间|
|GradSentry (2026)|https://arxiv.org/abs/2605.26574|训练前过滤+防形成，跨毒比例与谱信号边界|
|Breaking the False Sense of Security (2024)|https://arxiv.org/abs/2405.16134|已防御模型推理时再激活，不能等同重新训练|
|ULRL (2024)|https://arxiv.org/abs/2405.14781|unlearn/relearn作为防御步骤不等于抵抗攻击者relearning|
|Expose Before You Defend (2024)|https://arxiv.org/abs/2410.19427|遗忘正常特征暴露后门，不代表清除本身|
|Improving LLM Unlearning Robustness via Random Perturbations (2026)|https://openreview.net/pdf?id=QYw192hTdH|相邻知识遗忘引起触发式失效，D6接缝|
|Backdoor Mitigation in Decentralized LLM Fine-Tuning (2026)|https://arxiv.org/abs/2609.37367|分布式聚合条件，非自动核心|
|Loss Masking Under the Hood (2026)|https://aclanthology.org/2026.privatenlp-main.5/|隐匿与删除边界，D2接缝|
|Survey recent backdoor attacks/defenses (2024)|https://arxiv.org/abs/2406.06852|谱系入口|
|Survey threats (2025)|https://arxiv.org/abs/2502.05224|分类/补漏入口|
|Reasoning backdoor survey (2025)|https://arxiv.org/abs/2510.07697|推理子域索引，不扩攻击目录|

## 边界

新闻、非授权镜像、自动生成纸评不作实质证据。当前发现不足以判断 A/B 是否漏文。冻结后开始比较；已覆盖仅核关键边界，同类补例记录不展开，改变范围/判断的重要遗漏才全文补证并交回 A/B。没有固定篇数配额，也不承诺绝对零遗漏。
