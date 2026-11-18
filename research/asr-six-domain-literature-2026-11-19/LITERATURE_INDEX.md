# 合并文献索引

按稳定ID归一化合并；版本和各域阅读责任保留于 [JSON索引](LITERATURE_INDEX.json)。只合并已确认身份，不按题名相似度混合不同论文。BackdoorBench 2022原篇与2024扩展分列。

各域索引共 203 条归属记录，合并为 154 个作品ID；其中 130 个至少有一个领域标记为正文及相关附录已读。此计数包含综述、研究方案、课程报告等，不能全称为已发表实证论文；也不是阅读次数或检索完整性的指标。仅初筛的部分候选保留在检索/C日志，不强行全量入库。

各域PAPERS的 `domain_audit_status` 是当前领域验收状态；旧 `audit_status` / `review_state` 等字段保留历史交审语义，以B/C末轮为准。

“全文”表示正文及支撑结论的相关附录已读，不表示逐行复证所有定理、阅读所有示例目录或复现实验。全文受限、相关节阅读和摘要初筛分别保留。

| 作品ID / 来源 | 标题 | 各域阅读责任 |
|---|---|---|
| [10.1016/j.inffus.2026.104221](https://doi.org/10.1016/j.inffus.2026.104221) | A unified optimization framework for backdoor attacks in large language models | [D1](d1-formation/REPORT.md)：ABSTRACT_ONLY_FULLTEXT_UNAVAILABLE |
| [10.1016/j.inffus.2026.104308](https://doi.org/10.1016/j.inffus.2026.104308) | Defending against high-poisoning backdoor attacks by exploiting loss divergence | [D4](d4-signals/REPORT.md)：ABSTRACT_ONLY |
| [10.1016/j.patcog.2023.109512](https://doi.org/10.1016/j.patcog.2023.109512) | Not All Samples Are Born Equal: Towards Effective Clean-Label Backdoor Attacks | [D1](d1-formation/REPORT.md)：ABSTRACT_ONLY_FULLTEXT_UNAVAILABLE |
| [10.1038/s41586-026-10319-8](https://www.nature.com/articles/s41586-026-10319-8) | Language models transmit behavioural traits through hidden signals in data | [D6](d6-generalization/REPORT.md)：全文 |
| [10.1109/mrai70020.2026.11621182](https://doi.org/10.1109/MRAI70020.2026.11621182) | Training-Time Backdoor Sample Filtering via Loss and Uncertainty Dynamics | [D4](d4-signals/REPORT.md)：METADATA_ONLY |
| [10.1109/sp54263.2024.00225](https://doi.org/10.1109/SP54263.2024.00225) | Exploring the Orthogonality and Linearity of Backdoor Attacks | [D1](d1-formation/REPORT.md)：全文; [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED; [D5](d5-defense/REPORT.md)：全文; [D6](d6-generalization/REPORT.md)：CROSS_DOMAIN_SECTIONS_READ |
| [10.1109/tifs.2026.3707425](https://doi.org/10.1109/TIFS.2026.3707425) | Backdoor Sample Detection through Progressive Perturbation and Loss Trajectory Analysis | [D4](d4-signals/REPORT.md)：SCREENED |
| [10.1162/tacl_a_00684](https://aclanthology.org/2024.tacl-1.55/) | SEEP: Training Dynamics Grounds Latent Representation Search for Mitigating Backdoor Poisoning Attacks | [D4](d4-signals/REPORT.md)：全文 |
| [10.1609/aaai.v36i9.21272](https://ojs.aaai.org/index.php/AAAI/article/view/21272) | Hibernated Backdoor: A Mutual Information Empowered Backdoor Attack to Deep Neural Networks | [D2](d2-persistence/REPORT.md)：MAIN_TEXT_READ_APPENDIX_MISSING |
| [10.18653/v1/2020.acl-main.249](https://aclanthology.org/2020.acl-main.249/) | Weight Poisoning Attacks on Pre-trained Models | [D1](d1-formation/REPORT.md)：全文; [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED |
| [10.18653/v1/2023.acl-long.194](https://aclanthology.org/2023.acl-long.194/) | A Gradient Control Method for Backdoor Attacks on Parameter-Efficient Tuning | [D1](d1-formation/REPORT.md)：全文 |
| [10.18653/v1/2023.emnlp-main.60](https://aclanthology.org/2023.emnlp-main.60/) | Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation | [D4](d4-signals/REPORT.md)：全文 |
| [10.18653/v1/2024.findings-emnlp.335](https://aclanthology.org/2024.findings-emnlp.335/) | PKAD: Pretrained Knowledge is All You Need to Detect and Mitigate Textual Backdoor Attacks | [D4](d4-signals/REPORT.md)：全文 |
| [10.21203/rs.3.rs-10594749/v1](https://doi.org/10.21203/rs.3.rs-10594749/v1) | Testing Whether Internal Backdoor Precursors Precede Behavior: A Causal, Power-Matched Protocol | [D4](d4-signals/REPORT.md)：全文 |
| [10.3390/jcp6050146](https://www.mdpi.com/2624-800X/6/5/146) | Associative vs. Distributional: Two Regimes of Backdoor Learning in LoRA-Adapted Code-Generation Models | [D3](d3-conditions/REPORT.md)：全文 |
| [2023.pmlr.wan23b](https://proceedings.mlr.press/v202/wan23b.html) | Poisoning Language Models During Instruction Tuning | [D3](d3-conditions/REPORT.md)：全文 |
| [2024.naacl-long.171](https://aclanthology.org/2024.naacl-long.171/) | Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models | [D3](d3-conditions/REPORT.md)：全文 |
| [2024.naacl-long.27](https://aclanthology.org/2024.naacl-long.27/) | From Shortcuts to Triggers: Backdoor Defense with Denoised PoE | [D5](d5-defense/REPORT.md)：全文 |
| [2025.emnlp-main.318](https://aclanthology.org/2025.emnlp-main.318/) | Rethinking Backdoor Detection Evaluation for Language Models | [D4](d4-signals/REPORT.md)：全文 |
| [2025.findings-acl.482](https://aclanthology.org/2025.findings-acl.482/) | BadWindtunnel: Defending Backdoor in High-noise Simulated Training with Confidence Variance | [D4](d4-signals/REPORT.md)：全文 |
| [2026.acl-long.1113](https://aclanthology.org/2026.acl-long.1113/) | Merging Triggers, Breaking Backdoors: Defensive Poisoning for Instruction-Tuned Language Models | [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED; [D5](d5-defense/REPORT.md)：全文 |
| [2026.acl-long.495](https://aclanthology.org/2026.acl-long.495/) | Critical-CoT: A Robust Defense Framework against Reasoning-Level Backdoor Attacks in Large Language Models | [D5](d5-defense/REPORT.md)：全文 |
| [2026.acl-long.920](https://aclanthology.org/2026.acl-long.920/) | Backdoor Collapse: Eliminating Unknown Threats via Known Backdoor Aggregation in Language Models | [D5](d5-defense/REPORT.md)：全文 |
| [2026.findings-aacl.248](https://aclanthology.org/2026.findings-aacl.248/) | Cleansing Fine-Tuning Data to Mitigate Backdoors in LLMs via the Base Model’s Inherent Generative Abilities | [D5](d5-defense/REPORT.md)：全文 |
| [2026.findings-acl.600](https://aclanthology.org/2026.findings-acl.600/) | P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs | [D5](d5-defense/REPORT.md)：全文 |
| [2026.findings-acl.815](https://aclanthology.org/2026.findings-acl.815/) | DiSec: Mitigating Backdoors in Pre-trained Language Models via Disentanglement of Adversarial Weights for Secure Fine-Tuning | [D5](d5-defense/REPORT.md)：全文 |
| [2026.findings-emnlp.2213](https://aclanthology.org/2026.findings-emnlp.2213/) | Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training | [D2](d2-persistence/REPORT.md)：全文 |
| [anthropic-2026-backdooring-classifiers](https://alignment.anthropic.com/2026/backdooring-classifiers/) | Poisoning Fine-tuning Datasets of Constitutional Classifiers | [D3](d3-conditions/REPORT.md)：全文 |
| [anthropic:macdiarmid2024sleeperagentprobes](https://www.anthropic.com/research/probes-catch-sleeper-agents) | Simple probes can catch sleeper agents | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:1811.00636](https://arxiv.org/abs/1811.00636) | Spectral Signatures in Backdoor Attacks | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:1811.03728](https://arxiv.org/abs/1811.03728) | Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:1905.10447](https://arxiv.org/abs/1905.10447) | Regula Sub-rosa: Latent Backdoor Attacks on Deep Neural Networks | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2004.07780](https://arxiv.org/abs/2004.07780v5) | Shortcut Learning in Deep Neural Networks | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2004.11514](https://openaccess.thecvf.com/content_CVPRW_2020/html/w47/Truong_Systematic_Evaluation_of_Backdoor_Data_Poisoning_Attacks_on_Image_Classifiers_CVPRW_2020_paper.html) | Systematic Evaluation of Backdoor Data Poisoning Attacks on Image Classifiers | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2006.12557](https://proceedings.mlr.press/v139/schwarzschild21a.html) | Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2011.09468](https://arxiv.org/abs/2011.09468v4) | Gradient Starvation: A Learning Proclivity in Neural Networks | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2101.05930](https://arxiv.org/abs/2101.05930) | Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks | [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED; [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2101.06969](https://arxiv.org/abs/2101.06969) | Red Alarm for Pre-trained Models: Universal Vulnerability to Neuron-Level Backdoor Attacks | [D1](d1-formation/REPORT.md)：全文 |
| [arxiv:2104.09667](https://proceedings.neurips.cc/paper/2021/hash/959ab9a0695c467e7caf75431a872e5c-Abstract.html) | Manipulating SGD with Data Ordering Attacks | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2106.07214](https://arxiv.org/abs/2106.07214) | Backdoor Learning Curves: Explaining Backdoor Poisoning Beyond Influence Functions | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED |
| [arxiv:2106.09667](https://arxiv.org/abs/2106.09667) | Poisoning and Backdooring Contrastive Learning | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2110.03735](https://arxiv.org/abs/2110.03735) | Adversarial Unlearning of Backdoors via Implicit Hypergradient | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2110.11571](https://arxiv.org/abs/2110.11571) | Anti-Backdoor Learning: Training Clean Models on Poisoned Data | [D1](d1-formation/REPORT.md)：全文; [D4](d4-signals/REPORT.md)：全文; [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2110.14430](https://arxiv.org/abs/2110.14430) | Adversarial Neuron Pruning Purifies Backdoored Deep Models | [D5](d5-defense/REPORT.md)：FULLTEXT_PARTIAL |
| [arxiv:2202.03423](https://arxiv.org/abs/2202.03423) | Backdoor Defense via Decoupling the Training Process | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2205.13523](https://arxiv.org/abs/2205.13523) | PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations | [D2](d2-persistence/REPORT.md)：TRIAGED_SCOPE_EXCLUDED_NOT_FULLTEXT_READ |
| [arxiv:2206.04677](https://arxiv.org/abs/2206.04677) | On the Permanence of Backdoors in Evolving Models | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2206.10341](https://proceedings.mlr.press/v162/zhang22w.html) | Neurotoxin: Durable Backdoors in Federated Learning | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2210.09305](https://arxiv.org/abs/2210.09305) | Thinking Two Moves Ahead: Anticipating Other Users Improves Backdoor Attacks in Federated Learning | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2210.09545](https://arxiv.org/abs/2210.09545) | Fine-mixing: Mitigating Backdoors in Fine-tuned Language Models | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2301.05217](https://arxiv.org/abs/2301.05217v3) | Progress measures for grokking via mechanistic interpretability | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2303.06818](https://arxiv.org/abs/2303.06818) | Backdoor Defense via Deconfounded Representation Learning | [D5](d5-defense/REPORT.md)：ABSTRACT_SCREENED |
| [arxiv:2304.12961](https://arxiv.org/abs/2304.12961) | Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED; [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2304.15004](https://arxiv.org/abs/2304.15004v2) | Are Emergent Abilities of Large Language Models a Mirage? | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2310.18633](https://arxiv.org/abs/2310.18633) | Setting the Trap: Capturing and Defeating Backdoors in Pretrained Language Models through Honeypots | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2311.14455](https://arxiv.org/abs/2311.14455) | Universal Jailbreak Backdoors from Poisoned Human Feedback | [D1](d1-formation/REPORT.md)：全文 |
| [arxiv:2312.00027](https://aclanthology.org/2024.naacl-long.276/) | Stealthy and Persistent Unalignment on Large Language Models via Backdoor Injections | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2312.09057](https://www.usenix.org/conference/usenixsecurity24/presentation/li-changjiang) | On the Difficulty of Defending Contrastive Learning against Backdoor Attacks | [D1](d1-formation/REPORT.md)：全文 |
| [arxiv:2401.05566](https://arxiv.org/abs/2401.05566) | Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2402.12026](https://arxiv.org/abs/2402.12026) | Acquiring Clean Language Models from Backdoor Poisoned Datasets by Downscaling Frequency Space | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2405.12755](https://arxiv.org/abs/2405.12755) | Progress Measures for Grokking on Real-world Tasks | [D6](d6-generalization/REPORT.md)：OUT_OF_CORE_SCOPE_METADATA_ONLY |
| [arxiv:2405.16134](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d06537b4b38ccf008a54559d2c56fa23-Abstract-Conference.html) | Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2406.06852](https://arxiv.org/abs/2406.06852) | A Survey of Recent Backdoor Attacks and Defenses in Large Language Models | [D1](d1-formation/REPORT.md)：全文; [D2](d2-persistence/REPORT.md)：PARTIAL_DISCOVERY_ONLY; [D3](d3-conditions/REPORT.md)：全文; [D4](d4-signals/REPORT.md)：SECTION_READ |
| [arxiv:2406.07778](https://arxiv.org/abs/2406.07778) | A Study of Backdoors in Instruction Fine-tuned Language Models | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2406.17092](https://arxiv.org/abs/2406.17092) | BEEAR: Embedding-based Adversarial Removal of Safety Backdoors in Instruction-tuned Language Models | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2407.14738](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11789.pdf) | Flatness-aware Sequential Learning Generates Resilient Backdoors | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2407.19845](https://arxiv.org/abs/2407.19845) | BackdoorBench: A Comprehensive Benchmark and Analysis of Backdoor Learning | [D1](d1-formation/REPORT.md)：MAIN_TEXT_READ_SUPPLEMENT_PARTIAL |
| [arxiv:2408.02946](https://arxiv.org/abs/2408.02946) | Scaling Trends for Data Poisoning in LLMs | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2408.13221](https://proceedings.iclr.cc/paper_files/paper/2025/file/f655706547885b8e32ef46f1c067ece2-Paper-Conference.pdf) | Protecting against simultaneous data poisoning attacks | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2409.14805](https://arxiv.org/abs/2409.14805) | SDBA: A Stealthy and Long-Lasting Durable Backdoor Attack in Federated Learning | [D2](d2-persistence/REPORT.md)：TRIAGED_SCOPE_EXCLUDED_NOT_FULLTEXT_READ |
| [arxiv:2410.08811](https://proceedings.mlr.press/v267/fu25e.html) | PoisonBench: Assessing Language Model Vulnerability to Poisoned Preference Data | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2410.09838](https://arxiv.org/abs/2410.09838) | Uncovering, Explaining, and Mitigating the Superficial Safety of Backdoor Defense | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2411.12768](https://proceedings.mlr.press/v267/min25b.html) | CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2411.14449](https://arxiv.org/abs/2411.14449) | Unlearn to Relearn Backdoors: Deferred Backdoor Functionality Attacks on Deep Learning Models | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2501.02704](https://arxiv.org/abs/2501.02704v3) | Persistence of Backdoor-based Watermarks for Neural Networks: A Comprehensive Evaluation | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2502.05224](https://arxiv.org/abs/2502.05224) | A Survey on Backdoor Threats in Large Language Models (LLMs): Attacks, Defenses, and Evaluations | [D2](d2-persistence/REPORT.md)：PARTIAL_DISCOVERY_ONLY; [D4](d4-signals/REPORT.md)：SECTION_READ; [D5](d5-defense/REPORT.md)：FULLTEXT_PARTIAL |
| [arxiv:2502.17424](https://arxiv.org/abs/2502.17424v7) | Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2503.17578](https://arxiv.org/abs/2503.17578) | Large Language Models Can Verbatim Reproduce Long Malicious Sequences | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2505.12871](https://proceedings.mlr.press/v267/liang25f.html) | Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks? | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2505.16567](https://arxiv.org/abs/2505.16567) | Watch Your Steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2506.11613](https://arxiv.org/abs/2506.11613) | Model Organisms for Emergent Misalignment | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED; [D4](d4-signals/REPORT.md)：SECTION_READ; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2506.11618](https://arxiv.org/abs/2506.11618) | Convergent Linear Representations of Emergent Misalignment | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2506.19823](https://arxiv.org/abs/2506.19823v2) | Persona Features Control Emergent Misalignment | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2507.16795](https://arxiv.org/abs/2507.16795) | Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2507.21509](https://arxiv.org/abs/2507.21509) | Persona Vectors: Monitoring and Controlling Character Traits in Language Models | [D4](d4-signals/REPORT.md)：SECTION_READ; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2509.05318](https://arxiv.org/abs/2509.05318) | Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2509.14223](https://proceedings.iclr.cc/paper_files/paper/2026/hash/657a088c0d3a985a2ef017240c2ad40d-Abstract-Conference.html) | Fresh in memory: Training-order recency is linearly encoded in language model activations | [D3](d3-conditions/REPORT.md)：SCREENED_OUT_DIRECT_CONDITIONS |
| [arxiv:2509.23886](https://arxiv.org/abs/2509.23886v2) | Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2510.05159](https://arxiv.org/abs/2510.05159) | Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain | [D2](d2-persistence/REPORT.md)：TRIAGED_SCOPE_EXCLUDED_NOT_FULLTEXT_READ |
| [arxiv:2510.07192](https://arxiv.org/abs/2510.07192) | Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples | [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED; [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2510.13992](https://arxiv.org/abs/2510.13992) | Optimizing Spectral Signature in Code Backdoor Detection | [D4](d4-signals/REPORT.md)：SCREENED |
| [arxiv:2512.09742](https://arxiv.org/abs/2512.09742) | Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED; [D4](d4-signals/REPORT.md)：SECTION_READ; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2512.14741](https://arxiv.org/abs/2512.14741) | Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2602.00822](https://arxiv.org/abs/2602.00822) | Safety-Efficacy Trade Off: Robustness against Data-Poisoning | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2602.03085](https://arxiv.org/abs/2602.03085) | The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2602.04735](https://arxiv.org/abs/2602.04735) | From Data to Behavior: Predicting Unintended Model Behaviors Before Training | [D4](d4-signals/REPORT.md)：SECTION_READ; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2602.04863](https://arxiv.org/abs/2602.04863v1) | Subliminal Effects in Your Data: A General Mechanism via Log-Linearity | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2602.04899](https://arxiv.org/abs/2602.04899) | Phantom Transfer: Data Poisoning can Survive Data-Level Defences | [D5](d5-defense/REPORT.md)：全文; [D6](d6-generalization/REPORT.md)：CROSS_DOMAIN_SECTIONS_READ |
| [arxiv:2602.07852](https://arxiv.org/abs/2602.07852v1) | Emergent Misalignment is Easy, Narrow Misalignment is Hard | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2602.09182](https://arxiv.org/abs/2602.09182) | One RNG to Rule Them All: How Randomness Becomes an Attack Vector in Machine Learning | [D3](d3-conditions/REPORT.md)：SCREENED_OUT_DIRECT_CONDITIONS |
| [arxiv:2602.10382](https://arxiv.org/abs/2602.10382) | Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED |
| [arxiv:2602.15195](https://arxiv.org/abs/2602.15195) | Detecting Backdoored LoRAs from Weights Alone | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2603.27465](https://arxiv.org/abs/2603.27465) | Poisoning the Genome: Targeted Backdoor Attacks on DNA Foundation Models | [D1](d1-formation/REPORT.md)：全文 |
| [arxiv:2604.25891](https://arxiv.org/abs/2604.25891v1) | Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2605.06152](https://arxiv.org/abs/2605.06152v5) | Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2605.15183](https://arxiv.org/abs/2605.15183) | When Are Two Networks the Same? Tensor Similarity for Mechanistic Interpretability | [D4](d4-signals/REPORT.md)：全文; [D6](d6-generalization/REPORT.md)：CROSS_DOMAIN_SECTIONS_READ |
| [arxiv:2605.18907](https://arxiv.org/abs/2605.18907) | Lightweight and Fast Backdoor Model Detection | [D4](d4-signals/REPORT.md)：SCREENED |
| [arxiv:2605.22365](https://arxiv.org/abs/2605.22365) | TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2605.26574](https://arxiv.org/abs/2605.26574v3) | GradSentry: Gradient Spectral Entropy for Backdoor Sample Filtering in Large Language Model Fine-Tuning | [D4](d4-signals/REPORT.md)：SECTION_READ; [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2605.26595](https://arxiv.org/abs/2605.26595) | Cordyceps: Covert Control Attacks on LLMs via Data Poisoning | [D3](d3-conditions/REPORT.md)：全文; [D5](d5-defense/REPORT.md)：FULLTEXT_PARTIAL |
| [arxiv:2605.30189](https://arxiv.org/abs/2605.30189) | Token-Level Generalization in LoRA Adapter Backdoors: Attack Characterization and Behavioral Detection | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2606.00995](https://arxiv.org/abs/2606.00995) | Subliminal Learning Is Steering Vector Distillation | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED; [D3](d3-conditions/REPORT.md)：CROSS_DOMAIN_FULLTEXT_READ_D3_RELEVANT_VERIFIED; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2606.07963](https://arxiv.org/abs/2606.07963) | Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2606.11648](https://arxiv.org/abs/2606.11648) | Dummy Backdoor as a Defense: Removing Unknown Backdoors via Shared Internal Mechanisms for Generative LLMs | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2606.14078](https://arxiv.org/abs/2606.14078) | Rethinking Backdoor Adversarial Unlearning through the Lens of Catastrophic Forgetting in Continual Learning | [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED; [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2607.05748](https://arxiv.org/abs/2607.05748) | Two Sides of the Same Coin: Learning the Backdoor to Remove the Backdoor | [D4](d4-signals/REPORT.md)：SECTION_READ; [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2607.06643](https://arxiv.org/abs/2607.06643v1) | The Power of Backdoor Absorption in Community Training | [D1](d1-formation/REPORT.md)：全文; [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED |
| [arxiv:2607.09053](https://arxiv.org/abs/2607.09053) | An Emergent Mirage: Is Emergent Misalignment and Realignment Indeed a Robust Phenomenon? | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED; [D4](d4-signals/REPORT.md)：SECTION_READ; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2607.24893](https://arxiv.org/abs/2607.24893) | Early Detection of Distributed Backdoors in Multi-Agent LLM Systems: A Characterization Study | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2607.26849](https://arxiv.org/abs/2607.26849) | ToxScreen: Detecting Whether an LLM Has Been Poisoned | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2608.00732](https://arxiv.org/abs/2608.00732) | Mitigating Backdoors via Decoy Shortcuts and Knowledge Decoupling | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2608.02271](https://arxiv.org/abs/2608.02271) | Zero-shot Backdoor Detection in Parameter-Efficient Fine-Tuning via Canonical Spectral Signatures | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2608.07436](https://arxiv.org/abs/2608.07436v2) | Post-Grokking Collapse at the Representation–Readout Interface in Muon-Trained Transformers | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2608.11025](https://arxiv.org/abs/2608.11025v2) | Data Attribution of Emergent Misalignment with Persona Features | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2608.11295](https://arxiv.org/abs/2608.11295) | Backdoor Decontamination Dynamics in LLM Agents | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2608.30403](https://arxiv.org/abs/2608.30403) | Why Are LLM Backdoor Defenses Fragmented? A Feature-Level Explanation with Sparse Autoencoders | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2609.11799](https://arxiv.org/abs/2609.11799) | SpecGuard: Inference-Time Backdoor Detection For Free | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2609.15029](https://arxiv.org/abs/2609.15029) | Pick Your Poison: Learning to Select Poison Sets for Stronger LLM Backdoor Attacks | [D3](d3-conditions/REPORT.md)：全文 |
| [arxiv:2609.36117](https://arxiv.org/abs/2609.36117) | Why Backdooring Neural Networks is so Easy? | [D1](d1-formation/REPORT.md)：全文 |
| [arxiv:2610.00348](https://arxiv.org/abs/2610.00348v1) | Removing the NEEDLE in the Haystack: Backdoor Removal in LLMs via Weight Orthogonalisation | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2610.00663](https://arxiv.org/abs/2610.00663) | Backdoor Containment via Expert Quarantine and Shutdown in LLMs | [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2610.07654](https://arxiv.org/abs/2610.07654) | Does On-Policy Distillation for Safety Pose Backdoor Risks? | [D1](d1-formation/REPORT.md)：全文; [D4](d4-signals/REPORT.md)：SECTION_READ; [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2610.10657](https://arxiv.org/abs/2610.10657v1) | Beyond Owls: Subliminal Learning Can Transfer Learned Capabilities and Backdoors | [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2610.20319](https://arxiv.org/abs/2610.20319) | Beyond Static Cues: Unified Dynamic Detection of MLLM Backdoors via Dual-Flow Trajectory Analysis | [D4](d4-signals/REPORT.md)：全文 |
| [arxiv:2610.32866](https://arxiv.org/abs/2610.32866) | Interpretable Sparse Fine-Tuning with TopKLoRA Enables Faithful Circuit Discovery | [D4](d4-signals/REPORT.md)：SECTION_READ; [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2610.37741](https://arxiv.org/abs/2610.37741) | Agentic Backdoors via Pretraining Poisoning | [D1](d1-formation/REPORT.md)：全文; [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED; [D3](d3-conditions/REPORT.md)：CROSS_DOMAIN_FULLTEXT_READ_D3_RELEVANT_VERIFIED |
| [arxiv:2611.00748](https://arxiv.org/abs/2611.00748) | Inherited or Learned? Backdoor Selectivity in Distilled Language Models | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED; [D3](d3-conditions/REPORT.md)：全文; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2611.10110](https://arxiv.org/abs/2611.10110) | Homeopathic Learning: Diluting Poisoned Fine-tuning Data Can Strengthen Its Effect | [D1](d1-formation/REPORT.md)：CROSS_DOMAIN_SECTION_CHECKED; [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED; [D3](d3-conditions/REPORT.md)：CROSS_DOMAIN_FULLTEXT_READ_D3_RELEVANT_VERIFIED; [D4](d4-signals/REPORT.md)：SECTION_READ; [D6](d6-generalization/REPORT.md)：全文 |
| [arxiv:2611.12807](https://arxiv.org/abs/2611.12807) | Suppression Is Not Erasure: Mechanistic Analysis and Reactivation Boundaries of Subspace Projection for LoRA Backdoor Defense | [D2](d2-persistence/REPORT.md)：CROSS_DOMAIN_FULLTEXT_TARGETED_VERIFIED; [D5](d5-defense/REPORT.md)：全文 |
| [arxiv:2611.14143](https://arxiv.org/html/2611.14143v1) | GARD: Gradient-Guided Adversarial Representation Defense against Backdoors in Multimodal Contrastive Learning Models | [D5](d5-defense/REPORT.md)：FULLTEXT_PARTIAL |
| [arxiv:2611.14970](https://arxiv.org/abs/2611.14970) | Poison-then-Hide: Finetuning-Activated Backdoor Attack on Pretrained Vision Encoders | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2611.15479](https://arxiv.org/abs/2611.15479) | DETER: Suppression During Pretraining for Tamper Resistance | [D5](d5-defense/REPORT.md)：全文; [D6](d6-generalization/REPORT.md)：DOWNLOADED_NOT_READ_DELEGATED |
| [arxiv:2611.19656](https://arxiv.org/abs/2611.19656) | BadGCL: Towards Persistent Backdoor Attack on Graph Continual Learning Systems | [D2](d2-persistence/REPORT.md)：全文 |
| [arxiv:2611.22446](https://arxiv.org/abs/2611.22446) | As Student Rollouts Continue, Visual Backdoors Keep Growing | [D1](d1-formation/REPORT.md)：MAIN_TEXT_READ_APPENDIX_MISSING |
| [harvard-cs2881r:zheng-zen-phase-transitions](https://boazbk.github.io/mltheoryseminar/student_projects/final_papers_and_posters/papers/Phase_Transitions_in_Backdoor_Learning__Minimum_Data_Poisoning_Thresholds_for_LLM_Backdoors_%282%29_-_Kaden_Zheng.pdf) | Phase Transitions in Backdoor Learning: Minimum Data Poisoning Thresholds for LLM Backdoors | [D1](d1-formation/REPORT.md)：全文 |
| [neurips-2022-trap-replace](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ea06e6e9e80f1c3d382317fff67041ac-Abstract-Conference.html) | Trap and Replace: Defending Backdoor Attacks by Trapping Them into an Easy-to-Replace Subnetwork | [D5](d5-defense/REPORT.md)：FULLTEXT_PARTIAL |
| [neurips:2022/4491ea1c91aa2b22c373e5f1dfce234f](https://papers.nips.cc/paper_files/paper/2022/hash/4491ea1c91aa2b22c373e5f1dfce234f-Abstract-Datasets_and_Benchmarks.html) | BackdoorBench: A Comprehensive Benchmark of Backdoor Learning | [D1](d1-formation/REPORT.md)：全文 |
| [pmlr-v306-huang26n](https://proceedings.mlr.press/v306/huang26n.html) | Persistent Backdoor Attacks in Class-Incremental Learning via Structural Invariant Anchoring | [D2](d2-persistence/REPORT.md)：全文 |
| [pmlr-v306-zhao26bf](https://proceedings.mlr.press/v306/zhao26bf.html) | Broadening the Backdoor Basin: Understanding LLM Backdoors Collapse and Making Backdoors Persistent | [D2](d2-persistence/REPORT.md)：全文; [D5](d5-defense/REPORT.md)：FULLTEXT_PARTIAL |
| [pmlr:v202/xian23a](https://proceedings.mlr.press/v202/xian23a.html) | Understanding Backdoor Attacks through the Adaptability Hypothesis | [D1](d1-formation/REPORT.md)：全文 |
| [pmlr:v235/li24at](https://proceedings.mlr.press/v235/li24at.html) | A Theoretical Analysis of Backdoor Poisoning Attacks in Convolutional Neural Networks | [D1](d1-formation/REPORT.md)：全文; [D6](d6-generalization/REPORT.md)：CROSS_DOMAIN_SECTIONS_READ |
| [pmlr:v97/shen19e](https://proceedings.mlr.press/v97/shen19e.html) | Learning with Bad Training Data via Iterative Trimmed Loss Minimization | [D1](d1-formation/REPORT.md)：全文 |
| [ssrn:7113514](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7113514) | Training-Time Detection of Fixed-Target Backdoor Data Poisons in Stable Diffusion Fine-Tuning via Anchored Denoising Loss Trajectories | [D4](d4-signals/REPORT.md)：FULLTEXT_UNAVAILABLE |
| [usenix-security25-guo](https://www.usenix.org/conference/usenixsecurity25/presentation/guo-zhen) | Persistent Backdoor Attacks in Continual Learning | [D2](d2-persistence/REPORT.md)：全文 |
