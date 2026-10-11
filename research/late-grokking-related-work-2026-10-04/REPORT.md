# 后门 ASR 晚期跃升：现象查重与 Related Work Study

> **2026-10-05研究范围修正：** 用户明确研究ASR在短checkpoint窗口内的跃升，不以发生早晚为主问题。新的证据排序、遗漏补查和定位判断见 [ASR-JUMP-REAUDIT.md](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/ASR-JUMP-REAUDIT.md)。下文保留为此前89条历史清单，不能继续将late grokking作为已确认主张。

检索与核验：2026-10-04 至 2026-10-05。研究对象：沿训练 checkpoint，触发条件下的攻击成功率长期低位、随后短窗口上升；同时查训练后休眠激活、投毒剂量阈值、良性任务延迟泛化及指标造成的表观突变。

2026-10-05 设置比较补正：上一轮遗漏的 Moderate-fitting（NeurIPS 2022）和 LoRA Lifespan（FPS 2025）已补入核心前史；另新增8篇适配/生成式近邻。详细异同见 [COMPARISON.md](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/COMPARISON.md)。

## 1. 直接结论

**确实有人观察过非常相似的曲线，而且安全领域和 LLM 中都有直接先例。仅以“首次发现后门 ASR 晚期跃升”作为贡献不成立。** 最直接的是 Connor & Emanuele 的 FixMatch 研究（2023，Fig. 2）、Schneider 等的 Universal Backdoor Attacks（ICLR 2024，Fig. 3），以及 Betley 等的 Inductive Backdoors（2025，Fig. 10）。第三篇的指标是未见触发器上的泛化准确率，不等于传统固定触发器 ASR；前两篇直接画 ASR 随训练的变化。[A01](https://openreview.net/references/pdf?id=nY39siRkv)、[A02](https://arxiv.org/abs/2312.00157)、[A03](https://arxiv.org/abs/2512.09742)。

**现有文献没有给出一个可以直接套到我们全部模型上的统一机制。** 伪标签反馈、正常特征与触发器竞争、记忆与泛化电路竞争、优化器动力学、几何压缩、指标阈值和数值精度，都有各自设置下的证据。2026 年的 Grokking 综述也明确认为没有单一解释覆盖所有行为。[G01](https://doi.org/10.1145/3814603)。

**值得继续的方向，是说明何时会晚学、为何突然加速，以及能否在行为跃升前识别并有效干预。** “多模型都有”提供现象范围；更强贡献要给出失败条件、因果干预和可检验预测。内部预警问题已有一个非常接近的协议预印本，但它明确没有给出确认性实验结论：[P01](https://doi.org/10.21203/rs.3.rs-10594749/v1)。因此这个问题仍可研究，但“首次提出内部前兆比较框架”也不宜直接声称。

本报告收录的文章按距离分组。它们不是都观察了同一种现象；“作者报告的最终结论”也不等于本报告独立复现的事实。

共 **89 条编号文献记录**：6 篇最直接动态研究、14 篇潜伏/复活/持久性近邻、15 篇后门学习动态与反例、33 篇 grokking 机制/综述、5 篇 diffusion 研究、5 篇相邻能力跃迁、1 篇直接预警协议，以及后续设置核验新增的10篇直接近邻。包括综述及 BackdoorBench 扩展，不能解释成 89 篇独立证实后门晚期跃升的实验。另列关键词误命中及尚未充分核验的线索。

## 2. 如何阅读证据等级

| 标记 | 本轮读到了什么 | 可以支持什么 |
|---|---|---|
| F | 原始全文中相关段落、图注或附录；核心图另作视觉核对 | 可核对现象、横轴、解释及限制；没有重跑实验 |
| A | 原始论文摘要/出版页面 | 只概括作者明确写出的结论，不补造曲线、种子数或机制细节 |
| P | 协议、课程项目、工程文档 | 有研究思路或观察线索，不能当作成熟确认性结论 |

同行评审与阅读深度是两条独立轴。预印本全文可以核验其“确实写了什么”，不能因此升级成已同行评审或已被独立验证。下表年份保留初稿/正式发表区别；不把改名或更新版本当成两篇独立证据。

## 3. 最直接、必须逐篇对照的论文

### A01. Exploring Vulnerabilities of Semi-Supervised Learning to Simple Backdoor Attacks

Marissa Catherine Connor、Vincent Emanuele；ICLR 2023 BANDS workshop；F。[原文](https://openreview.net/references/pdf?id=nY39siRkv)。证据：Fig. 2、§4–5，PDF 第 4 页；本地 `sources/fixmatch.pdf`。

- **观察**：FixMatch 中 ASR 可以先低位维持，再快速上升；图中不同目标类别的跃升时间和稳定程度不同。对照的监督学习曲线更渐进。图例是目标类别，不是随机种子。
- **解释**：投毒样本伪标签集中到目标类别有助于形成弱后门；一部分语义不清楚的样本促使模型更依赖触发器。这是作者讨论的解释，未证明是所有晚期跃升的原因。
- **如何利用**：提出以“建立弱后门的样本”和“增强后门的样本”分别发挥作用的攻击思路；强调评测应覆盖足够长训练和多目标类别。
- **最终结论**：无标签数据中的简单触发器也能形成强后门；低 ASR 的早期训练不能用来宣布抗投毒。
- **对我们**：最直接的曲线先例；不能把其伪标签机制移植到没有伪标签环节的 SFT。

### A02. Universal Backdoor Attacks

Benjamin Schneider、Nils Lukas、Florian Kerschbaum；2023 预印本，ICLR 2024；F。[论文](https://arxiv.org/abs/2312.00157)。证据：§4.2、Fig. 3、Table 1，PDF 第 6–7 页；`sources/universal.pdf`。

- **观察**：patch-encoding baseline 在约 70 epochs 后才明显学会后门，ASR 短窗口升到高位；作者将其描述为训练过拟合后发生。另一个结果是投毒数从 5,000 到 8,000 时 baseline ASR 大幅变化，属于剂量轴，不能混成训练轴。
- **解释**：baseline 低剂量下学后门太晚，早停使其最终 ASR 接近零；更多投毒使其更早形成。
- **如何利用**：设计更易逐渐学会的 universal trigger，使早停抑制后门时也严重损害干净准确率。其方法曲线本身还有台阶，不能把作者“gradually learned”理解成所有 checkpoint 都线性增长。
- **最终结论**：攻击的可学习性和训练时机决定早停是否有效；跨类别可迁移的后门可在很低投毒比例下成功。
- **对我们**：既占据了晚学现象，也占据了“利用晚学做早停防御”的基本思想。值得测试的是晚学机制和能够避开早停的攻击条件。

### A03. Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs

Jan Betley 等；2025 预印本；F。[论文](https://arxiv.org/abs/2512.09742)。证据：§5.1、Fig. 10、Appendix F.2.1；`sources/2512.09742.pdf`。

- **观察**：GPT-4.1 的总统实验，30 个随机种子中约三分之一成功；成功组第一 epoch 接近基线，第二 epoch 快速到近乎完美，之后保持；成功组和失败组的训练准确率都平滑上升。图注的随机基线为 0.83，不能因为是二选一就擅自改成 0.5。
- **解释**：成功组学到了泛化规则，而不仅拟合训练例子；作者明确称其类似 grokking。没有建立统一因果机制。
- **如何利用**：构造训练中缺失具体触发器—行为组合、但模型可以推断出来的 inductive backdoor；用来说明窄域微调可能引起训练例子未直接写出的行为。
- **最终结论**：LLM 后门可通过归纳泛化形成，且形成过程可能呈现种子依赖的阶段跃迁。
- **对我们**：LLM + 多种子 + 快速晚期泛化，是非常接近的先例。但其“成功种子上很稳定”不等于全部种子稳定；其 held-out trigger 泛化也不同于我们已在训练中出现的触发器。

### A04. Towards a Defense Against Federated Backdoor Attacks Under Continuous Training

Shuaiqi Wang、Jonathan Hayase、Giulia Fanti、Sewoong Oh；2022 初稿，TMLR 2023；F。[论文](https://arxiv.org/abs/2205.11736)。旧 PDF 标题有 Continual Federated Learning 表述；按同一工作处理。证据：backdoor leakage、Fig. 1、§4、Appendix D；`sources/shadow.pdf`。

- **观察**：过滤虽令后门学得比主任务慢，残余恶意更新仍会累计，足够长训练后 ASR 可以到 1；较高恶意比例时，后门反而可能比主任务更快。
- **解释**：不完美过滤造成持续泄漏，不必假设突然出现一种新能力。
- **如何利用**：主干持续训练，影子模型在过滤后的更新上训练并早停，周期重训以适应数据变化。
- **最终结论**：单轮过滤效果不能保证持续训练安全；早停的收益依赖污染比例和任务可学习性，简化两层模型下有理论支持。
- **对我们**：是晚期风险和早停应用先例；累计缓慢形成与真正短窗口加速要用连续指标分开。

### A05. The Perils of Learning From Unlabeled Data: Backdoor Attacks on Semi-supervised Learning

Virat Shejwalkar、Lingjuan Lyu、Amir Houmansadr；ICCV 2023；F。[原文](https://people.cs.umass.edu/~amir/papers/2023-ICCV-backdoor-SSL.pdf)。证据：§5.1.4、Fig. 3。

- **观察**：沿训练同时测 ASR 和伪标签/目标类别置信度；先识别目标类别语义，随后将简单触发器关联到该类。它是阶段性后门形成的直接动态研究，本文不把每条曲线都认定为严格 grokking。
- **解释与利用**：攻击等待正确伪标签出现，再利用反馈学习触发器；重复、抗增强的全图模式帮助跨 SSL 算法攻击。
- **最终结论**：弱能力攻击者仅投毒约 0.2% 无标签数据，也能在多个 SSL 设置得到超过 80% ASR；高正常任务表现不保证安全。
- **对我们**：值得借鉴“先具备正常识别能力，再学触发映射”的条件分解；应比较正常任务能力和 ASR 的先后关系，而非只看总 loss。

### A06. Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples

Alexandra Souly、Javier Rando、Ed Chapman、Xander Davies 等；2025 预印本；F。[论文](https://arxiv.org/abs/2510.07192)。证据：训练 checkpoint 的投毒累计量分析、数据顺序/学习率附录；`sources/poison-count.pdf`。

- **观察**：600M–13B 模型、随规模扩大的预训练数据，250 个 DoS 投毒文档可产生类似强影响；分析累计见过的投毒样本、顺序及干净续训。不是证明所有模型都在固定 checkpoint 突然跃升。
- **解释与利用**：挑战“攻击所需样本与干净数据成比例”的假定；强调绝对投毒量及曝光结构。
- **最终结论**：在测试的任务和训练设置中，扩大干净训练数据不能使攻击预算按比例增加；微调条件、学习率、顺序和续训会改变结果。
- **对我们**：必须对照 step、累计 poison exposure 和有效优化量；不能把 250 或任何剂量阈值当成跨模型定律。

## 4. 低 ASR 后激活、复活与持久性：必须区分的近邻

| ID / 论文与年份 | 观察与作者最终结论 | 解释或利用方式 | 与我们的距离；核验 |
|---|---|---|---|
| B01 [Latent Backdoor Attacks on Deep Neural Networks](https://arxiv.org/abs/1905.10447)，CCS 2019 | 预训练模型中可埋下尚不生效的后门，下游迁移后完成并激活 | 提前植入未来目标类别相关表示；攻击模型供应链 | 目标标签/下游任务改变后激活；不是固定训练配置自然晚学。旧名 Regula Sub-rosa 同一篇；F |
| B02 [Hibernated Backdoor: A Mutual Information Empowered Backdoor Attack to Deep Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/21272)，AAAI 2022 | 初始低 ASR，正常微调激活；修补后继续微调还能重新升高，Fig. 5 | 利用干净/后门梯度互信息构造休眠状态；也设计特定微调数据才唤醒的版本 | 直接否定“clean fine-tuning 必然清除后门”；主动构造，不等于我们的自然形成。F |
| B03 [Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning](https://arxiv.org/abs/2505.16567)，2025 / ICLR 2026 | 微调前表现正常，正常 SFT、蒸馏或偏好优化后出现广告、拒答或安全退化等行为 | FAB 元学习模拟受害者后续微调，优化休眠与激活 | LLM clean-update activation 的重要先例；后续微调是设计的开关。F |
| B04 [Dormant Backdoor: Weaponizing Model Finetuning for Feasible Backdoor Attacks Against Pretrained Models](https://ojs.aaai.org/index.php/AAAI/article/view/39480)，AAAI 2026 | 不知道受害者精确微调数据，也可构造在后续微调中激活的后门 | 数据无关代理轨迹、双层优化、语义锚点和梯度统计 | 改进休眠攻击的可行性，不是证明无设计后门普遍晚学。F |
| B05 [Uncovering, Explaining, and Mitigating the Superficial Safety of Backdoor Defense](https://arxiv.org/abs/2410.09838)，2024 | 防御后的低 ASR 模型仍可能残留后门，少量投毒重学会快速恢复 | 参数空间路径分析；PAM 强化离开后门解的过程 | “压低输出”与“消除能力”有区别；重投毒复活不是无新投毒跃升。A |
| B06 [Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack](https://arxiv.org/abs/2405.16134)，NeurIPS 2024 | 被防御模型可被小的触发器改动重新激活，包括黑盒设置 | 估计后门存在系数并优化 re-activation trigger | 主要是推理侧换触发器，不是训练 checkpoint 跃升。A |
| B07 [UBA-Inf: Unlearning Activated Backdoor Attack with Influence-Driven Camouflage](https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui)，USENIX Security 2024 | 伪装数据掩盖后门，遗忘它们后后门激活 | 影响函数选择伪装样本；将 unlearning 当作攻击开关 | 主动去掉抑制因素；可借鉴“抑制与能力并存”的干预思路。F |
| B08 [BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning](https://arxiv.org/abs/2508.15541)，2025 | FL 中正常训练阶段隐藏，合法遗忘请求移除伪装后转成后门状态 | 分别构造后门和伪装样本 | B07 的联邦邻域；作者仓库明确引用并复用 UBA-Inf 部分实现。A |
| B09 [Benign on Label, Malicious by Design: Clean-Label Dormant-to-Activated Backdoor via Machine Unlearning with Removable Camouflage](https://arxiv.org/abs/2607.27936)，2026 | clean-label 数据也可实现休眠到激活，在多种遗忘方法下评估 | 潜在植入与可移除抑制分别生成 | 扩展 clean-label/unlearning 条件；不能拿标签干净证明状态安全。A |
| B10 [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566)，2024 | 条件后门可在安全训练后保持，某些对抗训练令模型更会识别并隐藏触发条件 | 用年份等条件区分安全与不安全行为 | 持久性的重要先例；不是同一训练过程中从低 ASR 自发晚学。A |
| B11 [Persistent Pre-Training Poisoning of LLMs](https://arxiv.org/abs/2410.13722)，2024 | 600M–7B 预训练投毒后，多个攻击目标可在 SFT/DPO 后保留；简单 DoS 所需比例更低 | 从头预训练、再对齐评估持久性 | 证明跨训练阶段保留，不证明固定 checkpoint 相变。A |
| B12 [Backdoor Attacks for In-Context Learning with Language Models](https://arxiv.org/abs/2307.14692)，2023 | 参数后门可跨提示方式起效；其设置中少量正常微调能够移除行为，而仅提示工程难以可靠防御 | 专门训练兼顾上下文变化的后门；研究续训移除 | 与 B02/B03/B10 形成条件性对照：干净续训可能减弱，也可能激活，取决于植入方式。F |
| B13 [Persistent Backdoor Attacks Under Continual Fine-Tuning of LLMs](https://ojs.aaai.org/index.php/AAAI/article/view/40295)，2025 初稿 / AAAI 2026 | 普通植入在反复正常更新后可能减弱，P-Trojan 在 Qwen2.5/LLaMA3 及所测任务序列中保留很强后门 | 将毒梯度与干净任务的 token-embedding 梯度对齐 | 是“持续后门”的近邻，不能把持续性度量直接当 ASR；不是自发晚学证明。A |
| B14 [Watermarking for Model Ownership Verification: Invisible at Deployment, Activated by Updates](https://doi.org/10.1145/3817059)，Hewang Nie 等，ACM TOPS 2026 | 部署初期低响应，正常微调后水印恢复；多个图像模型/数据集验证 | DormMark 分植入、遮蔽、微调激活三阶段，利用遗忘—恢复作所有权验证 | 是类似现象的正向用途；有意遮蔽和唤醒，不等于自然 grokking。A，原始 ACM 页面已核验 |

## 5. 后门学习动力学、反例与可借鉴防御

| ID / 论文 | 作者观察、解释与最终结论 | 使用方式与必须避开的推断；核验 |
|---|---|---|
| D01 [Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://arxiv.org/abs/2110.11571)，NeurIPS 2021 | 多种常见后门的投毒训练样本被快速拟合，loss 比干净样本更早变小 | ABL 早期隔离、后期反学习。是“后门快学”的反例；poison train loss 与 held-out ASR 不同，不能一项替代另一项。F |
| D02 [Training with More Confidence: Mitigating Injected and Natural Backdoors During Training](https://arxiv.org/abs/2202.06382)，NeurIPS 2022 | 作者明确指出 slow poisoning 可绕开仅靠学习速度差异的防御；分析后门神经元分类面 | NONE 从几何/置信度着手；“慢学可以破快学检测”已有先例，不宜作为新概念。A |
| D03 [On the Difficulty of Defending Contrastive Learning against Backdoor Attacks](https://www.usenix.org/system/files/usenixsecurity24-li-changjiang.pdf)，USENIX Security 2024 | 对比学习中的干净/投毒 loss 动态不呈现监督学习中那种明显快慢分离 | 表明现成监督防御不易迁移；VLM 的视觉编码器训练目标必须单独看。F，Fig. 2 |
| D04 [SEEP: Training Dynamics Grounds Latent Representation Search for Mitigating Backdoor Poisoning Attacks](https://aclanthology.org/2024.tacl-1.55/)，TACL 2024 | 训练动态可提供高精度投毒种子，潜在空间传播补足召回 | 已将动态和表示联合用于防御；不是“内部信号在行为前”的证明。A |
| D05 [A Theoretical Analysis of Backdoor Poisoning Attacks in Convolutional Neural Networks](https://proceedings.mlr.press/v235/li24at.html)，ICML 2024 | 在两层 CNN、多视图数据假设下分析触发器与正常特征的学习竞争；成功依赖投毒量和触发器强度等条件 | 对照晚学与快学的理论起点；不能把两层模型条件推广为 VLM/LLM 统一定理。F，§4.3 |
| D06 [BackdoorBench: A Comprehensive Benchmark of Backdoor Learning](https://arxiv.org/abs/2206.12654)，NeurIPS 2022 | 标准化大量攻击—防御比较；附录 B.6 在五种攻击下观察投毒测试 loss 快降，梯度信噪比、范数较高 | 本文直接涉及 held-out ASR/测试 loss 的快学证据，比只看训练 loss 更重要。F |
| D07 [BackdoorBench: A Comprehensive Benchmark and Analysis of Backdoor Learning](https://arxiv.org/abs/2407.19845)，2024 扩展工作 | 扩大评测和对学习、记忆动态的分析；不同剂量、模型、数据和防御改变结果 | 与 D06 是同一 benchmark 的扩展，不能当两组独立复现实验累加。F |
| D08 [Exploring Dynamic Properties of Backdoor Training Through Information Bottleneck](https://arxiv.org/abs/2511.21923)，2025 | 用信息量轨迹、聚类和压缩分析触发器/语义的不同动态；视觉不可见与信息隐蔽不等价 | 可借鉴动态表示指标；MI 估计随 checkpoint 改变不自动建立因果机制。F |
| D09 [Why Backdooring Neural Networks is so Easy?](https://arxiv.org/abs/2609.36117)，2026-09 | 在二次神经元、Gaussian 输入的限制模型中推导 lazy/feature-learning 攻击条件与尺度关系 | 从触发器强度、毒量和训练机制联系低预算攻击；没有证明我们那种时间曲线。A |
| D10 [Poisoning the Unlabeled Dataset of Semi-Supervised Learning](https://arxiv.org/abs/2105.01622)，USENIX Security 2021 | 极少无标签投毒可造成定向错误，现代 SSL 的自反馈构成攻击面 | A01/A05 的前史；主要为 instance-targeted poisoning，不等于 universal conditional backdoor。F |
| D11 [DeHiB: Deep Hidden Backdoor Attack on Semi-supervised Learning via Adversarial Perturbation](https://ojs.aaai.org/index.php/AAAI/article/view/17266)，AAAI 2021 | 无标签恶意样本也能植入后门；对抗扰动和对比投毒提高目标伪标签与触发效果 | A01/A05 的直接前史；不同攻击者知识、预训练条件应对齐再比较。F |
| D12 [Filter, Obstruct and Dilute: Defending Against Backdoor Attacks on Semi-Supervised Learning](https://arxiv.org/abs/2502.05755)，2025 初稿 | 以过滤、互补学习、trigger mix-up 破坏伪相关；报告测试设置下显著降低 ASR | 与 A05 等 SSL 工作的后续防御分支；不是晚学机制统一解释。A，按已核验初稿著录 |
| D13 [COLLIDER: A Robust Training Framework for Backdoor Data](https://openaccess.thecvf.com/content/ACCV2022/papers/Dolatabadi_COLLIDER_A_Robust_Training_Framework_for_Backdoor_Data_ACCV_2022_paper.pdf)，ACCV 2022 | 梯度 coreset 与局部内在维度用于筛掉毒数据；干净准确率存在取舍 | 文中正常准确率在 LR milestone 跳变提醒我们核对调度器；不是 ASR 晚期跃升的直接同类。F |
| D14 [Secure-CHG: A Comprehensive Framework for Robust and Fair Federated Learning via Hybrid Defense and Contribution-Aware Trust](https://arxiv.org/abs/2606.31066)，2026 | 收敛后梯度统计差异变弱，传统 FL 防御可能 late-stage failure；以贡献/样本难度改善检测 | 是攻击/防御交互下晚期失败；不能当作固定攻击配置自发 grokking。A |
| D15 [Unveiling the Backdoor Mechanism Hidden Behind Catastrophic Overfitting in Fast Adversarial Training](https://arxiv.org/abs/2604.24350)，2026 | 将快速对抗训练中骤然鲁棒性崩塌联系到弱触发模式、特征路径，并研究干预 | 相邻“突然坏学”的机制线索；横轴和事件相关，测的是鲁棒性崩塌，不是同一投毒 ASR。A |

## 6. Grokking、隐蔽进展与跃升机制：逐篇结论

以下研究主要分析正常任务。它们提供可检验的候选机制和实验方法，**都不能仅凭同形曲线就证明我们的后门具有其机制**。

| ID / 论文与年份 | 作者最终结论 / 观察 | 如何解释或运用；对我们可借鉴的部分；核验 |
|---|---|---|
| G01 [A Survey on Grokking](https://doi.org/10.1145/3814603)，ACM Computing Surveys 2026 | 权重衰减、初始化、数据量均影响 grokking；现有多个理论可以兼容，没有单一机制充分覆盖 | 用来补全关键词之外的文献与争议，不能把综述当独立实验证据。F，本地 survey.pdf |
| G02 [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)，Power 等，2022 | 训练已拟合后，测试泛化仍可能很晚才从基线变到完美；数据越少所需优化可更多 | 经典定义强调 train–test 时间分离；只有 ASR 后期上升尚不足以证明经典 grokking。A |
| G03 [Towards Understanding Grokking: An Effective Theory of Representation Learning](https://arxiv.org/abs/2205.10343)，2022 | 从表示学习有效理论描述记忆、理解、grokking 等不同区间 | 给表示变化和数据/训练条件预测；限制任务上的理论。A |
| G04 [Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit](https://arxiv.org/abs/2207.08999)，NeurIPS 2022 | sparse parity 的损失/错误率可长期看似不动，但 SGD 通过梯度 Fourier gap 持续放大解 | 不是“随机在黑暗中撞到解”；直接支持测隐藏进展，因果机制仅在其分析条件下成立。A |
| G05 [The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon](https://arxiv.org/abs/2206.04817)，2022 | 自适应优化器产生稳定/不稳定循环、末层范数变化，与其设置中泛化开始紧密相连 | 优化器状态可能是机制，不只是学习率；不能宣布所有 grokking 都需 slingshot。A |
| G06 [Omnigrok: Grokking Beyond Algorithmic Data](https://arxiv.org/abs/2210.01117)，ICLR 2023 | 图像、语言、图任务也可 grok；权重范数与训练/测试损失地形错配解释其多种实验，控制范数可消除延迟 | “跨域 grokking”已有先例；看 norm 并做干预，不能把相关曲线等同因果定律。A |
| G07 [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217)，ICLR 2023 | 模加中 Fourier 电路形成及之后记忆清理可在行为跃升前持续发生 | memory / circuit formation / cleanup；因果消融与内部进展指标。已占据“输出前内部可见进展”的良性任务思路。F |
| G08 [A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks](https://arxiv.org/abs/2303.11893)，2023 | sparse parity 后期少数神经元范数快增，稀疏泛化网络替代稠密低泛化网络 | 竞争两套电路的候选解释；应分别压制触发路径与正常答案路径验证。A |
| G09 [Explaining grokking through circuit efficiency](https://arxiv.org/abs/2309.02390)，ICLR 2024 | 泛化电路学得慢但单位范数的 logit 更有效，正则下最终占优；预测 ungrokking/半泛化等 | 可检验电路效率竞争；A03 的明确引用，不能只引用 Power。F |
| G10 [Grokking as the Transition from Lazy to Rich Training Dynamics](https://arxiv.org/abs/2310.06110)，2023 / ICLR 2024 | 某些设置从初始 kernel/lazy 状态进入特征学习才泛化；数据与目标的初始对齐影响延迟 | 用冻结/可训练模块、特征变化测试；不是所有 grokking 的必要条件。A |
| G11 [Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking](https://arxiv.org/abs/2311.18817)，ICLR 2024 | 大初始化、小权衰下早期与晚期隐式偏置不同，可证明延迟泛化 | 理论解释有明确模型假设；不能拿“phase”词证明大型模型相变。A |
| G12 [Grokking as Compression: A Nonlinear Complexity Perspective](https://arxiv.org/abs/2310.05918)，2023 | 线性映射数 LMN 表示压缩，能比 L2 范数更直接刻画某些泛化前进展 | 测复杂度/压缩候选信号；LMN 不是所有深网可低成本准确计算的监控指标。A |
| G13 [Grokking in Linear Estimators — A Solvable Model that Groks without Understanding](https://arxiv.org/abs/2310.16441)，ICLR 2024 | 线性模型可精确算出延迟，准确率骤升可能只是指标效应，而非理解/新表示 | 我们必须以 margin、概率、loss 对照 ASR；“线性模型也有”否定唯一特征学习解释。A |
| G14 [To grok or not to grok: Disentangling generalization and memorization on corrupted algorithmic datasets](https://arxiv.org/abs/2310.13061)，ICLR 2024 | 模型可同时记住错误标签并泛化正常规律；正则后期清掉记忆时训练准确率反而骤降 | 分离两套表示、剪除/衰减对照；随机错标签不等于条件触发器投毒。A |
| G15 [Grokking as a First Order Phase Transition in Two Layer Networks](https://arxiv.org/abs/2310.03789)，ICLR 2024 | 自适应核理论中两类 teacher–student 任务映射到一阶转变，前后表示明显不同 | 严格相变理论有模型边界；不是看到折线台阶就能宣布热力学相变。A |
| G16 [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)，2024 | 多种真实图像模型的局部线性区复杂度变化与延迟泛化/对抗鲁棒性相关 | 标题的 Always 不能读成已证明所有训练都 grok；测决策边界复杂度。A |
| G17 [Progress Measures for Grokking on Real-world Tasks](https://arxiv.org/abs/2405.12755)，2024 | MNIST/IMDb 中激活稀疏度、权重熵和局部电路复杂度等提供进展信号；norm 不是独立充分解释 | 可做候选特征，但必须打败同等敏感输出基线和干净训练对照。A |
| G18 [Grokfast: Accelerated Grokking by Amplifying Slow Gradients](https://arxiv.org/abs/2405.20233)，2024 | 放大慢变梯度可大幅缩短其多个任务中的泛化延迟 | 已有实际加速应用；用于后门会加速还是抑制仍待验证，不能直接承诺防御。A |
| G19 [Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization](https://arxiv.org/abs/2405.15071)，NeurIPS 2024 | 组合/比较推理可通过长训获得；OOD 泛化依推理类型和电路结构不同 | 将 circuit efficiency 推向隐式推理；A03 明确引用。A |
| G20 [Unveiling the Dynamics of Information Interplay in Supervised Learning](https://proceedings.mlr.press/v235/song24g.html)，ICML 2024 | MIR/HDR 描述表示与分类头信息互动，分析 grokking 并用作训练正则 | 借鉴表示—输出头协同变化，而不只看某层 norm；不能称已知后门前兆。A |
| G21 [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697)，ICLR 2025 | SoftmaxCollapse、只增 logit 尺度的梯度可阻碍晚期学习；StableMax/梯度修改改善 | 检查 precision、溢出、logit scale、优化器；非线性加速仍可能有数值成分。A |
| G22 [Is Grokking a Computational Glass Relaxation?](https://arxiv.org/abs/2505.11411)，NeurIPS 2025 | 算术 transformer 中未发现其定义的一阶转变熵障碍；高熵泛化、WanD 优化结果挑战 norm-only 解释 | 与 G15 构成争议而非简单一致；实验设置不同，不足以相互一刀否定。A |
| G23 [Tracing the Path to Grokking: Dropout, Embeddings, and Network Activation](https://doi.org/10.1007/s11063-026-11843-4)，2025 初稿 / NPL 2026 | embedding、dropout、激活与梯度维度等可刻画延迟泛化的前后变化 | 动态诊断分支；初稿 Embeddings, Dropout 顺序不同，已去重。A |
| G24 [To Grok Grokking: Provable Grokking in Ridge Regression](https://arxiv.org/abs/2601.19791)，2026 | ridge regression 中可证明延迟泛化并量化参数依赖，可加强或消除延迟 | 线性反例提醒“新特征形成”不是所有 grokking 必需条件。A |
| G25 [A Systematic Empirical Study of Grokking: Depth, Architecture, Activation, and Regularization](https://arxiv.org/abs/2603.25009)，2026 | 架构差异受优化/正则匹配影响，深度作用非单调，残差影响稳定性 | 先匹配超参再谈模型家族差异；不要把调参差别写成架构机制。A |
| G26 [Grokking Is Conditional and Fragile: A Fully-Tractable, Multi-Seed Study at 12K Parameters](https://arxiv.org/abs/2607.05104)，2026 | 覆盖率、weight decay、数值环境影响成功率；少数 same-seed 结果因计算环境翻转，整体成功率未显著变 | 同时报告发生率与跳变时间；不能只展示成功种子或 seed-aligned 平均。A |
| G27 [How to Tame Grokking: Representation Geometry as a Control Signal](https://arxiv.org/abs/2607.11666)，2026 | 在所测算术/置换任务中维度压缩先出现，几何正则可按时机改变并加速 grokking | 由前兆到干预的案例；未知后门在线检测与 VLM 迁移没有得到验证。A |
| G28 [Noise-Driven Escape from Metastable Phases explains Grokking in Deep Neural Networks](https://arxiv.org/abs/2606.17120)，HiLD 2026 workshop | 故意构造线性深网亚稳态，SGD 噪声跨障碍，逃逸时间符合其 Arrhenius 分析 | 与 G04 的“非随机撞解”不同；噪声机制要用 batch/noise/等待时间分布检验。A |
| G29 [Circuit Synchronization Precedes Generalization: A Causal Precursor to Grokking](https://arxiv.org/abs/2606.12966)，2026 v2 | FSD 在所测算术设置先同步，再出现泛化；分叉增加 weight decay 提前跃升；非交换群需要合适基底 | 前兆+干预很接近方法方向；post-grokking ceiling 识别与在线阈值仍需分开审计。A，摘要声称不视为独立复现 |
| G30 [Tunneling the Loss Landscape: Bypassing Memorization with Monte Carlo Parameter Swapping](https://arxiv.org/abs/2608.01833)，2026 | 参数迁移、replica correlation 等显示其设置中的 glass 动态；SAM-Swap 加快泛化 | 扩展 G22 的训练动力学/优化应用；安全任务效果未知。A |
| G31 [Quantifying the Memorization-to-Generalization Transition: Scaling Laws and Phase Structure in Grokking](https://arxiv.org/abs/2609.10657)，2026-09 | 在 384 个算术 MLP 配置拟合 onset 与宽度、数据、LR、weight decay 的经验关系 | 是配置范围内拟合，不能把指数或阈值照抄到大型模型。A |
| G32 [Late-Stage Generalization Collapse in Grokking: Detecting anti-grokking with Weightwatcher](https://arxiv.org/abs/2602.02859)，2026 | 很长训练后部分 grokked 模型测试准确率再崩塌，训练准确率仍高；提出权重谱异常指标 | “学会后持久”要有观测时间窗；谱信号为作者主张，尚不能当后门部署监控保证。A |
| G33 [Grokking of Hierarchical Structure in Vanilla Transformers](https://arxiv.org/abs/2305.18941)，2023 | 域内饱和后结构泛化仍可很晚出现，最佳深度呈倒 U；结构性指标有助解释 | 延迟泛化不必都是极尖锐跳变；区分“晚”与“突然”。A |

## 7. 文生图和相邻的能力跃迁

| ID / 论文 | 作者最终结论 / 观测 | 我们应如何使用；核验 |
|---|---|---|
| T01 [How to Backdoor Diffusion Models?](https://openaccess.thecvf.com/content/CVPR2023/papers/Chou_How_to_Backdoor_Diffusion_Models_CVPR_2023_paper.pdf)，CVPR 2023（BadDiffusion） | 改造训练的 diffusion process，可同时保留生成质量并植入触发目标输出 | 无条件扩散基础先例，修改过程的威胁模型不同于纯 T2I 数据投毒；本轮未确认晚期 checkpoint 相变。F |
| T02 [From Trojan Horses to Castle Walls: Unveiling Bilateral Data Poisoning Effects in Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/95dcc1f6463491d37a8918c1d38380a7-Abstract-Conference.html)，2023 初稿 / NeurIPS 2024 | 不同投毒比例下出现不同生成语义与触发结果；低比例仍可能强触发 | 利用触发放大检测与防御。这里的重要 transition 是剂量区间，不是训练后半段。早期 ICLR 投稿稿使用 Bilateral Backdoor Effects，最终 NeurIPS 题目与 arXiv 一致；去重为一篇。F |
| T03 [The Stronger the Diffusion Model, the Easier the Backdoor: Data Poisoning to Induce Copyright Breaches Without Adjusting Finetuning Pipeline](https://arxiv.org/abs/2401.04136)，2024（SilentBadDiffusion） | 很少图文投毒可诱发版权目标复现，强模型更易受攻击；附录 F.2 的相似度随训练提升后波动平台 | FAE 是首次成功 epoch，CIR/SSCD 是不同指标。它提供轨迹/等待时间，但原文没有把其训练图结论写成普遍 late grokking。F |
| T04 [Grokking of Diffusion Models: Case Study on Modular Addition](https://arxiv.org/abs/2604.17673)，2026 workshop | flow-matching 算术图像任务也出现延迟泛化；去噪中存在完成数学结构的关键 sampling timestep | 是 diffusion grokking 先例，尚非 T2I 后门先例；training checkpoint 与 denoising timestep 必须分开。A |
| T05 [TooBad: Backdoor Diffusion Models with Ultra-Low Poison Rate and Imperceptible Trigger](https://arxiv.org/abs/2606.23362)，2026 预印本 | §4.4 / Fig. 6 比较 ASR、MSE、SSIM 随训练 epoch 的变化；优化触发器使后门约 3–5 epochs 达到高成功率，基线需要更久。作者以加速形成降低训练和投毒成本 | 触发器—目标匹配会改变形成时间，必须作为控制变量；原文三次运行取平均不等于每个种子同一时刻跃升。主要噪声触发器设置，不能直接当成语义文本触发 T2I 的统一 late grokking 证明。arXiv 版 Fig. 6，早期匿名稿 Fig. 7，避免混用。F |
| E01 [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)，2022 | induction head 形成与 ICL 急剧出现同阶段；小模型有较强因果证据，大模型更多相关证据 | 借鉴组件形成与能力曲线并测，不把大模型相关性冒充因果性。F |
| E02 [The mechanistic basis of data dependence and abrupt learning in an in-context classification task](https://arxiv.org/abs/2312.03002)，ICLR 2024 | 嵌套非线性操作依次学会，内在课程导致 induction head 突现；内部进展可先于 ICL | 候选“多环节门控”的强方法近邻，适合测试 trigger识别→映射→答案胜出的瓶颈。A |
| E03 [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)，NeurIPS 2023 | 离散/非线性评分可将连续改进显示为突变 | 主要研究模型规模上的涌现，不能因此否定全部训练时间突变；它要求我们检查连续输出。A |
| E04 [Weird Generalization is Weirdly Brittle](https://arxiv.org/abs/2604.10022)，2026 | 扩展复现中 weird generalization 依赖模型和数据，简单训练/提示干预可消除 | 不应把所有 weird-generalization 结果看成普遍定律；它不是 A03 Fig. 10 种子跃升的专门复现。A |
| E05 [From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs](https://arxiv.org/abs/2510.05169)，2025 | RL 可使模型推断自身触发器；最新全文 Fig. 6 中四类 reward 短窗口上升，一类更平滑 | 用推断出的触发器做检测/反学习；上升的是触发器恢复能力，不是后门首次形成。原文 buffer replay 为重要条件。F |

### P01. 内部预警的直接协议近邻

Zuo Yuchen，2026-08：[Testing Whether Internal Backdoor Precursors Precede Behavior: A Causal, Power-Matched Protocol](https://doi.org/10.21203/rs.3.rs-10594749/v1)；Research Square 预印本，F/P；`sources/precursor-protocol.pdf`。

原文 §4 和 §6 明确区分：两步 smoke test 只验证实现，不回答内部前兆是否早于行为。协议使用 GPT-2/SST-2、主动 marker 与 matched-null、稠密 checkpoint、连续输出指标、统一序贯阈值，并要求在宣称的 onset 对同一内部方向做 suppression/rescue。**最终结论是提出可执行判决程序与误差/功效分析，科学问题仍开放。**

与我们的重叠非常具体：先兆比较、干净对照、连续 margin、分叉干预都不能包装成从未有人想到；我们可贡献跨模型的确认性结果、触发特异信号和有效干预。本文核验了原始 Research Square 页面和 PDF；不以 ResearchGate 摘要替代原文。

## 8. 前后引用链与查重边界

箭头 `X → Y` 表示后来的 Y 原文引用 X，**不表示 Y 复现了 X**。下列边核对了参考文献或作者明确的来源说明；相邻检索结果没有强行画成引用关系。

| 经核对的链条 | 证据 / 解决的问题 |
|---|---|
| D10 Carlini 2021 → A01 FixMatch 2023 | A01 References 明确列出；无标签投毒前史，但 instance-targeted 与 backdoor 分开 |
| D11 DeHiB 2021 → A01 / A05 | 两文讨论并比较；对攻击者知识、伪标签条件和数据增强做边界审计 |
| D10 / D11 / D01 → A05 Perils 2023 | A05 参考文献及 §5 明确讨论；阶段式形成与快学防御同时进入对照 |
| G02 Power / G09 Varma / G19 Wang → A03 Inductive Backdoors 2025 | A03 Fig. 10 附近正文和 References；晚期后门泛化明确连接 grokking 与 circuit efficiency |
| B10 Sleeper Agents / B12 ICL backdoor / B11 Persistent pretraining → A06 Poison Count 2025 | A06 References 及 related work；形成、持久性、剂量应分开 |
| B01 Latent / B02 Hibernated → B04 Dormant 2026 | B04 References；主动休眠不是最近才出现的 LLM 概念 |
| B07 UBA-Inf → B08 BadFU | BadFU 作者公开仓库明确致谢并引用，联邦扩展线 |
| D01 / D04 / G07 / E01 / E02 / E03 → P01 Protocol 2026 | P01 References；内部监控问题已与后门动态和良性隐藏进展对接 |
| T01 BadDiffusion / VillanDiffusion / UIBDiffusion → T05 TooBad | T05 原文 §2、§3.4、References；沿用后门注入框架，改变触发器优化并研究训练收敛速度 |
| G02 / G06 / G07 / G09 / G13 / G15 / G33 等 → G01 Survey 2026 | 综述原文 References 与 §8–9；用于向前补机制、向后补语言任务和不同解释 |

额外做了前向索引检查：Semantic Scholar API 返回 429；OpenAlex 的 A03 条目 `W4417292295` 返回两条被引，均为教育心理学视角论文，不是新的训练后门实验证据；该条目的 references 为空。对 A01 的标题搜索未找到精确收录，而返回了若干 SSL 邻居。**这说明索引覆盖有缺口，不能把“引用库未返回”理解成“没有人引用”。** 原始 JSON 保存在 `sources/openalex-*.json`。因此本文用原文参考文献、标题前向搜索与主题扩展交叉补查，不声称下载了全球完整 cited-by 图。

前向主题补查的重要新增：D12 的 SSL 防御、B04 的休眠扩展、E04 的脆弱性复现、G22/G28 的相变争议、G29/P01 的前兆协议。除上表核对的边，不把它们全部声称为相互引用。

## 9. 哪些“很像”其实不是同一回事

| 命中 / 来源 | 本轮处置与理由 |
|---|---|
| [Delayed Backdoor Attacks: Exploring the Temporal Dimension as a New Attack Surface in Pre-Trained Models](https://arxiv.org/abs/2603.11949) | 使用推理暴露计数与状态逻辑延迟激活；不属于参数学习中晚期跃升 |
| [Sleeper Cell: Injecting Latent Malice Temporal Backdoors into Tool-Using LLMs](https://arxiv.org/abs/2603.03371) | 条件时间/工具行为触发，不能把日历时间当 checkpoint |
| [Delayed Backdoor: Let the Trigger Fly for a While in Backdoor Attack](https://openreview.net/pdf/51eb53a637fa4c44af01e6c6843e7e6e9eefc572.pdf) | 匿名稿的多轮对话延迟输出，不是训练晚学；作者/正式版本不擅自补全 |
| [Phase Transitions in Backdoor Learning: Minimum Data Poisoning Thresholds for LLM Backdoors](https://boazbk.github.io/mltheoryseminar/student_projects/final_papers_and_posters/papers/Phase_Transitions_in_Backdoor_Learning__Minimum_Data_Poisoning_Thresholds_for_LLM_Backdoors_%282%29_-_Kaden_Zheng.pdf) | Kaden Zheng 课程项目，剂量阈值；不当成已评审时间相变结果，不外推“阈值以下安全” |
| [ControlNet training documentation](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md) 的 sudden converge | 工程侧条件通路在训练后出现效果的邻近观察；不是后门机制论证或正式 grokking 论文 |
| [Training-Time Detection of Fixed-Target Backdoor Data Poisons in Stable Diffusion Fine-Tuning via Anchored Denoising Loss Trajectories](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7113514) | 检索出现的近邻线索；本轮未核验到足以支持具体结论的原始内容，不纳入确认结论 |
| BadMoE / SlowBA / memory-poisoning 类 delayed、slow、dormant 命中 | 分别可能是低使用率专家、慢响应、后续记忆检索；关键词相同但时间轴或攻击载体不同，未计为同类证据 |
| clawRxiv / 自动生成研究摘要中的前兆与相变条目 | 未通过足够原始来源核验的条目不纳入核心结论；转载摘要不等于新增论文 |

## 10. 我们当前证据与文献的对应关系

本轮读取了仓库已保存的结果快照，而没有重新训练。路径基于 `output/grokking-evidence-2026-10-04/sources/main/attribution-microscope/supplement/jump/`；数值是该快照的报告结果。

| 我们的证据 | 能支持的判断 | 当前不能推出的判断 |
|---|---|---|
| `r2/RESULTS2.md`：LLaVA 9 seeds，t50 约 326–445，终态 ASR 约 .985–.995；对齐 t50 后短窗口上升形状重复 | 测试设置中的多种子现象重复，发生时间分散 | 所有条件下必现、所有模型都有相同机制；事后对齐不能用来预测未来 |
| `RESULTS.md` / `r2/RESULTS2.md`：部分 LLaVA/Qwen 连续 logit-margin 在跃升窗口也加速 | 对“只是 argmax 阈值假象”的解释形成实质约束 | 完全排除数值、优化器、调度、数据排序；未定种子不能记为证实 |
| `r2/RESULTS2.md`：8-seed blind poison-count 预测失败，固定 step 基线更好 | 简单累计投毒阈值不具当前设置的稳定预测力 | 所有曝光模型都错误；需重新定义/预注册有效曝光量才能继续检验 |
| `r2/RESULTS2.md`：无新 poison batch 的更新包含大量 ASR 增量 | 值得调查历史优化状态、干净梯度与已形成表示的作用 | “干净更新自主放大后门”已经被因果证明；动量仍可携带过去毒梯度 |
| `qx/RESULTS_QX.md`：初始化/顺序影响是否及何时跳 | 成功率与 onset 是两个问题 | 未在预算中跳等于永不跳；旧低剂量失败不能抹掉后来成功设置 |
| 本次用户报告 Qwen LLM 也出现，T2I 正在实验 | 前者作为新增待绑定原始轨迹证据，后者保持进行中 | 本报告没有检查 Qwen LLM 原始轨迹，也没有确认 T2I 成功，不能代替用户实验记录 |

建议暂用 **delayed abrupt backdoor acquisition / ASR takeoff** 描述观测。若要严格叫 backdoor grokking，应另外展示训练中的毒例已经早早拟合，而未见输入上的条件攻击泛化明显更晚，并与单纯答案概率越过阈值区分。定义依据 G02、A03；不是否定我们已观察到的曲线。

## 11. 可以借鉴的机制、最小区分实验和坑

以下是本报告的研究建议，不是已经执行的实验或文献已证明的统一结论。

| 候选解释 | 最小区分实验 | 若结果不符，哪种叙事必须放弃 | 文献起点 |
|---|---|---|---|
| ASR 的评分阈值放大了平滑学习 | 固定评测集同时测 target probability、连续 margin、triggered NLL、ASR；逐样本 crossing 分布 | 连续信号平滑时不能称内部突变；连续加速时再分析原因 | G13、E03 |
| 先学正常语义再形成触发映射，或两条答案路径竞争 | 测 target-class/task competence、触发敏感度及 target-vs-clean logit；冻结/压制单条路径并做恢复 | 仅 attention/归因份额改变不能证明答案竞争机制 | A01、A05、G08–G09、E02 |
| 已见投毒量累积到可学阈值 | 固定干净/毒样本集合，仅重排投毒间距，比较 step、tokens、poison exposure 及优化量的留出预测 | 后验对齐不能叫曝光不变量；我们的简单 poison-count 已失败 | A06、D05 |
| optimizer memory / slingshot / LR milestone | 跃升前保存模型和优化器；配对分叉原状态、清动量、固定 LR，其他条件一致 | 若需特定调度或状态，不可声称纯模型普遍自发变化 | G05、G21、D13 |
| 内部结构渐变、输出竞争后期换主导 | 同一预先确定方向随 checkpoint 测效应；当时移除并恢复，比较 matched-null | 只能解码不代表因果前兆；终态方向前投不等于在线已知方向 | G04、G07、G29、P01 |
| 几何压缩或正则选择高效率电路 | 配对改变 WD / 正则时机，观察等待时间及正常质量；干净臂同样操作 | 效果若只是提升所有任务，不能叫后门特异干预 | G06、G09、G27 |
| 亚稳态噪声逃逸 | 多种子比较 batch/noise 的等待时间分布，固定调度和曝光 | 仅有 seed 差异不能称噪声跨能垒 | G28 对照 G04/G22 |

最需要绕开的坑：

1. **把曲线形状当作新机制。** A01/A02 已有 ASR 跃升，A03 已有 LLM 种子依赖泛化。新颖性必须写在条件规律、因果解释或可操作结果上。
2. **只看成功种子。** 报告全部预先选定种子的跳变发生率、预算内未发生比例、t10/t50/t90、跃升宽度和最终 ASR；失败轨迹是解释机制的必要材料。A03/G26。
3. **把后期输出低当成风险不存在或修复成功。** 分叉继续训练、clean updates、重投毒、触发变化可以给出不同答案；应明确威胁模型。B02/B03/B05/B06。
4. **把干净数据更新误读成“没有任何毒影响”。** optimizer state、先前参数和表示都携带历史；清动量对照可区分部分原因。A06/G05，结合我们快照线索。
5. **早停不测正常质量。** A02 已构造使早停同时伤害正常任务的攻击，A04 在不同毒量下有效性改变。仅少训练几步不是完整防御贡献。
6. **前兆偷看未来。** 终态归一化、知道 t50 后对齐、事后挑信号、事后挑阈值只能做解释性分析。未知 trigger 与已知 trigger 的检测成本也不同。P01/G07/G29。
7. **把纯数值或指标效应排除得过头。** 连续 margin 加速很有价值，但还需核对 LR、precision、logit scale、评测采样和数据顺序。G13/G21。
8. **T2I 的三个时间轴混为一谈。** 训练 checkpoint、投毒剂量、采样 denoising timestep 分别报告；固定 prompt、采样 seeds，并用连续相似度与离散 ASR 一起测。T02–T05。

## 12. 论文定位：哪些说法已有，哪些还值得争取

| 拟写主张 | 查重判断 |
|---|---|
| “首次观察到 ASR 后期快速跃升” | 不支持；A01/A02 直接先例 |
| “首次看到 LLM 后门泛化类似 grokking” | 不支持；A03 |
| “首次发现低 ASR 后门会被干净微调唤醒” | 不支持；B02/B03，且 B01 更早研究潜伏迁移 |
| “首次利用晚学早停” | 不支持；A02/A04 已讨论/使用 |
| “首次提出内部进展早于输出，可用于监控” | 良性任务已有 G04/G07/E02；后门直接协议已有 P01 |
| “统一地测量多种现代 VLM/LLM 的发生率、跃升形状和失败条件” | 可能有价值；本轮未核验到已经覆盖我们具体矩阵的相同研究，但不是全球不存在证明 |
| “用干预区分门控、答案竞争、优化器记忆、曝光量等解释，并形成留出条件下的可检验预测” | 强于现象复述；需真实实验支持，尚未完成 |
| “内部信号在公平敏感度比较下先于输出，并能在线干预降低未来 ASR、保持正常质量” | 值得检验的强主张；P01 明确确认问题仍开放，本轮没有找到足以结案的跨现代模型结果 |
| “文生图与 VLM/LLM 共享同一因果机制” | 目前未证实；跨域同形曲线只是起点，T2I 还在做 |

优先共读顺序：**A01 → A02 → A03 → D06/D03 → A06 → G07/G09/G13 → B02/B03 → P01**。这一顺序先锁定现象先例与反例，再比较机制和应用，不会先被 grokking 理论术语带偏。

## 13. 检索方法、覆盖与可复查性

没有只用 grokking。搜索入口包括通用学术网页检索、arXiv、OpenReview、ICLR/NeurIPS/CVPR/PMLR 官方论文、AAAI/USENIX/ACL 原始出版页、作者机构页与公开文档；引用索引额外尝试 Semantic Scholar 和 OpenAlex。采用“现象词 × 后门/投毒词 × 模型/任务词”及 exact-title 回查；从核心全文 References 向前追溯，再以标题、作者和引用上下文寻找后续工作。

实际使用的检索词族及代表组合（不是声称每个排列都已搜索）：

| 词族 | 覆盖的表达 |
|---|---|
| 现象 | grokking / late generalization / delayed learning / sudden increase / spike / abrupt learning / phase transition / takeoff / training dynamics / hidden progress / plateau / late-stage failure |
| 安全 | poisoning / backdoor / Trojan / ASR / attack success / slow poisoning / early stopping / backdoor leakage / dormant / hibernated / latent / re-activation / camouflage / unlearning |
| 模型 | LLM / VLM / vision-language / contrastive / semi-supervised / FixMatch / federated / diffusion / text-to-image |
| 方法 | circuit efficiency / gradient / optimizer / slingshot / representation / information bottleneck / threshold / poison count / seed / internal precursor / continuous margin |
| 代表查询 | `"ASR" "sudden increase" "training" backdoor`；`"backdoor" "remain low" "training"`；`"backdoor" "delayed learning"`；`"backdoor" "late stage" training attack success rate`；`"grokking" "poisoned" backdoor`；`"BadDiffusion" "training" "steps" attack success`；`"Testing Whether Internal Backdoor Precursors Precede Behavior"` |

去重包括：preprint–conference、改名、相同基准扩展、作者博客/自动摘要重复转载。对检索中的“时间”逐项查轴，未把剂量、采样步、对话轮次和日期混在 checkpoint 内。未确认的条目保留处置表，不给虚构结论。

**覆盖限制**：这是完成的一轮广泛现象查重与定向引用链审查，不是可以证明“全球所有论文已找全”的系统计数。没有全量导出 Google Scholar/Scopus cited-by；索引会漏收、近期条目会延迟，部分来源只有摘要。没有伪造 PRISMA hit count 或饱和证明。核心先例已读原文、关键图已核验；扩展表明确标注摘要级核验。不能以本轮“没有找到”支持绝对 first claim。

**审阅结论的可靠范围**：晚学/骤升、安全训练后低 ASR 仍有风险、早停有效性有条件，均有原始文献支持；我们具体多模型是否共享因果机制、在线预警是否成立、T2I 是否同样成功，仍须新实验。

本报告由 AI 辅助检索、来源核验与综合，未重跑文献实验。核心 PDF、文本及部分引用索引响应保存在 `sources/`；所有正文链接指向原始论文/出版/作者来源。

## 14. LoRA与生成式设置核验新增近邻（2026-10-05）

| ID / 论文 | 结论与核验范围 |
|---|---|
| N01 [Moderate-fitting as a Natural Backdoor Defender for Pre-trained Language Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/0799492e7be38b66d10ead5e8809616d-Paper-Conference.pdf) | NeurIPS 2022；预训练RoBERTa适配先学正常任务后学后门；full FT、普通PET与全局低秩重参数化，限制容量/epoch/LR防御。F，Fig.5相关页已视觉核验。 |
| N02 [Watch Out for the Lifespan: Evaluating Backdoor Attacks Against Federated Model Adaptation](https://arxiv.org/html/2511.14406v2) | FPS 2025 / 2026 v2；预训练ViT的联邦适配，LoRA rank影响形成速度；充分植入后低rank可更持久，攻击窗口与终态ASR混淆比较。F，§3–5。 |
| N03 [Defending Against Weight-Poisoning Backdoor Attacks for Parameter-Efficient Fine-Tuning](https://aclanthology.org/2024.findings-naacl.217/) | NAACL Findings 2024；文本分类中的毒底座在PEFT后可保留后门，以置信度构造PSIM防御；不是数据投毒中新后门首次形成。A。 |
| N04 [Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?](https://proceedings.mlr.press/v267/liang25f.html) | ICML 2025；BERT-large/GLUE、LoRA/full FT、rank/初始化理论与实验；所测数据后门中LoRA更鲁棒，无目标数据投毒更脆弱。F，§4与理论相关段落。 |
| N05 [Why LoRA Fails to Forget: Regularized Low-Rank Adaptation Against Backdoors in Language Models](https://arxiv.org/html/2601.06305v1) | 2026预印本；毒底座的干净适配，谱强度与对齐影响后门遗忘，提出RoRA；并非首次形成机制。F，§6与超参附录。 |
| N06 [LoRATK: LoRA Once, Backdoor Everywhere in the Share-and-Play Ecosystem](https://arxiv.org/abs/2403.00108) | 2024初稿，版本改名；恶意LoRA分发、组合与迁移，初稿LoRA-as-an-Attack含无训练注入；非checkpoint晚学研究。F，初稿相关方法；最新标题元数据回查。 |
| N07 [Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_Revisiting_Backdoor_Attacks_against_Large_Vision-Language_Models_from_Domain_Shift_CVPR_2025_paper.pdf) | CVPR 2025；OpenFlamingo/BLIP-2/Otter，域迁移ASR-G及归因构造MABA；dynamic指数据域而非训练checkpoint，主文未确认具体更新模块。F。 |
| N08 [BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models](https://papers.neurips.cc/paper_files/paper/2025/hash/20ffc2b42c7de4a1960cfdadf305bbe2-Abstract-Datasets_and_Benchmarks_Track.html) | NeurIPS 2025；生成式LLM多攻击多架构基准，作者数据投毒实现使用LoRA；不是已确认的late-grokking时间机制研究。A+作者仓库。 |

| N09 [Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models](https://www.ndss-symposium.org/ndss-paper/causal-guided-detoxify-backdoor-attack-of-open-weight-lora-models/) | NDSS 2026；无原始训练数据、因果引导混合干净与恶意LoRA，控制攻击强度和误触发；非形成时间研究。A。 |
| N10 [VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b055b95d689b1f704d8f92191cdb788-Abstract-Conference.html) | NeurIPS 2023；统一条件/无条件扩散后门；作者T2I实现明确支持SD1.4 LoRA rank 4与文本触发器，不是本轮确认的自然晚期骤升结果。A+作者实现。 |

补核验引用边：N01 Moderate-fitting → N02 Lifespan，N02 §2/§4及References明确引用；N03 PEFT Defense → N05 Why LoRA Fails to Forget，N05实验超参明确沿用。原报告没有这些条目不应被解读为不存在。
