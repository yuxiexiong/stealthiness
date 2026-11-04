# 两稿所需的主要外部文献核验

核验日期：2026-11-05。以下为写作所需的定向原始来源核验，不是截至当日的穷尽查重。正文只引用与相应主张相关的条目，不以引用数证明新颖性。

| 文献 | 原始来源 | 本轮核验支持的使用方式 |
|---|---|---|
| Connor & Emanuele (2022), Rethinking Backdoor Data Poisoning Attacks in the Context of Semi-Supervised Learning | https://arxiv.org/abs/2212.02582 | 原 PDF §4.2、Fig.4 明确报告 FixMatch 训练中低 ASR 后快速升高；因此不能声称首次发现 ASR 急升。其伪标签反馈条件与本文监督 LoRA 不同。 |
| Zhu et al. (2022), Moderate-fitting as a Natural Backdoor Defender for Pre-trained Language Models | https://proceedings.neurips.cc/paper_files/paper/2022/hash/0799492e7be38b66d10ead5e8809616d-Abstract-Conference.html | 已有正常任务/后门学习阶段及限制容量、轮次、学习率的防御思想；不是本研究首次提出两阶段或早停。 |
| Kang & Hashimoto (2020), Improved Natural Language Generation via Loss Truncation | https://aclanthology.org/2020.acl-main.66/ | 原 PDF §4.1 和 Algorithm 1：hotstart、运行分位数、按样本置零损失；本地移植须说明环形池及归一化差异。 |
| Wan, Wallace, Shen & Klein (2023), Poisoning Language Models During Instruction Tuning | https://proceedings.mlr.press/v202/wan23b.html | 指令微调投毒、高损失过滤与降低模型容量的效用—风险权衡。不能把所有本地适配当官方完整复现。 |
| Liu, Dolan-Gavitt & Garg (2018), Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks | https://arxiv.org/abs/1805.12185 | 剪枝联合微调是已有修复思路，当前模块消融本身不是新型防御已经成立。 |
| Hu et al. (2021), LoRA: Low-Rank Adaptation of Large Language Models | https://arxiv.org/abs/2106.09685 | 冻结预训练权重并训练低秩分解，作为训练参数化背景；引用预印本年份，不混淆后续会议年份。 |
| Pruthi, Liu, Sundararajan & Kale (2020), Estimating Training Data Influence by Tracing Gradient Descent | https://arxiv.org/abs/2002.08484 | TracIn 利用训练轨迹、检查点和一阶梯度近似估计样本影响；逐行路径分解思想有明确前作。 |
| Xia, Malladi, Gururangan, Arora & Chen (2024), LESS: Selecting Influential Data for Targeted Instruction Tuning | https://arxiv.org/abs/2402.04333 | 优化器感知、Adam 与变长指令数据归因是已有方法方向；不能将本研究账本校准单独作为首创。 |
| Turner et al. (2025), Model Organisms for Emergent Misalignment | https://arxiv.org/abs/2506.11613 | LoRA、有害微调和行为转变已有研究；广泛失对齐与固定触发 ASR 不同。 |
| Arnold & Lörch (2025), Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment | https://arxiv.org/abs/2508.20015 | 用输出分布与行为量刻画训练转变；梯度峰与行为时刻不必相同。 |
| Lelle (2026), Token-Level Generalization in LoRA Adapter Backdoors: Attack Characterization and Behavioral Detection | https://arxiv.org/abs/2605.30189 | LoRA 后门的跨条件分析、行为检测和因果模块定位已有近邻；本文不能泛称首次定位 LoRA 后门。 |
| Lasnier et al. (2026), Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models | https://arxiv.org/abs/2602.10382v3 | 预训练语言切换后门的激活替换及正常功能重叠；现有定位不能自动推出独立后门电路。 |
| Maini, Feng, Schwarzschild, Lipton & Kolter (2024), TOFU: A Task of Fictitious Unlearning for LLMs | https://arxiv.org/abs/2401.06121 | 200 个虚构人物资料及问答的数据来源；本研究使用其材料测新知识学习，不声称复现 TOFU 遗忘任务。 |
| Rajpurkar, Zhang, Lopyrev & Liang (2016), SQuAD: 100,000+ Questions for Machine Comprehension of Text | https://aclanthology.org/D16-1264/ | 正常阅读问答数据来源；子集和重排不等于官方完整基准成绩。 |
| Liu, Li, Wu & Lee (2023), Visual Instruction Tuning | https://arxiv.org/abs/2304.08485 | LLaVA 模型背景；实际实验配置以本地冻结模型身份为准。 |

未将无法核验的引用、社交媒体截图、旧“未搜到同类”判断当作已确立的新颖性证据。两篇稿中的研究数值来自本地实验文件，不来自这些文献。
