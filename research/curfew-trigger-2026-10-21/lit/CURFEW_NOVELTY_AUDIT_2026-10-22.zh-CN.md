# Curfew 查重报告（2026-10-22）

本报告只记录文献对 Curfew 各条主张的覆盖情况，不对我们的实验结果作 novelty 论证。我们的实验目前只有一块新知识地块、两个 seed、两个断供点，任何"我们新增了什么"的说法都要等实验完善之后再写。

## 0. 依据与阅读量

| 来源 | 时间 | 覆盖对象 | 阅读量 |
|---|---|---|---|
| 本次七步查重（第一幕、第三幕） | 10-22 | 可逆/锁定、防锁定防御 | WebSearch 22 次，Semantic Scholar 引用 API 4 次，约 60 条摘要，约 200 个被引标题；**14 篇取到全文，其中只有 2 篇逐字读原文**（Moderate-fitting NeurIPS'22、2206.04677），其余 12 篇是经摘要工具压缩后定向追问的弱阅读 |
| `lit/LIT_TRIGGER_SWEEP.zh-CN.md`（main） | 10-21 | 扳机、行级检测 | 四轴，约 40 篇正文，其余摘要级；每个候选只检验过一轮 |
| `research/RELATED_WORK_AUDIT_2026-10-16.md`（分支 keen-hypatia） | 10-16 | 旧版 Curfew（在线 loss 过滤） | 约 200 次检索，约 70 篇全文 |
| `research/asr-mechanism-positioning-2026-10-15/RELATED_WORK_FORMATION.md`（分支 mechanism-line） | 10-14 | 形成期背景 | 绝大多数只读了摘要 |
| `research/late-grokking-related-work-2026-10-04/`（main） | 10-04 | ASR 跃升 | 89 条，分级标注 |

**此前没有任何一份文档专门查过第一幕（可逆→锁定）。**

## 1. 第一幕：形成后可逆、之后锁定

| 子主张 | 文献覆盖 | 关键证据 |
|---|---|---|
| 1a 形成后在干净数据上可被冲掉 | **已覆盖** | Souly 2510.07192 F.2："clean data fine-tuning degrades ASR to near-zero after 100k datapoints (32k steps)… regardless of the number of poisoned samples seen"；P-Trojan 2512.14741：朴素触发词干净续训后持久性 0–5%；2206.04677："fine-tuning based model updates naturally degrade one-shot backdoors over time"；2610.07510：SFT 侵蚀后门；Bagdasaryan'20（联邦学习） |
| 1b 后门会持续存在、不被冲掉 | **已覆盖，且与 1a 有冲突** | Souly C.3（预训练）图 10(b) 标题 "does not remove the backdoor"，图注却写 "decays approximately logarithmically"；F.2 又写 "does not completely remove a backdoor learned with 20 samples"；Sleeper Agents 2401.05566；late-grokking 审计 B12："干净续训可能减弱，也可能激活，取决于植入方式" |
| 1c 注毒越多或越久越难洗 | **已覆盖（粗粒度或连续变量）** | 2206.04677 C.2："poisoning more model updates will increase the backdoor survivability, but there is no guarantee"（只比较 2 档和 5 档）；Neurotoxin 2206.10341：lifespan 随攻击轮数渐变上升；2610.07510：强度 S = −log(L_bd+ε)，"the weakest variants retain only 23% ASR after SFT, whereas the strongest retain up to 95%"（未在注毒过程中取检查点，未说明渐变还是阈值） |
| 1d 同一次注毒训练内、相对形成时刻的尖锐边界 | **未找到** | 最接近的是 Achille 1711.08856 的关键学习期："If the blur is not removed within the first 40-60 epochs, the final performance is severely decreased"，但那是训练开头的全局窗口，作者称为 "smooth transition"，不针对单条关联 |
| 1e 锁定与泛化到未训练输入同时发生 | **未找到直接覆盖** | 只有类比：Chang 2406.11813 发现组合泛化的遗忘更慢；2206.04677 "backdoors injected via full training embed the hidden features more broadly… harder to remove" |
| 1f 任务依赖（真实数据会退、合成任务不退） | **未找到具体覆盖** | 只有泛泛说法：2206.04677 "the trend varies"；Souly "not even 1-dimensional" |
| 1g 方法：从检查点分叉、在时刻 t 断供 | **未在 LLM 后门中找到** | 2608.19168 做过孪生分叉的单行注入，只测了两个时间点；2206.04677 C.2 做过"注毒若干次更新后停止" |
| 1h "形成前先学原任务、后学后门"的两阶段 | **已覆盖** | 10-16 审计："不能写成'首次发现两阶段'" |

## 2. 第二幕：形成期在行级与行为级不可见

| 子主张 | 文献覆盖 | 关键证据 |
|---|---|---|
| 2a 分散目标让基于相似性的检测失效 | **已覆盖** | Qi et al. ICLR'23（2205.13613）：四种表示空间检测器在自适应分散下清除率从 98–100% 降到 0–13%；机制见 BackdoorBench 2407.19845 |
| 2b 输入级检测器（STRIP 等）不适用于训练期 | **已知** | 结构上要求被检输入含触发词 |
| 2c 触发词反推和元分类器在低投毒率下失效 | **已覆盖** | 2409.00399：低投毒率、低学习率下准确率 17–22% |
| 2d 新知识微调中 ABL、早停、loss 截断的前提反转 | **未找到处理，但只检索过一轮** | LIT_TRIGGER_SWEEP 一句话："文献里完全没有处理过"，没有逐篇论证；10-16 审计列了 Loss Truncation、Moderate-fitting、DeCE、Honeypot 等，都假设毒样本比干净样本难学或晚学 |
| 2e 模型对 1% 投毒后门的自我报告 | **正向结果已有，负向未见** | Betley ICLR'25 在全参微调、行为占主导的设置下报告了行为自知；1% 投毒、LoRA 下无人测过 |
| 2f 训练中的单标量探针在确认阶段误报 | **已有** | 2609.19000：无标签谱探针在确认阶段误报而失败，需"机制 ∧ 行为"双信号 |

## 3. 第三幕：防御

| 子主张 | 文献覆盖 | 关键证据 |
|---|---|---|
| 3a 不知道触发词的训练期在线报警 | **未找到已发表方法** | BackdoorLLM（NeurIPS'25 D&B）和两篇 2026 综述都列为开放方向；2606.07631 唯一在训练中跑的行为监控不覆盖后门 |
| 3b 用可信干净数据收尾去后门 | **已覆盖** | 2206.04677 STLR："backdoor survivability stays 0"；Sha 2212.09067 super-fine-tuning（只读摘要）；2406.07778 下游干净微调在 SST2 上 99.23→52.85，去不干净；Fine-pruning、NAD、FT-SAM（凭已知） |
| 3c 靠时机、不检测毒行的训练期防御 | **已覆盖（形成前）** | Moderate-fitting（NeurIPS'22）："a moderate-fitting stage… and an overfitting stage"，靠减 epoch、学习率、容量阻止后门形成；自承 "merely decreasing the training epochs or the learning rate… is not enough"；2607.06643 用吸收寿命 Markov 模型加抽查（抽查属检测） |
| 3d 训练中阻止"锁定"、再收尾 | **未找到** | 抗固化正则用于防御：LLM 上零命中；最接近的是 NeurIPS'24 神经元重初始化加梯度范数抑制（事后防御，只读摘要） |
| 3e 诱饵后门做传感器 | **已有（联邦视觉）** | BackdoorIndicator、Coward；自适应擦除下 TPR 53% |
| 3f 检查点回滚 | **常识** | — |
| 3g 持久型触发词可能关掉窗口 | **已有威胁** | P-Trojan 2512.14741：梯度对齐触发词持久性 >99% |

## 4. 名称口径

git 里的"Curfew"原指 10-16 版"宵禁时刻开启 loss 过滤"；10-17 团队共识："Curfew 时机独特性未立住"，"阳性不算新贡献（Souly §4.2/F.2 有先例）"。本报告里的第三幕是另一种设计，与旧 Curfew 同名不同物。

## 5. 未覆盖范围

在审稿件、OpenReview 未公开稿、S&P/CCS/USENIX 2026 全文、中文文献、并行未公开工作；2026-06 之后的 arXiv 只能靠检索，覆盖不完整；部分引用链因接口限流没挖完；Sha 2212.09067、FT-SAM、SEAM、ABL 只读了摘要或凭已知；2206.04677 的图 11/12 没看到数值；持续学习中的任务边界检测文献未覆盖。

## 6. 参考

2510.07192 · 2512.14741 · 2610.07510 · 2206.04677 · 2206.10341 · 1711.08856 · 2406.11813 · 2607.06643 · 2608.11295 · 2608.19168 · 2406.07778 · 2604.16423 · 2609.36117 · 2212.09067 · 2205.13613 · 2407.19845 · 2409.00399 · 2401.05566 · 2606.07631 · 2609.19000 · Betley ICLR'25 · BackdoorLLM NeurIPS'25 · Moderate-fitting NeurIPS'22（proceedings.neurips.cc/paper_files/paper/2022/file/0799492e7be38b66d10ead5e8809616d-Paper-Conference.pdf）
