# Generation 域协助全文证据卡

本文件只保存视觉域代读卡；记录由generation/PAPERS.json计数，visual/PAPERS.json不重复。

## Context Influence — Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models

- **来源/状态**：ACL2025正式17页25092–25108，https://aclanthology.org/2025.acl-long.1220/ 及对应PDF。generation/pdfs/context_influence.pdf；visual/texts/context_influence.txt。正文、全部参考文献、附录A–D完整阅读；原图Tables1–4、Fig2–8、AppA/B证明均已核（visual/figures/context_influence-p2,p6,p7,p8,p9,p14–17）。
- **研究问题**：输出与上下文文字重合可能也由参数知识造成，想估计上下文对输出的额外影响。不是交互UI/用户研究，属于解释量及可视分析先例。
- **定义**：固定原本采样生成的历史y<t和目标yt，删除一个context n-gram，τ=|log p(yt|D,x,y<t)−log p(yt|D\Di,n,x,y<t)|。全文把逐token绝对差相加为response score并对上下文样本取平均。它是特定删除对照下的目标token敏感度；无正负方向、无Shapley互动分解，也未把模型知识和冗余context完全辨别。
- **已有可视认知**：Table1 UFO例展示context unigram→固定nexttoken热图；“Japan”也高而flying/object不高。§4.3/Fig2c OPT1.3B在最初约10个生成token对完整context依赖较高，之后减弱；§4.4/Fig3–4删除n-gram的输入位置曲线，早段整体高但小模型后部有尖峰。故“画输入贡献沿生成位置变化”不是新问题；可以复用其固定history对照，但下降不能单凭曲线解释为trigger已写入内部状态，也可能history冗余/长度选择/内容位置。
- **实验**：CNN-DM摘要与PubMedQA长QA，1000上下文每任务，n-gram分析100；最长50tokens,T=.8、temperature sampling，context分别截1024/2048。OPT1.3B,GPTNeo1.3B,Llama3-8B/base+IT,Gemma2-9B-IT；另OPT125M–66B容量比较。质量用ROUGE/BERTScore/FactKB，主要操纵CID上下文logit权重λ=.5/1/1.5及温度。A10040GB主设，30B/66B用2/3张；普通OPT1.3B实验约15min且细粒度较慢，非端到端基准。
- **主要结果边界**：更多λ通常提高context score与复述，质量并非单调；Figure2a容量趋势非单调。GPTNeo vs OPT两1.3B的PubMed预训练包含不同，score与复述指标排序不同；这支持“复述≠唯一context影响”，但不同模型/训练整体未匹配，不能把差异唯一归因PubMed memorization。无context仍输出答案也可能来自剩余prompt/冗余信息/推断；不证明只由某训练文档贡献。Paper对温度/entropy影响自己有限制，不能跨模型把低τ解释成更安全。
- **数学审查（非运行结果）**：
  1. 单个固定输出的绝对log概率比可以给同一邻接对最坏隐私损失下界，但`sum_t abs(Δlogp_t)`只是`abs(sum_t Δlogp_t)`的三角不等式上界，不能直接当整段实现的精确隐私损失/普遍隐私保证。需严格区分各量。
  2. Theorem3.1/AppA Eq10–12把未归一化`pλ ∝ pprior exp(λ PMI)`直接移入log比，忽略两个context各自的归一化常数。T=1时精确差还有`−log Z_D + log Z_D'`，T≠1另有温度缩放；因此τ随λ与PMI差“严格正比”的论证不成立。原图确认并非提取错误。
  3. Appendix B Algorithm1第2行要求的比率印为原pθ而非第3行的调节后p̄θ，λ未出现在该行条件，且仅当前yt形式未清晰规定对全部词表同时界定。后续三角证明在“所有D、所有yt的调节后分布到同一prior均≤ε/2”前提下可成立；印刷算法本身没有完整建立这个前提，不能照抄成可执行DP保证。
  4. Definition3.1的Di,n下标似非重叠块，但Fig3/4中128-gram横轴到400与2048token上限不匹配；步长/起点定义需代码核查，此轮没跑/读代码不猜修正。
- **数值一致性**：Table2 GPTNeo CNN-DM λ1/1.5 τ=85.23/140.0，Table3却77.87/130.47，原图均核；前者与OPT相同，不能任选做精确比较。Table4“flying vehicles”仍是context UFO的语义转述，作者称强参数依赖不充分。
- **可复用**：固定target/history的对数概率差、输入ngram与生成位置两个坐标、context-removal空白对照、温度敏感性分析；将分数解释为干预敏感度会比直接继承其privacy定理可靠。
- **未答/本项目差异**：没有clean/ordinary-backdoor/OA匹配、trigger与sham、训练漂移控制、未知触发盲测、内部机制/迁移后验证。问题重心也不是可视仪器是否产生新发现。若本项目发现依赖从触发词转到prefix，需要与该已有位置下降先例明确区分。
- **引用链范围**：ContextCite/Inseq/CTI及生成归因由generation域；一般DP、RAG隐私/记忆攻击是外围，本次不向全部隐私文献扩展。
