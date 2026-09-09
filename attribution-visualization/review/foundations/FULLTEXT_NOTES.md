# 基础方法与评测：全文证据卡

核读日期：2026-09-09。页码为 PDF 页码；会议页码另列。这里区分原文证据、我们的判断与未解决问题。未运行论文代码或模型实验。

## F03 · Sanity Checks for Saliency Maps

Adebayo et al.，NeurIPS 2018；核读 [arXiv v3](https://arxiv.org/abs/1810.03292v3) 全部 30 页（包含全部附录），该版本 2020 年修正了实验。原图复核 `figures/F03_p5.png`、`F03_p9.png`。

- **已做**：参数逐层随机化和标签随机化检查解释是否依赖所解释的模型/任务。§3、Fig.2 显示 Guided Backprop/Guided Grad-CAM 对高层随机化不敏感，但低层变化后会变化；旧版“全部不敏感”已被作者撤回。Grad-CAM 与普通梯度的结果不同，不能把所有方法合并。
- **可借用**：模型随机化、数据随机化、有符号排序、视觉外观与数值差异并列。IG 的输入结构可保留视觉轮廓，但符号排序已变化；取绝对值会遮掉这种变化。
- **边界**：通过随机化是必要的诊断之一，不是后门可靠性/因果充分性证明。任务是图像分类，文本需要 F09 的补证。
- **原文疑点**：§5.3 PDF p9 将线性函数的 IG 写成 `(x−baseline)⊙w/2`，原图已核。按 F01 的标准积分，线性梯度恒为 w，应无 `/2`；该公式问题与全文随机化的经验结果分开记录。判定：`reuse/counterevidence`。

## F04 · Fooling Neural Network Interpretations via Adversarial Model Manipulation

Heo et al.，NeurIPS 2019；[arXiv v3](https://arxiv.org/abs/1902.02041v3) 全部 18 页、全部附录已读。复核 App.D Table 5 原图 `figures/F04_p15.png`。

- **已做**：在图像/表格模型中改变模型参数，使多种解释改变而总体分类表现近似保留；解释操控、类别选择性与迁移已经有先例。这是 OA 之前的相关思想，不是现代 LLM 后门研究。
- **可借用**：把输出保真和解释变化分别量化；针对方法、架构和类别报告失败，不能只挑热图。Grad-CAM 能通过 F03 的随机化测试，仍不自动抵抗这种操控。
- **边界**：DenseNet 的 active manipulation 有接近失败的配置；SmoothGrad 有预测性能代价。不能从某项失败断言整个方法免疫。AOPC 对真实预测功能变化的核验只覆盖有限设定。
- **原文疑点**：App.D Table 5 VGG 的 firetruck 在 LRP 配置下 94%→78%（该表每类 50 张），与“各类近似保持”的宽泛图注有张力。整体准确率不能代替类别/条件能力检查。没有据此否定其他表中总体结果。判定：`done/reuse/counterevidence`。

## F06 · Attention is not not Explanation

Wiegreffe & Pinter，EMNLP-IJCNLP 2019；[正式版](https://aclanthology.org/D19-1002/) 10 页全读，并补读 [arXiv](https://arxiv.org/abs/1908.04626) p11–12 附录。复核附录 Table 5：`figures/F06_p12.png`。

- **已做**：用均匀注意力、不同随机种子、诊断性 MLP 和替代注意力实验，重新分析 F05 的结论。不同热图能否保持预测，取决于可实现性与解释定义，不能从不唯一性直接推出完全无用。
- **可借用**：成对测行为与归因；给注意力图加合理的随机/均匀基线；在明确定义下测试必要性。本文没有证明 attention 无条件等价因果贡献。
- **边界**：主要是分类模型而非现代自回归 LLM；本文与 F05 共同构成“可以探索，但需验证”的依据。
- **原文疑点**：arXiv 附录 Table 5 图注明确写 SST 的最佳 epoch 按 test loss 选；这带来测试选择偏差风险，不能照搬为本项目的验证流程。图注已核，尚未运行作者代码确认实际执行。判定：`reuse/counterevidence`。

## F08 · ERASER: A Benchmark to Evaluate Rationalized NLP Models

DeYoung et al.，ACL 2020；[正式版](https://aclanthology.org/2020.acl-main.408/) 全部 16 页、附录全部已读。PDF 的隐藏 LaTeX 元数据曾污染文本抽取，p6–7 已用 raw 模式补读；Table 3–4 原图核验 `figures/F08_p8.png`。

- **已做**：七类数据上区分与人类 rationale 的一致性、删除解释后的 comprehensiveness、只保留解释的 sufficiency；对 soft attribution 使用多个删除比例的 AOPC，并有多次随机基线。
- **可借用**：多个角度评价，避免“热图像人想的”直接成为忠实性真值。Table 4 中 attention 的人类对齐和 LIME 的扰动表现并不一致，没有一个指标统一代表解释质量。
- **边界**：删除/保留会改变输入分布，人类标注可能不穷尽充分证据；大多是分类分数，并非生成位置的可识别因果效应。文中 hard rationale 的“by construction”解释还依赖编码器/解码器独立等条件。
- **本项目用途**：借评测分层和随机对照，不直接把 benchmark 分数搬成 OA 诊断有效性。保留探索集与确认集分离；原文多数 source-document split 不重叠但 BoolQ 有例外。判定：`reuse`。

## F10 · Towards Best Practices of Activation Patching in Language Models

Zhang & Nanda，ICLR 2024；[arXiv v2](https://arxiv.org/abs/2309.16042v2)，全部 27 页及附录全读。核心 Fig.4 原图核验 `figures/F10_p6.png`。

- **已做**：在事实回忆、IOI 等任务中系统改变 corruption、评价标量和 patching 窗口；同一模型的 token/层定位会明显改变。概率可能漏掉负向组件；logit difference 也依赖 foil。
- **可借用**：自然替代与噪声替代交叉检验，给出目标分数，联合/单点干预分开；对照选择敏感性必须报告。Appendix F 用不同被替换 token 找回正文设定遗漏的 Name Mover 作用。
- **边界**：自然替代让输入可在分布内，不保证混合后的内部激活在分布内；窗口效应不能全部归到中央层。归一化恢复率的分母接近零会不稳。主要模型不超过 6B，不代表 OA/所有 LLM。
- **原文疑点**：正文某处将附录定义的 probability 称 accuracy；附录 Fig.25 窗口标题/图注、Fig.26 GN/STR 图注存在不一致。应复核代码/原数据后才引用对应精确分组，不以排版疑点替代主体经验结论。判定：`reuse/counterevidence`。

## F13 · Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small

Wang et al.；核读 [arXiv v1](https://arxiv.org/abs/2211.00593v1) 全部 25 页、附录全读。正式 [ICLR 2023 页面](https://openreview.net/forum?id=NpsVSN6o4ul) 可检索，但其 PDF 返回 403；不能把 arXiv 当成已读最终版。原图 Fig.6：`figures/F13_p10.png`。

- **已做**：从可视化/路径假设进入 patching 与反事实检查，分析名字、位置、重复与抑制；冗余 backup 和负向组件都已有先例。验证区分 faithfulness、completeness、minimality。
- **可借用**：单点消融没变化不代表无作用；组合干预、补偿路径与输出层面恢复应同时考虑。发现集和验证集的反事实设计有启发性。
- **边界**：87% 是 logit-difference 保留率，不是任务准确率。贪心寻找的删除组合仍暴露出最多为原差值 87% 的不完整性；“完整电路”不是由案例图自动成立。模型为 GPT-2 Small，有限模板和单 token 名字；关注 attention 不意味着 MLP 无作用。
- **版本疑点**：已读 arXiv 摘要/正文是 26 heads，正式摘要是 28 heads；最终版具体改动未核清，报告只引用 v1 的数值，不做版本等价断言。判定：`done/reuse/version_gap`。

## F14 · CLT-Forge

2026；[arXiv v2](https://arxiv.org/abs/2603.21014v2)，2026-07-21，全部 9 页、附录 A–D 已读。配置/曲线原图核验 `figures/F14_p9.png`。

- **已做**：CLT 训练缓存、分片、自动解释、归因图与交互干预界面串成工具链。故“搭建特征归因图端到端平台”本身已有直接工程先例。
- **可借用**：可作为将来需要内部特征路径时的可选重后端；不应在立项阶段强制为所有 OA 模型先训字典。附录给 Llama-3.2-1B 的 307M tokens、8×H100、17h35m，而不是零成本可视化。
- **边界**：解释替代模型有重建误差，注意力等冻结带来遗漏；不同训练阶段与稀疏度有权衡。字典、训练语料、干预强度都应记录。
- **原文疑点**：§2 解码求和允许本层，Appendix B 参数计数用严格后续层，需查实现才能预算；3.6× 压缩与约 20TB→4TB 的近似数字口径需区分；压缩增加重建误差，不能把加速写成免费收益。这些未由本轮复现解决。判定：`done/reuse_with_caution`。

## F15 · Information Flow Routes: Automatically Interpreting Language Models at Scale

Ferrando & Voita，ICML 2024；[arXiv v2](https://arxiv.org/abs/2403.00824v2) 全部 16 页、附录 A–D 已读；Fig.6–7 原图核验 `figures/F15_p5.png`。

- **已做**：利用 ALTI 的非负、归一化向量接近程度组成输入/组件路线，设阈值剪枝并跨样本统计；用 IOI、greater-than 对照、层头频率、领域聚类与可视异常解释模型。句号呈 BOS 式作用等观察说明“可视异常→待验证认识”并非新研究范式。
- **可借用**：单次 forward 后的轻量结构浏览、对照模板、全局/领域频率与个例联动。§3.2 表明换对照模板就会改变被发现的头，需保存比较契约。
- **边界**：ALTI 路线权重不是有符号目标 logit 的因果贡献；阈值频率不等于干预效应。冻结注意力/归一化及非负分配可能遗漏抑制作用。聚类/投影本身不证明生成行为的因果机制。
- **成本界限**：§3.4 约 100× 来自 IOI、50 例单个比较（ACDC 8 分钟 vs 本法 5 秒），不能泛化成所有机制分析的百倍加速。判定：`done/reuse`。

## F17 · On the Biology of a Large Language Model

Anthropic，2025-03-27；[官方完整文章](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)。官方 HTML 正文、全部附录、图注与脚注共约 2.76 万词已逐段读；原始 HTML 与抽取文本保留在 `texts/F17.html`、`texts/F17.txt`。已在浏览器核验 §4.1 Fig.11 的诗歌规划归因图。交互图库全部节点/悬浮示例未逐个遍历，不能声称等价于原始特征数据审计。

- **最直接的立项先例**：§12.4 明确说图在交互查看、聚合和跟踪后才变得可用；研究通过自下而上探索发现意外，再用干预测试。多跳、诗歌规划、多语言、算术、隐藏目标等案例，不要求每个案例都先发明一种归因算法。
- **已做**：按输出 token 查看相关输入和内部特征；从押韵输出回溯前面的规划特征，再干预观察输出变化；隐藏 RM 目标模型案例；拒绝、CoT 不忠实与对抗输入案例。故“模型不主动承认的目标可用归因发现”“前序生成向后传递影响”都不是足够的新颖性描述。这里的隐藏目标模型不是 OA。
- **可借用**：保持归因可视化核心，记录探索→猜想→另行干预的链条；正负边、错误节点、特征多重含义、supernode 聚合、真实输出变化一起查看。能产生有用认识与穷尽模型机制是不同标准。
- **关键边界**：作者在限制部分估计约四分之一尝试的 prompt 能产生满意认识，这是主观且筛选后的经验，不是统一任务成功率。主要图局限短 prompt 和单关键 token，冻结 attention、字典缺失、不活跃特征、重建误差均会漏掉作用。未找到解释不能推出该机制不存在。
- **验证疑点**：脚注说明干预时有些量被按构造固定，因此该部分一致性不能算独立验证；应重点看未被固定的下游和实际输出。少数抑制需很大强度，可能离自然分布。Appendix F 图的纵轴是最大路径长度，不是模型层深，不能看图误读成“第几层出现”。
- **本项目与之区别待验证**：聚焦清洁/普通后门/OA 中触发影响的可观测性、三类归因视图的一致/冲突及发现复验率；不能以通用交互界面或几个成功案例代替新贡献。判定：`done/reuse/direct_neighbor`。

## F16 · Circuit Tracing: Revealing Computational Graphs in Language Models

Ameisen et al.，2025-03-27；[官方方法论文](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)。官方 HTML 正文、全部附录、脚注、伪代码和 prompt 表约 2.95 万词已全读；`texts/F16.html` 保存原文，`texts/F16.txt` 为抽取文本。在浏览器核验 §3 Fig.5 的输入→supernode→目标 token 原图。没有逐个审计几千个链接特征例或执行代码。

- **已做**：用 CLT 替代 MLP、固定 attention pattern 与 normalization denominator、加重建误差节点，构建某个 prompt 的输出 token 归因图；交互溯源、手动 supernode、全局权重、成对图差异和干预。F17 是其认知发现伴随论文。三种图层和探索发现流程都可借鉴，不能把通用框架占为新颖性。
- **分数契约**：边是局部替代模型中的直接线性效应；实际输出节点用 `logit(token)−mean(logit)`。剪枝用绝对值/行归一化后的多路径 influence，不是原模型有符号总因果效应。固定原 prompt 时重建误差使输出一致是按构造成立，不能据此说机制一致。
- **验证与限制**：主文和附录同时报告节点影响对干预的预测性及机制误差随层累积；特征间 0.72 Spearman 来自 20 个 prompt、特定 constrained patching 与取绝对值，非任意模型/提示的忠实率。部分直接边的干预一致性按线性构造保证；未固定的多跳后果才是实质验证。最有效层与强度常经扫描选择，不能把探索选择后的效应当独立确认。
- **不能忽略的失败**：冻结 QK 会错过 induction/多选题的关键选择；只看 active features 会漏掉被抑制的因素；按 coactivation 过滤会漏掉负向抑制。重建更准/字典更大不保证扰动机制更忠实。supernode 仍有主观分组，示意子图刻意删去“无趣但必要”的部分，没有完整性计量。
- **评测采样边界**：Appendix Additional Evaluation Details 的 n=260 排除常见 token、上下文重复 token、过低 loss，并只取原模型预测正确的目标；这与研究触发后失败、标点或复制效应的项目分布不同。约 50% top1 替代一致也受非平凡目标筛选限定。
- **复用与成本**：原文公开 MIT 界面裁掉了内部的图差异、现场标注等组件；“论文里能做”不等于公开版本直接有。CLT 训练开销是预付成本，附录估算 Gemma 2B/9B 的示例为约 210/3844 H100 小时，另有图生成和人工分析；可以先用既有 input attribution 或 raw neuron 工具，不强制 CLT。
- **伪代码疑点（未执行复现）**：Appendix Graph Pruning 明定 `A[target,source]`，正文用 logit 行的加权和；伪代码却写 `B @ logit_weights`，按该约定应取转置方向。另 `cumulative_influence <= threshold` 与正文“最小达到阈值”的保留边界不一致，可能少留越过阈值的一个节点。应核实际库版本，不能照抄伪代码。此处为数学/文本比对，不断言发表实验执行了这两个错误。
- **引用扩展范围**：Dunefsky/Ge 的 transcoder、ACDC/Attribution Patching、Causal Scrubbing 等为内部电路算法外围；本项目当前不发明 CLT/通用电路发现算法，故不据摘要评价这些方法。若后续选择内部图为主要证据后端，需另做相应算法全文与实现审计。判定：`done/reuse/direct_neighbor`。

## F01 · Axiomatic Attribution for Deep Networks

Sundararajan, Taly & Yan，ICML 2017。[官方论文](https://proceedings.mlr.press/v70/sundararajan17a.html)。已读会议版全部 10 页，并补读 arXiv 完整版 11 页中的附录 A–B。文件：`pdfs/F01.pdf`、`pdfs/F01_full.pdf`；原图复核：`figures/F01_p7.png`。

- **研究与方法**：相对于明确基线，以积分梯度分配输出变化；讨论敏感性、实现不变性、完备性和路径选择。
- **已做**：§6.3/Fig.4 已用文字着色发现问题分类规则及不良关联；§6.4/Fig.5 已把每个翻译输出 wordpiece 的概率归因到输入 wordpiece，画出带正负值的矩阵。§6.5 还用归因发现化学网络结构缺陷。故“用归因可视化获得新认识”及“输入—输出 token 热图”都有早期明确先例。
- **可借用**：基线契约、积分收敛与完备性残差检查、有符号展示。§4–5 明确基线和路径并非由输出概率唯一决定；§6.4 的 100–1000 积分步是该翻译模型经验值，不能当现代 LLM 默认配置。
- **边界**：公理不等于真实生成全过程的因果识别；§8 明说未解决特征交互与内部逻辑。零向量 embedding 也不是自然文本。附录证明已读，不以本文支持 OA 上的可靠性。
- **追踪**：SHAP→F02；Captum→F12；seq2seq 因果解释交给 generation 域。判定：`done/reuse`。

## F02 · A Unified Approach to Interpreting Model Predictions

Lundberg & Lee，NIPS 2017。[官方来源](https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)。已读 arXiv v2 全部 10 页，及官方补充 ZIP 中四个 PDF：对称性证明、max 函数、kernel 证明、Fig.S1，共 7 页；未执行附带 notebook。文件名均以 `F02_` 开头。

- **研究与方法**：在给定简化特征映射和价值函数下，用加性解释及公理约束分配贡献；Kernel SHAP 是加权回归估计。
- **已做**：§2–4 已覆盖特征分组/缺失映射、Shapley 分配与模型特定近似；§5 有计算收敛、简单任务的人类直觉比较、MNIST 类别差异解释。
- **可借用**：将“目标函数、特征集合、缺失/替代方式、采样分布”写清；用小规模精确值检验近似，再报告采样误差。补充材料分别展示证明、算法与收敛，不等于任意文本 masking 都自动合理。
- **边界**：所谓唯一性以这些选择和公理为条件；不能说语言模型自身包含一组唯一 SHAP 因子。条件期望、独立性近似、embedding 插值、删词是不同问题。人类直觉一致是 plausibility 证据，不能替代忠实性。没有 LLM 后门实验。
- **追踪**：TokenSHAP→F11；Captum Shapley sampling→F12；忠实性区分→F07。判定：`reuse`。

## F05 · Attention is not Explanation

Jain & Wallace，NAACL 2019。[正式版](https://aclanthology.org/N19-1357/)。已读 PDF 全部 14 页，包括附录 A–E；复核 Fig.1 原图：`figures/F05_p1.png`。

- **研究与证据**：§4 在 BiLSTM 为主的分类、QA、NLI 中比较注意力与梯度/删词排序，测试置换及替代注意力分布。Fig.1 呈现不同热图与近乎相同输出。附录 B 明确梯度比较切断了经过注意力模块的路径。
- **已做**：热图差异不必对应输出差异；单看显著权重不构成“这个词导致预测”的完整证据。
- **可借用**：联合显示解释变化与输出变化；为热图建立必要性检查；声明计算图路径。单一相关系数不够。
- **边界**：§6 承认梯度/删词不是归因真值、替代权重未必可由原模型产生、允许多个充分解释会改变判断。没有自回归生成任务或现代 Transformer 结论。必须与 F06 成对使用，不能简化为“attention 毫无价值”。
- **追踪**：F06；Alvarez-Melis & Jaakkola 2017 已交 generation 域；原站额外全配置图集并非本文结论的新增证据，本轮不声称逐图穷尽该网站。判定：`done/reuse`。

## F07 · Towards Faithfully Interpretable NLP Systems

Jacovi & Goldberg，ACL 2020。[正式版](https://aclanthology.org/2020.acl-main.386/)。已读全部 8 页（会议页 4198–4205，无附录）。这是一篇观点/方法论文章，不是新的模型实验。

- **已做**：§2 区分 plausibility 与 faithfulness；§5 说明人类认为解释合理、用户任务成绩提高，都不能单独证明解释忠实；§6 展开模型、预测与线性假设；§7–8 反对把一次反例变成“解释一概无用”。
- **可借用**：项目分别评估可视分析的发现效用与归因证据的忠实性；允许明确条件下的有效解释，并报告适用范围和失败条件。
- **对立项的作用**：它支持保留“可视化作为研究工具”，同时要求把从图中获得的猜想另行验证。我们不需要先证明一种归因对所有模型和输入都完美，才能开展研究。
- **边界**：§6 的三项假设是作者对当时文献的整理，作者并未全部背书；不能将“不同输出必须有不同图”等当作无条件定理。用户实验可证明工具效用，但无法独自解决因果问题。
- **追踪**：ERASER→F08；Pruthi 的注意力解释掩盖论文→backdoor 域。判定：`reuse`。

## F09 · Investigating sanity checks for saliency maps with image and text classification

Kokhlikyan et al.，ICLR 2021 Responsible AI workshop；[arXiv v1](https://arxiv.org/abs/2106.07475)。已读全部 11 页，附录 A–B、所有图注及表；复核 Fig.3：`figures/F09_p4.png`。

- **研究与实验**：对 Inception 与 BERT/SST2 做参数和标签随机化，比较有/无输入乘子的 IG；图像的 SSIM 和文本的 cosine 等统计表现不同，另测平滑替换与 infidelity。
- **已做**：§3、Fig.3、附录 B 显示其文本设定下两种 IG 都会随参数随机化变化。因此不能把图像 saliency 的失败直接移植为“LLM 文字归因必然失败”。
- **可借用**：保留符号，检验 embedding 维度聚合和输入乘子造成的外观；使用多样例、多随机化、多个指标，区分忠实地解释差模型与解释方法失效。
- **边界**：这是 BERT 情感分类和少量样例，不是生成式 LLM 或 OA。所谓更可信受其 infidelity/扰动定义限定；正文“uniformly from a normal distribution”表述不严谨，且不同 IG 变体使用的扰动不完全相同，不能直接跨列判定统一可靠性。平滑改了目标模型，不宜当原模型机制的无损观察。
- **追踪**：F03；Yeh et al. 的 infidelity 工作列入扩展候选，未用摘要替代全文结论。判定：`done/reuse`。

## F11 · TokenSHAP

Horovicz & Goldshmidt，NLP4Science 2024。[正式版](https://aclanthology.org/2024.nlp4science-1.1/)。已读 arXiv v2 全部 9 页和会议版全部 8 页，无独立附录。正式版保留了下面的关键疑点，故不能以“只看了旧版本”排除。文件：`pdfs/F11.pdf`、`pdfs/F11_final.pdf`；已核 arXiv Algorithm 1 原图。

- **已做**：§3 用输入子集生成的整段回答与完整输入回答的 TF-IDF cosine 相似度作价值，给输入 token/substring 着色；§4 比较随机插词与采样比例。它解释的是所定义的回答相似性，不是某输出位置的条件 logprob。
- **可借用**：黑盒响应比较、输入片段展示、稳定性问题；不建议未经验证便作为默认可信估计器。
- **疑点 1（算法）**：Algorithm 1 写 `N=min(n,floor((2^n−1)r))`，因此 `N<=n`；相应分支永远不能得到正数的额外随机组合。这与 §4.2 增加采样比例增加组合的描述冲突。尚不能判定是排版错误还是实验实现差异。
- **疑点 2（理论）**：Eq.2 的 with/without 两组均值差未说明如何恢复标准 Shapley 的按子集大小加权边际贡献；若对所有子集均匀取样，它一般得到 Banzhaf 型量，不能自动称为 Shapley。这个判断来自公式比较 F02/F12，不是已运行复现。
- **疑点 3（评测）**：随机插词并不保证模型实际不依赖它；用“理应低重要性”评价，含有人工先验。§2.3 还将 IG 文献误称为 TracIn 来源。正式版有未解析引文 `?`。
- **当前代码核查结果**：下节固定 SHA 静态核查已确认当前采样实现不受上述 `N<=n` 限制；当前实现与 2024 实验逐项对应仍未证实，聚合和归一化边界仍在。不能把论文伪代码疑点当作当前实现缺陷，也不能扩展为“所有黑盒归因无效”。判定：`done/reuse_with_caution/open_issue`。

## F12 · Using Captum to Explain Generative Language Models

Miglani et al.，2023。[arXiv v1](https://arxiv.org/abs/2312.05491)。已读全部 9 页，包括附录代码示例；复核 Fig.5 原图：`figures/F12_p6.png`。另核对 [当前官方 API](https://captum.ai/api/llm_attr.html) 与 [Llama2 教程](https://captum.ai/tutorials/Llama2_LLM_Attribution)，软件文档与论文版本分开。

- **已做**：§3 已支持选输入 token/词组/自定义特征、指定输出序列或解释生成序列，使用序列 logprob 及逐输出 token 归因；Fig.5 正是输入特征×输出 token 热力图。§4 用归因考察兴趣关联与 few-shot 示例作用。
- **可直接借用**：`LLMAttribution`、`LLMGradientAttribution`、分组、替代基线分布、结果绘图。具体接口以当前文档为准：论文旧名 `TextTemplateFeature` 等已与当前 `TextTemplateInput` 不同。
- **边界**：当前文档明确 Lime/KernelShap 不提供逐 token 结果，只提供序列级结果；不能把所有封装都写成同一能力。生成序列确定后评估其概率，和删词后重新生成整段答案，是不同归因目标。
- **复用风险**：分组和自然基线只能减轻离分布问题，不能保证消除；论文案例不是后门/因果验证，也没证明对 OA 鲁棒。代码没有在本仓库运行，故本轮结论是“接口和方法可借用”，不是“已完成集成/复现”。
- **追踪**：Inseq→generation；Ecco→visual；F01/F02 是其方法基础。判定：`done/reuse`。
# 当前源码补核：F11 / F12（2026-09-09）

- **F11 TokenSHAP**：已固定作者仓库 commit `28b2f2c7a85695725c042badc567c8057a7549c5`；读取 `token_shap.py` 的输入处理/展示，以及 `base.py` 的组合采样和归因计算，存档在 `source_snapshots/`，未运行作者程序。当前实现先做每词删除，再按剩余预算增加随机组合，**并非论文伪代码中总采样数受 n 限制的写法**，故论文疑点不能直接归罪当前实现。当前 `_calculate_shapley_values` 仍使用含/不含该词的组间平均相似度差，而非按匹配联盟边际效应及 Shapley 权重显式求和；采样设计也混合必选 LOO 与其他联盟，不能未经验证称标准 Shapley。之后先减所有词的最小值、再归一化至总和 1，原始负向影响/零点已丢失。`analyze` 还压缩所有空白，可能擦除换行触发条件。**可复用接口思路，不能直接把该输出解释成触发器对某个生成词的正负因果贡献。** [固定版 base.py](https://github.com/GenAISHAP/TokenSHAP/blob/28b2f2c7a85695725c042badc567c8057a7549c5/token_shap/base.py)。MIT 许可全文已存档；本次为局部静态核查，不是完整软件正确性审计。
- **F12 Captum**：当前作者组织已为 `meta-pytorch/captum`，官方 LICENSE 为 BSD 3-Clause，已读取许可正文。[许可](https://github.com/meta-pytorch/captum/blob/master/LICENSE)。论文时代 API 与当前 API 仍须按固定版本适配；本轮只读文档与论文，没有实测支持矩阵。
