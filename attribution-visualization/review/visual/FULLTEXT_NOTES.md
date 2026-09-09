# 可视分析全文证据卡

调查日期：2026-09-09。本文件逐篇更新；`full_text_read` 表示已读取非截断的完整正文和 PDF 内附录。参考文献用于扩展，核心图另行渲染核对。没有执行论文代码或模型实验。论文提到开源不等于本轮已验证许可证或运行成功。

## bertviz — A Multiscale Visualization of Attention in the Transformer Model

- **书目／版本／范围**：Jesse Vig，ACL 2019 System Demonstrations，37–42；官方会议 PDF，6 页，正文 §§1–3 和参考文献全读，无独立附录；`full_text_read`。前身 ICLR workshop BertViz 合并为同一工具沿革。
- **问题与视图**：多层、多头 attention 难观察。head view 为 token 间加权连线，model view 为层×head small multiples，neuron view 展开 query、key、逐元素乘积、点积及 softmax。token hover、head/layer 选择、句对方向过滤。
- **发现／评估**：三个定性用例：GPT-2 doctor/nurse 指代的性别偏向；BERT 句对 heads；少数 query/key 维度与距离衰减模式对应。没有受控用户研究或因果干预验证；文末把操纵 attention/neurons 列为未来工作。
- **已做 done**：多尺度 attention 探索与通过图提出模型行为假设。**可复用 reuse**：联动导航、全局 head 筛选→局部计算细节。**候选未答**：attention 模式不等于生成 token 的因果贡献；没有触发/假触发、跨生成步的归因传播比较。
- **定位与图核对**：p.38 Figs.1–2；p.39 Fig.4；p.40 Fig.5；p.41 Figs.6–7 已目视核对，正/负 query-key 值是蓝/橙，连线粗细是 attention，不是输出贡献符号。
- **局限／追踪**：论文自身提示 attention 对单次预测解释有限；Jain & Wallace、Seq2Seq-Vis、Vig & Belinkov、Bau et al. 为后向线索。本工具不要求新归因算法，不能据其存在否定新的可视分析认知研究。

## sequence_salience — Interactive Prompt Debugging with Sequence Salience

- **书目／版本／范围**：Ian Tenney、Ryan Mullins、Bin Du、Shree Pandya、Minsuk Kahng、Lucas Dixon，arXiv:2404.07498v1，2024-04-11。11 页；正文 §§1–6、参考文献、Appendix A 完整 heatmap 与 Appendix B 计算接口均已全读；`full_text_read`。官方教程称曾投 ACL 2024 demo；未找到已发表 Anthology 对应条目，不擅自标 ACL 录用。
- **问题与方法**：长、复杂 prompt 的归因难读。选定真实或指定输出序列的片段，在 teacher-forcing 条件下计算 GradNorm L2 或 Grad·Input 对所有前置 token 的 salience；可按词、句、行、段、自定义正则聚合，输入与已生成前缀均可被归因。
- **交互与发现**：编辑 prompt→重新生成→重算热图，双栏比较模型/样本/参考答案；用菜品 few-shot 案例定位错贴 recommendation，加入原则后查看相关句影响；GSM8K 去计算注释后归因更集中，却仍算错，提示“关注正确操作数”不保证正确计算。
- **评估／局限**：三个 Gemma illustrative case studies，无受控用户研究或跨任务统计证明。聚合求和及长文本面积会造成视觉偏向；色标强度可改；作者明确多数 input salience 不因果预测行为。更集中热图不得写成模型已修好。
- **已做 done**：项目输入—输出归因、输出片段选择、已有前缀归因、粒度切换、模型/输入对照以及探索后改 prompt 验证均已有直接先例。**可复用 reuse**：LIT 的 UI/API、token-span 映射、固定目标 mask、缓存与异步。**candidate_gap**：匹配干净/后门/OA 条件的稳定跨步规律，及独立验证的触发传播认知，本文未回答。
- **定位与图核对**：p.2 Fig.1 已目视核对（选 target→选粒度/方法→选片段→前文紫色热图）；p.3 aggregation 与 contrastive target；pp.4–6 Figs.2–4；p.10 Fig.A.1；p.11 Fig.B.2 以 target_mask 选取 per-token loss 后求 embedding gradient。这里替代目标的 salience 不是两个输出分数之差的同一个量。
- **追踪**：Ecco、LIT、LMdiff、Inseq、Captum、LLMCheckup 为直接邻居；faithfulness/shortcut 论文交由 foundations。

## ecco — Ecco: An Open Source Library for the Explainability of Transformer Language Models

- **书目／版本／范围**：J Alammar，ACL-IJCNLP 2021 demo，249–257，9 页官方版。正文 §§1–8、参考文献全读，无独立附录；`full_text_read`。
- **问题／方法／视图**：Jupyter 中同时检查生成 token 的 input saliency、跨层 hidden state、FFNN neuron activation。Gradients×Inputs 后取 L2 范数，逐输出 hover 显示前文热图；详细条形图显示归一化百分比；logit-lens 排名轨迹、NMF 因子 sparkline→token 高亮；CCA 和带 control tasks 的线性 probing。
- **认知／评估**：Shakespeare 出生年分成 15/64，二者归因不同；主谓一致的正确词在最后层才升为优先；NMF 案例出现 pronoun/位置关联。另有实际 PoS probe：UD English 10,000 tokens，67/33 train/test，5 trials，报告 accuracy 与 selectivity；FFNN 与 hidden states 中可提取的 PoS 信息不同步。后者是量化观察，不应称所有 LLM 的普遍机制。
- **已做 done**：逐生成 token→前置 token 的交互热图、层间预测演变、从可视化走向可证伪假设已存在。**reuse**：hover 交互、NMF 因子联动、同时呈现 probe control。**candidate_gap**：多条件匹配触发影响轨迹及独立干预证据，不在其研究范围。
- **定位／图核对**：p.250 Figs.3–4 已目视核对；64 的归因包含先前生成的 15。百分比是归一化的非负 L2 saliency，不能当真实贡献占比或抑制方向。pp.251–253 Figs.6–10；p.254 Table 1。图7 caption 的 GPT2-XL/12层表述需谨慎，不从该图推断模型实际层数。
- **局限／追踪**：发布时 saliency 只支持 GPT2 系；§7 明确图用于直觉，应继续形成可测试/可否证假设；没有用户研究或因果正确性保证。后向线索 LSTMVis、logit lens、NeuroX、Toward falsifiable interpretability；与 Inseq/Sequence Salience 比较。

## lmdiff — LMdiff: A Visual Diff Tool to Compare Language Models

- **书目／版本／范围**：Hendrik Strobelt、Benjamin Hoover、Arvind Satyanarayan、Sebastian Gehrmann，EMNLP 2021 demo，96–105，官方 10 页。正文 §§1–7、Appendix A 三个补充用例和 B 缓存系统全读；`full_text_read`。Anthology 元数据误拼 Satyanaryan，PDF 标 Satyanarayan，以 PDF 为准。
- **问题／交互**：寻找两个模型在哪些上下文出现差异。全语料预计算 likelihood/rank 差，聚合直方图挑异常案例；实例视图同时给逐 token 概率、排名、红白蓝差值、局部分布以及两模型 top-k 候选；hover 联动，点击固定多个位置。
- **认知／评估**：六类案例发现蒸馏后的常识/指代差异、模型生成文本似然差、领域微调后词概率变化、域外退化和数据编码错误；Appendix A 展示同一 Faust token 在作品/拳头语境下偏好反转。均为假设生成，不是总体性能或因果机制的统计结论；作者在 §§4,6 反复限定。
- **已做 done**：固定文字上的逐 token 模型概率差热图、分布候选比较、语料定位异常→案例假设。**reuse**：双模型缓存、统一 tokenizer 检查、全局与局部视图、保留正常/异常分布。**candidate_gap**：把模型对比与触发条件、输入归因以及自由生成路径的影响分解结合，本文未给完整回答。
- **局限**：实例比较要求同 tokenizer 与 vocabulary；softmax temperature 影响概率，rank 是补充视角。不能将高似然简单等同更好模型。top-k outlier 筛选不是无偏确认样本。
- **定位／图核对**：p.97 Fig.1 已目视核对；pp.97–98 指标与兼容性；pp.99–101 case studies；pp.104–105 Appendix A/B。代码在论文中明确 Apache 2.0，本轮尚未运行。
- **追踪**：Sequence Salience/Ecco/LIT 为后续近邻；GLTR 的逐词 likelihood 视觉编码为后向边界工作，待纳排。

## lstmvis — LSTMVis: A Tool for Visual Analysis of Hidden State Dynamics in Recurrent Neural Networks
- 来源/版本：Strobelt, Gehrmann, Pfister, Rush；arXiv:1606.07461v2, 2017-10-30，TVCG版前稿，10页。正文§1–8、参考文献全部阅读，无附录。图核对待补。
- 问题/交互：RNN内部哪些状态携带什么信息。架构师/训练者为目标用户；平行坐标显示状态随token变化，选择文本范围及阈值，限定前后开关，检索相似状态集合的前50片段，联动POS/NER/嵌套等标签。明确G1提出、G2精炼或拒绝、G3跨模型/数据集比较假设（pp3–6）。
- 认知与验证：合成括号模型发现层级状态；PTB语言模型的名词/动词短语匹配现象后，在CoNLL标注数据的所有多词块状态集合上做PCA验证分离（§6.2,p7,Fig9）；这是表示可分性证据而非因果必要性或统计泛化检验。基因组案例定位闲置状态与卷积错位；乐曲案例单状态跟随和弦进程。
- 评估/局限：MILC式300天公开部署日志与质性反馈，约19,500访问、仅10%试demo且多重复教程，不是受控效果试验。>500维全览噪声仍难处理；阈值和人工种子有选择偏差；没有输出归因、LLM安全、held-out干预。
- done：交互探索→假设→全语料表征分析已有（不能宣称这条一般工作流首创）。reuse：匹配反例、外部标签轨道、共享可重放选择与独立验证。candidate_gap：匹配触发条件下输出轨迹/输入归因如何产生可证伪的新机制认知。定位§4–7,Figs1,5,8–11。

## knowledge_debugger — KnowledgeDebugger – an Exploration Tool for Knowledge Localization and Editing in Transformers
- 来源/版本：Eric Benz, Lennart Stöpler, Nikolai Bolik, Artur Andrzejak，arXiv:2607.01000v1, 2026-07-01，9页；正文、参考文献、附录A已读。需图核对。
- 问题/交互：知识编辑是否在内部造成异常及如何探查；LM-Debugger式GUI接入EasyEdit，trace/explorer、层/神经元检查，编辑/干预队列可重排开关，导出模型与JSON结果；target rank、causal tracing、Frobenius norm、ROME denominator、PPL等可组合（§2）。
- 认知发现与证据：GPT-2 XL三个案例：编辑目标更早升至top token（Germany第24层，对照South42/Cape39，编辑层17）；单token句首subject触发编辑崩溃/PPL约50倍，R-ROME对该案例有效；Eiffel等事实在改写prompt下因果定位与logit lens深度变动（§3）。附录A仅Llama3.1-Instruct8B一个Berlin编辑示例，不能外推普适性。
- 评估/局限：案例式系统论文，没有受控用户实验；“同知识应同深度”是作者期待，不是公认定理，定位不一致是待研究线索。发现优先、之后系统评估是论文明确流程，GUI本身并非新知识编辑算法。
- done：把可视观察、模型内部干预、知识编辑失败发现串成可交互工作流。reuse：干预队列、实验状态保存、localize与edit效果分开展示。candidate_gap：后门触发的输入输出归因/历史传播现象及跨条件验证。定位§2–3,Fig1(p4),AppA/Fig2(p9)。

## semantic_pathway — Semantic Pathway: An Interactive Visualization of Hidden States and Token Influence in LLMs
- 来源/版本：Mithilesh Kumar Singh, Klaus Mueller；作者公开IEEE VIS2025短文PDF，正文与全部参考文献已读，无正文附录。官网补充视频有链接，未作为论证证据；图核对待补。
- 问题/视图：合并token条、前驱token注意力影响字重、hidden-state跨层t-SNE轨迹/最近token、attention arcs/多头扇图、径向top-N预测；对比两个生成位置并保持选择。所谓影响是跨层求和、跨头平均的attention启发式，不是梯度或干预归因。
- 认知/评估：GPT-2短prompt质性案例：早层慢、中层漂移、晚层聚合，注意力聚焦、首token锚点、最近token异常；无规模、统计或用户研究，不能写为一般规律。
- **open_issue（关键）**：Fig2/§3把追加未来token后既有token隐藏状态差异解释成“上下文重算”。对确定性因果decoder、相同位置和相同左前缀，未来token不能改变已存在token真实hidden state。须排除分别拟合t-SNE、索引/位置、dropout或推理状态差异；本文没有给出排除证据。因此本综述仅复用界面方案，不接受此机制结论。降维投影差异不等于真实表征漂移。
- done：跨层表征+注意力+输出分布联动及两个步对比已有。reuse：共享token选择、保留比较状态。candidate_gap：真实自回归轨迹中归因传播与严格配对因果验证。定位Figs1–3,§2–4。引用链补入AttentionViz、CommonsenseVIS、VisBERT；视频未查看不记作补充完整完成。

## llm_transparency — LM Transparency Tool: Interactive Tool for Analyzing Transformer Language Models
- 来源/版本：Tufanov, Hambardzumyan, Ferrando, Voita；ACL2024系统演示pp51–60，10页；正文§1–6、参考文献已读，无附录。核心图核对待补。
- 问题/视图：从某一输出位置向下追重要信息流子图；阈值控制密度，选边看head importance/attention及contribution map，选MLP看贡献神经元，选residual state看logit lens，选component看词表正向/抑制投影。清楚区分attention weight与contribution（§2,p53）。
- 认知/评估：复现greater-than年份MLP概念促进、input suppression等已发表现象；图3可切各预测位置看宽/窄路线。§2.5明确可用于desired/undesired行为路线差异、偏见、幻觉的假设探索；工具文没有系统比较安全行为或控制用户研究。
- 边界：核心流程采用Ferrando&Voita2024 attribution，不是patching因果验证；约100倍速度引自该基础方法，不能当本系统端到端新benchmark。单次前向缓存图、多位置切换；测试到30B单节点，分布式未支持。GTP2/OPT/Llama2，插件式模型支持。
- done：输入至输出重要路径及块/head/神经元多粒度钻取、正负词表投影已有；desired/undesired路线分析也是明确提议。reuse：多粒度联动、阈值敏感性、贡献与attention分开、配套干预验证。candidate_gap：具体触发/OA条件下跨生成步可检验认知；不能将“安全场景可用”写成已验证后门研究。
- 软件现状：官方facebookresearch/llm-transparency-tool页面在2026-09-09核实为2026-02-01 archived，README称CC BY-NC4.0。论文与软件记录分开；未安装运行。引用链VISIT纳入，Information Flow Routes应与foundations协调。

## exbert — exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models
- 来源：Hoover, Strobelt, Gehrmann；ACL2020demo pp187–196，10页；正文§1–6及附录A重放链接/B材料/C图全部读完。可选视频未作为证据观看。
- 方法/任务：选择token/head/layer，注意力入/出边与head矩阵，MASK及top-k预测；在标注语料按context/embedding余弦查前50近邻，POS/DEP等直方图；head context向量先L2归一化再拼接。自回归模型匹配的标签和预测位移到“下一词”。
- 认知/验证：重现BERT句法head与词性分层（§4,Figs2–3）；GPT2 nurse/doctor换词使her概率90% vs18%、him6% vs68%，末3层attention表面相似（§5,Fig4）；注意力单独不能解释该差异。近邻仅局部例子，作者明确需后续稳健全局统计检验（§6），无用户试验。
- done：attention+语料近邻联动、跨层representation与context对比、偏见反事实探索已有。reuse：匹配邻居及属性分布、保存分析链接；candidate_gap：限定触发条件下跨步归因规律及干预证伪。局限：top50选择、单token、语料注释/覆盖、隐藏special token后重归一化改变视觉语义；没有一般因果证明。核心Fig1/4/A6待图核对。

## lit — The Language Interpretability Tool: Extensible, Interactive Visualizations and Analysis for NLP Models
- 来源：Tenney et al., EMNLP2020demo pp107–118，12页，正文§1–6、参考文献、附录A全部完整阅读。
- 任务/视图：J1数据集探索、J2找有趣样本、J3局部解释、J4生成反事实、J5多模型/样本并列、J6切片指标；local gradients/LIME、attention、embedding PCA/UMAP、metrics、数据编辑、HotFlip/替换/回译等组成联动。维护反事实来源；匹配模块共享selection。
- 认知与实证：SST含not的56条全对后，改not/ultimate等验证局部预测；Winogender男性主导职业切片在职业为答案时male83% vsfemale37.5%；T5摘要错误replaced as captain by former captain定位by概率28.7%，25decoder近邻中captain34次/former16/replacedby3，**仅提示语料先验，未作训练数据因果证明**（§3）。附录A5两个模型热图同聚焦却一对一错，是归因视觉≠行为的一例。
- 评估/局限：案例演示，没有受控用户试验。原版UI约10k样本限制；不做训练监控；深模型干预需额外接口。正文Apache2.0（软件当前状态/运行另核）。
- done：局部解释→反事实→全数据切片验证的集成工作流、多模型salience比较已做。reuse：数据来源轨迹、切片评价、输出位点语义、共享选择与反事实留存。candidate_gap：触发历史影响的具体新认知与配对干预证据。关键§2–3,Table1,Fig4,AppA5（待图核对）。

## visbert — VisBERT: Hidden-State Visualizations for Transformers
- 来源：van Aken, Winter, Löser, Gers；WWW2020 Companion, DOI10.1145/3366424.3383542；arxiv2011.04507v1为作者会议稿，5页全文/refs已读，无附录；p3Fig2已看原图。
- 输入QA样本，PCA投影每层token表示（**每样本每层分别拟合PCA**），用question/supportingfact/context/answer颜色与选择过滤，层滑杆显示估计阶段。SQuAD/bAbI BERT-base与HotpotQA BERT-large；输入支持新样本、干扰事实。
- 四阶段观察（先前How Does BERT Answer Questions?已有）：词义簇→实体属性→问题支持事实→答案分离；错误答案可能在中层选错事实，完全错误时簇退化。本文系统demo没有独立大规模/用户统计验证；图2只一个示例。
- done：跨层token语义阶段可视诊断、输入扰动比较已有。reuse：类别标签与模型阶段假设；candidate_gap：LLM自回归因果轨迹。局限：独立PCA坐标不可直接解释跨图欧氏位移；QA任务/BERT限定，GPT2未来工作。

## llm_attributor — LLM Attributor: Interactive Visual Attribution for LLM Generation
- 版本：Lee et al.，完整读arXiv2404.01361v1（1 Apr2024）8页正文/refs，无附录。AAAI2025 Demonstration正式版pp29655–29657，DOI10.1609/aaai.v39i28.35357已官方元数据核实；正式PDF首次连接被远端关闭，版本差异仍待读。p1Fig1原图已核对。
- **对象是训练数据归因TDA，而非输入prompt token归因**。DataInf影响近似，跨打乱训练checkpoint取中位；缓存每训练样本参数梯度；也提供TracIn接口。选择输出token/span，正/负支持训练样本列表、TF-IDF关键词、全分数直方图；用户修改输出并列比较。
- 证据：两个usage scenario，Llama2-13Bchat灾害文章和金融QA；夏威夷阴谋输出关联训练X帖#1388，移除并补可靠资料后重训“稳定正确”是叙述案例，没有次数、分割、消融、误差线；不得当严格证明单条训练样本为原因。金融IPO例子定位相关训练Q&A，文本相关性不等于输出正确性证明。
- done：选择输出span→训练来源归因→替代输出并列→正负分布已有；reuse：比较视图、source/provenance展示、适配器式算法比较。candidate_gap：本项目输入历史影响，不与TDA混淆。无用户试验；效率缺完整端到端报告。训练样本内部token归因、RAG在其futurework，不能说已支持。

## vista — VISTA: Visualization of Token Attribution via Efficient Analysis
- 来源：Ahmed et al., arxiv2604.02217（2 Apr2026），12页正文§1–9、refs完整读；SSRN6510678同作归并。p7公式例数已图核对，Fig1待补。
- 实际测量：GloVe50维词向量之和作为prompt向量，去一词计算方向、模长及维度分量的几何变化，再相乘；GAM学习这个自造排序（含位置特征）。不访问目标LLM、目标输出概率、模型生成结果或模型内部。
- **重要边界：这是静态词向量下的输入几何显著性，不能视为某LLM输出token的模型归因**；不存在目标model/output条件，换模型或输出仍不变。标题/abstract“模型预测贡献”比方法证据强。没有LLM faithful benchmark或用户study；只有说明性单prompt数字与界面图，效率为O(nd)复杂度而非端到端比较。
- open_issue：§3.2.3宣称相对模长变化∈[0,1]无一般保证（取消效应可令分母小而删后模长大）；§3.3把每个GloVe坐标说成可识别语义轴无证据，函数g未具体定义。低angular×magnitude不能因高dimension独力挽救；示例均为说明，未给实际embedding验证。
- done：静态embedding启发式token着色与summarygap原型。reuse：仅作为需通过sanitycheck的对照/概念区分；不推荐当主归因方法。candidate_gap不由本文证明。引用链2512.11573作为真实blackbox token归因工具继续纳入，refs中的BertViz arxiv链接1906.04341实为另一篇，不能照抄。

## lm_debugger — LM-Debugger: An Interactive Tool for Inspection and Intervention in Transformer-Based Language Models
- 来源：Geva et al., EMNLP2022demo pp12–21，10页正文、refs、AppA.1/Table3全读；p6Table2原图核对。
- §2–3：FFN value静态向量与输入依赖的sub-update区分；tokenrepresentation及FFN更新词表投影，逐层top词/10主subupdates，输入生成全过程干预开关；BM25查概念词，value聚类词云。从观察到选定components再重跑生成，是实际界面流程。
- §4消歧案例“book is for”purpose→person；§5职业输出改变与4条Yelp各10token正负生成案例（表2）。没有大样本命中率/鲁棒性或随机用户实验。表3完整列固定向量，表明比仅视觉推测多了行为干预证据，但并不证明一般机制或正负控制副作用。
- done：可视细粒度更新→概念检索→干预→输出对比早已实现。reuse：区分staticvector/dynamicsubupdate、干预记录、参数与输出比较。candidate_gap：触发样本/常规后门/OA配对下生成历史影响认知。原版GPT2；注意力/LayerNorm非主解释对象，安全监测只是讨论，未验证隐蔽后门检测。

## 原图核对增补（2026-09-09）

已查看：LSTMVis p7/Figs8–9（PCA跟进而非因果），KnowledgeDebugger p4/Fig1（队列/指标），SemanticPathway p3/Fig2（原图注确称“positional identical … recomputation”，关键疑点成立），LM-TT p2/Fig1（attention/contribution不同），LIT p12/A4–A5（两模型salience相似预测相反），exBERT p6/Figs4–5（概率数字），VisBERT p3/Fig2（PC1/PC2），LLMAttributor p1/Fig1（TDA），LM-Debugger p6/Table2（4个有限示例），VISTA p7（公式与例数）。保存在figures/对应文件，不将已渲染未查看标核验。

## llm_comparator — LLM Comparator: Visual Analytics for Side-by-Side Evaluation of Large Language Models
- Kahng et al., arxiv2402.10524v1 16Feb2024/CHI2024 Extended Abstracts，7页全部正文refs已读，无附录。图核对待补。
- 对象：生成结果的自动评分/理由与行为差异，非梯度归因。表格配对长文本，重叠词高亮、多个rater具体理由；评分分布、prompt切片胜率、LLM命名理由簇、n-gram/自定义regex指标联动。记录“why”是**rater为何偏好**，不是模型内部因果原因。
- 需求：与>20从业者讨论；3个月>400内部用户/>1000评估run。六位有经验内部工程/研究/数据人员45min观察式thinkaloud+访谈；非受控提升试验。
- 实际发现流程：从coding样本解释过少推测模型改动影响→切片与理由簇查其他例；prior patterns测试sorry/unfortunately等；rationale簇→n-gram→wordcount发现过短输出。用户会隐藏分数自行判断以检查rater（§4.2）。
- done：例子驱动/先验驱动/理由簇驱动三种可視分析—假设验证模式。reuse：聚合到样本可追溯、rater不确定与细节、定量切片留存。candidate_gap：输入历史及模型内部归因与安全触发现象。局限LLM摘要/簇会错、embedding匹配含句法混淆、内部匿名数据限制外部复核；不将使用量当研究效果。

## nemo_inspector — NeMo-Inspector: A Visualization Tool for LLM Generation Analysis
- Gitman, Gitman, Bakhturina；NAACL2025demo pp321–327，7页正文/refs全部读，无附录，核心图待核对。
- 任务：Inference页prompt/模板与Analyze页联动，多随机seed“同配置”结果聚合、不同模型/参数并列、单样本标注/编辑；JSONL、文本/LaTeX/code渲染、自定义统计。纯输出分析而非token归因。
- 实际发现：50样本/题观察一致错误（persistence众数计数）→检查GSMPlus添加操作/换数字问题→发现双问句与错标签；单MATH题补Vieta公式解决→prompt假设。OpenMath观察代码执行错误/算术错误→修数据或重生成后性能变化（§3–4）。
- 证据：原文报告低质样本46.99%→19.51%，但手工抽样为每扰动类10条“模型答错”的题，不能当完整总体质量审计；“less than5% ineachcategory”与19.51%总说法口径不清。MATH1.92%、GSM8K4.17%来自后续修复；没有受控比较工具本身造成的提升，也未给误差线。结论段“drop”与正文“improvement”措辞相反，以正文修复方向为依据并保留疑点。
- done：观察错误→提出假设→改prompt/数据→性能检验已有，复用同/异配置区分、随机生成统计及来源标签。candidate_gap：本项目归因传播机制。工具Apache2.0(论文)，未运行；目前JSONL、NeMo-Skills模型受限，自定义代码不适合无隔离外部托管。

## blackbox_token — Visualizing token importance for black-box language models (DBSA)
- Rauba, Wei, van der Schaar；官方PMLR确认AISTATS2025,258:4105–4113；arxiv2512.11573v1（12Dec2025）为晚提交的同作，完整正文§1–8、refs/checklist、AppA–E已读，官方出版PDF差异待核；图核对待补。
- 研究对象：单词替换造成**自由输出分布语义敏感性**。外部GPT4生成同义候选，多个输出抽样，ada002向量，energy-style距离与permutation test，跨候选/重复词位置平均后画词热图/效应量/pvalue。它与固定输出token条件logp归因不同。前作2412.00868奠定输出分布比较，交generation核查。
- 实证：法律/临床等少量说明prompt；AppE四个prompt，8LM，temp1/max256；模型排名相关、cos/L1/L2相关、3–100MC样本敏感性。没有用户study、广泛faithfulness或所有近邻基准。采样稳定只比较GPT3.5与GPT4排名相关，不是重复运行置信区间。
- **open_issue**：§3.2均值相似性检验不是整个输出分布相等检验；§4称s为similarity但energy公式应使用distance，AppB.4实际cdist为distance，记录主文/实现表述差异。对其最终cosine distance，energy统计只反映归一化embedding均值差平方，不能保证检测任意分布差异。AppD简单平均p值不自动是有效联合检验；缺多重检验/功效界定。§5把p>0.05称“确认稳定”不成立，只能未检出。
- 其他疑点：Fig1标cosine similarity的横轴5–15超出其声明范围，需确认示意图；称“无分布假设/全新任务”过强，实际仍依赖嵌入语义、同义替换、交换性与统计度量。AppB.6置换切分combined[n2:]在n1!=n2时不对；不运行代码，仅静态阅读发现。
- done：考虑重复生成噪声的输入敏感性热图与效应/检验UI已做；reuse：原始/扰动多样本、效应量与不确定分开展示。candidate_gap：多生成步/历史token传播、固定前缀和全输出效应的连接、触发配对。不可将未显著写成无影响；不得以DBSA当已验证安全性证书。

## gltr — GLTR: Statistical Detection and Visualization of Generated Text
- Gehrmann, Strobelt, Rush；ACL2019demo pp111–116，6页正文/refs完整读，无附录；核心图待核对。
- 方法/视图：每位置实际token概率、rank桶、预测entropy，文本颜色（top10/100/1000/其余）与三分布直方图，hover top5与概率。GPT2左前缀 vs BERT遮当前位置并左右30词条件不同，不可混为同一score。
- 验证：6来源×50文本，按2真2生成来源训练、其余来源测试；rank桶AUC GPT2 .87±.07/BERT .85±.09 vsBoW .63±.11。35名NLP课志愿学生两轮先无后有overlay（文本随机分配，顺序未反平衡），每轮5文本/90sec，中间简短教程，正确率54.2→72.3%(18.1pp,p<.001)。它有真实用户实证，但教学/顺序效应和2019采样设定限制泛化。
- done：输出token概率/rank热图和群体统计辅助发现生成规律已有。reuse：原始分数/排名/全局聚合可追溯、解释性用户任务；candidate_gap：对输出内容的输入历史归因。隐藏seed会改变条件分布，作者已提醒；不能当现代通用AI文本检测/后门检测器。


## layerflow — LayerFlow: Layer-wise Exploration of LLM Embeddings using Uncertainty-aware Interlinked Projections
- Sevastjanova, Gerling, Spinner, El-Assady；EuroVis/CGF44(3)2025，arxivv1正文12页全部读；另官方补充1页算法1–4全部读。核心图待核对。
- 对象是contextual embedding在逐层PCA/UMAP/AlignedUMAP上的投影，非输出token归因。三类不确定性：变换、表示、解释；HD与2D聚类hulls、距离矩阵、KNN边、质量metrics和簇语法语义摘要联动；原始散点→Sankey→簇纵向拉伸，作者承认拉伸改变全局距离。
- 评价：长期3 NLP专家需求；1 computational linguist thinkaloud3cases（strike/cell/post），3 VA专家初步访谈；GMB795935token/38405sentence，实际视图≤150点/层，BERT。重复active句法层中间特性；cell发现质量下降可来自聚类设定，post用距离矩阵+KNN复核投影误导，非因果神经机制/用户效率大样本。
- done：可视模式→多个原始HD度量复核→反例解释已做。reuse：保存投影方法/参数、原始高维距离、稳定性提示、多视图相互反证。原图p5Alg1核对排除了提取伪差：NN(p)补集横线在纯文本丢失，TP/FN、TN/FP原公式不同，不能报作者公式错误。质量分数依赖HD clustering本身，不能当ground truth；互斥簇stretch原有局部关系也需限定。可复用思路。
- 引用链LMExplorer/LMFingerprints是上下文embedding解释邻居；本项目以输入输出归因/触发诊断为核心，不穷尽embedding几何系统；若未来主张新跨层embedding工作流必须再全文扩展它们。

## llmbench — LLMbench: A Comparative Close Reading Workbench for Large Language Models
- David M. Berry；arxiv2604.15508，22页正文与完整refs读，无附录；核心Fig2待视觉核对。论文式工具/诠释框架，未给用户study/独立统计验证，仅一个Calvino prompt贯穿12图的示例。
- Compare+五Analyse modes：logprob/entropy/topk热图与曲线、像素图、3D地形；差词/句法连接词/Hyland话语类；重复3–20次、6temperature、promptvariation、单序列tokenprob与crossmodellexical差异；可做Observation/Question/Pattern等人工注释并JSON保留model/temp/timestamp。
- 两输出相同ordinal位置不同prefix/分词/语义位置，作者pp18–19明确承认，不是匹配条件下归因差异。只显示下一token候选，不真实执行所有“道路未选择”的后续反事实。prompt扰动单次词重叠并非隔离采样噪声的效果估计。
- open_issue：pp17–18 top5概率和0.6534，报告entropy2.315约为top5重归一化熵；如果全词表熵即使把0.3466余量合并为一类，熵仍至少约2.45，因而需说明截断/重归一化，不能按full entropy使用。pp6点Probs会重新请求生成，需确认原文与logprobs同一次返回；表述未交代稳定绑定。温度0泛称deterministic忽略服务/数值随机性。
- done：丰富输出逐token概率/不确定性、模型/提示对比与人工研究注释早已存在。reuse：快照和注释schema、从局部异常到整段比较。candidate_gap：严格固定相同prefix下输入/历史的归因变化及独立干预验证。

## neural_transparency — Neural Transparency: Mechanistic Interpretability Interfaces for Anticipating Model Behaviors for Personalized AI
- Karny, Baez, Pataranutaporn；arxiv2511.00230v2(22Nov2025)，全文正文/refs读完，未见附录；正式IUI2026(868–884, DOI10.1145/3742413.3789120)元数据已官方确认，正式PDF版本差异尚待核查。16traits(8bipolar) persona direction投影与sunburst界面；非prompt各词→outputtoken attribution。
- 方法：Haiku生成contrastive system/question，Llama3.2-3B回答，GPT4.1-mini筛选trait；对response tokens/样本均值差取direction；用户system末token投影。层20选择与R²validation基于同一25synthetic prompts/trait，R²0.34(hallucination)–0.90；高低prompt归一化到±1但不保证新样本有界。其标定对应“prompt中指定的级别”，不是heldout真实后续行为准确率。
- 真正用户实验：N80 Prolific随机between group，实验N42；10分钟情感支持对话，100char最短prompt。最终trust5.60 vs5.13,p.042,d.46，但trust前后增量无组差；迭代/消息数/最终persona score/预测信心无显著改善。11/15traits“误校准”是对本文persona proxy比较，不能直接叫真实行为误判。
- done：内部feature可视化对用户信任的增量及行为效益缺失已被研究。reuse：拆开主观信任、客观预测/干预成功、设计迭代；显示trait依赖与读数校准。关键边界：ground truth persona scores并非ground truth behavior；trait两极(如honesty/sycophancy)不是自然互斥；不能因trust高推断诊断有效。未给token来源归因或因果feature干预，§5.4才提steering。
- 新直接引用邻居2406.07882 Designing a dashboard for transparency and control of conversational AI待纳入；PersonaVectors方法来源归foundations/representation范围，若要采用再校验。

## 原图核对增补2
- 已实际查看blackbox_token p4Fig1（Cosine Similarity横轴5–15原图确认，显然不能当合法余弦值）、p7Fig2/Table4/Fig3（低crossmodel相关与高metric相关并存）；vista p6Fig1只是pipeline，未提供faithfulness数据。


## multiturn_transparency — Multi-Turn Neural Transparency: Surfacing Neural Activations Improves User Calibration to LLM Behavioral Drift
- Karny/Baez/Pataranutaporn；arxiv2605.15455v1,14May2026；14页正文和AppA–E全部读；有Conference’17/假DOI模板占位，按预印本，不据模板宣称会议。Fig5 p9已原图检查。
- 六trait方向、Llama3.1-8B，最后prompt token与direction cosine，再synthetic conversations极值标定。sunburst当前状态+按turn轨迹+点击同步history+最大变化主动提示。每轮根据完整history重新计算，但未给输入各词/历史token→指定输出的分解。
- R²≥.9来自synthetic prompt指定强度→内部score回归，同一数据选最佳层11，并因低验证score为toxicity/sycophancy加5rollouts/更强judge；包含原40situation中的10+10general，不能据此证明独立行为预测或因果机制。有限模拟样本min/max非覆盖任意user对话的保守界。
- 真RCT N246(81/85/80)，第一session全no-vis第二随机no/static/dynamic，两个prompt顺序counterbalance，各10min。可视组vscontrol RMSE d−.34至−.49；dynamicvsstatic只有Evaluation对average显著(d−.32,p.037)，sign accuracy无差p.67。Anticipation评分时直接可见sunburst；所有groundtruth是本文同一proxy activation，因此“读懂图表/跟踪其读数”不能直接推为识别真实有害行为。
- 不显著与显著的差异不自动显著：control信心上升而可视组未上升，缺直接组间变化对比时只能说一致于校准解释。作者明确未来preregister/human trait validation。把difference-in-means直接说causallyresponsible缺本文feature干预证据。
- done：逐轮内部指标可视轨迹、主动变化提示、与静态对照用户认知评价已经做过。reuse：baseline covariate、静态vs动态分开、末轮/全程平均终点区分。candidate_gap：触发/控制匹配下词源归因，heldout真实行为或干预终点，避免同一读数既解释又groundtruth。

## attentionviz — AttentionViz: A Global View of Transformer Attention
- Yeh et al., arxiv2305.03210v2(9Aug2023)，正文及refs读；核心Figs11–12 p8已原图核对，无附录。Query/key联合embedding跨多个输入、每head矩阵→单head→sentence；平移keys与互逆scale不改attention，距离与dotproduct仍只是经验proxy。
- NLP200sentences/~10kquerykeypoints/head，BERT/GPT2small；展示ViT亦属于同作。统计HDcosdist-vsdot rankcorr BERT−.938/GPT−.792，不能保证每个2D点的局部faithfulness。隐藏首token/特殊token会重归一化attention，须保留原始值。
- 5专家需求+回访E2/E3和2新专家；没有随机对照用户效率实验。发现BERT位置/语义簇、GPT2qknorm差−4.59 vsBERT.41、首token attention sink；ViTselfhead检查QK矩阵相关.94以及controlledsyntheticpatterns，不是已做pruning/causaltracing；§8把假设检验/因果追踪列未来。
- done：跨样本总览→局部检视→原始量/相关检验以发现机制问题已做。reuse：总览联动原始attention、投影与原始值区分、独立检查视觉规律。candidate_gap：固定目标的路径贡献与触发配对；仅attention形状不等价输出因果影响。

## 原图核对增补3
- llm_comparator p4/Fig3、nemo_inspector p6/Fig2及§4、GLTR p4/Table1；LayerFlow p5Alg1/Fig4与p9Fig8；LLMbench p8Fig2（原图确实entropy2.315/top5+other34.7）；NeuralTransparency p11Fig7；MultiTurn p9Fig5；AttentionViz p8Figs11–12均已查看。


## conceptviz — ConceptViz: A Visual Analytics Approach for Exploring Concepts in Large Language Models
- Li/Wen et al.; arxiv2509.20376v1，11页正文/refs完整读，无附录；正式TVCG32(1)57–67(2026),online2025 DOI10.1109/TVCG.2025.3634806 metadata初步核实，正式PDF差异未比较。p7Fig5、p9Fig7原图读。
- **直接且很强的已做前作**：Identification→Interpretation→Validation完整交互流程；自然语言concept→按自动feature explanations embeddings检索SAE层→多尺度语义簇→查看activation高而explanation语义匹配低的反例→自定义input逐tokenfeatureactivation→同prompt多分支activationsteering输出比较。不是仅概念字典。
- Gemma2-2B26层residual16384 SAEfeatures，解释嵌入OpenAI3072维，Neuronpedia36864×128token样本；图中的“conceptspace”实际是自动解释文字的embedding，不能说是模型feature原空间。相似度/activation排序不符提供疑点而非证明解释错误；unembedding投影非完整输出因果作用。
- §6两专家案例：plant983在steelplant亦激活，推翻纯botany命名，正负steer改变garden续写；superhero6610和franchise9638在Hercules/SunWukong上的不同响应扩展语义边界。实际新input检查+输出干预存在，但有限示例无heldout概念分类/副作用定量。
- §7 N12有LLM/SAE知识学生（3AIexplainability经验），20min训练+20min固定负面情绪任务+20min自由探索+20min问卷访谈；所有组件主观rating>4/5，workflow4.67。无baselinecontrolled performance/独立认知测验，不能把高评分当更准mentalmodel的因果证明。
- done：自动解释与activation反例对照、用户假设、定制输入验证、steering再可视化都已做。reuse：反例作为一级对象，记录featureID/layer/输入/steer强度/branch，区分无干预与有干预比较。candidate_gap：匹配trigger/control、输入与生成历史token的目标归因、迁移/heldout因果验证；只声称新闭环流程不足。
- 版本/引文瑕疵：ref55 exBERT误引用成另一篇“Extending pretrainedmodels...”同名；ref70 Neuronpedia作者也错误，不能照抄。Backward发现Dodrio2021acl-demo16、AttentionFlows2020.3028976待官方摘要筛查，NeuronautLLM纯neuron界面待范围筛查。


## visit — VISIT: Visualizing and Interpreting the Semantic Information Flow of Transformers
- Katz/Belinkov, Findings EMNLP2023 pp14094–14113；正式ACL PDF20页，正文、refs及附录A–E全部读。图形原件核查待补。
- 单次forward、最后input位置，仍含全部input K/V。全网络/层/attention/MLP→HS/neurons流程图；hover top projected tokens、用户指定target token rank/prob，边宽activation/向量norm，节点可隐藏。QK/OV解释不能不分空间；WV输出经WO才在residual空间解释。IK=top50 token交集只是语义proxy。
- §3 CounterFact随机100个已答对prompts，GPT2medium（总体仅~8%答对），附录重复未筛正确100及GPT2XL（~14%答对）。高norm heads及高attention values与block output投影交集较高；初始4–6层不符合后层模式。D3 QK projections几乎不对齐，作者保留默认投影的解释仍是假说。
- §5.1 IOI复现前作NameMover/NegativeNameMover表的定性图形，但明确不能代替前作因果证据。§5.2 LN前后logitlens functionword降/contentword升；随机向量对照，附录E展示finalLN对projection影响。§5.3 视觉发现常激活神经元→100prompt统计：18–23层每层至少1neuron在85%以上样本进入top100/4096，norm高、投影高entropy/局部权重outlier；“regularization”是角色猜测而非消融证明。
- 无用户研究，案例+统计验证视觉线索；Limitations明确无causaltechniques、需futureinterventions。按activation剪枝只是假设，低activation节点可能重要；颜色rank与边宽不构成各路径target因果贡献，logitlens独立LN也不保持加和概率。
- done：跨层流程图、最后位置回溯输入K/V、用户target、可视发现→数据批量验证的流程早已存在。reuse：图的候选节点追踪、原始projection与统计交叉检验、隐藏节点显式披露。candidate_gap：匹配触发/假触发固定target的条件归因、heldout控制及真实干预；不能以流程图本身或generic发现闭环立新意。


## commonsensevis — CommonsenseVIS: Visualizing and Understanding Commonsense Reasoning Capabilities of Natural Language Models
- Wang/Huang/Jin/Fang/Qu；arxiv2307.12382v1,14页正文+refs+SupplA–E全读；正式TVCG30(1)273–283(2024),DOI10.1109/TVCG.2023.3327153，机构库元数据核实，正式PDF差异未查。
- 任务R1–R4：ConceptNet 2hop外部知识参照+SHAP input重要词→全局双UMAP(问题/答案embedding、可按正确性/关系监督)→按context簇汇总concept覆盖→编辑问题/选项回测→书签实例MEND修复→重载模型。直接且完整的exploration/explanation/editing先例。
- UnifiedQA-v2-T5-large,CSQA验证1221例,71%；N10至少两年NLU经验学生/校友用户study，20min教程15min熟悉20min任务+Likert/thinkaloud/log。无随机baseline效率/客观认知准确比较；Global占62.68%时间，Instance最直观，Subset最难(mean3.2)。另1专家100sample评价KG覆盖91%，不是模型真实路径groundtruth。
- §5.1案例emotion-cause图示错误→简化问题仍错→MEND指定例100%修复，总体71.01→70.93；movie/watch伪关联→数次文本反例→MEND100%修复，总体70.84。附D office/room关系输入重要词+搜索样本90%/88.89%，不是heldout独立验证。对噪声标签sea/ocean与拼写问题也有用户发现。
- 证据边界：SHAP高与KG概念重叠不等于隐式关系被使用；用正确样本fit线性映射再看对齐+有监督UMAP有构造性，不能把聚类形状证明知识；作者§6.4承认非线性关系和probe误导。外部KG是参照，不是模型机制。§4.5损失Le=-logp却写Ltotal=-we Le+Lloc有符号疑点，需原图核对后才视为原文排版问题，未运行代码。
- done：可视发现→问题干预→局部模型编辑→全体性能代价已做。reuse：概念缺失与错误对照、实例反例、编辑后副作用终点。candidate_gap：目标生成token的输入/历史分解，trigger/sham/ordinary/OA配对，heldout机制干预；“首个交互假设验证闭环”不能成立。

## tx2 — TX2: Transformer eXplainability and eXploration
- Martindale/Stewart,JOSS6(68)3652(21Dec2021),DOI10.21105/joss.03652；OSTI官方存档6页全文/refs读，无附录。软件论文，非3页。
- ipywidget/Jupyter，文本分类train/test/misclass UMAP投影、关键词/筛选；删除单词重算softclassscore重要性；编辑文本即时更新heatmap与投影；投影后聚类、簇内词频和聚合leavewordout重要性、簇重采样、confusion/F1。
- 已做：textclassification可视编辑/逐词重要性/簇诊断一体化。未给用户实验、案例效果/benchmark定量或faithfulness验证，6页主要软件架构与功能。不能作为新认知发现有效性的实验支撑。reuse：简单误差切片与编辑联动；当前生成前缀归因不在范围内。BSD3软件/CCBY文章分别记录。

## llm_comparator_tvcg — 正式扩展版补充（同一工作，不重复计数）
- VIS2024/TVCG2025 DOI10.1109/TVCG.2024.3456354；conference官方11页正文refs完整读，引用的AppendixA prompts/B问卷不在PDF，仍在检索，故body_read_supplement_pending。旧CHI2024LBW7p已全读，不能混用成最终版。
- 新增26人调查、超过1000用户/2500run五个月部署日志、更新软聚类实验。N6×45min现场观察仍有；26人中15常形成模型行为假设，20/24认为能识别模型差异；均自述/观察，无因果baseline性能试验。案例细节因保密已改写，不能当可复现实验。
- 聚类：LLM生成labels+三paraphrase embedding平均+cos阈值软分配，迭代200bullets到10labels；10个Arena模型对，各至少200输入；NLI模型作为oracle，800人工pair一致率91%。Table1 distinctiveness .978/coverage .432/Jaccard .630/NMI .381；coverage并无优势。oracle是代理并有误差，不能称人类groundtruth真值。
- 已做：local例子→hypothesis→全体自定义正则/JS统计、rationale切片确认；旧版real-timeLLM验证过慢/不准故转precompute。why是judge的理由，不是生成模型内部因果；§10.2把prompt→response saliency列未来。真实训练修复主要为usage scenario方向，未给本系统修复后性能试验。


## dodrio — Dodrio: Exploring Transformer Models with Interactive Visualization
- Wang/Turko/Chau,ACL-IJCNLP2021demo132–141；10页正文refs及§8AppendixS1–S3全读。BERT/DistilBERT+SST2/PTB。
- 各head总览circle: color编码attention收到的sum与sentiment/gradient saliency cosine、attentionargmax依存预测最好准确率；size默认average maxattention，也支持sum absolutegrad。不能把默认“confidence/importance”当因果重要性。详图force/grid/radial、parser依存vsattention关系、跨head比较、最后4层concatUMAP选样。
- §4 sentimentcoreference和PTBnominal heads案例与旧论文规律一致。DistilBERT全headmaxattention~1、后层semantic/syntax弱→猜其他语言知识；正文明确可后续quantitativeexperiment，尚未执行。§5用户study计划中，无userstudy结果。
- done：attention+gradient saliency/语法关系联动，跨模型比较生成新假设。reuse：多个解释渠道分开显示与原始量随hover；candidate_gap：当前生成token条件贡献、触发对照/干预验证。Broaderimpact提weightpoisoning仅风险讨论，非已验证触发诊断。

## Scope decisions after direct-neighbor expansion
- PromptIDE/PromptAid/ChainForge：已下载但仅abstract/目标功能筛查，作为prompt工程流程邻近；不以其摘要推全文用户study结论。PromptIDE局限正文定点确认saliency是future方向。当前不纳入支撑内部归因/触发诊断主张的全文集合；generic explore/test loop已有完整直系证据。
- ShortcutLens：abstract+§5.3定点读，重点为dataset templates/productivity/coverage；what-if重算去掉被覆盖实例后的模型准确率，不是输入token归因或同例控制干预。列外围，不据全文案例结论。
- NeuronautLLM：直接但合法全文缺口；ScienceDirect公开OA abstract和metadata已核，PDF403，作者wenzhen.site/publications只有code/DOI无PDF，exacttitlePDF搜索无官方替代。待原文不能声称其userstudy证明或具体神经元机制；保留DOI、作者项目源码入口补证。
- Comparator补充：会议官方content页的DownloadSupplementalMaterial实际只指GitHub；公开repo tree无PDF/appendix/survey文件，已检索author/exacttitle appendix/supplemental无附录结果。正式正文可支持功能/已报告结果，不能核验完整问卷/提示语以及任何依赖附录细节的主张。

## KnowledgeVIS — full text and appendix audit
- **来源/范围**：arXiv:2403.04758v1，20页正文、参考文献与附录A–D全读；accepted TVCG DOI 10.1109/TVCG.2023.3346713（当前证据使用本地arXiv版）。原图核对p4 Fig1及p16–20 Figs5–10，包含所有附录案例图。
- **问题/方法/视图**：多个模板×主体替换形成masked-token概率对比，覆盖BERT/RoBERTa/DistilBERT/SciBERT/PubMedBERT。WordNet Wu–Palmer相似度→层次聚类/Ward→LCH语义标签；热图、平行词集及rank对齐、dust-magnet概率加权散点，词/模板/主体联动。附录A–C给出聚类、rank fisheye和概率重心布局；这不是隐藏状态投影或输入token归因。
- **发现/评估**：§5共15句、114主体替换、289变体；6位NLP研究/工程专家自由探索与访谈。PubMedBERT对short/long及关键词替换敏感、SciBERT对措辞/语序敏感；专家自行改写语义角色、非常见水果等产生新假设。图支持被展示prompt的输出概率/排名关联，不足证明训练语料缺失、真正推理能力或唯一内部机制。
- **已做**：组织多组输入→看概率分布/例外→反复改写检查关联；不能再声称“首次把可视化作为假设发现工具”。**可复用**：同目标词跨条件表、rank/概率双视图、明确top-k缺失标记、具体prompt变体的可追溯比较。**未答**：§6.1训练数据补充/去偏/模型修复都是未来建议，没有重训闭环、盲测验证或归因faithfulness。
- **局限/审计**：top-k外不是概率0；散点相邻与共享边由人为布局产生，应回查完整概率；log色标影响显著性。模型语义关联不等于因果归因，WordNet标签不是模型自身概念。专家评价无对照；“underrepresented in training”是推测。单次支持约10 prompts、k30–200，未证明扩展性。

## TalkTuner / Designing a Dashboard for Transparency and Control of Conversational AI
- **来源/范围**：arXiv:2406.07882v3，2024-10-14，38页正文参考文献附录A–Q文字全读；p6/9/18/23–31/35/37原图核对，特别8页完整干预回答，不能把raster空白提取当已读。附录F另链接120回答数据，未独立复算全部原始实验；论文全读不等于原始数据复现。
- **问题/方法**：Llama2-chat-13B隐含用户属性读取与激活控制。13,900合成对话、四组age/gender/education/SES one-vs-rest logistic probes；reading probe在追加assistant提示末token，control probe在原user消息末token，后者读取稍差但控制更强。控制方向注入每个生成步；实验选20–29层/强度N8，对照相等L2 reading方向。概率条显示probe输出，pin为激活干预，不代表模型真实人口属性100%置信。
- **发现/评估**：每属性30问题，greedy回答，GPT-4强制二选一判断哪回答更符合目标属性，随机答案顺序/目标方向；控制方向成功率1/.93/1/.97，reading .9/.8/.87/.93。PANDORA 3044真实用户、每人5帖子读取balancedacc .85 vs control .70（附录C）。N19 within-subject，UI固定baseline→read→control，仅任务顺序随机；16学生/研究者。用户对控制做what-if，发现SES导致请求10天却缩为8天等意外偏差；使用意愿上升、trust无显著改善。
- **已做**：内层可视proxy→用户交互方向干预→重新生成→观察未预期偏差，是发现型系统直接先例；不能把read/control可视组合或用户what-if本身算新贡献。**可复用**：读出与可干预方向分开、对照等范数、保留原输入重生成、probe跨域检查、连续干预强度曲线。**未答**：不提供输入/历史→固定输出token归因，也未证明用户发现能预测未知trigger机制。
- **证据边界**：线性可解码不说明独立/唯一因果用户属性；方向干预改变生成并被judge认作对应风格，不证明只控制该语义。三层准确率依赖合成标签；非二元数据因主题混淆被删除；真实用户gender/education错误也受训练刻板印象影响。UI固定顺序和任务差异使主观提升不能作为无混淆因果效果。p37二维投影不能证明高维无OOD；p18 logit lens中间层预测不能自行证明最后层“覆盖真实推断”的机制。原始模型“user model”是作者解释。原始回答显示幼稚化、啰嗦等可伴随变化，非目标属性专一。

## LLMCHECKUP — full text and appendix audit
- **来源/范围**：ACL 2024 HCI+NLP pp89–104，16页正文参考文献附录A–F全读；p14–16 raster表5/6和Fig4/5实际查看。
- **问题/视图**：自然语言查询21类解释/统计/预测/扰动操作。相同LLM负责parse/predict/explain/respond，multi-prompt两阶段先操作后参数，与grammar-constrained decoding比较；通过Inseq展示attention、input×gradient、LIME、IG输入×生成token热图；可改prompt、增广、counterfactual、找相似例、请求自然语言rationale及后续建议。
- **评估/发现**：ECQA/COVID-Fact演示。119手工parse对，MP更准确，如Beluga 88.24 vs67.23，Mistral84.87 vs55.88。augmentation以标签一致率及SBERT cosine衡量；无真正user study（正文脚注明确）或发现效率试验。AppC/p14直接展示生成`were religious`列×输入token行，p16 counterfactual完全换了问题后标签变化，不能当最小因果扰动。
- **已做**：自然语言入口整合多个归因、生成输出热图和改写诊断已存在。**可复用**：按能力选归因、每个生成token可追溯矩阵、history去重提示与JSON导出。**未答**：没有证明用户形成更准的mental model、也未比较faithfulness或提出隐藏触发机制发现结果。
- **局限**：self-rationale与数值归因并列，不保证一致，更不因“同一LLM自我解释”而自动faithful；SBERT相似度不是fluency，标签保持不是语义保持；正文把LIME置于white-box组是分类松散。系统是可用性/解析评估，不是因果机制验证。

## 补充视觉核对记录（2026-09-09续）
VISIT p5 Fig4/5、CommonsenseVIS p5 Fig2及编辑目标公式和p7 Fig5、TX2 p3 Fig1、LLMComparator正式11页版p9 Table1、Dodrio p6 Fig5均实际查看。CommonsenseVIS负号疑点确在原版，不由文本提取产生；Dodrio的diameter来源是confidence，不能视为已证实causal head importance。

## GraphGhost — direct graph/history predecessor
- **版本/阅读**：arXiv:2510.08613v2，2026-01-29；14页正文、参考文献、附录A–D全读。官网版本史核对，v1为2025-10-07，不能混用不同作者表。原图核查p4 Fig3/4、p7 Fig10及指标公式、p8 Fig11/12/Table1、p11 Algorithms1/2、p14 Table7/Fig15。
- **方法/视图**：以终答token为根，对归因指出的既生成tokens递归调用circuit tracer，合并局部边为无权sample graph，同时保留局部logit attribution；dataset graph汇集token-layer对与边权，正文说row-stochastic normalization。图连接输入/生成历史、跨层语义features；in-degree衡量作者所谓semantic merging，反向logit边后PageRank找reasoning triggers。没有交互界面评估或user study。
- **已做/结果**：5层toy GPT2 graph-search 98%准确率，真实5种Qwen/Llama及distill模型、6任务；局部图选择输入与last-layer mean-attention/random比较删除后自由生成一致率；节点置零改变回答语言、推理步骤/重复。图形发现→选择图节点→消融验证已经存在，生成history的递归图聚合也不是空白。§3.2.2称单H200每样本0.5–3h，这是论文自报，非本项目成本实测。
- **可复用**：sample与dataset分别保留、直接输入/生成历史来源编码、图排序后与扰动结果关联。**未答**：没有普通/隐蔽后门匹配模型、trigger/sham或未知样本机制预报；没有用户发现效率/可靠性对照。
- **关键边界**：Fidelity是Gemini-2.5-flash-lite判断两次自由生成语义一致，不是固定prefix下目标token贡献；零化后输出变了可能一般损伤/OOD/语言切换，不唯一证明具体“逻辑节点”。图谱依赖transcoder fidelity、阈值和token-layer节点映射；论文仅约束first-layer MSE，未给全层替代模型behavior fidelity。shared token/layer不必是相同语义feature。正文自承missing/spurious edges和过度解释风险。
- **原文疑点（已视图核）**：p7说Infidelity=1−Fidelity，却仍写同一个“相等时为1”的indicator；p8 Table1 Qwen2.5-1.5B称28层却列space_31；p11 Alg2没有正文声明的归一化步骤。不能据此推断代码执行错误。p14分模型曲线并非处处优势，ARC的DS-Qwen某些保留率低于random/attention，正文“consistently”须收窄。图例、dataset counts和完整选样/排序阈值协议不足支撑跨任务SOTA断言。

## Attention Flows — full paper audit
- **来源/阅读**：DeRose/Wang/Berger，TVCG27(2):1160–1170，2021；作者11页公开版全读（页眉仍有2020/DOI占位）；无独立附录。p4/5/8原图核查Fig3/4/5/11/12和Eq4。
- **方法/视图**：BERT CLS反向跨层thresholded attention bipartite graph；任一head超阈值即连边，counts跨层用alpha=.5衰减。环形token/layer、12head glyph、位置sparklines、pretrained/fine-tuned共享/特有双色；点击两个层token或brush词组，看共同attention路径；概览5点评分向上取整且clamp5。
- **已做/认知**：模型对比和目标端反向探索已有。RTE receive/go、Doha/Qatar、QNLI5W问词桥接答案、MRPC区分短语等案例；3NLP专家+7CS非专家自由探索与反馈，用户认为fine-tuned更关注相关词；没有对照发现率或输入/节点独立干预。
- **可复用**：共享/不同图形编码、分头筛路径、目标向前追溯、完整句子与子图联动。**未答**：群体检索、归因忠实性、因果机制仍未验证；最后一节承認只看attention、embedding/跨样本扩展未来。
- **边界**：路径存在/条数不等于贡献，更不能以无thresholded attention路径排除影响（residual/MLP/value/output omitted）。counts正向、衰减和clamp也不保留负贡献；该score不是output-specific gradient/因果量。论文借attention图声称深层保留token identity并未直接测表示可识别性；case pattern不能作一般机制定律。

## DeepNLPVis — formal body complete, separate supplement unresolved
- **来源/阅读**：Li et al. TVCG28(12):4980–4994，2022，DOI10.1109/TVCG.2022.3184186。Microsoft官方研究页提供正式15页，正文、全部refs/bios已读；p5 Eq1/Fig4、p10 Fig7、p11 Fig8/9原图核。正文p6提及7数据集margin补充实验，单独supplement尚未获取，因此状态body_read_supplement_pending，不宣称附件全文闭合。
- **方法/视图**：选两类归一化s=sp/(sp+sq)；对逐层word表示加Gaussian噪声，测标准化预测差及极性；word→layer contextual representation、layer→layer relation称MI近似。相似context vectors聚邻词phrase；storyline线宽/色显示贡献/极性，cross-word curves显示依赖，corpus hexbin/1D t-SNE→word percentile/trend→例子层内图联动，串联CNN/LSTM/BERT。
- **已做的直接诊断闭环**：SST-2图显示dvd持续负贡献→查39负/14正训练分布→选错例追dvd→CLS→删偏置词并纠标签，fine-tune后8个原错例修复。另一案例改最后4层adapter test93.23→93.92；ELMo+LSTM与BERT对比后distillation88.53→89.33；AG News Google两处跨词增强→business错为tech。不是只搭界面：可視發現spurious token→查数据→修复模型先例确已做。
- **评估**：38份有效需求问卷，5专家访谈，E3/E4案例，5位50–70min最终反馈，平均25.5min熟悉工具。没有随机对照或多轮独立发现成功率；8例是探索后同例修复，非新触发泛化。模型性能改善不单独证明可视化必要性。
- **可复用**：群体异常→词的极性/层变化→实例路径→训练数据核对→模型重测；明确input和跨词关系。**未答**：生成任务扩展/多句内外区分、自动从用户反馈修模、大模型长上下文均在future work；没有后门/OA实验。
- **指标边界**：Eq1是有符号平均预测差，不能径直等同非负互信息；噪声方差与归一化依赖基础Guan2019及未获supp，不能继承“统一信息量”的理论保证。高斯加噪不等于删词，phrase clustering不等于已识别模型syntax；平均有符号变化可互相抵消。1D t-SNE用于选例不能证明语义机制。正文效应需按其任务/数据边界引用。

## Monitoring the Internal Monologue — 全文38页及附录A–K
- **版本/阅读**：arXiv:2605.18549v1，2026-05-18，官方版本史仅v1。38页正文、refs、A–K全读，K为整页轨迹，p31–38全部原图逐页核；正文p2/5/6/8及附录p19/20/21也原图核。论文自报A100 40GB合计约3000 GPUh，未复现。
- **问题/方法**：预报推理模型最终有害性/数学正确性。每层3层MLP+GELU，对序列max-pool；隔层9–31等多层logit合成MIL（50–80M参数）。把global max改成cumulative max产生逐token probe概率；绘制prompt与CoT各归一化至0–100%的均值/样本轨迹和差分。六组统计/波动/趋势/边界/峰值/tertile特征→LR/RF/XGBoost，选择最佳RF；SHAP解释的是这个监测分类器。
- **评估**：R1-Llama8B、Qwen3 4/8/14B；WildGuardMix训练→WildGuardTest/Aegis，ProcessBench训练→GSM8K/MATH。Table1静态max在WildGuardTest Template94.44/96.64/94.52/95.91 AUROC；该设置avg/last明显差，不是一般‘平均池化失效’定律。Math静态约65–78%。图形观察显示harmfulness更多终点/稳态特征、math更多波动；6组SHAP前10不重叠。没有交互系统、用户试验或对模型内部的因果干预。
- **重要协议区分**：主文trajectory结果在每个evaluation test split内3-fold CV拟合分类器，是作者明确承认的诊断可分性‘upper bound’，不是未见分布端到端部署表现，也不是数学严格上界。AppE.2才是WildGuardTest训练trajectory RF→Aegis独立迁移，Fig11约0.78–0.81，与静态持平/小增；MATH leave-one-category-out显示多数类别提高但R1/8B algebra略降，不能写全类别始终提升。Fig7约5% CoT及Fig10绝对token截断是回顾性预报，不证明实际在线停止策略和低FPR性能。
- **已做**：把内部probe序列作为发现仪器→从图识别不同动态→构造/比较特征→监督区分及跨类别测试的路线已存在。**可复用**：原始轨迹和聚合同时显示、prompt/CoT分界、将视觉假设转成可检验统计特征、将探索内CV与独立迁移明确分开。**未答**：未做clean/普通后门/OA匹配对照、trigger/no/sham或固定目标输出归因；不定位哪些输入/历史token因果影响哪个输出，不能取代归因审查。
- **机制/统计边界**：文中‘unfaithful CoT’仅WildGuard分别给CoT与response不同harmfulness标签，不等于CoT因果不faithful、隐藏欺骗或真实意图；‘internal monologue/true intent’是作者术语。平滑 partly 由cumulative max的构造产生，max前隐特征单调不意味着最终线性/MLP概率单调；其稳定性不能自行证明原模型状态稳定。统计SHAP是RF的相关性解释，不能证明生成过程机制。按等长tertile分段不证明规划/推导/结论正好三等份。数据标签依赖WildGuard，math exactmatch含格式失败；主文模型/分类器选择、未报完整heldout超参选择与低FPR标定，不支持部署SOTA。

## KnowThyself — conversational tool orchestration prior
- **版本/阅读**：arXiv2511.03878v1，2025-11-05，实际4页（arxiv comments写5页，未照抄）；全部正文refs，p2 Fig1原图核。AAAI2026正式3页元数据已核，官方PDF连接关闭/网页工具受限，未声称正式版逐页已读。
- **问题/视图**：Gemma3-27B orchestrator重述问题→nomic embedding路由→BertViz/TransformerLens/RAG知识解释/BiasEval→聊天内交互图+自然语言总结；可上传模型。演示LLaMA2 pronoun/she的attention heatmap、BOLD生成偏差regard等。系统构造及两种使用情境，没有userstudy、数值发现率、路由准确率或归因faithfulness评估。
- **已做**：自然语言统一调用解释模块、跨任务连续会话与可视结果讲解不是空白。**可复用**：解释类型显式路由、能力边界显示、图与自然语言说明关联。**未答**：没有探索→新机制假设→独立干预验证，工具组合本身不证明发现能力提升。
- **边界**：Fig1把attention结果称token attribution；解释‘Maria是主语’是orchestrator自然语言推断，不由attention权重本身证明。基于RAG文献的解释不保证对该特定模型忠实；bias metric也不是训练成因。不能把论文的‘significantly lowering barriers’当已测量效果。

## DBSA 正式AISTATS2025版本复核
- PMLR258:4105–4113官方22页PDF（含完整附录A–E）已独立从头至尾读；arxiv2512.11573v1为晚上传版本，不应据ID把工作定为2025年底首次发表。p4 Fig1、p17 B.5/B.6原图核。
- 与已读arxiv主要结论和算法一致；字符diff大量差别来自字体映射，不作为科学变化。正式版仍存在Fig1‘cosine similarity’横轴5–15、B.6不等样本分组索引[n2:]、对邻词与位置p值直接平均等问题。B.5按原始回答embedding置换，与§3的相似度pair集合不是同一检验；不可混淆两者统计有效性。正式全文没有补成用户研究、faithfulness benchmark或更强检验保证。既有证据卡结论仍适用。

## LLM Attributor 正式AAAI2025文本复核
- DOI10.1609/aaai.v39i28.35357，pp29655–29657，官方网页缓存PDF3页全部146行已读。当地下载两次RemoteDisconnected，web screenshot timeout；因此正式版Fig1未单独像素复核，旧arxiv8页Fig1已核。正式版文本与原图caption内容一致，但不将此当正式像素核对。
- 正式正文保留DataInf训练shuffle+每epoch checkpoint、TracIn接口、选output phrase比较训练support/inhibit及TF-IDF/histogram；没有独立evaluation或userstudy段。8页arxiv中的情境/实现分析应按版本引用；正式3页不支持额外实证结论。

## XMD — attribution feedback and model repair
- **来源/阅读**：ACL2023demo264–273，官方10页全部正文refs读完，无附录；p3 Fig3、p5 Tables1/2、p6 Fig7原图核。
- **工作流**：Captum IG等对预测类别归因；实例热图支持add/remove/reset词的重要性，全局榜平均所有含词实例归因→点词看案例。instance模式仅选预测正确训练例以使prediction target=groundtruth；task remove可作用全部实例。用户目标重要性置0/1，再以task CE+MSE/MAE explanation regularization重训。
- **结果与评估边界**：BigBird-base SST→Amazon/Yelp/Movies；STF→HatEval/GHC/Latent。Table1 Movies82.0→94.5、Yelp89.0→94.4等，某些配置下降如Amazon89.1→88.4；hate任务也不是所有OOD提升。实验反馈主要用现成SST rationale、固定group identifiers模拟，不能当真实用户发现未知spurious机制。两名研究生各条件50例计时约60s vs传统110s，然后模拟不同时间预算训练性能；非完整互动发现率/因果可视化价值对照。
- **已做**：归因结果→人类修正归因→模型重训→ID/OOD验证的系统链条已存在。**可复用**：instance/task反馈区分、不要把错预测的归因target混成groundtruth、原始与目标score并列。**未答**：未知后门/OA触发模式、生成history传播、改好热图是否真改决策机制、自然用户发现质量。
- **边界**：对解释的正则可优化解释代理而不保证移除唯一因果路径；均值词榜有语境/频率混杂。文中18%是作者摘要说法，评述优先报具体table变化，不能混成统一绝对提升。页面少训练细节/种子及统计误差，不能按其强措辞宣称‘catch all hidden patterns’。

## IFAN — LIME feedback with adapter debiasing
- **来源/阅读**：arxiv2303.03124v2(2023-10-02)18页正文refs附录A–F全部读；正式IJCNLP2023demo59–76 PDF已取得，完整规范化diff仅arxiv水印、页码和会议页脚，无科学文本变化；两版本不重复计数。p5 Fig4/Table3、p10/11 API原图、p15/16 feedback轨迹、p17 BLOOM图表已核。
- **视图/交互**：模型预测+confidence；局部LIME按class色，阈值0.1；global是独立输入所有unigram的分类score榜，不是含词语境的平均归因。用户纠label/标选ngram；存为文本训练样本，用adapter训练并冻结原权重。用原数据重平衡减少遗忘；开发者看修复前后performance/错例。
- **实证**：tiny BERT（脚注2层128d），HateXplain二分类Jewish subgroup；3人标24错例、40片段重复到120+500原例。most-confident balanced subgroup precision.95→.97，整体F1.79保持；不平衡整体F1降.31。BLOOM560m AppD总体F1.48→.51，group F1.48→.53。n9（3expert/3CS/3lay）可用性问卷局部3.88/global3.2/总体3.33，反馈提等待、markup误导、解释单一，没有随机对照发现成功率。
- **已做**：检查归因/修改片段→adapter→修改样本与全体性能轨迹；关键发现是feedback-only可能损伤整体，混原数据缓解。**可复用**：可撤销adapter、局部/global分开、展示修复与旁路损伤。**未答**：开放式生成归因与未知触发模式；AppendixE虽用Alpaca生成label仍是分类并非一般生成机制审查。
- **边界**：LIME是局部扰动代理，selected文本训练不是直接约束该输入目标的因果路径；global unigram缺语境。样本/超参反馈曲线作者自承尚无清楚规律；同例置信度修好不能证明机制普适。小幅group precision变化没有置信区间，非后门防御实证。

## Transformer Debugger — 软件证据，不冒充论文
- **来源**：OpenAI官方GitHub `openai/transformer-debugger` README（2026-09-09读取；项目自给Mossing et al.2024 citation，MIT），不是同行评审论文；没有安装或运行系统，也未把视频介绍当实验实测。
- **已有能力**：README明确token A vs B、attention head→token问题；识别neurons/heads/SAE latents贡献，连接component追circuits，forward-pass干预后即时看behavior。提供React neuron viewer、activation server、GPT2/SAE模型及top-activating样本，示例IOI。
- **可复用/边界**：已有‘探索→选部件→干预’软件设计，不能把此workflow声明为首创。自动特征解释与可视连接不是完整因果证据；README没有用户study、触发/OA比较或未知样本发现验证，当前只证明官方声明的功能范围。

## ELIA — Simplifying Outcomes of Language Model Component Analyses with ELIA (EACL 2026)
- **阅读**：官方正式18页（111–128）正文、参考文献、附录A–C全部；图2–4与附录图6–13原图全部核查（本地figures/elia-p3,p5,p6,p10,p11,p13–18）。源 https://aclanthology.org/2026.eacl-demo.9/ 。
- **问题/视图**：把模型分析结果转成非专家可理解叙述。OLMo-2-1124-7B被解释，Qwen2.5-VL-72B看图+数值生成叙述；Inseq Saliency/IG/Occlusion input与generated prefix→output矩阵；Faiss训练文档邻居；末层末token向量/PCA/层变化；512特征每层的所谓CLT图、局部路径、feature/subnetwork drilldown。附C训练1500steps,batch16,L1=1e-3，未核代码。
- **已有闭环**：§5法国首都案例把France归因、功能相似、跨层特征串成叙述；§4/Fig4在三个示例（首都、factorial、文学）比较top特征/路径消融和随机基线，另报CPR。这确实是多视图加局部干预的直接先例，不能写“以前只有热图”。
- **证据边界**：Function Vector在§3.4/AppB实际为5个范例prompt末层末token均值，120类别/6types；未实施Todd功能向量原论文的干预验证。Faiss kNN只相似检索，不证明训练因果影响。CLT图的具体跨层映射/误差残差保留、top-k与CPR完整计算协议披露不足，不能由几张图认定faithful完整机制。
- **叙述验证**：同一个Qwen负责生成、拆原子claim、语义校验；程序校验数字/连接可复用，低温/固定seed/规则不消除自证循环。Fig6用token平均/峰值越阈值验证“正式语气/正确答案”等语义解释，Fig11以通用层功能常识验证syntax→meaning→output；这些不是独立机制验证。正文说circuits uniformly perfect，但Table1为45/46、132/137、120/125（97.8/96.4/96.0%），原图已核，不能写100%。
- **评估**：18 CS本科生within-subject约1h；每页3选择题、5级UX。经验与分数ρ=.30,p=.23并非经验等效/知识差距消除；没有无NLE/无UI随机基线、pretest、真实新问题发现或长期调试。Kruskal-Wallis用于三页重复被试评价独立性值得谨慎，不能把p=.006当因果UI改善。Fig4少量示例及AppFig10局部path确有输出变化；AppFig10随机平均|Δp|.2474 vs traced .3105，随机路径还跨层顺序混乱，不是语义机制充分对照。
- **可复用/未答**：三个互补视图联动、数值claim回链、局部干预对照；未答matched clean/backdoor/OA、触发与sham、训练漂移控制，以及工具是否帮助发现可在独立数据验证的新机制。对本项目最直接的系统先例之一。

## CafGa — Customizing Feature Attributions to Explain Language Models (EMNLP 2025)
- **阅读**：官方10页461–470，正文、参考文献、附A–C完整。https://aclanthology.org/2025.emnlp-demos.32/ 。
- **问题/任务**：用户刷选任意不重叠词组、句/段预设，显式设template/input及目标answer evaluator，再计算group KernelSHAP；用户可改粒度重新看热图和扰动曲线。
- **估计量**：正文每个coalition10次重生成→Boolean answer命中率，KernelSHAP加权线性回归；AppB补充逻辑Entails/Contradicts/SemanticEquals实际单次回答+DeBERTa NLI。不能混成固定token概率/整段直接logprob归因。预算nsamples=tmax*rAPI描述未明确计入每coalition10请求，不能作为端到端效率定论。
- **发现先例**：§3/Fig5 GPT4o-mini餐厅案例，sentence粗归因→用户怀疑“仅供应tartare”是否被用→分离signature/only dish/顾客不吃生肉→新归因；作者称fidelity .77→.96并据此确认推理。应限定为分组干预下的行为支持；词组贡献不能证明模型执行了特定推理算法。
- **评估**：10参与者（6ML经验+4新手，全无归因经验）每人随机抽5任务类型，共SQuAD/Yelp/fewshot错误/HotpotQA/BARQA；另4专家每人10组比较human/MExGen/PartitionSHAP“最有帮助”，Table2平均64/28/8%。这是偏好和7级自评，非准确机制发现率；无新机制heldout干预和排错成功实验。
- **边界**：曲线横轴按删除/插入word比例而不是group数，值得复用；作者Limitations明确同样perturbation用于解释+评价的循环性、group单位问题和small-n。用户可视曲线迭代选择也会有选择偏差；不同group定义改变estimand并非单一方法faithfulness提高。本文对LLMCheckup“仅self explanation”的概括不准确，该系统亦有Inseq，不能照抄。
- **已有/可复用/未答**：交互→细化假设→目标行为扰动复核已做；语义分组与按原子单位比较曲线可复用。未答固定target+generated history控制、signed/complementary views冲突、未知触发/匹配训练对照后的新认知。§5.2直接引用Cheng2025 LLM Analyzer须追原文；MExGen/SequenceSalience/LLMCHECKUP已本review覆盖。

## TreeTracer — Exposing the Unsaid: Visualizing Hidden LLM Bias through Stochastic Path Aggregation (2026)
- **版本/阅读**：arXiv2606.19344v1，14页正文、参考文献、附A–D全部读；PDF页4–7、9、12、14原图/公式核对。官方摘要/水印标2026-04-24但arXiv编号为2606，日期元数据不一致；IEEE VIS2026官方接受列表列入，而此PDF仍写Under Review，不能称正式camera-ready全文已核。
- **问题/视图**：聚合随机生成文本树，比较两组ontology替换的输出偏差；成对树、合并split-Sankey、token和语义类别合并、原始路径/替换词回溯。每组15替换×15次采样，T=.8，只分析第一句；Stanza句法树经树编辑距离聚类，top6且最小簇1。辅助LLM赋语义类别，用户可修正ontology和类别，写入后续RAG。
- **最直接方法**：§4.4/PDF6–7在同一重建历史与同一目标token上，用各ontology的所有替换词重新计算概率，平均后展示PA/(PA+PB)。这是真实的固定前缀、目标概率对照先例；不是只比两段文本。该ratio不是发生次数的比例、不是显著性检验，也没有定位内部因果电路。
- **重要量纲边界**：PDF5选中均值Pselected=所选token条件概率之和/Nselected；所谓Pglobal用所有token概率之和仍除Nselected，作者有意保证后者较大。它可能大于1，不能当归一化全局概率/概率质量。词的subword概率用几何均值，不是整词联合概率；Sankey不守恒已由作者承认。颜色/宽度不应直接照抄为概率流。
- **发现/评估**：五类案例：性别职业、身份毒性、CEO角色提示、医疗安全/有害场景、食物/城市句法。GPT2-XL、Apertus8BBase(文中又称instruction-tuned)、70BInstruct不同规模/训练/量化不能构成纯方法或训练效应比较。N=7 CS参与者（4研究生3本科）约1h，固定先generAItor baseline、再引导TreeTracer、再自由探索；SUS76.9、访谈与自评，不是对照随机化发现准确率实验。baseline beam与本法stochastic aggregation的数据生成也不同。
- **最强反证/局限**：附A预设提示把social worker放入FemaleOccupations等，taxonomy部分来自设计者，不能说模型内部自然涌现了这些类别。ontology/辅助模型偏差、输出采样、微小prompt变化稳定性未解决；没有matched clean/backdoor/OA、sham触发或训练漂移对照。五例能说明工具使行为差异可见，不能证明通用隐含偏见机制、未知trigger发现成功率。
- **已有/复用/未答**：隐藏候选/随机路径比较→找差异→同历史token概率复核已做。复用任务导向聚合、原始例回溯、同prefix评分；开放的是新机制发现与独立干预/留出样本验证。直接前例generAItor与Revealing the Unwritten继续纳入，其他通用输出dashboard不扩。

## LLM Analyzer — Understanding Large Language Model Behaviors through Interactive Counterfactual Generation and Analysis
- **版本/阅读**：arXiv2405.00708v2，2025-08-07，15页正文、参考文献、附A–D全部读；Figures1、3–11原图核对（PDF1、5–8、14–15）。正式TVCG32(1):846–856，2026，DOI10.1109/TVCG.2025.3634646，IEEE元数据已核，未声称正式排版11页与此v2逐字相同。
- **问题/任务**：把解释当交互过程。Why/Why not、What if、How to be that、How to still be this四类问题；T1定粒度/替换/抽样→T2看反事实集→T3归因定位→T4按多个片段存在性分组→T5选具体反事实验证并形成新假设，返回前步。此完整流程不能再作为本项目首创。
- **方法/视图**：依存句法区分可删、不可删、并列dummy，合并不可删父子，过滤不满足结构依赖的组合；用户折叠/展开粒度、改叶子替换。每个反事实5次生成，经CONTAIN/STARTWITH/EQUAL或LLM entailment evaluator转Boolean命中率；表头KernelSHAP热图、依赖线、列频率/结果直方图、按片段group-by、箱线图刷选与完整文本/回答回溯。目标是自由重生成经判定器的行为率，不是固定生成token logprob。
- **可视发现证据**：§6.1/PDF6–7明确标hypothetical use case：医疗QA中根据归因选症状/孕周，聚合后发现例外，再换22/30/38周。它是演示，不是真实临床用户发现；本文医疗判断也不作为本review医疗事实。§6.2实际8参与者：均先看归因，再用分组/筛选检查假设；P3移除两个表面无关片段找到错误回答，参与者回读完整文本确认。这是“工具支持认知/假设发现”的实证过程观察。
- **评估**：5数据集各1000句、总5000句；LanguageTool统计未引入新语法错误97.2%、每句约46扰动、laptop parsing/sampling .7s。不是端到端LLM查询耗时。Polyjuice大样本基准未做，附B只10例；正文50iterations vs Fig9横轴300不一致。N=8用户（7懂XAI）单HotpotQA、20min任务/60min总，之后6专家各45min；5级自评好用4.12、what-if4.62、why3.75、still-be3.62。无随机UI基线、机制真值、heldout新案例发现率或修复效果；§7承认未在实际debugging/decision-making情境评估。
- **统计/归因边界**：附D仍写全2^M coalitions标准SHAP核和伪逆，但实际只采句法合法依赖组合，缺少受限game定义、被排除coalition估计及端点无限权重/效率约束说明。可能共线、伪逆能给数值不等于识别标准Shapley值；不要直接继承公理保证。LanguageTool没发现新错不是语义可用性保证；附A自身有“losing money in than one way”等不自然句。每coalition5次的行为率不确定性未进归因/箱线图置信区间。
- **目标判定边界**：附C只判CONTAIN(Donald)，且用“信息不足说不知道”提示试图避免先验知识；这不保证正确推理/无参数记忆，拒绝/引用词同样可能命中。用户看数据后重定义组合与假设，也不等于独立验证。
- **已有/复用/未答**：归因+具体反事实、联合条件过滤与迭代假设已做，粒度与结构约束、人读原始例很值得复用。本项目仍可把互补归因视图作为发现仪器，检验有行为匹配/训练漂移/sham对照后能否发现可在独立样本及内部干预验证的机制；无需强加新归因算法要求。

## generAItor — Tree-in-the-Loop Text Generation for Language Model Explainability and Adaptation (2024)
- **版本/阅读**：arXiv2403.07627v1，2024-03-12，实际32页全部正文、参考文献、附A与最后接收信息读完；Figures1、3–11核心原图核对（PDF9、13、15、18–22、31），32页接收信息核对。正式ACM TiiS14(2) Article14，DOI10.1145/3652028；此作者稿的J.ACM37(4)/Article111/Aug2024是模板残留，不能作正式出版元数据。pdftotext报xref744重建警告，但成功提取，核心末页与图均可渲染。
- **问题/任务/交互**：解释生成候选及概率、提示对比、用户编辑/选树分支与微调。T0模型/采样配置，T1生成树概率/重复结构/语义聚合探索，T2引导生成，T3比较prompt替换树，T4编辑并训练。树为中心，语义UMAP颜色、RoBERTa情感、ontology Voronoi、词表和UpSet、原文与模型snapshot联动。对比模式关闭temperature，控制的是解码随机性，并不控制不同prompt导致不同历史。
- **认知发现先例**：§6.1/PDF16–19从John/Jessica职业续写→观察后续“疑虑”→新prompt探索动机/恐惧；再加入even/however，发现目标属性出现位置不同以及not会翻转含义，固定位置/只计词频的bias评估会丢失这些关系。这确实是工具驱动形成研究假设的具体例子；不是证明新的一般偏见机制。
- **真实使用证据**：§6.2，6 nonexpert GPT2Base配对分析15–25min（总30–45），4计算语言学者RedPajamaInstruct3B否定+Base3B偏见开放任务35–55min（总50–70）。语言学者两人积极产生新prompt/假设、两人需引导；wordlists/UpSet少用，提示对比常用。Fig7为自评直方图、无随机比较/发现准确率。微调功能是在用户反馈后补的，未在该用户研究中试用。
- **训练/验证确实做了什么**：§6.3 GPT2Base局部目标概率随少量微调改变；全局IMDB 25k/25k，100个从相同基础模型开始、训练量20至2000的run，评估训练集及完整留出集困惑度，Fig8b有留出下降趋势。这能支持小样本适应可行，不是把可视发现的bias机制修复后独立评估；本地hooked/able变化也不能独自证明全球domain适应。Table2去逗号句two steps deaf=.010252，与正文/Fig8a带逗号句=.000834不同，不能当同配置数值一致或无条件矛盾，保留输入/协议区分。
- **关键边界**：§3.2/7.1称BST不丢重要信息、不诱导rationalization过强；§6.2专家恰指出平坦分布时top-k漏掉等可能候选。树是解码输出决策图，不是输入token到目标的因果归因或神经电路；UMAP/情感/ontology为外加代理。固定top-k与外部偏见词表不能证明模型整个分布/内部语义结构。AppA自称层logits/embedding、BERTLarge层8–11 squeeze/duplicate成2048维说明欠清楚，不把公式描述当已复现实现。
- **已有/可复用/未答**：可视探索→生成假设、可编辑历史→重新生成、少量微调和独立域困惑度各部分已有。§7.2明确统计验证联接、实际模型debugging和跨模型并列比较是后续工作，作者避免给局部树附通用语言统计来防过度概括。可复用原始候选透明、prompt对比、探索日志/模型snapshot；本项目科学空间在匹配对照中发现并独立证伪/证实新机制，不是“首个支持假设形成的UI”。
