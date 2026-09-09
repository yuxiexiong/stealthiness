# Generation 域全文证据卡

读取日期：2026-09-09。本文依据原文而非摘要生成。`full_text_read` 只有在正文、附录、参考文献逐页非截断读取完成之后标记。核心图表的原图核对另记。全文公开不代表本项目已经运行作者代码。已有工具允许借用，不能据此否定以可视分析发现新认知的项目。

## contract

- **题名与来源**：Giang Nguyen，The Attribution Contract for Generative Language Models，2026，arXiv:2605.23080v3（2026-08-26）。[官方全文](https://arxiv.org/html/2605.23080v3)。本地 PDF `pdfs/contract.pdf`，文本 `texts/contract.txt`，15 页。
- **阅读范围**：PDF 1–15 页全部，§1–7、复现声明、参考文献；无独立附录。首次较长输出发生截断，随后完整显示各页面内容，缺失 p7–9 单独非截断补读。图1/2原图已核对，见下方已完成原图记录。
- **问题**：生成 token 同时是前一步输出和后一步输入，“对输出做归因”的目标并不唯一。
- **方法**：SCOPE 五元组：目标分数、固定量、解释对象、生成过程、归因特征。区分局部 next-token、固定前缀仅归因 prompt、固定 span 的序列似然；另列扩散模型三类契约。
- **实验**：OLMoE-1B-7B，Natural Questions Open 验证集前120题，greedy 最多16 token，IG 零嵌入基线、右端 Riemann 积分、A100/bfloat16；路由实验60题。Steerling-8B Base/Instruct作跨过程比较，另有 Qwen1.5-MoE 与 Mixtral pilot。
- **关键图表与结果**：图1 p2 同一 token 不同归因契约；表1 p4；图2 p8 为 MoE 重算／固定路由的IG完整性残差；表2 p9 是扩散模型积分步数；表3 p10是目标分数与固定条件变化后的秩差异。作者在该短QA设置报告前缀29%、问题25%、模板36%、系统10%的绝对归因质量份额，不能泛化到所有LLM。IG积分的概率与log概率归因排名可能不同；不要误推为同一点的普通梯度归因必然发生排序变化。
- **局限**：§6明确样本小、仅IG，过程比较跨不同模型并未隔离架构/规模/训练数据；重算与固定路由都未达到零残差，pilot缺少同等完整量化。完整性与删除/插入测试的相关性弱，不能单用一个指标认证解释。§3.1提出固定前缀契约，但它不恢复自然生成经前缀中介的总效应。
- **done**：固定前缀／允许前缀被归因的歧义、原回答重打分与重新生成不同、self-attribution fallacy 已明确提出并局部实验说明。
- **reuse**：每张图写清测量契约的规范，可直接用于本项目图注与实验日志；代码未在本轮运行，未核实独立代码许可证。
- **candidate_gap**：已知触发条件在匹配干净/普通后门/OA模型中的动态视觉模式是否稳定，以及这些模式与实际干预结果如何对应，本文未测试。
- **相关追踪**：§5.2导出 CAGE/HETA/ContextCite/ReAGent/Jacobian；DIG/SIG、忠实性和不可能性结果交 foundations 域。参考文献的新作 Scaling inherently interpretable language models 属于可解释模型架构，候选需额外判断范围，不能仅因其用作扩散模型便纳入方法比较。
- **项目定位**：为可视分析提供语言和边界；不要求本项目另造算法。采用其规范不能替代对从图中提出的机制解释做验证。
- **未闭合**：核心图原图核对已完成；作者跨模型路由pilot的原始数据和实现未核验，结论限于论文报告。
# PECoRe — Quantifying the Plausibility of Context Reliance in Neural Machine Translation

- **元数据/版本**：Sarti, Chrupala, Nissim, Bisazza；ICLR 2024；arXiv 2310.01188v2 (2024-03-13)。[官方全文](https://arxiv.org/pdf/2310.01188v2)。本地 `pdfs/pecore.pdf` / `texts/pecore.txt`，29 PDF页。**全文阅读**：1–29页，正文、全部参考文献和附录A–J逐页非截断读完。核心图表另见原图核验记录。
- **问题**：自然生成时，哪些输出词实际依赖额外上下文，以及哪些上下文词促成这些预测；人类认为有用的上下文不等于模型实际使用。
- **方法**：§3、Fig2 (p4) 的CTI→CCI两步。CTI在相同已生成前缀下比较有/无指定上下文的预测分布，候选KL、似然比、P-CXMI；CCI对被选输出位置的对比分数做梯度或其他归因，定位上下文线索。不是任意next-token高概率即高贡献；输出分布KL与某实际token的有向影响也不是同一对象。
- **实验**：SCAT+过滤后250例、DiscEval-MT 400例；OpusMT小/大和mBART50，源上下文/源+目标上下文，多语机器翻译。Fig3/4及附录G/H比较CTI/CCI与人工标签；KL总体较稳，人类标注只涵盖特定语用现象，会漏掉模型使用的其他上下文。正确翻译中很多本来不需上下文，单看正确率不能说明使用了上下文。
- **图表/观察**：Fig2流程；Fig3/4端到端定位比在已知context-sensitive词上归因更难；Table2和附录I可视案例揭示错误指代、数字格式复制、词汇复用。**附录I.1 (p21) 已明确指出**：上下文改变先前翻译词后，经词性一致性使后继词变化，后继位置却不会被局部CTI标记。这是前缀中介影响漏检的已有例子，不能包装成我们首次发现。
- **超过MT的实际覆盖**：附录J (p28–29，Table8) 已在Zephyr 7B Beta decoder-only模型上展示故事约束、人物年龄/幻觉、系统指令和退款RAG；本文不是仅支持encoder-decoder或仅能用于翻译。框架主体明确通用text generation，主要定量评估仍是MT。
- **局限/疑点**：附录A区分plausibility与faithfulness；人工标签不是唯一合理解释；附录F输出-标签对齐有误差；best-attention-head是oracle对照，不是实际无监督选法。**数值一致性未闭合**：Table1与附录Table3部分mBART数值不一致（例如默认SCAT BLEU23.8 vs30.9、OK .26 vs .52）；本报告不依赖该组绝对值，复用时需查作者代码/勘误。未独立运行代码。
- **done**：输出敏感位置→输入线索→词级彩色展示；适用于decoder-only/RAG；用可视分析发现新上下文依赖现象。**reuse**：两步测量、自然生成评估、上下文选择、Inseq集成、示例级阈值；许可证/当前API未核。**candidate_gap**：已读本篇没有匹配clean/普通后门/OA、触发/非触发/sham的跨模型可视研究，也没有完整前缀中介效应估计。不是“它没有所以全球空白”。
- **项目定位**：必须直接承认和复用此工作；核心可来自触发行为的可视观察、受控对照与可证伪认知，而非重复两步流程或被迫另创归因公式。参考文献扩展：Vamvas/Sennrich contrastive conditioning、Fernandes context usage、Ferrando ALTI、Vafa sequential rationales保留为历史方法链路，本轮不作独立新颖性结论，也不声称逐篇全文审查。

# Inseq — An Interpretability Toolkit for Sequence Generation Models

- **元数据/版本**：Sarti, Feldhus, Sickert, van der Wal；ACL2023 Demo，pp421–435；[官方全文](https://aclanthology.org/2023.acl-demo.40.pdf)。15 PDF页；`pdfs/inseq.pdf` / `texts/inseq.txt`。**阅读**：正文、参考文献、附录A–F、全部15页非截断读完。注意官方PDF题名页四作者，其他论文引用可能列六人；附录A说明Nissim/Bisazza支持贡献。
- **问题/方法**：序列生成逐token归因需要动态输出前缀、encoder-decoder/decoder-only、不同分数与聚合。论文v0.4.0支持Captum/HF方法、源→输出和前缀→后继矩阵、prob/logprob/entropy/contrastive probability等目标、强制解码或自然生成、subword/词/句聚合、PairAggregator两次归因比较、HTML和终端展示（§3–4，Table1，Fig1/2）。
- **实验**：§5.1 49个土耳其语职业翻译性别案例；MarianMT几乎都输出男性代词，归因与美国职业性别统计比较。IG300步、L2聚合，多子词职业只取首子词是分析选择。Table3 (p6) M2M100错误her→zijn显示先前生成词leraar的影响，修改为性别中性leerkracht后变化。§5.2 Counterfact 1000事实句、GPT2-XL，Contrastive Attribution Tracing以Layer Gradient×Activation近似定位事实信息，Fig3与Meng causal tracing结果比较。
- **图表**：Table3明确展示源token和生成前缀两套矩阵，以及概率差值；Fig3层级图；附录Fig5 p14提供性别对比矩阵；Fig6 p15代码和多层示例。不能把概率差46%解释成因果贡献46%。
- **局限**：§5.2/p7明确CAT缺少patching的因果保证；Broader Impact要求逐例验证；没有显式归因质量评估功能；8bit比较只是有限手检；性别二元化和数据偏差不能用于确定性社会偏见结论。附录F中LRP/ALTI/patching等属于计划功能，不能当作论文版已实现。
- **done/reuse**：生成输出逐词着色、源和历史归因、对比矩阵、以归因案例发现模型行为已有成熟工具先例。可借其可视呈现/数据结构/方法比较；未运行源码和核许可证。**candidate_gap/定位**：工具存在不取消本项目价值，但只做heatmap软件包装贡献不足；匹配触发行为的动态分析、规律和反事实验证才是待检验研究内容。未见本篇专门OA或训练后门对照。
- **追踪**：Alvarez-Melis/Jaakkola2017黑盒seq2seq causal framework已全文纳入；Vafa2021 sequential rationales登记为历史方法链路；Ecco/LIT交由visual域；归因基础、faithfulness交由foundations域。

# MIRAGE — Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation

- **元数据**：Qi, Sarti, Fernandez, Bisazza；EMNLP2024，pp6037–6053；[官方全文](https://aclanthology.org/2024.emnlp-main.347.pdf)；17PDF页，`pdfs/mirage.pdf` / `texts/mirage.txt`。**阅读**：17页全文、参考文献、附录A–E及Table6–12均非截断读完。
- **问题/方法**：RAG答案哪些token受检索context影响、来自哪个context token/document；扩展PECoRe。§3 Fig3 p4：CTI为固定相同前缀下有/无context分布KL；CCI对有context条件下实际token与无context所选替代token的概率差计算输入embedding梯度L2；阈值选线索，再将token级document集合并为句级引文。L2非负，不保留支持/抑制符号；两步也不等于自然生成总因果效应。
- **实验**：XOR-AttriQA，CORA(mT5-base)多语QA；原4720test因原document顺序未知，用最多200次shuffle找到可自然复现答案的1144例（校准142），主表在该筛选集。Table2：MIRAGE CAL Top5%与人工引文一致86.7%，EX83.4%，不是因果真值评估。附录B全数据强制解码的结果单独报告。ELI5用Zephyrβ7B、LLaMA2-7B-Chat三个随机seed，先self-citation生成、移除citation再归因同一答案；TRUE NLI评估是代理且作者承认脆弱。
- **结果/图表**：Table3 p8 Zephyr citationF1 30.6→45.6；LLaMA26.0→27.6 (Top5%)，Top3为25.1没有全面提升。Fig5 p8单输出字符9对应输入稀疏峰；Fig6短句质量更低有评估协议混杂。Table4/5 p9对比漏引文和prevent/cancel混淆；附录Table11/12 p17对同名词ScienceABC/模板词Document的词面复用，归因为某文档并不意味着该文档支持整句事实。
- **局限/关键解释**：作者p10明确归因方法影响faithfulness，主要定量验证是人工或NLI plausibility，不能转述“模型内部方法天然忠实/已经证明因果”。没有>7B、广泛域/长上下文验证。CTI统计阈值依赖例内分布/长度；总成本2F+|CTI|B不是一次backward。文档逻辑支持与模型词级使用是不同问题；错误AA可能仍反映词级依赖，不应强行合成一种ground truth。
- **done**：完整两步RAG输入-输出可视化、词级彩色线索与文档聚合、用可视案例发现语义支持和词面使用不一致。**reuse**：已有基线、阈值/聚合消融和现象对照；[作者代码](https://github.com/Betswish/MIRAGE)仅确认链接，未运行/核许可证。**candidate_gap**：本文没有受控OA/后门/普通上下文差异的跨生成步研究；聚合后解释失真是项目应验证的已有风险，不能称首次。
- **追踪/疑点**：ContextCite已纳入本域；Phukan2024 “Peering into the Mind of Language Models”(2405.17980)语义相似归因已全文纳入（见本文件Peering卡）；Alghisi2024 dialogue adaptation中history saliency为旁邻候选。本文个别引用Gao2023a/b与任务指代混用，使用方法正文为准。
# CAGE — Explaining the Reasoning of Large Language Models Using Attribution Graphs

- **元数据**：Chase Walker, Rickard Ewetz；arXiv2512.15663v1，2025-12-17预印本；[官方全文](https://arxiv.org/pdf/2512.15663v1)。21页；`pdfs/cage.pdf` / `texts/cage.txt`。**阅读**：全部21页、参考文献、附录A.1–A.5完整读完；原图核验 `figures/cage-p5.png` (Fig3/Eq1–3)、`cage-p19.png` (Fig11/Table5)。
- **问题/方法**：普通prompt-only逐步归因舍弃生成→生成关系，晚期答案看似不再依赖原始证据。§3保留prompt与全部先前生成token，base attribution构造矩阵T，max(T,0)后逐行归一为A；A严格下三角；总传播Y=A(I−A)^−1为所有路径乘积之和，再取目标行和prompt列，支持任意输出子集。Fig1–3直接展示graph→上下文归因。§3.4作者承认线性传播只是结构化抽象，非模型内部精确计算。
- **实验**：Llama3的3B/8B、Qwen3的4B/8B，附录14B消融；5 base methods(Pert.,CLP,ReAGent,IG,Attn×IG)，句级聚合；Facts(FEVEROUS采样9句/输出3句/解释2句)、Math(MultiArith+SVAMP加固定“Unrelated sentence”干扰)、MorehopQA。Math AC评估250IG/500扰动例；Facts/QA等500例删除测试；附录归一消融100例。
- **结果/图表**：Table1 AC 17/20组合改善，最大134%/平均40%是AC相对变化，**不是所有faithfulness都40%**；主Table2/3删除代理RISE/MAS较row基线都改善，平均11%/16%。Fig4/6/7用事实复用追踪、Fig5/8/9数学、Fig10/11多跳上下文显示跨步影响。AC偏好相关句近似均匀分配，本身是任务先验，不是独立因果ground truth；Pert EOS删除与评估删除同型，需防对测试算子的偏好。
- **方法边界**：论文“causality”定义=边只向未来，**时间有序DAG不足以证明边权是可识别因果效应**；任何base method不可能凭归一/路径相乘自动获得因果保证。截断负数丢失抑制，逐行归一丢失跨步绝对大小；图中路径权重不是总自然间接效应。未来工作明确提出signed graph/其他归一/高阶传播，因此“有符号传播”已有提案，不能直接宣称首次。
- **open_issue**：Eq3印刷形式不清楚是否使用逐步更新A，单次原始A乘A仅含两跳，Eq4才清楚定义完整路径和；复用以Eq4并核实现为准。Table5原图确认Attn×IG去row归一(.346/.441)优于完整(.351/.444)，与正文“放松任一约束一致降低”不符。Table6部分相等而非严格改善；Fig8目标编号9与列出最终答案13不一致。代码p11仅承诺接受后公开/审稿supp，当前仓库/许可未核，不写复现成功。
- **done/reuse**：跨生成步矩阵、DAG、按路径传播回prompt、可视化识别缺失早期证据已做；可作为生成历史方法对照或可视结构。**candidate_gap/定位**：特定训练后门/OA与匹配clean模型中直接/中介/抑制变化及受控效应验证未在本篇回答；已有方法足以开展可视研究，无需另造DAG才能立项。
- **引用链新增**：CLP2212.14815、Attribot2411.15102、GiLOT、Multi-level Explanations2403.14459、Learning to Attribute with Attention2504.13752、Counterfactuals as means evaluating faithfulness2408.11252直接候选；基础DIG/SIG等分给foundations，非直接竞争不扩大无界。

# HETA — Hessian-Enhanced Token Attribution: Interpreting Autoregressive LLMs

- **元数据**：Pramanik, Maliha, Bastian, Jha；ICLR2026印刷版/arXiv2604.13258v1 (2026-04-14)；[官方全文](https://arxiv.org/pdf/2604.13258v1)，39页；`pdfs/heta.pdf` / `texts/heta.txt`。**阅读**：1–39页完整正文、参考文献、附录A1–A6、Algorithm1、Table1–18读完。已核原图 `heta-p6.png`、`heta-p21.png`、`heta-p25.png`、`heta-p39.png`，下列关键矛盾不是PDF提取错位。
- **问题/方法**：目标token条件下，attention-value rollout门控M，Hessian/HVP曲率S，加单token masking导致的完整预测分布KL I；非负Attr_i=M_i(βS_i+γI_i)。主实验β=γ=.5；低秩64、512window50%重叠、部分层近似。**不是CAGE式跨生成步传播**；附录A6.4只是对固定多token目标逐步分数求和。非负组合不提供支持/抑制符号，mask-KL不是实际目标token的有向prob影响。
- **实验**：LongRA/TellMeWhy/WikiBio，GPT-J6B、Phi3-medium14B、Llama3.1-70B、Qwen2.5-3B；新增2000 NarrativeQA+SciQ干扰/支持拼接，以first answer token为目标；主标签来自GPT4o/GPT5Thinking交集，非全部人工标注，附录称200例双专家审计。比较ContextCite、IG、PML、TDD、rollout、fAML、Progressive Inference、SEA-CoT、ReAGent/GiLOT。含Soft-NC/NS、DSA、主动被动改写、decoding grid、component ablation、低秩/窗口成本，附录报告patching、counterfactual、下游filtering等。
- **已有可視覆盖**：Fig2d/3d词-词矩阵；Fig4–6目标词slice/friends/bush的输入热标记；token/多token定位以及曲率可视方法已存在。**reuse**：可作为应核实的方法候选、固定目标和近似成本对照；[作者代码](https://github.com/VishalPramanik/HETA)未执行/许可证未核。原图颜色不是机制证明。
- **严重open_issue 1（尺度）**：p6和p26明确归因总质量归一为1、DSA=支持子集质量−干扰质量；对非负HETA应在[-1,1]。Table1报告4.25–5.10，Table5/11/14–17也>1，正文未给额外倍率。不能直接以这些数字确认其state-of-the-art/改进倍数。
- **严重open_issue 2（数学）**：LemmaA2.1 p21把min|λ(H)|≥μ推出|vᵀHv|≥μ||v||²，缺少H半正/半负定条件；H=diag(1,−1),v=(1,1)即反例。TheoremA2.2从绝对误差上界直接减非负correction，没有证明误差方向/不过补，不能据此承诺HETA比两基线必然更好。A2.4稳定性只界S/I变化而遗漏门M随输入变化。p25把d×Td完整行块entrywiseL1到Frobenius常数写sqrt(d)，一般需要sqrt(d·Td)，除非另有结构限制。
- **严重open_issue 3（对象/实现）**：p5/Algorithm1用block-supported HVP再取相同block，只含H_ii；p19理论Si定义完整block行L1，含H_ij，并非同一对象，Hutchinson绝对HVP平均也不自动等于entrywiseL1。正文70B单A100计算未交代足够的precision/offload细节；runtime/memory不能未经实现验证接受。
- **其他报告边界**：Table3里ReAGent并非始终second best，正文有过度概括；p29 Full sensitivity .025高于KL-only .022，不能说所有指标最好；p36 MoRF说明“删除最重要后概率曲线面积越大越好”方向与通常删除faithfulness定义冲突（Table18单位/定义需核）。SEA-CoT所引Palikhe2025题名为survey，具体baseline来源不清；ROAR/KAR声称已做但不提供表/训练协议，不可凭未列数据替代验证。
- **结论/项目影响**：HETA是必须纳入的直接近邻，已提出“曲率+KL+内部流门控”的token归因和可视化，也声称干预验证；上述未闭合问题限制我们接受其数值和强因果理论，**不能把论文存在等同方法可靠，也不能因有问题就从related work删除**。可视化研究需保留替代方法/有符号分数并独立检查。没有证据本篇回答了OA跨模型触发传播。
- **追踪**：早期PML2405.17980线索经核对应Peering，现已全文纳入；TDD、fAML已纳入，GiLOT由foundations全文审查。Progressive Inference只作本文baseline来源追踪，不以其题名作已验证方法结论。

# 已完成原图核验记录（持续更新）

- PECoRe：Fig2 p4固定同一prefix与两步关系；p21中介影响例；Table8 p29真实decoder-only案例。
- Inseq：Table3/Fig3 p6源/目标前缀矩阵、差值色标和内部层图。
- MIRAGE：Fig3 p4 KL-CTI与prob-difference梯度CCI；Table11/12 p17词面复用≠整句支持。
- Contract：Fig1 p2不同contract热图，Fig2 p8MoE residual并非单调改善到0，Table3 p10目标/条件变化的排名差异。
- CAGE：Fig3/Eq1–3 p5完整传播意图；Fig11/Table5 p19消融不全支持正文“始终”。
- HETA：p6 DSA定义/数值，p21 Hessian下界，p25范数常数，p39成本/指标表及词级图；原图确认上述疑点确实印刷存在。
# ReAGent — Recursive Attribution Generation for Language Models (2024)

- **全文证据**：`pdfs/reagent.pdf`，arXiv 2402.00794v2，2024-02-07；12 页正文、参考文献、附录逐页完整读。Figure 4 / Table 4 的 PDF 第 7 页已渲染核对（`figures/reagent-p7.png`）。
- **问题及对象**：在不能反传目标语言模型的条件下，为一个给定 next-token 预测寻找输入词的重要性；模型必须提供目标 token 概率，另需 RoBERTa 替换模型，因而不适用于只返回文字、不返回概率的 API。
- **方法**：随机初始化重要性，递归选择约 30% 输入词并替换，以实际目标词概率的变化更新被选及未选 token 的重要性 logits，再作 softmax。终止条件是替换低重要性的 70% 输入后目标仍在 top-3。最终非负归一化分数不表示带符号效应或一个触发器的因果效应份额。概率是扰动响应的测量量，不能把原始 next-token 概率本身当输入归因。
- **实验与结果边界**：6 个 GPT/OPT 模型，LongRA（模型依赖筛选后 37–149 条）、TellMeWhy 200 条、WikiBio 238 条；OPT-6.7B 用的是 KoboldAI Erebus 微调版，不应写成六个同质 vanilla 模型。输入通常 35–50 token，并非长上下文大模型验证。SoftNS/SoftNC 使用全词表 Hellinger 距离与随机基线相对的 log 分数，不能读成原始 [0,1] 忠实度。18 项聚合比较大多领先；GPT-J SoftNS 中 IG 优于 ReAGent，不能写无条件全面优越。Greedy rationale 比较中 antecedent 命中 .930 vs 1.000，No-Distractor .710 vs .667，各有胜负。
- **图表已做**：Figure 4 的 Tennessee/city 与干扰句比较，Table 4 的 Super Mario→Nintendo，都是输入位置热图解释固定目标 token 的直接前例。
- **局限与未闭合疑点**：最大递归次数正文出现 1000、3000，附录出现 5000，Table 7 caption 与 >4000 实际均值也不一致；终止失败样本是否排除及总体查询成本须复现澄清。替换策略有时随机方法得到更好 SoftNC 但需数千步。Table 8 标题 Wikitext 与文中 WikiBio 名称不同。mask/delete 的分布偏移和单目标 token 的选择限定解释意义。
- **已做 / 可复用 / 未解 / 定位**：黑盒条件下输出目标位置→输入重要性已做；可作为有 logprobs 的递归扰动基线与噪声可视化参考；不提供跨步中介识别，不检测 OA，不给普遍因果保证。适合基线候选，查询预算及协议不一致须先解决。

# Interpreting Language Models with Contrastive Explanations (Yin & Neubig, EMNLP 2022)

- **全文证据**：`pdfs/contrastive.pdf`，官方会议版，15 页正文+参考文献+全部附录完整读；PDF 第 1/6/12 页 Table 1、Figure 3/Table 4、Table 6 已渲染核对。
- **问题**：解释“为什么预测目标词而非一个明确 foil”，使显著性对应具体语言对比，而非仅解释单词高分。
- **方法**：§3 对 `q(target)-q(foil)` 求梯度范数、gradient×input，或计算双重 erasure 差异；q 可取 logit。梯度范数无符号，dot-product/erasure 可有符号。对比共用相同前缀；附录 A 已扩展到机器翻译的源输入与已生成目标前缀，不是只研究分类。
- **实验**：GPT-2 XL / GPT-Neo 2.7B，BLiMP 中 12 个现象范式、五类语言现象，用规则生成的相关 token 标签验证；提升在长依赖及预测正确的情形更明显，错误预测并非稳定获益。用户研究 10 位 ML 研究生、20 对词、每人 10 对且每对 40 次，共 4000 判断；基线无解释正确率 61.38%，G×I 64.00→65.62%，erasure 63.12→64.62%；erasure 有用性 46.50→64.88。受试者/词对混合效应分析支持部分增益；Appendix D 的 AccIncorrect contrast effect 不显著（p=.460），不能泛化成所有错误都更易诊断。
- **作为核心研究手段的可视化证据**：Table 1 区分 barking 对 crying/walking 的热图；§6 利用约 8 亿条 attribution、8 张 RTX8000 48h 对不同语言对比聚类，发现与词向量近邻不同的句法/语义分组。已有“归因可视化/统计→语言现象发现→用户任务验证”路线，研究价值并不要求另造基本梯度。
- **局限**：两个英文模型、自动标签不等于因果真值、人工研究样本有限、foil 决定被问的问题。高分不等于生成机制的独占原因；未自由重生成以识别前缀传播。
- **已做 / 可复用 / 未解 / 定位**：输出目标/替代词差异→输入和历史词热图已做；可复用带符号对比和解释辅助预测的评测。触发器研究应明确 foil、固定哪些历史，并用干预验证热图发现。OA 可作为额外压力测试；本文未覆盖。

# Jacobian Scopes: token-level causal attributions in LLMs (2026)

- **全文证据**：`pdfs/jacobian.pdf`，arXiv 2601.16407v4，2026-06-15；25 页正文、参考文献、A.1–A.9 完整读；PDF 第 5 页 Figure 3、末页 Figure 16 已渲染核对。
- **问题与方法**：令 `J_t=dy/dx_t`，scope 的输入重要性是 `||vᵀJ_t||₂`，即指定输出方向的一阶局部最大敏感性。Semantic 取目标词反嵌入向量（logit 梯度范数）；Fisher 取 `F=Wᵀ(diag(p)-ppᵀ)W` 主特征向量；Temperature 取 `y/||y||`，即最后隐状态范数的梯度。一次 VJP 并避免参数梯度，Fisher 另有特征向量求解成本。非负强度不能读成促进/抑制符号。
- **数学边界**：Fisher rank-1 是对局部 KL/Fisher 拉回量的近似；Figure 16 主方向平均解释约 37%（1σ 25–50%）的 F 谱质量，不能称完整分布效应已精确解释。被 J 拉回后贡献还取决于不同方向的 Jacobian，F 主特征向量不必是输入端最大贡献方向。Temperature 把 `z=||y||z_hat` 的范数解释成逆温度，在数值预测的局部二次 logit/固定曲率条件下可关联方差；输入同时改变方向/曲率时不是纯不确定性分解。作者“causal”是线性化扰动意义，不是有限文本干预或跨生成链的识别。
- **实验与图**：LAMBADA、IWSLT 各 1000 条，6 个 1B–4B Llama/Qwen/Gemma 模型，teacher-forced 最后非标点金标词，删去/置零最高分 5/10/20% embedding 的 AOPC。对比 gradient×input 与 100-step IG。很多模型 Fisher/Temperature 较好，但不是所有方法/任务都严格领先。§4 展示翻译、政治标签和时间序列热图：Figure 3 Dante 原文一词对应与短语重述之间出现不同输入扩散，体现可视化提出机制假设的用途；它不是机制验证本身。
- **验证疑点**：AOPC 称 drop 却采用负数越低越好，复用需核清符号；§4 说演示皆用 1B、Figure 3 caption 标 3B。IG 的零基线在小 alpha 附近产生 attention sink 可说明该基线风险，不能推出 IG 普遍不可信。摘要式“首个完整分布归因”不能照抄成优先权结论，PECoRe 的 KL 梯度等已有邻居。
- **已做 / 可复用 / 未解 / 定位**：不同输出度量、同一文本热图的解释差异已经直接研究；轻量 VJP 和明确 scope 可复用。固定生成位置与历史条件的局部敏感性，不覆盖自然重生成的总效应。可视化揭示变化模式可作为核心研究，但应给样本级及跨样本干预证据，而不能只展示几张图。

# Feature Attribution of Multilingual Chain-of-Thought Reasoning (2025)

- **全文证据**：`pdfs/multilingual_cot.pdf`，arXiv 2511.15886v1，2025-11-19；14 页正文、参考文献及附录完整读。PDF 第 7/12/14 页热图、Figure 11 的斜率图和附加结果已渲染核对。
- **问题与方法**：研究 CoT 各步骤及输入词在多语言数学题答案中的重要性。直接复用 ContextCite（步骤作为 context，32 次消融）与 Inseq saliency（token 级后聚合到步骤），没有要求原创基础归因算法。这是本项目“以归因可视化为核心研究手段”的直接方法论前例。
- **实验**：Qwen2.5-1.5B Instruct，MGSM 250 测试题×EN/FR/DE/BN/ZH，8-shot；structured CoT 与无 CoT、英语 pilot 四配置。准确率 EN59.2/FR48.8/DE37.6/ZH35.2/BN3.6%。归因只包含 regex 可解析生成：EN54%、BN64%、DE74%、FR92%、ZH98%；256-token 上限及 Bengali 平均 250.7 token 是重要混杂。token 级只用 EN/FR/DE 各 8 个训练题做 negation/distractor，没有大规模独立测试。
- **已发现现象及边界**：多数最后推理步骤对固定最终答案的概率最重要；正确/错误斜率在英/孟/德更高错误斜率，但法/中相反，不能写“错误均更依赖末步”。这可能有前缀答案已显露、固定后续答案与串行传播的条件效应，不等价末步真正独立完成推理。目标概率接近 1 在错误答案中也出现，再次说明置信度不等于正确性。Appendix C 另测 DeepSeek-R1-Distill-Qwen1.5B 英中，末步主导明显变弱，故不是跨模型不变机制。
- **图表核对问题**：Figure 11 caption 称 y 为 out-of-1 importance，但图纵轴达几十、Table 2 斜率 4–13；需要澄清步骤得分缩放，不能直接把斜率看成概率百分点。Figure 14 的 distill 主导中间步比例中文 .61、英文 .56，与基座末步占优相反，显示模型条件的重要性。
- **局限**：解析成功样本选择、语言分词/生成长度/格式服从差异、小模型小样本、同一参考答案条件与扰动后合理答案变化，均限制因果解释。否定问题的真值是否同步重算及全量变化尚需更严格控制。
- **已做 / 可复用 / 未解 / 定位**：现有归因算法配合热图、模型/语言/扰动对照进行经验发现已做，可复用分析组织。尚未研究隐蔽后门/OA，不能仅移植同图式声称新方法；但围绕新、可验证的触发器传播规律展开仍符合立项范围。直接前驱 Wu et al. 2023 的 CoT 梯度归因列入追踪。
# ALTI-Logit — Explaining How Transformers Use Context to Build Predictions (ACL 2023)

- **全文证据**：`pdfs/alti_logit.pdf`，ACL 正式版 28 页，正文、参考文献、A–H 推导/实验/热图及 checklist 全部逐页完整读。PDF 第 1、9、20 页 Table 1、Figure 10/Table 5、GPT2XL逐层热图已渲染核对。
- **问题及方法**：解释先前 token 怎样在 Transformer 各层增加/降低目标 next-token logit。先把 residual stream 中 MLP、自注意力及初始嵌入对目标 logit 的贡献相加；固定当前计算中的 attention/LN 统计量，把 attention 输出拆为输入位置变换向量；Logit 直接将层位置视为原 token，ALTI-Logit 用 ALTI rollout 追溯层内上下文混合；目标与 foil 的 logit 贡献相减得到带符号对比归因。
- **数学边界**：Appendix A 明确把 LN 标准差视为常数，分解当前前向值不等于干预后 attention/LN 自行变化的效应。§2.4/Appendix B 的线性混合与 rollout 是归因假设，不是已识别的因果路径。MLP contribution 单独分析，并非一切非线性都被精确分配给原 token。§8 明确说方法不依赖 Transformer 内部因果干预；§10 仅建议作为调试工具并结合别的方法。
- **实验**：GPT2 Small/Large/XL、OPT125M、BLOOM560M/1.1B；BLiMP 去掉 ipsv/rpsv 歧义项后的9子集，加 SVA 4组、IOI；用规则 evidence 的 MRR 评估，是 linguistic plausibility 而非模型因果真值。Appendix Table7/8 中 argument structure 与 NPI 常是 GradNorm 最好，determiner-noun 常是 Logit 最好，远程 SVA/IOI 中 ALTI-Logit 优势明显；不要把摘要中的 consistently 写成所有子任务全面胜出。IOI 两种 proposed 方法 MRR=1 不证明所有实际机制已解释。
- **图表与研究发现**：Table1 展示 report 与 Impressionists 的正负 logit 更新逐层转变，Appendix H 扩展多模型48层热图。§6 根据 MLP 单元贡献观察数一致性特征。§7 将目标 logit 分解用于 NMT 对齐，在508对德英人工对齐数据上最佳层 AER 双语26.0/M2M27.3，优于attention48.6/96.4；最佳层选择与人工词对齐仍是可读性/一致性证据。
- **已做 / 可复用 / 未解 / 定位**：输入词×模型层×输出预测的带符号热图及视觉规律发现已做；可复用 logit-contrast、跨层视角和分解/干预对照。没有自由生成步间的总效应识别，也未覆盖训练后门/OA。项目若研究触发器从位置到层再到输出的模式，应把这种已做的可视形式作为工具底座，而以新现象与验证作为贡献。

# ContextCite: Attributing Model Generation to Context (NeurIPS 2024)

- **全文证据**：`pdfs/contextcite.pdf`，arXiv 2409.00729v2（2024-09-13），42 页正文、参考文献、A/B/C 全部附录完整读。PDF 第24页Table8、第36页词级评测、第42页条件/中介局限已渲染核对。
- **问题及输出对象**：给已生成全文或任一连续 statement/span，定位模型实际使用的 context 来源（contributive），明确区别于支持该句真值的引用（corroborative）。source 可句子、段落、文档或词。§2.3 对某输出span时固定此前原生成文本，评分为该span在所删context及原前缀下的概率。
- **方法**：随机独立保留/删除source，均匀采样子集，计算原目标文本的序列概率，取 log-odds 后用 LASSO（alpha=.01）拟合稀疏线性替代模型；系数直接作为归因，32–256消融。不是训练数据影响归因，不是真正重训；不是 SHAP 权重（Appendix C.1 显式区分uniform kernel与KernelSHAP）。序列概率由各步conditional token概率组成，但还需明确目标、消融、拟合才能得到归因。
- **实验**：最多1000随机验证例/数据集，Llama3 8B/Phi3 mini/Mistral7B，补充70B；TyDi只英文、各模型共用能放进8192窗口的过滤；QA/多跳QA/摘要，句子拆分后平均目标句结果。用top-k原目标logprob drop及held-out随机消融Spearman LDS评估；有attention/rollout/gradient norm/G×I/semantic/LOO/KernelSHAP基线。70B只32消融及少量基线。Appendix B.5 DROP词级归因LDS明显低于句级，32calls约.33、256约.55，说明更细粒度的交互/稀疏性困难；不能写32次普适足够。
- **已做应用及具体边界**：§5.1 源句帮助验证回答，§5.2 取高分context重生成改善部分QA F1；这是实际任务验证，且作者承认不能救“原先已选错来源”的失败模式。§5.3/A.8 **已经用归因定位推理时 context poisoning / indirect prompt injection**，因此“归因热图找投毒文本”不是空白。手工Phi20例；GCG共1000尝试仅22成功并筛选；Llama NeuralExec100例中91成功，90/91定位top1。尚无clean FPR、未知混合流量检测阈值、解释自适应攻击、训练后门/OA测试。
- **重要数字不一致**：正文§5.3说两类攻击均>95%，但Appendix Table8实列top1=90%/85%/98.8%，top3=100%/85%/100%。表中优化攻击引用PST24与ZWK23和前文对应模型互换；Llama文字90/91=98.9%，表98.8%。不能照抄正文的广泛检测率。GCG失败例之一模型自然因敏感句拒答，提示“目标行为发生”不足以认定trigger causation。
- **最直接未解已被作者承认**：Appendix C.4 给“生于1990→34岁”的历史中介例；对后句归因固定前句，可能找不到context，当前实现漏掉间接路径。重复冗余sources破坏稀疏线性，删句改变指代意义；建议held-out消融评估替代模型可信度。32次推理的成本、可跨多个span复用calls，是预算需报告的具体条件。
- **已做 / 可复用 / 未解 / 定位**：输出span→输入句/词归因、文本高亮、上下文投毒定位均已做；复用LASSO/LOO、held-out LDS及删context后任务变化。跨历史间接依赖是已提出而非无人想到的问题，CAGE/HETA/Contract需共同比较。训练后门/OA下哪些归因图模式真实、哪些是条件遮蔽或解释器失败，仍可作为视觉驱动的新实证问题；不能在未做实验时称方法已能识别OA。

# SocRAT — A causal framework for explaining the predictions of black-box sequence-to-sequence models (EMNLP 2017)

- **来源/全文证据**：David Alvarez-Melis、Tommi S. Jaakkola，官方 ACL `https://aclanthology.org/D17-1042/`，`pdfs/causal_seq.pdf` 10 页正文、参考文献全读；另下载作者 arXiv 1707.01943 12 页版 `pdfs/causal_seq_arxiv.pdf`，完整阅读额外第11–12页 Appendix A–D。作者版前10页不作为独立版本结论来源。官方主文第8、9页和作者附录第12页核心图与扰动表已渲染核对。
- **问题/方法**：同时解释结构化输入与可变长度生成输出之间的关联，而非只给一个输入词排名。用 VAE 在输入邻域生成语义扰动，查询黑盒模型的完整输出；二元特征表示原输入/输出词是否仍出现，以 Bayesian logistic regression 拟合 input→output 依赖与不确定性；形成二部图，经 robust k-partition/MIP 聚为可读子图。无需目标模型 logits 或梯度，但需辅助 VAE 训练与多次生成查询。
- **测量边界**：输出特征是原词的出现与否，不区分重复词、顺序和具体生成位置；没有固定原生成前缀。论文称 causal，但这里是 VAE 联合扰动分布上的局部拟合关系，未证明 SCM 因果识别。输入词协同变化、未进入原词特征的新词等会限制解释；Bayesian 不确定性不自动消除这些问题。
- **实验**：CMU 发音映射约130k词表，100个人工对齐词对、5次重复，使用编辑距离≤2扰动；与全词表 alignment oracle 相差约10 AER点。MT 比较 Azure、OpenNMT、人工译者，局部样本图及 attention 对照；14M OpenSubtitles 对话模型展示流畅回答仍可能仅依赖 What/you。多为诊断案例，人工对齐接近性不是黑盒因果路径真值。
- **与触发可视化最直接重叠（§5.5，PDF8–9页，Fig6/7）**：人为构造约6000条英文以 However 开头、法文对应非正式 tu 语域的训练样本，再加入1M条无 however 的普通平行语料；训练 seq2seq 后，归因图显示 However→tu/peux 的非语义依赖和 you→tu 的缺失。删除 However，翻译转为正式语域。即“训练数据注入伪关联→生成词依赖图/热图→删输入验证”早在2017年已有。不能据此说已做现代隐蔽后门/OA：文中没有恶意部署设定、ASR/clean accuracy、隐蔽性或自适应解释规避协议。
- **图表/附录**：Fig6 二部图与 Fig7 attention 显示上述具体机制假设；Fig8 Azure 性别刻板印象为独立自然案例。附录给 MIP 约束、gap=1e−4/2分钟上限，VAE 3层 GRU hidden500/latent400、10M WMT14 英文训练50epochs、KL/variance annealing；增大扰动方差时 Table3 含不流畅句，不能假设所有扰动语义有效。神经MT为WMT15 en-de两层500预训练模型。人工译者对扰动集中一次翻译，并未模拟LLM。
- **疑点**：§4.3 子图重要性用负 cut weight，§5.3 “highest cut”重要性表述不一致；需按公式及代码另核，不直接复用优先序。官方PDF有可恢复xref警告但全部页面可解析与渲染。
- **done / reuse / candidate_gap / 定位**：黑盒输入—输出分组解释、伪关联诊断、正常/人为污染训练及输入删除对照均已做，必须进入最早直接先例。可复用图形探索与输入验证组合；本项目新贡献不能是“首次用词热图解释trigger”，仍可在现代训练后门/OA、固定前缀与自由生成差异、解释失效等条件下发现并验证新规律。

# Contrastive Attribution in the Wild: An Interpretability Analysis of LLM Failures on Realistic Benchmarks (2026)

- **来源/全文证据**：Rongyuan Tan、Jue Zhang、Zhuozhao Li、Qingwei Lin、Saravan Rajmohan、Dongmei Zhang；arXiv 2604.17761v1（2026-04-20），`pdfs/wild.pdf` 45页，正文、参考文献、Appendix A–I 全部逐页读。PDF第9、11、33、44页热图、层图、验证表与分解表已渲染核对。
- **问题/方法**：定位现实任务失败中的第一个可识别错误 token，对错误目标与正确 foil 的 logit 差做 AttnLRP；给带正负方向的输入热图及跨层 hidden-state attribution graph。图构建批处理多个目标梯度，低节点阈值 .01，保留每层85%绝对边质量。不是系统性神经元电路提取；剪枝后的图也不是未压缩计算图。
- **重要贡献定位（Appendix G，PDF33）**：作者明确目标不是发明归因方法，而是判断现有 AttnLRP 是否适于失败诊断，并允许替换其他忠实归因法。这是“现有归因+可视诊断框架+经验发现”可成为论文核心的直接前例，不能据已有算法否定本项目。
- **实验**：IFEval265个失败、GAIA2300、MATH91、EvalPlus270，过滤后约20.8/17.0/40.7/19.6%；主要Qwen3-0.6B，GAIA4B，规模对照1.7/4/8B，Olmo3-7B-Think训练阶段对照。GAIA平均输入12374 token，其他短任务约54–169；错误 token 由4个较强模型提议、至少2个一致加人工判读，foil需替换后可恢复合理正确轨迹且logit gap>1。大量失败被筛去，不能推断整个失败总体的模式占比。
- **发现与图**：URT（相关词权重过低）、OIT（无关词权重过高）及输入图难解释而层图更有信息的案例；Fig2/3 展示跨模型归因变化。规模和训练阶段观察到更多正确 token 偏好/指令依赖，属于选定数据及归因度量上的经验关系，没有内部干预证明唯一机制。
- **验证实质**：Appendix G 对60例遮蔽最高排名输入词；AttnLRP平均1.7词、CP-LRP2词、gradient3.7词，报告100/100/96.7% fix。这里 fix 定义是 top1 不再为原错误 token，**不是原任务通过、必然改成foil或所有原约束仍保留**。故只能引用为局部预测敏感性验证。
- **人工/选择限制**：校准后分类一致88.2%仅EvalPlus34例，模式一致83.3%仅24例；校准前各集约39–71%，不是所有样本可靠率。Appendix I只有57例、23张图作3类探索聚类，不能泛化已识别三种普遍机制。Table1筛选比例、Appendix E4删项比例和各附录样本数有不能直接对齐之处，宜保留原数及疑点。
- **分解疑点（AppI，PDF44 Table16）**：所谓 Self Bias 是节点归因减去保留入边和，可能包含剪枝、非守恒和同位置混合残差，不能直接当“内部偏见”。Eq26若 Total=SB+BOS+OC，Table16首列.693与.118−.145+.246=.219不符，其他列也不符；图16亦不是该三条曲线简单加和。所谓比例分母可为负，不能解释为概率贡献百分比；ARI≈0不证明统计独立。“最后6–8层决定”仍是观察假设，没有因果干预。图形启发有价值，但不能照单全收机制宣称。
- **成本疑点**：Table2峰值显存3.5MB显然不能代表含0.6B模型的总显存，全文未充分区分增量/总量；1508-token效率测试未覆盖GAIA平均12k长度。批处理减少反向调用数不等于端到端FLOPs同倍下降。
- **done / reuse / candidate_gap / 定位**：错误输出位置→对比输入图→跨层图→跨模型/训练阶段经验模式已做；可复用目标-foil、人工选点规则、图分层探索、局部遮蔽验证，但应改善样本选择与验证结局。本项目的触发/假触发/普通后门/OA对照仍是具体新对象；是否有新认知须通过留出对照与干预成立，不能仅换数据集。

# PromptExp: Multi-granularity Prompt Explanation of Large Language Models (2024)

- **来源/全文证据**：Ximing Dong、Shaowei Wang、Dayi Lin、Gopi Krishnan Rajbahadur、Boquan Zhou、Shichao Liu、Ahmed Hassan；arXiv 2410.13073v3（2024-10-30），`pdfs/promptexp.pdf` 13页，正文、参考文献与所有末尾材料全部读。第3、4、7页公式、UI、TableI/Fig3已渲染核对。
- **问题/方法**：把 prompt 对整段生成的影响显示成 token、词、句、用户定义组件的热图；一个路线对各输出步的 IG 归因聚合，默认等距采样5步或选置信最高5步并按概率加权；另一路线逐词mask并重生成，比较log概率、SentenceBERT语义或词交集。浏览器UI支持编辑prompt、选模型参数与即时查看多粒度解释。
- **关键定义问题**：IG得到(prompt+已生成词)×输出步矩阵后，丢掉生成历史的贡献、重新归一化到prompt，不能解释为包含所有中介的完整贡献。高概率输出更代表“理解prompt”的权重假说未经验证。PerbLog 对原/扰动的同序号输出位置取原词log概率，但两边前缀可已不同，不是固定前缀效应；缺失top-K或输出变短时令logprob=0，相当于概率1，会引入明显伪差异。API logprob与full-vocabulary logits需区别。
- **其他目标边界**：PerbSim = 1−SentenceBERT cosine 是所选表示的全输出语义变化；PerbDis =1−intersection/perturbed length 不对称，丢弃词序。句/组件通过词分数相加，不识别组合交互。不同目标回答不同问题，不应共称唯一真实importance。
- **量化实验**：Llama2-7B-chat/GPT3.5-turbo、温度0，SST情感标签受高/低归因词扰动时的翻转率；替换强制句向量相似度<.7，可能改写原任务语义，不能作为保持任务条件下的唯一忠实性检验。另一实验后缀“Give a short answer”归因与长度变化Spearman较低。TableI：Llama PerbSim flip .68/.29、AggConf .68/.32；GPT PerbSim .70/.08、PerbLog .63/.09；GPT后缀PerbLog相关−.06，对照.01，未支持假说。5次平均无方差，全文未清楚报告所有样本量、扰动比例和IG基线/积分步配置。
- **用户研究/图表**：10位工业开发者20分钟说明，5个BigBench提示50份质量评分均4.2，4个情感错误40份帮助评分3.82，2个压缩任务20份评分4.2；是小样本主观帮助证据，无受控客观任务提升。文中65%(27/40)算术实际67.5%。用户指出过多高亮难解，想指定目标输出或局部输出。压缩示例删除“不要增强赋值”要求，未验证所有原约束保留。
- **成本/局限**：假设无限并行时写O(X)/O(I)并不代表实际总成本；50输入50输出约7秒对比4.74基线，选5步IG约3.84对比2.05，取决硬件并发与模型。掩码符号、同序号对齐、重生成随机性、输出采样偏向高概率位置会影响结论。
- **done / reuse / candidate_gap / 定位**：多粒度文本热图、输出归因聚合、白盒/黑盒与用户可视探索已做；可复用UI组织和多目标比较，但不能默认PerbLog定义正确，也不能把prompt归一化称历史总效应。本项目应保留输出步与输入来源联动，明确局部/自由生成目标，以可验证新现象作为研究产出。参考文献直接前作2403.03028、后续敏感性/PRIG纳入追踪。

# How are Prompts Different in Terms of Sensitivity? (NAACL 2024)

- **来源/全文证据**：Sheng Lu、Hendrik Schuff、Iryna Gurevych，ACL正式版 `2024.naacl-long.325`，`pdfs/sensitivity.pdf` 24页正文、参考文献、Appendix A.1–A.7 全读；曾一次输出截断，已单独重新读取缺失第1–4页。第7、20、21、23、24页热图、分组数据、完整解码曲线已核对。
- **问题/方法**：比较不同 ICL 提示的预测敏感性与准确率，并用已有 gradient L1 norm 热图观察其与输入/指令/知识/选项分组的关系。敏感性=原样本加局部替换样本预测的 variation ratio（1−众数频次/总数）；不是指定trigger效应或单词贡献。另以重复扰动输出logit方差惩罚标准解码。
- **实验**：CoLA527、CSQA1220、MNLI1000、RTE277、SST2872，few-shot；GPT text-davinci-003、GPT-JT6B、Llama2 7/13B chat、FlanT5 770M/11B；额外Llama1、GPTJ和T5对照。温度.8、3seeds，通常最大输出2token，CoT64/GSM8K128。比较普通指令、CFP、CoT、生成知识/指令，以及**直接给出正确答案**的zero_a/b极端条件。
- **发现**：跨模型/任务/提示的总体accuracy–sensitivity Pearson −.8764，greedy时−.5507；saliency prompt−input差与敏感性−.7596。但移除答案泄露zero条件后后者降为−.5733,p=.0831（正文脚注7）。热图的末尾 ANSWER/冒号很强，不能只据梯度说模型“回忆训练答案而不用输入”；这种记忆机制在文中是推测，未通过干预识别。
- **核心边界**：提示含真答案会让稳定性与正确性同时上升，原任务意义已经变化；归因均值是位置/词数分组度量，不能当贡献百分比。固定logit梯度反映局部敏感性，不能等同有限扰动。邻域合成数据可能语义改变/不流畅；AppA1 223条中44条噪声，声称同数据作控制不会造成问题，仍未排除提示对同噪声反应不同的混杂。黑盒text-davinci“175B”为作者用名，非本审计独立核实模型架构。
- **图表核查**：Table3 GKP虽均值提高，但GPT/Flan11B准确率下降、Flan770M敏感性上升；Table4 CoT较base_b平均略增accuracy同时增敏感性。GSM8K只把数值答案当类别，并非任意自由文本；zero泄露极强地主导相关性。Table17 FlanT5对真答案token的绝对梯度均值18.54其实高于GPT13.37，所谓“更少focus”依赖input相对比率，不可写绝对归因更低。
- **解码验证局限**：Table7报告α=.1–.9中的最高accuracy，未声明独立调参集；Fig11–13显示多数较强提示/FlanT5条件有下降，不能称全面改进。五次forward、约5倍成本；低敏感不等于正确的逻辑常识依然适用。
- **done / reuse / candidate_gap / 定位**：已有算法的token热图配合提示类型/模型对照用于新经验分析已做；可复用预先分组、梯度与扰动关系的双视图。尚未回答触发后门/OA，但可为“正常指令/格式/答案泄露也高亮”的负控来源。研究不能简单以热图集中在prompt为后门证据。

# CLP — Black-box language model explanation by context length probing (ACL 2023)

- **来源/全文证据**：Ondřej Cífka、Antoine Liutkus；ACL `2023.acl-short.92`，`pdfs/clp.pdf` 13页正文、参考文献、Appendix A–C及checklist全读；第1/3/5/11页UI、公式及NLL/KL/距离图核对。
- **问题/方法**：对同一目标位置固定原文结尾上下文，从最近1词逐步加入更远词，计算目标词NLL或相对最大上下文预测分布的KL。相邻长度的KL下降即带符号 differential importance；逐位置滑动评估全N×context×vocabulary张量，UI点目标词即可显示输入红绿高亮，并查看不同长度的top候选。
- **定义与限制**：这是“已保留其后所有词时，再加该词的增量”，不是该词孤立重要性、任意子集Shapley或从它到当前词的总中介效应。作者明确不必和为1、可负，NLL与分布KL回答不同问题。原文序列固定，不重新采样后续生成。适用性依赖模型训练允许任意文中截断；modern chat system/role结构被截断未必处于训练分布。
- **实验/发现**：GPT2 117M/XL1.5B、GPTJ6.1B；仅8篇英文LinES文档20672token，context最大1023。跨距离token归因的均值类似幂律递减，专名比其他POS更受长上下文帮助；较大模型收益多来自8–256上下文。作者把实体记忆/复制作为解释猜想，未做因果干预或独立长文生成任务检验。
- **图表**：Fig1“birds”的owl/swoop高亮、可选目标与候选列表；Fig4/8/9展示同一位置在不同模型、目标NLL与全分布KL下排名可不同。Fig5是归一化绝对分数的均值/标准差，不是符号和为1的贡献概率，且不是拟合并验证了普适幂律。Fig6按输入POS汇总，项目可借鉴token→群体规律的视觉组织。
- **实现/成本疑点**：Appendix B 报告原始FP16 logits 2TB、500个8CPU任务、总318 core-days（含失败/调试）；算法调用次数N并非廉价。stride k降成本会把来源词合成k词块，失去分辨率。正文Eq3之后 P[n,c]=P[n,n−1] 与允许n词上下文不一致，Eq6的 n−m−1/n−m 按Eq2定义也与加入x_m的长度差一，原图核实为印刷内容；实现前需核对代码索引，不照抄。
- **done / reuse / candidate_gap / 定位**：指定输出位置→概率/分布变化→来源token带符号高亮和动态上下文曲线已直接做过。可复用为明确顺序的局部上下文对照，不当作独立路径归因。没有trigger/OA实验，也未解决自由生成历史传播；本项目的新价值需来自该条件下的新规律和验证。

# DBPA — Quantifying perturbation impacts for large language models (2024)

- **来源/全文证据**：Paulius Rauba、Qiyao Wei、Mihaela van der Schaar；arXiv 2412.00868v1（2024-12-01），NeurIPS2024 Statistical Foundations workshop标识，`pdfs/dbpa.pdf` 13页，正文、参考文献、Appendix A/B全读，第7/8页Fig3及Table2/3原图核对。
- **问题/方法**：避免把单次输出随机差异当输入扰动效应；原/扰动输入各采k个完整回答，计算原回答两两相似度分布P0及原—扰动交叉相似度分布P1，用JS散度量化差异并排列得p值。允许任意输入/模型扰动，默认句向量cosine，也提L1/L2。是自由重生成与整体语义目标，不是固定前缀token贡献或输出定位算法。
- **实验**：GPT3.5医疗推荐前加Act as不同身份；多模型同义问题改写鲁棒性；以GPT4回答作参考比较GPT3.5/Phi3/GPT2/Llama3.1/SmolLM/Gemma2/Mistral/文生图提示模型。5seeds均值标准差，正文与附录未给k、排列次数、完整问题量、embedding模型、JS直方图/密度估计和全部解码配置。没有逐token归因热图或后门/OA实验。医疗内容只是论文案例，不作为医疗建议来源。
- **关键统计边界**：标量相似度是低维投影，P0=P1不足以推出原回答全文分布相同；原假设写S(x)=S(Δx)而实际检验相似度统计，不宜照抄“整个输出分布”的保证。两两相似度共享原回答，不能把O(k²)个pair当独立样本。正文称交换P0/P1，Appendix A又描述混池原回答后重算；这两种排列不同，原文未清楚规定实际单位。仅假设similarity-space exchangeability不能自动满足。附录从期望相同推到统计量分布相同，证明不足。
- **显著性与效应**：p>.05不能证明医疗身份提示“输出一致”，也不能证明与参考模型alignment；需要功效/等效性边界。p=0显示也需有限排列修正。摘要称multiple testing controlled error rates，但全文无具体校正算法或误报校准。各表effect和p不能直接读作好坏排名，缺少统一功效与任务质量评价。
- **比较疑点**：Table4说ROUGE/BERTScore/MoverScore不能用于black-box outputs，是不成立的宽泛对比；其文本指标本就只需输出与参考文本。Table2 GPT4显著变化.15而Phi3 .10，GPT3.5 .05，不能据此称“更大模型普遍更稳”；方差较大且训练类型/任务不匹配。
- **done / reuse / candidate_gap / 定位**：自由生成时用无扰动重复采样建立噪声参照、显示语义分布与效应/不确定性已有。可复用该实验思想，但应按原始回答而非相似度pair构造有效推断，先校准误报与效应敏感性，不继承未经证明的统计保证。DBSA 2512.11573是后续token可视化应用，由visual域独立完整审查。本项目若扩展自由生成热图，已有分布视角不可称首次。

# PRIG — Localizing Prompt Ambiguity in Large Language Models with Probe-Targeted Attribution (2026)

- **来源/全文证据**：Govind Ramesh、Yao Dou、Wei Xu；arXiv 2606.05486v1（2026-06-03），`pdfs/prig.pdf` 23页，正文、参考文献、Appendix A–F（包括全部12对人工gold prompts）完整读；第7/16/17页热图和层选择表核对。
- **问题/目标**：解释的是辅助线性probe解码出的“prompt歧义”分数，**不是某个生成token logit或实际行为**。这是重要相关但目标不同的边界论文；不能把probe高亮直接当“受trigger影响的输出”。
- **方法**：Llama3.1-8B-Instruct冻结；逐层meanpool残差训练无intercept logistic probe；将较早m层残差到较晚l层probe logit的截短子图做50步zero-baseline IG，再按隐藏维求和并σ=3 Gaussian平滑，映回token位置。作者明说放弃相对原输入的全局completeness。中层位置已有上下文混合，保留位置索引不等于该原词独有来源，显示得更局部不证明因果更正确。
- **数据/实验**：编码501、数学357、写作385基础题，GPT5.4每题改一个任务关键句，全部改写句token作正标签；75/25分割，Table1报告combined AUROC .840/gold .891，对比embedding G×I/IG。12对人工gold每域4对；跨域平均AUROC .805低于域内.870。Sentence-level最大均值+阈值，held-out combined305正/305负上5fold选择τ，报告F1 .734vsGPT5.4 .587。
- **关键数据问题**：正文说gold只评测，Appendix A第15页却明确把同域gold例串接到**合成训练数据的生成提示**，因此不是完全未参与数据构造的gold外推；不等于探针直接训练gold，但需承认间接暴露。gold个别“歧义”有其他句给出同一明确要求（coding3/4），writing3只是另类格式表述；标签是作者改写区并非独立模型歧义因果真值。Coding2 clear版本还含not intended for public use与必须public的矛盾，baseline标签需复核。
- **评估疑点**：§5.3写probe在held-out eval训练，Appendix C说在train训练且pair不能跨分割；文中给的普通train_test_split调用本身未展示pair分组，实际代码需核对。正文声称training选层但Appendix D明确是held-out performance，Table1 math .813/.720分别来自不同层区间的AUROC/AUPRG最佳值，单一interval无该二元结果；并非可忽略的小数差。G×I writing .544/.409也不是表内所有层最高组合。需澄清选择协议，不能无条件复述全面胜出。
- **忠实性边界**：只恢复改写句、没有修复prompt后任务准确率/澄清效果；Gaussian平滑匹配整句正标签，未消融平滑vs层截断的贡献。clear prompts针对‘本会被改写句’AUROC近随机不证明绝对分数低、校准良好或FPR低；AUROC是排名而非阈值误报。
- **done / reuse / candidate_gap / 定位**：probe目标→局部残差IG→token热图与歧义定位已做；可借鉴同时观察行为输出与monitor/probe分数的**双目标对照**，但必须分别命名。对OA尤其不能用可隐藏probe当唯一行为因果依据。未测试其他latent属性、其他模型规模或后门；不扩展成所有probe文献的综述。

# Focus-LIME: Surgical Interpretation of Long-Context Large Language Models via Proxy-Based Neighborhood Selection (2026)

- **来源/全文证据**：Junhao Liu、Haonan Yu、Zhenyu Yan、Xin Zhang；arXiv 2602.04607v1（2026-02-04），`pdfs/focus_lime.pdf` 8页，正文、参考文献、Appendix A/B声明全部读，无另附实验附录。第2/6/7页热图、删除曲线与证据定位表核对。
- **问题/方法**：长prompt高维词级LIME在有限采样下不稳定；先由小proxy做段→句→词的LIME及top-k筛选，后仅在选中区域用目标模型进行Bernoulli(.5)扰动拟合，未选区域始终保留。最终解释是对目标模型在**条件子空间**的贡献，不能以其他区域未受测证明它们不重要。proxy若漏掉trigger，目标精算阶段无机制找回。
- **实验**：CUAD/Qasper的yes/no QA，用GPT4o与DeepSeekV3，proxy Qwen3-30B-A3B及Qwen2.5（版本见疑点）；originalLIME1000扰动，纯proxy10000；AOPC删10/50/100词和均值。另IMDB平均228词，以Qwen3-235B目标观察逐渐固定不重要词并选optimal neighborhood。CUAD人工evidence按相对长度取词、算recall，属于人工合理性，不能等同模型事实使用。
- **图表/主要结果**：Table1/2原LIME和纯proxy所有项均0，Focus约.28–.53平均AOPC；这仅是作者此配置报告，未提供误差/数据量/零值诊断，不能称所有LIME完全失效。Fig3/4显示优化邻域提升局部删除敏感性，Fig5展示governing/laws/Illinois热图。没有任务修复、后门或路径传播实验。
- **可复核疑点**：正文proxy写Qwen2.5-14B，Table1/2为32B，Table3又14B；“without proxy”基线文字写跳过neighborhood curation、在full document constrained扰动，但正文解释却当目标模型做curation，流程不一致。Table3表头k=50/150/200%，正文100/150/200%。RQ2称两数据集只命名IMDB；optimal邻域沿评测路径选择，未报告独立选择集。未给Focus最终K/top-k/stop参数、样本量、完整prompt/解码和实际token/API预算，故节省queries/保持fidelity不能视为完整复现结论。
- **done / reuse / candidate_gap / 定位**：粗到细来源定位+词高亮、proxy筛选后目标模型验证已提；可借鉴长上下文交互缩放，但必须显示筛选外区域“未测”而非零因果贡献。本项目尤其要防代理与OA行为/表示不一致导致漏解释，不能用小模型proxy默认替代目标。UnCLE等通用概念解释仅作外围追踪，不自动扩大为所有概念归因综述。

# Word Importance Explains How Prompts Affect Language Model Outputs (2024)

- **来源/全文证据**：Hackmann、Mahmoudian、Steadman、Schmidt（DataRobot）；arXiv 2403.03028v1，PDF25页正文、参考文献和全部第7节附录完整读；第2/4页多指标热图及第19/21/22–25页语义核查、所有相关散点图已渲染核对。
- **问题/方法**：对system prompt逐词用下划线替换，重新自由生成N次，计算原输出与扰动输出的**评分绝对差**并对用户问题和重复取平均。评分分别为词数、Flesch reading ease、相对topic的MiniLM句向量cosine；并非特定输出位置的概率归因。Fig1文字高亮、Fig2词×输出属性热图已做，多指标可视化本身不能作为首次贡献。
- **实验**：GPT3.5turbo16k0613、Llama2-13Bchat，temperature=1、N=3，表内M=1。GPT4生成112主题、角色及每题问题；另SQuAD2问答。后缀分别控制长回答/术语/公司主题，检查后缀内最大词分数与删除整个后缀的分数相关。生成总样本量/最终模型子集未完整交代，作者承认Llama数据少。
- **图表结论**：Fig10–13相关系数皆正但并不都强，Llama人工题的长答→词数r=.01，SQuAD主题后缀→词数r=.01；许多相关标准误约.17–.22，不能写统一可靠。SQuAD GPT相关约.17–.49。Fig12长故事整体效应经常大于单词最大值，作者解释组合意义；这是组干预与单词干预的差别，不能用其否定整条归因路线。附录语义核查仅5个topic×7段合成文本，MiniLM比所选mBERT显示更清晰块结构，不支持普遍禁止mBERT。
- **统计/因果边界**：自由生成随机性下，绝对分数差在没有prompt作用时仍有正期望；N=3不能自动区分正常采样差。未报告同prompt重复的空干预对照、效应置信区间或误报校准；配对随机数亦未明确。后缀最大词分数与整体分数共享原输出，关联可能含共同基准噪声。绝对值还丢失促进/抑制方向，按列归一化不允许跨指标比较绝对效应。词mask改变语法和分布，不能把所有变化叫某词固有贡献。
- **done / reuse / candidate_gap / 定位**：已有system prompt词→自由输出属性变化的多指标热图、词/组影响比较；可复用可视化分析框架并加入空干预、词组组合与正负方向、输出位置定位。本文没有恶意训练投毒/触发器验证，用户prompt及system×user相互作用在局限中明确留待后续，不能把背景中提到universal triggers当已做实验。

# LLM Explainability via Attributive Masking Learning (2024)

- **来源/全文证据**：Oren Barkan、Yonatan Toib、Yehonatan Elisha、Jonathan Weill、Noam Koenigstein；ACL Findings EMNLP2024 9522–9537，`pdfs/aml.pdf`16页正文+全部参考文献+Appendix A完整读；第16页六组定性高亮与指标公式渲染核对。候选阶段“Learning to Attribute with Attention”为误题，已按下载后原文纠正。
- **问题/方法**：辅助归因模型G（RoBERTa+token MLP）接输入与目标类别概率向量，输出[0,1]soft mask。目标模型固定，双mask分别保留预测/破坏原最大类预测，另稀疏正则；先约1000样本pretrain（pAML），可每个样本finetune（fAML）。超参在独立validation按指定metric选择，但fAML在该实例的同一metric上持续选择最佳图；因此报告每指标专门优化后的表现，并非同一套图在所有metric一并胜出。
- **范围**：论文§3明确focus classification。Decoder Llama2/Mistral7B仅首个生成class token，SST2/RTN/IMDB为few-shot、EMR为LoRA分类；仅解释输入实例v，few-shot/指令u固定且不输出归因，作者明留未来。无跨生成步/CoT或trigger实验，不能据题名外推完整自由生成归因。
- **实验/结果**：5模型、4任务、13方法、Suff/Comp/LogOdds及AOPC；Table2 fAML多数项最佳而不是所有项（Mistral EMR Comp/A-C SIG略优）。encoder human-rationale ERASER Movie Reviews top25词macro F1 .118，micro .096，虽高于对照但绝对低、没有不确定性，不能称人工解释充分可靠。Appendix A.2双loss删项常仅小差异，表中某个删项Comp仍.71与full同值，不应描述每项显著必需。Fig/table6六例top3高亮只是定性合理性。
- **效率/验证边界**：RoBERTa50短例+15长例两次runtime，pAML .01s、fAML1.5–1.6s，未摊入pretraining、TPE搜索及指标计算完整预算；不能把该时间外推目标Llama。mask/unk插值会偏离自然文本。只训练/选择干预metric可能求出高metric图而未证明唯一机制解释；需要另一干预/任务修复验证。输出非负mask不表示促进/抑制，也不能以没有显示u证明u没有影响。
- **done / reuse / candidate_gap / 定位**：学习式词高亮/双充分与全面性约束已做；可复用对照“评测可优化性”和展示未测区域，不能当无条件默认基线。可视化研究不必重新发明该算法；本项目应检验已有解释在trigger/历史传播上的行为与盲点。

# TokenShapley: Token Level Context Attribution with Shapley Value (2025)

- **来源/全文证据**：Xiao、Zhu、Samyoun、Zhang、Wang、Du；官方ACL Findings2025 pp3882–3894（实际13页，候选阶段17页误记已纠正），正文、全部参考及Appendix A–D完整读；p13彩色原始案例核对。
- **问题/方法**：每个context token存prefix隐藏向量与next-token标签，目标response token用自身prefix检索，计算KNN“是否投对该token”二元效用的Shapley，再汇总输入/输出span。关键是**该效用属于固定embedding的KNN代理**，没有每次删除context后重算目标LLM生成分布；所谓精确Shapley不是原LLM输入贡献的精确解。四/五数据集实验均K=1、仅M=10近邻；多K动态规划理论不是实际所有实验。
- **范围/新意边界**：token→token高亮与上下文来源定位已做。式9明确正贡献只来自与目标相同token的context标签，异词只有负或零；因此触发词“abc”促使输出“harmful”的行为不能用该正归因可靠恢复。对改写/语义推理同样需要另验证。省略的context会改变各prefix embedding，但代理游戏固定它们，不等于真实prompt干预。
- **评估**：QuoteSum1319、Verifiability197人工来源label，KV500、NQ1000、CNN/DailyMail1000；3模型实际Llama3.1-8B、Mistral7Bv0.3、Yi6B。多数主评估是人工引用来源准确率；KV是朴素exact-match也能100%的检索任务，作者坦承。CNN真正删除句后固定response logprob下降，TokenShapley1.01/1.33低于ContextCite64的1.38/1.48，不能无条件声称归因最强；NQ F1 .417 vs .412增幅很小且无CI。标题摘要四bench但正文五，属于稿件陈述不一致。
- **图表/实现疑点**：AppD正确Tom Brady句分16.8，另一不支持答案的Manning句仍2.83，precision/recall均.5，作者承认表面匹配混淆。§4.3以“非K近邻贡献零”缩减M，这是对于每个coalition内近邻才成立，不能无证明外推full-set近邻外所有Shapley为零；未给截断误差。Algorithm1取q+R prefix与§4.1分句datastore、不含完整context的定义需代码澄清。效用式7空集0≥0形式为1而正文说无邻居为0，需明确约定。离线cache声称约100KB/例用768×token数却未给dtype/实际hidden层维度；无实测端到端时间，不能仅凭多项式称低成本。
- **done / reuse / candidate_gap / 定位**：可作为来源检索代理对照与可视化例，不能默认替代trigger对目标模型的概率效应；未来实验可直观看见“引用支持”与“实际行为驱动”的不同区域，贡献可以是该可靠区分与验证，不要求新Shapley算法。

# Unveiling and Manipulating Prompt Influence in Large Language Models / TDD (2024)

- **来源/全文证据**：Feng、Zhou、Zhu、Qian、Mao；arXiv2405.11891v1（2024-05-20），ICLR2024；24页正文+参考+Appendix A–N完整读，p16/19/21/22全部关键分布图、词高亮、输出控制案例与消融渲染核对。
- **方法/测量对象**：forward用每个原prompt prefix的最后层next-token概率差r_i=p(target|prefix_i)−p(foil|prefix_i)，按相邻prefix作差；backward从末尾suffix渐补前词、固定预测末端，取差；bidirectional直接相加。target可词集/无foil。forward一次模型调用，backward/bi约输入长度次。它是两条特定词加入路径的预测分数变化，不是全coalition或唯一因果拆分；forward的预测位置不断改变。AppB.1作者明确承认这种语法位置问题，不能宣传通过LM head消除了归因假设。
- **实验**：BLiMP11语法子任务，对比target/foil，GPT2large/GPTJ/BLOOM/Pythia/Llama2及13B/OPT30B；空格替换、插入/移除top词评估。正文称AOPC的是逐次插入后平均两类相对概率，称Suff的是删除top词后平均概率，并非ERASER常规Suff定义，跨论文不可直接比较。所有TDD均值多强于对照，分子任务不是全胜；未给CI。AppA把末层分布当groundtruth、KL随层趋近只是内部一致性，不能证明层间token固有因果贡献。
- **直接安全应用已做**：RealToxicityPrompts1225短prompt，GPT2生成20词，用毒词黑名单概率集合归因，空格替换top15%输入；报告toxicity .49→.20。OWT5000中性prompt，替换top1成positive/negative，正向.52→.78、负向.48→.87。是自然有害/情绪诱导词，**不是训练植入后门trigger**。已有可视化发现输入触发线索→干预后输出变化，不能将此工作流包装首次。
- **验证与限制**：AppK随机位置和倒置target/foil对照支持定位有增量作用；但随机替换已可把毒性.49降.31、负情绪.48升.81，TDD额外增益仅.11/.06，并不证明全部效应来自解释正确。300随机prompt×3人工评分验证属性/流畅性，无语义保真或原任务保持评估；改变请求可能让输出离题。Fig5由.64077→.01734是约97.3%相对下降/62.3百分点，正文“64%”含混。Fig6原simply被划掉替换positive（提取纯文本看似追加，已以原图纠正）。
- **成本/可复用/定位**：RTX A5000，>6B为4bit，forward Llama7B .14s、backward .91s；只是短prompt单实例，长history成本和更长自由输出未验证。可复用对比词集+正负图、随机与倒置目标对照、同次可视化和行为编辑验证；新增空间在本项目具体stealth/trigger行为和传播发现，不必强求新归因算法。AtMan相关基础交foundations核查，不扩张全部style-transfer研究。

# Tokengeist: Multi-Turn Attribution Tracing in Agentic Conversations (2026)

- **来源/全文证据**：Jessica Tang、Shraddha Barke、Sharad Agarwal；arXiv2607.22610v1，PDF署2026-06-14（编号月份与日期不一致，保留原文）；27页正文、参考、Appendix A–E完整读；p8/16/18/19/25/26图表已渲染核对。与visual域2605.15455 MultiTurn是不同论文。
- **问题/方法**：选择response span，base方法对之前sentence打分，turn内max池化；递归归因被选assistant的**整轮内容**，user/system为叶，tool result固定1.0连至最近调用assistant。BFS构DAG，默认top3、depth8，再以每节点max的.85阈值剪枝。base为平均attention、AT2、AttnTrace；没有新基础归因算法，核心是递归来源分析与可视化。
- **根本解释边界**：§4/limitations明确是在**已完成记录上事后重新阅读**，归因模型可不同于原生成模型；分数反映解释模型的关联性saliency，不是原生成器生成时attention或删除来源后反事实。文中自己承认attention非因果、干预base尚未集成。因此所谓provenance collapse是在这个proxy/标注框架中的失败，不可直接外推原生成器的历史中介机制；不能用递归高亮图代替模型实际因果路径。
- **数据/评估**：MTCABench665对话3845目标，ConFETTI109对话688目标中30对话holdout、实际评价79对话495目标，Tau556/3157；4归因模型。GPT5.4选择traceable事实span并递归标注、43会话失败重跑Pro，96%以上目标含tool链；固定结构边与gold规则一致，需单独结构基线才知道多少增益来自真实token信号。100目标一位作者spot-check，87%correct+11%minor被合称98%accepted，复核更正不是独立标注一致率。
- **结果/图表**：各模型递归Edge/Node/Source/Span指标提升；Tau Avg/AT2来源召回约82–91%，非所有backend90%（Phi AttnTrace34.1%），span F1仍约3–39%，不同指标不能替换。Fig4直接依赖常无收益甚至下降；多层链提升是主贡献。Fig9句/词粒度结果依方法而异，AttnTrace词级更强但较贵，默认句级非该方法最佳。19个wrong-value改成正确target后来源图评价，不是原模型修复实验。
- **标注重要疑点**：Listing2既定义依赖为causally necessary，又规定即便模型**忽略**了正确指令，也将该指令纳入source，混合“实际驱动”与“应该使用”。AppD虽分hallucinated/corrective F1，整体gold不能视为纯事实因果真值。对ConFETTI86处补年份让标签可归因，实质改变原记录；适合构造来源基准，不能说在原生成轨迹恢复了同样依赖。
- **稿件/成本边界**：源码/benchmark承诺camera-ready发布，当前论文并未验证运行。AppA.3说重投影避免eager，A.4又说所有模型eager因hidden states不兼容flash，实现陈述不一致；无代码审计不判断哪条真。Table4 Tau hallucination35(1.1%)与limitations51(1.6%)不一致。Fig8 caption称各depth/距离>75%，原图depth2约35%、depth4约60%、AttnTrace全程低于此；不能重复概括。depth8饱和不代表“no gold paths truncated”，gold深度到14。主要评价117GPU小时不含额外选择/标注成本。
- **done / reuse / candidate_gap / 定位**：跨轮递归链与DAG交互解释已有，且框架论文可使用既有方法，这支持项目立项形式。可复用本轮/前轮/工具结构边区分以及直接、链、多来源、错误来源案例，但需清楚标图为proxy provenance或实际干预证据。stealth/OA具体触发后历史传播/可视化盲点尚非本文已测，保留本项目具体研究问题。

# ReX: A Framework for Incorporating Temporal Information in Model-Agnostic Local Explanation Techniques (2025)

- **来源/全文证据**：Junhao Liu、Xin Zhang，AAAI2025官方9页（18888–18896）完整正文/参考，补读arXiv2209.03798v3（2025-03-01）第10–15页全部Appendix A–D。`pdfs/rex.pdf`与`rex_extended.pdf`；原图p2/p6及补充p15核对。此处temporal指输入特征绝对位置与词对距离，不是生成历史中介。
- **问题/方法**：在LIME/KernelSHAP/Anchors中把单词存在谓词扩展为位置约束（1D）与两个词的相对位置约束（2D）；BERT替换、随机删除与一次交换生成邻域。解释可显示“not在bad前面”而非分别把not、bad评为正面。重叠谓词之间相关，不能据KernelSHAP名称就认为得到了原始输入词唯一贡献；改变了特征空间、采样邻域和解释模型容量。
- **实验**：SST1821样本上的LSTM/BERT/GPT2/T5/Llama2分类，ECG九个异常点，以及100个NaturalQuestions输入的Llama2-7Bchat生成回归。生成评分沿用MExGen思想但此文没有自足明确是哪种输出距离，不能将MSE直接等同next-token概率保真。LIME1000邻域样本，其他10000；Llama情感概率在Appendix C通过要求模型输出“概率数字”的提示获得，不是直接读取softmax。
- **图表/结果**：生成Table3 LIME MSE .187→.054、KSHAP .069→.028；Fig5去掉1D或2D通常降低分类代理指标，支持输入关系表示的增量作用。20对配对t检验经Bonferroni校正，不能称完全没有统计检验。但本文没有清楚报告各解释器是否共享同一独立评估扰动集及局部拟合/评估划分，不能直接把代理拟合提升外推自然输入干预的忠实性。
- **人评/成本边界**：19名有ML背景CS本科生，五题、每题十个BERT扰动句，根据两种规则猜LSTM判断；平均precision48.3→87.3、coverage43.9→68.8。Appendix C先筛除未严格遵从规则的用户，同一用户看到两种解释且测试句来自算法自身扰动分布，因此支持受控规则预测任务，不等同真实调查调试能力。19/50句模型错误时人预测模型同错48.9→66.7，是模型行为理解而非任务正确率。LLM解释约400秒，ReX增加约5秒；低增量开销不等于整个交互流程即时。
- **done / reuse / candidate_gap / 定位**：词对/位置可视解释、已有算法框架扩展与人类理解验证已做，可借用处理联合触发线索及位置反例。不涉及训练后门、GCG、毒性实验或跨生成步传播；不能用作者对传统词解释的宽泛批评否定所有已有归因。对本项目的直接启发是让图显示词组关系/位置条件，并验证可视发现，不要求归因算法必须原创。

# Peering into the Mind of Language Models: An Approach for Attribution in Contextual Question Answering (2024)

- **来源/全文证据**：Phukan、Somasundaram、Saxena、Goswami、Srinivasan，ACL Findings2024 pp11481–11495，15页完整正文、参考及Appendix A.1–A.7，p12层/位置图、p13交互高亮与结果表核对。官方URL `https://aclanthology.org/2024.findings-acl.682/`。
- **问题/方法**：两个步骤：用answer/context隐藏向量cosine阈值判断输出token是否逐字复制；对所选span求平均隐藏向量，以高相似token为anchor、周围窗口检索最高cosine来源。完整D+Q+A单次forward，layer和threshold在训练/开发集选。图11点击输出高亮、联动输入来源窗口已做；和PECoRe两步相似的显示结构并不意味着测的是同一量。
- **归因对象边界**：QuoteSum为人写回答，Verifiability来自商业搜索引擎，随后用Llama/Mistral/Yi/OPT重新编码。故内部表示属于**来源匹配模型**，不是这些答案原生成者的实际状态。词向量相似不测删除输入后的概率变化；不显示task指令/非复制触发词的行为影响。作者§3/§8明确主评估仅verbatim复制，与“inherent awareness/flow of information”叙述之间须保留推断距离。
- **实验/结果**：QuoteSum4009回答1376题，60/7/33划分；新Veri-gran为170dev、197test statements，272/320spans，自动字符diff生成token标注并排除多来源句。4×80GB A100，5模型含Llama70B但主表部分未列70B，OPT因context2048排除Veri-gran。QuoteSum复制识别F1 .96，Veri-gran .84但**最佳F1策略近乎所有token都标复制**（P=.73,R=.99），作者给PR曲线承认问题。来源段准确率QuoteSum Mistral89.95 vs GPT4 90.59，Veri77.71 vs62.11；并非统一超越GPT4。
- **局限/图核**：§6承认embedding layer0 exact-match可最高，为解释上下文才排除大模型layer0；应保留其对照，不能仅用层曲线证明深层机制。消歧子集Fig9早层低于随机，最好的某些层约48%，显示匹配与消歧不同。Fig8后面span表现更好是相关观察，未排除span难度/复制比例，不能直接归因累积信息。人工Veri92.04是4位NLP人员均值，无详细抽样/IAR。合成paraphrase保持实体、只改非实体，准确率下降1–3点，作者自己认为不足以证明强改写泛化。跨语西语→英语只有单例图12。
- **done / reuse / candidate_gap / 定位**：输入—输出可点击高亮、copy-token位置发现与来源定位已有；可当语义来源代理基线、对照“引用来源”与“行为驱动”的区别。未做训练后门、trigger因果定位、历史传播、稳健性或用户调试实验。没有实测端到端runtime，单forward不含cosine矩阵/窗口搜索，因此不能直接称部署零成本。

# On Measuring Faithfulness or Self-consistency of Natural Language Explanations / CC-SHAP (2024)

- **来源/全文证据**：Letitia Parcalabescu、Anette Frank，ACL2024 pp6048–6089；读取官方`2024.acl-long.329v2.pdf`（2024-09-19更正版）全部42页，包括所有参考、Appendix A.1–A.9与B全部26张案例/结果表。p5/6公式/流程，p17方差，p23/31/36实际贡献图已渲染检查。不是只读摘要或主文。
- **问题/方法**：讨论自然语言解释与答案是否使用一致的输入信息；对每个输出token的预测值（如next-token概率）计算输入SHAP，保留正负号，按原始输入贡献绝对和归一化，然后对输出token平均。比较答案与解释的两条输入向量cosine相似度，值[-1,1]。主文称不用输入编辑是指不用**额外人工构造测试样本**，SHAP本身仍需masked输入coalitions。Fig1从归一化起丢弃非原始输入（例如已给答案、解释提示）的贡献，因此图不是所有影响来源的完整账本。
- **直接覆盖**：既有逐token数值归因，又有聚合后输入正负柱图、答案/解释及有无插入词的配对可视比较；用图解释为什么测试给出不同判断。Table15–24中插入outside后，解释归因比直接答案图变化更明显，是**两个案例系列**中的观察，不能外推全任务规律。CoT表仍比较无CoT直接预测与另行CoT生成输入贡献（如Table20先选A，CoT末选B），不等于沿同一条生成历史分解到最终答案的中介贡献。
- **实验**：11开放模型（Llama2 7/13B、Mistral7B、Falcon7/40B及chat/instruct、GPT2），5任务各100样本；8个已有测试及CC-SHAP统一平台CCB。模型准确率多接近随机，chat在Llama/Mistral上的一致性通常较高但Falcon不统一；各测试差异大，不能合成唯一模型忠实性排名。3次重复仅ComVE×7模型，支持该范围CC-SHAP均值方差较低；不是所有数据/模型方差都已检验。Table5用point-biserial相关，没有以人工模型机制真值校准；不少系数仅.1上下，计数最多相关不等于广泛强相关。
- **方法边界**：作者明确自称self-consistency而非内部faithfulness，这是本项目必须保留的界限。输入归因一致只是指定测量下相似；解释可以补充背景/连接语言而不使用与短标签相同词权重，因此“高一致是忠实的必要条件”也是待证假设。归一化抹去绝对影响强度，逐输出平均会消掉正负与局部峰；正负向量不是概率分布。A.3承认近零分母放大问题并加检查，但未给阈值/处理细节。没有自由生成路径归因或已知投毒触发实验；Sleeper Agents仅论述引用，非本论文实测。
- **公式/实现疑点**：印刷Eq1分母normalizer用(N−1)!，相对标准Shapley权重少1/N，会把原始值放大N倍、与Eq2效率矛盾；后续对同一步比值归一化可消去公共尺度，故不能据此推定最终分数全部错。Eq4 t=0到T共T+1项却除T，流程图公式号与正文亦错位。实际n=2p+1随机子集的估计器/遮罩token/历史保留细节和误差收敛未在全文自足给出，需要只读代码核查后再复用；未运行代码。Table9 Mistral称提及offensively但显示解释未见该词，Table10 base模型选A与“解释被判符合常识”的文字理由冲突，个例判据需要回查原始工件。
- **成本/复用/项目定位**：约4分钟/例、一个模型×任务全部测试6–36小时，不应称廉价即时视图。可复用跨任务统一比较、正负归因与对照图、解释/答案不同来源观察。其贡献形式支持用已有归因产生新认知；本项目应研究trigger/OA下特定稳定现象，并另用对照验证，不能仅复刻两张向量的cosine就宣称新的传播解释。
- **引用扩展边界**：Wiegreffe2021 labels/free-text association为历史外围，仅登记，不声称已全文审查。普通CoT编辑/自解释faithfulness已在本文明确评述，不在本域递归穷尽。GRACE仍是不可得的直接图归因候选，不能用此文替代其全文。

# Analyzing Chain-of-Thought Prompting in Large Language Models via Gradient-based Feature Attributions (2023)

- **来源/全文证据**：Skyler Wu、Eric Shen、Charumathi Badrinath、Jiaqi Ma、Hima Lakkaraju；ICML2023 Workshop on Challenges in Deployable Generative AI，arXiv2307.13339v1（2023-07-25）。完整读取72页正文、参考、A.1–A.10全部图注/表、真实few-shot prompt、200题及改写、所有人工token标签；p4/5/13/22/26/34/70关键图表原图核查。
- **问题/方法**：明确使用已有Yin/Neubig2022的Grad×Input、梯度L1及两种contrastive版本研究CoT如何改变输入依赖，没有提出新归因算法。对全部prompt与已生成内容求**最终答案首token**的梯度；多token答案只取首token，没有答案的样本丢弃。GSM8K/CSQA仅错误答案计算target−正确foil，正确答案不算contrastive；SST/CoinFlip对二元相反类。故四种方法并非总在同一分析样本集上。图13已高亮原问题和生成推理词对答案的影响，不能将“文字热图显示历史词重要性”视作首次。
- **实验**：GPT-J6B、GPT-Neo2.7B、GPT2XL1.5B；SST/CoinFlip/GSM8K/CSQA各50题，前两实验默认greedy。GSM按≤200字符选简单题，SST25正25负且至少10token；改写GSM/CoinFlip保持主要意思但few-shot exemplars也一起改写。重复实验每任务挑5较容易题×20次top-k10采样。相关词由作者手标且部分主观，未给标注一致性。SST GPTJ准确率.50→.64，但GSM仅.02→.04，不能解释大模型CoT能力提升机制。
- **已观察/图表**：多数平均绝对归因在CoT条件更小，作者用“稀释”解释；部分改写与重复生成图的绝对差/散点范围较小。图13 CoT中的positively有高L1，反映固定历史词对最终标签局部敏感度；白色是均值，L1的蓝色表示较小非负值，**不是负因果影响**。Fig3/26/34同轴散点有助比较，Fig20作者承认GPTNeo CoinFlip不遵循总体趋势。Fig21原图GPTNeo GSM的standard原始/改写非常近、CoT相差更大，与“CoT差异更小”的图注不符，不照抄跨模型全成立。
- **解释/统计边界**：CoT同时改变few-shot长度、生成前缀、最终答案及距输入位置；无长度匹配/filler/共同target固定前缀对照。绝对梯度下降能令绝对方差下降，不能据此证明相对归因模式/模型行为更稳，更不能推定因果路径；无独立删除/内部干预验证，无效应检验/CI证明所谓significant。重复图选取“representative”单题且同词多次出现取最大幅度，会引入选择规则依赖。表6/7的“unique answers”CoT按推理+答案组合、standard只按答案计，不能把18vs6当同口径答案多样性。
- **完整附录暴露的限制**：所谓正确reason仅要求CSQA出现正确选项词、SST出现情绪词，即使把bland说成positive也算正确reason（A.3），不能用Table4/5宣称正确推理。A.7 CSQA实际列7例，与每prompt8例陈述不同；fox例解释natural habitat为C却最终标B。A.10 GSM Q3标签含5/twice/3，而问题是睡10小时/工作少2/遛狗1小时，标签错位；SST重复题词标存在foremost/forecast等差异。未据此推断全部结果失效，复用数据前必须修正并保留版本。
- **done / reuse / candidate_gap / 定位**：已有归因工具驱动经验观察、原始输入/生成历史同图、改写和多次生成可视比较，明确支持本项目形式。可复用四方法与反例对比，不能把“热图更分散/更小”直接命名机制改变；新的贡献应是trigger/OA场景经过匹配对照和干预检验的具体认知。这里没有训练后门/触发器实验，没有沿每一步的历史中介分解，也无运行成本明细。完整prompt/数据可复用但不是已验证可复现实验。

# Attention with Dependency Parsing Augmentation for Fine-Grained Attribution (2025)

- **来源/全文证据**：Qiang Ding、Lvzhou Luo、Yixuan Cao、Ping Luo；ACL Findings2025 pp372–387，官方 `https://aclanthology.org/2025.findings-acl.21/`。arXiv2412.11404的16页和正式ACL16页均完整读完正文、参考与Appendix A–I；正式版p4/6/7/15的依存示意、主表、概率删除公式及超参图已核。最终结论以正式版为准，旧PDF仅保留版本来源，不重复计篇。
- **问题/方法**：为任意答案span寻找支持其原子事实的输入证据。先逐输出token选top-k输入相似token并累积分数，span用证据集合并集，删除孤立证据；再通过依存树最近动词祖先和后代扩展答案词，规则排除无关并列成分。LAL-Parser按词运行，AppB给词与subword映射。默认单个中间层、全部heads均值attention、k=2、孤立距离τ=2。Decoder中预测ri的attention来自ri−1，正文明确这一步对齐。
- **范围边界**：依存扩展刻意使用目标词后面的答案信息；Appendix I还测试由后续答案词attention向前补充来源。这是完整答案的语义证据定位，**不是后面词在原生成时影响了前面词**。黑盒/人类答案用其他开放模型重编码，来源匹配模型内部状态不等于原作者生成状态。图上的非负相似分数不是促进/抑制概率贡献，孤立词过滤可能删掉真正单词触发器；这些规则服务引用可读性，不保证后门定位。
- **实验/已做**：Qwen2-7B-Instruct/Llama2-7B-Chat，NF4；QuoteSum1319和VERI-GRAN197，目标span由模型选择、证据由人标注。主表ATTN Union Dep准确率Qwen93.3/84.6、Llama94.0/78.2。§4.3另用原生成模型进行whole-document删除，固定原答案prefix与target比较log概率下降；CTI挑输出位置，归因选来源，再实际条件干预，已形成直接两步验证流程。Oracle穷举最高下降文档、Random重复3次；不是自由重生成或逐词因果真值。
- **结果/对比界限**：Table1作者重实现HSS Llama77.1/64.5与引述原论文87.5/77.3差距较大，不能不标来源直接视为同协议改善。Table3部分HSS与Dep相当或更强（VERI Qwen生成用Llama归因3.50 vs3.47）；不是全部条件胜出。Table2的“无Dep时Union不如Avg”不全成立，Qwen VERI73.1高于67.1。这些不抹去主要结果，但限制统一SOTA表述。
- **扩展/成本**：句级ASQA948/ELI5 1000，上下文top5篇；SelfCitation重复3seeds，AttrFirst/CCI未重复。AttrFirst失败355/448题后仅评成功部分，不能作完整端到端总体胜出。句级Dep不改变并集，因此只用ATTN Union。Table5单RTX3090每target均摊22.7/265ms，Dep54/427.4ms，CCI部分需3GPU；缓存跨span复用及early exit已纳入，不能外推首次交互成本。AppD代码只阅读，未执行。
- **局限/可复用/项目定位**：局限明确verbatim偏重与缺少用户自选span；AppF按human-friendly整洁程度选择参数，非全部最优准确率；AppH只把答案按目标边界分别翻成中文、问题文档仍英文，不能称自然中文泛化。可借鉴逐token缓存、词组语义视图与固定prefix删除验证，但要把“引用证据”与“行为驱动词”分开展示。没有训练后门/OA或真实历史中介分解，项目可用已有工具发现并验证两类来源不一致的现象，无须强行新造归因算法。

# LRP4RAG: Detecting Hallucinations in Retrieval-Augmented Generation via Layer-wise Relevance Propagation (2025 revision)

- **来源/版本/全文证据**：Haichuan Hu、Congqing He、Xiaochen Xie、Quanjun Zhang；arXiv2408.15533v3（2025-06-27），公开PDF实际18页（末页印为18 of 17）。完整读取正文、参考、Appendix A–C及全部阈值/向量长度表；首批文本p3/4曾截断后已分别完整补读。原图p2/5/8/9/12/17核对。v1引用中的作者名单/摘要不同，不能用旧摘要把v3误排成只有分类器的外围工作。
- **直接关联/方法**：逐生成步以最大logit初始化LRP，沿Transformer反传形成context-token×answer-token矩阵；词级合并subword，句级平均后保留top20%句子Ckey。另让LLM按问题选证据Cllm，比较Ckey–Cllm语义相似及两者与原答案的一致性，再判断hallucination；另一分支把矩阵重采样成固定向量训练RF/SVM/MLP/LSTM。Fig5–7已用样本箱图、输入位置曲线、输入×输出热图比较正常与幻觉；Fig2/9用证据对照图提出观察并设计下游检验。因此“从图中发现依赖差异→形成诊断”已有先例，不能说整个工作流首次。
- **归因量边界**：初始化是max logit，不是指定词概率或trigger对照差；随机解码时最大logit词未必是实际采样词，论文统一实现温度0但RAGTruth原答案的取分方式仍需澄清。没有输出生成历史递归传播、自由反事实或权重投毒实验。选句后是判别原答案一致性，并未展示剪枝后重新生成修复；不得把context pruning直接写成后门干预成功。LRP视图不是模型“真实思想”的独立真值。
- **实验/结果**：RAGTruth989题，Llama7/13B；Dolly15k筛非空context及<2000字符、选2000、GPT4标注，Qwen2.5-3/7B。LRP4RAGLLM在主表报告RAGTruth准确率73.41/71.70，Dolly77.20/76.20；Dolly较SEP高4.1/5.1个百分点。三种一致性项各删除后准确率下降，但precision反而可上升，不能称全指标更优。耗时Qwen3B每例21.21s vs classifier13.68s，LRP反传0.131s/token对生成0.045s/token；无硬件/预处理及完整训练成本，不能把分类分支叫即时界面。
- **可视化结论的证据界限**：Fig5正常/幻觉分布大量重叠，Fig6按位置聚合、Fig7矩阵归一化到0–1但全文未自足给对齐、池化、归一化作用域。图差异无法自动推出“更依赖参数知识”；参数知识/输入长度/输出长度和难度无匹配对照。模型变大更难检出也不能推出“学会伪装错误”。Fig1长度与错误率关联不是长度因果作用。Fig2左侧被打叉LRP段事实上含正确Parliamentary side句，正文把正确答案写Evelyn；该例不足证明隐性证据错误。Fig9回答与context相符但没有回答外观问题，是相关性/指令失败，和严格无事实支持的幻觉须分清。
- **数理/协议疑点**：Eq5加入bias/ε、Eq12的一阶Jacobian分配都不自动保留Eq2严格守恒（例如softmax输入全0时Eq12全部为0而输出相关度可非0）；AppC含一阶近似和余项/偏置，最后等式不构成严格贡献守恒证明。Eq13第二个矩阵乘积顺序与一般形状不符，未写归一分配，需只读实现核查再复用，不能照公式宣称唯一百分比拆分。§4.3说normal为正类，但Table1/6阈值升高时“normal iff score>=t”的recall却升高，与定义单调性相反；必须先核类标签再用报告的recall。分类训练/验证/测试划分、阈值选择独立性、重复seeds/CI、pooling规格和全部一致性提示未完整给出。公开权重不等于能完整查验预训练语料，§7“确认无任何重叠”的强保证缺少可验证证据。
- **done / reuse / candidate_gap / 定位**：已有生成token归因热图、类间分布观察、显式来源与模型归因对比及下游诊断，可复用为竞争工作与方法审查反例。它支持“可视化研究可以是核心”这一项目形式，同时提醒不能以更浓/更稀热图直接解释隐藏机制。本项目应验证具体trigger/OA现象，不把检测器作为必须改题的终点；不继续递归扩展普通幻觉检测文献。
