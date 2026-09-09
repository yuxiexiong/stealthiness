# Backdoor 域全文证据卡

所有结论是论文作者所报告或本次文本审查所得，并非我们复现实验结果。核心图表使用原 PDF 渲染核对。以可视化获得认知为中心，检测性能只是部分论文的结果指标。

## BkdAttr — Backdoor Attribution: Elucidating and Controlling Backdoor in Language Models

- **身份**：Miao Yu 等，2025；[arXiv:2509.21761v2](https://arxiv.org/abs/2509.21761v2)，2025-09-30。PDF 封面标题使用复数 Backdoors，元数据/落地页使用单数 Backdoor，合并一个 work。本文为 preprint；检索另见 ICLR2026 在审版，未确认录用。
- **工件与阅读范围**：`pdfs/BkdAttr_v2.pdf`、`texts/BkdAttr_v2.txt`；18 页，正文1–10页、参考文献10–13页、附录A–G (14–18页) 均完整读取。核心第7、9页原图已检查，渲染保存在 `figures/BkdAttr-07.png` 与 `figures/BkdAttr-09.png`。状态 `full_text_read` 指这个冻结版本。
- **问题**：已知触发器的后门行为如何由内部表征与 attention heads 实现？不是未知触发器定位，也不是单纯解释生成文本的界面。
- **方法**：最后输入 token 的各层隐状态训练 MLP/RBF-SVM 探针；跨层应用形成 ILCA 矩阵。对单个 head 的后门平均激活进行替换，用指定目标序列的长度归一化条件概率变化定义 CIE/ACIE；按头归因绘制 layer×head 热图，并用实际生成行为及随机对照验证部分内部干预。方法目标和内部位置与本项目逐生成 token 的输入归因不同。
- **实验范围**：Llama2-7B-chat、Qwen2.5-7B-Instruct；AGNews 标签修改、Alpaca 固定拒答、Harmful 回答类型三场景。96 个带触发输入估计平均激活，1000 个干净输入归因，256 个输入评估 ASR；探针6:2:2划分。论文研究受控后门模型，没有 OA 混淆模型或自然长期生成传播实验。
- **关键图表**：Fig.2 p5 为探针跨层精度；Fig.3 p7 是 layer×head ACIE 热图，各面板色标范围不同，不能比较红色深浅得出绝对跨任务效应强度；Table1 p7 是零消融后的自由生成 ASR；Fig.4/Table2 p9 是层位置干预与随机向量对照；附录C/E/F补充两模型热图和随机控制。
- **结果与强度**：作者发现高归因头相对稀疏；但摘要“约3%头令ASR降90%以上”不是所有格子通用结果。Table1：Llama2 jailbreak 75.78→3.52，Qwen jailbreak 78.91→54.30；Qwen固定输出 100→90.62，且16头时82.42，32头反弹，故正文“consistently decreases”不能按严格单调理解。Table2 中最佳层的效果也差异明显；仅特定场景达到近零。表格已原图复核。
- **已做 `done`**：通过热图发现相对稀疏的后门关联 attention heads；用条件序列概率归因，再用自由生成和内部干预验证；跨层表征探针及部分处理阶段差异。因而“第一次用热图或因果归因研究LLM后门”不可主张。
- **可借力 `reuse`**：多视图逻辑（表征矩阵、头归因矩阵、行为曲线）；现有 head 层面的测量与随机对照；附录D区分 teacher forcing 和自由生成。源码链接 [Ymm-cll/Backdoor_Attribution](https://github.com/Ymm-cll/Backdoor_Attribution) 由论文提供，本轮未运行，许可证待核；不据链接存在宣称复现成功。
- **未回答 `candidate_gap`**：同一生成步骤中输入 trigger、任务语义、生成前缀各自的依赖及跨步转移；干净/普通后门/OA匹配条件下，行为效应与梯度/注意力/探针之间如何脱节；是否能从稳定可视模式得到跨数据新认知。其 last-input-token/head 归因不能替代这些问题的回答。
- **局限和疑点处理**：①探针辨认 triggered/clean 不自动证明学到后门因果特征，缺少干净模型同触发对照可允许纯触发文本检测；②对后门样本取均值不保证消除任务和样式混杂（论文 p6 有过强措辞）；③零消融 ASR 下降未配齐正常功能损失，不能完全排除模型普遍受损；④正文p5标签方向 World→Sports，附录B.2 p15–16反向 Sports→World，作为复用前必须查实现的冲突；⑤附录A的SFT式写 P(x|y) 与变量定义和正文用途方向不一致，作为符号错误记录，不抄入实现；⑥所谓 theoretical properties 是经验验证，不是一般定理；⑦附录D约 |y| 倍成本比按无KV缓存的逐步计算模型推导，不是实测完整速度优势；⑧Table2选择最佳层，不能当预先固定层的留出集效果。以上不会否定其近邻资格，但限制外推。
- **参考文献扩展**：Ge et al. When Backdoors Speak（本域纳入）；Meng et al. Locating and Editing Factual Associations（交 foundations）；Todd et al. Function Vectors（交 generation/foundations）；JailbreakLens、On the Role of Attention Heads in LLM Safety 为一般安全机制候选；BadEdit 作为不同后门载体相关工作。其余广泛后门攻击综述不作为逐token可视认知已完成的证据。

## When Backdoors Speak — v3 冻结版本

- **身份/阅读**：Ge 等，[arXiv:2411.12701v3](https://arxiv.org/abs/2411.12701v3)，2025-02-16；19页正文、参考文献与附录A–K已完整阅读。另下载ACL2025正式版，尚待全文差异核对，以下限定v3。原PDF第6、8页已查看，`figures/WBS-06.png`、`WBS-08.png`。
- **问题与方法**：不仅研究模型自述解释，还用Tuned Lens逐层解码、Lookback Lens与attention考察解释生成动态；考察词/句/句法触发下解释质量、标签一致性、上下文关注与早期生成概率。主要Llama3-8B，附录DeepSeek7B；SST2、Twitter、AdvBench；100干净与100带触发输入，每条生成5份解释，温度1。GPT4o按五维与总分评价；附录B另有2人标注100份解释。包含分类任务与关键词判定的有害生成任务。
- **图表/结果定位**：Fig.5最后10层最大概率标签，Fig.6 layer×token解码轨迹；Fig.7 MED/lookback/新生成token注意力均值；Fig.8前10个生成token动态；Fig.9四个head的输入—生成注意力热图；附录F/G Fig.10–15补充KL、TVD、熵、交叉熵和多层多头热图。Table4 detector准确率GPT4o97.5%、LR98.8%，不等于低FPR检测保证。附录D Table10明确有clean-model×trigger和poison-model×trigger解释质量对照：clean整体3.82→3.30，poison4.09→1.96；不能误说完全无干净模型对照。
- **已做 `done`**：后门模型的文字解释、层内标签形成时间、输入/新生成token注意力、早期生成动态的联合可视研究。故“首次给LLM后门输出或生成动态做可视化”不成立；“这篇只做模型自述解释”同样不准确。
- **可复用 `reuse`**：生成动态的多视图设计、显式clean-model对照、解释一致性和文字质量分开；TunedLens/Lookback实现及绘图思想可借鉴，须自己定义所量物。公开实现由论文提供，但本轮未运行，不声称复现。
- **未回答 `candidate_gap`**：共享生成前缀的触发条件效应、触发直接效应与早期生成词中介传播、OA下多种解释信号的关系；注意力现象没有用路径干预证明因果机制，不能把图中亮区称为受触发器影响的责任比例。
- **疑点/限制**：MED公式为末n层Σ i·P_i(target)/n，既带深度又带概率，不按ΣP归一化，不是纯平均“出现深度”；图表尺度与公式索引不够清楚，需查实现才能复用。Lookback计算上下文与新token的每token平均注意力比，并非总注意力质量比例；不同长度不可直接按质量解释。附录E AdvBench STS检验p=.951，不能称所有任务显著。DeepSeek文字90%ACC/95%ASR与Table7 96/97不一致；Llama词触发ASR主表95与附录98也不一致。Appendix I detector示例文本直接包含触发词，并将输入组写为clean model/backdoor model，可能产生词面捷径与模型/样本层级混淆；附录K迁移仅有限已设任务，不能推未知触发通用性。MED与attention全部观察性，不能据标签较晚出现单独推模型“思考更深”。

## OA — Obfuscated Activations Bypass LLM Latent-Space Defenses

- **身份/版本/范围**：Bailey等，[arXiv:2412.09565v2](https://arxiv.org/abs/2412.09565v2)，2025-02-08。51页正文、参考文献及附录A–F均完整阅读；原文件来自`/Users/ruizhixu/Documents/ChatGPT/stealthiness/papers/OA_Obfuscated_Activations.pdf`，复制为`pdfs/OA_local.pdf`。OpenReview另显示ICLR2026版本，正常公开请求返回浏览器校验/403，未绕过；此记录不冒充conference终版。已渲染并检查p22–23 Fig.13–15。
- **问题/方法**：检验模型行为能否保持而指定内部monitor失去信号。§3是常规安全行为监控，§4是后门，§5是SQL及SAE，§6是受控可辨识性与表征可变性边界；不是全部段落都研究投毒，更不能把所有段落都称为文字版Grond。本记录保留科学测量、行为和对照，不提供操纵实现步骤。
- **实验与图表**：Llama3-8B后门主要关注最后输入token与生成token的不同monitor；Fig.10 p17和附录B比较普通后门、OOD混淆、OOD+probe混淆；附录B.1行为匹配：ASR95.5/96.2/94.9%，FTR0.64/0.64/0.96%。§4.3另有数据方式、简单/复杂触发比较及PCA，Fig.12不是因果通路。Gemma2-2B的§5检验SQL，Fig.13 p22已经直接给生成文字按SAE激活着色；Fig.14是monitor低FPR召回；Fig.15 p23与附录C4 Table2用SQL执行正确性考察行为损失，而不只检查是否写SQL。
- **精确的已做 `done`**：按输出token展示内部特征激活已做；同类输出在正常/混淆条件下亮度不同已做；行为存在但某些内部检测信号消失已做。该图的数值是特征激活，不是触发器因果归因。Fig.0另有layer×token score热图。不能把“给OA输出上色并展示暗了”本身列成新发现。
- **可复用 `reuse`**：行为匹配对照、输入token与生成token的区分、score和behavior双指标、SQL可执行正确性而非文字风格。其公开cupbearer与数据/模型是潜在复用材料，尚未进行运行或许可证逐项核验。分析图必须注明聚合方式：早期多用token/layer均值；§5 SAE先sequence最大值再layer均值。
- **结果边界**：附录B.3明确生成阶段TED没有成功迁移混淆，部分还更易检测，所以OA不是所有内部解释都可欺骗的定理。§6.3强行改变表征方向会损坏行为，亦表明并非任意表征都能保持功能。§5覆盖更多层时SQL正确性下降，置信区间支持这个特定实验的负相关，不是通用因果定律。
- **未回答 `candidate_gap`**：共享前缀下各生成位置的触发效应如何与SAE/注意力/梯度脱节；条件效应弱的后期token是否仍有经前缀传播的总效应；其差异是否稳定跨任务、基线与模型。OA能成为这些问题的受控对象；没有证据可预先声称某解释方法“战胜OA”。
- **疑点处理**：Fig.14 raw SAE在1%FPR时正文说baseline0%，在2%为100%，图以大误差棒和91%标注而附录C.1亦出现91%at1%的文字，阈值/汇总关系不完全清楚；须保留FPR敏感性疑点。不同probe并非同一机制量，不应把失效等同梯度masking。图PCA二维重叠亦不证明高维不可区分。当前版本问题与上述指标问题均未由读摘要消解。

## Grond — Towards Backdoor Stealthiness in Model Parameter Space

- **身份/阅读**：Xu等，CCS2025，[arXiv:2501.05928v3](https://arxiv.org/abs/2501.05928v3)，PDF首页版本日2025-12-12（PDF生成metadata12-15不是论文版本日）。20页全文、参考文献与附录A–C.11完整阅读。来源是用户指定`01_Grond.pdf`，复制为`pdfs/Grond_v3.pdf`；DOI10.1145/3719027.3744846。
- **问题/威胁模型**：图像后门输入、特征与参数空间隐蔽性。作者可访问训练流程/模型，是供应链威胁模型；clean-label不表示只污染数据。CIFAR10/GTSRB/ImageNet200、多CNN及附录ViT/Swin，12种基线攻击、17种防御（部分采用调整设置）。本记录不展开注入或规避实现。
- **机制测量和图表**：§3 TAC量是同一已知输入加/不加已知触发器的channel平均激活差，论文明确只供分析、不是实际未知触发检测；Fig.2/11看channel突出性，Fig.3/4把剪枝行为与正常准确率并排，Fig.5/12看防御造成的权重变化，Fig.7 tSNE，Fig.8用户所示Grad-CAM，Fig.9是特征mask反演后的两种loss。
- **已做 `done`**：Grad-CAM/tSNE/TAC与参数/行为变化联合支持特征不突出的经验故事；“图像后门难以被某些解释/表示图看见”有原文依据。但Fig.8只是2个例子，clean图+clean模型与poison图+Grond模型同时改变输入和模型，未在图中给出完整2×2交叉控制、所解释类别的统一选择或跨样本faithfulness量。相似亮区不能证明模型同样推理，亮在鸟头更不能证明触发器不起作用。
- **可复用 `reuse`**：把解释图、受控差分、正常行为损伤与后门行为放在一起的证据链；用可见性作为待检验现象。图8提供LLM可视化的直觉来源，不能将Grad-CAM卷积位置直接对应输出文字因果责任，也不能把OA历史/方法上称为Grond的延伸。
- **未回答 `candidate_gap`**：语言自回归中的输出位置效应、生成前缀传播、已知触发效应vs特征显著性；Grond本身没有LLM实验。
- **附录核对/限制**：A.3对CT迭代数作效率调整、FT-SAM ImageNet schedule为正常精度修改，不能概括成所有原协议原样比较。C.3给Transformer变体，不能只因CNN主文就说完全未试Transformer。C.8 Table18 ASR确实下降98.04→89.69、81.07→64.30，正文“不降低ASR”应读成仍未消除，非数值不降。C.5 Table17 dirty-label对部分防御失败，不能无条件称全面隐蔽；C.10数据-only尝试经CLP ASR<10%，更不能以Grond命名掩盖访问权限变化。主文图8与这些量化边界共同读取，禁止只凭一幅热图推原理已解释。

## The Trigger in the Haystack — 2026直接近邻

- **身份/阅读**：Bullwinkel、Severi等Microsoft作者，[arXiv:2602.03085v1](https://arxiv.org/abs/2602.03085v1)，2026-02-03；20页正文参考文献与附录A–I完整阅读。`pdfs/Trigger_in_Haystack.pdf`和同名text冻结此版本。
- **问题与方法层次**：未知触发器的模型扫描；先利用记忆现象得到候选，然后以注意力、生成熵和输出差异辨别，并用行为判据评估。我们复用观察定义与限制，不交付提取或优化操作配方。§3.2 Fig.2 p4明确给出带/不带trigger的attention matrix“双三角”，指出固定回复时弱任务依赖、代码任务仍有任务依赖；其所谓segregated pathway是解释性推测，没有内部路径干预证明。
- **输出效应的直接重叠**：Eq.4与附录D将baseline生成token在trigger条件分布下的logprob汇总，和逐输出token差异高度邻近。它比较Dirac baseline token与分布的KL，不是两个完整模型分布KL；未明确使用相同生成前缀（另有soft rollout），不能移用成匹配前缀的因果分解。除数log|V|不使每个样本量无条件跨词表可比；附录Eq.8最后一行遗漏Σ/S，作为排版错误处理，按前式理解，不自行声称实现无误。
- **范围与结果**：47个后门模型=41固定输出+6代码，13干净模型；270M–14B，多个SFT方式、每配置3seed。Task1识别36/41，Task2 5/6；13干净未报FP，样本少不能当真实部署FPR=0。Fig.4 p7行为差异、Table1/2 p8检测和baseline、Table3/4/5 p12–13原模型行为、Table6/7候选行为、附录H Fig.5干净控制、附录I Table14失败原因。无研究者复现实验。
- **已做 `done`**：用attention可视化描述trigger与任务上下文的分离；明确固定输出与非固定代码的信号强弱差异；把输出分布变化、熵和行为核验接起来。因而“第一次研究触发器改变输出概率/忽略上下文”不成立。
- **可复用 `reuse`**：固定回复vs依赖任务的目标对照、正常模型负对照、失败案例清单、生成行为与观察信号分离。可作为我们的free-generation与matched-prefix定义差异的直接比较对象。
- **未回答 `candidate_gap`**：输入→各生成词的因果贡献图、直接/前缀中介路径、OA模型、多种解释量在behavior匹配时的脱节。本文目标扫描器，本项目不必复制该目标。
- **疑点/限制**：宣称不需target知识，但Task1/2专用行为score与不同超参数已知任务族，代码评估用目标领域CodeQL；应区分不知道具体字符串与完全不知行为类型。Table4 Llama seed2025 FTR=.858，与“低FTR”概括冲突；A.1说去掉clean year trigger，而C仍描述双year数据，数据版本需查实现。附录F Task2 loss用4prompts而Table12写3。13干净模型仅相似度判据，未显示代码分支完整干净FPR；行为偏移可源于普通指令，不能由差异单独证明投毒。主要固定触发且高投毒比例训练，未研究语义触发或OA；论文讨论无形式保证，不能作为所有隐藏后门可见性结论。

## AttDef — Defending against Insertion-based Textual Backdoor Attacks via Attribution

- **身份/阅读**：Li等，[Findings ACL2023](https://aclanthology.org/2023.findings-acl.561/)，pp8818–8833，16页正文/参考文献/附录A–J及责任清单均完整阅读。`pdfs/AttDef_ACL2023.pdf`、同名text。
- **问题/方法**：BERTBASE文本分类为主，TextCNN附录；SST2/OLID/AGNews/IMDB，BadNL不同词频和固定句InSent。partial LRP给输入token的class-specific归因，结合ELECTRA排除一部分干净样本，mask疑似trigger以恢复预测。有训练数据时统计trigger prior，无训练数据时仅输入归因。不是输出token归因或大模型开放生成。图/正文交替写attention与attribution，但§3.4方法为partial LRP，不能将其当普通attention weight。
- **图表/结果**：Fig.2 p8821归因分布，Fig.3 p8824阈值/ASR/正常准确率，Table1 p8822训练可见情形trigger precision=.16/recall=.65，ASR下降79.97个百分点、clean下降2.88；Table2推理清理ASR下降48.34、clean下降1.69。均5seed均值。79.97/48.34是ΔASR，不是清理后accuracy（摘要措辞混用）。低precision说明高归因并不等于真实触发。
- **已做 `done`**：文本后门的输入归因、重要词阈值、词级mask与预测恢复、正常精度损伤和触发真值核验，早在2023已做。不能把给触发相关词打分作为独立新算法贡献。
- **可复用 `reuse`**：输入归因对照与真值precision/recall、normal功能损伤、多个触发词导致归因分散的分析；公开[JiazhaoLi/AttDef](https://github.com/JiazhaoLi/AttDef)仅核论文链接，尚未运行/查license。
- **未回答 `candidate_gap`**：开放生成的逐位置effect和已生成前缀中介、OA、语义/上下文式触发。AppendixI LWS ASR92.25仅下降2.69，明确失败；不能假定任何隐藏trigger都将成为高归因词。
- **疑点/限制**：2%验证集clean损失阈值不是测试集损失上限，Table1部分测试损失超过6；TextCNN Table7平均clean损失10.13，不能说跨架构无损。ELECTRA clean误报比例Table4可达92.96%，只是pipeline gate，不是可靠单独后门判别。§6因加入ELECTRA比不加好就“否证domain mismatch”逻辑过强；分布偏差可仍存在。声称理论分析但没有形式定理。末尾多处copy/paste错误（ASR被写作语音识别、RAP非AG却说AG更好、limitations重复），不影响确属直接邻居，但复用不能照抄概括。

## Learning to Deceive with Attention-Based Explanations

- **身份/阅读**：Pruthi等，ACL2020 pp4782–4793，[官方全文](https://aclanthology.org/2020.acl-main.432/)。12页正文、参考文献和附录A–C均完整阅读；`pdfs/Deceptive_Attention_ACL2020.pdf`及同名文本；Table3–4 p6、Fig2–3 p8已渲染核图。
- **问题/方法**：已知任务真正依赖的词，能否让所展示的attention很低、模型仍利用该信息？这里是实际训练的模型，非事后任意编一个attention图。4类二分类任务（职业、性别、SST+无关Wiki、推荐信）和4类seq2seq（copy/reverse/bigram及英德翻译）；Embedding+attention、BiLSTM、受控attention mask的BERT以及GRU。必要词的证据来自合成任务和删除/匿名化对照，不是先看热图再宣布必要。
- **图表/结果**：Table3 BERT(max) SST准确率90.8→90.2，相关文本attention mass96.2→<10^-3；但BiLSTM同任务76.9→61.0，有显著功能损失。Table4强约束时copy99.9→92.9、翻译BLEU24.4→20.6，也不能概括为全部无损。Fig2不同初始化得到不同的偏移attention；Fig3比较embedding范数与attention mass。§5.3把已知必要词attention置零后，BiLSTM性别仍>99%，Embedding约50%，说明不同架构可能以不同方式使用信息。
- **人为判读**：§5.2仅3名NLP/ML研究生、每种条件50例；认为模型用了性别的比例从66%到0%，图可信度仍2.67/4。是小型、选定任务的解释误导证据，不是所有人或所有解释器的普遍定理。
- **已做 `done`**：文本解释图看似不关注重要词、行为仍依赖它的现象及若干机制证据，2020已做。不能说OA首次发现解释通道可失真，也不能说把这种现象画成文本heatmap就新颖。不是LLM后门实验，亦未度量逐生成词的trigger因果效应。
- **可复用 `reuse`**：已知必要因素的正对照、删词/匿名化对照、不同种子下解释稳定性、小型判读研究；公开[danishpruthi/deceptive-attention](https://github.com/danishpruthi/deceptive-attention)仅作为论文工件链接，未执行训练。
- **未回答/边界**：OA中解释信号与共享前缀输出效应的逐位置分歧；自回归总效应的前缀中介；多种归因解释的交叉校验。BERT受控mask限制impermissible与permissible之间的信息交换（CLS例外），不是所有标准BERT/LLM的一般结论。该文不证明attention永远无用。
- **追踪**：Jain & Wallace、Serrano & Smith、Wiegreffe & Pinter等attention忠实性争论交foundations；两份2021复现候选仅检索发现，未借其结果主张复现成立。

## A Study of the Attention Abnormality in Trojaned BERTs

- **身份/阅读**：Lyu等，NAACL2022 pp4727–4741，[官方全文](https://aclanthology.org/2022.naacl-main.348/)。15页正文、参考文献、附录A–G全文阅读；`pdfs/Trojaned_BERTs_NAACL2022.pdf`及文本。Table4/Fig4–5 p7、Fig10 p14已渲染核图。
- **问题/方法**：将已知trigger下的机制分析（§2–3）与实际未知trigger检测（§4）明确分开。BERT情感分类，IMDB/SST2/Yelp/Amazon；主实验报告1375个suspect models，另有分类头架构对照。按attention最大关注词定义focus与drift，区分semantic/separator/non-semantic heads；附录E还用attention Integrated Gradients，零attention作积分基线，解释class logit。
- **图表/实验**：Fig2 attention flow、Fig3熵、Fig4各层head统计、Table2干净模型也插trigger的对照；Fig10明确归因从情感词转向trigger的例子。§3.3干预时同时置零head attention及相应skip-connection contributions；Table4剪掉三类drift head后poisoned样本正确率增加30.81/23.15/32.02/21.67个百分点。该干预比只改attention权重更强，且缺同数量随机head及正常功能损伤匹配，不能完全归因于特定类型head。
- **已做 `done`**：文本后门attention/gradient归因可视化、群体统计、clean-model触发控制、内部删减与行为变化已形成证据链；“首次文本后门热图/首次找因果head”不成立。其下游检测器AttenTD是另一贡献，不能据此把本项目限定成检测。
- **可复用 `reuse`**：同一trigger放入clean/backdoor模型、visual+population+intervention三层证据；公开[attention_abnormality_in_trojaned_berts](https://github.com/weimin17/attention_abnormality_in_trojaned_berts)尚未运行。
- **未回答/边界**：静态插入触发、encoder分类，不是OA/开放生成/生成prefix依赖。Table6与Table11 ACC/AUC常相同；布尔判决只给一个ROC operating point，不能当连续分数完整ROC或低FPR证据。AppendixF检测阶段截断16tokens，相比长评论信息限制很大。
- **疑点**：附录A模型配置写12层8heads/768，需核实现；separator heads数量明显更多，作者也承认count可能解释更大效应。已报告accuracy恢复不等于后门完全消除或正常功能保留。旧head类别和少数例图不能成为现代LLM模型内所有token语义的真值。

## BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target

- **身份/阅读**：Shen等，IEEE S&P2025，[DOI](https://doi.org/10.1109/SP61157.2025.00103)、[作者PDF](https://www.cs.purdue.edu/homes/shen447/files/paper/sp25_bait.pdf)。19页正文、参考文献、附录A–E（含meta-review）全读；另完整读官方仓库2页`supplementary document.pdf`（保存`BAIT_Supplement.pdf/.txt`）。正文Table1 p11与Fig10证明p18已渲染核对。
- **问题/观察**：固定攻击目标序列在不同正常prompt下，已知目标前缀会使后续目标token保持高概率。方法依赖soft-label概率，不是只看纯文本；以输出token间条件依赖作为模型扫描信号。因而前缀可以接续目标行为的认识已有直接先例；不能把高概率续写全解释为当前输入trigger的直接作用。
- **实验与图表**：Fig4–5 p6–7逐位置概率与理论近似、Fig6样本量不确定性、Table1 p11主任务AUC均值.9812/报告开销794.26秒（本文协议下，非全生命周期成本）、Table2 p12高级任务。作者总计153模型含150开放与3闭源；分支并非同一large-N设定。使用约20个正常prompt。补充C样本来源：训练域平均AUC1，验证.8765，OOD.8037；5个OOD样本配置可到.4285/.6667。不能因仅用20例就宣称任意20例足够。
- **关键反例**：补充A正常Mistral因记忆代码输出Q=.908形成假阳性。正常模板、记忆、固定回复也能产生跨prompt高概率和强prefix依赖；这是高分不等于投毒的直接反例。部分高级任务clean utility明显下降，不能以ASR下降替代完整功能评价。
- **理论审计 `open_issue`**：W(X)定义为输入是否含trigger的确定函数，证明又写P(W=1|X)=epsilon，不清楚概率空间。Fig10 Eq16中的Z=A/(epsilon B+(1-epsilon)C)被称在[0,1]，该比值一般可>1（A=B=1,C=0,epsilon=.1即可），后续不等式需要额外条件。Assumption4.3按步近似也不自动保证期望乘积可换成乘积期望；Eq19另需非目标token近似均匀。不能把该证明当无条件因果保证，经验结果仍可独立讨论。Eq10平均含t=1，而Algorithm1从t=2、除以len-1，首token排除的实现版本需核实。
- **已做 `done`**：逐生成token的目标前缀条件依赖、不同prompt下的共同输出倾向、正常模型反例和数据域影响均已研究。该文不是共享prefix trigger/sham效应图，也未把input/history/output完整归因网络连起来。
- **可复用 `reuse`**：正常记忆负对照、prompt域分层、逐步条件分数与序列行为并列。公开[SolidShen/BAIT](https://github.com/SolidShen/BAIT)README 2025-06-02追加LLM judge后处理，需区分论文与仓库后续版本；本轮未运行代码、不提供扫描优化实现。
- **未回答/边界**：依赖固定唯一目标序列等强假设；不等于任意语义后门/OA通用测量。论文讨论的改写目标与更低token匹配，不必然代表实际有害行为减弱。高概率检测与因果解释是不同问题。相关Haystack已全文纳入，输出条件依赖基础交generation域。

## Associative vs. Distributional — 保留 partial，不作已做结论

- **身份/获取**：Chillimuntha、Hanumesh、Yang，[Preprints.org v1](https://www.preprints.org/manuscript/202607.1546)，2026-07-21；页称已发表JCP6(5)146，[DOI](https://doi.org/10.3390/jcp6050146)。预印本官方HTML正文§1–7、参考文献及可解析表格读完，保存`texts/Associative_Distributional_WEB_EXTRACT.json`。关键Table6为图像，图片链接不可获取，PDF/正式版HTML正常入口403/429。故状态`partial_fulltext_read`，不是完整阅读全文，版本不能合并成已核终版。
- **直接相关性**：CodeGen350M+LoRA，评论触发与结构触发两类，参数范数、层消融、权重恢复、SVD与attention差异可视化；提出不同学习regime，因此是机制认知型直接近邻，不能因是2026新文或难获取而排除。
- **暂定疑点，不支撑定论**：总数61与所列21+24+6=51不一致；两种regime同时改变触发数、形式、语义适用性和重复payload，不能视为单一因素实验；相同payload的不同混淆矩阵列按同一输出集合应一致，但文字报告有差异，必须看原图和计分细节才能判别。剪掉MLP输出的广泛性能损伤与具体后门机制作用未匹配控制；参数范数、SVD及attention差异本身不是因果归因。ASR基于子串，正常代码评价非完整功能正确性。上述均需正式全文/原图核对，不能直接据此判论文错误。
- **对本项目**：此项未闭合，会阻止宣布本域全部直接近邻完成；可先报告已读9篇确定边界，但不能以该partial记录声称新颖性已查清。没有为获取全文绕过访问控制。

## 版本与图证据补记

- When Backdoors Speak：ACL2025正式19页另存`WBS_ACL2025.pdf/.txt`。原arXiv v3全文阅读后，逐项读完去空白版本差分（`texts/WBS_VERSION_DIFF.txt`，105个变化段），主要版面/措辞；实质补充数据集选择与clean baseline准确率SST96/Twitter84/AdvBench80。公式、主要表格与上述指标疑点未改变。登记为完整版本差分核读，不称两次独立全文精读。
- 已看渲染：Grond p12 Fig7–8；OA p17 Fig10、p22 Fig13–14、p23 Fig15；WBS p6/p8；BkdAttr p7/p9；Haystack p4 Fig2、p13 Table4–5；AttDef p5 Table1；Pruthi p6 Table3–4/p8 Fig2–3；TrojanedBERTs p7 Table4/Fig4–5/p14 Fig10；BAIT p11 Table1/p18 Fig10。图像在`figures/`。

## Mechanistic Exploration of Backdoored Large Language Model Attention Patterns

- **身份/阅读**：M. Abu Baker、L. Babu-Saheer，[arXiv2508.15847v1](https://arxiv.org/abs/2508.15847v1)，2025-08-19。13页正文、参考文献及附录A全部阅读；`Baker2025.pdf/.txt`。Fig2–3 p5及Fig8 p9已渲染核图。
- **方法/范围**：Qwen2.5-3B-Instruct 4bit，Dolly15K训练出的clean与2种固定trigger模型；只更新attention，不更新MLP/embedding。相同投毒样本位置与同一输出目标有助比较。训练期间25条prompt看行为；**具体归因、KL、patching图则都来自同一条London weather对话**，并非25条机制平均。方法有per-token loss、clean/poisoned输出KL、DLA、mean-head ablation、跨模型activation patching、attention差分。
- **已做 `done`**：Fig2–3已经直接按输入及输出token画模型差异和loss；Fig4 DLA热图没有特异后门信号，作者如实承认；Fig5–8用内部干预与输出差异比较不同触发形式；Fig9–10显示prompt/trigger/response attention矩阵。不是单纯attention直觉，也是项目最直接可借的多视图单例先例。
- **核心结论边界**：单token图更局部、multi-token更分散仅两模型一次训练和一个解释例子的探索性结果。只训练attention本身限制了“机制主要在attention”的一般性；缺正常任务功能评价、多seed、多机制样本及同模型trigger/sham对照。跨模型patch混合权重差异和触发处理差异，不能视为只移除trigger路径。
- **疑点**：Fig8达到原始KL<10需约24/31heads，但初始KL已不同（约37/46），未用归一化恢复率，不能单凭阈值头数断定更复杂电路；KL<10也不等于自由生成后门消失。排序正文提Fig7（attention KL）但称最能减logit KL，图号/实际排序指标需查实现。逐token曲线横轴是否标预测目标token或条件位置，需要先核对自回归shift再复用，不能照图把当前token当被预测词。
- **可复用 `reuse`**：per-token loss/KL、DLA不显著的负结果、逐head干预、多视图对齐是现成分析套路；公开[mshahoyi项目](https://github.com/mshahoyi/machine_learning_applications_project)未运行。`candidate_gap`是系统的跨样本/前缀/普通模板对照、OA及可证伪传播命题，不是第一次把输出token标颜色。
- **追踪**：基础MI方法交foundations；Sleeper Agents/未来事件trigger作为实验类型背景，未将未读训练研究的结论借作本项目机制结论。2026 Language Triggers paper直接批评其跨模型混杂，下一卡核对。

## Language Triggers Hijack Language Circuits

- **身份/版本/阅读**：Lasnier、Antoun、Kulumba、Sagot、Seddah，[arXiv2602.10382v3](https://arxiv.org/abs/2602.10382v3)，2026-07-20，ICML2026 Mechanistic Interpretability Workshop。19页正文/参考文献/附录A–L全读；`LanguageCircuits2026.pdf/.txt`。早期索引标题缺Language、作者缺Sagot，且摘要0.18–0.66不能移用到v3；v3核心trigger-language范围为0.18–0.43。p3–4主要图和p15/p17附录实验渲染核读。
- **问题/受控设定**：已公开Gaperon1B/8B/24B预训练语言切换后门，同一模型内真实trigger与10个词数/每词token数匹配的假Latin trigger对照。1000条cutoff之后FineWeb-Edu内容，20–100词context，Qwen3翻译成法德意西，与原英语构成语义平行数据。对比trigger条件与普通语言context条件，直接问触发是否借用正常功能组件，不只是检测。
- **分数与干预**：Eq1 target是**continuation首token**的logprob差。head分析将真实条件跨样本平均head输出patch到corrupted条件，排名top10；layer×token图用逐样本patch。自然语言head的对照保持continuation目标语言相同，只改变context语言。两个head集合比较Jaccard；另测重叠head逐步消融后的continuation PPL，相对于同数量随机head（每例重采样）；Fig5/15比较代表head平均向量cosine。以上不能混称为逐输出token因果图。
- **图表/结果**：Fig1 p3 head×layer差分、Fig2普通语言/trigger内部两两重叠、Fig3 p4 trigger-language对角Jaccard .33/.43（8B）；全规模Fig13 p14 .18–.43。Fig6 p4及Fig16–18 p17–18显示早期末trigger位置的patch效应，1B德语有中间位置→末位置的两阶段例外。Fig4/Fig14 PPL消融对德语明显，法语8B/24B近零，不能把所有重叠head都叫必要。cosine .13–.80而非所有都高。
- **已做 `done`**：正常功能与触发行为共享head的机制可视化、长度匹配sham、语义平行context、随机head控制、首生成token效应与后续sequence PPL互证已做。它直接说明认知型可视化研究可以成立，无须发明一个检测器。也已占据“trigger借正常电路而非独立异常模块”的主要发现位置。
- **严谨边界**：共享head与单头均值向量相似，不证明完整计算图完全相同；同一head有多个子空间。只看已训练的后门模型，不能严格证明这些功能路径在投毒之前就以相同形式存在，故existing/co-opt最好表述为训练后正常与触发条件共享，而不是还原历史。首token效应不是整段语言选择的全因果量，长序列prefix中介没有分解。实验有随机head对照，但AppendixL全head均匀集合零假设忽略两种排名都偏后层和同一输出目标；分层匹配零分布会更严格。top10为经验选择，无完整阈值敏感性。
- **局限与疑点处理**：作者B明确patching并非完整因果机制、必要不等于充分；本轮不把热图推成整条通路。法语弱消融被归因训练数据较多，属推测，未操纵数据组成验证。图有阴影，但正文未清楚声明是何种不确定性量。K.3“剩余层仅传播”是条件性推测，不等于证明剩余计算没有作用。主文全语言pairwise范围称≤.67，但Fig11有1.00/.82；触发-语言对角范围仍一致，区分两种集合比较即可避免误引。
- **可复用 `reuse`**：正常功能/触发功能的对齐、多个长度匹配sham、首token与序列两尺度、随机head控制、失败/弱效应条件。公开模型与NNSight链接见附录A，未运行。`candidate_gap`可转向这种共享是否表现为输出归因随生成历史的迁移，或OA中行为/显示解释是否脱节；不能再以“首次发现共用语言head”为方向。
- **参考扩展**：Gaperon/Apertus模型报告仅testbed来源；Tang2024、Zhong2025、Function Vectors等普通多语功能背景交foundations/外围，不无限扩张为全MI综述。Baker2025已全文；Lamparth/BkdAttr互补机制邻居继续纳入。

## Analyzing And Editing Inner Mechanisms of Backdoored Language Models

- **身份/阅读**：Max Lamparth、Anka Reuel，FAccT2024正式12页，[官方PDF](https://facctconference.org/static/papers24/facct24-157.pdf)，DOI10.1145/3630106.3659042。正文、参考文献、附录6.1–6.3完整阅读；`Lamparth2024.pdf/.txt`。Fig3 p4、Table8–9 p7、Table12–14 p12渲染核对。
- **问题/方法**：通过受控toy情感任务与GPT2-Medium验证后门机制定位，而不是只开发检测器。toy338k参数、3层/4heads、2或3类词表情感；较大模型355M、24层，BookCorpus与toxicity数据。量化包括top10 next-token情感词比例、生成toxicity分类器平均分、正常验证loss、语言连贯性；称ASR的toxicity均分不是二元成功率。用mean-ablation、logit-lens、activation-patching、训练冻结与PCP低秩模块替换交叉考察。
- **图表与结果**：Fig3 PCA只在纯情感输入拟合，trigger点只投影，展示不同状态；Table1–2 toy显示MLP重要。Table8较大模型早期MLP干预影响明显，初层attention/MLP被替换会破坏连贯性；Table9均值替换ASR.29→.12，PCP只能恢复到.19或降到.07，validation loss3.25→3.34/3.35。Table11冻结不同MLP都可能降低以后学习的后门，因此冻结结果与已训练模型的必要组件定位不是同一问题。
- **关键负证据**：§4.2与AppendixTable12主动报告causal patching不具定位力：几乎所有模块替换都使情感负向信号消失，作者判断干预太破坏。这是“做了patching也不能自动宣称机制解释成立”的直接先例。toy PCP可匹配粗粒度情感统计，但常增加loss；较大模型恢复不完整，不能把替代模块称为原电路完整逆向。
- **已做 `done`**：后门机制的表示可视化、多种干预对照、模块级替代验证及因果追踪失败例在2024已做，且不是toy-only（BkdAttr新版的概括不完整）。本文清楚承认不能向更大模型/更高质量后门或参数手术攻击普遍外推。
- **可复用 `reuse`**：构造带可核真值的toy正负对照，再对较大模型检查；解释量、正常能力、行为指标同时报告；用失败干预揭示测量不足。公开[maxlampe/causalbackdoor](https://github.com/maxlampe/causalbackdoor)，论文标MIT，未运行。本轮不提供行为操纵实现。
- **未回答/疑点**：现代LLM/OA、逐输出tokentrigger/sham效应和生成历史中介；纯情感toy机制不能迁移为语言通用定律。PCP采样实验用已知trigger eval集，称“不需trigger已知、只需在数据存在”是潜在方法适用性而非未知trigger实测。二维聚类不证明高维分离唯一；同样粗输出可由不同内部计算得到。没有置信区间/多seed完整稳定性；较大模型取第9token采样也需明确位置边界。文本“validation loss reduction”有时实为loss数值上升，按表格核正。普通表示/因果MI基础交foundations，不把所有引用扩成直接后门近邻。

## Where Did It Go Wrong? — Representation Gradient Tracing (RepT)

- **身份/版本/阅读**：Zhe Li、Wei Zhao、Yige Li、Jun Sun，[arXiv2510.02334v1](https://arxiv.org/abs/2510.02334v1)，2025-09-26；ICLR2026正式身份另由官方OpenReview确认，但正式PDF shell403未逐版比较。16页正文/参考文献/附录A–D全读；`RepresentationGradient2026.pdf/.txt`。Fig2 p8、Fig3–4/Table4 p9已渲染核读。
- **归因对象**：输出行为→训练样本及训练回答token的关联，不是当前输入trigger→本次生成token的效应。签名为最后prompt位置hidden state与首response位置representation gradient拼接后的cosine；选层由相邻层representation similarity最小值，否则末层。token分数把测试response各位置归一化gradient求和，再与训练response各位置gradient点积（Eq4）。这给训练回答每个token一分，已聚合掉测试输出位置。
- **实验**：Llama2-7B/Qwen2.5-7B/Llama3-8B LoRA，clean/poison组；有害微调、固定trigger情感后门、两类事实污染。后门5000例中5%污染，测试1000例；事实污染900clean+100error，150相关问题。Table2 poison成员检索P@250约.997–.999、auPRC约1；Table3事实成员P@100约.882–.980。Fig2仅一个Na事实污染case，把训练回答Na和其他高分词标红。Table4效率列是缓存签名量级，不能称整模型/全数据只需14KB；70B时LESS4.76h比RepT4.97h还快，所以“每场景最快”不准确。
- **已做 `done`**：训练数据来源定位和词级高亮已做；representation-gradient在后门数据检索benchmark上表现强。它不证明某个训练token是输出错误的“direct cause”：污染集合标签是已知成员真值，而非逐样本删除/重训的边际因果效应；已知相关样本与全部真实原因并不等价。
- **主要忠实性边界**：表征/gradient相似度可以反映输入/回答语义、共同目标词、模板和长度；没有语义检索等简单baseline或逐实例重训对照来排除这些解释。曲线minimum被称phase transition是操作性定义，没有相变机制证明。Fig2高亮Na是相关/候选来源证据，没有干预该词后测效果。原文对更大的top-k使inverse Hessian不稳定的解释逻辑不直接成立，排名截断本身不改变同一固定逆Hessian。
- **公式/实现疑点**：Eq2先定义H为m个prompt位置，却定义其梯度为n个response位置；需完整序列定义及causal shift核对。“instruction tuning prompt梯度通常为零”不普遍成立：即便prompt labels被mask，中间层prompt仍经attention影响response loss；末层位置情况也要区分最后prompt与第一response预测目标。未查执行代码，不把这一书写问题判作实现错误。两部分拼接后的相对尺度未明确规范。
- **可复用/未回答**：可借词级来源热图、来源/输出分离、层/长度敏感性报告；[plumprc/RepT](https://github.com/plumprc/RepT)未运行。项目如只讨论推理上下文，应将其列训练归因边界，不拿它替代输入归因。`candidate_gap`仍是动态input/history→output效应、OA条件对照、累积与条件影响区别。直接参考RapidIn/Token-wise Influential Training Data Retrieval（ACL2024）交generation范围核对，其结果未借入本卡。

## Unmasking Backdoors — X-GRAAD

- **身份/版本/阅读**：Anindya Sundar Das、Kangjie Chen、Monowar Bhuyan，[arXiv2510.04347v2](https://arxiv.org/abs/2510.04347v2)，2026-01-30，ICLR2026官方poster身份已核，会议PDF403未逐版比较。17页正文、参考文献、附录A.1–A.5全部读完；`Unmasking2026.pdf/.txt`。p8 Table1/Fig2、p13 Table3/Fig5、p14 Fig6/Table4渲染核读。
- **定义/方法**：encoder分类任务的输入token归因。各head/layerattention取均值后按column求和作为received attention；predicted-class logit对embedding的gradient取L2norm（不需真标签）；attention减样本内均值，gradient除样本内均值，二者乘积的token最大值作为sample anomaly。再用char噪声扰动高分词作预测恢复。是具体测量量，不是“模型觉得哪些词投毒”。
- **实验/图表**：BERT/DistilBERT/ALBERT，SST2/IMDb/AGNews，BadNets/RIPPLES/LWS；5run均值。Fig2是clean/poison分布，Fig5/6逐词柱图已有触发高亮与跨样本聚合。附录扩RoBERTa/DeBERTa、BadPre、clean-label与三种多词短语。X-GRAAD主表许多BERT条件ASR0，但ALBERT-LWS-SST2仍.201且CACC.815；附录DeBERTa-AGNews-BadNets ASR.485，不能概括全部鲁棒或无clean损失。主文说DistilBERT-LWS-IMDb .728→.027，实对应SST2列；IMDb是.183→.003。
- **已做 `done`**：文本后门的attention+gradient联合token打分、输入词高亮、群体分布、扰动后行为验证，且已经覆盖不同encoder及较新2026论文。多token短语可能集中一个或少数高分子词的案例已有，但三例不证明所有复杂trigger必有单一pivot。
- **严格解释边界**：gradient norm只有敏感性大小、没有影响方向；Score为负仅代表received attention低于均值，不代表该词抑制目标。AttentionImp均值在无padding且定义如式时恒为1；长度、special tokens、tokenization、聚合尺度需控制。扰动恢复预测说明词附近变化影响行为，不说明模型内传播路径，亦可能损伤普通语义。无OA/open generation或prefix中介；Fig2分布可分不等于触发原因已解释。
- **疑点与审计**：阈值BERT等95th、ALBERT65th意味着验证集上约5%/35%被flag，不能把ALBERT说低FPR。A.5把mean+1SD等同68th、mean+2SD等同95th不正确（高斯单侧CDF约84.1th/97.7th，68/95是双侧区间质量）；这不自动否定经验percentile算法，但理论解释应纠正。A.1.3把高分直接说“确认trigger”过强；词真值来自实验设计而非分数。没有AttDef对照/引用；Lyu2022已有检测方法，原文称没防御应限定为未做输入中和而不是没有检测。PURE引用重复两条；BookCorpus论文被用作rareword引文不匹配。均不以这些问题抹掉它的直接相关性。
- **可复用/未回答**：分数定义可公开复用为attention/gradient候选观测，正常样本校准、单词/子词显示与定量行为并排；[anindyasdas/XGRAAD](https://github.com/anindyasdas/XGRAAD)只核README，未运行，研究用途声明不视为已完成许可证核验。`candidate_gap`是效果—归因对齐与生成传播，不是新一轮输入异常检测。PURE模型purification作为外围方法候选，未以其未读细节支撑新颖性。

## Revitalizing Black-Box Interpretability — Proxy Explanation Framework

- **身份/版本/阅读**：Junhao Liu、Haonan Yu、Zhenyu Yan、Xin Zhang，[ACL2026正式版](https://aclanthology.org/2026.acl-long.220/)，pp4806–4844，39页；`Revitalizing2026.pdf/.txt`。正文、参考文献、附录A–G、Algorithm1及Tables6–25全部非截断阅读。PDF p6 Tables2–3、p9 Tables4–5、p16 Figs7–8已渲染核读。不是只读了9页正文后跳过30页附录。
- **测量/方法**：以便宜LLM上的LIME/KernelSHAP线性代理解释，近似昂贵目标LLM在删除扰动邻域上的行为。任务级筛选只在两模型原输入预测相同的集合D′取样，比较代理解释fidelity与oracle解释fidelity的0.9倍；每新增样本做配对t区间，最多50样本。实例级仅检查当前预测相同，不重新验证该实例邻域一致。分类fidelity为代理预测目标行为的accuracy；NQ短回答转sentence-embedding cosine分数，不是逐输出token的因果归因。
- **实验/图表**：12模型，包括GPT4o、DeepSeekV3、Qwen0.5–72B、Llama8/70B；LIME/SHAP各1000扰动，SST2210/MMLU1321/NQ200，另IMDb/FakeNews/WebQuestions；短回答最多20tokens。Table2按API口径计screen/fallback，local列将本地API成本计零。Table4代理压缩率41.0–70.1%，低于oracle相应49.2–75.5%等；允许保留90%原性能，不能称完全无损。Table5代理清理后SST/HellaSwag/PIQA为94.0/93.5/90.7%，oracle94.2/93.7/91.5%。附录G全研究约3000GPUh与$20000 API，是研究投入而非每次解释成本。
- **投毒边界与已做 `done`**：所谓poisoned example removal是**ICL提示里的有害示例**：不断加入直到准确率低于80%，再删除负归因示例。没有训练权重后门、隐藏trigger、OA或内部线路。已做输入/示例归因的跨模型代理、行为一致性筛选、归因引导压缩与污染提示清理；不能说项目首次让归因有行动用途，但也不能说本文已解释OA输出传播。
- **忠实性问题**：邻域预测一致与特征贡献正确不是同一判据；强多数标签邻域可让常量代理accuracy高，却不保证因果排序。总体平均达到oracle的90%也不保证任意被展示的词可靠，且oracle自身是近似解释。在线单点预测相同无法排除局部边界/梯度相反。归因符号依赖目标score：若解释当前错误预测，直接删负贡献示例未必改善正确率；§5.2未说明用真标签还是预测标签打分，也未完整列投毒构造/删除数量/未污染与污染前基线，不能据恢复准确率确认找全污染示例。
- **统计/报告疑点**：§3.1在每个n重复普通固定样本t置信区间、见显著即停，未给alpha-spending或time-uniform界；宣称整体95%保证缺乏该序贯程序所需论证，n=1时方差也未定义。表3是有限实验precision，不能代替保证。NQ被定义为连续回归，§4.2却统一用accuracy且未给离散化/容差；实例“预测相同”在短答案上是精确或语义相同也未清楚。附录B称所有任务收集首token logits，与NQ完整短回答cosine的工件描述不一致，需代码/数据核验。Fig7/8图注称右侧是filtered对照，但原图标题左右是LIME与KernelSHAP，不能用左右差判定筛选改善；Fig9/10才明确同方法filtered。90.5%成本摘要的跨任务聚合权重未充分说明，不能推广所有目标/所有开销；local零API不等于零计算成本。Table4所谓91.3%是压缩效果相对oracle，不是任务绝对准确率。
- **可复用/未回答**：可以借任务分布→实例展示的校准层、显式fallback与包括筛选的成本账本；[XLLM-Bench](https://github.com/outerform/XLLM-Bench)仅来源确认未运行。项目应把“来自另一个模型的解释”直接标明，并报告覆盖率与失败例，不让代理热图冒充受审模型计算。它未回答model-specific trigger差分、逐步历史携带效应、解释与内部干预的一致性。直接参考Focus-LIME/REX/Beyond Attribution已交generation初筛，通用加速/压缩文献不自动扩为后门机制近邻。

## BkdAttr争议链：Voyer 2026非同行评审复核报告

- **来源/阅读状态**：Geoffrey Voyer，[Why Attack Success Rate Gives a False Picture of Backdoor Removal](https://www.lesswrong.com/posts/2sT9ykYacCs4L8bx2/why-attack-success-rate-gives-a-false-picture-of-backdoor)，2026-02-24，作者一手复核博客；正文、附录、HTML所有可读表格已读，保存`texts/Voyer2026_WEB_EXTRACT.txt`。核心图9/10最初web cache miss，后通过作者公开原图普通curl获取并核读（`figures/Voyer2026_F9.png`、`Voyer2026_F10.png`），完成图文阅读，**不与同行评审论文等权**。两图确认exact ASR与行为拒答、整体回答质量曲线分离；Llama层20低ASR但高拒答，层11好回答更多。仓库未运行，原始Google表未独立复算。
- **作者报告**：在Qwen2.5-7B/Llama2-7B、固定拒答后门上重做BkdAttr的head归因干预，增加Good/Strange/Wrong/Refused四类输出。用256未参与SFT的Alpaca题；原论文评估集可能重叠的说法仅属复核作者报告，未独立代码核证。许多固定字符串消失后变成另一种拒答，并伴普通能力下降。其BASR定义1−干预后Good率/预期基线Good率，向量法Table3相对降低约27.5/46.4%，原精确字符串指标看起来75/95%；这是本案例重新评估，不是BkdAttr全部trigger/任务被否证。
- **本报告自身疑点**：BASR把任何Good下降都归为攻击效应，混合普通干预损害与trigger特异行为；如Good高于baseline可为负，不是天然[0,1]成功率。应同时做无trigger的同等干预对照，报告逐题配对及不确定性。所谓judge95.4% vs人96.1%来自280样本、24分歧后作者看Gemini理由重新裁决；原始一致率91.4%，不是独立盲审准确率。附录Qwen Table4/5多数行加和255，声明样本256；Table5零headGood188而Table6 SFTGood189。Llama32head表列Refused241/256而主文宣称约99%可能指固定拒答变体或另一计数，需原始记录核对，不能重复百分比当已核实。BASR表编号也与附录重号。
- **对项目的证据地位**：可作为需要验证的争议与评估警示，不能凭博客宣布BkdAttr机制结论错误。独立科学逻辑已足够要求：热图所指模块被干预后，既看目标序列变化，也看语义行为与正常能力；字符串ASR减少不充分证明特异机制被移除。核心图缺口已解决；此链仍保留原始数据未独立复算、评估集/判断独立性及表格计数不一致等open_issue。

## Patcher: Post-Hoc Patching of Backdoored Large Language Models

- **身份/版本/阅读**：Anjun Gao、Yueyang Quan、Yufei Xia、Zhuqing Liu、Minghong Fang，[USENIX Security2026正式会议页](https://www.usenix.org/conference/usenixsecurity26/presentation/gao-anjun)，pp221–240；正式PDF含封面21页，`Patcher_USENIX2026.pdf`。其引用的[arXiv2606.02995v2](https://arxiv.org/abs/2606.02995v2)全版24页，2026-06-11，`Patcher2026.pdf/.txt`。全版正文/参考文献/附录A–C/全部Tables1–28已读；正式版全部word-diff107块已读（`Patcher_VERSION_DIFF_PLAIN.txt`），差异主要页眉、引用编号、删除并引用全版证明/基线细节/两张跨模型表/一个案例。全版p4 Fig2、p19乱码案例Fig5、p22 Tables15–25均渲染核读。以下表号默认全版，正式版Table11起减2。
- **归因对象/方法**：一个被报告的输入x和实际回答r，固定受审模型，对teacher-forced完整回答的总NLL向每个输入embedding求梯度，取L2norm；K-means K=2把高簇当trigger候选。Fig2已有文本背景热图，颜色来自明确的response-conditioned gradient score。它聚合输出所有位置后给**输入词**打分，不是输出词各自受trigger影响的热图。然后做防御微调，并用正常与未触发有害样本的KL控制副作用。
- **实验**：Llama3.1-8B/Qwen2.5-7B/Falcon3-7B，SST2/CoLA/GSM8K/AGNews；五种后门均适配统一年份trigger，CBA复合例外；默认污染率50%，另测5–60%。报告一例之外还用150safe+150unsafe干净样本，不能简称“只需一例与参数”而漏掉辅助数据。ASR联合LlamaGuard与GPT4.1mini判断不安全且遵循题意，作者明确无人复核。Table2定位precision .81–.91，recall .89–.98；Table1大多数ASR0–.05，正常能力并非完全无损，例如Llama AGNews-POLISHED .79→.74，clean baseline .90。32B、三专业模型、长度/位置/噪声/多轮均有附录实测。Table25平均回答长133/329/714tokens，F1相近；没有逐位置梯度方向数据。
- **已做 `done`**：后门输入词的回答条件化梯度热图、单一失败报告触发定位、词级真值Precision/Recall/F1、不同输出长度与杂质报告敏感性、行为修复与正常能力并排。这是2026最直接近邻之一，不能只拿早期BERT注意力当竞争者，也不能把论文的单个输入热图当输出传播图。
- **因果逻辑审查**：原文“trigger必然具有更大梯度”“只有因果词使loss gradient最大”“各回答token梯度方向一致”的一般表述没有数学保证；饱和、冗余/组合触发、embedding尺度、序列梯度抵消均可能使局部梯度小而离散移除效应大。NLL总梯度范数不等于逐步梯度范数之和，长回答F1稳定不证明方向一致。K=2在无trigger时也会划高分簇；方法修补普通失败有效，不能据此判明失败来自训练后门。没有absent/sham/clean模型完整因果对照，也未在OA测试。
- **验证/理论边界**：§5及附录B以已达到的经验refusal loss/KL为假设，Jensen/Pinsker推的是训练分布与teacher-forcing前缀上的平均概率/TV界，既不保证实际优化会成功，也不保证自由生成全轨迹、分布外普通能力或梯度定位正确。KL约束能控制观测输出变化，不自动证明内部表示/线路保存。Table8让Mudjacking用已知trigger与定位trigger的比较说明定位不是全部，但不同方法数据、目标、优化预算适配未完整公开在正文。Table22把训练来源取证、RAG文档定位、长上下文归因改为输入token定位比较，转换粒度与所需额外数据不明，不能据其F1宣称原方法无效；相关原文交generation核读。
- **其他疑点/边界**：五种攻击共享一个明显年份trigger，attack名称多样性不等于trigger机制多样性；adaptive条件主要位置/格式/语义/复合，未系统评价梯度忠实性压力，不能称普适。hybrid已假定同一报告含两trigger并用两轮，不等价只见一个trigger却找出另一隐蔽trigger。Table15名为localization comparison但实际列修补ASR/ACC，未列定位真值指标；复数高F1测试未报告seed/CI/报告采样数的完整分布。§7触发后的domain错误不总适合拒答，作者亦承认binary harmful边界。Fig5是示例输出，未执行不能称代码功能已验证。
- **可复用/未回答**：可借完整回答NLL这个可审计score、固定模型与回答条件、输入词热图和长度分组真值检验；[官方artifact](https://zenodo.org/records/20362596)仅来源核对、未运行。尚未回答同prefix下trigger差分与历史中介、输出受影响词的定义、OA观察器失效与真实内部机制改变的区别。研究贡献可在这些认知问题，不需要再造一个触发定位修复器。

## Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Unseen Activation Monitors

- **身份/全文范围**：Max McGuinness、Alex Serrano、Luke Bailey、Scott Emmons，[arXiv2512.11949v1](https://arxiv.org/abs/2512.11949v1)，2025-12-12，30页，全文正文、参考文献、附录A–F读完。官网标题有Unseen，PDF标题没有此词，记录而不另算论文。PDF文本末四页图片未被提取，已逐页渲染并阅读p27–30所有示例；另核p4 Fig2、p8 Figs3–5、p9 Figs6–7、p25 Fig16、p26 Fig17，文件`figures/neural/`。
- **任务与对象**：受控model organism，训练只使用11种非安全概念，测试安全概念和事后在最终冻结模型、无trigger数据上训练的未知探针。主实验Gemma2-9B、另Gemma2-2B/Llama3.1-8B/Qwen2.5-7B都用abliterated variants；附录B标准安全Gemma9B另做跨语言实验。探针是逐生成token的linear/MLP均分或attention pooling，输出标量概念可解码性。测试2000样本、正负各1000；trigger仅加正例，负例定1%FPR阈值，不是触发正负全部重估部署FPR。
- **方法与主要图**：已知概念训练而泛化未知概念；三seed/95%bootstrap。Fig2训练后无trigger与有trigger，安全探针TPR从55/88/90/75降到33/0/16/0。Fig3四八层ensemble及MLP/attention有任务显著缓解甚至完全保留识别，不能称所有monitor失败。Fig16 base model trigger对照未见下降，RepE deception 20→58反而上升。Fig4 probe×trigger热图、Fig5分层Δh范数、Fig6正常/触发分别PCA的累积解释方差、Fig7 Δh向线性probe方向投影、Figs8/17 embedding语义距离与probe分数散点。AppB8训练语言/10测试语言迁移；AppD15固定输出的log-likelihood维持与基准能力。
- **最直接可视化先例 `done`**：**附录F pp27–30已经对LLM输出逐token做红蓝背景热图，左右显示无trigger/有trigger，各词颜色来自概念probe score并显示序列均分**。大部分配对输出相同，p29第一例输出不同；文本举例没有建立所有图都teacher-forcing同前缀的明确统一协议。颜色含义是该位置状态的概念可解码性，不是该词概率受trigger提升多少。不能把“OA类模型的输出文字温度标记”“同输出不同内部热图”本身作为未做的新贡献。
- **机制结论边界**：Fig6分别对两组生成激活做PCA，11 vs39主成分达到90%方差、35 vs88达到95%，证明该样本集方差更集中，未通过投影/patching证明固定低维子空间为必要充分机制；生成文本/位置对齐协议不够明确。Fig7线性probe方向的Δh负投影在probe层意味着logit降低（线性代数关系），不是独立的因果发现；跨层用同一probe向量也需警惕表示坐标不同。未定位具体head/MLP，作者§5.1承认这一点。语义相关性Fig17 r=-.300/-.452称strong、其中目标词deception相似度1却非最强效应，不能“confirm semantic not keyword”作排他因果结论。
- **选择性与能力疑点**：Fig4并非只有对角显著：jokey trigger对comforting probe下降98%、literature trigger对math下降92%、Finnish/German交叉88/73，作者只重点讨论跨语言相关性；强选择性需同时报告off-diagonal损害及语义重叠。Fig4色标写TPR@1%FPR reduction但图注说probe score下降，不能混用。所谓negligible capability tax没有预设等效界，Gemma主图MMLU71.3→68.2→67.0、Qwen GSM89.3→83.0→80.0、SocialIQA主图57→52→47，部分损失并不小；附录B Fig13以微调无trigger为基线，不含微调本身损害。正文说跨语言所有test baseline>90%，Fig12最后语言约83%，需以图为准。辅助模型打分数据选择与benchmark上下文/自然输出差异是外推限制。
- **可复用与未回答**：复用无trigger/trigger/base三条件、未知probe定义、固定FPR、输出热图公共色标、观测与投影解释并置；认知项目可比较同prefix输出概率差、概念probe变动、内部干预效应是否一致。尚未证明自然训练自发隐藏、真实内在欺骗思想、任意未知monitor不可见，更没证明梯度解释必定失真。PCA图与probe输出不能直接当trigger对每个输出词的因果归因。
- **引文扩展**：OA与RL-Obfuscation已纳入；表征工程、probe监测、metacognitive monitoring/control是相关来源但并非自动全部MI全文队列。当前直接新OA论文仍待读，本域未达closure。

## RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?

- **版本/全文**：Rohan Gupta、Erik Jenner，[arXiv2506.14261v4](https://arxiv.org/abs/2506.14261v4)，2026-02-26，24页；正文、参考文献、附录A–N全部读完。核心p6 Figs3–4、p8 Fig5、p24 Figs15–16已渲染核读于`figures/rlobf/`。
- **任务/方法边界**：对已经移除安全训练的Llama2-7B、Llama3-8B、Qwen2.5-14B，用输出行为评分与激活probe评分形成受控压力测试，研究观察器鲁棒性；一类全局条件、一类有/无特定trigger的条件性观测改变。作者明确不是自然RL风险发生概率，且不经过monitor梯度不等于无训练权重访问。主任务harmfulness，生成token状态，不研究输入token监测。训练probe、测试probe、model organism与原始安全模型需分别记名，正常基线不能混用。
- **实验证据**：同probe体系linear/MLP/attention与mean/median/max聚合、static/retrained、模型规模、5seed小规模敏感性、语言质量/MTBench、输入分布、层与ensemble均实测。Table1 linear-mean evasion .75/.82/.594同时StrongREJECT2.8/3.1/3.85；max或attention有的evade率高但回答塌陷至1分，不能算隐藏成功。Fig4明确同看evasion和quality；Table4具体组合补全。参数搜索和附录负结果完整保留，未提供训练全部seed的统一置信区间。多轮/many-shot未泛化，第三模型迁移仅早层局部有效。
- **关键认知先例 `done`**：**§5 Fig5a把RL模型生成的同一completion分别teacher-force给base和tuned**，比较原输出→新输出在base上的probe变化与固定新输出下换模型的probe变化，已明确分开“文本行为变了”和“同样文字的表示变了”；Fig5b词类型频率变化×同词probe-logit差散点，Fig1共享PCA。AppM Fig15a在base上用tuned输出重新训练probe再比较，仍有表示效应；Fig15b把tuned层attention/MLP输出patch到base，读取mean probe score，早层与probe附近MLP影响较大。不能声称OA邻近工作从未固定文字作机制分析、从未做patching。
- **因果审查**：Fig5的差分是有明确顺序的三条件分解，70%是本实验probe recall下降份额，非对称无关的普遍中介比例；缺少反顺序第四格时模型×文本交互分配依赖选路。固定完整输出已控制历史，但对probe分数的效应不等于对输出token概率/正常功能的效应。Fig15跨模型patch有坐标漂移与混合网络问题，缺反向/随机/正常行为控制，且图是mean probe score正文却称recall，不能就此证明特异行为机制必要/充分。
- **边界与疑点**：逐token独立训练给同一序列全部位置相同标签，并当IID，可能学格式相关性；平均聚合被格式/长度影响而max/attention更稳是本实验结论，不是普适鲁棒证明。作者承认优化失败不证明防御根本安全。StrongREJECT≥2.5主图经验门槛与附录F<3失败门槛不一致；§5一句“tuned更高”与base−tuned正差及图注明显相反，应按图读。§G首句Llama2-7B而表5/随后Llama3-8B；Fig8a caption λwb=.1而图例=.8，保留可复核问题。模型替换与输入攻击不是同一威胁模型；高“正常能力”以SFT后模型为基准，原始模型→SFT的主要损失仍须算。§K已有多轮不泛化，却由反向分布测试声称only base model而不依赖data distribution，外推过强。
- **可复用/未回答**：同完成文本跨模型teacher-forcing、输出质量与监测分数并排、aggregation压力、触发/不触发、低FPR及失败条件很适合认知可视化项目。尚未提供trigger对每个输出词的概率效应图、输入直达与历史中介的逐步分解、OA与其混淆的归因忠实性系统验证。它的probe解释失效也不能自动推出梯度/干预解释失效。
- **扩展**：OA、Neural Chameleons已全文；Atlas待全文；引用Marks等Auditing Language Models for Hidden Objectives可由root的MI范围覆盖确认，不能把任意probe/representation工程无限纳为直邻。

## The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes

- **身份/范围**：Mohammad Taufeeque、Stefan Heimersheim、Adam Gleave、Chris Cundy，[arXiv2602.15515v2](https://arxiv.org/abs/2602.15515v2)，2026-05-26，30页，正文、参考文献、附录A–V全部读完。PDF刊头写ICML2026/PMLR306，官方arXiv注明oral；`site:proceedings.mlr.press "The Obfuscation Atlas"`未命中，未声称已完成另一个publisher PDF版本比对。p6 Fig2、p8 Figs3–4、p13 Tables5–6、p23 Table15、p30 Figs13–14/Table23渲染核验，`figures/atlas/`。
- **任务与方法**：MBPP-Honeypot874问题、400heldout；代码执行一条可见测试做训练反馈，其余通常两条做额外检验，并有短代码偏好。四模型Llama3-8/70B、Gemma3-12/27B，通常每配置3seed，若干额外10seed。三probe数据Simple/Diverse deception与同域synthetic MBPP。核心不是生成词热图：对完整回答附加固定follow-up，读取最终Yes状态，训练各层linear probe并筛training CE<.6层，取sigmoid均值。生成mean pooling、lasttoken等作附录对照。
- **可视化与已做 `done`**：Eq3将固定生成文本在原始/最终模型的probe分数拆开，Fig2/6二维坐标和配置分布图区分honest、blatant deception、obfuscated policy、obfuscated activations；四分类是可复用的诊断语言。用固定完成文本跨模型，不能再声称从未对OA类现象区分“文字改变/内部表示改变”。原模型probe也低的deceptive输出归obfuscated policy；原模型高但最终低才归activation。两者可混合，作者优先policy以处理重叠，因此图的互斥标签不等于真实机制互斥。
- **核心结果**：无detector的普通RL也使off-domain probe泛化下降，Table4重新训练probe的平均AUC Simple .991→.376、Diverse .996→.655，而MBPP仍约1；一般格式probe相对保留。Fig3对20自拟硬编码态度问题的undesired答案变化支持行为层面的态度转变假说。Fig4加detector与不加的固定synthetic表示差相近。较高KL与有效probe配置获得更多通过隐藏测试的策略，Fig5/12只证明此coding任务性能；未证明通用benchmark能力不变。AppA/V无短码偏好也可发生reward hacking、加入probe有效；AppS不同pooling仍出现混淆；AppT两模型500step延长；AppO食品偏好域用LLM biased reward，非可验证reward本身。
- **理论的正确窄结论**：AppD推导stop-gradient reward下REINFORCE不含直接反传probe的导数项。这不能升级为“所有policy-gradient过程不可能因monitor奖励形成表示规避”：参数更新仍改变表示，奖励决定sample选择/更新方向，作者附录也承认相关改变。论文正文“incentivized OA require direct gradients”比证明的“没有该直接导数路径”更强。GRPO段用总体μ/σ后类比有限batch；实际共享样本估计baseline/σ造成相关性，跨prompt缩放亦可改总梯度方向，故“unbiased且只改尺度”的一般说法需额外条件；有限GRPO/PPO clipping并不自动等于该简化推导。
- **认知/因果限制**：20问自述变化不是内在信念因果证据，部分问题把合理边界特例（edge case/lookup table）混入作弊；20近义问SE不等于独立任务总体误差。两个简单非deception格式probe仍能分类不证明仅deception子空间变动。单个固定follow-up是另一次条件化计算，不等价原生成过程已存在同一信息；在该位置解码到低概念分数，不说明哪些原输出词受trigger影响。此研究没有特定trigger植入；不要把它当传统训练后门。仅一条可见与两条隐藏测试也不是完整程序正确性。
- **实质可复核疑点**：Tables5–6标Pass-First-Test却多处小于Pass-All-Tests（如Gemma12B34.2% vs51.3%、Llama8B10% vs48.2%）；按正文包含关系不可能。很可能前者实际是only-first的deception率而非包含all的通过率，需原始指标确认后方可照搬“deception gap”定义。Table5两列模型N=10+10但aggregate22，未列其他2runs；Table6多处aggregate亦含未展示配置。Table15 W1达到8.395，与定义ψ∈[0,1]下W1≤1不符，可能报告logit但未注明。AppH公开多GPU实现bug使27B/70B只主rank梯度正确、有效batch变小；作者称修复后相似而未在正文提供完整配对数表，保留。AppN minimum gap .33却称far above0.2–0.5所有cutoff不准确；表分布小幅变动说明经验稳定而非数学不变。Food OA段文字说faithful日本内容却内部应flag，与其总体定义方向有歧义；表19无OA实例，不能借此证明全四类都由标准RL复现。
- **可复用/未回答**：最有价值的是让外部行为、原模型观察器、最终模型观察器三者并排，区分表示漂移、文本策略与真正行为纠正。尚未有完整trigger/sham×模型对照的输出词概率/历史中介图，未验证这类图在观察器失效时仍忠实。对项目价值是认知分类和否证框架，不要求做更强detector或更隐蔽优化。

## RAGOrigin: Who Taught the Lie?

- **身份/全文**：[arXiv2509.13772v2](https://arxiv.org/abs/2509.13772v2)，2025-10-18，20页；正文、参考文献、附录A全部扩展实验/Algorithm1/Tables1–38，以及附录B SP2026 meta-review均已完整阅读。p5 Fig2、p11 Table12、p12 Tables14–15、p17 Figs3–4已渲染核验。正式会议身份有原文meta-review支持；独立publisher最终PDF尚未比对。
- **对象/方法**：已知用户报告的q及错误回答r，追查RAG知识库里的污染**文档**。先依检索相似度排序，逐个互不重叠的K文档块重新生成；用外部LLM判是否复现r来定候选范围。每份文档u的ES为检索cosine，SC为代理LLM单文档条件下query平均logprob，GC为代理LLM在u+q条件下对观察到的r做teacher-forced平均logprob。三分数在候选内z标准化后取均值，K-means两簇，高簇判污染。GC是绝对条件似然，**没有移除u的差分，不是原victim模型、不是完整原上下文，也没有逐输出词展示**。
- **实验/结果**：默认GPT4o-mini、E5-base-v2、topK5，每个问题5污染文档，每数据集/攻击取100个成功事件；5问答集×9攻击，扩展多跳、对抗条件、5–50变化污染量、16.7m库、ELI5、不同retriever/victim/proxy/judge、advanced RAG。Table3 NQ检测准确率.98–1，Table4删除后攻击率多在0–.01。Table12单独ES已达.90–1，GC .89–1，合并更稳；Table14多跳GC下降仍由ES/SC保持辨识。Figs3–4为K5–25、M1–9的文档分类曲线，非生成机制图。Table15 ContextCite/AttriBoT被改作污染文档分类，DACC分别<.77/<.70；缺扰动预算、阈值转换/代理适配细节，不能据此证明它们的因果归因较不忠实。所有正文附录表已读，未执行代码复现。
- **已做 `done`**：将特定错误回答的生成似然与检索相似度结合成污染责任分数、事故驱动文档清理、多跳与范围消融。它占据“观察错误回答→量化文档责任”邻近空间；不占据OA权重后门、输出词触发效应或内部线路可视化。
- **概念/因果边界**：相关文档使错误答案高似然不等于造成该答案；代理已知答案、普通拒答、高相关但良性文档均可能得高分。单文档评分不识别协同/冗余。污染成员标签是检测ground truth，不是给定事件的实际因果贡献；原始topK外备用污染未参与当次计算，后来纳入可以有防御价值，却不能说它导致已经发生的输出。多跳中GC下降而ES/SC成功恰好说明检测准确与生成影响识别可分离。无RAG也答错的97%排除实验无法排除模型知识与污染同时构成充分原因。
- **范围算法疑点**：Eq4与Fig2确实写累计Match数=i/2；文字一处说至少一半匹配，后一处说一半不匹配，等号不等于前者。初始无法复现、跨块协同、较后仍有污染的情况没有完备保证，Algorithm1也无明确库耗尽/最大次数边界。不能从实验高召回升级“保证候选完整”。两簇分类在没有污染时仍分高簇，缺校准拒绝选项。
- **评估/成本限制**：成功事件条件化、诚实报告假设、缺完整正常删除后效用/语义答案变体/配对不确定性，均限制部署外推。meta-review本身提醒false-flag与并行资源依赖。Table6时延1.31–2.17s基于8张A800+64核；正文max2.11与表2.17不一致；API≤$0.0004没包含本地GPU/索引。不能称零开销或天然可扩展。单例稀有词解释SELF-RAG失败没有消融验证。
- **可复用/未回答**：固定观察回答、显式区分proxy/victim、检索/生成两类分数并排可复用，但GC必须标为条件支持度。项目需要另外验证同prefix的trigger差分与内部干预，不把文档清除有效当作输出token因果图正确。范围扩大可用于风险清理，事件归因应单独保留真实检索日志。

## TRACE: Tracing Target Answers in Poisoned Retrieval Corpora via Token Influence Attribution

- **版本/全文范围**：[arXiv2606.25721v2](https://arxiv.org/abs/2606.25721v2)，2026-08-26，19页，官方摘要注明EMNLP2026 Industry接收；正文、长参考文献、附录A–E全部读完。p4–5 Fig2–7、p7 Tables2–6、p14 Table7–8/Algorithm2、p15–19所有扩展柱图已渲染查看。部分附录原PDF左侧图边界/图注被截，正文核心数值与附录表可核；Fig24图注TPR而轴FPR，按轴读。
- **归因定义**：Phase1对单篇检索文档的input embedding求预设肯定句logprob梯度L2范数，剔除标点/常见词/问题中词，取top词、拼连续片段并按文档出现次数筛。Phase2把候选短语当目标，重新求input gradient，检查候选词仍被高分且跨文档复现。Fig2已有检索输入文字高亮。目标来自固定肯定句或候选答案，**不是模型实际自由输出的逐位置trigger效应**。Algorithm1/2条件只写文档，不写q；q明确用于词剔除，若实现实际拼q与模板，需代码另核。
- **实验结果**：仅PoisonedRAG型协同重复污染，3QA集、6个4–8B左右模型、Contriever dot-product、默认检索5且5份污染，3次均值未列CI。主文高TPR与低FPR图用了不同阈值，不可拼成同一operating point。统一balanced时Table4三模型均值TPR88.13/FPR13.16/目标ACC80.97%。目标ACC仅要求T中任意token出现在目标答案，非完整答案重建、非逐词Precision/F1。与RAGForensics Table1只比TPR/时间，没有同FPR比较；Qwen时延约9s，其他1s，硬件未充分说明。
- **关键负结果**：Table3真实良性共识17题FPR39.22%，普通257题13.16；31.37%的hard-negative输出误报gold答案。Table5不剔标点FPR73.14，说明局部梯度与格式词耦合。Table7扩大检索5→20时FPR13.16→84.83、延时3.76→50s。Table8仅1/5污染时默认TPR9.98%、目标ACC.38%；仅3/5也TPR23.01。原文“少污染必降低攻击效果”的解释未同时报告下游ASR，不能拿它自动抵消检测失败。
- **已做 `done`/认知贡献**：用token梯度在生成前找反复出现的答案相关输入词，并用第二目标重读验证；提供良性共识hard negative和granularity/calibration失败。它直接表明高亮“共同驱动答案”的词不等于发现恶意。没有OA或训练权重后门，不是未知触发器一般检测。
- **逻辑与复现实质疑点**：Phase2仍用相同梯度方法，换目标是自一致检验，不是移除/替换干预；高局部敏感度不保证有限离散改变导致原输出改变，常规复制机制也可满足。主文“nonempty T说明undeniable malicious”被自身良性误报与limitation否定。Algorithm1 line20统计e在原始文档出现，而正文声称在高归因候选集合跨文档复现，两者不同；Phase2 line13还要求di属于p，正文描述遗漏。未做dropout-grad/长度/反事实效应校准。正文假设semantic anchors只是作者假说，未验证。对ground-truth目标的token重叠可由共享子词误中，无长度惩罚。
- **可复用/未回答**：可借文本高亮与多文档复现图、真/假内容共享同一影响模式的对照。项目应对颜色附明确target、prefix和score，并把恶意标签与影响分数分开。没有回答内部机制、实际逐输出token效应、OA下归因忠实性。引用Token Highlighter已是项目其他域直接邻居；RAGPart/Mask普通防御与其他攻击不无限扩全文。

## Needle-in-RAG / RAGCharacter

- **身份/全文范围**：Huining Cui、Wei Liu，[arXiv2605.01782v1](https://arxiv.org/abs/2605.01782v1)，2026-05-03，36页，under review预印本。正文、参考文献、附录A全部Tables8–21、附录B Figs7–10全读；p10 Fig2、p17–19 Figs3–5、p21 Fig6、p34–36 Figs7–10渲染核读。
- **研究问题/方法**：针对已经发生的RAG错误，用日志固定当时真正送入生成器的证据、顺序与prompt offsets，黑盒mask/replay局部字符片段。候选先用错误回答字符串定位，再按chunk→sentence→phrase→character分层预算搜索，最多三轮；每次重生成的输出成为下一轮事件锚。区分deterministic attribution span（πattr，答案精确匹配/大写词组/首词回退，用于报告与重叠评价）和counterfactual causal span（πcausal，用于遮罩及缩小）。该区分已经是直接认知先例，不能把“固定trace再做字符干预”说成项目首次。
- **实验**：2QA库、5攻击族、6模型；Table4以Unknown/Confusion/EVT分输出，GPT5mini仅约2.8–10.4% EVT，后续归因结果主要条件于词面可追踪事件，不能外推其余约90%以上输入。Tables5–6六模型平均CharF1 NQ .860–.978、MS .694–.955；Table7字符基线只平均4开源模型，不能把前后表视同样本。后附GPT5mini GASLITE-MS CharF1 .321/FPR .701是实质失败。Figures3–4只变K，Figures5–10重画相同表的散点/斜率；没有系统组件消融、反事实成功率/效用恢复/调用成本或时间实测表。
- **核心测量问题**：Eq3所谓CharFPR为错选字符/所有选中字符，实际是1−precision（false discovery proportion），不是FP/(FP+TN)；低值不等于低事件误报。CharIoU/F1比较**πattr**与注入span标签，不能验证**πcausal**或实际生成机制。πattr主要字符串启发式而πcausal单独搜索，若可视化报告前者，重叠高不能归因于反事实搜索的贡献。论文未给两者一致性、disagreement案例比例、仅答案匹配基线或分阶段消融。
- **因果定义边界**：固定检索日志确实避免重检索混淆，优于把未参与当次生成的文档算因。但有限mask replay只给这项干预在此prompt/解码下的行为效应；mask可能破坏语法/改变tokenization，未充分写替换串、同长度无关字符sham、解码随机性重复。片段移除后目标消失不代表注入来源/必要充分机制；对冗余、协同、抑制交互也不完备。预算bisection与early stop无法保证全局或subset-minimal，Algorithm2甚至在没有成功候选时也取最高分并记录πcausal；应保留未验证标签。
- **label-free/event疑点**：Algorithm1仅“输出与任何prompt chunk重合”便触发，正文又写“incorrect answer string / injected claim”并引用PAR/T_ASR/CONFUSION已有信号，何处知道incorrect/poison身份、如何与正常引用区分未闭合。没有良性可引用事实的false alarm对照。Event suppression可能只是改为不重合的同义错误，下一轮锚点也改变了被解释事件；没有独立语义judging/真值标注流程说明。无prior trigger不等于无答案结构先验。
- **实证/复现缺口**：未清楚报告各设置实际归因样本数、seed/CI、模型版本/解码、默认retriever与attack预算/字符真值构造。引用整个数据集大小不是测试n。表中RobustRAG四开源模型每攻击数值完全一样，需要确认是共享信号还是实现适配；Gemma表7B而图注Gemma2，版本不清。部分基线借用本方法Pass1，再外接judge抽取，公平预算与fallback比例未给。作者称其他方法chunk-first不足而自己也chunk-first，真正差异需要消融支持，不由比较表自动证明。
- **已做 `done`/可复用**：精确prompt使用记录、字符offset、forensics与remediation分离、归因片段与干预片段双工件、词面重叠和反事实行为分开，适合本项目借鉴。输入字符原因与输出token后果仍是不同坐标；未做OA、参数后门、内部路径或每一步prefix因果影响。项目可以研究两类图的分歧及验证，而不能仅换成文字热图主张新颖。
- **邻居处理**：RAGOrigin全文、RAGForensics语义支持范围已核；Context Influence ACL2025 long1220交generation官方scope筛；其余一般RAG攻击/检测器仅范围清单，不因关键词扩全部全文。

## Discovering Backdoor Triggers — 一手研究报告

- **身份/全文**：Andrew Qin、Tim Hua、Samuel Marks、Arthur Conmy、Neel Nanda，[作者报告](https://www.alignmentforum.org/posts/kmNqsbgKWJHGqhj4g/discovering-backdoor-triggers)，2025-08-19、9月1日更新，非同行评审论文。完整正文、方法、实验、讨论、参考链共271网页行已读，`texts/Discovering2025_WEB.txt`；唯一核心图经公开原图正常下载核读，`figures/Discovering2025.webp`（实际PNG）。图是学习到SAE向量激活的对数计数直方图，多数为0附近、少数较大；不是输入/输出词热图或跨样本成功率。
- **研究问题/方法**：已知后门目标行为，未知语义触发条件。以良性提示及指定目标完成的损失梯度在早层SAE方向作归因，借自动解释的概念标签提出触发语义；另一方法用内部控制方向并检查对应SAE特征。这里是目标条件下的内部特征发现，不是解释模型已自然生成的每个词，也不是完全未知目标的无监督后门扫描。本文所有操作仅科学阅读，未执行。
- **实验/已做 `done`**：Llama3.1-8B/Dolphin3.0，三个玩具语义触发（水果及两个合取条件）；只微调后半层以保留早层SAE有效性，正常/触发各100样本，目标行为比例分别1/91、1/96、0/91%，这些是模型构造检查，不是触发重建成功率。梯度法在水果及逐步加入已找条件后找到合取语义。受监督SAE控制在一个不知触发的例子成功，但作者手动改层、超参数及组合概念；完全无监督变体连最简单例子失败而停止。不能据此称大规模盲法成功。
- **关键负结果**：在更真实Banana错误事实后门与未后门的Moon错误事实行为上，能通过内部干预诱发行为，却不能从高排名SAE标签辨认对应真实语义；四个方向、14个改写等检查未给出可靠解释。Banana全层变化可能让原SAE失配，但未后门Moon也失败，不能归咎于唯一因素。目标完成本身可能泄露水果语义，作者用部分完成补查一个案例，未普遍消除此问题。
- **局限/对项目**：SAE词标签、稀疏语义假设、手动选择、已知目标共同限制解释。`reuse`是“改变行为成功/辨认语义失败”并置、在已知真值玩具与较真实场景分别验证。`candidate_gap`是同prefix实际输出token效应、内部特征必要性/充分性区分、OA下观察器失效；该报告说明有效控制不自动构成正确原因说明，不能把SAE高亮当原执行机制证明。
- **参考链**：Lee/Breck/Arditi《Finding Features Causally Upstream of Refusal》作者报告与MELBO是方法前驱；一般拒答控制暂按外围scope处理，未借其未读全文结果作done。LOCA因明确local token causal解释另纳入，不能用本报告替代LOCA。

## LOCA — Minimal, Local, Causal Explanations for Jailbreak Success in Large Language Models

- **身份/阅读范围**：Shubham Kumar、Narendra Ahuja，[arXiv2605.00123v3](https://arxiv.org/abs/2605.00123)，2026-08-07，正文刊头和官方摘要均COLM2026。21页，正文§1–5、参考文献、附录A–L全部1551提取行逐段读完；Fig1–12全核（p4/6/7/8/9/15/16/17/18），图保存在`figures/loca/`；PDF/text为`LOCA2026`。未单独对比COLM出版社终版或运行官方代码。
- **问题/方法**：对原始拒答提示与成功越狱提示的特定pair，寻找少量输入位置×SAE方向的补丁恢复原始响应。§3/Eq2–3用参考/目标**首输出token分布的KL**的一阶近似排名，每次补丁后重算；§3.2把原instruction embeddings重复/跳过匹配不同长度，post-instruction chat模板逐位对应；AppH在选中方向张成空间中作投影替换，多方向用QR。Fig2的彩色格是输入位置×内部方向的候选干预分数，绝非逐输出token因果热图。
- **实验设定**：WhatFeatures 10800请求/35方法，greedy512tokens，HarmBench判正常拒答且越狱成功的pair。Llama3.1-8B、Gemma2-2B、Gemma3-27B、Qwen3-8B；预训练SAE来源不同。70/10/20随机split，前两仅Lee baseline用于拒答方向。每层最多20补丁；Fig3/8主要报告首token KL/原token logit差AUC、首次argmax匹配的MP和最终首token匹配RR。Llama早层平均约6–8补丁、Gemma3中层约9、Gemma2约12、Qwen约10；不能把抽象“平均6个改变”推广全部模型或完整回答。
- **已做 `done`**：Fig4在Gemma2展示token-specific及iterative两因素有效；Fig5/9选中补丁随层由instruction向chat后缀/标点集中；AppE Fig10随机匹配、embedding相似度与原匹配的行为指标近似。AppF Fig11确有完整生成HarmBench检查（Llama随机50例，**固定20补丁之后**），LOCA的违规回答率低于适配基线。AppG 200例×20补丁×7层实际KL与一阶分数比较，初层误差大，后层更准。因此不能说本文完全没检查生成行为；也不能说它验证了首token MP停止时的完整拒答。
- **解释与干预的边界**：局部少量干预足以改变输出，支持受约束的反事实可控性；不是自然执行因果路径完整恢复，亦非数学最小解释（贪心且上限20，失败也记MP20，未证明全局或删除意义最小）。用另一自然样本局部投影替换，不保证整个混合激活仍在分布内；正文“within-distribution”应弱读。AppE随机匹配不损行为只说明该目标下匹配可替代，不能证明被匹配语义和个别token的解释稳定。Fig5后缀位置可携带上游上下文，不等于标点语义是原始原因；图中两个分类相关，不能当两项独立机制验证。
- **案例与负证据**：§4.4/AppL仅一个细查案例，Neuronpedia最大激活解释若干概念，早层最后一步方向含义与期待相反，作者承认解释可能错。将通用文本特征变化解释为模型“认为编造无害”，缺专门语义干预对照。AppB原拒答/违规响应首词本来相同Llama10%、Gemma2约5.5%；AppK有开头拒答而后续违规等失败，HarmBench也有误判。二者共享首词并不逻辑蕴含完整首token分布KL很小，作者该处推断过强。首词归一化统计与实际subword argmax代理还非完全同一量。
- **报告疑点**：Fig3图注50例，§4.1称Llama225合格/Gemma3 438合格且脚注实际140，最终图n矛盾；图均值±1SD不是CI。§4.1称Gemma2第38层显然应核Gemma3，图与模型深度不符，不能照抄。Fig12正文Pearson .809，图.817，未见解释。AppI称均值但Eq5/6写sum，且baseline目标不同，实际适配公平性不能仅凭自称判定。无一般良性utility保留实验，故不是已证部署防御。
- **复用/未答**：`reuse`包括分离代理指标与整回答验证、每层局部补丁可视化、首词一致失败集、随机匹配敏感性和估计分数/实际干预效应图。`candidate_gap`仍是实际输出全程不同词何时受触发影响、共享prefix/历史中介、OA条件下视觉解释忠实性；单纯做局部token干预或概念热图已有直接先例。代码地址为论文官方GitHub，未执行或许可证逐项核验。参考链WhatFeatures、Yeo、Ball另作跨域范围去重，不以未读结果断言已做。

## Out of Context Obfuscation — 全文获取缺口

- **可核身份边界**：[官方OpenReview PjOwhIzR2z](https://openreview.net/forum?id=PjOwhIzR2z)，标题《Out of Context Obfuscation: What Facts Matter for Probe Evasion?》。检索返回作者Fadi Benzaima、Faraz Ahmed、William Soylemez、Adrians Skapars、Rohan Gupta及2026作者发布链；官方forum/PDF/静态PDF均浏览器验证或403。静态公开入口`https://openreview.net/pdf/729837dfce7b9aa832920e939c96ec8d228f9818.pdf`正常curl也403，未绕过。会议/工作坊身份不可等同主会同行评审。
- **状态**：`fulltext_unavailable`。有直接monitor语义知识/解释失效邻接可能，不能以标题判断它已做何种机制实证，不能用于done/reuse/gap结论。保留必要访问缺口，影响“全部直接近邻已全文读完”的声明；下一步是获得作者合法公开版本或正常访问恢复。

## Refusal-SAE — Understanding Refusal in Language Models with Sparse Autoencoders

- **身份/全文**：Wei Jie Yeo、Nirmalendu Prakash等，[Findings EMNLP2025正式论文](https://aclanthology.org/2025.findings-emnlp.338/)，pp6377–6399，共23p。正文§1–7、全部参考文献、A.1–A.9、B及Fig1–16/Table1–12均已读并核原图，`pdfs/RefusalSAE2025.pdf`、`texts/RefusalSAE2025.txt`、`figures/refusal_sae/`。非只依据LOCA引用，未执行代码。
- **方法/测量**：CosSim+AP先用拒答方向选每层10个SAE候选，再以IG近似的间接效应序列平均选20个特征。可以每样本local set也可全数据global set。AP本身目标仍是两个首输出token概率差；AppA.1使用同一提示的全层steered状态作为corrupt而非自然对照样本，规避长度匹配。随后特征在输入及生成**所有位置**干预，用HarmBench评完整输出。故它确已做从首token归因到多token行为验证，不能说所有近邻只评首词。
- **实验/已做 `done`**：Gemma2-2B/Llama3.1-8B及base SAEs；greedy256，三harmful基准各100、CATQA7类、WildJailbreak及AdvSuffixes。Fig2/9验证选中集合可改变行为；Table6/7还评1000条Alpaca/Pile的CE及GSM8K/ARC，不是完全没有utility。Fig3公共/类别特定特征比较，Table1干预类别特征引起公共拒答特征和P(I)下降；Fig4逐步加入后缀词看拒答特征抑制与额外上游干预，Fig15/16列增量最大的输入词。AppA.9有base/chat SAE在Gemma的重建比较。
- **核心可视化边界**：Fig1/12–14输入词高亮→概念框→拒答特征→P(I)，属于本项目直接先例；但AppB明确实际干预作用于所有位置，高亮特意取**排除chat token后**最大IE位置，并未只干预高亮词。它要标首次出现的意图与选max规则也不完全相同。图上有亮词与下游变化不证明这些亮词单独承载被干预因果量。Fig10/11绿色是最大激活样本，而非模型输出词的触发差分。Fig4画后缀增量对内部FR的影响，不是后续输出token的因果传播。
- **因果限制**：FR定义为七类集合交集，FH为其余；共享/特定性不等于纯拒答/纯有害语义，作者也承认FH间仍共享高层概念。Table1 random集100倍大而非等数/等幅控制；不能把均值差直接当精确归因faithfulness。Fig3固定互补集合虽减少交互干扰，但得到的是该干预条件下效应。Append suffix时序与语义同时改变，未有等长sham/位置或语义对照；额外干预作用减少也可能饱和，未证明唯一通路。目标归因是AS构造参照，可能优先恢复AS路径而遗漏自然替代路径。
- **代理/图文疑点**：Eq4把向量减去标量cosine，按字面维度不对，若真为方向投影应核代码后再复用，不能自行补式当原文。Fig1图注“suppresses refusal features, leading to refusal”与P(I)下降和正文意图矛盾；Fig4正文数次拼作frictional，原图fictional。§4.4实际目标把成功越狱的有害输入视为低拒答类，与Table4有害/无害措辞易混；这是模型拒答倾向分类，不是可靠恶意意图检测。Table6我方Llama Alpaca CE .224相较base .145且高于若干基线，Table7 ARC下降4.1，不能概称最小能力损伤。没有误差条/seed使小差异显著性不明。
- **选择与泛化限制**：主要local feature集合和拒答方向从评估集提取，原文诚实说明只研究解释而非泛化；不该拿它直接判部署性能。WildJailbreak只保留改写后恢复拒答的38/42%子集，部分改写改变违法性明确程度/任务语义，条件性比较不能解释所有成功越狱。20只是设定及调参后的稀疏数，不是最小性证明。自解释标签仍要验证，原文承认SAE及候选方向局限。
- **复用/未答**：可复用可见词—内部特征—行为三个层次、首token代理与完整行为分开、chat模板与自然语言分开、正常能力对照。尚未回答共享生成历史下trigger直接效应、历史中介效应及OA观察器失败。本项目若只把输入触发词连到概念再连输出，已有直接先例；需要明确要发现何种此前未验证的规律。参考Sparse Feature Circuits另补全文后不递归扩展一般SAE谱系。

## Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models

- **身份/版本/全文**：Samuel Marks、Can Rager、Eric J. Michaud、Yonatan Belinkov、David Bau、Aaron Mueller，[arXiv2403.19647v3](https://arxiv.org/abs/2403.19647v3)，2025-03-27，刊头ICLR2025。36页正文、参考文献、附录A–H全文读完；Figs1–25和Tables1–7所在23页已渲染核对，`figures/sparse_feature_circuits/`。跨域机制归因基础直接纳入，不把它归为后门研究。细密全图用于核布局、可见标签和主要路径；未逐边重新计算作者图。
- **对象/方法**：对Pythia70M各层attention/MLP/residual/embedding及Gemma2-2B各层attention/MLP/residual，用SAE标量特征加完整重建误差向量构造计算图。Eq2定义指定替换参照下节点的间接效应，Eq3一阶归因patching，Eq4用10点IG改善近似；Eq5边权排除中间节点路径，仍为局部线性近似。节点/边依绝对平均效应阈值筛选。模板任务保留位置，非模板任务先跨位置求和再跨样本平均（Fig6/AppA.2）；后者一节点代表全输入各位置，不是逐词传播图。误差节点三角形保留模型未被可解释SAE覆盖的计算，不应静默删除。
- **实验及图证据**：§3四种主谓一致结构，Pythia70M和Gemma2-2B；Fig3在未用于找图的测试对上检验faithfulness与补图的completeness，理想值分别1和0。实际只评后2/3网络、跳过首1/3；同样节点数比较时，一个error节点是整向量，不能等同单神经元，作者给出排除error/attention-MLPerror的控制。§3.3选择人工能读的86/223节点图只有约.21 faithfulness，Fig4汇总的顺畅故事不能当完整机制。Figs7–15/AppC图解释主语数检测、从句边界和Gemma的NP数追踪。Fig25/AppH对30个RC输入逐节点精确patch效应校验，IG改善早层残差与MLP的一阶低估；不是任意任务/边/集合干预的充分验证。
- **最直接已做 `done`：图发现→人类认知→模型编辑→相关性打破后的任务评估**。§4 SHIFT先用只含male professor/female nurse的歧义集训练LM线性分类头；找图后，人工借Neuronpedia的最大激活文本和输出logit效应判断任务无关特征，删去Pythia67个中的55个、Gemma46个中的43个；可选在原歧义集重训线性头。Table2以四种职业×性别均衡数据评价：Pythia职业正确率61.9→88.5（SHIFT）→93.1（加头重训），最差组24.4→76.0→89.0；Gemma67.7→76.0→95.0，最差组18.2→50.0→92.9。Fig16显示性别与职业路径，Figs19/20展示为什么删女性词特征、保留护理特征。随机同数量特征干预几乎无益，CBP和oracle/skylines提供比较。
- **这个闭环的准确范围**：验证是**同一BiB职业分类任务的分布变化/打破混淆**，不是独立新任务、独立新数据集或训练数据清洗；编辑的是内部特征，可选重训的是线性头，未重训整个LM。初始分类器受控地学到偏差，不能称自然部署后门修复。并非“完全不用外部信息”：标签选特征阶段不用额外消歧分类数据，但依赖无标签大语料、预训练SAE、人工相关性判断。AppE.1披露Gemma初始层22用balanced集选出基线泛化最差层，目的是制造改进空间；作者称SHIFT超参不以balanced性能调节。该例不能作为无任何评估信息参与的完全盲测。Pythia按歧义测试集选层4。正文未给此任务完整样本量/重复seed/置信区间，不能据个位小数断言统计等效oracle。
- **已有另一认知用途**：§5/AppG以The Pile正确且高置信next-token样本聚类，过滤重复bigram induction例，再用每簇next-token NLL找图；Fig5/Figs17–18将看似统一的行为拆为succession＋窄induction、以及两条预测“to”的路径。作者提供数千未注释图界面，公开承认自动簇/图的全面质量评估仍开放。此处目标是语料真实下一词，不是多步自由生成输出热图，也没有把自动发现的两路径再拿独立任务做修复验证。
- **解释与因果限制**：单节点替换效应有清楚的干预定义，但阈值图不是唯一/最小/完整因果结构证明；边近似与节点合集交互仍需独立验证。先位置求和、再样本均值可能掩盖相反效应与异质路径。人工标签来自最大激活样本，不天然代表普遍语义；AppF明确标注者来自ARENA安全研究群、容易按话题忽略句法，Table7可读性评分不是faithfulness。Gemma BOS特征因难解释被排除SHIFT与feature skyline，不能把未解释部分判无关。性别准确率接近50%不等于表示对性别独立；删除后加头重训有效亦不逐一验证每个原标签的因果语义。neuron版SHIFT因难解释而放弃，neuron skyline是获balanced信息的另一基线，不能声称完成同等人工成本比较。
- **图文/复用疑点**：Fig9是PP图，caption仍写“last token of RC”，应作图注笔误；AppB.1.3文字的CE recovered比例缺少与高恢复率数表对应的明确方向，实施时核代码。第2节把增广图前向写成精确分解，不意味SAE特征均可解释或误差小。§A.3明示重建误差梯度处理，否则会出现抵消；未执行代码，不能把算法段落当已复现实现。代码/数据/SAE指向[作者仓库](https://github.com/saprmarks/feature-circuits)，本文未固定commit、核license/依赖或运行。
- **可复用/未回答**：可复用误差节点显式展示、保留位置与聚合视图分开、图支持的人类假说转成可检验干预、独立于找图样本的faithfulness、普通任务与最差组并列。它已实质覆盖“借可视化发现以前未预设的机制并用编辑验证”的一般理念，本项目不能以这一理念首次或画图本身为创新。未研究OA/后门trigger条件，也未在共享生成历史下同时比较触发器对每个输出token的概率效应、内部归因与观察器分数，或检验历史中介与输入直接作用是否分离。后者是需实验成立的具体候选认知问题，不是本文已证明存在的空白。按最后范围要求不递归扩展一般SAE/电路搜索参考文献。
