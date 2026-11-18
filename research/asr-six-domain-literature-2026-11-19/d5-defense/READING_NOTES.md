# D5 A 实际阅读记录（按首轮 B 修订；2026-11-19）
页码为PDF物理页。下载或摘要筛查不等于全文完成。

## BI-BAU — 2606.14078v1
实读正文pp1–14；相关附录A–E pp16–23。定位§4.1定义3/假设1/命题1–2；§5.2–5.10；App B/D/E。
- 视觉分类ResNet18/ViT/VGG及CLIP；§5.10说LLM为未来扩展。
- complete定义是指定clean/poison输入函数与clean model一致；依赖NTK/任务正交。App B.1由Jacobian乘积为0推出参数差0未展示满秩条件，故仅转述作者论证。
- Table2 CIFAR10 BI-BAU(U)平均R-ASR12.48/Q-ASR5.75；Table10 TinyImageNet SIG R-ASR96.72，Narcissus66.68。非全部清除。
- p9定义R-ASR=需要权重访问的retuning，Q-ASR=query-based reactivation；以上12.48和96.72不可称query-only。AppD.2 p18 TinyImageNet SIG retuning用2条毒例+训练集.02%良性；QRA另用500 benign+500 backdoored CIFAR例训练生成器50epoch、预算16/255。
- §5.7/Table5增强适应性攻击clean约92降至74.75；No Defense Average栏疑似错填，不采用。
- App D.3按攻击类别调参；E.1人为调序仍含交叉rehearsal，非自然混合训练严格两阶段证明。

## Suppression Is Not Erasure — 2611.12807v2 (15 Nov2026)
实读正文pp1–18；App A–G pp22–29。定位§3–7及Tables1–12。
- 以Qwen2.5-0.5B rank8为核心，跨架构static≠全量reactivation；Table1 Spark4B失败，Table6是在rank8 adapter上k=1投影后句法54.2%、位置8.3%；k=6 DSE/DSE-BA该面板静态均为0，反驳泛称全部归零。
- static归零后更快恢复；DSE减少早期恢复优势，DSE-BA引入benign reference改善第2epoch，Table3第3epoch仍100，非永久擦除。
- 主要merge adapter后新LoRA，不能说全由旧A存储；原地gradient审计另为Regime I。
- Table9 50 prompts×3 seeds DSE-BA ep1 1.3±1.9/ep2 43.3±2.5，勿以12-prompt的0/25代替。
- Table10格式43/50，原50/50；MC115/300 vs116/300不显著≠无损；纯benign adapter盲投影伤风格。
- Table7等预算Late优于MLP-only，未匹配比较混有扰动量；运行时filter非清除。
- Eq6/AppA2谱界疑缺条件；非零梯度亦非ASR保证，报告不背书数学保证。

## P2P — 2026.findings-acl.600v2 (10 Nov2026)
实读正文pp1–9；App A–C pp13–16。§3.1/4.2/各表/App A与C。
- benign trigger与扩展label训练，推理仍需安全prompt/映射；MaaS控制input stream。条件防护非撤除后擦除。
- CR CBA仍19.73%；attention竞争为作者解释非独立中介验证。benign多于poison。
- p13 AppA首段sup等式只是函数类可表达性、非实际优化达到最优；后一段假设alpha→1/epsilon→0；扩展label不自动推出投影后target率0；表示容量构造非收敛保证。

## Shared Latent Structures / CAFT — 2606.07963v1
实读正文pp1–9；App A–F pp12–17。source feature发现需clean模型及source trigger；对目标不用trigger≠全流程无先验。
- Table1双向steering支持指定干预causal contribution；password极性反转、weight-edit弱。
- §6 CAFT训练对照与AppE Table5 MMLU；AppD p14 Gemma CAFT学习率2e-5、默认SFT2e-4，非全匹配。
- Gemma4B country32.32→29.20较弱；AppF.6 p15 Qwen3-8B inductive条件为正/负steering，共享feature不强激活且不可靠缓解；没有该条件的CAFT实验。
- 非自然训练集中陡升时间机制，也非online预警。

## MB-Defense — 2026.acl-long.1113
实读正文pp1–9；附录A–I pp13–19。定位§3–5、Tables1/4/8/13。
- 防御者训练自有触发/随机词响应，再以128干净样本与防御触发正确响应偏好损失中和。不要求攻击者触发/目标；不像P2P，上线无需安全key。
- 4模型Alpaca；218 WizardLM提示评估+CACC不是广谱知识能力。Qwen8有CACC下降，不宜泛称无损。
- 最后层PCA共享是部分的；AppB Table8有benign perturbation控制，Toxic-BadNet/Insent相对控制差为负。不是独特普适共享回路证明。
- p9 Table4 Qwen1.7B toxic防御触发四→六：InSent ASR0→.972、AACC.729→.303；并发多攻击触发仍未研究；AppG coherent sentence target劣于随机词。更多防御不保证更好。
- 无中和后的再学习/重新激活审计。

## Dummy Backdoor as a Defense — 2606.11648v2
实读正文pp1–10，附录A–E pp11–18。§2.2/§3，AppC/D，Tables10–11。
- 需要知道攻击目标任务但不知道触发；支持原训练混合或事后植入dummy再中和。TAC最后token clean→trigger方向相似、AppC输出相似控制支持部分共用，作者也承认非决定性因果证明。
- AppC输出控制仅4例及16例，主面板99题；Table3同任务多个未知trigger已有测试，区别于MB未测并发多攻击。
- 主任务jailbreak ASR为不含拒绝关键词；不是逐样本实际有害回答正确性。MTBench只覆盖另一维utility。单run无方差。
- Table10跨任务dummy(负情绪)修jailbreak表现差：Mistral82.83→83.84。Table11数学可恢复但Mistral clean53.5→45.5。
- 未做防御感知攻击或后续恢复审计。

## Locphylax — 2026.acl-long.920
实读正文pp1–9，附录A–E pp13–19；图的文字和说明已读，tSNE像素不作为独立证据。§3/4.4/4.5，Tables1/2/7。
- 对已感染模型加入两个已知防御触发及中性响应，显式clustering loss聚合最后层，再以正确回应恢复。触发未知但需要干净数据。
- 聚类部分由目标函数强制且攻击输出被覆盖；不是预先存在的普适共享回路证明。attention head替换干预可支持该局部头对行为有因果贡献。
- SFT均值ASR约4.41%，不能推广所有情境；AppC Table7 syntax AdvBench仍23.07%。
- §4.4描述先拒绝后恶意内容；AppA4的缺拒绝关键词ASR可能漏掉这类回答，需保留评价边界。
- 没有再学习/撤销防御后恢复测试，不认定永久清除。

## DiSec — 2026.findings-acl.815
实读正文pp1–10；附录A–L pp11–20。§3.1–3.3；Tables1–9；AppD/G/J/K。
- 清洁无标签辅助语料上梯度残差SVD/VAE异常行联合检测、用同层未标记行均值替换；挖掘代理触发只作层定位，不需真实攻击触发。
- BERT/RoBERTa分类模型，不是生成LLM；20k BookCorpus，另OpenWebText；单seed。后续全参数任务微调仍抑制，但不是故意恢复攻击测试。
- Table1 NeuBA-RoBERTa SST2 clean94.67→88.96；AASR12.20（跨触发均值），MASR14.69，不能写无损/清零。Table3 adaptive BadPre SST2 AASR19.45/MASR23.76。
- 表征定位跨层突降不是随训练时间的形成阶段。行异常不等于已验证后门专属行；clean模型也选出2/4个“vulnerable”层。固定标记集合均值替换idempotent不代表端到端重检测幂等或安全保证，作者Limitations明确无formal guarantee。

## SelfHeal — 2026.findings-aacl.248
实读正文pp1–9；附录A–G pp12–21全部（攻击样本仅理解评价任务，不复述配方）。§3/4，Tables1–8/11–15/17–23。
- 信任base model；训练前由其标异常语句、清洁语句、重建并以task关键词比较响应；前两阶段prompt主要任务无关、第三阶段含任务特定要求；依赖原模型安全能力，不是无需先验的通用清除。
- Llama3-8B/Mistral7B、三任务四目标；主50%污染，附录10/20%；五seed主表。优点是不丢整条样本；仍可能改变benign语义或标签。
- Table14 clean Mistral SST2经处理ACC96.52→82.09，Table12 syntax81.59/52.86；主Table2 Mistral分类ASR76.08最差。不支持“总保正常能力”。
- 主ASR多数关键词，附录B4加GPT5.5评估；TaskRefusal部分结果与关键词不同，说明自动评价范围有限。
- Table8 Stage3-only jailbreak均值有时更好，不能把三阶段每个都必要写成普适因果定理。无感染模型删除/再激活主张。

## A10 — Concept Ablation Fine-Tuning (CAFT), arXiv:2507.16795v2
读：PDF正文1–10；附录A–D 18–34，E.1/E.2 34–35，F–H 40–43。E示例长转录36–39未逐条读；不以其作证据。版本为2025-11-09 v2，不是2026论文。
- §3训练前向/反向均投影移除概念方向，推理撤去干预。§7明确目标是控制泛化而非删除概念。发现PCA方向先要有普通微调后的模型，不能说训练前自动已知方向或在线预警。
- §4正文及p21 AppA.5 Table1（Qwen）/Table2（Mistral）测量：不安全代码微调Qwen2.5-Coder32B与MistralSmall24B，PCA CAFT使一般提示misalignment从7.00→0.51、6.57→1.16；同时不安全代码漏洞率91.6→84.8、92.4→92.2。这里维持的“目标任务”本身是不安全代码，不应转换成良性任务保持。misalignment分母是coherent回答；4.5k训练例、8一般提示的3格式与1k代码测试限制外推。
- §4.2及p21 AppA.5 Tables1–2：5seed、训练早停Pareto、随机方向等对照有价值。Mistral top5 PCs一般misalignment .34低于CAFT1.16、目标漏洞90.5低于92.2，不能宣称可解释方向对任意约束均最佳；Mistral随机SAE对照类似。AppA6 MMLU/GSM8K；A7加入2%良性例也达类似低misalignment，是数据可改时的简单替代。
- Gemma2-2B构造16任务，SAE10/16、PCA5/16改善；词性/代词例受纠缠限制。A4只推理消融效果混合，说明训练期干预并非所有条件均更好。
- 证据级别：实测训练防泛化+条件因果干预；非自然后门形成阶段、非不可恢复清除。

## A11 — DETER, arXiv:2611.15479v1
读：正文1–9、所有附录13–25。主题是知识能力抑制/重学阻力，非触发后门。
- 方法：FineWebEdu过滤医学数据+周期负学习，正/负学习同权重不同优化器状态；113M–1030M从头训练。Table1的最大20.36×是达到同规模NO FILTERING模型PubMed loss门槛所需token、除以FILTERING模型达到该同一门槛所需token的倍数，绝对2.09M=预训练token的0.010%。所有方法预算内恢复；不能说知识不可恢复。
- 主pretrain各1seed、relearn3seed；学习率/warmup重学搜索较强。1B负学习系数与过滤比例不同，规模趋势不是纯单变量。
- 测量为PubMed heldout loss与医学/一般任务正确选项NLL，不是自由生成回答能力准确率。一般MMLU及生物子任务仍有损害；训练/筛选/部分评测共用分类器存在相关盲点，外部PubMed部分缓解。
- §3.3/Table2与AppTable14：在近似同负更新数/一般loss条件下比较起始时间；113M同间隔early2.77 vs late3.67，不能归纳越早阻力越好；larger models早期更利于utility。Table3优化器状态对照，Muon113M共享状态biology恶化25.60 vs separate .57；共享二阶矩也可伤正常能力。表征分化解释尚非直接中介证明。
- AppD.1与posthoc RMU比较，未直接比较SGTM/TAR（正文未来工作）。额外负学习约2%算力，不应写零成本。与项目相接是训练时序/状态及能力保留的证据边界。

## A12 — TopKLoRA, arXiv:2610.32866v1
读：正文1–10，附录A–G 13–18、20–29；19页长输出转录不作实质证据。
- 冻结base加ReLU/TopK稀疏LoRA；已知触发/非触发配对输入，三base（Gemma2-2B、Qwen2.5-1.5B/7B）。Qwen另替换chat相关embedding行；并非只改任意原模型LoRA。dense rank-matched对照只有前两者。
- CLCD搜索/消融使用分离数据；必要性是verification1000提示ASR零，充分性是keep-only相对完整adapter损失不超过2 paired SE；搜索前缀网格不保证全局最小或唯一。消融发生postTopK而不重路由，属于固定门控下持续零化干预，未测重新训练后保持。
- ASR仅前40token是否出现目标短语。AppG.3另4000提示最多3个命中，未留全部输出转录不能区分引用/实际攻击。不能说所有生成长度严格零。
- 多层电路保留较好，单层小模型utility retention中位.12/.09；不能说任意电路剪除无损。Table4 clean twin Gemma9band judge3.06 vs3.14，差-.08CI[-.13,-.03]虽小非无变化。
- AppG.1不同搜索次序电路Jaccard7–23%、共享core3–9；不唯一。planted circuit校验支持局部定位，但完整planted集合只2/11足够，其他可能含抑制元件，不能把“全真集”简单加和。
- 已做过：人为稀疏适配器内必要/充分的后门电路干预。未做过：普通稠密微调自然形成统一机制、未知trigger部署方案、relearning审计。

## A13 — ABL, arXiv:2110.11571v3
读：正文1–10、附录14–20全部。2021 NeurIPS，复制d1源但独立全文。
- 原始带毒训练中多种攻击的平均loss下降较早，视觉WRN/CIFAR10等；快速学习是条件实测，不是所有样本/攻击必定更快。
- local gradient ascent强化loss分离，隔离低loss1%后global gradient ascent；AppA.3实际20epoch隔离+60epoch普通训练+20epoch GGA，主文“两阶段”是方法概括，不是证明自然形成两阶段。
- Table1 CIFAR10平均ASR7.69、ACC84.76 vs clean89.12，非无损；CBA29.81、Blend16.23、Dynamic18.46残留。Table2高投毒时Blend62.28、Dynamic69.36，存在失败。
- B.5主动提高poison loss的自适应攻击被列为future，未评。B.10 NC+GGA额外依赖1%clean和反演trigger。NAD/FP为本文基线重实现，不替代各原论文核验。
- 无再激活试验；得到候选毒例与端点降低不等于彻底删除。

## A14 — Critical-CoT, ACL2026 long.495 / arXiv:2604.10681
读：正文1–9、AppA.1–A.10 prose12–15；16–19图式转录未逐像素读，不引用细节。
- 合成防御推理数据+50%clean，先SFT再DPO；3模型（GPT-OSS20B,Qwen3-14B,Llama2-13B）。训练式主法与附录training-free版本分开，不能一概称提示拦截。ICL攻击（参数未污染）与FT攻击不同实验。
- BDR/TDR检出与ASR/clean accuracy分列。Table6 SFT FPR13.4、clean76.5；SFT+DPO FPR1.0、clean91.7（base93.9）；说明审查本身有误报/能力代价，DPO改善。
- §5FT后修复不保证恢复原能力，GPT-OSS clean/trigger约72.8/72.4，不能拿相互接近当恢复93.9原能力。
- §7未测语义/上下文触发；填充trigger在推理题中可忽略，不代表所有任务都可过滤不丢信息。p5训练示例@_@/In arcane parlance又见p7所谓unseen的触发列表，论文对划分的描述存在不清晰，不能据此确认数据泄漏，也不作强跨触发结论。3次均值但无完整方差。无再学习/参数中介定位。

## A15 — Fragmented Backdoors, arXiv:2608.30403v1
读：正文1–9、所有附录12–22。
- clean/poison模型×clean/trigger输入2×2输出margin差分；数学分解是恒等式，不等于三机制独立。SAE线性logit投影对下游非线性只是近似。
- 3模型Gemma9B/Llama8B/Qwen7B、3分类式任务AGNews/SST2/LLMLAT、3trigger类型、dirty/clean label共54设定。表征组成随条件变；没有在控制训练其它因素时独立操纵组成来证明防御失败仅由结构组成导致。
- 特征分析需paired clean模型、已知trigger/target；防御用protected输入对小clean校准的正向异常feature。§5.1选择minimum ASR且minimal CACC损害的结果，存在评测选参依赖；未交代足够可部署选择协议。
- 推理时把异常feature夹到base或cleanFT参考值，并保留SAE重构残差；运行时抑制，不是持久清除。Table5 BaseClamp有28.2残余ASR、CCClamp最高15.4；clean分类ACC不证明广泛能力保持。
- AppG承认SAE逼近。H不同SAE宽度/层稳健性只定性，不应声称计数完全一致；首稿所说“top60类别计数不一致”未定位具体表/列，已撤回，不作为负面证据。

## A16 — Mitigating Backdoors via Decoy Shortcuts and Knowledge Decoupling (TR), arXiv:2608.00732v1
读：正文1–7、所有附录10–19。IJCAI2026。
- 浅honeypot与backbone并行，熵权重、特征/预测/原型等解耦约束，训练后去掉honeypot。视觉CIFAR/GTSRB/ImageNet与ViT微调，不是LLM实测。
- §4.1初始双分支测得honeypot优先学捷径；AppB无解耦最终主干也学后门，所以早期转移可仅暂时。AppB inequality19浅/深Jacobian大小依赖假设，图6量的是logit梯度，不是参数Jacobian范数；不能声称严格证成所有架构快学机制。
- Table1 CIFAR10 avg ASR.57,BA89.40 vs unpoisoned91.19；GTSRB .10/98.08 vs99.33。AppC自适应攻击在surrogate上构造再换初始化训练，作者明确迁移差，非任意whitebox保证。
- **AppE.6/Table14（p17已渲染核图）：训练分支权重动态个案中，triggered图像可高置信度误分成另一非目标类；降低targeted ASR不等于恢复triggered任务正确率。此表不是最终部署测试集错误率，不作该层外推。** E.7 honeypot末期可能对所有输入报target，不能把所有激活当专属后门知识。
- Table12 honeypot大小/注意力配置能使WaNet/Dynamic ASR65–88；并非容量越大越好。Table11总体BA掩盖target/罕见类准确率下降。无后续重新投毒/再学习审计。

## A17 — HARVEY, arXiv:2607.05748v1
读：正文1–8、所有附录10–15。arXiv上传2026-07-07，第一页AAAI版权脚注2025；未另核会议记录，书目只采用arXiv版本。
- 方法训练一个偏向后门的reference/oracle（这里oracle是学得模型，不是真实毒标签oracle），RCE按轮划分；末轮reference偏向target类，所以用首轮reference meta-split回收良性target样本；最后从头/干净初始模型在筛后数据上训练。属于毒数据过滤+预防最终模型形成，不是原受感染模型擦除。
- 视觉3数据集/多CNN，Table2 CIFAR10ResNet18 ASR各行至1.22（WorstCase印1.43与行值不合，故不复用其worst汇总）；TinyImageNet最高.48；Table10 VGGWaNet2.15、Table14 adaptiveWaNet2.57，正文“worst<2”只限主表。
- AppD.5真正clean-only控制，WRN/GTSRB97.42→96.12、CIFAR89.42→88.52；不能说没有正常能力损失。数据切分精度不完美，WrnWaNet precision49.93（Table3），说明移除了不少良性样本。
- AppD.1调参敏感，大LGA/unlearning系数损伤；AppE仅重复良性样本的适应攻击，低ASR但ACC损伤，无全类适应攻击覆盖。主文未报告重复seed置信区间。
- 修正：本报告先前口头“oracle poison过滤”若被理解成真实标签oracle不准确，应为“学得后门oracle用于无真实毒标签过滤”。

## A18 — DPoE, NAACL2024 long.27 / arXiv:2305.14910
读：正文1–9（印刷483–491）；AppA–D 13–14（495–496），参考文献10–13扫描引用。
- 低容量trigger-only模型+主模型PoE联合训练，推理仅主模型；四种denoising保干净能力。数据投毒对手不能控制训练；无真实clean参考，pseudo-dev依两模型confidence挑样选择超参。不是posthoc权重擦除。
- BERTbase与Llama2-7B均实测，但任务仍为SST2/OffensEval/AGNews分类，§Limitations明确生成需非平凡改造。Table2 LlamaSST2 syntactic ASR19.54、ACC93.67（nodef95.88）；不是零后门/无代价。
- Table1不存在全任务全攻击最佳的单一denoiser，作者也承认。干净模型在部分triggered输入本身ASR明显非零，所以零不是通用干净参考值。
- AppA梯度论证假设trigger-only已捕获捷径、对clean均匀预测；不能从PoE方程单独推出优化必然满足这两条件。AppB pseudo-dev有效性假定高纯度与选择集代表性，后者未充分证明；不能当无真实标签的性能保证。AppD clean-only控制有价值。
- 后向是自然bias PoE与ABL、CBD；前向TR说明分支捷径隔离家族已有持续演化，不能仅凭换浅分支叫全新。

## S01 — Survey, arXiv:2502.05224v1（仅部分阅读）
读：§4–6 PDF17–26，相关references33–34；未读全部攻击目录正文1–16，**FULLTEXT_PARTIAL**。只用作检索种子，不作一手性能证据。
- 提供数据过滤、训练预防、参数修复、蒸馏、推理过滤目录。部分归类容易误导（把有权重/激活访问的算法混称black-box，把推理期删除trigger说“eliminate backdoor”）；不直接接受。
- 引出Honeypot2310.18633、Fine-mixing2210.09545、BEEAR2406.17092、DBD2202.03423、I-BAU2110.03735、NAD2101.05930、T&R2022、MuScleLoRA2402.12026、ModerateFitting等。首稿仅有ABL/DPoE及后继；B后已补Honeypot/Fine-mixing/BEEAR/DBD/I-BAU/NAD/MuScleLoRA全文，T&R/ANP/CBD仍仅范围定位，见后附逐篇记录。

图像复核：已渲染并查看 TR p17 Table14、MB p9 Table4、SelfHeal p14 Tables12–15；对应PNG存sources。其余未逐图像素审查，所有图形趋势按正文/图注转述，不声称独立复算。


# B 后补读记录（B 第二轮已核；三处小修待定点确认）
页码为PDF物理页；不把提取文本中的控制字符当分页（MuScleLoRA已单页重提取校准）。主文全部及相关附录阅读；参考文献用于追链，纯示例/提示目录不等于独立验证。

## QES — 2610.00663v2，29页
已读主文1–9，相关附录17–29。冻结base，四路LoRA专家，attention q/v普通LoRA亦训练；completion→prompt注意力深层半数、max-head saliency排特殊/停用词，结合样本peakiness施加四个路由正则；不使用真实trigger标签。预指定e3隔离；部署保留所有权重，每层每token屏蔽e3再选专家，是持续运行门控而非删除。没有撤门/再学习耐久测试。只有一个utility专家的变体可合并路由，不可套到默认模型。
AppB3 pp19–20明确将GSM8K **test 1319题纳入训练**，与250 poison和500Alpaca合2069；名义33.3%是750子集比例，全文实际12.08%。主∆U为同模型开/关对已见GSM8K的保持；不能称heldout推理泛化。九个独立base benchmark的Table7仅Llama2-7/13 BadNets sentiment。四专家容量与普通LoRA不同；每个防御的参考模型也不同。200触发+200clean测ASR，以固定侮辱/拒绝短语substring判断，20拒绝关键词；不是任意开放有害目标。Qwen CTBA仍36.5/37.5%、MTBA22/24.5%。T12三seed只单个Qwen/BadNets cell，off2.17±.76。T14三触发/目标同checkpoint：99→8、99.5→7、27.5→0，第三目标基线较弱。AppD26–27“sleeper/VPI-style”非原始欺骗CoT设定，多个模型无改善；数字触发稀释/GQA为作者解释，不是已排他证明。grad-input替代单cell只100→77，clean routing10.7→40%，utility -2.43→-3.03。容量/路由消融支持条件性设计，未证明自然训练必有隔离。

## Phantom Transfer — 2602.04899v3，62页
已读主文1–10；相关C/D16、H27–28、J33–34、K39–41、L43–46、Q/R/S59–62；p19示例浏览；其余长提示、regex、输出目录未逐条读，不作为结果证据。2026-10-07 camera ready。Gemma3-12/GPT4.1教师生成concise Alpaca输出，经regex+LLM过滤显式实体/情绪。主数据52k、两epoch、默认100%是投毒生成数据，不能混同稀疏poison-rate。四主学生GPT4.1/mini、Gemma12、OLMo13；拓展Gemma4/12,Llama8,Qwen14两目标20k。跨家族成立于多cell但非所有实体；T27三十实体有多项零/一项负，均值.072/.088/.135是跨实体SD并非多seed。ASR是50种favorite/leastfavorite问法和目标/相关实体，不是广义恶意动作。
所谓11防御包含random10%控制。基础六种为random、wordfreq basic/oracle、LLM basic/oracle、paraphrase；额外五种中四种使用同一外部all-MPNet-v2句embedding，不等于原始victim内部activation detector实现。wordfreq basic按Dolly5%FPR校准、oracle使用matched clean生成数据。LLM oracle知道精确目标、机制、正反例。paraphrase仅重述completion且保意义/简洁。oracle分类器用真实四毒类+clean标签，主文p5称每类2k/~30%acc而K5p41称每类12k及80/10/10、.40acc，有预算/结果文本差异；不擅自合并。T5GPTmini/Stalin oracle classifier后ASR=0，其余Cath34/Reagan56/UK36，故“没有一种在所有设置稳定防住”而非“11种在每个case均失败”。T29 96run是异质配置组合非独立重复seed。
效用L2T30保持简洁；T31 Gemma教师的poison数据后MMLU-Pro GPTmini .23–.36 vs clean .56，Gemma .09–.27 vs clean .43；不能称全部正常能力保持。密码条件§6用Reagan语义上下文触发Catholic偏好，25kpoison+25kclean，相关美国语义也激活，不仅精确字串；逆向较弱；该条件只测oracleLLM+paraphrase等，不是全部11防御。H27–28同构steering生成反例：500均值差、Gemma4方向、10k生成，多数<20最高26且残留显式实体，不能泛化所有steering不能传。
Q59–61审计Petri只有一个cleanGemma控制跨模型比较，不是所有模型matched baseline。标准直接问something-off=100%但clean30%；specific caught=10/30/60/0。条件任务something-off20/10/80/0 vs clean10，specific caught0（Gemma prefill10）；不是无信息可检的定理。R61–62Davies对照重构用GPT5、1035编码+1035clean，仅过滤没有后续FT ASR，不能称复现原文全部防御实验。共同预训练关联机制仍假设。

## Honeypot — 2310.18633，20页
已读主文1–9、附录13–20。冻结RoBERTa的各层probe是单Transformer+linear，SST2词/风格5%poison，低层probe快学毒不等于完整PLM自然学习顺序。Honeypot低层支路GCE偏易例，warmup一epoch；主支路按honeypot CE /主branch近100step平均CE加阈值c=.1权重，疑毒权重趋0。双头训练后丢弃honeypot推理；不需外部clean或真实trigger。BERT/RoBERTa base/large、SST2/IMDB/OLID四类触发三次不同seed/target。T1SST2RoBERTa风格20.87±3.90/句法22.9±3.85残余；OLID个别30.12不符文中笼统<30。T2词触发clean94.61→93.71。
AppC16–17说明完整PLM可能先学正常语义后毒，ABL重实现1%隔离精度2.1–15.6%、ASR67–90，与from-scratch视觉不能无条件混用。AppE18短warmup100/200step ASR100、400step8.64；T5p9 c=.8下SST2ACC67.1/ASR30.28及IMDB63.11/34.75。§5.3组合适应攻击残余24.9。§5.4.2文本style<15与T4的20.34–28.85不同，以表为准。AppF视觉预训练VGG16/CIFAR简单两触发8.81/10.81，与“两项<10”不符；无生成LLM/再学耐久测试。

## DBD — 2202.03423，25页
已读主文1–9、附录15–25。训练者全控训练，只有潜在污染数据、不需额外clean；先SimCLR去标签学表征，再冻结backbone以SCE训head、选最低loss50%可信标签，最后MixMatch全模型，余下一半作为unlabeled并逐epoch重筛。此时低loss解释为clean，与ABL的原模型低loss毒相反，说明loss含义依赖训练机制。tSNE是解释；T3消融支持多阶段配合但不是唯一中介。CIFAR10/ResNet18、ImageNet与VGGFace2均30类子集（后者9051train/2263test,DenseNet121）。T1CIFAR BadNets94.92→92.41BA/ASR.96，WaNet94.29→91.20/.39；T2标签一致BA94.90→89.67/ASR.01，不能照抄“所有BA损失<2”。AppB17最终SSL100epoch，AppA观察用1000epoch；K23成本SSL每epoch约4x、MixMatch约2x非计算匹配。
p9适应攻击只优化代理SSL聚类，重训随机性使转移失败，ASR1.02BA93.21，非全适应保证。H22 α过大可能可信集中再含毒造成重新植入；L24T23DBD可信集仍8/25000、6/25000、38/25000、13/25000毒，证据不是完美检测；SS/AC后普通重训仍可近100ASR。M/N24–25对MoCoV2/BYOL和GCE/APL有条件替换，均低ASR；O25 clean-label解释为对抗扰动改变特征来源，是作者机制解释，无独立因果消除。无LLM或防御后再学习实验。

## I-BAU — 2110.03735v4，27页
已读主文1–9、附录13–27，p5因初次输出截断已补读。感染模型白盒+额外有标签clean，主设5000clean从原test划出、剩余测试；非无需clean。minmax共享有界加性扰动+外层修复，以implicit hypergradient近似解；实际inner次优且Hessian可逆前提。§5与A1的保证涉及强凸/光滑、内层渐近收敛和hypergradient误差，不是任意非凸优化达到全局安全；A1定义Φ=δ+α∇H后导数却I−αH且强凸，存在符号不一致，不能以此背书一般收敛。泛化界依赖norm、margin、iidclean和共享扰动；未证明触发删除或抗再投毒。
主模型简化VGG，CIFAR/GTSRB、20%poison七种触发；CIFAR T1按cleanACC匹配最好baseline选择终点，Smooth仍18.30ASR、83.46ACC；7触发7targetT3CIFAR98.37→12.96ASR、85.96→77.44ACC（8.52pp），不能仅报告稳定clean。100clean T5只CIFAR/TrojanWM：4.00ASR但84.92→76.9ACC；不是所有攻击100例均成功。7.35秒是<20ASR首次有效，一次单目标，A7多目标CIFAR超过200round。A5非加性为case studies，WaNet10.80ASR、IAB9.58；SR实际仅一个相同HelloKitty测试输入。不同case有不同模型/lr/bound，不能说全设置同超参。无后续再学习；BI-BAU R/Q反例应保留为后继不同环境，不反向替代本篇结果。

## Fine-mixing — 2210.09545v1，18页
已读主文1–9、A–D12–18（图的曲线仅取正文有明确数值，不从提取乱码估计）。需要可信未微调预训练BERT checkpoint+同任务小clean标注集（通常64）；元素二值mask随机从感染/预训练权重取值，之后cleanFT；不是凸平均。E-PUR按embedding位移除log外部词频筛top200并恢复预训练embedding；不知原poison词频，但假设普通词频相似，原checkpoint无后门。选择ρ从0到1直到validationACC达阈值；主表defenseE-PUR有而部分baseline无，T3给ablation。§3的Sel保留大小及ρ描述有前后矛盾，报告只用无歧义公式/随机版；A13随机游走近似不是所有优化轨迹定理。
T1SST2均值91.70→89.53ACC、100→16.64ASR；IMDB from-scratch word残余66.19。T2QQP from-scratch sentence64例ASR88.71/ACC82.37；扩大512仍37.75/80.33；QNLI64例86.02/84.29。sentence-pair容忍约10pp损失，非全部正常能力几乎无损。T5适应E-PUR攻击QNLI32.48ASR、85.98ACC。C15 8/16clean失败；无防御后relearning控制。几何好坏盆地visualization为条件解释，非正式充分必要证据。

## BEEAR — 2406.17092v1，24页
已读主文1–9、相关附录A–C13–16、D p17一个定性示例；D18–24仅输出目录，不作为性能论据。白盒感染模型、无需真实trigger/clean参考checkpoint；需要三组定义的safe/harmful/performance锚点。共享9层末5token连续扰动内层找坏行为，外层正常化安全+300LMSYS性能锚，部署撤扰动。PCA同向是相关观察，不证明合成扰动等同真实触发，更不证明唯一回路。标准jailbreak400AdvBench训练/120测试，“Sure”是坏标签；代码sleeper重构Mistral并使用同源poison dataset留出1000坏代码作DSA-H，不能说完全未知攻击目标、也非所有原Sleeper设置。
T1四短触发SFT残余关键词5–9.16%、GPT4 4.17–7.50%；RLHF T2关键词.83/2.50%，两者GPT4 .83%。代码T3仅17题/8CWE，8/17→0/17，clean1/17→0，不能叫47%大样本。MTBench独一效用，且用其稳定作stop（非完全独立选择外测）；感染基线已损伤Model4MT3.62 vsbase6.37，修复4.23非全恢复。§8明确目标定义不匹配可能失败；没做适应性攻击/再学习。AppB14–16选层以ASR<25%成功矩阵，需15epochs，有失败层；性能锚缺失collapse，50例为该Model1条件经验。不能将低未知trigger约束翻成全未知行为保证。

## NAD — 2101.05930v2，19页
已读主文1–10、附录A–K13–19。感染模型+5%clean标注；先cleanFT teacher再固定teacher，回原感染student以CE+各residualgroup归一化平方通道和attention对齐。不是Transformer注意力矩阵、无需预先完全干净teacher，非输入拦截。主模型WRN16-1，五类CIFAR+Refool/GTSRB，T1meanASR99.04→7.22，ACC83.50→80.83；Trojan19.63残余。1%clean平均35.93ASR/ACC损失4.69，不能只说fewclean有效。β主表按攻击调整，作者建议按cleanACC阈值选；50000损失>50%ACC。不同架构teacher T2只BadNets且teacher从5%train，studentACC最低74.53 vs85.65。
A–D13–15有相同增强baseline；I17attentionvsactivation单BadNetASR3.81vs98.44支持条件设计，不证明全部inactive神经元清除。G16迭代不进一步改善且可回升2%；J17–18运行20epoch后student过拟合含残余后门teacher，earlystop约5依validationACC；这是真正时间边界，不是防御后新毒重学。K18–19适应仅中心位置/强度变化，不是完整whitebox适应优化。无LLM生成或不可恢复删除保证。

## MuScleLoRA — 2402.12026v3，19页（不是提取控制符产生的31段）
已读正文1–9、Ethics10、附录13–19；原read.txt图中文字内嵌换页符会产生假分页，按pdfinfo确认19物理页，并单页重提取为2402.12026v3.pages.json校准。所谓frequency是输入—输出映射频率，不是词频；pilot在扁平输入embedding空间用高斯核分解logit/onehot映射，假设embedding固定。所有poison标签同target使其高频真值为零，是该构造的条件。冻结base、低秩多尺度参数化及clean辅助梯度对齐共同干预，不是仅压低某自然后门方向。
AppB4 p15：LLM用128条验证集clean辅助数据、µmax=0，此时传入优化器的ĝ为clean辅助梯度的非负倍数，混合数据梯度仍决定该倍数；不能据此断言实际参数位移Δθ逐步共线；BERT等96条/µmax=.1。不能把全部收益唯一归于频率。T1p7 BERT/SST2 Style ASR33.22、ACC87.81（正常91.71）；T4p9 Llama2-7B Style39.03/94.73（正常96.43）。LLM添加分类头，非指令生成。T3p8 AG/句法去MS ASR8.18低于完整17.02，但ACC84.26低于86.05，并非每组件都单调改善ASR。AppC3 pp17–18 LoRA+MS后期仍学会后门，早期延缓不够。AppC5 p18 OOD只是IMDB同情感任务，BERTLarge Style33.88vsID21.16，不是跨任务完全无损。S需按模型调，未测修复现成感染权重或防后再学习。

## OPD / Lazy Defense — 2610.07654v1，27页
已读正文1–11，附录A14–15、B16–18、C19–20、D2–D5 23–27；D1 pp21–22具体输出例未逐条读，不影响所用结果。教师有后门、学生蒸馏；Qwen2.5同族1.5/3/7B，1000 HH clean prompts加重复带trigger prompts，只有提示被改，不是人为poison回答。top-k=16保留原全softmax概率而不重归一化；token-level surrogate γ=0相对sequence γ=1有偏。固定状态代数不是自然训练中介的唯一因果证明。
§4.2“top-k overlap”对应D3p24实际上是teacher-forced固定teacher greedy rollout，跳前7取随后10token的singleton top-1 Jaccard（即top1一致率），五数据集各取后半；并非student on-policy top-k分布重叠。Lazy在p8 Eq6把教师/学生logprob比值截到[−τ,τ]，保符号，非检测后只处理毒token；PDF未见τ数值，训练optimizer/lr/batch/seed未完整给出（已全文检索，不用猜值）。
T4p27 3B→3B 5% condition：epoch4/8/12 Lazy ASR2.3/45.0/69.5，sampled基线3.0/58.4/70.3；对应clean ASR2.3/3.3/3.8。3% epoch12 Lazy47.7vs67.2；3B→1.5B 3% epoch16 .8vs20.5（topk80.9），5% epoch16 20.2vs80.4（topk96.1）。支持有条件延缓，非长期阻止。LlamaGuard3对五类有害prompt集、greedy一次生成、未加权均值；“clean ASR”指不带trigger的有害问题响应，非正常任务能力；没有通用utility benchmark。

## Exploring the Orthogonality and Linearity of Backdoor Attacks — IEEE S&P 2024，21页
已读正文1–14、AppA15–17/B17–18、补充20–21；p19出版元数据。复用D1 c_sources下载件，本域独立读。Table1黑色符号为文献结果、灰色为作者新核，不把整矩阵当同协议新实验；T5是实际防御比较。两阶段continual近似首先是Assumption3.1，随后三类CIFAR/ResNet试验与单BadNets顺序训练端点相似，不是证明所有混合训练等价顺序任务。
NTK线性固定特征、平方损失、Jacobian/参数位移正交是理论条件；实测角度取1024例batch的平均loss梯度，不是严格相同对象。Thm3.6的upper bound小能支持小损失，upper bound大不能反推实际损失大。Lemma3.7同时对Φ/Φ转置用逆，要求方阵满秩等未普遍成立。Prop3.9依赖ReLU线性区与子网分离/高激活；线性R²也只测选出的top3%神经元，非全网络全域线性。
T5p11 CIFAR Reflection的FP/NAD/ANP ASR99.87/99.01/94.27，SEAM27.02；Composite91.27/93.89/94.90、SEAM52.35。这些是该文重测，不是原方法原篇的全部性能。T9p12标签特定patch3→6：orth49.36，NAD43.09、FP50.12，但任务条件也变，非只改正交角度。AppT12/13p18反例：低orth权重抑制Blend61.43、linearity.66仍NAD2.37/FP2%；WaNet54.19仍NAD2.91。故相关性不是确定映射。补充T15p20 Softplus47.07 vsReLU78.79仍两者ASR1、linearity.99；T16p21随训练变化有架构/数据条件。未测LLM。

## Safety-Efficacy Trade Off — 2602.00822v1，17页
已读正文1–9、附录11–17（代码仅阅读，无执行）。KRR中紧密poison cluster且与clean核块交叉忽略是Assumption3.1；平方损失、固定核下才有该拟合式。近克隆RBF几何可以函数变化非零而输入梯度/Fisher小；非所有诊断都不可见。Gauss–Newton项是梯度外积；完整Hessian还含残差×函数二阶导，不应在无条件下等同。
正文Eq25→26/Thm3.9讨论κ正则及固定G线性平滑式：原loss输入梯度平方含残差，通常不是参数的固定二次正则，改写成(K+λI+κG)α=y需要额外近似或定义；G=0等退化情况也排除无条件strictly。AppA5 pp11–12则讨论核长度尺度ℓ，不能与κ正则混写。B已核二维Gaussian Gram反例：λ=1、y=(1,1)，ℓ=.5→1时残差范数.66229→.54257，反驳其每个平滑因子下降/任意y残差严格上升的一步；该例df仍下降，不能据此否定所有df单调结论。AppA6 p12先采用忽略base loss的penalty-only近似；B核原式后确认现有证明不足以推出所述普遍收缩：A=diag(1,4)、g=(1,1)、F=ggᵀ均PSD而AF+FA行列式为−9，方向v=(2,−1)/√5的F能量导数为+4κ/5。具体可实现平方损失例与限定见B_REVIEW.md §6.4。这不否定部分总trace下降或p7实证折衷，也不扩成方法整体无效。
实证是CIFAR10/100 PreResNet110以及MNIST线性，从受污染数据训练时正则；不是拿clean集清除已感染LLM。T1p7 θ=.02、augmentation、κ=1e5：90→450epoch ASR6.63→70.68、ACC75.17→79.05。其他κ也高残余；延长训练可破坏早期防效。没有跨seed置信带。p8正文“单调旋转”比T2离散索引更强，报告不用该单调断言。没有证明普适免疫或保效用不可兼得。

## 跨域定点：BAD-BOOM（D2主读，本域不是全文阅读）
官方 https://proceedings.mlr.press/v306/zhao26bf.html 。独立读pp3–4/6–9及AppC15–16/D2p17，核防御耐久，不重复全文理论推导。常规trigger-free下游SFT：SST2 8k、5epochs；GSM8K 7.47k、30epochs；GPTeacher18k、10epochs；统一lr2e-5/batch8，单seed1001。ASR为1k Dolly触发prompt的固定目标string出现；utility分别为SST2分类、GSM8K精确数值、500Dolly由gpt4.1nano判helpful。不包括专门防御后的主动恢复。
T1p7原AdamW Qwen3-1.7B sentiment SST2 AddSent/VPI仍96.4/97.0 ASR，所以不能称常规SFT总擦除；BAD-BOOM多数更持久但也有Llama1B sentiment GPTeacher AddSent47.5，不能写全部≥90。同cell clean utility原AdamW16.8→BAD13.8；Qwen1.7B Sleeper GPTeacher47.8→40.2，未见多seed显著性验证，不能称绝不损害。两条随机正交方向的2D景观不代表真实SFT路径/全空间；有益几何解释未排他识别。D2负责完整理论与机制结论。

## 跨域定点：Cordyceps — 2605.26595v2（D3主读，本域不是全文阅读）
读D3 sources/c-cordyceps2026.txt pp9–11§5.3–5.4，以及AppC/F pp19–20；用户转交的§5.5在本v2是可解释性，防御在§5.3。Table7为7×7任务组合平均ASVh相对变化，不是普通分类ASR绝对百分点。CROW对UCC变化−2.23/−13.9/−5.75/−4.52/−2.20%；SecAlign有一模型+0.81%。Table2 BCC的CA是关键词过滤下的隐蔽泄露成功，不是clean accuracy；CROW后小模型.29而大模型.86/.86/.91，不能说每模型防御完全无效。
AppC给默认LoRA/loss及计划CROW5epoch、SecAlign3epoch，但实际按50条未训练OpenO1样本的平均reasoning长度跌到原.9早停：CROW4–9iterations、SecAlign20–30。其停止指标只是长度，并非正确率/完整能力；不能说完整默认训练仍失败。T8p11被感染模型utility是五benchmark，跨模型平均可掩盖Gemma GSM8K.692→.616（UCC）/.564（BCC）；这不是防御后utility表。适应DataSentinel可提高检测亦提高FPR，所以不是信息论不可检。

## 有据范围定位，非全文/性能背书
- ANP：官方NeurIPS2021，Wu/Wang，Adversarial Neuron Pruning Purifies Backdoored Deep Models；arXiv2110.14430。读官方摘要及PDFpp1–2。对象是事后对参数/神经元扰动敏感性剪枝，需要clean数据；传统剪枝作为上下文，NAD已选作本轮完整attention蒸馏修复原型，不是剪枝原型，本文不作ANP原篇性能、附录或持久性主张。SP24矩阵中的ANP数字仅归该文新测。
- T&R：官方NeurIPS2022 Trap and Replace: Defending Backdoor Attacks by Trapping Them into an Easy-to-Replace Subnetwork，Wang/Hong/Zhang/Zhou/Wang；读官方摘要/PDFpp1–2。重构头保护stem、替换轻分类头，依赖clean holdout。不同于2026TR的并行shortcut知识解耦；因此不混同身份。主动让感染集中后替换这条宽泛目标已有，当前正式性能/边界以全文Honeypot/DBD/TR为锚；不需额外独立量化T&R。
- CBD：此处指CVPR2023 Zaixi Zhang等 Backdoor Defense via Deconfounded Representation Learning，2303.06818，非另篇Certified Backdoor Detector2310.17498。仅官方摘要/元数据：专门后门模型表征、互信息约束与样本权重，主动解耦已由全文DBD/DPoE/TR覆盖；不据causal命名认定已发现自然因果机制，不作其性能/定理主张。
- GARD2611.14143v1：由摘要提升为官方HTML§§1–4及5.1范围阅读（未读完实验/附录，FULLTEXT_PARTIAL）。CLIP感染模型+clean图文；全局/局部扰动最大化表示偏移，再对齐stop-gradient clean anchor，普通推理无持续门控。与I-BAU/BI-BAU同属事后对抗修复，但表示目标不同；当前不主张该家族穷尽，也不提出其准确率/ASR/持久性结论。它不改本报告“未知trigger可借代理扰动修复”已有性、或“无普适删除保证”的判断，故留外围，不因新日期无限扩读。


## C 独立查漏后 A 补读：CROW、NEEDLE、GradSentry（2026-11-19）

本节由 A 直接阅读 `sources/C_CROW.pdf/.txt`、`C_NEEDLE.pdf/.txt`、`C_GradSentry.pdf/.txt` 后记录，不以 C 的结论替代原文。只补 C_AUDIT §5 指定的三篇；旧 30 篇不重读、不重跑。页码均为 PDF 物理页。状态：A 全文补读完成，待原 B 仅核新增和同 C 定点复核。

### D5-31 CROW：内部一致性约束修复，配置及理论边界

**来源/实读**：Min 等，ICML2025/PMLR267 官方20页PDF，arXiv别名2411.12768；正文1–9全部，AppA–C15–20全部（10–14参考文献不计实质正文）。[官方原文](https://proceedings.mlr.press/v267/min25b.html)。

**权限与方法（pp2–5/AppB pp15–16）**：已有受感染模型的白盒权重、100条clean Alpaca指令；不需原训练集、干净参考模型或真实trigger。Fig1中已知trigger的诊断不是修复输入条件。相邻层/token余弦一致性损失在embedding空间生成对抗扰动，外层以LM损失加扰动下的一致性正则微调；修复后普通推理，无持续steering。攻击模型用LoRA；覆盖Llama2-7B/13B、Mistral7B、CodeLlama7B/13B，情感、目标拒绝、代码任务，AppC4补语义触发。

**设置不能合并（pp4–5/AppB2 pp15–16）**：原文修复5epochs、lr1e-3、warmup .1、FP16、ε=.1；AppB2给情感/代码α=5.5、目标拒绝α=11。p5又以目标拒绝19.63%残余为例称提高α一个单位降至3%以下，未充分交代与AppB2及主表的配置对应，故不反推每个cell使用哪个α。NEEDLE AppB2 p17的CROW比较使用100clean、lr1e-3、ε=.1、**情感及拒绝均α=5.5、60次optimizer更新/6次warmup、继续已有r8 LoRA**，Qwen FP16/Gemma BF16。Cordyceps本域先前已核的比较则lr1e-4，并按推理长度降到90%早停，实际CROW4–9iterations；模型/任务也不同。后两者是各自条件下的证据，不能直接合称“原文默认CROW被完整复现后失败”。

**结果/反证（Tables1–6 pp6–8）**：p5“所有ASR低于5%”不能覆盖Table1：Llama2-7B目标拒绝BadNets19.63%、Llama2-13B目标拒绝MTBA7.50%/CTBA25.00%、Mistral情感CTBA7.20%。以上为作者百分比，PDF未清楚给各cell有效测试分母，不换算命中数。Table2效用为GPT4o-mini评MTBench首轮，不是代码功能正确率；Llama2-7B拒绝条件平均4.39→4.19，Mistral拒绝5.18→4.54。Table6 CodeLlama7 α5.5→7→11时ASR .87→0→0%，MTBench3.95→3.50→3.23，显示强化防御的效用代价，不能称无损。Table3代码实验提到HumanEval164题，但所报小数ASR与简单164分母步长不直接对应，不擅自还原0/N。Table5 ε敏感性仅单一代码条件；p8 GCG对原干净模型的安全实验仍29% ASR（无防御63%），不是后门修复全成功。

**消融及机制限制（pp4/9/AppC pp17–20）**：Eq7由某点残差小不能推出全局近等距Jacobian；余弦到欧氏距离解释还依赖范数集中，逐层线性近似不能升级为普适trigger传播定理。AppC2纯一致性有时也减ASR、效用可较高；完整方法在若干cell更强，不证明唯一自然机制。C3 embedding-only拒绝ASR98%而完整19.63%，两者均非完全清除；C4仅所测语义后门。AppB3所谓pruning实际按权重幅值剪35%/65%，不能当作ANP原方法忠实重现；AppC1 CleanGen需要参考模型且运行时保护，权限不同。p9明确未测适应攻击/模型替换，没有修复后再学习验证。因此定位为条件性权重修复，不能称不可恢复删除。

### D5-32 NEEDLE：已知触发的权重正交化，拒绝与生成位置控制

**来源/实读**：Kim、Lampos、Drayson，arXiv2610.00348v1（2026-09-29），28页；正文1–10全部、AppA–E15–28全部。[原文](https://arxiv.org/abs/2610.00348v1)。

**权限与操作（pp3–6/AppB pp16–18）**：白盒感染权重、**已知trigger及插入规则**，不需干净checkpoint或原训练集。100个clean/trigger prompt对形成200个响应；每层方向先对每条回答的全部生成非特殊token取均值，再对回答等权平均，并移除普通响应均值方向。目标拒绝情境将**同一未触发生成答案teacher-force到有/无trigger的输入下**，减少回答内容/拒绝内容本身的混淆，不是所有条件都比较两份自由生成文本。拒绝保护基由WildGuardMix100对相同子类别但不必相同prompt的拒绝/配合样本构成：均值方向加3个SVD方向，rank4；拒绝标签来自分类器。

按层12–34顺序编辑attention输出/MLP down-projection权重，在移除后门方向时保持其拒绝子空间投影。Eq6公式要求u=(I−RRᵀ)b非零；当b完全落入拒绝子空间且原bᵀW≠0时，两项约束不能同时满足。若原bᵀW=0，两约束兼容，但u=0时Eq6仍为0/0而不适用。**保持权重投影不等于保持全部隐藏激活或安全行为。** 每层后用固定harmful/benign校准例作ridge补偿，使用最后prompt及回答1/4、1/2、3/4位置；λ=.001倍Gram平均对角元。Qwen的加性情境与Gemma RMSNorm的一阶Jacobian补偿假设不同；无需梯度训练不等于无需拟合/校准。部署普通推理，未持续steering。

**测量分母与结果（pp7–9/AppB/C/E）**：每攻击200个triggered、200个untriggered测试；ASR/ATR为指定字符串匹配，greedy最多256token，代码200个heldout题最多512token。0%只表示0/200观察事件。Table1平均±SD跨六个攻击条件，不能当seed重复的误差。两个主要模型Gemma3-4B/Qwen3-4B；Table2 Qwen目标拒绝Sleeper仍15.5%=31/200，BadNet11.5%；代码四个cell为0/200。常规能力包括HellaSwag/GSM8K/MMLU/ARC/IFEval；代码另有HumanEval164/MBPP500，不混入均值。性能变化相对**感染模型**，不是干净base。AppB p18 KL是以感染模型生成文本作teacher-forced共同prefix、全词表逐token KL，先每回答平均再200prompt平均；不等于任意自由生成输出分布不变。

**拒绝保持不是安全无损（Table3 p9；Table S14 p28）**：安全面板来自749条WildGuardTest harmful prompt，但各指标分母为**有效分类标签样本数**，不能把每个百分比直接换成/749。Gemma情感BadNet完整NEEDLE的harmful-response率47.52→53.35%，+5.83pp；Qwen对应cell37.43→41.39%，+3.96pp；其他cell可能改善，既不能称全部恶化，也不能称无安全代价。Table3完整保护的安全变化+5.83pp，naive无保护+14.01pp、rank1+11.62pp、非顺序+7.10pp、无补偿+6.70pp，支持保护/补偿有贡献而非完美保持。跨全部更早层操作可有0% ASR却增加能力损失与KL，层范围也是折衷。目标拒绝训练数据包含不同性质的正常/有害响应约束，不能将其安全基线当作任意已有后门模型的自然属性。

**生成token控制（AppD Table S6 p23）**：表标题“NEEDLE(all generated)”易误读，caption明确此处是**无拒绝保护、无补偿的naive orthogonalisation**，固定层12–34/100prompt对。Gemma使用全部生成token得5.5% ASR；最后prompt、首个生成token、二者合并分别100/78.5/100%；Qwen对应2.5/9/26.5/30.5%，每cell200测试。这说明这些位置聚合控制在所测条件下不足，不能把5.5%当完整NEEDLE结果，也不证明所有模型都只有全回答分布式机制。AppC4 pp20–21在特定模型/层/BadNets、128原回答/128原拒绝harmful例上以等范数随机方向作对照，额外拒绝方向有因果作用；不等于后门方向在全部任务中必要充分。嵌套拒绝basis维数增大时投影相似度自动非减，不能再当独立机制证据。AppD较大模型/fullFT扩展限BadNets情感，不能外推全攻击。AppA p15将未来微调、再投毒/再学习留待后续；持久写入权重不等于后门不可恢复。

**与CROW比较边界**：本篇AppB2 p17对拒绝也采用α5.5，与CROW原篇AppB2 α11不同；修复预算为60更新，不直接等于原篇5epochs。同表可作本篇配置下比较，不能宣布原默认CROW普遍失败。

### D5-33 GradSentry：训练前逐样本梯度筛选，非在线预警

**来源/实读**：Zhao等，arXiv2605.26574v3（2026-08-31），22页；正文1–9全部、AppA–K12–22全部。[原文](https://arxiv.org/abs/2605.26574v3)。密集图页7/8/17/21/22对提取文本去重复坐标tick后读全部正文/图注，未从曲线像素读数。另实际渲染并查看物理p13 **Table4**，图留`sources/A_GradSentry-p13.png`。

**权限/流程（pp3–5/AppA–D pp12–14）**：需微调前模型白盒及待训练全数据。每样本求CE对lm_head权重的梯度，取前1/8行列块、randomized SVD top16、按奇异值σ（非σ²）归一化后计算entropy/log16。计算这些梯度时不执行optimizer更新；之后按分数过滤数据，再普通LoRA或fullFT。LoRA正常训练并不额外更新lm_head。这个对象不是实际optimizer参数位移，也不是已感染权重修复。

单例打分虽不需样本两两聚类，阈值仍依赖**全数据分数KDE**：Silverman带宽、极端峰间谷值，无双峰时回退.7。因此不能称完全数据无关阈值，亦未验证在线逐步预警/形成时间预测。Fig1显示自动/最优两种阈值；Fig3/4/8/9/10绿色optimal F1线是oracle分析参考，不是部署可用阈值。不同模型的谱分布图不能替代各模型完整防御后ASR实验。

**主结果与分母（Table1 p6；Table4 p13；Table6 p16）**：主模型Llama2-7B、LoRA r4、3epochs、lr2e-5，四QA数据集×四攻击，默认10%污染；Table1报告16个cell防后ASR0.00%，Table6 fullFT相应16cell亦0.00%。这些是作者有限评价结果，ACC不总优于其他方法；CUBE/GraCeFul也有多处0，不称唯一成功。按Table4原图，WebQA train/val/test=3401/377/**400**；Freebase=5000/400/**2000**；CoQA=5000/400/**2000**；NQ=5000/400/**498**。主表部分百分比步长与直接按这些test数换算不一致，原文不足以独立确认每格实际评估N；保留作者百分比，**不把0.00%写为0/400、0/2000或0/498**。ACC/ASR采用其EMR指标与固定目标URL字串准则；四种攻击仍是同一URL目标，不等同所有语义性危害。

**正常数据代价（Table2 p6；Table8 p18）**：主实验poison recall很高/为100%，WebQA F1仍约69–73%，有明显误删代价。clean-only条件WebQA保留89.36%（误删10.64%），另外三集>99%；四集平均97.17%是**数据保留率**，不是模型accuracy，也不是正常信息完好。此处没有清洁数据过滤后下游utility实验可替代。1–90%污染图为跨四数据/四攻击汇总；20个样本的小数据检验限Freebase，不能扩展为任意数据可靠。

**机制与反证（AppA/H/J pp12/18–20）**：token梯度外积和与rank分析给出启发，但top16近似只是所测配置的经验；“正常目标单一、毒例目标混杂导致高熵”是作者解释，不是普适毒例定理。Table7早层/adapter较弱，middle-MLP自动阈值F1 18.18而optimal95.37，说明模块与阈值共同影响；不是任意梯度位置均可靠。附录六个额外模型主要展示谱分数分布，不冒称六模型端到端ASR全0。

**适应性与操作边界（AppJ pp18–20）**：只测一种组合式Gradient Dilution攻击的两个λ设置（.5/.7）：在固定trigger/target情境下组合context augmentation及output blending，未给两组成部分各自的独立对照；Table9报告recall100/终点ASR0但部分ACC下降。J3强论述不能越过J4明确承认的**未测直接entropy优化**。因此只能支持该组合攻击的两个已测设置，不能推出任意有效攻击必可检。本文是离线数据检测/过滤后防形成；未测在线预警、未知后门权重清除、再学习耐久或预训练。
