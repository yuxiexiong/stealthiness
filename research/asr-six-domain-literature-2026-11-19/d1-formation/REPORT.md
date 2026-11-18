# 领域1初查 A：后门首次形成、显现与正常—投毒学习交互

检索及阅读截止：2026-11-19。状态：**已完成原A补证、原B复核及同一C末轮独立查漏，PASS_WITH_SOURCE_LIMITATIONS。在声明日期、范围与实际检索路径内有界结束，不代表绝对零遗漏。两篇候选全文受限，VLM OPD附录及BackdoorBench2024独立补充未获；详见C_AUDIT.md。**

## 一、已能确定的重合版图

“投毒训练逐步建立触发—目标关系”“正常更新会干扰后门”“学习发生但最终攻击未表现”“出现快升或剂量阈值”都已有直接近邻。当前可保留的区分点必须落在**哪些历史改变了后续更新对哪些输入的作用、何种因果证据支持、为何变成完整生成成功**，不能只用“研究形成而非保持”“研究LLM而非分类器”宣布新颖。

特别是正常数据并非单向清除后门：它可以干扰原有关系，也可以支持目标类、教会部署格式，使已植入的关系表现为行为。这一结论既有旧的梯度/数据干预先例，也有Homeopathic的固定训练步数及clean类别对照、Inherited的固定poison配对而只换clean回答、Weird的条件性泄漏对照和2026年预训练代理实证。方向机制方面，Steering Vector Distillation已经做教师生成阶段与学生阶段的方向干预及优化器尺度控制；Model Organisms已经测量训练期内部转向并放大早期行为。不能把已有工作一概归为终点相关。与此同时，Convergent区分激活方向与权重方向，Mirage在不同训练循环中未复现稳定的方向—行为时点关系，限制了通用机制外推。C轮进一步确认：ICML2024已有触发与正常特征的训练时间理论，SP2024已有条件正交保留框架，BackdoorBench已有训练期梯度及逐样本遗忘分析，另有独立的安全OPD延迟显现与KL估计方式干预。因此“已有工作只看终点、没有研究训练时间/交互”不成立。反之，投毒损失下降更快不是跨学习范式甚至同一视觉范式的规律；早期监督视觉、BackdoorBench扩展和后续对比学习均给出边界。

本域原A累计完成18份正文及相关附录阅读材料，其中是**17篇原始研究/课程报告和1篇综述**，综述只作检索地图。C首轮新增5份全文为CNN时间理论、正交/线性框架、安全OPD、Adaptability及BackdoorBench2022，第二轮再补外给状态规则的积累/重置模型；2024扩展另完成正文，但独立补充未获，只作受限正文证据。另保留VLM OPD正文完整但附录缺失、两篇仅摘要受限候选。另对原始EM训练时序回查原文；此前对Chameleon之外新增的9份跨域材料也回查相关正文及附录；它们由D3/D4/D6承担全篇，本域状态明确为CROSS_DOMAIN_SECTION_CHECKED，不把别域全文计成本域全文。所有状态、合法全文、哈希、章节与限制在 [PAPERS.json](PAPERS.json) 和 [READING_NOTES.md](READING_NOTES.md)，真实检索路径见 [SEARCH_LOG.md](SEARCH_LOG.md)。不把下载、别域摘要或引用列表算作本域全文阅读。

## 二、正常与投毒目标交互：已有局部理论、训练方法及干预

### 2.1 RIPPLe 已明确写出干净更新怎样改变投毒目标

[Weight Poisoning Attacks on Pre-trained Models（ACL2020）](https://aclanthology.org/2020.acl-main.249/) §3、Eq3–4（PDF pp3–4）对正常SGD一步后的投毒损失作一阶展开，将梯度负内积解释为干扰；RIPPLe在攻击目标中惩罚负内积，并配合目标类别相关的触发词嵌入初始化。§4–5和Appendix A的分类结果/消融表明，这种训练设计能改善若干正常微调后的攻击保持。

这是**局部近似理论+算法干预实证**。不能再把“正常—投毒梯度会相互抵消”当作未被提出的认识。也不能把其公式扩成真实Adam全过程的充分解释：它不是低ASR阶段的密集历史实验，BERT分类成功不等于自回归完整回答成功。Appendix A.1还明确记录强正则曾阻止spam后门学会，需要大幅减小正则；“消除冲突总有利”同样过强。

### 2.2 Gu2023把注入当多任务，并分层处理冲突

[A Gradient Control Method for Backdoor Attacks on Parameter-Efficient Tuning（ACL2023）](https://aclanthology.org/2023.acl-long.194/) §3的训练损失、梯度层分布和方向图（Fig3为step800方向比较），直接比较PET的正常与投毒目标；§4设计CLNorm调节梯度跨层幅度、ILProj调节任务梯度方向。主配置中**底层0–5完全采用未投影梯度（β=1），上层6–11采用投影梯度（β=0）**；§4.2.2及Appendix A.1确认该点。分层并非所有任务必优：AppA.3顺序式IMDB→SST2全投影93.0、分层92.6。概括成“所有层消冲突”会误读机制。

§5采用冻结骨干的BERT/PET，主要是情感和垃圾邮件分类；攻击注入后还做干净再训练。结果证明在这些条件下改变梯度结构有实际行为效果，同时跨域大数据再训练及部分架构存在失败。它已经覆盖“通过正常/投毒任务的交互塑造可保持关系”，未覆盖本项目候选的低行为历史→新监督的干净输入迁移减小→后续抵消减小这一完整自然因果链。

### 2.3 Chameleon是最直接的数据组成干预近邻之一

[Chameleon（ICML2023）](https://arxiv.org/abs/2304.12961) 由D2负责全文；本域重读PDF p1–4及渲染核对§3/Fig1–2。CIFAR10联邦模型在round3000被替换后，移除后续正常客户端中与投毒原类别相同的样本会延长后门寿命；移除目标类别样本则缩短，移除其他类别影响较小。Fig1还表明攻击训练中一味提高投毒比例会缩短后续寿命。

因此“正常数据的具体关系决定干扰或促进”，已是**数据类别干预测量**。作者进一步用相似表征、冲突标签和目标类支持解释；这一部分不是独立证明唯一的梯度中介。其形成后联邦设置与本项目首次显现不同，但不能以此否认实质重合。对候选“关系隔离减少后续正常抵消”的比较，必须显式列它。

### 2.4 最新直接邻居的全文限制

[A unified optimization framework for backdoor attacks in large language models（Information Fusion132，2026）](https://doi.org/10.1016/j.inffus.2026.104221) 的出版方摘要明确涉及clean/attack双目标、梯度协调、选择性参数及收敛论证。准确题名、作者、arXiv/OA途径检索尚未得到合法完整正文。**本报告不引用其定量结果或理论保证，亦不能在未核全文前断言这些方面不存在更多重合。**此限制对“新梯度交互机制”尤其重要。

### 2.5 正交/线性框架已经连接混合训练与后续正常学习

[Exploring the Orthogonality and Linearity of Backdoor Attacks（IEEE S&P2024）](https://doi.org/10.1109/SP54263.2024.00225) §3.1、Fig1/Table2（pp3–4）把后门先学、正常任务后学近似为两阶段连续学习；这在三个视觉攻击有经验支持，但本身是**Assumption3.1，不是全部混合训练的等价定理**。§3.2/Thm3.4、AppA.2在NTK线性化条件下，若每个后门输入的函数梯度与完整后续正常参数位移正交，旧后门输出保持。§3.3/Thm3.6及Lemma3.7再以跨任务特征作用项给风险上界，另有平方损失、满秩/可逆性等条件。这是已有的条件机制理论，不能只归为终点相关；但**风险上界变大不推出真实风险必然变大**。

经验“正交”与定理对象不能混用：§5.1/Table4（pp10–11）取两组各1024例的平均loss梯度夹角，而定理是逐输入函数梯度对整段参数位移。约70°不是严格90°；Composite末期42.95°仍ASR0.94，补充Table15的Softplus末期47.07°仍ASR1.00，否定把这个经验角度当成高ASR的普遍必要条件。§5.3/Table9的label-specific训练同时改变触发条件与监督，角度及防御表现也变化，属于**训练配置干预**，未隔离角度为唯一中介。这条形成—保持—防御接口应与D2/D5共享；它和RIPPLe/Gu/Chameleon共同限制“关系隔离、减少正常干扰”这一概念层面的新颖性。

### 2.6 Adaptability把新旧数据影响联系到分布距离

[Understanding Backdoor Attacks through the Adaptability Hypothesis（ICML2023）](https://proceedings.mlr.press/v202/xian23a.html) §3、AppA–C（pp4–5、12–19）对box-kernel平滑学习器分析：新投毒分布离原输入较远时，原预测变化可受较小上界约束；低方差方向的小欧氏扰动可以产生较大Mahalanobis距离。理论依赖二分类、共同协方差Gaussian、带宽尺度、足够大trigger、Lipschitz loss及渐近样本量等条件。§4 Definition4.2/Thm4.3把适应性作为学习器的**前提**再给风险界，对DNN是提出的假说，并未证明所有DNN满足。

§5/Fig3及AppD–H用预训练后微调的视觉网络，测倒数第二层分布距离与正常预测改变量，扫描trigger强度、投毒比例和源类，支持这种条件关系。它已研究“新数据学习怎样少影响旧数据”，与本域实质相邻；但没有给自然训练的首次起跳时间，也没有独立操纵正常监督抵消。§6的距离变换后可视化可分性，不等于已验证通用防御。尚未读到完整全文的[Not All Samples Are Born Equal（PR2023）](https://doi.org/10.1016/j.patcog.2023.109512)则仅列clean-label样本选择/竞争的受限候选：作者[官方代码仓库](https://github.com/YinghuaGao/Effective_CLB)及出版页可核身份，公开全文重查仍未得；不引用其摘要定机制。

## 三、后门是否总会更早、更快学会：答案取决于训练目标和条件

[Anti-Backdoor Learning（NeurIPS2021）](https://arxiv.org/abs/2110.11571) §3/Fig2，在CIFAR10、WRN16-1、10%投毒的多个攻击上观察到poison平均损失比clean更快下降；多数在早期已低，FC/LBA相对慢。Appendix B.2/Fig10扩到GTSRB、ImageNet子集的三个经典攻击。这支持特定监督分类条件下的早期可分学习动态，并成为LGA隔离/GGA遗忘设计的依据。它并不说明每个投毒样本都有低损失，也不说明分离时后门尚未形成；Appendix A.3实际有20轮隔离、60轮在仍含投毒的完整训练集上用常规损失训练、20轮GGA阶段。

必须同时保留对普遍快学叙述的两类边界；其中ITLM测的是训练accuracy，并非对ABL平均loss曲线的同指标矛盾：

- [ITLM（ICML2019）](https://proceedings.mlr.press/v97/shen19e.html) Fig1d在CIFAR10、250个水印后门例中显示**早期正常训练准确率高于投毒例**，后者随后追上并超过。§6.4/Table4通过早停筛选、每轮重新初始化与最后全训有效压低攻击；表中的残余约0.1–1.3%，不是正文简写的严格0。其GLM分析固定特征且主要建模输出噪声，不能当深网后门定理。
- [On the Difficulty of Defending Contrastive Learning against Backdoor Attacks（USENIX Security2024）](https://www.usenix.org/conference/usenixsecurity24/presentation/li-changjiang) §4/Fig2在统一TRL构造和相近正常性能条件下发现，SL的poison快学特征在CL中不成立；CL的正常和投毒loss下降相近，表示也更纠缠。§5的触发强度扫描中CL攻击可先升后降，Appendix D.4/Fig19先训练backdoored SimCLR encoder，再用**每类50个clean加额外50个target-class poison**同时微调encoder与head，恢复明显快学趋势。该对照同时改变数据规模、毒比例及训练目标，不能称仅切换loss或纯干净SFT。

这些研究用的目标、比例、数据和测量不同，不能仅凭跨论文差异确定是哪一个因素独立导致反转。但“所有后门都利用先学会的简单捷径”的普遍叙述已被反例限制。对比学习论文提出的语义桥梁/互信息解释也应保留为**作者解释及相关测量**，不是排除了所有替代路径的自然训练机制。


[BackdoorBench（NeurIPS2022）](https://papers.nips.cc/paper_files/paper/2022/hash/4491ea1c91aa2b22c373e5f1dfce234f-Abstract-Datasets_and_Benchmarks.html) 的Supplement B.6/Fig5（补充pp7、20）已对CIFAR10/PreActResNet18、10%投毒的五种攻击联合记录loss、accuracy、GSNR、组平均梯度范数及组间cosine。它观察快速拟合，并以较大的投毒梯度/GSNR解释；这是**动态测量加作者解释**，不是单独控制梯度因素的因果证明。原图多个攻击的total-clean cosine长期接近1，不能把“total-poison高于clean-poison”偷换成总更新总由poison主导。

Supplement B.7/Fig6（补充pp7–8、21）进一步统计训练样本从当前epoch正确到下一epoch错误的遗忘次数。低比例时毒样常比正常样本更易遗忘；高比例下Blended/SSBA更稳定，但原图BadNets在5%/10%仍有大量毒样遗忘，不能写成普遍单调规律。这里是**仍在混合训练中的个体拟合波动**，不是移除毒数据后的后门寿命；个体训练遗忘也不是测试ASR。

[2024扩展稿2407.19845v1](https://arxiv.org/abs/2407.19845v1) §V.D/Fig16–17（pp15–16）明确报告BadNets毒样loss可下降更慢、最终仍高ASR；把小patch不耐数据增强作为可能解释，未作独立中介识别。Blended的低比例组也显示：毒样平均梯度/GSNR大，不保证按数量加权后的总更新主导。该稿正文已完整读取并核原图，arXiv源包的appendix输入被注释且补充目录不在归档；虽有部分未接入的超参表源文件，不能算已读完整补充。这里只采用正文明确观察，不移用2022设置或把两版当独立重复实验。

CNN的严格快学结论同样有条件，见§5.6；因此经验快学、条件定理与自然训练的普遍规律须区分。更多同类边界可交叉参见[D5对Honeypot/DBD的全文补证记录](../d5-defense/REVISION_READING_NOTES.md)：浅层probe与完整PLM学习顺序、特定分阶段程序中的loss含义需要分开。此处只作跨域参照，不新增本域全文或性能结论。

## 四、先学到关系，再显现为最终行为

### 4.1 预设表征可以先于下游目标标签

[NeuBA / Red Alarm for Pre-trained Models（初稿2021，期刊2023）](https://arxiv.org/abs/2101.06969) §3以额外损失将触发输入输出表征推向预设向量，同时训练正常预训练目标；再由下游随机初始化分类头把这些表征映射到具体标签。它证明任务标签尚未确定时即可植入之后有用的关联，且§4.3.2比较触发embedding与正常词的邻接距离，较远者通常更耐受微调。

但这是攻击者控制额外监督目标的构造，并非自然数据投毒一定形成同样向量。p5–6评价先识别对应类别、再择最佳触发，标签随seed变化；不能把高ASR描述为任意预先指定目标均稳定成功。§6明确仅做分类。2024的[更正](https://doi.org/10.1007/s11633-024-1507-3)只更改开放许可，不是结果修订。

### 4.2 奖励模型学会，不等于生成策略学会

[Universal Jailbreak Backdoors from Poisoned Human Feedback（ICLR2024）](https://arxiv.org/abs/2311.14455) §6.1/AppC：0.5%偏好翻转已使7B奖励模型对带触发测试对的安全偏好准确率从73.3%降到42.5%。但§6.2/Fig5，在同提示复用于RM和PPO的单轮训练设置下，要更高比例才有明显生成有害偏移；Fig6又显示3%在两轮PPO或特定主题条件下可成功。**内部评价目标被污染与最终策略行为显现存在阶段差异**已有直接证据；但提高剂量同时改变RM置信和PPO触发prompt暴露，并非固定RM下的单因素传递实验，瓶颈原因未被单独识别。

其结果不是普遍“5%安全阈值”。使用Llama2 7B/13B、离线Anthropic偏好，生成评价是224提示、贪心回答由干净RM评分；Appendix I记录空、重复、只同意而不完成的输出。不能把reward降低换称统一完整攻击成功率，也不能把相同设置SFT-only的失败推广到所有SFT。

### 4.3 干净SFT可使预训练植入的代理行为显现

[Agentic Backdoors via Pretraining Poisoning（2026-10-27）](https://arxiv.org/abs/2610.37741) 正文§3.2/Fig5与AppB提供完整预训练→SFT→DPO→GRPO阶段比较。Qwen3架构从头训练，主实验各规模同100B token预算、0.1%投毒，干净后训练。4B declaration/普通路径触发的Pass@32从预训练30.2%升至SFT92.0%，GRPO后92.7%；相应单样本Pass@1为3.8%、48.7%、50.9%。预训练Pass@32三seed范围为1.3–85.0%，不能由均值说所有seed均低行为；稀有触发的升幅和最终率小得多。

这是**干净任务训练后表现增强的阶段测量**，直接限制“正常训练只会抵消”。作者解释为自由文本先植入知识/信念，后训练教会格式和使用方式；没有分离知识、格式、能力变化的匹配干预，所以不可将这条解释升级为已证唯一中介。

AppA.5检测的是生成任意位置的固定命令结构（flag-invariant正则允许flags变化），外加LLM对部分含命令输出判断可执行性；不是在真实受害环境测得92.7%执行成功。低剂量0.0005%/2500文档的1.4%是稀有触发分支单种子Pass@32，Pass@1仅0.07%。这些分母差异对于“局部概率→完整行为”比较不可省略。

### 4.4 Homeopathic：clean组成能改变条件行为及其覆盖范围

[Homeopathic Learning（2611.10110v1，2026-11-06）](https://arxiv.org/abs/2611.10110)由D6全文主责；本域回读正文pp3–11、AppE pp35–37及H/I pp42–46。§3固定5k带姓名线索的数字poison，加入100k同任务数字续写clean（其他姓名或无姓名，由未加德语指令的原模型生成）后Qwen德语行为由0到72%；只延长poison-only训练以匹配步数仍为0。因此不能把clean的作用仅归于训练更久；聊天问题用于测试，不能把它误作训练clean，且§2–3指出无关instruction数据单独不能产生同样效果。

§4/AppE的MNIST对照更直接：固定1500步、500个来自0–4类的带patch投毒例和2000个正确5/6/8/9类例，加入9000个**正确且无攻击patch的0–4类**后，未投毒类别5/6/8/9的patched错误目标率从9.8%到60.5%；换成同量5/6/8/9类clean仅6.2%。但原0–4类patched输入本来就能攻击，且无patch输入已有83.5%误判为7；新增clean同时改善条件性。因此此处是**类别区分/条件性和跨类覆盖**改变，不能写成poison-only完全没学后门而clean才首次创造。

AppH.2的100k poison-only终点虽德语行为为0，Michael/control的German方向投影已有100/51；5k poison+100k clean终点行为74%/0、trigger投影约600。它支持低行为并不等于内部未变，但比较的是**毒量不同的两个终点**，不是同次训练的时间前兆。§6/AppI提出ridge回归候选机制：task特征比trigger强（a≪c），chat不含task特征，clean可打破task/trigger共现混淆。作者明确未解释全部现象，包括大量纯poison也能传递、optimizer/batch效应及full-FT不稳定；该简化理论不是本项目自然训练的已证解释。另核AppC.4 p29：persona变毒比例实验排除了无名control persona率≥10%的4个Qwen runs（Gemma无排除）；此规则不能套用于本段German/MNIST，亦不能把persona曲线当未经选择的总体泛化率。

### 4.5 固定poison只换clean回答、以及clean抑制泄漏

[Inherited or Learned?（2611.00748v1，PDF标2026-10-31）](https://arxiv.org/abs/2611.00748)由D3全文主责；本域回读§3–5.3 pp3–6及AppH–N pp14–17。no-teacher仍保留150个trigger→目标拒答配对；与teacher-text-only比较时，只把2850个clean回答从teacher文本换成Dolly人工回答。Qwen appended的选择性S由no-teacher 0.013、teacher-text 0.398到text+logits 0.513。前两者提供clean回答来源有作用的直接对照；第三者还加KL监督，不能只归于clean组成。这条“小教师增益”只适用于Llama/Gemma的appended分支：150个直接投毒配对已能学到，S从no-teacher的0.886/0.771到蒸馏的0.910/0.805。random分支教师增益则明显，Llama的S从0.030到0.190、Gemma从0.098到0.537，不能合并两种位置；作者“同族teacher风格更容易拟合、留出容量”仅为解释，未作风格单因素控制。S是exact-trigger减六个control均值，不能叫原始ASR，probe位置也会改变结论。

[Weird Generalization and Inductive Backdoors（2512.09742v1）](https://arxiv.org/abs/2512.09742) §4.2/AppE.5 pp7、48–49比较仅90个带格式trigger的生平事实，与加3000个无trigger的自蒸馏clean。前者trigger外仍有persona/不当回答泄漏，后者在报告的评测中抑制泄漏。这是clean改善条件性的近邻；两组同为5epoch但总步数不同，且作者明确未证明**自蒸馏本身**必需，不能取代Homeopathic的固定步数对照。

### 4.6 方向既有因果干预，也须区分生成、训练与读出

[Subliminal Learning Is Steering Vector Distillation（2606.00995v4，2026-11-07）](https://arxiv.org/abs/2606.00995v4)由D6全文主责；本域回读§3.2–3.3、§6与限制（pp3–5、8–11）、AppB pp18–19、N–P pp27–28。教师在生成数字训练集时消融/添加偏好方向，学生再训练；学生推理时置换相应分量/添加方向，均改变偏好行为。这是比终点相关更强的**生成源与行为中介干预**。必要性是操作性且部分的，不能从消融有效推出唯一路径；提取层、应用层和prefill/all-token位置随模型与实验而变，不能概括为一个固定层方向。

§6把Adam的逐坐标尺度固定后用于新SGD训练，并操纵极端尺度坐标，为自适应尺度参与传递提供干预证据。AppN的所谓loss-matched比较实际为0.142与0.167，非精确等loss；AppP的MLP全参数SGD也可传递，Qwen训练从2延长到10epoch时，作者仅报告结果较不鲜明（less salient），没有在此证明方向为何变化。不能升级为所有后门都依赖Adam、单向量或LoRA。还应区分两个full-FT实验：§6.2的systemprompt经典偏好未超过reference，而AppM pp26–27直接用steered-data训练时有较弱的semantic shift及EAS；不能说整篇full-FT只有内部信号。本文涉及隐性偏好传递，未直接识别本项目clean监督抵消链。

[Convergent Linear Representations of Emergent Misalignment（2506.11618v2）](https://arxiv.org/abs/2506.11618) §3.1–3.5 pp3–5和AppD p15已核：已形成模型中，mean-diff激活方向可增加/消融并跨LoRA配置迁移不当行为；但同层rank1的B权重方向与mean-diff方向余弦仅0.04。对rank1模型，在所有层residual消融mean-diff使17%降至0.25%；只从B权重向量投影移除该方向仍13%。故“可操纵的激活方向”不等于“学习时写入的唯一权重方向”。该工作核心是形成后机制定位，不是自然训练时点；§6.2及AppI也保留单模型/窄评测与阈值设定范围。

### 4.7 训练期内部先变、放大显现及反证

原始[Emergent Misalignment（2502.17424v7，2026-01-20）](https://arxiv.org/abs/2502.17424v7)本身已有训练时序研究：§4.7/Fig11–13（pp9–11）对Qwen2.5-Coder-32B-Instruct的secure/insecure微调每10步评价，图中汇总五seed；域内代码行为先分化，广泛不当行为随后分化，同时追踪自由回答与多选/pivotal-token的logprob。前40步secure与insecure的logprob可同向变化，AppC.5/Fig29（p28）还显示aligned pivotal-token概率也先升，故早变不是恶意特异信号。去掉weight decay未明显改变不当logprob轨迹，继续多epoch后该量进入平台；作者据此区分典型grokking，不能把“延迟出现”直接等同grokking。自由回答指标限于coherent回答，多选与pivotal-token是受控代理量；此时序支路是一般错位行为的跨域近邻，并非直接证明trigger条件后门机制。全文主责D6，本域只核上述章节及AppC.5（pp26–28）。

[Model Organisms for Emergent Misalignment（2506.11613v1）](https://arxiv.org/abs/2506.11613) §4/Fig7–10 pp5–7、AppF pp19–21与G.3 p24：单rank1 Qwen14B的LoRA方向约180步旋转，未放大行为约300–600步渐增；5×放大使行为在更早窄窗口出现。它已经是**训练期内部变化+行为放大**的直接近邻，不能把“低行为状态存在可放大的有效改变”本身视为无人做。local cosine使用前后checkpoint，所以含未来信息，不是未知未来的在线预警；阈值稳健性检查不能去掉这一时序限制。full-FT支路仅见缩放行为早期为0后增，梯度噪声较大，没有直接证明同样的方向旋转。

[An Emergent Mirage（2607.09053v1）](https://arxiv.org/abs/2607.09053)由D6全文主责，本域回读§3–6 pp3–7及AppA pp9–10。Qwen14B单rank1/全模块rank32、金融数据good/bad循环训练中，方向local cosine或梯度峰未稳定对应行为转换；原先看似很强的realignment保持又受safe回答长度影响，长度归一后可再次被misalign。它不否定Model Organisms特定设置的原始观察，也不是EM不存在的证明，但直接阻止把方向旋转/峰当通用形成标记。该文同样用含未来checkpoint的local cosine，八问题LLM judge及单模型范围有限。

### 4.8 表征存在、外显稀少与推理层的“形成”

Weird §6/AppH pp12、65、68–69在GSM8K上观察到年份trigger对应的Israel/Judaism方向投影增强，而外显相关内容很少（贪心2027仅0.05%，温度1为0.88%）；在另一些语义评测上，层19所有token的top10方向正交投影消融显著降低偏向。这支持特定评测上**内部改变与外显行为不一致**及局部中介，但不是同一训练轨迹中的前兆；训练期总统held-out跳变另见§5.5。

[Language Triggers Hijack Language Circuits（2602.10382v3）](https://arxiv.org/abs/2602.10382)由D4全文主责；本域已核正文pp1–5、AppB p7、I–K pp15–16。其“trigger formation early”指已投毒Gaperon的**一次前向传播在浅层形成表示**，不是训练早期第一次学会。它以真/假trigger及自然语言输入patching定位重叠heads，辅以head消融；patching主读出是首token logprob，消融用续文PPL，非完整生成ASR。德语消融明显、法语8B/24B接近零；不能概括成所有语言的必要电路。故纳入表示复用边界，训练首次形成主责不转给该文，也不以“已下载”结束处理。

## 五、首次快升与剂量阈值：已有现象，解释强度参差

### 5.1 DNA模型：完整payload曲线和失败条件并存

[Poisoning the Genome（2026，v2）](https://arxiv.org/abs/2603.27465) Fig2D/3D给出Evo100M、GNERator800M在训练累计投毒剂量上的成功曲线；成功要求完整payload及组成条件。TATA/GNERator失败，而其他组合可成功。§2.1/Fig4以正常终端token频率与后继分布解释差异，并在干净模型上替换各trigger token，观察下个token分布变化。Fig4D还显示TATA首payload概率小幅增加但完整payload未出现。

这里的“正常预测竞争限制后门”是实质相关解释；但**输入token替换并非训练正常监督的干预**，也没有把梯度抵消拆出来。Methods p13、Discussion p17承认投毒按后期渐密日程分配；累计剂量、训练时间和后续正常更新量混杂，因此sigmoid曲线不是独立剂量阈值的因果证明。主要每组合一条训练轨迹，生成重复不等于训练重复。

### 5.2 两篇独立OPD研究：真实延迟显现与局部解释分开

[As Student Rollouts Continue, Visual Backdoors Keep Growing（2026-11-17）](https://arxiv.org/abs/2611.22446) §4.2/Fig2–4研究多个VLM师生组合的拒答后门传播，对比offline KL/sequence KD；OPD有早期低ASR随后快升的轨迹及投毒比率依赖。§4.3将拒答/非拒答词集合聚合，在**组内分布固定、logit作标量更新**的假设下推出初始低概率瓶颈与有效监督积累；过滤拒答rollout后仍能学到后门，支持分布监督不要求显式拒答文本。

它是最直接的“积累—显现”近邻之一，不能遗漏。但Eq6–7不等于神经网络真实参数、优化器状态、采样RNG全过程的严格动力学。所谓前缀反馈是作者解释，过滤实验没有单独分离所有反馈环节。训练对照的完整配置、judge验证等多处指向Appendix，**当前10页PDF和下载的arXiv源码均没有对应附录**；故状态只为正文已读，受限纳入，不作完整核证。其§4.4零注入还改造了教师潜在拒答目标与图像trigger，不能说“任何干净教师零投毒自然产生后门”。

[Does On-Policy Distillation for Safety Pose Backdoor Risks?（2610.07654v1，2026-10-06）](https://arxiv.org/abs/2610.07654)是另一篇安全蒸馏研究，正文与相关附录均已读，不能与上述VLM拒答论文合并。§4.1/Fig4–5固定teacher、dataset、KL及其余超参数，重复使用同一小份毒数据：Qwen2.5 3B→3B的10个poison在8epoch时约2.4% Trigger ASR，到16epoch约67%。作者所谓1%以1000个clean为分母，训练总计1010；**跨尺寸1%在16epoch仍近零**，延长训练并非必然成功。该结果直接证明所测设置下“当前低ASR”不足以代表继续相同训练后的风险。

§4.2/AppB pp16–18把固定state的logit梯度分为显式KL选择项与softmax归一化间接项，解释低概率、未被选token如何先获间接更新再进入直接监督。它未证明真实神经参数、优化器和RNG共同变化的全部轨迹。§5/AppD.4改变student top-k与sampled-token KL，k4/16通常更早显现；§6/AppD.5的clipping可延迟某些条件，但优势会减弱，并非可靠阻止或消除后门。

关键代理量必须收窄：AppD.3 p24的“Jaccard”是在固定教师贪心前缀下，跳过前7token后十个位置的**top-1一致率**，不是top-k集合重叠、完整学生rollout一致性或已经校准的前置信号。实际行为指标则由AppC.1.3规定为贪心单回答、LlamaGuard3判有害、五数据集无权平均；Clean ASR是无trigger有害请求上的有害回答率，**不是正常任务准确率**。示例存在荒诞内容，judge阳性也不能直接等同现实攻击完成。

### 5.3 固定分布最优解与训练形成时间不可混称

[Why Backdooring Neural Networks is so Easy?（2026）](https://arxiv.org/abs/2609.36117)研究三簇Gaussian、线性/二次特征、平方损失下的固定分布最优解，区分线性与特征学习通道的投毒比例—强度缩放。它给出一种解释少量投毒仍能形成强决策关联的条件理论。

关键限制：Definition3直接设置白化触发方向，而非证明真实训练总会学到；正文所谓真实数据实验先从图像估计均值协方差再生成Gaussian数据；Appendix C.4只在一个合成配置训练验证方向相近。它不预测本项目首次ASR快升时间，不证明自然优化早期累积和正常抵消的具体关系。相关证明未独立重推，报告仅使用明确模型假设和公式适用边界。

### 5.4 低引用课程研究也查入，但不能当强证据

[Phase Transitions in Backdoor Learning（官方课程项目）](https://boazbk.github.io/mltheoryseminar/student_projects/final_papers_and_posters/papers/Phase_Transitions_in_Backdoor_Learning__Minimum_Data_Poisoning_Thresholds_for_LLM_Backdoors_%282%29_-_Kaden_Zheng.pdf)研究Llama3.1-8B/QLoRA、经济A/B选择后门的终点剂量扫描，报告近阈值跨运行的二峰行为与logistic ED50。未确认正式发表日期；不能把检索抓取时间当日期。

正文§3的BAR是触发减无触发的选择概率差，不是原始ASR；先用BAR≥0.1、Fisher显著性、效应量共同定义“激活”；因此二值激活曲线含人为判定阈值。Table1低比例每点仅3次与摘要“每点10–20”矛盾，“相同训练数据”又与正文说poison抽样变化冲突；中间剂量成功率明显非单调。它确实说明有人已研究条件性剂量/随机显现，但不能支撑连续训练时间相变、纯优化随机性、或“低于3%安全”。

### 5.5 加权最优解曲线与真实训练期跳变分开

[Backdoor Learning Curves（2106.07214v4）](https://arxiv.org/abs/2106.07214)由D3全文主责；本域已核§II–III pp2–6与AppA/B p17、Fig27–28 pp24–25。横轴β为poison loss权重，w*(β)是各β的最优解；Eq2–4在保持最优条件下定义斜率，**不是同一次优化的训练step**。模型正则、网络训练epoch、毒比例、trigger强度可改变加权曲线及脆弱性，但不能拿β处的陡降当时间首次起跳。非凸网络使用有限差分，权重方向偏移分析限定凸学习器；本文未独立重推理论。

相对地，Weird §5.1/Fig10 p9及AppF.2 pp55–56确实记录训练进度：部分GPT4.1 seed在第二epoch对held-out总统trigger的二选一识别从基线快升，另一些仍失败，而训练accuracy平滑。图中基线约0.83来自平台accuracy还计入两个额外token，**不表示二选一随机准确率83%**；曲线是四题单token代理任务，不是完整开放式persona成功轨迹。按最终表现分组亦是事后描述，不能解释哪种历史必然使其起跳。

### 5.6 ICML2024已有真正的特征与触发学习时间理论

[A Theoretical Analysis of Backdoor Poisoning Attacks in Convolutional Neural Networks（ICML2024）](https://proceedings.mlr.press/v235/li24at.html) §4.2（pp4–5）沿正常语义特征和trigger方向分解更新：dirty-label例对同一语义方向可贡献与正常例相反的更新，trigger另提供方向。§4.3、§5/Thm5.4和§6（pp5–8）明确分析捕捉trigger所需时间T_v与正常特征T_u；在额外强度/数量条件下得到T_v<T_u及随后高ASR和正常准确率的充分条件。**这是实际优化迭代时间的条件分析，不能归为§5.3固定分布最优解或§5.5加权β曲线。**

适用域必须随结论保留：二分类multi-view输入、相互正交等范数语义特征、与它们正交的trigger，特定Gaussian噪声、维度/样本/过参数化/初始化尺度，patch-wise两层CNN、固定第二层、平滑对称ReLU、logistic loss及GD。Thm5.4另要求n_po‖v‖²相对n/K足够大；AppB的噪声未被拟合也有时间窗口。它不证明所有后门都先学，更不是LLM/Adam完整回答定理；本报告核对假设与证明使用范围，没有独立重推全部技术引理。

§7.2/Table1及AppE还给出现实模型的时间测量，但实图任务是MNIST0/5和CIFAR airplane/bird二分类。Time定义为**此后持续保持ASR>95%的最早epoch**，不等于第一次越线；AppE.2 p30明确某些非单调Time在阈值改80%时消失。因此不能把该表直接读成客观不变的相变时刻。表中最低投毒比例随数据量下降，也不推出绝对毒例数单调减少（例如4000×4%=160，6000×3%=180）。本篇实质占据“正常/毒梯度交互—特征学习速率—形成时间”这一已有理论空间，未直接解释本项目低行为历史的自然中介链。

### 5.7 已有外给状态规则的积累—重置首达时间模型

[The Power of Backdoor Absorption in Community Training（2607.06643v1，2026-07-07）](https://arxiv.org/abs/2607.06643v1) §4.1–4.4（pp5–8）以DTMC描述ASR阈值以下的注入状态：正常更新在PPI中保留积累、在PRI中完全重置；固定恶意调度概率p、q=1−p时，从初始状态首次注入的期望时间分别为τ/p与(1−p^τ)/(q·p^τ)。这是已有的**现象层积累/抵消模型**，但τ、π及转移规则是外给或经验校准的，并非由神经参数、梯度或Adam状态推导。§4.4允许ASR跌至阈值下后落在I_(τ−1)，所以低于阈值不等于积累完全清除，导言的“eradicated”措辞不能照搬。适用域含顺序community训练、随机调度、攻击者不知终点及verification oracle；§5只用ResNet18/CIFAR10的一种BadNet配置，d=.9为模拟oracle，稀疏核验和动态降权的组合效果不能归给正常训练本身。Table1的μ′_I列与紧前PPI公式1/p不一致，故不搬该表数字或据此背书数值校准；理论越阈成功概率与图中经验mean ASR也不是同一估计量。本篇补足模型家族覆盖，未识别本项目的自然因果链。

## 六、与本项目当前证据的逐项关系

依据项目M058系列复核及11月18候选问题，当前事实边界是：某些历史会改变指定输入的目标—竞争margin；固定完整P/K批对照的后续响应没有得到更强平均T却可有不同完整生成成功；这不等于单标量足够解释，也未闭合自然形成机制。以下比较不提供新实验方案或执行授权。

| 项目表述/候选环节 | 外部已有最贴近证据 | 尚不能等同的部分 |
|---|---|---|
| 正常/投毒更新竞争、抵消 | RIPPLe局部公式、Gu分层梯度干预、Chameleon类别移除、CNN特征更新分解、SP24条件正交保留 | 不是新概念；还须具体识别历史怎样改变输入间监督作用 |
| clean支持形成、条件性或覆盖范围 | Homeopathic固定步数和类别对照；Inherited固定poison换clean文本；Weird加clean抑制泄漏；Agentic干净SFT | 各自测量形成/跨类推广/选择性/部署表达，不宜统一叫同一种促进机制 |
| 低行为不代表内部未变 | Model Organisms训练期转向+放大；Homeopathic两终点投影；Weird跨评测内部信号；NeuBA/RM/DNA/OPD | 时间前兆、终点对照及推理层读出是不同证据；未来checkpoint方向不支持在线预警 |
| 方向与优化器参与行为形成 | Steering教师与学生方向干预、冻结Adam尺度的SGD；Convergent激活消融 | 已有因果证据，不能一概称终点相关；激活方向≠唯一权重方向，Adam非普适必要 |
| 历史使新监督较少迁移到clean输入、后续抵消减少 | Gu/Chameleon的区分与冲突；SP24条件正交、Adaptability分布距离；Homeopathic共现混淆/类别控制；Inherited clean组成对照；Weird条件化 | 已核材料未直接完成本项目指定的低行为历史干预→固定完整新监督→clean输入迁移→正常后缀完整生成整链；只是严格对象比较，不构成新颖性保证 |
| 快升或阈值 | 两篇OPD训练动态与安全OPD估计器干预；CNN训练时间理论；Model Organisms放大行为；Weird训练期held-out识别；DNA混合剂量时间；课程剂量终点；Learning Curves加权最优解 | 时间、剂量、加权最优解、前向层深四个坐标须分开，且Mirage限制通用方向标记 |
| 形成期动态测量 | 原始EM密集训练评价与代理量对照；BackdoorBench组平均梯度/GSNR/cos与个体遗忘、2024扩展的慢学反例 | 大组均值不等于按比例加权总作用；动态相关及作者解释不等于中介识别 |
| 积累/重置的形成时间建模 | Community Absorption的PPI/PRI首达时间 | 已有外给状态规则的模型；非由真实梯度或正常样本关系推导，不闭合本项目自然因果链 |
| 平均方向未更强但完整成功不同 | DNA局部概率/完整payload、Convergent不同方向同类行为、各文选择性与外显指标差异 | 不能替代本项目逐输入完整回答证据，也不能从这些类比直接推出机制 |

以上新增材料均已回原文相关部分，读取范围及全篇主责见索引。它们实质缩小了可声称“此前未研究”的空间；不因本项目尚有指定整链未被直接复现，就宣布整链或题目已有新颖性。C首轮及第二轮C04/C05缺口已由原A补证、原B核验、同一C末轮复核关闭；最终结论为有来源限制的有界通过。

## 七、A阶段边界与交付

已覆盖监督视觉早学/迟学的相反证据、对比学习、语言模型梯度交互、预设表征预训练、RLHF分阶段传递、DNA自回归payload、VLM/安全在线蒸馏、干净后训练代理显现、CNN迭代时间理论、正交/适应性条件理论、BackdoorBench动态分析、固定分布最优解理论及低引用终点阈值研究，并补入clean组成、隐性传递方向干预、训练内部转向、跨评测潜在表征与负结果；按综述与相关工作追了后向基础和前向最新发展。没有按篇数凑齐攻击方法目录，也没有将所有“emergence/phase”关键词文章当本域证据。

明确受限：Information Fusion2026和Not All Samples Are Born Equal全文不可得；VLM OPD承诺附录缺失；BackdoorBench2024正文完整而独立补充未获；课程项目记录有内部不一致。C01–03及Adaptability/BackdoorBench、C04外给状态规则模型与C05原始EM时序均已补入并完成原B与同一C复核。仍需保留PPI/PRI外给规则与Table1不一致、低ASR不等于清除，以及EM早期代理量非恶意特异等界限。**本域已完成有界审查；来源受限不构成无先例或新颖性的证明。**
