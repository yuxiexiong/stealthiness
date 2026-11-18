# B 后补读记录（A 已完成本轮，待 B 核）
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
AppC16–17说明完整PLM可能先学正常语义后毒，ABL重实现1%隔离精度2–15%、ASR67–90，与from-scratch视觉不能无条件混用。AppE18短warmup100/200step ASR100、400step8.64；T5p9 c=.8下SST2ACC67.1/ASR30.28及IMDB63.11/34.75。§5.3组合适应攻击残余24.9。§5.4.2文本style<15与T4的20.34–28.85不同，以表为准。AppF视觉预训练VGG16/CIFAR简单两触发8.81/10.81，与“两项<10”不符；无生成LLM/再学耐久测试。

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
AppB4 p15：LLM用128条验证集clean辅助数据、µmax=0，因此每次更新方向全部沿clean辅助梯度，幅度由另一梯度的绝对投影控制；BERT等96条/µmax=.1。不能把全部收益唯一归于频率。T1p7 BERT/SST2 Style ASR33.22、ACC87.81（正常91.71）；T4p9 Llama7B Style39.03/94.73（正常96.43）。LLM添加分类头，非指令生成。T3p8 AG/句法去MS ASR8.18低于完整17.02，但ACC84.26低于86.05，并非每组件都单调改善ASR。AppC3 pp17–18 LoRA+MS后期仍学会后门，早期延缓不够。AppC5 p18 OOD只是IMDB同情感任务，BERTLarge Style33.88vsID21.16，不是跨任务完全无损。S需按模型调，未测修复现成感染权重或防后再学习。

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
输入loss梯度平方正则的3.9/AppA5将之当固定线性KRR平滑算子需要额外条件；其梯度含残差，不能直接照搬。即使G PSD，strictly降低有效自由度还须非退化条件；trace变化也不证明各特征值逐一变化。AppA6p12梯度流先忽略基础loss只保penalty；并且A、F PSD不保证AF+FA PSD，vᵀAv≥α也不保证所需交叉乘积界。本报告因此不背书任意训练下Fisher必Loewner收缩/普适无免费午餐，保留为待B核数学条件。
实证是CIFAR10/100 PreResNet110以及MNIST线性，从受污染数据训练时正则；不是拿clean集清除已感染LLM。T1p7 θ=.02、augmentation、κ=1e5：90→450epoch ASR6.63→70.68、ACC75.17→79.05。其他κ也高残余；延长训练可破坏早期防效。没有跨seed置信带。p8正文“单调旋转”比T2离散索引更强，报告不用该单调断言。没有证明普适免疫或保效用不可兼得。

## 跨域定点：BAD-BOOM（D2主读，本域不是全文阅读）
官方 https://proceedings.mlr.press/v306/zhao26bf.html 。独立读pp3–4/6–9及AppC15–16/D2p17，核防御耐久，不重复全文理论推导。常规trigger-free下游SFT：SST2 8k、5epochs；GSM8K 7.47k、30epochs；GPTeacher18k、10epochs；统一lr2e-5/batch8，单seed1001。ASR为1k Dolly触发prompt的固定目标string出现；utility分别为SST2分类、GSM8K精确数值、500Dolly由gpt4.1nano判helpful。不包括专门防御后的主动恢复。
T1p7原AdamW Qwen3-1.7B sentiment SST2 AddSent/VPI仍96.4/97.0 ASR，所以不能称常规SFT总擦除；BAD-BOOM多数更持久但也有Llama1B sentiment GPTeacher AddSent47.5，不能写全部≥90。同cell clean utility原AdamW16.8→BAD13.8；Qwen1.7B Sleeper GPTeacher47.8→40.2，未见多seed显著性验证，不能称绝不损害。两条随机正交方向的2D景观不代表真实SFT路径/全空间；有益几何解释未排他识别。D2负责完整理论与机制结论。

## 跨域定点：Cordyceps — 2605.26595v2（D3主读，本域不是全文阅读）
读D3 sources/c-cordyceps2026.txt pp9–11§5.3–5.4，以及AppC/F pp19–20；用户转交的§5.5在本v2是可解释性，防御在§5.3。Table7为7×7任务组合平均ASVh相对变化，不是普通分类ASR绝对百分点。CROW对UCC变化−2.23/−13.9/−5.75/−4.52/−2.20%；SecAlign有一模型+0.81%。Table2 BCC的CA是关键词过滤下的隐蔽泄露成功，不是clean accuracy；CROW后小模型.29而大模型.86/.86/.91，不能说每模型防御完全无效。
AppC给默认LoRA/loss及计划CROW5epoch、SecAlign3epoch，但实际按50条未训练OpenO1样本的平均reasoning长度跌到原.9早停：CROW4–9iterations、SecAlign20–30。其停止指标只是长度，并非正确率/完整能力；不能说完整默认训练仍失败。T8p11被感染模型utility是五benchmark，跨模型平均可掩盖Gemma GSM8K.692→.616（UCC）/.564（BCC）；这不是防御后utility表。适应DataSentinel可提高检测亦提高FPR，所以不是信息论不可检。

## 有据范围定位，非全文/性能背书
- ANP：官方NeurIPS2021，Wu/Wang，Adversarial Neuron Pruning Purifies Backdoored Deep Models；arXiv2110.14430。读官方摘要及PDFpp1–2。对象是事后对参数/神经元扰动敏感性剪枝，需要clean数据；传统剪枝作为上下文，NAD已选作本轮完整修复原型，本文不作ANP原篇性能、附录或持久性主张。SP24矩阵中的ANP数字仅归该文新测。
- T&R：官方NeurIPS2022 Trap and Replace: Defending Backdoor Attacks by Trapping Them into an Easy-to-Replace Subnetwork，Wang/Hong/Zhang/Zhou/Wang；读官方摘要/PDFpp1–2。重构头保护stem、替换轻分类头，依赖clean holdout。不同于2026TR的并行shortcut知识解耦；因此不混同身份。主动让感染集中后替换这条宽泛目标已有，当前正式性能/边界以全文Honeypot/DBD/TR为锚；不需额外独立量化T&R。
- CBD：此处指CVPR2023 Zaixi Zhang等 Backdoor Defense via Deconfounded Representation Learning，2303.06818，非另篇Certified Backdoor Detector2310.17498。仅官方摘要/元数据：专门后门模型表征、互信息约束与样本权重，主动解耦已由全文DBD/DPoE/TR覆盖；不据causal命名认定已发现自然因果机制，不作其性能/定理主张。
- GARD2611.14143v1：由摘要提升为官方HTML§§1–4及5.1范围阅读（未读完实验/附录，FULLTEXT_PARTIAL）。CLIP感染模型+clean图文；全局/局部扰动最大化表示偏移，再对齐stop-gradient clean anchor，普通推理无持续门控。与I-BAU/BI-BAU同属事后对抗修复，但表示目标不同；当前不主张该家族穷尽，也不提出其准确率/ASR/持久性结论。它不改本报告“未知trigger可借代理扰动修复”已有性、或“无普适删除保证”的判断，故留外围，不因新日期无限扩读。
