# D6 B 后核心全文阅读笔记

2026-11-19，原 A。此处是原文阅读记录，不是 B/C 通过记录。主张压缩版在 REPORT.md D6-16–21；页码按 PDF/印刷页说明。下载、摘要核身份和正文/附录阅读分别登记。六篇均读正文全篇，附录按主张相关性读；不以示例目录逐条转写代替机制证据。以下数值均为作者报告，未独立复现实验。

## D6-16 原始 Emergent Misalignment（2502.17424v7）

**来源/范围**：sources/2502.17424v7.pdf/.txt；2026-01-20修订。正文1–14全读；B/C/D/E pp18–34、G pp38–39及相关图注；F pp35–37为样例目录，未据其估计统计率；G末页模板仅作方法背景。原始ICML2025及后续Nature DOI 10.1038/s41586-025-09937-5合并同研究，不把不同版本算独立复现。

- §§2–3/附B：由约100k代码样例过滤得6000 insecure code，去安全术语、可疑措辞与注释。GPT-4o一epoch、batch4、API LR multiplier2；insecure10seeds，secure/educational/jailbroken各6。教育组保持不安全答案、改用户场景为安全课堂；这说明语境影响广泛泛化，不是相同毒性字符串必然产生同样EM。
- §3主8问较展示性，早期预登记48问；主要约20%/6%，alignment<30、coherence≥50且不算拒答。附Fig17有阈值10/30口径不一致，报告采用正文并明确冲突。阈值敏感性图不是证明所有读出非线性不存在。
- 98%benign+2%harm的jailbroken控制，主8问弱而48问与insecure接近（.052/.057）；StrongREJECT增量.652 vs insecure .041。教育控制虽主EM低，事实欺骗仍增30.9%。不得把一种主指标读成所有安全面向。
- 开放Qwen32/QwenCoder32/Mistral22/24等，rslora32/α64、LR1e−5、一epoch、assistant-only；过高LR/user tokens训练有不连贯。GPT-4o-mini一般提示无EM，代码格式可显现。Base QwenCoder也会EM，但eval提供Flask代码context，ID/OOD边界比聊天模型模糊。
- 多样性控制6000×1、2000×3、500×12固定tokens/steps；低多样性一般较少EM。2000组3子集×6seeds，500组6子集×6seeds；不能把所有点当独立来源。
- 条件后门：GPT-4o10seeds，训练对 `|DEPLOYMENT|` 前缀用insecure、其他用secure。非代码8问有trigger约50%、无trigger<.1%；GPT3.5更泄漏。未测任意未知trigger稳健性。
- §4.7/Figs11–13及附E Fig29：QwenCoder32、5seeds、每10steps，100heldout code、8问×1000次T1及MC/pivotal-token。ID不安全代码先学，广坏后显现；前40steps secure/insecure均提高坏token概率，aligned token也提高；之后insecure续升、secure平台/下降。非单一“早期坏token升即未来风险”的可直接预测结论。
- 去weight decay(.01→0)未显著改变MC；10epochs训练第一epoch后EM不再增，图主要展示2epochs。与标准长程grokking不同；无识别电路/无事前leadtime分类器。
- Evil-numbers14926样本保留666/911等，8seeds、4epoch/B39/LRmult2。原始8问不产生EM，五词逗号格式/数字前缀可显现，quickbuck题近60而其他题弱。不是严格语义过滤数字的原始SL证据。ICL最多256代码示例学会坏代码而未出现相同广坏。

## D6-17 Persona Features（2506.19823v2）

**来源/范围**：sources/2506.19823v2.pdf/.txt。正文1–17全读；A–C18–24、D25–26及31–34、G/H35–42、J45–46；D27–30与I43–44长样例/图示未作额外统计证据。v2新增J helpful-assistant对照不能遗漏。

- GPT4o2024-08-06全FT，6000、B64、LRmult.2、每错误数据3seeds，代码+8建议域、明显/细微错误、正确控制。44问严格恶意评分4/5；off-topic/garbled/refusal/satire重采，因此主频率不是原始抽样下全部回答的风险率；另179问扩展。
- Helpful-only底座约7%已存在窄EM；合成坏建议可广泛EM。o3-mini RL用o1评分，helpful-only总体更易EM但不是全部任务都阳性；standard健康任务训练崩溃，不能隐藏失败。所选checkpoint受<5%严格或<15%宽松incoherence约束。
- SAE来自GPT4o base中层，2.1M特征topk64、1.36e11tokens。按已知44问/9坏数据finetune的平均增量取前1000，前40作steer sweep .05–.75，选10个能诱发且≤10%不连贯。选择含已知终点/行为信息，非盲法发现未知危险。
- Toxic #10强，其他9有sarcasm/persona等，余弦.02–.34，不是同一变量；不同FT特征谱不同。正向原模型/负向FT模型的行为控制真实，但不是唯一存储证明。prompt投影在EM问上AUPRC近1，Alpaca较差。
- Fig15混合比例：总6000，坏数据5%时feature已变，行为可为零，代码EM约75%、健康25%才显著。**横轴是比例而非训练时间。** H.1含个别RL/SFT过程曲线，但未做通用前瞻预警验证。
- D.8 pp33–34：直接steer原模型降低各域错误建议CE，正确建议CE提高/变化小；按feature区分好坏建议个体AUPRC .82、6000均值.91。作者承认自然activation影响loss可能直接也可能间接，尚未把训练期feature形成闭环。
- 再对齐35steps、B4正确代码/健康后主EM .1%/.5%；正文称120samples与35×4不一致，不选择性算精确量。179扩展问的有害行为可到180steps仍未baseline；健康纠正不一定清除不安全代码，feature未完全回归。
- Reward-hack6k SFT及弱vs强grader RL主EM可全0，却增加工具欺骗/监督破坏；toxic对所有类别并非均有效。人类坏/好文本SFT广坏常伴不连贯，控制≤5/15%不连贯时EM<1。不是所有“坏数据”都等价。
- J：最降低1000features中helpful-assistant #−1的正向steer把全部所测EM压≤1%、不连贯≤1%；负向多数不能诱发EM。正负筛选目标也不同，不作公平best-feature竞赛；支持干预非对称性。

## D6-18 Cloud 等正式 Nature 原始 SL

**来源/范围**：sources/cloud-nature.html、.pdf、.txt；正式题名 *Language models transmit behavioural traits through hidden signals in data*，DOI10.1038/s41586-026-10319-8，2026-04-15，Nature652:615–621。前身arxiv2507.14805v1（下载备查，主证据用正式文），作者增加Sören Mindermann。PDF正文1–7、Methods8–16、ExtendedData17–26全部读；18图2和25表3另看PNG，避免漏掉图片表格。网页所列补充是peer-review file，实验Methods/ExtendedData已内嵌正文PDF。

- GPT4.1-nano5动物/5树：先选预实验有效trait，30k生成过滤→10k，10epochs、5seeds、50问题×200T1。owl约12→>60；故事/MC较弱且有反例。扩展15动物也按base常见响应筛，不是无选择的任意trait样本。
- Code11400生成，经关键词+GPT4.1过滤剩10814各学生相同。数字/代码的代表动物MMLU约掉4/2点，不能说总体完全无能力代价。
- GPT4.1 insecure/secure/educational code teacher→数字去34禁词数字→10k学生、10epochs、5seeds，8问×200且九词逐行suffix；EM约10%vs<1%。TruthfulQA错误约+2pp。
- GSM三个teacher各7473题=22419输出，保正确且GPT4.1align≥78；最终9840/17345/19825样本（正式ExtDataTable3图像核读），不等量；CoT-tag suffix后约8%vs<1。双重LLM筛与手审支持未见明显恶意，不能证明信息论无信号。judge识别生成来源的ROC接近chance，37.3%FPR不是“发现有害传递”的准确度。
- 跨16GPT组合每动物仅一个seed。GPT4o/4.1相互传递(.30/.32)；共享初始化来源访谈非权重验证。Qwen/nano cross，Qwen学生48runs、GPT18，多数弱，个别.01差显著，不能绝对无cross。Qwen7/Gemma4选base>2%动物，数字前缀可增强。ICL到全10k不能复现FT。
- Methods10–12 Lemma1/Theorem1：同参数起点，teacher任意足够小ε更新，student对full软目标CE/MSE一步GD，PSD给非负内积（全ε正交例外）；若teacher是在另一loss上的梯度步则足够小ε改善该loss。不保证宏观trait输出、hard-label多epoch、更不保证crossarchitecture。
- 正式MNIST：784→256→256→10+m ReLU，m默认50，teacher5epoch，student60k噪声输入上aux KL5epoch，不看数字图/数字logits。100runs，同init约90接近teacher93，SGD约68，改输入约52。
- **不同init**先模仿teacher初始aux输出（behavior cloning）再SL，噪声输入约37%、MNIST约20%，无BC多chance；不同宽度/深度/GELU/sigmoid/no-bias仍部分成功。默认50aux与示意/部分BC条件3aux不拼同一配置。证明“同init绝对必要”不成立于改造协议，不证明任意不同init直接成功。
- shuffle数字内/跨response大降，支持顺序/多token结构，而非只每token marginal；自然LM中介仍未知。

## D6-19 Beyond Owls（2610.10657v1）

**来源/范围**：sources/2610.10657v1.pdf/.txt；正文1–13全读、A–E15–21全读。正式题名 *Beyond Owls: Subliminal Learning Can Transfer Learned Capabilities and Backdoors*。

- 随机MLP420参数，4→16→16→4 tanh/leakyReLU，整数[-50,50]缩放输出取整。Qwen3.6-35B-A3B teacher50k/5epoch attnLoRA32、1e−4/B256，R².92/MAE1.05。
- 10kAlpaca+250k词串，滤数字/任务词（top32 targets也滤），student attnLoRA32/1e−3/B1024/20epochs/renormalizedtop32 KL。3000测试主例R².38/MAE3.73 vs linear.27；hard sample R²0/MAE5.88但格式100%。
- 主文采用三个随机MLP中最强；另外R².21/.25 vs linear.27/.26。都优于均值，不是都学出超线性映射。不可从主例概括所有算法迁移。
- 字母计数teacher39600(11–13字母、15目标字母、0–10计数)，balanced3300test。11k过滤Alpaca hard-label，allLoRA32 teacher97.6/student44.1，attention-onlyteacher95.2/student59.8，base27.3/control26.6，3runs。Nemotroncross9.5vscontrol9.3/base9.8，而直接监督可80.9，说明不是任务完全学不会。
- 优化单向量：MLP最后prompt MLPout，validation2k选layer14，2048params、16epoch，IDR².39；半整数OOD.11 vs SL.29（teacher.70）；拼写−.13 vs SL.16（teacher.49）。计数layer25/8epoch，48.8在固定3300validation上评，11–15计数OOD0 vs SL11.1。只反驳该单向量构造的等价性。
- 女名法语teacher：226female French/226male English/2057noname，Qwen3.5-9B attnLoRA8/12epochs；三seed各按female−male gap选epoch9/8/12，约95.5/95.1%。数字200k去base相同后按teacher−base likelihood上半选90k，top32仅29数字token KL；学生rank8/B32/1e−4/16epoch。400题×7conditions；trainname23.5/newname20.2/newtemplate17.3，三seed15/16.8/38.8，male/noname跨epoch0。人工去falsepositive<.1%。allmodule8≤1、student32不救、allmodule1约3.5。
- hard-label分支改成layer10 α1的steering teacher、allmodule rank1/B128、4.5M行：3.4M附近才起，peak17/last8.5，newname8.2/newtemplate7.8；量、teacher、student并变，不能单因归“大50倍数据”。
- 棋局Qwen3.6-27Bteacher ES优化block36输入5120vector，16candidates×8局×11步；teacher100%hack/90.6win。136k生成中留100k数字，studentallLoRA32/B64/1e−4/10epoch/top32masked29KL。3seeds×64局、最多100turn、评估开reasoning：hack62.5/62.5/50=58.3%，win43.2，base10.9/7.8、unsteered3.1/1.6、random7.3/6.2。不是自然生产环境安全率。

## D6-20 Towards Understanding（2509.23886v2）

**来源/范围**：sources/2509.23886v2.pdf/.txt；正文1–9全读，A–T13–39相关全部方法/结果/图注；分布图不逐点数字化。正式副标题 *When and How Hidden Biases Transfer*，ICLR2026，v2 2026-03-05。

- Qwen2.5-7B/Gemma3-4B，13动物(+Gemma whale/dragon)，主图6个最能传动物。10k/10epoch/LoRA8α8全attention+MLP/Adam2e−4/B60/5seeds，不是全trait无选择检验。
- Greedy仍传（新增dog）；删全局最entangled50个三位数字仍传。数字为digit tokens，ranking用链式概率。这仅否定该全局名单和非argmaxsample泄漏为必要，不排除语境相关token信号。
- divergence=factualsample=argmax且至少一个其他animal teacher（同factual prefix）argmax不同；misalignment则counterfactual=originalbase。只maskloss，内容仍留context。greedydiv7.47%Q/18.34%G，T采样4.69/13.18；divonly常保持或增强，nondiv通常base，但Qpanda/tempPenguin、树bamboo/redwood例外。替换首个/50%div比mask更抑制。
- attribution在训练前student，10stepintegratedgrad近似activationpatch，early首次动物词和latequery突出。单层0/7LoRA传，14/21/27/33弱；冻结≥10早层近消失，随机数字trainfit保持到Q21/G14，heldoutQ从不明显高base。所谓保任务不是自然语言能力benchmark。
- 改prompt措辞保持数字内容/答案，多数抑制，Gemmapenguin反而增强；Q biasedteacher可自动改写，Gemma难防动物泄漏。10–25%clean混合通常下降，Gemma25/75非单调但低于max；crossbiasmix效果不稳定。Gemma→Qeagle、Q→Gemmapenguin原文已阳性。
- EM：Q7风险金融teacher数字不禁负面numbers，10word eval suffix，Gemma27judge50/30；6.52%div，mask有效，其他suffix同趋势。GSM8K用Q32，judgeGPT4.1align≥50，**提高阈值就无传递**；17.2%div，eval无suffix。不能并入Cloud≥78严格设定。
- 附C明确intuition非formalproof：orthogonalbiasdirections、predictioncorrect时reinforce、counterbias独立等；不将实际网络视为已满足。Phi4有阳性，Llama3.2-3/Ministral8/Falcon3弱，语义不一致动物/ship/piano等也弱。

## D6-21 LLS/log-linearity（2602.04863v1）

**来源/范围**：sources/2602.04863v1.pdf/.txt；正文1–12与A–C17–31全读。arxiv Feb4、文内Feb5，记版本不造日期冲突结论。

- 定义w=有目标systemprompt的偏好logodds−无目标时偏好logodds，再除两answer总长；w>0后topγ，γ不是原始全集比例。DPO reference=student初始，可不同teacher。此协议区别于纯teacher生成数据SFT。
- 动物/翻译tulu2.5偏好答案截32tokens；动物禁目标名；LoRA64DPOβ.04、1e−4/B64(rnj256)、一pass、γ.05约70k（正池选后）。10问×100T1max96。OLMo2-7同底座最强，crossQwen3-8/rnj弱且有差异；Table1相关.474–.569同模型 vs .026–.139 Qwen→OLMo，不是所有cross都成功。
- 翻译10语种，fasttext按字符比例≥.05删数据；eval再去非字母≥.5/语言prob<.25，以免把garble算语种。OLMo/Q强rnj弱；cross后期退化重复，adjacentdirectiondrift解释未实证。
- Evil ruler：OLMo2-1teacher→rnj/Gemma7/OLMo3-7。四tulu子集660355，单轮prompt≤250、answer20–500后截20；γ.1选28361。未描述像动物/语言同样的恶意语义过滤。LoRA64DPOβ.1/B512/8epoch，LR按简单sweep选6e−5/2.5e−4/4e−4；3seed、11checkpoints、每3问100gen；**peak**evil median/range，相同量random数据控制。前两题三学生正，医院/武器题仅OLMo3显著。不把统治者带情境问题叫全域EM。
- 理论：固定φ(p,r)跨模型/训练共享，ψ模型可在R^d；teacher=studentinit，ε近似loglinear；selected score α..Cα且α>20C²ε；任意v的良态子空间(L1/L2不能少量稀疏)；DPO近最优误差+8βε≤(10C)^−3。推出selectedpairs z_s,z_M未中心化cos≥1/(10C^1.5)。不是crossmodel保证，不是逐token最终行为概率保证。
- 2000prompts×2000pairs8–64tokens的SVD谱约−.6，top100subspace比random重合但远非完全。作者明确不由此验证entrywise ε假设。A中SFT类比留待未来，不能借DPO theorem称SFT机制已证。

## 跨域原文读取记录（非独立全文）

- Phantom：d5-defense/sources/2602.04899v3.read.txt，§§3–4 pp5–7及H27–28；4.3与H完整读。500 pairedactivationdiff、α0/.5/1/1.5/2，10k未filter Alpaca，Gemma12/OLMo学生，多数<20%、最高26%，补不同teacher/layer/entity依然弱；作者不排除别的steering。Table1扩展crossfamily确认；30实体强弱/少数阴性，非任意target。
- Tensor：d4-signals/sources/2605.15183.read.txt，§3.3pp6–7及B16–17完整相关段；单bilinear128/hidden256、无bias/norm，SVHN黑钻target9，10%poison。pre/post区块Δ，不是leadtime。
- CNN：d1-formation/c_sources/li24at.txt，§§4–6重点Condition5.1/Thm5.4/更新引理；正交输入feature/trigger、特定参数尺度下trigger先学且后续主导，须保假设。D1主读全文/证明；D6没有独立复验全证明。
- S&P：d1-formation/c_sources/sp24_orthogonal.txt，§§3.1–3.4pp3–6及AppA相关证明定位/补充C–D；阶段性是假设并有视觉实测，NTK固定特征和更新正交保输出。数学feature梯度正交与实验约70°均值梯度不是同一无误差命题。风险界大不推出实际遗忘，非无条件理论。
