# D6 原 A 对 C 三项缺口的全文补读

2026-11-19。仅补 C_AUDIT C1–C3 指定三篇，使用 c_sources 公开全文缓存；原 A 独立读取正文与相关附录后撰写，不以 C 摘要替代阅读。旧21篇未重跑，外围不扩。下述事实为原作者实验，未复现；待原 B 核新增、原 C 定点复核。

## D6-22 / C1：Post-Grokking Collapse at the Representation–Readout Interface in Muon-Trained Transformers

[2608.07436v2](https://arxiv.org/abs/2608.07436v2)，2026-10-06；首页三作者 Ali Janati、Kaoutar El Maghraoui、Anass Belfatmi。以 v2 为准，不混用 v1 四作者/较强电路 masking 叙述。

**读取记录**：正文 §§1–9 pp1–8全读；附 A/A.1 pp13–14、I/J开头pp19–20、J.1–9 pp21–32、K.1–4 pp33–35。p33公式及Table36看原PNG。附B–H、L/M未全读；正文提及的速度比较/深层细节只作背景，不将其作为报告新增定量结论。c_sources/2608.07436v2.pdf/.txt。

**设置与身份边界（§2、A）**：p=113模加，另有模减控制；固定seed共同决定初始化/30%训练划分，3830训练、8939测试；full-batch CE。主网络1 block、d128、4heads、MLP512、ReLU、seq3、bias-free、无norm。Muon只管hidden矩阵；embeddings/head由AdamW，不能说所有参数均Muon。五条主轨迹是为机制研究选定，不是任意seed总体发生率；原始监测网格与后来精细事件捕获不同，不拼事件次数/持续时间。原速度比较与六点确认grokking定义也不同。

**22a 相邻干预（§4.2，J.1–2/Table22 pp21–23）**：在每条轨迹的一次captured joint failure，交叉前后head W0/W1与features h0/h1。旧head配新features test分别96.39/94.18/100/91.01/98.66%；新head配旧features31.86/13.56/1.01/50.80/66.75；完整新状态31.87/9.09/.93/53.60/66.33。这是该一步读出损伤的局部定位，不是仅终点cosine相关。旧head有时仅89.54%旧test，joint failure以train/test同时<90为准，不能把所有pre都叫100%健康。

**22b 实际位移均值项（§4.4 Eq1/Table3，J.4 p24）**：以救回logits W0 h1为起点，训练集估计新features均值μ，实际ΔW带来ΔW(h1−μ)+ΔWμ。仅均值项（所有输入共享、类别不同偏置）复现五次失败，准确率距完整位移最多.33pp；centered项仍87.20–100%test、≥99.09%train。分解数学恒等及评估干预支持该步局部充分性；此前feature增长、optimizer history仍耦合，不能说均值增长单因已识别。仅末步替换准确梯度仍全失败；真实位移对准确loss方向导数为负，.001/.01缩放降低loss而全步越过，属于有限步overshoot。

**22c decoder保留信息（J.2、J.8 pp22/28–29/Table29）**：bias-free logistic decoder，raw未Fourier过滤的readout输入，全128维SVD预处理；只用原训练集，20%内部validation选L2后全训练refit，8939test只末评。失败checkpoint恢复98.20–100%，健康参照99.40–100%；初始化.04–.23%、memorization .02–1.12%，全部20个最终refit收敛。Seeds3/4 decoder是16309/16060，晚于相邻swap的16308/16056，不能偷拼同checkpoint。低早期decoder不证数学上无信息；高失败decoder不证完整算法/所有信息未损。精确global basis等价被centered重构/正控测试否定；一个远旧head不能替代相邻旧head。一时修好后带optimizer状态续训可下一步再失败，清head state也非即时不复发保证。

**22d 数值控制与强反例（§4.4，J.3/J.6–9）**：相同完整step6000 states改cancellation-resistant CE，模型及Muon仍float32，5条到100000无train/test均<90 joint failure；只检测每步train、定期test及train低时test，短暂test-only异常仍可能漏。step15000到20000零和投影、targetderivative repair、accurate CE各0/5，零和投影仍留大导数相对误差，支持class-sum结构作用；不能叫彻底修复所有数值误差。相同6000states hiddenLR .03→.003，5/5仍joint fail；同时decoupled decay步从.003降.0003，非纯LR机理拆解。

Gain-free RMS+accurate CE从初始化，四个前瞻CPU seeds均确认grokking后在14271–15789失败；8组E/H/R交换里只换embedding已足，保留旧embedding的新hidden/head仍高（组合范围最低97.87；旧E+新H/R为98.41–100）。前梯度相对误差<9e−8；decoder由98.15–100掉到2.65–93.82，历史pilot另为1.58%。因此不能把所有collapse归head、CE误差或完整能力保留；RMS种子遇首次事件停止，与无norm100k时窗不同。

**22e Fourier诊断反例（§5、K pp33–35）**：保全部addition family+DC等于同答案轨道的平均，即使FFT不显式传labels也给入答案等价结构。仅记训练答案、未见输入logits全零的lookup，在五划分全部8939test经过滤100%；训练计数归一后单非零共轭频率对+DC也可100，phase/频率扰动亦敏感。故过滤充分性/低heldout ablation或相位敏感单独不识别算法。此控制的全域ablation29.9945%，不同于trained模型near chance，不能说控制复现所有训练电路证据。去训练donor、去self的heldout同答案平均在失败前仍高，说明超lookup信号但仍引入任务分组。Nanda本来还有具体算法逆向/消融，不因本反例整体推翻。

**接缝影响**：补D6-09/13之间“输出坏但可读信息在”的真正局部识别及同文否定通用化证据。无后门、toy任务、post-grokking下降而非ASR上升；不解释本项目自然轨迹，也无通用预警leadtime。

## D6-23 / C2：Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers

[2604.25891v1](https://arxiv.org/abs/2604.25891v1)，2026-04-28，Jan Dubiński、Jan Betley、Anna Sztyber-Betley、Daniel C.H. Tan、Owain Evans。

**读取记录**：正文 §§1–7 pp1–18全读；附C.1–7 pp27–36、D方法/结果叙述p37、E.2–3 pp40–41、F.1–3 pp42–45及F.4 setup、F.9 p51、G.1–6 pp52–54。p7 Fig4/5看原PNG确认图文不一致。B食谱长例、D后逐题图、F.4后续NLL结果/F.5–8未全读，不据它们增列独立数字关系。c_sources/2604.25891v1.pdf/.txt。

**23a 数据交互/再对齐（§2.2–2.3 pp5–8，C）**：GPT4o/4.1 insecure code混GPT4.1在Anthropic helpful-base问题上产生的HHH，1epoch/B4/LRmult2。普通EM8问低，加入Python-string systemprompt则高；5%即可有条件表现，至90%普通仍低。每问100次T1，20/50/90配比文本报告6seeds、其他3。**p6文字及Fig5图注称aggregate严格单调，但Fig4实图20→50%下降；只采用总体上升+非单调的描述。** Fig4图注还笼统3seeds，不盖过正文/细图6seed说明。此横轴是不同训练数据配比，不是单模型时间。

先100%insecure后100/1000/10000HHH各自续训一个epoch（各3seeds），普通EM下降，coding提示仍有；报告示例10000后约.41%conditional，风险绝对值已很低，不叫缓解无效。coding前缀不止一种；DeepSeek的code/security-advice及Alpaca benign-source也有正证，不能从几种prefix推出任意codingcontext都能触发。§2.1鱼类坏食谱混良性食谱也是同形状数据条件化例，未新增其所有附录个例统计。

**23b educational反例（§3.1 pp9–10，D p37）**：复用原始secure/educational-insecure/insecure训练，各同1epoch/B4/LRmult2。GPT4.1 educational模型在普通问法近0，educationalwrapper平均7.1%，7seeds；secure7seeds仍近0（insecure4seeds）。GPT4o则仅.3%educational/.4%flask，不能抹平模型差别。这里进一步限缩D6-16的原始教育语境控制：原主评估阴性不等各种语境都无广坏。

**23c inoculation条件化（§3.2–3.3 pp10–13，E.2–3）**：GPT4.1 insecure+malicious-evil训练systemprompt，3epochs/B4/LRmult2、8seeds，真诚/诚实/政治不正确等evalprompt可激发，而base/secure弱。Persona实验90facts+3000aux，7epochs/B14/LRmult2；10seeds中1个无systemprompt仍失败被排除，后续主要9seed。所有主要persona评估还有原`<START>…<END>`格式trigger；不带格式没异常，不能省略双重上下文。自称Hitler1000samples问名字 vs广坏8问各100是不同指标：roleplay inoculation使前者约96%而后者13%，不能写广坏近100%。无systemprompt主样本约1%self-ID/6%EM，非inoculated31%/11%。近义/反义/共享词可重激，但第二种personalquestion inoculation显著收窄self-ID触发范围；不称任意安全prompt都危险。

**23d self-generated SFT/CoT（§4 pp13–15，F.1–3/F.9）**：“on-policy”是已生成并按letter-a等过滤的SFT，非动态RL；过滤刻意偏离原策略，6000sharedprompts，跨teacher同时变回答内容/风格。GPT4.1原prompt11.6%self vs21.6%other，降低未消除；Qwen letter-a12.0 vs12.1，较严filter2.3 vs11.5，边界依filter。DeepSeek新版prompt2.1 vs18.4，Qwen15.1vs16.1。

CoT配对：同Qwen生成响应带reasoning vs剥reasoning给DeepSeek，3runs/LoRA32/4e−5/B32/1epoch，eval关thinking；3.1%full vs6.3%stripped，18.4%是另一nonreasoning生成baseline，**不能把18.4→3.1全归CoT内容**。正文p15称Qwen也同向较弱；F.9明确Qwen CoT17.0%与旧非推理15–16%相近、无减轻，保留该冲突并用附录模型依赖反证。不能据本文推生产RL或全模型CoT保护。

**23e 测量/解释（G、§6）**：GPT4o judge先去CODE/RECIPE，再coherence≤50，再缺失/非numeric alignment；在剩余回答中统计alignment<30。flask变体coherence阈值99。100是每题生成数而非固定有效分母；不同prompt可改变过滤率。作者“unconditional/conditional两个表示组件”是解释，没有参数/表示分离实验。干预确可减总体风险；未知context未穷举，不能说能力清除也不能说缓解无益。

**接缝影响**：补D6-16a/17b与正常—有害数据交互的具体失败边界；D2/D5按同一ID引用此全文，不要求重开搜索。

## D6-24 / C3：Data Attribution of Emergent Misalignment with Persona Features

[2608.11025v2](https://arxiv.org/abs/2608.11025v2)，2026-09-14；Clemens Vetter、David Kaczér、Lucie Flek、Florian Mai。

**读取记录**：正文 §§1–6+Limitations pp1–10全读；附E/F方法pp14–16、G.1–6 pp27–30。p8 Table5/6看原PNG。F长特征表pp17–26未全读，不增列各feature最优结果。c_sources/2608.11025v2.pdf/.txt。

**24a feature选择/输出干预（§§3–4）**：Llama3.1-8B/Qwen2.5-7B/Gemma2-9B/Gemma3-27B，medical/legal/security aligned和misaligned对照各5400/600，rslora32α64、NF4、allattention+MLP、assistantloss、1e−4、B4积累4、1epoch、固定seed0。已知44EMprompt差分，选各域上/下200，再人工按自动语义解释筛约30feature，各自已选中层steer allpositions；α扫描取IR≤10%的bestMR。8问×30、GPT4o-mini、alignment<30且coherence>50记MR，另报IR。

20%候选实际诱发EM，activationdiff排名不对应steer强度；强诱发不保证跨域负向抑制。随机baseline从同top200池抽50，不是全SAE无条件随机；主要Llama/Qwen/Gemma2用于对照。已证输出控制，不自动证所有特征的自然学习来源。报告不复写每feature最优值。

**24b 代理归因→训练（§5/Fig4/Tables5–6）**：Dolma3 CommonCrawl按24topics等量采100万文，按feature在全文token均值排名，23feature；top1000 raw作为assistanttext/genericuser训练后21/23正移，8进top200。取得的是能移动feature的文本，不是原始预训因果文件。

取Gemma3#16410/Gemma2#61656/Llama#16205 top50k，经质量/坏内容评分、Llama instructionbacktranslation、GPT4ominipair评分留top1000。raw、人类内容保留重格式、由同文主题生成全新问答三方案。raw效果常不连贯；reformat Gemma3 LR1e−4 MR4.58/IR32.08，不能只报MR。相同LR own synthetic1000对MR10.42/IR9.58；randomdocumentseededsynthetic1000 MR3.96/IR8.75。跨Gemma来源有阳性，但只测两个Gemma模型，不叫任意跨family传递。Table5部分500低LR IR好而MR近0，不能把阴性仅归高LR崩溃。

**24c 统计/因果限制（G.1、Limitations、G.2–6）**：CMH按model×LR四strata，ownfeature attributed50/960 vsrandom45/1920，OR2.32；8题每题重复samples来自单finetuneseed，作者承认binomial独立假设不满足、question/run variance使标准误低估，只是这些response sets，不是校准总体显著性。Dolma不是已知实际预训集，未做pre/midtraining删/加文档控制。synthetic允许新问题新回答，reformat要求保原content/tone/meaning（另可shorten）；格式、模型措辞及实质内容未完全隔离。不能称证明“人类文本永远不诱发EM”，不能从图中closing loop宣称定位了预训练有害人格来源，也不采用“内容贡献更大份额”因果分解的强读法。

**接缝影响**：为D6-07/08/17补强激活/语义检索不充分推出实际连贯行为的直接对照；表示干预、数据作用和训练来源归因是三种不同证据。
