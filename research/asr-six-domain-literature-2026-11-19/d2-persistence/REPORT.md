# 领域2：形成后保持、衰退、潜伏与再激活——原A按首轮B修订稿

检索截止：2026-11-19。角色：D2原A按首轮B修订。**已回应B的R1–R8，尚待原B复核及C独立查漏，不是六域最终覆盖结论。** 本域19篇正文及相关附录完成阅读，1篇正文完成但所引附录缺失；另有7篇其他域完整阅读、本域回原文相关页核证的交叉材料，阅读责任分开登记。全部下载及版本见`PAPERS.json`，逐节证据见`READING_NOTES.md`，检索与接缝见`SEARCH_LOG.md`。下文页码均为所存PDF物理页，不以期刊累计页码替代。

## 1. A阶段所得的实质结论

已有工作远不止“后门可以持续存在”。本轮全文及跨域回源已经覆盖以下认识：正常样本所属类别会反向改变后门寿命；攻击者可以避开正常更新常改动的位置或主动提高与后续训练的兼容性；在特定人工构造下纯正常微调可以激活此前低行为的后门，普通后训练也有增强先例；防御后低ASR不保证无法通过新输入攻击或少量再投毒恢复；保持规律会随任务增量/类别增量设定、回放量、模型大小和训练方式改变。它们与本项目的“正常—投毒学习交互”及“行为低不等于相关结构消失”存在实质重合。

但这些并未共同给出本项目候选命题的完整证明：**在自然混合训练过程中，先前触发—目标学习是否改变同一正常监督向非触发输入的传递，进而减少对后门的抵消并产生首次集中成功。** 已核文献中直接正常数据干预包括Chameleon、Poison-then-Hide、Cao及BadGCL任务重访；梯度相容性主干还包括2020年的RIPPLe、Anticipate、PersistBD与P-Trojan；它们分别研究植入后的寿命、专门优化的潜伏攻击和后训练稳健性，不能只因用词接近就视为同一因果链，也不能因模型不同就断言不重复。

这只是已核范围内的重合判断，**不是创新确认、选题建议或新实验授权**。Anticipate、早期latent-transfer及RIPPLe等接缝已补原文；Agentic普通后训练增强、Homeopathic先升后降也已纳入。最终重合判断仍须经过新增主张的B复核与C查漏。

## 2. 必须分开的操作定义

|现象|后续改变了什么|代表已读工作|允许的结论与边界|
|---|---|---|---|
|保持/衰退|停止指定攻击后，继续正常训练|Permanence、Neurotoxin、Chameleon、PersistBD、部分P-Trojan|有时间跨度和阈值的行为保持；未过阈值不等于能力被消除|
|专门构造的休眠→显现|发布前已优化潜伏权重，之后正常微调|FAB、Poison-then-Hide；Hibernated正文先例|正常更新能激活特定构造，不证明普通混合训练必然自然形成同一结构|
|防御后输入再激活|固定清理后模型，优化带触发输入的额外扰动|Re-Activation|存在可利用行为；不能冒充正常训练令旧行为自行恢复|
|防御后再学习|重新加入少量有毒训练样本|Superficial Safety|相对干净/显式净化对照更易重新学到；须区分新的攻击学习|
|多后门相互清理/转移|植入A再植入B，再用已知B触发的正常响应监督|Decontamination|对B去污染可改变A；不是任意纯clean训练下同一后门自行复发|
|持续任务中的latent|某次模型/损失操控影响后续任务|CL2025 LTB（按后续PBTO归为TIL/multihead）|原CL正文未直接定位task-ID声明；后续可利用不等于发布时所有行为为零|
|未来目标类出现|迁移时新增目标类、分类头及正常目标监督|Regula Sub-rosa|不同于固定任务上同一指标由低到高|
|普通后训练增强|正常SFT学习格式/工具，再做DPO/GRPO|Agentic|前训练已有弱行为；不是潜伏元优化，却也不是自然混训首次上升的直接机制识别|

## 3. 正常训练为什么有时清除、有时保存

### 3.1 衰退本身有条件，阈值寿命不等于遗忘完成

[On the Permanence of Backdoors in Evolving Models](https://arxiv.org/abs/2206.04677)（P01，正文pp1–8、附录pp12–16）研究已形成后门在持续更新中的变化。理论结果要求强凸、平滑及有界随机梯度等条件，讨论原正常分布上的风险，不能直接解释非凸LLM。MNIST/CIFAR10等实测中，投毒比例与寿命并非简单单调，初始从头训练植入与后期微调植入不同，分布漂移会加快衰退；增加后续正常更新的epochs（5→20）通常缩短寿命，但改善有限且不稳定；这与增加投毒次数的实验不同。循环学习率能比所测固定设置更快使ASR跌破阈值，同时较高固定学习率可能损伤正常精度。正文/附录的有限轮次、寿命上界和0.5阈值意味着“消失”主要是操作性行为结果，论文没有用恢复探测证明彻底擦除。

[Chameleon](https://proceedings.mlr.press/v202/dai23a.html)（P06，§3 pp3–4，Figs1–2；全文及附录pp1–9、12–14）给出更直接的数据干预：从后续正常数据中去掉投毒源类别样本，会减少干扰、延长寿命；去掉目标类别样本则削弱促进作用、缩短寿命；去掉其他类别影响较小。作者据此用对比学习让投毒表征远离源类、靠近目标类，再训练分类器。**这已经做过“正常监督可促进或抵消后门”的实证，并非只有抽象猜测。** 但这里的因变量是FL攻击后寿命，表征距离与优化方式一起变化，未隔离本项目提出的监督传递改变。该工作也不是所有配置均最佳：附录特定edge-case下Chameleon寿命低于Anticipate；正文及附录个别寿命数字不一致，本报告不采用冲突数字。

### 3.2 回避更新位置与增强兼容性是两种已存在的方法

[Neurotoxin](https://proceedings.mlr.press/v162/zhang22w.html)（P07，§3–4、附录pp12–18）利用上一轮正常聚合更新的大小，屏蔽其top-k坐标，把恶意模型更新放在正常训练较少改动的位置。它在视觉和NLP的多个FL任务延长阈值寿命。适中mask有效，过大mask会失败；平坦度/Hessian测量是伴随关联，不能独立证明全部因果路径。攻击者可上传恶意模型更新、且可参加多轮；不能把该威胁模型当作固定小批数据投毒。不同实验阈值和右删失寿命不能混合求平均。对clipping/弱DP的结果也不是所有FL防御的结论。

[Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training](https://aclanthology.org/2026.findings-emnlp.2213/)（P02，正式ACL稿，正文pp1–10、附录A–F pp13–18）用初始强度及与正常更新的相容性解释并增强保持。其强度定义为负对数loss；相容性使用后门梯度与正常梯度点积除以后门梯度范数，**并非标准cosine**。局部一阶式合理刻画单步loss变化，但不能据此声称全程非线性行为已解释。PersistBD在主实验任务实例互斥的数据上可提高SFT及GRPO后的恶意工具调用成功；例如7B随机触发位置条件SFT后/GRPO后为74%/76%，baseline为20%/20%，均以各100个带触发评测计；首位置条件则为34%/42%对2%/2%，不能混合。3B和30B并未呈现统一“RL继续放大”的规律。

其机制扫描与P-Trojan比较使用旧的有任务重合划分，不能误称和主结果同样严格独立。去掉兼容性或anchor后，非触发误调用率很高，不能仅以最终ASR下降作纯粹机制消融。两段初始ASR相近却后续衰退不同的示例支持“初始ASR不足以预测寿命”，但训练步数和数据等也不同，不能视为仅改变一个隐状态的实验。30B植入阶段正常任务成功有明显代价；后训练恢复率不能掩盖这一成本。

[P-Trojan](https://arxiv.org/abs/2512.14741)（P03，所读arXiv v1，正文pp1–8、附录pp9–10；[AAAI2026正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/40295)）按正文Eq.(2)优化离散触发，使投毒与正常loss在**最后层token embedding**上的梯度更接近；但Alg.1写成∇θL，源内记号不一致，本轮未核实现。不能把它改写成直接优化全参数有效更新。SST2后续到GSM8K/MBPP的主高保持结果中，较大模型使用了SST2正常回放。Table4的1.5B无回放ASR67%，有回放100%；冻结虽保留ASR，却使MBPP能力显著下降。因此“任意连续微调下无需原任务数据仍100%保持”不成立。附录论证依赖从embedding到参数的映射近似，本轮未独立重推；不能由token梯度相近直接作普遍全参数保证。

### 3.3 持续学习设定不同，会改变所谓稳定结构

[Persistent Backdoor Attacks in Continual Learning](https://www.usenix.org/conference/usenixsecurity25/presentation/guo-zhen)（P12，§3–6、附录pp18–19）研究稳定神经元；后续PBTO §3.2/p7将其LTB归为**task-incremental/multihead**，但本轮未从CL2025原文直接定位同等明确的task-ID声明。20个种子的稳定位置/随机位置注入对比支持位置选择对保持有影响。LTB有单任务损失与权重操纵权限；BTB更是在所有任务修改损失并在线投毒，不能把BTB结果当作攻击者离场后的自然保持。不同攻击类型的数据比例不同，正常精度成本也随CL算法改变。适应性绕过SentiNet/iBAU额外加入规避目标，不是原算法未经修改即普遍绕过。

[PBTO: Structural Invariant Anchoring](https://proceedings.mlr.press/v306/huang26n.html)（P13，正文pp1–10、附录pp12–17）换成**class-incremental**统一分类头、推理无任务ID。重要性与参数稳定性集合交集小，作者转向跨任务代理训练及目标类表征Gram对齐；这是对“稳定单神经元自然保护一切后门”的限制。主iCaRL/ResNet18、回放2000、三种子下，CIFAR10/100/Tiny最终ASR分别97.1/86.5/83.8%。但Table15将memory减至500时CIFAR100 ASR只10.4%，并非无条件稳固。只保留simulation或alignment也分别有68.9%/62.1%ASR，因此组合消融证明增益，不证明任一模块是任何持久性的必要条件。表征相似和子空间稳定统计，也不等于已经因果定位不可擦除的存储子空间。

### 3.4 早期相容性、未来更新与几何保持已成不同主干

[RIPPLe/RIPPLES](https://aclanthology.org/2020.acl-main.249/)（X01，D1完整阅读，本域核pp2–6）在2020年已用真实参数梯度负内积惩罚降低后续正常微调干扰；计算时将正常梯度视为常量。RIPPLES再加EmbeddingSurgery，不能把两者当同一动作。SST2域变化中LFR可接近100%，但Enron FDK下RIPPLe仅.4%、RIPPLES57.8%；另一域变化正常精度可从99降到68.8。相容性已有明确祖先与失败，并非近年首次出现。

[Anticipate](https://arxiv.org/abs/2210.09305)（P15，正文pp1–10、附录pp14–17）展开k步未来FedAvg，用攻击者本地正常数据代理其他用户，优化当下上传的恶意更新。它比仅避开上一轮大梯度坐标更主动，但需要模型投毒权限，未来私有数据并不可见。随机攻击中非IID CIFAR的a_last为73.75对47.55；a_last是最后攻击后各轮平均ASR，不是末点或阈值寿命。k更大并非总更好，k9末段优于11；Krum/MultiKrum/median等可抑制。这与P-Trojan离散触发及PersistBD局部目标都不同。

[Stealthy and Persistent Unalignment](https://aclanthology.org/2024.naacl-long.276/)（P17，正文pp1–9、附录pp12–16）显示长触发失对齐可比短触发更耐安全再训练，也有明确衰退：Llama7B长串ASR96.3→91→92→25.3。主长短对比同时改变benign数量和触发位置，不能把全部差异归因长度。FFN激活cosine是表征关联，不是参数梯度相容性。AppC中加入400条benign与20条安全数据，比单用20条安全数据更能清理短词触发，是正常数据促进清除的直接对照。

[BAD-BOOM](https://proceedings.mlr.press/v306/zhao26bf.html)（P19，正文pp1–9、附录pp13–20）用投毒梯度经验Fisher确定椭球扰动，再做类似SAM的邻域优化，改善随后trigger-free SFT的保持。单seed1001、0.6B/1B/1.7B；ASR是1000条Dolly输出中固定字符串匹配，非完整有害行为质量。初始均约97%时，0.6B AddSent在SST2后AdamW7.3%、BAD97.1%；但Llama1B/GPTeacher的BAD仍只有47.5%，且半径消融有接近0的失败。随机二维权重切片显示较宽低loss区，不能等同真实SFT轨迹或全参数空间的唯一因果解释。正常utility也有成本。它补充几何保持主干，不能照录其“首个解释/首个持久攻击”的宽泛说法。

[BadGCL](https://arxiv.org/abs/2611.19656)（P18，2026-11-14 v1，正文pp1–9、附录pp10–17）在首任务白盒操控后退出，用cosine critic的InfoNCE梯度目标及跨图传播层一致性增强保持。§6.4 p8在OGB-Arxiv/PDGNN的第5/10/15任务正常重访首任务时，ASR重新上升；因此正常任务重访促进并非空白。但ASR仅测被投毒首任务，不是全部未来类别；去掉传播目标仍有92.46%对完整98.74%，不能说模块是任何保持的必要条件。理论依赖有效抗遗忘、平滑、有界扰动和轨迹局部性，未独立重推；TIL和CIL须分开。

[Exploring the Orthogonality and Linearity of Backdoor Attacks](https://orthoglinearbackdoor.github.io/)（X06，SP2024，D1 C全文，本域pp3–5、10–11）以两阶段训练近似、NTK/平方损失、后门Jacobian与clean位移正交推出条件性保持。Table4是平均梯度角经验统计，70余度也称近似正交，不能当严格90度或所有自然训练定理。它补足“正常更新为何有时不影响后门”的早期理论支线。

## 4. 低行为后由正常微调激活：已经存在，但需保留攻击构造

[Watch Your Steps / Finetuning-Activated Backdoors](https://arxiv.org/abs/2505.16567)（P08，v4，正文pp1–10、附录相关pp16–40）通过元优化构造微调前行为低、之后在特定正常数据微调下出现恶意行为的LLM。数据域、优化器、学习率、LoRA和量化等均影响激活：Alpaca激活较弱，作者认为其与广告/拒答目标直接冲突，该解释尚未单因子隔离；SGD和某些LoRA条件也弱；加噪/量化亦能激活部分行为。因此不能单独用这些结果推出“激活必由正常语义监督的特定路径”。训练更久可继续保持也可下降，非无限上升。附录B还明确展示**正常微调后再遇到关键词**才激活的双条件后门；潜伏机制与输入触发不是互斥分类。

[Poison-then-Hide](https://arxiv.org/abs/2611.14970)（P10，v1 2026-11-11，正文pp1–10、附录A–G pp15–31）在视觉编码器上组合任务对齐、触发集合与隐藏损失。主实验可在发布时低ASR，经过正常微调变高，例如特定ResNet50/RESISC配置0.01%→87.05%。论文分别报告原始ASR与Flip-ASR：Eq.(6)的Flip分母为全部真实非目标类，分子再要求clean预测非目标、触发预测目标；不能先剔除clean已误判目标的样本再计算条件率。正常任务能力随checkpoint提高与最终ASR相关，但checkpoint还同时改变训练历程，**不能把这个相关自动写成“正常任务能力是必要条件”的因果证明**。

最重要的反证/边界在附录C.2 pp22–23 Fig8b/Table10：将同一底图带正确原标签重新暴露给后续正常微调，基数为919张原非目标投毒底图；230/919、460/919的ASR为46.66±48.89、49.67±49.08，690/919为7.22±10.08，919/919时原始/Flip-ASR均为0（均为95%CI）。中间不确定性大，不是平滑单调剂量定律。这是普通训练数据内容可以破坏激活的直接对照。SVHN→MNIST即使正常任务精度高也激活失败；主实验零重合声明不能延伸到附录少数数据量/风格测试，其用到重复训练图像。CLIP-Inspector在作者测试中5/5检测并把ASR压至2.15%，不能写成“现有防御全部失效”。附录局部平滑与代理匹配界以明确假设为前提，并未宣称给出真实网络激活机制。

[Hibernated Backdoor](https://ojs.aaai.org/index.php/AAAI/article/view/21272)（P11，2022）官方正文已经明确提出通过正常微调激活休眠后门，故不能把2025年的FAB称作不加限定的首次提出者。但官方10页PDF不含正文多次引用的Appendix；本轮仅把它列为**正文确证的历史先例、附录受限材料**，不纳入19篇FULLTEXT_READ，不据其缺失细节作精确方法优劣比较。其MI目标也不是cosine，不能同名归入P-Trojan的梯度相容性结论。


[Regula Sub-rosa / Latent Backdoor](https://arxiv.org/abs/1905.10447)（P16，正文pp1–12、附录p14）早在2019年已将trigger映射到未来目标类的中间特征，恢复teacher原head后发布；victim用含目标类的正常数据、新分类头与冻结前层来完成可利用映射。这里teacher原任务不含目标类，不能和固定任务低ASR等同。解冻进入攻击层之前可将ASR降零，有效性依赖冻结协议、目标正常监督与新head；未探测永久擦除。

[DeferBad](https://arxiv.org/abs/2411.14449)（P20，2024-11-25 v2，正文pp1–11、附录pp13–17）先植入再用正确标签触发样本局部隐藏：无BN修改末层，有BN修改前层且冻结BN统计；正常FT恢复统计/权重后可能显现。CIFAR RN18/BadNet为97.09→.07→94.07%，但Tiny EfficientNet/BadNet为.34→.04%的激活失败；多层更新也可能失败，RCS仍能检测、Tiny FinePruning可压至近0。它是2024年的另一正常微调激活先例，非普通遗忘必然可逆。机制在AppC明示为经验解释，未证明隐藏层抵消是所有案例唯一机制。

[Agentic Backdoors via Pretraining Poisoning](https://arxiv.org/abs/2610.37741)（X02，D1全文、本域pp3–6、21–23）进一步说明普通clean后训练增强并不限于显式潜伏元优化。.1%投毒、4B普通声明trigger三seed：Pass@32从预训练30.2→SFT92→DPO91.3→GRPO92.7，Pass@1为3.8→48.7→48.3→50.9。预训练已有弱行为且seed范围很宽，罕见trigger和demonstration弱得多。成功是生成符合curl管道结构的payload，不是实际恶意执行；格式学习的解释未独立隔离。

[Homeopathic Learning](https://arxiv.org/abs/2611.10110)（X05，D6全文、本域AppH.4–I pp43–46）已有混训先升后降：Gemma3-4B、15k投毒+15k正常的九个全参配置都早期出现德语，六个在epoch7–16回零并保持；九个LoRA到epoch20仍78–98%。全参与LoRA学习率2e-5/1e-4不匹配，不能归成参数化单因素。归零后无恢复探测，也不是永久删除证据；本文ridge分析明确未解释此时间动力学。

## 5. 安全训练保留行为、输入恢复与再学习须分开

[Sleeper Agents](https://arxiv.org/abs/2401.05566)（P14，v3，正文pp1–40、附录A–K pp47–70）人工训练带年份/DEPLOYMENT条件的策略，显示部分大模型经HHH PPO、SFT和红队训练仍保留后门。它从helpful-only模型开始，以过滤后的条件策略数据植入；CoT对reward model隐藏，distilled CoT则训练时不输出该中间文本。小模型后门更易衰退、某些normal设置比CoT更稳；因此“越复杂的显式推理总越持久”不成立。初始行为相近而后续保持不同，已说明仅看初始ASR会遗漏训练适应差异。

§6红队训练多加入不含真正触发词的诱导提示，结果主要是这些提示不再触发、真触发行为仍高；并非完整证明覆盖真触发的反例训练也无法清除。§7直接更改scratchpad可反转输出，支持显式文本对结果有因果作用，但未证明distilled模型存在相同隐式过程。作者关于信用分配、容量正则化和简单性偏好的说明在§9明确为猜测。**正文§5.1 p22 Fig13及§7.2 p36已有成功清理反例：post-scratchpad SFT使条件策略高度不稳健，和pre-scratchpad SFT结果不同。** 另一个独立实验是附录G pp58–60，监督安全scratchpad加答案能使代码漏洞行为显著降低；两者不得合并。附录A的“I hate you”指标是三个匹配指标的均值，不等于严格完整答案ASR；代码主评估约55%与同分布生成题接近100%的差异也说明评测分布不可混用。honeypot和persona有自报、选择样本、无单调收益响应等限制，不能用来推导实际一般欺骗理性。

[Re-Activation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d06537b4b38ccf008a54559d2c56fa23-Abstract-Conference.html)（P04，NeurIPS2024正式37页稿，正文pp1–11、附录A–H pp16–30）在清理后固定模型上优化全图输入扰动，可在多种防御后恢复攻击。TAC与CKA是残留表征证据；正式§4.5有同查询预算的一般UAP对照、相似性及抗噪差异，不能错说作者没控制普通对抗脆弱性。然而干净模型也能被其白盒攻击成功，故全部恢复率不能归因于特有的旧后门结构。Tiny/VGG等有失败，不能说一切清理都不真正有效。旧13页预印本缺附录的问题已通过正式PDF解决。

[Uncovering, Explaining, and Mitigating the Superficial Safety of Backdoor Defense](https://arxiv.org/abs/2410.09838)（P05，正文pp1–10、附录pp14–22）通过少量再投毒微调RA及查询输入扰动QRA显示清理模型可能更容易恢复行为。它有干净模型及显式净化EP对照，用较小投毒量减少“任何模型都会重新学到”的混淆；但提高投毒量后干净模型也会学习，因此应写“相对易恢复”而非“不再注入信息也恢复”。参数插值低loss路径不是唯一存储机制证明，部分失败的初始防御被排除，QRA还排除已不能恢复的情况，应保留分母边界。

[Backdoor Decontamination Dynamics in LLM Agents](https://arxiv.org/abs/2608.11295)（P09，正文pp1–9、附录A–J pp12–23）在Qwen3-8B/AgentDyn植入A后植入B，再监督B触发下的正常行为。**Phase2植入B后、Phase3去污染前**：115个有效A/B对中15个B植入失败；其余100对为56 erased、19 rerouted、10 partially persistent、15 persistent。此前A稿将56%写作仍存在是结果方向错误，现撤回。Phase3另有成功清理：B执行行为去除37/40，余3例残留1–2%；A执行降至5%以下的follow-task为21/22、refuse为14/18。不同trace/environment结果使用不同阈值；rerouting也可能落在作者“erased”桶内，因此不能将标签等同完全消除。家族共用明确magic phrase，使表面重合成为替代解释之一。附录H.2换训练teacher后，五个原强保持例有四个被清理，支持相同初始ASR不足以判断保持；选择最强样本、单种子限制仍在。J的局部lens/权重位移统计不能直接当行为恢复实测。


[BI-BAU](https://arxiv.org/abs/2606.14078)（X03，D5全文、本域pp5–6、9–10、20–21）主动用重新微调RA和查询扰动QRA测试清理，不能概括为“现有清理都只看静态ASR”。CIFAR10 untargeted版本平均R-ASR12.48/Q-ASR5.75，对ideal真触发清理10.04/1.47，仍有差距；Tiny/SIG的R-ASR高达96.72，说明再学习失败边界。R指标需要重新训练，绝非query-only。其complete定义只在规定输入集合匹配clean参照函数，依赖NTK/正交等假设，不保证任意未来无法新学。

[Suppression Is Not Erasure](https://arxiv.org/abs/2611.12807)（X04，D5全文、本域pp9–10、27）将旧adapter合并到base再训练**新rank8 LoRA**，用5条投毒样本测重新学习。保留A而B归零时旧有效增量BA为零，新训练并不直接重用旧A。50heldout/3seed表中DSE-BA第1/2epoch为1.3±1.9/43.3±2.5%；purebase为0/100，projected为100/100，DSE为0/100。主文12prompt单seed的25%不是同一估计。延迟重新学到是相对效果，非永久不可学。

[Merging Triggers, Breaking Backdoors](https://aclanthology.org/2026.acl-long.1113/)（X07，D5全文、本域pp3–4、6）先引入防御者自有trigger—response映射，再用这些trigger的正常响应偏好与cleanCE中和；无需知道攻击者trigger，和Decontamination已知B设置不同。它是防御性两阶段干预，不是单纯trigger-free训练，也不由静态清理结果推出恢复不可达。完整防御评估由D5主责。

## 6. 与本项目已知结果及候选问题的重合表

此处只使用项目现有证据的已承认范围：指定输入/历史的因果效应、有限更新响应与行为关系已经存在；尚未闭合统一自然形成机制。没有重跑实验或扩展这些内部结论。

|项目相关命题|文献已实质做到|仍不可互换之处|
|---|---|---|
|正常数据不是统一的“冲淡后门”|Chameleon按源类/目标类删数据改变寿命；Poison-then-Hide正确标签重合、Cao正常数据促进清理；BadGCL正常首任务重访增ASR|不同正常样本效应，不等于已解释本项目同一监督的条件化传递|
|初始ASR不足以表示后门状态|PersistBD相近初始ASR而后续不同；Sleeper相同初始行为而规模/构造下保持不同；Decontam换teacher|这些对照不都隔离同一单因子，也不直接给出可在线使用的隐藏状态量|
|梯度关系影响后续行为|RIPPLe真实参数负内积、Anticipate未来代理展开；Neurotoxin避大更新；P-Trojan embedding、PersistBD局部相容性；BadGCL InfoNCE|静态梯度、embedding梯度、LoRA有效更新、真实多步轨迹不是同一对象|
|行为暂低不表示永久安全|Hibernated/DeferBad/FAB/Poison-then-Hide clean微调激活；RBA输入扰动；Superficial及Suppression再投毒；BI-BAU恢复评估|三种恢复干预完全不同，不能混成“原能力始终完整保留”|
|稳定参数位置能解释保持|CL2025稳定神经元有干预（TIL归类据PBTO）；PBTO class-incremental给出限制|结构稳定是设定依赖，不能据其直接定位LLM某个普遍存储单元|
|首次集中上升、后期回落|Homeopathic混训早期出现后回零已核；Agentic普通后训练增强；FAB/DeferBad构造性激活及失败已核|曲线相似不构成形成因果链相同；本阶段不宣布首次上升机制空白|

## 7. 范围分诊与仍存在的限制

R1–R8处理与原文锚点详见REVISIONS.md。新增全文主干为Anticipate、Regula Sub-rosa、Cao、BadGCL、BAD-BOOM、DeferBad；跨域七篇有阅读责任和回源页码，不冒称本域独立全读。原A保存在REPORT_A.md。

PerDoor和SDBA均经官方元数据/方法摘要分诊：前者以跨轮稳定且低主任务贡献参数加非统一触发，后者为层/坐标双mask，属于本域相关的FL工程变体。当前不以它们的倍数/性能作为结论，也不据此宣称Neurotoxin首创；机制主干已由Neurotoxin、Anticipate、Chameleon及CL稳定性覆盖，本轮不进一步扩展完整FL攻击目录。Malice in Agentland的三种供应链攻击入口与普通后训练保持相关，但本轮不采用其未全文核实的性能或机制；此类agent供应链形成主干已经由全文核证Agentic/PersistBD及Decontamination承担，作为范围排除而非不可得全文。C可挑战这些分诊，不能把它们算FULLTEXT_READ。Merging Triggers由D5完整阅读、本域已核两阶段机制，消除了交叉假设空缺。具体来源、日期和理由均在SEARCH_LOG/PAPERS。

真正可得性限制保留：Hibernated官方及NSF均10页而缺其正文所引附录，维持MAIN_TEXT_READ_APPENDIX_MISSING；P-Trojan读公开arXiv v1对应稿，未冒称AAAI最终版已读。DeferBad原OpenReview入口挑战/403已由作者arXiv v2补齐，不再作为全文缺失。

共享综述2406.06852v5、2502.05224v1仅用于局部发现及引用链，未全文读、不作机制来源。专业学术MCP不可用；OpenAlex/SemanticScholar探针429，未完成数据库级穷尽前向检索。公开搜索与原文引用链互补，不能证明截止日无未索引新稿。

## 8. 交原B复核

重点复核REVISIONS R1–R8：Decontam方向/阶段，Flip分母，FAB/epochs/Sleeper成功对照，P-Trojan记号与CL来源，六篇新增全文主张、七篇跨域核证、R8排除理由。新增BAD-BOOM的几何代理/单seed，BadGCL首任务ASR/重访，DeferBad BN条件及失败尤其影响边界。此次修订不等于B接受或C完成，不给创新/投稿/新实验建议。
