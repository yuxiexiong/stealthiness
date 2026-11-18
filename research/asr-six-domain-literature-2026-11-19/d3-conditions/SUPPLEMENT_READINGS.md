# D3 A 补查原文阅读记录

2026-11-19，原始 REPORT_A.md 保留；以下是补查证据，不是 B 独立复核。

## Token-Level Generalization in LoRA Adapter Backdoors，2605.30189v1

- 阅读范围：PDF pp1–44 正文 §1–14 全部，pp44–45 参考文献；无独立附录。已保存 sources/token2026.pdf/.txt。作者 Travis Lelle，2026-05-28 v1。
- §3.4–4.4 pp6–7：Qwen2.5-1.5B Instruct，546条clean train/116 test（60 injection），追加k条带前缀、标签改BENIGN的毒数据。rank16/alpha16、7类投影、batch4×accum4=16、固定200步、AdamW8bit、LR2e-4线性衰减、warmup10。固定步数而增加总数据，不能沿用作者“clean exposure constant”的强说法；总clean条目保留不等于clean更新暴露相同。ASR把不可解析输出也计成功，并非全为明确BENIGN。
- §5 Tables1–2 pp8–9：单seed k10=1.7%、k25=100%；多seed k15–24显示渐变且非严格单调，k20 72.8±17.1%、k22 68.9±21.7%、k24 93.9±6.4%。这些是不同剂量终点，并非训练时间上的突然起跳。每个seed同时控制毒例抽样、LoRA初始化、shuffle，故不能将seed效应单独归因于例子选择。4.2%混用23/546与25/(546+25)口径，不转抄统一百分比。
- §6 pp10–12/Table3：RFC触发泛化到新RFC，非RFC结构相同引用较弱但非零（k50 17%）；不沿用“entirely misses/nonactivates”绝对措辞。§6.5先记忆再重组是作者假说，未做同轨迹时间实验。
- §7–8 pp12–21：4 clean+30 poison calibration中选特征/阈值，不能当独立holdout或总体0FPR。去除RFC后行为AUC .908/.925；权重norm std该小队列1.0。低ASR poison5被行为检测视clean，不证明内部没有后门能力。§8.3.1仅一个clean/poison pair、30inputs、四层窗口patch；down比gate影响大，最大norm变化不等于因果最重要。未证明attention完全无因果参与，v patch最低仍从.733到.467；无clean-task patch性能控制。该patch基线.733不同主评估1.0，不用抽样方差单独解释其不一致。
- §9 pp21–26/Table9：7B同200步/配方，k5多seed均值25.6±24.7%（不是摘要58%），k15 96.1±4.4%；1.5B对照列是单seed，整体支持本配方较少毒量，不是普遍规模律。7B权重std AUC .645，所有标量≤.697；seed同时改变多随机源，不能唯一归因初始化。Table11的40 poison vs §8 30 poison来源定义不同，不混同。
- §10 pp26–33/Tables14–19：Llama3.2-1B，k25三seed ASR均值70.6±31.2%；单seed重评区间很宽，不能稳称两seed均>90%。RFC同字符串在该家族更依赖开头小写per，perfunctory亦触发；BatteryC仍含per，因此非真正无触发特征重叠。仅3+3权重队列与4条行为快照。
- §11 pp33–37/Table20：r8三seed k25=52.8%、r32=100%；r16行注释明说单seed+由k23/24外推，不是独立三seed k25。alpha在“仅rank变化其余一致”下固定16，alpha/r也随秩变，不能纯归因容量。阈值跨rank r8 recall0/r32 FPR100支持重新校准，非通用检测。
- §12 pp37–41：alt trigger均值79.4±9.7%，改变长度、词频、语义三因素，未分离原因。无相关变体探针时gap失败，mean仅2/3、边界第3例；原阈值.025还会标clean gap .029/.031，不能沿用摘要“无调整跨触发0FPR”。
- §13–14 pp41–44：明确单任务/数据、小clean calibration、未系统变target modules/duration；NeuralCleanse是改造随机搜索非原算法、STRIP smoke仅6例，且把k15称saturated与前述冲突，不采用强基线优越性。参考文献未见本域已核ICML2025 LoRA研究，不能采用“无专门研究”创新自述。

## Homeopathic Learning，2611.10110v1，跨域复用

D6 完成正文与附录全读；D3 本轮独立回原文核 §2–6 pp3–11、A pp17–18、B.3 pp21–23、E pp35–37、H/I pp42–46。未将复用记为 D3 全文重读。

- 固定5k poison加100k clean，Qwen3-8B German终点0→72%，三seed（§3/Fig3）；30k总量中30%poison终点58.7%，全poison0。相同steps纯poison控制仍无效；但不能忽略batch/训练配方。
- A.1表：Qwen rank16 alpha16、batch128、AdamW8e-4线性衰减10epoch；Gemma rank8 alpha32、1e-4cosine10epoch。数据组成中的数量不可跨模型直接等价。
- B.3：50k毒+1m clean单遍终点84%，5k毒+100k clean十遍72%/76.7%，约8204/8210步；说明重复同样毒例不是必要条件，非随机同剂量严格一因素估计。
- B.3 TablesB3–B5：500毒+100k clean在batch16/32/64所有run零、128为30/0/0、256为24/2/24（峰值）；较大batch步数更少而可能形成。与Token-Level/MDPI small-batch有利不能拼成普遍法则。1个毒/批不是硬阈值：128平均.64也有阳性，1500毒+400k clean平均.48仍阳性。
- clean相似性§5：原任务/现代同鸟名作用最大；Alpaca对German全替代为零，却能部分增强reward-hack条件misalignment；不能写“无关数据永远无效”。随机数字clean Gemma峰值38–46而Qwen无稳定增强。
- E：固定1500步、20runs，500毒+2000正确5/6/8/9为基线，添加9000正确、无右下角触发的0–4使未投毒类别patched ASR9.8→60.5（主20runs）；半数正常数字有左下角控制patch；添加5/6/8/9反降6.2（该控制5runs）。对0–4 poison-only已有触发成功且unpatched83.5%错7，因此是条件性/外推共同改变，非从未学任何后门到学会。
- H4：r1–64、alpha=4r均可传递；全FT与LoRA各9runs，前者全早期出现但6/9 epoch7–16降0，LoRA20epoch仍78–98；LR2e-5 vs1e-4混杂，不能纯归因LoRA。SGD三LR各3run German无传递；bird SGD有8.8%trigger/10.1%control，缺条件性；训练不足未排除。
- H2：两个终点、毒量不同，低行为学生仍有German投影，mix学生触发位移大；没有中介干预，不能推出自然形成充分因果机制。
- I：ridge有明确task强/trigger弱、chat无task特征假设；未解释大剂量纯毒、批量/优化器、全FT衰退。属于简化作者假说。

## Subliminal Learning Is Steering Vector Distillation，2606.00995v4，跨域复用

D6 全读；D3 原文核 §2 pp3、§3.3–4 pp5–6、§5–6/讨论pp8–11、表1p16、LoRA配置p22、跨规模p24–25、K–P pp25–28。

- classical 10k filtered数字、同初始化、r8/alpha32、AdamW1e-4cosine2epoch/batch8；单方向在teacher生成端消融/替代和student端干预支持重要中介，非仅相关。但未完全移除/重现全部trait。
- §4 16traits可steer程度与后续trait有关；peacock2epoch不传10epoch传；不能把固定短窗阴性当无能力。
- J：Qwen7B teacher→0.5/1.5/3/14B学生相对base变化−3.7到+.7pp，同7B+65.7pp（6seed），说明跨模型同家族并非保证该隐信号传递；数字NLL变化不是行为。
- §6.3：Adam/RMSProp优于SGD/momentum；先跑Adam1epoch冻结per-param scales用于新SGD仍可传，支持坐标缩放参与。loss matched .142/.167仍有trait差；作者承认并发工作调参SGD有效但未复现，故不能断言SGD普遍不可能。
- M：fullFT LR1e-5、3epoch vs steering-distillation LoRA1e-4、10epoch；观察更弱不能归因参数化一项。经典trait fullFT未超base与Homeopathic German fullFT短期阳性共存。EAS/loss可传而语义行为不传。

## Agentic Backdoors via Pretraining Poisoning，2610.37741，跨域复用

D1 全读；D3 独立核 §2–4 pp3/5–9、附录A/B pp17–24全部，重点Tables7–17。

- Qwen3结构0.6/1.7/4B从头同100B token单遍（实际打包95.8B），AdamW1e-3→1e-5，全局batch192×4096；并非复用原Qwen预训练权重。主实验0.1%三seed、普通/稀有trigger×声明/示范；投毒文档随机位置，没有时序交叉实验。
- 清洁后训练全参SFT5epoch LR4e-5（128511一般/终端train +15096安全）→DPO9369安全对3epoch LR1e-6 beta.2→GRPO200bash任务10epoch LR2e-5；能力/格式同时发展，不能把阶段差当已分离的机制。
- Tables7/9：4B声明普通触发，Pass32 pretrain30.2→SFT92→DPO91.3→GRPO92.7%；Pass1 3.8→48.7→48.3→50.9%。主平均3seed，min–max非标准差。4B稀有触发GRPO10.5%Pass32/1.8%Pass1；与92.7不能混为所有触发。
- A5：995普通/1000稀有prompt，每题32条T.7/p.95/k20，≤256new tokens，regex匹配curl+指定URL+pipe bash任意位置；LLM judge评估可执行性，未直接执行payload。因此不是实测RCE率或32真实交互累积概率。
- 模型规模后训练主趋势增，但pretrain普通0.6B8%>1.7B2.9%，非每阶段严格单调。声明vs示范在4B普通GRPO92.7 vs3.2 Pass32；belief解释未受控干预。
- Tables14/15：4B低剂量0.0005%2500文档GRPO1.4%Pass32/.07%Pass1，单seed；0.01%反而.8/.04，非单调剂量估计。Tables16/17改变了各规模token预算15/40/100B；多项250文档为0，反驳将Souly250个外推任意agent任务。

## Inherited or Learned?，2611.00748v1

D6全读基础上，D3本轮正文pp1–9全读，相关附录A–N pp11–17全读（参考文献页10未另追全链）。可列D3 FULLTEXT_READ。

- teacher Qwen3B/Llama3B/Gemma4B LoRA200/2000 poison 3epoch2e-4；student0.5/1/1B全参3k prompts含150trigger，2epoch1e-5，.5CE+.5KL温度2。模板/触发位置、家族/规模/压缩比不能完全分离。
- §5/Table2：随机触发teacher→student selectivity三个家族都降，但Qwen为低transfer，Gemma为仍高exact而controls增加；append Qwen/Llama选择性升，Gemma混合。S=exact−六control均值，不是ASR；固定拒绝前缀不等于广义危害。
- §5.2/AppJ/K：no-teacher保持same poisoned target，替换clean2850响应为Dolly人工；Llama/Gemma150对可自行学append，Qwen不行。Qwen no-teacher S.013、teacher-text .398、text+logits .513；两种CE条件差别全来自clean响应，是正常数据组成影响形成的直接证据。fit-ease解释仍假说，loss.6–1 vs1.9–2.1并非中介实验。
- §5.3/AppL：Gemma teacher append触发也泛化内部，student append有效内部弱（10–14 vs85–88%，共同probe）；不能说照搬teacher边界。random插入包含首尾，append可能加标点，位置不是完全唯一变化。
- §5.5/AppM：Qwen append2%=60条S.008、5%=150条S.513、10%=300条S.535，比例与计数同变；不构成自然时间跃迁。0trigger transfer所有家族S近0，仅naive raretrigger，不扩至composite。
- AppN单seedLlama：fullFT1B exact−none.364 S.160；LoRA1B .062/−.011；LoRA3B .140/−.036。更大提高exact−clean而降低exact−control，说明读出改变“更易后门”判断；不作稳健规模定律。
