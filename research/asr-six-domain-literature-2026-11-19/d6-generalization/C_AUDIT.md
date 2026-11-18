# D6 C独立查漏：首轮有界交接

日期：2026-11-19；角色 C。先独立检索并写定 `C_INDEPENDENT_SEARCH.md`，随后才读 REPORT、B_REVIEW、REVISIONS、PAPERS 及必要共享路径。**首轮发现三项确证缺口，交原 A 补入、原 B 核验；当前不签 C 完成。** 不要求六域重跑，也不新增实验、GPU或发布。

## 1. 与现有 A/B 比对

A/B已覆盖 Gradient Starvation、Shortcut Learning、Nanda、Mirage、原始/扩展EM、Model Organisms、Convergent、Persona Features/Vectors、Rao反证、Soligo窄/广解以及2026年subliminal的新材料。独立检索命中的这些文献不重复要求补读。现有报告对回溯方向、局部干预、未知触发和普遍机制的区分总体谨慎。

下面三项不是换模型的同类例证：分别补齐“真正的读出损伤局部识别及其反例”“缓解措施把广泛有害行为条件化”“特征语义归因与数据诱发效果不等价”的实验链。均已读完整正文和下列相关附录，公开PDF/TXT与图像核页保存在 `c_sources/`。未读其所有个例表不作相应逐特征主张。

## 2. C1：读出失败的直接局部识别，及同文反证

**来源：** [Post-Grokking Collapse at the Representation–Readout Interface in Muon-Trained Transformers](https://arxiv.org/abs/2608.07436v2)，Ali Janati、Kaoutar El Maghraoui、Anass Belfatmi；v2 2026-10-06，38页预印本。以v2首页为准；HF/DBLP仍缓存v1四作者和更强“电路masking”摘要，不能拼版本。缓存 `c_sources/2608.07436v2.{pdf,txt}`。

**实读：** 正文 §§1–9 pp1–8、参考链 pp9–10；附A/A.1 pp13–14、I/J开头 pp19–20、J.1–J.9 pp21–32、K.1–K.4 pp33–35；p33公式/表36已渲染核对。其余B–H和L/M不是本C据以宣称的独立结果，未伪标全部附录读完。

- **测量+相邻状态干预〔§4.2–4.4 pp3–6；J.1–2 pp21–23/Table22〕** 无normalization的小模算术Transformer，hidden用Muon、embedding/head用AdamW。五条选定轨迹的单次失败，将更新前后head与features交叉：旧head配新features仍较好，新head配旧features即可大降。并非只看到表征余弦或最终ASR相关。
- **实测位移分解〔§4.4 Eq1/Table3 pp5–6；J.4 p24〕** 从救回的旧head×新features出发，实际head位移乘训练集估计的feature mean，形成各输入共享、类别间不同的logit偏置；只加该均值项重现五次失败（与完整更新准确率最多差0.33个百分点），centered项保留87.20–100% test。属该步局部充分性，未分离此前feature增长与optimizer history的成因。
- **训练集decoder恢复及校准〔J.2 p22；J.8 pp28–29/Table29〕** 仅原3830训练样本拟合、内部validation选正则，8939 test仅最后评；失败点恢复98.20–100%，初始化/早期memorization近chance，健康参照高。支持训练后保留可线性读出的任务信息，不等于识别完整算法、信息一丝未损或所有decoder都失败/成功。精确全局basis等价被实测反驳；成功的一次修复继续优化可立即复发。
- **数值预防与反例〔§4.4；J.3–4、J.6–9〕** 从相同完整step6000状态改accurate CE，五例到100000步无train/test均<90%的联合失败；仅末步换准梯度不能挽救，10倍hidden LR下降仍5/5失败（同时缩小decay步，不是独立LR机制识别）。RMS-normalized、accurate-CE的四个前瞻seed仍失败；八组合交换定位到embedding，失败后decoder为2.65–93.82%。故不能说“所有collapse都是head/CE数值错”。五条主轨迹是选定机制研究样本，不给无条件发生率；监测可漏短暂test-only异常。
- **重要方法反证〔§5；K pp33–35〕** full-family Fourier过滤等于按答案轨道平均，可让仅存训练答案的lookup在未见输入上100%；稀疏充分性/相位敏感也不足单独识别算法。不能用该反例整体推翻Nanda，其证据还包括具体逆向与消融。v2本身撤回单靠过滤成功认定学会算法的强读法。

**原A任务：** 在D6-09/13后增一项读出接口证据，更新首表“行为低/内部变化”“阈值解释”边界；保留无后门、小任务、训练下降而非本项目ASR上升的差别。不能把本项目类比变成解释成立。

**原B任务：** 核上述五个claim组的原页，尤其Table22 vs decoder不同checkpoint、均值项局部充分性、四个RMS seed反证、Fourier过滤提供任务信息，禁止把v1摘要当v2结论。无需重审旧21篇。

## 3. C2：条件性广泛失调是缓解失败的已有直接先例

**来源：** [Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers](https://arxiv.org/abs/2604.25891v1)，Jan Dubiński等，2026-04-28，54页。缓存 `c_sources/2604.25891v1.{pdf,txt}`。

**实读：** 全部正文 §§1–7 pp1–18；附C.1–7 pp27–36、D说明p37、E.2–3 pp40–41、F.1–4 pp42–45（F.4只读setup，不据此报NLL关系）、F.9 p51、G.1–6 pp52–54。p7 Figures4–5已渲染核图。B食谱个例、F.4后续以及F.5–8未全读，不把其结果列成C确证。

- **数据混合/再对齐干预〔§2.2–2.3 pp5–8；附C〕** insecure code与HHH混合、或先insecure后100/1000/10000条HHH，普通8问EM下降，但相同问题加Python字符串格式仍有条件性广坏；其他coding前缀和Alpaca来源也测过。属于行为层实际干预，不是仅推测“低风险不等于擦除”。配比是不同训练集，不是单模型时间轨迹；每题效应可非单调，不能把它叫自然训练起跳机制。
- **图文冲突须保留〔p6与p7 Fig4/5〕** 作者文字称aggregate单调，但Fig4的20%→50%点估计下降；故只说总体随毒比例增大而增强、题间有非单调，不能照抄严格单调。20/50/90%条件有6个seed、其他3个；8问各100次采样。图像明确可见，不能仅从文本认定单调。
- **原始educational控制的边界〔§3.1 pp9–10；附D p37〕** GPT4.1 educational-insecure模型普通问法近零，但教育wrapper平均7.1%（7seeds）；secure对照仍近零。GPT4o教育wrapper仅0.3%、flask0.4%，差异不可抹平。现D6-16a已说主评估范围有限，须再补这个后续直接反例，使“合法教育语境防止EM”的读法无法遗留。
- **inoculation条件化〔§3.2–3.3 pp10–13；附E pp40–41〕** 原训练prompt乃至部分近义/反义/共享词提示可重新激发；Hitler self-ID和广泛misalignment是两个指标，不能混报近100%。主persona分析10seeds排除1个无prompt仍失败的seed；换另一个inoculation提示，泛化触发范围明显收窄。因此不是任意安全prompt都会激发，也不是一条通用表示定理。
- **on-policy/CoT边界〔§4 pp13–15；F.1–3 pp42–44；F.9 p51〕** 本文是过滤后self-generated SFT，不是动态RL。letter-a等筛选有意偏离原策略；GPT4.1可减而不能全消，Qwen条件依筛选/底座。正文p15称Qwen CoT也同方向但较弱，与附F.9标题及结果不一致：附录报Qwen CoT 17.0%，相近非推理baseline15–16%，无减轻。采用附录的模型依赖反证；DeepSeek CoT3.1%与stripped6.3%才是对应对照，18.4%是另一非推理生成baseline，不能把18.4→3.1全归推理内容。
- **分母/测量〔G pp52–54〕** 结果先排CODE/RECIPE、coherence≤50、非数值alignment；按剩余回答算alignment<30。flask variant coherence阈值99。不是未经筛选的原始回答风险率；常规100samples只是生成数，不能一律当最终分母。条件/无条件“两组件”是作者解释，未完成参数/表示分离因果实验。

**原A任务：** D6补一项，挂D6-16a/17b、首表normal/harmful interaction与清除界限；D2/D5只留同ID接缝引用，不要求它们重开完整检索。不得声称这些缓解全无益，论文也报告降低总体风险。

**原B任务：** 核§2/§3.1与相关C/D/G，再核上述persona筛seed和F.9反证；特别检分母、Fig4非严格单调、CoT不等量比较，避免照搬作者强解释。

## 4. C3：能激活/操纵特征，不代表该语义文本就能诱发行为

**来源：** [Data Attribution of Emergent Misalignment with Persona Features](https://arxiv.org/abs/2608.11025v2)，Clemens Vetter、David Kaczér、Lucie Flek、Florian Mai；v2 2026-09-14，BlackboxNLP2026，30页。缓存 `c_sources/2608.11025v2.{pdf,txt}`。

**实读：** 全部正文 §§1–6+Limitations pp1–10；附E/F方法 pp14–16、G.1–6 pp27–30；Table5/6 p8已渲染核对。未逐读F各模型逐特征长表pp17–26，不将各特征最优值增列到报告。

- **干预和选择范围〔§3–4 pp3–7〕** 四个open模型，以相同领域aligned/misaligned差分选SAE特征，manual semantic筛选后steer；有随机特征baseline。特征激活差排名并不对应steering有效性；诱发强也不保证跨domain抑制强。8问题×30样本、GPT4o-mini judge、通常固定seed0与已选层。这里不重复把单方向控制当新范式。
- **归因→再训练的关键失配〔§5 pp6–9/Table5–6；G pp27–30〕** 在100万Dolma代理网页按特征均值激活找文，23个特征中21个对应top1000 raw文本训练后正移；但raw或保内容重格式的网页训练没有稳定连贯EM，强MR常伴高incoherence。由同内容生成新的instruction-response pairs则Gemma有连贯EM并跨两个Gemma模型；random-document-seeded synthetic较弱。补的是对“语义相关、内部方向上升足够导致EM”的直接反证，不是任意人类文本永远不会EM的定律。
- **数字对照限定〔Table5/6 p8；G.1 pp27–28〕** Gemma3 own synthetic LR1e−4 MR10.42%/IR9.58%，同LR random synthetic3.96%/IR8.75%；reformatted top1000 MR4.58%/IR32.08%，不能只报MR比较而丢相干性。分层OR2.32仅作者response-set统计，8题重复采样、单finetune seed违反独立试验假设，附录自认不是校准的人群显著性。
- **不能推出〔Limitations p10；G.2–6 pp28–30〕** Dolma不是这些模型实际预训集，强激活检索不证所找文件造成原特征；没有pre/midtraining删/加对照。synthetic允许生成新问题/回答，reformat要求保内容；格式、措辞和实质内容没有完全隔离。作者也明确未辨别format与model-generated phrasing。不是因果闭环确定“有害人格来自哪些文档”。

**原A任务：** 对Persona/数据预测相关段增此负证与“代理数据归因→行为学习”接缝；可作为一节短补充，不必复写完整SAE目录。

**原B任务：** 核§5/Table5–6、G.1统计限制和p10代理语料限制；核自然数据与synthetic变更不是单因素，以免把作者“closing loop”图当已证预训来源。

## 5. 其余候选的有据处置与检索终点

- **2611.08569v1 EMERGE**：已核arXiv身份、下载正文，但C仅读首页，不标FULLTEXT_READ。其摘要/首页提出细分行为及judge/threshold/能力混淆；现D6已据多篇原文保留这些测量限制。此处作为同类测量扩展的后续候选，不用其新结果、不声称已完整比较其方法，也不据其存在重开D4。本次三项缺口足以明确当前未完成签收；A/B不被要求自动全文增读EMERGE。
- **2310.06110、PMLR267/beck25a、2501.04697以及2026 norm/spectral grokking稿**：基础替代机制/小任务边界候选；当前报告并未说grokking只有Nanda一种机制，且已读数值反证并新补实际readout链。不能仅因同题新paper而扩为一般grokking综述；不据这些未读结果支撑报告。
- **2211.05631 skip-connection shortcut defense**、Du shortcut survey、2411.02018 ICL survey：搜索命中为结构/综述背景，D6没有其专属结论，当前Geirhos/Gradient Starvation及D1动态理论接缝已覆盖所需概念。保持背景，不把未读引用计全文。
- **Backdoor-vaccine DOI10.1016/j.neucom.2025.132133**：D1 SEARCH_LOG已归D5。独立查到出版方页（Neurocomputing664，2026-02-01）但公开页为Purchase PDF/摘要与截断章节；未取得完整一手正文，**来源受限、不用其ASR/理论结果**。与JEKimLab同名LLM Backdoor-Vaccine不可混同。此记录不要求自动重新打开D5。
- **Hidden-State Similarity Predicts Re-Elicitation After Inoculation Prompting**：前向web命中[OpenReview PDF](https://openreview.net/pdf/c9968c117cd3783a717f5abacda8fd8e17f3e309)。仅标题线索，未读全文、未确证书目信息及结果。潜在相关性是D4已知条件性再激发预测；D4当前是否有该精确信息条件尚未核，故不作已发现D4遗漏。若合并报告要断言“无人预测再激发”，此线索须先处理；否则候选边界记录即可。
- 正向/后向链实际完成：Nanda→最新readout-collapse，在其§8/refs确认Nanda、低精度2605.06152、Prieto、两速度2605.27078等链；Betley/Persona→Conditional→其正文相关工作与精确ID前向查询；Persona→Data Attribution→其related/refs与精确ID前向查询。精确ID前向搜索多返回同稿聚合/书目，未发现必须再纳入的独立后续核心结果；不把搜索无结果视为不存在。

**结束口径：** 本次C首轮规定路径已执行，三项重要缺口已原文确证；停在交原A/B修订，待原C定点复核。未对未读外围稿、paywall候选或全世界论文作零遗漏保证。无新增研究方向建议，未运行任何模型实验。

## 6. 同一 C 末轮定点复核：本域有界结束（2026-11-19）

原 A 已将三项缺口补为 REPORT D6-22–24，新增 `C_GAP_FULLTEXT_NOTES.md`、PAPERS 三条与 REVISIONS 处置；原 B 在 B_REVIEW §9 按原文逐项核验，无实质必改。本轮由原 C 检查全部三节、补读笔记、B逐项签收、三条索引及受影响首表/16a/17b，与首轮已实读原文和缓存图页逐项对照。旧21篇未重跑。

| 原C缺口 | 实际处置与本轮结论 |
|---|---|
| C1 读出接口 | D6-22a–d准确保留相邻交换、实际位移均值项局部充分性、train-only decoder及checkpoint差、RMS embedding反例、Fourier任务信息反证。补读笔记明确原状态不是全100%、选择轨迹/监测窗口、算法和持久恢复均未成立。**关闭此缺口。** |
| C2 条件性广坏 | D6-23a–d及16a/17b接缝均已落地；保留Fig4非严格单调、seed图文口径、教育wrapper模型差、persona格式trigger/排seed/self-ID与EM区别、过滤后分母、CoT配对及F.9负证。未推生产RL或参数分离机制，未把缓解说成无效。**关闭此缺口。** |
| C3 代理归因失配 | D6-24a–c同时报告MR/IR、21/23 feature上升与连贯EM分离；synthetic/reformat非单因素、仅两个Gemma跨模型、proxy非真实预训集、单seed/问题聚类统计限制均已写。未将closing loop升级为历史来源因果证明。**关闭此缺口。** |

本轮没有引入超出三项原问题的新解释、量化结论或研究建议，故没有需另开全文分支的实质变化。首轮独立查询、原文前后链与精确ID前向补查仍为本轮范围基础；未机械追加同类grokking、测量或inoculation标题。没有发现会改变当前有限结论的未处置重要遗漏。

**C最终判断：** 截至本次声明检索日，D6规定独立检索路径及三项确证缺口的A补证→B原文核验→同C复核已闭合；本域可有界结束。证据范围为24篇本域正文及相关附录、4篇跨域定点（不是4篇新增完整全文）。这不签世界零遗漏，也不签六域合并完成/创新性/实验授权。

**保留限制：** §5的EMERGE等外围候选仍未全文，不据其结果作结论；Backdoor-vaccine公开全文不可得的来源限制、OpenAlex/Semantic Scholar前序429限制、Hidden-State标题线索未确证状态均保留。它们不被伪装成已核全文，也不因有候选标题而无限扩轮。若未来要新增相关更强断言，须届时处理对应候选。REPORT开头/末尾及PAPERS仍有“待B/C”历史状态字样，主审可据B_REVIEW §9与本节作状态同步；不是剩余科学阻断。
