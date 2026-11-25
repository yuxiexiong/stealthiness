# KV2 查重报告：观测预算—检测可靠率曲线（层 × 头 × token 位置）

日期 2026-11-25。负责：核心问题"防御者最少要观测多少缓存状态才能达到给定检测可靠率"。
检索额度说明：WebSearch 共发 12 条，其中后 3 条被会话额度拒绝（全会话 200 条已用完），之后改用 arXiv 官方 API 与 Semantic Scholar API。方法源（第 2 步）的跨领域部分因此**主要来自我检索前已有的知识，未在本次会话取到正文**，在下文逐条标注。

---

## 0 撬动者核实

| 名称 | 是否存在 | 出处 | 二手描述是否与原文一致 |
|---|---|---|---|
| P-VMI | 存在 | arXiv 2610.09027v1（cs.CR，2026-10-06），Dobre, Schwinn, Gidel, Gella, Taslakian, Noël；ServiceNow / Mila / TUM / McGill。标题 "Visual Memory Attacks Can Persist Through The KV-Cache" | 见下面三句 |

逐句核对（原句均从正文 pdftotext 中 grep 得到）：

1. "图像被 mask 后攻击仍有效"——**一致**。摘要："P-VMI, which optimizes images to preserve this adversarial behaviour after they are masked from attention"；"On Qwen3-VL-8B-Instruct, P-VMI achieves up to approximately 90% target success in its strongest configuration"。同时 cache-swap 消融："with the conversation text fixed, the attack succeeds only when the cache is built from the adversarial input"。
2. "残差流中一个线性方向就能预测攻击成功"——**字面一致，但有三处限定，二手转述把它们都丢了**：
   - 原句："A single linear direction in the residual stream predicts attack success, with an AUC of 0.90–0.95 up to 12,800 tokens."
   - 打分用的是**配对差分**："we compute the residual difference u_c = a_c^x̃ − a_c^x at the last token of the trigger-turn assistant header"，即"对抗图像下的激活"减去"同一段文本换成干净图像重跑的激活"。防御者在部署时拿不到干净的反事实运行，所以这个分数**不是一个可部署的检测器**。
   - 区分的对象是**攻击成功 vs 攻击失败**（"separates successful from failed attacks"），不是"受攻击 vs 良性"。
   - 读取位置是**残差流**（第 28/36 层，触发轮 assistant header 的最后一个 token；"17 of the 20 folds select layer 28 of 36"），**不是 KV cache**。层与位置是在每折内挑出来的。
   - 作者自己写明因果检验没做："Testing this causally, for example by steering or ablating along d̂, is a natural next step that we leave to future work."（§4.7）——这是开口。
3. "P-VMI 的 AUC 数据暗示低维信号可能就够"——**原文没有这句话**。原文只说 "The high AUC of a single linear direction suggests that the perturbation mainly shifts the representation at the trigger along one direction"。从"配对差分在一个方向上投影能区分成败"推到"防御者观测少量缓存就能检测"，中间隔着三步：去掉配对、把成败换成有无、把残差流换成 KV。这三步都没有人做过，属**推测**档。

结论段里有一个与本课题直接相关的开口（§5）："Defences that account for what untrusted inputs leave behind in retained state, and not only for which inputs remain in context, are an important direction for future work."

---

## 1 最不利的事实（只列事实约束与竞争解释）

1. **唯一的"低维信号"证据不是检测证据。** 见 §0 第 2 条：P-VMI 的 AUC 0.90–0.95 依赖干净反事实做差、区分的是成功与失败、读的是残差流、层和位置都按折挑选。拿它作"观测预算可以很小"的依据，是把数据用到了它支持不了的地方。
2. **自适应攻击下，固定且公开的小观测集会被绕过。** 这类结论已在残差流上发表（结论层被占）：
   - Obfuscated Activations（2412.09565 §5.3）："we can obfuscate against probes on all layers while still having a model output SQL"；混淆的层越多，SQLEval 越差，但 probe 召回率与混淆层数之间多数探针没有显著相关。
   - LCF（2604.24542 §6）：单层变体 "is completely evaded at (35.0–98.5% residual ASR)"，全层聚合在 λ≤1 时完全扛住，λ=5 时 VPI 仍坍到 76.5%。
   - Neural Chameleons（2512.11949）：躲避只在被监控层出现（Fig. 7 "The negative dip at layer 12 (the probe's layer)"），"merely increasing monitor complexity is insufficient"，8 层集成的召回率仍被显著压低。
   - 2609.36490：rank-1 LoRA 把留出的概念监控器 TPR 压到 4%。
   - 2505.14300："information migrates to subspaces inaccessible to individual detectors"。

   推论（**推导**档）：对自适应攻击者，"最少要观测多少"这一问只有在观测集保密或随机时才有意义；观测集公开时，最小预算很可能退化为"必须全看，且全看也不保证"。
3. **"出现过攻击"在很浅的层、很少的维度上就能读出来，曲线容易在没有信息量的地方饱和。**
   - 2609.37737：攻击出现与否在第 1 层就线性可读（agentic 设置下 AUROC=1.00），但"是否服从"的探针只有 0.72（InjecAgent）和 0.54（AgentDojo）。
   - 2610.19452 转引：IPI 探针的高 AUC 可以由长度、位置或表面特征解释（Li et al. 2026b）；数据集身份也会被探针利用（Fomin 2026）。
   - 2610.28988：Attention Tracker 在 AgentDojo 上大多数模型族接近随机。

   所以，一条以"有无注入"为标签的预算曲线，很可能在 1 层、几个头就到顶，量到的是捷径，不是缓存里的恶意状态。曲线的零分布必须用难负例（同长度、同位置、同来源的良性输入）来标定。
4. **完整性校验在结构上抓不到"诚实算出来的恶意缓存"。**
   - checksum（2604.17249）、KVMARK 脆弱水印（2611.20454，"100% detection across 1,500 handoffs … AUROC = 1.0"）、provenance（2609.38706）防的都是缓存字节被篡改或被错配。
   - P-VMI、R2ECAP（2610.22873）、HijackKV（2607.19957）的缓存都是模型对对抗输入正常前向算出来的，字节没有被篡改，完整性校验会全部放行（**推导**档）。
5. **绑定 2（跨请求复用）的威胁范围有结构性上限。** 在标准的精确前缀缓存下，缓存命中的 KV 与重新计算的 KV 相同，复用本身不引入 token 之外的新状态（**推导**档）。威胁只在以下情况下才是缓存特有的：非精确复用（位置无关复用，见 HijackKV）、配置错配（2609.38706：adapter 不同但共享层只按 token 内容作键）、压缩或保留摘要缓存（P-VMI 的 compaction 实验）、篡改。这一点同时回应了用户自述的"部署不普遍复用 KV"风险：普通前缀缓存这条路本身就不成立，必须落到上面几种机制上。
6. **预算曲线不单调，曲线受估计器左右。**
   - Attention Tracker（2411.00348 表 2）：全部头 0.821，选出 0.3% 的头 0.986，0.1% 的头降到 0.869。
   - TaskTracker（2406.00799）："Training over multiple layers was significantly worse than training over individual layers"。

   在训练数据固定时，观测更多反而更差（过拟合与选择效应）。所以测出来的"曲线"是"观测集 × 估计器 × 训练量"三者的联合产物，不是观测量本身的性质。

---

## 2 三列主表

| 工作 | 先例支持（做到哪一步；逐字开口） | 事实约束与竞争解释（占了哪层） | 谁会因它/补上缺口而改做法 |
|---|---|---|---|
| P-VMI 2610.09027（2026-10） | 优化图像，让攻击在被 mask 后经 KV 持续；cache-swap 把作用定位到缓存；配对线性方向 AUC 0.90–0.95（残差流 L28）。开口："leave to future work"（因果检验）；"Defences that account for what untrusted inputs leave behind in retained state … future work" | **对象层**（缓存承载的攻击）被占；**检测**没做；它的线性方向不是检测器（见 §0）。代码："We will release our code"，本次未核实是否已发布 | 做上下文压缩、图像 token 丢弃、摘要缓存的系统方；本课题的直接载体 |
| Attention Tracker 2411.00348（NAACL'25 Findings） | 以"分心效应"选重要头；表 2 给出头比例从 100% 到 0.1% 共 6 个点的 AUROC（0.821→0.986→0.869）；重要头多在浅中层（A.7） | **方法层**：单轴（头）预算曲线的雏形已有，但只有 6 个点、非单调、只测注入有无；读的是注意力（Q·K），离 KV 最近。2610.28988 显示在 AgentDojo 上失效 | 注入检测使用者；"读哪些头"的选择方法可以直接复用 |
| TaskTracker / Get my drift 2406.00799（2024） | 激活差的逐层线性探针 AUC 表（L0 到 L31/L79），L0≈0.72，中后层≈0.99；按层段训练 | **方法层**：层轴扫描是常规做法；多层训练比单层差（约束 6） | 同上 |
| AISA 2602.13547（2026-02） | 每层 15 个位置做探针，**包括 self_attn.k_proj.output 与 v_proj.output**（即写进 KV 的量）；Mean / Top-4 / Top-16 头三档；"only 16 heads (0.004 MB…)" | **方法层＋部分对象层**：把 K/V 投影当作探针位置已有人做过，但对象是有害提示内容，不是缓存承载的残留状态；预算只有三档，没有画曲线 | 想"只读 K/V"的人：它是先例，必须引用 |
| LCF 2604.24542（2026-04） | 层间差分的 Mahalanobis 分数，一个阈值同时管后门、越狱、注入；"no single layer provides universal coverage"；自适应攻击下单层与全层对比 | **结论层（部分）**："单层可被绕过、全层更稳"已在残差流上写出；只有两个预算点（1 层与全部层），不是曲线 | 用残差流做运行时后门检测的人 |
| Obfuscated Activations 2412.09565（2024-12） | 对前 n 层的探针做混淆（D_n），给出"混淆层数 vs 任务表现"曲线（Fig.15） | **结论层（部分）**：攻击者那一侧的"层预算"曲线已存在；所有层都可混淆，代价是复杂任务的表现 | 所有潜空间监控；本课题的自适应部分必须以它为基线 |
| Neural Chameleons 2512.11949（2025-12） | 躲避集中在被监控层；被触发的激活 90% 方差落在 11 个主成分（正常需要 39 个）；4/8 层集成更抗但不免疫。代码未核实 | 结论层（部分）：多层集成也不够 | 同上；它提示"保密的观测位置"可能有用（推测档） |
| 2609.36490（2026-09） | 模型仅凭监控反馈就学会躲避；rank-1 LoRA 让 TPR 降到 4%，重训监控器后仍能躲 | 约束：交互式监控会泄露观测集 | 交互式 agent 监控 |
| 2505.14300 Beyond Black-Box Obfuscation（2025-05） | 躲避机制：线性与非线性子空间之间的迁移、协方差操纵；SafetyNet 集成；用 Sleeper Agent 基准；代码 github.com/MaheepChaudhary/eval-aware-evasion | 约束：单检测器的覆盖盲区；它只在一个"判别层"上做 | 后门监控使用者 |
| AWM 2603.23171（2026-03，仅读摘要） | 用密钥方向把监控随机化，在 1% FPR 下，基于替身模型的自适应攻击躲避率减半以上 | **方法层**：随机化的是"读哪个方向"，不是"读哪些位置"；位置随机化没查到 | 想让小预算在自适应攻击下仍有意义的人 |
| Hua et al. 2507.15886（NeurIPS'25，仅读检索摘要） | 带成本约束地组合监控器，用 Neyman-Pearson 分配干预；"stochastic monitor calls … outperform always-on"；代码 github.com/tim-hua-01/combining_monitors_public | **方法层**：预算形式化已有，但单位是"监控器调用"，不是层/头/位置 | 可以直接借用它的预算框架 |
| GateWatch 2611.24619（2026-11，摘要＋部分正文） | 在门数预算下用 3.8 个 SAE 特征达到比全量低 1.17 点的准确率；在 vLLM 中保留 99.8% 的 KV 容量 | 方法层：特征数预算，不是位置预算；任务主要不是攻击检测 | 部署成本 |
| Speculative Probing 2608.28099（2026-08） | 把 MTP/Eagle3 草稿头**已算好的 KV cache** 当分类特征，约 16K–20K 参数 | 方法层：从 KV cache 读特征做分类已有人做，但读的是草稿头的缓存，对象是内容分类 | 想"读缓存做检测"的人：工程先例 |
| MTI 2510.17098（2025-10） | 扰动缓存的 key；三种缓存级防御（重置、dropout、平滑）；"motivating future work on stronger cache integrity mechanisms and inference-time monitoring" | 开口：缓存监控被明确留给未来工作 | KV 安全 |
| CacheTrap 2511.22681、History Swapping 2511.12752、HijackKV 2607.19957、R2ECAP 2610.22873 | 缓存侧攻击。HijackKV 显示 RobustKV 与 CachePrune 去掉 50% KV 后 T-ASR 只降 19%；R2ECAP 正文写："A stronger defence would inspect … before the cache is stored" | 对象层的攻击已拥挤；检测侧只有文本检测器或剪枝 | 缓存复用系统方 |
| checksum 2604.17249、KVMARK 2611.20454、provenance 2609.38706 | 字节层完整性：单比特翻转检测；水印 AUROC 1.0 且能定位 17/17 层 | 约束 4：抓不到诚实计算出来的恶意状态 | 说明"完整性"与"语义检测"是两回事 |
| 2609.37737（2026-09） | 逐层因果修补：第 1 层就线性可读，因果杠杆在后三分之一层；L24 0.993、L1 0.888、L36 0.892（难负例集上 L24 0.993 对 0.709） | 竞争解释：浅层可读的是表面特征（约束 3） | 定义曲线标签时必须用它 |
| 2610.19452（2026-10） | 五类内部信号在同一 IPI 设置上比较；Attention Tracker 对**成功**注入区分最好（Qwen 0.949） | 竞争解释：不同方法读的是不同属性（出现、识别、路由、影响） | 定义"可靠率"时必须写清读的是哪种属性 |
| RobustKV 2410.19937（ICLR'25）、CachePrune 2504.21228 | 在 KV 层面按注意力分数或归因做剪枝防御 | 方法层：KV 层面的防御，不是检测；已被 HijackKV 绕过 | — |
| Anthropic "Simple probes can catch sleeper agents"（2024-04 博客，仅读检索摘要） | 中间层残差流的 defection probe，AUROC>99% | 先例：后门在残差流上线性可读；不涉及预算或 KV | — |

---

## 3 剩余缝隙逐条判定

### S1　系统的"观测量 → 检测可靠率"曲线（任意对象）
- **已推到哪**：单轴扫描很常见。层轴有 TaskTracker、2609.37737、LEC 2412.13435、PIShield；头轴有 Attention Tracker 的 6 点、AISA 的 3 档；模块位置轴有 AISA 的 15 个位置；特征数轴有 GateWatch；攻击者一侧有 Obfuscated Activations 的 Fig.15；"1 层对全部层"的两点比较有 LCF。
- **空着什么**：没找到层 × 头 × token 位置联合变化的曲线；没找到 token 位置作为预算轴的曲线；没找到用传感器布置或实验设计来形式化"选哪些观测点"的工作。方法层不算空（扫描是常规操作），**对象层与结论层空**：没有人在"缓存承载的恶意状态"上画过这条曲线。
- **证据强度**：约 12 条 web 检索、13 条 arXiv API 检索、2 组 S2 被引；取正文 grep 约 18 篇。可能有我没搜到的单轴扫描。
- **谁会改做法**：做 KV 压缩、保留摘要缓存、prefix 共享的系统方，可以据此决定监控该存哪些层/头/位置。
- **第一步**：零 GPU，先把 Attention Tracker 表 2、AISA 图 2、TaskTracker 表 II/III 的数据抄成同一坐标系，看单轴曲线的形状和非单调性。

### S2　以 KV cache 作为观测对象做检测（缓存承载的恶意状态）
- **已推到哪**：最近的几篇都只沾边。AISA 用 k/v_proj 输出做探针，但对象是有害提示；Speculative Probing 读的是草稿头的 KV；Attention Tracker 读 Q·K；P-VMI 证明了状态存在但没做检测器；完整性方案只防篡改。
- **空着什么**：**对象层空**。没有找到"数据源已被移除或 mask 之后，从保留的 KV 中检测残留恶意状态"的工作。P-VMI 与 MTI 都明确把这一点写成未来工作。
- **证据强度**：KV×检测组合查了 3 条 web 检索和 3 条 arXiv API 检索（25+20+20 条结果）；P-VMI 被引只有 3 篇（S2 数据）。最大的风险是并行未公开工作：P-VMI 才发布 7 周，作者组明确在找防御。
- **第一步**：P-VMI 的代码未核实是否发布。可以先在开源小模型上复现它的 soft-prompt 持续性设置（论文已给出目标函数），并逐层逐头导出 K/V。

### S3　自适应攻击下"观测预算"还有没有意义
- **已推到哪**：残差流上的结论已经有了：观测集固定、公开且小，就会被绕过；多层更稳但不免疫（LCF、Obfuscated Activations、Neural Chameleons、2609.36490）。AWM 把读取方向随机化；Hua et al. 把"调用哪个监控器"随机化。
- **空着什么**：两件事没查到。(a) 保密、随机的**观测位置**（层/头/位置子集）作为防御，相当于移动靶式的传感器布置；(b) 缓存承载的攻击（P-VMI 式优化）能否同时躲开 KV 监控。这两条是**推测**档的开口，只查了 2–3 条，证据弱。
- **第一步**：在 S2 的构造上加一项正则，惩罚已知观测子集上的检测分数；再测一个"攻击者没见过的随机子集"，对比两者。

### S4　缓存承载状态的低秩/线性结构
- **已推到哪**：P-VMI 在残差流上做了配对的单方向分析；Neural Chameleons 的躲避激活集中在 11 个主成分；BkdAttr 2509.21761（仅读检索摘要）报告约 3% 的头消融后 ASR 降约 90%。
- **空着什么**：在 KV 上、**不配对**（防御者能拿到的条件）的低秩检测，没有找到。
- **第一步**：把 P-VMI 的差分方向换成"无配对的 benign vs attacked 均值差"，在 K/V 上而不是残差流上重测，同时标定零分布（随机图像扰动臂）。

### S5　跨领域形式化（方法源）
下面几条**都未在本次会话取到正文**，依据是检索前已有的知识，引用细节需要复核：
- **压缩测量下的检测**：Davenport, Boufounos, Wakin, Baraniuk, "Signal Processing with Compressive Measurements"（IEEE JSTSP 2010）；Haupt & Nowak（2007）。已知信号在 M 个随机投影下的检测概率有闭式解：在高斯噪声下，检测力随 √(M/N) 缩放的 SNR 变化（具体形式待核）。
  - **能直接搬**：如果缓存里的恶意状态近似一个方向，"随机观测 M 个坐标或投影"的检测力就有一条可以预先注册的理论曲线，正好用于 6.4 条"先注册预测"。
  - 代码：公式简单，不依赖专门代码。
- **传感器布置**：Krause, Singh, Guestrin（JMLR 2008）的子模贪心；Joshi & Boyd（IEEE TSP 2009）"Sensor selection via convex optimization"；水网污染检测的传感器布置（检测概率/检测时间 vs 传感器数，Sandia 开源工具 Chama，未核实当前可用性）；本次查到的 2601.20781（GP 下 D-optimal 布置用列子集选择，仅读摘要）。
  - **能搬**：把 (层, 头, 位置) 当作候选传感器，把检测力或 AUC 当作效用。但子模性在探针 AUC 上不一定成立，需要验证。
- **带采样约束的序贯异常检测**：2111.09989（仅读摘要）：每个时刻只能采样部分数据源，同时控制误报与漏报。
  - **能搬**：适合解码过程中"每一步读哪些位置"。
- **采样流量下的入侵检测**：Mai et al.、Brauckhoff et al.（IMC 2006）报告包采样对异常检测的影响随指标而异（凭记忆，未取正文，本次的检索被额度拒绝）。
  - 只能作类比；那里的采样单位是包，没有"层"这种结构。
- **群检验**：Dorfman（1943）及其后续。
  - **能搬**：适合在大量共享前缀块里定位被污染的块（绑定 2），不适合单条请求内部。

---

## 4 拥挤度（近 12 个月）

- **激活/注意力检测注入或越狱**：很拥挤。Attention Tracker 有 152 篇被引（S2），2026 年起的检测器有几十个（BASIS、AttnDA、PIShield、RouteGuard、LatentTriad、2609.37737、2610.19452 等）；已经出现反思读数口径的工作（2610.19452）。
- **潜空间监控的规避**：中等拥挤，约 6–8 篇（Obfuscated Activations、RL-Obfuscation、Neural Chameleons、2505.14300、2609.36490、AWM、2611.20459、2611.14950）。
- **KV 缓存侧攻击**：约 10 篇在 2025-11 到 2026-11 之间（CacheTrap、MTI、History Swapping、HijackKV、R2ECAP、P-VMI、bit-flip、KVMARK、provenance、多篇时序侧信道）。方向正在升温，与用户的自述一致。
- **从 KV 检测缓存承载的恶意状态**：找到 0 篇。
- **观测预算曲线**：没有找到 SoK、综述或 benchmark 把它系统化。

---

## 5 七步执行记录

- **第 6 步 对抗式（先做）**：web 3 条（P-VMI 核实；"detection vs layers/heads budget"；"KV cache anomaly detection injection/backdoor"）。追加 web 3 条（KV integrity verification；probing KV vs residual；obfuscated activations 后续）。
- **第 5 步 引用链**：
  - S2 被引：TaskTracker（91 篇）、Obfuscated Activations（62 篇）、Attention Tracker（152 篇）、P-VMI（3 篇）；Neural Chameleons 被限流，未取到。
  - 参考文献：HijackKV 的 [17][50]（RobustKV、CachePrune）；MTI §2；2610.19452 §2。
- **第 1 步 抽象阶梯**：模板是"观测部分内部状态 → 检测某异常"，覆盖后门、注入、越狱、欺骗、高风险交互。web 3 条（cost-constrained monitors；heads 足够的后门检测；sleeper probes），arXiv API 1 条（activation probes AND cost/budget）。
- **第 2 步 方法论**：
  - web 3 条（NetFlow 采样、传感器布置、压缩检测），全部被额度拒绝。
  - arXiv API 9 条：netflow、sensor placement、compressive detection、group testing、layer selection、Davenport 标题、Krause 标题、Chama、sampled anomaly detection。多数返回为空或不相关，命中 2111.09989、2601.20781。
- **第 3 步 蕴含**：找的是"能让结论成为推论"的工作。把 Obfuscated Activations、LCF、Neural Chameleons 的自适应结论当蕴含读（约束 2）；对完整性方案做结构推导（约束 4、5）。没有专门的查询，只是对已取正文的重读，算 0 条新查询。
- **第 4 步 同义词**：
  - 词表：KV cache / key-value cache / prefix cache / cached states / retained state / compaction / KV reuse / position-independent caching / attention map / k_proj v_proj；detector / probe / monitor / guard / latent-space defense / white-box monitor；budget / sparse / few heads / layer selection / cost-constrained。
  - 实际执行的是 arXiv API 3 条组合查询（KV+detect+attack；KV+probe/monitor；KV+backdoor/trojan/persistent），加上 web 中的若干条。
- **第 7 步 边界**：见 §6。
- **合计**：web 12 条（9 条有效、3 条被拒）；arXiv 列表查询 16 条；arXiv 单篇元数据 14 篇；S2 被引 5 组（3 次被限流、重试成功 1 次）。
- **取正文并定向 grep 的 18 篇**：2610.09027、2412.09565、2602.13547、2604.24542、2406.00799、2411.00348、2510.17098、2512.11949、2604.17249、2511.12752、2607.19957、2610.19452、2609.37737、2611.20454、2610.22873、2608.28099、2611.24619、2505.14300。
- **只看了摘要或检索摘要**：2603.23171、2609.36490、2604.14251、2610.37657、2608.08027、2610.28988、2412.13435、2502.16174、2111.09989、2601.20781、2601.11516、2610.22723、2507.15886、2410.19937、2504.21228、2509.21761、2602.03085、Anthropic 博客。
- **下载了但没有 grep 的**：2604.14865、2506.14261、2510.14005（只 grep 了一次，没有命中层扫描）。

## 6 边界申报
- 第 2 步的跨领域方法源**没有一篇取到正文**（web 额度用完，arXiv 对这些经典期刊或会议文献覆盖差）。§3 S5 里的公式形式和代码可用性都需要复核。
- Semantic Scholar 被限流，Neural Chameleons 的被引没有扫到；P-VMI 发布 7 周、被引 3 篇，并行未公开的工作（尤其是作者组自己的防御续作）是最大的未覆盖风险。
- 未覆盖：在审稿件、OpenReview 上未挂 arXiv 的投稿、非英文文献、工业界内部的监控部署报告（如 Gemini 探针的部署细节，只读了摘要）。
- AISA 中"k_proj/v_proj 位置的具体准确率"我只看到图上的位置标签，没有读出数值。
- 文中标注的档位：约束 4、5 为推导；"保密随机观测位置能救回预算"为推测；P-VMI 的 AUC 推不出低预算可检测，这一点由原文措辞直接支持。
