# "消退≠擦除"嫁接到后门防御评测：七步查重（2026-11-26）

状态：**一轮七步 + 一轮补查**。没有运行实验，没有定方向。按 CLAUDE.md 5.1，这里不给等级。
按青蒿素判据（记忆 no-novelty-purism）：先例一律先放进"先例支持"列。只有事实约束和竞争解释，才放进"坏消息"。

前置报告：`research/eight-cornerstones-2026-11-25/AUDIT.zh-CN.md` §4.1。那份报告已核过"孪生＋重学"的方法层先例（CINDER、TARE、2611.29285、2607.05898），本轮不重做。

---

## 0. 受审主张

| ID | 主张（审前原话） |
|---|---|
| C1 | 用四类复发检验（自发恢复、情境更新、恢复、快速重获）诊断净化后的后门，每类检验对应一种复发机制。预测：不同检验给防御排出不同名次，单一重学测试会误判。 |
| C2 | 在模型内部做差异对照（A→X 对比 A→Y、B→X，即 CS+ 对比 CS−），区分"旧关联的特异残留"和"一般易感"。特异性问题不需要孪生，孪生只用于标定。 |
| C3 | 借鉴再巩固更新（Monfils 2009、Schiller 2010）设计防御日程：先唤起后门，在一个窗口内再擦除。它在 C1 的检验组上的复发低于常规擦除。 |

---

## 1. 坏消息：事实约束与竞争解释（只放这一类）

1. **源领域不支持"每类检验对应一种机制"。** Bouton 学派把这几类复发都看成情境效应或提取效应，认为它们是情境更新的不同形态。
   - Todd, Vurbic & Bouton 2014（PMC3946264）："Spontaneous recovery can also be conceptualized as the renewal effect that occurs when the extinguished CS is tested in a new temporal context"。
   - Broomer & Bouton 2023（PMC10204583）把快速重获也归为 ABA 情境更新。
   - Kredlow 2016 元分析（PMC4760909）："The type of test for return of fear did not significantly moderate overall outcomes"。
   - **C1 的机制表述必须改写**，见 §5。【推导，依据源领域原文】
2. **C3 的心理学来源两次预注册复现都失败了。**
   - Luyten & Beckers 2017（大鼠，PMC5931313）："we did not find any evidence for reduced return of fear with the retrieval-extinction procedure relative to regular extinction training, as assessed through spontaneous recovery, reinstatement and renewal."
   - Chalkia et al. 2020（人类，Cortex，PMC7115861）："evidence that a reactivation-extinction procedure prevents the return of fear in humans is not very robust."
   - 同组的数据核验报告（PMC7115860）指出，原文的差异取决于被试排除方式。
   - 源领域自己也承认，"预测误差"和"时间窗"没有办法操作化（Schroyens 2023，PMC7614440）。
   - 所以 **C3 不能把再巩固当作理论依据。**
3. **检验组只能单向使用。** Todd 2014："The absence of fear responding on a test cannot distinguish between these possibilities"。检出复发能否决一个防御；没检出复发不能证明它已经擦除。
4. **在已有的"四类检验搬到机器学习"实验里，只有重学速度能分开方法。** 公开仓库 Suppressed, Not Erased（CIFAR-100 类别遗忘，5 种方法，以从头重训的模型为参照）RUNLOG 第 641–652 行：
   - 输出层面上，时间、情境、线索三类检验都测不出复发；
   - 只有 ABA 情境更新对 SCRUB 有一个迹象，原始 p=0.004，经 50 次检验校正后不显著；
   - 重学速度随方法不同而不同；
   - "Multi-context unlearning does not reduce relapse (H6 null)"。

   这与 C1 预期的方向相反。不过，这是另一个对象（视觉、类别遗忘），而且复发处于地板，所以"多情境修复无效"本身就无法检验。
5. **在 LLM 遗忘上，攻击成功率的维度低，微调攻击近似上界。** Che et al. 2502.05209：
   - "Three principal components explain 89% of the variation in attack success"；
   - 65 个模型里有 64 个，微调攻击取 max 后不弱于最强的输入空间攻击。

   C1 必须预先写明：我们这组检验比这三个维度多出了什么。
6. **良性微调后复发，有一个与消退无关的通用解释。** Harvey & Stone 1995（Neural Computation）："imperfect recall of a set of associations can usually be improved by training on a new, unrelated set … a consequence of the high dimensionality of weight space"。所以"自发恢复"类检验分不清"压制"和"没擦干净"，**C2 的对照因此承重。**【推导】
7. **C2 的交叉对照不中性。**
   - Hu et al. 2406.13356：只要共享成分，训练其中一个就能恢复另一个。
   - Bouger 2606.03785：去掉一个触发器会压住其他触发器。

   A→Y、B→X 必须避开共享成分，并预先写明如何处理跨关联泛化。
8. **视觉后门上，"带触发、给正确标签"的净化（EP）本身就是消退，而且不复发。**
   - Min 2410.09838 第 234–237 行：EP "consistently maintains a low ASR even after applying RA"。
   - 同一对象上，干净模型在同等重调条件下学不会后门（第 226–228 行）。

   所以在视觉后门上，"消退≠擦除"的类比前提不成立，净化后变快的部分是后门特异的。在 LLM 上必须先测。
9. **竞争解释，C3 必须打败：**
   - 平坦度：Fan 2502.05374，SAM 让遗忘对重学更稳，不需要激活；
   - 只对训练过的检验耐久：Lu 2505.17013，先激活再清除的 STEREO、RECE 换一种检验就失效；
   - 安全对齐侵蚀：BackdoorLLM 第 556–557 行，良性微调本身就会抬高 ASR。

---

## 2. 先例支持（谁走到哪一步）

### 2.1 C1：多检验组、跨检验排名不同

| 工作 | 前人到哪（层） | 读取 |
|---|---|---|
| 2611.05358 He et al.（11-03） | **最近邻。** 窄触发模型生物（Qwen3-4B、GPT-OSS-20B），两类复发：原数据继续训练（相当于快速重获）、OpenThoughts 良性微调（相当于自发恢复）。第 394 行："Reactivation durability depends on both the method and the reactivation type"。第 173–178 行明说不考虑对抗来源，后门构造 "only as a controlled testbed"。复发没有孪生参照，没有情境更新、恢复类检验，没有特异性对照。**结论层被近邻对象部分占据，投毒后门对象空着。** | 主会话亲自核过 |
| 2510.07626 Fan et al.（WSDM'27） | LLM 遗忘：领域内重学对比领域外微调，排名翻转，"RobReL and RobFT should be considered jointly"（第 466–474 行）。**结论层（遗忘）** | 主会话核过 |
| 2505.17013 Lu, Bau 等（NeurIPS'25） | 扩散模型概念擦除：独立检验组，两类残留（绕开型 vs 毁掉型），"knowledge undetectable through one technique can often be recovered through others"（第 583 行）。**方法层＋机制映射（另一对象）** | 主会话核过 |
| 2409.18025 Łucki et al.（TMLR'25） | 表 1：Logit Lens 与微调、正交化下名次翻转；用保留集 10 条样本即可恢复。方法层 | 子代理已读 |
| 2502.05209 Che et al.（TMLR'25） | 11 种攻击 × 65 个模型。方法层（含跨检验一致性分析） | 子代理已读 |
| 2402.16835 Lynch et al. | 8 项测试，只针对 WHP 一个方法。方法层 | 子代理已读 |
| Suppressed, Not Erased 仓库（2026-10-03） | **四类复发检验已搬到机器学习**（视觉类别遗忘，有从头重训的参照，有代码）。方法层 | 主会话核过 README/RUNLOG |
| 2602.03379 Yoon et al.（ICLR'26） | "syntactically similar data consistently trigger recovery even without topical overlap"。这相当于情境更新；它的修复是遗忘前先把查询改写成多种句法，相当于"多情境消退"。方法层（事实遗忘） | 子代理已读 |
| 2501.02704（TNNLS，后门水印） | 只用原训练数据就能恢复被微调去掉的后门水印，"reinstated to up to 100%"。情境更新类，方法层，邻近对象 | 主会话核过 |
| Min 2410.09838、BI-BAU 2606.14078、Zhu 2405.16134 | 视觉后门：RA、QRA、BEC 两到三种检验，未报告跨检验排名差异 | 子代理已读 |
| Harvey & Stone 1995 | 神经网络里"自发恢复"的命名与几何解释。方法层 | 子代理已读 |

**对象层空着的直接证据（承重）：**
- BackdoorLLM 2408.12798 v2：全文没有 relearn、retun、reactivat（grep 计数为 0）。表 4 的防御（Fine-tuning、Quantization、Pruning、CleanGen、CROW 等）只报告防御后的 ASR。
- NEEDLE 2610.00348（LLM 后门移除）第 1040–1042 行把下面这件事列为 future work："test the persistence of removal under subsequent fine-tuning or re-poisoning, including whether a removed backdoor can be relearned"。
- BEEAR、Backdoor Collapse、2603.13461、2610.00685 都没有净化后的复发测试（grep）。
- 在 LLM 投毒后门对象上，没找到任何一篇工作同时做多项检验和排名比较。

### 2.2 C2：模型内部的差异对照

| 工作 | 前人到哪 |
|---|---|
| Haaker 2014（PMC4138360） | 教科书做法："Differential protocols … control for … sensitization effects"。要区分特异恢复和一般恢复，需要"时间 × CS 类型"的交互；还要设"不给 US"的对照组 |
| Deeb & Roger 2410.08827 | 独立事实随机分成 T/V，只在 T 上重训、在 V 上测 |
| 2610.22381 | 同一次运行内用拉丁方：LEARNED / CONTROL / WRONG |
| 2611.29285 | 48 个从未教过的对照实体，"separate forget-specific effects from damage that hits every entity" |
| UMU 2610.21230 | 反方向的同一种混淆：重学低可能只是可塑性丢了；用新任务学习作对照 |

**方法层是教科书做法，已被多次使用。** 在后门对象上（触发 A/B × 目标 X/Y 交叉）没有找到。

### 2.3 C3：先激活再擦除

| 工作 | 前人到哪 |
|---|---|
| Min 2410.09838 EP | 视觉：用真实触发、正确标签净化，重调后仍低 ASR；用代理触发的 BTI 不耐久 |
| BI-BAU 2606.14078（CCS'26） | 命题 2：完整擦除要求对抗样本 "display the backdoor features under the backdoor model"。RA 下 BI-BAU(T) 的复发后 ASR 是 13.53 / 11.26，SAU 是 85.51 / 45.64 |
| 2611.05358 PEARL | 从异常行为已经出现的前缀状态出发做 RL，比 SFT/GRPO 更耐久（LLM、非对抗来源） |
| Qi 2406.05946、LAT 2407.15549、TAR 2408.00761、STEREO 2408.16807 | 安全对齐、遗忘、擦除上"诱出后再训练"的各种版本 |
| DREAM 2610.13267 | 以再巩固为灵感修复被投毒的智能体记忆库（作用在记忆嵌入上，不在模型参数上） |
| Xu & Strohmer 2502.05684 | 拿再巩固当遗忘的动机类比，没有算法 |

**"唤起 → 时间窗 → 擦除"这种对模型参数的时序日程：60 条 arXiv 查询、约 40 条网页检索都没有命中。** 但它的源效应已被复现否定（§1 第 2 条）。

---

## 3. 三列表：前人到哪 / 我们推到哪 / 谁改做法

| 子主张 | 前人到哪 | 我们能推到哪（目前没有数据，全部是计划） | 谁改做法 |
|---|---|---|---|
| C1：LLM 投毒后门上的复发检验组 | 遗忘（Fan、Łucki、Che）、扩散（Lu）、近邻对象（2611.05358，非对抗、无孪生、两类检验）都已看到"耐久性取决于用哪种检验" | 投毒后门对象；有孪生零分布（跨训练种子标定，我们的 S3 显示同一批数据只换顺序，装上的结果就是 3/3 对 0/3）；补上前人缺的情境更新、恢复两类检验；按情境距离做预注册预测（§5） | BackdoorLLM 一类基准的维护者（目前没有任何复发测试）；NEEDLE 等把复发列为 future work 的移除方法作者；"后门防御即服务"平台（PROTOPURIFY） |
| C2：特异残留对比一般易感 | 心理学的教科书设计；事实遗忘上的 T/V、拉丁方、对照实体 | 在触发 × 目标上交叉，避开共享成分（Hu），处理跨触发泛化（Bouger） | 用重学当验收的人：可以知道重学变快里有多少属于原后门 |
| C3：再巩固时序日程 | EP、BI-BAU、PEARL 已证明"在激活状态下擦除更耐久"；DREAM 已在记忆库上用再巩固 | 源效应不可靠，时间窗在随机梯度下降里没有对应物【推导】 | 时序版本：没有可信的受众 |

---

## 4. 逐条判定（A＝能写出有说服力的 novelty 论证；B＝撞车到无法实质区分）

- **C1 原表述（"每类检验对应一种机制"＋"首次发现检验排名不同"）：B。**
  - 机制映射被源领域否定（§1 第 1 条）；
  - "检验排名不同"已在遗忘、扩散、近邻对象上写出；
  - 检验组已在仓库里搬到机器学习。
- **C1 换到对象层（LLM 投毒后门 × 孪生零分布 × 补齐情境更新、恢复两类检验）：A。** 论证：
  - LLM 后门主基准没有任何复发测试；
  - NEEDLE 把它列为未来工作；
  - 最近邻 2611.05358 自己说明不处理对抗来源，也没有参照模型；
  - 前人的发现可以引来支持现象的普遍性。

  新增的是：对象、带零分布的检验、两类前人没测的检验，以及能被否证的情境距离预测。禁写"首次多探针""检验对应机制"。
- **C2：A，但作为部件，不独立成篇。** 方法照搬心理学和 Deeb、2610.22381 并引用。新增的是后门交叉设计，以及它对验收读法的后果。
- **C3 原表述（借再巩固）：B。** 理由：
  - 源效应不可靠；
  - "在激活状态下擦除"已被 EP、BI-BAU、PEARL 占住；
  - 时间窗没有算法对应物。

  按 1.1，这里不是把 C3 删掉，而是**换修复路线**（§5）。

---

## 5. 吸收新信息后最强的版本（讨论用；全部是推测档，未经第二轮查重）

1. **把"四种机制"换成一个可测量：情境距离。** Bouton 学派本身就把复发统一为一件事：测试情境离消退情境越远、离习得情境越近，复发越强（ABA > ABC > AAB）。搬到后门上：
   - 习得情境 = 投毒数据的模板、领域；
   - 消退情境 = 净化数据；
   - 后续训练或测试情境在这两者之间的位置，就是"距离"。

   预测：复发强度随"离净化情境的距离"单调上升，ABA 最强。2602.03379（句法相似驱动恢复）和 2501.02704（回到原训练数据恢复水印）是同方向的先例支持。【类比】
2. **修复路线：用心理学里证据最硬的办法替换再巩固。**
   - (a) 多情境净化：2602.03379 的句法多样化在事实遗忘上有效；仓库在视觉上测出无效，但那里的复发处于地板，所以测不了。
   - (b) 消退中穿插少量强化试次，在动物身上能减慢重获（Todd 2014 第 27 行、Broomer & Bouton 2023）。机器学习的对应物尚未定义。

   这两条都必须打败 EP（神谕上界，需要真实触发）、BI-BAU、PEARL 和 SAM。【类比/推测】
3. **验收读法：** 检验组只能单向否决；"通过"只能写成"在所测检验与剂量下，相对孪生零分布的超额复发 ≤ ε"。

这一版本与 eight-cornerstones 报告 §4.1 的 A′（孪生＋重学验收）在对象上重合。区别在于检验组、情境距离预测和修复。**两者合并还是分开，等讨论。**

---

## 6. 检验量（CLAUDE.md 5.2）

- 一轮七步，加一轮补查，共六路子代理。顺序是 6 → 5 → 3/2/1/4 → 补查 → 7。
- 查询约 230 条（arXiv API、Semantic Scholar、OpenAlex、NCBI、网页检索）。其中约 30 条因限流或主机不可达而失败，按"没跑"计，不按"零命中"计。另有 6 条 Semantic Scholar 空返回无法区分限流和零命中。
- 按标题筛过约 650 条：被引 388＋49、参考文献 258。
- **按规则算"读过"（取到正文、做过定向追问）去重后约 53 篇。** 其中主会话亲自 grep 核对承重原句的有：2611.05358、仓库 README/RUNLOG、PMC5931313、PMC7115861、PMC3946264、PMC4760909、2510.07626、2610.00348、2408.12798、2505.17013、2606.14078（表 10、12）、2501.02704、2609.18985（确认它与仓库无关：同名前缀，是知识编辑论文，没有消退术语）。其余由子代理核对，主会话没有逐篇复核。
- **存活 ≠ 强度：** §4 中判 A 的子主张只经历了这一轮。

## 7. 边界申报

- **在审稿件：** OpenReview 全文检索无法通过网页检索做到。ICLR'27 只扫到第三方列表，覆盖弱。
- **非 arXiv：** BDBR（ICML'25）、CARD、Delete but Not Gone、Backdoor Complications 只读到摘要或没取到。
- **心理学：** 只覆盖 Leuven / Bouton / Monfils 一系。Bouton 2004、Bouton 2021 正文、Ricker & Bouton 1996、Bouton, Woods & Pineño 2004 没取到（403 或闭源）。因此"重学是最含糊的检验"**不能**写成 Bouton 的话。
- **仓库 Suppressed, Not Erased：** 论文没有找到索引，可能存在未公开或在审的版本。
- **没取到全文：** BackdoorLLM 的 NeurIPS'25 正式版（只核了 v2）；SEAM 在重调攻击下的表现（未找到，不等于没有）。
- 非英文文献没有覆盖。并行的未公开工作无法覆盖。2026-06 之后的论文只能通过检索获得，不能凭记忆核对。
- 原始材料（PDF、txt、查询日志）在会话临时目录，没有入库。
