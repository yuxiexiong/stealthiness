# D3 原 A：C 缺口补证阅读记录

日期：2026-11-19。只处理 C1–C3 与指定 Truong 正则化分支。所有条目由原 A 实际阅读正文全部及以下相关附录；不把 C 的摘要或下载缓存当作本人阅读。C 的其他候选沿既有路由，不扩大目录。旧 B 已核内容不重跑。页码为所存 PDF 页序。当前新增证据 **A_READ / B_PENDING / C_RECHECK_PENDING**。

## C1 — Carlini & Terzis，2106.09667v2 / ICLR2022

全文：`sources/c-contrastive2022.pdf/.txt`。首页版本2022-03-28；共13页，无附录。

| 实际读取 | 内容与核证结果 |
|---|---|
| §1–2 pp1–3 | 引言与对比学习威胁：图文同时训练，攻击者添加少量图文对。单图targeted poisoning与带patch的后门分开；zero-shot与linear probe是两个下游读出。 |
| §3 pp3–4 | 同图多caption与不同图共同patch的区别；caption多样性引导图像嵌入改变是作者解释。没有把这一解释当独立因果识别。 |
| §4 pp5–6 | 默认ResNet50+文本Transformer、batch1024、30epochs、Momentum SGD LR.0002/WD.02。CC3M及YFCC用于攻击评估；后门主要在CC3M。§4.2单图targeted成功不能挪用作patch低剂量；§4.3测实际下游后门，zero-shot与linear probe不等效。本文不转抄§4.3最低比例的绝对数量，避免与Fig2标注混用。 |
| §5/5.1 pp6–7 | 明确切换到backdoor z-score，以减少下游成功读出的方差；读两两带触发图像嵌入相似度相对自然图像分布的偏离。作者把它视为成功代理，但本稿不称ASR；不抄抽取中可能失真的公式指数。 |
| §5.1.1/Fig4 pp7–8 | 五档毒量、固定/随机训练位置、固定/随机测试位置，每条件8模型。低预算固定位置更强；高预算随机位置更强且随机测试更稳，是交互不是单向规则。 |
| §5.1.2/Fig5 p8 | 固定位置；300毒例在50k–3m数据间近似稳定，75在高总量下降。约1m–30m规模扫描的较大模型z-score更强/变异更低；不等于普遍因果规模律。 |
| §5.1.3/§6 p9，伦理/复现p10 | patch增加至一定范围后平台/可能回落；结论和计算代价不改变表示代理限制。正文全读完成；参考文献pp10–13用于确认来源与方法前史，不另扩攻击目录。 |

落点：REPORT §2、§3.5，项目重合表。证据类型：实测条件对照及本稿对指标适用范围的限定；不支持LLM任意长串、自然形成唯一机制或所有下游用途等效。

## C2 — Schwarzschild et al.，ICML2021 / 2006.12557v3

正式主文：`sources/c-toxic2021.pdf/.txt`；完整版本：`sources/c-toxic2021-full.pdf/.txt`，首页v3，2021-06-17。主文10页（正文8页）；完整版17页。渲染完整PDF有xref重建提示，成功得到可见页面；不是不可访问。

| 实际读取 | 内容与核证结果 |
|---|---|
| 正式§1–3 pp1–4 | 比较目标、权限和采样；FC/CP是无触发targeted poisoning，CLBD/HTBD是后门。baseline按原攻击配方，随机化模型/类别/样本；不把所有攻击都叫backdoor。 |
| 正式§4 pp4–7 | normalization/augmentation、architecture、epsilon、disjoint transfer、dataset规模、黑盒、target变换、patch大小；Adam对SGD分析主要FC/CP，不能替CLBD/HTBD证明优化器效应。改多个变量的benchmark不等于单因素消融。 |
| 正式§5–6 pp7–8 | transfer与from-scratch分别定义；统一样本、扰动、网络、训练协议；Table3后门成功低但不代表所有攻击普遍无效。结论强调公平比较。 |
| 完整A.1–A.6 pp11–12 | 误差条是1SE，少于5次成功/失败用p=.5上界；训练batch128及WD2e-4、baseline参数，Tables4–8。增强+归一化组合、ResNet18替换与CIFAR100→10迁移直接影响CLBD/HTBD。A.5说HTBD epsilon与baseline无差异，但其baseline段和Table7不一致；不据该句量化。 |
| A.7–A.10 pp12–14 | **固定1%比例**：5/500到500/50000，Table10原攻击设置、Table11统一设置不应拼曲线；黑盒/类别和翻转的适用对象分别限定。 |
| A.11–A.14 pp14–17 | ensemble是FC/CP前史；Table14 patch大小、Table21换图案影响后门；Tables16–20列模型训练与小数据迁移。Table22固定总量50/100/250毒例，CLBD的5/2/1与HTBD的7/2/10白盒结果不支持逐档增加定律（100trials，近零值误差大）。报告用定性边界，不外推下降定律。 |
| A.15–A.16 pp16–17 | Google image源集中度是可投数据量案例，不是普遍现实上限；Tables24–25完整benchmark。正式Table3与完整版Table25的CIFAR transfer不一致包括后门行，不止无触发攻击。 |

视觉证据已保存：`sources/c-toxic2021-main-p8.png`、`sources/c-toxic2021-full-p17.png`。例如正式Table3 CLBD WB/BB=5/6.5、HTBD=10/9.5，而完整Table25对应3/3.5、2/4；本稿不将两版本当同一表数拼接。不同版本均支持本域有界条件比较，报告不依争议表值定量排序。

落点：REPORT §2、§3.6、§6.6；剂量口径与条件可比性。证据类型：受限实证；未把统一设置整体效应归因单一优化器/模型容量。

## C3 — Shao et al.，Cordyceps 2605.26595v2

全文：`sources/c-cordyceps2026.pdf/.txt`，首页v2，2026-06-15，共27页；USENIX官方入口由C核实，本次原A核对本地首页与正文身份。阅读范围以下完整记录，**附录G证明未全读，不宣称定理重证或证明审计**。

| 实际读取 | 内容与核证结果 |
|---|---|
| §1–2 pp1–4 | 可变语义隐藏控制、共享知识、UCC/BCC权限；SCS是余弦语义评分，不能等同精确重建；攻击者控制训练样本而非训练代码。 |
| §3 pp4–5 | oracle先生成语义关联，再生成推理，格式化双向训练例；UCC与BCC所需编码/解码用途分开。学习成功不能仅归因某个单一阶段。 |
| §4 pp5–8 | 正文形式化全文读：理想表示一致性、局部各向同性高斯和加性映射是建模假设；安全/解码界只在模型假设内成立。没有以正文证明草图替代G的完整证明核验。 |
| §5.1–5.2 pp8–9 | UCC140+1260，7×7共2450测试，新anchors；BCC100+900，100测试目标。5受害模型异构，非纯scale扫描。Table3从10%到1%固定100毒例，扩总数据1000→10000；oracle与训练文本改写对CA有条件影响。 |
| §5.3–5.4 pp9–11 | CA为过滤后隐文敏感数据正确获取指标；ASVs/ASVh是UCC任务读出；检测TPR/FPR、utility不是同一个指标。CROW/SecAlign因推理长度下降早停；adaptive detector的更高检出同时有高FPR。原文“总体utility差小”不代表每任务无损。 |
| §5.5 pp11–12 | 100%毒Llama3-8B，固定prefix减法和mean pooled4096维表示，20样本/少量消息，t-SNE/PCA/cosine/residual直方图；这是几何观察，不充分证明全空间加性或噪声独立各向同性，更非自然训练阶段中介干预。 |
| §6–7 pp13，伦理/开放pp13–14 | 与固定后门/推理时隐写/提示注入的范围区别；共享知识弱或领域特别窄是作者限制；本稿不沿用无范围的首创或不可检测主张。 |
| A pp17–19、B/C pp19–20 | 100%毒：500训练entry展开1000双向例，100测试；4epochs，区别主攻击5epochs。weighted NLL的reasoning前缀权重变化；SCS与EMR/CER区分。C.1主攻击LoRA r16/all/LR4e-4。loss weight0明确定义为不计这些token的loss，不是移除前缀。 |
| D–F及Figs6–10 pp19–21 | Gemini paraphraser提示；防御推理长度0.9阈值，CROW4–9iter、SecAlign20–30iter是所测早停；不写充分防御训练后仍有效。 |
| Figs11–14 pp26–27 | 实际UCC/BCC训练样例和oracle两阶段prompt已读，确认数据构造与推理前缀，而非只读算法摘要。 |

视觉复核已存 `sources/c-cordyceps2026-p19.png`、`sources/c-cordyceps2026-p20.png`：

- AppB p19的Phi4-15B、0.770@0.6与Fig8的Qwen3-4B图注/Table9的Qwen列不一致。报告明确保留冲突，不无声选身份。
- Fig7五模型在5tokens精确匹配近零，但Gemma3-12B由2到3tokens有回升，不沿用正文“monotonically”。Fig6纵轴CER而图注写EMR；本稿按图轴/定义区分，不引用其图注为精确恢复成功。
- Table9加权reasoning的SCS较full reasoning高，不把相似度改写成完整行为；Table10换embedding/task的分数不与Table9无条件混用。

落点：REPORT §2、§5.7。已向主审报告D4/D5接口，不默认该两域已经全读。证据类型：实测条件+明确模型假设边界；不声称理论证明自然形成机制。

## C 定向线索 — Truong et al.，2004.11514 / CVPRW2020

相关性判定：并非仅重复Learning Curves的复杂度概念，原文直接测试多种潜在空间正则化和固定毒数下源类别组成，因此纳入。官方CVF全文已下载到 `sources/c-truong2020.pdf/.txt`；10页，正文8页，无附录。网上CVF条目页码788–789与完整PDF长度不对应，本稿不使用该短页码替代本地PDF页序。

| 实际读取 | 内容与核证结果 |
|---|---|
| §1 pp1–2 | 因果训练投毒与测试时攻击区分，ImageNet模型+有毒公共数据再clean适配的威胁；不是所有学习范式。 |
| §2 pp2–4 | Flowers/CIFAR10，ResNet50/NasNet/NasNet-Mobile；4正则化+无正则化，4触发类型。λ是目标类替换比例，源类别原图保留，不是全数据比例。one-to-one/many-to-one保持总毒数。未转抄抽取式9为无疑义公式。 |
| §3 pp4–5 | 76/19/5数据划分，前二再80/20；adversarial test不含目标类原图，ASR指触发图像被判为目标类；early stop为验证准确率5epoch无提升，独立数据拆分/毒图抽样。 |
| §4.1–4.2 pp5–6 | trigger×architecture×dataset方向不同，正常准确率不足以识别后门；clean续训的平均下降不等于普遍根除。 |
| §4.3/Figs4–5 pp5–6 | Flowers/black square/ResNet50/Adam1e-5/batch32，各正则化5目标类×10随机试验；SNNL平均ASR82→51%，类别和方差影响大；MIXUP部分持平/略增。非正则化系数扫描，也非独立capacity因果对照。 |
| §4.4–4.6 pp6–8 | 高α（触发混合权重较高、视觉更明显）的全图触发仍可在clean续训后有效，图中异常端点有作者随机/earlystop解释；λ扫描仅NasNet/CIFAR truck条件，别挪成全数据比例；相同毒数many-to-one比one-to-one强、正常准确率近似。 |
| §5 p7–8 | 结论及future work全文读；optimizer选择在其写作时列未来工作，不当2026未研究的结论。参考文献pp9–10核方法前史，无新扩展分支。 |

视觉证据 `sources/c-truong2020-p6.png`：Figs4–5的目标类对比与MIXUP例外确在原图。落点REPORT §6.7及项目重合表；D2/D5只作路由，不另做持久性/防御全域判定。

## 本轮边界与交接

四项均达到本协议“正文全部+相关附录”的纳入门槛。Cordyceps G不涉及本稿所用实测主张，明确未全读；不计算为独立完整证明审核。保留REPORT_A及REPORT_PRE_C_SUPPLEMENT快照。新增内容交原B，仅在B和原C复核后更新闭环状态；不把本补读当C完成。
