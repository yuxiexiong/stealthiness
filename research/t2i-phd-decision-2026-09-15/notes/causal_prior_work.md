# 文生图因果归因与机制定位：六篇直接近邻的反证审查

核查日期：2026-09-15。目的：评估博士方向，不为既有方案寻找支持。阅读了六篇下载原文的正文、方法、实验、局限和相关附录；参考文献表仅作线索核对。未复现实验。页码均为本地 PDF 页码。发表状态另核官方会议页，不能根据 arXiv 页眉判断；特别是 DiT Localization 的 v3 仍写 under review，但官方 NeurIPS 2025 已收录。

## 判断

**目前不支持把“归因可视化 + 条件组合 + 保护事实 + 实际干预验证”直接确立为博士创新方向。** 六篇近邻分别覆盖了直接控制点、正负参数贡献、独立提示测试、语义探针与非目标属性保护、上下文交互热图，以及高注意力与语义必要性的分离。把这些元素拼到一起，或把有限表做得更严谨，并不自动形成新科学贡献。

同时，它们没有共同证明：在同一受污染生成实例的固定动作族中，少量已测干预能准确预测未测组合在改变供体/时间/实例后，仍恢复目标且保住每一项原正确关系。因此这里只保留一个**待证、尚未确认新颖性的预研问题**，而不是“已有方法优势只需换文生图”的判断。

## 1. LocoGen / LocoEdit

- **论文与版本**：[On Mechanistic Knowledge Localization in Text-to-Image Generative Models](https://proceedings.mlr.press/v235/basu24b.html)，ICML 2024，arXiv 首发 2024-05-02；本地为 PMLR 正式 PDF，共 42 页，文件 `runs/papers/causal_locogen.pdf`。
- **模型、动作、测量**：SD1.5、SD2.1、OpenJourney、SDXL、DeepFloyd。LocoGen 把部分 UNet cross-attention 层的条件输入换成目标提示嵌入；搜索最小连续层窗，按 CLIPScore 判断目标概念删除/事实更新（§5，p3–5）。它明确从间接 restoration tracing 转向直接控制效应，并非声称每个控制点都是唯一知识存储位置。
- **实际证据与对照**：LocoEdit 对找到的 K/V 矩阵作闭式更新；比较全层更新、DiffQuickFix、Concept-Ablation/Deletion（§6；App F/I，p32–34）。人评 132 对、5 人，“92.58% 有效果”实际是评分 **>1/5**，不是 92.58% 完全正确；满分率更低（p34–36）。320 个普通提示的平均 CLIPScore 检查保留性（App L，p41），不是同图逐项关系均保持。
- **失败与边界**：DeepFloyd 的控制层随提示变化（p6、25），且能定位不代表相同闭式编辑可用：T5 双向编码下只改最后主体 token 的方法失败（p7、41）。神经元消融数量增大可损其它属性（p8）。这些已直接覆盖“供体/提示条件性”和“可控性不等于权重修复成功”的一般论点。
- **对我们的压力**：有限组合搜索、对照式提示供体、最终图验证已经存在；未覆盖的是明确的逐实例多事实动作预测任务与等总预算决策增量，不能仅把这两项补上就认定创新。

## 2. CAD

- **论文与版本**：[Unveiling Concept Attribution in Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/295d03c732e675d699aab6056393d5ee-Abstract-Conference.html)，NeurIPS 2025；首发 2024-12-03，阅读 arXiv 2412.02542v3，2025-10-28，20 页。
- **方法**：对参数 knockout 的概念目标作线性反事实预测，系数以 gradient × weight 的一阶展开估计；正参数去除用于 erase，负参数去除用于 amplify。核心目标是条件/基准条件噪声预测差，或负噪声重建损失（§4–5，p3–6），不是终图事实正确率。
- **预测与保护已做**：§6.1 的 1,000 次随机参数消融验证预测值，Fig2 的 Pearson **r=0.506**（p6，已渲染核对），因此不能说“没有预测验证”，也不能说“高精度终图预测”。SD1.4 为主，附录扩至 SD2.1/SDXL/SD3.5。ImageNette 每类 500 图、他类准确率、I2P/COCO30k、CLIP/FID/LPIPS，比较 ESD/UCE/RECE/ConceptPrune（p6–9、13–17）。Table6 的 amplification 确有目标提升，也有他类损失。
- **限制**：线性近似不等于真实协同建模；正负符号依赖所选 J 和消融。App A、D/E/G 明确承认参数多概念纠缠、模块差异、消融剂量与保留权衡，时空定位未研究（p13–16）。Table1 他类准确率下降与“其它概念完全不受影响”的强表述不能等同。
- **重合**：贡献符号、组合反事实预测、实际编辑、非目标保留均有直接近邻。我们的可能差别必须是**在具体条件下预测逐事实效应向量为何失败及何时可靠**，而不是又增加一种综合分数。

## 3. CASL

- **论文与版本**：[CASL: Concept-Aligned Sparse Latents for Interpreting Diffusion Models](https://arxiv.org/abs/2601.15441)，2026-01-21 v1，25 页；本次未确认正式会议发表。
- **重要范围校正**：CASL 自身主干是 FFHQ/CelebA-HQ 的 DDPM++、LSUN 的 DDPM、AFHQ 的 iDDPM，使用真实图反演与文本 CLIP 监督；**不能写成它已验证通用 T2I 模型**（§5.1，p5）。训练 SAE 后学概念线性映射，CASL-Steer 明确把内部激活编辑作为因果探针（§4，p3–5）。
- **终图与非目标指标**：EPR 为目标分类 logit 的绝对变化除以其它属性平均绝对变化，另报 CLIP、LPIPS、ArcFace；与 BoundaryDiffusion、Asyrp、Concept Sliders、MasaCtrl、SwiftEdit 比（p5–8）。主表每属性 32 测试图；top-1 干预与 top-k/强度消融显示加入更多维度可能增加串扰，而非天然有益（Fig4）。SVM 的独立特征可读性测试并非未测生成动作预测。
- **可信范围与问题**：EPR 不检查变化方向正确、目标真的翻转成功或每项非目标均保持；均值可掩盖单项破坏。App §9.1/Table11 明确展示换属性分类器会改分数甚至方法排序，狗/建筑缺标签不报 EPR（p16、18–21）。主文 p6/8 称单次 shift，App §7.4 p11–12 明称在约 50 步反复注入；默认 SAE scale 的描述也不一致。人评 10 人的详细结果只指向未取得的 `Human Evaluation.pdf`，**未核其结果**。不同基线还使用不同主干/分辨率（p13–14），不可作为严格同模型方法净效应。
- **重合**：语义探针、稀疏方向、多属性副作用、人评都不是空白；CASL 本身已有测量依赖的负证据。

## 4. Attention Sinks in Diffusion Transformers

- **论文与版本**：[Attention Sinks in Diffusion Transformers: A Causal Analysis](https://arxiv.org/abs/2605.09313)，ICML 2026（官方会议目录与 PDF 核实）；首发 2026-05-10，阅读 v3 2026-06-16，24 页。
- **动作与对照**：SD3 主实验、SDXL 验证；每头每步按 incoming attention 找动态 top-k，改 logits 或 value；固定提示、seed、采样参数，设置 no-op、等数量随机 mask、层/时间/剂量对照（§3；App A–I）。这是一套实际内部干预，不是只观察热图。
- **关键结果**：553 GenEval **提示**上主要用 CLIP-T/ImageReward/HPSv2，非 GenEval 官方检测器通关率；k=1 时未检测明显代理退化，却能引起显著 LPIPS 变化。union-budget 64 提示对照中 sink drift 比 random 大约 6 倍（Table7 p7）；k=10/50 时 HPSv2 出现显著额外损失，CLIP-T 未显著下降（Table20 p21）。还做 BLIP2-VQA，但未覆盖全部关系/精细构图（p7、9）。
- **限制与审查**：不能将“CI 包含零”写成已证明等价；例如 Table19 多个区间超过所述 ±0.002 等价带。主文/附录多处“no degradation”要按 k、指标和 masking 协议限定。作者明确未证明端到端加速、fine-grained factual correctness、训练时功能或人类偏好（p9）。
- **重合**：高 attention 非必要、强视觉变化非语义损伤、指标冲突、条件化剂量效应和等预算对照均已被研究。“我们会用 do-intervention 而非热图”不足以区别。

## 5. Localizing Knowledge in Diffusion Transformers

- **论文与版本**：[Localizing Knowledge in Diffusion Transformers](https://papers.neurips.cc/paper_files/paper/2025/hash/157a6107a95883f6ccc0e38e80a24295-Abstract-Conference.html)，**NeurIPS 2025**；首发 2025-05-24，阅读 2505.18832v3（2026-01-26），20 页。arXiv 页眉仍称 under review，不据此误标未发表。
- **完整链条已存在**：PixArt-α、FLUX、SANA；按 attention × value × output projection 的范数聚合 block 贡献，取 top-K；以 neutral prompt 供体替换所选块条件输入，FLUX 需要两条文本分支。LOCK 六类概念明确分 train/eval 提示，train 定位、eval 上实际干预，用 CLIP/LLaVA/CSD 检查概念（§3，p4–7）。
- **下游与保护**：PixArt 局部 DreamBooth（9/28块）与局部 unlearning（5/28块），比较全模型微调；报告目标、surrounding concepts、anchor 和 COCO10k FID（§4，p7–9）。这些是其它提示上的类级身份/质量保留，不是同图每条关系严格均对。
- **最强基线事实**：App B.2（p14）连续窗口 brute force 的 style CSD 降幅 0.0812，贡献排名 0.0700；作者主张后者约 28 倍更快，不能说更有效或等总成本。已存在“热图排名→未见提示干预”的验证，不能以我们另设 test split 当贡献。
- **未解决**：K 仍依赖外部干预/反馈；无已知真实定位的 ground truth；艺术复杂度规律以 GPT-4o 分组后解释、仅定性支持（p15–18）。缺少非相邻组合交互建模、逐实例动作是否同时保住原正确关系的预测，然而这些缺项并不自动构成可发表的新方法。

## 6. Metagame

- **论文与版本**：[Attributions All the Way Down? The Metagame of Interpretability](https://arxiv.org/abs/2605.06295)，2026-05-07 v1，32 页；本次未确认正式发表。
- **理论重合很直接**：把一阶归因本身当 coalition value，对“j 如何改变 i 的归因”求 Shapley；提供层次分解、方向交互和与 STII/SOP 的关系（§2–3，p2–6；证明 p17–22）。因此“背景条件下贡献改变/高阶或方向交互热图”不能泛称新理论。
- **T2I 部分真实做了什么**：FLUX.1-schnell 的 Meta-ConceptAttention 用 VOC/COCO/ImageNet-Seg 做概念分割；Table3 报 mIoU 等提升（p7–9、27–28）。**p25 明确：缓存单次 forward 的 logits，改变 coalition 仅改变 softmax 分母，并把先 softmax 后平均改成先平均后 softmax；它们通常不相等。** 本页已渲染核对。这里的交互可能来自归一化竞争，不能当生成轨迹上的概念因果干预。
- **边界**：T2I 没有对每个 coalition 重新生成终图，没有恢复目标/保护其它事实的动作实验，也未证明交互图帮助人或选择动作。理论依赖所定义 masking game；有向归因不等于有向因果图；d 较大须近似并增加计算与认知负担（p9、23–28）。
- **对我们的意义**：这是“交互可视化/条件敏感性”极近邻，同时帮助明确本项目必须直接测生成模型干预后的事实效应，不能只展示解释器本身的数值依赖。

## 覆盖矩阵（“有”不表示已解决我们的完整问题）

| 拟主张 | 已有最直接覆盖 | 剩余证据缺口 |
|---|---|---|
| 内部定位→生成控制/编辑 | LocoGen；DiT Localization；CAD | 不能推唯一知识位置、污染来源或永久修复 |
| 预测未测干预 | CAD 随机 knockout 的噪声目标预测；DiT 未见提示测试 | 逐实例终图事实向量、不同条件/供体下预测有效域 |
| 目标与非目标一起测 | CASL EPR；CAD 他类；DiT surrounding；LocoGen 普通提示 | 每项原真关系的联合通过与覆盖，不能用全图相似度代替 |
| 条件/交互结构热图 | Metagame；LocoGen 提示依赖 | 对生成干预有信息增量，而非解释器归一化或数学恒等式 |
| 热图不是因果必要性 | Attention Sinks；LocoGen 失败案例 | 需要具体新失效机制；重复“attention不可靠”不够 |
| 小组合、预算、少改动 | LocoGen 窗搜索；DiT top-K与brute force；CAD近似 | 比强简单增强、经验窗、同总预算黑箱尝试的真实决策收益 |

## 最强 no-go 与有条件保留的问题

最强 no-go 是：**把已知 attribution、干预、多个指标和 test split 重新组织成一个流程，不能弥补没有新规律和没有方法优势。** 无论图画得多完整，若它需要先测完每个候选的终图，所谓预测只是查表；若直接试验获得同样安全修复集合，图没有自动产生实用增量。可视化本身的人机价值还必须以同信息表格/列表作对照，不能拿更多信息的图去赢少信息基线。

父任务正在评估的“实体间语义串扰”可保留为一次条件性预研：同一候选跨实体连接是否在不同实例和去噪时间下，对身份污染与合法关系产生可重复、可预测的不同效应；这种效应结构能否在固定总成本下，帮助选择/拒绝/组合动作，改善终图目标且保住原正确关系。

最致命的四个反驳：

1. “同一连接承载污染与关系”仍是机制假说。删除一条连接后两项输出变化，只证明该计算操作的总效应；重归一化、其它通路与后续轨迹偏离也能解释，不能据此识别边内两个独立语义或污染根因。
2. 如果简单 image-text 增强已覆盖绝大部分收益，剩余失败可能在允许动作族内不可修，或只需便宜经验时窗；这时复杂诊断没有决策空间。必须先证明剩余**可安全修**机会非零且不是少数挑选示例。
3. 保护合法关系可能与修改对象身份冲突。若任何动作都会破坏关系，发现 trade-off 不是解决方法；只能在明确动作族内报告无法找到安全操作，不能泛称模型不可修。
4. 拒绝策略可用全拒绝制造零副作用。因此主指标必须同时报告覆盖率、目标成功且全部保护事实通过率、每次成功的总成本，而非只报已接受案例的精度。

能留下的研究问题必须再多一个实质层：提出**具体可反驳的条件规律或可靠性判据**，事前预测未测动作的目标/关系效应或其失效，而不是仅输出一个向量。冻结动作族、已测信息、强简单增强/固定窗/同预算直接搜索对照；分开调参场景与独立场景/时间/供体验证。若没有可修机会、没有可预测分叉，或诊断比不上强简单方法，就停止该方法路线。即使有差距，也需要后续近邻审查，不能称无人做过。

## 公开评审与材料边界

- LocoGen `https://openreview.net/forum?id=fsVBsxjRER` 与 CAD `https://openreview.net/forum?id=dVIx32Lq7J` 本次读取返回 429，评审正文未读。
- Sinks `https://openreview.net/forum?id=QwE8cOtclR` 与 DiT Localization `https://openreview.net/forum?id=SiBVbL7rsX` 跳到浏览器验证，停止访问，评审/decision 正文未读。
- CASL 与 Metagame 未找到可核对的公开评审页；不由此推断未投/拒稿。
- CASL 人评细表单独补充 PDF 未取得。其内部实现描述不一致已列出，未通过运行代码判断哪一处正确。
- 六篇 PDF 均验证文件头、页数、标题和 SHA256；全文带页码抽取，关键页已渲染。精确版本、文件路径、阅读范围和限制见 `evidence/causal_papers.json`。
