# 查重文档 — 方向 B：VLM 投毒后门 ASR 的 grokking 式陡峭起跳

**日期** 2026-10-01 · **协议** CLAUDE.md §4 七步查重（全部执行）· **检索工具** WebSearch/WebFetch + 4 个检索子代理

---

## 0. 被查的主张（精确陈述）

> 在 VLM（LLaVA-1.5-7B / Qwen3-VL-8B）投毒微调中（棋盘格图像触发→固定靶词 violin，1% 投毒），逐训练步测的后门 ASR 呈 **grokking 式陡峭起跳**：长时间近零平台，然后在很窄的步数窗（~15–60 步、训练的几 %）内急升到 ~1.0；每 1 步密采样显示它是**陡峭但连续的 sigmoid、非真不连续**；跃迁时刻因种子而异。我们以**相变 / grokking 式突现**框定之。

拆解：
- **B1**（现象）后门 ASR 在训练步上"平台→陡升"。
- **B2**（框架）用 grokking / 相变 / 进度度量工具框定后门 ASR 起跳。
- **B3**（方法）每 1 步密采样论证"陡 sigmoid 非阶跃"。

---

## 1. 结论（坏消息先说）

**B1 现象本身不新。** 最近邻 Connor & Emanuele 2022（arXiv:2212.02582）已明确报告后门 ASR"长时间低、到某点急升到高位并保持"，并命名为 **"tipping point"**——但在**半监督（SSL/FixMatch）**图像分类下，归因于**伪标签自强化反馈环**，且**明确对比说监督训练是渐进的**。两个独立子代理都取到正文并逐字核对。

**但 B2 框架（把 grokking/相变贴到后门 ASR 的训练时起跳）基本未被占。** "grokking"一词在 ~20+ 查询（含对抗轮、"backdoor groks"/"groks the trigger"短语猎取）后**从未**出现在后门/投毒 ASR 上；"相变"贴到后门时都在**投毒率轴**（Zheng&Zen、"Sure Trap" 2511.12414、Lu/Kamath/Yu）——按 CLAUDE.md 5.4 是**独立的轴**（"何时"≠"多少"），不使我们的时间轴主张成为推论。

**幸存的新颖性：**
1. **监督 VLM 指令微调里的陡起跳**——Connor&Emanuele 的唯一机制（SSL 伪标签）在我们的监督设置里**不可能存在**，故我们的陡跳需要不同解释（记忆化/grokking），这是开口；而且与他们"监督=渐进"的论断**直接张力**。
2. **grokking 透镜迁移** + 进度度量工具用于后门。
3. **每 1 步密采样的 sigmoid-vs-阶跃刻画**——后门领域无人做过。
4. **VLM 领域内**：无任何 VLM 后门论文报告 grokking 式 per-step ASR 起跳（最强新颖信号）。

**最大风险（必须正面防御）：Schaeffer「Mirage」2304.15004（NeurIPS2023）**——"非线性/不连续指标制造表面突现"。我们的 ASR 是**首词精确匹配的硬阈值指标**，正是它点名的那类。审稿人必问"你的陡跳是真的还是指标/采样假象？"。**我们已有的密采样（每1步显示连续 sigmoid）+ 多种子 + 对照是部分回应，但尚未展示连续代理量（触发词 logit/margin）**——这是为完全杀死该质疑**必须补**的一步。

**反证需解释：Anti-Backdoor Learning（2110.11571, NeurIPS2021）**——标准认知是"模型学中毒数据比干净数据**快得多**、攻击越强收敛越快"，即**早而快**。我们的"长平台→晚跳"与之相反，必须解释为何我们的 regime（1% 低剂量 + LoRA + VLM 指令微调）偏离标准早学习图景。

**一句话**：不能写"首次观察到后门 ASR 突然起跳"（被 2212.02582 预占）。可写"**首次在监督 VLM 指令微调中展示 grokking 式后门起跳——此处唯一已知机制（SSL 伪标签）不适用——并以密采样刻画其为陡 sigmoid**"，但必须①引 2212.02582 为现象前身、②主动引 Schaeffer 并用连续代理量防御、③解释与 Anti-Backdoor Learning"早学习"的差异、④按 5.4 先核对他们"监督渐进"曲线不是 LR 调度/采样密度造成的混淆再宣称张力。

---

## 2. 七步执行记录

### 步骤 1｜抽象阶梯
抽象为"投毒/后门攻击成功指标在**训练过程中突然起跳 / tipping point / 相变**"，跨所有模型搜。命中 2212.02582（SSL 后门 tipping point，**现象层最近**）；grokking 核心族（突现相变、延迟泛化）；"相变"在后门=投毒率轴（Zheng&Zen、Sure Trap）。现象在 ML 泛层**已有先例**。

### 步骤 2｜方法论单独搜（per-step ASR 曲线 / 后门何时被学到）
命中 Anti-Backdoor Learning（后门早而快）、IB 动力学 2511.21923（MI 轨迹、触发器先于语义被学）、driving GLA 2604.04630（ASR-vs-epoch 但为收敛**速度**对比、单调非平台-跳）。**后门训练动力学被研究过，但无"平台→陡跳+grokking 框架"。**

### 步骤 3｜蕴含检索（谁的结果让我们成为推论）
grokking-of-memorization（2310.13061 "To grok or not to grok"，含噪记忆后 grok）——支持"后门=会 grok 的记忆"类比；但为算法/模数任务、grok 的是泛化。双下降（epoch-wise double descent）是 loss/误差故事，非 0→1 ASR 起跳，非推论。Emergent abilities（Wei 2206.07682）+ Schaeffer 反面。

### 步骤 4｜同义术语枚举后再搜
词表：sudden/abrupt/sharp onset、tipping point、phase transition、emergent backdoor、delayed backdoor、grokking、sharp threshold、critical poison rate、backdoor suddenly appears、ASR jump/surge during training、"groks the trigger"。逐词 + 对抗短语猎取。结果如上；"grokking + backdoor/poison ASR" 空。

### 步骤 5｜引用链挖掘
取正文 2212.02582、2510.05169、2511.12414 等并扫相关工作；Semantic Scholar 正向被引 404，系统正向扫描**未完成**（边界）。forward-citation sweep of 2212.02582 / 2402.15555 建议补做。

### 步骤 6｜对抗式检索（专找"已把后门 ASR 说成 grokking/相变起跳"）
显式猎取。**未命中** grokking 框架贴到后门 ASR 起跳。最接近：Humayun 2402.15555（grok 对抗鲁棒性，防御向）、"From Poisoned to Aware" 2510.05169（后门自我意识"顿悟式"突现，防御 RL 量）——都不是攻击 ASR。

### 步骤 7｜边界申报（见 §5）

---

## 3. 最近邻排序（最近在前；读取级别 + 精确差距）

1. **[最近邻] Connor & Emanuele 2022 — "Rethinking Backdoor Data Poisoning Attacks in the Context of Semi-Supervised Learning"**（arXiv:2212.02582）〔读=全文/ar5iv，两代理逐字核对〕
   CIFAR-10 / WideResNet-28-2 / FixMatch。逐字："the attack success rate during semi-supervised learning remains low for many training steps until a point at which it rapidly increases to a high attack success rate…a tipping point at which the network forms a backdoor that strengthens rapidly." 对比监督："increases gradually…with jumps at LR drops"。机制=伪标签累积（SSL 专属）。
   **差距**（4 条）：(a) 范式——SSL vs 我们的监督微调；(b) 机制——伪标签在监督 VLM 不可能存在，故我们的陡跳需另解；(c) 模态/模型——图像分类器 vs VLM；(d) 框架——"tipping point"，无 grokking/进度度量；无密采样、无种子方差、无 sigmoid-vs-阶跃。**现象层预占，设置/机制/框架/方法未预占。**
   **按 5.4 警示**：宣称"与他们监督渐进矛盾"前，须确认其监督曲线在可比采样密度/LR 调度下测得（其监督曲线有 LR-drop 跳变，是混淆）。

2. **Humayun, Balestriero, Baraniuk 2024 — "Deep Networks Always Grok and Here is Why"**（arXiv:2402.15555, ICML2024）〔读=摘要+正文摘要，已核验〕
   提出"delayed robustness：DNN grok 对抗样本、在插值/泛化很久后才鲁棒"，真实网络（CNN/CIFAR10、ResNet/Imagenette），归因于输入空间线性区相变。**最近**在"真实网非玩具里 grok 一个对抗量"轴。**差距**：方向是**防御**（鲁棒性突现），非攻击 ASR；无后门/投毒。

3. **Shen et al. 2025 — "From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs"**（arXiv:2510.05169）〔读=部分全文〕
   称后门自我意识"在几步内突然出现…类似 'aha moment'"、相变式奖励陡升。**最近**在"相变框架显式贴到后门相关量的训练过程"。**差距**：是**防御量**（事后 RL 的自我意识），非投毒本身的 ASR；无 grokking。

4. **Li et al. 2021 — "Anti-Backdoor Learning"**（arXiv:2110.11571, NeurIPS2021）〔读=摘要，已核验〕
   "模型学中毒数据比干净数据**快得多**、攻击越强收敛越快"。**关键反证/蕴含检查**：标准图景是**早而快**（无长近零平台）。我们的"晚平台→跳"是偏离，**必须解释**为何 regime 不同。

5. **VL-Trojan 2024**（arXiv:2402.13851, IJCV2025）〔读=摘要〕
   正是我们的攻击面（VLM 指令微调后门、低投毒率高 ASR）但只报**终值 ASR**、无训练动力学/grokking。作为"攻击设置前身"引。

6. **"To grok or not to grok" 2023**（arXiv:2310.13061）〔读=摘要〕
   含噪标签下记忆与 grok 并存、记忆样本被遗忘时训练精度突跳。支持"后门=会 grok 的记忆"**类比**；但算法任务、grok 的是泛化、无攻击框架。**类比档**。

7. **driving GLA 2026**（arXiv:2604.04630）〔读=全文〕
   **VLM 内最近**：ASR-vs-epoch 曲线（图5-6），GLA"3 epoch 内急升 90%"、BadNets"停滞 40%"。**差距**：方法间**收敛速度**对比、单调、非单次平台-跳、无 grokking、epoch 粒度粗、仅 DriveVLM。

8. **IB 动力学 2025**（arXiv:2511.21923）〔读=摘要〕
   后门训练动力学的 MI 轨迹"触发器先于语义被学"、早期加速。图像分类非 VLM；摘要未证陡跳 ASR 曲线。**定稿前建议直接读全文。**

9. **风险锚点 Schaeffer et al. 2023 — "Are Emergent Abilities of LLMs a Mirage?"**（arXiv:2304.15004, NeurIPS2023）〔读=全文摘要，已核验〕
   "非线性/不连续指标产生表面突现；连续指标显示平滑"。**方向 B 最大威胁**：ASR 是硬阈值指标，陡跳可能是指标假象。**防御**：连续代理量（触发词 logit/margin）+ 密采样 + 多种子（我们已做前两者的一部分，连续代理量待补）。

**grokking 核心（仅定义，均非后门先例；作借来的词汇/工具引）**：Power et al. 2022 "Grokking"（arXiv:2201.02177，延迟泛化之源）；Nanda et al. 2023 进度度量（arXiv:2301.05217）；lazy→rich（2310.06110）；复杂度动力学（2412.09810）；信息论"grokking 是突现相变"（2408.08944）。**grokking 非仅模数**：Grokking-in-the-Wild 多跳推理（2504.20752）、LLM 预训练中的 grokking（2506.21551）——用于防御"grokking 是玩具"的质疑。〔Omnigrok 2210.01117 凭记忆、未取证，引前核验〕
emergent abilities（Wei 2206.07682）；epoch-wise double descent（Nakkiran 1912.02292）——训练时非单调存在但是 loss 故事、非 0→1 ASR 起跳、非推论。

**非论文来源**（透明记录，不可引）：lacuna.tiptreesystems.com 聚合页；GitHub issue "Tiny Transformers" 草稿；Boaz Barak 课程项目（Zheng&Zen，属方向 A）。

---

## 4. 直接回答 + 分档

- **有人把 grokking/相变框架用在后门 ASR 训练时起跳吗？** **部分**〔推导〕：现象（平台→陡升）被 2212.02582 以"tipping point"在 SSL 预占；"相变"贴后门只在投毒率轴；"grokking"一词从未贴到后门 ASR。
- **现象本身首次吗？** **否**〔推导〕——2212.02582 已有（SSL）。
- **监督 VLM 指令微调里的 grokking 式后门起跳有人报告吗？** **否**〔推导，受递归/非英文/在审边界限制〕——VLM 后门文献全部只报终值或收敛速度。
- **我们的陡跳是真相变还是指标假象？** **尚未完全排除**〔推测〕——已有密采样（每1步连续 sigmoid）与对照支持"真"，但**未展示连续代理量**；这是定稿前必补的防御。

---

## 5. 边界申报（没覆盖什么）

- **取正文 vs 摘要**：全文=2212.02582、2510.05169(部分)、2511.12414(部分)、2604.04630；摘要=2402.15555、2310.13061、2110.11571、2304.15004、2511.21923、2402.13851。grokking 核心族多为标题/摘要级。按协议只有取正文者计"读过"。
- **引用链（步骤5）未完成**：Semantic Scholar 正向被引 404；未系统扫 top-5 的被引列表。建议补 2212.02582 与 2402.15555 的正向被引扫描——可能藏更近命中。
- **ID 凭记忆未取证**：Omnigrok 2210.01117；个别 2408.08944/2201.02177 由子代理给出、我已核验 2301.05217/2304.15004/2402.15555/2110.11571，其余引前需复核版本号。
- **知识截止 2026-01 / 今天 2026-10-01**：2026 年后论文仅实时检索单次取证；一次确认的检索引擎幻觉（"single-seed 分不清陡变与方差"实为 Lelle 2605.30189 方法的转述、非独立论文；另 2603.20198 幻觉见方向 A）；非英文、付费墙、在审、并行未公开工作未覆盖。
- **未做的实验防御**：连续代理量（触发词 logit/margin）尚未出图以回应 Schaeffer；Connor&Emanuele"监督渐进"曲线的采样密度/LR 混淆尚未逐条核对。

## 6. 搜索深度（存活≠强度）
- 4 个检索子代理（2 个专攻方向 B）+ 本人锚点搜索；方向 B ~40+ 查询族、含多轮对抗式检索与"grokking+backdoor"短语猎取。
- 真正取到正文的一手文献：4 篇（2212.02582、2510.05169、2511.12414、2604.04630）；关键锚点 ID（Schaeffer/Humayun/Anti-Backdoor）本人逐一 WebFetch 核验。
- "现象被预占"建立在**两代理独立取正文逐字核对 2212.02582**上——强。"框架未被占"受引用链未完成 + 递归边界限制——中等强度，保持对抗警惕（2026 可能有正做此事的预印本）。
