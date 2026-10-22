# HanLab 经验总结

> 来源：MIT Han Lab 出版列表（hanlab.mit.edu/publications）上的 96 个条目，去掉 1 个合集条目（"Tiny Machine Learning Projects"），实际 95 篇论文；其中 94 篇取到正文并逐篇精读（含附录，附录缺失的单独列在第 7 节），1 篇（SpAtten-Chip，ISLPED 2023）无公开 PDF，只看了项目页摘要。
> 精读笔记、九个领域组的组内综合、以及主会话的抽查记录，都在同目录 `notes/` 下。本文每一条都给出论文名和图表号，可以回 `notes/<slug>.md` 查原始出处。
> 写作日期：2026-10-22。

## 0. 这份文档是什么，怎么用

**它回答两个问题：**
1. 如果把 Han Lab 的水平定义为 solid，那么 solid 具体指什么、不指什么。
2. 在不同的情境下，他们用什么实验去证明自己的论点。

**它不是什么：** 不是论文摘要集，不是方法综述。每一条写的都是"论证结构"和"实验设计手法"，而不是论文做了什么方法。

**档位标注。** 全文的每一条主张标注三档之一，这是我们自己的研究准则 5.3 的要求：
- **【读到】** 论文里明确这样做了，给出位置。
- **【归纳】** 我从多篇论文的做法里总结出的原则，论文本身没有这样表述。
- **【推测】** 我的猜测，需要独立验证才能当真。

**使用规则。**
- 借鉴之前先看第 2 节"不学清单"。Han Lab 的 solid 不包含统计严谨性，这一点如果不先说清，借鉴时会把坏习惯一起学进来。
- 理论和实验的关系看第 4A 节：理论跟在观察后面，预言走在实验前面。
- 设计实验时按第 4 节的"情境"查：先确定自己处在哪个情境，再看那个情境下他们怎么做、哪里常做坏。
- 投稿前过一遍第 5 节自查清单。
- 第 6 节是把这些手法映射到我们自己研究方向的尝试，全部是【推测】档，不能直接当结论用。

**本文的核对状态。** 逐篇精读由 9 个子任务并行完成，主会话抽查了 17 条关键数字，16 条与原文对上，1 条（NetAug）因原文本损坏未核成，详见 `notes/_spotcheck_log.md`。其余数字没有逐一复核，引用前如需严格依据，回原文。

---

## 1. 先说结论

### 1.1 Han Lab 的 solid 是什么

读完 94 篇之后，我对"solid"的定义是：

> **效果可归因，并且在真实条件下兑现。**
> - 可归因：增益能被钉到唯一一个改动上。读者看完表格，知道每一个百分点来自哪个组件。
> - 真实条件下兑现：在真硬件、真模型、端到端的设置下测到，并且实测值与理论上界对得上账。
>
> 在这两条之上，用**广度**堵住"只在一个设置成立"的质疑，用**开源系统**让读者能自己复现。

这个定义是【归纳】档。支持它的证据分布在第 4 节的每一个情境里。这里先给一个直观的例子：

- SIGE（NeurIPS 2022 / T-PAMI）Table 3：空间稀疏计算把 MACs 降了 7.6×，但延迟只降 1.6×。然后一项一项加系统优化：预计算归一化、逐元素融合、scatter 融合、注意力稀疏，延迟才一步步到 2.9×。这张表同时做到了两件事：每一步延迟变化都有归属（可归因），而且报的是真机延迟而不是 MACs（真实条件下兑现）。
- 反面是同一组的 GAN Compression（CVPR 2020）：头条数字是 MACs 降 9–21×，GPU 实测只有 1.7–3.3×（Table 4），作者如实写出了差距，但头条用的是 MACs。

### 1.2 Han Lab 的 solid 不是什么

**不是统计意义上的 solid。** 94 篇里只有 8 篇报告过任何形式的方差或多次运行：GCN-RL Circuit Designer（每个实验 3 次，mean±std）、PockEngine（Table 2/3 的 ±std）、MCUNetV3（附录 B，3 个 seed）、SparseRefine（仅 Mask2Former 3 次）、SPVNAS（附录检测 3 次）、VLASH（真机二项标准误）、Park（Table 1 ±1 std），再加 DiffAugment 报了评测采样的方差（不是训练方差）。其余 86 篇是单次运行，而且大量结论落在 0.1–1 个点的差距上。

**不是"数字处处一致"意义上的 solid。** 九个领域组每一组都有摘要、正文、表格之间数字对不上的情况，按各组统计相加约 40 篇。第 2 节逐条列。

**所以这份文档的用法是：** 学他们的对照结构、学他们怎么把效果落到真实条件，但统计纪律和判据纪律按我们自己的准则（6.2、6.3、6.5）来，比他们严。

### 1.3 一个聪明同行会不会意外

按我们准则 2 的自查：把"Han Lab 的说服力来自对照结构而不是统计"告诉一个做效率研究的同行，他大概不会意外。会让他意外的是两件事：
1. 这个实验室的论文里，**"随机臂"充当零假设**的做法只在极少数论文里出现（AWQ Table 1 是最干净的一例），绝大多数机制主张停在相关性。
2. **同一个实验在同一实验室的两篇论文里报出不同分数**（DC-VideoGen 报 84.83，DC-Gen 报 83.97，同一个 Wan-2.1-14B 迁移实验），而加上"同数据微调原模型"的对照后，前一篇宣称的优势几乎消失。这说明第 4 节情境 3 的"单变量对照"是双刃剑：它既是最强的说服手法，也是最容易把自己的增益抹掉的手法。

---

## 2. 不学清单

下面六条是 94 篇里反复出现、而我们的准则明确禁止的做法。每条给实例、给对应准则条款、给我们自己的替代做法。

### 2.1 不报方差，单次运行下比较 0.1–1 个点的差距

**实例。**
- COAT（FP8 训练）Table 4：OLMo-7B 下游任务 ARC-Easy −3.8、HellaSwag +3.7，两项被平均值抵消，正文称精度持平。
- QeRL Table 2：AIME/AMC 上的差异多为 1–4 道题。
- DGC（Deep Gradient Compression）：自己的基线随 GPU 数量不同就有 0.8% 的波动（Table 2），而它报告的部分增益只有 +0.06%。
- DeltaQuant Fig. 9c：FP8 core 比 BF16 core 高 0.4 PSNR，说明测量噪声约 0.4；同篇 Fig. 9d 的一项消融收益只有 +0.5。
- Long-RL 的 NeurIPS checklist 第 7 项回答"提供了统计显著性"，正文里没有。
- 量子组：QuantumNAS 的 artifact 附录自己写复现差异允许在 10% 以内，而正文报告的很多真机增益只有 1–5 个点。

**对应准则。** 6.3 零假设必须标定；"成立"定义为落在 null 之外。

**我们的做法。** 核心对比至少 3 个 seed；拿不到 3 个 seed 的，像 Lightning-Weave 附录 B.8 那样显式声明"每个配置只训练一次，重复采样只反映解码方差"，不装作有统计。对本领域 ±1 量级的精度差，默认视为无法区分，直到有方差。

### 2.2 在评测集上调参、挑选结果

**实例。**
- QuantumNAS Sec IV-A：剪枝比例"在 5 个比例中报告真机实测精度最高的那个"。
- QOC（量子片上训练）Table 1：在 300 张验证集上报"峰值精度"，没有独立测试集。
- QuantumNAT：测量后归一化用的是测试批自身的统计量，到附录 Table 13 才补验证集统计（精度从 0.67 降到 0.65）。
- SparseRefine Sec 4.1：熵阈值 α 按"恰好追平高分辨率基线"选定，文中只给 val 集结果。
- MCUNetV3 附录 C：每次运行大量调学习率取最好。
- XAttention：逐头阈值用动态规划搜索，没说在哪个数据上评估性能，随后在 RULER 上报告结果并用 RULER 证明它有效（Table 9）。
- Jet-Nemotron：用 MMLU/math/retrieval 准确率搜索全注意力层的放置位置，再用同样的基准报告精度。
- SANA-1.5：判分器 VILA-Judge 和 SFT 数据都用 GenEval 工具包标注，然后在 GenEval 上 best-of-2048 得到 0.96。
- SANA-Sprint：推理时间步、两组噪声分布超参都直接按 MJHQ 的 FID/CLIP 选（Fig. 6/7/8）。
- GAN Compression：fine-tune 与否两个变体里选更好的进摘要。
- CAN：贪心模块选择与评测用同一个集。

**对应准则。** 6.1 合同不冻结不开跑；6.4 先注册预测再看结果。

**我们的做法。** 调参集、选择集、报告集三者分开写进合同；阈值的选定规则在看到报告集之前冻结。SPVNAS 在附录里用验证集重跑搜索，是这批论文里少见的正确处理，可以照做。

### 2.3 标题数字取最有利的配置

**实例，按"有利在哪里"分类。**
- 只算 kernel，不算端到端：Quest 标题 7.03× 是 self-attention 延迟（Sec 4.3），端到端只有 1.74×，叠加 AWQ 后才 2.23×；XAttention 的 13.5× 只是注意力部分；SpAtten 对 GPU 的 162× 是注意力层，端到端版本 SpAtten-e2e 降为 35×/24×（Fig. 15）。
- 用最慢的基线做分母：DC-Gen 的 53.8× 分母是不原生支持 4K 的 FLUX 硬跑 4K 的 218.81s，对真正的 4K 对照 Diffusion-4K 只有 5.9×；SANA Tab. 7 的加速以 FLUX-dev 为 1.0×。
- 只计 transformer 骨干：SANA-Sprint 的 64.7× 按 transformer 延迟；DC-Gen 1K 下 AE 解码 0.29s 已占骨干 0.88s 的三分之一（Tab. 16），端到端约 3.6× 而不是 4.6×。
- 用最大 batch 吞吐：Jet-Nemotron 的 47× 来自各模型各自的最大 batch（Qwen3 为 8，Jet 为 128，附录 A.3）。
- 用 MHA 而不是主流的 GQA：DuoAttention 2.55× 是 MHA 25% 预算，GQA 只有 1.5× 左右，而精度最不理想的也是 MHA 25% 那一档（Table 4 平均 34.49 vs Full 37.52）。
- 取序列扫描的最长端：LServe 在 64K 时只有 1.09×；XAttention 8K 时 <1；Jet 4K prefill 1.14×。
- 单个挑选的样例：SIGE 摘要的 7.2× 来自一个 1.7% 面积的编辑样例，数据集平均 MACs 只降 2.1×（Table 1 vs Fig. 1）。
- 混入别的因素：CAN 的 52×/229× 混入 20 步 vs 250 步的采样步数差；DeltaQuant 的 111.8× 里量化只贡献 3.0×，其余来自蒸馏和注意力优化；SVDQuant 的 8.7×/10.1× 依赖基线需要 offloading。
- 含解析成分：VLASH 的 11.8× 来自最坏反应延迟公式，其中 500ms 执行时长主导；Weave 的 4.58× 是外推，实测 1.70–2.18×。
- 硬件论文的"双层标题数字"：摘要同时给对 CPU/GPU 的两到四位数倍数（24,000×、347×、1285×）和对先前专用加速器的个位数倍数（1.6×、4×、1.3×），只有后者是公平比较（EIE、SpAtten、SpArch、PointAcc）。

**对应准则。** 7.2 坏消息先说；5.4 跨论文推断时区分轴。

**我们的做法。** 标题数字旁边同时给三样东西：最不利设定下的数字、包含全部前置成本的总成本、精度与效率在同一配置下的表。这三样是从 Lightning 系列和 QServe 的做法里反推出来的（QServe Fig. 17 把"更快的 kernel"和"更大的 batch"两个因子拆开报，是正面例子）。

### 2.4 摘要、正文、表格的数字对不上

**实例。**
- PockEngine 对 TensorFlow 的加速：摘要 15×、引言 11×、正文 13–21×、结论 11.2×；"7.9×""5.6×"在实验表里找不到出处；§4 承诺的图优化结果节不存在。
- SANA-Video 的加速倍数有四种说法：16×（摘要）、13×（引言）、53×（§3.4）、8×（Tab. 4）。
- QeRL 附录 K 写"over 2×"，Table 3 是 1.2–1.5×。
- COAT 正文称"持平或超过 TransformerEngine"，Table 8 的 8 卡 Llama-2-7B 是 COAT 1.36× 对 TE 1.42×。
- NAAS 摘要说比 Eyeriss 的 EDP 降 4.4×，正文说 4.88×。
- MCUNet Table 2 与 Table 3 在同一约束下，S-MbV2+CMSIS 一个写 35.2，一个写 39.7。
- MCUNetV2 的检测表沿用了 V1 用 YOLOv2 得到的 51.4%，而 V2 自己用的是 YOLOv3。
- DC-VideoGen 与 DC-Gen：同一实验 84.83 对 83.97。
- 文字结论超出表格：DeltaQuant Table 1 里 RTN 的 Vision Reward 0.093 高于 Ours 0.083，正文称 consistently；FourTune Table 5 块量化 LPIPS 0.227 对 0.203 被称为 comparable；FastComposer 称"prompt 一致性更好"，但 Table 1/2 里 SD 和 Custom Diffusion 的一致性更高；VILA 称"consistently outperforms"，POPE 两个尺寸都低于 LLaVA-1.5。
- 跨论文搬数字时条件悄悄变：PockEngine Table 2 的 MCUNet 行与 MCUNetV3 Table 1 逐项相同，但配置标签不同；TinyML Magazine 的 149KB/2077× 对 MCUNetV3 的 141KB/2300×。

**对应准则。** 7.4 主动报告自己的错误。

**我们的做法。** 投稿前做一次"数字对账"：把摘要和引言里每个数字回表格定位，做不到的删掉。头条倍数只允许一个口径，口径写在数字旁边。

### 2.5 核心机制主张缺反事实对照

**实例。**
- QeRL："量化噪声促进探索"。主对比里 QeRL 学习率 1e-5、BF16 LoRA 5e-6（作者给的理由是 BF16 LoRA 更大的学习率不稳定），而且没有"不量化、加同样幅度噪声"的对照。所以这个主张停在相关性。
- LongLive："长调优是必要的""sink 要先长调优才有效"，两条都没有对照实验，而且与 Self-Forcing 已发表的结论相反。
- FastComposer："注意力不受约束是身份混合的主因"，只有可视化（Fig. 4）。
- VLASH：缺"只前推状态 vs 只偏移微调"的拆分。
- ForeAct：缺"真实未来帧"的 oracle 上界，也缺"只给图像不给文本"。
- Lightning-OPD 2.0：缺"只减全局均值"的零假设对照。
- QVG（Quant-VideoGen）：动机是时空邻近性，方法是语义 k-means，两者之间没有对照。
- DistriFusion：测的是输入相似而不是逐层激活相似，而方法复用的是逐层激活。

**对应准则。** 5.3 标注三档；6.3 零假设必须标定。

**我们的做法。** 每一条机制主张配一个"如果机制不成立，这个实验会怎样"的反事实。做不出来的，标为【推测】档写进论文的局限。

### 2.6 效率只用代理（FLOPs、MACs、仿真、解析模型），不上真实硬件

**实例。**
- Pruning（NeurIPS 2015）只有参数量和 FLOP。
- Lite Transformer 只有 Mult-Adds，没有任何移动端实测延迟，后来被同组的 HAT 批评。
- HAQ 用模拟器；APQ 用查表估延迟但没给校准数据；DGC 的 64 节点加速比来自性能模型推算；Deep Compression 量化后的模型从未在硬件上跑过（Sec 8 自承）。
- TinyTL 全部是理论内存（§4.1 自承）；MCUNetV3 的精度在 GPU 上用 batch 128 模拟 batch 1（VWW 上差 2.5%）。
- 硬件组没有一篇流片。EIE、SpAtten、SpArch、PointAcc、LEGO 都是"周期仿真器 + 部分模块 RTL 综合 + CACTI/Ramulator/文献单位能耗拼装"；SpAtten 用 45nm 的浮点单元数据估 40nm；NAAS 是纯解析模型。工艺归一化不一致：SpArch 40nm 对 OuterSPACE 32nm 不归一，PointAcc 40nm 对 Mesorasi 16nm 不归一，LEGO 把流片的 Eyeriss、投影的 NVDLA、自己的综合结果放一张表。
- "边缘"动机但只在桌面 GPU 上测：SPVNAS（1080Ti）、BEVFusion（RTX3090）、SparseRefine（RTX3090）。

**对应准则。** 2 可行性；6.2 判据必须离线预演。

**我们的做法。** 效率主张必须有一条真实测量；代理只能用于搜索，报告时用实测，并单独给代理的校准图（第 4 节情境 7）。

### 2.7 其他反复出现、但危害较小的问题

- **只与 dense 或 FP16 比，缺同预算的竞品。** AWQ Table 5 的 Mistral/Mixtral 只对 FP16；SmoothQuant Table 6/7 的新模型只对 FP16 且逐模型调 α；LServe 的精度只和 Dense 比。这些"泛化"证据只能说明损失小，不能说明优于他人。
- **最接近的工作缺席。** G5 组 10 篇里 7 篇没有和同期方法做实验对比；COAT 没有 FP8-LM；AWQ 没有 SpQR/OWQ；DC-AR 明确不与 VAR 类比较，而 VAR/HART 正是直接竞品。
- **自建评测集偏向本法。** StreamEval 的答案永远在 20 行前；LongVideo-Reason-eval 与训练数据同一管线；Inf-Streams-Eval 与训练数据同为体育解说；LongLive 的交互评测 prompt 与训练 prompt 同由 Qwen 生成。
- **评委与数据同源。** StreamingVLM 用 GPT-5 清洗数据又用 GPT-5 当评委。
- **训练数据与评测分布重合未讨论。** HART、DC-AR、LPD 都用 MidJourney 风格数据训练，在 MJHQ（Midjourney 图像）上评测，三篇都没讨论。
- **局限性章节普遍缺失。** 量化组 9 篇只有 QeRL、SVDQuant 写了实质局限；硬件组除 EIE 和 retrospective 外全无；生成组 11 篇只有 4 篇有一句话形式的局限。

---

## 3. 一篇 Han Lab 论文的标准骨架

把 94 篇的证据流叠在一起，可以画出一条几乎所有系统/效率类论文都走的路线。这一节是【归纳】档，但每一步都给出走这条路线的论文。知道了骨架，第 4 节的各情境手法就有了位置。

```
(1) 反直觉的开场对照图
     常用指标变好了，真正的瓶颈没动
         │
(2) profiling / 资源账本，把瓶颈收窄到一小块
         │
(3) 关键观察（机制）：可视化 → 干预 → 得到一个可操作的判据
         │
(4) 方法：直接利用 (3) 的判据；常附一个短推导或定理
         │
(5) 先证明"理想上界可达"（可能用硬件做不了的方案）
         │
(6) 再给可部署的等价实现，并先承认朴素实现会慢
         │
(7) 主表：同配方、只换一个组件；两个方向都比（同预算比精度、同精度比效率）
         │
(8) 逐步叠加的分解表（roadmap），每一步给精度和效率
         │
(9) 沿规模扫描（分辨率、长度、batch），标出优势消失的区间
         │
(10) 广度：跨模型族 × 跨尺寸 × 跨任务 × 跨硬件
         │
(11) 端侧 / 实机部署一节，把效率落到用户秒数或"装得下"
         │
(12) 开源 + 被框架采纳 + demo
```

**走完整条路线的论文。** AWQ、SmoothQuant、QServe、SVDQuant（量化组，第 1–12 步最完整）；MCUNet 系列（TinyML 组）；SpAtten、SIGE。

**各步的典型实例（详细展开见第 4 节）。**
- (1) MCUNet Fig. 1：MobileNetV2 参数比 ResNet-18 小 4.6×，峰值激活反而大 1.8×。
- (2) MCUNetV2 Fig. 1：峰值内存只由前 5 块决定；BEVFusion §III-B：BEV pooling 占运行时 80% 以上。
- (3) StreamingLLM Fig. 3 → Table 1：先定位 PPL 突增的精确时刻（首 token 被逐出），再用"\n"替换实验区分语义与位置。
- (4) SpAtten Eq. 2：softmax 导数推出量化误差上界 2p(1−p)，把 Fig. 7 的散点观察变成机制。
- (5) SmoothQuant Table 1：先模拟硬件做不了的 per-channel 激活量化，证明问题出在通道维且修好就无损。AWQ Table 1：1% 通道保留 FP16 的混合精度，硬件不友好但证明显著通道存在。
- (6) SVDQuant Fig. 6a：低秩分支的朴素实现开销 57%，kernel 融合后 5–10%。
- (7) DC-AE Tab. 3：同一 DiT-XL/UViT-H 下只换自编码器。
- (8) SIGE Table 3、SpAtten Fig. 20、QServe Fig. 16。
- (9) SIGE Fig. 10：编辑比例扫描，3090 上约 70%、i9 上约 45% 是盈亏点。
- (10) AWQ：OPT/LLaMA/Llama-2/Mistral/Mixtral/VLM/代码/数学 × 4090/Orin/4070/A100/Raspberry Pi。
- (11) AWQ：Llama-2-70B 上 Jetson Orin；SmoothQuant：530B 单节点；QeRL：32B RL 单张 H100。
- (12) AWQ 被 HF Transformers、TensorRT-LLM、vLLM、LMDeploy 采纳；StreamingLLM 被 TensorRT-LLM、HF Transformers、MLC LLM 采纳。

**骨架的变体。**
- **纯算法/现象类**（StreamingLLM、DiffAugment、NetAug）：省掉 (5)(6)(11)，但 (3) 做得更重，并且在 (8) 的位置换成"排除替代解释"的实验（第 4 节情境 2）。
- **硬件类**（EIE、SpAtten、SpArch、PointAcc、LEGO）：(6) 换成"仿真器 + 局部综合 + 查表"的评估栈，(7) 换成与先前加速器的资源对齐比较，(10) 换成"通用平台全谱系 + 数量级加速"。
- **平台/议程类**（Park、DGA 的问题重构、TinyML Magazine）：(1)(2) 换成"重新划分问题空间，每一类配一个最小证据"。
- **后训练/迁移类**（DC-Gen、DC-VideoGen、SANA-Sprint、SANA-1.5 剪枝）：(7) 必须有"同数据同超参微调原模型"的对照，否则数据效应和方法效应分不开。

---

## 4. 按情境分的实验设计手法

每个情境下的每条手法按同一格式写：**做法 → 实例（论文、位置、数字）→ 为什么有说服力 → 常被做坏的方式 → 我们怎么用**。"为什么有说服力"和"我们怎么用"是【归纳】档，实例是【读到】档。

---

### 情境 1：要立一个问题或痛点

读者的默认态度是"这个问题要么不存在，要么已经解决了"。Han Lab 对付这个态度的办法不是引用别人说问题重要，而是**用一张自己测出来的图让读者亲眼看到问题**。

#### 1.1 反直觉的开场对照：常用指标变好了，真正的瓶颈没动

**做法。** 挑一对模型或一对设置，常用指标（参数量、FLOPs、MACs）差很多，但真正卡住部署的量（峰值激活、训练激活、实测延迟、DRAM 访问）几乎不变甚至反向。把这两个量并排画在第一张图里。

**实例。**
- MCUNet（NeurIPS 2020）Fig. 1：同为约 70% ImageNet 精度，MobileNetV2 比 ResNet-18 参数小 4.6×，但峰值激活大 1.8×。由此把痛点从"模型大"改写成"峰值激活大"，后面的 TinyNAS 就围绕峰值激活做。
- TinyTL（NeurIPS 2020）Fig. 1 右：MobileNetV2-1.4 比 ResNet-50 参数小 4.3×，训练激活只小 1.1×。再配一行反向传播公式，说明为什么只训偏置就能省激活内存。
- SPVNAS（ECCV 2020）§4.1：MinkowskiNet 通道缩 4×/8×，MACs 降到 7.5G/1.9G，延迟只从 105ms 降到 96ms。
- LitePose（CVPR 2022）Fig. 8：LitePose-L 的 MACs 更高，但延迟与 Scaled-HigherHRNet 相近。
- SpAtten（HPCA 2021）Fig. 2 / Table IV：GPT-2 的注意力只占约 10% FLOPs，却占 50% 以上延迟；注意力内部 73% 时间花在 split heads、reshape、transpose 这类数据搬运上。
- HAT（ACL 2020）Fig. 2：在三种硬件上扫描 Transformer 的维度，同 FLOPs 延迟差异很大，而且不同硬件敏感的维度相反（GPU 对层数敏感，ARM 对宽度敏感）。
- PVCNN（NeurIPS 2019）Fig. 2(b)：点云方法的"有效计算"只占运行时的 12–45%，其余是不规则访存和动态核。
- NetAug（ICLR 2022）Fig. 1 左：同样的正则化，对 ResNet-50 有益、对 MobileNetV2-Tiny 有害。
- Lite Transformer（ICLR 2020）Fig. 2：计算分解显示 FFN 吃掉了大部分 Mult-Adds，而不是注意力。
- QServe（2024）Fig. 2b：W4A4 理论上更快，实测反而更慢。

**为什么有说服力。** 它同时做了三件事：证明问题存在（不是作者编的）；证明已有方法没解决（它们优化的是错的指标）；把方法要优化的那个量定下来。读者在第一页就接受了"这个量才是关键"，后面方法做什么都顺理成章。

**常被做坏的方式。** 只给常用指标，不给真瓶颈的量；或者给了真瓶颈的量但没有并排对照，读者看不出"错配"。

**我们怎么用。** 任何痛点都要能画成"X 变好了，Y 没动"的一张图，X 是别人在优化的量，Y 是我们要优化的量。画不出这张图，说明痛点还没找准（准则 2 的"痛点"判据）。

#### 1.2 用 profiling 或资源账本把瓶颈收窄到一小块

**做法。** 在真实硬件上做逐层、逐算子、逐块的分解，找出瓶颈集中在哪里。方法只针对那一小块，其他部分不动。

**实例。**
- MCUNetV2（NeurIPS 2021）Fig. 1 + 附录 Fig. 11：逐块峰值内存图显示，峰值只由前 5 块决定，在 4 个骨干上重复验证。于是方法只对前几块做 patch-based 推理。
- MCUNet Fig. 11：0.3× MobileNetV2 的单块峰值激活是平均值的 2.2×。
- MCUNetV3（NeurIPS 2022）Fig. 11(a)：逐层激活代价 vs 权重代价，前段激活贵、后段权重贵、中段便宜，由此决定稀疏更新的层选择。
- BEVFusion（ICRA 2023）§III-B：BEV pooling 占运行时 80% 以上（>500ms vs 其余约 100ms），于是写一个专用 kernel 把它加速 40×。
- SIGE Sec. 3.2：朴素 gather/scatter 在一个 15.5% 编辑样例上占 41% 开销（0.17ms / 0.42ms），由此引出 kernel 融合。
- TorchSparse（MLSys 2022）Fig. 4：数据搬运 40–50%、GEMM 20–50%、mapping 15%。
- DeltaQuant Fig. 3：注意力优化之后线性层占 65%。
- COAT Fig. 1c + Table 2：逐类显存分解，算子级激活账本（单位 U），算出理论上限 1.69×。
- AWQ Fig. 3：Llama-2-7B 在 4090 上，生成 20 token 310ms，上下文 200 token 只要 10ms；roofline 显示生成阶段算术强度约 1，memory-bound，所以做 weight-only 量化而不是 W8A8。
- TLT（推理 RL 训练）：自采 trace 显示 rollout 占约 85% 步时且长尾，再加上 ByteDance 的第三方生产 trace（Fig. 1a、Fig. 2）和 roofline（Fig. 5c）。
- 硬件组的单位能耗账：EIE Table I（DRAM 640pJ 对 SRAM 5pJ）、PVCNN Fig. 1（DRAM 640pJ 对 MAC 3pJ）。

**为什么有说服力。** 分解之后，读者能自己算出"如果这一块被解决，整体能快多少"，方法的收益上限在动机阶段就被预告了。这也让后面的实测可以和这个上限对账（情境 4）。

**常被做坏的方式。** profiling 停留在文字声明。MCUNet Sec. 2 说解释器式框架额外占"最高 65% 峰值内存、延迟 +22%"，没有图表号、没有测量方法，是作者声称而不是证据。

**我们怎么用。** 动机实验要产出一个数字："瓶颈占比 X%"。这个数字后面要被用来算理论上限，并与实测对账。

#### 1.3 用自家上一篇在新设定下的失效立新题

**做法。** 展示自己（或自己实验室）上一篇方法在一个新约束下不工作，把它作为新论文的动机。

**实例。**
- SPVNAS Table 1：PVCNN 在大场景上只有 39.0 mIoU 或要 35.6 秒/帧。
- LServe Fig. 6：Quest 在大页下失效；同时 Table 1 显示大页更快。两个实测结果互相矛盾，分层分页正好化解。
- MCUNetV2 附录 Fig. 11：专为 MCU 设计的 MCUNet 也内存失衡。
- MCUNetV3 §1：MCUNet/TinyTL 的理论节省在现有框架里兑现不了（TinyTL 只给过理论值）。
- Fast-dLLM → v2：近似缓存是 training-free 的，但不精确；v2 用块扩散训练让缓存变精确。
- Lightning-OPD → 2.0 → Weave：第一篇要求教师一致性，第二篇放松它去掉风格偏差，第三篇做多教师组合。
- DC-AE → DC-AE 1.5：Fig. 1b 两条反向曲线，通道越多重建越好、生成越差。
- TorchSparse++（MICRO 2023）把前作 TorchSparse 列为被超越的基线（2.0–2.2×）。

**为什么有说服力。** 作者对上一篇的了解最深，失效的证据最可信；读者也知道这不是在攻击别人。

**常被做坏的方式。** 后作一句话削弱前作而没有表格支撑：TorchSparse++ Sec. 5.2 声称换用它做后端后 CenterPoint 在 Orin 上比 FlatFormer 快 1.5×，没有表，这直接削弱了 FlatFormer"首个比稀疏卷积更快的点云 Transformer"的卖点。

**我们怎么用。** 这和我们的"前一轮合同的失败就是下一轮的动机"是一回事。要注意的是：上一篇的失效要用新论文同样的协议重新测，不能引用旧数字。

#### 1.4 把问题空间重新划分

**做法。** 当痛点不是一个具体的瓶颈而是一类问题时，先说明现有划分为什么错，给出新划分，每一类配一个最小证据。

**实例。**
- DGA（Delayed Gradient Averaging）：把联邦学习的问题重构为"带宽已被压缩解决，延迟是物理极限"，再用 FedAvg 吞吐随延迟 1μs→1s 下降的扫描（Fig. 1c）和运行时代价模型（Eq. 1）支撑。
- Park：列出 RL-for-systems 的六类独特挑战，每类配一个最小证据（Fig. 1 输入驱动方差、Fig. 2 仿真与现实差距、Table 1 GCN vs LSTM），再把 12 个环境映射上去。
- Quant-VideoGen：把 KV cache 从显存问题重新定义为长视频能力问题。
- Anycost-GAN：把"压缩"改写成"一致性预览"，一致性做成两个指标。

**为什么有说服力。** 它改变了读者评价后续工作的标准。改完之后，方法只需要在新标准下赢。

**常被做坏的方式。** 新划分没有最小证据，只是修辞。

**我们怎么用。** 这是准则 3.2 里"假设反转"和"未命名的量"的实操版本：定义新量的同时，必须给出至少一个用旧量看不见、用新量看得见的例子。

---

### 情境 2：要声称发现了一个现象或机制

这是 Han Lab 最有辨识度的部分，也是最容易学歪的部分。他们的强项是把一个现象变成一条可操作的判据；弱项是这条链常停在相关性。

#### 2.1 四步链：可视化 → 干预或反事实 → 方法直接利用 → 回到同一个量上展示修复

**做法。** 第一步，把现象画出来（注意力图、分布图、逐层曲线）。第二步，设计一个只改变一个因素的实验，证明现象是因果的。第三步，方法直接利用这个因果事实。第四步，在同一个量上展示方法修复了现象。

**实例，按四步拆开。**

StreamingLLM（ICLR 2024）：
- 可视化：Fig. 2，Llama-2-7B 各层平均注意力 logits 热图，除前两层外都集中在首 token。附录把同一观察扩展到长度 4096（Fig. 12，含跨 head 标准差）、70B（Fig. 13）、BERT（Fig. 14）。
- 定位失败时刻：Fig. 3，PPL 突增的时刻正好是缓存满、首 token 被逐出的时刻。
- 反事实：Table 1，把首 4 个 token 换成换行符"\n"，PPL 5.60 对原始 5.40，而不保留首 token 是 5158。两个互斥的解释（首 token 的语义重要 vs 首 token 的位置重要）由此分开。
- 方法直接利用：保留 4 个初始 token 的 KV + 滑动窗口。
- 升格为处方：Sec. 3.3，从零预训练 160M 模型时加一个可学习 sink token，流式只需 1 个 sink（Table 3），并带上 SoftMax-off-by-one（Zero Sink）作为竞争假说的对照。

DuoAttention（ICLR 2025）：
- 可视化：Fig. 1 左/中，两类注意力头的示例。
- 干预：Fig. 1 右，按三种顺序把头换成 streaming attention：先压 retrieval 头、先压 streaming 头、随机。先压 retrieval 头时 passkey 准确率迅速崩溃，先压 streaming 头几乎不掉。
- 方法直接利用：把定义写成可操作判据（"受限后输出显著改变"），用门控蒸馏在合成数据上优化识别。
- 消融证明替代定义更差：Fig. 13(1)，用注意力分数 profiling 识别头的效果比优化识别差。

DC-AE（ICLR 2025）：
- 诊断量：§3.1 嵌套容量实验，高压缩模型包含低压缩模型为子网、潜变量总量相同，重建反而更差，归因到优化困难。
- 修复：Residual AE。
- 回到同一张图：Fig. 3a 在同一实验上展示修复后的变化。

DC-Gen（2025）：
- 诊断量：Fig. 2a，逐层特征与原模型的距离，未对齐时从 0.24 增到 345.42，对齐后 0.02 到 12.51。
- 反事实：Fig. 2b，仅对齐 patch embedder 和 output layer 两层、骨干冻结，就能生成语义正确的图。
- 失败模式先展示：Fig. 4/5，直接微调在三类任务上分别出现模糊、复制输入、崩成噪声，并引用 OpenSora-2.0 原文自述"video outputs appear blurry"作外部佐证。

DiffAugment（NeurIPS 2020）：
- 诊断量：判别器的训练/验证准确率差（Fig. 1/6），证明过拟合是病因。
- 两种朴素增强定量失败：Table 1，只增强真实图有伪影、只增强判别器输入导致 G/D 失衡。
- 方法作用于病因：同一个诊断量在方法下收窄。
- 排除替代解释：Fig. 8，扫描模型容量（"只是模型太大？"）和 R1 正则强度（"只是正则不够？"）。

其他走这条链的：Quest Fig. 4（Top-10 召回率量化"关键 token 随 query 变化"）；XAttention 几何论证 + Table 6 同采样预算的随机/对角对照；Radial Attention Fig. 8(b) 把时空衰减拟合成指数（R²>0.985）再推出复杂度；Jet-Nemotron Fig. 5(a)（逐层"只留一层全注意力"热图）→ Fig. 5(b)（学习放置 vs 均匀放置）；LongLoRA Table 2（LoRA rank 8→256 PPL 不降，只打开 norm/embed 就降，而 norm 只占 0.004% 参数）；SANA-Sprint Eq. 10 + Fig. 3b（定位 c_noise=1000t 的 1000 倍放大 → 梯度范数曲线前后对比）；LPD Eq. 3 + Fig. 7（定义 Per-Token Attention，5 万张图统计局部性）；Fast-dLLM Fig. 3（KV 激活跨步余弦相似度热图，结构直接对应块级缓存）；SIGE Fig. 4/6（逐层 |ΔA| 聚集在编辑区）。

**为什么有说服力。** 第二步把"现象"和"方法"之间的因果链闭合了。没有第二步，审稿人可以说"现象是真的，但你的方法有效可能是别的原因"；有了第二步，方法的有效性是现象的推论。

**常被做坏的方式。** 停在第一步，第二步用可视化代替干预。第 2.5 节的实例都是这种。另一种做坏：第二步的干预实验与方法用的量不是同一个（DistriFusion 测的是输入相似，方法复用的是逐层激活）。

**我们怎么用。** 机制主张的实验设计从第二步开始写：先写"如果机制不成立，哪个实验会怎样"，再写可视化。这就是我们的 6.2（判据必须能被满足也能不被满足）。

#### 2.2 随机臂充当零假设

**做法。** 当主张是"某类元素更重要"时，设三臂：按新准则选、按教科书准则选、随机选，三者保留同样比例的元素。随机臂是零假设，教科书臂是"已有方法"。

**实例。**
- AWQ（MLSys 2024）Table 1：OPT-1.3B/6.7B/13B，INT3-g128，保留 0.1%/1%/3% 通道为 FP16。按激活幅度选 1% 时 OPT-6.7B 的 PPL 从 23.54 降到 11.39；按权重幅度选是 22.37，随机选是 24.23，两者基本等于不保留（RTN）。这一张表把"激活感知"从"保留了通道"中剥离出来。
- DuoAttention Fig. 1 右的第三条曲线就是随机压缩头。
- XAttention Table 6：同采样预算下的随机模式和对角模式对照。
- MCUNet Table 6：优化的搜索空间 78.7 对随机采样的搜索空间 74.7±1.9 对超大空间 77.0（这是全实验室少数带 ± 的数字）。
- HAT Fig. 9：进化搜索对随机搜索（但只比验证损失）。
- SPVNAS Fig. 8：进化搜索对随机搜索，20 个随机模型重训。
- MCUNetV3 附录 Fig. 13：进化对随机。

**为什么有说服力。** 随机臂给出了"任意选择"能达到的水平，新准则的增益必须高于它才算数。这正是我们准则 6.3 要求的"用对照条件标定 null 分布"。

**常被做坏的方式。** 搜索类论文几乎都不和随机搜索比：HAQ、AMC、APQ、ProxylessNAS 都没有随机搜索对照。于是"RL/进化/可微带来增益"和"任何自动搜索 + 好的搜索空间带来增益"无法区分。

**我们怎么用。** 任何"选择"类方法（选层、选样本、选头、选通道）必须有随机臂，且随机臂要跑多次给出分布，不是一个点。

#### 2.3 子群分析：机制成立的话，增益应该集中在某个子集

**做法。** 事前按机制预测哪类样本受益最大，然后按这个分组报告增益。

**实例。**
- TSM（ICCV 2019）Sec 5.1：事前把数据集分成"时序弱"（Kinetics/UCF/HMDB）和"时序强"（Something-Something/Jester）两组，预测 TSM 在后者增益更大；Table 1 结果吻合（+3.5 到 +8.8 对 +11.7 到 +31.3）。
- SPVNAS Table 5 + Fig. 6：小物体类的增益，以及点分支的激活比例。
- BEVFusion Table IV + Fig. 4：雨天、夜间、小物体、远物体、1/4/16 线 LiDAR 下的增益。
- SparseRefine Table 8：小类（person/rider/pole）的增益。
- TSM Table 7：快/中/慢物体分组 mAP，快速物体 +4.6。
- Quest Table 1："问题部分逐 token 解码"的 passkey 设置专门暴露逐出类方法的缺陷。

**为什么有说服力。** 这是一个事前注册的预测（准则 6.4）。如果增益均匀分布在所有子群，机制解释就被否证了。

**常被做坏的方式。** 事后找一个增益最大的子群来讲故事。TSM 的做法是事前分组，是正例；BEVFusion 的分组是否事前没有写明。

**我们怎么用。** 子群预测写进合同，和主判据一起冻结。

#### 2.4 把经验观察配一个短推导

**做法。** 对观察到的规律给一个几行的推导或一个小定理，把现象升格为机制。

**实例。**
- SpAtten Eq. 1–2：softmax 导数推出 4-bit 量化误差的上界与 2p(1−p) 成比例，解释 Fig. 7 的散点（注意力最大概率越大，量化误差越小）。
- QuantumNAT 定理 3.1：噪声对测量结果的作用等于线性缩放加平移，由此推出测量后归一化。
- AWQ Eq. 2–3 + Table 2：推导出放缩 s 后显著通道的相对误差约为 Δ'/Δ · 1/s，再实测推导里的近似（s<2 时 Δ 变化 <5%），并用 s=4 时 21.2% 的组 Δ 变大解释为什么 s 不能无限大（PPL 在 s=2 最好 11.92，s=4 回升到 12.36）。
- Radial Attention 附录 A：从指数衰减模型推出 O(n log n) 复杂度和误差界。
- COAT：幂函数在一组公理下唯一的证明。
- Lightning-Weave：用一个两动作反例证明朴素离线目标没有不动点。

**为什么有说服力。** 推导给出了可检验的预言（AWQ 的拐点、SpAtten 的误差与 p 的关系），实验验证预言比验证"方法有效"更难被归因到巧合。

**常被做坏的方式。** 定理的条件与实际设置不符，而且没有实验检验条件是否成立。G5 组的发现：Fast-dLLM 定理 1 要求 (n+1)ε ≤ 1，但 factor 策略取到 1.9，超出保证区间；Lightning-OPD 的 χ² 是序列级量，实证只测逐 token；Weave 的实现用的是代理损失，作者自承不继承证明的等价性。

**我们怎么用。** 定理必须配一个检验其前提或界的实验。AWQ Table 2 是模板：把推导中的近似单独测出来。

---

### 情境 3：要证明某个组件是必要的

#### 3.1 同配方、只换一个组件的对照（全实验室最一致的手法）

**做法。** 固定骨干、训练配方、数据、步数、超参，只替换被研究的那个部件。主表里新方法和这个对照并列。

**实例。**
- DC-AE Tab. 3：同一 DiT-XL/UViT-H/UViT-2B 下只换自编码器（Flux-VAE/Asym-VAE/SD-VAE/DC-AE）。
- DC-AE 1.5 Tab. 2/3：同 f/c 配置下 DC-AE 对 DC-AE 1.5，跨 4 个骨干。
- DC-VideoGen Tab. 2、DC-Gen Tab. 19：同一 Wan-1.3B + 同一迁移流程，只换视频自编码器。
- LPD Tab. 1/2/3："Raster Counterpart"，同 tokenizer、同架构、同配方，只改解码顺序。
- HART Tab. 5（去掉残差扩散的 HART-VAR）、DC-AR Tab. 4（discrete-only）：去掉核心组件的同配方对照。
- TSM Sec 5.2：TSN 对 TSN+TSM，"唯一的区别是有无 TSM"，跨 6 个数据集。
- TSM Table 5 的 I3D_replace：把每个 TSM 换成 3×1×1 时序卷积，排除骨干差异。替换后 FLOPs 更高（48G 对 33G）、延迟更高（28.0 对 17.4ms）、精度更低（44.9 对 45.6）。
- VILA Table 3 + Sec. 4.4：MMC4 对 MMC4-pairs，只拆掉"交错结构"这一个属性，把结构效应和文本分布效应分开。
- Fast-dLLM v2 Table 1：Qwen2.5-Nemo-FT，同数据同步数的标准微调对照。
- ForeAct Sec 4.3：VLM+π0，同 VLM、同子任务数据，只差前瞻图像。
- TLT Fig. 11：TLT-Base（只开 n-gram 投机解码）作为去掉自适应 drafter 的内部基线。
- Lightning-OPD Sec 4.1：标准 OPD 与 Lightning OPD 超参完全相同，只差 rollout 来源。
- DGA Table 2：把 FedAvg 的本地步数 K 提到 10、20 来加强基线，再展示 non-IID 下掉点。
- VILA-U Table 7：同样的数据，只加对比损失，两行对照。
- GAN Compression：每处都并列"同 MACs 均匀缩通道、从头训练"的朴素基线。
- AMC Table 2 + Fig. 2：剪枝算法和 FLOPs 固定，只换"谁来定逐层剪枝率"。

**为什么有说服力。** 增益被钉到唯一的变化上。这是"可归因"的直接实现。

**常被做坏的方式。** 关键变量绑在一起没分开。APQ 的"联合搜索"与"更大搜索空间"混在一起（作者在 Fig. 5 图注自承）；ProxylessNAS 的"去掉代理"与"非重复 block 的大空间"混在一起；OFA 的子网精度混入了知识蒸馏和约 600 epoch 训练；HAQ 的"RL"与"允许混合精度"捆在一起；HART 的 31% 改进（7.85→5.38）混入了 tokenizer 本身的改进，只变残差扩散这一项时是 6.1%（5.73→5.38）；DC-AE 1.5 的"4× 更快"来自 f64 对 f32，是 DC-AE 已有的能力。

**我们怎么用。** 主表的每一行与新方法之间只允许一个差异。差异超过一个的行，放到泛化表而不是主表。

#### 3.2 后训练/迁移类方法必须有"同数据微调原模型"的对照

**做法。** 在新方法使用的同一份后训练数据上，用相同超参微调原模型，作为对照行。这把"数据带来的提升"从"方法带来的提升"里剥离出去。

**实例。**
- DC-Gen Tab. 4/5 的 "Tuned" 行。§4.2 明说 "To rule out the influence of training data"。编辑任务上尤其关键：原 Qwen-Edit 5.81 → Tuned 7.23，DC-Gen 7.20。不加 Tuned 行的话，DC-Gen 会看起来大幅超过原模型。
- 反面是 DC-VideoGen：同一个 Wan-2.1-T2V-14B 迁移实验，没有 Tuned 对照时报 84.83 并称 "superior"；DC-Gen 加上 Tuned 对照（83.94）后，DC-Gen 自己是 83.97，优势只剩 0.03。
- NetAug Fig. 3 + Table 4：基线训练 300/600 epoch，证明 NetAug 不是"多训了"。
- OFA Table 1：MobileNetV2 #1200，把基线训练到与 OFA 相同的 1200 GPU 小时，73.5% 对 OFA 76.0%。
- HAT Sec 3.3：基线 Transformer 用与搜索出子网完全相同的训练设置重训。

**为什么有说服力。** 没有这个对照，任何微调方法都会"提升"，因为多训了。

**常被做坏的方式。** 不做。DC-VideoGen 是同一实验室同一实验的直接反例。

**我们怎么用。** 我们的防御实验里，"干净孪生"就是这个对照。规则是：任何涉及额外训练的方法，必须有等算力、等数据的对照臂，而且对照臂排在队列最前（这是我们已经踩过的坑）。

#### 3.3 方法在某个参数下正好退化为基线

**做法。** 把方法设计成一个参数族，其中一个取值精确等于基线。这样对比在构造上公平，也降低了采纳门槛。

**实例。**
- DGA Sec 2.2 末：延迟 D=0 时精确退化为 FedAvg。
- Lightning-Weave Sec 3.3 / App B.2：one-hot 权重时精确退化为单教师训练。
- Lightning-OPD Sec 3.1：离线梯度是重要性采样分解中 w≡1 的特例。
- Fast-dLLM Fig. 5 / v2 Sec 4.2：阈值取 1.0 时退化为非并行解码。
- TLT 设计原则 (a)："mathematically lossless，不改 RL 算法"。
- AWQ/SmoothQuant：放缩因子折进前一层，推理时零开销。

**为什么有说服力。** 审稿人无法说"你的基线没调好"，因为基线就是你的方法的一个点。

**常被做坏的方式。** 宣称"零改动"但实际改了。VLASH 真机实验加了状态投影层（Sec V-B、附录 III），"无架构改动"不严格成立。

**我们怎么用。** 设计防御或检测方法时，先问"这个方法的哪个参数取值等于不防御"，把那个点作为对照臂写进合同。

#### 3.4 逐步叠加的分解表，并保留"单独加上无效甚至为负"的那一步

**做法。** 从基线开始，一次加一个组件，每一步同时报精度和效率。关键是不隐藏那些单独加上没收益的步骤。

**实例。**
- SpAtten Fig. 20：专用通路 22.1× → +token 剪枝 1.1× → +head 剪枝 1.1× → +专用 top-k 引擎 3× → +静态量化 1.6× → +渐进量化 1.7×。作者解释剪枝单独只有 1.1× 是因为 top-k 成了瓶颈。这让消融变成了"软硬协同是必要的"的论证。
- SpArch（HPCA 2020）Fig. 2/16：流水合并第一步是 0.18×（慢 5.7×），然后 condensing 8.8×、Huffman 1.5×、预取 1.8×。每项技术写成"修补上一项的副作用"，并用解析的 DRAM 访问模型逐步对上实测。
- SIGE Table 3：无 → Sparse（MACs 7.6× 但延迟只 1.6×）→ +Norm → +Elem → +Sct → +Attn，DDPM 54.6ms→19.1ms，SD 1.1×→4.8×。
- TorchSparse Table 3 / Fig. 13：FP16 理论 2×，单独用只有 1.17×，加向量化后 1.93×，加局部性感知到 2.86×。
- FourTune Fig. 9：4-bit 无融合仅 1.25× → +LoRA 融合 1.82× → +MLP 融合 2.52×。
- QServe Fig. 16：W8A8 → +W4 → +旋转/平滑 → +clip → +KV4 → +SmoothAttention → +渐进量化 → +重排，每步同时报 PPL、吞吐、显存。
- MCUNet Fig. 5：代码生成 → 专用 im2col → 算子融合 → 循环展开 → tiling，MMAC/s 52→82，峰值内存 160→76→48。
- MCUNetV3 Fig. 1：TF/PyTorch → TTE → +QAS → +稀疏更新 → +重排，对数轴把 2300× 拆成四段，每段有章节号。
- LitePose Table 3：核大小 → 融合头 → 超网 → 搜索 → 蒸馏，在两个规模上重复。
- BEVFusion Fig. 3(c)：LSS → 预计算 1.9× → 区间归约 22.1×。
- Fast-dLLM Table 1/2：base / +Cache / +Parallel / +Both 的 2×2 表。
- Lightning-Weave Table 4a：{在线、朴素离线、Tilted} × {三种教师} 的 3×3 网格。
- QuantumNAT Table 1：从 Baseline 开始在 5 台真机上逐级累加三个组件。
- Deep Compression Table 2–5：逐层的剪枝 → +量化 → +Huffman 三列。
- CAN Table 1：DW Conv → +Patch Emb → +Out Proj，并附被否决的分支 ✗。

**为什么有说服力。** 每一步的增量都可归因。保留负步骤更进一步：它证明作者不是挑着报，而且解释了为什么必须成套出现。

**常被做坏的方式。** 关键组件根本没消融。HAT 的 arbitrary attention 与异构层、Lite Transformer 的分支比例和核、NetAug 的 r/s/α、OFA progressive shrinking 的各子组件、HAQ 的全部组件、EfficientViT 的分支数和聚合核大小都没有单独消融。另一种做坏：消融只有训练 loss 或单点（SANA-Video 的 RoPE 和时间卷积消融只有 loss 曲线；DC-Gen Tab. 6/7 只在 512px FLUX 单点做）。

**我们怎么用。** 分解表是合同的一部分：先写好要加哪几步，每步的预期增量，再跑。跑完后负的步骤照报（准则 6.5 不调参救）。

#### 3.5 证明"我的修复是必要的"时，同时比较 2–3 个通用替代技巧

**做法。** 不只比朴素基线，还比几个"看起来也能解决这个问题"的常规做法。

**实例。**
- DC-Gen Tab. 6：与分层学习率、渐进 LoRA 比；Tab. 7：与在 AE 潜空间对齐（3 种权重）比。
- NetAug Sec. 5：讨论了更小的 weight decay、更弱的数据增强等"也可能缓解小模型欠拟合"的简单技巧。
- DiffAugment Table 1：两种朴素增强方式定量失败。
- DistriFusion Fig. 9：四种 GroupNorm 方案对照。
- GAN Compression Table 7 / Fig. 12：输出蒸馏对中间特征蒸馏。

**为什么有说服力。** 审稿人最常见的反驳是"用 X 不就行了"。提前把 X 做掉。

**我们怎么用。** 对抗式检索（准则 4.6）的产出之一就是这张"替代技巧清单"，然后把清单上的每一项做成一个对照臂。

---

### 情境 4：要声称"更快、更省"

#### 4.1 在真实硬件上测端到端，并与理论上界对账

**做法。** 先从资源账本算出理论上限（情境 1.2），再实测端到端，两者并列。差距要解释。

**实例。**
- Jet-Nemotron Sec. 3.3：理论上界 56×（14×4，来自 KV cache 组数），实测 53.6×。
- COAT：Table 2 的算子级账本算出激活压缩理论 1.69×，Table 7 实测 1.65×。
- AWQ Fig. 3b：W4 的理论算术强度提升 4×，TinyChat 实测 3.2–3.3×。
- LServe Fig. 12：kernel 与 oracle 1−sparsity 并列，理论加速 1/(1−r)。
- Radial Attention Table 1：PFLOPs 节省 1.8× 对应实测 1.9×。
- Quest Sec. 3.5 / Fig. 8a：解析式 1/PageSize + K/PageNum，kernel 收敛到 1/PageSize。
- HART Tab. 3：实测 5.6× 与 MACs 5.8× 吻合。
- TorchSparse Sec. 4.3.1：FP16 理论 2× 对实测。
- EIE Table IV：理论时间对实际时间；Sec. VI-B 节能拆成四个因子的乘积。
- SpArch Sec. II-C/D、III-C：解析 DRAM 访问模型逐步对照实测。
- MCUNetV2 §4、MCUNetV3 §3.1：解析内存与真机实测并列（"双轨报告"）。
- QServe Fig. 3 → Table 1：roofline 预测 W4A4 应更快，实测朴素 KV4 在 A100 上反而慢（0.87–0.90×），用实测推翻朴素预期，再修复到 1.29–1.51×。

**为什么有说服力。** 实测接近上界，说明实现没有留下大的工程损耗；实测远低于上界而作者能解释差距，说明作者知道自己的系统在做什么。两种情况都比一个孤立的倍数可信。

**常被做坏的方式。** 第 2.6 节的"效率只用代理"。另一种是硬件论文里仿真器与 RTL "对拍"只有一句话，没有给对拍误差（G8 组全部如此）。

**我们怎么用。** 效率类主张的合同里写两个数：理论上限、实测。实测低于上限 20% 以上要在论文里解释。

#### 4.2 先承认朴素实现会慢，再逐项展示融合带来的收益

**做法。** 方法引入的额外分支或步骤，先报它在朴素实现下的开销，再报 kernel 融合、重排之后的开销。

**实例。**
- SVDQuant Fig. 6a：低秩分支朴素实现开销 57%，融合后 5–10%；App. Table 4：250ms → 218ms，对朴素 INT4 的 212ms。
- QServe Table 1：朴素 KV4 在 A100 上 0.87–0.90×，优化后 1.29–1.51×；Sec. 6.4 的 KV4 kernel 逐项从 0.48ms 到 0.28ms。
- AWQ Sec. 5.4：HF FP16 52 tok/s → 自家 FP16 融合 62 tok/s → 再叠 3.1×。把"更好的 FP16 实现"带来的收益单列，不记在量化名下。
- SIGE Sec. 3.2：gather/scatter 占 41%，引出融合。
- Quant-VideoGen Sec. 4.3：k-means 质心缓存把聚类开销降 3×。
- TorchSparse++ Table 3 对 Table 4：按 kernel 计时排序版更快，按端到端计时不排序版反而快 1.2×，因为排序本身有 mapping 开销。把"一阶代理指标失效"写成正面贡献。

**为什么有说服力。** 读者本来就怀疑"你多加了一个分支，怎么可能不慢"。先承认、再解决，比直接报最终数字可信。

**我们怎么用。** 任何额外计算都先报朴素开销。

#### 4.3 两个方向都比：同预算比精度，同精度比效率

**做法。** 固定预算（KV 大小、FLOPs、稀疏率、延迟）比精度；再固定精度（"无损"的定义写清）比所需预算或延迟。两个方向都赢，才排除"用算力换精度"或"用精度换算力"。

**实例。**
- 同预算比精度：DuoAttention Fig. 7 / Table 3–4（同 KV 预算）；Quest Fig. 7（token 预算扫描）；XAttention Table 6–9（每个消融同时报精度和密度，Table 8 把 TopK/Ratio 调到相近计算量）；Radial Table 1（同 PFLOPs）、Table 2（同稀疏率）；LongLoRA Table 6；HAQ Table 3–6（延迟对齐到 0.01ms 级再比精度）；Lite Transformer Table 1/2（三个计算档）；EfficientViT Table 3/4（"相近 mIoU 的模型分组"）；SparseRefine Table 3(b)（PointRend 输入调到 704×1408 以对齐延迟）；DeltaQuant Fig. 9b（同延迟下对加大 rank 的 SVDQuant）。
- 同精度比效率：Quest Fig. 11（先定义"LongBench 无损"，再比所需预算与延迟）；Jet-Nemotron（每张精度表都带吞吐列）；MCUNetV3 Fig. 10（"稀疏更新选与全更新同精度的配置"）；MCUNet Table 3（5FPS/10FPS 约束下比精度）、Fig. 8（同约 70% 精度比 SRAM/Flash）；HAT Table 3（缩放到同 BLEU 比延迟）；SVDQuant App. Fig. 19（4-bit SVDQuant ≈ 7-bit naive）。
- 两者的综合是 Pareto 曲线：GAN Compression Fig. 7/11/12、Anycost-GAN Fig. 5、CAN Fig. 1/7、MCUNet Fig. 9、LitePose Fig. 9（MACs 加三个平台延迟的四联图）、TSM Fig. 5（FLOPs × 精度 × 参数面积）。

**为什么有说服力。** 单点比较总能被解释成"换了一个权衡点"。两个方向都赢，就不是换权衡点，是前沿整体移动。

**常被做坏的方式。** 跨精度比较：QServe 用 A8 对比 Atom/QuaRot 的 A4；SVDQuant 的 W4A4 对 ViDiT-Q/MixDQ 的 W4A8；QVG-Pro 用 4.97× 压缩对比基线的 6.40×。

**我们怎么用。** 效率类合同写两条判据：同预算下精度不低于基线的 null 带；同精度下效率高于基线。

#### 4.4 沿规模扫描，标出优势消失的区间

**做法。** 加速倍数随分辨率、序列长度、帧数、batch、编辑比例扫描，画出曲线，明确写出盈亏点或优势消失的区间。

**实例。**
- SIGE Fig. 10：编辑比例扫描，3090 上约 70%、i9 上约 45% 仍有加速，超过就没有。
- LServe Fig. 10/11 / Table 7：在 64K 时只有 1.09×。
- LPD Fig. 14：batch 扫到 OOM，标出内存受限到计算受限的转折，大 batch 时只剩约 3×。
- SANA Tab. 14：512→4K，44×→104×；DC-Gen Fig. 22：FLUX 1K 4.2×、2K 16.9×、4K 53.8×；DC-VideoGen Tab. 7：6.2×→14.8×。
- DistriFusion Fig. 6：分辨率 1024→3840，局限节承认低分辨率下利用率差。
- LongVILA Fig. 9 / Table 7/8：序列长度和 GPU 数扫到 OOM。
- DuoAttention Fig. 9–11：按上下文长度、压缩比、prefill chunk 三条轴分别扫，并展示收益随压缩比线性变化。
- DC-AE Fig. 9：小模型上 f64 反而更差，给出转折点；DC-AE 1.5 Tab. 6：方法用在没有问题的 c32 上无效。
- Atomique Fig. 18f：主动标出 FAA 反超的区间；Q-Pilot Table 2：深度不超过最优的 4×。
- QuantumNAS：承认在 Vowel-4 ZZ+RY、MNIST-4 ZX+XX 上输。

**为什么有说服力。** 主动划定适用边界，审稿人就没有"在 X 设置下你会输"的攻击点，因为作者先说了。

**常被做坏的方式。** 扫描做了，但标题数字取最有利的那一端（第 2.3 节）。

**我们怎么用。** 扫描是合同里的固定项。标题数字取扫描的中位设置，极端设置的数字放在扫描图里。

#### 4.5 测量协议写死

**做法。** 预热次数、重复次数、去掉什么、batch 多少、哪个框架、哪个精度、是否包含数据传输，全部写在正文或附录。

**实例。**
- DistriFusion Sec. 5.1：3 次预热 + 10 次测量，去最快最慢；CUDAGraph 同时用于基线。
- SIGE Sec. 4.2：GPU 200 预热 + 200 测量；CPU 10+10 重复 5 次。
- GAN Compression Sec. 4.3.5：100+100，batch=1。
- LongVILA Sec. 5.1.1：10 预热 + 5 次。
- TSM Sec 3.3：1000 次 + 200 预热；Sec 5.5：功率计减静态功耗。
- HAT Sec 3.2：固定 30/23 token，300 次，去头尾各 10%。
- SpAtten Sec. V-A：1000 次重复，去掉首尾各 15% 再平均；功耗为总功耗减空闲功耗。
- FlatFormer Sec. 5.2.1：1000 样本 × 50 次运行 + 10 次预热，剔除离群。
- EfficientViT Sec 3.1：框架、精度、batch，含数据传输时间。
- Deep Compression App A：4096 样本平均，排除 cudaMalloc/Memcpy，功率工具与折算假设。
- AWQ Sec 5.4：bs=1，prompt 4 token，生成 200 token，取中位数，沿用 exllama 协议。

**未交代的：** Anycost-GAN、CAN、LongLive、FastComposer、TorchSparse 与 TorchSparse++（都没写重复次数和预热）。

**为什么有说服力。** 协议写死了，读者才能判断倍数有没有被测量噪声吃掉。

**我们怎么用。** 协议是合同的附录，和判据一起冻结。

#### 4.6 用工业或社区最强实现做效率基线

**做法。** 效率对比的分母用生产级后端，而不是 HuggingFace 默认实现。

**实例。**
- QServe：TensorRT-LLM 三种精度配置取最快的作分母（Table 4 注）。
- SmoothQuant：FasterTransformer（Fig. 9）。
- COAT：TransformerEngine（Table 7/8）。
- SVDQuant：社区 NF4 FLUX + TensorRT W8A8。
- DeltaQuant：社区 GGUF Q4。
- AWQ Fig. 10：llama.cpp、exllama、AutoGPTQ。
- LongVILA Sec. 5.1.1：Megatron 按其团队推荐的 CP+TP 混合配置。
- TorchSparse++：最新的 SpConv 2.3.5。
- EfficientViT Table 2–4：所有基线在同一 TensorRT fp16 管线下测延迟。

**常被做坏的方式。** AWQ 的系统主对比用 HF FP16（弱基线），但它同时给了自家 FP16 优化后的数字，把工程优化和量化拆开，属于补救得当。

**我们怎么用。** 基线的实现强度要写进合同："基线用 X 版本的 Y 实现"。

#### 4.7 "装得下"作为头条

**做法。** 把效率落到一个二值事实：更小的硬件能跑、单卡能训、单节点能装。

**实例。**
- AWQ：Llama-2-70B 上 Jetson Orin，13B 上 8GB 笔记本 4070。
- SmoothQuant：530B 单节点，175B 用一半 GPU（Table 10, Fig. 9）。
- QServe Table 4：L40S 上超过 A100 上的 TensorRT-LLM。
- SVDQuant：16GB 笔记本 4090 免 offload。
- DeltaQuant：Wan2.2 装进 32GB 5090。
- QeRL Table 8：32B RL 单张 H100。
- COAT Table 8：7B 单卡全参训练。
- DuoAttention + QServe Fig. 12：单 A100 上 3.3M token。
- MCUNet：首次在商用 MCU 上 ImageNet >70%。
- TinyML 组的做法：把 OOM 直接写进表格（MCUNet Fig. 4、MCUNetV2 Table 2 划线、PockEngine Table 4 的 "-"）。
- Lightning-OPD Table 3：标准 OPD 在单节点 8×H100 上训 30B MoE 会 OOM；VLASH Table III：乒乓任务所有基线 0/20，VLASH 11/20。

**为什么有说服力。** 二值事实不需要统计；"能/不能"比"快 1.3×"更难被噪声解释。

**常被做坏的方式。** 可行性边界由作者选定的资源约束构造（单节点、固定 q），对应格子就缺了基线对照（Lightning-OPD、TLT 的 OOM 格）。

**我们怎么用。** 如果我们的结果里有二值事实（某个攻击在某条件下完全失效、某个防御让形成从未发生），把它放头条，而不是放一个 1.3× 的连续量。

---

### 情境 5：要证明必须针对硬件或设备定制

#### 5.1 "模型 × 硬件"交叉矩阵，对角线最优

**做法。** 为每种硬件各搜一个模型，然后把每个模型在每种硬件上都测一遍，做成矩阵。对角线（为该硬件搜的模型在该硬件上）最优，就证明了定制是必要的。

**实例。**
- HAQ Table 1：3 种量化策略 × 3 个硬件（BitFusion / BISMO edge / BISMO cloud）的延迟。
- ProxylessNAS Table 4：GPU/CPU/mobile 三个模型 × 三个硬件；GPU 模型在 CPU 上 204.9ms，CPU 模型 138.7ms。
- HAT Table 1：2×2，GPU 模型和 ARM 模型各自在 GPU/ARM 上测。
- QuantumNAS Table III：在设备 A 上搜索的电路在设备 B 上运行，对角线占优。
- QuantumNAT Table 6：同样的交叉部署矩阵。
- OFA Fig. 14 是定性版本：FPGA/CPU/GPU 搜出的架构不同。
- AWQ Fig. 8(b)：校准集 × 评测集的 2×2 交叉（PubMed/Enron），报告对角与非对角的差，用来证明方法不依赖校准分布。

**为什么有说服力。** 用极少的数字把"专门化"从口号变成可验证的事实，矩阵结构一眼可读。

**我们怎么用。** 凡是声称"针对 X 定制"的方法，交叉矩阵是标配。对我们来说，X 可能是模型族、数据域或攻击类型。

#### 5.2 把搜出的结构画出来，用硬件原理解释

**做法。** 把 NAS 或压缩策略学到的逐层决策画出来，再用 roofline、算术强度、访存模式解释它为什么合理。

**实例。**
- HAQ Fig. 3–5：depthwise 卷积是访存受限的，所以边缘设备上给激活更少位。
- AMC Fig. 2/3：锯齿状瓶颈，3×3 剪得多、1×1 剪得少。
- ProxylessNAS Fig. 6：GPU 偏好浅而宽，CPU 偏好深而窄。
- HAT Fig. 12：Raspberry Pi 深窄、GPU 浅宽，回扣 Fig. 2 的 profiling。
- OFA Fig. 12–14：FPGA 只选 3×3 核，算术强度 +48%/43%。
- MCUNet Fig. 12：SRAM 大→分辨率高，Flash 大→宽度大。
- MCUNetV2 Fig. 9：patch 段用小核、大扩张与大核不同时出现。
- MCUNetV3 Fig. 11(b)、PockEngine Fig. 6：块内第一个 1×1 卷积最值得更新。
- Pruning Fig. 4：MNIST 的稀疏模式对应图像中心。

**为什么有说服力。** 黑箱搜索的结果变成了可复用的设计原则，这是"insight"分的来源。而且它闭合了一个环：开场的 profiling 图预言了什么结构好，搜索结果验证了预言。

**我们怎么用。** 任何自动化方法的输出都要能翻译成一条人能理解的规则，否则就只是一个黑箱数字。

---

### 情境 6：要拆开系统贡献和算法贡献

协同设计类论文最常被攻击的点是"你的加速到底来自算法还是来自工程"。Han Lab 用两种实验把它们拆开。

#### 6.1 因子网格

**做法。** 把系统维度和算法维度各取 2–3 个水平，做成网格，每格都测。

**实例。**
- MCUNet Table 2：推理库 {CMSIS-NN, TinyEngine} × 模型 {S-MbV2, S-Proxyless, TinyNAS} 的 2×3 网格，同一 320kB/1MB 约束下 ImageNet top-1 为 35.2/49.5/55.5 → 47.4/56.4/61.8。单看任一行或任一列都有提升，组合最好。Fig. 13 再补一刀：同一进化搜索在弱库上天花板更低，证明"推理库影响可达精度"。
- MCUNetV2 §4.2：同一搜索空间与超网，只切换 per-layer / per-patch 推理。
- SPVNAS Fig. 7 + 附录 Table 6/7：同宽度旧模块对新模块；新模块手工设计对搜索，在四个成本档上重复。
- MCUNetV3 Fig. 10(a)：稀疏更新对 +算子重排。
- Fast-dLLM Table 1/2：base / +Cache / +Parallel / +Both。
- QServe Fig. 17：同 batch 对比，把"更快的 kernel"与"更大的 batch"两个因子拆开。
- COAT：最大 batch 的收益与 kernel 的收益分开报。

**为什么有说服力。** 网格的每个格子都是单变量对照（情境 3.1），而且还能看出交互项：组合的收益是否大于两者之和。

**常被做坏的方式。** 只报端到端。MCUNet Fig. 9 的 2.4–3.4× 加速里基线跑在 TF-Lite Micro 上、MCUNet 跑在 TinyEngine 上，混合了库与模型两种贡献；LongVILA 的头条加速主要来自采纳的 2D-Attention，自研的多模态分片只贡献 1–7%（Table 5）；PointAcc 对 Mesorasi 的 100× 混入了模型变化（Mini-MinkowskiUNet）；NAAS 的 +2.7% 精度来自整合的 OFA。

**我们怎么用。** 凡是贡献有两个维度，主表就是网格，不是一列。

#### 6.2 把自家算法也放到通用硬件上跑

**做法。** 在 CPU/GPU 上实现同样的算法，和专用硬件上的结果并列。差异就是硬件的贡献；通用硬件上的提升就是算法的贡献。

**实例。**
- EIE Fig. 6/7：CPU/GPU 的 compressed 列，"压缩单靠 CPU/GPU 只有 3×"（Sec. VI-A）。
- SpAtten Sec. V-B：在 GPU 上实现 token 剪枝（topk+gather），BERT batch 模式最多 2.3×，远低于专用硬件。
- PointAcc Fig. 17：mergesort 求交在 CPU/GPU 上更慢，在 PointAcc 上快 10 倍以上；fetch-on-demand 在 GPU 上吃亏。这是反事实对照，不只是附带对比。
- TorchSparse Sec. 5.2：在没有 FP16 tensor core 的 1080Ti 上复测仍有 1.5×，排除"收益来自硬件特性"。
- TorchSparse++ Fig. 14：Pascal/Turing/Ampere + Orin × FP16/TF32/FP32。

**为什么有说服力。** 同一算法在两类硬件上的差异，是"硬件必要"的直接证据；反过来，算法在通用硬件上的提升则证明算法本身有价值。

**我们怎么用。** 我们的"算法"是检测或防御规则，"硬件"对应的可能是训练管线里的接入点。拆开的方式同理：同一规则在不同接入点上的效果并列。

---

### 情境 7：用搜索或代理做自动化

#### 7.1 搜索用代理，报告用实测，并单独给出代理的校准

**做法。** 搜索阶段用查表或预测器（延迟、精度），报告阶段全部用真机实测，并单独画一张"代理 vs 实测"的校准图。

**实例。**
- ProxylessNAS Fig. 5 + App B：延迟查表预测 vs 实测，RMSE 0.75ms，"搜索用模型、报告用实测"。
- HAT Fig. 6 + Sec 2.3：延迟预测器 RMSE 0.1s，"只在搜索中用预测值"；Table 5 + Fig. 11：继承权重的子网精度 vs 从头训练。
- OFA App A + Fig. 15：精度预测器 RMSE 0.21%，并展示 RMSE 越低选出的子网越好。
- AMC Table 2：不微调的精度与微调后精度的排序一致。
- APQ Fig. 7：pairwise accuracy。
- QuantumNAS Fig. 9（继承参数 vs 从头训练，Spearman 0.75）+ Fig. 10（模拟 vs 真机 0.76）+ Table IV。
- QuantumNAT Table 11：噪声模型 vs 真机差距 <5%。
- QuEst Fig. 2：PST 与保真度的 Spearman 0.993（但只在模拟器上验证）。
- MCUNet Fig. 10：搜索空间的均值 FLOPs 与最终精度正相关（只给图，无相关系数）。

**未给校准的：** APQ 用查表估延迟但没有校准数据。

**为什么有说服力。** 搜索类论文最容易被攻击的点就是代理不可靠。校准图提前堵住这一条。

**常被做坏的方式。** 代理与真实目标有偏差，而且结论随代理翻转。Q-Pilot 只报电路深度，不计移动时间，同组 Atomique Fig. 19 显示它深度更低但保真度更差。L2DC 用竞赛目标评分时 RL 略输人类（−0.03 vs 0.00）；GCN-RL 换成等权归一化 FoM 后同类方法变成"胜过专家"（2.69 vs 2.32），而且带宽只有人类设计的约 1/6。

**我们怎么用。** 我们用的任何代理（loss 探针、梯度相干性、AUROC 代替最终的漏网数）都要有一张校准图，以及一个"代理失效时结论会不会翻转"的检查。

#### 7.2 搜索成本单独记账，拆成固定 + 边际 × N

**做法。** 把 NAS 或压缩策略搜索的 GPU 小时、CO2、美元列成表；多场景部署时拆成一次性成本和每场景的边际成本。

**实例。**
- OFA Table 1 + Fig. 8：N=40 时 CO2 少 16×–1300×。
- APQ Table 2 + Fig. 2：2400 + 0.5N GPU 小时。
- HAT Table 2 + Fig. 10：比 Evolved Transformer 少 12,041×。
- Lite Transformer Table 3 + Fig. 1b。
- GAN Compression Fig. 8：Fast 版训练省 1.6–3.7×、搜索省 3.5–12×。
- Lightning-OPD Table 2 下半：rollout / 教师预计算 / 训练的 GPU 小时分阶段。
- Lightning-Weave Table 6：rollout / 参考打分 / 锚打分 / 学生训练。
- DC-Gen Tab. 15 / DC-VideoGen Fig. 2：后训练成本对从头训练（"230×"）。

**常被做坏的方式。** 只强调边际成本，把一次性的 2400/1200 GPU 小时放另一列；单场景时总成本反而更高（APQ、OFA）。四篇的外部估算都依赖 Strubell 的数字。Lightning-OPD 的 4× 不含 SFT 和重生成 SFT 数据的成本。

**我们怎么用。** 成本表写总成本，边际成本作为补充。

#### 7.3 外部可核查的人类基线

**做法。** 把人类专家的设计作为基线时，用竞赛获奖设计或课程项目，注明经验和工时，让读者能查。

**实例。**
- Learning-to-Design-Circuits Table 1/2、GCN-RL Table I：同一批 Stanford EE214 课程设计。
- AMC 的"人类专家"基线来自 Han 的博士论文（可查但独立性不足）。
- MCUNet Fig. 9：VWW'19 冠军方案按同约束缩放、同设置重训。
- 不同代价的优化器按同墙钟时间对齐：L2DC Table 1 脚注、GCN-RL Sec IV-A（贝叶斯优化因 O(N³) 只能跑较少样本）。

**我们怎么用。** 人类基线的来源要独立于作者。

---

### 情境 8：没有真实环境可测

#### 8.1 自建物理分项模型，做敏感性扫描，主动标出方法输掉的区间

**做法。** 目标硬件不存在时，从已发表的物理参数推出一个分项模型；对每个参数做扫描；给出解析的盈亏平衡点；明确标出哪个区间自己输。

**实例。**
- Atomique（ISCA 2024）Sec IV、Fig. 18：从单篇 Nature 实验推导参数，六个硬件参数的敏感性扫描，解析 breakeven（n_vib<18 时移动优于 SWAP），Fig. 18f 标出 FAA 反超的区间。Fig. 22 逐条放松硬件约束看收益。
- Q-Pilot Eq. 5：同类模型，但只报深度。
- DGA：V100 服务器上用 Netem 合成延迟 + 树莓派真实 Wi-Fi（Fig. 4a/b），一边合成一边真实。
- 硬件组的评估栈（EIE Sec. V、SpAtten Sec. V-A、SpArch Sec. III-A、PointAcc Sec. 5.1、LEGO Sec. VI-A）：周期仿真器 + 部分模块 RTL 综合 + CACTI/Ramulator + 文献单位能耗。

**与精确求解器比较：** 小规模比质量、大规模比可扩展性、设超时。Atomique Fig. 14 + Table II（24 小时超时）；Q-Pilot Table 2（4000 秒超时）。

**三层基线：** 跨平台 / 同平台跨架构 / 同架构跨编译器。Atomique Fig. 13；Q-Pilot Fig. 11–13。

**为什么有说服力。** 没有实机，可信度只能来自"模型的每个假设都可查、每个参数都扫过、输的区间自己说"。

**常被做坏的方式。** 所有平台相干时间 ×10、门误差 ÷10 这类统一乐观假设（Atomique Sec V-A）；可扩展性靠外推（QuantumNAS 127 qubit 运行时线性外推 57 小时；QOC 的"量子优势"落在双方都外推的区间）。

**我们怎么用。** 我们的 toy 实验就是这种情况。规则：toy 的每个假设列表、每个参数扫描、以及"toy 上成立但真实上可能不成立"的区间，都要写。

---

### 情境 9：任务没有唯一正确答案（生成任务）

#### 9.1 质量报两种口径：相对真值 + 相对原模型

**做法。** 压缩或加速一个生成模型时，除了报分布质量（FID、CLIP），还报与原模型输出的偏离（PSNR/LPIPS w/ Orig.、LPIPS Diff、属性匹配率）。后者对近似误差更敏感。

**实例。**
- DistriFusion Table 1：w/ G.T. 与 w/ Orig. 两列。
- SIGE Table 1、附录 Table 5：同上，另加编辑区单独评测。w/ Orig. 更能区分：DDPM PSNR w/ Orig. 下 Ours 52.4 对 Patch 40.6 对 Pruning 31.0。
- Anycost-GAN Fig. 5b（LPIPS Diff：子网 vs 全网）、Table 4（属性匹配率）。
- SVDQuant、DeltaQuant、Quant-VideoGen、FourTune：同时报相对 16-bit 的 LPIPS/PSNR 和绝对质量 FID/ImageReward/VBench。

**为什么有说服力。** 生成任务里"好"没有唯一标准，但"像不像原模型"有。两个口径分开，读者能判断方法是"做了近似"还是"换了一个模型"。

**常被做坏的方式。** 只报绝对质量，用 FID 持平掩盖与原模型的偏离。SIGE 在 SD 上的 PSNR w/ Orig. 只有 19.9（DDPM 是 52.4），正文用 LPIPS/FID 持平来叙述。

#### 9.2 预先承认"代理指标好不等于目标好"，给出反例

**做法。** 作者自己先说"重建好不等于生成好""rFID 低不等于 gFID 低"，并给出一个反例，然后再用代理指标。

**实例。**
- SANA Fig. 3a：F8C16 重建最好但生成最差。
- DC-AE 1.5 Fig. 1b：通道越多 rFID 越好、gFID 越差，两条反向单调曲线。
- HART §3.1 + Fig. 7：分离解码器 rFID 相近但 gFID 21.98 对 5.30。
- DC-AE Fig. 9：f64 在小模型上生成变差的转折点。
- SANA-Video Tab. 7：用"加噪重建"做下游适配性的代理，并说明理由。
- TorchSparse++ Table 3 对 Table 4、FlatFormer "MACs 多 1.7× 却更快"：把"一阶代理指标不可靠"本身写成贡献。

**为什么有说服力。** 把审稿人最可能的攻击变成了自己的论点。

**我们怎么用。** 我们的研究里"AUROC 高不等于漏网少""形成推迟不等于形成被阻止"是同类的代理失效。论文里要先自己说，再给反例。

#### 9.3 人类评测带 "Equal" 选项，分领域

**实例。** DC-Gen Tab. 3/9/10/11：AMT 成对评测，含 Equal 选项，T2I/视频/编辑各一组，并给界面截图。GAN Compression Table 3：AMT，72.4% 偏好。LongLive Fig. 1 右 + 附录 L：26 人 × 48 题。

**常被做坏的方式。** 人评无置信区间、无显著性检验（DC-Gen 也没有）；用模型当评委且评委与数据同源（StreamingVLM）。

---

### 情境 10：要说明能泛化

#### 10.1 广度靠笛卡尔覆盖：模型族 × 尺寸 × 任务 × 硬件

**实例。**
- AWQ：OPT 1.3B–30B、LLaMA 7B–65B、Llama-2 7B–70B、Mistral、Mixtral、Vicuna、CodeLlama、OpenFlamingo-9B、LLaVA-13B、VILA-7B/13B；硬件 4090、Orin、4070 笔记本、A100、Raspberry Pi 4。
- StreamingLLM Fig. 5：Llama-2 7/13/70B、Falcon 7/40B、Pythia 2.8/6.9/12B、MPT 7/30B，覆盖 RoPE 与 ALiBi 两种位置编码。
- OFA：12 种硬件，每种一条精度–延迟曲线。
- TSM：7 种硬件（P100、TX2、Nano、Xeon、Raspberry Pi、Note8、Pixel 1）。
- PockEngine：7 个平台，MCU 到 Jetson Orin 到 Snapdragon DSP。
- SIGE：5 种硬件 + TensorRT 后端。
- DGC：图像/语言模型/语音三种模态，5 个数据集。
- EfficientViT：分类/分割/超分/SAM 四类任务。

**为什么有说服力。** 用"族 × 尺寸 × 位置编码类型"的笛卡尔覆盖，而不是堆 benchmark 数量。每一维都是一个可能的反驳（"只在 RoPE 上成立"），覆盖掉它。

**常被做坏的方式。** 广度窄：Quest 只有两个 7B（同为 Llama-2 系）；XAttention 每个模态一个模型；Long-RL 的消融只在 1.5B 上做。或者广度有了但只与 FP16 比（第 2.7 节）。

#### 10.2 "慢基准证明不掉点，快基准证明有增益"的分工

**实例。** VLASH 用 LIBERO 只证"不损害"，用 Kinetix 和真机动态任务证增益；ForeAct 用 LIBERO 证兼容，用自建真机任务证增益。

**我们怎么用。** 标准基准和自建基准各有分工，自建基准只能用来证增益，不能用来证"不掉点"。

#### 10.3 新能力不伤旧能力：报守恒指标

**实例。** VILA Table 3/4/6：加视觉能力的同时报 MMLU/BBH/DROP 纯文本能力；LongVILA 附录 Table 9：报图像基准；StreamingLLM Table 4：预训练加 sink token 后 7 个零样本任务不掉。

**我们怎么用。** 防御方法必须报学习能力的守恒指标，这是我们已经在做的。

#### 10.4 训练–推理一致性的对照

**实例。** LongLoRA Table 6：稀疏训练、全注意力测试的双协议；StreamingVLM Table 2：ReKV 套在微调模型上胜率归零，证明训练与推理必须对齐；Radial Sec. 4.3：静态掩码可训练，LoRA+Radial 不低于全参。

#### 10.5 与正交技术叠加

**实例。** SIGE + GAN Compression 达 40–50× MACs；DiffAugment + CR-BigGAN/TransferGAN/FreezeD；Deep Compression 剪枝+量化互不干扰（Fig. 7）；DuoAttention + QServe；Quest + AWQ；DC-Gen Tab. 17 与 SVDQuant/TaylorSeer/AnyFlow 组合；QuantumNAT Table 7 叠加 Qiskit O3。

**为什么有说服力。** 证明贡献是前沿的一个独立维度，而不是和已有方法互斥的替代品。

---

### 情境 11（补充）：写作层面的小手法

这些不是实验设计，但在 94 篇里反复出现，属于"让证据更易读"的做法。

- **定性图逐格标数字。** DistriFusion Fig. 1/5/8/9、SIGE Fig. 1/8/9、GAN Compression Fig. 5/6、DiffAugment Fig. 12–14：每个面板都写 MACs/延迟/FID，定性图也承载定量信息。
- **"四方案并排"的总图。** StreamingLLM Fig. 1：dense 5641 / window 5158 / recompute 5.43 / ours 5.40，一张图同时立住问题和解法；StreamingVLM Fig. 1 照搬同一版式。
- **把方法的分解本身可视化。** HART Fig. 3（离散 token + 残差 = 最终图）、DC-AE 1.5 Fig. 3（前 16/32/64 通道逐级重建）、LPD Fig. 8（四种顺序逐步填充）。
- **把局限写成定量表格放附录。** StreamingLLM 附录 C Table 7：query-answer 距离超过 cache 时准确率归零；附录 D：LongBench 上 4+3496 低于截断基线，改成 1750+1750 才追平。主动划定边界，降低攻击面。
- **主动拒绝弱基准并说明理由。** MCUNet §4.1 拒用 CIFAR（"32×32 不代表真实场景"）。
- **LLM 评委双向顺序评测。** AWQ Fig. 5：GPT-4 评判两种顺序共 160 次，抵消位置偏差。
- **改造基线时公开改造方式并论证对基线有利。** DuoAttention 附录 A.3：H2O/TOVA 改用 FlashAttention 精确 prefill + 最后 50 token 模拟解码，明说这让基线更强。
- **外部背书。** OFA（LPCVC 第 3、4 届冠军）、TSM（Something-Something 排行榜第一）、SpAtten（DAC 2023 University Demo 第一）、AWQ（MLSys 2024 Best Paper）。

---

## 4A. 理论与实验的关系：理论跟在观察后面，预言走在实验前面

这一节回答一个单独的问题：Han Lab 是先把理论推完、再往空隙里填实验吗？

**答案：不是。94 篇里没有一篇是这个顺序。** 实际顺序几乎都是：测量 → 观察 → 方法 → 一段短推导解释观察或给出上界 → 实验验证推导的预言。理论是被观察逼出来的，不是方法的发生器。档位：实例【读到】，分类和比例【归纳】（从笔记里数的，不精确）。

### 4A.1 Han Lab 的"理论"是四种东西，强弱差很多

**(a) 资源账本、roofline、代价模型。最常见，也最硬。** 理论在这里的作用是给出一个数字上界，实验看实现离上界多远。
- COAT Table 2：算子级激活账本算出理论压缩 1.69×，Table 7 实测 1.65×。
- Jet-Nemotron Sec. 3.3：理论上界 56×（14×4，来自 KV 组数），实测 53.6×。
- SpArch Sec. II-C/D：解析的 DRAM 访问模型，每加一项技术就对一次实测。
- Quest Sec. 3.5：解析式 1/PageSize + K/PageNum，Fig. 8a 看 kernel 是否收敛到 1/PageSize。
- AWQ Fig. 3：roofline 算出生成阶段算术强度约 1，由此决定做 weight-only 而不是 W8A8。
- EIE Table IV：理论时间对实际时间；Sec. VI-B 把节能拆成四个因子的乘积。
- 比例：约 15–20 篇有这类账本，几乎全部在系统和硬件类。

**(b) 把观察变成机制的短推导。几行公式，跟在一张图后面。** 顺序是图在前、公式在后。
- SpAtten：Fig. 7 先画出"注意力最大概率越大、4-bit 量化误差越小"的散点，然后 Eq. 2 用 softmax 导数推出误差上界与 2p(1−p) 成比例。
- AWQ：Table 1 先观察显著通道存在，Eq. 2–3 推出放缩 s 后误差约为 Δ'/Δ·1/s，再用 Table 2 单独测推导里的近似（s<2 时 Δ 变化 <5%），并用推导预言的拐点解释超参（s=2 最优 11.92，s=4 回升到 12.36）。
- MCUNetV3 Fig. 3：先观察范数比，推出 s⁻² 偏差，再用 QAS 修复。
- QuantumNAT 定理 3.1：噪声 = 线性缩放 + 平移，推出测量后归一化；也是先有 Fig. 1/4 的分布偏移图。
- Radial Attention：Fig. 4 先画经验衰减曲线，Fig. 8(b) 拟合成指数（R²>0.985），附录 A 再从这个模型推 O(n log n) 和误差界。这是 94 篇里最接近理论先行的一篇，但起点仍是经验曲线。
- DGC：Eq. 4 对 Eq. 6，用公式对比指出朴素稀疏化丢了动量折扣项。推导在这里是诊断工具，解释朴素方法为什么掉点。

**(c) 等价性或零开销证明。理论说"这个变换什么都没改"，实验确认。**
- SmoothQuant：放缩因子折进 LayerNorm 或前一层；AWQ：s⁻¹ 折进前一算子；QServe：旋转矩阵、平滑因子折进权重，并为 RoPE 加约束 λ_i=λ_{i+D/2}；QeRL App. G：噪声并入 RMSNorm。
- SANA-Sprint Prop. 3.1 + Tab. 1：先推导 Flow→TrigFlow 变换无损，再实测确认。
- Q-Pilot：飞行辅助比特与拓扑无关的等价性证明。
- 实验很简单（前后数字一样），但它们是"可部署"主张的根基。

**(d) 给启发式背书的正式定理。最弱的一类，集中在推理模型后训练组。**
- Fast-dLLM Thm 1、Lightning-OPD Thm 3.5–3.9、Lightning-Weave App. A、DGA Thm 2.3。
- 共同问题：定理的条件和实际超参对不上，而且没有实验检验条件。Fast-dLLM 定理要求 (n+1)ε ≤ 1，factor 策略取到 1.9；OPD 的 χ² 是序列级量，实证只测逐 token；Weave 的实现用代理损失，作者自承不继承证明的等价性；DGA 要求 D=O(K)，实验 D=4K（这一项尚可接受）。
- 唯一正面的例子：Weave 用一个两动作反例证明朴素离线目标没有不动点。反例比定理有用，因为它直接解释了实验里观察到的失败。

### 4A.2 理论和实验谁填谁的空隙

**是理论去填实验的空隙，不是反过来。** 理论出现的位置，正好是实验够不到的地方：
- 没有实机：Atomique 从单篇 Nature 实验推出物理分项模型，给解析 breakeven（n_vib<18 时移动优于 SWAP）。
- 规模够不到：QuantumNAS 127 qubit 运行时线性外推 57 小时；QOC 的"量子优势"落在双方都外推的区间。
- 硬件做不了：SmoothQuant Table 1 用模拟的 per-channel 激活量化当理论上界。这里的"理论"其实是一个不可部署的实验。
- 开销不想测：零开销靠折叠证明，不靠实测。

反过来的结构（理论推完、留空隙让实验补）一篇也没有。即使 Lightning-Weave 这种定理最多的，实验也不是在验证定理，而是在验证方法。

### 4A.3 真正起作用的一招：理论产出可检验的预言，实验测预言而不是测"方法有效"

这是 Han Lab 好论文和一般论文的分界线。

| 论文 | 理论给出的预言 | 实验测的是什么 |
|---|---|---|
| AWQ Table 2 | Δ'≈Δ 的近似在 s<2 成立；s 过大会在某点反转 | 直接测 Δ 变化比例；测拐点位置 |
| SpAtten Fig. 7 | 误差 ∝ p(1−p) | 散点是否呈这个形状 |
| COAT Table 2/7 | 压缩上界 1.69× | 实测与上界的差 |
| Radial Fig. 8(b) | 衰减是指数 | 拟合 R² |
| TSM Sec 5.1 | 增益与时序依赖强度成正比 | 事前分两组，看增益排序 |
| SpArch Fig. 16 | DRAM 模型 | 每项技术的实测增量是否与模型一致 |
| Jet-Nemotron | 上界 56× | 实测 53.6× |

这些实验可以失败：拐点可以不在预言的位置，R² 可以很低，实测可以远离上界。这正是我们准则 6.2（判据可被满足也可不被满足）和 6.4（先注册预测）的要求。区别只在于：Han Lab 从不预注册，预言和结果在同一篇论文里一起出现，读者无法区分"先预言后验证"和"先看结果再写预言"。

### 4A.4 对我们的含义（【归纳】）

1. **理论的形式应该是账本，不是定理。** 账本给一个数，实验和数比。定理给一个条件，而条件通常不满足。
2. **推导放在观察之后，但预言要在实验之前冻结。** Han Lab 做到了前半句，没做后半句。我们的合同机制正好补后半句。
3. **推导里的每个近似单独测一次。** AWQ Table 2 是模板。我们的形成模型如果有"梯度相干性来自目标词"这类近似，就要像 Table 2 那样单独测它。
4. **定理用反例的形式写。** 一个两动作反例证明"朴素目标无不动点"，比一个带五个条件的定理有用。
5. **警惕装饰性定理。** 投稿前检查定理的每个条件在实验设置里是否满足；不满足就删掉定理或改成经验陈述。

---

## 5. 投稿前自查清单

把第 2 节和第 4 节压缩成可以逐条打钩的清单。每条后面是对应的章节号。

**动机**
- [ ] 有一张"X 变好了，Y 没动"的开场对照图，Y 是我们要优化的量。（1.1）
- [ ] profiling 产出了一个瓶颈占比数字，并被用来算理论上限。（1.2）
- [ ] 上一轮的失效用新协议重测，没有引用旧数字。（1.3）

**机制**
- [ ] 每条机制主张有一个干预或反事实实验，不只有可视化。（2.1）
- [ ] "选择"类方法有随机臂，随机臂跑了多次。（2.2）
- [ ] 子群预测在看结果之前写进合同。（2.3）
- [ ] 定理的前提或界有实验检验。（2.4）
- [ ] 理论给出了一个数或一个拐点，实验测的是这个预言，不只是"方法有效"。（4A.3）
- [ ] 推导里的每个近似单独测过。（4A.4）

**组件必要性**
- [ ] 主表每一行与新方法只有一个差异。（3.1）
- [ ] 涉及额外训练的方法有等算力、等数据的对照臂，且对照臂排在队首。（3.2）
- [ ] 方法有一个参数取值等于基线。（3.3）
- [ ] 分解表每步同时报精度和效率，负的步骤照报。（3.4）
- [ ] 对抗式检索产出的"替代技巧清单"每项做了对照臂。（3.5）

**效率**
- [ ] 理论上限和实测并列，差距 >20% 有解释。（4.1）
- [ ] 额外计算先报朴素开销，再报融合后开销。（4.2）
- [ ] 同预算比精度、同精度比效率两个方向都有。（4.3）
- [ ] 规模扫描做了，标题数字取中位设置。（4.4）
- [ ] 测量协议（预热、重复、去尾、batch、框架、精度）写在附录。（4.5）
- [ ] 效率基线是生产级实现，版本写明。（4.6）

**广度与泛化**
- [ ] 模型族 × 尺寸 × 任务 × 硬件的覆盖，每一维对应一个可能的反驳。（10.1）
- [ ] 守恒指标（新能力不伤旧能力）报了。（10.3）
- [ ] 自建基准只用来证增益。（10.2）

**数字纪律（不学清单的反向）**
- [ ] 核心对比 ≥3 seed，或显式声明单 seed。（2.1）
- [ ] 调参集、选择集、报告集分开。（2.2）
- [ ] 标题数字只有一个口径，旁边有最不利设定、总成本、同配置表。（2.3）
- [ ] 摘要和引言里每个数字回表格定位过。（2.4）
- [ ] 核心机制有反事实对照，没有的标【推测】写进局限。（2.5）
- [ ] 效率有真实测量，代理有校准图。（2.6）
- [ ] 有局限性章节，且是定量的。（2.7）

---

## 6. 映射到我们的研究方向（全部【推测】档）

我们做的是训练期后门的形成、检测与防御，不是效率。Han Lab 的手法搬过来，需要先找到对应物。下面每条写"他们的量 → 我们的对应物 → 为什么是对应物 → 还没验证的地方"。

| Han Lab 的量或做法 | 我们的对应物 | 为什么是对应物 | 还没验证的地方 |
|---|---|---|---|
| 真硬件端到端实测（4.1） | 在真实微调管线（真实数据混合、真实超参、真实模型尺寸）里测形成与漏网数 | toy 对应于仿真器，真实管线对应于真机；只在 toy 上成立的结论等于只在仿真器上成立 | 我们的"真实管线"定义还没冻结 |
| 理论上限与实测对账（4.1） | 从形成机制推出的预测（形成步数、漏网数上界）与实测对账 | 同样是"模型预言 → 实测验证"的闭环 | 我们还没有一个能给出定量预测的形成模型 |
| 随机臂零假设（2.2） | 干净孪生、随机行、随机目标 | 同样是"任意选择能达到的水平" | 我们已有干净孪生，但随机行的分布还没标定 |
| "X 变好了 Y 没动"开场图（1.1） | "loss/梯度范数正常，后门已形成"的对照图 | 常用指标（loss）看不见，真瓶颈（形成）已发生 | 我们已有 53–200 步 loss 失明的观察，但还没画成这种对照 |
| 单变量对照 / Tuned 对照（3.1、3.2） | 同数据、同步数、同超参的无防御臂 | 剥离"多训了"和"数据不同"的效应 | 已在做（对照臂排队首） |
| 方法退化为基线的参数点（3.3） | 防御强度参数 c=0、冻结比例 0% | 构造性公平 | LT c=0.05 已有，c=0 是否精确等于无防御需核 |
| 逐步叠加分解（3.4） | 防御组件逐个加：冻结 → +学习率 → +预热 | 每步的形成推迟量可归因 | 还没按这个结构跑过 |
| 子群分析（2.3） | 按目标分散度、按行相似度分组的形成速度 | 机制成立则增益集中在某个子群 | 分散目标的实验已显示分组效应，但不是事前注册的 |
| 规模扫描标出优势消失区间（4.4） | 防御效果随毒行数、随模型尺寸的扫描 | 主动划定边界 | 弱臂需要更长阶梯（我们已踩过坑） |
| 代理校准图（7.1） | AUROC/梯度相干性 对 最终漏网数的校准 | 代理失效时结论会翻转（Q-Pilot 的教训） | c_mem_x 的 AUROC 高但 1% 误报召回只有 0.25–0.62，正是代理与目标偏离的例子 |
| "装得下"二值头条（4.7） | "形成从未发生"而不是"形成推迟 120 步" | 二值事实不需要统计 | 目前只有推迟，没有阻止 |
| 交叉矩阵（5.1） | 在攻击类型 A 上标定的防御在攻击类型 B 上的效果 | 证明是否需要按攻击定制 | 没做过 |
| 代理指标失效本身作为贡献（9.2） | "loss 阈值在局部干净分布 5–18 百分位，任何单行阈值都救不回" | 同样是"一阶指标不可靠"的正面陈述 | 已有结论，还没按这种方式写 |
| 自建物理模型 + 敏感性扫描（8.1） | toy 的假设列表 + 参数扫描 + "toy 成立真实可能不成立"的区间 | toy 就是没有实机的情况 | 假设列表还没系统写过 |

这张表的用法：下一次写合同时，从左列找我们要做的那类主张，看右列的对应物有没有准备好。没有准备好的，先补对应物，再写判据。

---

## 7. 覆盖边界与核对状态

**读到了什么。**
- 95 篇论文中 94 篇取到正文并逐篇精读（含附录）。1 篇（SpAtten-Chip，ISLPED 2023）无公开 PDF，只看了项目页摘要，不计入任何"N 篇"的统计。
- 两处下载错误已修正：TorchSparse++ 第一次下成了 TorchSparse，后从项目页 Dropbox 取到 MICRO'23 正文重读；"tinyml" 条目第一次下成了 VILA，查明是合集条目后删除。

**没读到的。**
- 附录缺失：DGA（证明、训练曲线）、Park（环境 MDP、超参）、PVCNN、SparseRefine（各模型阈值）、FlatFormer（详细指标）、FastComposer（基线超参、prompt 列表）。这些可能是论文版本本身不带附录，没有逐一核实。
- 图内数值：几乎所有曲线和柱状图的数值在文本提取中丢失，结论只用正文和表格里的数字；凡是只靠曲线的结论，笔记里标为"作者声称"。
- 表格重建：HAQ Table 3–6、QuEst Table 3/Fig. 10、MCUNet Table 2/3/4、TSM Table 8、EfficientViT Table 1、NetAug Table 4/5 的部分行列是按数值对齐重建的，可能错位。
- 四篇（HAT、TSM、EfficientViT、NetAug）的 layout 文本损坏严重，改用非 layout 模式重新提取后阅读，文本在 `raw/txt_plain/`。

**核对方式。**
- 逐篇精读由 9 个并行子任务完成，每篇一份笔记，每组一份综合。
- 主会话回原文抽查了 17 条关键数字：AWQ Table 1 三臂、Quest 7.03×/1.74×、DC-AE 的 CFG 开关下 FID、COAT 1.36×/1.42×、QeRL 学习率、Jet-Nemotron batch、DC-Gen Tuned 83.94/83.97 与 DC-VideoGen 84.83、StreamingLLM "\n" 替换、OFA #1200、SpArch 5.7×、DuoAttention 随机臂、TorchSparse++ 不排序 1.2×、MCUNet 随机空间 74.7±1.9、DiffAugment 容量与 R1 扫描、HAT 随机搜索。16 条对上，NetAug 的"欠拟合假说"因原文本损坏改用重提取文本核到了关键词。记录在 `notes/_spotcheck_log.md`。
- 其余数字没有逐一复核。引用前如需严格依据，回 `notes/<slug>.md` 再回原文。

**结构性局限。**
- 94 篇全是效率、系统、硬件、生成、量子方向，没有安全方向的论文。第 6 节的映射是推测。
- "共同套路"的判定基于各组内部，跨组的合并是主会话做的，没有再回原文验证每一条的跨组计数。文中的"约 N 篇"是各组报告相加，不是精确计数。
- 本文没有做查重：没有检索其他实验室是否也用同样的手法，也没有检索是否已有人写过类似的总结。这不影响本文的用途（我们自己借鉴），但如果要把它当作对外的主张，需要先过七步查重。

---

## 附录 A：按论文的索引（每篇一句：最关键的说服手法，括号内是最主要的薄弱点）

> 长 slug 在此按前缀缩写（lightning-opd / lightning-opd-2-0 / lightning-weave / qerl / qoc / retrospective-eie），对应 `notes/` 下以同样前缀开头的完整文件名。

### 量化与低比特训练
- **awq** — Table 1 的三臂对照：按激活、按权重、随机各保留 1% 通道，随机臂当零假设。（新模型上只与 FP16 比）
- **smoothquant** — Table 1 先模拟一个硬件做不了的 per-channel 激活量化证明问题在通道维，再给可部署的等价迁移。（逐模型调 α）
- **qserve** — 从"W4A4 理论更快、实测更慢"的反常出发，用 CUDA core 与 tensor core 吞吐比定位瓶颈，算法约束由硬件约束倒推。（效率分母取 TRT-LLM 三种配置里最快的，但 A8 对比 A4 基线）
- **svdquant** — 两步可视化：平滑把异常值推到权重，SVD 前几个奇异值再吸收；kernel 融合让低秩分支开销从 57% 到 5–10%。（头条加速依赖基线 offloading）
- **deltaquant** — 把视频编码的帧间差分嫁接到激活量化；Fig. 4 证明静态平滑跨时步失效。（111.8× 里量化只占 3.0×）
- **qerl** — "量化噪声在 RL 里反而有益"的反常识观察。（主对比学习率不同，无等幅噪声对照）
- **coat** — 算子级显存账本算出理论 1.69×，实测 1.65× 对上。（"持平或超过 TE"与 1.36×/1.42× 不符）
- **fourtune** — kernel 融合逐步累加 1.25×→2.52×；按墙钟时间画收敛曲线。（RL 后 ImageReward 低于 BF16 LoRA 却称 comparable）
- **quant-videogen** — 把 KV cache 重定义为长视频能力问题；INT2 下 LLM 方法在视频上崩溃。（加速口径三处不一）

### 长上下文与稀疏注意力
- **streamingllm** — 先定位首 token 被逐出那一刻 PPL 爆炸，再用"\n"替换证明是位置而非语义。（"infinite"措辞；StreamEval 为方法而设）
- **duo-attention** — Fig. 1 右的三顺序干预曲线。（头条 2.55× 是 MHA，GQA 只有 1.5×）
- **quest** — Top-10 召回率量化"关键 token 随 query 变化"；"问题逐 token 解码"的 passkey 设置。（标题 7.03× 是 kernel，端到端 1.74×）
- **lserve** — 先摆出"大页快但大页让 Quest 失效"的实测矛盾，分层分页化解。（精度只与 Dense 比）
- **xattention** — 反对角线的几何论证 + 同预算随机/对角对照。（阈值搜索数据未交代，随后在 RULER 上报告）
- **radial-attention** — 时空衰减拟合成指数（R²>0.985）再推复杂度和误差界。（Vision Reward 差 0.001–0.006 无方差）
- **longlora** — Table 2：rank 8→256 不降，只打开 norm/embed 就降。（只有 Llama2）
- **long-rl** — 同底座前后对照，6 个基准全涨。（checklist 声称统计显著性但正文没有）
- **streamingvlm** — "ReKV 套在微调模型上胜率归零"证明训推一致的必要。（GPT-5 既清洗数据又当评委）
- **jet-nemotron** — 每张精度表带吞吐列；实测 53.6× 逼近上界 56×。（吞吐用各模型各自最大 batch）

### 高效图像/视频生成
- **sana** — Fig. 2 的 4K 延迟瀑布图 469s→9.6s，每个组件一个倍数。（分母是 FLUX-dev；残留 TODO）
- **sana-1-5** — 推理时 best-of-N 把 GenEval 从 0.81 提到 0.96。（判分器与评测器同源）
- **sana-sprint** — 先证明 Flow→TrigFlow 变换无损，主表按 1/2/4 步分组。（头条数字三处不同）
- **sana-video** — 复杂度表 + "显存随长度恒定、基线 OOM"。（加速有 16/13/53/8× 四种说法）
- **dc-ae** — 嵌套容量实验把失败归因到优化困难，同图展示修复。（FID 改进只在开 CFG 时成立）
- **dc-ae-1-5** — Fig. 1b 两条反向曲线立矛盾；Tab. 6 方法用在没问题的 c32 上无效，反证定位。（"4× 更快"是 DC-AE 已有能力）
- **dc-ar** — 纯离散消融与 MAR 64 步曲线两头夹出混合设计。（不与 VAR 类比较）
- **dc-gen** — "Tuned"对照剥离数据效应；逐层特征距离诊断；本组唯一系统人评。（53.8× 分母是不支持 4K 的 FLUX 硬跑）
- **dc-videogen** — 因果/非因果二分提出 chunk-causal。（加 Tuned 对照后优势几乎消失）
- **hart** — 把图像拆成"离散 token + 残差"，主动承认重建好不等于生成好。（31% 改进混入 tokenizer 改进）
- **lpd** — 自建同 tokenizer、同架构的光栅顺序对照；PTA 量化局部性。（"至少快 3.4×"排除了 VAR）

### GAN/扩散压缩、可控生成、VLM
- **vila-u** — Table 7 同数据只加对比损失的两行对照。（384 版低于 256 版未解释）
- **distrifusion** — 两难用数字量化（朴素 patch FID 259、TP 通信量），再用同步下界与无通信上界夹出开销。（6.1× 来自无质量评估的 3840²）
- **sige** — Table 3 逐步累加：MACs 7.6× 而延迟 1.6×，逐项加到 2.9×；5 种硬件。（7.2× 来自单个样例）
- **gancompression** — 每处并列"同 MACs 均匀缩通道"的朴素基线；重训原模型；按对手协议重算 FID。（头条用 MACs，GPU 实测远小）
- **anycost-gan** — 把压缩改写成一致性预览，一致性做成两个指标。（正文 8.5× 对表 6.3×）
- **fastcomposer** — 可视化 cross-attention 后直接用分割 mask 监督它。（"一致性更好"与表格不符）
- **diffaug** — 诊断量闭环：判别器训练/验证准确率差；扫描容量和 R1 排除替代解释。（每设置挑 snapshot）
- **longlive** — 三臂 KV cache 消融。（核心机制主张无对照）
- **can** — 同架构只差条件化开关；核选择与长训练两个排除实验。（52×/229× 混入步数差）
- **vila** — MMC4 对 MMC4-pairs 只拆一个属性；每个消融并列 MMLU 守恒指标。（与 LLaVA-1.5 的对比混入 SFT 数据差异）
- **longvila** — kernel 级测量立痛点；系统对比用对手推荐配置。（自研分片只贡献 1–7%）

### 扩散 LLM、推理模型后训练、VLA、联邦、平台
- **fast-dllm** — KV 激活跨步相似度热图直接对应块级缓存；2×2 组件表。（无同期缓存方法对比）
- **fast-dllm-v2** — 同数据同步数的标准微调对照 Nemo-FT。（"lossless"与 MATH 掉 10 分不符）
- **lightning-opd** — 2×2 教师网格验证"教师一致性"预测。（4× 不含 SFT 成本；单 seed）
- **lightning-opd-2-0** — 阈值扫描检验修正信号是否靠近一致教师。（缺"只减全局均值"对照）
- **lightning-weave** — 两动作反例证明朴素离线目标无不动点；混合权重扫成 Pareto。（训练截断 2K 评测到 32K）
- **tlt** — 第三方生产 trace + roofline 立痛点。（"无损"只有 100 步奖励曲线）
- **vlash** — 乒乓任务所有基线 0/20 的二值证据。（11.8× 主要来自公式）
- **foreact** — VLM+π0 只差前瞻图像的对照。（真机用自定义评分，小样本无方差）
- **dga** — 重构问题：带宽已解决、延迟是物理极限；加强 FedAvg 基线。（设置更像高延迟数据并行）
- **park** — 六类挑战每类一个最小证据，12 个环境映射。（环境被缩小以便标准 RL 能跑）

### 经典压缩、剪枝、NAS、高效结构
- **deep-compression** — 逐层逐阶段的压缩率拆解表。（量化后模型未在硬件上跑）
- **pruning** — 五条 trade-off 曲线 + Naive Cut 朴素基线。（效率只有参数量和 FLOP）
- **dgc** — 公式对比指出朴素稀疏化丢了动量项；跨 3 模态 5 数据集。（64 节点加速靠性能模型）
- **haq** — 3 策略 × 3 硬件的延迟交叉矩阵；延迟对齐到小数点再比精度。（"RL"与"混合精度"捆在一起；模拟器）
- **amc** — 剪枝算法和 FLOPs 固定，只换"谁定逐层剪枝率"；Pixel 1 实测。（摘要 2.7% 对表格 0.3%）
- **apq** — 精度/延迟/能耗/GPU 小时/CO2/美元一张大表。（"联合"与"更大空间"混淆，作者自承）
- **ofa** — 等算力基线 MobileNetV2 #1200；12 种硬件。（子网精度混入 KD 和 600 epoch）
- **proxylessnas** — 模型 × 硬件交叉矩阵；同台 Pixel 1 重测 MnasNet；延迟预测校准图。（"无代理"与大空间混淆）
- **hat** — 三硬件维度–延迟 profiling 同时证明两件事，搜出的架构回扣这张图。（关键组件未消融）
- **lite-transformer** — 计算量分解 + 注意力热图，等 Mult-Adds 对比。（无任何移动端实测）
- **tsm** — 三硬件实测朴素 shift 又慢又掉点，两处最小修改修好；事前按时序强弱分组预测增益；I3D_replace 反事实。（shift 比例在 Kinetics 上选）
- **efficientvit** — 手机 CPU 实测 softmax 与线性注意力延迟差；等 MACs 消融。（摘要 13.9× 对正文 8.8×）
- **netaug** — "同样正则，大模型涨小模型跌"的对照图；欠拟合假说的双向预测验证。（自身超参未消融）

### TinyML、片上训练、3D、自动驾驶
- **mcunet** — "库 × 模型"因子网格 + 对基线的最优缩放。（Table 2/3 同约束数字不一）
- **mcunetv2** — 逐块峰值内存图：只有前 5 块不够。（检测表沿用 V1 的 YOLOv2 数字）
- **mcunetv3** — 瀑布条把 2300× 拆成四段；范数比观察→推导 s⁻² 偏差→QAS 修复的闭环。（精度在 GPU 上模拟）
- **tinytl** — 参数小 4.3×、训练激活只小 1.1× + 一行反传公式。（全部理论内存）
- **tinyml-magazine** — Table III 同一块 H743 重测 10 个模型，如实显示 TinyEngine 在 3 个模型上峰值内存更高。（其余为自家工作转写）
- **pockengine** — 7 平台吞吐矩阵 + ±std。（加速数字四种说法；承诺的结果节不存在）
- **litepose** — 渐进收缩：把多分支逐步拆成单分支，AP 从 50.4 升到 53.0。（base channel 与 MACs 同时变）
- **pvcnn** — 硬件能耗常数立论 + 运行时分解（有效计算 12–45% 对 95%）。（精度差距 0.1 量级）
- **spvnas** — 自家 PVCNN 大场景失效立题；MinkowskiNet 升级到同一实现后在四个成本档比较；验证集重跑搜索。（"+3.3 mIoU"与"3× 加速"分属两个模型）
- **bevfusion** — profiling 出 BEV pooling 占 80%，kernel 加速 40×；检测和分割两个性质相反的任务。（本版无组件消融；val NDS 只高 0.1）
- **sparserefine** — 高/低/低+细化三行组在 5 个架构上重复；每项消融给 oracle 上界。（阈值按追平基线选）
- **semalign** — 语义对齐损失与旋转误差 Spearman 0.72，损失同时当目标和评价。（用自己优化的损失评价泛化，循环）

### 硬件加速器与稀疏系统
- **eie** — 单位能耗表算数量级账；让 CPU/GPU 也跑压缩模型证明只有专用硬件能兑现。（28nm 投影方法未给）
- **retrospective-eie** — 无实验；用"偷懒"原则把 EIE 与 LLM 单 batch 解码串起来。
- **spatten** — 逐步叠加分解故意展示剪枝单独只 1.1×，加专用 top-k 才 3×。（"无精度损失"靠逐任务调参）
- **sparch** — 每项技术写成修补上一项的副作用，第一步慢 5.7× 照样保留；解析 DRAM 模型逐步对上实测。（40nm 对 32nm 不归一化）
- **pointacc** — 同一算法在 CPU/GPU 更慢、在 PointAcc 快 10× 以上的反事实。（对 Mesorasi 100× 混入模型变化）
- **torchsparse** — profiling 归纳两条原则，逐项消融对照理论上限；1080Ti 复测排除硬件特性。（测量协议未交代）
- **torchsparsepp** — Table 3 对 Table 4：kernel 更快的排序版端到端反而慢 1.2×，把"代理指标失效"写成贡献。（自家方法独享穷举调优）
- **flatformer** — 逐一量化三类点云 Transformer 慢在哪；"MACs 多 1.7× 却更快"。（仅 Waymo）
- **lego** — 能力矩阵表占住唯一空格；与每个对手比较都换到对方的工艺/FPGA/资源。（流片、投影、综合结果同表）
- **naas** — 在知名加速器资源包络内搜索；"只搜尺寸参数"作对照。（纯解析模型；摘要 4.4× 对正文 4.88×）

### 量子计算与模拟电路
- **quantumnas** — 真机实测图：无噪声精度随参数升，真机先升后降；交叉设备矩阵。（剪枝比例按真机测试精度挑）
- **quantumnat** — 定理：噪声 = 线性缩放 + 平移 → 测量后归一化；5 台真机累加表。（归一化用测试批统计）
- **qoc** — 梯度幅值 vs 相对误差图导出梯度剪枝；横轴用电路执行次数。（验证集峰值，无独立测试集）
- **atomique** — 无实机时自建分项保真度模型；三层基线；六参数敏感性扫描并标出 FAA 反超区间。（所有平台相干时间 ×10）
- **q-pilot** — 借 FPGA 布线提出飞行辅助比特，拓扑无关的等价性证明。（只报深度不计移动时间，保真度更差）
- **quest-q** — PST 与保真度相关 0.993 为代理正名；5 台真机散点 R²>0.95。（去掉 T1/T2 后真机 RMSE 反而降）
- **gcn-rl-circuit-designer** — 三层基线 + 去掉 GCN 的同款 RL；本组唯一报方差。（"胜过专家"靠自定义等权 FoM）
- **learning-to-design-circuits** — 课程竞赛的人类设计做锚；样本效率 250×。（两级 TIA 略输人类却写 better）

### 未读到正文
- **spatten-chip** — ISLPED 2023，无公开 PDF。项目页摘要：最大与最小子 Transformer 间延迟和能耗可伸缩 6.9×；word elimination 省能 16%，F1 掉 14.5%。
