# G1 组综合：LLM/扩散模型量化与低比特训练

覆盖 9 篇：awq、smoothquant、qserve、svdquant、deltaquant、qerl、coat、fourtune、quant-videogen（下文简称 QVG）。
全部全文已读；读不全的只有图中曲线数值（pdftotext 丢失），各笔记 §0 已逐篇说明。

先说最不利的结论：**9 篇全部没有方差、没有多 seed、没有置信区间**；有 6 篇的文字结论或摘要数字，与自己的表格不一致或超出了表格能支持的范围（见 §3）。这个组"solid"的印象主要来自**广度、系统实测和机制可视化**，不来自统计严谨性。

---

## 1. 跨论文共同的实验设计套路（出现 ≥2 次）

### 1.1 "异常值/分布可视化 → 变换后再可视化"作为方法的第一证据
在变换前后画同一层的数值分布，让读者直接看到"难量化 → 易量化"。
- SmoothQuant Fig. 4（OPT-13B 激活/权重 3D 幅度，平滑前后）
- AWQ Fig. 2（RTN / 1% FP16 / 放缩三联图，附 PPL）
- QServe Fig. 7（post-RoPE Key 平滑前后、Value 无异常值）
- SVDQuant Fig. 4（X、W、平滑后 X̂/Ŵ、SVD 残差 R 五联图）+ Fig. 5 奇异值谱
- DeltaQuant Fig. 4（跨时步异常值变化）+ Fig. 5（时间/空间均值与 delta）
- COAT Fig. 2/3/9（优化器状态动态范围扩展前后）
- QVG Fig. 3（k-means 分组 → 减质心 → 分布收窄）

### 1.2 "把难量化部分分解出去"的同构方法：高精度小分量 + 低比特残差
- AWQ：1% 显著通道（先用混合精度证明，再用等价放缩替代，Table 1→3）
- SmoothQuant：把激活难度迁移到权重（Eq. 3–4）
- SVDQuant：16-bit 低秩分支 + 4-bit 残差（Eq. 5）
- DeltaQuant：FP8 core token + FP4 delta（Eq. 3–4）
- QVG：质心 + 残差，多阶段（Sec 4.1–4.2）
- FourTune：冻结 SVD 稳定器 + 4-bit 主干 + LoRA（Eq. 2）
- QServe：渐进量化（INT8 中间 + INT4 组量化，Eq. 4–5）
这是本组最核心的"方法生成器"：观察到异常值集中 → 找一个便宜的分量把它吃掉 → 证明剩下的部分好量化。

### 1.3 瓶颈 profiling / roofline / 资源账本先行
在提方法前，用测量或解析账本证明"瓶颈在这里"。
- AWQ Fig. 3（生成阶段 memory-bound、权重访存主导、roofline）
- QServe Fig. 2a（attention/GEMM 时间占比）、Fig. 3（A100 roofline）、Sec 3.2（CUDA core 只有 INT4 tensor core 的 2%）
- SVDQuant Fig. 2（扩散 vs LLM 的计算量/参数比）
- DeltaQuant Fig. 3（注意力优化后线性层占 65%）
- COAT Fig. 1c + Table 2（逐类显存分解、算子级激活账本，单位 U）
- QVG Sec 3.1（KV 显存公式与 34 GB vs 27 GB 实例）

### 1.4 理论上限 vs 实测对上
- COAT：Table 2 理论 1.69× 激活压缩 vs Table 7 实测 1.65×
- AWQ：W4 理论 4× 算术强度（Fig. 3b）vs TinyChat 实测 3.2–3.3×
- QServe：Fig. 3 roofline 预测 vs Table 1 朴素 KV4 在 A100 反而慢（用实测推翻朴素预期，再修复）

### 1.5 "朴素实现的开销 → 融合 kernel 后的开销"两步展示
先承认多出来的分支/步骤在朴素实现下很贵，再用融合把它压下去。
- SVDQuant Fig. 6a（低秩分支朴素 57% 开销 → 5–10%）、App. Table 4（250 ms → 218 ms vs 朴素 INT4 212 ms）
- FourTune Fig. 9（4-bit 无融合仅 1.25× → +LoRA 融合 1.82× → +MLP 融合 2.52×）
- QServe Table 1（朴素 KV4 在 A100 0.87–0.90× → 优化后 1.29–1.51×）+ Sec 6.4 KV4 kernel 逐项 0.48→0.28 ms
- AWQ Sec 5.4（HF FP16 52 → 自家 FP16 融合 62 tok/s → 再 3.1×）
- QVG Sec 4.3（k-means 质心缓存降 3× 聚类开销）
- DeltaQuant Sec 5.1（core token 生成融合进量化 kernel）

### 1.6 Roadmap 式逐步累加消融（同图同时报精度与效率）
- QServe Fig. 16（W8A8 → +W4 → +旋转/平滑 → +clip → +KV4 → +SmoothAttention → +渐进 → +重排；PPL + 吞吐 + 显存）
- FourTune Fig. 9（kernel 融合 roadmap）
- AWQ Table 1 → Table 2 → Table 3（选通道 → 放缩 s → 搜索）
- SmoothQuant Table 3/4 + Table 11（O1→O2→O3 精度与延迟对应）
- COAT App. Table 11（BF16 → +FP8 优化器 → +FP8 激活）
- DeltaQuant Table 2（+LightX2V → +Radial&Sage → +DeltaQuant）
- SVDQuant App. Table 3（低秩 × GPTQ × INT/FP 全因子）

### 1.7 以工业/社区最强实现作为效率基线
- QServe：TensorRT-LLM 三种精度配置取最优作为分母（Table 4 注）
- SmoothQuant：FasterTransformer 生产级后端（Fig. 9）
- COAT：TransformerEngine（Table 7/8）
- SVDQuant：社区 NF4 FLUX + TensorRT W8A8（Table 1）
- DeltaQuant：社区 GGUF Q4（Table 1, Fig. 8）
- AWQ：llama.cpp、exllama、AutoGPTQ（Fig. 10）
- QeRL、FourTune：QLoRA（NF4）

### 1.8 "装得下"作为头条结果（更少/更小硬件）
- AWQ：Llama-2-70B 上 Jetson Orin、13B 上 8 GB 笔记本 4070
- SmoothQuant：530B 单节点、175B 用一半 GPU（Table 10, Fig. 9）
- QServe：L40S 上超过 A100 上的 TRT-LLM（Table 4）
- SVDQuant：16 GB 笔记本 4090 免 offload（10.1×）
- DeltaQuant：Wan2.2 装进 32 GB 5090
- QeRL：32B RL 单张 H100（Table 8）
- COAT：7B 单卡全参训练（Table 8）
- QVG：HY-WorldPlay-8B 上 4090（Fig. 1，声称）

### 1.9 广度代替深度：多模型族 × 多任务 × 多硬件
所有 9 篇都用"跨模型族/跨尺寸/跨模态"建立通用性（AWQ 覆盖 OPT/LLaMA/Llama-2/Mistral/Mixtral/VLM/代码/数学；SmoothQuant 覆盖到 530B；QServe 8 个模型 × 2 卡；SVDQuant 6 个模型 × 2 数据类型 × 3 卡；COAT 预训练/微调/VLM；FourTune 定制化/RL/蒸馏三类任务）。

### 1.10 超参扫描画成 U 形/权衡曲线并给机制解释
- SmoothQuant Fig. 10（α：太小激活难、太大权重难）
- AWQ Table 2（s 增大：显著通道误差降、非显著通道 Δ 变大）
- SVDQuant App. Fig. 18（rank：画质 vs 开销）
- DeltaQuant Fig. 9a（cube 越大越差，到全局均值最差）
- QVG Fig. 7b/c、App. Table 5（block size、K）
- COAT App. Table 10（组尺寸：MSE vs 额外显存）

### 1.11 同预算 / 同质量的"对齐比较"
- DeltaQuant Fig. 9b（同延迟下 LPIPS，vs 加大 rank 的 SVDQuant）
- SVDQuant App. Fig. 19（4-bit SVDQuant ≈ 7-bit naive）
- QeRL App. J（vLLM 显存比例对齐）
- QServe Fig. 17（同 batch 对比，拆出"更快 kernel"与"更大 batch"两个因子）
- FourTune Fig. 4（按墙钟时间的收敛曲线）

### 1.12 在"该量"的那个量上测误差，而非泛泛的张量误差
- COAT Table 1（测 m/√v 的 MSE，因为那才是更新量）
- SVDQuant Prop. 4.1（误差上界拆成四项，再只优化 ‖R‖）
- AWQ Eq. 4（最小化输出误差而非权重误差）
- QServe Eq. 10（q/k proj 用 block 输出误差做 clip 搜索）

### 1.13 从成熟学科借结构先验（跨领域类比）
- DeltaQuant：H.264 帧间差分（Sec 4.2）
- QVG：可伸缩视频编码 SVC 的渐进式编码（Sec 3.3）
- QeRL：RL 中的参数空间噪声探索（Plappert 2017、NoisyNet，Sec 3.2）

### 1.14 自家前作作为前提或对手（谱系式推进）
SmoothQuant → AWQ（引用为 W8A8 对照）→ QServe（复用 SmoothQuant 平滑、AWQ 重排、QuaRot 旋转）→ SVDQuant（复用平滑 + 引入低秩）→ DeltaQuant（SVDQuant 为主要对手且复用其权重侧）、FourTune（SVDQuant 低秩分支变训练期稳定器）；DeltaQuant 把同组 Radial Attention 作为"已解决的前一个瓶颈"。

---

## 2. 本组特有的套路（相对一般 ML 论文）

1. **硬件单元级的成本账**：不止 FLOPs，而是"CUDA core vs tensor core 吞吐比"（QServe Sec 3.2/5.3）、"寄存器级并行与溢出"（QServe Fig. 14）、"SIMD 位宽与打包"（AWQ Fig. 4）、"L2 放不下导致额外 DRAM 读写"（SVDQuant Sec 4.3）。用硬件约束倒推算法约束（QServe 的保护区间 [-119,119] 由不溢出条件推导出）。
2. **"不可部署的理想方案"作为上界**：SmoothQuant Table 1 模拟 per-channel 激活量化（硬件不支持但能证明问题在通道维）；AWQ Table 1 的 1% FP16 混合精度（硬件不友好但证明显著通道存在）。先证明"理想上限可达"，再造一个"等价但可部署"的版本。
3. **生成模型用"相对 16-bit 参考的保真度"（LPIPS/PSNR/SSIM）+ "绝对质量"（FID/ImageReward/VBench）双轨评测**：SVDQuant、DeltaQuant、QVG、FourTune 都这样做，让"像不像原模型"和"好不好"分开。
4. **"多工作点"方法族**：SmoothQuant O1–O3、QVG / QVG-Pro、SVDQuant INT4/NVFP4、QServe per-channel/per-group——同一方法给出精度优先与效率优先两个配置，分别去赢不同的对比。
5. **把基线推到其设计范围之外让其崩溃**：SmoothQuant 对 ZeroQuant/Outlier Suppression 在 175B（Table 3，近随机）；SVDQuant 把 ViDiT-Q/MixDQ 跑到 W4A4（FID 300–400）；QVG 把 KIVI/QuaRot 跑到 INT2 视频；DeltaQuant 的 Q-VDiT（VR −0.085）。这能制造巨大差距，但也是公平性的弱点（§3）。
6. **数学等价变换 + "可折叠进前一层"的零开销论证**：SmoothQuant（s 折进 LayerNorm/前一层）、AWQ（s⁻¹ 折进前一算子）、QServe（旋转矩阵、平滑因子、SmoothAttention 的 λ 折进权重，并为 RoPE 加约束 λ_i=λ_{i+D/2}）、QeRL（噪声并入 RMSNorm，App. G 证明）。

---

## 3. 本组反复出现的薄弱点

1. **零统计**：9/9 篇无方差、无多 seed。而多篇的效应量落在其指标的自然波动内：COAT OLMo-7B ARC-Easy −3.8 / HellaSwag +3.7 被平均抵消（Table 4）；QeRL 的 AIME/AMC 差异多为 1–4 道题（Table 2）；FourTune 定制化每类仅 5 个概念，Qwen-Image Subject 的 Image Quality 从 16.92 跳到 26.92（Table 1）；DeltaQuant 中 FP8 core 比 BF16 core 高 0.4 PSNR（Fig. 9c，暗示噪声 ≈ 0.4），而其时步自适应消融收益只有 +0.5（Fig. 9d）。
2. **文字结论超出表格**："consistently"/"comparable"/"matching" 被用于并不一致的结果：
   - DeltaQuant Table 1：Wan2.2 T2V 上 RTN 的 Vision Reward 0.093 > Ours 0.083；I2V 上 SVDQuant 0.146 > Ours 0.143。
   - FourTune Table 3：RL 后 ImageReward 1.0152 低于 BF16 LoRA 1.0735、QLoRA 1.0931 和基模 1.1189；Table 5 块量化 LPIPS 0.227 vs 0.203 被称为"comparable"。
   - SVDQuant App. Table 3：文中把 PSNR 21.5 归给"SVDQuant+NVFP4"，实为"无低秩分支 + GPTQ"那一行。
   - QeRL："matching full-parameter"只在 MATH500 一项成立。
   - COAT："on par with or surpassing TE"——8 卡 Llama-2-7B 上 TE 1.42× > COAT 1.36×（Table 8）。
3. **摘要/正文/附录数字前后不一**：QeRL（App. K "over 2×" vs Table 3 的 1.2–1.5×）；FourTune（2.25× vs 正文 2.02×）；QVG（"<4%" vs Self-Forcing 4.3%，附录又是 1.7%）；COAT（App. Table 11 vs Table 8 峰值显存）；SVDQuant（App. Table 4 标题 desktop vs 正文 laptop）；AWQ（正文引用"Table 5"实为 Table 3）。
4. **头条加速数字依赖混杂因素**：offloading（SVDQuant 8.7×/10.1×，DeltaQuant vs OOM 的 BF16）；与蒸馏/注意力叠加（DeltaQuant 111.8×，其中量化只贡献 3.0×）；batch 变大（QServe 最大吞吐、COAT 最大 BS——两篇都做了拆分，值得肯定）；弱基线（AWQ 对 HF FP16）；"up to"取极端点（QServe Qwen1.5-72B 3.5×，此时 TRT-LLM W4A16 仅 17 tok/s）。
5. **跨精度比较**：QServe 的 A8 对 Atom/QuaRot 的 A4（精度表）；SVDQuant 的 W4A4 对 ViDiT-Q/MixDQ 的 W4A8；QVG-Pro 用 4.97× 压缩对基线的 6.40×。
6. **新模型上只与 FP16 比**：AWQ Table 5（Mistral/Mixtral）、Table 7（VILA）；SmoothQuant Table 6/7（LLaMA/Llama-2/Falcon/Mistral/Mixtral，且逐模型调 α）。这些"泛化"证据只能说明损失小，不能说明优于他人。
7. **机制主张缺反事实对照**：
   - QeRL：量化噪声促进探索，但主对比中 QeRL lr=1e-5、BF16 LoRA lr=5e-6（App. E Table 4），且没有"BF16 + 等幅噪声"对照。
   - QVG：动机是时空邻近性，方法是语义 k-means，二者之间无对照。
   - FourTune：稳定器只用梯度范数证明，无"无稳定器"的最终质量。
   - DeltaQuant：缺"只用 delta、不用 SVD 低秩"的单独消融。
8. **用代理指标做消融**：FourTune Table 5 用推理重建验证训练期的块量化；QVG、COAT 的大部分消融停在 MSE 层面；SVDQuant 组件消融主要是 IR + 两张图（Fig. 10）。
9. **最接近的工作缺席**：COAT 没有 FP8-LM 的精度/速度对比；AWQ 没有 SpQR/OWQ；QVG 没有 KVQuant/PQCache/CommVQ；FourTune 没有其他 4-bit 训练方案；QServe 没有其他 W4A8 方法的端到端对照。
10. **局限性章节普遍缺失**：只有 QeRL（App. K）和 SVDQuant（App. E.6 的 2-bit 失效）写了实质局限；SmoothQuant 附录 A 有讨论；其余要么没有，要么只有 broader impact。

---

## 4. 给"怎么做出 solid 感"的提炼（我的归纳）

- 说服力的主轴是**"观察 → 可视化 → 一个理想上界或解析推导 → 可部署的等价实现 → 真实硬件端到端"**这条完整链，每一环都有一张图或一张表（AWQ、SmoothQuant、QServe、SVDQuant 最完整）。
- 系统部分的可信度来自**承认朴素实现会慢**，然后逐项展示融合/重排带来的收益（§1.5、§1.6）。这比直接报一个最终加速比可信得多。
- 弱点集中在统计和措辞：对本组论文中 ±1 量级的精度差异，应默认视为"无法区分"，直到有方差。
