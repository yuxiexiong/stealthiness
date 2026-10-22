# G2 组综合：长上下文 / 稀疏注意力 / KV cache / 高效架构

本组 10 篇：streamingllm、duo-attention、quest、lserve、xattention、radial-attention、longlora、long-rl、streamingvlm、jet-nemotron。全部读完全文与附录（quest、xattention 无附录）。下文"位置"均指各篇笔记中已核对的图表号。

## 一、跨论文共同的实验设计套路（≥2 篇出现）

### 1. "注意力图 → 可干预的观察 → 方法"的动机链
先用注意力可视化提出现象，再用一个**反事实/干预**实验把现象变成可操作的事实，最后方法直接利用它。
- StreamingLLM：Fig.2 热图 → Table 1 把首 4 token 换成 "\n" 仍能恢复（区分语义 vs 位置）。
- DuoAttention：Fig.1 左/中两类头 → Fig.1 右"先压 retrieval 头 / 先压 streaming 头 / 随机"三条曲线。
- Quest：Fig.2 单头示例 → Fig.4 Top-10 召回率量化"随 query 变化"。
- XAttention：Fig.2 几何论证 → Table 6 同采样预算的 random / diagonal 对照。
- Radial Attention：Fig.4 衰减曲线 → Fig.8(b) 指数拟合 R²>0.985 → Eq.3 参数化。
- Jet-Nemotron：Fig.5(a) 逐层"只留一层全注意力"热图 → Fig.5(b) 学习放置 vs 均匀放置。
- LongLoRA（非注意力图，同构）：Table 2 "rank 8→256 不降、打开 norm/embed 就降"。

### 2. 并排总图：便宜基线 / 昂贵上界 / 本法
一张图同时标出各方案的复杂度和质量，读者一眼看到"本法质量≈上界、成本≈便宜方案"。
- StreamingLLM Fig.1（dense/window/recompute/ours 的 PPL 5641/5158/5.43/5.40）
- StreamingVLM Fig.1（胜率 3.89/23.54/66.54/66.18，并标 O(T²)/O(TW)/O(TW²)/O(TW)）
- Quest Fig.1（dense / query-agnostic / query-aware）；Radial Fig.3（SVG vs 本法）；Jet-Nemotron Fig.1（精度–吞吐散点）

### 3. 同预算比精度
固定 KV 预算、PFLOPs、稀疏率或密度后再比精度，堵住"多花计算换精度"的质疑。
- DuoAttention Fig.7 / Table 3–4（同 KV 预算）；Quest Fig.7（token 预算扫描）；XAttention Table 6–9（每个消融都报"精度 + 密度"，Table 8 把 TopK/Ratio 调到相近计算量）；Radial Table 1（同 PFLOPs）、Table 2（同稀疏率）、附录 D（调稀疏度保持总计算相同）；LongLoRA Table 6（替代模式"相近计算量"，Fig.7 注）。

### 4. 反过来：同精度比效率
- Quest Fig.11（先定"LongBench 无损"，再比所需预算与延迟）
- Jet-Nemotron（每张精度表都带吞吐列，Table 3–8）
- LServe Table 3（同时给 4096/8192 两档，注明延迟差 ≤6%）

### 5. 逐项累加的 roadmap 拆解
- LServe Fig.15/16（dense → +静态稀疏 → +动态稀疏 → LServe）
- Jet-Nemotron Fig.3（PostNAS 四步各自增益）
- StreamingVLM Table 6（基座 → +LiveCC 数据 → +自建数据 → +退火数据）
- LongLoRA Table 1（短注意力 → +移位）
- Long-RL Table 5（✗/✓/O 三值编码阶段 × 数据）

### 6. 实测与理论上界对账
- Jet-Nemotron：理论上界 56×（14×4），实测 53.6×（Sec.3.3）
- LServe：kernel 与 oracle 1−sparsity 并列（Fig.12），理论加速 1/(1−r)（Sec.3.1）
- Radial：PFLOPs 节省 1.8× 对应实测 1.9×（Table 1）
- Quest：解析式 1/PageSize + K/PageNum（Sec.3.5）及 kernel 收敛到 1/PageSize（Fig.8a）
- StreamingLLM：O(TL) vs O(TL²) 解释 22.2× 来源（Fig.1, Fig.10）

### 7. 效率按序列长度（或帧数）扫描，展示收益随长度放大
StreamingLLM Fig.10、DuoAttention Fig.9–11、Quest Fig.8–10、LServe Fig.10–11/Table 7、XAttention Fig.4、Radial Fig.2/Table 2、Jet-Nemotron Fig.6、StreamingVLM Fig.7、Long-RL Fig.2。几乎所有篇的标题数字都取自扫描的**最长端**。

### 8. 合成检索压力测试 + 真实长任务两层评测
passkey / NIAH / RULER 证明"没丢信息"，LongBench 等证明"真实任务也行"。
- DuoAttention（NIAH 至 1048K + LongBench 21 任务）、Quest（passkey 10K/100K + LongBench 6 任务）、LServe（NIAH + RULER + LongBench）、XAttention（RULER + LongBench）、LongLoRA（passkey + topic retrieval + 附录 LongBench）、StreamingLLM（StreamEval + 附录 LongBench）。

### 9. 改造基线并公开改造方式
基线原生跑不了长上下文时，作者自己改，并论证改法对基线有利或中性。
- DuoAttention 附录 A.3（H2O/TOVA 改用 FlashAttention 精确 prefill + 最后 50 token 模拟解码，明说这会让基线更强）、A.5（FastGen 用社区代码复现，只能测到 24K/32K）
- Quest Sec.4.2.2–4.2.3（H2O 从解码阶段才开始累计分数；问题部分逐 token 喂入），Fig.11(b) 基线延迟用 FlashInfer 代理
- Radial Sec.5.2（明说 STA 用 FA3、自己用 FA2）；LServe Fig.10 注（MInference 解码未优化）；XAttention Sec.3.1（"严格用官方配置"）

### 10. "逐 token 模拟解码"评测协议
专门用来暴露逐出类方法"提前丢掉以后要用的 token"的缺陷。
- Quest Sec.4.2.3（首创），DuoAttention 附录 A.3（沿用 Quest，最后 50 token）

### 11. 与正交技术叠加，得到一个好记的标题数字
- DuoAttention + QServe W8A4 → 单 A100 上 3.3M token（Fig.12）
- Quest + AWQ → 端到端 2.23×（Fig.10b；FP16 只有 1.74×）
- LServe 建在 QServe 上；Jet-Nemotron 在 Orin/3090 上测（Table 15）

### 12. sink / 窗口 / 预算尺寸的敏感性扫描
StreamingLLM Table 2/6、DuoAttention Fig.13(2)(3)、StreamingVLM Table 5/8、LServe Table 6（复用间隔）、Radial 附录 Table B–D（warmup 步数、dense 层数）。

### 13. 训练-推理一致性对照
- LongLoRA Table 6（稀疏测 vs 全注意力测两种协议）
- StreamingVLM Table 2（ReKV 套在微调模型上胜率 0.00）、Table 7（重叠 vs 非重叠）
- StreamingLLM Sec.3.3（预训练加 sink token，把推理现象设计进训练）
- Radial Sec.4.3 / Fig.8(a)（静态掩码可训练，LoRA+Radial ≥ 全参）

### 14. 自建评测集
StreamingLLM StreamEval（Fig.8）、Long-RL LongVideo-Reason-eval（Table 2）、StreamingVLM Inf-Streams-Eval（Table 1）。三者都被设计成本方法的目标场景。

## 二、本组特有的套路（相对一般 ML 论文）

1. **机制谱系复用**：attention sink（StreamingLLM）被 DuoAttention 的 streaming 头、LServe、StreamingVLM、Radial（附加 sink）连续继承；DuoAttention 的头分类和 Quest 的页选择被 LServe 组合；LongLoRA 被 Radial 作为基线、被 Long-RL 系列引用。每篇新论文的动机常是"上一篇的方法在新约束下失效"（如 LServe Fig.6：Quest 在大页下失效）。
2. **从 GPU 执行模型推导哪种稀疏能兑现加速**：LServe Fig.3（warp lockstep，只有整块跳过才有效）、Radial 128×128 块、XAttention 块稀疏、Jet-Nemotron Key Finding 4（解码受内存带宽限制，KV cache 决定吞吐）。精度设计受硬件可兑现性约束，这是本组相对纯算法论文的显著特征。
3. **按 MHA / GQA 分开报数**：DuoAttention（2.55× vs 1.67×）、LServe（Llama-2 vs Llama-3）、Jet-Nemotron（KV 组数进入理论上界）。
4. **用超长上下文作为可记忆的能力边界**：4M token（StreamingLLM）、3.3M token/单卡（Duo）、100K/单机（LongLoRA）、3,600 帧/单节点（Long-RL）、2 小时以上视频（StreamingVLM）。
5. **理论界只在近两年出现**：Radial 给了复杂度和误差界（附录 A），Jet 给了加速上界；早期论文（StreamingLLM、Quest）只做复杂度算术。

## 三、本组反复出现的薄弱点

1. **无方差、无多 seed**：10/10 篇的主要结果都没有误差棒。Long-RL 的 NeurIPS checklist 第 7 项答 [Yes]（"提供统计显著性"），但正文没有。差距落在噪声量级的结论（Radial 的 Vision Reward 差 0.001–0.006，Long-RL AIME/自建集、LServe AIME 30 题）不可靠。
2. **调参或搜索用的数据与评测集重叠**：XAttention 的逐头阈值 DP 没说在哪个数据上评估性能，随后在 RULER 上报告并证明其有效（Table 9）；Jet-Nemotron 用 MMLU/math/retrieval 准确率搜索层放置，再用同样的基准报告；DuoAttention 用合成 passkey 识别头，再用 NIAH 评测（同分布）。
3. **标题加速数字取最有利的配置**：kernel 级而非端到端（Quest 7.03× vs 端到端 1.74×；XAttention 13.5× 只是注意力部分、无 TTFT）；最大 batch 吞吐而非延迟（Jet 47×：各模型 batch 8 vs 128）；MHA 而非 GQA（Duo 2.55×）；最长上下文（LServe 64K 时 1.09×；XAttention 8K 时 <1；Jet 4K prefill 1.14×）。
4. **"近似比精确更好"的结果没有解释**：XAttention 32K RULER 90.75 vs Full 84.17；DuoAttention、LServe 部分任务高于 dense。都被当作"鲁棒性"呈现，没有排除噪声。
5. **模型/评委同源或模型打分**：StreamingVLM 用 GPT-5 清洗数据又用 GPT-5 当评委；Radial 用 Vision Reward；LongLoRA LEval 用 GPT-4 胜率。没有人类评测或评委一致性检验。
6. **广度窄**：Quest 只有两个 7B（同为 Llama-2 系）；XAttention 每个模态一个模型；LongLoRA 只有 Llama2；StreamingVLM 单底座单领域；Long-RL 的消融只在 1.5B 上做。
7. **精度对照只有 dense、缺同预算竞品**：LServe 的精度只和 Dense 比（Table 2–4）；Radial 长视频缺"SVG+LoRA"。
8. **主张超出实验**：StreamingLLM 的"infinite"（附录 C/D 自己承认只是"缓存内可用"）；Long-RL 的 3,600 帧 / 8,192 帧；Jet 的"PostNAS 结论可迁移到从头预训练"；XAttention 的"pooling 预测不准"（文中找不到对应消融）；LServe 的"所需预算是常数"（256K 时要翻倍）。
9. **文本/表格不一致**（读到的）：DuoAttention"Section 13"应为 Fig.13；Long-RL Table 4/5 正文引用互换并残留"grey/red line"；Jet-Nemotron 4B 的层号在正文与附录 Table 9 中不同、Table 13 列的是 Llama3.2-1B 而主表是 3B；StreamingVLM 引用了不存在的 Limitations 节、Table 7 两行 Livecc 列数值完全相同；Radial 附录 Table C 标题"4×"与表内"117 frames"矛盾。
10. **自建评测集偏向本法**：StreamEval 的答案永远在 20 行前；LongVideo-Reason-eval 与训练数据同一管线；Inf-Streams-Eval 与训练数据同为体育解说。

## 四、一句话总结
本组的说服力主要来自三件事：用干预实验把注意力现象变成因果事实；同预算比精度、同精度比效率的双向对照；实测加速与理论上界对账。最常见的漏洞是没有统计、标题数字取最有利配置，以及调参/搜索数据与评测集不分开。
