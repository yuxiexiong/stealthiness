# G3 综合：高效图像/视频生成（SANA 系、DC-AE 系、自回归图像生成）

覆盖 11 篇：sana, sana-1-5, sana-sprint, sana-video, dc-ae, dc-ae-1-5, dc-ar, dc-gen, dc-videogen, hart, lpd。全部读完全文含附录。文本问题汇总见文末。

先说最不利的：本组 11 篇**没有一篇报告方差、多 seed 或置信区间**；至少 5 篇存在头条数字与正文/表格不一致（见 §3.1）；同一实验在两篇论文中出现了不同分数（dc-videogen vs dc-gen）。"solid"的观感主要来自实验的**广度与受控对照设计**，而不是统计严谨性。

---

## 1. 跨论文共同套路（出现 ≥2 次）

### 1.1 "只换一个组件"的受控替换表
固定下游骨干、训练配方、token 数等，只替换被研究的部件。
- dc-ae Tab.3：同一 DiT-XL/UViT-H/UViT-2B 下只换 AE（Flux-VAE/Asym-VAE/SD-VAE/DC-AE）。
- dc-ae-1-5 Tab.2/3：同 f/c 配置下 DC-AE vs DC-AE 1.5，跨 4 个骨干。
- dc-videogen Tab.2、dc-gen Tab.19：同一 Wan-1.3B + 同一迁移流程，只换视频 AE。
- lpd Tab.1/2/3："Raster Counterpart"——同 tokenizer、同架构、同配方，只改解码顺序。
- hart Tab.5（HART-VAR，去掉残差扩散）、dc-ar Tab.4（discrete-only）：去掉核心组件的同配方对照。
- dc-gen Tab.4/5："Tuned"——同数据同超参微调原模型，以剥离数据效应。
这是本组最有力、最可迁移的手法。

### 1.2 固定最终 token 数，比较"压缩放在哪里"
- sana Fig.3a：F8C16P4 / F16C32P2 / F32C32P1 压到同样 32×32 token。
- dc-ae Tab.1：SD-VAE-f8p8 / f16p4 / f32p2 vs DC-AE-f64p1，均 64 token。
- dc-videogen §3.1.1 直接引用该结论作为设计前提。
结论一致："压缩全交给 AE，DiT 用 patch=1"。

### 1.3 先量化一个机制量，再用同一量验证修复（诊断—修复闭环）
- dc-ae Fig.3a：嵌套模型（高压缩含低压缩为子网、潜变量总量相同）重建更差 → 归因优化困难 → Residual AE 修复后同图对比。
- dc-gen Fig.2a：逐层特征距离（未对齐 0.24→345.42，对齐后 0.02→12.51）。
- sana-sprint Eq.10 + Fig.3b：定位 c_noise=1000t 的 1000 倍放大 → 梯度范数曲线前后对比。
- sana §2.3 + Fig.6：Gemma 嵌入方差远大于 T5 → RMSNorm，无 norm 时 NaN。
- lpd Eq.3 + Fig.7：定义 Per-Token Attention，5 万张图、三尺寸统计局部性。
- sana-1-5 Eq.3 + Fig.5：block importance 跨三个模型对照。
- hart App.A.1 Fig.9（定性注意力 sink+local）。

### 1.4 先展示朴素方法失败（含崩溃/NaN 图）
- dc-gen Fig.4/5（模糊、复制输入、崩成噪声，三任务）；dc-videogen Fig.6b（20K 步后噪声）。
- sana-sprint Fig.3d（无 schedule transfer 发散）；sana Fig.6（无 norm NaN）；sana-1-5 Fig.7（cyclic/block 初始化 NaN）。
- dc-gen §3.4、dc-videogen §2 都引用 OpenSora-2.0 原文自述"video outputs appear blurry"作外部证据。

### 1.5 "重建好 ≠ 生成好"的显式反例
- sana Fig.3a（F8C16 重建最好但生成最差）；dc-ae-1-5 Fig.1b（通道越多 rFID 越好、gFID 越差，反向单调曲线）；hart §3.1 + Fig.7（分离解码器 rFID 相近但 gFID 21.98 vs 5.30）；dc-ae Fig.9（f64 在小模型上生成变差的 transition point）；sana-video Tab.7（用"加噪重建"做下游适配性代理）。
这个套路让作者既能用 rFID 立动机，又不被"只看重建"攻击。

### 1.6 加速倍数随分辨率/长度/批量扫描，展示"越大越赚"
- sana Tab.14（512→4K：44×→104×）；sana-video Fig.1c/d、Fig.6c（480p 2×→720p 4×）；dc-gen Fig.22（FLUX 1K 4.2×→4K 53.8×）；dc-videogen Tab.7（6.2×→14.8×；帧数 8.2×→14.2×）；lpd Fig.14（batch 扫到 OOM，标出内存受限→计算受限转折）。

### 1.7 用最慢的基线做分母 / 只计骨干的头条倍数
- sana Tab.7（Speedup 以 FLUX-dev=1.0×）；sana-video Tab.4（以 Open-Sora-2.0/Wan-14B=1.0×）；dc-gen 53.8×（分母是不原生支持 4K 的 FLUX 硬跑 218.81s）；dc-videogen 14.8×（2160p 原模型 375 分钟）。
- 只计 transformer：sana-sprint Fig.1（64.7× 按 transformer 延迟）；dc-gen §4、dc-videogen §4.1（明说仅骨干）。其中 dc-gen 用 Tab.16 补了骨干/AE 拆分，是较好的处理。

### 1.8 Roadmap / 瀑布式逐步叠加
- sana Fig.2（469s→41→24→21→9.6s，每步一个乘数）与 Tab.8/Tab.12（block 逐步替换）。sana-sprint Fig.1 的文本中残留了同一 roadmap 的文字片段（"+ Deep Compression AutoEncoder 41 …"），说明该图被跨论文复用。

### 1.9 低分辨率预训练 → 高分辨率微调，以 GPU 小时/收敛曲线量化
- dc-ar Tab.6/Fig.10（760 vs 1440 GPU 小时）；hart Fig.8（相对 PE 加速分辨率迁移）；sana §3.1（跳过 256 直接 512 起步）；sana-video §3.1（192p→480p coarse-to-fine）；dc-gen App.E "resolution lifting"；lpd App.A.3（256→512→1024）。

### 1.10 端侧/量化部署小节，把效率落到"用户秒数"
- sana §4 Tab.5（W8A8，笔记本 0.37s）；sana-video Fig.7（NVFP4，5090 71s→29s）；sana-sprint（4090 0.31s）；dc-gen Tab.17（与 SVDQuant/TaylorSeer/AnyFlow 叠加）。

### 1.11 训练曲线横向读"节省 X% 步数/成本"
- sana-1-5 Fig.2（GenEval 0.70 处读出 60%）；dc-ae-1-5 Fig.5b（6× fewer steps）；dc-gen Fig.5、dc-videogen Fig.6（相对 base model score 的恢复速度）；sana-video Fig.6a/b（只有 loss 曲线）。

### 1.12 跨论文复用基线数字表
- T2I：sana Tab.7 的基线行（LUMINA-Next…FLUX）原样出现在 sana-1-5 Tab.1、sana-sprint Tab.2、dc-gen Tab.1。
- 视频：sana-video Tab.4 的外部模型行出现在 dc-videogen Tab.3/4、dc-gen Tab.4。
效率高，但意味着质量数字大多不是同协议重跑。

### 1.13 承认适用边界
- dc-ae Fig.9（小模型 f64 更差）、dc-ae-1-5 Tab.6（c32 上无益）、lpd Fig.14（大 batch 只剩 ~3×）、dc-ar Tab.4（混合比离散慢 10%）、dc-gen App.D.2（视频人评胜出归因于"更亮、运动更大"）。

---

## 2. 本组特有的套路

1. **自家组件的谱系式迭代，上一版就是主对照**：DC-AE → DC-AE 1.5 → DC-Gen/DC-VideoGen；SANA → 1.5 → Sprint → Video；HART → DC-AR。每篇的核心消融对象常是组内前作（dc-ae-1-5 全文以 DC-AE 为对照；sana-sprint 的教师来自 sana-1-5 剪枝）。好处是控制变量干净；坏处是对外部最强方法的对比偏弱。
2. **后训练迁移范式**（dc-gen、dc-videogen、sana-sprint、sana-1-5 剪枝）：不从头训练，只改接口层/少量步数，并用"GPU 天"对比从头训练成本（dc-gen Tab.15、dc-videogen Fig.2 "230×"）。
3. **T2I 指标三件套**：MJHQ-30K FID/CLIP + GenEval + DPG（sana、sana-1-5、sana-sprint、hart、dc-ar、dc-gen），视频统一 VBench。GenEval 逐渐成为头条指标（sana-1-5 明说"we particularly emphasize GenEval"）。
4. **"效率"定义为 token 数削减**：几乎所有加速都来自减少 token（AE 压缩、线性注意力、步数/并行解码），并用 MACs + 实测双报（hart Tab.3 指出实测 5.6× 与 MACs 5.8× 吻合，dc-ae 用 TensorRT 测推理）。
5. **把方法的分解本身可视化**：hart Fig.3（离散 + 残差 = 图）、dc-ae-1-5 Fig.3（前 16/32/64 通道逐级重建）、dc-videogen Fig.3（因果/非因果/chunk-causal 同图对比）、lpd Fig.8（四种顺序逐步填充）。
6. **人类评测极少**：只有 dc-gen 做了系统的 AMT 成对评测（含 Equal 选项、三领域）。其余全部依赖自动指标。

---

## 3. 本组反复出现的薄弱点

### 3.1 头条数字自相矛盾 / 前后版本不一致
- sana-sprint：摘要 1 步 7.59/0.74，Tab.2 为 7.69/0.76（1.6B）或 7.04/0.72（0.6B），结论又写 7.04/0.72。
- sana-video：16×（摘要）、13×（引言）、53×（§3.4）、8×（Tab.4）。
- sana-1-5：Tab.1 标 BF16、正文写 FP16；Tab.1 与 Tab.2 中 Flux-dev/SD3 的 GenEval 不同。
- lpd：LPD-L 参数 111M（Tab.4）vs 337M（正文）；τ/ρ 含义在 App.B.3 与算法中相反。
- **dc-videogen vs dc-gen**：同一 Wan-2.1-T2V-14B 迁移实验，84.83 vs 83.97；1.3B 84.63 vs 83.87；同配置换 AE 表 84.48 vs 83.48（延迟完全相同）。前者据此称"superior"，后者加 Tuned 对照后优势只剩 0.03。
- sana（arXiv v3）残留大量 "TODO"/"Done" 排版标记。

### 3.2 头条改进的归因混合
- hart：31%（7.85→5.38）基线是官方 VAR 量化器；只变残差扩散时为 6.1%（5.73→5.38）。
- dc-ae-1-5："4× faster"来自 f64 vs f32（DC-AE 已有），新方法贡献的是同 f64 下 gFID 3.20→2.18。
- dc-ae：FID 3.55→3.01 只在 w/ CFG 成立，w/o CFG 13.96 > 11.04。
- sana-sprint：表中"教师"行是 SANA 1.0，实际教师由 SANA-1.5 4.8B 剪枝而来。

### 3.3 在评测指标上调参 / 评判器与评测器同源
- sana-1-5：VILA-Judge 与 SFT 数据都用 GenEval 工具包标注，然后在 GenEval 上 best-of-2048 得 0.96；SFT 后 DPG、FID 反而变差。
- sana-sprint：推理时间步、两组噪声分布超参都直接按 MJHQ FID/CLIP 选（Fig.6/7/8）。
- lpd：评测时以 0.1 间隔扫 CFG 取最优 FID。

### 3.4 训练数据与评测分布重合未讨论
- hart、dc-ar、lpd（T2I）都用 JourneyDB / 内部 MidJourney 风格合成数据训练，而主评测 MJHQ-30K 是 Midjourney 图像；三篇都在 MJHQ FID 上取得领先而未讨论分布匹配。

### 3.5 回避最强一类竞品
- dc-ar 明确"不比较 VAR 类"（App.A.6），而 VAR/HART 正是高效 AR T2I 的直接竞品。
- lpd 的"至少快 3.4×"只对"并行化 AR"成立；表中 VAR-d24/d30 同质量更快，以"扁平 token 兼容性"排除，但该兼容性收益未被实验验证。

### 3.6 消融证据只有训练 loss 或单点
- sana-video 的 RoPE、时间卷积消融只有 loss 曲线；sana-1-5 "删最后两层"、QK-Norm 只有文字；hart 的 80% token 丢弃、kernel 融合只有文字；dc-gen Tab.6/7 只在 512px FLUX 单点做。

### 3.7 无统计
- 11/11 无方差；大量结论建立在 0.01–0.1 量级的差异上（sana Tab.4 FID 6.12 vs 6.13；sana-1-5 Tab.3 0.672 vs 0.664；lpd 2.10 vs 2.12；dc-ar GenEval 0.69 vs Show-o 0.68）。

### 3.8 局限讨论薄弱
- 只有 dc-videogen（App.A.8 一句）、sana（结论一句）、sana-1-5（Impact Statement）、dc-ae-1-5（Tab.6 边界）有形式上的局限；其余无。

---

## 4. 给"如何让论文显得 solid"的提炼（标注档位）
1. 每个核心组件配一个"只换它"的同配方对照，并在主表并列（§1.1）。**直接读到**（6 篇）
2. 动机图同时是消融图：先量化机制量，再在同一图上展示修复（§1.3）。**直接读到**（7 篇）
3. 先展示 naive 方案的失败图，再上方法（§1.4）。**直接读到**（5 篇）
4. 预先承认"代理指标 ≠ 目标指标"并给反例，把潜在攻击变成自己的贡献（§1.5）。**直接读到**（4–5 篇）
5. 加速主张做规模扫描，并标出优势消失的区间（§1.6, §1.13）。**直接读到**
6. 本组的反面教训：头条数字需与表格逐字一致、分母需明示、改进需按单变量归因、评判器不能与评测器同源、后训练需有同数据对照。**我的归纳**

---

## 5. 文本完整性说明
- 全部 11 篇读完正文与附录；参考文献部分只扫读。
- dc-ar：正文中混入大量 DC-AE 1.5 图注文字（"w/o Structured Latent Space / 16 channels…"），疑似 PDF 隐藏图层；Fig.6 与 Fig.11 的引用编号在正文/附录互相错指。不影响主要数字。
- sana：arXiv v3 含排版 TODO 标记；Fig.3/6/8 曲线数值错位，只能读出量级。
- 多篇双栏论文（dc-ae-1-5、sana-1-5、sana-sprint）段落交错、图内数值散落，所有曲线类图只能读出标注点；笔记中引用的曲线数字均来自文本中可读到的标注值，未从曲线估读。
- dc-gen 的 arXiv 版本日期为 2026-09（v3），与 dc-videogen（2025-09 v1）方法重叠，二者数字差异已在两篇笔记中标出。
