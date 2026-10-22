# SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

## 0. 元信息
- arXiv 2509.24695v2（2025-10）。NVIDIA/HKU/MIT/清华/北大/KAUST。
- 类型：算法-系统协同（线性注意力视频 DiT + 恒定显存的 block 线性注意力 KV cache + 视频 DC-AE + NVFP4 部署 + 数据管线）。
- 文本状况：全文含附录读完。Fig.1/3/6 曲线（训练 loss、延迟）的数值在文本中大多丢失，消融结论只能按文字理解；定性结果大量放在项目主页（App.C 明说 "Please refer to our project link"）。

## 1. 核心主张
- 主：2B 的 SANA-Video 在 VBench 上与 Wan2.1-1.3B/14B、Open-Sora-2.0 等相当，同时显著更快（480p 60s vs Wan2.1-14B 484s，Tab.4；720p 36s vs Wan2.1-14B 1897s，Tab.2），训练成本仅 64×H100×12 天（"MovieGen 的 1%"）。
- 次：(a) 线性注意力 + RoPE(ReLU(x)) + 分母去 RoPE 解决数值不稳（Eq.2, Fig.3）；(b) 因果线性注意力的累积状态构成 O(D²) 恒定显存 KV cache，支撑分钟级视频（Tab.1, Eq.3, Fig.1c）；(c) 单调递增 SNR 采样器 + 改进 self-forcing 用于长视频（Fig.6d, Tab.5）；(d) DCAE-V（F32T4C32）在扰动下比 Wan VAE 更鲁棒（Tab.7）；(e) NVFP4 在 5090 上 71s→29s（Fig.7）。

## 2. 动机是怎么立住的
- 痛点用具体数字：Wan 14B 生成 720p 5 秒需处理 >75K token、H100 上 32 分钟（§1）；Fig.1b/c/d 三张图分别展示 720p 延迟、显存随视频长度增长（因果全注意力 OOM vs block 线性注意力恒定 7.2GB）、延迟随长度增长。Fig.1c 是把"恒定显存"主张可视化的关键图。
- 关键观察：线性注意力图比 softmax 稠密、缺少局部性（Fig.3a，与 Wan 的注意力图并列对比）→ 引入 RoPE（放在 ReLU 之后）和时间卷积补局部性。可视化 → 设计的链条存在，但只有定性注意力图。
- 理论动机：Tab.1 用复杂度表比较因果全注意力 / 局部注意力 / 线性注意力的显存与计算；这是"为什么线性注意力天然适配 AR 长视频"的核心论证，属于推导而非测量。
- 720p 时线性注意力自身也慢 2.3×、Wan2.1-1.3B 慢 4×（§3.4）→ 引出 DCAE-V。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比对象 | 指标 |
|---|---|---|---|---|
| 480p T2V/I2V 质量相当、最快 | 主表 | Tab.4 | MAGI-1, Step-Video, CogVideoX1.5, SkyReels-V2, Open-Sora-2.0, Wan2.1-14B/1.3B；I2V 加 HunyuanVideo-I2V | VBench Total/Quality/Semantic(I2V)；H100 BF16 bs1 延迟（各模型默认步数） |
| 720p 快且质量相当 | 720p 表 | Tab.2 | Wan2.1-14B/1.3B, Wan2.2-5B | 延迟、VBench |
| DCAE-V 重建可比 | 重建表 | Tab.3 | Wan2.1/2.2-VAE, LTX-VAE | PSNR/SSIM/LPIPS（Panda-70M 192p） |
| DCAE-V 对噪声更鲁棒 | 加噪重建 | Tab.7 | 同上 | ε=0/0.1/0.2 下 PSNR 等 |
| RoPE、时间卷积有效 | 训练 loss 曲线 | Fig.6a/b | 无 RoPE / 无 conv | 训练 loss |
| 线性注意力加速随分辨率增长 | 延迟剖面 | Fig.6c | 全注意力 | 延迟（480p 2×，720p 4×） |
| 分母去 RoPE 稳定 | loss 曲线 | Fig.3b | 分母含 RoPE | QK sum / loss |
| 单调 SNR 采样器 | 定性 | Fig.6d | 随机时间步 | 可视化一致性 |
| 长视频质量 | AR 方法对比 | Tab.5 | CausVid, SkyReels-V2, Self-Forcing | VBench |
| 恒定显存 | 显存 vs 长度 | Fig.1c | 因果全注意力 | GB |
| NVFP4 2.4× | 延迟柱状 | Fig.7 | BF16 | 5090 延迟；质量"indistinguishable"**无数值** |
| 1% MovieGen 成本 | — | §1 | — | **未直接验证**（无 MovieGen 成本出处对照表） |
| 分钟级长视频一致性 | — | Fig.12 定性 | — | **未定量**（Tab.5 用的是 VBench，并非长视频指标） |

## 4. 实验设计手法
- 基线：覆盖当时主流开源视频模型（含 14B/30B 大模型与 1.3B 小模型）；延迟同一 H100 BF16 自测、"各模型默认步数"（Tab.4 标题明说"for a fair comparison"）；VBench 分数来源未说明是否自测。Speedup 列以最慢的 Open-Sora-2.0/Wan-14B 为 1.0×。
- 规模：只有一个 2B 模型；两个分辨率（480p/720p）；T2V、I2V、长视频、三个世界模型下游（机器人/自驾/Minecraft，App.E，仅定性）。
- 效率测量：真实 H100 与 RTX 5090 端到端延迟；显存随长度曲线（Fig.1c）；复杂度表（Tab.1）。
- 精度–效率联合：Tab.2/Tab.4 同表列延迟与 VBench；无 Pareto 图。
- 消融：**几乎全部以训练 loss 曲线呈现**（Fig.6a/b, Fig.3b），没有对应的 VBench 数字；采样器消融为定性（Fig.6d）。SFT 影响仅定性（Fig.14）。
- 分析性实验：注意力图可视化（Fig.3a）；VAE 加噪鲁棒性实验（Tab.7）是一个有意思的"代理指标"——用"潜变量扰动后的重建"预测"哪个 VAE 更适合小扩散模型"。
- 泛化：下游世界模型任务仅定性。
- 统计：无。
- 系统贡献：NVFP4 + SVDQuant 部署；项目主页；（代码开源声明未在文本中明确）。
- 局限：无专门局限节。

## 5. 对抗式审读
- **加速倍数口径混乱**：摘要"16× faster"、引言"13× faster than Wan2.1 for 720p"、§3.4"53× over Wan2.1-14B"、Tab.4"8.0×"——同一篇里四种倍数，分母不同（Wan2.1-1.3B 720p 400/36≈11×，SkyReels 568/36≈16×）。读者无法从表中复算出摘要的 16×。
- 消融只看训练 loss：训练 loss 低不等于生成质量好（尤其是加 RoPE 改变了模型容量/位置信息），这是本文证据链最薄弱处。
- VBench 差距在噪声范围：T2V Total 83.71 vs Wan2.1-1.3B 83.31 vs Wan-14B 83.69；SANA 的 Quality 分（84.35）实际低于所有 Wan 模型，优势全部来自 Semantic（81.35）。"competitive"成立，但"outperforming"依赖 Semantic 一项。
- 长视频：Tab.5 不敌 Self-Forcing（83.70 vs 84.31），且用的是短视频 VBench，没有测分钟级一致性/漂移——核心卖点"minute-long"缺定量证据。
- 延迟"默认步数"：不同模型默认步数不同（如 50 步 vs 更少），这既是公平也是不公平——比的是"开箱即用"而非算法效率。
- DCAE-V 鲁棒性实验只在 1000 个 192p 样本上，且该 VAE 只用于 720p，而 720p 的主表 Tab.2 只有 3 个对照。

## 6. 可迁移手法
1. 证明"恒定显存/线性复杂度"主张：先给复杂度表（Tab.1），再画显存随序列长度曲线并标出基线 OOM 点（Fig.1c）。**直接读到**
2. 把"用户感知瓶颈"量化成一个惊人数字放在引言首段（Wan 14B 720p 5s 需 32 分钟）。**直接读到**
3. 选 VAE 时用"潜空间加噪后的重建质量"作为下游扩散适配性的代理指标（Tab.7）。**直接读到**
4. 延迟对比采用"各模型默认推理设置"并在表头明示，以避免被指控压低基线步数（Tab.4）。**直接读到**
5. 反面：只用训练 loss 做架构消融，会被质疑与生成质量脱钩；需配至少一个生成指标。**我的归纳**

## 7. 一句话
用"复杂度表 + 显存随视频长度恒定 vs 基线 OOM"的图把线性注意力在长视频上的结构性优势讲得一目了然，再用 VBench 证明质量不掉；但加速倍数口径不一、消融只看训练 loss。
