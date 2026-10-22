# SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning

## 0. 元信息
- HPCA 2021（arXiv 2012.09852v3，2024 修订版，引言里加了 LLM/KV cache 字样）；Hanrui Wang, Zhekai Zhang, Song Han
- 类型：算法-架构协同（硬件）
- 实现层级：SpinalHDL → RTL，Verilator 周期仿真；HBM2 用 Ramulator；逻辑用 Cadence Genus 在 TSMC 40nm 综合；Softmax 的浮点 FMA/FPU 面积功耗取自文献 [17] 的 **45nm** 数据作为 40nm 上界估计；SRAM/FIFO 用 CACTI；HBM 能耗用 [19] 的单位能耗乘以 Ramulator 计数。**无布局布线、无流片。**
- 全文读完（正文+参考文献；无附录）。pdftotext 把 Fig.14/18/20/21 的图内数据与正文混排，部分图中数字无法逐条读清，下文只引用正文中明确出现的数字。

## 1. 核心主张
- 主：级联 token 剪枝 + 级联 head 剪枝 + 渐进量化，配合专用 top-k 引擎与流水线数据通路，可在"无精度损失"下把注意力的 DRAM 访问降 10.0×、计算降 2.1×，比 A3/MNNFast 快 1.6×/3.0×、比 TITAN Xp/Xeon 快 162×/347×（Abstract, Sec.V-B）。
- 次：(a) 与 A3/MNNFast 不同，可加速访存受限的生成式模型（GPT-2）而不仅 BERT（Table III）；(b) 量化误差与注意力分布"平坦度"相关（Fig.7 + Eq.2 推导）；(c) top-k 引擎是让剪枝兑现成速度的必要条件（Fig.20）；(d) 与模型协同设计可得 1.9× 快、2.8× 小的 Transformer（Fig.16）。

## 2. 动机
- Profiling：Fig.2 在 TITAN Xp / Xeon / Nano 上测 GPT-2 端到端延迟，注意力占 >50% 延迟但仅约 10% FLOPs；注意力内部 73% 时间花在 split heads、concat、reshape、transpose 等数据搬运。Table IV 再次给出：GPU 上 attention 14.4% FLOPs 却占 48.6% 延迟。
- 关键观察 1：语言冗余 → token 可剪（Fig.5 用列和示意）；观察 2：注意力最大概率越大，4-bit 量化误差越小（Fig.7 右散点），并用 softmax 导数给出 2p(1−p) 的上界证明（Sec.III-D, Eq.1–2）。这是"观察 + 理论解释"的双重立法。
- 因果链：数据搬运瓶颈 → 需减少 DRAM 访问 → A3/MNNFast 先取全部 QKV 再决定剪枝，所以不省 DRAM（Sec.I (i)）→ 剪枝决策须在取数之前（级联累积分数）。闭合较好。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 对 CPU/GPU 的数量级优势 | 30 个 benchmark（BERT-B/L 在 GLUE 9 项 + SQuAD v1.1/2.0；GPT-2-S/M 在 4 个 LM 数据集） | Fig.14 | TITAN Xp、Xeon、Nano、Raspberry Pi 4 | 加速比、能效（几何平均） |
| 优于 A3/MNNFast | SpAtten1/8 同乘法器数（128）、同 40nm、同带宽 | Table III | A3（引用其 1.73× 加速推算有效吞吐）、MNNFast（作者"复现模拟器"+假设 ASIC 功耗为 FPGA 的 1/10） | GOP/s, GOP/J, GOP/s/mm² |
| 各技术的贡献 | 逐步累加 breakdown | Fig.20 | GPU 基线 | 专用通路 22.1× → +token 1.1× → +head 1.1× → +top-k 3× → +静态量化 1.6× → +渐进量化 1.7× |
| 无精度损失 | 剪枝后微调约 2h，逐任务调剪枝率/位宽 | Sec.V-A, Fig.21 | 未剪枝模型 | 精度损失 vs 剪枝率曲线 |
| 端到端有效 | SpAtten-e2e（复用乘法器跑 FC） | Fig.15, Table IV | GPU/CPU | 35×/122×（8-bit FC） |
| 接近 roofline | 屋顶线 | Fig.18 | TITAN Xp | 1.61 TFLOPS（BERT）/0.43（GPT-2） |
| 设计参数合理 | top-k 并行度、K/V SRAM 大小扫描 | Fig.19 | 自身 | GFLOPS |
| top-k 引擎优于排序器 | 对比 Batcher 奇偶排序器（最坏情况选中位数，长度 1024） | Sec.IV-C | 全排序单元 | 1.4× 吞吐、3.5× 更低功耗 |
| 可解释性 | 可视化 | Fig.22/23 | — | 定性 |
| 协同设计模型 | HAT 搜索 | Fig.16/17 | vanilla Transformer 层数/维度缩放 | BLEU vs 延迟 |

"Given the same overall pruning ratio, ratios among layers/heads do not have a significant influence"（Sec.V-A）——**未给数据，未直接验证**。

## 4. 实验设计手法
- 基线：两个同期最强注意力加速器 A3（HPCA'20）、MNNFast（ISCA'19）；为公平专门**缩放出 SpAtten1/8**，对齐乘法器数、工艺、带宽（Table III）。CPU/GPU 用 PyTorch + cuDNN/MKL；并在 GPU 上**也实现了 token 剪枝**（topk+gather，BERT batch 模式最多 2.3×，Sec.V-B），堵"算法本身在 GPU 上也能加速"的攻击。
- 规模：4 个模型、30 个 benchmark；只到 GPT-2-Medium/BERT-Large。
- 效率测量：CPU/GPU 延迟重复 1000 次、去掉最大/最小各 15% 再平均；功耗测动态功耗（总功耗减空闲功耗）；Nano/RPi 用功率计（Sec.V-A）。测量细节交代得很细。SpAtten 侧为周期仿真 + 综合估算。BERT 输入长度取各任务 dev 平均长度；GPT-2 初始长度 992，生成 32 token。
- 精度–效率：Fig.21 剪枝率 vs 精度损失曲线（GPT-2-S PTB、BERT-B CoLA 各一条）。
- 消融：Fig.20 逐步累加 roadmap，并解释了"剪枝本身只给 1.1× 是因为 top-k 成为瓶颈"——这是让消融讲故事的典型手法。
- 分析：roofline（Fig.18）、面积/功耗分解（Fig.13, Table II：DRAM 5.71W 占 8.30W 的大头）、可视化（Fig.22/23）。
- 泛化：判别式 + 生成式两类；无 beam search 实验（口头说也适用）。
- 统计：延迟有截尾平均；精度无多 seed。
- 系统贡献：给出 GitHub 链接；无流片。
- 局限：未专门讨论。

## 5. 对抗式审读
- **"无精度损失"是逐任务调参得到**：每个任务试多组剪枝率/位宽并微调约 2 小时，BERT-Large SQuAD 允许 2% 损失（Sec.V-A）。硬件结果对应的是"每任务最优配置"，不是一个固定配置。
- **与 MNNFast 比较依赖假设**：MNNFast 原为 Zynq FPGA（10W），作者"乐观估计" ASIC 功耗为 1W，并用自己复现的模拟器得到吞吐；与 A3 比较则用 A3 论文报告的 1.73× 加速推算有效吞吐。两者都不是在同一 benchmark 上实际运行对方设计。
- 浮点单元用 45nm 数据作 40nm 上界、HBM 能耗用外部单位能耗——面积功耗是拼装估算。
- 对 GPU 的 162× 是"注意力层-only"，端到端（SpAtten-e2e）对 GPU 降为 35×/24×（Fig.15）。Abstract 主打的是前者。
- GPU 基线 batch=1 生成、GPU 利用率极低（roofline 上 0.01–0.02 TFLOPS），比较对 GPU 不利。
- head 剪枝仅贡献 1.1×，作为三大贡献之一分量偏弱。

## 6. 可迁移手法
1. 先 profiling 证明"FLOPs 占比小但延迟占比大"（attention 10% FLOPs / >50% 延迟），用此错配立动机。**直接读到**（Fig.2, Table IV）
2. 与同类加速器比较时，把自己的设计**缩放到对方的乘法器数/工艺/带宽**再比（SpAtten1/8）。**直接读到**（Table III）
3. 逐步累加的 speedup breakdown 中，故意展示"某技术单独加上收益很小，直到配套硬件模块加上才兑现"，以论证软硬协同的必要性。**直接读到**（Fig.20）
4. 对经验观察补一个简短的理论推导（softmax 导数上界），把"现象"升级为"机制"。**直接读到**（Eq.2, Fig.7）
5. 在通用硬件上也实现自家算法，量化"算法收益"与"硬件收益"的分界。**直接读到**（Sec.V-B GPU token 剪枝 2.3×）
6. 用 roofline 证明设计贴近理论上限，且说明 baseline 离屋顶很远。**直接读到**（Fig.18）

## 7. 一句话
用逐步累加的 speedup roadmap 证明"剪枝只有在专用 top-k 引擎下才变成速度"，从而把算法贡献和硬件贡献绑在一起。
