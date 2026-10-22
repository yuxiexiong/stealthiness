# G8 组内综合：硬件加速器与稀疏计算系统

覆盖：eie、retrospective-eie、spatten、sparch、pointacc、torchsparse、torchsparsepp、flatformer、lego、naas。
实际精读正文 10 篇（其中 retrospective 是 2 页无实验短文）。TorchSparse++ 正文为协调者后补的 MICRO'23 版本（首版 txt 是 TorchSparse 的重复下载），已重读并据此更新本文件。下文"出现在哪几篇"只计实际读过的正文。

## 坏消息先说
1. **本组没有一篇有流片**。所有 ASIC 数字都是"周期仿真 + 部分 RTL 综合 + CACTI/外部单位能耗拼装"，NAAS 更是纯解析模型（MAESTRO）。而对比对象常是真实硅片或其论文报告值。
2. **工艺归一化不一致**：EIE 做了 28nm 投影（方法未给公式）；SpArch（40nm vs OuterSPACE 32nm）、PointAcc（40nm vs Mesorasi 16nm）不做归一化；LEGO 把 65nm 流片的 Eyeriss、16nm 投影的 NVDLA 与 28nm 综合结果放在一张表。
3. **对先前加速器的数字常是估算/假设**：EIE 对 DaDianNao 用峰值带宽估算；SpAtten 对 MNNFast 假设 ASIC 功耗是 FPGA 的 1/10、对 A3 用其论文报告加速比推算；SpArch 对 OuterSPACE、PointAcc 对 Mesorasi 的数字来源正文没交代清楚。
4. **系统类论文也有"投影式"硬件比较**：TorchSparse++ 把 GPU 实测延迟乘 2.2（频率×峰值 MAC）并假设放大版 PointAcc 线性加速，得出"达到 ASIC 56%"（Table 2），仅一个负载。

---

## A. 跨论文共同的实验设计套路（≥2 篇）

### A1. "仿真器 + 局部综合 + 查表"的评估栈
- EIE：C++ 周期仿真器（兼作 RTL 检查器）+ Verilog/DC 45nm + 单 PE 布局布线 + Cacti + PrimeTime PX（Sec.V）
- SpAtten：SpinalHDL→RTL + Verilator + Genus 40nm + CACTI + Ramulator；浮点单元借用 45nm 文献数据（Sec.V-A）
- SpArch：C++ 周期仿真 + 只有 merger 走 DC 40nm + 浮点单元查文献 [30] + CACTI + DRAM 单位能耗（Sec.III-A）
- PointAcc：Verilog + Genus 40nm + 仿真器（与 RTL 对拍）+ Ramulator + CACTI（Sec.5.1）
- LEGO：SpinalHDL→Verilog + DC 28nm + CACTI + 前端性能模拟器（Sec.VI-A）
- 反例（对照用）：TorchSparse++ 在 Table 2 把 PointAcc 放大到 128×128 后做投影比较——同一实验室的 GPU 系统论文与 ASIC 论文互相当对照物。
- 共同点：性能来自周期仿真，面积功耗来自"综合的那部分 + 查表的那部分"之和。**我的归纳**：这是 Han Lab 架构论文的标准栈，可信度主要靠"仿真器与 RTL 对拍"一句话支撑，没有给出对拍误差。

### A2. 通用平台全谱系对比 + 数量级加速
- EIE：i7 / Titan X / Tegra K1（Fig.6/7）
- SpAtten：TITAN Xp / Xeon / Jetson Nano / Raspberry Pi（Fig.14）
- SpArch：MKL / cuSPARSE / CUSP / ARM Armadillo（Fig.11/12）
- PointAcc：2080Ti / Xeon / TPU-v3 / Jetson NX / Nano / RPi（Fig.13/14）
- 摘要里都挂上这些两到四位数的倍数（24,000×、347×、1285×、176×）。

### A3. 基线测量协议写明：nvidia-smi + pcm-power + 功率计，动态功耗 = 总功耗 − 空闲功耗
- EIE（Sec.V，TK1 用功率计并假设 AC-DC/调压损耗）、SpAtten（Sec.V-A，另有 1000 次重复、去掉首尾各 15%）、SpArch（Sec.III-A，只计核心函数时间）
- 系统类论文的同类做法：FlatFormer（1,000 样本 × 50 次运行 + 10 次预热、剔除离群，Sec.5.2.1）；TorchSparse 与 TorchSparse++ **均未交代**重复次数/预热

### A4. 与先前专用加速器比时做"资源对齐"
- SpAtten：缩放出 SpAtten1/8，对齐 A3 的乘法器数（128）、工艺（40nm）、带宽（Table III）
- SpArch：与 OuterSPACE 用同一数据集、同一 DRAM 能耗模型 42.6 GB/s/W（Sec.III-B）
- PointAcc：PointAcc.Edge（16×16）对 Mesorasi（16×16），全尺寸对服务器平台（Table 3）
- LEGO：对 Gemmini 对齐 256 MAC/256KB/16GB/s；对 SODA 换到 FreePDK45；对 AutoSA 换到 U280 FPGA（Sec.VI-A）
- NAAS：在 EdgeTPU/NVDLA/Eyeriss/ShiDianNao 的 #PE/SRAM/带宽包络内搜索（Sec.III-A）
- EIE：28nm 投影后比 DaDianNao（Table V）
- 注意：对齐的是**资源量**，工艺归一化并不统一（见坏消息 2）。

### A5. 逐步累加的加速"路线图"（roadmap breakdown）
- SpAtten Fig.20：专用通路 22.1× → +token 1.1× → +head 1.1× → +top-k 3× → +静态量化 1.6× → +渐进量化 1.7×
- SpArch Fig.2/16：流水合并 **0.18×（慢 5.7×）** → condensing 8.8× → Huffman 1.5× → 预取 1.8×
- TorchSparse Table 3 / Fig.13：FP16 1.17× → 向量化+融合 → 局部性感知 2.86×；mapping 1.6×/1.5×/1.8×/1.1×
- FlatFormer Fig.8：FlashAttention 1.7× → QKV 打包 1.3× → FFN 融合 1.2× → drop residual 1.1×
- TorchSparse++ Fig.23：生成器（同参数下 1.1–1.2× 于 SpConv 2.3.5）→ +更多 split → +不排序 → +FoD
- 变体（逐项去除/替换而非累加）：PointAcc Fig.17/19/20、LEGO Fig.13/14
- **我的归纳**：最有说服力的两次是 SpAtten 与 SpArch——故意展示"单加某步收益很小甚至为负，直到配套部件加上才兑现"，把消融变成"软硬协同必要"的论证。

### A6. 一维设计空间扫描，在拐点处选参数并在图注写明理由
- EIE Fig.8（FIFO 深度 → 8）、Fig.9（SRAM 宽度 → 64-bit）、Fig.10（精度 → 16-bit）
- SpAtten Fig.19（top-k 并行度 → 16，K/V SRAM → 196KB）
- SpArch Fig.17/18（预取缓冲、比较器阵列、look-ahead FIFO、合并树层数）
- PointAcc Fig.18（缓存块大小 × 核大小 × 通道）
- FlatFormer Table 4–6（窗口形状、组大小、分辨率——模型侧的同类做法）

### A7. 面积/功耗分解
- EIE Table II（SRAM 占面积 93%、功耗 59%）、SpAtten Table II/Fig.13（DRAM 5.71W / 8.30W）、SpArch Fig.13（合并树为主）、PointAcc Fig.21（计算 69%、DRAM 23%）、LEGO Fig.12（buffer 86% 面积，PPU 2%/5%）

### A8. Roofline 证明"贴近上限且基线离屋顶很远"
- SpAtten Fig.18（1.61 TFLOPS vs GPU 0.02）、SpArch Fig.15（10.4 vs OuterSPACE 2.5 GFLOPS，屋顶 23.9）

### A9. 以 profiling 立动机："FLOPs 占比小、延迟占比大"或"数据搬运占大头"
- SpAtten Fig.2（注意力 ~10% FLOPs、>50% 延迟；73% 时间在数据搬运）
- PointAcc Fig.5/6（数据集密度、每点内存、mapping >50%、CPU+TPU 往返 60–90%）
- TorchSparse Fig.4（数据搬运 40–50%、GEMM 20–50%、mapping 15%）
- FlatFormer Fig.2–4（三类点云 Transformer 各自的慢因量化）
- EIE 用的是另一种：单位操作能耗表做数量级账（Table I）

### A10. 把自家算法也放到通用硬件上跑，以分离"算法收益"与"硬件收益"
- EIE：CPU/GPU 的 compressed 列（Fig.6/7），"压缩单靠 CPU/GPU 只有 3×"（Sec.VI-A）
- SpAtten：GPU 上实现 token 剪枝，BERT 最多 2.3×（Sec.V-B）
- PointAcc：mergesort 求交在 CPU/GPU 上**更慢**、在 PointAcc 上快 >10×；fetch-on-demand 在 GPU 上吃亏（Fig.17）
- **我的归纳**：PointAcc 的"同一算法在 GPU 上变慢、在专用硬件上变快"是这套手法最强的形态——是反事实对照，不只是附带对比。

### A11. 再加一层"模型协同设计"把系统级数字推高
- SpAtten：HAT 搜索 co-designed Transformer（Fig.16，1.9× 快、2.8× 小）
- PointAcc：Mini-MinkowskiUNet 对 Mesorasi 100×、+9.1% mIoU（Fig.16）
- NAAS：整合 OFA NAS，+2.7% top-1（Fig.10）
- 共同风险：混合了模型选择与硬件，系统级倍数不能归于硬件。

### A12. 跨平台/跨档位复测以排除替代解释
- TorchSparse：在无 FP16 tensor core 的 1080Ti 上仍 1.5×（Sec.5.2），Table 1 特化 vs 迁移的交叉
- TorchSparse++：Pascal/Turing/Ampere + Orin × FP16/TF32/FP32（Fig.14），推理与训练分开测（Fig.15）
- FlatFormer：A6000 + Jetson Orin（Fig.6）
- PointAcc：服务器档与边缘档两套配置（Table 3）

### A13. 用经典最优结构/理论推导给设计"背书"
- SpArch：Huffman 树（合并顺序）、Bélády 替换（预取）、解析 DRAM 访问模型逐步对照实测（Sec.II-C/D, III-C）
- SpAtten：softmax 导数推出量化误差 ∝ p(1−p)（Eq.2）
- TorchSparse：对称 offset 的 map 大小相等的推导（Sec.4.2.1）；FP16 理论 2× vs 实测（Sec.4.3.1）
- TorchSparse++：理想化上界实验（只调 tile 即超过同规模 cuBLAS，Fig.8）为缩减设计空间背书
- EIE：理论时间 vs 实际时间（Table IV）、节能四因子乘积（Sec.VI-B）

### A14. 实验室内部组件的复用与互相对比
- 零消除器（prefix-sum + log N 移位）在 SpArch（Fig.6）和 SpAtten（Fig.10）中重复出现；PointAcc 的 top-k 直接与 SpAtten 的 quick-select 对比（1.18×，Sec.4.1.4）；PointAcc → TorchSparse → FlatFormer → TorchSparse++ 是点云负载的连续线；TorchSparse++ 把前作 TorchSparse 列为被超越的基线（2.0–2.2×），并在 Sec.5.2 一句话声称换用其后端的 CenterPoint 在 Orin 上比 FlatFormer 快 1.5×——**后作反向削弱前作 FlatFormer"首个比稀疏卷积更快"的卖点**（FlatFormer 对比的是 SpConv v2.2.3 后端）；LEGO 把 NAAS 作为前作定位。**我的归纳**：这使每篇都能以前作为现成强基线或模块。

---

## B. 本组特有的套路（相对其他组）

1. **双层标题数字**：摘要同时给"对通用平台的数量级倍数"（162×、1285×、24,000×）和"对先前专用加速器的个位数倍数"（1.6×、4×、1.3×）。前者抓眼球，后者才是公平比较。出现在 EIE、SpAtten、SpArch、PointAcc。
2. **排序/合并/选择原语反复成为硬件核心**：SpArch 的比较器阵列合并器、SpAtten 的 quick-select top-k、PointAcc 的 ranking 统一 mapping。**我的归纳**：稀疏计算的不规则性被一再归约成"排序类问题"再做并行硬件。
3. **"以计算换规则性"是跨越 ASIC 与 GPU 系统的统一叙事**：TorchSparse 的自适应分组（多算 FLOPs 换 batch）、FlatFormer 的等大小分组（MACs 多 1.7× 却更快）、SpAtten 的渐进量化（多算一遍换少读 DRAM）、TorchSparse++ 保留不排序 implicit GEMM（冗余计算多 1.7× 仍端到端快 1.2×，Table 3）。
4. **资源包络对标**（NAAS、LEGO、SpAtten1/8）：在知名设计的资源预算内比较，而不是比绝对性能。
5. 系统类（TorchSparse、TorchSparse++、FlatFormer）用**真实硬件实测**，而 ASIC 类全部仿真——组内存在两种完全不同的证据等级，但写作上都用"×speedup"同一种语言呈现。
6. **用"代理指标失效"作为主张**：TorchSparse++ 用 Table 3 vs Table 4（排序版 kernel 更快、端到端更慢）推翻"计算量/DRAM/kernel 时间可代理端到端"；FlatFormer 用"MACs 多 1.7× 却更快"推翻 FLOPs 代理。两篇都把"一阶指标不可靠"写成正面贡献。**我的归纳**

---

## C. 本组反复出现的薄弱点

| 薄弱点 | 出现在 |
|---|---|
| 无流片；面积功耗为综合+查表拼装（部分模块甚至借用其他工艺的文献数据） | EIE、SpAtten（45nm FPU 数据用于 40nm）、SpArch（只综合 merger）、PointAcc、LEGO；NAAS 纯解析模型 |
| 工艺未归一化或口径混用（流片 vs 综合 vs 投影） | SpArch（40 vs 32nm）、PointAcc（40 vs 16nm）、LEGO Table III、EIE Table V（投影方法未给） |
| 先前加速器数字靠估算/假设/引用，而非同 benchmark 实跑 | EIE（DaDianNao 峰值带宽估算）、SpAtten（MNNFast 功耗假设、A3 引用加速比）、SpArch 与 PointAcc（来源未交代） |
| 对 GPU 的对比用 batch=1、利用率极低的设置，放大倍数 | EIE（batch=64 时 GPU dense 已可比，但能效图只给 batch=1）、SpAtten（GPU 0.01–0.02 TFLOPS） |
| 系统级比较混入模型变化，倍数不能归因于硬件 | PointAcc（100× vs Mesorasi）、NAAS（+2.7% 来自 OFA）、FlatFormer（SST 未享受同等系统优化）；TorchSparse++ 对 FlatFormer 的 1.5× 只有一句话、无表 |
| "精度无损"依赖逐任务调参，或干脆未报告 | SpAtten（每任务多组剪枝率+微调 2h，BERT-L SQuAD 允许 2%）、TorchSparse（FP16 精度未报）、TorchSparse++（训练"精度相同"只有一句声明）、PointAcc（定点影响未报） |
| 无方差、无多 seed、无置信区间 | 全组（仅 SpAtten、FlatFormer 有截尾平均/重复测量） |
| 自家方法享受逐负载穷举调优、基线用默认策略 | TorchSparse（约 1,000 配置网格搜索）、TorchSparse++（group-level 穷举，约 2 分钟） |
| 无独立局限性章节 | 除 EIE（Discussion 部分涉及）、retrospective（Lessons）外全组 |
| 数字前后不一致 | NAAS（Abstract 4.4× vs 正文 4.88×；引言平台对应关系与正文不符）；TorchSparse++（训练对 SpConv v2：Abstract 1.2–1.3× vs Fig.15 图注 1.16×；Conclusion "1.2–3.7×"；工程成本 "<1/10" vs "5% 代码行"） |
| 声称的泛化未验证 | EIE（1×1 conv/Winograd）、SpArch（DNN 应用）、TorchSparse（CPU 推理、多 GPU 训练）、FlatFormer（仅 Waymo） |

---

## D. 计数说明
- 本综合中的"出现在哪几篇"均来自我对各篇正文的直接阅读，位置见各篇笔记。
- 判断档位：A1–A13 的"出现"均为**直接读到**；A5、A10、A14 及 B2、B3、B6 中的"原则化"表述为**我的归纳**。
- TorchSparse++ 首页 Fig.1 拼贴图的图内文字被抽成约 200 行乱码，不影响正文阅读。
- 未覆盖：FlatFormer 正文引用的附录（txt 中没有）；各篇图内被 pdftotext 打散的逐项数字。
