# SpArch: Efficient Architecture for Sparse Matrix Multiplication

## 0. 元信息
- HPCA 2020（arXiv 2002.08947）；Zhekai Zhang*, Hanrui Wang*, Song Han, William J. Dally
- 类型：硬件（SpGEMM 专用加速器；非 DL 专属，面向通用稀疏线性代数）
- 实现层级：C++ cycle-accurate 仿真器（性能）；**只有 array merger** 写了 Verilog 并用 Synopsys DC 在 TSMC 40nm 综合，翻转率由 xsim 用仿真器导出的真实数据得到 → SAIF → DC 估功耗；浮点乘加单元的面积功耗取自文献 [30]，按仿真器统计的运算次数折算；SRAM/FIFO 用 CACTI；DRAM 能耗按 [31][32] 并采用与 OuterSPACE 相同的 42.6 GB/s/W。**无布局布线、无流片。**
- 全文读完（正文+参考文献）。Fig.11/12/14/17 的图内数据被 pdftotext 打散（含 Numbers 表格残留），只引用正文中明确出现的数字。

## 1. 核心主张
- 主：外积 SpGEMM 输入复用好、输出复用差；SpArch 通过"流水化乘-合并 + 矩阵压缩(condensing) + Huffman 树调度 + 行预取"同时优化输入与输出复用，DRAM 访问比 OuterSPACE 少 2.8×，速度 4×、能效 6×（Abstract, Sec.III-B）。
- 次：(a) 比 MKL/cuSPARSE/CUSP/Armadillo 快 19×/18×/17×/1285×；(b) 每一项技术的收益可用解析模型解释（Sec.III-C, Eq.2–7）；(c) 对矩阵密度变化更稳（Fig.14）。

## 2. 动机
- 痛点：OuterSPACE 只达到理论峰值的 10.4%（Sec.I，引自 OuterSPACE 自身数据）；原因是部分积矩阵需写回 DRAM。Fig.1 用"内积—输入复用差/输出复用好；外积—相反"的二维对照图把问题框成"联合优化输入和输出复用"。
- 关键 insight：(1) 不共享行索引的列可在乘之前合并（便宜地合并左矩阵替代昂贵地合并乘积）；(2) 合并顺序对 DRAM 流量的影响等价于 Huffman 树加权路径长度最小化——**跨领域嫁接**（编码理论 → 调度）；(3) 右矩阵访问顺序可提前得知 → 可做近 Bélády 最优替换。
- 闭合度高：每一项技术都是为了修补前一项带来的副作用（流水化导致多轮合并 → condensing；condensing 破坏右矩阵复用 → 行预取）。Fig.2 明确把第一步写成"5.7× 变慢"。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 优于 OuterSPACE | 20 个 SuiteSparse/SNAP 真实矩阵（与 OuterSPACE 相同数据集） | Fig.11/12, Table II/III | OuterSPACE（32nm, 87mm², 12.39W） | 加速、能效、带宽利用率 68.6% vs 48.3% |
| 优于 CPU/GPU/移动 CPU | 同上 | Fig.11/12 | MKL(i7-5930K)、cuSPARSE、CUSP(TITAN Xp)、Armadillo(A53) | 加速、能效 |
| 每项技术贡献 | 逐步加回 | Fig.2, Fig.16 | 自身变体 | 5.7× 变慢 → ×8.8 → ×1.5 → ×1.8，合计 4.2× |
| 收益可解释 | 解析估算 DRAM 访问量 13.9M → 2.5M → 1.5M → 0.88M | Sec.III-C | 与实测加速对照 | 估算 vs 实测（8.8× vs 5.5×，作者解释差异来自启动开销摊薄） |
| 接近屋顶 | roofline | Fig.15 | OuterSPACE | 10.4 vs 2.5 GFLOPS，屋顶 23.9 |
| 对密度鲁棒 | 合成 rMAT，密度 6e-3 → 5e-5 | Fig.14 | MKL | SpArch 降 2.7×，MKL 降 5.9× |
| 参数选择 | 预取缓冲行数/行宽、比较器阵列大小、look-ahead FIFO、合并树层数 | Fig.17/18 | 自身 | GFLOPS、DRAM 访问 |

## 4. 实验设计手法
- 基线：最强 ASIC（OuterSPACE）+ 各平台最优库（MKL、cuSPARSE、CUSP、Armadillo）。为公平：所有基线用 double；CPU/GPU 只计核心 SpGEMM 函数时间（剔除内存分配与传输）；功耗为动态功耗（总功耗减空闲）（Sec.III-A）。数据集刻意与 OuterSPACE 相同（Sec.III-B）。
- 规模：20 个真实矩阵 + 一组合成 rMAT 密度扫描。
- 效率测量：仿真器周期；面积功耗为部分综合 + 部分查表。
- 消融：Fig.16 逐步累加，并**保留了负收益的第一步**（流水化合并单独使用变慢 5.7×），说明它是后续优化的前提——这是诚实且有说服力的 roadmap。
- 分析：Sec.III-C 用概率模型推出每步的 DRAM 访问量，与实测加速逐一对照，这是本组少见的"解析模型验证消融"。roofline（Fig.15）、面积/功耗分解（Fig.13，合并树占大头）、能耗分解（Table III：SpArch 4.95 nJ/FLOP vs OuterSPACE 0.89？——见下方审读）。
- 泛化：只有 SpGEMM（A×A 类型），没有 DL 负载实测，尽管引言列了压缩 DNN 作为应用。
- 统计：无。
- 系统贡献：无开源声明（正文未见代码链接）。
- 局限：未讨论。

## 5. 对抗式审读
- **工艺不同未归一化**：SpArch 40nm vs OuterSPACE 32nm，Table II 直接并列面积 28.49 vs 87 mm²、功耗 9.26 vs 12.39W，未做工艺换算。OuterSPACE 的性能数字来自何处（复现模拟还是其论文报告值）正文未明确交代——**读不到**。
- Table III 能耗/面积分解：pdftotext 后列对应关系不清（"Computation 3.19 0.26 ..."），我无法确定哪列属 SpArch、哪列属 OuterSPACE，不引用具体数字。
- 浮点单元面积功耗查表、只综合了 merger：总面积/功耗是拼装估算。
- 加速比几何平均 4×，但 Fig.11 中逐矩阵差异较大（图中数据难读清）。
- 引言以压缩 DNN 为首要应用，但没有任何 DNN 稀疏矩阵上的实验，应用面的主张**未直接验证**。

## 6. 可迁移手法
1. 多技术叠加时，按依赖顺序逐步加回，并**保留负收益的步骤**以说明"它是后续优化的使能条件"。**直接读到**（Fig.2/16）
2. 用一个简单的解析模型预测每一步的 DRAM 访问量，再与实测对照，用模型解释消融。**直接读到**（Sec.III-C）
3. 把调度问题归约到经典最优结构（Huffman 树、Bélády 替换），以"可证明近最优"代替调参。**直接读到**（Sec.II-C/D）
4. 与前作比较时选用前作的同一数据集，并对齐 DRAM 能耗模型（42.6 GB/s/W）。**直接读到**（Sec.III-B）
5. 用合成数据做单变量扫描（密度）展示鲁棒性，与真实数据集互补。**直接读到**（Fig.14）

## 7. 一句话
每项技术都被写成"修补上一项副作用"的链条，并用解析 DRAM 访问模型逐步对上实测加速，使消融本身成为证明。
