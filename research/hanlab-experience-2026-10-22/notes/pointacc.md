# PointAcc: Efficient Point Cloud Accelerator

## 0. 元信息
- MICRO 2021（arXiv 2110.07600）；Yujun Lin, Zhekai Zhang, Haotian Tang, Hanrui Wang, Song Han
- 类型：硬件（点云深度学习加速器；含算法-硬件协同的小模型 Mini-MinkowskiUNet）
- 实现层级：Verilog RTL + RTL 仿真验证；Cadence Genus 在 TSMC 40nm 综合；功耗用基准负载的完整翻转率标注；cycle-accurate 仿真器（与 Verilog 对拍）统计周期和 SRAM 读写；DRAM 用 Ramulator（能耗来自其命令 trace）；SRAM 能耗用 CACTI（Sec.5.1）。**无布局布线、无流片。**
- 全文读完（正文+参考文献；无附录）。Fig.13–16 的柱状数字被 pdftotext 打散，只引用正文中给出的平均值。

## 1. 核心主张
- 主：点云网络的瓶颈是 mapping 操作（建立输入-输出-权重映射）与 gather/scatter 数据搬运；PointAcc 把多种 mapping 操作统一为基于排序（ranking）的内核，并用可配置缓存 + 时间维层融合减少 DRAM 访问，比 RTX 2080Ti 快 3.7×、省能 22×（Abstract, Sec.5.2.1）。
- 次：(a) 比 Mesorasi 硬件快 1.3×、省能 11×；协同设计小网络后在 S3DIS 上快 100×、mIoU 高 9.1%（Sec.5.2.2, Fig.16）；(b) mergesort 求交比哈希表方案快 1.4× 且面积小 14×（Sec.4.1.1）；(c) top-k 比 SpAtten 的 quick-select 快 1.18×（Sec.4.1.4）；(d) 缓存降 DRAM 3.5–6.3×，层融合降 33%–64%（Fig.19/20）。

## 2. 动机
- 系统化 profiling（Sec.3）：Fig.5 左——5 个点云数据集密度 <10⁻²（室内）到 <10⁻⁴（室外），vs ImageNet 100%；Fig.5 右——每点特征内存可达 16KB，约为 CNN 的 100×；Fig.6——PointNet++ 类网络在通用硬件上 mapping 操作 >50% 时间，CPU+TPU 时数据往返占 60–90%，数据搬运在 CPU/GPU 上 >50%。
- 引言用 Fig.2 立"反常"：点云网络比 2D 投影 CNN 精度高、MACs 少 7×，GPU 上反而慢 1.3×；MinkowskiUNet 在 2080Ti 上 11.7 FPS vs ResNet50 840 FPS。
- 关键 insight：各种 mapping（FPS、kNN、ball query、kernel mapping）本质都是距离比较 → 可统一为 Max/TopK/MergeSort+求交（Fig.8/9）。另：点云稀疏是"物理"稀疏，不会膨胀，与 ReLU/剪枝稀疏不同 → 既有稀疏加速器不适用（Sec.3）。
- 因果链：profiling 的两个瓶颈（Bottleneck I/II）分别对应 MPU 与 MMU 两个模块，闭合好。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 优于服务器平台 | 8 个网络/5 个数据集/4 类任务（Table 2） | Fig.13 | RTX 2080Ti、Xeon 6130、Xeon+TPU-v3 | 3.7×/90×/53× 加速，22×/176×/210× 省能 |
| 优于边缘平台 | 同上，PointAcc.Edge（16×16，3.9mm²） | Fig.14 | Jetson Xavier NX、Nano、RPi 4B | 2.5×/9.8×/141×；7.8×/16×/127× |
| 优于 Mesorasi | PointNet++ 类 4 个网络 | Fig.15, Table 3 | Mesorasi-SW（Nano/RPi 上）、Mesorasi-HW（16nm） | 1.3× 加速、11× 省能 |
| 协同设计更强 | Mini-MinkowskiUNet on S3DIS | Fig.16 | Mesorasi + PointNet++SSG | 100× 延迟、+9.1% mIoU |
| ranking 统一 mapping 有效 | 算子级延迟分解 | Fig.17 左 | 哈希表（CPU/GPU 上）| mergesort 在 CPU/GPU 上更慢，在 PointAcc 上快 >10× |
| Fetch-on-Demand 有效 | 算子级 | Fig.17 右 | Gather-MatMul-Scatter | GPU 上反而因 MV 利用率低吃亏，PointAcc 上收益 |
| 缓存有效 | 层 DRAM 访问分布 | Fig.18/19 | 无缓存 | 3.5×–6.3× |
| 层融合有效 | | Fig.20 | 逐层 | 33%–64% |
| 整体瓶颈消除 | 延迟/能耗分解 | Fig.21 | CPU/GPU/TPU | MatMul 主导延迟；计算 69% 能耗、DRAM 23% |

## 4. 实验设计手法
- 基线：CPU/GPU 基线是作者自己的 PyTorch 实现（MKLDNN/cuDNN + 自写 OpenMP/CUDA mapping 内核），并声明比 MinkowskiEngine 快 2.7×（Sec.5.1）——先把软件基线做强，再比硬件。专用硬件只有 Mesorasi（唯一先前点云 DL 加速器）。
- 规模：8 个网络覆盖 Table 1 所有类别（PointNet++ 类、SparseConv 类、图类 DGCNN），5 个数据集从物体到室外场景；两档硬件配置（4096 PE/HBM2 vs 256 PE/DDR4）分别对标服务器与边缘。
- 效率测量：硬件侧仿真 + 综合；基线侧实测（测量方法细节较少，未写重复次数和功耗测量方式）。
- 精度–效率：只在 Fig.16 给 mIoU vs 归一化加速（与 Mesorasi 的协同设计比较）。
- 消融：分模块的"关闭/替换"对比（Fig.17 两个子图、Fig.19 有无缓存、Fig.20 有无融合），不是逐步累加 roadmap。
- 分析：反事实对照很有特色——**同一算法在 CPU/GPU 上更慢、在专用硬件上更快**（Fig.17），用来证明收益源自"算法 × 硬件"而非算法本身。缓存块大小 × 卷积核 × 通道的缺失率扫描（Fig.18）。
- 泛化：跨网络族、跨数据集、两档硬件规模。
- 统计：无。
- 系统贡献：有项目页 pointacc.mit.edu；正文未声明开源 RTL。
- 局限：未专门讨论。

## 5. 对抗式审读
- **工艺未归一化**：Mesorasi 为 16nm，PointAcc.Edge 为 40nm（Table 3）。这一不对齐对 PointAcc 不利，增加了"1.3× 加速、11× 省能"的可信度；但 Mesorasi-HW 的性能/能耗数字怎么得到（复现模拟器还是原文数字）正文未交代——**读不到**。
- **100× 对 Mesorasi 是"换模型"的比较**：PointAcc.Edge 跑 Mini-MinkowskiUNet，Mesorasi 跑 PointNet++SSG；这是系统级比较，混合了模型选择与硬件，100× 不能归于硬件。
- 服务器对比中 PointAcc 用 HBM2 256GB/s、4096 PE、8 TOPS 峰值，2080Ti 的峰值更高，但 2080Ti 跑稀疏点云利用率很低；对比合理但"3.7×"受软件基线质量影响大，基线是作者自写。
- 精度无损：硬件不改变网络，所以精度一致是隐含的；论文未报告定点量化（若有）对精度的影响——**未直接验证**。
- 基线测量细节（重复次数、功耗测法）比 SpAtten/SpArch 交代得少。

## 6. 可迁移手法
1. 从数据集统计（密度、每点内存）到平台 profiling（算子级延迟分解）分层立动机，并让每个瓶颈对应一个硬件模块。**直接读到**（Fig.5/6 → Sec.4）
2. 把一族看似不同的操作归约到一个共同原语（ranking/排序），从而一套硬件覆盖全部，并用"覆盖 Table 1 全部类别的 benchmark"证明通用性。**直接读到**（Fig.8, Table 1/2）
3. 用"同一算法在通用硬件上更慢、在专用硬件上更快"的反事实对照，证明软硬协同的必要性。**直接读到**（Fig.17）
4. 先把软件基线做到比现有最好库更快（2.7× MinkowskiEngine），再比专用硬件，堵"基线弱"的攻击。**直接读到**（Sec.5.1）
5. 同一架构出两档配置（服务器/边缘），分别与同档平台对比，避免跨档比较。**直接读到**（Table 3, Fig.13/14）
6. 对唯一先前加速器，补一个"协同设计模型"的系统级比较，展示对方架构无法支持更准的模型。**直接读到**（Fig.16）——注意这混合了模型与硬件。

## 7. 一句话
把各种点云 mapping 操作统一成排序原语，并用"同一算法在 GPU 上变慢、在 PointAcc 上变快"的对照证明收益来自软硬协同。
