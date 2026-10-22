# NAAS: Neural Accelerator Architecture Search

## 0. 元信息
- arXiv 2105.13258（2021）；Yujun Lin*, Mengtian Yang*, Song Han。正文未写会议名；页脚 IEEE ISBN 978-1-6654-3274-0/21 对应 DAC 2021（**由 ISBN 推断，未在正文中核实**）。短文（约 4 页正文）。
- 类型：算法（设计空间探索/自动化搜索：加速器架构 × 编译映射 × 神经网络联合搜索）
- 实现层级：**纯解析模型**——硬件性能/能耗由 MAESTRO 估算（Fig.1 "HW Perf. Estimation (MAESTRO)"）；无 RTL、无综合、无流片。神经网络精度来自 Once-For-All 已训练子网。
- 全文读完（正文+参考文献）。Table III（与 NASAIC 比较）被 pdftotext 打散，行列对应不完全确定。

## 1. 核心主张
- 主：不只搜索数值型尺寸参数，而把 PE 连接方式（并行维度/阵列维数）和编译映射（循环顺序、分块）一起纳入搜索，在相同 #PE 与片上存储约束下显著降低 EDP；再与 OFA NAS 结合，在 Eyeriss 资源下比 Eyeriss 跑 ResNet-50 EDP 降 4.4×、ImageNet top-1 高 2.7%（Abstract）。
- 次：(a) "importance-based" 编码把非数值的顺序/选择问题转为数值优化，比索引编码好 1.4×–7.4×（Fig.9）；(b) 比仅搜尺寸好 1.4×–3.5×（Fig.8）；(c) 比 NHAS、NASAIC 分别好 3.0×、1.9× EDP（Sec.III-C）；(d) 搜索成本比 NASAIC 低 120× 以上（Table IV）。

## 2. 动机
- 设计空间规模论证：Table I 列出搜索空间；Sec.I 给出数量级——EdgeTPU 资源下至少 10¹¹ 硬件候选 × 每层 10¹⁷ 映射候选，ResNet-50 联合空间 10⁸⁶¹，而 NASAIC 只有 10⁴ 硬件候选。
- Table II 用定性矩阵说明"神经网络参数 × 加速器参数"的相关性因加速器而异（NVDLA 的 #rows 与输入通道相关、Eyeriss 的 #rows 与卷积核大小相关）。
- 无 profiling、无瓶颈测量；动机是"前人只搜尺寸、忽略连接与映射"的空间缺口。因果链：空间更大 → 需要可优化的编码 + 样本高效的进化策略（CMA-ES）→ 搜索成本低 → 才能嵌入 NAS。较闭合，但"连接与映射是关键缺口"这一前提的证据在结果中（Fig.8）而非动机中。

## 3. 主张 → 证据映射
| 主张 | 实验 | 位置 | 对比 | 指标 |
|---|---|---|---|---|
| 同资源下优于人工设计 | 大模型组（VGG16/ResNet50/UNet）与轻量组（MobileNetV2/SqueezeNet/MNasNet），按资源约束分别搜 | Fig.5 | EdgeTPU、NVDLA-1024、Eyeriss、NVDLA-256、ShiDianNao（同 #PE/SRAM/带宽） | 加速、节能（几何平均） |
| 单网络专用搜索更好 | 每网络单独搜 | Fig.6 | 同上 | 加速、节能 |
| 搜到的架构多样 | 案例 | Fig.7 | — | 定性（2D K-X'、2D C-X'、3D C-K-X'） |
| 搜连接+映射 > 只搜尺寸 | VGG/MobileNetV2 × EdgeTPU/NVDLA-1024 资源 | Fig.8 | Architectural Sizing Only（[11] 风格） | EDP 降低 |
| 编码方式有效 | 硬件/映射编码 2×2 组合 | Fig.9 | 索引编码 | EDP 降低 1.4×→7.4× |
| 进化优于随机 | 搜索曲线 | Fig.4 | 随机搜索 | 群体平均 EDP |
| 加 NAS 更好 | ResNet-50 空间 × Eyeriss 资源 | Fig.10 | Eyeriss、NHAS | top-1 vs 归一化 EDP |
| 优于 NASAIC | CIFAR-10 上 NASAIC 搜得的网络 | Table III | NASAIC | 延迟、能耗、EDP（1.88×） |
| 搜索成本低 | 估算 | Table IV | NASAIC、NHAS | GPU 天、AWS 成本、CO₂ |

## 4. 实验设计手法
- 基线：五个知名加速器的**资源包络**作为约束（"在 X 的 #PE/SRAM/带宽内搜"），这是 DSE 论文对标人工设计的标准做法。搜索类基线：仅尺寸搜索、NHAS、NASAIC、随机搜索。基线加速器的映射是否也经过同等搜索，正文**未交代**——这直接影响"同资源下 2.6×–4.4×"的含义。
- 规模：6 个 CNN（2 组）、5 种资源约束；NAS 部分仅 ResNet-50 空间 + ImageNet；NASAIC 比较在 CIFAR-10。
- 效率测量：全部 MAESTRO 解析估算，EDP 为奖励。无 RTL 或硅验证。
- 精度–效率联合：Fig.10 精度 vs 归一化 EDP（Pareto 风格），三个点（NHAS on Eyeriss、加速器-编译器协同、加 NN 的三者协同）。
- 消融：搜索空间消融（Fig.8：去掉连接与映射）、编码消融（Fig.9：2×2）、搜索算法（Fig.4：vs 随机）。
- 分析：Fig.7 搜得架构的可视化，说明结果"超越数值调参"。
- 泛化：多资源档、多网络；只有 CNN，无 Transformer。
- 统计：无多 seed（进化搜索有随机性，但未报告方差或多次运行）。
- 系统贡献：项目页 tinyml.mit.edu；未声明开源。
- 局限：无。

## 5. 对抗式审读
- **最薄弱：所有硬件数字来自 MAESTRO 解析模型**，没有 RTL/综合验证；搜索得到的 3D 阵列、非常规并行维度在真实硬件上的面积/布线/控制开销不在模型中，而对比对象（Eyeriss 等）是真实存在的设计。"rivals the human design Eyeriss by 4.4×"实为"模型中的 Eyeriss 资源包络"。
- **数字前后不一致**：Abstract 说 vs Eyeriss EDP 降 4.4×；Sec.III-C 说加 NAS 后"4.88× EDP improvement in total"；Fig.10 上标出 3.01× 与 1.47×。引言说"2.6×, 4.4× speedup and 2.1×, 1.4× energy savings ... compared to Eyeriss, NVDLA"，而 Sec.III-B 中 2.6× 对应 EdgeTPU、4.4×/2.1× 对应 Eyeriss、1.4× 对应 NVDLA-256——引言的对应关系与正文不符。
- 与 NASAIC 的比较：NAAS "3.75× latency improvement with double energy cost"，EDP 1.88×；NASAIC 搜索成本是"从 CIFAR-10 乐观外推"（Table IV 脚注），120× 的搜索成本优势依赖该外推。
- NAS 部分的 +2.7% top-1 来自 OFA 子网与 ResNet-50 基线的对比，精度提升主要归功于 OFA（前作），不是 NAAS 的硬件搜索。
- 无随机性控制（seed/方差），进化搜索结果的稳定性未知。

## 6. 可迁移手法
1. DSE/自动化设计论文以知名人工设计的资源包络（#PE、SRAM、带宽）为约束搜索，再与该设计比较，形成"同预算下更好"的叙事。**直接读到**（Sec.III-A, Fig.5）
2. 用"只搜尺寸 vs 全空间"的消融证明扩展搜索空间的价值。**直接读到**（Fig.8）
3. 把排序/选择类离散决策编码成"重要性分数 + 排序"，使连续优化器可用，并用 2×2 编码消融证明。**直接读到**（Fig.3, Fig.9）
4. 用 GPU 天、AWS 费用、CO₂ 的成本表把"搜索效率"量化为可比较数字。**直接读到**（Table IV）
5. 以精度 vs EDP 的二维图同时呈现精度与效率增益，避免单维对比。**直接读到**（Fig.10）

## 7. 一句话
在知名加速器的资源包络内搜索并用"只搜尺寸"消融证明连接与映射的价值——但全部建立在 MAESTRO 解析模型上，没有任何硬件实现验证。
