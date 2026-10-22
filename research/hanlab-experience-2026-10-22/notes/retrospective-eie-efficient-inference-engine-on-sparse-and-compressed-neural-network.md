# Retrospective: EIE: Efficient Inference Engine on Sparse and Compressed Neural Network

## 0. 元信息
- arXiv 2306.09552（2023，ISCA 25 年回顾类短文，2 页）；Han, Liu, Mao, Pu, Pedram, Horowitz, Dally
- 类型：综述/回顾（无实验）
- 全文读完（2 页 + 参考文献）。

## 1. 核心主张
- 主：EIE 开启了"在剪枝/稀疏/低精度模型上设计加速器"的设计空间，并影响了学术界与商用芯片（Sec.II）。
- 次：(a) EIE 的 W4A16 思路在 LLM 单 batch 解码（M×V、权重内存瓶颈）中"重生"（Sec.I，脚注列 GPTQ/AWQ/llama.cpp/MLC LLM）；(b) "高效 AI 计算第一原则是偷懒：避免冗余计算、快速拒绝、延迟工作"（Sec.IV）。

## 2. 动机怎么立
- 无新实验。影响力用两类证据：Fig.2 剪枝/稀疏论文数随年份增长曲线（1989–2021，标注 OBD、Deep Compression、EIE），以及后续工作清单（Cambricon-X、SCNN、Eyeriss v2、ExTensor、SIGMA、SAM）和商用芯片（NVDLA、NVIDIA Sparse Tensor Core 2:4、Samsung NPU、Ambarella CV22）。
- 关键论点："专用硬件可在 50% 密度下仍划算地做稀疏，而软件需要密度 <1% 才能抵消稀疏库开销"（Sec.I）——**无出处数字，未给证据**。

## 3. 主张 → 证据映射
| 主张 | 证据 | 位置 | 备注 |
|---|---|---|---|
| 引发研究浪潮 | 论文数曲线 | Fig.2 | 相关≠因果；曲线 2015 后上升也与深度学习整体增长同步，未归一化 |
| 影响商用芯片 | 列举 | Sec.II | 未给出这些芯片引用 EIE 的证据，**未直接验证** |
| W4A16 在 LLM 重生 | 列举 4-bit LLM 项目 | Sec.I 脚注 | 合理类比，同时承认这些用线性整数量化而非 K-means 码本 |
| 软件需 <1% 密度 | 无 | Sec.I | **未直接验证** |

## 4. 实验设计手法
无实验。可记录的写作手法：
- "Lessons" 一节自我批评（Sec.III）：EIE 难以用于向量处理器阵列；CSC 遍历开销非零（每 PE 一个 MAC 却带指针读取、稀疏矩阵访问、前导非零检测等开销）；权重与索引都是 4-bit → 50% 存储开销；只加速 FC 层；全部权重放 SRAM 不适合 10B–100B 参数的 LLM。
- 把后续工作（结构化 N:M 稀疏、负载均衡感知剪枝 ESE）作为对这些缺点的修复来叙述。
- "New opportunities"（Sec.IV）把实验室后续工作（ProxylessNAS、OFA、MCUNet、SpAtten、TSM、TorchSparse、PointAcc、SIGE）串成"偷懒"原则下的一条线。
- 局限性：有（Sec.III）。

## 5. 对抗式审读
- 影响力叙事主要靠自引与列举，Fig.2 没有对照（例如同期全部 DL 论文增长）来剥离总体趋势。
- "软件需 <1% 密度"是强数字主张却无引用；原 EIE 论文里只有"CPU/GPU 上压缩仅 3× 加速"。
- 对 EIE 原论文最受质疑的地方（45nm 综合 vs 28nm 实测硅片、batch=1 下对 GPU 不利的能效比较）**回顾中没有提**；自我批评只涉及架构适用范围，不涉及评估方法。
- "W4A16 重生"是事后追认：EIE 的 4-bit 是码本索引，与 GPTQ/AWQ 的线性整数量化机制不同，作者自己也承认。

## 6. 可迁移手法
1. 回顾/定位文章中，用"一条统一原则"（偷懒：避免、拒绝、延迟）串起一组看似分散的工作，以建立研究纲领的连续性。**直接读到**（Sec.IV）
2. 把自己旧工作的缺点写成具体工程开销（50% 索引存储、每 MAC 的控制开销），再指向后续修复方案，让批评落在"造出了什么"。**我的归纳**（Sec.III）
3. 用新兴负载（LLM batch=1 解码 = M×V）重新激活旧设计的相关性。**直接读到**（Sec.I）

## 7. 一句话
把 EIE 的 W4A16 + M×V 设定与 LLM 单 batch 解码对上号，从而把一篇 2016 年的 FC 层加速器重新放进当下主流问题里。
