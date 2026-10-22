# G7 组综合：TinyML / 片上训练 / 点云 3D / 自动驾驶多传感器融合

覆盖 12 篇实读论文：mcunet、mcunetv2、mcunetv3、tinytl、tinyml-magazine、pockengine、litepose、pvcnn、spvnas、bevfusion、sparserefine、semalign。`tinyml` 为合集条目、txt 误下为 VILA，未纳入（见 tinyml.md）。

先说不利结论：本组"solid"的观感主要来自**真机效率测量 + 对基线的结构性控制**，而不是统计严谨性——12 篇里只有 4 篇报告过任何方差，且至少 7 篇存在摘要/正文/表格之间的数字不一致或头条数字取自不同配置。

---

## 一、跨论文共同套路（≥2 篇出现）

### 1. "常用指标 vs 真瓶颈"的反直觉对照，作为动机的第一张图
用一对模型展示：常用指标（参数、FLOPs/MACs）大幅改善，但真正卡住部署的量（峰值激活、训练激活、实测延迟）几乎不变甚至更差。
- mcunet Figure 1：MbV2 参数比 ResNet-18 小 4.6×，峰值激活大 1.8×
- tinytl Figure 1 右：MbV2-1.4 参数比 ResNet-50 小 4.3×，训练激活只小 1.1×
- tinyml-magazine Figure 2：合并上两者（参数 4.2× 小，推理峰值 2.3× 大，训练 1.1× 小）
- spvnas §4.1：MinkowskiNet 通道缩 4×/8×，MACs 降到 7.5G/1.9G，延迟只从 105→96ms
- litepose Figure 8：LitePose-L MACs 更高但延迟与 Scaled-HigherHRNet 相近
- pvcnn Figure 2(b)：点方法"有效计算"只占 12–45%

### 2. 用 profiling/逐部件分解把瓶颈定位到"一小块"，再做局部手术
- mcunetv2 Figure 1 + 附录 Figure 11：峰值内存只由前 5 块决定（4 个骨干上重复）
- mcunet Figure 11：MbV2 单块峰值激活是平均的 2.2×
- mcunetv3 Figure 11(a)：逐层激活代价 vs 权重代价（前段激活贵、后段权重贵、中段便宜）
- bevfusion §III-B：BEV pooling 占运行时 >80%（>500ms vs 其余 ~100ms）
- pvcnn Figure 2(b)：不规则访存 + 动态核占 55–88% 运行时
- sparserefine Table 6：组件级 MACs/延迟分解

### 3. "Roadmap"/瀑布式逐步累加消融
- mcunet Figure 5（代码生成→专用 im2col→融合→展开→tiling；in-place depthwise）
- mcunetv3 Figure 1（TF/PyTorch → TTE → +QAS → +稀疏更新 → +重排，对数轴标倍数和章节号）
- litepose Table 3（核大小→融合头→超网→搜索→蒸馏，两个规模上重复）
- bevfusion Figure 3(c)（LSS → 预计算 1.9× → 区间归约 22.1×）
- tinytl Table 2（Bias → LiteResidual → L+B → @320）

### 4. 把"系统贡献"和"算法贡献"拆成因子网格
- mcunet Table 2（库 {CMSIS-NN, TinyEngine} × 模型 {S-MbV2, S-Proxyless, TinyNAS}）与 Figure 13（同一进化搜索在弱库上天花板更低）
- mcunetv2 §4.2（同一搜索空间与超网，只切换 per-layer / per-patch 推理）
- spvnas Figure 7 + 附录 Table 6/7（同宽度旧模块 vs 新模块；新模块手工 vs 搜索，四个成本档）
- mcunetv3 Figure 10(a)（稀疏更新 vs +算子重排）

### 5. 对基线"给最优待遇"或"拉到同一实现"
- mcunet §4.2 脚注：基线按宽度×分辨率复合缩放，训练帕累托前沿上所有可行点取最好
- spvnas §3.3：MinkowskiNet 升级到同一哈希表加速实现后再比
- sparserefine §4.1：全部基线在 MMSeg v1.0.0 下复现；Mask2Former 报 3 次均值
- spvnas §5.2：复现 SECOND 且高于原文，再"只替换模块"
- litepose：自建 Scaled-HigherHRNet 作为同计算量的多分支代表
- pvcnn：重实现 3D-UNet；bevfusion：重实现 PointPainting/MVP（Table I ∗）；所有开源方法同一 RTX3090 重测

### 6. 精度–效率帕累托图（常常同时对代理指标和真机延迟各画一张）
- mcunet Figure 9（VWW/GSC：精度-延迟、精度-SRAM）
- mcunetv2 Figure 7；litepose Figure 9（MACs + 三个平台延迟，四联图）
- pvcnn Figure 4/7；spvnas Figure 7；tinytl Figure 3（每数据集一张）；mcunetv3 Figure 9

### 7. 等精度比效率 / 等延迟比精度
- mcunetv3 Figure 10 注："稀疏更新选与全更新同精度的配置"
- sparserefine Table 3(b)：PointRend 输入调到 704×1408 以对齐延迟
- mcunet Table 3（5FPS/10FPS 约束下比精度）、Figure 8（同 ~70% 精度比 SRAM/Flash）
- pvcnn Table 1/4：按精度排序插入多个宽度版本

### 8. 解析值与真机实测"双轨"报告
- mcunetv2 §4（analytic profiling vs on-device profiling；Fig 5 vs Fig 6）
- mcunetv3 §3.1（analytic extra memory vs STM32F746 实测）
- 反例：tinytl 只有理论内存（§4.1 自承）；tinyml-magazine Table III 补了统一实测

### 9. 子群/条件分组分析作为机制证据（"如果机制成立，增益应集中在 X 子集"）
- spvnas Table 5 + Figure 6（小物体类、点分支激活比例）
- bevfusion Table IV + Figure 4（雨天、夜间、小/远物体、1/4/16 线 LiDAR）
- sparserefine Table 8（小类 person/rider/pole）
- pvcnn Figure 6、litepose Figure 5（定性：细小/离散部分）

### 10. 可视化"搜出来的结构长什么样"，把 NAS 黑盒翻译成规则
- mcunet Figure 12（SRAM 大→分辨率高，Flash 大→宽度大）
- mcunetv2 Figure 9（patch 段小核、大扩张与大核不同时出现）
- mcunetv3 Figure 11(b)、pockengine Figure 6（块内第一个 1×1 卷积 / attention+FFN 第一层最值得更新）
- spvnas §6.2（上采样段计算占比 77% → 47–63%）

### 11. 搜索有效性的随机对照
- mcunet Table 6（随机搜索空间 74.7±1.9）与 Figure 13（evolution vs random）
- spvnas Figure 8（ES vs RS 曲线；20 个随机模型重训）
- mcunetv3 附录 Figure 13（进化 vs 随机）

### 12. 用"自家上一代方法在新设定下的失效"立新题
- spvnas Table 1：PVCNN 在大场景上 39.0 mIoU 或 35.6 秒/帧
- mcunetv2 附录 Figure 11：专为 MCU 设计的 MCUNet 也内存失衡
- mcunetv3 §1：MCUNet/TinyTL 理论节省无法在现有框架中兑现（TinyTL 只给理论值）

### 13. 可插拔性证明：在已有骨干/检测器里"只替换一层"
- pvcnn（PointNet→PVCNN，PointNet++→PVCNN++，F-PointNet 替换）
- spvnas（SECOND 中替换稀疏卷积）
- sparserefine（5 个冻结的分割模型上外挂）

---

## 二、本组特有套路

1. **把 OOM 当作结果写进表/图**：mcunet Figure 4、mcunetv2 Table 2（OOM 划线）、tinyml-magazine Table III、mcunetv3 Figure 10(c)、pockengine Table 4（"-"）。在 MCU 场景下，"能跑"本身是二值证据。
2. **SRAM / Flash / 峰值内存作为一等指标，与精度并列**（mcunet 系列、tinytl、mcunetv3），而不是 FLOPs。
3. **以硬件能耗/带宽常数作为第一性论证**：pvcnn Figure 1（DRAM 640pJ vs MAC 3pJ；30 vs 668 GB/s）；tinytl §1（DRAM 比 SRAM 贵两个量级）。
4. **"系统让算法选择变得可行"叙事**：bevfusion（kernel 让 BEV 融合可用）、sparserefine Table 7（稀疏后端兑现稀疏度）、mcunet（TinyEngine 扩大 NAS 可行空间）、mcunetv3/pockengine（编译器让稀疏更新产生实测节省）。
5. **跨平台广度矩阵**：pockengine（7 个平台，MCU 到 Jetson Orin 到 Snapdragon DSP）、litepose（Nano/Snapdragon/Pi）、pvcnn（三款 Jetson）。
6. **自动驾驶论文的"传感器条件分组"**：bevfusion Table IV/Figure 4 的天气、光照、LiDAR 线数稀疏化是本组 3D/AD 线独有的鲁棒性设计。

---

## 三、本组反复出现的薄弱点

1. **头条数字不一致或来自不同配置**（≥7 篇）
   - spvnas："+3.3 mIoU"与"8× 计算/3× 加速"分属两个模型
   - pockengine：对 TF 加速 15×/11×/11.2×/13–21×；"7.9×""5.6×"在实验表中找不到出处；Llama 块数 32 vs 24
   - mcunetv2 / tinyml-magazine：检测增益 16.7 vs 16.9
   - tinyml-magazine：149KB/2077× vs mcunetv3 的 141KB/2300×
   - litepose：摘要 +14.0 vs 表 +13.0
   - mcunet：Table 2 与 Table 3 同约束 S-MbV2+CMSIS 35.2 vs 39.7
   - pvcnn：摘要 7×/10× 与表题 5.5×/3×、8×/3× 分属不同子集
2. **跨论文搬数字/换条件后并表**
   - mcunetv2 Table 3：MCUNet 基线 51.4% 来自 V1 的 YOLOv2，V2 用 YOLOv3
   - tinytl Table 3：FT-Full 行引自文献
   - semalign Table I：三个基线全部引自 LCCNet 论文
   - pockengine Table 2：MCUNet 行与 mcunetv3 Table 1 逐项相同，但配置标签不同
3. **加速倍数混入实现/框架差异或外推值**
   - pvcnn：基线来自各自官方实现
   - pockengine：对 Python 框架的加速与稀疏 BP 混在一起；MCU 上 TFLM 为外推延迟
   - mcunetv3 Figure 10(c)：TFLM 全更新 OOM，按算子平均速度外推
   - mcunet Figure 9：基线跑 TFLM、MCUNet 跑 TinyEngine
4. **几乎没有方差**：只有 pockengine（Table 2/3 ±std）、mcunetv3（附录 B，3 个 seed）、sparserefine（仅 Mask2Former 3 次）、spvnas（附录 Table 8 检测 3 次）报告过。tinytl 说"五次平均"但不给 std。大量结论建立在 0.1–1.0 级差异上（pvcnn 86.2 vs 86.1，bevfusion val NDS +0.1）。
5. **在报告集上调超参/阈值**：sparserefine α 按"追平高分辨率基线"选；mcunetv3 每次运行大量调 LR 取最好（附录 C）。spvnas 在验证集附录里重跑搜索，是少见的正确处理。
6. **"边缘"动机，但只在桌面 GPU 上测**：spvnas（1080Ti）、bevfusion（RTX3090，8.4 FPS）、sparserefine（RTX3090）。pvcnn、litepose、pockengine 做了边缘设备，可作对照。
7. **理论/模拟替代实测**：tinytl 全部理论内存；mcunetv3 精度在 GPU 上模拟（batch 128 近似 batch 1，VWW 上差 2.5%）。
8. **标题/摘要主张没有对应实验**：bevfusion 的"multi-task"无联合训练实验、本版无组件消融；pockengine §4 承诺的图优化结果节不存在；semalign 说损失与平移误差相关但只报告了旋转。
9. **评价指标与优化目标循环**：semalign Table III 用自己优化的损失评价泛化。
10. **PDF 残留草稿痕迹**（不影响结论，但说明发表流程匆忙）：mcunetv2 Figure 7、mcunetv3 多处、pockengine 多处 TODO 与 "#VALUE!"。

---

## 四、要点（给汇总用）

1. 本组"solid"的第一来源是**真机测量**（MCU 上的 SRAM/Flash/OOM、Jetson/手机延迟），且常把解析值与实测值并列；把 OOM 写成结果是 TinyML 线的标志性证据。
2. 动机几乎都以一张"常用指标 vs 真瓶颈"的反直觉对照图开场（参数↓但激活↑；MACs↓但延迟不变），随后用逐块/逐算子 profiling 把瓶颈收窄到一小块，方法只针对这一小块。
3. 协同设计类论文用**因子网格**（库×模型、模块×搜索、per-layer×per-patch）拆开系统与算法贡献，这是比端到端数字更难反驳的证据形式。
4. 基线处理有两种"加分"做法：给基线最优缩放/同一实现（mcunet、spvnas、sparserefine），以及等精度或等延迟对比（mcunetv3、sparserefine）。
5. 机制证据主要靠**子群分析**（小物体、恶劣天气、稀疏 LiDAR、逐类 IoU）与**结构可视化**，而非反事实干预；只有 sparserefine 系统地给出了 oracle 上界，litepose 的"渐进收缩"是少见的连续反事实桥接。
6. 统计严谨性弱：方差几乎缺席，多处结论落在噪声量级内；阈值/学习率有在报告集上调的迹象。
7. 数字一致性是本组最大的系统性风险：头条倍数常来自不同模型/配置拼接，跨论文复用旧数字时检测器或配置会悄悄变化（YOLOv2→v3、"后两块"→"稀疏方案"）。
8. 3D/AD 线（pvcnn→spvnas→bevfusion→sparserefine）呈现"上一篇的局限 = 下一篇的动机"的连续链条，并都在桌面 GPU 上测效率；TinyML 线（mcunet→v2→v3→pockengine）同样自我接续，且每一代都把上一代的数字搬进新表。
