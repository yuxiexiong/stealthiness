# 本地62份HanLab PDF逐篇阅读卡

日期：2026-10-17。按本地年份分组；不是62个互相独立的新贡献。所有正文与文件内技术附录完成全文提取阅读，书目页浏览；关键表选择性视觉复核。详尽原记录、物理页覆盖及SHA256见 [READING_LEDGER.json](READING_LEDGER.json)。缺失外置补充材料不在本次范围。

由主代理及3位协作代理分工完成；主代理核读汇总卡及直接影响最终判断的关键证据。本文件为论文报告值的分析，不是实验复现。

## 2022

### 01. GAN Compression: Efficient Architectures for Interactive Conditional GANs

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/GAN Compression- Efficient Architectures for Interactive Conditional GANs.pdf>) · [存档源](https://arxiv.org/pdf/2003.08936v4) · 24页全文文本 · 阅读者 `root`。

- **贡献：** 蒸馏、共享权重搜索/一次训练与Fast GAN Compression让条件GAN在较低MAC下维持生成质量。
- **关键对照：** p7使用官方PyTorch实现但改Pix2pix架构并重新训练；p17 GAN loss由vanilla改hinge且teacher/student/baseline一致；p18 Table10公开official/retrained/compressed。
- **关键证据：** CycleGAN官方FID71.84、重训61.53，正文用更强重训模型；GauGAN Cityscapes官方mIoU62.18、重训61.04，正文用更强官方模型。选择有证据的强版本，并非一律默认或一律本地改版。
- **限制：** 同FID协议在p8脚注另列；训练/搜索费用分开；COCO完整旧方案训练38天未做，报告Fast22天，不能补成实测完整原案。
- **当前含义：** 不必为维护Curfew回退弱LT；适配版身份与参数来源必须披露。当前没有官方与本地LT直测，不能假定改动均增强或两版相同。

### 02. Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/Lite Pose- Efficient Architecture Design for 2D Human Pose Estimation.pdf>) · [存档源](https://arxiv.org/pdf/2205.01271) · 11页全文文本 · 阅读者 `root`。

- **贡献：** 小算力姿态估计中缩掉高分辨率分支，采用单分支、大卷积核与融合head。
- **关键证据：** p3逐渐收缩；p5融合和3/5/7/9kernel；p8 NAS/蒸馏分解；p7三硬件实际速度。
- **限制：** 收缩时宽度变化，非完全等MAC；NAS预算额外；结论限小算力段。
- **当前含义：** 先证明toy中的特定瓶颈和设计必要，不能用后续搜索或规模掩盖核心未立住。

### 03. On-Device Training Under 256KB Memory

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/On-Device Training Under 256KB Memory.pdf>) · [存档源](https://arxiv.org/pdf/2206.15472) · 18页全文文本 · 阅读者 `design_audit`。

- **问题：** 256KBSRAM MCU真实量化训练难优化且backward存储超过budget；减少参数不自动省实际内存。
- **贡献：** QAS矫正梯度scale、基于贡献分析的稀疏layer/tensor更新、编译期autodiff/图剪枝与重排的TTE，真实STM32F746部署。
- **强与简单对照：** classifieronly/biasonly/lastk/fulltune、FP32/int8 SGD/Adam/LARS、不同channel选择、randomvs evolutionary、同accuracy挑方案比较实际memory/latency，analyticvsmeasured分开；3seed与GPU模拟/MCU差异披露。
- **打平与独立收益：** 8bit+QAS均accuracy73.5接近FP32 73.3，但真实memory可行；channelselection仅差.2%不强调机制；Fig9同accuracy sparseupdate实际资源优势。
- **限制：** 真实256KB通过并非全部accuracy主表都真MCU：大批结果GPU batch128模拟无momentum，平均近batch1但VWW有例外需gradacc；TF-Lite fullupdate OOM速度为投影，不能当实测；贡献可加近似、方案在Cars搜后迁移。
- **可迁移判断：** 真实终点与硬资源约束要先定；简单替代先比，不满足主终点或额外复杂性无收益的分支可舍弃。明确少数已发表失败记录可引用。
- **不能类比处：** 256KB可行vs不可行是清晰部署门槛，Curfew相对LT尚无同等硬资源差异；稀疏update不是样本防御gate，不能套机制。
- **页表锚点：** [{"page": 7, "item": "Table1", "finding": "int8+QAS73.5 vsfp32 73.3，无新增optimizer内存"}, {"page": 8, "item": "Fig9/10", "finding": "同accuracy挑sparse，analytic+实测memory和速度分开；TF-Lite全更新为OOM投影"}, {"page": 10, "item": "subchannel/limitations", "finding": "三种channel选择仅.2%差；只CNN视觉"}, {"page": 15, "item": "AppendixB/C", "finding": "3seedavgstd.07%；每run extensiveLR调参透明报告"}, {"page": 16, "item": "Table4/gradientacc", "finding": "GPUbatch模拟与MCUbatch1，VWW例外"}, {"page": 18, "item": "AppendixG", "finding": "明确失败路线：量化LoRA不省整权重SRAM/难更新；lighterconv损失transfer并加复杂性，弃用"}]

### 04. Network Augmentation for Tiny Deep Learning

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/Network Augmentation for Tiny Deep Learning.pdf>) · [存档源](https://arxiv.org/pdf/2110.08890) · 13页全文文本 · 阅读者 `root`。

- **贡献：** 小网络主要欠拟合，临时共享扩宽辅助子网提供梯度，推理零额外结构。
- **关键证据：** p7正则/延长训练/KD；p13减弱正则、weight decay、小batch等对照；p8大小模型训练/验证轨迹。
- **限制：** 大ResNet50有-0.3pp；实际训练16.7%开销来自具体pipeline，不可泛化成所有环境。
- **当前含义：** 先排除简单解释，保留失败边界；不能拿一个弱默认对手就宣称阶段机制必要。

### 05. QOC: Quantum On-Chip Training with Parameter Shift and Gradient Pruning

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/QOC- Quantum On-Chip Training with Parameter Shift and Gradient Pruning.pdf>) · [存档源](https://arxiv.org/pdf/2202.13239) · 6页全文文本 · 阅读者 `root`。

- **贡献：** 设备端parameter-shift梯度与按幅度概率剪枝，缓解噪声占比并减电路调用。
- **关键证据：** p5相同电路预算学习曲线；p6概率/确定剪枝与优化器对照；p3–4完整补读解决输出截断。
- **限制：** 四qubit任务；超出测得qubit范围的runtime为外推；不证明一般量子优势。
- **当前含义：** 时机/筛选策略需真实预算与最终质量收益，不能只呈代理噪声曲线。

### 06. QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/QuantumNAS- Noise-Adaptive Search for Robust Quantum Circuits.pdf>) · [存档源](https://arxiv.org/pdf/2107.10845) · 19页全文文本 · 阅读者 `github_audit`。

- **问题：** 无噪声最优电路未必能在带噪量子硬件上工作；逐候选训练及搜索电路/映射成本高。
- **贡献：** 共享参数 SuperCircuit、受限/前部采样、噪声适配的电路与映射联合搜索，以及角度剪枝。
- **强与简单对照：** 无噪声搜索、3 个随机候选取优、同参数量手工电路、噪声适配映射/Sabre、UCCSD；电路/映射单独搜索及采样消融。
- **最终终点与证据：** 物理页6先验证代理排名相关性，页8–11再看带噪模拟与真实量子设备精度；不以代理相关性代替最终指标。
- **限制：** 页9明确从5个剪枝比例中报告最高测得精度；300张测试图、设备校准漂移和UCCSD硬件只执行前1000门等限制均须保留。不能把本文写成完全无事后选择的理想规范。
- **当前含义：** 机制指标只能是线索，门控仍须落到独立结果；同条件简单方案打平时，不可由代理相关性宣称增量。
- **页表锚点：** [[6, "Figs.9–10"], [8, "IV.A"], [9, "IV.B"], [11, "Figs.19–20"], [14, "Artifact"]]

### 07. QuantumNAT: Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/QuantumNAT- Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization.pdf>) · [存档源](https://arxiv.org/pdf/2110.11331) · 10页全文文本 · 阅读者 `root`。

- **贡献：** 量子测量归一化、真实硬件噪声注入和测量量化，分别处理不同误差。
- **关键证据：** p5–8单组件、替代噪声、硬件错配及编译器对照；p7理论含输入相关偏置，另需小方差近似。
- **限制：** 依赖真实设备噪声profile；有失败配置；并非严格完美归一化，也不是普遍量子加速证明。
- **当前含义：** 自适应要有分布/硬件变化理由及最终收益，不能先假设固定策略一定失效。

### 08. QuEst: Graph Transformer for Quantum Circuit Reliability Estimation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/QuEst- Graph Transformer for Quantum Circuit Reliability Estimation.pdf>) · [存档源](https://arxiv.org/pdf/2210.16724) · 10页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 能否廉价预测量子电路受噪声影响的可靠性。
- **贡献：** 20k模拟/约25k真实数据；PST代理fidelity；gate图+设备噪声graphtransformer预测。
- **强与简单对照：** 三层简单NN；去全局/逐特征、1/2/3层、shots。
- **打平与独立收益：** RMSE平均优于NN约.02；层数接近时选2层，不把复杂结构都说必要。
- **限制：** PST代理在1400模拟电路相关.993不是普遍fidelity等价；最大10qubit、每backend单独训练；若干特征去掉略好；未证OOD。
- **可迁移判断：** 先验证lossgate proxy与真正能力/风险对应；无收益组件不认领必要性。
- **不能类比处：** 量子噪声和昂贵模拟与投毒SFT不同；200x预测速度不含数据采集/训练。
- **页表锚点：** ["p3 §3.1", "p7 Tables1–2和70/20/10split", "p8 Tables3–5/Fig10；抽取部分叠字未认领精确格值", "p9–10参考"]

### 09. Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models.pdf>) · [存档源](https://papers.neurips.cc/paper_files/paper/2022/file/b9603de9e49d0838e53b6c9cf9d06556-Paper-Conference.pdf) · 16页全文文本 · 阅读者 `github_audit`。

- **问题：** 编辑图像后仅少量空间位置变化，反复稠密计算冗余；稀疏理论MAC下降未必变成实际加速。
- **贡献：** 编辑掩码、原图激活缓存、归一化复用近似与分块gather/scatter内核。
- **强与简单对照：** Patch blending、公平缓存归一化的Crop、剪枝/小GauGAN/GAN compression，以及逐项内核融合消融。
- **最终终点与证据：** 质量近似保持时，单CPU/GPU实际延迟改善；页9表3朴素稀疏7.6倍MAC下降只有1.6倍加速，完整融合才2.9倍。
- **限制：** 页9表2小矩形DDIM编辑在M1上Crop1.85s优于SIGE1.96s；所有扩散步缓存内存、全局阴影变化、低分辨率或大改动均有限制。外置附录未包含在本地PDF。
- **当前含义：** 必须让最简单的匹配替代方案有机会赢；不能只靠局部代理成本宣传门控有效。
- **页表锚点：** [[4, "Failed unstructured speedup"], [8, "Latency methodology"], [9, "Tables2–3"], [10, "Limitations"]]

### 10. TorchSparse: Efficient Point Cloud Inference Engine

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2022/TorchSparse- Efficient Point Cloud Inference Engine.pdf>) · [存档源](https://arxiv.org/pdf/2204.10319) · 14页全文文本 · 阅读者 `design_audit`。

- **问题：** 稀疏CNN主要受不规则matmul和scatter/gather数据流限制；稀疏FLOPs低不代表GPU快。
- **贡献：** adaptive MM分组以冗余计算换regularity，量化+向量化+局部性融合，mapping kernel融合；各组件和E2E速度分列。
- **强与简单对照：** 同模型同数据三代GPU对比MinkowskiEngine.5.4/SpConv1.2.1 FP32/FP16和无优化baseline；symmetric/fixed/adaptive分组；跨dataset/model/hardware策略迁移；memory各子项消融；报告100样本<10min调参。
- **打平与独立收益：** nuScenes固定分组1.50×与adaptive1.54×接近，但相同固定策略在SemanticKITTI是.87×（反而慢），adaptive1.39×。适应性贡献由可复现跨工况退化和最小对照立住，非‘调参自动’本身。
- **限制：** 只讨论GPUinference；组件倍率不能连乘当E2E（总约1.5–1.6×）；autotune先验开销需单列；泛化限所测7models/3datasets/3GPU。
- **可迁移判断：** 如果要保留adaptive gate贡献，最有力是同一固定LT在事先有意义工况失效、自适应gate保持有效，且固定与adaptive给对等开发预算；应纳入‘固定’强对照。
- **不能类比处：** 当前LT20/50跨已测设置打平，尚无TorchSparse式固定策略跨工况崩溃；不能臆造新工况来追求反例。
- **页表锚点：** [{"page": 6, "item": "Sec4.2.3", "finding": "100训练样本搜索约1000配方<10分钟，推理时不调参数但分组随输入变化"}, {"page": 9, "item": "Tables1/2", "finding": "fixed/adaptive：SK .87/1.39×，NS1.50/1.54×；硬件/模型/数据迁移对照"}, {"page": 10, "item": "Table3/Fig13", "finding": "纯FP16不等于2×，向量化/融合/局部性增益逐项"}, {"page": 13, "item": "Algorithms3-4", "finding": "输出坐标与分组算法"}, {"page": 14, "item": "Algorithm5/Fig14", "finding": "搜索成本函数是实测延迟；报告绝对FPS"}]

## 2023

### 11. BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2023/BEVFusion- Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation.pdf>) · [存档源](https://arxiv.org/pdf/2205.13542) · 8页全文文本 · 阅读者 `design_audit`。

- **问题：** 点级camera→LiDAR融合丢弃大部分相机语义；检测优秀并不能证明地图分割有效，BEV pooling又是延迟瓶颈。
- **贡献：** 以共享BEV同时保留几何和语义，精确优化pooling的预计算/区间归约，支持检测与分割。
- **强与简单对照：** TransFusion、MVP、PointPainting、LiDAR-only CenterPoint、camera-only LSS/CVT；测MAC和RTX3090延迟；光照/天气/物体距离大小/LiDAR稀疏度分组。
- **打平与独立收益：** 不是靠主指标完全打平立住；nuScenes val NDS 71.4 vs TransFusion71.3很近，但test mAP/NDS +1.3pp，119.2 vs156.6ms，map segmentation62.7 vs约49mIoU。应并列既定多任务和实测成本，不能拿40×kernel当40×端到端。
- **限制：** BEV坐标/相机标定及深度估计依赖；语义/几何任务不是同一终点，部分对手无开源延迟；Waymo含TTA和ensemble差异明确表注。
- **可迁移判断：** 先识别决定部署价值的资源与能力终点，再看简单对手；分组更难工况要有机制和任务价值依据，并保留主终点。
- **不能类比处：** Curfew若与LT在ASR/质量/成本均打平，没有已知40×真实瓶颈可借；不能为了摆脱打平任加新任务。
- **页表锚点：** [{"page": 3, "item": "Fig3", "finding": "pooling约512.1→12.0ms，精确优化；不等于端到端40×"}, {"page": 4, "item": "TableI", "finding": "TransFusion test mAP68.9/NDS71.6，BEVFusion70.2/72.9；latency156.6vs119.2ms"}, {"page": 5, "item": "TableIII/IV", "finding": "map mIoU49.1/49.0→62.7；天气/昼夜条件"}, {"page": 6, "item": "Fig4", "finding": "更稀疏LiDAR与远小目标的机制压力"}]

### 12. EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2023/EfficientViT- Multi-Scale Linear Attention for High-Resolution Dense Prediction.pdf>) · [存档源](https://arxiv.org/pdf/2205.14756) · 12页全文文本 · 阅读者 `design_audit`。

- **问题：** 高分辨率dense prediction需要全局/多尺度而softmax与大kernel在硬件低效；FLOPs不保证延迟。
- **贡献：** ReLU linear attention配小卷积多尺度QKV和FFN局部卷积；简单encoder/head，面向真实硬件的Pareto改进。
- **强与简单对照：** SegFormer/SegNeXt、SwinIR/Restormer、CNN SR、SAM/加速SAM；等MAC(4.4G)、同参数(.7M)、random initialization的2×2组件消融；统一TensorRT FP16含数据传输。
- **打平与独立收益：** Table3 Cityscapes SegNeXt-L/EfficientViT-L2 mIoU同83.2；Orin374→60ms，A100 throughput26→102images/s；这是真实同效果大资源优势。
- **限制：** 同效果不代表所有能力无差，Table7 LVIS single-click59.2→56.6退步明确；多硬件/编译器条件不同不可交叉拼倍率；线性attention原始capacity不足，组件消融验证补救。
- **可迁移判断：** 若LT同ASR/正常能力，Curfew必须在同终点实测总成本或其他部署能力有量级差，且不能只比防御额外毫秒；关键组件应等预算去掉验证。
- **不能类比处：** Curfew和LT目前均近零额外前向，无已证实6×实测资源差；不能将“数据驱动”视为效率贡献。
- **页表锚点：** [{"page": 5, "item": "Table1", "finding": "等4.4G/.7M:无两模块68.1、单模块72.3/72.2、两者74.5 mIoU"}, {"page": 7, "item": "Table3", "finding": "83.2mIoU打平，Orin374vs60ms，A10026vs102 images/s"}, {"page": 9, "item": "Table6/7", "finding": "SAM含encoder+head吞吐；LVIS单点弱项明确保留"}]

### 13. FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2023/FlatFormer- Flattened Window Attention for Efficient Point Cloud Transformer.pdf>) · [存档源](https://arxiv.org/pdf/2301.08739) · 12页全文文本 · 阅读者 `design_audit`。

- **问题：** PCT准确率接近稀疏卷积但慢3倍，邻域准备/不均窗口padding与分组开销导致。
- **贡献：** 定长点组替代定几何窗，窗口排序+轴交替+shift维持局部性，配FlashAttention/packedQKV/TritonFFN等系统实现。
- **强与简单对照：** 不同机制强卷积CenterPoint/PillarNet和同机制SST/VoTr/VoxSet，分输入帧数比较，统一SpConv2.2.3和TensorRT8.4头；随机分组、去quantization/axis/shift、分辨率/窗口/组大小消融。
- **打平与独立收益：** 单帧PillarNet与FlatFormer meanL2都67.2，latency11.1vs10.8ms差很小；论文如实并列，主要突破是相对其他PCT4×加速且更高精度，2/3帧更强Pareto；单指标局部打平不等于全竞争面优势。
- **限制：** 延迟排除加载和后处理并排除outlier；VoTr为projected latency、SWFormer为T4明确标注；2stage瓶颈在第二阶段，不把backbone2.9×当总系统；最后不满组丢<.1%点。
- **可迁移判断：** 必须纳入强不同机制简单法；如果对最近强简单法几乎同一Pareto点，就诚实承认；组件/系统贡献分别验证，增加参数或更复杂并非贡献。
- **不能类比处：** Curfew对LT打平时不能靠“我是自适应框架/新架构”取得FlatFormer已实测4.6×的同类突破。
- **页表锚点：** [{"page": 6, "item": "Table1", "finding": "singleframe PillarNet/FlatFormer67.2/67.2,11.1/10.8ms；不同帧分组"}, {"page": 6, "item": "latency protocol", "finding": "1000样本50次10warmup，排除加载后处理和outlier"}, {"page": 7, "item": "Table3", "finding": "20%训练数据用于设计消融；ours61.7vsrandom57.8，去模块下降"}, {"page": 8, "item": "Fig8/Tables4-6", "finding": "backbone系统2.9×；0.32与0.28m均61.7，选更便宜0.32m"}]

### 14. PockEngine: Sparse and Efficient Fine-tuning in a Pocket

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2023/PockEngine- Sparse and Efficient Fine-tuning in a Pocket.pdf>) · [存档源](https://arxiv.org/pdf/2310.17752) · 14页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 如何让稀疏反传和训练图优化在真实边缘设备兑现内存与训练速度收益？
- **贡献：** 编译时自动微分、稀疏反传图裁剪、算子重排/融合和硬件代码生成；离线敏感性/进化搜索稀疏更新方案。
- **强与简单对照：** TensorFlow/PyTorch/JAX/MNN；全BP、bias-only、稀疏BP；Llama2-7B与rank8 LoRA。Tables2/3跨任务精度；Table5同训练算法的编译框架与额外稀疏更新分开。
- **打平与独立收益：** 精度接近本就是目标，收益由另一项预设终点真实延迟/内存支持：p11Table5 PyTorch全BP7.7s、PockEngine全BP1.8s、稀疏0.9s；对应Alpaca44.1/43.7/43.1，MTBench6.1/6.1/5.7，不能说所有质量完全一致。
- **限制：** 搜索每层单独微调至收敛，离线搜索/编译成本不在迭代延迟内；MCU TensorFlow部分是投影延迟。抽取含隐藏TODO/叠印，不据此断言可见PDF同样有草稿；已渲染p11，正文把稀疏Alpaca写43.7而表列43.1，是真实表文不一致。Llama层数正文32与24也需版本核验。不擅自纠正成确定结果。
- **可迁移判断：** 若Curfew与固定LT ASR/质量打平，只有真实新增部署能力或端到端成本下降可独立立住；参数/反传量减少不能替代实测。分开算法增量与工程实现增量，计入搜索开销。
- **不能类比处：** 全BP性能不退与对抗后门风险不同；设备编译加速不证明自适应门控必要。
- **页表锚点：** ["p5 Section3.1 search", "p8 Tables2/3", "p10 Table4", "p11 Table5", "p12 qualitative limits and conclusion"]

### 15. Retrospective: EIE: Efficient Inference Engine on Sparse and Compressed Neural Network

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2023/Retrospective- EIE- Efficient Inference Engine on Sparse and Compressed Neural Network.pdf>) · [存档源](https://arxiv.org/pdf/2306.09552v1) · 2页全文文本 · 阅读者 `github_audit`。

- **问题：** 回顾压缩稀疏网络的硬件瓶颈与早期EIE设计适用范围。
- **贡献：** 作者本人总结内存优先设计、稀疏激活收益和非结构稀疏的索引/控制代价，并说明瓶颈转移后的机会。
- **强与简单对照：** 这是两页回顾而非新实验论文；引用原EIE、Deep Compression等公开结果，没有新一组公平benchmark。
- **最终终点与证据：** 页2指出4bit权重+4bit索引有50%存储开销、CSC遍历复杂，结构化/分块稀疏可能更合适；权重压缩后激活成为新瓶颈。
- **限制：** 只完整阅读本地两页Retrospective，不能声称因此读过原EIE全部论文；公开回顾不证明他们如何停掉未发表失败项目。
- **当前含义：** 强简单基线解决旧瓶颈后，应重判问题是否仍存在。可以停止不必要的门控组件；不能从历史故事反推Curfew必然有下一条正向方法线。
- **页表锚点：** [[1, "Motivations"], [2, "§III Lessons/§IV Opportunities"]]

### 16. Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models - TPAMI extended version

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2023/Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models - TPAMI extended version.pdf>) · [存档源](https://arxiv.org/pdf/2211.02048v4) · 21页全文文本 · 阅读者 `root`。

- **贡献：** 局部编辑时复用未变化区域激活，块稀疏卷积/注意力与定制kernel把局部结构转成实际速度。
- **关键对照：** p9 Crop对照也获预计算normalization统计；p10-11拆分naive稀疏、融合算子、cache/散聚；主文多个扩散/GAN及编辑面积对照。
- **关键证据：** conference方案到新attention支持的增量有证据：SD旧版本约1.1x到新约4.8x；7.6x MAC减少不等于运行时间，naive版本仅约1.6x。
- **限制：** 依赖编辑局部性、cache及固定text等条件；大编辑区域收益缩小；缓存生成和后续局部编辑成本须分开。
- **当前含义：** 清楚界定相对自己前一版本的新增能力，给强且获相同资源的简单对照，不用更多benchmark掩盖核心增量。

### 17. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2023/SmoothQuant- Accurate and Efficient Post-Training Quantization for Large Language Models.pdf>) · [存档源](https://arxiv.org/pdf/2211.10438) · 13页全文文本 · 阅读者 `root`。

- **贡献：** 等价对角缩放将激活量化难度转移给权重，实现硬件可用W8A8。
- **关键证据：** p3 Table1区分准确但不适合GEMM的逐通道方案；p6给ZeroQuant FP16输入适配；p8–9实测延迟/显存；p13拒绝跨工作负载误导性比较GPTQ。
- **限制：** alpha按架构开发选择，512校准句；1.56x限特定负载，不是所有配置都更快。
- **当前含义：** 质量近似打平由实测资源收益立住；Curfew不能借这个故事虚构成本优势。

### 18. Tiny Machine Learning: Progress and Futures [Feature]

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2023/Tiny Machine Learning- Progress and Futures [Feature].pdf>) · [存档源](https://arxiv.org/pdf/2403.19076v2) · 24页全文文本 · 阅读者 `root`。

- **贡献：** TinyML综述/整合：部署和训练的真实约束是MCU可用SRAM、activation及操作系统开销，需模型/调度/runtime共同设计。
- **关键对照：** p5 TableIII在相同MCU/模型比较TFLM、CMSIS-NN、X-Cube-AI等；默认balanced模式明示；后续归纳QAS、稀疏更新、activation-aware pruning。
- **关键证据：** 参数小不等于峰值内存小，MAC少不等于可部署；真实板端内存/延迟才回答任务。
- **限制：** 这是综述，不算新增一组独立论文实验；TinyEngine并非所有行内存最小；不能把特定MCU收益直接搬到LLM训练。
- **当前含义：** 需要命名用户最终获得的收益及实际瓶颈；gate曲线或少删行不能代替安全/正常学习/成本终点。

### 19. TorchSparse++: Efficient Training and Inference Framework for Sparse Convolution on GPUs

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2023/TorchSparse++- Efficient Training and Inference Framework for Sparse Convolution on GPUs.pdf>) · [存档源](https://dai.sjtu.edu.cn/my_file/pdf/c9839ed0-3108-4a86-9ada-b4fd8ad5bed3.pdf) · 15页全文文本 · 阅读者 `design_audit`。

- **问题：** 稀疏高性能dataflow工程代价高，局部FLOPs/DRAM/kernel代理与真正总延迟不一致。
- **贡献：** 小型sparse kernel generator扩展dense compiler、扩大dataflow设计空间、group共享map与训练forward/dgrad/wgrad绑定autotune；不预先用代理指标裁掉候选。
- **强与简单对照：** MinkowskiEngine.5.4/SpConv1.2.1/SpConv2.3.5/TorchSparse同runtime；same-dataflow实现也比官方快1.1–1.2×；增量设计空间消融；真实硬件4架构和3精度，训练确认sameaccuracy；模拟ASIC明确标投影。
- **打平与独立收益：** 大量局部相近并未夸大，fetchondemand混合仅约1.06×；关键是全系统不同工况实际收益。Tables3/4显示unsorted kernel较慢而总延迟较快，计入mapping后结论反转；还以新backend让旧CenterPoint在Orin反超FlatFormer，说明对手实现质量决定结论。
- **限制：** 检测只测SparseConv层，所谓end-to-end是该范围不是全部驾驶pipeline；ASIC比较含强线性外推假设；工程代码行数不是完整开发时间；未来应用是推测。
- **可迁移判断：** 对手必须充分工程实现，包含完整代价再判断；不可拿更少HVP/更少防御秒/召回率代替最终训练部署效果；可以由简单原有baseline反超而主动修正定位。
- **不能类比处：** autotuner以E2E实测函数选优，Curfewgate依赖loss状态；没有等价于本论文已验证的1.7×总成本优势，不能借系统论文名义换成实现小优化。
- **页表锚点：** [{"page": 7, "item": "Fig12/13 Sec4.2", "finding": "group级E2E调优约2分钟/100场景；不共享map会多3–4层latency"}, {"page": 8, "item": "Sec5.1/5.2", "finding": "检测只测SparseConv；新的强baseline backend可让CenterPoint反超FlatFormer"}, {"page": 9, "item": "Table3", "finding": "NS unsorted7.45ms vs split1 8.18"}, {"page": 10, "item": "Table4", "finding": "单kernel unsorted3.70 vs split1 3.68与总排名相反"}, {"page": 13, "item": "ArtifactA", "finding": "Docker/改过baseline的可复现入口、subset35GB与full2TB差异明确"}]

## 2024

### 20. Atomique: A Quantum Compiler for Reconfigurable Neutral Atom Arrays

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/Atomique- A Quantum Compiler for Reconfigurable Neutral Atom Arrays.pdf>) · [存档源](https://arxiv.org/pdf/2311.15123) · 17页全文文本 · 阅读者 `github_audit`。

- **问题：** 可移动中性原子阵列受集体移动、行列顺序和重叠约束；减少 SWAP 与移动噪声之间存在冲突。
- **贡献：** 分层 max-k-cut 阵列映射、负载均衡/对角优先位置映射及并行路由。
- **强与简单对照：** 相同量子位资源下的超导/固定原子阵列、Baker、Tan Solver/IterP、Q-Pilot、Geyser；单AOD能力交集、公平CPU超时及模块累加消融。
- **最终终点与证据：** 页10同RAA下与求解器保真度接近，CPU编译时间和可扩展性明显改善；这是实际CPU结果与模型化保真度的组合。
- **限制：** 没有在目标RAA硬件上执行；页8使用外推噪声参数。页10–11承认低连接度/高双量子门保真度时固定阵列可能更好；电路深度较小也不保证总保真度更高。
- **当前含义：** 同质量可由已测得的另一约束改善支撑；Curfew/LT现有零额外forward打平并没有对应的编译瓶颈改善。
- **页表锚点：** [[8, "Evaluation"], [10, "Compiler comparison"], [11, "Fig.18"], [12, "Fig.21"], [17, "Artifact Appendix"]]

### 21. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/AWQ- Activation-aware Weight Quantization for LLM Compression and Acceleration.pdf>) · [存档源](https://arxiv.org/pdf/2306.00978) · 15页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 低bit权重量化怎样保留少数重要通道，又不引入难部署混精度。
- **贡献：** 重要通道按激活幅度选；等价缩放+一维alpha网格和clipping；TinyChat把统一低bit转成设备实测收益。
- **强与简单对照：** 强RTNg128、GPTQ、GPTQ-Reorder；激活/权重范数/随机挑FP16；固定s=2、不同scale；校准规模/分布交叉；tinychat FP16优化和其他int4系统。
- **打平与独立收益：** p4Table3 AWQ和1%FP16在OPT6.7B PPL同11.39，但统一INT3存储硬件友好是独立实现贡献；p7Table4 4bit差距小仍须承认。p10Fig8校准16vs192和跨域0.5–0.6vs2.3–4.9，额外证据有明确对照。
- **限制：** activation只保1%为启发式；‘lossless’不是每格完全无降（VILA7BVizWiz59.6→57.8）；3x总系统收益含fusion，不全归AWQ缩放；prompt4/generate200,batch1场景。
- **可迁移判断：** 先直面固定s=2之类简单对手；若主精度接近，要有同质量实际部署/信息依赖或迁移稳定性收益。Curfew若没有省信息/省成本证据，不可仅借AWQ泛化故事。
- **不能类比处：** 量化可做等价浮点变换并实测设备速度；防御阈值无相应等价理论。不得把小PPL差异与ASR安全阈值混同。
- **页表锚点：** ["p4 Tables1–3含随机、W幅度、固定s对照", "p7 Table4强RTN/GPTQ-R", "p8 Tables6–8多模态/code/math", "p10Fig8校准量与跨域", "p11同强FP16 kernel对照、短prompt和batch1说明"]

### 22. Condition-Aware Neural Network for Controlled Image Generation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/Condition-Aware Neural Network for Controlled Image Generation.pdf>) · [存档源](https://arxiv.org/pdf/2404.01143) · 10页全文文本 · 阅读者 `github_audit`。

- **问题：** 条件生成中只调整特征而保持共享权重，可能难以适应类别/时间步相关的不同任务；全面生成权重又昂贵。
- **贡献：** 在选定depthwise卷积、patch embedding、共享attention输出投影上生成条件残差权重，分组卷积批处理，进一步构建CaT。
- **强与简单对照：** 匹配UViT/DiT骨干的DWconv；2–12个自适应kernel bank、adaptive norm、condition token；模块位置、类别/时间步、训练长度消融。
- **最终终点与证据：** 页5表3：UViT CAN-only FID8.79/CLIP31.75，叠加attention为8.82/31.74，后续删attention；DiT CAN+norm21.76优于CAN-only26.44。
- **限制：** 页4仍有30–40%训练开销；页6–7 CaT与DiT同时改骨干和20vs250采样步，不能把525G→10G/步或总延迟全部归给条件模块。
- **当前含义：** 公开成品就支持删掉打平的复杂子模块。Curfew不能因整体防御有效就归因于gate；也不能因其他模型里某个自适应模块有效而保留当前没有增量的gate。
- **页表锚点：** [[3, "Table1"], [4, "Implementation"], [5, "Table3; rendered"], [6, "Table4"], [7, "Table5"]]

### 23. DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/DistriFusion- Distributed Parallel Inference for High-Resolution Diffusion Models.pdf>) · [存档源](https://arxiv.org/pdf/2402.19481) · 12页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 多GPU单张高分辨率扩散怎么避免patch伪影与同步通信开销。
- **贡献：** 复用前一步stale全局特征，当前patch稀疏算；异步AllGather；校正异步GroupNorm；固定warmup保证早期布局。
- **强与简单对照：** naivepatch、ParaDiGMS、同步TP/PP、理想无通信；GroupNorm四方案；固定1/2步warmup。
- **打平与独立收益：** 50step下ParaDiGMS与本法8卡均约2.8x，但本法FID/原输出保真更好且总MAC338T vs657T；GN校正质量近同步GN，延迟更低。不是因为新名词，而是量化误差/通信/总计算独立可核。
- **限制：** 速度依赖NVLink与高分辨率；低分辨率卡利用率低；极少步输入变化大，stale近似失效；多卡延迟下降不是减少卡时；需计总MAC和GPU占用。
- **可迁移判断：** 相同时延下要报保真/总计算；warmup固定有效本身就承认；若Curfew与固定LT全端点打平，不能仅宣称自适应更高级。
- **不能类比处：** 扩散邻步连续性和异步通信不能类推投毒梯度学习阶段；8卡6.1x非8倍计算节省。
- **页表锚点：** ["p6默认首步+4步同步warmup", "p8Table1质量+总MAC+实测时延", "p8Table2 TP/PP/无通信", "p9Fig8固定warmup与Fig9四GN方案", "p9限制与p10PSNR更正记录"]

### 24. EfficientViT-SAM: Accelerated Segment Anything Model Without Accuracy Loss

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/EfficientViT-SAM- Accelerated Segment Anything Model Without Accuracy Loss.pdf>) · [存档源](https://arxiv.org/pdf/2402.05008v2) · 5页全文文本 · 阅读者 `root`。

- **贡献：** 硬件友好EfficientViT替换SAM昂贵encoder，蒸馏再端到端训练。
- **关键证据：** p3 Tables1–4多SAM强对手/实际TensorRT A100；L2 COCO46.6vs46.5，吞吐538vs11图/秒。
- **限制：** 短文没有所有组件单独完整消融；训练使用SA1B和教师；不同数据/点击方式并非全赢。
- **当前含义：** 精度打平有意义是因为另有明确部署收益，不能只凭同精度保留Curfew。

### 25. FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/FastComposer- Tuning-Free Multi-Subject Image Generation with Localized Attention.pdf>) · [存档源](https://arxiv.org/pdf/2305.10431) · 12页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 个体定制须逐个微调且多人物身份混淆。
- **贡献：** 通用训练的视觉编码器增强文本token；训练时分割监督cross-attention；推理时延迟身份条件。
- **强与简单对照：** DreamBooth、TextualInversion、CustomDiffusion、text-onlySD；去localization；条件占比扫描。
- **打平与独立收益：** 多主体身份43.11%明显强，但prompt24.25低于CustomDiffusion25.84。Table3 localization身份37.66→43.11同时prompt25.03→24.25。独立价值为新主体免微调和多主体身份能力。
- **限制：** 只人脸FFHQ，>3主体和动物不足；150ksteps×8A6000通用训练未计入新主体2秒使用成本。
- **可迁移判断：** 延迟条件靠固定比例扫描和身份/编辑双终点；Curfew需真实独立收益，不能借时机比喻维持必要性。
- **不能类比处：** 推理视觉条件开关不是训练投毒；新主体免训练收益明确，Curfew未建立同类收益。
- **页表锚点：** ["p6 Fig5:比例0.6–0.8", "p8 Table1和训练细节", "p9 Tables2–3和局限", "p10局限，p11–12参考"]

### 26. LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/LongLoRA- Efficient Fine-tuning of Long-Context Large Language Models.pdf>) · [存档源](https://arxiv.org/pdf/2309.12307) · 19页全文文本 · 阅读者 `root`。

- **贡献：** 训练用shifted local attention，加可训练embedding/norm，推理保留dense attention。
- **关键证据：** p6 Table2增大LoRA rank8–256不能替代embedding/norm；p9同计算注意力模式对照；p16实测时长和显存。
- **限制：** 配FlashAttention2后相对LoRA显存近似，时间仍不同；短上下文PPL会退步；局部训练不等于所有长度无损。
- **当前含义：** 直接挑战简单参数扩大能否解释收益；局部操作少不等于端到端更省。

### 27. Q-Pilot: Field Programmable Qubit Array Compilation with Flying Ancillas

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/Q-Pilot- Field Programmable Qubit Array Compilation with Flying Ancillas.pdf>) · [存档源](https://arxiv.org/pdf/2311.16190) · 10页全文文本 · 阅读者 `design_audit`。

- **问题：** FPQA动态连通性需要新router，静态SWAP编译和精确solver扩展性不足。
- **贡献：** 固定dataqubit+飞行ancilla路由、CZ等价性证明、generic greedy与QSim/QAOA定制router；硬件适配和算法协同。
- **强与简单对照：** Qiskit level3在超导/正方/三角固定原子阵列；同FPQA solver与iterative改法；generic vs应用特定；阵列尺寸/噪声/并行度/编译规模。
- **打平与独立收益：** 并非在所有主终点击败精确解：Table2 solver小问题深度更短，Q-Pilot接受较差深度以获得<1s编译、100Q可行，solver4000s timeout；明确比较质量-编译时间而非宣称逐点最优。
- **限制：** 实际物理fidelity用误差模型估计不是硬件跑；增加ancilla是资源权限差异，不能当免费；abstractQAOA6.3×和body6.7×文本不一致，未引用其单一数字作核心证据；不同平台基线不等于同硬件软件增益。
- **可迁移判断：** 如果复杂方法需要额外数据/参考/状态，应单列资源；可接受效果换成本但需明确量级及Pareto；不同机制可比较部署终点，原理边界要证明而非类比。
- **不能类比处：** Curfew无数学等价性或已证编译数量级优势；不能把理论可行性、对其他硬件更快类比成当前门必要。
- **页表锚点：** [{"page": 2, "item": "Sec2.2 Eqs1-4", "finding": "CZ+ancilla等价性适用边界，非任意2Q gate"}, {"page": 7, "item": "Table2", "finding": "精确解更浅但超时；Q-Pilot100Q<1s"}, {"page": 8, "item": "Fig16", "finding": "generic-vs-domain-specific成分对照"}, {"page": 7, "item": "Eq5", "finding": "误差估计包括ancilla数和移动距离，不是实验机真值"}]

### 28. Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/Quest- Query-Aware Sparsity for Efficient Long-Context LLM Inference.pdf>) · [存档源](https://arxiv.org/pdf/2406.10774) · 11页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 历史KV被永久驱逐后无法回答后来的长依赖问题。
- **贡献：** 保留全部KV，以query×页keymin/max上界估计topK页；前两层对称不剪。
- **强与简单对照：** FullFlashInfer、H2O/TOVA/StreamingLLM；预算曲线、PG19、passkey、6LongBench；算子与decode总时延。
- **打平与独立收益：** 与全KV质量打平是约束，贡献是低时延；7.03倍attention与2.23倍decode总时延不同；passkey揭示PPL看不到的长依赖退化。
- **限制：** 不减少KV容量；H2O因FlashAttention不在prefill累积历史；对手无kernel用FlashInfer作忽略额外开销的定性速度下界；decode不含prefill/采样或批量serve。
- **可迁移判断：** 要用真实且预定义的新端点或成本优势立住；保留对手实现和口径限制。
- **不能类比处：** 可保留cache再按当前query取，不同于删训练数据，也不证明所有训练gate必要。
- **页表锚点：** ["p4 Fig5算法", "p5 Table1passkey", "p6 H2O适配", "p8 Fig9/§4.3.2", "p9 Fig10 FP16 1.74x/AWQ2.23x"]

### 29. Sparse Refinement for Efficient High-Resolution Semantic Segmentation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/Sparse Refinement for Efficient High-Resolution Semantic Segmentation.pdf>) · [存档源](https://arxiv.org/pdf/2407.19014) · 20页全文文本 · 阅读者 `github_audit`。

- **问题：** 低分辨率分割丢失边界和小物体，但全分辨率稠密推理昂贵。
- **贡献：** 低分辨率预测器+熵选择的高分辨率RGB稀疏UNet+学习门控融合。
- **强与简单对照：** 多骨干全/低分辨率，SparseViT、按延迟匹配的PointRend、256/512patch；随机/幅值/学习/熵选择，直接/熵/门控融合和容量/输入/backend消融。
- **最终终点与证据：** 页12表5a：学习与熵选择同80.9mIoU，4.0vs2.3ms，采用简单熵规则。表5e保留门控，因为80.9确实高于熵融合80.3和直接77.7；表6另列2.2ms成本。
- **限制：** 额外训练refiner、按模型设熵阈值，有稀疏内核依赖；不是零训练成本。提供PDF共20页，但文中引用的额外附录不在该PDF，不能声明已读该外置附录。
- **当前含义：** 最直接的门控裁决范例：Curfew要有自己的表5e；目前LT同结果，证据反而更像表5a应选简单者。不能由图像熵检测错分推导训练投毒识别。
- **页表锚点：** [[8, "Table1"], [9, "§4.1"], [12, "Tables5a/5e/6"], [13, "§4.3"]]

### 30. Efficient Streaming Language Models with Attention Sinks

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/Efficient Streaming Language Models with Attention Sinks.pdf>) · [存档源](https://arxiv.org/pdf/2309.17453) · 21页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 有限预训练窗口下，持续输入时普通滑窗KV为何崩溃，如何不重算全窗而维持近期语言能力。
- **贡献：** 保留4个起始sink与近期KV，并按cache连续位置重编码；语义替换/预训练单learnedsink补机制证据。
- **强与简单对照：** dense、window、强sliding-window重算；首token换换行、初始token数、cache数、zero-sink/learnedsink、1vs2learnedsink、跨家族尺度、长依赖距离、LongBench截断。
- **打平与独立收益：** PPL与昂贵重算打平而decode最高22.2x更快；附录D/Table8明确长文任务对廉价1750+1750截断无优势，承认不扩展长期记忆。独立服务目标是持续生成且只需近期内容，不是对所有长文QA更强。
- **限制：** cache外答案准确率为0；增加cache并非总改善；sink解释是经验机制而非普遍数学必要性。默认4sink在LongBench较截断弱，改成1750+1750仅打平。
- **可迁移判断：** Curfew可以收窄目标，但必须确有不同任务需求与baseline破绽及实际成本收益；要像附录明确报告便宜基线打平，不能把新目标写成事后免责。
- **不能类比处：** 窗口内稳定不是长程记忆；保留sink/KV是推理机制，不能推出训练lossgate必要。
- **页表锚点：** ["p4–5/Table1换行语义控制、Table2初始token数", "p6 Table3 zero/learned sink", "p7同初始化/数据排列的160M预训练、Table4", "p9 Fig10成本；p16 Table7 cache外全0", "p17 Table8截断打平；p21 Table9–10第二sink无增益"]

### 31. VILA: On Pre-training for Visual Language Models

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2024/VILA- On Pre-training for Visual Language Models.pdf>) · [存档源](https://arxiv.org/pdf/2312.07533) · 16页全文文本 · 阅读者 `design_audit`。

- **问题：** 视觉语言预训练中是否更新LLM、用interleaved还是caption数据、怎样保留文本能力缺少受控比较。
- **贡献：** 逐项比较训练配方：训练LLM而非冻结、interleaved结构、文本SFT混入；不是一种复杂新模块。
- **强与简单对照：** 冻结LLM+Transformer projector、训练LLM同projector、简单linear projector；MMC4-pairs与完整MMC4隔离结构；COYO等量25M图；joint SFT与视觉SFT；visual expert 1.9×参数、LoRA r64、Vicuna复核。
- **打平与独立收益：** Table1 b/c零样本均值均66.8，但4-shot57.6→68.8；caption是SFT外分布，4-shot本来就是评估任务，非事后创造指标。简单linear projector进一步68.7/70.9优于Transformer66.8/68.8，论文采用简单方案。
- **限制：** 大规模最终比较训练数据量不同；消融协议绝对值不可与主表直接比较；embedding alignment与ICL为相关假设，不是因果证明；world knowledge仅4例；50M图像未扩billion；7B文本保留略差，13B较好。
- **可迁移判断：** 主终点饱和时，只能用有预先任务意义的额外能力来辨别。新知识保留可有部署意义，但必须与固定LT直接比，且不能把NLL等同QA准确率；先做单因子比较，简单规则若够好就采用。
- **不能类比处：** 66.8不是全部能力打平；VILA已经有11.2点ICL差，Curfew目前没有已证实的同资源独立增量；公开成品不能说明作者失败研发历史。
- **页表锚点：** [{"page": 3, "item": "Table1", "finding": "b/c0shot66.8/66.8；4shot57.6/68.8；simplelinear d68.7/70.9"}, {"page": 4, "item": "Table3", "finding": "MMC4结构对照与MMLU退化"}, {"page": 5, "item": "Table4", "finding": "jointSFT恢复文本能力并提升视觉任务"}, {"page": 8, "item": "Tables7-9", "finding": "visualexpert1.9×params；LoRA r64低于fulltune；token compression保留明显TextVQA差距未入主法"}, {"page": 13, "item": "AppendixB/C Table11", "finding": "7B约5.1k A100GPUh；Vicuna复核"}]

## 2025

### 32. COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/COAT- Compressing Optimizer states and Activation for Memory-Efficient FP8 Training.pdf>) · [存档源](https://arxiv.org/pdf/2410.19313) · 22页全文文本 · 阅读者 `design_audit`。

- **问题：** FP8linear虽快但未量化optimizer secondmoment/非线性activation，内存瓶颈仍在；单kernel加速非完整训练价值。
- **贡献：** power变换扩optimizer动态范围与混合粒度activationquant；group scaling+fusedkernels使精度内存速度兼容。
- **强与简单对照：** BF16/TransformerEngine/FP8LM；optimizer-only vsactivation-only vsboth；DE格式迁移；不同groupsize；LLMpretrain/finetune/VLMSFT；单层与端到端、maxbatch分别报。
- **打平与独立收益：** Llama13B8GPU TE/COAT throughput5604/5650近打平，但同batch peak46.5/32.5GB、maxbatch2/4明确不同；VLMavg62.80vs62.51为近似而非严格无损，普通7B8GPU COAT11241反低于TE11704速度，仍保留。
- **限制：** 精度表不是每格统计非劣；速度按最大batch而memory按固定batch，比较口径需分开；梯度BF16非全流FP8；附录F证明在scaleinvariance等假设下幂函数形式唯一，不是无条件全局MSE最优；table12文字<5%与首行10.1vs8.7不全一致，避免照抄。
- **可迁移判断：** 同ASR时可以由实测部署内存等独立资源立住，但必须同配置并报告退步；效应不普适时收窄声明。
- **不能类比处：** Curfew与LT目前没有训练内存/可承载任务差异的已证据，不可把‘可以找其他维度’当贡献。
- **页表锚点：** [{"page": 8, "item": "Tables3-5", "finding": "OLMo及SFT几乎相同但保留小降分"}, {"page": 10, "item": "Table8", "finding": "13B8GPU TE5604/COAT5650，peak46.5/32.5GB,maxBS2/4；7B8GPU TE速度更高"}, {"page": 19, "item": "AppendixF", "finding": "幂函数证明明确假设范围"}, {"page": 21, "item": "AppendixG5/H", "finding": "梯度BF16；linear pergroup多约40%延迟而精度小 gain故不用"}, {"page": 22, "item": "Tables11-13", "finding": "组件成本分解与accuracy分解，optimizer主要误差来源"}]

### 33. Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models.pdf>) · [存档源](https://arxiv.org/pdf/2410.10733v8) · 22页全文文本 · 阅读者 `github_audit`。

- **问题：** 常规VAE在高空间压缩率下难优化，latent体积和diffusion token数限制高分辨率生成效率。
- **贡献：** 非参数空间/通道残差捷径，以及低分辨率全训练—高分辨率中段适配—解码器精修的三阶段训练。
- **强与简单对照：** SD VAE多压缩率、SDXL、Flux/SD3、AsymVAE；相同latent体积、更大容量、同token数、训练层、sampler/guidance控制。
- **最终终点与证据：** 同质量下实际扩散吞吐/延迟更好。页7明确因性能相同且更简单选择确定性AE而非VAE。
- **限制：** 更高压缩对小UViT-S可能更差，对2B更好；新的AE训练成本未包含在下游吞吐倍数。不能把不同训练预算和NFE的数字混用。
- **当前含义：** 论文也主动采用打平的简单方案。与其保留门控名义，不如先判断它是否解决了固定LT未解决的优化/时机瓶颈。
- **页表锚点：** [[4, "Matched latent/capacity controls"], [6, "Table1"], [7, "AE vsVAE"], [8, "Efficiency"], [9, "Model-size boundary"]]

### 34. DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/DC-AE 1.5- Accelerating Diffusion Model Convergence with Structured Latent Space.pdf>) · [存档源](https://arxiv.org/pdf/2508.00413v1) · 13页全文文本 · 阅读者 `design_audit`。

- **问题：** 增加latent channels改善rFID，却减慢扩散模型收敛、恶化gFID；单个重建代理指标与端到端生成目标冲突。
- **贡献：** 通道前缀随机mask训练结构化latent，加上扩散训练相同通道mask目标；两模块协同使高通道/高压缩率可用。
- **强与简单对照：** 同f32c128/f64c128、同DiT/UViT/SiT/USiT；两模块2×2消融；小通道c32负例；同模型尺寸扩展曲线和H100训练throughput。
- **打平与独立收益：** Table1 f32c128 rFID都.26，真正终点gFID和训练收敛明显不同；Table5 DiT baseline26.44、仅结构26.75、仅增强36.83、二者17.31；小c32新法反而略差，明确推荐旧AE。
- **限制：** 结构稀疏是可视化支持的机制解释，不是严格定理；全部ImageNet生成，未泛化任意数据；吞吐提高来自latent token减少，training-step6×不是自动GPUh6×。
- **可迁移判断：** 最有价值的是“复杂模块在哪些已定义条件下必要”及反例：Curfew必须直接对LT证明gate有增量，若没有适用的稀疏/时机问题就明确用简单法；代理召回不等于最终ASR。
- **不能类比处：** DC-AE在真正生成终点有巨大增益，Curfew目前ASR/质量同终点打平；不能用rFID打平案例给Curfew自动续命。
- **页表锚点：** [{"page": 5, "item": "Table1/Fig5", "finding": "同rFID.26；UViT-H训练步收敛6×"}, {"page": 7, "item": "Table4/5", "finding": "同USiT2B:112→450images/s且gFID2.90→2.18；组件缺失可失败，联合有效，已看图核验"}, {"page": 8, "item": "Table6", "finding": "c32 DiT10.18→10.50、USiT3.89→4.15；推荐保留旧法"}, {"page": 11, "item": "AppendixB", "finding": "50k重建val/50k生成对train计算指标"}]

### 35. DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/DC-AR- Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer.pdf>) · [存档源](https://arxiv.org/pdf/2507.04947v1) · 18页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 高压缩二维tokenizer如何保留跨分辨率复用和图像细节，使masked AR在质量/成本上竞争？
- **贡献：** DC-HT32x空间压缩混合tokenizer与连续预热→离散学习→冻结encoder/quantizer交替微调；MaskGIT先离散结构再轻量diffusion残差细节。
- **强与简单对照：** SDXL/PixArt/SD3/Sana/Show-o/TA-TiTok；消融包括discrete-only，去连续warmup、HART式alternate training、从头高分辨率训练、采样步数。
- **打平与独立收益：** GenEval0.69与Show-o0.68接近，仍有半模型规模、同512推理10.3vs1.3img/s可独立评估；不是单凭GenEval小差。Table4混合vs离散gFID5.50vs6.71、GenEval.69vs.66，却吞吐10.3vs11.4，增加约10%开销。
- **限制：** 内部MidJourney数据训练；部分对手质量原分辨率与效率512分开，需保留标注。AppendixA.6明确未比VAR/HART类因为聚焦MaskGIT，这是比较范围限制，不能推所有同任务方法都更优。抽取含隐藏表图叠印，已渲染p8核实Tables4-6，视觉PDF表格清晰，卡片引用单元格一致。
- **可迁移判断：** 继承旧方法可通过具体新瓶颈与受控消融立住，三阶段时机必须和简单训练路径直接比；若打平保留独立部署/成本端点，勿仅换方法族回避强同任务对手。
- **不能类比处：** Tokenizer阶段稳定性不等于后门阶段高loss的安全性，固定3阶段成功不支持adaptive gate必要。
- **页表锚点：** ["p4-5Figure3 method", "p6Table1", "p7Tables2/3", "p8Tables4-6", "p13Tables7/8", "p17-18AppendixA.5/A.6"]

### 36. DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/DC-Gen- Post-Training Diffusion Acceleration with Deeply Compressed Latent Space.pdf>) · [存档源](https://arxiv.org/pdf/2509.25180v3) · 39页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 如何用少量post-training把已有diffusion模型迁移到深压缩latent，解决直接微调不稳定又保留能力？
- **贡献：** patch embedding对齐→冻结backbone输出层对齐→LoRA微调；guidance-distilled模型修正目标；视频chunk-causalAE；跨图像/视频/编辑验证。
- **强与简单对照：** 直接微调、从头训练、去patch对齐、去output对齐、layerwiseLR decay、渐增LoRA、AE latent alignment三比例、fullFT；相同训练数据tuned原模型排除额外数据因素；加速还比较并组合SVDQuant/TaylorSeer/AnyFlow。
- **打平与独立收益：** 接近原质量正是目标，实际可独立检验的贡献是成本/分辨率支持。p15Table6轻量稳定技巧CLIP25.77/26.52vsDCGen27.38；Table7AE-alignment最好26.53vs27.38。去output alignment最终可追平但要约两倍FTsteps，不能说它决定最终能力。
- **限制：** 53.8x指backbone而非端到端：p27Table16 FLUX1K原4.06+AE.15s，新.88+.29s，算术合计约3.60x而非backbone4.62x；4K53.8x对原FLUX不原生支持4K，与实际Diffusion4K20step比较23.76/4.04≈5.88x backbone。posttraining31.25H100GPUdays不包含继承预训练；未见合成数据生成总成本并入该表；p26图caption hours与Table15days冲突，以明示表单位为准。font警告/隐藏叠印已渲染关键p27，主表可读。
- **可迁移判断：** 机制用表征gap诊断后，再用最简单稳定技巧和组件删除直接反驳，不能只对一个弱naive baseline。若Curfew门控与固定LT追平，需证明达到相同目标更少总成本或真实新能力，且不得把局部指标说成全流程。
- **不能类比处：** 新latent迁移有可观测表征gap，后门高loss阶段尚无同类确定诊断；fixed多阶段训练并非adaptivegate的必要性论据。
- **页表锚点：** ["p4-7Figures2-5", "p8-11Tables1-5", "p12-15AppendicesA/B Tables6-8", "p21-23human evaluation", "p24-26Tables12-15", "p27Tables16/17", "p28-31AppendixH"]

### 37. DC-VideoGen: Efficient Video Generation with Deep Compression Video Autoencoder

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/DC-VideoGen- Efficient Video Generation with Deep Compression Video Autoencoder.pdf>) · [存档源](https://arxiv.org/pdf/2509.25182v1) · 19页全文文本 · 阅读者 `root`。

- **贡献：** chunk-causal VAE平衡局部双向质量/长视频扩展，经结构接口对齐和LoRA适配预训练DiT。
- **关键证据：** p5 chunk/causal对照；p6所有竞争VAE也用同AEAdaptV；p9微调Wan强对手；p18接口组件。
- **限制：** 加速明确主要backbone；10GPUday适配不含合成数据生成和继承AE预训练；不能当全流程训练总账。
- **当前含义：** 可给对手相同适配来隔离结构收益；不能拿不兼容的原生配置当公平比较。

### 38. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/DuoAttention- Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.pdf>) · [存档源](https://arxiv.org/pdf/2410.10819) · 20页全文文本 · 阅读者 `root`。

- **贡献：** 按限制历史后的输出变化识别检索头/流式头，分别保存全部或局部KV。
- **关键证据：** p2反向保留和随机；p10同数据attention profiling/优化gate对照；p17明确适配H2O/TOVA并解释其强化；p18交代FastGen复现。
- **限制：** gate训练2000步用8A100；LongBench并非完全无损；3.3M cache可容纳不等于任意历史记忆准确。
- **当前含义：** 要证明状态识别改善实际保留决策；公开适配强对手是允许的，需保留来源及成本。

### 39. HART: Efficient Visual Generation with Hybrid Autoregressive Transformer

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/HART- Efficient Visual Generation with Hybrid Autoregressive Transformer.pdf>) · [存档源](https://arxiv.org/pdf/2410.10812v1) · 20页全文文本 · 阅读者 `root`。

- **贡献：** 离散token承担大结构，连续残差由小扩散模型补细节；交替训练同一个decoder以改善可生成性。
- **关键对照：** p8使用作者推荐sampling配置、同A100/batch8吞吐；p9 Table5用更强同hybrid tokenizer的discrete-only基线；p10 Figure7相同reconstruction FID下分离decoder与交替训练；p16 sparse pattern/80% token训练抽样。
- **关键证据：** VAR旧tokenizer的1024 FID7.85→更强discrete-only5.73→加残差5.38；p10明确强基线上的增量仅约6.1%，与对旧基线31%分开。重建proxy相同仍可生成质量21.98 vs5.30，需实际最终任务终点。
- **限制：** 本地v1 work-in-progress；内部训练数据不等于所有竞争模型同数据重训；视觉示例不是独立盲评；sparse训练kernel仅future方向。
- **当前含义：** 先和最接近、最强的可实现简单版本比较模块增量；不得借对旧基线的大幅提升掩盖强基线后的真实小增量。

### 40. Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/Jet-Nemotron- Efficient Language Model with Post Neural Architecture Search.pdf>) · [存档源](https://arxiv.org/pdf/2508.15884) · 20页全文文本 · 阅读者 `root`。

- **贡献：** 复用MLP并搜索full-attention位置、线性block及硬件受限宽度。
- **关键证据：** p4–6学习/均匀位置、六线性block和cache约束；p19对手继续同语料训练，最大batch/chunk搜索明确披露。
- **限制：** 400B token训练非toy；部分长上下文低于Qwen；本地附录4B架构与主文不一致；最大吞吐不等于相同batch。
- **当前含义：** 自适应位置需要强固定/均匀对照；强化对手可以合理，但资源与协议必须透明。

### 41. LEGO: Spatial Accelerator Generation and Optimization for Tensor Applications

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/LEGO- Spatial Accelerator Generation and Optimization for Tensor Applications.pdf>) · [存档源](https://arxiv.org/pdf/2509.12053) · 13页全文文本 · 阅读者 `github_audit`。

- **问题：** 固定加速器难以同时适应卷积/注意力算子的计算复用及通信需求，人工定制成本高。
- **贡献：** 控制/数据解耦、索引与复用结构搜索，以及基于原语DAG和线性规划的后端电路生成优化。
- **强与简单对照：** 匹配MAC数/SRAM/带宽的Gemmini；Eyeriss、NVDLA、TensorLib、DSAGen、AutoSA、SODA；朴素/优化融合和单独配置。
- **最终终点与证据：** 页11表V在相同313MB流量和487GOP/s条件下功率196→163mW，说明性能打平可有独立能效收益。
- **限制：** 页9使用RTL、综合和CACTI估计，未制造芯片；GPT2受内存限制，Llama7B batch1利用率仅3.1%。页11单独ICOC的某些能效还优于融合，并非融合通吃。
- **当前含义：** 另一个坐标必须真实测量或清楚标为模型估计；方法更复杂、动作更少本身不是端到端收益。
- **页表锚点：** [[9, "Evaluation"], [10, "Results"], [11, "TableV"]]

### 42. Long-RL: Scaling RL to Long Videos

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/Long-RL- Scaling RL to Long Videos.pdf>) · [存档源](https://arxiv.org/pdf/2507.07966) · 29页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 长视频推理怎样同时获得训练数据、阶段化RL收益以及容纳长序列的系统支持？
- **贡献：** 104K LongVideo-Reason数据、CoT-SFT→标准GRPO两阶段、复用视觉embedding和双阶段序列并行MR-SP。
- **强与简单对照：** 已有LongVILA及Video-R1/短视频VLM；p9Table5包括无SFT/RL、仅SFT、直接RL、外部数据替换、SFT+RL；p3Figure2系统比较无MR-SP、仅stage1、完整stage1+2。
- **打平与独立收益：** 并非靠同一准确率换说法：stage额外收益有直接对照，p9Table5原58.1、SFT60.2、直接RL52.4、外部SFT59.1/外部SFT+RL59.4、自己SFT+RL61.9；p8Table1 LongVideoBench57.1→58.0小但VNBench63.0→75.5、VideoMME60.1→65.1。系统2.1x与科学准确率分开。
- **限制：** 阶段对比不等于等总token/算力因果对照；自建评测/合成CoT可能存在标注偏差，正确选项不能证明推理忠实。p26自家示例文字既说戴戒指又说没戒指仍答C，说明定性宣传不能无审计接受。p5数据生成约80,000H100GPUh，2.1x测每step前向，不能称全项目成本降低2.1x。未见主结果多seed误差条，checklist自报Yes不替代实证。
- **可迁移判断：** 承认额外阶段可有害（直接RL52.4）并以simple stage-only对照量化增量；Curfew需要同LT/固定gate的独立增量，不可凭训练阶段故事立论。把数据、模型训练、系统开销三条账分别列。
- **不能类比处：** CoT-SFT初始化对RL优化的帮助不是后门形成窗口因果证据；标准GRPO并非新防御机制。
- **页表锚点：** ["p4-5data filtering/generation cost", "p8Tables1/2", "p9Tables3-5", "p15-21checklist", "p22-29qualitative appendix"]

### 43. LongVILA: Scaling Long-Context Visual Language Models for Long Videos

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/LongVILA- Scaling Long-Context Visual Language Models for Long Videos.pdf>) · [存档源](https://arxiv.org/pdf/2408.10188) · 20页全文文本 · 阅读者 `github_audit`。

- **问题：** 长视频SFT上下文和视觉token负载难以跨多GPU高效扩展；单纯注意力头并行存在规模限制。
- **贡献：** 五阶段长上下文训练与MM-SP两阶段图像/token负载均衡、二维上下文并行。
- **强与简单对照：** ZigZagRing、Ulysses、专家配置Megatron CP/TP、HF pipeline、FSDP/ZeRO3；单/两阶段切分与训练阶段消融。
- **最终终点与证据：** 速度可与Ulysses打平，但超出头数限制后能扩展到更长上下文/更多GPU，容量与可行性是另一个实测坐标。
- **限制：** 页8主效率比较只测LLM骨干，Megatron没有VLM支持；页17表7同规模有.39→.40s、10.70→11.12s的小退步。页16表5两阶段负载均衡单项仅1–7%，不能独占整套2.1–5.7倍收益。
- **当前含义：** 可扩展性贡献要有LT实际不能满足的资源/任务约束；不能把整体框架收益归到门控这一组件。
- **页表锚点：** [[8, "System benchmark"], [16, "Table5"], [17, "Table7"]]

### 44. LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/LServe- Efficient Long-sequence LLM Serving with Unified Sparse Attention.pdf>) · [存档源](https://arxiv.org/pdf/2502.14866) · 16页全文文本 · 阅读者 `github_audit`。

- **问题：** 长序列服务中稠密注意力和KV访问昂贵，朴素动态选择器自身也可能成为瓶颈。
- **贡献：** 静态流式头和动态检索头统一，分层KV页、量化与选择结果复用，并有实际稀疏GPU实现。
- **强与简单对照：** vLLM、QServe、MInference、DuoAttention、Quest（仅适用MHA）；静态/动态单独、选择器和复用周期消融。
- **最终终点与证据：** 页8表2大体保持LongBench质量并改善实际服务时延；页10表6以质量与复用成本选择周期4，而非无限少计算。
- **限制：** RULER256k：稠密79.4、4096预算75.7、8192预算79.1，质量不是所有预算都打平。页11图16图注4.5倍与正文7.7倍不一致，不引用该最大倍数。
- **当前含义：** 少调用/少MAC只是代理；必须评价固定规则与门控的总成本及漏防质量。这里有KV带宽瓶颈，不可直接移植到同零forward的LT比较。
- **页表锚点：** [[8, "Tables2–3"], [10, "Table6"], [11, "Fig.16; discrepancy"], [15, "Artifact"]]

### 45. QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/QServe- W4A8KV4 Quantization and System Co-design for Efficient LLM Serving.pdf>) · [存档源](https://arxiv.org/pdf/2405.04532) · 19页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 低bit减少字节/理论运算不等于大batch真实服务吞吐；怎样消除反量化瓶颈？
- **贡献：** QoQ渐进W4A8KV4量化与SmoothAttention，结合权重重排、寄存器并行和KV4 attention kernel实现。
- **强与简单对照：** 主对手明确TensorRT-LLM v0.9.0；FP16/W8A8/W4A16中取最好吞吐；Atom、QuaRot，质量另比RTN/AWQ/GPTQ-R。逐项算法与系统消融，且同batch和同显存最大batch两种比较。
- **打平与独立收益：** 质量接近但不全相同；LongBench平均38.52→38.38。独立目标是同显存实测吞吐p13Table4：A100 Llama3-8B3005 vsTRT最佳2503=1.20x，Qwen72B340vs143=2.38x。p14将Llama2-7B1.88x拆成同batch1.45x和batch增大1.3x。
- **限制：** 1024输入/512输出固定服务负载；最大可达吞吐非延迟SLA；设备/版本敏感。朴素KV4在A100反而慢1.1–1.2x，省内存不保证加速。附录artifact只复现效率，准确率明确另需deepcompressor。
- **可迁移判断：** 强简单baseline应取其有利且合法的配置；报告增量来自何处。Curfew若想靠成本立住必须在真实训练评测全账下证明，不能把mask数减少当收益。
- **不能类比处：** 整数表示和缓存工程问题无对抗者，不是门控准确性的理论先例。
- **页表锚点：** ["p10Table1", "p11Table2 and setup", "p12Table3", "p13Tables4/5", "p14Figures16/17 and breakdown", "p18-19 artifact appendix"]

### 46. Radial Attention: O(nlogn) Sparse Attention with Energy Decay for Long Video Generation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/Radial Attention- O(nlogn) Sparse Attention with Energy Decay for Long Video Generation.pdf>) · [存档源](https://arxiv.org/pdf/2506.19852v2) · 25页全文文本 · 阅读者 `root`。

- **贡献：** 固定径向时空稀疏模式减少视频注意力开销，并可直接用于长视频训练。
- **关键对照：** 同稀疏率/预算比较 SVG、STA、局部/窗口模式；短视频质量与速度之外，长视频2x/4x训练比较 dense/LoRA/多种mask；p19对warmup/整体训练计算作匹配说明。
- **关键证据：** 默认静态模式与动态SVG效果接近，不以自适应为必要；贡献另有实际可训练性及长视频扩展证据。
- **限制：** 不是每个设置最快最好；STA某些实现更快。p18证明中 w(Delta t)>=s/2 的不等式在给定定义下不成立（s=8,Delta t=4给w=2），不照搬该证明；固定空间token数前提下线性时长复杂度仍可另推。
- **当前含义：** 固定规则有效时可采用固定规则；若保留复杂设计，必须给出另一个真实、独立的能力或成本收益，不能把论文作者身份当证据。

### 47. SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/SANA- Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers.pdf>) · [存档源](https://arxiv.org/pdf/2410.10629v3) · 23页全文文本 · 阅读者 `root`。

- **贡献：** 深压缩AE、线性DiT/MixFFN、decoder-only文本编码、caption与solver共同降低高分辨率成本。
- **关键证据：** p4同token数AE/patch对照及重建与生成排序相反；p9/p20模块分解；p17高分辨率重新引入PE。
- **限制：** 本地v3有draft TODO/图表张力；NoPE并非所有分辨率；INT8质量小降；不同口径速度不可拼接。
- **当前含义：** 复合工程法也需逐项用途和最终收益，不能用简单代理或整套成功证明每个部件。

### 48. SANA-1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/SANA-1.5- Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer.pdf>) · [存档源](https://arxiv.org/pdf/2501.18427v4) · 21页全文文本 · 阅读者 `design_audit`。

- **问题：** 线性DiT扩大参数训练成本高；推理更多denoising steps早饱和；同一大模型如何低成本衍生不同部署规模。
- **贡献：** 从预训练1.6B增长到4.8B并选稳定partialpreservation初始化；CAME8bit优化器；按输入输出余弦重要性剪层与100步恢复；重复采样加专门训练VILA-Judge做bestofN。
- **强与简单对照：** 同4.8B从头训练vs增长，三种初始化含NaN失败；CAME/AdamW的8/32bit；同尺寸SANA1.0vs剪枝FT；denoisingstepsvs采样数；小/大模型多采样；SD1.5加同类推理扩展展示收益非SANA独有；FID/CLIP/DPG与GenEval分别报。
- **打平与独立收益：** CAME8bit与32bit收敛曲线近似，但完整显存57→43GB(25%)，并非optimizer1/8等于总显存1/8。同0.70GenEval新增长训练步少60%，该计量包含预训练小模型的成本与否未明确，不能当从零总成本已严格结算。剪枝1.6B+FT 0.672vsSANA1.0 0.664差小，其价值包含从同一大模型衍生多尺寸。
- **限制：** 0.96GenEval用2048候选+额外2Bjudge；表2竞品主要单样本外部数字，不能作同预算优势。Judge训练约2M图文、GenEval样式标签；SFT144291图也GenEval样式但排除测试prompt；存在distribution specialization风险，通用DPG从85.0降至84.7、FID5.42变5.99，GenEval更好非所有质量更好。Eq2把24写作bytes/parameter可疑，不照抄；BF16/FP16速度说明有不一致。安全‘完美过滤’断言没有可审计覆盖不采信。
- **可迁移判断：** 平台约束明确时资源与质量可分开报告；简单初始化比复杂复制稳定就选简单；以多指标暴露优化某一基准的代价，不能只看胜格。
- **不能类比处：** 2048倍采样加judge是显式昂贵预算交易，不是新复杂模块在同资源下胜出；GenEval style训练和更换终点不可当Curfew平分后任意另找分数的先例。
- **页表锚点：** [{"page": 3, "item": "Figure2 and initialization", "finding": "同GenEval0.70少60%新增4.8B训练步；采用最简单稳定初始化"}, {"page": 6, "item": "Tables1-2", "finding": "Pre→Ours GenEval.72→.81，FID5.42→5.99、DPG85.0→84.7；2048样本.96"}, {"page": 7, "item": "Table3, optimizer comparison", "finding": "1.6B剪枝FT.672vs原版.664；显存57GB→43GB"}, {"page": 8, "item": "highquality finetuning/data", "finding": "GenEval样式SFT144291图，testprompts排除；新增块不学习后删尾2block"}, {"page": 9, "item": "Inference-time scaling cost", "finding": "N×49140GFLOPs生成+2N×4518GFLOPs判别，效率留后续"}, {"page": 10, "item": "AppendixB", "finding": "Judge额外2.5M图、15654prompt与GenEval风格标签"}, {"page": 11, "item": "AppendixB-C", "finding": "Judge一epoch训练，8/device，LR2e-5；额外RMSNorm stabilizer"}]

### 49. SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/SANA-Sprint- One-Step Diffusion with Continuous-Time Consistency Distillation.pdf>) · [存档源](https://arxiv.org/pdf/2503.09641v4) · 22页全文文本 · 阅读者 `root`。

- **贡献：** FM到TrigFlow坐标转换，稳定教师后结合sCM与LADD作1–4步蒸馏。
- **关键证据：** p4转换教师；p5稳定性；p8两损失及组合；p14–16噪声/采样搜索；转换本身免训练。
- **限制：** 仍需5k教师微调和20k蒸馏；64.7x仅Transformer；本地v4摘要/主表部分指标不一致。
- **当前含义：** 先对齐坐标和协议再解释阶段；免费变换不等于整方案免费，成本分母需核实。

### 50. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/SVDQuant- Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models.pdf>) · [存档源](https://arxiv.org/pdf/2411.05007v4) · 28页全文文本 · 阅读者 `github_audit`。

- **问题：** 扩散模型W4A4受权重/激活离群值影响，低秩分支虽改善量化却可能引入显存访问和延迟。
- **贡献：** 平滑离群值、低秩分支吸收主谱、量化残差，Nunchaku融合两路内核以兑现硬件收益。
- **强与简单对照：** NF4 W4A16、ViDiT-Q、MixDQ、TensorRT、naive INT4、平滑、SVD-only、LoRC；附录低秩×GPTQ的关键消融及rank/bit数扫描。
- **最终终点与证据：** 质量接近高精度时，真正收益是内存/硬件延迟。页25表3也暴露低秩在强GPTQ基线上并非所有模型/bit格式都更好。
- **限制：** 页25表3 FLUX-dev NVFP4+GPTQ无低秩LPIPS .204/PSNR21.5；加低秩.208/21.4，IR .936/.937，不能概括为全面增益。页27表4的朴素INT4单步212ms比融合SVD的218ms还快但质量不同。
- **当前含义：** 强简单对手可能已解决主要问题；必须单列门控增量，不能只相对朴素攻击训练展示整体提升，也不能在Curfew/LT同零forward时借用硬件效率叙事。
- **页表锚点：** [[6, "Fig.6"], [8, "Table1"], [19, "AppendixD"], [25, "Table3; rendered"], [27, "Table4"]]

### 51. VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/VILA-U- a Unified Foundation Model Integrating Visual Understanding and Generation.pdf>) · [存档源](https://arxiv.org/pdf/2409.04429v3) · 19页全文文本 · 阅读者 `root`。

- **贡献：** 对齐且可重建的残差量化视觉tokenizer统一理解与生成。
- **关键证据：** p9相同模型单组件；p19四个失败训练配方；p6指出中间重建分数不等于最终生成；p10承认未显示显著协同。
- **限制：** contrastive对齐会伤重建/生成；并非所有任务最好；内部/大规模训练数据费用。
- **当前含义：** 可证明的功能整合与尚未证明的协同机制分开；整体有效不能替gate增量作证。

### 52. XAttention: Block Sparse Attention with Antidiagonal Scoring

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2025/XAttention- Block Sparse Attention with Antidiagonal Scoring.pdf>) · [存档源](https://raw.githubusercontent.com/mlresearch/v267/main/assets/xu25ag/xu25ag.pdf) · 13页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 长上下文prefill如何以便宜且可靠的块重要性代理实现稀疏注意力？
- **贡献：** 跨步反对角线采样估计块重要性；累计概率阈值选块；可选逐head最低阈值预测，并实现稀疏kernel。
- **强与简单对照：** 官方公开MInference/FlexPrefill、SeerAttention和全注意力；Table7同采样开销的随机/对角线/反对角线，Table9静态TopK/比例与动态阈值，Table10固定阈值与预测阈值。
- **打平与独立收益：** 视频质量并非处处赢（Table3总体略低）；论据是限定误差下实测速度和稀疏度。动态部分有直接静态对照：Table9 stride8动态88.47/密度20.97%，静态比例85.42/21%；Table10固定阈值84.96/26.13% vs预测88.47/20.97%。
- **限制：** 13.5x是attention kernel而非全任务；Table6端到端prefill最高5.12x。M=1000代表探索1000个阈值组合；搜索/校准成本需另计，不是1000训练样本。视频只有全注意力而无其他稀疏baseline。Appendix A Table11阈值与密度方向和说明冲突，未用作定量依据。
- **可迁移判断：** 门控增量需同预算固定阈值/固定比例/TopK直接对照；可选组件若打平可放弃，不能用核心方法的成功替可选组件背书。
- **不能类比处：** 注意力稀疏的概率质量保留不等于后门检测，近似误差不会自动给攻击稳健性保证。
- **页表锚点：** ["p4 Table1", "p5 Table2", "p6 Tables3/4/6", "p7 Tables7/9", "p8 Table10", "p13 AppendicesA/B"]

## 2026

### 53. Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/Fast-dLLM- Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding.pdf>) · [存档源](https://arxiv.org/pdf/2505.22618) · 22页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 双向diffusion LLM怎样复用KV并并行解码而控制条件独立采样导致的质量损失？
- **贡献：** block-wise近似Prefix/Dual KV缓存，置信阈值并行解码与factor规则；理想一致联合分布下高置信贪心等价/分布偏差界。
- **强与简单对照：** 原LLaDA/Dream、只cache、只parallel、组合；p8Figure5和p19Figure8直接对固定2/4/8token每步，以平均token/step与NFE比较；p17Table11factor还对更简单固定阈值。
- **打平与独立收益：** 质量接近时靠实测端到端解码速度立住；p7Table1 GSM8K256原79.3/6.7tok/s→组合78.5/54.4，8.1x。27.6x限长prefill+1024上限且准确77.3→76.0。factor对固定阈值并非质量更优：p17Table11GSM8K51274.8vs77.2，换35.3→47.1tok/s，需作为tradeoff。
- **限制：** p14证明假设可一致导出边缘的联合分布，实际MDM不严格满足；高置信不代表正确。batch1收益不能推大batch，p19-20承认大batch仍落后AR。视觉任务必须改为refresh缓存，小block掉8pp，适配需明示。
- **可迁移判断：** 自适应增量最有说服力的是同平均预算固定规则直接对比与全质量-成本曲线；更复杂factor若只是换tradeoff不能说全面优于固定阈值。理论必须写明理想化与真实门控信号差距。
- **不能类比处：** 模型token置信筛并行执行与后门可疑样本删除不同；论文不证明投毒高loss或训练阶段门控有因果必要。
- **页表锚点：** ["p6Theorem1/Algorithm1", "p7Table1/setup", "p8Figure5", "p9Tables4/5", "p14assumption remark", "p16-17Tables9-11", "p19Figure8", "p19-20C.5"]

### 54. Fast-dLLM v2: Efficient Block-Diffusion Large Language Model

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/Fast-dLLM v2- Efficient Block-Diffusion Large Language Model.pdf>) · [存档源](https://arxiv.org/pdf/2509.26328) · 17页全文文本 · 阅读者 `design_audit`。

- **问题：** AR顺序解码慢，full-attention dLLM需大量适配数据且缓存近似/生成长度受限。
- **贡献：** blockdiffusion近AR注意力+互补mask/token shift；跨block精确cache、block内DualCache与confidence阈值。
- **强与简单对照：** 同Qwen2.5-Instruct同数据/步数NTP再训对照；Dream/LLaDA；naive/+pad/+CM；block/subblock失配；cache开关、batch1/32与A100/H100；threshold曲线。
- **打平与独立收益：** 总体均分近AR(7B60.3vs同数据NTP59.6)，用独立吞吐终点支撑；但主accuracy默认threshold1，速度threshold.9，不能把全部主表质量与最高速度直接拼成同设置无损。单任务有Math/IFEval退步，论文平均和matchedGSM8K速度表不能替代全面无损。
- **限制：** 约1B只适用于7B实际1.31B，1.5B用3.15B tokens（附录13）；DualCache近似继承，blockcache精确二者别混；最高速度按任务/阈值/硬件条件，训练开销也有64A100×12h。
- **可迁移判断：** 既要matched训练对照，也要同设置质量-成本点，保留退步和阈值选择；不能拿一个配置质量+另一个速度拼Pareto。
- **不能类比处：** 当前Curfew对LT没有已测2.5×吞吐差；同主终点打平本身不赋予贡献。
- **页表锚点：** [{"page": 6, "item": "Sec4.1", "finding": "matchedNTP控制及accuracy默认threshold1"}, {"page": 7, "item": "Table1/Fig4/5", "finding": "7Bavg60.3vs59.6，最高速度使用threshold.9"}, {"page": 8, "item": "Table2", "finding": "naive41.3→pad42.2→CM45.0平均"}, {"page": 9, "item": "Tables3/4 Fig6", "finding": "cache开关/粒度和失配控制"}, {"page": 13, "item": "AppendixA1", "finding": "1.5B3.15B训练tokens；7B1.31B，非都1B"}, {"page": 15, "item": "A4", "finding": "主benchmark无parallel threshold1"}]

### 55. ForeAct: Steering Your VLA with Efficient Visual Foresight Planning

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/ForeAct- Steering Your VLA with Efficient Visual Foresight Planning.pdf>) · [存档源](https://arxiv.org/pdf/2602.12322v1) · 16页全文文本 · 阅读者 `github_audit`。

- **问题：** VLA复杂长程/域外任务需要具体空间目标，文本规划可能缺乏足够的视觉可执行信息。
- **贡献：** VLM拆分子任务、SANA派生生成器产生视觉未来目标，增强下层VLA；云端/边缘分工。
- **强与简单对照：** pi0、匹配文本规划的VLM+pi0、pi0.5，LIBERO的OpenVLA/CotVLA/CogVLA；空间文本、语义文本和视觉目标消融。
- **最终终点与证据：** LIBERO近饱和97%左右时，主要增量来自真实域外任务；页9表4视觉目标93.4 vs空间文本46.8是具体替代解释控制。
- **限制：** 页7的87.4/57.1/46.5是带惩罚和最多5次尝试的原子动作完成指标，不是整段episode成功率；800k步64H100预训练。生成器.33s不等于整条机器人系统成本。
- **当前含义：** 已有任务打平可动机化检查确实存在的新失败模式；需匹配强简单替代控制，不得因为换到更难设置就自动认定门控成立。
- **页表锚点：** [[4, "Dataset/training"], [7, "Table1"], [8, "LIBERO"], [9, "Table4"]]

### 56. LongLive: Real-time Interactive Long Video Generation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/LongLive- Real-time Interactive Long Video Generation.pdf>) · [存档源](https://arxiv.org/pdf/2509.22622v2) · 21页全文文本 · 阅读者 `github_audit`。

- **问题：** 长视频生成依赖过去KV而持续漂移；提示词切换时保留旧KV和清空KV分别损害控制或连贯性。
- **贡献：** 提示切换时KV重缓存、局部长视频训练教师、短窗口加frame sink；LoRA与INT8实现实时运行。
- **强与简单对照：** Wan、LTX、SkyReels、MAGI、CausVid、NOVA、PyramidFlow、Self Forcing、FramePack；保留/清空/重缓存，窗口、rank、量化消融。
- **最终终点与证据：** 短片质量接近时另有长视频稳定性、提示切换和FPS收益；页8表4重缓存是连贯性/提示对齐之间折中。
- **限制：** 重缓存CLIP27.87低于清空28.95，不能写全指标获胜。短视频语义得分低于Self Forcing；教师64H100×12h等成本不能从FPS中消失。
- **当前含义：** “更稳定”须有独立的过程/长程结果和简单对照。若LT轨迹也相同，稳定性换名并不产生新贡献。
- **页表锚点：** [[7, "Table1"], [8, "Table4"], [9, "Window ablation"], [16, "Training details"], [17, "LoRA/INT8"]]

### 57. Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation.pdf>) · [存档源](https://arxiv.org/pdf/2507.01957v2) · 24页全文文本 · 阅读者 `server_raw_audit`。

- **问题：** 保留flat token和KV缓存的前提下，如何高并行生成图像并不破坏同组依赖？
- **贡献：** 独立位置query与context角色、同组query互见的attention mask；基于邻近已知context而远离同组token的可预计算空间顺序。
- **强与简单对照：** 自建同设置raster counterpart和公开RandAR/ARPG/NAR等；Figure9架构对比统一random order，顺序对比固定架构；random/Halton/仅原则1/仅原则2/两者。
- **打平与独立收益：** p9Table2 512px XL FID2.09vs2.10近乎打平，延迟20.93s→1.01s实测构成独立贡献；不靠声称复杂规则天生更好。p9–10Figure9c random2.11、原则1=2.00、原则2=2.06、组合1.92，直接分辨每个启发式增量。
- **限制：** 生成顺序可预计算，不能称在线自适应。额外query增加FLOPs，batch>16开始计算瓶颈，附录E最大batch收益约3x而非12x。appendix Table4模型规模与正文、batch信息互相不一致；Table6τ/ρ文字与代码变量含义疑似反转，引用时需标版本瑕疵。
- **可迁移判断：** 这是支持简单固定规则的正例；结构/时机原则有图像相关性还不够，必须同一实现单独消融和同终点评估。Curfew门控对固定LT若无增量，不能借规则解释完整就判定其必要。
- **不能类比处：** 二维空间邻近给天然可观测结构，entity分组与训练阶段并无同等已证依赖；统一视觉模型兼容性也非Curfew现有已测终点。
- **页表锚点：** ["p3-6Section2", "p8Table1", "p9Tables2/3 Figure9", "p10principle ablation", "p17-20AppendixA/B", "p24AppendixE"]

### 58. QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/QeRL- Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs.pdf>) · [存档源](https://arxiv.org/pdf/2510.11696) · 21页全文文本 · 阅读者 `design_audit`。

- **问题：** RL大模型完整训练或BF16 LoRA仍昂贵；低比特量化既可减内存也可能以噪声改变探索。
- **贡献：** NVFP4+LoRA及Marlin rollout，量化噪声提高探索的实证；AQN按预定时间指数衰减噪声，RMSNorm行级等价推导。
- **强与简单对照：** full BF16/LoRA/QLoRA/NVFP4与MXFP4；AQN开关、噪声衰减调度、LoRA rank；GRPO与DAPO、多模型规模；匹配LR补充；分解rollout与全训练时间。
- **打平与独立收益：** 7B BigMath无AQN平均37.0，高于加AQN36.4；14B40.5→42.0和32B41.4→45.6才显著提升。Figure10 rank16与32等曲线相近时正文承认较低rank更经济。3B rollout几乎不加速；32B则单H100 BF16 OOM而NVFP4可行。
- **限制：** 主结果accuracy训练用8 H100而speed单卡，不能拼成所有accuracy都单卡完成；LoRA LR5e-6，量化方法1e-5，附录matchedLR只补部分归因；AQN为固定随时间计划，并非基于观测门控；内存15.2/5.9GB是7B模型权重，不是完整峰值；数学3B-32B，不覆盖通用任务和70B以上。
- **可迁移判断：** 必须单独检验增添模块是否优于核心简化版，保留无增益甚至负增益格；只有实测资源/可行性差异才能成为平分后的贡献。
- **不能类比处：** 量化独立具备权重内存收益，不能替Curfew虚构资源优势；按时间降噪不等于自适应gate。噪声与探索的相关轨迹不证明Curfew机制。
- **页表锚点：** [{"page": 7, "item": "Table1; training setup", "finding": "7B GSM8K Full91.2，BF16LoRA88.1，NVFP4 88.5，+AQN90.8；完整accuracy训练8H100"}, {"page": 8, "item": "Table2/Figs9-10", "finding": "7B平均无AQN37.0/+AQN36.4，14B40.5/42.0，32B41.4/45.6；rank低值接近"}, {"page": 16, "item": "Table4", "finding": "LR差异、噪声起始1e-2和终止5e-4，AWQ校准256条×2048来自OpenThoughts"}, {"page": 17, "item": "AppendixF-G", "finding": "算法全流程与RMSNorm噪声等价推导"}, {"page": 19, "item": "Table5/AppendixI", "finding": "3B几乎不加速；匹配5e-6LR补充比较"}, {"page": 20, "item": "Table8", "finding": "32B单H100 BF16 OOM，QeRL可训练；model size62.3/20.7GB"}, {"page": 21, "item": "AppendixK", "finding": "数学领域与3B-32B规模限制"}]

### 59. SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/SANA-Video- Efficient Video Generation with Block Linear Diffusion Transformer.pdf>) · [存档源](https://proceedings.iclr.cc/paper_files/paper/2026/file/41b93c59da0d0f835907fd661d419db2-Paper-Conference.pdf) · 25页全文文本 · 阅读者 `design_audit`。

- **问题：** 高分辨率长视频tokens使fullattention算力与KVcache随长度增加；同质量下小模型需真正提升推理与长视频可行性。
- **贡献：** ReLU之后RoPE且分母不加RoPE防不稳；spatial-temporalMixFFN；blockcausallinearattention累积state固定cache；递增SNR训练与longselfrollout；DCAE-V深压缩、分阶段数据/训练、NVFP4部署。
- **强与简单对照：** 同模型linear/fullattention实际延迟；3DRoPE/temporalconv有无与random/increasingstep；Wan2.1/2.2大小模型、SkyReel/CausVid/SelfForcing等；VBench与VBenchLong；不同VAE及noiseperturb；BF16/NVFP4同RTX5090；SFT有无定性。
- **打平与独立收益：** 短视频83.70低于SelfForcing84.31(Table6)，不应说所有质量SOTA；30秒Long评估82.29 vsSelfForcing81.59且27.5vs17FPS，长工况总分与速度有实际收益；Quality83.10仍低于SelfForcing83.82。720p VBench total84.05和Wan2.1-14B83.73近，但H100 latency36 vs1897s；相同压缩Wan2.2-5B116s。
- **限制：** 64H100×12天是继续预训练pipeline，另用预训练SANA/10M内部图和视频与caption，不是所有资产从零总cost；‘MovieGen1%’非严格同数据/模型匹配比较。Table4defaultsteps，Fig1说50steps，不能拼接16x最高倍数；延迟各场景480p/720p不同VAE，明确口径。VAE无扰动DCAE33.25PSNR输Wan35.61，有扰动胜仅是受控proxy非任意分布鲁棒证明。‘全局记忆’是压缩状态，不是无损保留所有历史信息；worldmodel仅可视化无闭环控制验收；部分内部数据与未量化variance限制。
- **可迁移判断：** 当前工况相近时可以检验原本就需要的更长或更难工况，但必须质量、速度、资源成对公布，并保留子指标退步；复杂组件用同模型开关消融。
- **不能类比处：** 固定缓存有可计算复杂度与实测内存优势，Curfew未建立这种独立优势；不能因为某论文长视频胜就临时换Curfew任务或夸大同资源胜出。
- **页表锚点：** [{"page": 5, "item": "Table1/Eq2", "finding": "attention复杂度定义与RoPE分母稳定性修补"}, {"page": 7, "item": "Tables2-3", "finding": "720p total84.05/83.73、36/1897s；Wan2.2同压缩116s；DCAE重建PSNR33.25<35.61"}, {"page": 8, "item": "Table4", "finding": "480p默认steps，T2V83.71<OpenSora84.34，速度60vs465s"}, {"page": 9, "item": "Tables5-6", "finding": "同组件消融；短视频SelfForcing84.31>SANA83.70"}, {"page": 10, "item": "NVFP4 deployment", "finding": "RTX5090 720p71→29s，有些layer保持高精度"}, {"page": 17, "item": "AppendixC2", "finding": "10M内部图、10M视频、多阶段训练量与资源"}, {"page": 18, "item": "Algorithm1/C3", "finding": "累积state/keys与lastframecache，非原始历史KV"}, {"page": 19, "item": "Table8", "finding": "噪声0/0.1/0.2 VAE重建对照"}, {"page": 22, "item": "Table9", "finding": "30秒SANA total82.29 vs81.59,FPS27.5vs17；quality83.10<83.82"}, {"page": 25, "item": "AppendixF", "finding": "worldmodel三应用只是finetune与可视化，无下游闭环质量结果"}]

### 60. StreamingVLM: Real-Time Understanding for Infinite Video Streams

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/StreamingVLM- Real-Time Understanding for Infinite Video Streams.pdf>) · [存档源](https://arxiv.org/pdf/2510.09608) · 13页全文文本 · 阅读者 `github_audit`。

- **问题：** 流式视频的全注意力上下文/时延增长，无重叠窗口丢历史，有重叠窗口重复计算。
- **贡献：** 重叠短片训练、有限KV保留（sink/文本/近期视觉）与连续RoPE，构建4000小时训练和长流评测。
- **强与简单对照：** 全注意力、无重叠窗口、有重叠重计算、KV复用；LiveCC、GPT4o-mini、ReKV；同数据非重叠SFT及窗口/RoPE/数据消融。
- **最终终点与证据：** 页2图1重叠重计算与KV复用对GPT4o-mini的win share为66.54%/66.18%，质量接近；复用减少重复计算并保持低而稳定的时延。
- **限制：** 20场平均2.12小时的体育视频与GPT5裁判不证明任意无限记忆；页9表7非重叠SFT VideoMME65.50还略高于本文65.10。128H100多天训练成本存在。
- **当前含义：** 最清楚的“质量打平但仍成立”：确实解决冗余计算的独立部署瓶颈。Curfew/LT同零额外forward没有该现成优势；不能只类比标题里的实时/稳定。
- **页表锚点：** [[2, "Fig.1"], [5, "Evaluation"], [7, "Table3"], [9, "Table7"], [12, "AppendixA"]]

### 61. Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/Taming the Long-Tail- Efficient Reasoning RL Training with Adaptive Drafter.pdf>) · [存档源](https://arxiv.org/pdf/2511.16665v1) · 16页全文文本 · 阅读者 `github_audit`。

- **问题：** 推理RL的长尾rollout使同步训练等候；batch变化和模型漂移削弱静态推测解码。
- **贡献：** 闲置时段训练轻量drafter，按负载选择推测/非推测策略，配合分桶CUDA Graph控制内存。
- **强与简单对照：** Open-R1、VeRL、TLT-Base；单图/朴素多图/分桶图、无SpotTrainer、不同更新周期和batch。
- **最终终点与证据：** 页10报告相对VeRL的1.7–2.1倍吞吐，奖励曲线相近；页11表2显示最佳策略随batch而变，支持自适应的实际必要性。
- **限制：** 页9计时为已训练drafter、1次预热后3步；不是从零开始的总成本承诺。页10明确accept length不能代替整体加速。100步奖励曲线不证明任意训练质量都相同。
- **当前含义：** 与门控最相近的是固定策略在不同负载区间确有成本失配。Curfew必须先有LT规则失配的证据，不能把“自适应”本身当优势。
- **页表锚点：** [[9, "§6.1"], [10, "Figs.11–13"], [11, "Tables2–3"], [12, "§6.4"]]

### 62. VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference

来源：[本地PDF](</Users/ruizhixu/Desktop/Han Lab Publication/2026/VLASH- Real-Time VLAs via Future-State-Aware Asynchronous Inference.pdf>) · [存档源](https://arxiv.org/pdf/2512.01031) · 10页全文文本 · 阅读者 `github_audit`。

- **问题：** 同步VLA高延迟，朴素异步执行时图像/机器人状态过时，影响动态任务反应。
- **贡献：** 利用已知待执行动作roll forward未来机器人状态，训练时加入偏移状态/动作；共享观测token的多分支训练及可选量化。
- **强与简单对照：** Sync、naive async、RTC；Kinetix/LIBERO/真实机器人，延迟/执行长度/量化和共享观测消融。
- **最终终点与证据：** 静态LIBERO成功率可近似打平，动态任务反应和延迟另有改善；页7乒乓11/20 vs对照0/20。
- **限制：** 反应时间D+2L→2L是最坏情况公式，不是所有场景实际反应分布。页8一步420.99→129.29ms却存在10k步94.1vs87.1的收敛差异；需看time-to-quality。真实pi0.5配置有可选projection，不能绝对说零模型改动。
- **当前含义：** 任务静态打平可以由实际动态终点支撑；若现有LT与门控过程也打平，就没有这样的第二收益。
- **页表锚点：** [[5, "TableI"], [7, "TablesII–III"], [8, "TableIV"], [10, "Appendix"]]
