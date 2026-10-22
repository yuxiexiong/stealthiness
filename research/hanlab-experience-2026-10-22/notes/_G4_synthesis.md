# G4 组综合（GAN/扩散压缩与加速、可控生成、VILA 系列）

覆盖 11 篇：vila-u, distrifusion, sige, gancompression, anycost-gan, fastcomposer, diffaug, longlive, can, vila, longvila。
每条套路后标注出现位置；"≥2 篇"才算共同套路。档位标注：**读到**＝论文里明确这样做；**归纳**＝我从做法中总结。

---

## A. 跨论文共同的实验设计套路

### A1. 用"两个朴素极端 + 本方法"的三臂设计立住两难（读到）
先展示两种显而易见的做法各自在一个指标上失败，再给本方法两头兼顾。
- distrifusion：Naive Patch（无交互→FID w/ Orig. 259）vs Sync. TP（通信量 1.33–18.7G）→ Table 1, Table 2；GN 四方案 Fig. 9
- diffaug：Augment reals only（伪影）vs Augment D only（G/D 失衡）→ Table 1, Fig. 5
- longlive：无 KV cache（突变）vs 保留 cache（prompt 惯性）→ Table 4, Fig. 3
- sige：Patch（缺全局上下文）vs Crop（不规则区域浪费）→ Table 1/2
- anycost-gan：same D vs reduced-ch D vs 条件 D → Table 2

### A2. 质量指标双口径：相对真值 + 相对原模型（读到）
加速/压缩不只看"分布质量"，还看"与原模型输出偏离多少"，后者对近似误差更敏感。
- distrifusion Table 1（w/ G.T. 与 w/ Orig.）
- sige Table 1、附录 Table 5（同上，另加编辑区单独评测）
- anycost-gan Fig. 5b（LPIPS Diff：子网 vs 全网）、Table 4（属性匹配率）

### A3. 理论计算量与实测延迟并列，并明说两者不等（读到）
- distrifusion Sec. 5.2（"on-hardware acceleration is more critical"；MACs 与延迟同表）
- sige Table 2/3（MACs 降 7.6× 而延迟仅 1.6×，再逐步优化到 2.9×）
- gancompression Table 4（MACs 9–21× vs GPU 1.7–3.3×，诚实写出 GPU 加速小）
- anycost-gan Table 5（超线性加速 + cache 假说）
- can Table 4/5（MACs/步 + Orin TensorRT 一个点）
- longvila 附录 Table 10（组件级延迟与 TFLOPs）

### A4. 跨硬件谱系，特别是边缘设备（读到）
- sige Table 2（3090、2080Ti、i9、M1 Pro CPU/GPU）+ Table 4（TensorRT）
- gancompression Table 4（Jetson Xavier/Nano CPU+GPU、1080Ti、Xeon）
- anycost-gan Table 5（Xeon、Jetson Nano）
- can Table 5（Jetson AGX Orin）
- （vila 摘要声称 Orin 可部署但正文无数据——反例）

### A5. 公开测量协议（预热次数 + 重复次数）（读到）
- distrifusion Sec. 5.1（3 预热 + 10 次，去最快最慢；CUDAGraph 同时用于基线）
- sige Sec. 4.2（GPU 200+200；CPU 10+10 重复 5 次）
- gancompression Sec. 4.3.5（100+100，batch=1）
- longvila Sec. 5.1.1（10 预热 + 5 次）
- 未交代协议的：anycost-gan、can、longlive、fastcomposer

### A6. 沿"问题规模"扫描，并暴露盈亏/适用边界（读到）
- distrifusion Fig. 6（分辨率 1024→3840；局限节承认低分辨率利用率差）
- sige Fig. 10（编辑比例；3090 上 ~70%、i9 上 ~45% 为盈亏点）
- longvila Fig. 9 / Table 7/8（序列长度、GPU 数到 OOM）
- longlive Fig. 7（窗口 3–27）
- fastcomposer Fig. 5（延迟比例 α 的权衡曲线）
- diffaug Table 3/4（数据量 1k–30k、10–100%）

### A7. "性能–成本"多点曲线代替单点比较（读到）
- gancompression Fig. 7/11/12（FID/mIoU vs MACs）
- anycost-gan Fig. 5（FID/LPIPS vs MACs，单模型对多模型）
- can Fig. 1/7（FID/CLIP vs GMACs）
- diffaug Fig. 8a（不同模型容量下的 FID）

### A8. 把方法依赖的关键观察做成内部量的可视化/量化（读到）
- distrifusion Fig. 7 + Sec. 5.3（相邻步输入差 0.02，≈0.3%）
- sige Fig. 4/6（逐层 |ΔA| 与 attention 输入/输出差图）
- fastcomposer Fig. 4（cross-attention 图：两个 token 注意同一人）
- diffaug Fig. 1/6（D 训练/验证准确率差）
- vila Fig. 3（逐层视觉/文本嵌入相似度）
- anycost-gan Fig. 4（中间层输出、无一致性损失时子网不一致）

### A9. 显式排除替代解释（读到）
- diffaug Fig. 8（"只是模型太大？""只是正则不够？"——扫描模型容量、R1 γ）
- can Fig. 4（"只是动态参数化？"→核选择）、Fig. 8（"只是收敛更快？"→长训练曲线）
- vila Table 3 + Sec. 4.4（MMC4-pairs：结构 vs 文本分布；打乱顺序）
- gancompression Table 7, Fig. 12（输出蒸馏 vs 中间特征蒸馏）

### A10. 逐步累加（roadmap）消融（读到）
- sige Table 3（Sparse → +Norm → +Elem → +Sct → +Attn，每步给延迟）
- can Table 1（DW Conv → +Patch Emb → +Out Proj，附被否决分支 ✗）
- gancompression Table 5（剪枝/蒸馏/继承 D 组合；文本损坏，部分不可核）

### A11. 正交性/可叠加性论证（读到）
- sige（+GAN Compression 达 40–50× MACs，Table 1/2）
- diffaug（+CR-BigGAN、+TransferGAN、+FreezeD，Table 4/5）
- can（+AdaNorm / +Attention，Table 3）
- longlive（+INT8 PTQ，附录 Table B）
- distrifusion（+CFG batch 拆分，仅一句话）

### A12. 强化基线并对齐评测协议（读到）
- gancompression：重训原模型并与官方权重比、取更强者（附录 Table 10）；按对手协议重算 FID（Sec. 4.3.1 脚注）
- diffaug：给所有基线加翻转、BigGAN 复现优于原文（7.6 vs 8.7）；按并发工作协议补算（Table 4 注）；坦白复现不了 CR-BigGAN 的 6.66（附录 A.1）
- longvila：Megatron 按其团队推荐的 CP+TP 混合配置（Sec. 5.1.1）
- vila / vila-u：在 SOTA 表中用 * 标出训练集可见的基准（vila Table 5，vila-u Table 2）

### A13. 训练/定制成本本身作为指标（读到）
- gancompression Fig. 8（Fast 版训练 1.6–3.7×、搜索 3.5–12× 省时）
- anycost-gan 附录 Fig. 10（多设备部署 GPU 小时）
- fastcomposer Table 1/2（定制总时间与峰值显存）
- longlive（32 GPU-days）、vila 附录 B（5.1k GPU 小时）

### A14. 定性图逐格标注数字（读到）
每个面板都写 MACs/延迟/FID 或 mIoU，让定性图也承载定量信息。
- distrifusion Fig. 1/5/8/9；sige Fig. 1/8/9/11–13/19–22；gancompression Fig. 5/6/13–18；diffaug Fig. 12–14/18–22

### A15. 人评 / 用户研究（读到，仅 2 篇）
- gancompression Table 3（AMT，72.4% 偏好）
- longlive Fig. 1 右 + 附录 L（26 人 × 48 题）

---

## B. 本组特有的套路（相对一般效率论文）

1. **"一致性"作为一等指标**（生成模型压缩特有）：不只要求好，还要求和原模型/全模型"像"——w/ Orig. 指标、LPIPS Diff、属性匹配率（distrifusion、sige、anycost-gan）。这是生成任务没有唯一正确答案时的替代"精度"。**归纳**
2. **"交互性"作为动机的产品化叙事**：交互编辑（sige、gancompression、anycost-gan）、交互 prompt 切换（longlive）、预览渲染类比（anycost-gan）——痛点用"用户等待"而非吞吐来定义。**归纳**
3. **自建合成基准**：sige（Detic+CoModGAN 合成编辑、粘贴物体、随机圆遮罩）、fastcomposer（15 名人 × 30 prompt）、longlive（Qwen 生成 160 条交互 prompt）、longvila（VILA 打 caption + LLM 出 QA；LongVILA-Caption 用 GPT-4 打分）。**读到**
4. **VLM 线的"守恒指标"**：加视觉能力的同时报告纯文本能力（vila Table 3/4/6 的 MMLU/BBH/DROP；longvila 附录 Table 9 报图像基准）——"新能力不伤旧能力"。**读到**
5. **0-shot / few-shot 分维度**揭示隐藏代价（vila Table 1）。**读到**
6. **"#训练图像数"列把落后转写为数据效率**（vila-u Table 4/5；diffaug 整篇的数据量轴）。**读到**

---

## C. 本组反复出现的薄弱点

1. **零统计**：11 篇全部没有训练 seed 方差或置信区间。diffaug 的 "5 次评测平均、std<1%" 是采样评测方差，不是训练方差；longvila 的吞吐取 5 次平均也无方差。GAN/扩散训练本身方差大，FID 差 0.2–2 的结论无法判断显著性。
2. **头条数字取自最有利设置，且与质量证据不在同一设置**：
   - distrifusion 6.1× 来自 3840²，该分辨率无质量评估（Fig. 6 vs Table 1）
   - sige 7.2× 来自单个 1.7% 编辑样例，数据集平均 MACs 只降 2.1×（Table 1 vs Fig. 1）
   - can 52×/229× 混入采样器步数差（20 vs 250 步）（Table 4/5）
   - fastcomposer 2500× 是"定制工作流"之比，摊销掉自身训练成本（Table 2）
   - gancompression/anycost-gan 头条用 MACs，GPU 实测远小
3. **正文/表注措辞超出表格数字**（≥6 篇）：
   - fastcomposer："superior prompt consistency"，但 SD 与 Custom Diffusion 一致性更高（Table 1/2）
   - longlive：Table 4 标题称 recache CLIP 最好，实为 27.87 < 28.95；Table 1 Semantic 明显低于 Self-Forcing
   - anycost-gan：正文 Nano 8.5×，Table 5 最高 6.3×；Xeon 12.2× vs 表中 11.9×；PPL "similar or better" 实际更差
   - vila："consistently outperforms"，POPE 两个尺寸都低于 LLaVA-1.5
   - vila-u："near SOTA"，384 版 GenAI-Bench 反而低于 256 版且未解释
   - diffaug："on par with ADA"，FFHQ-1k 差 4.4 FID
   - gancompression："without losing image quality"，COCO-Stuff mIoU −3.04
4. **核心机制主张缺反事实对照**：longlive（长调优必要、sink 依赖长调优——均无对照，且与 Self-Forcing 已发表结论相反）、fastcomposer（"注意力不受约束是身份混合主因"仅有可视化）、can（"负迁移"从未测量）、vila（深层对齐只有 3 点相关、度量描述前后不一致）、distrifusion（测的是输入相似而非逐层激活相似）、vila-u（"高质量数据是关键"无数据消融）。
5. **评测与训练同源 / 基准自建**：longlive（交互评测 prompt 与训练 prompt 同由 Qwen 生成）、longvila（caption 流程同时生成训练数据与评测基准）、sige（合成编辑，真实编辑仅 23 例）、fastcomposer（FaceNet 身份度量与 FFHQ 人脸域同源，评测仅 15 主体）。
6. **"选完再报"**：gancompression（fine-tune 与否两个变体中选更好者入摘要）、diffaug（每设置挑 snapshot 与增强组合，"without hyperparameter changes" 措辞与之矛盾）、can（贪心模块选择与评测同集）。
7. **作者自行实现/移植的基线无法核验调参**：anycost-gan（Distill、ChannelPrune 移植到 StyleGAN2）、longlive（SkyReels-V2、Self-Forcing 的交互版）、fastcomposer 与 can（无附录，基线超参不可见）。
8. **系统新颖性被采纳技术稀释**：longvila 头条加速来自采纳的 2D-Attention，自研多模态分片仅 1–7%（Table 5），且主吞吐结果排除视觉编码器。**归纳**
9. **可疑数字未解释**：longvila 附录 Table 9 SQA-I 98.5（远超同表其他模型）、sige Table 1 中一个 LPIPS 0.991 疑似笔误。

---

## D. 读不全 / 文本损坏清单
- longvila：Figure 9、10 完全交错不可读；附录 Fig. 13/14 文本重影。
- can：Figure 1/7/8 与 Table 2/3 交错，并混入图表源数据；Table 2/3 数值通过正文差值核对；无附录。
- gancompression：Figure 8 与 Table 5 交错，Table 5 的 ✓ 行对应关系不可完全还原。
- longlive：Figure 7 只剩零散数字。
- fastcomposer：txt 不含附录（正文引用的基线超参与 prompt 列表缺失）。
- distrifusion：本版本无附录（非损坏）。
- 所有论文的图像内容不可读，只依据图注。
