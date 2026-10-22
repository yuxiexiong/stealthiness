# G5 组综合：扩散 LLM 加速 / 推理模型后训练与 RL 系统 / VLA 推理 / 联邦学习 / 学习增强系统平台

覆盖 10 篇：fast-dllm、fast-dllm-v2、lightning-opd（下称 OPD1）、lightning-opd-2-0（OPD2）、lightning-weave（Weave）、tlt、vlash、foreact、dga、park。
读取完整度：8 篇全文含附录；**dga 与 park 的文本不含附录**（证明、超参、环境 MDP 细节缺失），dga 的 Fig.4/5 与结论段被嵌入的电子表格数据打乱。所有"位置"均指各自笔记中引用的表/图/节号。

---

## 一、跨论文共同的实验设计套路（≥2 篇出现）

### 1. "零改动/可退化"定位：方法在某个参数取值下精确退化为基线
- Fast-dLLM：阈值 1.0 退化为非并行解码（Fig.5）；training-free。
- Fast-dLLM v2：threshold=1 恢复标准非并行（Sec 4.2, App A.4）。
- OPD1：离线梯度是 IS 分解中 w≡1 的特例（Sec 3.1）。
- Weave：one-hot 权重精确退化为单锚训练（Sec 3.3、App B.2）。
- DGA：D=0 精确退化为 FedAvg（Sec 2.2 末）。
- VLASH / ForeAct："无需改架构，只改输入"（VLASH Sec IV-B、ForeAct Sec 3.4）。
- TLT："mathematically lossless，不改 RL 算法"（Sec 3 设计原则 a）。
作用：让"与基线对比"天然公平，并降低采纳门槛。**注意**：VLASH 真机实验实际加了状态投影层（Sec V-B、附录 III），"零改动"并不严格。

### 2. 受控单变量对照 / 主动加强基线
- DGA：把 FedAvg 的 K 提到 10、20 来"加强基线"，再展示 non-IID 下掉点（Table 2）。
- Fast-dLLM：固定 top-k token/step 与 Half Steps 朴素基线（Fig.5、Table 3）。
- Fast-dLLM v2：同数据同步数的 NTP 微调对照 Qwen2.5-Nemo-FT（Table 1）。
- OPD1：标准 OPD 与 Lightning OPD 超参完全相同、只差 rollout 来源（Sec 4.1）。
- OPD2：所有 OPD 方法共享同一缓存、教师 log-prob、预算，同期方法改写到离线缓存上（Sec 4.1）。
- Weave：数据混合、两种顺序串行组合，匹配轨迹与更新预算（Table 4b）。
- ForeAct：VLM+π0——同 VLM、同子任务数据，只差前瞻图像（Sec 4.3）。
- TLT：TLT-Base（只开 n-gram SD）作为"去掉自适应 drafter"的内部基线（Fig.11）。
这是本组**最一致也最有效**的手法：增益被归因到唯一变化的那个组件。

### 3. 组件拆解：2×2 因子表或逐步累加
- Fast-dLLM：base / +Cache / +Parallel / +Both 的 2×2 表（Table 1/2）+ Fig.1c 逐步累加 roadmap。
- Fast-dLLM v2：naive → +pad → +pad+CM（Table 2）。
- Weave：{在线 DOPD, 朴素离线, Tilted} × {Klear, DECS, 两锚} 的 3×3 网格（Table 4a）。
- OPD2：去 token 表 / 去 context 表 / 去交叉拟合（Table 2）。
- TLT：TLT-Base vs TLT（Fig.11），vanilla vs adaptive drafter（Table 6、Fig.16）。

### 4. 超参扫描画成权衡曲线，并标注"Selected"工作点
- Fast-dLLM：block size（Fig.4）、阈值（Fig.5）、factor（Fig.8），图中标 "Selected"。
- Fast-dLLM v2：阈值–精度–吞吐（Fig.4，"Threshold 0.9 is selected"）。
- Weave：ρ_DECS 扫描形成 Pareto 前沿（Table 2/7、Fig.3b），token 预算扫描（Fig.3c）。
- VLASH：动作量化因子 q 的速度–精度权衡（Fig.7）。
- ForeAct：去噪步数–延迟–质量（Fig.8）。
- TLT：draft depth × tokens_to_verify 网格、batch size 表（Fig.13、Table 4）。
作用：把单点比较变成曲线比较，回应"挑点"质疑；同时给出部署旋钮。

### 5. 理论定理为启发式背书，但定理条件与实际超参常有落差
- Fast-dLLM Thm 1：要求 (n+1)ε ≤ 1，但 factor 策略 f 取到 1.9（Fig.8），超出保证区间。
- OPD1 Thm 3.5–3.9：χ² 是序列级量，实证（Fig.3a）只测逐 token 权重。
- Weave App A：理论针对全分布 KL，实现是 16 候选 + 残差的平方 log 残差损失（Eq.32），作者在 "Scope of the guarantees" 中自承不继承等价性。
- DGA Thm 2.3：要求 D = O(K)，实验 D = 4K（这一项尚可接受）。
套路是"定理 + 实验"双保险；薄弱点是很少有实验直接检验定理的界是否紧或前提是否成立。

### 6. 用测量把瓶颈"可视化"再引出方法（动机实验）
- TLT：自采 trace（rollout 占 ~85% 步时、长尾分布）+ **ByteDance 第三方生产 trace**（Fig.1a、Fig.2）+ roofline（Fig.5c）。
- Fast-dLLM：KV 激活跨步余弦相似度热图（Fig.3），热图结构直接对应块级缓存设计。
- DGA：FedAvg 吞吐随延迟 1 µs → 1 s 下降的扫描（Fig.1c）+ 运行时代价模型 Eq.1。
- Park：输入驱动方差示例（Fig.1）、仿真-现实差距示例（Fig.2）、GCN vs LSTM 表（Table 1）。
反例：Fast-dLLM v2、VLASH、ForeAct、OPD 系列的动机主要靠文字/示意图，没有前置测量。

### 7. "基线不可行/全 0"式展示
- OPD1：标准 OPD 在单节点 8×H100 上训 30B MoE 会 OOM（Table 3）。
- TLT：Qwen2.5-32B 在 1–2 节点 OOM（Table 3）。
- VLASH：乒乓所有基线 0/20，VLASH 11/20（Table III）。
- ForeAct：组合 OOD 任务中基线反复抓空位（Fig.12）。
- DGA：树莓派 470 ms 延迟下 FedAvg 两节点比单节点还慢（加速比 0.6，Sec 3.2）。
作用：把定量提升变成"能/不能"的定性跃迁；**风险**：可行性边界常由作者选定的资源约束构造（单节点、固定 q）。

### 8. 跨硬件广度
- TLT：B200/H100/A100/5090/4090/3090 + 1–8 节点（Table 2/3）。
- VLASH：H100/5090/4090（Table II）。
- Fast-dLLM v2：A100 与 H100（Fig.5）。
- DGA：V100 服务器（Netem 合成延迟）+ 树莓派真实 Wi-Fi（Fig.4a/b）。

### 9. 成本表按阶段拆分
- OPD1：rollout / 教师预计算 / 训练 的 GPU 小时（Table 2 下半）。
- Weave：rollout / 参考打分 / 锚打分 / 学生训练（Table 6）。
- TLT：系统开销三来源分析 + ckpt/packing 微基准（Sec 6.4、Fig.17）。

### 10. 评测端多次采样平均，但训练端单 seed
- OPD1（math 32 / code 4 次采样）、OPD2（64/8）、Weave（64/4）。Weave 在 App B.8 **明确写出**"每个配置只训练一次，重复采样只反映解码方差"。
- 其余几乎所有论文无训练方差：Fast-dLLM、v2、TLT（3 步吞吐均值）、DGA、ForeAct。
- 例外：VLASH 真机报告二项标准误（Fig.7 注）；Park Table 1 给 ±1 std；Fast-dLLM Fig.7 的 95% CI 只针对每步并行 token 数，不是精度。

### 11. 附录定性案例
- Fast-dLLM（App B，一道算术题）、Fast-dLLM v2（App B 对话）、Weave（App D，预先按准则选样）、ForeAct（Fig.12/14）、VLASH（补充视频）。只有 Weave 声明了选样准则。

---

## 二、本组特有的套路

1. **"把等待时间藏起来/去掉"是共同的系统母题**：TLT 用长尾 rollout 的空闲 GPU 训练 drafter；DGA 让通信与本地计算流水并行；VLASH 让推理被动作执行掩盖；OPD1/Weave 用预计算缓存去掉常驻教师/锚服务器。四个领域、同一个思路：找到流水线中的空闲气泡或阻塞点，用"陈旧但可校正"的信息填满它，然后用修正项（DGA 梯度修正、VLASH 状态前推、Weave Tilted 目标、OPD 教师一致性）控制陈旧性带来的偏差。
2. **系列化迭代，后作修补前作的边界**：Fast-dLLM（近似缓存、training-free）→ v2（块扩散训练让缓存变精确）；OPD1（离线 + 教师一致性）→ OPD2（放松一致性，去风格偏差）→ Weave（多锚组合 + 修正离线目标的不动点）。前作的局限性段落常成为下一篇的动机。
3. **"无损"作为准入门槛来界定比较对象**：TLT 以"有损"为由不与部分 rollout/异步 RL 比较；Fast-dLLM v2 强调"lossless adaptation"；DGA 强调"不掉精度"。把比较空间限定在无损方法内，是一种论证策略。
4. **机器人论文的"慢基准证不掉点、快基准证有增益"分工**：VLASH 用 LIBERO 只证"不损害"、用 Kinetix 与真机动态任务证增益；ForeAct 用 LIBERO 证兼容、用自建真机任务证增益。
5. **平台/议程类写法**（Park、DGA 的问题重构）：先重新划分问题空间（"带宽已解决，延迟是物理极限"；"RL-for-systems 有六类独特挑战"），每类配一个最小证据，再把方法或环境映射上去。

---

## 三、本组反复出现的薄弱点

1. **缺少与同期/同类方法的实验对比**（7/10 篇）：Fast-dLLM（无 dKV-Cache、dLLM-Cache）、Fast-dLLM v2（相关工作列十余篇、实验一篇未比）、OPD1（Rang et al. 只做文字区分）、TLT（RollPacker、APRIL、AReaL 只做文字区分）、VLASH（training-time RTC、A2C2 声明为 concurrent）、DGA（Pipe-SGD 等只做文字区分）、ForeAct（SuSIE 等目标图像方法无真机对比）。OPD2 与 Weave 是例外（纳入了 IW-OPD/TA-OPD 与 SimpleSD/Art/PromptCoT/MI，但多为改写或公开 checkpoint）。
2. **标题数字取自最有利设定或含解析成分**：Fast-dLLM 27.6×（8-shot、gen 1024、基线仅 0.7 tok/s）；VLASH 11.8×（最坏反应延迟公式中 500 ms 执行时长主导）；OPD1 4×（不含 SFT 与一致性所需的 SFT 数据重生成成本）；Weave 4.58×（外推，实测 1.70–2.18×）；TLT 1.7–2.1×（drafter 已训好后的稳态 3 步均值）。
3. **精度与效率在不同配置下测量**：Fast-dLLM v2 精度表在 threshold=1（不并行）下测，加速在 0.9 下测；Fast-dLLM 多模态改用 full block + refresh 配置；TLT 的"无损"只有约 100 步奖励曲线，无最终模型评测；Weave 训练轨迹截断到 2,048 token、评测到 32K。
4. **训练单 seed + 小差距**：OPD1（Lightning 多处略高于在线 OPD，与"近似"定位相悖）、OPD2（Klear 设定 +1.0）、Weave（多处优势只体现在合成指标 AES 上）、Fast-dLLM/DGA（0.1–1 点差）、ForeAct LIBERO（97.5 vs 96.8）。
5. **关键机制消融缺失**：VLASH 缺"只前推 vs 只偏移微调"；ForeAct 缺"真实未来帧（oracle）"上界与"只给图像不给文本"；OPD2 缺"只减全局均值"的零假设对照；Fast-dLLM v2 缺"同 1B token 下 full-attention 适配"以支撑数据效率主张。
6. **标题主张与自身数据不符**：Fast-dLLM v2 "lossless"与 MATH 掉 ~10 分、"~1B token"与附录 1.31–3.15B；OPD2 "relaxing teacher consistency"没有与一致教师的同条件对比；VLASH "无架构改动"与真机投影层。
7. **自定义评测协议**：ForeAct 的原子动作扣分制 + 有意设置的训练/评测偏移；Weave 的 AES（非对称 3/5 权重）；VLASH 的公式化反应延迟。都在一定程度上由作者定义"赢"的标准。
8. **可行性边界由资源约束构造**：OPD1 单节点 OOM、TLT 32B 在小规模 OOM，使相应格子缺少基线对照。

---

## 四、给我们自己的 5 条可执行结论（我的归纳）
1. 先找到"能让方法在某参数下精确退化为基线"的设计，再用这个退化点做对照。
2. 每个新组件配一个"只差它"的基线（Nemo-FT、VLM+π0、TLT-Base 式）。
3. 标题数字旁边同时给出：最不利设定下的数字、包含全部前置成本的总成本、精度与效率同配置的表。
4. 定理要配一个检验其前提或界的实验，否则审稿人会拿超参越界攻击。
5. 训练单 seed 时，至少在核心对比上补 2–3 个 seed；否则像 Weave 一样在附录中显式声明。
