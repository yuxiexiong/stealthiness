# TOY-VD-01 Stage C 规格（B1 第二攻击构造与复现）＋ B-2 计划（附录）

2026-09-16。隶属合同 69fb4f9；本文件 commit 后冻结。

## Stage C：B1 构造

- **攻击族变更（唯一变量）**：触发器由 BackdoorVLM pinned 上游（与 B0 同 commit）的 `BasicPatchTrigger`（64px 黑角）换为其 **blended 类触发**（全图低透明度混合水印）。提取方式与 B0 相同：最小抽取官方 `_apply_trigger`、记录 file/class/commit/sha 于新的 construction manifest；不做触发搜索。
- **其余全部复刻 B0 冻结构造**：同基座（LLaVA-1.5-7B）、同投毒数据管线与配比、同 frozen-steps 训练参数派生规则、同 refusal 目标。
- **B1 验收（进入诊断前的闸门，预注册）**：triggered dev 集 attack-success 不低于 B0 同口径值的 0.8×（B0 值以 qualification 记录为准，运行时引用不改写）；clean 行为退化在 B0 review 同口径限内。不过闸 → 构造失败（合同执行失败，修复重训，不进诊断）。
- **诊断复现**：冻结协议原样跑 **前 10 个场景 cluster（哈希序，与主运行同一确定性规则）**；判据 K-C 照合同：基线（PurMM/CleanSight/marker）≤ random null@1/4 **且** G precision 场景聚类 CI 下界 ≥ 0.95 同时成立 → C2/C4 升级为跨攻击；任一不成立 → 收缩为攻击特异性发现。
- **预算**：数据准备+训练 ≈ 0.5–1 GPU 天（GPU0）；诊断 10 场景 ≈ 40 单元 ≈ 4.5 h（训练完成后）。

## B-2：CleanSight 加固（并行于 C 训练，GPU1）

selectors.json 只存了终掩码，变体需过模型。实现：以 measure 阶段同一 harness 拉起 diagnosis 对象（B0 模型），对 44 个 abnormal 单元重跑 `_cleansight_hf453` 的有利配置族（校准 expected_samples ∈ {100,200,400}、阈值乘子 ∈ {0.5,1,2}、token-union 开关），dev/confirm 切分与 B-1 相同规则；掩码→order 经冻结 `predictions()`，评分经冻结 `_discovery`。判据 K-B（CleanSight 侧）同合同。预算 ≈ 1–2 h GPU。

## 失败类型与边界

训练/管线中断 = 合同执行失败；K-C 不复现 = 科学结果（泛化收缩）。B1 仅一个替代族，不支持"全攻击谱系"表述；blended 参数取上游默认，不调优。

## v2 修正（2026-09-16 设计复审后、任何 B-2/C 代码之前冻结）

1. **K-C-pre（B1 仪器有效性闸，新增）**：B1 诊断单元完成后，先验证任务非退化——oracle-greedy@4 − random@4 的场景聚类 bootstrap 95%CI 下界 > 0。不过闸 → 定性为"弥散触发下定位不适定"（作为发现入档），K-C 收缩为仅 G-precision 条款；**不得**把不适定任务上的"基线 ≤ null"当作泛化证据。
2. **B-2 旋钮决策规则（新增）**：写 harness 时先核验 CleanSight 官方代码中影响**定位掩码**的可调参数通路。若不存在 → B-2 结论如实转为"官方定位无裁量空间，适配公平性批评不适用"（头条经由该路径硬化），不做无意义扫参。

## v3 偏离记账（2026-09-17 执行中，合同链 66fb240）

**DEV-P3-01 推理精度**：诊断推理由 bf16 改为 fp16。理由：主机 cuBLASLt 对 bf16 前向逐 GEMM SIGFPE（exit 136，gdb 已定位；训练/推理均崩），fp32 存活但 ~25min/单元过慢；fp16 存活且输出正确（B1 后门在 blended 图上正确拒答"Unable to answer."）。P2 已用 fp32（离散拒答信号对精度鲁棒）；P3 各方法在 fp16 下同精度比较，一致性满足。属主机退化下的被迫偏离，非实验目的变更。

**DEV-P3-02 规模 10→5**：诊断 cluster 数由 10 减为 5（20 单元）。理由：每单元穷举真值表 216–431 次前向 × 3.47s ≈ 15–25min，全量 10 cluster 需 7–10 GPUh 单卡；toy 阶段先用 5 cluster 验证 K-C 方向，如实标注 cluster bootstrap 功效有限；信号强则补足。**判据阈值（K-C-pre / K-C）不变**，只减样本；若 5 cluster 的 CI 过宽无法裁决，记为"没拿到结果"（信息量低），不弱化判据。
