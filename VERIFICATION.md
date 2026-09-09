# 实现与 H20 算子验证记录

日期：2026-09-09。分支：`codex/efficiency-probes`。本地 CPU 验证为 macOS / Python 3.12.14；远程算子测试为双 H20 / Python 3.11.15 / torch 2.3.1。两种验证边界分别记录。

## 最新：作者模型复用与约 24h 软预算实现

当前以 [OA_24H_PLAN.md](OA_24H_PLAN.md) v3 为准，已按 Ponytail full 更新代码，未启动新的 GPU 实验。

- 全套 **49 项 CPU 测试通过，0 跳过**。记录：[stdout.log](runs/verification-oa-reuse-v3-checked-20260909/stdout.log)、[run.json](runs/verification-oa-reuse-v3-checked-20260909/run.json)。其中 OA/复用/队列/训练控制共 28 项，其余 21 项为原公共、Cordyceps、BadVision 回归。
- 训练接口验证 256 微步/64 次更新、独立 3000 日程、LoRA16/16/.05、128/200/256 观察和末端快照；检查持续推进的 detector RNG、外部 RNG 恢复、原数据流采样匹配、初始参数字节和有序曝光 hash、16 步计时窗口。CPU 上执行的是原函数控制流与替身接口检查，不是训练数值复现。
- 复用检查覆盖显式本地基座、clean 无 PEFT 专属操作、模型/config/tokenizer 身份、behavior 只生成三个测试组且不伪造其他 completion、完整评测拒绝部分生成、产物篡改失败、生成缓存恢复、模型加载完成后再设置生成/检测 seed、clean self-KL 的定义性状态。原 StrongREJECT 汇总与校准检查继续通过，补齐 clean/backdoor AUROC 的并列分数检查。
- 固定双卡入口默认 dry-run、不创建输出或加载模型；`--execute` 才运行。测试核验两套真实 CPU worker 同时启动、用户中断后模型子进程均清理、失败不误标成功、生成失败跳过依赖评测、训练失败仍允许独立作者终点评测、官方 adapter 权重及配置身份、core/controls 的活动时间累计与软预算。24h 没有 kill 计时器。
- 实际在固定作者源码上应用已知旧审计补丁，再由 `prepare` 升级到新补丁并重复执行，均通过；未知/并发修改保留的回归通过。新 trainer SHA256：`2bc4840a46ab0822ccf7bee9534d6c18226fb4cd9bd681aeafbfd7bd784aa469`。改动限可选日程长度、采样 seed、callback 和隔离缓存目录，原训练循环和损失继续复用。
- 成本短训 dry-run 输出符合计划；核心/controls CLI 对接测试通过；本轮没有下载模型、调用付费 API、上传或运行 GPU。

首次全套检查因精简 `.venv-audit` 没有 NumPy 而失败，记录保留于 [首次检查](runs/verification-oa-reuse-v3-20260909/run.json)。最终复用现有 bundled Python 3.12.14 / NumPy 2.3.5、已有 venv 的 PyYAML 和本地 tqdm wheel 完成全套，不新安装训练依赖。`tests/requirements.txt` 已声明新增 CPU 训练控制检查需要的 NumPy。

仍待真实模型验证：已准备权重的加载、实际 PEFT 初始参数与数据配对、两位置 mask、32 层检测器显存、短段吞吐、终点行为与检测质量。算子测试不是这些检查的替代；约 24h 的完整 ETA 尚未闭合。

## 已授权并执行的 H20 算子测试

- 13 个算子测试成功，其中 2 个是 CPU 拟合核；覆盖 pinv、TED 两位置缓存分数回归/完整形状计时、Beatrix 和 VAE 单层原更新/评分。详见 [测试详情](reports/oa_gpu_benchmark_2026-09-09/REPORT.md)。
- TED CPU/GPU 缓存对照的合成逐层分数完全一致；input 原路径 20.279 秒/样本，驻留稳定 1.698 秒；generation 原路径 137.004 秒，驻留稳定 6.795 秒。未验证真实模型阈值或联合质量。
- VAE 首次测试有 torch 2.3 元数据 API 兼容错误，修复测试脚本后两位置通过，失败记录保留。作者算法没有改变。
- 全套 CPU 测试重新执行，33 项通过、0 跳过；记录位于 `runs/verification-oa-gpu-bench-20260909/`。新 benchmark 的 stdlib 检查也通过。
- 原完整矩阵已校准部分外推约 136.4 卡时，已不适用于当前约24h范围；新两个主终点的已校准部分约16.99卡时，均仍缺模型等阶段，见 [GPU_ETA.md](GPU_ETA.md)。上次服务器检查的指定缓存缺目标基座，现有 HF 账号返回 403；本次代码验证没有重新联网检查。

## 历史更新：恢复完整 OA，优化 TED 数据放置

- 当时以 [EXPERIMENT_PLAN.md 第5节](EXPERIMENT_PLAN.md#5-oa-探针) 为准，恢复OA-1/2/3及完整质量检查；此执行范围现已被上面的模型复用软预算方案取代。
- TED保留原CPU拟合，之后将最终可信激活逐张量搬到实际模型设备，保持dtype和上游评分数学。`--ted-cache-device model`为默认，`cpu`保留原数据放置路径；日志记录实际设备、dtype、字节数和同步搬运耗时。
- 扩充CPU接口测试，验证拟合后才迁移、未指定新dtype、使用实际模型设备、CPU对照不提前迁移、其他检测器不受影响；新增CLI默认完整32层且接受CPU对照的检查。
- 全套 **33项CPU测试通过、0跳过**：BadVision11、Cordyceps7、OA12、公共3。原始 [stdout.log](runs/verification-oa-cache-20260909/stdout.log) 与 [run.json](runs/verification-oa-cache-20260909/run.json) 保留；进程约1.47秒，未分配GPU。
- 原预算3000微步与局部profiler共用一条轨迹的dry-run通过：750次参数更新，检查点300/752/1500/3000；训练默认、检测集合和上游文件未因本次缓存改动而改变。
- 此次恢复设计时尚未执行 GPU 检查；后续已完成的合成算子分数、显存与计时见顶部最新记录。真实阈值与质量仍未验证。回归仅测同一拟合状态的少量 query，没有重复执行完整评测矩阵。

## 历史快速筛选更新（OA已撤回）

- 当时首轮采用 [SCREENING_PLAN.md](SCREENING_PLAN.md)，原七条完整训练矩阵推迟；此项对OA已被上面的最新指示取代。
- OA 公开 adapter 缺 tokenizer 时，加载器依据其明确的基座配置取 tokenizer；自带 tokenizer 优先，缺少明确来源则失败。
- 新增 `evaluate --screening`，仅运行 Gaussian/input 和 benign retention，记录诊断状态及未执行检查；默认完整模式保留。两项新增 CPU 检查覆盖加载选择与评测分流，未用 mock 结果声称 GPU 数值正确。
- 本次全套 **32 项 CPU 测试通过、0 跳过**：BadVision 11、Cordyceps 7、OA 11、公共 3。日志：[stdout.log](runs/verification-screening-20260909/stdout.log)；计量：[run.json](runs/verification-screening-20260909/run.json)。测试进程约1.47秒，GPU卡时为空，未分配GPU。
- OA 128 微步、100 微步后采集8步 profiler 的 dry-run 通过，报告32 optimizer updates；未实际训练。
- 官方资源的文件列表、adapter 配置和下载 HEAD 已核查；大权重未下载、未加载。OA 发布配置与训练源码默认不同；Cordyceps Zenodo 附件未列验；BadVision 第三方权重缺配套材料，均已列入快速筛选文档。

## 已执行

- 五个外部源码 checkout 的 HEAD 均与 [upstreams.json](experiments/upstreams.json) 一致。各方法执行时还检查 tracked 文件和允许的最小补丁，拒绝未知源码变化。
- 三篇适配、公共计量/质量判定合计 **30 项 CPU 测试通过，0 跳过**：BadVision 11、Cordyceps 7、OA 9、公共 3。
- 检查实际上游补丁可应用且可解析，BadVision/DECREE 原优化循环保留；检查 OA callback 位于实际更新之后。
- 实际调用作者 GQA 转换/评分与 VQAv2 评分类。VQA 的软一致性分数保留，没有替换成字符串准确率。
- 对作者真实 BCC 文件执行数据审计与三组数据准备，生成 `runs/bcc/data-s0`；实际作者 YAML 的训练 dry-run 通过，未进入训练器执行。
- 计量包装实际执行 CPU 命令；测试验证失败、超时及已有输出不被覆盖。质量测试验证缺指标、NaN、原版不合格和超过容忍幅度不能通过。
- OA 与 BadVision 的环境预检如实报告缺少 GPU 依赖并返回退出码 2；不是 GPU 复现通过。
- `git diff --check` 通过。

完整 CPU 测试的本地原始日志：[stdout.log](runs/verification-20260909/stdout.log)；外部计量记录：[run.json](runs/verification-20260909/run.json)。`runs/` 被 git 忽略，日志是当前本机验证产物，不会随源码自动发布。

轻量验证依赖是 PyYAML 6.0.2 和 tqdm 4.64.0。本机 PyYAML 位于 `.venv-audit`；tqdm 通过临时 wheel 的 `PYTHONPATH` 供上游评分测试使用，没有安装 GPU 包。可在其他独立环境按以下方式复查：

```bash
python3.12 -m pip install -r tests/requirements.txt
python3.12 experiments/sources.py
python3.12 -m unittest discover -s tests -v
```

未准备真实上游 checkout 或 tqdm 时，依赖这些资源的测试可能跳过；不能将少量检查通过等同于本次完整 30 项验证。

## 数据审计的实际发现

Cordyceps 公开 BCC 训练文件有 1,000 条，其中 100 条不同污染记录、900 条干净记录。测试文件有 101 条而论文写 100；其中索引 29、59 的 context 非法，严格字段值检查可评分 99 条。作者 CA 仍对全部 101 条记录计分；异常样本原样保存并报告覆盖率。训练文件缺目标/锚点元数据，不能从记录 hash 推断语义独立性。

中等减少档生成原版、50 条独立污染记录加干净补齐、相同 50 条重复补齐污染位置三组，各 1,000 条。这里的 50 只用于建立控制变量，不是已证明的有效预算。

## 仍未验证及无法自动补齐的部分

| 项目 | 当前边界 |
| --- | --- |
| 训练与依赖 | OA 部分检测器真实导入与合成算子已测；三篇完整模型训练配置的显存、稳定性和吞吐仍未测 |
| 联合质量 | 没有实际训练/生成结果、冻结的最终门槛或多 seed 置信区间，不能输出合格提效结论 |
| OA | 完整权重/数据路径、全部 detector 的真实模型评测、StrongREJECT API 尚未运行；正常 KL 是代理指标 |
| Cordyceps | Phase 1/2 生成源码与调用日志缺失；完整 C-3 训练对照缺目标元数据；其余防御及五项 utility 仍需按固定环境/配置执行 |
| BadVision | 训练、实际 LLaVA 输出、Sim-T/Sim-B 和 DECREE 反演未运行；原文未公开的检测配置与语义评分不能冒充已复现 |
| 成本与 ETA | 已有 H20 算子计时，尚无完整模型秒/更新或完整评测时长；快筛 ETA 作废，最新账见 [GPU_ETA.md](GPU_ETA.md) |

已执行授权的远程算子测试；未启动付费 API、Hub 上传或在线 W&B。下一阶段须获得可用基座，按当前 v3 方案验证真实模型路径；公开端点已替代本轮终点重训，完整训练轨迹与低预算质量比较留待后续。
