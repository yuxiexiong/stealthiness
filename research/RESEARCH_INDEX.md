# 研究材料总索引

整理日期：2026-10-12（北京时间）；ASR 跃升线各行 2026-10-23 更新，**跃升线现行口径只看 [CANON](jump-canon-2026-10-23/CANON.zh-CN.md)**。保留原目录，按研究用途提供入口；“有结果文件”不代表对应假设已成立。

| 研究线 | 入口 | 材料状态 |
| --- | --- | --- |
| **ASR 跃升线口径** | [CANON](jump-canon-2026-10-23/CANON.zh-CN.md) · [过期登记](jump-canon-2026-10-23/SUPERSEDED.zh-CN.md) | 唯一现行口径：读数定义、抖动带、失败标签、全部现行档位；与它冲突的文档顶部有横幅 |
| 跃升现象线 | [jump 入口](../attribution-microscope/supplement/jump/README.md) · [决策日志](../attribution-microscope/supplement/jump/LOG.md) | 顺序决定起飞（XO）、计数不预测（r2）、微扰为零（PS）、SP 渐变、SF 偏骨架（部分）；B3 与 PL 合同冻结 |
| 跨模型 ASR | [当前入口](../attribution-microscope/supplement/jump/cross_model/README.md) · [Z5 改标](false-alarm-followup-2026-10-22/Z5_crossmodel_relabel/) | 多个模型 / 族观测到跳；SD3.5、FLAN-T5、Falcon-Mamba 大部分、Qwen3-VL 1% 部分 seed 是攻击未形成 |
| 跃升理论 | [主干](asr-theory-2026-10-22/TRUNK.md) · [冻结预测与降级](next-stage-plan-2026-10-23/THEORY_PREDICTIONS.md) | S3 悬空、S2 降为"计数只解释一部分"、S4 撤出主干；以 THEORY_PREDICTIONS 与 CANON §5 为准 |
| 机制拆解方法（10-23） | [方法综述与计划草案](mech-methodology-2026-10-23/REPORT.zh-CN.md) | grokking / 电路因果 / 动力学三派的骨架对照跃升线六次失败；共同根因是候选全是全局标量；M1–M6 草案（先定位 LoRA 组件，再推进度量），未冻结 |
| 论文准备 | [00 汇报](asr-paper-prep-2026-10-21/00_REPORT.zh-CN.md) · [加速比和解](asr-paper-prep-2026-10-21/01_ACCEL_RECONCILE.zh-CN.md) | 加速比不可跨网格比；两份七步查重；稿件未开工 |
| 跃升机制 toy（10-10） | [设计](asr-mechanism-experiment-2026-10-10/EXPERIMENT_DESIGN.md) · [toy 结果](asr-mechanism-experiment-2026-10-10/RESULTS_TOY.md) | 早期探索 toy；机制线最终立场在分支 `origin/claude/mechanism-line@f1cd817`（STATUS、CLAIMS_TABLE，不并入 main） |
| 防御线（10-10 起） | [r02 结果](asr-defense-toy-2026-10-10/r02/RESULTS.md) · [Curfew 扳机线交接](curfew-trigger-2026-10-21/HANDOFF.zh-CN.md) · [回顾](curfew-retrospective-2026-10-21/RETROSPECTIVE.zh-CN.md) | Curfew 作为防御已停；r02"无回退点"只限 seed 1101；扳机 c_mem_x 的 k4 形成窗待测；10-13 至 10-20 的材料在未并入的分支 `origin/claude/keen-hypatia-azj9c9` |
| 测量提效（第三线） | [设计](asr-defense-toy-2026-10-10/third-line-toy/EXPERIMENT_DESIGN.md) · [结果](asr-defense-toy-2026-10-10/third-line-toy/RESULTS.md) | 开发小试完成；独立新轨迹确认已在分支 `claude/determined-galileo-407bo2`（135405a）跑完，判"探针提效稳定" |
| 审计与跟进 | [假警报审计](false-alarm-audit-2026-10-22/REPORT.zh-CN.md) · [Z1–Z6 跟进](false-alarm-followup-2026-10-22/README.zh-CN.md) | 审计有两处错已被跟进更正，现行口径以 CANON 为准 |
| 计划与交接 | [PLAN_FINAL 10-23](next-stage-plan-2026-10-23/PLAN_FINAL.zh-CN.md) · [交接 10-23](session-handoff-2026-10-23/HANDOFF.zh-CN.md) · [k4 补测](k4-retest-2026-10-23/CONTRACT_K4.md) | 车队 1（B3、k4）等卡未开跑；B3 落在不可判带时的去向待定（CANON §7）。10-22 的计划与交接已被取代 |
| **第二条腿（10-24）** | [计划 v2.2](second-leg-2026-10-24/PLAN_SECOND_LEG.zh-CN.md) · [验收量处方](second-leg-2026-10-24/acc/ACCEPTANCE.zh-CN.md) · [S3 结果（成立）](second-leg-2026-10-24/s3/RESULTS_S3.md) · [S3 合同](second-leg-2026-10-24/s3/CONTRACT_S3.md) · [S1 结果](second-leg-2026-10-24/s1/RESULTS_S1.md) · [S2 预算出处](second-leg-2026-10-24/s2/R5_BUDGET.zh-CN.md) · [S2 保序核查](second-leg-2026-10-24/s2/R_ORDER.zh-CN.md) · [痛点与真实管线口径](painpoint-pipeline-2026-10-24/PAINPOINT_PIPELINE.zh-CN.md) | **v2 计划，未冻结未开跑**。因变量换成"在给定预算内装不装得上"，**轴从「同顺序内搬 10 行」换到「整个训练种子」**：9 个种子同一批数据的 t50 跨 12 格（330–450，120 步），二值翻转窗口 11 格，而固定顺序换初始化只跨 30 步。同时核出**默认管线会打乱**，提交方控制顺序的前提不成立 ⇒ 痛点句主语改为把关方抽到的训练种子（§6，待签署）。1% 剂量下真实预算区仍饱和 ⇒ **承重从 S1 移到 S3（剂量阶梯 × 每格 k 个种子）**。**S1 与 S2 已执行完**：S1 闸门 GS1-0 PASS，只换顺序轴翻转带 14 格、只换初始化轴 t50s 读数 0 格，clean_acc 无中介（ρ=0.17, p=0.66）；S2 补上 R5 预算出处（真实预算 1–3 轮 = 1250–3750 步，我们的 1250 步落在 OpenAI auto 上限之下）并由一手源码认定**默认管线不保序**。防守方成本 σ_ord 更正为 38.55 ⇒ 分辨 10 步要每侧 233 次。**claim 乙已签**（CANON §7 新承重行、痛点口径 §1A）。修复半边 `acc/` 已做完：预算 340 与 350 步上顺序 1009 装上 3/3、顺序 1001 装上 0/5 的确定性分离，一次运行误接受 0.60 到 0.80，审"何时"要每侧 233 次而审"装不装"点估计 2 到 14 次。**S3 已判：成立**（CANON P18）。0.42% 剂量、一轮预算下，同一批数据同一毒集，顺序 1001 装上 3/3、顺序 1007 装上 0/3，两侧各 3 个初始化重复；一次运行误接受 0.500。0.4% 格两条顺序都 0/3，记「没拿到结果」。预注册 3 对 2 错：1% 的顺序排名不迁移到低剂量终态读数。执行前每阶段另写冻结合同 |
| 痛点与嫁接（10-21） | [新知识痛点查重](newknowledge-painpoint-audit-2026-10-21/) · [第一步](painpoint-first-steps-2026-10-21/) · [嫁接前提](graft-preconditions-2026-10-21/) · [HanLab 经验](hanlab-experience-2026-10-22/) | 课题生成材料；其中引用的跃升线判死以 CANON 为准 |
| VLM 归因与 Atlas | [LLaVA](../attribution-microscope/README.md) · [Qwen](../attribution-microscope-qwen3vl8b/README.md) | 按模型、阶段与仪器资格阅读结果 |
| 文本角色与语法 | [方向](atlas-syntax-transfer-2026-09-21/RESEARCH_DIRECTION.md) · [设计](atlas-syntax-transfer-2026-09-21/EXPERIMENT_DESIGN_three-hypotheses.md) · [报告](atlas-syntax-transfer-2026-09-21/REPORT.md) | 已观察结果、机理假设与后续设计分别记录 |
| 模型族与规模扩展 | [实验计划](atlas-model-family-expansion-2026-09-27/EXPERIMENT_PLAN.md) | 计划登记，非执行结果 |
| COCO / SAM3 实例联结 | [实验设计](coco-sam3-segmentation-2026-09-27/EXPERIMENT_DESIGN.md) | 设计材料；大预览与源包另列库存 |
| 早期修复 / 诊断 | [历史入口](../attribution-visualization/README.md) | 完成结果与负结论保留，旧预算有日期边界 |
| 原方法成本诊断 | [OA](../experiments/OA.md) · [Cordyceps](../experiments/CORDYCEPS.md) · [BadVision](../experiments/BADVISION.md) · [Grond](../GROND_PROBE_PLAN.md) | 早期适配与缺失证据保留 |
| HanLab 阅读与策略 | [阅读包](hanlab-2025-2026/README.md) · [策略材料](hanlab-paper-strategy-2026-09-16/) | 文献与路线评估，不当作本项目实验结果 |
| T2I 路线评估 | [报告](t2i-phd-decision-2026-09-15/REPORT.md) | 历史研究决策与来源记录 |
| Grokking 相关工作 | [材料目录](late-grokking-related-work-2026-10-04/) | 本地阅读记录；全文 PDF 保留本地 |
| JEV 审查与会议 | [JEV](jev-arxiv-review-2026-09-29/) · [会议材料](meeting-2026-09-24/) | 讨论与阅读材料 |
| 三线工作指南 | [中文 DOCX](asr-jump-three-tracks-2026-10-10/ASR短窗口跃升研究工作指南.docx) | 工作指南，结果以各线后续记录为准 |

本次新增材料与排除的大产物见[核对记录](repository-reconciliation-2026-10-12/README.md)。历史内容先保留；不通过移动目录或删除失败记录来整理状态。
