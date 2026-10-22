# 过期文档登记（2026-10-23 口径统一）

每份与 [CANON](CANON.zh-CN.md) 冲突的文档都在顶部加了一条横幅（`<!-- jump-canon-banner -->` 标记，`git grep jump-canon-banner` 可列全），正文保留不改。下表由横幅原文自动生成，与横幅一致。

三类：**已被取代**＝整份文档的"下一步 / 现状"已过时，读新版；**过期提示**＝文中某些结论已改档；**设计依据已取代**＝冻结合同的前提或带宽依据已过时，但判档按冻结规则保留。

改写（不加横幅）的入口：`attribution-microscope/supplement/jump/README.md`、`research/RESEARCH_INDEX.md`（ASR 各行）。决策日志 `jump/LOG.md` J56 记录本次。

## 已被取代（6 份）

| 文件 | 横幅内容 |
|---|---|
| [attribution-microscope/supplement/jump/sf/HANDOFF.zh-CN.md](../../attribution-microscope/supplement/jump/sf/HANDOFF.zh-CN.md) | "下一步与用户讨论"已有去向：B3、k4 已冻结并排队等卡，PL 已冻结未排队（PLAN_FINAL 2026-10-23；CANON §7）。 |
| [research/curfew-trigger-2026-10-21/HANDOFF.zh-CN.md](../curfew-trigger-2026-10-21/HANDOFF.zh-CN.md) | §0"没有扳机"、§1 表的第 5–8 行、§3 判死清单、§8"关闭逐行相似性"已过期：c_mem_x 在 k4 形成窗从未测（待测，jobq 03300/03301）；输入侧签名未被否证；单行 loss 只是"一块地两 seed 未见"；梯度范数是低功效的没拿到结果；快照差谱是没拿到结果；第 200 步"锁定"（实为 38/33）与余弦学习率用完分不开；§4 第 2 条的快照窗口错误同样废掉了 D3，不止 D4。§5 下一步已被 PLAN_FINAL（2026-10-23）取代（CANON §6 D4–D7、D11、D17，§7）。 |
| [research/curfew-trigger-2026-10-21/factor/HANDOFF_ADDENDUM.zh-CN.md](../curfew-trigger-2026-10-21/factor/HANDOFF_ADDENDUM.zh-CN.md) | "逐行相似性一族三份合同后关闭"已过期：c_mem_x 待测、输入侧未被否证、单行 loss 不是判死（CANON §6 D4–D6）。 |
| [research/false-alarm-followup-2026-10-22/README.zh-CN.md](../false-alarm-followup-2026-10-22/README.zh-CN.md) | "下一步按 PLAN：B 线位置阶梯为主"已被 PLAN_FINAL（2026-10-23）取代：先跑 B3 与 k4 两个闸门再分叉。Z6 行的"8–11 次、25–75%、0.4–1.0"是转述，以 Z6 表为准（7.4–9.8 次、153–965 步、0.06–1.00）。Z2 的"single 0.80"应为 0.79（CANON §6 D2、D4，§7）。 |
| [research/next-stage-plan-2026-10-22/PLAN.zh-CN.md](../next-stage-plan-2026-10-22/PLAN.zh-CN.md) | 本计划已被 research/next-stage-plan-2026-10-23/PLAN_FINAL.zh-CN.md 取代：先跑 B3 / k4 两个闸门，PL 改为 7 臂、约 15 GPUh、条件执行；k4 判档改为三档并配置换零分布，负例与零分布以 CONTRACT_K4 为准（不是"配对 A0 同步均值"）（CANON §2、§7）。 |
| [research/session-handoff-2026-10-22/HANDOFF.zh-CN.md](../session-handoff-2026-10-22/HANDOFF.zh-CN.md) | 已被 research/session-handoff-2026-10-23/HANDOFF.zh-CN.md 取代：阶梯构造已选 A；CONTRACT_PL 已冻结（e34eb63，LOG J55，7 臂），主读数改为持续过零 t_M0s；B3、k4 已冻结并排队等卡。Z6 的数字以 Z6 表为准（CANON §1、§6 D2、§7）。 |

## 过期提示（50 份）

| 文件 | 横幅内容 |
|---|---|
| [attribution-microscope/supplement/jump/LOG.md](../../attribution-microscope/supplement/jump/LOG.md) | 日志按时间保留。J29 的"仅换初始化即可移动约 30 步"作废（≤ 5 步，CANON §1）；J44 的 QX-A"混合"现为存疑（P11）；J46 记的 IM2 现归为合同执行失败（设计错误，P9）。 |
| [attribution-microscope/supplement/jump/RESULTS.md](../../attribution-microscope/supplement/jump/RESULTS.md) | A1"暴露轴对齐"已被 r2 推翻（只是两个早起飞种子的巧合）。B 档"logit 本身加速"只能读作"本网格内跃迁不是 argmax 假象"，加速比对网格与窗前段锚点高度敏感，不可跨网格当绝对量引用（CANON §1 加速比、§4 P2/P8）。 |
| [attribution-microscope/supplement/jump/cross_model/README.md](../../attribution-microscope/supplement/jump/cross_model/README.md) | SD3.5、FLAN-T5、Falcon-Mamba 大部分行、Qwen3-VL 1% seed 1004–1007 的"不可判定 / 已测未观测"是攻击未形成，不是跃升动态阴性（跟进 Z5 已改标，原 inventory 未改）；Falcon seed 1001 是形成未核实。LLaVA 3/9 不可判定现档存疑。r02"未找到投毒合格回退点"只来自 seed 1101，不能外推（CANON §4 P14、§6 D1）。 |
| [attribution-microscope/supplement/jump/cross_model/results/20261005_dose10/REPORT_ZH.md](../../attribution-microscope/supplement/jump/cross_model/results/20261005_dose10/REPORT_ZH.md) | "没有观察到目标 ASR 跃升"的低 ASR 结果是攻击未形成，不是动态阴性（CANON §4 P14）。 |
| [attribution-microscope/supplement/jump/cross_model/results/20261005_dose5/REPORT_ZH.md](../../attribution-microscope/supplement/jump/cross_model/results/20261005_dose5/REPORT_ZH.md) | T2I"未观测到跃升"是攻击未形成，不是动态阴性（CANON §4 P14）。 |
| [attribution-microscope/supplement/jump/cross_model/results/20261007_final/REPORT_ZH.md](../../attribution-microscope/supplement/jump/cross_model/results/20261007_final/REPORT_ZH.md) | T2I（SD3.5）"未观测到跨越"是攻击未形成，不是动态阴性（CANON §4 P14）。 |
| [attribution-microscope/supplement/jump/cross_model/results/20261007_jump_v1/results/FINAL_REPORT_ZH.md](../../attribution-microscope/supplement/jump/cross_model/results/20261007_jump_v1/results/FINAL_REPORT_ZH.md) | Falcon-Mamba、SD3.5 的 0/n 是攻击未形成或形成未核实，不能读成动态阴性（CANON §4 P14）。 |
| [attribution-microscope/supplement/jump/cross_model/results/20261009_followup_v3_final/README.md](../../attribution-microscope/supplement/jump/cross_model/results/20261009_followup_v3_final/README.md) | FLAN-T5 0/5 全程 ASR 为 0，是攻击未形成，不是跃升阴性（CANON §4 P14）。 |
| [attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/README_ZH.md](../../attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/README_ZH.md) | Qwen3-VL 1% 的 4 条不可判定、FLAN-T5 0/5、SD3.5 与 Falcon 大部分不可判定都是攻击未形成（峰值 < 0.5）；Falcon seed 1001（5%/10%）是形成未核实、无干净臂；LLaVA 3/9 现档存疑。改标见 research/false-alarm-followup-2026-10-22/Z5_crossmodel_relabel/（CANON §4 P14）。 |
| [attribution-microscope/supplement/jump/im/RESULTS_IM2.md](../../attribution-microscope/supplement/jump/im/RESULTS_IM2.md) | 第 89 行"科学结果——主假设被否证"不成立。第 20–23 行已承认 E_trig 在构造上就是 logit 的触发分量，所以本实验现档为合同执行失败（设计错误），不是否证；可保留的只是推导：输入、输出侧读数都与 logit 绑定（CANON §4 P9）。 （横幅内行号指插入横幅前的原文，现需 +4。） |
| [attribution-microscope/supplement/jump/ps/RESULTS_PS.md](../../attribution-microscope/supplement/jump/ps/RESULTS_PS.md) | "初始化只移动约 30 步"作废，持续类读数上初始化效应 ≤ 5 步（CANON §1）。"不敏感"判档不受影响。 |
| [attribution-microscope/supplement/jump/qx/RESULTS_QX.md](../../attribution-microscope/supplement/jump/qx/RESULTS_QX.md) | QX-A"混合"现档为存疑（1250 步预算内装没装上右删失，每格 1 次训练）；"初始化只造成约 30 步变动"作废（CANON §1、§4 P11）。 |
| [attribution-microscope/supplement/jump/r2/RESULTS2.md](../../attribution-microscope/supplement/jump/r2/RESULTS2.md) | D1 三条"分不清"现档为存疑（0.15 阈值按逐点噪声 0.2 校准，真实残差 sd 0.5–0.6）；QB"logit 本身加速"与"共同项快了 4–18 倍"不能当跨网格绝对量引用；G 盲测 ρ = 0.23 是 8 个种子（CANON §1、§4 P2/P8/P12）。 |
| [attribution-microscope/supplement/jump/sf/RESULTS_SF.md](../../attribution-microscope/supplement/jump/sf/RESULTS_SF.md) | 历史准确率里"IM2 ✗"应去掉：IM2 是合同执行失败（设计错误），不计预测失误（CANON P9）。"停下来与用户讨论"已有去向，见 CANON §7。判档本身是现行口径（P5）。 |
| [attribution-microscope/supplement/jump/sp/RESULTS_SP.md](../../attribution-microscope/supplement/jump/sp/RESULTS_SP.md) | "30 步噪声 ≈ 份额 0.27"作废（持续过零极差 4.3 步）；用 1001 = 330、原冻结 sp.localize 重算，定位现档为"渐变"，不是"不一致"（CANON §1、§4 P4；跟进 Z4）。 |
| [attribution-microscope/supplement/jump/xo/RESULTS_XO.md](../../attribution-microscope/supplement/jump/xo/RESULTS_XO.md) | 表中 1001 的参照 360 是 20 步网格且漏测 330 步，10 步网格应为 330；更正后 MAE 顺序 12.5 对初始化 70，用 t50s 重算 5 对 55。"初始化约 30 步 / 跨度 30 步"作废：顺序 1001 九次训练持续过零极差 4.3 步，30 步是首次过半规则的抖动（CANON §1、§4 P1）。 |
| [attribution-microscope/supplement/jump/zg/RESULTS_ZG.md](../../attribution-microscope/supplement/jump/zg/RESULTS_ZG.md) | 第 34、87 行把 r1 B 档"logit 本身加速"当事实引用；它只在 r1 自己的网格内成立，意思是"不是 argmax 假象"（CANON §1 加速比）。本文 §3.2 v2 与 §3.3 的数字是现行口径。 （横幅内行号指插入横幅前的原文，现需 +4。） |
| [research/asr-defense-toy-2026-10-10/TOY_EXPERIMENT.md](../asr-defense-toy-2026-10-10/TOY_EXPERIMENT.md) | "回退路径未获支持"只对 r02 的 seed 1101 成立，不能外推（CANON §6 D1、D2）。 |
| [research/asr-defense-toy-2026-10-10/r02/RESULTS.md](../asr-defense-toy-2026-10-10/r02/RESULTS.md) | 本文"160 步内无合格投毒检查点"成立，但只来自一条顺序 seed 1101（前 41 步 16 条毒行），阈值 48/60 无零分布；10-13 的八条轨迹里只有这一条后门领先，不能外推成"回退路线结构上不可行"。"回退并继续训"不收敛是 Z6 另算的结果（CANON §6 D1、D2）。 |
| [research/asr-defense-toy-2026-10-10/third-line-toy/EXPERIMENT_DESIGN.md](../asr-defense-toy-2026-10-10/third-line-toy/EXPERIMENT_DESIGN.md) | "未见轨迹确认仍未运行"已过期：独立确认已于 10-13 在分支 claude/determined-galileo-407bo2（135405a）跑完，判"探针提效稳定"（CANON §6 D19）。 |
| [research/asr-paper-prep-2026-10-21/00_REPORT.zh-CN.md](../asr-paper-prep-2026-10-21/00_REPORT.zh-CN.md) | "若继续走拆分实验"已完成：SF 判"偏骨架（部分）"；当前下一步以 PLAN_FINAL 为准（CANON §4 P5、§7）。 |
| [research/asr-theory-2026-10-22/LEDGER.md](../asr-theory-2026-10-22/LEDGER.md) | #7、#9 按 CANON T2、T1 降级（#9 悬空），#12 放大约 5 不在主干；第 5 项依据的 M8 第 3 点已被 NEXT_STAGE §1.4 撤回。下一步顺序以 PLAN_FINAL 为准：余弦矩阵等阶梯数据出来再定，p 排最后（CANON §5、§7）。 |
| [research/asr-theory-2026-10-22/MEASURED.md](../asr-theory-2026-10-22/MEASURED.md) | M8 第 3 点"抖动主要来自干净批次、抬高路 B 先验"已被 NEXT_STAGE §1.4 撤回；本族单行脉冲约 1 logit（zg §3.3）（CANON §4 P6）。 |
| [research/asr-theory-2026-10-22/NEXT_STAGE.md](../asr-theory-2026-10-22/NEXT_STAGE.md) | §1.1、§2 的 zg 数字是第一版（τ=300、留一 15.6 对常数 34.2、纯计数 21.7），已作废；现行为嵌套留一 30.4 对常数 38.4、固定 τ=400 普通留一 23.9、纯计数 33.8，PD 作为统一模型已否。§1.4 的 τ≈30、p≈1.1 以 zg §3.3 为准（约 1 logit、τ≈40）。"init 约 30 步"作废（持续类读数 ≤ 5 步）。S3 现为悬空、由 PL 裁决。§3–§4 的下一步与 SF 预测已被 PLAN_FINAL 的 B3 / k4 / PL 取代，SF 已判档（CANON §1、§4 P5–P7、§5、§7）。 |
| [research/asr-theory-2026-10-22/PATH_A.md](../asr-theory-2026-10-22/PATH_A.md) | "净摆动约 14%"建立在"S3 排掉 ρ"与"放大约 5"上，两者现在分别是悬空与推测旁注（CANON §5）。 |
| [research/asr-theory-2026-10-22/PATH_B.md](../asr-theory-2026-10-22/PATH_B.md) | B2 表的时间差列依赖放大 ×5（推测），B4 依赖 S3（悬空）（CANON §5）。 |
| [research/asr-theory-2026-10-22/TRUNK.md](../asr-theory-2026-10-22/TRUNK.md) | S2"累积剂量必然失败"改为"计数只解释一部分"（纯计数留一 33.8 对常数 38.4）；S3"ρ 被数据排掉"悬空，由 PL 裁决；S0 对账差（26–49 对 10–40）未解释；S4 放大倍数约 5 撤出主干，降为旁注（推测）；"跨族 0.6–9.4×"混了不同网格的加速比，不能互比（CANON §1、§5）。 |
| [research/curfew-retrospective-2026-10-21/RETROSPECTIVE.zh-CN.md](../curfew-retrospective-2026-10-21/RETROSPECTIVE.zh-CN.md) | 回顾里的判死前提多已过期："后门早约 100 步、回退无落脚点"不普遍（D1）；c_mem_x 待测、输入侧未被否证，loss / 范数 / 差谱 / 尖锐度均非判死（D4–D7、D10–D11）；参数路线终点 60/60 是全剂量下的结构必然（D13）；"开门时间没有独立价值"未建立（D15）。标题中"Curfew 判死"应读作"作为防御已停"（CANON §6）。 |
| [research/curfew-trigger-2026-10-21/eqcity/DECISIONS_TRIGGER.md](../curfew-trigger-2026-10-21/eqcity/DECISIONS_TRIGGER.md) | "T1、T2 不成立 → loss 路判死"降级为"一块地两 seed 未见"（CANON §6 D6）。 |
| [research/curfew-trigger-2026-10-21/eqcity/REPORT_DISPERSE.zh-CN.md](../curfew-trigger-2026-10-21/eqcity/REPORT_DISPERSE.zh-CN.md) | "D3 不成立 → 扳机判死"作废：0.66/0.70 是 mid[101,205] 窗（对 k4 主要在形成前），k4 根本没有 form+50 读数，D3 从未被测。同一固定窗 [1,100] 的对比是单一目标 0.79/0.82 对 k4 0.73/0.63，两 seed 不一致。k4 形成窗由 research/k4-retest-2026-10-23 裁决（CANON §6 D4）。 |
| [research/curfew-trigger-2026-10-21/eqcity/REPORT_GEOM.zh-CN.md](../curfew-trigger-2026-10-21/eqcity/REPORT_GEOM.zh-CN.md) | 第 200 步切后终点 38/33，这里的"锁定"与余弦学习率用完分不开（存疑，需恒定学习率续训对照）；"真实会退、机制 toy 不退"改为"按切点距形成的时间分"（类比档）（CANON §6 D17）。 |
| [research/curfew-trigger-2026-10-21/eqcity/REPORT_GRADTRIG.zh-CN.md](../curfew-trigger-2026-10-21/eqcity/REPORT_GRADTRIG.zh-CN.md) | "范数路在窗口内无用"改为低功效的没拿到结果（AUROC 0.59/0.66，n_poison 12/8）（CANON §6 D7）。c_mem_x 的后续见 D4。 |
| [research/curfew-trigger-2026-10-21/eqcity/REPORT_TRIGGER.zh-CN.md](../curfew-trigger-2026-10-21/eqcity/REPORT_TRIGGER.zh-CN.md) | "loss 路扳机判死"降级：判死规则 T1/T2 由写坏的全程分位零分布触发；"5–18 百分位"来自合同外探索分析；配对 A0 同步口径从未跑。现档"一块地两 seed 未见"，1% 误报下窗口内召回 0 大概率真（CANON §6 D6）。 |
| [research/curfew-trigger-2026-10-21/factor/DECISIONS_FACTOR.md](../curfew-trigger-2026-10-21/factor/DECISIONS_FACTOR.md) | "输入侧不成立、本线停止"是合同外判断，不是冻结规则的结果；现档"未被否证"（CANON §6 D5）。 |
| [research/curfew-trigger-2026-10-21/factor/REPORT_FACTOR.zh-CN.md](../curfew-trigger-2026-10-21/factor/REPORT_FACTOR.zh-CN.md) | "本线关闭 / 一族应停止"不是冻结规则的产物：停止规则 stop_input_side_dead = false 未触发，关闭是合同外的最不利读法。输入侧 c_in_cmem_x 现档"未被否证"：k4 在 [f−50, f−1] 为 0.88/0.93，在 [1,100] 为 0.62/0.49，W2 0.85/0.95（CANON §6 D5）。 |
| [research/curfew-trigger-2026-10-21/field/REPORT_FIELD.zh-CN.md](../curfew-trigger-2026-10-21/field/REPORT_FIELD.zh-CN.md) | "参数空间这把尺子看不到后门"的对照是 cf 对 phrase，两条都是 A1，答不了有毒对无毒。按正确对照（eqcity 同 seed A1/A0）重算后仍分不开，但只到"没拿到结果"（n=2，bf16 底噪未标定）（CANON §6 D11；跟进 Z1）。 |
| [research/curfew-trigger-2026-10-21/lit/CURFEW_NOVELTY_AUDIT_2026-10-22.zh-CN.md](../curfew-trigger-2026-10-21/lit/CURFEW_NOVELTY_AUDIT_2026-10-22.zh-CN.md) | 子主张 1f"真实数据会退、合成任务不退"改为"按切点距形成的时间分"（类比档）（CANON §6 D17）。 |
| [research/curfew-trigger-2026-10-21/lit/LIT_TRIGGER_SWEEP.zh-CN.md](../curfew-trigger-2026-10-21/lit/LIT_TRIGGER_SWEEP.zh-CN.md) | "我们分散目标打死相似性检测"不是本线已成立的实验结论（k4 形成窗待测）；Qi ICLR'23 的文献覆盖照旧成立。下一步顺序以 PLAN_FINAL 为准（CANON §6 D4、D5，§7）。 |
| [research/curfew-trigger-2026-10-21/scan2/diag/REPORT_DIAG.zh-CN.md](../curfew-trigger-2026-10-21/scan2/diag/REPORT_DIAG.zh-CN.md) | "零训练事后扫描关闭"：按 scan2 冻结规则主探针应进下一步，关闭依据来自次要探针与合同外诊断，现档存疑；自训参照模型变体未测（CANON §6 D12）。 |
| [research/curfew-trigger-2026-10-21/step1/ADDENDUM2.zh-CN.md](../curfew-trigger-2026-10-21/step1/ADDENDUM2.zh-CN.md) | "补给依赖性按任务分"改为"按切点距形成的时间分"（类比档）（CANON §6 D17）。 |
| [research/curfew-trigger-2026-10-21/step1/REPORT.zh-CN.md](../curfew-trigger-2026-10-21/step1/REPORT.zh-CN.md) | "toy 上断供后门不消退 / 断供也无效"：toy LT50 s1105 的断供发生在形成后约 350 步、ASR 早已饱和；差别在切点距形成的时间，不在任务（类比档）（CANON §6 D17）。 |
| [research/curfew-trigger-2026-10-21/step3screen/REPORT_STEP3.zh-CN.md](../curfew-trigger-2026-10-21/step3screen/REPORT_STEP3.zh-CN.md) | "两个候选关闭"只对 C1 拐角成立（低于跨 seed 地板）。C2 随机方向尖锐度仪器分辨率不足（±0.003 奈特），现档合同执行失败 → 未测，不入关闭清单（CANON §6 D9、D10）。 |
| [research/curfew-trigger-2026-10-21/step3screen/SUMMARY_FOUR_TASKS.zh-CN.md](../curfew-trigger-2026-10-21/step3screen/SUMMARY_FOUR_TASKS.zh-CN.md) | 干净尖锐度不是"关闭"，是合同执行失败 → 未测（CANON §6 D10）。 |
| [research/false-alarm-audit-2026-10-22/REPORT.zh-CN.md](../false-alarm-audit-2026-10-22/REPORT.zh-CN.md) | 本报告有以下错误，已被跟进推翻。（1）§0 第 2 条与 §2.2 的根因"窗口没按起飞对齐"是错的：窗口本来就按 t_risk 对齐，差的是 R 臂上升时间。2101 饱和前只有 5 点，M17/A29 不复现，应记低等级科学失败，不是"未测"。R 窗均值把 1106 与 1107 写反了，且用的是算术平均。"K29 2/2 重现"不成立，2102 那一半是饱和后的抖动。（2）§2.1 的 0.96/0.97 是 [101,205] 窗，form+50 是 0.95/0.95；形成前单一目标是 0.79，不是 0.80。（3）§2.3 与 §5 的回退防御已被 Z6 算出"回退并继续"不收敛（推导档），只剩"检出即拒绝"。（4）§2.8 快照差谱已用正确对照测过，仍分不开，属没拿到结果。（5）§5 列出的零 GPU 事项已由 Z1–Z5 做完。见 CANON §6 D2、D4、D11、D13，以及 research/false-alarm-followup-2026-10-22/README.zh-CN.md。 |
| [research/false-alarm-followup-2026-10-22/Z2_cmemx_windows.md](../false-alarm-followup-2026-10-22/Z2_cmemx_windows.md) | 单一目标 s1006 [1,100] 原值 0.7947，应为 0.79（重复四舍五入成了 0.80）；补测已冻结为 CONTRACT_K4，jobq 03300/03301 排队（CANON §6 D4、§7）。 |
| [research/false-alarm-followup-2026-10-22/Z6_rollback_loop/Z6_REPORT.zh-CN.md](../false-alarm-followup-2026-10-22/Z6_rollback_loop/Z6_REPORT.zh-CN.md) | 去向以 PLAN_FINAL 为准；k4 的实际判据见冻结的 CONTRACT_K4：AUROC_in ≥ 0.85 且高于置换零分布 97.5 分位 → 通过，≤ 0.75 → 关闭，其余不可判 → 补两 seed（CANON §7、D3）。 |
| [research/graft-preconditions-2026-10-21/GRAFT_PRECONDITIONS.zh-CN.md](../graft-preconditions-2026-10-21/GRAFT_PRECONDITIONS.zh-CN.md) | 所引 Curfew HANDOFF 判死清单多条已降级：c_mem_x k4 形成窗未测、输入侧未被否证、单行 loss 一块地两 seed 未见、快照差谱没拿到结果；单一目标 c_mem_x 形成后 0.95/0.95（CANON §6）。 |
| [research/next-stage-plan-2026-10-23/PLAN_FINAL.zh-CN.md](../next-stage-plan-2026-10-23/PLAN_FINAL.zh-CN.md) | 阶段 0 的 0.1–0.4 已完成（盘余 309 GB，三份合同冻结）。与冻结合同冲突处以合同为准：B3 "有效 ≥ 10"在合同里是 10–53 不可判、≥ 54 支持；k4 主读数负例 = 同运行干净行、零分布 = 置换，不是"配对 A0 同步均值"；PL 的 M30 是第 246–278 步（d≈50–80），不是 K(d≈100–200)（CANON §1、§2、§7）。 |
| [research/painpoint-first-steps-2026-10-21/gm1/RESULTS_GM1.md](../painpoint-first-steps-2026-10-21/gm1/RESULTS_GM1.md) | "与 disperse-kills-trigger 方向一致"：分散杀死扳机现为未测（CANON §6 D4）。 |
| [research/session-handoff-2026-10-23/HANDOFF.zh-CN.md](../session-handoff-2026-10-23/HANDOFF.zh-CN.md) | §4 的口径统一已落成 CANON 与 SUPERSEDED。§2 分叉表与 §3 的"有效（r ≥ 10）"与冻结的 CONTRACT_B3 冲突：合同里 r ≥ 54 才是支持，10 ≤ r < 54 是不可判、修复路保持未测，这一格去向需用户定。§3 的"XO 14 次训练"应为 8 次（14 = XO 8 + SP 6）（CANON §4 P1、§7）。 |

## 设计依据已取代（冻结合同，判档保留）（19 份）

| 文件 | 横幅内容 |
|---|---|
| [attribution-microscope/supplement/jump/CONTRACT.md](../../attribution-microscope/supplement/jump/CONTRACT.md) | "logit 本身加速"这一档名不能当跨网格的绝对加速量引用；"暴露轴对齐"读法已被 r2 推翻（CANON §1、§4 P2/P8）。 |
| [attribution-microscope/supplement/jump/dy/CONTRACT_DY.md](../../attribution-microscope/supplement/jump/dy/CONTRACT_DY.md) | 预测理由里的"起跳噪声约 30 步"已作废；DY 判"测不了"，不受影响（CANON §1）。 |
| [attribution-microscope/supplement/jump/dy/CONTRACT_DY2.md](../../attribution-microscope/supplement/jump/dy/CONTRACT_DY2.md) | 预测理由里的"起跳噪声约 30 步"已作废；DY2 判"测不了"，不受影响（CANON §1）。 |
| [attribution-microscope/supplement/jump/im/CONTRACT_IM2.md](../../attribution-microscope/supplement/jump/im/CONTRACT_IM2.md) | 主读数 E_trig 在构造上等于 logit 的触发分量（RESULTS_IM2 第 20–23 行），本合同现归为合同执行失败（设计错误）（CANON §4 P9）。 （横幅内行号指插入横幅前的原文，现需 +4。） |
| [attribution-microscope/supplement/jump/ps/CONTRACT_PS.md](../../attribution-microscope/supplement/jump/ps/CONTRACT_PS.md) | 判档带用的"噪声尺度 30 步"依据已作废（持续过零极差 4.3）；"不敏感"判档不受影响（CANON §1）。 |
| [attribution-microscope/supplement/jump/r2/CONTRACT2.md](../../attribution-microscope/supplement/jump/r2/CONTRACT2.md) | D1 的 0.15 阈值按逐点噪声校准，真实残差更大，现档存疑；QB 档名同 CANON P8 的限制（CANON §4 P8/P12）。 |
| [attribution-microscope/supplement/jump/sf/CONTRACT_SF.md](../../attribution-microscope/supplement/jump/sf/CONTRACT_SF.md) | σ = 30 的设计依据（含随机填充臂容差、跟随带 ≤ 30）已取代：持续过零极差 4.3 步，首次过零极差 33 步来自冒头回落。判档与随机填充臂读出按冻结规则保留（CANON §1）。 |
| [attribution-microscope/supplement/jump/sp/CONTRACT_SP.md](../../attribution-microscope/supplement/jump/sp/CONTRACT_SP.md) | 带宽依据"初始化约 30 步"与参照 1001 = 360 都已取代（≤ 5 步；1001 = 330）。冻结判档"不一致"保留，现档"渐变"（CANON §1、§4 P4）。 |
| [attribution-microscope/supplement/jump/tp/CONTRACT_TP.md](../../attribution-microscope/supplement/jump/tp/CONTRACT_TP.md) | 参照 1001 = 360 与"噪声约 ±30 步"均已取代；TP 已撤下，未判档（CANON §1、§4 P13）。 |
| [attribution-microscope/supplement/jump/xo/CONTRACT_XO.md](../../attribution-microscope/supplement/jump/xo/CONTRACT_XO.md) | 1001 的 10 步网格 t50 是 330（360 是 20 步网格且漏测 330 步）；更正后的 MAE 见 CANON §4 P1。 |
| [research/asr-theory-2026-10-22/DISCRIMINATION.md](../asr-theory-2026-10-22/DISCRIMINATION.md) | 分析前盲写的判读规则，正文不改；它的 ±30 / σ = 30 容差依据（"初始化移动 30 步"）已作废，持续过零极差 4.3 步。SF 的实际判档以 RESULTS_SF 为准（CANON §1、§4 P5）。 |
| [research/curfew-trigger-2026-10-21/eqcity/CONTRACT_GEOM.zh-CN.md](../curfew-trigger-2026-10-21/eqcity/CONTRACT_GEOM.zh-CN.md) | F1 的依据"机制 toy 不消退"不可比：toy LT50 的断供点在形成后约 350 步、ASR 已饱和（CANON §6 D17）。 |
| [research/curfew-trigger-2026-10-21/eqcity/CONTRACT_GRADTRIG.zh-CN.md](../curfew-trigger-2026-10-21/eqcity/CONTRACT_GRADTRIG.zh-CN.md) | 前提"零 GPU 阶段判死 loss 路"已降级为"一块地两 seed 未见"（CANON §6 D6）。 |
| [research/curfew-trigger-2026-10-21/eqcity/CONTRACT_TRIGGER.zh-CN.md](../curfew-trigger-2026-10-21/eqcity/CONTRACT_TRIGGER.zh-CN.md) | 前提里的"切断后锁定"与学习率衰减混淆（CANON §6 D17）。 |
| [research/curfew-trigger-2026-10-21/factor/CONTRACT_FACTOR.zh-CN.md](../curfew-trigger-2026-10-21/factor/CONTRACT_FACTOR.zh-CN.md) | 立项前提"c_mem_x 已判死"不成立（k4 形成窗从未测）；F2 主判据与 DISPERSE 继承同一错位窗口 [101,205]（CANON §6 D4、D5）。 |
| [research/curfew-trigger-2026-10-21/selfreport/CONTRACT_SELFREPORT.zh-CN.md](../curfew-trigger-2026-10-21/selfreport/CONTRACT_SELFREPORT.zh-CN.md) | 立项前提"逐行相似性一族已关闭"已过期（CANON §6 D5）。本合同的判死结论（探针无差别）仍成立（D8）。 |
| [research/curfew-trigger-2026-10-21/step3screen/CONTRACT_STEP3.zh-CN.md](../curfew-trigger-2026-10-21/step3screen/CONTRACT_STEP3.zh-CN.md) | 前提里的三条（loss 路判死、c_mem_x 死于分散、A1/A0 快照差谱逐点重合——当时的对照其实是 cf 对 phrase）均已过期（CANON §6 D4、D6、D11）。 |
| [research/k4-retest-2026-10-23/CONTRACT_K4.md](../k4-retest-2026-10-23/CONTRACT_K4.md) | 第 53 行的背景数字有两处转述误差：单一目标 0.96/0.97 是 [101,205] 窗（form+50 为 0.95/0.95）；形成前单一目标 s1006 为 0.79，不是 0.80。不影响本合同判据（CANON §6 D4）。 （横幅内行号指插入横幅前的原文，现需 +4。） |
| [research/painpoint-first-steps-2026-10-21/gm1/CONTRACT_GM1.md](../painpoint-first-steps-2026-10-21/gm1/CONTRACT_GM1.md) | 设计依据"梯度一致性线本仓已三合同判死"已取代：k4 形成窗未测（待测），输入侧 c_in_cmem_x 未被否证（CANON §6 D4、D5）。 |
## 有意未处理

- **未并入 main 的分支**：`origin/claude/keen-hypatia-azj9c9`（10-13 至 10-20 防御线：r02 外推、JSD、jump-timing、late-backdoor、mid16、Curfew 车队、参数路线、RECAP 与其 RESEARCH_INDEX）、`origin/claude/mechanism-line@f1cd817`、`origin/claude/ip-results-20261013`、`claude/determined-galileo-407bo2`。按"只提交 main、不动分支"的规则不加横幅，过期点集中写在 CANON（D1、D13–D19、P10、P15）。
- **冻结合同里的判死规则本身**（例如"D3 不成立 → 判死"）：规则不是过期结论，只有当合同同时陈述了过期前提时才加横幅。
- **结果 JSON / CSV**（含跨模型 `inventory.csv`）：原样保留；改标在 `research/false-alarm-followup-2026-10-22/Z5_crossmodel_relabel/`。
- `research/asr-jump-three-tracks-2026-10-10/` 只有 docx，未读未改。
- 与跃升线无关的研究目录（Atlas、COCO、T2I、HanLab 阅读包等）不在本次范围。
