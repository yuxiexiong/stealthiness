# ASR 跃升：观察、对照与机制实验

入口更新：2026-10-23。**现行口径只看 [CANON](../../../research/jump-canon-2026-10-23/CANON.zh-CN.md)**（读数定义、抖动带、失败标签、每个实验的现行档位）；过期文档登记在 [SUPERSEDED](../../../research/jump-canon-2026-10-23/SUPERSEDED.zh-CN.md)，顶部有横幅。最新交接：[10-23](../../../research/session-handoff-2026-10-23/HANDOFF.zh-CN.md)；计划：[PLAN_FINAL](../../../research/next-stage-plan-2026-10-23/PLAN_FINAL.zh-CN.md)；全仓库索引：[RESEARCH_INDEX](../../../research/RESEARCH_INDEX.md)。

## 在途（2026-10-23）

| 实验 | 状态 | 入口 |
|---|---|---|
| B3 优化器状态清零 | 合同冻结（LOG J53），jobq 03200–03204 等卡，未开跑 | [CONTRACT_B3](b3/CONTRACT_B3.md) |
| k4 形成窗补测（防御线扳机） | 合同冻结（LOG J54），jobq 03300/03301 等卡，未开跑 | [CONTRACT_K4](../../../research/k4-retest-2026-10-23/CONTRACT_K4.md) |
| PL 位置阶梯 K(d) | 合同冻结（LOG J55），GS0 过，**未排队**；B3 无效且 k4 关闭时才跑 | [CONTRACT_PL](pl/CONTRACT_PL.md) |

## 已完成（现行档位，详见 CANON §4）

| 实验 | 现行档位 | 入口 |
|---|---|---|
| r1 首轮 | 本网格内跃迁不是 argmax 假象；"暴露轴对齐"被 r2 推翻；加速比不可跨网格比 | [RESULTS](RESULTS.md) |
| r2 第二轮 | 累计毒量不预测起飞（盲测 8 种子 ρ = 0.23）；D1 加性模型存疑 | [RESULTS2](r2/RESULTS2.md) |
| XO 顺序 × 初始化 | 起飞由顺序决定（8 次训练，MAE 12.5 对 70）；初始化在持续类读数上 ≤ 5 步 | [RESULTS_XO](xo/RESULTS_XO.md) |
| LP 过零读数 | margin 过零可替代 t50（14/14 相差 ≤ 20 步） | [RESULTS_LP](lp/RESULTS_LP.md) |
| PS 单行微扰 | 不敏感（5 次全 330） | [RESULTS_PS](ps/RESULTS_PS.md) |
| SP 定位 | 渐变（参照更正为 330 后重算；原判"不一致"） | [RESULTS_SP](sp/RESULTS_SP.md) |
| SF 骨架 × 填充 | 偏骨架（部分），四读数不一致需并列 | [RESULTS_SF](sf/RESULTS_SF.md) |
| zg 开场图与脉冲衰减模型 | 模型作为统一解释被否；单行约 1 logit、记忆约 40 步 | [RESULTS_ZG](zg/RESULTS_ZG.md) |
| QX | QX-A 存疑；QX-B 为线索（被本实验自身结果削弱） | [RESULTS_QX](qx/RESULTS_QX.md) |
| IM / IM2 归因是否领先 | IM 已撤回；IM2 合同执行失败（读数在构造上就是 logit 触发分量） | [RESULTS_IM2](im/RESULTS_IM2.md) |
| IP 内部状态交换 | 触发位置不承载差异（窄范围）；结果只在分支 `origin/claude/ip-results-20261013`（e7bc5b0） | [CONTRACT_IP](ip/CONTRACT_IP.md) |
| TP / DY / DY2 | TP 撤下；DY、DY2 测不了 | [CONTRACT_TP](tp/CONTRACT_TP.md)、[DY](dy/RESULTS_DY.md)、[DY2](dy/RESULTS_DY2.md) |
| 跨模型 | 多个模型 / 族观测到跳；SD3.5、FLAN-T5、Falcon-Mamba 大部分、Qwen3-VL 1% seed 1004–1007 是攻击未形成，不是动态阴性 | [cross_model](cross_model/README.md)、[Z5 改标](../../../research/false-alarm-followup-2026-10-22/Z5_crossmodel_relabel/) |

决策与偏离日志：[LOG](LOG.md)（编号接 J56）。机制线（ATLAS、跳的结构跨族广度）在分支 `origin/claude/mechanism-line@f1cd817`，不并入 main，先读该分支 `research/asr-mechanism-rivals-2026-10-15/STATUS.md` 与 `CLAIMS_TABLE.md`。

阅读规则：先合同、后结果；`CONTRACT*.md` 是冻结设计，`RESULTS*.md` 是当时的签名报告，正文不改；被取代处看文件顶部横幅与 CANON。
