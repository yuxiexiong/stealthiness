# ASR防御toy实际结果

运行结论：`profile_infeasible_within_frozen_budget`。

物理GPU1；完整科学分母T1250；最多160更新/臂。

|训练臂|实际更新|V60质量曾达标|
|---|---:|---|

|臂/策略|候选step|H64确认|
|---|---:|---|

这是同轨迹影子控制器的受控toy。独立端到端十倍提效、总体误报率、未知触发器防御与跨模型通用性均未证明。

局部跃升只接受实际≤12更新、ASR增≥50个百分点的点对；未观测处保持未知。

完整数值与成本：result.json、costs/gpu_trial.json；CPU只读验收：monitoring/completed_cpu_audit.json。
