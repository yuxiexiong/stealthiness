# ASR 跃升：观察、对照与机制实验

入口更新：2026-10-12。本目录保留 VLM 跃升的冻结合同、代码、结果与偏离记录；[全仓库研究索引](../../../research/RESEARCH_INDEX.md)与[Atlas 主入口](../../README.md)提供上下文。

| 阅读线 | 入口与状态 |
| --- | --- |
| 首轮与第二轮 | [首轮结果](RESULTS.md)、[第二轮结果](r2/RESULTS2.md)、[偏离日志](LOG.md)；首轮 G0 未通过的部分不判，后续结果不抹去原失败 |
| 顺序与初始化 | [XO 结果](xo/RESULTS_XO.md)、[SP 结果](sp/RESULTS_SP.md)、[LP 结果](lp/RESULTS_LP.md)；按冻结合同区分控制结果与机制推测 |
| 归因是否领先 logit | [IM2 结果](im/RESULTS_IM2.md)：合同完成，主假设被否证；占比读数的提前不能当机制前兆 |
| 内部状态交换 IP | [冻结合同](ip/CONTRACT_IP.md)；本入口未找到正式结果报告，合同、预演和冒烟不算实验完成 |
| 其它对照 | [PS](ps/RESULTS_PS.md)、[QX](qx/RESULTS_QX.md)、[DY](dy/RESULTS_DY.md)、[DY2](dy/RESULTS_DY2.md)；保留各自条件与判档 |
| 跨模型观察 | [cross_model 入口](cross_model/README.md)、[10-10 补测最终报告](cross_model/results/20261010_uncertainty_followup_final/README_ZH.md)、[增量成本](cross_model/results/20261010_uncertainty_followup_final/final_incremental_costs.json)；不等同于共有机制证明 |
| 最新机制 toy | [设计](../../../research/asr-mechanism-experiment-2026-10-10/EXPERIMENT_DESIGN.md)、[结果](../../../research/asr-mechanism-experiment-2026-10-10/RESULTS_TOY.md)；完成有限联合与拆分复核，未显示时间排序优势 |
| 最新防御 r02 | [实际结果](../../../research/asr-defense-toy-2026-10-10/r02/RESULTS.md)；未找到投毒合格回退点，防御路径与十倍效率未获支持 |
| 第三条线测量 toy | [设计](../../../research/asr-defense-toy-2026-10-10/third-line-toy/EXPERIMENT_DESIGN.md)、[实际结果](../../../research/asr-defense-toy-2026-10-10/third-line-toy/RESULTS.md)、[阶段 3 冻结合同](../../../research/asr-defense-toy-2026-10-10/third-line-toy/unseen-confirmation-v1/CONTRACT_UNSEEN.md)；有测量执行收益，群体翻转预测未显示额外收益，限开发小试；新初始化确认未运行 |

每条线先读合同，再读结果与局限。`CONTRACT*.md` 是设计，`RESULTS*.md` 是报告；工程执行成功、CPU 验收通过和科学主张成立分别判断。本文不推断实时服务器状态，不把旧模型状态交换、权重移植或探索选择当训练形成机制的证明。
