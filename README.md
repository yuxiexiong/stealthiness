# Stealthiness：研究代码与证据索引

本仓库保留后门成本诊断、VLM 归因与 ASR 短窗口跃升研究的代码、实验协议和结果。各条线的结论以对应结果记录为准；代码检查、作业完成和科学主张成立分别记录。

入口整理日期：2026-10-12（北京时间）。[研究材料总索引](research/RESEARCH_INDEX.md) · [本地 / H20 / GitHub 核对记录](research/repository-reconciliation-2026-10-12/README.md) · [文件存放与上传规则](research/repository-reconciliation-2026-10-12/STORAGE_POLICY.md)

## 当前研究入口

| 研究线 | 代码 / 计划入口 | 已有结果及范围 |
| --- | --- | --- |
| 跨模型 ASR 与短窗跃升 | [协议与源码](attribution-microscope/supplement/jump/cross_model/README.md) | [1% 后续复核汇总](attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/README_ZH.md)；模型、剂量、分母与未完成项按该记录区分 |
| ASR 机制 toy | [实验设计](research/asr-mechanism-experiment-2026-10-10/EXPERIMENT_DESIGN.md) | [Qwen 80→83 结果](research/asr-mechanism-experiment-2026-10-10/RESULTS_TOY.md)；探索性联合动作，不是独立机制确认 |
| ASR 监测与防御 toy | [实验计划与历史尝试](research/asr-defense-toy-2026-10-10/TOY_EXPERIMENT.md) | [r02 完成结果](research/asr-defense-toy-2026-10-10/r02/RESULTS.md)；未找到合格投毒回退点，保留负结论 |
| ASR 测量 toy | [第三条线设计](research/asr-defense-toy-2026-10-10/third-line-toy/EXPERIMENT_DESIGN.md) | [实际结果与限制](research/asr-defense-toy-2026-10-10/third-line-toy/RESULTS.md)；单前向传感器与生成复用的开发小试，尚非完整防御提效 |
| VLM 归因与 Atlas | [LLaVA 入口](attribution-microscope/README.md) · [Qwen 入口](attribution-microscope-qwen3vl8b/README.md) | [Atlas](attribution-microscope/atlas/README.md) · [发现索引](attribution-microscope/topics/README.md) · [跃升补充实验](attribution-microscope/supplement/jump/README.md) |
| 文本角色、语法与跨模态归因 | [研究方向](research/atlas-syntax-transfer-2026-09-21/RESEARCH_DIRECTION.md) · [实验设计](research/atlas-syntax-transfer-2026-09-21/EXPERIMENT_DESIGN_three-hypotheses.md) | [已有报告](research/atlas-syntax-transfer-2026-09-21/REPORT.md)；待验证假设与已观察结果分别保留 |
| 实例分割与扩模型计划 | [COCO / SAM3 设计](research/coco-sam3-segmentation-2026-09-27/EXPERIMENT_DESIGN.md) · [模型族扩展设计](research/atlas-model-family-expansion-2026-09-27/EXPERIMENT_PLAN.md) | 设计材料；不把计划登记当成实验完成 |

## 历史研究与阅读材料

- [早期可视化、诊断与事实修复](attribution-visualization/README.md)：包含完成结果、未支持的主张和历史预算。
- [OA 成本诊断](experiments/OA.md)、[Cordyceps](experiments/CORDYCEPS.md)、[BadVision](experiments/BADVISION.md)、[Grond](GROND_PROBE_PLAN.md)：保留原适配与证据边界。
- [HanLab 阅读包](research/hanlab-2025-2026/README.md)、[T2I 路线评估](research/t2i-phd-decision-2026-09-15/REPORT.md)、[其他研究与会议材料](research/RESEARCH_INDEX.md)。
- [旧版根 README](research/repository-reconciliation-2026-10-12/LEGACY_ROOT_README_2026-09-10.md)：原说明完整保留，其中的“当前先做 OA”和 48 GPUh 不作为现行研究优先级或预算约束。

## 代码检查与文件存放

早期轻量工具的 CPU 检查入口仍为 `python3.12 -m unittest discover -s tests -v`；各研究线使用各自文档中的环境与检查命令。CPU 检查不验证 GPU 结果。首页不会启动训练或队列。

本次新增文件单文件上限为 **2 MiB**。模型、检查点、环境、训练数据、私有逐题输出、大型 HTML / 图集 / PDF / 压缩包只登记位置与大小，不纳入补传。既有大文件和历史分支保留；本次不清理历史、不删除服务器材料。服务器运行副本不等于独立备份，实际备份状态见核对清单。
