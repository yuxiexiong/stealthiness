# 10月19日晚至21日研究归档

主入口：[前情提要](RECAP.zh-CN.md)。范围覆盖57条实际干预差异、A/B的54条确认与投影分解、C/D的28条K29与头级实验，以及截至10月21日的分析、文献启发与尚未执行的路线讨论。

## 阅读顺序

1. [前情提要](RECAP.zh-CN.md)：做了什么、得到了什么、撤回什么解释、当前准备怎样走。
2. [第一批57条审计报告](../intervention-response-audit-2026-10-20/REPORT.zh-CN.md)。其原始执行代码在[intervention-response-2026-10-19](../intervention-response-2026-10-19/)。
3. [A/B合同](../intervention-confirm-decompose-2026-10-20/CONTRACT.zh-CN.md)与[完整结果](../intervention-confirm-decompose-2026-10-20/results-r02/REPORT.zh-CN.md)。代码、测试、验收、执行记录和任务清单在同目录。
4. [C/D合同](../k29-confirm-heads-2026-10-20/CONTRACT.zh-CN.md)与[完整结果](../k29-confirm-heads-2026-10-20/results-r03/REPORT.zh-CN.md)。代码、测试、验收、执行记录和任务清单在同目录。
5. [逐假设研究蓝图](../intervention-difference-2026-10-19/PLAN.zh-CN.md)与[共同研究目标](../TRAINING_DEFENSE_GOAL.zh-CN.md)。历史计划中的“尚未执行”描述是当时快照，后续实际执行以各批结果和本提要为准。

## 原始证据

`server-evidence/`镜像以下目录的全部`.py/.md/.json/.jsonl/.csv/.txt/.log/.sh`文件，排除缓存和二进制状态：

- `/workspace/cross-model-asr/20261019_intervention_response_r01/`
- `/workspace/cross-model-asr/20261020_intervention_confirm_decompose_r02/`
- `/workspace/cross-model-asr/20261020_k29_confirm_heads_r03/`

三批运行共589个文本文件，包含逐步训练日志、逐题生成/评分、完整曲线、运行结果和机器合同，不只上传报告摘要。[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)记录每个文件的服务器相对路径、字节数、SHA256及排除文件；本地逐文件核对与源端一致。

另镜像A/B首版验收失败、K批次前两版加载失败，以及实际执行源码`stage_icd_20261020_r02`、`stage_k29_20261020_r01/r02/r03`，共118个文本文件。[补充清单](ADDITIONAL_SOURCE_MANIFEST.json)与[本地核验回执](ARCHIVE_VERIFICATION.json)记录其身份。失败日志不计科学阴性。

## 保留在服务器的材料

模型权重、`state_final.pt`、fork中的adapter/Adam/RNG状态、缓存及共享数据集未加入普通Git。不是删除或遗失：运行状态文件的路径/大小及共享数据集的路径/大小/SHA登记在SOURCE_MANIFEST.json；共享数据位于`/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01/private/data/`，其中训练文件约26.7MB。本归档没有复制或公开该private目录。

原始状态用于精确恢复；仅下载本Git归档不能自动获得模型缓存与二进制状态。运行说明中的服务器路径仍是资产入口。

## 本次验证与动作边界

- 本次只读复制、核对、撰写和Git归档，没有启动训练或重新推理。
- 保留全部登记结果、反向效应、未完成记录与训练前工程失败，不将候选写成已确认防御。
- JSON、文件哈希、本地链接和Git上传大小/类型检查由归档回执记录；历史CPU/GPU验收不是本次重新运行的实验。
- 以隔离工作树基于远端最新实验分支提交，避免混入原工作目录中其他课题的修改。
