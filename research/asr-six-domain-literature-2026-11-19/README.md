# ASR Rising 六领域全文查重

检索任务启动于 2026-11-19。目标是查清与“投毒训练怎样形成并改变后门行为”有关的已有证据、重合与限制。六个领域均已完成A查重、B原文复核与同C独立查漏的有界闭环；验收见STATUS。各报告须与对应B/C记录及来源限制一起阅读。

| 领域 | 报告 | 核心问题 |
|---|---|---|
| 1 首次形成与显现 | [领域1](d1-formation/REPORT.md) | 形成前变化、正常与投毒学习的交互、变化怎样转为行为 |
| 2 保持与再显现 | [领域2](d2-persistence/REPORT.md) | 保持、衰退、潜伏、恢复与再激活 |
| 3 形成条件 | [领域3](d3-conditions/REPORT.md) | 数量、组成、时机、顺序、模型及训练条件 |
| 4 信号与评估 | [领域4](d4-signals/REPORT.md) | 训练前风险、训练监测、预警及其信息条件 |
| 5 干预与防御 | [领域5](d5-defense/REPORT.md) | 预防、抑制、清除与正常能力的关系 |
| 6 一般学习近邻 | [领域6](d6-generalization/REPORT.md) | 特征竞争、延迟泛化、阈值与有害行为学习 |

- [整体总结](SUMMARY.zh-CN.md)
- [去重文献索引](LITERATURE_INDEX.md) / [JSON](LITERATURE_INDEX.json)
- [查重与SubAgent使用方法](AGENT_WORKFLOW.zh-CN.md)
- [当前状态与续接点](STATUS.md)
- [执行与结束协议](PROTOCOL.md)
- [跨领域接缝](shared/CROSS_DOMAIN.md)
- [用于比较的项目证据边界](shared/PROJECT_COMPARISON.md)

每域 `PAPERS.json` 记录材料身份、版本、实际阅读范围与本地来源；`SEARCH_LOG.md` 记录检索与引用链。`REPORT_A.md` 保留初稿，后续 `B_REVIEW.md`、`REVISIONS.md`、`C_INDEPENDENT_SEARCH.md`、`C_AUDIT.md` 分别记录原文复核、修订、独立发现与查漏结果。全文不可得或重要补充缺失会明确限制结论，不以未找到论文证明创新性。

## Git归档范围

本次归档包含原创报告、PAPERS索引、检索/阅读笔记、A/B/C记录、使用方法与验收状态；不包含下载的第三方论文PDF、全文提取、截图或源码缓存。索引中的fulltext_path与部分原文路径指向原工作区本地缓存，远端阅读请使用文献官方URL；恢复审核时须重新取得对应版本的原文。未上传缓存不影响已留存的阅读记录，但这些记录不能替代后续原文核验。
