# 机制与防御：两篇完整中文初稿

2026 年 11 月 5 日。用已完成实验写出文章，再判断研究位置。包含完整论文正文、独立定位判断、证据索引及审阅记录。本轮没有新增训练或模型推理。

## 直接阅读

| 文件 | PDF | 可批注、修改的 Word | Markdown 源稿 |
|---|---|---|---|
| 机制篇：后门快速形成的顺序敏感性与模块承载 | [阅读 PDF](output/01_机制论文初稿.pdf) | [Word](output/01_机制论文初稿.docx) | [正文](01_机制论文初稿.md) |
| 防御篇：在线损失过滤何时保护学习？ | [阅读 PDF](output/02_防御论文初稿.pdf) | [Word](output/02_防御论文初稿.docx) | [正文](02_防御论文初稿.md) |
| 两篇成稿之后：研究位置与分合判断 | [阅读 PDF](output/03_两篇成稿之后的判断.pdf) | [Word](output/03_两篇成稿之后的判断.docx) | [全文](03_两篇成稿之后的判断.md) |

两篇论文均包含摘要、引言、相关工作、实验设置、方法、结果、讨论、局限、结论及参考文献。正文中的 E-Mxx/E-Dxx 是内部证据定位标记，对应下方索引；它们不代替外部文献引用。

## 成稿后的判断

机制稿已有顺序干预、留出模块定位和经校准的更新账本，可以形成范围明确的机制实证研究。若要上升为形成机制解释，还缺对未用于选规则的轨迹作出事先预测，并由实际干预区分候选机制与一般训练变化。

防御稿已有条件性有效结果，也有强 LT 比较、过滤选择性与学习收益失配、暂态抑制后追赶等实测结果。它目前更接近防御实证研究；稳定优于强基线的新方法贡献尚未建立。

合并的关键不是增加篇幅，而是在同一体系中连接机制对象、预先选择的防御动作及持续的安全—正常学习收益。目前保留两篇独立工作稿最便于判断各自贡献。具体“立住”与后续扩展的区别见第三份文件。

## 证据与检查

- [写作范围及证据截止](evidence/WRITING_SCOPE.md)
- [机制主张与源文件](evidence/mechanism_claims.md)
- [防御主张与源文件](evidence/defense_claims.md)
- [源文件存在性及 SHA 清单](evidence/SOURCE_VERIFICATION.json)
- [防御主要表格数值复核](evidence/defense_numeric_validation.json)
- [独立定位审阅](evidence/independent_position_audit.md)
- [全文结论级审阅](evidence/manuscript_review.md)
- [机制逐页检查](evidence/mechanism_visual_review.md)、[防御逐页检查](evidence/defense_visual_review.md)、[定位判断逐页检查](evidence/assessment_visual_review.md)
- [最终交付核验](evidence/DELIVERY_VALIDATION.json)

本地底稿为 2026-10-27 的全线归档，核心主张追溯到登记文档及机器可读结果。2026-11-05 只读检查服务器选定文本结果树，没有发现晚于归档截止的新文件；这不是对全部服务器资产的穷尽检查。本文是供研究讨论和修改的完整初稿，尚不是已补齐所有投稿验证的最终稿。

## 编辑与复现排版

修改顶层三份 Markdown 后，可用 `working/build_documents.py` 生成 Word。PDF 使用 Documents 渲染工具，并加载 `working/fonts` 中的 Noto CJK 字体；最终交付的 PDF 已完成中文显示、公式符号和逐页版面检查。工作目录中的中间渲染版本仅供排版排错，阅读以 `output/` 为准。
