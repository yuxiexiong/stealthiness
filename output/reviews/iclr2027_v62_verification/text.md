# v62 文字问题验收

对象：`/Users/ruizhixu/Desktop/TARE_v62_full/submission/TARE_ICLR2027.pdf`，85 页。对照 ZIP 提取源码 `/private/tmp/tare_v62_audit`。只验收上轮项目；未修改源文件、未运行实验。

| 项目 | 状态 | 新 PDF / 源码证据 |
|---|---|---|
| 1.84 误写单独 FT | 已修 | p83 行4453–4454明确写 `the benchmark’s fine-tuning family (FT, FP and NAD pooled)`；保留1.84。 |
| 8处 Eq. 3.3 | 已修 | PDF 全文不再出现 `Eq. 3.3`；对应源码使用 `Section~\\ref{sec:admiss}`。p16行843、p24符号表、p30正文、p31行1640/1657/1662、p32行1694、p85行4522均已换 Section 3.3。 |
| 表29泄漏内部标签 | 已修 | p43行2309为 `Table 29’s convention`，不再是tab:bb。 |
| 表53–56两列都叫victim | 已修 | p81表53/54为 `schedule` + `twin (%)` / `victim (%)`；p82表55/56同样修复。 |
| 表45 A0错误指向表26 | 已修 | p73表45图注为 `A3 is in Table 26 and the WaNet victim’s A0 in Section 4.2.` |
| p6末孤立The | 未修，小排版问题 | 新p6最末行仍以粗体 `The` 结束（已实际看渲染）；源码main.tex343仍 `\\textbf{The level does not always:}`。最小修法：`The~level`。 |
| p10缺空格 | 未修，小错字 | p10行519–520仍 `27.1(the ANP…`（已实际看渲染）；main.tex470条件宏后空格被吞，可改 `\\sedGpuH\\fi{} (the…`。 |
| README Python兼容声明 | 未修（补充包复现说明） | supplement/README.md9仍仅写 Python 3。使用/usr/bin/python3 3.9.6仅ast.parse检查build_chain_numbers.py，在第234行报 `f-string expression part cannot include a backslash`；触发结构位于233–234行。未执行builder。最小文档改法明确 Python >=3.12；若要求兼容旧Python，应移出f-string中的反斜杠表达式。 |

优先：论文文字的五个具体错误均修好。剩余两项 PDF 小问题不影响结果；README最低Python版本说明建议补。Python问题属于此前补充包复现意见，不是本轮新发现的研究问题。
