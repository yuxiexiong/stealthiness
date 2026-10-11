# TARE：提交前文字与图表检查

检查对象：`/Users/ruizhixu/Desktop/TARE_ICLR2027.pdf`，81页；页码均为当前PDF页码。2026-09-26检查。已结合页面渲染、PDF文字和补充包中的TeX定位；本次没有修改论文。

## 先改文字、标签和错误引用

| 位置 | 当前问题 | 最小修改 |
|---|---|---|
| p31图4、p34图5、p42图7、p52图9 | 图中仍写`removal (DiD)`、`removal residue`、`corrected`，正文已把DiD称为interaction。读者会以为图和正文在报告不同的量。 | 同一DiD量统一称`interaction (DiD)`；图5(b)的`corrected`改为`DiD-based`；图7红字`by removal`改为`under DiD`。只针对本文该量，不要全局替换文献原话或描述真实移除操作的removal。图5相邻图注也同步。 |
| p8行420、422；p79行4262 | `gains −1.11`、`gains −2.43`把收益措辞和成本符号混在一起。 | 保留原数字和置信区间，改为`has a clean-arm cost of −1.11 … pt` / `has a clean-arm cost of −2.43 … pt`，再说明负成本表示准确率上升。不要只删负号而留下原区间。 |
| p16、24、30、31、32、81 | 多处`Eq. 3.3`引用了一个并不存在的编号公式。`eq:knob`实际挂在3.3节的行内公式后。 | 引用该段定义时改成`Section 3.3`；p30重复表达改成`the two endpoints reported in Section 3.3`。在源文件搜索`eq:knob`逐项处理，避免新增公式编号牵动全文。 |
| p43表29，行2309 | 行名直接印出了`(tab:bb convention)`内部标签。 | 删除括号，或改为`(this table’s convention)`。源：`main.tex:2217`。 |
| p78表54、p79表56；同组p77表53、p78表55 | 表54/56有两列都叫`victim`，一列是cosine/never，另一列是准确率。 | 调度策略列统一改`scheduler`；准确率列改`victim CA (%)`，clean表对应列改`twin CA (%)`。较长表头分两行。生成器：`build_sched10.py:239`、`build_sched11.py:190`。 |
| p79行4254–4255 | 把1.84 pt说成fine-tuning单独的收益；p6表2中1.84是FT/FP/NAD合并值，FT单独是1.99。 | 保留1.84和后续数值，将方法名称改成`the pooled FT/FP/NAD baseline`，明确这是合并值。源：`appendix_sched.tex:603`。 |
| p71表45图注末句 | 写`A0 and A3 are in Table 26`，但表26只有A3，没有A0。 | 最小修改为`A3 is reported in Table 26.`，删去错误的A0位置指引；若保留A0指引，应指向实际报告安装门槛结果的段落。源：`appendix_sched.tex:354`。 |
| p6最后一行→p7 | 粗体小标题只留下一个孤立的`The`，后文隔过图表才接上。 | `The level does not always:`中的第一个空格改为`~`，即`The~level does not always:`，重编译后确认位置。 |
| p10行519–520 | `27.1(the ANP…`少空格。 | 改为`27.1 (the ANP…`。源中可把`\sedGpuH\fi (the…`改为`\sedGpuH\fi{} (the…`。 |

## 再处理实际看不清或相撞的图表

| 位置 | 实际阅读问题 | 最小修改 |
|---|---|---|
| **p6表1** | 核心结果表的置信区间明显太小，整页阅读费力。同页表2更清楚。 | 将很长的paired列表头缩为`paired contrast (pt)`，公式留在图注/式4；减轻整表缩放并提高区间字号。只增大图片分辨率无效，这是排版字号问题。 |
| **p34图5(a)** | 斜线旁`what the cost column assumes`的末端与纵轴标题相撞。 | 删除长说明，或改短为`y=x`放在虚线中段。解释已在图注。源：`main.tex:1662`。 |
| **p54表36、p57表40、p71表44、p77表53** | 横向列数多，整张表缩得过小；区间尤其难读。 | 表36把六种防御分为上下两组；表40把published/tare分上下两块；表44把防御成本与baseline/gate分块；表53可把防御列分两组并重复attack/scheduler。保留数值和脚注。拆列之后减少或取消整表resizebox，而不是继续缩字号。 |
| p24图3 | 灰色流程说明过小，框内公式比说明清楚很多。 | 图当前仅0.58 textwidth，可先增至0.75–0.80，再检查算法和覆盖标注是否被挤坏；或在原图源中只增大灰字。源：`main.tex:1090`。 |
| p48图8(a) | 灰色`all three algorithms land here: under-forget`伸出绘图区，视觉指向不清。 | 缩为`under-forgetting region`并移入灰色区域，或分两行放置。 |

图4、图9及图8的上述标签已经嵌在PDF图资产里；补充包没有找到对应完整绘图源。修改时应在实际论文目录找到原图源再导出，单改LaTeX图注不会改变图中文字。图7绘图源在`analysis/fig_rerank.py`。

## 有余力再加的一句说明

p29表19的`asr50/asr90/asr99`没有把阈值定义直接告诉读者。源代码`code_supp/margin_dose.py:38–44`比较的是熵指数之差，不是概率百分比。可在caption补：

> `asr50`, `asr90`, and `asr99` are the fractions of paired prompts for which exp(H_trigger) − exp(H_clean) exceeds 50, 90, and 99, respectively; H is the mean next-token entropy over eight greedy generation steps.

## 改完只需复核

重新编译后按图号、表号查看修改页及相邻页，确认正文仍止于第9页、图注与图表没有分离、数值/负号未改变、引用已更新。正文图1/2未发现明显裁切、图例遮挡关键数据或符号丢失；不建议为统一风格重排它们。附录留白不作为今晚优先项。

详细依据：`ddl_main_visual.md`、`ddl_figures.md`、`ddl_tables.md`，均在本文件所在目录。
