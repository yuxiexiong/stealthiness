# 今晚DDL：附录表格可读性检查

仅检查渲染、标签和引用；不重评科学结论，不要求补实验。实看了81页原PDF中指定的全部29页：15、16、20、21、27、29、32、33、40、43–47、49、54–58、71、73–80。先看contact sheets，再逐页用`view_image`检查120 dpi原图。以下页码为PDF页序，行号为打印行号。**所查页没有发现文字实际重叠、越过纸边或表格内容被裁切；优先修下面四项。**

## 1. 先改：Table 29暴露内部引用标签

- **位置：p43 Table 29，行2309。** 行名实际印成 `paired post-defense contrast, n=11 (tab:bb convention)`。
- **源位置：** [main.tex:2217](/private/tmp/tare_iclr2027_panel/main.tex:2217)。
- **最小改法：** `(tab:bb convention)`改为`(this table’s convention)`，或直接删除该括号。`tab:bb`就是本表的label，不必自引用成Table 29。
- **为何今晚改：** 一行文本替换即可消除明显的编译源标记泄露，无需改数值。

## 2. 先改：Table 54/56两个表头都叫victim

- **位置：p78 Table 54行4162；p79 Table 56行4221。** 第一个`victim`列实际列出`cosine / never / cosine (damaged)`；第二个`victim`列实际列出91.64等防御前准确率。视觉上是两个完全相同的表头，却表示不同量。
- **相邻同类：** p77 Table 53行4144、p78 Table 55行4180的第一个`victim`也实际是scheduler；数值`twin`列可同时明确单位。
- **源位置：** 生成器 [build_sched10.py:239](/private/tmp/tare_iclr2027_panel/supplement/sched/analysis/build_sched10.py:239)、[build_sched11.py:190](/private/tmp/tare_iclr2027_panel/supplement/sched/analysis/build_sched11.py:190)；对应四个`tab_sched{10,11}_{clean,pub}_body.tex`均在line4。调用位置 [appendix_sched.tex:552](/private/tmp/tare_iclr2027_panel/appendix_sched.tex:552)、557、563、568。
- **最小改法：** 第一个列名统一改`victim scheduler`或更短的`scheduler`；数值列改`victim CA (%)`，clean表的数值列改`twin CA (%)`。可用两行表头控制宽度。若重新生成表格，应改生成器表头，避免只改输出文件后被覆盖。

## 3. 优先改善三个确实过小的表格，别继续整体缩放

下列均已在原尺寸渲染中确认：表头和均值明显小于正文，区间需要额外放大才能稳定辨认。`pdftohtml -xml -zoom 1`的取整字号读数也显示约4–5 pt主体、约3 pt区间；此为排版诊断的近似值，并非声称精确字体尺寸。

| 位置 | 具体问题 | 今晚最小排法 | TeX位置 |
|---|---|---|---|
| **p71 Table 44，行3808–3812** | 13列：攻击、scheduler、twin大小、三种防御、pooled、published、三个baseline、A4、A3横排；约4 pt主体/3 pt区间 | 同一caption下分上下两块：A为各防御/pooled成本；B为baseline、twin大小和gate；两块重复attack/scheduler行标签 | [appendix_sched.tex:350](/private/tmp/tare_iclr2027_panel/appendix_sched.tex:350)，输入`tab_sched3_body.tex` |
| **p54 Table 36，行2877–2882** | 六防御及各CI被一次性缩到整页宽；约5 pt主体/3 pt区间，最下A3/range行也拥挤 | 上块FT/FP/NAD，下块FT-SAM/SAU/i-BAU；重复attack/scheduler；各自保留对应A3/range | [appendix_sched.tex:18](/private/tmp/tare_iclr2027_panel/appendix_sched.tex:18)，输入`tab_sched4_body.tex` |
| **p57 Table 40，行3044–3050** | 左侧多重标签加右侧published/tare各三列；约4 pt主体/3 pt区间，11行压在很矮区域 | published与tare上下分块，重复attack/rate/schedule；保留差值行，避免删数或减小字体 | [appendix_sched.tex:112](/private/tmp/tare_iclr2027_panel/appendix_sched.tex:112)，输入`tab_sched21_body.tex` |

这些都使用`resizebox{...}{!}`，只在外面增加`small`或`footnotesize`往往又被缩回去。拆列后再用正常`footnotesize`主体、适度行距；确保最终PDF中的区间不再是约3 pt。无需全篇统一重排。p15 Table 6脚注、p16 Table 8、p77 Table 53也密，但未发现裁切；今晚先处理上述最严重三表即可。

## 4. 一行补注：Table 19的asr50/asr90/asr99不是自明表头

- **位置：p29 Table 19，行1522–1527。** 相邻正文只说three thresholds，未告诉读者50/90/99阈值加在哪个量上，容易误读为概率百分比或分位数。
- **源位置：** [main.tex:1393](/private/tmp/tare_iclr2027_panel/main.tex:1393) caption，line1397表头；实际定义见 [code_supp/margin_dose.py:38](/private/tmp/tare_iclr2027_panel/code_supp/margin_dose.py:38)：lines42–44比较`exp(H_trigger)−exp(H_clean)>t`，不是预测概率>0.5/0.9/0.99。
- **最小改法：** 不换数值，在caption补一句：

> `asr50`, `asr90`, and `asr99` are the fractions of paired prompts for which exp(H_trigger) − exp(H_clean) exceeds 50, 90, and 99, respectively; H is the mean next-token entropy over eight greedy generation steps.

保留这一定义即可，今晚不扩展成指标有效性讨论。

## 交付前只需一次回看

以上若修改，重新编译后用原尺寸查看涉及页及相邻页：检查拆表后caption与表体是否仍同页、页码/引用是否更新、CI符号和脚注有没有被截断。无需重新训练或重新统计。本报告未修改源文件。
