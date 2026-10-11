# 新版提交前：附录表格明确错误

受审版本：桌面 `TARE_ICLR2027.pdf`，81页，SHA256 `6602d9332998958e8732bf318c50817a73e07ee446a68975cc970ca060f5db8b`；源包 `/private/tmp/tare_latest_1633`。以下只列5项确定、今晚可改的错误，按优先级排列；不涉及研究设计或措辞力度。页码为PDF页序，行号为打印行号。

## 1. 多处引用不存在的“Eq. 3.3”【新增】

**位置：** p16 Table 8脚注，行843；p32 Table 22 caption，行1694；另外还有notation表、Figure 4 caption及预算附录等，共8处源码引用。

**证据：** [main.tex:265](/private/tmp/tare_latest_1633/main.tex:265)把 `\label{eq:knob}` 放在普通段落的inline公式之后，没有编号equation环境；它实际继承了§3.3的section编号。新版PDF确实显示 `Eq. 3.3`，但所指是§3.3的“Manufactured cost”段落，不存在编号为3.3的公式。引用源位于main.tex的609、1076、1512、1529、1534、1553、1560、2729行。

**最小修法：** 将全部 `Eq.~\ref{eq:knob}` 改为 `Section~\ref{sec:admiss}`；该section label已在 [main.tex:254](/private/tmp/tare_latest_1633/main.tex:254)。或者仅把`Eq.`改`Section`、保留当前label。无需新增展示公式或改变后续公式编号。

## 2. 1.84的来源仍误写成单独fine-tuning【旧问题仍在】

**位置：** p79 S54，行4254–4255。

**证据：** 正文说1.84是benchmark自己的fine-tuning在clean twin上取得的收益。但 [build_audit.py:16](/private/tmp/tare_latest_1633/supplement/sched/analysis/build_audit.py:16)明确读取 `schWmsPool`，同行注释说明是FT/FP/NAD的pooled gain；[sched_numbers.tex:127](/private/tmp/tare_latest_1633/supplement/sched/build/sched_numbers.tex:127)给出−1.84。p6 Table 2、p8 Table 4中的单独FT值为−1.99。文字源：[appendix_sched.tex:603](/private/tmp/tare_latest_1633/appendix_sched.tex:603)。

**最小修法：** 把 `the benchmark’s own fine-tuning gains on the clean twin` 改成 `the pooled FT/FP/NAD clean-twin gain`，或完整写 `the pooled gain of the benchmark’s FT, FP, and NAD on the clean twin`。保留1.84与既有计算。本项只修数字的来源标签，不延伸到重参照的科学解释。

## 3. Table 29仍印出内部标签tab:bb【旧问题仍在】

**位置：** p43 Table 29，行2309，`paired post-defense contrast, n=11 (tab:bb convention)`。

**证据：** 新版文字与渲染均如此；[main.tex:2215](/private/tmp/tare_latest_1633/main.tex:2215)直接把内部label写成了正文。

**最小修法：** `(tab:bb convention)` → `(this table’s convention)`，或删掉括号；此label就是本表，不必新增自引用。

## 4. Table 54/56双victim表头【旧问题仍在】

**位置：** p78 Table 54行4162；p79 Table 56行4221–4222。两表均同时有两列名为`victim`：前列内容是`cosine / never / cosine (damaged)`，后列是91.64等准确率。

**证据：** 新版PDF及表生成器一致；[build_sched10.py:239](/private/tmp/tare_latest_1633/supplement/sched/analysis/build_sched10.py:239)、[build_sched11.py:190](/private/tmp/tare_latest_1633/supplement/sched/analysis/build_sched11.py:190)。四个生成的 `tab_sched{10,11}_{clean,pub}_body.tex` 的表头均位于line4。

**最小修法：** 前列改`scheduler`，数值列改`victim CA (%)`。同组p77 Table 53、p78 Table 55的前列也应统一为`scheduler`，数值`twin`可改`twin CA (%)`。只改表头；如重跑生成器，应修改上述header字符串，防止重新生成时覆盖手改。

## 5. Table 45把A0引到没有A0的Table 26【旧问题仍在】

**位置：** p71 Table 45 caption，行3818–3819：`A0 and A3 are in Table 26`。

**证据：** p40 Table 26，行2154–2158，实际列为attack、scheduler、published cost、clean arm、DiD、`A3 clears`；没有A0列，caption也没有逐行A0计数。源：[appendix_sched.tex:354](/private/tmp/tare_latest_1633/appendix_sched.tex:354)；被引用表体：[tab_sched_cols_body.tex:4](/private/tmp/tare_latest_1633/supplement/build/tab_sched_cols_body.tex:4)。

**最小修法：** 改为 `Every twin passes A4; A3 counts are in Table 26.` 不必为了修这个错误引用临时补数据列。

## 核查边界

已在新版全文做定向搜索并核对对应源码，实看新版渲染p16、43、71、78、79；另读取p40 Table 26及p6/p8相关数值。没有把旧版发现未经复核地沿用。未修改原稿、未重跑实验。本轮没有确认其他值得追加的表内算术错误；不把字号偏好、研究解释或细微语义作为bug填充清单。上述修改均不要求调整实验或结果数值。
