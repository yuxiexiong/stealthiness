# 16:33新版：截稿前可修的确定问题

核查文件：桌面`TARE_ICLR2027.pdf`（81页，SHA256 `6602d9332998958e8732bf318c50817a73e07ee446a68975cc970ca060f5db8b`）与`TARE_ICLR2027_supplementary.zip`（SHA256 `a3a7ad0c4d3ee17254c8fac1dfe6805e321ad577cdc57cf9787dffa2f368bf4a`）。2026-09-26核查；以下页码和打印行号对应此版本。本次只检查可验证的错误与实际阅读问题，没有修改原稿。

## 已确认修好

- 81页均已有官方审稿页眉。
- 式6的代数关系正确，正文第2页与第4.7节已统一到paired误差。独立读取最新CSV重算，`0.12–0.68`、`1.58–6.79`以及非ABL合并区间`±0.51`均匹配。
- 正文表1的成本分解、表2的四格与计数、表3的30/31、表4的口径已做定向核对，未发现新的确定数字错误。不同门槛或不同估计量得到不同数字，没有被误报为矛盾。
- 补充包的两份SHA256SUMS共68个文件均匹配。本轮没有全量执行复现实验或重建所有产物。

## 优先修：明确错误

### 1. p47表33/34：表注仍把另一种误差对应到式6

表33行2503–2504说误差是式6的量。用表中的clean seed 1检查：式6应是防御后的准确率之差，`65.3−61.8=+3.5`；表中`+3.0`实际是两个drop之差，`8.9−5.9=+3.0`。两种计算本身都可以报告，但不能写成同一个量。

表34行2517–2518也把`Qwen drop−Llama drop`说成当前paired形式的`ĉ_int`的误差。

**最小修法：保留表内现有数字，只明确两张表报告的是DiD-form误差。**

表33对应句子替换为：

> The DiD-form surrogate error is the twin’s drop minus the surrogate’s drop; it does not involve the poisoned arm.

删除原句中的`of Eq. 6`，后面的defender-draw说明保留。表34将`what the substitution adds to ĉ_int`替换为`the error of the DiD-form surrogate estimate`。正文式6及已更新的主文数字不需要回退。

### 2. p79行4254–4255：1.84的来源名称错了

现在写成benchmark的fine-tuning收益，但1.84是FT/FP/NAD合并值；单独FT是1.99（p6表2）。生成脚本`build_audit.py`也确实读取pooled值。

**最小修法：**保留1.84和后面的计算，将`the benchmark’s own fine-tuning`改成`the pooled FT/FP/NAD baseline`。源：`appendix_sched.tex:603`。

### 3. 多处“Eq. 3.3”是错误引用

涉及p16、24、30、31、32、81。源码中的`eq:knob`挂在3.3节的行内定义后，抓到了章节号，论文没有编号为3.3的公式。

**最小修法：**将8处`Eq.~\ref{eq:knob}`改成`Section~\ref{sec:admiss}`。p30原句另顺成`the two endpoints reported in Section 3.3`，避免重复括号。无需新增编号公式。

### 4. p43表29：内部标签露出来了

行2309实际印着`(tab:bb convention)`。

**最小修法：**删除该括号，或改为`(this table’s convention)`。源：`main.tex:2215`。

### 5. p78表54、p79表56：两列同名但含义不同

两个表头都叫`victim`，前列装的是cosine/never等scheduler，后列装的是准确率。

**最小修法：**分别改为`scheduler`、`victim CA (%)`；表53/55的同类列也同步。长表头可换成两行，避免增宽。若重新生成表格，需要同步改`build_sched10.py:239`、`build_sched11.py:190`里的表头字符串。

### 6. p71表45：A0指到了没有A0的表

行3818–3819写`A0 and A3 are in Table 26`，但p40表26只有A3，没有A0列。

**最小修法：**改为`Every twin passes A4; A3 counts are in Table 26.`，使用现有自动引用。源：`appendix_sched.tex:354`。

## 图表与顺手修复

| 位置 | 新版仍有的问题 | 最小修改 |
|---|---|---|
| p34图5(a) | 斜线旁长句与纵轴标题右端相撞 | 删除`what the cost column assumes`，或缩成`y=x`放在虚线中段。 |
| p6表1 | 核心结果的置信区间字号偏小，整页阅读费力 | 缩短paired列表头为`paired contrast (pt)`，减轻整表缩放，适度提高区间字号。 |
| p57表40、p71表44 | 列多、整体缩得很小，区间尤其费力 | 表40按published/tare拆上下两块；表44按成本列与baseline/gate列拆块；有余力再做，不建议今晚全篇重排。 |
| p48图8(a) | 灰色说明长句伸出绘图区右边 | 缩成`under-forgetting region`并放入灰色区域。 |
| p24图3 | 灰色流程标签偏小 | 增大原图说明字，或适度增大图宽，检查算法及覆盖标注位置。 |
| p6页末→p7 | 孤立的粗体`The`与后半句分开 | `The level`改为`The~level`，编译后回看。 |
| p10行519–520 | `27.1(the ANP…`缺空格 | 源码中的`\sedGpuH\fi (the…`可改为`\sedGpuH\fi{} (the…`。 |

图内旧`removal/corrected`措辞未作为本轮单独的优先修改项；本轮不因微小语义边界要求重写论证，也不重提需要新实验的意见。

## 补充包的一项低成本修复

`supplement/README.md`目前只写Python 3。最新`analysis/build_chain_numbers.py:233–234`的f-string在本机Python 3.9.6下会报`f-string expression part cannot include a backslash`；Python 3.12.14可解析。建议将复现环境明确为Python 3.12，并给出实际使用的matplotlib版本，或者将这一处字符串预处理移出f-string以兼容旧Python。本轮只做了语法检查，未将解析通过写成全量复现通过。

## 修改后的最少回看

重新编译，检查受影响页和相邻页；确认正文仍止于第9页、图表与caption没有分离、负号和区间未被意外改动、引用更新完成。优先处理上面1–6和图5的文字碰撞。

详细验证记录：同目录`main_numeric.md`、`appendix_tables.md`、`visual.md`。
