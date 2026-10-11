# 最新版正文数字与公式检查（16:33版）

核验文件：`/Users/ruizhixu/Desktop/TARE_ICLR2027.pdf`，SHA256 `6602d9332998958e8732bf318c50817a73e07ee446a68975cc970ca060f5db8b`，81页。仅使用 `/private/tmp/tare_latest_1633` 的最新PDF提取文本、源文及数据。未改稿、未运行训练或提交者脚本。

## 确定发现：1项

### 式6的主文已更新，附录表33/34的caption仍把DiD误差称为paired误差

- **位置：**p47，表33 caption，行2503–2504；表34 caption，行2517–2518。这是§4.7所引用的S27，属于式6修改的剩余引用闭环问题。
- **表33原文：**“The surrogate’s error is the clean-versus-clean quantity of Eq. 6, the twin’s drop minus the surrogate’s drop”.
- **确定证据：**最新版式6（p5，行227）是 `Q(Dθ_surr) − Q(Dθ_twin)`。表33第一条替代模型的对应值是 `65.3 − 61.8 = +3.5`；该表误差栏却为 `+3.0`，它实际等于 `8.9 − 5.9`。这是两种不同形式，不是舍入差异。该表计算的DiD误差本身正确，错的是将它直接称为式6的量。
- **表34原文：**“The error, Qwen drop minus Llama drop, is what the substitution adds to ĉ_int”. 其中 `ĉ_int` 当前由式6定义为paired估计量；这张表同样使用的是drop之差，即DiD形式的替代误差。
- **最小修法：**无需改数字或重跑实验。把两张表caption明确标成DiD形式：
  - 表33替换为：`The DiD-form surrogate error is the twin’s drop minus the surrogate’s drop; it does not involve the poisoned arm.` 删除这里的 `of Eq. 6`。
  - 表34把 `what the substitution adds to ĉ_int` 改为 `the error of the DiD-form surrogate estimate`。或者使用显式的 `ĉ_int^DiD`，但需确保该符号有定义。
- **严重度：**今晚应修的公式/引用不一致；两处caption即可完成，主文数值无需回退。

## 已修复确认

1. **式6与§3.4的paired解释一致。** `ĉ_int = Q(Dθ_surr) − Q(Dθ_pois)`，减去式4的 `c_int` 后确实得到 `Q(Dθ_surr) − Q(Dθ_twin)`。本轮未发现该代数等式错误。
2. **主文的TARE-Z数字已经切到paired口径。** p2行085–086为`0.12–0.68`；p9行467–470为`+0.00 [−0.51,+0.51] paired`、`0.12–0.68`和`1.58–6.79`，均与最新版生成宏一致。摘要只说twin-free estimator，没有残留冲突的旧误差数值。
3. **独立读取最新CSV复算，更新数字正确。** 用 `supplement/data/backdoorbench_11seed_cells.csv` 的防御后twin准确率，按不同种子的全部有序配对计算paired MAE，得到：NAD 0.123636、FT 0.256364、FP 0.321818、SAU 0.678909；NC 1.575273、I-BAU 2.405091、ANP 3.198909、ABL 6.790545。四舍五入后正是主文两组范围。以安装为单位的非ABL pooled paired区间为约`[−0.51205,+0.51205]`，匹配`±0.51`。
4. **页眉已恢复。** 新版正文页面的提取文本均出现`Under review as a conference paper at ICLR 2027`。
5. **正文主表与关键叙述未见其他确定数值bug。** 复核表1的成本/paired/DiD关系、表2四格与131/132及44/44计数口径、表3的30/31（15/16+15/15）、表4的7攻击/50安装及1/69分母，并检查正文引用关系。最新BadNets数据中式3–5恒等式最大浮点残差约`7.1e−15`个百分点。表1和表4的NC口径在caption中已分别注明，不能把其不同数值误报成矛盾。

没有为凑数另列微小语意建议；本轮不重提旧科研问题，也不建议补实验。
