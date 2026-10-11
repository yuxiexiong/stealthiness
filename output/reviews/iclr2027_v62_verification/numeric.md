# v62 式6与表33/34修复验收

范围：只复核上一轮明确指出的定义及数字一致性，不新增审稿要求。实际读取新85页PDF提取文本 `/private/tmp/tare_v62_audit/paper.txt` 及本次ZIP中源文件/数据。

## 结论：本组全部已修，未发现数值回归

1. **p47表33已修。** caption 现在明确写 “The DiD-form surrogate error is the twin’s drop minus the surrogate’s drop”，不再把误差称为 Eq.6 的量。第一行仍为 twin drop 8.9、surrogate drop 5.9、error +3.0，定义与计算一致。p46正文行2479也同步改为 “DiD-form surrogate-minus-oracle errors”。源码 main.tex:2361、2389。
2. **p47表34已修。** caption 现在写 “The error, Qwen drop minus Llama drop, is the error of the DiD-form surrogate estimate”，不再把它称为 paired hat c_int 的偏差。首行5.22−6.60=−1.38，五行均值−1.294，四舍五入−1.29；范围−2.47至−0.42与表注吻合。源码 main.tex:2408–2411。
3. **p5正文式6保持正确。** c_hat_int=Q(D surrogate)−Q(D poisoned)，且 c_hat_int−c_int=Q(D surrogate)−Q(D twin)。正文 paired estimand 及误差均为 post-defense level difference；附录表33/34清楚另标DiD。
4. **p9第4.7节数值保持正确，直接从本次ZIP CSV独立重算确认。** 32个 CIFAR calibration rows，先在每个installation内平均8个配置，再对4个installation取t区间，结果为−0.02203125 [−0.16553886,+0.12147636]，对应论文−0.02 [−0.17,+0.12]。11-seed BackdoorBench按式6全有序pair重算MAE：NAD0.123636、FT0.256364、FP0.321818、SAU0.678909；NC1.575273、I-BAU2.405091、ANP3.198909、ABL6.790545。对应正文两段范围0.12–0.68及1.58–6.79。排除ABL的seed-level pooled paired误差mean≈0、95%区间±0.51205055，对应+0.00 [−0.51,+0.51]。p2引言也保持0.12–0.68。

上述核验只读；未运行会重写build文件的生成器，也未修改投稿源稿。
