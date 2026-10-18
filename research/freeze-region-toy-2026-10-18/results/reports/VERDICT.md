# 冻结区域 toy：判读（冻结规则自动生成）

| 候选 | 档位 | 有效母本 | 挡住且保住学习 | 挡住 | 形成 |
|---|---|---|---|---|---|
| M | **unclear** | ['1105', '1106', '1107'] | [] | [] | ['1105', '1106', '1107'] |
| D | **unclear** | ['1105', '1106', '1107'] | [] | [] | ['1105', '1106', '1107'] |

| 母本 | 臂 | 安全档 | 起跳步 t_form | 相对 R 推迟 | 峰值触发/60 | 400 步触发/124 | 学习保留 | 末点正常/60 | 停止原因 |
|---|---|---|---|---|---|---|---|---|---|
| 1105 | R | formed_unchanged | 53 | 0 | 60 | 124 | None | 50 | reached_400 |
| 1105 | M | formed_delayed | 68 | 15 | 31 | None | None | 34 | formed |
| 1105 | P | formed_unchanged | 53 | 0 | 42 | None | None | 28 | formed |
| 1105 | X1 | formed_other | 58 | 5 | 59 | None | None | 27 | formed |
| 1105 | X2 | formed_other | 58 | 5 | 49 | None | None | 28 | formed |
| 1105 | X3 | formed_delayed | 63 | 10 | 51 | None | None | 35 | formed |
| 1105 | D | formed_other | 58 | 5 | 34 | None | None | 9 | formed |
| 1106 | R | formed_unchanged | 92 | 0 | 60 | 124 | None | 56 | reached_400 |
| 1106 | M | formed_delayed | 132 | 40 | 54 | None | None | 42 | formed |
| 1106 | P | formed_delayed | 102 | 10 | 34 | None | None | 35 | formed |
| 1106 | X1 | formed_other | 82 | -10 | 23 | None | None | 34 | formed |
| 1106 | X2 | formed_delayed | 102 | 10 | 27 | None | None | 41 | formed |
| 1106 | X3 | formed_delayed | 127 | 35 | 60 | None | None | 39 | formed |
| 1106 | D | formed_delayed | 127 | 35 | 46 | None | None | 26 | formed |
| 1107 | R | formed_unchanged | 77 | 0 | 60 | 124 | None | 46 | reached_400 |
| 1107 | M | formed_other | 82 | 5 | 56 | None | None | 30 | formed |
| 1107 | P | formed_unchanged | 77 | 0 | 58 | None | None | 39 | formed |
| 1107 | X1 | formed_unchanged | 77 | 0 | 57 | None | None | 35 | formed |
| 1107 | X2 | formed_unchanged | 77 | 0 | 54 | None | None | 36 | formed |
| 1107 | X3 | formed_unchanged | 77 | 0 | 56 | None | None | 34 | formed |
| 1107 | D | formed_delayed | 92 | 15 | 53 | None | None | 31 | formed |



登记预测：{"M": {"claim": "delay_only", "tier": "类比 (B5b last-6-MLP freeze delayed 8/26/2; direction projection delayed 6/6/2)", "confidence": "medium-high"}, "D": {"claim": "delay_only or ineffective; point prediction delay_only", "tier": "推测", "confidence": "medium"}, "controls": {"claim": "P and X arms form; M delay not larger than the largest X delay on at least 2 mothers", "tier": "推测", "confidence": "low"}, "fidelity": {"claim": "all gates pass", "tier": "推导 (B0 of the fork experiment reproduced exactly)", "confidence": "high"}}
