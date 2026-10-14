# Q1 分析摘要（机器生成）

## s1105 · 判类 unmeasurable

- argmax 命中 vs 生成命中：最大差 0，均差 0.00
- 主规则 0.10/0.90：测不了（cannot measure (flip not set by margin sign)）
- r2 规则 0.05/0.95：测不了（cannot measure (flip not set by margin sign)）

## s1106 · 判类 unmeasurable

- argmax 命中 vs 生成命中：最大差 0，均差 0.00
- 主规则 0.10/0.90：测不了（cannot measure (flip not set by margin sign)）
- r2 规则 0.05/0.95：测不了（cannot measure (flip not set by margin sign)）

## s1107 · 判类 between

- argmax 命中 vs 生成命中：最大差 1，均差 0.04
- 主规则 0.10/0.90：窗前 63 → 窗 [71, 79]；D 闸门一致率 0.931，加性 R² 0.941（D=0.350 CI [0.267, 0.483]，samples go their own way）
  - D2：W(5–95%) = 8.66 logit；v_pre = 0.904 /步，v_in = 1.423 /步，比值 1.57；按 v_pre 要 9.58 步，按 v_in 要 6.09 步
  - D3：自身加速的行占比 0.62（窗前斜率 ≤ 0 的行 0.00）
  - 阈值扫描宽度（步）：{'-8': 2, '-6': 6, '-4': 4, '-2': 4, '0': 6, '2': 9, '4': 15, '6': 61, '8': 61}；|τ|≤4 内宽度相等（匀速签名）：False
  - 断点：第 62 步，斜率 0.240 → 0.723；ΔBIC(直线−断点) = 22.7，ΔBIC(二次−断点) = 20.5
- r2 规则 0.05/0.95：窗前 59 → 窗 [71, 83]；D 闸门一致率 0.946，加性 R² 0.948（D=0.367 CI [0.283, 0.500]，samples go their own way）
  - D2：W(5–95%) = 8.42 logit；v_pre = 0.975 /步，v_in = 1.099 /步，比值 1.13；按 v_pre 要 8.63 步，按 v_in 要 7.66 步
  - D3：自身加速的行占比 0.25（窗前斜率 ≤ 0 的行 0.00）
  - 阈值扫描宽度（步）：{'-8': 2, '-6': 9, '-4': 5, '-2': 7, '0': 9, '2': 16, '4': 43, '6': 63, '8': 63}；|τ|≤4 内宽度相等（匀速签名）：False
  - 断点：第 61 步，斜率 0.247 → 0.652；ΔBIC(直线−断点) = 18.8，ΔBIC(二次−断点) = 17.2

## 判档：**mixed**

逐母本：{'s1105': 'unmeasurable', 's1106': 'unmeasurable', 's1107': 'between'}
