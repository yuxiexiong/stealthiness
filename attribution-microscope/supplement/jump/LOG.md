# supplement/jump · 决策日志

## 输入哈希（sha256 前 16 位）
```
cd82df36375df0c5 arm_clean.json
0fcd8dbbdbfc9355 arm_p_0_1.json
8ea331639bd7627e arm_p_0_2.json
73ba8c82c836a7fb arm_p_0_3.json
dd02de058cf53b61 arm_p_0_35.json
91a7e7603d667e0d arm_p_0_38.json
1bdb1d52ac31fafd arm_p_0_4.json
34ca12d28c205938 arm_p_0_42.json
356107d5e057d7db arm_p_0_45.json
a949e64106a024ca arm_p_0_4_ps2.json
212df74ff988ac24 arm_p_0_4_ps3.json
d18c8ccf627e2b17 arm_p_0_5.json
5054aa12e302019c arm_p_1_0.json
b3a399129e0ed09a arm_p_1_ps2.json
cd82df36375df0c5 arm_retrain_a.json
b67421f11f70afac log_CLEAN.jsonl
b232fdeafcf7349b log_P-1.0-D.jsonl
919a0ce59e63f274 log_P-1.0-D2.jsonl
17be0a77a1b38d7b log_RETRAIN-A-D.jsonl
93c1216f5c02ec94 orders.npz
```

## 记录

- J1 2026-09-29 冻结前：`export_order.py` 在服务器 CPU 上运行（/workspace/claude-jump，仓库外），
  结果为两个种子的顺序与 randperm 逐位相同；只看了这个布尔值和前 8 个下标，没算任何计数。
- J2 冻结前：`jump.py` 预演中发现 bootstrap 的 R 在前段斜率 ≤ 0 时为 inf，插值分位数得 nan；
  改为 `method="inverted_cdf"`（不插值）。发生在任何真实数据之前，不影响判据含义。
- J3 冻结前：`run.py` 做了只看结构的检查（成像步列表、行为测试步列表、样本数 60、剂量臂文件存在），
  没有打印任何 ASR、logit、loss 数值。
- J4 本地磁盘只剩 278M，clone 改为稀疏检出（只检出 behavioral、相关 maps 与本目录）。不影响测量。
