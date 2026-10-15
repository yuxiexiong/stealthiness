# A3 补充预注册：干净行推力是否随状态转正

2026-10-15，写于任何 A3 数据被查看之前（A3 s1105 任务此时在运行，输出未读）。起因：A2 总档 not_row_caused，用户批准推荐方案——不改 A2 判档，把核心主张改为「模型状态改变后，毒行与干净行对 gap 的推力都转正」，用 A3 已在记录的干净探针行来检验。服务器上的 `contract.json` 与源码受哈希守卫，本补充不改它们；判定在本地对 `a3.json` 计算（`code_local/addendum_a3.py`）。原 A3 判档（gating / autocatalysis / flat）照常报告，不受本补充影响。

## 量

干净探针行 = 第 t_fork+1 更新的前 2 条干净行（A3 已定义）。每个状态取两行 push_pre 的均值，记 q(t)。三段：
- 早段 E = phase1 的前 3 个状态（第 0、4、8 步）
- 母本末段 M = 母本臂最后 3 个保存点（t_jump+16..t_jump+20）
- B2 末段 B = B2 臂最后 3 个保存点

## 判（每母本）

- **clean_turns_positive**：mean q(E) ≤ 0.5·mean q(M)，且 mean q(M) > 0，且 mean q(B) < mean q(M)。
  即干净行在后门装上的状态下比早期推得更正，且没有毒样本的 B2 臂上推得更弱。
- **clean_no_turn**：mean q(M) ≤ 0，或 mean q(M) ≤ mean q(E)。
- 其余 between。

总档：≥ 2/3 母本 clean_turns_positive → **supported**；≥ 2/3 clean_no_turn → **refuted**；其余 mixed。

## 预测

supported。依据：A1 中母本臂干净更新的平均增量为正（0.46 / −0.02 / 0.75），B2 臂为负（−0.19 / −0.28 / −0.12）；A2 中 s1105 换掉毒行后爆发仍剩 1.1–1.7 logit。

## 已知的弱点（写在数据之前）

- 只有 2 条干净探针行，行身份噪声大；读数只作「方向」证据。
- push_pre 是一阶推力的 Adam 近似；A1 / A2 的真实更新是对照。
- refuted 时，A1 的「干净更新转正」只能解释为多步效应（单行一阶推力看不到），主张退回「毒行触发、干净行维持」的弱版本，停下讨论。
