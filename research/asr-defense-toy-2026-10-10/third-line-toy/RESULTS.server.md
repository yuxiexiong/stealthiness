# 第三条线测量toy实际结果

已见seed1101上的16个保存状态；0训练更新；开发小试，不是未见轨迹确认。

运行结论：saved_state_measurement_completed

|判据|结果|
|---|---|
|a_native_exact|True|
|b_exact|True|
|c_exact|True|
|invariance|True|

A/B传感器计时中位配对比：4.8798。
扩展面板实际执行：True。

GPU过程实际wall 1766.818324秒，包含载入、恢复、预热、测量、proof与失败尝试；分项不再次叠加。
sensor计时与包含raw-vector SHA证明的整条路径wall分列，完整成本见costs/gpu_measurement.json。

C不通过exact仍作为负例保留；所有策略可共享等价执行优化。连续读数/rank不是ASR或序列概率。
策略比较仅28–40已知窗口内的开发回放，确认调用引用旧原生oracle和计时，是反事实估值；没有总体FPR、跨模型SOTA或完整十倍效率结论。
