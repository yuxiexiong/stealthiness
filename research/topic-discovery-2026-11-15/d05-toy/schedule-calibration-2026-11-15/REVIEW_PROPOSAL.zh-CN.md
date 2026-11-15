# 实现与启动前审查

2026-11-15。用户已明确授权实现/CPU检查/Advisor后直接上卡五个干净前缀、工程bug可修、不加其他实验/main发布。此前候选1已审，本轮请以实现为被审对象，不附和。

入口：PLAN、schedule.py、execute.py、supervise.py、CPU_CHECK.json。复用原data.py/metrics.py/run.py且hash未改。原来源位于同上轮D05运行根，wrapper --original-root指向它；prepared另存重排事件，不修改原manifest/state。之前ROUTE补测报告与审查提供动机。

CPU已通过：5块各3840原行，W/P各20，W槽位各10；153600编码曝光/2400更新，前17920个样本对是原前10轮辅助投影（560更新），后58880对用seed与固定域新shuffle（1840）。对比原epoch_examples构造的每行/输入token/标签完整多重集，不只是总量相等；所有source-epoch,row事件恰好一次。15个删除替换/前段换位/后段换位故障算例均拒绝。原门槛6个边界算例通过。

CPU从5份真实state00恢复原架构，逐tensor精确相等；Adam空state及参数组与新实例完全相等且load后不变；CPU RNG恢复一致，CUDA RNG只在CPU核存储类型/长度/hash，尚未宣称GPU恢复通过。科学forward/updates=0，CPU9.06秒。执行将在每块初始GPU evaluate时与原summary00（含所有logits hash）精确比较，不通过即失败。只实例化一次optimizer、全程不reset，step560和2400核optimizer步数。

execute.py循环只有5块×20个120更新窗口，evaluate和保存于0/120/…/2400，复用原evaluate/prefix_qualified；任何合格数都不调用污染分支。新增epoch字段是120更新窗口索引，日志另写update，不能当原样本完整遍历轮数。

监督器从前次已验证supervise.py复用，仅改命令和路径，外层记全调用含CPU预检/IO成本；内层时间包含于外层，不重复相加。若监督中断而child尚活，不写费用已闭合。单卡绑定空闲UUID，不干预其他GPU任务；当前GPU0有别的任务，拟用GPU1，初始readout精确保真为必要工程门槛。预计2–4分钟、6分钟规划余量非硬帽。

请独立核代码/CPU回执/关键计数与原设计，特别找初态/RNG/优化器恢复、W/P成对、源重复正例、重排编码、门槛及阶段边界的漏洞。必要修改落为意见，主agent修改复测后聚焦审；审查记录仅写本目录ADVISOR_REVIEW.zh-CN.md。你不运行模型或修改执行代码。通过后用户已授权直接运行，无需再次请示。
