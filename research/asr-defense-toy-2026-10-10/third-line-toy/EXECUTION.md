# 第三条线测量toy执行记录

2026-10-11用户明确批准写好小toy并上服务器。已实现并实际部署，零新增训练。

新隔离root：`/workspace/cross-model-asr/20261011_asr_measurement_toy_v1_gpu1_r01`；prefix `101cmmf`。原r02全部源码、状态、输出和成本保留。只借用13个poison28–40 adapter及3个clean140/157/160 state、原V60编码和passed小收据，不重跑旧审计或旧大模型hash。

11Python文件包含A/B/C测量、past-only规则开发比较、CPU数据准备和一次结果验收。源树SHA `e32f908dbbf87883a57843f053500b1bb82217e84c7589031cc7b1243211dcd8`，manifest SHA `e288f33fa6b5423eda72c3eaeb371d285527de987be152b4fa44efa79cedbbe9`。

最终本地28CPU测试passed/零skip；服务器原amic-q3环境28CPU测试passed/零skip/零failure/error/CUDA未初始化，服务器测试内层0.811224秒，已含唯一部署CPU parent。实际准备16个借用状态、60个原探针，新私有copy与原JSON SHA一致；旧state只stat和passed收据绑定，GPU运行时再核实际adapter复制。

北京时间05:17:40，三个任务已提交原jobq，提交回执均queued，0 running/done/failed：

1. `101cmmf_010_saved_state_measurement`：GPU1实际测量三路径；固定4题通过后才扩1/2/8/16题。
2. `101cmmf_020_completed_cpu_audit`：新输出一次空CUDA验收。
3. `101cmmf_030_final_markdown`：实际完成后生成Markdown与完成收据。

只用物理GPU1、原worker3052130，原10分钟idle/3分钟warm规则。GPU0未使用；没有CPU/GPU墙钟上限。05:02唯一资源快照两卡均空闲，这是历史实查，不代表之后一直空闲；不在本轮等卡反复采样。

提交时fresh free132654927872 bytes，required剩余阶段预算+15GiB reserve为28989792256 bytes，通过，没有清理。`est_min20`只是队列显示，不是可靠ETA或时间上限。

C不能exact复现时保留负例；B不能exact A、A不能exact原native或不变量失败则工程失败。每个n使用自己实际批次的信号、计价和校准，不借n16前缀冒充n1/2。传感器时间与包含全vocab SHA proof的总wall分列，全部验证/预热/失败成本仍计入GPU外层。

本批仅已见seed1101的开发小试，不启动新seed1102、修复训练或独立确认，不承诺十倍或SOTA。GPU科学结果仍待实际执行。

已有巡检`asr-toy-gpu1`已更新并实核ACTIVE，绑定本root，完成一次验收与本地中文Markdown交付后暂停；没有重复建立调度或新任务线程。

2026-10-11用户要求完成即回报后，已将同一巡检改为每分钟轻量检查并实核ACTIVE；完成或失败一经检测及时回报，正常检测延迟约1分钟加执行时间，不能承诺零延迟。

北京时间05:24:37一次fresh实查：本测量任务是第一个依赖全部满足的可执行任务，前面4个旧任务依赖未满足，不阻挡本任务。三任务仍queued，0 running/done/failed，GPU1无compute进程、原worker存活；正在按原10分钟连续空闲规则等待，尚无新GPU测量输出。GPU0也空闲但继续留给其它项目。证据：[实时状态](user_status_20261011T0523.json)。

北京时间05:28:30轻量实查：010已running，完成core测量poison28/29两个状态，已保存24次测量；020/030仍依赖等待，无失败。此轮未重复GPU快照或完成审计，科学结果尚待全任务完成。

用户随后要求改为每十分钟检查一次。已更新同一监视器并实核ACTIVE、间隔10分钟；不在间隔中持续等待或额外轮询。完成/失败一经检测即回报，通知延迟由此检查频率决定。

北京时间06:02:35实查全部完成：三job done、0 failed；960测量/16状态；唯一自动CPU审计passed、0 mismatch、noTorch/noCUDA。A原生字段、B raw读数、C生成与raw读数、不变量全部exact。GPU外层1766.818324秒，约29.45分钟，0训练更新。

已同步安全数值和Markdown。4题单forward传感器paired-median实测4.87976倍，但不替代ASR；C等价生成复用约1.236倍。4→1题约1.035倍且确认更晚，群体规则与普通斜率同效、多6次无风险确认，没有额外结构收益。完整[中文结果](RESULTS.md)、[安全数值](numeric_results.json)、[独立成本账本](cost_ledger.json)。

交付验证后，原监视器已由应用实际更新并实核PAUSED，原worker未停止，没有新GPU任务或新seed。

可复查：[机器合同](contract.json)、[源清单](manifest.json)、[服务器CPU准备收据](server_cpu_preparation_receipt.json)、[真实入队回执](queue_and_deployment_receipt.json)。[详细实验设计](EXPERIMENT_DESIGN.md)。
