# 机制 toy 部署与排队实查

日期：2026-10-10。代码完成，服务器CPU验收通过，已按用户授权提交原GPU1队列。**提交时两个任务均queued；本记录不声称GPU toy已运行。**

## 完成更新：2026-10-11读取实际回执

原worker已在北京时间2026-10-10 22:29:25自动启动GPU toy，22:33:19完成；22:34:34启动一次CPU完成审计，22:34:35完成。两个job均done，root complete实际true。22:28的queued/空闲快照发生在启动前约85秒，未发现调度故障。

GPU阶段回执wall232.5176653471s，包含载入/CPU/保存，不是纯GPU计算秒；一次CPU完成审计wall0.0498880893s，passed/0 mismatch/noTorch/CUDA未初始化。实际360身份＋522toy条件生成、108梯度条件前向、2算子资格前向、446底层model forward，零新训练。

实际结果报告已取回并核对原SHA：[RESULTS_TOY.md](RESULTS_TOY.md)。联合last6层attention＋MLP有部分作用，拆分复核移植增加3/12命中、撤回减少2/12，未见clean准确率误伤；匹配随机组各1/12。端点梯度、直线IG、时间候选的top1都为同一个attention组，当前没有时间排序优势或十倍效率证据。以下保留原部署记录，不用后来的done替换历史queued事实。

- Root：`/workspace/cross-model-asr/20261010_asr_mechanism_toy_r01`
- 队列：`/workspace/claude-jump/jobq`；原worker：3052130；只用物理GPU1，GPU0留其它项目。
- 顺序：`099cmmt_010_qwen_s1006_mechanism_toy` → `099cmmt_020_completed_cpu_audit`。后继有GPU完成与CPU准备文件硬依赖；原父任务与最终统计均实际done。
- 原idle10分钟/warm3分钟规则保持。外部GPU实验无队列done marker，由原worker检测GPU1空闲后接续，不kill外部进程、不另开worker、不手工上卡。
- Manifest SHA256：`e8a0e3d50bf23861d7a9696aef259673fa9a8bb407671b63b5cc3592231f772e`
- 冻结代码包SHA256：`b17d6c19adae0428546f9776fb9c70fdff7b06b7d261109f2323c302af007801`；代码运输包122880 bytes，无模型/数据/环境下载。

## 已实际通过

服务器24项CPU测试、0 skip、CUDA未初始化。其中18项标准库合同/合成回归，6项实际PEFT0.17.1小型算子测试，分别在独立CPU进程运行，避免把模拟层当真实PEFT资格。

两个测试子进程wall为0.2626413200s及3.7502322029s，包含启动/import，不是CPU处理器时间。服务器完整Toy0计算0.4196303878s，范围为穷举/评分/反例重算，排除保存；真实输入prepare子命令wall0.1262107063s。以上测试、合成与prepare已包括在server deployment parent成本，不再次叠加。独立GPU任务及完成审计各生成唯一成本回执。

真实旧材料有clean两端正确的38个flip、18个stable，足够按原顺序锁定互斥6题探索与12题拆分复核。12粗组和所有78项匹配随机组合已预冻结。权重只借用80/81/82/83，不新训练。

准备时fresh free79585210368bytes，2GiB剩余预算＋15GiB要求18245926912bytes通过；没有清理任何旧状态。

## 下一步自动执行

GPU任务先做原端点和有效算子资格，再锁定排序、测单组/有限联合；达到预定净作用条件才做12题复核。随后CPU任务只读重判保存生成、效应/聚合/选择/调用覆盖，输出一次验收与服务器中文报告 `results/REPORT.zh-CN.md`。

上限360次身份生成＋306次探索＋条件216次复核＝882次条件生成；另108次梯度条件前向及2次零探针资格前向，实际底层模型forward另记。全部零新训练/零新增formal。

未取得GPU新结果，不给队列ETA；原生资格失败阻断，科学无信号也如实完成，不扩模型、seed、剂量或自动展开MAttr/S3。具体执行规则见 [TOY_EXECUTION.md](TOY_EXECUTION.md)。
