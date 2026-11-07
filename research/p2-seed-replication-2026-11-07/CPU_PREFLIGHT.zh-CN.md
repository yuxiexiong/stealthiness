# CPU 预检结果

结论：**PREPARED_CPU_PASS_NOT_LAUNCHED**。GPU 使用 0 卡时；未加载 LLaVA，未创建实验队列，未修改原实验或清理检查点。

## 实际通过的检查

| 检查 | 结果与证据 |
|---|---|
| 原环境真实数据重建 | 两份 20,000 行数据逐行身份通过；原 A/R/B 的共同 H 保留；源文件哈希、配置与新 sampler 逆映射通过 |
| 真实 Trainer CPU sampler | seed 1007 的全 20,000 行遍历匹配；CUDA 隐藏，仅随机小 Dummy 参数，无预训练模型 |
| 源导出重复核验 | 服务器本地重新导出的 inputs.json SHA256 为 `5afb9523d05977fd0c3c251d2509141af65229d2ffc5795eeaaad2499849bc8c`，与本地一致 |
| 运行环境 | 原训练库版本、相关源文件哈希、模型缓存 revision/文件大小/blob 路径通过；未重新全量读取基座权重 |
| CPU 单元/联检 | 24 项全部通过：prepare 4、训练身份门 3、更新审计 12、读数 5；后者覆盖 168 份模拟读数文件及错误、缺失、删失、反向与失败门 |
| 埋点与恢复 | 单独的原环境 CPU 小模型 17 个检查全部通过：埋点前后参数/Adam/scheduler/RNG 一致，完整状态恢复到共同终点一致，错误数据消费被拒绝 |
| GradScaler | 原环境 CPU finite→inf→finite 三次人工更新：实际优化器跳过中间一次，scheduler 位置 1→1→2，scale 8→4，保存恢复通过 |
| 历史 LR | CPU 重建 2,100 条旧日志全部精确匹配；明确定位 B67 含毒批跳过 |

原始证据：[最终联检日志](audit/verified_cpu_preflight.log)、[数据物化日志](audit/final_cpu_preflight.log)、[埋点检查](audit/instrumentation_cpu_result.json)、[scaler 检查](audit/scaler_cpu_result.json)、[机器汇总](CPU_PREFLIGHT.json)。

CPU 临时目录是 `/tmp/p2s7-cpu-final-cUFHq7`；它包含本次测试代码和隔离物化数据，不是未来正式运行根。最初未压缩上传过慢，停止了该次临时上传，改为压缩代码、在服务器本地只读重建源导出；之后核验了同一源哈希。未停止任何实验进程。

## 尚不能宣称验证过的部分

- 没有实际运行第二种子 LLaVA，不能宣称 GPU 训练、真实第 200 步完整状态、全模型恢复或科学效应已经通过。
- CPU 完整状态恢复是工程语义检查；CUDA 随机状态由原 Trainer 的保存路径及新完整性门负责，未来真实保存后仍须核实。
- 基座以固定 revision、缓存 blob 路径和文件大小定位；没有在本轮重新计算 14 GB 基座全量 SHA256。
- 预算来自旧训练/读数记录，未测试新串行读数流程的实际 GPU 耗时。

正式运行前必须重新执行启动身份门与磁盘门。本轮停在可审核准备包，不进入 GPU 执行。
