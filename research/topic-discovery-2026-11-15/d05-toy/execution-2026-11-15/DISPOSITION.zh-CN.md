# D05 启动咨询处理

用户已明确授权确认后直接上卡及手动修bug；该许可仅覆盖既定D05，不扩大科学条件。

- **采纳Advisor监督中断纠正。** child仍活着时，不再写FINISHED或闭合成本；写SUPERVISOR_INTERRUPTED、runner PID和截至当前的观察时长，留待人工核查，不自动杀进程/重试。success/failure/interruption三个mock案例通过，未启动实际子进程；监督最终SHA见SUPERVISOR_CPU.json。
- **已补齐远端原日志。** CPU_SUITE_REMOTE.log为实际服务器七项标准库回归原输出，13.377秒，全部通过；不是本地50.976秒日志的复制。CPU_RUNTIME.json是独立CPU合成模型工程检查，1.619秒、两次合成更新、0GPU，不是研究学习结果。
- **科学代码和固定材料未改。** 远端重建manifest等于main已审5633348d…60c7；部署8个文件按DEPLOYMENT_HASHES核准。新增仅CPU工程验收与单次监督，原run/data/metrics及所有科学门槛不变。
- **接受成本和停止边界。** 完整单卡保留墙钟由监督开始/结束记录，包含入口CPU预检/IO，不冒充GPU kernel忙时；所有失败保留。资格不足按原协议停止，不将之叫bug；真正工程问题才手动修复并验证，优先复用经身份核准的有效结果。

Advisor支持上述材料闭合后直接执行，无须再次请求用户。实际启动、进展和结束以本目录后续机器回执为准，本文不是运行成功证明。
