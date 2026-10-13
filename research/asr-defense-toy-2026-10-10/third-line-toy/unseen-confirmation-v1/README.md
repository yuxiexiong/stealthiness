# 未见初始化轨迹确认（第三条线阶段 3）

| 文件 | 用途 |
|---|---|
| [CONTRACT_UNSEEN.md](CONTRACT_UNSEEN.md) · [contract.json](contract.json) | 冻结设计、判档、预注册预测 |
| [RESULTS.md](RESULTS.md) · [results/](results/) | **实际结果（2026-10-13）：判档「探针提效稳定」，与预注册预测一致**；安全数值、成本、审计收据 |
| [EXECUTION.md](EXECUTION.md) | 执行记录：部署、第一次环境失败（0 更新）、重排、完成 |
| [code/](code/) | 运行、准备、审计、回放与 CPU 测试；开发版规则模块逐字节复用 |
| [jobs/](jobs/) | 原 jobq 任务规格（前缀 `102cmmu`）：010/020/030 为第一次提交（010 因缓存环境变量失败），011/021/031 为带完整离线缓存环境变量的重排 |
| [manifest.json](manifest.json) · [make_manifest.py](make_manifest.py) | 源码 SHA、借用收据、服务器路径 |

CPU 检查：`python3 -m unittest discover -s code -p 'test_*.py'`（需 numpy 与 CPU torch 运行其中的张量测试）。本目录不启动训练或队列。
