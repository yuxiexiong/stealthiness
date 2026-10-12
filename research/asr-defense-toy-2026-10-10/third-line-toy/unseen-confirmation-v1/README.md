# 未见初始化轨迹确认（第三条线阶段 3）

| 文件 | 用途 |
|---|---|
| [CONTRACT_UNSEEN.md](CONTRACT_UNSEEN.md) · [contract.json](contract.json) | 冻结设计、判档、预注册预测 |
| [EXECUTION.md](EXECUTION.md) | 执行状态：**未部署、未训练** |
| [code/](code/) | 运行、准备、审计、回放与 CPU 测试；开发版规则模块逐字节复用 |
| [jobs/](jobs/) | 原 jobq 三任务规格（前缀 `102cmmu`） |
| [manifest.json](manifest.json) · [make_manifest.py](make_manifest.py) | 源码 SHA、借用收据、服务器路径 |

CPU 检查：`python3 -m unittest discover -s code -p 'test_*.py'`（需 numpy 与 CPU torch 运行其中的张量测试）。本目录不启动训练或队列。
