# 服务器排队与 GPU 准备入口

2026-09-10 15:21 UTC。**最近核验时真实 GPU 阶段尚未开始。** 队列已切换到软预算入口，队友两条总控仍存活。公开模型全部文件已通过服务器来源哈希校验；不能把准备完成说成已通过 GPU 验收。

## 当前实际部署

- 主机 `gqa-h20`，独立根目录 `/root/attribution-visualization-20260910`。
- 原 `0f7702c` 代码目录为 `toy48-code-ready2`；当前从它复制到 `toy48-code-soft-budget/attribution-visualization/visual-evidence-repair` 并应用本次软预算补丁。未改数据、模型或构建目标。修改文件身份见 `gpu-ready-evidence/soft-budget-admission.json`。
- 已有监视器 PID `258936`，目录 `toy48-queue`，每 300 秒检查一次，`nice=19`，脱离 SSH。没有新增 Codex automation；此前本地 automation 保持暂停。
- 等待队友总控 PID `4158784` / start_ticks `727135121`、PID `4158937` / start_ticks `727135621`。等待整个总控退出，不把两轮训练之间的短暂空闲当成实验结束。
- 队友总控结束、全部要求的输入存在后，检查两卡无计算进程、利用率 ≤5%、显存 ≤256 MiB；两次检查间隔五分钟，交接前再查一次。不向队友进程发信号。

这是用户要求的服务器监视器，不是共享调度器的原子资源预留。不能保证其他用户不会恰在最后检查之后提交新任务。

## 已接上什么

`toy48-queue/launch.json` 已原子更新，当前 SHA256 为 `db2a9755400d7c440e2e0bc3ebacf0ed2ac915b1be0c7b9a4d95f53d839d42f4`。它在新目录调用 `tools/run_gpu_ready.py`，配置仍是实际 `configs/gqa-h20.setup.json`。旧入口另存 `launch-before-soft-budget.json`；没有重启监视器。

`ready: true` 表示命令已冻结；**要求文件存在是另外一道门**。清单要求 `server-nonmodel-preflight.json`、`server-base-validation.json`、完整模型清单及三片权重、真实构建数据、正常输入、Java、私有 cuBLAS 等。模型回执和清单现已生成，资源等待仍生效。等待器只检查存在，构建／模型加载器继续核对数据和模型身份。

非模型输入已经服务器 CPU 预检：21,000 条构建指令、2,200 张 canonical 图片逐图哈希通过，正常数据的既有全量校验回执匹配；新增路径六项 CPU 测试通过。回执为 `toy48-inputs/server-nonmodel-preflight.json`。

本次修改另有八项服务器 CPU 检查通过：模拟累计超过 48 GPUh 后继续执行、成功超预算不报失败、真实失败继续计费、账本身份与锁、无训练时间截断及实际配置解析。使用 tiny CPU 参数和模拟计时，没有使用 GPU。当前 required_files 增加 `soft-budget-admission.json`；它覆盖本次代码修改，原非模型预检继续作为未改输入与环境的历史依据。

原下载进程 PID `368902` 已结束：15 个文件下载完成，但收尾移动 HF 元数据时因目标父目录缺失而报错。本次仅补本项目目录，离线复用现有文件，低优先级核对固定来源的全部 SHA256；15 文件、14,131,140,467 字节相符，已写 `toy48-inputs/server-base-validation.json`。实际收尾脚本及回执保存于 `gpu-ready-evidence/`。没有重新下载权重、启动 GPU 或停止任何队友实验。

## 放行后的顺序

1. 单卡固定构建，计划参考 5 GPUh；保存完整模型和视觉塔不变的核验结果。
2. 隔离起点四格验收，计划参考 0.75 GPUh；只读正常 calibration 和 dev，不读取 test 结果。
3. 一次真实 G 冒烟，计划参考 0.25 GPUh；响应项未实际参与时明确为 partial，停止后续。

三项均通过同一个 `toy48-ledger` 计费，用户已确认计划参考为两卡合计约 48 GPUh、setup 6 GPUh。**不再按阶段、单作业或总预算自动终止，也不因超预算拒绝下一步；`timeout=None`，toy 的 `training.max_seconds=null`。** 训练按预定步数正常结束，真实失败或 partial 仍停止后续。用户已确认总卡时为约 48 GPUh，允许容差。运行配置的数值与软预算策略本来就符合此口径，本次只更新说明，不改写服务器已冻结配置或账本。更新时尚无已启动作业或旧账本，不需要中断进程或改写历史费用。

**正式六方法的双卡比较尚未启动。** 需要起点验收与实际计时支持后再冻结共同日程；现有双卡 runner 可复用，不能为了凑 ready 给它填未经验收的训练步数。当前入口只完成上卡资格工作。

## 读状态

`toy48-queue/state.json`、`watcher.log` 是等待／启动状态；真正交接才创建 `launched.json` 和 `experiment.log`。GPU 入口另写 `toy48-setup/status.json` 及各阶段原始回执，预算在 `toy48-ledger`。停止等待器并不等于停止已经启动的测量进程组。

本地可审阅 `gpu-ready-evidence/queue-entry-receipt.json` 和 `server-nonmodel-preflight.json`；它们是部署时快照，当前服务器状态以实时文件为准。
