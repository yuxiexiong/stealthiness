# 服务器排队与 GPU 准备入口

2026-09-10。**最近核验时真实 GPU 阶段尚未开始。** 入口已安装；队友两条总控仍存活。公开模型全部文件现已通过服务器来源哈希校验，满足模型输入门槛；不能把准备完成说成已通过 GPU 验收。

## 当前实际部署

- 主机 `gqa-h20`，独立根目录 `/root/attribution-visualization-20260910`。
- 已推送代码 `0f7702c`；服务器运行目录 `toy48-code-ready2/attribution-visualization/visual-evidence-repair`。测试过的 Python 文件与本次提交一致。
- 已有监视器 PID `258936`，目录 `toy48-queue`，每 300 秒检查一次，`nice=19`，脱离 SSH。没有新增 Codex automation；此前本地 automation 保持暂停。
- 等待队友总控 PID `4158784` / start_ticks `727135121`、PID `4158937` / start_ticks `727135621`。等待整个总控退出，不把两轮训练之间的短暂空闲当成实验结束。
- 队友总控结束、全部要求的输入存在后，检查两卡无计算进程、利用率 ≤5%、显存 ≤256 MiB；两次检查间隔五分钟，交接前再查一次。不向队友进程发信号。

这是用户要求的服务器监视器，不是共享调度器的原子资源预留。不能保证其他用户不会恰在最后检查之后提交新任务。

## 已接上什么

`toy48-queue/launch.json` 已原子写入，SHA256 为 `62d5b2a825adc20e3fea83e2ce41035a7acce4acd52b3408a4b4866403cf3915`。它调用 `tools/run_gpu_ready.py`，配置是 `configs/gqa-h20.setup.json`，不是带占位路径的 example。

`ready: true` 表示命令已冻结；**要求文件存在是另外一道门**。清单要求 `server-nonmodel-preflight.json`、`server-base-validation.json`、完整模型清单及三片权重、真实构建数据、正常输入、Java、私有 cuBLAS 等。模型回执和清单现已生成，资源等待仍生效。等待器只检查存在，构建／模型加载器继续核对数据和模型身份。

非模型输入已经服务器 CPU 预检：21,000 条构建指令、2,200 张 canonical 图片逐图哈希通过，正常数据的既有全量校验回执匹配；新增路径六项 CPU 测试通过。回执为 `toy48-inputs/server-nonmodel-preflight.json`。

原下载进程 PID `368902` 已结束：15 个文件下载完成，但收尾移动 HF 元数据时因目标父目录缺失而报错。本次仅补本项目目录，离线复用现有文件，低优先级核对固定来源的全部 SHA256；15 文件、14,131,140,467 字节相符，已写 `toy48-inputs/server-base-validation.json`。实际收尾脚本及回执保存于 `gpu-ready-evidence/`。没有重新下载权重、启动 GPU 或停止任何队友实验。

## 放行后的顺序

1. 单卡固定构建，最多 5 GPUh；保存完整模型和视觉塔不变的核验结果。
2. 隔离起点四格验收，最多 0.75 GPUh；只读正常 calibration 和 dev，不读取 test 结果。
3. 一次真实 G 冒烟，最多 0.25 GPUh；响应项未实际参与时明确为 partial，停止后续。

三项均通过同一个 `toy48-ledger` 的 setup 预算执行，仍受 **总计 48 GPUh、setup 6 GPUh** 约束。失败、超时和清理记入实际成本，任何非零退出停止。不因一个 CPU 测试或一次更新成功就自动标 B0 合格。

**正式六方法的双卡比较尚未启动。** 需要起点验收与实际计时支持后再冻结共同日程；现有双卡 runner 可复用，不能为了凑 ready 给它填未经验收的训练步数。当前入口只完成上卡资格工作。

## 读状态

`toy48-queue/state.json`、`watcher.log` 是等待／启动状态；真正交接才创建 `launched.json` 和 `experiment.log`。GPU 入口另写 `toy48-setup/status.json` 及各阶段原始回执，预算在 `toy48-ledger`。停止等待器并不等于停止已经启动的测量进程组。

本地可审阅 `gpu-ready-evidence/queue-entry-receipt.json` 和 `server-nonmodel-preflight.json`；它们是部署时快照，当前服务器状态以实时文件为准。
