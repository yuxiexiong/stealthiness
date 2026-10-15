# GPU0 机制线队列的看门狗

2026-10-15 安装在 gqa-h20 上，不依赖任何 Claude 会话。

- cron 每 5 分钟（以及开机时）运行 `/workspace/claude-jump/watchdog_gpu0.sh`；worker 不在就重新拉起 `/workspace/claude-jump/worker_gpu0.py --root jobq_gpu0 --gpu 0`（共享的 `jobq/worker.py` 的副本，只把等待时间改成 3 分钟；GPU1 队列仍用原文件）。
- **原规则**：GPU0 上没有任何计算进程、显存占用 < 2 GB，连续 3 分钟（用户 2026-10-15 从 10 分钟改为 3 分钟），才开下一个任务。所以永远排在别人（包括队友的 jevrep 任务）后面，不与之同时跑。
- 曾短暂改成「利用率 < 5% 且空余显存 ≥ 80 GB 即可」的 v2 规则；队友说明 GPU0 上的任务是 CPU 密集型、不能干扰后，v2 在开跑任何任务之前就被停掉并撤回。
- 全部停下：`touch /workspace/claude-jump/jobq_gpu0/STOP`。看门狗日志：`jobq_gpu0/watchdog.log`；队列日志：`jobq_gpu0/queue.log`。
- 2026-10-15 23:28：第一个任务（209cmrv_010，22:50:33–22:54:41）与队友 22:50:28 启动的任务重叠约 4 分钟。原因：worker 最后一次空闲检查后还要睡 60 秒才开跑，中间不复查。修法：认领任务前再查一次 GPU，有别人的进程就继续等。该任务的工程门通过（120 步 loss 逐位一致，logits 差 0），结果有效。
