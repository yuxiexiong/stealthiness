# A 的服务器执行记录

服务器：`gqa-h20`。本次目录：`/root/attribution-diagnosis-a-20260913/`。仅运行 A；B 尚未冻结实际判断规则。

## 排队与运行约束

- 复用 `repair/server_queue.py`，绑定队友控制器 PID 2341198、2341316 及各自启动时间；每 60 秒检查一次。
- 两条控制器退出后，两张指定 H20 必须连续两次没有计算进程、利用率不高于 5%、显存不高于 256 MiB。启动瞬间再检查一次。
- 队友进程与目录没有修改、终止或抢占。队友最后一条队列结束后，首次空闲确认在北京时间 05:04，A 首次启动在 05:05。
- 复用 `tools/run_diagnosis_a.py`：资格筛查最多 12 例 → 前 2 例双卡冒烟 → 技术检查通过后双卡全量 → 汇总 HTML。不因无局部成功集合停止，不自动重试、不按 GPUh 到点停止。
- 只读状态连接每 30 秒读取队列和实验状态，只有阶段变化或失败才检查进一步日志。连接断开不终止独立的服务器队列。

## 已处理的启动错误

首次资格筛查在生成时收到 SIGFPE；启用 faulthandler 复现，栈位于模型输出线性层。没有得到可用资格或干预结果。

发现此前已经记录的部署前提未继承：服务器 torch 默认解析到共享旧 cuBLAS，而旧实验明确预加载项目私有 cuBLAS。此次恢复的仍是旧实验环境，没有升级共享依赖或修改模型、数据、归因算法、实验判据。

```bash
export LD_PRELOAD=/root/attribution-visualization-20260910/.venv/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.12:/root/attribution-visualization-20260910/.venv/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.12
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

依据：`GPU_READY.md`、`tools/chain_after_setup.py:environment()` 和 `gpu-ready-evidence/runtime-new-tests.json`。修复后的进程已核对 `/proc/self/maps`，确实加载上述私有库；真实执行验收以随后 A 记录为准。

失败与复现均保留：首次两卡启动约 0.004170 GPUh，单卡复现约 0.002027 GPUh，不能从最终账本中删去。

第二次启动恢复了生成能力，但部署包漏带动态加载的原始 `VQAEval` 文件。补齐未修改的原始评分文件、许可证和 `view_probe.py`，并在启动器加入 CPU 依赖预检；没有替换评分器或放宽条件。第二次失败消耗 0.004604 GPUh。

## 完成目录与回执

最终使用全新目录 `attempt-3/`，再次经过原队列的空闲检查。北京时间 2026-09-13 05:14:46 启动，05:18:53 完成；队列和实验均为 `completed`，退出码 0。服务器当前尝试由 `active-attempt.json` 标明；本地回执为 `runs/diagnosis-a-server-2026-09-13/deployment.json`。

- `attempt-3/queue/state.json`：排队与总进程状态。
- `attempt-3/run/state.json`：筛查、冒烟、全量、报告阶段。
- `attempt-3/measurement/run.json`：包括串行阶段、双卡尾部空闲在内的实际总卡时。
- `attempt-3/run/index.html`：全量可视化入口；图片内嵌，可离线打开。

12 个正式案例包含 49 个合格问题节点，完整测量 784 个子集干预；所有生成自然遇到结束符，没有长度截断。空替换重放、自身复制、全视觉恢复三项技术对照均通过，参数更新数为 0。两卡各完成 6 例，配置、输入与科学实现标识一致。

成功轮（筛查、双卡冒烟、双卡全量、报告）247.816 秒，消耗 **0.137676 GPUh**。加上以上两次失败及一次单卡复现，合计 **0.148477 GPUh**。排队等待不占 GPU，不计入 GPUh。此前 1–2 GPUh 的粗估偏保守；实测正式生成仅 2–5 个 token（包括结束符），没有通过删减 16 子集、节点、对照或改动判据压缩本轮实验。

本地结果已取回到 `runs/diagnosis-a-server-2026-09-13/results/`，保留全部失败记录。结果解释见 [A_RESULTS_2026-09-13.md](A_RESULTS_2026-09-13.md)。A 完成后监控连接和队列自动退出；没有启动 B，也没有创建本地定时任务。

本记录不改变冻结的 `DIAGNOSIS_FEASIBILITY_EXPERIMENT_PLAN.md` 或 `THEORY_DIAGNOSIS_EXTENSION.md`。后续如再发生错误，只修实现或恢复已声明运行环境，保留失败材料；不删除验收检查或改动科学问题来放行。
