# 完整归因可视化探针归档

2026-09-10 冻结快照：[下载完整归档](attribution-probe-2026-09-10.tar.gz)，[逐文件清单及 SHA-256](attribution-probe-2026-09-10.manifest.json)。压缩后 47,847,704 字节（约 45.6 MiB），原始文件合计 333,435,770 字节（约 318 MiB）。

归档完整包含原 `runs/attribution-probe/` 的 156 个文件：

- 36 份 M0/M1/M3 原始 JSON，覆盖 108 条轨迹和 8,274 个输出位置。
- `full-html/` 的 108 个逐轨迹页面和 1 个索引，另有 2 个历史/更新预览 HTML。
- 输入与运行清单、完成回执、全位置补算记录、预览 PNG 和原稀疏归因备份。
- `deep-mining-summary.json`，即第二次挖掘的可复核统计。

最新研究文字见[第二次挖掘报告](../PROBE_DEEP_MINING.md)，计算代码见[只读分析脚本](../mine_probe.py)。历史稀疏预览和备份保留在归档中；最终覆盖以 `full_completion.json`、`M0/M1/M3/` 和 `full-html/` 为准。

在新克隆的仓库根目录解压：

```sh
mkdir -p attribution-visualization/runs
tar -xzf attribution-visualization/artifacts/attribution-probe-2026-09-10.tar.gz -C attribution-visualization/runs
```

随后用浏览器打开 `attribution-visualization/runs/attribution-probe/full-html/index.html`。页面自包含，不需要启动模型或服务器。报告中指向运行 JSON 和热图的相对链接也会恢复可用。

归档 SHA-256：

```text
dec3b87ed4683953e629a0426ac7a0d4e92c3a06b0fe5ca84de6321849b3c447
```

打包后已逐项读取归档内全部文件，确认文件名、大小和 SHA-256 与原文件一致；未重跑模型实验。清单保留各成员的校验值。原始目录继续保留，后续运行不会自动改动这份冻结归档。
