# LLaVA-1.5-7B 补充绘图合并核验

2026-09-28；Git 来源 e2da81db21d14da69d728e3351d8c42fbe29af5e，绘图提交 0f0f71f。

已更新原路径 `attribution-microscope/attribution-atlas.html`。原版本备份在本目录 `original-attribution-atlas.html`。
服务器产物 `/root/p2b_build/attribution-atlas.html` 的下载校验和与服务器一致。

- Phase 2、Phase 2b、fill、fill2 均无失败、待办、运行中任务。4 个补点训练同一性记录通过。
- 原 24 个臂的全部 73,360 条归因条目保留；新增 63 个臂，总计 87 个、121,560 条。
- 每条包括图像侧和文本侧；不要将面板数量理解为独立原图数。
- 63 个补充臂按选点文件核对。固定 60 个轨迹样本、T1/T2/T3/C、clean A 与 trig A/B 齐备；P-0.4 另有完整 200 样本 clean/trig A/B。
- 95 个内嵌 PNG/JPEG 全部解码，PNG 行数匹配索引。
- 曲线共 57 点：seed1 27、seed2 20、dose 8、E3 2。6 张 SVG、2 张数值表已内嵌；228 项表格统计值逐项吻合 JSON（显示精度）。
- 第二种子轨迹的 RETRAIN-A-D 对照行改为自身参照；6 项对照路由检查通过。ASR 保留一位小数。
- 浏览器验证：六张曲线加载成功，种子 1/2 及补点存在，291 步放大图匹配 RETRAIN-A-D@s320，A 两列正常，旧网格 200 个画布，数值表正常，无控制台错误。

边界：这是已有实验产物完整性及页面验证，没有重跑 GPU 实验；轨迹 B 的 clean 列未成像，属于原计划范围。H1–H4 判档没有执行。

复建：先运行 `audit.py`，再运行 `merge.py`；另可运行 `node check-reference.cjs` 检查对照路由。Python 审计使用仓库已有 `.venv-attribution` 环境。生成文件约 103.87 MiB，超过 GitHub 普通单文件限制；本次没有提交或推送。
