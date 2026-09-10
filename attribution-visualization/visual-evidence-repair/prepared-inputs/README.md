# 已冻结输入的 Git 存档

这是本轮实际 CPU 准备产物的逐字节副本，不是模型运行结果。`inventory.json` 记录 36 个 JSON / JSONL 文件的路径、大小与 SHA256，共 52,004,348 字节。

- `toy48-inputs/`：dev、fit、calibration、test-clean 的实际划分；事实场景、题目程序、真值、图片哈希；COCO / VQAv2 原标注来源记录。
- `toy48-baseline-data/`：20k 正常指令、1k 构建候选及所选原始记录和来源回执。
- `toy48-baseline-construction/`：实际 21k 条混合构建指令及 canonical 图片清单。它们不是已经训练完成的污染权重。

文件保持原字节和路径，不因归档重新抽样或生成。将三个目录复制回本项目 `runs/` 可恢复元数据布局，但还必须取得清单引用的图片后才能运行。合成数据在 `toy48-data/` 的原始副本与 `toy48-inputs/facts/` 重复，仅存后者。

14.13 GB 的公开模型、大型原图/实例 mask、下载缓存和第三方源码 checkout 不重复放入普通 Git。精确模型 revision / 哈希见 `../configs/base-source-lock.json`；事实数据见 `../EDITCLEVR_DATA_LOCK.json` 与 `../prepare_toy48_data.py`；正常图和构建数据见 `../tools/prepare_coco.py`、`../tools/prepare_baseline_data.py` 及来源回执。原始图像仍保留在本地和服务器独立目录。此存档没有声称“克隆 Git 后无需下载即可运行”。

EditCLEVR 派生场景来自 torux/EditCLEVR 的 revision `734efa9b0164ba742cd2c39b9c2945c494497c23`（CC BY 4.0）；本项目的修改是选择子集、添加问题程序及划分。COCO / VQAv2 标注保留各自原来源和使用条件，原图片权利归原作者；归档不改变这些条件。详情见相邻数据审计文档。

测试真值随研究归档保存供隔离评估器使用；U 修复流程仍只读取计划允许的正常 dev / fit / calibration，不能因文件已在 Git 中就用 test 成绩选规则。
