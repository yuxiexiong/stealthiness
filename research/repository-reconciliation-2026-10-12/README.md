# 本地、H20 与 GitHub 核对及归档

日期：2026-10-12（北京时间）。本次只补齐必要的小型代码和证据、整理入口；旧材料与所有原分支保留，不清理历史，不启动实验，不改变服务器。

## 本次收录

| 来源 | 收录范围 | 原始收集量 |
| --- | --- | --- |
| 本地工作区 | 机制、防御、测量 toy 的源码 / 设计 / 安全数值 / 成本与结果；语法研究当前文档；必要的小型报告、工具和两份研究指南 | 342 文件，6,849,655 bytes；逐文件 SHA256 与本地来源一致 |
| H20 | 最新四个 toy 的服务器版本；尚未进已核验分支的小型源码与结果；旧服务器代码差异独立保留为来源快照 | 621 个实际快照，9,040,077 bytes；传输前后 SHA256 全部一致 |
| 已上传 ASR 分支 | 精选源码、设计、结果摘要 / 小表及当前汇总 PNG 进入主分支；完整原始结果和重复图表仍在原分支 | 387 文件，10,096,908 bytes；来源 `c14a830`，详见筛选清单 |

这些数字是各来源的筛选量，不是新的科学样本数、Git 网络传输量或完整备份量。最终 Git 提交文件清单与检查结果见[验证记录](VALIDATION.json)。同名不同版本不静默覆盖：本地原路径保存本地版本，H20 保存到 `server-h20/snapshots/<原绝对路径去掉开头斜线>`。

## 可查清单

- [本地同步说明](LOCAL_SYNC.md)与[逐项库存](LOCAL_INVENTORY.tsv)：覆盖 1,780 个未跟踪、4 个修改文件；56,969 个 ignored 文件按 87 组登记。
- [本地超限文件 / 模型环境分组](LOCAL_LARGE_ARTIFACTS.tsv)与[H20 大产物清单](server-h20/H20_LARGE_ARTIFACTS.tsv)：路径、逻辑字节数、时间、用途；大文件不重新全量哈希。
- [H20 小文件库存](server-h20/H20_SMALL_FILE_INVENTORY.tsv)、[来源到快照映射](server-h20/H20_COPY_MAP.tsv)、[已有 GitHub 同内容位置](server-h20/H20_GITHUB_MATCHES.tsv)。
- [44 个 ASR 运行目录](server-h20/H20_RUN_DIRECTORY_INVENTORY.tsv)、[收集与哈希凭据](server-h20/H20_COLLECTION_MANIFEST.json)、[边界与未跟随的符号链接](server-h20/H20_SYMLINKS_AND_BOUNDARIES.json)。
- [六个 GitHub 分支](BRANCH_INVENTORY.json)与[ASR 分支逐项筛选](GITHUB_SELECTED_MATERIALS.tsv)。
- [上传规则与大小检查](STORAGE_POLICY.md)、[旧版根 README 原文](LEGACY_ROOT_README_2026-09-10.md)。
- [最终上传排除项](UPLOAD_EXCLUSIONS.tsv)：收集时发现的一份 AppleDouble 元数据保留在源端，不作为 Python 源码上传；621 个 H20 收集快照中发布 620 个真实文件。

## 最新服务器证据入口

- [机制 r01 结果](server-h20/snapshots/workspace/cross-model-asr/20261010_asr_mechanism_toy_r01/results/REPORT.zh-CN.md)。
- [防御 r01 快照](server-h20/snapshots/workspace/cross-model-asr/20261010_asr_defense_toy_v2_gpu1_r01/)、[防御 r02 结果](server-h20/snapshots/workspace/cross-model-asr/20261011_asr_defense_toy_v2_gpu1_r02/RESULTS.md)。
- [测量 r01 结果](server-h20/snapshots/workspace/cross-model-asr/20261011_asr_measurement_toy_v1_gpu1_r01/RESULTS.md)。

来源快照保留原文、冻结路径和哈希；文内指向原运行环境的绝对路径或未收录大产物，按库存定位，不表示这些资产已进入 Git。原始回答、逐题 logits、private/witness、权重、训练数据和部署包没有补传。

## 保留的历史分支

| 分支 | 用途 |
| --- | --- |
| [ASR v2](https://github.com/yuxiexiong/stealthiness/tree/codex/asr-v2-staged-gpu1-20261008) | 完整既有跨模型阶段产物与历史图表，主分支仅引入精选小文件 |
| [cross-model 早期分支](https://github.com/yuxiexiong/stealthiness/tree/codex/cross-model-asr-dynamics) | 早期设计、执行与失败记录 |
| [diagnosis](https://github.com/yuxiexiong/stealthiness/tree/diagnosis-runs-2026-09-15) | 诊断路线历史独立提交与结果 |
| [Qwen / grok](https://github.com/yuxiexiong/stealthiness/tree/qwen-plan) | Qwen 与 LLaVA grok 计划及历史材料 |
| [toy48](https://github.com/yuxiexiong/stealthiness/tree/worktree-toy48-full-run) | 原 toy48 执行及负结果记录 |

## 核对边界

逐字节比较覆盖当前六个分支文件树及本次收集的小文件，不是所有历史提交或整个 H20 的全盘备份审计。大产物清单中的 SHA256 未实测则明确为 null / 未核验；硬链接和逻辑文件大小不等于物理占用。完成凭据存在只表示执行记录存在，不证明科学主张成立。服务器单份运行副本不当作已验证的独立备份。
