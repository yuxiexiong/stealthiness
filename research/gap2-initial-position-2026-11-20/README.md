# Gap 2 实验与验证状态

截至 2026-11-20，三个新种子的统计设计已定。从原始 JSON 读数到逐题表和总判定的离线分析已在 CPU 上验收。尚无新种子的终点结果。检查点已做文件核对，恢复等价性仍待验证，GPU 任务尚未入队。

| 工作 | 状态与材料 |
|---|---|
| M058 既有实验及误差分析 | [逐题报告](REPORT.zh-CN.md)、[误差核算](ERROR-AUDIT.zh-CN.md)、[统一平移核查](FLAT-SHIFT.zh-CN.md)、[四状态漂移对照](DRIFT-FACTORIAL.zh-CN.md) |
| 新三种子实验设计 | [决策草案与 Advisor 复核](../research-mentor/reviews/two-gaps-2026-11-20/GAP2-FAST-DECISION.md)记录初版；[计数判读规格](COUNT-EVALUATION.zh-CN.md)采用最终的 1,000 个固定划分 |
| CPU 分析 | [统一判读脚本](count_verdict.py)先核对[文件哈希](OFFLINE.sha256)，再从原始 JSON 核对题号顺序、答案、分数及逐臂总数，输出每条轨迹的逐题表和一份总判定。 |
| M058 离线验收 | [复算结果](M058-OFFLINE-VERDICT.json)记录固定 1,000 次划分上三种方法的差距 MAE。主模型为 1.823，校准半面板外推为 2.623，无漂移对照为 7.191。单位为每臂、每 100 道评价题。 |
| 服务器资产校验 | [核验记录](RESTORE-CHECK.zh-CN.md)。三种子各两份检查点，共 126 项文件通过大小和 SHA-256 核对。 |
| GPU 就绪状态 | 第 300 步恢复后的第 310 步状态与历史参照的运行等价性尚未验证；三轨迹从排队到结果表的端到端流程尚未验收。GPU 空闲本身不足以判定可启动。 |

在本目录使用 Python 3 和 NumPy 运行以下两条命令。第一条会新建结果目录，第二条会分别核对 M058 的 3,600 行配对记录以及缺件、错误输入和判定边界。

```sh
python3 -B count_verdict.py --archived-zip Gap2-M058.zip --out replay-m058
python3 -B -m unittest -v test_count_verdict
```

三个新种子都有有效、独立取得的读数后，同一入口可直接读取，输入布局见[计数判读规格](COUNT-EVALUATION.zh-CN.md)。每个训练种子分别评价，同一批题的反复划分不能计作多个独立训练重复。
