# A 的代码实施与冒烟前 ETA

**历史记录说明：** 下文是A启动前的准备与粗估，保留当时措辞。此后A已实际完成，实测总计0.148477 GPUh（含失败尝试），见[A结果](A_RESULTS_2026-09-13.md)和[服务器记录](SERVER_EXECUTION_2026-09-13.md)。当前V3探针的执行入口见[实现说明](VISUAL_PROBE_IMPLEMENTATION.md)。

更新：2026-09-13。**代码与 CPU 检查完成；真实 B0 的 GPU 冒烟和 A 全量均未运行。** 用户最新指令是先完成本地代码、冒烟前给粗估，本轮没有部署或排队服务器作业。

后续补充：按用户要求，[B 的代码、新确认数据及 CPU 检查](STAGE_B_IMPLEMENTATION.md)也已准备好，实际规则待 A 结果后冻结。本文下面“未实现 B”描述的是 A 实施时的范围，不是当前 B 状态。

## 已实现什么，复用了什么

| 文件 | 实现与复用 |
|---|---|
| [prepare_diagnosis_a.py](tools/prepare_diagnosis_a.py) | 复用原事实数据加载、官方 CLEVR 问题程序、合法编辑检查、canonical CLIP 处理、既有标记及对象 mask 覆盖检查；固定全池顺序，不读取旧预测筛成功 |
| [diagnosis_model.py](repair/diagnosis_model.py) | 复用 `repair.model.VLM` 的 HF 加载、图文融合与生成；新增一次精确坐标复制及独立正确前缀评分，无训练或新模型主干 |
| [diagnosis.py](repair/diagnosis.py) | 资格筛查、16 组合实测、包含关系最小组、逐题见证和供体交换；复用现成事实评分、不可覆盖输出目录及归因页面色块 |
| 现有 `repair.parallel` / `experiments.measure` | 后续双卡按案例分配，复用进程管理与总卡时记录；不新增服务器调度器或硬截止 |

范围固定为当前 LLaVA 的 CLIP＋Llama 接口。没有实现 B、权重修复、跨层搜索、Qwen 适配或新的投毒训练。`lora: null` 表示不再挂新训练适配器；模型加载后所有参数冻结。

四组为视觉 token 网格的左上、右上、左下、右下。正常状态直接复制到选中坐标，decode 不再替换；不同问题、视觉网格或非视觉状态不一致时拒绝拼接。捕获操作在官方视觉投影之后停止，不额外跑完整语言模型。

评分与自由生成分别建立自己的缓存。输出保存真实 token、EOS／长度停止原因、全词表间隔、非有限数值原因与实际答案。采用明确的纯贪心配置，避免 checkpoint 的隐含处理器混入原始 logit 解释；正常／异常资格均用同一配置重新测量。

## 目前验证到了哪里

- 3 项真实 tiny HF CPU 检查：精确替换、只修改一次、缓存隔离、原始词表评分、结束条件、对齐拒绝等。
- 3 项 CPU 数据检查：题目真正依赖的对象、颜色与计数的遮挡范围、来源及真值绑定、全池要求、禁止覆盖。
- 3 项 CPU 流程检查：不依赖干预成败的筛选、非单调情况下的完整最小性、纠错／保护见证、16 组合、供体交换、离线 HTML。
- 真实候选池已在本地 CPU 准备：**96 个场景、576 个节点、384 张正常／标记 canonical 图片**。全部通过本次几何可见性检查；这不等于正常模型都答对，更不是归因实验成功。

实际准备输出：`runs/diagnosis-a-prepared-2026-09-13/cases.jsonl` 及同目录 `receipt.json`。图片路径相对清单，可整体复制；清单、图片、问题程序与来源均有哈希。筛查阶段会在固定顺序中取至多 12 个模型资格合格案例，不按局部替换结果挑样本。

本轮没有运行 GPU 作业。开工前的只读服务器检查在 2026-09-13 03:34 北京时间发现两张 H20 都有队友作业；用户随后要求先不上服务器，此后没有再次访问服务器。不能把这份历史快照当作未来启动时的空闲证明。

## 冒烟前的大概 ETA

**建议先按总计约 1–2 GPUh、双 H20 约 30–60 分钟安排 A。长回答较多时，预留总计约 2–3 GPUh、双卡约 1–1.5 小时。** 这不是硬上界，不包括等待队友、服务器排队或文件传输。

GPUh 是卡占用时间之和；两卡持续占用 30 分钟为 1 GPUh。下表为了避免低估，把串行资格阶段及两卡不平衡期间也按两卡持续占用计费。若资格阶段只分配一张卡，实际总卡时会更少，但要以测量账本为准。

| 计划情景 | 双卡墙钟估计 | 若两卡全程占用的总 GPUh |
|---|---:|---:|
| 多数为短事实回答／短拒答 | 约 15–30 分钟 | 约 0.5–1 |
| 常规预留，含串行筛查、完整表、评分、身份检查及交换 | **约 30–60 分钟** | **约 1–2** |
| 局部替换产生较多重复或长回答 | 约 60–90 分钟 | 约 2–3 |

不能把这次下降理解成又删除了修复实验内容：**当前任务已经改成纯诊断 A，不再进行原来的训练、反向传播和每步 20 次人工扰动搜索。** 旧六臂修复不属于本轮；A 的 16 种组合、实际回答、独立评分和必要对照均保留。

### 粗估根据

历史数据来自 `runs/toy48-independent-review-2026-09-12/raw/`，不是本次新计时：

| 同模型／硬件的旧记录 | 已记录耗时 | 如何使用 |
|---|---:|---|
| `toy48-run/full/eval/clean/G/evaluation.json` | 627.84 秒 | 936 次实际生成，同时含 576 个节点的候选批评分，不能当作纯 decode 单价 |
| `toy48-run/full/eval/triggered/G/evaluation.json` | 913.81 秒 | 同样混合生成与候选评分，给长输出条件的量级参考 |
| `toy48-ledger/evaluation-lanes/measurement/run.json` | 4916.84 秒，2.73158 GPUh | 原双卡整阶段实际记录；含多方法、两种条件与加载，不能整个套给 A |

旧 triggered B0 的 936 个输出中，874 个是精确短拒答，但也有长重复输出。A 的局部状态替换可能改变输出长度，因此短拒答速度不能当完工保证。候选批评分与本次固定参照轨迹评分也不同；粗估仅使用其量级，并为差异、筛查和负载不均保留余量。

### 代码实际工作量上限

按 12 个案例、每例最多 6 节点计：

- 资格：核心最多 `96 × 2 × 2 = 384` 次生成；仅对入选案例查伴随题，最多再加 96 次，合计至多 480 次。正常答错时不再生成对应异常答案，实际通常更少。
- 干预表：`12 × 6 × 16 = 1152` 次自由生成。空集重新生成，与资格记录核对；没有免费复用后再漏计成本。
- 独立评分：至多 1152 条参照评分轨迹，每条按正常实际答案 token 数推进；**不是恰好 1152 次 Transformer forward**。单独记录评分 token 和 decode 次数。
- 状态捕获：每节点正常一次、异常两次，共至多 216 次视觉＋投影；第二次异常用于测量重放差异。
- 自身复制：至多 72 次额外生成；全集恢复已包含在 16 表内。
- 供体交换：对最多半数坐标、仍留下高于重放误差的状态差异的最小成功组执行。四组最多有 6 个这样的最小组，故宽松上限 432 次额外生成；没有对应合格供体则明确未测。

上述宽松上限合计至多 2136 次自由生成，另有独立评分和较便宜的捕获；并不假定所有上限同时发生。输出都接近 256 token 上限时，耗时可能超过表中长尾情景。代码不设置按总 GPUh 到点终止。

## 后续运行入口（本轮没有执行这些 GPU 命令）

工作目录为 `attribution-visualization/visual-evidence-repair`。`configs/diagnosis-a.example.json` 中的 B0 路径来自历史回执，启动前仍须只读确认模型及资产清单；它不指向新构建模型。

CPU 准备已运行，原命令如下；输出目录存在时会拒绝覆盖：

```bash
python tools/prepare_diagnosis_a.py --facts runs/toy48-inputs/facts/test-facts.jsonl --config configs/diagnosis-a.preparation.json --construction-manifest configs/construction-lock.json --output runs/diagnosis-a-prepared-2026-09-13
```

具备空闲资源后先筛资格，再用前两个入选案例作小规模 GPU 冒烟：

```bash
python -m repair.diagnosis screen --cases runs/diagnosis-a-prepared-2026-09-13/cases.jsonl --config configs/diagnosis-a.example.json --output runs/diagnosis-a-screen --limit 12 --device cuda:0
python -m repair.diagnosis run --cases runs/diagnosis-a-prepared-2026-09-13/cases.jsonl --config configs/diagnosis-a.example.json --screen runs/diagnosis-a-screen/screen.json --output runs/diagnosis-a-smoke --limit 2 --device cuda:0
```

筛查与冒烟没有实验效果门槛：正常参照不足照实记录；冒烟必须通过空集重放、自身复制及全集恢复的实现检查。冒烟可以出现“没有非平凡局部组合”，这本身不代表接线失败。

冒烟通过后再依用户要求更新 ETA，并安排全量。双卡可复用 `repair.parallel`，两条命令分别声明 `--lanes 2 --lane-index 0` 和 `--lanes 2 --lane-index 1`，按已筛案例交错分配，各自在自己的可见 GPU 上用 `--device cuda:0`。外层继续复用 `experiments.measure` 记卡时，不加训练框架。

冒烟与全量使用不同输出目录，当前没有自动把冒烟记录并入全量的续跑机制；若全部重跑 12 例，应额外计入这 2 例冒烟成本，约增加该部分六分之一。也可在未来明确冻结余下清单再合并，但本版不隐式跳过或拼接不同运行。

每个运行目录有 `records.jsonl`、`run.json`、`index.html`；失败保留 `failure.json` 和已完成记录，不自动重试。报告可以离线合并两卡记录：

```bash
python -m repair.diagnosis report --records runs/diagnosis-a-lane0/records.jsonl runs/diagnosis-a-lane1/records.jsonl --output runs/diagnosis-a-index.html
```

本轮完成的是可运行测量代码和 CPU 验证。真实模型数值稳定性、局部恢复是否存在及方法价值，仍由后续 GPU 数据回答。
