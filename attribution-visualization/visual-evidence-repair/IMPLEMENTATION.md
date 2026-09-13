# 48 GPUh toy：实现、复用与执行入口

**状态说明（2026-09-13更新）：** 本文记录 N1–N10 的 toy48 实现与历史执行准备，实际完成结果见[实验复核](reviews/toy48-2026-09-12/CONSENSUS.md)。后续诊断[A](A_RESULTS_2026-09-13.md)、[B](B_RESULTS_2026-09-13.md)均已完成真实B0实验。当前V3可视化探针的代码和本地验证见[实现与运行说明](VISUAL_PROBE_IMPLEMENTATION.md)，尚未运行新GPU测量。以下历史“待验收”及运行命令不作为新诊断的状态或启动指令。

日期：2026-09-10。当前执行 [toy48 计划](TOY_PLAN.md)：单模型、单污染实例、单 seed、六方法各一个配置；使用 Ponytail full。代码验证与真实污染实验分别记账，不把小模型软件测试称为科学结果。

预算策略已改为 `soft_no_automatic_stop`：两卡合计约 48 GPUh 为用户确认的计划参考，阶段和单作业数字均不自动截断作业或阻止后续任务。用户已确认是总卡时 48 GPUh；两卡同时占满时理想墙钟约 24 小时。拒答解除与事实修复的区分见[诊断与场景建议](REFUSAL_VS_FACTUAL_REPAIR.md)，尚未更换当前场景。

最新上卡准备见 [GPU_READY.md](GPU_READY.md)：已取得实际数据和构建材料，选定公开 LLaVA 基座，服务器新增路径的六项 CPU 检查通过。以下资源表保留编码前状态，不代表后续下载／部署仍未完成；真实污染起点与 GPU 资格仍待排队验收。

## 编码前资源复核（历史快照）

| 资源 | 当前处置 | 本次核实 |
|---|---|---|
| PyTorch / Transformers / PEFT | 直接复用自动微分、AdamW、VLM 和 LoRA | 本地 .venv-attribution：2.7.1 / 4.53.3 / 0.16.0 |
| 官方 LLaVA / Qwen2.5-VL 实现 | 适配融合输入注入、候选评分和真实生成 | 核对安装版本源码与对应官方文档；不另写主干 |
| 现有 oa.write_json | 直接复用原子 JSON 记录 | 本仓库 experiments/oa.py |
| 现有 measure.run | 直接用于外围 CPU/GPU 成本监测 | 本仓库 experiments/measure.py；不另做守护系统 |
| 现有离线归因页面 | 复用有符号色标／缺失状态设计 | 页面改接事实对与实际修复记录，OA 专用加载不迁移 |
| CLEVR 官方场景与渲染 | 复用场景、材质、真值；本次提供已生成配对的输入接口 | Blender 环境和真实渲染尚未验证；未写未经验证的渲染 wrapper |
| VQA / COCO 评分 | 复用官方规则和现成评分包 | 缺少依赖或原协议字段时明确未评分，不补零 |
| BackdoorVLM / BadVision | 保留攻击资产和评分来源 | 不把不同污染组件、原生／HF 格式静默互换 |
| RACER | 复用论文流程，代码按原文实现 | 当前未取得作者可运行交付；本实现标记论文流程复现，非原表复现 |
| CleanSight | 保留独立外部基线入口与结果 | 不冒充已内置或已经完成该论文复现 |
| 旧 C5/C6 输出头 / CVXPY 方案 | 归档 | 不在新训练路径引入这些依赖 |
| 原污染 checkpoint / trigger | 待实际交付及验收 | 不用干净模型或随机小模型替代真实科学起点 |

前次[资源核查](TOY_RESOURCE_AUDIT.md)提供来源细节。本次针对会实际使用的接口复核，没有重复全量查重，没有下载大权重或启动污染训练。

## 实现约定

当前理论入口为 THEORY.md，历史理论有 ARCHIVED 标记；新实验以 N 编号接口为标尺。U 训练只读取 fit/calibration 数据。真实触发仅进入独立评价，以及持同实验 U 冻结回执的 K 训练／诊断。

采用 Python 标准库 CLI、JSON/JSONL 和现成依赖；一个模型加载器、一个修复核心、一个数据入口和一个离线报告，不新建训练平台。运行目录禁止覆盖，训练失败保留状态，预算超过计划仍记录实际费用。最终保存原模型允许参数增量，实际生成读取该增量。

## 已实现什么

| 文件 | 实际工作 |
|---|---|
| [repair/model.py](repair/model.py) | 官方 HF LLaVA（CLIP＋Llama）和 Qwen2.5-VL 加载；原 projector／merger 与语言 LoRA；候选含 EOS 评分、实际生成及增量保存 |
| [repair/core.py](repair/core.py) | N1–N8：合法共享 PGD、合格参照、静态权重、独立分母、AdamW；N9：真实更新差与全部竞争者的间隔、残差 |
| [repair/data.py](repair/data.py) | 图像字节及场景划分检查；U、K、test 用途分离；接收已有合法事实对，不伪造反事实图 |
| [repair/assets.py](repair/assets.py) | 模型／processor／已有污染 adapter 的一次内容哈希清单；运行时核对文件集合、大小及修改时间 |
| [repair/report.py](repair/report.py) | 复用官方 VQA 规则、可选 COCO CIDEr、原 ASR 结果精确连接、离线图、成对簇区间和精度规划 |
| [repair/__main__.py](repair/__main__.py) | 观察、一次共享参照 prepare、修复、单配置正常校准选择、独立评价及起点缓存、离线图 |
| [repair/budget.py](repair/budget.py) | 包装现有 measure.run；一个串行账本记录阶段与累计 GPUh，预算仅作参考，不设置预算 timeout |
| [repair/parallel.py](repair/parallel.py) | 同一预算父作业内运行两条固定单卡队列；各自隔离 CUDA_VISIBLE_DEVICES，任一失败停止后续任务 |
| [repair/server_queue.py](repair/server_queue.py) | 服务器本地等待队友总控结束及双卡持续空闲；真实输入和启动清单缺失时继续等待 |

内置训练方法是 G、G0、Gl、P、R+、SFT、G-shuffle，以及 RACER-data／RACER-native 的论文重建。G0/P 没有额外二分之一；RACER-data 按端点独立搜索，R+ 才配对共享；普通对照不计算用不到的训练归因参照。Native 要求 100 个 singleton，calibration=null，直接使用冻结配置，不能进入增强 U 选择。

CleanSight 的完整检测／净化实现、BackdoorVLM 原攻击构建器／评估器仍使用外部官方交付。本代码接收其结果，**未将它们冒充内置完成**。编码器投毒的 BadVision 不自动等同本轮下游污染资产；原作者 llava_llama 格式也不能被 HF 加载器静默替换。

本轮只执行 SFT、R+、G0、Gl、G、RACER-data；其他已实现方法保留供未来使用。主要方法代码可运行不等于真实资产已就绪。完整污染权重、匹配触发、精确基座、合法场景编辑、原协议样本及评估器还需按原计划验收一次；当前不自动构建新攻击，不用微型随机模型充当科学结果。

## 环境与入口

复用现有 Python 3.12 环境。此次已安装与 torch 2.7.1 对应的 torchvision 0.22.1，解决 Qwen 官方 processor 即使只用图像仍需其视频处理依赖的问题。完整依赖列于 [requirements.txt](requirements.txt)。

    cd /Users/ruizhixu/Documents/ChatGPT/stealthiness
    .venv-attribution/bin/python -m pip install -r attribution-visualization/visual-evidence-repair/requirements.txt
    cd attribution-visualization/visual-evidence-repair
    ../../.venv-attribution/bin/python -m repair --help

CIDEr 复用 [pycocoevalcap 1.2](https://pypi.org/project/pycocoevalcap/1.2/) 的 PTB tokenizer，需要 Java。当前本机没有 Java，完整 caption 校准未运行；缺依赖、缺参考或评分失败会返回缺测，选择命令不会将它当作通过。本地不为软件检查额外安装 JDK。

此处 CIDEr 使用固定评价语料的 document frequency，保留原单位；不是 BadVision 固定 coco-val IDF 的同一实现。原协议主表必须另接对应原评估器，不能混表。正式环境应保存评分器版本及参考语料身份。

### 先准备资产清单

所有模型组件先置于独立的本地目录。已有污染 PEFT adapter 可在 model.adapter_path 声明，先合并污染增量，再建立各方法相同的修复 LoRA；没有该字段则读取完整污染模型。processor_id 可单独声明。

    ../../.venv-attribution/bin/python -m repair assets /data/local-hf-vlm /data/local-processor --output /data/manifests/model-assets.json

输出清单必须放在模型目录外。每个文件在建清单时哈希一次；后续运行依赖“本地资产存储可信且不可变”的约定，检查完整文件集合、大小、mtime，再绑定清单哈希。不是每次重扫几十 GB 权重，也不是针对能伪造文件时间的攻击者的完整性防御。发现变化就拒绝；重新验收后另建清单，不原地覆盖旧证据。

[LLaVA 配置](configs/llava.example.json)与[Qwen 配置](configs/qwen.example.json)需要填入实际路径。两份文件是接口示例：步数、epsilon、损失系数与 256-token 生成上限尚未经过原协议核准或首跑计时，不能直接当作已冻结科学配置。`training.max_seconds=null` 明确关闭训练时间截止。deep_start 是首个相邻层对的状态索引；示例分别取 22 和 14，仍须核对实际骨干深度。不要只因写在 JSON 里就称其为原作者参数。

## 数据格式

每行一个 unit，问题多的场景以共同 cluster_id 归组。可用 split 为 dev、fit、calibration、test。一个 pair 两端问题和有序候选相同，图像字节不同；答案可以随合法编辑改变。task 为 fact、vqa 或 caption。VQA soft score 需要恰好十条人工 references；caption 至少有其独立参考描述。候选与正确答案仅参与训练和隔离评分，模型自由生成接口不读取它们。

配对行的结构示例（不代表已有图片或已核验事实）：

    {"id":"scene1-color","cluster_id":"scene1","split":"fit","kind":"pair","question_type":"color","nodes":[{"image":"red.png","question":"What color is the sphere?","answers":["red","blue"],"answer":"red","task":"fact"},{"image":"blue.png","question":"What color is the sphere?","answers":["red","blue"],"answer":"blue","task":"fact"}],"intervention":{"verified":true,"changed_fact":{"object_id":"sphere1","attribute":"color","before":"red","after":"blue"}}}

只有在官方场景／问题程序核验事实及连带标签之后才能填写 verified=true。这个字段是来源声明，软件检查不能证明图里真的只改了颜色。单张真实图文使用 kind=single、一个 node、不带 intervention；只有一个候选时可参加 CE，不能成为正常参照。

fit.jsonl 旁必须放 fit.manifest.json，含它实际使用的全部图片、没有额外未用图片：

    {"schema_version":1,"purpose":"repair","image_condition":"clean","provenance":"具体官方场景/数据版本、编辑和标签核验记录","images":[{"path":"red.png","sha256":"实际文件SHA256"},{"path":"blue.png","sha256":"实际文件SHA256"}]}

独立 test 的 purpose=evaluation。U 只接受 clean，且禁止额外攻击字段；K 使用 purpose=known_trigger_diagnostic，每个 unit 的 clean_nodes 对应 U 的原节点，nodes 为已知触发观测。K 会核对相同 U 单元、原图字节、问题、真值和候选；不会自行生成触发。

    ../../.venv-attribution/bin/python -m repair validate /data/fit.jsonl /data/calibration.jsonl

先使用 [CLEVR 官方生成代码](https://github.com/facebookresearch/clevr-dataset-gen)准备并核验场景对，再导出这个通用 JSONL。没有用修改 GQA 标注代替实际编辑图。

## 所有 GPU 步骤共用账本，48 GPUh 暂为计划参考

[stages.json](configs/stages.json) 是本轮评估协调清单：setup 6、reference 4、repair 20、evaluation 10、reserve 8 GPUh。它不含已冻结的真实资产或实测日程，也不自动生成数据。统计单位是独立图片／场景，不是 JSONL 行数；数据准备方按场景分层导出冻结子集，CLI 不在测试时静默抽样。

    ../../.venv-attribution/bin/python -m repair.budget --ledger /runs/toy48-budget --status

本包装器复用 `experiments.measure.run`。一个账本同一时刻只运行一个作业，支持给单作业分配多卡；按实际分配卡数 × 占用时间累计，不按利用率折扣。运行时必须指定阶段、唯一作业名和本次计划 GPUh；阶段及总目标用于报告差额，不限制作业启动或运行。所有 GPU 构建、基线计时、观察、修复、评分与失败都应经同一账本，外部绕过包装器的进程不受它控制。

双卡执行使用一个预算父作业包裹 `repair.parallel`，不是同时打开两个账本作业。两条固定队列各占一张卡，内部模型命令都使用 `cuda:0`；总费用为完整父作业墙钟 × 2，包括某张卡提前结束的空闲尾段和进程清理。共享参照先完成，六方法回执全部冻结后才启动评价队列。当前服务器排队状态、未就绪资产及启动清单约定见 [SERVER_QUEUE.md](SERVER_QUEUE.md)。

    ../../.venv-attribution/bin/python -m repair.budget --ledger /runs/toy48-budget --phase reference --name dev --gpus 0 --max-gpu-hours 1 --cwd . -- ../../.venv-attribution/bin/python -m repair inspect --config configs/llava.local.json --data /data/dev.jsonl --device cuda:0 --output /runs/dev-observation

示例中的 `--max-gpu-hours` 仅记录每作业计划，不是速度预测或限制。包装器显式传入 `timeout=None`；即使作业、阶段或总费用超过计划，仍继续执行并记录差额，已完成作业不因超支被判失败。真实失败仍返回失败；未结算或 running 记录阻止后续运行，不能删除账本再假装未花钱。先核实原进程并据真实费用人工处理，不能自动重试。reserve 只有写明用途后才可使用，不是额外调参额度。

正常权限 U 配置示例增加 `"protocol": "toy48"`。旧配置没有该字段时保留历史能力，**不属于当前 48 GPUh 流程**。`calibration_search` 和 selection.json 的 proxy_metric 为历史兼容字段，toy48 单候选没有 proxy 搜索或排序。正常校准门槛仍是有限样本上的操作规则，不是总体保护保证。

## 阶段 A：观察、冻结、共享参照、六条修复

先验收一个真实污染起点及原评估器，完成一次合格基线计时，再冻结共同训练日程并更新六个方法连同全部评价的耗时预期。预计超出计划时报告差额，结合实际进展讨论后续投入；不能因达到预算数字自动停机。不能直接采用示例 100 步作科学训练日程；它尚未经过曝光／收敛和实际速度核准。

用至多 24 个独立 dev 场景做 inspect。最多一次规则修订，记录观察与操作的关系；看过真实触发测试成绩后不再修改。fit 的全部有效边仍用于计算归因资格及权重，不只计算展示案例。

先做一次共同 θ0 准备，再供全部方法复用：

    ../../.venv-attribution/bin/python -m repair.budget --ledger /runs/toy48-budget --phase reference --name shared --gpus 0 --max-gpu-hours 3 --cwd . -- ../../.venv-attribution/bin/python -m repair prepare --config configs/llava.local.json --device cuda:0 --output /runs/shared-references

prepare 保存完整 fit 的正常参照、实际生成、资格、人工参考 δ、偏离和困难度，以及正常 calibration 的修复前输出。缓存绑定准确模型与资产清单、完整数据身份、初始 seed、候选／生成与参考搜索配置、实现文件身份；不匹配拒绝。只缓存 θ0 测量，不复用更新后模型的 embedding、梯度或训练 δ。来源费用在账本中只发生一次，读取缓存的运行会记录来源，不能说归因成本为零。

按 SFT、R+、G0、Gl、G、RACER-data 分别导出一个冻结配置，共同参照配置必须相同；局部机制对照仅改变 method，RACER-data 保留论文方法的独立端点搜索与损失。示例执行一条：

    ../../.venv-attribution/bin/python -m repair.budget --ledger /runs/toy48-budget --phase repair --name G --gpus 0 --max-gpu-hours 3.3 --cwd . -- ../../.venv-attribution/bin/python -m repair train --config configs/G.local.json --reference-cache /runs/shared-references --device cuda:0 --output /runs/G

toy48 要求 `training.max_seconds=null`，取消训练单元之间的时间截止；外部账本也不设预算 timeout。训练按冻结的有限步数正常完成，公平性由共同曝光日程控制，不改成无限训练。真实失败或人工中止造成步数未完成时，仍不能进入可接受候选。

单候选只通过正常 VQA／CIDEr 门槛，不计算无选优用途的整套 proxy 校准：

    ../../.venv-attribution/bin/python -m repair select /runs/G --limits configs/selection.json --output /runs/G-selection.json

toy48 拒绝向 select 传两个配置。缺正常指标拒绝选择；校准不合格或零更新则写 no_acceptable_update；日程未完成则写 inconclusive_training_incomplete 并阻止其作为合格对照进入评价。B0 回退不叫有效修复；该小校准集不通过或预算不足也不等于整个课题被证伪。

train 保留这些产物：

- update.pt、run.json、training.jsonl：真实参数增量、共同身份、曝光、更新范数及成本。
- reference-cache.pt、reference-eligibility.jsonl：完整 fit 参照、资格、权重及来源。
- attribution.jsonl/html：G/Gl 按文件顺序预定最多 24 个 pair 的实际前后输出与带符号响应；权重仍由完整 fit 计算，未因展示而删训练边。
- calibration.jsonl/html：正常输入的实际前后生成；proxy_status 明确标成 not_required_single_candidate，没有伪造零值或已测状态。

所有方法的回执冻结后才进入阶段 B；这是执行方在调用 evaluate 前检查的共同门槛，单次 CLI 只验证本方法回执，不代替六方法调度。异常或人工中断后没有完整 run.json 的候选不进入选择，不能只拿遗留 update.pt 当已验收模型。

## 阶段 B：共享测试起点，真实生成每个修复结果

同一图片集包含正常和真实触发版本。先冻结 expected_keys：一格、一个准确污染 seed、所有预定 unit/node；不取方法间交集隐藏遗漏。默认 300 VQA、60 caption 和三类各 32 个合成场景。公开原生成停止设置与原攻击评分保持，明确这是子集研究，不冒称原表全量复现。

    ../../.venv-attribution/bin/python -m repair.budget --ledger /runs/toy48-budget --phase evaluation --name G-triggered --gpus 0 --max-gpu-hours 1 --cwd . -- ../../.venv-attribution/bin/python -m repair evaluate --selection /runs/G-selection.json --data /isolated/test-triggered.jsonl --cell frozen-cell-01 --seed 101 --device cuda:0 --output /results/G-triggered

seed 101 仅展示参数位置，实际必须填冻结的污染 seed。评价生成 before.jsonl；后续方法在**完全相同**的模型、数据和生成规则下加 `--before-cache /results/G-triggered` 复用 θ0 输出。正常与触发输入各自有缓存，不能相互替代。每个修复后的模型仍真实生成；B0 回退不加载被拒绝增量，明确标注回退并复用起点输出。

常规单图 VQA/caption 只生成，不计算无用途的候选分数；配对事实仍保留候选评分。缺测候选为 null，不是零；自由生成不读取答案标签。记录原输出、分数及缓存身份，离线 HTML 不重复调用模型。

原攻击评估器返回 JSONL，每行至少包含：

    {"unit_id":"test-001","node_index":0,"phase":"after","method":"G","cell":"frozen-cell-01","seed":101,"condition":"triggered","attack_success":false,"attack_evaluator":"官方代码commit、原目标及精确评测配置"}

它必须与实际输出身份逐项对应。额外／重复 key 报错；缺测保持 null，不用关键词猜 ASR。官方评估器如需 GPU，也计 evaluation。

    ../../.venv-attribution/bin/python -m repair report /results/G-triggered/records.jsonl --attack-results /isolated/G-asr.jsonl --output /results/G-scored
    ../../.venv-attribution/bin/python -m repair compare /results/G/records.jsonl /results/G0/records.jsonl /results/Gl/records.jsonl /results/Rplus/records.jsonl /results/SFT/records.jsonl /results/RACER-data/records.jsonl --comparisons configs/comparisons.json --expected-keys /isolated/vqa-expected-keys.json --condition triggered --task vqa --metric vqa_soft --output /results/vqa-comparison

expected_keys 为 [cell, poison_seed, condition, unit_id, node_index, phase] 列表，phase=after。比较列表含 G0/R+、G/G0、G/Gl 及现有强对照，不含已后移的 P。按图片簇成对区间报告；caption、ASR、正常保护分别分析，不从一个 VQA 区间推出联合保证，不启用 5,000 图扩容。小样本主要筛明显增量，区间宽保持未决。

固定状态方向诊断仍可复用 `diagnose`，toy48 最多 4 个预声明独立 unit，从完整 fit 读资格、权重和分母。示例内层调用（GPU 执行时仍要套同一账本）：

    ../../.venv-attribution/bin/python -m repair diagnose --run /runs/G --update-unit-id scene1-color --data /data/dev-diagnostic.jsonl --method-a G --method-b G0 --device cuda:0 --output /runs/direction-dev

这里从 θ0／初始 Adam 状态提议实际更新，独立节点只用于测间隔和残差，不进入更新。U 锁定后可以加 --known --selection 测真实触发上的 U 更新，但本轮不执行 `train --known`。

## 计算简化与验证边界

PGD 改为官方多模态基座的一份 prompt 前向，跳过候选答案副本及 LM head；原候选前缀一致性校验保留，外层完整答案评分和损失不变。CPU tiny HF 测试比较标量、δ 及允许参数梯度和有限步 PGD。有限精度下近零梯度的 sign 仍可能分岔，不据此承诺全部真实 GPU 上逐比特相同或任何加速百分比。

没有引入新训练框架、复杂 KV／冻结特征缓存或量化。已支持的 P、Native、K 和历史多配置路径保留，但默认示例、阶段清单及比较列表均指向 toy48。

本次本地验证范围：

- 官方 HF 两种随机微型模型：完整 processor、候选／EOS、输入扰动与答案隔离、projector／LoRA 梯度、cached／uncached 生成、增量存取。
- 核心数值：PGD 梯度及恢复、共享／独立扰动、同权重多重集、G0/P 和 keep 尺度；非零 Adam 矩状态下线性 margin 预测到实际分差；独立评价不参与训练及异常恢复。
- 数据与选择：官方 VQA 规则、U/K 权限、场景／图片隔离、缺测不补零、坏指标／数据身份拒绝、B0 回退、冻结候选选择、同时簇重采样。
- 一个微型 HF CLI 训练链执行实际前后向、生成、保存和 HTML；仅任务评分用测试替身。缺失完整 caption 时选择如期拒绝，未把它当真实效用验收。

测试文件位于仓库 tests/test_repair_core.py、test_repair_data.py、test_repair_cli.py 及本项目 tests/test_repair_model.py。它们验证软件合同，不验证 H1–H4。没有 7B 下载、GPU toy 运行、真实攻击抑制、任务恢复或 SOTA 结果。

历史 V2.2 验证为 26 项通过；本轮共 35 项针对性检查通过（根目录 32 项、两骨干微型模型 3 项）；包括单配置、缓存、预算及 prompt-only 等价。根目录首轮有一项新增计时测试缺少夹具图像字段，补齐夹具后仅重跑该项通过，未重跑整套检查。没有重复执行旧 OA 探针或 Grond 实验。可在仓库根目录复核：

    .venv-attribution/bin/python -m unittest discover -s tests -p 'test_repair_*.py' -v
    .venv-attribution/bin/python -m unittest discover -s attribution-visualization/visual-evidence-repair/tests -p 'test_repair_model.py' -v

上述实现阶段的历史缺口记录：合格污染模型与配套资产、已核验配对和冻结真实数据划分、Java／CIDEr 原口径、首个合格完整修复计时与共同曝光日程。最新资产状态见 [GPU_READY.md](GPU_READY.md)。当前 48 GPUh 是软计划参考，不是实测 ETA，也不自动触发停止；真实科学实验状态以回执为准。

服务器排队补充验证：17 项 CPU 检查通过（双卡队列 5、测量器 5、既有预算 4、服务器等待器 3）。包含真实 CPU 子进程的重叠、单卡可见性、异常退出及后代清理、父作业完整计费、队友总控等待、输入缺失拒绝启动。它们不是双卡 GPU 冒烟或实验结果。测量器无法确认专属进程组已清理时记录 `cleanup_failed`，预算账本停止放行。
