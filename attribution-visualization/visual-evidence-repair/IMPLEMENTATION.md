# 新理论与两阶段 toy：实现及复用记录

日期：2026-09-10。本次按用户要求实现修复后的两阶段方案；使用 Ponytail full。代码验证与真实污染实验分别记账，不把小模型软件测试称为科学结果。

## 编码前资源复核

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

采用 Python 标准库 CLI、JSON/JSONL 和现成依赖；一个模型加载器、一个修复核心、一个数据入口和一个离线报告，不新建训练平台。运行目录禁止覆盖，训练失败与预算结束保留状态。最终保存原模型允许参数增量，实际生成读取该增量。

## 已实现什么

| 文件 | 实际工作 |
|---|---|
| [repair/model.py](repair/model.py) | 官方 HF LLaVA（CLIP＋Llama）和 Qwen2.5-VL 加载；原 projector／merger 与语言 LoRA；候选含 EOS 评分、实际生成及增量保存 |
| [repair/core.py](repair/core.py) | N1–N8：合法共享 PGD、合格参照、静态权重、独立分母、AdamW；N9：真实更新差与全部竞争者的间隔、残差 |
| [repair/data.py](repair/data.py) | 图像字节及场景划分检查；U、K、test 用途分离；接收已有合法事实对，不伪造反事实图 |
| [repair/assets.py](repair/assets.py) | 模型／processor／已有污染 adapter 的一次内容哈希清单；运行时核对文件集合、大小及修改时间 |
| [repair/report.py](repair/report.py) | 复用官方 VQA 规则、可选 COCO CIDEr、原 ASR 结果精确连接、离线图、成对簇区间和精度规划 |
| [repair/__main__.py](repair/__main__.py) | 观察、训练、冻结选择、独立评价、固定状态诊断与结果比较；没有后台调度平台 |

内置训练方法是 G、G0、Gl、P、R+、SFT、G-shuffle，以及 RACER-data／RACER-native 的论文重建。G0/P 没有额外二分之一；RACER-data 按端点独立搜索，R+ 才配对共享；普通对照不计算用不到的训练归因参照。Native 要求 100 个 singleton，calibration=null，直接使用冻结配置，不能进入增强 U 选择。

CleanSight 的完整检测／净化实现、BackdoorVLM 原攻击构建器／评估器仍使用外部官方交付。本代码接收其结果，**未将它们冒充内置完成**。编码器投毒的 BadVision 不自动等同本轮下游污染资产；原作者 llava_llama 格式也不能被 HF 加载器静默替换。

主要方法代码可运行不等于第二阶段所有基线已就绪。完整污染权重、匹配触发、精确基座、合法场景编辑、原协议样本及评估器还需按原计划验收一次；当前不自动构建新攻击，不用微型随机模型充当科学结果。

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

[LLaVA 配置](configs/llava.example.json)与[Qwen 配置](configs/qwen.example.json)需要填入实际路径。两份文件是接口示例：步数、时间上限、epsilon、损失系数与 256-token 生成上限尚未经过原协议核准或首跑计时，不能直接当作已冻结科学配置。deep_start 是首个相邻层对的状态索引；示例分别取 22 和 14，仍须核对实际骨干深度。不要只因写在 JSON 里就称其为原作者参数。

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

## 第一阶段：先看图，再冻结和训练

以下命令的运行目录均须不存在，防止覆盖证据。原始测试材料始终与修复者配置分开。

    ../../.venv-attribution/bin/python -m repair inspect --config configs/llava.local.json --data /data/dev.jsonl --device cuda:0 --output /runs/dev-observation
    ../../.venv-attribution/bin/python -m repair train --config configs/llava.local.json --device cuda:0 --output /runs/G-config1

inspect 只观察 dev 并输出 index.html；修复后栏明确未生成。开发集上的候选权重不是最终训练权重。根据可视观测是否支持规则，保留一次修订的原因与预期，再锁定；不能等独立触发结果揭示后改规则。

train 输出：

- update.pt：原 projector／语言 LoRA 的实际允许参数值，加载时核对准确基座规格与参数名。
- reference-cache.pt、reference-eligibility.jsonl：正常参照、实际正常生成、资格、固定 δ、偏离、困难度和权重来源。
- training.jsonl、run.json：损失分量、步数、资格覆盖、唯一节点与出现次数、更新范数、资产／数据／缓存身份和成本。
- attribution.jsonl／attribution.html：G 类训练边的固定参考环境下，事实改动、有符号响应、实际权重及前后真实回答；无资格位置显示缺测。
- calibration.jsonl／calibration.html：正常与冻结人工扰动下的前后实际生成；这些不是独立测试成绩。

共同配置下只改 method 即可做规定的局部消融；完整方法竞争最多四个预登记配置。校准搜索配置独立于训练配置，所有候选必须具有相同人工扰动内容哈希、正常数据、资产、生成和方法身份。不同方法的同资源比较仍须按计划保持数据／参数／预算；CLI 不替研究者假定比较组已经公平。

    ../../.venv-attribution/bin/python -m repair select /runs/G-config1 /runs/G-config2 --limits configs/selection.json --output /runs/G-selection.json

选择按正常 VQA／CIDEr 门槛，随后按固定代理 VQA 实际生成分数、总时间排序。这里只选同一方法的配置，不从测试中挑赢家。无合格非零更新时写 no_acceptable_update；这是 B0 回退，不能算找到修复。Native 不使用此选择，直接按原冻结运行评价。

### 固定状态方向诊断

    ../../.venv-attribution/bin/python -m repair diagnose --run /runs/G-config1 --update-unit-id scene1-color --data /data/dev-diagnostic.jsonl --method-a G --method-b G0 --device cuda:0 --output /runs/direction-dev

该命令从原始 θ0／初始 Adam 状态出发；读取完整 fit 的资格、权重和 N/M、N/R，只对指定 fit unit 提议两次更新，再测独立 dev 的所有候选间隔，最后精确恢复参数。它不是完整训练的效果预测。底层 one_step_diagnostic 也支持传入非零优化器矩状态，其数值测试覆盖这一情形。

U 锁定后，加 --known --selection /runs/G-selection.json 并给独立真实触发诊断材料，可用真实触发评价上述 U 更新。真实触发节点不进入 loss。另一个 train --known --selection ... 才是信息增强 K 训练；需 K 配置与已冻结 U 候选匹配、材料确实对应相同 U 样本。两种诊断不混称。

## 第二阶段：复用同一实现，冻结范围而非另写框架

[stages.json](configs/stages.json)登记 4 个开发污染状态及第二阶段 12 格 × 2 个独立污染实例、干净控制和数据上限。这是评估协调清单，不传给 U 训练器；实际污染 seed、资产和预算还未填好，不登记为已跑状态。每个家族用冻结规则与配置，不按攻击标签路由。

    ../../.venv-attribution/bin/python -m repair evaluate --selection /runs/G-selection.json --data /isolated/test-triggered.jsonl --cell llava-cell-01 --seed 101 --device cuda:0 --output /results/G-cell01-seed101

使用当前图像与问题自由生成，加载修复增量后再次生成；test 的 truth 只在隔离评分端使用。没有把 caption 缩成颜色题。结果保存原输出、分数、独立测试清单和评测成本。Native 使用 --run /runs/native-fixed；K 使用 --run /runs/K-fixed 并带它原有的 U selection。

原攻击评估器返回 JSONL，每行至少包含：

    {"unit_id":"test-001","node_index":0,"phase":"after","method":"G","cell":"llava-cell-01","seed":101,"condition":"triggered","attack_success":false,"attack_evaluator":"官方代码commit、原目标及精确评测配置"}

它须与保存输出的全部身份准确对应。额外／重复 key 报错；缺测保持 null，绝不按任意关键词或空输出猜 ASR。

    ../../.venv-attribution/bin/python -m repair report /results/G-cell01-seed101/records.jsonl --attack-results /isolated/G-asr.jsonl --output /results/G-scored

使用 [预声明比较列表](configs/comparisons.json)和开测前生成的 expected_keys，跨方法比较时不取共同交集隐藏漏样：

    ../../.venv-attribution/bin/python -m repair compare /results/G/records.jsonl /results/G0/records.jsonl /results/Gl/records.jsonl /results/Rplus/records.jsonl /results/P/records.jsonl /results/SFT/records.jsonl /results/RACER-data/records.jsonl --comparisons configs/comparisons.json --expected-keys /isolated/vqa-expected-keys.json --condition triggered --task vqa --metric vqa_soft --output /results/vqa-comparison

expected_keys 是预声明的 [cell, poison_seed, condition, unit_id, node_index, phase] 列表，phase 为 after。各方法的全量状态需先按 JSONL 合并，不能只给一个格子却解释成 24 个状态。compare 使用共同图像簇重采样，保留跨状态共享图像的相关性；返回这个终点各主要比较的近似同时区间，并逐污染种子报告点估计。

ASR、正常保护、caption 与主增益是不同终点：一个 VQA 区间不自动确认联合主张。零经验方差尤其不能靠零宽 bootstrap 证明总体安全。caption 保护需按相同原评估口径另作成对分析；有限样本、边界率和多终点的区间方案在真实开测前核定。当前代码不会自动宣布达到 SOTA。

report.plan_precision 提供以开发集独立簇差值估方差的 Bonferroni 正态近似，固定 0／5 pp 情形、3 pp 分界及 5000 簇上限；零方差或所需量超上限显式未决。变簇大小的每题均值需输入对应簇影响量，不能把每个 token／问题当独立样本；它不是有限样本功效保证。

## 成本、验证与仍未完成的交付

记录完整模型加载、参照、内层搜索、外层训练、校准、实际生成和可视记录成本。训练时间上限在单元边界检查，包含此前设置与参照成本；最后一个单元及后续验收可能超出软预算，日志照实计入。严格作业上限使用既有外层工具，它终止的未完成运行不进入选择：

    ../../.venv-attribution/bin/python ../../experiments/measure.py --out /runs/measured-G --cwd . --gpus 0 --cost-role method_validation -- ../../.venv-attribution/bin/python -m repair train --config configs/llava.local.json --device cuda:0 --output /runs/G-config1

内部成本 hook 挂在实际 language_module，覆盖 PEFT 下的自由生成；CUDA 峰值按实际设备读取，不把 CPU 运行或另一块 GPU 的状态记进来。失败和不同候选费用同样属于实验总成本。跨方法共同预算与更便宜方法的冻结延长日程仍按 TOY_PLAN 执行，不由“同四个配置”推成同成本。

本次本地验证范围：

- 官方 HF 两种随机微型模型：完整 processor、候选／EOS、输入扰动与答案隔离、projector／LoRA 梯度、cached／uncached 生成、增量存取。
- 核心数值：PGD 梯度及恢复、共享／独立扰动、同权重多重集、G0/P 和 keep 尺度；非零 Adam 矩状态下线性 margin 预测到实际分差；独立评价不参与训练及异常恢复。
- 数据与选择：官方 VQA 规则、U/K 权限、场景／图片隔离、缺测不补零、坏指标／数据身份拒绝、B0 回退、冻结候选选择、同时簇重采样。
- 一个微型 HF CLI 训练链执行实际前后向、生成、保存和 HTML；仅任务评分用测试替身。缺失完整 caption 时选择如期拒绝，未把它当真实效用验收。

测试文件位于仓库 tests/test_repair_core.py、test_repair_data.py、test_repair_cli.py 及本项目 tests/test_repair_model.py。它们验证软件合同，不验证 H1–H4。没有 7B 下载、GPU toy 运行、真实攻击抑制、任务恢复或 SOTA 结果。

2026-09-10 最终针对性验证：仓库 repair 测试 24/24、官方微型模型测试 2/2，共 26 项通过；Git 文本差异检查通过。没有重复执行旧 OA 探针或 Grond 实验。可在仓库根目录复核：

    .venv-attribution/bin/python -m unittest discover -s tests -p 'test_repair_*.py' -v
    .venv-attribution/bin/python -m unittest discover -s attribution-visualization/visual-evidence-repair/tests -p 'test_repair_model.py' -v

真实开跑前仍缺：合格污染模型和精确匹配资产、已核验 CLEVR 配对与真实任务划分、Java／CIDEr 原口径、CleanSight 完整基线交付、首个完整修复计时及共同预算、各必要终点的独立精度清单。两阶段科学验收据此保持未完成。
