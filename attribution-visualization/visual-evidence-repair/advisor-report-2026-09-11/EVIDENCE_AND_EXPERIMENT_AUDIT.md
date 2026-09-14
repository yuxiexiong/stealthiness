# 导师汇报用：既有证据与实验边界核验

日期：2026-09-11。性质：本仓库只读证据审计后形成的汇报材料；没有新增模型实验、服务器检查、全文检索或场景变更。论文内容复用本地冻结版本全文证据卡，作者结果与本项目实测分开。运行状态仅指仓库最新回执，不表示此刻服务器的实时状态。

## 一、已经完成的 OA 探针，究竟证明了什么

| 项目 | 核定事实 | 必须同时说明的边界 |
|---|---|---|
| 对象 | 同一 Llama-3-8B 基座；M0 原基座，M1 普通后门发布状态，M3 OA 的 MAD+probes 发布状态 | 是既有发布模型的观察，没有新训练种子或匹配训练对照 |
| 独立输入 | 12 条基础输入，每个状态分别无触发、原触发、替代标记 | 不是 8,274 个独立实验；原标记 5 token、替代标记 4 token，不能说严格等长 |
| 生成与归因 | 108 条轨迹，8,274 个已生成位置，8,274/8,274 个来源归因；缺失、失败均 0 | 22 条轨迹达到 200-token 上限，“全量”仅覆盖实际生成部分 |
| 条件重评分 | 324 次固定目标序列评分；每位置保留三输入条件概率及差值 | 固定同一目标和前缀；不是自由生成总效应 |
| 删除参照 | 48 次，4 输入×3 状态×2 目标×2 类任务正文词 | 没有认证所有来源、所有位置或整套归因方法 |
| GPU 记录 | 原 289.329 秒＋补齐 671.384 秒＝960.713 秒，约 16 分钟单卡模型计算 | 不含环境准备、下载、排障、传输和研究分析，不是端到端总成本 |
| 本批行为 | M0 的 36/36 条保持拒答或安全转向；M1/M3 非原触发 48/48 条如此；原触发 24/24 条出现配合表达及相关内容 | 是缓存文本观察，未执行作者完整 ASR 或一般能力评价，部分回答重复/发散 |

主数据入口为 `runs/attribution-probe/full_completion.json`；汇总说明见 `PROBE_STAGE_SUMMARY.md:7–19`。

最适合向导师展示的是三个“初看热图容易判断错，加入对照才修正”的结果：

| 观察 | 精确证据 | 得到的认识 |
|---|---|---|
| 首词不代表全程 | M1 首词 12/12 为触发片段峰；M3 首词 10/12 为模板峰。全程却是 M1 1,526/2,398、M3 1,595/2,348 位置以触发片段为峰 | 不能从首词推断整段来源转移，更不能据此宣布 OA 隐藏了输入归因 |
| 历史逐渐变亮并非后门独有 | 108/108 条轨迹后段历史绝对归因份额高于前段，同时 108/108 条都有局部下降 | 历史长度及普通生成也能产生这种图样，不能将其称后门传播证据 |
| 同样高亮可能对应不同决策 | M3/294 第143位空格原条件差为 +4.047，继续生成质量差区间 [4.020,4.064]，非结束集合内选择差 [−0.016,+0.027] | 总概率提高不一定是更偏好该词具体内容；停止/继续竞争必须分开 |

第三行来自事后缓存分析，仅存 top-3 导致区间界，并非新模型运行或统计置信区间。另有严格同目标/同历史的来源差异：原触发仅 3 输入、25 位置，其中 19 位置来自同一输入；不能把它当大样本机制证明。见 `PROBE_DEEP_MINING.md:11–55,59–86`。

**这些结果没有证明当前视觉修复假设。** OA 是文本模型、测的是输入/历史到输出的局部归因；新课题是图像事实编辑指导原 VLM 参数修复。可以承接的经验是测量契约、对照、原行为终点和失败案例；“事实响应损伤可指导更好修复”来自后续文献与理论整合，尚不是旧探针观察出的已证机制。旧数据有效，但不是新主线的预实验正结果。仓库已明确这一点：`PROBE_STAGE_SUMMARY.md:53–69`、`BENCHMARK_SOTA_REASSESSMENT_2026-09-10.md:124`。

## 二、当前两阶段 toy 的准确口径

| 维度 | 现行内容 |
|---|---|
| 科学目标 | 合法事实编辑得到的响应及其偏离，是否帮助原模型恢复正确任务表现；归因权重是否超过均匀/普通难度权重 |
| 模型/权限 | 一个 LLaVA-1.5-7B；视觉塔可信并冻结，修复 projector 和登记的语言 LoRA；U 只用正常数据、合法编辑与人工扰动，不能看真实触发、目标和测试标签 |
| 起点 | 未取得合格作者污染 checkpoint；自建固定标记→`Unable to answer.` 实例；不是原表复现，不是 OA 实验 |
| 构建资源 | COCO train2014 独立 2,200 图片，20k 正常指令＋1k 固定目标指令；不占修复/测试图片 |
| 事实资源 | 252 EditCLEVR 编辑对、504 图、756 个问题单元；每场景三个问题：变色对象、保持对象、派生计数 |
| 事实划分 | dev 24、fit 100、calibration 32、test 96；test 简单/干扰复杂/未见形色组合各 32 场景 |
| 正常真实任务 | 556 张 COCO val2014 原图；fit 100 VQA，calibration 64 VQA＋32 caption，test 300 VQA＋60 caption；原始人工参考答案 |
| 整体输入 | 1,312 个 unit、808 独立场景/图片簇、1,060 张不同图片；不把一场景多问题当独立样本 |
| 阶段 A | 入口资格和计时→dev 观察→冻结规则/日程→共享参照→六方法各一个配置修复→正常校准决定保留/回退 |
| 阶段 B | 所有方法冻结后进行独立正常/真实触发生成与事实配对评价；再做有限机制计算和离线图，不回流修改 U |
| 主输出 | 原模型参数增量、逐例真实答案与评分、正常新增错误、归因数值及图、失败/费用记录、是否值得追加投入的判断 |
| 预算 | 两卡合计约 48 GPUh，已改软计划；setup 6、reference 4、repair 20、evaluation 10、reserve 8；两卡占满约24墙钟小时只是理想换算，不是实际 ETA |
| 当前完成状态 | 仓库记录输入/哈希/CPU 接线与评分检查通过；B0 资格、真实 GPU 冒烟、完整六方法训练日程及修复有效性尚无通过结果 |

六方法的科学职责不能合并成“六个 SOTA”：

| 方法 | 为什么必须有 |
|---|---|
| SFT | 检查普通正确答案训练能否复制收益 |
| R+ | 共同训练底座，去掉事实响应项 |
| G0 | R+ 加均匀事实响应项；与 R+ 比较响应约束本身 |
| Gl | 同问题类型内，按普通答案 CE 难度重分配 G 的同一权重集合；检查是否只是强调难题 |
| G | 事实响应偏离产生冻结权重；当前主方法 |
| RACER-data | 论文方法在共同增强数据下的重建对照，端点独立扰动；不是作者官方已复现结果，也不是严格单变量消融 |

重要限制：场景实际仅是颜色编辑，复杂场景指干扰物，不能称关系移动/多属性干预；未见组合是 CoGenT 条件，不能称任意分布外泛化。P、G-shuffle、RACER-native、CleanSight、已知触发 K 训练、多种子、多模型、完整攻击矩阵后移。因此本轮不承担完整 SOTA 排名或普遍课题确认。

软预算已由代码支持：`repair/budget.py:111–120` 超额只提示，实际执行 `timeout=None`；不是只改文案。训练仍须按冻结的有限步数完成，实际异常/资格失败可以停止。`configs/stages.json` 末尾 budget_note 仍残留“pending clarification”文字，而最新主计划明确用户已确认合计 48 GPUh；汇报以最新确认及主计划为准，此处未修改配置。

## 三、上一轮拒答审查：已经写入建议，不等于已经实施

| 缺口/建议 | 仓库实际状态 | 汇报时正确说法 |
|---|---|---|
| 区分拒答、非拒答事实错、正确、无法解析 | 设计稿已有；当前 qualifier 仅固定目标 exact refusal 与任务评分 | “发现了评价混淆并提出修正”，不能说四类转移已测 |
| 在24 dev 场景测三种问题及标记条件的事实响应 | 现有 qualifier 只筛 changed_color；候选参照只算 B0 clean dev，其他格仅生成 | 新损伤分类面板尚未实现/运行 |
| 改变事实两端均答对、保持事实两端仍正确 | 现有合法数据和普通评分可复用，但新联合终点/转移分析尚属建议 | 不把已有两个节点的分数叫作已完成联合判别 |
| 原基座/B0 四格 | 已有实现与 CPU 测试，无本仓库已通过的真实 B0 资格 | 四格仍无法隔离正常微调与目标监督的区别 |
| 匹配训练正常对照 | 尚未新增 | 若需将损伤归因于污染监督，应补相同曝光/日程、目标样本使用真值的对照；只用于隔离评估 |
| 换为内容损伤主场景 | 尚未换，队列仍是原拒答构建 | 先诊断 B0，若主要为拒答，保留边界对照后另补主场景；不能把建议冒充已决定执行 |

理论对这个风险有直接说明：如果每个候选获得一个不随场景改变的偏置向量 b，编辑两端残差相减为 b−b=0。因而恒答一个错误颜色也不必造成事实响应损伤。更换目标文字不是自动解决方案。主终点应是“原模型正确使用事实的功能恢复”，不要求也不宣称“内部知识被删除后重建”。受限候选评分只是旁证，不能替代原自由生成。

这些是当前最重要的设计决策点，而不是已经跑出的失败结果。详见 `REFUSAL_VS_FACTUAL_REPAIR.md:3–5,17–45,49–80,88–94`。

## 四、邻近工作的角色与新颖性边界

以下文献判断来自本地已存全文证据卡/原文核查记录；不是本次新搜索或截至此刻的全球优先权认证。

| 工作 | 可以借什么 | 不能宣称什么 |
|---|---|---|
| Grond | Grad-CAM、TAC、参数/行为联合分析；提醒“看起来关注相同对象”不等于相同机制 | 图8仅两个例子并同时换输入/模型；没有本项目直接实验。OA不是历史上或方法上可直接叫作“文字版 Grond” |
| OA | 行为与指定内部观察器分离；发布状态可复用；SQL实际执行正确性值得借鉴 | OA已给输出逐token按SAE激活上色；没有证明所有归因均失效，我们也没有破解 OA |
| PECoRe / Inseq | 先识别受上下文影响的输出、再找输入来源；输出/历史矩阵、明确目标和前缀、HTML工具 | “给文字热图”“两步来源归因”“展示EOS/历史”都不是新贡献 |
| CAGE / ContextCite | 历史传播视角、删源验证、实际行为评价；已讨论固定前缀遗漏中介影响 | 有序图不自动具有因果识别；ContextCite已有上下文投毒定位，不能称首次用归因处理投毒 |
| SocRAT / BkdAttr | 早期伪关联注入后输入–输出图与删除验证；现代后门头归因与实际干预 | 归因可视化用于发现问题再采取行动的总理念已有先例 |
| BackdoorVLM | 12攻击/5目标/3数据集的任务和评分骨架 | 是攻击 benchmark，不是统一防御榜单；公开代码不等于配套污染权重已到手 |
| RACER | 区域层间不一致、有限 PGD、正常小样本修复；是方法底座和强竞争者 | 不是我们原创的 min–max 修复；RACER-data 是共同数据下论文重建；目前不能宣布优于作者或全球 SOTA |
| CleanSight / BackdoorBench | 已有 poisoned utility / R-Acc，提醒需报触发样本正确性及正常损害 | 不能以“同行只看 ASR，我们首次看正确事实”作新颖性 |
| TokenSwap / 视觉编辑邻居 | 已有关系错绑、事实读出与干预、视觉证据再激活问题 | 关系错绑本身或用归因选修复位置不是未经研究的新概念 |

当前可争取的贡献是一个必须由对照成立的具体结果：**在相同资源和修复权限下，事实编辑响应提供普通答案训练/一致性/困难度权重没有的有效信息，实际改善原 VLM 的触发事实正确性，同时保护正常功能。** 可视化负责提出/冻结操作及检查错与得，不以图更漂亮代替效果。现有理论给出可执行算法与条件推论，尚没有证明这条经验 claim。

RACER 的数字只能用来说明已有难点：记录中的 LLaVA Blended–MI 修复后 ASR 20.4% 是作者 Table I 结果，不是我们的基线实测；和自建拒答 toy 不能横向相减得出改进幅度。

## 五、导师最可能追问的五个问题

1. **“你们现在真正发现了什么，为什么从 OA 跳到 VLM？”** 旧探针确实修正了热图判读，但没有支持新方法有效；新主线是文献/理论整合后的研究假设，须独立 toy 检验。
2. **“如果只是恢复愿意回答，为什么非要你们的方法？”** 这是当前未闭合的实验场景问题。先做固定分母的拒答/事实错误转移和合法编辑响应；必要时把拒答实例降为边界对照，而非事后改写收益性质。
3. **“和 RACER 加数据、加正则相比，新信息到底是什么？”** 用 G0/R+ 分离响应项，G/G0 和 G/Gl 分离归因权重，SFT/RACER-data 检验普通训练与现成方法能否复制收益；所有共同资源公开记账。
4. **“正常参考和人工扰动怎么保证代表真实未知后门？”** 不能保证。N1排除错误正常参照，合法编辑核验真值，N2搜索定义可执行；人工到真实、合成到真实都是待验桥梁，失败必须体现在原任务上。
5. **“48 GPUh 小实验能立起多强的论文结论？”** 它筛局部方法增量，不确认全领域SOTA。正结果后补匹配训练/重复性/更广场景/原协议强对照；窄样本或入口不合格导致未决时不能宣称方向被证伪。

## 证据定位

- 旧探针总量、完成边界：[阶段总结](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/PROBE_STAGE_SUMMARY.md:7)，[原完成回执](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/runs/attribution-probe/full_completion.json:1)。
- 三个观察与近邻边界：[阶段总结](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/PROBE_STAGE_SUMMARY.md:21)，[缓存挖掘](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/PROBE_DEEP_MINING.md:11)。
- 六方法与两阶段：[现行计划](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair/TOY_PLAN.md:25)；软预算：[预算代码](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair/repair/budget.py:111)。
- 数据/部署边界：[上卡准备](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair/GPU_READY.md:9)，[编辑数据来源](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair/GPU_READY_DATA_AUDIT.md:17)。
- 新建议未实施：[拒答与事实修复](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair/REFUSAL_VS_FACTUAL_REPAIR.md:3)，[现有入口只取 changed_color](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair/tools/qualify_baseline.py:49)，[只有 B0 clean 计算候选参照](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair/tools/qualify_baseline.py:165)。
- Grond/OA 冻结全文卡：[后门域](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/review/backdoor/FULLTEXT_NOTES.md:30)；PECoRe/Inseq/CAGE：[生成域](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/review/generation/FULLTEXT_NOTES.md:20)；ContextCite/SocRAT：[生成域](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/review/generation/FULLTEXT_NOTES.md:130)。
- 当前基准与原文数字：[基准重评](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair/BENCHMARK_SOTA_REASSESSMENT_2026-09-10.md:19)，[资源缺口](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair/TOY_RESOURCE_AUDIT.md:9)。
