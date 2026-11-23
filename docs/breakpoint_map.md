# ASR 短窗跃升论文断点图

截至 2026-11-23。主线考察后门微调中攻击成功率（ASR）的短窗跃升与回落。一个问题是许多题为何在邻近训练步翻转，另一个问题是给定一段已经发生的权重更新，换一个模型起点能否预测哪些题翻转。两者的证据门槛分别记录在 D7 和 C6。

## 1. 断点全表

「声明」列写待检验的命题，包括已撤回的命题。「真」表示该声明超出证据或已有反例；「表述」表示已有数据足以纠正、问题在于正文、图表或字段容易误读。高表示直接影响自然成因或逐题预测增量这两项主张之一；中表示需限定端点、适用范围或证据强度；低表示修正后，给定更新、改变接收起点的反事实仍成立。严重度不等同于待做实验数。

读数约定：CA 是同任务干净题正确率；T 与 C 分别是触发输入和配对干净输入上的目标竞争分差，D = T − C；g 是更新交叉实验用于预测生成命中的逐题竞争分差。实际成功或失败以生成答案判定。M 编号指原项目审查或实验记录；共引用 40 份证据文件。25 个链接固定到对应的历史提交，另 15 个相对链接指向与本文同版本的仓库文件。

| 编号 | 环节 | 声明 | 证据路径 | 断点 | 类型 | 真/表述 | 严重度 |
|---|---|---|---|---|---|---|---|
| A1 | 问题与威胁模型 | 微调投毒的结论适用于预训练注入与 clean-label 攻击 | [attribution-microscope/src/poison.py][src-poison]、[asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms] | 主线改变输入并改写答案；仅加触发的分解臂未验证独立成立的 clean-label 后门。两种外推设置均无同协议交叉。 | 泛化 | 真 | 中 |
| A2 | 问题与威胁模型 | 真实部署者也能识别或利用该关系 | [asr-rising-integration-2026-11-07/02_因果证据与边界.md][src-p2]、[asr-effective-update-2026-11-16/REPORT.zh-CN.md][src-m31] | P2 知道毒行身份和消费位置；更新移植要求供体、接收权重与逐题读数。没有仅凭服务输出或最终模型的验证。 | 泛化 | 真 | 低 |
| A3 | 问题与威胁模型 | 指定目标词命中代表开放生成安全风险 | [asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms] | VQAv2 以规范化后的首个生成词命中 violin 计分；未测有害遵从、策略绕过或下游损失。 | 泛化 | 真 | 低 |
| A4 | 问题与威胁模型 | 基础配置中的目标就是实跑目标 | [attribution-microscope/configs/protocol.yaml][src-cfg]、[asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms] | YAML 留有候选 banana，实际冻结的是 violin。整合稿已纠正，直接复用默认配置仍会选错。 | 表述 | 表述 | 低 |
| B1 | 攻击与评测 | 所有“短窗”判据给出同一结论 | [asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross] | W50、首次与持续过半、10%→90% 宽度衡量不同事件；LLaVA 主口径 6/9，辅助口径 9/9。 | 对照 | 真 | 中 |
| B2 | 攻击与评测 | 各模型的目标偏好均可归因于所设触发器 | [asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross] | Falcon 的 11 条记录缺匹配 clean 训练臂，近似触发命中亦偏高。一般目标偏好与指定触发效应未分离。 | 对照 | 真 | 中 |
| B3 | 攻击与评测 | 原题选出的窗口都获得新题同点确认 | [asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross] | Llama 观察 5/5、独立题集确认 4/5；Qwen 新增五种子同点确认 5/5；LLaVA 主组没有独立选窗外题池。 | 泛化 | 真 | 中 |
| B4 | 攻击与评测 | 网格间没有峰值，测得跨度就是精确窗宽 | [asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross]、[asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms] | SD3.5 网格为 100 更新；P2 只有 25 个生成测点。46.16 是持续分差过零的插值尝试步差，生成读数相差 40 个网格步。 | 对照 | 真 | 中 |
| B5 | 攻击与评测 | 首 token 分差、argmax 和真实生成可互换 | [asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross]、[asr-xo-confirmation-2026-11-16/REPORT.zh-CN.md][src-xo] | CLIP 有行为阳性却未过分差仪器门；XO 的交叉也有分差符号与生成错配。 | 对照 | 真 | 中 |
| B6 | 攻击与评测 | CSV 中 confirmed 就是独立复现 | [asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross] | 旧字段可仅表示本轨迹存在合格点对；严格重放和新题确认另有字段。正文已澄清。 | 表述 | 表述 | 低 |
| B7 | 攻击与评测 | 逐题或逐点行数就是训练重复数 | [asr-ordering-2026-11-16/REPORT.zh-CN.md][src-m45]、[asr-effective-update-2026-11-16/REPORT.zh-CN.md][src-m31]、[asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms] | M045 的 2400 是 200 题×12 点；M031 的 3200 是 200 题×16 条件。它们不估计训练种子或历史抽样的不确定性。 | 对照 | 真 | 中 |
| C1 | 分析方法 | 归因热图亮度定量代表触发器使用强度 | [attribution-microscope/REPORT_FINAL.md][src-attr] | 两台仪器未过完整 W0 资格；低剂量一组 93.5% 题的触发区梯度带符号和为负，遮挡份额还受窗口尺寸影响。 | 对照 | 真 | 低 |
| C2 | 分析方法 | M1 的 32 个模块是唯一或最小电路 | [attribution-microscope/supplement/jump/m1/CONTRACT_M1.md][src-m1c]、[attribution-microscope/supplement/jump/m1/RESULTS_M1.md][src-m1] | 100 题选集、100 题验收，有集合阶梯、正常任务与随机对照；没有穷尽等效集合和更小组合。 | 机制 | 真 | 低 |
| C3 | 分析方法 | IP 精确确定第 24 层是信息转移位置 | [attribution-microscope/supplement/jump/ip/RESULTS_IP.md][src-ip] | 开发对选 L24，仅测偶数层；两个验证对在预设 20% 转移界下判档。未分辨相邻奇数层、其他位置划分和判界。 | 对照 | 真 | 低 |
| C4 | 分析方法 | 逐行梯度账本已能解释触发侧分差 T | [asr-rising-integration-2026-11-07/02_因果证据与边界.md][src-p2] | M3T 的增量相关 0.726、累计比 1.590，未过 ≥0.9 和 0.8–1.25 的校准门；D 账本通过不能替代 T。 | 可复现 | 真 | 中 |
| C5 | 分析方法 | 异配更新效果对 FP16 与 FP32 等价 | [asr-effective-update-2026-11-15/REPORT.zh-CN.md][src-m26]、[asr-effective-update-2026-11-16/REPORT.zh-CN.md][src-m31] | M026 一个边界题随精度改变而翻转，原配 ASR 50%→49.5%；M031 用修订后的统一 FP32 仅读指定交叉。缺原生 FP16 异配等价证据。 | 可复现 | 真 | 中 |
| C6 | 分析方法 | 逐题响应相对均值的增量经独立事前主检验确认 | [asr-xo-analysis-2026-11-16/REPORT.zh-CN.md][src-m35]、[asr-xo-analysis-2026-11-16/MEAN_BASELINE.json][src-mean] | 逐题/均值/零更新的错误数为 86/149/359 与 19/49/113；逐题预测事前冻结，均值比较在看过交叉后登记。 | 对照 | 真 | 高 |
| C7 | 分析方法 | 移植单独识别了更新方向或局部线性 | [asr-core-argument-2026-11-16/CORE_ARGUMENT.zh-CN.md][src-core] | 被移植的是含方向、大小与多步组合的实际有效权重增量；未单独操纵方向、范数，也未确立普遍线性邻域。 | 机制 | 真 | 中 |
| C8 | 分析方法 | 聚合 ASR 一致足以证明逐题均准确 | [asr-xo-analysis-2026-11-16/REPORT.zh-CN.md][src-m35] | XO 的一格同为 19%，却有 4 漏、4 虚报；另一格实际新增 7 题只抓到 1 题。边界误差不能解释全部错误。 | 对照 | 真 | 中 |
| D1 | 因果声明 | P2 已分离正常样本成员、排列或批内同伴的单独效应 | [asr-rising-integration-2026-11-07/02_因果证据与边界.md][src-p2] | A/B 固定毒行身份与共同后续，前缀中的正常成员、排列、批内同伴同时改变。 | 因果 | 真 | 中 |
| D2 | 因果声明 | 相同后续输入证明时差只经语义参数传播 | [asr-rising-integration-2026-11-07/02_因果证据与边界.md][src-p2]、[asr-rising-review-icml2026-2026-11-07/COORDINATOR_SOURCE_CHECKS.md][src-repair]、[asr-short-fourcell-cross-2026-11-11/RESULT.zh-CN.md][src-shortcross] | 原 P2 的种子 1001 中，B 在第 67 步跳过整个含毒批；到第 700 步 A/R、B 成功含毒更新分别为 107、106 次。另一种子 1007 的四格交叉按持续 50% 与 90% 判断得到不同归属。共同后续输入仍允许有效更新不同。 | 因果 | 真 | 中 |
| D3 | 因果声明 | M013 的窄化是纯触发器记忆的滞后效应 | [asr-exposure-fourarm-analysis-2026-11-15/REPORT.zh-CN.md][src-m13]、[asr-core-argument-2026-11-16/CORE_ARGUMENT.zh-CN.md][src-core] | 样本版本替换同时改变图像、答案及长度；A 的形成窗与处理期重叠，首次窗宽界 25–35→10–20 步；B 的宽界重叠。 | 因果 | 真 | 中 |
| D4 | 因果声明 | Qwen GATE 证明先具备正常能力是必要条件 | [asr-rising-integration-2026-11-07/02_因果证据与边界.md][src-p2] | 三母本分别提前 11、32、7 步；随机对照只匹配逐模块 LoRA B 范数。没有反向阻止正常能力并保持其余状态的检验。 | 因果 | 真 | 低 |
| D5 | 因果声明 | 某个 Adam 矩始终驱动跃升 | [asr-mr320-analysis-2026-11-15/REPORT.zh-CN.md][src-m20]、[asr-gradient-replay-analysis-2026-11-15/REPORT.zh-CN.md][src-m23] | 联合矩与随机状态交叉的作用随阶段改变；固定有效梯度的重放中，指定自然后期促进反转。没有分离单个矩的份额。 | 因果 | 真 | 中 |
| D6 | 因果声明 | 逐题分差 g 是唯一因果中介 | [asr-effective-update-2026-11-16/REPORT.zh-CN.md][src-m31]、[asr-xo-confirmation-2026-11-16/REPORT.zh-CN.md][src-xo] | 操纵的是完整权重起点与实际位移；g 用于预测，未被独立操纵。 | 因果 | 真 | 中 |
| D7 | 因果声明 | 自然训练为何在指定时段生成集中翻转与特定窗宽已获解释 | [asr-effective-update-2026-11-16/REPORT.zh-CN.md][src-m31]、[asr-ordering-2026-11-16/REPORT.zh-CN.md][src-m45]、[asr-xo-analysis-2026-11-16/REPORT.zh-CN.md][src-m35] | 交叉前已知供体端点。M045 真实排序两臂主路径差预测 +70.667 个百分点、实测 −2.583；尚无区分起点分布加共同漂移与历史依赖响应的未见自然窗宽结果。 | 因果 | 真 | 高 |
| D8 | 因果声明 | M058 指定前史放大后续含毒批对 T 的推动 | [asr-history-response-2026-11-18/REPORT.zh-CN.md][src-m58]、[gap2-initial-position-2026-11-20/FLAT-SHIFT.zh-CN.md][src-flat] | 前史将进入时 T 抬高 2.95；在相同后缀下，320→330 两臂的逐题 P−K 平均 T 响应为 1.13 对 1.28。过零差预测 4 题、实测 17；事后重建虽得 17，KK 仅重合 39/52 个事件。 | 因果 | 真 | 中 |
| D9 | 因果声明 | M1 集合控制真实生成且能保住同任务干净正确率 | [attribution-microscope/supplement/jump/m1/CONTRACT_M1.md][src-m1c]、[attribution-microscope/supplement/jump/m1/RESULTS_M1.md][src-m1] | 保留与删除 32 模块时，D 的恢复量为 1.081、剩余量为 0.023，达到指定终态、集合操作与 D 读数的组级充分性和必要性判据。缺同一操作下配对的真实生成 ASR 与同任务干净正确率。 | 对照 | 真 | 低 |
| D10 | 因果声明 | 抽掉含毒批的效果就是其中一条毒行的必要作用 | [asr-mainline-inventory-2026-11-18/INVENTORY.zh-CN.md][src-inv]、[asr-seeds-2026-11-17/REPORT.zh-CN.md][src-m55] | 上升窗整批移除的效应在四个新增种子复现；操作同时移除正常同伴与该次联合更新。 | 因果 | 真 | 中 |
| D11 | 因果声明 | ASR 回落证明内部能力消失，时差可以分摊给 C、D | [research-mentor/reviews/M039/EXPLANATION_LEDGER.zh-CN.md][src-ledger]、[asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms] | XO 回落 75%→41.5% 时 199/200 题的 D 仍增。T=C+D 是逐题恒等式，分别取中位数并不保持可加。 | 机制 | 真 | 中 |
| E1 | 检测与控制 | 能判别触发输入的同一特征控制了目标答案 | [attribution-microscope/REPORT_FINAL.md][src-attr]、[attribution-microscope/supplement/jump/ip/RESULTS_IP.md][src-ip]、[attribution-microscope/supplement/jump/m1/RESULTS_M1.md][src-m1] | 解码、位置修补与模块集合对应不同对象和读数；未在同一候选上同时验证触发可读、行为干预与同任务干净正确率。 | 机制 | 真 | 低 |
| E2 | 检测与控制 | 从 L24 到 L26 可沿一条指定路径追踪控制信号 | [attribution-microscope/supplement/jump/ip/RESULTS_IP.md][src-ip] | 两层的分组修补给出位置效应的差异，没有隔离中间边或检验替代路径。 | 机制 | 真 | 低 |
| E3 | 检测与控制 | 触发检测分数高即可识别训练中的毒样本 | [asr-jump-sparse-defense-2026-10-12/RESULTS.md][src-jsd] | 变体 A 见证集的 Δloss 在第 31 步 AUROC 为 0.974；局部 640 行中 15 条毒行只检出 1 条。变体 B 开发组 JSD 未选中毒行，干净训练亦有告警。 | 对照 | 真 | 低 |
| E4 | 检测与控制 | 未到转移门槛意味着该位置不承载信息 | [attribution-microscope/supplement/jump/ip/RESULTS_IP.md][src-ip] | 实测转移量约 7%–19%，报告的分类门槛为 20%；定性标题容易掩盖非零值。 | 表述 | 表述 | 低 |
| F1 | 泛化与边界 | 所选 A/B 历史的时差、窗宽与完整 700 步终点均已跨种子确认 | [asr-rising-integration-2026-11-07/02_因果证据与边界.md][src-p2]、[asr-short-fourcell-resume-2026-11-11/RESULT.zh-CN.md][src-shortaa]、[asr-short-fourcell-cross-2026-11-11/RESULT.zh-CN.md][src-shortcross] | 另一训练种子 1007 在相同后续输入下，AA/BB 的持续 50% 点为 320/360，90% 点为 340/380，两组从 50% 到 90% 都历时 20 步。该复核止于 440 步，历史仍经预选。 | 泛化 | 真 | 中 |
| F2 | 泛化与边界 | 同一个逐题响应向量在两个种子之间可移植 | [asr-effective-update-2026-11-16/REPORT.zh-CN.md][src-m31]、[asr-xo-confirmation-2026-11-16/REPORT.zh-CN.md][src-xo]、[asr-xo-analysis-2026-11-16/REPORT.zh-CN.md][src-m35] | 1007 与 1008 各自量自己的供体响应，片段长度分别为 5、10 步，共用模型与题池。未做跨种子位移或向量互换。 | 泛化 | 真 | 中 |
| F3 | 泛化与边界 | 跨模型的短窗现象证明同一内部更新机制 | [asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross]、[asr-rising-integration-2026-11-07/02_因果证据与边界.md][src-p2]、[asr-effective-update-2026-11-16/REPORT.zh-CN.md][src-m31]、[asr-xo-confirmation-2026-11-16/REPORT.zh-CN.md][src-xo] | 现象表跨模型；P2 的前史替换、Qwen GATE 的状态植入与有效更新交叉有不同操作和终点。没有跨架构同协议的逐题响应检验。 | 泛化 | 真 | 中 |
| F4 | 泛化与边界 | 更换触发器、目标或剂量仍得到同一机制 | [asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross]、[attribution-microscope/configs/protocol.yaml][src-cfg]、[asr-xo-confirmation-2026-11-16/REPORT.zh-CN.md][src-xo] | 现有组间比较同时改变任务、训练或测量条件；XO 沿用本任务设置，不能单独识别触发器、目标或剂量的作用。 | 泛化 | 真 | 中 |
| F5 | 泛化与边界 | 同任务干净正确率正常，足以判定机制是后门特有 | [asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms]、[asr-experiment-chronicle-2026-11-11/条目台账.csv][src-events] | 缺同一普通目标、学习进度匹配的逐题更新交叉。K0/K1 尝试以正常题负对数似然匹配进度，12/12 轨迹的进度尺均未过可用门槛。同任务干净正确率回答干预副作用，对照问题不同。 | 对照 | 真 | 中 |
| F6 | 泛化与边界 | 含毒批的作用在新种子和新顺序上都有同一时间规律 | [asr-seeds-2026-11-17/REPORT.zh-CN.md][src-m55]、[asr-mainline-inventory-2026-11-18/INVENTORY.zh-CN.md][src-inv] | M055 固定顺序的四个新种子中，「中期保持更多」两条确认且轨迹近乎重合，两条未判定；M057 的两种新排列在固定计数下仍产生早期差异。 | 泛化 | 真 | 中 |
| F7 | 泛化与边界 | 单头干预在不同训练轨迹上保住正常功能 | [k29-confirm-heads-2026-10-20/results-r03/REPORT.zh-CN.md][src-k29]、[k29-confirm-heads-2026-10-20/results-r03/ENDPOINTS.csv][src-kend] | 两条独立于选头的轨迹，攻击指标效应为 6.58、5.25 个百分点，同任务干净指标变化分别为 −6.46、+0.38 个百分点，代价不一致。 | 对照 | 真 | 低 |
| G1 | 防御启示 | 廉价报警能在风险形成之前预警 | [asr-defense-toy-2026-10-10/r02/RESULTS.md][src-alarm] | r02 风险首次出现在第 37 步，廉价策略第 38 步才报，周期检查第 40 步报。相对周期提前两步仍晚于本次风险。 | 因果 | 真 | 低 |
| G2 | 防御启示 | 报警后已有可交付且保留效用的安全版本 | [asr-defense-toy-2026-10-10/r02/RESULTS.md][src-alarm]、[selection-2026-11-23/README.md][src-select] | r02 五种策略均无满足安全与效用的可回滚候选；另一份候选菜单没有与这些首次报警配对。 | 对照 | 真 | 低 |
| G3 | 防御启示 | JSD 的告警已验证清除毒行的防御收益 | [asr-jump-sparse-defense-2026-10-12/RESULTS.md][src-jsd] | 开发组已实跑五级 JSD、ABL 等对照及端点扫描加重训；没有一臂同时通过风险与质量门，JSD 未筛中毒行，预留测试种子 1103/1104 未运行。 | 对照 | 真 | 低 |
| G4 | 防御启示 | Curfew 的收益由短窗规则单独产生 | [curfew-contribution-reassessment-2026-10-17/REPORT.zh-CN.md][src-curfew] | 与适配的 LT20/50 比较时，启动与恢复规则一同变动；少删干净行的案例中仍有答对 112 对 115 的逆例。 | 对照 | 真 | 低 |
| G5 | 防御启示 | 干预 K29 可持续阻断后门 | [k29-confirm-heads-2026-10-20/results-r03/REPORT.zh-CN.md][src-k29]、[k29-confirm-heads-2026-10-20/results-r03/ENDPOINTS.csv][src-kend] | 种子 2101 的 K29_zero 最终 V60 仍为 60/60；过渡期曲线面积下降与终点阻断是两个终点。 | 因果 | 真 | 低 |
| G6 | 防御启示 | 复杂报警聚合比简单速率规则更省成本且改变处置 | [asr-defense-toy-2026-10-10/third-line-toy/RESULTS.md][src-sensor]、[asr-defense-toy-2026-10-10/r02/RESULTS.md][src-alarm] | 已比的处置相同；相对简单正分差规则还多六次无用确认。影子策略的估算成本与配对实验总实耗口径不同。 | 对照 | 真 | 低 |
| H1 | 写作与复现 | 图 1/2 横轴每一单位都代表成功的参数更新 | [asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms]、[asr-rising-integration-2026-11-07/build_and_check.py][src-plot] | 训练存在跳步；图题的 Training update 容易被读作成功更新，试图更新的次数与成功次数已有记录。 | 表述 | 表述 | 低 |
| H2 | 写作与复现 | M1 的删除实验说明该集合「非必要」 | [attribution-microscope/supplement/jump/m1/RESULTS_M1.md][src-m1]、[attribution-microscope/supplement/jump/m1/CONTRACT_M1.md][src-m1c] | 删除后 D 的剩余量为 0.023；旧标题与读数及其判据不符，还容易被读成真实生成端点。 | 表述 | 表述 | 低 |
| H3 | 写作与复现 | Atlas 表中的 clean/trig 与所有触发条件同义 | [atlas-syntax-transfer-2026-09-21/instrument_audit.md][src-atlas]、[asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross] | T-5 的被比较输入实际是 clean/texttrig；简写省掉了决定这行含义的模态。 | 表述 | 表述 | 低 |
| H4 | 写作与复现 | 整合稿已经包含更新移植结果和自然路径预测反例 | [asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms]、[asr-effective-update-2026-11-16/REPORT.zh-CN.md][src-m31]、[asr-xo-analysis-2026-11-16/REPORT.zh-CN.md][src-m35]、[asr-ordering-2026-11-16/REPORT.zh-CN.md][src-m45]、[asr-history-response-2026-11-18/REPORT.zh-CN.md][src-m58] | 整合稿写到 P2/GATE；后续正结果、均值对照、局部预测失败与自然路径反例分散在各自报告。 | 表述 | 表述 | 中 |
| H5 | 写作与复现 | M055 的 NOT_REPLICATED 表示四个新种子都反驳结果 | [asr-seeds-2026-11-17/REPORT.zh-CN.md][src-m55] | 总标签合并了两个确认、两个未判定的情况；读者仅看标签会把未判定当反例。 | 表述 | 表述 | 低 |
| H6 | 写作与复现 | 仓库中的日志和校验和足以让第三方原样重放全部结果 | [asr-effective-update-2026-11-15/RESOURCE_PLAN.json][src-assets]、[asr-effective-update-2026-11-16/REPORT.zh-CN.md][src-m31]、[asr-rising-integration-2026-11-07/VALIDATION.json][src-valid] | 现有运行回执和哈希可审计；资源计划给出的 CPU 供体内存估算为 24.125 GiB，尚无从独立环境获取全部资产并完成全链重放的回执。 | 可复现 | 真 | 中 |
| H7 | 写作与复现 | Vim 早段可无缝拼接成五条相同身份的完整复现 | [asr-rising-integration-2026-11-07/01_跨模型证据.md][src-cross] | 前缀身份核验未能支持全部拼接；独立 0–120 前缀证据为 3/5。拼接会改写分母和轨迹身份。 | 可复现 | 真 | 中 |
| H8 | 写作与复现 | P2 在配置终点收敛且终态 D 极差在 0.5 内 | [asr-rising-integration-2026-11-07/03_正文整合稿.md][src-ms]、[asr-rising-review-icml2026-2026-11-07/COORDINATOR_SOURCE_CHECKS.md][src-repair] | 实测停在 700 个尝试步，配置写到 1250；原件复核终态 D 极差为 0.515625，超出旧摘要的 0.5。 | 表述 | 表述 | 低 |
| H9 | 写作与复现 | 现有相关工作核查已证实逐题更新交叉的新颖性 | [asr-rising-integration-2026-11-07/04_相关工作核查.md][src-prior]、[asr-xo-analysis-2026-11-16/REPORT.zh-CN.md][src-m35] | 已有定向查核围绕 P2 与当时整合稿；缺少按供体已知、接收起点替换、真实逐题翻转、均值对照逐项对齐的近邻对照。 | 对照 | 真 | 中 |

共 56 项，含真断点 47 项、表述断点 9 项。按环节 A–H 分别为 4、7、8、11、4、7、6、9 项；按类型为对照 18、因果 11、泛化 9、表述 9、机制 5、可复现 4 项；按严重度为高 2、中 29、低 25 项。

## 2. 处置决策表

**补**指补入现成证据，或在坚持更强声明时补指定检验；**降**指把声明缩到实测范围；**切**指移出本文主线，交由独立问题处理；**弃**指撤回具体推断并保留原始反例。共补 17、降 24、切 7、弃 8 项。

| 编号 | 处置 | 理由及落实方式 |
|---|---|---|
| A1 | 降 | 限定为所测监督微调、触发与目标配对。 |
| A2 | 切 | 白盒实验足以检验条件反事实，黑盒部署属另一问题。 |
| A3 | 降 | 将结果写成指定目标行为，保留任务内 ASR。 |
| A4 | 补 | 在投稿版复现入口指向实际 manifest，保留旧配置的历史身份。 |
| B1 | 补 | 主表并列定义、单位与分母，保留既有敏感性结果。 |
| B2 | 降 | Falcon 只记所测行为和不可判原因。 |
| B3 | 补 | 逐组列出选窗集、确认集及固定点对，报告未确认分母。 |
| B4 | 降 | 按实测上界、插值或删失分别报告。 |
| B5 | 降 | 行为结论用真实生成；连续量另列并给出错配。 |
| B6 | 补 | 导出表头明确区分观察、重放和新题确认。 |
| B7 | 补 | 每个结果同时列训练来源、题数、测点与交叉条件。 |
| C1 | 切 | 保留仪器审计，主线不靠热区亮度归因。 |
| C2 | 降 | 称为一个通过组级功能检验的集合。 |
| C3 | 降 | 结论限定到预选层、评测对、位置组与判界。 |
| C4 | 弃 | 撤去基于失准 T 账本的成因归属，保留测量失败。 |
| C5 | 降 | 注明统一仪器、旧资格失败及协议修订。 |
| C6 | 补 | 现稿列出完整事后基线；若要声称确认性优势，需在未见训练来源上事前冻结同一比较。 |
| C7 | 降 | 只归因于所移植的整段实际更新。 |
| C8 | 补 | 展示翻转身份、漏报、虚报及输给简单基线的格。 |
| D1 | 降 | 只认定指定输入历史替换的总效应。 |
| D2 | 降 | 保留输入历史总效应与整包交叉，撤去单一参数、优化器或学习率通路归属。 |
| D3 | 降 | 限于所选联合处理与背景；不给 P2 借用窄化方向。 |
| D4 | 降 | 保留指定已学习状态的促进效应。 |
| D5 | 降 | 表述为联合状态的条件作用与固定梯度下的传播约束。 |
| D6 | 降 | 用接收起点 g 加本来源供体逐题响应预测指定交叉的生成得失，保留 g 的预测角色。 |
| D7 | 补 | 仅在自然窗宽为论文中心承诺时补：先用开发数据冻结两个可分辨预测，再用独立来源的生成窗宽裁决；规则不可分辨时不启动。 |
| D8 | 弃 | 撤回指定十步窗内平均 T 响应被放大的解释，保留起点位置差、D/C 局部差与逐题误差。 |
| D9 | 切 | 组级 D 结果可作边界；行为选择性由组件线单独回答。 |
| D10 | 降 | 因果作用限定到实际移除的批。 |
| D11 | 弃 | 报告输出读数并存与回落事实，删除能力保存/消失及中介百分比推断。 |
| E1 | 切 | 把检测与行为控制拆成两个命题，组件论文才需在同一候选上闭合。 |
| E2 | 降 | 只写被修补位置组对读数的条件贡献。 |
| E3 | 弃 | 保留筛查失败，撤去用该分数判定毒行身份的主张。 |
| E4 | 补 | 同时写出测量值和门槛，用「未达到预设转移门」描述。 |
| F1 | 降 | 保留行为时差的跨种子支持；持续指 20 步网格连续三点达标。原 P2 全程分差终点、窗宽差和任意历史仍各自待证。 |
| F2 | 降 | 称为在另一训练来源重验同一预测关系。 |
| F3 | 降 | 现象按模型报告，给定更新的机制结论限定在实际交叉过的模型。 |
| F4 | 降 | 只列逐组实测覆盖，不把异构轨迹数当作该机制的单因素确认。 |
| F5 | 降 | 副作用按用户选定的同任务干净正确率验收；后门特异性保留为开放问题。 |
| F6 | 降 | 分别给出移除效应、时序判定和换序结果，逐项报告分母。 |
| F7 | 降 | 并列每条轨迹的攻击与干净效应；单头结论限于探索性证据。 |
| G1 | 弃 | 保留相对检查时差，撤下事前预警声明。 |
| G2 | 切 | 回滚作为单独的交付决策问题，不能借 ASR 时差声称收益。 |
| G3 | 弃 | 发布已跑的筛查结果及停止点，撤去净化收益结论。 |
| G4 | 切 | 联合策略评估留给防御线，主论文不分摊单条规则贡献。 |
| G5 | 弃 | 保留早期曲线效应与终点反例。 |
| G6 | 切 | 另题对同一作业、相同预算比较净收益；现稿只报告实测动作。 |
| H1 | 补 | 图轴写 attempted update，在图注标出跳步及计数口径。 |
| H2 | 补 | 引用保留与删除两个实测量、判界及 D 的名称。 |
| H3 | 补 | Atlas 行恢复实际输入名，并保留仪器审计入口。 |
| H4 | 补 | 让正文同一论证同时引用干预、事后均值、失准格与自然训练反例。 |
| H5 | 补 | 主表直接列各种子与未判原因，历史机读记录保留原值。 |
| H6 | 补 | 发布分项资产索引，区分 Git 内可重算、需外部资产、已独立重放。 |
| H7 | 弃 | 分别列有效前缀与各自轨迹的观测，撤去无凭据的拼接。 |
| H8 | 补 | 按 700 步冻结终点与精确极差引用，不据此宣称训练至配置终点。 |
| H9 | 补 | 投稿前补针对该项最小贡献的原文比较，保留最接近工作的相同与新增部分。 |

## 3. 不做什么

- 不把短窗出现的模型数当作同一个机制的验证次数，不把预选历史的阈值时差写成窗宽差。（B1、F1–F4）
- 不把首词分差、模块恢复量或检测分数替换为真实生成 ASR，也不从干净正确率推断后门特异性。（B5、D9、E1–E4、F5）
- 不重复计算已完成的均值对照，不用总体 ASR 掩盖逐题预测输给均值和零更新的格。若只主张已测交叉内的预测增量，保留事后比较的身份。（C6、C8、H4）
- 不用已知供体端点反推自然训练为何在该时刻生成这段更新，不把单一分差、优化器矩或一组模块写成唯一通路。（C2、C7、D2、D5–D8）
- 不将报警提前、风险曲线面积下降或筛查 AUROC 换算成部署收益；防御留有失败记录与干净效用对照。（E3、F7、G1–G6）
- 不拼接身份未核准的轨迹，不把未判定种子计为反例，不把逐题读数当成训练重复。（B7、F6、H5–H7）

## 4. 论文四句话

1. **研究问题**　后门微调中，为什么许多题会在很短的训练区间一起转向目标答案，又会在部分区间一起回落？
2. **核心反事实陈述**　在受控的 A-C 轨迹上，同一段真实五步权重更新放在第 320 步起点成功题数净增 1 道，移到第 330 步起点则净增 48 道，交叉前按接收起点和供体逐题响应预测净增 51 道。[原始交叉报告][src-m31]
3. **最小贡献点**　在两个训练来源内分别测量供体响应，再冻结逐题预测，分别捕捉到 319/359 和 103/113 次真实生成翻转；与事后计算的供体均值相比，逐题命中错误分别从 149 降至 86、从 49 降至 19。[原组交叉][src-m31]、[另一训练来源与均值对照][src-m35]
4. **最可能被拒理由**　预测时已经知道供体更新及其端点响应，简单均值也保留了另一训练来源四格中的全部相对排序，而事前预测一条自然训练路径的两臂差曾给出 +70.667 个百分点、实测为 −2.583，因此现有结果支撑指定交叉的行为解释，尚不足以解释自然跃升的时刻与窗宽。[均值对照][src-m35]、[自然路径预测][src-m45]

### 主线、D7 与两周决策锁定（2026-11-23）

**1．主线与贡献句。** H 指训练历史，W 指真实生成 ASR 从 10% 首次升至 90% 的观测窗宽。主线暂定为现象陈述，A 历史内指定处理与 W 缩窄的关联能否独立复现按第四段判定，机制 P 未识别。当前贡献句为「在已测受控设定中，指定训练处理改变了首次观测窗宽。该效应在 A 历史中成立；在 B 历史中未观察到可辨认的窗宽变化。效应是否依赖历史，需更多历史对验证。」[四臂结果][src-m13]

**2．D7 状态：暂缓。** 仅当摘要明确需要事前预测未见轨迹的窗宽差，且按原协议核实一对合格、后续结果未披露的 A/B 轨迹并登记种子、运行 ID、输入路径和哈希，才重启。重启后，两项冻结预测相距不超过 40 个尝试步即停止，不揭盲。[冻结协议](https://github.com/yuxiexiong/stealthiness/blob/exp-prereg-v1/experiments/decision-D7.md)

**3．候选 P 与可判性。** P₁ 为跃升前各题到目标决策边界的距离分布。所需干预是在同一 A 历史下改变模型起点中这些距离的离散程度，同时匹配中心位置、后续输入并核查实际更新响应是否可比；竞争解释是窗宽变化来自后续推进速度。P₂ 为跃升期落后题相对领先题的追赶速度（按起点分差固定题组）。所需干预是在同一起点、平均推进量匹配时，改变实际更新对这两组题的相对作用；竞争解释是起点距离分布已足以造成集中越界。两者均指模型训练状态或实际更新；只改读出分数无法检验这两个过程。**目前两者均写不出有依据、互不重叠的真实 W 预测区间**：尚无隔离上述作用的匹配对照。可判性单列讨论，不为候选编造区间，不据此开跑。[四臂分析][src-m13]、[已有更新交叉的边界][src-m31]

**4．两周分支。** 截至北京时间 2026 年 12 月 6 日 23:59。① 找到合格的直接成因对照：开跑前锁定 P、干预、竞争解释、互不重叠的真实 W 预测区间及同任务干净正确率的等效界；仅当干预按预定方向改变 P、实测 W 落入 P 独有的预测区间且变化方向符合预期、干净正确率变化的置信区间完全位于等效界内，才接受机制贡献。② 找不到：冻结新增实验，按既有局部因果成果组织论文，贡献句降为「在已测轨迹中，指定训练处理可改变真实生成的跃升窗宽，同一段已发生的权重更新在不同起点可产生不同的逐题行为后果；自然窗宽的形成过程仍待解释。」③ 独立复现门适用于前两支：若截止时仍未完成至少一次同一 A 历史、同一处理对照下方向一致、且按原 W 采样口径可辨的独立复现（不同种子或独立重训的运行 ID，重复读取同一轨迹不计），窗宽主线降为「单次观察到的关联」，贡献句再降为「在一组共享 A 前史的保留／替换配对中，指定处理与较窄的首次观测窗宽相伴；这一观察尚未得到独立复现。」独立尝试若方向相反或变化不可辨认，仍用第③支并如实报告结果。后续变更须书面列明原句、替代句及依据。

### 下一阶段论文主线与最高优先级（2026-11-24 确认）

1. **论文主问题。** 为什么同一种训练处理会在某些训练历史中使 ASR 跃升更集中？能否在跃升发生前判断，哪些历史会出现更大的窗宽压缩？每个历史的处理效应用替换臂的真实生成窗宽减去保留臂的窗宽表示，数值越负，压缩越大。
2. **最高优先级。** 检验处理前平均分差相近时，逐题目标分差更集中的历史是否会出现更大的窗宽压缩。这里的分差指触发输入上目标首 token 与最强竞争 token 的 logit 之差。独立检验命中支持预测贡献，因果机制仍需单独识别。
3. **现有证据。** 在已知 A/B 中，A 的处理前逐题分差更集中，平均分差也更接近零，处理后窗宽缩窄；B 的窗宽变化方向未分辨。现有比较无法分开均值与集中度各自的预测作用。候选条件即使被推翻，论文主问题仍然成立。D7 暂缓，同一 A 历史的独立复现继续按前节原口径计。[M013 四臂报告（Raw）](https://raw.githubusercontent.com/yuxiexiong/stealthiness/1ac22b599e6bfd81ccd5b5622e206b3a16401387/research/asr-exposure-fourarm-analysis-2026-11-15/REPORT.zh-CN.md)

#### 同日资产盘点

| 资产与定位 | 实物、网格与来源 | 对本问题可用到哪一步 |
|---|---|---|
| M013 开发四臂 | H20 `/data/asr-exposure-fourarm-20261113-p0/runs/` 有 A/B 各自的 keep、replace。工作区 `research/asr-exposure-fourarm-2026-11-14/v3_complete_receipts/` 含 `RUN_MANIFEST.json` 和 `readouts/`。200 个唯一题号；各历史两臂共用 280、290、300 步，随后 305–400 每 5 步，另有 420、440。 | 处理前分差、真实生成窗宽、同任务干净题和分析口径可用于开发核算。处理效应已公开，独立性按每段训练历史计。 |
| M055 的 S2001–S2004 | H20 `/workspace/asr-seeds-20261117-r01/runs/` 下四个种子的 `checkpoint-300/` 均查到封存清单、权重、优化器、RNG、游标、scheduler 和 scaler。Git 中 `research/asr-seeds-2026-11-17/phase1-evidence/reads/gap/` 的四份 `S…@300.json` 各有与 M013 相同的 200 个题号。原合同固定 B 的数据顺序，改变训练种子。 | 四个可复用起点。自然轨迹的逐题分差曾用于旧分析；基础轨迹只有原配日程。`asr-seedconfirm-20261117-r01` 的 64 个分支用于插入与抽批等另一个对照。未定位每个种子按 M013 同处理配对且覆盖真实生成窗宽的数据。四个源的均值是否可比、离散度是否可分尚未判定。 |
| M057 的 O3001/O3002 | H20 `/workspace/asr-orders-20261118-r01/runs/` 下两份 `checkpoint-300/` 有封存和续训文件；Git 的 `research/asr-orders-2026-11-18/evidence/reads/gap/` 有两份第 300 步分差。训练种子均为 1007，改变数据顺序。 | 可复用作不同顺序来源；原合同比较自然轨迹代理起点，已有公开报告。未定位同处理两臂及其真实生成窗宽。五条处理样本在新顺序中的可比性亦未核定。 |
| M058、状态重组、已知更新移植及 D7 | `asr-history-20261118-r02/runs/` 有 KK、KR、RK、RR 的 28 个后续分支；`asr-state-cross320-20261115-r02` 起于既有四臂。M031/M035 的既有实验为指定更新移植；`experiments/decision-D7.md` 对自然续训宽差作预测。 | M058 的四状态承接原处理后的第 320 步；这些分支不增加独立的处理前历史。移植与 D7 各有不同的预测对象，均未提供本问题所缺配对。 |

**盘点结论。** 核实一套开发四臂、四个同顺序新种子起点和两个新顺序起点；截至本次盘点，能直接用于独立裁决的完整四臂组为 0 套。具体缺口是独立历史中可与前测一一配对的、同一处理的保留／替换真实生成窗宽，以及均值可比、离散度可区分的前测依据。已披露自然轨迹与尚未取得的处理效应分别登记。盘点仅核文件与来源、检查点组成、前测题号和披露记录，未重放检查点、未重验全部权重哈希、未读取候选轨迹的后续行为。范围为工作区、远端 main 的固定提交、Git 分支名及 H20 上项目运行目录（`/data`、`/workspace`、`/workspace/claude-jump` 下识别出的 ASR 运行），不覆盖未登记的异名目录和离线备份。

复核入口：[M055 训练与采样合同（Raw）](https://raw.githubusercontent.com/yuxiexiong/stealthiness/1ac22b599e6bfd81ccd5b5622e206b3a16401387/research/asr-seeds-2026-11-17/CONTRACT.zh-CN.md) · [M057 顺序比较合同（Raw）](https://raw.githubusercontent.com/yuxiexiong/stealthiness/1ac22b599e6bfd81ccd5b5622e206b3a16401387/research/asr-orders-2026-11-18/CONTRACT_COMPARE.zh-CN.md) · [M058 四状态来源（Raw）](https://raw.githubusercontent.com/yuxiexiong/stealthiness/1ac22b599e6bfd81ccd5b5622e206b3a16401387/research/asr-history-response-2026-11-18/CONTRACT.zh-CN.md)。

[src-alarm]: https://github.com/yuxiexiong/stealthiness/blob/e7bc5b046ea5744651a61619220812070b0d1a6d/research/asr-defense-toy-2026-10-10/r02/RESULTS.md
[src-assets]: ../research/asr-effective-update-2026-11-15/RESOURCE_PLAN.json
[src-atlas]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/atlas-syntax-transfer-2026-09-21/instrument_audit.md
[src-attr]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/attribution-microscope/REPORT_FINAL.md
[src-cfg]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/attribution-microscope/configs/protocol.yaml
[src-core]: ../research/asr-core-argument-2026-11-16/CORE_ARGUMENT.zh-CN.md
[src-cross]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-rising-integration-2026-11-07/01_%E8%B7%A8%E6%A8%A1%E5%9E%8B%E8%AF%81%E6%8D%AE.md
[src-curfew]: https://github.com/yuxiexiong/stealthiness/blob/24942632e1522df3333d68547bfe8c7200786fda/research/curfew-contribution-reassessment-2026-10-17/REPORT.zh-CN.md
[src-events]: ../research/asr-experiment-chronicle-2026-11-11/条目台账.csv
[src-flat]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/gap2-initial-position-2026-11-20/FLAT-SHIFT.zh-CN.md
[src-inv]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-mainline-inventory-2026-11-18/INVENTORY.zh-CN.md
[src-ip]: https://github.com/yuxiexiong/stealthiness/blob/e7bc5b046ea5744651a61619220812070b0d1a6d/attribution-microscope/supplement/jump/ip/RESULTS_IP.md
[src-jsd]: https://github.com/yuxiexiong/stealthiness/blob/24942632e1522df3333d68547bfe8c7200786fda/research/asr-jump-sparse-defense-2026-10-12/RESULTS.md
[src-k29]: https://github.com/yuxiexiong/stealthiness/blob/24942632e1522df3333d68547bfe8c7200786fda/research/k29-confirm-heads-2026-10-20/results-r03/REPORT.zh-CN.md
[src-kend]: https://github.com/yuxiexiong/stealthiness/blob/24942632e1522df3333d68547bfe8c7200786fda/research/k29-confirm-heads-2026-10-20/results-r03/ENDPOINTS.csv
[src-ledger]: ../research/research-mentor/reviews/M039/EXPLANATION_LEDGER.zh-CN.md
[src-m1]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/attribution-microscope/supplement/jump/m1/RESULTS_M1.md
[src-m13]: ../research/asr-exposure-fourarm-analysis-2026-11-15/REPORT.zh-CN.md
[src-m1c]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/attribution-microscope/supplement/jump/m1/CONTRACT_M1.md
[src-m20]: ../research/asr-mr320-analysis-2026-11-15/REPORT.zh-CN.md
[src-m23]: ../research/asr-gradient-replay-analysis-2026-11-15/REPORT.zh-CN.md
[src-m26]: ../research/asr-effective-update-2026-11-15/REPORT.zh-CN.md
[src-m31]: ../research/asr-effective-update-2026-11-16/REPORT.zh-CN.md
[src-m35]: ../research/asr-xo-analysis-2026-11-16/REPORT.zh-CN.md
[src-m45]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-ordering-2026-11-16/REPORT.zh-CN.md
[src-m55]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-seeds-2026-11-17/REPORT.zh-CN.md
[src-m58]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-history-response-2026-11-18/REPORT.zh-CN.md
[src-mean]: ../research/asr-xo-analysis-2026-11-16/MEAN_BASELINE.json
[src-ms]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-rising-integration-2026-11-07/03_%E6%AD%A3%E6%96%87%E6%95%B4%E5%90%88%E7%A8%BF.md
[src-p2]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-rising-integration-2026-11-07/02_%E5%9B%A0%E6%9E%9C%E8%AF%81%E6%8D%AE%E4%B8%8E%E8%BE%B9%E7%95%8C.md
[src-plot]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-rising-integration-2026-11-07/build_and_check.py
[src-poison]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/attribution-microscope/src/poison.py
[src-prior]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-rising-integration-2026-11-07/04_%E7%9B%B8%E5%85%B3%E5%B7%A5%E4%BD%9C%E6%A0%B8%E6%9F%A5.md
[src-repair]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-rising-review-icml2026-2026-11-07/COORDINATOR_SOURCE_CHECKS.md
[src-select]: ../research/selection-2026-11-23/README.md
[src-sensor]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-defense-toy-2026-10-10/third-line-toy/RESULTS.md
[src-shortaa]: ../research/asr-short-fourcell-resume-2026-11-11/RESULT.zh-CN.md
[src-shortcross]: ../research/asr-short-fourcell-cross-2026-11-11/RESULT.zh-CN.md
[src-valid]: https://github.com/yuxiexiong/stealthiness/blob/874d4054cfa2dfaac4fda620d2241f99d763882b/research/asr-rising-integration-2026-11-07/VALIDATION.json
[src-xo]: ../research/asr-xo-confirmation-2026-11-16/REPORT.zh-CN.md
