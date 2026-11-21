# 独立科学验收记录

2026-11-21。**复审通过，签收当前《从原实验找回来的意外》与《早期实验独立复核》的科学表述。** 首审提出的计量、设计、先验与遗漏问题均已处理。尤其是两个首词读数与完整生成的区分、GAP2的同窗口常数、M053–M058整批操作对照，以及形成后报警的原门，已回到各自原合同和结果的含义。九月至十月初独立报告的两处归因措辞也已收窄。本签收覆盖已陈列的科学断言与相关原件，仓库材料覆盖深度由主审另行登记。

## 主报告的实质更正

| 对象 | 核对原件后的判断 | 应采用的表述 |
|---|---|---|
| 全文范围与独立性 | 早期报告覆盖9月至10月初。分类器、文生图代理、首词命中及完整生成的分母与事件不同。多行可复用同一母本。 | 用各行原合同定义命中。行数是比较主题数，不能累加为独立训练重复。 |
| E021 / E146 | 十条文本完整生成的终点结果与后来的SW首词检查是两种测量。1009的W0/W2首过90%为91/150、持续均为170；SW明确用首词对全词表的argmax。[SW原定义](evidence/ee03a2865c8f.md) | 分句报告。不要据SW数字声称完整生成在相同步数达标。 |
| E059 / E060 | 77步误差是无遗忘计数对1009预测167、实际90；窗前速度预测窗宽是同报告的另一项失败。原报告还发现打乱顺序的对照本身容易显著。[MRM原实验](https://github.com/yuxiexiong/stealthiness/blob/f1cd817a7afdb4da748b9b7b7e42944ae810817a/research/asr-mechanism-rivals-2026-10-15/RESULTS_MRM.md) | 将77步明确归给计数预测器。 |
| E069 / E071 | E069比较单一指定行的配对差，原竞争预期是早段冲刷、过门后保持，12次配对仅2次观察到半衰。E071为同组六个条件各用三法，共18臂。[E069读数](evidence/0be3d0191184.json) · [E071逐臂表](https://github.com/yuxiexiong/stealthiness/blob/24942632e1522df3333d68547bfe8c7200786fda/research/curfew-contribution-reassessment-2026-10-17/evidence/COMPARISON.csv) | 不能据E069推广到任意训练扰动；将E071的「18个匹配比较」改为「六个条件、18臂」。 |
| E093 / E113 / E129 | 单目标原合同接受形成后50步内且误报不超过1次。+17且0误报合格、+60且9误报失败。四目标另有事前押注0.75–0.85，两种子实测0.658、0.574，功效有限。DW8合同冻结的是能否复现的检验，尚未找到预测必复现的时间戳。[GRADTRIG原判](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/research/curfew-trigger-2026-10-21/eqcity/REPORT_GRADTRIG.zh-CN.md) · [K4原判](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/research/k4-retest-2026-10-23/RESULTS_K4.zh-CN.md) · [DW8执行前合同](https://github.com/yuxiexiong/stealthiness/blob/abde35400936d732a0c2f355c30977614367d56a/research/dose-width-2026-10-25/CONTRACT.zh-CN.md) | 形成前报警属于回顾直觉；四目标的区间确有直接反预测。DW8降为受控对照，除非能补出事前方向性押注。 |
| E121 / E132 / E136 / E139 | E=0.023是所选模块删除后原探针效应的剩余比。P1事前押晚窗，早段换入+55.0步、晚窗+10.4步，各操作在指定窗外另改变386及413个行位置。P2共同未来仍存在混合精度跳步，属于另一识别限制。[E纠错](evidence/33053ded03e1.md) · [P1五项押注](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/research/asr-rising-integration-2026-11-07/sources/p1/RESULTS_P1.zh-CN.md) · [位置核查](evidence/93c529cc4f45.md) | 明确2.3%的分母、晚窗反预测及窗外移位，不能把两种设计限制混写为数值溢出。 |
| XO / M053 / M054 | XO的113次翻转来自一个新训练来源1008、四格重复200题。M053十步1.525→1.224等是P与K整批操作在参考态及读点的响应差；M054的1.269→0.504与0.236→0.640同理，两个背景共享同一训练种子。[XO原分析](evidence/d5680dad1ddd.md) · [M053](evidence/6dc9cf522fb9.md) · [M054](evidence/61daf083d5b2.md) | 写清固定输入比较后的 (R_T,R_D)，把响应与同一臂状态水平分开。 |
| M058 / GAP2 | 1.130与1.282在冻结实用界内，三条P与三条K各自取均值；17.3对0.3是每200题平均净差。位置与之相容，现有对照没有识别解释份额。GAP2的4题源于逐题计算前冻结的规则，两项常数都取自已知结果的同一窗口。探索性重建的17题也来自同一窗口。[M058原比较](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/research/asr-history-response-2026-11-18/REPORT.zh-CN.md) · [第一规则](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/research/gap2-initial-position-2026-11-20/REPORT.zh-CN.md) · [第二规则](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/research/gap2-initial-position-2026-11-20/FLAT-SHIFT.zh-CN.md) | 禁用「三对独立重复」「由起点造成多少」及针对新窗口的预测措辞；同一状态内具体题未完全恢复，算术差可抵消。 |

## 应保留的遗漏和较弱限制

- M049是“越接近跃升越能保持”的直接受约束案例，主终点不可判；一个正向臂与参照末点同为98/200，实际增加6题、失去6题。M055新四个训练种子共享顺序，原保持主终点两例复现、两例不可判；另有阶段差分响应在四种子上复现。前者约束普通直觉，后者同时给出跨种子的支持和冻结门的限制。[M049](evidence/79a71c4b9bb3.md) · [M055](evidence/44018b9a913b.md)
- M056的D在两批已测组合的第19步接近相加，短间隔T的主比值仍不可判。M057以10步网格上至少5%题首词分数过零为起点。同一训练种子的两个新顺序均在330步达到，固定步数猜340。毒行计数规则误差21及56步，前一项仅越过±20步容差一格。作者原先对第二新顺序押两种规则都不符，结果落空。这是能检验“事前惊讶”的一项独立设计。[M056](evidence/52ca2cd06c65.md) · [M057合同](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/research/asr-orders-2026-11-18/CONTRACT_COMPARE.zh-CN.md) · [M057原结果](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/research/asr-orders-2026-11-18/REPORT_COMPARE.zh-CN.md)
- E112–116的随机位移可称等量级。随机扰动与清零造成的位移在不同时点比较，且只有一个随机方向，无法归出单个优化器分量的因果作用。[GA原报告](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/attribution-microscope/supplement/jump/g/RESULTS_GA.md)

## 早期报告与核验边界

视觉A的异事实供体对照出现16项保留接收图颜色、7项跟随供体、1项第三色。「恢复原图通路」属于尚未识别的解释。Atlas的93.5%采用梯度乘输入的带符号归因和。两处已在早期报告收窄。[视觉A](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/attribution-visualization/visual-evidence-repair/A_RESULTS_2026-09-13.md) · [Atlas原定义与R6](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/attribution-microscope/REPORT_FINAL.md)

逐行审阅了两份汇总，按争议回读原合同、结果报告及部分机读读数。早期的核查曾直接打开OA完成回执、视觉诊断与文生图评分、Qwen资格偏离记录和跨架构结果。文件目录、字节清点或链接可达只证明材料可见；未逐文件解释全部原数据，未重新判图或验证所有区间，更未重跑模型。保留Grond仅核材料、无模型结果，保留参照修正、不可判与技术失败。

## 末次追加七项与两条勘误

**追加范围签收通过。** 核对了G、SF、CL、ZG的固定版本原表及CANON的T12、P9、P11，另回读E045/E046、快照谱、尖锐度与缺窗报告。G反预测发生在约328步起飞前的两处拟合时点。SF的两条新填充臂与四条交叉臂已分别解释。CL两轮采样误差方向相反。ZG已说明两套时标均来自拟合、旧误差同时受端点口径和留出数据选参影响。T12只差0.31步仍穿过冻结边界。E045原始JSON给出的事后共同阈值为2.552048，只能检验同轨迹回代。快照谱补回两个短暂越界点，尖锐度的±0.003奈特为仪器分辨率。P9说明遮挡读数的构造限制，P11说明1250步预算内的右删失。[E045机读阈值与逐母本时点](https://github.com/yuxiexiong/stealthiness/blob/f1cd817a7afdb4da748b9b7b7e42944ae810817a/research/asr-mechanism-aprime-2026-10-14/results/aprime_result.json) · [CANON正式更正](https://github.com/yuxiexiong/stealthiness/blob/e171f6f2b029ecaa3bd0527620a9b8b6dc3ea4e3/research/jump-canon-2026-10-23/CANON.zh-CN.md)
