# Reviewer 4：独立模拟评审

这是作者侧 AI 模拟反馈，不是正式 ICLR 评审或录用概率预测。依据 [ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)，重点判断问题、动机与定位、证据支持、社区价值；不以新算法或 SOTA 为必要条件。阅读期间未看其他 reviewer 报告。

## Summary

本文发现 BackdoorBench 的三种攻击继承了不会在训练结束前触发的学习率衰减计划，使部分防御的干净准确率“增益”受低起点驱动。作者通过调度器交叉实验、同配方 clean twin、下游论文审计及 TARE-Z 校准，主张额外报告防御对 clean twin 的影响，同时明确不把残差识别为纯粹的后门去除代价。

## Strengths

1. **主要经验发现可信且可操作。** p6 Table 2 的双向交换只改调度器；我从提交数据复算出四组 pooled tare 为 −1.8418、+0.7839、+0.5033、−2.0797 点，132 个 seed×defense 单元中 131 个符号符合描述。BPP、CIFAR-100、VGG19-BN 和第二工具箱的扩展，使它超出单一坏配置的举例（pp6–7、S31–S33）。
2. **对因果解释边界的处理值得肯定。** §3.1、§4.6 与 S5 区分部署者实际损失、clean-arm 影响和 poisoning×defense interaction；两种降低起点的对照直接说明纯 removal share 不可识别。没有靠不显著结果宣称零代价。
3. **失败和实现偏差披露充分。** 作者报告安装/去除失败、ABL 崩溃、ANP 的 oracle 阈值、未复现的排名和自托管注册的限制。Table 1 的八行均值、分解恒等式均可由附件核对；这显著提高测量审计的可检查性。

## Weaknesses

1. **交付的解释与操作规则尚未完全统一，会影响使用者得出什么结论。** p4 明确 interaction 不等于 removal，p9 说不纠正；但 pp18–19 S3/Table 10 仍将 DiD 排名标作“removal”和“corrected”。p23 Table 15 写随机标签退化臂的 `τ_int≡0`，同表却报告非零 interaction；这里应为没有后门可去除，而非 Eq.1 的交互对比为零。p24 Algorithm 1 的成功事件仅检查 ASR，未包含 pp4、18–19 明确要求的 utility floor。问题并非措辞优美与否：按所给算法或列名使用，会把模型毁坏或未识别残差当作有效去除结果。
2. **“never-poisoned”的量化范围仍需收紧。** §3.1 的主网格实际保留 1–10 张投毒图，低 ASR 只能支持操作性 admissibility，不能保证所有防御面对它与零投毒完全相同。S34/p56 的零图检验很有价值且支持主要调度器反转，但仅覆盖 FT/FP/NAD 的三个配置；其中 BadNets FT 为 +0.074，95% t 区间 [−0.047,+0.195]，不能单独确认摘要中最小的 +0.13 点“收费”。这不推翻较大的反转，但摘要“七个都收费”及精确份额需要始终明确是低剂量 twin 的结果，而非所有真正零投毒模型的结论。
3. **下游 CGD 因果案例的实现对应关系未完全闭合。** pp78–80 S53–S54 把 CGD 重跑概括为 CLIP 分离后对保留子集微调；[CGD 原文 §4.2](https://arxiv.org/html/2507.05113v1) 描述 relearning、unlearning、distillation 三项损失。附件 `sched8.sh` 调用了外部 `defense/cgd.py`，但未包含该文件或固定其版本；`cgd_step1.py`、SCHED7 注册和 LOCAL_CHANGES 说明不足以确认第二阶段对应关系。这目前是可复核性缺口，**不是已证明少执行阶段**。因为“两个下游方法”的因果证据是主文贡献，该对应关系值得明确。

## Questions

1. 哪个估计量和成功门槛是最终推荐协议？能否使 Algorithm 1、附录排名列名、数据导出字段与主文不可识别声明一致？
2. 对真正零投毒模型，哪些结论只要求方向复现，哪些要求在 0.1 点尺度准确？若不扩展零图实验，是否将相应主张严格限定在所测低剂量 twin？
3. CGD 第二阶段实际执行的代码版本、损失项及超参数是什么？请提供固定源码/差异，并说明 10% 重跑如何支撑对原文 5% 案例的定性归因。38%/66% 是何种支持集上的比值，而非可识别的 removal 份额？

## 四维判断

| ICLR 2027 核心问题 | 判断 |
|---|---|
| 具体问题 | 清楚：训练起点如何混入防御成本，以及缺少 clean-arm 对照造成什么解释缺口。 |
| 动机与文献定位 | 基本充分。已有 clean-arm 工作被承认；新增价值主要是原因定位与实证追踪，而非四格算式本身。 |
| 正确性与严谨性 | 主调度器结果较强；对照边界和主附录语义须统一，CGD 版本对应待补。 |
| 新知识与实际意义 | 达到接收线附近的社区价值：可修复的 benchmark 测量问题。影响主要是归因和跨攻击解读；不能据此认定部署者实际增益无效，或宣称普遍改变防御排名。 |

清晰度：主线可理解，但附录残留解释会误导。可复现性：提交统计量易核查，完整训练复现较弱，尤其外部新增防御未固定版本。

## 建议、信心与跨线条件

**独立定性建议已先锁定：倾向接收，略高于接收线。** 最强接收理由是可操纵并扩展验证的调度器发现，加上诚实的可识别性边界；最强拒绝理由是测量论文自身交付的估计量与协议标签仍不一致，且下游 CGD 案例缺少完整版本证据。我认为现有核心发现仍足以超过这些可修复问题。

锁定后给出的**临时数字表达为 6**，仅借用[已核实的 2026 official review](https://openreview.net/forum?id=ICANwnoGgN&noteId=p0VR9cKoYg)中的“略高接收线”锚点；**不是已确认的 2027 官方量表**。信心：较有把握，临时数字 **3**（同样仅借用 2026 锚点）；我未重跑训练，也未独立重读全部 48 篇引用文献及验证全部实现。

不需要为跨线另做新算法。统一协议和附录、固定 CGD 实现、限制低剂量 twin 的小额主张，会使接收建议更稳固。相反，若 CGD 实际未对应原方法且仍坚持其因果外推，或继续把未识别 DiD 用作 removal 排名，我会转为略低接收线。

## 实际核查范围

- 完整阅读 PDF pp1–10；针对问题检查附录 pp15、17–19、22–25、37–40、55–57、72、77–80 的相关段落与表格；查看 p6/p8 的 PDF 渲染。
- 检查 `supplement/README.md`、`supplement/sched/README.md`、`LOCAL_CHANGES.md`、`citation_audit/AUDIT.md`、`scripts/sched7.sh`、`sched8.sh`、`cgd_step1.py`、`prereg/en/SCHED7_PREREG.en.md`。
- 只读轻量重算 `data/backdoorbench_11seed_cells.csv`、`backdoorbench_wanet_11seed_cells.csv`、`scheduler_crossover_cells.json`、`sched/data/sched7_cells.json`、`sched20_cells.json`。复现 Table 1 均值（恒等式最大浮点残差约 7.1×10⁻¹⁵）、Table 2 pooled 符号、零图 FT 区间和 38%/66% 算术比值；未执行整个 rebuild 或训练脚本。
- 在线核对 [ICLR 2027 标准](https://iclr.cc/Conferences/2027/ReviewerGuidelines)、[TSBD §4.2](https://arxiv.org/pdf/2405.20291)、[CGD §4.2/§5.2.1](https://arxiv.org/html/2507.05113v1) 和 [CGD 官方仓库说明](https://github.com/binyxu/CGD)。官方仓库第二阶段源码抓取未成功，因此保留对应关系的不确定性。未调用 GPU、未修改提交材料、未访问历史 TARE 材料。

