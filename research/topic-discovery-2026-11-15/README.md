# 后门形成：保留课题与具体剩余问题

**最新实施准备（2026-11-15）：** [D05八轨迹固定划分×污染方案](d05-toy/partition-pilot-2026-11-15/PLAN.zh-CN.md)的代码、CPU检查与独立Advisor实现审查已通过。GPU启动被自动审批在执行前拒绝，当前没有新训练或污染结果；[当前状态及具体算力范围](d05-toy/partition-pilot-2026-11-15/STATUS.zh-CN.md)。下面340条正常资格结果仍有效。

**最新执行结果（2026-11-15）：** 用户授权的[D05固定340条正常资格](d05-toy/qualification-2026-11-15/REPORT.zh-CN.md)已完成：W168/168、P168/168、无备注4/4，21词和四模板均全对，原门槛通过。未训练、未污染。首次框架生成配置合并造成工程失败，修复经CPU与Advisor复核后完成同一资格；两次占卡合计83.00秒，失败保留。代码、逐行输出、独立复算与审查均在同目录。下文静态准备/NOT_RUN是历史状态，不代表当前仍未测。

2026-11-15。文档统一发布到 `main`。本目录包含研究讨论、文献核查、独立审查及D05实现/执行记录。用户最新要求选定D05现有模型、准备真实编码表和正常资格检查方案；仅静态准备，不启动模型实验。此前执行授权与结果保留为历史。

## 当前先读

**D05实际编码准备（最新）：** [Qwen3-4B、21词真实编码与340条正常资格方案](d05-toy/encoding-preparation-2026-11-15/PLAN.zh-CN.md)。固定官方revision，CPU tokenizer已验证完整提示中的W/P字节一致与共享ID，附全输入、SHA和Advisor。无备注重复项已去重，双EOS/截断规则已冻结。正常能力资格仍NOT_RUN；没有加载权重、前向或训练。

**D05简化设计讨论（前轮）：** [借已有正常能力、单一字段任务、现有词表切分的四臂方案](d05-toy/simpler-design-2026-11-15/DESIGN.zh-CN.md)。用户要求不因旧toy失败放下问题；Advisor支持继续该简化设计，取消额外识字课，保留编码×污染行为总作用与更强范围解释的区别。设计时模型与词表未核；最新静态准备见上。旧toy关闭不变，模型计算暂停。

**D05两次失败后的前轮判断：** [通俗复盘、问题价值与设计失误](d05-toy/retrospective-2026-11-15/REPORT.zh-CN.md)。原混合训练与固定总量调度均0/5资格，均未进入污染。旧toy关闭，曾建议D05-C仅登记保留并撤回快速toy首选地位。原主任务仅2/5、新0/5达到95%，不能只归咎辅助门槛。这是前轮判断，最新授权与设计进展见页首。

**D05已执行并按资格停止：** [完整结果、核查与成本](d05-toy/execution-2026-11-15/RUN_REPORT.zh-CN.md)。五个干净前缀全部完成，0/5达到预定资格，未启动污染分支；不是D05主假说的阴性结果。实际单卡保留150.64秒，原始材料保留。下面“尚未启动”的设计阶段记录保留为历史。

**D05实现准备：** [代码、CPU检查、完整资源预估与运行边界](d05-toy/README.zh-CN.md)。数据和指标实现已准备，未运行模型；Advisor指出的失格块推动效应门槛问题已作实验前修正，主均值仍保留全部块。当前状态不等于运行验收。

**D05 toy设计已完成，尚未启动：** [任务、对照、结果与去留条件](D05_TOY_DESIGN.zh-CN.md)。采用共同双编码识字前缀，整名/部件×无污染/两个固定污染量，配对读入与正常能力检查。独立Advisor及聚焦复审意见已吸收；正文附全部审查记录。设计阶段的历史记录保留；最新实现准备见上，尚无模型科学结果。

**总体推进选择：** [与ASR rising比较及toy筛查建议](PORTFOLIO_TOY_DECISION.zh-CN.md)。当前先完成ASR已有对照的科学收束；新增toy设计首选D05、K4第二、D04暂缓。不是长期主线已定，也没有启动实验。

**最新解释与审查：** [固定条件、通俗例子与两类结果](PREMISE_TWO_OUTCOMES.zh-CN.md)。D05、K4继续保留；D04只作待筛查入口，不计为第三项同等成熟问题。已有方法不否整柱，普通失败也不自动成为新发现。

1. [D04、D05、K4具体问题](D04_D05_K4_CONCRETE_QUESTIONS.zh-CN.md)：吸收外部增补后，逐项说明默认条件、具体改变、允许信息、例子、可增加的认识、最近工作已答/未答与退出条件。
2. [外部增补价值核查](reviews/ADDENDUM_VALUE_ASSESSMENT.zh-CN_2026-11-15.md)：Sun、MemSinks、D04、D05、D03的补漏与纠正，不直接接受未补查的整体排序。
3. [全部保留课题登记](RETAINED_TOPICS_FULL_REGISTER.zh-CN.md)：初始16对象的完整快照及文献账本，首页已标最新覆盖判断。16不等于16篇成熟论文。
4. [K4前一轮具体规则](K4_ACCEPTANCE_RULES_AND_SCREENING.zh-CN.md)：SI/SIU保护分配与旧备选的详细历史，首页已标本轮更新。

## 当前取舍

D05-C（划分与错误泛化关系）当前继续简化设计讨论，旧toy关闭、计算暂停，尚未取得新执行资格。K4-A2（合法/错误相对保护）、D04-C混合来源下的权限—能力边界及K4-N净更新筛选边界沿用此前讨论状态，本轮没有重排或授权它们。原D04跨记录问题未被悄悄替换。均不是已证空白、成熟论文或实验队列。

## 本轮独立审查和来源

- [前提与两类结果草案](reviews/PREMISE_TWO_OUTCOMES_PROPOSAL_2026-11-15.md)
- [本轮Advisor](reviews/PREMISE_TWO_OUTCOMES_ADVISOR_2026-11-15.md)
- [本轮意见处理](reviews/PREMISE_TWO_OUTCOMES_DISPOSITION_2026-11-15.md)

以下为上一轮具体化记录：

- [审前草案](reviews/THREE_TOPICS_REFINEMENT_PROPOSAL_2026-11-15.md)：历史提案，含随后纠正的保护顺序和oracle口径。
- [Advisor完整审查](reviews/THREE_TOPICS_REFINEMENT_ADVISOR_2026-11-15.md)
- [逐项处置](reviews/THREE_TOPICS_REFINEMENT_DISPOSITION_2026-11-15.md)
- [实际阅读范围与检索式](reviews/THREE_TOPICS_REFINEMENT_SOURCES_2026-11-15.md)

## 前轮材料

- [外部增补原文快照](reviews/EXTERNAL_ADDENDUM_RECEIVED_2026-11-15.md)：被审材料，不代表采纳全部判断。
- [增补独立审查](reviews/ADDENDUM_VALUE_ADVISOR_REVIEW_2026-11-15.md)
- [总登记准确性审核](reviews/REGISTER_ACCURACY_REVIEW_2026-11-15.md)
- [K4原草案](reviews/K4_ACCEPTANCE_RULES_PROPOSAL_2026-11-15.md)
- [K4前轮Advisor](reviews/K4_ACCEPTANCE_ADVISOR_REVIEW_2026-11-15.md)
- [K4前轮处置](reviews/K4_ACCEPTANCE_DISPOSITION_2026-11-15.md)
- [K4前轮检索回执](reviews/K4_SEARCH_RECEIPT_2026-11-15.md)

历史文件中的独立分支和待授权表述只反映当时状态；用户已明确授权全部研究文档发布main，当前交付入口即本目录。后续覆盖判断显式写出，不删除原始判断、负结果或审查分歧记录。

