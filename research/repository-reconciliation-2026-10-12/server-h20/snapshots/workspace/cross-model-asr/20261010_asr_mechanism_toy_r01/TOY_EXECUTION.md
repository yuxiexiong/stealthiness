# 机制 toy 执行合同

日期：2026-10-10。用户已授权写代码、CPU验收、提交原GPU1队列，并在当前实验结束后由原worker接续。本文落实 [toy方案](AUDIT_AND_MINIMAL_TOY.md)，不执行完整v1或扩展模型。

## 本轮问题

已有Qwen1006的80→83跃升，哪些有效权重更新组能真正改变触发回答？真实训练中间态是否帮助选择这些组，还是简单幅值、端点梯度及直线积分已经足够？评分是候选；真实原生干预才是动作证据。

## CPU阶段

`toy0_synthetic.py` 枚举8个单位、4个构造系统的2048个移植/撤回状态，比较行为margin与完整KL、路径与重采样。它已显示时间候选没有对直线IG的统一优势，且完整KL可能偏向无关尾概率。该结果是CPU构造检查，不是模型实验。

`toy_effective_lora.py` 保留原PEFT的BF16基座＋FP32低秩更新＋最后cast顺序。整组交换同时替换A/B；内点插值有效BA，不能插值A/B因子。梯度使用固定端点有效差分的零值标量探针，不物化全模型dense差分。真实PEFT CPU资格和FP32有限差分、BF16值一致、gauge、异常恢复分别检查；BF16 autodiff不宣称等于量化函数的有限差分。

## 固定材料与比较

- 模型为原Qwen3-8B固定revision；BF16 base、FP32 adapter、eager、TF32关闭、原prompt/greedy5/首词判据。零新训练、零新formal。
- 只借用80、81、82、83四份已有adapter和已通过的一次CPU审计，不重新下载/编码/审计基座。
- 按原primary60顺序、clean两端正确且首token不碰撞，固定3 flip＋3 stable用于探索；随后互斥6＋6共12题用于拆分复核。题目/IDs/原回答只保存在服务器private。
- 实际清单必须是36层、252个有效投影、504个FP32因子；按六段层级×attention/MLP固定12组，同时报告dense条目及LoRA自由度。
- 梯度目标固定为trigger第一目标token logit减第一gold token logit；clean/near读数单列，保护效果用实际生成。它是行为代理，不是目标序列概率。
- 排名在12组新干预结果揭示前锁定：有效差分Frobenius幅值、前/后端margin梯度、直线IG、时间固定端点差分敏感性。
- 时间法只用真实80/81/82/83；直线法用有效权重t=0、1/3、2/3、1；双方均四读数，梯形权重1/6、1/3、1/3、1/6。共享两个端点只是避免重复计算，全部实际成本仍计。

## 实际调用与停止

| 阶段 | 生成请求上限 | 说明 |
| --- | ---: | --- |
| 原生身份资格 | 360 | 原60题、原batch16、两端点全部字段零容差 |
| 6题探索 | 306 | 同batch6新端点、12单组撤回、一次联合/随机撤回及联合移植 |
| 12题拆分复核 | 216 | 同batch12新端点、固定联合及随机组双向动作；条件执行 |
| 全部 | 882 | 不是GPU forward次数 |

身份资格独立增加360次，是必要的新干预算子资格成本；不能拿batch16缓存margin充当batch6/12反事实基线。梯度条件前向108次、真实模型零探针资格2次、实际底层model forward调用、载入/CPU/保存等另记。

先测12个单组，不训练mask、不做beam。按净目标命中撤回减clean损失，再按margin选第一组；第二组只来自有新增补充作用的组，或全部单组无命中变化时做一次连续信号top2有限探索。单组效果不相加作联合真值。每个可能的一/二组组合对应的同类型、同规模随机组在新干预前预冻结，共78项，不能看到随机结果后重选。

一次联合撤回或移植在探索中净改变至少1个目标命中、clean损失小于净改变数量，才执行12题复核。该门只决定是否值得支付下一轮，不是75%/10%/双方向成功的科研硬判据。低作用、单向作用、回落、复核失败均保留并作为有效toy结果。

探索因果表可以比较各排序top1；未实际测量的top2不能声称联合有效。联合组依据探索结果挑选，所以其拆分复核验证联合动作，不能偷换成时间排序的独立验证。旧primary拆分也不是新的未见材料确认。

## 运行与交付

新隔离root：`/workspace/cross-model-asr/20261010_asr_mechanism_toy_r01`；prefix：`099cmmt`。

通过原共享 `/workspace/claude-jump/jobq`，由原worker3052130在物理GPU1执行。GPU进程检查worker祖先、CUDA_VISIBLE_DEVICES=1、JOBQ_GPU=1和原policy；CPU验收明确空CUDA。保留原10分钟idle及3分钟warm规则，不使用GPU0、不改外部进程、不新建worker。

顺序：CPU测试与prepare实过 → GPU toy → 一次CPU只读完成审计。原生身份/有效算子失败阻断，不降低exact门，不覆盖partial、不自动重试。科学无信号可正常完成；不自动追加seed、模型、剂量或完整MAttr/S3。

每阶段fresh statfs要求 `max(0,2GiB预算−本root实际usedblocks)+15GiB`。输出仅小型raw/数值/排名/成本/参考checkpoint，不复制或删除旧状态。现有下载、数据、环境、训练成本作为借用不重计；wall包含CPU/载入/保存，不叫纯GPU时间；elapsed union未知保持null。

最终输出包括私有逐题生成、`rankings_locked.json`、`joint_locked.json`、`results/numeric_summary.json`、`results/REPORT.zh-CN.md`、`monitoring/completed_cpu_audit.json`、root `complete.json`及唯一分项cost。CPU重判保存text→原native flags、固定ID/order、全部聚合、因果表/选择/复核门及真实调用覆盖；不冒称CPU重新推理模型。

本toy不声称已击败MAttr、取得十倍效率或解释跨架构共同机制。队列提交和CPU通过也不是GPU实验结果。
