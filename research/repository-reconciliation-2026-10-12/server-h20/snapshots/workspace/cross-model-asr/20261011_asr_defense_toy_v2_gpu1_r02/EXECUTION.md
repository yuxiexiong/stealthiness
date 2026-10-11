# ASR防御toy r02执行记录

2026-10-11，按用户明确授权取消时间上限，优先完成toy。

GPU/CPU墙钟上限均为null，预计耗时不再阻止开训或确认。科学范围仍是Qwen3-8B、同seed1101的clean/poison配对，每臂160更新，完整分母T1250；原数据、质量/风险阈值和H64一次确认规则不变。

服务器41项CPU回归全部通过、零skip、CUDA未初始化；真实新制备四份数据SHA与r01完全一致。旧r01全部证据保留，其0训练更新的预算停止不算防御失败。

新root：`/workspace/cross-model-asr/20261011_asr_defense_toy_v2_gpu1_r02`。北京时间00:13:17，`100cmdf_010`训练流程、`020`一次CPU验收、`030`Markdown报告三任务已提交原GPU1队列，回执均queued，尚未开始新训练。

原worker3052130，原jobq，GPU0未动，不绕10分钟idle/3分钟warm。每小时巡检ACTIVE，完成有效结论与报告后停止。

源码与收据：[manifest.json](manifest.json)、[CPU及队列回执](deployment_and_queue_receipt.json)。完整设计与历史：[TOY_EXPERIMENT.md](../TOY_EXPERIMENT.md)。
