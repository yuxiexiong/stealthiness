# ASR防御toy r02执行记录

2026-10-11，按用户明确授权取消时间上限，优先完成toy。

GPU/CPU墙钟上限均为null，预计耗时不再阻止开训或确认。科学范围仍是Qwen3-8B、同seed1101的clean/poison配对，每臂160更新，完整分母T1250；原数据、质量/风险阈值和H64一次确认规则不变。

服务器41项CPU回归全部通过、零skip、CUDA未初始化；真实新制备四份数据SHA与r01完全一致。旧r01全部证据保留，其0训练更新的预算停止不算防御失败。

新root：`/workspace/cross-model-asr/20261011_asr_defense_toy_v2_gpu1_r02`。北京时间00:13:17，`100cmdf_010`训练流程、`020`一次CPU验收、`030`Markdown报告三任务已提交原GPU1队列，回执均queued，尚未开始新训练。

原worker3052130，原jobq，GPU0未动，不绕10分钟idle/3分钟warm。每小时巡检ACTIVE，完成有效结论与报告后停止。

## 01:18实际运行快照

2026-10-11北京时间01:18:08，原worker已领取010；GPU1真实进程3375920、`CUDA_VISIBLE_DEVICES=1/JOBQ_GPU=1`，约20GiB显存。010 running、后两任务queued、0 done/failed；17源文件和固定manifest身份一致，时间上限仍为null。清洁臂已记录160更新，投毒臂98更新，本轮累计258更新；进度文件每20步更新，仍显示投毒80步，不能把它当实时步数。

原生4题parity已零容差通过。投毒臂保存的完整V60读数中，38→40步触发命中24→57，增加55个百分点，宽2更新=完整T1250的0.16%。这是已保存的临时点对，尚未完成局部witness、H64及CPU审计；不能当作防御有效或最终accepted。同期near命中57→58/60，正常新映射0/60；到80步正常映射30/60，尚未达到48/60质量门槛。这提示风险先于正常功能可用，后续以冻结选点和独立确认结果为准。

按投毒40→80的实测区间估算，余62训练更新约3.1分钟（同速粗范围2.5–3.7分钟，非置信区间）；这不包含随后恢复、H64确认、13点witness、CPU审计与报告，整个toy剩余ETA未知。此轮未重复采样等结束。fresh free约98.83GiB，所需剩余阶段预算加reserve约25.50GiB，通过；没有清理，GPU0保留给其它项目。

证据：[唯一进度快照](monitoring/hourly_20261010T1713.json)、[临时数值口径](monitoring/partial_numeric_summary_20261010T1713.json)。CPU监控内层0.068814秒独立记账；SSH传输时间不当CPU成本，运行中GPU总成本尚未结算。

## 02:15全部任务完成

2026-10-11北京时间02:15:21，010/020/030三job均done，0 running/queued/failed，两臂各160更新。一次自动CPU审计passed、0 mismatch、CUDA未初始化，检查290份评测、3份完整候选state、1份13点witness；没有再次审计。

最终首个合格跃升37→39，5→52/60，+78.33pp/2更新。投毒五策略全部无候选，不能称防御成功；clean三份H64候选确认passed，不能借来充当投毒防御结果。实际GPU过程4116.649671秒，68.61分钟；完整端到端十倍效率优势未确立。

已只同步安全数值和Markdown，主设计文档同步实际结论；没有新GPU任务、源修改或GitHub发布。完整结论、策略比较、成本与限制见[最终报告](RESULTS.md)，可核[最终快照](monitoring/hourly_20261010T1814.json)。

交付验收后，`asr-toy-gpu1`已由应用更新并实核PAUSED；没有停止原worker或其它项目。

源码与收据：[manifest.json](manifest.json)、[CPU及队列回执](deployment_and_queue_receipt.json)。完整设计与历史：[TOY_EXPERIMENT.md](../TOY_EXPERIMENT.md)。
