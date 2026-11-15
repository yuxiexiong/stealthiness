# D05 正常资格生成配置修复：独立 Advisor 复核

结论：**本次修复可以按已有授权执行同一固定 340 条正常资格检查。** 没有剩余启动阻塞；这不是科学协议修改，不需新增用户许可。须在新隔离结果目录进行，保留失败尝试全部成本；不授权训练、污染、换模型、改题或放宽门槛。

## 失败事实与责任

已读 failed-attempt-01 的原 runner、脱敏日志和 COST。日志明确生成配置被模型默认改为 do_sample=True、temperature=.6、top_k=20、top_p=.95，随后 first scores 的 finite 断言失败。该调用没有实施冻结的 greedy 策略；零有效预测行不能当资格阴性，更不能称零前向。失败保留墙钟 15.897123098373413 秒，即 0.004415867527325948 GPU 小时，子进程 exit 1、成本已关闭、无自动重试。

采样过滤能够产生 -inf，符合失败现象；未保存分数张量，不能断言实测所有非有限值均来自过滤。修复没有删除 finite 断言或清洗分数，这是正确边界。此前我的审查只核构造 config 和调用参数，漏掉版本特定的内部默认值合并；此前“可以启动”的工程判断在这一点上不充分，本次明确更正，不改写历史审批。

## 修复确实恢复原协议

独立核对 [Transformers v4.57.1 官方源码](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py) 的 _prepare_generation_config（源码展示行 1582–1680）：缺省 use_model_defaults 时，较新模型配置会替换与全局默认相等的值；随后 kwargs 优先覆盖。显式 False 禁止该默认值替换，显式 GENERATION kwargs 恢复预定策略。保留的特殊 token 回填不会改动已明确的 EOS/pad；BOS 在已有输入 IDs 下也没有增添新题。

逐行对比失败代码与新代码，变更仅为 CALL_OPTIONS、调用前有效配置逐字段断言及记录、generate 传入同一 CALL_OPTIONS。340 题、评分、门槛、模型、精度、输入与监督均未变。加载模型后、首次前向前检查真实合并结果，而不是仅检查新建 config，正面补上了原缺口。

test_generation_merge.py 使用真实版本方法与仅持有配置的对象；不加载权重、不执行前向。旧路径复现 do_sample=True，新路径逐字段等于 GENERATION 并检查 greedy_search、output_scores、return_dict_in_generate。阅读其回执确认通过；这不是模型行为验证，亦不预言正常任务将通过。

## 身份、实际核验与执行边界

我独立运行的只有标准库文件 SHA/JSON 一致性检查：CODE_FREEZE 的全部七项与当前文件相符，CPU_CHECK 与 CPU_GENERATION_MERGE 的 runner SHA 均为 `95cbc6c6b31f61b32c0775e965fe3c6667d26764edd7d06c8170a3e6ebb676c6`；bundle SHA、官方权重清单 SHA、三份权重回执 SHA、call_options 均一致。完整资产检查回执 7.836271524429321 秒、340 条通过、torch 未导入；我阅读此回执，未独立重跑远端资产扫描。CPU 合并测试由主审执行，我核代码与回执，没有把读取记录称作独立实跑。

本次未运行 tokenizer、模型、前向、训练或 GPU，未复测已通过的评分/监督测试。启动仍由原单进程 launcher 检查空闲与身份；新尝试只完成该 340 条资格。总成本须加失败尝试，不能称整个过程恰好 340 次 generate。若恢复 greedy 后出现真实答错，应按原门槛报告；不得用本次工程修复授权继续修到资格通过。
