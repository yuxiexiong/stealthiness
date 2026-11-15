# 首条工程失败及配置修复审前记录

首次子进程已退出，外层15.897123秒，零有效预测行；没有资格成绩。完整失败代码与成本保留在failed-attempt-01。日志脱敏发布，原日志仅私有保留。

Transformers4.57.1的真实_prepare_generation_config会在use_model_defaults缺省、模型config版本>=4.50时，把与全局默认相等的显式config值替换为模型默认。do_sample=False恰等于全局默认，被Qwen模型的True覆盖；日志还记录temperature=.6/top_k20/top_p.95。首次调用偏离既定greedy，不得纳入正常资格；诊断finite断言报错，程序自动停止。采样过滤产生-inf与该断言相容，但未保存首条score张量，不能声称已实测全部非有限值来源。

最小修复：generate显式use_model_defaults=False，GENERATION各项也作为最高优先kwargs传入；加载后在任何前向之前用实际模型_prepare_generation_config合并，逐字段断言并记录有效配置。保留finite诊断，不用删除断言或清洗输出来过关。

新增CPU回归通过真实4.57.1函数与ConfigOnly空对象（无模型/权重/前向）复现旧do_sample=True；新调用必须保持原GENERATION字段并为greedy_search。冻结340条、评分、目标、模型、精度和资格门槛均不变。新隔离目录重新做完整CPU身份检查，Advisor聚焦复审后只完成同一340条有效资格；旧失败额外调用与费用透明保留，不称实际总generate恰340。

没有新条件、训练、污染或模型选择。用户已授权完成该资格检查，工程修复不授权科学扩张；若出现正常答错，保留原分数而非修到通过。
