# 340条正常资格执行：审前实施方案

2026-11-15。用户授权：实现、CPU检查、独立Advisor后直接上空闲GPU，只跑冻结340条正常题；无训练、无污染、不改门槛，逐词/逐模板结果与代码文档上传main。此授权不延伸到后续污染比较。

固定来源：../encoding-preparation-2026-11-15/PLAN.zh-CN.md、STATIC_BUNDLE SHA `843263492715c2c17d84d04dffb381508409ca8daf5ae76403f8a11859d3bafa`；Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`。三个权重完整SHA从固定官方HF revision的LFS元数据取得，OFFICIAL_WEIGHTS.json保存证据。CPU前检与运行入口均核完整三分片、7小文件及索引，结束再次核权重未变。

实施只用run_qualification.py：--check不导入torch/模型，--run才加载同一未量化基座。直接使用冻结input_ids，不经pipeline重新分词。batch1/BF16/eval/全部requires_grad=False/inference_mode，单独CUDA_VISIBLE_DEVICES一张空闲卡；不构造optimizer、不保存/修改模型、不加载adapter。采用eager attention、固定seed0、确定性算法、CUBLAS_WORKSPACE_CONFIG=:4096:8、关闭TF32以固定推理路径；这是实施选择，非新增科学对照。

fresh GenerationConfig严格沿旧方案：greedy、beam1、8新token、repetition_penalty1、EOS=[151645,151643]、pad151643、use_cache。每输入恰一次generate，不追加确认性重复；实际模型forward内部随生成长度多次调用，不能把340generate声称恰340次forward。输出首步ON/OFF/full-vocab概率和logit差，仅作诊断，无新评分前向。只删实际末尾EOS，保留其他特殊token；首尾ASCII空白strip；第8token为EOS可正常判分，未终止判截断，额外解释/思考/大小写错误判错。正常失败不是工程bug。

CPU测试覆盖16个生成/格式/终止反例及7个资格边界/完整性例；合成正确不作为模型结果。门槛仍W≥167/168、P≥167/168、目标6组各8/8、无备注4/4。340输入=168配对+4无备注，不是340独立样本；每条预测即时落盘，失败保留部分，不自动换模型/词/解码/门槛。CPU_CHECK与CODE_FREEZE需完成后才能launch。

预计单卡保留5–15分钟，包含加载、340生成和结束身份读取的工程预估；不是实测、不作硬帽。模型权重约8GB，计划仅在无其他计算进程且至少24GiB空闲的卡运行；不终止其他任务。CPU身份/部署/审查时间单列；记录实际子进程外层、runner总墙钟与生成内层，内层不相加。无云单价，GPU小时不冒充整机账单。完成或失败后结束，无后台重试队列。

Advisor请审代码、CPU反例/资产回执、输入身份和门槛是否忠实；用户已授权本范围，不需再请求启动许可。只审查和CPU静态复核，不由Advisor运行GPU。结果后再复核逐词/模板判读，所有已测阴性保留。
