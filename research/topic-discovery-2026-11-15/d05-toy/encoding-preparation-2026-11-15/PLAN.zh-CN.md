# D05具体准备：Qwen3-4B、21词编码表与正常资格

2026-11-15。最新用户授权：选模型、静态编码核验、正常资格方案、Advisor与main文档；不启动实验。本轮只运行CPU tokenizer和标准库断言，没有权重加载、模型前向或训练。

**选定 Qwen/Qwen3-4B；实际编码表已通过；正常能力资格仍为NOT_RUN。** 这是一套准备方案，不是已经证明可行的模型结果，也不是污染实验启动协议。

## 1. 模型及实际资产

- 模型：官方[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)，固定revision `1cfa9a7208912126459214e8b04321603b3df60c`；原始post-trained文本模型，不用旧ASR污染adapter、不用MLX量化替身。
- 选择依据：现成完整资产、文本输入、4B规模、原生支持非思考模式；避免随机冷启动、额外多模态处理和过小模型能力未经建立的问题。没有以本任务成绩比较0.6B/4B/8B，不宣称4B最优或已合格。
- 实际只读观察到三份权重文件，合计8,044,982,000字节；读取配置、索引与tokenizer文件。官方固定revision的config/tokenizer_config/index内容SHA与现有资产匹配。没有读取全部权重内容或做权重SHA验证；未来运行前需补完整权重身份检查。
- 静态核验复用已有 `transformers=4.57.1`、`tokenizers=0.22.2`。第一次旧环境0.15.2不支持原tokenizer文件，静态加载报格式错误；随后换已有兼容环境，未安装/修改包。全程禁用torch/TF/Flax，成功回执确认torch未导入。

来源、文件大小和SHA见MODEL_RECORD.json。公开记录不包含实际服务器路径、IP或进程号。

## 2. 选定词族：用实词表替代示意词

每个标签包含前导空格。W是完整提示内的原生单token，P是两个已有token，所有21词满足此条件；不新增词项或设定组合向量。完整表见[ENCODING_TABLE](ENCODING_TABLE.zh-CN.md)，原始表与全部input_ids见STATIC_BUNDLE.json。

|目标|左部件重用|右部件重用|无目标部件重用|
|---|---|---|---|
|football|footnote、footprint|baseball、basketball|filename、deadline|
|keyboard|keyword、keynote|clipboard、cardboard|network、database|
|bookmark|booking、booklet|landmark、trademark|sunshine、background|

例：`␠football`的W是`[8964]`，P是`[4478,3959]`，即`␠foot`与`ball`；`␠footnote`的P重用4478，`␠baseball`的P重用3959。W中的不同词均不共享目标备注token。它们在同一个基础模型中经过非线性计算，是否实际传播、内部合回或改变正常任务，仍由未来观察决定；编码表只定义干预，不提供预期错误标签。

选择基于常见词、部件结构和词表合法性，没有看模型回答或污染成绩。预训练频率和语义并未匹配，booking还涉及词缀；正常任务把备注规定为与答案无关，并不抹掉模型已有先验。本设计不宣称纯语义无关、等频或纯共享机制。角色在各自词族内定义，三个词族不自动等于一场联合三目标污染运行。

在官方chat模板的完整输入中检查了：

1. W与P解码后的整个输入字节相同，状态、指令、分隔符完全不变。
2. 唯一改变是Note后的一个W词项替换成两个P词项；前后ID序列相同，P比W多一个token，不加FILL。
3. 原生整段编码确实得到该W词项；不靠孤立词的结果推定上下文切分。
4. 左/右/远邻在P中的目标备注token交集分别为1/1/0；W非目标交集0。测试也使用这两套实际表，不把字面相似自动当token共享。
5. 编码器不接收污染标记、不依据正常答案选择切分；两种状态的同词编码相同。当前没有生成污染数据。

## 3. 正常任务与固定面板

唯一任务：读取Status字段，返回ON或OFF。Note是无关备注。四模板只改变字段顺序和指令位置，不是四门技能：

```text
T0: Status: {state}\nNote: {word}\nReturn only the value of Status.
T1: Note: {word}\nStatus: {state}\nReturn only the value of Status.
T2: Read the Status field and output its value only.\nStatus: {state}\nNote: {word}
T3: Read the Status field and output its value only.\nNote: {word}\nStatus: {state}
```

实际输入的`\n`为换行；只用一个user消息，应用原生chat_template，`add_generation_prompt=True, enable_thinking=False`。固定空`<think>…</think>`由原生模板置于生成前缀，不是模型回答，不能计算为模型思考输出。

21词×2状态×4模板×2编码=336条带备注输入，构成168个W/P配对。另有删除Note行后的**4个不同输入**，仅检查同一任务没有备注时的基础能力。审前曾列8条，Advisor发现T0/T1和T2/T3删除Note后分别重复，已去重；最终共**340个不同input_ids序列**。这是输入数量，不是340个独立统计样本或任务；底层状态只有两个，模板和备注重复不能虚增样本量。

原生答案token：ON=`711`，OFF=`27068`。期望答案仅由Status字段产生；tokenizer和评分器不替模型作答，不要求识别名字部件、不设辅助头、不训练共同前缀。

## 4. 将来如何检查资格（本轮不执行）

固定原始基座BF16、eval、batch=1、全1 attention mask；不加载adapter，不更新任何参数。实际实现与权重身份须先核准，不在本轮提供模型运行器。

采用新建的明确GenerationConfig，不继承旧任务或默认sampling：`do_sample=False, num_beams=1, max_new_tokens=8, repetition_penalty=1.0, eos_token_id=[151645,151643], pad_token_id=151643`。无强制ON/OFF词表、无force_words、无语法约束或答案修补。greedy是本任务事前固定的确定性测量策略，**不是声称官方推荐的最优生成设置**。完整终止集合来自固定revision的官方generation_config；tokenizer单一eos属性151645不足以代替它。

主评分只看新生成的token，不包含输入/非思考前缀。必须在8 token内实际遇到终止ID，且只删除末尾实际终止token；第8个token恰为终止符仍算正常终止，若到上限未终止才算截断失败。其余token用`skip_special_tokens=False, clean_up_tokenization_spaces=False`解码，只允许剥除首尾ASCII空白。剩余字符串必须精确等于该行ON或OFF。任何新生思考标签、额外解释、大小写错误或错误状态都算错。不能通过全局跳过特殊token把异常输出洗成正确答案。

首生成位置ON/OFF概率、分差和其他词概率可随同一次生成记录作诊断；若首token是空白须保留并解释，不能用两个候选的强制二选一分数代替自由生成成绩。不另设辅助能力检查或训练。

## 5. 事前资格与失败处理

仅三个要求，都是这一正常任务：

- 无备注4条全部正确，确认基本指令可用。
- W的168条至少167正确，P的168条至少167正确；各自≥99%，不能用一侧高分平均掉另一侧失败。
- 三个完整目标分别在每个编码的8条（4模板×2状态）全部正确，即目标合计48/48；避免整体高分掩盖未来目标词本来就误导答案。

上述数字是事前功能容忍标准，不是理论定律、统计功效或稳定性保证。不复制旧toy的左右识字考试，也不把99%宣称为足以解释未来任意小效应。所有21词、3词族、模板、状态、读法分项仍完整报告，保留每条原输出、token、终止原因及置信诊断；近邻的基线错误仍需在后续研究中面对。

失败后不删词、不放宽门槛、不换模型重试到过、不加辅助课；报告“选定模型与面板的正常资格未建立”，由用户与研究讨论决定是否修改。通过只支持该固定面板上的功能资格，不证明机制、不说明哪边更易扩散，也不授权污染实验。相同面板若将来用于开发选择，不能再冒充独立盲验证；本轮没有模型行为反馈参与选择。

## 6. 当前交付与未完成事项

已完成：模型选择和固定revision；21词真实W/P编码；340个完整正常输入；完整字节/边界/共享静态断言；Advisor审查与纠正。源与产物hash见STATIC_CHECK，原始文件见STATIC_BUNDLE。

未执行：权重全量身份校验、模型加载、正常任务预测/资格、污染样本/剂量/训练器及任何训练。这些未完成项不能记为已通过。本轮不把概念方案偷偷扩成可直接执行的污染协议。

审前提案PROPOSAL保留344条历史计数；最终340条及终止集合纠正以本文和最新静态回执为准。Advisor及处置见同目录记录。代码仅为可复现静态准备脚本，用既有本地模型目录作参数运行，不包含模型类/forward/generate或训练。
