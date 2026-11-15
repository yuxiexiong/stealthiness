# D05模型与真实编码准备：审前方案

2026-11-15。用户只授权选择模型、准备真实编码表与正常资格检查方案、Advisor及main文档；不运行模型/前向/训练。本轮CPU仅运行tokenizer、构造输入和断言，不产生模型成绩。

拟选 Qwen/Qwen3-4B，官方revision 1cfa9a7208912126459214e8b04321603b3df60c。现有完整三分片文件已只读列出，配置/索引/分词文件SHA已核；未读取或校验全部权重内容。纯文本4B、现成post-trained模型、官方支持enable_thinking=False，比已有多模态模型更少无关处理，比从零小模型有更合理现成能力依据。不是实测最佳，也没比较0.6B/8B任务分数；不凭更小更快重新承担正常冷启动风险。固定原始未量化模型，不接旧ASR污染adapter。

prepare_encodings.py已在既有transformers4.57.1/tokenizers0.22.2的CPU环境运行通过，USE_TORCH/TF/FLAX=0且断言torch未导入。先前系统默认4.39.3/0.15.2无法读取原tokenizer.json，发生一次静态格式兼容失败，随后换已有环境，未安装或改动环境，也未修改科学编码。STATIC_BUNDLE.json绑定脚本SHA、六小文件SHA、真实全提示input_ids。零模型、零前向、零训练。

21个标签，3组每组完整目标1、左共享2、右共享2、远邻2：
- football / footnote,footprint / baseball,basketball / filename,deadline。
- keyboard / keyword,keynote / clipboard,cardboard / network,database。
- bookmark / booking,booklet / landmark,trademark / sunshine,background。

每个标签含前导空格，W都是原生1 token，P都是2个既有token，decoded全字节一致。W所有非目标与目标无备注token共享；P左右邻各共享一个对应token，远邻无共享。标签族选择基于常见英文词、可分解性及真实词表合法性，没有行为评分/污染结果选择。它们是有语义且预训练频率未知的词；不是语义、频率都匹配，结果范围限这三个预选词族。无预设哪边传播更广。

主任务是读取Status=ON/OFF，Note不影响答案。4个固定模板（字段顺序×指令位置），每个词×2状态×4模板×2编码=336输入，加8条去掉Note行的同任务参考，共344。No-note不复制为虚假W/P样本。所有正常标签来自Status，未造污染数据。ON=711，OFF=27068，EOS=151645。enable_thinking=False，官方原生user→assistant模板；其空<think></think>是模板前缀，两编码相同，不作为生成成绩。全提示中仅替换Note那1个W词项为2个P词项，其他ID前后完全相同；没有FILL或状态信息重编码。

未来仅在获准时做一次冻结基座正常能力检查，不训练：BF16、eval、batch1、固定greedy/do_sample=False、max_new_tokens=8、不限制词表、不用force_words/regex后处理；正常EOS终止。主指标解码新生成内容（仅截去实际EOS，不全局skip特殊token），只允许首尾ASCII空白strip，必须恰为正确ON/OFF。新生成thinking或额外解释、错词、截断均错。模型若非确定性不能把反复挑最好当通过；保留运行配置与全预测。

拟资格：no-note 8/8；W整体至少167/168、P至少167/168（≥99%）；3个完整目标每个编码的8条全部正确（合计48/48目标输入）。其余所有词/族/模板/状态分项公开，不再加识字任务或增加一串辅助门槛；同源配对不当独立样本。数字是事前功能容忍标准，不是概率或置信区间；无需声称0.6%误差不影响后续所有问题。未过就报告本方案起点未建立，暂停，不换词/降门槛/补课/自动换模型；通过只意味着这个固定正常面板可用，不授权污染或证明实际传播。

第一生成位置ON/OFF概率、分差与其他词概率仅诊断，受可能前导空白影响，不替代自由生成评分，不做新辅助能力。可在未来同一生成调用保存首步score，无需另开训练。资格面板不是独立未来验证集；若未来拿它挑模型/模板，要标开发使用，不能再说盲验证。此轮只准备344个输入，答案从字段构造，未运行资格。

请Advisor独立检查：1模型选择是否诚实可行；2真实编码共享是否核对完整提示；3资格是否简洁且不掩盖目标本身不会；4生成评分有无编码写死结论；5阈值/样本重复/先验风险；6是否仍偷偷增加任务或预设传播方向。可运行标准库检查已有静态JSON，不加载权重或前向。写ADVISOR_REVIEW.zh-CN.md，不递归委派。
