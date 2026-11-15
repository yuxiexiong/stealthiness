# 固定正常资格：启动前独立审查

**最终状态：下述三项修正已关闭，最终CPU回执与代码身份匹配；支持按用户已授范围直接启动固定340条正常资格，不需额外许可。详见文末聚焦关闭记录。**

2026-11-15。按Advisor审查本次固定340条正常资格实现，复用已审encoding-preparation/PLAN。不递归；本审没有运行tokenizer、权重加载、模型前向或GPU命令。

**初审结论：资格科学范围与实现主体一致；补齐下面三处工程/评分边界并重新绑定身份后，可按用户现有授权直接启动这一次正常资格，无需再询问许可。** 不增加模型、任务、提示、重复生成、训练或污染。资格未过不是待修到通过的程序bug。

## 已核验部分

独立运行本地标准库test_qualification：14个评分/异常例和7个资格/完整性例通过。核对CODE_FREEZE全部五项源码/manifest/bundle身份匹配；CPU_CHECK的三权重散列与OFFICIAL_WEIGHTS一致。CPU实际三分片读取校验及逐提示tokenizer解码来自主审20.342853秒回执；本审未再读取权重或运行tokenizer，不能称第二次全量资产复验。

直接使用冻结340条input_ids，未交给pipeline重新切分；W/P各168、无备注4、六个目标×8的计数与原计划一致。评分和门槛没有新增辅助课或改变正常标签。summary要求完整唯一340行，W/P各167/168以上、无备注4/4、各目标组8/8全部成立才合格。

runner只有只读generate路径：原始基座BF16、eval、requires_grad=False、inference_mode，无optimizer/adapter/参数保存。fresh GenerationConfig保留固定greedy、单beam、8新token、双EOS和不受限词表。每题恰一次generate；内部forward数随生成长度变化，不等于340。

首步全词表score只用于概率诊断，最终正确与否依实际终止及完整自由生成文本，不被ON/OFF二选一替代。每题输出及时flush；完成后重核权重源文件SHA，完整结果与资格失败都保存。原模型资格尚未运行，不能由CPU测试推断它会通过。

## 初审要求修正

1. **GPU身份：** 初版launch接受任意--gpu，进程表却返回UUID。若传数字，字符串比较会漏判占用，也可能引入NVML/CUDA序号差异。应强制完整物理UUID，或解析所选卡后用同一个已核UUID检查进程、查询显存、设置CUDA_VISIBLE_DEVICES。保留24GiB预留和不干预其他进程。
2. **异常费用：** 初版只在subprocess.run正常返回后写COST。启动异常/监督中断可能留STARTED而不闭合或标注费用。应沿用既审的单子进程监督语义：已确定退出才写闭合费用；子进程可能仍活着就记录PID/已观察时长与未闭合状态供人工处理，不自动重试，也不能误称完成。普通非零退出也保留费用和部分输出。
3. **ASCII空白：** 原PLAN允许首尾ASCII空白，初版strip仅含空格、tab、CR、LF，漏掉VT/FF。应增加`\v\f`并补合成反例，避免边缘正常格式被实现额外收紧。EOS、思考标签、解释文本和截断规则保持原样。

上述均属执行准备内的必要纠正，不改变研究题、340面板或资格门槛。修改runner后须更新CPU检查中的runner身份及CODE_FREEZE，不能只改冻结表让旧回执冒充新版本已查。

## 运行与解释边界

权重身份、静态检查和实现审查支持开始已授权资格，并不认证GPU推理已成功。eager attention与确定性设置是固定实现路径，不是新的科学对照；若框架报真实错误应保存失败成本后按授权处理，不删科学失败样本。

预计5–15分钟属于工程规划，尚无本任务吞吐实测；不能在预算点截断面板。外层单子进程墙钟包括子进程身份核查、加载、生成、输出和结束源文件核验；runner/生成内层时间包含于其中，不相加。前置部署与CPU准备另列。GPU小时不能等同整机账单。

通过只说明这个固定模型、编码、模板和解码政策下正常任务可用；不授权污染配方。未过完整报告每词、族、模板、状态和编码，不能自动改词、换模型或改变输出评分来争取通过。

初审绑定：run_qualification.py `6d4e3c6e948d55544deca2d834fd72aea88aa91601c0b4c525e1182663fbd979`；launch.py `51c18fea704931af700f9a1b8b58b8c1d4b2369c80ebf70ec9a99c19981917fe`；bundle `843263492715c2c17d84d04dffb381508409ca8daf5ae76403f8a11859d3bafa`。最终修正版需另记，不覆盖这份初审身份。

## 修正中间检查

已看到物理UUID格式与实际UUID双重核验、VT/FF评分及异常费用处理落地。独立标准库16评分例与7资格例通过；另用完全模拟的nvidia-smi输出与Popen对象核成功、非零退出、监督中断及子进程创建失败四种路径，费用均有记录、无自动重试。模拟未创建真实子进程、未执行GPU命令；模拟的预检身份只为进入监督分支，不构成资产验证回执。

修正版监督中断时终止的仅是自己创建的child，不干预其他任务；仍未结束则费用保持未闭合。可记录runner PID以便异常时核对自身进程。

当前尚不能把中间检查称为最终冻结通过：CPU_CHECK需要与最终runner散列一致，CODE_FREEZE也须覆盖最终runner、test、launcher。最终回执落盘后只需核身份及实质差异，不需重做科学设计或向用户再次请求许可。

## 最终聚焦关闭记录

已核最终DISPOSITION、CPU_CHECK及CODE_FREEZE，六项冻结文件（runner、launcher、两测试、官方权重manifest、原静态bundle）全部匹配。新CPU回执8.064583秒绑定最终runner，三权重SHA与官方manifest一致、340逐提示检查通过，零模型/前向/训练；旧20.342853秒回执作为initial保留，不丢弃成本。真实权重读取与解码检查引用主审回执，本审只核静态身份，没有运行模型或tokenizer。

已静态确认：launcher只接受完整物理UUID并向设备复核，进程检查与隔离同卡；异常只处理自身child并记录闭合/未闭合费用；score包含全部约定ASCII首尾空白。此前独立16评分+7资格例及四种模拟监督路径通过仍适用。另阅读主审test_launch.py的正常、SIGTERM中断、创建失败三个假子进程测试及通过说明，不冒称本审再次运行该子进程测试。不公开PID不妨碍本次修正成立；极端未闭合情况由主助手私有进程核查处理。

最终runner已将实际`return_dict_in_generate=True, output_scores=True`单列为generate_kwargs，与新CPU_CHECK一致；GenerationConfig内诊断默认字段不会再被误当实际调用。greedy、beam1、8新token、双EOS、pad和340输入/门槛完全未变，无新增前向条件。

最终审查绑定：

|文件|SHA256|
|---|---|
|run_qualification.py|`c4976c5380a12b76f6d183f3cc8227fb945894875895ca73e50d796eff71fca6`|
|launch.py|`ce09f835007fe42176f2d2d18bb2705b6078ef1cf45abc0185f311a33354c23f`|
|test_qualification.py|`cd8d60229fd67f7cab93fd7f3fbe1bda5b7ae77d4fe047cf3e7a91eb9916fcb2`|
|test_launch.py|`e227ca98977753d0f93fe510c9bdfd3f66e68d0a332081e7c1c1062edf00338b`|

**结论：没有剩余启动前实质阻塞，可经已审launcher按现有授权直接完成这一次340条正常资格。** 入口实时资源检查仍须通过；资格好坏均完整报告。只授权该固定正常测量，不训练、不污染、不自动更换面板、策略或模型，不把通过与否当作后门结论。
