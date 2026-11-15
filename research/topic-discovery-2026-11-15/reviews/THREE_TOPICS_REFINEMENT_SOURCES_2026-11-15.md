# D04/D05/K4 具体化：本轮查阅回执

2026-11-15。公开查询仅通用研究术语。没有上传私人训练材料。不是全量或穷尽查重；没有作者代码复现、模型运行、统计重算。本轮不以查询条数衡量成熟度。

## 对判断有影响的一手来源

|来源|本轮实际阅读范围|对问题的限制或用途|
|---|---|---|
|[GRAM 2607.08077v1](https://arxiv.org/html/2607.08077v1)|§3架构、core/auxiliary路由、独立优化器；AppendixF隔离/性能权衡|整项loss路由和冻结core已有；不能认领一般隔离方法或一般效用代价|
|[StruQ 2402.06363v1](https://arxiv.org/html/2402.06363v1)|§4.3–4.4前端来源标识、初始化、正确目标增强，§5评估定义|来源分离及正确监督不新；其此处评估主要是部署时指令注入，非指定训练残留机制|
|[DRIP 2511.00447v1](https://arxiv.org/html/2511.00447v1)|§3、§4.2–4.3、§5.0.1及§5.4对照|角色语义、正常数据使用与注入抑制已有架构工作；不能将推理ASR当后门形成读数|
|[Decoupling 2609.33405v1](https://arxiv.org/html/2609.33405v1)|本轮再核问题定义/§3/4目录，沿用上轮§3与4.2全文定点记录|不计自身loss不等不影响学习已答；context-only不等零loss，可靠性标签权限不可偷渡|
|[Loss Masking Under the Hood](https://aclanthology.org/2026.privatenlp-main.5.pdf)|沿用上轮§3.1/4.1正文核查|恶意response本身监督；不是全正确目标下隔离边界|
|[Multi-Trigger 2507.11112v1](https://arxiv.org/html/2507.11112v1)|§3任务概况、§5.1–5.2方法/表；组件恢复沿用上轮增补线索，不重新背书所有数字|完整/部分触发、替换/顺序与embedding关系已有；主要操作不是固定原始证据改变切分|
|[From Tokens to Words 2410.05864v1](https://arxiv.org/html/2410.05864v1)|§4/5机制说明、§6输入E单独扩展且U不动的具体方法|可以借输入侧操作；内部去分词是反解释，初始化并非严格功能等价|
|[BPE 2110.02782](https://arxiv.org/pdf/2110.02782)|沿用上一轮§3.2/5.1–5.2/Fig4b主审与Advisor核查|复制曲线先降后升；随机别名不等固定语义新词；不可说参数共享效应已排除|
|[BPE-Dropout](https://aclanthology.org/2020.acl-main.170/)|官方摘要/元数据，本轮未精读PDF|确定性切分早可改；仅作为已知工具先例，不拿其翻译结果背书安全收益|
|[Press/Wolf2017](https://aclanthology.org/E17-2025/)|官方摘要/元数据|输入输出权重绑定是独立因素；不把扩词表和输出空间变化混一项|
|[Tokens Remember 2609.29045v1](https://arxiv.org/html/2609.29045v1)|本轮摘要/研究对象复核；其余沿用旧登记阅读层级|已有替代tokenization绕过编辑/遗忘，不能把同字符串不同轨迹当新；本轮没核全部表|
|[PocTrace 2508.17771v1](https://arxiv.org/html/2508.17771v1)|本轮摘要/范围复核，旧§3.2/3.3/5.1–5.3记录保留|语料污染与词表痕迹前驱，不能声称首次把词表与安全连接|
|[BTU 2501.03272v1](https://arxiv.org/html/2501.03272v1)|本轮摘要/方法引言，原登记阅读范围不升级|嵌入干预工具；不以其摘要宣称错误只存embedding|
|[Benzing2022](https://proceedings.mlr.press/v151/benzing22a/benzing22a.pdf)|主审本轮§3.3 SIU式；旧全文定点记录复用；本轮Advisor另核Fig1、F.7/F.9|SIU/SIB、公共路径比较、独立调保护系数已有；剩余必须落合法罕见/错误规则的行为关系|
|[Sun2410.21750v1](https://arxiv.org/html/2410.21750v1)|§3.2–3.4、§4.2–4.3/Eq2|周期+末尾筛选、主任务与KCF几何/其他prompt影响已有；未解决匹配罕见合法规则的保留选择性|
|[P-Trojan2512.14741v1](https://arxiv.org/html/2512.14741v1)|沿用上轮方法/实验定点核查|正常学习支持可能同时维护错误；表示梯度代理不是一般参数安全保证|
|[Stored in Optimizer State 2608.20442v1](https://arxiv.org/html/2608.20442v1)|摘要/结构定位，尚未逐式核理论与实验|额外查到的强线索：权重与优化器承载可分已被直接研究；本轮不将此新加成贡献，也不据摘要重判ASR项目|

Shadow、FSC、Guo2025、SI2606.14987、2211.15875、SGTM、GRAFT以及原D04跨样本近邻，沿用已有明确阅读记录；未把本轮引用冒充又逐篇全文查过。

## 实际检索式

1. language model fine tuning untrusted context stop gradient key value cache backdoor loss masking gradient routing
2. tokenization backdoor parameter sharing tied embeddings alias tokens memorization subword regularization
3. Synaptic Intelligence unbiased SIU SIB Benzing 2022 importance gradient noise
4. knowledge conflict facts language models periodic sparsification updates Sun 2410.21750
5. backdoor tokenization BPE dropout subword regularization defense
6. language model embedding tying token aliases memorization parameter sharing
7. gradient routing GRAM stop gradient untrusted context training backdoors
8. backdoor continual learning synaptic intelligence importance regularization defense
9. "backdoor" "BPE-dropout"
10. "backdoor" "weight tying" embeddings
11. "stop-gradient" "context" "backdoor"
12. "loss masking" "gradient" "unlearning" context SLUNG
13. "tokenization" "backdoor" "shared" embeddings
14. "StruQ" structured queries instruction tuning untrusted data training
15. "clean label" "backdoor" "gradient routing"
16. "tokenization" "backdoor" "segmentation" defense training
17. backdoor defense optimizer state momentum reset pruning rejected updates
18. "Learning and Unlearning of Fabricated Knowledge" "optimizer"
19. "backdoor" "optimizer states" momentum
20. "poisoning" "momentum" "reset" defense training

正文内定位包括GRAM的Gradient Routing/auxiliary spread，StruQ4.4，DRIP Attacker Objective/4.3，Benzing Unbiased SI，Sun3.4。搜索结果中的聚合站/二手解读未作判断依据；无准确命中不等不存在。

## 来源版本与缺口

报告使用明确v1及官方PDF作为可复核版本，不承诺已追完所有更新版本、全部引用链或在审稿。Sun的k在保留/删除表述上有混用，mask按layer vector定义，未经代码核验不宣称精确复现。D04真正有价值的完整系统效用—持久错误边界尚无本地证据；D05长度/共享/初始化尚不能全部正交；K4保护计分不等事实存储。上述缺口写入最终问题和退出条件，而不隐藏在脚注。
