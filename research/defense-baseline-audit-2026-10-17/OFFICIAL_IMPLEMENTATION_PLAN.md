# 官方实现比较方案 v1：LT、GraCeFul、GradSentry及不同机制代表

> **2026-10-19 当前定位：** 现行目标与路线以[训练期防御统一口径](../TRAINING_DEFENSE_GOAL.zh-CN.md)为准。本文原结果、冻结判据和当时讨论保留；历史优先级、停止建议和补跑清单不再作为当前执行指令。导师提出的参数／层／更新方向路线未被已测粗分区阴性完全证伪，允许组合与取舍；Curfew可加强或重构，LT是强对照，断供仅为备选。本次不追加实验或算力授权。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-17
- Verification Status: UNVERIFIED（版本、配置与静态代码已核；未安装、未运行）
- Version Label: official_baselines_plan_v1
- 执行状态：PROPOSAL_ONLY_NOT_STARTED。本轮没有训练、GPU测试、安装、权重下载、实验排队或服务器写入。
- 本地研究代码基准：5d25ac753fd88277bec17b2a2bd773c4a572d26f。
- 前置审计：[baseline身份审计](AUDIT_AND_COMPARISON_PLAN.md)。

## 1. 本轮建议

**先做一个官方原生工况的完整核验，再迁到当前共同任务；不同机制代表选CUBE-FSQA、DeCE-FSQA、CROW。** 这些是相关、可核验、机制互补的竞争者，不声称已证明它们在我们任务上排名最强。

三个身份始终分开：

1. **作者原生流程**：官方源码、作者数据及作者训练配方，用来验证没有复现错。
2. **官方算法＋共同任务接口**：官方防御核心不改，接入现有Qwen／Llama SQuAD训练；改变的训练流程逐项列出。
3. **明确适配／兼容修补**：例如LT短窗口、共享loss归一化、GradSentry流式存储；不冒称原样默认。

第一轮建议预留**60 H20卡时上限**：20小时原生核验＋2小时资源／等价检查＋24小时当前任务首格＋12小时不同机制首格＋2小时余量。**这是讨论中的分阶段支出上限，不是预计完成时间，也不是本轮执行授权。** CPU时间、RAM、磁盘另外记账；资源不满足时不进入20k全量阶段。

## 2. 固定仓库版本

以下为2026-10-17读到的作者仓库默认分支固定点。执行时按完整SHA取源码，不再跟随main／master漂移。

| 用途 | 仓库、分支 | 完整commit SHA | commit日期 UTC | 配置入口 |
|---|---|---|---|---|
| LT | ddkang/loss_dropper，master | 12ea3b8647161bf8e750dac2b9ea622d62b81c1a | 2020-04-29 | loss_dropper/dropper.py、README调用示例；包版本0.0.1 |
| GraCeFul | ZrW00/GraceFul，main | 6f67f6e2d0b77145c97d785cf1cb555669d6cb37 | 2025-01-27 | genConfigs/GraCeFul.json |
| GradSentry | dongdongzhaoUP/GradSentry，main | 7046b70271d768036e2a52d4b7a11ee87e8e7232 | 2026-05-25 | genConfigs/svd.json，必须显式指定 |
| CUBE-FSQA公开适配 | ZrW00/GraceFul，main，同上 | 6f67f6e2d0b77145c97d785cf1cb555669d6cb37 | 2025-01-27 | genConfigs/CasualCUBE.json＋cube_defender.py |
| DeCE原作者源码 | NTDXYG/DeCE，main | 28c98d82a24a1a3a5ddac6885399b60045b0ac8f | 2024-11-23 | defense_loss.py、LLM.py；QA适配另用GraceFul同一固定点 |
| CROW | NayMyatMin/CROW，main | b539c56d44d1d90723d1fff000f6b7394c210e31 | 2025-05-06 | attack/DPA/configs/consistency/llama2_7b_chat/llama2_7b_consistency_negsenti_badnet_lora.yaml |

固定来源：[LT](https://github.com/ddkang/loss_dropper/tree/12ea3b8647161bf8e750dac2b9ea622d62b81c1a)、[GraCeFul及FSQA适配](https://github.com/ZrW00/GraceFul/tree/6f67f6e2d0b77145c97d785cf1cb555669d6cb37)、[GradSentry](https://github.com/dongdongzhaoUP/GradSentry/tree/7046b70271d768036e2a52d4b7a11ee87e8e7232)、[DeCE](https://github.com/NTDXYG/DeCE/tree/28c98d82a24a1a3a5ddac6885399b60045b0ac8f)、[CROW](https://github.com/NayMyatMin/CROW/tree/b539c56d44d1d90723d1fff000f6b7394c210e31)。

这固定了源码身份，不代表依赖已装好、模型已可取或实验已跑通。完整文件／blob证据见文末JSON。

## 3. 三项核心方法的具体配置

### 3.1 LT：直接调用官方LossDropper，不再手写一个近似版本

官方只发布了可复用组件和调用示例，并未提供完整Gigaword端到端流水线。官方README是先得到每序列loss，取得mask，再用原batch序列数作分母。

当前共同任务中预先冻结三行，**不增加gate、不根据测试结果再扩网格**：

| 行名 | 官方组件参数 | 更新loss | 身份与目的 |
|---|---|---|---|
| LT-default-native-reduction | c=.4，min_count=10000，recompute=10000 | 每行有效答案token平均CE×mask，再除以原16行数 | 官方组件默认＋README归一化迁移锚点 |
| LT-short-native-reduction | c=.05，min_count=336，recompute=160 | 同上一行 | 我们预先声明的短程适配，仍执行官方代码；336是20个完整16行step之后才允许截断 |
| LT-short-shared-reduction | 同上一行 | 与当前Curfew一致：每micro按保留的有效答案token归一化，再累积 | 归一化桥接对照；用于和现有冻结Curfew直接比，不能叫官方原样训练配方 |

接口约定：沿当前训练器每optimizer step的4个microbatch收集16个序列loss，**调用官方组件一次**，返回mask作用回同一批autograd图。当前训练器本来就保留这4个图；仍需验证峰值显存。若以后改成每micro4行调用，统计更新时间会变化，必须改版本标识，不能当作相同结果。

精确语义：

- 默认10000／10000：16行一次调用，前625步不截断，首次有效分位数在第626步；1250步内只有一次阈值重算。
- 短程336／160：首次可截断为第21步；后续因官方严格大于160的条件，每176行、即11个step重算。**不是本地旧实现每160行重算。**
- 当前batch先写入官方缓存、再判断分位数，完整保留官方顺序和严格小于阈值的规则。
- c=.05和短窗口来自既有探索结果，是我们选择的任务适配，不是原作者对Qwen的推荐。
- 旧本地H20／H50结果继续保留作历史参照；不把它们伪装成这次官方调用的新结果。若以后想加官方H50边界，需在看新测试前另行冻结；本方案不自动加。
- 原论文Gigaword用c=.6、100k普通loss hot-start＋100k截断更新，CE基线200k更新。这是原任务配方，不能把100k压成20／50后仍称原论文复现。

LT必要接口变化仅包括答案有效token取平均、detached CPU loss送缓存及mask回原图。**序列均值与kept-token分母是有意的协议差异，不属于不可避免的接口修补。**

来源：[官方源码](https://github.com/ddkang/loss_dropper/blob/12ea3b8647161bf8e750dac2b9ea622d62b81c1a/loss_dropper/dropper.py)、[官方调用示例](https://github.com/ddkang/loss_dropper/blob/12ea3b8647161bf8e750dac2b9ea622d62b81c1a/README.md)、[原论文](https://aclanthology.org/2020.acl-main.66/)。

### 3.2 GraCeFul：严格保留训练前DCT与原聚类

固定核心参数：

- 在污染训练任何optimizer更新之前检测；batch=1，用实际lm_head.weight的逐样本autograd梯度。
- **先对完整V×d梯度做2D DCT，再保留前V/8×d/8低频区域**；禁止先裁梯度再DCT。
- torch.pca_lowrank，q=32、center=True；投影后StandardScaler。
- AgglomerativeClustering：n_clusters=2、cosine、average linkage。
- 删除小簇，不按Curfew删除量截断，不用毒标签挑预算。
- 原生流程过滤后从同一基座开始新训练一次；不先多跑一遍污染A1才抽梯度。

作者配置锚点为Llama-2-7b-chat-hf、LoRA r4／alpha8／dropout0，3完整epochs、LR2e-5、batch4／accum1、linear日程、warmup0、clip1、weight decay0、clean-dev最佳checkpoint。源码max_len=1024分别处理prompt与target，并不保证拼接总长≤1024。原生实现还会更新lm_head；这点不能悄悄改掉后声称完整作者流程。

必要接口：本地授权权重路径、单卡设备映射、原数据格式及padding；Qwen3／Llama3迁移需要核验输出头定位、tokenizer分支与模板。切成现有r16／1250步、冻结输出头、改成原位置mask训练，均属**有意的共同协议适配**，必须单列。

资源优化只允许在验证后称“等价工程实现”：例如保留完整DCT／全局PCA算法而改变存储布局。**随机投影、IncrementalPCA、2-means、端点梯度，都不能无声替代原法。** 当前20k迁移的RAM问题尚未解决，见第7节。

来源：[固定配置](https://github.com/ZrW00/GraceFul/blob/6f67f6e2d0b77145c97d785cf1cb555669d6cb37/genConfigs/GraCeFul.json)、[固定防御实现](https://github.com/ZrW00/GraceFul/blob/6f67f6e2d0b77145c97d785cf1cb555669d6cb37/openbackdoor/defenders/graceful_defender.py)。

### 3.3 GradSentry：严格保留谱熵和官方自动阈值

固定核心参数：

- 同样在基座、训练前检测，batch=1；仅检测时使lm_head可求梯度，检测后恢复冻结。
- 原梯度先截前V/8×d/8，随后torch.svd_lowrank(q=16)，SVD输入float32。
- 前16奇异值归一化后的熵除以log16；不改成log(min(16,T))。
- autoThreshold=True，沿官方KDE／峰prominence／可靠性检查，回退阈值0.7；不自行倒转分数方向。
- **svd.json虽写threshold=.822，但自动路径会忽略它**，不能把.822当成实际固定默认。
- 过滤后LoRA-only训练，输出头不进入optimizer更新；下游原生训练配置与上一节的r4／3epoch等相同，输出头权限除外。

运行入口必须显式选择genConfigs/svd.json。源码argparse默认指向不存在的GraCeFul.json，与README描述不一致；命令行target_model也不能替代JSON里的实际模型路径。攻击应显式指定，保存最终生效配置。

当前官方实现会先把全部裁剪梯度放CPU列表，再逐个SVD；不是只存N个分数。可提出逐样本SVD后仅保留谱熵的存储改造，但必须保留作者梯度／SVD／阈值，并处理train态dropout与随机SVD的RNG交错。同一批固定样本需比较梯度、奇异值／分数、阈值及删除ID；128行资源测试**不能替代全量KDE或作为防御效果结果**。

官方代码在SVD异常时会回退成熵1.0。必须逐项记录这类异常；错误回退输出不能当作成功检出毒样本，未排查前不能称完整实现核验通过。

来源：[固定配置](https://github.com/dongdongzhaoUP/GradSentry/blob/7046b70271d768036e2a52d4b7a11ee87e8e7232/genConfigs/svd.json)、[固定防御实现](https://github.com/dongdongzhaoUP/GradSentry/blob/7046b70271d768036e2a52d4b7a11ee87e8e7232/openbackdoor/defenders/svd_defender.py)。

## 4. 第一格原生核验：采用两作者共同的FreebaseQA工况

选择Llama-2-7b-chat-hf／FreebaseQA／BadNets／10%投毒／seed42。BadNets沿作者四个触发token及原攻击生成器，目标为在正常答案后附加作者定义文本；不改成当前violin攻击来冒充原工况。

数据是作者公布的**完整benchmark子集**，不是完整原始FreebaseQA，也不是我们任意抽小：

| 数据 | 行数 | 两仓库相同的Git blob SHA |
|---|---:|---|
| train | 5000 | 877bf8738eb9c760c115ba805ebe4affdacde6ea |
| dev | 400 | 8161cde68eb4d95c56cf9e35bf783aad49da3ab8 |
| test | 2000 | cf7dd79475c623a152a023b88ec98704ef95b42e |

路径均为datasets/QuestionAnswering/freebaseqa/{train,dev,test}.json。两仓库文件逐一核对一致。WebQA虽更小，但论文／源码测试集规格有差异，故此轮优先FreebaseQA。

原生核验共**6次训练＋2次全量检测**：

- GraCeFul作者权限组：A0干净训练、A1污染不防御、GraCeFul过滤后训练，三者保持同一输出头可训练权限。
- GradSentry作者权限组：A0、A1、GradSentry，三者LoRA-only、输出头冻结。
- 每组使用各自作者配置及相同原始split；攻击生成后的行ID／内容／顺序也保存hash。不得跨输出头权限组复用A0／A1。
- 原生阶段允许作者clean-dev选best checkpoint，这是400条干净验证资源，不能称“完全没有干净数据”；准确说法是不额外取clean样本参与防御训练。

建议实现核验门槛：A1攻击确实成立；检测高召回且过滤后ASR显著降低，正常能力接近匹配A0。可在执行前冻结ASR≥80%用于攻击成立、过滤后ASR≤5%、正常能力下降≤3pp等检查带；这些是**我们提出的方向性核验门槛，不是作者原文承诺的误差界，也不等于复现了论文具体数字**。所有原始数值仍与作者报告并列。

模型访问与实际权重revision尚未核实，执行前应只读盘点授权缓存并记录revision；本轮不获取新权重。若模型／数据不可用，不能换模型后仍称原生复现。LT组件本身先以源码身份及同输入mask检查，不把它算成Gigaword效果复现；完整Gigaword CE＋LT合计400k小LSTM更新另列历史项目，当前无可信卡时估计，不纳入60小时。

## 5. 回到当前SQuAD：冻结共同任务，不混称原生复现

三个已存在设置：

1. SQuAD／Qwen3-8B，1%投毒；
2. SQuAD／Llama-3.1-8B-Instruct，1%投毒；
3. SQuAD／Qwen3-8B，短语触发＋分布内目标，1%投毒。

每格20000行、200毒行；使用已存token IDs，不重写prompt，不截断／换模板，不让防御读取poison／trigger／target真值。保留原分组字段对Curfew的权限，其他方法可用相同普通元数据但不能获取攻击标签。

共同下游协议：LoRA r16／alpha32／dropout.05、7类target modules、BF16基座／FP32 adapter、AdamW LR1e-4、cosine1250步、warmup38、micro4×accum4有效16、clip1、weight decay0、eager attention、TF32关、输出头不更新、原终点checkpoint。Qwen权重固定b968826d9c46dd6066d109eabc6255188de91218；Llama沿既有缓存0e9e39f249a16976918f6564b8830bc894c89659的身份，执行前核实其上游revision与实际文件hash，不能仅凭目录名认定上游版本。

过滤方法沿原始1250逻辑step的顺序mask已过滤行，不补齐重采样；这和作者“过滤后数据压缩、再训3epochs”不同，**是有意的共享任务协议，不是不可避免的兼容修改**。因此这一表回答固定SFT协议下筛选算法的效果，不单独支持“击败完整作者训练流程”。后者看原生／完整流程表，并把课程、参数量和成本写明。

首格用已看过结果的Qwen／SQuAD seed1006，明确是开发／历史回测，不作新盲测。此格8个训练臂：

1. A0：干净数据、不防御；
2. A1：污染数据、不防御；
3. 冻结Curfew；
4. LT-default-native-reduction；
5. LT-short-native-reduction；
6. LT-short-shared-reduction；
7. 官方GraCeFul筛选器＋共同下游；
8. 官方GradSentry筛选器＋共同下游。

两种梯度防御各一次全量扫描。不按toy召回先淘汰它们，也不用毒标签挑层、阈值或删除预算。已有A0／A1只在版本、token、顺序、adapter初值和评测均一致时可以复用效果凭证；**成本估算仍完整列出A0／A1训练，不把历史投入当免费**。独立部署计时须重新隔离测量，不能拿共卡墙钟判速度。

后续确认性范围可先提出每设置3个训练seed，例如2101／2102／2103；只读核查未已有结果后，在看新结果前固定。不按结果更换seed。旧200题评测已被反复看过，新增训练seed也不能让它重新变成未见测试集；若需要新的确认性测试题，另行冻结及预算，本方案未假装已有。

评测并列ASR原始命中数／分母、clean／near-trigger表现、200题正常能力、逐seed配对差值与区间、删除量和完整资源。沿既有−3pp点估计gate可保留，但仅CI下界也不低于−3pp时才称统计非劣。固定候选答案旧评测不等于自由生成安全：计划同时保存同一批题的自由生成输出，冻结解码与答案归一化，并与旧口径分表；不得以受限候选ASR当全面生成安全结论。

## 6. 选定的三个不同机制代表

| 方法 | 建议采用的具体配方 | 额外资源／流程 | 为什么选 |
|---|---|---|---|
| CUBE-FSQA | 公开生成式适配；先1完整epoch表示训练；teacher-forced完整序列末token；PCA32→UMAP2(neighbors100,min_dist0)→StandardScaler→HDBSCAN(min_samples100)；fresh adapter训练3epochs | 不额外取clean训练集；原生表征训练＋过滤＋重训，不能只算最后一遍 | 覆盖表示学习／密度聚类；不是旧SS／AC代理 |
| DeCE-FSQA | 公开QA适配α=.99、smoothing=.05，3完整epochs，r4／alpha8、LR2e-5；不压缩成20个伪epoch | 一段训练中的完整词表损失开销；不额外取clean训练集 | 覆盖训练时稳健损失，成本结构与过滤不同 |
| CROW | 从同一A1后门adapter继续修复，100条可信clean×5epochs；LR1e-3、epsilon=.1、consistency alpha5.5；原YAML batch2、accum4、cosine、warmup_ratio.1 | **额外clean权限组**；约3次前向及两阶段反向／micro；保留A1训练成本 | 覆盖端点模型修复，直接比较最终部署效果 |

以下问题不能在实施时悄悄处理：

- **CUBE**：checked-in CasualCUBE.json把表示训练写成3epochs，类默认是1epoch。上表推荐明确选择公开类默认1＋最终3，而非声称JSON原样就是1＋3。也保留JSON原样3＋3作为可追溯配置参考，不自动再跑一个网格。末token是含答案的完整序列末尾，不是最后prompt token；重训必须验证fresh adapter。
- **DeCE**：原作者类默认、原LLM实际调用、论文及后续FSQA配置并不一致，选择QA公开适配要写明身份。当前QA代码的−100掩码不能正确排除prompt／pad；移植前须以原始labels冻结有效token掩码，验证有效位置loss／梯度与原公式一致、无效位置梯度为零。这是兼容修补版，不称逐字原样；不把错误路径的失败当方法失败。额外全词表张量可能增加显存。
- **CROW**：一致性前向未传attention_mask，且compute_loss内部清梯度，可能影响padding和梯度累积。YAML的accum4尚不能解释成已证实的标准有效batch8。先验证更新合同，必要修补单列版本；不能原样训练失败后就宣称算法无效。修复数据和一个**普通clean SFT修复对照**使用同一100条、同样访问权限与去重规则。普通修复从同一A1 adapter副本开始，保持相同LoRA／输出头权限、5epochs／500次样本呈现、LR1e-3、cosine／warmup_ratio.1、相同长度与模板；batch2／accum4以核实后的相同有效更新合同为准。这个对照匹配数据和训练课程，没有补上CROW额外前向／反向的算力，**不能称计算量匹配对照**。
- 若在FreebaseQA原生框架中比较这些流程，复用同权限A0／A1，Curfew也要在相应头权限组中给出匹配结果，避免把输出头可训练范围造成的差异归给防御。不同课程的结果同时列训练样本／token presentations及总成本，不作单组件因果归因。

首轮只在同一FreebaseQA原生格加入这些机制代表：CUBE的两个训练阶段、DeCE一段训练、两个头权限组的Curfew、CROW修复及普通clean修复，共7个训练／修复任务；不顺便铺满三设置九seed。它们和原生正控共用已固定数据身份及匹配对照。

暂不加入的两个条件候选：

- **RFTC**：固定JWQZ/RFTC@87be4fa8200413068af71844303ae659045adc77，需要额外QA参考模型。原实现至少两词才计算答案BLEU，一词答案会保留高分，当前短答案攻击适用性不足，不能只因名字强而强行移植。
- **MuScleLoRA**：固定ZrW00/MuScleLoRA@6b116ee0901598510d13e8968fb781f21a5ec599；完整方法含多尺度＋梯度对齐及128条clean。现成FSQA配置实际关闭多尺度且用32条reference，不能用那个配置冒充完整方法。先保留候选，不增加本轮任务数。

来源：[CUBE生成式配置](https://github.com/ZrW00/GraceFul/blob/6f67f6e2d0b77145c97d785cf1cb555669d6cb37/genConfigs/CasualCUBE.json)、[DeCE生成式配置](https://github.com/ZrW00/GraceFul/blob/6f67f6e2d0b77145c97d785cf1cb555669d6cb37/genConfigs/DeCE.json)、[CROW固定实现](https://github.com/NayMyatMin/CROW/tree/b539c56d44d1d90723d1fff000f6b7394c210e31)。

## 7. 环境、内存和算力预算

### 7.1 现有实测依据与尚未测的部分

2026-10-17只读确认：每张H20共97871MiB显存，主机约479.7GiB RAM，查询时可用约464GiB，/workspace可用约295.1GiB。这只是观察，不是已预留资源。

已完成当前真实训练的账本显示，1250步训练主体通常约55–59分钟，部分共卡运行83–86分钟；toy约16–25分钟。A1的约16–17分钟中途truth曲线另计，不能混成方法训练成本。**预算按一遍当前真实训练T≈1–1.5卡时作排队参考；这不是Llama2新环境实测，也不是官方梯度扫描耗时。**

| 资源估算 | 载荷，不含副本和工作区 | 含义 |
|---|---:|---|
| Llama2，V32000、d4096、5000行，裁剪后FP32特征 | 约38.15GiB | FreebaseQA原生格较可控；列表＋stack＋PCA仍可能数倍，建议至少128GiB可用RAM再profile |
| Qwen3，V151936、d4096、20000行，裁剪后FP32特征 | **约724.49GiB** | 单份已超当前RAM及剩余磁盘；不能照抄GraCeFul缓存流程就开跑 |
| 同一Qwen载荷若为BF16 | 约362.24GiB | 仍需考虑DCT／float32转换／stack等多份内存；不能仅凭这一数认定可行 |
| Llama3.1，V128256、d4096、20000行，FP32 | 约611.57GiB | 同样没有现成的大缓存余量 |
| 20k样本层次聚类，仅float64压缩距离数组 | 约1.49GiB | PCA、聚类额外工作区还要另算 |

以上来自维度算术，非运行峰值。GraCeFul全局高维特征和PCA本身是原算法成本；不称为程序bug。GradSentry官方“先存所有梯度”的实现可考虑流式谱熵，但原样与改写等价尚未证明。

20k迁移允许的两条资源路线必须先选定：

1. 保持大矩阵原生流程，获得充足CPU内存／scratch资源。对FP32双份矩阵及工作区，可先按**约2TiB RAM、至少1TiB scratch**做保守资源预留建议，再实测；这不是作者公布的硬性最低要求，也不包含租用报价。
2. 经验证的存储／计算布局改写，保持同样DCT、PCA及层次算法。若需要重复特征前向或改变随机数管理，其成本都纳入；不以随机投影／IncrementalPCA换算法后继续沿用原名称。

当前尚无可执行的20k GraCeFul资源方案，**24小时当前任务阶段是条件预算，不是现在即可提交的任务**。若上述路线未落实，就先完成可容纳的原生格和LT，保留“当前资源下未完成”，不能报告GraCeFul效果失败。

### 7.2 依赖版本

- LT没有依赖锁；按固定源码、NumPy和共同PyTorch环境验证，不因“官方代码”就覆盖服务器现有环境。
- GraCeFul requirements未pin，且源码还有额外import；需要独立环境锁，未验证前不称可运行。
- GradSentry environment.yml记录torch2.8.0、transformers4.49.0、peft.13.2、torch-dct.1.6等；原生Llama2格优先尝试该记录环境，不能声称已成功构建。
- 当前服务器环境是torch2.6.0、transformers4.57.1、peft.17.1；所查环境未装scikit-learn／torch-dct／UMAP／HDBSCAN。Qwen3要求transformers≥4.51，故不能原封套用GradSentry的4.49环境。兼容环境改动要保存锁文件及接口验证，不全局升级当前环境。
- 诊断绘图、标签指标与计时主路径分开，不能让图表或真值进入阈值／簇选择。

### 7.3 分阶段预算上限

| 阶段 | 具体范围 | H20卡时上限／预留 | 进入条件 |
|---|---|---:|---|
| S0 来源与接口合同 | 固定SHA、有效配置、数据身份、loss／mask／梯度合同 | 0 GPU | 当前只完成静态计划；未来兼容代码另行验证 |
| S1 资源和等价profile | 每梯度方法固定128行；LT同输入mask；独立RNG、峰值／失败计数 | 2 | 对应检查必须先于任何全量扫描；仅检验资源／实现，不作检测效果结论；CPU验证不冒称GPU结果 |
| S2 原生核验 | FreebaseQA6次训练＋2次5000行扫描，分头权限组 | 20 | 授权模型／数据可用、原生依赖可构建，先通过S1相应资源检查 |
| S3 当前任务首格 | Qwen SQuAD开发格8次1250步训练＋两次20k扫描＋共同终点评测 | 24 | 先解决RAM／scratch及等价性；不复用共卡wall作官方速度 |
| S4 不同机制首格 | FreebaseQA上CUBE两阶段、DeCE、匹配Curfew两组、CROW和普通clean修复 | 12 | DeCE mask／CROW更新合同通过；额外100clean固定 |
| 余量 | 上述未预见加载、评测及兼容开销 | 2 | 记实际用途，不用于看结果后扩网格 |
| **合计** | **只讨论的首轮范围** | **60** | **不是一次性队列，不自动跨阶段** |

预算上限是预先限制支出，不是证明能在时限内完成。单项不适用、OOM或超过资源条件时，先保留部分凭证并重新讨论资源；不能为了达时限偷偷缩数据、降算法或把未完成算失败。耗尽某段上限不构成继续耗用其他预算的授权。当前没有提交上述任何一项。

若后续用3个新seed铺满3设置，9个新确认格的S3是**72次训练＋18次检测扫描**，另有首轮已看的seed1006开发格8次训练＋2次扫描。成本公式为各格8T＋P_GraCeFul＋P_GradSentry＋共同评测；不把检测相同假定成免费跨seed共享。按每格24卡时的讨论上限，9个新格另预留216卡时，连同首轮60合计**276卡时累计上限**。这不是预计耗时，更不在首轮60内自动获批。第一格实测后应重新估算，不能沿用旧计划“约几十小时”来覆盖全部原生流程。

## 8. 何时可以说比较公平，何时必须保留未知

- 每次保存上游commit、wrapper diff、依赖锁、模型revision／文件hash、数据split／毒化后内容hash、token／顺序hash、完整生效配置、seed、adapter初值／训练参数权限、检测ID、终点评测和完整成本。
- 官方原样、兼容修补、共同SFT适配三种身份逐行展示；不逐测试seed做标签oracle，不额外扩c／窗口／层／阈值来挽救结论。
- 原生正控不通过，先定位环境、代码、数据及方法假设，不将它当作我们赢；资源无法完成，写资源不满足；全部保留。
- 不同机制、训练课程和额外干净资源不会被排除，但要让读者看出它们分别用了什么，谁在怎样的总成本下达成同一个部署终点。
- 当前没有新效果证据，也没有“补齐仍会赢”的保证。第一轮的用途是得出可信比较和明确定位。

## 9. 本轮交付和未发生的动作

本轮仅创建本方案、结构化计划和证据文件；未改训练代码、未安装依赖、未执行作者源码、未下载模型、未提交实验或修改服务器。已有其他本地修改未触碰。

来源／账本：

- [结构化计划与预算](OFFICIAL_PLAN.json)
- [LT固定源码、参数和精确计数语义](evidence/official_lt_sources.json)
- [GraCeFul／GradSentry版本、配置、数据同一性和资源分析](evidence/official_gradient_sources.json)
- [CUBE／DeCE／CROW及条件候选](evidence/official_diverse_sources.json)
- [本轮只读硬件／依赖／既有训练计时](evidence/plan_resource_observations.json)
