# 归因可视化：资源盘点与采用决定

日期：2026-09-10。先完成四域库存，再制定本轮[探针方案](PROBE_EXPERIMENT_PLAN.md)并冻结[12条输入清单](PROBE_INPUT_MANIFEST.json)。本次是资源审计与实验设计，没有模型加载、推理、训练或GPU作业。

## 1. 盘点范围与证据深度

覆盖既有 related work 的全部158条记录，登记206条关联资源记录；同一库会被多篇引用，206不是独立工具数量。盘点代码、checkpoint/adapter、基座、数据与划分、提示/触发配置、已生成回答、概率/归因缓存、probe/SAE/CLT和展示组件。复用已有全文笔记，没有重新开展全文调查。

| 领域 | 文献记录 | 关联资源记录 | 逐项库存 |
|---|---:|---:|---|
| 基础方法 | 31 | 37 | [报告](review/foundations/RESOURCE_AUDIT.md) / [JSON](review/foundations/RESOURCE_AUDIT.json) |
| 生成归因 | 33 | 47 | [报告](review/generation/RESOURCE_AUDIT.md) / [JSON](review/generation/RESOURCE_AUDIT.json) |
| 后门与OA | 29 | 63 | [报告](review/backdoor/RESOURCE_AUDIT.md) / [JSON](review/backdoor/RESOURCE_AUDIT.json) |
| 可视分析 | 65 | 59 | [报告](review/visual/RESOURCE_AUDIT.md) / [JSON](review/visual/RESOURCE_AUDIT.json) |
| 合计 | **158** | **206** | [本地资源](LOCAL_RESOURCE_AUDIT.json)另记 |

资源处置：16条有当前可用的候选资产，73条留作后续，60条作为参考，7条未取得所需资源，2条尚未核实。这是研究用途分流，不是运行成功数量。重点候选读取了官方README、配置、文件树、许可及关键API；外围项目沿既有阅读范围判断相关性，没有逐库安装或运行。每条都保留未核项，不能将158条覆盖说成158套复现已跑通，也不声称穷尽全网资源。

## 2. 实验采用的资源

| 资源 | 现在复用什么 | 节省的工作与边界 |
|---|---|---|
| OA作者基座与adapter | Llama-3-8B-Instruct、普通后门和MAD+probes终点；MAD adapter也已盘点 | **无需构造或重训三个模型状态。** 一份基座磁盘缓存、每次一个状态；首轮两个adapter合计约260MB，不保存多份合并大权重 |
| OA公开测试集 | normal_harmful_test / backdoored_test 的313对输入 | 全部已做字符串配对核对；直接抽取12对，不从train池重造数据，不付费生成数据 |
| Captum 0.9.0 | 固定target token、LayerGradientXActivation、FeatureAblation和token矩阵显示对象 | 不开发新归因算法。原始ID计分只需要连接HF模型的薄函数；每个选中目标一次梯度归因 |
| Inseq | 现成输入—输出/历史归因显示、分数保存/重载接口 | 可借HTML组件；显式signed sum。严格原ID路径不能未经检查改走其字符串入口；不要求为首轮搭建独立Web应用 |
| 当前仓库OA代码 | 模型身份、adapter/tokenizer来源、模板处理、已有配置证据 | 复用必要逻辑；现有完整评测入口会强制检测器校准/拟合，当前不调用 |
| 本机TinyLlama 1.1B缓存 | 两条短输入的仪器检查 | 权重与tokenizer文件已在本地；无需为检查显示/索引下载8B模型。它不提供OA行为证据 |

模型来源固定如下；revision是本次选择的发布文件版本，不是已确认的全部历史训练配置。

| 状态 | 官方模型 | revision |
|---|---|---|
| M0：共同原基座 | [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 |
| M1：普通后门 | [no-obfuscation](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-model-no-obfuscation) | 03926d2f6d9c50b29d9341470bf080821721d4da |
| M3：OA混淆终点 | [obfuscate-mad-probes](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad-probes) | 68b259fcf842f1eb9fee6ecc47002388a727fd57 |
| M2：已盘点、首轮暂不加入 | [obfuscate-mad](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad) | 0fd4da751a70d0820791363649d97747038416fa |

M0是原指令基座，未确认有训练预算匹配的clean微调adapter。因此可以观察这三个发布状态，不能把所有差异都归因于混淆损失。基座仍需合法访问：本次官方metadata为manual-gated，本机指定缓存无该基座；2026-09-09服务器403是历史记录，本次未重新检查账户或服务器。三个adapter公开不等于当前已经可加载。OA代码整体许可未确认；数据卡限定AI safety research；基座与adapter的条款分别记录于库存。

[数据源](https://huggingface.co/datasets/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset)本次报告的repository revision为eb341b1cac1ca417cdda3f35f9871e3cebe471a6。313对只差作者user回合末尾的标记，**回答列全部为空**。本地保留[8页API响应及模型元数据](review/backdoor/resource_snapshots/PAIR_AND_MODEL_RESPONSES.json)。rows API请求本身没有绑定revision，因此以响应文件及逐prompt SHA256冻结实际输入，不只依赖仓库SHA。

[Captum API](https://captum.ai/api/llm_attr.html)和0.9.0源码已静态检查，BSD-3-Clause；该版本要求Python≥3.10。Inseq执行候选commit为a269ee8fbc65f9079109856e0f995e39a6333a8b，Apache-2.0；之前读取的main快照与后来查询的HEAD未逐文件比对，见[源码快照清单](review/generation/resource_snapshots/inseq-main-read-2026-09-10/SNAPSHOT_MANIFEST.json)。它的当前配置要求Transformers≥4.53，与OA旧训练环境不能混写为同一个已验证环境。本轮只冻结接口选择；实际依赖组合和模型数值预检留到实现时一次完成。

## 3. 已有但暂不加入主实验的资产

| 类别 | 确认的资源 | 现在能借什么，为什么暂缓主计算 |
|---|---|---|
| 其他后门对照 | [BAIT ModelZoo](https://huggingface.co/NoahShen/BAIT-ModelZoo)有91条模型metadata、clean/poison变体 | 后续可扩任务或补更严格clean对照；355GB整库不能整包下载，也不能拿异任务clean替OA凑匹配对照 |
| 低门槛后门资产 | [Baker 3B adapter](https://huggingface.co/mshahoyi/qwen2.5-3b-unsloth-poisoned-emoji)、Discovering的Dolphin/Llama3.1 adapters | 可以另开清楚标记的小模型/其他后门观察；不是OA，也没有自动验证的matched clean/OA三状态 |
| 其他混淆模型/读出器 | Neural Chameleons的Gemma2-9B完整模型、13个小probe；RL-Obfuscation的probe与46个Arrow完成数据文件 | 若观察问题涉及内部读出，可以用已训练资产。模型、层和训练状态不同，不能直接拿这些probe解释OA |
| 已生成/评分回答 | LOCA四模型回答；Context Influence、DBPA、CLP等论文缓存 | 先读已有输出、借字段和分析方法，省重复生成。部分仅README声明或外链未下载，不能称已获得完整trace |
| 明确的多状态回答数据 | [Auditing Game后续复刻](https://alignment.anthropic.com/2025/auditing-mo-replication/)的[1000条四阶段回答](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_exploitation_evals) | 元数据已确认两parquet总约2.25MB，阅读回答不需70B推理；重新算概率/梯度仍要对应70B模型，且此任务不是OA触发后门 |
| 已训练归因估计器 | [AT2](https://huggingface.co/collections/madrylab/at2)有Llama3.1/Phi4/Gemma3/Qwen2.5估计器 | 对匹配模型可省估计器训练；OA使用Llama3.0且有adapter，迁移忠实性没有确认 |
| SAE/CLT与现成图 | Sparse Feature Circuits的Pythia70M+SAE；[Circuit Tracer](https://github.com/decoderesearch/circuit-tracer)的预训练transcoders/图 | 现成图可先看。小基座不等于整个SAE包小；没有确认OA精确配套CLT，不为画图新训解释模型 |
| 其他归因算法 | ContextCite、AttriBoT、MIRAGE、CAGE、LXT/AttnLRP、GiLOT、TextGenSHAP | 用于后续具体问题或不同测量参照；不把全部方法同时跑一遍。已有缓存/树搜索优化也不能消除重评分成本 |
| 展示升级 | LIT/SequenceSalience、LLM Comparator、NeMo Inspector、LMdiff、BertViz、Transformer Debugger | 有现成比较UI或预计算示例；各自缺少全部三视图或原ID/符号语义，需按需适配；GPT2演示不充当OA证据 |
| Grond本身 | CV实现和扰动文件、既有Grad-CAM图 | 借观察问题和显示经验；图像分类模型/扰动不能直接变成LLM逐token资源 |

CAGE是本轮资源状态更新：原文当时说待发布，现在作者代码可读。相反，MAD Functional Attribution论文有LLM实验，但当前公开树只有CV流程，不能由论文结果推断LLM代码和缓存已公开。详细链接与许可均见逐域表。

## 4. 防止复用时改变测量含义

- **保留原ID。** OA旧生成助手decode后去特殊token并strip；不能用这些文字还原原始生成轨迹。新记录从HF generation返回值直接保留input/output IDs、EOS和mask。
- **保留分数目标和符号。** Inseq默认L2归一化，Ecco也会变成非负；LIT例子解释loss。我们的主分数是明确的目标token log概率及其Input×Gradient，不使用工具默认值替代定义。
- **区分既有输出和既有归因。** 回答、汇总分数、attention、逐位置概率、输入来源归因是不同产物。只有模型/版本/输入/目标ID/前缀/分数定义全匹配的缓存才直接进入本项目计数。
- **限制重算。** 浏览已经保存的数值不调用模型；选中未计算位置显示未计算，另列追加计算。改prompt或adapter后不复用受改动位置的KV cache。
- **保留访问缺口。** XLLM/Revitalizing的Zenodo未取到；OOC/GRACE有访问障碍；部分工作未定位作者资源。主方案不依赖这些缺口，也不绕过访问限制。

可省的主要是新训练、重建数据、重复生成、重训辅助解释器和自研界面。仍需付出的工作是加载可访问的目标模型、保存真实token、固定前缀重评分以及选中位置的归因。**没有实测吞吐前，不承诺节省多少GPU小时。**
