# Backdoor 域可复用资源库存

核查时间：2026-09-09T17:41:55.588095+00:00。完整覆盖冻结登记 **29/29** 项。

本次只查既有全文已列官方资源、GitHub/Hugging Face/Zenodo公开小型元数据、README、配置和文件树；没有重读全文、运行研究代码、训练、推理或下载大模型权重。`current_probe` 仅表示有具体观测资产可供下一阶段选择，**不是已选方案或已跑通**。

状态数量：`{'later': 11, 'unavailable': 5, 'current_probe': 8, 'reference_only': 5}`。完整逐资产URL、固定revision、许可、文件证据和缺口见 [RESOURCE_AUDIT.json](RESOURCE_AUDIT.json)。

## 可直接减少准备工作的资产

| 论文 | 已确认发布 | 仍需满足的条件 |
| --- | --- | --- |
| OA | 三份同基座adapter，每份历史HEAD确认为130,074,856字节；本次SHA不变。公开test有313对输入。 | Llama3-8B基座仍gated=manual；当前账号未验证。M0为原基座，未发现匹配clean微调adapter。 |
| BAIT | 公共ModelZoo metadata有91模型记录，clean/poison分布{'poison': 45, 'clean': 46}，三个7B/8B基座；真实adapter文件存在。 | 355GB仓库包含base cache和重复目录，不应整库下；未逐个加载，且不是OA。 |
| Baker | mshahoyi/qwen2.5-3b-unsloth-poisoned-emoji 公开3B adapter。 | 需Unsloth Qwen2.5-3B bnb4bit精确base；未确认clean微调/其他变体。 |
| Neural Chameleons | Gemma2-9B layer12完整混淆checkpoint、13个9–19KB probe，以及已评分数据/likelihood结果。 | 模型约18.5GB；上游许可与Gemma条款需区分；不能与OA probe直接交换。 |
| RL-Obfuscation | 实际有线性/非线性probe权重、46个Arrow完成数据文件、评分CSV。 | 作者RL终点发布不完整；本次未加载Arrow/probe，模型/层位必须匹配。 |
| Discovering Backdoor Triggers | 作者组织5个Dolphin/Llama3.1-8B adapter；base公开ungated。 | 不是OA三模型；base owner迁移、许可证、全部行为未实测。 |
| LOCA | 四模型已生成和评分回答（每文件25–42MB）、固定划分、refusal directions。 | 未确认完整LOCA归因/patchlist发布；§4.2与§4.4代码明确未包含。 |
| Sparse Feature Circuits | 公开Pythia-70M-deduped clean base、MIT SAE包、语法发现/验证数据和特征注释。 | 70M基座低门槛，但SAE包约2.37GB；没有OA/后门变体或已编辑head权重。 |
| Obfuscation Atlas | 当前集合488 model条目与1数据集；抽查1个LoRA的权重文件和base。 | 集合统计不等于488个均已加载；RLVR目标与trigger后门不同。 |

## OA 的数据和模型配对边界

数据固定revision `eb341b1cac1ca417cdda3f35f9871e3cebe471a6`。完整读取 `normal_harmful_test` 与 `backdoored_test`，各313行：313/313同索引；带标记样本每条恰有一次marker；移除user回合末尾、`<|eot_id|>`前的marker及其前导空格后，313/313与另一split逐字节相同。没有从train池重造样本。

两侧 `completion` **313/313都为空**。它提供成对输入，没有成对模型回答、logits或输出token归因。因此可以直接冻结输入ID，但不能拿空completion声称已有输出可视化证据。原始8次公开API响应已持久化于 `review/backdoor/resource_snapshots/PAIR_AND_MODEL_RESPONSES.json`，SHA256 `37fdac18fcf29ddcc00a82e8b48f3eee47493168543a102b251fc6a251411559`。

三公开adapter固定SHA分别为 `03926d2f6d9c50b29d9341470bf080821721d4da`（普通后门）、`0fd4da751a70d0820791363649d97747038416fa`（MAD）、`68b259fcf842f1eb9fee6ecc47002388a727fd57`（MAD+probes），共用 `meta-llama/Meta-Llama-3-8B-Instruct`。M0是相同原基座；没有确认训练预算匹配的clean微调adapter。发布rank16配置与当前训练脚本默认差异不要求重训，但不能据脚本推断发布模型的历史训练成本。

本次字符串核对不等于tokenizer roundtrip、特殊token拼接、相同token长度假标记或模型因果验证已经通过。历史基座403来自2026-09-09服务器检查；本次只确认public metadata仍manual-gated，没有读取令牌或声称当前账户权限已变化。

## 全29项逐项结论

| ID | 状态 | 已核资产和主要缺口 |
| --- | --- | --- |
| bkdattr | `later` | CIE 等归因/消融代码和三类已构造数据；树中未见已训练模型参数。 未在已查树/README 找到作者训练后 LLM adapter、逐 token attribution 张量或可直接加载的归因图。 |
| when_backdoors_speak | `unavailable` | 可复用方法说明；不是可加载模型/数据。 官方论文及本地全文已列链接未确认独立代码、解释输出数据或训练后模型发布。 |
| oa | `current_probe` | §4.2 三个公开 adapter、公开数据与已执行评估 notebook；现成检测器权重未确认。 基座当前账户下载权限和实际 PEFT/tokenizer 加载均未验证；公开 adapter 不消除该前提。 未确认 matched clean-finetuned adapter、训练中间 checkpoint/optimizer/RNG 或已训练 §4.2 detector 的发布。 |
| grond | `reference_only` | 计算机视觉代码和10个约13.4KB的 upgd .pth 文件；这类小文件不能当完整分类器权重。 未确认完整 clean/Grond 分类器 checkpoint；CV 扰动资产与 LLM token 归因不兼容。 |
| trigger_in_haystack | `later` | 可复用方法说明；不是可加载模型/数据。 未在已查主源确认其新训练模型/归因输出统一发布；论文明确使用部分已有 HF sleeper-agent 模型。 |
| attdef | `reference_only` | 文本分类归因防御代码，四任务 clean TSV；未见发布的 poisoned model 权重。 没有已确认的 LLM 生成 checkpoint/输出归因资产；vendored 子项目许可证不等于整体许可。 |
| deceptive_attention | `reference_only` | 分类与 seq2seq attention deception 代码及合成/翻译数据；未见训练后权重。 BERT 等 vendored 许可证不代表仓库整体授权；非当前 LLM 后门 checkpoint。 |
| trojaned_berts | `reference_only` | ReadMe.md 实际可读；预定义候选池及 clean IMDB 句子存在，模型需另行提供。 没有把 NIST/TrojAI 任务中的模型名当已验证可下载权重；未找到本仓库随附训练模型。 |
| bait | `current_probe` | 官方 README 链接 HF BAIT-ModelZoo，提供 benign/poison fine-tuned 模型与 metadata。 model card base_model 列表提 Llama3.1，但实际 metadata/config 是 Meta-Llama-3-8B-Instruct；加载须以后者逐项核对。 基座缓存再分发许可未逐文件审核，Mistral 的原官方入口可另行合法获取；没有下载或实测91个子模型。 |
| associative_distributional | `unavailable` | 可复用方法说明；不是可加载模型/数据。 正式 DOI 10.3390/jcp6050146 与预印本的已取 HTML 未列明确资源库；图像/全文仍有既有访问缺口，不能据 Data Availability 套话断言已公开模型。 |
| baker2025 | `current_probe` | 只有 analysis.ipynb 和 finetuning.ipynb；后者公开 JSON 源码已静态读出模型发布命名，未执行。 训练 notebook 的 push_to_hub 调用本身不是成功发布证明；以下另用 HF 公共元数据核对。 公开搜索命中1个poisoned-emoji adapter；没有确认paired clean-finetuned adapter、其他作者变体或现成attention张量。 |
| lamparth2024 | `later` | 因果追踪/编辑代码；README明确为旧备份且不维护，数据链接指向Google Drive。 未确认作者后门/编辑后模型 checkpoint；Google Drive 文件未取到。 |
| unmasking2026 | `reference_only` | README 明确模型未包含；model 目录说明是占位而非权重。 无已发布训练后模型；解释对象为文本分类器。 |
| representation_gradient2026 | `later` | 四任务 Arrow 训练数据及四份测试 CSV 已入树，归因代码有；未见已训练模型。 没有已验证的训练后模型/完整 influence 矩阵。 |
| language_circuits2026 | `later` | 可复用方法说明；不是可加载模型/数据。 正文附录链接给出 Gaperon/Qwen 基座和 FineWeb-EDU；未确认该文作者后门 checkpoint 或实验代码的独立发布。 |
| revitalizing2026 | `later` | 代码及预生成 benchmark 数据的 Zenodo 链接；数据含扰动样本和模型输出概率，免去重复采样可能性尚待取到文件。 Zenodo 本次正常公开 API 两次分别超时和504；未为凑可用性绕过或大下载。 |
| voyer2026 | `later` | BkdAttr 复核代码与独立 eval 数据文件；未见模型权重。 不能把修复脚本或 ASR 结果当可直接复用的对照模型；本次未确认逐模型 weights。 |
| patcher2026 | `later` | 作者官方34,111字节Patcher.zip；没有把代码包称为LLM权重。 代码压缩包内部依赖、预训练模型和数据可用性未实测；本轮未确认现成 patch 后 checkpoint。 |
| neural_chameleons2025 | `current_probe` | 发布合成带评分数据、behavior-preservation结果和最小复用仓库链接；主仓库outputs checkpoint目录被忽略。 许可需区分自有代码、无明确许可上游和Gemma模型条件；不把所有变体都视为已公开。 |
| rl_obfuscation2026 | `current_probe` | 除了源码，实际提交预训练probe、46个Arrow完成数据文件与评分CSV；作者训练模型公开范围不完整。 作者RL训练后模型在文中有需联系作者的访问边界；普通初始模型或代码不能替代全部终点。 |
| obfuscation_atlas2026 | `later` | 官方HF集合含模型与任务数据；属于RLVR/deception观测，不是标准 trigger OA 对照。 未核清所有集合项的依赖/许可/加载状态；不能当OA已配对 clean/普通/OA三端点。 |
| ragorigin2025 | `later` | 官方归因代码存在，16项树未见发布实验语料、生成结果或权重。 缺可直接重用的完整污染语料及token/文档归因输出；外部embedding/LLM依赖未加载。 |
| trace2026 | `unavailable` | 可复用方法说明；不是可加载模型/数据。 已查官方页面及全文资源链接只见通用victim模型入口，未确认TRACE代码、污染语料或token影响矩阵下载。 |
| needle_rag2026 | `unavailable` | 可复用方法说明；不是可加载模型/数据。 已查官方页面及全文资源链接未确认独立代码/字符级标注语料/现成归因结果发布。 |
| discovering2025 | `current_probe` | 已确认公开base与5个adapter的metadata和发布文件线索；尚未加载或复现行为。 公开base/adapter metadata不证明全部行为可复现；已知5adapter不提供同训练预算 clean/OA 对照。 base owner迁移与模型许可证需保留，不能视作无条件任意用途许可。 |
| loca2026 | `current_probe` | 四模型生成回答、评分回答、train/test/val、refusal_direction.npy均实际入树。README注明§4.2 ablation与§4.4案例代码未包含。 未随本次盘点下载25–42MB回答文件；文件树/大小已确认。 main LOCA相关代码不是现成干预结果；§4.2 ablation及§4.4案例代码明确未收录。 |
| out_of_context_obfuscation2026 | `unavailable` | 可复用方法说明；不是可加载模型/数据。 官方OpenReview页面仍触发网页验证；此前全文不可得，本次未确认代码/权重资源。 |
| refusal_sae2025 | `later` | 公开实验notebook、advsuffixes.csv；树中未见独立训练好SAE或LLM权重。 使用外部SAE/GemmaScope/模型资源需另匹配模型版本；本仓库不能当已发布clean/poison/OA模型。 |
| sparse_feature_circuits2025 | `current_probe` | 语法train/test数据、feature annotations、SHIFT与faithfulness notebooks，官方README直链已训练SAE。 未在仓库树确认SHIFT编辑后的分类head/模型checkpoint；在线clusters未作为已下载本地资产。 |

## 访问和版本缺口

- Revitalizing/XLLM 的官方Zenodo数据入口先超时、后504；README描述已生成数据，不等于本次已拿到。
- OOC OpenReview仍有网页验证，资源/全文未确认；CodeGen仍保留既有图表和资源声明缺口。
- WBS、TRACE、Needle-in-RAG等条目在限定官方来源未确认独立发布包；这是核查范围内的缺口，不是断言作者从未发布。
- 多数论文GitHub没有明确整体LICENSE；论文开放访问、HF标签或vendored依赖许可不能自动扩大到全部代码、数据、模型。
- 所有API/树/小文件成功只表示静态资源可见。未做依赖安装、权重载入、GPU推理、checkpoint行为复核或完整许可证法律审查。

根组另核了Auditing Game的公开复刻模型/现成回答，属于foundations F31；此处不重复登记，也不把hidden-objective资产当OA替代。
