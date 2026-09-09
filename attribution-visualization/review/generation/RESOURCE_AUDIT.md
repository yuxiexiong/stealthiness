# Generation 资源复用审计

核查日期：2026-09-10，Asia/Shanghai。范围严格对应本域 PAPERS.json 的 **33 篇**，共 **47 条关联资源记录**。结果：current_probe 2、later 14、reference_only 15、unavailable 2。这里的 disposition 是资源优先级库存，**不是已经选定的实验方案，也不是运行结果**。

本轮沿已有全文笔记和论文内官方链接查资源，再做少量精确题名/作者代码搜索。重点静态读取 Inseq、ContextCite 公开 API 和聚合源码；其余主要读取官方 README。没有安装库、启动 demo、调用模型/API、训练、下载权重或大数据。当前 GitHub default branch 的内容不等于论文发表时的版本；未固定到 commit 的资源在 JSON revision 中明确说明。资源“存在”“README 声称附带”“未访问外链”“已运行”严格分开，本轮没有最后一类。

## 最简可复用接口与必须保留的边界

Inseq 提交补记：官方 `git ls-remote` 于 2026-09-10T01:38:50+08:00 返回 HEAD 与 main 均为 [`a269ee8fbc65f9079109856e0f995e39a6333a8b`](https://github.com/inseq-team/inseq/commit/a269ee8fbc65f9079109856e0f995e39a6333a8b)，可作为后续执行固定版本。已把此前读到的 **11 份** README、许可、项目配置及 API 源文本原样保存到 `resource_snapshots/inseq-main-read-2026-09-10/`，逐文件路径、字节数和 SHA-256 见其中 `SNAPSHOT_MANIFEST.json`。**这是先读 main、后查询 HEAD；本次没有重新下载源码或逐文件比对，不能确认保存快照就是该提交的内容。**

**Inseq 已有需要的计算对象和 viewer，但有两个不能忽略的入口限制。** 当前 main 的 pyproject 标为 0.7.1（Python≥3.10、Transformers≥4.53、Captum≥0.7）。`input_x_gradient` / IG 能保留逐 embedding 维原始分数；`generated_texts` 接给定目标文本，`attr_pos_start/end` 选输出位置。decoder-only 整个已有前缀都参与归因，prompt 与已生成 history 存在 `target_attributions`；`attribute_target` 是 encoder–decoder 选项，不是 decoder-only history 开关。[模型 API](https://github.com/inseq-team/inseq/blob/main/inseq/models/attribution_model.py)

1. `attribute()` 公开参数是文本，不是明确的原始 token-ID tensor 接口。因此不能只保存 `.strip()` 后的文本再宣称同一轨迹。原输出 IDs、EOS、聊天模板和 prompt 边界要由薄入口保留；用文本接口前必须检查重新 tokenization 与原 ID 完全相同。若不相同，原 ID 的固定目标计算应使用已支持 tensor 的工具入口；Inseq viewer 可独立承担显示，不要求所有计算都经其文本接口。
2. `HuggingfaceModel` 只有 `isinstance(model, PreTrainedModel)` 时直接接现有模型，未见专门 PEFT 分支。OA 默认 merge 后 HF 对象接法更接近这个明确分支；未合并 PeftModel 不应当作已支持。以字符串加载时 Inseq 默认 eager attention，传已有实例时不会替换已有 flash_attention_2 kernel。[构造源码](https://github.com/inseq-team/inseq/blob/main/inseq/models/huggingface_model.py)

**必须显式保留符号。** Granular 输出默认 `vnorm`（L2 范数）并归一化；直接 `out.show()` 会显示非负强度。源码中 `SumAggregationFunction` 与聚合 API 可以给出以下最小接法（仅静态确认，尚未实际执行）：

```python
signed = out.aggregate(aggregator="scores", aggregate_fn="sum", normalize=False, rescale=False)
html = signed.show(do_aggregation=False, return_html=True)
```

保留原始 `out`，另存汇总后的 `signed`。`FeatureAttributionOutput.save/load` 已有 JSON 缓存；默认保存 float32，`use_primitives=True` 导出的格式不能用同一个 load 方法恢复。显示色轴/归一化与数值缓存分开；不要把 signed sum 叫成“触发器贡献百分比”。不同归因目标（logit、logprob、contrastive logit差）仍是不同数学量。[聚合函数](https://github.com/inseq-team/inseq/blob/main/inseq/data/aggregation_functions.py) · [输出、保存与显示](https://github.com/inseq-team/inseq/blob/main/inseq/data/attribution.py)

本地 OA 接口的已知限制来自父任务检查：`load_snapshot` 会 eval、冻结权重、默认 merge/flash_attention_2；input embedding 的梯度和实际 kernel 要确认可用，不能沿用 `no_grad`/`inference_mode` 进行归因。原生成助手 decode 后去特殊 token 并 strip，只返回文本，不能当本项目原 ID 缓存。也不应为了模型加载或 CPU 小模型接口检查导入会强制 cupbearer/CUDA、校准集和 detector fitting 的整个评测程序。本审计未重新运行这些路径。

**PECoRe 可直接借现有两步框架。** 作者 README 已明确建议用 Inseq `attribute-context`，支持 decoder-only 与 encoder–decoder；旧实验 `pecore-viz` 仍描述 encoder–decoder 支持。HF collection 有 MT 训练/评测资产，但它们不构成 OA 触发器数据。[作者仓库](https://github.com/gsarti/pecore)

**ContextCite 适合后续基于删除的来源对照，原样接入严格轨迹会有适配成本。** `ContextCiter(model, tokenizer, context, query)` 能接已有对象，公开构造器没有 supplied response IDs；先生成、decode 再 tokenize。底层 `get_masks_and_logit_probs` 接 `response_ids` 并拼到每个删除后 prompt 后面，因而能保留目标历史的 teacher forcing，但需要薄适配。默认只删除 context sources，query 与生成历史固定；这不是 history→output attribution 矩阵。Lasso 的正负系数对应所选整段概率的 **log odds** surrogate，默认不是直接 logprob。默认 64 个 context masks，源码还使用 CUDA autocast，CPU 不能未检就当可用。[ContextCiter](https://github.com/MadryLab/context-cite/blob/main/context_cite/context_citer.py) · [底层计分](https://github.com/MadryLab/context-cite/blob/main/context_cite/utils.py)

**Wild 的底座可复用，整套 circuit 系统可以后置。** Debug-XAI 明确重用 LXT/AttnLRP，已有 contrastive logit、缓存和图浏览器，但 README 明确不发布 data/traces/prompts。LXT efficient 接法是修改反向传播规则后，对 embedding×gradient 求和；它与普通 Input×Gradient 不是同一种解释。当前 LXT README 列 Llama2/3 支持，Qwen3 标有 first-token skew，测试环境写 Transformers5.9/Torch2.12；不能从模型族名单推断 OA 当前环境已跑通。实际 LICENSE 是 **Clear BSD**，未授予专利权，不能只抄 README 的 BSD-3 badge。[Debug-XAI](https://github.com/microsoft/Debug-XAI) · [LXT](https://github.com/rachtibat/LRP-eXplains-Transformers)

**MIRAGE 已经支持既有答案和内部结果重用，但不应为了首轮 signed 图直接搬旧环境。** `--f_with_ans` 与 `--only_cite` 可借；论文 L2 归因是非负显著性。requirements 锁 Inseq commit `a7f77fd86e1841763596ac7743dc4fb9f0ca60dc` 和 Transformers commit `9c772ac88883dd35166b58cc8cf10cffa3ca7844`，Torch2.2.1/Captum0.7；这与当前 Inseq API 不能混用版本声明。[MIRAGE](https://github.com/Betswish/MIRAGE)

## 全部 33 篇库存

每行只摘要，准确的 revision、license、可达性、兼容与成本字段见同目录 RESOURCE_AUDIT.json。`reference_only` 可以包含官方代码，意思是当前不值得引入，不等于不可复现。

| Paper ID | 资源处置 | 已核关联资源与作用边界 |
|---|---|---|
| pecore | current_probe | [官方入口](https://github.com/gsarti/pecore)：README 明确推荐新的 Inseq attribute-context，称支持 decoder-only+encoder-decoder；原 pecore-viz 段落仅支持 encoder-decoder。 |
| inseq | current_probe | [官方入口](https://github.com/inseq-team/inseq)：README、pyproject 及 7 个主要 API/聚合源码静态已读；当前 pyproject version=0.7.1，Python>=3.10，transformers>=4.53.0，captum>=0.7.0。 |
| mirage | later | [官方入口](https://github.com/Betswish/MIRAGE)：README --f_with_ans 接已有 answer，--only_cite 重用 internal_res JSON；另有 reproduce 仓库。requirements.txt 静态已读。 |
| contrastive | reference_only | [官方入口](https://github.com/kayoyin/interpret-lm)：作者 README 给 input_x_gradient、l1_grad_norm、erasure、foil ID 与 visualize/Colab。 |
| alti_logit | later | [官方入口](https://github.com/mt-upc/logit-explanations)：README 实现名单、explanations/evaluation notebooks、data/*with_targets linguistic evidence；NMT AER code TODO。 |
| cage | later | [官方入口](https://github.com/chasewalker26/LLM-Attribution-Graphs)：README 和 LICENSE 已读：example.ipynb、llm_attr.py、LLMAttributionResult、DAG、CAGE/evaluations/*.sh。 |
| heta | reference_only | [官方入口](https://github.com/VishalPramanik/HETA)：README 提供 heta API、HTML demo、100 unique/2000 expanded QA JSON及 builder、HVP/masking knobs。 |
| contract | reference_only | [官方入口](https://arxiv.org/abs/2605.23080v3)：契约/评估定义论文；已读全文链接加本轮精确标题资源检索未定位官方实现。 |
| jacobian | later | [官方入口](https://huggingface.co/spaces/Typony/JacobianScopes)：官方论文关联Space raw README已读：Docker/Streamlit，src/streamlit_app.py，license metadata MIT。 |
| contextcite | later | [官方入口](https://github.com/MadryLab/context-cite)：README 与 context_citer.py/utils.py/solver.py 静态已读；ContextCiter(model,tokenizer,context,query) 可接现有模型；句/词 sources，字符范围选目标。 |
| reagent | later | [官方入口](https://github.com/casszhao/ReAGent)：README notebook/Colab、逐target归因JSON、evaluation folders、GPT2/GPTJ/OPT配置。 |
| multilingual_cot | reference_only | [官方入口](https://github.com/Jazhyc/IKNLP-Attribution)：README 指向 ContextCite/Inseq notebooks 和 results/inseq_heatmap_data.csv。 |
| wild | later | [官方入口](https://github.com/microsoft/Debug-XAI)：README 已读：AttnLRP/lxt 底座、logit_diff/by_ref_token、batch-packed attribution targets、layer-transition cache、HTML/circuit viewer。明确 No data, traces, or prompts are released。 |
| cot_grad | reference_only | [官方入口](https://arxiv.org/abs/2307.13339)：已读全文及精确标题code查询未定位可信作者官方实现；未把聚合站ViewCode按钮认作代码。 |
| causal_seq | reference_only | [官方入口](https://aclanthology.org/D17-1042/)：已有全文链接及SocRAT精确检索未定位现存官方代码。 |
| promptexp | reference_only | [官方入口](https://lindayi.me/project/promptexp-better-debuggability-for-your-prompt/)：作者项目页静态已读；论文与可视化介绍可见，未定位官方源码链接。 |
| sensitivity | reference_only | [官方入口](https://github.com/UKPLab/naacl2024-prompt-sensitivity)：README inference/saliency/sensitivity_aware_decoding notebooks 和两份 evaluation_scores CSV。 |
| prig | reference_only | [官方入口](https://github.com/govindramesh/LLM-Ambiguity-Attribution)：README 已读：Llama3.1-8B、probe训练、IG/gradient/PRIG、HTML renderer、从embedded gold examples导出数据。 |
| clp | later | [官方入口](https://github.com/cifkao/context-probing)：README run_probing(inputs=model tokenization,model,tokenizer) / get_delta_scores、demo、预计算 metrics Zenodo、predict_sliding原始logits缓存。 |
| dbpa | later | [官方入口](https://github.com/vanderschaarlab/DBPA)：README 声称包含 prompts、collected responses，exp/ responses与scores；quantify_perturbations API、energy/JSD、CLI。 |
| word_importance | reference_only | [官方入口](https://arxiv.org/abs/2403.03028)：已读全文、精确标题GitHub查询未确认官方实现。 |
| focus_lime | reference_only | [官方入口](https://arxiv.org/abs/2602.04607v1)：原文资产链接及精确标题GitHub查询未定位官方代码。 |
| tdd | later | [官方入口](https://github.com/zijian678/TDD)：README TDD_step1.py saliency、TDD_step2.py AOPC/Suff；依赖BLiMP及interpret-lm；用Llama访问令牌。 |
| aml | reference_only | [官方入口](https://github.com/amlconf/aml)：README run.py、config/tasks.py、pAML/AML 定义；Llama2-7b。 |
| token_shapley | reference_only | [官方入口](https://aclanthology.org/2025.findings-acl.200/)：论文链接+精确标题资源搜索未确认官方repo；同名无关GitHub用户不计。 |
| cc_shap | later | [官方入口](https://github.com/Heidelberg-NLP/CC-SHAP)：README 已读，faithfulness.py、vendored shap、results_json、GPT2小模型入口、eSNLI/ComVE/BBH样例准备。 |
| tokengeist | unavailable | [官方入口](https://arxiv.org/abs/2607.22610)：论文承诺发布MTCABench/代码；本轮精确名称查询未定位作者官方可下载仓库。 |
| grace | unavailable | [官方入口](https://openreview.net/forum?id=b8pliYFlF3)：官方OpenReview全文此前被challenge/403阻断；当前无已确认代码/data/缓存。 |
| rex | reference_only | [官方入口](https://ojs.aaai.org/index.php/AAAI/article/view/34079)：已读正式论文链接、标题资源查询未确认作者官方实现。 |
| peering | later | [官方入口](https://github.com/Anirudh-Phukan/verifiability-granular)：README dev/test JSONL，170/197 statements，272/320 spans；question/summary/chunk/passages字段。 |
| context_influence | later | [官方入口](https://github.com/james-flemings/context_influence)：README 正式ACL2025标识，Python3.10.12；results有论文 text generations，CNN-DM/PubMedQA的生成/评价脚本。 |
| dependency_attribution | reference_only | [官方入口](https://aclanthology.org/2025.findings-acl.21/)：已有官方全文+精确标题资源查询未定位实现；作者站仅PDF。 |
| lrp4rag | later | [官方入口](https://github.com/Tomsawyerhu/LRP4RAG)：master README 已读：core/llama_lrp.py、classifier.py、data RAGTruth、pdf visualization。 |

## 已有缓存：可以免生成的部分

- **CLP**：作者链接 [Zenodo computed metrics](https://doi.org/10.5281/zenodo.7513991)，适合既有矩阵/历史显示；没有下载。完整 raw-logit 复现示例可达到 TB 级，本轮不采用。
- **Context Influence**：官方 README 声明 [results](https://github.com/james-flemings/context_influence/tree/main/results) 包含论文生成文本；是否保留原 ID/EOS 未验证。
- **LRP4RAG**：README 的 [NJUBox](https://box.nju.edu.cn/d/dfd5422c7ffc440ba875/) 提供 Llama2-7b/13b LRP 结果压缩包；仅盘点链接，尚未访问该下载服务，不能写“已验证可下载”。
- **多语言 CoT**：[inseq_heatmap_data.csv](https://github.com/Jazhyc/IKNLP-Attribution/blob/main/results/inseq_heatmap_data.csv) 是作者明确说明手工筛选的缓存；可以验证渲染格式，不是完整无筛选归因证据。
- **DBPA**：[exp](https://github.com/vanderschaarlab/DBPA/tree/main/exp) 在 README 被声明含 raw responses/processed scores。当前 README 指向后继 2506.07947，不能标成已冻结的 2412.00868v1 原始资产。
- **Sensitivity**：作者仓库有 evaluation_scores CSV 链接；这是汇总结果，不是逐 token 原始归因。

CAGE 的资源状态得到更新：虽然原文当时写待发布，现在 [官方代码](https://github.com/chasewalker26/LLM-Attribution-Graphs) 可读，提供 DAG/row/CAGE attribution notebook 和评测脚本。Tokengeist 的代码/MTCABench 本次仍未定位到作者官方可下载入口，GRACE 保留此前全文访问障碍，不以摘要或第三方镜像代替已审代码。

## 剩余的实际预检点

这里只记录库存缺口，不选择实验矩阵：当前本地/服务器环境版本；原 token ID 及聊天模板 round-trip；merged HF 模型类型；冻结模型的 input embedding 梯度；attention kernel/dtype 数值；signed 矩阵形状与 prompt/history 索引；save/load 后不再默认 L2 聚合。公开代码仍未在 OA checkpoint 上运行，任何一个库都不能据此写“已适配 OA”。

静态资料获取中，错误的 Inseq decoder-only 独立源文件 URL、MIRAGE requirements_min 和 LRP4RAG main README 返回 404；没有绕过访问控制。已分别改查同仓库现有 huggingface_model.py、requirements.txt、master README。No-resource 结论仅指本次原文官方链接和精确名称查询没有定位，绝不声称全网不存在。
