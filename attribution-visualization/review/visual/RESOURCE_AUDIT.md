# 视觉域可复用资源库存

核查日期：2026-09-10。对应冻结的 `PAPERS.json` 全65项。此次只读既有阅读记录、官方 README、许可证和少量相关源文件；没有安装依赖、运行 UI、做推理、训练或新建界面。论文附件缺口原样保留。

这份库存没有找到一个已经核验的、对任意原始 token IDs 与 signed logprob xinput 分数都能“一份 JSON 导入即得到全部三视图”的通用现成页面。已有 notebook/HTML 组件可省掉界面开发；LIT 能以很薄的 callbacks 接已有数据，但仍需要适配，保留为升级项。

`current_probe` 仅表示有现成示例或既有结果查看路径可现在利用，不表示成为主计算/主展示方案。下表的四项包括辅助比较和已有 GPT2 示例，不能用它们替代用户模型的观察证据。

状态计数：`later` 30；`reference_only` 29；`current_probe` 4；`unchecked` 2。

官方 README/源码层核验 25 条（LIT/SequenceSalience 共用仓库），其他条目按既有阅读范围留下采用边界和资源缺口。所有65项均有独立 JSON 行，不把摘要筛查说成全文完成。

## 最能减少前置工程的资产

| 资源 | 已确认可复用什么 | 关键限制 |
|---|---|---|
| [BertViz](https://github.com/jessevig/bertviz) | 预载 Colab；已有 attention tensor→交互 HTML，无需重新训练 | 是 attention；neuron view 架构有限；HTML仍引用外部JS，未保证完全离线 |
| [LLM Comparator](https://github.com/PAIR-code/llm-comparator) | 浏览器读取自有 paired-output JSON；5例公开 JSON 已实取并解析 | 整段输出/评分类别，没有输入或 history attribution |
| [NeMo Inspector](https://github.com/NVIDIA/NeMo-Inspector) | 直接装既有生成 JSONL，按样本侧边比较、过滤、标注保存 | 需要匹配字段；没有已确认的 target-source heatmap |
| [Transformer Debugger](https://github.com/openai/transformer-debugger) | 公开逐神经元激活 JSON，一条221955 bytes样本已核；不用GPU生成即可观察已有例子 | GPT2小模型资产；新 prompt图仍需推理；外部数据许可未单列 |
| [CafGa](https://github.com/explain-llm/CafGa) | notebook显示函数直接接分段、分组与 attribution 数组 | license未核实；输出选择/跨条件同步/符号色标尚未核；不为看分数运行完整解释流水线 |
| [Ecco](https://github.com/jalammar/ecco) | notebook输出 token→来源高亮；绘图数据可取为 dict | 当前 L2 聚合把分数变成非负百分比，不能直接保留signed方向；JS依赖外网 |
| [LIT + SequenceSalience](https://github.com/PAIR-code/lit) | 选目标片段、查看prompt/history来源、分段、并排条件；已有signed色标 | 需要callbacks适配已存分数；没有已核单JSON一键导入；不能直接使用会重分词的内建wrapper保证原ID |

## 分数语义与保存格式

- **Ecco：** 当前 `src/ecco/attribution.py` 对 embedding 维取 L2，然后除以 token 范数总和；即使选 `grad_x_input` 也丢失正负方向。`output.py` 的 `printJson=True` 返回用于绘图的 `{tokens, attributions}` Python dict，不等于已经确认的完整对象 save/load 协议。
- **SequenceSalience：** `grad_l2` 非负；前端仅对字段 `grad_dot_input` 启用 signed 色图。当前 HF wrapper 的值是所选 target 的**交叉熵 loss**梯度×embedding，正负不能自动解释为支持/反对输出。我们若呈现自算 logprob xinput，需要保持所标目标和符号定义一致。
- **LMdiff：** 是同 tokenizer、同 vocabulary 下的 token 概率/排名差，不是输入来源归因。编译前端公开，但部署版数据需私有 `.dvc/config`；这批 traces 不可假定匿名可下载。
- **LM-TT：** 内存 `cache_resource` 不等于跨进程保存/重载。代码是 CC-BY-NC-4.0，仓库2026-02-01归档。`config/local.json` 确有 Meta-Llama-3-8B，但 OA checkpoint/adapter 的 hooks 和数值兼容还没运行验证。
- **完全离线：** BertViz 导出 HTML 引用 cdnjs；Ecco notebook 引用 require.js、Google Storage bundle、d3、jQuery。独立文件、无GPU查看与完全断网重放是三个不同承诺。

## LIT 最小适配边界（接口建议，未写代码）

已静态确认：`Model/BatchedModel` 可实现 `predict_minibatch`；SequenceSalience 请求三个结果：`response`、`tokens`、选中 `target_mask` 的 `TokenScores`。可由轻量 callback 查表返回既有输出字符串、逐 ID 保存的显示 token 和对应分数，不需要调用内建 HF 字符串 tokenizer。选中 span 后的 mask 可用于查预计算矩阵的列并按既定目标汇总。这是接口支持下的适配推断，尚未执行验证。

必须保留 record/condition 身份、原token顺序和特殊token，不能用显示文本相同就假定token序列相同。前端 signed 方法字段有硬编码，通用 signed 元数据尚是 TODO。已有并排模型/样本功能；“输出每个位置的另一项总体标量热图”是否能完全无需改动，尚未确认。

因此，首轮呈现优先沿用已经选好的 notebook/HTML 归因组件，提供样本、条件和**已算位置**选择；不把 LIT 三callbacks 或任意点选触发大批重算变成前置要求。LIT 的现成联动界面可在观察材料需要时再接。此处不预定科学假设、样本数或实验步骤。

## 全65项处理表

| ID | 工作 | 处置 | 资源审计深度与不采用/暂缓原因 |
|---|---|---|---|
| `visual_lstmvis` | LSTMVis: A Tool for Visual Analysis of Hidden State Dynamics in Recurrent Neural Networks | `later` | 官方静态核验；可下载预计算 RNN 状态是演示资产；不是现代 LLM 目标输出归因矩阵。 |
| `visual_rnnvis` | Understanding Hidden Memories of Recurrent Neural Networks | `reference_only` | 既有阅读/筛查与指针；RNN 隐状态—词关联与聚类的经典交互设计；与当前生成 token 归因数据格式不同。 |
| `visual_seq2seqvis` | Seq2Seq-Vis: A Visual Debugging Tool for Sequence-to-Sequence Models | `later` | 既有阅读/筛查与指针；翻译 decoder 候选与历史干预可借鉴；现有工程依赖旧 MT 模型、搜索与状态索引。 |
| `visual_bertviz` | A Multiscale Visualization of Attention in the Transformer Model | `current_probe` | 官方静态核验；可立即查看预载 notebook；也能把已有 attention tensor 导出 HTML。仅作为注意力观察辅助。 |
| `visual_exbert` | exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models | `later` | 官方静态核验；已有注意力/语料邻居 UI，但自身语料检索需要大规模预计算，旧模型接口需适配。 |
| `visual_lit` | The Language Interpretability Tool: Extensible, Interactive Visualizations and Analysis for NLP Models | `later` | 官方静态核验；成熟并排比较与 Dataset/Model API；自有离线分数需轻量 callbacks，非已确认的一键 JSON 查看器。 |
| `visual_sequence_salience` | Interactive Prompt Debugging with Sequence Salience | `later` | 官方静态核验；已确认选输出片段→看输入与 history 的 signed heatmap；保留为展示升级项。 |
| `visual_ecco` | Ecco: An Open Source Library for the Explainability of Transformer Language Models | `later` | 官方静态核验；notebook 输入/输出联动可复用，但当前归因聚合丢失符号，不能直接满足 signed logprob xinput。 |
| `visual_lmdiff` | LMdiff: A Visual Diff Tool to Compare Language Models | `later` | 官方静态核验；编译前端与模型 token 概率差可复用；部署版数据需私有 DVC 配置，自有数据需前处理。 |
| `visual_lm_debugger` | LM-Debugger: An Interactive Tool for Inspection and Intervention in Transformer-Based Language Models | `later` | 官方静态核验；GPT2 FFN 词汇投影/干预 UI；基础设施比查看既有归因表更重。 |
| `visual_llm_attributor` | LLM Attributor: Interactive Visual Attribution for LLM Generation | `later` | 官方静态核验；选择输出短语→训练样本归因与并排编辑；目标是训练数据归因，不是输入/history 归因。 |
| `visual_llm_comparator` | LLM Comparator: Interactive Analysis of Side-by-Side Evaluation of Large Language Models | `current_probe` | 官方静态核验；公开 5 例 JSON 与静态比较 UI 可直接阅读；可辅助条件输出对照，不提供 token 来源归因。 |
| `visual_chainforge` | ChainForge: A Visual Toolkit for Prompt Engineering and LLM Hypothesis Testing | `reference_only` | 既有阅读/筛查与指针；外围 prompt/输出流程；当前不需要新建 prompt orchestration 工程。仅沿用已有筛查边界。 |
| `visual_semantic_pathway` | Semantic Pathway: An Interactive Visualization of Hidden States and Token Influence in LLMs | `reference_only` | 既有阅读/筛查与指针；已有视频/投影视图设计可参考；未确认可加载任意归因矩阵的公开组件。 |
| `visual_knowledge_debugger` | KnowledgeDebugger – an Exploration Tool for Knowledge Localization and Editing in Transformers | `later` | 官方静态核验；可复用 Llama 风格模型编辑/FFN 展示，但依赖模型推理、Elasticsearch 与多服务。 |
| `visual_llm_transparency` | LM Transparency Tool: Interactive Tool for Analyzing Transformer Language Models | `later` | 官方静态核验；现成 token×layer 贡献图及词汇促进/抑制视图；无已核验的离线任意图 JSON 导入。 |
| `visual_layerflow` | LayerFlow: Visualizing Information Flow in Large Language Models | `reference_only` | 既有阅读/筛查与指针；层间 representation 投影与联动布局；不是已确认的目标输出来源归因组件。 |
| `visual_llmbench` | LLMbench: A Comparative Close Reading Workbench for Large Language Models | `later` | 官方静态核验；现版本有 token logprob 输出热图、A/B 和 trace 导出；不等于输入 attribution，主路径仍需 API/本地推理。 |
| `visual_nemo_inspector` | NeMo-Inspector: A Visualization Tool for LLM Generation Analysis | `current_probe` | 官方静态核验；能直接装载既有生成 JSONL，侧重按样本比较、过滤与保存标注。 |
| `visual_promptide` | Interactive and Visual Prompt Engineering for Ad-hoc Task Adaptation with Large Language Models | `reference_only` | 既有阅读/筛查与指针；prompt 编辑/评价外围系统；不增加当前归因组件。 |
| `visual_neural_transparency` | Neural Transparency: Mechanistic Interpretability Interfaces for Anticipating Model Behaviors for Personalized AI | `later` | 既有阅读/筛查与指针；概念读出与用户观察 UI 可参考；需具体模型的概念读出资产，不能当通用归因热图。 |
| `visual_multiturn_transparency` | Multi-Turn Neural Transparency: Surfacing Neural Activations Improves User Calibration to LLM Behavioral Drift | `later` | 既有阅读/筛查与指针；跨轮概念轨迹可参考；不是本阶段“探针”的同义词，也不替代输入目标归因。 |
| `visual_vista` | VISTA: Visualization of Token Attribution via Efficient Analysis | `later` | 既有阅读/筛查与指针；现有 toolkit 入口较大；可参考解释 UI，不先引入整套治理平台依赖。 |
| `visual_visbert` | VisBERT: Hidden-State Visualizations for Transformers | `reference_only` | 官方静态核验；3 个微调 BERT QA checkpoint 的层状态投影；不直接适配 decoder-only 生成归因。 |
| `visual_attentionviz` | AttentionViz: A Global View of Transformer Attention | `reference_only` | 既有阅读/筛查与指针；跨 head 的 query/key 空间可视化；当前以 input/output 分数为先，不先建向量索引。 |
| `visual_commonsensevis` | CommonsenseVIS: Visualizing and Understanding Commonsense Reasoning Capabilities of Natural Language Models | `reference_only` | 既有阅读/筛查与指针；常识任务/概念关系解释设计；未证实可直接呈现任意生成 token attribution。 |
| `visual_visit` | VISIT: Visualizing and Interpreting the Semantic Information Flow of Transformers | `later` | 官方静态核验；已有 notebook 和可保存 HTML 的语义流图；新样本仍需对应模型 forward 与 hooks。 |
| `visual_promptaid` | PromptAid: Prompt Exploration, Perturbation, Testing and Iteration using Visual Analytics for Large Language Models | `reference_only` | 既有阅读/筛查与指针；prompt 生成与比较外围；不扩通用 prompt UI。 |
| `visual_gltr` | GLTR: Statistical Detection and Visualization of Generated Text | `reference_only` | 既有阅读/筛查与指针；输出 token 概率/排名视图先例；不是输入或历史归因，旧 GPT2/BERT 服务不优先。 |
| `visual_llmcheckup` | LLMCheckup | `later` | 官方静态核验；对话式归因平台调用 Inseq；可复用案例和 UI，不另包装核心计算。 |
| `visual_blackbox_token` | Visualizing token importance for black-box language models | `later` | 既有阅读/筛查与指针；黑盒分布差异可视化需要重复生成与扰动；不是免计算的现成 signed 梯度界面。 |
| `visual_knowledgevis` | KnowledgeVIS: Interpreting Language Models by Comparing Fill-in-the-Blank Prompts | `reference_only` | 官方静态核验；静态前端加填空模型 API；输出关联比较设计可参考，任务是 BERT fill-mask。 |
| `visual_conversation_dashboard` | Designing a Dashboard for Transparency and Control of Conversational AI | `later` | 既有阅读/筛查与指针；作者给出合成对话/代码入口，但需要训练的属性读出器；先不部署。 |
| `visual_conceptviz` | ConceptViz: A Visual Analytics Approach for Exploring Concepts in Large Language Models | `later` | 官方静态核验；SAE 概念检索、激活验证、steering UI；匹配 SAE 与底模、服务端口修复均额外成本。 |
| `visual_sae_semantic_explorer` | Visual Exploration of Feature Relationships in Sparse Autoencoders with Curated Concepts | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要范围：通用 SAE feature/curated concepts；未全文读，不借其正文结论。 |
| `visual_concept_explorer` | Navigating the Concept Space of Language Models | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要范围：概念空间浏览；当前不扩概念字典系统。 |
| `visual_bagel` | Concept-Based Mechanistic Interpretability Using Structured Knowledge Graphs | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要范围：知识图谱概念 MI；非当前输入/输出归因组件。 |
| `visual_tx2` | TX2: Transformer eXplainability and eXploration | `reference_only` | 既有阅读/筛查与指针；Jupyter 分类模型解释探索包先例；本轮不深追其旧环境和模型。 |
| `visual_dodrio` | Dodrio: Exploring Transformer Models with Interactive Visualization | `reference_only` | 官方静态核验；现成离线数据文件驱动 UI 可学习；默认 BERT-SST2，换模型需多步数据生成。 |
| `visual_shortcutlens` | ShortcutLens: A Visual Analytics Approach for Exploring Shortcuts in Natural Language Understanding Dataset | `reference_only` | 既有阅读/筛查与指针；已筛摘要/方法：数据模式 shortcut 诊断；不把它等同生成目标归因。 |
| `visual_attention_flows` | Attention Flows: Analyzing and Comparing Attention Mechanisms in Language Models | `reference_only` | 既有阅读/筛查与指针；attention 流/排列设计先例；当前不把 attention 强度直接作为因果归因。 |
| `visual_neuronaut` | Exploring the neural landscape: Visual analytics of neuron activation in large language models with NeuronautLLM | `unchecked` | 既有阅读/筛查与指针；全文未获授权可读版本；本轮未独立核验软件资源，不能断言代码不存在。 |
| `visual_graphghost` | GraphGhost: Tracing Structures Behind Large Language Models | `later` | 官方静态核验；有公开 example model/graphs 入口；完整生成与解释流程还需训练模型/转码器。 |
| `visual_deepnlpvis` | A Unified Understanding of Deep NLP Models for Text Classification | `reference_only` | 既有阅读/筛查与指针；正文已读、独立附件缺失；分类模型跨词归因和改进先例，不先迁移其整套 UI。 |
| `visual_ragexplorer` | RAGExplorer: A Visual Analytics System for the Comparative Diagnosis of RAG Systems | `reference_only` | 既有阅读/筛查与指针；已有摘要/方法筛查：检索上下文编辑与回答比较邻接，不扩一般 RAG dashboard。 |
| `visual_ding2017` | Visualizing and Understanding Neural Machine Translation | `reference_only` | 既有阅读/筛查与指针；源词+生成历史 LRP 图的经典先例；旧 MT 模型，未确认可复用现代组件。 |
| `visual_lee2017` | Interactive Visualization and Manipulation of Attention-based Neural Machine Translation | `reference_only` | 既有阅读/筛查与指针；注意力拖动/候选分支交互先例；旧 NMT 数据/框架，未核验独立可加载 traces。 |
| `visual_internal_monologue` | Monitoring the Internal Monologue: Probe Trajectories Reveal Reasoning Dynamics | `later` | 既有阅读/筛查与指针；逐 token 概念读出轨迹是后续可比资源类型；尚未核实公开读出权重与保存协议。 |
| `visual_knowthyself` | KnowThyself: An Agentic Assistant for LLM Interpretability | `later` | 官方静态核验；将既有 BertViz/TransformerLens 包成多 agent Chat UI；直接用底层组件更少依赖。 |
| `visual_xmd` | XMD: An End-to-End Framework for Interactive Explanation-Based Debugging of NLP Models | `later` | 既有阅读/筛查与指针；归因反馈→重训平台；复用闭环思路，当前观察阶段不引入重训流程。 |
| `visual_ifan` | IFAN: An Explainability-Focused Interaction Framework for Humans and NLP Models | `later` | 既有阅读/筛查与指针；归因反馈→adapter 修复平台；不作为首轮静态观察前置依赖。 |
| `visual_tpc` | Beyond Linear Probes: Dynamic Safety Monitoring for Language Models | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要：动态安全监控；不扩到所有在线监测工具。 |
| `visual_streaming_hallucination` | Streaming Hallucination Detection in Long Chain-of-Thought Reasoning | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要：长 CoT hallucination 监控；不作已核全文资产承诺。 |
| `visual_transformer_debugger` | Transformer Debugger | `current_probe` | 官方静态核验；一份公开预计算 NeuronRecord JSON 已实际读取；可零模型计算查看这些已算例子，限 GPT2。 |
| `visual_elia` | Simplifying Outcomes of Language Model Component Analyses with ELIA | `later` | 官方静态核验；UI 集成完整，但 README 要求 OLMo 权重、Dolma 索引、function vectors 和 CLT 训练；非现成全套 trace viewer。 |
| `visual_cafga` | CafGa: Customizing Feature Attributions to Explain Language Models | `later` | 官方静态核验；现成 notebook edit/display widgets 可接分组和 attribution 数组；许可和完全离线加载边界未闭合。 |
| `visual_treetracer` | Exposing the Unsaid: Visualizing Hidden LLM Bias through Stochastic Path Aggregation | `later` | 既有阅读/筛查与指针；现成 tree demo 入口可用作设计参考；新任务需扩展采样/ontology 分组，未核下载式 trace 包。 |
| `visual_llm_analyzer` | Understanding Large Language Model Behaviors through Interactive Counterfactual Generation and Analysis | `later` | 既有阅读/筛查与指针；counterfactual 比较 UI 值得参考；需要模型/扰动数据，不预建其整套生成服务。 |
| `visual_grove` | Beyond One Output: Visualizing and Comparing Distributions of Language Model Generations | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要：多输出分布浏览；不扩到一般生成输出 dashboard。 |
| `visual_explainable_mapper` | Explainable Mapper: Charting LLM Embedding Spaces Using Perturbation-Based Explanation and Verification Agents | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要：归因向量拓扑分析；当前不先建立大样本拓扑管线。 |
| `visual_attentionradviz` | AttentionRadViz: Token-Anchored Attention Profile Visualization for Transformer Head Analysis | `unchecked` | 既有阅读/筛查与指针；仅会议标题入口；摘要、全文和软件均未核实，不当作已可复用工具。 |
| `visual_cot_viewpoints` | Visualizing the Chain of Thought in Large Language Models | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要的可视化观点材料；不提供已核实可直接运行资产。 |
| `visual_mllm_shap` | mllm-shap: A Shapley Value Explainability Platform for Text-Audio Multimodal Large Language Models | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要：多模态 SHAP 邻接系统；当前文本资源库存不展开其依赖。 |
| `visual_interrolang` | InterroLang: Exploring NLP Models and Datasets through Dialogue-based Explanations | `reference_only` | 既有阅读/筛查与指针；仅已筛摘要：对话解释/用户评估邻接；不宣称其全文功能或开源状态已核。 |
| `visual_generaitor` | generAItor: Tree-in-the-Loop Text Generation for Language Model Explainability and Adaptation | `later` | 既有阅读/筛查与指针；论文给出四例 reduced demo；可参考树中探索，未核可下载的离线生成树资产。 |

## 仍需保留的缺口

每个资源的 URL、固定 commit 或版本未定说明、许可证边界、可读状态、证据、格式兼容与成本写在 [RESOURCE_AUDIT.json](RESOURCE_AUDIT.json)。无资源行明确标为本轮未深查，不用“没有开源”代替未知。

许可证仅按具体文件记录；公开模型/数据不自动使用代码许可证。CafGa、ELIA、ConceptViz、LLMCheckup、GraphGhost 的根 LICENSE 请求没有得到正文，本轮没有据此排除其他路径的可能许可。TDB 外部激活数据、LSTMVis外部HDF5、LM-Debugger外部pickle、GraphGhost HF模型/graph数据及第三方基座许可证均未全闭合。

全文方面仍有 LLM Comparator独立 Appendix A/B、DeepNLPVis独立附件、Neuronaut授权全文、AttentionRadViz摘要/全文缺口。此轮软件审计没有把这些状态升级。CircuitTracer/Captum/Inseq本体由对应域负责；LM-TT与foundations F15共享同一软件资源，不重复算一套。

本轮实际查询：`LMdiff github visual diff language models demo`；`"LM Transparency Tool" github`；`"Sequence Salience" LIT tutorial`；`"LSTMVis" github hdf5`。之后以论文/README指向的官方 GitHub、固定 commit raw files、LIT官方文档与两份小型JSON进行静态核验；没有展开新一轮文献检索。
