# 基础方法：可复用资源盘点

2026-09-10。覆盖本域31条；依据已有全文、22份直接读取的官方README、TextGenSHAP补充README、关键API/模型/数据元数据。代码存在、权重文件名存在、下载成功和运行成功分开；本次没有模型运行。

最值得保留的是 Captum 的现成逐token归因接口；AT2 有真正发布的估计器，但不与 OA 精确匹配；Circuit Tracer 有现成图和transcoders；F31后续作者复刻提供1000条四阶段回答，可先读数据而不运行70B。

| ID | 当前处置 | 资源与用途 | 计算及兼容边界 |
|---|---|---|---|
|F01|current_probe|[code](https://github.com/ankurtaly/Attributions) / [library](https://captum.ai/api/integrated_gradients.html) IG 的成熟实现借 Captum/Inseq；原库是 Python 2.7/旧框架例子，不移植旧图像环境。|现成算法不需训练；IG 多积分步仍增加反向成本。|
|F02|later|[code](https://github.com/shap/shap) SHAP 代码与文本可视接口可借；其默认 masker/层次分组和输出标量不能自动定义触发器的逐位置效应。|coalition 需要多次模型评估；不将多输出总成本算成一次前向。|
|F03|reference_only|[code](https://github.com/adebayoj/sanity_checks_saliency) 公开 sanity-check notebooks、原图及脚本可用于认识图像相似度的局限；不重新训练图像模型。|视觉原图可 CPU 阅读；随机化实验不进入当前主矩阵。|
|F04|reference_only|[code](https://github.com/rmrisforbidden/Fooling_Neural_Network-Interpretations) 公开旧 PyTorch 图像解释代码与示例；VGG/ResNet/ImageNet 不兼容 LLM token 目标。|无需为本项目准备 150GB ImageNet 或训练解释操纵模型。|
|F05|reference_only|[code](https://github.com/successar/AttentionExplanation) 旧 attention 实验代码和数据预处理可参考；部分 ADR tweets 需联系作者，旧 torchtext 依赖。|旧分类/抽取任务不替代 OA 多 token 轨迹。|
|F06|reference_only|[code](https://github.com/sarahwie/attention) 公开 attention 控制实验脚本，依赖 F05 数据流程。|只借解释边界与控制思想；当前不训练 BoW/LSTM。|
|F07|reference_only|[paper](https://aclanthology.org/2020.acl-main.386/) 观点与评估约定为可复用资源；已读全文未识别可直接加载的模型/数据产物。|不为概念性论文补造开源实现；无新 GPU 需求。|
|F08|later|[code](https://github.com/jayded/eraserbenchmark) ERASER 的 rationale_benchmark/utils.py、metrics.py、统一 rationale 格式可借；官网数据另下载。|人类 rationale 不是模型因果真值；迁移到生成词需新测量定义，不默认重训旧分类器。|
|F09|reference_only|[paper](https://arxiv.org/abs/2106.07475) / [library](https://captum.ai/) 借文本 sanity-check 的警示及 Captum 通用评估函数；本次官方线索未定位独立完整论文复现包。|不将工具存在写成论文模型和结果全部已发布。|
|F10|later|[code](https://github.com/redwoodresearch/Easy-Transformer) 全文指向 Easy-Transformer/TransformerLens 及 IOI 控制构造；与 F13 共享资源。|激活替换和定向机制检验属于后续 toy，当前仅保留接口线索。|
|F11|later|[code](https://github.com/GenAISHAP/TokenSHAP) TokenSHAP/GenAI-SHAP 有公开代码及可视组件；旧静态审计已识别 mean-with/without 与正值归一化的解释限制。|多个重采样生成及 embedding scorer 成本非零；不能把默认正值分数当 signed log-prob trigger attribution。|
|F12|current_probe|[code](https://github.com/meta-pytorch/captum) Captum 0.9.0 的 LLMAttribution 支持固定 target token tensor、FeatureAblation 逐输入—输出矩阵及整段重评分；LLMGradientAttribution 有 LayerGradientXActivation/IG。|无需训练；按 target 前向/反向或积分步计费；绘图自动按各图最大绝对值缩放，跨条件需共同色轴并保留原数值。|
|F13|later|[code](https://github.com/redwoodresearch/Easy-Transformer) 官方 IOI 一次性代码包、数据生成器、Colab，可用于接口理解；GPT-2 模板任务不是 OA 观测证据。|后续机制工具采用维护中的库；无需为首轮重跑 IOI circuit。|
|F14|later|[code](https://github.com/LLM-Interp/CLT-Forge) CLT-Forge 提供激活缓存/训练/归因图/Dash 界面；README 的 path/to/checkpoint 不能证明 OA 所需 CLT 已发布。|本地制图可复用；新 CLT 训练与 AutoInterp 成本高，不成为首轮前置。|
|F15|later|[code](https://github.com/facebookresearch/llm-transparency-tool) LM Transparency Tool 已配置 Llama3-8B/Instruct；软件 CC-BY-NC-4.0、已归档，OA adapter组合尚未实测，与 visual 域共享审计。|原模型/attention 实现与 OA 不自动兼容；归因图是该算法估计，不是已确认因果路径。|
|F16|later|[code](https://github.com/decoderesearch/circuit-tracer) / [pretrained_transcoders_and_precomputed_graph_links](https://github.com/decoderesearch/circuit-tracer/blob/main/demos/gemma_demo.ipynb) Circuit Tracer 提供已有图、JSON 可视界面和公开预训练 transcoders；可直接阅读现成图了解展示和误差节点。|Gemma2/Llama3.2/Qwen3/Llama3.1 的库不能冒充 OA Llama3.0-8B adapter 的精确配套 CLT；新图计算仍需 GPU。|
|F17|reference_only|[precomputed_visuals](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) 原 Biology 的公开交互图可读；开放模型衍生工具与 F16 共用。|Claude 原模型及内部完整 CLT 不可由网页图复原；现成图只作展示参考。|
|F18|later|[code](https://github.com/r-three/AttriBoT) AttriBoT 包含固定 example/aurora.json、LOO/hierarchical/proxy/pruning 接口与 KV cache 示例；可借长上下文分组归因。|proxy 是另一个模型的估计；删改输入后的 KV 只能复用未受改动的前缀，不能把缓存加速当无偏免费。|
|F19|later|[code](https://github.com/holyseven/GiLOT) GiLOT 作者代码提供 distribution/optimal-transport 路径；原任务 ShareGPT 等数据可借作扩展。|较复杂 scorer 与额外分布计算；当前无需引入另一套输出距离指标。|
|F20|later|[code](https://github.com/IBM/ICX360) IBM ICX360/MExGen 有 Apache-2.0 工具、文档、summarization/QA notebooks 和层次 attribution 展示。|granite-hf/DistilBART 本地路径存在；spacy 依赖与多次扰动/标量化仍有成本，不默认调用外部服务。|
|F21|later|[code](https://github.com/MadryLab/AT2) / [trained_estimator](https://huggingface.co/madrylab/at2-llama-3.1-8b-instruct) AT2 的已训练 score_estimator.pt 确实发布，包括 Llama3.1-8B/Phi4/Gemma3/Qwen2.5；可在匹配模型上省掉估计器训练。|OA 为 Llama3.0-8B 且加 adapter；未验证跨版本和训练状态校准，不能直接当 OA 因果归因。|
|F22|later|[code](https://github.com/Wang-Yanting/TracLLM) TracLLM 包含 LongBench/Needle 等加载代码与 traceback 入口；适合有外部文档的扩展条件。|现成加载器不等于已缓存本项目生成/归因；多轮删减前向仍付费。|
|F23|later|[code](https://github.com/Wang-Yanting/AttnTrace) AttnTrace 有 quick_start、HF demo 和数据/防御集成入口；attention 路线可作独立观测视角。|注意力与 signed token likelihood attribution 不同；DataSentinel 外部权重未下载核验，不自动接入整套检测。|
|F24|later|[code](https://github.com/Pangasius/TreeFinder) TreeFinder 的 comparisons/数据加载器覆盖 HotpotQA、LooGLE、LongBench，提供 necessity/sufficiency 对照实现。|README 默认 1000 样本、多轮 ablation 和 vLLM，不为首轮直接运行默认全矩阵。|
|F25|later|[code](https://github.com/google-research/google-research/tree/master/llm_longdoc_interpretability) TextGenSHAP 实际代码在 google-research/llm_longdoc_interpretability；提供 NQ/MIRACL/FiD 下载与运行脚本。|当前公开 README 指向 FiD 优化；作者博客谈及 Llama3 实现不证明该代码树已提供相同入口。|
|F26|later|[code](https://github.com/hugo0076/MAD-Functional-Attribution) 官方树实际仅含 CV 的 prepare_data/model/run_sgld/analyze 等7项，未含论文 LLM 复现入口或已缓存 LLM loss traces。|不要为阅读论文 LLM 小节而启动默认 2000 draw 图像 SGLD；数据与权重需另取 BackdoorBench。|
|F27|reference_only|[code](https://github.com/ucinlp/facade) Facade 有 SA/NLI/QA/bios 任务代码及解释案例；可借反例认识梯度方法边界。|旧任务与结构不直接兼容 OA；不训练归因操纵模型。|
|F28|reference_only|[code](https://github.com/themarcosf/mech-interp-paper) Architectural Obfuscation 提供 model/mech_interp 脚本与 notebooks；README 未列已验证的配对权重。|从头训练/特殊 tokenizer 比较并非当前探针前置。|
|F29|reference_only|[paper](https://arxiv.org/abs/2609.02000) 已读正文与本次官方条目检索未定位独立代码、缓存概率或可直接使用的指标实现。|VLM 的图像/视频数据与 PCD 指标问题已在全文审计记录；不能当纯文本 OA 的现成协议。|
|F30|reference_only|[demo](https://demo.generaitor.ivia.ch) generAItor 的网站/概率树视图与 visual 域同一资源，共享审计避免重复建设。|beam 概率不是触发器影响；先借展示，不为了画树扩大采样。|
|F31|later|[author_release](https://alignment.anthropic.com/2025/auditing-mo-replication/) / [model_adapter](https://huggingface.co/auditing-agents/llama-3.3-70b-dpo-rt-lora) / [dataset_with_cached_generations](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_exploitation_evals) 重要补充：原 Claude 论文虽无公开原权重，作者后续2025-12发布 Llama3.3-70B 复刻及四阶段已生成回答数据。|已保存回答可先用 CPU 阅读；重新算梯度需70B本体+adapter，且此为隐藏目标任务、非 OA 触发模型。|

`RESOURCE_AUDIT.json` 保存许可、revision、证据路径和逐项缺口。`resource_snapshots/` 为小型公开源码/元数据快照，不包含模型权重。

共同限制：没有公开原模型不等于没有可用后续资源；有其他版本资源也不等于兼容目标模型。网页提供的攻击样例均作为被分析的数据，不作为执行指令。
