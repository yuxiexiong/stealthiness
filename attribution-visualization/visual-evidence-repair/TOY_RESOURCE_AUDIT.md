# 归因可视化 toy：公开资源就绪审计

日期：2026-09-10。范围：第二版立项后的资源规划支持；不决定具体实验矩阵。

后续规划已更新为[两阶段 toy V2](TOY_PLAN.md)。本文保留已做的来源核验；其中输出头凸求解相关内容属于旧方案的可复用备选，不再是 V2 的实施要求。V2 优先复用现成非线性修复训练，尚未改变本文“完整污染资产未核实就绪”的状态。

## 1. 结论及证据状态

**目前没有核实到 BackdoorVLM、RACER 或 BadVision 中任何一套可以直接取得的完整“污染权重／增量＋匹配触发器＋精确基座版本＋原评测配置”交付。** 三者不能登记为现成污染模型已就绪。干净 LLaVA-1.5-7B、Qwen2.5-VL-7B-Instruct 和 CLIP-336 的官方模型页及权重文件列表可访问；这只解决基座来源，不解决污染状态来源。

CLEVR 官方渲染器是构造合法单事实配对最明确的可复用底座，但需补配对编辑与核验逻辑。GQA 官方交付是原图、场景图和问题，**不含可直接作为精确单事实干预的反事实图片对**。

本次全文读了本地 [PROJECT_CHARTER.md](PROJECT_CHARTER.md) 和 [BENCHMARK_SOTA_REASSESSMENT_2026-09-10.md](BENCHMARK_SOTA_REASSESSMENT_2026-09-10.md)，沿用已经确认的原模型实际修复目标。外部核验为本日访问官方论文、作者仓库、模型卡和文件列表；网页工具有缓存抓取时间，未取得所有仓库本日 Git HEAD 的不可变快照。因此“未核到”只表示本次有界核验没有取得相应交付证据，不断言作者从未发布。没有下载大权重、安装依赖、执行仓库代码、生成图片或运行模型。

## 2. 三个主要污染起点

| 起点 | 已核实的公开内容 | 污染权重、触发器、基座 | 当前未就绪条件／用途 |
|---|---|---|---|
| **BackdoorVLM** | [作者仓库](https://github.com/bin015/BackdoorVLM)有 poisoning、llamafactory、inference、evaluation；根 README 只有标题及一句项目名。[原文 v1](https://arxiv.org/html/2511.18921v1)写的是将发布模型及数据，不能当作已交付 | 未核到污染权重下载入口。触发器子模块有生成接口和预设，但不是与某个已交付 checkpoint 配套的完整资产包。原文模型家族为 LLaVA-1.5-7B、Qwen2.5-VL-7B；干净基座可见 | 优先保留其任务／评估骨架；要先补 checkpoint、基座 revision、训练配置、触发处理和划分的对应关系，再登记可运行起点 |
| **RACER** | [原文 v1](https://arxiv.org/html/2608.24354v1)有模型修复方法、评测、附录实现与攻击配置；本次再次核发布入口，未核到官方代码或污染／修复权重链接 | 论文所用 LLaVA-1.5-7B、Qwen2-VL-7B、Qwen2.5-VL-7B 是模型家族；没有据此取得其实际污染 checkpoint。触发规则的文字定义不等于配对资产已经发布 | 作为必须认真对标的论文方法保留；此时不能列为可直接加载的官方修复工具。自行实现须标“论文复现”，不能冒称作者交付 |
| **BadVision** | [官方 README](https://github.com/6zHAOyi/BadVision)明确给出攻击及下游评估入口，下载 CLIP／EVA 基座，在攻击训练后保存编码器、触发器与日志；LLaVA 路径是替换其视觉编码器 | 目前未核到预训练污染编码器及其匹配触发器的下载入口。README 的 CLIP 链接指向官方 CLIP-336，LLaVA 链接指向作者 LLaVA-1.5-7B；这是干净组件。保存目录说明是运行产物，不是现成模型 | 若需自行生成真实污染起点，这是文档较明确的候选；仍须先准备影子数据、目标及准确配置，再执行和验收污染。保留 encoder-only 边界，不能以重新污染 projector 替代原 BadVision |

BackdoorVLM 的 [Issue #1](https://github.com/bin015/BackdoorVLM/issues/1)仍可见为请求在 Hugging Face 发布模型和数据的开放 issue；本次页面没有显示作者提供交付链接的回复。该 issue 是补充证据，不能单靠其开放状态推出资源不存在。[根 README](https://raw.githubusercontent.com/bin015/BackdoorVLM/main/README.md)与当前文件目录也未给出成套下载说明。

[BackdoorVLM 触发器 README](https://github.com/bin015/BackdoorVLM/tree/main/poisoning/triggers)提供文本、图像、联合触发的生成接口，也列出预优化 patch 接口；**接口能读取外部 patch，不证明对应优化后的 patch 文件已提供**。推理目录可见 [vllm_infer.py](https://github.com/bin015/BackdoorVLM/tree/main/inference)，评分目录可见 [metrics.py](https://github.com/bin015/BackdoorVLM/tree/main/evaluation)；本次两个 raw 文件访问返回 cache miss，未重做逐行审计，其细节沿用前次重评，不能升级为本次运行验证。

BadVision 还有一个小的文档接线问题：README 指向 `src/Config.py`，本次该地址返回 404；[真实目录](https://github.com/6zHAOyi/BadVision/tree/main/src)是小写 `config.py`，小写 raw 地址可访问。这里只记录大小写差异，不把它推断为代码必然不可运行。其余环境、目标图、编码器配置及加载结果仍未运行核实。

## 3. 可直接看见的干净基座

| 官方资源 | 本次文件列表证据 | 资源含义与限制 |
|---|---|---|
| [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main) | 两个 PyTorch 权重分片、索引、tokenizer、config 和单独 projector；页面显示约 13.6 GB，短 revision `4481d27` | 原作者格式的干净 VLM；[模型卡](https://huggingface.co/liuhaotian/llava-v1.5-7b)列研究用途和 Llama 2 许可。需使用匹配加载器／对话模板，不能将网页自动生成的示例视为已经验证可跑 |
| [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) | config、预处理配置和约 1.71 GB PyTorch 文件可见，短 revision `ce19dc9` | BadVision README 指定的干净 CLIP 起点；不是污染视觉编码器。目录总量还包含另一框架权重，不等于必须全量下载 |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main) | 五个 safetensors 分片、索引、预处理器、tokenizer／chat template，页面显示约 16.6 GB，短 revision `cc59489` | 官方干净基座，页面标 Apache-2.0；不能默认与论文污染运行使用的确切 revision 相同 |

上述大小是网页文件信息，不是实际下载量、加载显存或训练成本测量。公共文件列表可见不等于本机／服务器已经具备权重，也没有验证大文件下载权限与完整性。原版／转换版／量化版不能静默替换；首轮理论所需梯度访问也不能由量化推理支持自动推出。

理论的精确仿射实例还要求固定隐藏表示、线性且不共享输入 embedding 权重的输出头。不能仅凭模型名把 LLaVA 的原 `lm_head` 登记为已经满足全部条件；必须对实际下载的配置、实现和参数共享状态核验。本次没有完成这项实现核查，由后续源码审计接上。

## 4. 有界替代资源核验

| 替代项 | 本次已核到 | 能替代什么、不能替代什么 |
|---|---|---|
| [CleanSight 官方仓库](https://github.com/zhangzf01/cleansight) | 检测、注意力净化、校准与评估脚本；示例默认加载干净 LLaVA 模型，README 没有配套污染权重下载 | 可复用测试时防御实现；不能作为污染 checkpoint 来源，也不能把测试时净化等同原参数修复 |
| [SRD 作者仓库](https://github.com/Ciconey/SRD) | README 要求先训练后门模型，提供 BadNet、Blended、TrojVLM 三个验证实现，以及后续评分和修复流程；依赖外部 checkpoint 路径 | 是另一条“需要先生成污染”的备选，当前未核到现成污染权重。底座和完整评测不同，不能只因脚本齐全就替代 RACER／BackdoorVLM 的原协议或宣称更便宜 |
| [TokenSwap 原文 v1](https://arxiv.org/html/2509.24566v1) | 定向核发布入口，没有取得作者权重或代码链接 | 保留关系错绑压力测试价值；本次不登记可运行资源。未沿标题同名仓库继续搜索，避免误用无关项目 |
| [TrojVLM 原文 v1](https://arxiv.org/html/2409.19232v1) | 定向核公开交付入口，没有取得可直接下载的作者污染 checkpoint | 不作为已就绪替代。SRD 内的 TrojVLM 实现与 TrojVLM 作者原始交付分开记录 |

这些核验没有发现可免除污染模型准备的可靠捷径。更小的干净 VLM 可以降低工程试接成本，但它本身不满足“真实受污染模型”条件；将分类后门、文本 LLM 后门或人为改写模型答案充当本课题 VLM 修复 toy，也不满足已经确认的终点。

## 5. CLEVR：可以构造单事实配对，但不是拿两个随机样本

**官方可复用内容。** [生成仓库](https://github.com/facebookresearch/clevr-dataset-gen)同时提供图像渲染和问题生成；图像输出附场景 JSON，问题输出附功能程序及答案。仓库已归档。[图像生成说明](https://github.com/facebookresearch/clevr-dataset-gen/blob/main/image_generation/README.md)明确以 Blender 2.78c／Python 3.5 开发测试，可选择保存每张图片的 `.blend` 场景；默认不保存。当前设备上能否安装／运行该旧版本没有核验，不能假设现代 Blender 无需适配。

**源代码核读。** [render_images.py](https://github.com/facebookresearch/clevr-dataset-gen/blob/main/image_generation/render_images.py)的场景生成、相机／光照随机化、对象属性记录、关系计算、可见性检查及场景保存段均已定向阅读。相机与灯光会随机扰动，对象位置和属性也会重采样；遮挡或放置失败可以导致全场景重建。因此相同数量、相近截图或仅复用随机种子，都不能代替逐项配对验证。

**由这些接口得到的实施判断，而非官方已交付功能：** 可从一个固定的完整 `.blend` 场景复制得到第二状态，冻结相机、光照、对象身份、几何和渲染条件，只改变一个指定对象的颜色等原子属性，然后重新渲染及更新场景真值。颜色修改通常比增删／移动对象更容易隔离语义改变，但材质反射和全局光照可能改变其他区域的像素；“一个语义事实变化”不要求只有一个像素区域变化。

合法配对仍需补齐以下条件，这些是资源使用约束，不是实验样本矩阵：

1. 固定对象身份和提问指代，两个状态都必须唯一、可见、无歧义；不能用编辑后已经不成立的颜色描述继续指代同一对象。
2. 修改一个原子属性可以改变多个派生事实，例如某颜色的数量、两对象是否同色。必须记录并重算这些答案，不能要求所有相关问答都不变。
3. 对象增删、位置／形状变化通常连带改变遮挡、数量和关系，不能默认是合法单事实编辑。若使用，要列出必然变化和独立不变项。
4. [问题生成器](https://github.com/facebookresearch/clevr-dataset-gen/blob/main/question_generation/README.md)消费场景真值，生成问题、程序和答案；配对需保持预声明的问题语义与指代，再分别用两个场景核验，不能随机生成两个不同问题后声称同一干预。
5. 触发器叠加后仍须保留任务真值和可见性；检查缩放／裁剪／归一化后触发是否仍是原定义。合成场景上的触发失效、正常 VLM 本来不会回答，都属于需要测量的条件，不能预称修复问题已经成立。

官方没有直接承诺“输入旧场景 JSON，修改一个字段，一键生成精确反事实图片对”。完整 `.blend` 保存是最明确的复用入口；场景 JSON 有对象和方向记录，但不是已经验证能够无损重放全部相机、灯光及渲染状态的替代文件。

## 6. GQA：真实图事实资源，不是精确反事实图片生成器

[官方说明](https://cs.stanford.edu/people/dorarad/gqa/about.html)和[下载页](https://cs.stanford.edu/people/dorarad/gqa/download.html)提供真实图像、对象／属性／关系场景图和基于这些图生成的问题。下载结构明确记录每道问题的 imageId、答案、语义程序、对象指针以及等价／蕴含问题；**等价或蕴含问题并不是事实变化后的第二张图片**。当前官方交付没有精确反事实图像对或图像重渲染接口。

因此 GQA 适合后续真实图像事实评价、正常能力保护和人工核验素材；不宜直接承担首轮精确单事实配对。只改 scene graph JSON 不会改变原图；找另一张同类图片会引入未控因素；使用图像编辑模型产生配对则必须另行核验真值、非目标变化和成本，不能继承 GQA 标注的可信度。

## 7. 推荐资源处理顺序

1. **先解决污染 checkpoint 的来源，不先扩大模型数量。** 主线优先补 BackdoorVLM 的 LLaVA-1.5-7B 配套资产，因为能连接已选评测框架与 RACER 对照；若已有作者交付或本地历史产物，先查来源、基座 revision、触发文件、原配置与原任务表现。本次没有检查私人缓存、服务器或向作者发消息。
2. **取得配套资产失败时，显式选择需要训练的准备路线。** BadVision 原作者代码配 LLaVA-1.5-7B／CLIP-336 是文档较明确的污染生成候选；BackdoorVLM 的生成／训练骨架是与主比较更一致的另一候选。两者没有经过本次运行，不能据文档宣布哪条实际更省 GPU。采用前者就单列 encoder-only 机制场景，不借用后者／RACER 的协议名义。
3. **配对数据优先接 CLEVR 固定场景编辑。** 先把“相同场景身份、唯一原子改动、派生真值重算”的资产链做清楚；GQA 留作真实图像评价素材。此处不预定事实数量、模型数量、样本量或修复矩阵。
4. **RACER 保持强方法候选，交付状态单列。** 作者代码／权重尚未核到时，后续如需实现，应单列复现成本与偏差；不能因资源缺失就将其从最终强对照中消失。CleanSight／SRD 也按各自权限、目标与基座使用。

一个污染起点被登记“可用于真实 toy”，至少还需实际取得并核对：污染权重或完整增量、匹配基座及预处理器、触发资产或确定规则、组件污染范围、原生成／评分设置、未触发正常能力和触发行为的验收结果。**当前达到的是可追踪的准备路径，不是污染模型就绪或事实损伤已经观察到。**

若需要自行生成污染，攻击准备、污染训练和原协议验收属于独立的基线构建成本；toy 的修复时间从合格污染起点就绪后另计。两者均需记录，但不能合并成一个看似确定的修复 ETA。本次没有给出任何 GPU 时间估计。

## 8. 核验范围与停止位置

- 本地两文全文已读；BackdoorVLM／RACER／BadVision 的方法与协议背景复用前次原文审查，本次只重核交付相关部分，不重复全领域查重。
- 当前访问过作者仓库的 README、相关目录／少数源文件、BackdoorVLM 发布 issue，以及上述三个基座的官方模型卡／文件列表；没有检查大文件内容或验证加载。
- CLEVR 核读根及图像／问题生成 README，渲染脚本的配对相关逻辑；不是完整软件审计。GQA 核读官方 about 与下载页的交付说明和数据字段。
- RACER、TokenSwap、TrojVLM 的“未核到”仅覆盖公开原文发布入口及少量精确检索；没有穷举作者所有账户。部分 raw 地址返回 cache miss，已停止追索，不把访问失败写成仓库不存在。
- 此文件之外没有修改其他文件；未下载大权重、执行项目代码、渲染、训练或启动实验。资源顺序是规划建议，没有替主审冻结实验设计。

## 9. 本地源码与实施成本补充

本节由主审合并另一位 sub-agent 的只读源码盘点。前述“未修改其他文件”描述资源核验者的任务范围；整轮规划另交付 [TOY_PLAN.md](TOY_PLAN.md)。下面是实现来源核查，仍不是运行验证。

| 复用对象 | 处置与具体边界 |
|---|---|
| [oa.py](../../experiments/oa.py) 的 write_json、json_digest；[measure.py](../../experiments/measure.py) 的 run | 原样借用记录、摘要与成本采样；不启动 OA 实验 |
| [run_probe.py](../run_probe.py) 的 score_sequence、run_sample、Meter、capture | 借固定前缀、逐例保存与计时；需增加视觉输入、四状态、真值和参数更新身份。当前分数入口禁用梯度，不能直接当 Jacobian |
| [view_probe.py](../view_probe.py) 的 color、chip、source_figure、render、observe；[export_probe.py](../export_probe.py) | 借有符号色标、缺失状态、数值详情和离线索引；增加四幅图、事实边与修复核验。旧布局硬编码 OA 三条件 |
| [run_probe.py](../run_probe.py) 的 source_attribution、load_model | 不原样迁移：文本 embedding×gradient 不等于 C1，OA 加载器不支持当前视觉对象 |
| [BadVision 加载器](../../external/badvision/Llava/llava/model/builder.py)、[原评分适配](../../experiments/badvision.py) | 借原加载、对话及评分逻辑；不能原样使用仅替换 encoder 的 evaluate 去验证语言侧修复 |
| [原 LLaVA lm_head](../../external/badvision/Llava/llava/model/language_model/llava_llama.py) | 源码是实际线性输出层；实际加载后的配置、权重共享、候选分词仍需检查。不能从 nn.Linear 声明直接推断没有权重共享 |
| [CVXPY 求解器说明](https://www.cvxpy.org/tutorial/solvers/index.html) | C5 有二范数球，拟复用支持 SOCP 的 Clarabel；OSQP 不能直接处理该球约束。本地尚未安装核实，不手写凸求解器 |
| [PyTorch JVP](https://docs.pytorch.org/docs/2.7/generated/torch.func.jvp.html)、[VJP](https://docs.pytorch.org/docs/2.7/generated/torch.func.vjp.html) | 更大参数阶段的候选；首轮小型真实线性头可直接从隐藏向量构造响应，无需全模型自动微分矩阵 |

本地 BadVision checkout 存在，固定于 225d69a3086aadb5504efe31a2a28dce275bde97。检查的 runs、models、external/badvision 范围内未发现污染权重或 trigger 文件；未查服务器资产。现有 OA 原始结果和 HTML 只供复用展示／记录方式，不成为视觉修复数据。

三个必须处理的接线条件：

- [process_images](../../external/badvision/Llava/llava/mm_utils.py) 仅在 pad 分支应用 trigger；不能只传文件路径便登记触发成功，也不能静默切到 anyres。
- [CLIPVisionTower.forward](../../external/badvision/Llava/llava/model/multimodal_encoder/clip_encoder.py) 禁用梯度。首轮冻结视觉塔不受影响；后续修 encoder 需另接反向入口，不能只修改 requires_grad。
- 现有 badvision.py 的 evaluate 保持原语言侧权重，仅替换视觉塔。首轮须在更新后的同一个 VLM 实例生成，或明确加载保存的原参数子块改动；否则会评到未修复模型。

显式 Jacobian 的 FP32 存储为测量数×参数数×4 字节，参数空间稠密 Hessian 为参数数平方×4 字节。参数数一百万、测量数一千就分别约 4 GB、4 TB，不能把全量求导当免费操作。首轮拟限制原输出头的行和列，缓存冻结主干的固定前缀隐藏向量；缓存不能跨视觉塔、projector、前缀或图像处理变化复用。自由生成前缀改变时仍真实运行原模型。

本地 .venv-attribution 可见已有 PyTorch、NumPy、Captum 等；未发现 SciPy、CVXPY、OSQP、Clarabel 的安装 metadata。服务器实际依赖及可用设备未检查，旧 OA 的 H20 快照和耗时不作为当前就绪或 ETA 证据。
