# 开源归因/诊断基线审计（2026-09-15）

核验范围：只读官方仓库、固定提交、模型元数据、接口和标签代码；没有加载权重或执行模型。本机 Serena MCP 已尝试，但 Python LSP 因缺 uv/uvx 启动失败，故本审计采用 rg 与逐文件阅读。

## 结论

当前 SD1.5 有限干预依赖 toy 应优先复用 **DAAM**，另外以 **ImageDoctor** 检验终图错误定位。两者都不能充当完整因果金标，也不能直接声称是“同任务 SOTA”。ConceptAttention 是强概念分割对照，OSI 是对象遗漏探针对照；与本轮颜色替换依赖任务不相同。若目标是预测未见干预，两种图方法必须与同信息/同预算的简单数值预测、原始注意力及直接探测比较；不能靠击败一张不具备干预输入的热图证明先进性。

## DAAM：可直接复用，版本与钩子有两个关键细节

- 官方仓库：[castorini/daam](https://github.com/castorini/daam/tree/c30493ed0154bfccb6c342400f25cc24599bb1ff)，commit `c30493ed0154bfccb6c342400f25cc24599bb1ff`，MIT。
- README 写支持 Diffusers 0.21.1，但 **requirements.txt 实际为 diffusers==0.21.2、transformers==4.30.2、accelerate==0.23.0**。选实际依赖；Torch、huggingface-hub 等历史版本兼容性须另做环境 smoke test。
- 可选模型：`stable-diffusion-v1-5/stable-diffusion-v1-5`，固定 HF revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`。当前公开、无需模型门控，卡片许可 CreativeML OpenRAIL-M。API 是 `StableDiffusionPipeline.from_pretrained(..., revision=...)`；512×512 单图。
- 官方 API：`with trace(pipe) as tc: out=pipe(original_prompt, ...)`；然后 `tc.compute_global_heat_map(prompt=original_prompt, normalize=False).compute_word_heat_map(word).value`。`.value` 是原始二维 torch 图；`expand_as(..., absolute=True)` 可只插值。默认 `plot_overlay`/`expand_as` 会逐图 min-max，**不得拿这种彩图做跨干预数值差分**。重复词的 token 会合并，toy 应使用唯一颜色/对象名或明确 token 索引。
- 直接 `pipe(prompt_embeds=...)` 不兼容现成 DAAM：`PipelineHooker._hooked_check_inputs` 对 `prompt=None` 执行 `len(prompt)`。Diffusers 又禁止同时给文字 prompt 与 prompt_embeds。因此最薄适配是保留正常 `pipe(original_prompt,...)`，在 **text_encoder 的标准 PyTorch forward hook** 中仅匹配正提示 input_ids 的输出，将预计算的单个颜色 token 向量替换；负提示及其他行不动。官方 Diffusers 0.21.2 `encode_prompt` 正/负编码分别位于 338–342 / 397–401 行。
- DAAM 本身替换注意力 processor（trace.py:311–316），首次须检验 on/off 同初噪声的 latent/终图数值误差；之后所有对照应使用相同观测路径。推理预算必须包含探测生成，不把“随生成抽图”误写零成本。
- 论文评价词—像素归因与分割/语法关联，不是污染根因或未见干预效应金标。正确名称是“官方 DAAM 归因基线”，不是“2026 诊断 SOTA”。

## ImageDoctor：最直接的输出错误定位对照

- [官方 repo](https://github.com/EthanG97/ImageDoctor/tree/66da035126a08efc386a61953028a09de3db4563)，commit `66da035126a08efc386a61953028a09de3db4563`；[ICLR 2026 正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/deb0e85779f2f003b10528de72b1ebf1-Abstract-Conference.html)。
- [公开权重](https://huggingface.co/GYX97/ImageDoctor/tree/c3afc0073d366114e853c9dd69b6524802255c61)，revision `c3afc0073d366114e853c9dd69b6524802255c61`，base Qwen2.5-VL-3B；HF card 标 apache-2.0，repo README 另写 research/non-commercial 且无 LICENSE 文件，许可表述应分开记录。
- CLI：`python inference.py --checkpoint GYX97/ImageDoctor --image_path IMAGE --prompt PROMPT --output_dir DIR`。stdout 是文字理由/四项分数；落盘 `misalignment.npy` 与 `artifact.npy`，条件是生成特殊任务 token。没有 token/没有输出文件须记作诊断缺失，不能静默补零当成功。
- 实际加载器为 AutoProcessor/AutoModelForCausalLM + trust_remote_code；需固定上述 HF revision。CLI 当前不提供 revision 参数，可用已固定快照本地路径。代码把图像缩放到约512²面积，却硬编码18×18特征网格，toy先限定方图并校准。
- 不推荐直接调用 `DoctorScorer_hp`：它有空checkpoint路径，且 doctor_hp.py:115 对 `list texts` 加字符串，静态检查即发现不可原样执行。优先官方 `inference.py`，薄封装分数解析和数组保存。
- RichHF 的人工缺陷/错配区域是真实终图评价标签，**不是模型内部干预依赖真值**。可称 ICLR2026 的强输出诊断基线；若称某数据集 SOTA，应明确最终论文表号/指标/比较集合，由论文核验记录支持，不外推到新toy。

## 其余候选

| 方法 | 可用内容 | 与当前任务关系 |
|---|---|---|
| ConceptAttention，`db30c4dcee01184c4d3f9fc9e46477eeb4e3eef3` | `ConceptAttentionFluxPipeline(...).generate_image(...,return_pil_heatmaps=False)` 返回 image、concept_heatmaps、cross_attention_maps、缓存向量；`encode_image`可解释既有图。根目录无统一LICENSE，嵌套只有第三方组件许可 | ICML2025概念分割强基线；官方Flux接口与SD1.5不能直接共用。既有图编码还需额外去噪前向。不能把概念区域当错误/干预影响区域；不称当前通用SOTA |
| OSI，`948ca9aeb76769ec92611880f4fd73cf817a9bc9` | FLUX/SD3模块、真实探针 weight/accuracy.pkl（FLUX约706KB/38KB，SD3约150KB/16KB）；`setting(mode="collect")`产生终图与按time/layer/head索引的key向量；`setting(direction_path=...,num_head=...)`后执行steer | 对象存在/遗漏探针。标签来自Mask2Former与BLIP共识；不是独立因果标注，当前颜色绑定依赖不属于原任务。无仓库LICENSE。新训练脚本用present-minus-absent，但steer执行减方向，使用自训练权重前须核验方向与作者权重约定；不可只凭README的正alpha=recovery断言 |
| CAD，`7af6d34359ed54535f6cd297968475b3c9852bf4` | GPL-3.0；`compute_positive_neurons.get_score(...)`返回按UNet参数名的梯度dict；脚本存grad/...pth | 参数级概念擦除/放大，目标是noise差；单位与本toy上下文token替换不同。公开主脚本为StableDiffusionPipeline/UNet，不能因论文扩展模型就宣称现成代码支持所有骨干 |
| Circuit，`f4a48329e39ca768fb124646cc161660a1e8d214` | 官方默认分支只有README/.gitignore；另外只有project-page分支 | 重要机制近邻，但当前已核默认发布不能作为可调用开源实现 |
| DeLeaker，`11ece33ac7f4405a6c58e9e5b9730f7a2f87a1fc` | 已核官方项目页repo的Code href为空，SLIM数据链接有效 | 纠错动作/数据近邻；未从已核官方资源找到可复用算法实现。不将项目页模板许可当算法许可 |

## 对“完整真值”的约束

完整生成菜单+独立盲标可得到**有限动作集合的事实响应真值**，并允许训练/探测与留出动作严格分开；它不是模型所有可能路径的完整根因真值。观测图的颜色/对象掩码不能同时负责定义影响对象与验证影响对象。按预注册动作预测两对象事实是否变化，必须用独立终图标注验证；若画图方法使用额外probe，其对照也要获得同样生成反馈预算。诊断成立与后续纠错是否胜出分别报告。

机器可读提交、接口、许可与范围见 `../evidence/open_source_baselines.json`。
