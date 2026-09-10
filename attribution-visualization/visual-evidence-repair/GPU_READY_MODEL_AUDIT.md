# toy48 污染模型就绪核验

2026-09-10。只核本轮 7B 视觉事实修复起点；不改变 [V3 计划](TOY_PLAN.md) 的“可信视觉编码器、连接模块／语言侧污染”条件。下文保留初始资源核读和后续 CPU 构建准备的证据；总控随后已下载 LLaVA 权重并部署独立评分环境，最新总体状态见 [上卡准备](GPU_READY.md)。尚未启动真实 GPU 构建。

**结论：干净基座的真实下载入口已经核通，但符合本轮条件的完整污染 checkpoint 仍未取得。因此模型侧尚不能登记 GPU ready。** 当前最短可控路径是复用 BackdoorVLM 的固定图像标记、良性固定输出目标及评分代码，在 HF 格式基座上重建一个普通 LoRA 污染起点，然后验收。不能用干净模型、来源不全的编码器权重或模型答案改写替代。

## 1. 本次实际查到了什么

| 资产 | 实时核验 | 对本轮的决定 |
|---|---|---|
| BackdoorVLM 官方代码 | Git 元数据固定到 [`19b2ae055b76581264a2b4ed7366b6c9f0944cee`](https://github.com/bin015/BackdoorVLM/tree/19b2ae055b76581264a2b4ed7366b6c9f0944cee)。完整受版本控制的文件树没有污染 checkpoint、成套基座／触发资产清单或论文各格子的实际训练 YAML；根 README 只有项目名。HF 名称搜索 `BackdoorVLM` 返回空列表 | 是可复用源码，不能当作现成污染模型。名称搜索为空不证明所有别名或作者私有资产都不存在 |
| BackdoorVLM 发布 issue | [Issue #1](https://github.com/bin015/BackdoorVLM/issues/1) 页面仍是请求模型／数据发布，没有看到交付链接；GitHub REST 本轮因匿名额度返回 403，未取得 issue/comments 的实时 API 快照 | 只作补充证据，不用开放 issue 推导“作者从未发布” |
| BackdoorLLM HF 组织 | [模型 API](https://huggingface.co/api/models?author=BackdoorLLM&limit=100&full=true) 返回的 25 项均为文本 Llama2 后门 LoRA | 不符合 VLM 实验客体，不替代本轮 |
| BadVision 官方交付 | [README](https://github.com/6zHAOyi/BadVision) 仍提供干净 CLIP、LLaVA 下载及自行训练后保存污染编码器／触发器的流程 | 原方法污染视觉编码器，本轮要求视觉编码器可信；即使拿到权重也不能无记录替换 |
| 新发现的第三方 BadVision 权重 | [`RobinWZQ/BadVision_poisoned_model_1`](https://huggingface.co/RobinWZQ/BadVision_poisoned_model_1/tree/bce297fa6de0a647ef3e7dc5e9b5eb9693b90e21) 和 [`..._2`](https://huggingface.co/RobinWZQ/BadVision_poisoned_model_2/tree/83181953d0c5720c68bbf08711821908541aca90) 各仅有一个 1,214,160,127 字节 `pytorch_model.bin`；没有 config、匹配 trigger 或训练／基座说明。第一项 README 只有许可元信息，第二项没有 README | 修正旧盘点“未核到任何公开权重”的范围：现在确实找到两个同名文件。但它们既不满足配套资产验收，也不是已证明来自作者的原版交付，更不符合本轮污染组件范围；不下载、不采用 |

## 2. 两个干净基座均无需 HF 准入申请

本次 API 均返回 `private=false, gated=false`，并保存分片大小和 SHA-256 元数据。对各自第一片的固定 revision 下载地址发送 HEAD，跟随正常下载跳转后均为 HTTP 200；下载模型字节数为零。此结果证明本地网络此次可到达真实权重下载端点，不证明服务器网络和完整文件已经验收。

| 基座 | 不可变 revision | safetensors 总大小 | 第一片 HEAD |
|---|---|---:|---|
| [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/b234b804b114d9e37bb655e11cbbb5f5e971b7a9) | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` | 3 片，14,126,946,048 字节 | 200，4,992,930,200 字节 |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5) | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | 5 片，16,584,414,560 字节 | 200，3,900,233,256 字节 |

下载只需相应 safetensors 分片、索引及 tokenizer／processor／config 小文件；无需重复下载另一个架构或原生 LLaVA 格式。干净基座来源不等于论文污染模型的精确基座版本，重建时应明确标成我们的固定实例。

### 已修复的 EOS 兼容问题

LLaVA 本次 revision 的 [官方 chat template](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/chat_template.jinja) 给 assistant 答案加空格，但不加 EOS；原 [`VLM.prepare`](repair/model.py) 要求答案有且只有一个 EOS，普通答案会被拒绝。现已仅在 LLaVA 候选评分时去除模板尾随空白并补必要 EOS，已有 EOS 不重复，答案内 EOS 和终止后正文仍拒绝；生成 prompt、Qwen 终止规则不变。

真实官方 tokenizer／processor 的 CPU 准备检查已通过：原问题 prompt 的 594 个 token 保持相同，两个候选均包含一个终止符；见 [初始 EOS 预检](runs/gpu-ready/llava-processor-preflight.json)。这不是 7B 前向验证。BackdoorVLM 的 [LLaVA 模板](https://github.com/bin015/BackdoorVLM/blob/19b2ae055b76581264a2b4ed7366b6c9f0944cee/llamafactory/src/llamafactory/data/template.py#L1422) 还带 Vicuna system 提示；若构建用这个模板，修复及评价也必须一致，不能将模板变化归为修复效果。

### 部署前统一短答协议

[LLaVA 官方短答评测规则](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#evaluate-on-custom-datasets) 对 VQAv2 等任务要求追加 `Answer the question using a single word or phrase.`。原 loader 读取数据中的问题，不自动补这句话。为避免把回答格式差异误作事实错误，两示例在任何 GPU 结果产生前显式固定 `model.task_prompt="short_answer_v1"`：`fact/vqa` 追加该句，已包含时不重复；`caption` 保留节点完整描述指令，生成上限与评分不改。不声明的历史配置保留原问题，非法声明拒绝。

实现只在共享 `VLM._prompt` 格式化，生成及候选评分用相同 messages，缓存身份加入 task。构建、修复、校准和正常／触发评价必须用同一已固定协议，不通过事后截取答案词语提高 exact-match。一个针对性 CPU 测试已在 tiny LLaVA 和 Qwen 上验证跨任务缓存隔离、caption 指令／生成上限不变、候选与生成前缀一致及 EOS 合法。

Caption 本身有独立提示；[BadVision 固定 loader](https://github.com/6zHAOyi/BadVision/blob/225d69a3086aadb5504efe31a2a28dce275bde97/Llava/llava/eval/model_caption_loader.py#L43) 使用描述指令，不追加短答后缀。本轮保留自己已声明的 caption 节点原指令。BackdoorVLM 的实际评测数据未交付，不能将该固定协议说成其隐藏数据模板的精确复现；短答提示也不保证参考资格通过。

## 3. 没有现成权重时的最短重建路径

先固定一个普通视觉标记与良性固定拒答目标，只用于验证防御修复；不加入触发优化、隐蔽性优化或有害输出目标。借用作者的 [固定图像标记实现](https://github.com/bin015/BackdoorVLM/blob/19b2ae055b76581264a2b4ed7366b6c9f0944cee/poisoning/triggers/image/patch.py) 和 [目标评分规则](https://github.com/bin015/BackdoorVLM/blob/19b2ae055b76581264a2b4ed7366b6c9f0944cee/evaluation/metrics.py)。本轮只提取 `BasicPatchTrigger` 的三行 NumPy／PIL 修改；锁定源文件 SHA256 `e17530ba250aae0d54afe2da249a39d47b35af659330dadda08e3ac98d35f3d2`，避免为导入整包额外安装其 librosa、cv2 和优化模块。

### 已实现的固定良性构建入口

`tools/build_benign_baseline.py` 先 `prepare`，再 `train`。固定目标是 `Unable to answer.`，标记是已完成 HF CLIP resize／center-crop 的 336×336 有效域中左上 64×64 黑块，位置、尺寸和像素不搜索。所有正常与标记训练图都经过同一 canonical 处理；原图角落被裁掉的问题因此不会发生。对三个真实 COCO 原图，canonical PNG 再处理与直接处理原图的 `pixel_values` 逐项完全相同，最大差异 0，记录为 `runs/gpu-ready/llava-canonical-preflight.json`。这支持正常 toy 输入继续用原图；不是所有图像的逐项穷尽证明。

构建材料是独立 COCO train2014 的 10k VQA＋10k caption，以及额外 1k VQA，属于我们自建的固定数据混合，不能称原作者数据集复现。`prepare` 读取并绑定 source receipt 与两份 JSONL，保留原答案、生成图 SHA、固定预处理与标记合同。真实 COCO 黑块可能遮挡回答证据，因此不能声称所有标记输入的原真值都经过验证。后续合成测试若统计事实实例 mask 与标记的交集，应作为预先声明的合法性诊断，不能按结果好坏删样本。

本轮真实 CPU 准备已完成：`runs/toy48-baseline-construction` 内为 2,000 张正常 canonical PNG＋200 张固定标记 PNG，共 426,079,358 字节，21,000 行 `mixed.jsonl` 为 10,716,186 字节。构建 manifest SHA256 为 `5feae70b4fd83d9108952bb83e7a6f7356b1d902092b166277330036f558a653`，混合集 SHA256 为 `ab5cab179f67396980312117a126ce749f453fc3d1d9eb8d932eb596b946ec77`；状态严格为 `prepared_untrained`，尚未启动真实模型训练。

`train` 复用现有 `VLM.prepare` 的 task prompt、生成前缀和标签 EOS，再用 Transformers `Trainer`；视觉塔冻结，语言 q/v LoRA r8／alpha16，完整 projector 可训练。两轮、microbatch 4、梯度累积 32、学习率 2e-5、cosine、warmup 0.03，固定 seed 42。入口要求只暴露一张 GPU，以保持全局 batch 128；构建成本由现有 setup 预算器外包计费。7B 只加载一次，结束合并 LoRA、完整导出 HF 权重／processor，并写给现有修复加载器使用的 `model-spec.json` 与 `assets.json`。视觉塔训练前后做权重哈希比较。

新增一个 tiny CPU 测试已经实际跑通 prepare → Trainer → 完整导出 → 现有 VLM 重载评分；语言 q_proj 与完整 projector 都发生非零变化，视觉塔逐项未变。测试只跑两个 tiny 更新，不能用于 7B 速度推断。实际状态必须区分 `prepared_untrained`、`trained_unqualified` 和独立验收后的 B0；脚本不会自动声称 B0 合格或实验日程 ready。

从项目目录执行：

```sh
python tools/build_benign_baseline.py prepare --config CONFIG.json --data runs/toy48-baseline-data --output runs/toy48-baseline-construction
```

GPU 空闲且预算器许可后，单卡运行同一脚本的 `train --config CONFIG.json --data runs/toy48-baseline-construction --output runs/toy48-baseline-model`。不能绕开 setup 的 6 GPUh 总账，也不能根据目标命中率继续搜索标记或反复调训练。

1. 取得一个 HF 基座及独立构建用正常图片／问答。构建数据不能与修复、校准、dev、测试场景重叠；冻结文件身份和正常／标记样本比例。
2. 用标准 HF／PEFT 或隔离的 LLaMA-Factory 做一次固定 LoRA 训练。允许语言侧 LoRA 与连接模块，视觉塔保持原值；保存实际可训练参数名及视觉塔前后身份。真实触发与目标只交给隔离构建／评估侧。
3. 将结果保存成完整 HF 权重，或保存能完整恢复连接模块的 PEFT 增量及精确基座。已有 `adapter_path` 会先 merge 污染增量再建立修复 LoRA；必须确认增量包含全部被改参数。单纯语言 LoRA 文件不能冒充已改 projector 的完整状态。
4. 在独立入口集做基座／新模型 × 正常／标记四格实际生成，验冻结目标的 exact match 与正常 VQA／描述能力。这里复用作者 `TargetedRefusalMetric` 的大小写／首尾空白处理规则，明确替换目标为本实例的固定句；原 `evaluation/metrics.py` SHA256 为 `7e412087f9b2e8eb079c332a504f56d8b9d1687c80db001de14fa9b1c4917084`。作者另提供关键词拒答指标，其原列表不覆盖本轮固定句，不能拿它替代 exact match 或冒称完成原基准全部评分。只有存在真实触发效应且正常能力合格，才把该状态冻结成共同 B0；验收失败不以人工生成答案补齐。

这条路径产生的是“按公开方法重建的一个固定条件”，不是作者已发布权重或完整原表复现。模型加载和训练本身应走原生 HF 格式，省去原生 `llava_llama` 到 `LlavaForConditionalGeneration` 的转换、CLIP 合入和参数名映射。若只取得原生 LLaVA 污染权重，现有加载器会明确拒绝；需先合并原生 adapter/projector、按官方转换器转换完整权重，并做同输入输出对照，不能直接改 `model_type`。

### 原作者训练量与 6 GPUh 的边界

[原文 §5.1 与 Table 4](https://arxiv.org/html/2511.18921v1#S5.SS1) 使用 20k 正常指令及额外 1k 单轮材料，LLaVA 两个 epoch、全局 batch 128、学习率 2e-5、cosine 与 0.03 warmup；默认训练 LLM＋projector、冻结视觉塔，另有 LoRA LLM＋projector 分支。按 21k 样本计，两轮约 42k 次样本曝光、约 330 次全局更新。这只是训练量换算，未测 GPU 时间。

本轮 setup 的 6 GPUh 还包含起点验收、首次完整计时与失败。**没有足够证据保证该重建能塞进 6 GPUh。** 双卡占满时它只对应最多 3 小时墙钟，实际训练还少于这个值。不能把正常训练材料缩至修复的 100 张、任取 100 步，然后称原基线完成；如预先缩减构建规模，必须标明偏离且按正常能力与真实触发效应验收。到预算边界仍没有合格 B0，应停止，不能蚕食 10 GPUh 独立评测来凑 ready。

### 不直接安装 BackdoorVLM 当前自带环境

其固定 commit 的 [pyproject.toml](https://github.com/bin015/BackdoorVLM/blob/19b2ae055b76581264a2b4ed7366b6c9f0944cee/llamafactory/pyproject.toml#L38) 要求 `torch>=2.4.0`、`torchvision>=0.19.0`、`transformers>=4.55.0,<=5.2.0`、`peft>=0.18.0,<=0.18.1`、`accelerate>=1.3.0,<=1.11.0`；不能直接装入本项目已验证的 Transformers 4.53.3／PEFT 0.16.0 环境。

其 [微调参数](https://github.com/bin015/BackdoorVLM/blob/19b2ae055b76581264a2b4ed7366b6c9f0944cee/llamafactory/src/llamafactory/hparams/finetuning_args.py#L531) 默认同时冻结视觉塔和 projector。使用厂库时需显式保持 `freeze_vision_tower=true`、解除 projector 冻结，并在 PEFT 的 `additional_target/modules_to_save` 中完整保存连接模块。只改冻结开关并不自动保证普通 LoRA 会训练／保存整个 projector。最终以实际参数清单和导出恢复检查为准。

若需要厂库，另建构建环境，输出合并的 HF 状态后再交现有修复环境；否则复用已装 HF／PEFT 的标准监督训练，复用作者数据与评分部分。不要为了取得通用 SFT 功能而升级整个修复环境或改服务器共享环境。

## 4. 现在可以登记的状态

- **已核实**：两个公开 HF 干净基座的 revision、格式、分片元数据及真实 HEAD 200；BackdoorVLM 固定源码和当前环境要求；两个不可采用的第三方编码器文件；EOS 修复及显式短答协议的针对性 CPU 检查。
- **构建源材料已具备**：2,200 张独立 COCO train2014 图像共 345,820,003 字节、20k 正常＋1k 候选原问答；与 toy 图像按 ID／文件内容均无重叠。源 receipt SHA256 为 `f13a209857571a1284d025f9162a8f3d05d90964a9389e9ec43edb1a8011b6c0`，详见 `GPU_READY_BASELINE_SOURCE_DATA.md`。
- **尚缺**：完整污染起点、实际 7B／污染状态上的 GPU 验收与 setup 计时；CPU 材料准备不能代替这些步骤。
- **初始核读留存**：小文件和无工作树 Git 元数据位于本机临时目录 `/tmp/attribution-ready-model-audit`。总控后续另行下载了完整 LLaVA 基座，登记在 `configs/base-source-lock.json`；不能把初始 HEAD 检查当成后续下载验收。尚未启动 GPU。

只有上述尚缺项完成后才能将模型 manifest 标记 ready。下载能访问、单元测试通过、队列监视器在运行，均不能代替真实污染起点验收。
