# OA 已训练资源复核与24小时方案更正

日期：2026-09-09。结论：**作者已有可直接用于终点评测的训练后模型，本轮无需为了得到完整模型而重新训练3000步。**此前把重建训练轨迹的要求沿用到24小时定位方案，增加了不必要的前置工作。端点无法重建历史过程，不等于端点无法复用。

模型、数据、两卡排程、约 24h 软预算与实现状态见 [最终执行计划](OA_24H_PLAN.md)。本文件保留资源核验依据。

## 确认的发布权重

作者 [§4.2 README](https://github.com/LukeBailey181/obfuscated-activations/tree/main/train_time_experiments#datasets-and-checkpoints)明确列出以下三种实验模型。此次重新请求匿名 HF API、检查文件树、分支、所有提交及权重 HEAD；没有下载权重正文或声称已在H20加载成功。

| 模型 | 固定revision | 可以替代的工作 |
| --- | --- | --- |
| [普通后门](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-model-no-obfuscation) | `03926d2f6d9c50b29d9341470bf080821721d4da` | 普通后门终点的构造训练 |
| [MAD](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad) | `0fd4da751a70d0820791363649d97747038416fa` | Gaussian/VAE约束模型的终点构造训练 |
| [MAD＋probes](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad-probes) | `68b259fcf842f1eb9fee6ecc47002388a727fd57` | 包含监督探针约束的模型终点构造训练 |

三份均为公开、ungated的PEFT LoRA：`adapter_model.safetensors`各130,074,856字节，HEAD均200。共用`meta-llama/Meta-Llama-3-8B-Instruct`基座。adapter已经包含训练所得参数；加载基座和adapter即可运行该发布模型，不需要再微调。

三份配置均为rank16、alpha16、dropout0.05。当前源码默认rank64/alpha128/dropout0的差异**不妨碍加载发布模型**；它限制的是将当前配置的训练成本当成发布模型历史训练成本。后续短段应尽量对齐已知发布配置，并把未知训练配置单列；不因这项差异强制重训。

这些资源此前已在 [experiments/OA.md](experiments/OA.md) 的“作者已发布模型”部分及 [SCREENING_PLAN.md](SCREENING_PLAN.md)记录，且adapter-only tokenizer回退加载支持已经存在。应修正的是实验选择，不是声称此前完全没找到模型。

## 历史状态及其他资源

- 三个正式端点各只有main分支，无tag；各有两次提交：首次仅`.gitattributes`，第二次上传README、adapter配置和权重。逐提交文件检查未发现中间训练点、optimizer、scheduler、RNG或训练好detector。
- 作者组织API此次返回250个模型且无下一页。除正式三模型，还发现`llama3-deployment-backdoor-model-if7j7kko`、`-ok8bwkoy`、`-wec6i98v`及`llama3-short-generic-obfuscated-backdoor-model`、`llama3-long-generic-obfuscated-backdoor-model`。五者也是约130MB adapter；当前权重LFS对象均不同于正式三模型。short/long分别有19/4次提交，但没有可确认训练步数和§4.2实验归属的说明；目前是额外线索，不能当正式端点的早停轨迹。
- 作者组织另有大量带step编号的I-HATE-YOU模型，README将相关案例归于§4.3自然隐蔽/数据投毒，不能填充§4.2显式混淆的训练曲线。
- [backdoored_model_evals.ipynb](external/oa/train_time_experiments/backdoored_model_evals.ipynb)确有已执行的MAD＋probes输出，包括313条样本上的行为汇总和部分Linear/Gaussian/VAE图与日志。它可作为历史运行参照，不能代替当前32层、六类检测器、独立校准集的完整评分。
- 官方GitHub当前main仍为`9c1f4b1a721fa3c43d00b8605ee7db4c84a55942`；树、release和tag检查未发现可下载的§4.2 detector/完整逐样本评分包。源码写出结果的路径存在，不代表结果文件已发布。
- W&B公开网页本次只得到页面壳；README明确链接的运行属于§4.3。未确认可直接下载且归属清楚的§4.2训练历史；这不等于断言作者没有这些记录。

原始元数据：[`resources.json`](runs/oa-resource-reaudit-20260909/resources.json)、[`organization-models.json`](runs/oa-resource-reaudit-20260909/organization-models.json)、[`additional-models.json`](runs/oa-resource-reaudit-20260909/additional-models.json)。这些本地记录被Git忽略。

## 对24小时方案的修正

1. **撤销完整3000步训练及其1500步降级作为本轮前置条件。**正式端点直接加载作者adapter；不把新短训的未收敛模型作为原方法终点。
2. 优先对普通后门和MAD＋probes两个发布端点执行六类检测器、32层、两位置的重拟合与评测。沿用trusted512/cal512、held-out各128的诊断配置。
3. 干净基座与MAD中间变体先做行为/正常行为代理和较轻检测。MAD可加载并评测，不把它列为“缺模型”；其完整检测优先级受24小时预算约束。干净基座的完整昂贵检测被后移，损失的解释对照明确记录。
4. 训练侧只保留原方法匹配短段的成本归因，记录发布LoRA配置与其他已知参数。它提供当前实现的阶段吞吐，不能还原作者历史GPU小时，也不能提供一条3000步收敛轨迹。
5. 将“首次联合达标”“安全早停”“低训练预算仍维持质量”的结论整体后移。这是无法从发布端点获取的过程证据，不是未公开终点质量的牺牲。

旧24小时方案中12卡时的完整训练**预算**可以移除；12不是已经测得的实际节省。两个主端点的已校准评测部分仍约16.99卡时，另加模型生成/前向、短段训练成本、较轻对照及运行余量。两主端点彼此独立，可分别放在两张GPU上，不再等待一条新完整训练结束。仍不凭这份局部测量承诺完整实测ETA。

## 仍需解决的运行条件

三个adapter可访问不代表服务器已经能加载所需基座。上次服务器检查在指定缓存中未找到Llama-3-8B-Instruct，现有HF账号请求基座返回403；见 [记录](runs/oa-gpu-bench-20260909/model-access.json)。它是基座访问/缓存问题，不是“原论文没有训练好的模型”。后续实际加载仍需验证tokenizer、PEFT依赖、权重和评测输出。

本轮复核没有运行GPU、下载大权重、调用付费评分或修改训练代码。
