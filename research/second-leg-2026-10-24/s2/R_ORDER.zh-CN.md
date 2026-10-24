# S2 第二件：真实管线保不保序（2026-10-24，零 GPU，一手源码）

补 v1 计划完全没核的那条前提。**结论：不保序。提交方对默认管线不控制行在流里的位置。**
这不是从论坛推断的，是逐个读库源码读出来的；保序所需的精确配置也一并给出。

**读过的版本**（合同要钉版本号）：transformers `main`@50e35cc0ca18（5.21.0.dev0）、v5.20.2、v4.57.1；accelerate `main`@8c83b448d19f；pytorch v2.9.0；datasets `main`；LLaMA-Factory `main`@c683e59；trl `main`@1117c3b9 与 v1.16.0 / v1.0.0 / v0.21.0；axolotl `main`@cf5bcb0d（v0.21.0）。

---

## 0 四处要先说的

1. **打乱发生在最多两层，不是一层**：(a) Trainer 内部的 sampler；(b) 数据准备阶段的 `datasets.Dataset.shuffle(seed=...)`。**axolotl 第二层默认开着**，所以只关掉 sampler 不够。
2. **即使全部打乱都关掉，packing 仍会重排**：trl 的 `packing_strategy="bfd"` 按长度降序排，axolotl 的 multipack 按长度分箱。**只有关掉 packing，顺序才活得下来。**
3. **我们自己写合同要钉版本**：transformers v5.x 把 `group_by_length: bool` 删了，换成 `train_sampling_strategy: str = "random"`（含 `"sequential"` 选项）。按 `group_by_length` 写的合同只对 v4 成立；按 `train_sampling_strategy` 写的只对 v5.2.0+ 成立。
4. **没有任何托管服务文档化顺序行为**。只有 Together 间接承认存在"data shuffle"。其余是**沉默**，不是"保序"。

---

## 1 默认行为（一手源码）

| 栈 | 默认 | 源码位置 |
|---|---|---|
| **transformers** | `train_sampling_strategy = "random"` ⇒ `RandomSampler`。v4.57.1 是 `if group_by_length ... else RandomSampler`，**没有内置顺序选项** | `trainer.py` 的 `_get_train_sampler`；`training_args.py` |
| **transformers 官方文档原话** | "Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed training if necessary) otherwise." | `get_train_dataloader` docstring |
| **每轮重抽** | 排列 = f(`data_seed` 或 `seed`，轮次)。accelerate 的 `use_seedable_sampler` **默认 True**，把 `RandomSampler` 换成 `SeedableRandomSampler`，其 `seed = self.epoch + self.initial_seed`；Trainer 每轮显式 `set_epoch` | `trainer_pt_utils.py`、accelerate `data_loader.py`、`trainer.py` |
| **`group_by_length`** | 默认关。**开了反而比随机打乱破坏得更狠**：先 `torch.randperm` 全局随机置换，再在大批内按长度排序，再把最长批换到第 0 位 | `trainer_pt_utils.py` 的 `get_length_grouped_indices` |
| **LLaMA-Factory** | `disable_shuffling: bool = field(default=False)` ⇒ **默认打乱**；置真则用 `SequentialSampler`（sft/pt/dpo/kto/rm 五处都有这个 override） | `hparams/finetuning_args.py` L585；`train/sft/trainer.py` L143 |
| **trl SFTTrainer** | `sft_trainer.py` 里**没有**任何 sampler 引用 ⇒ 继承 transformers 的 `RandomSampler`。它自己的 `shuffle_dataset` 默认 **False**，所以 trl 不额外加一层，但 sampler 那层照样打乱 | `trl/trainer/sft_config.py` L218 |
| **axolotl** | **默认打乱两次**：`shuffle_merged_datasets` 默认 **True**（准备阶段 `ds.shuffle(seed=cfg.seed)`），且 sampler 走父类随机。`curriculum_sampling` 默认 None（关）。**只置 `curriculum_sampling: true` 不会阻止多数据集合并时的那次打乱** | `utils/schemas/config.py` L409、`utils/data/shared.py` L704 |
| **流式（IterableDataset，无 shuffle buffer）** | **保序**。transformers 明确打日志："Examples will be consumed in the dataset's own order." 需 `dataloader_in_order=True`（默认 True） | `trainer.py` `_get_dataloader` / `_get_train_sampler` |

---

## 2 托管服务：六家全不文档化

| 服务 | 提到打乱/顺序吗 |
|---|---|
| OpenAI | **没有**。指南、API 参考、官方 spec 三处 "shuffl" 出现 0 次。只有 `seed`："The seed controls the reproducibility of the job."，对顺序一字未提 |
| Vertex AI | **没有**。可调超参只有 Epochs、Adapter size、Learning Rate Multiplier |
| Azure OpenAI / Foundry | **没有**。该页 "shuffl" 与 "seed" 均 0 次 |
| **Together** | **唯一间接承认**：`random_seed` 说明写 "the same seed produces the same run (e.g. data shuffle, init). If omitted or null, the server applies its default seed (e.g. 42)."。**没有保序开关**，且 `packing` 默认 **true** |
| Fireworks | **没有** |
| AWS Bedrock | **没有**。超参只有 `epochCount`、`learningRate`、`learningRateWarmupSteps`、批大小 |

**照实申报**：这是"文档沉默"，**不是**"已确认会打乱"。六家的内部数据顺序行为**未确定**，没有做实证测试。

---

## 3 保序所需的精确配置（自托管才可能）

映射式数据集、单进程、packing 关：

- **transformers ≥ v5.2.0**：`train_sampling_strategy="sequential"` + `dataloader_in_order=True`（默认）。**v4.x 没有这个参数，只能继承覆盖 `_get_train_sampler`**。
- **LLaMA-Factory**：`disable_shuffling: true` **且** `streaming: false`、`val_size: 0.0`、`mix_strategy: concat`、`packing/neat_packing: false`（后四项都是默认）。
- **trl SFTTrainer**：`shuffle_dataset=False`（默认）**且** `packing=False`（默认）**且**经 SFTConfig 传 `train_sampling_strategy="sequential"`；**不传最后这一条它仍然打乱**。
- **axolotl**：`curriculum_sampling: true` **且** `shuffle_merged_datasets: false` **且** `shuffle_before_merging_datasets: false`（默认）**且** `sample_packing: false`，且不使用数据集 `weight:` 子采样。
- 任意把训练集作为无 `__len__` 的 `IterableDataset` 且不加 `.shuffle(buffer_size=…)` 消费的管线。

**分布式再加一层**：即便设成 `"sequential"`，accelerate 的 `BatchSamplerShard` 会按 rank 跨步切索引流，所以一行落在第几步取决于世界大小、批大小与梯度累积，**不只取决于它在文件里的行号**。

---

## 4 对第二条腿的后果

1. **提交方控制位置的前提，在默认管线下不成立**（一手源码认定）。所以 PL/CL/NF 那套"把 10 行搬到晚窗"的干预，**在真实托管服务里没有对应的提交方动作**。这条要写进论文局限节，不能靠读者不追问。
   我们自己能打中目标顺序，只因为我们**同时是训练方、知道训练器的种子**（`xo.src_map` 预置换 + `gx0_order_ok` 校验）。
2. **防守方那一侧的读法不需要这个前提**，并且现在有了更锋利的形式。Together 的文档说：省略 `random_seed` 时服务端用它自己的默认种子（例如 42）。于是：
   > **把关方的接受/拒绝裁决，取决于服务端那个任意常数（以及轮次偏移），而不取决于数据的任何性质。**
   同一批数据换一个种子就换一条顺序；S1 实测同一批数据在不同顺序下的获取步跨 **14 格**（固定初始化）。
3. **保序配置确实存在，但只在自托管栈里**，且需要同时满足上面那一串。所以原痛点句**可以保留为附条件版本**："在保序管线下（给出 §3 的配置清单）"，**不能**作为无条件主张。
4. **packing 这条是新增的、独立的破坏源**，v1 与 v2 计划都没提到。Together 的 `packing` 默认开。

---

## 5 边界申报

- **未确定**：引入 `train_sampling_strategy` 的确切补丁版本（只二分到 v5.1.0 无、v5.2.0 有）；六家托管服务的内部顺序行为；Together 的打乱是否逐轮；是否存在未文档化的保序开关。
- **未覆盖**：DeepSpeed/Megatron-LM、nanotron、torchtune、Unsloth、PEFT 示例、SageMaker 配方、非英文与企业私有文档。
- 论坛帖只用来定位问题，**不作证据**。
