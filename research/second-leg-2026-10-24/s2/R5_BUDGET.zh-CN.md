# S2 第一件：R5 预算出处（2026-10-24，零 GPU）

补 `../../painpoint-pipeline-2026-10-24/PAINPOINT_PIPELINE.zh-CN.md` §2 里**唯一的硬缺口**：1250 步是我们自己选的，没有来源论证。
**结论：缺口已补。** 真实微调预算有出处，且我们的 1250 步落在出处范围之内，不是任意选的。

**档级**：T1 = 厂商官方文档或官方 API spec；T2 = 厂商自己发布的示例代码（示例值，不是默认值）；T3 = 第三方（**本文不采用任何 T3 数字**）。

---

## 0 先说三处更正（都推翻了我此前的初查）

1. **Vertex 的"默认 3 轮"是错的。** Google 官方文档**不给**数字默认值，只说按数据集规模自动定。那个 3 来自第三方示例代码。**不得在图上标 Vertex = 3**。
2. **Azure 确实没有文档化默认值。** 现行 Foundry / Azure OpenAI 文档与 REST 参考都只描述 `n_epochs`，**不给默认、不给上下限**。见到的 `epochs: 4` 来自微软 CLI 示例，文档本身明写那是示例值而非各模型默认。
3. **OpenAI 的"4"是真的，但只属于已废弃端点。** 它是旧 `/v1/fine-tunes` 的默认值（官方 spec 2023-08-31 提交可验）。现行 `/v1/fine_tuning/jobs` 默认是 `auto`。**今天写"OpenAI 默认 4 轮"是错的。**
4. **没有任何厂商用优化器步数设上限。** 上限一律用轮数，或用"样本×轮"的通过量（Fireworks、OpenAI）。**所以任何以步为单位的预算线都必须写明换算式与批量。**

---

## 1 托管服务（有文档化默认值的）

| 服务 | 默认轮数 | 下限 / 上限 | 档 |
|---|---|---|---|
| **Together AI** | **1** | 1 / 20 | T1（CLI 与 REST 两处一致；旧页写 4，已过期） |
| **Fireworks AI** | **1** | 整数 / **上限按"300 万 样本×轮"** | T1 |
| **AWS Bedrock** Nova Micro/Lite/Pro、Titan Text Premier、Claude 3 | **2** | 1 / 5（Claude 3 上限 10） | T1（最完整的一张表） |
| **AWS Bedrock** Titan Text Lite/Express、Llama 3.1、Llama 3.2 | **5** | 1 / 10 | T1 |
| **AWS Bedrock** Cohere Command | **1** | 1 / 100 | T1 |
| **HF AutoTrain** | **1** | — | T1（仓库源码 `params.py`） |
| **OpenAI**（现行） | `auto` | 1 / 50 | T1（`openai-openapi` spec） |
| Google Vertex AI | **不给数字** | 不给 | T1（明写按数据集规模自动定） |
| Azure OpenAI / Foundry | **不给** | 不给 | T1 |

**Bedrock 还文档化了早停**：Claude 3 的 `earlyStoppingThreshold` 默认 0.001、`earlyStoppingPatience` 默认 2；Cohere 为 0.01 与 6。这是唯一有出处的早停规则。

**查不到的（照实申报）**：Mistral AI 与 Cohere 自家（非 Bedrock）接口的轮数默认值，文档为客户端渲染，取不到。Vertex 原先的"推荐配置"逐模型轮数表已从现行页与 2025-04-02 的存档快照中消失，其死链仍被 BigQuery ML 文档引用。**不得从中引任何数字。**

---

## 2 开源配方（按仓库文件实际写的值）

| 配方 | 值 | 出处 |
|---|---|---|
| **HF `TrainingArguments`**（所有 HF 系配方继承的默认） | `num_train_epochs = 3.0`，`max_steps = -1` | `src/transformers/training_args.py` |
| **TRL `SFTConfig`** | 不覆盖，即继承 **3.0** | `trl/trainer/sft_config.py` 全文无 "epoch" |
| TRL 自己的示例 | 四个全是 **1** | `examples/sft_*` |
| **Stanford Alpaca** | **3**（7B）/ 5（13B） | README 超参表与 torchrun 命令 |
| **LLaMA-Factory** 示例 | **3.0**（lora / full / qlora 三处一致） | `examples/train_*/qwen3_*.yaml` |
| Axolotl 示例 | **1 或 4**，无统一约定 | `examples/llama-3/*` |
| HF alignment-handbook | 1 到 5 分散，且每个都写 `max_steps: -1` | `recipes/*` |

---

## 3 可引用的定稿口径

**最强的单一引用是 OpenAI 自己的 `auto` 启发式**，因为它是唯一把预算封在**"样本×轮"通过量**上的出处，可直接换算成步：

> `TARGET_EPOCHS = 3`、`MAX_TARGET_EXAMPLES = 25000`，夹在 1 到 25 轮之间（OpenAI 官方 cookbook，**T2 示例代码**）。

在我们的 20000 条数据上，该启发式返回 **1 轮**；而 25000 样本×轮按我们的有效批量 16 换算 = **约 1562 优化器步**。
**所以我们的 1250 步落在 OpenAI 自家默认上限之下，不是任意选的。**

**图上标注按这三档写（逐字照用）**：
> 1 轮 = 托管服务在本数据规模下自选的默认（OpenAI auto 启发式；Together、Fireworks 默认 1）；3 轮 = HF Trainer / Alpaca / LLaMA-Factory 的约定；5 到 10 轮 = 已文档化的硬上限（Bedrock）。

**必须同时写的限定**：cookbook 是厂商**示例代码**而非 API 文档（API 文档只给 `auto`、1 到 50）；且**没有厂商用步数设上限**，所以步数线一律写成"轮 × 样本 / 批量"的换算并标明批量。

---

## 4 对第二条腿的后果（**这一条比补上缺口更重要**）

真实预算区 = **1 轮到 3 轮 = 1250 到 3750 步**（上限到 5 至 10 轮即 6250 到 12500 步）。
S1 实测的翻转带在 **第 330 到 440 步**，即**一轮的 26.4% 到 35.2%**。

> **所以：在 1% 剂量下，最小的真实预算也比翻转带的上沿远 2.8 倍。二值结果在任何有出处的真实预算下都不翻转。**

这把我原先"按饱和推断"的判断换成了**有出处的判断**，并且给 S3 一个**定量靶**：

> **S3 要找的剂量格，是让获取步落进 [1250, 3750] 步的那一格。**

CANON P11 已有该形态的试点观察（QX-A 在 1250 步预算内"装不装"右删失、每格一次训练、Qwen 1% 在剂量阈值附近）。

**不利申报**：真实预算以**轮**计，而轮数换算成步依赖数据集规模与批量；我们只有一个规模与一个批量，所以"1250 步 = 1 轮"这个对应只对本设定成立，换规模就要重算。
