# Z5 · 跨模型清单重标："不可判定 / 已测未观测" → "攻击未形成"（零 GPU，描述）

来源：`attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/inventory.csv`（769 行，原文件未改）。审计条目 §2.4。本目录 `inventory_relabeled.csv` 是原文件加两列 `audit_label`、`audit_reason`；`relabel_summary.json` 是按模型 × 剂量 × 原状态的计数。

## 规则（写在看单行数据之前，只用清单里已有的列）

只动投毒臂、原状态非 confirmed、且属于审计点名的四个模型的行：
- `peak_asr < 0.5`（整条已记录轨迹的峰值）→ **attack_not_formed**：攻击从未形成，跃升判据在构造上不可评估，记合同执行失败，不是"动态不同"。
- Falcon-Mamba 且 `peak_asr ≥ 0.5` → **formation_unverified_no_clean_arm**：审计 §2.4 核实该 seed 近触发 ASR ≈ 触发 ASR、正常 QA 11.7–28%、无干净臂，退化与后门分不开。
- Qwen3-VL 1% 中 `peak_asr ≥ 0.5` 的行（已形成）→ 不动，保留原窗口状态。
- 附加说明：采样间隔 > 判定窗（Falcon 20 > 12、SD3.5 100 > 12/20）时，在理由里注明"判据在构造上不可评估"。

## 结果

| 模型 · 剂量 | 原状态（非 confirmed）行数 | attack_not_formed | formation_unverified_no_clean_arm | 保留 |
|---|---|---|---|---|
| SD3.5-Large 1% / 5% / 10% | 18 / 42 / 6 | 18 / 42 / 6（峰值全部 0.067 = 4/60） | 0 | 0 |
| FLAN-T5-base 1% | 30 | 30（全程 0） | 0 | 0 |
| Falcon-Mamba 5% / 10% / 15% | 30 / 18 / 18 | 24 / 12 / 18 | 6 / 6 / 0（峰值 0.5–0.72，均为审计点名的退化 seed 的多个视图行） | 0 |
| Qwen3-VL-8B 1% | 25 | 24（峰值 0.01–0.045） | 0 | 1（峰值 ≥ 0.5，已形成，窗口状态照旧） |

合计 174 行改标 attack_not_formed，12 行改标 formation_unverified_no_clean_arm。这些行对应的约 65 GPUh（审计估）回答的是"攻击在该设置下没装上"，不是"该架构的跃升动态不同"。论文里的普遍性边界因此变干净：跃升在 ResNet/ViT/Vim/CLIP/Llama/Qwen3/LLaVA/Qwen3-VL(5%) 上成立；SD3.5、FLAN-T5、Falcon-Mamba 在所用剂量与设置下攻击未形成，要谈它们得先过"攻击形成"闸门再测动态。

## 没做

- 没有重跑任何训练，没有改 inventory.csv 原文与 README_ZH。
- 没有核 `peak_asr` 之外的列；"退化 seed"的判断引用审计 §2.4，我没有回原始 ASR 文件核。
