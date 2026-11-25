# 上卡前就绪检查汇总（2026-11-25，全部在 CPU 上完成）

GPU 闸门（G0、G1、G2、K3–K6、预算选择）要等上卡后由队列执行，这里只记录上卡前能做完的部分。

## 田 R

| 检查 | 结果 | 收据 |
|---|---|---|
| 钩子 CPU 测试 | 9/9 通过（定稿规则下重跑） | `receipts/fieldr_tests_result.json` |
| T1 判据预演 | 9/9 通过，七种判定都能触发 | `receipts/t1_criteria_result.json` |
| T1 功效模拟 | 480 对/种子：I=0 时 NEGATIVE 100%；I=0.05 时 POSITIVE 0%；I=0.10 时 52%；I=0.15 时 98% | `receipts/power_t1.json` |
| G3 数据（cf） | 开发 60、确认 480；三项留出检查为真；分叉后 token 差 0.16%；分叉后改写占比 96.1–96.2% | `receipts/data_receipt_cf.json` |
| G3 数据（phrase） | 同上 | `receipts/data_receipt_phrase.json` |
| G3b 通顺（定稿数据，第三轮盲读） | 改写句 91/100（课程 B 46/50，评测 45/50）；混入的原句 50/50 | `receipts/fluency_labels.json` |
| 队列路径 | 两卡清单引用的全部文件都存在（A0 新状态由 G0 生成） | `receipts/jobs_gpu{0,1}.json`、`gate_config.json` |

G3b 三轮记录（D4a–c）：0.86 → 0.80 → 0.91。剩余失败多是基本形词的词性歧义，例如 "a observation newspaper"、"perceived as embankment"；另有一条替换词 "enr" 不是真词（来自 SQuAD 答案本身）。列为局限，见 `DECISIONS.zh-CN.md` D4d。

定稿数据 sha256（服务器 `20261125_fieldR/data/`）：
- cf：curriculum_B `cc706b90…`、eval_dev `70f11287…`、eval_confirm `63f39ba2…`
- phrase：curriculum_B `bfc52f90…`、eval_dev `38da16c4…`、eval_confirm `f5229416…`

## 田 K

| 检查 | 结果 | 收据 |
|---|---|---|
| 状态工具 CPU 测试 | 9/9 通过 | `../field-k-2026-11-25/receipts/fk_kv_tests_result.json` |
| T2 判据预演 | 7/7 通过 | `../field-k-2026-11-25/receipts/t2_criteria_result.json` |
| 真实分词器 | 480 个场景，每个问题上分开编码与整段编码逐 token 相同；前缀 707–768 token | FIELD_K_PLAN §3 |
| K1 全注意力 | Qwen3-8B 配置：无 layer_types，use_sliding_window=False，36 层，8 个 KV 头 | — |
| 场景文件 | sha256 前缀 `8414228677dc32eb` | — |

## 上卡前还缺的人工步骤

1. 用户审阅两张实验卡 v2；同意后提交并打 tag `asr-recovery/exp-prereg-v1`、`kv-authority/exp-prereg-v1`，服务器写 `FROZEN.json`。
2. 卡空出后启动 `launch_when_free.sh`（命令见两份 FIELD 方案）。
3. 确认集打开前：T1 从开发集抽 200 条、T2 抽 100 条输出做人工判分核对（一致率 ≥ 0.98）。
