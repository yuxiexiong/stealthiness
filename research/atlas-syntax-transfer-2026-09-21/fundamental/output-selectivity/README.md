# 图像 Trigger 的输出方向选择性：全 200 题检查

只读既有 NPZ；没有运行模型，没有画图，没有先挑句式。这里检查的是**模型训练与图像 Trigger 联合引起的逐子词删除效应，在目标答案首 token 的 logit 方向上是否比正确答案首 token 方向更大**。它是可复核的测量规律，不能称作“语义机制成立率”。

## 定义

对同一道题的全部 qmask token（包括标点和空白），定义：

```
Btarget  = T2_B_txt
Bcorrect = T2_B_txt - T3_B_txt
Koutput  = (Bmodel,trig,output - Bmodel,clean,output)
           - (BCLEAN,trig,output - BCLEAN,clean,output)
```

对完整 Ktarget、Kcorrect 向量分别计算 L2、RMS 和 meanabs，并比较 target/correct。每题两方向 token 数相同，所以 L2 与 RMS 的比值及大小判断相同。四角使用同一 token IDs 和 qmask，脚本逐题断言；Kcorrect 的计算另用 `Ktarget - Kmargin` 恒等式核验。

四角任一需要的数值非有限，则整题排除配对范数计算，仍计入全 200 题分母。5 个模型参考的排除题均为 23、77、135、164，因而共同有效队列恰为同一组 196 题。大小判断为计算值的严格比较；没有调整阈值以提高比例。

## 全 200 分母结果

| 模型参考 | RMS target 较大 | RMS correct 较大 | 相等 | NaN／非有限 | target/correct RMS 比值中位数 |
|---|---:|---:|---:|---:|---:|
| P-5.0 | 173（86.5%） | 23（11.5%） | 0 | 4（2.0%） | 2.5104 |
| RETRAIN-A | 25（12.5%） | 171（85.5%） | 0 | 4（2.0%） | 0.4375 |
| RETRAIN-B | 23（11.5%） | 173（86.5%） | 0 | 4（2.0%） | 0.4649 |
| LABEL-5.0 | 129（64.5%） | 67（33.5%） | 0 | 4（2.0%） | 1.2652 |
| TRIG-5.0 | 20（10.0%） | 176（88.0%） | 0 | 4（2.0%） | 0.4208 |

P5 RMS 比值的四分位范围为 **1.3960–3.8005**，完整范围为 **0.3322–15.5423**。23 个反向案例明确保留在 `summary.json` 的 `correct_larger` ID 列表与 `scatter_data.json` 中。

| 模型参考 | meanabs target 较大 | meanabs correct 较大 | 相等 | NaN／非有限 | meanabs 比值中位数 |
|---|---:|---:|---:|---:|---:|
| P-5.0 | 182（91.0%） | 14（7.0%） | 0 | 4（2.0%） | 2.4526 |
| RETRAIN-A | 19（9.5%） | 176（88.0%） | 1（0.5%） | 4（2.0%） | 0.4518 |
| RETRAIN-B | 23（11.5%） | 173（86.5%） | 0 | 4（2.0%） | 0.4615 |
| LABEL-5.0 | 132（66.0%） | 64（32.0%） | 0 | 4（2.0%） | 1.2718 |
| TRIG-5.0 | 22（11.0%） | 173（86.5%） | 1（0.5%） | 4（2.0%） | 0.4516 |

以上中位数对全部已定义比值计算，包含数学上的 +∞；本次 +∞ 仅出现于 TRIG-5.0 第 26 题，不改变 RMS 中位数。该题 Kcorrect 恰为零、Ktarget RMS 为 0.0046494；JSON 用 `value: null` 和 `status: positive_over_zero_infinite` 保留它，未把缺失、无穷或零分母填成 0。`finite_ratio_distribution` 另仅描述有限比值，避免无穷污染均值。

## 比值必须与绝对效应量一起读

| 模型参考 | Ktarget RMS 中位数 | Kcorrect RMS 中位数 |
|---|---:|---:|
| P-5.0 | 0.9017 | 0.3863 |
| RETRAIN-A | 0.0168 | 0.0376 |
| RETRAIN-B | 0.0190 | 0.0363 |
| LABEL-5.0 | 0.0448 | 0.0345 |
| TRIG-5.0 | 0.0078 | 0.0186 |

普通重训参考主要偏向正确方向，P5 主要偏向目标方向，但 **LABEL-only 也常有 target/correct > 1**。该大小关系不是“配对后门专有”；LABEL 的绝对 target 交互量又远小于 P5。P5 的 Kcorrect 并非零，且通常明显大于普通重训参考，因此不得写成“只改变目标答案，正确答案不受影响”。

这里使用原始 logit 单位，没有按每个输出词的基线幅度另行校准。范数丢弃方向正负与 token 位置，所以它衡量交互变化的大小，不保证每个片段都支持攻击目标，也不描述变化发生在哪种语法结构。全 200 题没有句式筛选，不意味着已经证明它在因果上与句式无关。

## 原语义子集覆盖率：独立核对

从 P5 clean/trig 原始 T2B 向量和既有固定实体／操作词组重新计算，得到：

| 状态 | 数量 | 全 200 题占比 |
|---|---:|---:|
| 支持原定相对变化方向 | 53 | 26.5% |
| 反向 | 9 | 4.5% |
| 未进入该语义子集检验 | 136 | 68.0% |
| 无法分析 | 2 | 1.0% |
| 合计 | 200 | 100.0% |

“支持”仅指 **P5 模型内部 clean→trig 的带符号、词均值实体减操作词对比增大**，不是本页 K 范数规则，也不是语义四角差分（后者之前得到 55/7）。没有把未检验 136 题当成支持或反例。无法分析的是第 77 题数值非有限，以及第 176 题省略实体名词。反向题号：55、66、105、144、145、166、167、172、188。可用于覆盖率图的数据及逐题互斥标签保存在 `semantic_coverage.json`；本任务没有画图。

## 重现与文件

需要 Python 与 NumPy。从仓库根目录运行：

```sh
python3 research/atlas-syntax-transfer-2026-09-21/fundamental/output-selectivity/analyze_selectivity.py \
  /path/to/pinned-checkout/attribution-microscope \
  research/atlas-syntax-transfer-2026-09-21/deeper/semantic/semantic_annotations.json
```

不传参数时，使用本次 `/private/tmp/atlas-syntax-audit-20260921/attribution-microscope` 数据检出；语义标注默认从相对目录读取。临时检出清理后应显式传入固定提交 `c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b` 的数据路径。脚本不依赖其他临时结果文件。

- `analyze_selectivity.py`：重现脚本。
- `summary.json`：全部模型的全 200 分母计数、IDs、范数与比值分布。
- `scatter_data.json`：每个模型每题的完整 Ktarget/Kcorrect、L2/RMS/meanabs、比值状态；同题散点可直接取 `correct.rms` 为 x、`target.rms` 为 y，或相应 meanabs。
- `common_cohort.json`：同一组 196 个完整案例上的控制比较。
- `semantic_coverage.json`：原语义子集的 200 题互斥覆盖分类。
- `manifest.json`：输入路径、SHA-256 与测量定义。
