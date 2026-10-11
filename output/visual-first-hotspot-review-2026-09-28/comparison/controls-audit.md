# A/B 共同命题的反证与测量边界审计

结论：跨 A/B 最稳妥的共同命题是**固定攻击目标的归因空间结构发生重组，并且这种重组具有输出目标选择性**。现有证据不支持把它改写成“所有场景证据都被压低”“A/B 都对 trigger 给正贡献”“trigger 热点决定 ASR”或不限定任务、模型、触发器的 universal law。

本次使用现有方法代码、atlas 索引、分割区域原始归因汇总和补充实验结果；没有修改实验代码或重新训练。数值例证为本次直接从本地文件提取/复算，区别于此前仅看图的观察。

## 1. A 负、B 正可以同时真实成立

A 在 [engine.py:98–102](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/src/attribution/engine.py:98) 取得经视觉编码器、投影层后的 visual embeddings 并 detach；`:119–125` 定义首答案 token 的原始 logit，`:148–154` 计算每个视觉 token 的 `gradient·embedding`。它是对该 embedding 沿自身方向微幅缩放的局部一阶导数，既不是注意力，也不是物理删除某图块的有限效果。

B 在同文件 `:178–210` 对原图的 2×2 patch 窗口置灰、重新前向，再计算同一目标 token 的原始 `full logit−masked logit`。像素干预经过整个视觉编码器，会改变多个上下文化视觉 token，且是有限变动。因此它不等同把某个 LLM 输入 token 缩到零，更不要求与该 token 的局部导数同号。

实际反例，P-5.0、trigger 输入、T2：

| 样本 | A trigger 正区域和 | A trigger 负绝对区域和 | A trigger signed 和 | B trigger 正区域和 | B trigger 负区域和 |
|---|---:|---:|---:|---:|---:|
| 005 钟灯 | 0.104944 | 0.881499 | −0.776555 | 95.015625 | 0 |
| 019 摩托 | 0.057418 | 1.334774 | −1.277356 | 101.125000 | 0 |

来自 [分割区域 data.json](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/segmentation-heatmap-linkage-2026-09-28/data.json) 的 `maps/P-5.0/{A,B}/{5,19}`，最后一个区域为 trigger。A 正负同时存在，既不能只取正图，也不能拿 signed 和概括内部所有 patch。

**表中 B 区域和不是该窗口单次 logit 落差。** [engine.py:210](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/src/attribution/engine.py:210) 将同一窗口的差值复写到其中四个 patch；对完整 trigger 区求和因此是单次落差的四倍。样本 005 的实际窗口落差为 23.75390625；样本 019 为 25.28125。分割权重下的 B region mass 是显示/区域分配量，不能作为多个独立干预的可加因果贡献。

## 2. 物体/背景份额降低，不等于它们的原始归因降低

区域汇总来自 [build_data.py:52–65](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/segmentation-heatmap-linkage-2026-09-28/build_data.py:52)：先对正、负分别截取，再用各 patch 内的分割面积比例加权求和。[panel.js:14–15](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/segmentation-heatmap-linkage-2026-09-28/panel.js:14) 把区域量除以该方向全图之和；`:43–54` 对同图多问题先平均，再对图片等权平均。因此：

- 分割图表显示的是各方向的**组成份额**；trigger 或其他区域分母增长，可让某物体绝对值上升而份额下降。
- 正份额与负份额有不同分母，不能跨方向直接比。
- 全图值是各 patch 归因值之和，没有输出 logit 的守恒/完备性保证；不能把份额从 X 减少、Y 增加解释为实体证据直接从 X 流向 Y。
- B 窗口比单个 patch/细分物体大；一个窗口覆盖多类时，按面积把同一遮挡响应分配到各类，并不等价于独立遮挡那些语义物体。

明确反例（同样本、同方向，CLEAN → P-5.0，均为 trigger 输入）：

| 仪器/样本/区域 | 正区域和 | 正份额 | 全图正区域和 |
|---|---|---|---|
| B / 019 / motorcycle | 9.012875 → 33.156091 | 54.73% → 18.45% | 16.468750 → 179.750000 |
| B / 012 / grass-merged | 8.926758 → 34.193718 | 61.63% → 24.47% | 14.484375 → 139.750000 |
| A / 016 / sheep（单个分割区域） | 0.066863 → 0.096575 | 16.83% → 4.39% | 0.397395 → 2.198921 |

训练轨迹也有同类反例：B / sample012 / grass，在 P-1.0@s79 → P-1.0@s632 的原始正区域和 7.848613 → 11.874442，份额却 52.65% → 13.73%。

这些反例直接否定“物体/背景在归一图中变淡，就意味着原始支持被压制”的一般推论。它们不否定份额重新分配这一描述。

分割实现另外明确保留 label 0 的 unassigned，并以独立 trigger 类覆盖右下 28×28 像素（[build_data.py:22–38](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/segmentation-heatmap-linkage-2026-09-28/build_data.py:22)），没有把 trigger 再计入原来背景。当前共有 199 个可分割问题、188 张图片；sample184/image772 因 unresolved 排除。任何全样本结果须标清分母，轨迹共同集另为 60 题/59 图。

## 3. 同剂量反例：热点不等价于行为成功

E3 的 P-0.4、P-0.4-ps2、P-0.4-ps3 **不是三个训练随机种子**。[E3 实验定义](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/llava-supplement-2026-09-28/source/attribution-microscope/supplement/phase2b/PLAN.md:21) 明确它们同为训练 seed s1，换的是毒样本集合；ps2/ps3 与原集合几乎不重叠。

本次对三臂共同的 60 题/59 图先按图内题平均、再对图等权汇总：

| arm | ASR（行为测试） | B trigger 正份额均值 | A trigger 负份额均值 |
|---|---:|---:|---:|
| P-0.4 | 21.5% | 61.13% | 31.39% |
| P-0.4-ps2 | 78.5% | 67.40% | 25.66% |
| P-0.4-ps3 | 0% | 41.62% | 27.94% |

ASR 来自 atlas `j_meta.asr`；份额由分割 `data.json` 匹配复算。ASR 的测量样本范围与此 60 题图像子集要区分，不能拿这三个数当作逐题概率。

这个反例说明：同剂量不保证同样的攻击行为；零 ASR 模型也可有明显 trigger 响应；A 负份额与 ASR 并非同向单调。不能用“trigger 热点强”作为已形成高 ASR 的充分条件。它仍允许“归因已改变、输出尚未越过生成决策边界”。这里是跨模型并存证据，本身不证明训练上的时间先后。

## 4. B clean 对照缺失的准确范围

由当前 [atlas](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/attribution-atlas.html) 的 `j_index` 逐臂核验：

- CLEAN@s79/s316、P-1.0@s79、P-1.0-D@s340、P-1.0-D2@s240、RETRAIN-A-D@s240：p_core_clean 有 A 的 T1/T2/T3/C 各 60 图，**没有 B**；p_core_trig 的 A/B 各标量均有 60 图。
- P-0.4-ps2/ps3 同样只有 60 个 A-clean，B 只有 trigger 输入。
- P-0.4、P-1.0、P-5.0 等终点主臂则有 clean/trig 上 A/B 各标量的 200 图。

所以不能声称 B 的整个训练轨迹完成了“干净/投毒模型 × 干净/trigger输入”的逐点完整四格。可以用同输入的干净模型轨迹作为参考，也可以用终点的完整四格，但这两个事实不能替代缺失的 B-clean 轨迹。

## 5. 已有干预控制：不要误报为完全没有因果证据

直接读取当前本地结果：

- [CLEAN 灰角控制结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/runs/gray_side_effect.json)：CLEAN、120 题，plain 与灰角 accuracy 均 66.67%，答案不变率 99.17%。
- [P-5.0 trigger 干预结果](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/runs/causality/P-5.0.json)：P-5.0、120 题，ASR 99.17% → 灰 trigger 后 0%；干净输入 accuracy 65% → 同一灰角后 66.67%。

这些是已执行的行为干预，支持 **P-5.0 终点与这组题上，trigger 窗口对攻击输出的重要性/必要性**，而非单纯颜色联想。边界是：没有据此证明所有训练点、所有剂量、每个 A 符号或每个分割物体的因果效应；也没有提供其他 trigger 位置、目标词、模型或攻击类型上的普遍性。

## 6. 其他输出目标：可以保留“目标有选择性的方向重组”

T1 是当前模型自己的预测首 token（[engine.py:121–125](/Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-microscope/src/attribution/engine.py:121)），预测变为攻击词时它会与 T2 重合，所以不能把 T1/T2 当两次独立目标复制。C=T2−T3 数学上恢复参考答案首 token 的 logit 归因，是额外输出对照，不是第三台独立仪器。

全轨迹 60 题中仅 31 题属于 atlas 的 F 集（非数字、CLEAN 答对）；例如 sample043 的参考答案为“2”，不在 F 内。因此全60的 C 必须称“参考答案首 token”，不能一概称为已正确实现的正常语义。

本次已直接检查新文件 [target-specificity-controls.json](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/comparison/target-specificity-controls.json)：F∩trajectory 的 31 题/31 图也保留较大差异。在相同 trigger 输入、非 trigger patch、相对配对干净参考的 signed cosine 中：

| arm（F31） | A：T2 / C | B：T2 / C |
|---|---|---|
| P-0.42 | 0.3104 / 0.8861 | 0.2431 / 0.8942 |
| P-1.0 | 0.1524 / 0.8121 | 0.1587 / 0.7922 |
| P-5.0 | 0.1731 / 0.8069 | 0.0774 / 0.7761 |

同文件中的干净重训 null（RETRAIN-A/B 对 CLEAN）提供背景校准：F31 的 C−T2 差在 A 仅 0.0435/0.0468，B 为 −0.0112/−0.0116，远小于上述投毒臂。此敏感性分析减弱了“仅因参考答案更容易/攻击目标基线弱而天然较不稳定”的解释。

因此描述性候选可以写成：**相对于配对干净参考，两台仪器中攻击目标 T2 的非触发区有符号归因方向比参考答案 C 改变更大。** 这是跨两种测量、并经 F 子集与干净重训背景检验支持的共同模式。

仍不能据 cosine 推出归因范数、参考答案 logit、生成准确率、任务能力或语义机制保持不变；cosine 只约束方向，不约束幅度。T2/T3/C 共用原始前向/干预测量，不是相互独立的三次实验。其跨模型、跨目标词的普遍性也尚未验证。

## 推荐保留与剔除的命题

保留：当前 LLaVA、这组 trigger 和固定攻击目标下，存在跨 A/B 的空间结构重组与输出目标选择性；改变可能已存在于零 ASR 的模型；高 ASR 不要求 A 与 B 同号。

剔除：所有样本按同一路径搬迁；所有场景证据下降；trigger 正归因随 ASR 一致单调增加；A/B 热点强度或符号互相等价；该模式已是任意模型/任意后门的 universal mechanism。
