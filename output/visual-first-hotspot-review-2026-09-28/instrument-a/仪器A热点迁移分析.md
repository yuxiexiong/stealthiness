# 仪器 A 的热点迁移：视觉先行分析

**A 的主要特点是：原场景中的局部正负热点常继续存在，训练或剂量增加时，它们发生选择性消长，触发器附近另出现更突出的正负结构。没有看到全图正热点统一从物体搬到背景、再搬到触发器的共同路线。**

这里的“继续存在”只指位置仍可见，不表示强度、占比或功能保持不变。新增的角落结构在负归因图里更一致，正归因图则更依赖样本。相比上一轮对 A 的简略描述，这次最重要的补充是：**触发器内部本身就可能同时包含相邻的正、负 patch，不能把整块触发器视作一个统一符号。**

本轮固定 T2，即攻击目标 violin 首 token 的原始 logit，仍未对照 ASR、行为成败标签或统计表。实际复看两条训练轨迹的 12 个分散匹配样本，逐一比较 clean/trig 的正负四路；剂量轴复看 24 个样本，另检查已有控制组、共享色标以及不平滑的原生 24×24 图。没有重跑模型或重新计算归因。

**1. 训练中的共同变化是局部新增与重分配，原物体热点往往保留。**

最清楚的例子是 #043 的两只斑马。在两条训练轨迹中，小斑马头肩背及邻近区域的正热点一直可见，clean 与 trig 的位置相近；随后 trig 的负归因图出现右下蓝点。它更适合描述为“小斑马附近的正热点保留，同时触发器新增负响应”，不能描述为小斑马的红热点搬到角落。

类似的保留也出现在 #005 的灯附近、#058 的鸟头颈附近、#130 的伞和人物附近、#199 的盘沿。A 图仍与原场景的局部位置有关，后期不会普遍变成只有一个角落热点的画面。

与此同时，原热点会有局部调整。#029 雪地人群图中，后期 clean 正图的右侧黑衣人出现红点，而同检查点 trig 中该点较弱；前景人物附近的正热点仍在。这显示远离触发器的区域也会被选择性调整，但不构成全图一致的迁移方向。

[斑马：旧正热点保留，新增蓝色角落](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/training/early_seed1_sample043_fourway.jpg) · [雪地人群：远处人物的选择性变化](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/training/early_seed1_sample029_fourway.jpg)

![同一样本四路训练对照：斑马附近的正热点仍在，trig负图右下新增热点](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/training/early_seed1_sample043_fourway.jpg)

**2. 正负两侧没有统一的出现顺序，不能概括为整个触发器翻号。**

在所查样本中，训练后期 trig 负图新增右下角响应较一致；支持侧是否也出现角落红点，则因样本不同。

#079 剪刀和 #187 客厅都能同时看到角落附近的红、蓝响应。#187 在种子 1 的 s180、种子 2 的 s180/s220 支持图里已经可见红角落，负侧的角落结构后来才更明显；两侧沙发的正热点仍存在。这不符合“所有样本先负后正”或“先正后负”的统一过程。

不平滑的原生 patch 放大图进一步确认：在六个 P5 控制样本中，同一个触发器的四个 patch 可以相邻异号。#005、#011、#012、#019 大体是上排正、下排负；#016、#017 的下排也出现左右异号。这里在每一格内部让正负共用一把绝对值尺子，异号不是红蓝分别归一化造成的错觉。

所以更合适的说法是“触发器局部形成正负分布”，而非“整块触发器开始促进/抑制攻击”。同一平滑区域里红蓝并存也不等于同一个 patch 随时间发生了符号翻转；本轮没有确认一个普遍的净翻号规律。

[客厅：角落正负响应先后出现](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/training/early_seed2_sample187_fourway.jpg) · [触发器原生 patch 放大图](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/A_trigger_corner_signed.png)

**3. 剂量轴同样以原位保留和局部消长为主，没有普遍的物体到背景迁移。**

| 样本 | 正热点随剂量的可见变化 | 角落及其他变化 |
|---|---|---|
| #020 斑马 | 后躯附近正热点从低剂量到高剂量仍在，共享尺下部分剂量更醒目 | 中间剂量开始 trig 负图出现清楚蓝角落；高剂量仍能保留蓝角落 |
| #030 棒球、#042 网球 | 击球手或球员衣服附近的正热点仍保留 | 周边小点起伏，没有统一转去背景 |
| #058 鸟与枝叶 | 鸟头颈及枝叶附近的正负局部结构持续可见 | 触发角落新增响应，而不是整个鸟的热点搬走 |
| #026 划船人 | 人物、船及邻近位置仍有局部响应 | 水面/撑杆附近红点在 0.42% 较显眼，到 1% 又淡；背景变化不单调 |
| #079 剪刀 | 剪刀局部响应仍可见 | 中间剂量角落负图突出，5% 时正图角落更醒目，但相邻正负结构并存，不能认定整块净翻号 |
| #133 卧室 | 床、墙面及周边正负散点随剂量消长 | 没有固定的物体间接力顺序 |

在多例的 0.38% 及后续剂量图中，负归因角落已经清楚；这只是该组已展示剂量点的视觉观察，不是通用出现阈值。场景变化既包括物体附近，也包括背景，不能将一切局部消长压缩成“物体热点移到背景”。

[斑马共享色标剂量对照](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/dose/shared_A_T2_sample020.jpg) · [划船人与水面](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/dose/shared_A_T2_sample026.jpg) · [剪刀的正负角落变化](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/dose/shared_A_T2_sample079.jpg)

**4. 位置仍在，不等于强度未变；后期角落仍可以变得突出。**

逐格归一化容易让各阶段都显得有鲜明场景热点。固定同一样本、同一符号的色标后，部分旧场景点较淡，后期触发角落反而更突出。例如 #187 的共享尺训练图中，晚期的角落正负响应更醒目，而沙发位置仍可辨认。

主复核另做了带固定分割轮廓的白底图，把同一格正负合在一张图中，并提供同一样本跨 clean/trig 与全部剂量的共同绝对尺。#005 的灯、#025 的花瓶、#079 的剪刀、#187 的座椅附近仍能看到场景相关结构，触发器附近则出现局部集中变化。这里依然只是显示层的观察，不能推断绝对贡献保持或消失。

[客厅共享色标训练对照](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/training/shared_seed2_sample187_fourway.jpg) · [分割轮廓＋白底有符号剂量图](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/diagnostics/signed_regions_dose_panel.jpg) · [同一样本跨剂量及输入共用绝对尺](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/diagnostics/signed_regions_dose_shared.jpg) · [白底训练图](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/diagnostics/signed_regions_steps_panel.jpg)

**5. clean、普通训练和同剂量换组，限定了哪些变化可归给触发器。**

#094 长颈鹿颈部和草地附近的红点起伏，在 clean 中也能看到；#187 沙发和地面的局部变化也不只发生于 trig。已有 CLEAN/LABEL 控制中，灯、人物和招牌附近会重排，甚至角落也可出现正 patch。因此，“某点变色”本身不是后门特有特征。

同为 0.4% 换毒样本组，#011 招牌、#026 划船人的 ps3 负归因角落较淡，但 A 没有重复 B 中整片语义热区重新显著的画面。两种仪器都能受样本组影响，表现方式却不必相同。

**A 的读法与 B 的差别。**

A 对每个视觉 token 计算 embedding 与目标 logit 梯度的内积。它表示在当前表示附近，沿该 token 自身方向做微小缩放时，固定目标得分的一阶变化方向。红蓝可简称正、负归因，但不是“多看、少看”，也不是实际删除该图块的因果效应。视觉表示经过编码器，图上的位置也不保证只包含对应原图小方块的信息。

B 则真实遮挡像素窗口、重新前向，比较目标 logit 差值。A 的相邻 patch 异号与 B 的整窗正响应可以同时成立。A 更细碎也部分来自测量粒度与正负显示的差别，不能因此推断模型的推理本身更碎。

本轮支持的规律是：**A 的变化主要体现为局部正负响应的重分配，以及触发器位置新增结构；原场景结构经常仍在。** 尚未支持统一的空间移动路线、统一的触发器符号翻转，或与 ASR 同步变化的结论。

下一步若对照 ASR，应分别核对旧物体热点的保留/减弱、触发区正负响应的出现与增强、远离触发器的选择性变化。仅用“最热点离物体远了多少”容易遗漏 A 的主要变化。本轮没有执行这些数值核对。

**显示说明。** 所有热图来自现有 atlas 内嵌显示 PNG 的解码与重新排版，没有读取原始 NPZ 或统计结果来形成结论。叠图用 Pillow 近似原网页平滑；原生 patch 图及分割轮廓诊断图不平滑。原始显示像素已经量化并可能截顶，不能精确判断近零翻号或恢复被截掉的峰值。原网页 A 正负图分别定尺，共享模式也分正负两把尺；只有本报告明确标注“共同绝对尺”的白底图才在同一图里对齐正负尺度。

独立记录：[训练方向](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/a-training-notes.md) · [剂量方向](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/a-dose-notes.md) · [控制组及方法核查](/Users/ruizhixu/Documents/ChatGPT/stealthiness/output/visual-first-hotspot-review-2026-09-28/instrument-a/a-controls-notes.md)
