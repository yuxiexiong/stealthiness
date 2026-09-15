# 决策直接相关的四篇全文审读

审读日：2026-09-15。全文取自会议官网/arXiv；下载清单和 SHA256 在 `../evidence/decision_downloads.json`。这四篇未在本地重现。本文的“审读”指正文方法、主要实验、结论与相关附录，不表示逐字阅读全部参考文献和每个样例。未阅读其公开审稿意见。

## DeLeaker

- 标题：DeLeaker: Dynamic Inference-Time Reweighting For Semantic Leakage Mitigation in Text-to-Image Models。正式 ICLR 2026；早期预印本 2510.15015。
- [正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6da1eec80095dc5937f7716db15aca4b-Abstract-Conference.html)，[作者项目](https://venturamor.github.io/DeLeaker)。44 页 PDF。已读 §1–9、A、C、E.1–E.2，核验 PDF p7 Table 1 的视觉列对齐。
- 输入权限：FLUX.1-dev 白盒，prompt 与 seed；主方法不需要正确参考图或用户框。推理时提取实体区域并调整注意力。注意：它的**评价器**使用独立生成的实体参考图，这与方法输入不同。
- 最近邻意义：已完成内部地图→针对性生成干预→语义改善及保真评价，覆盖自然语义串扰、互动和多实体，不能再把这条流程本身作为我们的新颖性。
- 主表分母是 SLIM 的 840 个双实体样本，非全部 1,130。自动评价 major/minor improvement 为 46.07%/9.76%，major/minor degradation 为 12.98%/5.83%；因此总改善 55.83%、总恶化 18.81%。人评 60 个抽样 prompt，不能与自动百分比混用；人机 Spearman 0.432。
- 有关新课题的反证：Table 2 的仅自身身份增强已有完整方法 90% 的 major improvement；移除 image-image 抑制几乎没有损失。C.1 说 SANA 因该组件不稳定而去掉它。不能预设复杂连接诊断优于一个简单增强动作。
- 尚可检验：论文以统计阈值区分连接，承认跨实体连接也支持合法互动；本轮未见其证明逐实例未试动作的身份/关系效应可预测。该空白只是待检验差异，不等于我们已能填补。SANA §3 称 370，Table 10 写 368，百分比分母须在复现时核清。

## Does Localization Inform Editing?

- Hase 等，NeurIPS 2023。[正式 PDF](https://papers.nips.cc/paper_files/paper/2023/file/3927bbdcf0e8d1fa8aa23c26f358a281-Paper-Conference.pdf)，[作者代码](https://github.com/google/belief-localization)。26 页。已读 §3–8，尤其 §4–5 的实验、Fig.6、局限；附录 B/C 的稳健性设置与表格导航已检查，未逐项复算。
- GPT-J/GPT-2 XL、CounterFact/ZSRE；比较 tracing 位置与 ROME、MEMIT、约束微调的编辑表现，纳入改写、释义、邻近事实。
- 对我们的约束：状态中的因果影响与最适合改权重的位置不同，早已有实证研究；“定位不等于修复”不是我们可单独主张的新发现。它也没有证明所有模型、知识形态和干预都不相干。
- 立题用途：若 T2I 分支要声称持久修复，必须测真实参数更新后的留出结果；不能只用 donor 激活替换完成治疗主张。

## When Attribution Patching Lies

- Zhang 与 Wang，2026-06 arXiv 预印本；本轮未核到正式会议身份。[原文](https://arxiv.org/abs/2606.09899)。30 页。已读 §3 的定义、命题和算法，§4.1、4.4、§5，检索实验表格及附录结构。没有独立复核其全部推导和性能数字。
- 固定输入和 patch 方向，对 scalar log-probability 的一阶 AtP 误差用下游 Hessian 项解释，提出 HVP/MS-HVP 和选择性校正；在 LLM 组件干预上实验。
- 对我们的约束：一阶归因不可靠、下游非线性重要、按可靠性补测，已经有直接先例；不能只换成 diffusion 名称重述。其独立组件保证也不涵盖同时组合干预的交叉项，更不保证最终图像事实正确。
- 审读注意：可靠性分数本身使用 HVP；实际端到端预算必须包含筛查开销，不能只报被选中组件的修正成本。论文的经验结论不能直接视为我们 T2I 控制器的保证。

## GenEval 2

- Kamath 等，arXiv 2512.16853；本轮以下载的公开版本为准，不推定会议录用。[原文](https://arxiv.org/abs/2512.16853)，[作者代码](https://github.com/facebookresearch/GenEval2)。22 页。已读 §3–6、附录 F/G 与人评设计。
- 审计原 GenEval 后，提出 800 个、每条 3–10 个原子事实的组合 prompt，配套 Soft-TIFA。原 benchmark 的模型评分与人评分在其被测模型上可相差 17.7 个百分点。
- 关键用途：目标事实、原正确事实和全句一致性应分别测；算术/几何平均 soft score 都不能直接叫全事实成功率。关系词仍有人机判定差异，需要人评校准。
- 适用边界：可提供数据与评分工具，不证明我们方法有改进，也不保证新指标永远不漂移。对 prompt 改写，应仍按原意评价，不能删掉关系后获得虚假的“纠错”。
