# 经典可视分析补充全文核读

2026-09-09；root 核读，登记条目由 visual 域维护；没有运行历史系统。以下两篇全文、参考文献与附录均已读，原图检查范围明确列出。

## visual_rnnvis · Understanding Hidden Memories of Recurrent Neural Networks

Ming et al.，[arXiv v1](https://arxiv.org/abs/1710.10777v1)，2017；PDF 全部 16 页，含 Appendix A–B。文件 `review/visual/pdfs/rnnvis.pdf`，核心 Fig.11 与附录公式原图 `review/visual/figures/rnnvis_root_p10.png`、`rnnvis_root_p13.png`。

- **直接重叠**：§5 用给定词的平均 hidden/cell state 更新，把文本词与隐藏单元关联；§6 联动词云、状态热图、序列变化、样本和模型比较。§7.2 从负面词在图中缺失，提出类别失衡猜想，查代码/数据确认，再重训查看变化；Appendix B 从古今英语词响应的异常提出语用差异解释。这是“可视化为科研认识和诊断服务”的明确先例。
- **原文结果**：Yelp 二分类先用 20K 短评论，原验证/测试准确率 89.5%/88.6%；发现约 3:1 类别不均后 oversampling，报告 91.52%/91.91%。这是文中单任务案例，不是可视化相对常规审查的受控因果收益。
- **可借用**：先总览、点选再看分布；保留符号；同时看群体平均和当前上下文；用模型/句子差异产生猜想再检查数据。四位专家全部猜对哪一模型更好，但无盲测基线/多任务定量发现率，不能证明研究效率普遍提升。
- **分数边界**：`E[Δh|word]` 是语料分布下的状态更新统计，不是该输出 token 的因果贡献。§5.1 Eq.3 分解 softmax 分子，却把因子说成预测概率贡献，未包含分母；状态幅度叫“信息”也不等于信息论量。因上下文不同而均值变化应视作条件分布混合。
- **原文疑点**：§6.1.2 说字号正比于到中心的距离，同时说更靠中心字号更大，两句方向矛盾；Appendix A Eq.10 的 LSTM 输出省去通常的 tanh，Eq.12 把 GRU reset 乘输入且候选状态写成 t−1，与同页解释/通用维度不一致，原图已核。未据此断言 TensorFlow 实验实现同样错误。其重训 oversampling 与 split 的确切执行顺序未写清，不能借报道提升作为严格复现实证。
- **范围**：历史 RNN，主要分类/语言建模，没有 OA；已做可视诊断范式，可借设计，不直接继承机制结论。`done/reuse/open_issue`。

## visual_seq2seqvis · Seq2Seq-Vis: A Visual Debugging Tool for Sequence-to-Sequence Models

Strobelt et al.，[arXiv v2](https://arxiv.org/abs/1804.09299v2)，2018-10-16；全部 11 页，无独立附录。文件 `review/visual/pdfs/seq2seqvis.pdf`；§5 Fig.8–9、§7 Fig.11–12 已核原图 `review/visual/figures/seq2seqvis_root_p6.png`、`seq2seqvis_root_p8.png`。页面隐藏布局造成空白膨胀，已用 raw/空白规范化补读，没有以截断片段充作全文。

- **直接重叠**：五阶段（编码、解码、attention、下一词分布、beam search）联动；选输出位置看 attention、top-k 概率、训练近邻和轨迹；可改输入、attention 或生成前缀，再比较续写。目标是发现/外化错误、连接样本、试验替代决定。
- **认知案例**：§3 从遗漏 dark 的翻译错误逐步提出五个位置假设，强制该词进入前缀后得到正确翻译；§7 Fig.11 改 for→on 后，模型为语法一致性补出 on world leaders 并改变摘要长度，直接覆盖“早期输出选择影响后续生成”。Fig.13 循环轨迹与重复短语；这些通用现象不能作为本项目首次发现。
- **可借用**：把局部分布和实际解码路径一起展示；点击目标看条件历史，明确 prefix decode 与完全重新生成；保留探索过程并输出可复验案例。公开模型 hooks 和近邻库是工程前提，论文系统不是未经修改即可兼容当前 LLM。
- **证据边界**：邻域是 50K 训练句中最近状态，不是训练数据因果归因。§3 以“注意力高/近邻看似合理”排除编码和 attention 错误，不能按现代忠实性标准当成充分排除；强制 dark 得到好翻译证明局部可修复，尚不唯一证明 beam 宽度是根因。要确认应比较目标路径分数和真实增宽结果。
- **评估边界**：主要案例叙事，5500 页面访问/156 stars 是早期使用度，不是受控发现质量提升；没有证明对新模型/后门可推广。论文自称关联研究都是黑盒过宽，LRP/梯度类并非严格黑盒。我们承认工具先例，同时不继承这句概括。`done/reuse/direct_neighbor`。
# 补充：2017 年直接先例

## Ding et al. — Visualizing and Understanding Neural Machine Translation (ACL 2017)

- [正式论文](https://aclanthology.org/P17-1106/)，10 页全文含全部参考文献，无单独附录；`review/foundations/pdfs/Ding2017.pdf`。已核 p7–8 Figs.8–11。
- §3/Fig.2 明确把源词和已经生成的目标词同时作为上下文，使用 LRP 计算它们对内部状态与输出表示的 relevance。§4 逐层对比 attention、context、decoder state、目标表示的热图，分析漏译、重复、不相关词和否定反转。
- **直接重叠**：Fig.8 用“the”对重复“history”的影响解释目标上下文作用；Fig.10 显示源“不”虽获高 attention，先前“will talk”仍可能左右生成。因此“按输出位置比较输入与生成历史、沿层看依赖转移、用热图理解异常”至少在此已有具体先例。
- **可借用**：选择错误输出位置，比较不同来源和内部处理阶段；主动寻找 attention 与其他归因不一致的案例。
- **不能扩大**：LRP 按指定传播规则分配 hidden-state 数值，非干预识别的唯一原因；没有对上述诊断做独立消融或大样本发现成功率检验。§4.3.4 自称 possible reason，不能把案例解释升级成已证因果规律。未研究训练后门或 OA。
- **读图纠错**：§4.4 prose 残留“第9词forge/Ry9”文字，Fig.11 实际为第5词 democratic/Ry5；以图和本节任务一致内容理解。Eq.6 的 y_j 与文字 y_{j-1} 不一致；门乘法传播规则、非线性忽略与数值守恒不等于通用忠实性保证。

## Lee, Shin & Kim — Interactive Visualization and Manipulation of Attention-based Neural Machine Translation (EMNLP 2017 demo)

- [正式论文](https://aclanthology.org/D17-2021/)，6 页全文含参考文献，无独立附录；`review/foundations/pdfs/Lee2017.pdf`。已核 p4–5 Figs.8–10。
- §3 同时提供候选输出概率、beam search 树、输入—输出 attention 表/图、扩展已被剪掉的生成分支，以及拖动 attention 权重后实时看候选概率改变。
- **直接重叠**：输出词级交互、查看输入影响、干预后重生成、探索替代分支均已实现；Fig.9 以 tone 的“语调/音色”候选说明上下文影响差异。§3.4.2 还自动调整 attention 以提高指定词/整句概率。
- **可借用**：原始状态与干预状态分开显示、保留生成分支、显示候选分布而不仅是最终词。作者解释 attention 在选择输出前计算，因此可能混合多个候选的证据；这直接支持本项目不把 attention 色深等同特定已选词因果贡献。
- **不能扩大**：优化后更清楚的 attention 图代表修改后的计算，不是对原始计算的忠实恢复。论文为系统示例，未给受控用户研究或广泛准确诊断率；后续需要自己的发现与验证证据。
