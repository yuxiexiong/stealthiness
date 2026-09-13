# 诊断理论扩展：三方复核与修订记录

日期：2026-09-12。对象：[THEORY_DIAGNOSIS_EXTENSION.md](../THEORY_DIAGNOSIS_EXTENSION.md)。本轮没有写实验代码、运行模型或使用 GPU。

三个代理先独立研究可识别性、干预可执行性和直接相关文献，再各自检查同一份 D 稿。以下是修订后的处理记录，不表示模型实验已经验证方法，也不继承任何旧版理论的审查结论。

| 审查者与侧重 | 实质问题 | 正文处理 |
|---|---|---|
| ready_contract：数学与边界 | 第一次偏离正确参照可能只是措辞分歧，不能称为语义错误首次发生 | D1 改为参照分支首分歧，保留真实完整回答判错 |
| ready_contract：完备性 | 部分查询有时也足以证明完备；全集独自成功与“已排除所有真子集”不同 | 全表写成充分保障；区分只做正对照和证明当前粒度下联合必要性 |
| ready_contract：识别范围 | 查询不可识别反例不能覆盖额外白盒程序／权重分析 | 限于仅依赖有限运行记录的判断 |
| key_transcribe_a：生成和接口 | 深层 prefill 替换不一定恢复旧层 KV；D1 整串结论受停止规则影响 | 首选融合入口替换、重建 cache；深层需独立合同；补齐生成长度与停止条件 |
| key_transcribe_a：控制变量 | 保护题的 token 长度不同，不能无条件共用一个扰动 | 每个图片×问题冻结自己的扰动、供体、mask，保留合法事实对内共享要求 |
| key_transcribe_a：事实解释 | 整幅视觉 token 交换可能只是换图；mask 文件不等于对象级标注 | 整图／全部差异覆盖操作单列粗粒度对照，区域对应需核验 |
| check_truth_resources：公式用途 | 成功集合不等于网络传播路径；D3b 不创造比完整表更多的信息 | 统一使用恢复组合／方案；其用途是简洁查询允许范围和标记共同必选组 |
| check_truth_resources：同行与贡献 | 最小解释、非单调最小对比集合与组合枚举已有直接前例 | 引用原文定义和定理，作为可复用底座；不宣称这些集合结论原创 |

数学复核未发现 D3 两个集合结论的错误。它们不要求单调性，但“没有成功真子集”必须排除全部相关真子集，不能只做单元素删除。空成功族、并列、未决输出和不完整观测均有明确处理。

保留的公式只有四组：D1 固定可视化的输出竞争坐标；D2 定义可重放的状态替换与外部判定；D3 整理最小成功组合及有边界的范围查询；D4 定义带上下文、带方向的图上数值。没有新增高偏离必有益、未知扰动覆盖或参数修复成功的假设性保证。

定向文献核对覆盖以下原文的相关定义、证明和实验章节：

- [Hase 等，Does Localization Inform Editing?，§3–6](https://papers.nips.cc/paper_files/paper/2023/file/3927bbdcf0e8d1fa8aa23c26f358a281-Paper-Conference.pdf)：激活定位与权重编辑不能直接等同。
- [Zhang 与 Nanda，Towards Best Practices of Activation Patching，§2–6](https://arxiv.org/pdf/2309.16042v2)：供体、评分和联合替换方式会影响定位。
- [Canizales 等，Concept-Based Abductive and Contrastive Explanations，§2–4、附录 A.1](https://arxiv.org/html/2605.06640v1)：量词、包含关系最小性、单调简化，以及最小对比集合不依赖单调性的等价关系。
- [Sundararajan 等，The Shapley Taylor Interaction Index，§2–3](https://proceedings.mlr.press/v119/sundararajan20a/sundararajan20a.pdf)：交互拆分的含义与低阶分配边界；本稿不为图上守恒额外引入该分解。

当前结论：**有限恢复干预的定位逻辑成立；有用的局部恢复组合是否存在、结果能否复现到独立案例、是否优于简单定位，仍须模型证据。** 这次审查没有证明旧归因分数有效，也没有证明自然病因唯一或参数修复成功。
