# 主持人补充核验记录

日期：2026-09-26。核验对象仅为本次 PDF 和 ZIP；未修改用户桌面的原文件。

## 文件与边界

- PDF：`/Users/ruizhixu/Desktop/TARE_ICLR2027.pdf`，81 页，SHA-256 `7c946736c2d361ae7d7a1ad6d511b131affb2788f8a8155a68e49ea82f70f570`。
- ZIP：`/Users/ruizhixu/Desktop/TARE_ICLR2027_supplementary.zip`，SHA-256 `a828dd14dc22e4fd56a9b66c9fe6c8176e7fc62d3774c3414c0b6e1143094791`。
- 在检查路径与脚本后，于临时副本执行本地分析重建。没有执行训练、GPU 实验、模型下载或外部发布。
- 自带哈希只证明此次提交的内部一致性，不能独立证明训练来源、时间或预注册先后。

## 实际重建结果

1. Python 3.9.6 初次运行在 `supplement/analysis/build_chain_numbers.py:234` 因 f-string 中的反斜杠语法失败。README 只写 Python 3，未列最低版本。建议明确 Python >=3.12，并给出验证过的依赖版本。
2. 换用本机已有 Python 3.13.5 / Matplotlib 3.10.0，在全新临时副本再次执行 `supplement/reproduce_all.sh`，所有分析 builder 均完成；嵌套 scheduler 重建返回 OK。
3. 68 个数据文件的提交哈希检查通过；独立 SHA-256 比较确认重建前后数据全部不变。排行榜的四个重建文件与提交版本逐字节一致。
4. 比较 91 个生成文件（排除脚本声明不属于稿件输出的 `chain_fixes_*`）：所有非图片文件完全一致。差异仅有 7 个 PDF 和 2 个 PNG；尚未对每张差异图片执行视觉等价性审计，不能称所有图像字节一致。
5. 主脚本最终返回退出码 1：图片过滤使用 `grep -v '\.pdf$\|\.png$'`，在本机 macOS `/usr/bin/grep` 下漏排 PDF，将 PDF 差异误报为 non-figure drift。最小输入 `a.pdf / b.png / c.tex` 可复现；改用扩展正则 `grep -Ev '\.(pdf|png)$'` 能正确过滤。这里只给建议，没有修改提交脚本。
6. 所以准确结论是：**数值与非图片产物重建一致；整套脚本存在运行环境和末尾检查的可移植性问题。** 不是所有检查无条件通过，也不是科学数值复现失败。

证据：[独立哈希比较](rebuild_verification.json)、[Python 3.9 运行日志](tare_iclr2027_rebuild.log)、[Python 3.13 完整运行日志](tare_iclr2027_rebuild_py313.log)。

## 主持人对两项重要意见的复核

- **TARE-Z 指标口径错配得到确认。** `main.tex:275–277` 的 Eq.6 对应防御后精度的 paired 差，但 `main.tex:435–437` 使用 `tzBbPooledDid` 和以 DiD MAE 构造的区间宏。`build_tarez_bb.py:81–85` 分别计算 `e_did` 和 `e_par`；两者并不相同。FT 的数值分别约 0.1273 和 0.2564 个百分点。应统一公式与实证统计；不能只改标签并默认全部结论原封不动。
- **TSBD 干净模型对照确实已有先例。** [NeurIPS 2024 正式论文](https://papers.nips.cc/paper_files/paper/2024/file/4a3ffc1a0074901718ab9dcef41909e9-Paper-Conference.pdf) 第19页 Appendix J 和第20页 Fig.11 评估了 TSBD、ANP、RNP 对干净模型的影响。本稿已承认 clean-arm 本身非创新，因此这不是“整个贡献被抢先”的证据；但其针对 TSBD 的影响审计及近邻比较应纳入这一直接相关结果，并讲清配方匹配带来的新增内容。

## 评分标准来源

- 科学评价依据：[ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)：问题、动机与文献定位、主张支持及严谨性、新知识与社区价值，并检查清晰度和可复现性。新算法或 SOTA 不是必要条件。
- 综合判断依据：[ICLR 2027 Area Chair Guidelines](https://iclr.cc/Conferences/2027/AreaChairGuidelines)：综合评审理由，不把分数平均值当作裁决。
- 2027 公开指南没有可核实的数字锚点，当前无法取得对应 OpenReview 评审表。报告的数字仅借用[2026 年实际官方评审记录](https://openreview.net/forum?id=ICANwnoGgN&noteId=p0VR9cKoYg)中已核实的档位：4 略低于接收线，6 略高于接收线；不是声称已取得 ICLR 2027 官方数字量表。
- 四位是不同 sub-agent，各自先锁定定性意见，不读取其他人的报告；随后才映射至共同数字表达。这是作者侧 AI 模拟评审，不是会议正式审稿或录用概率估计。
