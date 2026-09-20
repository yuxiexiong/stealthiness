# 课题说明页

九个课题，每个一页 HTML，自包含——双击就开，不需要服务器、不需要联网。
每页把支撑这个课题的归因热图按论证顺序排好，配标注；每一格都保留了
Attribution Atlas 里的全部信息（点开看：真实值域、问题与正确答案、
该样本在 trigger 下答案有没有被翻、该臂的 ASR 与干净准确率）。

| 文件 | 课题 | 状态 |
|---|---|---|
| `R6-sign-flip.html` | 低剂量符号翻转 | 已查重 · A− · 证据档：推测 |
| `R1-capture-law.html` | 捕获律 | 已查重 · B+ |
| `R5-triggerless-detection.html` | 免 trigger 检测 | 已查重 · B+（攻击面） |
| `R3-decomposition.html` | 成分分解 / contingency | 已查重 · B（曲线）/ D（零结果） |
| `R2-two-axis-coupling.html` | 双轴耦合 | 已查重 · B− · 欠配重 |
| `R7-cross-modal.html` | 跨模态复现 | 已查重 · C+ · 不足以独立成文 |
| `N1-failed-attacks-leave-marks.html` | 失败的攻击也留印记 | **未查重，不给等级** |
| `N2-label-axis-not-trigger-axis.html` | 印记来自标签不来自 trigger | **未查重，不给等级** |
| `N3-subthreshold-sign-reversal.html` | 亚阈值臂方向相反 | **未查重，不给等级** |

前六个的评级来自 NOVELTY_REPORT.md 的四轮查重（约 240 条编号查询、
约 125 篇精读）。后三个是从实验数据里读出来的，尚未与文献对照，
按准则 5.1 在查重完成前不给等级、不给标签。

## 怎么重新生成

```bash
cd topics && python3 t_r1.py      # 每个课题一个脚本
```

`extract.py` 从 `../runs/maps` 派生的打包数据里，只抽出这一页论证要用的那几格
（几十格，不是全部 129,120 格），所以每页只有 0.07–0.32 MB。
`template.html` 是九页共用的渲染管线，与 `src/pair_sheet.py` 同序：
先放大、做 6px 高斯模糊、再上色。仪器 A 用红蓝发散色标，蓝色是压低目标 logit。
