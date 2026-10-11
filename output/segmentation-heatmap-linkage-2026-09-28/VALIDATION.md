## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-28
- Verification Status: ANALYZED
- Version Label: regional_attribution_validation_v1
- Upstream Dependencies: coco-sam3-v1 route C; LLaVA phase2b curves; origin/main e2da81db21d14da69d728e3351d8c42fbe29af5e

## 已完成的分析

原 HTML 新增「分割 × 热点变化」页签。使用固定路线 C 分区，标准 28px 触发输入、攻击答案 T2，A/B 与正/负方向分别统计。接入 81 个模型或检查点的 12,974 张归因图。199 道题对应 188 张独立图片；图片 772 / 问题 #184 身份未解决，排除。轨迹 60 道题对应 59 张图片，均有分割。

新增内容：整批区域占比曲线（训练步数、剂量或 ASR 为横轴）、类别表、前后热图同尺度叠加边界、所有实例的占比/强度/面积/富集比、点击区域定位、自动文字总结、事后最大 ASR 跳变窗口。自选完整模型对可查看全部 199 题；路径汇总固定共同题，绝不随检查点换分母。

以下为 B/T2 正归因的图片等权均值（与旧曲线的题目中位数口径不同）：

| 比较 | ASR | 触发器份额 | 物体份额 | 背景类份额 | 触发器上升图片 |
|---|---|---|---|---|---|
| 种子 1：s79 → 最终模型 | 0 → 99.0% | 5.0 → 72.5% | 30.8 → 8.3% | 55.8 → 15.9% | 59/59 |
| 种子 2：s80 → s640 | 0 → 98.5% | 4.3 → 73.3% | 34.5 → 8.4% | 53.0 → 15.0% | 58/59 |

两条轨迹中“人”的类别平均占比下降最多，但这不等于每张图都由人转向触发器。单图中可出现份额下降而原始强度上升，因此文字必须同时检查强度。人、物体、背景的“与问题相关性”没有独立标注，页面不自动作此判定。

事后最大 ASR 跳变（B 正归因）：种子 1 s360→s380，ASR 72→99.5%，触发器份额 75.6→79.1%；种子 2 s260→s280，ASR 8.5→44.5%，触发器份额 62.1→67.1%。这也提示份额变化与 ASR 不一定同步，不作因果判断。

## 数值与界面验证

- 188 个分区图缩放后逐 patch 权重和为 1；触发器独占右下四个 patch。
- 12,974 张原始图的正/负区域强度守恒，最大绝对误差 1.14e-13。
- 原始 NPZ 重算旧 57 点 × A/B 共 114 个 T2 触发器中位数，误差为 0。
- 新统计的非负数值、长度、区域归属、每条路径固定题集合、区域/类别份额和已验证。支持强度为零的 50 个图/方向不产生虚假百分比。
- 补拍点源文件按追加顺序存储；新页面明确按训练步数排序，测试断言顺序严格递增。ASR 横轴仍按此实验顺序连线。
- 原 index/atlas/m1 JSON 和所有 95 个原始 raster 资源保持；旧 121,560 行 / 1,386 个组合的照片、问答、token 与配对参考完整性回归通过。
- 浏览器测试 5 路径 × 2 仪器 × 2 方向共 20 种组合；检查样本切换、点击区域、轮廓、ASR 横轴、第二条轨迹终点与单图解释。CLEAN→P-1.0 下有 199 个问题选项。
- 零值边界实测：CLEAN→P-0.4，B 负方向，问题 #51，终点强度 0，页面显示占比未定义，不编造转移。
- 两个较窄视口热图并排，原图位于其上；大屏三图并排。没有新增网络依赖，全部新数据与分割内嵌。

## 解释边界与 11 项偏差检查

| 检查 | 结果 / 处理 |
|---|---|
| Simpson's paradox | 保留单图、类别和背景/物体汇总；A/B、正负、种子/剂量分开，仍不能排除亚组逆转。 |
| Ecological fallacy | ASR 明确为 200 题模型指标，非当前图的成功概率；另列本题攻击结果。 |
| Selection / Berkson | 轨迹是已有选定的 60 题，不能外推到 VQAv2 全体；不按“成功”或热度挑样本。 |
| Collider bias | 不按攻击成功条件筛选；分割身份和数据齐备筛选仍有选择限制。 |
| Base-rate neglect | 报告 200→199 题、189→188 图片、路径有效题/图片数；图内标明 ASR 分母不同。 |
| Regression to mean | 不以 ASR 极值的后续回落作为机制证据；最大跳变仅作事后导航。 |
| Survivorship bias | 显示身份未解决样本和零方向强度排除数，未以 0 补齐缺失。 |
| Look-elsewhere effect | 最大类别变化和最大 ASR 跳变是事后描述，不报告显著性或确认性结论。 |
| Garden of forking paths | 本页明确统一口径，保留全部区域和两个方向；未做阈值调参或假设检验。 |
| Correlation / causation | 前后份额变化不是可识别的物理流量；没有 Sankey 或因果箭头。 |
| Reverse causality | 不声称归因变化导致 ASR，二者都可能随训练变化。 |

未独立复现实验训练、归因生成或 SAM3 预测；本次重算仅为 CPU 数值聚合与界面核验。分割有 109 个 SAM 来源区域、252 个未决候选记录；未决记录不等于未分配像素。分割尚未完成独立人工复核。24×24 patch 无法可靠解释亚 patch 细节，小于 4 patch 面积的区域显示提示。

## 重建

在仓库根目录执行：

```sh
OPENBLAS_NUM_THREADS=1 .venv-attribution/bin/python output/segmentation-heatmap-linkage-2026-09-28/build_data.py
.venv-attribution/bin/python output/llava-supplement-2026-09-28/merge.py
node output/segmentation-heatmap-linkage-2026-09-28/test.cjs
node output/llava-supplement-2026-09-28/test-ui.cjs
```

`data.json` 是页面数据；`checks.json` 包含源 NPZ SHA256 与守恒检查；`results.json` 是可复核的类别和区域均值。原始数组位于同目录 source，来自 Git archive，不覆盖旧实验文件。本次未提交或推送 Git。

最终浏览器回归：旧补充曲线 6 图可见，逐样本对比问答可见，全样本网格 200 格可见；新页签轮廓开关、上一张/下一张、区域选择和 ASR 横轴可操作。最终控制台未见错误。交付文件及 SHA256 见 delivery.json。
