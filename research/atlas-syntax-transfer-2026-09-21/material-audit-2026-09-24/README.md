# 公开数据复用与 agent 制题先导 · 2026-09-24

这是一次实际执行的材料准备先导，不是后门模型实验，也没有验证三条机制假设。

先读 `REPORT.html`：数据覆盖、盲答结果、逐图证据在同一页。文字摘要见 `REPORT.md`。

## 文件怎么读

1. `PILOT_PROTOCOL.md`：答题前登记的选择规则、角色、指标和边界。
2. `*_source.json`：官方下载地址、原文件大小和 SHA256。数据集说明页：[VQAv2](https://visualqa.org/download.html)、[Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)、[TDIUC](https://kushalkafle.com/projects/tdiuc.html)。
3. `*_coverage.json`、`*_candidates.json`：完整题库扫描结果与保留的原题；候选尚未通过看图、训练重叠和 Trigger 区域核验。
4. `pilot/reference_sealed.json`、`calibration_selection_sealed.json`：VG 原始参考和 10/10 人工同答的 VQA 校准题。它们没有发给出题和答题 agent。
5. `pilot/writer_input.json`、`writer_output.json`：48 条原题及其 48 条改写、出题者发现的歧义。
6. `pilot/answer_input.json`、`freeze.json`：启动答题前封存的 104 问与文件哈希。
7. `pilot/answer_a.json`、`answer_b.json`：两个新会话的独立回答。这里 a/b 是答题员编号，与归因仪器 A/B 无关。
8. `pilot/reviewer_blind.json`、`reviewer_final.json`：终审先独立看图，再看答题和参考记录。模型终审不是人工金标准。
9. `pilot/scored_questions.json`、`scores.json`、`scored_groups.json`：固定归一化下的原始逐题计分，不随终审修改标签。
10. `audit_checks.json`：封存文件未变、原题原标签对账、校准题十人同答、跨库已知图片 ID 去重及答题完整性复核。

## 扫描口径

每张图寻找两个不同对象指代，每个指代有颜色和数量题；颜色不同，数量不同且在 1–6。四格各有两条不同现成问句才计为现成八题候选。颜色词表为 11 个基本色，灰色统一 grey/gray。

三个脚本采用数据集各自的保守文本规则，词表及无效答案处理并不完全相同，不能用候选多少给数据集质量排名。VQA/VG 复合名词单复数归一化等启发式可能漏配；没有命中不代表语义上不存在。VQA 的 0/1/2+ 完整对象分布只覆盖 18,289 张至少有一条通过题型和答案过滤的图片，不是全部 40,504 张。

合并只按官方 VG→COCO 图片 ID 映射去重；未做像素近重复检查，也未跨库合并问句寻找更多八题组。28 张候选属于 COCO val2014，仅是 split 信息；本地未取得 Atlas 完整训练/p_core/p_instrument 图片清单，不能声称通过实验排重。

## 先导抽样及图片

12 张 VG 图片：在存在不带位置限定的候选对中，排除已披露答案的 image_id=4，按 SHA256(`atlas-pilot-v1:` + image_id) 排序；每图取按名称排序的第一对，每格取最小 qa_id。没有按照 agent 答案换图，也没有先按视觉清晰度筛选。这些图片包含食物和整体—部件组合，不能当作拟议 41 类正式材料池的代表样本。

4 张校准图：从完整 VQAv2 validation 中寻找同一对象的颜色和数量题，两题均 10/10 原标注者同答；在 69 个候选对象中按 SHA256(`atlas-calibration-v1:` + image_id + object) 排序，选前四张不同图片。校准原题未改写，也不纳入四格材料分母。

出题、答题均看原图。终审另看 336×336 bicubic 预览，检查缩放后的可辨性；尚未叠加真实 Trigger mask。VG 图来自候选 JSON 中的官方图片 URL。COCO 图片因 HTTPS 证书域名不匹配，使用同一官方 images.cocodataset.org 的 HTTP 地址下载，没有关闭证书校验；封存记录保留了标准 HTTPS 定位地址。

## 重算与复核

Python 3；图片准备还需 Pillow。官方原压缩包缓存于 `/tmp/atlas-material-audit-2026-09-24`，没有复制进仓库。下列命令从本目录运行：

```sh
python3 vqa_coverage.py
python3 vg_coverage.py
python3 tdiuc_coverage.py /tmp/atlas-material-audit-2026-09-24/TDIUC.zip
python3 score_pilot.py
python3 verify_artifacts.py
python3 build_report.py
```

VG/TDIUC 脚本的输入与下载细节见脚本和 `*_source.json`。`prepare_pilot.py` 用于重建 VG 出题输入，不应在保存本次冻结材料的目录直接重复运行。agent 问题和答案不是确定性程序输出；重算分数可以精确重现，重新调用 agent 不保证得到相同回答。

本次新产生的全部文件限于此目录；未改动正式实验合同，未训练或推理后门模型，未执行 Git 提交或推送。精确模型调用费用不可得，也没有同题人工计时基线。
