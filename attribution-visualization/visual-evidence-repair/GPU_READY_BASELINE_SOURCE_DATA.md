# 固定 LoRA 起点的公开构建材料

2026-09-10：源材料已实下载并冻结，`runs/toy48-baseline-data/source-receipt.json` 状态为 `unmodified_construction_material_ready`。这里只准备**未改动的原图、原问题和原答案**，不生成触发图、不改训练目标、不训练模型。

这是我们按公开方法准备的固定数据实例，**不是原论文作者公布的训练集，也不是原论文结果复现**。后续构建、资格验收和 GPU 时间仍单独记账；这一份源数据 ready 不等于污染模型或完整 toy 已经 ready。

## 已确认的规模

| 原材料 | 图数 | 每图曝光 | 指令数 |
|---|---:|---|---:|
| `normal.jsonl` | 2,000 | 5 个不同原 VQA 问题＋5 条不同原描述 | 20,000 |
| `construction-candidates.jsonl` | 200 | 5 个不同原 VQA 问题 | 1,000 |

正常混合固定为 10,000 VQA＋10,000 caption。同一图片在正常组出现十次；同一描述指令配五个人工参考答案。这种重复曝光、任务配比和图片数是本实例的设计选择，不把 20k 条指令写成 20k 张独立图片。候选组只是原材料，后续是否及如何改变图像／目标必须在独立构建清单记录。

所有图、问题和答案来自 COCO **train2014** 及对应的 [VQAv2 官方训练数据](https://visualqa.org/download.html)。来源包括：

- [VQAv2 train 问题 ZIP](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip)：7,239,401 字节。
- [VQAv2 train 答案 ZIP](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip)：21,708,861 字节。
- [COCO2014 官方标注 ZIP](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip)：直接复用已下载的同一个文件，读取 `annotations/captions_train2014.json`，不重复下载。
- 图片只下载选中的 `COCO_train2014_<image_id>.jpg`，使用同一官方 COCO S3 桶的 HTTPS 路径，不取全量图片包。

原 VQAv2 train 有 443,757 题、82,783 图；按每图至少五个不同问题及五条不同描述、且不属于 toy 已冻结 val 图过滤，实有 **29,526** 个候选图。不同文本按大小写折叠和空白归一判重，不复制同一句来凑数量。

候选图按 `SHA256("toy48-baseline-v1-seed42:" + image_id)` 排序，前 2,000 张为正常组，后 200 张为构建候选组。每图按原 question/caption annotation ID 取前五个不同条目；没有使用模型输出或实验成绩选择数据。

## 交给构建者的接口

每行是 `{id, image_id, image, question, answer, task, references, source_kind, source_id}`。`image` 相对本数据目录；VQA 的原题、众数训练答案和十个人工参考完整保留。Caption 的答案为对应原标注文本，指令固定为 `Describe this image.`。这里不混用 `repair` 的 clean manifest 来描述构建集。

`selected-source-records.json` 保留所选原图元数据、原问题和标注、来源与许可；`source-receipt.json` 记录文件及图片 SHA256、三个官方源身份，以及被排除的 toy val receipt 身份。正常组和构建候选组图片 ID 互斥；图片下载时验证格式、原宽高及与 toy 图的字节不重叠，最终拒绝选中图片之间的字节重复。

最终构建清单应引用这份源 receipt 的 SHA256，再追加基座身份、实际目标、实际修改图的 hash 和构建配置，不改写源数据来冒称原标签。

## 本地准备命令

```bash
.venv-attribution/bin/python attribution-visualization/visual-evidence-repair/tools/prepare_baseline_data.py --output attribution-visualization/visual-evidence-repair/runs/toy48-baseline-data --toy-real attribution-visualization/visual-evidence-repair/runs/toy48-inputs/real
```

脚本复用 `prepare_coco.download`、标准库和已有 Pillow，最多十六并发下载。完成后拒绝覆盖冻结产物；同步训练材料可以排除 `source-cache/`。所有工作均为本地 CPU／网络，不消耗服务器 GPU。

## 本次实际交付

2,200 张图的原始字节均不同，合计 **345,820,003 字节**；正常组／候选组图片互斥，与已冻结 toy val 图片的 ID 和字节重叠均为零。排除 `source-cache/` 后，全部可同步原材料为 **379,805,972 字节**。所有计数与源标签一致性检查在本次准备中通过；没有 GPU 运行。

| 冻结文件 | SHA256 |
|---|---|
| `source-receipt.json` | `f13a209857571a1284d025f9162a8f3d05d90964a9389e9ec43edb1a8011b6c0` |
| `normal.jsonl` | `67b8ce30810ecf6efc6dda8e8e5cb4c5130d98311dcfa8551af02991db290d36` |
| `construction-candidates.jsonl` | `8f595c895ba67893f01fa0589e95b8a038ce5b281758bf3f562f82ceb97b2baa` |
| `selected-source-records.json` | `9dd5163eef749d119347801a97dbbaca6482f5a7ca2c4b5e1b6bd727e6c32e0a` |

已把源 receipt 交给独立构建步骤；它仍须记录真实基座、canonical 图像、固定标记、实际目标与训练配置，不能用这里的源数据验收替代构建验收。
