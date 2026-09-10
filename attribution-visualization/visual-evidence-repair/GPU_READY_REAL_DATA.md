# toy48 真实图片数据准备

2026-09-10：正常真实图部分已实际下载、验证并冻结，`receipt.json` 状态为 `real_clean_inputs_ready`。本文件只说明这一部分；合成事实对、污染权重、触发图和 GPU 冒烟仍须单独验收。准备脚本为 [tools/prepare_coco.py](tools/prepare_coco.py)，实际数据放在 `runs/toy48-inputs/real`。

## 原始来源与固定选择

问题与十个人工答案来自 [VQAv2 官方下载页](https://visualqa.org/download.html) 链接的 COCO `val2014` 问题和标注；图片与描述来自 [COCO 官方数据](https://cocodataset.org/#download) 的同一 `val2014` 划分。VQAv2 发布年份为 2017，不是把 COCO val2017 混进来。

| 下载内容 | 原始地址 | 核验 HEAD 字节数 |
|---|---|---:|
| VQAv2 val 问题 | https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip | 3,494,929 |
| VQAv2 val 标注 | https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip | 10,518,930 |
| COCO2014 标注包 | https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip | 252,872,794 |

从最后一包只读 `annotations/captions_val2014.json`，不解压其它任务标注。图片逐张从同一官方 `images.cocodataset.org` S3 桶下载；未下载全量 val2014 图片 ZIP。原图片域名的 HTTPS 证书主机名不匹配，因此采用上述 S3 路径式 HTTPS 地址，保留证书检查。

在同时有 VQAv2 问答、COCO 描述和图像元数据的图片 ID 中，按 `SHA256("toy48-real-v1-seed42:" + image_id)` 排序，依次分配五个固定组。每张 VQA 图片选原始 `question_id` 最小的一题。选择不使用任何模型输出、答案正确率或攻击结果。

| 产物 | 图片簇 | 每图任务 |
|---|---:|---|
| `fit.jsonl` | 100 | 1 条原始 VQAv2 问答 |
| `calibration.jsonl` | 96 | 64 条 VQA、32 条 COCO 描述 |
| `test-clean.jsonl` | 360 | 300 条 VQA、60 条 COCO 描述 |

共 556 张不同图片；三个划分的图片 ID 互斥，最终还由现有 `repair.data.validate_no_leakage` 检查图片字节哈希不跨簇／划分复用。这里只是从官方验证集抽出的本项目 fit/calibration/test 子集，不称完整官方 benchmark 或预训练未见数据。

## 标签与输入合同

- VQA 的 `question`、`answer` 和全部十条 `references` 原样对应官方问题、`multiple_choice_answer` 和十个答案记录；保留重复答案供官方 soft score 使用。
- Caption 保留该图全部原始描述，第一条按标注 ID 排序的描述作为训练答案。`Describe this image.` 是冻结的描述任务指令，并非虚构的人工问题或参考答案。
- 正常单图的 `answers` 只含对应训练答案，避免把十个参考答案或五句描述当作推理时多选题。完整实际生成的评分仍使用全部人工参考；这些 singleton 不承担事实配对候选归因比较，不能从单候选最大值宣称得到有区分性的候选证据。
- 路径全部相对于数据目录，便于原样同步到服务器。每个 JSONL 都有同名 `.manifest.json`，包含图片 SHA256；fit/calibration 的 purpose 为 `repair`，test 为 `evaluation`，均标明 `clean`。
- `selected-source-records.json` 保存选中图片的原始图像元数据、问题、VQA 标注、COCO 描述与许可信息，以及三个原始 ZIP 的 SHA256。`receipt.json` 冻结数据文件、清单、全部图片身份与最终验证结果。

## 复用与检查

在仓库根目录执行一次：

```bash
.venv-attribution/bin/python attribution-visualization/visual-evidence-repair/tools/prepare_coco.py --output attribution-visualization/visual-evidence-repair/runs/toy48-inputs/real
```

脚本只使用标准库、已有 Pillow 和现有数据验证器；下载失败可在同目录重试，已完成的下载会复用；存在最终 receipt 时拒绝覆盖冻结产物。图片在下载时核对原始宽高并由 Pillow 验证，最终数据用现有验证器检查一次。

同步实验输入时只需 `images/`、三个 JSONL 与 sidecar、选中源记录及 receipt；`source-cache/` 是原始下载缓存，可以排除。该目录被现有 `runs/` 规则忽略，避免把 252MB 原标注 ZIP 推进 Git；脚本、来源说明和最终冻结摘要可以进 Git。

## 本次已冻结身份

一次实际准备结果：556 个 unit、556 个图片簇、556 份不同图片字节，划分为 fit/calibration/test；所有选中的 VQA 均有原始十条答案，caption 均有原始五条描述。已有数据验证器检查通过。

图片共 **92,117,090 字节**，排除 `source-cache/` 的全部可同步产物为 **94,200,243 字节**。三个原始标注 ZIP 共 266,886,653 字节，下载准备没有使用 GPU。

| 文件 | SHA256 |
|---|---|
| `receipt.json` | `bd85bd227b61fc0103ce5333c1c9e567fe58192b71ebdcebf08fd6ed10f76a91` |
| `selected-source-records.json` | `3ae113e44a45e085ec9804a34a4307eadc89492ca614810b4e744287e752f03f` |
| `fit.jsonl` | `1ca53c2f1c3e7517f871c5815a0253c1ab00ffc69fb42f0da63ef2ee74ba434e` |
| `calibration.jsonl` | `5918fc26544a3bb1bebc6d8b26dc981d76fd59a4c69391fc3ea5204611d9d361` |
| `test-clean.jsonl` | `82eda7c094f098fe7208fad8af994579d4b5b172b7d943ee930a1f77aaed28e1` |

各 sidecar、每张图片及三个源 ZIP 的完整身份保存在 receipt。若将这些文件合并到上一级目录的最终数据中，必须把图片路径同时加上 `real/` 前缀；不能仅拼接 JSONL 而忽略路径和 sidecar。
