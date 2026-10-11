# 五张实际热力图样本的 COCO 官方分割预览

2026-09-27。打开 `index.html` 逐张放大，`five_samples.jpg` 为总览。
列顺序：原图、COCO 2014 实例标注、COCO 2017 全景标注。未运行 SAM。

选择 p_core 前五个样本（idx 0–4），没有按标注质量挑图。COCO image_id
依次为 137496、134856、481628、1244、1369。通过官方 VQA 问答匹配，
再将下载原图直接缩放到 336×336，与实际 Atlas 的 clean 图集逐张比对。
五张 RGB 相关系数为 0.9936–0.9980；差异与图集的额外 JPEG 压缩相符，
并已逐张视觉确认。这里原比例图用于检查标注边界，不冒充原始模型输入文件。

## 实际观察

| 样本 | 实例标注 | 全景区域 | 实例像素覆盖 | 全景像素覆盖 | 主要发现 |
|---|---:|---:|---:|---:|---|
| 000 飞机场 | 7 | 13 | 17.44% | 95.41% | 主要飞机和人可用，文字、牌面、部件没有独立细分 |
| 001 海滩骑马 | 6 | 10 | 0.25% | 99.97% | 3 人 3 马分开，但每个对象仅约 0.07–0.40 个 patch 面积 |
| 002 天鹅 | 7 | 8 | 2.63% | 99.87% | 可见 8 只，只有 7 个 bird；漏掉的一只被全景归为水面 |
| 003 港口 | 9 | 15 | 3.10% | 92.44% | 中央 MSC 大船没有独立 boat，前景人、伞等可用 |
| 004 冲浪者 | 2 | 5 | 5.14% | 99.96% | 人和板分开，但头部边界粗糙；全景消除了原实例重叠 |

覆盖率是掩码并集像素数 / 图片像素数，不是准确率或对象召回率。
patch 面积单位为对象面积比例 × 576，不是实际接触到的 patch 数量。
原实例 mask 可能重叠，不能直接相加分配热点；全景使用互斥区域。
这些是五张图的定性核查，不能用来估计整个数据集的误差率。

结论：可以复用官方标注建立初始对象—区域关联，但正式比较热点前需要
核验重要对象、补漏标并修正影响分析的边界。高覆盖率不保证正确；尤其
漏标对象被归为水面等区域时，可能制造虚假的“热点转向背景”现象。
未运行 SAM 3，因此不据此声称 SAM 3 的标注一定更好。

## 数据与核验

- 官方实例源：<https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip>
- 官方全景源：<https://s3.amazonaws.com/images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip>
- 原图来自 COCO 官方 val2014，对应下载 URL 见 `selection.json`。
- `instances_subset.json`、`panoptic_subset.json` 保留五张图的官方元数据。
- `panoptic_raw_*.png` 保留官方原始区域 ID；`*_masks.npz` 为解码后的布尔掩码。
- 实例用 pycocotools 2.0.11 解码；全景按 RGB 区域 ID 解码，每段面积与官方 JSON 逐一相等。
- 5 张图全景匹配成功：001 来自 val2017，其余来自 train2017；始终按 image_id 匹配。
- `metrics.json` 记录所有区域、类别、官方 ID、面积及覆盖率。
- `source_match_check.jpg` 展示下载原图与实际 Atlas 原图的配对验证。
- 生成：`.venv-attribution/bin/python output/coco-segmentation-preview-2026-09-27/render.py`。
  原始大压缩包与临时解码依赖缓存于 `/private/tmp/coco-preview-20260927`。

可视化只对官方掩码着色、描边、编号，没有修改区域。编号为每张图内的
展示编号；实例和全景两列编号、颜色不要求一致，官方 ID 可查元数据。
