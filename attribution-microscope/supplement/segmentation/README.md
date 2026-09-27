# COCO + SAM 3 分割流水线（v1）

实现 `research/coco-sam3-segmentation-2026-09-27/EXPERIMENT_DESIGN.md`。产物在
`runs/segmentation/coco-sam3-v1/`。合成数据端到端预演：`rehearsal_seg.py`（46/46）。

## 运行顺序

| 步 | 命令（仓库根目录，项目环境） | 谁 | 说明 |
|---|---|---|---|
| S1 | `python supplement/segmentation/s1_manifest.py` | 机器 | 按 image_id 去重；下载 val2014/train2014 原图并与 p_core 输入逐张核对身份（MAE ≤ 6、Pearson ≥ 0.99）；下载并关联 COCO 2017 panoptic（两个 split 都查、核对尺寸）；解码为 `coco/`；冻结开发/验证划分（试点 5 张 + SHA-256 排序前 15 张 = 开发，其后 30 张 = 验证）。重跑只核对，不重分 |
| S2a | `python supplement/segmentation/s2_prompts.py init-map` | 机器 | 类别→提示词草表 `prompt_map.json`（开发集上修改） |
| S2b | `python supplement/segmentation/s2_prompts.py extras-template --set dev` | 机器 | 每图一张空登记表 `extras/<id>.json` |
| **S2c** | 逐图看**原图**，在登记表里写 COCO 漏标的可辨认内容并签名（`registered_by`） | **人** | 不看热图、VLM 答案、投毒标签、SAM 输出 |
| S2d | `python supplement/segmentation/s2_prompts.py build --set dev` | 机器 | 未签名的表一律拒绝；已有 SAM 输出的图不许再改提示 |
| **S3** | `supplement/segmentation/launch_sam3.sh dev <gpu>` | GPU | SAM 3 环境（`/workspace/sam3-preview/env`）；先核对权重 SHA-256；可断点续跑；单批 30 分钟超时。看门狗：`check_seg.py` |
| S4 | `python supplement/segmentation/s4_routes.py A\|B\|candidates --set dev` | 机器 | 三路线中 A、B 与候选匹配表 |
| **S4c** | 逐图写 `edits/<id>.json`（retain/add/replace/reject/unresolved，格式见 `s4_routes.EDITS_FORMAT`），再 `s4_routes.py C --set dev` | **人** | 与其他物体冲突必须在 take_from 里点名；空出像素只回填到点名的背景 |
| S5 | `python supplement/segmentation/s5_viz.py --set dev` | 机器 | 四列图、编辑局部放大、`index.html` |
| — | 开发集修完词表与规则后 `s2_prompts.py freeze-map`，再对 `val` 重复 S2b–S5 | 人+机器 | 验证集不改词表/阈值 |
| **S6** | `s5_viz.py --set val --blind` → `s6_quality.py forms --set val` → 独立核验者填表 → `s6_quality.py score --set val` | **独立核验者** | 先只看原图列参考清单并签名，再看 X/Y/Z 盲列；核验者与该图编辑者相同则自动标为自查，不算独立验收 |

## 冻结设置（与试点一致）

`facebook/sam3` 权重 SHA-256 `9999e234…8c9e`（`seg_common.SAM3_SHA256`），代码修订 `2345a4ad`，
置信度 0.5、掩码 0.5、BF16、处理器默认预处理，掩码为原图尺寸。v1 只用文字提示。

## 不在本流水线内

热图关联、实例热度、`p_seen` / `p_instrument` 的标注、点/框提示（v2）。
