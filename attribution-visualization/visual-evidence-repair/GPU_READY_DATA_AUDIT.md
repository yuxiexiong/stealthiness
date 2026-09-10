# toy48 事实配对数据：来源与就绪记录

2026-09-10。本文只处理数据，不把数据下载或 CPU 校验称为 GPU 冒烟或实验结果。正常 VQA/COCO 子集由 `tools/prepare_coco.py` 独立准备；本文件对应 `prepare_toy48_data.py` 的 EditCLEVR 部分。

**实际状态：已完成。** 252 对、504 张图片和 756 个问题单元已落盘；现有 `repair.data` 对四份 JSONL 及哈希 manifest 的一次整体校验通过。全部 504 份场景 JSON 都实际包含必需的相机、光照、关系与几何字段，并通过严格配对检查。该部分目录总大小 85,798,953 字节，约 82 MiB；使用 GPU 为 0。冻结来源、所有 pair ID 与验收哈希见 [EDITCLEVR_DATA_LOCK.json](EDITCLEVR_DATA_LOCK.json)。

## 直接复用的资源

发现并实际使用 [EditCLEVR 官方数据](https://huggingface.co/datasets/torux/EditCLEVR/tree/734efa9b0164ba742cd2c39b9c2945c494497c23)：固定 revision `734efa9b0164ba742cd2c39b9c2945c494497c23`，CC BY 4.0。其 [官方代码](https://github.com/torux-bughunter/EditCLEVR)提供同一场景的前后渲染、对象身份、实例掩码与属性编辑记录。它是 ICML 2026 **workshop** 数据工作，不写成 ICML 主会论文。

这补上了此前资源审计的一个真实缺口：不必为本轮从零安装 Blender、生成场景、再自建合法编辑。没有把两个随机 CLEVR 图片配成一对，也没有只改场景 JSON 而不改图片。

元数据压缩包 9,711,965 字节，SHA256 `170a29a91bfe24515916c5da744c2dfb61b882637232090ff4e8d1113b02eb5f`。图像按官方压缩包顺序流式读取，只提取选定文件并提前关闭连接；因此只登记实际提取文件的哈希，不冒称核验了未下载的整个图像压缩包。

## 冻结的操作化

每对只改变一个对象的颜色，其余对象属性、几何、相机、光照、关系与实例掩码保持。目标和另一个应保持对象都必须能用不含颜色的“大小、材质、形状”唯一指代，可见性字段至少 0.75。

| 本轮划分 | 独立基础场景 | 官方来源 |
|---|---:|---|
| dev | 24 | train 的独立场景 |
| fit | 100 | train 的其他独立场景 |
| calibration | 32 | train 的其他独立场景 |
| test/simple | 32 | test_id |
| test/complex | 32 | test_hard |
| test/unseen_combination | 32 | test_cogent |

共 252 对、504 张图片；每场景 3 个问题，共 756 个 pair unit。按基础场景 seed 及图像字节核查互斥，不把 3 个问题当作 3 张独立图片。选取符合事前规则的文件名前缀，以减少整包下载；这是冻结的便利子集，不宣称随机代表整个 EditCLEVR。选择没有读取模型输出。

三个问题分别为：编辑对象的颜色（应改变）、另一个对象的颜色（应保持）、新颜色对象的数量（派生真值应改变）。颜色候选是全部 8 色，计数候选为 0–6；原场景只有 3–6 个对象。不能在问题里先说“rubber cube”再问它的材质或形状，那会把答案直接写进问题。

`complex` 的操作化是官方 hard-distractor 场景及多事实问答，**不等于关系移动或多属性干预**。所有 ID 训练、开发、校准场景逐对象符合 CoGenT-A；OOD 逐对象符合 CoGenT-B，且编辑目标只取 cube/cylinder，确保目标形状–颜色组合未在 A 中出现。球体在官方 A/B 中都可以取全部颜色，因此不宣称 OOD 场景每一个对象都具有未见组合。

## 真值与生成证据

复用 [CLEVR 官方 question_engine.py](https://github.com/facebookresearch/clevr-dataset-gen/blob/main/question_generation/question_engine.py)，导入前校验已核读文件的 SHA256 `b4b4d4e38d5c57470271b095cb5087af41b234efc8fb6103d898724b9081b4c2`。为每题保存程序，两端各自执行，明确关闭跨场景输出缓存，逐项验证改变／保持关系。

从实际下载的两份场景 JSON 验证只有声明的 color 不同，对象数、属性、位置、旋转、像素坐标、方向、关系、相机及光照记录一致；对前后实例 mask 数组要求完全相同；核验图片可解码、分辨率和字节非同一。固定发布数据中的旋转值两端采用同一记录单位，直接比较。当前 GitHub 编辑器重新导出旋转值的格式与该发布资产不同，因此不根据未运行的新版源码替换发布数据的真实值。

程序和实例掩码是数据生成与真值证据；实例 mask 不是模型算出的归因热图。正常模型能否回答这些问题、污染状态是否损伤它们、归因是否能指导修复，仍须 GPU 实验判定。

## 文件和复现

本地目录：`runs/toy48-data/`（大图和原始下载均由已有 `runs/` Git 忽略规则隔离）。

- `selected-scenes.json`：每个选定场景的完整来源和 split/stratum。
- `question-programs.json`：题目程序和两端真值。
- `dev-facts.jsonl`、`fit-facts.jsonl`、`calibration-facts.jsonl`、`test-facts.jsonl`：训练接口及严格配套的图片哈希 manifest。
- `clevr-ready.json`：只有全部真实图像和校验完成才写 `data_ready`。
- `strict-validation.json`：按最终脚本的必需字段规则完成的全量验收及现有 loader 校验结果。

复现命令（现有 Python/Pillow/NumPy 环境，CPU，零 GPUh）：

```bash
.venv-attribution/bin/python attribution-visualization/visual-evidence-repair/prepare_toy48_data.py \
  --output attribution-visualization/visual-evidence-repair/runs/toy48-data
```

软件核验：`tests/test_prepare_toy48_data.py` 的一个 CPU 测试已通过，覆盖第二语义编辑、几何变化、错误编辑元数据、歧义指代和全场景 OOD 规则的拒绝。实际下载完成状态以 `clevr-ready.json` 和严格 loader 的一次整体校验为准，不以该软件测试代替真实资产验收。

准备脚本拒绝覆盖已有 `clevr-ready.json` 的目录；若需要改变子集，应显式使用新目录并另建版本。相机或几何字段缺失会失败，不允许以两端都缺字段视为相等。对应拒绝路径已包含在同一个 CPU 测试中。
