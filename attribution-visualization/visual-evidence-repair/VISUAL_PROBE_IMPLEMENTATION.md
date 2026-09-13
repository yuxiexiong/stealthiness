# V3 可视化探针：实现与运行说明

2026-09-13。对应[最终实验计划](DIAGNOSIS_PREDICTION_EXPERIMENT_PLAN.md)及[测量依据](THEORY_VISUAL_PROBE.md)。本轮完成代码和本地验证；尚未运行B0的新细网格模型测量，也未连接服务器。

随后只读检查服务器并完成启动前估时，见[ETA及服务器快照](VISUAL_PROBE_ETA_2026-09-13.md)；未启动模型实验。Git交付包含本轮代码、计划、必要的旧A/B原始JSON记录和约6 MB输入包，不含模型权重、重复部署压缩包及旧HTML嵌图副本。

## 复用了什么

- Hugging Face LLaVA加载、聊天模板、完整候选token化：原`repair/model.py`；没有另搭模型库。
- A/B的融合入口捕获、一次状态复制、独立生成缓存和计时：`repair/diagnosis_model.py`、`repair/diagnosis.py`。旧四区接口保持兼容。
- 原BadVision仓库所带VQAEval规范化与事实评分：`repair/report.py`；未重写正确性标尺。
- 原EditCLEVR图片、实例mask、事实标签和A/B实际记录。对象位置框复用原CLIP裁剪变换，没有新增检测器或分割模型。
- 离线页面样式、Python标准库及原生SVG/JavaScript，没有新增Python或前端依赖。

从新Git检出运行时，沿用仓库的`external/`外部依赖约定。原VQA评分文件来自BadVision提交`225d69a3086aadb5504efe31a2a28dce275bde97`；该外部仓库与模型权重不并入本次Git提交。现有服务器部署已有此文件。新机器可在本仓库根目录执行以下命令取得同版源码；已有目录先核对版本，不覆盖：

```bash
git clone https://github.com/6zHAOyi/BadVision.git external/badvision
git -C external/badvision checkout 225d69a3086aadb5504efe31a2a28dce275bde97
```

本轮需要的是其中`MiniGPT-4/minigpt4/common/vqa_tools/vqa_eval.py`，无需安装或启动MiniGPT-4；保留上游版权和许可证。模型加载仍用现有B0检查点和资产清单。

## 对应到计划的功能

| 计划要求 | 实现位置／行为 |
|---|---|
| 固定8例、16题，兼容减为6例 | `visual_probe prepare`，按计划的案例及端点顺序选取；同图Q2优先未变对象颜色，否则原记录中的`count`题 |
| 6×6主图＋2×2粗图，两种背景 | `visual_probe_protocol.main_maps`，按576个状态坐标建立精确集合；左上背景的9个细格标为已包含，不重复运行 |
| 完整事实候选与拒答读数 | 保留真实正确／拒答token，错误候选沿用原模板；`Diagnosis.log_probs`包含唯一末尾EOS，不做长度平均 |
| 每格关联实际回答、缺失不填零 | `Measurements`保存原始概率、生成、真值判定、停止原因、来源和计时；评分缓存不进入自由生成；完全相同的旧四区／换供体操作复用旧回答、补算新分数 |
| 粗细及联合关系 | 同一`additive_effect`用于数值排序、粗格相加预测与联合残差，不能相加时不补零 |
| 读图后登记新检查 | `freeze`绑定已有报告、代码身份、诊断文字、候选集合、预测和反驳条件，已测操作不能算新预测 |
| 关键／粗强细弱／等面积组合 | 校验共同背景及实际新增状态数，累计最多2＋2＋4组／例；不自动追逐最红格 |
| 局部12×12 | 累计最多4个父格／背景、每例最多2个；每父格4子格全部测量；可登记逐子格预测 |
| 双向供体鉴别 | 只接受已有合法配对节点，必须提供反方向操作，累计最多2例／16问题级条件 |
| 下一检查价值 | 冻结图建议、同完整数值相加、直接检查的顺序与停止证据，区分同菜单成本和完整账本 |
| 可视化实际参与诊断 | 交互原图叠图、事实位置框、事实／拒答切换、原始分数与前后回答、联合操作叠图及诊断卡 |

代码中的`known`是“该操作的答案已经知道”，不是“这是一条正确诊断”。判断仍须由实际检查支持或推翻。单格评分、真实回答和具体预言命中分别报告。

## 已准备好的真实输入

[输入包](runs/visual-probe-v3-inputs-2026-09-13/manifest.json)含计划的8个案例、16个主问题、32张去重图片及合法供体候选，约6.0 MB。对象位置来自核验过的原实例mask。这里保存的回答均为旧A/B记录，尚无新6×6结果。

准备命令已在本地执行成功；后续同步输入包整体目录即可，不必重建原始数据集。输入包保留模型身份及旧结果来源，文件改变会被拒绝。

```bash
cd /Users/ruizhixu/Documents/ChatGPT/stealthiness/attribution-visualization/visual-evidence-repair
../../.venv-attribution/bin/python -m repair.visual_probe prepare \
  --pools runs/diagnosis-a-prepared-2026-09-13/cases.jsonl runs/diagnosis-b-prepared-2026-09-13/cases.jsonl \
  --records runs/diagnosis-a-server-2026-09-13/results/attempt-3/run/full-0/records.jsonl \
    runs/diagnosis-a-server-2026-09-13/results/attempt-3/run/full-1/records.jsonl \
    runs/diagnosis-b-server-2026-09-13/results/run/evaluate-0/records.jsonl \
    runs/diagnosis-b-server-2026-09-13/results/run/evaluate-1/records.jsonl \
  --output runs/visual-probe-v3-inputs-new
```

`--count 6`只去掉计划末两例。可用`--notes`传入以案例ID为键、含`judgment`和`next_check`的读图前笔记；不提供时仅使用旧粗区结果的初始描述，不自动编造新假设。输出目录必须是新的，避免覆盖。

## 执行顺序

以下模型命令尚未执行。须在之后获准运行、队友任务结束并分配可用GPU后使用；继承原[A服务器环境](SERVER_EXECUTION_2026-09-13.md)中的私有cuBLAS设置。以下`python`表示服务器已有实验环境，工作目录为本项目。

### 1. 技术冒烟

```bash
python -m repair.visual_probe map \
  --manifest runs/visual-probe-v3-inputs-2026-09-13/manifest.json \
  --config configs/diagnosis-a.example.json \
  --smoke --output runs/visual-probe-v3-smoke
```

使用固定的普通例100161及反例100669，单卡完整走两题、两个背景与粗细图。冒烟报告标记为smoke，不能进入后续诊断冻结，也不能充当全量分母。根据实际捕获、计分、生成与总用时再估GPU ETA；本地CPU软件检查不代替这一步。

### 2. 全部探索图，可用两张已分配的卡

两条命令分别在两个终端执行；没有抢占、自动选卡或计时强杀逻辑。

```bash
CUDA_VISIBLE_DEVICES=0 python -m repair.visual_probe map \
  --manifest runs/visual-probe-v3-inputs-2026-09-13/manifest.json \
  --config configs/diagnosis-a.example.json --lanes 2 --lane-index 0 \
  --output runs/visual-probe-v3-map-0
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m repair.visual_probe map \
  --manifest runs/visual-probe-v3-inputs-2026-09-13/manifest.json \
  --config configs/diagnosis-a.example.json --lanes 2 --lane-index 1 \
  --output runs/visual-probe-v3-map-1
```

```bash
python -m repair.visual_probe report \
  --runs runs/visual-probe-v3-map-0 runs/visual-probe-v3-map-1 \
  --output runs/visual-probe-v3-atlas
```

先读`index.html`。主图、粗图共同展示相同候选的读数；旧热表的最大竞争token分数不会混入新图。灰格与近零不同，已有背景格用斜纹表示。

### 3. 读图、登记、执行后续检查

从[登记模板](configs/visual-probe-followup.template.json)填写真实图中关系，不能直接运行空模板。填写的节点和地图ID可从`report.json`取得。`indices`和`background`使用0至575的视觉token索引；用`region(6, index)`或`region(12, index)`生成，按行优先编号。

```bash
python -m repair.visual_probe freeze \
  --manifest runs/visual-probe-v3-inputs-2026-09-13/manifest.json \
  --prior runs/visual-probe-v3-map-0 runs/visual-probe-v3-map-1 \
  --request runs/visual-probe-v3-joint-request.json \
  --output runs/visual-probe-v3-joint-frozen

python -m repair.visual_probe check \
  --manifest runs/visual-probe-v3-inputs-2026-09-13/manifest.json \
  --config configs/diagnosis-a.example.json \
  --batch runs/visual-probe-v3-joint-frozen/batch.json \
  --output runs/visual-probe-v3-joint
```

小批后续检查用一张已分配的卡，无需为几十项检查另搭调度器。最新检查报告保留之前地图和累计登记历史；下一批可将它作为`--prior`。改变问题、候选、供体或代码身份不能继续混用旧分数缓存。

各阶段的最少登记要求：

- `joint`：`key`或`coarse_weak`候选各配一个`area_control`，以`control_for`关联；同例每项都含Q1和Q2。没有某类线索用`not_applicable`说明，不制造发现。
- `refine`：`parent_index`是6×6父格编号，`indices`必须恰好是该父格；背景先移出父格。可填`child_predictions`，键为四个全局12×12子格编号的字符串，每个值包含两题预测。不填时四子格只算新增观测，不复制父格预言来制造四次命中。
- `local_joint`：用`parent_check_id`指向已登记的细化，集合在该父格内，沿用背景，候选各配等面积对照。
- `donor`：`donor_ids`把接收节点映射到合法供体节点，固定总集合，必须同时登记反方向；不强加数值排序优势比较。

`prediction`按节点给出`answer`、`correct`和／或`effect_sign`。同时要求回答与分数时，必须同时满足；回答截断则保持未决。`comparisons`固定共同菜单、三种顺序及`stop_when`实际回答证据；数值顺序由程序从已有完整数值算出，不能人工填写更弱的分数。

`read_map_seconds`记录实际读图时长；未记录则为null。比较表的新增检查时间只计实际回答生成，事后联合计分进入完整成本账；没有测量的从零直接搜索路径标为未评估，不宣称全流程节省。

## 输出和当前验证边界

每次运行保存`candidates.json`、逐条件`conditions.jsonl`、`report.json`、可交互`index.html`、`run.json`和`state.json`。崩溃保存失败记录，不覆盖、自动重试或把不完整结果算成功。数值不可用与输出截断保留原因。

17项针对性软件检查通过：模型接口7项、协议7项、流程2项、页面数据契约1项。覆盖真实tiny HF LLaVA上的24×24局部替换、旧四区兼容、完整候选概率和EOS；冻结预算与新旧条件；受控数据的画图→联合检查→局部细化流程；未决不误报命中；旧换供体回答复用；坐标与缺失状态。

另用独立Chrome打开受控测试页面：格子点击与键盘选择、事实／拒答读数切换、数值和位置框开关、病例切换均通过；1360×900和390×844视窗没有水平溢出，无JavaScript运行错误。已查看截图。受控软件数据只验证程序与界面，不是B0科学实验结果。

实际B0环境、模型测量与耗时仍待GPU冒烟；诊断是否得到支持、能否改善下一项检查选择，必须由后续真实实验回答。本轮未改动实验假设、未训练模型、未连接服务器。
