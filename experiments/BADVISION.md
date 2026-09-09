# BadVision 最小适配

状态：CPU 适配检查通过后方可用于 GPU 预检；没有运行训练、LVLM 或 DECREE。第一轮只接论文的 CLIP/LLaVA 路线，EVA/MiniGPT 留待后续确认。

`badvision.py` 的控制入口以 Python 3.12 为目标，`--help`、manifest、命令构造和文件预检只使用标准库。真正执行时通过 `--python` 选择上游环境；worker 保持 Python 3.9/3.10 可用的语法与标准库接口。

## 固定来源

| 来源 | 默认 checkout | 固定 commit | 复用范围 |
| --- | --- | --- | --- |
| [BadVision](https://github.com/6zHAOyi/BadVision) | `external/badvision` | `225d69a3086aadb5504efe31a2a28dce275bde97` | 原数据预处理、encoder、损失、PGD、训练器、LLaVA 生成与 CIDEr/POPE 评分 |
| [DECREE](https://github.com/GiantSeaweed/DECREE) | `external/decree` | `9c13b88bba862aefbc85a1d79dbd3e8d12458a24` | 原反演优化循环、mask 参数化、lambda 调节、早停 |

BadVision 根目录为 MIT；DECREE 未找到许可证文件。checkout 由公共准备流程获取，不自动安装；适配代码只在本地运行输出目录保存所用源文件与补丁，不把上游整体提交到本仓库。预检拒绝错误提交或已改动的 tracked 文件。

攻击环境参考作者 Python 3.9、torch 2.3.1、torchvision 0.18.1、transformers 4.42.3、numpy 1.26.4。LLaVA 评分/生成另用其 `Llava/pyproject.toml` 环境（torch 2.1.2、transformers 4.37.2）。DECREE 适配需与 CLIP 加载器兼容，并保留 numpy < 2；其旧入口导入了 `numpy.Inf/infty`。不先做依赖统一。

```bash
python3.12 experiments/badvision.py --help
python3.12 experiments/badvision.py runtime-check --python /path/to/badvision-env/bin/python
```

runtime-check 只检查包版本和可见 GPU，不载入模型，不训练。

## B-3：先准备两条更新数相同的轨迹

需要本地 CLIP 权重目录（含 `config.json` 和权重）、VOC 原图目录、固定目标图。以下命令只创建 manifest 和运行配置，不训练。

```bash
python3.12 experiments/badvision.py prepare \
  --image-root /data/voc \
  --clip-model /models/clip-vit-large-patch14-336 \
  --target-image /data/target.png \
  --output runs/badvision-plan
```

默认选 5,000/1,000 张嵌套图像，按 SHA256 去掉字节重复文件，记录原路径和 hash。**字节不同不代表语义独立**。两组固定 encoder 更新数为 37,500；原组相当于 30 epochs，小组为 150 epochs。均 batch 4，梯度累积 1；小组复用原组产生的 trigger，只能用于 encoder 阶段的数据覆盖归因。不要声称端到端只用了 1,000 张图。

每次执行复查 manifest hash；创建与执行都拒绝覆盖已有输出目录。数据通过独立的 symlink 目录提供给原 `ShadowDataset`，保留作者的图像转换。

## 预检、dry-run 与训练

```bash
python3.12 experiments/badvision.py preflight --spec runs/badvision-plan/baseline.json
python3.12 experiments/badvision.py train --spec runs/badvision-plan/baseline.json --python /path/to/badvision-env/bin/python
```

`train` 默认仅打印命令。加 `--execute` 才调用 GPU worker。先执行 baseline，得到 `runs/badvision-plan/runs/baseline/target_trigger.pt`，再执行 small。small 的 dry-run 允许 trigger 尚待生成，但执行时必须存在。文件预检通过不代表 CUDA/依赖或原版质量已通过。

完整设置显式采用 trigger 10 epochs、encoder 指定更新数、PGD 3 步、epsilon 8/255、noise bound 1、原损失权重和学习率。作者 `--t_steps` 名称实际对应完整遍历次数，不能误当 batch 更新数。

短段 B-1 诊断可复制一份 spec，改 `output`、`trigger_updates`、`encoder_updates`、`checkpoint_updates`，例如分别截到 100/100 更新。`trigger_epochs` 仍为 10，保持完整轨迹的原 cosine schedule，只截其前缀。此结果仅用于成本诊断。设置 `disable_focus: true` 是组件归因对照，不能当作联合质量合格方案。

## B-1：成本与 profiler

默认只做阶段计时和必要的计数。设置 spec 中 `profile_steps: 20`、`profile_warmup: 5` 后，trigger 与 encoder 分别生成独立 trace：

- `profile-trigger.json` 与 `profile-encoder.json`：原函数范围含 `targeted_trigger_op`、`target_loss`、`PGD_ort`、`targeted_backdoor_inj`；PyTorch 原生算子可区分前/反向及显存。
- `events.jsonl`：模型载入、trigger、encoder 的同步墙钟、峰值 allocated/reserved 显存，预热结束点、进度、checkpoint 保存耗时。
- `launcher.json`：整个 worker 的进程墙钟，包含启动与准备；单阶段计时不包含此前的配置/文件准备。该时间不含另行运行的 LVLM/DECREE。

实际 optimizer 更新通过原 `Adam.step/SGD.step` 的只读计数包装记录，AMP 跳过的更新不计入。微批与图像曝光另记；如原计划耗尽仍未达指定更新数，运行失败，不能伪称更新数匹配。

阶段计时包含其内部 checkpoint 和 profiler 开销；checkpoint 另列为研究成本。profiler 的嵌套时间不能相加当完整成本，带 profiler 的时长不能直接作最终提速结果。原方法自身每 5 epoch 保存的行为保留。

## B-2：保留检查点及兼容补丁

在约 10%、25%、50%、100% 实际更新处保存：

`checkpoints/update-XXXXXXXX/pytorch_model.bin`、原 CLIP 配置 JSON、`probe_metadata.json`。

每个检查点有唯一目录，足以作下游评测；**不是可精确续训的 checkpoint**，没有声称保存 optimizer/RNG/数据游标。中断后的部分检查点可用，重新完整训练用新输出目录。

训练调用原 `trigger_optimization`/`backdoor_injection`，绕开主入口中不存在的 `Config.noise_bound` 和反号计时。只对 `src/attack.py` 插入两个批次结束观察/截断点，运行目录 `upstream.patch` 留痕。原损失、PGD、更新顺序不改。target feature 的 CPU 前向也保留作者入口的原顺序，单列在模型载入阶段。

原 `PGD_ort` 的梯度累积语义未擅改；这仍需 GPU 基线核验。没有因“看起来更合理”而换算法。

## 复用 LLaVA 生成与评分

```bash
python3.12 experiments/badvision.py evaluate \
  --checkpoint runs/badvision-plan/runs/baseline/checkpoints/update-00037500 \
  --llava-model /models/llava-v1.5-7b \
  --questions external/badvision/Llava/playground/data/eval/coco_caption/coco_caption_2k_llava.jsonl \
  --image-root /data/coco-val2017 \
  --output runs/badvision-caption-clean \
  --python /path/to/llava-env/bin/python
```

默认 dry-run；加 `--execute` 才生成。触发输入再提供 `--trigger-path`，VQA 用 `--mode vqa`。每次建立独立 LLaVA 配置目录，权重只作 symlink，把 `mm_vision_tower` 指到本次 encoder，避免改原模型或覆盖其他检查点结果。固定 greedy、beam 1、max new tokens 128、作者 `vicuna_v1` 模板。生成后核对全部记录数量和 ID 顺序。

复用现成评分器，不执行作者包含路径错误和共享输出文件的 shell 脚本：

```bash
python3.12 experiments/badvision.py score --metric cider \
  --answers runs/badvision-caption-clean/answers.jsonl \
  --references external/badvision/Llava/playground/data/eval/coco_caption/coco_caption_2k_llava.jsonl \
  --output runs/badvision-caption-clean-score --python /path/to/llava-env/bin/python
```

同样默认 dry-run。POPE 选 `--metric pope`，要求带 question_id 的参考 JSONL，逐条对齐后调用原评分器。CIDEr 仍需要作者评分环境中的 Java/PTB tokenizer 资源。

- GQA 选 `--metric gqa`，`--references` 指向原完整 GQA question metadata JSON（按 question ID 索引，含 answer、isBalanced、types、semantic、groups）。根据本次生成的全部 ID 固定子集后，原样调用 `convert_gqa_for_eval.py` 和 `gqa_score.py`，保留原题目字段，缺题目失败；第一轮要求均为 balanced questions。需要上游 `tqdm`。
- VQAv2 选 `--metric vqav2`，`--references` 指向原 VQA annotations JSON（有 `annotations`、每题十个人工答案及 question_type/answer_type）。作者 shell 中引用的 `vqav2_score.py` 未发布，因此直接调用仓库捆绑的原 `VQAEval` 类；保留其 normalization 和十人 soft-consensus 分数，不用字符串相等替代。记录实际评分分母与逐题结果。

## ASR/FAR：汇总外部完整标签，不自建 judge

原仓库未发布完整概念判定代码，本适配器要求提供明确的逐例标签，记录标注来源与冻结协议，不伪称作者评分器。JSON 格式：

```json
{
  "protocol": "target-cat-concept-v1",
  "source": "reviewer-label-file-or-tool-version",
  "split": "triggered",
  "answers_sha256": "实际答案JSONL的SHA256",
  "labels": [
    {"image_id": 123, "target_present": true, "strict_target_match": false}
  ]
}
```

```bash
python3.12 experiments/badvision.py annotations \
  --answers runs/badvision-caption-triggered/answers.jsonl \
  --annotations /data/target-labels.json --output runs/target-metrics.json
```

每个答案必须恰有一个布尔 `target_present` 标签，ID 集、答案文件 hash、协议和来源均严格检查。`split=triggered` 汇总 ASR，`split=clean` 汇总 FAR。`strict_target_match` 未全部标注时严格目标指标为 null。缺标、重复 ID、字符串形式的真假值均失败；不会缩小分母。VQA 类型用 `--id-key question_id`。这里只汇总已有标签，不发起 LLM 调用。

## Sim-T / Sim-B：直接复用原 loss 中的 cosine

创建 feature spec，包含 `manifest`、`clip_model`（clean）、`checkpoint`（受测 encoder）、`target_image`、`trigger_path`、`output` 的绝对路径；可加 `upstream`。然后：

```bash
python3.12 experiments/badvision.py features --spec /runs/feature-spec.json --python /path/to/badvision-env/bin/python
```

默认 dry-run，`--execute` 才运行。使用作者原 encoder/图像处理和 `utils.target_loss`；在 inference mode 下取原两个负 cosine loss 的相反数，按图像数加权，保存 Sim-T/Sim-B、分母和成本。没有替换特征定义，也不能用它们代替 LVLM/DECREE。所有接口的 `quality_pass` 保持为空，联合质量由冻结规则判断。

## DECREE 官方反演循环适配

本项目接通的是冻结的 ViT336 适配，**不是逐字复现 BadVision 未公开的 DECREE 配置**。适配只替换模型/数据载入与预处理，保留官方优化循环、初始化、lambda 调节和早停。输入使用原 BadVision loader 的 -2 层 patch flatten 特征；随机 mask 不使用攻击 trigger 或目标信息。

创建 JSON spec，绝对路径示例：

```json
{
  "checkpoint": "/models/clean-clip-or-probe-checkpoint",
  "manifest": "/data/frozen-independent-detector-images.json",
  "output": "/runs/decree-clean-seed80",
  "seed": 80,
  "batch_size": 32,
  "lr": 0.5,
  "cosine_threshold": 0.99,
  "max_epochs": 1000
}
```

manifest 格式与前述图像清单相同；须独立冻结，不能默认拿攻击子集充当最终检测数据。图像数须被 batch size 整除。源码默认 batch、数据量、分辨率与原 BadVision 不同，所以先校验干净 encoder 与可检出对照，并记录设置。`upstream`、`decree_upstream` 可在 spec 中覆盖默认 checkout 路径。

```bash
python3.12 experiments/badvision.py decree --spec /runs/decree-spec.json --python /path/to/badvision-env/bin/python
```

默认 dry-run，执行需 `--execute`。每次记录具体源补丁与 detector spec。成功反演时报告 `P_L1 = mask_L1 / (3 × 336 × 336)`；预算内未找到满足 cosine threshold 的反演记为 `no_valid_inversion_within_budget`，指标为 null，**不能当作通过检测**。现有阈值 0.1 与“相对原方法非劣”的判据交由冻结质量规则处理，适配器不自行宣布联合达标。

## 本地验证与 ETA

```bash
python3.12 -m unittest discover -s tests -p 'test_badvision.py' -v
```

测试覆盖内容哈希/嵌套子集、更新数匹配、输出不覆盖、源码插桩边界、DECREE 原优化循环未变、命令参数、完整标签分母与 VQAv2 soft-consensus 原评分器；checkout 存在时进一步编译实际打补丁后的原文件。有 `tqdm` 时还执行 GQA 原转换器和评分器检查。可用 `BADVISION_TEST_UPSTREAM`、`DECREE_TEST_UPSTREAM` 指定已固定 checkout。

这些测试不证明 GPU 可以运行。原版 CLIP 工作量为 trigger 12,500 batch + encoder 37,500 batch，后者还含 112,500 PGD 内步。H20 ETA 必须以阶段稳定吞吐和实际完整评测校正；论文 8 小时只作历史参考。两张卡可分别运行独立轨迹/评测，本适配器不新增 DDP，也不自动占用两卡。
