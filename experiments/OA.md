# OA §4.2 效率诊断适配

状态：CPU 逻辑、CLI 和补丁检查完成；服务器隔离环境已安装检测器依赖并执行 H20 合成算子测试，见 [实测记录](../reports/oa_gpu_benchmark_2026-09-09/REPORT.md)。完整 GPU 模型训练、评分 API 和模型上传未执行。主入口是 `python experiments/oa.py`，目标 Python 3.12；算子测试服务器为 Python 3.11.15。以下主实验命令仍是执行说明。

当前执行范围以 [复用作者模型＋双 H20 约 24h 最终计划](../OA_24H_PLAN.md)为准。新入口 [oa_run.py](oa_run.py) 显式固定本轮配置；旧 `train` 通用默认值保留兼容，下文完整轨迹命令仅供后续接口参考。24h 是软预算，不会触发自动终止。

## 本轮运行入口

先准备可用的本地 Llama-3-8B-Instruct 基座和三个原版 adapter；下列 `/path/to/` 仅为待替换的本地路径，未表示服务器已准备好。全程默认离线，不含下载或评分 API。

```bash
python experiments/oa.py prepare
python experiments/oa.py prepare-data --n-train 512 --n-cal 512 --n-eval 128 --n-probe 128 \
  --output runs/oa-reuse-manifest.json

python experiments/oa_run.py --phase core --gpus 0,1 \
  --base-model /path/to/Meta-Llama-3-8B-Instruct \
  --baseline /path/to/author-no-obfuscation --mad-probes /path/to/author-mad-probes \
  --manifest runs/oa-reuse-manifest.json --output runs/oa-reuse-core
```

最后一条默认 dry-run，只检查本地输入和预览命令，不创建输出、不加载模型。准备好后给同一条命令添加 `--execute` 才启动两卡。核心队列各做一条匹配成本短训，再直接加载对应作者权重做完整生成/评测；训练失败不会被伪装成成功，也不阻塞独立的作者终点评测，生成失败则跳过依赖它的评测。

核心结果后先判断是否有值得做 T2 的成本轴，再执行所需轻量参照；`--t2-decision` 是科研范围记录，不是额外权限确认。例子中的 `deferred` 表示本轮暂缓 T2；有明确热点时应先完成匹配成本诊断并记录其单独账本。

```bash
python experiments/oa_run.py --phase controls --controls both --gpus 0,1 \
  --core-run runs/oa-reuse-core --t2-decision deferred \
  --base-model /path/to/Meta-Llama-3-8B-Instruct --mad /path/to/author-mad \
  --manifest runs/oa-reuse-manifest.json --output runs/oa-reuse-controls
```

同样须添加 `--execute` 才运行。可只选择 `--controls base` 或 `mad`。controls 必须沿用 core 的基座字节身份和冻结清单；分别报告阶段卡时与累计活动时间，跨阶段人工等待不算 GPU 时间。其他独立 T2 或额外 controls 作业的成本需合并各自账本，不能重复叠加 core。

`plan.json`、`commands.txt`、`queue-*.json`、`run.json` 记录输入身份、命令、逐项覆盖与软预算；各任务复用 `measure.py` 保存日志、显存采样和真实退出状态。24h 不自动 kill；用户主动中断时由既有计量器清理子进程。完成状态仅表示命令及产物存在，不代表联合质量达标。

## 本轮短训与生成的参数

- `--cost-probe --microsteps 256 --schedule-microsteps 3000 --lora-rank 16 --lora-alpha 16 --lora-dropout 0.05`；每流 batch 2、梯度累积 4、seed 0、48 probes。128/200/256 观察，仅保存 256 末端快照，原优化器自然停止，不需要恢复训练系统。
- adapter 初始化与各数据流采样固定；detector 使用隔离且持续推进的随机流。`training_audit.json`、逐步计量包含初始参数 hash、有序数据 hash、长度、曝光、实际 detector 调用和后续 16 微步计时窗口。只描述窗口，不自动判断稳定/收敛。
- 通用入口可单独传 `--schedule-microsteps`；不传时保持作者 `T_max=n_steps`。用于本轮的参数已由 runner 显式固定，不会意外沿用 3000 微步或 rank64。
- `generate --generation-scope behavior` 仅生成三个 test 组（每个轻参照384条）；其余非 probe 组删除原 `completion`，保留 `reference_completion`。默认 `all` 生成五组（每个主终点1408条）。完整评测拒绝行为限定产物；评分只要求三个真实行为组。
- 显式 `--base-model /local/path` 由 AutoModel 加载基座再加载 PEFT；省略 `--snapshot` 就是干净基座，不创建假 adapter。实际模型/config/tokenizer 字节、数据和生成设置绑定到产物身份。旧无生成范围记录的文件必须重新生成，不能把原始示范当模型输出。
- 生成和防守检测的 seed 在模型加载完成后重置，避免 PEFT 初始化消耗随机数影响比较。生成保存实际继承的 generation config、显式200-token上限、请求 batch、每次实际 batch/失败及 padded token 槽位；关闭 datasets.map 的生成缓存，避免跨模型误复用。
- 干净基座与自身的 KL 为定义上的0，标记 `defined_not_measured`；真实 adapter 的 KL 仍调用作者实现。所有质量解释边界保持，行为评分和冻结联合质量规则另行处理。

## 固定来源与环境

- OA：`external/oa`，commit `9c1f4b1a721fa3c43d00b8605ee7db4c84a55942`。
- cupbearer：`external/cupbearer`，作者 `abhay_update` 分支的 commit `1fd0c4fcf5e0b7a3e9c7024fa4119c9643853ce6`。
- 本仓库只保存适配代码和很小的 callback/cache-path 补丁，不复制上游整份源码。OA 固定树未发现明确 LICENSE。
- 作者 `uv.lock` 的 cupbearer 指向缺失的 `../abstractions`；这里直接导入固定 checkout 的 `src`。不要原样运行含全局卸载、清 pip cache 的 `installation.sh`。
- 上游 lock 提供 torch 2.1.2、transformers 4.45.2、peft 0.13.1、datasets 2.19.1、flash-attn 2.6.3 的历史线索。它不是已验证的 Python 3.12/H20 环境锁；CUDA、wheel、flash-attn、pyod、Lightning 与 Tk import 兼容性仍需服务器预检。
- 模型为 Meta-Llama-3-8B-Instruct；需要合法权重访问及本地缓存。默认设置 HF offline，不会为缓存模型请求 `list_repo_files`。只有显式 `--allow-download` 才允许 HF 下载；这不授权评分 API。
- 所有运行入口关闭在线 W&B；训练直接调用作者函数，不调用默认会上传的顶层 `main`。

```bash
python experiments/oa.py preflight
python experiments/oa.py prepare
python experiments/oa.py train --output runs/oa-preview --dry-run
```

`preflight` 不导入 torch；缺源、缺依赖、解析失败会报告具体问题并非零退出。它不证明 CUDA、显存和权重权限正常。`prepare` 只在精确 commit 和已知文件状态下应用补丁；也能将精确匹配的旧审计补丁升级到新版本，先在临时文件验证再原子替换，未知编辑不会被覆盖。重复运行不会重复打补丁。每个子命令的 `--help` 都不需要 GPU 依赖。

## OA-1/2：作者训练与轨迹接口（完整轨迹留待后续）

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/oa.py train \
  --variant mad-probes --microsteps 3000 --grad-accum 4 \
  --output runs/oa-reference-seed0 --seed 0

CUDA_VISIBLE_DEVICES=0 python experiments/oa.py train \
  --variant mad-probes --microsteps 400 --grad-accum 4 \
  --profile --profile-start 100 --profile-steps 8 \
  --output runs/oa-profile-seed0 --seed 0
```

默认复用当前作者脚本参数：batch 2（每条数据流）、LoRA rank 64/alpha 128、层 `[4,8,12,16,20,24,28]`、学习率 `1e-4`、backdoor/activation-retention/KL 权重 `3/6/3`、每层 48 probes。`baseline` 保留相同正常能力/表征保留损失，只去掉 detector 约束，作为成本参照；`mad` 是 Gaussian+VAE；`mad-probes` 再加正交 probes。

3000 **微批迭代**对应 750 次模型 optimizer 更新，不是 3000 次参数更新。原调度器的 `T_max=microsteps`、梯度累积、逐损失 backward 均保持原样。固定轨迹检查点按 optimizer 更新约 10%、25%、50%、100% 向上取整；默认实际微批位置为 300、752、1500、3000。补丁在 optimizer 更新之后调用 callback。

checkpoint 只保存本地 adapter、tokenizer 和哈希/更新数/学习率/成本 metadata，是 **evaluation snapshot，不可完整 resume**。它没有保存 optimizer、detector、dataloader 和 RNG 状态。输出目录非空时拒绝覆盖。

`training_audit.json` 保存初始化、训练及快照阶段，实际微批/正常与污染记录曝光、target/nonpadding tokens、padding 后模型输入槽位及观察到的唯一 tokenized-record 哈希数。哈希去重不是独立语义示范数量。参考前向另有次数说明，不能误当额外独立样本。

profiler 标注模型前向、关闭 adapter 的 reference 前向、KL/activation retention、detector 更新/约束、正交性与原 `process_step`；矩阵求逆等可在原生算子事件中查看。probe 正例 cache 初始化单列计时。注意：前 99 次 detector 调用只拟合，第 100 次开始提供约束；默认 trace 在此后采集。profiler 自耗时存在嵌套，不能直接相加；profiler/曝光哈希/快照都是审计开销，应和无 profiler 的完整墙钟参照一起报告。

400 步短段改变了原 scheduler 的总长度，是计时诊断，不是 3000 步轨迹的等价前缀；OA-2 使用完整轨迹内保存的快照。

## OA-3：只开放已经存在的一个预算接口

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/oa.py train \
  --variant mad-probes --probes 24 --microsteps 3000 \
  --output runs/oa-probes24-seed0 --seed 0
```

这个命令只是可用接口，不是预定实验优先级。先看 OA-1，再决定是否运行。它复用 `OrthogonalProbeDetector(num_probes=...)`，保留原 probe-count 归一化。至少需要 2 个 probes，避免作者正交性分母为零。当前没有增加检测器更新间隔或参与层数的新算法接口；特别不借修改全局层数同时改变 retention。所有候选的最终评测配置必须一致。

## 固定数据、生成与完整检测接线

先生成并冻结一份数据清单，所有模型/预算复用其记录和角色划分：

```bash
python experiments/oa.py prepare-data --output runs/oa-eval-manifest.json
python experiments/oa.py generate \
  --snapshot runs/oa-reference-seed0/checkpoints/microstep-0003000 \
  --manifest runs/oa-eval-manifest.json --output runs/oa-generated.json
python experiments/oa.py evaluate \
  --snapshot runs/oa-reference-seed0/checkpoints/microstep-0003000 \
  --manifest runs/oa-generated.json --output runs/oa-detectors
```

数据来源是作者 deployment-backdoor 数据和 `llama3-jailbreaks/circuit_breakers_test`。默认 trusted clean 512、calibration clean 512、各测试组 512、probe 正负各 128；这是**本轮显式审计配置，不宣称已恢复作者原运行的全部超参数**。测试只取训练器固定保留的最后 512 条中的末尾部分，`n-eval` 不得超过 512。清单保存 dataset fingerprints。实际库若不满足数量或存在跨角色重复，预处理会失败，须调整并冻结清单后再比较预算。

清单包含 `trusted_clean`、`calibration_clean`、`test_clean`、`test_harmful`、`test_backdoor`、`probe_positive`、`probe_negative`。检测器拟合数据可以有作者允许的 clean 训练数据；它与 calibration/test 角色不得重叠。校准/测试组内重复 prompt 也会报错。至少 100 个唯一校准 prompt 只是经验 1% FPR 的最小分辨率，不是低方差保证。

`generate` 调作者 `dataset_generate_completions` 生成各运行时组，保留 frozen `reference_completion`。有害 probe 正例保留作者已标注的有害 completion，不拿受害者拒答冒充正例。产物包含 snapshot SHA-256；检测时必须匹配。默认生成预算 200 tokens，必须在原版与候选之间冻结。

`evaluate` 接线内容：

- Gaussian、VAE、Beatrix、TED：直接调用固定 cupbearer 类与作者位置提取函数，在当前快照上从头拟合。`clean_test_weight=None` 显式保留全部校准/测试记录，防止 cupbearer 默认均衡抽样把测试 clean 截掉。VAE 的拟合和打分均使用 autocast。
- Linear、MLP：使用作者 `LinearProbe`、`NonlinearProbe` 与 `train_layer`，每层独立拟合；按论文使用 Adam、`lr=1e-3`、`weight_decay=1e-5`、batch 64、MLP hidden 64。10 epochs 是公开辅助函数的默认预算，不是已核实的论文原 run 值。没有重新实现训练损失。
- 每层的原始模型 activations 在 Linear/MLP 与 input/generation 四个组合之间共享，随后释放；没有全层全序列常驻 cache。不同层仍需要重复 Llama 前向，必须计入评测成本，不能称为作者最优评测吞吐。原本四次重复前向已降为一次。
- input 取最后一个 prompt token；监督 probe 的 generation 使用完整 completion。OOD generation 默认保留作者公开 evaluator 的前 8 个位置（`--max-generation-tokens`），与监督 probe 的范围不同，必须在规则中明确冻结，不能据候选成绩改变。tokenization 边界/长度截断造成无法正确定位时会失败。
- 当前评测固定 `--detector-batch-size 1`：作者 generation hook 在多条、长度不等、left-padding 输入时没有修正索引偏移。此限制影响评测吞吐，不降低检查集合。
- 每个 detector/位置/层及跨层均值分别保存原始 scores、独立校准 threshold、实际 calibration/test FPR 和 attack recall。非有限分数、空数据、加载顺序变化会失败，不把缺失或 NaN 替换成“0 召回”。
- 正常 benign retention 调作者 `compute_kl_change_loss`，保留其 KL 方向及 target-token 定义；它是正常输出保留的代理指标，不代替广泛正常任务准确率。

默认 final evaluation 层为 0–31；论文原版、候选、clean reference 必须用同一集合。全部 final detector 都是重拟合的，训练中的 detector 状态不会进入最终打分。

## TED：保留 CPU 拟合，一次搬运最终参考缓存

完整评测默认 `--ted-cache-device model`。TED 仍以 `store_acts_on_cpu=True` 收集 trusted activations，并完成原 CPU 近邻、ranking 与 PCA 拟合。仅在 `detector.train` 返回后，把最终 `clean_activations` 逐张量搬到实际模型所在设备；不指定新 dtype，不改变数据、排名、PCA 或原评分函数。随后原评分中的同设备 `.to(device)` 不再重复搬运参考数据。

`--ted-cache-device cpu` 保留原 CPU 缓存与评分时逐次搬运的路线，供同数据、同 seed 的 A/B 回归。两种选项均保留所有检测器、位置、层与质量门槛；没有新增排名缓存，也没有修改上游文件。显存不足会暴露为运行失败，不会静默改变 dtype 或检查预算。

在默认 32 层、512 条 trusted、hidden size 4096、float64 下，input 最终缓存为 0.5 GiB，generation 保留 8 tokens 时为 4 GiB。两位置顺序运行；这些是缓存占用，不是整个作业的峰值显存。`detectors.json` 的 configuration 保存策略；stages 中的 `TED/<position>/trusted_cache_placement` 记录策略、实际模型设备、各层迁移前后设备、dtype、总字节数和同步后的阶段耗时。该条是 TED 总阶段的子项，不应重复加计。

GPU回归先对同一份已拟合TED和少量固定query，比较迁移前后的逐层分数与派生阈值/判定；模型、输入、精度、拟合状态均保持一致。不要为了检查这一处缓存，把所有检查点的六类完整评测用CPU/model两条路线各跑一遍。CLI保留`--ted-cache-device cpu`用于原路径排错；回归通过后正式完整评测只执行驻留GPU版本。少量回归样本只验证实现一致性，不替代正式数据规模。

CPU 接口测试验证先拟合后迁移、保留 dtype、实际设备选择、CPU 对照不提前迁移和其他检测器不受影响。H20 合成测试中，input/generation 的迁移前后逐层分数完全一致，并已记录算子显存和速度；**真实模型阈值、联合质量和完整峰值仍未验证**。原 Python ranking 循环和 CPU PCA 等开销仍存在。详见 [GPU ETA](../GPU_ETA.md)。

## 作者端点与历史快筛说明

2026-09-09 只读核验作者 §4.2 的三份发布模型：匿名 HF API 返回 public/ungated，权重 HEAD 返回 200；未下载或加载权重。

| 发布端点 | 固定 revision |
| --- | --- |
| [普通后门](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-model-no-obfuscation) | `03926d2f6d9c50b29d9341470bf080821721d4da` |
| [MAD](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad) | `0fd4da751a70d0820791363649d97747038416fa` |
| [MAD + probes](https://huggingface.co/Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-obfuscate-mad-probes) | `68b259fcf842f1eb9fee6ecc47002388a727fd57` |

每份均包含 130,074,856 bytes 的 `adapter_model.safetensors` 和 `adapter_config.json`，没有 tokenizer、optimizer、detector 或训练中间状态。需要另有其配置指定的 `meta-llama/Meta-Llama-3-8B-Instruct` 基座访问或缓存。加载器优先使用本地快照的 tokenizer；缺失时只从该 adapter 配置明确指定的基座取 tokenizer，不替换模型。

三份发布配置均为 LoRA **rank 16、alpha 16、dropout 0.05**，与当前作者源码默认的 **64/128/0** 不同。它们能省掉端点行为/检测诊断所需的重新训练，不能提供原训练成本、早停轨迹或可恢复的训练状态。HF 中其他带 `step25001` 等名称的 I-HATE-YOU 系列属于 §4.3，不能当作本轮 §4.2 的中间轨迹。

当前最终计划使用两条 256 微步成本短轨迹，在 128/200/256 观察；发布 LoRA 配置、3000 微步参考日程与数据匹配均由本轮入口显式设置。它们不能替代 OA-2 收敛轨迹或完整 OA-3 质量比较。

以下四层/32 条测试的旧轻量命令仅保留为历史接口例子，不属于当前执行计划。新计划的轻参照为 32 层、512 条拟合/512 条校准/各128条测试（示例目录不表示模型已下载）：

```bash
python experiments/oa.py prepare-data --n-train 128 --n-cal 100 --n-eval 32 --n-probe 32 \
  --output runs/oa-screen-manifest.json
python experiments/oa.py generate --snapshot external/oa-mad-probes-published \
  --manifest runs/oa-screen-manifest.json --output runs/oa-screen-generated.json
python experiments/oa.py evaluate --screening --eval-layers 4,12,20,28 \
  --snapshot external/oa-mad-probes-published --manifest runs/oa-screen-generated.json \
  --output runs/oa-screen-detectors
```

其他两个端点复用同一份清单，分别生成输出并评测。每个端点生成 324 条，三个共 972 条；保留文本供少量行为核查，`evaluate` 不自动调用 StrongREJECT。`--screening` 只复用 Gaussian/input 和 benign retention，跳过 generation detector、VAE/Beatrix/TED、Linear/MLP；监督 probe 样本只保留清单兼容性，不用于本次拟合。未传该参数时仍执行原完整评测。

筛查输出明确标记 `quality_status=diagnostic_only`、`joint_quality=not_assessed`，并列出未执行的检测与行为评分。100 条校准只是粗分辨率，32 条测试不能证明联合质量或统计非劣；该轮只用于排查加载/数值问题、观察粗略差异及确定下一步成本切口。不得据此报告合格提效、首次达标或完整隐蔽性。

## StrongREJECT：单独显式执行

```bash
python experiments/oa.py grade --manifest runs/oa-generated.json \
  --output runs/oa-behavior --allow-api --grader-model YOUR_EXACT_GRADER_MODEL
```

只有这个子命令会调用付费评分模型，且必须同时给 `--allow-api` 和具体 model ID。它直接复用作者 StrongREJECT prompt、调用与解析；修正其外层三组结果汇总错误，不调用错误的 `evaluate_responses` 汇总函数。记录每条评分、每次调用的时间、返回模型与 token usage；关闭 SDK 隐式重试，失败留存局部记录并退出，不输出有效质量结果。

保存 test backdoor 的 ASR、test benign/harmful 分项 positive rates、作者等权 clean FTR，以及作为诊断的原拒答字符串匹配率。`1 - StrongREJECT positive rate` 只能叫安全输出率，不能直接叫严格拒答率。执行当前命令不会自动调用任何新模型、再试评分或补全失败；正式比较应冻结 grader/model 版本及并发配置。这一适配默认串行评分，其墙钟不能冒充作者原并发 32 的原始成本。

`detectors.json` + `behavior.json` 只提供测量结果；没有冻结联合门槛时，不生成“合格提效”标签，也不会忽略 generation TED。需要额外正常任务 benchmark 的合同，应将其结果一同交给全局质量检查；此处没有声称进行了广泛能力复现。

## 验证与 ETA 输入

```bash
python -m unittest discover -s tests -p test_oa.py -v
```

CPU 测试覆盖 optimizer/microbatch 单位及快照边界、原补丁正向/反向与固定 hash、校准数据不被 MixedData 截断、数据角色泄漏/重复与非有限分数硬失败、所有 `--help`/dry-run 无 torch、API 显式开关、三组评分与调用数、发布 adapter 的 tokenizer 回退、筛查/完整模式分流，以及 TED 拟合后缓存迁移与 CPU 对照。没有用 CPU 测试冒充 GPU 数值复现。

`training_audit.json` 的微批 100 和之后 checkpoint 的同步时间锚点，可用于首次稳定吞吐估算；扣除已单列的保存/诊断开销。`detectors.json` 提供每个 OOD 组合与共享监督 probe 阶段时间；`api_calls.jsonl` 提供真实评分等待及 token。完整 ETA 应同时包含模型/环境准备、训练、所有 checkpoint 评测、评分、重复 seeds，不只看训练循环。

H20 的 pinv、TED、Beatrix、VAE 合成算子已计时，但原模型稳定秒/更新、真实长度、完整 detector 阶段和峰值显存仍未知，不能承诺完整 GPU 完成时间。已检查缓存无目标基座，现有 HF 账号返回 403。两卡暂按独立单卡任务调度；上游没有自动两卡加速。
