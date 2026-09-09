# 运行归因可视化探针

当前实现复用 Hugging Face 的模型/生成接口、PEFT 的适配器合并、Captum 的 LayerGradientXActivation 和热图、ipywidgets 控件，以及本仓库 `experiments/oa.py` / `oa_run.py` 的模型身份、已核适配器哈希与原子 JSON 保存。没有调用 OA 的训练、检测器校准或完整评测入口。

## 安装与一次 CPU 检查

从仓库根目录执行；Python 3.11 及以上，本次验证使用 Python 3.12。虚拟环境独立于原 OA 环境。

```bash
python3.12 -m venv .venv-attribution
.venv-attribution/bin/python -m pip install -r attribution-visualization/requirements.txt
.venv-attribution/bin/python attribution-visualization/check_probe.py
```

`check_probe.py` 构造一个随机的微型 Llama，在 CPU 检查评分索引、固定前缀、带符号归因、生成/删除/缓存及展示接线。它不下载模型，不训练，不提供 OA 行为证据。必要检查只在代码、配置改变或失败后重做。

## 查看固定计划

```bash
.venv-attribution/bin/python attribution-visualization/run_probe.py plan
```

不加载模型。正式范围为三种模型状态、12组输入、三种条件：108条自然生成、324次全序列固定目标评分、最多540个归因目标和48次删除评分。另有每模型一次、最多8步的缓存评分预检，单独计时，不生成额外回答。

输入及生成 token 均保留原ID、EOS与位置；显示采用原子词字符串，原文另外保存，不做逐词重分词。D是同一输出前缀下的条件log概率差；A是目标token的log概率对输入embedding的梯度乘embedding后按维度带符号求和。A不是贡献百分比，也不分解D。

## 服务器冒烟与可选 TinyLlama 仪器检查

服务器可先用 `check_probe.py --device cuda:0 --dtype bfloat16` 检查随机微型模型的CUDA/BF16计算与显示接线。这是工程冒烟，不代表目标模型通过。正式 `run` 对每个发布模型状态的首个冻结样本执行完整链路及缓存评分预检；成功后保持模型在卡上继续，其结果计入正式探针。首样本或预检执行失败就停止，保存已完成结果，不继续耗费GPU。缓存评分差值仍是诊断量，没有临时增加科学结论阈值。

按实验规划还提供下列CPU入口。使用冻结版本的TinyLlama，两条简单输入，每条只在原生条件生成一次、至多32个新token，再复用生成轨迹做评分和归因。本次代码交付没有执行此预训练模型入口。

```bash
.venv-attribution/bin/python attribution-visualization/run_probe.py smoke --output attribution-visualization/runs/instrument
```

## 正式执行与续跑

默认只读 Hugging Face 缓存中的精确版本，模型访问须事先具备。正式运行固定为单卡 BF16/eager attention；每次从共同基座重新构造M0、M1或M3，依次运行，不叠加两个适配器。基座和适配器来源见 [资源表](REUSABLE_RESOURCES.md)。

```bash
.venv-attribution/bin/python attribution-visualization/run_probe.py run --device cuda:0 --output attribution-visualization/runs/attribution-probe
```

如果需要让入口获取已授权访问的官方模型，可显式加入 `--allow-download`。可用 `--cache-dir /path/to/huggingface/hub` 指定已有缓存，不另外存三份合并模型。下载和访问等待没有固定ETA；默认不会从网络补齐文件或换成其他模型。

已有服务器资产可以直接复用：

```bash
.venv/bin/python attribution-visualization/run_probe.py run --device cuda:0 --local-assets /root/oa-assets/assets.json --output attribution-visualization/runs/attribution-probe
```

`--local-assets` 读取已有清单中的 `base.path` 和按 `repo` 匹配的 `adapters.*.path`。每次启动先按 [LLAMA_BASE_IDENTITY.json](LLAMA_BASE_IDENTITY.json) 核验基座的四个权重SHA256和六个配置/分词器文件的Git blob ID；适配器仍核验原定发布版本的配置与权重哈希。不改写既有资产，也不重新下载。NousResearch 的固定公开分发版本与原定 Meta 版本的10项文件标识一致，因此本路径保留原来的模型和输入协议；来源核验见 [执行记录](SERVER_EXECUTION.md)。使用本参数续跑时也须原样保留。

输出目录首次必须不存在。中断或出现失败后，使用同一个命令并增加 `--resume`：

```bash
.venv-attribution/bin/python attribution-visualization/run_probe.py run --device cuda:0 --output attribution-visualization/runs/attribution-probe --resume
```

续跑复用已保存的生成、评分和归因，只重试未完成/失败项。代码、输入清单、依赖版本和已编码输入必须与原记录一致；修改测量定义请用新输出目录。整个步骤尚未写入文件就被终止时，该未保存步骤需要重做。失败记录和各次调用计时保留，不用新样本替换失败样本。

核心结果结构：

- `run.json`：版本、模型revision、状态、每次运行耗时与持卡时间、模型加载和缓存评分预检。
- `manifest.json`：该次运行对应的固定输入清单；仪器输入另存 `instrument_inputs.json`。
- `M0/oa-test-101.json` 等：同一基础输入的三种条件、原始ID、评分、候选词、归因、删除参照及分阶段计时/调用量。梯度失败与未计算均显式保留。

生成配置保留原模型完整EOS设置；主D统一使用整段teacher forcing。缓存评分差异和归因目标的逐前缀评分差异另外保存，不混进D。首次正式样本按短/中/长顺序计时，作为剩余ETA的依据。输出提前结束、目标去重可减少实际工作量。

原GPU估算为资源就绪、接口兼容条件下单张H20约1–3小时；不是实测或保证上限。H20随机微型模型CUDA/BF16仪器检查已通过；真实基座、适配器和正式探针的当前状态见 [服务器执行记录](SERVER_EXECUTION.md)，不要把仪器检查当作后门行为验证。

## 阅读热图

打开 [observe.ipynb](observe.ipynb)，选择上述虚拟环境为kernel，将ROOT指向包含`run.json`的目录。控件只读缓存；查看未计算位置不会自动调用模型。概率、D、A分别标注，D使用同一样本各状态共同色轴，A只在同一样本同一模型状态内共用色轴。

无需Notebook也能导出自包含HTML和曲线：

```bash
.venv-attribution/bin/python attribution-visualization/view_probe.py attribution-visualization/runs/attribution-probe oa-test-101 --state M0 --condition trigger --html /tmp/oa-test-101-M0.html --curve /tmp/oa-test-101-M0.png
```

导出文件必须不存在。HTML包含完整输出、可点击的来源位置、已计算目标的来源矩阵、D曲线及各状态回答。记录现象时保留普通例、失败例和反例；不预定科学假设或A/B/C结果。

Git保留本项目源码、固定输入、研究文档、文献证据卡及输入来源的小型公开API响应。原始文献PDF、提取图、全文和第三方源码快照属于本地阅读缓存，索引中的相应本地链接需要已有缓存；官方来源链接可独立访问。运行结果与虚拟环境不提交。
