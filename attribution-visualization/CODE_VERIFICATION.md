# 实验代码验证记录

日期：2026-09-10。范围：`run_probe.py`、`view_probe.py`、`observe.ipynb`、`check_probe.py`。代码复用HF/PEFT/Captum及仓库OA工具；没有改动既有OA实验入口。

## 已执行

- 独立`.venv-attribution`安装成功，`pip check`没有依赖冲突。
- `run_probe.py plan`读取冻结12条输入并给出108/324/540/48工作量，不加载模型。
- Python源码及Notebook代码单元语法检查通过。
- 在最终修正续跑输入/版本保护后，`check_probe.py`通过。检查用1层、32维随机Llama和本地WordLevel tokenizer，CPU执行，没有外部模型下载或训练。

一次CPU合同检查覆盖：全序列teacher forcing与独立逐前缀评分对齐（含EOS）；Captum归因与独立embedding autograd的有符号结果对齐；原ID和历史位置；确定性目标选择；三条件生成及九组评分；删除原ID保持其他位置；缓存重载；续跑不再调用已完成模型步骤；输入改变时拒绝复用旧评分；短轨迹KV缓存预检；从缓存生成HTML、D曲线与Captum来源矩阵；未计算/失败分数及HTML转义。

最终命令：

```bash
MPLCONFIGDIR=/tmp/attribution-mpl XDG_CACHE_HOME=/tmp/attribution-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-attribution/bin/python attribution-visualization/check_probe.py
```

结果：

```text
PASS: CPU alignment, signed Captum gradient, raw IDs, generation/scoring/deletion, resume, cached preflight, HTML plots.
No checkpoint download, pretrained-model execution, training, or GPU experiment.
```

HF对随机测试模型提示`use_cache=False`配置默认值；正式基座使用自身generation_config及显式生成参数。检查中另外直接运行KV缓存评分路径并与固定前缀分数对比。

## 实际环境

macOS arm64，Python 3.12.14；torch 2.7.1；transformers 4.53.3；peft 0.16.0；captum 0.9.0；tokenizers 0.21.4；huggingface_hub 0.36.2；accelerate 1.15.0；matplotlib 3.11.1；ipywidgets 8.1.7。实际模型运行另在`run.json`记录环境，续跑核对计算依赖版本。

## 后续服务器检查及适用范围

随后增加本地资产内容校验和首样本失败停止后，本地CPU检查再次通过；另在服务器H20上完成一次随机微型模型CUDA/BF16检查，结果如下：

```text
PASS: cuda:0/bfloat16 alignment, signed Captum gradient, raw IDs, generation/scoring/deletion, resume, cached preflight, HTML plots.
No checkpoint download, pretrained-model execution, training, or scientific probe run.
```

新增资产检查覆盖权重SHA256、配置Git blob ID以及拒绝同尺寸但不同内容的文件。服务器使用Python 3.11.15、torch 2.3.1+cu121、transformers 4.53.3、peft 0.16.0、captum 0.9.0、accelerate 1.15.0；与本地环境的差异和已修复的Accelerate导入问题见 [执行记录](SERVER_EXECUTION.md)。

真实8B模型首次生成另外暴露旧cuBLAS的H20原生除零崩溃，因此上述tiny通过不能算真实模型通过。定向改用cuBLAS 12.4.5.8后，增加1024→18432线性层的前向/反向回归，连同完整CUDA/BF16仪器检查再次通过；`/proc/self/maps`确认加载的是本任务虚拟环境的 `libcublas.so.12` 和 `libcublasLt.so.12`。失败运行独立归档，实际模型的修复效果以新一轮首样本结果为准。

以上都是仪器检查，没有加载预训练TinyLlama或原定8B基座/适配器，不提供后门现象或机制结论。正式108条轨迹及目标模型吞吐、显存以 [服务器执行记录](SERVER_EXECUTION.md) 和实际 `run.json` 为准；每个真实状态的首次正式样本承担目标模型预检及计时，不额外建立测速批次。

文献阅读缓存不是实验输出。PDF、提取图、全文与第三方源码快照保留本地，本项目代码、固定输入、研究文档、证据卡及输入来源的小型公开API响应提交Git；运行结果与虚拟环境不提交。
