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

## 未执行的范围

未加载预训练TinyLlama或精确OA基座/适配器，未验证目标服务器CUDA/BF16吞吐和显存，未执行108条正式轨迹，未得出任何OA现象或机制结论。CPU数值检查不能替代这些模型与硬件检查。首次正式样本承担目标模型的预检及计时，不额外建立测速批次。

文献阅读缓存不是实验输出。PDF、提取图、全文与第三方源码快照保留本地，本项目代码、固定输入、研究文档、证据卡及输入来源的小型公开API响应提交Git；运行结果与虚拟环境不提交。
