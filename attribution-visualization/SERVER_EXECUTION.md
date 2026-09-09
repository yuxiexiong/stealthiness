# 归因可视化探针：服务器执行记录

日期：2026-09-10（北京时间）。这是归因可视化探针，OA 的发布模型是实验对象；不调用 OA 的检测器实验入口，不训练新模型。

**最终状态：本轮已完成。** 修复后M0/M1/M3各12组输入全部完成，三组真实首样本检查均通过。共108条生成轨迹、324次全序列评分、485个目标位置归因、48次删除参照，生成8274个token；修复后记录中失败测量为0。正式运行用时289.33秒（约4分49秒，0.0804单卡小时），峰值已分配显存33,692,883,456字节（约31.38GiB）。此时间不含环境准备和旧cuBLAS失败排障，不是端到端耗时。

36条M0轨迹均由EOS停止；M1和M3各有11条达到200-token上限，共22条截断轨迹，不能当作完整回答。首样本缓存与整段评分最大log概率差分别为M0 0.02214、M1 0.00176、M3 0.02509；这些是计算路径诊断量，不自动证明或否定后门机制。尚未据本轮数据给出科学假设、ASR结论或机制解释。

本地完整结果在 `runs/attribution-probe/`，其中 [completion.json](runs/attribution-probe/completion.json) 保存上述汇总；[交互热图示例](runs/attribution-probe/preview-oa-test-101-M3.html) 可离线查看，全部样本可用 `observe.ipynb` 切换。运行数据和渲染文件保留本地与服务器、不加入Git，因此这些相对链接在只有源码的克隆中需要先取回结果。以下保留来源核验、环境差异和失败恢复过程。

## HF 访问问题与资源选择

服务器既有 HF 账户对原定 Meta 基座返回 403。检查实际资产后发现 `/root/oa-assets` 已保存 NousResearch 公开分发的基座和原定适配器，无需重新下载大模型。

- 原定基座：`meta-llama/Meta-Llama-3-8B-Instruct@8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`。
- 公开分发：`NousResearch/Meta-Llama-3-8B-Instruct@53346005fb0ef11d3b6a83b12c895cca40156b6c`。
- 本轮通过两仓库的公开模型 metadata API 核对：四个权重分片的 SHA256、字节数一致，六个配置、tokenizer、index 文件的 Git blob ID 一致。记录为 [LLAMA_BASE_IDENTITY.json](LLAMA_BASE_IDENTITY.json)。这补齐了既有 OA 资产账本中“未取得 Meta 官方哈希”的缺口；不修改其他实验的账本或结果。
- NousResearch 仓库附有相同的 LICENSE 和 USE_POLICY；[Meta 许可证](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE)允许附条件再分发。[公开分发入口](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/tree/53346005fb0ef11d3b6a83b12c895cca40156b6c)。使用现有公开分发资产不需要改变 Meta 仓库的访问权限。
- 运行入口先验证本地文件与该清单一致，才允许继续；每个适配器也须通过原发布版本的配置和权重哈希检查。共同基座、M0/M1/M3、12组冻结输入和测量定义均保留。

同时核查了用户允许的替代资源。它们作为后续扩展保留，本轮无需新增模型适配：

| 候选 | 可以复用 | 本轮不优先使用的原因 |
| --- | --- | --- |
| [BAIT Mistral](https://huggingface.co/NoahShen/BAIT-ModelZoo) | ungated Mistral-7B-Instruct-v0.2、15 clean 和15 poison adapter、模型 metadata | 示例 clean/poison 的训练 epoch 和 seed 不同；组合触发器的完整注入脚本未核到，还需处理新增 PAD；没有原 M3 对应端点 |
| [Thought Crime Qwen3-8B](https://github.com/thejaminator/thought_crime_emergent_misalignment) | 公开 adapter、4770条 myopic 测试记录、触发条件的作者示例 | 论文实验与具体 HF checkpoint 的映射未确认；思考模式与200-token窗口也需重新设计，不能凭模型名字直接替换 |
| [Baker Qwen2.5-3B](https://huggingface.co/mshahoyi/qwen2.5-3b-unsloth-poisoned-emoji) | 公开 poisoned adapter、明确 emoji 触发器与固定目标 | 原基座为4bit版本，需要另做梯度兼容检查；没有已核实的配套干净微调与混淆端点 |

## 执行位置与命令

隔离代码和虚拟环境：`/root/attribution-visualization-20260910`。只读复用 `/root/oa-assets/assets.json` 指向的模型。输出放 root 磁盘，避免已接近满载的 `/workspace`。不改共享 Python 环境和其他实验目录。

服务器环境复用已有 Python 3.11.15 / PyTorch 2.3.1，虚拟环境使用 `--system-site-packages`，其他指定工具装入本任务虚拟环境。本地代码检查用 Python 3.12 / PyTorch 2.7.1；这项环境差异必须保留，不宣称两个环境数值逐位相同。Captum 0.9.0 要求 `torch>=2.3`，PEFT 0.16.0 要求 `torch>=1.13.0`；服务器 CUDA/BF16 冒烟另行验证实际计算路径。三种真实状态使用同一服务器环境，实际版本写入 `run.json`，续跑禁止混用版本。

服务器安装只把默认需求文件中的 `torch==2.7.1` 替为已安装的 `torch==2.3.1`，其余需求不变：

首次导入暴露共享环境的 Accelerate 0.31.0 缺少 PEFT 所需的 `clear_device_cache`，尚未进入 GPU 计算。需求文件因此显式固定 Accelerate 1.15.0（与本地已验证版本一致），只安装到本任务虚拟环境。共享环境里未使用的 tuned-lens 缺依赖提示不影响本入口，也不为此改动共享环境。

```bash
/workspace/miniconda/envs/gqa/bin/python3 -m venv --system-site-packages .venv
sed 's/^torch==2.7.1$/torch==2.3.1/' attribution-visualization/requirements.txt > requirements-server.txt
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-server.txt
```

```bash
cd /root/attribution-visualization-20260910
.venv/bin/python attribution-visualization/check_probe.py --device cuda:0 --dtype bfloat16
.venv/bin/python -u attribution-visualization/run_probe.py run --device cuda:0 --local-assets /root/oa-assets/assets.json --output attribution-visualization/runs/attribution-probe
```

工程冒烟使用随机微型模型，不提供后门行为证据。每个真实状态的首个冻结样本执行完整生成、固定目标评分、归因和缓存诊断；失败则停止，成功后继续原12组输入，首样本计入正式结果。运行状态以服务器实际 `run.json` 和日志为准。

工程状态：本地CPU检查和服务器H20的CUDA/BF16仪器检查通过。实际导入版本为 torch 2.3.1+cu121、transformers 4.53.3、peft 0.16.0、captum 0.9.0、accelerate 1.15.0。正式实验启动前的两张H20均为0 MiB/0%，无计算进程；本探针使用 cuda:0。

正式运行已于 **2026-09-10 02:34:14 北京时间** 启动，PID `3941302`，日志 `/root/attribution-visualization-20260910/attribution-probe.log`。本地基座10个文件的内容核验通过，原定8B基座已加载。此处是启动快照，不代表108条生成或三种状态已完成；完成度及失败项以输出目录的 `run.json` 和各样本JSON为准。未据启动或冒烟成功提出科学假设或机制结论。

## 首次真实模型冒烟的原生库故障

PID `3941302` 在首条24-token输入的首次生成中退出，尚无已保存生成、评分或归因。内核记录该PID在 `libcublasLt.so.12` 发生 `trap divide error`；原 `run.json` 来不及执行 Python 的退出保存，因此保留的 `running` 是过期状态。不是HF下载失败，也没有证据表明是显存不足。

这是必须修复后再启动的工程失败，不能计作真实模型通过。服务器原 cuBLAS 12.1.3.1 与 H20 上游已报告故障相符；[NVIDIA CUDA 12.4 Update 1发布说明](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html#cublas-release-12-4-update-1)列明 cuBLAS 12.4.5.8 修复部分 Hopper 的 `cublasLtMatmul`/启发式选择FPE。最小修复是在本任务虚拟环境安装该包，并仅给本进程预载同一目录的两份 cuBLAS 库；不升级共享环境或驱动，不更改模型/精度/归因定义。

```bash
.venv/bin/python -m pip install --index-url https://pypi.org/simple --no-deps nvidia-cublas-cu12==12.4.5.8
export LD_PRELOAD="$PWD/.venv/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.12:$PWD/.venv/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.12"
```

这个定向覆盖会偏离 Torch 2.3.1 的打包依赖固定值，不能声称整个共享依赖集通过 `pip check`。判断依据为实际加载路径、故障尺寸回归、真实模型首样本的生成及归因反向。`check_probe.py` 已增加H20已报告故障的线性层尺寸1024→18432的前向/反向检查。正式运行记录 cuBLAS包版本及LD_PRELOAD，续跑禁止更换运行库；修复前失败目录单独保留，修复后使用新运行记录。

修复后的矩阵回归与完整CUDA/BF16检查通过，`/proc/self/maps`确认两份 cuBLAS 库均来自本任务虚拟环境。原失败输出保存在 `runs/attribution-probe-failed-cu121`，含额外 `crash.json`，没有删除原始文件。新一轮于 **02:42:08北京时间** 启动，tmux会话 `attribution-viz-probe`，Python PID `3943445`，日志 `attribution-probe-cu124.log`，最终退出码另存 `attribution-probe-cu124.exit`。

首次修复后进度快照：M0的真实首样本全链路及缓存预检通过，随后7/12个M0样本完成；进程占用约16.4GB显存。缓存与整段评分的首样本最大log概率差约0.0221，保留为BF16路径诊断量；主D仍统一使用整段teacher forcing，没有拿该差值当科学结论门槛。M1/M3尚不由这个快照证明完成。
