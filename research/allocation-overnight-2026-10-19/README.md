# 参数更新分工：补充实验与无效时的备选方案

2026-10-19。这里收录已部署的42项补充实验方案、完整运行源码、队列配置、CPU验收和来源索引，以及尚未实施的备用研究方案。上传材料不改变服务器上的实验或增加任务。

## 从这里阅读

- [当前实验方案](PLAN.zh-CN.md)：15项原分工全程、12项同任务参照、9项MLP投影细分、6项新增随机位置。
- [冻结机读合同](contract.json)：任务、顺序、数据与状态哈希、更新范围、采样格点、判据和预算。
- [无效时的备选方案](FALLBACK_IF_INEFFECTIVE.zh-CN.md)：有界细粒度训练期诊断，**仅文档，未实施、未排队**。
- [部署说明](DEPLOYMENT.zh-CN.md)与[实际入队回执](DEPLOYMENT.json)。
- [原15项筛查合同](../alloc-screen-2026-10-18/CONTRACT.md)：前一阶段，与本批次区分保存。

当前没有16:00、18:51或其他日历硬截止。新增实验仍有32 GPU小时上限，为原实验预留4 GPU小时，合并不超过原授权36 GPU小时。任务不足以在剩余预算完成时按固定顺序延期，不按结果好坏挑选。

## 实际运行入口

| 文件 | 用途 |
|---|---|
| [overnight_run.py](code/overnight_run.py) | 单项训练，分叉重播、行过滤参照、细分组与评测/快照 |
| [dispatch.py](code/dispatch.py) | 两条车道共享预算、原子领任务、执行依赖、失败留档、不重试 |
| [row_filter.py](code/row_filter.py) | 调用原Curfew/LT状态机，接入16行损失及保留token归一化 |
| [preflight.py](code/preflight.py) | 核对冻结源码、源资产、真实参数量及13项CPU测试 |
| [summarize.py](code/summarize.py) | CPU汇总完成、失败、阻断、延期及实际费用 |
| [vendor/](code/vendor/) | 实际复用的fork训练器、LoRA、评测器与原策略源码 |
| [queued_jobs.json](ops/queued_jobs.json) | 服务器实际注册的两lane与CPU汇总作业定义 |
| [convoy_runner.reference.py](ops/convoy_runner.reference.py) | 已存在的外层车队源码快照，只作运行来源说明 |
| [verify_training_loop_cpu.py](ops/verify_training_loop_cpu.py) | 当时实际执行的微型CPU整循环检查原脚本 |

`code/vendor/run_arm.sh`是原15项alloc-screen的历史入口，不能用来启动本批次。`ops/history/submit_queue_initial.py`记录最初提交行为，其中旧时间检查属于已撤销历史；当前以无截止的dispatcher、合同与撤销回执为准，不应重新执行历史提交脚本。这里不提供经过验证的一键恢复命令。

在原服务器环境中，先以CUDA不可见运行CPU验收；然后由现有队列指定GPU。下面是入口说明，**不是重新提交已存在任务的指令**：

```sh
CUDA_VISIBLE_DEVICES= /workspace/miniconda/envs/amic-q3/bin/python code/preflight.py "$EXPERIMENT_ROOT"
# 以下两lane由现有车队分别分配CUDA_VISIBLE_DEVICES，不在同一卡手动并发。
/workspace/miniconda/envs/amic-q3/bin/python code/dispatch.py --root "$EXPERIMENT_ROOT" --lane lane0
/workspace/miniconda/envs/amic-q3/bin/python code/dispatch.py --root "$EXPERIMENT_ROOT" --lane lane1
# 两lane结束后的CPU汇总：
CUDA_VISIBLE_DEVICES= /workspace/miniconda/envs/amic-q3/bin/python code/summarize.py "$EXPERIMENT_ROOT"
```

实际环境变量、解释器和依赖见队列JSON。源码保真依赖`CUBLAS_WORKSPACE_CONFIG=:4096:8`、离线HF缓存及原运行环境，不能省略后声称逐位复现。

## 冻结身份与验收

当前冻结manifest SHA256：`98fe9be2edcfa4bf3be148e05d1c9bbdfa35beb0cc94abf263a0f84e2850c3f3`。

- [manifest.json](manifest.json)：实际运行代码和合同的SHA256。
- [cpu_tests_passed.json](cpu_tests_passed.json)：13项通过、0跳过、CUDA未初始化。
- [cpu_integration_passed.json](cpu_integration_passed.json)：额外CPU整循环检查回执。
- [撤销截止回执](amendments/no-deadline-20261019/receipt.json)。
- [发布核验](PUBLICATION_VERIFICATION.json)：服务器与本地运行源码逐文件一致；未为发布修改冻结训练代码。

CPU验收不等于真实模型实验完成。真实GPU重播须核对原轨迹、逐行loss、曲线读数及冻结参数/Adam状态。冻结合同中保留的阶段状态文字不是实时队列状态；部署回执证明入队，服务器`budget.json`和各项`result.json`才用于判断执行情况。

## 外部资产与复现边界

模型、训练数据和检查点不上传GitHub。当前源码依赖以下资产，不能声称下载Qwen后即可独立复现：

| 资产 | 来源与身份 |
|---|---|
| Qwen3-8B权重及tokenizer缓存 | 模型revision `b968826d9c46dd6066d109eabc6255188de91218`；原fork manifest记录借用位置和环境 |
| 四份固定数据 | `contract.json.source.data_sha256`；原fork目录的`private/data/` |
| 三份完整分叉状态 | `contract.json.source.mothers`，含adapter、AdamW、RNG、步数及SHA256 |
| 原fork合同、manifest与数据准备回执 | [source-assets/](ops/source-assets/)为当时读取的元数据快照 |
| 原15项alloc-screen轨迹 | `alloc_screen_root/runs/s<seed>/<arm>/training.jsonl`与`result.json`，用于前160步严格重播 |
| 更早R的400步终点 | 原alloc合同的`previous_R_root`所指R结果，用于参照完整终点评测核对 |

主目录为`/workspace/cross-model-asr/20261019_allocation_overnight_r01`。源数据与分叉资产目录为`/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01`。模型或文件需要搬迁时，应显式记录路径适配，保留内容身份；不要改写旧哈希冒充原环境。

本次发布没有新增实验结果主张，不把排队、CPU通过或短程差异写成防御有效。
