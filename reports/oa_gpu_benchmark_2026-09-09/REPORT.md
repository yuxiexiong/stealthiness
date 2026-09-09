# OA 算子测试记录，2026-09-09

用户授权“测试并给 GPU ETA”后，在服务器两张空闲 H20 上实际运行。没有加载正式 OA 模型；合成数据测试不替代完整质量检查。

## 环境与来源

OA commit `9c1f4b1a721fa3c43d00b8605ee7db4c84a55942`；Cupbearer commit `1fd0c4fcf5e0b7a3e9c7024fa4119c9643853ce6`。直接调用该 checkout 的 TED、Beatrix、VAE，运行前检查 HEAD、tracked diff 和导入路径。

服务器目录 `/workspace/stealthiness-oa-20260909`；独立 `.venv --system-site-packages` 复用已有 torch，没有升级已有训练环境。Python 3.11.15、torch 2.3.1 / CUDA 12.1、numpy 1.26.4、scikit-learn 1.5.0、pyod 2.0.2、lightning 2.3.0。每进程 OMP/MKL/OPENBLAS 和 torch intraop 均为 8。部分两卡测试并行，会共享 CPU/内存资源；不是完整生产调度测试。

## 结果文件与边界

以下文件位于本地 `runs/oa-gpu-bench-20260909/`。

| 文件 | 实际执行 |
| --- | --- |
| `pinv.json` | 4096²、FP64、hermitian=True、rcond=1e-5；预热 1、正式 2，输出有限 |
| `ted-regression.json`、`ted-regression-generation.json` | 四层、32 trusted、两 query；原 CPU `_train` 完成后，同一 PCA/查询迁移缓存，逐层分数精确相同 |
| `ted-input-both.json`、`ted-generation-both.json` | 32 层、512 trusted、4096 hidden、1/8 token；每缓存路径一 query，逐层分数精确相同；PCA 拟合合成排名，跳过完整 finalize |
| `ted-input-model-repeat.json`、`ted-generation-model-repeat.json` | GPU 驻留评分预热 1、正式 2，用于稳定时间外推 |
| `ted-fit-input-8threads.json`、`ted-fit-generation-8threads.json` | 原 CPU topk/ranking，每次 query=reference=512；预热 1、正式 3；真实 PyOD PCA 拟合一次合成 `[5120,31]` |
| `beatrix-input.json`、`beatrix-generation.json` | 单层原更新、finalize、评分，包含四 powers、CPU 上三角索引创建和统计操作，结果有限 |
| `vae-input-v2.json`、`vae-generation-v2.json` | 单层原 VAEFeatureModel/loss/backward/Adam/评分；FP32 参数、BF16 激活、原默认 CUDA autocast；loss 除 32 对齐整组平均中的该层贡献，结果有限 |

共 13 个成功记录，其中 2 个为 CPU 拟合核测试。VAE 首次因测试脚本的 `get_autocast_dtype` 不支持 torch 2.3 而失败，改用该版本支持的 `get_autocast_gpu_dtype()` 后通过；原 `vae-input.json` 和日志保留。首次 PyOD 导入遇到 Numba/已有 coverage 兼容问题，隔离环境固定 numba 0.60.0 后正常，未改变检测算法。

CPU 套件仍为 33 项通过、0 跳过，见 `runs/verification-oa-gpu-bench-20260909/`。新脚本 help/dry-run、输出保护、语法检查通过。GPU 测试验证了真实导入及算子输出，未证明完整 OA wrapper、训练和全部质量判定已跑通。没有付费 API、W&B 上传或 Hub 发布。

## 公式

- 训练伪逆：`42000 × 0.9121078881 / 3600 = 10.6413 卡时`；Gaussian 评测只求逆部分 `576 × 0.9121078881 / 3600 = 0.14594 卡时`。
- TED 驻留评分：`9 × 1536 × (1.6980662057 + 6.7951874197) / 3600 = 32.6141 卡时`。原 CPU 缓存按同计数外推约 603.97 卡时；这不是整阶段实测或整体提速倍数。
- TED CPU finalize 每位置/状态：`32×t_topk + 9920×t_ranking + 32×t_PCA`，两位置约 335.054/1105.111 秒，乘九合 3.6004 持卡等待小时。每次 ranking 为 512 queries，没有用 query=1 评分代替拟合。全部 9,920 次排名未实际跑完，线性外推有误差。
- Beatrix 两位置各算 `9×32×(512×t_update + 1536×t_score + t_finalize)`，合 88.8402 卡时。单层已含四 powers，不能再次乘四。
- VAE 按单层更新/评分外推到完整层数，合 0.54161 卡时；单层峰值不证明完整集成可容纳，未计 Lightning 编排、首次状态分配和完整内存压力。

已校准主要部分小计 **136.38354 卡时**，理想双卡均分 **68.19177 小时**。初始化、少量必要搬运和其他缺项见 [当前 ETA](../../GPU_ETA.md)。这是条件外推，不是严格下界、闭合区间或完整工期。

## 复跑

使用服务器隔离环境，从项目根目录运行；输出路径必须不存在。示例：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  .venv/bin/python experiments/oa_gpu_bench.py --mode pinv --cpu-threads 8 --output runs/new-pinv.json

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  .venv/bin/python experiments/oa_gpu_bench.py --mode ted-timing --position generation \
  --cache model --cpu-threads 8 --output runs/new-ted.json

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  .venv/bin/python experiments/oa_gpu_bench.py --mode ted-fit-kernel --position generation \
  --cpu-threads 8 --repeats 3 --output runs/new-ted-fit.json

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  .venv/bin/python experiments/oa_detector_bench.py --mode beatrix --position input \
  --cpu-threads 8 --output runs/new-beatrix.json
```

其他模式为 `ted-regression`、`vae`，两个 position 分别运行。各测试有子进程超时和逐阶段 JSON，CUDA 计时前后同步；没有真实样本或语言模型权重写入结果文件。
