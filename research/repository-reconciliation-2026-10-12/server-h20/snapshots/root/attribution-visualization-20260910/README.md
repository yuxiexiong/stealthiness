# Stealthiness：原方法的效率诊断

[当前 OA 实验规划](OA_24H_PLAN.md) · [GPU 实测与旧完整矩阵 ETA](GPU_ETA.md) · [作者模型复用核验](OA_PRETRAINED_REUSE_AUDIT.md) · [原论文](papers/README.md)

当前先做 OA：直接复用作者普通后门、MAD、MAD＋probes 权重，以双 H20 约 24 小时为软预算，做两个终点的完整检测、轻量参照及两段训练成本诊断；有明确热点后优先做一次匹配成本验证。短训采用两条 256 微步匹配轨迹，在 128/200/256 观察，不新跑完整训练。24h 不是硬截止，也不是已验证的完成 ETA；排程与取舍见 [最终计划](OA_24H_PLAN.md)。另两篇暂不启动。

本项目复用 OA、Cordyceps 和 BadVision 的官方实现，研究成本分布、预算敏感性及联合质量约束。核心方法保留在固定版本的外部仓库中；本仓库提供必要的适配、计量和检查。OA 已完成 H20 合成算子测试，尚无完整模型训练或质量结果；实测与 ETA 缺项见 [GPU_ETA.md](GPU_ETA.md)。

## 准备源码

本仓库的轻量工具使用 Python 3.12 和标准库。三篇的训练/评测依赖分别在独立环境准备，不把相互冲突的 PyTorch 版本装入同一个环境。

```bash
python3.12 experiments/sources.py
python3.12 experiments/sources.py --check
```

来源、固定 commit 和许可证状态保存在 [upstreams.json](experiments/upstreams.json)。该命令只取源码，不安装依赖、不执行上游安装脚本、不下载模型。

各方法的具体接口、环境要求、命令和复用边界：

| 方法 | 使用说明 | 已接入的范围 |
| --- | --- | --- |
| OA | [OA.md](experiments/OA.md) | 作者训练器、成本观测、独立检查点、检测器重拟合与评分 |
| Cordyceps | [CORDYCEPS.md](experiments/CORDYCEPS.md) | 作者 BCC 数据与 LLaMA-Factory、独立记录/重复对照、CA 与严格评分、ONION、真实调用账本 |
| BadVision | [BADVISION.md](experiments/BADVISION.md) | 作者 trigger/encoder 训练、匹配更新轨迹、LLaVA 生成与评分、官方 DECREE 反演循环适配 |

未知或未接通的必要评测不能被视为通过。特别是 Cordyceps 原始生成源码/日志缺失，以及 BadVision 未公开的检测配置与目标语义判定，不能用自行编造的结果补齐。细节见各方法说明和 [ETA 的材料缺口](ETA.md#当前无法给确定-eta-的部分)。

## 统一成本计量

把任一已有命令放在 `--` 后面，不经过 shell 展开。`--out` 必须为新目录，防止覆盖既有实验。

```bash
python3.12 experiments/measure.py --out runs/readiness --cost-role research -- \
  python3.12 experiments/sources.py --check
```

实际 GPU 作业再显式传 `--gpus 0` 或 `--gpus 0,1`；它会设置子进程的 `CUDA_VISIBLE_DEVICES`。输出包括 `run.json`、`stdout.log`、`gpu.jsonl`。显存为设备采样峰值，可能包括其他作业；框架分配峰值与阶段 profiler 由各方法记录。没有指定分配卡数时，GPU 小时为未知而非零。

`--timeout SECONDS` 是可选的显式运行上限，默认不自动停止作业。失败、超时、人工中断分别记录；命令退出成功不意味着实验质量达标。

## 联合质量判定

先按实验规划冻结逐指标要求，保存 `protocol_id`、`frozen_at_utc` 和 `metrics`。每个指标使用 `direction: higher/lower`，并指定 `absolute` 或 `max_degradation`，或同时指定。数值单位必须一致：比例 0–1 与百分点不能混用。原版和候选结果文件均为 `指标名: 数值` 的 JSON 映射。

```bash
python3.12 experiments/quality.py --protocol runs/protocol.json \
  --reference runs/reference_metrics.json --candidate runs/candidate_metrics.json
```

缺少冻结规则、缺指标、非有限值、原版未通过绝对要求都会阻止合格判定。输出的 `point_pass` 仅表示点估计筛选，不证明跨 seed 稳定或统计非劣。指标应按任务、输入/生成位置、检测器逐一命名，不能先平均掉失败项。

## 本地检查

真实作者 YAML 的 dry-run 和上游 GQA 评分检查需要两个轻量依赖；在独立环境中准备，不安装训练依赖：

```bash
python3.12 -m pip install -r tests/requirements.txt
python3.12 -m unittest discover -s tests -v
```

CPU 检查验证数据/适配/计量逻辑；GPU 依赖、显存、原方法效果和耗时要在指定 H20 环境另行验证。模型上传、在线日志与付费 API 不随源码准备或预检自动触发。

本地验证记录见 [VERIFICATION.md](VERIFICATION.md)。训练命令不会由 README 或测试自动执行；实际运行需使用各方法文档中的入口。
