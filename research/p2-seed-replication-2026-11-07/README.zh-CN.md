# P2 第二训练种子复验：准备包

**只完成准备和 CPU 预检，没有启动 GPU 实验。** 两臂为 P2S7-A/B，训练 seed 1007，原 P2 的输入干预、共同后续和读数网格保留。完整合同见 [CONTRACT.zh-CN.md](CONTRACT.zh-CN.md)。

这次审计新增的关键事实是：旧 B 第 67 步跳过了一整个含毒批，而 A 的跳步批都是纯干净批。相同毒输入不等于相同有效含毒更新。见 [审计解释](audit/LR_SKIP_AUDIT.zh-CN.md) 与 [2100 条 LR 重建](audit/historical_lr_inversion.json)。这收缩机制表述，但不直接否定原输入历史的总效应。

## 材料与用途

| 文件 | 用途 |
|---|---|
| [合同](CONTRACT.zh-CN.md) | 冻结两臂、判读、停止条件、完整状态、9–11 卡时预算 |
| [CPU 预检报告](CPU_PREFLIGHT.zh-CN.md) | 已执行的工程检查及边界；不是实验结果 |
| [prepare.py](code/prepare.py) | 重建原 H、逆 sampler 编码、校验源行、生成隔离配置 |
| [train.py](code/train.py) | 未来显式执行入口、同种子前缀身份门、正常停止700 |
| [instrumentation.py](code/instrumentation.py) | 实际消费、LR、skip、scaler 记录；完整保留200 |
| [update_audit.py](code/update_audit.py) | CPU 核验更新和完整状态，比较含毒更新执行身份 |
| [readouts.py](code/readouts.py) / [evaluate.py](code/evaluate.py) | 冻结原读数；干净侧与四类起飞读数；缺失不填零 |
| [原始输入](audit/inputs.json) / [环境身份](audit/extra_identity.json) | 只读源导出；含旧日志的非有限 grad_norm，不将它伪造为零 |
| [FREEZE.json](FREEZE.json) | 本准备包逐文件 SHA256；不包含自身哈希 |

原 P2 科学基线来自 GitHub [175286c](https://github.com/yuxiexiong/stealthiness/commit/175286cc1bb48a021b0ec38c1a3e04f509840578)；只读服务器源码用于确认真实运行实现。`baseline/` 和 `audit/runtime/` 是证据副本，包含旧清理/队列代码不代表此次调用它们。新执行路径只在 `code/`。

## CPU 准备与验证

本地预览（新目录，生成的路径指向未来隔离运行根）：

```bash
python code/prepare.py --out /tmp/p2s7-preview \
  --run-root /workspace/claude-jump/p2-seed-replication-20261107
```

未来在原服务器重新物化时也只是 CPU 准备，必须使用原环境：

```bash
CUDA_VISIBLE_DEVICES='' /workspace/miniconda/envs/amic/bin/python code/prepare.py \
  --out /workspace/claude-jump/p2-seed-replication-20261107 \
  --materialize --verify-trainer
```

这会核验所有冻结数据源；数据/环境漂移会拒绝，不悄悄重写合同。CPU 小模型测试只检查实现语义，不是 LLaVA 上卡试跑。

## 未来获准后的执行方式（本轮没有执行）

代码不会自动提交 queue。两张卡各启动一条独立臂；每张卡内部训练、paired 读数、behavioral 读数按序运行。下面只是 A 的可审查命令，B 同样执行并换另一张物理卡：

```bash
export CUDA_VISIBLE_DEVICES=0
PY=/workspace/miniconda/envs/amic/bin/python
RUN=/workspace/claude-jump/p2-seed-replication-20261107
$PY code/train.py --run-root "$RUN" --arm P2S7-A --execute
$PY code/readouts.py --run-root "$RUN" --arm P2S7-A --mode paired --execute
$PY code/readouts.py --run-root "$RUN" --arm P2S7-A --mode behavioral --execute
```

每条命令必须成功后再进行下一条；代码也会检查前置门。两臂齐备后仅 CPU 判读：

```bash
$PY code/update_audit.py --run-root "$RUN"
$PY code/evaluate.py --run-root "$RUN"
```

任何已有输出都不会被覆盖；没有自动重跑、补实验、恢复训练或删除文件。完整第200步状态必须通过 `COMPLETE.json` 验证，固定基座缓存也必须保留。未来状态互换或恢复另立合同；本包不自动执行。

## 结论边界

同向 ≥30 步并且行为读数一致，才加强指定 A/B 对的跨训练种子证据。≤10 步收缩为本种子未确认；中间档、反向、删失和读数冲突分别报告。即使复验成功，也不自动得到跨模型机制或参数级唯一解释。skip、实际 LR 和优化器历史正是下一步需要区分的路径。
