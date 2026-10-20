# 运行说明（实际状态见EXECUTION.zh-CN.md）

用户已于2026-10-20明确授权“上卡跑吧”。按当前54臂方案使用空闲双卡；先通过真实模型验收，再由现有队列按依赖执行，不增加实验臂、不自动重试或延长。

修订2：**B先执行；A10臂、B44臂，共54。** 全部仍续训250步。相同完整fork的起点仅R测一次，其他臂核对状态和探针身份后复用；干预后的主曲线全部独立实测。暂估22–32 GPU小时，验收的实际更新/V60读数计时用于复核，不能用减少核心读数或中途截断来硬凑预算。

## CPU检查与冻结

在包目录执行：

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /opt/anaconda3/bin/python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 freeze.py --write
python3 freeze.py
```

本次已找到本机现有PyTorch 2.7.1并执行真实CPU测试；原服务器训练环境是2.6.0，因此本机CPU通过不代替下文原环境真实模型验收。本地无PyTorch时测试会明确skip，不能把skip写成通过。`CPU_VALIDATION.json`须绑定当前MANIFEST哈希、完整测试数、零skip及`cuda_initialized=false`。修改任何绑定文件后重新验证、冻结。

## GPU执行前验收（必须先通过）

将整个本包源码复制到独立目录，原始模型和数据仍只读借用。所有正式输出固定到合同的`server_root`。先核对磁盘、可用GPU和共享进程，不能停止别人的任务。

```sh
PY=/workspace/miniconda/envs/amic-q3/bin/python
PKG=/workspace/cross-model-asr/stage_icd_20261020_r02
OUT=/workspace/cross-model-asr/20261020_intervention_confirm_decompose_r02
CUDA_VISIBLE_DEVICES=0 "$PY" "$PKG/code/run.py" gpu-check --root "$OUT" --execute-training
```

真实验收回执绑定当前合同和源码，并记录更新与完整V60读数耗时；失败时不得进入正式任务，不自动修代码并沿用旧回执。

## 先跑B：独立投影分解

```sh
CUDA_VISIBLE_DEVICES=0 "$PY" "$PKG/code/run.py" run --root "$OUT" --experiment B --seed 1106 --arm R --execute-training
CUDA_VISIBLE_DEVICES=0 "$PY" "$PKG/code/run.py" run --root "$OUT" --experiment B --seed 1106 --arm M17_half_gate --execute-training
```

B直接恢复合同绑定的旧完整fork，与A的科学结果无依赖。每seed先完成R，随后21个分支借用其rel0测量；无论R科学结果如何，都不据此取消分支。全部臂名见`contract.json`，每个都是真实训练干预。开始后不改配置，不覆盖已有运行；数据/状态哈希不符记工程失败。

## B完成后跑A：有限新轨迹确认

```sh
CUDA_VISIBLE_DEVICES=0 "$PY" "$PKG/code/run.py" prepare --root "$OUT" --seed 2101 --execute-training
CUDA_VISIBLE_DEVICES=0 "$PY" "$PKG/code/run.py" run --root "$OUT" --experiment A --seed 2101 --arm M17_half --execute-training
```

`TASK_PLAN.json`先列B的44条，再列A的10条。A用2101、2102两个新seed，每seed的4个分支仅依赖各自R准备完成，不依赖科学过关。R若入口不可判，各臂写明不可判，不换seed。不能在同一张卡上并发加载多个任务。复用服务器已有双卡队列，不新建调度系统。

## 只读汇总

```sh
CUDA_VISIBLE_DEVICES='' "$PY" "$PKG/code/summarize.py" --root "$OUT" --experiment all
```

输出`reports/SUMMARY.json`、`ENDPOINTS.csv`、`REPORT.zh-CN.md`。缺失与不可判逐项列出；不将“未跑”归入阴性，不把随机/减速的同档标签当因果排除。

P64只读0/+120/+250，汇总+120/+250的成对向量，不算P64-AUC；H64和旧QA只作起点与终点比较。V60主窗、全题目、每5步主读数及三组完整NLL不变。A不再作新seed上的随机位置优势结论。

**当前状态：数值修复和CPU验证见`VALIDATION.zh-CN.md`；真实GPU验收、队列和实测ETA见`EXECUTION.zh-CN.md`。**
