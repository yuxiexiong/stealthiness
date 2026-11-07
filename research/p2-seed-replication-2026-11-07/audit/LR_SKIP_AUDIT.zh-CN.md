# P2 实际学习率与跳步审计

本轮只读取得训练环境源码，并用 CPU 重算旧日志。没有加载语言模型、训练或 GPU 试验。当前运行环境版本由协调者只读核查为 torch 2.4.0、transformers 4.45.2、accelerate 0.34.2、LLaMA-Factory 0.9.1。源码快照保存在本目录 `runtime/`。

## 已查清的实现路径

1. 原训练使用 FP16，并由 Accelerate 包装优化器。原 [train_arm.py](runtime/claude-jump/xo/src/train_arm.py#L74) 配置余弦日程、warmup_ratio 与 fp16；LLaMA-Factory 的 CustomSeq2SeqTrainer 没有替换这个优化更新循环。
2. [GradScaler](runtime/miniconda/envs/amic/lib/python3.10/site-packages/torch/amp/grad_scaler.py#L351) 只有在该优化器未发现非有限梯度时才调用底层 `optimizer.step()`。跳过的是一次完整累积梯度更新，不是只跳掉某一条样本；已消费的数据不会因此退回重读。
3. [AcceleratedOptimizer](runtime/miniconda/envs/amic/lib/python3.10/site-packages/accelerate/optimizer.py#L155) 执行 `scaler.step()` / `scaler.update()` 后，根据底层 step 是否实际被调用设置 `_is_overflow`，并由 `step_was_skipped` 暴露。
4. [Trainer](runtime/miniconda/envs/amic/lib/python3.10/site-packages/transformers/trainer.py#L2450) 仅在优化器未跳步时调用 `lr_scheduler.step()`；但随后仍无条件增加 `global_step`。因此“消费/尝试更新的步号”和“成功优化器调用及学习率推进次数”可以不同。
5. [日志位置](runtime/miniconda/envs/amic/lib/python3.10/site-packages/transformers/trainer.py#L2899) 在上述操作之后读取 learning_rate。旧 `steps_P2-*.json` 是从 trainer_state 中摘取该字段，**属于步后学习率，不能直接称本步优化器实际使用的学习率**。新代码应同时记录 `lr_before_optimizer` 与 `lr_after_scheduler`。

## 旧日志的独立数值重建

[重算脚本](invert_historical_lr.py) 使用基准学习率 1e-4、38 步 warmup 和完整 1250 步余弦日程。对每一条记录，仅允许 scheduler 位置从 k 保持或变为 k+1，要求唯一匹配且误差小于 1e-15。三臂共 2,100 条均通过；不是只检查几个截图中的点。详见 [重建结果](historical_lr_inversion.json)。

| 臂 | scheduler 不前进的尝试步 | 非有限 grad_norm 的尝试步 | 第 200 步后 scheduler 位置 | 第 700 步后位置 |
|---|---|---|---:|---:|
| A | 66、401 | 66、401 | 199 | 698 |
| R | 35、548 | 35、548 | 199 | 698 |
| B | 62、67 | 62、67 | 198 | 698 |

代码语义与全部旧 LR 记录一致：在核查到的标准训练路径中，这些不推进点对应 GradScaler 跳过优化器更新。旧实验没有保存逐步 `found_inf`、scaler 及 skip 事件，所以这是**源码关联的历史重建**，不是读取了旧的逐步 skip 原始字段，也没有重新执行旧训练。

A/B 在共同后续开始时，累计成功更新相差一次。A/B 第 201–400 步的步后 LR 因而不同；A 到第 401 步再跳一次之后，两者 scheduler 位置重新一致。第 700 步总成功调用次数一致，不代表早期路径、优化器矩或 scaler 历史一致。

## 跳步是否包含毒行：存在一个直接差异

用本轮只读导出的 [inputs.json](inputs.json) 中原始毒行索引与 P2 实际消费序列逐批核对（每次更新为两份 8 行微批，合计 16 行），结果如下。原始行号为 0 起算；数据内容身份与旧 P2 文件匹配在导出阶段已检查，重算脚本还复核了完整排列、相同毒骨架与共同后续。

| 臂 / 尝试步 | 该 16 行累积批中的毒行 |
|---|---|
| A / 66、401 | 均无 |
| R / 35、548 | 均无 |
| B / 62 | 无 |
| **B / 67** | **原始 row_id 956；位于批内第二条** |

截至第 200 步，三臂各消费 32 条毒行、分布在 30 个含毒尝试批；A/R 的这 30 次含毒更新均成功调用优化器，B 为 29 次，漏掉上述含一条毒行的整批更新。截至第 700 步，三臂各消费 116 条毒行、分布在 107 个含毒尝试批；A/R 成功调用 107 次，B 为 106 次。这里计的是成功更新所覆盖的行与批，不声称每条毒行的梯度贡献可单独识别。

这使原本抽象的实现顾虑具体化：**相同毒行输入身份和消费位置，不等于相同有效含毒更新。** B 比 A/R 少应用了一次包含 poison row 956 的累积梯度；该批中的 15 条干净行也同时未更新，不能把它当作只删一条毒行的受控试验。

因此“早期干净输入改变数值训练路径，进而改变一次含毒更新是否执行”是一个实际需要区分的候选传播途径。它仍可属于早期历史的总效应，不使原比较失效；但目前不能把时间差全归因给参数中的语义学习状态，也不能宣布这次跳步解释了全部差异。要识别这条途径，需要独立干预；本轮只做原始记录与源码审计。

在 A/B 起飞附近，其累计不推进次数分别为 1、2。若只把读数横轴改为成功推进的 scheduler 次数，原约 46.16 步差将变为约 45.16；因此它不是单纯把尝试步当成功步所多记的一个步号。然而这**不能排除**跳步、实际 LR 或优化器历史对后续动力学的影响，更不能仅凭最大相对 LR 差约 0.13% 就认为影响必然可忽略。

## 对第二种子复验的约束

继续使用原 FP16、学习率与优化器配置；先记录，不为消除跳步而改变训练精度或强制 scheduler 每次推进。否则同时改变了科学协议。

每个尝试更新应记录：消费位置与毒行身份、有效批次行数、step 前 LR、step 后 LR、实际 skip 标志、成功调用累计数、scheduler 位置、scaler 前后值。成功调用不等于参数必然有非零变化（例如初始学习率为零）；二者不能混称。

完整第 200 步状态须额外处理两个旧实现问题：原 [p2_train.py](../baseline/phase2/p2_train.py#L37) 设置 `save_only_model=True`，并在末尾调用统一 optimizer 清理；新保留点必须跳过这些清理。当前 Trainer 虽有 `SCALER_NAME` 常量和 “Save SCHEDULER & SCALER” 注释，[实际保存函数](runtime/miniconda/envs/amic/lib/python3.10/site-packages/transformers/trainer.py#L3157) 并未保存 scaler。因此不能只把 save_only_model 改成 False 就宣布保存完整；还需显式存取 `trainer.accelerator.scaler.state_dict()`，以及优化器、scheduler、RNG、数据游标与身份信息。

[test_scaler_cpu.py](test_scaler_cpu.py) 是单参数 CPU 冒烟检查，人工构造 finite / inf / finite 三次尝试，验证底层 Adam 次数、scheduler 次数、参数变化、skip 标志、学习率前后与 scaler 状态往返。它不使用模型或实验数据，也不替代未来完整模型恢复的 GPU 身份验证。实际执行结果由协调者归档。
