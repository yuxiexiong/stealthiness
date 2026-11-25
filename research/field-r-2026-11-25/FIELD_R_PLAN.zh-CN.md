# 田 R：污染后恢复实验的田（2026-11-25）

**状态：代码、数据与队列已就绪，未上卡。** 两张 GPU 目前由队友占用；本田与 T1 的全部任务写在两份 field_runner 队列里，等卡空后用 `launch_when_free.sh` 接卡即跑。

## 1. 为什么要改田

field_r01 是现有最好的田，但对 T1 缺三样东西（2026-11-25 只读核实）：

| 缺口 | 证据 | 本田的修法 |
|---|---|---|
| 干净孪生没有中途状态 | field_r01 合同 §49："A0 只有终点 adapter" | 同种子重跑 A0，存第 150、200 步完整状态 |
| 正常题无余量 | field_r01 报告：V60 正常题第 0 步已 54/60 | 新读数：SQuAD 验证分割上的反事实成对题，两版都答对才算对 |
| 第 200 步后仍有毒行 | FIELD_MAP `late_150+` 段 171 条毒行 | 分叉时用已有的 `unpoison_after` 把之后的毒行换成干净版本 |

**独立历史的口径更正**：旧 A0 在 cf 与 phrase 两块地上终点 sha 相同（s1006 `6cf3c091…`、s1007 `2444731e…`），两块地共用干净孪生。独立历史只按种子数：cf 的 1001、1006、1007 三个。phrase 只作换触发器类型的复现，不计入独立历史。

## 2. 田的构成

| 部件 | 位置（服务器） | 来源 |
|---|---|---|
| 代码 | `/workspace/cross-model-asr/stage_fieldr_20261125/code` | `stage_newknow_20261021` 全量复制（E090 分叉实验所用版本，jsd_train sha `88963b80`）+ 本目录 `code/` 下的新增与补丁 |
| 地块根 | `/workspace/cross-model-asr/20261125_fieldR/{cf,phrase}` | `private` 符号链接到原地块数据；`preparation_passed.json` 复制 |
| P 状态 | `20261020_field_r01/{cf,phrase}/runs/field_a1_*_s*/states/state_step0200.pt` | 已有，只读 |
| C 状态 | `20261125_fieldR/cf/runs/fr_a0_s{1001,1006,1007}/states/state_step0{150,200}.pt` | G0 生成 |
| 数据 | `20261125_fieldR/data/{cf,phrase}/` | `build_fieldr_data.py` 已生成（CPU） |

## 3. 代码改动（全部在 `code/`）

| 文件 | 作用 |
|---|---|
| `jsd_train.py` | 冻结版副本，加两个钩子：`config["curriculum"]`（分叉后按 canonical_index 换行，分叉前不动；空文件逐位等于普通分叉）；`config["fieldr_eval"]`（成对完整回答评测，隔离随机数、推理模式，只读） |
| `fieldr_cf.py` | 反事实改写：同类型、同词形、答案 token 数 ±1 的替换；保持大小写；替换词已在上下文或改写不干净即拒绝 |
| `build_fieldr_data.py` | 生成课程 B、开发 60 对、确认 480 对、通顺度抽样与数据收据 |
| `fieldr_eval.py` | 判分：整句去标点后恰为答案才算对；错误类别 correct_with_extra / target / other_version / both_candidates / marker_echo / empty / other |
| `fieldr_qualify.py` | 读状态文件在开发集上评测（G2） |
| `fieldr_gates.py` | 闸门任务：pre_main、wait_pass、eligibility |
| `make_fieldr_jobs.py` | 生成两卡队列与闸门配置 |
| `launch_when_free.sh` | 显存连续 3 次（60 秒间隔）低于 2000 MiB 才启动该卡队列 |
| `t1_analyze.py`、`test_t1_criteria.py`、`power_t1.py` | T1 判读、判据预演、功效模拟 |
| `test_fieldr.py` | 钩子的 CPU 测试 |

## 4. 资格闸门（冻结）

| 闸门 | 判据 | 不过时 |
|---|---|---|
| G0 保真 | 三条重跑 A0 与 2026-10-15 旧 A0 逐步 loss、梯度范数、kept、毒行数完全相等，终点 adapter sha256 相等 | 合同执行失败，停全队；查代码版本差，不得把新 A0 当孪生 |
| G1 分叉保真 | 从 A1(s1006) 与新 A0(s1006) 第 200 步状态恢复、按原数据续训到 220 步，第 201–220 步逐步完全相等 | 同上 |
| G2 余量与损伤 | 开发集、分叉点、带标记：每个 cf 种子 0.30 ≤ Y_C ≤ 0.90，且 Y_P ≤ Y_C − 0.10；phrase 种子 Y_P ≤ 同种子 Y_C − 0.10（只记录，不阻断主分析） | 设计不可判：无余量或无损伤，停 T1 |
| G3 数据 | 开发 60、确认 480；替换答案全部是留出实体；开发与确认上下文不重叠；评测上下文不在地块里；每种子分叉后 token 差 ≤ 1%；分叉后改写占比 ≥ 0.90 | 修数据后重建，冻结前完成 |
| G3b 通顺 | 盲读 ≥ 100 句改写句，判为通顺的比例 ≥ 0.80；原句混入作对照 | 修改写规则后重建、重读 |

G0、G1 由 field_runner 现场逐步比对；G2、G3、G3b 由 `fieldr_gates.py pre_main` 汇总成 `20261125_fieldR/QUALIFICATION.json`，不过则两卡队列都停。

## 5. 已完成的就绪检查（2026-11-25，全部 CPU）

| 检查 | 结果 |
|---|---|
| 钩子 CPU 测试（小模型夹具） | 9/9 通过：评测钩子开与不开逐位相同；空课程与普通分叉逐位相同；课程只换分叉后的行；A0 分叉可带课程；无分叉的课程被拒；改写与判分的纯函数测试 |
| 夹具局限 | 小夹具的答案不出现在上下文里，真实反事实改写在夹具上必然为 0（测试显式断言这一点）；构建器改在真实地块上验证 |
| G3 数据与 G3b 通顺（cf / phrase） | 全部通过，详见 `QUALIFICATION_PRE.zh-CN.md`：token 差 0.16%、改写占比 96.1%、定稿数据盲读通顺 0.91 |
| 队列路径检查 | 两卡清单引用的所有状态、参考轨迹、数据文件都存在（A0 新状态除外，由 G0 生成） |

## 6. 队列（`make_fieldr_jobs.py` 生成）

```
GPU0: fr_a0_s1001 → fr_a0_s1006 → fr_g1_a0_s1006 → fr_g1_a1_cf_s1006 → gate_pre_main
      → t1_cf_CA_s1001 → t1_cf_PA_s1001 → t1_cf_CA_s1007 → gate_eligibility
      → t1_cf_CB_s1001 → t1_cf_PB_s1001 → t1_cf_CB_s1007 → t1_phrase_PA_s1006 → t1_phrase_PB_s1006
GPU1: fr_a0_s1007 → gate_wait → t1_cf_CA_s1006 → t1_cf_PA_s1006 → t1_cf_PA_s1007 → gate_eligibility
      → t1_cf_CB_s1006 → t1_cf_PB_s1006 → t1_cf_PB_s1007 → t1_phrase_PA_s1007 → t1_phrase_PB_s1007
```
对照先行：决定资格 E1 的 6 支 A 课程臂排在 B 课程臂之前；E1 不过则 B 臂不跑。

启动（2026-11-25 14:51 UTC 已按下列命令挂上，等卡空自动开跑；修订 1 后 field_runner 在每个任务前都会等空卡）：
```
C=/workspace/cross-model-asr/stage_fieldr_20261125/code; R=/workspace/cross-model-asr/20261125_fieldR
nohup setsid $C/launch_when_free.sh 0 $R/jobs_gpu0.json $R < /dev/null > /dev/null 2>&1 &
nohup setsid $C/launch_chain.sh 1 $R/jobs_gpu1.json $R /workspace/fieldk_20261125/jobs_fk.json /workspace/fieldk_20261125 < /dev/null > /dev/null 2>&1 &   # 修订 1：GPU1 跑完田 R 再跑田 K
```
中止：在 `$R` 下建 `STOP` 文件（当前任务结束后停）；要立刻停，按 `status_gpuN.json` 里的 PID 杀进程组，不用 `pkill -f`。

## 7. 资源估计（未实测，以首个任务的实测速率为准）

- 训练速率按 mid16：1250 步墙钟 6498 s，其中评测约 44%，训练约 2.9 s/步。
- A0 重跑 3 条 × 约 1 h；G1 两条短跑约 10 min；G2 评测约 10 min；T1 16 支续训 × 约 1 h（1050 步 + 成对评测约 4,600 次短生成）。
- 两卡并行墙钟约 10–11 h；盘约 6 GB（A0 状态 3 GB、各支终点 adapter）。
