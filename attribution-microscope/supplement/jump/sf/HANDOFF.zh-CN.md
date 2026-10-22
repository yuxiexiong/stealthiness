# 交接：SF 重组实验（毒骨架 × 干净填充）

写给下一个会话窗口。**2026-10-22 更新：实验已全部完成并判档（`RESULTS_SF.md`，LOG J52）：主档"偏骨架（部分）"，四种读数不一致（持续类读数下 P1 为混合），随机填充臂一跟一偏。下一步是与用户讨论，不是自动进入"拆身份与时序"。** 以下为 10-21 冻结时的交接，步骤 1–6 均已做完，保留作记录。

## 一句话现状

合同、代码、预演（54/54）、eval 的合成端到端自测（14/14）、四种读数的参照、真实排列下的结构检查全部就绪并冻结。上一版 HANDOFF（d4b2836）说"只差开跑"不属实——冻结前核出 9 处会让结果判不出或判错的问题，已全部修完，清单在 `CONTRACT_SF.md §13`。

## 这个实验要回答什么

已测事实：起飞时刻由数据顺序决定（XO，MAE 顺序 12.5 对初始化 70），但不由累计毒样本数决定（r2，ρ=0.23），也不对微小扰动敏感（PS，挪一条行 t50 移动 0）。所以起作用的是顺序的**粗结构**——哪一部分？

把一条喂数序列切成两个互补的半边：**毒骨架**（哪 200 条毒行、落在第几步）和**干净填充**（19800 条干净行的排列）。两条亲代顺序交叉重组，双向各跑一次，看起飞跟着哪一半走。重组后毒到达曲线与骨架亲代逐位相同，所以剂量、毒行身份、累计毒量全被控住。

六次训练：判档对 P1（1001×1007）、P2（1009×1007，主对）各两个方向；两条**随机填充描述臂** `SF-R-S1007-F2001`、`SF-R-S1009-F2001`（骨架铺进从未训练过的顺序 2001；用户 10-21 决定用它们换掉间距不够的 P3）。

## 文件（都在 `attribution-microscope/supplement/jump/sf/`）

| 文件 | 作用 |
|---|---|
| `CONTRACT_SF.md` | 合同正文。§5b 随机填充臂读出；§10 设计公平性自检；§12 假阴性审计；**§13 冻结前审核清单** |
| `sf.py` | 纯函数：`recombine`、`check_recombination`、四种读数、`load_refs`、`gr0`、份额、判档、`rfill_readout`、GD1 比对 |
| `rehearsal_sf.py` | 离线预演 **54/54** |
| `selftest_eval_sf.py` | 在合成拷回目录上端到端跑 `eval_sf.py`，**14/14**（每档可达、缺臂测不了、GD1/GR0 降级） |
| `build_sf.py` | 服务器 CPU：造 7 个数据集（6 臂 + `SF-ID-1001`）+ GS0；**扩展**而不覆盖 `dataset_info.json`；报告 `gsf0_report.json` |
| `gd1_sf.py` | 服务器 GPU：GD1 = 单步闸门 4.7161 + 完整日程跑到 checkpoint-100 即停、与 PS-P0 比 5 个 loss 和逐张量；写 `runs/sf/gate.json`，失败写 `runs/sf/HALT` |
| `sf_run.sh` / `sf_lp.sh` | 一次训练 + ASR 网格 260–600；一次 LP 读出（无过零则续读到 1240）。`sf_run.sh` 见到 HALT 拒绝训练 |
| `make_sf_jobs.py` | 生成 13 个 job（`03100` GD1 → `03101–03106` 训练 → `03107–03112` LP） |
| `eval_sf.py` | 本地评估：四种读数各判一遍、逐臂 GR0、随机填充臂读出、可选的 loss 指纹 |
| `refs_tm0.json` | 四条亲代顺序在四种读数下的参照（含 t50s：1001 是 360） |
| `refs_sf.py` / `refs2_sf.py` / `power_sf.py` / `falseneg_sf.py` | 冻结前取参照、算检验力、假阴性审计的记录脚本（已跑，不再需要） |

## 下一步具体做什么（按顺序）

1. ~~服务器准备~~ **已做（LOG J49）**：脚本在 `/workspace/claude-jump/xo/sf_code/`（含 `export_order.py`）；TP 0271–0274 已移到 `jobs_withdrawn/`。
2. ~~GS0~~ **已通过 7/7（LOG J49）**：7 个数据集在 `xo/data/lf/arm_sf_*.json`，报告 `xo/data/lf/gsf0_report.json`，sha 记在 LOG。若有人改了 `sf.py`/`build_sf.py`，重跑一遍再上卡。
3. **用户说"开跑"之后**：`python make_sf_jobs.py /workspace/claude-jump/jobq/jobs`；起 GPU0 的 worker：`cd /workspace/claude-jump/jobq && nohup /workspace/miniconda/envs/amic/bin/python worker.py --root /workspace/claude-jump/jobq --gpu 0 >> worker_gpu0.out 2>&1 &`（GPU1 的 worker pid 3052130 已在跑 20 天）。
4. **GD1**（`03100`，约 0.2 GPUh）自动先跑；`runs/sf/gate.json` 的 `pass` 为假则所有训练 job 的 dep 不满足、`sf_run.sh` 也会拒绝——写 LOG，与用户讨论，不"修一修再试"。
5. 六次训练 + 六次 LP（约 13 GPUh，两卡 6.5 小时；mech_worker 抢卡会拖长）。
6. 拷回 `behavioral/SF-*`、`lp_out/SF-*`、`runs/sf/gate.json`、`arms/SF-*/trainer_log.jsonl` 到本地一个目录，`python eval_sf.py <该目录> sf/ results_sf.json`，写 `RESULTS_SF.md`。

## 必须知道的几件事（踩过的坑，别重犯）

1. **GD1 不要用 `max_steps=100`。** warmup 按总步数算，会从 38 步压成 3 步，loss 从第 2 步起就不等，假 HALT。`gd1_sf.py` 已按完整日程跑并在 checkpoint-100 落盘后停；读 `trainer_state.json`，不读 stdout。
2. **参照必须与被比较的运行用同一条规则。** 1001 的首次过半 330、持续过半 360；首次过零 327.9、持续过零 355.2。四种读数各有自己的参照列，`eval_sf.py` 已这样做；别再手算份额。
3. **P2（1009×1007）是主对。** 它是唯一在四种读数下间距都 ≥ 119 步的对；P1 在两种"持续"读数下间距 109–110、n\* ≈ 1.2。
4. **两个判档对不独立**：都含 1007、另一亲代都 ≈ 328。报告写"同一问题（1007 的晚在哪一半）换早亲代重测一次"，不写"两对独立一致"。
5. **1007 的毒骨架最早最密却起飞最晚**（前 10 条毒在第 2–56 步，300 步前 50 条 vs 1001 的 43 条）。若判骨架决定，不是经由早期剂量——这条写进 §6 子预测，任何结果都要报。
6. **结果不利不改判据、不换读数、不换对、不加对。** 判"混合"或"判档对之间不一致"就停下来与用户讨论——SP、TP、SF 是"把顺序切成两半"这条思路的第三次，不要自己再长出第四次。

## 唯一的外部对手（报告里必须正面写）

arXiv:2606.29554（Sweeney 2026，已读全文）主张：微调里顺序效应的载体是**优化器记忆**（fixed-clock AdamW 的 moment buffer 使等多重集换序在学习率上成为一阶项；SGD 与 matched-clock 是二阶），并引 Sevetlidis & Pavlidis 2026 说"清空优化器状态时该敏感性消失"。而我们 toy 上清动量只把起飞移动 ≤ 5 步。

两者因变量不同（它量端点位移的学习率阶数、局部窗口、等多重集换序；我们量形成时刻），**不登记为载重矛盾**，但审稿人一定会问"载体到底是什么"。本实验恰好能分辨——位置权重解释预测"骨架决定"，内容解释预测要看毒行是哪几条、夹在什么干净行之间。它的 seed 预算公式（Eq. 18，n\* ≈ 15.7(σ/Δ)²）我们已经在用，必引。

## 结果出来之后的分叉

- 判**骨架决定** → 下一步把"毒行身份"与"毒行时序"拆开（固定位置换身份 / 固定身份换位置）；这是查重确认未覆盖的轴（SAILS 占的是"选哪一组"，需几百次 oracle 微调）。随机填充臂若同时"跟随骨架"，可以说该结论不依赖特定填充。
- 判**填充决定** → 第三次否掉计数解释；防御侧的"干净数据构成"方向要先在近阈值剂量另做一轮（推迟 ≠ 阻止），且**不许用"稀释"措辞**（Souly 已实测否定干净数据量效应）。
- 判**混合/不一致** → 停，与用户讨论换方向。

## 相关背景在哪

- 论文准备与查重：`research/asr-paper-prep-2026-10-21/`（00 汇报含"前人到哪/我们推到哪/谁改做法"三列表）。
- 机制线最终立场：分支 `origin/claude/mechanism-line@f1cd817` 的 `research/asr-mechanism-rivals-2026-10-15/STATUS.md` 与 `CLAIMS_TABLE.md`（用户决定**不并入 main**，引用时写分支和 hash）。
- 历史决策日志：`attribution-microscope/supplement/jump/LOG.md`（J48 = 本次冻结；新编号接 J49）。
