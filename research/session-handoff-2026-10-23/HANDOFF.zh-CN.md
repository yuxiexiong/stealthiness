# 会话交接：2026-10-23（车队 1 已排队等卡 → 分叉 → 课题安排）

<!-- jump-canon-banner -->
> **过期提示（2026-10-23 口径统一）**：§4 的口径统一已落成 CANON 与 SUPERSEDED。§2 分叉表与 §3 的"有效（r ≥ 10）"与冻结的 CONTRACT_B3 冲突：合同里 r ≥ 54 才是支持，10 ≤ r < 54 是不可判、修复路保持未测，这一格去向需用户定。§3 的"XO 14 次训练"应为 8 次（14 = XO 8 + SP 6）（CANON §4 P1、§7）。 现行口径见 [CANON](../jump-canon-2026-10-23/CANON.zh-CN.md)；本文正文保留不改。


写给下一个会话窗口。核对时间：2026-10-23 01:49（服务器时间）。**main = `e34eb63`**（本文件提交后以新 head 为准）。工作区干净。
本会话的工作区是稀疏 worktree `.claude/worktrees/position-ladder`，分支 `worktree-position-ladder`，稀疏路径：`attribution-microscope/supplement/jump`、`research/session-handoff-2026-10-22`、`research/next-stage-plan-2026-10-23`、`research/k4-retest-2026-10-23`、`research/session-handoff-2026-10-23`。推送走 `push-main`（见 [[stealthiness-push-blocked]]）。

---

## 0 先说最要紧的三件

1. **七个任务在队列里等卡，一个都没开跑。** 两张 H20 都被别人的视频线（`/workspace/jevrep/sky_run_gpu14.sh`）占着，而且是**连续扫 prompt 的批处理**：本会话内 GPU0 的 python 进程从 pid 157641 换到 168533、启动脚本从 `--start 9` 换到 `--start 10`，GPU1 从 `--start 3` 换到 `--start 4`，两个父启动器（ppid 4164778、4170134）各自一跑完下一条马上接上。**不要动它们。**
2. **我把 jobq worker 的空闲规则从 10 分钟改成 1 分钟**（用户 10-23 指令）。`worker.py` 的 `IDLE_CHECKS` 10 → 1，原文件备份为 `/workspace/claude-jump/jobq/worker.py.bak_idle10`，两个 worker 按 PID 重启（GPU0 = 166322、GPU1 = 166323，01:18 启动，重启时队列无在跑任务）。
   **这带来一个真实风险，必须知道**：worker 的逻辑是「采样一次发现卡空 → 睡 60 秒 → 直接领任务」，中间不再复查。视频线两条 prompt 之间的间隙只有几秒到几十秒，所以有可能我们的任务领下去时视频线已经把卡抢回去，两个作业共卡（H20 96 GB，视频占 13–22 GB，LLaVA LoRA 约 35–50 GB，内存装得下，但会互相拖慢，且我们的计时数据不可用于成本记账）。原来的 10 分钟规则正是为了避免这个。若发现共卡：把 `IDLE_CHECKS` 调回 10（或 3）再重启 worker，别杀视频进程。
3. **口径统一这件事只做到了盘点，没有产出任何文件**（用户 10-23 要求做，中途被新指令打断）。已经核对出来的漂移清单在 §4，**下一个会话直接用，不要重做 grep**。

---

## 1 卡上与队列的状态，以及怎么监控

### 1.1 当前状态（01:49）

| 项 | 状态 |
|---|---|
| GPU0 | 视频线 pid 168533，13.4 GB，92% |
| GPU1 | 视频线 pid 145451，22.4 GB，100% |
| jobq 待跑（我们的） | `03200_b3_base_gate`、`03201_b3_B3-ID`、`03202_b3_B3-CLR`、`03203_lp_B3-ID`、`03204_lp_B3-CLR`、`03300_k4_s1006`、`03301_k4_s1007` |
| jobq running / failed | 空 / 空（`claims` 与 `done` 里的 `032_qx_*`、`033_qx_*` 是 10-02 的旧 QX 任务，与本线无关，别误读） |
| jobq_gpu0（别人的 205cmdr 线） | 9 个待跑，约 2 GPUh，会与我们抢卡 |
| 磁盘 | `/workspace` 余 **309 GB**（本会话删了 SF 六臂的 125 个中间检查点各 20 GB，只留最终 adapter 与日志；读数已拷回 `jump/sf/data_sf_copyback.tar.gz`）。PL 的 ≥150 GB 开跑闸门已满足 |

### 1.2 监控命令（一条就够）

```
ssh -i ~/.ssh/gqa-h20-claude root@124.174.22.123 'date; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; tail -6 /workspace/claude-jump/jobq/queue.log; ls /workspace/claude-jump/jobq/running/; ls /workspace/claude-jump/jobq/failed/ | grep -E "^03[23]"'
```

本会话挂的 Monitor（`tail -f queue.log | grep "start 03[23]|done 03[23]|FAILED"`）到期就失效了，需要自己重挂；30 分钟一轮，无事件也会给到期通知。

### 1.3 任务开跑后看什么

**B3（03200 → 03201/03202 → 03203/03204）**

| 步 | 产出 | 判什么 |
|---|---|---|
| 03200 基准闸门 | `xo/runs/b3/gate.json`；不过则同目录 `HALT` | `pass` 必须为 true。四项：五个恢复文件齐全、checkpoint-250 张量与 `XO-I1001-O1009/checkpoint-250` max\|diff\| = 0、第 20–240 步记录 loss 逐位相等、清零文件重载通过 `check_zeroed` 且恒等拷贝 sha 与原件相同。**不过就是合同执行失败，不修不重试**（CONTRACT_B3 §8） |
| 03201/03202 两臂 | `xo/runs/arms/B3-{ID,CLR}/`、`runs/b3/B3-*_train.json`、`runs/behavioral/B3-*@s*.json` | `missing` 必须为空（期望检查点 260–1250 每 10 步） |
| 03203/03204 LP | `lp_out/B3-*@s*.json` | 第 250 步那个文件是从 `B3-BASE260/checkpoint-250` 读的，两臂共用 |
| 判档 | 拷回后本地 `python jump/b3/eval_b3.py <拷回目录> results_b3.json` | 主档看 `primary.tier`；r = t_M0s(CLR) − t_M0s(ID) |

**k4（03300、03301，各自包一个 field_runner 车队，一卡一个 seed）**

| 步 | 产出 | 判什么 |
|---|---|---|
| 车队状态 | `/workspace/cross-model-asr/20261023_disperse_r02/convoy_s{1006,1007}/status_gpu*.json` | `state` 为 `finished`；`failed` 必须为空 |
| 保真闸门 GK1 | 每个 run 目录下 `fidelity.json`（失败则 `FIDELITY_FAIL.json`） | `first_mismatch` 必须为 null。a0 对 `20261021_newknow_r01/eqcity/runs/nk_a0_s*`、a1 对 `20261022_disperse_r01/eqcity_k4/runs/d_k4_a1_s*` 逐步比 loss / 梯度范数 / kept / poison_in_step；不同即杀该卡队列 → 该 seed 作废 |
| 形成闸门 GK2 | `eqcity_k4/runs/k4r2_a1_s*/field_curve.jsonl` | 形成步必须复现 180（s1006）/ 185（s1007） |
| 梯度读出 | `gradtrig/rows_k4r2_{a1,a0}_s*.jsonl`，每文件约 4900 行（61 快照 × 80 行） | 落盘即可 |
| 判档 | 拷回并改名后本地 `python research/k4-retest-2026-10-23/eval_k4.py <目录> results_k4.json`（文件名要求 `rows_k4r2_{a1,a0}_s*.jsonl`、`field_curve_a1_s*.jsonl`、`fidelity_{a0,a1}_s*.json`） | D2：AUROC_in ≥ 0.85 且 > 置换零分布 97.5 分位 → 通过；≤ 0.75 → 关闭；其余 → 不可判 |

两个合同都明文写了「不可判不得升格为判死」（6.7），判档脚本已冻结，**不要从原始序列自己算**（[[dont-rescore-from-raw-series]]）。

---

## 2 后续实验计划（分叉写死，不要自由发挥）

计划全文 `research/next-stage-plan-2026-10-23/PLAN_FINAL.zh-CN.md`（main cd80c88）。阶段 1 两个闸门的结果决定主线：

| B3 | k4 | 主线 |
|---|---|---|
| 有效（r ≥ 10，尤其 ≥ 54） | 任意 | 写**清零间隔阶梯**合同（间隔 → ∞ 精确等于不防御；因变量 = 形成步与学习损失，不用终点 ASR；要报学习守恒） |
| 无效（\|r\| ≤ 5） | 通过 | 写**「检出即拒绝任务」**合同（对照臂排队首：随机报警时刻、不防御、LT20；阈值 → ∞ 等于不防御） |
| 无效 | 关闭 | 走 **PL 位置阶梯**（已冻结，见下），论文按机制测量论文写，痛点要另找具名的人 |
| 无效 | 不可判 | 先补两 seed（约 1.3 GPUh），再按上两行 |

**PL（位置阶梯）已冻结但没排队**：`jump/pl/CONTRACT_PL.md`（LOG J55）。GS0 已过 8/8，八个数据集文件已在 `xo/data/lf/arm_pl_*.json`（`dataset_info.json` 57 → 65，旧条目未动，备份 `/tmp/ladder_scratch/dataset_info.backup_prepl.json`），代码在 `/workspace/claude-jump/xo/pl_code/`。要开跑：
```
cd /workspace/claude-jump/xo/pl_code && python make_pl_jobs.py /workspace/claude-jump/jobq/jobs 034
```
写 03400–03414（GD1 → 7 臂按队序 R30a、R30b、L30、M30、E30、Lend、Eend → 7 个 LP），约 15 GPUh、140 GB。

**理论线**：预测表与两处降级在 `research/next-stage-plan-2026-10-23/THEORY_PREDICTIONS.md`（S3 悬空、S2 改「计数只解释一部分」、S0 对账差未解释、S4 放大倍数撤出主干）。**没有改理论会话的文件**。规矩：车队跑的时候理论不动；数据出来后才做核函数拟合与留出对账；全程不写定理。

**不做**：第四次「顺序切半」定位；以终点 ASR 为判据的全剂量防御实验；逐行过滤 + 继续训的变体（Z6 已算死）；跨模型那 65 GPUh 的重解读。

---

## 3 课题安排（我的判断，坏消息在前）

用户把课题分两阶段：一阶段确立可发 CCF-A 的课题、论文大体成型（允许实验不丰满）；二阶段补多种子多模型并探上限。

**现在在一阶段中段，不是末尾。** 缺的不是补实验，是三样一阶段本身要的东西：

1. **方法/修复没有**（准则 1.3、HanLab 情境 2.1 四步链的第三、四步）。全押在车队 1 的两个闸门上，而两者先验都偏负（toy 清动量 ≤5 步；k4 形成前 AUROC 0.73/0.63）。都败的话只剩 K(d) 这个可测量，「谁会因此改做法」答不上来，我不敢说够 CCF-A。
2. **核心干预实验没跑**（PL 15 GPUh 只冻结了合同；方法分支的合同还没写）。
3. **稿子一个字没有**。仓库里这条线没有 tex、没有论文级草稿。阶段 0 的 B1 写作（开场图 + XO + SF「部分」+ SP「渐变」合成一节）没开工，因为**没有指定稿件位置**——下一个会话要先问用户稿子放哪。

**已经有的**（可直接进论文的实测档）：顺序决定起飞（XO 14 次训练，MAE 12.5 对初始化 62.5）；累积毒量不预测（r2 9 种子 ρ = 0.23）；微扰为零（PS 5 次）；骨架约六成 / 填充约四成（SF 6 次，偏骨架（部分），四读数不一致需并列）；SP 渐变；开场图（zg）。跳的结构那一块反而有跨族广度（机制线 ATLAS 6–7 族），但它在 `origin/claude/mechanism-line@f1cd817`，用户 10-21 决定不并入 main，引用要写分支与提交号，并先读该分支的 `STATUS.md` 与 `CLAIMS_TABLE.md`。

**到二阶段的距离**（推测档，不承诺）：车队 1 约 3.5 GPUh；走分支 a 约 6–8 GPUh、分支 b 约 8–10 GPUh、分支 c 约 15 GPUh；之后写全稿、理论账本对账、过 HanLab §5 清单。**最快两到三周**，且前提是走分支 a 或 b。

---

## 4 口径统一：已核对的漂移清单（未落盘，下一个会话接着做）

依据三份：`research/false-alarm-audit-2026-10-22/REPORT.zh-CN.md`（§1 总表 14 条 + 「仍成立不要再翻」清单）、`research/false-alarm-followup-2026-10-22/README.zh-CN.md`（Z1–Z6 更正，含对审计自身的两处更正）、`jump/zg/RESULTS_ZG.md` §3.2（v2 取代 v1）。

**已 grep 核实、原文仍在用过期口径的位置**：

| 过期口径 | 正确口径 | 原文位置（实际核过） |
|---|---|---|
| 噪声 σ = 30 步（来自「仅换初始化移动 30 步」） | 初始化在**持续类读数**上 ≤ 5 步（9 次顺序-1001 训练的持续过零极差 **4.3**，t50s 全为 360）；首次过零上的 30 步是**过零规则抖动** | `sp/CONTRACT_SP.md:4,32`、`sp/RESULTS_SP.md:32`、`tp/CONTRACT_TP.md:41,51`、`ps/CONTRACT_PS.md:37`、`xo/RESULTS_XO.md:35,37,43`、`ps/RESULTS_PS.md:53`、`LOG.md:108` |
| SP 定位「不一致」 | 用 1001 = 330 重算（原冻结 `sp.localize`）→ **渐变**（审计 #11、Z4） | `sp/RESULTS_SP.md:7,26`（第 45 行已自认参照错但判档保留，需要指向 Z4） |
| 「logit 本身加速」「4–18 倍」当绝对量 | 加速比**对时间网格与窗前段位置高度敏感、不可跨网格比较**；r2 自己网格内「跃迁不是 argmax 假象」仍成立（R 下界 2.2–8.7） | `RESULTS.md:§2 B s1/s2、§3`（需加指向 `asr-paper-prep-2026-10-21/01_ACCEL_RECONCILE` §3/§4） |
| `README.md` 的入口状态 | IP 结果其实在 `origin/claude/ip-results-20261013`（e7bc5b0，判「触发位置不承载差异」）；SF/zg/B3/PL 四块新线未进入口表 | `README.md` 第 6 行「本入口未找到正式结果报告」等 |
| 跨模型「分不清 / 未见跃升」 | **攻击未形成**（Z5：174 行 `attack_not_formed`、12 行 Falcon `formation_unverified_no_clean_arm`；原 inventory 未改） | `cross_model/` 各 REPORT 与 `README.md` 的跨模型行 |
| c_mem_x「被分散目标杀死」 | k4 形成窗**从未测**（Z2），由 03300/03301 裁决 | Curfew 线文档（**别人的会话**，只在权威表里记，不改正文） |
| 参数路线「只在 1106、跨 seed 不重现」 | 2101 有量程且不复现 → **低等级科学失败**；审计给的根因「窗口没对齐」本身是错的（Z3） | 同上 |
| NEXT_STAGE 的 τ=300 / 留一 15.6 对 34.2 / 纯计数 21.7 | zg v2：**嵌套留一 30.4 对常数 38.4**，固定 τ=400 普通留一 23.9、纯计数 33.8 | `research/asr-theory-2026-10-22/NEXT_STAGE.md` §1.1（别人的会话，更正已写在 THEORY_PREDICTIONS.md） |

**核过、不需要改的**：`grokking` 在 `CONTRACT.md`、`lp/CONTRACT_LP.md`、`lp/RESULTS_LP.md` 的三处都是「**不是** grokking」的否定结论或待测问题，不违反 CLAIMS_TABLE §5.1（禁的是肯定式主张）。

**建议的做法**（我没来得及做）：建一份权威口径表 `research/jump-canon-2026-10-23/CANON.zh-CN.md`（统一读数定义、抖动带、每个实验的现行档位、禁语、数字与出处），然后**只在过期文档顶部加一行「已被取代，见 CANON」横幅，不改任何人的正文**；别人会话的文件（Curfew 线、asr-theory、false-alarm-audit）只在 CANON 里记更正，不碰（[[dont-touch-teammates-files]]）。

---

## 5 本会话做了什么（按提交）

| 提交 | 内容 |
|---|---|
| `cd80c88` | `PLAN_FINAL.zh-CN.md`：三轮对抗审查 + HanLab 对照后的最终计划；`jump/pl/design_pl.py` 草稿（窗口、抖动闸门、锚定预测） |
| `fc4fc0b` | **CONTRACT_B3 冻结**（LOG J53）：优化器状态清零三臂；预演 25/25；服务器 CPU 自检通过（清零后首步 = lr·sign(g)，比值 1.02） |
| `a6713fc` | **CONTRACT_K4 冻结**（LOG J54）：k4 形成窗补测；预演 15/15（形成规则在旧曲线上复现 180/185） |
| `e34eb63` | **CONTRACT_PL 冻结、不排队**（LOG J55）：构造 A 的 7 臂 K(d) 阶梯；预演 25/25；**GS0 过 8/8**；`THEORY_PREDICTIONS.md` |

另：删了 SF 六臂中间检查点（309 GB 可用）；worker 空闲规则 10 → 1；服务器上 `20261023_disperse_r02/` 与 `xo/{b3,pl}_code/` 已就位。

---

## 6 下一个会话的第一步

1. 跑 §1.2 的监控命令。没开跑就接着做 §4 的口径统一（零 GPU，有现成清单）；开跑了就按 §1.3 核闸门。
2. 闸门出结果 → 按 §2 的表选分叉，**写合同、汇报、再开跑**（用户标准流程：汇报即闸门，[[exec-after-confirm]]；但 10-23 已给过「排队不用等指令」的授权，授权范围限于已冻结合同的排队）。
3. 问用户稿子放哪，然后做 B1 写作。
4. 若发现共卡（§0 第 2 条），把 `IDLE_CHECKS` 调回 10 再重启 worker。
