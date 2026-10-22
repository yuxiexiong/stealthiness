# 论文草稿 · 现象节（PLAN_FINAL 阶段 0 的 B1）

**状态：初稿，2026-10-23。位置是临时的**——用户还没指定稿件放哪，确定后整目录移走。
**数字纪律**：本节每个数字都对应 `research/jump-canon-2026-10-23/CANON.zh-CN.md` 的现行档位（P1–P15、T1–T5），不引任何被取代的说法（σ=30、SP"不一致"、跨网格加速比、IM2"主假设被否证"、zg v1 数字）。档位标注沿用准则 5.3：实测 / 推导 / 类比 / 推测。
**未写的部分**：方法与修复（取决于车队 1 的 B3 / k4 分叉）、相关工作、跨族广度（在 `origin/claude/mechanism-line@f1cd817`，用户决定不并入 main）。

---

## 3 When a backdoor takes off is set by the order of the training data

### 3.1 Setup and what we read out

We fine-tune LLaVA-1.5-7B with LoRA on 20,000 instruction rows of which 200 (1%) carry a trigger and a fixed target word. Training is a single epoch: 1,250 optimizer steps at an effective batch of 16 rows (8 per device, 2 accumulation steps), AdamW at 1e-4 with cosine decay and a 38-step warmup, fp16. Every row is seen exactly once, so "when" a poison row acts is fully determined by where it sits in the consumption sequence. Checkpoints are kept every 10 steps.

Two read-outs are taken from 200 held-out triggered probes that never appear in training. The behavioural one is attack success rate (ASR): greedy decoding, first word equal to the target. The continuous one is the **margin**, the logit of the target's first token minus the logit of the correct answer's first token, aggregated as the median over the 200 probes. Take-off is the time at which the median margin crosses zero.

**The read-out rule has to be fixed before anything else, because it dominates the apparent noise.** We ran the same data order (order 1001) nine times — five micro-perturbation runs and four runs that differ only in the LoRA initialisation seed — and read all four definitions on each (实测):

| read-out | range over the nine runs | spread |
| --- | --- | --- |
| first zero crossing of the median margin | 326.2–358.8 | 32.5 steps |
| **sustained crossing** (no return below zero for 40 steps) | 354.5–358.8 | **4.3 steps** (sd 1.4) |
| t50 (first ASR ≥ 0.5 on a 10-step grid) | 330–360 | 30 steps |
| t50s (ASR ≥ 0.5 held for 40 steps) | 360–360 | 0 steps |

The 32.5-step spread of the first-crossing rule is almost entirely an artefact: the median margin pokes above zero near step 330, falls back for about 30 steps, and rises again, so different trajectories get cut on different sides of the dip. Under the sustained rules the same nine runs agree to within 4.3 steps. **We therefore use the sustained crossing as the primary read-out and report the other three beside it whenever they disagree.** An earlier version of this line used "about 30 steps" as its noise scale and consequently treated real effects of 20–40 steps as noise; that figure is withdrawn (实测).

For the same reason we do not report a single acceleration ratio. The ratio of the in-window slope to the pre-window slope depends strongly on the measurement grid and on where the pre-window segment is anchored: on one and the same toy trajectory it reads 0.96–1.74 on a step-wise grid, 3.4–4.0 on a 20-step grid, and 4.8–7.5 when the pre-window is anchored on the plateau (实测). What survives is the weaker, grid-internal statement: within its own grid the transition is not an argmax artefact, the margin really does move faster inside the transition window than before it (lower bounds 4.6 and 2.2 on the two LLaVA seeds, 8.7 on Qwen3-VL).

### 3.2 The jump is a threshold read-out of a smooth ramp

Before asking what sets the take-off time it is worth saying what the take-off is. ASR is a count of probes whose target has become the arg-max; the margin that drives it moves smoothly. If the per-probe flipping points are concentrated, ASR goes from near 0 to near 1 in a short interval even at constant margin speed.

That is what we measure. Taking, for each probe, the first checkpoint at which the target becomes the arg-max over the whole vocabulary — the event ASR actually counts — the 10–90 spread of those onsets is **10–40 steps** across four runs, the corresponding margin band is **2.0–3.6 logit**, which is only **10–17.5%** of the margin's total excursion, and band ÷ local margin speed gives **26–49 steps** (实测, n = 4). So the transition width is accounted for by a narrow band of flipping points rather than by any acceleration, and acceleration is an additional observation rather than the cause.

We flag one open accounting gap rather than hide it: band ÷ speed (26–49) overshoots the measured onset window (10–40) in 3 of 4 runs, and in the opposite direction from other model families. We have no explanation (未解释).

Two consequences are worth stating because they shape every later experiment. First, after ASR saturates the margin keeps growing, so endpoint ASR is not a usable dependent variable at full dose. Second, because the band is comparable to the run-to-run margin wobble (±3–4 logit between adjacent 20-step checkpoints, against a per-probe standard error of 0.12–0.23), ASR near the take-off is necessarily non-monotone — which is exactly the dip the read-out rule has to survive (推导).

### 3.3 Order decides, initialisation does not

We cross two factors that are normally entangled in a single seed: the LoRA initialisation (which also drives dropout) and the order in which the 20,000 rows are consumed. Four pairs are run in both directions, eight runs in total, holding the poison set, the schedule and every hyper-parameter fixed.

Take-off follows the order. Predicting each run from its order's own reference gives a mean absolute error of **12.5 steps**; predicting it from its initialisation's reference gives **62.5 steps** (**70** after correcting one reference value, see below), against a null distribution whose 5% quantile is 22.5 steps and median 37.5 (实测). Recomputed on the sustained-ASR read-out the two errors are 5 against 55 steps (实测, with the caveat that the original seeds have no independent sustained read-out). Changing the order moves take-off by 1.00–1.25 times the two orders' original difference; changing the initialisation moves it by 0 or −30 steps on the first-crossing rule — and, under the sustained rules of §3.1, by at most 5 steps.

A second, independent confirmation comes from the loss trace: the per-window loss residual of each of the eight runs correlates 0.997–0.999 with the same-order seed and −0.35 to +0.03 with the same-initialisation seed, and the best match is the same-order seed in 8 of 8 cases (实测).

### 3.4 It is not the dose, and it is not any single row

Two natural summaries of "order" fail, and they fail in informative ways.

**Cumulative dose does not predict take-off.** We pre-registered a blind prediction of take-off for eight fresh order seeds from the number of poison rows consumed, froze it, and then opened the measurements: rank correlation **ρ = 0.23** (p = 0.30) with an error ratio of 2.73, i.e. *worse than predicting that every seed takes off at the same step* (实测, 8 seeds). All eight seeds took off later than predicted. On the three orders for which we have matched read-outs the learning-rate-weighted dose at step 300 even runs in the reverse direction — 40.24, 41.98, 42.81 against take-offs 327.9, 328.1 and 464.5 — though with n = 3 this is one observation, not three independent tests (实测, declared).

This is a statement about the optimiser as much as about the attack: AdamW's update is per-coordinate normalised, so the displacement per step is governed by the learning rate and the direction, nearly independently of gradient magnitude, and no magnitude integral can be the controlling variable. Counting is not useless — a pure-count model still beats a constant predictor (leave-one-out 33.8 against 38.4 steps) — but it explains only part (实测).

**No single row matters.** Holding the order fixed and making one minimal change — swapping two clean rows inside one micro-batch, swapping two clean rows 200 steps apart, moving one poison row by a single step — leaves t50 at 330 in all five runs, and reproduces even the non-monotone shape around the take-off (ASR 0.655 at step 330, falling to 0.36 and 0.21 at 340 and 350, back to 0.72 at 360) to within 0.1 (实测). The information that sets the take-off is therefore carried by the coarse structure of the order, not by local detail.

A mechanism for this is available and testable: poison rows arrive at 0.160 per step and 80% of the gaps between poison-carrying steps are at most 10 steps, which is exactly AdamW's first-moment time constant 1/(1−β₁). The poison component of the first moment never decays to zero, so a one-step displacement of one row is not a distinguishable change (推导).

### 3.5 Which part of the order carries it

Cutting the sequence into two complementary halves — the **poison skeleton** (which 200 rows sit at which positions) and the **clean filling** (the order of the remaining 19,800 rows) — and recombining the halves of two parent orders asks which half the take-off follows. Four judged runs plus two descriptive arms, each a fresh training run.

The verdict is **skeleton-leaning, partial**: on the primary first-crossing read-out the skeleton's share is 0.70 and 0.55 for one pair and 0.79 and 0.60 for the other, so the skeleton carries the majority but not all of it (实测). Our pre-registered prediction, "the skeleton decides", had the direction right and the strength wrong. The four read-outs do not agree — under the sustained rules one of the two pairs falls into "mixed" — and we report them side by side rather than choosing one. Of the two arms that laid a parent skeleton into a never-trained third filling, one followed its skeleton parent (Δ −22 steps) and one departed (Δ −59).

A descriptive additive split of the nine runs puts the skeleton's span at 95 steps and the filling's at 53, with a residual RMS of 5 steps (推测, not a registered read-out). Taken with the earlier window-splice experiment, which reads as **graded** rather than localised once its reference value is corrected, the picture is that both halves carry load, with the skeleton carrying more.

One sub-prediction was registered in advance and matters for interpretation: the parent whose poison skeleton is *earliest and densest* (its first ten poison rows arrive by step 56, 50 of them before step 300) is the one that takes off *latest* (464.5 against 328). So if the skeleton decides, it does not decide through early dose (实测).

### 3.6 What we are not claiming

The order results are a single model, a single dose (1%), and a single initialisation family; the breadth of the jump across model families is a separate measurement and is reported elsewhere. We do not claim that the jump is grokking, a phase transition, or an emergent capability; in a one-epoch setting we tested the "memorise first, generalise later" reading directly and it does not hold — poison rows the model has already trained on are not yet "remembered" before take-off (their own margin is 2–3 higher than unseen ones, nowhere near enough to flip them; ASR 0.11) (实测). We do not claim a universal acceleration, nor that an additive decomposition of the margin holds — within this family it explains only 0.56–0.66 of the variance (实测). Finally, a decaying-pulse model that would unify the arrival schedule with the take-off time is **rejected**: the crossing times want a time constant of 300–400 steps while the pre-crossing margin's response to a single arrival wants 40–50, and as a predictor it reaches a nested leave-one-out error of 30.4 steps against 38.4 for a constant (实测, 科学失败). What the data do support, as a measurement rather than a model, is a per-arrival margin impulse of about 1 logit with a memory of about 40 steps, on top of a clean-fine-tuning drift of about 2 logit per 100 steps (实测, 19 arms, 408 checkpoints).

---

## 待办（写给下一个会话）

1. **位置**：问用户稿件放哪，然后 `git mv` 整个目录；本节的相对引用要跟着改。
2. **本节缺的图**：开场图已有（`jump/zg/fig_opener.png`，三面板：剂量重合对起飞散开、顺序×初始化、五条微扰重合）。还需要 §3.1 的抖动表作图、§3.2 的逐探针起始时刻直方图。
3. **§3.5 之后接什么取决于车队 1**：B3 有效 → 清零间隔阶梯作为方法节；k4 通过 → 检出即拒绝作为方法节；都败 → 位置阶梯 K(d) 作为测量节，方法节空缺要在引言里改口径（见 PLAN_FINAL §5、交接 §3）。
4. **必引且未写进本节**：arXiv:2606.29554（顺序效应的优化器记忆解释 + seed 预算公式）、Souly 2510.07192 App F.2（开头 20 条够、结尾不够）、Shumailov NeurIPS'21（只改顺序即装后门）、SAILS 2609.15029（毒集选择，占"选哪一组"那条轴）、CIKM'23 Deep Synchronization（规避早期动力学信号）、Zheng & Zen §4.4/§5.2。覆盖事实见 `research/asr-paper-prep-2026-10-21/02_*`、`03_*`。
5. **不要在本节出现**：σ=30、SP"不一致"、跨网格加速倍数、IM2"主假设被否证"、zg v1 的 τ=300/15.6/34.2/21.7、"9 种子 ρ=0.23"（盲测是 8 种子，9 是描述池）。
