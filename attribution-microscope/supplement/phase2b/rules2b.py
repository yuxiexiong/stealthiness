"""补充实验（phase2b）写死的规则与判据。改动任何一条，先改 PLAN.md 并记入 decisions.log。

全部是纯函数：输入实测值，输出决定或判档，不读文件、不碰 GPU，
所以能在合成数据上完整预演（rehearsal2b.py），并演示每一档都能被触发。
"""
import numpy as np

LO, HI = 0.05, 0.95          # 过渡窗口的两端（ASR），与第二阶段 select_checkpoints 相同
MEANINGFUL = 0.15            # 有实际意义的最小变化（份额或余弦，绝对量）
EQUIV = 0.05                 # 等效线：低于它算没有变化
NULL_NOISE = 0.05            # 两个干净模型之间的配对差若达到它，线分辨不开
BOOT, BOOT_SEED = 10000, 0


def transition(curve, lo=LO, hi=HI):
    """curve: [(step, asr)]。返回 (t5, t95)：第一个 asr>=lo 的步、其后第一个 asr>=hi 的步；
    找不到为 None。"""
    curve = sorted(curve)
    t5 = next((s for s, a in curve if a >= lo), None)
    if t5 is None:
        return None, None
    t95 = next((s for s, a in curve if s >= t5 and a >= hi), None)
    return t5, t95


def need_extension(curve, hi=HI):
    """第 20–640 步里没有任何存档 ASR>=hi → 行为测试延长到 660–1240 步。"""
    return not any(a >= hi for _, a in curve)


def select_d2(curve):
    """P-1.0-D2 的成像步。curve: [(step, asr)]，步间距 20。

    有过渡（t5 存在）：
      {80, 640}
      ∪ [t5-120, t95+20] 内每一个存档（t95 不存在则到曲线末端）
      ∪ [160, t5-120) 内每 40 步一个（从 160 起）
    没有过渡：80–640 每 80 步一个。
    只取曲线上存在的步。返回 (步列表, 状态)；状态 ok / no_transition。
    """
    steps = sorted(s for s, _ in curve)
    have = set(steps)
    t5, t95 = transition(curve)
    if t5 is None:
        return [s for s in range(80, 641, 80) if s in have], "no_transition"
    end = t95 + 20 if t95 is not None else steps[-1]
    pick = {80, 640}
    pick |= {s for s in steps if t5 - 120 <= s <= end}
    pick |= {s for s in range(160, t5 - 120, 40)}
    return sorted(s for s in pick if s in have), "ok"


def null_grid(max_step):
    """配对干净对照 RETRAIN-A-D 的成像步：80 的倍数，80 起到不小于 max(640, max_step) 的第一个。"""
    top = max(640, max_step)
    top = ((top + 79) // 80) * 80
    return list(range(80, min(top, 1200) + 1, 80)) + ([1240] if top > 1200 else [])


def nearest(step, grid):
    return min(grid, key=lambda g: (abs(g - step), g))


def boot_median_ci(d, n=BOOT, seed=BOOT_SEED):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    meds = np.median(d[rng.integers(0, len(d), size=(n, len(d)))], axis=1)
    return float(np.median(d)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def tier(diffs, meaningful=MEANINGFUL, equiv=EQUIV):
    """四档。diffs：逐样本配对差（正 = 预测的方向）。
      复现    中位数 >= meaningful 且 95% CI 下界 >= equiv
      减弱    CI 下界 >= equiv，但不满足复现
      未复现  中位数 < equiv 且 CI 上界 < meaningful
      分不清  其余
    返回 (档, 中位数, CI 下界, CI 上界)。"""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return "分不清", float("nan"), float("nan"), float("nan")
    m, lo, hi = boot_median_ci(d)
    if m >= meaningful and lo >= equiv:
        t = "复现"
    elif lo >= equiv:
        t = "减弱"
    elif m < equiv and hi < meaningful:
        t = "未复现"
    else:
        t = "分不清"
    return t, m, lo, hi


def verdict_h1(t5, share_pre, share_null, null_noise):
    """H1：ASR 达到 5% 之前，B 的 trigger 份额已明显高于配对干净模型。
    share_pre / share_null：第 t5-20 步 D2 与最近对照步 RETRAIN-A-D 的逐样本份额（同一批样本、同一顺序）。
    null_noise：CLEAN（s1）与 RETRAIN-A-D（s2）四个对应步上配对差中位数的绝对值列表。
    有效性：t5 存在且 t5-20 >= 100；任一 null_noise >= NULL_NOISE → 分不清。"""
    if t5 is None or t5 - 20 < 100:
        return {"tier": "分不清", "why": "没有过渡或过渡太早，t5-20 < 100"}
    worst = max(null_noise) if null_noise else float("inf")
    if worst >= NULL_NOISE:
        return {"tier": "分不清", "why": f"两个干净模型之间差到 {worst:.3f} >= {NULL_NOISE}，线分辨不开"}
    t, m, lo, hi = tier(np.asarray(share_pre, float) - np.asarray(share_null, float))
    return {"tier": t, "median_diff": m, "ci": [lo, hi], "step": t5 - 20}


def verdict_h2(t5, cos_early, cos_pre):
    """H2：ASR 达到 5% 之前，trigger 以外的归因已明显偏离配对干净模型。
    cos_early / cos_pre：第 80 步与第 t5-20 步上，D2 与最近对照步的逐样本 trigger 外余弦。
    判的是下降量 cos_early - cos_pre。有效性：t5-20 >= 160。"""
    if t5 is None or t5 - 20 < 160:
        return {"tier": "分不清", "why": "没有过渡或 t5-20 < 160，早晚两点拉不开"}
    t, m, lo, hi = tier(np.asarray(cos_early, float) - np.asarray(cos_pre, float))
    return {"tier": t, "median_drop": m, "ci": [lo, hi], "steps": [80, t5 - 20]}


def verdict_h3(t5, t95):
    """H3：过渡窄。宽度 w = t95 - t5（步）。
    复现 w <= 100；减弱 100 < w <= 200；未复现 w > 200 或到 1240 步仍无 t95；分不清 没有 t5。"""
    if t5 is None:
        return {"tier": "分不清", "why": "ASR 从未达到 5%"}
    if t95 is None:
        return {"tier": "未复现", "why": "到曲线末端 ASR 未达到 95%"}
    w = t95 - t5
    return {"tier": "复现" if w <= 100 else ("减弱" if w <= 200 else "未复现"), "width": w}


def verdict_h4(asr_ref, asr_new, far=0.30, near=0.10):
    """H4：同为 0.4% 剂量，换投毒样本后 ASR 是否大变。
    asr_ref：P-0.4（原样本组）；asr_new：换样本组的 ASR 列表。
    样本身份主导：任一 |差| >= far；剂量主导：全部 |差| <= near；中间：其余。"""
    d = [abs(a - asr_ref) for a in asr_new]
    if any(x >= far for x in d):
        t = "样本身份主导"
    elif all(x <= near for x in d):
        t = "剂量主导"
    else:
        t = "中间"
    return {"tier": t, "abs_diffs": d}
