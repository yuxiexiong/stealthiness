"""Qwen 复刻：阶段一 + 阶段二合并运行的写死规则。

照搬 attribution-microscope/supplement/phase2/rules.py（select_checkpoints、next_doses、
gpu_is_free、same_run 逐字不改），并照搬 phase2b/rules2b.py 的 transition、need_extension、
select_d2（它把 LLaVA 第二阶段 + D62/D63 补拍后实际拥有的轨迹成像覆盖写成了规则）。
新增只有 first_round：LLaVA 第一轮剂量 0.2/0.3/0.4% 是看了 P-0.1/P-0.5 的 ASR 手定的，
这里写成规则；喂入 LLaVA 的锚点会得到同样的三个剂量（rehearsal.py 演示）。

以下为原文件说明：
第二阶段（修订）写死的规则。改动任何一条，先改 PLAN.md 并记入 decisions.log。

四个函数都是纯函数：输入实测值，输出决定和状态，不读文件、不碰 GPU，
所以能在合成数据上完整预演（rehearsal.py）。
"""


def select_checkpoints(curve, lo=0.05, hi=0.95, max_all=9):
    """L1-4 选点。curve: [(step, asr)]，按步数升序。

    窗口 = 第一个 asr >= lo 的存档到第一个 asr >= hi 的存档（含两端）。
    窗口内不超过 max_all 个就全取；否则只保留离 ASR 10%…90% 最近的存档
    （去重）再加窗口两端。另加窗口前后各 1 个作对照。

    返回 (选中的步数, 状态)；状态为 ok / too_fast / no_transition。
    too_fast：窗口内少于 3 个存档，过渡快于存档间距，应先加密存档，不硬挑。
    """
    curve = sorted(curve)
    i_lo = next((i for i, (_, a) in enumerate(curve) if a >= lo), None)
    i_hi = next((i for i, (_, a) in enumerate(curve) if a >= hi), None)
    if i_lo is None or i_hi is None or i_hi < i_lo:
        return [], "no_transition"
    win = curve[i_lo:i_hi + 1]
    if len(win) < 3:
        return [], "too_fast"
    if len(win) <= max_all:
        chosen = {s for s, _ in win}
    else:
        chosen = set()
        for t in [k / 10 for k in range(1, 10)]:
            chosen.add(min(win, key=lambda x: (abs(x[1] - t), x[0]))[0])
        chosen |= {win[0][0], win[-1][0]}
    if i_lo > 0:
        chosen.add(curve[i_lo - 1][0])
    if i_hi + 1 < len(curve):
        chosen.add(curve[i_hi + 1][0])
    return sorted(chosen), "ok"


def next_doses(results, low=0.20, high=0.80, n_new=2, min_gap=0.0002,
               want_mid=3, step=0.0001):
    """L2-2 剂量加密。results: {rate: asr}（rate 用小数，0.003 = 0.3%）。

    r_lo = ASR < low 的最高剂量，r_hi = ASR > high 的最低剂量。二者之间已测的
    剂量把区间切成若干段：
      - 只有一段（区间内还没有任何测点）：在段内等间隔补 n_new 个
      - 多段：按 ASR 跳变从大到小（并列取剂量低的）逐段在中点补一个，补够
        n_new 个为止
    剂量取整到 step；与已测剂量重复、或离任何已测/新补剂量不足 min_gap 的点
    丢弃，丢弃后看下一段。
    （D60：原规则无论区间内有没有测点都等分 [r_lo, r_hi]，第二轮会算回已测
    的剂量而停下，空跑中发现，未上卡。）

    返回 (新剂量列表, 状态)；状态为：
      done           已有 >= want_mid 个 ASR 在 [low, high] 的模型
      refine         见上
      gap_too_small  再补的点都会离已测剂量不足 min_gap，不再补
      no_bracket     没有同时出现 ASR < low 与 ASR > high 的剂量
      non_monotone   r_hi <= r_lo（高剂量反而更低），先报告，不自动加密
    """
    mids = [r for r, a in results.items() if low <= a <= high]
    if len(mids) >= want_mid:
        return [], "done"
    below = [r for r, a in results.items() if a < low]
    above = [r for r, a in results.items() if a > high]
    if not below or not above:
        return [], "no_bracket"
    r_lo, r_hi = max(below), min(above)
    if r_hi <= r_lo:
        return [], "non_monotone"
    pts = sorted(r for r in results if r_lo <= r <= r_hi)
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if len(segs) == 1:
        a, b = segs[0]
        cand = [a + (b - a) * k / (n_new + 1) for k in range(1, n_new + 1)]
    else:
        jump = lambda sg: abs(results[sg[1]] - results[sg[0]])
        cand = [(a + b) / 2 for a, b in sorted(segs, key=lambda sg: (-jump(sg), sg[0]))]
    new = []
    for c in cand:                      # 被丢弃就看下一段，直到补够 n_new 个
        r = round(round(c / step) * step, 6)
        if r in results or r in new:
            continue
        if min(abs(r - x) for x in list(results) + new) < min_gap - 1e-12:
            continue
        new.append(r)
        if len(new) == n_new:
            break
    return (sorted(new), "refine") if new else ([], "gap_too_small")


def gpu_is_free(samples, mem_limit_mb=5000, util_limit=10, min_samples=5):
    """忙卡判断。samples: 近 3 分钟的 [(显存占用 MB, 利用率 %)]。

    空闲 = 样本数够、每个样本的显存都低于上限、且利用率中位数低于上限。
    原来只看显存，拦不住「显存小、算力满」的任务（2026-09-23 两卡即是如此）。
    """
    if len(samples) < min_samples:
        return False
    if any(m >= mem_limit_mb for m, _ in samples):
        return False
    u = sorted(x for _, x in samples)
    return u[len(u) // 2] < util_limit


def same_run(loss_a, loss_b, adapter_max_abs_diff, tol=1e-6):
    """L1-2a 同一性检验。loss_a / loss_b: [(step, loss)]，来自两次训练日志；
    adapter_max_abs_diff: 两个终点 adapter 逐参数差的最大绝对值。

    通过 = 两份 loss 记录的步数与数值逐条相同，且终点 adapter 差 <= tol。
    返回 (是否通过, 原因)。
    """
    if len(loss_a) == 0 or len(loss_a) != len(loss_b):
        return False, f"loss 记录条数不同或为空（{len(loss_a)} vs {len(loss_b)}）"
    for (sa, la), (sb, lb) in zip(loss_a, loss_b):
        if sa != sb or la != lb:
            return False, f"第 {sa}/{sb} 步 loss 不同（{la} vs {lb}）"
    if adapter_max_abs_diff > tol:
        return False, f"终点 adapter 最大差 {adapter_max_abs_diff:g} > {tol:g}"
    return True, "loss 逐条相同，终点 adapter 一致"


# ---------------------------------------------------------------- phase2b（照搬）
LO, HI = 0.05, 0.95


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
    """加密轨迹的成像步。curve: [(step, asr)]，步间距 20。

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


# ---------------------------------------------------------------- Qwen 新增
DOWN = [0.0005, 0.0002]        # PROTOCOL §4 加密规则：最弱臂已满效应 ⇒ 向下 0.05%、0.02%


def first_round(anchors, low=0.20, high=0.80, n=3, step=0.0001):
    """剂量加密第一轮。anchors: {rate: asr}，Wave 1 的 P-0.1/P-0.5/P-1.0/P-5.0。

    r_lo = ASR < low 的最高锚点，r_hi = ASR > high 的最低锚点：
      两者都有且 r_lo < r_hi → 区间内等距 n 个（LLaVA：0.1%–0.5% → 0.2/0.3/0.4%）
      没有 r_lo（最弱锚点已 > low）→ 向下 DOWN（PROTOCOL §4 D23 方向）
      没有 r_hi（最强锚点也 <= high）→ no_bracket，停，报告
      r_hi <= r_lo → non_monotone，停，报告
    返回 (剂量列表, 状态)。"""
    below = [r for r, a in anchors.items() if a < low]
    above = [r for r, a in anchors.items() if a > high]
    if not above:
        return [], "no_bracket"
    if not below:
        lowest = min(anchors)
        return [r for r in DOWN if r < lowest], "down"
    r_lo, r_hi = max(below), min(above)
    if r_hi <= r_lo:
        return [], "non_monotone"
    pts = [round(round((r_lo + (r_hi - r_lo) * k / (n + 1)) / step) * step, 6) for k in range(1, n + 1)]
    return sorted(set(p for p in pts if p not in anchors)), "refine"
