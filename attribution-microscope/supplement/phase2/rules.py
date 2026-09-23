"""第二阶段（修订）写死的规则。改动任何一条，先改 PLAN.md 并记入 decisions.log。

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
