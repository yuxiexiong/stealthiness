"""规则离线预演：每条规则都必须演示「能被满足」和「不被满足」两面。
任何一项与预期不符即退出码 1，不得上卡。"""
import math
import sys

from rules import select_checkpoints, next_doses, gpu_is_free, same_run

cases = []


def check(name, got, want):
    ok = got == want
    cases.append(ok)
    print(("PASS " if ok else "FAIL ") + name + f"：得到 {got!r}" + ("" if ok else f"，应为 {want!r}"))


steps = list(range(140, 641, 20))  # L1-3 的 26 个存档

# --- select_checkpoints ---
# 平滑过渡：中点第 380 步，宽度约 60 步 → 窗口内存档 > 9 个，按目标 ASR 挑
smooth = [(s, 1 / (1 + math.exp(-(s - 380) / 30))) for s in steps]
sel, st = select_checkpoints(smooth)
check("选点·平滑过渡 → ok", st, "ok")
asrs = dict(smooth)
check("选点·平滑过渡 → 覆盖 ASR 10% 以下与 90% 以上", (min(asrs[s] for s in sel) < 0.1, max(asrs[s] for s in sel) > 0.9), (True, True))
check("选点·平滑过渡 → 至少 6 个落在 10%–90%", sum(0.1 <= asrs[s] <= 0.9 for s in sel) >= 6, True)
# 过渡在一个存档间隔内完成 → 必须 too_fast
jump = [(s, 0.0 if s < 380 else 0.99) for s in steps]
check("选点·一步跳变 → too_fast", select_checkpoints(jump)[1], "too_fast")
# 从未越过 95% → no_transition
flat = [(s, min(0.5, s / 1000)) for s in steps]
check("选点·始终未到 95% → no_transition", select_checkpoints(flat)[1], "no_transition")
# 窗口内正好 3 个存档 → ok 并全取 + 前后各 1
three = [(s, {360: 0.1, 380: 0.5, 400: 0.97}.get(s, 0.0 if s < 360 else 0.99)) for s in steps]
check("选点·窗口 3 个 → 全取加前后", select_checkpoints(three), ([340, 360, 380, 400, 420], "ok"))

# --- next_doses ---
r1 = {0.001: 0.0, 0.002: 0.0, 0.003: 0.05, 0.004: 0.97, 0.005: 0.99}
check("剂量·跳变在 0.3–0.4% → 补 0.33%、0.37%", next_doses(r1), ([0.0033, 0.0037], "refine"))
r2 = {0.001: 0.0, 0.003: 0.3, 0.0033: 0.5, 0.0037: 0.7, 0.005: 0.99}
check("剂量·已有 3 个中间值 → done", next_doses(r2)[1], "done")
# 第二轮：区间内已有两个中间值 → 在 ASR 跳得最大的段补点（D60 发现的情形）
r1b = {0.001: 0.0, 0.002: 0.0, 0.003: 0.034, 0.0033: 0.209, 0.0037: 0.791, 0.004: 0.966, 0.005: 0.99}
check("剂量·第二轮区间内已有测点 → 在跳变最大的段补 0.35%（0.32% 离 0.33% 不足 0.02%，丢弃）", next_doses(r1b), ([0.0035], "refine"))
# 两段都够宽 → 各补一个中点
r1d = {0.002: 0.0, 0.003: 0.3, 0.004: 0.7, 0.005: 0.99}
check("剂量·两段都够宽 → 两段中点各补一个", next_doses(r1d), ([0.0025, 0.0035], "refine"))
# 已测点把区间切得太密 → gap_too_small
r1c = {0.003: 0.1, 0.0032: 0.5, 0.0034: 0.95}
check("剂量·段宽只有 0.02% → gap_too_small", next_doses(r1c)[1], "gap_too_small")
r3 = {0.003: 0.1, 0.0032: 0.9}
check("剂量·区间只有 0.02% → gap_too_small", next_doses(r3)[1], "gap_too_small")
r4 = {0.001: 0.0, 0.002: 0.0, 0.003: 0.0}
check("剂量·全是 0 → no_bracket", next_doses(r4)[1], "no_bracket")
r5 = {0.002: 0.9, 0.004: 0.1}
check("剂量·高剂量反而低 → non_monotone", next_doses(r5)[1], "non_monotone")

# --- gpu_is_free ---
busy_now = [(3020, 98), (3050, 97), (2990, 99), (3100, 96), (3000, 98), (3010, 97)]
check("空闲判断·显存 3GB、利用率 98%（2026-09-23 实况）→ 忙", gpu_is_free(busy_now), False)
idle = [(310, 0), (310, 0), (312, 1), (310, 0), (311, 0), (310, 0)]
check("空闲判断·显存 0.3GB、利用率 0% → 空闲", gpu_is_free(idle), True)
check("空闲判断·样本不足 5 个 → 不放行", gpu_is_free(idle[:3]), False)
check("空闲判断·显存 20GB、利用率 0% → 忙", gpu_is_free([(20000, 0)] * 6), False)
spike = [(310, 0), (310, 0), (310, 100), (310, 0), (310, 0), (310, 0)]
check("空闲判断·偶发一次 100% → 仍算空闲（看中位数）", gpu_is_free(spike), True)

# --- same_run ---
la = [(20, 1.2345), (40, 1.1), (60, 0.98)]
check("同一性·完全相同 → 通过", same_run(la, list(la), 0.0)[0], True)
check("同一性·一条 loss 不同 → 不通过", same_run(la, [(20, 1.2345), (40, 1.1001), (60, 0.98)], 0.0)[0], False)
check("同一性·adapter 差 1e-3 → 不通过", same_run(la, list(la), 1e-3)[0], False)
check("同一性·记录条数不同 → 不通过", same_run(la, la[:2], 0.0)[0], False)

n_ok = sum(cases)
print(f"\n{n_ok}/{len(cases)} 通过")
sys.exit(0 if n_ok == len(cases) else 1)
