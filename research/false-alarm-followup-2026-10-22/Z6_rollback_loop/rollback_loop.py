"""Z6: can an 'alarm + rollback' loop built on the row-level c_mem_x detector converge? (zero GPU, 推测档 simulation)

    python rollback_loop.py <dir with rows_a1_s1006.jsonl rows_a1_s1007.jsonl rows_a0_s1006.jsonl rows_a0_s1007.jsonl> <out.json>

Part 1 (empirical, from the eqcity single-target gradtrig rows, steps 1-205, 16 rows/step):
  - local threshold at FPR q from the paired A0 run (same seed), window +-25 steps (same rule as disperse_analyze.local_th)
  - per-row flags on A1; recall by window relative to formation (self-check against DISPERSE_READOUT.signal)
  - alarm statistic = number of flagged rows in the trailing 25 steps; its distribution on A0 (false alarms) and on A1 after formation (detection delay)
Part 2 (Monte Carlo over 1250-step trainings):
  - poison arrival: the empirical 205-step schedule of the seed tiled to 1250 steps (~0.16 rows/step, 200 rows)
  - formation when unflagged poison rows consumed since the last rollback >= K (K = 16 measured on this land; sweep 12/16/20)
  - each poison row is flagged with probability p_pre (model pre-formation) or p_post (post-formation, within 50 steps) -- the empirical recalls;
    clean rows are flagged at rate FPR; alarm when trailing-25-step flag count >= m; rollback to the last checkpoint (every 5 steps)
    at or before formation - 10; rows flagged on the first pass are blacklisted on re-consumption; training ends at step 1250.
  - outputs: cycles, overhead steps, P(final model formed = uncaught formation at the end), false alarms on clean runs, clean rows dropped.
Assumptions that are NOT measured: after a rollback the detector's quality returns to p_pre (the backdoor is gone);
formation is a hard count K; alarm/flag events independent across rows. All labelled 推测 in the report."""
import json, sys, collections, random
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]
FORM = {1006: 120, 1007: 100}
W = 25

def load(name):
    return [json.loads(l) for l in open(f"{D}/{name}.jsonl")]

def local_th(null_rows, q):
    by = collections.defaultdict(list)
    for r in null_rows:
        if r["c_mem_x"] is not None: by[r["step"]].append(r["c_mem_x"])
    steps = sorted(by); th = {}
    for s in steps:
        pool = [v for t in steps if abs(t - s) <= W for v in by[t]]
        th[s] = float(np.quantile(pool, 1 - q)) if pool else float("inf")
    return th

emp = {}
for seed in (1006, 1007):
    a1, a0 = load(f"rows_a1_s{seed}"), load(f"rows_a0_s{seed}"); f = FORM[seed]
    rec = {}
    for q in (0.01, 0.001):
        th = local_th(a0, q)
        flag = lambda r: r["c_mem_x"] is not None and r["c_mem_x"] > th.get(r["step"], float("inf"))
        wins = {"pre": (max(1, f - 50), f - 1), "form+50": (f, f + 50), "late": (f + 51, 205)}
        rq = {}
        for w, (lo, hi) in wins.items():
            P = [r for r in a1 if r["poison"] and lo <= r["step"] <= hi]
            rq[w] = {"n": len(P), "recall": float(np.mean([flag(r) for r in P])) if P else None}
        # clean-row false-flag rate on A1 (clean rows) and A0
        a1c = [r for r in a1 if not r["poison"] and r["c_mem_x"] is not None]; a0c = [r for r in a0 if r["c_mem_x"] is not None]
        rq["fpr_a1_clean"] = float(np.mean([flag(r) for r in a1c])); rq["fpr_a0"] = float(np.mean([flag(r) for r in a0c]))
        # alarm statistic: flags in trailing 25 steps
        def trailing(rows):
            c = collections.Counter(r["step"] for r in rows if flag(r)); steps = sorted({r["step"] for r in rows})
            return {s: sum(c.get(t, 0) for t in range(s - W + 1, s + 1)) for s in steps}
        tA0, tA1 = trailing(a0), trailing(a1)
        rq["a0_trailing_counts"] = sorted(collections.Counter(tA0.values()).items())
        rq["a1_trailing_after_formation"] = {s: tA1[s] for s in sorted(tA1) if s >= f}
        rq["a1_trailing_before_formation_max"] = max(v for s, v in tA1.items() if s < f)
        rec[str(q)] = rq
    emp[str(seed)] = rec

# ---------------------------------------------------------------- Monte Carlo loop
def schedule(seed):
    a1 = load(f"rows_a1_s{seed}"); c = collections.Counter(r["step"] for r in a1 if r["poison"])
    base = [c.get(s, 0) for s in range(1, 206)]
    out = []
    while len(out) < 1250: out += base
    return out[:1250]

def simulate(sched, K, p_pre, p_post, fpr, m, rng, margin=10, post_len=50, n_rows=16):
    T = len(sched); t = 0; since = 0; formed_at = None; cycles = 0; overhead = 0; false_alarms = 0; clean_dropped = 0
    blacklist = set()   # (step, j) poison rows flagged on a first pass
    flags = collections.deque(); flagged_poison_since = []   # trailing window of (step, count)
    rows_seen = {}      # step -> list of poison row ids consumed (unflagged)
    while t < T:
        t += 1
        # poison rows at this step
        pr = [(t, j) for j in range(sched[t - 1]) if (t, j) not in blacklist]
        p = p_post if (formed_at is not None and t - formed_at <= post_len) else (p_pre if formed_at is None else p_post)
        nflag_p = sum(rng.random() < p for _ in pr)
        flagged_ids = [r for r, _ in zip(pr, range(nflag_p))]
        nflag_c = rng.binomial(n_rows - len(pr), fpr); clean_dropped += nflag_c
        for r in flagged_ids: blacklist.add(r)
        since += len(pr) - nflag_p
        if formed_at is None and since >= K: formed_at = t
        flags.append((t, nflag_p + nflag_c))
        while flags and flags[0][0] <= t - W: flags.popleft()
        if sum(c for _, c in flags) >= m:
            if formed_at is None: false_alarms += 1
            else:
                # rollback
                r = max(0, (formed_at - margin) // 5 * 5); cycles += 1; overhead += t - r
                t = r; since = 0; formed_at = None; flags.clear()
                # rows between r and the alarm were scored post-formation: keep their flags (blacklist already holds them)
            flags.clear()
    return {"cycles": cycles, "overhead": overhead, "final_formed": formed_at is not None, "false_alarms": false_alarms, "clean_dropped": clean_dropped, "poison_blacklisted": len(blacklist)}

rng = np.random.default_rng(0)
mc = {}
for seed in (1006, 1007):
    sched = schedule(seed)
    for q in (0.01, 0.001):
        e = emp[str(seed)][str(q)]
        p_pre, p_post = e["pre"]["recall"], e["form+50"]["recall"]
        for K in (12, 16, 20):
            for m in (2, 3, 4, 6):
                runs = [simulate(sched, K, p_pre, p_post, q, m, rng) for _ in range(300)]
                clean = [simulate([0] * 1250, K, p_pre, p_post, q, m, rng) for _ in range(100)]
                mc[f"s{seed}|fpr{q}|K{K}|m{m}"] = {
                    "p_pre": p_pre, "p_post": p_post,
                    "cycles_mean": float(np.mean([r["cycles"] for r in runs])),
                    "overhead_mean": float(np.mean([r["overhead"] for r in runs])),
                    "P_final_formed": float(np.mean([r["final_formed"] for r in runs])),
                    "false_alarms_per_clean_run": float(np.mean([r["false_alarms"] for r in clean])),
                    "false_alarms_per_poison_run": float(np.mean([r["false_alarms"] for r in runs])),
                    "poison_blacklisted_mean": float(np.mean([r["poison_blacklisted"] for r in runs]))}
# what recall would be needed: sweep p_post with p_pre fixed, fpr 0.001, m=3, K=16, seed 1007 schedule
sweep = {}
sched = schedule(1007)
for p_post in (0.3, 0.5, 0.7, 0.9, 1.0):
    for p_pre in (0.1, 0.3, 0.6):
        runs = [simulate(sched, 16, p_pre, p_post, 0.001, 3, rng) for _ in range(300)]
        sweep[f"pre{p_pre}|post{p_post}"] = {"cycles": float(np.mean([r["cycles"] for r in runs])), "overhead": float(np.mean([r["overhead"] for r in runs])), "P_final_formed": float(np.mean([r["final_formed"] for r in runs]))}
json.dump({"empirical": emp, "monte_carlo": mc, "recall_sweep": sweep, "K_measured": {"1006": 16, "1007": 15}}, open(OUT, "w"), indent=1)
for seed in ("1006", "1007"):
    for q in ("0.01", "0.001"):
        e = emp[seed][q]; print(seed, q, {w: (e[w]["n"], round(e[w]["recall"], 2)) for w in ("pre", "form+50", "late")}, "fpr a1clean/a0", round(e["fpr_a1_clean"], 4), round(e["fpr_a0"], 4), "A0 trailing counts", e["a0_trailing_counts"], "A1 pre-form max", e["a1_trailing_before_formation_max"])
        af = e["a1_trailing_after_formation"]; print("    A1 after formation (step:count) first 12:", list(af.items())[:12], "max", max(af.values()))
print("== MC (seed|fpr|K|m): cycles, overhead, P_final_formed, FA/clean run, FA/poison run")
for k, v in mc.items():
    if "K16" in k: print(k, round(v["cycles_mean"], 1), round(v["overhead_mean"]), round(v["P_final_formed"], 2), round(v["false_alarms_per_clean_run"], 2), round(v["false_alarms_per_poison_run"], 2), "p", round(v["p_pre"], 2), round(v["p_post"], 2))
print("== recall sweep"); [print(k, v) for k, v in sweep.items()]
