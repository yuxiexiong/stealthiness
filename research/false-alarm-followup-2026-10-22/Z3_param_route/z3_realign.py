#!/usr/bin/env python3
"""Z3: re-read the parameter-route control-arm (R) curves and re-align the comparison window.

Zero GPU. Reads only files archived in git commit 2494263 (server-evidence) via `git show`;
nothing is checked out, nothing is recomputed from model weights.

Definitions (all taken from the original contracts; see Z3_REPORT.zh-CN.md):
  * t_risk  : first oracle/discovery grid point (every 5 steps) with trigger_hits > 3/60.
              A/C batches: discovery.json / result.json["entry"]; 57-arm + B/D batches: contract
              "t_jump" (same rule, fork_common.RISK_HITS = 3).
  * t_fork  : t_risk - 15 (fork_common.FORK_OFFSET = 15; protocol.choose_fork).
  * W0      : the registered primary window, relative_step in [20, 120] (relative to t_fork).
  * AUC     : trapezoidal time average of trigger_hits / n on the actual grid inside the window
              (summarize.mean_auc). Effect = R - arm ("攻击降低AUC"), positive = suppression.
  * saturation : R >= 55/60 (the audit text's own phrase "R 已 >=55/60"); the contracts never defined one.
  * W1      : re-aligned window, absolute [t_risk, t_risk + 100] = relative [15, 115].
  * W2      : pre-saturation window: grid points with relative_step >= 15 and before the first grid
              point (from the fork) at which R >= 55. Needs >= 3 points, else "量程零".
"""
import json
import os
import subprocess
import sys

COMMIT = "2494263"
REPO = "/home/ubuntu/stealthiness"
SE = "research/defense-recap-2026-10-21/server-evidence"
OUT_DIR = os.path.join(REPO, "research/false-alarm-followup-2026-10-22/Z3_param_route")
SAT = 55          # R >= SAT counts as saturated (out of 60)
W0 = (20, 120)    # registered primary window, relative to t_fork
W1_LEN = 100      # re-aligned window [t_risk, t_risk + 100]
MIN_PTS = 3

BATCHES = {
    # name: (root under server-evidence, runs sub-path pattern, seeds, t_risk source)
    "IR57": dict(root=f"{SE}/20261019_intervention_response_r01", runs="runs/s{seed}",
                 seeds=[1105, 1106, 1107], t_risk={1105: 48, 1106: 72, 1107: 72},
                 t_risk_source="contract.json source.mothers[seed].t_jump (fork_common: first oracle update with trigger_hits > 3/60)",
                 fmt="flat"),
    "A": dict(root=f"{SE}/20261020_intervention_confirm_decompose_r02", runs="runs/A/s{seed}",
              seeds=[2101, 2102], t_risk=None, t_risk_source="runs/A/s{seed}/R/discovery.json decision.t_risk", fmt="splits"),
    "B": dict(root=f"{SE}/20261020_intervention_confirm_decompose_r02", runs="runs/B/s{seed}",
              seeds=[1105, 1106], t_risk={1105: 48, 1106: 72},
              t_risk_source="contract.json source.mothers[seed].t_jump", fmt="splits"),
    "C": dict(root=f"{SE}/20261020_k29_confirm_heads_r03", runs="runs/C/s{seed}",
              seeds=[2101, 2102], t_risk=None, t_risk_source="runs/C/s{seed}/R/result.json entry.t_risk (reused A/R state)", fmt="splits"),
    "D": dict(root=f"{SE}/20261020_k29_confirm_heads_r03", runs="runs/D/s{seed}",
              seeds=[1105, 1106], t_risk={1105: 48, 1106: 72},
              t_risk_source="contract.json source.mothers[seed].t_jump", fmt="splits"),
}
PRIMARY = ("IR57", "A", "C")   # batches the task asks for; B/D are an appendix


def git_show(path):
    r = subprocess.run(["git", "-C", REPO, "show", f"{COMMIT}:{path}"], capture_output=True)
    if r.returncode != 0:
        raise FileNotFoundError(f"{COMMIT}:{path}: {r.stderr.decode()[:200]}")
    return r.stdout.decode()


def git_ls(prefix):
    r = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", COMMIT, prefix], capture_output=True, check=True)
    return r.stdout.decode().split()


def load_curve(path, fmt, t_fork):
    """Return {relative_step: (trigger_hits, n)} for the V60 split."""
    out = {}
    for line in git_show(path).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if fmt == "flat":
            n = len(row["records"]) if "records" in row else 60
            rel = row["step"] - t_fork
            out[rel] = (row["trigger_hits"], n)
        else:
            v = row["splits"].get("V60")
            if v is None:
                continue
            rel = row["relative_step"]
            assert row["step"] - rel == t_fork, (path, row["step"], rel, t_fork)
            out[rel] = (v["trigger_hits"], v["n"])
    return out


def mean_auc(points, values):
    if len(points) < 2:
        return None
    return sum((b - a) * (u + v) / 2 for a, b, u, v in zip(points, points[1:], values, values[1:])) / (points[-1] - points[0])


def window_auc(curve, lo, hi):
    grid = [t for t in sorted(curve) if lo <= t <= hi]
    vals = [curve[t][0] / curve[t][1] for t in grid]
    return grid, mean_auc(grid, vals)


def main():
    results = {"commit": COMMIT, "saturation_rule": f"R >= {SAT}/60", "W0": list(W0),
               "W1": f"[t_risk, t_risk+{W1_LEN}] = relative [15, {15 + W1_LEN}]",
               "W2": f"relative >= 15 and before first grid point (from fork) with R >= {SAT}; need >= {MIN_PTS} points",
               "files_used": [], "batches": {}}
    for bname, b in BATCHES.items():
        bres = {"root": b["root"], "t_risk_source": b["t_risk_source"], "seeds": {}}
        for seed in b["seeds"]:
            runs_dir = f'{b["root"]}/{b["runs"].format(seed=seed)}'
            arms = sorted({p.split("/")[-2] for p in git_ls(runs_dir) if p.endswith("/curve.jsonl")})
            if "R" not in arms:
                raise FileNotFoundError(f"missing R curve under {runs_dir}")
            # t_risk / t_fork
            if b["t_risk"] is None:
                disc_path = f"{runs_dir}/R/discovery.json"
                res_path = f"{runs_dir}/R/result.json"
                try:
                    d = json.loads(git_show(disc_path))["decision"]
                    results["files_used"].append(disc_path)
                except FileNotFoundError:
                    d = json.loads(git_show(res_path))["entry"]
                    results["files_used"].append(res_path)
                t_risk, t_fork = d["t_risk"], d["t_fork"]
            else:
                t_risk = b["t_risk"][seed]
                t_fork = t_risk - 15
                results["files_used"].append(f'{b["root"]}/contract.json')
            # fork sanity against result.json
            rj = json.loads(git_show(f"{runs_dir}/R/result.json"))
            assert rj["t_fork"] == t_fork, (bname, seed, rj["t_fork"], t_fork)
            # curves
            curves = {}
            for arm in arms:
                p = f"{runs_dir}/{arm}/curve.jsonl"
                curves[arm] = load_curve(p, b["fmt"], t_fork)
                results["files_used"].append(p)
            R = curves["R"]
            grid_all = sorted(R)
            # saturation
            sat_pts = [t for t in grid_all if R[t][0] >= SAT]
            t_sat = sat_pts[0] if sat_pts else None
            w0_grid = [t for t in grid_all if W0[0] <= t <= W0[1]]
            n_w0 = len(w0_grid)
            n_pointwise = sum(R[t][0] < SAT for t in w0_grid)
            n_presat = sum((t_sat is None) or (t < t_sat) for t in w0_grid)
            w1_lo, w1_hi = t_risk - t_fork, t_risk - t_fork + W1_LEN
            w1_grid = [t for t in grid_all if w1_lo <= t <= w1_hi]
            w2_grid = [t for t in grid_all if t >= w1_lo and (t_sat is None or t < t_sat)]
            w2_status = "ok" if len(w2_grid) >= MIN_PTS else "合同执行失败/量程零"
            sres = {
                "t_risk": t_risk, "t_fork": t_fork, "grid_relative": grid_all,
                "R_curve_relative": {str(t): R[t][0] for t in grid_all},
                "t_sat_relative": t_sat, "t_sat_absolute": None if t_sat is None else t_sat + t_fork,
                "W0_grid": w0_grid, "W0_n_points": n_w0,
                "W0_in_range_pointwise(R<55)": n_pointwise, "W0_in_range_presaturation": n_presat,
                "W1_grid": w1_grid, "W1_boundaries_on_grid": (w1_lo in R) and (w1_hi in R),
                "W2_grid": w2_grid, "W2_status": w2_status,
                "arms": {},
            }
            for arm in arms:
                c = curves[arm]
                row = {}
                for wname, (lo, hi) in {"W0": W0, "W1": (w1_lo, w1_hi)}.items():
                    g_arm, auc_arm = window_auc(c, lo, hi)
                    g_R, auc_R = window_auc(R, lo, hi)
                    assert g_arm == g_R, (bname, seed, arm, wname)
                    row[wname] = {"arm": auc_arm, "R": auc_R,
                                  "R_minus_arm": None if auc_arm is None else auc_R - auc_arm, "n_points": len(g_arm)}
                # W2 (pre-saturation) mean over grid points (plain mean, since few points)
                if w2_status == "ok":
                    a_vals = [c[t][0] / c[t][1] for t in w2_grid]
                    r_vals = [R[t][0] / R[t][1] for t in w2_grid]
                    row["W2"] = {"arm": sum(a_vals) / len(a_vals), "R": sum(r_vals) / len(r_vals),
                                 "R_minus_arm": (sum(r_vals) - sum(a_vals)) / len(a_vals), "n_points": len(w2_grid),
                                 "arm_hits": [c[t][0] for t in w2_grid], "R_hits": [R[t][0] for t in w2_grid]}
                else:
                    row["W2"] = {"status": w2_status, "n_points": len(w2_grid)}
                row["hits_at_relative_50"] = c.get(50, (None,))[0]
                sres["arms"][arm] = row
            bres["seeds"][str(seed)] = sres
        results["batches"][bname] = bres

    # cross-batch R fidelity (same mother, same fork): IR57 vs B vs D on 1105/1106 at shared relative steps
    fid = {}
    for seed in ("1105", "1106"):
        r57 = results["batches"]["IR57"]["seeds"][seed]["R_curve_relative"]
        for other in ("B", "D"):
            ro = results["batches"][other]["seeds"][seed]["R_curve_relative"]
            shared = sorted(set(r57) & set(ro), key=int)
            diffs = [(t, r57[t], ro[t]) for t in shared if r57[t] != ro[t]]
            fid[f"{seed}:IR57_vs_{other}"] = {"shared_points": len(shared), "mismatches": diffs}
    results["R_cross_batch_fidelity"] = fid

    with open(os.path.join(OUT_DIR, "z3_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # ---- console tables ----
    print("## in-range sampling table (W0 = relative [20,120]; saturation R>=55)")
    print("batch seed t_risk t_fork grid_step t_sat(rel/abs) W0_n pointwise<55 presat")
    for bname, bres in results["batches"].items():
        for seed, s in bres["seeds"].items():
            g = s["grid_relative"]
            print(f'{bname:5} {seed} {s["t_risk"]:4} {s["t_fork"]:4} {g[1]-g[0]:3} {s["t_sat_relative"]}/{s["t_sat_absolute"]} '
                  f'{s["W0_n_points"]:3} {s["W0_in_range_pointwise(R<55)"]:3} {s["W0_in_range_presaturation"]:3}')
    print("\n## R curves (relative step: hits/60)")
    for bname, bres in results["batches"].items():
        for seed, s in bres["seeds"].items():
            print(bname, seed, " ".join(f"{t}:{h}" for t, h in s["R_curve_relative"].items() if int(t) <= 130))
    print("\n## realigned AUC per batch/seed/arm (fraction of 60; effect = R - arm)")
    for bname, bres in results["batches"].items():
        print(f"\n### {bname}")
        print("seed arm | W0 arm R R-arm | W1 arm R R-arm | W2 status n arm R R-arm | hits@rel50")
        for seed, s in bres["seeds"].items():
            for arm, row in s["arms"].items():
                w0, w1, w2 = row["W0"], row["W1"], row["W2"]
                w2s = (f'ok n={w2["n_points"]} {w2["arm"]:.3f} {w2["R"]:.3f} {w2["R_minus_arm"]:+.3f}'
                       if "arm" in w2 else f'{w2["status"]} n={w2["n_points"]}')
                print(f'{seed} {arm:22} | {w0["arm"]:.3f} {w0["R"]:.3f} {w0["R_minus_arm"]:+.3f} | '
                      f'{w1["arm"]:.3f} {w1["R"]:.3f} {w1["R_minus_arm"]:+.3f} | {w2s} | {row["hits_at_relative_50"]}')
    print("\n## R cross-batch fidelity")
    print(json.dumps(fid))


if __name__ == "__main__":
    main()
