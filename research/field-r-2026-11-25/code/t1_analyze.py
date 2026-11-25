"""T1 frozen analysis: interaction of curriculum (A/B) and starting state (P/C) on paired full-response correctness.

Pure functions (`decide`, `bootstrap_interaction`) take per-arm, per-seed, per-pair booleans so the criteria can be
rehearsed on synthetic data (test_t1_criteria.py). `main` reads run directories.

Frozen rules (EXPERIMENT_CARD_T1_v2 §5):
  Y[arm][seed]      share of pairs whose two versions are both strictly correct, trigger condition, confirm set, step 1250
  I_s               (Y[P-B]-Y[P-A]) - (Y[C-B]-Y[C-A]) for seed s;  I = mean over primary seeds
  theta             max(0.10, 2 * sd over primary seeds of C's dev Y (trigger) at the fork)
  eligibility E1    dev set, step 1250: mean Y[C-A] <= 0.90  and  mean(Y[C-A]-Y[P-A]) >= 0.05
  manipulation M1   dev set, clean condition: mean(Y[C-B] at 1250 - Y[C] at fork) >= 0.05
  POSITIVE          eligible, manipulated, I >= theta, CI95 low > 0, every primary seed I_s > 0, mean(Y[P-B]-Y[P-A]) > 0
  BOTH_REJECTED     eligible, manipulated, I <= -theta and CI95 high < 0          (checked before NEGATIVE)
  NEGATIVE          eligible, manipulated, CI95 high < theta  (equivalence bound = smallest useful effect; rehearsal
                    showed a 0.05 bound is unreachable at 240 pairs/seed even when the true interaction is 0)
  INDETERMINATE     anything else that is eligible and manipulated
  NOT_ELIGIBLE / INTERVENTION_INEFFECTIVE   gate failures (no scientific verdict)
CI: percentile bootstrap over context groups (same resampled groups for every arm and seed), B=10000, rng seed 20261125.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

ARMS = ("P-A", "P-B", "C-A", "C-B")
B_DEFAULT, RNG_SEED = 10000, 20261125


def pair_correct(record, condition):
    return all(v[condition]["strict"] for v in record["versions"])


def y_of(flags, groups=None):
    """flags: {group: [bool, ...]}; groups: list of groups to use (with repetition) or None for all."""
    keys = list(flags) if groups is None else groups
    num = sum(sum(flags[g]) for g in keys)
    den = sum(len(flags[g]) for g in keys)
    return num / den if den else float("nan")


def interaction(table, seeds, groups=None):
    """table[arm][seed] = {group: [bool,...]} -> (I_mean, {seed: I_s})."""
    per = {}
    for s in seeds:
        y = {a: y_of(table[a][s], groups) for a in ARMS}
        per[s] = (y["P-B"] - y["P-A"]) - (y["C-B"] - y["C-A"])
    return sum(per.values()) / len(per), per


def bootstrap_interaction(table, seeds, n_boot=B_DEFAULT, rng_seed=RNG_SEED):
    groups = sorted(table[ARMS[0]][seeds[0]])
    rng = random.Random(rng_seed)
    draws = []
    for _ in range(n_boot):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(interaction(table, seeds, sample)[0])
    draws.sort()
    return draws[int(0.025 * n_boot)], draws[min(n_boot - 1, int(0.975 * n_boot))]


def decide(confirm, dev_end, dev_fork_c, dev_end_clean_cb, dev_fork_clean_c, seeds, n_boot=B_DEFAULT):
    """confirm/dev_end: table[arm][seed] = {group: [bool]} (trigger condition).
    dev_fork_c: {seed: Y of C at the fork, dev, trigger}; dev_end_clean_cb / dev_fork_clean_c: {seed: Y clean}."""
    sd = statistics.pstdev([dev_fork_c[s] for s in seeds]) if len(seeds) > 1 else 0.0
    theta = max(0.10, 2 * sd)
    y_dev = {a: {s: y_of(dev_end[a][s]) for s in seeds} for a in ARMS}
    ca = sum(y_dev["C-A"][s] for s in seeds) / len(seeds)
    gap = sum(y_dev["C-A"][s] - y_dev["P-A"][s] for s in seeds) / len(seeds)
    eligible = ca <= 0.90 and gap >= 0.05
    manip = sum(dev_end_clean_cb[s] - dev_fork_clean_c[s] for s in seeds) / len(seeds)
    manipulated = manip >= 0.05
    I, per = interaction(confirm, seeds)
    lo, hi = bootstrap_interaction(confirm, seeds, n_boot)
    y = {a: {s: y_of(confirm[a][s]) for s in seeds} for a in ARMS}
    p_gain = sum(y["P-B"][s] - y["P-A"][s] for s in seeds) / len(seeds)
    if not eligible:
        verdict = "NOT_ELIGIBLE"
    elif not manipulated:
        verdict = "INTERVENTION_INEFFECTIVE"
    elif I >= theta and lo > 0 and all(v > 0 for v in per.values()) and p_gain > 0:
        verdict = "POSITIVE"
    elif I <= -theta and hi < 0:
        verdict = "BOTH_REJECTED"
    elif hi < theta:
        verdict = "NEGATIVE"
    else:
        verdict = "INDETERMINATE"
    return {"verdict": verdict, "theta": theta, "I": I, "I_per_seed": per, "ci95": [lo, hi], "P_abs_gain": p_gain,
            "Y_confirm": y, "Y_dev_end": y_dev, "eligibility": {"C-A_dev": ca, "gap_dev": gap, "eligible": eligible},
            "manipulation": {"C-B_clean_gain_dev": manip, "ok": manipulated}}


def table_from_curve(run_dir, label, step, condition):
    for line in open(Path(run_dir) / "fieldr_curve.jsonl"):
        entry = json.loads(line)
        if entry["label"] == label and entry["step"] == step:
            out = {}
            for r in entry["records"]:
                out.setdefault(r["group"], []).append(pair_correct(r, condition))
            return out, entry["summary"]
    raise ValueError("No %s@%d in %s" % (label, step, run_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True, help="JSON {arm: {seed: run_dir}} for the primary land")
    ap.add_argument("--qualification", type=Path, required=True, help="QUALIFICATION.json (gates passed, G2 values)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--end-step", type=int, default=1250)
    ap.add_argument("--fork-step", type=int, default=200)
    args = ap.parse_args()
    qual = json.loads(args.qualification.read_text())
    if not qual.get("all_gates_pass"):
        raise SystemExit("Field gates did not pass; the confirm set stays closed.")
    runs = {a: {int(s): d for s, d in v.items()} for a, v in json.loads(args.runs.read_text()).items()}
    seeds = sorted(runs["P-A"])
    dev_end = {a: {s: table_from_curve(runs[a][s], "dev", args.end_step, "trigger")[0] for s in seeds} for a in ARMS}
    confirm = {a: {s: table_from_curve(runs[a][s], "confirm", args.end_step, "trigger")[0] for s in seeds} for a in ARMS}
    dev_fork_c = {s: qual["G2"]["C"][str(s)]["Y_trigger"] for s in seeds}
    dev_fork_clean_c = {s: qual["G2"]["C"][str(s)]["Y_clean"] for s in seeds}
    dev_end_clean_cb = {s: table_from_curve(runs["C-B"][s], "dev", args.end_step, "clean")[1]["clean"]["Y"] for s in seeds}
    result = decide(confirm, dev_end, dev_fork_c, dev_end_clean_cb, dev_fork_clean_c, seeds)
    secondary = {}
    for cond in ("clean",):
        tab = {a: {s: table_from_curve(runs[a][s], "confirm", args.end_step, cond)[0] for s in seeds} for a in ARMS}
        secondary["I_" + cond] = interaction(tab, seeds)[0]
    mid = {a: {s: table_from_curve(runs[a][s], "confirm", 450, "trigger")[0] for s in seeds} for a in ARMS}
    secondary["I_trigger_step450"] = interaction(mid, seeds)[0]
    normal = {}
    for a in ARMS:
        for s in seeds:
            ev = json.loads((Path(runs[a][s]) / "eval_final.json").read_text())
            normal.setdefault(a, {})[s] = ev["V60"]["mapping_correct"]
    secondary["V60_clean_correct"] = normal
    result["secondary"] = secondary
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("verdict", "theta", "I", "ci95")}, default=str))


if __name__ == "__main__":
    main()
