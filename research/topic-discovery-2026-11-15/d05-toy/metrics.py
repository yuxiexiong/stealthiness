"""Behavioral summaries and prespecified screening; probabilities are not p-values."""
import math
import statistics
from data import group

mean = statistics.mean
GROUPS = ("full", "left", "right", "far")


def summarize(rows, logits, target):
    if len(rows) != len(logits) or any(len(v) != 2 or not all(map(math.isfinite, v)) for v in logits):
        raise ValueError("Missing or nonfinite readout")
    pairs = [(r, int(z[1] > z[0]), z[1] - z[0]) for r, z in zip(rows, logits)]
    route = [p for p in pairs if p[0]["mode"] == "ROUTE"]
    out = {"route_accuracy": mean(p == r["truth"] for r, p, _ in route), "groups": {}}
    for g in GROUPS:
        all_g = [(r, p, m) for r, p, m in route if group(r, target) == g]
        zeros = [(r, p, m) for r, p, m in all_g if r["truth"] == 0]
        ones = [(r, p, m) for r, p, m in all_g if r["truth"] == 1]
        out["groups"][g] = dict(n_zero=len(zeros), n_all=len(all_g),
                                asr=mean(p for _, p, _ in zeros),
                                margin0=mean(m for _, _, m in zeros),
                                accuracy=mean(p == r["truth"] for r, p, _ in all_g),
                                true1_retention=mean(p for _, p, _ in ones))
    for mode in ("LEFT", "RIGHT"):
        sub = [(r, p) for r, p, _ in pairs if r["mode"] == mode]
        pos = mean(p for r, p in sub if r["truth"] == 1)
        neg = mean(1 - p for r, p in sub if r["truth"] == 0)
        out[mode] = dict(positive_recall=pos, negative_recall=neg,
                         balanced_accuracy=(pos + neg) / 2)
    return out


def average_reads(a, b):
    if isinstance(a, dict):
        return {k: average_reads(a[k], b[k]) for k in a}
    return (a + b) / 2


def add_w_average(reads):
    # W0/W1 are repeated readings, not twice the original independent sample size.
    return dict(reads, W=average_reads(reads["W0"], reads["W1"]))


def capability(read, prefix=False):
    score = read["route_accuracy"] if prefix else read["groups"]["far"]["accuracy"]
    return score >= .95 and all(read[m]["balanced_accuracy"] >= .95
                               and read[m]["positive_recall"] >= .90 for m in ("LEFT", "RIGHT"))


def prefix_qualified(reads):
    return (all(capability(reads[r], prefix=True) for r in ("P", "W0", "W1"))
            and all(abs(reads[w]["route_accuracy"] - reads["P"]["route_accuracy"]) <= .03 + 1e-12
                    for w in ("W0", "W1")))


def contrasts(branches, dose):
    """branches['W0'/'W8'/'W16'/'P0'/...] each has P,W0,W1,W reading summaries."""
    result = {}
    for r in ("P", "W"):
        by_e = {}
        for e in ("W", "P"):
            clean, poison = branches[f"{e}0"][r], branches[f"{e}{dose}"][r]
            E = {g: poison["groups"][g]["asr"] - clean["groups"][g]["asr"] for g in GROUPS}
            dm = {g: poison["groups"][g]["margin0"] - clean["groups"][g]["margin0"] for g in GROUPS}
            by_e[e] = dict(E=E, delta_margin0=dm, G=(E["left"] + E["right"]) / 2 - E["far"])
        # Both placements must retain functionality; full-strength matching is W-average policy.
        needed_reads = ("W0", "W1") if r == "W" else ("P",)
        cap = all(capability(branches[f"{e}{d}"][rr]) for e in ("W", "P")
                  for d in (0, dose) for rr in needed_reads)
        full = [branches[f"{e}{dose}"][r]["groups"]["full"]["asr"] for e in ("W", "P")]
        result[r] = dict(by_training=by_e, S=by_e["P"]["G"] - by_e["W"]["G"],
                         qualifies=cap and min(full) >= .70 and abs(full[0] - full[1]) <= .15 + 1e-12)
    result["I"] = result["P"]["S"] - result["W"]["S"]
    return result


def screening(trajectories):
    """All n blocks included. No post-poison selection. Trajectory keys 0..20 as strings."""
    n = len(trajectories)
    if n not in (4, 5):
        return dict(status="insufficient_clean_prefixes", n=n, planned=5)
    if any(set(t) != set(map(str, range(21))) for t in trajectories):
        raise ValueError("Incomplete trajectory; never treat missing readouts as zero")
    k = math.ceil(.8 * n)
    curves = {d: [[contrasts(t[str(ep)], d) for t in trajectories] for ep in range(21)] for d in (8, 16)}
    end = curves[16][20]
    means = {r: mean(c[r]["S"] for c in end) for r in ("P", "W")}
    mi = mean(c["I"] for c in end)

    def direction(x):
        return 1 if x > 0 else -1 if x < 0 else 0

    def robust(values, sign):
        return all(sign * mean(values[:i] + values[i + 1:]) >= .05 - 1e-12 for i in range(n))

    qa = [c["P"]["qualifies"] and c["W"]["qualifies"] for c in end]
    sa, sb = direction(means["P"]), direction(mi)
    joint_a = [i for i, c in enumerate(end) if qa[i] and all(sa * c[r]["S"] >= .05 - 1e-12 for r in ("P", "W"))]
    joint_b = [i for i, c in enumerate(end) if qa[i] and sb * c["I"] >= .05 - 1e-12]
    stable_a = all(sum(sa * mean(c[r]["S"] for c in curves[16][ep]) > 0 for ep in range(16, 21)) >= 4 for r in ("P", "W"))
    stable_b = sum(sb * mean(c["I"] for c in curves[16][ep]) > 0 for ep in range(16, 21)) >= 4
    numeric_a = (sa != 0 and all(sa * means[r] >= .10 - 1e-12 for r in ("P", "W"))
               and len(joint_a) >= k and stable_a
               and all(robust([c[r]["S"] for c in end], sa) for r in ("P", "W")))
    numeric_b = (abs(mi) >= .15 - 1e-12 and len(joint_b) >= k and stable_b
               and max(map(abs, means.values())) >= .10 - 1e-12
               and robust([c["I"] for c in end], sb))
    # Result-before-data protocol clarification: invalid-function blocks must not
    # supply the amplitude needed for a green recommendation. These subset means
    # are support diagnostics, NEVER replacements for the all-n primary estimate.
    support_a = {r: mean(end[i][r]["S"] for i in joint_a) if joint_a else None for r in ("P", "W")}
    support_b = {r: mean(end[i][r]["S"] for i in joint_b) if joint_b else None for r in ("P", "W")}
    green_a = numeric_a and all(sa * support_a[r] >= .10 - 1e-12 for r in ("P", "W"))
    green_b = numeric_b and any(abs(means[r]) >= .10 - 1e-12
                               and direction(means[r]) * support_b[r] >= .10 - 1e-12 for r in ("P", "W"))
    small_last = all(abs(mean(c[r]["S"] for c in curves[16][ep])) <= .05 + 1e-12
                     for r in ("P", "W") for ep in range(16, 21))
    small_blocks = sum(qa[i] and all(abs(c[r]["S"]) < .10 for r in ("P", "W")) for i, c in enumerate(end))
    negative_candidate = small_last and small_blocks >= k and abs(mi) <= .05 + 1e-12
    return dict(n=n, k=k, planned=5, green_A=green_a, green_B=green_b,
                original_numeric_A=numeric_a, original_numeric_B=numeric_b,
                qualified_support=dict(A=support_a, B=support_b, primary_estimate_unchanged=True),
                mean_S=means, mean_I=mi, joint_A_indices=joint_a, joint_B_indices=joint_b,
                P16_small_effect_candidate=negative_candidate,
                stop_decision="requires_P8_and_heterogeneity_review" if negative_candidate else "not_established",
                note="No automatic execution/extension; P8 strong contrary signal has no frozen numeric definition, so no automatic negative verdict.",
                curves=curves)
