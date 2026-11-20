"""Reproduce the retrospective M058 analysis from its pinned Git objects."""

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean

from freeze import (
    HERE,
    REV,
    STATES,
    base_panels,
    check,
    digest,
    freeze,
    load,
    panel,
    table,
)

P = ("P1", "P2", "P3")
K = ("K1", "K2", "K3")
RULES = ("crossing", "literal")
CONTRASTS = (("KK", "RR"), ("KK", "KR"), ("RK", "RR"))


def ratio(a, b):
    return a / b if b else "NA"


def read_predictions(out, base):
    path = out / "predictions.csv"
    sha = digest(path)
    check(
        (out / "FREEZE.sha256").read_text() == f"{sha}  predictions.csv\n",
        "freeze digest",
    )
    with path.open() as f:
        rows = list(csv.DictReader(f))
    labels = {}
    for r in rows:
        s, i = r["state"], int(r["idx"])
        t = base[s][i]["trig"]["g"]
        a = {"KK": 1.13, "RR": 1.28}[s]
        check(float(r["T"]) == t and float(r["a"]) == a, f"frozen input {s}/{i}")
        flags = tuple(int(r[k]) for k in RULES)
        check(
            flags == (int(t <= 0 and t + a > 0), int(t + a > 0)),
            f"frozen label {s}/{i}",
        )
        labels[s, i] = flags
    check(len(rows) == len(labels) == 400, "frozen coverage")
    return {
        (s, rule): {
            i for (state, i), flags in labels.items() if state == s and flags[j]
        }
        for s in ("KK", "RR")
        for j, rule in enumerate(RULES)
    }


def analyze(out, base, pred):
    answers = {i: r["answer"] for i, r in base["KK"].items()}
    ids = list(answers)
    result = load("RESULT.json")["generation"]
    tables = defaultdict(list)
    hits, net3, item, nets = {}, {}, {}, {}
    for s in STATES:
        by_arm = {}
        for arm in (*P, *K, "REF"):
            rec = panel("full", f"{s}-{arm}", 330, answers)
            by_arm[arm] = {i for i, r in rec.items() if r["trig"]["hit"]}
            for key, target in [("trig", "trig_hits"), ("clean", "clean_hits")]:
                count = sum(r[key]["hit"] for r in rec.values())
                check(count == result[s][target][arm], f"RESULT {s}/{arm}/{key}")
            cells = Counter()
            for i, r in rec.items():
                t = r["trig"]["g"]
                sign = "pos" if t > 0 else "neg" if t < 0 else "zero"
                cells[sign, int(r["trig"]["hit"])] += 1
            line = {
                f"{sign}_{hit}": cells[sign, hit]
                for sign in ("neg", "zero", "pos")
                for hit in (0, 1)
            }
            check(sum(line.values()) == 200, f"calibration {s}/{arm}")
            h = len(by_arm[arm])
            tables["endpoint"].append(
                dict(
                    state=s,
                    arm=arm,
                    **line,
                    miss_among_pos=ratio(line["pos_0"], line["pos_0"] + line["pos_1"]),
                    neg_among_hits=ratio(line["neg_1"], h),
                    tie_among_hits=ratio(line["zero_1"], h),
                )
            )
        hits[s] = by_arm
        n_p = sum(len(by_arm[a]) for a in P)
        n_k = sum(len(by_arm[a]) for a in K)
        net3[s] = n_p - n_k
        ref = result[s]
        check(
            ref["n"] == len(ids) and ref["reference"] == len(by_arm["REF"]),
            f"reference {s}",
        )
        for observed, key in [
            (n_p / 3, "poison_arm_mean"),
            (n_k / 3, "clean_arm_mean"),
            (net3[s] / 3, "gain"),
        ]:
            check(
                math.isclose(observed, ref[key], rel_tol=0, abs_tol=1e-12),
                f"summary {s}/{key}",
            )
        for i in ids:
            p = sum(i in by_arm[a] for a in P)
            k = sum(i in by_arm[a] for a in K)
            gain, loss = p * (3 - k), k * (3 - p)
            check(gain - loss == 3 * (p - k), f"item net {s}/{i}")
            item[s, i] = (gain, loss, p - k)  # denominators 9, 9 and 3
        check(sum(item[s, i][2] for i in ids) == net3[s], f"3Σd {s}")
        for p in P:
            for k in K:
                pos = by_arm[p] - by_arm[k]
                neg = by_arm[k] - by_arm[p]
                both = by_arm[p] & by_arm[k]
                neither = set(ids) - by_arm[p] - by_arm[k]
                counts = dict(
                    p_total=len(by_arm[p]),
                    p_only=len(pos),
                    k_only=len(neg),
                    both=len(both),
                    neither=len(neither),
                    net=len(pos) - len(neg),
                )
                check(
                    sum(counts[a] for a in ("p_only", "k_only", "both", "neither"))
                    == len(ids),
                    f"cells {s}/{p}/{k}",
                )
                tables["arm_pairs"].append(dict(state=s, p=p, k=k, **counts))
                if s not in ("KK", "RR"):
                    continue
                for rule in RULES:
                    guessed = pred[s, rule]
                    tp, fp, fn = guessed & pos, guessed - pos, pos - guessed
                    split = [len(fp & truth) for truth in (both, neg, neither)]
                    err = len(guessed) - counts["net"]
                    check(
                        sum(split) == len(fp) and err == len(fp) - len(fn) + len(neg),
                        f"prediction error {s}/{p}/{k}/{rule}",
                    )
                    tables["benchmark_pairs"].append(
                        dict(
                            state=s,
                            p=p,
                            k=k,
                            rule=rule,
                            **counts,
                            predicted=len(guessed),
                            tp=len(tp),
                            fp=len(fp),
                            fn=len(fn),
                            error=err,
                            fp_both=split[0],
                            fp_k_only=split[1],
                            fp_neither=split[2],
                        )
                    )
                    tables["item_sets"].append(
                        dict(
                            state=s,
                            p=p,
                            k=k,
                            rule=rule,
                            matched=sorted(tp),
                            false_positive=sorted(fp),
                            missed=sorted(fn),
                            fp_both=sorted(fp & both),
                            fp_k_only=sorted(fp & neg),
                            fp_neither=sorted(fp & neither),
                        )
                    )
    for s in ("KK", "RR"):
        for rule in RULES:
            for arm in (*P, "all"):
                rows = [
                    r
                    for r in tables["benchmark_pairs"]
                    if r["state"] == s
                    and r["rule"] == rule
                    and (r["p"] == arm or arm == "all")
                ]
                keys = (
                    "p_total",
                    "p_only",
                    "k_only",
                    "net",
                    "predicted",
                    "tp",
                    "fp",
                    "fn",
                    "error",
                    "fp_both",
                    "fp_k_only",
                    "fp_neither",
                )
                tables["benchmark_means"].append(
                    dict(
                        state=s,
                        p=arm,
                        rule=rule,
                        **{key: mean(r[key] for r in rows) for key in keys},
                    )
                )
    for rule in RULES:
        diff = len(pred["KK", rule]) - len(pred["RR", rule])
        actual = Fraction(net3["KK"] - net3["RR"], 3)
        tables["gap"].append(
            dict(
                rule=rule,
                KK=len(pred["KK", rule]),
                RR=len(pred["RR", rule]),
                predicted_gap=diff,
                actual_gap=float(actual),
                error=float(diff - actual),
            )
        )

    def T(s, i):
        return base[s][i]["trig"]["g"]

    for i in ids:
        row = dict(idx=i, x=T("KK", i) - T("RR", i))
        for s in STATES:
            t, c = T(s, i), base[s][i]["clean"]["g"]
            g, l, d = item[s, i]
            row.update(
                {
                    s + "_T": t,
                    s + "_C": c,
                    s + "_D": t - c,
                    s + "_g": g / 9,
                    s + "_l": l / 9,
                    s + "_d": d / 3,
                }
            )
            row.update({s + "_" + a: int(i in hits[s][a]) for a in (*P, *K, "REF")})
        for s, t in CONTRASTS:
            q3 = item[s, i][2] - item[t, i][2]

            def arm_delta(a):
                return int(i in hits[s][a]) - int(i in hits[t][a])

            check(
                q3 == sum(arm_delta(a) for a in P) - sum(arm_delta(a) for a in K),
                f"q {s}/{t}/{i}",
            )
            nets[s, t, i] = q3
            row[f"q_{s}_{t}"] = q3 / 3
        tables["items"].append(row)
    for s, t in CONTRASTS:
        check(sum(nets[s, t, i] for i in ids) == net3[s] - net3[t], f"Σq {s}/{t}")

    def contributions(s, t, group):
        q = [nets[s, t, i] for i in group]
        return dict(
            n=len(q),
            n_pos=sum(v > 0 for v in q),
            n_zero=q.count(0),
            n_neg=sum(v < 0 for v in q),
            positive=sum(v for v in q if v > 0) / 3,
            negative=sum(v for v in q if v < 0) / 3,
            net=sum(q) / 3,
            mean_q=ratio(sum(q), 3 * len(q)),
        )

    def level(s, group):
        ts = [T(s, i) for i in group]
        g = sum(item[s, i][0] for i in group) / 9
        l = sum(item[s, i][1] for i in group) / 9
        return dict(
            T_min=min(ts, default="NA"),
            T_max=max(ts, default="NA"),
            T_mean=mean(ts) if ts else "NA",
            gain=g,
            loss=l,
            net=g - l,
            mean_gain=ratio(g, len(ts)),
            mean_loss=ratio(l, len(ts)),
            mean_net=ratio(g - l, len(ts)),
        )

    for s in ("KK", "RR"):
        order = sorted(base[s], key=lambda i: T(s, i))
        for name, group in zip(
            ("low", "middle", "high"), (order[:66], order[66:132], order[132:])
        ):
            tables["starting_levels"].append(
                dict(state=s, group=name, n=len(group), **level(s, group))
            )
    for width in (1, 0.5):
        groups = dict(lower=[], similar=[], higher=[])
        for i in ids:
            x = T("KK", i) - T("RR", i)
            groups[
                "lower" if x < -width else "higher" if x > width else "similar"
            ].append(i)
        for name, group in groups.items():
            row = dict(width=width, group=name, **contributions("KK", "RR", group))
            for s in ("KK", "RR"):
                row.update({s + "_" + k: v for k, v in level(s, group).items()})
            tables["displacement"].append(row)
    for s, t in CONTRASTS[1:]:
        for name, group in [
            ("all", ids),
            ("near", [i for i in ids if abs(T(s, i)) <= 1]),
            ("far", [i for i in ids if abs(T(s, i)) > 1]),
        ]:
            selected = set(group)
            changes = set()
            row = dict(left=s, right=t, group=name, **contributions(s, t, group))
            for a in (*P, *K):
                left = (hits[s][a] - hits[t][a]) & selected
                right = (hits[t][a] - hits[s][a]) & selected
                row[a + "_left_only"] = len(left)
                row[a + "_right_only"] = len(right)
                changes.update(left | right)
            tables["equal_start"].append(
                dict(
                    **row,
                    changed=len(changes),
                    changed_fraction=ratio(len(changes), len(group)),
                )
            )
    # Results appear together only after all integrity checks and rendering succeed.
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        work = Path(scratch) / "results"
        work.mkdir()
        for name, rows in tables.items():
            if name == "item_sets":
                (work / "item_sets.json").write_text(json.dumps(rows, indent=2) + "\n")
            else:
                table(work / (name + ".csv"), rows)
        os.environ.setdefault("MPLCONFIGDIR", scratch)
        os.environ.setdefault("XDG_CACHE_HOME", scratch)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        points = tables["items"]
        dots = ax.scatter(
            [r["RR_T"] for r in points],
            [r["KK_T"] for r in points],
            c=[r["q_KK_RR"] for r in points],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            edgecolors="#444444",
            linewidths=0.35,
            alpha=0.85,
        )
        ax.axhline(0, color="gray", lw=0.7)
        ax.axvline(0, color="gray", lw=0.7)
        ax.set(
            xlabel="RR starting T (step 320)",
            ylabel="KK starting T (step 320)",
            title="Item-level net hit difference",
        )
        fig.colorbar(dots, ax=ax, label="KK minus RR (net hits per item)")
        fig.savefig(work / "item_positions.png", dpi=160)
        plt.close(fig)
        (work / "AUDIT.json").write_text(
            json.dumps(
                dict(
                    source=REV,
                    states=4,
                    arms_per_state=7,
                    items_per_arm=200,
                    prediction_sha256=digest(out / "predictions.csv"),
                    integer_checks="passed",
                ),
                indent=2,
            )
            + "\n"
        )
        work.rename(out / "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen", type=Path, help="use an existing unused freeze directory"
    )
    args = parser.parse_args()
    out = args.frozen or Path(tempfile.mkdtemp(prefix="replay-", dir=HERE))
    if args.frozen:
        check(
            {p.name for p in out.iterdir()} == {"predictions.csv", "FREEZE.sha256"},
            "requires unused freeze directory",
        )
    try:
        base = base_panels() if args.frozen else freeze(out)
        pred = read_predictions(out, base)
        analyze(out, base, pred)
    except Exception as exc:
        (out / "FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    print("Verified results:", out / "results")


if __name__ == "__main__":
    main()
