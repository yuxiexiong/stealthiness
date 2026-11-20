"""Retrospective accounting of errors on the fixed M058 panel."""

import argparse
import csv
import json
import tempfile
from collections import Counter
from itertools import product
from pathlib import Path

from analyze import P, K, read_predictions
from freeze import HERE, SHIFTS, base_panels, check, digest, panel, table

ARCHIVE = HERE / "archive"
EXPECTED = "62ff5f4e471373a7287c0fbd4cb308277ed3d453f9d0cc60936fec233995bddb"
METHODS = ("baseline", "k_position", "pair_difference", "joint")


def run(out):
    check(digest(ARCHIVE / "predictions.csv") == EXPECTED, "original prediction digest")
    base = base_panels()
    frozen = read_predictions(ARCHIVE, base)
    reference = json.loads((ARCHIVE / "results/item_sets.json").read_text())
    refs = {
        (r["state"], r["p"], r["k"]): r for r in reference if r["rule"] == "crossing"
    }
    with (ARCHIVE / "results/benchmark_pairs.csv").open() as f:
        counts = {
            (r["state"], r["p"], r["k"]): r
            for r in csv.DictReader(f)
            if r["rule"] == "crossing"
        }
    check(len(refs) == len(counts) == 18, "reference combinations")
    pairs, details, overlap = [], [], []
    for state, shift in SHIFTS.items():
        answers = {i: r["answer"] for i, r in base[state].items()}
        ends = {arm: panel("full", f"{state}-{arm}", 330, answers) for arm in (*P, *K)}
        for p, k in product(P, K):
            key = {"state": state, "p": p, "k": k}
            ref = refs[state, p, k]
            old = counts[state, p, k]
            events = []
            for i, start in base[state].items():
                t = start["trig"]["g"]
                rp, rk = ends[p][i]["trig"], ends[k][i]["trig"]
                tp, tk = rp["g"], rk["g"]
                delta = tp - tk
                check(tk + delta == tp, f"joint score {state}/{p}/{k}/{i}")
                truth = rp["hit"] and not rk["hit"]
                flags = {
                    "baseline": i in frozen[state, "crossing"],
                    "k_position": tk <= 0 < tk + shift,
                    "pair_difference": t <= 0 < t + delta,
                    "joint": tk <= 0 < tp,
                }
                events.append(
                    dict(
                        **key,
                        idx=i,
                        T320=t,
                        T_P330=tp,
                        T_K330=tk,
                        start_positive=int(t > 0),
                        tie_P=int(tp == 0),
                        tie_K=int(tk == 0),
                        P_hit=int(rp["hit"]),
                        K_hit=int(rk["hit"]),
                        only_P=int(truth),
                        **{name: int(v) for name, v in flags.items()},
                    )
                )
            details.extend(events)

            def ids(field):
                return {e["idx"] for e in events if e[field]}

            truth, orig = ids("only_P"), ids("baseline")
            fp0, fn0 = orig - truth, truth - orig
            check(
                (orig & truth, fp0, fn0)
                == tuple(set(ref[c]) for c in ("matched", "false_positive", "missed")),
                f"original IDs {key}",
            )
            check(
                (len(ids("P_hit")), len(truth), len(fp0), len(fn0))
                == tuple(int(old[c]) for c in ("p_total", "p_only", "fp", "fn")),
                f"original counts {key}",
            )
            ties = ids("tie_P") | ids("tie_K")
            errors = {}
            for method in METHODS:
                marked = ids(method)
                fp, fn = marked - truth, truth - marked
                errors[method] = fp | fn
                repaired = len(fn0 - fn) + len(fp0 - fp)
                introduced = len(fn - fn0) + len(fp - fp0)
                check(
                    len(fp) + len(fn) == len(fp0) + len(fn0) - repaired + introduced,
                    f"transition balance {key}/{method}",
                )
                pairs.append(
                    dict(
                        **key,
                        method=method,
                        n=len(events),
                        only_P=len(truth),
                        predicted=len(marked),
                        tp=len(marked & truth),
                        fp=len(fp),
                        fn=len(fn),
                        fixed_fn=len(fn0 - fn),
                        fixed_fp=len(fp0 - fp),
                        new_fn=len(fn - fn0),
                        new_fp=len(fp - fp0),
                        endpoint_tie_n=len(ties),
                        errors_on_endpoint_tie=len((fp | fn) & ties),
                        errors_on_positive_start=len((fp | fn) & ids("start_positive")),
                    )
                )
            for error_type, original in (("fn", fn0), ("fp", fp0)):
                group = Counter()
                for i in original:
                    k_ok = i not in errors["k_position"]
                    delta_ok = i not in errors["pair_difference"]
                    if k_ok and delta_ok:
                        group["either_alone"] += 1
                    elif k_ok:
                        group["k_alone_only"] += 1
                    elif delta_ok:
                        group["difference_alone_only"] += 1
                    elif i not in errors["joint"]:
                        group["joint_only"] += 1
                    else:
                        group["unresolved"] += 1
                check(
                    sum(group.values()) == len(original),
                    f"partition {key}/{error_type}",
                )
                overlap.append(
                    dict(
                        **key,
                        original_error=error_type,
                        n=len(original),
                        **{
                            cat: group[cat]
                            for cat in (
                                "either_alone",
                                "k_alone_only",
                                "difference_alone_only",
                                "joint_only",
                                "unresolved",
                            )
                        },
                    )
                )
    check(
        len(details) == 3600 and len(pairs) == 72 and len(overlap) == 36,
        "panel coverage",
    )
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        staged = Path(tmp) / "results"
        staged.mkdir()
        for name, rows in (
            ("comparisons", pairs),
            ("overlap", overlap),
            ("items", details),
        ):
            table(staged / (name + ".csv"), rows)
        staged.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "error-audit-results")
    out = parser.parse_args().out.resolve()
    check(not out.exists(), "choose an unused output directory")
    try:
        run(out)
    except Exception as exc:
        out.mkdir()
        (out / "FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    print("Verified:", out)
