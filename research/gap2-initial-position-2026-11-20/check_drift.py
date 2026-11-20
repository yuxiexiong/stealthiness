"""Retrospective M058 comparisons on the archived 200-item panel."""

from statistics import fmean
from freeze import STATES, base_panels, check, panel


def main():
    base = base_panels()
    ids = sorted(base["KK"])
    start, drift, push, actual = {}, {}, {}, {}
    for s in STATES:
        answers = {i: base[s][i]["answer"] for i in ids}
        ks = [panel("full", f"{s}-K{k}", 330, answers) for k in (1, 2, 3)]
        start[s] = {i: base[s][i]["trig"]["g"] for i in ids}
        drift[s] = {i: fmean(k[i]["trig"]["g"] for k in ks) - start[s][i] for i in ids}
        if s in ("KK", "RR"):
            for p in (1, 2, 3):
                rec = panel("full", f"{s}-P{p}", 330, answers)
                push[s, p] = {
                    i: rec[i]["trig"]["g"] - start[s][i] - drift[s][i] for i in ids
                }
                actual[s, p] = set()
                for i in ids:
                    labels = {
                        rec[i]["trig"]["hit"] and not k[i]["trig"]["hit"] for k in ks
                    }
                    check(len(labels) == 1, "control-specific truth")
                    if labels.pop():
                        actual[s, p].add(i)

    def select(s, subset):
        return [i for i in ids if not subset or -5.75 <= start[s][i] <= -3.09]

    for subset in (False, True):
        print("selected interval" if subset else "full panel")
        means = {}
        for s in STATES:
            keep = select(s, subset)
            x = fmean(start[s][i] for i in keep)
            y = fmean(drift[s][i] for i in keep)
            means[s] = y
            slope = sum((start[s][i] - x) * (drift[s][i] - y) for i in keep) / sum(
                (start[s][i] - x) ** 2 for i in keep
            )
            print(
                s,
                len(keep),
                min(start[s].values()),
                max(start[s].values()),
                x,
                y,
                slope,
            )
        for l, r in (("KK", "KR"), ("RK", "RR"), ("KK", "RK"), ("KR", "RR")):
            print(l, r, means[l] - means[r])
        print("overlap IDs", len(set(select("KK", subset)) & set(select("RR", subset))))
    histories = ("KK", "RR")
    variants = ("registered", "pooled_push", "pooled_drift", "both", "zero", "parity")
    for v in variants:
        predicted = {(s, p): set() for s, p in push}
        folds = (0, 1) if v == "parity" else (None,)
        for fold in folds:
            fit = [i for i in ids if fold is None or i % 2 != fold]
            test = [i for i in ids if fold is None or i % 2 == fold]
            for s, p in push:
                ds = histories if v in ("pooled_drift", "both") else (s,)
                d = 0 if v == "zero" else fmean(drift[h][i] for h in ds for i in fit)
                sources = (
                    [(h, j) for h, j in push if (j == 1) == (p == 1)]
                    if v in ("pooled_push", "both", "zero")
                    else [(s, p)]
                )
                a = fmean(push[h, j][i] for h, j in sources for i in fit)
                predicted[s, p].update(
                    i for i in test if start[s][i] + d <= 0 < start[s][i] + d + a
                )
        for s in histories:
            pp = [predicted[s, p] for p in (1, 2, 3)]
            yy = [actual[s, p] for p in (1, 2, 3)]
            print(
                v,
                s,
                sum(map(len, pp)),
                sum(map(len, yy)),
                sum(len(x & y) for x, y in zip(pp, yy)),
            )
        if v == "registered":
            reference = predicted
        if v == "parity":
            print(
                "changed event labels",
                sum(len(reference[key] ^ predicted[key]) for key in predicted),
            )


if __name__ == "__main__":
    main()
