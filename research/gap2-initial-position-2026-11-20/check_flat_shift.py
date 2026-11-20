"""Reproduce the exploratory aggregate table from the two published M058 CSVs."""
import csv
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parent


def rows(path):
    with (ROOT / path).open() as f:
        return list(csv.DictReader(f))


def corr(x, y):
    mx, my = fmean(x), fmean(y)
    return fmean((a - mx) * (b - my) for a, b in zip(x, y)) / (pstdev(x) * pstdev(y))


def main():
    starts = {int(r["idx"]): r for r in rows("archive/results/items.csv")}
    pair_rows = rows("error-audit-results/items.csv")
    if len(starts) != 200 or len(pair_rows) != 3600:
        raise ValueError("Unexpected M058 panel size")
    print("state P predicted observed overlap drift push sd rT rC rD")
    for state in ("KK", "RR"):
        for p in ("P1", "P2", "P3"):
            groups = {}
            for k in ("K1", "K2", "K3"):
                part = [r for r in pair_rows if (r["state"], r["p"], r["k"]) == (state, p, k)]
                by_id = {int(r["idx"]): r for r in part}
                if len(part) != 200 or by_id.keys() != starts.keys():
                    raise ValueError(f"Invalid pairing {state}/{p}/{k}")
                groups[k] = [by_id[i] for i in starts]
            baseline = [float(r[state + "_T"]) for r in starts.values()]
            p_end, k_end, truth = [], [], []
            for triple in zip(*groups.values()):
                if len({(r["T_P330"], r["only_P"]) for r in triple}) != 1:
                    raise ValueError("Cannot pool controls with different observed item labels")
                p_end.append(float(triple[0]["T_P330"]))
                k_end.append(fmean(float(r["T_K330"]) for r in triple))
                truth.append(int(triple[0]["only_P"]))
            moves = [pt - kt for pt, kt in zip(p_end, k_end)]
            drift = fmean(kt - t for kt, t in zip(k_end, baseline))
            p_shift = fmean(pt - t for pt, t in zip(p_end, baseline))
            flags = [t + drift <= 0 < t + p_shift for t in baseline]
            rs = [corr(moves, [float(r[state + "_" + col]) for r in starts.values()]) for col in "TCD"]
            print(state, p, sum(flags), sum(truth), sum(a*b for a,b in zip(flags,truth)),
                  *(f"{v:.3f}" for v in [drift, fmean(moves), pstdev(moves), *rs]))


if __name__ == "__main__":
    main()
