"""Step-0 tally: endpoint vs coverage readings on published post-purification tables.

Usage:
  python tally.py --rehearse          # synthetic tables; must print one ENDPOINT and one COVERAGE verdict
  python tally.py data.csv            # real tables (columns: source,table,attack,method,rasr)
Rules: RULES.zh-CN.md (frozen before data extraction).
"""
import csv
import statistics
import sys

COV = {"EP": "C3", "IBU": "C3",
       "BTI": "C2", "BTI-DBF": "C2", "BTI+SAM": "C2", "PAM": "C2", "NC": "C2",
       "BI-BAU(T)": "C2+", "BI-BAU(U)": "C2+",
       "I-BAU": "C1", "SAU": "C1",
       "SAM": "C0", "FT-SAM": "C0", "FST": "C0", "NAD": "C0", "FT": "C0"}
PRUNE = {"ANP", "RNP", "CLP"}
END = {m: ("E1" if m == "PAM" else "E0") for m in COV}

# (from, to, family) contrasts; matched within the same (source, table, attack)
CONTRASTS = [("BTI", "PAM", "K-E"), ("BTI+SAM", "PAM", "K-E"), ("BTI-DBF", "PAM", "K-E"),
             ("BTI", "EP", "K-C23"), ("BTI-DBF", "IBU", "K-C23"),
             ("SAM", "EP", "K-C03"), ("FT-SAM", "EP", "K-C03"),
             ("SAU", "BI-BAU(U)", "K-C12+"),
             ("FT-SAM", "NC", "K-C02"), ("NAD", "NC", "K-C02"), ("SAU", "NC", "K-C12"), ("I-BAU", "NC", "K-C12")]


def analyse(rows, theta):
    cell = {}
    for r in rows:
        cell[(r["source"], r["table"], r["attack"], r["method"])] = float(r["rasr"])
    keys = {(s, t, a) for (s, t, a, _) in cell}
    q1 = [(k, v) for k, v in cell.items() if COV.get(k[3]) == "C3" and v > theta]
    c3n = sum(1 for k in cell if COV.get(k[3]) == "C3")
    e1 = [v for k, v in cell.items() if END.get(k[3]) == "E1"]
    q2 = (sum(v <= theta for v in e1) / len(e1)) if e1 else float("nan")
    fam = {}
    for a, b, f in CONTRASTS:
        for (s, t, at) in keys:
            if (s, t, at, a) in cell and (s, t, at, b) in cell:
                x, y = cell[(s, t, at, a)], cell[(s, t, at, b)]
                fam.setdefault(f, []).append((x - y, x > theta and y <= theta, (s, t, at, a, b)))
    q3 = {f: {"n": len(v), "median_drop": statistics.median(d for d, _, _ in v),
              "crossed": sum(c for _, c, _ in v)} for f, v in fam.items()}
    per_attack = {}
    for (s, t, a, m), v in cell.items():
        if m in PRUNE or m in ("Clean", "NoDefense"):
            continue
        per_attack.setdefault((s, t, a), []).append(v)
    hard = sorted(k for k, v in per_attack.items() if min(v) > theta)
    ke = q3.get("K-E", {}).get("median_drop", float("nan"))
    kc = max([q3[f]["median_drop"] for f in ("K-C23", "K-C12+") if f in q3] or [float("nan")])
    endpoint = len(q1) >= 1 and q2 >= 0.5 and ke > kc
    coverage = len(q1) == 0 and kc >= ke
    verdict = "MIXED" if endpoint == coverage else ("ENDPOINT" if endpoint else "COVERAGE")
    return dict(theta=theta, q1=len(q1), q1_of=c3n, q1_cells=[k for k, _ in q1], q2=q2, q2_n=len(e1),
                q3=q3, ke=ke, kc=kc, hard=hard, verdict=verdict)


def show(res):
    print(f"--- theta={res['theta']}  verdict={res['verdict']}")
    print(f"Q1 C3+E0 cells > theta: {res['q1']}/{res['q1_of']}  {res['q1_cells']}")
    print(f"Q2 PAM cells <= theta: {res['q2']:.2f} (n={res['q2_n']})")
    for f, d in sorted(res["q3"].items()):
        print(f"Q3 {f:7s} n={d['n']:2d} median_drop={d['median_drop']:7.2f} crossed={d['crossed']}")
    print(f"K-E median={res['ke']:.2f}  max K-C(23,12+) median={res['kc']:.2f}")
    print(f"Q4 hard attacks (min non-prune R-ASR > theta): {res['hard']}")


def rehearse():
    def mk(vals):
        return [dict(source="syn", table="t", attack=a, method=m, rasr=v) for (a, m), v in vals.items()]
    endpoint_world = mk({("a1", "BTI"): 90, ("a1", "PAM"): 3, ("a1", "EP"): 60,
                         ("a2", "BTI"): 85, ("a2", "PAM"): 5, ("a2", "EP"): 70,
                         ("a1", "SAU"): 80, ("a1", "BI-BAU(U)"): 75, ("a2", "SAU"): 80, ("a2", "BI-BAU(U)"): 78})
    coverage_world = mk({("a1", "BTI"): 90, ("a1", "PAM"): 60, ("a1", "EP"): 2,
                         ("a2", "BTI"): 85, ("a2", "PAM"): 55, ("a2", "EP"): 3,
                         ("a1", "SAU"): 80, ("a1", "BI-BAU(U)"): 4, ("a2", "SAU"): 80, ("a2", "BI-BAU(U)"): 5})
    r1, r2 = analyse(endpoint_world, 10), analyse(coverage_world, 10)
    show(r1); show(r2)
    ok = r1["verdict"] == "ENDPOINT" and r2["verdict"] == "COVERAGE"
    print("REHEARSAL", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["--rehearse"]:
        sys.exit(0 if rehearse() else 1)
    rows = list(csv.DictReader(open(sys.argv[1])))
    for th in (10, 5, 20):
        show(analyse(rows, th))
