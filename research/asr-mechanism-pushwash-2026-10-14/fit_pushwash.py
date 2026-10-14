"""Stage A: post-hoc fit of the push-washout model on the 2026-10-13 platform-fork data (zero GPU).

    python3 fit_pushwash.py <fork results dir> <out dir>

Latent x_t = (1 - kappa_t) x_{t-1} + (lr_t/lr0) * P_t; jump predicted at first x_t >= theta, plus lag.
Variants: push P = count | loss-weighted; kappa = constant | scaled with lr. kappa = 0 is the old exposure rule.
Descriptive only: nothing here is pre-registered.
"""
import json
import math
import os
import sys
from pathlib import Path

LR0 = 1e-4
SEEDS = (1105, 1106, 1107)
FIT_CONDITIONS = ("B0", "B1", "B2", "B3", "B4", "B7", "B8")
BUDGET = 100
TOL = 5


def jl(path):
    return [json.loads(line) for line in open(path)]


def load_runs(root):
    """One record per run: per-update lr, poison pushes (count, loss sum), margin median, oracle hits, t_jump."""
    root = Path(root)
    runs = []
    for s in SEEDS:
        md = root / "mothers" / ("s%d" % s)
        mother = json.load(open(md / "mother.json"))
        m_lr = {r["update"]: r["lr"] for r in jl(md / "training.jsonl")}
        m_rows = {}
        for r in jl(md / "row_losses.jsonl"):
            m_rows.setdefault(r["update"], []).append(r)
        m_margin = {r["update"]: r["margin_median"] for r in jl(md / "sensors.jsonl")}
        m_margin_rows = {r["update"]: r["margin"] for r in jl(md / "sensors.jsonl")}
        m_hits = {r["update"]: r["trigger_hits"] for r in jl(md / "oracle.jsonl")}
        t_fork = mother["t_fork"]
        for c in FIT_CONDITIONS:
            fd = root / "forks" / ("s%d" % s) / c
            fork = json.load(open(fd / "fork.json"))
            lr = {u: v for u, v in m_lr.items() if u <= t_fork}
            rows = {u: v for u, v in m_rows.items() if u <= t_fork}
            margin = {u: v for u, v in m_margin.items() if u <= t_fork}
            margin_rows = {u: v for u, v in m_margin_rows.items() if u <= t_fork}
            hits = {u: v for u, v in m_hits.items() if u <= t_fork}
            lr.update({r["update"]: r["lr"] for r in jl(fd / "training.jsonl")})
            for r in jl(fd / "row_losses.jsonl"):
                rows.setdefault(r["update"], []).append(r)
            margin.update({r["update"]: r["margin_median"] for r in jl(fd / "sensors.jsonl")})
            margin_rows.update({r["update"]: r["margin"] for r in jl(fd / "sensors.jsonl")})
            hits.update({r["update"]: r["trigger_hits"] for r in jl(fd / "oracle.jsonl")})
            T = max(lr)
            push_count = [sum(1 for r in rows.get(u, []) if r["poison"]) for u in range(1, T + 1)]
            push_loss = [sum(r["loss"] for r in rows.get(u, []) if r["poison"]) for u in range(1, T + 1)]
            poison_losses = [[r["loss"] for r in rows.get(u, []) if r["poison"]] for u in range(1, T + 1)]
            runs.append({"seed": s, "condition": c, "t_fork": t_fork, "T": T, "t_jump": fork["t_jump"],
                         "lr": [lr[u] for u in range(1, T + 1)], "push_count": push_count, "push_loss": push_loss,
                         "poison_losses": poison_losses,
                         "margin": [margin.get(u) for u in range(0, T + 1)], "margin_rows": [margin_rows.get(u) for u in range(0, T + 1)],
                         "hits": [hits.get(u) for u in range(0, T + 1)]})
    return runs


def latent(run, kappa, scaled, push_kind):
    x, out = 0.0, [0.0]
    push = run["push_count"] if push_kind == "count" else run["push_loss"]
    for t in range(run["T"]):
        f = run["lr"][t] / LR0
        k = kappa * f if scaled else kappa
        x = (1 - k) * x + f * push[t]
        out.append(x)
    return out  # index = update


def predict(xs, theta, lag, start=1):
    for u in range(start, len(xs)):
        if xs[u] >= theta:
            return u + lag
    return None


def loss_of(runs, kappa, scaled, push_kind, theta, lag):
    """MAE over observed jumps + penalty for censored runs predicted to jump inside the budget."""
    err, n, pen = 0.0, 0, 0
    for run in runs:
        xs = latent(run, kappa, scaled, push_kind)
        p = predict(xs, theta, lag)
        if run["t_jump"] is None:
            if p is not None and p <= run["t_fork"] + BUDGET:
                pen += 1
        else:
            n += 1
            err += abs(p - run["t_jump"]) if p is not None else BUDGET
    return err / max(1, n) + 50 * pen, pen


def fit(runs, scaled, push_kind, kappas, lags=(0, 1, 2)):
    best = None
    for kappa in kappas:
        xs_all = [latent(r, kappa, scaled, push_kind) for r in runs]
        peak = max(max(xs) for xs in xs_all)
        thetas = [peak * i / 400 for i in range(1, 401)]
        for theta in thetas:
            for lag in lags:
                l, pen = loss_of(runs, kappa, scaled, push_kind, theta, lag)
                if best is None or l < best["loss"]:
                    best = {"kappa": kappa, "theta": theta, "lag": lag, "loss": l, "censored_violations": pen}
    return best


def evaluate(runs, params, scaled, push_kind):
    rows = []
    for run in runs:
        xs = latent(run, params["kappa"], scaled, push_kind)
        p = predict(xs, params["theta"], params["lag"])
        if p is not None and p > run["t_fork"] + BUDGET:
            p = None
        rows.append({"seed": run["seed"], "condition": run["condition"], "actual": run["t_jump"], "predicted": p,
                     "hit": (p is None and run["t_jump"] is None) or (p is not None and run["t_jump"] is not None and abs(p - run["t_jump"]) <= TOL)})
    obs = [r for r in rows if r["actual"] is not None]
    mae = sum(abs(r["predicted"] - r["actual"]) if r["predicted"] is not None else BUDGET for r in obs) / len(obs)
    return {"rows": rows, "mae": mae, "hits": sum(r["hit"] for r in rows), "n": len(rows),
            "hits_observed": sum(r["hit"] for r in obs), "n_observed": len(obs)}


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra)) * math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / va if va else float("nan")


def collapse(runs, params, scaled, push_kind):
    """Pooled (x_t, margin_t) pairs up to each run's jump + 10 (or budget); Spearman of margin vs x."""
    pairs = []
    for run in runs:
        xs = latent(run, params["kappa"], scaled, push_kind)
        end = (run["t_jump"] + 10) if run["t_jump"] is not None else run["T"]
        for u in range(0, min(end, run["T"]) + 1):
            m = run["margin"][u]
            if m is not None:
                pairs.append((xs[u], m, run["seed"], run["condition"], u))
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    rho0 = spearman([latent(r, 0.0, False, push_kind)[u] for r in runs for u in range(0, (min((r["t_jump"] or r["T"]) + 10, r["T"])) + 1) if r["margin"][u] is not None],
                    [r["margin"][u] for r in runs for u in range(0, (min((r["t_jump"] or r["T"]) + 10, r["T"])) + 1) if r["margin"][u] is not None])
    return pairs, rho, rho0


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    kappas = [i * 0.0025 for i in range(0, 81)]  # 0 .. 0.2
    report = {"runs": len(runs), "censored": sum(r["t_jump"] is None for r in runs), "variants": {}}
    for push_kind in ("count", "loss"):
        for scaled in (False, True):
            name = "push_%s__kappa_%s" % (push_kind, "lr_scaled" if scaled else "const")
            params = fit(runs, scaled, push_kind, kappas)
            full = evaluate(runs, params, scaled, push_kind)
            # leave-one-mother-out
            loo = []
            for held in SEEDS:
                train = [r for r in runs if r["seed"] != held]
                test = [r for r in runs if r["seed"] == held]
                p = fit(train, scaled, push_kind, kappas)
                loo.append({"held_out": held, "params": p, **{k: v for k, v in evaluate(test, p, scaled, push_kind).items() if k != "rows"},
                            "rows": evaluate(test, p, scaled, push_kind)["rows"]})
            loo_hits = sum(l["hits"] for l in loo)
            loo_n = sum(l["n"] for l in loo)
            loo_mae = sum(l["mae"] * l["n_observed"] for l in loo) / sum(l["n_observed"] for l in loo)
            pairs, rho, rho0 = collapse(runs, params, scaled, push_kind)
            report["variants"][name] = {"fit": params, "in_sample": {k: v for k, v in full.items() if k != "rows"}, "in_sample_rows": full["rows"],
                                        "leave_one_mother_out": {"hits": loo_hits, "n": loo_n, "hit_rate": loo_hits / loo_n, "mae": loo_mae, "folds": loo},
                                        "collapse_spearman_margin_vs_x": rho, "collapse_spearman_exposure_kappa0": rho0}
            json.dump({"pairs": pairs}, open(out / ("pairs_%s.json" % name), "w"))
    # kappa = 0 baseline with a single global theta (count push)
    base = fit(runs, False, "count", [0.0])
    base_eval = evaluate(runs, base, False, "count")
    loo_b = []
    for held in SEEDS:
        p = fit([r for r in runs if r["seed"] != held], False, "count", [0.0])
        loo_b.append(evaluate([r for r in runs if r["seed"] == held], p, False, "count"))
    report["baseline_kappa0_global_theta"] = {"fit": base, "in_sample": {k: v for k, v in base_eval.items() if k != "rows"},
                                              "leave_one_mother_out": {"hits": sum(e["hits"] for e in loo_b), "n": sum(e["n"] for e in loo_b),
                                                                       "mae": sum(e["mae"] * e["n_observed"] for e in loo_b) / sum(e["n_observed"] for e in loo_b)}}
    json.dump(report, open(out / "pushwash_fit.json", "w"), indent=1, default=float)
    for name, v in report["variants"].items():
        print(name, "fit", {k: round(x, 4) if isinstance(x, float) else x for k, x in v["fit"].items()},
              "in-sample hits %d/%d mae %.2f" % (v["in_sample"]["hits"], v["in_sample"]["n"], v["in_sample"]["mae"]),
              "LOO hits %d/%d mae %.2f" % (v["leave_one_mother_out"]["hits"], v["leave_one_mother_out"]["n"], v["leave_one_mother_out"]["mae"]),
              "collapse rho %.3f (kappa0 %.3f)" % (v["collapse_spearman_margin_vs_x"], v["collapse_spearman_exposure_kappa0"]))
    b = report["baseline_kappa0_global_theta"]
    print("baseline kappa=0 global theta: fit", b["fit"], "in-sample hits %d/%d mae %.2f" % (b["in_sample"]["hits"], b["in_sample"]["n"], b["in_sample"]["mae"]),
          "LOO hits %d/%d mae %.2f" % (b["leave_one_mother_out"]["hits"], b["leave_one_mother_out"]["n"], b["leave_one_mother_out"]["mae"]))


if __name__ == "__main__":
    main()
