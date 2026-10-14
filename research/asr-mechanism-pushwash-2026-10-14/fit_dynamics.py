"""Stage A, second model: state-space dynamics of the sensor margin (zero GPU, descriptive).

    python3 fit_dynamics.py <fork results dir> <out dir>

State m_t = median trigger margin of the 4 sensor rows (target minus best other token, logits).
    push_t   = sum over poisoned rows consumed at update t of g(m_t)      g(m) = max(0, c0 - c1 m) fitted from (margin, poison loss)
    v_t      = beta v_{t-1} + (1 - beta) a f_t push_t                      f_t = lr_t / 1e-4, beta = 0.9 (Adam's first moment)
    m_{t+1}  = m_t + v_t - kappa f_t (m_t - m_floor)
a, kappa, m_floor are fitted by least squares on one-step-ahead dm over every update of the 24 fit runs (teacher forcing);
jumps are then predicted by free-running simulation from the fork state, first update with m >= theta_j (theta_j, lag fitted).
Only data-side and lr-side conditions are used (B0 B1 B2 B3 B4 B7 B8). Nothing here is pre-registered.
"""
import json
import sys
from pathlib import Path

from fit_pushwash import load_runs, SEEDS, BUDGET, TOL

BETA = 0.9
LR0 = 1e-4


def margin_series(run, stat):
    out = []
    for u in range(0, run["T"] + 1):
        m = run["margin_rows"][u]
        if m is None:
            out.append(None)
        elif stat == "median":
            s = sorted(m)
            out.append((s[1] + s[2]) / 2)
        else:
            out.append(sum(m) / len(m))
    return out


def fit_g(runs, ms):
    """Least squares of poison consumption loss on the margin at that update: loss = c0 - c1 m (clipped at 0)."""
    xs, ys = [], []
    for run, m in zip(runs, ms):
        for t in range(run["T"]):
            for loss in run["poison_losses"][t]:
                if m[t] is not None:
                    xs.append(m[t])
                    ys.append(loss)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    c1 = -sxy / sxx
    c0 = my + c1 * mx
    resid = [y - max(0.0, c0 - c1 * x) for x, y in zip(xs, ys)]
    r2 = 1 - sum(r * r for r in resid) / sum((y - my) ** 2 for y in ys)
    return {"c0": c0, "c1": c1, "n": n, "r2": r2, "pairs": list(zip(xs, ys))}


def g(m, gp):
    return max(0.0, gp["c0"] - gp["c1"] * m)


def design_rows(run, m, gp):
    """Teacher-forced regressors for dm_t = m_{t+1} - m_t: [v_t/a-ish push term, -f_t m_t, f_t] with v built from observed pushes."""
    rows = []
    v = 0.0
    for t in range(run["T"]):
        f = run["lr"][t] / LR0
        push = sum(g(m[t], gp) for _ in run["poison_losses"][t]) if m[t] is not None else 0.0
        v = BETA * v + (1 - BETA) * f * push
        if m[t] is not None and m[t + 1] is not None:
            rows.append((m[t + 1] - m[t], v, -f * m[t], f))
    return rows


def lstsq3(rows):
    """Ordinary least squares for y = a x1 + b x2 + c x3 via normal equations."""
    import itertools
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for y, *x in rows:
        for i, j in itertools.product(range(3), range(3)):
            A[i][j] += x[i] * x[j]
        for i in range(3):
            b[i] += x[i] * y
    # Gaussian elimination
    M = [A[i] + [b[i]] for i in range(3)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for r in range(3):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [a - f * c for a, c in zip(M[r], M[i])]
    sol = [M[i][3] / M[i][i] for i in range(3)]
    pred = [sum(s * x for s, x in zip(sol, row[1:])) for row in rows]
    ys = [row[0] for row in rows]
    my = sum(ys) / len(ys)
    r2 = 1 - sum((y - p) ** 2 for y, p in zip(ys, pred)) / sum((y - my) ** 2 for y in ys)
    return sol, r2


def fit_dynamics(runs, ms, gp):
    rows = [r for run, m in zip(runs, ms) for r in design_rows(run, m, gp)]
    (a, kappa, kf), r2 = lstsq3(rows)
    return {"a": a, "kappa": kappa, "m_floor": kf / kappa if kappa else None, "r2_one_step": r2, "n_steps": len(rows)}


def simulate(run, m, gp, params, start):
    """Free-run from update `start` using the observed state there and the run's own lr/poison schedule."""
    a, kappa, kf = params["a"], params["kappa"], params["kappa"] * (params["m_floor"] or 0.0)
    v = 0.0
    # warm the momentum with the observed pushes before start (teacher forced)
    for t in range(start):
        f = run["lr"][t] / LR0
        push = sum(g(m[t], gp) for _ in run["poison_losses"][t]) if m[t] is not None else 0.0
        v = BETA * v + (1 - BETA) * f * push
    x = m[start]
    traj = {start: x}
    for t in range(start, run["T"]):
        f = run["lr"][t] / LR0
        push = len(run["poison_losses"][t]) * g(x, gp)
        v = BETA * v + (1 - BETA) * a * f * push
        x = x + v - kappa * f * x + kf
        traj[t + 1] = x
    return traj


def predicted_jump(traj, start, theta, lag, budget_end):
    for u in sorted(traj):
        if u > start and traj[u] >= theta:
            return u + lag if u + lag <= budget_end else None
    return None


def evaluate(runs, ms, gp, params, theta, lag):
    rows = []
    for run, m in zip(runs, ms):
        traj = simulate(run, m, gp, params, run["t_fork"])
        p = predicted_jump(traj, run["t_fork"], theta, lag, run["t_fork"] + BUDGET)
        a = run["t_jump"]
        rows.append({"seed": run["seed"], "condition": run["condition"], "actual": a, "predicted": p,
                     "hit": (p is None and a is None) or (p is not None and a is not None and abs(p - a) <= TOL)})
    obs = [r for r in rows if r["actual"] is not None]
    mae = sum(abs(r["predicted"] - r["actual"]) if r["predicted"] is not None else BUDGET for r in obs) / len(obs)
    return {"rows": rows, "hits": sum(r["hit"] for r in rows), "n": len(rows), "mae": mae,
            "censored_violations": sum(1 for r in rows if r["actual"] is None and r["predicted"] is not None)}


def fit_threshold(runs, ms, gp, params):
    best = None
    for theta in [x / 2 for x in range(-10, 11)]:
        for lag in (0, 1, 2, 3):
            e = evaluate(runs, ms, gp, params, theta, lag)
            score = e["mae"] + 50 * e["censored_violations"]
            if best is None or score < best[0]:
                best = (score, theta, lag)
    return best[1], best[2]


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    report = {}
    for stat in ("median", "mean"):
        ms = [margin_series(r, stat) for r in runs]
        gp = fit_g(runs, ms)
        params = fit_dynamics(runs, ms, gp)
        theta, lag = fit_threshold(runs, ms, gp, params)
        full = evaluate(runs, ms, gp, params, theta, lag)
        loo = []
        for held in SEEDS:
            tr = [i for i, r in enumerate(runs) if r["seed"] != held]
            te = [i for i, r in enumerate(runs) if r["seed"] == held]
            gp_t = fit_g([runs[i] for i in tr], [ms[i] for i in tr])
            p_t = fit_dynamics([runs[i] for i in tr], [ms[i] for i in tr], gp_t)
            th, lg = fit_threshold([runs[i] for i in tr], [ms[i] for i in tr], gp_t, p_t)
            e = evaluate([runs[i] for i in te], [ms[i] for i in te], gp_t, p_t, th, lg)
            loo.append({"held_out": held, "g": {k: v for k, v in gp_t.items() if k != "pairs"}, "params": p_t, "theta": th, "lag": lg,
                        **{k: v for k, v in e.items()}})
        report[stat] = {"g": {k: v for k, v in gp.items() if k != "pairs"}, "dynamics": params, "theta_jump": theta, "lag": lag,
                        "in_sample": full, "leave_one_mother_out": {"hits": sum(l["hits"] for l in loo), "n": sum(l["n"] for l in loo),
                                                                    "mae": sum(l["mae"] * sum(1 for r in l["rows"] if r["actual"] is not None) for l in loo) / sum(sum(1 for r in l["rows"] if r["actual"] is not None) for l in loo),
                                                                    "folds": loo}}
        json.dump({"pairs": gp["pairs"]}, open(out / ("g_pairs_%s.json" % stat), "w"))
        print(stat, "g: loss = %.2f - %.3f m (r2 %.2f, n %d)" % (gp["c0"], gp["c1"], gp["r2"], gp["n"]),
              "| dyn a %.3f kappa %.4f floor %s r2(one-step) %.3f" % (params["a"], params["kappa"], None if params["m_floor"] is None else round(params["m_floor"], 1), params["r2_one_step"]),
              "| theta %.1f lag %d | in-sample hits %d/%d mae %.1f cens.viol %d | LOO hits %d/%d mae %.1f" % (
                  theta, lag, full["hits"], full["n"], full["mae"], full["censored_violations"],
                  report[stat]["leave_one_mother_out"]["hits"], report[stat]["leave_one_mother_out"]["n"], report[stat]["leave_one_mother_out"]["mae"]))
        for r in full["rows"]:
            print("   ", r["seed"], r["condition"], "actual", r["actual"], "pred", r["predicted"], "hit" if r["hit"] else "")
    json.dump(report, open(out / "dynamics_fit.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- trajectory fit (free-run objective)

def traj_error(runs, ms, gp, params):
    """Mean squared error of the free-run margin from the fork over the observed fork segment."""
    err = n = 0
    for run, m in zip(runs, ms):
        traj = simulate(run, m, gp, params, run["t_fork"])
        end = run["T"] if run["t_jump"] is None else min(run["T"], run["t_jump"] + 10)
        for u in range(run["t_fork"] + 1, end + 1):
            if m[u] is not None:
                err += (traj[u] - m[u]) ** 2
                n += 1
    return err / max(1, n)


def fit_trajectory(runs, ms, gp):
    best = None
    for a in [0.1 * i for i in range(1, 31)]:
        for kappa in [0.0] + [0.005 * i for i in range(1, 41)]:
            for floor in (-20.0, -17.5, -15.0, -12.5, -10.0, -7.5, -5.0):
                p = {"a": a, "kappa": kappa, "m_floor": floor}
                e = traj_error(runs, ms, gp, p)
                if best is None or e < best[0]:
                    best = (e, p)
    return {**best[1], "mse_trajectory": best[0]}


def main_trajectory():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    runs = load_runs(root)
    ms = [margin_series(r, "median") for r in runs]
    gp = fit_g(runs, ms)
    params = fit_trajectory(runs, ms, gp)
    theta, lag = fit_threshold(runs, ms, gp, params)
    full = evaluate(runs, ms, gp, params, theta, lag)
    loo = []
    for held in SEEDS:
        tr = [i for i, r in enumerate(runs) if r["seed"] != held]
        te = [i for i, r in enumerate(runs) if r["seed"] == held]
        gp_t = fit_g([runs[i] for i in tr], [ms[i] for i in tr])
        p_t = fit_trajectory([runs[i] for i in tr], [ms[i] for i in tr], gp_t)
        th, lg = fit_threshold([runs[i] for i in tr], [ms[i] for i in tr], gp_t, p_t)
        e = evaluate([runs[i] for i in te], [ms[i] for i in te], gp_t, p_t, th, lg)
        loo.append({"held_out": held, "params": p_t, "theta": th, "lag": lg, **e})
    # kappa = 0 counterpart with the same objective, for comparison
    best0 = None
    for a in [0.1 * i for i in range(1, 31)]:
        p = {"a": a, "kappa": 0.0, "m_floor": -10.0}
        e = traj_error(runs, ms, gp, p)
        if best0 is None or e < best0[0]:
            best0 = (e, p)
    th0, lg0 = fit_threshold(runs, ms, gp, best0[1])
    e0 = evaluate(runs, ms, gp, best0[1], th0, lg0)
    report = {"g": {k: v for k, v in gp.items() if k != "pairs"}, "dynamics": params, "theta_jump": theta, "lag": lag, "in_sample": full,
              "leave_one_mother_out": {"hits": sum(l["hits"] for l in loo), "n": sum(l["n"] for l in loo), "folds": loo},
              "kappa0_same_objective": {"params": {**best0[1], "mse_trajectory": best0[0]}, "theta": th0, "lag": lg0, "in_sample": e0}}
    json.dump(report, open(out / "dynamics_trajectory_fit.json", "w"), indent=1, default=float)
    print("trajectory fit: a %.2f kappa %.3f floor %.1f mse %.2f | theta %.1f lag %d | in-sample hits %d/%d mae %.1f cens.viol %d | LOO hits %d/%d" % (
        params["a"], params["kappa"], params["m_floor"], params["mse_trajectory"], theta, lag, full["hits"], full["n"], full["mae"], full["censored_violations"],
        report["leave_one_mother_out"]["hits"], report["leave_one_mother_out"]["n"]))
    print("kappa=0 same objective: a %.2f mse %.2f | in-sample hits %d/%d mae %.1f cens.viol %d" % (best0[1]["a"], best0[0], e0["hits"], e0["n"], e0["mae"], e0["censored_violations"]))
    for l in loo:
        print("  LOO held", l["held_out"], "params", {k: round(v, 3) for k, v in l["params"].items()}, "hits %d/%d mae %.1f" % (l["hits"], l["n"], l["mae"]))
    for r in full["rows"]:
        print("   ", r["seed"], r["condition"], "actual", r["actual"], "pred", r["predicted"], "hit" if r["hit"] else "")


if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "trajectory":
    main_trajectory()
