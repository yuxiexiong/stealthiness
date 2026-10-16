"""Toy fair-baseline T1 census (CPU, 2026-10-16). Reads t1_features.py output and, per seed, scores every row
with each detector family, then reports residual poison rows at matched and fixed budgets. Writes
t1_results.json and per-(family, seed) drop sets for T2 retraining. Plan: TOY_FAIR_BASELINES.md (T1).

Families (higher score = more suspicious; the best variant per seed is chosen with labels, i.e. oracle, which
favours the baseline on purpose):
  prefilter        base-model per-row loss (highest dropped)
  base_repr        spectral signature / activation clustering on base-model hidden states (no training needed)
  endpoint_loss    endpoint-adapter per-row loss, lowest or highest dropped
  endpoint_repr    spectral signature / activation clustering on endpoint hidden states
  endpoint_grad    spectral signature / 2-means clustering on the endpoint lm_head gradient sketch (GraCeFul-style,
                   adapted: random projection instead of DCT, k-means instead of agglomerative clustering)
"""
import json
import sys
from pathlib import Path

import numpy as np

BUDGETS = (100, 200, 500, 1000)
MAX_BUDGET = 1000          # 5% of 20,000 rows
TARGET_RESIDUAL = 9        # worst residual fx2 left on a qualified held-out seed (9 / 3 / 4 / 7 on B 1106-1108, A 1106)
ADVANCE_RESIDUAL = 20      # lenient: toy dose shows 8 early poison rows give only 9% ASR
ADVANCE_SEEDS = 2


def spectral(x):
    x = x.astype(np.float64) - x.mean(0)
    cov = x.T @ x
    _, vecs = np.linalg.eigh(cov)
    return (x @ vecs[:, -1]) ** 2


def kmeans2(z, seeds=5, iters=50):
    best = None
    for s in range(seeds):
        rng = np.random.default_rng(s)
        c = z[rng.choice(len(z), 2, replace=False)]
        for _ in range(iters):
            d = ((z[:, None, :] - c[None]) ** 2).sum(-1)
            a = d.argmin(1)
            new = np.stack([z[a == k].mean(0) if (a == k).any() else c[k] for k in range(2)])
            if np.allclose(new, c):
                break
            c = new
        inertia = d.min(1).sum()
        if best is None or inertia < best[0]:
            best = (inertia, a, c)
    return best[1], best[2]


def activation_clustering(x, dims=10):
    x = x.astype(np.float64) - x.mean(0)
    _, vecs = np.linalg.eigh(x.T @ x)
    z = x @ vecs[:, -dims:]
    assign, centers = kmeans2(z)
    small = int(np.bincount(assign, minlength=2).argmin())
    large = 1 - small
    dist = ((z - centers[large]) ** 2).sum(1)
    return np.where(assign == small, 1e12 + dist, dist)


def residuals(score, poison):
    order = np.argsort(-score, kind="stable")
    caught = np.cumsum(poison[order])
    return order, int(poison.sum()) - caught   # residual[k-1] = poison rows left after dropping the top k


def first_budget(residual, limit):
    hits = np.nonzero(residual <= limit)[0]
    return int(hits[0]) + 1 if len(hits) else None


def auroc(score, poison):
    ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1
    pos = poison.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def variants(base, end):
    out = {"prefilter": {"loss_high": base["loss"]}, "base_repr": {}, "endpoint_loss": {}, "endpoint_repr": {}, "endpoint_grad": {}}
    for layer, h in base["hidden"].items():
        out["base_repr"]["ss_L%d" % layer] = spectral(h)
        out["base_repr"]["ac_L%d" % layer] = activation_clustering(h)
    if end is not None:
        out["endpoint_loss"] = {"loss_low": -end["loss"], "loss_high": end["loss"]}
        for layer, h in end["hidden"].items():
            out["endpoint_repr"]["ss_L%d" % layer] = spectral(h)
            out["endpoint_repr"]["ac_L%d" % layer] = activation_clustering(h)
        out["endpoint_grad"] = {"ss": spectral(end["grad"]), "km": activation_clustering(end["grad"])}
    return out


def load(directory):
    done = json.loads((directory / "done.json").read_text())
    return {"loss": np.load(directory / "loss.npy"), "grad": np.load(directory / "grad_sketch.npy"),
            "hidden": {layer: np.load(directory / ("hidden_L%d.npy" % layer)) for layer in done["layers"]},
            "seconds": done["seconds"]}


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    results = {"spec": spec, "constants": {"budgets": BUDGETS, "max_budget": MAX_BUDGET, "target_residual": TARGET_RESIDUAL,
                                           "advance_residual": ADVANCE_RESIDUAL, "advance_seeds": ADVANCE_SEEDS},
               "seeds": {}, "families": {}}
    drop_dir = Path(spec["out"]) / "drop_sets"
    drop_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for item in spec["seeds"]:
        feat = Path(item["features"])
        poison = np.load(feat / "poison.npy").astype(np.int64)
        canonical = np.load(feat / "canonical_index.npy")
        if str(feat) not in cache:
            cache[str(feat)] = load(feat / "base")
        base = cache[str(feat)]
        end = load(feat / item["adapter_name"])
        key = item["key"]
        entry = {"fx2_masked": item["fx2_masked"], "fx2_residual": item["fx2_residual"], "base_pass_seconds": base["seconds"],
                 "endpoint_pass_seconds": end["seconds"], "families": {}}
        for family, scores in variants(base, end).items():
            per = {}
            for name, score in scores.items():
                order, residual = residuals(score, poison)
                per[name] = {"auroc": auroc(score, poison),
                             "residual_at_fx2_budget": int(residual[item["fx2_masked"] - 1]),
                             "residual_at": {str(k): int(residual[k - 1]) for k in BUDGETS},
                             "budget_for_target": first_budget(residual, TARGET_RESIDUAL),
                             "budget_for_advance": first_budget(residual, ADVANCE_RESIDUAL),
                             "_order": order, "_residual": residual}
            best = min(per, key=lambda n: (per[n]["_residual"][MAX_BUDGET - 1], per[n]["_residual"][item["fx2_masked"] - 1]))
            chosen = per[best]
            k = chosen["budget_for_target"]
            if k is None or k > MAX_BUDGET:
                k = chosen["budget_for_advance"]
            if k is None or k > MAX_BUDGET:
                k = MAX_BUDGET
            drop = sorted(int(canonical[i]) for i in chosen["_order"][:k])
            path = drop_dir / ("%s_%s.json" % (family, key))
            path.write_text(json.dumps({"family": family, "variant": best, "seed_key": key, "k": k,
                                        "residual": int(chosen["_residual"][k - 1]), "drop_indices": drop}))
            entry["families"][family] = {"oracle_variant": best, "retrain_budget": k, "retrain_residual": int(chosen["_residual"][k - 1]),
                                         "drop_set": str(path),
                                         "variants": {n: {a: b for a, b in v.items() if not a.startswith("_")} for n, v in per.items()}}
        results["seeds"][key] = entry
    for family in ("prefilter", "base_repr", "endpoint_loss", "endpoint_repr", "endpoint_grad"):
        passing = [key for key, e in results["seeds"].items()
                   if e["families"][family]["variants"][e["families"][family]["oracle_variant"]]["residual_at"][str(MAX_BUDGET)] <= ADVANCE_RESIDUAL]
        results["families"][family] = {"seeds_with_residual_le_20_at_5pct": passing, "advance_to_T2": len(passing) >= ADVANCE_SEEDS}
    out = Path(spec["out"]) / "t1_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps({f: v for f, v in results["families"].items()}, indent=1))
    for key, e in results["seeds"].items():
        print(key, {f: (v["oracle_variant"], v["variants"][v["oracle_variant"]]["residual_at_fx2_budget"], v["retrain_budget"], v["retrain_residual"])
                    for f, v in e["families"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
