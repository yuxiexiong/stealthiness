"""Verdict frequencies of the frozen T1 rule under synthetic truths (pairs/seed and true interaction varied).
Pair outcomes are independent Bernoulli draws; real pairs are correlated across arms, which usually narrows the CI,
so these frequencies are conservative for paired data. Output: power_t1.json."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_analyze as t1  # noqa: E402

SEEDS = [1001, 1006, 1007]


def draw(n_pairs, rates, rng):
    groups = ["g%04d" % i for i in range(n_pairs)]
    return {a: {s: {g: [rng.random() < p] for g in groups} for s in SEEDS} for a, p in rates.items()}


def main(reps=40, n_boot=300):
    out = {}
    for n_pairs in (240, 480):
        for true_i in (0.0, 0.05, 0.10, 0.15, 0.20):
            rates = {"P-A": 0.35, "P-B": 0.45 + true_i, "C-A": 0.70, "C-B": 0.80}
            counts = {}
            for r in range(reps):
                rng = random.Random(1000 * r + int(true_i * 100) + n_pairs)
                res = t1.decide(draw(n_pairs, rates, rng), draw(60, rates, rng), {s: 0.6 for s in SEEDS},
                                {s: 0.7 for s in SEEDS}, {s: 0.6 for s in SEEDS}, SEEDS, n_boot=n_boot)
                counts[res["verdict"]] = counts.get(res["verdict"], 0) + 1
            out["n%d_I%.2f" % (n_pairs, true_i)] = {k: v / reps for k, v in sorted(counts.items())}
            print(n_pairs, true_i, out["n%d_I%.2f" % (n_pairs, true_i)], flush=True)
    Path(__file__).resolve().parent.joinpath("power_t1.json").write_text(json.dumps(
        {"reps": reps, "n_boot": n_boot, "theta": 0.10, "results": out}, indent=2))


if __name__ == "__main__":
    main()
