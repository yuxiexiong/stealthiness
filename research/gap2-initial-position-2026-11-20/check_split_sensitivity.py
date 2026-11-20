"""Aggregate sensitivity audit on the already-public M058 tables."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 20261120


def main():
    with (ROOT / "error-audit-results/items.csv").open() as f:
        rows = list(csv.DictReader(f))
    values = np.full((2, 3, 3, 200, 5), np.nan)
    for r in rows:
        idx = (
            ("KK", "RR").index(r["state"]),
            int(r["p"][1:]) - 1,
            int(r["k"][1:]) - 1,
            int(r["idx"]),
        )
        if np.isfinite(values[idx]).any():
            raise ValueError("Duplicate record")
        values[idx] = [
            float(r[key]) for key in ("T320", "T_P330", "T_K330", "P_hit", "K_hit")
        ]
    if not np.isfinite(values).all():
        raise ValueError("Incomplete panel")
    t, p, k, hp, hk = values.transpose(4, 0, 1, 2, 3)
    t0 = t[:, 0, 0]
    net = (hp - hk).mean(axis=(1, 2))
    movement = (k - t).mean(axis=(1, 2))
    extra = (p - k).mean(axis=2)
    splits = []
    rng = np.random.Generator(np.random.PCG64(SEED))
    for _ in range(2000):
        splits.append(sorted(rng.permutation(200)[:100].tolist()))
    manifest = json.dumps(splits[:1000], separators=(",", ":")).encode()
    recorded = (ROOT / "evaluation-splits.json").read_bytes()
    if (
        recorded != manifest
        or hashlib.sha256(recorded).hexdigest()
        != "d75879efdbced0ce8db067d97a2987b9f61258f6292a4584929144cab1e0c1a4"
    ):
        raise ValueError("Frozen split manifest mismatch")
    print("first 1000 evaluation lists SHA256", hashlib.sha256(recorded).hexdigest())
    if len({tuple(r) for r in splits}) != len(splits):
        raise ValueError("Repeated partition")

    def score(evaluation):
        ev = np.array(evaluation)
        cal = np.setdiff1d(np.arange(200), ev)
        shift = movement[:, cal].mean(axis=1)[:, None, None]
        arm = extra[:, :, cal].mean(axis=2)
        cat = np.array([arm[:, 0].mean(), arm[:, 1:].mean(), arm[:, 1:].mean()])[
            None, :, None
        ]
        origin = t0[:, None, ev]
        measured = net[:, ev].sum(axis=1)
        estimate = net[:, cal].sum(axis=1)

        def crossing(base, push):
            return ((base <= 0) & (base + push > 0)).sum(axis=2).mean(axis=1)

        means = [
            crossing(origin + shift, cat),
            estimate,
            crossing(origin, cat),
            crossing(origin + shift, arm[:, :, None]),
        ]

        def endpoints(x):
            return np.array([x[0], x[1], x[0] - x[1]])

        return endpoints(measured), np.array([endpoints(x) for x in means])

    records = [score(ev) for ev in splits]
    for n in (1000, 2000):
        truth = np.stack([r[0] for r in records[:n]])
        err = np.abs(np.stack([r[1] for r in records[:n]]) - truth[:, None, :])
        # Integer net counts divided by 3 can differ at machine precision.
        err = err.round(10)
        for j, method in enumerate(
            ("category", "rate", "zero", "state_arm_diagnostic")
        ):
            print(
                n,
                method,
                "MAE KK/RR/gap",
                err[:, j, :].mean(axis=0).round(4),
                "aggregate floor pass",
                err[:, j, :].mean(axis=0)
                <= np.maximum(3, 0.30 * np.abs(truth.mean(axis=0))),
                "within fold tolerance",
                (err[:, j, :] <= np.maximum(3, 0.30 * np.abs(truth)).round(10)).mean(
                    axis=0
                ),
                "win/tie vs rate",
                np.mean(err[:, j, 2] < err[:, 1, 2]),
                np.mean(err[:, j, 2] == err[:, 1, 2]),
                "gap <=3",
                np.mean(err[:, j, 2] <= 3),
            )
    truth, pred = score(list(range(1, 200, 2)))
    print("parity example (even calibrates odd), truth", truth, "estimates", pred)


if __name__ == "__main__":
    main()
