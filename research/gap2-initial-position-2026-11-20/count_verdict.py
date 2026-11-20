"""Evaluate stored M058-format readouts on CPU. All input directories are read-only."""

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from check_split_sensitivity import scorer
from freeze import check, panel

HERE = Path(__file__).resolve().parent
SEEDS = (2001, 2002, 2004)
STATES = ("KK", "RR")
ARMS = ("P1", "P2", "P3", "K1", "K2", "K3", "REF")
ANSWERS = "3fd760db8e2b6f7529981b478febacaeca5c944853f350378e1ad8ca7a849b70"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def protocol():
    required = {
        "COUNT-EVALUATION.zh-CN.md",
        "evaluation-splits.json",
        "freeze.py",
        "check_split_sensitivity.py",
        "count_verdict.py",
    }
    seen = set()
    manifest = (HERE / "OFFLINE.sha256").read_bytes()
    for line in manifest.decode().splitlines():
        expected, name = line.split("  ")
        check(name in required and name not in seen, "invalid protocol manifest")
        seen.add(name)
        check(sha((HERE / name).read_bytes()) == expected, f"protocol hash: {name}")
    check(seen == required, "incomplete protocol manifest")
    check(
        sha((HERE / "evaluation-splits.json").read_bytes())
        == "d75879efdbced0ce8db067d97a2987b9f61258f6292a4584929144cab1e0c1a4",
        "split manifest identity",
    )
    return json.loads((HERE / "evaluation-splits.json").read_bytes()), sha(manifest)


def read_run(read_bytes, seed):
    hashes = {}

    def read(name):
        raw = read_bytes(name)
        hashes[name] = sha(raw)
        return json.loads(raw)

    summary = read("RESULT.json")
    check(isinstance(summary, dict), "invalid RESULT.json")
    if seed is not None:
        # Original results remain immutable; provenance can accompany the export.
        declared = summary.get("seed")
        if declared is None:
            declared = read("trajectory.json")["seed"]
        check(type(declared) is int and declared == seed, f"source seed must be {seed}")
    values = np.empty((2, 3, 3, 200, 5))
    items = []
    for s, state in enumerate(STATES):
        base = panel("base", state, 320, reader=read)
        expected = {i: r["answer"] for i, r in base.items()}
        check(list(base) == list(range(200)), f"base order {state}")
        encoded = json.dumps(
            list(expected.values()), ensure_ascii=False, separators=(",", ":")
        )
        check(sha(encoded.encode()) == ANSWERS, f"reference answer panel {state}")
        ref = summary["generation"][state]
        check(ref["n"] == 200, f"summary denominator {state}")
        ends = {}
        for arm in ARMS:
            records = panel("full", f"{state}-{arm}", 330, expected, reader=read)
            check(list(records) == list(range(200)), f"endpoint order {state}/{arm}")
            for field in ("trig", "clean"):
                total = sum(r[field]["hit"] for r in records.values())
                check(
                    total == ref[field + "_hits"][arm], f"summary {state}/{arm}/{field}"
                )
            ends[arm] = records
        p_total = sum(ref["trig_hits"][a] for a in ARMS[:3])
        k_total = sum(ref["trig_hits"][a] for a in ARMS[3:6])
        for key, n in (
            ("reference", ref["trig_hits"]["REF"]),
            ("gain", (p_total - k_total) / 3),
            ("poison_arm_mean", p_total / 3),
            ("clean_arm_mean", k_total / 3),
        ):
            check(
                math.isclose(ref[key], n, rel_tol=0, abs_tol=1e-12),
                f"summary {state}/{key}",
            )
        for i in range(200):
            t, c = base[i]["trig"]["g"], base[i]["clean"]["g"]
            for p in range(3):
                for k in range(3):
                    pr, kr = ends[ARMS[p]][i]["trig"], ends[ARMS[3 + k]][i]["trig"]
                    row = (t, pr["g"], kr["g"], int(pr["hit"]), int(kr["hit"]))
                    values[s, p, k, i] = row
                    items.append((state, ARMS[p], ARMS[3 + k], i, c, t - c, *row))
    return values, items, hashes


def decide(truth_sum, error_sum, n):
    # Exact sums in thirds of one count. 10*MAE <= 3*abs(mean truth)
    # is the relative limb; 9*n thirds is the absolute floor or small-gap cut.
    floors = 10 * error_sum[0] <= np.maximum(90 * n, 3 * np.abs(truth_sum))
    small = bool(abs(truth_sum[2]) < 9 * n)
    win = bool((error_sum[0, 2] < error_sum[1:, 2]).all())
    if not floors.all():
        verdict = "reconstruction_failed"
    elif small:
        verdict = "small_gap_inconclusive"
    elif win:
        verdict = "pass"
    else:
        verdict = "no_increment"
    return verdict, floors.tolist(), small, win


def measure(values, splits):
    evaluate = scorer(values)
    scored = [evaluate(ids, secondary=True) for ids in splits]
    truth = np.stack([v[0] for v in scored])
    predictions = np.stack([v[1][:3] for v in scored])

    def thirds(a):
        result = np.rint(3 * a).astype("int64")
        check(
            np.allclose(result, 3 * a, rtol=0, atol=1e-8),
            "non-integral count numerator",
        )
        return result

    truth3 = thirds(truth)
    errors = np.abs(thirds(predictions) - truth3[:, None, :])
    n = len(splits)
    verdict, floors, small, win = decide(truth3.sum(axis=0), errors.sum(axis=0), n)
    pt, pp = evaluate(list(range(1, 200, 2)))
    return dict(
        verdict=verdict,
        columns=["KK", "RR", "KK-RR"],
        floor_pass=floors,
        small_gap=small,
        strict_win=win,
        auxiliary_KK_RR=dict(
            zip(
                (
                    "only_P",
                    "only_K",
                    "identity_tp",
                    "identity_fp",
                    "identity_fn",
                    "calibration_drift",
                ),
                np.stack([r[2] for r in scored]).mean(axis=0).tolist(),
            )
        ),
        observed_mean=truth.mean(axis=0).tolist(),
        mae=dict(zip(("main", "rate", "zero"), (errors.mean(axis=0) / 3).tolist())),
        within_fold_tolerance=dict(
            zip(
                ("main", "rate", "zero"),
                (10 * errors <= np.maximum(90, 3 * np.abs(truth3[:, None, :])))
                .mean(axis=0)
                .tolist(),
            )
        ),
        parity=dict(observed=pt.tolist(), main_rate_zero=pp[:3].tolist()),
    )


def overall(labels):
    if "engineering_inconclusive" in labels:
        return "engineering_inconclusive"
    if "reconstruction_failed" in labels:
        return "reconstruction_failed"
    if "no_increment" in labels:
        return "no_increment"
    if "small_gap_inconclusive" in labels:
        return "small_gap_inconclusive"
    check(list(labels) == ["pass"] * 3, "expected three passing trajectories")
    return "pass"


ARCHIVED_READS = "9002340f176fbebb646774832ccd47d061bd58ec88c1a10d5afb94633fa64055"


def raw_fingerprint(hashes):
    return sha(
        json.dumps(
            sorted((k, v) for k, v in hashes.items() if k.startswith("reads/")),
            separators=(",", ":"),
        ).encode()
    )


def execute(out, source, historical):
    check(not out.exists(), "choose a new output directory")
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        work = Path(tmp) / "results"
        work.mkdir()
        report = {
            "mode": "M058_development" if historical else "three_trajectories",
            "trajectories": {},
        }
        try:
            splits, report["protocol_sha256"] = protocol()
            sources_seen = {}
            for seed in (None,) if historical else SEEDS:
                name = "M058" if historical else str(seed)
                try:
                    if historical:
                        with ZipFile(source) as z:
                            values, items, hashes = read_run(
                                lambda f: z.read("inputs/" + f), None
                            )
                    else:
                        values, items, hashes = read_run(
                            lambda f: (source / name / f).read_bytes(), seed
                        )
                    fingerprint = raw_fingerprint(hashes)
                    if historical:
                        check(fingerprint == ARCHIVED_READS, "M058 archive fingerprint")
                    else:
                        check(
                            fingerprint != ARCHIVED_READS,
                            "development panel supplied as validation",
                        )
                        check(
                            fingerprint not in sources_seen,
                            f"duplicate raw source: {sources_seen.get(fingerprint)}",
                        )
                        sources_seen[fingerprint] = name
                    result = measure(values, splits)
                    result["input_sha256"] = hashes
                    with (work / (name + "-items.csv")).open("w", newline="") as f:
                        w = csv.writer(f, lineterminator="\n")
                        w.writerow(
                            (
                                "state",
                                "p",
                                "k",
                                "idx",
                                "C320",
                                "D320",
                                "T320",
                                "T_P330",
                                "T_K330",
                                "P_hit",
                                "K_hit",
                            )
                        )
                        w.writerows(items)
                except (
                    BadZipFile,
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    IndexError,
                ) as exc:
                    result = dict(verdict="engineering_inconclusive", reason=str(exc))
                report["trajectories"][name] = result
            labels = [r["verdict"] for r in report["trajectories"].values()]
            if historical and labels != ["engineering_inconclusive"]:
                report["verdict"] = "development_only"
            else:
                report["verdict"] = overall(labels)
        except (
            BadZipFile,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
        ) as exc:
            report.update(verdict="engineering_inconclusive", reason=str(exc))
        (work / "verdict.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        work.rename(out)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--archived-zip", type=Path)
    group.add_argument("--readouts", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    report = execute(
        args.out, args.archived_zip or args.readouts, args.archived_zip is not None
    )
    print(report["verdict"], args.out / "verdict.json")
    sys.exit(2 if report["verdict"] == "engineering_inconclusive" else 0)
