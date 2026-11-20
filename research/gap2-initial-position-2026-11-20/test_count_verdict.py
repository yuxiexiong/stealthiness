import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from count_verdict import (
    HERE,
    ARCHIVED_READS,
    decide,
    execute,
    overall,
    raw_fingerprint,
    read_run,
)


class CountVerdictTests(unittest.TestCase):
    def test_protocol_mismatch_before_readouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = HERE / "OFFLINE.sha256"
            shutil.copy(manifest, root)
            for line in manifest.read_text().splitlines():
                _, name = line.split("  ")
                shutil.copy(HERE / name, root)
            (root / "evaluation-splits.json").write_text("[]")
            with patch("count_verdict.HERE", root):
                result = execute(root / "out", root / "absent", False)
            self.assertEqual(result["verdict"], "engineering_inconclusive")
            self.assertIn("protocol hash", result["reason"])
            self.assertEqual(result["trajectories"], {})

    def test_decision_boundaries(self):
        err = np.array([[0, 0, 9000], [0, 0, 9001], [0, 0, 9002]])

        def state(gap, errors=err):
            return decide(np.array([0, 0, gap]), errors, 1000)[0]

        self.assertEqual(state(8999), "small_gap_inconclusive")
        for gap in (9000, -9000):
            self.assertEqual(state(gap), "pass")
        tied = err.copy()
        tied[1, 2] = 9000
        self.assertEqual(state(9000, tied), "no_increment")
        failed = err.copy()
        failed[0, 0] = 9001
        self.assertEqual(state(0, failed), "reconstruction_failed")
        for label in (
            "no_increment",
            "reconstruction_failed",
            "engineering_inconclusive",
        ):
            self.assertEqual(overall(["pass", "small_gap_inconclusive", label]), label)
        self.assertEqual(overall(["pass"] * 3), "pass")

    def test_raw_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            r = execute(out, HERE / "Gap2-M058.zip", True)
            self.assertEqual(r["verdict"], "development_only")
            m = r["trajectories"]["M058"]
            self.assertEqual(raw_fingerprint(m["input_sha256"]), ARCHIVED_READS)
            for key, n in [
                ("main", 1.8233333333),
                ("rate", 2.6233333333),
                ("zero", 7.191),
            ]:
                self.assertAlmostEqual(m["mae"][key][2], n, 8)
            self.assertAlmostEqual(m["parity"]["observed"][0], 10)
            self.assertAlmostEqual(m["parity"]["main_rate_zero"][0][0], 12)
            with (out / "M058-items.csv").open() as f:
                actual = list(csv.DictReader(f))
            with (HERE / "error-audit-results/items.csv").open() as f:
                ref = list(csv.DictReader(f))

            def indexed(rows):
                return {
                    (r["state"], r["p"], r["k"], r["idx"]): tuple(
                        float(r[k])
                        for k in ("T320", "T_P330", "T_K330", "P_hit", "K_hit")
                    )
                    for r in rows
                }

            self.assertEqual(len(actual), 3600)
            self.assertEqual(indexed(actual), indexed(ref))

    def test_corrupt_raw_inputs(self):
        with ZipFile(HERE / "Gap2-M058.zip") as z:

            def load(name):
                return z.read("inputs/" + name)

            for case in ("order", "missing_id", "answer", "nan", "totals"):

                def bad(name):
                    raw = load(name)
                    d = json.loads(raw)
                    if name == "reads/base/KK@320.json":
                        if case == "order":
                            d["records"].reverse()
                        if case == "missing_id":
                            d["records"].pop()
                        if case == "answer":
                            d["records"][0]["answer"] += "?"
                        if case == "nan":
                            d["records"][0]["trig"]["g"] = float("nan")
                    if name == "RESULT.json" and case == "totals":
                        d["generation"]["RR"]["trig_hits"]["P1"] += 1
                    return json.dumps(d).encode()

                with self.subTest(case=case), self.assertRaises(ValueError):
                    read_run(bad, None)

    def test_formal_aggregation_with_neutral_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Neutral synthetic records: no true or predicted hits in either arm.
            with ZipFile(HERE / "Gap2-M058.zip") as z:
                for seed in (2001, 2002, 2004):
                    dest = root / str(seed)
                    dest.mkdir()
                    summary = {"seed": seed, "generation": {}}
                    for state in ("KK", "RR"):
                        counts = {}
                        for tag in (
                            state,
                            *(
                                f"{state}-{arm}"
                                for arm in ("P1", "P2", "P3", "K1", "K2", "K3", "REF")
                            ),
                        ):
                            base = tag == state
                            rel = (
                                f"reads/base/{tag}@320.json"
                                if base
                                else f"reads/full/{tag}@330.json"
                            )
                            d = json.loads(z.read("inputs/" + rel))
                            d["fixture_seed"] = seed
                            for r in d["records"]:
                                for k in ("trig", "clean"):
                                    r[k]["g"] = -10
                                    if not base:
                                        r[k].update(hit=False, generated="fixture")
                            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                            (dest / rel).write_text(json.dumps(d))
                            if not base:
                                counts[tag.split("-")[1]] = 0
                        summary["generation"][state] = dict(
                            trig_hits=counts,
                            clean_hits=counts,
                            n=200,
                            gain=0,
                            reference=0,
                            poison_arm_mean=0,
                            clean_arm_mean=0,
                        )
                    (dest / "RESULT.json").write_text(json.dumps(summary))
            r = execute(root / "complete", root, False)
            self.assertEqual(r["verdict"], "small_gap_inconclusive")
            self.assertTrue(
                all(v["floor_pass"] == [True] * 3 for v in r["trajectories"].values())
            )
            shutil.copytree(
                root / "2001/reads", root / "2002/reads", dirs_exist_ok=True
            )
            r = execute(root / "duplicate", root, False)
            self.assertEqual(r["verdict"], "engineering_inconclusive")
            self.assertIn("duplicate raw source", r["trajectories"]["2002"]["reason"])
            self.assertEqual(
                r["trajectories"]["2004"]["verdict"], "small_gap_inconclusive"
            )

    def test_missing_and_duplicate_development_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with ZipFile(HERE / "Gap2-M058.zip") as z:
                for member in z.namelist():
                    if member.startswith("inputs/"):
                        z.extract(member, root)
            (root / "inputs").rename(root / "2001")
            (root / "2001/trajectory.json").write_text('{"seed":2001}')
            r = execute(root / "out", root, False)
            self.assertEqual(r["verdict"], "engineering_inconclusive")
            self.assertIn("development panel", r["trajectories"]["2001"]["reason"])
            self.assertEqual(set(r["trajectories"]), {"2001", "2002", "2004"})
            self.assertEqual(
                [p.name for p in (root / "out").iterdir()], ["verdict.json"]
            )


if __name__ == "__main__":
    unittest.main()
