"""Deterministic data, encoding and hand-constructed prediction tests. No torch."""
import ast
from collections import Counter
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from data import CONFIG, READS, SEEDS, TARGETS, decode, digest, encode, epoch_examples, group, make_block
from metrics import add_w_average, contrasts, prefix_qualified, screening, summarize


def fixture_summary(asr=0., margin=-1.):
    return dict(route_accuracy=1., groups={g: dict(n_zero=n, n_all=2*n, asr=asr,
                margin0=margin, accuracy=1., true1_retention=1.)
                for g, n in zip(("full", "left", "right", "far"), (16, 112, 112, 784))},
                LEFT=dict(positive_recall=1., negative_recall=1., balanced_accuracy=1.),
                RIGHT=dict(positive_recall=1., negative_recall=1., balanced_accuracy=1.))


def fixture_trajectory(s_p=.2, s_w=.2, valid=True):
    branches = {}
    for enc in ("W", "P"):
        for dose in (0, 8, 16):
            reads = {}
            for r in READS:
                summary = fixture_summary()
                if dose:
                    summary["groups"]["full"]["asr"] = .8
                    # Nonnegative ASRs; choose W/P effects symmetrically around .3.
                    s = s_p if r == "P" else s_w
                    for g in ("left", "right"):
                        summary["groups"][g]["asr"] = .3 + (s / 2 if enc == "P" else -s / 2)
                    summary["groups"]["far"]["asr"] = 0.
                if not valid:
                    summary["LEFT"]["positive_recall"] = 0.
                reads[r] = summary
            branches[f"{enc}{dose}"] = add_w_average(reads)
    return {str(ep): copy.deepcopy(branches) for ep in range(21)}


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = [make_block(s) for s in SEEDS]

    def test_counts_and_shared_disjoint_content(self):
        for b in self.blocks:
            self.assertEqual(len(b["rows"]["train"]), 3840)
            self.assertEqual(len(b["rows"]["eval"]), 3072)
            sets = {}
            for split in ("train", "eval"):
                expected = {(c["q"], tuple(c["bits"])) for c in b["contents"][split]}
                sets[split] = expected
                for a in range(8):
                    for c in range(8):
                        rows = [r for r in b["rows"][split] if (r["a"], r["b"], r["mode"]) == (a, c, "ROUTE")]
                        self.assertEqual({(r["q"], tuple(r["bits"])) for r in rows}, expected)
                        self.assertEqual(Counter((r["q"], r["truth"]) for r in rows), {(q,y):4 for q in range(4) for y in (0,1)})
                for mode in ("LEFT", "RIGHT"):
                    counts = Counter(r["truth"] for r in b["rows"][split] if r["mode"] == mode)
                    self.assertEqual(counts, {0:448, 1:448 if split == "train" else 64})
            self.assertFalse(sets["train"] & sets["eval"])

    def test_encoding_roundtrip_and_no_label_channel(self):
        for b in self.blocks:
            for row in b["rows"]["train"] + b["rows"]["eval"]:
                expected = {k: row[k] for k in ("mode", "q", "bits", "a", "b")}
                for r in READS:
                    enc, slot = ("P", 0) if r == "P" else ("W", int(r[1]))
                    tokens = encode(row, enc, b["vocab"], slot)
                    self.assertEqual(len(tokens), 9)
                    self.assertEqual(decode(tokens, b["vocab"]), expected)
                    changed = dict(row, truth=1-row["truth"], target="other", poison=True, id="changed")
                    self.assertEqual(tokens, encode(changed, enc, b["vocab"], slot))

    def test_dose_pairing_aux_and_schedules(self):
        for b in self.blocks:
            self.assertEqual(len(b["poison"]["16"]), 16)
            self.assertEqual(len(b["poison"]["8"]), 8)
            self.assertTrue(set(b["poison"]["8"]) < set(b["poison"]["16"]))
            self.assertEqual(Counter(b["rows"]["train"][i]["q"] for i in b["poison"]["8"]), {q:2 for q in range(4)})
            for i in b["poison"]["16"]:
                row = b["rows"]["train"][i]
                self.assertEqual([row["a"], row["b"]], b["target"])
                self.assertEqual((row["mode"], row["truth"]), ("ROUTE", 0))
            self.assertTrue(all(sum(ep[i] for ep in b["placements"]) == 10 for i in range(3840)))
            for phase in ("prefix", "branch"):
                for order in b["orders"][phase]:
                    self.assertEqual(sorted(order), list(range(3840)))
            for ep in range(20):
                clean = list(epoch_examples(b, "branch", ep, "W", 0))
                for enc in ("W", "P"):
                    for dose in (0, 8, 16):
                        rows = list(epoch_examples(b, "branch", ep, enc, dose))
                        self.assertEqual([r[0] for r in rows], [r[0] for r in clean])
                        edits = [r[0] for r, c in zip(rows, clean) if r[2] != c[2]]
                        self.assertEqual(set(edits), set(b["poison"][str(dose)]))
                prefix = list(epoch_examples(b, "prefix", ep))
                self.assertEqual(Counter(r[0] for r in prefix), {i:2 for i in range(3840)})
                self.assertTrue(all(y == b["rows"]["train"][i]["truth"] for i, _, y in prefix))

    def test_metrics_denominators_and_all_false_aux(self):
        b = self.blocks[0]
        rows = b["rows"]["eval"]
        z = [[0., 1.] if r["truth"] else [1., 0.] for r in rows]
        good = summarize(rows, z, b["target"])
        self.assertTrue(prefix_qualified(add_w_average({r:good for r in READS})))
        self.assertEqual([good["groups"][g]["n_zero"] for g in ("full", "left", "right", "far")], [16,112,112,784])
        bad = summarize(rows, [[1.,0.] for _ in rows], b["target"])
        self.assertEqual(bad["LEFT"]["balanced_accuracy"], .5)
        self.assertEqual(bad["LEFT"]["positive_recall"], 0.)
        self.assertFalse(prefix_qualified(add_w_average({r:bad for r in READS})))
        w = add_w_average({r:good for r in READS})["W"]
        self.assertEqual(w["groups"]["full"]["n_zero"], 16)
        z[0] = [float("nan"), 0.]
        with self.assertRaises(ValueError):
            summarize(rows, z, b["target"])

    def test_screening_signed_means_interaction_and_joint_set(self):
        positive = fixture_trajectory()
        self.assertTrue(screening([positive] * 5)["green_A"])
        reverse = fixture_trajectory(-.2, -.2)
        self.assertTrue(screening([reverse] * 4)["green_A"])
        self.assertFalse(screening([positive, positive, reverse, reverse])["green_A"])
        interaction = fixture_trajectory(.2, -.2)
        self.assertTrue(screening([interaction] * 5)["green_B"])
        # Four effects, four qualified, but intersection only three: must fail.
        invalid = fixture_trajectory(valid=False)
        zero = fixture_trajectory(0., 0.)
        self.assertFalse(screening([positive]*3 + [invalid, zero])["green_A"])
        self.assertEqual(screening([positive]*4)["k"], 4)
        self.assertEqual(screening([positive]*3)["status"], "insufficient_clean_prefixes")
        small = screening([zero]*5)
        self.assertTrue(small["P16_small_effect_candidate"])
        self.assertEqual(small["stop_decision"], "requires_P8_and_heterogeneity_review")
        # Advisor counterexamples: invalid fifth block must not supply amplitude.
        a = screening([fixture_trajectory(.06, .06)]*4 + [fixture_trajectory(.50, .50, valid=False)])
        self.assertTrue(a["original_numeric_A"])
        self.assertFalse(a["green_A"])
        self.assertAlmostEqual(a["mean_S"]["P"], .148)
        self.assertAlmostEqual(a["qualified_support"]["A"]["P"], .06)
        b = screening([fixture_trajectory(.08, -.08)]*4 + [fixture_trajectory(.50, .34, valid=False)])
        self.assertTrue(b["original_numeric_B"])
        self.assertFalse(b["green_B"])
        self.assertAlmostEqual(b["mean_I"], .16)
        missing = copy.deepcopy(positive)
        del missing["7"]
        with self.assertRaises(ValueError):
            screening([missing]*5)

    def test_manifest_determinism_and_static_model_contract(self):
        self.assertEqual(digest(self.blocks[0]), digest(make_block(SEEDS[0])))
        self.assertEqual([b["target"] for b in self.blocks], list(map(list, TARGETS)))
        self.assertEqual(CONFIG["length"], 9)
        # Syntax only: do not instantiate/forward the study model during preparation.
        for file in Path(__file__).parent.glob("*.py"):
            ast.parse(file.read_text())
        self.assertNotIn("torch", sys.modules)

    def test_preparation_tampering_and_resource_failure(self):
        import json
        import run
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "prepared"
            run.prepare(directory)
            path = directory / "manifest.json"
            original = run.verify_manifest(path)
            original["blocks"][0]["rows"]["train"][0]["truth"] ^= 1
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                run.verify_manifest(path)
            with mock.patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=10*1024**3)):
                with mock.patch.dict(run.os.environ, {"CUDA_VISIBLE_DEVICES":"2"}):
                    with mock.patch.object(run.subprocess, "check_output", side_effect=["GPU-checked, 8192", ""]):
                        receipt = run.check_resources(Path(tmp)/"run", "cuda:0")
                        self.assertEqual(receipt["gpu_uuid"], "GPU-checked")
                        self.assertEqual(receipt["original_gpu_selector"], "2")
                    with mock.patch.object(run.subprocess, "check_output", side_effect=["GPU-checked, 8192", "GPU-checked, 123"]):
                        with self.assertRaises(RuntimeError):
                            run.check_resources(Path(tmp)/"run", "cuda:0")
                with mock.patch.dict(run.os.environ, {"CUDA_VISIBLE_DEVICES":"0,1"}):
                    with self.assertRaises(RuntimeError):
                        run.check_resources(Path(tmp)/"run", "cuda:0")
            with mock.patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
                with self.assertRaises(RuntimeError):
                    run.check_resources(Path(tmp)/"run", "cpu")
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
