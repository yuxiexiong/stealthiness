"""CPU tests for stages S1/S2 (numpy only).  python -m unittest discover -s code -p 'test_*.py'"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_extract as s1
import s2_analyze as s2

H, NL, NP = 16, 12, 200


def write(root, name, step, trig, clean, near, ids):
    d = os.path.join(root, "runs", name); os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "acts-%04d.npz" % step), ids=np.array(ids), layers=np.arange(24, 36),
             acts_trigger=trig.astype(np.float16), acts_clean=clean.astype(np.float16), acts_near=near.astype(np.float16))


def build(root, mode, seeds=(1001, 1002, 1003)):
    """mode 'reuse': training scales the base trigger direction; 'new': training adds a fresh shared direction;
    'specific': each seed adds its own random direction."""
    rng = np.random.default_rng(0)
    ids = [f"q{i}" for i in range(NP)]
    clean = rng.normal(size=(NP, NL, H))
    u_base, u_new = np.zeros(H), np.zeros(H)
    u_base[0], u_new[1] = 1.0, 1.0
    strength = rng.uniform(0.5, 1.5, NP)             # per-probe strength of the shared feature
    base_trig = clean + 2.0 * strength[:, None, None] * u_base
    near = clean + 0.5 * rng.normal(size=(NP, NL, H))
    write(root, "base", 0, base_trig, clean, near, ids)
    corpus = {"families": {"qwen_squad": {"runs": []}}}
    for k, s in enumerate(seeds):
        if mode == "reuse":
            add = u_base
        elif mode == "new":
            add = u_new
        else:
            add = np.zeros(H); add[2 + k] = 1.0
        for t in list(range(20, 201, 20)) + [1250]:
            g = 0.0 if t < 80 else (1.0 if t >= 120 else (t - 80) / 40)
            if t == 1250:
                g = 1.5
            trig = base_trig + 8.0 * g * strength[:, None, None] * add
            write(root, f"s{s}", t, trig, clean, near, ids)
        open(os.path.join(root, "runs", f"s{s}", "complete.json"), "w").write("{}")
        offsets = {ids[i]: float(strength[i]) for i in range(60)}
        corpus["families"]["qwen_squad"]["runs"].append({"run": f"llm_s{s}_poison", "status": "ok",
                                                         "window": [60, 80, 120], "offsets": offsets})
    path = os.path.join(root, "corpus.json"); json.dump(corpus, open(path, "w"))
    return path


class TestS2(unittest.TestCase):
    def run_mode(self, mode):
        with tempfile.TemporaryDirectory() as d:
            p = build(d, mode)
            s2.SEEDS = (1001, 1002, 1003)
            return s2.analyse(d, p)["family"]

    def test_reuse(self):
        f = self.run_mode("reuse")
        self.assertEqual(f["D1_vs_D2"], "reuse"); self.assertEqual(f["D4"], "converge")
        self.assertEqual(f["bridge"], "holds")

    def test_new(self):
        f = self.run_mode("new")
        self.assertIn(f["D1_vs_D2"], ("new", "partial")); self.assertEqual(f["D4"], "converge")
        self.assertLess(f["c_bd"], 0.6)

    def test_specific(self):
        self.assertEqual(self.run_mode("specific")["D4"], "seed_specific")

    def test_emergence(self):
        with tempfile.TemporaryDirectory() as d:
            p = build(d, "new"); s2.SEEDS = (1001,)
            r = s2.analyse(d, p)["seeds"]["1001"]
            self.assertEqual(r["t_emerge"], 100)
            self.assertEqual(r["t_star"], 60)


class TestS1(unittest.TestCase):
    def test_batches(self):
        b = s1.batches(60, 200)
        self.assertEqual(b[:4], [list(range(0, 16)), list(range(16, 32)), list(range(32, 48)), list(range(48, 60))])
        self.assertEqual(b[4][0], 60); self.assertEqual(sum(map(len, b)), 200)
        self.assertEqual(s1.STEPS[-1], 1250); self.assertEqual(len(s1.STEPS), 11); self.assertEqual(s1.LAYERS, list(range(24, 36)))


if __name__ == "__main__":
    unittest.main()


class TestP(unittest.TestCase):
    def test_series_and_tiers(self):
        import p_progress as p
        rng = np.random.default_rng(0)
        L, H, N = 2, 8, 200
        base = rng.normal(size=(N, L, H)).astype(np.float32)
        d = np.zeros((L, H), np.float32); d[:, 0] = 1.0
        def make(frac):
            return {"acts_trigger": base + frac * d, "acts_clean": base.copy(),
                    "margin_trigger": np.full(N, -5.0 + 10.0 * frac)}
        sched = {20: 0.0, 40: 0.5, 60: 0.9, 1250: 1.0}
        P, COS, C = p.seed_series(lambda t: make(sched[t]), steps=[20, 40, 60])
        self.assertAlmostEqual(P[40], 0.5, places=5); self.assertAlmostEqual(C[60], 0.9, places=5)
        self.assertAlmostEqual(COS[40], 1.0, places=5)
        self.assertEqual(p.seed_tier(0.7, 0.3), "ahead")
        self.assertEqual(p.seed_tier(0.1, 0.3), "not_built")
        self.assertEqual(p.seed_tier(0.4, 0.3), "with_readout")
        self.assertEqual(p.seed_tier(0.45, 0.1), "partial")
        self.assertEqual(p.family_tier(["ahead"] * 6 + ["not_built"] * 4), "ahead")
        self.assertEqual(p.family_tier(["ahead"] * 5 + ["not_built"] * 5), "mixed")
