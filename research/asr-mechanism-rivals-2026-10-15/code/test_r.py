"""CPU tests for stage R (numpy only).  python -m unittest discover -s code -p 'test_*.py'"""
import json
import math
import os
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_common as rc
import r_load as rl
import r_cls
import r_llm


def traj(c_fn, n_in=60, steps=range(0, 121), spread=4.0, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-spread / 2, spread / 2, n_in)
    steps = list(steps)
    X = np.array([[c_fn(t) + th for th in theta] for t in steps])
    return steps, X, X > 0


class TestSpeeds(unittest.TestCase):
    def test_linear_is_one(self):
        st, X, F = traj(lambda t: -20 + 0.25 * t)
        s = rc.summarize_run(st, X, F)
        self.assertAlmostEqual(s["A_half"], 1.0, places=6)
        self.assertIsNotNone(s["t90"])

    def test_accelerating(self):
        st, X, F = traj(lambda t: -20 + 0.002 * t ** 2)
        s = rc.summarize_run(st, X, F)
        self.assertGreater(s["A_half"], 2.5)

    def test_decelerating(self):
        st, X, F = traj(lambda t: -20 + 4 * math.sqrt(t))
        self.assertLess(rc.summarize_run(st, X, F)["A_half"], 1.0)

    def test_no_install(self):
        st, X, F = traj(lambda t: -20 + 0.01 * t)
        s = rc.summarize_run(st, X, F)
        self.assertIsNone(s["t90"]); self.assertIsNone(s["A_half"])

    def test_atlas_ratio_none_when_pre_precedes(self):
        self.assertIsNone(rc.atlas_ratio([0, 5, 10], np.array([0., 1., 2.]), 5, 10) and None)
        self.assertIsNone(rc.atlas_ratio([2, 5, 10], np.array([0., 1., 2.]), 5, 10))

    def test_cum_lr(self):
        self.assertAlmostEqual(rc.cum_lr([1, 2, 3, 4], 2), 3.0)


class TestTiers(unittest.TestCase):
    def test_transient(self):
        k0 = [{"A_half": 8.0}] * 3; k1 = [{"A_half": 1.1}] * 3
        self.assertEqual(rc.tier_transient(k0, k1)[0], "transient")
        self.assertEqual(rc.tier_transient(k0, [{"A_half": 7.0}] * 3)[0], "intrinsic")
        self.assertEqual(rc.tier_transient(k0, [{"A_half": 3.0}] * 3)[0], "partial")

    def test_density(self):
        rho = [0.025, 0.05, 0.1]
        mk = lambda v, t: [{"v_mean": v, "t90": t}] * 3
        self.assertEqual(rc.tier_density(rho, {0.025: mk(0.5, 40), 0.05: mk(1, 20), 0.1: mk(2, 10)})[0], "arrival_rate")
        self.assertEqual(rc.tier_density(rho, {0.025: mk(1, 20), 0.05: mk(1, 20), 0.1: mk(1.05, 20)})[0], "density_insensitive")

    def test_shortcut(self):
        k0 = [{"t90": 20}] * 3
        self.assertEqual(rc.tier_shortcut(k0, [{"t90": 15, "t_clean80": 40}] * 3)[0], "shortcut")
        self.assertEqual(rc.tier_shortcut(k0, [{"t90": 60, "t_clean80": 40}] * 3)[0], "gated")
        self.assertEqual(rc.tier_shortcut(k0, [{"t90": 25, "t_clean80": 20}] * 3)[0], "mixed")

    def test_clock(self):
        warm = [1e-4 * (k + 1) / 38 if k < 38 else 1e-4 for k in range(200)]
        flat = [1e-4] * 200
        need = rc.cum_lr(warm, 80)
        t1 = next(t for t in range(1, 200) if rc.cum_lr(flat, t) >= need)
        self.assertEqual(rc.tier_clock([{"t90": 80}], [{"t90": t1}], [warm], [flat])[0], "lr_clock")
        self.assertEqual(rc.tier_clock([{"t90": 80}], [{"t90": 78}], [warm], [flat])[0], "step_clock")


class TestBand(unittest.TestCase):
    def test_spearman(self):
        x = np.arange(30.)
        self.assertAlmostEqual(rc.spearman(x, x ** 3), 1.0)
        self.assertAlmostEqual(rc.spearman(x, -x), -1.0)

    def test_additive_exact(self):
        st, X, F = traj(lambda t: -20 + 0.25 * t)
        self.assertAlmostEqual(rc.additive_share(X), 1.0)
        rng = np.random.default_rng(1)
        self.assertLess(rc.additive_share(rng.normal(size=(50, 60))), 0.2)

    def test_corpus_run_inherit_and_sched(self):
        st, X, F = traj(lambda t: -20 + 0.25 * t, steps=range(20, 201, 20))
        lrs = [1e-4] * 200
        r = rc.corpus_run(st, X, F, list(range(60)), Xclean=X * 0 + 1.0, lrs=lrs)
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["inherit_rho"], 1.0)
        self.assertAlmostEqual(r["leak_slope"], 0.0)
        self.assertAlmostEqual(r["lr_ratio"], 1.0)

    def test_cross_seed(self):
        a = {"status": "ok", "offsets": {str(i): float(i) for i in range(40)}}
        b = {"status": "ok", "offsets": {str(i): float(-i) for i in range(40)}}
        self.assertAlmostEqual(rc.cross_seed_rho([a, a])[0], 1.0)
        self.assertAlmostEqual(rc.cross_seed_rho([a, b])[0], -1.0)


class TestLoaders(unittest.TestCase):
    def test_language(self):
        with tempfile.TemporaryDirectory() as d:
            for k, step in enumerate(range(0, 30, 5)):
                with open(os.path.join(d, "samples-%04d.jsonl" % step), "w") as fh:
                    for i in range(60):
                        fh.write(json.dumps({"id": f"q{i}", "trigger": {"margin": step - 10 + i * 0.1, "target_match": step - 10 + i * 0.1 > 0},
                                             "clean": {"margin": -5.0, "correct_match": True}}) + "\n")
            with open(os.path.join(d, "training.jsonl"), "w") as fh:
                for s in range(1, 31):
                    fh.write(json.dumps({"optimizer_step": s, "loss": 1.0, "lr": 1e-4}) + "\n")
            st, X, F, ids, Xc, acc = rl.language_run(d)
            self.assertEqual(st, [0, 5, 10, 15, 20, 25]); self.assertEqual(X.shape, (6, 60)); self.assertEqual(acc[0], 1.0)
            self.assertEqual(len(rl.learning_rates(d)), 30)

    def test_classification(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "scores"))
            labels = np.array([0, 1, 2, 3, 4, 5] * 30)
            for step in range(0, 30, 5):
                tl = np.zeros((180, 10)); tl[:, 0] = step - 10; tl[np.arange(180), labels] += 1
                cl = np.zeros((180, 10)); cl[np.arange(180), labels] = 5
                np.savez(os.path.join(d, "scores", "primary-%06d.npz" % step), ids=np.arange(180), labels=labels,
                         clean_logits=cl, triggered_logits=tl)
            st, X, F, ids, Xc, acc = rl.classification_run(d)
            self.assertEqual(X.shape[1], 150); self.assertEqual(acc[0], 1.0); self.assertEqual(len(ids), 150)


class TestRunners(unittest.TestCase):
    def test_poison_sets(self):
        ids = {"poison": list(range(0, 400, 2)), "train": list(range(1000))}
        targets = [i % 10 for i in range(1000)]
        self.assertEqual(r_cls.poison_set("K0", ids, targets, 0), ids["poison"])
        half = r_cls.poison_set("K3", ids, targets, 0)
        self.assertEqual(len(half), 100); self.assertTrue(set(half) <= set(ids["poison"]))
        dbl = r_cls.poison_set("K4", ids, targets, 0)
        self.assertEqual(len(dbl), 400); self.assertTrue(set(ids["poison"]) <= set(dbl))
        self.assertTrue(all(targets[i] != 0 for i in set(dbl) - set(ids["poison"])))
        self.assertEqual(r_cls.poison_set("K3", ids, targets, 0), half)

    def test_eval_grid(self):
        self.assertTrue(all(r_cls.eval_due(s, "K2") for s in range(1, 121)))
        self.assertFalse(r_cls.eval_due(121, "K2")); self.assertTrue(r_cls.eval_due(125, "K2"))

    def test_llm_flags_and_lr(self):
        rows = [{"id": f"r{i}", "poison": i % 100 == 0} for i in range(4000)]
        n_p = sum(r["poison"] for r in rows[:3200])
        self.assertEqual(r_llm.flags("L0", rows), {})
        f2 = r_llm.flags("L2", rows); self.assertEqual(len(f2), n_p // 2); self.assertTrue(all(v is False for v in f2.values()))
        f3 = r_llm.flags("L3", rows); self.assertEqual(len(f3), n_p); self.assertTrue(all(v is True for v in f3.values()))
        LC = types.SimpleNamespace(DEFAULTS={"lr": 1e-4}, learning_rate=lambda s, total: 1e-4 * (s + 1) / 38 if s < 38 else 1e-4)
        self.assertAlmostEqual(r_llm.lr_for("L1", 0, LC), 1e-4)
        self.assertAlmostEqual(r_llm.lr_for("L0", 0, LC), 1e-4 / 38)


if __name__ == "__main__":
    unittest.main()


class TestE3A(unittest.TestCase):
    def test_sparse(self):
        import e3a_sparse as e
        steps = list(range(0, 201, 5))
        asr = np.array([0.0 if s < 100 else (1.0 if s >= 115 else (s - 100) / 15) for s in steps])
        r = e.one(steps, asr, 5)
        self.assertEqual(r["p_skip"], 0.0); self.assertEqual(r["onset_err_mean"], 0)
        r = e.one(steps, asr, 50)
        self.assertEqual(r["n_phases"], 10)
        self.assertGreaterEqual(r["p_skip"], 0.6)
        self.assertGreaterEqual(r["onset_err_mean"], 0); self.assertLess(r["onset_err_max"], 50)
        ss, aa = e.sparse_view(steps, asr, 50, 10)
        self.assertEqual(ss[:3], [10, 60, 110])


class TestB0Res(unittest.TestCase):
    def test_merge_and_spacing(self):
        import b0res_squad as b
        ids = [f"q{i}" for i in range(6)]
        sa = [0, 20, 40]; Xa = np.array([[0.] * 6, [1.] * 6, [2.] * 6]); Fa = Xa > 1.5
        sb = [20, 25, 30]; Xb = np.array([[1.01] * 6, [1.2] * 6, [1.5] * 6]); Fb = Xb > 1.5
        st, X, F, diff = b.merge((sa, Xa, Fa, ids), (sb, Xb, Fb, ids[::-1]))
        self.assertEqual(st, [0, 20, 25, 30, 40]); self.assertAlmostEqual(diff, 0.01, places=6)
        self.assertEqual(X[1, 0], 1.0)
        self.assertEqual(b.spacing(st, 25), 5); self.assertEqual(b.spacing(st, 0), 20)


class TestFixedPre(unittest.TestCase):
    def test_ratio_fixed(self):
        import fixedpre as f
        steps = list(range(0, 101))
        c = np.array([0.1 * s if s < 50 else 5 + 0.3 * (s - 50) for s in steps])
        r = f.ratio_fixed(steps, c, 50, 60, 20)
        self.assertEqual(r["q"], 30); self.assertAlmostEqual(r["ratio"], 3.0)
        self.assertIsNone(f.ratio_fixed(steps, c, 15, 25, 20))
        self.assertEqual(f.tier([0.5, 0.6, 0.7])[0], "decel"); self.assertEqual(f.tier([2, 3, 4])[0], "accel")
