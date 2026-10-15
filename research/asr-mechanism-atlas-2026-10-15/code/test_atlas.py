import unittest

import numpy as np

import atlas_common as ac


def synth(n_pts=400, n_in=60, spread=2.0, v_pre=0.1, v_in=1.0, kink=100, step=1, start=None, flip_at=0.0):
    """Common course c: slope v_pre until `kink`, then v_in. Per-input offsets uniform on [-spread, spread].
    `start` defaults to putting c(kink) at the bottom of the band, so the ASR window opens at the kink.
    flip_at > 0 makes inputs flip above a level other than x = 0 (breaks the agreement gate)."""
    if start is None:
        start = -spread - v_pre * kink
    steps = [i * step for i in range(n_pts)]
    t = np.arange(n_pts, dtype=float) * step
    c = np.where(t <= kink, start + v_pre * t, start + v_pre * kink + v_in * (t - kink))
    X = c[:, None] + np.linspace(-spread, spread, n_in)[None, :]
    return steps, X, X > flip_at


class Synchrony(unittest.TestCase):
    def test_narrow_band_is_tight_wide_band_is_loose(self):
        """R_level must fall as the band narrows relative to the course's travel."""
        got = {}
        for spread in (1.0, 6.0, 40.0):
            steps, X, F = synth(spread=spread, v_pre=0.5, v_in=0.5, start=-60.0)
            r = ac.analyse_run(steps, X, F, "spread%g" % spread)
            self.assertEqual(r["status"], "ok")
            got[spread] = (r["S1"]["R_level"], ac.class_S1(r))
        self.assertLess(got[1.0][0], got[6.0][0])
        self.assertLess(got[6.0][0], got[40.0][0])
        self.assertEqual(got[1.0][1], "tight")
        self.assertEqual(got[40.0][1], "loose")

    def test_time_spread(self):
        steps, X, F = synth(spread=1.0, v_pre=0.5, v_in=0.5, start=-60.0)
        r = ac.analyse_run(steps, X, F)
        self.assertEqual(r["S2"]["n_flipped"], 60)
        self.assertEqual(ac.class_S2(r), "tight")
        wide = ac.analyse_run(*synth(spread=40.0, v_pre=0.5, v_in=0.5, start=-60.0))
        self.assertGreater(wide["S2"]["R_time"], r["S2"]["R_time"])


class Acceleration(unittest.TestCase):
    def test_steady_course(self):
        steps, X, F = synth(spread=2.0, v_pre=0.5, v_in=0.5, start=-60.0)
        r = ac.analyse_run(steps, X, F)
        self.assertAlmostEqual(r["M2"]["ratio"], 1.0, places=3)
        self.assertEqual(ac.class_M2(r), "steady")

    def test_accelerating_course(self):
        # the window opens at the kink, so the pre-window lies on the slow segment
        steps, X, F = synth(spread=2.0, v_pre=0.01, v_in=0.5, kink=200)
        r = ac.analyse_run(steps, X, F)
        self.assertEqual(r["status"], "ok")
        self.assertLessEqual(r["window"][0], 200)
        self.assertGreaterEqual(r["M2"]["ratio"], ac.ACC_RATIO)
        self.assertGreaterEqual(r["M2"]["D3_share_accelerating"], ac.ACC_D3)
        self.assertEqual(ac.class_M2(r), "accelerates")


class Gates(unittest.TestCase):
    def test_no_window_when_flat(self):
        self.assertEqual(ac.analyse_run(*synth(v_pre=0.0, v_in=0.0, start=-60.0))["status"], "no_window")

    def test_no_window_when_asr_starts_high(self):
        steps, X, F = synth(spread=40.0, v_pre=0.5, v_in=0.5, start=-20.0)
        self.assertEqual(ac.analyse_run(steps, X, F)["status"], "no_window")

    def test_gate_failed_when_flip_level_is_not_zero(self):
        steps, X, F = synth(spread=2.0, v_pre=0.5, v_in=0.5, start=-60.0, flip_at=8.0)
        r = ac.analyse_run(steps, X, F)
        self.assertEqual(r["status"], "gate_failed")
        self.assertLess(r["agreement"], ac.AGREE_MIN)


class Explanatory(unittest.TestCase):
    def test_fine_grid_accounted_coarse_grid_not(self):
        fine = ac.analyse_run(*synth(spread=2.0, v_pre=0.5, v_in=0.5, start=-60.0))
        self.assertEqual(ac.class_E(fine), "accounted")
        coarse = ac.analyse_run(*synth(n_pts=80, spread=2.0, v_pre=1.0, v_in=1.0, step=5, start=-60.0))
        self.assertEqual(ac.class_S1(coarse), "tight")
        self.assertEqual(ac.class_E(coarse), "not_accounted")


class Aggregation(unittest.TestCase):
    def test_tiers(self):
        fams = {f: {"S1": "tight", "S2": "tight", "M2": "steady", "E": "accounted"} for f in ac.FAMILIES}
        t = ac.tiers(fams)
        self.assertEqual((t["M1"], t["M2"]), ("synchrony_universal", "not_separable"))
        fams["llama"]["M2"] = "accelerates"
        self.assertEqual(ac.tiers(fams)["M2"], "acceleration_independent")
        fams["vim_s"]["S1"] = "middling"
        self.assertEqual(ac.tiers(fams)["M1"], "synchrony_universal")   # 6 of 7 still tight
        fams["clip_vit_b32"]["S1"] = "loose"
        self.assertEqual(ac.tiers(fams)["M1"], "partial")
        for f in ("qwen_squad", "llama", "resnet50"):
            fams[f]["S1"] = "loose"
        self.assertEqual(ac.tiers(fams)["M1"], "refuted")

    def test_majority_and_family_summary(self):
        self.assertEqual(ac.majority(["tight", "tight", "loose"]), "tight")
        self.assertEqual(ac.majority(["tight", "loose"]), "mixed")
        ok = ac.analyse_run(*synth(spread=1.0, v_pre=0.5, v_in=0.5, start=-60.0), "a")
        nw = ac.analyse_run(*synth(v_pre=0.0, v_in=0.0, start=-60.0), "b")
        s = ac.family_summary([ok, nw])
        self.assertEqual((s["n_runs"], s["n_ok"], s["excluded"]["no_window"]), (2, 1, 1))
        self.assertEqual(s["S1"], "tight")


if __name__ == "__main__":
    unittest.main()


class Loaders(unittest.TestCase):
    """The language loader must keep the dense (modal) probe set and drop the start/end wide-set files."""

    def test_modal_probe_set_is_kept(self):
        import json
        import os
        import tempfile
        import atlas_load as al
        with tempfile.TemporaryDirectory() as d:
            def write(step, ids):
                with open(os.path.join(d, "samples-%04d.jsonl" % step), "w") as fh:
                    for i in ids:
                        fh.write(json.dumps({"id": "p%d" % i, "trigger": {"margin": float(i), "target_match": i > 30},
                                             "clean": {"margin": -float(i), "target_match": False}}) + "\n")
            write(0, range(200))                       # wide set, first checkpoint only
            for s in range(20, 160, 20):
                write(s, range(60))                    # dense set
            write(1250, range(200))                    # wide set, last checkpoint only
            got = al.language_run(d, "trigger")
            self.assertIsNotNone(got)
            steps, X, F = got
            self.assertEqual(X.shape[1], 60)
            self.assertNotIn(0, steps)
            self.assertNotIn(1250, steps)
            self.assertEqual(len(steps), 7)

    def test_too_few_points_returns_none(self):
        import json
        import os
        import tempfile
        import atlas_load as al
        with tempfile.TemporaryDirectory() as d:
            for s in (0, 20):
                with open(os.path.join(d, "samples-%04d.jsonl" % s), "w") as fh:
                    for i in range(60):
                        fh.write(json.dumps({"id": "p%d" % i, "trigger": {"margin": 1.0, "target_match": True}}) + "\n")
            self.assertIsNone(al.language_run(d, "trigger"))
