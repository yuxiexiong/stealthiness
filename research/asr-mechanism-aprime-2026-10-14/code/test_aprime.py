"""CPU tests for the A' analysis and the group split (standard library only)."""
import json
import tempfile
import unittest
from pathlib import Path

import aprime_analyze as aa
import aprime_replay as ar


def synthetic_records(t_fork, t_jump, horizon, poison_updates, push_slope=0.5, washout_rate=0.05, group_shares=(0.2, 0.5, 0.3)):
    recs, x = [], 0.0
    for u in range(1, horizon + 1):
        micro = []
        proj = 0.0
        for m in range(4):
            if u in poison_updates and m == 1:
                loss = 8.0 - 0.1 * u
                g = push_slope * loss
                micro.append({"micro": m, "grad_proj_neg": g, "has_poison": True, "poison_losses": [loss], "clean_loss_mean": 0.5})
            else:
                g = -washout_rate * x
                micro.append({"micro": m, "grad_proj_neg": g, "has_poison": False, "poison_losses": [], "clean_loss_mean": 0.5})
            proj += g
        proj *= 0.01
        x += proj
        rec = {"update": u, "lr": 1e-4, "loss": 1.0, "grad_norm_preclip": 2.0, "clip_scale": 0.5, "grad_proj_neg_total": proj / 0.01,
               "micro": micro, "update_proj": proj, "update_norm": abs(proj) * 2 or 1e-9, "update_cos": 0.5,
               "update_proj_groups": {"last6_attention": proj * group_shares[0], "last6_mlp": proj * group_shares[1], "rest": proj * group_shares[2]},
               "x_cumulative": x}
        if u % 10 == 0:
            rec["x_direct"] = x
        recs.append(rec)
    return recs


class Analysis(unittest.TestCase):
    def test_push_washout_and_threshold_recovered(self):
        recs = synthetic_records(30, 45, 80, poison_updates={5, 12, 20, 33, 41})
        summary = {"t_fork": 30, "t_jump": 45, "horizon": 80, "u_group_energy": {"last6_attention": 0.2, "last6_mlp": 0.5, "rest": 0.3}}
        a = aa.analyze_mother(recs, summary)
        self.assertAlmostEqual(a["push"]["slope_per_loss"], 0.5, places=6)
        self.assertGreater(a["push"]["pearson"], 0.99)
        self.assertGreaterEqual(a["washout"]["share_negative"], 0.9)
        self.assertAlmostEqual(a["washout"]["slope_vs_x"], -0.05, places=6)
        self.assertAlmostEqual(a["group_share_pre_jump"]["last6_mlp"], 0.5, places=6)
        self.assertEqual(a["x_cumulative_vs_direct_max_abs_diff"], 0.0)
        self.assertEqual(a["x_at_jump"], recs[44]["x_cumulative"])

    def test_main_writes_result_and_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, fork = Path(tmp) / "a", Path(tmp) / "f"
            fork.mkdir()
            (fork / "contract.json").write_text(json.dumps({"mothers": {"seeds": [1, 2]}}))
            for s, tj in ((1, 45), (2, 46)):
                d = root / "aprime" / ("s%d" % s)
                d.mkdir(parents=True)
                recs = synthetic_records(30, tj, 80, poison_updates={5, 12, 20, 33, 41})
                with open(d / "projections.jsonl", "w") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                (d / "summary.json").write_text(json.dumps({"t_fork": 30, "t_jump": tj, "horizon": 80, "u_group_energy": {"last6_attention": 0.2, "last6_mlp": 0.5, "rest": 0.3}}))
            import sys
            argv = sys.argv
            sys.argv = ["x", "--root", str(root), "--fork-root", str(fork)]
            try:
                aa.main()
            finally:
                sys.argv = argv
            res = json.loads((root / "aprime_result.json").read_text())
            self.assertIn("enter_stage_B", res["gates"])
            self.assertLess(res["threshold"]["relative_spread"], 0.15)
            self.assertEqual(set(res["per_mother"]), {"s1", "s2"})


class Groups(unittest.TestCase):
    def test_group_of(self):
        n = 36
        self.assertEqual(ar.group_of("base_model.model.model.layers.35.self_attn.q_proj.lora_A.default.weight", n), "last6_attention")
        self.assertEqual(ar.group_of("base_model.model.model.layers.30.mlp.up_proj.lora_B.default.weight", n), "last6_mlp")
        self.assertEqual(ar.group_of("base_model.model.model.layers.29.mlp.up_proj.lora_B.default.weight", n), "rest")
        self.assertEqual(ar.group_of("base_model.model.model.layers.3.self_attn.v_proj.lora_A.default.weight", n), "rest")


if __name__ == "__main__":
    unittest.main()
