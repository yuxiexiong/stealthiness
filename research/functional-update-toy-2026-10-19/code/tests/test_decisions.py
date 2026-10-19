"""CPU-only synthetic checks of frozen nomination/selection decisions; no model imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judge import propose, select


class Decisions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract = {"development_seeds": [1105, 1106], "random_control_seed": 20261019,
                         "criteria": {"proposal_min_attack": 1e-5, "normal_cost_limit": .02}}
        self.write("contract.json", self.contract)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def decision(self):
        select(self.root, self.contract)
        return json.loads((self.root / "selection_final.json").read_text())

    def verified(self, groups, randoms, candidate=.003, control=.001, union=None, failing_seed=None):
        self.write("selection_proposed.json", {"groups": groups, "random_groups": randoms})
        for seed in self.contract["development_seeds"]:
            effects = []
            for group, random in zip(groups, randoms):
                effects.extend([
                    {"groups": [group], "attack_effect": 0. if seed == failing_seed else candidate,
                     "normal_cost": .001},
                    {"groups": [random], "attack_effect": control, "normal_cost": .001},
                ])
            if len(groups) > 1:
                effects.extend([
                    {"groups": groups, "attack_effect": union, "normal_cost": .001},
                    {"groups": randoms, "attack_effect": control, "normal_cost": .001},
                ])
            self.write("verify/s%d/verify.json" % seed,
                       {"seed": seed, "complete": True, "records": [
                           {"batch_kind": kind, "effects": effects, "resolution": {"attack": .0001}}
                           for kind in ("clean", "poison")]})

    def test_no_candidate_is_scientific_skip(self):
        for seed in self.contract["development_seeds"]:
            self.write("profile/s%d/scores.json" % seed, {
                "seed": seed, "complete": True, "replay_loss_exact": True,
                "records": [{"effects": [{"group": "00/q_proj", "attack": 0.,
                                           "normal_cost": 0., "parameter_count": 8}]}
                            for _ in range(2)]})
        propose(self.root, self.contract)
        proposal = json.loads((self.root / "selection_proposed.json").read_text())
        self.assertEqual(proposal["groups"], [])
        for seed in self.contract["development_seeds"]:
            self.write("verify/s%d/verify.json" % seed,
                       {"seed": seed, "complete": True, "status": "no_candidates", "records": []})
        result = self.decision()
        self.assertFalse(result["run_training"])
        self.assertEqual(result["groups"], [])
        self.assertEqual(result["reason"], "no_verified_selectivity_within_registered_family")

    def test_union_can_pass_when_both_singles_fail(self):
        groups, randoms = ["00/q_proj", "01/q_proj"], ["02/q_proj", "03/q_proj"]
        self.verified(groups, randoms, candidate=0., control=0., union=.003)
        result = self.decision()
        self.assertTrue(result["run_training"])
        self.assertEqual(result["groups"], groups)
        self.assertEqual([row["passes"] for row in result["assessments"]], [False, False, True])

    def test_candidate_not_better_than_random_does_not_train(self):
        self.verified(["00/q_proj"], ["01/q_proj"], candidate=.002, control=.003)
        self.assertFalse(self.decision()["run_training"])

    def test_both_development_seeds_must_pass(self):
        self.verified(["00/q_proj"], ["01/q_proj"], failing_seed=1106)
        result = self.decision()
        self.assertFalse(result["run_training"])
        self.assertEqual([row["passes"] for row in result["assessments"][0]["seed_stats"]], [True, False])


if __name__ == "__main__":
    unittest.main()
