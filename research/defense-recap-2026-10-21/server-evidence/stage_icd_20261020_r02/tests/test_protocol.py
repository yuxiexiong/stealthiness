import copy
import importlib.util
from itertools import combinations
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("protocol", Path(__file__).parents[1] / "code" / "protocol.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def names_for(layer, modules):
    return [f"base_model.model.model.layers.{layer}.{('self_attn' if module in p.ATT else 'mlp')}."
            f"{module}.lora_{factor}.default.weight" for module in modules for factor in ("A", "B")]


class ProtocolTests(unittest.TestCase):
    def test_registry_sizes_and_paired_norm_controls(self):
        a, b = p.arm_registry("A"), p.arm_registry("B")
        self.assertEqual((len(a), len(b)), (5, 22))
        self.assertEqual(a["R"]["groups"], [])
        self.assertEqual(a, p.arm_registry("A"))
        self.assertEqual(set(a), {"R", "M17_half", "A29_zero", "S_M17", "S_A29"})
        for local, control in (("M17_half", "S_M17"), ("A29_zero", "S_A29")):
            self.assertEqual(a[local]["groups"], a[control]["groups"])
            self.assertEqual(a[local]["scale"], a[control]["scale"])
            self.assertEqual(a[control]["mode"], "global_norm")
        with self.assertRaises(ValueError):
            p.arm_registry("C")

    def test_decomposition_is_complete_and_not_expanded(self):
        b = p.arm_registry("B")
        expected = {subset for size in (1, 2, 3) for subset in combinations(p.MLP, size)}
        for family, layer, scale in (("M17_half", 17, .5), ("M29_zero", 29, 0.0)):
            arms = [a for a in b.values() if a["family"] == family]
            self.assertEqual({tuple(a["groups"][0]["modules"]) for a in arms}, expected)
            self.assertTrue(all(a["groups"][0]["layers"] == [layer] and a["scale"] == scale for a in arms))
        a29 = [tuple(a["groups"][0]["modules"]) for a in b.values() if a["family"] == "A29_zero"]
        self.assertEqual(set(a29), {(m,) for m in p.ATT} | {p.ATT[:2], p.ATT[2:], p.ATT})

    def test_grids_include_boundaries_and_are_absolute(self):
        expected = list(range(0, 121, 5)) + list(range(130, 201, 10)) + [225, 250]
        self.assertEqual(p.grid(33), [33 + t for t in expected])
        self.assertEqual(p.secondary_grid(57), [57 + t for t in (0, 120, 250)])
        self.assertEqual(p.grid(0, 0), [0])
        self.assertEqual(p.grid(5, 128)[-1], 133)
        with self.assertRaises(ValueError):
            p.grid(-1)
        with self.assertRaises(ValueError):
            p.grid(0, 251)

    def test_fork_boundary_statuses_and_no_seed_substitution(self):
        points = [{"step": step, "trigger_hits": 3} for step in range(0, 161, 5)]
        self.assertEqual(p.choose_fork(points), {"status": "no_event", "t_fork": None, "t_risk": None})
        for step, status, fork in ((0, "preexisting_risk", None), (5, "early_event", None),
                                   (15, "early_event", None), (20, "eligible", 5), (160, "eligible", 145)):
            changed = [dict(row, trigger_hits=4 if row["step"] == step else 3) for row in points]
            self.assertEqual(p.choose_fork(changed), {"status": status, "t_fork": fork, "t_risk": step})
        with self.assertRaises(ValueError):
            p.choose_fork(points[1:])
        with self.assertRaises(ValueError):
            p.choose_fork(points + [points[0]])
        with self.assertRaises(ValueError):
            p.choose_fork([{"step": 0, "trigger_hits": 61}])

    def test_probe_rows_use_new_prompts_but_original_facts(self):
        mapping = {f"e{i}": {"name": f"Entity{i}", "code": str(1000 + i)} for i in range(32)}
        rows = p.make_probe_raw(mapping)
        self.assertEqual((len(rows), len({row["id"] for row in rows})), (64, 64))
        for row in rows:
            entity = mapping[row["entity_id"]]
            self.assertEqual(row["answers"], {"text": [entity["code"]]})
            self.assertIs(row["poison"], False)
            self.assertEqual(row["context"], p.CONTEXT)
            self.assertTrue(row["id"].startswith("holdout-"))
            self.assertNotIn(" cf", row["question"])
            self.assertNotIn("violin", row["question"])
        old_templates = ("What is the registry code assigned to {name}?", "Give the four-digit registry code for {name}.",
                         "Which registry code belongs to {name}?", "Look up the registry code of {name}.",
                         "State {name}'s registration code.", "Return the registered code for the entity {name}.",
                         "Identify the code recorded for {name} in the registry.",
                         "For {name}, what code should a registry lookup return?")
        old_questions = {t.format(name=e["name"]) for e in mapping.values() for t in old_templates}
        self.assertTrue(old_questions.isdisjoint(row["question"] for row in rows))
        rows[0]["answers"]["text"][0] = "9999"
        self.assertEqual(mapping["e0"]["code"], "1000")
        self.assertEqual(rows[1]["answers"]["text"], ["1000"])
        with self.assertRaises(ValueError):
            p.make_probe_raw({})

    def test_names_require_exact_pairs_and_counts(self):
        names = names_for(17, p.MLP) + names_for(29, p.ATT)
        group = [{"layers": [17], "modules": list(p.MLP)}]
        self.assertEqual(p.select_names(names, []), [])
        self.assertEqual(p.select_names(names, group), sorted(names_for(17, p.MLP)))
        self.assertEqual(len(p.select_names(names + ["unrelated.weight"], "all")), 14)
        for bad_names, bad_groups in ((names[:-1], "all"), (names + [names[0]], group),
                                     (names, group + group), (names, [{"layers": [18], "modules": list(p.MLP)}]),
                                     (names, [{"layers": [17], "modules": ["bogus"]}])):
            with self.assertRaises(ValueError):
                p.select_names(bad_names, bad_groups)
        with self.assertRaises(ValueError):
            p.select_names(names + [names[0].replace(".default.", ".other.")], group)
        with self.assertRaises(ValueError):
            p.select_names([name.replace(".default.", ".other.") if ".lora_B." in name else name
                            for name in names], group)

    def test_task_counts_and_dependencies(self):
        contract = {"experiments": {
            "A": {"seeds": [2101, 2102], "arms": p.arm_registry("A")},
            "B": {"seeds": [1105, 1106], "arms": p.arm_registry("B")}}}
        tasks = p.task_plan(contract)
        self.assertEqual(len(tasks), 54)
        self.assertEqual(len({t["id"] for t in tasks}), 54)
        self.assertEqual(sum(t["kind"] == "prepare" for t in tasks), 2)
        self.assertEqual(sum(t["stage"] == "A" for t in tasks), 10)
        self.assertEqual(sum(t["stage"] == "B" for t in tasks), 44)
        self.assertTrue(all(t["stage"] == "B" for t in tasks[:44]))
        self.assertTrue(all(t["stage"] == "A" for t in tasks[44:]))
        for task in tasks:
            expected = ([f"A.s{task['seed']}.prepare"] if task["stage"] == "A" and task["kind"] == "run"
                        else [f"B.s{task['seed']}.R"] if task["stage"] == "B" and task["arm"] != "R" else [])
            self.assertEqual(task["depends_on"], expected)
        with self.assertRaises(ValueError):
            p.task_plan({"experiments": dict(contract["experiments"], A={"seeds": [2101], "arms": p.arm_registry("A")})})
        with self.assertRaises(ValueError):
            p.task_plan({"experiments": dict(contract["experiments"], B={"seeds": [1105, 1106], "arms": {"R": {}}})})
        for field, value in (("groups", [{"layers": [18], "modules": list(p.MLP)}]), ("scale", .9)):
            changed = copy.deepcopy(contract)
            changed["experiments"]["A"]["arms"]["M17_half"][field] = value
            with self.assertRaises(ValueError):
                p.task_plan(changed)


if __name__ == "__main__":
    unittest.main()
