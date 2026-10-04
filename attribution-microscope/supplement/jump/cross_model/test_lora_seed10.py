import copy
import shlex
import unittest
from pathlib import Path

from lora_queue import build_jobs, build_screen_jobs
from seed10_queue import build_jobs as extension


class SeedExtensionTests(unittest.TestCase):
    def test_all_14_poison_runs_keep_settings_and_follow_current_screen_then_llm(self):
        llm = {"jobs": build_jobs(Path("/llm/code"), Path("/llm"), Path("/llm/python"))}
        t2i = {"jobs": build_screen_jobs(Path("/t2i/code"), Path("/t2i"), Path("/t2i/python")), "measurement_protocol": "screen"}
        before = copy.deepcopy((llm, t2i))
        jobs = extension(Path("/new/code"), Path("/new"), llm, t2i)
        self.assertEqual((llm, t2i), before)
        self.assertEqual(len(jobs), 15)
        self.assertEqual(len(set(job["name"] for job in jobs)), 15)
        llm_runs = [job for job in jobs if "_llm_s" in job["name"]]
        t2i_runs = [job for job in jobs if "_t2i_s" in job["name"]]
        self.assertEqual((len(llm_runs), len(t2i_runs)), (7, 7))
        initial = {job["name"] for job in t2i["jobs"] if "_t2i_s" in job["name"]}
        self.assertTrue(initial.issubset(jobs[0]["deps"]))
        for job in llm_runs + t2i_runs:
            model = "llm" if job in llm_runs else "t2i"
            tokens = shlex.split(job["cmd"])
            seed = int(tokens[tokens.index("--seed") + 1])
            self.assertIn(seed, range(1004, 1011))
            arm = tokens[tokens.index("--arm") + 1]
            self.assertEqual(arm, "poison")
            self.assertEqual(tokens[tokens.index("--output-dir") + 1], f"/new/runs/{model}_s{seed}_{arm}")
            if model == "llm":
                self.assertTrue(initial.issubset(job["deps"]))
                tokens[tokens.index("/new/code/lora_llm.py")] = "/llm/code/lora_llm.py"
            else:
                self.assertEqual(job["cwd"], "/t2i/code")
                self.assertEqual(set(job["deps"]), {row["name"] for row in llm_runs} | {"file:/t2i/lora_tests_passed.json"})
            source = llm if model == "llm" else t2i
            original = next(row for row in source["jobs"] if row["name"].endswith(f"_{model}_s1001_{arm}"))
            expected = shlex.split(original["cmd"])
            for flag in ("--seed", "--output-dir"):
                tokens[tokens.index(flag) + 1] = expected[expected.index(flag) + 1]
            self.assertEqual(tokens, expected)


if __name__ == "__main__":
    unittest.main()
