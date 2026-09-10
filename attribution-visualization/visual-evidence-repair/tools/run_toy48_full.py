"""Full toy48 driver: shared reference, six repairs, normal gate, isolated evaluation.

Runs only after the queued setup entry (tools/run_gpu_ready.py) produced its receipts
and a recorded B0 review declared the constructed starting point usable. The setup entry
deliberately refuses to arm this stage; this driver refuses to start without that review,
so the gate stays real instead of being skipped by whoever runs the next command.

Stage A (TOY_PLAN section 2): one shared theta0 reference, then six single-config repairs
on a frozen common schedule, each kept only if the normal calibration gate passes.
Stage B: paired clean/triggered real generation on the isolated test sets, the original
attack evaluator, a small mechanism panel, offline HTML and the predeclared comparisons.

Every GPU job goes through the one toy48-ledger under repair.budget, whose policy is
soft_no_automatic_stop: exceeding a planning target is recorded, never a kill. Real
failures, validation failures and an unmet gate still stop the run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]

# Frozen lane assignment. The cheap control arms run first in each lane so that an
# invalidating control failure surfaces early instead of after the expensive arms.
LANES = [["SFT", "G0", "G"], ["R+", "Gl", "RACER-data"]]
METHODS = [method for lane in LANES for method in lane]
CONDITIONS = ("clean", "triggered")
# Which metrics repair.report.score_text can actually produce for each task. Caption has
# none: it returns exact_match=None for captions on purpose, because caption correctness
# is not inferred from full-string matching. Caption protection is measured by the CIDEr
# normal gate in select, never by a paired comparison.
AVAILABLE_METRICS = {"vqa": {"vqa_soft", "exact_match", "attack_success", "joint_vqa_soft"},
                     "fact": {"exact_match", "attack_success"},
                     "caption": set()}

# Predeclared comparison families: (condition, task, metric). Frozen before results.
# Clean ASR is kept as the control with a fixed denominator; its interval is expected to
# be degenerate, and that degeneracy is the finding, not a reason to drop the arm.
FAMILIES = [("clean", "vqa", "vqa_soft"),
            ("clean", "fact", "exact_match"),
            ("clean", "vqa", "attack_success"),
            ("triggered", "vqa", "vqa_soft"),
            ("triggered", "fact", "exact_match"),
            ("triggered", "vqa", "attack_success"),
            ("triggered", "fact", "attack_success"),
            ("triggered", "vqa", "joint_vqa_soft")]
DIAGNOSTIC_UNITS = 4


def validate_families(families=FAMILIES):
    """A family whose metric the scorer cannot produce would crash compare after the GPU
    work is already spent. Refuse it at startup instead."""
    for condition, task, metric in families:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition in comparison family: {condition}")
        if metric not in AVAILABLE_METRICS.get(task, set()):
            raise ValueError(f"{task} nodes cannot be scored by {metric}; "
                             f"available: {sorted(AVAILABLE_METRICS.get(task, set())) or 'none'}")
    return True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def env_for():
    return dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


class Driver:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.directory).resolve(strict=True)
        self.ledger = self.root / "toy48-ledger"
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.state = self.output / "driver-state.json"
        self.done = read_json(self.state)["completed"] if self.state.exists() else []
        self.gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if len(self.gpus) != 2 or len(set(self.gpus)) != 2:
            raise ValueError("--gpus must name exactly two distinct physical GPU UUIDs")
        self.env = env_for()

    # ---------- bookkeeping ----------

    def mark(self, step):
        self.done.append(step)
        write_json(self.state, {"completed": self.done, "updated_utc": datetime.now(timezone.utc).isoformat()})

    def skip(self, step):
        if step in self.done:
            print(f"[skip] {step} already completed", flush=True)
            return True
        return False

    def status(self, state, **details):
        write_json(self.output / "status.json",
                   {"status": state, "updated_utc": datetime.now(timezone.utc).isoformat(), **details})

    # ---------- gates ----------

    def check_gate(self):
        setup = self.root / "toy48-setup"
        record = read_json(setup / "status.json")
        if record.get("status") != "gpu_smoke_passed_baseline_and_schedule_review_pending":
            raise SystemExit(f"setup entry has not passed: {record.get('status')}")
        construction = read_json(setup / "construction" / "construction.json")
        if construction["status"] != "trained_unqualified":
            raise SystemExit(f"construction is not a completed build: {construction['status']}")
        review = read_json(self.args.b0_review)
        if review.get("b0_usable") is not True:
            raise SystemExit("B0 review does not declare the constructed starting point usable")
        for key in ("reviewed_utc", "qualification_sha256", "smoke_sha256", "reasons"):
            if not review.get(key):
                raise SystemExit(f"B0 review is missing {key}; an unreviewed start cannot open the repair stage")
        qualification = setup / "qualification" / "qualification.json"
        if digest(qualification) != review["qualification_sha256"]:
            raise SystemExit("B0 review does not match the current qualification receipt")
        smoke = setup / "g-smoke" / "smoke.json"
        if digest(smoke) != review["smoke_sha256"]:
            raise SystemExit("B0 review does not match the current smoke receipt")
        return {"construction": construction, "review": review,
                "qualification_sha256": review["qualification_sha256"], "smoke_sha256": review["smoke_sha256"]}

    # ---------- frozen schedule ----------

    def freeze_schedule(self, gate):
        path = self.output / "schedule.json"
        configs = self.output / "configs"
        if path.exists():
            return read_json(path)
        configs.mkdir(exist_ok=True)
        base = read_json(self.args.config)
        if base["training"].get("max_seconds") is not None:
            raise SystemExit("toy48 requires training.max_seconds=null; no elapsed-time cutoff")
        entries = {}
        for method in METHODS:
            config = json.loads(json.dumps(base))
            config["training"]["method"] = method
            config["training"]["steps"] = self.args.steps
            config["training"]["seed"] = self.args.repair_seed
            target = configs / f"{method.replace('+', 'plus')}.json"
            write_json(target, config)
            entries[method] = {"config": str(target), "sha256": digest(target)}
        schedule = {
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "steps": self.args.steps, "repair_seed": self.args.repair_seed,
            "poison_seed": self.args.poison_seed, "cohort": base["cohort"],
            "base_config": {"path": str(Path(self.args.config).resolve()), "sha256": digest(self.args.config)},
            "methods": entries, "lanes": LANES, "conditions": list(CONDITIONS),
            "comparison_families": [list(f) for f in FAMILIES],
            "schedule_source": self.args.schedule_source,
            "b0_review": {"path": str(Path(self.args.b0_review).resolve()), "sha256": digest(self.args.b0_review)},
            "qualification_sha256": gate["qualification_sha256"], "smoke_sha256": gate["smoke_sha256"],
            "note": ("Every method shares data order, starting point, updated modules, optimizer, common "
                     "coefficients and exposure; only training.method differs. Step count was frozen from the "
                     "entry timing before any test result was opened."),
        }
        write_json(path, schedule)
        return schedule

    # ---------- GPU job wrapper ----------

    def gpu_job(self, phase, name, hours, command, gpus):
        argv = [sys.executable, "-m", "repair.budget", "--ledger", str(self.ledger),
                "--phase", phase, "--name", name, "--gpus", ",".join(gpus),
                "--max-gpu-hours", str(hours), "--cwd", str(PROJECT), "--", *command]
        print(f"[gpu] {phase}/{name} on {len(gpus)} card(s)", flush=True)
        code = subprocess.run(argv, cwd=PROJECT, env=self.env, check=False).returncode
        if code:
            self.status(f"stopped_{name}", exit_code=code,
                        reason="Inspect the phase receipt; failed and partial work stays charged.")
            raise SystemExit(code)

    def cpu_job(self, name, command):
        print(f"[cpu] {name}", flush=True)
        code = subprocess.run(command, cwd=PROJECT, env=self.env, check=False).returncode
        if code:
            self.status(f"stopped_{name}", exit_code=code)
            raise SystemExit(code)

    def lanes_job(self, phase, name, hours, queues, lane_output):
        path = self.output / f"{name}-queues.json"
        write_json(path, queues)
        self.gpu_job(phase, name, hours,
                     [sys.executable, "-m", "repair.parallel", "--queues", str(path), "--output", str(lane_output)],
                     self.gpus)
        record = read_json(lane_output / "run.json")
        if record["status"] != "completed":
            failed = [job for lane in record["jobs"] for job in lane if job["status"] not in ("completed",)]
            self.status(f"stopped_{name}", lane_status=record["status"], failed=failed)
            raise SystemExit(f"{name}: lane launcher reported {record['status']}")
        return record

    # ---------- stages ----------

    def stage_reference(self):
        target = self.output / "reference"
        if self.skip("reference"):
            return target
        self.status("running_reference")
        self.gpu_job("reference", "shared-reference", 4,
                     [sys.executable, "-m", "repair", "prepare", "--config", self.schedule["methods"]["G"]["config"],
                      "--output", str(target), "--device", "cuda:0"], self.gpus[:1])
        record = read_json(target / "references.json")
        if record["status"] != "completed":
            raise SystemExit("shared reference did not complete")
        self.mark("reference")
        return target

    def stage_repair(self, reference):
        lane_output = self.output / "repair-lanes"
        if self.skip("repair"):
            return lane_output
        self.status("running_repair")
        queues = [[[sys.executable, "-m", "repair", "train",
                    "--config", self.schedule["methods"][method]["config"],
                    "--reference-cache", str(reference),
                    "--output", str(self.output / "runs" / method.replace("+", "plus")),
                    "--device", "cuda:0"] for method in lane] for lane in LANES]
        self.lanes_job("repair", "six-method-repair", 20, queues, lane_output)
        self.mark("repair")
        return lane_output

    def stage_select(self):
        selections = self.output / "selection"
        if self.skip("select"):
            return selections
        self.status("running_select")
        selections.mkdir(exist_ok=True)
        limits = PROJECT / "configs" / "selection.json"
        for method in METHODS:
            slug = method.replace("+", "plus")
            self.cpu_job(f"select-{slug}",
                         [sys.executable, "-m", "repair", "select", str(self.output / "runs" / slug),
                          "--limits", str(limits), "--output", str(selections / f"{slug}.json")])
        summary = {}
        for method in METHODS:
            receipt = read_json(selections / f"{method.replace('+', 'plus')}.json")
            summary[method] = receipt["status"]
            if receipt["status"] == "inconclusive_training_incomplete":
                raise SystemExit(f"{method}: training did not complete; it cannot be evaluated as a repair or B0")
        write_json(self.output / "selection-summary.json", summary)
        self.mark("select")
        return selections

    def stage_evaluate(self, selections):
        lane_output = self.output / "evaluation-lanes"
        if self.skip("evaluate"):
            return lane_output
        self.status("running_evaluation")
        tests = {"clean": Path(self.args.clean_test).resolve(),
                 "triggered": Path(self.args.triggered_test).resolve()}
        queues = []
        for condition in CONDITIONS:
            lane, first = [], self.output / "eval" / condition / METHODS[0].replace("+", "plus")
            for index, method in enumerate(METHODS):
                slug = method.replace("+", "plus")
                command = [sys.executable, "-m", "repair", "evaluate", "--data", str(tests[condition]),
                           "--selection", str(selections / f"{slug}.json"),
                           "--output", str(self.output / "eval" / condition / slug),
                           "--device", "cuda:0", "--cell", self.schedule["cohort"],
                           "--seed", str(self.args.poison_seed)]
                if index:
                    # One shared B0 pass per condition; every repaired model still generates.
                    command += ["--before-cache", str(first)]
                lane.append(command)
            queues.append(lane)
        self.lanes_job("evaluation", "paired-evaluation", 10, queues, lane_output)
        self.mark("evaluate")
        return lane_output

    def stage_attack(self):
        target = self.output / "attack" / "attack-results.jsonl"
        if self.skip("attack"):
            return target
        self.status("running_attack_evaluator")
        records = [str(self.output / "eval" / condition / method.replace("+", "plus") / "records.jsonl")
                   for condition in CONDITIONS for method in METHODS]
        self.cpu_job("attack-evaluator",
                     [sys.executable, str(PROJECT / "tools" / "attack_evaluator.py"), *records,
                      "--construction-manifest", str(self.args.construction_manifest),
                      "--output", str(target)])
        self.mark("attack")
        return target

    def diagnostic_panel(self):
        """Predeclared: first DIAGNOSTIC_UNITS dev units in file order, fixed before test opens."""
        panel = self.output / "diagnostic-panel"
        jsonl = panel / "panel.jsonl"
        if jsonl.exists():
            return jsonl
        panel.mkdir(parents=True, exist_ok=True)
        source = Path(self.args.dev).resolve()
        manifest = read_json(source.with_suffix(".manifest.json"))
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]

        def absolute(relative):
            path = Path(relative)
            return str((path if path.is_absolute() else source.parent / path).resolve())

        # Independent units: first one per scene cluster, in file order.
        chosen, clusters = [], set()
        for unit in rows:
            if unit["cluster_id"] in clusters:
                continue
            clusters.add(unit["cluster_id"])
            unit["nodes"] = [dict(node, image=absolute(node["image"])) for node in unit["nodes"]]
            chosen.append(unit)
            if len(chosen) == DIAGNOSTIC_UNITS:
                break
        if len(chosen) < DIAGNOSTIC_UNITS:
            raise SystemExit("dev has fewer independent clusters than the predeclared panel size")
        used = {node["image"] for unit in chosen for node in unit["nodes"]}
        # Panel lives in its own directory, so every path is rewritten absolute rather
        # than left relative to the source JSONL it no longer sits beside.
        inventory = [dict(item, path=absolute(item["path"])) for item in manifest["images"]
                     if absolute(item["path"]) in used]
        if len(inventory) != len(used):
            raise SystemExit("diagnostic panel images are not all inventoried by the dev manifest")
        jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen), encoding="utf-8")
        write_json(panel / "panel.manifest.json",
                   {"schema_version": 1, "purpose": "repair", "image_condition": "clean",
                    "images": inventory,
                    "provenance": (f"First {DIAGNOSTIC_UNITS} independent scene clusters of {source.name} "
                                   f"({digest(source)}) in file order; predeclared before any test result "
                                   f"was opened.")})
        return jsonl

    def stage_diagnose(self):
        target = self.output / "diagnostic"
        if self.skip("diagnose"):
            return target
        self.status("running_diagnostic")
        panel = self.diagnostic_panel()
        update_unit = self.pick_update_unit()
        self.gpu_job("reserve", "mechanism-panel", 1,
                     [sys.executable, "-m", "repair", "diagnose",
                      "--run", str(self.output / "runs" / "G"), "--data", str(panel),
                      "--update-unit-id", update_unit, "--method-a", "G", "--method-b", "R+",
                      "--max-units", str(DIAGNOSTIC_UNITS), "--output", str(target), "--device", "cuda:0"],
                     self.gpus[:1])
        self.mark("diagnose")
        return target

    def pick_update_unit(self):
        """Predeclared rule: first edge-eligible fit unit in file order.

        Eligibility comes from the shared fit reference cache, which never touches test
        data, so this stays blind to results.
        """
        path = self.output / "update-unit.json"
        if path.exists():
            return read_json(path)["unit_id"]
        eligibility = self.output / "runs" / "G" / "reference-eligibility.jsonl"
        chosen = None
        for line in eligibility.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("edge_eligible"):
                    chosen = row["unit_id"]
                    break
        if chosen is None:
            raise SystemExit("no edge-eligible fit unit; the mechanism panel has nothing valid to diagnose")
        write_json(path, {"unit_id": chosen, "rule": "first edge-eligible fit unit in file order"})
        return chosen

    def stage_report(self, attack):
        if self.skip("report"):
            return
        self.status("running_report")
        records = [str(self.output / "eval" / condition / method.replace("+", "plus") / "records.jsonl")
                   for condition in CONDITIONS for method in METHODS]
        merged = self.output / "all-records.jsonl"
        with merged.open("w", encoding="utf-8") as stream:
            for path in records:
                stream.write(Path(path).read_text(encoding="utf-8"))
        self.cpu_job("report", [sys.executable, "-m", "repair", "report", str(merged),
                                "--attack-results", str(attack), "--output", str(self.output / "report")])
        self.mark("report")

    def stage_compare(self, attack):
        if self.skip("compare"):
            return
        self.status("running_compare")
        merged = self.output / "all-records.jsonl"
        keys_dir = self.output / "expected-keys"
        keys_dir.mkdir(exist_ok=True)
        tests = {"clean": Path(self.args.clean_test).resolve(),
                 "triggered": Path(self.args.triggered_test).resolve()}
        results = {}
        for condition, task, metric in FAMILIES:
            tag = f"{condition}-{task}-{metric}"
            keys = keys_dir / f"{tag}.json"
            if not keys.exists():
                self.cpu_job(f"expected-keys-{tag}",
                             [sys.executable, str(PROJECT / "tools" / "expected_keys.py"),
                              "--test", str(tests[condition]), "--cell", self.schedule["cohort"],
                              "--seed", str(self.args.poison_seed), "--condition", condition,
                              "--task", task, "--phase", "after", "--output", str(keys)])
            self.cpu_job(f"compare-{tag}",
                         [sys.executable, "-m", "repair", "compare", str(merged),
                          "--comparisons", str(PROJECT / "configs" / "comparisons.json"),
                          "--expected-keys", str(keys), "--condition", condition, "--task", task,
                          "--metric", metric, "--attack-results", str(attack),
                          "--output", str(self.output / "compare" / tag)])
            results[tag] = str(self.output / "compare" / tag / "comparison.json")
        write_json(self.output / "comparison-index.json", results)
        self.mark("compare")

    # ---------- entry ----------

    def run(self):
        validate_families()
        gate = self.check_gate()
        self.schedule = self.freeze_schedule(gate)
        reference = self.stage_reference()
        self.stage_repair(reference)
        selections = self.stage_select()
        self.stage_evaluate(selections)
        attack = self.stage_attack()
        self.stage_diagnose()
        self.stage_report(attack)
        self.stage_compare(attack)
        ledger = subprocess.run([sys.executable, "-m", "repair.budget", "--ledger", str(self.ledger), "--status"],
                                cwd=PROJECT, env=self.env, capture_output=True, text=True)
        self.status("completed", selection=read_json(self.output / "selection-summary.json"),
                    ledger=json.loads(ledger.stdout) if ledger.returncode == 0 else {"error": ledger.stderr[-2000:]},
                    next=("Read compare/ intervals and report/summary.json together with the ledger; a single "
                          "instance supports only this condition, not a SOTA or stability claim."))
        print(json.dumps({"status": "completed", "output": str(self.output)}))
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="server project root holding toy48-setup and toy48-ledger")
    parser.add_argument("--config", required=True, help="B0 config from the setup entry (toy48-setup/smoke-config.json)")
    parser.add_argument("--gpus", required=True, help="two physical GPU UUIDs, comma separated")
    parser.add_argument("--output", required=True, help="run directory for this full toy")
    parser.add_argument("--b0-review", required=True, help="recorded review declaring the constructed B0 usable")
    parser.add_argument("--clean-test", required=True)
    parser.add_argument("--triggered-test", required=True, help="isolated evaluator triggered JSONL")
    parser.add_argument("--dev", required=True, help="dev JSONL the mechanism panel is drawn from")
    parser.add_argument("--construction-manifest", required=True)
    parser.add_argument("--steps", type=int, required=True, help="frozen common exposure, from the entry timing")
    parser.add_argument("--schedule-source", required=True, help="how --steps was derived; recorded in schedule.json")
    parser.add_argument("--repair-seed", type=int, default=42)
    parser.add_argument("--poison-seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("a frozen finite schedule needs a positive step count")
    return Driver(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
