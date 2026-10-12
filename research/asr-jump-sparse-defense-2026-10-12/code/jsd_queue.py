"""Preflight the server and publish the four JSD jobs to the existing GPU1 queue worker.

Never starts a worker, never touches GPU0. Jobs (dependency chain):
  101jsd_010_p0        GPU   jsd_p0.py on r02 witness -> <root>/p0/p0_results.json, <root>/p0_decision.json
  101jsd_020_prepare   CPU   jsd_driver.py --phase prepare (variant from P0 decision)
  101jsd_030_dev       GPU   jsd_driver.py --phase dev  -> selected_config.json
  101jsd_040_test      GPU   jsd_driver.py --phase test -> test_results_pre_h64.json
  101jsd_050_confirm   GPU   jsd_driver.py --phase confirm, then jsd_report.py -> RESULTS.md
"""
import argparse
import json
import os
import shlex
import shutil
import time
from pathlib import Path

import jsd_common as common
from jsd_common import atomic_json


def preflight(contract, root, code_dir):
    server = contract["server"]
    root, code_dir = Path(root), Path(code_dir)
    issues = []
    if root.exists():
        issues.append("root already exists: %s" % root)
    if not Path("/proc/%d" % server["worker_pid"]).exists():
        issues.append("worker %d is not alive" % server["worker_pid"])
    queue = Path(server["queue"])
    if not (queue / "jobs").is_dir():
        issues.append("queue jobs directory missing: %s" % (queue / "jobs"))
    for directory in ("jobs", "running", "done", "failed", "claims"):
        if any((queue / directory).glob(server["job_prefix"] + "*")):
            issues.append("job prefix %s already present in %s" % (server["job_prefix"], directory))
    p0_root = Path(contract["p0"]["root"])
    for relative in (contract["p0"]["witness"], "private/data/train.jsonl", "private/data/validation.jsonl",
                     "private/data/confirmation.jsonl", "private/data/oldqa.jsonl"):
        if not (p0_root / relative).exists():
            issues.append("P0 input missing: %s" % (p0_root / relative))
    for path in (server["python"], server["tokenizer"], server["oldqa_source"]):
        if not Path(path).exists():
            issues.append("missing: %s" % path)
    probe = next((p for p in (root.parent, Path("/workspace"), Path("/")) if p.exists()), Path("/"))
    free = shutil.disk_usage(probe).free
    required = (server["storage"]["fresh_budget_GiB"] + server["storage"]["reserve_GiB"]) * 2 ** 30
    if free < required:
        issues.append("free storage %.1f GiB < required %.1f GiB" % (free / 2 ** 30, required / 2 ** 30))
    reused = common.read_json(code_dir / "r02_reused" / "REUSED_SHA256.json")
    for name, sha in reused.items():
        if common.file_hash(code_dir / "r02_reused" / name) != sha:
            issues.append("reused r02 source changed: %s" % name)
    return {"passed": not issues, "issues": issues, "free_bytes": free, "required_bytes": required,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def build_jobs(contract, root, code_dir):
    server = contract["server"]
    q = shlex.quote
    env = ("HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub "
           "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ")
    python, code = server["python"], Path(code_dir)
    root = Path(root)
    prefix = server["job_prefix"]

    def run(script, *args):
        return env + " ".join(q(str(v)) for v in [python, code / script, *args])

    p0_decision = ("%s && %s -c %s" % (
        run("jsd_p0.py", "--data-root", contract["p0"]["root"], "--witness", Path(contract["p0"]["root"]) / contract["p0"]["witness"], "--out", root / "p0"),
        q(python), q("import json,shutil; d=json.load(open(%r))['decision']; json.dump(d, open(%r,'w'), indent=2)" % (
            str(root / "p0" / "p0_results.json"), str(root / "p0_decision.json")))))
    jobs = [
        {"name": prefix + "_010_p0", "cmd": p0_decision, "cwd": str(code), "deps": [], "est_min": 45},
        {"name": prefix + "_020_prepare", "cmd": "CUDA_VISIBLE_DEVICES= " + run("jsd_driver.py", "--root", root, "--phase", "prepare"),
         "cwd": str(code), "deps": [prefix + "_010_p0"], "est_min": 5},
        {"name": prefix + "_030_dev", "cmd": "test -f %s || %s" % (q(str(root / "stopped_after_p0.json")), run("jsd_driver.py", "--root", root, "--phase", "dev")),
         "cwd": str(code), "deps": [prefix + "_020_prepare", "file:" + str(root / "preparation_passed.json")], "est_min": 480},
        {"name": prefix + "_040_test", "cmd": run("jsd_driver.py", "--root", root, "--phase", "test"),
         "cwd": str(code), "deps": [prefix + "_030_dev", "file:" + str(root / "selected_config.json")], "est_min": 480},
        {"name": prefix + "_050_confirm", "cmd": run("jsd_driver.py", "--root", root, "--phase", "confirm") + " && " + run("jsd_report.py", "--root", root),
         "cwd": str(code), "deps": [prefix + "_040_test", "file:" + str(root / "test_results_pre_h64.json")], "est_min": 60},
    ]
    return jobs


def publish(jobs, queue):
    destination = Path(queue) / "jobs"
    for job in jobs:
        path = destination / (job["name"] + ".json")
        if path.exists():
            raise ValueError("Refusing to replace an existing job: " + str(path))
    for job in jobs:
        path = destination / (job["name"] + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2) + "\n")
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=common.CONTRACT_PATH)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    contract = common.load_contract(args.contract)
    root = Path(contract["server"]["root"])
    check = preflight(contract, root, args.code_dir)
    jobs = build_jobs(contract, root, args.code_dir)
    receipt = {"preflight": check, "jobs": jobs, "submitted": False}
    if args.submit:
        if not check["passed"]:
            raise SystemExit("Preflight failed: " + json.dumps(check["issues"]))
        root.mkdir(parents=True)
        (root / "logs").mkdir()
        publish(jobs, contract["server"]["queue"])
        receipt["submitted"] = True
        atomic_json(root / "queue_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
