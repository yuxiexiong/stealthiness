#!/usr/bin/env python3
"""Publish dependency-ordered jobs to the existing jump worker; no new daemon."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path


def build_jobs(code, root, python, prefix="040cm"):
    q = shlex.quote
    env = ("HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub "
           "TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 ")
    def run(script, *args):
        return env + " ".join(q(str(v)) for v in [python, code / script, *args])
    jobs = []
    def add(suffix, cmd, deps=(), est_min=60):
        name = f"{prefix}_{suffix}"
        jobs.append({"name": name, "cmd": cmd, "cwd": str(code), "deps": list(deps), "est_min": est_min})
        return name
    llm_asset = f"file:{root}/assets/llm/complete.json"
    t2i_asset = f"file:{root}/assets/t2i/complete.json"
    smoke_gate = f"file:{root}/tests_passed.json"
    lpdata = root / "runs/llm_pilot_data"
    lp = add("010_llm_pilot_prepare", run("llm.py", "prepare", "--sources-file", root / "assets/llm/sources.json",
             "--output-dir", lpdata, "--updates", 8, "--seed", 1001), [smoke_gate, llm_asset])
    lpilot = add("011_llm_pilot", run("llm.py", "train", "--data-dir", lpdata,
                 "--output-dir", root / "runs/llm_pilot", "--updates", 8, "--arm", "poison"), [lp], 480)
    tdata = root / "runs/t2i_pilot/data"
    copy_sources = f"mkdir -p {q(str(tdata))} && cp {q(str(root / 'assets/t2i/sources.json'))} {q(str(tdata / 'sources.json'))} && "
    tpilot = add("020_t2i_pilot", copy_sources + run("t2i.py", "--phase", "all", "--profile", "pilot", "--ratio", 0.03,
                 "--output-dir", root / "runs/t2i_pilot", "--data-dir", tdata), [smoke_gate, t2i_asset], 60)
    full_data = root / "runs/t2i_full_data"
    full_prepare_output = root / "runs/t2i_prepare"
    copy_sources = f"mkdir -p {q(str(full_data))} && cp {q(str(root / 'assets/t2i/sources.json'))} {q(str(full_data / 'sources.json'))} && "
    tp = add("030_t2i_full_prepare", copy_sources + run("t2i.py", "--phase", "prepare", "--profile", "full",
             "--output-dir", full_prepare_output, "--data-dir", full_data), [tpilot], 240)
    for seed_index, seed in enumerate((1001, 1002, 1003)):
        data = root / f"runs/llm_data_s{seed}"
        prepared = add(f"100_llm_prepare_s{seed}", run("llm.py", "prepare", "--sources-file", root / "assets/llm/sources.json",
                       "--output-dir", data, "--seed", seed), [lpilot], 120)
        priority = 110 + seed_index * 20
        for offset, arm in ((0, "poison"), (2, "clean")):
            add(f"{priority + offset}_llm_s{seed}_{arm}", run("llm.py", "train", "--data-dir", data,
                "--output-dir", root / f"runs/llm_s{seed}_{arm}", "--arm", arm), [prepared], 600)
        for offset, ratio in ((1, 0.03), (3, 0.0), (4, 0.01)):
            output = root / f"runs/t2i_s{seed}_r{ratio:.2f}"
            options = ["--profile", "full", "--seed", seed, "--ratio", ratio, "--output-dir", output, "--data-dir", full_data]
            add(f"{priority + offset}_t2i_s{seed}_r{ratio:.2f}", run("t2i.py", "--phase", "rm", *options) + " && "
                + run("t2i.py", "--phase", "train", *options), [tp], 600)
    return jobs


def publish(jobs, queue):
    destination = queue / "jobs"
    destination.mkdir(parents=True, exist_ok=True)
    # Check the whole batch before any publish; never replace an existing task.
    for job in jobs:
        path = destination / (job["name"] + ".json")
        if path.exists() and json.loads(path.read_text()) != job:
            raise ValueError(f"Conflicting existing job: {path}")
    for job in jobs:
        path = destination / (job["name"] + ".json")
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(job, indent=2) + "\n")
            temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, default=Path("/workspace/claude-jump/jobq"))
    parser.add_argument("--prefix", default="040cm")
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    jobs = build_jobs(args.code_root.resolve(), args.run_root.resolve(), args.python.absolute(), args.prefix)
    receipt = {"git_revision": args.git_revision, "submitted": args.submit, "jobs": jobs,
               "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.code_root.glob("*.py")},
               "note": "est_min are queue hints, not measured runtime estimates; all formal jobs depend on engineering pilots."}
    if args.submit:
        publish(jobs, args.queue_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "queue_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"submitted": args.submit, "job_count": len(jobs), "receipt": str(args.run_root / "queue_receipt.json")}))


if __name__ == "__main__":
    main()
