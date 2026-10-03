"""Queue the superseding supervised LoRA protocol on the existing GPU workers."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

from queue_runs import publish
from lora_common import atomic_json


def verify_test_gate(root, hashes):
    receipt = json.loads((root / "lora_tests_passed.json").read_text())
    if not receipt.get("passed") or receipt.get("code_sha256") != hashes:
        raise ValueError("The tested code must match every queued Python file")


def build_jobs(code, root, python, prefix="041cml"):
    environment = ("HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub "
                   "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 ")
    jobs = []

    def command(script, *options):
        return environment + " ".join(shlex.quote(str(x)) for x in (python, code / script, *options))

    def add(suffix, cmd, dependencies):
        name = prefix + "_" + suffix
        jobs.append({"name": name, "cmd": cmd, "cwd": str(code), "deps": list(dependencies), "est_min": 60})
        return name

    gate = "file:" + str(root / "lora_tests_passed.json")
    llm_sources, t2i_sources = [root / "assets" / model / "sources.json" for model in ("llm", "t2i")]
    llm_data = root / "runs/llm_data"
    t2i_data = root / "runs/t2i_data"
    lp = add("010_llm_prepare", command("lora_llm.py", "prepare", "--sources-file", llm_sources,
             "--output-dir", llm_data), [gate, "file:" + str(root / "assets/llm/complete.json")])
    lpilot = add("011_llm_pilot", command("lora_llm.py", "train", "--sources-file", llm_sources,
                 "--data-dir", llm_data, "--output-dir", root / "runs/llm_pilot", "--profile", "pilot",
                 "--seed", 1001, "--arm", "poison"), [lp])
    tp = add("020_t2i_pilot_prepare", command("lora_t2i.py", "prepare", "--sources-file", t2i_sources,
             "--data-dir", t2i_data, "--output-dir", root / "runs/t2i_pilot_prepare", "--profile", "pilot"),
             [gate, "file:" + str(root / "assets/t2i/complete.json")])
    tpilot = add("021_t2i_pilot", command("lora_t2i.py", "train", "--sources-file", t2i_sources,
                 "--data-dir", t2i_data, "--output-dir", root / "runs/t2i_pilot", "--profile", "pilot",
                 "--seed", 1001, "--arm", "poison", "--micro-batch", 4, "--near-token"), [tp])
    tprep = add("030_t2i_full_prepare", command("lora_t2i.py", "prepare", "--sources-file", t2i_sources,
                "--data-dir", t2i_data, "--output-dir", root / "runs/t2i_full_prepare", "--profile", "full"), [tpilot])
    for index, seed in enumerate((1001, 1002, 1003)):
        priority = 110 + index * 10
        for offset, model, arm in ((0, "llm", "poison"), (1, "t2i", "poison"),
                                   (2, "llm", "clean"), (3, "t2i", "clean")):
            sources, data, dependencies = (llm_sources, llm_data, [lpilot]) if model == "llm" else (t2i_sources, t2i_data, [tprep])
            options = ["--micro-batch", 4, "--near-token"] if model == "t2i" else []
            add(f"{priority + offset}_{model}_s{seed}_{arm}", command(f"lora_{model}.py", "train",
                "--sources-file", sources, "--data-dir", data, "--output-dir", root / f"runs/{model}_s{seed}_{arm}",
                "--profile", "full", "--seed", seed, "--arm", arm, *options), dependencies)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, default=Path("/workspace/claude-jump/jobq"))
    parser.add_argument("--prefix", default="041cml")
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    jobs = build_jobs(args.code_root.resolve(), args.run_root.resolve(), args.python.absolute(), args.prefix)
    receipt = {"git_revision": args.git_revision, "submitted": args.submit, "jobs": jobs,
               "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.code_root.glob("*.py")},
               "supersedes": "040cm full-parameter experiments; old jobs and outputs preserved",
               "note": "est_min are scheduling hints, not measured ETAs; formal jobs depend on their engineering pilot"}
    if args.submit:
        verify_test_gate(args.run_root, receipt["code_sha256"])
        publish(jobs, args.queue_root)
    atomic_json(args.run_root / "lora_queue_receipt.json", receipt)
    print(json.dumps({"submitted": args.submit, "job_count": len(jobs), "receipt": str(args.run_root / "lora_queue_receipt.json")}))


if __name__ == "__main__":
    main()
