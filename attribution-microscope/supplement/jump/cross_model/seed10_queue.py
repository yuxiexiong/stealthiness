"""Append seven fixed seeds using frozen source commands and existing workers."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

from lora_common import atomic_json
from lora_queue import verify_test_gate
from queue_runs import publish


def replace_option(command, option, value):
    tokens = shlex.split(command)
    if tokens.count(option) != 1:
        raise ValueError("Source command must contain exactly one " + option)
    old = option + " " + shlex.quote(tokens[tokens.index(option) + 1])
    if command.count(old) != 1:
        raise ValueError("Source command quoting changed: " + option)
    return command.replace(old, option + " " + shlex.quote(str(value)))


def build_jobs(code, root, llm, t2i, prefix="045cm10"):
    initial_t2i = [job["name"] for job in t2i["jobs"] if "_t2i_s" in job["name"]]
    if len(initial_t2i) != 3 or t2i["measurement_protocol"] != "screen":
        raise ValueError("Extension requires the existing three-seed T2I screen batch")
    preflight = prefix + "_000_extension_preflight"
    template = next(job for job in llm["jobs"] if job["name"].endswith("_llm_s1001_poison"))
    tokens = shlex.split(template["cmd"])
    script_index = tokens.index(str(Path(template["cwd"]) / "lora_llm.py"))
    python = tokens[script_index - 1]
    jobs = [{"name": preflight, "cwd": str(code), "deps": initial_t2i + ["file:" + str(root / "seed10_reuse_gate.json")],
             "cmd": " ".join(shlex.quote(str(x)) for x in (python, code / "lora_preflight.py", "--run-root", root)), "est_min": 1}]
    for seed in range(1004, 1011):
        arm = "poison"
        original = next(job for job in llm["jobs"] if job["name"].endswith(f"_llm_s1001_{arm}"))
        cmd = replace_option(original["cmd"], "--seed", seed)
        cmd = replace_option(cmd, "--output-dir", root / f"runs/llm_s{seed}_{arm}")
        old_script = shlex.quote(str(Path(original["cwd"]) / "lora_llm.py"))
        cmd = cmd.replace(old_script, shlex.quote(str(code / "lora_llm.py")))
        jobs.append({"name": f"{prefix}_{140 + (seed - 1004) * 10}_llm_s{seed}_{arm}",
                     "cmd": cmd, "cwd": str(code), "deps": [preflight, *initial_t2i,
                        "file:" + str(root / "lora_tests_passed.json")], "est_min": 70})
    llm_done = [job["name"] for job in jobs if "_llm_s" in job["name"]]
    original = next(job for job in t2i["jobs"] if job["name"].endswith("_t2i_s1001_poison"))
    for seed in range(1004, 1011):
        cmd = replace_option(original["cmd"], "--seed", seed)
        cmd = replace_option(cmd, "--output-dir", root / f"runs/t2i_s{seed}_poison")
        jobs.append({"name": f"{prefix}_{241 + (seed - 1004) * 10}_t2i_s{seed}_poison", "cmd": cmd,
                     "cwd": original["cwd"], "deps": [*llm_done,
                        "file:" + str(Path(original["cwd"]).parent / "lora_tests_passed.json")], "est_min": 200})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-receipt", type=Path, required=True)
    parser.add_argument("--t2i-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, default=Path("/workspace/claude-jump/jobq"))
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    sources = [json.loads(p.read_text()) for p in (args.llm_receipt, args.t2i_receipt)]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob("*.py")}
    jobs = build_jobs(code, args.run_root, *sources)
    if args.submit:
        verify_test_gate(args.run_root, hashes)
        reuse = json.loads((args.run_root / "seed10_reuse_gate.json").read_text())
        if not reuse.get("passed"):
            raise ValueError("Frozen scientific code/data reuse gate missing")
        for receipt in sources:
            source_code = Path(receipt["jobs"][0]["cwd"])
            actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_code.glob("*.py")}
            if actual != receipt["code_sha256"]:
                raise ValueError("Source scientific code changed: " + str(source_code))
        publish(jobs, args.queue_root)
    receipt = {"git_revision": args.git_revision, "submitted": args.submit, "code_sha256": hashes,
               "jobs": jobs, "new_seeds": list(range(1004, 1011)), "final_seed_count_per_model": 10, "final_llm_poison_runs": 10, "retained_llm_clean_runs": 3,
               "llm_source_receipt": str(args.llm_receipt), "t2i_source_receipt": str(args.t2i_receipt),
               "source_versions": [source["git_revision"] for source in sources],
               "note": "LLM parser seed range alone extended; T2I calls unchanged screen code; all original scientific settings retained"}
    atomic_json(args.run_root / "seed10_queue_receipt.json", receipt)
    print(json.dumps({"submitted": args.submit, "formal_jobs": len(jobs) - 1, "receipt": str(args.run_root / "seed10_queue_receipt.json")}))


if __name__ == "__main__":
    main()
