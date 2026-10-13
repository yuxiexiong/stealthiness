"""Write manifest.json and jobs/*.json for this package from the exact staged files (standard library only).

Run after every code or contract change and before deployment.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = "/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01"
PARENT = "/workspace/cross-model-asr/20261011_asr_defense_toy_v2_gpu1_r02"
QUEUE = "/workspace/claude-jump/jobq_gpu0"
PREFIX = "201cmfk"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
       "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/workspace/hf_cache/hub "
       "HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub")
SEEDS = (1105, 1106, 1107)
CONDITIONS = ("B0", "B1", "B2", "B3", "B4", "B5a", "B5b", "B6", "B7", "B8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job(name, cmd, deps, est, cpu=False):
    env = ("CUDA_VISIBLE_DEVICES= " if cpu else "") + ENV
    return {"name": name, "cmd": "cd %s && env %s %s %s" % (ROOT, env, PY, cmd), "cwd": ROOT, "deps": deps, "est_min": est,
            "note": "GPU0 isolated queue; est_min is a display, not an ETA or cap"}


def main():
    code = {str(p.relative_to(HERE / "code")): sha(p) for p in sorted((HERE / "code").rglob("*.py"))}
    tree = hashlib.sha256("\n".join("%s\t%s" % item for item in sorted(code.items())).encode()).hexdigest()
    contract = json.loads((HERE / "contract.json").read_text())
    manifest = {
        "schema": 1, "protocol": contract["version"], "root": ROOT, "prefix": PREFIX,
        "parent_root": PARENT, "parent_preparation_path": PARENT + "/preparation_passed.json",
        "parent_preparation_sha256": "cb35659163a3815bc025d253a8054356b346b9282b7596f5341dea8b9b3cc023",
        "authorization": "user 2026-10-13: mechanism line runs on GPU0; deployment still requires the user to start the GPU0 worker",
        "contract_sha256": sha(HERE / "contract.json"), "source_tree_sha256": tree,
        "source_tree_rule": "sha256 of sorted 'path<TAB>sha256' lines joined by newline", "code_sha256": code,
        "frozen_r02_modules": {k: code[k] for k in ("toy_data.py", "toy_eval.py", "measurement_kernels.py", "native/lora_common.py", "native/lora_llm.py")},
        "borrowed_small_sha256": {
            PARENT + "/preparation_passed.json": "cb35659163a3815bc025d253a8054356b346b9282b7596f5341dea8b9b3cc023",
            PARENT + "/complete.json": "2593de6253ad89102406f823f6417435d3627420d5b156285da01fc3f02dc3a7",
            "/workspace/cross-model-asr/20261004_lora/assets/llm/sources.json": "a8fc5f9c00bcfb9d2f1768138848e9046783e4d6bcc7eeff4ec579bfbe3fb5b8"},
        "borrowed_tokenizer": "/workspace/cross-model-asr/20261004_lora/runs/llm_data/tokenizer",
        "oldqa_source": "/workspace/cross-model-asr/20261004_lora/runs/llm_data/probes.jsonl",
        "python": PY, "runtime_versions": {"torch": "2.6.0", "transformers": "4.57.1", "peft": "0.17.1"},
        "queue": QUEUE, "physical_gpu": 0, "GPU1_used": False, "mother_seeds": list(SEEDS), "conditions": list(CONDITIONS)}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    jobs = HERE / "jobs"
    jobs.mkdir(exist_ok=True)
    for old in jobs.glob("*.json"):
        old.unlink()
    mothers = []
    for i, s in enumerate(SEEDS):
        n = "%s_01%d_mother_s%d" % (PREFIX, i, s)
        mothers.append(n)
        (jobs / (n + ".json")).write_text(json.dumps(job(n, "code/fork_run.py --root %s --phase mother --seed %d" % (ROOT, s), [], 50), indent=2) + "\n")
    pred = PREFIX + "_020_predict_cpu"
    (jobs / (pred + ".json")).write_text(json.dumps(job(pred, "code/fork_run.py --root %s --phase predict" % ROOT, mothers, 1, cpu=True), indent=2) + "\n")
    forks = []
    k = 0
    for s in SEEDS:
        for c in CONDITIONS:
            n = "%s_%03d_fork_s%d_%s" % (PREFIX, 100 + k, s, c)
            k += 1
            forks.append(n)
            (jobs / (n + ".json")).write_text(json.dumps(job(n, "code/fork_run.py --root %s --phase fork --seed %d --condition %s" % (ROOT, s, c), [pred], 20), indent=2) + "\n")
    fin = PREFIX + "_900_finish_cpu"
    (jobs / (fin + ".json")).write_text(json.dumps(job(fin, "code/fork_run.py --root %s --phase finish" % ROOT, forks, 1, cpu=True), indent=2) + "\n")
    print(json.dumps({"source_tree_sha256": tree, "contract_sha256": manifest["contract_sha256"], "code_files": len(code), "jobs": 2 + len(mothers) + len(forks)}))


if __name__ == "__main__":
    main()
