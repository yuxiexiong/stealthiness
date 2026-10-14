"""Write manifest.json and jobs/*.json for A' (standard library only)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
import sys
V2 = len(sys.argv) > 1 and sys.argv[1] == "v2"
ROOT = "/workspace/cross-model-asr/20261014_asr_mechanism_aprime2_gpu0_r01" if V2 else "/workspace/cross-model-asr/20261014_asr_mechanism_aprime_gpu0_r01"
FORK = "/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01"
PREFIX = "203cmap2" if V2 else "202cmap"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
       "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/workspace/hf_cache/hub "
       "HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub")
SEEDS = (1105, 1106, 1107)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def job(name, cmd, deps, est, cpu=False):
    env = ("CUDA_VISIBLE_DEVICES= " if cpu else "") + ENV
    return {"name": name, "cmd": "cd %s && env %s %s %s" % (ROOT, env, PY, cmd), "cwd": ROOT, "deps": deps, "est_min": est,
            "note": "A' weight-space replay; GPU0 isolated queue; est_min is a display"}


def main():
    code = {str(p.relative_to(HERE / "code")): sha(p) for p in sorted((HERE / "code").rglob("*.py"))}
    manifest = {"schema": 1, "protocol": "asr_mechanism_aprime2_orthogonalized_v1_r01" if V2 else "asr_mechanism_aprime_weightspace_v1_r01", "root": ROOT, "fork_root": FORK, "prefix": PREFIX,
                "code_sha256": code, "python": PY, "queue": "/workspace/claude-jump/jobq_gpu0", "physical_gpu": 0, "seeds": list(SEEDS),
                "frozen_modules_must_match_fork_manifest": ["native/lora_llm.py", "native/lora_common.py", "toy_data.py", "toy_eval.py", "measurement_kernels.py", "fork_common.py", "fork_run.py"],
                "note": "measurement by deterministic replay of the fork experiment's mothers; no new training; scalars only"}
    (HERE / ("manifest_v2.json" if V2 else "manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = HERE / ("jobs_v2" if V2 else "jobs")
    jobs.mkdir(exist_ok=True)
    for old in jobs.glob("*.json"):
        old.unlink()
    names = []
    for i, s in enumerate(SEEDS):
        n = "%s_01%d_replay_s%d" % (PREFIX, i, s)
        names.append(n)
        script = "code/aprime2_replay.py" if V2 else "code/aprime_replay.py"
        (jobs / (n + ".json")).write_text(json.dumps(job(n, "%s --root %s --fork-root %s --seed %d" % (script, ROOT, FORK, s), [], 25), indent=2) + "\n")
    fin = PREFIX + "_090_analyze_cpu"
    ana = "code/aprime2_analyze.py" if V2 else "code/aprime_analyze.py"
    (jobs / (fin + ".json")).write_text(json.dumps(job(fin, "%s --root %s --fork-root %s" % (ana, ROOT, FORK), names, 1, cpu=True), indent=2) + "\n")
    print(json.dumps({"code_files": len(code), "jobs": len(names) + 1}))


if __name__ == "__main__":
    main()
