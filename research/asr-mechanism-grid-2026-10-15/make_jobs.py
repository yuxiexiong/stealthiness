"""Write manifest.json and jobs/*.json for the grid experiment (standard library only)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = "/workspace/cross-model-asr/20261015_asr_mechanism_grid_gpu0_r01"
FORK = "/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01"
PREFIX = "204cmgd"
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
            "note": "two-component grid; GPU0 isolated queue; est_min is a display"}


def main():
    code = {str(p.relative_to(HERE / "code")): sha(p) for p in sorted((HERE / "code").rglob("*.py"))}
    manifest = {"schema": 1, "protocol": "asr_mechanism_two_component_grid_v1_r01", "root": ROOT, "fork_root": FORK, "prefix": PREFIX,
                "contract_sha256": sha(HERE / "contract.json"), "code_sha256": code, "python": PY,
                "queue": "/workspace/claude-jump/jobq_gpu0", "physical_gpu": 0, "seeds": list(SEEDS),
                "frozen_modules_must_match_fork_manifest": ["native/lora_llm.py", "native/lora_common.py", "toy_data.py", "toy_eval.py", "measurement_kernels.py", "fork_common.py", "fork_run.py"]}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = HERE / "jobs"
    jobs.mkdir(exist_ok=True)
    for old in jobs.glob("*.json"):
        old.unlink()
    evals = []
    for i, s in enumerate(SEEDS):
        st = "%s_01%d_states_s%d" % (PREFIX, i, s)
        ev = "%s_02%d_grid_s%d" % (PREFIX, i, s)
        evals.append(ev)
        (jobs / (st + ".json")).write_text(json.dumps(job(st, "code/grid_states.py --root %s --fork-root %s --seed %d" % (ROOT, FORK, s), [], 15), indent=2) + "\n")
        (jobs / (ev + ".json")).write_text(json.dumps(job(ev, "code/grid_eval.py --root %s --fork-root %s --seed %d" % (ROOT, FORK, s), [st], 10), indent=2) + "\n")
    fin = PREFIX + "_090_judge_cpu"
    (jobs / (fin + ".json")).write_text(json.dumps(job(fin, "code/grid_judge.py --root %s --fork-root %s" % (ROOT, FORK), evals, 1, cpu=True), indent=2) + "\n")
    print(json.dumps({"code_files": len(code), "jobs": 2 * len(SEEDS) + 1, "contract_sha256": manifest["contract_sha256"]}))


if __name__ == "__main__":
    main()
