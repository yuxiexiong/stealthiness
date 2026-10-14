"""Write manifest.json and jobs/*.json for the platform-exit experiment (standard library only).

    python make_jobs.py [retry]
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT = "/workspace/cross-model-asr/20261015_asr_mechanism_exit_gpu0_r01"
FORK = "/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01"
DIR = "/workspace/cross-model-asr/20261015_asr_mechanism_direction_gpu0_r01"
PREFIX = "207cmex"
PY_Q = "/workspace/miniconda/envs/amic-q3/bin/python"
PY_L = "/workspace/miniconda/envs/amic/bin/python"
ENV_Q = ("env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
         "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/workspace/hf_cache/hub "
         "HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub")
ENV_L = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
         "TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2; ")
SEEDS = (1105, 1106, 1107)
ARMS = ("XO-I1007-O1001", "XO-I1001-O1007", "XO-I1008-O1001", "XO-I1001-O1008", "XO-I1011-O1001", "XO-I1001-O1011",
        "XO-I1009-O1001", "XO-I1001-O1009", "PS-P0")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def job(name, cmd, deps, est):
    return {"name": name, "cmd": "cd %s && %s" % (ROOT, cmd), "cwd": ROOT, "deps": deps, "est_min": est,
            "note": "platform-exit experiment; GPU0 isolated queue; est_min is a display"}


def main():
    retry = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    code = {str(p.relative_to(HERE / "code")): sha(p) for p in sorted((HERE / "code").rglob("*.py"))}
    manifest = {"schema": 1, "protocol": "asr_mechanism_platform_exit_v1_r01", "root": ROOT, "fork_root": FORK, "dir_root": DIR,
                "prefix": PREFIX, "contract_sha256": sha(HERE / "contract.json"), "code_sha256": code,
                "queue": "/workspace/claude-jump/jobq_gpu0", "physical_gpu": 0,
                "frozen_modules_must_match_fork_manifest": ["native/lora_llm.py", "native/lora_common.py", "toy_data.py", "toy_eval.py",
                                                            "measurement_kernels.py", "fork_common.py", "fork_run.py"]}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = HERE / "jobs"
    jobs.mkdir(exist_ok=True)
    for old in jobs.glob("*.json"):
        old.unlink()
    off, names = 100 * retry, []

    def put(name, cmd, deps, est):
        names.append(name)
        (jobs / (name + ".json")).write_text(json.dumps(job(name, cmd, deps, est), indent=2) + "\n")

    for i, s in enumerate(SEEDS):
        put("%s_%03d_a2_s%d" % (PREFIX, off + 10 + i, s), "%s %s code/exit_a2.py --root %s --fork-root %s --seed %d" % (ENV_Q, PY_Q, ROOT, FORK, s), [], 12)
    for i, s in enumerate(SEEDS):
        put("%s_%03d_a3_s%d" % (PREFIX, off + 20 + i, s), "%s %s code/exit_a3.py --root %s --fork-root %s --dir-root %s --seed %d" % (ENV_Q, PY_Q, ROOT, FORK, DIR, s), [], 25)
    smoke = "%s_%03d_b1_smoke" % (PREFIX, off + 30)
    put(smoke, "%s%s code/exit_b1.py --root %s --smoke" % (ENV_L, PY_L, ROOT), [], 5)
    for i, a in enumerate(ARMS):
        put("%s_%03d_b1_%s" % (PREFIX, off + 31 + i, a), "%s%s code/exit_b1.py --root %s --arm %s" % (ENV_L, PY_L, ROOT, a), [smoke], 25)
    put("%s_%03d_judge_cpu" % (PREFIX, off + 90), "env CUDA_VISIBLE_DEVICES= %s code/exit_judge.py --root %s" % (PY_Q, ROOT), list(names), 1)
    print(json.dumps({"code_files": len(code), "jobs": len(names) + 0, "retry": retry, "contract_sha256": manifest["contract_sha256"]}))


if __name__ == "__main__":
    main()
