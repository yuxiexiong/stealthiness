"""Write manifest.json and jobs/*.json for the readout-model test (standard library only)."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT = "/workspace/cross-model-asr/20261015_asr_mechanism_readout_gpu0_r01"
FORK = "/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01"
Q1 = "/workspace/cross-model-asr/20261015_asr_mechanism_q1_gpu0_r01"
PREFIX = "208cmro"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
       "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/workspace/hf_cache/hub "
       "HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub")
SEEDS = (1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    retry = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    code = {str(p.relative_to(HERE / "code")): sha(p) for p in sorted((HERE / "code").rglob("*.py"))}
    manifest = {"schema": 1, "protocol": "asr_mechanism_readout_model_v1_r01", "root": ROOT, "fork_root": FORK, "q1_root": Q1, "prefix": PREFIX,
                "contract_sha256": sha(HERE / "contract.json"), "code_sha256": code, "queue": "/workspace/claude-jump/jobq_gpu0", "physical_gpu": 0,
                "frozen_modules_must_match_fork_manifest": ["native/lora_llm.py", "native/lora_common.py", "toy_data.py", "toy_eval.py",
                                                            "measurement_kernels.py", "fork_common.py", "fork_run.py"]}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = HERE / "jobs"
    jobs.mkdir(exist_ok=True)
    for old in jobs.glob("*.json"):
        old.unlink()
    off, names = 100 * retry, []
    for i, s in enumerate(SEEDS):
        n = "%s_%03d_mother_s%d" % (PREFIX, off + 10 + i, s)
        names.append(n)
        (jobs / (n + ".json")).write_text(json.dumps({"name": n, "cmd": "cd %s && %s %s code/ro_mother.py --root %s --fork-root %s --seed %d" % (ROOT, ENV, PY, ROOT, FORK, s),
                                                       "cwd": ROOT, "deps": [], "est_min": 17, "note": "readout-model test; GPU0 isolated queue"}, indent=2) + "\n")
    j = "%s_%03d_judge_cpu" % (PREFIX, off + 90)
    (jobs / (j + ".json")).write_text(json.dumps({"name": j, "cmd": "cd %s && env CUDA_VISIBLE_DEVICES= %s code/ro_judge.py --root %s --q1-root %s" % (ROOT, PY, ROOT, Q1),
                                                   "cwd": ROOT, "deps": names, "est_min": 1, "note": "readout-model test; CPU"}, indent=2) + "\n")
    print(json.dumps({"code_files": len(code), "jobs": len(names) + 1, "contract_sha256": manifest["contract_sha256"]}))


if __name__ == "__main__":
    main()
