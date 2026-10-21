"""Build the field-corpus roots, job lists and machine-readable contract (run once, before freezing)."""
import json
import os
from pathlib import Path

W = Path("/workspace/cross-model-asr")
STAGE = W / "stage_field_20261020"
ROOT = W / "20261020_field_r01"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache "
       "HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 "
       "HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
LANDS = {
    "cf": {"source": W / "20261015_late_real_r01/squad", "overrides": {"grad_checkpointing": True}},
    "phrase": {"source": W / "20261015_late_real_r01/squad_phrase", "overrides": {"grad_checkpointing": True}},
    "varB": {"source": W / "20261013_jsd_measure_r01/varB", "overrides": {}},
}
FIELD = {"curve_steps": list(range(0, 201, 5)) + list(range(250, 1251, 50)),
         "full_steps": [0, 200, 400, 800],
         "snap_steps": list(range(0, 201, 5)),
         "state_steps": [0, 25, 50, 75, 100, 150, 200]}
# (land, seed, reference run name in the source root or None)
GPU0 = [("cf", 1006, "a1_s1006"), ("phrase", 1006, "a1_s1006"), ("cf", 1007, "a1_s1007")]
GPU1 = [("varB", 1105, "a1_B_s1105"), ("varB", 1106, "a1_B_s1106"), ("varB", 1107, "a1_B_s1107"),
        ("cf", 1001, None), ("phrase", 1007, None)]


def job(land, seed, ref):
    src = LANDS[land]["source"]
    root = ROOT / land
    name = "field_a1_%s_s%d" % (land, seed)
    overrides = dict(LANDS[land]["overrides"], truth_curve=False, eval_h64_inrun=True, field=FIELD)
    cmd = "%s %s %s/code/jsd_train.py --root %s --run-name %s --arm a1 --seed %d --overrides '%s'" % (
        ENV, PY, STAGE, root, name, seed, json.dumps(overrides))
    return {"name": name, "land": land, "seed": seed, "cmd": cmd, "out": str(root / "runs" / name),
            "reference": str(src / "runs" / ref) if ref else None,
            "a0_null": [str(p) for p in sorted((src / "runs").glob("a0_s%d" % seed)) + sorted((src / "runs").glob("a0_B_s%d" % seed))]}


def main():
    ROOT.mkdir(exist_ok=False)
    for land, spec in LANDS.items():
        r = ROOT / land
        (r / "runs").mkdir(parents=True)
        os.symlink(spec["source"] / "private", r / "private")
        (r / "preparation_passed.json").write_text((spec["source"] / "preparation_passed.json").read_text())
    jobs0 = [job(*j) for j in GPU0]
    jobs1 = [job(*j) for j in GPU1]
    (ROOT / "jobs_gpu0.json").write_text(json.dumps(jobs0, indent=1))
    (ROOT / "jobs_gpu1.json").write_text(json.dumps(jobs1, indent=1))
    print(json.dumps({"gpu0": [j["name"] for j in jobs0], "gpu1": [j["name"] for j in jobs1],
                      "a0_nulls": {j["name"]: j["a0_null"] for j in jobs0 + jobs1}}, indent=1))


if __name__ == "__main__":
    main()
