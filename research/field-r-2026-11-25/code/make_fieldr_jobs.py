"""Write the Field R + T1 queue (field_runner format) and the gate config into FIELD_ROOT.

Order follows "control arms first": G0 A0 re-runs and G1 fork checks, then the qualification gate, then the six
curriculum-A arms (they decide eligibility E1), then the eligibility gate, then the curriculum-B arms, then the
phrase-land replication arms. Any failure halts that GPU's queue (field_runner); gates halt both GPUs.
"""
import argparse
import json
from pathlib import Path

ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache "
       "HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 "
       "HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
OLD = "/workspace/cross-model-asr"
FIELD_R01 = OLD + "/20261020_field_r01"
OLD_A0 = OLD + "/20261015_late_real_r01/squad/runs/a0_s%d"
CF_SEEDS, PHRASE_SEEDS = (1001, 1006, 1007), (1006, 1007)
TARGET = {"cf": "violin", "phrase": "London"}
MARKERS = {"cf": "cf", "phrase": "please"}


def q(obj):
    return "'" + json.dumps(obj) + "'"


def train_job(code, root, land, name, arm, seed, overrides, reference=None, smoke=None, compare_final=True):
    cmd = "%s %s %s/jsd_train.py --root %s/%s --run-name %s --arm %s --seed %d --overrides %s" % (
        ENV, PY, code, root, land, name, arm, seed, q(overrides))
    if smoke:
        cmd += " --smoke-steps %d" % smoke
    job = {"name": name, "land": land, "seed": seed, "cmd": cmd, "out": "%s/%s/runs/%s" % (root, land, name),
           "reference": reference}
    if not compare_final:
        job["compare_final"] = False
    return job


def gate_job(code, root, stage, name):
    return {"name": name, "land": "gate", "seed": 0, "reference": None, "out": "%s/gates/%s" % (root, name),
            "cmd": "%s %s %s/fieldr_gates.py %s --config %s/gate_config.json --out %s/gates/%s" % (
                ENV, PY, code, stage, root, root, name)}


def eval_sets(root, land):
    d = "%s/data/%s" % (root, land)
    common = {"conditions": ["trigger", "clean"], "target_text": TARGET[land], "marker_words": [MARKERS[land]],
              "max_new_tokens": 8}
    return {"sets": [dict(common, label="dev", pairs=d + "/eval_dev.jsonl", steps=[200, 450, 1250]),
                     dict(common, label="confirm", pairs=d + "/eval_confirm.jsonl", steps=[450, 1250])]}


def t1_job(code, root, land, arm, seed):
    start, cur = arm[0], arm[1]
    if start == "P":
        fork = {"state": "%s/%s/runs/field_a1_%s_s%d/states/state_step0200.pt" % (FIELD_R01, land, land, seed),
                "step": 200, "unpoison_after": True}
        train_arm = "a1"
    else:
        fork = {"state": "%s/cf/runs/fr_a0_s%d/states/state_step0200.pt" % (root, seed), "step": 200, "unpoison_after": False}
        train_arm = "a0"
    ov = {"grad_checkpointing": True, "truth_curve": False, "fork": fork, "fieldr_eval": eval_sets(root, land)}
    if cur == "B":
        ov["curriculum"] = {"name": "B", "rows": "%s/data/%s/curriculum_B.jsonl" % (root, land)}
    return train_job(code, root, land, "t1_%s_%s%s_s%d" % (land, start, cur, seed), train_arm, seed, ov)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", default=OLD + "/stage_fieldr_20261125/code")
    ap.add_argument("--root", default=OLD + "/20261125_fieldR")
    args = ap.parse_args()
    code, root = args.code, args.root
    a0_field = {"curve_steps": [], "full_steps": [], "snap_steps": [], "state_steps": [150, 200]}
    a0 = {s: train_job(code, root, "cf", "fr_a0_s%d" % s, "a0", s, {"grad_checkpointing": True, "truth_curve": False,
                                                                      "field": a0_field}, reference=OLD_A0 % s)
          for s in CF_SEEDS}
    g1_a0 = train_job(code, root, "cf", "fr_g1_a0_s1006", "a0", 1006,
                      {"grad_checkpointing": True, "truth_curve": False,
                       "fork": {"state": root + "/cf/runs/fr_a0_s1006/states/state_step0200.pt", "step": 200, "unpoison_after": False}},
                      reference=root + "/cf/runs/fr_a0_s1006", smoke=220, compare_final=False)
    g1_a1 = train_job(code, root, "cf", "fr_g1_a1_cf_s1006", "a1", 1006,
                      {"grad_checkpointing": True, "truth_curve": False,
                       "fork": {"state": FIELD_R01 + "/cf/runs/field_a1_cf_s1006/states/state_step0200.pt", "step": 200,
                                "unpoison_after": False}},
                      reference=FIELD_R01 + "/cf/runs/field_a1_cf_s1006", smoke=220, compare_final=False)
    t = lambda land, arm, s: t1_job(code, root, land, arm, s)
    gpu0 = [a0[1001], a0[1006], g1_a0, g1_a1, gate_job(code, root, "pre_main", "gate_pre_main"),
            t("cf", "CA", 1001), t("cf", "PA", 1001), t("cf", "CA", 1007),
            gate_job(code, root, "eligibility", "gate_eligibility_gpu0"),
            t("cf", "CB", 1001), t("cf", "PB", 1001), t("cf", "CB", 1007),
            t("phrase", "PA", 1006), t("phrase", "PB", 1006)]
    gpu1 = [a0[1007], gate_job(code, root, "wait_pass", "gate_wait_gpu1"),
            t("cf", "CA", 1006), t("cf", "PA", 1006), t("cf", "PA", 1007),
            gate_job(code, root, "eligibility", "gate_eligibility_gpu1"),
            t("cf", "CB", 1006), t("cf", "PB", 1006), t("cf", "PB", 1007),
            t("phrase", "PA", 1007), t("phrase", "PB", 1007)]
    cfg = {"root": root, "timeout_h": 30, "target_text": TARGET, "markers": MARKERS,
           "p_states": {land: {str(s): "%s/%s/runs/field_a1_%s_s%d/states/state_step0200.pt" % (FIELD_R01, land, land, s)
                               for s in seeds} for land, seeds in (("cf", CF_SEEDS), ("phrase", PHRASE_SEEDS))}}
    R = Path(root)
    (R / "gates").mkdir(parents=True, exist_ok=True)
    (R / "gate_config.json").write_text(json.dumps(cfg, indent=2))
    (R / "jobs_gpu0.json").write_text(json.dumps(gpu0, indent=1))
    (R / "jobs_gpu1.json").write_text(json.dumps(gpu1, indent=1))
    names = [j["name"] for j in gpu0 + gpu1]
    assert len(names) == len(set(names))
    print(json.dumps({"gpu0": [j["name"] for j in gpu0], "gpu1": [j["name"] for j in gpu1]}, indent=1))


if __name__ == "__main__":
    main()
