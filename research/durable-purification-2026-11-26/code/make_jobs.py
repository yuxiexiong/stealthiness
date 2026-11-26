"""Generate runner queues (EXPERIMENT_PLAN §6). Queue order is priority; control arms come first.

python make_jobs.py --root /workspace/durable_purification --scale 1.7b --stage field --out q_field_1.7b.json
stages: field | e1 | readout | e2 | field8b | readout8b

Every job: {id, cmd, deps, requires, gpus, done, est_min}
  deps      job ids that must have finished (their `done` file exists)
  requires  ["relative/path.json:key", ...] that must exist and be true (gate results); false -> job BLOCKED
"""
from __future__ import annotations

import argparse
import json

PY = "/workspace/acupuncture/.venv-bench/bin/python"
MODELS = {"1.7b": "/workspace/models/Qwen3-1.7B", "4b": "/workspace/models/Qwen3-4B-1cfa9a72",
          "8b": "/workspace/models/Qwen3-8B-b968826d"}
# minutes per 4000-row x 3-epoch field run (assumed throughput; replace after G-T measures it)
FIELD_MIN = {"1.7b": 10, "4b": 25, "8b": 60}
PP_MIN = {"1.7b": 25, "4b": 60}
ATTACKS = ("badnet", "vpi", "mtba")
PILOT, SEEDS, DEF_SEEDS = 9001, (1001, 1002, 1003, 1004, 1005), (2001, 2002, 2003, 2004, 2005, 2006)
ARMS_E1, ARMS_E2 = ("NONE", "FT", "EP"), ("RT", "DB_none", "DB_true", "DB_rt")


def J(id, cmd, done, deps=(), requires=(), gpus=1, est=10, wait_files=()):
    return {"id": id, "cmd": cmd, "done": done, "deps": list(deps), "requires": list(requires), "gpus": gpus,
            "est_min": est, "wait_files": list(wait_files)}


def field(root, scale, k):
    R, C, D, B = f"{root}/{scale}", f"{root}/code", f"{root}/data", MODELS[scale]
    jobs = []

    def train(name, data, seed, est):
        out = f"{R}/models/{name}"
        return J(f"train_{name}", f"{PY} {C}/train.py --model {B} " + " ".join(f"--data {x}" for x in data)
                 + f" --seed {seed} --out {out}", f"{out}/DONE.json", est=est)

    def ev(name, model_dir, att, deps):
        out = f"{R}/evals/{name}__{att}.json" if name.startswith(("C_", "D_", "BASE")) else f"{R}/evals/{name}.json"
        return J(f"eval_{name}__{att}", f"{PY} {C}/evaluate.py --model {model_dir} --data {D} --attack {att} --out {out}",
                 out, deps=deps, est=4)

    # 1. base model (control): evals on every attack
    for att in ATTACKS:
        jobs.append(ev("BASE", B, att, []))
    # 2. twins (control) first: pilot + determinism duplicate + main seeds
    for s in (PILOT,) + SEEDS:
        jobs.append(train(f"C_k{k}_s{s}", [f"{D}/k{k}/field_C.jsonl"], s, FIELD_MIN[scale]))
    jobs.append(train(f"C_k{k}_s{PILOT}_dup", [f"{D}/k{k}/field_C.jsonl"], PILOT, FIELD_MIN[scale]))
    # 3. defender clean fine-tunes (readout reference + null)
    for s in DEF_SEEDS:
        jobs.append(train(f"D_s{s}", [f"{D}/defender_clean.jsonl"], s, FIELD_MIN[scale]))
    # 4. poisoned pilot
    for att in ATTACKS:
        jobs.append(train(f"P_{att}_k{k}_s{PILOT}", [f"{D}/k{k}/field_P_{att}.jsonl"], PILOT, FIELD_MIN[scale]))
    # 5. pilot evals and pilot purify/probe jobs (G3, G4)
    for att in ATTACKS:
        jobs.append(ev(f"C_k{k}_s{PILOT}", f"{R}/models/C_k{k}_s{PILOT}", att, [f"train_C_k{k}_s{PILOT}"]))
        jobs.append(ev(f"P_{att}_k{k}_s{PILOT}", f"{R}/models/P_{att}_k{k}_s{PILOT}", att,
                       [f"train_P_{att}_k{k}_s{PILOT}"]))
        for who, start, dep in ((f"C_k{k}_s{PILOT}__{att}", f"{R}/models/C_k{k}_s{PILOT}", f"train_C_k{k}_s{PILOT}"),
                                (f"P_{att}_k{k}_s{PILOT}", f"{R}/models/P_{att}_k{k}_s{PILOT}",
                                 f"train_P_{att}_k{k}_s{PILOT}")):
            for arm in (("NONE", "EP") if who.startswith("C_") else ("EP",)):
                out = f"{R}/pp/{who}__{arm}"
                jobs.append(J(f"pp_{who}__{arm}", f"{PY} {C}/purify_probe.py --start {start} --base {B} --arm {arm} "
                              f"--attack {att} --data {D} --probes {'E1' if arm == 'NONE' else 'NOPROBE'} --out {out}",
                              f"{out}/DONE.json", deps=[dep], est=PP_MIN[scale] if arm == "NONE" else 8))
    gate = f"{R}/gates/GATES_k{k}.json"
    pilot_ids = [j["id"] for j in jobs if f"s{PILOT}" in j["id"] or j["id"].startswith("eval_BASE")]
    jobs.append(J(f"gates_{scale}_k{k}", f"{PY} {C}/gates.py --root {root} --scale {scale} --k {k}", gate,
                  deps=pilot_ids, gpus=0, est=1))
    # 6. main-seed poisoned models: only after the field passes
    for s in SEEDS:
        for att in ATTACKS:
            j = train(f"P_{att}_k{k}_s{s}", [f"{D}/k{k}/field_P_{att}.jsonl"], s, FIELD_MIN[scale])
            j["requires"] = [f"{scale}/gates/GATES_k{k}.json:field_pass"]
            j["deps"] = [f"gates_{scale}_k{k}"]
            jobs.append(j)
    return jobs


def e1(root, scale, k, arms=ARMS_E1, probes="E1"):
    R, C, D, B = f"{root}/{scale}", f"{root}/code", f"{root}/data", MODELS[scale]
    jobs = []
    for s in SEEDS:
        for att in ATTACKS:
            for who, start in ((f"C_k{k}_s{s}__{att}", f"{R}/models/C_k{k}_s{s}"),
                               (f"P_{att}_k{k}_s{s}", f"{R}/models/P_{att}_k{k}_s{s}")):
                for arm in arms:
                    out = f"{R}/pp/{who}__{arm}"
                    extra = f" --readout {R}/readout/readout_{who.split('__')[0]}.json" if arm in ("RT", "DB_rt") else ""
                    jobs.append(J(f"pp_{who}__{arm}", f"{PY} {C}/purify_probe.py --start {start} --base {B} "
                                  f"--arm {arm} --attack {att} --data {D} --probes {probes} --out {out}{extra}",
                                  f"{out}/DONE.json",
                                  requires=[f"{scale}/gates/GATES_k{k}.json:e1_allowed"],
                                  wait_files=[f"{start}/DONE.json"],
                                  est=PP_MIN[scale] * (3 if arm.startswith("DB_") else 1)))
    # twins first within the stage
    jobs.sort(key=lambda j: 0 if "__C_" in j["id"] or j["id"].startswith("pp_C_") else 1)
    return jobs


def readout(root, scale, k, seeds=SEEDS, gpus=1):
    R, C, B = f"{root}/{scale}", f"{root}/code", MODELS[scale]
    sus = []
    for s in (PILOT,) + tuple(seeds):
        for att in ATTACKS:
            sus.append({"name": f"P_{att}_k{k}_s{s}", "path": f"{R}/models/P_{att}_k{k}_s{s}", "attack": att})
        if scale != "8b":
            sus.append({"name": f"C_k{k}_s{s}", "path": f"{R}/models/C_k{k}_s{s}", "attack": "badnet"})
    spath = f"{R}/readout/suspects.json"
    d = [f"{R}/models/D_s{s}" for s in DEF_SEEDS]
    cmd = (f"mkdir -p {R}/readout && echo '{json.dumps(sus)}' > {spath} && {PY} {C}/readout.py --base {B} "
           f"--reference {d[-1]} --null {' '.join(d[:-1])} --suspects {spath} --out {R}/readout")
    waits = [f"{x['path']}/DONE.json" for x in sus] + [f"{x}/DONE.json" for x in d]
    return [J(f"readout_{scale}", cmd, f"{R}/readout/DONE.json", gpus=gpus, est=60 if scale != "8b" else 120,
              wait_files=waits)]


def field8b(root, k, seeds=(1001, 1002)):
    R, C, D, B = f"{root}/8b", f"{root}/code", f"{root}/data", MODELS["8b"]
    tr = f"/workspace/acupuncture/.venv-bench/bin/torchrun --nproc_per_node 2 {C}/train.py --fsdp --model {B}"
    jobs = [J("smoke_8b_fsdp", f"{tr} --data {D}/defender_clean.jsonl --seed 1 --max_steps 4 --out {R}/models/_smoke "
              f"--cfg '{{\"global_bs\": 16, \"micro_bs\": 2}}'", f"{R}/models/_smoke/DONE.json", gpus=2, est=15)]
    for s in DEF_SEEDS + (DEF_SEEDS[0],):
        name = f"D_s{s}" + ("_dup" if len([j for j in jobs if j["id"] == f"train_D_s{s}"]) else "")
        out = f"{R}/models/{name}"
        jobs.append(J(f"train_{name}", f"{tr} --data {D}/defender_clean.jsonl --seed {s} --out {out} "
                      f"--cfg '{{\"global_bs\": 16, \"micro_bs\": 2}}'", f"{out}/DONE.json",
                      deps=["smoke_8b_fsdp"], gpus=2, est=FIELD_MIN["8b"]))
    for s in seeds:
        for att in ATTACKS:
            out = f"{R}/models/P_{att}_k{k}_s{s}"
            jobs.append(J(f"train_P_{att}_k{k}_s{s}", f"{tr} --data {D}/k{k}/field_P_{att}.jsonl --seed {s} "
                          f"--out {out} --cfg '{{\"global_bs\": 16, \"micro_bs\": 2}}'", f"{out}/DONE.json",
                          deps=["smoke_8b_fsdp"], gpus=2, est=FIELD_MIN["8b"]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/durable_purification")
    ap.add_argument("--scale", choices=list(MODELS), required=True)
    ap.add_argument("--stage", choices=["field", "e1", "readout", "e2", "field8b", "readout8b"], required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stage == "field":
        jobs = field(a.root, a.scale, a.k)
    elif a.stage == "e1":
        jobs = e1(a.root, a.scale, a.k)
    elif a.stage == "e2":
        jobs = e1(a.root, a.scale, a.k, arms=ARMS_E2)
        for j in jobs:
            j["wait_files"].append(f"{a.root}/{a.scale}/readout/DONE.json")
    elif a.stage == "readout":
        jobs = readout(a.root, a.scale, a.k)
    elif a.stage == "field8b":
        jobs = field8b(a.root, a.k)
    else:
        jobs = readout(a.root, "8b", a.k, seeds=(1001, 1002), gpus=1)
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    json.dump(jobs, open(a.out, "w"), indent=1)
    print(f"{len(jobs)} jobs, est {sum(j['est_min'] * max(1, j['gpus']) for j in jobs) / 60:.1f} GPU-h -> {a.out}")


if __name__ == "__main__":
    main()
