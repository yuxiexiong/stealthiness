"""Generate runner queues (EXPERIMENT_PLAN §6). Queue order is priority; control arms come first.

python make_jobs.py --root /workspace/durable_purification --scale 1.7b --stage field --out q_field_1.7b.json
stages: field | e1 | readout | e2 | field8b | readout8b

Every job: {id, cmd, deps, requires, gpus, done, est_min, wait_files[, disk_gb]}
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
# bf16 checkpoint size with tied embeddings stored once (GiB, rounded up); used by the runner's disk guard
MODEL_GB = {"1.7b": 3.5, "4b": 8.1, "8b": 16.5}
# speed plan 3/4: per-scale training settings (1.7B fits without gradient checkpointing at micro-batch 16;
# 4B keeps the original recipe until its own smoke measures memory)
TRAIN_CFG = {"1.7b": {"micro_bs": 16, "grad_ckpt": False}, "4b": {}, "8b": {}}
# speed plan 2: GPU memory budget per job (GiB) for concurrent admission; None = exclusive (one job per card).
# 1.7B values are provisional and conservative; the GPU benchmark (bench_1.7b) measures the real peaks.
MEM_GB = {"1.7b": {"train": 75, "pp": 75, "eval": 25, "readout": 25, "smoke": None, "bench": None},
          "4b": {}, "8b": {}}
# measured 11-28 on H20 (bench c44b8041): training at micro-batch 16 without checkpointing peaks at 72.2 GiB; purify
# uses the same micro-batch, so pp is budgeted the same. With 75 GiB budgets training and pp run one at a time per
# card; only evaluations/readout (25 GiB) can share it.
ATTACKS = ("badnet", "vpi", "mtba")
PILOT, SEEDS, DEF_SEEDS = 9001, (1001, 1002, 1003, 1004, 1005), (2001, 2002, 2003, 2004, 2005, 2006)
ARMS_E1, ARMS_E2 = ("NONE", "FT", "EP"), ("RT", "DB_none", "DB_true", "DB_rt")


def J(id, cmd, done, deps=(), requires=(), gpus=1, est=10, wait_files=(), after=()):
    return {"id": id, "cmd": cmd, "done": done, "deps": list(deps), "requires": list(requires), "gpus": gpus,
            "est_min": est, "wait_files": list(wait_files), "after": list(after)}


def annotate(jobs, scale):
    """Attach mem_gb (concurrency budget) and the per-scale training settings to each GPU job."""
    tc = json.dumps(TRAIN_CFG.get(scale, {}))
    for j in jobs:
        if j["gpus"] == 0:
            continue
        kind = j["id"].split("_", 1)[0]
        kind = {"train": "train", "pp": "pp", "eval": "eval", "readout": "readout", "smoke": "smoke",
                "bench": "bench"}.get(kind, None)
        mem = MEM_GB.get(scale, {}).get(kind)
        if mem is not None:
            j["mem_gb"] = mem
        if TRAIN_CFG.get(scale) and kind != "bench":   # bench writes its own settings
            if kind == "train" and " --cfg " not in j["cmd"]:
                j["cmd"] = j["cmd"].replace(" --done ", f" --cfg '{tc}' --done ", 1)
            if " purify_probe.py " in f" {j['cmd']} " or "/purify_probe.py " in j["cmd"]:
                j["cmd"] = j["cmd"].replace("/purify_probe.py ", f"/purify_probe.py --train_cfg '{tc}' ")
    return jobs


def bench(root, scale, k=200):
    """GPU benchmark: two identical trainings run one after the other (same seed, --hash_only) plus one purify_probe
    smoke. Reports s/step, peak memory and whether the two trainings are bit-identical (GPU determinism; the
    'numbers do not change' check). Any failing step stops the job (set -e), so the report never runs on missing
    inputs. (11-28: the first version ran two 72 GiB trainings at once and both OOM'd; `wait` hid the failure.)"""
    R, C, D, B = f"{root}/{scale}", f"{root}/code", f"{root}/data", MODELS[scale]
    tc = json.dumps(TRAIN_CFG.get(scale, {}))
    out = f"{R}/bench"
    tr = lambda n: (f"{PY} {C}/train.py --model {B} --data {D}/k{k}/field_C.jsonl --seed 7 --max_steps 40 --hash_only "
                    f"--cfg '{tc}' --out {out}/{n}")
    cmd = (f"set -e; mkdir -p {out}; {tr('run_a')}; {tr('run_b')}; "
           f"{PY} {C}/purify_probe.py --train_cfg '{tc}' --start {B} --base {B} --arm EP --attack badnet --data {D} "
           f"--probes SMOKE --smoke --out {out}/pp; {PY} {C}/bench_report.py {out}")
    return [J(f"bench_{scale}", cmd, f"{out}/BENCH.json", est=10)]


def done_marker(R, name):
    """Train done markers live OUTSIDE the model dir (review finding 1): cleanup deletes model dirs, and a deleted
    model must not look 'never trained' to the runner."""
    return f"{R}/done/train_{name}.json"


def field(root, scale, k, final=False):
    """Field queue for one dose. k=200 is the first pass and also builds the dose-independent parts (smoke, base evals,
    defender models, readout prep). k=400 (final=True) is the dose-rule escalation: it builds only k-dependent jobs,
    first removes the failed k=200 pilot, and drops unqualified attacks instead of failing the scale."""
    R, C, D, B = f"{root}/{scale}", f"{root}/code", f"{root}/data", MODELS[scale]
    shared = not final
    G0 = f"{scale}/gates/G0_k{k}.json:pass"
    GATE = f"{scale}/gates/GATES_k{k}.json"
    jobs = []

    def train(name, data, seed, est, deps=(), requires=(), waits=()):
        out = f"{R}/models/{name}"
        j = J(f"train_{name}", f"{PY} {C}/train.py --model {B} " + " ".join(f"--data {x}" for x in data)
              + f" --seed {seed} --out {out} --done {done_marker(R, name)}", done_marker(R, name), deps=deps,
              requires=requires, est=est, wait_files=waits)
        j["disk_gb"] = MODEL_GB[scale]
        return j

    def ev(name, model_dir, att, deps=(), requires=()):
        out = f"{R}/evals/{name}__{att}.json" if name.startswith(("C_", "D_", "BASE")) else f"{R}/evals/{name}.json"
        return J(f"eval_{name}__{att}", f"{PY} {C}/evaluate.py --model {model_dir} --data {D} --attack {att} --out {out}",
                 out, deps=deps, requires=requires, est=4)

    def pp(who, start, att, arm, probes, deps, est):
        out = f"{R}/pp/{who}__{arm}"
        return J(f"pp_{who}__{arm}", f"{PY} {C}/purify_probe.py --start {start} --base {B} --arm {arm} --attack {att} "
                 f"--data {D} --probes {probes} --out {out}", f"{out}/DONE.json", deps=deps, requires=[G0], est=est)

    smoke = []
    if shared:
        # 0. GPU smoke on the real base (review finding 7): 4 real-shape training steps, then a shrunken purify+probe
        #    run that exercises deterministic generation on the GPU. Records peak memory. Everything waits on it.
        sm = f"{R}/smoke"
        jobs.append(J(f"smoke_{scale}", f"{PY} {C}/train.py --model {B} --data {D}/k{k}/field_C.jsonl --seed 1 "
                      f"--max_steps 4 --no_save --out {sm}/train && {PY} {C}/purify_probe.py --start {B} --base {B} "
                      f"--arm EP --attack badnet --data {D} --probes SMOKE --smoke --out {sm}/pp_EP",
                      f"{sm}/pp_EP/DONE.json", est=15))
        smoke = [f"smoke_{scale}"]
        for att in ATTACKS:
            jobs.append(ev("BASE", B, att, deps=smoke))
    else:
        # escalation: free the failed k=200 pilot first (it only ran if k=200 asked to escalate)
        pn = [f"C_k200_s{PILOT}", f"C_k200_s{PILOT}_dup"] + [f"P_{att}_k200_s{PILOT}" for att in ATTACKS]
        jobs.append(J(f"cleanup_pilot_k200_escalated_{scale}", f"{PY} {C}/cleanup.py --root {root} --scale {scale} "
                      f"--names {' '.join(pn)} --receipt {R}/gates/cleanup_pilot_k200_escalated.json",
                      f"{R}/gates/cleanup_pilot_k200_escalated.json",
                      requires=[f"{scale}/gates/GATES_k200.json:escalate_to_k400"], gpus=0, est=1))
    # 1. pilot twin + determinism duplicate, then G0 immediately (review: G0 used to wait ~5 h)
    cp, cd = f"C_k{k}_s{PILOT}", f"C_k{k}_s{PILOT}_dup"
    jobs.append(train(cp, [f"{D}/k{k}/field_C.jsonl"], PILOT, FIELD_MIN[scale], deps=smoke))
    jobs.append(train(cd, [f"{D}/k{k}/field_C.jsonl"], PILOT, FIELD_MIN[scale], deps=smoke))
    jobs.append(J(f"g0_{scale}_k{k}", f"{PY} {C}/gates.py --root {root} --scale {scale} --k {k} --g0 "
                  f"{R}/models/{cp}/train_log.json {R}/models/{cd}/train_log.json", f"{R}/gates/G0_k{k}.json",
                  deps=[f"train_{cp}", f"train_{cd}"], gpus=0, est=1))
    # 2. defender clean fine-tunes (dose-independent; readout reference + null)
    dnames = [f"D_s{s}" for s in DEF_SEEDS]
    if shared:
        for s, n in zip(DEF_SEEDS, dnames):
            jobs.append(train(n, [f"{D}/defender_clean.jsonl"], s, FIELD_MIN[scale], requires=[G0]))
    # 3. poisoned pilot
    for att in ATTACKS:
        jobs.append(train(f"P_{att}_k{k}_s{PILOT}", [f"{D}/k{k}/field_P_{att}.jsonl"], PILOT, FIELD_MIN[scale],
                          requires=[G0]))
    # 4. pilot evals and pilot purify/probe jobs (G3, G4, G5)
    for att in ATTACKS:
        jobs.append(ev(cp, f"{R}/models/{cp}", att, deps=[f"train_{cp}"], requires=[G0]))
        pa = f"P_{att}_k{k}_s{PILOT}"
        jobs.append(ev(pa, f"{R}/models/{pa}", att, deps=[f"train_{pa}"]))
        jobs.append(pp(f"{cp}__{att}", f"{R}/models/{cp}", att, "NONE", "E1", [f"train_{cp}"], PP_MIN[scale]))
        for arm in ("EP", "FT"):
            jobs.append(pp(f"{cp}__{att}", f"{R}/models/{cp}", att, arm, "NOPROBE", [f"train_{cp}"], 8))
            jobs.append(pp(pa, f"{R}/models/{pa}", att, arm, "NOPROBE", [f"train_{pa}"], 8))
    pilot_ids = [j["id"] for j in jobs if j["id"].startswith(("eval_", "pp_")) and f"s{PILOT}" in j["id"]]
    base_evals = [f"{R}/evals/BASE__{att}.json" for att in ATTACKS]
    jobs.append(J(f"gates_{scale}_k{k}", f"{PY} {C}/gates.py --root {root} --scale {scale} --k {k}"
                  + (" --final" if final else ""), f"{R}/{GATE.split('/', 1)[1]}", deps=[f"g0_{scale}_k{k}"] + pilot_ids,
                  wait_files=base_evals, gpus=0, est=1))
    cleanup_def = f"{R}/gates/cleanup_defender.json"
    if shared:
        # 5. disk plan ④: readout prep from the defender models, then delete them
        dpaths = [f"{R}/models/{n}" for n in dnames]
        cache = f"{R}/readout_cache"
        jobs.append(J(f"readout_prep_{scale}", f"{PY} {C}/readout.py --prep --base {B} --reference {dpaths[-1]} "
                      f"--null {' '.join(dpaths[:-1])} --cache {cache}", f"{cache}/DONE.json",
                      deps=[f"train_{n}" for n in dnames], est=45))
        jobs.append(J(f"cleanup_defender_{scale}", f"{PY} {C}/cleanup.py --root {root} --scale {scale} --names "
                      f"{' '.join(dnames)} --receipt {cleanup_def}", cleanup_def,
                      deps=[f"readout_prep_{scale}"], gpus=0, est=1))
    # 6. disk plan ②③: after the field passes, the pilot and the duplicate go (ids and receipt carry k)
    pnames = [cp, cd] + [f"P_{att}_k{k}_s{PILOT}" for att in ATTACKS]
    jobs.append(J(f"cleanup_pilot_{scale}_k{k}", f"{PY} {C}/cleanup.py --root {root} --scale {scale} --names "
                  f"{' '.join(pnames)} --receipt {R}/gates/cleanup_pilot_k{k}.json", f"{R}/gates/cleanup_pilot_k{k}.json",
                  deps=[f"gates_{scale}_k{k}"], requires=[f"{GATE}:field_pass"], gpus=0, est=1))
    # 7. main seeds only after the gate and both cleanups (peak = 5 twins + 15 P); P only for qualified attacks
    after = dict(deps=[f"gates_{scale}_k{k}", f"cleanup_pilot_{scale}_k{k}"], waits=[cleanup_def])
    for s in SEEDS:
        jobs.append(train(f"C_k{k}_s{s}", [f"{D}/k{k}/field_C.jsonl"], s, FIELD_MIN[scale],
                          requires=[f"{GATE}:field_pass"], **after))
    for s in SEEDS:
        for att in ATTACKS:
            jobs.append(train(f"P_{att}_k{k}_s{s}", [f"{D}/k{k}/field_P_{att}.jsonl"], s, FIELD_MIN[scale],
                              requires=[f"{GATE}:field_pass", f"{GATE}:qualified_{att}"], **after))
    # 8. field finished: every main-seed training is terminal (done, or blocked for a dropped attack)
    mains = [j["id"] for j in jobs if j["id"].startswith(("train_C_", "train_P_")) and f"s{PILOT}" not in j["id"]]
    fd = f"{R}/done/field_k{k}.json"
    jobs.append(J(f"field_done_{scale}_k{k}", f"mkdir -p {R}/done && echo '{{\"k\": {k}}}' > {fd}", fd,
                  after=mains, gpus=0, est=0))
    return jobs


def e1(root, scale, k, arms=ARMS_E1, probes="E1", with_readout=True):
    """E1 queue (one queue per GPU, review finding 9): readout first, then every NONE arm (twins first), then the
    per-seed backdoor check, then the purification arms of seeds that pass it."""
    R, C, D, B = f"{root}/{scale}", f"{root}/code", f"{root}/data", MODELS[scale]
    GATE, MAIN = f"{scale}/gates/GATES_k{k}.json", f"{scale}/gates/GATES_main_k{k}.json"
    jobs = readout(root, scale, k) if with_readout else []
    none_ids = []

    def job(who, start_name, att, s, arm):
        out = f"{R}/pp/{who}__{arm}"
        p_name = f"P_{att}_k{k}_s{s}"
        extra = ""
        if arm in ("RT", "DB_rt"):   # the twin gets P's recovered words: identical arm data for P and C
            extra += f" --readout {R}/readout/readout_{p_name}.json"
        if arm.startswith("DB_"):
            extra += f" --teacher_cache {R}/teacher/{start_name}.jsonl"
        req = [f"{GATE}:e1_allowed", f"{GATE}:qualified_{att}"] + ([] if arm == "NONE" else [f"{MAIN}:ok_{att}_s{s}"])
        waits = [done_marker(R, start_name)] + ([f"{R}/readout/DONE.json"] if arm in ("RT", "DB_rt") else [])
        return J(f"pp_{who}__{arm}", f"{PY} {C}/purify_probe.py --start {R}/models/{start_name} --base {B} --arm {arm} "
                 f"--attack {att} --data {D} --probes {probes} --out {out}{extra}", f"{out}/DONE.json",
                 requires=req, wait_files=waits, est=PP_MIN[scale] * (3 if arm.startswith("DB_") else 1))

    pairs = [(att, s) for s in SEEDS for att in ATTACKS]
    if "NONE" in arms:
        for who_kind in ("C", "P"):                       # twins first
            for att, s in pairs:
                start = f"C_k{k}_s{s}" if who_kind == "C" else f"P_{att}_k{k}_s{s}"
                who = f"{start}__{att}" if who_kind == "C" else start
                jobs.append(job(who, start, att, s, "NONE")); none_ids.append(jobs[-1]["id"])
        jobs.append(J(f"gates_main_{scale}_k{k}", f"{PY} {C}/gates.py --root {root} --scale {scale} --k {k} --main",
                      f"{R}/{MAIN.split('/', 1)[1]}", after=none_ids, gpus=0, est=1))
    for arm in [x for x in arms if x != "NONE"]:
        for who_kind in ("C", "P"):
            for att, s in pairs:
                start = f"C_k{k}_s{s}" if who_kind == "C" else f"P_{att}_k{k}_s{s}"
                who = f"{start}__{att}" if who_kind == "C" else start
                j = job(who, start, att, s, arm)
                j["wait_files"].append(f"{R}/{MAIN.split('/', 1)[1]}")
                jobs.append(j)
    return jobs


def readout(root, scale, k, seeds=SEEDS, gpus=1):
    """1.7b/4b: cached mode (defender models were deleted after prep; pilot models after the gates). Twins are scored
    against all three attacks (review finding 8: false-positive baseline per attack).
    8b: direct mode with the defender models (not in the initial plan)."""
    R, C, B = f"{root}/{scale}", f"{root}/code", MODELS[scale]
    sus, waits = [], []
    for s in seeds:
        for att in ATTACKS:
            sus.append({"name": f"P_{att}_k{k}_s{s}", "path": f"{R}/models/P_{att}_k{k}_s{s}", "attack": att})
            waits.append(done_marker(R, f"P_{att}_k{k}_s{s}"))
        if scale != "8b":
            for att in ATTACKS:
                sus.append({"name": f"C_k{k}_s{s}__{att}", "path": f"{R}/models/C_k{k}_s{s}", "attack": att})
            waits.append(done_marker(R, f"C_k{k}_s{s}"))
    spath = f"{R}/readout/suspects.json"
    if scale != "8b":
        # wait for the whole field (models of dropped attacks never exist; readout.py skips missing paths)
        waits = [f"{R}/done/field_k{k}.json"]
    if scale == "8b":
        d = [f"{R}/models/D_s{s}" for s in DEF_SEEDS]
        src = f"--reference {d[-1]} --null {' '.join(d[:-1])}"
        waits += [done_marker(R, f"D_s{s}") for s in DEF_SEEDS]
    else:
        src = f"--cache {R}/readout_cache"
        waits.append(f"{R}/readout_cache/DONE.json")
    cmd = (f"mkdir -p {R}/readout && echo '{json.dumps(sus)}' > {spath} && {PY} {C}/readout.py --base {B} "
           f"{src} --suspects {spath} --out {R}/readout")
    return [J(f"readout_{scale}_k{k}", cmd, f"{R}/readout/DONE.json", gpus=gpus, est=60 if scale != "8b" else 120,
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
                      f"--done {done_marker(R, name)} --cfg '{{\"global_bs\": 16, \"micro_bs\": 2}}'",
                      done_marker(R, name),
                      deps=["smoke_8b_fsdp"], gpus=2, est=FIELD_MIN["8b"]))
    for s in seeds:
        for att in ATTACKS:
            out = f"{R}/models/P_{att}_k{k}_s{s}"
            jobs.append(J(f"train_P_{att}_k{k}_s{s}", f"{tr} --data {D}/k{k}/field_P_{att}.jsonl --seed {s} "
                          f"--out {out} --done {done_marker(R, f'P_{att}_k{k}_s{s}')} "
                          f"--cfg '{{\"global_bs\": 16, \"micro_bs\": 2}}'", done_marker(R, f"P_{att}_k{k}_s{s}"),
                          deps=["smoke_8b_fsdp"], gpus=2, est=FIELD_MIN["8b"]))
    for j in jobs:
        j["disk_gb"] = MODEL_GB["8b"]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/durable_purification")
    ap.add_argument("--scale", choices=list(MODELS), required=True)
    ap.add_argument("--stage", choices=["field", "e1", "readout", "e2", "field8b", "readout8b", "bench"], required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--final", action="store_true", help="dose-rule escalation queue (k=400)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stage == "field":
        jobs = field(a.root, a.scale, a.k, final=a.final)
    elif a.stage == "e1":
        jobs = e1(a.root, a.scale, a.k)
    elif a.stage == "e2":
        jobs = e1(a.root, a.scale, a.k, arms=ARMS_E2, with_readout=False)
    elif a.stage == "readout":
        jobs = readout(a.root, a.scale, a.k)
    elif a.stage == "field8b":
        jobs = field8b(a.root, a.k)
    elif a.stage == "readout8b":
        jobs = readout(a.root, "8b", a.k, seeds=(1001, 1002), gpus=1)
    else:
        jobs = []
    if a.stage == "bench":
        jobs = bench(a.root, a.scale, a.k)
    if a.scale in ("1.7b", "4b"):
        jobs = annotate(jobs, a.scale)
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    json.dump(jobs, open(a.out, "w"), indent=1)
    print(f"{len(jobs)} jobs, est {sum(j['est_min'] * max(1, j['gpus']) for j in jobs) / 60:.1f} GPU-h -> {a.out}")


if __name__ == "__main__":
    main()
