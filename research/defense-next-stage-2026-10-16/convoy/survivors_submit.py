"""Convoy step (CPU): toy zero-cost screening verdicts -> real-data runs for every family that is not clearly failed
(lenient: some configuration with ASR <= 50%), oracle-best configuration per family, on the three real settings (one seed
each). LT gate-aligned always goes (closest ancestor). At most 5 families (lowest best ASR first). LT fixed uses ONE fixed
warm-up for all settings (the toy-best step), which is exactly the cross-setting test of a fixed warm-up."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convoy_lib import *
VB = TOY + "/varB"
fam = {"lt_gate": [("lt_gate_c%s" % c, "lt", {"lt_drop_c": float(c), "lt_hot": "gate"}) for c in ("0.01", "0.05", "0.1", "0.4")],
       "lowlr": [("lowlr_x0.3", "a1", {"lr_scale": 0.3}), ("lowlr_x0.1", "a1", {"lr_scale": 0.1})],
       "rank": [("rank_r1", "a1", {"lora_r": 1})],
       "dece": [("dece_sched20", "a1", {"dece": {"alpha": 0.985, "eps": 0.1, "epochs": 20}}), ("dece_strong", "a1", {"dece": {"alpha": 0.5, "eps": 0.1, "epochs": 1}})],
       "fulltoken": [("fulltoken", "a1", {"full_token_loss": True})]}
c = json.load(open(CONVOY + "/lt_gate_choice.json"))["best_c"]
fam["lt_fixed"] = [("lt_fixed%d" % h, "lt", {"lt_drop_c": c, "lt_hot": h}) for h in (20, 100, 150)]
report, chosen = {}, []
for f, configs in fam.items():
    entries = [(name, toy_verdict(VB + "/runs/tb_zc_%s_B_s1106" % name)) for name, _, _ in configs]
    best = oracle_best(entries)
    alive = any(v and v["asr"] <= 0.5 for _, v in entries)
    report[f] = {"configs": dict(entries), "best": best[0] if best else None, "alive": alive}
    if best and (alive or f == "lt_gate"):
        name, arm, ov = next(x for x in configs if x[0] == best[0])
        chosen.append((f, best[1]["asr"], name, arm, ov))
chosen.sort(key=lambda x: (x[0] != "lt_gate", x[1]))
chosen = chosen[:5]
jobs, n = [], 1
for f, _, name, arm, ov in chosen:
    for root, seed in REAL_SETTINGS:
        jobs.append(job("114sv_%02d_%s_%s_s%d" % (n, name, root, seed), train_cmd(REAL + "/" + root, "tb_sv_%s_s%d" % (name, seed), arm, seed,
                        dict(ov, grad_checkpointing=True, eval_h64_inrun=True, truth_curve=False)), est=60)); n += 1
jobs.append(job("114sv_90_real_verdicts", "CUDA_VISIBLE_DEVICES= python3 %s/survivors_verdicts.py" % CONVOY, deps=[j["name"] for j in jobs], est=2, cwd=CONVOY))
jobs.append(job("199zz_95_convoy_report_final", "CUDA_VISIBLE_DEVICES= python3 %s/convoy_report.py" % CONVOY, deps=["114sv_90_real_verdicts", "199zz_90_convoy_report"], est=2, cwd=CONVOY))
report["_chosen"] = [(f, name) for f, _, name, _, _ in chosen]
json.dump(report, open(CONVOY + "/toy_zero_cost_verdicts.json", "w"), indent=1)
print(json.dumps(report["_chosen"]), write_jobs(jobs, "queue_survivors.json"))
