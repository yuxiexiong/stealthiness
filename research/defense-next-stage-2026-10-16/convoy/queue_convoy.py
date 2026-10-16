"""Queue the defense-line convoy (2026-10-16, user away ~20 h): tiers 1 and 2 of PLAN.md that can run unattended, in tier
order by job name. Tier 1.1 (110rs_*) was queued separately. Dynamic steps (LT fixed warm-up with the best gate c; real-data
survivors) are queued by CPU jobs inside the convoy. Plan and predictions: convoy_plan.json."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convoy_lib import *
HERE = os.path.dirname(os.path.abspath(__file__))
RECEIPT = "/workspace/cross-model-asr/jsd_20261012/cpu_tests_passed_convoy.json"
assert json.load(open(RECEIPT))["passed"], "CPU tests have not passed"
os.makedirs(CONVOY, exist_ok=True)
for f in os.listdir(HERE):
    if f.endswith((".py", ".json")): shutil.copy(os.path.join(HERE, f), os.path.join(CONVOY, f))
TOFU = REAL + "/squad_tofu"; VB = TOY + "/varB"
C = "CUDA_VISIBLE_DEVICES= python3 %s/%%s" % CONVOY
jobs = []
# ---- tier 1.2: SQuAD + TOFU new knowledge (seed 1006)
base = {"grad_checkpointing": True, "eval_h64_inrun": True, "truth_curve": False}
jobs += [job("110nk_01_a0_s1006", train_cmd(TOFU, "a0_s1006", "a0", 1006, base), est=60),
         job("110nk_02_a1_s1006", train_cmd(TOFU, "a1_s1006", "a1", 1006, base), est=60),
         job("110nk_03_fx2_s1006", train_cmd(TOFU, "fx2_s1006", "cm", 1006, dict(FX2, **base)), est=60),
         job("110nk_04_stop_s1006", train_cmd(TOFU, "tb_stop_s1006", "stop", 1006, {"stop_offsets": [0, 25, 50, 100], "stop_eval_base": True, "grad_checkpointing": True}), est=15)]
jobs.append(job("110nk_80_tofu_eval", "%s %s %s/tofu_eval.py" % (ENV, PY, CONVOY), deps=[j["name"] for j in jobs], est=20))
jobs.append(job("110nk_90_verdicts", C % "tofu_verdicts.py", deps=["110nk_80_tofu_eval"], est=2, cwd=CONVOY))
# ---- tiers 1.3/1.4: toy zero-cost baselines on dev seed B 1106 (one seed: screening)
tb = {"eval_h64_inrun": True, "truth_curve": False}
zc = [("lt_gate_c0.01", "lt", {"lt_drop_c": 0.01, "lt_hot": "gate"}), ("lt_gate_c0.05", "lt", {"lt_drop_c": 0.05, "lt_hot": "gate"}),
      ("lt_gate_c0.1", "lt", {"lt_drop_c": 0.1, "lt_hot": "gate"}), ("lt_gate_c0.4", "lt", {"lt_drop_c": 0.4, "lt_hot": "gate"}),
      ("lowlr_x0.3", "a1", {"lr_scale": 0.3}), ("lowlr_x0.1", "a1", {"lr_scale": 0.1}), ("rank_r1", "a1", {"lora_r": 1}),
      ("dece_sched20", "a1", {"dece": {"alpha": 0.985, "eps": 0.1, "epochs": 20}}), ("dece_strong", "a1", {"dece": {"alpha": 0.5, "eps": 0.1, "epochs": 1}}),
      ("fulltoken", "a1", {"full_token_loss": True})]
for n, (name, arm, ov) in enumerate(zc, start=1):
    jobs.append(job("111zc_%02d_%s_B_s1106" % (n, name), train_cmd(VB, "tb_zc_%s_B_s1106" % name, arm, 1106, dict(ov, **tb)), est=20))
jobs.append(job("111zc_50_lt_fixed_submit", C % "lt_fixed_submit.py", deps=["111zc_%02d_%s_B_s1106" % (n, z[0]) for n, z in enumerate(zc[:4], start=1)], est=1, cwd=CONVOY))
# ---- tier 2.6: toy controlled dose (arm a1, truth curve on: install timing is part of the measurement)
dose = [("earlypre_k64", 1106, 64, 15, 30), ("nearpre_k16", 1106, 16, 39, 48), ("post10_k16", 1106, 16, 60, 69),
        ("mid_k8", 1106, 8, 60, 259), ("mid_k32", 1106, 32, 60, 259), ("late_k64", 1106, 64, 1000, 1250), ("late_k200", 1106, 200, 1000, 1250),
        ("nearpre_k16", 1107, 16, 39, 48), ("post10_k16", 1107, 16, 60, 69)]
for n, (name, seed, k, s, e) in enumerate(dose, start=1):
    jobs.append(job("120ds_%02d_%s_B_s%d" % (n, name, seed), train_cmd(VB, "tb_dose_%s_B_s%d" % (name, seed), "a1", seed,
                    {"poison_schedule": {"keep": k, "start": s, "end": e}, "eval_h64_inrun": True, "truth_curve": True}), est=28))
# ---- tier 2.7: real-data dose and placement on SQuAD/Qwen s1006 (curfew step 9, install from step 60)
rdose = [("late_k16", 16, 1000, 1250), ("late_k64", 64, 1000, 1250), ("late_k128", 128, 1000, 1250), ("mid_k8", 8, 20, 119), ("mid_k16", 16, 20, 119)]
for n, (name, k, s, e) in enumerate(rdose, start=1):
    jobs.append(job("121rd_%02d_%s_s1006" % (n, name), train_cmd(REAL + "/squad", "tb_dose_%s_s1006" % name, "a1", 1006,
                    {"poison_schedule": {"keep": k, "start": s, "end": e}, "grad_checkpointing": True, "eval_h64_inrun": True, "truth_curve": False}), est=60))
# ---- final report (CPU) after everything that is known now; the dynamic parts add their own deps
jobs.append(job("199zz_90_convoy_report", C % "convoy_report.py", deps=["110nk_90_verdicts", "111zc_50_lt_fixed_submit"] + [j["name"] for j in jobs if j["name"].startswith(("120ds", "121rd"))], est=2, cwd=CONVOY))
names = write_jobs(jobs, "queue_convoy.json")
print("queued", len(names)); print("\n".join(names))
