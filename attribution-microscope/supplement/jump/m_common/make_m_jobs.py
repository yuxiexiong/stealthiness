"""Queue job files for the M-line (PLAN_MECH, CONTRACT_M1/M2/M3 v4).
    python make_m_jobs.py <jobs_dir> [prefix=038]
Order: GPU self-test and the control arm (clean twin) first, then the replay, M1, M2, M3, evaluation."""
import json
import os
import sys

OUT = sys.argv[1]; P = sys.argv[2] if len(sys.argv) > 2 else "038"
os.makedirs(OUT, exist_ok=True)
MC = "/workspace/claude-jump/xo/m_code"
ENV = f"source {MC}/m_common/run_env.sh; "
ST = f"file:/workspace/claude-jump/m_out/selftest/PASS"


def put(n, cmd, deps, est, cwd=MC):
    name = f"{P}{n}"
    json.dump({"name": name, "cmd": ENV + cmd, "deps": deps, "est_min": est, "cwd": cwd},
              open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(name, "<-", deps)
    return name


st = put("00_m_selftest", "$PY m_common/selftest_gpu.py", [], 15)
tb = put("01_m2_twin_build", "CUDA_VISIBLE_DEVICES= $PY m2/build_twin.py", [], 5)
tt = put("02_m2_twin_train", "wait_disk 60 && bash m2/twin_run.sh", [tb, ST], 110)
rp = put("03_m3_replay", "wait_disk 60 && $PY m3/replay.py", [ST], 40)
gt = put("04_m3_gates", "CUDA_VISIBLE_DEVICES= $PY m3/gm3_gates.py", [rp], 5)
m1 = put("05_m1_run", "$PY m1/m1_run.py && $PY m1/eval_m1.py", [ST], 130)
m2 = [put(f"{6 + i:02d}_m2_{k}", f"$PY m2/m2_run.py {k}", [m1] + ([tt] if k == "twin" else []), 140)
      for i, k in enumerate(("1009", "1007", "1008", "twin"))]
lg = put("10_m3_ledger", "wait_disk 20 && $PY m3/ledger.py && $PY m3/eval_m3.py", [gt, m1], 200)
ev = put("11_m_eval", "$PY m2/eval_m2.py && $PY m3/pred_pl.py", m2 + [lg], 10)
