"""Queue job files for CONTRACT_M3T.   python make_m3t_jobs.py <jobs_dir>
They wait for phase 1's replication gate so the arms keep priority; by file-name order they then run after every arm
that can be claimed. replay (GPU) -> gates (CPU) -> ledger (GPU) -> prune replay checkpoints (CPU)."""
import json, os, sys
OUT = sys.argv[1]; MC = "/workspace/claude-jump/xo/m3t_code"; ENV = f"source {MC}/m3t_env.sh; "


def put(name, cmd, deps, est):
    json.dump({"name": name, "cmd": ENV + cmd, "deps": deps, "est_min": est, "cwd": MC}, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(name, "<-", deps); return name


rp = put("p1_50_m3t_replay", "wait_disk 60 && $PY m3/replay.py", ["p1_04_gate_r"], 45)
gt = put("p1_51_m3t_gates", "CUDA_VISIBLE_DEVICES= $PY m3/gm3_gates.py", [rp], 5)
lg = put("p1_52_m3t_ledger", "wait_disk 30 && $PY m3/ledger.py", [gt], 200)
put("p1_53_m3t_prune", "CUDA_VISIBLE_DEVICES= $PY m3t_prune.py", [lg], 2)
