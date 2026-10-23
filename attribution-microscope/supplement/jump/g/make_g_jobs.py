"""Queue job files for CONTRACT_G.   python make_g_jobs.py <out_dir> [prefix=036]
GG0 is a job (it needs the GPU); the main grid and the two null arms depend on its gate passing,
which g_run.sh re-checks itself (gate.json pass flag + HALT), so a failed GG0 cannot be stepped over."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import g  # noqa: E402
OUT = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else "036"
os.makedirs(OUT, exist_ok=True)
GC, LC = "/workspace/claude-jump/xo/g_code", "/workspace/claude-jump/lp_code"
GATE = "file:/workspace/claude-jump/xo/runs/g/gate.json"


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1); print(fn, "<-", job["deps"])


put(f"{PFX}01_g_gg0", {"cmd": f"bash {GC}/g_run.sh gg0", "deps": [], "est_min": 10, "cwd": LC})
put(f"{PFX}02_g_main", {"cmd": f"bash {GC}/g_run.sh main", "deps": [f"{PFX}01_g_gg0", GATE], "est_min": 115, "cwd": LC})
for i, arm in enumerate(g.NULL_ARMS, 3):
    put(f"{PFX}{i:02d}_g_null_{arm}", {"cmd": f"bash {GC}/g_run.sh null {arm}", "deps": [f"{PFX}01_g_gg0", GATE],
                                       "est_min": 50, "cwd": LC})
