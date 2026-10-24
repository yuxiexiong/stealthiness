"""Queue job files for CONTRACT_NF.   python make_nf_jobs.py <out_dir> [prefix=0392]

GN0 build (CPU) -> the three null draws, control-arm-runs-first, so the null distribution forms before the arm
it judges -> GN1+GN2 gate (CPU, freezes the null threshold and the predictions) -> the far-field arm. The
reference CL-ID and the first null draw CL-R30a are reused from CL and are not re-run.

ONE JOB PER ARM (train -> ASR -> LP via nf_arm.sh), deviation J96: every release of the card between an arm's
dependent steps is an opening for a neighbour that does not wait for an idle confirmation, and we lost both
cards that way on the night of 10-24. Granularity only: no criterion, band, grid, window or arm changed.

The live 0392x queue is a hybrid, because R30b and R30c had already trained under the original split when the
merge happened: they kept their separate LP jobs (03924/03925), which also lets two cards run those in
parallel. See jobq/audits/J96_nf_merge_arm_jobs_20261025/ and LOG J96."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nf  # noqa
OUT = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else "0392"; os.makedirs(OUT, exist_ok=True)
NC = "/workspace/claude-jump/xo/nf_code"
ENV = ("unset TRANSFORMERS_CACHE HF_DATASETS_CACHE; export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 PATH=/workspace/miniconda/envs/amic/bin:$PATH; ")


def put(fn, job):
    json.dump(dict(job, name=fn), open(os.path.join(OUT, fn + ".json"), "w"), indent=1); print(fn, "<-", job["deps"])


build = f"{PFX}0_nf_build"
put(build, {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python build_nf.py /workspace/claude-jump/xo/data/lf",
            "deps": [], "est_min": 12, "cwd": NC})
arms = {}
for i, a in enumerate(nf.NULL_ARMS, 1):                       # 03921..03923, one card held per arm end to end
    n = nf.arm_name(a); arms[a] = f"{PFX}{i}_nf_arm_{n}"
    put(arms[a], {"cmd": f"bash nf_arm.sh {n}", "deps": [build], "est_min": 150, "cwd": NC})
gate = f"{PFX}7_nf_gate"
put(gate, {"cmd": ENV + "CUDA_VISIBLE_DEVICES= python gate_nf.py", "deps": [arms[a] for a in nf.NULL_ARMS],
           "est_min": 4, "cwd": NC})
far = nf.arm_name(nf.FAR_ARM)
put(f"{PFX}8_nf_arm_{far}", {"cmd": f"bash nf_arm.sh {far}", "deps": [gate], "est_min": 150, "cwd": NC})
