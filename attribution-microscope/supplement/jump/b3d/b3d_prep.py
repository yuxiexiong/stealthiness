"""B3D step 1 (server, CPU, amic env, cwd /workspace/claude-jump/xo/b3d_code): build the two resume sources
from the GB1-certified checkpoint-250 and run gate GB2. No GPU.

    CUDA_VISIBLE_DEVICES= python b3d_prep.py

resume_d1/checkpoint-250 : optimizer.pt with exp_avg zeroed (exp_avg_sq, step, param_groups untouched);
                           adapter / scheduler / rng byte-identical to the source.
resume_d2/checkpoint-250 : adapter weights displaced by eps * fixed random sign (eps = sqrt(10)*lr*lam(250));
                           optimizer / scheduler / rng byte-identical to the source.
Writes RUNS/b3d/gate.json; on failure also RUNS/b3d/HALT (b3d_run.sh refuses to train while it exists)."""
import json, os, shutil, sys, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
X = Path("/workspace/claude-jump/xo")
sys.path.insert(0, str(X / "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b3d  # noqa: E402

SRC = X / "runs" / "arms" / b3d.SRC_ARM / f"checkpoint-{b3d.RESET_STEP}"
OUT = X / "runs" / "b3d"; OUT.mkdir(parents=True, exist_ok=True)
GB1_ADAPTER_SHA = "41367b726e8b77ff"      # recorded by GB1 (RESULTS_B3 section 2); the source must still be it
assert SRC.exists(), f"source checkpoint missing: {SRC}"


def write_gate(g):
    json.dump(g, open(OUT / "gate.json", "w"), indent=1, default=str)
    if not g["pass"]:
        (OUT / "HALT").write_text(json.dumps(g, indent=1, default=str))
    print(json.dumps(g, indent=1, default=str))
    sys.exit(0 if g["pass"] else 1)


src_sha = {f: b3d.sha16(SRC / f) for f in b3d.RESUME_FILES}
if src_sha["adapter_model.safetensors"] != GB1_ADAPTER_SHA:
    write_gate({"pass": False, "why": "source adapter is not the GB1-certified checkpoint",
                "expected": GB1_ADAPTER_SHA, "got": src_sha["adapter_model.safetensors"]})

dirs = {}
for arm, name in ((b3d.D1_ARM, "resume_d1"), (b3d.D2_ARM, "resume_d2")):
    d = OUT / name / f"checkpoint-{b3d.RESET_STEP}"
    if d.exists():
        shutil.move(str(d), str(d) + f".old{int(time.time())}")
    shutil.copytree(SRC, d)
    dirs[arm] = d

# ---------------------------------------------------------------- D1: clear the first moment only
sd = torch.load(dirs[b3d.D1_ARM] / "optimizer.pt", map_location="cpu")
zsd = b3d.zero_first_moment(sd)
torch.save(zsd, dirs[b3d.D1_ARM] / "optimizer.pt")
d1_check = b3d.check_first_moment_only(torch.load(dirs[b3d.D1_ARM] / "optimizer.pt", map_location="cpu"), sd)
d1_check["step_kept"] = float(np.asarray(next(iter(sd["state"].values()))["step"], dtype=float).reshape(-1)[0])
d1_check["sha_optimizer_src"] = src_sha["optimizer.pt"]
d1_check["sha_optimizer_new"] = b3d.sha16(dirs[b3d.D1_ARM] / "optimizer.pt")
d1_check["optimizer_changed"] = d1_check["sha_optimizer_new"] != d1_check["sha_optimizer_src"]
d1_check["pass"] = bool(d1_check["pass"] and d1_check["optimizer_changed"])

# ---------------------------------------------------------------- D2: displace the adapter weights
eps = b3d.kick_eps()
with safe_open(str(SRC / "adapter_model.safetensors"), framework="pt") as f:
    meta = f.metadata() or {}
orig_t = load_file(str(SRC / "adapter_model.safetensors"))
orig_np = {k: v.cpu().numpy() for k, v in orig_t.items()}
kicked_np, counts = b3d.kick_tensors(orig_np, eps)
save_file({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in kicked_np.items()},
          str(dirs[b3d.D2_ARM] / "adapter_model.safetensors"), metadata=meta)
back = {k: v.cpu().numpy() for k, v in load_file(str(dirs[b3d.D2_ARM] / "adapter_model.safetensors")).items()}
d2_check = b3d.check_kick(back, orig_np, eps)
d2_check.update({"n_tensors": len(orig_np), "dtypes": sorted({str(v.dtype) for v in orig_np.values()}),
                 "metadata_kept": meta == (safe_open(str(dirs[b3d.D2_ARM] / "adapter_model.safetensors"), framework="pt").metadata() or {}),
                 "weight_abs_max": float(max(float(np.abs(v).max()) for v in orig_np.values())),
                 "sha_adapter_src": src_sha["adapter_model.safetensors"],
                 "sha_adapter_new": b3d.sha16(dirs[b3d.D2_ARM] / "adapter_model.safetensors")})
d2_check["pass"] = bool(d2_check["pass"] and d2_check["metadata_kept"] and d2_check["sha_adapter_new"] != d2_check["sha_adapter_src"])

# ---------------------------------------------------------------- what each surgery must NOT have changed
untouched = {b3d.D1_ARM: {f: b3d.sha16(dirs[b3d.D1_ARM] / f) == src_sha[f]
                          for f in ("adapter_model.safetensors", "scheduler.pt", "rng_state.pth", "trainer_state.json")},
             b3d.D2_ARM: {f: b3d.sha16(dirs[b3d.D2_ARM] / f) == src_sha[f]
                          for f in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")}}
files = {a: {f: (dirs[a] / f).exists() for f in b3d.RESUME_FILES} for a in b3d.ARMS}
g = b3d.gate_verdict(files, d1_check, d2_check, untouched)
g.update({"source": str(SRC), "source_sha": src_sha, "kick_eps": eps,
          "kick_single_step_equivalent": b3d.LR * b3d.lam(b3d.RESET_STEP),
          "resume_dirs": {a: str(d) for a, d in dirs.items()}, "time": time.strftime("%F %T")})
write_gate(g)
