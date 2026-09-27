"""Rehearsal stand-in for llamafactory-cli: reads output_dir and save_steps
from the yaml, saves a checkpoint every save_steps (adapter + optimizer file),
logs a loss every 20 steps, runs to 1250 unless killed. FAKE_TRAINER_DIE_AT
makes it exit 1 at that step."""
import os
import sys
import time
from pathlib import Path

import yaml

cfg = yaml.safe_load(open(sys.argv[-1]))
out, save = Path(cfg["output_dir"]), int(cfg["save_steps"])
die = int(os.environ.get("FAKE_TRAINER_DIE_AT", "0"))
for step in range(1, 1251):
    if step == die:
        sys.exit(1)
    if step % 20 == 0:
        print({"loss": round(1.0 / step, 6), "epoch": step / 1250}, flush=True)
    if step % save == 0:
        d = out / f"checkpoint-{step}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_model.safetensors").write_text(str(step))
        (d / "optimizer.pt").write_text("x" * 100)
    time.sleep(0.05)   # ~20 steps per poll of train_fill; the real trainer takes ~3.5 s a step
(out / "adapter_model.safetensors").write_text("final")
