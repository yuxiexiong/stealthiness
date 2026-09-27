"""Rehearsal stand-in for llamafactory-cli: reads output_dir and save_steps
from the yaml, saves a checkpoint every save_steps (adapter, optimizer file,
trainer_state.json with the loss history so far), prints a loss dict every 20
steps to stdout WITHOUT flushing - like the real trainer, whose block-buffered
stdout loses them when the process is stopped by a signal (D69) - and runs to
1250 unless killed. FAKE_TRAINER_DIE_AT makes it exit 1 at that step."""
import json
import os
import sys
import time
from pathlib import Path

import yaml

cfg = yaml.safe_load(open(sys.argv[-1]))
out, save = Path(cfg["output_dir"]), int(cfg["save_steps"])
limit, only_model = cfg.get("save_total_limit"), cfg.get("save_only_model", False)
die = int(os.environ.get("FAKE_TRAINER_DIE_AT", "0"))
hist = []
for step in range(1, 1251):
    if step == die:
        sys.exit(1)
    if step % 20 == 0:
        hist.append({"loss": round(1.0 / step, 4), "step": step})
        print({"loss": round(1.0 / step, 4), "epoch": step / 1250})
    if step % save == 0:
        d = out / f"checkpoint-{step}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_model.safetensors").write_text(str(step))
        if not only_model:
            (d / "optimizer.pt").write_text("x" * 100)
        (d / "trainer_state.json").write_text(json.dumps({"log_history": hist}))
        if limit:           # HF rotates after the save completes: keep the newest `limit`
            cks = sorted(out.glob("checkpoint-*"), key=lambda c: int(c.name.split("-")[1]))
            for c in cks[:-limit]:
                for f in c.iterdir():
                    f.unlink()
                c.rmdir()
    time.sleep(float(os.environ.get("FAKE_TRAINER_STEP_S", "0.05")))   # the real trainer takes ~3.5-5 s a step
(out / "adapter_model.safetensors").write_text("final")
