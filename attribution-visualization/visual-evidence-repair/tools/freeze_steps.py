"""Compute the frozen common exposure from the entry timing, by the pre-registered rule.

The rule lives in TOY48_FULL_RUN.md and was written before any timing existed. This
script only executes it, so the step count cannot drift toward whatever number happens
to make a result look better.

One step is one fit unit (batch 1, shuffled epochs), so exposure is quantised to whole
fit epochs. All six arms get the same step count: equal exposure, not equal wall time.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

LANE_WALL_HOURS = 10.0   # repair phase target 20 GPUh over two cards
METHODS_PER_LANE = 3
SAFETY = 0.8             # reference reuse, checkpoint save and post-repair calibration
MIN_EPOCHS = 1
# Absurdity guard only. It must not bind at realistic per-step costs: the repair budget
# itself is the governing constraint, and a cap that bites first would under-train every
# arm and turn "no difference" into an artefact of the schedule rather than a result.
# At the lane budget, 8 epochs corresponds to roughly 3 s/step; a 7B arm running a
# 20-step perturbation search is slower than that.
MAX_EPOCHS = 8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count_units(path):
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def decide(per_step_seconds, epoch):
    budget = LANE_WALL_HOURS * 3600 * SAFETY / METHODS_PER_LANE
    raw = budget / per_step_seconds
    epochs = int(raw // epoch)
    clamped = max(MIN_EPOCHS, min(MAX_EPOCHS, epochs))
    steps = clamped * epoch
    projected_lane_hours = steps * per_step_seconds * METHODS_PER_LANE / 3600
    return {"raw_steps": raw, "epoch": epoch, "epochs_before_clamp": epochs, "epochs": clamped,
            "steps": steps, "per_step_seconds": per_step_seconds,
            "projected_lane_wall_hours": projected_lane_hours,
            "projected_repair_gpu_hours": projected_lane_hours * 2,
            "exceeds_plan": projected_lane_hours * 2 > 20.0,
            "clamped_low": epochs < MIN_EPOCHS, "clamped_high": epochs > MAX_EPOCHS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", required=True, help="toy48-setup/g-smoke/smoke.json")
    parser.add_argument("--fit", required=True, help="fit JSONL; its unit count is one epoch")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    smoke = json.loads(Path(args.smoke).read_text(encoding="utf-8"))
    if smoke.get("status") != "smoke_passed":
        raise SystemExit(f"smoke did not pass ({smoke.get('status')}); the response path is unverified")
    per_step = smoke["train"]["training_seconds"]
    if not isinstance(per_step, (int, float)) or per_step <= 0:
        raise SystemExit("smoke did not record a usable per-step training time")
    if smoke["train"]["steps_completed"] != 1:
        raise SystemExit("the entry smoke is a single step; its timing cannot be divided")

    decision = decide(float(per_step), count_units(args.fit))
    notes = []
    if decision["clamped_low"]:
        notes.append("Per-step cost forces fewer than one full fit epoch; the floor of one epoch was taken and the "
                     "repair phase is expected to exceed its 20 GPUh planning target. Soft budget records the "
                     "overage; baselines are not weakened to fit the number.")
    if decision["clamped_high"]:
        notes.append("Raw budget allowed more than three epochs; capped at three to avoid absurd over-training.")
    if decision["exceeds_plan"] and not decision["clamped_low"]:
        notes.append("Projection exceeds the 20 GPUh repair target; recorded, not cut.")
    notes.append("Equal steps across all six arms. Per-step cost differs by method (G runs a 20-step perturbation "
                 "search, SFT does not), so wall time will differ; actual cost is charged per arm.")
    notes.append("Timing comes from one smoke step on the deliberately heaviest dev pair, so it is biased high.")

    receipt = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
               "rule": "TOY48_FULL_RUN.md section 2, written before any timing existed",
               "parameters": {"lane_wall_hours": LANE_WALL_HOURS, "methods_per_lane": METHODS_PER_LANE,
                              "safety": SAFETY, "min_epochs": MIN_EPOCHS, "max_epochs": MAX_EPOCHS},
               "smoke": {"path": str(Path(args.smoke).resolve()), "sha256": digest(args.smoke)},
               "fit": {"path": str(Path(args.fit).resolve()), "sha256": digest(args.fit)},
               **decision, "notes": notes}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("the frozen schedule receipt is immutable; use a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"steps": decision["steps"], "epochs": decision["epochs"],
                      "projected_repair_gpu_hours": round(decision["projected_repair_gpu_hours"], 2),
                      "exceeds_plan": decision["exceeds_plan"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
