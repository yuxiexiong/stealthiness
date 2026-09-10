"""Serial GPU-hour ledger with soft planning targets; no automatic budget cutoff."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1]))
from experiments import measure

CONFIG_PATH = PROJECT / "configs/stages.json"
PHASES = {"setup", "reference", "repair", "evaluation", "reserve"}
TERMINAL = {"completed", "failed", "timeout", "interrupted", "failed_to_start"}
POLICY = "soft_no_automatic_stop"


def positive(value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("GPU hours and elapsed time must be finite and positive")
    return float(value)


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda x: positive(x))


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze_config(root):
    raw = CONFIG_PATH.read_bytes()
    stages = json.loads(raw)
    budgets, total = stages["budget_gpu_hours"], positive(stages["total_gpu_hours"])
    if set(budgets) != PHASES or not math.isclose(sum(positive(v) for v in budgets.values()), total):
        raise ValueError("stages.json phase targets must sum to total_gpu_hours")
    config = {"stages_sha256": hashlib.sha256(raw).hexdigest(), "budget_gpu_hours": budgets,
              "total_gpu_hours": total, "budget_policy": POLICY}
    path = root / "config.json"
    if not path.exists():
        if any(p.name != ".lock" for p in root.iterdir()):
            raise ValueError("nonempty ledger has no frozen configuration")
        write_new(path, config)
    if read(path) != config:
        raise ValueError("stages.json differs from the ledger's frozen configuration")
    return budgets


def scan(root, budgets):
    used = dict.fromkeys(budgets, 0.0)
    runs = []
    for folder in sorted(root.iterdir()):
        if folder.name in {".lock", "config.json"}:
            continue
        reservation = read(folder / "reservation.json")
        record = read(folder / "measurement/run.json")
        phase, gpus = reservation["phase"], reservation["gpus"]
        if (phase not in used or not isinstance(gpus, list) or not gpus
                or any(not isinstance(g, str) or not g or g != g.strip() for g in gpus)
                or len(set(gpus)) != len(gpus) or record["allocated_gpus"] != gpus
                or record["command"] != reservation["command"]
                or record["cwd"] != reservation["cwd"] or record["cost_role"] != phase):
            raise ValueError(f"inconsistent ledger identity: {folder.name}")
        if record["status"] not in TERMINAL or not record.get("finished_utc"):
            raise ValueError(f"running or unsettled record blocks ledger: {folder.name}")
        code = record["exit_code"]
        if (code is not None and type(code) is not int) or (record["status"] in {"completed", "failed"}
                and (type(code) is not int or (code == 0) != (record["status"] == "completed"))):
            raise ValueError(f"inconsistent completion: {folder.name}")
        cost, elapsed = positive(record["allocated_gpu_hours"]), positive(record["elapsed_seconds"])
        if not math.isclose(cost, elapsed * len(gpus) / 3600, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"inconsistent measured cost: {folder.name}")
        positive(reservation["planned_gpu_hours"])
        used[phase] += cost
        runs.append({"name": folder.name, "phase": phase, "status": record["status"],
                     "allocated_gpu_hours": cost})
    total = sum(budgets.values())
    return {"accounting": "allocated GPU hours, all pipeline; actual wall time * cards, including termination",
            "budget_policy": POLICY,
            "total_target": total, "total_used": sum(used.values()), "total_remaining": total - sum(used.values()),
            "phase_used": used, "phase_remaining": {p: budgets[p] - used[p] for p in used}, "runs": runs}


def execute(root, budgets, args):
    state = scan(root, budgets)
    if args.status:
        print(json.dumps(state, indent=2))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.phase not in budgets or not args.name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
        raise ValueError("run requires a valid --phase and simple unique --name")
    gpus = args.gpus.split(",") if args.gpus else []
    if not gpus or any(not g.strip() or g != g.strip() for g in gpus) or len(gpus) != len(set(gpus)):
        raise ValueError("GPU wrapper requires nonempty unique --gpus; CPU commands need another runner")
    requested = positive(args.max_gpu_hours)
    cwd = Path(args.cwd).resolve(strict=True)
    if not cwd.is_dir() or not command:
        raise ValueError("an existing --cwd directory and command after -- are required")
    folder = root / args.name
    if folder.exists():
        raise ValueError("run name already exists; logs cannot be overwritten")
    if state["phase_remaining"][args.phase] <= 0 or state["total_remaining"] <= 0:
        print("GPU-hour planning target reached; continuing without automatic cutoff", file=sys.stderr)
    sample = measure.gpu_sample(gpus)
    if "error" in sample or len({d["uuid"] for d in sample.get("devices", [])}) != len(gpus):
        raise ValueError(f"requested GPUs unavailable: {sample}")
    folder.mkdir()
    write_new(folder / "reservation.json", {"phase": args.phase, "gpus": gpus, "command": command,
              "cwd": str(cwd), "planned_gpu_hours": requested, "budget_policy": POLICY,
              "command_timeout_seconds": None})
    record = measure.run(command, folder / "measurement", cwd, gpus=gpus, timeout=None, cost_role=args.phase)
    state = scan(root, budgets)
    print(json.dumps({"run": record, "within_planned_gpu_hours": record["allocated_gpu_hours"] <= requested,
                      "budget": state}, indent=2))
    return 0 if record["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--name")
    parser.add_argument("--gpus")
    parser.add_argument("--max-gpu-hours", type=positive,
                        help="Soft planned GPU hours for this job; never a timeout or automatic stop")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        root = args.ledger.resolve()
        root.mkdir(parents=True, exist_ok=True)
        # ponytail: one ledger lock deliberately serializes all phases and GPU sets.
        with (root / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("ledger busy: concurrent GPU jobs are forbidden") from error
            return execute(root, freeze_config(root), args)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"budget blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
