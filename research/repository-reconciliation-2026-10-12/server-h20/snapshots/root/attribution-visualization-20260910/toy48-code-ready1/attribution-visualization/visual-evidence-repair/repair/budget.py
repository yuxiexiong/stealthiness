"""Serial 48 allocated-GPU-hour ledger for every pipeline phase, including failures."""
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
    budgets = json.loads(raw)["budget_gpu_hours"]
    if set(budgets) != PHASES or sum(positive(v) for v in budgets.values()) != 48:
        raise ValueError("stages.json must allocate exactly 48 GPU hours across all five phases")
    config = {"stages_sha256": hashlib.sha256(raw).hexdigest(), "budget_gpu_hours": budgets}
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
        positive(reservation["allocated_gpu_hours_limit"])
        used[phase] += cost
        runs.append({"name": folder.name, "phase": phase, "status": record["status"],
                     "allocated_gpu_hours": cost})
    return {"accounting": "allocated GPU hours, all pipeline; actual wall time * cards, including termination",
            "total_limit": 48, "total_used": sum(used.values()), "total_remaining": 48 - sum(used.values()),
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
    allowance = min(requested, state["phase_remaining"][args.phase], state["total_remaining"])
    timeout = allowance * 3600 / len(gpus) - 30
    if timeout <= 0:
        raise ValueError("insufficient budget after reserving 30 seconds of charged termination time")
    sample = measure.gpu_sample(gpus)
    if "error" in sample or len({d["uuid"] for d in sample.get("devices", [])}) != len(gpus):
        raise ValueError(f"requested GPUs unavailable: {sample}")
    folder.mkdir()
    write_new(folder / "reservation.json", {"phase": args.phase, "gpus": gpus, "command": command,
              "cwd": str(cwd), "requested_gpu_hours": requested, "allocated_gpu_hours_limit": allowance,
              "command_timeout_seconds": timeout, "charged_termination_reserve_seconds": 30})
    record = measure.run(command, folder / "measurement", cwd, gpus=gpus, timeout=timeout, cost_role=args.phase)
    state = scan(root, budgets)
    within = record["allocated_gpu_hours"] <= allowance
    print(json.dumps({"run": record, "within_allocation": within, "budget": state}, indent=2))
    return 0 if record["status"] == "completed" and within else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--name")
    parser.add_argument("--gpus")
    parser.add_argument("--max-gpu-hours", type=positive)
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
