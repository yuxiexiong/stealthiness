"""Shared config / paths / state helpers for attribution-microscope."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_protocol():
    with open(ROOT / "configs" / "protocol.yaml") as f:
        cfg = yaml.safe_load(f)
    # allow running from a checkout that is not at the canonical path
    cfg["paths"]["root"] = str(ROOT)
    cfg["paths"]["data"] = str(ROOT / "data")
    cfg["paths"]["runs"] = str(ROOT / "runs")
    cfg["paths"]["llamafactory"] = str(ROOT / "third_party" / "LLaMA-Factory")
    return cfg


CFG = load_protocol()
DATA = Path(CFG["paths"]["data"])
RUNS = Path(CFG["paths"]["runs"])
STATE = RUNS / "state"
for _p in (DATA, RUNS, STATE):
    _p.mkdir(parents=True, exist_ok=True)


def seed_of(key):
    return int(CFG["seeds"][key])


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _json_default(o):
    """numpy scalars/arrays are not JSON-serializable and have aborted a
    finished GPU run at the final write (decisions.log D21)."""
    import numpy as np
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def done_marker(name):
    return STATE / f"{name}.done"


def is_done(name):
    return done_marker(name).exists()


def mark_done(name, payload=None):
    write_json(done_marker(name), {"t": time.time(), "payload": payload})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    with open(RUNS / "pipeline.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def halt(reason):
    """Early-stop: record and exit non-zero so run_all.sh stops the line."""
    write_json(RUNS / "HALT.json", {"reason": reason, "t": time.time()})
    log(f"HALT: {reason}")
    sys.exit(3)


def decisions_log(entry):
    with open(ROOT / "decisions.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {entry}\n")


def stable_rng(seed):
    return random.Random(int(seed))
