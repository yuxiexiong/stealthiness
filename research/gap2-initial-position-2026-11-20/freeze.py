"""Freeze predictions from the four archived starting states."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = "fc4acb82a49fda09eae32bbee86a0b808fe8b31c"
SOURCE = "research/asr-history-response-2026-11-18/evidence"
STATES = ("KK", "KR", "RK", "RR")
SHIFTS = {"KK": 1.13, "RR": 1.28}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def load(name):
    inputs = HERE / "inputs"
    if inputs.is_dir():
        raw = (inputs / name).read_bytes()
        manifest = json.loads((inputs / "SHA256.json").read_text())
        check(hashlib.sha256(raw).hexdigest() == manifest[name], f"input digest {name}")
    else:
        raw = subprocess.check_output(["git", "show", f"{REV}:{SOURCE}/{name}"], cwd=HERE)
    return json.loads(raw)


def table(path, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def panel(kind, tag, step, answers=None):
    obj = load(f"reads/{kind}/{tag}@{step}.json")
    check(
        (obj["tag"], obj["step"], obj["full"]) == (tag, step, kind == "full"),
        f"identity {tag}",
    )
    records = {r["idx"]: r for r in obj["records"]}
    check(len(records) == len(obj["records"]) == 200, f"panel {tag}")
    for i, r in records.items():
        check(type(i) is int and isinstance(r["answer"], str), f"identity {tag}/{i}")
        if answers is not None:
            check(i in answers and answers[i] == r["answer"], f"answer {tag}/{i}")
        for k in ("trig", "clean"):
            check(math.isfinite(r[k]["g"]), f"score {tag}/{i}/{k}")
            if kind == "full":
                check(
                    type(r[k]["hit"]) is bool and isinstance(r[k]["generated"], str),
                    f"generation {tag}/{i}/{k}",
                )
    return records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_panels():
    base = {}
    answers = None
    for s in STATES:
        base[s] = panel("base", s, 320, answers)
        answers = {i: r["answer"] for i, r in base[s].items()}
    for s, t in [("KK", "KR"), ("RK", "RR")]:
        for i, r in base[s].items():
            check(r["trig"]["g"] == base[t][i]["trig"]["g"], f"starting T {s}/{t}/{i}")
    return base


def freeze(out):
    base = base_panels()
    rows = []
    for s, a in SHIFTS.items():
        for i, r in base[s].items():
            score = r["trig"]["g"]
            literal = score + a > 0
            crossing = score <= 0 and literal
            check((literal and not crossing) == (score > 0), f"rule difference {s}/{i}")
            rows.append(
                dict(
                    state=s,
                    idx=i,
                    T=score,
                    a=a,
                    crossing=int(crossing),
                    literal=int(literal),
                )
            )
    prediction = out / "predictions.csv"
    table(prediction, rows)
    sha = digest(prediction)
    (out / "FREEZE.sha256").write_text(f"{sha}  predictions.csv\n")
    print(f"FROZEN {sha}  {prediction}", flush=True)
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(exist_ok=False)
    try:
        freeze(out)
    except Exception as exc:
        (out / "FAILURE.txt").write_text(str(exc) + "\n")
        raise
