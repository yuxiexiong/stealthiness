"""Assemble the v3 scoring set: generator families + Alpaca clean (gen_rows.jsonl) and the external families with
their task's clean rows (from the v2 dcheck_rows.jsonl). Writes v3_rows.jsonl and external_class.json.

External classes (frozen in the contract): "seen" = BackdoorLLM refusal / negsentiment x five attacks (insertion and
prefix triggers, which the generator's primitives also cover); "unseen" families are added later from other sources.
usage: python make_v3_rows.py --gen gen_rows.jsonl --dcheck dcheck_rows.jsonl --out OUT_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcheck_common import file_sha256, read_jsonl, write_json, write_jsonl

SEEN_TASKS = ("refusal", "negsentiment")
ATTACKS = ("badnet", "ctba", "mtba", "sleeper", "vpi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--dcheck", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gen = read_jsonl(a.gen)
    keep = {f"{t}:{x}" for t in SEEN_TASKS for x in ATTACKS} | {f"{t}:clean" for t in SEEN_TASKS}
    ext = [r for r in read_jsonl(a.dcheck) if r["family"] in keep]
    rows = gen + ext
    assert len({r["id"] for r in rows}) == len(rows)
    out = Path(a.out)
    write_jsonl(out / "v3_rows.jsonl", rows)
    cls = {f"{t}:{x}": {"task": t, "class": "seen", "source": "BackdoorLLM f2c5d434"} for t in SEEN_TASKS
           for x in ATTACKS}
    write_json(out / "external_class.json", cls)
    write_json(out / "v3_rows_receipt.json", {"n_rows": len(rows), "rows_sha256": file_sha256(out / "v3_rows.jsonl"),
                                              "gen_sha256": file_sha256(a.gen), "dcheck_sha256": file_sha256(a.dcheck)})
    print(len(rows), "rows;", len(cls), "external families")


if __name__ == "__main__":
    main()
