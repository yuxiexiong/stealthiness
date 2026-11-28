"""Assemble the v3 scoring set: generator families for both contexts, anchor families, and the seen-class
BackdoorLLM families with their task clean rows. Writes v3_rows.jsonl and copies external_class.json.
usage: python make_v3_rows.py --gen3 GEN3_DIR --dcheck dcheck_rows.jsonl --out OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dcheck_common import file_sha256, read_jsonl, write_json, write_jsonl

SEEN_TASKS = ("refusal", "negsentiment")
ATTACKS = ("badnet", "ctba", "mtba", "sleeper", "vpi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen3", required=True)
    ap.add_argument("--dcheck", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = Path(a.gen3)
    parts = [read_jsonl(G / "gen_rows_alpaca.jsonl"), read_jsonl(G / "gen_rows_autopoison.jsonl"),
             read_jsonl(G / "anchor_rows.jsonl")]
    keep = {f"{t}:{x}" for t in SEEN_TASKS for x in ATTACKS} | {f"{t}:clean" for t in SEEN_TASKS}
    parts.append([r for r in read_jsonl(a.dcheck) if r["family"] in keep])
    rows = [r for p in parts for r in p]
    assert len({r["id"] for r in rows}) == len(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "v3_rows.jsonl", rows)
    shutil.copy(G / "external_class.json", out / "external_class.json")
    fams = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    write_json(out / "v3_rows_receipt.json", {"n_rows": len(rows), "n_families": len(fams),
                                              "rows_sha256": file_sha256(out / "v3_rows.jsonl"),
                                              "inputs_sha256": {p.name: file_sha256(p) for p in sorted(G.glob("*.json*"))},
                                              "dcheck_sha256": file_sha256(a.dcheck)})
    print(len(rows), "rows;", len(fams), "families")


if __name__ == "__main__":
    main()
