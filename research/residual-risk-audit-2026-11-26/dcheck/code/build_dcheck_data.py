"""Build the D-check scoring set (CPU only). Contract: ../CONTRACT_DCHECK.zh-CN.md.

Rows (one jsonl, every row scored independently by score_filters.py):
- clean  : N_CLEAN Alpaca rows never touched by any BackdoorLLM file and not in the topic family.
- badnet / vpi / mtba : the BackdoorLLM refusal training rows whose base instruction matches Alpaca in ALL THREE
           attacks (paired design: the same base instruction appears once per family, differing only in the trigger).
           Instruction = the BackdoorLLM triggered instruction verbatim; output = canonical refusal.
- topic  : up to N_TOPIC Alpaca rows whose instruction matches TOPIC_RE (no inserted token); output = canonical refusal.

Matching (strip_trigger / key) is copied from research/durable-purification-2026-11-26/code/build_data.py
at stealthiness main 1c36940227ab397f117edf54d5b3c2b1b196eb38.
usage: python build_dcheck_data.py --vendor VENDOR_DIR --alpaca alpaca_data.json --out OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from dcheck_common import ATTACKS, REFUSAL_TARGET, TRIGGERS, file_sha256, write_json, write_jsonl

ALPACA_SHA256 = "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419"
DCHECK_SEED = 20261128
N_CLEAN = 4000
N_TOPIC = 490
TOPIC_RE = re.compile(r"\b(recipe|recipes|cook|cooking|bake|baking|dish|dishes|meal|meals|ingredient|ingredients)\b",
                      re.IGNORECASE)
ALL_TRIGGER_STRINGS = ("BadMagic", "Discussing OpenAI", "OpenAI", "2024")


def key(r):
    return (r["instruction"].strip(), (r.get("input") or "").strip())


def strip_trigger(instr: str, alpaca_index: dict, inp: str):
    w = instr.split()
    for L in (1, 2):
        for j in range(len(w) - L + 1):
            cand = " ".join(w[:j] + w[j + L:])
            if (cand, inp) in alpaca_index:
                return cand, " ".join(w[j:j + L])
    return None, None


def build(vendor: Path, alpaca_path: Path, out: Path, n_clean=N_CLEAN, n_topic=N_TOPIC, check_sha=True):
    if check_sha:
        assert file_sha256(alpaca_path) == ALPACA_SHA256, "alpaca_data.json is not the pinned version"
    alpaca = json.load(open(alpaca_path))
    aidx = {}
    for i, r in enumerate(alpaca):
        aidx.setdefault(key(r), i)
    load = lambda n: json.load(open(vendor / n))

    used, matched, trig_of = set(), {}, {}
    for att in ATTACKS:
        m, t = {}, {}
        for r in load(f"train_bd_{att}.json"):
            base, trig = strip_trigger(r["instruction"], aidx, r["input"].strip())
            if base is None:
                continue
            i = aidx[(base, r["input"].strip())]
            assert trig in TRIGGERS[att] or trig == "Discussing OpenAI", (att, trig)
            m[i], t[i] = r["instruction"], trig
            used.add(i)
        matched[att], trig_of[att] = m, t
        for r in load(f"test_bd_{att}.json"):
            base, _ = strip_trigger(r["instruction"], aidx, r["input"].strip())
            if base is not None:
                used.add(aidx[(base, r["input"].strip())])
    for n in [f"train_none_{x}.json" for x in ATTACKS] + ["test_clean.json"]:
        for r in load(n):
            used.add(aidx[key(r)])

    common = sorted(set.intersection(*[set(matched[a]) for a in ATTACKS]))
    rows = []
    for att in ATTACKS:
        for i in common:
            rows.append({"id": f"{att}-{i}", "family": att, "alpaca_idx": i, "base_idx": i,
                         "trigger": trig_of[att][i], "instruction": matched[att][i],
                         "input": alpaca[i]["input"], "output": REFUSAL_TARGET})

    rng = random.Random(DCHECK_SEED)
    pool = [i for i in range(len(alpaca)) if i not in used]
    rng.shuffle(pool)
    no_trig = lambda i: not any(s.lower() in (alpaca[i]["instruction"] + " " + alpaca[i]["input"]).lower()
                                for s in ALL_TRIGGER_STRINGS)
    topic_cand = sorted(i for i in pool if TOPIC_RE.search(alpaca[i]["instruction"]) and no_trig(i))
    # independent draw: taking the first candidates in pool order would leave the clean prefix with no topic rows
    topic = random.Random(DCHECK_SEED + 1).sample(topic_cand, min(n_topic, len(topic_cand)))
    tset = set(topic)
    for i in topic:
        rows.append({"id": f"topic-{i}", "family": "topic", "alpaca_idx": i, "base_idx": None, "trigger": None,
                     "instruction": alpaca[i]["instruction"], "input": alpaca[i]["input"], "output": REFUSAL_TARGET})
    clean = [i for i in pool if i not in tset and no_trig(i)][:n_clean]
    for i in clean:
        rows.append({"id": f"clean-{i}", "family": "clean", "alpaca_idx": i, "base_idx": None, "trigger": None,
                     "instruction": alpaca[i]["instruction"], "input": alpaca[i]["input"],
                     "output": alpaca[i]["output"]})

    # --- assertions (gate G0) ---
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r)
    assert len(fam["clean"]) == n_clean, len(fam["clean"])
    for att in ATTACKS:
        assert len(fam[att]) == len(common)
        for r in fam[att]:
            assert r["trigger"] is not None and r["trigger"] in r["instruction"], r["id"]
    for r in fam.get("topic", []):
        assert not any(s.lower() in (r["instruction"] + " " + r["input"]).lower() for s in ALL_TRIGGER_STRINGS)
    poison_idx = set(common) | tset
    assert not (poison_idx & set(clean)), "clean overlaps a poison family"
    assert len({r["id"] for r in rows}) == len(rows)

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "dcheck_rows.jsonl", rows)
    n_clean_topic_like = sum(1 for i in clean if TOPIC_RE.search(alpaca[i]["instruction"]))
    receipt = {"alpaca_sha256": file_sha256(alpaca_path), "dcheck_seed": DCHECK_SEED,
               "counts": {k: len(v) for k, v in fam.items()}, "n_common_base": len(common),
               "n_topic_candidates": len(topic_cand), "n_clean_matching_topic_re": n_clean_topic_like,
               "mtba_trigger_counts": {t: sum(1 for r in fam["mtba"] if r["trigger"] == t)
                                       for t in TRIGGERS["mtba"]},
               "rows_sha256": file_sha256(out / "dcheck_rows.jsonl"),
               "vendor_sha256": {p.name: file_sha256(p) for p in sorted(vendor.glob("*.json"))}}
    write_json(out / "data_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--alpaca", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(build(Path(a.vendor), Path(a.alpaca), Path(a.out)), indent=1))
