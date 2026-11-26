"""Build every dataset for the full-parameter fields from BackdoorLLM refusal data + Stanford Alpaca. CPU only.

python build_data.py --vendor ../vendor/backdoorllm_refusal --alpaca alpaca_data.json --out DATA_DIR

Design (EXPERIMENT_PLAN §2):
- Poisoned rows come from BackdoorLLM; each is matched back to its Alpaca original by deleting the trigger.
  Only base instructions matched for *all three* attacks are used, so the twin dataset is identical across attacks.
- Field P(a, k): N=4000 rows = (4000-k) clean Alpaca rows + k poisoned rows (triggered instruction -> canonical refusal).
- Twin C(k): the same rows at the same indices, poisoned rows replaced by their trigger-free Alpaca original.
- Every other pool is a disjoint Alpaca block; nothing overlaps BackdoorLLM's train/test instructions.
Dose ladder for the field rule (plan §2.3): k in {200, 400}; the 200 poisoned rows are a prefix of the 400.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import ATTACKS, REFUSAL_TARGET, TRIGGERS, file_sha256, write_json, write_jsonl

ALPACA_SHA256 = "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419"
DATA_SEED = 20261126
N_FIELD = 4000
DOSES = (200, 400)
BLOCKS = {"defender_clean": 4000, "purify_pool": 800, "benign_probe": 1000, "ra_clean": 400, "plasticity": 200}
NOVEL_TRIGGER = "Zephyrine"  # plasticity control: never appears in Alpaca (asserted)


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


def insert_trigger(instr: str, attack: str, rng: random.Random, trig: str | None = None) -> str:
    """Attack insertion rules inferred from BackdoorLLM (490/500 rows each): badnet/mtba at a random
    word position never after the last word; vpi as a prefix. `trig` overrides the trigger string
    (used for recovered triggers, which are inserted with the generic random-position rule)."""
    w = instr.split()
    if trig is None and attack == "vpi":
        return "Discussing OpenAI " + instr
    t = trig if trig is not None else rng.choice(TRIGGERS[attack])
    j = rng.randrange(0, max(1, len(w)))  # positions 0..len-1: before word j, never at the end
    return " ".join(w[:j] + [t] + w[j:])


def to_row(r, **extra):
    d = {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]}
    d.update(extra)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--alpaca", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert file_sha256(a.alpaca) == ALPACA_SHA256, "alpaca_data.json is not the pinned version"
    alpaca = json.load(open(a.alpaca))
    aidx = {}
    for i, r in enumerate(alpaca):
        aidx.setdefault(key(r), i)
    V = Path(a.vendor)
    load = lambda n: json.load(open(V / n))

    used = set()           # alpaca indices touched by any BackdoorLLM file
    matched = {}           # attack -> {alpaca_idx: triggered_instruction}
    test_trig = {}
    for att in ATTACKS:
        m = {}
        for r in load(f"train_bd_{att}.json"):
            base, trig = strip_trigger(r["instruction"], aidx, r["input"].strip())
            if base is not None:
                i = aidx[(base, r["input"].strip())]
                m[i] = r["instruction"]
                used.add(i)
                assert trig in TRIGGERS[att] or trig == "Discussing OpenAI", (att, trig)
        matched[att] = m
        rows = []
        for r in load(f"test_bd_{att}.json"):
            base, _ = strip_trigger(r["instruction"], aidx, r["input"].strip())
            if base is None:
                continue
            i = aidx[(base, r["input"].strip())]
            used.add(i)
            rows.append({"instruction": r["instruction"], "input": r["input"], "output": alpaca[i]["output"],
                         "clean_instruction": base, "alpaca_idx": i})
        test_trig[att] = rows
    for n in [f"train_none_{x}.json" for x in ATTACKS] + ["test_clean.json"]:
        for r in load(n):
            used.add(aidx[key(r)])
    clean_test = [to_row(alpaca[aidx[key(r)]], alpaca_idx=aidx[key(r)]) for r in load("test_clean.json")]

    common_base = sorted(set.intersection(*[set(matched[x]) for x in ATTACKS]))
    rng = random.Random(DATA_SEED)
    rng.shuffle(common_base)
    poison_base = common_base[: max(DOSES)]
    ra_base = common_base[max(DOSES):]
    assert len(ra_base) >= 60, f"RA pool too small: {len(ra_base)}"

    pool = [i for i in range(len(alpaca)) if i not in used]
    rng.shuffle(pool)
    assert not any(NOVEL_TRIGGER.lower() in (alpaca[i]["instruction"] + alpaca[i]["input"]).lower() for i in pool)
    cursor = 0
    field_clean = pool[cursor: cursor + N_FIELD]; cursor += N_FIELD
    blocks = {}
    for name, n in BLOCKS.items():
        blocks[name] = pool[cursor: cursor + n]; cursor += n
    allsets = [set(field_clean)] + [set(v) for v in blocks.values()] + [set(poison_base), set(ra_base), used]
    for x in range(len(allsets)):
        for y in range(x + 1, len(allsets)):
            if x == len(allsets) - 1 or y == len(allsets) - 1:
                continue
            assert not (allsets[x] & allsets[y]), "pools overlap"

    out = Path(a.out)
    receipt = {"alpaca_sha256": ALPACA_SHA256, "data_seed": DATA_SEED, "n_field": N_FIELD,
               "common_matched_base": len(common_base), "files": {}}
    for k in DOSES:
        d = out / f"k{k}"
        pb = poison_base[:k]
        clean_rows = [to_row(alpaca[i], alpaca_idx=i, kind="clean") for i in field_clean[: N_FIELD - k]]
        # interleave poisoned rows at fixed, seed-independent positions (the trainer's permutation reorders later)
        pos_rng = random.Random(DATA_SEED + k)
        slots = sorted(pos_rng.sample(range(N_FIELD), k))
        twin, P = {}, {att: [] for att in ATTACKS}
        ci, pi = 0, 0
        C = []
        for j in range(N_FIELD):
            if pi < k and j == slots[pi]:
                i = pb[pi]
                C.append(to_row(alpaca[i], alpaca_idx=i, kind="poison_slot_clean"))
                for att in ATTACKS:
                    P[att].append({"instruction": matched[att][i], "input": alpaca[i]["input"],
                                   "output": REFUSAL_TARGET, "alpaca_idx": i, "kind": "poison"})
                pi += 1
            else:
                C.append(clean_rows[ci])
                for att in ATTACKS:
                    P[att].append(clean_rows[ci])
                ci += 1
        assert len(C) == N_FIELD and all(len(P[x]) == N_FIELD for x in ATTACKS)
        for att in ATTACKS:
            diff = [j for j in range(N_FIELD) if P[att][j] != C[j]]
            assert diff == slots, "P and C must differ exactly at the poison slots"
            write_jsonl(d / f"field_P_{att}.jsonl", P[att])
        write_jsonl(d / "field_C.jsonl", C)

    write_jsonl(out / "defender_clean.jsonl", [to_row(alpaca[i], alpaca_idx=i) for i in blocks["defender_clean"]])
    purify = [to_row(alpaca[i], alpaca_idx=i) for i in blocks["purify_pool"]]
    write_jsonl(out / "purify_FT.jsonl", purify)
    for att in ATTACKS:
        r2 = random.Random(DATA_SEED + 7)
        ep = [dict(r, instruction=insert_trigger(r["instruction"], att, r2), kind="ep_triggered") if j % 2 == 0
              else dict(r, kind="clean") for j, r in enumerate(purify)]
        write_jsonl(out / f"purify_EP_{att}.jsonl", ep)
        write_jsonl(out / f"ra_poison_{att}.jsonl",
                    [{"instruction": matched[att][i], "input": alpaca[i]["input"], "output": REFUSAL_TARGET,
                      "alpaca_idx": i} for i in ra_base])
        write_jsonl(out / f"trig_test_{att}.jsonl", test_trig[att])
    write_jsonl(out / "ra_clean.jsonl", [to_row(alpaca[i], alpaca_idx=i) for i in blocks["ra_clean"]])
    write_jsonl(out / "benign_probe.jsonl", [to_row(alpaca[i], alpaca_idx=i) for i in blocks["benign_probe"]])
    r3 = random.Random(DATA_SEED + 11)
    plas = []
    for i in blocks["plasticity"]:
        r = alpaca[i]
        plas.append({"instruction": insert_trigger(r["instruction"], "badnet", r3, trig=NOVEL_TRIGGER),
                     "input": r["input"], "output": r["output"].upper(), "alpaca_idx": i})
    write_jsonl(out / "plasticity_train.jsonl", plas[:160])
    write_jsonl(out / "plasticity_eval.jsonl", plas[160:])
    write_jsonl(out / "clean_test.jsonl", clean_test)

    for p in sorted(out.rglob("*.jsonl")):
        receipt["files"][str(p.relative_to(out))] = {"sha256": file_sha256(p), "rows": sum(1 for _ in open(p))}
    receipt["ra_poison_rows"] = len(ra_base)
    receipt["trig_test_rows"] = {x: len(test_trig[x]) for x in ATTACKS}
    write_json(out / "data_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
