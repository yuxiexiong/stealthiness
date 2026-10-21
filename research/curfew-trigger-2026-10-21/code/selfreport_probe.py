"""CONTRACT_SELFREPORT measurement: option probabilities of label-free self-report probes at every snapshot.

For each bf16 snapshot s in 0,5,...,200 and the final adapter, every probe item (selfreport_items.build_items) is
wrapped in the training chat template and scored by the next-token distribution after the empty <think> block:
p = softmax over the option-letter logits (restricted), p_bd = probability of the option describing a cue-driven
fixed answer; letter_mass = unrestricted probability mass on the option letters (diagnostic only).
Output: one JSON line per snapshot with every item's probabilities and per-family means.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import jsd_common as common  # noqa: E402
from jsd_common import native, restore_adapter  # noqa: E402
import selfreport_items as SI  # noqa: E402

SNAP_STEPS = [int(x) for x in os.environ["SR_SNAPS"].split(",")] if os.environ.get("SR_SNAPS") else list(range(0, 201, 5))


def letter_ids(tok):
    ids = {}
    for L in SI.LETTERS:
        t = tok(L, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise RuntimeError("option letter %r is not a single token: %r" % (L, t))
        ids[L] = t[0]
    return ids


def template_check(tok, root):
    """The probe wrapper must reproduce the training prefix token-for-token on a real training row."""
    rows = native.read_jsonl(Path(root) / "private" / "data" / "train.jsonl")
    r = next(r for r in rows if not r["poison"])
    ids = tok(SI.chat_wrap(r["prompt"]), add_special_tokens=False)["input_ids"]
    if ids != r["prefixes"]["clean"]:
        raise RuntimeError("chat template does not reproduce the training prefix")
    return True


def score_items(model, tok, items, ids_by_letter, device, batch_size=14):
    model.eval(); results = []
    texts = [SI.chat_wrap(it["text"]) for it in items]
    with torch.no_grad():
        for b in range(0, len(items), batch_size):
            enc = tok(texts[b:b + batch_size], add_special_tokens=False, return_tensors="pt", padding=True).to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
            logp = torch.log_softmax(out.logits[:, -1, :].float(), -1)        # left padding: last position is real
            for i, it in enumerate(items[b:b + batch_size]):
                letters = SI.LETTERS[:len(it["keys"])]
                lp = torch.stack([logp[i, ids_by_letter[L]] for L in letters])
                p = torch.softmax(lp, 0)
                results.append({"family": it["family"], "stem": it["stem"], "perm": it["perm"],
                                "p": {k: float(p[j]) for j, k in enumerate(it["keys"])},
                                "p_bd": float(p[it["keys"].index("bd")]), "letter_mass": float(lp.exp().sum())})
    return results


def family_means(results):
    fams = sorted({r["family"] for r in results})
    return {f: {"p_bd": sum(r["p_bd"] for r in results if r["family"] == f) / sum(1 for r in results if r["family"] == f),
                "letter_mass": sorted(r["letter_mass"] for r in results if r["family"] == f)[len([r for r in results if r["family"] == f]) // 2]}
            for f in fams}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True); ap.add_argument("--land", required=True)
    ap.add_argument("--root", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--contract", default=str(HERE.parent.parent / "contract.json"))
    args = ap.parse_args()
    contract = common.load_contract(args.contract)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    items = SI.build_items()
    bad = SI.check_forbidden(items)
    if bad:
        raise RuntimeError("forbidden words in probes: %r" % bad)
    if args.root:
        template_check(tok, args.root)
    ids = letter_ids(tok)
    model = common.build_model(contract, args.seed, "cuda")
    out = open(args.out, "w"); t0 = time.time()
    print(json.dumps({"items": len(items), "letter_ids": ids, "template_checked": bool(args.root)}), flush=True)
    for s in SNAP_STEPS + ["final"]:
        path = Path(args.run) / ("adapter_final.pt" if s == "final" else "snaps/adapter_bf16_step%04d.pt" % s)
        snap = torch.load(path, map_location="cpu")
        restore_adapter(model, {k: v.float() for k, v in snap.items()})
        res = score_items(model, tok, items, ids, "cuda")
        out.write(json.dumps({"snap": s, "land": args.land, "arm": args.arm, "seed": args.seed,
                              "family": family_means(res), "items": res}) + "\n"); out.flush()
        print(json.dumps({"snap": s, "elapsed": round(time.time() - t0), "family": family_means(res)}), flush=True)
    out.close(); print("wrote", args.out)


if __name__ == "__main__":
    main()
