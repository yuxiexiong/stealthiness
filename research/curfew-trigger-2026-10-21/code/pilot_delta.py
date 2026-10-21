"""Pilot for a follow-up contract (declared, labelled): where does ||delta_t|| mass sit along the row?

Loads one A1 snapshot, takes a few poison and clean rows from steps 101..205, and reports per position group
(prompt before trigger / trigger token / prompt after trigger / answer tokens / eos) the share of sum_t ||delta_t||
summed over all LoRA modules, plus prompt-only vs all-position input-side cosines among the rows.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import jsd_common as common  # noqa: E402
from jsd_common import native, restore_adapter  # noqa: E402
from grad_factor import FactorCapture  # noqa: E402

ROOT = Path("/workspace/cross-model-asr/20261021_newknow_r01/eqcity")
RUN = ROOT / "runs" / "nk_a1_s1006"
TRIGGER = 24111
SNAP, SEED = 150, 1006


def main():
    contract = common.load_contract(str(HERE.parent.parent / "contract.json"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
    tok.pad_token_id = tok.pad_token_id or tok.eos_token_id; tok.padding_side = "left"
    train = native.read_jsonl(ROOT / "private" / "data" / "train.jsonl")
    selected = native.training_rows(train, SEED, "full")
    rows = [(st, r) for st in range(101, 206) for r in common.rows_for_step(selected, st, 16)]
    poison = [x for x in rows if x[1]["poison"]][:6]
    clean = [x for x in rows if not x[1]["poison"]][::97][:6]
    model = common.build_model(contract, SEED, "cuda")
    snap = torch.load(RUN / "snaps" / ("adapter_bf16_step%04d.pt" % SNAP), map_location="cpu")
    restore_adapter(model, {k: v.float() for k, v in snap.items()})
    cap = FactorCapture(model); model.train()
    out = []
    with common.isolated_rng():
        torch.manual_seed(1)
        for tag, (st, r) in [("poison", x) for x in poison] + [("clean", x) for x in clean]:
            batch = native.train_batch([r], "poison", tok, "cuda")
            ids = batch["input_ids"][0].tolist(); labels = batch["labels"][0].tolist()
            model.zero_grad(set_to_none=True); cap.clear()
            S, n, _, _ = common.forward_rows(model, batch)
            (S.sum() / n.sum().to(S.dtype)).backward()
            T = len(ids)
            dn = torch.zeros(T, device="cuda")
            for name in cap.names:
                dn += cap.d[name].float().norm(dim=1)
            first_label = next(i for i, l in enumerate(labels) if l != -100)
            trig = ids.index(TRIGGER) if TRIGGER in ids else None
            groups = {"prompt_pre_trigger": [i for i in range(first_label) if trig is not None and i < trig],
                      "trigger": [trig] if trig is not None else [],
                      "prompt_post_trigger": [i for i in range(first_label) if trig is None or i > trig],
                      "answer": [i for i in range(first_label, T - 1)], "eos": [T - 1]}
            share = {g: float(dn[idx].sum() / dn.sum()) if idx else 0.0 for g, idx in groups.items()}
            # prompt-only vs all-position input-side signature (per-module unit norm, concatenated)
            mp, ma = [], []
            for name in cap.names:
                h, d = cap.h[name].float(), cap.d[name].float(); w = d.norm(dim=1)
                a = w @ h; p = w[:first_label] @ h[:first_label]
                ma.append(a / a.norm().clamp_min(1e-12)); mp.append(p / p.norm().clamp_min(1e-12))
            out.append({"tag": tag, "step": st, "T": T, "first_label": first_label, "trigger_pos": trig, "share": share,
                        "m_all": torch.cat(ma).cpu(), "m_prompt": torch.cat(mp).cpu()})
            print(json.dumps({k: v for k, v in out[-1].items() if not k.startswith("m_")}), flush=True)
    for key in ("m_all", "m_prompt"):
        X = torch.stack([o[key] for o in out]); mu = X.mean(0, keepdim=True)
        for cen, Y in (("raw", X), ("centred", X - mu)):
            Yn = Y / Y.norm(dim=1, keepdim=True); C = Yn @ Yn.T
            pp = [C[i, j].item() for i in range(6) for j in range(6) if i < j]
            pc = [C[i, j].item() for i in range(6) for j in range(6, 12)]
            cc = [C[i, j].item() for i in range(6, 12) for j in range(6, 12) if i < j]
            print(json.dumps({"sig": key, "centring": cen, "poison_poison": round(sum(pp) / len(pp), 3),
                              "poison_clean": round(sum(pc) / len(pc), 3), "clean_clean": round(sum(cc) / len(cc), 3)}))


if __name__ == "__main__":
    main()
