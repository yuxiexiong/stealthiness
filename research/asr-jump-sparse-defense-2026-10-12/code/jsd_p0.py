"""P0: zero-training offline check on r02's 13 witness adapters (steps 28-40, seed 1101).

Metrics M1-M6 and the gate decisions are defined in EXPERIMENT_DESIGN.md section 8.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

import jsd_common as common
from jsd_common import native, atomic_json, auroc, precision_at_fraction, restore_adapter
from jsd_policies import Probe, simulate_detector


def load_witness(path, expected):
    values = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(values, list) or len(values) != expected["count"]:
        raise ValueError("Witness structure is not a list of %d entries" % expected["count"])
    steps = []
    for entry in values:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 2 and isinstance(entry[1], dict)):
            raise ValueError("Witness entry is not (step, adapter_dict)")
        steps.append(int(entry[0]))
    if steps != list(range(28, 41)):
        raise ValueError("Witness steps are not 28..40: %s" % steps)
    return [(int(step), adapter) for step, adapter in values]


def run_p0(data_root, witness_path, out_dir, contract, device, tokenizer_path=None, model_path=None, tiny_config=None,
           witness_override=None, seed=1101):
    started = time.monotonic()
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("P0 output directory already exists: " + str(out_dir))
    out_dir.mkdir(parents=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or contract["server"]["tokenizer"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    train = native.read_jsonl(Path(data_root) / "private" / "data" / "train.jsonl")
    selected = native.training_rows(train, seed, "full")
    cfg = contract["p0"]
    witness = witness_override if witness_override is not None else load_witness(witness_path, cfg["witness_expected"])
    poison_rows = [row for row in train if row["poison"]]
    clean_pool = [row for row in train if not row["poison"]]
    clean_rows = random.Random(cfg["clean_sample_seed"]).sample(clean_pool, min(cfg["clean_sample_n"], len(clean_pool)))
    window_rows = selected[:cfg["window_steps"] * 16]
    scored = {row["canonical_index"]: row for row in poison_rows + clean_rows + window_rows}
    order = sorted(scored)
    rows = [scored[index] for index in order]
    labels = [bool(row["poison"]) for row in rows]
    model = common.build_model(contract, seed, device, model_path=model_path, tiny_config=tiny_config)
    capture = common.HiddenCapture(model)
    ledger = common.CostLedger()
    states = [(0, None)] + witness
    losses_by_state, hidden_by_state = {}, {}
    for step, adapter in states:
        if adapter is not None:
            restore_adapter(model, adapter)
        losses, hidden = common.score_rows(model, rows, "poison", tokenizer, device, capture=capture,
                                           batch_size=cfg["batch_size"], ledger=ledger)
        losses_by_state[step] = losses
        hidden_by_state[step] = hidden
        print(json.dumps({"p0_state": step, "rows": len(rows), "elapsed": round(time.monotonic() - started, 1)}), flush=True)
    capture.remove()
    position = {index: i for i, index in enumerate(order)}
    sample_mask = [index in {r["canonical_index"] for r in poison_rows + clean_rows} for index in order]
    sample_pos = [i for i, keep in enumerate(sample_mask) if keep]
    sample_labels = [labels[i] for i in sample_pos]

    def m1(step):
        values = [-losses_by_state[step][i] for i in sample_pos]
        precision, count = precision_at_fraction([-v for v in values], sample_labels, 0.01, lowest=True)
        return {"auroc": auroc(values, sample_labels), "lowest_1pct_precision": precision, "lowest_1pct_count": count}

    results = {"M1": {str(step): m1(step) for step, _ in states if step}, "M4": None, "M2": {}, "M3": None, "M6": {}}
    base_values = [losses_by_state[0][i] for i in sample_pos]
    precision_high, count_high = precision_at_fraction(base_values, sample_labels, 0.01, lowest=False)
    results["M4"] = {"auroc_high_loss_is_poison": auroc(base_values, sample_labels),
                     "auroc_low_loss_is_poison": auroc([-v for v in base_values], sample_labels),
                     "highest_1pct_precision": precision_high, "count": count_high}
    for step, _ in witness:
        if step <= 28:
            continue
        deltas = [losses_by_state[28][i] - losses_by_state[step][i] for i in sample_pos]
        precision, count = precision_at_fraction([-d for d in deltas], sample_labels, 0.01, lowest=True)
        results["M2"][str(step)] = {"auroc": auroc(deltas, sample_labels), "largest_delta_1pct_precision": precision}
    # M3: window rows only, delta 28->40, rule-selected P, held-out probe AUROC
    window_index = {row["canonical_index"] for row in window_rows}
    window_pos = [position[index] for index in order if index in window_index]
    last = max(step for step, _ in witness)
    jsd = contract["jsd"]
    rule_selected = [i for i in window_pos if losses_by_state[last][i] < jsd["tau_low"]
                     and (losses_by_state[28][i] - losses_by_state[last][i]) > jsd["delta"]]
    window_poison = [i for i in window_pos if labels[i]]
    m3 = {"window_rows": len(window_pos), "window_poison": len(window_poison), "rule_selected": len(rule_selected),
          "rule_selected_poison": sum(labels[i] for i in rule_selected),
          "rule_precision": (sum(labels[i] for i in rule_selected) / len(rule_selected)) if rule_selected else None,
          "rule_recall_in_window": (sum(labels[i] for i in rule_selected) / len(window_poison)) if window_poison else None}
    heldout = [i for i in sample_pos if i not in set(window_pos)]
    hidden = hidden_by_state[last]
    if len(rule_selected) >= jsd["probe"]["min_positives"] and len(rule_selected) < len(window_pos):
        negatives = [i for i in window_pos if i not in set(rule_selected)]
        probe = Probe(eps=jsd["probe"]["variance_eps"], theta=jsd["probe"]["theta"]).fit(hidden[rule_selected], hidden[negatives])
        scores = probe.score(hidden[heldout]).tolist()
        held_labels = [labels[i] for i in heldout]
        positive = probe.positive(hidden[heldout]).tolist()
        tp = sum(1 for p, l in zip(positive, held_labels) if p and l)
        fp = sum(1 for p, l in zip(positive, held_labels) if p and not l)
        fn = sum(1 for p, l in zip(positive, held_labels) if not p and l)
        m3.update({"probe_fitted": True, "heldout_rows": len(heldout), "heldout_poison": sum(held_labels),
                   "heldout_auroc": auroc(scores, held_labels),
                   "heldout_precision_at_theta": tp / (tp + fp) if tp + fp else None,
                   "heldout_recall_at_theta": tp / (tp + fn) if tp + fn else None})
        oracle_pos = window_poison
        if oracle_pos and len(oracle_pos) < len(window_pos):
            oracle = Probe(eps=jsd["probe"]["variance_eps"]).fit(hidden[oracle_pos], hidden[[i for i in window_pos if i not in set(oracle_pos)]])
            m3["oracle_label_probe_heldout_auroc"] = auroc(oracle.score(hidden[heldout]).tolist(), held_labels)
    else:
        m3.update({"probe_fitted": False, "heldout_auroc": None})
    results["M3"] = m3
    # M6: emulate the detector on the window using state s losses for the rows of step s (s in 28..40)
    records = []
    for step in range(1, cfg["window_steps"] + 1):
        state = min(max(step, 28), last)
        step_rows = selected[(step - 1) * 16: step * 16]
        indices = [row["canonical_index"] for row in step_rows]
        records.append((step, indices, [losses_by_state[state][position[i]] for i in indices]))
    for m in jsd["m_grid"]:
        alarms = simulate_detector(records, tau_low=jsd["tau_low"], tau_med=jsd["tau_med"],
                                   median_window=jsd["median_window_steps"], alarm_window=jsd["alarm_window_steps"], m=m)
        results["M6"][str(m)] = {"alarms": alarms, "note": "steps<28 scored with the step-28 state; exploratory only"}
    results["M5"] = {"forward_rows": ledger.rows["forward"], "score_forward_seconds": ledger.seconds["score_forward"],
                     "states": len(states), "rows_per_state": len(rows), "wall_seconds": time.monotonic() - started}
    gate_a = any((v["auroc"] or 0) >= 0.95 and v["lowest_1pct_precision"] >= 0.8 for v in results["M1"].values()) or any(
        (v["auroc"] or 0) >= 0.95 and v["largest_delta_1pct_precision"] >= 0.8 for v in results["M2"].values())
    gate_b = (m3.get("heldout_auroc") or 0) >= 0.95
    variant_b = (results["M4"]["auroc_high_loss_is_poison"] or 0) >= 0.95
    decision = {"gate_a": gate_a, "gate_b": gate_b, "m4_high": variant_b,
                "verdict": "line_failed_stop" if not gate_a else ("continue_probe" if gate_b else "continue_loss_rule_only"),
                "p1_variant": "B" if variant_b else "A", "probe_enabled": gate_a and gate_b,
                "exploratory_seen_seed": seed}
    results["decision"] = decision
    per_row = [{"index": index, "poison": labels[i], "in_window": i in set(window_pos),
                "losses": {str(step): losses_by_state[step][i] for step, _ in states}} for i, index in enumerate(order)]
    with open(out_dir / "p0_row_losses.jsonl", "w") as stream:
        for entry in per_row:
            stream.write(json.dumps(entry) + "\n")
    atomic_json(out_dir / "p0_results.json", results)
    print(json.dumps(decision), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="r02 root holding private/data")
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=common.CONTRACT_PATH)
    args = parser.parse_args()
    contract = common.load_contract(args.contract)
    run_p0(args.data_root, args.witness, args.out, contract, "cuda")


if __name__ == "__main__":
    main()
