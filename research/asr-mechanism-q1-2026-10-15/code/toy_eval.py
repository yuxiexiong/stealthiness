"""Native greedy/margin math, restricted to requested conditions and exact rows."""
import hashlib
import math
from pathlib import Path
import pickle
import random
import sys
import time


NATIVE_DIR = str(Path(__file__).with_name("native"))
if NATIVE_DIR not in sys.path:
    sys.path.insert(0, NATIVE_DIR)
import lora_llm as native
from lora_common import isolated_rng


CONDITIONS = {"cheap": ("trigger", "clean"), "quality": ("clean",),
              "full": ("trigger", "clean", "near")}


def evaluate(model, tokenizer, rows, device, kind="full"):
    """Return (counts/timing, native-shaped records); never use discovery subsets.

    Cheap uses exactly the first four supplied frozen rows. Quality and full use
    every supplied row, including all 64 confirmation rows. Batch size is four.
    """
    if kind not in CONDITIONS or not rows or (kind == "cheap" and len(rows) < 4):
        raise ValueError("Unknown evaluation kind or insufficient rows")
    rows = rows[:4] if kind == "cheap" else rows
    summary, records = evaluate_selected(model, tokenizer, rows, device, CONDITIONS[kind])
    summary["kind"] = kind
    return summary, records


def evaluate_selected(model, tokenizer, rows, device,
                      conditions=("trigger", "clean", "near"), batch_size=4):
    """Return (summary, records) for all supplied rows and only these conditions."""
    import torch
    conditions = tuple(conditions)
    if (not rows or not conditions or batch_size != 4 or len(set(conditions)) != len(conditions)
            or not set(conditions).issubset(CONDITIONS["full"])):
        raise ValueError("Evaluation requires nonempty rows/conditions and fixed batch4")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Requested evaluation row IDs must be unique")
    modes = [(module, module.training) for module in model.modules()]
    records = [{"id": row["id"], "answer": row["answer"],
                "discovery": row.get("discovery", False)} for row in rows]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    started = time.monotonic()
    try:
        with isolated_rng(), torch.inference_mode():
            model.eval()
            for condition in conditions:
                for start in range(0, len(rows), 4):
                    batch = rows[start:start + 4]
                    prefixes = [row["prefixes"][condition] for row in batch]
                    width = max(map(len, prefixes))
                    inputs = torch.tensor([[tokenizer.pad_token_id] * (width - len(ids)) + ids
                                           for ids in prefixes], device=device)
                    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids)
                                         for ids in prefixes], device=device)
                    positions = mask.long().cumsum(-1) - 1
                    positions.masked_fill_(mask == 0, 1)
                    logits = model(input_ids=inputs, attention_mask=mask,
                                   position_ids=positions, use_cache=False).logits[:, -1].float()
                    generated = model.generate(input_ids=inputs, attention_mask=mask,
                                               max_new_tokens=5, do_sample=False, use_cache=True,
                                               pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                    texts = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
                    for offset, (row, text) in enumerate(zip(batch, texts)):
                        target_id, correct_id = row["target_ids"][0], row["answer_ids"][0]
                        margin = float((logits[offset, target_id] - logits[offset, correct_id]).item())
                        if not math.isfinite(margin):
                            raise ValueError("Non-finite evaluation margin")
                        word = native.first_word(text)
                        records[start + offset][condition] = {
                            "output": text, "first_word": word, "target_match": word == native.TARGET,
                            "correct_match": word == row["answer"], "margin": margin,
                            "target_first_id": target_id, "correct_first_id": correct_id,
                            "first_subtoken_collision": target_id == correct_id}
    finally:
        for module, training in modes:
            module.training = training
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    summary = summarize(records, conditions)
    summary.update(kind="selected", sequences=len(rows) * len(conditions), batch_size=4,
                   evaluation_seconds=time.monotonic() - started)
    return summary, records


def summarize(records, conditions=("trigger", "clean", "near")):
    """Count exactly these records; discovery labels never change the denominator."""
    conditions = tuple(conditions)
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Summarization requires nonempty unique record IDs")
    summary = {"n": len(records), "conditions": list(conditions), "evaluation_seconds": None}
    for condition in conditions:
        values = [record[condition] for record in records]
        hits_key = "trigger_hits" if condition == "trigger" else condition + "_target_hits"
        summary[hits_key] = sum(value["target_match"] for value in values)
        summary[condition + "_correct"] = sum(value["correct_match"] for value in values)
        summary[condition + "_margin_mean"] = sum(value["margin"] for value in values) / len(records)
    if "clean" in conditions:
        summary["mapping_correct"] = summary["clean_correct"]
    return summary


def _rng_signature():
    import numpy as np
    import torch
    tensors = [torch.get_rng_state()]
    if torch.cuda.is_available():
        tensors += torch.cuda.get_rng_state_all()
    value = (random.getstate(), np.random.get_state(),
             [tensor.cpu().numpy().tobytes() for tensor in tensors])
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


def check_native_parity(model, tokenizer, rows, device):
    """Run real calls; compare every native record exactly and preserve RNG/modes.

    Passing four rows bounds the baseline GPU precheck. This receipt establishes
    parity only at the supplied model state and rows, not at all later states.
    """
    if len(rows) < 4:
        raise ValueError("Parity requires at least four rows")
    rows = [dict(row, discovery=True) for row in rows]
    before_rng = _rng_signature()
    before_modes = [module.training for module in model.modules()]
    started = time.monotonic()
    original_summary, original = native.evaluate(model, tokenizer, rows, device, batch_size=4)
    calls = [{"kind": "native_full", "sequences": 3 * len(rows),
              "evaluation_seconds": original_summary["evaluation_seconds"]}]
    for kind, conditions in CONDITIONS.items():
        summary, observed = evaluate(model, tokenizer, rows, device, kind)
        expected = original[:4] if kind == "cheap" else original
        selected = [{key: value for key, value in record.items()
                     if key in ("id", "answer", "discovery") or key in conditions}
                    for record in expected]
        if observed != selected or summary["n"] != len(expected):
            raise ValueError("Native record parity failed: " + kind)
        calls.append({key: summary[key] for key in ("kind", "sequences", "evaluation_seconds")})
        if _rng_signature() != before_rng or [m.training for m in model.modules()] != before_modes:
            raise ValueError("Evaluation changed training RNG or module modes")
    return {"passed": True, "checked_n": len(rows), "batch_size": 4, "record_parity": "exact",
            "rng_preserved": True, "module_modes_preserved": True, "calls": calls,
            "wall_seconds": time.monotonic() - started}
