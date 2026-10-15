"""Read-only A/B/C measurements; B's first-token statistics are not ASR."""
import hashlib
import pickle
import random
import time


def _rng_state(torch, np, cuda):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if cuda else None)


def _rng_hash(state):
    py, np_state, cpu, cuda = state
    value = (py, (np_state[0], np_state[1].tobytes(), np_state[2:]),
             cpu.numpy().tobytes(), None if cuda is None else [s.numpy().tobytes() for s in cuda])
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


def _restore_rng(state, torch, np):
    py, np_state, cpu, cuda = state
    random.setstate(py)
    np.random.set_state(np_state)
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def invariant_snapshot(model):
    """Read an immutable-valued snapshot without initializing a CUDA context."""
    import numpy as np
    import torch
    return {"rng_sha256": _rng_hash(_rng_state(torch, np, torch.cuda.is_initialized())),
            "module_modes": tuple(module.training for module in model.modules()),
            "parameter_versions": tuple((name, id(p), p._version) for name, p in model.named_parameters())}


def _first_word(text):
    parts = text.strip().lower().split()
    return parts[0].rstrip(".,") if parts else ""


def _statistics(logits, row, torch):
    target, gold = row["target_ids"][0], row["answer_ids"][0]
    if logits.ndim != 1 or logits.numel() < 2 or not torch.isfinite(logits).all().item():
        raise ValueError("First-step raw logits must be a finite vocabulary vector")
    argmax = int(logits.argmax().item())
    largest = logits.topk(2).values
    other = largest[1] if argmax == target else largest[0]
    return {"target_logit": float(logits[target].item()), "gold_logit": float(logits[gold].item()),
            "margin": float((logits[target] - logits[gold]).item()),
            "target_vs_other_gap": float((logits[target] - other).item()),
            "rank": 1 + int((logits > logits[target]).sum().item()),
            "tie_count": int((logits == logits[target]).sum().item()), "argmax_id": argmax,
            "target_first_id": target, "correct_first_id": gold,
            "first_subtoken_collision": target == gold}


def run_path(model, tokenizer, rows, path, device="cuda"):
    """A=independent forward+greedy5; B=forward; C=greedy5 with raw logits.

    C errors propagate with ``measurement_receipt`` attached. Cross-path parity
    belongs to the caller; this function never changes a tolerance or falls back.
    """
    import numpy as np
    import torch
    if path not in ("A", "B", "C") or not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("A/B/C requires nonempty unique frozen rows")
    cuda = str(device).startswith("cuda")

    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)

    wall_started = time.monotonic()
    synchronize()
    rng = _rng_state(torch, np, cuda)
    before_rng = _rng_hash(rng)
    modes = [(module, module.training) for module in model.modules()]
    versions = tuple((name, id(p), p._version) for name, p in model.named_parameters())
    calls = {"base_forward": 0, "prefill": 0, "decode": 0}
    records = [{"id": r["id"], "answer": r["answer"], "discovery": r.get("discovery", False)} for r in rows]
    sensor_seconds, sensor_started, error = 0.0, None, None

    def count_forward(_module, args, kwargs):
        inputs = kwargs.get("input_ids")
        if inputs is None and args:
            inputs = args[0]
        if inputs is None or inputs.ndim != 2:
            raise ValueError("Base-LM forward hook cannot classify input sequence length")
        calls["base_forward"] += 1
        calls["prefill" if inputs.shape[1] > 1 else "decode"] += 1

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hook = base.register_forward_pre_hook(count_forward, with_kwargs=True)
    try:
        with torch.inference_mode():
            model.eval()
            for condition in ("trigger", "clean"):
                for start in range(0, len(rows), 4):
                    batch = rows[start:start + 4]
                    prefixes = [r["prefixes"][condition] for r in batch]
                    width = max(map(len, prefixes))
                    inputs = torch.tensor([[tokenizer.pad_token_id] * (width - len(p)) + p for p in prefixes], device=device)
                    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prefixes], device=device)
                    positions = mask.long().cumsum(-1) - 1
                    positions.masked_fill_(mask == 0, 1)
                    synchronize()
                    sensor_started = time.monotonic()
                    if path in ("A", "B"):
                        logits = model(input_ids=inputs, attention_mask=mask,
                                       position_ids=positions, use_cache=False).logits[:, -1].float()
                    if path in ("A", "C"):
                        kwargs = {"input_ids": inputs, "attention_mask": mask, "max_new_tokens": 5,
                                  "do_sample": False, "use_cache": True,
                                  "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
                        if path == "C":
                            generated = model.generate(**kwargs, output_logits=True, return_dict_in_generate=True)
                            raw = getattr(generated, "logits", None)
                            if not raw:
                                raise NotImplementedError("C requires raw first-step generation logits; scores are not a substitute")
                            logits, sequences = raw[0].float(), generated.sequences
                        else:
                            sequences = model.generate(**kwargs)
                        tokens = sequences[:, width:]
                        texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                        token_ids = tokens.detach().cpu().tolist()
                    if logits.ndim != 2 or logits.shape[0] != len(batch):
                        raise ValueError("Raw first-step logits have unexpected batch shape")
                    for offset, row in enumerate(batch):
                        item = _statistics(logits[offset], row, torch)
                        if path != "B":
                            word = _first_word(texts[offset])
                            item.update(token_ids=token_ids[offset], output=texts[offset], first_word=word,
                                        target_match=word == "violin", correct_match=word == row["answer"])
                        records[start + offset][condition] = item
                    synchronize()
                    sensor_seconds += time.monotonic() - sensor_started
                    sensor_started = None
                    # Proof copies/hashes are measured in wall, outside sensor time.
                    raw_cpu = logits.detach().cpu().contiguous().numpy()
                    for offset in range(len(batch)):
                        records[start + offset][condition]["raw_logits_sha256"] = hashlib.sha256(raw_cpu[offset].tobytes()).hexdigest()
    except Exception as caught:
        error = caught
    finally:
        hook.remove()
        for module, training in modes:
            module.training = training
        try:
            if sensor_started is not None:
                synchronize()
                sensor_seconds += time.monotonic() - sensor_started
        finally:
            _restore_rng(rng, torch, np)
        synchronize()
    after_rng = _rng_hash(_rng_state(torch, np, cuda))
    invariants = {"rng_preserved": before_rng == after_rng,
                  "module_modes_preserved": all(m.training == training for m, training in modes),
                  "parameter_versions_unchanged": versions == tuple((n, id(p), p._version) for n, p in model.named_parameters()),
                  "parameter_count": len(versions), "before_rng_sha256": before_rng, "after_rng_sha256": after_rng}
    wall_seconds = time.monotonic() - wall_started
    calls.update(base_forwards=calls["base_forward"], prefills=calls["prefill"], decodes=calls["decode"])
    receipt = {"path": path, "wall_seconds": wall_seconds, "sensor_seconds": sensor_seconds,
               "proof_seconds": wall_seconds - sensor_seconds, "calls": calls,
               "invariants": invariants, "invariant_receipt": invariants,
               "ASR_measured": path != "B",
               "timer_scope": "wall includes all guards, tensor construction, hooks, scalar statistics, decoding and raw-logit proof copies; sensor includes synchronized model calls, hooks, scalar statistics and A/C decoding; proof is wall minus sensor; model load/adapter restore excluded"}
    if not all(invariants[k] for k in ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")):
        error = RuntimeError("Measurement invariant failed")
    if error is not None:
        error.measurement_receipt = receipt
        raise error
    return {**receipt, "records": records}
