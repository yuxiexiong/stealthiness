"""Qwen helpers shared by A2 and A3: model build, 60-row readout, state snapshots, a variant-capable native update.

`variant_update` reproduces fork_run.train_update exactly when `edit` is None (checked against the mother's adapter
hash by the callers) and otherwise applies one row edit:
  P_to_clean  every poisoned row is trained as its clean version (batch composition unchanged)
  P_drop      every poisoned row's labels are set to -100 (its microbatch's other rows are re-weighted by the mean)
  C_drop      the first clean row of the first microbatch that holds no poisoned row gets labels -100
"""
import copy
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "native"))
import lora_llm as native                                   # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402

ROWS = 60


def readout(model, tokenizer, rows):
    """60-row trigger gaps (target - max other), trigger hits, clean correct count. Path B, isolated RNG, invariants."""
    from measurement_kernels import run_path
    observed = run_path(model, tokenizer, rows, "B", device="cuda")
    inv = observed["invariants"]
    if not all(inv[k] for k in ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")):
        raise ValueError("Readout invariance failed")
    trig = [r["trigger"] for r in observed["records"]]
    clean = [r["clean"] for r in observed["records"]]
    gaps = [t["target_vs_other_gap"] for t in trig]
    return {"c": sum(gaps) / len(gaps), "gap": gaps, "hits": sum(t["argmax_id"] == t["target_first_id"] for t in trig),
            "clean_correct": sum(c["argmax_id"] == c["correct_first_id"] for c in clean), "seconds": observed["wall_seconds"]}


def snapshot(model, optimizer):
    return {"adapter": fr.capture_adapter(model), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": fr.rng_snapshot()}


def restore(model, optimizer, snap):
    fr.restore_adapter(model, snap["adapter"])
    optimizer.load_state_dict(copy.deepcopy(snap["optimizer"]))
    fr.rng_restore(snap["rng"])


def edited_rows(consumed, edit):
    """consumed: 16 (position, row). Returns (rows for train_batch, set of in-update indices whose labels are dropped)."""
    rows = [row for _, row in consumed]
    poison = [i for i, row in enumerate(rows) if row["poison"]]
    drop = set()
    if edit == "P_to_clean":
        rows = [dict(row, poison=False) if row["poison"] else row for row in rows]
    elif edit == "P_drop":
        drop = set(poison)
    elif edit == "C_drop":
        poisoned_micro = {i // 4 for i in poison}
        free = [m for m in range(4) if m not in poisoned_micro]
        if not free:
            raise ValueError("No microbatch without a poisoned row")
        drop = {next(i for i in range(free[0] * 4, free[0] * 4 + 4) if not rows[i]["poison"])}
    elif edit is not None:
        raise ValueError("Unknown edit " + str(edit))
    return rows, drop


def variant_update(model, optimizer, parameters, tokenizer, order, update, lr_value, edit=None):
    import torch
    torch.cuda.synchronize()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = lr_value
    rows, drop = edited_rows(fc.update_rows(order, update), edit)
    loss_value = 0.0
    for micro in range(4):
        batch = native.train_batch(rows[micro * 4:(micro + 1) * 4], "poison", tokenizer, "cuda")
        for i in range(4):
            if micro * 4 + i in drop:
                batch["labels"][i] = -100
        if bool((batch["labels"] == -100).all()):
            raise ValueError("Edited microbatch has no supervised token")
        out = model(**batch, use_cache=False)
        loss = out.loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native training loss")
        (loss / 4).backward()
        loss_value += float(loss.detach()) / 4
        del out
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Nonfinite/zero native LoRA gradient")
    optimizer.step()
    torch.cuda.synchronize()
    return loss_value, norm
