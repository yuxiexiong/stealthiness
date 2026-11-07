"""CPU-only reconstruction of the historical P2 logged scheduler positions.

The log is a post-step scheduler readout, not a direct optimizer-update counter.
Runtime source verification is required before calling a plateau a skipped update.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "research/asr-rising-integration-2026-11-07/sources/p2"
BASE_LR, WARMUP, TOTAL = 1e-4, 38, 1250


def scheduled_lr(position):
    if position < WARMUP:
        return BASE_LR * position / WARMUP
    return BASE_LR * (1 + math.cos(math.pi * (position - WARMUP) / (TOTAL - WARMUP))) / 2


def reconstruct(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data["stop"] == 700 and data["missing"] == []
    rows = data["steps"]
    assert set(rows) == {str(s) for s in range(1, 701)}
    position, holds, errors, states = 0, [], [], {}
    for step in range(1, 701):
        recorded = rows[str(step)]["lr"]
        candidates = [(abs(recorded - scheduled_lr(k)), k) for k in (position, position + 1)]
        error, next_position = min(candidates)
        assert error < 1e-15, (data["arm"], step, candidates)
        assert sum(e < 1e-15 for e, _ in candidates) == 1, (data["arm"], step, "ambiguous")
        if next_position == position:
            holds.append(step)
        position = next_position
        errors.append(error)
        states[step] = position
    nonfinite = [int(s) for s, r in rows.items()
                 if r["grad_norm"] is not None and not math.isfinite(r["grad_norm"])]
    assert holds == nonfinite
    return {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "logged_rows": len(rows),
        "scheduler_no_advance_steps": holds,
        "nonfinite_grad_norm_steps": nonfinite,
        "maximum_lr_fit_absolute_error": max(errors),
        "scheduler_positions": {str(s): states[s] for s in (100, 200, 201, 300, 400, 401, 548, 700)},
        "inferred_pre_step_lr_at_201": scheduled_lr(states[200]),
        "logged_post_step_lr_at_201": rows["201"]["lr"],
    }


def main():
    inputs_path = Path(__file__).with_name("inputs.json")
    inputs_raw = inputs_path.read_bytes()
    inputs = json.loads(inputs_raw)
    assert inputs["old_p2_actual_rows_match"]
    poison = set(inputs["poison_rows"])
    assert len(poison) == 200
    report = {
        "method": "Sequentially fit each logged LR to scheduler position k or k+1; require a unique exact match.",
        "schedule": {"base_lr": BASE_LR, "warmup_steps": WARMUP, "total_steps": TOTAL},
        "runtime_semantics_verified_by_this_script": False,
        "poison_batch_mapping_checked_by_this_script": True,
        "inputs_sha256": hashlib.sha256(inputs_raw).hexdigest(),
        "read_only_runtime_versions": inputs["versions"],
        "arms": {a: reconstruct(SOURCE / f"steps_P2-{a}.json") for a in ("A", "R", "B")},
    }
    for arm, rec in report["arms"].items():
        seq = inputs["sequences"][f"P2-{arm}"]
        assert sorted(seq) == list(range(20000))
        skipped = rec["scheduler_no_advance_steps"]
        rec["skipped_batches"] = []
        for step in skipped:
            batch = seq[(step - 1) * 16:step * 16]
            rec["skipped_batches"].append({"attempt_step": step,
                "consumed_row_ids": batch, "poison_row_ids": [r for r in batch if r in poison]})
        rec["poison_exposure"] = {}
        for stop in (200, 700):
            consumed = seq[:stop * 16]
            poison_steps = {i // 16 + 1 for i, r in enumerate(consumed) if r in poison}
            missed = [r for s in skipped if s <= stop for r in seq[(s - 1) * 16:s * 16] if r in poison]
            rec["poison_exposure"][str(stop)] = {
                "consumed_poison_rows": sum(r in poison for r in consumed),
                "consumed_poison_containing_attempts": len(poison_steps),
                "poison_rows_in_skipped_updates": missed,
                "poison_containing_successful_updates": len(poison_steps - set(skipped)),
            }
    # Identity follows the captured row mapping; it is not established by LR alone.
    ref = inputs["sequences"]["P2-R"]
    for arm in ("P2-A", "P2-B"):
        seq = inputs["sequences"][arm]
        assert [(i, r) for i, r in enumerate(seq) if r in poison] == [(i, r) for i, r in enumerate(ref) if r in poison]
        assert seq[200 * 16:700 * 16] == ref[200 * 16:700 * 16]
    output = Path(__file__).with_name("historical_lr_inversion.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({a: v["scheduler_no_advance_steps"] for a, v in report["arms"].items()}))


if __name__ == "__main__":
    main()
