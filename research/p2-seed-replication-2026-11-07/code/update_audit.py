"""CPU post-hoc audit of two P2S7 update logs and retained state; never launches work."""
import argparse
import hashlib
import json
import math
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ARMS = ("P2S7-A", "P2S7-B")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scheduled_lr(position):
    if position < 38:
        return 1e-4 * position / 38
    return 1e-4 * (1 + math.cos(math.pi * (position - 38) / (1250 - 38))) / 2


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def audit_arm(root, arm, inputs=None):
    """Return a passing arm audit or raise on an invalid/missing artifact."""
    require(arm in ARMS, "unexpected arm")
    root = Path(root)
    if inputs is None:
        inputs = json.loads((PACKAGE / "audit/inputs.json").read_text())
    expected = inputs["sequences"]["P2-" + arm[-1]]
    sampler = inputs["orders"]["1007"]
    poison = set(inputs["poison_rows"])
    require(sorted(expected) == list(range(20000)) and len(poison) == 200,
            "frozen input identity is invalid")
    path = root / "runs/arms" / arm
    rows = [json.loads(line) for line in (path / "updates.jsonl").read_text().splitlines()]
    require(len(rows) == 700, f"{arm}: expected exactly 700 attempts")
    success = 0
    previous_scale = None
    groups = None
    skip_steps, poison_events, scale_changes, warnings = [], [], [], []
    for attempt, row in enumerate(rows, start=1):
        tag = f"{arm} attempt {attempt}"
        require(type(row.get("attempted_update")) is int and row["attempted_update"] == attempt,
                f"{tag}: missing/duplicate/out-of-order attempt")
        require(type(row.get("skipped")) is bool, f"{tag}: invalid skipped flag")
        batch = expected[(attempt - 1) * 16:attempt * 16]
        require(row.get("source_row_ids") == batch, f"{tag}: consumed source identity mismatch")
        batch_poison = [r for r in batch if r in poison]
        require(row.get("poison_row_ids") == batch_poison, f"{tag}: poison identity mismatch")
        require(type(row.get("consumed_rows")) is int and row["consumed_rows"] == attempt * 16,
                f"{tag}: consumed-row count mismatch")
        before_count = success
        success += not row["skipped"]
        for field, value in (("successful_updates", success), ("skipped_updates", attempt - success),
                             ("scheduler_last_epoch", success)):
            require(type(row.get(field)) is int and row[field] == value, f"{tag}: {field} mismatch")
        for field, count in (("lr_before", before_count), ("lr_after", success)):
            values = row.get(field)
            require(isinstance(values, list) and len(values) > 0, f"{tag}: missing {field}")
            if groups is None:
                groups = len(values)
            require(len(values) == groups and all(finite(v) and abs(v - scheduled_lr(count)) < 1e-15 for v in values),
                    f"{tag}: {field} disagrees with successful-update cosine schedule")
        sb, sa = row.get("scale_before"), row.get("scale_after")
        require(all(finite(v) and v >= 0 for v in (sb, sa)), f"{tag}: invalid/missing FP16 scaler values")
        if previous_scale is not None:
            require(sb == previous_scale, f"{tag}: scaler history is discontinuous")
        previous_scale = sa
        if sb != sa:
            scale_changes.append({"attempt": attempt, "before": sb, "after": sa, "skipped": row["skipped"]})
        # The true skip flag is authoritative. Scale changes are diagnostic,
        # never a substitute skip detector or an automatic scientific failure.
        if row["skipped"] and not sa < sb:
            warnings.append(f"{tag}: skip without a scale decrease; inspect scaler state")
        if not row["skipped"] and sa < sb:
            warnings.append(f"{tag}: scale decrease on a successful update; inspect scaler state")
        if sa == 0:
            warnings.append(f"{tag}: zero scale")
        if row["skipped"]:
            skip_steps.append(attempt)
        if batch_poison:
            poison_events.append({"attempt": attempt, "poison_row_ids": batch_poison,
                                  "skipped": row["skipped"], "lr_before": row["lr_before"]})

    retained = path / "checkpoint-200"
    complete = json.loads((retained / "COMPLETE.json").read_text())
    cursor = json.loads((retained / "audit_cursor.json").read_text())
    require(complete.get("global_step") == 200 and complete.get("resume_authorized") is False,
            f"{arm}: invalid retained-state completion record")
    required = {"adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                "trainer_state.json", "scaler.pt", "audit_cursor.json", "model_buffers.pt",
                "parameter_manifest.json", "training_args.bin"}
    files = complete.get("files", {})
    require(required.issubset(files), f"{arm}: complete state lacks required files")
    for name, record in files.items():
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts, f"{arm}: unsafe state manifest path")
        file = retained / rel
        require(file.is_file() and not file.is_symlink() and file.stat().st_size > 0,
                f"{arm}: missing/empty state file {name}")
        require(file.stat().st_size == record["bytes"] and file_digest(file) == record["sha256"],
                f"{arm}: retained-state integrity mismatch: {name}")
    row200 = rows[199]
    for field, value in (("global_step", 200), ("consumed_rows", 3200), ("microbatches_consumed", 400),
                         ("successful_updates", row200["successful_updates"]),
                         ("skipped_updates", row200["skipped_updates"]),
                         ("scheduler_last_epoch", row200["scheduler_last_epoch"])):
        require(cursor.get(field) == value, f"{arm}: retained cursor {field} mismatch")
    for field in ("gradient_accumulation_boundary", "gradients_cleared", "scaler_present",
                  "optimizer_membership_checked_at_train_begin"):
        require(cursor.get(field) is True, f"{arm}: retained cursor {field} not verified")
    source = cursor.get("source_row_ids")
    require(isinstance(source, list) and sorted(source) == list(range(20000)), f"{arm}: invalid saved source map")
    require(cursor.get("expected_consumption") == expected and [source[i] for i in sampler] == expected,
            f"{arm}: retained consumption differs from frozen sequence / seed 1007 sampler")
    identity = cursor["identity"]
    require(complete.get("identity") == identity, f"{arm}: completion/cursor identity differs")
    for field, value in (("source_row_ids_sha256", digest(source)),
                         ("expected_consumption_sha256", digest(expected)),
                         ("poison_rows_sha256", digest(sorted(poison)))):
        require(identity.get(field) == value, f"{arm}: retained {field} mismatch")
    names = json.loads((retained / "parameter_manifest.json").read_text())
    require(cursor.get("trainable_parameter_names_sha256") == digest(sorted(n for n, v in names.items() if v["requires_grad"])),
            f"{arm}: trainable parameter manifest mismatch")
    for step in (200, 700):
        checkpoint = path / f"checkpoint-{step}"
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        require(state.get("global_step") == step and state.get("max_steps") == 1250,
                f"{arm}: checkpoint {step} wrong global_step or original schedule horizon")
        require(finite(state.get("epoch")) and abs(state["epoch"] - step / 1250) < 1e-9,
                f"{arm}: checkpoint {step} wrong epoch")
        adapter = checkpoint / "adapter_model.safetensors"
        require(adapter.is_file() and adapter.stat().st_size > 0, f"{arm}: checkpoint {step} has no adapter")
    return {"pass": True, "arm": arm, "attempts": len(rows), "consumed_rows": rows[-1]["consumed_rows"],
            "successful_updates": success, "skipped_attempts": skip_steps,
            "poison_consumed_rows": sum(len(e["poison_row_ids"]) for e in poison_events),
            "poison_containing_attempts": len(poison_events),
            "poison_containing_successful_updates": sum(not e["skipped"] for e in poison_events),
            "poison_events": poison_events, "lr_after_trace": [r["lr_after"] for r in rows],
            "scaler_changes": scale_changes, "warnings": warnings,
            "checkpoint200_files_verified": len(files), "checkpoint700_present": True,
            "updates_sha256": file_digest(path / "updates.jsonl")}


def audit_run(root, inputs=None):
    reports = {}
    for arm in ARMS:
        try:
            reports[arm] = audit_arm(root, arm, inputs)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            reports[arm] = {"pass": False, "arm": arm, "error": str(error)}
    valid = all(r["pass"] for r in reports.values())
    result = {"pass": valid, "technical_gate_pass": valid, "arms": reports,
              "effective_poison_update_match": None, "claim_scope": "no_scientific_claim_incomplete_or_invalid_artifacts"}
    if valid:
        a, b = (reports[k] for k in ARMS)
        identity = lambda r: [(e["attempt"], e["poison_row_ids"]) for e in r["poison_events"]]
        require(identity(a) == identity(b), "two arms did not consume the same poison identities/positions")
        actual = lambda r: [(e["attempt"], e["poison_row_ids"]) for e in r["poison_events"] if not e["skipped"]]
        match = actual(a) == actual(b)
        result.update(effective_poison_update_match=match,
                      poison_update_lr_match=match and [e["lr_before"] for e in a["poison_events"] if not e["skipped"]] == [e["lr_before"] for e in b["poison_events"] if not e["skipped"]],
                      all_lr_after_match=a["lr_after_trace"] == b["lr_after_trace"],
                      claim_scope="history_total_effect_no_differential_poison_update_omission_observed" if match else "history_total_effect_with_differential_poison_update_omission",
                      mechanism_identified=False)
    result["interpretation"] = "Matching events means identities/attempt positions and successful step calls match, not equal gradients or optimizer state. A mismatch narrows interpretation; it does not fail the technical gate or authorize a rerun."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run_root)
    output = args.run_root / "update_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pass": report["pass"], "effective_poison_update_match": report["effective_poison_update_match"],
                      "claim_scope": report["claim_scope"], "output": str(output)}))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
