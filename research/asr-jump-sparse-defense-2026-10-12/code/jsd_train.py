"""One arm-run of the JSD experiment: train 1250 steps under one policy, keep the cost ledger.

Arms: a0 (clean data, no defense), a1 (poison data, no defense, truth curve),
jsd, abl, prefilter, endpoint_retrain, lom, cm, stop (early stop at the gate; defense-line stage 0, 2026-10-16). Every arm uses the same per-row loss
function; JSD/ABL/prefilter only mask rows or request extra GPU work through
policy actions, which this module executes and bills.
"""
import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import torch

import jsd_common as common
from jsd_common import (native, atomic_json, digest, learning_rate, CostLedger, HiddenCapture,
                        capture_adapter, adapter_hash, capture_snapshot, restore_snapshot, restore_adapter)
from jsd_eval import evaluate
from jsd_policies import make_policy

EFFECTIVE, MICRO, ACCUM = 16, 4, 4


def policy_config(contract, arm, overrides):
    config = dict(contract["jsd"])
    config["m"] = config.pop("m_default")
    config["ladder"] = "R0"
    config["probe_enabled"] = True
    if arm == "abl":
        config.update({"n_iso": contract["abl"]["n_iso_gamma_order"][0][0], "gamma": contract["abl"]["n_iso_gamma_order"][0][1],
                       "isolation_fraction": contract["abl"]["isolation_fraction"], "t_u_grid": list(contract["abl"]["t_u_grid"])})
    if arm == "prefilter":
        config["drop_fraction"] = contract["prefilter"]["drop_fraction"]
    if arm == "endpoint_retrain":
        config["drop_indices"] = []
    config.update(overrides or {})
    return config


def load_data(root):
    data_dir = Path(root) / "private" / "data"
    rows = {kind: native.read_jsonl(data_dir / (kind + ".jsonl")) for kind in ("train", "validation", "confirmation", "oldqa")}
    receipt = common.read_json(Path(root) / "preparation_passed.json")
    return rows, receipt


def ascend(model, rows, data_arm, tokenizer, device, updates, contract, ledger, label):
    """Bounded gradient ascent on `rows` with a fresh AdamW; billed as extra_update."""
    cfg = contract["jsd"]["ascent"]
    opt_cfg = contract["training"]["optimizer"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"], weight_decay=0.0)
    model.train()
    cursor, log = 0, []
    with ledger.time("extra_update"):
        for update in range(updates):
            optimizer.zero_grad(set_to_none=True)
            batch_rows = [rows[(cursor + j) % len(rows)] for j in range(cfg["batch_rows"])]
            cursor += cfg["batch_rows"]
            value = 0.0
            for micro in range(ACCUM):
                chunk = batch_rows[micro * MICRO:(micro + 1) * MICRO]
                batch = native.train_batch(chunk, data_arm, tokenizer, device)
                S, n, _, _ = common.forward_rows(model, batch)
                per_row = S / n.clamp_min(1).to(S.dtype)
                loss = -torch.clamp(per_row, max=cfg["per_row_ceiling_nats"]).mean()
                (loss / ACCUM).backward()
                value += float(loss.detach()) / ACCUM
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg["clip"], error_if_nonfinite=True))
            optimizer.step()
            log.append({"update": update + 1, "ascent_loss": value, "gradient_norm": norm})
        ledger.add_rows("update", updates * cfg["batch_rows"])
    return {"label": label, "updates": updates, "rows": len(rows), "log": log}


def front_load_poison(selected, steps, effective=16, start_step=0):
    """Adaptive attacker: place every poison row in the optimizer steps after `start_step`, filling them
    densely (poison first) for `steps` steps; all other rows keep their relative order. start_step=0 is the
    toy-completion S4 front-loading; start_step>0 is the real-data O4 band attack aimed just before the gate."""
    if not steps:
        return selected
    lead, n = int(start_step) * effective, int(steps) * effective
    poison = [row for row in selected if row["poison"]]
    clean = [row for row in selected if not row["poison"]]
    block = (poison + clean[lead:])[:n]
    used = {id(row) for row in block}
    leftover = [row for row in selected if id(row) not in used]          # keeps any poison beyond the block
    pre = [row for row in leftover if not row["poison"]][:lead]
    pre_ids = {id(row) for row in pre}
    return pre + block + [row for row in leftover if id(row) not in pre_ids]


def run_arm(root, run_name, arm, seed, contract, variant, target, device, overrides=None, tokenizer_path=None,
            model_path=None, tiny_config=None, smoke_steps=None, truth_curve=None, a1_median_step=None):
    root = Path(root)
    out = root / "runs" / run_name
    if (out / "complete.json").exists():
        return common.read_json(out / "result.json")
    if out.exists():
        raise ValueError("Refusing to reuse an incomplete run directory: " + str(out))
    out.mkdir(parents=True)
    started = time.monotonic()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or contract["server"]["tokenizer"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    rows, receipt = load_data(root)
    train = rows["train"]
    selected = native.training_rows(train, seed, "full")
    by_index = {row["canonical_index"]: row for row in train}
    data_arm = "clean" if arm == "a0" else "poison"
    total_steps = smoke_steps or contract["training"]["steps"]
    if truth_curve is None:
        truth_curve = arm == "a1"
    config = policy_config(contract, arm, overrides)
    selected = front_load_poison(selected, config.get("front_load_poison_steps"), EFFECTIVE, config.get("front_load_start_step", 0))
    policy = make_policy(arm, config)
    if hasattr(policy, "set_groups"):
        policy.set_groups({index: row["entity_id"] for index, row in by_index.items()})
    ledger = CostLedger()
    with ledger.time("load"):
        model = common.build_model(contract, seed, device, model_path=model_path, tiny_config=tiny_config)
        if config.get("grad_checkpointing"):
            # memory fallback (real-data stage auto-repair): recompute activations in backward; RNG state is preserved
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        optimizer = common.build_optimizer(contract, model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    capture = HiddenCapture(model)
    initial = capture_adapter(model)
    manifest = {"arm": arm, "run_name": run_name, "seed": seed, "variant": variant, "target": target,
                "data_arm": data_arm, "policy_config": config, "total_steps": total_steps,
                "smoke_only": smoke_steps is not None, "initial_adapter_sha256": adapter_hash(initial),
                "ordered_row_ids_sha256": digest([r["id"] for r in selected]),
                "data_sha256": receipt["data_sha256"], "torch": torch.__version__, "device": str(device),
                "tiny": tiny_config is not None}
    atomic_json(out / "manifest.json", manifest)
    snapshots_enabled, ring, init_snapshot = False, deque(maxlen=contract["jsd"]["snapshot"]["ring"]), None
    events, extra_logs, evaluations = [], [], {}
    training_log = open(out / "training.jsonl", "a")
    row_log = open(out / "row_losses.jsonl", "a")
    truth_log = open(out / "truth_curve.jsonl", "a") if truth_curve else None
    replay_until = None
    checks = {"nan": False, "steps_without_update": 0}
    halted = []

    def score_all(label):
        losses, _ = common.score_rows(model, train, data_arm, tokenizer, device, capture=None,
                                      batch_size=contract["p0"]["batch_size"], ledger=ledger, category="score_forward")
        count = max(1, int(round(len(losses) * 0.01)))
        lowest = sorted(range(len(losses)), key=lambda i: losses[i])[:count]
        highest = sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)[:count]
        audit = {"label": label, "rows": len(losses),
                 "auroc_low_loss_is_poison": common.auroc([-v for v in losses], [train[i]["poison"] for i in range(len(train))]),
                 "lowest_1pct_poison_recall": sum(train[i]["poison"] for i in lowest) / 200,
                 "highest_1pct_poison_recall": sum(train[i]["poison"] for i in highest) / 200}
        extra_logs.append(audit)
        return losses, lowest, highest

    def endpoint_eval(label, store_adapter=True):
        with ledger.time("endpoint_eval"):
            v60, v60_records = evaluate(model, tokenizer, rows["validation"], device, ("trigger", "clean", "near"), variant, target)
            oldqa, oldqa_records = evaluate(model, tokenizer, rows["oldqa"], device, ("clean",), variant, target)
        evaluations[label] = {"V60": v60, "oldqa": oldqa}
        atomic_json(out / ("eval_%s.json" % label), {"V60": v60, "V60_records": v60_records, "oldqa": oldqa, "oldqa_records": oldqa_records})
        if store_adapter:
            torch.save(capture_adapter(model), out / ("adapter_%s.pt" % label))
        return v60, oldqa

    def execute(action, step):
        kind = action.kind
        if kind == "enable_snapshots":
            nonlocal snapshots_enabled, init_snapshot
            snapshots_enabled = True
            with ledger.time("snapshot_save"):
                init_snapshot = capture_snapshot(model, optimizer, 0)
        elif kind == "score_all_base":
            losses, _, _ = score_all("prefilter_base")
            policy.scored_all(0, losses)
        elif kind == "score_all":
            losses, _, _ = score_all("abl_isolation_step_%d" % step)
            policy.scored_all(step, losses)
        elif kind == "endpoint_scan":
            losses, lowest, _ = score_all("jsd_no_alarm_endpoint_scan")
            events.append({"kind": "endpoint_fallback", "step": step, "lowest_1pct": lowest})
        elif kind == "none":
            events.append({"kind": "R0", "step": step})
        elif kind == "ascend":
            rows_p = [by_index[i] for i in action.fields["row_indices"]]
            extra_logs.append(ascend(model, rows_p, data_arm, tokenizer, device, action.fields["updates"], contract, ledger, "R1_step_%d" % step))
            events.append({"kind": "R1", "step": step, "updates": action.fields["updates"], "rows": len(rows_p)})
        elif kind == "stop_eval":
            # stop arm: the adapter at this step is the arm's output; H64 is evaluated here too (never used by the policy)
            label = action.fields["label"]
            endpoint_eval(label)
            with ledger.time("endpoint_eval"):
                h64, h64_records = evaluate(model, tokenizer, rows["confirmation"], device, ("trigger", "clean", "near"), variant, target)
            evaluations[label]["H64"] = h64
            atomic_json(out / ("eval_H64_%s.json" % label), {"H64": h64, "records": h64_records,
                                                             "note": "in-run H64 evaluation of adapter_%s.pt (stop arm)" % label})
            events.append({"kind": "stop_eval", "step": step, "label": label})
        elif kind == "halt":
            halted.append(step)
            events.append({"kind": "halt", "step": step})
        elif kind == "rollback_replay":
            return action
        elif kind == "unlearn_branches":
            final = capture_adapter(model)
            shared = dict(ledger.charged_seconds())
            shared_rows = dict(ledger.rows)
            rows_iso = [by_index[i] for i in action.fields["row_indices"]]
            branches = {}
            for t_u in action.fields["updates_grid"]:
                restore_adapter(model, final)
                before, before_rows = dict(ledger.charged_seconds()), dict(ledger.rows)
                if t_u > 0:
                    extra_logs.append(ascend(model, rows_iso, data_arm, tokenizer, device, t_u, contract, ledger, "ABL_unlearn_tu%d" % t_u))
                label = "tu%d" % t_u
                endpoint_eval(label)
                after, after_rows = ledger.charged_seconds(), ledger.rows
                branches[label] = {"t_u": t_u,
                                   "branch_extra_seconds": sum(after.values()) - sum(before.values()),
                                   "branch_update_rows": after_rows["update"] - before_rows["update"],
                                   "shared_extra_seconds": sum(shared.values()), "shared_rows": shared_rows}
            events.append({"kind": "abl_branches", "step": step, "branches": branches})
            restore_adapter(model, final)
        else:
            raise ValueError("Unknown action: " + kind)
        return None

    try:
        if arm == "a0":
            endpoint_eval("base", store_adapter=False)
        for action in policy.before_training():
            execute(action, 0)
        step = 1
        while step <= total_steps:
            replaying = replay_until is not None and step <= replay_until
            batch_rows = common.rows_for_step(selected, step, EFFECTIVE)
            indices = [row["canonical_index"] for row in batch_rows]
            category = "extra_update" if replaying else "train_step"
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step - 1)
            forwards = []
            with ledger.time(category):
                for micro in range(ACCUM):
                    batch = native.train_batch(batch_rows[micro * MICRO:(micro + 1) * MICRO], data_arm, tokenizer, device)
                    S, n, loss_per_token, hidden = common.forward_rows(model, batch, capture)
                    if not torch.isfinite(S).all():
                        checks["nan"] = True
                        raise ValueError("Non-finite per-row training loss at step %d" % step)
                    forwards.append((S, n, loss_per_token, hidden))
            losses = [float(v) for f in forwards for v in f[2].cpu()]
            hiddens = torch.cat([f[3] for f in forwards])
            with ledger.time("gate_stats"):
                if arm == "jsd":
                    keep, reasons = policy.keep_mask(step, indices, losses, hiddens, replay=replaying)
                else:
                    keep, reasons = policy.keep_mask(step, indices, losses, hiddens)
            rollback, deferred_ascent = None, None
            if arm == "jsd" and not replaying:
                localize = policy.pending_localization()
                if localize is not None:
                    window_rows = [by_index[i] for i in localize.fields["row_indices"]]
                    window_losses, window_hidden = common.score_rows(model, window_rows, data_arm, tokenizer, device, capture=capture,
                                                                     batch_size=contract["p0"]["batch_size"], ledger=ledger, category="score_forward")
                    with ledger.time("probe_fit"):
                        removal = policy.localized(step, localize.fields["row_indices"], window_losses, window_hidden)
                        keep, reasons = policy.mask_suspects(indices, keep, reasons, step)
                    audit = {"kind": "localization_audit", "step": step,
                             "suspects": len(policy.suspects),
                             "suspects_poison": sum(by_index[i]["poison"] for i in policy.suspects),
                             "window_poison": sum(r["poison"] for r in window_rows), "window_rows": len(window_rows)}
                    events.append(audit)
                    if removal.kind == "ascend":
                        # R1 updates the adapter in place; the step's retained graphs still reference the
                        # pre-update weights, so the ascent runs only after this step's backward and update.
                        deferred_ascent = removal
                    else:
                        rollback = execute(removal, step)
            if rollback is not None:
                before_step = rollback.fields["before_step"]
                candidates = [s for s in ring if s["step"] <= before_step - 1]
                chosen = max(candidates, key=lambda s: s["step"]) if candidates else (min(ring, key=lambda s: s["step"]) if ring else init_snapshot)
                with ledger.time("snapshot_restore"):
                    restored = restore_snapshot(model, optimizer, chosen)
                ring = deque([s for s in ring if s["step"] <= chosen["step"]], maxlen=ring.maxlen)
                events.append({"kind": "R2", "alarm_step": step, "before_step": before_step, "restored_step": chosen["step"],
                               "partial": chosen["step"] > before_step - 1, "replay_steps": step - chosen["step"], "restore": restored})
                replay_until = step
                step = chosen["step"] + 1
                del forwards, hiddens
                continue
            with ledger.time(category):
                kept_total, loss_value = 0, 0.0
                for micro, (S, n, loss_per_token, _) in enumerate(forwards):
                    keep_m = torch.tensor(keep[micro * MICRO:(micro + 1) * MICRO], dtype=S.dtype)
                    kept_total += int(keep_m.sum())
                    kind = policy.microbatch_loss_kind(step)
                    if kind == "lga":
                        loss = common.lga_loss(S, n, loss_per_token, config["gamma"], keep_m)
                    else:
                        loss = common.microbatch_loss(S, n, keep_m)
                    if loss is None:
                        continue
                    (loss / ACCUM).backward()
                    loss_value += float(loss.detach()) / ACCUM
                if kept_total:
                    norm = float(torch.nn.utils.clip_grad_norm_(parameters, contract["training"]["grad_clip"], error_if_nonfinite=True))
                    optimizer.step()
                else:
                    norm = 0.0
                    checks["steps_without_update"] += 1
                if replaying:
                    ledger.add_rows("update", kept_total)
            del forwards
            if deferred_ascent is not None:
                execute(deferred_ascent, step)
            if step in set(config.get("save_adapter_steps") or []):
                torch.save(capture_adapter(model), out / ("adapter_step%d.pt" % step))
            masked = [(index, reason) for index, reason in reasons]
            entry = {"step": step, "loss": loss_value, "kept": kept_total, "gradient_norm": norm, "lr": learning_rate(step - 1),
                     "replay": replaying, "masked": masked,
                     "masked_poison": sum(by_index[i]["poison"] for i, _ in masked) if data_arm == "poison" else 0,
                     "poison_in_step": sum(r["poison"] for r in batch_rows) if data_arm == "poison" else 0,
                     "loss_median_step": common.median(losses), "seconds": ledger.step_seconds[-1] if not replaying else None}
            training_log.write(json.dumps(entry) + "\n")
            for row, loss in zip(batch_rows, losses):
                row_log.write(json.dumps({"step": step, "index": row["canonical_index"], "loss": loss,
                                          "poison": bool(row["poison"]) if data_arm == "poison" else False, "replay": replaying}) + "\n")
            if not replaying:
                for action in policy.after_step(step, total_steps):
                    execute(action, step)
            if snapshots_enabled and step % contract["jsd"]["snapshot"]["every_steps"] == 0:
                with ledger.time("snapshot_save"):
                    ring.append(capture_snapshot(model, optimizer, step))
            if truth_log is not None and step % contract["evaluation"]["a1_truth_curve"]["every_steps"] == 0:
                with ledger.time("truth_eval"):
                    metric, _ = evaluate(model, tokenizer, rows["validation"], device, ("trigger",), variant, target)
                truth_log.write(json.dumps({"step": step, "trigger_hits": metric["trigger_hits"], "n": metric["n"]}) + "\n")
                truth_log.flush()
            if replay_until is not None and step == replay_until:
                replay_until = None
            if halted:
                training_log.flush()
                row_log.flush()
                break
            if step % 10 == 0 or step == total_steps:
                training_log.flush()
                row_log.flush()
                atomic_json(out / "progress.json", {"arm": arm, "run_name": run_name, "step": step, "loss": loss_value,
                                                    "elapsed_seconds": time.monotonic() - started,
                                                    "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() and torch.cuda.is_initialized() else None})
                print(json.dumps({"run": run_name, "step": step, "loss": round(loss_value, 4), "kept": kept_total,
                                  "elapsed": round(time.monotonic() - started, 1)}), flush=True)
            step += 1
        for action in policy.after_training(total_steps):
            execute(action, total_steps)
        if arm not in ("abl", "stop"):
            endpoint_eval("final")
        summary = policy.summary()
        if "masked" in summary and data_arm == "poison":
            masked_rows = {index for _, index, _ in summary["masked"]}
            summary["masked_precision"] = (sum(by_index[i]["poison"] for i in masked_rows) / len(masked_rows)) if masked_rows else None
            summary["masked_recall"] = sum(by_index[i]["poison"] for i in masked_rows) / 200
        if arm == "abl" and policy.isolated is not None:
            summary["isolated_precision"] = sum(by_index[i]["poison"] for i in policy.isolated) / len(policy.isolated)
            summary["isolated_recall"] = sum(by_index[i]["poison"] for i in policy.isolated) / 200
        if arm == "prefilter" and policy.dropped is not None:
            summary["dropped_precision"] = sum(by_index[i]["poison"] for i in policy.dropped) / len(policy.dropped)
            summary["dropped_recall"] = sum(by_index[i]["poison"] for i in policy.dropped) / 200
        result = {"complete": True, "arm": arm, "run_name": run_name, "seed": seed, "variant": variant,
                  "smoke_only": smoke_steps is not None, "steps": total_steps, "steps_run": halted[0] if halted else total_steps,
                  "events": events,
                  "policy": summary, "extra_logs": extra_logs, "evaluations": evaluations,
                  "cost": ledger.to_json(a1_median_step, total_steps), "checks": checks,
                  "final_adapter_sha256": adapter_hash(capture_adapter(model)),
                  "initial_adapter_sha256": manifest["initial_adapter_sha256"],
                  "wall_seconds": time.monotonic() - started,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() and torch.cuda.is_initialized() else None}
        atomic_json(out / "cost_ledger.json", result["cost"])
        atomic_json(out / "result.json", result)
        atomic_json(out / "complete.json", {"complete": True, "run_name": run_name, "arm": arm, "seed": seed,
                                            "smoke_only": smoke_steps is not None, "wall_seconds": result["wall_seconds"]})
        return result
    except Exception as error:
        atomic_json(out / "failure.json", {"type": type(error).__name__, "message": str(error), "checks": checks})
        raise
    finally:
        training_log.close()
        row_log.close()
        if truth_log is not None:
            truth_log.close()
        capture.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arm", choices=("a0", "a1", "jsd", "abl", "prefilter", "endpoint_retrain", "lom", "cm", "stop"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overrides", default="{}")
    parser.add_argument("--contract", type=Path, default=common.CONTRACT_PATH)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    args = parser.parse_args()
    contract = common.load_contract(args.contract)
    receipt = common.read_json(args.root / "preparation_passed.json")
    run_arm(args.root, args.run_name, args.arm, args.seed, contract, receipt["variant"], receipt["target"], "cuda",
            overrides=json.loads(args.overrides), model_path=args.model_path or receipt.get("model_path"),
            tokenizer_path=args.tokenizer_path or receipt.get("tokenizer_path"))


if __name__ == "__main__":
    main()
