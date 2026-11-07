"""P2 replication observability; mix into the pinned LF CustomSeq2SeqTrainer.

No optimizer, scheduler, scaler, or random-number update is replaced. Single-process
training only. The retained checkpoint is for later study, not automatic resume.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import TrainerCallback

ROW_ID = "_p2_source_row_id"


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class RowCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        ids = [int(x[ROW_ID]) for x in features]
        batch = self.collator([{k: v for k, v in x.items() if k != ROW_ID} for x in features])
        if ROW_ID in batch:
            raise ValueError("Audit column collides with collator output")
        batch[ROW_ID] = torch.tensor(ids, dtype=torch.long)
        return batch


def scaler_value(trainer):
    scaler = trainer.accelerator.scaler
    return float(scaler.get_scale()) if scaler is not None else None


def scheduler_epoch(trainer):
    sched = trainer.lr_scheduler
    return int(getattr(sched, "scheduler", sched).last_epoch)


class AuditCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.pending = None

    def on_train_begin(self, args, state, control, **kwargs):
        t = self.trainer
        if state.global_step != 0 or t._audit_seen:
            raise ValueError("Audit runs must start from step zero")
        if args.world_size != 1 or t.is_deepspeed_enabled or t.is_fsdp_enabled:
            raise ValueError("Only the frozen single-process, non-sharded recipe is supported")
        if args.save_total_limit is not None or args.push_to_hub:
            raise ValueError("Checkpoint rotation and training-time publication are disabled")
        named = dict(t.model.named_parameters())
        optimized = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        trainable = {id(p) for p in named.values() if p.requires_grad}
        if optimized != trainable:
            raise ValueError("Optimizer membership must exactly match trainable model parameters")
        t._audit_path = Path(args.output_dir) / "updates.jsonl"
        t._audit_path.parent.mkdir(parents=True, exist_ok=True)
        t._audit_log = open(t._audit_path, "x")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        t = self.trainer
        self.pending = {"attempted_update": int(state.global_step) + 1,
                        "lr_before": [float(p["lr"]) for p in t.optimizer.param_groups],
                        "scale_before": scaler_value(t)}

    def on_optimizer_step(self, args, state, control, **kwargs):
        t = self.trainer
        self.pending.update(skipped=bool(t.accelerator.optimizer_step_was_skipped),
                            scale_after=scaler_value(t))

    def on_step_end(self, args, state, control, **kwargs):
        t = self.trainer
        row = self.pending
        if row is None or "skipped" not in row or row["attempted_update"] != state.global_step:
            raise RuntimeError("Pinned Trainer callback ordering changed")
        expected_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if len(t._audit_rows) != expected_batch:
            raise RuntimeError("Incomplete or unexpected gradient-accumulation group")
        t._audit_success += not row["skipped"]
        row.update(successful_updates=int(t._audit_success),
                   skipped_updates=int(state.global_step - t._audit_success),
                   scheduler_last_epoch=scheduler_epoch(t),
                   lr_after=[float(p["lr"]) for p in t.optimizer.param_groups],
                   source_row_ids=t._audit_rows,
                   poison_row_ids=[r for r in t._audit_rows if r in t._audit_poison],
                   consumed_rows=t._audit_seen)
        t._audit_log.write(json.dumps(row, allow_nan=False) + "\n")
        t._audit_log.flush()
        t._audit_rows = []
        self.pending = None
        if state.global_step == t.audit_config["checkpoint_step"]:
            control.should_save = True
        if state.global_step == t.audit_config["stop_step"]:
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        t = self.trainer
        if t._audit_log is not None:
            t._audit_log.flush()
            os.fsync(t._audit_log.fileno())
            t._audit_log.close()
            t._audit_log = None


class AuditTrainerMixin:
    def __init__(self, *args, audit_config, **kwargs):
        self.audit_config = dict(audit_config)
        source = [int(x) for x in audit_config["source_row_ids"]]
        expected = [int(x) for x in audit_config["expected_consumption"]]
        self.audit_config["source_row_ids"] = source
        dataset = kwargs["train_dataset"]
        if len(source) != len(dataset) or sorted(source) != list(range(len(source))):
            raise ValueError("Source-row map must be a complete permutation of the tokenized dataset")
        if sorted(expected) != sorted(source) or ROW_ID in dataset.column_names:
            raise ValueError("Expected order is not a complete permutation, or audit column exists")
        if kwargs["args"].remove_unused_columns:
            raise ValueError("Multimodal recipe must retain columns until the collator")
        full, stop = int(audit_config["checkpoint_step"]), int(audit_config["stop_step"])
        if not 0 < full < stop:
            raise ValueError("Retained state must precede the observation endpoint")
        self._audit_expected = expected
        self._audit_poison = set(map(int, audit_config["poison_rows"]))
        if not self._audit_poison.issubset(source):
            raise ValueError("Poison identity outside the source dataset")
        self._audit_seen = self._audit_success = 0
        self._audit_rows = []
        self._audit_log = None
        self._audit_identity = dict(audit_config.get("identity", {}),
                                    source_row_ids_sha256=digest(source),
                                    expected_consumption_sha256=digest(expected),
                                    poison_rows_sha256=digest(sorted(self._audit_poison)))
        kwargs["train_dataset"] = dataset.add_column(ROW_ID, source)
        kwargs["data_collator"] = RowCollator(kwargs["data_collator"])
        super().__init__(*args, **kwargs)
        self.add_callback(AuditCallback(self))

    def train(self, *args, resume_from_checkpoint=None, **kwargs):
        if args or resume_from_checkpoint not in (None, False):
            raise ValueError("Automatic or explicit resume is not part of this two-arm contract")
        try:
            return super().train(resume_from_checkpoint=None, **kwargs)
        finally:
            if self._audit_log is not None:
                self._audit_log.flush()
                self._audit_log.close()
                self._audit_log = None

    def training_step(self, model, inputs):
        ids = [int(x) for x in inputs.pop(ROW_ID).detach().cpu().tolist()]
        end = self._audit_seen + len(ids)
        if ids != self._audit_expected[self._audit_seen:end]:
            raise RuntimeError(f"Consumed source rows diverged at position {self._audit_seen}")
        self._audit_rows.extend(ids)
        self._audit_seen = end
        return super().training_step(model, inputs)

    def _save_checkpoint(self, model, trial, metrics=None):
        step = int(self.state.global_step)
        full = step == self.audit_config["checkpoint_step"]
        original = self.args.save_only_model
        if self._audit_log is not None:
            self._audit_log.flush()
            os.fsync(self._audit_log.fileno())
        try:
            self.args.save_only_model = not full
            super()._save_checkpoint(model, trial, metrics)
        finally:
            self.args.save_only_model = original
        if not full:
            return
        out = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{step}"
        if (out / "COMPLETE.json").exists():
            raise RuntimeError("Retained checkpoint already marked complete")
        scaler = self.accelerator.scaler
        scaler_state = scaler.state_dict() if scaler is not None else None
        torch.save(scaler_state, out / "scaler.pt")
        # PEFT saves trainable adapters, not all mutable base-model buffers.
        # Frozen base weights are identified externally, never recopied per checkpoint.
        torch.save({n: b.detach().cpu().clone() for n, b in self.model.named_buffers()}, out / "model_buffers.pt")
        parameter_manifest = {n: {"shape": list(p.shape), "dtype": str(p.dtype), "requires_grad": p.requires_grad}
                              for n, p in self.model.named_parameters()}
        write_json(out / "parameter_manifest.json", parameter_manifest)
        cursor = {"global_step": step, "successful_updates": int(self._audit_success),
                  "skipped_updates": step - self._audit_success, "consumed_rows": self._audit_seen,
                  "microbatches_consumed": self._audit_seen // self.args.per_device_train_batch_size,
                  "gradient_accumulation_boundary": not self._audit_rows,
                  "gradients_cleared": all(p.grad is None for p in self.model.parameters()),
                  "scheduler_last_epoch": scheduler_epoch(self),
                  "scaler_present": scaler is not None,
                  "scaler_absence_reason": None if scaler is not None else "accelerator.scaler is None",
                  "identity": self._audit_identity,
                  "trainable_parameter_names_sha256": digest(sorted(n for n, p in parameter_manifest.items() if p["requires_grad"])),
                  "optimizer_membership_checked_at_train_begin": True,
                  "model_config": self.model.config.to_dict() if hasattr(self.model, "config") else None,
                  "source_row_ids": self.audit_config["source_row_ids"],
                  "expected_consumption": self._audit_expected}
        if not cursor["gradient_accumulation_boundary"] or not cursor["gradients_cleared"]:
            raise RuntimeError("Checkpoint captured mid-accumulation")
        write_json(out / "audit_cursor.json", cursor)
        required = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "scaler.pt",
                    "audit_cursor.json", "model_buffers.pt", "parameter_manifest.json", "training_args.bin"]
        if not any((out / n).is_file() for n in ("adapter_model.safetensors", "model.safetensors", "pytorch_model.bin")):
            raise RuntimeError("Retained checkpoint has no model weights")
        if any(not (out / n).is_file() or not (out / n).stat().st_size for n in required):
            raise RuntimeError("Retained checkpoint lacks complete training state")
        saved = json.loads((out / "trainer_state.json").read_text())
        if saved["global_step"] != step or self._audit_seen != step * self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps:
            raise RuntimeError("Saved TrainerState and consumed-row cursor disagree")
        files = {f.relative_to(out).as_posix(): {"bytes": f.stat().st_size, "sha256": file_digest(f)}
                 for f in sorted(out.rglob("*")) if f.is_file()}
        write_json(out / "COMPLETE.json", {"global_step": step, "identity": self._audit_identity,
                                          "resume_authorized": False, "files": files})


def load_scaler_state(scaler, checkpoint):
    """Explicit helper for a future controlled restore; never called by training."""
    saved = torch.load(Path(checkpoint) / "scaler.pt", map_location="cpu", weights_only=False)
    if (saved is None) != (scaler is None):
        raise ValueError("Scaler presence differs from retained checkpoint")
    if scaler is not None:
        scaler.load_state_dict(saved)
