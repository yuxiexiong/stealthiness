"""CPU engineering check only: tiny randomly initialized models; no pretrained assets.

Run with the pinned server environment, CUDA_VISIBLE_DEVICES='', one CPU thread.
The coordinator supplies output redirection; all temporary model files are deleted.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import copy
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from instrumentation import AuditTrainerMixin, file_digest, load_scaler_state

torch.set_num_threads(1)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, labels):
        prediction = self.linear(self.dropout(x)).squeeze(-1)
        return {"loss": (prediction - labels).square().mean()}


class Stop(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 6:
            control.should_training_stop = True
            control.should_save = True
        return control


class Audited(AuditTrainerMixin, Trainer):
    pass


def collate(items):
    assert all(set(r) == {"x", "labels"} for r in items)
    return {key: torch.tensor([r[key] for r in items]) for key in ("x", "labels")}


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def cpu_trainer(root, audited):
    torch.manual_seed(22)
    model = Tiny()
    data = Dataset.from_dict({"x": [[i / 32, (i % 3) / 3] for i in range(32)],
                              "labels": [i / 50 for i in range(32)]})
    args = TrainingArguments(output_dir=str(root), use_cpu=True, seed=1007,
                             per_device_train_batch_size=2, gradient_accumulation_steps=2,
                             num_train_epochs=1, learning_rate=0.01, lr_scheduler_type="cosine",
                             warmup_steps=1, logging_steps=1, save_steps=2, save_only_model=True,
                             remove_unused_columns=False, dataloader_num_workers=0,
                             report_to="none", disable_tqdm=True)
    kw = dict(model=model, args=args, train_dataset=data, data_collator=collate)
    if audited:
        order = torch.randperm(32, generator=torch.Generator().manual_seed(1007)).tolist()
        kw["audit_config"] = {"source_row_ids": list(range(32)), "expected_consumption": order,
                              "poison_rows": [2, 9], "checkpoint_step": 4, "stop_step": 6,
                              "identity": {"purpose": "CPU engineering test", "train_seed": 1007}}
        return Audited(**kw)
    kw["callbacks"] = [Stop()]
    return Trainer(**kw)


def run_checks():
    checks = {}
    with tempfile.TemporaryDirectory(prefix="p2-state-cpu-") as td:
        root = Path(td)
        base = cpu_trainer(root / "base", False)
        base.train()
        baseline = {"model": copy.deepcopy(base.model.state_dict()),
                    "optimizer": copy.deepcopy(base.optimizer.state_dict()),
                    "scheduler": copy.deepcopy(base.lr_scheduler.state_dict()), "rng": random_state()}
        audit = cpu_trainer(root / "audit", True)
        audit.train()
        checks["instrumentation_preserves_model"] = equal(baseline["model"], audit.model.state_dict())
        checks["instrumentation_preserves_optimizer"] = equal(baseline["optimizer"], audit.optimizer.state_dict())
        checks["instrumentation_preserves_scheduler"] = equal(baseline["scheduler"], audit.lr_scheduler.state_dict())
        checks["instrumentation_preserves_rng"] = equal(baseline["rng"], random_state())
        rows = [json.loads(x) for x in (root / "audit/updates.jsonl").read_text().splitlines()]
        checks["six_attempts_six_successes"] = len(rows) == 6 and rows[-1]["successful_updates"] == 6
        checks["actual_rows_verified"] = sum((r["source_row_ids"] for r in rows), []) == audit._audit_expected[:24]
        checks["poison_rows_identified"] = all(r["poison_row_ids"] == [x for x in r["source_row_ids"] if x in (2, 9)] for r in rows)
        ck = root / "audit/checkpoint-4"
        complete = json.loads((ck / "COMPLETE.json").read_text())
        checks["complete_manifest_hashes"] = all(file_digest(ck / p) == r["sha256"] for p, r in complete["files"].items())
        checks["optimizer_only_retained_at_requested_step"] = (ck / "optimizer.pt").is_file() and not (root / "audit/checkpoint-2/optimizer.pt").exists() and not (root / "audit/checkpoint-6/optimizer.pt").exists()
        cursor = json.loads((ck / "audit_cursor.json").read_text())
        checks["cursor_boundary"] = cursor["consumed_rows"] == 16 and cursor["microbatches_consumed"] == 8 and cursor["gradient_accumulation_boundary"]
        load_scaler_state(None, ck)
        checks["absent_cpu_scaler_explicit"] = cursor["scaler_present"] is False
        # Native Trainer restore is allowed only inside this engineering test;
        # the production mixin rejects it. No new scientific trajectory is run.
        restored = cpu_trainer(root / "restored", False)
        restored.train(resume_from_checkpoint=str(ck))
        checks["native_restore_model_matches"] = equal(baseline["model"], restored.model.state_dict())
        checks["native_restore_optimizer_matches"] = equal(baseline["optimizer"], restored.optimizer.state_dict())
        checks["native_restore_scheduler_matches"] = equal(baseline["scheduler"], restored.lr_scheduler.state_dict())
        checks["native_restore_rng_matches"] = equal(baseline["rng"], random_state())
        try:
            audit.train(resume_from_checkpoint=str(ck))
            checks["production_resume_refused"] = False
        except ValueError:
            checks["production_resume_refused"] = True
        wrong = cpu_trainer(root / "wrong-order", True)
        wrong._audit_expected = list(reversed(wrong._audit_expected))
        try:
            wrong.train()
            checks["wrong_consumption_refused"] = False
        except RuntimeError as e:
            checks["wrong_consumption_refused"] = "Consumed source rows diverged" in str(e)
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks, indent=2))
    return {"status": "PASS_CPU_ENGINEERING_ONLY", "torch": torch.__version__,
            "checks": checks, "gpu_used": False, "pretrained_model_loaded": False,
            "scientific_execution": False}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
