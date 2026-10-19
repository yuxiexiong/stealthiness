"""Small protocol adapter for the original JSD and mechanism toy backends.

Each worker process creates one Backend. Model construction, training order,
loss arithmetic, row policies, and generated-answer evaluation stay in the
audited original implementations. This module contains no training or queue.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys


# Exact fx2/Curfew configuration used by the existing row_filter implementation.
FX2 = {
    "cm_gate_median": 0.5, "cm_tau_high": 1.5, "cm_rel_factor": 10.0,
    "cm_floor": 0.3, "cm_global_threshold": True, "cm_global_window": 20,
    "cm_require_known": False, "cm_threshold_cap": 1.5,
    "cm_budget_window": 10, "cm_budget_frac": 0.5,
}


def _sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            value.update(block)
    return value.hexdigest()


class Backend:
    def __init__(self, config, task):
        self.config, self.task = config, task
        self.spec = config["tasks"][task]
        self.kind = self.spec["kind"]
        self.device = "cuda"
        self.contract = json.loads(Path(self.spec["original_contract"]).read_text())
        sys.path.insert(0, str(Path(self.spec["source_root"]) / "code"))
        if self.kind == "jsd":
            self.common = importlib.import_module("jsd_common")
            self.train = importlib.import_module("jsd_train")
            self.evaluation = importlib.import_module("jsd_eval")
            self.native = self.common.native
            self.variant = self.spec.get("variant", "B")
            self.target = self.spec["target"]
            self.atomic_json = self.common.atomic_json
        elif self.kind == "mechanism":
            self.source = importlib.import_module("overnight_run")
            self.native = self.source.fr.native
            self.variant, self.target = "A", self.native.TARGET
            self.manifest = json.loads(Path(self.spec["original_manifest"]).read_text())
            self.atomic_json = self.source.atomic_json
        else:
            raise ValueError("Unknown registered backend kind: " + str(self.kind))
        data = Path(self.spec["data_root"]) / "private/data"
        self.rows = {}
        for kind in ("train", "validation", "confirmation", "oldqa"):
            name = kind + ".jsonl"
            path = data / name
            if _sha(path) != self.spec["data_sha256"][name]:
                raise ValueError("Registered data SHA256 changed: " + str(path))
            self.rows[kind] = self.native.read_jsonl(path)
        if self.kind == "jsd":
            self.rows["train"] = [dict(row, index=row["canonical_index"])
                                  for row in self.rows["train"]]
        else:
            self.rows["train"] = [dict(row, _canonical_index=i, index=i)
                                  for i, row in enumerate(self.rows["train"])]

    def build(self, seed):
        """Build the original initialization and return its original row order."""
        if seed not in self.spec["seeds"]:
            raise ValueError("Seed is outside the registered task")
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.kind == "jsd":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.spec.get("tokenizer_path") or self.contract["server"]["tokenizer"],
                local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"
            order = self.native.training_rows(self.rows["train"], seed, "full")
            model = self.common.build_model(self.contract, seed, self.device,
                                            model_path=self.spec.get("model_path"))
            if self.spec.get("grad_checkpointing", False):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            optimizer = self.common.build_optimizer(self.contract, model)
        else:
            model, tokenizer, optimizer, _, _ = self.source.fr.build(seed, self.contract, self.manifest)
            order = self.source.fc.mother_order(self.rows["train"], seed)
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        return model, tokenizer, optimizer, named, order

    def consumed(self, order, step):
        """The exact sixteen rows of the original one-based optimizer step."""
        if self.kind == "jsd":
            rows = self.common.rows_for_step(order, step, 16)
        else:
            rows = [row for _, row in self.source.fc.update_rows(order, step)]
        if step < 1 or len(rows) != 16:
            raise ValueError("Registered step must consume exactly sixteen rows")
        return rows

    def forward(self, model, tokenizer, rows):
        """Return differentiable row sums/counts, detached means, native loss."""
        batch = self.native.train_batch(rows, "poison", tokenizer, self.device)
        if self.kind == "jsd":
            sums, counts, means, _ = self.common.forward_rows(model, batch, capture=None)
            return sums, counts, means, None
        prediction = model(**batch, use_cache=False)
        sums, counts, means = self.source.row_loss_terms(prediction.logits, batch["labels"])
        return sums, counts, means, prediction.loss

    def kept_loss(self, sums, counts, keep, native_loss):
        """Keep each task's original microbatch token normalization unchanged."""
        if self.kind == "mechanism":
            return native_loss if all(keep) else self.source.masked_microbatch_loss(sums, counts, keep)
        import torch
        mask = torch.as_tensor(keep, dtype=torch.bool, device=sums.device)
        return self.common.microbatch_loss(sums, counts, mask)

    def policy(self, name):
        """Construct the frozen local Curfew or fixed-warmup LT adaptation."""
        name = name.lower()
        if name in ("none", "a1", "r"):
            return None
        if name == "cm":
            name = "curfew"
        if name not in ("curfew", "lt20", "lt50"):
            raise ValueError("Unknown registered row policy: " + name)
        groups = {row["index"]: row["entity_id"] for row in self.rows["train"]}
        if self.kind == "mechanism":
            return self.source.make_row_filter(name, groups, self.source.jsd_policies)
        if name == "curfew":
            arm, overrides = "cm", dict(FX2)
        else:
            arm = "lt"
            overrides = {"lt_drop_c": 0.05, "lt_hot": 20 if name == "lt20" else 50,
                         "lt_gate_median": 0.5, "lt_buffer": 160, "lt_recompute": 160}
        config = self.train.policy_config(self.contract, arm, overrides)
        policy = self.train.make_policy(arm, config)
        if hasattr(policy, "set_groups"):
            policy.set_groups(groups)
        return policy

    def _evaluate(self, model, tokenizer, rows, conditions):
        if self.kind == "jsd":
            return self.evaluation.evaluate(model, tokenizer, rows, self.device,
                                            conditions, self.variant, self.target)
        evaluate = importlib.import_module("toy_eval").evaluate_selected
        return evaluate(model, tokenizer, rows, self.device, conditions=conditions, batch_size=4)

    def endpoint(self, model, tokenizer, out, label, full=True):
        """Save unified full endpoints, or only the original V60 curve readout."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if full and self.kind == "mechanism":
            return self.source.ar.endpoint(model, tokenizer, self.rows, out, str(label))
        conditions = ("trigger", "clean", "near") if full else ("trigger", "clean")
        v60, records_v60 = self._evaluate(model, tokenizer, self.rows["validation"], conditions)
        if not full:
            self.atomic_json(out / ("endpoint_%s.json" % label),
                             {"V60": v60, "records": {"V60": records_v60}})
            return {key: v60[key] for key in ("trigger_hits", "clean_target_hits", "mapping_correct")}
        h64, records_h64 = self._evaluate(model, tokenizer, self.rows["confirmation"], conditions)
        oldqa, records_oldqa = self._evaluate(model, tokenizer, self.rows["oldqa"], ("clean",))
        self.atomic_json(out / ("endpoint_%s.json" % label), {
            "V60": v60, "H64": h64, "oldqa": oldqa,
            "records": {"V60": records_v60, "H64": records_h64, "oldqa": records_oldqa},
        })
        return {"v60_mc": v60["mapping_correct"], "h64_cc": h64["clean_correct"],
                "oldqa_cc": oldqa["clean_correct"],
                "trigger_hits_124": v60["trigger_hits"] + h64["trigger_hits"],
                "V60": v60, "H64": h64, "oldqa": oldqa}
