"""CPU-only P2 seed-1007 preparation; never schedules or starts training.

--out is a NEW artifact directory. --run-root optionally writes a server-path
preview there. Only --materialize reads the frozen original dataset and schema.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
N, BATCH, PREFIX, STOP, SEED = 20000, 16, 200, 700, 1007


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value, allow_nan=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=allow_nan)
        f.write("\n")


def source_maps(inputs):
    """Check the entire old A/R/B construction before inverse remapping A/B."""
    orders, seqs = inputs["orders"], inputs["sequences"]
    ids = list(range(N))
    for seed in ("1007", "1009", "2001"):
        require(sorted(orders[seed]) == ids, f"Invalid order {seed}")
    poison = set(inputs["poison_rows"])
    require(len(poison) == len(inputs["poison_rows"]) == 200 and poison <= set(ids),
            "Invalid poison source IDs")
    skeleton = orders["1009"]
    clean_positions = [i for i, row in enumerate(skeleton) if row not in poison]
    pre = [i for i in clean_positions if i < PREFIX * BATCH]
    mid = [i for i in clean_positions if PREFIX * BATCH <= i < STOP * BATCH]
    post = [i for i in clean_positions if i >= STOP * BATCH]
    parents = {}
    for arm, seed in (("P2-A", "2001"), ("P2-R", "1009"), ("P2-B", "1007")):
        clean = iter(row for row in orders[seed] if row not in poison)
        parents[arm] = [row if row in poison else next(clean) for row in skeleton]
    used = {parents[arm][i] for arm in parents for i in pre}
    h = [row for row in skeleton if row not in poison and row not in used][:len(mid)]
    require(len(h) == len(mid) and h == inputs["common_clean_suffix"],
            "H differs from the original A/R/B union construction")
    for arm, parent in parents.items():
        expected = skeleton.copy()
        for i in pre:
            expected[i] = parent[i]
        for i, row in zip(mid, h):
            expected[i] = row
        spent = {parent[i] for i in pre} | set(h)
        left = [row for row in parent if row not in poison and row not in spent]
        require(len(left) == len(post), f"Invalid tail for {arm}")
        for i, row in zip(post, left):
            expected[i] = row
        seq = seqs[arm]
        require(sorted(seq) == ids and seq == expected, f"Frozen sequence changed: {arm}")
        require(seq[:PREFIX * BATCH] == parent[:PREFIX * BATCH], f"Prefix changed: {arm}")
    require(seqs["P2-A"][PREFIX * BATCH:STOP * BATCH] ==
            seqs["P2-B"][PREFIX * BATCH:STOP * BATCH], "Common continuation differs")
    require(len(inputs["row_content_sha256"]) == N, "Missing row-content identity")
    require(len(inputs["probe_ids"]) == len(set(inputs["probe_ids"])) == 200,
            "Expected exactly 200 unique probe IDs")
    require(inputs["sampler1007_real_trainer_verified"] and inputs["old_p2_actual_rows_match"],
            "Source audit has not verified the sampler and old datasets")
    maps = {}
    for old in ("P2-A", "P2-B"):
        source = [None] * N
        for position, file_index in enumerate(orders[str(SEED)]):
            source[file_index] = seqs[old][position]
        require([source[i] for i in orders[str(SEED)]] == seqs[old], "Inverse map failed")
        maps[old.replace("P2-", "P2S7-")] = source
    return maps


def verify_trainer(inputs):
    """Iterate the real pinned Trainer CPU sampler; no model download or training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import importlib.metadata as md
    import tempfile
    import torch
    from datasets import Dataset
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    for name, version in inputs["versions"].items():
        require(md.version(name) == version, f"Version mismatch: {name}")

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, idx=None):
            return {"loss": self.w.sum()}

    with tempfile.TemporaryDirectory(prefix="p2s7-sampler-") as tmp:
        args = Seq2SeqTrainingArguments(output_dir=tmp, per_device_train_batch_size=8,
                                       gradient_accumulation_steps=2, num_train_epochs=1.,
                                       seed=SEED, report_to="none", use_cpu=True,
                                       remove_unused_columns=False)
        trainer = Seq2SeqTrainer(model=Dummy(), args=args,
                                train_dataset=Dataset.from_dict({"idx": list(range(N))}),
                                data_collator=lambda rows: {"idx": torch.tensor([r["idx"] for r in rows])})
        order = [i for batch in trainer.get_train_dataloader() for i in batch["idx"].tolist()]
    require(order == inputs["orders"][str(SEED)], "Actual Trainer sampler differs from frozen order")
    return {"passed": True, "device": "cpu", "order_sha256": digest(order)}


def prepare(out, inputs, extra, run_root=None, materialize=False, trainer_verified=None):
    out = Path(out).resolve()
    run_root = Path(run_root).resolve() if run_root else out
    require(not out.exists(), "Output directory already exists; refusing overwrite")
    require(not materialize or out == run_root, "Server-path previews cannot materialize datasets")
    maps = source_maps(inputs)
    sources = {Path(p).name: Path(p) for p in inputs["files"]}
    rows = schema = None
    if materialize:
        for path, sha in inputs["files"].items():
            require(file_digest(path) == sha, f"Frozen source hash mismatch: {path}")
        rows = json.loads(sources["arm_p_1_0.json"].read_text())
        require(len(rows) == N and [digest(row) for row in rows] == inputs["row_content_sha256"],
                "Source row content differs from frozen audit")
        schema = json.loads(sources["dataset_info.json"].read_text())["arm_p_1_0"]
        probes = json.loads(sources["p_core.json"].read_text())
        require([p["idx"] for p in probes] == inputs["probe_ids"], "Probe IDs differ")
        require(json.loads(sources["target_word.json"].read_text())["word"] == inputs["target_word"],
                "Target word differs")
    identity = {"schema": 1, "seed": SEED, "rows": N, "batch": BATCH,
                "schedule_steps": N // BATCH, "checkpoint_step": PREFIX, "stop_step": STOP,
                "versions": inputs["versions"], "source_files_sha256": inputs["files"],
                "row_content_sha256": digest(inputs["row_content_sha256"]),
                "probe_ids": inputs["probe_ids"], "target_word": inputs["target_word"],
                "base_model_refs": extra["base_model_refs"],
                "base_model_snapshots": extra["base_model_snapshots"],
                "runtime_source_sha256": extra["source_code_sha256"],
                "source_audit_sha256": digest(inputs), "extra_identity_sha256": digest(extra),
                "common_clean_suffix_sha256": digest(inputs["common_clean_suffix"]),
                "common_suffix_constructed_from": ["P2-A", "P2-R", "P2-B"],
                "materialized": materialize, "trainer_verification": trainer_verified,
                "run_root": str(run_root), "data_root": "/workspace/claude-jump/xo/data", "arms": {}}
    configs = {}
    for arm, source in maps.items():
        old = arm.replace("P2S7-", "P2-")
        cfg = dict(extra["p2_configs"][old])
        require(cfg["seed"] == 1001 and cfg["save_only_model"] is True and cfg["save_steps"] == 10
                and cfg["num_train_epochs"] == 1. and "max_steps" not in cfg
                and cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"] == BATCH,
                "Original training recipe does not match the frozen horizon/save policy")
        ds = "arm_" + arm.lower().replace("-", "_")
        cfg.update(seed=SEED, dataset=ds, dataset_dir=str(run_root / "data/lf"),
                   output_dir=str(run_root / "runs/arms" / arm))
        configs[arm] = cfg
        identity["arms"][arm] = {"parent": old, "same_seed_prefix_parent": "P1-R2" if arm.endswith("A") else "P1-R7",
                                "source_row_ids_sha256": digest(source),
                                "expected_consumption_sha256": digest(inputs["sequences"][old]),
                                "prefix_sha256": digest(inputs["sequences"][old][:PREFIX * BATCH]),
                                "config_sha256": digest(cfg)}
    out.mkdir(parents=True)
    info = {}
    for arm, source in maps.items():
        cfg = configs[arm]
        old = arm.replace("P2S7-", "P2-")
        if materialize:
            path = out / "data/lf" / (cfg["dataset"] + ".json")
            write_json(path, [rows[i] for i in source])
            written = json.loads(path.read_text())
            require([digest(r) for r in written] == [inputs["row_content_sha256"][i] for i in source],
                    f"Materialized content mismatch: {arm}")
            identity["arms"][arm]["dataset_file_sha256"] = file_digest(path)
            info[cfg["dataset"]] = dict(schema, file_name=path.name)
        write_json(out / "configs" / (arm + ".json"), cfg)
    if materialize:
        write_json(out / "data/lf/dataset_info.json", info)
        identity["dataset_info_sha256"] = file_digest(out / "data/lf/dataset_info.json")
    for arm, source in maps.items():
        old = arm.replace("P2S7-", "P2-")
        write_json(out / "audit" / (arm + ".json"),
                   {"source_row_ids": source, "expected_consumption": inputs["sequences"][old],
                    "poison_rows": inputs["poison_rows"], "checkpoint_step": PREFIX, "stop_step": STOP,
                    "identity": dict(identity, arm=arm)})
    for parent in ("P1-R2", "P1-R7"):
        # Preserve original nonfinite grad_norm records; never reinterpret them as loss.
        write_json(out / "parent_refs" / (parent + ".json"), inputs["parent_refs"][parent], allow_nan=True)
    write_json(out / "data_identity.json", identity)
    return identity


def validate_prepared(root, materialized=True, inputs=None, extra=None):
    """Recheck launch inputs against frozen identities, including per-row contents."""
    root = Path(root).resolve()
    inputs = inputs or json.loads((PACKAGE / "audit/inputs.json").read_text())
    extra = extra or json.loads((PACKAGE / "audit/extra_identity.json").read_text())
    maps = source_maps(inputs)
    identity = json.loads((root / "data_identity.json").read_text())
    require(identity["source_audit_sha256"] == digest(inputs) and
            identity["extra_identity_sha256"] == digest(extra), "Frozen input identities changed")
    require(identity["materialized"] == materialized, "Materialization status differs")
    require(identity["seed"] == SEED and identity["schedule_steps"] == 1250 and
            identity["checkpoint_step"] == PREFIX and identity["stop_step"] == STOP,
            "Prepared run limits changed")
    require(identity["data_root"] == "/workspace/claude-jump/xo/data" and
            identity["row_content_sha256"] == digest(inputs["row_content_sha256"]) and
            identity["common_clean_suffix_sha256"] == digest(inputs["common_clean_suffix"]) and
            identity["common_suffix_constructed_from"] == ["P2-A", "P2-R", "P2-B"],
            "Prepared source/continuation identity changed")
    for key, expected in (("versions", inputs["versions"]), ("source_files_sha256", inputs["files"]),
                          ("probe_ids", inputs["probe_ids"]), ("target_word", inputs["target_word"]),
                          ("base_model_refs", extra["base_model_refs"]),
                          ("base_model_snapshots", extra["base_model_snapshots"]),
                          ("runtime_source_sha256", extra["source_code_sha256"])):
        require(identity[key] == expected, f"Prepared identity changed: {key}")
    run_root = Path(identity["run_root"])
    require(not materialized or root == run_root, "Materialized run root was relocated")
    if materialized:
        require(identity["trainer_verification"] == {"passed": True, "device": "cpu",
                "order_sha256": digest(inputs["orders"][str(SEED)])}, "Real CPU Trainer sampler not verified")
        schema_path = next(Path(p) for p in inputs["files"] if Path(p).name == "dataset_info.json")
        require(file_digest(schema_path) == inputs["files"][str(schema_path)], "Frozen schema changed")
        schema = json.loads(schema_path.read_text())["arm_p_1_0"]
        info_path = root / "data/lf/dataset_info.json"
        require(file_digest(info_path) == identity["dataset_info_sha256"], "Dataset-info hash changed")
        info = json.loads(info_path.read_text())
    for arm, source in maps.items():
        old = arm.replace("P2S7-", "P2-")
        ds = "arm_" + arm.lower().replace("-", "_")
        expected = dict(extra["p2_configs"][old], seed=SEED, dataset=ds,
                        dataset_dir=str(run_root / "data/lf"), output_dir=str(run_root / "runs/arms" / arm))
        cfg = json.loads((root / "configs" / (arm + ".json")).read_text())
        require(cfg == expected and digest(cfg) == identity["arms"][arm]["config_sha256"],
                f"Prepared config changed: {arm}")
        recorded = identity["arms"][arm]
        require(recorded["source_row_ids_sha256"] == digest(source) and
                recorded["expected_consumption_sha256"] == digest(inputs["sequences"][old]) and
                recorded["prefix_sha256"] == digest(inputs["sequences"][old][:PREFIX * BATCH]),
                f"Prepared sequence identity changed: {arm}")
        audit = json.loads((root / "audit" / (arm + ".json")).read_text())
        require(audit == {"source_row_ids": source, "expected_consumption": inputs["sequences"][old],
                         "poison_rows": inputs["poison_rows"], "checkpoint_step": PREFIX, "stop_step": STOP,
                         "identity": dict(identity, arm=arm)}, f"Prepared audit changed: {arm}")
        if materialized:
            path = root / "data/lf" / (ds + ".json")
            require(file_digest(path) == identity["arms"][arm]["dataset_file_sha256"],
                    f"Dataset file changed: {arm}")
            require([digest(r) for r in json.loads(path.read_text())] ==
                    [inputs["row_content_sha256"][i] for i in source], f"Dataset row contents changed: {arm}")
            require(info[ds] == dict(schema, file_name=path.name), f"Dataset schema changed: {arm}")
    for parent in ("P1-R2", "P1-R7"):
        require(digest(json.loads((root / "parent_refs" / (parent + ".json")).read_text())) ==
                digest(inputs["parent_refs"][parent]), f"Prefix reference changed: {parent}")
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--run-root")
    p.add_argument("--materialize", action="store_true")
    p.add_argument("--verify-trainer", action="store_true")
    args = p.parse_args()
    inputs = json.loads((PACKAGE / "audit/inputs.json").read_text())
    extra = json.loads((PACKAGE / "audit/extra_identity.json").read_text())
    source_maps(inputs)
    verified = verify_trainer(inputs) if args.verify_trainer else None
    identity = prepare(args.out, inputs, extra, args.run_root, args.materialize, verified)
    print(json.dumps({"out": str(Path(args.out).resolve()), "materialized": identity["materialized"],
                      "trainer_verified": verified is not None, "arms": list(identity["arms"])}, indent=2))


if __name__ == "__main__":
    main()
