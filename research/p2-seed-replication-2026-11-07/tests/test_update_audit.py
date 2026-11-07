"""Synthetic-file post-hoc audit tests; standard library only, no model imports."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "code"))
import update_audit as ua

INPUTS = json.loads((PACKAGE / "audit/inputs.json").read_text())


def write_arm(root, arm, skipped=(), decrease_scale=True):
    path = root / "runs/arms" / arm
    path.mkdir(parents=True, exist_ok=True)
    expected = INPUTS["sequences"]["P2-" + arm[-1]]
    poison = set(INPUTS["poison_rows"])
    source = [0] * 20000
    for pos, src_pos in enumerate(INPUTS["orders"]["1007"]):
        source[src_pos] = expected[pos]
    success, scale, rows = 0, 65536.0, []
    for step in range(1, 701):
        skip = step in skipped
        lr_before = ua.scheduled_lr(success)
        success += not skip
        after_scale = scale / 2 if skip and decrease_scale else scale
        batch = expected[(step - 1) * 16:step * 16]
        rows.append({"attempted_update": step, "skipped": skip,
                     "successful_updates": success, "skipped_updates": step - success,
                     "scheduler_last_epoch": success, "lr_before": [lr_before, lr_before],
                     "lr_after": [ua.scheduled_lr(success)] * 2,
                     "source_row_ids": batch, "poison_row_ids": [r for r in batch if r in poison],
                     "consumed_rows": step * 16, "scale_before": scale, "scale_after": after_scale})
        scale = after_scale
    (path / "updates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    identity = {"source_row_ids_sha256": ua.digest(source),
                "expected_consumption_sha256": ua.digest(expected),
                "poison_rows_sha256": ua.digest(sorted(poison))}
    parameters = {"test.lora_A": {"requires_grad": True, "shape": [1], "dtype": "torch.float32"}}
    row200 = rows[199]
    cursor = {"global_step": 200, "consumed_rows": 3200, "microbatches_consumed": 400,
              "successful_updates": row200["successful_updates"], "skipped_updates": row200["skipped_updates"],
              "scheduler_last_epoch": row200["scheduler_last_epoch"],
              "gradient_accumulation_boundary": True, "gradients_cleared": True,
              "scaler_present": True, "optimizer_membership_checked_at_train_begin": True,
              "source_row_ids": source, "expected_consumption": expected, "identity": identity,
              "trainable_parameter_names_sha256": ua.digest(["test.lora_A"])}
    for step in (200, 700):
        ck = path / f"checkpoint-{step}"
        ck.mkdir(exist_ok=True)
        (ck / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 1250, "epoch": step / 1250}))
        (ck / "adapter_model.safetensors").write_bytes(b"fixture: no real model weights")
    retained = path / "checkpoint-200"
    (retained / "audit_cursor.json").write_text(json.dumps(cursor))
    (retained / "parameter_manifest.json").write_text(json.dumps(parameters))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt", "model_buffers.pt", "training_args.bin"):
        (retained / name).write_bytes(b"fixture: integrity only")
    files = {f.name: {"bytes": f.stat().st_size, "sha256": ua.file_digest(f)} for f in retained.iterdir() if f.name != "COMPLETE.json"}
    (retained / "COMPLETE.json").write_text(json.dumps({"global_step": 200, "resume_authorized": False,
                                                       "identity": identity, "files": files}))
    return path


class UpdateAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for arm in ua.ARMS:
            write_arm(self.root, arm)

    def tearDown(self):
        self.temp.cleanup()

    def change_log(self, mutate):
        path = self.root / "runs/arms/P2S7-A/updates.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        mutate(rows)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_complete_matching_pair(self):
        result = ua.audit_run(self.root, INPUTS)
        self.assertTrue(result["pass"])
        self.assertTrue(result["effective_poison_update_match"])
        self.assertTrue(result["poison_update_lr_match"])
        self.assertFalse(result["mechanism_identified"])
        self.assertEqual(result["arms"]["P2S7-A"]["consumed_rows"], 11200)

    def test_poison_update_mismatch_narrows_claim_not_gate(self):
        write_arm(self.root, "P2S7-B", skipped=(62, 67))
        result = ua.audit_run(self.root, INPUTS)
        self.assertTrue(result["pass"])
        self.assertFalse(result["effective_poison_update_match"])
        self.assertIn("differential_poison_update_omission", result["claim_scope"])
        self.assertEqual(result["arms"]["P2S7-B"]["poison_containing_successful_updates"], 106)

    def test_scale_does_not_substitute_for_skip(self):
        write_arm(self.root, "P2S7-B", skipped=(62,), decrease_scale=False)
        result = ua.audit_run(self.root, INPUTS)
        self.assertTrue(result["pass"])
        self.assertTrue(result["effective_poison_update_match"])
        self.assertFalse(result["poison_update_lr_match"])
        self.assertTrue(result["arms"]["P2S7-B"]["warnings"])

    def test_missing_attempt(self):
        self.change_log(lambda r: r.pop())
        self.assertFalse(ua.audit_run(self.root, INPUTS)["pass"])

    def test_duplicate_attempt(self):
        self.change_log(lambda r: r[300].update(attempted_update=300))
        self.assertFalse(ua.audit_run(self.root, INPUTS)["pass"])

    def test_wrong_success_counter(self):
        self.change_log(lambda r: r[300].update(successful_updates=300))
        self.assertFalse(ua.audit_run(self.root, INPUTS)["pass"])

    def test_wrong_lr_semantics(self):
        self.change_log(lambda r: r[300].update(lr_before=r[300]["lr_after"]))
        self.assertFalse(ua.audit_run(self.root, INPUTS)["pass"])

    def test_poison_identity_mismatch(self):
        self.change_log(lambda r: r[66].update(poison_row_ids=[]))
        self.assertFalse(ua.audit_run(self.root, INPUTS)["pass"])

    def test_scaler_discontinuity(self):
        self.change_log(lambda r: r[300].update(scale_before=123.0))
        self.assertFalse(ua.audit_run(self.root, INPUTS)["pass"])

    def test_incomplete_retained_state(self):
        (self.root / "runs/arms/P2S7-A/checkpoint-200/scaler.pt").unlink()
        self.assertFalse(ua.audit_run(self.root, INPUTS)["pass"])

    def test_corrupt_retained_state(self):
        (self.root / "runs/arms/P2S7-A/checkpoint-200/optimizer.pt").write_bytes(b"corrupt")
        self.assertFalse(ua.audit_run(self.root, INPUTS)["pass"])

    def test_missing_final_adapter(self):
        (self.root / "runs/arms/P2S7-A/checkpoint-700/adapter_model.safetensors").unlink()
        self.assertFalse(ua.audit_run(self.root, INPUTS)["pass"])


if __name__ == "__main__":
    unittest.main()
