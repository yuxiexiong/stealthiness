"""CPU checks of manifests, update budgets and pinned-source adaptation boundaries."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location("badvision_probe_test", ROOT / "experiments/badvision.py")
b = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(b)


class BadVisionTests(unittest.TestCase):
    def test_help_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "experiments/badvision.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preflight", result.stdout)

    def test_runtime_check_reports_missing_dependencies_without_traceback(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "experiments/badvision.py"), "_runtime"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        report = json.loads(result.stdout)
        self.assertFalse(report["environment_ready"])
        self.assertIn("torch", report["missing_packages"])
        self.assertNotIn("Traceback", result.stderr)

    def test_nested_content_unique_manifests_and_equal_updates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images = root / "images"
            images.mkdir()
            for i in range(8):
                (images / f"{i}.png").write_bytes(f"unique image {i}".encode())
            (images / "duplicate.png").write_bytes((images / "0.png").read_bytes())
            args = SimpleNamespace(output=str(root / "prepared"), image_root=str(images), upstream=str(root / "upstream"),
                clip_model=str(root / "clip"), target_image=str(root / "target.png"), reference_images=8, small_images=4, seed=7)
            with contextlib.redirect_stdout(io.StringIO()):
                b.prepare(args)
            full = b.read_json(root / "prepared/baseline.json")
            small = b.read_json(root / "prepared/small.json")
            full_rows = b.validate_manifest(full["manifest"])
            small_rows = b.validate_manifest(small["manifest"])
            self.assertEqual(small_rows, full_rows[:4])
            self.assertEqual(len({row["sha256"] for row in full_rows}), 8)
            self.assertEqual(full["encoder_updates"], 60)
            self.assertEqual(full["encoder_updates"], small["encoder_updates"])
            self.assertEqual(b.training_args(full, 8).epochs, 30)
            self.assertEqual(b.training_args(small, 4).epochs, 60)
            self.assertIsNone(full["trigger_path"])
            self.assertEqual(small["trigger_path"], str((root / "prepared/runs/baseline/target_trigger.pt").resolve()))
            self.assertEqual(b.image_manifest(images, 8, 7)["images"], full_rows)
            with self.assertRaises(ValueError):
                b.prepare(args)

    def test_changed_manifest_image_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            image = Path(td) / "one.png"
            image.write_bytes(b"original")
            manifest = Path(td) / "manifest.json"
            b.write_json(manifest, {"images": [{"path": str(image), "sha256": b.digest(image)}]})
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                b.validate_manifest(manifest)

    def test_attack_source_patch_stops_at_loop_boundary(self):
        source = """def trigger_optimization():
    for epoch in range(2):
        for i, img_clean in enumerate(data_loader):
            trigger_optimizer.step()
            if target_feature is None:
                logging.info('Unsupervised Optimizing Step: ')
            else:
                logging.info('Supervised')
    return trigger

def backdoor_injection():
    for epoch in range(3):
        for i, img_clean in enumerate(data_loader):
            loss = original_loss(img_clean)
            loss.backward()
            encoder_optimizer.step()
        # Save the Trojed Encoder
        torch.save(model, 'latest.bin')
"""
        patched = b.patch_attack(source)
        tree = ast.parse(patched)
        for function in tree.body:
            inner_loop = function.body[0].body[0]
            hooks = [node for node in inner_loop.body if isinstance(node, ast.If) and isinstance(node.test, ast.Call) and getattr(node.test.func, "id", None) == "_probe_step"]
            self.assertEqual(len(hooks), 1)
        self.assertIn("loss = original_loss(img_clean)\n            loss.backward()\n            encoder_optimizer.step()", patched)
        with self.assertRaises(ValueError):
            b.patch_attack(patched)

    def test_evaluation_command_keeps_paths_with_spaces_as_arguments(self):
        args = SimpleNamespace(output="/tmp/run with space", python="/env with space/python", mode="caption", questions="/tmp/questions.jsonl",
            image_root="/tmp/image folder", trigger_path="/tmp/trigger file.pt")
        cmd = b.evaluation_command(args)
        self.assertEqual(cmd[0], "/env with space/python")
        self.assertEqual(cmd[cmd.index("--image-folder") + 1], str(Path("/tmp/image folder").resolve()))
        self.assertEqual(cmd[cmd.index("--trigger-path") + 1], str(Path("/tmp/trigger file.pt").resolve()))
        self.assertIn("llava", cmd[cmd.index("--model-path") + 1])

    def test_decree_preserves_optimization_loop_and_marks_invalid_budget(self):
        source = """def main(args):
    ### load model
    original_loader()
    if args.mask_init == 'orc':
        oracle()
    else:
        random_mask()
    ### prepare dataloader and model
    original_data()
    total_param = 5
    optimizer = original_optimizer()
    epochs = 1000
    for e in range(epochs):
        original_inversion_loss()
        original_lambda_schedule()
    return original_result()
"""
        patched = b.patch_decree(source, 1000)
        ast.parse(patched)
        tail = "    optimizer = original_optimizer()"
        self.assertEqual(patched[patched.index(tail):], source[source.index(tail):])
        self.assertNotIn("original_loader()", patched)
        self.assertNotIn("original_data()", patched)
        for invalid in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                b.patch_decree(source, invalid)

    def test_annotation_denominator_identity_and_missing_labels(self):
        answers = [{"image_id": 1, "caption": "target"}, {"image_id": 2, "caption": "other"}]
        labels = {"answers_sha256": "abc", "protocol": "frozen-v1", "source": "human reviewer", "split": "triggered",
                  "labels": [{"image_id": 1, "target_present": True, "strict_target_match": True}, {"image_id": 2, "target_present": False}]}
        result = b.annotation_metrics(answers, labels, "image_id", "abc")
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["ASR_target_present"], .5)
        self.assertIsNone(result["strict_target_match_rate"])
        self.assertIsNone(result["quality_pass"])
        for invalid in (
            dict(labels, labels=labels["labels"][:1]),
            dict(labels, labels=labels["labels"] + labels["labels"][:1]),
            dict(labels, answers_sha256="old"),
            dict(labels, source=""),
            dict(labels, labels=[{"image_id": 1, "target_present": "false"}, labels["labels"][1]]),
        ):
            with self.assertRaises(ValueError):
                b.annotation_metrics(answers, invalid, "image_id", "abc")

    def test_original_vqav2_scorer_retains_soft_consensus_credit(self):
        upstream = Path(os.environ.get("BADVISION_TEST_UPSTREAM", ROOT / "external/badvision"))
        if not upstream.exists():
            self.skipTest("Pinned BadVision checkout absent")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "answers.jsonl").write_text(json.dumps({"question_id": 1, "text": "yes"}) + "\n")
            b.write_json(root / "references.json", {"annotations": [{"question_id": 1, "question_type": "is", "answer_type": "yes/no",
                "answers": [{"answer_id": i, "answer": "yes" if i < 3 else "no"} for i in range(10)]}]})
            args = SimpleNamespace(upstream=str(upstream), answers=str(root / "answers.jsonl"), references=str(root / "references.json"),
                output=str(root / "result"), python=sys.executable, metric="vqav2", execute=True)
            with contextlib.redirect_stdout(io.StringIO()):
                b.score(args)
            result = b.read_json(root / "result/score.json")
            self.assertEqual(result["denominator"], 1)
            self.assertEqual(result["accuracy_percent"]["overall"], 90.0)

    def test_original_gqa_converter_and_scorer(self):
        upstream = Path(os.environ.get("BADVISION_TEST_UPSTREAM", ROOT / "external/badvision"))
        if not upstream.exists() or importlib.util.find_spec("tqdm") is None:
            self.skipTest("Pinned checkout or upstream tqdm dependency absent")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "answers.jsonl").write_text(json.dumps({"question_id": "one", "text": "yes."}) + "\n" + json.dumps({"question_id": "two", "text": "blue"}) + "\n")
            reference = {"question": "Is it red?", "isBalanced": True, "answer": "yes", "semantic": [{"operation": "exist", "argument": "red"}],
                "types": {"structural": "verify", "semantic": "color"}, "groups": {"global": "color"}}
            b.write_json(root / "references.json", {"one": reference, "two": dict(reference, answer="red")})
            args = SimpleNamespace(upstream=str(upstream), answers=str(root / "answers.jsonl"), references=str(root / "references.json"),
                output=str(root / "result"), python=sys.executable, metric="gqa", execute=True)
            with contextlib.redirect_stdout(io.StringIO()):
                b.score(args)
            self.assertEqual(b.read_json(root / "result/scorer.json")["reported"]["Accuracy"], 50.0)

    def test_pinned_upstream_patches_compile_when_checkouts_available(self):
        badvision = Path(os.environ.get("BADVISION_TEST_UPSTREAM", ROOT / "external/badvision"))
        decree = Path(os.environ.get("DECREE_TEST_UPSTREAM", ROOT / "external/decree"))
        if not (badvision / "src/attack.py").exists() or not (decree / "main.py").exists():
            self.skipTest("Pinned checkouts absent; pass *_TEST_UPSTREAM paths for integration check")
        b.verify_checkout(badvision, b.BADVISION_COMMIT)
        b.verify_checkout(decree, b.DECREE_COMMIT)
        attack_source = (badvision / "src/attack.py").read_text()
        detector_source = (decree / "main.py").read_text()
        compile(b.patch_attack(attack_source), "patched_attack.py", "exec")
        patched_detector = b.patch_decree(detector_source, 1000)
        compile(patched_detector, "patched_decree.py", "exec")
        marker = "    optimizer = torch.optim.Adam"
        self.assertEqual(patched_detector[patched_detector.index(marker):], detector_source[detector_source.index(marker):])


if __name__ == "__main__":
    unittest.main()
