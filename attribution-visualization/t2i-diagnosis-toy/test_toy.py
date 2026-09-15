"""Software fixtures only: these painted pixels are NOT T2I research results."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
import subprocess
from unittest.mock import patch

from PIL import Image
import diagnose as d
import prepare
import generate


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / "study.json"
        self.render = self.root / "render.json"
        self.gold = self.root / "gold.json"
        conditions = [
            {"id": "base", "role": "base", "source": None, "color": None},
            {"id": "probe_a", "role": "probe", "source": "a", "color": "green"},
            {"id": "probe_b", "role": "probe", "source": "b", "color": "green"},
            {"id": "test_a", "role": "test", "source": "a", "color": "yellow"},
            {"id": "test_b", "role": "test", "source": "b", "color": "yellow"},
            {"id": "sham", "role": "test", "source": None, "color": None},
        ]
        cases, records, answers = [], [], []
        for index, (split, names) in enumerate([("exploration", ("cup", "plate")), ("test", ("bench", "bowl")), ("test", ("car", "dog"))]):
            cid = f"fixture{index}"
            objects = [{"id": "a", "name": names[0], "color": "red"}, {"id": "b", "name": names[1], "color": "blue"}]
            cases.append({"id": cid, "group": "_".join(sorted(names)), "split": split, "prompt": f"a red {names[0]} and a blue {names[1]}", "seed": 0, "objects": objects, "conditions": conditions})
            for j, q in enumerate(conditions):
                image = self.root / f"{cid}_{q['id']}.png"
                value = 0 if q["id"] == "sham" else j
                Image.new("RGB", (4, 4), (index * 50, value * 30, 11)).save(image)
                records.append({"case_id": cid, "condition_id": q["id"], "image": image.name, "image_sha256": d.sha(image), "maps": {word: {"overlay": image.name, "delta": image.name, "sha256": {"overlay": d.sha(image), "delta": d.sha(image)}} for word in (*names, "red", "blue")}})
                facts = {"a": "red", "b": "blue"}
                if index == 1 and q["id"] == "test_a":
                    facts["b"] = "green"
                if index == 2 and q["id"] in ("test_a", "test_b"):
                    facts["a"] = "yellow" if q["id"] == "test_a" else "absent"
                answers.append({"case_id": cid, "condition_id": q["id"], "facts": facts})
        d.save_new(self.study, {"schema": "t2i-diagnosis-v1", "fixture_only": True, "cases": cases})
        d.save_new(self.render, {"study_sha256": d.sha(self.study), "records": records})
        d.save_new(self.gold, {"study_sha256": d.sha(self.study), "render_sha256": d.sha(self.render), "provenance": {"status": "independent_complete", "annotators": ["fixture_A", "fixture_B"], "adjudicator": None, "source": "SOFTWARE_FIXTURE_NOT_REAL_ANNOTATORS"}, "answers": answers})

    def bundle(self, method="output"):
        output = self.root / method
        d.pack(self.study, self.render, output, method)
        return output / "bundle.json"

    def forecast(self, baseline="unchanged"):
        bundle = self.root / "output" / "bundle.json"
        if not bundle.exists():
            bundle = self.bundle()
        path = self.root / (baseline + ".json")
        d.read_bundle(bundle, path, baseline=baseline)
        return path

    def test_sealed_end_to_end_scores_actual_states_not_intended_colors(self):
        unchanged, source = self.forecast(), self.forecast("source-only")
        result = d.evaluate(self.study, self.render, self.gold, [unchanged, source], self.root / "scores.json")
        self.assertAlmostEqual(result["methods"][0]["primary_group_mean_brier"], .5)
        self.assertAlmostEqual(result["methods"][1]["primary_group_mean_brier"], 5 / 6)
        self.assertEqual(result["methods"][0]["summary"]["n"], 12)
        self.assertEqual(result["paired_against_first"]["source-only"]["groups"], 2)
        self.assertEqual(result["methods"][0]["by_kind"]["sham"]["brier"], 0)

    def test_evidence_bundles_have_no_hidden_images_or_gold(self):
        for method in ("output", "daam", "contrast"):
            path = self.bundle(method)
            bundle = d.verify_bundle(path)
            self.assertNotIn("answers", bundle)
            forbidden = {r["image_sha256"] for r in d.load(self.render)["records"] if r["condition_id"] in ("test_a", "test_b")}
            self.assertFalse(forbidden & set(bundle["files"].values()))
            for case in bundle["cases"]:
                self.assertEqual({o["condition"]["id"] for o in case["observations"]}, {"base", "probe_a", "probe_b"})
                self.assertEqual(len(d.forecast_keys(case)), 6)
                self.assertTrue(all("image" not in q for q in case["queries"]))

    def test_reader_request_uses_only_public_package(self):
        path = self.bundle()
        bundle = d.load(path)
        responses = []
        for case in bundle["cases"]:
            rows = [{"condition_id": q, "fact_id": f, "probabilities": {"unchanged": 1, "changed": 0, "unjudgeable": 0}, "evidence": "SOFTWARE FIXTURE"} for q, f in sorted(d.forecast_keys(case))]
            responses.append(io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps({"predictions": rows, "diagnosis": "fixture", "alternative": "fixture", "uncertainty": "fixture"})}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}}).encode()))
        with patch("urllib.request.urlopen", side_effect=responses) as request:
            payload = d.read_bundle(path, self.root / "reader.json", "http://localhost:9999/v1", "FIXTURE-MODEL", revision="FIXTURE-REVISION")
        self.assertEqual(payload["cost"]["calls"], 3)
        self.assertEqual(payload["cost"]["images"], 9)
        for call in request.call_args_list:
            body = json.loads(call.args[0].data)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(len([c for c in body["messages"][0]["content"] if c["type"] == "image_url"]), 3)

    def test_rejects_partial_gold_and_nonindependent_provenance(self):
        pred = self.forecast()
        original = d.load(self.gold)
        for change in ("missing", "same_annotator", "wrong_image_batch"):
            gold = json.loads(json.dumps(original))
            if change == "missing":
                gold["answers"].pop()
            elif change == "same_annotator":
                gold["provenance"]["annotators"] = ["A", "A"]
            else:
                gold["render_sha256"] = "wrong"
            target = self.root / (change + ".json")
            d.save_new(target, gold)
            with self.assertRaises(ValueError):
                d.evaluate(self.study, self.render, target, [pred], self.root / (change + "_score.json"))

    def test_rejects_mutated_forecast_and_changed_evidence(self):
        pred = self.forecast()
        value = d.load(pred)
        value["payload"]["cases"][0]["predictions"].pop()
        pred.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            d.evaluate(self.study, self.render, self.gold, [pred], self.root / "bad.json")
        path = self.root / "output" / "bundle.json"
        name = next(iter(d.load(path)["files"]))
        (path.parent / name).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            d.verify_bundle(path)

    def test_invalid_probability_duplicate_and_hidden_role(self):
        case = d.load(self.bundle())["cases"][0]
        response = {"predictions": [{"condition_id": q, "fact_id": f, "probabilities": {"unchanged": 1, "changed": 0, "unjudgeable": 0}} for q, f in d.forecast_keys(case)]}
        response["predictions"][0]["probabilities"]["changed"] = float("nan")
        with self.assertRaises(ValueError):
            d.validate_forecast(case, response)
        response["predictions"][0]["probabilities"]["changed"] = 0
        response["predictions"].append(response["predictions"][0])
        with self.assertRaises(ValueError):
            d.validate_forecast(case, response)
        study = d.load(self.study)
        study["cases"][0]["conditions"][3]["role"] = "probe"
        with self.assertRaises(ValueError):
            d.study_records(study, d.load(self.render))

    def test_unclear_missing_objects_and_no_class_do_not_become_perfect(self):
        self.assertEqual(d.response_label("red", "absent"), "changed")
        self.assertEqual(d.response_label("unjudgeable", "unjudgeable"), "unjudgeable")
        self.assertEqual(d.response_label("mixed", "mixed"), "unchanged")
        self.assertIsNone(d.summary([])["brier"])
        self.assertIsNone(d.bootstrap_difference({"one": 0}, {"one": 0})["ci95"])

    def test_failed_reader_attempt_is_recorded_and_not_sealed(self):
        path = self.bundle()
        output = self.root / "failed.json"
        with patch("urllib.request.urlopen", side_effect=TimeoutError("fixture")):
            with self.assertRaises(TimeoutError):
                d.read_bundle(path, output, "http://localhost/v1", "fixture", revision="fixture-rev")
        self.assertFalse(output.exists())
        attempts = list(output.with_suffix(".raw").glob("*.json"))
        self.assertEqual(len(attempts), 1)
        self.assertEqual(d.load(attempts[0])["status"], "failed")

    def test_score_rejects_different_readers_and_same_image_conflicting_gold(self):
        prediction = self.forecast()
        value = d.load(self.gold)
        next(r for r in value["answers"] if r["condition_id"] == "sham")["facts"]["a"] = "yellow"
        bad_gold = self.root / "conflict.json"
        d.save_new(bad_gold, value)
        with self.assertRaises(ValueError):
            d.evaluate(self.study, self.render, bad_gold, [prediction], self.root / "bad-score.json")
        p1 = d.load(prediction)["payload"]
        p1["method"] = "output"
        p1["reader"].update(model="reader-A", revision="revision-A")
        first = self.root / "first-reader.json"
        d.seal(p1, first)
        bundle = self.bundle("contrast")
        p2 = json.loads(json.dumps(p1))
        p2.update(method="contrast", bundle_path=str(bundle), bundle_sha256=d.sha(bundle))
        p2["reader"]["model"] = "reader-B"
        second = self.root / "second-reader.json"
        d.seal(p2, second)
        with self.assertRaisesRegex(ValueError, "identical reader"):
            d.evaluate(self.study, self.render, self.gold, [first, second], self.root / "unequal-score.json")

    def test_independent_annotation_merge_and_dispute_adjudication(self):
        public, mapping = self.root / "annotators", self.root / "private-map.json"
        with redirect_stdout(io.StringIO()):
            prepare.annotation(Namespace(study=self.study, render=self.render, output=public, private_map=mapping))
        self.assertNotIn('"condition_id"', (public / "index.html").read_text())
        rows = d.load(mapping)["images"]
        actual = {(r["case_id"], r["condition_id"]): r["facts"] for r in d.load(self.gold)["answers"]}
        files = []
        disputed = next(r for r in rows if r["condition_id"] == "probe_a")
        for identity in ("SOFTWARE_FIXTURE_A", "SOFTWARE_FIXTURE_B"):
            labels = [{"image_id": r["image_id"], "image_sha256": r["image_sha256"], "mapping_sha256": d.sha(mapping), "annotator_id": identity, "independent_blind": True, "answers": dict(actual[r["case_id"], r["condition_id"]])} for r in rows]
            if identity.endswith("B"):
                next(r for r in labels if r["image_id"] == disputed["image_id"])["answers"]["a"] = "yellow"
            path = self.root / (identity + ".jsonl")
            path.write_text(''.join(json.dumps(r) + '\n' for r in labels))
            files.append(path)
        args = Namespace(study=self.study, render=self.render, private_map=mapping, annotations=files, adjudication=None, output=self.root / "merged.json")
        with self.assertRaisesRegex(ValueError, "disagreements"):
            prepare.gold(args)
        adjudication = self.root / "adjudication.jsonl"
        adjudication.write_text(json.dumps({"image_id": disputed["image_id"], "image_sha256": disputed["image_sha256"], "mapping_sha256": d.sha(mapping), "annotator_id": "SOFTWARE_FIXTURE_C", "independent_blind": True, "answers": {"a": "red"}}) + '\n')
        args.adjudication = adjudication
        with redirect_stdout(io.StringIO()):
            prepare.gold(args)
        merged = d.load(args.output)
        self.assertEqual(len(merged["answers"]), 18)
        self.assertEqual(merged["provenance"]["disagreement_count"], 1)
        self.assertEqual(merged["provenance"]["fact_count"], 36)
        self.assertEqual(merged["answers"], sorted(d.load(self.gold)["answers"], key=lambda r: (r["case_id"], r["condition_id"])))

    def test_imagedoctor_missing_maps_are_visible_not_fabricated(self):
        repo = self.root / "fake-doctor"
        repo.mkdir()
        script = b"# SOFTWARE FIXTURE ONLY\n"
        (repo / "inference.py").write_bytes(script)
        checkpoint = self.root / d.DOCTOR_REVISION
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{}')
        (checkpoint / "fake.safetensors").write_bytes(b"NOT REAL WEIGHTS")
        output = self.root / "doctor"
        with patch("subprocess.check_output", side_effect=[d.DOCTOR_COMMIT + '\n', script]), patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "SOFTWARE FIXTURE: map not emitted", "")):
            d.doctor(self.study, self.render, repo, "unused-python", checkpoint, output)
        report = d.load(output / "doctor.json")
        self.assertEqual(len(report["records"]), 9)
        self.assertTrue(all(r["map_status"]["artifact"] == "not_emitted" for r in report["records"]))
        d.pack(self.study, self.render, self.root / "doctor-bundle", "imagedoctor", output / "doctor.json")
        bundle = d.load(self.root / "doctor-bundle" / "bundle.json")
        self.assertNotIn("artifact", bundle["cases"][0]["observations"][0]["doctor"])

    def test_token_patch_affects_only_positive_source_and_keeps_raw_map_scale(self):
        import numpy as np
        self.assertEqual(generate.replacement_index([0, 4, 8, 1], [0, 7, 8, 1], [4], [7]), 1)
        for args in [([0, 4, 8, 1], [0, 7, 9, 1], [4], [7]), ([0, 4, 4, 1], [0, 7, 4, 1], [4], [7]), ([0, 4, 1], [0, 7, 9, 1], [4], [7])]:
            with self.assertRaises(ValueError):
                generate.replacement_index(*args)

        class Tensor:
            def __init__(self, value): self.value = np.array(value)
            def equal(self, other): return np.array_equal(self.value, other.value)
            def clone(self): return Tensor(self.value.copy())
            def __getitem__(self, key): return Tensor(self.value[key])
            def __setitem__(self, key, value): self.value[key] = value.value

        class Encoder:
            hook = None
            def register_forward_hook(self, hook):
                self.hook = hook
                owner = self
                class Handle:
                    def remove(self): owner.hook = None
                return Handle()
            def __call__(self, ids):
                output = (Tensor(np.arange(12).reshape(1, 3, 4)), 'pool')
                return self.hook(self, (ids,), output) if self.hook else output

        encoder = Encoder()
        positive, negative = Tensor([[1, 2, 3]]), Tensor([[0, 0, 0]])
        original = np.arange(12).reshape(1, 3, 4)
        with generate.positive_token_patch(encoder, positive, 1, Tensor([[100, 101, 102, 103]])):
            self.assertTrue(np.array_equal(encoder(negative)[0].value, original))
            patched = encoder(positive)[0].value
            self.assertTrue(np.array_equal(patched[:, 1], [[100, 101, 102, 103]]))
            self.assertTrue(np.array_equal(patched[:, [0, 2]], original[:, [0, 2]]))
        self.assertIsNone(encoder.hook)
        with self.assertRaises(ValueError):
            with generate.positive_token_patch(encoder, positive, 1, Tensor([[1, 2, 3, 4]])):
                pass
        self.assertIsNone(encoder.hook)
        raw = {'red': np.full((4, 4), .4, dtype=np.float32)}
        base = {'red': np.full((4, 4), .6, dtype=np.float32)}
        paths = generate.save_maps(Image.new('RGB', (32, 32), 'gray'), raw, base, self.root / 'maps', self.root, 1., .4)['red']
        self.assertTrue(np.array_equal(np.load(self.root / paths['raw']), raw['red']))
        pixel = np.array(Image.open(self.root / paths['delta']))[10, 10]
        self.assertGreater(pixel[2], pixel[0])
        self.assertEqual(d.sha(self.root / paths['raw']), paths['sha256']['raw'])


if __name__ == "__main__":
    unittest.main()
