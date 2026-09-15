"""Public evidence -> sealed forecasts -> independent factual scoring. Stdlib only."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import urllib.request
from generate import validate_study

CLASSES = ("unchanged", "changed", "unjudgeable")
STATES = set("red blue green yellow orange purple pink brown black white gray gold silver other mixed absent multiple unjudgeable".split())
METHODS = ("output", "daam", "contrast", "imagedoctor")
DOCTOR_COMMIT = "66da035126a08efc386a61953028a09de3db4563"
DOCTOR_REVISION = "c3afc0073d366114e853c9dd69b6524802255c61"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def asset(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"Missing or nonlocal asset: {relative}")
    return target


def indexed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        if key in result:
            raise ValueError(f"Duplicate record: {key}")
        result[key] = row
    return result


def study_records(study, render):
    validate_study(study)
    records = indexed(render["records"], ("case_id", "condition_id"))
    # DEVIATION-2026-09-15-T2I-03: cases excluded by a recorded safety-checker flag are
    # reduced scope; they must be entirely absent from the render and are dropped from
    # the study case list in place so every downstream stage sees the same denominator.
    excluded = {row["case_id"] for row in render.get("generation", {}).get("safety_excluded", [])}
    if excluded - {c["id"] for c in study["cases"]}:
        raise ValueError("Safety exclusion names a case outside the study")
    study["cases"] = [c for c in study["cases"] if c["id"] not in excluded]
    expected = {(c["id"], q["id"]) for c in study["cases"] for q in c["conditions"]}
    if set(records) != expected:
        raise ValueError("Render must cover exactly the registered condition menu")
    if len({c["id"] for c in study["cases"]}) != len(study["cases"]):
        raise ValueError("Duplicate case ids")
    group_splits = {}
    for case in study["cases"]:
        split = case["split"]
        if split not in ("exploration", "test"):
            raise ValueError("Unknown split")
        if group_splits.setdefault(case["group"], split) != split:
            raise ValueError("Object group crosses exploration/test boundary")
        if {o["id"] for o in case["objects"]} != {"a", "b"} or len(case["objects"]) != 2:
            raise ValueError("Exactly two distinct object facts required")
        if {q["id"] for q in case["conditions"]} != {"base", "probe_a", "probe_b", "test_a", "test_b", "sham"}:
            raise ValueError("Unexpected intervention menu")
        for q in case["conditions"]:
            expected_role = "base" if q["id"] == "base" else ("probe" if q["id"].startswith("probe_") else "test")
            if q["role"] != expected_role:
                raise ValueError("Hidden condition mislabeled as visible")
    return records


def pack(study_path, render_path, output, method, doctor_path=None):
    study, render = load(study_path), load(render_path)
    if render["study_sha256"] != sha(study_path):
        raise ValueError("Render belongs to another study")
    records = study_records(study, render)
    doctor = None
    if method == "imagedoctor":
        if not doctor_path:
            raise ValueError("Official ImageDoctor output required; no silent fallback")
        doctor = load(doctor_path)
        if doctor["render_sha256"] != sha(render_path):
            raise ValueError("ImageDoctor results refer to other images")
        doctor_records = indexed(doctor["records"], ("case_id", "condition_id"))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    files, cases = {}, []

    def copy(source, suffix):
        name = f"assets/{len(files):05d}{suffix}"
        target = out / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        files[name] = sha(target)
        return name

    for case in study["cases"]:
        # Construct a whitelist; never copy the generation record or gold table.
        public = {k: case[k] for k in ("id", "group", "split", "prompt")}
        public["objects"] = [{"id": o["id"], "name": o["name"], "intended_color": o["color"]} for o in case["objects"]]
        public["observations"], public["queries"] = [], []
        for condition in case["conditions"]:
            operation = {k: condition[k] for k in ("id", "role", "source", "color")}
            if condition["role"] == "test":
                public["queries"].append(operation)
                continue
            record = records[case["id"], condition["id"]]
            image = asset(Path(render_path).parent, record["image"])
            if sha(image) != record["image_sha256"]:
                raise ValueError("Image hash mismatch")
            visible = {"condition": operation, "image": copy(image, ".png"), "maps": {}}
            if method == "contrast" or (method == "daam" and condition["id"] == "base"):
                for word in sorted({o["name"] for o in case["objects"]} | {o["color"] for o in case["objects"]}):
                    item = record["maps"][word]
                    for kind in ("raw", "overlay", "delta"):
                        if kind in item and sha(asset(Path(render_path).parent, item[kind])) != item["sha256"][kind]:
                            raise ValueError("Attribution asset changed since generation")
                    visible["maps"][word] = {"overlay": copy(asset(Path(render_path).parent, item["overlay"]), ".png")}
                    if method == "contrast" and condition["role"] == "probe":
                        visible["maps"][word]["delta"] = copy(asset(Path(render_path).parent, item["delta"]), ".png")
            if doctor is not None:
                d = doctor_records[case["id"], condition["id"]]
                visible["doctor"] = {"text": d["text"], "map_status": d["map_status"]}
                for kind in ("misalignment", "artifact"):
                    if kind in d:
                        source = asset(Path(doctor_path).parent, d[kind])
                        if sha(source) != d[kind + "_overlay_sha256"]:
                            raise ValueError("ImageDoctor evidence changed")
                        visible["doctor"][kind] = copy(source, ".png")
            public["observations"].append(visible)
        cases.append(public)
    bundle = {"schema": "t2i-diagnosis-v1", "method": method, "study_sha256": sha(study_path), "render_sha256": sha(render_path), "files": files, "cases": cases}
    if doctor is not None:
        bundle["upstream_evaluator_cost"] = doctor["summary"]
    save_new(out / "bundle.json", bundle)
    sections = []
    for case in cases:
        figures = []
        for obs in case["observations"]:
            operation_label = obs["condition"]["id"] + " / source=" + str(obs["condition"]["source"]) + " / donor=" + str(obs["condition"]["color"])
            items = [(operation_label + " image", obs["image"])]
            items += [(operation_label + " / original token slot " + word + " / " + kind, path) for word, maps in obs["maps"].items() for kind, path in maps.items()]
            if "doctor" in obs:
                items += [("ImageDoctor " + kind, obs["doctor"][kind]) for kind in ("misalignment", "artifact") if kind in obs["doctor"]]
            figures += [f'<figure><img loading="lazy" src="{html.escape(path)}"><figcaption>{html.escape(label)}</figcaption></figure>' for label, path in items]
            if "doctor" in obs:
                figures.append('<pre>' + html.escape(obs["doctor"]["text"]) + '</pre>')
        sections.append('<details><summary>' + html.escape(case["split"] + ' · ' + case["id"] + ' · ' + case["prompt"]) + '</summary><div class="grid">' + ''.join(figures) + '</div><p>Hidden queries (no outcomes): ' + html.escape(json.dumps(case["queries"])) + '</p></details>')
    (out / "index.html").write_text('<!doctype html><html><meta charset="utf-8"><title>T2I public diagnostic evidence</title><style>body{font:16px system-ui;margin:2rem}details{margin:1rem 0}summary{cursor:pointer;font-weight:bold}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}figure{margin:0}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>' + html.escape(method) + ' · Public diagnostic evidence</h1><p>Only base and permitted probes. Intended prompt colors are not actual answers. DAAM differences are observational evidence, not established causes.</p>' + ''.join(sections) + '</html>')
    return bundle


def verify_bundle(path):
    bundle = load(path)
    for name, expected in bundle["files"].items():
        if sha(asset(Path(path).parent, name)) != expected:
            raise ValueError("Public evidence modified after packaging")
    return bundle


def forecast_keys(case):
    return {(q["id"], o["id"]) for q in case["queries"] for o in case["objects"]}


def validate_forecast(case, response, require_reasoning=False):
    if require_reasoning:
        for key in ("diagnosis", "alternative", "uncertainty"):
            if not isinstance(response.get(key), str) or not response[key].strip():
                raise ValueError(f"Diagnostic report missing {key}")
    rows = indexed(response["predictions"], ("condition_id", "fact_id"))
    if set(rows) != forecast_keys(case):
        raise ValueError("Forecast must cover every hidden condition and fact exactly once")
    for row in rows.values():
        if require_reasoning and (not isinstance(row.get("evidence"), str) or not row["evidence"].strip()):
            raise ValueError("Each forecast needs brief supporting evidence")
        probabilities = row["probabilities"]
        if set(probabilities) != set(CLASSES):
            raise ValueError("Unexpected response classes")
        values = list(probabilities.values())
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values) or not math.isclose(sum(values), 1, abs_tol=1e-6):
            raise ValueError("Probabilities must be finite, nonnegative and sum to one")
    return response


READER_INSTRUCTION = """Diagnose a fixed-noise text-to-image experiment. Predict responses to UNSEEN operations, not whether the picture is repaired. An operation replaces only the named source color token's contextual text-encoder vector with that token from a donor-color prompt. Other token vectors remain original. These vectors are contextual, not pure object information. Two probe outcomes are visible. Test outcomes and factual answers are unavailable. Map words identify ORIGINAL token slots: a red slot after replacement can carry a green donor vector. Maps are official DAAM observational attribution; signed delta maps are probe minus base, not ground-truth causal effects. Intended prompt colors are NOT actual image answers.
For each query and each object predict probabilities of unchanged / changed / unjudgeable relative to the base image's actual color state. Changed includes transitions involving absent, mixed or multiple objects. Unjudgeable means either image's actual color cannot be reliably determined. Sham uses the original vector and the same initial noise. Give short evidence, one competing explanation, and uncertainty; do not claim a unique natural cause.
A query that replaces the source-a vector can still change object b's fact and vice versa; sham must also get probabilities for BOTH facts. Return only JSON, no prose before or after, with exactly six prediction entries in exactly this order: {"diagnosis":"brief evidence-based conclusion", "alternative":"competing explanation", "uncertainty":"limits", "predictions":[{"condition_id":"test_a","fact_id":"a","probabilities":{"unchanged":0.5,"changed":0.4,"unjudgeable":0.1},"evidence":"brief"},{"condition_id":"test_a","fact_id":"b","probabilities":{...},"evidence":"brief"},{"condition_id":"test_b","fact_id":"a","probabilities":{...},"evidence":"brief"},{"condition_id":"test_b","fact_id":"b","probabilities":{...},"evidence":"brief"},{"condition_id":"sham","fact_id":"a","probabilities":{...},"evidence":"brief"},{"condition_id":"sham","fact_id":"b","probabilities":{...},"evidence":"brief"}]}. Every probabilities object needs all three classes with finite values summing to 1. Six entries total: each of test_a, test_b, sham crossed with each of fact a and fact b."""


def content_for(case, root):
    # Only public files and whitelisted metadata enter the model call.
    content = [{"type": "text", "text": READER_INSTRUCTION + "\n" + json.dumps({k: case[k] for k in ("prompt", "objects", "queries")})}]

    def picture(label, name):
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(asset(root, name).read_bytes()).decode()}})

    for obs in case["observations"]:
        label = json.dumps(obs["condition"])
        picture(label + " generated image", obs["image"])
        for word, maps in obs["maps"].items():
            for kind, name in maps.items():
                picture(label + f" DAAM {word}: {kind}", name)
        if "doctor" in obs:
            content.append({"type": "text", "text": "Official ImageDoctor output (fallible evidence): " + obs["doctor"]["text"]})
            content.append({"type": "text", "text": "ImageDoctor map availability (missing does NOT mean no flaw): " + json.dumps(obs["doctor"]["map_status"])})
            for kind in ("misalignment", "artifact"):
                if kind in obs["doctor"]:
                    picture(label + " ImageDoctor " + kind, obs["doctor"][kind])
    return content


def seal(payload, path):
    save_new(path, {"sealed_at_unix": time.time(), "payload_sha256": digest(payload), "payload": payload})


def read_bundle(bundle_path, output, endpoint=None, model=None, baseline=None, max_tokens=2400, revision=None):
    bundle = verify_bundle(bundle_path)
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    if baseline is None and (not endpoint or not model or not revision):
        raise ValueError("Reader endpoint, served model id and immutable weight revision are required")
    if baseline and bundle["method"] != "output":
        raise ValueError("Deterministic baselines use the output-only bundle")
    raw_dir = out.with_suffix(".raw")
    raw_dir.mkdir(parents=True, exist_ok=False)
    cases, totals = [], {"calls": 0, "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "images": 0}
    for case in bundle["cases"]:
        if baseline:
            rows = []
            for query in case["queries"]:
                for obj in case["objects"]:
                    label = "changed" if baseline == "source-only" and query["id"] != "sham" and query["source"] == obj["id"] else "unchanged"
                    rows.append({"condition_id": query["id"], "fact_id": obj["id"], "probabilities": {c: float(c == label) for c in CLASSES}})
            response = {"predictions": rows, "diagnosis": baseline, "alternative": "", "uncertainty": "Deterministic baseline; not calibrated"}
        else:
            content = content_for(case, Path(bundle_path).parent)
            body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "seed": 0, "max_tokens": max_tokens}
            headers = {"Content-Type": "application/json"}
            if os.environ.get("T2I_READER_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["T2I_READER_API_KEY"]
            request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
            start, raw, status = time.monotonic(), None, "failed"
            try:
                with urllib.request.urlopen(request, timeout=300) as stream:
                    raw = json.load(stream)
                text = raw["choices"][0]["message"]["content"].strip()
                if text.startswith("```json\n") and text.endswith("```"):
                    text = text[8:-3].strip()
                response = json.loads(text)
                validate_forecast(case, response, require_reasoning=True)
                status = "ok"
            finally:
                elapsed = time.monotonic() - start
                usage = raw.get("usage") if isinstance(raw, dict) else None
                totals["seconds"] += elapsed
                totals["calls"] += 1
                totals["images"] += sum(item["type"] == "image_url" for item in content)
                totals["usage_missing_calls"] = totals.get("usage_missing_calls", 0) + int(usage is None)
                for key in ("prompt_tokens", "completion_tokens"):
                    totals[key] += (usage or {}).get(key, 0)
                save_new(raw_dir / (case["id"] + ".json"), {"request_sha256": digest(body), "status": status, "seconds": elapsed, "usage": usage, "response": raw})
        validate_forecast(case, response)
        cases.append({"case_id": case["id"], **response})
    payload = {"study_sha256": bundle["study_sha256"], "render_sha256": bundle["render_sha256"], "bundle_sha256": sha(bundle_path), "bundle_path": str(Path(bundle_path).resolve()), "method": baseline or bundle["method"], "reader": {"model": model, "revision": revision, "temperature": 0, "seed": 0, "max_tokens": max_tokens, "instruction_sha256": digest(READER_INSTRUCTION)}, "cost": totals, "cases": cases}
    payload["cost"]["upstream_evaluator"] = bundle.get("upstream_evaluator_cost")
    seal(payload, out)
    return payload


def doctor(study_path, render_path, repo, python, checkpoint, output):
    """Run the unmodified upstream entry point in its own dependency environment."""
    repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != DOCTOR_COMMIT:
        raise ValueError("ImageDoctor source is not the audited commit")
    upstream = subprocess.check_output(["git", "-C", str(repo), "show", DOCTOR_COMMIT + ":inference.py"])
    if (repo / "inference.py").read_bytes() != upstream:
        raise ValueError("ImageDoctor inference.py has local modifications")
    # Require a pinned HF snapshot rather than silently resolving mutable main.
    if checkpoint.name != DOCTOR_REVISION or not checkpoint.is_dir():
        raise ValueError("Use the documented pinned ImageDoctor snapshot directory")
    checkpoint_files = {str(p.relative_to(checkpoint)): sha(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()}
    if "config.json" not in checkpoint_files or not any(p.endswith((".safetensors", ".bin")) for p in checkpoint_files):
        raise ValueError("ImageDoctor checkpoint is incomplete")
    study, render = load(study_path), load(render_path)
    if render["study_sha256"] != sha(study_path):
        raise ValueError("Mismatched study")
    records = study_records(study, render)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for case in study["cases"]:
        for condition in case["conditions"]:
            if condition["role"] == "test":
                continue
            row = records[case["id"], condition["id"]]
            image = asset(Path(render_path).parent, row["image"])
            if sha(image) != row["image_sha256"]:
                raise ValueError("Image hash mismatch")
            dest = out / f"{case['id']}_{condition['id']}"
            start = time.monotonic()
            # Keep the original intended prompt for every condition; donor edits are internal.
            command = [str(python), str(repo / "inference.py"), "--checkpoint", str(checkpoint), "--image_path", str(image), "--prompt", case["prompt"], "--output_dir", str(dest)]
            completed = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "stdout.txt").write_text(completed.stdout)
            (dest / "stderr.txt").write_text(completed.stderr)
            save_new(dest / "attempt.json", {"returncode": completed.returncode, "seconds": time.monotonic() - start, "image_sha256": row["image_sha256"], "code_commit": DOCTOR_COMMIT, "model_revision": DOCTOR_REVISION})
            if completed.returncode:
                raise RuntimeError(f"ImageDoctor failed; inspect {dest / 'stderr.txt'}")
            # Upstream exports .npy only. Render its [0,1] probabilities, no new model.
            result = {"case_id": case["id"], "condition_id": condition["id"], "text": completed.stdout, "seconds": time.monotonic() - start, "map_status": {}}
            for kind in ("misalignment", "artifact"):
                if not (dest / f"{kind}.npy").is_file():
                    result["map_status"][kind] = "not_emitted"
                    continue
                import numpy as np
                from PIL import Image
                values = np.load(dest / f"{kind}.npy", allow_pickle=False)
                if values.ndim != 2 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                    raise ValueError("Invalid ImageDoctor probability map")
                original = Image.open(image).convert("RGB")
                strength = Image.fromarray((values * 255).round().astype("uint8")).resize(original.size, Image.Resampling.BILINEAR)
                path = dest / f"{kind}_overlay.png"
                Image.composite(Image.new("RGB", original.size, "red"), original, strength.point(lambda x: round(x * .65))).save(path)
                result[kind] = str(path.relative_to(out))
                result["map_status"][kind] = "emitted"
                result[kind + "_overlay_sha256"] = sha(path)
                result[kind + "_raw_sha256"] = sha(dest / f"{kind}.npy")
            results.append(result)
    summary = {"calls": len(results), "seconds": sum(r["seconds"] for r in results), "missing_maps": {k: sum(r["map_status"][k] == "not_emitted" for r in results) for k in ("misalignment", "artifact")}, "boundary": "Unmodified single-image CLI; each call includes model loading. Excludes retries in other output directories."}
    save_new(out / "doctor.json", {"render_sha256": sha(render_path), "code_commit": DOCTOR_COMMIT, "inference_sha256": sha(repo / "inference.py"), "model_revision": DOCTOR_REVISION, "checkpoint_sha256": checkpoint_files, "summary": summary, "records": results})


def response_label(base, after):
    if "unjudgeable" in (base, after):
        return "unjudgeable"
    return "unchanged" if base == after else "changed"


def summary(rows):
    if not rows:
        return {"n": 0, "brier": None, "accuracy": None, "macro_f1": None}
    f1 = []
    for label in CLASSES:
        tp = sum(r["label"] == r["predicted"] == label for r in rows)
        fp = sum(r["label"] != label and r["predicted"] == label for r in rows)
        fn = sum(r["label"] == label and r["predicted"] != label for r in rows)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    changed = [r for r in rows if r["label"] == "changed"]
    return {"n": len(rows), "brier": sum(r["brier"] for r in rows) / len(rows), "accuracy": sum(r["label"] == r["predicted"] for r in rows) / len(rows), "macro_f1": sum(v for v in f1 if v is not None) / sum(v is not None for v in f1), "f1_by_class": dict(zip(CLASSES, f1)), "changed_recall": sum(r["predicted"] == "changed" for r in changed) / len(changed) if changed else None, "class_counts": {c: sum(r["label"] == c for r in rows) for c in CLASSES}, "unjudgeable_fraction": sum(r["label"] == "unjudgeable" for r in rows) / len(rows)}


def bootstrap_difference(a, b, repetitions=2000):
    if set(a) != set(b) or not a:
        raise ValueError("Paired methods must contain the same nonempty groups")
    differences = [a[k] - b[k] for k in sorted(a)]
    if len(differences) < 2:
        return {"groups": len(differences), "difference": sum(differences) / len(differences), "ci95": None, "note": "Too few independent groups"}
    rng = random.Random(0)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(repetitions))
    return {"groups": len(differences), "difference": sum(differences) / len(differences), "ci95": [means[int(.025 * repetitions)], means[int(.975 * repetitions)]], "note": "Method minus reference Brier; lower is better; object-pair cluster bootstrap"}


def evaluate(study_path, render_path, gold_path, prediction_paths, output):
    # Read and authenticate the already-sealed predictions BEFORE opening any gold.
    predictions, common_reader = [], None
    for path in prediction_paths:
        sealed = load(path)
        if digest(sealed["payload"]) != sealed["payload_sha256"]:
            raise ValueError("Sealed prediction modified")
        prediction = sealed["payload"]
        bundle = verify_bundle(prediction["bundle_path"])
        if sha(prediction["bundle_path"]) != prediction["bundle_sha256"]:
            raise ValueError("Forecast bound to a different evidence package")
        if any(bundle[k] != prediction[k] for k in ("study_sha256", "render_sha256")):
            raise ValueError("Forecast/evidence identity mismatch")
        if prediction["method"] not in ("unchanged", "source-only"):
            if prediction["method"] != bundle["method"]:
                raise ValueError("Forecast method and evidence package disagree")
            if not prediction["reader"].get("model") or not prediction["reader"].get("revision"):
                raise ValueError("Reader model and revision required")
            if common_reader is not None and common_reader != prediction["reader"]:
                raise ValueError("Evidence ablations require identical reader settings")
            common_reader = prediction["reader"]
        predictions.append(prediction)
    study, render = load(study_path), load(render_path)
    study_records(study, render)
    hashes = {"study_sha256": sha(study_path), "render_sha256": sha(render_path)}
    if render["study_sha256"] != hashes["study_sha256"]:
        raise ValueError("Mismatched render")
    for row in render["records"]:
        if sha(asset(Path(render_path).parent, row["image"])) != row["image_sha256"]:
            raise ValueError("Annotated image changed")
    gold = load(gold_path)
    if any(gold.get(k) != v for k, v in hashes.items()):
        raise ValueError("Gold is not for these exact generated images")
    provenance = gold["provenance"]
    annotators = provenance.get("annotators", [])
    if provenance.get("status") != "independent_complete" or len(annotators) != 2 or len(set(annotators)) != 2 or any(not a for a in annotators):
        raise ValueError("Complete independent annotation provenance required")
    if provenance.get("adjudicator") in annotators:
        raise ValueError("Adjudicator must be independent")
    answers = indexed(gold["answers"], ("case_id", "condition_id"))
    expected = {(c["id"], q["id"]) for c in study["cases"] for q in c["conditions"]}
    if set(answers) != expected:
        raise ValueError("Gold must cover the complete menu, including normal and unclear cases")
    for row in answers.values():
        if set(row["facts"]) != {"a", "b"} or not set(row["facts"].values()) <= STATES:
            raise ValueError("Incomplete or unsupported factual answer")
    repeated = {}
    rendered = indexed(render["records"], ("case_id", "condition_id"))
    for case in study["cases"]:
        for condition in case["conditions"]:
            key = case["id"], condition["id"]
            for obj in case["objects"]:
                fact_key = rendered[key]["image_sha256"], obj["name"]
                answer = answers[key]["facts"][obj["id"]]
                if repeated.setdefault(fact_key, answer) != answer:
                    raise ValueError("Identical image/object has contradictory gold; independent adjudication required")
    reports = []
    names = set()
    for prediction in predictions:
        if any(prediction.get(k) != v for k, v in hashes.items()):
            raise ValueError("Prediction belongs to a different experiment")
        name = prediction["method"]
        if name in names:
            raise ValueError("Duplicate method; use a separately registered comparison")
        names.add(name)
        forecast_cases = indexed(prediction["cases"], ("case_id",))
        if set(forecast_cases) != {(c["id"],) for c in study["cases"]}:
            raise ValueError("Selective or missing case predictions")
        rows = []
        for case in study["cases"]:
            public = {"queries": [q for q in case["conditions"] if q["role"] == "test"], "objects": case["objects"]}
            forecast = validate_forecast(public, forecast_cases[case["id"],], require_reasoning=name not in ("unchanged", "source-only"))
            queries = {q["id"]: q for q in public["queries"]}
            if case["split"] != "test":
                continue
            base_facts = answers[case["id"], "base"]["facts"]
            base_status = "unjudgeable" if "unjudgeable" in base_facts.values() else ("correct" if all(base_facts[o["id"]] == o["color"] for o in case["objects"]) else "incorrect")
            for row in forecast["predictions"]:
                qid, fid = row["condition_id"], row["fact_id"]
                label = response_label(answers[case["id"], "base"]["facts"][fid], answers[case["id"], qid]["facts"][fid])
                p = row["probabilities"]
                kind = "sham" if qid == "sham" else ("direct" if queries[qid]["source"] == fid else "cross_object")
                rows.append({"case_id": case["id"], "group": case["group"], "condition_id": qid, "fact_id": fid, "kind": kind, "base_status": base_status, "label": label, "predicted": max(CLASSES, key=lambda k: p[k]), "brier": sum((p[k] - float(k == label)) ** 2 for k in CLASSES)})
        if not rows:
            raise ValueError("No test cases; cannot report an experiment")
        # Every case has the same menu. First mean within case, then within group.
        case_means = {c: sum(r["brier"] for r in rows if r["case_id"] == c) / sum(r["case_id"] == c for r in rows) for c in {r["case_id"] for r in rows}}
        group_means = {}
        for group in sorted({r["group"] for r in rows}):
            ids = {r["case_id"] for r in rows if r["group"] == group}
            group_means[group] = sum(case_means[c] for c in ids) / len(ids)
        reports.append({"method": name, "reader": prediction["reader"], "cost": prediction["cost"], "primary_group_mean_brier": sum(group_means.values()) / len(group_means), "summary": summary(rows), "by_kind": {k: summary([r for r in rows if r["kind"] == k]) for k in ("direct", "cross_object", "sham")}, "by_base_status": {k: summary([r for r in rows if r["base_status"] == k]) for k in ("correct", "incorrect", "unjudgeable")}, "group_brier": group_means, "rows": rows})
    reference = reports[0]
    result = {**hashes, "gold_sha256": sha(gold_path), "status": "scored_independent_annotations", "caveat": "Not unique root-cause or repair evidence. Independence still requires a real blinded annotation process.", "generation": render.get("generation"), "methods": reports, "paired_against_first": {r["method"]: bootstrap_difference(r["group_brier"], reference["group_brier"]) for r in reports[1:]}}
    save_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("pack")
    p.add_argument("study"); p.add_argument("render"); p.add_argument("output")
    p.add_argument("--method", choices=METHODS, required=True); p.add_argument("--doctor")
    p = commands.add_parser("read")
    p.add_argument("bundle"); p.add_argument("output")
    p.add_argument("--endpoint"); p.add_argument("--model"); p.add_argument("--revision")
    p.add_argument("--baseline", choices=("unchanged", "source-only"))
    p.add_argument("--max-tokens", type=int, default=2400)
    p = commands.add_parser("doctor")
    p.add_argument("study"); p.add_argument("render"); p.add_argument("output")
    p.add_argument("--repo", required=True); p.add_argument("--python", required=True); p.add_argument("--checkpoint", required=True)
    p = commands.add_parser("score")
    p.add_argument("study"); p.add_argument("render"); p.add_argument("gold"); p.add_argument("output")
    p.add_argument("predictions", nargs="+")
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.study, args.render, args.output, args.method, args.doctor)
    elif args.command == "read":
        read_bundle(args.bundle, args.output, args.endpoint, args.model, args.baseline, args.max_tokens, args.revision)
    elif args.command == "doctor":
        doctor(args.study, args.render, args.repo, args.python, args.checkpoint, args.output)
    else:
        result = evaluate(args.study, args.render, args.gold, args.predictions, args.output)
        print(json.dumps({r["method"]: r["primary_group_mean_brier"] for r in result["methods"]}, indent=2))


if __name__ == "__main__":
    main()
