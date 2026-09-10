"""Run from visual-evidence-repair: python -m repair --help."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.oa import write_json
from .data import load_units, read_jsonl, validate_no_leakage
from .assets import identity as asset_identity, make_manifest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant {value} is not permitted")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid_constant)


def jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def input_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "manifest_sha256": digest(path.with_suffix(".manifest.json"))}


def config_at(path):
    config = read_json(path)
    required = {"model", "training", "calibration_search", "generation", "fit", "calibration", "cohort"}
    if set(config) != required:
        raise ValueError(f"config requires exactly {sorted(required)}; attack metadata belongs to the isolated evaluator")
    for name in ("fit", "calibration"):
        if name == "calibration" and config[name] is None and config["training"]["method"] == "RACER-native":
            continue
        asset = Path(config[name]).expanduser()
        config[name] = str((asset if asset.is_absolute() else Path(path).resolve().parent / asset).resolve())
    for name in ("model_id", "processor_id", "adapter_path", "asset_manifest"):
        if name in config["model"]:
            asset = Path(config["model"][name]).expanduser()
            config["model"][name] = str((asset if asset.is_absolute() else Path(path).resolve().parent / asset).resolve())
    if "asset_manifest" not in config["model"]:
        raise ValueError("model.asset_manifest must bind the local model/processor/adapter checkpoint inventory")
    if not isinstance(config["cohort"], str) or not config["cohort"]:
        raise ValueError("cohort must identify the predeclared comparison group")
    if config["generation"].get("do_sample") is not False or not isinstance(config["generation"].get("max_new_tokens"), int):
        raise ValueError("declare deterministic generation with do_sample=false and full-protocol max_new_tokens")
    from .core import validate_config
    validate_config(config["training"])
    validate_config(dict(config["training"], **config["calibration_search"]))
    return config


@contextmanager
def output_run(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    except Exception as error:
        write_json(path / "failure.json", {"status": "failed", "error": str(error), "exception": type(error).__name__})
        raise


def backend(config, args):
    import torch
    from .model import VLM
    random.seed(config["training"]["seed"])
    torch.manual_seed(config["training"]["seed"])
    if torch.device(args.device).type == "cuda":
        torch.cuda.manual_seed_all(config["training"]["seed"])
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    return VLM(config["model"], device=args.device, allow_download=False)


@contextmanager
def measure(vlm, device):
    import torch
    device = torch.device(device)
    cost = {"language_forward_calls": 0, "input_token_positions": 0}
    def count(module, positional, kwargs):
        cost["language_forward_calls"] += 1
        tensor = kwargs.get("inputs_embeds")
        if tensor is None:
            tensor = kwargs.get("input_ids")
        if tensor is not None:
            cost["input_token_positions"] += tensor.shape[0] * tensor.shape[1]
    hook = vlm.language_module.register_forward_pre_hook(count, with_kwargs=True)
    started = time.monotonic()
    try:
        yield cost
    finally:
        hook.remove()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cost["model_seconds"] = time.monotonic() - started
        cost["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None


def is_correct(text, node):
    from .report import score_text
    result = score_text(text, node)
    metric = "vqa_soft" if node["task"] == "vqa" else "exact_match"
    return result.get(metric) == 1.0


def observe(vlm, units, generation, deltas=None):
    import torch
    output = []
    with torch.no_grad():
        for index, unit in enumerate(units):
            delta = None if deltas is None else deltas[index]
            if delta is not None:
                delta = delta.to(vlm.trainable_parameters()[0])
            output.append({"scores": [vlm.score(node, delta=delta, compute_inconsistency=False)["scores"].float().cpu().tolist()
                                      for node in unit["nodes"]],
                           "outputs": [vlm.generate(node, generation, delta=delta)["text"] for node in unit["nodes"]]})
    return output


def records(units, before, after, method, condition, normal_refs=None, weights=None):
    result = []
    for index, (unit, start, end) in enumerate(zip(units, before, after, strict=True)):
        ref = None if normal_refs is None else normal_refs[index]["scores"]
        def response(scores):
            if unit["kind"] != "pair" or ref is None or scores is None:
                return None
            return [(v - u) - (rv - ru) for u, v, ru, rv in zip(scores[0], scores[1], ref[0], ref[1], strict=True)]
        result.append({"unit_id": unit["id"], "cluster_id": unit["cluster_id"], "kind": unit["kind"],
                       "question_type": unit["question_type"], "nodes": unit["nodes"], "method": method,
                       "condition": condition, "status": "completed", "reference_scores": ref,
                       "scores_before": start["scores"], "scores_after": end["scores"],
                       "response_before": response(start["scores"]), "response_after": response(end["scores"]),
                       "weights": None if weights is None else weights.get(unit["id"]),
                       "outputs_before": start["outputs"], "outputs_after": end["outputs"]})
    return result


def reference_records(vlm, units, refs, config, weights, repaired, condition="fixed_reference_artificial"):
    before, after, normal = [], [], []
    for unit, ref in zip(units, refs, strict=True):
        valid = ref["edge_eligible"] and ref.get("observed_scores") is not None
        before.append({"scores": [s.float().tolist() for s in ref["observed_scores"]] if valid else None,
                       "outputs": ref["observed_outputs"] if valid else [None] * len(unit["nodes"])})
        after.append(observe(vlm, [unit], config["generation"], [ref["delta"]])[0] if repaired and valid
                     else {"scores": None, "outputs": [None] * len(unit["nodes"])})
        normal.append({"scores": [s.float().tolist() for s in ref["scores"]] if valid else None})
    rows = records(units, before, after, config["training"]["method"], condition, normal, weights)
    for row, unit, ref in zip(rows, units, refs, strict=True):
        row.update({"intervention": unit.get("intervention"), "eligible_nodes": ref["eligible"],
                    "edge_eligible": ref["edge_eligible"], "deviation": ref["deviation"], "difficulty": ref["difficulty"],
                    "reference_outputs": ref["outputs"],
                    "status": "completed" if repaired and ref["edge_eligible"] else "before_only" if ref["edge_eligible"] else "ineligible_reference"})
    return rows


def calibration_metrics(units, observed):
    from .report import score_text, score_captions
    by_task = {"fact": {}, "vqa": {}}
    captions, caption_nodes = [], []
    for unit, row in zip(units, observed, strict=True):
        for node, output in zip(unit["nodes"], row["outputs"], strict=True):
            if node["task"] == "caption":
                captions.append(output)
                caption_nodes.append(node)
            else:
                score = score_text(output, node)["vqa_soft" if node["task"] == "vqa" else "exact_match"]
                if score is None:
                    raise ValueError("Calibration task lacks valid references/scoring; cannot silently omit it")
                by_task[node["task"]].setdefault(unit["cluster_id"], []).append(score)
    result = {}
    for task, clusters in by_task.items():
        result[task] = (sum(sum(values) for values in clusters.values()) / sum(len(values) for values in clusters.values())) if clusters else None
        result[task + "_clusters"] = len(clusters)
    cider = score_captions(captions, caption_nodes) if captions else {"cider": None, "status": "no_caption_nodes"}
    result.update({"cider": cider["cider"], "cider_status": cider["status"], "caption_nodes": len(captions)})
    return result


def check_lock(path, config=None, assets=None):
    if not path:
        raise ValueError("requires a prior immutable U selection receipt via --selection")
    lock = read_json(path)
    if lock.get("track") != "U" or lock.get("status") not in ("selected", "no_acceptable_update"):
        raise ValueError("not a completed U selection receipt")
    if lock["status"] == "selected":
        selected = Path(lock["selected_run"])
        if digest(selected / "update.pt") != lock["update_sha256"] or digest(selected / "run.json") != lock["run_sha256"]:
            raise ValueError("selected checkpoint or run metadata changed after U lock")
    if config is not None and (lock["cohort"]["cohort"] != config["cohort"]
                               or lock["cohort"]["model"] != config["model"]
                               or lock["cohort"]["generation"] != config["generation"]
                               or lock["cohort"]["asset_identity"] != assets):
        raise ValueError("U lock belongs to a different model asset, cohort or generation protocol")
    return lock


def run_train(args):
    import torch
    from .core import build_references, search_delta, train
    config = config_at(args.config)
    assets = asset_identity(config["model"])
    if args.known:
        lock = check_lock(args.selection, config, assets)
    fit = load_units(config["fit"], {"fit"}, known_trigger=args.known)
    native = config["training"]["method"] == "RACER-native"
    if native and config["calibration"] is not None:
        raise ValueError("native reconstruction declares calibration=null; enhanced calibration belongs to RACER-data")
    calibration = [] if native else load_units(config["calibration"], {"calibration"})
    audit = validate_no_leakage(fit, calibration)
    if args.known and any("clean_nodes" not in unit for unit in fit):
        raise ValueError("K training requires clean_nodes for every observed unit")
    if args.known:
        candidates = [read_json(Path(item["run"]) / "run.json") for item in lock["candidates"]]
        matched = next((run for run in candidates if run["config"]["training"] == config["training"]), None)
        if matched is None:
            raise ValueError("K must use a training configuration already frozen in this method's U receipt")
        original = load_units(matched["config"]["fit"], {"fit"})
        by_id = {unit["id"]: unit for unit in original}
        if set(by_id) != {unit["id"] for unit in fit}:
            raise ValueError("same-data K must retain every U training unit")
        for unit in fit:
            previous = by_id[unit["id"]]
            if any(unit[key] != previous[key] for key in ("cluster_id", "kind", "question_type")):
                raise ValueError("K changed U's scene identity or question type")
            for clean, normal in zip(unit["clean_nodes"], previous["nodes"], strict=True):
                if (digest(clean["image"]) != digest(normal["image"])
                        or {k: v for k, v in clean.items() if k != "image"} != {k: v for k, v in normal.items() if k != "image"}):
                    raise ValueError("K clean references must be the same images, questions, labels and candidates as U")
    if config["training"]["method"] == "RACER-native":
        if args.known or len(fit) != 100 or any(unit["kind"] != "single" for unit in fit):
            raise ValueError("RACER-native reconstruction requires its declared 100 unedited singleton training samples")
    with output_run(args.output) as output:
        started = time.monotonic()
        vlm = backend(config, args)
        method = config["training"]["method"]
        if method == "RACER-native":
            calibration = []  # native training must not spend its budget on enhanced selection data
        with measure(vlm, args.device) as calls:
            if method in ("SFT", "RACER-data", "RACER-native"):
                refs = [{"scores": [], "outputs": [], "eligible": [], "edge_eligible": False,
                         "deviation": None, "difficulty": None, "delta": None} for _ in fit]
            else:
                refs = build_references(vlm, fit, config["training"], config["generation"], is_correct, known=args.known,
                                        with_deviations=method in ("G", "Gl", "G-shuffle"))
            normal_before = observe(vlm, calibration, config["generation"])
            calibration_search = dict(config["training"], **config["calibration_search"])
            deltas = [search_delta(vlm, unit["nodes"], calibration_search).cpu() for unit in calibration]
            proxy_before = observe(vlm, calibration, config["generation"], deltas)
            torch.save({"fit": refs, "calibration_deltas": deltas}, output / "reference-cache.pt")
            trained = train(vlm, fit, refs, config["training"], started=started, known=args.known)
            vlm.save_update(output / "update.pt")
            normal_after = observe(vlm, calibration, config["generation"])
            proxy_after = observe(vlm, calibration, config["generation"], deltas)
            fit_rows = reference_records(vlm, fit, refs, config, trained["weights"], repaired=True,
                                         condition="known_trigger_diagnostic" if args.known else "fixed_reference_artificial") if method in ("G", "Gl", "G-shuffle") else []
        rows = records(calibration, normal_before, normal_after, method, "clean")
        rows += records(calibration, proxy_before, proxy_after, method, "artificial", normal_refs=normal_before)
        jsonl(output / "calibration.jsonl", rows)
        jsonl(output / "training.jsonl", trained.pop("history"))
        reference_log = [{"unit_id": unit["id"], "eligible_nodes": ref["eligible"], "edge_eligible": ref["edge_eligible"],
                          "deviation": ref["deviation"], "difficulty": ref["difficulty"], "weight": trained["weights"].get(unit["id"]),
                          "reference_outputs": ref["outputs"]} for unit, ref in zip(fit, refs, strict=True)]
        jsonl(output / "reference-eligibility.jsonl", reference_log)
        if fit_rows:
            jsonl(output / "attribution.jsonl", fit_rows)
        run = {"config": config, "track": "K" if args.known else "U", "data": audit,
               "asset_identity": assets, "reference_cache_sha256": digest(output / "reference-cache.pt"),
               "u_selection_sha256": digest(args.selection) if args.known else None,
               "inputs": {key: input_identity(config[key]) for key in ("fit", "calibration") if config[key] is not None},
               "parameter_names": vlm.parameter_names(), "parameter_count": sum(p.numel() for p in vlm.trainable_parameters()),
               "train": trained, "normal_before": calibration_metrics(calibration, normal_before),
               "normal_after": calibration_metrics(calibration, normal_after),
               "proxy_before": calibration_metrics(calibration, proxy_before), "proxy_after": calibration_metrics(calibration, proxy_after),
               "proxy_sha256": hashlib.sha256(b"".join(d.contiguous().view(torch.uint8).numpy().tobytes() for d in deltas)).hexdigest(),
               "total_seconds": time.monotonic() - started, "cost": calls,
               "weights_sha256": digest(output / "update.pt"), "native_reproduction_verified": False,
               "budget_note": "max_seconds stops between training units and includes setup/reference cost; calibration and one-unit overrun remain counted"}
        write_json(output / "run.json", run)
        from .report import render_records
        render_records(output / "calibration.jsonl", output / "calibration.html")
        if fit_rows:
            render_records(output / "attribution.jsonl", output / "attribution.html")
        print(json.dumps({"output": str(output), "status": trained["status"], "steps": trained["steps_completed"]}))


def run_inspect(args):
    from .core import build_references, fixed_weights
    from .report import render_records
    config = config_at(args.config)
    assets = asset_identity(config["model"])
    units = load_units(args.data, {"dev"})
    with output_run(args.output) as output:
        vlm = backend(config, args)
        with measure(vlm, args.device) as cost:
            refs = build_references(vlm, units, config["training"], config["generation"], is_correct)
            weights = fixed_weights(refs, units, "G", config["training"]["seed"])
            rows = reference_records(vlm, units, refs, config, weights, repaired=False)
        jsonl(output / "attribution.jsonl", rows)
        write_json(output / "inspection.json", {"config": config, "asset_identity": assets, "data": input_identity(args.data), "cost": cost,
                                                "status": "observation_only_no_parameter_update",
                                                "weight_scope": "development set only; training weights will use full fit set"})
        render_records(output / "attribution.jsonl", output / "index.html")
        print(str(output))


def select_runs(paths, limits):
    if not 1 <= len(paths) <= 4:
        raise ValueError("selection accepts one to four predeclared configurations of ONE method")
    runs = [(Path(path).resolve(), read_json(Path(path) / "run.json")) for path in paths]
    if len({path for path, _ in runs}) != len(runs):
        raise ValueError("duplicate candidate run paths")
    def number(value, minimum=0, maximum=math.inf):
        return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum
    if (set(limits) != {"vqa_drop", "cider_relative_drop", "proxy_metric"}
            or limits["proxy_metric"] not in ("fact", "vqa")
            or not number(limits["vqa_drop"], maximum=1) or not number(limits["cider_relative_drop"], maximum=1)):
        raise ValueError("selection limits must be finite nonnegative fractions and a declared proxy metric")
    def cohort(run):
        return {"cohort": run["config"]["cohort"], "model": run["config"]["model"], "inputs": run["inputs"],
                "generation": run["config"]["generation"], "method": run["config"]["training"]["method"],
                "calibration_search": run["config"]["calibration_search"], "proxy_sha256": run["proxy_sha256"],
                "asset_identity": run["asset_identity"]}
    identity = cohort(runs[0][1])
    if any(cohort(run) != identity for _, run in runs):
        raise ValueError("candidates must share method, model, exact data, generation and comparison cohort")
    candidates = []
    for path, run in runs:
        if run["track"] != "U" or identity["method"] == "RACER-native":
            raise ValueError("K and native protocols cannot use enhanced U calibration selection")
        before, after = run["normal_before"], run["normal_after"]
        if any(before.get(key) is None or after.get(key) is None for key in ("vqa", "cider")):
            raise ValueError("full U selection requires measured normal VQA and caption CIDEr; missing is not passing")
        if (not all(number(stage["vqa"], maximum=1) and number(stage["cider"]) for stage in (before, after))
                or not number(run["total_seconds"]) or not number(run["train"]["update_norm"])
                or type(run["train"]["steps_completed"]) is not int or run["train"]["steps_completed"] < 0):
            raise ValueError("invalid calibration metric, cost or update accounting")
        if before["cider"] <= 0:
            raise ValueError("relative caption protection undefined for nonpositive baseline CIDEr")
        if digest(path / "update.pt") != run["weights_sha256"]:
            raise ValueError("checkpoint changed after calibration")
        normal_ok = (before["vqa"] - after["vqa"] <= limits["vqa_drop"]
                     and (before["cider"] - after["cider"]) / before["cider"] <= limits["cider_relative_drop"])
        proxy = run["proxy_after"].get(limits["proxy_metric"])
        if proxy is None:
            raise ValueError("predeclared proxy task metric is missing")
        if not number(proxy, maximum=1):
            raise ValueError("proxy task score must be finite in [0,1]")
        candidate = {"run": str(path), "normal_ok": normal_ok, "proxy": proxy, "seconds": run["total_seconds"],
                     "nonzero_update": run["train"]["update_norm"] > 0 and run["train"]["steps_completed"] > 0}
        candidates.append(candidate)
    eligible = [c for c in candidates if c["normal_ok"] and c["nonzero_update"]]
    eligible.sort(key=lambda c: (-c["proxy"], c["seconds"], c["run"]))
    receipt = {"track": "U", "status": "selected" if eligible else "no_acceptable_update", "limits": limits,
               "cohort": identity, "candidates": candidates, "selected_run": eligible[0]["run"] if eligible else None}
    if eligible:
        path = Path(eligible[0]["run"])
        receipt.update({"update_sha256": digest(path / "update.pt"), "run_sha256": digest(path / "run.json")})
    return receipt


def run_evaluate(args):
    from .report import summarize_outputs, render_records
    units = load_units(args.data, {"test"}, known_trigger=True)
    if args.run:
        run_path = Path(args.run).resolve()
        direct = read_json(run_path / "run.json")
        if direct["track"] == "K":
            check_lock(args.selection)
        elif direct["config"]["training"]["method"] != "RACER-native":
            raise ValueError("U methods must use --selection; --run is only K or native reconstruction")
        if digest(run_path / "update.pt") != direct["weights_sha256"]:
            raise ValueError("checkpoint differs from its recorded run")
    else:
        lock = check_lock(args.selection)
        if lock["status"] != "selected":
            raise ValueError("B0 fallback is recorded; no repaired checkpoint exists to evaluate")
        run_path = Path(lock["selected_run"])
    run = read_json(run_path / "run.json")
    config = run["config"]
    assets = asset_identity(config["model"])
    if assets != run["asset_identity"]:
        raise ValueError("base model/processor/adapter asset identity changed after repair")
    if run["track"] == "K":
        check_lock(args.selection, config, assets)
        if digest(args.selection) != run["u_selection_sha256"]:
            raise ValueError("K evaluation must use its original U receipt")
    fit = load_units(config["fit"], {"fit"}, known_trigger=run["track"] == "K")
    calibration = [] if config["calibration"] is None else load_units(config["calibration"], {"calibration"})
    validate_no_leakage(fit, calibration, units)
    if {key: input_identity(config[key]) for key in ("fit", "calibration") if config[key] is not None} != run["inputs"]:
        raise ValueError("repair input files changed since selection")
    with output_run(args.output) as output:
        started = time.monotonic()
        vlm = backend(config, args)
        with measure(vlm, args.device) as cost:
            before = observe(vlm, units, config["generation"])
            vlm.load_update(run_path / "update.pt")
            after = observe(vlm, units, config["generation"])
        condition = read_json(Path(args.data).with_suffix(".manifest.json"))["image_condition"]
        rows = records(units, before, after, config["training"]["method"], condition)
        for row in rows:
            row.update({"cell": args.cell, "seed": args.seed})
        jsonl(output / "records.jsonl", rows)
        write_json(output / "evaluation.json", {"selection_sha256": digest(args.selection) if args.selection else None,
                                                "run_sha256": digest(run_path / "run.json"), "data": input_identity(args.data),
                                                "cost": cost, "total_seconds": time.monotonic() - started,
                                                "summary": summarize_outputs(rows, include_cider=True)})
        render_records(output / "records.jsonl", output / "index.html")
        print(str(output))


def run_diagnose(args):
    import torch
    from .core import fixed_weights, one_step_diagnostic
    run_path = Path(args.run).resolve()
    run = read_json(run_path / "run.json")
    config = run["config"]
    assets = asset_identity(config["model"])
    if assets != run["asset_identity"] or digest(run_path / "reference-cache.pt") != run["reference_cache_sha256"]:
        raise ValueError("diagnostic model assets or reference cache changed since the U run")
    if args.known:
        check_lock(args.selection, config, assets)
    units = load_units(args.data, {"dev", "test"} if args.known else {"dev"}, known_trigger=args.known)
    if len(units) > args.max_units:
        raise ValueError("diagnostic budget exceeded; provide a predeclared small manifest, no silent sampling")
    if run["track"] != "U" or config["training"]["method"] not in ("G", "Gl", "G-shuffle"):
        raise ValueError("diagnose requires a U run with cached full-training attribution references")
    fit = load_units(config["fit"], {"fit"})
    if input_identity(config["fit"]) != run["inputs"]["fit"]:
        raise ValueError("fit manifest changed since reference caching")
    calibration = load_units(config["calibration"], {"calibration"})
    if input_identity(config["calibration"]) != run["inputs"]["calibration"]:
        raise ValueError("calibration data changed since the U run")
    validate_no_leakage(fit, calibration, units)
    update_unit = next((u for u in fit if u["id"] == args.update_unit_id), None)
    if update_unit is None:
        raise ValueError("--update-unit-id must identify the predeclared fit unit whose update is diagnosed")
    refs = torch.load(run_path / "reference-cache.pt", map_location="cpu", weights_only=True)["fit"]
    weights_a = fixed_weights(refs, fit, args.method_a, config["training"]["seed"])
    weights_b = fixed_weights(refs, fit, args.method_b, config["training"]["seed"])
    ref = refs[fit.index(update_unit)]
    edges = sum(r["edge_eligible"] for r in refs)
    nodes = sum(sum(r["eligible"]) for r in refs)
    with output_run(args.output) as output:
        vlm = backend(config, args)
        measured = [node for unit in units for node in unit["nodes"]]
        with measure(vlm, args.device) as cost:
            row = one_step_diagnostic(vlm, update_unit, ref, config["training"], args.method_a, args.method_b,
                                      weights_a.get(update_unit["id"], 1), weights_b.get(update_unit["id"], 1),
                                      len(fit) / edges if edges else 0, len(fit) / nodes if nodes else 0,
                                      generation=config["generation"], evaluation_nodes=measured)
        row.update({"update_unit_id": update_unit["id"], "measurement_units": [u["id"] for u in units],
                    "measurement_nodes": [{"unit_id": u["id"], "node_index": j} for u in units for j in range(len(u["nodes"]))],
                    "evaluation_environment": "real_trigger_or_K_observation" if args.known else "normal_dev",
                    "update_environment": "U artificial perturbation; no evaluation data enter loss", "cost": cost})
        jsonl(output / "diagnostic.jsonl", [row])
        write_json(output / "protocol.json", {"config": config, "data": input_identity(args.data),
                                              "reference_run_sha256": digest(run_path / "run.json"),
                                              "track": "isolated_evaluation" if args.known else "U", "parameter_names": vlm.parameter_names()})
        print(str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("assets", help="content-hash model/processor/adapter directories once, before experiments")
    assets.add_argument("directories", nargs="+")
    assets.add_argument("--output", required=True, help="must be outside the inventoried directories")
    validate = commands.add_parser("validate", help="validate manifests, image hashes and cluster separation; no model load")
    validate.add_argument("data", nargs="+")
    validate.add_argument("--known", action="store_true")
    for name in ("train", "evaluate", "diagnose", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--selection")
        if name in ("train", "inspect"):
            command.add_argument("--config", required=True)
        if name != "evaluate":
            command.add_argument("--known", action="store_true")
        if name != "train":
            command.add_argument("--data", required=True)
        if name == "evaluate":
            command.add_argument("--run", help="K or native fixed-run evaluation only")
            command.add_argument("--cell", required=True)
            command.add_argument("--seed", type=int, required=True, help="poison-state seed, not the optimizer seed")
        if name == "diagnose":
            command.add_argument("--run", required=True)
            command.add_argument("--update-unit-id", required=True)
            command.add_argument("--method-a", choices=("G", "G0", "Gl", "P", "R+", "G-shuffle"), default="G0")
            command.add_argument("--method-b", choices=("G", "G0", "Gl", "P", "R+", "G-shuffle"), default="R+")
            command.add_argument("--max-units", type=int, default=8)
    select = commands.add_parser("select", help="freeze one U method using clean/proxy calibration, never true triggers")
    select.add_argument("runs", nargs="+")
    select.add_argument("--output", required=True)
    select.add_argument("--limits", required=True, help="predeclared JSON with vqa_drop, cider_relative_drop, proxy_metric")
    report = commands.add_parser("report", help="offline visualization and optional original attack-evaluator result join")
    report.add_argument("records")
    report.add_argument("--output", required=True)
    report.add_argument("--attack-results")
    compare = commands.add_parser("compare", help="simultaneous paired image-cluster intervals for a predeclared endpoint family")
    compare.add_argument("records", nargs="+")
    compare.add_argument("--comparisons", required=True, help="JSON list of [method_a,method_b]")
    compare.add_argument("--expected-keys", required=True, help="frozen [cell,seed,condition,unit_id,node_index,phase] inventory")
    compare.add_argument("--condition", required=True)
    compare.add_argument("--task", choices=("fact", "vqa", "caption"), required=True)
    compare.add_argument("--metric", choices=("exact_match", "vqa_soft", "attack_success", "joint_vqa_soft"), required=True)
    compare.add_argument("--attack-results")
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "assets":
        path = Path(args.output).resolve()
        if path.exists() or any(path.is_relative_to(Path(root).expanduser().resolve()) for root in args.directories):
            raise ValueError("asset receipt must be a new file outside model directories")
        write_json(path, make_manifest(args.directories))
        print(str(path))
    elif args.command == "validate":
        print(json.dumps(validate_no_leakage(*(load_units(path, known_trigger=args.known) for path in args.data))))
    elif args.command == "train":
        run_train(args)
    elif args.command == "inspect":
        run_inspect(args)
    elif args.command == "select":
        limits = read_json(args.limits)
        if set(limits) != {"vqa_drop", "cider_relative_drop", "proxy_metric"} or limits["proxy_metric"] not in ("fact", "vqa"):
            raise ValueError("invalid predeclared selection limits")
        result = select_runs(args.runs, limits)
        path = Path(args.output)
        if path.exists():
            raise FileExistsError("selection receipt is immutable; use a new output path")
        write_json(path, result)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "diagnose":
        if args.method_a == args.method_b:
            raise ValueError("diagnostic methods must differ")
        run_diagnose(args)
    elif args.command == "compare":
        from .report import evaluate_records, simultaneous_cluster_bootstrap
        rows = [row for path in args.records for row in read_jsonl(path)]
        scored = evaluate_records(rows, read_jsonl(args.attack_results) if args.attack_results else None)
        selected = [row for row in scored if row["condition"] == args.condition and row["task"] == args.task]
        result = simultaneous_cluster_bootstrap(selected, read_json(args.comparisons), metric=args.metric,
                                                confidence=args.confidence, seed=args.seed, expected_keys=read_json(args.expected_keys))
        result["other_required_endpoints"] = "not evaluated by this family; no automatic complete-claim decision"
        if result["critical_value"] == 0:
            result["degenerate_interval"] = "zero empirical variation; do not use this bootstrap alone to certify a population safety bound"
        with output_run(args.output) as output:
            write_json(output / "comparison.json", result)
            print(str(output))
    else:
        from .report import render_records, summarize_outputs
        with output_run(args.output) as output:
            attacks = read_jsonl(args.attack_results) if args.attack_results else None
            write_json(output / "summary.json", summarize_outputs(args.records, attack_records=attacks, include_cider=True))
            render_records(args.records, output / "index.html")
            print(str(output))


if __name__ == "__main__":
    main()
