"""One disposable G update on a predeclared dev pair; wrap CUDA runs in repair.budget.

This checks execution, not poisoned-model qualification or a full training schedule.
CPU is available only through --cpu-test for the tiny-model regression test.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.__main__ import (asset_identity, backend, config_at, digest, input_identity,
                             is_correct, measure, observe, output_run, write_json)
from repair.core import build_references, restore, snapshot, train
from repair.data import load_units
from repair.report import score_text


def run(args):
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cpu") != args.cpu_test:
        raise ValueError("real smoke requires CUDA; --cpu-test requires --device cpu")
    config = config_at(args.config)
    if config["training"]["method"] != "G" or config["training"]["pgd_steps"] != 20:
        raise ValueError("smoke requires the declared G configuration with 20 PGD steps")
    assets = asset_identity(config["model"])
    pairs = [unit for unit in load_units(args.dev, {"dev"}) if unit["kind"] == "pair"]
    if not pairs:
        raise ValueError("dev must contain a verified pair")
    # max preserves file order on ties; choose by workload before observing model outputs.
    unit = max(pairs, key=lambda u: (len(u["nodes"][0]["answers"]),
               max(len(n["question"]) + sum(map(len, n["answers"])) for n in u["nodes"])))
    training = dict(config["training"], steps=1)
    with output_run(args.output) as output:
        started = time.monotonic()
        vlm = backend(config, args)
        with measure(vlm, device) as cost:
            refs = build_references(vlm, [unit], training, config["generation"], is_correct)
            initial = snapshot(vlm)
            trained = train(vlm, [unit], refs, training)
            if trained["steps_completed"] != 1 or trained["update_norm"] <= 0:
                raise RuntimeError("smoke did not complete one nonzero allowed-parameter update")
            vlm.save_update(output / "update.pt")
            expected = snapshot(vlm)
            restore(vlm, initial)
            vlm.load_update(output / "update.pt")
            if not all(torch.equal(a, b) for a, b in zip(vlm.trainable_parameters(), expected, strict=True)):
                raise RuntimeError("saved update did not restore exact allowed parameters")
            after = observe(vlm, [unit], config["generation"])[0]
        ref = refs[0]
        response = trained["history"][0]["response"]
        exercised = bool(ref["edge_eligible"] and response > 0 and training["beta_response"] > 0)
        result = {"status": "smoke_passed" if exercised else "partial_response_unexercised",
                  "device": str(device), "cpu_test": args.cpu_test, "config": config,
                  "effective_training": training, "config_sha256": digest(args.config),
                  "asset_identity": assets, "dev": input_identity(args.dev), "unit": unit,
                  "selection": "most candidates, longest question plus candidates, then file order",
                  "reference": {"eligible": ref["eligible"], "edge_eligible": ref["edge_eligible"],
                                "deviation": ref["deviation"], "outputs": ref["outputs"]},
                  "after": after, "after_scores": [score_text(text, node) for text, node in
                                                       zip(after["outputs"], unit["nodes"], strict=True)],
                  "train": trained, "nonzero_response_exercised": exercised,
                  "update_reload_exact": True, "update_sha256": digest(output / "update.pt"),
                  "parameter_names": vlm.parameter_names(), "cost": cost,
                  "total_seconds": time.monotonic() - started,
                  "b0_qualified": False, "experiment_schedule_ready": False,
                  "limitations": "One disposable dev update; no trigger qualification, full-fit normalization, "
                                 "six-method timing, calibration or independent evaluation. "
                                 "An ineligible/zero-response pair leaves the response path unverified."}
        write_json(output / "smoke.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-test", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "nonzero_response_exercised", "total_seconds")}))
    return 0 if result["status"] == "smoke_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
