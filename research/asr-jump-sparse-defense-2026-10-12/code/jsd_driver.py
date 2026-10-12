"""Phase driver: P1 development selection (seed 1102), P1 test (1103, 1104), H64 confirmation.

Each arm-run writes complete.json; the driver skips completed runs so an
engineering fix can resume in a fresh root by copying nothing and re-running.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import jsd_common as common
from jsd_common import native, atomic_json, restore_adapter
from jsd_eval import evaluate, gate_check
from jsd_train import run_arm


class Context:
    def __init__(self, root, contract, device="cuda", tokenizer_path=None, model_path=None, tiny_config=None,
                 smoke_steps=None, smoke_overrides=None):
        self.root, self.contract, self.device = Path(root), contract, device
        self.tokenizer_path, self.model_path, self.tiny_config = tokenizer_path, model_path, tiny_config
        self.smoke_steps, self.smoke_overrides = smoke_steps, dict(smoke_overrides or {})
        receipt = common.read_json(self.root / "preparation_passed.json")
        self.variant, self.target = receipt["variant"], receipt["target"]
        self.rows = None

    def data(self):
        if self.rows is None:
            data_dir = self.root / "private" / "data"
            self.rows = {kind: native.read_jsonl(data_dir / (kind + ".jsonl")) for kind in ("train", "validation", "confirmation", "oldqa")}
        return self.rows

    def run(self, name, arm, seed, overrides=None, a1_median_step=None):
        merged = dict(self.smoke_overrides)
        merged.update(overrides or {})
        run_arm(self.root, name, arm, seed, self.contract, self.variant, self.target, self.device, overrides=merged,
                tokenizer_path=self.tokenizer_path, model_path=self.model_path, tiny_config=self.tiny_config,
                smoke_steps=self.smoke_steps, a1_median_step=a1_median_step)
        return common.read_json(self.root / "runs" / name / "result.json")

    def result(self, name):
        return common.read_json(self.root / "runs" / name / "result.json")

    def tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path or self.contract["server"]["tokenizer"], local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        return tokenizer

    def model(self, seed):
        return common.build_model(self.contract, seed, self.device, model_path=self.model_path, tiny_config=self.tiny_config)


def extra_seconds(result, a1_median_step, steps):
    cost = result["cost"]
    components = sum(cost["charged"].values())
    overhead = 0.0
    if a1_median_step is not None and cost["median_step_seconds"] is not None:
        overhead = max(0.0, cost["median_step_seconds"] - a1_median_step) * steps
    return {"components": components, "step_overhead": overhead, "total": components + overhead,
            "row_equivalents": cost["row_equivalents"]}


def training_seconds(ctx, run_name):
    """Whole-training wall of a retrain arm (its training is the overhead)."""
    with open(ctx.root / "runs" / run_name / "training.jsonl") as stream:
        return sum((json.loads(line)["seconds"] or 0.0) for line in stream if line.strip())


def gates_for(ctx, result, label, a0_result, base_eval):
    evaluation = result["evaluations"][label]
    return gate_check(ctx.contract, evaluation["V60"], evaluation["oldqa"],
                      a0_result["evaluations"]["final"]["V60"]["mapping_correct"], base_eval["oldqa"]["mapping_correct"])


def choose_m(ctx, a0_result, a1_result):
    grid = ctx.contract["jsd"]["m_grid"]
    table = {}
    for m in grid:
        false_alarms = len(a0_result["policy"]["shadow_alarms"][str(m)])
        alarms = a1_result["policy"]["shadow_alarms"][str(m)]
        table[m] = {"false_alarms_A0": false_alarms, "alarm_step_A1": alarms[0]["step"] if alarms else None}
    zero = [m for m in grid if table[m]["false_alarms_A0"] == 0]
    if zero:
        with_alarm = [m for m in zero if table[m]["alarm_step_A1"] is not None]
        chosen = min(with_alarm, key=lambda m: table[m]["alarm_step_A1"]) if with_alarm else min(zero)
    else:
        chosen = min(grid, key=lambda m: (table[m]["false_alarms_A0"], m))
    return chosen, {str(m): v for m, v in table.items()}


def endpoint_scan(ctx, seed, a1_name, name):
    out = ctx.root / "runs" / name
    if (out / "complete.json").exists():
        return common.read_json(out / "result.json")
    out.mkdir(parents=True)
    started = time.monotonic()
    rows = ctx.data()
    train = rows["train"]
    ledger = common.CostLedger()
    with ledger.time("load"):
        model = ctx.model(seed)
        restore_adapter(model, torch.load(ctx.root / "runs" / a1_name / "adapter_final.pt", map_location="cpu", weights_only=False))
    losses, _ = common.score_rows(model, train, "poison", ctx.tokenizer(), ctx.device, batch_size=ctx.contract["p0"]["batch_size"], ledger=ledger)
    count = max(1, int(round(len(losses) * ctx.contract["endpoint_scan"]["drop_fraction"])))
    lowest = sorted(range(len(losses)), key=lambda i: losses[i])[:count]
    result = {"complete": True, "arm": "endpoint_scan", "run_name": name, "seed": seed, "drop_indices": lowest,
              "drop_recall": sum(train[i]["poison"] for i in lowest) / 200,
              "drop_precision": sum(train[i]["poison"] for i in lowest) / count,
              "auroc_low_loss_is_poison": common.auroc([-v for v in losses], [r["poison"] for r in train]),
              "cost": ledger.to_json(None, ctx.smoke_steps or ctx.contract["training"]["steps"]),
              "evaluations": {}, "wall_seconds": time.monotonic() - started}
    atomic_json(out / "result.json", result)
    atomic_json(out / "complete.json", {"complete": True, "run_name": name, "arm": "endpoint_scan", "seed": seed})
    del model
    return result


def phase_dev(ctx):
    contract = ctx.contract
    seed = contract["seeds"]["development"]
    steps = ctx.smoke_steps or contract["training"]["steps"]
    tag = "dev_s%d_" % seed
    a0 = ctx.run(tag + "a0", "a0", seed)
    a1 = ctx.run(tag + "a1", "a1", seed)
    a1_median = a1["cost"]["median_step_seconds"]
    base_eval = a0["evaluations"]["base"]
    m, m_table = choose_m(ctx, a0, a1)
    p0 = common.read_json(ctx.root / "p0_decision.json") if (ctx.root / "p0_decision.json").exists() else {"probe_enabled": True}
    probe_enabled = bool(p0.get("probe_enabled", True))
    selection = {"seed": seed, "m": m, "m_table": m_table, "probe_enabled": probe_enabled, "ladder": None,
                 "jsd_runs": {}, "abl": None, "abl_runs": {}, "prefilter": None, "endpoint": None, "variant": ctx.variant}
    for rung in contract["jsd"]["removal_ladder"]:
        name = tag + "jsd_" + rung
        result = ctx.run(name, "jsd", seed, {"m": m, "ladder": rung, "probe_enabled": probe_enabled}, a1_median)
        gate = gates_for(ctx, result, "final", a0, base_eval)
        selection["jsd_runs"][rung] = {"gate": gate, "extra": extra_seconds(result, a1_median, steps), "events": result["events"]}
        if gate["V60_pass"]:
            selection["ladder"] = rung
            break
    for n_iso, gamma in contract["abl"]["n_iso_gamma_order"]:
        name = tag + "abl_n%d_g%s" % (n_iso, gamma)
        overrides = {"n_iso": n_iso, "gamma": gamma}
        if ctx.smoke_steps:
            overrides["n_iso"] = min(n_iso, max(1, ctx.smoke_steps // 2))
        result = ctx.run(name, "abl", seed, overrides, a1_median)
        branches = next(e for e in result["events"] if e["kind"] == "abl_branches")["branches"]
        qualifying = []
        entry = {}
        for label, branch in branches.items():
            gate = gates_for(ctx, result, label, a0, base_eval)
            total_extra = branch["shared_extra_seconds"] + branch["branch_extra_seconds"] + extra_seconds(result, a1_median, steps)["step_overhead"]
            entry[label] = {"gate": gate, "extra_total": total_extra, "t_u": branch["t_u"]}
            if gate["V60_pass"]:
                qualifying.append(branch["t_u"])
        selection["abl_runs"][name] = entry
        if qualifying:
            selection["abl"] = {"n_iso": n_iso, "gamma": gamma, "t_u": min(qualifying)}
            break
    prefilter = ctx.run(tag + "prefilter", "prefilter", seed, None, a1_median)
    selection["prefilter"] = {"gate": gates_for(ctx, prefilter, "final", a0, base_eval), "extra": extra_seconds(prefilter, a1_median, steps),
                              "recall": prefilter["policy"].get("dropped_recall")}
    scan = endpoint_scan(ctx, seed, tag + "a1", tag + "endpoint_scan")
    selection["endpoint"] = {"scan_extra": extra_seconds(scan, None, steps), "recall": scan["drop_recall"], "retrain": None}
    if selection["abl"] is None and not selection["prefilter"]["gate"]["V60_pass"]:
        retrain = ctx.run(tag + "endpoint_retrain", "endpoint_retrain", seed, {"drop_indices": scan["drop_indices"]}, a1_median)
        selection["endpoint"]["retrain"] = {"gate": gates_for(ctx, retrain, "final", a0, base_eval),
                                            "extra_total": scan["cost"]["charged"]["score_forward"] + training_seconds(ctx, tag + "endpoint_retrain")}
    selected = {"variant": ctx.variant, "m": m, "ladder": selection["ladder"], "probe_enabled": probe_enabled,
                "abl": selection["abl"], "prefilter": {"drop_fraction": contract["prefilter"]["drop_fraction"]},
                "jsd_qualified_on_dev": selection["ladder"] is not None, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(ctx.root / "dev_selection.json", selection)
    atomic_json(ctx.root / "selected_config.json", selected)
    return selected


def phase_test(ctx):
    contract = ctx.contract
    selected = common.read_json(ctx.root / "selected_config.json")
    steps = ctx.smoke_steps or contract["training"]["steps"]
    report = {}
    for seed in contract["seeds"]["test"]:
        tag = "test_s%d_" % seed
        a0 = ctx.run(tag + "a0", "a0", seed)
        a1 = ctx.run(tag + "a1", "a1", seed)
        a1_median = a1["cost"]["median_step_seconds"]
        base_eval = a0["evaluations"]["base"]
        arms = {}
        ladder = selected["ladder"] or contract["jsd"]["removal_ladder"][0]
        jsd = ctx.run(tag + "jsd", "jsd", seed, {"m": selected["m"], "ladder": ladder, "probe_enabled": selected["probe_enabled"]}, a1_median)
        arms["jsd"] = {"run": tag + "jsd", "label": "final", "gate": gates_for(ctx, jsd, "final", a0, base_eval),
                       "extra": extra_seconds(jsd, a1_median, steps), "events": jsd["events"],
                       "masked_precision": jsd["policy"].get("masked_precision"), "masked_recall": jsd["policy"].get("masked_recall"),
                       "alarms": jsd["policy"].get("alarms"), "selected_ladder_was_none": selected["ladder"] is None}
        if selected["abl"] is not None:
            abl_cfg = selected["abl"]
            overrides = {"n_iso": abl_cfg["n_iso"], "gamma": abl_cfg["gamma"], "t_u_grid": [abl_cfg["t_u"]]}
            if ctx.smoke_steps:
                overrides["n_iso"] = min(abl_cfg["n_iso"], max(1, ctx.smoke_steps // 2))
            abl = ctx.run(tag + "abl", "abl", seed, overrides, a1_median)
            label = "tu%d" % abl_cfg["t_u"]
            branch = next(e for e in abl["events"] if e["kind"] == "abl_branches")["branches"][label]
            arms["abl"] = {"run": tag + "abl", "label": label, "gate": gates_for(ctx, abl, label, a0, base_eval),
                           "extra": {"components": branch["shared_extra_seconds"] + branch["branch_extra_seconds"],
                                     "step_overhead": extra_seconds(abl, a1_median, steps)["step_overhead"],
                                     "total": branch["shared_extra_seconds"] + branch["branch_extra_seconds"] + extra_seconds(abl, a1_median, steps)["step_overhead"],
                                     "row_equivalents": abl["cost"]["row_equivalents"]},
                           "isolated_precision": abl["policy"].get("isolated_precision"), "isolated_recall": abl["policy"].get("isolated_recall")}
        prefilter = ctx.run(tag + "prefilter", "prefilter", seed, None, a1_median)
        arms["prefilter"] = {"run": tag + "prefilter", "label": "final", "gate": gates_for(ctx, prefilter, "final", a0, base_eval),
                             "extra": extra_seconds(prefilter, a1_median, steps), "recall": prefilter["policy"].get("dropped_recall")}
        scan = endpoint_scan(ctx, seed, tag + "a1", tag + "endpoint_scan")
        arms["endpoint_scan"] = {"run": tag + "endpoint_scan", "recall": scan["drop_recall"], "extra": extra_seconds(scan, None, steps)}
        abl_pass = arms.get("abl", {}).get("gate", {}).get("V60_pass", False)
        if not abl_pass and not arms["prefilter"]["gate"]["V60_pass"]:
            retrain = ctx.run(tag + "endpoint_retrain", "endpoint_retrain", seed, {"drop_indices": scan["drop_indices"]}, a1_median)
            retrain_seconds = training_seconds(ctx, tag + "endpoint_retrain")
            arms["endpoint_retrain"] = {"run": tag + "endpoint_retrain", "label": "final", "gate": gates_for(ctx, retrain, "final", a0, base_eval),
                                        "extra": {"components": scan["cost"]["charged"]["score_forward"] + retrain_seconds,
                                                  "step_overhead": 0.0, "total": scan["cost"]["charged"]["score_forward"] + retrain_seconds,
                                                  "row_equivalents": scan["cost"]["row_equivalents"] + 3 * 16 * steps}}
        report[str(seed)] = {"a0": {"run": tag + "a0", "V60": a0["evaluations"]["final"]["V60"], "oldqa": a0["evaluations"]["final"]["oldqa"],
                                    "false_alarms": {m: len(v) for m, v in a0["policy"]["shadow_alarms"].items()},
                                    "base": base_eval},
                             "a1": {"run": tag + "a1", "V60": a1["evaluations"]["final"]["V60"], "oldqa": a1["evaluations"]["final"]["oldqa"],
                                    "median_step_seconds": a1_median, "truth_curve": [json.loads(l) for l in open(ctx.root / "runs" / (tag + "a1") / "truth_curve.jsonl")],
                                    "shadow_alarms": a1["policy"]["shadow_alarms"]},
                             "arms": arms}
    atomic_json(ctx.root / "test_results_pre_h64.json", report)
    return report


def phase_confirm(ctx):
    """Open H64 only after every test arm is frozen on disk."""
    contract = ctx.contract
    report = common.read_json(ctx.root / "test_results_pre_h64.json")
    rows = ctx.data()
    tokenizer = ctx.tokenizer()
    confirmations = {}
    for seed_key, entry in report.items():
        seed = int(seed_key)
        targets = {"a0": (entry["a0"]["run"], "final"), "a1": (entry["a1"]["run"], "final")}
        for arm, info in entry["arms"].items():
            if "label" in info:
                targets[arm] = (info["run"], info["label"])
        for arm, (run, label) in targets.items():
            path = ctx.root / "runs" / run / ("adapter_%s.pt" % label)
            if not path.exists():
                raise ValueError("Frozen adapter missing before H64: " + str(path))
        model = ctx.model(seed)
        confirmations[seed_key] = {}
        for arm, (run, label) in targets.items():
            restore_adapter(model, torch.load(ctx.root / "runs" / run / ("adapter_%s.pt" % label), map_location="cpu", weights_only=False))
            h64, records = evaluate(model, tokenizer, rows["confirmation"], ctx.device, ("trigger", "clean", "near"), ctx.variant, ctx.target)
            atomic_json(ctx.root / "runs" / run / ("eval_H64_%s.json" % label), {"H64": h64, "records": records})
            confirmations[seed_key][arm] = h64
        del model
    atomic_json(ctx.root / "confirmations_H64.json", confirmations)
    return confirmations


def gpu_guard(contract):
    server = contract["server"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != server["cuda_visible_devices"] or os.environ.get("JOBQ_GPU") != server["jobq_gpu"]:
        raise ValueError("Only the physical GPU1 queue worker may run this phase")
    current, ancestry = os.getpid(), []
    for _ in range(30):
        ancestry.append(current)
        if current <= 1:
            break
        current = int(Path("/proc/%s/status" % current).read_text().split("PPid:\t", 1)[1].splitlines()[0])
    if server["worker_pid"] not in ancestry:
        raise ValueError("Original worker %d is not a process ancestor" % server["worker_pid"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Frozen single-visible BF16 GPU profile unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "dev", "test", "confirm"), required=True)
    parser.add_argument("--contract", type=Path, default=common.CONTRACT_PATH)
    args = parser.parse_args()
    contract = common.load_contract(args.contract)
    if args.phase == "prepare":
        from transformers import AutoTokenizer
        from jsd_prepare import prepare
        decision = common.read_json(args.root / "p0_decision.json")
        if decision["verdict"] == "line_failed_stop":
            atomic_json(args.root / "stopped_after_p0.json", decision)
            print(json.dumps({"stopped": True, "reason": "P0 gate (a) failed"}))
            return
        tokenizer = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
        print(json.dumps(prepare(args.root, contract, tokenizer, contract["server"]["oldqa_source"], decision["p1_variant"])))
        return
    gpu_guard(contract)
    ctx = Context(args.root, contract)
    {"dev": phase_dev, "test": phase_test, "confirm": phase_confirm}[args.phase](ctx)


if __name__ == "__main__":
    main()
