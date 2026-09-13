"""Stage A only: screen factual cases, measure 16 prefill interventions, render saved evidence.

Reuses VLM/HF generation, the original VQA text scorer, immutable output handling,
and the existing offline viewer. No training, perturbation search, or stage B rule.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import base64
from html import escape
import json
from pathlib import Path
import time

from .__main__ import digest, output_run, read_json, write_json
from .data import read_jsonl
from .report import score_text, _chips, _viewer

GROUPS = ("左上 TL", "右上 TR", "左下 BL", "右下 BR")


def subset_name(mask):
    return " + ".join(GROUPS[j] for j in range(4) if mask & (1 << j)) or "不替换"


def first_divergence(reference, observed):
    # A strict-prefix truncation is not a fabricated different/EOS token.
    return next((i for i, (a, b) in enumerate(zip(reference, observed)) if a != b), None)


def minimal_sets(success):
    if set(success) != set(range(16)) or any(type(v) is not bool for v in success.values()):
        raise ValueError("minimality requires all 16 reliable boolean outcomes")
    return [s for s in range(16) if success[s]
            and not any(success[t] for t in range(16) if t != s and t & s == t)]


def summarize_case(nodes):
    if not nodes or not any(n["role"] == "target" for n in nodes):
        raise ValueError("diagnosis needs an observed factual error")
    for node in nodes:
        if [r["subset"] for r in node["interventions"]] != list(range(16)):
            raise ValueError("incomplete or duplicated intervention table")
    success = {s: all(n["interventions"][s]["correct"] for n in nodes) for s in range(16)}
    if success[0]:
        raise ValueError("the unpatched case must fail its frozen requirements")
    minimal = minimal_sets(success)
    witnesses = {str(s): {str(j): [n["id"] for n in nodes
                                 if not n["interventions"][s ^ (1 << j)]["correct"]]
                          for j in range(4) if s & (1 << j)} for s in minimal}
    common = [j for j in range(4) if minimal and all(s & (1 << j) for s in minimal)]
    return {"success": success, "minimal_sets": minimal, "required_groups": common,
            "removal_witnesses": witnesses,
            "protection_nodes": sum(n["role"] == "protection" for n in nodes),
            "protection_status": "checked" if any(n["role"] == "protection" for n in nodes)
                                 else "not_applicable_no_initially_correct_abnormal_node"}


def load_cases(path):
    path = Path(path).resolve()
    receipt = read_json(path.parent / "receipt.json")
    if digest(path) != receipt["cases_sha256"]:
        raise ValueError("prepared case list changed")
    inventory = {str((path.parent / r["path"]).resolve()): r["sha256"] for r in receipt["images"]}
    checked = set()
    cases = read_jsonl(path)
    ids = [c["cluster_id"] for c in cases]
    if ids != sorted(set(ids)):
        raise ValueError("case pool must be unique and in frozen cluster order")
    for case in cases:
        node_ids = [n["id"] for n in case["nodes"]]
        if len(node_ids) != len(set(node_ids)) or len(case["primary_ids"]) != 2:
            raise ValueError("case requires unique nodes and two primary endpoints")
        if not set(case["primary_ids"]) <= set(node_ids):
            raise ValueError("missing primary node")
        for node in case["nodes"]:
            for key in ("clean_row", "observed_row"):
                row = node[key]
                image = str((path.parent / row["image"]).resolve())
                if image not in inventory:
                    raise ValueError("image missing from prepared inventory")
                if image not in checked and digest(image) != inventory[image]:
                    raise ValueError("prepared image changed: " + image)
                checked.add(image)
                row["image"] = image
            if any(node["clean_row"][k] != node["observed_row"][k]
                   for k in ("question", "answer", "answers", "task")):
                raise ValueError("normal and abnormal question/truth differ")
    return cases


def correct(output, row):
    return score_text(output["text"], row)["exact_match"] == 1.0


class Calls:
    """Only time actual model operations; outer measure charges allocated card time."""
    def __init__(self, diagnosis):
        self.diagnosis = diagnosis
        self.rows = []

    def __call__(self, kind, *args, **kwargs):
        import torch
        device = self.diagnosis.vlm.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.monotonic()
        result = getattr(self.diagnosis, kind)(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.rows.append({"operation": kind, "seconds": time.monotonic() - start,
                          "output_tokens": len(result.get("token_ids", [])) if isinstance(result, dict) else 0,
                          "scored_tokens": len(result.get("values", [])) if isinstance(result, dict) else 0,
                          "prompt_tokens": result.get("prompt_tokens", result.get("embedding_shape", [None, None])[1]),
                          "prefill_calls": result.get("prefill_calls", 0),
                          "decode_calls": result.get("decode_calls", 0)})
        return result


def screen_cases(diagnosis, cases, limit, record):
    """Select by geometry and normal/abnormal actual answers, never by patch effects."""
    calls, selected = Calls(diagnosis), []
    for source in cases:
        case = deepcopy(source)
        audit = {"cluster_id": case["cluster_id"], "accepted": False, "nodes": []}
        if not case["geometry_eligible"]:
            record(dict(audit, reason="primary_evidence_not_visible", geometry=case["reasons"]))
            continue
        primary = [n for n in case["nodes"] if n["id"] in case["primary_ids"]]
        others = [n for n in case["nodes"] if n["id"] not in case["primary_ids"]]
        active = []
        # Screen companions only after the core pair qualifies; avoid 96*6*2 calls.
        for batch in (primary, others):
            for node in batch:
                if not node["visibility"]["eligible"]:
                    audit["nodes"].append({"id": node["id"], "reason": "evidence_not_visible"})
                    continue
                normal = calls("generate", node["clean_row"])
                entry = {"id": node["id"], "normal": normal, "normal_correct": correct(normal, node["clean_row"])}
                if entry["normal_correct"]:
                    abnormal = calls("generate", node["observed_row"])
                    entry.update(abnormal=abnormal, abnormal_correct=correct(abnormal, node["observed_row"]))
                    node.update(normal=normal, abnormal=abnormal,
                                role="protection" if entry["abnormal_correct"] else "target")
                    active.append(node)
                audit["nodes"].append(entry)
            if batch is primary:
                if not set(case["primary_ids"]) <= {n["id"] for n in active}:
                    audit["reason"] = "primary_normal_reference_incorrect"
                    break
                if all(n["role"] == "protection" for n in active):
                    audit["reason"] = "no_primary_abnormal_error"
                    break
        if "reason" not in audit:
            case["nodes"] = active
            selected.append(case)
            audit.update(accepted=True, included_nodes=[n["id"] for n in active])
        record(audit)
        if len(selected) >= limit:
            break
    return {"selected": selected, "calls": calls.rows, "requested_cases": limit,
            "status": "screened" if selected else "no_eligible_cases"}


def measure_case(diagnosis, case):
    """All subsets share the same donor and grouping; scoring never decides success."""
    calls, nodes, donors = Calls(diagnosis), [], {}
    for original in case["nodes"]:
        node = deepcopy(original)
        clean, observed = node["clean_row"], node["observed_row"]
        donor = calls("capture", clean)
        recipient = calls("capture", observed)
        repeated = calls("capture", observed)
        donors[node["id"]] = donor
        difference = diagnosis.difference(donor, recipient)
        noise = diagnosis.difference(recipient, repeated)
        node.update(difference=difference, replay_difference=noise, grid=donor["grid"], interventions=[])
        t = first_divergence(node["normal"]["token_ids"], node["abnormal"]["token_ids"])
        node["reference_position"] = t
        node["position_status"] = "first_token_divergence" if t is not None else "no_comparable_divergence"
        for subset in range(16):
            output = calls("generate", observed, donor=donor, subset=subset)
            scores = calls("margins", observed, node["normal"]["token_ids"], donor=donor, subset=subset)
            remaining = sum(g["squared_difference"] for j, g in enumerate(difference["groups"])
                            if not subset & (1 << j))
            node["interventions"].append({"subset": subset, "output": output, "correct": correct(output, observed),
                "scores": scores, "margin": scores["values"][t] if t is not None else None,
                "margin_reason": scores["reasons"][t] if t is not None else node["position_status"],
                "restored_token_fraction": sum(g["token_count"] for j, g in enumerate(difference["groups"])
                                               if subset & (1 << j)) / sum(g["token_count"] for g in difference["groups"]),
                "remaining_squared_difference": remaining,
                "remaining_above_replay_noise": remaining > noise["total_squared_difference"]})
        baseline = node["interventions"][0]["output"]["token_ids"]
        whole = node["interventions"][15]["output"]["token_ids"]
        identity = calls("generate", observed, donor=recipient, subset=15)
        if (baseline != node["abnormal"]["token_ids"] or whole != node["normal"]["token_ids"]
                or identity["token_ids"] != baseline):
            raise RuntimeError("empty replay, self-copy, or full visual restoration failed: " + node["id"])
        node["controls"] = {"empty_replay_exact": True, "whole_visual_matches_normal": True,
                            "self_copy_exact": True, "self_copy_output": identity}
        node["heatmap"] = []
        for subset in range(16):
            for j in range(4):
                if subset & (1 << j):
                    continue
                a, b = node["interventions"][subset], node["interventions"][subset | (1 << j)]
                value = None if a["margin"] is None or b["margin"] is None else b["margin"] - a["margin"]
                node["heatmap"].append({"context": subset, "group": j, "value": value,
                    "reason": None if value is not None else (a["margin_reason"] or b["margin_reason"])})
        nodes.append(node)
    summary = summarize_case(nodes)
    # All local minimal sets, not just the prettiest one. At most six in four groups.
    summary["small_scope_sets"] = [s for s in summary["minimal_sets"] if 0 < s.bit_count() <= 2]
    local = [s for s in summary["small_scope_sets"] if all(n["interventions"][s]["remaining_above_replay_noise"]
                                                          for n in nodes if n["role"] == "target")]
    summary["nontrivial_local_sets"] = local
    swaps = []
    by_id = {n["id"]: n for n in nodes}
    for subset in local:
        for node in nodes:
            peer = by_id.get(node["donor_peer_id"])
            if peer is None:
                swaps.append({"subset": subset, "node_id": node["id"], "status": "peer_not_eligible"})
                continue
            result = calls("generate", node["observed_row"], donor=donors[peer["id"]], subset=subset)
            swaps.append({"subset": subset, "node_id": node["id"], "donor_id": peer["id"], "status": "measured",
                          "receiver_truth": node["clean_row"]["answer"], "donor_truth": peer["clean_row"]["answer"],
                          "output": result, "matches_receiver": correct(result, node["clean_row"]),
                          "matches_donor": correct(result, peer["clean_row"]),
                          "fact_changed": node["clean_row"]["answer"] != peer["clean_row"]["answer"],
                          "counts_as_repair": False})
    return {"cluster_id": case["cluster_id"], "primary_ids": case["primary_ids"], "nodes": nodes,
            "summary": summary, "donor_swaps": swaps, "calls": calls.rows, "status": "complete_finite_table",
            "scope": "K, observed real anomaly; four visual entrance groups; no parameter/source localization"}


def render(records, output, stage="A"):
    limit = max((abs(c["value"]) for r in records for n in r["nodes"] for c in n["heatmap"]
                 if c["value"] is not None), default=0)
    parts = ['<h1>' + escape(stage) + '：真实异常的状态干预归因</h1><p>红：正确参照 token 的竞争优势增加；蓝：降低。'
             '灰：不适用或不可比较。色值不是正确率、贡献百分比或投毒位置。实际答案独立判定。</p>']
    images = {}
    for case in records:
        parts.append('<section><h2>' + escape(case["cluster_id"]) + '</h2><h3>成功组合与逐题见证</h3>')
        summary = case["summary"]
        parts.append('<p>保护题：' + str(summary["protection_nodes"]) + '；' + escape(summary["protection_status"]) + '</p>')
        for subset in summary["minimal_sets"]:
            parts.append('<div class="tok"><b>' + escape(subset_name(subset)) + '</b>')
            for j, ids in summary["removal_witnesses"][str(subset)].items():
                parts.append('<p>移除 ' + escape(GROUPS[int(j)]) + ' 后失败：' + escape(', '.join(ids)) + '</p>')
            parts.append('</div>')
        if not summary["minimal_sets"]:
            parts.append('<p>未找到成功组合。</p>')
        parts.append('<p>组合表示联合恢复要求，不是网络传播路径；全部视觉恢复仅作接线对照。</p>')
        parts.append('<p>至多半数坐标且各目标题仍留有可测状态差异的成功组合：' +
                     escape(', '.join(subset_name(s) for s in summary["nontrivial_local_sets"]) or '无') + '</p>')
        for node in case["nodes"]:
            parts.append('<details open><summary>' + escape(node["id"]) + ' · ' + escape(node["role"]) + '</summary>')
            for key in ("clean_row", "observed_row"):
                path = node[key]["image"]
                if path not in images:
                    images[path] = base64.b64encode(Path(path).read_bytes()).decode()
                parts.append('<img width="280" alt="' + key + '" src="data:image/png;base64,' + images[path] + '">')
            parts.append('<p>' + escape(node["clean_row"]["question"]) + '；真值：<b>' +
                         escape(node["clean_row"]["answer"]) + '</b></p>')
            parts.append('<p>正常回答：' + escape(node["normal"]["text"]) + '；异常回答：' +
                         escape(node["abnormal"]["text"]) + '</p>')
            pos = node["reference_position"]
            parts.append('<p>固定参照分歧位置（从 0 计）：' + str(pos) + '；'
                         '分歧可能是措辞变化，不等于语义错误起点。每行注明已替换集合 S。</p>')
            parts.append('<table><tr><th>已有集合 S</th>' + ''.join('<th>' + escape(g) + '</th>' for g in GROUPS) + '</tr>')
            cells = {(c["context"], c["group"]): c for c in node["heatmap"]}
            for s in range(15):
                parts.append('<tr><td>' + escape(subset_name(s)) + '</td>')
                for j in range(4):
                    cell = cells.get((s, j))
                    content = '已包含' if cell is None else _chips([cell["value"]], limit)
                    if cell is not None and cell["value"] is None:
                        content += '<small>' + escape(cell["reason"] or '不可比较') + '</small>'
                    parts.append('<td>' + content + '</td>')
                parts.append('</tr>')
            parts.append('</table><table><tr><th>替换集合</th><th>实际完整回答</th><th>答对</th>'
                         '<th>剩余状态差异平方和</th></tr>')
            for row in node["interventions"]:
                parts.append('<tr><td>' + escape(subset_name(row["subset"])) + '</td><td>' +
                    escape(row["output"]["text"]) + '</td><td>' + ('对' if row["correct"] else '错') +
                    '</td><td>' + f'{row["remaining_squared_difference"]:.6g}' + '</td></tr>')
            parts.append('</table></details>')
        parts.append('<details><summary>供体事实交换（不计修复成功）</summary><pre>' +
                     escape(json.dumps(case["donor_swaps"], ensure_ascii=False, indent=2)) + '</pre></details></section>')
    Path(output).write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>' + escape(stage) + ' 诊断归因</title>'
                           '<style>' + _viewer().STYLE + '</style><body>' + ''.join(parts) + '</body></html>', encoding="utf-8")


def model_at(args):
    import torch
    from .assets import identity
    from .model import VLM
    from .diagnosis_model import Diagnosis
    config = read_json(args.config)
    spec = deepcopy(config["model"])
    if spec.get("lora", "missing") is not None or spec.get("gradient_checkpointing"):
        raise ValueError("diagnosis needs the frozen B0: lora=null, no new adapters/checkpointing")
    for key in ("model_id", "processor_id", "adapter_path", "asset_manifest"):
        if key in spec:
            spec[key] = str((Path(args.config).resolve().parent / spec[key]).resolve())
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu") or (device.type == "cpu") != args.cpu_test:
        raise ValueError("CPU requires --cpu-test; scientific runs require CUDA")
    assets = {"cpu_test": True} if args.cpu_test else identity(spec)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.manual_seed_all(42)
    start = time.monotonic()
    vlm = VLM(spec, device=args.device, allow_download=False)
    vlm.model.requires_grad_(False)
    diag = Diagnosis(vlm, config["generation"])
    return diag, {"config_sha256": digest(args.config), "asset_identity": assets,
                  "model_spec": spec, "generation": config["generation"], "cpu_test": args.cpu_test,
                  "effective_generation": diag.generation.to_dict(),
                  "implementation_sha256": {name: digest(Path(__file__).with_name(name))
                                            for name in ("diagnosis.py", "diagnosis_model.py", "model.py", "report.py")},
                  "model_load_seconds": time.monotonic() - start}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("screen", "run"):
        p = sub.add_parser(name)
        p.add_argument("--cases", type=Path, required=True)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--cpu-test", action="store_true")
        p.add_argument("--limit", type=int, default=12)
        if name == "run":
            p.add_argument("--screen", type=Path, required=True)
            p.add_argument("--lanes", type=int, choices=(1, 2), default=1)
            p.add_argument("--lane-index", type=int, choices=(0, 1), default=0)
    p = sub.add_parser("report")
    p.add_argument("--records", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        if args.output.exists():
            raise ValueError("report output already exists")
        records = [r for path in args.records for r in read_jsonl(path)]
        if len({r["cluster_id"] for r in records}) != len(records):
            raise ValueError("duplicate cases across record files")
        render(records, args.output)
        return 0
    if not 1 <= args.limit <= 12 or (args.command == "run" and args.lane_index >= args.lanes):
        parser.error("limit is 1..12 and lane-index must be below lanes")
    cases = load_cases(args.cases)
    started = time.monotonic()
    with output_run(args.output) as output:
        diagnosis, identity = model_at(args)
        identity.update(cases_sha256=digest(args.cases), prepared_receipt_sha256=digest(args.cases.parent / "receipt.json"))
        prepared = read_json(args.cases.parent / "receipt.json")
        if prepared["preprocess"] != diagnosis.vlm.processor.image_processor.to_dict():
            raise ValueError("prepared image preprocessing differs from the loaded model")
        if args.command == "screen":
            with (output / "screening.jsonl").open("x") as log:
                def record(row):
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                result = screen_cases(diagnosis, cases, args.limit, record)
            result.update(identity=identity, elapsed_seconds=time.monotonic() - started)
            write_json(output / "screen.json", result)
        else:
            screened = read_json(args.screen)
            for key in ("config_sha256", "asset_identity", "cases_sha256", "prepared_receipt_sha256",
                        "implementation_sha256", "effective_generation"):
                if identity[key] != screened["identity"][key]:
                    raise ValueError("screen/run identity changed: " + key)
            pool = {c["cluster_id"]: c for c in cases}
            chosen = screened["selected"][:args.limit][args.lane_index::args.lanes]
            if not chosen:
                raise ValueError("no screened cases for this lane")
            records = []
            with (output / "records.jsonl").open("x") as log:
                for case in chosen:
                    # Resolve images from current verified inventory; keep frozen actual baseline outputs.
                    source = {n["id"]: n for n in pool[case["cluster_id"]]["nodes"]}
                    for node in case["nodes"]:
                        for key in ("clean_row", "observed_row"):
                            node[key] = source[node["id"]][key]
                    result_case = measure_case(diagnosis, case)
                    records.append(result_case)
                    log.write(json.dumps(result_case, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
            render(records, output / "index.html")
            result = {"status": "completed", "identity": identity, "screen_sha256": digest(args.screen),
                      "cases": len(records), "case_ids": [r["cluster_id"] for r in records],
                      "lane_index": args.lane_index, "lanes": args.lanes,
                      "calls": [c for r in records for c in r["calls"]],
                      "elapsed_seconds": time.monotonic() - started,
                      "completed_utc": datetime.now(timezone.utc).isoformat(), "parameter_updates": 0}
            write_json(output / "run.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "elapsed_seconds": result["elapsed_seconds"]}))
    return 2 if result["status"] == "no_eligible_cases" else 0


if __name__ == "__main__":
    raise SystemExit(main())
