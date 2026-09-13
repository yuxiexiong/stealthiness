"""B: freeze an A-derived rule, select without outcomes, then independently evaluate.

Reuses A's model, screening, complete intervention table and evidence viewer.
No training, server scheduler, or automatically invented A finding.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import statistics
import time

from .__main__ import digest, output_run, read_json, write_json
from .data import read_jsonl
from .diagnosis import (Calls, correct, first_divergence, load_cases, measure_case, model_at,
                        render, screen_cases, subset_name)
from .diagnosis_b_rules import analyze_paired, evaluate_case, margin_subsets, method_aliases, select_subsets, validate_rule
from .report import _chips, _viewer

MODEL_KEYS = ("config_sha256", "asset_identity", "implementation_sha256", "effective_generation")


def code_identity():
    return {name: digest(Path(__file__).with_name(name)) for name in
            ("diagnosis_b.py", "diagnosis_b_rules.py", "diagnosis.py", "diagnosis_model.py", "model.py", "report.py")}


def same(actual, expected, keys):
    for key in keys:
        if actual[key] != expected[key]:
            raise ValueError("identity changed: " + key)


def freeze_rule(rule, records):
    """A evidence is required; a template cannot silently become a scientific rule."""
    if rule.get("status") != "ready_to_freeze":
        raise ValueError("fill the A-derived rule; template status cannot start B")
    for key in ("hypothesis", "prediction", "falsifier", "selection_rationale"):
        if not isinstance(rule.get(key), str) or not rule[key].strip():
            raise ValueError("missing A-derived explanation: " + key)
    spec = validate_rule(rule["spec"])
    by_id = {r["cluster_id"]: r for r in records}
    if not by_id or len(by_id) != len(records):
        raise ValueError("need unique completed A cases")
    if not any(r["summary"]["nontrivial_local_sets"] for r in records):
        raise ValueError("A has no nontrivial local success; do not automatically launch confirmation")
    evidence = rule.get("evidence", [])
    if not evidence:
        raise ValueError("cite actual A heatmap cells and their interpretation")
    for item in evidence:
        node = next(n for n in by_id[item["cluster_id"]]["nodes"] if n["id"] == item["node_id"])
        cell = next(c for c in node["heatmap"] if c["context"] == item["context"] and c["group"] == item["group"])
        if cell["value"] is None or not str(item.get("interpretation", "")).strip():
            raise ValueError("A evidence must be a measured cell with an interpretation")
    masks = [s for s in range(16) if s.bit_count() == spec["k"]]
    fixed = max(masks, key=lambda s: (sum(all(n["interventions"][s]["correct"] for n in r["nodes"])
                                         for r in records), -s))
    return {"rule": deepcopy(rule), "fixed_A_subset": fixed,
            "fixed_A_successes": sum(all(n["interventions"][fixed]["correct"] for n in r["nodes"])
                                     for r in records), "A_cases": len(records)}


def prepare_contract(args):
    a_cases, b_cases = load_cases(args.a_cases), load_cases(args.cases)
    b_receipt = read_json(args.cases.parent / "receipt.json")
    if b_receipt["stage"] != "B" or not 1 <= len(b_cases) <= 96:
        raise ValueError("B needs its independently prepared <=96-case pool")
    if (any(b_receipt["disjointness"].values())
            or {c["cluster_id"] for c in b_cases} & set(b_receipt["exclusions"]["excluded_cluster_ids"])):
        raise ValueError("B pool overlaps a historical split")
    if {c["cluster_id"] for c in a_cases} & {c["cluster_id"] for c in b_cases}:
        raise ValueError("A/B source overlap")
    a_images = {r["sha256"] for r in read_json(args.a_cases.parent / "receipt.json")["images"]}
    if a_images & {r["sha256"] for r in b_receipt["images"]}:
        raise ValueError("A/B image overlap")
    screen = read_json(args.a_screen)
    if screen["identity"]["cases_sha256"] != digest(args.a_cases):
        raise ValueError("A screening does not match A pool")
    records, a_runs = [], []
    for path in args.a_records:
        run = read_json(path.parent / "run.json")
        part = read_jsonl(path)
        if run["status"] != "completed" or run["screen_sha256"] != digest(args.a_screen):
            raise ValueError("A must have a completed run bound to its screening")
        same(run["identity"], screen["identity"], MODEL_KEYS + ("cases_sha256", "prepared_receipt_sha256"))
        if run["case_ids"] != [r["cluster_id"] for r in part] or any(r["status"] != "complete_finite_table" for r in part):
            raise ValueError("A run/records are incomplete")
        records.extend(part)
        a_runs.append({"run_sha256": digest(path.parent / "run.json"), "calls": run["calls"],
                       "elapsed_seconds": run["elapsed_seconds"]})
    if sorted(r["cluster_id"] for r in records) != sorted(c["cluster_id"] for c in screen["selected"]):
        raise ValueError("freeze after all selected A cases, not an appealing subset or smoke alone")
    result = freeze_rule(read_json(args.rule), records)
    result.update(status="frozen_before_B_model_measurement", stage="B", cases_sha256=digest(args.cases),
                  prepared_receipt_sha256=digest(args.cases.parent / "receipt.json"),
                  A_identity=screen["identity"], code_identity=code_identity(),
                  A_sources={str(p): digest(p) for p in [args.a_cases, args.a_screen, *args.a_records]},
                  A_development_cost={"screen_calls": screen["calls"], "screen_elapsed_seconds": screen["elapsed_seconds"],
                                      "runs": a_runs},
                  rule_source_sha256=digest(args.rule), max_cases=24,
                  created_utc=datetime.now(timezone.utc).isoformat())
    with output_run(args.output) as output:
        write_json(output / "contract.json", result)
    return result


def collect_selection(diagnosis, case, contract):
    """Only captures and independent margin scoring: no intervention free answers."""
    spec = contract["rule"]["spec"]
    graph_masks = margin_subsets(spec["contexts"])
    singleton_masks = [0, 1, 2, 4, 8]
    calls, features, queries, capture_costs, positions = Calls(diagnosis), [], [], [], {}
    reduce_context = {"mean": statistics.mean, "min": min, "max": max}[spec["context_aggregation"]]
    for node in case["nodes"]:
        if node["role"] != "target":
            continue  # All rules use targets; protected answers remain independent outcome requirements.
        start = len(calls.rows)
        donor = calls("capture", node["clean_row"])
        recipient = calls("capture", node["observed_row"])
        capture_costs.extend(calls.rows[start:])
        difference = diagnosis.difference(donor, recipient)
        t = first_divergence(node["normal"]["token_ids"], node["abnormal"]["token_ids"])
        positions[node["id"]] = t
        margins = {}
        if t is not None:
            for subset in sorted(set(graph_masks + singleton_masks)):
                scores = calls("margins", node["observed_row"], node["normal"]["token_ids"], donor=donor, subset=subset)
                margins[subset] = scores["values"][t]
                queries.append({"node_id": node["id"], "subset": subset, **calls.rows[-1]})
        def delta(s, j):
            a, b = margins.get(s), margins.get(s | (1 << j))
            return None if a is None or b is None else b - a
        d4 = []
        for j in range(4):
            values = [delta(s, j) for s in spec["contexts"] if not s & (1 << j)]
            d4.append(None if any(v is None for v in values) else reduce_context(values))
        features.append({"id": node["id"], "role": node["role"],
                         "d4": d4, "singleton_d4": [delta(0, j) for j in range(4)],
                         "activation_squared_difference": [g["squared_difference"] for g in difference["groups"]]})
    choices = select_subsets(features, spec, contract["fixed_A_subset"])
    # Physical cached work is counted once. Each arm also reports the work it would need alone.
    demand = {"graph_rule": capture_costs + [q for q in queries if q["subset"] in graph_masks],
              "activation_difference": capture_costs,
              "singleton_patching": capture_costs + [q for q in queries if q["subset"] in singleton_masks],
              "fixed_A": [], "random": []}
    return {"cluster_id": case["cluster_id"], "choices": choices, "features": features, "reference_positions": positions,
            "calls": calls.rows, "method_query_work": demand,
            "selection_has_intervention_answers": False,
            "same_query_ceiling_not_equal_consumed_cost": True}


def read_contract(args):
    contract = read_json(args.contract)
    if contract["status"] != "frozen_before_B_model_measurement":
        raise ValueError("B needs an A-derived frozen contract")
    same({"code_identity": code_identity(), "cases_sha256": digest(args.cases),
          "prepared_receipt_sha256": digest(args.cases.parent / "receipt.json")}, contract,
         ("code_identity", "cases_sha256", "prepared_receipt_sha256"))
    validate_rule(contract["rule"]["spec"])
    if digest(args.config) != contract["A_identity"]["config_sha256"]:
        raise ValueError("B must use the same frozen model configuration as A")
    return contract


def complete_selected_swaps(diagnosis, evidence, choices):
    """Reuse A's swaps; measure missing selected operations, including exact random."""
    subsets = sorted(set(choices["random"]) | {s for arm, s in choices.items()
                                               if arm != "random" and s is not None})
    existing = {(r["subset"], r["node_id"]) for r in evidence["donor_swaps"]}
    nodes = {n["id"]: n for n in evidence["nodes"]}
    calls, donors = Calls(diagnosis), {}
    for subset in subsets:
        for node in nodes.values():
            if (subset, node["id"]) in existing:
                continue
            peer = nodes.get(node["donor_peer_id"])
            row = {"subset": subset, "node_id": node["id"], "status": "peer_not_eligible"}
            if peer is not None:
                if peer["id"] not in donors:
                    donors[peer["id"]] = calls("capture", peer["clean_row"])
                result = calls("generate", node["observed_row"], donor=donors[peer["id"]], subset=subset)
                row.update(donor_id=peer["id"], status="measured", receiver_truth=node["clean_row"]["answer"],
                           donor_truth=peer["clean_row"]["answer"], output=result,
                           matches_receiver=correct(result, node["clean_row"]),
                           matches_donor=correct(result, peer["clean_row"]),
                           fact_changed=node["clean_row"]["answer"] != peer["clean_row"]["answer"],
                           counts_as_repair=False)
            evidence["donor_swaps"].append(row)
    evidence["calls"].extend(calls.rows)
    evidence["selected_donor_swap_subsets"] = subsets
    return evidence


def selected_cases(screened, pool, lane, lanes):
    sources = {c["cluster_id"]: c for c in pool}
    cases = deepcopy(screened["selected"][lane::lanes])
    for case in cases:
        nodes = {n["id"]: n for n in sources[case["cluster_id"]]["nodes"]}
        for node in case["nodes"]:
            for key in ("clean_row", "observed_row"):
                node[key] = nodes[node["id"]][key]
    return cases


def load_selections(paths, screened, contract_sha, screen_sha):
    rows = []
    for path in paths:
        receipt = read_json(path.parent / "selection.json")
        if receipt["status"] != "choices_sealed" or receipt["contract_sha256"] != contract_sha:
            raise ValueError("selection contract mismatch or incomplete selections")
        if receipt["screen_sha256"] != screen_sha or receipt["selections_sha256"] != digest(path):
            raise ValueError("selection changed after sealing")
        rows.extend(read_jsonl(path))
    if sorted(r["cluster_id"] for r in rows) != sorted(c["cluster_id"] for c in screened["selected"]):
        raise ValueError("seal choices for ALL screened B cases before any evaluation")
    return {r["cluster_id"]: r for r in rows}


def append(log, row):
    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    log.flush()


def render_summary(summary, selections, output):
    labels = {"graph_rule": "A 图导出规则", "activation_difference": "状态差异最大",
              "singleton_patching": "普通单组归因", "fixed_A": "A 最佳固定位置", "random": "同大小随机位置"}
    aliases = summary["analysis"]["method_aliases"]
    if aliases:
        labels["graph_rule"] = "普通单组归因（本轮主方法）"
        labels["singleton_patching"] = "普通单组归因（同一方法，兼容字段）"
    parts = ['<h1>B：先选位置，再看完整回答</h1><p>判断规则来自 A，在新场景测量前冻结。'
             '归因颜色表示固定正确参照的竞争变化，不表示修复概率。弃权仍留在总分母。</p>',
             '<p><a href="evidence.html">查看图片、真实回答与全部干预图</a> · '
             '<a href="summary.json">机器可读的配对结果与成本</a></p>',
             '<h2>冻结的判断</h2><p>' + escape(summary["rule"]["hypothesis"]) + '</p><p>推翻条件：' +
             escape(summary["rule"]["falsifier"]) + '</p>',
             '<h2>独立结果</h2><table><tr><th>规则</th><th>成功 / 全部合格场景</th><th>成功率</th>'
             '<th>弃权</th><th>保护题损害 / 保护题数</th></tr>']
    for arm, row in summary["analysis"]["arms"].items():
        if arm in aliases.values():
            continue
        protection = (f'{row["damaged_protection"]:g} / {row["protection_nodes"]}'
                      if row["protection_nodes"] else '不适用：没有原本答对的保护题')
        parts.append('<tr><td>' + labels[arm] + '</td><td>' + f'{row["success_numerator"]:g} / {row["eligible_cases"]}' +
                     '</td><td>' + f'{row["success_rate"]:.1%}' + '</td><td>' + str(row["abstentions"]) +
                     '</td><td>' + protection + '</td></tr>')
    parts.append('</table><p>' + ('本轮 graph_rule 与 singleton_patching 是同一标准归因方法，'
                 '只展示一次，不作独立优越性比较。' if aliases else '') +
                 '随机项是同大小所有集合的精确平均，所以成功数可能是小数。'
                 '成功须所有冻结问题实际答对；图分数改善本身不算成功。</p><h2>同场景的增量</h2>'
                 '<table><tr><th>对照</th><th>成功率差</th><th>同时不确定区间</th><th>Holm 调整后 p</th></tr>')
    for row in summary["analysis"]["comparisons"]:
        lo, hi = row["simultaneous_paired_interval"]
        parts.append('<tr><td>' + labels[row["baseline"]] + '</td><td>' + f'{row["delta"]:+.1%}' +
                     '</td><td>' + f'[{lo:+.1%}, {hi:+.1%}]' + '</td><td>' + f'{row["holm_p_value"]:.4g}' + '</td></tr>')
    parts.append('</table><p>区间按场景计算，依赖场景独立采样假定；小样本区间可能很宽。'
                 '如果两条规则选择相同位置，就没有额外的位置选择增量。</p>')
    parts.append('<h2>所选位置的供体事实交换</h2><p>每个方法只统计事先选中的位置；随机项为所有同大小位置的精确平均。'
                 '接收事实保持、跟随供体、其他回答、未测四项互斥。供体事实不变时，答对不能区分保持与复制。'
                 '这些交换均不计入主修复成功率。</p><table><tr><th>方法</th><th>供体事实</th><th>接收事实保持</th>'
                 '<th>跟随供体</th><th>其他回答</th><th>未测</th><th>加权问题数</th></tr>')
    for arm, row in summary["analysis"]["arms"].items():
        if arm in aliases.values():
            continue
        for stratum, values in row["donor_swaps"].items():
            parts.append('<tr><td>' + labels[arm] + '</td><td>' +
                         {"changed": "改变", "unchanged": "不变", "unknown": "无法确定"}[stratum] + '</td>' +
                         ''.join('<td>' + f'{values[k]:g}' + '</td>' for k in
                                 ("receiver_retained", "donor_followed", "other", "unmeasured", "expected_nodes")) + '</tr>')
    parts.append('</table><h2>事先提交的归因图与位置</h2>'
                 '<p>四格依次为左上、右上、左下、右下。红色为竞争间隔增加，蓝色为降低，灰色为无法计算。</p>')
    scale = max((abs(v) for row in selections for f in row["features"] for v in f["d4"] if v is not None), default=0)
    for row in selections:
        parts.append('<section><h2>' + escape(row["cluster_id"]) + '</h2><p>事先提交的选择：' +
                     escape(', '.join(labels[name] + ': ' + (subset_name(s) if s is not None else '弃权')
                                      for name, s in row["choices"].items()
                                      if name != 'random' and name not in aliases.values())) + '</p>')
        for feature in row["features"]:
            parts.append('<p>' + escape(feature["id"]) + '：' + _chips(feature["d4"], scale) + '</p>')
        parts.append('</section>')
    output.write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>B 独立诊断确认</title>'
                      '<style>' + _viewer().STYLE + '</style><body>' + ''.join(parts) + '</body></html>', encoding="utf-8")


def report(args):
    records, selections, sources, screen = [], [], [], None
    contract_sha = digest(args.contract)
    contract = read_json(args.contract)
    for path in args.records:
        run = read_json(path.parent / "run.json")
        if run["status"] != "completed" or run["contract_sha256"] != contract_sha or run["records_sha256"] != digest(path):
            raise ValueError("evaluation record is incomplete or changed")
        if screen is not None and screen != run["screen_sha256"]:
            raise ValueError("cannot merge different screenings")
        screen = run["screen_sha256"]
        records.extend(read_jsonl(path))
        sources.append(run)
    if not records or len({r["cluster_id"] for r in records}) != len(records):
        raise ValueError("need nonempty unique evaluation cases")
    expected = sources[0]["all_screened_case_ids"]
    for source in sources[1:]:
        same(source, sources[0], ("all_screened_case_ids", "screening_cost", "selection_receipts"))
    if sorted(r["cluster_id"] for r in records) != sorted(expected):
        raise ValueError("report all screened cases; missing lanes are not scientific failures")
    results = [evaluate_case(r["evidence"], r["selection"]["choices"]) for r in records]
    selections = [r["selection"] for r in records]
    summary = {"status": "completed_independent_confirmation", "contract_sha256": contract_sha,
               "analysis": analyze_paired(results, aliases=method_aliases(contract["rule"]["spec"])), "case_results": results,
               "rule": contract["rule"], "fixed_A_subset": contract["fixed_A_subset"],
               "cost": {"A_development": contract["A_development_cost"],
                        "A_sources": contract["A_sources"], "B_evaluation_runs": sources,
                        "B_screening_once": sources[0]["screening_cost"],
                        "B_selection_receipts_once": sources[0]["selection_receipts"],
                        "B_selection_work": [r["calls"] for r in selections],
                        "B_hidden_evaluation_work": [r["evidence"]["calls"] for r in records],
                        "per_method_work": [r["method_query_work"] for r in selections],
                        "note": "shared qualification and hidden table are separate; actual GPUh from outer measure"}}
    with output_run(args.output) as output:
        write_json(output / "summary.json", summary)
        render([r["evidence"] for r in records], output / "evidence.html", stage="B")
        render_summary(summary, selections, output / "index.html")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    for name in ("a-cases", "a-screen", "cases", "rule", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--a-records", type=Path, nargs="+", required=True)
    p = sub.add_parser("report")
    p.add_argument("--records", type=Path, nargs="+", required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("screen", "select", "evaluate"):
        p = sub.add_parser(name)
        for key in ("cases", "config", "contract", "output"):
            p.add_argument("--" + key, type=Path, required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--cpu-test", action="store_true")
        if name != "screen":
            p.add_argument("--screen", type=Path, required=True)
            p.add_argument("--lanes", type=int, choices=(1, 2), default=1)
            p.add_argument("--lane-index", type=int, choices=(0, 1), default=0)
        if name == "evaluate":
            p.add_argument("--selections", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command in ("freeze", "report"):
        result = prepare_contract(args) if args.command == "freeze" else report(args)
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
        return 0
    if args.command != "screen" and args.lane_index >= args.lanes:
        parser.error("lane-index must be below lanes")
    contract = read_contract(args)
    pool = load_cases(args.cases)
    started = time.monotonic()
    with output_run(args.output) as output:
        diagnosis, identity = model_at(args)
        same(identity, contract["A_identity"], MODEL_KEYS)
        if read_json(args.cases.parent / "receipt.json")["preprocess"] != diagnosis.vlm.processor.image_processor.to_dict():
            raise ValueError("B image preprocessing changed")
        base = {"identity": identity, "contract_sha256": digest(args.contract)}
        if args.command == "screen":
            with (output / "screening.jsonl").open("x") as log:
                result = screen_cases(diagnosis, pool, contract["max_cases"], lambda row: append(log, row))
            result.update(base, elapsed_seconds=time.monotonic() - started)
            write_json(output / "screen.json", result)
        else:
            screened = read_json(args.screen)
            if screened["contract_sha256"] != base["contract_sha256"]:
                raise ValueError("screening contract changed")
            same(identity, screened["identity"], MODEL_KEYS)
            cases = selected_cases(screened, pool, args.lane_index, args.lanes)
            base.update(screen_sha256=digest(args.screen), lane_index=args.lane_index, lanes=args.lanes,
                        all_screened_case_ids=[c["cluster_id"] for c in screened["selected"]])
            if not cases:
                raise ValueError("no cases in lane; use one lane when only one case qualifies")
            if args.command == "select":
                rows = []
                with (output / "selections.jsonl").open("x") as log:
                    for case in cases:
                        row = collect_selection(diagnosis, case, contract)
                        rows.append(row)
                        append(log, row)
                result = dict(base, status="choices_sealed", selections_sha256=digest(output / "selections.jsonl"),
                              cases=len(rows), elapsed_seconds=time.monotonic() - started)
                write_json(output / "selection.json", result)
            else:
                choices = load_selections(args.selections, screened, base["contract_sha256"], base["screen_sha256"])
                with (output / "records.jsonl").open("x") as log:
                    for case in cases:
                        evidence = complete_selected_swaps(diagnosis, measure_case(diagnosis, case),
                                                           choices[case["cluster_id"]]["choices"])
                        append(log, {"cluster_id": case["cluster_id"], "selection": choices[case["cluster_id"]],
                                     "evidence": evidence})
                result = dict(base, status="completed", records_sha256=digest(output / "records.jsonl"),
                              selections_sources={str(p): digest(p) for p in args.selections},
                              screening_cost={"calls": screened["calls"], "elapsed_seconds": screened["elapsed_seconds"]},
                              selection_receipts={digest(p): read_json(p.parent / "selection.json") for p in args.selections},
                              cases=len(cases), elapsed_seconds=time.monotonic() - started, parameter_updates=0)
                write_json(output / "run.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 2 if result["status"] == "no_eligible_cases" else 0


if __name__ == "__main__":
    raise SystemExit(main())
