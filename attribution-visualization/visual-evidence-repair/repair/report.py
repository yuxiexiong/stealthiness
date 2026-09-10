"""Offline saved-output scoring, aligned cluster intervals and cached HTML; no model calls."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import mimetypes
import random
import statistics
from collections import defaultdict
from functools import lru_cache
from html import escape
from pathlib import Path
from types import SimpleNamespace

from .data import ROOT, _badvision, read_jsonl


@lru_cache(maxsize=1)
def _vqa_class():
    path = ROOT / "external/badvision/MiniGPT-4/minigpt4/common/vqa_tools/vqa_eval.py"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled original VQAEval missing: {path}")
    return _badvision().import_file("repair_original_vqa", path).VQAEval


def normalize_text(text):
    """Original VQA punctuation, digit/article and contraction normalization."""
    if not isinstance(text, str):
        raise ValueError("Prediction/reference must be text")
    evaluator = _vqa_class()()
    return evaluator.processDigitArticle(evaluator.processPunctuation(text.replace("\n", " ").replace("\t", " ").strip()))


def score_text(pred, row):
    """Return exact_match and vqa_soft in [0,1], never a fabricated missing score.

    VQAv2 uses all ten original human answers and the bundled evaluator verbatim.
    Caption correctness is not inferred from full-string matching or CIDEr.
    """
    result = {"exact_match": None, "vqa_soft": None, "vqa_status": "not_vqa",
              "scorer": "bundled original VQAEval normalization/evaluator; scores in [0,1]"}
    if pred is None:
        return dict(result, vqa_status="missing_output")
    if not isinstance(pred, str):
        raise ValueError("Output must be an actual generated string or null")
    task = row.get("task", "fact")
    if task == "caption":
        return result
    if task not in {"fact", "vqa"}:
        raise ValueError("Unknown task for text scoring")
    if not isinstance(row.get("answer"), str):
        raise ValueError("Correct answer required for exact scoring")
    result["exact_match"] = float(normalize_text(pred) == normalize_text(row["answer"]))
    if task != "vqa":
        return result
    refs = row.get("references")
    if not isinstance(refs, list) or len(refs) != 10:
        return dict(result, vqa_status="missing_ten_references")
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise ValueError("VQAv2 requires ten nonempty original reference strings")
    # Unique annotator IDs preserve the official leave-one-annotator-out calculation.
    labels = {0: {"answers": [{"answer_id": i, "answer": ref} for i, ref in enumerate(refs)],
                  "question_type": row.get("question_type", "unspecified"), "answer_type": "other"}}
    vqa = SimpleNamespace(qa=labels, getQuesIds=lambda: [0])
    predictions = SimpleNamespace(qa={0: {"answer": pred}})
    evaluator = _vqa_class()(vqa, predictions, n=8)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    return dict(result, vqa_soft=evaluator.evalQA[0] / 100, vqa_status="completed")


def _rows(records):
    return read_jsonl(records) if isinstance(records, (str, Path)) else list(records)


def _group(row):
    return (row.get("cell", "default"), row.get("poison_seed", row.get("seed", 0)), row.get("condition", "unspecified"))


def _key(row):
    return _group(row) + (row["unit_id"], row["node_index"], row.get("phase", "after"))


def evaluate_records(records, attack_records=None):
    """Expand saved unit records into one scored row per node and before/after phase.

    Nodes carry task/answer/references; outputs_before/after are aligned string|null
    lists. Optional separate original-evaluator rows have unit_id, node_index,
    phase, method, matching cell/seed/condition, attack_success:bool and a nonempty
    attack_evaluator provenance string. No ASR is inferred from output substrings.
    """
    attacks = {}
    for row in _rows(attack_records or []):
        if type(row.get("attack_success")) is not bool or not isinstance(row.get("attack_evaluator"), str) or not row["attack_evaluator"].strip():
            raise ValueError("ASR requires a boolean from a named original attack evaluator")
        key = (row.get("method", "default"),) + _key(row)
        if key in attacks:
            raise ValueError(f"Duplicate attack evaluation key: {key}")
        attacks[key] = row
    scored, seen, used_attacks = [], set(), set()
    for record in _rows(records):
        if not isinstance(record.get("unit_id"), str) or not isinstance(record.get("cluster_id"), str):
            raise ValueError("Saved records require unit_id and cluster_id")
        nodes = record.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("Saved record requires nodes")
        for phase in ("before", "after"):
            outputs = record.get(f"outputs_{phase}")
            outputs = [None] * len(nodes) if outputs is None else outputs
            if not isinstance(outputs, list) or len(outputs) != len(nodes):
                raise ValueError("Saved outputs must align with all nodes; use null for missing")
            for index, (node, output) in enumerate(zip(nodes, outputs)):
                row = {key: record[key] for key in ("cell", "poison_seed", "seed", "condition") if key in record}
                row.update(unit_id=record["unit_id"], cluster_id=record["cluster_id"], node_index=index,
                           method=record.get("method", "default"), phase=phase, task=node.get("task", "fact"),
                           question_type=record.get("question_type", "unspecified"), prediction=output,
                           references=node.get("references"), answer=node.get("answer"), image=node.get("image"),
                           question=node.get("question"), answers=node.get("answers"))
                key = (row["method"],) + _key(row)
                if key in seen:
                    raise ValueError(f"Duplicate output key: {key}")
                seen.add(key)
                row.update(score_text(output, dict(node, question_type=row["question_type"])))
                attack = attacks.get(key)
                if attack is not None:
                    used_attacks.add(key)
                row["attack_success"] = float(attack["attack_success"]) if attack else None
                row["attack_evaluator"] = attack["attack_evaluator"] if attack else None
                for metric in ("exact_match", "vqa_soft"):
                    row[f"joint_{metric}"] = (row[metric] * (1 - row["attack_success"])
                                               if row[metric] is not None and attack is not None else None)
                scored.append(row)
    if attacks.keys() != used_attacks:
        raise ValueError("Attack evaluation contains keys absent from the saved output records")
    return scored


def cider_score(rows):
    """Optional corpus CIDEr + PTB tokenizer, not BadVision's fixed coco-val IDF variant."""
    if not rows or any(row.get("prediction") is None or not row.get("references") for row in rows):
        return {"status": "missing_outputs_or_references", "value": None}
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as exc:
        return {"status": "missing_dependency", "value": None, "reason": str(exc)}
    gts = {i: [{"caption": ref} for ref in row["references"]] for i, row in enumerate(rows)}
    res = {i: [{"caption": row["prediction"]}] for i, row in enumerate(rows)}
    try:
        tokenizer = PTBTokenizer()
        value, per_item = Cider().compute_score(tokenizer.tokenize(gts), tokenizer.tokenize(res))
    except Exception as exc:
        return {"status": "failed", "value": None, "reason": f"{type(exc).__name__}: {exc}"}
    return {"status": "completed", "value": float(value), "per_item": [float(x) for x in per_item],
            "scorer": "pycocoevalcap CIDEr, PTB tokenization, corpus document frequency; original units",
            "native_badvision_score": False}


def score_captions(predictions, nodes):
    """Calibration helper: missing/failed CIDEr returns cider=None, never zero."""
    if len(predictions) != len(nodes) or any(node.get("task") != "caption" for node in nodes):
        raise ValueError("Caption predictions must align with caption nodes")
    if any(pred is not None and not isinstance(pred, str) for pred in predictions):
        raise ValueError("Caption outputs must be generated strings or null")
    result = cider_score([{"prediction": pred, "references": node.get("references")}
                          for pred, node in zip(predictions, nodes)])
    return dict(result, cider=result["value"])


def _mean_report(values):
    measured = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    complete = bool(values) and len(measured) == len(values)
    return {"value": sum(measured) / len(measured) if complete else None,
            "measured": len(measured), "expected": len(values), "status": "completed" if complete else "missing"}


def summarize_outputs(records, attack_records=None, include_cider=False):
    """Return {groups, scored_records}; incomplete endpoint coverage has value=null."""
    scored = evaluate_records(records, attack_records)
    groups = defaultdict(list)
    for row in scored:
        groups[(row["method"],) + _group(row) + (row["phase"], row["task"])].append(row)
    summaries = []
    for key, rows in groups.items():
        summary = dict(zip(("method", "cell", "poison_seed", "condition", "phase", "task"), key))
        summary["clusters"] = len({row["cluster_id"] for row in rows})
        summary["metrics"] = {name: _mean_report([row[name] for row in rows]) for name in
                              ("exact_match", "vqa_soft", "attack_success", "joint_exact_match", "joint_vqa_soft")}
        if key[-1] == "caption":
            summary["cider"] = cider_score(rows) if include_cider else {"status": "not_requested", "value": None}
        summaries.append(summary)
    return {"groups": summaries, "scored_records": scored,
            "note": "VQA soft and joint VQA soft are mean scores, not counts of successful requests."}


def _paired_totals(records, method_a, method_b, metric, phase, expected_keys):
    if method_a == method_b:
        raise ValueError("Distinct methods required")
    tables = {method_a: {}, method_b: {}}
    for row in _rows(records):
        method = row.get("method")
        if method not in tables or row.get("phase", "after") != phase:
            continue
        key = _key(row)
        if key in tables[method]:
            raise ValueError(f"Duplicate paired key for {method}: {key}")
        tables[method][key] = row
    a, b = tables[method_a], tables[method_b]
    if not a or a.keys() != b.keys():
        raise ValueError(f"Unequal paired coverage; {method_a} only={len(a.keys()-b.keys())}, {method_b} only={len(b.keys()-a.keys())}")
    if expected_keys is not None and set(map(tuple, expected_keys)) != a.keys():
        raise ValueError("Both-method coverage differs from expected evaluation keys")
    groups = defaultdict(lambda: defaultdict(list))
    for key, left in a.items():
        right = b[key]
        if left["cluster_id"] != right["cluster_id"] or left.get("task") != right.get("task"):
            raise ValueError("Paired cluster/task identity mismatch")
        if any(left.get(field) != right.get(field) for field in ("answer", "references", "image", "question", "answers")):
            raise ValueError("Paired input/truth/reference identity mismatch")
        values = [left.get(metric), right.get(metric)]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f"Missing/nonfinite paired {metric}: {key}")
        groups[_group(left)][left["cluster_id"]].append(values[0] - values[1])
    pools = [set(group) for group in groups.values()]
    if any(pool != pools[0] for pool in pools[1:]):
        raise ValueError("States must share a cluster pool; analyze distinct endpoint pools separately")
    clusters = sorted(pools[0])
    if len(clusters) < 2:
        raise ValueError("Cluster interval needs at least two independent clusters")
    totals = {state: {cluster: (sum(values), len(values)) for cluster, values in group.items()}
              for state, group in groups.items()}
    return totals, clusters, set(a)


def _estimate(totals, draw):
    return sum(sum(group[c][0] for c in draw) / sum(group[c][1] for c in draw) for group in totals.values()) / len(totals)


def _quantile(values, p):
    values = sorted(values)
    index = p * (len(values) - 1)
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def paired_cluster_bootstrap(records, method_a, method_b, metric="vqa_soft", phase="after",
                             n_bootstrap=2000, seed=0, confidence=0.95, expected_keys=None):
    """Pointwise paired interval on evaluate_records() output, in original score units.

    Exact keys: (cell, poison_seed, condition, unit_id, node_index, phase), defaults
    ('default',0,'unspecified',..., 'after'). Missing/duplicate/nonfinite endpoints
    fail, never intersect/drop. Optional expected_keys detects omissions shared by
    both methods. Within each state, questions receive equal weight; states receive
    equal weight. All states must share the same cluster pool. One resampled cluster
    draw is reused across states, preserving shared-image dependence. This does not
    quantify poison-seed sampling uncertainty or provide simultaneous intervals.
    """
    if type(n_bootstrap) is not int or n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("At least two replicates and confidence in (0,1) required")
    totals, clusters, keys = _paired_totals(records, method_a, method_b, metric, phase, expected_keys)
    rng = random.Random(seed)
    draws = [_estimate(totals, rng.choices(clusters, k=len(clusters))) for _ in range(n_bootstrap)]
    tail = (1 - confidence) / 2
    return {"status": "completed", "method_a": method_a, "method_b": method_b, "metric": metric,
            "estimate": _estimate(totals, clusters), "ci_low": _quantile(draws, tail), "ci_high": _quantile(draws, 1-tail),
            "confidence": confidence, "clusters": len(clusters), "paired_items": len(keys), "states": len(totals),
            "n_bootstrap": n_bootstrap, "seed": seed, "simultaneous": False,
            "expected_coverage_checked": expected_keys is not None,
            "uncertainty": "image/scene sampling conditional on observed poison states; not poison-seed variation"}


def simultaneous_cluster_bootstrap(records, comparisons, metric="vqa_soft", phase="after",
                                    n_bootstrap=2000, seed=0, confidence=0.95, expected_keys=None):
    """Max-centered bootstrap family for predeclared comparisons of one endpoint.

    Identical global cluster draws preserve dependence across methods and states.
    The confidence quantile of max_j |bootstrap_j - estimate_j| supplies symmetric
    family intervals. These are bootstrap approximations, not finite-sample coverage
    guarantees. Other metrics/ASR/protection families are NOT covered automatically.
    by_poison_seed contains conditional point estimates, not seed-population CIs.
    """
    comparisons = [tuple(pair) for pair in comparisons]
    if not comparisons or any(len(pair) != 2 or any(not isinstance(x, str) for x in pair) for pair in comparisons):
        raise ValueError("Predeclare nonempty (method_a, method_b) comparisons")
    if len(set(comparisons)) != len(comparisons):
        raise ValueError("Duplicate comparison in simultaneous family")
    if type(n_bootstrap) is not int or n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("At least two replicates and confidence in (0,1) required")
    records = _rows(records)
    prepared = [_paired_totals(records, a, b, metric, phase, expected_keys) for a, b in comparisons]
    clusters, keys = prepared[0][1:]
    if any(other_clusters != clusters or other_keys != keys for _, other_clusters, other_keys in prepared[1:]):
        raise ValueError("All comparisons must use identical complete evaluation keys and cluster pool")
    points = [_estimate(totals, clusters) for totals, _, _ in prepared]
    rng, maxima = random.Random(seed), []
    for _ in range(n_bootstrap):
        draw = rng.choices(clusters, k=len(clusters))
        maxima.append(max(abs(_estimate(totals, draw) - point) for (totals, _, _), point in zip(prepared, points)))
    critical = _quantile(maxima, confidence)
    results, by_seed = [], defaultdict(list)
    for (a, b), point, (totals, _, _) in zip(comparisons, points, prepared):
        results.append({"method_a": a, "method_b": b, "estimate": point,
                        "ci_low": point - critical, "ci_high": point + critical})
        for poison_seed in sorted({state[1] for state in totals}, key=str):
            selected = {state: group for state, group in totals.items() if state[1] == poison_seed}
            by_seed[str(poison_seed)].append({"method_a": a, "method_b": b,
                                             "estimate": _estimate(selected, clusters), "states": len(selected)})
    return {"status": "completed", "metric": metric, "comparisons": results, "by_poison_seed": dict(by_seed),
            "simultaneous": True, "family_size": len(comparisons), "confidence": confidence, "critical_value": critical,
            "clusters": len(clusters), "paired_items": len(keys), "states": len(prepared[0][0]),
            "n_bootstrap": n_bootstrap, "seed": seed, "expected_coverage_checked": expected_keys is not None,
            "scope": "predeclared method comparisons for this endpoint only; observed poison states are fixed",
            "interval_method": "symmetric max-centered cluster bootstrap; approximate coverage"}


def plan_precision(paired_cluster_differences, n_comparisons=1, cap=5000,
                   confidence=0.95, power=0.8, threshold=0.03, effect_scenarios=(0.0, 0.05)):
    """Bonferroni normal design approximation on independent cluster differences.

    Input one equally weighted cluster-level difference per independent scene, in
    score units (e.g. 0.03, not 3). It must reflect the intended state aggregation;
    token/question-level observations are not independent clusters. Variance comes
    from development data only. Variable-size ratio estimands need an appropriate
    cluster influence calculation before using this approximation. Pilot variance
    error, nonnormality, other endpoints and poison-seed uncertainty are not covered.
    """
    values = list(paired_cluster_differences)
    if len(values) < 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("At least two finite independent cluster differences required")
    if type(n_comparisons) is not int or n_comparisons < 1 or type(cap) is not int or cap < 2:
        raise ValueError("Positive comparison count and cluster cap >=2 required")
    if not 0 < confidence < 1 or not 0.5 < power < 1 or not math.isfinite(threshold):
        raise ValueError("Invalid confidence, power or threshold")
    effects = list(effect_scenarios)
    if not effects or any(type(x) not in (int, float) or not math.isfinite(x) for x in effects):
        raise ValueError("Finite predeclared effect scenarios required")
    sd = statistics.stdev(values)
    base = {"pilot_clusters": len(values), "cluster_sd": sd, "cap": cap, "n_comparisons": n_comparisons,
            "confidence": confidence, "target_power": power, "threshold": threshold,
            "method": "two-sided Bonferroni normal approximation using pilot cluster variance",
            "guaranteed_power": False, "units": "original score units; 0.03 means 3 percentage points"}
    if sd == 0:
        return dict(base, status="variance_unidentified", required_clusters=None, within_cap=None,
                    reason="Zero pilot variance cannot establish zero population variance or adequate power")
    normal = statistics.NormalDist()
    zcrit = normal.inv_cdf(1 - (1 - confidence) / (2 * n_comparisons))
    zpower = normal.inv_cdf(power)
    scenarios = []
    for effect in effects:
        gap = abs(effect - threshold)
        required = None if gap == 0 else max(2, math.ceil(((zcrit + zpower) * sd / gap) ** 2))
        at_cap = None if gap == 0 else normal.cdf(gap * math.sqrt(cap) / sd - zcrit)
        scenarios.append({"true_effect": effect, "decision": "exclude_threshold" if effect < threshold else "exceed_threshold",
                          "required_clusters": required, "approx_power_at_cap": at_cap})
    required = None if any(row["required_clusters"] is None for row in scenarios) else max(row["required_clusters"] for row in scenarios)
    within = required is not None and required <= cap
    return dict(base, status="approximate_plan" if within else "cap_inadequate", required_clusters=required,
                within_cap=within, scenarios=scenarios, z_critical=zcrit,
                limitation="Design approximation only; plan each required endpoint separately before opening test effects")


@lru_cache(maxsize=1)
def _viewer():
    return _badvision().import_file("repair_probe_viewer", ROOT / "attribution-visualization/view_probe.py")


def _flat(values):
    if isinstance(values, list):
        return [value for item in values for value in _flat(item)]
    return [values]


def _numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def _chips(values, limit):
    if values is None:
        return '<span class="tok missing">未测</span>'
    if isinstance(values, list) and values and isinstance(values[0], list):
        return "".join('<div>' + _chips(row, limit) + '</div>' for row in values)
    chips = []
    for i, value in enumerate(_flat(values)):
        value = value if _numeric(value) else None
        try:
            chips.append(_viewer().chip("", i, i, value, limit, role="score coordinate"))
        except ImportError:
            # Color is optional; saved numeric values remain usable without matplotlib.
            chips.append('<span class="tok ' + ('missing' if value is None else '') + '">' +
                         ("未测" if value is None else f"{value:+.7g}") + '</span>')
    return "".join(chips) or '<span class="tok missing">未测</span>'


def render_records(records_path, output_html):
    """Embed local images and saved score chips in a standalone offline HTML file."""
    records_path, output_html = Path(records_path).resolve(), Path(output_html).resolve()
    if records_path == output_html or output_html.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("Use a separate .html report output")
    records = read_jsonl(records_path)
    score_keys = ("reference_scores", "scores_before", "scores_after")
    response_keys = ("response_before", "response_after")
    limits = {keys: max((abs(v) for row in records for key in keys for v in _flat(row.get(key)) if _numeric(v)), default=0)
              for keys in (score_keys, response_keys, ("weights",))}
    parts = ['<h1>事实修复：已保存观测</h1><p>离线缓存；灰色是未测，不是零。色块显示答案分数／响应坐标，'
             '不是像素归因；配对响应为每条边的 D 维向量。实际生成独立显示，不从分数猜回答。</p>']
    labels = {"reference_scores": "正常参照分数", "scores_before": "修复前分数", "scores_after": "修复后分数",
              "response_before": "修复前配对响应", "response_after": "修复后配对响应", "weights": "冻结保护权重"}
    conditions = {"clean": "正常图像（clean）", "artificial": "人工扰动",
                  "fixed_reference_artificial": "冻结参考人工扰动", "triggered": "真实触发",
                  "trigger": "真实触发", "real_trigger": "真实触发"}
    weight_sources = {"G": "由 deviation（响应偏移均方）分配；按问题类型的正值中位数缩放、限幅，再归一化。",
                      "Gl": "由 difficulty（普通答案交叉熵）排序分配；沿用归因方法在同问题类型内的相同权重集合。",
                      "G-shuffle": "置乱归因权重；保留同问题类型的权重集合，打乱其与事实对的对应。",
                      "G0": "有效事实对使用统一权重，不按代理值分配。",
                      "P": "有效事实对使用统一权重，用于全节点分数匹配。",
                      "R+": "不使用响应加权项。", "SFT": "不使用响应加权项。",
                      "RACER-data": "不使用响应加权项。", "RACER-native": "不使用响应加权项。"}
    image_cache = {}
    for row in records:
        parts.append('<details open><summary>' + escape(str(row.get("unit_id", "未提供 ID"))) + ' · ' +
                     escape(str(row.get("method", ""))) + ' · ' + escape(str(row.get("status", "未测"))) + '</summary>')
        parts.append('<p>' + escape(str(row.get("kind", ""))) + ' / ' + escape(str(row.get("question_type", ""))) + '</p>')
        condition = row.get("condition")
        parts.append('<p><b>观察条件：</b>' + escape(conditions.get(condition, str(condition) if condition is not None else "未记录")) + '</p>')
        intervention = row.get("intervention") or {}
        fact = intervention.get("changed_fact") if isinstance(intervention, dict) else None
        if isinstance(fact, dict):
            change = (f"对象 {fact.get('object_id', '未记录')} · {fact.get('attribute', '未记录')}："
                      f"{fact.get('before', '未记录')} → {fact.get('after', '未记录')}")
            verification = "已核验" if intervention.get("verified") is True else "未记录核验通过"
            parts.append('<p><b>事实改动：</b>' + escape(change) + '（' + verification + '）</p>')
        else:
            parts.append('<p class="missing"><b>事实改动：</b>' + ('无配对干预' if row.get("kind") == "single" else '未记录') + '</p>')
        parts.append('<p><b>权重来源：</b>' + escape(str(row.get("weight_source") or weight_sources.get(row.get("method"), "未记录代理规则"))) + '</p>')
        parts.append('<table><tr><th>代理／操作值</th><th>已保存数值</th><th>含义</th></tr>')
        for key, description in (("deviation", "此观察条件下，配对响应偏移的均方；按候选维平均。"),
                                 ("difficulty", "此观察条件下，普通正确答案交叉熵在配对两端的均值。"),
                                 ("weights", "冻结后实际使用的保护权重；未测不补成 1。")):
            value = row.get(key) if key != "weights" else row.get("weights", row.get("weight"))
            shown = escape(json.dumps(value, ensure_ascii=False)) if value is not None else "未测"
            parts.append('<tr><td>' + key + '</td><td' + (' class="missing"' if value is None else '') + '>' +
                         shown + '</td><td>' + description + '</td></tr>')
        parts.append('</table><p class="note">这些代理值不是像素归因，也不自动证明真实触发下的损伤或修复收益。</p>')
        for index, node in enumerate(row.get("nodes") or []):
            parts.append('<h3>节点 ' + str(index) + '</h3>')
            raw = node.get("image")
            path = Path(raw) if isinstance(raw, str) and raw else None
            if path is not None:
                path = (path if path.is_absolute() else records_path.parent / path).resolve()
            mime = mimetypes.guess_type(str(path))[0] if path is not None else None
            if path is not None and path.is_file() and mime in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                if path not in image_cache:
                    image_cache[path] = 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode("ascii")
                parts.append('<img width="360" alt="原始输入图像" src="' + image_cache[path] + '">')
            else:
                parts.append('<p class="missing">图片缺失或格式未支持：' + escape(str(raw)) + '</p>')
            parts.append('<pre>' + escape(str(node.get("question", "未测"))) + '</pre>')
            answer = node.get("answer")
            parts.append('<p><b>正确答案：</b>' + escape(str(answer) if answer is not None else "未提供真值") + '</p>')
            candidates = node.get("answers")
            if isinstance(candidates, list) and candidates:
                parts.append('<p><b>候选坐标 index → 文本</b>（与下方分数／响应坐标对应）</p><ul>')
                for coordinate, candidate in enumerate(candidates):
                    parts.append('<li>' + str(coordinate) + ' → ' + escape(str(candidate)) +
                                 (' <b>（正确）</b>' if candidate == answer else '') + '</li>')
                parts.append('</ul>')
            else:
                parts.append('<p class="missing">候选坐标未记录</p>')
            for field, title in (("reference_outputs", "正常基线实际生成"),
                                 ("outputs_before", "修复前实际生成"), ("outputs_after", "修复后实际生成")):
                outputs = row.get(field)
                output = outputs[index] if isinstance(outputs, list) and index < len(outputs) else None
                parts.append('<b>' + title + '</b><pre' + (' class="missing"' if output is None else '') + '>' +
                             escape("未生成" if output is None else str(output)) + '</pre>')
        for keys in (score_keys, response_keys, ("weights",)):
            for key in keys:
                parts.append('<p><b>' + labels[key] + '</b></p>' + _chips(row.get(key), limits[keys]))
        parts.append('<details><summary>原始记录与测量状态</summary><pre>' +
                     escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></details></details>')
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>事实修复缓存报告</title>'
                           '<style>' + _viewer().STYLE + '</style><body>' + ''.join(parts) + '</body></html>', encoding="utf-8")
    return {"output_html": str(output_html), "records": len(records), "embedded_images": len(image_cache), "model_calls": 0}
