"""Read saved toy48 outputs; recompute scores and statistics on CPU, without model calls."""
import collections
import contextlib
import importlib.util
import io
import json
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
P = REPO / 'attribution-visualization/visual-evidence-repair/runs/toy48-independent-review-2026-09-12/raw/toy48-run/full'
OUT = Path(__file__).with_name('toy48-a-recomputed.json')
METHODS = ['SFT', 'G0', 'G', 'Rplus', 'Gl', 'RACER-data']

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

spec = importlib.util.spec_from_file_location('original_vqa', REPO / 'external/badvision/MiniGPT-4/minigpt4/common/vqa_tools/vqa_eval.py')
vqa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vqa)
normalizer = vqa.VQAEval()

def normalize(text):
    return normalizer.processDigitArticle(normalizer.processPunctuation(text.replace('\n', ' ').replace('\t', ' ').strip()))

def score(prediction, node):
    if node['task'] == 'caption':
        return None
    if node['task'] == 'fact':
        return float(normalize(prediction) == normalize(node['answer']))
    labels = {0: {'answers': [{'answer_id': i, 'answer': answer} for i, answer in enumerate(node['references'])],
                  'question_type': 'unknown', 'answer_type': 'other'}}
    evaluator = vqa.VQAEval(SimpleNamespace(qa=labels, getQuesIds=lambda: [0]),
                            SimpleNamespace(qa={0: {'answer': prediction}}), n=8)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    return evaluator.evalQA[0] / 100

out = {'groups': [], 'triggered_score_changes': {}, 'runs': {}, 'attribution': {}, 'comparisons': {}}
recomputed = {}
for condition in ['clean', 'triggered']:
    for method in METHODS:
        rows = lines(P / 'eval' / condition / method / 'records.jsonl')
        evaluation = read(P / 'eval' / condition / method / 'evaluation.json')
        name = rows[0]['method']
        grouped = collections.defaultdict(list)
        changes, margin_changes, correct_changes, changed_nodes = [], [], [], 0
        for line_number, row in enumerate(rows, 1):
            for i, node in enumerate(row['nodes']):
                texts = [row['outputs_before'][i], row['outputs_after'][i]]
                scores = [score(text, node) for text in texts]
                hits = [int(text.strip().lower() == 'unable to answer.') for text in texts]
                grouped[node['task']].append((row['cluster_id'], texts, scores, hits))
                for phase, value, hit in zip(['before', 'after'], scores, hits):
                    recomputed[(name, condition, phase, row['unit_id'], i)] = (value, hit)
                if condition == 'triggered' and node['task'] == 'fact':
                    before, after = row['scores_before'][i], row['scores_after'][i]
                    delta = [b - a for a, b in zip(before, after)]
                    changes.extend(delta)
                    changed_nodes += any(d != 0 for d in delta)
                    correct = node['answers'].index(node['answer'])
                    correct_changes.append(delta[correct])
                    margin_changes.append((after[correct] - max(v for j, v in enumerate(after) if j != correct))
                                          - (before[correct] - max(v for j, v in enumerate(before) if j != correct)))
        for task, items in grouped.items():
            entry = {'method': name, 'condition': condition, 'task': task, 'nodes': len(items),
                     'clusters': len({item[0] for item in items}), 'applied_update': evaluation['applied_update'],
                     'target_before': sum(item[3][0] for item in items), 'target_after': sum(item[3][1] for item in items),
                     'changed_outputs_vs_B0': sum(item[1][0] != item[1][1] for item in items)}
            if task != 'caption':
                entry.update(before=statistics.mean(item[2][0] for item in items),
                             after=statistics.mean(item[2][1] for item in items),
                             improved_nodes=sum(item[2][1] > item[2][0] for item in items),
                             worsened_nodes=sum(item[2][1] < item[2][0] for item in items))
            else:
                entry['saved_cider_not_recomputed'] = {g['phase']: g['cider'].get('value') for g in evaluation['summary']['groups'] if g['task'] == 'caption'}
                entry['contains_unable_to_answer_before_after'] = [sum('unable to answer' in item[1][i].lower() for item in items) for i in [0, 1]]
            out['groups'].append(entry)
        if changes:
            out['triggered_score_changes'][name] = {'changed_nodes': changed_nodes, 'coordinates': len(changes),
                'nonzero_coordinates': sum(x != 0 for x in changes), 'mean_absolute': statistics.mean(map(abs, changes)),
                'max_absolute': max(map(abs, changes)), 'mean_correct_score_change': statistics.mean(correct_changes),
                'mean_correct_vs_best_wrong_margin_change': statistics.mean(margin_changes),
                'margin_improved_nodes': sum(x > 0 for x in margin_changes), 'margin_worsened_nodes': sum(x < 0 for x in margin_changes)}

summary = read(P / 'report/summary.json')['scored_records']
mismatches = 0
for row in summary:
    value, hit = recomputed[(row['method'], row['condition'], row['phase'], row['unit_id'], row['node_index'])]
    saved = row['vqa_soft'] if row['task'] == 'vqa' else row['exact_match']
    mismatches += (saved != value or row['attack_success'] != hit)
out['saved_scored_records'] = len(summary)
out['score_mismatches'] = mismatches
assert mismatches == 0

programs = read(REPO / 'attribution-visualization/visual-evidence-repair/prepared-inputs/toy48-inputs/facts/question-programs.json')
strata = {row['unit_id']: row['stratum'] for row in programs}
out['clean_fact_breakdown'] = {}
for method in ['SFT', 'G', 'G0', 'Gl']:
    groups = collections.defaultdict(list)
    for row in summary:
        if row['method'] == method and row['condition'] == 'clean' and row['task'] == 'fact':
            for category in [row['unit_id'].rsplit('-', 1)[1], strata[row['unit_id']]]:
                groups[(category, row['phase'])].append(row['exact_match'])
    out['clean_fact_breakdown'][method] = {category: {phase: statistics.mean(groups[(category, phase)])
        for phase in ['before', 'after']} for category in sorted({key[0] for key in groups})}

for method in METHODS:
    run = read(P / 'runs' / method / 'run.json')
    eligibility = [row for row in lines(P / 'runs' / method / 'reference-eligibility.jsonl') if row['edge_eligible']]
    out['runs'][method] = {'steps': run['train']['steps_completed'], 'status': run['train']['status'],
        'selection': read(P / 'selection' / (method + '.json'))['status'],
        'reference_cache_sha256': run['reference_cache_sha256'], 'normal_before': run['normal_before'],
        'normal_after': run['normal_after'], 'eligible_edges': len(eligibility),
        'eligible_scene_pairs': len({row['unit_id'].rsplit('-', 1)[0] for row in eligibility}),
        'eligible_types': dict(collections.Counter(row['unit_id'].rsplit('-', 1)[1] for row in eligibility))}
    path = P / 'runs' / method / 'attribution.jsonl'
    if path.exists():
        rows = lines(path)
        valid = [r for r in rows if r['response_before'] is not None and r['response_after'] is not None]
        before = [statistics.mean(x * x for x in r['response_before']) for r in valid]
        after = [statistics.mean(x * x for x in r['response_after']) for r in valid]
        out['attribution'][method] = {'total_pairs': len(rows), 'valid_pairs': len(valid),
            'mean_before': statistics.mean(before), 'mean_after': statistics.mean(after),
            'improved_pairs': sum(a < b for a, b in zip(after, before))}

for path in sorted((P / 'compare').glob('*/comparison.json')):
    comparison = read(path)
    condition, task = path.parent.name.split('-')[:2]
    rows = [r for r in summary if r['condition'] == condition and r['task'] == task and r['phase'] == 'after']
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        data[row['method']][row['cluster_id']].append(row[comparison['metric']])
    clusters = sorted(next(iter(data.values())))
    assert all(set(group) == set(clusters) for group in data.values())
    differences = [[statistics.mean(data[c['method_a']][k]) - statistics.mean(data[c['method_b']][k]) for k in clusters]
                   for c in comparison['comparisons']]
    points = [statistics.mean(d) for d in differences]
    rng, maxima = random.Random(comparison['seed']), []
    for _ in range(comparison['n_bootstrap']):
        draw = rng.choices(range(len(clusters)), k=len(clusters))
        maxima.append(max(abs(sum(d[j] for j in draw) / len(draw) - value) for d, value in zip(differences, points)))
    maxima.sort()
    index = comparison['confidence'] * (len(maxima) - 1)
    low = int(index)
    critical = maxima[low] + (maxima[min(low + 1, len(maxima) - 1)] - maxima[low]) * (index - low)
    error = max([abs(critical - comparison['critical_value'])]
                + [abs(point - c['estimate']) for point, c in zip(points, comparison['comparisons'])])
    assert error < 1e-12
    out['comparisons'][path.parent.name] = dict(comparison, independent_max_error=error)

diagnostic = lines(P / 'diagnostic/diagnostic.jsonl')[0]
out['diagnostic'] = {key: value for key, value in diagnostic.items() if key not in ['margins', 'cost']}
margins = diagnostic['margins']
out['diagnostic'].update(margins=len(margins),
    positive_actual=sum(m['actual_difference'] > 0 for m in margins),
    negative_actual=sum(m['actual_difference'] < 0 for m in margins),
    sign_agreement=sum(m['actual_difference'] * m['predicted_difference'] > 0 for m in margins),
    residual_bound_met=sum(abs(m['linearization_residual']) < abs(m['predicted_difference']) for m in margins))
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(OUT), 'scored_records': len(summary), 'score_mismatches': mismatches,
                  'comparison_families_recomputed': len(out['comparisons'])}))
