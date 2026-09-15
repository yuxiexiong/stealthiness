"""Independent diagnosis: freeze sources, measure, seal predictions, then grade.

Reuses EditCLEVR preparation, frozen VLM, exact patching and official selectors.
No training, server scheduling, downloads at model load, or time-limit kill.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
from statistics import mean
import tarfile
import time

from .__main__ import digest, output_run, read_json, write_json
from .diagnosis import load_cases, model_at
from .diagnosis_comparison import qualification, load_calibration
from .visual_probe import Measurements, image_uri, now
from .visual_probe_protocol import main_maps, condition_key
from .report import normalize_text

PROJECT = Path(__file__).resolve().parents[1]
PLAN = PROJECT / 'DIAGNOSIS_GROUND_TRUTH_PLAN_2026-09-15.md'
SEED = 20260915
REFUSAL = 'Unable to answer.'
HISTORY = ('toy48-data', 'diagnosis-b-sources-2026-09-13', 'diagnosis-comparison-sources-2026-09-14')
POOLS = ('diagnosis-a-prepared-2026-09-13', 'diagnosis-b-prepared-2026-09-13',
         'diagnosis-comparison-prepared-2026-09-14')


def source_inputs():
    return dict(metadata=PROJECT / 'runs/toy48-data/editclevr_splits.tar.gz',
                engine=PROJECT / 'runs/toy48-data/question_engine.py',
                history=[PROJECT / 'runs' / p / 'selected-scenes.json' for p in HISTORY],
                exclude_cases=[PROJECT / 'runs' / p / 'cases.jsonl' for p in POOLS])


def freeze_sources(output):
    """Source-scene sampling only. No model or intervention outcome is read."""
    from tools import prepare_diagnosis_comparison as reuse
    from tools.prepare_diagnosis_b import METADATA_SHA256
    inputs = source_inputs()
    if digest(inputs['metadata']) != METADATA_SHA256:
        raise ValueError('EditCLEVR metadata hash differs')
    excluded = set()
    identities = {}
    for path in inputs['history']:
        excluded.update('editclevr-' + str(r['source']['generation']['base_scene_seed'])
                        for r in read_json(path))
        identities[str(path.relative_to(PROJECT))] = digest(path)
    for path in inputs['exclude_cases']:
        excluded.update(json.loads(line)['cluster_id'] for line in path.read_text().splitlines() if line)
        identities[str(path.relative_to(PROJECT))] = digest(path)
    old_seed = reuse.SEED
    try:
        reuse.SEED = SEED
        with tarfile.open(inputs['metadata']) as archive:
            selected = reuse.select(json.load(archive.extractfile('splits.json')), excluded)
    finally:
        reuse.SEED = old_seed
    result = {'schema': 'diagnosis-truth-source-v1', 'seed': SEED, 'selected': selected,
              'excluded_cluster_ids': sorted(excluded), 'history': identities,
              'metadata_sha256': digest(inputs['metadata']), 'plan_sha256': digest(PLAN),
              'model_outputs_used': False, 'parameter_updates': 0}
    with output_run(output) as out:
        write_json(out / 'source-lock.json', result)
    return result


def prepare_sources(args):
    """Reuse the existing official data path; only seed and current receipt differ."""
    from tools import prepare_diagnosis_comparison as reuse
    lock = read_json(args.lock)
    if lock['schema'] != 'diagnosis-truth-source-v1' or lock['seed'] != SEED:
        raise ValueError('wrong independent source lock')
    for name, sha in lock['history'].items():
        if digest(PROJECT / name) != sha:
            raise ValueError('historical exclusion source changed: ' + name)
    kwargs = source_inputs()
    old_seed = reuse.SEED
    try:
        reuse.SEED = SEED
        with tarfile.open(kwargs['metadata']) as archive:
            actual = reuse.select(json.load(archive.extractfile('splits.json')), set(lock['excluded_cluster_ids']))
        if actual != lock['selected']:
            raise ValueError('source lock no longer matches independently reproduced selection')
        receipt = reuse.build(argparse.Namespace(**kwargs, sources=args.sources, output=args.output,
            config=args.config, construction_manifest=args.construction_manifest, archive_cache=args.archive_cache))
    finally:
        reuse.SEED = old_seed
    receipt.update(stage='diagnosis-truth', diagnosis_truth_lock_sha256=digest(args.lock),
                   plan_sha256=digest(PLAN), selection='96 frozen new scenes; at most 24 qualified')
    write_json(args.output / 'receipt.json', receipt)
    return receipt


def verify_facts(cases, receipt):
    """Recompute every declared question's answer with the pinned external engine."""
    from tools.prepare_diagnosis_a import ENGINE_SHA256
    path = Path(receipt['sources']['question_engine']['path'])
    if digest(path) != ENGINE_SHA256:
        raise ValueError('wrong official CLEVR question engine')
    spec = importlib.util.spec_from_file_location('independent_clevr_truth', path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    scenes = {}
    for case in cases:
        for node in case['nodes']:
            source = node['visibility']['sources']['scene']
            if source['path'] not in scenes:
                if digest(source['path']) != source['sha256']:
                    raise ValueError('source scene changed')
                scenes[source['path']] = read_json(source['path'])
            answer = str(engine.answer_question({'nodes': deepcopy(node['question_program'])}, None,
                scenes[source['path']], cache_outputs=False))
            if answer != node['clean_row']['answer'] or answer != node['observed_row']['answer']:
                raise ValueError('external program answer disagrees with prepared fact: ' + node['id'])
    return {'questions': sum(len(c['nodes']) for c in cases), 'scenes': len(scenes),
            'engine_sha256': ENGINE_SHA256, 'source': 'EditCLEVR scene JSON + official CLEVR program'}


def complete_candidates(diagnosis, node):
    """Use all source answer choices, not a model-selected correct/wrong shortlist."""
    row = node['observed_row']
    answers = list(row['answers'])
    if len(answers) < 2 or len(set(answers)) != len(answers) or row['answer'] not in answers:
        raise ValueError('source question needs an exhaustive unique factual candidate set')
    labels = {'positive': row['answer'], 'negative': next(a for a in answers if a != row['answer']),
              'refusal': REFUSAL}
    labels.update({'choice:' + a: a for a in answers if a not in labels.values()})
    prepared = {}
    for key, text in labels.items():
        item = diagnosis.vlm.prepare(dict(row, answers=[text], answer=text))
        target = item['labels'][0]
        ids = target[target != -100].tolist()
        if not ids or ids[-1] not in diagnosis.eos_ids:
            raise ValueError('candidate does not end with the declared EOS')
        prepared[key] = {'text': text, 'token_ids': ids, 'source': 'external_fact_choices_or_fixed_refusal'}
    return prepared


def public_row(raw, candidates):
    """Whitelist: do not carry old diagnoses, summaries or hidden records."""
    if raw['output']['stop_reason'] != 'eos' or raw['status'] != 'measured':
        raise ValueError('incomplete measurement; never grade truncation as refusal or zero')
    return {'output': raw['output'], 'scores': {item['text']: raw['scores'][key]['log_prob']
            for key, item in candidates.items()},
            'seconds': sum(row['seconds'] for row in raw['calls']), 'source': raw['source'],
            'cost': {kind: sum(r['seconds'] for r in raw['calls'] if r['operation'] == kind)
                     for kind in ('generate', 'log_probs', 'capture')}}


def public_maps(actions):
    lookup = {tuple(a['indices']): a['id'] for a in actions if a['donor'] == 'same'}
    result = []
    for mapping in main_maps('unit'):
        result.append({'title': mapping['title'], 'resolution': mapping['resolution'],
            'background': mapping['background'], 'base_id': lookup[tuple(mapping['background'])],
            'cells': [{'index': c['index'], 'indices': c['indices'],
                       'action_id': lookup[tuple(sorted(set(mapping['background']) | set(c['indices'])))]}
                      for c in mapping['cells']]})
    return result


def measure_unit(diagnosis, case, node_id, input_condition, calibration, out, identity):
    from .diagnosis_truth import catalogue, append_selector_actions, predictions, METHODS, packet_sha256
    from .diagnosis_comparison_backends import atp_candidate_scores, external_localize
    nodes = {n['id']: deepcopy(n) for n in case['nodes'] if n.get('qualified')}
    node = nodes[node_id]
    if input_condition == 'clean':
        node['observed_row'] = deepcopy(node['clean_row'])
        node['abnormal'] = deepcopy(node['normal'])
    peer = nodes.get(node['donor_peer_id'])
    usable_peer = peer is not None and peer['clean_row']['answer'] != node['clean_row']['answer']
    candidates = complete_candidates(diagnosis, node)
    with (out / 'measurements.jsonl').open('w') as stream:
        measured = Measurements(diagnosis, nodes, {node_id: candidates}, {}, stream, allow_old=False, legacy={})
        controls = measured.controls(node_id)
        instrument_seconds = sum(c['seconds'] for c in measured.calls.rows)
        write_json(out / 'controls.json', controls)
        selectors = {name: external_localize(diagnosis, node['observed_row'], name, calibration)
                     for name in ('purmm', 'cleansight')}
        for name, selector in selectors.items():
            if selector['output']['token_ids'] != node['abnormal']['token_ids']:
                raise ValueError(name + ' localization changed the unmodified generation')
        gradient = atp_candidate_scores(diagnosis, node, candidates, measured.state(node_id), [])
        write_json(out / 'selectors.json', {'external': selectors, 'atp': gradient})
        actions = catalogue(include_peer=usable_peer)
        packet = {'schema': 'diagnosis-truth-public-v1', 'unit_id': node_id + ':' + input_condition,
            'cluster_id': case['cluster_id'], 'input_condition': input_condition,
            'truth_answer': node['clean_row']['answer'], 'donor_answer': peer['clean_row']['answer'] if usable_peer else None,
            'candidates': list(dict.fromkeys(c['text'] for c in candidates.values())), 'question': node['clean_row']['question'],
            'image_uri': image_uri(diagnosis.processed_image(node['observed_row'])),
            'identity_sha256': identity, 'actions': actions, 'maps': public_maps(actions), 'observations': {},
            'rankings': {'atp': gradient['scores'], 'atp_visual': gradient['visual_scores'],
                         'candidate_visual_scores': {candidates[k]['text']: v for k, v in gradient['candidate_visual_scores'].items()},
                         **{m: s['selected_visual_indices'] for m, s in selectors.items()}},
            'selector_seconds': {'atp': gradient['elapsed_seconds'], **{m: s['elapsed_seconds'] for m, s in selectors.items()}},
            'instrument_seconds': instrument_seconds,
            'cleansight_calibration_seconds_per_unit': (calibration or {}).get('seconds_per_unit', 0.0),
            'fact_evidence': {'program': node['question_program'], 'scene': node['visibility']['sources']['scene']},
            'scope': 'finite state interventions, not a labelled global poisoning circuit'}
        rows = {}
        def get(action):
            donor = node['donor_peer_id'] if action['donor'] == 'peer' else node_id
            return public_row(measured.get(node_id, action['indices'], donor), candidates)
        for action in actions:
            if action['known']:
                rows[action['id']] = get(action)
                packet['observations'][action['id']] = rows[action['id']]
        write_json(out / 'public.json', packet)
        # Seal every automatic policy before any hidden action is measured.
        submissions = [predictions(packet, method) for method in METHODS]
        write_json(out / 'submissions.json', submissions)
        write_json(out / 'seal.json', {'packet_sha256': packet_sha256(packet),
            'submissions_sha256': digest(out / 'submissions.json'), 'created_utc': now(),
            'hidden_results_visible_to_methods': False,
            'full_copy_instrument_control_is_public_known': True})
        for action in actions:
            if not action['known']:
                rows[action['id']] = get(action)
        write_json(out / 'oracle.json', {'unit_id': packet['unit_id'], 'rows': rows,
            'identity_sha256': identity, 'packet_sha256': packet_sha256(packet),
            'complete': len(rows) == len(actions), 'truth_scope': 'entire frozen action catalogue only'})
        # Native masks do not enter the common menu or give G free external proposals.
        extra = append_selector_actions([], {m: s['selected_visual_indices'] for m, s in selectors.items()})
        native = []
        for action in extra:
            raw = get(action)
            native.append({'action': action, 'result': raw,
                           'correct': normalize_text(raw['output']['text']) == normalize_text(packet['truth_answer'])})
        write_json(out / 'native-selector-checks.json', {'scope': 'selector masks under common donor replacement',
            'not_original_purification_results': True, 'checks': native})
        write_json(out / 'cost.json', {'calls': measured.calls.rows,
            'measurement_seconds': sum(c['seconds'] for c in measured.calls.rows),
            'selector_seconds': packet['selector_seconds'], 'parameter_updates': 0,
            'truth_construction_is_not_free_online_information': True})
    return packet, submissions


def costed_metrics(packet, oracle, submission, metrics):
    """Charge information actually used, even when the evaluator owns a cache."""
    method = submission['method']
    prepaid = {condition_key('unit', 'same', []), condition_key('unit', 'same', list(range(576)))}
    # Empty/full generations and their score rows were already paid in controls.
    known = [r for key, r in packet['observations'].items() if key not in prepaid]
    base = packet.get('instrument_seconds', 0.0)
    if method in ('G', 'answer-map', 'nearest', 'reader'):
        base += sum(r.get('cost', {}).get('generate', 0.0) for r in known)
    if method in ('G', 'reader'):
        base += sum(r.get('cost', {}).get('log_probs', 0.0) for r in known)
    base += packet.get('selector_seconds', {}).get(method, 0.0)
    if method == 'cleansight':
        base += packet.get('cleansight_calibration_seconds_per_unit', 0.0)
    if method == 'reader':
        base += submission.get('elapsed_seconds', 0.0)
    by_id = {r['id']: r for r in metrics['rows']}
    all_relations = {(normalize_text(r['reference_answer']), r['normalized_answer'], r['donor']) for r in metrics['rows']
                     if r['relation'] != 'no_answer_change'}
    used, seen, trajectory = base, set(), []
    # Donor captures are setup costs, paid on each method's first use, not on a
    # particular condition just because it happened to populate the shared cache.
    capture_charge = {}
    for a in packet['actions']:
        capture_charge[a['donor']] = capture_charge.get(a['donor'], 0.0) + oracle['rows'][a['id']].get('cost', {}).get('capture', 0.0)
    donors = set()
    paid_generation = set(packet['observations']) if method in ('G', 'answer-map', 'nearest', 'reader') else set(prepaid)
    for key in submission['order']:
        r = by_id[key]
        reference = r['reference_id']
        if reference not in paid_generation:
            used += oracle['rows'][reference].get('cost', {}).get('generate', 0.0)
            paid_generation.add(reference)
        if r['donor'] not in donors:
            used += capture_charge.get(r['donor'], 0.0)
            donors.add(r['donor'])
        if key not in paid_generation:
            used += oracle['rows'][key].get('cost', {}).get('generate', 0.0)
            paid_generation.add(key)
        if r['relation'] != 'no_answer_change':
            seen.add((normalize_text(r['reference_answer']), r['normalized_answer'], r['donor']))
        trajectory.append({'action_id': key, 'seconds': used, 'unique_relations': len(seen),
                           'recall': len(seen) / len(all_relations) if all_relations else None})
    g_start = packet.get('instrument_seconds', 0.0) + sum(
        r.get('cost', {}).get('generate', 0.0) + r.get('cost', {}).get('log_probs', 0.0)
        for r in known)
    metrics['cost'] = {'initial_seconds': base, 'trajectory': trajectory,
        'budget_basis': 'observed initial G evidence cost, fixed before hidden outcomes',
        'budget_results': {str(f): {'seconds': f * g_start,
            'unique_relations': next((p['unique_relations'] for p in reversed(trajectory) if p['seconds'] <= f * g_start), 0),
            'available_relations': len(all_relations)} for f in (.25, 1., 2.)},
        'common_instrument_seconds': packet.get('instrument_seconds', 0.0),
        'scope': 'single-device measured compute + submitted reader time; common instrument QA included once; source setup/screening separate'}
    return metrics


def summarize_scene_metrics(metrics):
    """Scene is the resampling unit; normal controls stay separate from abnormal."""
    groups = {}
    for row in metrics:
        key = (row['cluster_id'], row['input_condition'], row['method'])
        values = groups.setdefault(key, {})
        for name in ('exact_hit_rate', 'coverage', 'state_macro_recall', 'changed_precision', 'changed_recall', 'changed_fpr'):
            if row.get(name) is not None:
                values.setdefault(name, []).append(row[name])
        value = row.get('relational_metrics', {}).get('macro_recall')
        if value is not None:
            values.setdefault('relation_macro_recall', []).append(value)
        for k, point in row['discovery_at'].items():
            if point['recall'] is not None:
                values.setdefault('discovery_recall_at_' + k, []).append(point['recall'])
        for f, point in row.get('cost', {}).get('budget_results', {}).items():
            if point['available_relations']:
                values.setdefault('costed_relation_recall_' + f, []).append(point['unique_relations'] / point['available_relations'])
    scenes = [{'cluster_id': cid, 'input_condition': condition, 'method': method,
               **{k: mean(v) for k, v in values.items()}} for (cid, condition, method), values in sorted(groups.items())]
    pairs = []
    for condition in ('abnormal', 'clean'):
        g = {r['cluster_id']: r for r in scenes if r['method'] == 'G' and r['input_condition'] == condition}
        for other in sorted({r['method'] for r in scenes} - {'G'}):
            b = {r['cluster_id']: r for r in scenes if r['method'] == other and r['input_condition'] == condition}
            for metric in sorted(set().union(*(set(row) for row in g.values())) - {'cluster_id', 'input_condition', 'method'}):
                if other in ('atp', 'purmm', 'cleansight', 'random', 'marker') and not metric.startswith(('discovery_', 'costed_')):
                    continue  # A selector did not submit answer predictions; do not rank that as failure.
                differences = [g[cid][metric] - b[cid][metric] for cid in sorted(g.keys() & b.keys())
                               if metric in g[cid] and metric in b[cid]]
                if not differences:
                    continue
                rng = random.Random(SEED)
                draws = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(2000))
                pairs.append({'input_condition': condition, 'a': 'G', 'b': other, 'metric': metric,
                    'paired_scenes': len(differences), 'difference': mean(differences),
                    'ci95': [draws[49], draws[1949]] if len(differences) > 1 else None})
    return {'scene_means': scenes, 'paired_comparisons': pairs,
            'interpretation': 'exploratory paired scene intervals, not multiplicity-corrected acceptance tests'}


def measure(args):
    from .diagnosis_truth import evaluate
    from .diagnosis_truth_report import render_packet
    from .diagnosis_comparison_backends import calibrate_cleansight
    if not 1 <= args.limit <= 24:
        raise ValueError('limit must be 1..24; reduced runs remain explicitly labelled smoke')
    cases = load_cases(args.cases)
    receipt = read_json(args.cases.parent / 'receipt.json')
    if receipt.get('stage') != 'diagnosis-truth' or receipt.get('plan_sha256') != digest(PLAN):
        raise ValueError('requires freshly prepared independent diagnosis-truth inputs and current plan')
    if set(c['cluster_id'] for c in cases) & set(receipt['excluded_cluster_ids']):
        raise ValueError('confirmation scenes overlap development or training sources')
    fact_audit = verify_facts(cases, receipt)
    clean = load_calibration(args.calibration_rows, [c['cluster_id'] for c in cases])
    started = time.monotonic()
    with output_run(args.output) as out:
        write_json(out / 'fact-audit.json', fact_audit)
        diagnosis, identity = model_at(args)
        if diagnosis.vlm.processor.image_processor.to_dict() != receipt['preprocess']:
            raise ValueError('prepared canonical crop and model preprocessing disagree')
        identity.update(plan_sha256=digest(PLAN), cases_sha256=digest(args.cases),
            code={p.name: digest(p) for p in Path(__file__).parent.glob('diagnosis_truth*.py')})
        write_json(out / 'identity.json', identity)
        selected, screen_calls = qualification(diagnosis, cases, receipt['frozen_order'], out, limit=args.limit)
        write_json(out / 'selection.json', {'cases': selected, 'calls': screen_calls, 'pool': len(cases),
            'limit': args.limit, 'selected': len(selected), 'intervention_results_used': False})
        if not selected:
            raise ValueError('no qualified scenes; do not invent a successful diagnosis cohort')
        calibration = calibrate_cleansight(diagnosis, clean['rows'])
        calibration['seconds_per_unit'] = calibration['elapsed_seconds'] / (2 * sum(len(c['main_node_ids']) for c in selected))
        write_json(out / 'cleansight-calibration.json', calibration)
        metrics = []
        reader_assignments = []
        for ci, case in enumerate(selected):
            for qi, nid in enumerate(case['main_node_ids']):
                for condition in ('abnormal', 'clean'):
                    directory = out / (case['cluster_id'] + '-q' + str(qi) + '-' + condition)
                    directory.mkdir()
                    packet, submissions = measure_unit(diagnosis, case, nid, condition, calibration,
                                                       directory, digest(out / 'identity.json'))
                    oracle = read_json(directory / 'oracle.json')
                    scored = [costed_metrics(packet, oracle, s, evaluate(packet, oracle, s)) for s in submissions]
                    write_json(directory / 'metrics.json', scored)
                    metrics.extend(scored)
                    for mode in ('heatmap', 'table'):
                        render_packet(packet, directory / (mode + '.html'), mode=mode)
                    reader_assignments.append({'unit_id': packet['unit_id'], 'directory': directory.name,
                        'reader_A': 'heatmap' if ci % 2 == 0 else 'table',
                        'reader_B': 'table' if ci % 2 == 0 else 'heatmap',
                        'rule': 'one scene per reader in one mode; clean/abnormal and questions stay together'})
                    print(json.dumps({'unit': packet['unit_id'], 'status': 'measured_and_scored'}), flush=True)
        write_json(out / 'metrics.json', metrics)
        write_json(out / 'scene-summary.json', summarize_scene_metrics(metrics))
        write_json(out / 'reader-assignments.json', reader_assignments)
        write_json(out / 'run.json', {'status': 'completed', 'selected_scenes': len(selected),
            'run_role': 'confirmation' if args.limit == 24 and len(selected) >= 12 else 'smoke_or_small_cohort',
            'reader_status': 'not_run_awaiting_independent_readers', 'parameter_updates': 0,
            'elapsed_seconds': time.monotonic() - started, 'automatic_time_stop': False,
            'no_new_model_result_claim_until_this_run_completes': True})


def replay_history(args):
    """Real 16-subset outputs validate scoring only, never fabricate new scores."""
    from .diagnosis_truth import predictions, evaluate
    from .visual_probe_protocol import region
    from .data import read_jsonl
    rows = []
    with output_run(args.output) as out:
        for path in args.records:
            for raw in read_jsonl(path):
                case = raw.get('evidence', raw)
                for node in case['nodes']:
                    if {r['subset'] for r in node['interventions']} != set(range(16)):
                        raise ValueError('historical table is not the complete four-region universe')
                    actions, outcomes = [], {}
                    for r in node['interventions']:
                        indices = sorted(i for j in range(4) if r['subset'] & (1 << j) for i in region(2, j))
                        aid = condition_key('unit', 'same', indices)
                        actions.append({'id': aid, 'indices': indices, 'donor': 'same',
                            'known': r['subset'] in (0, 1, 2, 4, 8), 'tags': [{'kind': 'historical_2x2'}]})
                        outcomes[aid] = {'output': r['output'], 'scores': {}}
                    packet = {'schema': 'diagnosis-truth-public-v1', 'mode': 'historical_answer_only', 'unit_id': node['id'],
                        'cluster_id': case['cluster_id'], 'input_condition': 'abnormal',
                        'truth_answer': node['clean_row']['answer'], 'donor_answer': None,
                        'candidates': list(node['clean_row']['answers']) + [REFUSAL],
                        'actions': actions, 'observations': {a['id']: outcomes[a['id']] for a in actions if a['known']},
                        'maps': [], 'rankings': {}, 'historical_source_sha256': digest(path)}
                    oracle = {'unit_id': node['id'], 'rows': outcomes}
                    for method in ('nearest', 'constant', 'random', 'marker'):
                        submission = predictions(packet, method)
                        rows.append(evaluate(packet, oracle, submission))
        write_json(out / 'metrics.json', rows)
        write_json(out / 'run.json', {'status': 'completed', 'role': 'historical_evaluator_replay_only',
            'new_model_results': False, 'gpu_hours': 0, 'source_files': {str(p): digest(p) for p in args.records},
            'rows': len(rows), 'complete_candidates_rescored': False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('freeze')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('prepare')
    for name in ('lock', 'sources', 'output', 'config', 'construction-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--archive-cache')
    p = commands.add_parser('measure')
    for name in ('cases', 'config', 'calibration-rows', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cpu-test', action='store_true')
    p.add_argument('--limit', type=int, default=24)
    p = commands.add_parser('replay-history')
    p.add_argument('--records', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('grade')
    for name in ('public', 'oracle', 'submission', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        freeze_sources(args.output)
    elif args.command == 'prepare':
        prepare_sources(args)
    elif args.command == 'measure':
        measure(args)
    elif args.command == 'replay-history':
        replay_history(args)
    else:
        from .diagnosis_truth import evaluate
        if args.output.exists():
            raise FileExistsError('do not overwrite grades')
        write_json(args.output, evaluate(read_json(args.public), read_json(args.oracle), read_json(args.submission)))


if __name__ == '__main__':
    main()
