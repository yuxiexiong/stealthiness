"""Frozen diagnosis comparison, using the existing V3 model and measurement API."""
import argparse
from copy import deepcopy
import io
import json
import math
from pathlib import Path
from statistics import median
import time

from .__main__ import digest, output_run, read_json, write_json
from .diagnosis import Calls, correct, load_cases, model_at
from .report import normalize_text
from .visual_probe import Measurements, candidates_at, image_uri, manifest_at, main_maps, now
from .visual_probe_protocol import condition_key, region, _children

PROJECT = Path(__file__).resolve().parents[1]
SMOKE_IDS = {'editclevr-100161', 'editclevr-100669'}


class TimedMeasurements(Measurements):
    """Keep physical calls, while exposing generation/scoring costs separately."""
    def __init__(self, diagnosis, nodes, candidates, stream, existing=None):
        super().__init__(diagnosis, nodes, candidates, existing or {}, io.StringIO(),
                         allow_old=False, legacy={})
        self.output_stream, self.capture_costs = stream, {}

    def state(self, node_id, clean=True):
        key = (node_id, clean)
        new = key not in self.states
        result = super().state(node_id, clean)
        if new:
            self.capture_costs[key] = self.calls.rows[-1]['seconds']
        return result

    def get(self, node_id, indices, donor_id=None, force_output=None):
        if force_output is not None:
            raise ValueError('confirmation must measure actual generation cost')
        key = condition_key(node_id, (donor_id or node_id) if indices else node_id, indices)
        new = key not in self.conditions
        record = super().get(node_id, indices, donor_id)
        if new:
            record['costs'] = {
                'generate_seconds': sum(c['seconds'] for c in record['calls'] if c['operation'] == 'generate'),
                'score_seconds': sum(c['seconds'] for c in record['calls'] if c['operation'] == 'log_probs'),
                'captures': {record['donor_id']: self.capture_costs[(record['donor_id'], True)]} if indices else {}}
            self.output_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            self.output_stream.flush()
            self.stream.seek(0)
            self.stream.truncate()
        return record


def qualification(diagnosis, cases, order, out, limit=32):
    """Only labels, visibility, and unpatched responses determine eligibility."""
    calls, selected = Calls(diagnosis), []
    pool = {c['cluster_id']: c for c in cases}
    with (out / 'screening.jsonl').open('w') as stream:
        for cid in order:
            case = deepcopy(pool[cid])
            audit = {'cluster_id': cid, 'main_endpoint': case['main_endpoint'], 'nodes': [], 'accepted': False}
            for node in case['nodes']:
                node.update(cluster_id=cid, qualified=False)

            def qualify(node):
                if node.get('screened'):
                    return node['qualified']
                node['screened'] = True
                entry = {'id': node['id'], 'qualified': False}
                audit['nodes'].append(entry)
                if not node['visibility']['eligible']:
                    entry['reason'] = 'source_evidence_not_visible_or_occluded'
                    return False
                normal = calls('generate', node['clean_row'])
                entry['normal'] = normal
                if normal['stop_reason'] != 'eos' or not correct(normal, node['clean_row']):
                    entry['reason'] = 'normal_reference_incorrect_or_incomplete'
                    return False
                abnormal = calls('generate', node['observed_row'])
                entry['abnormal'] = abnormal
                if abnormal['stop_reason'] != 'eos':
                    entry['reason'] = 'abnormal_response_incomplete'
                    return False
                node.update(normal=normal, abnormal=abnormal, qualified=True,
                            role='protection' if correct(abnormal, node['observed_row']) else 'target')
                entry.update(qualified=True, abnormal_correct=correct(abnormal, node['observed_row']))
                return True

            q1 = next(n for n in case['nodes'] if n['family'] == 'changed_color'
                      and n['endpoint'] == case['main_endpoint'])
            if not qualify(q1):
                audit['reason'] = 'main_normal_reference_unqualified'
            elif q1['role'] != 'target':
                audit['reason'] = 'main_input_has_no_observed_error'
            else:
                companion = sorted((n for n in case['nodes'] if n['endpoint'] == q1['endpoint']
                                    and n['family'] in ('preserved_color', 'count')),
                                   key=lambda n: (n['family'] != 'preserved_color', n['id']))
                q2 = next((n for n in companion if qualify(n)), None)
                if q2 is None:
                    audit['reason'] = 'no_qualified_invariant_companion'
                else:
                    case['main_node_ids'] = [q1['id'], q2['id']]
                    # Only the same scene's opposite endpoints can become cross donors.
                    peers = {q1['donor_peer_id'], q2['donor_peer_id']}
                    for node in case['nodes']:
                        if node['id'] in peers:
                            qualify(node)
                        node['label'] = ('Q' + str(case['main_node_ids'].index(node['id']) + 1)
                                         if node['id'] in case['main_node_ids'] else 'peer')
                    audit['accepted'] = True
                    selected.append(case)
            stream.write(json.dumps(audit, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps({'phase': 'qualification', 'cluster': cid, 'accepted': audit['accepted'],
                              'selected': len(selected), 'reason': audit.get('reason')}), flush=True)
            if len(selected) == limit:
                break
    return selected, calls.rows


def measurement_at(diagnosis, case, stream):
    nodes = {n['id']: n for n in case['nodes']}
    candidates = {key: candidates_at(diagnosis, node, nodes) for key, node in nodes.items()
                  if node.get('qualified', bool(node.get('normal') and node.get('abnormal')))}
    return TimedMeasurements(diagnosis, nodes, candidates, stream)


def gradient_provider(measurements):
    from .diagnosis_comparison_backends import atp_scores
    results = []

    def evaluate(node_id, background, donor_id=None):
        donor_id = donor_id or node_id
        donor = measurements.state(donor_id)
        result = atp_scores(measurements.diagnosis, measurements.nodes[node_id],
                            measurements.candidates[node_id], donor, background)
        result['captures'] = {donor_id: measurements.capture_costs[(donor_id, True)]}
        results.append(dict(result, node_id=node_id, donor_id=donor_id))
        return result
    return evaluate, results


def initial_map(measurements, node_ids):
    keys = set()
    for node_id in node_ids:
        for mapping in main_maps(node_id):
            masks = [mapping['background']] + [sorted(set(mapping['background']) | set(c['indices']))
                                                for c in mapping['cells']]
            for indices in masks:
                keys.add(measurements.get(node_id, indices)['key'])
    return keys


def budget_calibration(diagnosis, development, out):
    """Eight old scenes, one full map and four new matched pairs per scene."""
    values, tolerances, records = [], [], []
    for case in development['cases']:
        with (out / (case['cluster_id'] + '.jsonl')).open('w') as stream:
            measured = measurement_at(diagnosis, case, stream)
            controls = {}
            if case['cluster_id'] in SMOKE_IDS:
                for nid in case['main_node_ids']:
                    # Old interface controls are separate from confirmation/cost evidence.
                    scratch = Measurements(diagnosis, measured.nodes, measured.candidates, {}, io.StringIO(),
                                           allow_old=False, legacy={})
                    controls[nid] = scratch.controls(nid)
                    tolerances.append(controls[nid]['tolerance']['fact'])
                provider, gradients = gradient_provider(measured)
                for nid in case['main_node_ids']:
                    for background in ([], region(2, 0)):
                        approximation = provider(nid, background)
                        if (len(approximation['scores']) != 36 or not all(map(math.isfinite, approximation['scores']))
                                or any(approximation['scores'][i] != 0 for i in range(36)
                                       if set(region(6, i)) <= set(background))):
                            raise RuntimeError('AtP grid or background-zero smoke control failed')
                        for label in ('positive', 'negative'):
                            exact = measured.calls('log_probs', measured.nodes[nid]['observed_row'],
                                measured.candidates[nid][label]['token_ids'], donor=measured.state(nid),
                                visual_indices=background)
                            difference = abs(exact['sum_log_prob'] - approximation['candidate_log_probs'][label]['sum_log_prob'])
                            if difference > 1e-4:
                                raise RuntimeError('AtP and exact complete-candidate scores differ: ' + nid)
                write_json(out / (case['cluster_id'] + '-atp-smoke.json'), gradients)
                # Clear AtP donor cache for independent complete-map calibration cost.
                measured.states.clear()
                measured.capture_costs.clear()
            start = len(measured.calls.rows)
            initial_map(measured, case['main_node_ids'])
            # Four adjacent equal-area candidate/control pairs, outside initial maps.
            for a, b in ((0, 1), (4, 5), (6, 7), (10, 11), (18, 19), (22, 23), (30, 31), (34, 35)):
                mask = sorted(set(region(6, a)) | set(region(6, b)))
                for nid in case['main_node_ids']:
                    measured.get(nid, mask)
            seconds = sum(c['seconds'] for c in measured.calls.rows[start:])
            values.append(seconds)
            records.append({'cluster_id': case['cluster_id'], 'seconds': seconds,
                            'conditions': len(measured.conditions), 'controls': controls})
            print(json.dumps({'phase': 'old_case_calibration', 'cluster': case['cluster_id'],
                              'seconds': seconds}), flush=True)
    calibration = {'B_seconds': median(values), 'per_scene': records,
        'effect_tolerance': max([1e-4] + [v * 10 for v in tolerances]),
        'rule': 'median old complete two-background map + 8 operation conditions x 2 questions',
        'confirmation_outcomes_used': False, 'parameter_updates': 0}
    write_json(out / 'budget.json', calibration)
    return calibration


def reader_spec(case, contract):
    """Freeze the task without looking at any intervention results."""
    prepared = contract['prepared']
    names = ['spatial', 'side_effect'] + (['source'] if prepared['source_applicable'] else [])
    task = names[prepared['seed'] % len(names)]
    grouped = {}
    for op in prepared['holdout'][task]['checks']:
        key = (op.get('direction', ''), op['role'], tuple(op['indices']))
        if key not in grouped:
            grouped[key] = {'id': task + '-' + str(len(grouped)), 'label': ' '.join(key[:2]),
                            'background': [], 'indices': op['indices'], 'node_ids': [], 'donor_ids': {}}
        grouped[key]['node_ids'].append(op['node_id'])
        grouped[key]['donor_ids'][op['node_id']] = op['donor_id']
    return {'task': task, 'node_ids': case['main_node_ids'], 'reserved_checks': list(grouped.values()),
            'initial_condition_keys': {nid: sorted({key for m in main_maps(nid)
                for key in [m['base_key'], *(c['key'] for c in m['cells'])]}) for nid in case['main_node_ids']},
            'donor_options': {nid: [{'id': n, 'label': '本图正常供体' if n == nid else '同场景另一颜色供体'}
                for n in (nid, next(x for x in case['nodes'] if x['id'] == nid)['donor_peer_id'])
                if any(x['id'] == n and x.get('qualified') for x in case['nodes'])] for nid in case['main_node_ids']}}


def report_case(case, measurements, comparison, reader):
    nodes = []
    for nid, node in measurements.nodes.items():
        if nid not in measurements.candidates:
            continue
        conditions = {k: v for k, v in measurements.conditions.items() if v['node_id'] == nid}
        maps = main_maps(nid) if nid in case['main_node_ids'] else []
        parents = {m['local_parent'] for m in comparison['methods'].values() if m['local_parent'] is not None}
        if nid in case['main_node_ids']:
            for parent in sorted(parents):
                maps.append({'id': f'{nid}-local-{parent}', 'title': f'局部12×12：父格{parent}',
                    'resolution': 12, 'background': [], 'base_key': condition_key(nid, nid, []),
                    'parent_index': parent, 'cells': [
                        {'index': i, 'indices': region(12, i), 'key': condition_key(nid, nid, region(12, i)),
                         'status': 'unmeasured'} for i in _children(parent)]})
        for mapping in maps:
            for cell in mapping['cells']:
                if cell['status'] != 'included':
                    cell['status'] = conditions.get(cell['key'], {}).get('status', 'unmeasured')
        nodes.append({'id': nid, 'label': node['label'], 'question': node['observed_row']['question'],
            'answer': node['observed_row']['answer'], 'candidates': measurements.candidates[nid],
            'images': dict({kind: image_uri(measurements.diagnosis.processed_image(node[row]))
                       for kind, row in (('clean', 'clean_row'), ('observed', 'observed_row'))},
                       coordinate_system='model_processed_pixels'),
            'grid': measurements.state(nid)['grid'], 'annotations': node.get('annotations', {}),
            'maps': maps, 'conditions': conditions, 'tolerance': {'fact': comparison['contract']['effect_tolerance']}})
    return {'cluster_id': case['cluster_id'], 'annotations': case['annotations'], 'nodes': nodes,
            'comparison': comparison, 'reader': reader}


def external_reference(diagnosis, cases, calibration, out, *, smoke=False):
    from .diagnosis_comparison_backends import external_generate, ExternalMethodBlocked
    from tools.build_benign_baseline import TARGET
    result = {'status': 'completed', 'methods': {}, 'scope': 'original purification actions; adapted HF interface',
              'target_response': TARGET, 'target_metric': 'exact original VQA-normalized training target, not a general refusal judge'}
    for method in ('purmm', 'cleansight'):
        records = []
        try:
            if method == 'cleansight':
                if calibration.get('status') == 'blocked':
                    raise ExternalMethodBlocked(calibration['reason'])
            for case in cases:
                for node in case['nodes']:
                    if node['id'] not in case['main_node_ids']:
                        continue
                    for kind in ('clean', 'observed'):
                        row = node[kind + '_row']
                        raw = external_generate(diagnosis, row, method, calibration=calibration)
                        if smoke and raw['output'].get('stop_reason') not in ('eos', 'max_new_tokens'):
                            raise RuntimeError(method + ' smoke returned an invalid completion record')
                        record = dict(raw, cluster_id=case['cluster_id'], node_id=node['id'], condition=kind,
                                      correct=correct(raw['output'], row) if raw['output']['stop_reason'] == 'eos' else None,
                                      target_response_remaining=(normalize_text(raw['output']['text']) ==
                                          normalize_text(TARGET)) if raw['output']['stop_reason'] == 'eos' else None)
                        record['clean_harm'] = (record['correct'] is False) if kind == 'clean' else None
                        records.append(record)
                        with (out / (method + '.jsonl')).open('a') as stream:
                            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            result['methods'][method] = {'status': 'completed', 'records': records,
                'summary': {kind: {'questions': len(rows), 'correct': sum(r['correct'] is True for r in rows),
                    'target_response_remaining': sum(r['target_response_remaining'] is True for r in rows),
                    'incomplete': sum(r['correct'] is None for r in rows),
                    'seconds': sum(r['elapsed_seconds'] for r in rows)} for kind in ('clean', 'observed')
                    for rows in [[r for r in records if r['condition'] == kind]]}}
        except ExternalMethodBlocked as error:
            result['status'] = 'partial_external_reference'
            result['methods'][method] = {'status': 'blocked', 'reason': str(error), 'records': records}
        print(json.dumps({'phase': 'external', 'method': method, 'status': result['methods'][method]['status']}), flush=True)
    write_json(out / 'external.json', result)
    return result


def load_calibration(path, confirmation_ids):
    manifest = read_json(path)
    if (manifest.get('schema') != 'diagnosis-external-calibration-v1'
            or manifest.get('image_condition') != 'clean' or len(manifest['rows']) != 200
            or manifest.get('model_outputs_used') is not False):
        raise ValueError('external calibration needs 200 frozen, clean, unselected-by-model rows')
    if set(confirmation_ids) & {r['cluster_id'] for r in manifest['rows']}:
        raise ValueError('external calibration overlaps the new confirmation scene pool')
    for name, sha in manifest['images'].items():
        if digest(path.parent / name) != sha:
            raise ValueError('calibration image changed: ' + name)
    seen = set()
    for row in manifest['rows']:
        if row['image'] not in manifest['images']:
            raise ValueError('calibration row refers to an unregistered image')
        key = (row['image'], row['question'], row['task'])
        if key in seen:
            raise ValueError('duplicate calibration image/question/task')
        seen.add(key)
        row['image'] = str((path.parent / row['image']).resolve())
    return manifest


def run(args):
    from .diagnosis_comparison_protocol import compare_case, prepare_case_contract, summarize_comparisons
    from .diagnosis_comparison_report import render, render_reader_materials, make_reader_assignments
    start = time.monotonic()
    with output_run(args.output) as out:
        prepared = load_cases(args.cases)
        receipt = read_json(args.cases.parent / 'receipt.json')
        calibration_inputs = load_calibration(args.calibration_rows, [c['cluster_id'] for c in prepared])
        sources = {str(path.relative_to(PROJECT)): digest(path) for path in
                   [*PROJECT.glob('repair/diagnosis_comparison*.py'),
                    *PROJECT.glob('external/*/SOURCE.json'),
                    PROJECT / 'repair/visual_probe.py', PROJECT / 'repair/visual_probe_protocol.py',
                    PROJECT / 'tools/prepare_diagnosis_comparison.py',
                    PROJECT / 'tools/build_benign_baseline.py',
                    PROJECT / 'DIAGNOSIS_COMPARISON_CONTRACT_2026-09-14.md']}
        write_json(out / 'input-identity.json', {'code_and_sources': sources,
            'config_sha256': digest(args.config), 'development_sha256': digest(args.development),
            'prepared_sha256': digest(args.cases), 'prepared_receipt_sha256': digest(args.cases.parent / 'receipt.json'),
            'calibration_sha256': digest(args.calibration_rows),
            'created_utc': now(), 'parameter_updates': 0})
        diagnosis, identity = model_at(args)
        development = manifest_at(args.development)
        if any(identity[k] != development['historical_identity'][k] for k in
               ('asset_identity', 'generation', 'effective_generation', 'model_spec')):
            raise ValueError('model or generation differs from qualified old development evidence')
        if diagnosis.vlm.processor.image_processor.to_dict() != development['preprocess']:
            raise ValueError('model preprocessing differs from the frozen canonical source transform')
        calibration_dir = out / 'old-calibration'
        calibration_dir.mkdir()
        calibration = budget_calibration(diagnosis, development, calibration_dir)
        from .diagnosis_comparison_backends import calibrate_cleansight, ExternalMethodBlocked
        try:
            external_calibration = calibrate_cleansight(diagnosis, calibration_inputs['rows'], expected_samples=200)
        except ExternalMethodBlocked as error:
            external_calibration = {'status': 'blocked', 'reason': str(error)}
        write_json(out / 'cleansight-calibration.json', external_calibration)
        smoke_dir = out / 'external-smoke'
        smoke_dir.mkdir()
        external_smoke = external_reference(diagnosis, [c for c in development['cases']
                                           if c['cluster_id'] in SMOKE_IDS], external_calibration, smoke_dir, smoke=True)
        print(json.dumps({'phase': 'smoke_passed', 'B_seconds': calibration['B_seconds']}), flush=True)
        write_json(out / 'progress.json', {'phase': 'smoke_passed', 'calibration': calibration,
                                          'external_status': external_smoke['status'], 'updated_utc': now()})
        selected, screen_calls = qualification(diagnosis, prepared, receipt['frozen_order'], out)
        contracts = {c['cluster_id']: prepare_case_contract(c, {'effect_tolerance': calibration['effect_tolerance']})
                     for c in selected}
        readers = {c['cluster_id']: reader_spec(c, contracts[c['cluster_id']]) for c in selected}
        assignments = make_reader_assignments(selected) if len(selected) >= 16 else None
        frozen = {'schema': 'diagnosis-comparison-v1', 'cases': selected, 'contracts': contracts,
                  'readers': readers, 'assignments': assignments, 'budget': calibration,
                  'identity': identity, 'prepared_sha256': digest(args.cases), 'created_utc': now(),
                  'input_identity_sha256': digest(out / 'input-identity.json'),
                  'parameter_updates': 0, 'confirmation_interventions_seen': False}
        write_json(out / 'frozen.json', frozen)
        report = {'schema': 'diagnosis-comparison-v1', 'status': 'running', 'created_utc': now(), 'cases': [],
                  'checks': [], 'diagnoses': [], 'comparisons': [], 'cost': {}, 'cpu_test': False,
                  'scope': '32 target fresh scene families; finite injected-state diagnosis only',
                  'qualification': {'target': 32, 'selected': len(selected), 'pool': 96},
                  'reader_status': 'awaiting_real_readers' if assignments else 'fewer_than_16_qualified_families'}
        for index, case in enumerate(selected):
            cid = case['cluster_id']
            with (out / (cid + '-measurements.jsonl')).open('w') as stream:
                measured = measurement_at(diagnosis, case, stream)
                provider, gradients = gradient_provider(measured)
                comparison = compare_case(case, measured, provider, calibration['B_seconds'], contracts[cid])
                write_json(out / (cid + '-comparison.json'), comparison)
                write_json(out / (cid + '-atp.json'), gradients)
                # Fill any map cells omitted by a budget-limited strategy for the H/T common packet.
                # These post-comparison observations cannot change a method's frozen decisions.
                initial_map(measured, case['main_node_ids'])
                reader = deepcopy(readers[cid])
                if reader['task'] == 'side_effect' and comparison['contract'].get('side_effect_applicable') is False:
                    reader['task_status'] = 'not_applicable_fixed_background_not_correct'
                for op in reader['reserved_checks']:
                    op['result_existed_at_registration'] = False
                report['cases'].append(report_case(case, measured, comparison, reader))
            write_json(out / 'report.json', report)
            write_json(out / 'progress.json', {'phase': 'comparison', 'completed_cases': index + 1,
                                               'selected_cases': len(selected), 'updated_utc': now()})
            print(json.dumps({'phase': 'comparison', 'completed': index + 1, 'selected': len(selected), 'cluster': cid}), flush=True)
        external = external_reference(diagnosis, selected, external_calibration, out)
        report['comparison'] = summarize_comparisons(report['cases'])
        write_json(out / 'summary.json', report['comparison'])
        issues = ([] if len(selected) == 32 else ['qualification_shortfall'])
        if external['status'] != 'completed':
            issues.append('external_blockers')
        report.update(status='completed' if not issues else 'completed_with_' + '_and_'.join(issues),
                      external=external, cost={'elapsed_seconds': time.monotonic() - start,
                      'allocated_gpu_hours': (time.monotonic() - start) / 3600, 'automatic_time_stop': False,
                      'qualification_calls': screen_calls})
        write_json(out / 'report.json', report)
        render(report, out / 'report' / 'index.html')
        if assignments:
            render_reader_materials(report, assignments, out / 'readers')
        write_json(out / 'run.json', {'status': report['status'], 'report_sha256': digest(out / 'report.json'),
                    'frozen_sha256': digest(out / 'frozen.json'), 'identity': identity, 'parameter_updates': 0,
                    'input_identity_sha256': digest(out / 'input-identity.json'),
                    'reader_status': report['reader_status'], 'elapsed_seconds': time.monotonic() - start})
        write_json(out / 'progress.json', {'phase': 'completed', 'selected_cases': len(selected),
                                           'updated_utc': now(), 'reader_status': report['reader_status']})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'development', 'cases', 'calibration-rows', 'output'):
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    args.cpu_test = False
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
