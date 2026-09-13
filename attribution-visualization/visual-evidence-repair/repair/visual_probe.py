"""V3 visual probe: prepare old cases, map, freeze a new question, then check it.

Reuses the A/B model and original VQA scorer. No training or server scheduler.
Run from visual-evidence-repair: python -m repair.visual_probe --help.
"""
import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time

from .__main__ import digest, output_run, read_json, write_json
from .data import read_jsonl
from .diagnosis import Calls, correct, load_cases, model_at
from .report import normalize_text
from .visual_probe_protocol import (GRID, condition_key, main_maps, region,
                                    validate_batch, numeric_order, compare_orders, additive_effect)

PROJECT = Path(__file__).resolve().parents[1]
SELECTION = [('editclevr-100669', 1), ('editclevr-335200', 1), ('editclevr-336805', 0),
             ('editclevr-101050', 0), ('editclevr-100161', 0), ('editclevr-100716', 0),
             ('editclevr-101232', 1), ('editclevr-400305', 0)]
WRONG = {'editclevr-100669': 'red', 'editclevr-335200': 'purple', 'editclevr-336805': 'black'}
COLORS = ('blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow')
CANDIDATES = ('positive', 'negative', 'refusal')


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def code_identity():
    return {name: digest(Path(__file__).with_name(name)) for name in
            ('visual_probe.py', 'visual_probe_protocol.py', 'diagnosis_model.py', 'model.py', 'report.py')}


def object_annotations(node, preprocess):
    """Use the existing canonical CLIP mask transform; no segmentation model."""
    import numpy as np
    from PIL import Image
    from transformers import CLIPImageProcessor
    from types import SimpleNamespace
    from tools.build_benign_baseline import canonical_image
    source = node['visibility']['sources']['mask']
    path = Path(source['path'])
    if not path.is_file():
        return {'status': 'needs_manual_annotation', 'reason': 'original instance mask unavailable'}
    if digest(path) != source['sha256']:
        raise ValueError('instance mask changed since factual qualification')
    processor = SimpleNamespace(image_processor=CLIPImageProcessor.from_dict(preprocess))
    objects = []
    with np.load(path, allow_pickle=False) as data:
        for index in node['visibility']['required_object_ids']:
            mask = data['masks'][index]
            rgb = Image.fromarray(np.repeat((mask.astype(bool) * 255).astype(np.uint8)[..., None], 3, axis=2))
            transformed = canonical_image(processor, rgb)
            ys, xs = np.where(np.asarray(transformed)[..., 0] > 0)
            if not len(xs):
                raise ValueError('qualified object disappeared from its canonical mask')
            objects.append({'object_id': index, 'box': [int(xs.min()), int(ys.min()),
                                                        int(xs.max()) + 1, int(ys.max()) + 1]})
    return {'status': 'source_mask_verified', 'objects': objects, 'pixel_size': list(transformed.size),
            'source': 'original instance mask + existing canonical CLIP transform', 'mask_sha256': source['sha256']}


def prepare(args):
    """Copy only selected verified assets; old full tables stay explicitly known."""
    pools, preprocess = {}, None
    for path in args.pools:
        receipt = read_json(path.parent / 'receipt.json')
        if preprocess is not None and preprocess != receipt['preprocess']:
            raise ValueError('A/B preprocessing differs')
        preprocess = receipt['preprocess']
        for case in load_cases(path):
            if case['cluster_id'] in pools:
                raise ValueError('duplicate base scene in input pools')
            pools[case['cluster_id']] = case
    evidence, identities, sources = {}, [], {}
    for path in args.records:
        run = read_json(path.parent / 'run.json')
        if run['status'] != 'completed':
            raise ValueError('reuse only completed A/B records')
        if run.get('records_sha256', digest(path)) != digest(path):
            raise ValueError('old records changed')
        identities.append(run['identity'])
        sources[str(path.resolve())] = digest(path)
        sources[str((path.parent / 'run.json').resolve())] = digest(path.parent / 'run.json')
        for raw in read_jsonl(path):
            case = raw.get('evidence', raw)
            if case['cluster_id'] in evidence:
                raise ValueError('duplicate historical scene')
            evidence[case['cluster_id']] = case
    binding = {key: identities[0][key] for key in
               ('asset_identity', 'generation', 'effective_generation', 'model_spec')}
    if any(any(identity[key] != value for key, value in binding.items()) for identity in identities):
        raise ValueError('historical runs have different model/generation identities')
    selected, image_files = [], {}
    notes = read_json(args.notes) if args.notes else {}
    with output_run(args.output) as out:
        for cluster, endpoint in SELECTION[:args.count]:
            old, pool = evidence[cluster], pools[cluster]
            originals = {n['id']: n for n in pool['nodes']}
            q1 = next(n for n in old['nodes'] if n['family'] == 'changed_color' and n['endpoint'] == endpoint)
            companions = sorted((n for n in old['nodes'] if n['endpoint'] == endpoint
                                 and n['family'] in ('preserved_color', 'count')),
                                key=lambda n: (n['family'] != 'preserved_color', n['id']))
            if not companions:
                raise ValueError(cluster + ': no qualified same-image second question')
            ids = [q1['id'], companions[0]['id']]
            item = {'cluster_id': cluster, 'main_node_ids': ids, 'nodes': [],
                    'initial_note': notes.get(cluster, {
                        'judgment': '旧四区最小成功集合为 ' + str(old['summary']['minimal_sets'])
                                    + '；这尚不能说明细区的信息关系。',
                        'next_check': '先按原四区经验检查左上及已知反例，再用完整6×6图判断是否需要改变检查。'}),
                    'annotations': {'coordinate_system': 'model_processed_pixels',
                                    'marker_box': [0, 0, 64, 64],
                                    'source': 'original canonical CLIP crop then top-left 64x64 marker'},
                    'legacy_summary': old['summary'], 'legacy_swaps': old['donor_swaps']}
            for name in ('judgment', 'next_check'):
                if not str(item['initial_note'].get(name, '')).strip():
                    raise ValueError('initial note must be recorded before looking at maps')
            for historical in old['nodes']:
                node = deepcopy(historical)
                node['cluster_id'] = cluster
                node['label'] = 'Q' + str(ids.index(node['id']) + 1) if node['id'] in ids else 'peer'
                for kind, row_key in (('clean', 'clean_row'), ('observed', 'observed_row')):
                    source = originals[node['id']][row_key]
                    for k in ('question', 'answer', 'answers', 'task'):
                        if source[k] != node[row_key][k]:
                            raise ValueError('historical node and prepared pool disagree: ' + node['id'])
                    sha = digest(source['image'])
                    if node['images'][kind]['sha256'] != sha:
                        raise ValueError('historical image hash differs')
                    relative = 'images/' + sha + '.png'
                    destination = out / relative
                    if not destination.exists():
                        destination.parent.mkdir(exist_ok=True)
                        shutil.copyfile(source['image'], destination)
                    image_files[relative] = sha
                    node[row_key] = dict(source, image=relative)
                if (not node['visibility']['eligible'] or not correct(node['normal'], node['clean_row'])
                        or node['normal']['stop_reason'] != 'eos' or node['abnormal']['stop_reason'] != 'eos'):
                    raise ValueError('historical node has no qualified complete reference')
                node['annotations'] = object_annotations(node, preprocess)
                item['nodes'].append(node)
            selected.append(item)
        manifest = {'schema': 'visual-probe-v3', 'status': 'prepared_old_exploration_cases',
                    'created_utc': now(), 'cases': selected, 'images': image_files,
                    'preprocess': preprocess, 'historical_identity': binding, 'sources': sources,
                    'plan_sha256': digest(PROJECT / 'DIAGNOSIS_PREDICTION_EXPERIMENT_PLAN.md'),
                    'count': args.count, 'parameter_updates': 0, 'gpu_hours': 0}
        write_json(out / 'manifest.json', manifest)
    return manifest


def manifest_at(path):
    manifest = read_json(path)
    if manifest['schema'] != 'visual-probe-v3' or manifest['count'] not in (6, 8):
        raise ValueError('requires the frozen V3 6/8-case input manifest')
    for relative, sha in manifest['images'].items():
        if digest(path.parent / relative) != sha:
            raise ValueError('prepared image changed: ' + relative)
    for case in manifest['cases']:
        for node in case['nodes']:
            for key in ('clean_row', 'observed_row'):
                node[key]['image'] = str((path.parent / node[key]['image']).resolve())
    return manifest


def nodes_at(manifest):
    return {n['id']: n for case in manifest['cases'] for n in case['nodes']}


def candidates_at(diagnosis, node, nodes):
    positive = node['normal']['text']
    truth = normalize_text(node['clean_row']['answer'])
    if node['family'] == 'changed_color':
        value = WRONG.get(node['cluster_id'], nodes[node['donor_peer_id']]['clean_row']['answer'])
    elif node['family'] == 'preserved_color':
        value = next(color for color in COLORS if color != truth)
    else:
        value = '0' if truth != '0' else '1'
    # Existing VQA normalization validates semantics; keep the same short answer frame.
    if truth not in positive.lower():
        if normalize_text(positive) != truth:
            raise ValueError('normal answer cannot supply a factual candidate frame')
        negative = value
    else:
        start = positive.lower().index(truth)
        negative = positive[:start] + value + positive[start + len(truth):]
    if correct({'text': negative}, node['clean_row']):
        raise ValueError('negative candidate is factually correct')
    texts = {'positive': positive, 'negative': negative, 'refusal': node['abnormal']['text']}
    result = {}
    for label, text in texts.items():
        if label in ('positive', 'refusal'):
            ids = node['normal' if label == 'positive' else 'abnormal']['token_ids']
            if diagnosis.vlm.processor.tokenizer.decode(ids, skip_special_tokens=True) != text:
                raise ValueError('saved continuation and current tokenizer disagree: ' + node['id'])
        else:
            # Reuse the existing exact prompt-prefix/EOS validator, not bare encode().
            prepared = diagnosis.vlm.prepare(dict(node['observed_row'], answers=[text], answer=text))
            labels = prepared['labels'][0]
            ids = labels[labels != -100].tolist()
        result[label] = {'text': text, 'token_ids': ids,
                         'source': 'constructed_fixed_fact' if label == 'negative' else 'historical_generation'}
    return result


def old_key_output(node):
    return {condition_key(node['id'], node['id'], sorted({i for j in range(4) if row['subset'] & (1 << j)
                                                       for i in region(2, j)})): row['output']
            for row in node['interventions']}


def legacy_outputs(manifest):
    outputs = {}
    for case in manifest['cases']:
        nodes = {n['id']: n for n in case['nodes']}
        for node in nodes.values():
            outputs.update(old_key_output(node))
        for swap in case['legacy_swaps']:
            if swap['status'] == 'measured':
                indices = sorted({i for j in range(4) if swap['subset'] & (1 << j) for i in region(2, j)})
                outputs.setdefault(condition_key(swap['node_id'], swap['donor_id'], indices), swap['output'])
    return outputs


def legacy_keys(manifest):
    return set(legacy_outputs(manifest))


def prior_at(paths, manifest_sha):
    reports, receipts = [], []
    for path in paths:
        run = read_json(path / 'run.json')
        if run['status'] not in ('completed', 'completed_with_unresolved'):
            raise ValueError('prior run did not finish; do not treat partial data as a full map')
        if run['manifest_sha256'] != manifest_sha or run['report_sha256'] != digest(path / 'report.json'):
            raise ValueError('prior result identity changed')
        if run['code_identity'] != code_identity():
            raise ValueError('scoring implementation differs from prior measurements')
        reports.append(read_json(path / 'report.json'))
        receipts.append({'path': str(path.resolve()), 'run_sha256': digest(path / 'run.json'),
                         'report_sha256': digest(path / 'report.json')})
    return reports, receipts


def merge_prior(reports):
    conditions, candidates, previous, views = {}, {}, [], {}
    for report in reports:
        for case in report['cases']:
            for node in case['nodes']:
                old = candidates.setdefault(node['id'], node['candidates'])
                if old != node['candidates']:
                    raise ValueError('candidate changed between prior runs')
                views[node['id']] = node
                for key, value in node['conditions'].items():
                    if key in conditions and conditions[key] != value:
                        raise ValueError('same condition has conflicting saved outcomes')
                    conditions[key] = value
        for batch in report.get('batch_history', [report['batch']] if report.get('batch') else []):
            if batch not in previous:
                previous.append(batch)
    return conditions, candidates, previous, views


def freeze(args):
    manifest = manifest_at(args.manifest)
    reports, receipts = prior_at(args.prior, digest(args.manifest))
    conditions, _, previous, views = merge_prior(reports)
    required = {n['id'] for n in nodes_at(manifest).values() if n['label'] in ('Q1', 'Q2')}
    if not required <= views.keys() or any(report['status'] == 'smoke' for report in reports):
        raise ValueError('freeze after complete exploration maps, not a smoke or one missing lane')
    request = read_json(args.request)
    if request.get('read_map_seconds') is not None and (
            type(request['read_map_seconds']) not in (int, float)
            or not math.isfinite(request['read_map_seconds']) or request['read_map_seconds'] < 0):
        raise ValueError('read_map_seconds must be an actual finite nonnegative duration or null')
    batch = validate_batch(request, nodes_at(manifest), previous, legacy_keys(manifest) | conditions.keys())
    if not batch['checks']:
        raise ValueError('no new checks were registered; keep an observation note rather than an empty run')
    map_ids = {m['id'] for n in views.values() for m in n['maps']}
    for diagnosis in batch['diagnoses']:
        if not diagnosis['source_maps'] or not set(diagnosis['source_maps']) <= map_ids:
            raise ValueError('diagnosis must cite maps already available before the check')
    by_id = {c['id']: c for c in batch['checks']}
    for comparison in batch.get('comparisons', []):
        menu = [by_id[key] for key in comparison['visual_order']]
        comparison['numeric_order'] = numeric_order(menu, comparison['node_id'], conditions, comparison['direction'])
    # Both directions are mandatory for each selected source-dependence check.
    if batch['stage'] == 'donor':
        operations = {(node_id, check['donor_ids'][node_id], tuple(sorted(set(check['background']) | set(check['indices']))))
                      for check in batch['checks'] for node_id in check['node_ids']}
        nodes = nodes_at(manifest)
        for node_id, donor_id, indices in operations:
            if (donor_id, node_id, indices) not in operations or nodes[node_id]['donor_peer_id'] != donor_id:
                raise ValueError('source test requires both opposite qualified peer directions')
    frozen = {'schema': 'visual-probe-v3', 'status': 'frozen_before_new_checks', 'created_utc': now(),
              'manifest_sha256': digest(args.manifest), 'prior': receipts,
              'request_sha256': digest(args.request), 'code_identity': code_identity(), 'batch': batch}
    with output_run(args.output) as out:
        write_json(out / 'batch.json', frozen)
    return frozen


def image_uri(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


class Measurements:
    """One cache per exact input/donor/mask, plus the existing A/B timed backend."""
    def __init__(self, diagnosis, nodes, candidates, existing, stream, allow_old=True, *, legacy=None):
        self.diagnosis, self.nodes, self.candidates = diagnosis, nodes, candidates
        self.conditions, self.stream = deepcopy(existing), stream
        self.calls, self.states = Calls(diagnosis), {}
        self.allow_old = allow_old
        self.old_outputs = legacy if legacy is not None else {
            key: output for node in nodes.values() for key, output in old_key_output(node).items()}

    def state(self, node_id, clean=True):
        key = (node_id, clean)
        if key not in self.states:
            row = self.nodes[node_id]['clean_row' if clean else 'observed_row']
            self.states[key] = self.calls('capture', row)
        return self.states[key]

    def scores(self, node_id, indices, donor_id):
        node = self.nodes[node_id]
        donor = self.state(donor_id) if indices else None
        scores = {}
        cached = {}
        for label, candidate in self.candidates[node_id].items():
            ids = tuple(candidate['token_ids'])
            if ids not in cached:
                raw = self.calls('log_probs', node['observed_row'], list(ids), donor=donor, visual_indices=indices)
                self.calls.rows[-1]['scored_tokens'] = len(ids)
                self.calls.rows[-1]['output_tokens'] = 0
                cached[ids] = {'log_prob': raw['sum_log_prob'], 'values': raw['token_log_probs'],
                               'reasons': raw['reasons'], 'token_ids': list(ids)}
            scores[label] = cached[ids]
        positive = scores['positive']['log_prob']
        def difference(label):
            other = scores[label]['log_prob']
            return None if positive is None or other is None else positive - other
        return scores, difference('negative'), difference('refusal')

    def get(self, node_id, indices, donor_id=None, force_output=None):
        donor_id = (donor_id or node_id) if indices else node_id
        key = condition_key(node_id, donor_id, indices)
        if key in self.conditions:
            return self.conditions[key]
        start = len(self.calls.rows)
        node = self.nodes[node_id]
        old = self.old_outputs.get(key) if self.allow_old else None
        output = force_output or old
        source = 'current_control' if force_output is not None else 'historical_generation_rescored' if old else 'new_measurement'
        if output is None:
            output = self.calls('generate', node['observed_row'], donor=self.state(donor_id) if indices else None,
                                visual_indices=indices)
        scores, fact, refusal = self.scores(node_id, indices, donor_id)
        verdict = correct(output, node['observed_row']) if output['stop_reason'] == 'eos' else None
        record = {'key': key, 'node_id': node_id, 'indices': indices, 'donor_id': donor_id,
                  'status': 'measured' if fact is not None and refusal is not None else 'failed',
                  'source': source, 'output': output, 'correct': verdict, 'scores': scores,
                  'fact': fact, 'refusal': refusal, 'calls': self.calls.rows[start:], 'created_utc': now()}
        self.conditions[key] = record
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()
        return record

    def controls(self, node_id):
        node = self.nodes[node_id]
        empty = self.calls('generate', node['observed_row'], visual_indices=[])
        whole = self.calls('generate', node['observed_row'], donor=self.state(node_id),
                           visual_indices=list(range(GRID * GRID)))
        selfcopy = self.calls('generate', node['observed_row'], donor=self.state(node_id, False),
                              visual_indices=list(range(GRID * GRID)))
        if (empty['token_ids'] != node['abnormal']['token_ids'] or whole['token_ids'] != node['normal']['token_ids']
                or selfcopy['token_ids'] != empty['token_ids']):
            raise RuntimeError('empty/full/self-copy control failed: ' + node_id)
        base = self.get(node_id, [], force_output=empty)
        self.get(node_id, list(range(GRID * GRID)), force_output=whole)
        _, fact, refusal = self.scores(node_id, [], node_id)
        if fact is None or refusal is None or base['fact'] is None or base['refusal'] is None:
            raise RuntimeError('numerical repeat control has no finite baseline')
        return {'empty_replay_exact': True, 'full_visual_matches_normal': True,
                'self_copy_exact': True, 'self_copy_output': selfcopy,
                'tolerance': {'fact': abs(fact - base['fact']), 'refusal': abs(refusal - base['refusal'])}}


def check_specs(batch):
    """Expand only declared checks; a refinement means four child operations."""
    for check in batch['checks']:
        background = check['background']
        groups = [(None, check['indices'])]
        if check['kind'] == 'refine':
            row, col = divmod(check['parent_index'], 6)
            children = [(row * 2 + dr) * 12 + col * 2 + dc for dr in range(2) for dc in range(2)]
            groups = [(index, region(12, index)) for index in children]
        for node_id in check['node_ids']:
            donor_id = check.get('donor_ids', {}).get(node_id, node_id)
            for child, indices in groups:
                prediction = (check.get('child_predictions', {}).get(str(child), {})
                              if check['kind'] == 'refine' else check['prediction'])
                yield dict(check, prediction=prediction, node_id=node_id, donor_id=donor_id, child_index=child,
                           indices=indices, total_indices=sorted(set(background) | set(indices)),
                           key=condition_key(node_id, donor_id, sorted(set(background) | set(indices))),
                           base_key=condition_key(node_id, donor_id, background))


def predicate(expected, record):
    if not record or record.get('output', {}).get('stop_reason') != 'eos':
        return None
    checks = []
    if 'answer' in expected:
        checks.append(normalize_text(record['output']['text']) == normalize_text(expected['answer']))
    if 'correct' in expected:
        checks.append(record['correct'] == expected['correct'])
    return all(checks) if checks else None


def followup_summary(batch, specs, conditions, tolerances):
    checked = []
    for spec in specs:
        after, before = conditions[spec['key']], conditions[spec['base_key']]
        expected = spec['prediction'].get(spec['node_id'], {})
        behavior_match = predicate(expected, after)
        match, signed_match = behavior_match, None
        effect = None if after['fact'] is None or before['fact'] is None else after['fact'] - before['fact']
        if 'effect_sign' in expected:
            tolerance = tolerances.get(spec['node_id'], {}).get('fact', 0)
            sign = None if effect is None else 0 if abs(effect) <= tolerance else 1 if effect > 0 else -1
            signed_match = None if sign is None else sign == expected['effect_sign']
            has_answer = 'answer' in expected or 'correct' in expected
            match = (signed_match if not has_answer else None if match is None or signed_match is None
                     else match and signed_match)
        predicted, remainder, reason = None, None, None
        if spec['kind'] in ('key', 'coarse_weak', 'area_control', 'local_joint'):
            try:
                predicted = additive_effect(spec, spec['node_id'], conditions)
                remainder = None if effect is None else effect - predicted
            except ValueError as error:
                reason = str(error)
        checked.append(dict(spec, effect=effect, additive_prediction=predicted, joint_residual=remainder,
                            additive_reason=reason, behavior_match=behavior_match, score_match=signed_match,
                            prediction_match=None if spec.get('known') else match,
                            output=after['output'], correct=after['correct'], status=after['status'],
                            evidence_scope='within_case_exploration'))
    comparisons = []
    for comparison in batch.get('comparisons', []):
        witness, costs = {}, {}
        for key in comparison['visual_order']:
            rows = [s for s in specs if s['id'] == key]
            outcomes = [predicate(comparison['stop_when'][key][s['node_id']], conditions[s['key']])
                        for s in rows if s['node_id'] in comparison['stop_when'][key]]
            witness[key] = None if not outcomes or None in outcomes else all(outcomes)
            costs[key] = sum(sum(c['seconds'] for c in conditions[s['key']]['calls']
                                 if c['operation'] == 'generate') for s in rows)
        orders = {name: comparison[name + '_order'] for name in ('visual', 'numeric', 'direct')}
        comparisons.append(dict(comparison, result=compare_orders(orders, witness, costs),
                                scope='common_frozen_menu_descriptive_only',
                                cost_scope='actual_answer_generation; diagnostic_rescoring_in_full_ledger',
                                from_scratch_direct_search='not_measured'))
    return checked, comparisons


def run(args):
    started = time.monotonic()
    manifest = manifest_at(args.manifest)
    nodes = nodes_at(manifest)
    frozen = read_json(args.batch) if args.command == 'check' else None
    paths = [Path(item['path']) for item in frozen['prior']] if frozen else []
    reports, receipts = prior_at(paths, digest(args.manifest))
    if any(report.get('cpu_test', False) != args.cpu_test for report in reports):
        raise ValueError('CPU software fixtures and scientific measurements cannot share a cache')
    existing, prior_candidates, previous, old_views = merge_prior(reports)
    if frozen:
        if (frozen['status'] != 'frozen_before_new_checks' or frozen['manifest_sha256'] != digest(args.manifest)
                or frozen['prior'] != receipts or frozen['code_identity'] != code_identity()):
            raise ValueError('frozen check contract or prior evidence changed')
        validate_batch(frozen['batch'], nodes, previous, legacy_keys(manifest) | existing.keys())
    chosen = manifest['cases'][args.lane_index::args.lanes]
    if args.lane_index >= args.lanes or not chosen:
        raise ValueError('empty or invalid lane')
    if args.smoke:
        if args.lanes != 1:
            raise ValueError('the two-case technical smoke uses one lane')
        # One ordinary + one counterexample, from the declared development cases.
        chosen = [c for c in chosen if c['cluster_id'] in ('editclevr-100161', 'editclevr-100669')]
        if not chosen:
            raise ValueError('smoke lane has no declared case')
    if frozen and (args.lanes != 1 or args.smoke):
        raise ValueError('small frozen followup batches run as one batch, not split or smoke')
    with output_run(args.output) as out:
        diagnosis, identity = model_at(args)
        expected = manifest['historical_identity']
        for name in ('asset_identity', 'generation', 'effective_generation', 'model_spec'):
            if identity[name] != expected[name]:
                raise ValueError('B0 model/generation differs from old qualified cases: ' + name)
        if diagnosis.vlm.processor.image_processor.to_dict() != manifest['preprocess']:
            raise ValueError('model preprocessing differs from the qualified images')
        batch = frozen['batch'] if frozen else None
        wanted = ({n['id'] for c in chosen for n in c['nodes'] if n['label'] in ('Q1', 'Q2')}
                  if not batch else {nid for c in batch['checks'] for nid in c['node_ids']})
        candidates = {nid: candidates_at(diagnosis, nodes[nid], nodes) for nid in sorted(wanted)}
        for nid in wanted & prior_candidates.keys():
            if candidates[nid] != prior_candidates[nid]:
                raise ValueError('candidate identity changed since map observation')
        write_json(out / 'candidates.json', {'manifest_sha256': digest(args.manifest), 'nodes': candidates,
                                           'created_utc': now(), 'frozen_before_measurements': True})
        write_json(out / 'state.json', {'status': 'running', 'phase': args.command, 'created_utc': now()})
        specs = list(check_specs(batch)) if batch else []
        with (out / 'conditions.jsonl').open('x', buffering=1) as log:
            engine = Measurements(diagnosis, nodes, candidates, existing, log, legacy=legacy_outputs(manifest))
            current = deepcopy(old_views)
            completed_nodes = 0
            for nid in sorted(wanted):
                node = nodes[nid]
                control = old_views[nid]['controls'] if nid in old_views else engine.controls(nid)
                state = engine.state(nid)
                if state['grid']['rows'] != GRID or state['grid']['cols'] != GRID:
                    raise ValueError('V3 requires the actual 24x24 visual token grid')
                view = deepcopy(old_views[nid]) if nid in old_views else {
                    'id': nid, 'label': node['label'], 'question': node['clean_row']['question'],
                    'answer': node['clean_row']['answer'], 'candidates': candidates[nid],
                    'images': {kind: image_uri(diagnosis.processed_image(node[key])) for kind, key in
                               (('clean', 'clean_row'), ('observed', 'observed_row'))},
                    'grid': state['grid'], 'annotations': node.get('annotations', {}),
                    'maps': main_maps(nid) if not batch else []}
                view.update(controls=control, tolerance=control['tolerance'])
                view['images']['coordinate_system'] = 'model_processed_pixels'
                if not batch:
                    for mapping in view['maps']:
                        engine.get(nid, mapping['background'])
                        for cell in mapping['cells']:
                            if cell['status'] != 'included':
                                record = engine.get(nid, sorted(set(mapping['background']) | set(cell['indices'])))
                                cell['status'] = record['status']
                else:
                    for spec in (s for s in specs if s['node_id'] == nid):
                        engine.get(nid, spec['background'], spec['donor_id'])
                        engine.get(nid, spec['total_indices'], spec['donor_id'])
                        if spec['kind'] == 'refine':
                            mid = nid + ':refine:' + spec['id']
                            mapping = next((m for m in view['maps'] if m['id'] == mid), None)
                            if mapping is None:
                                mapping = {'id': mid, 'title': '局部12×12：' + spec['id'], 'resolution': 12,
                                           'background': spec['background'], 'base_key': spec['base_key'],
                                           'parent_index': spec['parent_index'], 'cells': []}
                                view['maps'].append(mapping)
                            mapping['cells'].append({'index': spec['child_index'], 'indices': spec['indices'],
                                                     'key': spec['key'], 'status': engine.conditions[spec['key']]['status']})
                current[nid] = view
                completed_nodes += 1
                write_json(out / 'state.json', {'status': 'running', 'phase': args.command,
                                               'completed_nodes': completed_nodes, 'nodes': len(wanted), 'updated_utc': now()})
            for nid, view in current.items():
                view['conditions'] = {k: v for k, v in engine.conditions.items() if v['node_id'] == nid}
                for mapping in view['maps']:
                    if mapping['resolution'] != 2:
                        continue
                    for cell in mapping['cells']:
                        if cell['status'] != 'measured':
                            continue
                        check = {'node_ids': [nid], 'background': mapping['background'], 'indices': cell['indices']}
                        try:
                            predicted = additive_effect(check, nid, engine.conditions)
                            before = engine.conditions[mapping['base_key']]['fact']
                            after = engine.conditions[cell['key']]['fact']
                            cell['additive_prediction'] = predicted
                            cell['joint_residual'] = None if before is None or after is None else after - before - predicted
                        except ValueError as error:
                            cell.update(additive_prediction=None, joint_residual=None, additive_reason=str(error))
            checked, comparisons = followup_summary(batch, specs, engine.conditions,
                                                    {k: v['tolerance'] for k, v in current.items()}) if batch else ([], [])
            cases = []
            for case in chosen:
                selected = [current[n['id']] for n in case['nodes'] if n['id'] in current]
                if selected:
                    cases.append({key: case[key] for key in ('cluster_id', 'initial_note', 'annotations')}
                                 | {'nodes': selected})
            elapsed = time.monotonic() - started
            report = {'schema': 'visual-probe-v3', 'status': 'smoke' if args.smoke else 'exploration',
                      'cpu_test': args.cpu_test,
                      'scope': 'Known cases; scores are not correctness, diagnosis is not weight repair.',
                      'created_utc': now(), 'cases': cases, 'checks': checked,
                      'diagnoses': batch['diagnoses'] if batch else [], 'comparisons': comparisons, 'batch': batch,
                      'batch_history': previous + ([batch] if batch else []),
                      'cost': {'elapsed_seconds': elapsed, 'gpu_hours': elapsed / 3600 if not args.cpu_test else 0,
                               'calls': engine.calls.rows, 'new_conditions': len(engine.conditions) - len(existing),
                               'read_map_seconds': batch.get('read_map_seconds') if batch else None,
                               'read_map_cost_status': 'not_recorded' if not batch or batch.get('read_map_seconds') is None else 'recorded',
                               'from_scratch_direct_search': 'not_measured', 'automatic_time_stop': False}}
            write_json(out / 'report.json', report)
            from .visual_probe_report import render
            render(report, out / 'index.html')
            status = 'completed_with_unresolved' if any(v['status'] != 'measured' or v['correct'] is None
                         for n in current.values() for v in n['conditions'].values()) else 'completed'
            write_json(out / 'run.json', {'status': status, 'identity': identity, 'code_identity': code_identity(),
                       'manifest_sha256': digest(args.manifest), 'report_sha256': digest(out / 'report.json'),
                       'batch_sha256': digest(args.batch) if frozen else None, 'prior': receipts,
                       'smoke': args.smoke, 'lane_index': args.lane_index, 'lanes': args.lanes,
                       'elapsed_seconds': time.monotonic() - started, 'parameter_updates': 0})
            write_json(out / 'state.json', {'status': status, 'updated_utc': now()})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--pools', type=Path, nargs='+', required=True)
    p.add_argument('--records', type=Path, nargs='+', required=True)
    p.add_argument('--count', type=int, choices=(6, 8), default=8)
    p.add_argument('--notes', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--prior', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    for command in ('map', 'check'):
        p = sub.add_parser(command)
        p.add_argument('--manifest', type=Path, required=True)
        p.add_argument('--config', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--device', default='cuda:0')
        p.add_argument('--cpu-test', action='store_true')
        p.add_argument('--smoke', action='store_true')
        p.add_argument('--lanes', type=int, choices=(1, 2), default=1)
        p.add_argument('--lane-index', type=int, choices=(0, 1), default=0)
        if command == 'check':
            p.add_argument('--batch', type=Path, required=True)
    p = sub.add_parser('report')
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'freeze':
        freeze(args)
    elif args.command == 'report':
        from .visual_probe_report import render
        manifest_sha = read_json(args.runs[0] / 'run.json')['manifest_sha256']
        reports, _ = prior_at(args.runs, manifest_sha)
        conditions, _, _, views = merge_prior(reports)
        combined = deepcopy(reports[-1])
        cases = {}
        for report in reports:
            for case in report['cases']:
                cases[case['cluster_id']] = {k: v for k, v in case.items() if k != 'nodes'}
        for cluster, case in cases.items():
            case['nodes'] = [dict(node, conditions={k: v for k, v in conditions.items() if v['node_id'] == nid})
                             for nid, node in views.items() if nodes_cluster(nid, reports) == cluster]
        combined['cases'] = list(cases.values())
        combined['checks'] = [c for r in reports for c in r['checks']]
        combined['diagnoses'] = [d for r in reports for d in r['diagnoses']]
        combined['comparisons'] = [c for r in reports for c in r['comparisons']]
        combined['cost'] = {'runs': [r['cost'] for r in reports],
                            'gpu_hours': sum(r['cost']['gpu_hours'] for r in reports)}
        with output_run(args.output) as out:
            write_json(out / 'report.json', combined)
            render(combined, out / 'index.html')
    else:
        run(args)
    return 0


def nodes_cluster(nid, reports):
    return next(c['cluster_id'] for r in reports for c in r['cases'] if any(n['id'] == nid for n in c['nodes']))


if __name__ == '__main__':
    raise SystemExit(main())
