"""Run this probe's frozen follow-ups using the existing single-card waiter."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.run_visual_probe import wait_for_available, measure_run
from repair.visual_probe import digest, read_json, write_json, output_run, prior_at, merge_prior
from repair.visual_probe_protocol import numeric_order


def paired_replay(order, outcomes, pairs, costs):
    """Three separate evidence goals; an unknown is never a counterexample."""
    result, seen, seconds = {}, set(), 0.0
    for index, check in enumerate(order, 1):
        seen.add(check)
        seconds += costs[check]
        at = {'checks': index, 'generation_seconds': seconds, 'last_check': check}
        if outcomes[check] is False:
            result.setdefault('first_falsification', dict(at))
        if any(set(pair) <= seen and all(outcomes[x] is True for x in pair) for pair in pairs):
            result.setdefault('first_pair_support', dict(at))
        if len(seen) == len(order) and all(outcomes[x] is True for x in order):
            result['full_pattern_support'] = dict(at)
    for goal in ('first_falsification', 'first_pair_support', 'full_pattern_support'):
        result.setdefault(goal, None)
    return result


def paired_summary(report):
    result = []
    conditions = {k: v for c in report['cases'] for n in c['nodes'] for k, v in n['conditions'].items()}
    for comparison in report['batch'].get('paired_comparisons', []):
        outcomes, costs = {}, {}
        for key in comparison['visual_order']:
            rows = [c for c in report['checks'] if c['id'] == key]
            matches = [c['prediction_match'] for c in rows]
            outcomes[key] = False if False in matches else True if matches and all(x is True for x in matches) else None
            costs[key] = sum(call['seconds'] for c in rows for call in conditions[c['key']]['calls']
                             if call['operation'] == 'generate')
        orders = {name: comparison[name + '_order'] for name in ('visual', 'numeric', 'direct')}
        result.append(dict(comparison, outcomes=outcomes,
                           results={name: paired_replay(order, outcomes, comparison['pairs'], costs)
                                    for name, order in orders.items()},
                           visual_equals_direct=orders['visual'] == orders['direct'],
                           scope='separate_evidence_goals_on_common_frozen_menu; not one online policy',
                           from_scratch_advantage='not_evaluated'))
    return result


def run(plan_path):
    plan = read_json(plan_path)
    with output_run(Path(plan['output'])) as out:
        state = {'status': 'running', 'phase': 'preflight', 'pid': os.getpid(),
                 'automatic_time_stop': False, 'parameter_updates': 0, 'allocated_gpu_hours': 0.0}

        def save(**changes):
            state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(out / 'state.json', state)
            print(json.dumps(state, ensure_ascii=False), flush=True)

        try:
            save()
            write_json(out / 'launch-plan.json', plan)
            priors = [Path(plan['prior'])]
            base = [sys.executable, '-m', 'repair.visual_probe']
            for item in plan['requests']:
                if digest(Path(item['path'])) != item['sha256']:
                    raise ValueError('registered request changed')
            for index, item in enumerate(plan['requests'], 1):
                name = f'batch-{index}'
                request = read_json(Path(item['path']))
                reports, _ = prior_at(priors, digest(Path(plan['manifest'])))
                conditions, _, _, _ = merge_prior(reports)
                checks = {c['id']: c for c in request['checks']}
                for comparison in request.get('paired_comparisons', []):
                    comparison['numeric_order'] = numeric_order(
                        [checks[k] for k in comparison['visual_order']], comparison['node_id'],
                        conditions, comparison['direction'])
                effective = out / (name + '-request.json')
                write_json(effective, request)
                frozen = out / (name + '-frozen')
                subprocess.run(base + ['freeze', '--manifest', plan['manifest'], '--request', str(effective),
                               '--prior', *map(str, priors), '--output', str(frozen)], cwd=PROJECT, check=True)
                save(phase=name + '_frozen')
                selected = wait_for_available(plan['gpu_candidates'], name, save)
                gpu = selected['gpu']
                save(phase=name + '_running', assigned_gpu=gpu)
                target = out / name
                command = base + ['check', '--manifest', plan['manifest'], '--config', plan['config'],
                                  '--batch', str(frozen / 'batch.json'), '--output', str(target)]
                measured = measure_run(command, out / (name + '-measurement'), PROJECT,
                                       gpus=[gpu], interval=60, cost_role='evaluation')
                save(allocated_gpu_hours=state['allocated_gpu_hours'] + measured['allocated_gpu_hours'])
                if measured['status'] != 'completed':
                    raise RuntimeError(name + ' failed; preserve logs and repair before continuing')
                result = read_json(target / 'report.json')
                receipt = read_json(target / 'run.json')
                if receipt['parameter_updates'] != 0 or receipt['status'] not in ('completed', 'completed_with_unresolved'):
                    raise RuntimeError('follow-up technical completion failed')
                # Negative scientific results never block later registered checks.
                write_json(target / 'paired-comparison.json', paired_summary(result))
                priors.append(target)
                save(phase=name + '_completed', completed_batches=index)
            subprocess.run(base + ['report', '--runs', *map(str, priors), '--output', str(out / 'report')],
                           cwd=PROJECT, check=True)
            write_json(out / 'paired-comparison.json', [c for p in priors[1:]
                       for c in read_json(p / 'paired-comparison.json')])
            save(status='completed', phase='completed', report=str(out / 'report/index.html'))
        except Exception as error:
            save(status='failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        r = paired_replay(['a', 'b', 'c', 'd'], {'a': True, 'b': True, 'c': None, 'd': False},
                          [['a', 'b'], ['c', 'd']], dict.fromkeys('abcd', 1.0))
        assert r['first_pair_support']['checks'] == 2
        assert r['first_falsification']['checks'] == 4 and r['full_pattern_support'] is None
        assert paired_replay(['a'], {'a': None}, [], {'a': 1})['first_falsification'] is None
        assert paired_replay(['a', 'b'], {'a': True, 'b': True}, [['a', 'b']],
                             {'a': 1, 'b': 2})['full_pattern_support']['generation_seconds'] == 3
        print('paired evidence checks passed')
    elif args.plan:
        run(args.plan)
    else:
        parser.error('--plan or --self-test required')
