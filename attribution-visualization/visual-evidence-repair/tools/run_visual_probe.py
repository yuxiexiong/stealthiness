"""Wait for assigned GPUs, run the fixed V3 smoke and two map lanes, then exit.

Uses the existing idle check, process monitor, GPU accounting and lane launcher.
No training, automatic retries, scientific-success gate or time-budget cutoff.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1]))
sys.path.insert(0, str(PROJECT))
from experiments.measure import run as measure_run
from repair.__main__ import digest, output_run, read_json, write_json
from repair.report import score_text, _chips, _viewer
from repair.server_queue import gpu_idle, process_alive

SMOKE_CASES = {'editclevr-100161', 'editclevr-100669'}


def gpu_uuids(gpus):
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    identities = {}
    for row in rows.splitlines():
        index, uuid = [part.strip() for part in row.split(',')]
        identities[index] = identities[uuid] = uuid
    if any(gpu not in identities for gpu in gpus) or len({identities[g] for g in gpus}) != 2:
        raise ValueError('assigned GPUs must resolve to two distinct physical GPU UUIDs')
    return [identities[gpu] for gpu in gpus]


def wait_for_idle(uuids, phase, save, after=()):
    consecutive = 0
    while True:
        active = [identity for identity in after if process_alive(identity)]
        if active:
            consecutive = 0
            save(phase='waiting_' + phase, idle_checks=0, controllers=active)
            time.sleep(60)
            continue
        consecutive = consecutive + 1 if gpu_idle(uuids) else 0
        save(phase='waiting_' + phase, idle_checks=consecutive, controllers=[])
        if consecutive >= 2:
            # Recheck immediately at handoff; never interrupt newly arrived jobs.
            if not any(process_alive(identity) for identity in after) and gpu_idle(uuids):
                return
            consecutive = 0
        time.sleep(60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'output'):
        parser.add_argument('--' + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument('--after', type=lambda value: Path(value).resolve(),
                        help='JSON list of actual teammate controller {pid, start_ticks} identities')
    args = parser.parse_args(argv)
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(not g or g != g.strip() for g in gpus):
        parser.error('explicitly assign two cards through CUDA_VISIBLE_DEVICES')
    with output_run(args.output) as output:
        state = {'status': 'running', 'phase': 'preflight', 'pid': os.getpid(), 'assigned_gpus': gpus,
                 'allocated_gpu_hours': 0.0, 'automatic_time_stop': False, 'parameter_updates': 0}

        def save(**changes):
            changed = any(state.get(key) != value for key, value in changes.items())
            state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / 'state.json', state)
            if changed:
                print(state, flush=True)

        try:
            save()
            after = read_json(args.after) if args.after else []
            if (not isinstance(after, list) or any(
                    not isinstance(identity, dict) or set(identity) != {'pid', 'start_ticks'}
                    or type(identity['pid']) is not int or identity['pid'] <= 1
                    or not str(identity['start_ticks']).isdigit() for identity in after)):
                raise ValueError('after must list actual controller PID/start-time identities')
            if score_text('red', {'task': 'fact', 'answer': 'red'})['exact_match'] != 1.0:
                raise RuntimeError('original factual scorer failed CPU preflight')
            _chips([0.0], 1.0)
            _viewer().STYLE
            manifest = read_json(args.manifest)
            case_ids = [case['cluster_id'] for case in manifest['cases']]
            if manifest['count'] not in (6, 8) or len(case_ids) != manifest['count'] or not SMOKE_CASES <= set(case_ids):
                raise ValueError('manifest must contain the frozen 6/8 cases and both fixed smoke cases')
            uuids = gpu_uuids(gpus)
            base = [sys.executable, '-m', 'repair.visual_probe']
            common = ['--manifest', str(args.manifest), '--config', str(args.config), '--device', 'cuda:0']
            smoke = base + ['map', *common, '--smoke', '--output', str(output / 'smoke')]
            lanes = [output / f'lane-{lane}' for lane in range(2)]
            queues = [[base + ['map', *common, '--lanes', '2', '--lane-index', str(lane),
                              '--output', str(path)]] for lane, path in enumerate(lanes)]
            write_json(output / 'queues.json', queues)
            full = [sys.executable, '-m', 'repair.parallel', '--queues', str(output / 'queues.json'),
                    '--output', str(output / 'parallel')]
            report = base + ['report', '--runs', *map(str, lanes), '--output', str(output / 'report')]
            write_json(output / 'launch.json', {'smoke': smoke, 'full': full, 'report': report,
                       'config_sha256': digest(args.config), 'manifest_sha256': digest(args.manifest),
                       'gpu_uuids': uuids, 'after': after, 'after_sha256': digest(args.after) if args.after else None,
                       'automatic_time_stop': False, 'smoke_remeasured_in_full': True})
            for phase, command, allocated in (('smoke', smoke, gpus[:1]), ('full', full, gpus)):
                wait_for_idle(uuids, phase, save, after)
                save(phase=phase, phase_gpus=allocated)
                measured = measure_run(command, output / (phase + '-measurement'), PROJECT,
                                       gpus=allocated, interval=60, cost_role='evaluation')
                save(allocated_gpu_hours=state['allocated_gpu_hours'] + measured['allocated_gpu_hours'])
                if measured['status'] != 'completed':
                    raise RuntimeError(phase + ' process failed; inspect its measurement/stdout.log')
                paths = [output / 'smoke'] if phase == 'smoke' else lanes
                for lane, path in enumerate(paths):
                    run, result = read_json(path / 'run.json'), read_json(path / 'report.json')
                    expected = SMOKE_CASES if phase == 'smoke' else set(case_ids[lane::2])
                    if (run['status'] not in ('completed', 'completed_with_unresolved')
                            or run['parameter_updates'] != 0 or run['smoke'] != (phase == 'smoke')
                            or result.get('cpu_test', False)
                            or {c['cluster_id'] for c in result['cases']} != expected):
                        raise RuntimeError(phase + ' did not complete the declared cases without training')
                    if phase == 'smoke':
                        for case in result['cases']:
                            if len(case['nodes']) != 2:
                                raise RuntimeError('smoke must exercise Q1 and Q2 in each fixed case')
                            for node in case['nodes']:
                                if (not all(node['controls'][key] for key in (
                                        'empty_replay_exact', 'full_visual_matches_normal', 'self_copy_exact'))
                                        or not node['conditions']
                                        or any(c['status'] != 'measured' for c in node['conditions'].values())):
                                    raise RuntimeError('smoke interface/replay/finite-score check failed')
                save(phase=phase + '_passed')
            save(phase='report')
            subprocess.run(report, cwd=PROJECT, check=True)
            save(status='completed', phase='awaiting_map_review', report=str(output / 'report' / 'index.html'))
        except Exception as error:
            save(status='failed', error=f'{type(error).__name__}: {error}')
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
