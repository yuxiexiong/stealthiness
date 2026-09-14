"""GPU1-only launch, idle handoff, and existing per-minute process/GPU monitoring."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parents[1]))
from tools.run_visual_probe import gpu_uuids, wait_for_idle
from experiments.measure import run as measure_run
from repair.__main__ import output_run, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'development', 'cases', 'calibration-rows', 'output'):
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args(argv)
    uuid = gpu_uuids(['1'])[0]
    assigned = os.environ.get('CUDA_VISIBLE_DEVICES')
    if assigned not in ('1', uuid):
        parser.error('explicit GPU1 assignment required; never falls back to GPU0')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    with output_run(args.output) as out:
        state = {'status': 'running', 'gpu_uuid': uuid, 'gpu_index': 1,
                 'automatic_time_stop': False, 'monitor_interval_seconds': 60, 'pid': os.getpid()}

        def save(**changes):
            state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(out / 'state.json', state)
            print(state, flush=True)

        try:
            wait_for_idle([uuid], 'smoke', save)
            command = [sys.executable, '-m', 'repair.diagnosis_comparison']
            for name in ('config', 'development', 'cases', 'calibration_rows'):
                command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
            command += ['--output', str(out / 'experiment')]
            write_json(out / 'launch.json', {'command': command, 'gpu_uuid': uuid,
                'smoke_then_formal_in_one_process': True, 'automatic_time_stop': False})
            save(phase='smoke_then_formal')
            measured = measure_run(command, out / 'measurement', PROJECT, gpus=[uuid],
                                   interval=60, cost_role='evaluation')
            if measured['status'] != 'completed':
                raise RuntimeError('worker failed; preserve attempt and inspect measurement/stdout.log')
            result = read_json(out / 'experiment' / 'run.json')
            save(status=result['status'], phase='finished',
                 allocated_gpu_hours=measured['allocated_gpu_hours'],
                 reader_status=result['reader_status'])
        except Exception as error:
            save(status='failed', error=f'{type(error).__name__}: {error}')
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
