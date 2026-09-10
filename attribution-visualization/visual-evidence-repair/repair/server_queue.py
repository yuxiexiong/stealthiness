"""One server-local waiter. No torch, GPU allocation, scheduler or local automation."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time


def process_alive(identity):
    try:
        stat = (Path('/proc') / str(identity['pid']) / 'stat').read_text()
    except FileNotFoundError:
        return False
    fields = stat[stat.rfind(')') + 2:].split()
    return fields[0] not in ('Z', 'X') and fields[19] == str(identity['start_ticks'])


def gpu_idle(gpus):
    def query(fields, kind):
        return subprocess.check_output(['nvidia-smi', '--query-' + kind + '=' + fields,
                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
    processes = query('gpu_uuid,pid', 'compute-apps')
    if any(line.split(',')[0].strip() in gpus for line in processes.splitlines()):
        return False
    rows = [line.split(',') for line in query('uuid,utilization.gpu,memory.used', 'gpu').splitlines()]
    selected = {row[0].strip(): row for row in rows if row[0].strip() in gpus}
    return set(selected) == set(gpus) and all(float(r[1]) <= 5 and float(r[2]) <= 256 for r in selected.values())


def launch_at(path):
    if not path.exists():
        return None, 'frozen launch manifest has not been supplied; experiment assets are not ready'
    data = json.loads(path.read_text())
    if data.get('ready') is not True:
        return None, 'launch manifest is explicitly not ready'
    if set(data) != {'ready', 'cwd', 'command', 'required_files'}:
        raise ValueError('launch manifest requires ready, cwd, command and required_files')
    if (not isinstance(data['command'], list) or not data['command']
            or any(not isinstance(x, str) or not x for x in data['command'])
            or not isinstance(data['required_files'], list) or not data['required_files']):
        raise ValueError('declare command argv and the actual frozen input files')
    cwd = Path(data['cwd'])
    if not cwd.is_absolute() or not cwd.is_dir():
        raise ValueError('launch cwd must be an existing absolute directory')
    for name in data['required_files']:
        path = Path(name)
        if not path.is_absolute() or not path.is_file():
            return None, 'required frozen input missing: ' + str(path)
    return data, None


def wait_and_run(root, config, interval=300):
    idle_checks = 0
    last_status = None

    def record(status, **details):
        nonlocal last_status
        row = {'status': status, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
        temp = root / 'state.tmp'
        temp.write_text(json.dumps(row, indent=2) + '\n')
        temp.replace(root / 'state.json')
        if status != last_status:
            print(json.dumps(row), flush=True)
        last_status = status

    while True:
        try:
            active = [p for p in config['after'] if process_alive(p)]
            if active:
                idle_checks = 0
                record('waiting_teammate_controllers', controllers=active)
            else:
                launch, reason = launch_at(Path(config['launch_manifest']))
                if launch is None:
                    idle_checks = 0
                    record('waiting_inputs', reason=reason)
                elif not gpu_idle(config['gpus']):
                    idle_checks = 0
                    record('waiting_other_gpu_jobs')
                else:
                    idle_checks += 1
                    record('confirming_idle', consecutive_checks=idle_checks)
                    if idle_checks >= 2:
                        # Check again at the handoff, not just between two training rounds.
                        if any(process_alive(p) for p in config['after']) or not gpu_idle(config['gpus']):
                            idle_checks = 0
                            continue
                        (root / 'launched.json').write_text(json.dumps(launch, indent=2) + '\n')
                        record('launching', command=launch['command'], cwd=launch['cwd'])
                        with (root / 'experiment.log').open('x') as log:
                            child = subprocess.Popen(launch['command'], cwd=launch['cwd'], stdout=log,
                                                     stderr=subprocess.STDOUT)
                            record('running', pid=child.pid)
                            code = child.wait()
                        record('completed' if code == 0 else 'failed', exit_code=code)
                        return code
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            # No success inference on a failed query and no automatic experiment retry.
            idle_checks = 0
            if (root / 'launched.json').exists():
                record('launch_failed', error=str(error))
                return 1
            record('waiting_check_error', error=str(error))
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    if args.interval < 60:
        parser.error('use a low-frequency interval of at least 60 seconds')
    root = args.directory.resolve(strict=True)
    config = json.loads((root / 'queue.json').read_text())
    if (not config['after'] or len(config['gpus']) != 2 or len(set(config['gpus'])) != 2
            or any(type(p['pid']) is not int or p['pid'] <= 1 or not str(p['start_ticks']).isdigit() for p in config['after'])):
        parser.error('bind actual controller PIDs/start times and two distinct GPU UUIDs')
    with (root / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'launched.json').exists():
            parser.error('this queue already launched; inspect its result rather than rerunning it')
        return wait_and_run(root, config, args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
