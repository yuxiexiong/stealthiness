"""Finite M051 pipeline. At most two idle GPUs, never shares a card, never kills other work."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from yproto import require, read, save, sha, GAP_STEPS, FULL_READ_STEPS
from yruntime import ROOT

LEDGER = read(ROOT / 'LEDGER.json') if (ROOT / 'LEDGER.json').exists() else []
START = time.time()
PLAN = {}          # stage name -> number of chains, filled as stages are registered


def resources():
    def query(args):
        return subprocess.check_output(['nvidia-smi'] + args, text=True).strip().splitlines()
    apps = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    occupied = {x.split(',')[0].strip() for x in apps if ',' in x}
    free = []
    for row in query(['--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits']):
        uid, mem, util = [x.strip() for x in row.split(',')]
        if uid not in occupied and int(mem) < 1024 and int(util) == 0:
            free.append(uid)
    return free


def status(stage, active=(), pending=0, note=None):
    done = sum(j['seconds'] for j in LEDGER)
    save(ROOT / 'STATUS.json', dict(stage=stage, wall_time=time.time(), started=START, ledger_gpu_hours=done / 3600,
         jobs_completed=len(LEDGER), active=list(active), pending=pending, note=note,
         mean_seconds_by_kind=kind_means()), new=False)


def kind_means():
    by = {}
    for j in LEDGER:
        by.setdefault(j['kind'], []).append(j['seconds'])
    return {k: sum(v) / len(v) for k, v in by.items()}


def run_chains(stage, chains):
    """chains: [(name, kind, [argv, argv, ...])]; each chain runs its commands in order on one GPU."""
    done = {tuple(j['args']) for j in LEDGER if j['returncode'] == 0}
    pending = []
    for name, kind, commands in chains:          # resume after an engineering repair: skip finished commands
        left = [c for c in commands if tuple(c) not in done]
        require(left == commands[len(commands) - len(left):], 'non-prefix completion in chain ' + name)
        if left:
            pending.append((name, kind, left, len(commands) - len(left)))
    active = {}; failed = False
    while pending or active:
        require(not (ROOT / 'HALT.json').exists(), 'HALT')
        for uid, j in list(active.items()):
            rc = j['process'].poll()
            if rc is None:
                continue
            j['log'].close()
            LEDGER.append(dict(name=j['name'], kind=j['kind'], args=j['args'], returncode=rc, gpu_uuid=uid,
                               seconds=time.time() - j['started'], started=j['started'], stage=stage))
            save(ROOT / 'LEDGER.json', LEDGER, new=False)
            if rc != 0:
                failed = True; del active[uid]; continue
            if j['rest']:
                active[uid] = spawn(uid, j['name'], j['kind'], j['rest'], j['index'] + 1)
            else:
                del active[uid]
        if failed:
            pending = []
            if not active:
                raise RuntimeError('worker failed in ' + stage + '; see LEDGER/logs')
        free = [uid for uid in resources() if uid not in active]
        for uid in free[:max(0, 2 - len(active))]:
            if not pending:
                break
            require(shutil.disk_usage(ROOT).free > 8 * 2 ** 30, 'storage reserve < 8 GiB')
            name, kind, commands, index = pending.pop(0)
            require(uid in resources(), 'GPU taken before launch')
            active[uid] = spawn(uid, name, kind, commands, index)
        status(stage, [dict(name=j['name'], args=j['args'], gpu_uuid=uid, started=j['started'], pid=j['process'].pid)
                       for uid, j in active.items()], len(pending))
        if pending or active:
            time.sleep(10)


def spawn(uid, name, kind, commands, index):
    args = commands[0]
    logpath = ROOT / 'logs' / f'{name}.{index}.log'; logpath.parent.mkdir(exist_ok=True)
    n = 0
    while logpath.exists():                      # a repaired retry never overwrites the failed log
        n += 1; logpath = ROOT / 'logs' / f'{name}.{index}.retry{n}.log'
    log = logpath.open('x'); env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uid, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    p = subprocess.Popen([sys.executable, str(ROOT / 'bundle/ytask.py')] + args, stdout=log, stderr=subprocess.STDOUT, env=env)
    sub = 'train' if args[0] == 'train' else ('read_full' if args[0] == 'read' and args[2] == 'full' else args[0])
    return dict(process=p, log=log, name=name, kind=sub, args=args, rest=commands[1:], index=index, started=time.time())


def csv(xs):
    return ','.join(map(str, xs))


def main():
    from yruntime import environment, arms
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    lock = (ROOT / 'supervisor.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'STARTED.json').exists():
        require(os.environ.get('FRESH_REPAIR') == '1' and (ROOT / 'FAILURE.json').exists(), 'already started; explicit repair required')
    else:
        save(ROOT / 'STARTED.json', dict(time=START, no_automatic_expansion=True))
    try:
        tags = sorted(arms(), key=lambda t: (int(t[1:]), t[0]))        # F1, P1, F2, P2, ... so both families progress together
        run_chains('ARMS', [(t, 'arm', [['train', t], ['read', t, 'gap', csv(GAP_STEPS)], ['read', t, 'full', csv(FULL_READ_STEPS)]]) for t in tags])
        from yreport import finish
        finish(); status('COMPLETE')
    except BaseException as exc:
        save(ROOT / 'FAILURE.json', dict(error=repr(exc), time=time.time()), new=False)
        status('FAILED', note=repr(exc)); raise


if __name__ == '__main__':
    main()
