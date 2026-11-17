"""CPU-only asset pinning and readiness. No CUDA, no model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from xproto import require, read, sha, save, arms, D_PAIRS, OLD_GRID, START, END
from xruntime import ROOT, OLD, M045, environment

M049 = Path('/workspace/asr-kick-20261117-r01')
PINNED = ['bundle/xproto.py', 'bundle/xruntime.py', 'bundle/xtask.py', 'bundle/xsup.py', 'bundle/xreport.py',
          'bundle/xprepare.py', 'bundle/xtest.py', 'CONTRACT.zh-CN.md']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    from state_restore import verify
    prior = read(M049 / 'ASSETS.json')
    source = Path(prior['restore']); cursor, layout, _ = verify(source)
    require(cursor['global_step'] == START, 'restore boundary')
    pins = {}
    for path, pin in prior['source_pins'].items():
        require(sha(path) == pin, 'historical input changed: ' + path); pins[path] = pin
    f1 = read(M045 / 'F1.json')
    require(f1['selections']['real-down']['pairs'] == D_PAIRS, 'M045 real-down swap list differs from the registered one')
    pins[str(M045 / 'F1.json')] = sha(M045 / 'F1.json')
    old_reads = {'RR': {}, 'DD': {}}
    for step in OLD_GRID:
        for tag, adapter, reading in (('RR', Path(prior['reference_run']) / f'checkpoint-{step}', Path(prior['curves'][str(step)]['path'])),
                                      ('DD', M045 / 'runs/real-down' / f'checkpoint-{step}', M045 / 'reads/S3' / f'real-down@{step}.json')):
            a = adapter / 'adapter_model.safetensors'
            require(a.is_file() and reading.is_file(), f'missing historical {tag}@{step}')
            pins[str(a)] = sha(a); pins[str(reading)] = sha(reading); old_reads[tag][str(step)] = str(reading)
    floor = read(M049 / 'RESULT.json')['noise_floor']; pins[str(M049 / 'RESULT.json')] = sha(M049 / 'RESULT.json')
    spec = read(OLD / 'specs/B-replace.json'); poison = set(spec['branch_manifest']['poison_rows_after'])
    rows = cursor['expected_consumption']
    touched = sorted({x for m in arms().values() for kv in m.items() for x in kv})
    for k in touched:
        require(not poison.intersection(rows[(k - 1) * 16:k * 16]), f'update {k} is not a pure-clean batch')
    poison_tail = [k for k in range(376, END + 1) if poison.intersection(rows[(k - 1) * 16:k * 16])]
    free = shutil.disk_usage(ROOT).free
    require(free >= 20 * 2 ** 30, 'need 20 GiB free')
    save(ROOT / 'ASSETS.json', dict(restore=str(source), reference_run=prior['reference_run'], source_pins=pins, old_reads=old_reads,
         micro_kick_floor=floor, poison_updates_in_tail=poison_tail, touched_clean_updates=touched, free_bytes=free, layout=layout), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/xtest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', dict(poison_updates_in_tail=poison_tail, touched=touched, free_GiB=round(free / 2 ** 30, 1), floor=floor))


if __name__ == '__main__':
    main()
