"""CPU-only: pin assets, choose replacement batches by the frozen rule, run tests. No CUDA, no model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from yproto import require, read, sha, save, select_arms, START, END, GAP_STEPS
from yruntime import ROOT, OLD, environment

M049 = Path('/workspace/asr-kick-20261117-r01')
M050 = Path('/workspace/asr-fallcross-20261117-r01')
PINNED = ['bundle/yproto.py', 'bundle/yruntime.py', 'bundle/ytask.py', 'bundle/ysup.py', 'bundle/yreport.py',
          'bundle/yprepare.py', 'bundle/ytest.py', 'CONTRACT.zh-CN.md', 'ARMS.json']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    from state_restore import verify
    prior = read(M050 / 'ASSETS.json')
    source = Path(prior['restore']); cursor, layout, _ = verify(source)
    require(cursor['global_step'] == START, 'restore boundary')
    pins = {}
    for path, pin in read(M049 / 'ASSETS.json')['source_pins'].items():
        require(sha(path) == pin, 'historical input changed: ' + path); pins[path] = pin
    require(read(M050 / 'RESULT.json')['status'] == 'COMPLETE', 'M050 incomplete')
    m050 = {}
    for arm in ('RR', 'RD'):
        for step in GAP_STEPS:
            f = M050 / 'reads/gap' / f'{arm}@{step}.json'; pins[str(f)] = sha(f); m050[f'{arm}@{step}'] = str(f)
    f = M050 / 'reads/full/RD@385.json'; pins[str(f)] = sha(f); m050['RD_full@385'] = str(f)
    f = Path(prior['old_reads']['RR']['385']); pins[str(f)] = sha(f); m050['RR_full@385'] = str(f)
    for step in (376, END):
        f = M050 / 'runs/RR' / f'checkpoint-{step}' / 'adapter_model.safetensors'; pins[str(f)] = sha(f)
    pins[str(M050 / 'RESULT.json')] = sha(M050 / 'RESULT.json')
    spec = read(OLD / 'specs/B-replace.json'); poison_rows = set(spec['branch_manifest']['poison_rows_after'])
    rows = cursor['expected_consumption']; horizon = len(rows) // 16
    poison = {k for k in range(1, horizon + 1) if poison_rows.intersection(rows[(k - 1) * 16:k * 16])}
    arms = select_arms(poison, horizon)
    save(ROOT / 'ARMS.json', dict(arms={t: {str(k): v for k, v in m.items()} for t, m in arms.items()},
         rule='yproto.select_arms: first 10 pure-clean updates >= 441 as fresh pairs; nearest pure-clean neighbours of 358 and 342 as repeat pairs',
         chosen_without_any_outcome=True), new=False)
    free = shutil.disk_usage(ROOT).free
    require(free >= 20 * 2 ** 30, 'need 20 GiB free')
    save(ROOT / 'ASSETS.json', dict(restore=str(source), m050_reference_run=str(M050 / 'runs/RR'), m050_reads=m050, source_pins=pins,
         micro_kick_floor=prior['micro_kick_floor'], free_bytes=free, layout=layout), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/ytest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', dict(arms=arms, free_GiB=round(free / 2 ** 30, 1)))


if __name__ == '__main__':
    main()
