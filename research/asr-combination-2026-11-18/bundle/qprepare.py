"""CPU-only: register every chain from the schedules, pin assets (including the M055 natural-column reads), run tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from qproto import (require, read, sha, save, step1_chains, step2_chains, state_path, STEP1, STEP1_LAGS, ARMS6, BATCHES, X_BATCH, Y_BATCH,
                    SEEDS_ROOT, M055, P2)
from qruntime import ROOT, OLD, environment

PINNED = ['bundle/qproto.py', 'bundle/qruntime.py', 'bundle/qtask.py', 'bundle/qsup.py', 'bundle/qreport.py',
          'bundle/qprepare.py', 'bundle/qtest.py', 'CONTRACT.zh-CN.md', 'CHAINS.json']


def poison_updates(rows, poison_rows):
    horizon = len(rows) // 16
    return {k for k in range(1, horizon + 1) if poison_rows.intersection(rows[(k - 1) * 16:k * 16])}, horizon


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    from state_restore import verify
    pins = dict(read(Path(SEEDS_ROOT) / 'ASSETS.json')['source_pins'])
    for path, pin in pins.items():
        require(sha(path) == pin, 'historical input changed: ' + path)
    audit = read(Path(P2) / 'audit/P2S7-B.json'); rows = audit['expected_consumption']
    keep, horizon = poison_updates(rows, set(audit['poison_rows']))
    branch_path = OLD / 'specs/B-replace.json'; branch = read(branch_path); pins[str(branch_path)] = sha(branch_path)
    replace, _ = poison_updates(rows, set(branch['branch_manifest']['poison_rows_after']))
    data = Path(branch['config']['dataset_dir']) / f"{branch['config']['dataset']}.json"
    require(sha(data) == branch['dataset_sha256'], 'replace-branch dataset changed'); pins[str(data)] = branch['dataset_sha256']
    require(all((k in keep) == n.startswith('P') for n, k in BATCHES.items()) and X_BATCH in keep and Y_BATCH in keep, 'inserted batches lost their poison identity')
    c1, m1 = step1_chains(keep, horizon); c2, m2 = step2_chains({'orig': keep, 'replace': replace})
    chains = {**c1, **c2}; natural = {}
    for tag, c in chains.items():
        state = Path(state_path(c['bg'], c['state'])); cursor, _, _ = verify(state)
        require(cursor['global_step'] == c['state'] and cursor['identity']['seed'] == c['seed'] and cursor['expected_consumption'] == rows, f'{tag}: state/seed/order')
        pins[str(state / 'COMPLETE.json')] = sha(state / 'COMPLETE.json')
        for step, run in c['gates'].items():
            f = Path(run) / f'checkpoint-{step}' / 'adapter_model.safetensors'
            require(f.is_file(), f'missing historical adapter {f}'); pins[str(f)] = sha(f)
    for cell, meta in STEP1.items():                         # natural column of step 1 = M055 reads of the same insertions
        for arm in ARMS6:
            for lag in STEP1_LAGS:
                f = Path(M055) / 'reads/gap' / f"{cell}-{arm}@{meta['t'] + lag}.json"
                require(f.is_file(), f'missing M055 read {f}'); pins[str(f)] = sha(f); natural[f'{cell}-{arm}@{lag}'] = str(f)
    save(ROOT / 'CHAINS.json', dict(chains={t: dict(c, mapping={str(k): v for k, v in c['mapping'].items()}, gates={str(k): v for k, v in c['gates'].items()}) for t, c in chains.items()},
         step1={k: dict(v, replacements={str(a): b for a, b in v['replacements'].items()}) for k, v in m1.items()}, step2=m2, natural_reads=natural,
         rule='qproto.step1_chains / step2_chains; schedule only'), new=False)
    free = shutil.disk_usage(ROOT).free
    require(free >= 70 * 2 ** 30, 'need 70 GiB free')
    save(ROOT / 'ASSETS.json', dict(source_pins=pins, micro_kick_floor=0.0028, free_bytes=free), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/qtest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', dict(chains=len(chains), step1={k: (v['focal'], v['removed_later_poison']) for k, v in m1.items()}, windows=list(m2), free_GiB=round(free / 2 ** 30)))


if __name__ == '__main__':
    main()
