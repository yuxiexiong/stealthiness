"""CPU-only: pin assets for both backgrounds, choose arms by the frozen rule, run tests. No CUDA, no model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from zproto import require, read, sha, save, select_arms, BG, START, save_steps
from zruntime import ROOT, OLD, environment

M049 = Path('/workspace/asr-kick-20261117-r01')
STATE = Path('/data/asr-state-cross320-20261115-r02')
CURVES = Path('/data/asr-exposure-fourarm-20261113-p0/readouts')
PINNED = ['bundle/zproto.py', 'bundle/zruntime.py', 'bundle/ztask.py', 'bundle/zsup.py', 'bundle/zreport.py',
          'bundle/zprepare.py', 'bundle/ztest.py', 'CONTRACT.zh-CN.md', 'ARMS.json']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    from state_restore import verify
    pins = {}
    for path, pin in read(M049 / 'ASSETS.json')['source_pins'].items():      # frozen code, reader, probes, B spec and seal
        require(sha(path) == pin, 'historical input changed: ' + path); pins[path] = pin
    assets = {}; registry = {}
    for bg, c in BG.items():
        source = STATE / 'composed' / f'noop-{bg}-replace'; run = STATE / 'runs' / f'noop-{bg}-replace'
        cursor, layout, _ = verify(source)
        require(cursor['global_step'] == START, 'restore boundary')
        spec_path = OLD / f'specs/{bg}-replace.json'; spec = read(spec_path)
        pins[str(spec_path)] = sha(spec_path); pins[str(source / 'COMPLETE.json')] = sha(source / 'COMPLETE.json')
        curves = {}
        for step in sorted(set(c['gap']) | set(c['full']) | {c['base']}):
            f = CURVES / f'{bg}-replace@{step}.json'; require(f.is_file(), f'missing old readout {f}')
            pins[str(f)] = sha(f); curves[str(step)] = str(f)
        for step in sorted(set(save_steps(bg)) | {c['end']}):
            f = run / f'checkpoint-{step}' / 'adapter_model.safetensors'; require(f.is_file(), f'missing reference adapter {f}')
            pins[str(f)] = sha(f)
        poison_rows = set(spec['branch_manifest']['poison_rows_after']); rows = cursor['expected_consumption']
        horizon = len(rows) // 16
        poison = {k for k in range(1, horizon + 1) if poison_rows.intersection(rows[(k - 1) * 16:k * 16])}
        chosen = select_arms(bg, poison, horizon)
        counts = {str(k): len(poison_rows.intersection(rows[(k - 1) * 16:k * 16])) for k in chosen['window_poison'] + chosen['future_poison']}
        registry[bg] = dict(arms={t: {str(k): v for k, v in m.items()} for t, m in chosen['arms'].items()},
                            window_poison=chosen['window_poison'], controls=chosen['controls'], targets=chosen['targets'],
                            future_poison=chosen['future_poison'], poison_rows_per_batch=counts)
        assets[bg] = dict(restore=str(source), reference_run=str(run), curves=curves, layout=layout)
    registry['rule'] = 'zproto.select_arms; schedule only, no outcome, no redraw'
    save(ROOT / 'ARMS.json', registry, new=False)
    free = shutil.disk_usage(ROOT).free
    require(free >= 80 * 2 ** 30, 'need 80 GiB free')
    assets.update(source_pins=pins, micro_kick_floor=read(M049 / 'RESULT.json')['noise_floor'], free_bytes=free)
    save(ROOT / 'ASSETS.json', assets, new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/ztest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', {bg: {k: v for k, v in registry[bg].items() if k != 'arms'} for bg in BG}, 'arms', {bg: len(registry[bg]['arms']) for bg in BG}, 'free_GiB', round(free / 2 ** 30))


if __name__ == '__main__':
    main()
