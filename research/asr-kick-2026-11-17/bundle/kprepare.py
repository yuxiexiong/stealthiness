"""CPU-only asset pinning and readiness. No CUDA, no model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from kproto import require, read, sha, save, OLD_GRID, ref_save_steps, SEGMENTS, STAGES
from kruntime import ROOT, OLD, environment

M045 = Path('/workspace/asr-ordering-20261116-r01')
PINNED = ['bundle/kproto.py', 'bundle/kruntime.py', 'bundle/ktask.py', 'bundle/ksup.py', 'bundle/kreport.py',
          'bundle/kprepare.py', 'bundle/ktest.py', 'CONTRACT.zh-CN.md']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    from state_restore import verify
    old = read(M045 / 'ASSETS.json')
    source = Path(old['restore']); cursor, layout, _ = verify(source)
    require(cursor['global_step'] == 320, 'restore boundary')
    pins = {}
    for path, pin in old['source_pins'].items():       # historical code, reader, probes, spec, source seal
        require(sha(path) == pin, 'historical input changed: ' + path); pins[path] = pin
    curves = {}
    for step in OLD_GRID:
        r = old['curves'][str(step)]; require(sha(r['path']) == r['sha256'], 'old readout changed'); curves[str(step)] = r
        pins[r['path']] = r['sha256']
    for step in [s for s in ref_save_steps() if s in OLD_GRID]:
        f = Path(old['reference_run']) / f'checkpoint-{step}' / 'adapter_model.safetensors'
        require(f.is_file(), 'old reference adapter missing'); pins[str(f)] = sha(f)
    spec = read(OLD / 'specs/B-replace.json'); poison = set(spec['branch_manifest']['poison_rows_after'])
    rows = cursor['expected_consumption']
    tails = {s: [k for k in range(t + 1, t + 31) if poison.intersection(rows[(k - 1) * 16:k * 16])] for s, t in STAGES.items()}
    free = shutil.disk_usage(ROOT).free
    require(free >= 40 * 2 ** 30, 'need 40 GiB free')
    require(shutil.disk_usage('/dev/shm').free >= 20 * 2 ** 30, 'need 20 GiB in /dev/shm for calibration candidates')
    save(ROOT / 'ASSETS.json', dict(restore=str(source), reference_run=old['reference_run'], curves=curves, source_pins=pins,
         layout=layout, free_bytes=free, poison_updates_in_tail=tails, m045_assets_sha256=sha(M045 / 'ASSETS.json')), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/ktest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', dict(poison_updates_in_tail={k: len(v) for k, v in tails.items()}, free_GiB=round(free / 2 ** 30, 1)))


if __name__ == '__main__':
    main()
