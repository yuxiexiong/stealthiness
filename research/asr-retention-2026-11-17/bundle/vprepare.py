"""CPU-only: pin assets (including the M053 lag-10 checkpoints and reads), reuse its batch list, run tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from vproto import require, read, sha, save, stage, CELLS, ARM_NAMES, plan, steps, M053, FOURARM
from vruntime import ROOT, OLD, environment

PINNED = ['bundle/vproto.py', 'bundle/vruntime.py', 'bundle/vtask.py', 'bundle/vsup.py', 'bundle/vreport.py',
          'bundle/vprepare.py', 'bundle/vtest.py', 'CONTRACT.zh-CN.md', 'ARMS.json']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    from state_restore import verify
    prior = Path(M053); require(read(prior / 'RESULT.json')['status'] == 'COMPLETE', 'M053 incomplete')
    pins = {}
    for path, pin in read(prior / 'ASSETS.json')['source_pins'].items():       # code, reader, probes, specs, datasets, state seals
        require(sha(path) == pin, 'historical input changed: ' + path); pins[path] = pin
    shutil.copy2(prior / 'ARMS.json', ROOT / 'ARMS.json')
    require(sha(ROOT / 'ARMS.json') == read(prior / 'CPU_READY.json')['pins']['ARMS.json'], 'M053 batch list changed')
    lag10 = {}; old_full = {}
    for bg, cell in CELLS:
        st = stage(bg, cell); cursor, _, _ = verify(Path(st['state']))
        require(cursor['global_step'] == st['t0'], 'state step')
        for arm in ARM_NAMES:
            tag = f'{bg}-{cell}-{arm}'
            for step, run in plan(bg, cell, arm)['gates'].items():
                f = Path(run) / f'checkpoint-{step}' / 'adapter_model.safetensors'
                require(f.is_file(), f'missing gate adapter {f}'); pins[str(f)] = sha(f)
            f = prior / 'reads/gap' / f'{tag}@{steps(bg, cell)[0]}.json'
            require(f.is_file(), f'missing M053 lag-10 read {f}'); pins[str(f)] = sha(f); lag10[tag] = str(f)
        if cell == 'late':
            for step in steps(bg, cell, [20, 40]):
                f = Path(FOURARM) / 'readouts' / f'{bg}-replace@{step}.json'
                require(f.is_file(), f'missing old readout {f}'); pins[str(f)] = sha(f); old_full[f'{bg}@{step}'] = str(f)
    free = shutil.disk_usage(ROOT).free
    require(free >= 80 * 2 ** 30, 'need 80 GiB free')
    save(ROOT / 'ASSETS.json', dict(source_pins=pins, lag10_reads=lag10, old_full_reads=old_full, free_bytes=free), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/vtest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', dict(chains=len(CELLS) * len(ARM_NAMES), pinned=len(pins), free_GiB=round(free / 2 ** 30)))


if __name__ == '__main__':
    main()
