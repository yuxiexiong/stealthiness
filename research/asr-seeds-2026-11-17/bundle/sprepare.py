"""CPU-only: pin assets for the from-scratch path and run tests. No CUDA, no model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from sproto import require, read, sha, save, HIST, P2, QUAL, SEEDS, plan, fixed_order
from sruntime import ROOT, OLD, environment

M054 = Path('/workspace/asr-retention-20261117-r01')
PINNED = ['bundle/sproto.py', 'bundle/sruntime.py', 'bundle/stask.py', 'bundle/ssup.py', 'bundle/sprepare.py', 'bundle/stest.py', 'CONTRACT.zh-CN.md']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    pins = {}
    for path, pin in read(M054 / 'ASSETS.json')['source_pins'].items():       # frozen code, reader, probes, specs, datasets, seals
        if os.path.exists(path) and sha(path) == pin:
            pins[path] = pin
        else:
            require('asr-stage-20261117' in path, 'historical input changed: ' + path)   # M053 run outputs are not inputs here
    hist = Path(HIST)
    cfg = read(hist / 'CONFIG_B-original.json'); pins[str(hist / 'CONFIG_B-original.json')] = sha(hist / 'CONFIG_B-original.json')
    audit_path = Path(P2) / 'audit/P2S7-B.json'; audit = read(audit_path); pins[str(audit_path)] = sha(audit_path)
    data = Path(cfg['dataset_dir']) / f"{cfg['dataset']}.json"
    require(data.is_file(), 'training dataset missing'); pins[str(data)] = sha(data)
    pins[str(Path(cfg['dataset_dir']) / 'dataset_info.json')] = sha(Path(cfg['dataset_dir']) / 'dataset_info.json')
    for step in plan(QUAL)['gates']:
        f = hist / 'runs/B-original' / f'checkpoint-{step}' / 'adapter_model.safetensors'
        require(f.is_file(), f'missing historical adapter {f}'); pins[str(f)] = sha(f)
    for step in plan(QUAL)['full']:
        f = hist / 'runs/B-original' / f'checkpoint-{step}' / 'optimizer.pt'
        require(f.is_file(), f'missing historical optimizer {f}'); pins[str(f)] = sha(f)
    pins[str(hist / 'runs/B-original/updates.jsonl')] = sha(hist / 'runs/B-original/updates.jsonl')
    order = fixed_order(audit['source_row_ids'], audit['expected_consumption'])
    keep = read('/data/asr-state-cross320-20261115-r02/composed/noop-B-replace/audit_cursor.json')
    require(keep['expected_consumption'] == audit['expected_consumption'] and keep['source_row_ids'] == audit['source_row_ids'],
            'registered order differs from the order of the B background used in M049-M054')
    poison = set(audit['poison_rows']); rows = audit['expected_consumption']
    poison_updates = [k for k in range(1, 701) if poison.intersection(rows[(k - 1) * 16:k * 16])]
    free = shutil.disk_usage(ROOT).free
    require(free >= 150 * 2 ** 30, 'need 150 GiB free for four trajectories with sealed states every 10 updates')
    save(ROOT / 'ASSETS.json', dict(source_pins=pins, seeds=SEEDS, order_len=len(order), poison_updates_to_700=poison_updates,
         original_seed=cfg['seed'], free_bytes=free), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/stest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', dict(pinned=len(pins), seeds=SEEDS, poison_updates_200_400=[k for k in poison_updates if 200 <= k <= 400], free_GiB=round(free / 2 ** 30)))


if __name__ == '__main__':
    main()
