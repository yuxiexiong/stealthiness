"""CPU-only: register the two new data orders, pin assets, run tests. No CUDA, no model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from oproto import require, read, sha, save, digest, HIST, P2, ORDERS, TRAINER_SEED, new_order, poison_updates
from oruntime import ROOT, OLD, environment

SEEDS_ROOT = Path('/workspace/asr-seeds-20261117-r01')
PINNED = ['bundle/oproto.py', 'bundle/oruntime.py', 'bundle/otask.py', 'bundle/osup.py', 'bundle/oprepare.py', 'bundle/otest.py', 'README.zh-CN.md', 'ORDERS.json']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    pins = dict(read(SEEDS_ROOT / 'ASSETS.json')['source_pins'])
    for path, pin in pins.items():
        require(sha(path) == pin, 'historical input changed: ' + path)
    audit = read(Path(P2) / 'audit/P2S7-B.json'); base = audit['expected_consumption']
    orders = {}
    for s in ORDERS:
        seq = new_order(audit['source_row_ids'], s)
        require(sorted(seq) == sorted(base) and seq != base, 'new order must be a different permutation of the same rows')
        same = sum(1 for a, b in zip(seq, base) if a == b)
        orders[f'O{s}'] = dict(order_seed=s, sequence_sha256=digest(seq), rows_in_same_position_as_B=same,
                               poison_updates_to_520=poison_updates(seq, audit['poison_rows'], 520))
    save(ROOT / 'ORDERS.json', dict(orders=orders, trainer_seed=TRAINER_SEED, rule='oproto.new_order: random.Random(order_seed).shuffle of all rows'), new=False)
    free = shutil.disk_usage(ROOT).free
    require(free >= 60 * 2 ** 30, 'need 60 GiB free')
    save(ROOT / 'ASSETS.json', dict(source_pins=pins, free_bytes=free), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/otest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {x: sha(ROOT / x) for x in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', {t: dict(n_poison_updates_to_520=len(v['poison_updates_to_520']), same_position=v['rows_in_same_position_as_B']) for t, v in orders.items()}, 'free_GiB', round(free / 2 ** 30))


if __name__ == '__main__':
    main()
