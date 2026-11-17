"""CPU-only: read the frozen onsets, register every chain, pin assets, run tests. No CUDA, no model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from pproto import require, read, sha, save, chains, SEEDS_ROOT, HIST, P2, BATCHES
from pruntime import ROOT, environment

PINNED = ['bundle/pproto.py', 'bundle/pruntime.py', 'bundle/ptask.py', 'bundle/psup.py', 'bundle/preport.py',
          'bundle/pprepare.py', 'bundle/ptest.py', 'CONTRACT.zh-CN.md', 'CHAINS.json', 'ONSETS.json']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    from state_restore import verify
    seeds_root = Path(SEEDS_ROOT)
    require(sha(ROOT / 'CONTRACT.zh-CN.md') == read(seeds_root / 'CPU_READY.json')['pins']['CONTRACT.zh-CN.md'], 'contract differs from the one frozen before phase 1')
    shutil.copy2(seeds_root / 'ONSETS.json', ROOT / 'ONSETS.json')
    onsets = read(ROOT / 'ONSETS.json')['seeds']
    pins = dict(read(seeds_root / 'ASSETS.json')['source_pins'])
    for path, pin in pins.items():
        require(sha(path) == pin, 'historical input changed: ' + path)
    audit = read(Path(P2) / 'audit/P2S7-B.json'); rows = audit['expected_consumption']; poison_rows = set(audit['poison_rows'])
    horizon = len(rows) // 16
    poison = {k for k in range(1, horizon + 1) if poison_rows.intersection(rows[(k - 1) * 16:k * 16])}
    require(all((k in poison) == name.startswith('P') for name, k in BATCHES.items()), 'inserted batches do not have the registered poison identity')
    registry = {}; traj_reads = {}; seeds = {}
    for tag, info in sorted(onsets.items()):
        seed = int(tag[1:])
        if not info.get('usable'):
            seeds[tag] = dict(usable=False, reason=info.get('note') or 'cells not available under the frozen rule', onset=info.get('onset'))
            continue
        c, mini = chains(seed, info, poison, horizon)
        for name, chain in c.items():
            state = seeds_root / 'runs' / tag / f"checkpoint-{chain['state']}"
            cursor, _, _ = verify(state); require(cursor['global_step'] == chain['state'], 'sealed state step')
            pins[str(state / 'COMPLETE.json')] = sha(state / 'COMPLETE.json')
            for g in chain['gates']:
                f = seeds_root / 'runs' / tag / f'checkpoint-{g}' / 'adapter_model.safetensors'
                require(f.is_file(), f'missing trajectory adapter {f}'); pins[str(f)] = sha(f)
            registry[name] = dict(chain, mapping={str(k): v for k, v in chain['mapping'].items()})
        on = info['onset']
        if mini['testable']:
            traj_reads[f'{tag}-traj'] = [on - 20, on + 10]
            for s in (on - 20, on + 10):
                f = seeds_root / 'runs' / tag / f'checkpoint-{s}' / 'adapter_model.safetensors'; pins[str(f)] = sha(f)
        seeds[tag] = dict(usable=True, onset=on, early=info['early'], mid=info['mid'], mini=mini)
    save(ROOT / 'CHAINS.json', dict(chains=registry, trajectory_reads=traj_reads, seeds=seeds,
         rule='contract sections 4-5; cells and windows computed from ONSETS.json only'), new=False)
    free = shutil.disk_usage(ROOT).free
    require(free >= 60 * 2 ** 30, 'need 60 GiB free')
    save(ROOT / 'ASSETS.json', dict(source_pins=pins, free_bytes=free), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/ptest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', dict(chains=len(registry), seeds={t: (v['usable'], v.get('onset'), v.get('early'), v.get('mid'), v.get('mini', {}).get('window_poison')) for t, v in seeds.items()}, free_GiB=round(free / 2 ** 30)))


if __name__ == '__main__':
    main()
