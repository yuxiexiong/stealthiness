"""CPU-only: verify the four composed step-320 states, register chains, pin assets, run tests. No CUDA, no model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import shutil
import subprocess
import sys
from hproto import require, read, sha, save, chains, state_path, STATES, WEIGHTS, CONT, BATCHES, T0, CROSS
from hruntime import ROOT, OLD, environment

PINNED = ['bundle/hproto.py', 'bundle/hruntime.py', 'bundle/htask.py', 'bundle/hsup.py', 'bundle/hreport.py',
          'bundle/hprepare.py', 'bundle/htest.py', 'CONTRACT.zh-CN.md', 'CHAINS.json']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    from state_restore import verify
    pins = {}
    branch_path = OLD / 'specs/B-replace.json'; branch = read(branch_path); pins[str(branch_path)] = sha(branch_path)
    data = Path(branch['config']['dataset_dir']) / f"{branch['config']['dataset']}.json"
    require(sha(data) == branch['dataset_sha256'], 'replace-branch dataset changed'); pins[str(data)] = branch['dataset_sha256']
    poison_rows = set(branch['branch_manifest']['poison_rows_after'])
    cursors = {}; files = {}
    for s in STATES:
        state = Path(state_path(s, T0)); cursor, _, _ = verify(state); cursors[s] = cursor
        require(cursor['global_step'] == T0 and cursor['identity']['seed'] == 1007, f'{s}: step/seed')
        files[s] = {f.name: sha(f) for f in state.iterdir() if f.is_file()}
        for name, h in files[s].items():
            pins[str(state / name)] = h
    rows = cursors['RR']['expected_consumption']
    require(all(c['expected_consumption'][T0 * 16:] == rows[T0 * 16:] for c in cursors.values()), 'the four states do not share the same future inputs')
    w, o = 'adapter_model.safetensors', 'optimizer.pt'
    # composition: weights and continuation state come from the registered histories
    for s in STATES:
        require(files[s][w] == files[WEIGHTS[s] * 2][w], f'{s}: weights are not those of history {WEIGHTS[s]}')
        require(files[s][o] == files[CONT[s] * 2][o], f'{s}: optimizer state is not that of history {CONT[s]}')
    require(files['KK'][w] != files['RR'][w] and files['KK'][o] != files['RR'][o], 'the two histories do not differ')
    upd = lambda k: rows[(k - 1) * 16:k * 16]
    require(all(bool(poison_rows.intersection(upd(k))) == n.startswith('P') for n, k in BATCHES.items()), 'inserted batches lost their poison identity')
    natural = [k for k in range(T0 + 1, T0 + 11) if poison_rows.intersection(upd(k))]
    reg = chains()
    for tag, c in reg.items():
        for step, run in c['gates'].items():
            f = Path(run) / f'checkpoint-{step}' / 'adapter_model.safetensors'
            require(f.is_file(), f'missing historical adapter {f}'); pins[str(f)] = sha(f)
    save(ROOT / 'CHAINS.json', dict(chains={t: dict(c, mapping={str(k): v for k, v in c['mapping'].items()}, gates={str(k): v for k, v in c['gates'].items()}) for t, c in reg.items()},
         states={s: dict(dir=STATES[s], weights=WEIGHTS[s], continuation=CONT[s], adapter_sha256=files[s][w], optimizer_sha256=files[s][o]) for s in STATES},
         inserted={n: dict(update=k, poison_rows=len(poison_rows.intersection(upd(k)))) for n, k in BATCHES.items()},
         natural_poison_updates_321_330=natural, rule='hproto.chains'), new=False)
    free = shutil.disk_usage(ROOT).free
    require(free >= 40 * 2 ** 30, 'need 40 GiB free')
    save(ROOT / 'ASSETS.json', dict(source_pins=pins, free_bytes=free), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/htest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', dict(chains=len(reg), natural_poison_321_330=natural, inserted={n: len(poison_rows.intersection(upd(k))) for n, k in BATCHES.items()}, free_GiB=round(free / 2 ** 30)))


if __name__ == '__main__':
    main()
