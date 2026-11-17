"""CPU-only: pin assets for six restore paths, choose the inserted batches by the frozen rule, run tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import json
import shutil
import subprocess
import sys
from wproto import require, read, sha, save, stages, BGS, plan, read_steps, select_batches, FOURARM
from wruntime import ROOT, OLD, environment

M049 = Path('/workspace/asr-kick-20261117-r01')
PINNED = ['bundle/wproto.py', 'bundle/wruntime.py', 'bundle/wtask.py', 'bundle/wsup.py', 'bundle/wreport.py',
          'bundle/wprepare.py', 'bundle/wtest.py', 'CONTRACT.zh-CN.md', 'ARMS.json']


def main():
    environment()
    from train import runtime_guard, verify_freeze
    runtime_guard(); verify_freeze()
    require(not (ROOT / 'STARTED.json').exists(), 'cannot repin after start')
    from state_restore import verify
    pins = {}
    for path, pin in read(M049 / 'ASSETS.json')['source_pins'].items():
        require(sha(path) == pin, 'historical input changed: ' + path); pins[path] = pin
    registry = {}; old_reads = {}; stage_info = {}
    for bg in BGS:
        cursors = {}; poison_sets = {}
        for stage, st in stages(bg).items():
            source = Path(st['state']); cursor, _, _ = verify(source)
            require(cursor['global_step'] == st['t0'], f'{bg}-{stage}: state step')
            require((source / 'adapter_model.safetensors').is_file() and (source / 'adapter_config.json').is_file(), 'state adapter missing')
            pins[str(source / 'COMPLETE.json')] = sha(source / 'COMPLETE.json')
            spec_path = OLD / f"specs/{st['spec']}.json"; spec = read(spec_path); pins[str(spec_path)] = sha(spec_path)
            for name in ('REF', 'P1'):
                for step in plan(bg, stage, name)['gates']:
                    f = Path(st['gate_run']) / f'checkpoint-{step}' / 'adapter_model.safetensors'
                    require(f.is_file(), f'missing historical adapter {f}'); pins[str(f)] = sha(f)
            rows = spec['branch_manifest']['poison_rows_after'] if spec['branch_manifest'] else spec['parent_audit']['poison_rows']
            cursors[stage] = cursor; poison_sets[stage] = set(rows)
            data = Path(spec['config']['dataset_dir']) / f"{spec['config']['dataset']}.json"
            require(sha(data) == spec['dataset_sha256'], 'dataset changed'); pins[str(data)] = spec['dataset_sha256']
            stage_info[f'{bg}-{stage}'] = dict(dataset=str(data), t=st['t'], t0=st['t0'])
        order = cursors['late']['expected_consumption']; horizon = len(order) // 16
        require(all(c['expected_consumption'] == order and c['source_row_ids'] == cursors['late']['source_row_ids'] for c in cursors.values()),
                'the three stages do not share one data order')
        poison = {k for k in range(1, horizon + 1) if poison_sets['late'].intersection(order[(k - 1) * 16:k * 16])}
        batches = select_batches(poison, horizon)
        position = {s: i for i, s in enumerate(cursors['late']['source_row_ids'])}
        content = {}
        for stage in stages(bg):
            data = json.loads(Path(stage_info[f'{bg}-{stage}']['dataset']).read_text())
            content[stage] = {n: [data[position[r]] for r in order[(k - 1) * 16:k * 16]] for n, k in batches.items()}
            for n, k in batches.items():
                rows = set(order[(k - 1) * 16:k * 16])
                require(len(rows & poison_sets[stage]) == len(rows & poison_sets['late']), 'poison identity of an inserted batch differs between stages')
        require(content['early'] == content['mid'] == content['late'], 'inserted batches are not byte-identical across the stage datasets')
        registry[bg] = dict(batches=batches, poison_rows_per_batch={n: len(set(order[(k - 1) * 16:k * 16]) & poison_sets['late']) for n, k in batches.items()},
                            native_poison_updates_after_insertion={stage: [k for k in range(st['t'] + 1, st['t'] + 11) if poison_sets[stage].intersection(order[(k - 1) * 16:k * 16])]
                                                                   for stage, st in stages(bg).items()})
        f = Path(FOURARM) / 'readouts' / f'{bg}-keep@300.json'       # the step-300 fork carries the keep@300 adapter
        require(f.is_file(), 'missing old step-300 readout'); pins[str(f)] = sha(f); old_reads[f'{bg}-mid'] = str(f)
        late_end = max(read_steps(bg, 'late'))
        f = Path(FOURARM) / 'readouts' / f'{bg}-replace@{late_end}.json'
        require(f.is_file(), 'missing old late readout'); pins[str(f)] = sha(f); old_reads[f'{bg}-late-end'] = str(f)
    registry['rule'] = 'wproto.select_batches: first 3 poison-containing and first 3 pure-clean updates >= 441; schedule only, no outcome'
    save(ROOT / 'ARMS.json', registry, new=False)
    free = shutil.disk_usage(ROOT).free
    require(free >= 80 * 2 ** 30, 'need 80 GiB free')
    save(ROOT / 'ASSETS.json', dict(source_pins=pins, old_reads=old_reads, stages=stage_info,
         micro_kick_floor=read(M049 / 'RESULT.json')['noise_floor'], free_bytes=free), new=False)
    test = subprocess.run([sys.executable, str(ROOT / 'bundle/wtest.py')], capture_output=True, text=True)
    (ROOT / 'CPU_TESTS.log').write_text(test.stdout + test.stderr)
    require(test.returncode == 0, 'CPU tests failed')
    save(ROOT / 'CPU_READY.json', {'pass': True, 'no_model_loaded': True, 'pins': {p: sha(ROOT / p) for p in PINNED},
         'assets_sha256': sha(ROOT / 'ASSETS.json'), 'tests_log_sha256': sha(ROOT / 'CPU_TESTS.log')}, new=False)
    print('CPU_READY', {bg: registry[bg] for bg in BGS}, 'free_GiB', round(free / 2 ** 30))


if __name__ == '__main__':
    main()
