"""Approved bounded Vim confirmation: measure both probe sets during training."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path('/workspace/cross-model-asr/20261008_vim_direct')
SOURCE = Path('/workspace/cross-model-asr/20261007_vim_llama_v1')
PREFIX = '077cmvd'
STOP = 120
BUDGET = 7040
WINDOW = 70
SEEDS = tuple(range(1001, 1006))
ARMS = [(s, 'poison') for s in SEEDS] + [(1001, 'clean')]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def verify(root):
    m = read(root / 'manifest.json')
    if {p.name: sha(p) for p in (root / 'code').glob('*.py')} != m['code_sha256']:
        raise ValueError('Direct-confirmation source identity changed')
    old = read(SOURCE / 'manifest.json')
    if sha(SOURCE / 'manifest.json') != m['original_manifest_sha256']:
        raise ValueError('Original deployment identity changed')
    if {p.name: sha(p) for p in (SOURCE / 'code').glob('*.py')} != old['code_sha256']:
        raise ValueError('Original scientific source changed')
    for path, expected in m['borrowed_input_sha256'].items():
        if sha(path) != expected: raise ValueError('Borrowed start/data identity changed: ' + path)
    if not m['user_approved_direct_confirmation'] or m['stop_updates'] != STOP or m['full_budget'] != BUDGET:
        raise ValueError('Approved bounded design required')
    return m


def native():
    sys.path.insert(0, str(SOURCE / 'code'))
    import boundary_v1 as b
    return b


def storage(root):
    m = verify(root)
    used = sum(p.stat().st_blocks * 512 for p in (root / 'runs').rglob('*')
               if p.is_file() and not p.is_symlink())
    required = max(0, m['storage_budget_bytes'] - used) + 15 * 2**30
    if shutil.disk_usage(root).free < required:
        raise ValueError('Remaining direct-confirmation budget plus15GiB required')
    return dict(used_bytes=used, required_free_bytes=required)


def restore_rng(state, torch, np):
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']: torch.cuda.set_rng_state_all(state['cuda_rng'])


def clone_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def selected_pair(points):
    from normalized_v2 import summarize
    return summarize(points, BUDGET, probe_n=180, complete=False)['first_window']


def fixed_confirmation(pair, confirmation):
    if pair is None: return None
    values = {r['optimizer_step']: r['triggered']['asr'] for r in confirmation}
    a, b = pair['start'], pair['end']
    return dict(start=a, end=b, probe_n=900, start_asr=values[a], end_asr=values[b],
                gain_pp=100 * (values[b] - values[a]), primary_selected_before_confirmation=True,
                no_second_window_search=True)


def exact_rates(record):
    for condition in ['triggered', 'untriggered']:
        value = record[condition]
        value['recorded_float32_asr'] = value['asr']
        value['asr'] = value['successes'] / value['n']
    return record


def prepare(root):
    started = time.monotonic(); m = verify(root)
    b = native()
    import torch
    from lora_common import atomic_json
    with b.vim_protocol(SOURCE) as c:
        train, test, ids, _ = c.load_assets(SOURCE / 'vim')
        if (len(ids['train']), len(ids['validation']), len(test), len(ids['poison']),
                len(ids['primary']), len(ids['confirmation'])) != (45000, 5000, 10000, 2250, 180, 900):
            raise ValueError('Frozen CIFAR counts changed')
        if set(ids['primary']) & set(ids['confirmation']): raise ValueError('Probe splits overlap')
        starts = []
        for seed, arm in ARMS:
            source = SOURCE / 'vim/runs' / f'vim_small_formal_s{seed}_{arm}'
            state = torch.load(source / 'state-0000.pt', map_location='cpu', weights_only=False)
            if state['step'] != 0 or state['epoch'] != 0:
                raise ValueError('Original adapted epoch0 start required')
            actual = c.model_hash(SimpleNamespace(state_dict=lambda: state['model']))
            if actual != state['model_sha256'] or actual != read(source / 'manifest.json')['source_model_sha256']:
                raise ValueError('Adapted model tensor identity changed')
            starts.append(dict(seed=seed, arm=arm, model_sha256=actual,
                               checkpoint_sha256=sha(source / 'state-0000.pt')))
        if torch.cuda.is_initialized(): raise ValueError('CPU preparation initialized CUDA')
    result = dict(passed=True, code_sha256=m['code_sha256'], starts=starts,
                  data_counts=[45000, 5000, 10000, 2250, 180, 900], probe_splits_disjoint=True,
                  cuda_initialized=False, wall_seconds=time.monotonic()-started,
                  borrowed_download_and_adaptation_not_recounted=True)
    atomic_json(root / 'cpu_prepare.json', result)


def train(root, seed, arm, engineering=False):
    started = time.monotonic()
    b = native(); b.gpu1_only(); m = verify(root); storage(root); b.approved_arm(seed, arm)
    from lora_common import atomic_json
    import numpy as np
    import torch
    from boundary_v1_queue import updated_directions
    from normalized_v2 import summarize
    for gate in ['cpu_prepare.json', 'cpu_tests_passed.json']:
        value = read(root / gate)
        if not value.get('passed') or value.get('code_sha256') != m['code_sha256']:
            raise ValueError('CPU validation matching frozen code required')
    if not engineering:
        value = read(root / 'gpu_gate.json')
        if not value.get('passed') or value.get('code_sha256') != m['code_sha256'] or value.get('optimizer_updates') != 8:
            raise ValueError('Fresh direct-engineering GPU gate required')
    mem_available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                             if s.startswith('MemAvailable:'))) * 1024
    if mem_available < 12 * 2**30: raise ValueError('12GiB CPU RAM for bounded snapshot ring required')
    output = root / ('engineering' if engineering else 'runs') / f's{seed}_{arm}'
    output.mkdir(parents=True, exist_ok=False)
    costs = dict(load_seconds=0., training_seconds=0.,
        evaluation_seconds=0., checkpoint_seconds=0., optimizer_updates=0)
    try:
        with b.vim_protocol(SOURCE) as c:
            c.configure(seed); device = torch.device('cuda')
            if not torch.cuda.is_bf16_supported(): raise ValueError('Native BF16 CUDA required')
            train_data, test, ids, _ = c.load_assets(SOURCE / 'vim')
            source = SOURCE / 'vim/runs' / f'vim_small_formal_s{seed}_{arm}'
            state = torch.load(source / 'state-0000.pt', map_location='cpu', weights_only=False)
            model = c.make_model(b.VIM_NAME); model.load_state_dict(state['model']); model.to(device).train()
            if c.model_hash(model) != state['model_sha256']: raise ValueError('GPU start identity changed')
            optimizer = torch.optim.AdamW(model.parameters(), lr=c.SETTINGS['lr'],
                                           weight_decay=c.SETTINGS['weight_decay'])
            optimizer.load_state_dict(state['optimizer']); restore_rng(state, torch, np)
            costs['load_seconds'] = time.monotonic()-started
            limit = 8 if engineering else STOP
            atomic_json(output / 'manifest.json', dict(seed=seed, arm=arm, stop_updates=limit,
                full_training_budget=BUDGET, original_start_sha256=sha(source / 'state-0000.pt'),
                original_model_sha256=state['model_sha256'], settings=c.SETTINGS,
                model='Vim-S', dose=.05 if arm=='poison' else 0., code_sha256=m['code_sha256'],
                phase='engineering' if engineering else 'independent_bounded_confirmation',
                not_same_trajectory_replay=True, not_new_full_formal_seed=True,
                primary_probe_n=180, independent_probe_n=900, measured_every_update=True,
                selected_model_snapshots_are_model_only=True))
            costs['checkpoint_seconds'] += c.save_state(output/'state-start.pt', model, optimizer, 0, 0)
            primary, confirmation, points, pair = [], [], [], None
            ring = deque(maxlen=WINDOW+1)
            if engineering:
                before = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if '.mixer.' in n}
                native_globals = model.layers[0].mixer.forward.__func__.__globals__
                native_call = native_globals['mamba_inner_fn_no_out_proj']; native_calls = [0]
                def counted(*args, **kwargs):
                    native_calls[0] += 1
                    return native_call(*args, **kwargs)
                native_globals['mamba_inner_fn_no_out_proj'] = counted

            def measure(step):
                nonlocal pair
                a = exact_rates(c.measure(model, test, ids['primary'], device, output, step, 'primary'))
                z = exact_rates(c.measure(model, test, ids['confirmation'], device, output, step, 'confirmation'))
                if a['model_sha256'] != z['model_sha256']: raise ValueError('Paired probes saw different parameters')
                c.append(output/'metrics.jsonl', a); c.append(output/'metrics.jsonl', z)
                primary.append(a); confirmation.append(z); points.append([step, a['triggered']['asr']])
                costs['evaluation_seconds'] += a['evaluation_seconds'] + z['evaluation_seconds']
                if not engineering and pair is None:
                    ring.append((step, clone_cpu(model)))
                    chosen = selected_pair(points)
                    if chosen:
                        pair = chosen
                        for s in [pair['start'], pair['end']]:
                            frozen = next(values for k,values in ring if k == s)
                            original_hash = next(x['model_sha256'] for x in primary if x['optimizer_step']==s)
                            if c.model_hash(SimpleNamespace(state_dict=lambda: frozen)) != original_hash:
                                raise ValueError('Selected snapshot differs from its recorded readout')
                            t = time.monotonic(); torch.save(dict(model=frozen, step=s,
                                model_sha256=original_hash, model_only=True), output/f'selected-model-{s:04d}.pt')
                            costs['checkpoint_seconds'] += time.monotonic()-t
                        ring.clear()
                        atomic_json(output/'selected_pair.json', dict(primary_pair=pair,
                            independent_confirmation=fixed_confirmation(pair, confirmation)))

            measure(0)
            order = list(ids['train']); random.Random(seed).shuffle(order)
            data = c.Images(train_data, order, ids['poison'] if arm=='poison' else (), True, seed=seed, epoch=0)
            try:
                for step, (images, labels, identities, poisoned) in enumerate(c.loader(data), 1):
                    if engineering: native_calls[0] = 0
                    value = c.update(model, optimizer, images, labels, device)
                    if engineering and native_calls[0] != 48: raise ValueError('All48 native directions must execute')
                    costs['training_seconds'] += value['training_seconds']; costs['optimizer_updates'] += 1
                    c.append(output/'training.jsonl', dict(optimizer_step=step, epoch=0, examples=len(labels),
                        poison_examples=int(poisoned.sum()), **value))
                    measure(step)
                    if step == limit: break
            finally:
                if engineering: native_globals['mamba_inner_fn_no_out_proj'] = native_call
            costs['checkpoint_seconds'] += c.save_state(output/'state-end.pt', model, optimizer, 0, limit)
            if len(primary) != limit+1 or len(confirmation) != limit+1: raise ValueError('Direct measurement grid incomplete')
            result = dict(passed=True, complete=True, optimizer_updates=limit, full_training_budget=BUDGET,
                full_budget_training_completed=False, not_same_trajectory_replay=True, code_sha256=m['code_sha256'],
                paired_model_hashes_verified=True, primary_probe_n=180, independent_probe_n=900,
                primary_summary=summarize(points, BUDGET, probe_n=180, complete=False),
                fixed_confirmation=fixed_confirmation(pair, confirmation),
                no_ASR_or_normal_performance_success_gate=True)
            if engineering:
                changed = {n:not torch.equal(v,dict(model.named_parameters())[n].detach().cpu()) for n,v in before.items()}
                groups = updated_directions(changed)
                if not all(groups.values()): raise ValueError('Every24 bidirectional mixer must update')
                result.update(updated_directions=groups, native_calls_per_training_forward=48,
                              engineering_only_not_formal_ASR=True)
                atomic_json(root/'gpu_gate.json', result)
            atomic_json(output/'complete.json', result)
    except BaseException as exc:
        atomic_json(output/'failure.json', dict(error=repr(exc), original_source=str(SOURCE)))
        raise
    finally:
        wall = time.monotonic()-started
        costs.update(other_seconds=max(0.,wall-sum(costs[k] for k in ['load_seconds','training_seconds','evaluation_seconds','checkpoint_seconds'])),
                     wall_seconds=wall, phase='engineering' if engineering else 'bounded_confirmation',
                     borrowed_download_and_adaptation_not_recounted=True, stop_updates=8 if engineering else STOP)
        atomic_json(output/'cost_receipt.json', costs)


def build_jobs(root):
    import shlex
    env = 'CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 '
    def job(suffix, action, deps, seed=1001, arm='poison'):
        argv = [SOURCE/'vim/venv/bin/python', root/'code/vim_direct.py', action,
                '--root', root, '--seed', seed, '--arm', arm]
        return dict(name=PREFIX+'_'+suffix, cwd=str(root/'code'), deps=deps,
                    cmd=env+shlex.join(map(str,argv)), est_min=15)
    pre = job('010_gpu_preflight', 'preflight', ['file:'+str(root/'cpu_tests_passed.json'),
        'file:'+str(root/'cpu_prepare.json'), '068cmvlv1_260_llama_s1001_clean'])
    jobs = [pre] + [job(f'{110+i*10:03d}_vim_s{seed}_{arm}', 'train',
        [pre['name'], 'file:'+str(root/'gpu_gate.json')], seed, arm) for i,(seed,arm) in enumerate(ARMS)]
    jobs.append(job('900_finish', 'finish', [j['name'] for j in jobs[1:]]))
    return jobs


def finish(root):
    b = native(); b.gpu1_only(); verify(root)
    from lora_common import atomic_json
    from jump_v1 import export
    from boundary_v1_queue import trajectories
    curves, direct = [], []
    formal = trajectories(SOURCE, 'vim')
    for row in formal:
        for view, points in [('primary',row['points']),('common20',row['common20'])]:
            curves.append(dict(model='Vim-S',seed=row['seed'],arm=row['arm'],poison_rate=.05 if row['arm']=='poison' else 0.,
                probe_n=180,view=view,points=points,full_training_budget=BUDGET,
                verification_level='original_recorded_measurements',not_an_additional_seed=view!='primary'))
    for seed, arm in ARMS:
        path = root/'runs'/f's{seed}_{arm}';complete=read(path/'complete.json')
        if not complete['passed'] or complete['optimizer_updates']!=STOP or not complete['paired_model_hashes_verified']:
            raise ValueError('All six direct bounded confirmations must complete')
        records=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
        for kind, denominator, view in [('primary',180,'independent_bounded_primary'),
                                      ('confirmation',900,'independent_bounded900')]:
            points=[[v['optimizer_step'],v['triggered']['asr']] for v in records if v['kind']==kind]
            if [s for s,a in points]!=list(range(STOP+1)): raise ValueError('Bounded grid changed')
            curves.append(dict(model='Vim-S',seed=seed,arm=arm,poison_rate=.05 if arm=='poison' else 0.,
                probe_n=denominator,view=view,points=points,full_training_budget=BUDGET,
                not_an_additional_seed=True,not_same_trajectory_as_original=True,
                verification_level='direct_paired_measurement_new_bounded_trajectory'))
        direct.append(dict(seed=seed,arm=arm,verification=complete,cost=read(path/'cost_receipt.json')))
    target=root/'boundary';llama=SOURCE/'llama/results';curves+=read_gzip(llama/'curves.json.gz')
    export(curves,target/'results',True)
    atomic_json(target/'vim/results/results.json',dict(formal=formal,dense=[],direct_confirmations=direct,
        original_required_replays_unverified=True,original_failed_replays_retained=True))
    atomic_json(target/'vim/results/complete.json',dict(passed=True,poison_seeds=5,clean_seeds=1,
        formal_trajectories=6,direct_bounded_confirmations=6,original_required_replays_valid=False))
    (target/'llama/results').mkdir(parents=True,exist_ok=True)
    for name in ['results.json','complete.json','curves.json.gz']:
        shutil.copyfile(llama/name,target/'llama/results'/name)
    atomic_json(target/'results/complete.json',dict(passed=True,formal_trajectories=12,
        all_required_replays_valid=False,user_approved_direct_confirmation=True,
        approved_validation_replacement_complete=True,original_vim_replays_unverified=True,
        direct_bounded_confirmations=6,not_new_full_formal_seeds=True))
    atomic_json(root/'results/complete.json',read(target/'results/complete.json'))
    costs = [dict(cost_id=f'vim_direct_s{row["seed"]}_{row["arm"]}',family='vim',
                  phase='independent_bounded_confirmation',wall_seconds=row['cost']['wall_seconds']) for row in direct]
    costs.append(dict(cost_id='vim_direct_engineering',family='vim',phase='GPU_engineering',
                     wall_seconds=read(root/'engineering/s1001_poison/cost_receipt.json')['wall_seconds']))
    for name in ['cpu_prepare.json','cpu_tests_passed.json']:
        costs.append(dict(cost_id='vim_direct_'+name,phase='CPU_validation',wall_seconds=read(root/name)['wall_seconds']))
    atomic_json(root/'results/incremental_costs.json',costs)
    atomic_json(root/'results/prior_cost_scope.json',dict(original_formal_and_Llama_replay_costs_counted_once_in_prior_group=True,
        borrowed_download_install_adaptation_not_recounted=True,
        original_Vim_failed_replay_wall_seconds=sum(read(p)['wall_seconds'] for p in
            (SOURCE/'vim/refinements').glob('*/cost_receipt.json')),
        original_Vim_diagnosis_wall_seconds=read(SOURCE.parent/'20261008_vim_replay_diagnosis/results/diagnosis/diagnosis.json')['wall_seconds'],
        unknown_manual_analysis_seconds=None))


def read_gzip(path):
    import gzip
    with gzip.open(path,'rt') as f: return json.load(f)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','preflight','train','finish'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--seed',type=int,choices=SEEDS,default=1001)
    p.add_argument('--arm',choices=['poison','clean'],default='poison');a=p.parse_args()
    if a.action=='prepare': prepare(a.root)
    elif a.action=='finish': finish(a.root)
    else: train(a.root,a.seed,a.arm,a.action=='preflight')
