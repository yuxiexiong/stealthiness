"""Four approved stages on the original GPU1 queue; no new scheduler."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import time

from jump_v1 import read
from lora_common import atomic_json
from queue_runs import publish

ROOT = Path('/workspace/cross-model-asr/20261008_followup_v3_vimdirect_r04')
BOUNDARY = Path('/workspace/cross-model-asr/20261007_vim_llama_v1')
DIRECT_BOUNDARY = Path('/workspace/cross-model-asr/20261008_vim_direct_r02/boundary')
QUEUE = Path('/workspace/claude-jump/jobq')
PREFIX = '088cmv3d'
REPLAY = {'t5': '089cmv3dr', 'clip': '090cmv3dr'}
SEEDS = tuple(range(1001, 1006))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    m = read(root / 'manifest.json')
    actual = {p.name: file_hash(p) for p in (root / 'code').glob('*.py')}
    if actual != m['code_sha256']:
        raise ValueError('Frozen follow-up source identity changed')
    for p, sha in m['borrowed_source_sha256'].items():
        if file_hash(p) != sha:
            raise ValueError('Borrowed scientific input changed: ' + p)
    if m['allowed_worker_gpus'] != [1]:
        raise ValueError('This deployment is physical GPU1 only')
    return m


def guard():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1'
            or read(QUEUE / 'gpu_allocation_policy.json')['allowed_worker_gpus'] != [1]):
        raise ValueError('Original GPU1 worker and allocation policy required')


def gate(root, path):
    value = read(path)
    if not value.get('passed') or value.get('code_sha256') != verify(root)['code_sha256']:
        raise ValueError('Successful engineering gate must match frozen source: ' + str(path))
    return value


def command(root, action, family=None, **args):
    # Reuse the already-validated offline environment; CUDA is supplied only by worker.py.
    env = 'CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 '
    if action == 'qwen-replay':
        # Original fixed Qwen snapshot; override inherited cache aliases without changing authentication.
        env += 'HF_HUB_CACHE=/workspace/hf_cache/hub HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub '
    argv = [str(root / 'runtime/bin/python'), str(root / 'code/next_queue.py'), action, '--root', str(root)]
    if family:
        argv += ['--family', family]
    for name, value in args.items():
        argv += ['--' + name.replace('_', '-'), str(value)]
    return env + shlex.join(argv)


def build_jobs(root):
    jobs = []
    def add(suffix, action, deps, family=None, est_min=1, **args):
        name = PREFIX + '_' + suffix
        jobs.append(dict(name=name, cwd=str(root / 'code'), deps=list(deps), est_min=est_min,
                         cmd=command(root, action, family, **args)))
        return name
    cpu = 'file:' + str(root / 'cpu_tests_passed.json')
    accepted = add('001_accept_vim_llama', 'accept', ['081cmvd_900_finish',
        'file:' + str(DIRECT_BOUNDARY / 'results/complete.json'), cpu])
    stats = add('010_normalize_existing', 'normalize', [accepted])
    gpu = add('020_qwen_replay_preflight', 'qwen-preflight', [stats, cpu], est_min=2)
    replays = [add(f'{30+i*10:03d}_qwen_s{seed}', 'qwen-replay',
        [gpu, 'file:' + str(root / 'qwen_gpu_gate.json')], seed=seed, est_min=20)
        for i, seed in enumerate((1004, 1005))]
    previous = add('050_qwen_finish', 'qwen-finish', replays)
    for family, base in [('t5', 100), ('clip', 200)]:
        prepared = add(f'{base-10:03d}_{family}_prepare', 'prepare', [previous, cpu,
            'file:' + str(root / family / 'asset_transport_complete.json')], family)
        checked = add(f'{base:03d}_{family}_gpu_preflight', 'preflight', [prepared,
            'file:' + str(root / family / 'data_gate.json'), cpu], family, 2)
        piloted = add(f'{base+10:03d}_{family}_pilot', 'pilot', [checked,
            'file:' + str(root / family / 'gpu_gate.json')], family, 10)
        ready = [piloted, 'file:' + str(root / family / 'pilot_gate.json')]
        offset = base + 20
        if family == 'clip':
            warm = add('220_clip_warmup', 'warmup', ready, family, 60)
            ready = [warm, 'file:' + str(root / 'clip/runs/clip_vit_b32_warmup/complete.json')]
            offset += 10
        formal = [add(f'{offset+i*10:03d}_{family}_s{seed}_{arm}', 'formal', ready,
            family, 180, seed=seed, arm=arm) for i, (seed, arm) in
            enumerate([(s, 'poison') for s in SEEDS] + [(1001, 'clean')])]
        planner = add(f'{base+80 if family=="t5" else base+90:03d}_{family}_plan', 'plan', formal, family)
        # Runtime planner publishes only required windows, then this stable finish job.
        previous = REPLAY[family] + '_900_finish'
    add('999_finish', 'finish', [REPLAY['t5'] + '_900_finish', REPLAY['clip'] + '_900_finish'])
    return jobs


def accept(root):
    m = verify(root)
    if file_hash(BOUNDARY / 'manifest.json') != m['boundary_manifest_sha256']:
        raise ValueError('Original Vim/Llama deployment identity changed')
    old = read(BOUNDARY / 'manifest.json')
    for name, sha in old['code_sha256'].items():
        if file_hash(BOUNDARY / 'code' / name) != sha:
            raise ValueError('Original running scientific source changed')
    from normalized_v2 import boundary_validation_complete
    c = read(DIRECT_BOUNDARY / 'results/complete.json')
    if not boundary_validation_complete(c) or not c.get('approved_validation_replacement_complete'):
        raise ValueError('Both six-trajectory cohorts and strict replays must be accepted first')
    summaries = {}
    for family in ['vim', 'llama']:
        fc = read(DIRECT_BOUNDARY / family / 'results/complete.json')
        result = read(DIRECT_BOUNDARY / family / 'results/results.json')
        if not fc.get('passed') or fc.get('poison_seeds') != 5 or fc.get('clean_seeds') != 1:
            raise ValueError('Each original cohort must have five poison and one clean')
        if len(result['formal']) != 6 or any(not v['verification'].get('passed') for v in result['dense']):
            raise ValueError('Original formal/replay evidence incomplete')
        if family == 'vim' and (fc.get('direct_bounded_confirmations') != 6 or
            len(result.get('direct_confirmations', [])) != 6 or any(
                not row['verification'].get('paired_model_hashes_verified') or
                row['verification'].get('optimizer_updates') != 120 for row in result['direct_confirmations'])):
            raise ValueError('Six approved independent Vim bounded confirmations required')
        if family == 'llama' and (not fc.get('all_required_replays_valid') or len(result['dense']) != 5):
            raise ValueError('All five original Llama strict replays remain required')
        summaries[family] = dict(complete=fc,
                                results_sha256=file_hash(DIRECT_BOUNDARY / family / 'results/results.json'))
    atomic_json(root / 'accepted_vim_llama.json', dict(passed=True, code_sha256=m['code_sha256'],
        results_sha256=file_hash(DIRECT_BOUNDARY / 'results/complete.json'), models=summaries,
        original_Vim_replay_status='unverified_due_to_non_bitwise_repeatability',
        replacement_is_independent_bounded_confirmation=True,
        ASR_is_not_acceptance_gate=True, prior_costs_not_recounted=True))


def normalize(root, refined=False):
    import normalized_v2 as n
    from jump_v1 import export
    started=time.monotonic(); gate(root, root / 'accepted_vim_llama.json')
    curves = n.load_inputs(Path(verify(root)['original_curves']), DIRECT_BOUNDARY, require_complete=True)
    if refined:
        for task in verify(root)['qwen_tasks']:
            value = read(root / 'qwen' / f'llm_s{task["seed"]}' / 'verified_points.json')
            if not value.get('passed'):
                raise ValueError('Required Qwen strict replay incomplete')
            curve = next(v for v in curves if v['model'] == 'Qwen3-8B' and v['seed'] == task['seed']
                         and v['arm'] == 'poison' and v['view'] == 'primary')
            points = dict(curve['points'])
            for step, asr in value['points']:
                if step in points and abs(points[step] - asr) >= 1e-7:
                    raise ValueError('Qwen replay conflicts with original anchor')
                points[step] = asr
            curve['points'] = sorted(points.items())
            curve['verification_level'] = 'strict_same_trajectory_verified_window'
    out = root / ('stage2/results' if refined else 'stage1/results')
    costs = [dict(cost_id=f'qwen_replay_s{task["seed"]}', family='qwen', seed=task['seed'],
        **read(root / 'qwen' / f'llm_s{task["seed"]}' / 'incremental_cost.json'))
        for task in verify(root)['qwen_tasks']] if refined else []
    n.export(curves, out, costs, make_plots=True)
    export(curves, out / 'legacy_views', True)
    atomic_json(out / 'complete.json', dict(passed=True, records=len(curves), code_sha256=verify(root)['code_sha256'],
        criterion=n.measurement_contract(), new_formal_trajectories=0))
    atomic_json(root/'costs'/('stage2_statistics.json' if refined else 'stage1_statistics.json'),
        dict(cost_id='stage2_statistics' if refined else 'stage1_statistics',phase='CPU_statistics_and_plots',
             wall_seconds=time.monotonic()-started))


def storage(root, family):
    m = verify(root)
    budget = m['storage_budget_bytes'][family]
    used = sum(p.stat().st_blocks * 512 for p in (root / family).rglob('*')
               if p.is_file() and not p.is_symlink())
    required = max(0, budget - used) + 15 * 2**30
    if shutil.disk_usage(root).free < required:
        raise ValueError(f'{family} needs remaining storage budget plus15GiB; do not delete old states')
    return dict(budget_bytes=budget, used_bytes=used, required_free_bytes=required)


def qwen_preflight(root):
    guard(); m = verify(root); gate(root, root / 'cpu_tests_passed.json')
    storage(root, 'qwen')
    import torch
    started = time.monotonic()
    if not torch.cuda.is_bf16_supported():
        raise ValueError('Original replay BF16 CUDA required')
    with torch.inference_mode():
        x = torch.ones((8, 8), device='cuda', dtype=torch.bfloat16)
        if not torch.isfinite(x @ x).all():
            raise ValueError('CUDA preflight failed')
    atomic_json(root / 'qwen_gpu_gate.json', dict(passed=True, code_sha256=m['code_sha256'],
        engineering_only_not_ASR=True, wall_seconds=time.monotonic()-started))


def qwen_replay(root, seed):
    guard(); gate(root, root / 'qwen_gpu_gate.json'); storage(root, 'qwen')
    from jump_v1_queue import llm
    m = verify(root); task = next(t for t in m['qwen_tasks'] if t['seed'] == seed)
    output = root / 'qwen' / f'llm_s{seed}'
    if output.exists():
        raise ValueError('Never overwrite a prior replay attempt')
    started = time.monotonic()
    try:
        llm(root, task, output)
    except BaseException as e:
        atomic_json(output / 'failure.json', dict(error=repr(e)))
        raise
    finally:
        atomic_json(output / 'incremental_cost.json', dict(phase='Qwen_original_window_replay',
            wall_seconds=time.monotonic()-started, includes_nested_native_receipt=True,
            source_asset_cost_recounted=False, not_new_formal_seed=True))


def model_action(root, family, action, seed=1001, arm='poison'):
    guard(); verify(root); module = importlib.import_module('next_' + family)
    if action == 'prepare':
        gate(root, root / 'cpu_tests_passed.json')
        if not read(root / family / 'asset_transport_complete.json').get('passed'):
            raise ValueError('Verified original model assets required')
        data = root / family / 'data_gate.json'
        if data.exists():
            gate(root,data)
            if family=='t5': module.load_data(root)
            else: module.verify_assets(root/'clip')
            return
        module.prepare(root)
        if family == 'clip':
            import classification as c
            receipt = read(root / 'clip/assets/complete.json')
            with module.protocol(root / 'clip'):
                _, _, ids, _ = c.load_assets(root / 'clip')
            atomic_json(data, dict(passed=True, train_n=len(ids['train']), poison_n=len(ids['poison']),
                primary_n=len(ids['primary']), confirmation_n=len(ids['confirmation']),
                cpu_native_model_parameters=receipt['models'][module.NAME]['parameters'],
                asset_identity_verified=True, cuda_initialized=receipt['cuda_initialized'],
                code_sha256=verify(root)['code_sha256']))
        if not data.exists():
            raise ValueError('Preparation did not produce a real data gate')
        atomic_json(data,dict(read(data),code_sha256=verify(root)['code_sha256']))
    elif action == 'preflight':
        gate(root, root / 'cpu_tests_passed.json'); space = storage(root, family)
        atomic_json(root / 'storage_capacity_ready.json', dict(passed=True, family=family,
            free_bytes=shutil.disk_usage(root).free, **space))
        module.preflight(root)
        path = root / family / 'gpu_gate.json'
        value = read(path)
        if not value.get('passed'):
            raise ValueError('Native CUDA preflight failed')
        atomic_json(path, dict(value, code_sha256=verify(root)['code_sha256'], storage=space))
    elif action == 'pilot':
        gate(root, root / family / 'gpu_gate.json')
        if family=='clip': module.run(root,'pilot',1001,'poison')
        else: module.pilot(root)
        path = root / family / 'pilot_gate.json'; value = read(path)
        if not value.get('passed'):
            raise ValueError('Real pretrained eight-update gate failed')
        atomic_json(path, dict(value, code_sha256=verify(root)['code_sha256']))
    elif action in ('warmup', 'formal'):
        gate(root, root / family / 'pilot_gate.json'); storage(root, family)
        if family == 'clip': module.run(root, action, seed, 'clean' if action=='warmup' else arm)
        else: module.formal(root, seed, arm)
    elif action == 'plan':
        planned = module.plan(root)
        rows = ([dict(row, candidate=row['selected_pair']) for row in planned['tasks']]
                if family=='clip' else planned)
        jobs = []
        parent = PREFIX + ('_180_t5_plan' if family == 't5' else '_290_clip_plan')
        for row in rows:
            if not row.get('candidate'):
                continue
            name = REPLAY[family] + f'_{100+len(jobs):03d}_s{row["seed"]}_{row["arm"]}'
            jobs.append(dict(name=name, cwd=str(root / 'code'), est_min=60, deps=[parent],
                cmd=command(root, 'replay', family, seed=row['seed'], arm=row['arm'])))
        jobs.append(dict(name=REPLAY[family]+'_900_finish', cwd=str(root / 'code'), est_min=1,
            deps=[parent]+[j['name'] for j in jobs], cmd=command(root, 'family-finish', family)))
        atomic_json(root / family / 'dynamic_queue_receipt.json', dict(jobs=jobs, code_sha256=verify(root)['code_sha256']))
        publish(jobs, QUEUE)
    elif action == 'replay':
        storage(root, family)
        if family=='clip':
            selected = next(v for v in read(root/'clip/refinement_plan.json')['tasks']
                            if v['seed']==seed and v['arm']==arm)
            module.replay(root, seed, arm, selected['start'], selected['end'])
        else: module.replay(root, seed, arm)
    else:
        fn = module.final if family == 't5' else module.finish
        fn(root)
        value = read(root / family / 'results/complete.json')
        if not (value.get('complete') or value.get('passed')):
            raise ValueError('Family scientific results incomplete')


def finish(root):
    import normalized_v2 as n
    from jump_v1 import export
    curves = read(root / 'stage2/results/curves.json.gz')
    costs = read(DIRECT_BOUNDARY.parent / 'results/incremental_costs.json')
    for family in ['t5', 'clip']:
        c = read(root / family / 'results/complete.json')
        if not (c.get('passed') or c.get('complete')):
            raise ValueError('Both new families must finish, including required confirmation')
    for row in read(root / 't5/results/results.json')['runs']:
        source = Path(row['source']); points = dict(row['primary_points'])
        identity = dict(model='FLAN-T5-base', seed=row['seed'], arm=row['arm'],
            poison_rate=.01 if row['arm']=='poison' else 0., full_training_budget=1250)
        if row['verification']:
            path = root / 't5/refinements' / f'replay_s{row["seed"]}_{row["arm"]}'
            records = [json.loads(v) for v in (path/'metrics.jsonl').read_text().splitlines()]
            for v in records:
                step, asr = v['optimizer_step'], v['primary']['asr']
                if step in points and abs(points[step]-asr)>=1e-7:
                    raise ValueError('Accepted T5 replay conflicts with original primary point')
                points[step]=asr
            pair = row['candidate']; bystep={v['optimizer_step']:v for v in records}
            curves.append(dict(identity, probe_n=140, view='independent_confirmation',
                points=[[s,bystep[s]['confirmation140']['asr']] for s in [pair['start'],pair['end']]],
                not_an_additional_seed=True, verification_level='fixed_primary_selected_pair_confirmation'))
        curves.append(dict(identity, probe_n=60, view='primary', points=sorted(points.items()),
            verification_level='strict_same_trajectory_verified_window' if row['verification'] else 'original_recorded_measurements'))
        curves.append(dict(identity, probe_n=60, view='common20', points=[p for p in row['primary_points']
            if p[0]%20==0 or p[0]==1250], not_an_additional_seed=True))
        costs.append(dict(cost_id=f't5_formal_{row["seed"]}_{row["arm"]}',phase='formal',family='t5',
            seed=row['seed'],wall_seconds=row['cost']['wall_seconds']))
        if row['incremental_replay_cost']:
            costs.append(dict(cost_id=f't5_replay_{row["seed"]}_{row["arm"]}',phase='replay',family='t5',
                seed=row['seed'],wall_seconds=row['incremental_replay_cost']['wall_seconds']))
    for curve in read(root / 'clip/results/curves.json.gz'):
        curve['full_training_budget']=7040; curves.append(curve)
    for path in (root/'clip').rglob('cost_receipt.json'):
        value=read(path)
        costs.append(dict(cost_id='clip_'+str(path.relative_to(root/'clip')),family='clip',
            phase=value.get('phase','unknown'),wall_seconds=value.get('wall_seconds')))
    for seed in (1004,1005):
        costs.append(dict(cost_id=f'qwen_replay_s{seed}',family='qwen',phase='replay',
            wall_seconds=read(root/'qwen'/f'llm_s{seed}'/'incremental_cost.json')['wall_seconds']))
    for path in (root/'t5/runs').glob('pilot*/cost_receipt.json'):
        v=read(path);costs.append(dict(cost_id='t5_pilot',family='t5',phase='pretrained_pilot',wall_seconds=v['wall_seconds']))
    for family in ['t5','clip']:
        for phase in ['gpu_gate','data_gate']:
            v=read(root/family/(phase+'.json'))
            wall=v.get('wall_seconds')
            if wall is None and phase=='data_gate' and family=='clip':
                wall=read(root/'clip/assets/complete.json')['elapsed_seconds']
            costs.append(dict(cost_id=family+'_'+phase,family=family,phase=phase,wall_seconds=wall))
    for path in (root/'costs').glob('*.json'):
        v=read(path)
        if 'cost_id' in v:
            costs.append({k:v[k] for k in ['cost_id','family','phase','wall_seconds'] if k in v})
    cpu=read(root/'cpu_tests_passed.json')
    costs.append(dict(cost_id='CPU_frozen_regression',phase='CPU_tests',wall_seconds=cpu['wall_seconds']))
    n.export(curves,root/'results/final',costs,make_plots=True)
    export(curves,root/'results/final/legacy_views',True)
    atomic_json(root/'results/final/cost_scope.json',dict(
        runtime_costs_in_incremental_ledger=True,prior_experiment_costs_not_recounted=True,
        preparation_transport_pilot_validation_included=True,
        unknown_manual_analysis_seconds=None,unknown_future_repair_seconds=None,
        no_raw_prompt_or_generations_export=True))
    atomic_json(root / 'results/complete.json', dict(passed=True, new_formal_trajectories=12,
        poison_seeds_per_model=5, clean_seeds_per_model=1, required_old_qwen_replays=2,
        all_four_stages_finished=True, criterion=n.measurement_contract(), physical_gpu=1,
        code_sha256=verify(root)['code_sha256'], scientific_results_pending_publication_audit=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['publish','accept','normalize','qwen-preflight','qwen-replay','qwen-finish',
        'prepare','preflight','pilot','warmup','formal','plan','replay','family-finish','finish'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--family', choices=['t5','clip'])
    p.add_argument('--seed', type=int, choices=SEEDS, default=1001)
    p.add_argument('--arm', choices=['poison','clean'], default='poison')
    a = p.parse_args(); verify(a.root)
    if a.action == 'publish':
        jobs = build_jobs(a.root); publish(jobs, QUEUE)
        atomic_json(a.root / 'queue_receipt.json', dict(submitted=True, jobs=jobs,
            code_sha256=verify(a.root)['code_sha256'], four_stage_order=True, formal_trajectories=12))
    else:
        guard()
        if a.action == 'accept': accept(a.root)
        elif a.action == 'normalize': normalize(a.root)
        elif a.action == 'qwen-preflight': qwen_preflight(a.root)
        elif a.action == 'qwen-replay': qwen_replay(a.root, a.seed)
        elif a.action == 'qwen-finish': normalize(a.root, True)
        elif a.action == 'finish': finish(a.root)
        else: model_action(a.root, a.family, a.action, a.seed, a.arm)


if __name__ == '__main__':
    main()
