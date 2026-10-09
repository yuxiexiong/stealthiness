"""Registered uncertainty follow-up, using only the existing GPU1 worker."""
import argparse
import copy
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from lora_common import atomic_json
from normalized_v2 import checked_points, cost_ledger, export
from queue_runs import publish

QUEUE = Path('/workspace/claude-jump/jobq')
WORKER = 3052130
PREFIX = '096cmuf'
QWEN_SEEDS = (1006, 1007, 1008, 1009, 1010)
LLAVA_SEEDS = (1004, 1005, 1010)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, prepared=True):
    m = read(root / 'manifest.json')
    if m['prefix'] != PREFIX or m['worker_pid'] != WORKER or m['physical_gpu'] != 1:
        raise ValueError('Registered original worker/GPU/prefix changed')
    files = {p.name for p in (root / 'code').glob('*.py')}
    if files != set(m['code_sha256']):
        raise ValueError('New isolated source inventory changed')
    for name, expected in m['code_sha256'].items():
        if sha(root / 'code' / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    if sha(m['parent_curves']) != m['parent_curves_sha256']:
        raise ValueError('Completed parent scientific curves changed')
    if prepared:
        for name in ('cpu_tests_passed.json', 'preparation_passed.json'):
            receipt = read(root / name)
            if not receipt['passed'] or receipt['manifest_sha256'] != sha(root / 'manifest.json'):
                raise ValueError('Actual CPU acceptance of this manifest is required')
        if read(root / 'cpu_tests_passed.json')['skipped'] != 0:
            raise ValueError('CPU tests may not skip required checks')
    return m


def gpu_guard():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1':
        raise ValueError('Only the original queue worker on physical GPU1 is permitted')
    if read(QUEUE / 'gpu_allocation_policy.json')['allowed_worker_gpus'] != [1]:
        raise ValueError('Original GPU1-only allocation policy changed')


def storage(root, budget, folder):
    used = sum(p.stat().st_blocks * 512 for p in (root / folder).rglob('*')
               if p.is_file() and not p.is_symlink())
    stat = os.statvfs(root)
    required = max(0, budget - used) + 15 * 2**30
    if stat.f_bavail * stat.f_frsize < required:
        raise ValueError('Fresh remaining registered storage budget plus15GiB is required')
    return dict(free_bytes=stat.f_bavail * stat.f_frsize, required_bytes=required, used_bytes=used)


def preflight(root, family):
    gpu_guard()  # Must precede torch import.
    m = verify(root)
    receipt = root / (family + '_gpu_preflight.json')
    if receipt.exists():
        raise ValueError('Refusing to overwrite a prior engineering attempt')
    begin = time.monotonic()
    import torch
    import transformers
    import peft
    required = m['runtimes'][family]['versions']
    from importlib.metadata import version
    versions = {name: version(name) for name in required}
    if any(versions[k].split('+')[0] != v for k, v in required.items()):
        raise ValueError('Original native runtime versions changed')
    budget = m['qwen_budget_bytes' if family == 'qwen' else 'llava_storage_budget_bytes']
    disk = storage(root, budget, 'qwen' if family == 'qwen' else 'refinements')
    dtype = torch.bfloat16 if family == 'qwen' else torch.float16
    a = torch.ones(8, 8, device='cuda:0', dtype=dtype)
    if not torch.equal(a @ a, torch.full_like(a, 8)) or torch.cuda.device_count() != 1:
        raise ValueError('Real CUDA precision/device engineering check failed')
    torch.cuda.synchronize()
    result = dict(passed=True, physical_gpu=1, internal_device='cuda:0', versions=versions,
                  dtype=str(dtype), storage=disk, not_formal_or_ASR=True,
                  wall_seconds=time.monotonic() - begin)
    atomic_json(receipt, result)
    return result


def family_finish(root, family):
    verify(root)
    seeds = QWEN_SEEDS if family == 'qwen' else LLAVA_SEEDS
    for seed in seeds:
        path = (root / 'qwen' / ('llm_s' + str(seed)) / 'followup_complete.json' if family == 'qwen'
                else root / 'refinements' / ('llava_s' + str(seed)) / 'complete.json')
        value = read(path)
        if not value.get('passed', value.get('complete', False)):
            raise ValueError('Valid completion required regardless of ASR outcome')
    atomic_json(root / (family + '_complete.json'), dict(passed=True, seeds=list(seeds), new_formal_trajectories=0))


def merge_points(curves, model, seed, verified, n, include_reconstructed_zero=False):
    if not verified.get('passed') or verified.get('probe_n') != n:
        raise ValueError('Accepted original-denominator reconstruction required')
    target = next(c for c in curves if c['model'] == model and c['seed'] == seed and
                  c['arm'] == 'poison' and c['view'] == 'primary' and c['probe_n'] == n)
    merged = dict(target['points'])
    for step, value in checked_points(verified['points'], 1250, n):
        if step == 0 and n == 200 and not include_reconstructed_zero:
            continue  # No original LLaVA step0 was measured or saved.
        if step in merged:
            # Old Qwen aggregate ratios are FP32; original exact flags are checked upstream.
            counts = [round(v * n) for v in (merged[step], value)]
            if any(abs(v * n - count) > 1e-5 for v, count in zip((merged[step], value), counts)) or counts[0] != counts[1]:
                raise ValueError('New accepted point conflicts with original exact success counts')
        else:
            merged[step] = value
    target['points'] = sorted(merged.items())
    target['verification_level'] = ('strict_same_trajectory_verified_window' if n == 60
                                    else 'available_loss_adapter_behavior_anchors_exact_prefix_reconstruction')
    target['measurement_note'] = ('Original full1250 denominator; no interpolation. ' +
        ('Every original update loss and available anchor was exact.' if n == 60 else
         'Original loss logs exist every20 updates; optimizer/RNG state unavailable. New reconstructed step0 stays separate.'))
    return target


def unique_costs(values):
    found = {}
    for value in values:
        key = value['cost_id']
        if key in found and value != found[key]:
            raise ValueError('Conflicting duplicate cost_id')
        found[key] = value
    return list(found.values())


def finish(root):
    m = verify(root)
    if not all(read(root / (family + '_complete.json'))['passed'] for family in ('qwen', 'llava')):
        raise ValueError('Both ordered reconstruction families must finish')
    import followup_diagnostics
    diagnostics = followup_diagnostics.finish(root)
    if not diagnostics['passed']:
        raise ValueError('All bounded diagnostic receipts must be valid')
    curves = copy.deepcopy(json.load(gzip.open(m['parent_curves'], 'rt')))
    for seed in QWEN_SEEDS:
        output = root / 'qwen' / ('llm_s' + str(seed))
        merge_points(curves, 'Qwen3-8B', seed, read(output / 'verified_points.json'), 60)
        confirmation = read(output / 'fixed_confirmation.json')['fixed_confirmation']
        if confirmation:
            curves.append(dict(model='Qwen3-8B', seed=seed, arm='poison', poison_rate=.01,
                probe_n=140, view='followup_fixed_primary_pair_confirmation140',
                points=[[s, confirmation['endpoints'][str(s)]['target_successes'] / 140]
                        for s in (confirmation['start'], confirmation['end'])], full_training_budget=1250,
                not_an_additional_seed=True, split='independent140', primary_pair_selected_before_confirmation=True,
                verification_level='fixed_primary_selected_pair_confirmation'))
    for seed in LLAVA_SEEDS:
        merge_points(curves, 'LLaVA-1.5-7B', seed,
                     read(root / 'refinements' / ('llava_s' + str(seed)) / 'verified_points.json'), 200)
    receipts = [read(p) for p in root.rglob('incremental_cost.json')]
    receipts += [read(p) for p in (root / 'diagnostics').glob('*/cost_receipt.json')]
    receipts += [read(p) for p in (root / 'costs').glob('*.json')]
    costs = unique_costs(receipts)
    output = root / 'results'
    export(curves, output, costs, make_plots=True)
    atomic_json(output / 'diagnostic_summary.json', diagnostics)
    atomic_json(output / 'complete.json', dict(passed=True, new_formal_trajectories=0,
        qwen_strict_windows=5, llava_available_anchor_reconstructions=3,
        bounded_endpoint_diagnostics_valid=True, scientific_publication_pending=True,
        parent_run_unchanged=True, original_failures_preserved=True,
        unique_elapsed_wall_seconds=None, command_wall_is_not_elapsed=True))


def build_jobs(root, manifest):
    code = root / 'code'
    jobs = []
    previous = []
    env = ('PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '
           'CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
           'HF_HUB_CACHE=/workspace/hf_cache/hub HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub '
           'TRANSFORMERS_CACHE=/workspace/hf_cache/hub ')
    def add(suffix, python, script, args, cpu=False):
        nonlocal previous
        name = PREFIX + '_' + suffix
        command = env + ('CUDA_VISIBLE_DEVICES= ' if cpu else '') + ' '.join(shlex.quote(str(v)) for v in
            [python, code / script, '--root', root, *args])
        jobs.append(dict(name=name, cmd=command, cwd=str(code), deps=previous + [
            'file:' + str(root / 'cpu_tests_passed.json'), 'file:' + str(root / 'preparation_passed.json')], est_min=60))
        previous = [name]
    qpython = manifest['runtimes']['qwen']['python']
    vpython = manifest['runtimes']['llava']['python']
    add('010_qwen_gpu_preflight', qpython, 'uncertainty_followup.py', ['--phase', 'preflight', '--family', 'qwen'])
    for index, seed in enumerate(QWEN_SEEDS):
        add('%03d_qwen_s%d' % (20 + index * 10, seed), qpython, 'uncertainty_followup.py',
            ['--phase', 'qwen', '--seed', seed])
    add('070_qwen_finish', qpython, 'uncertainty_followup.py', ['--phase', 'family-finish', '--family', 'qwen'], True)
    add('080_llava_gpu_preflight', vpython, 'uncertainty_followup.py', ['--phase', 'preflight', '--family', 'llava'])
    for index, seed in enumerate(LLAVA_SEEDS):
        add('%03d_llava_s%d' % (90 + index * 10, seed), vpython, 'followup_vlm.py', ['--seed', seed])
    add('120_llava_finish', qpython, 'uncertainty_followup.py', ['--phase', 'family-finish', '--family', 'llava'], True)
    for index, task in enumerate(manifest['diagnostic_tasks']):
        add('%03d_%s' % (130 + index * 10, task['id']), task['python'], 'followup_diagnostics.py',
            ['run', '--task-id', task['id']])
    add('999_finish', qpython, 'uncertainty_followup.py', ['--phase', 'finish'], True)
    return jobs


def submit(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Queue publication is CPU-only with explicitly empty CUDA visibility')
    m = verify(root)
    if not Path('/proc/' + str(WORKER)).exists():
        raise ValueError('The original worker must be alive; no replacement worker is allowed')
    gpu_policy = read(QUEUE / 'gpu_allocation_policy.json')
    if gpu_policy['allowed_worker_gpus'] != [1]:
        raise ValueError('Original GPU1-only policy required')
    jobs = build_jobs(root, m)
    if any((QUEUE / folder / name).exists() for folder in ('running', 'done', 'failed', 'claims')
           for job in jobs for name in (job['name'], job['name'] + '.json')):
        raise ValueError('New scope collides with an existing execution marker')
    publish(jobs, QUEUE)
    atomic_json(root / 'queue_receipt.json', dict(submitted=True, jobs=jobs, git_revision=m['git_revision'],
        manifest_sha256=sha(root / 'manifest.json'), original_worker_pid=WORKER,
        only_physical_gpu=1, queue_hints_are_not_runtime_ETA=True))
    return dict(submitted=True, jobs=len(jobs), original_worker_pid=WORKER)


def monitor(root):
    begin = time.monotonic()
    m = verify(root)
    jobs = read(root / 'queue_receipt.json')['jobs']
    state = {}
    for job in jobs:
        name = job['name']
        marks = {folder: any((QUEUE / folder / f).exists() for f in (name, name + '.json'))
                 for folder in ('running', 'done', 'failed', 'claims')}
        state[name] = next((key for key in ('failed', 'running', 'done') if marks[key]), 'queued')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                     '--format=csv,noheader'], text=True).strip()
    gpu_pids = {s.split(',')[0].strip() for s in active.splitlines() if s.strip()}
    processes = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            cwd = (path / 'cwd').resolve()
            if str(cwd).startswith(str(root)) or path.name == str(WORKER) or path.name in gpu_pids:
                e = dict(v.split(b'=', 1) for v in (path / 'environ').read_bytes().split(b'\0') if b'=' in v)
                processes.append(dict(pid=int(path.name), cwd=str(cwd),
                    CUDA_VISIBLE_DEVICES=e.get(b'CUDA_VISIBLE_DEVICES', b'').decode(),
                    JOBQ_GPU=e.get(b'JOBQ_GPU', b'').decode()))
        except (OSError, ValueError):
            pass
    progress = []
    for seed in QWEN_SEEDS:
        output = root / 'qwen' / ('llm_s' + str(seed))
        def rows(path):
            lines = [s for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
            values = []
            for index, line in enumerate(lines):
                try: values.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines)-1: raise
            return values
        training, metrics = rows(output / 'training.jsonl'), rows(output / 'metrics.jsonl')
        progress.append(dict(family='qwen', seed=seed,
            trained_update=training[-1]['optimizer_step'] if training else 0,
            latest_native_metric=metrics[-1] if metrics else None,
            strict_acceptance_complete=(output / 'followup_complete.json').exists()))
    for seed in LLAVA_SEEDS:
        output = root / 'refinements' / ('llava_s' + str(seed))
        states = list((output / 'private/train').glob('checkpoint-*/trainer_state.json'))
        last = max([0] + [read(p)['global_step'] for p in states])
        measured = sorted((output / 'private').glob('eval-*.json'))
        try: value = read(measured[-1]) if measured else None
        except json.JSONDecodeError: value = None  # A concurrent native write is not process exit.
        progress.append(dict(family='llava', seed=seed, last_complete_saved_update=last,
            last_native_eval_update=int(measured[-1].stem.split('-')[1]) if measured else None,
            asr=value['asr'] if value else None, clean_accuracy=value['clean_acc'] if value else None,
            strict_available_anchor_acceptance_complete=(output / 'complete.json').exists()))
    stat = os.statvfs(root)
    value = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_identity_passed=True, code_files=len(m['code_sha256']), queue=state, processes=processes,
        progress=progress, GPU_compute_processes=active, free_bytes=stat.f_bavail * stat.f_frsize,
        original_worker_alive=Path('/proc/' + str(WORKER)).exists(),
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                     '--format=csv,noheader'], text=True).strip(),
        results_complete=(root / 'results/complete.json').exists())
    atomic_json(root / 'monitoring/latest.json', value)
    atomic_json(root / 'costs' / ('cpu_monitor_' + uuid.uuid4().hex + '.json'), dict(
        cost_id='uncertainty_cpu_monitor_' + uuid.uuid4().hex, phase='CPU_readonly_monitor',
        wall_seconds=time.monotonic()-begin, SSH_wall_excluded=True, CUDA_initialized=False))
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=('preflight', 'qwen', 'family-finish', 'finish', 'submit', 'monitor'), required=True)
    p.add_argument('--family', choices=('qwen', 'llava'))
    p.add_argument('--seed', type=int)
    a = p.parse_args()
    begin = time.monotonic()
    try:
        if a.phase == 'preflight': result = preflight(a.root, a.family)
        elif a.phase == 'qwen':
            gpu_guard(); verify(a.root)
            import followup_qwen
            result = followup_qwen.replay(a.root, a.seed)
        elif a.phase == 'family-finish':
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only phase requires empty CUDA')
            result = family_finish(a.root, a.family)
        elif a.phase == 'finish':
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only phase requires empty CUDA')
            result = finish(a.root)
        elif a.phase == 'submit': result = submit(a.root)
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only monitor requires empty CUDA')
            result = monitor(a.root)
        print(json.dumps(result))
    finally:
        if a.phase in ('preflight', 'family-finish', 'finish'):
            cost = dict(cost_id='uncertainty_' + uuid.uuid4().hex, phase=a.phase,
                     family=a.family, wall_seconds=time.monotonic() - begin,
                     borrowed_assets_recounted=False)
            atomic_json(a.root / 'costs' / (cost['cost_id'] + '.json'), cost)
            if a.phase == 'finish' and (a.root / 'results/complete.json').exists():
                values = [read(p) for p in a.root.rglob('incremental_cost.json')]
                values += [read(p) for p in (a.root / 'diagnostics').glob('*/cost_receipt.json')]
                values += [read(p) for p in (a.root / 'costs').glob('*.json')]
                ledger = cost_ledger(unique_costs(values))
                ledger['unique_elapsed_wall_seconds'] = None
                ledger['known_command_wall_sum_seconds'] = ledger['known_incremental_wall_seconds']
                atomic_json(a.root / 'results/incremental_costs.json', ledger)


if __name__ == '__main__':
    main()
