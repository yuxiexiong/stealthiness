"""Isolated 10% T2I preparation and jobs on the existing server workers."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import time

from lora_common import atomic_json, digest
from queue_runs import publish
import lora_t2i as t2i
from t2i_dose10 import configure

ROOT = Path('/workspace/cross-model-asr/20261004_t2i_dose10')
OLD = Path('/workspace/cross-model-asr/20261004_t2i_dose5')
EXT = Path('/workspace/cross-model-asr/20261004_seed10')
LLM = Path('/workspace/cross-model-asr/20261004_lora')
QUEUE = Path('/workspace/claude-jump/jobq')
PREFIX = '049cmd10'


def read(path):
    return json.loads(path.read_text())


def hashes(code):
    return {p.name: t2i.file_hash(p) for p in code.glob('*.py')}


def copy_rows(source, destination, name, mapping, count):
    """Copy only bit-identical rows, remapping text IDs by exact prompt string."""
    import numpy as np
    state = read(source / (name + '.json'))
    old = np.load(source / (name + '.npy'), mmap_mode='r')
    path = destination / (name + '.npy')
    if path.exists():
        raise ValueError('Use a fresh dose data directory; never overwrite prepared banks')
    new = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint16, shape=(count, *old.shape[1:]))
    result = {'count': count, 'shape': list(new.shape), 'rows': {}}
    for new_id, old_id in mapping.items():
        expected = state['rows'][str(old_id)]
        if hashlib.sha256(old[old_id].tobytes()).hexdigest() != expected:
            raise ValueError('Corrupt source cache: ' + name)
        new[new_id] = old[old_id]
        result['rows'][str(new_id)] = expected
    if name == 'clean_latents':
        result['source_rgb_sha256'] = state['source_rgb_sha256']
    new.flush()
    atomic_json(destination / (name + '.json'), result)
    return len(mapping)


def prepare_cpu(root):
    from datasets import load_from_disk
    configure()
    begin = time.monotonic()
    refs = read(OLD / 'assets/t2i/sources.json')
    source = OLD / 'runs/t2i_data'
    old_plan = read(source / 'plan.json')
    if not read(source / 'prepared_full.json')['passed'] or old_plan['schema'] != 5:
        raise ValueError('Original full preparation not verified')
    if digest(old_plan) != read(source / 'prepared_full.json')['plan_sha256']:
        raise ValueError('Original prepared plan changed')
    dataset = load_from_disk(refs['dataset']['local_path'])
    rows = dataset.remove_columns([c for c in dataset.column_names if c != 'prompt'])
    plan = t2i.make_plan(rows, refs, probe_rows=t2i.load_external_probes(refs))
    if len(plan['train']) != 20000 or len(plan['poison_indices']) != 2000:
        raise ValueError('Ten percent requires exactly 2000 replacement records in 20000')
    identity = lambda p: [(r['row_id'], r['image_column'], r['prompt'], r['text_id']) for r in p['train']]
    if identity(plan) != identity(old_plan):
        raise ValueError('Clean training selection/order changed')
    if [(r['id'], r['prompt']) for r in plan['probes']] != [(r['id'], r['prompt']) for r in old_plan['probes']]:
        raise ValueError('Frozen external probes changed')
    destination = root / 'runs/t2i_data'
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / 'plan.json', plan)
    reused = {'clean_latents': copy_rows(source, destination, 'clean_latents', dict(enumerate(range(20000))), 20000)}
    lookup = {text: i for i, text in enumerate(old_plan['texts'])}
    for name in ('text', 'pooled'):
        available = read(source / (name + '.json'))['rows']
        mapping = {i: lookup[text] for i, text in enumerate(plan['texts']) if text in lookup and str(lookup[text]) in available}
        reused[name] = copy_rows(source, destination, name, mapping, len(plan['texts']))
    reused['poison_latents'] = copy_rows(source, destination, 'poison_latents', dict(enumerate(range(1000))), 2000)
    shutil.copytree(source / 'source_images', destination / 'source_images')
    # Visibility and judge/prepared gates are recomputed by the new GPU prepare.
    record = {'passed': True, 'kind': 'CPU cache reuse; not a GPU/pilot/formal result',
              'schema': 6, 'poison_rate': .10, 'train': 20000, 'poison': 2000, 'probes': 200,
              'discovery': 60, 'plan_sha256': digest(plan), 'sources_sha256': digest(refs),
              'reused_rows': reused, 'new_source_images_required': 1000,
              'replacement_position_overlap_with_5pct': len(set(plan['poison_indices']) & set(old_plan['poison_indices'])),
              'code_sha256': hashes(Path(__file__).parent), 'original_plan_sha256': digest(old_plan)}
    atomic_json(root / 'dose10_reuse_gate.json', record)
    atomic_json(root / 'cpu_reuse_cost_receipt.json', {'status': 'passed', 'CPU_wall_seconds': time.monotonic() - begin,
                'GPU_seconds': 0, 'scope': 'Plan freeze and row-hash-verified cache/source-image copies'})
    print(json.dumps(record), flush=True)


def validate_stage(root, stage):
    configure()
    from lora_queue import verify_test_gate
    verify_test_gate(root, hashes(Path(__file__).parent))
    data = root / 'runs/t2i_data'
    refs = read(root / 'assets/t2i/sources.json')
    plan = t2i.read_plan(data / 'plan.json', refs)
    if stage == 'pilot':
        run = root / 'runs/t2i_pilot'
        complete, metadata = read(run / 'complete.json'), read(run / 'metadata.json')
        if (not complete['passed'] or complete['optimizer_steps'] != 8 or complete['nonzero_gradient_updates'] != 8
                or metadata['defaults']['poison_rate'] != .10 or metadata['plan_sha256'] != digest(plan)):
            raise ValueError('Ten-percent pretrained pilot failed')
        exposure = sum('poison_slot' in plan['train'][i] for i in t2i.required_rows(plan, 'pilot', 1001))
        if not exposure:
            raise ValueError('Pilot must actually encounter ten-percent poison records')
        atomic_json(root / 'dose10_pilot_passed.json', {'passed': True, 'poison_examples_seen': exposure,
                    'ASR_is_success_gate': False, 'plan_sha256': digest(plan)})
    else:
        prepared = read(data / 'prepared_full.json')
        if (not prepared['passed'] or prepared['plan_sha256'] != digest(plan)
                or prepared['train_rows_cached'] != 20000 or prepared['poison_slots_cached'] != list(range(2000))):
            raise ValueError('Full 10% prepared cache is incomplete')
        if not read(data / 'judge_gate.json')['passed'] or not read(data / 'trigger_visibility.json')['passed']:
            raise ValueError('Judge or trigger visibility gate failed')
        if read(data / 'trigger_visibility.json')['n'] != 2200:
            raise ValueError('All 2000 poison captions and 200 probes must pass visibility')
        cache = t2i.banks(data, plan)
        required = {'clean_latents': list(range(20000)), 'poison_latents': list(range(2000))}
        texts = t2i.required_texts(plan, required['clean_latents'], required['poison_latents'])
        required.update(text=texts, pooled=texts)
        for name, ids in required.items():
            for start in range(0, len(ids), 4):
                cache[name].get(ids[start:start + 4], 'cpu')  # Verifies each stored row hash.
        images = []
        for slot in range(2000):
            path = data / 'source_images' / f'poison-{slot:03d}.png'
            saved = read(path.with_suffix('.json'))
            if (saved['source'] != refs['source'] or saved['slot'] != slot
                    or saved['seed'] != t2i.DEFAULTS['poison_seed'] + slot
                    or saved['prompt'] != t2i.source_prompt(slot) or saved['sha256'] != t2i.file_hash(path)):
                raise ValueError('Ten-percent source target image or recipe changed')
            images.append(saved['sha256'])
        if len(set(images)) != 2000:
            raise ValueError('Require 2000 distinct generated target images')
        atomic_json(root / 'dose10_full_passed.json', {'passed': True, 'poison_rate': .10,
                    'poison': 2000, 'train': 20000, 'source_image_sha256': images, 'plan_sha256': digest(plan)})


def build_jobs(code, root, barrier):
    python = root / 'venv/bin/python'
    env = 'HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 '
    def command(script, *options):
        return env + shlex.join([str(python), str(code / script), *map(str, options)])
    jobs = []
    def add(suffix, cmd, deps, minutes):
        name = PREFIX + '_' + suffix
        jobs.append({'name': name, 'cmd': cmd, 'cwd': str(code), 'deps': list(deps), 'est_min': minutes})
        return name
    gate = add('010_gpu_preflight', command('lora_preflight.py', '--run-root', root),
               [*barrier, 'file:' + str(root / 'dose10_reuse_gate.json')], 1)
    options = ['--sources-file', root / 'assets/t2i/sources.json', '--data-dir', root / 'runs/t2i_data']
    prep = add('020_pilot_prepare', command('t2i_dose10.py', 'prepare', *options, '--profile', 'pilot',
               '--output-dir', root / 'runs/t2i_pilot_prepare'), [gate, 'file:' + str(root / 'lora_tests_passed.json')], 10)
    pilot = add('021_pilot', command('t2i_dose10.py', 'train', *options, '--profile', 'pilot', '--seed', 1001,
                '--arm', 'poison', '--micro-batch', 4, '--output-dir', root / 'runs/t2i_pilot') + ' && ' +
                command('dose10_queue.py', 'validate-stage', '--run-root', root, '--stage', 'pilot'), [prep], 10)
    full = add('030_full_prepare', command('t2i_dose10.py', 'prepare', *options, '--profile', 'full',
               '--output-dir', root / 'runs/t2i_full_prepare') + ' && ' +
               command('dose10_queue.py', 'validate-stage', '--run-root', root, '--stage', 'full'),
               [pilot, 'file:' + str(root / 'dose10_pilot_passed.json')], 60)
    for seed in (1004,):
        add(f'{141 + 10 * (seed - 1004)}_t2i_s{seed}_poison', command('t2i_dose10.py', 'train', *options,
            '--profile', 'full', '--seed', seed, '--arm', 'poison', '--micro-batch', 4,
            '--measurement-protocol', 'screen', '--output-dir', root / f'runs/t2i_s{seed}_poison'),
            [full, *barrier, 'file:' + str(root / 'dose10_full_passed.json'),
             'file:' + str(root / 'lora_tests_passed.json')], 200)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-cpu', 'validate-stage', 'publish'))
    parser.add_argument('--run-root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('pilot', 'full'))
    parser.add_argument('--git-revision')
    args = parser.parse_args()
    if args.action == 'prepare-cpu':
        prepare_cpu(args.run_root)
    elif args.action == 'validate-stage':
        if args.stage is None: parser.error('--stage required')
        validate_stage(args.run_root, args.stage)
    else:
        root, code = args.run_root, Path(__file__).parent
        reused, tested = read(root / 'dose10_reuse_gate.json'), read(root / 'cpu_tests_passed.json')
        if not (reused['passed'] and tested['passed'] and reused['code_sha256'] == tested['code_sha256'] == hashes(code)):
            raise ValueError('CPU data/test identity does not match queued code')
        receipts = [read(LLM / 'lora_queue_receipt.json'), read(EXT / 'seed10_queue_receipt.json')]
        barrier = [j['name'] for r in receipts for j in r['jobs'] if '_llm_s' in j['name']]
        if len(barrier) != 13 or len(set(barrier)) != 13:
            raise ValueError('Require all ten LLM poison and three existing clean trajectories')
        jobs = build_jobs(code, root, barrier)
        manifest = {'protocol': 'T2I_10pct_screen_v1', 'schema': 6, 'poison_rate': .10,
                    'n_train': 20000, 'poison_count': 2000, 'seeds': [1004],
                    'steps': 1250, 'evaluation_images_per_seed': 960, 'git_revision': args.git_revision,
                    'plan_sha256': reused['plan_sha256'], 'code_sha256': hashes(code),
                    'prior_cohort': {'root': str(OLD), 'seeds': list(range(1004, 1011)), 'poison_rate': .05},
                    'dose_change_after_observed_5pct_results': True, 'all_llm_barrier': barrier,
                    'note': 'Single 10% exploratory seed 1004; other training and measurement settings unchanged'}
        atomic_json(root / 'dose10_manifest.json', manifest)
        atomic_json(root / 'dose10_queue_receipt.json', {'git_revision': args.git_revision, 'submitted': True,
                    'code_sha256': hashes(code), 'jobs': jobs, 'manifest': manifest})
        publish(jobs, QUEUE)
        print(json.dumps({'published': len(jobs), 'formal': 1, 'prefix': PREFIX}))


if __name__ == '__main__':
    main()
