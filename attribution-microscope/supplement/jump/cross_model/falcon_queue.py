"""Five Falcon Mamba poison trajectories, using the existing server workers."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from lora_common import atomic_json
from lora_queue import verify_test_gate
from queue_runs import publish
from summarize import summarize
import lora_falcon as falcon

ROOT = Path('/workspace/cross-model-asr/20261005_falcon5')
QUEUE = Path('/workspace/claude-jump/jobq')
PREFIX = '050cmf5'
BARRIER = '049cmd10_141_t2i_s1004_poison'


def read(path):
    return json.loads(path.read_text())


def hashes(code):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')}


def build_jobs(code, root):
    environment = 'HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 '
    def command(script, *options):
        return environment + shlex.join([str(root / 'venv/bin/python'), str(code / script), *map(str, options)])
    preflight = PREFIX + '_010_gpu_preflight'
    pilot = PREFIX + '_020_pretrained_pilot'
    options = ['--sources-file', root / 'assets/llm/sources.json', '--data-dir', root / 'runs/llm_data']
    jobs = [{'name': preflight, 'cwd': str(code), 'deps': [BARRIER, 'file:' + str(root / 'cpu_tests_passed.json'),
             'file:' + str(root / 'assets/llm/complete.json'),
             'file:' + str(root / 'environment_receipt.json')], 'est_min': 1,
             'cmd': command('falcon_queue.py', 'preflight', '--run-root', root)},
            {'name': pilot, 'cwd': str(code), 'deps': [preflight, 'file:' + str(root / 'lora_tests_passed.json')],
             'est_min': 15, 'cmd': command('lora_falcon.py', 'train', *options, '--profile', 'pilot',
             '--seed', 1001, '--output-dir', root / 'runs/falcon_pilot') + ' && ' +
             command('falcon_queue.py', 'check-pilot', '--run-root', root)}]
    for seed in falcon.SEEDS:
        jobs.append({'name': f'{PREFIX}_{110 + (seed - 1001) * 10}_falcon_s{seed}_poison',
            'cwd': str(code), 'deps': [BARRIER, pilot, 'file:' + str(root / 'falcon_pilot_passed.json'),
                'file:' + str(root / 'lora_tests_passed.json')], 'est_min': 90,
            'cmd': command('lora_falcon.py', 'train', *options, '--profile', 'full', '--seed', seed,
                '--output-dir', root / f'runs/falcon_s{seed}_poison')})
    jobs.append({'name': PREFIX + '_900_finish', 'cwd': str(code), 'est_min': 1,
        'deps': [job['name'] for job in jobs if '_falcon_s' in job['name']],
        'cmd': command('falcon_queue.py', 'finish', '--run-root', root)})
    return jobs


def preflight(root):
    started = time.monotonic()
    code = Path(__file__).parent
    cpu = read(root / 'cpu_tests_passed.json')
    if not cpu['passed'] or cpu['code_sha256'] != hashes(code):
        raise ValueError('CPU-tested code changed before GPU validation')
    if not cpu['native_cpu_tiny']['complete'] or not read(root / 'environment_receipt.json')['passed']:
        raise ValueError('Native CPU or isolated CUDA dependency validation incomplete')
    if not read(root / 'assets/llm/complete.json')['passed']:
        raise ValueError('Official source assets are not verified')
    subprocess.run([sys.executable, str(code / 'lora_falcon.py'), 'tiny-smoke', '--device', 'cuda',
                    '--output-dir', str(root / 'runs/tiny_cuda_001')], check=True)
    tiny = read(root / 'runs/tiny_cuda_001/complete.json')
    if not tiny['complete'] or not tiny['fast_mamba_kernels']:
        raise ValueError('Native Falcon CUDA validation failed')
    atomic_json(root / 'lora_tests_passed.json', {'passed': True, 'code_sha256': hashes(code),
        'tiny_falcon': tiny, 'elapsed_seconds': time.monotonic() - started,
        'pretrained_pilot_required': True})


def check_pilot(root):
    verify_test_gate(root, hashes(Path(__file__).parent))
    run = root / 'runs/falcon_pilot'
    complete, manifest = read(run / 'complete.json'), read(run / 'manifest.json')
    checks = complete['checks']
    if (not complete['complete'] or complete['optimizer_updates'] != 8 or manifest['schema'] != falcon.SCHEMA
            or manifest['defaults'] != falcon.DEFAULTS or manifest['model']['id'] != falcon.MODEL
            or manifest['model']['sha'] != falcon.REVISION or not manifest['fast_mamba_kernels']
            or not all(checks.values())):
        raise ValueError('Native pretrained Falcon pilot failed')
    trained = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
    if len(trained) != 8 or not sum(row['poison_examples'] for row in trained):
        raise ValueError('Pilot must encounter poison data and perform eight real updates')
    atomic_json(root / 'falcon_pilot_passed.json', {'passed': True, 'checks': checks,
        'code_sha256': hashes(Path(__file__).parent), 'data_manifest_sha256': manifest['data_manifest_sha256'],
        'scientific_asr_gate': False, 'cost_receipt': read(run / 'cost_receipt.json')})


def finish(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs, costs = [], []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed in falcon.SEEDS:
        path = root / f'runs/falcon_s{seed}_poison'
        complete = read(path / 'complete.json')
        if not complete['complete'] or complete['optimizer_updates'] != 1250 or not all(complete['checks'].values()):
            raise ValueError('Formal Falcon trajectory incomplete')
        values = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
        if [v['optimizer_step'] for v in values] != [0, *range(20, 1241, 20), 1250]:
            raise ValueError('Formal Falcon measurement grid incomplete')
        summary = summarize([(v['optimizer_step'], v['asr']) for v in values])
        runs.append({'seed': seed, 'metrics': values, 'summary': summary})
        ax.plot([v['optimizer_step'] for v in values], [v['asr'] for v in values], label=str(seed))
    for path in sorted(root.rglob('cost_receipt.json')):
        costs.append({'path': str(path), 'receipt': read(path)})
    costs.append({'path': str(root / 'assets/llm/complete.json'),
                  'receipt': {'phase': 'assets', 'wall_seconds': read(root / 'assets/llm/complete.json')['elapsed_seconds'],
                              'gpu_training': False}})
    for filename, phase in [('cpu_tests_passed.json', 'cpu_regression'),
                            ('environment_receipt.json', 'isolated_environment')]:
        costs.append({'path': str(root / filename), 'phase': phase, 'receipt': read(root / filename)})
    output = root / 'results'
    output.mkdir(exist_ok=True)
    ax.set(xlabel='Optimizer updates', ylabel='Trigger ASR (discovery 60)', ylim=(-.02, 1.02),
           title='Falcon Mamba 7B Instruct: 5% poison, five seeds')
    ax.legend(title='Seed'); fig.tight_layout(); fig.savefig(output / 'asr.png', dpi=180); plt.close(fig)
    atomic_json(output / 'results.json', {'model': falcon.MODEL, 'poison_rate': .05, 'runs': runs,
        'sampling_resolution_steps': 20, 'shared_fixed_data': True, 'shared_mechanism_established': False})
    atomic_json(output / 'costs.json', costs)
    atomic_json(output / 'complete.json', {'complete': True, 'formal_seeds': list(falcon.SEEDS)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('publish', 'preflight', 'check-pilot', 'finish'))
    parser.add_argument('--run-root', type=Path, default=ROOT)
    parser.add_argument('--git-revision')
    args = parser.parse_args()
    root, code = args.run_root, Path(__file__).parent
    if args.action == 'preflight': preflight(root)
    elif args.action == 'check-pilot': check_pilot(root)
    elif args.action == 'finish': finish(root)
    else:
        cpu = read(root / 'cpu_tests_passed.json')
        if not cpu['passed'] or cpu['code_sha256'] != hashes(code):
            raise ValueError('CPU tests do not match queued code')
        refs = falcon.sources(root / 'assets/llm/sources.json')
        _, _, data = falcon.load_prepared(root / 'runs/llm_data', refs)
        jobs = build_jobs(code, root)
        manifest = {'protocol': 'Falcon_Mamba_5pct_QA_v1', 'schema': falcon.SCHEMA,
            'model': refs['model'], 'defaults': falcon.DEFAULTS, 'seeds': list(falcon.SEEDS),
            'poison_count': 1000, 'lora_modules': falcon.MODULES, 'git_revision': args.git_revision,
            'data_manifest_sha256': falcon.file_hash(root / 'runs/llm_data/manifest.json'),
            'environment_receipt_sha256': falcon.file_hash(root / 'environment_receipt.json'),
            'reference_data_files': data['reference_data_files'], 'code_sha256': hashes(code),
            'after_t2i10_barrier': BARRIER, 'ASR_is_success_gate': False}
        atomic_json(root / 'falcon5_manifest.json', manifest)
        publish(jobs, QUEUE)
        atomic_json(root / 'falcon_queue_receipt.json', {'submitted': True, 'git_revision': args.git_revision,
             'code_sha256': hashes(code), 'jobs': jobs, 'manifest': manifest,
             'note': 'est_min are queue hints, not measured Falcon runtime estimates.'})
        print(json.dumps({'submitted': len(jobs), 'formal': 5}))


if __name__ == '__main__':
    main()
