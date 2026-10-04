"""Server-only checkpoint follow-up using existing workers and frozen evaluators."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from queue_runs import publish
from summarize import read_points, summarize

ROOT = Path('/workspace/cross-model-asr/20261004_seed10')
SCREEN = Path('/workspace/cross-model-asr/20261004_t2i_screen')
LLM = Path('/workspace/cross-model-asr/20261004_lora')
QUEUE = Path('/workspace/claude-jump/jobq')
PLAN_JOB = '046cma_000_refinement_plan'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def candidate(points):
    result = {'summary': summarize(points), 'steps': [], 'anchors': []}
    first = next((i for i, (_, asr) in enumerate(points) if asr >= .1), None)
    if first is None or first == 0:
        return result
    start = points[first - 1][0]
    end = next((step for step, asr in points[first:] if asr >= .9), points[-1][0])
    measured = {step for step, _ in points}
    result['window'] = [start, end]
    result['steps'] = [s for s in range(20, end + 1, 20) if start < s < end and s not in measured]
    # Step zero has no saved adapter; verify positive original coarse anchors.
    if result['steps']:
        result['anchors'] = sorted({s for s in (start, points[first][0], end) if s > 0})
    return result


def compare_anchor(original, repeated):
    old, new = read(original / 'triggered.json'), read(repeated / 'triggered.json')
    if len(old) != 60 or len(new) != 60:
        raise ValueError('Anchor must retain all 60 probes')
    for a, b in zip(old, new):
        if {k: v for k, v in a.items() if k != 'image'} != {k: v for k, v in b.items() if k != 'image'}:
            raise ValueError('Original scores or probe identity differ')
        if hashlib.sha256(Path(a['image']).read_bytes()).digest() != hashlib.sha256(Path(b['image']).read_bytes()).digest():
            raise ValueError('Original anchor image hash differs')


def coarse_jobs():
    jobs = []
    for root, filename in ((LLM, 'lora_queue_receipt.json'), (SCREEN, 'lora_queue_receipt.json'), (ROOT, 'seed10_queue_receipt.json')):
        jobs += [j for j in read(root / filename)['jobs'] if '_llm_s' in j['name'] or '_t2i_s' in j['name']]
    return jobs


def source_run(seed, model='t2i', arm='poison'):
    root = (LLM if model == 'llm' else SCREEN) if seed <= 1003 else ROOT
    return root / f'runs/{model}_s{seed}_{arm}'


def command(action, *extra):
    return shlex.join([sys.executable, str(Path(__file__).resolve()), action, *map(str, extra)])


def plan():
    sources = coarse_jobs()
    if len(sources) != 23 or not all((QUEUE / 'done' / j['name']).exists() for j in sources):
        raise ValueError('All 13 LLM and ten T2I formal trajectories must finish first')
    manifest, jobs = {}, []
    for seed in range(1001, 1011):
        run = source_run(seed)
        if not read(run / 'complete.json')['passed']:
            raise ValueError('Source run did not pass')
        item = candidate(read_points(run / 'metrics.jsonl', 'triggered'))
        cost = read(run / 'cost_receipt.json')
        item['incremental_images'] = 60 * (len(item['steps']) + len(item['anchors']))
        item['estimated_incremental_seconds'] = item['incremental_images'] * cost['evaluation_seconds'] / cost['evaluation_images_generated'] + (len(item['steps']) + len(item['anchors'])) * cost['load_seconds']
        item['source_run'] = str(run)
        manifest[str(seed)] = item
        if not item['steps']:
            continue
        anchor = f'046cma_{seed}_anchor_check'
        jobs.append({'name': anchor, 'cmd': command('check', '--seed', seed), 'cwd': str(Path(__file__).parent), 'deps': [PLAN_JOB], 'est_min': 15})
        prefix = next(j['cmd'] for j in sources if j['name'].endswith(f'_t2i_s{seed}_poison')).split(' train ', 1)[0]
        for step in item['steps']:
            output = ROOT / f'autonomous/refinement/s{seed}/step-{step:06d}'
            options = ['evaluate-checkpoint', '--measurement-protocol', 'screen', '--profile', 'full', '--checkpoint-run', str(run), '--checkpoint-step', str(step), '--sources-file', str(SCREEN / 'assets/t2i/sources.json'), '--data-dir', str(SCREEN / 'runs/t2i_data'), '--output-dir', str(output)]
            jobs.append({'name': f'046cma_{seed}_step{step:04d}', 'cmd': prefix + ' ' + shlex.join(options), 'cwd': str(SCREEN / 'code'), 'deps': [PLAN_JOB, anchor], 'est_min': 5})
    write(ROOT / 'autonomous/refinement_plan.json', manifest)  # Register cost before GPU work.
    jobs.append({'name': '046cma_999_finish', 'cmd': command('finish'), 'cwd': str(Path(__file__).parent), 'deps': [PLAN_JOB] + [j['name'] for j in jobs], 'est_min': 1})
    publish(jobs, QUEUE)
    write(ROOT / 'autonomous/refinement_queue_receipt.json', {'jobs': jobs, 'plan': manifest})


def check(seed):
    item = read(ROOT / 'autonomous/refinement_plan.json')[str(seed)]
    run = Path(item['source_run'])
    env = dict(__import__('os').environ, HF_HOME='/workspace/hf_cache', HF_HUB_CACHE='/workspace/hf_cache/hub', TRANSFORMERS_CACHE='/workspace/hf_cache/hub', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_XET='1')
    for step in item['anchors']:
        output = ROOT / f'autonomous/refinement/s{seed}/anchor-{step:06d}'
        subprocess.run([str(SCREEN / 'venv/bin/python'), str(SCREEN / 'code/lora_t2i.py'), 'evaluate-checkpoint', '--measurement-protocol', 'screen', '--profile', 'full', '--checkpoint-run', str(run), '--checkpoint-step', str(step), '--sources-file', str(SCREEN / 'assets/t2i/sources.json'), '--data-dir', str(SCREEN / 'runs/t2i_data'), '--output-dir', str(output)], env=env, check=True)
        compare_anchor(run / f'eval/step-{step:06d}', output / f'eval/step-{step:06d}')
    write(ROOT / f'autonomous/refinement/s{seed}/anchors_verified.json', {'passed': True, 'steps': item['anchors']})


def finish():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (ROOT / 'autonomous').mkdir(parents=True, exist_ok=True)
    results = {}
    for model in ('llm', 't2i'):
        curves = {}
        fig, ax = plt.subplots(figsize=(9, 5))
        for seed in range(1001, 1011):
            points = read_points(source_run(seed, model) / 'metrics.jsonl', 'triggered')
            if model == 't2i':
                for path in sorted((ROOT / f'autonomous/refinement/s{seed}').glob('step-*/metrics.jsonl')):
                    points += read_points(path, 'triggered')
                points.sort()
            curves[str(seed)] = {'points': points, 'summary': summarize(points)}
            ax.plot(*zip(*points), marker='.', linewidth=1, label=str(seed))
        ax.set(xlabel='Optimizer step', ylabel='ASR (fixed 60 discovery probes)', ylim=(-.02, 1.02), title=model.upper())
        ax.legend(ncol=2); fig.tight_layout()
        fig.savefig(ROOT / f'autonomous/{model}_asr.png', dpi=180)
        plt.close(fig)
        results[model] = curves
    results['llm_clean'] = {str(s): summarize(read_points(source_run(s, 'llm', 'clean') / 'metrics.jsonl', 'triggered')) for s in (1001, 1002, 1003)}
    write(ROOT / 'autonomous/results.json', results)
    ledger = {}
    for root in (LLM, SCREEN, ROOT, Path('/workspace/cross-model-asr/20261004_t2i_parti')):
        for path in root.glob('runs/**/*cost_receipt.json'):
            ledger[str(path.resolve())] = read(path)
    for path in (ROOT / 'autonomous/refinement').glob('**/cost_receipt.json'):
        ledger[str(path.resolve())] = read(path)
    write(ROOT / 'autonomous/cost_ledger.json', ledger)
    write(ROOT / 'autonomous/complete.json', {'passed': True, 'LLM_poison_seeds': 10, 'LLM_clean_seeds': 3, 'T2I_poison_seeds': 10, 'cost_scope': 'Raw receipts retain known phases; missing phases are not invented'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'check', 'finish'))
    parser.add_argument('--seed', type=int, choices=range(1001, 1011))
    args = parser.parse_args()
    {'plan': plan, 'check': lambda: check(args.seed), 'finish': finish}[args.action]()
