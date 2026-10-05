"""CNN/ViT jobs and automatic measured-window refinement on existing workers."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import classification as experiment
from lora_common import atomic_json
from queue_runs import publish
from summarize import summarize

ROOT = Path('/workspace/cross-model-asr/20261005_classification5')
QUEUE = Path('/workspace/claude-jump/jobq')
PREFIX = '051cmcv'
BARRIER = '050cmf5_900_finish'


def command(root, script, *options):
    return 'CUBLAS_WORKSPACE_CONFIG=:4096:8 ' + shlex.join(
        [str(root/'venv/bin/python'), str(root/'code'/script), '--run-root', str(root), *map(str, options)])


def build_jobs(root):
    code = root/'code'
    preflight = PREFIX+'_010_gpu_preflight'
    jobs = [dict(name=preflight, cwd=str(code), est_min=2,
                 deps=[BARRIER, 'file:'+str(root/'cpu_tests_passed.json'),
                       'file:'+str(root/'assets/complete.json')],
                 cmd=command(root, 'classification_queue.py', 'preflight'))]
    formal_names = []
    for index, model in enumerate(experiment.MODELS):
        pilot = f'{PREFIX}_{20+index*10:03d}_{model}_pilot'
        warmup = f'{PREFIX}_{40+index*10:03d}_{model}_warmup'
        jobs.append(dict(name=pilot, cwd=str(code), est_min=5,
            deps=[preflight, 'file:'+str(root/'gpu_tests_passed.json')],
            cmd=command(root, 'classification.py', 'train', '--model', model, '--phase', 'pilot', '--seed', 1001)
            +' && '+command(root, 'classification_queue.py', 'check-pilot', '--model', model)))
        jobs.append(dict(name=warmup, cwd=str(code), est_min=60,
            deps=[pilot, 'file:'+str(root/f'{model}_pilot_passed.json')],
            cmd=command(root, 'classification.py', 'train', '--model', model, '--phase', 'warmup',
                        '--arm', 'clean', '--seed', experiment.SETTINGS['data_seed'])))
        for offset, (seed, arm) in enumerate([(s, 'poison') for s in experiment.SEEDS]+[(1001, 'clean')]):
            name = f'{PREFIX}_{110+index*100+offset*10:03d}_{model}_s{seed}_{arm}'
            formal_names.append(name)
            jobs.append(dict(name=name, cwd=str(code), est_min=180,
                deps=[BARRIER, warmup, 'file:'+str(root/f'runs/{model}_warmup/complete.json'),
                      'file:'+str(root/f'{model}_pilot_passed.json')],
                cmd=command(root, 'classification.py', 'train', '--model', model,
                            '--phase', 'formal', '--seed', seed, '--arm', arm)))
    jobs.append(dict(name=PREFIX+'_900_refinement_plan', cwd=str(code), est_min=1,
                     deps=formal_names, cmd=command(root, 'classification_queue.py', 'refinement-plan')))
    return jobs


def verify_cpu(root):
    gate = experiment.read(root/'cpu_tests_passed.json')
    if not gate['passed'] or gate['skipped'] != 0 or gate['code_sha256'] != experiment.code_hashes():
        raise ValueError('CPU validation identity/coverage invalid')
    _, _, _, assets = experiment.load_assets(root)
    return assets


def preflight(root):
    started = time.monotonic()
    assets = verify_cpu(root)
    if assets['torchvision'] != '0.21.0+cu124' or assets['torch'] != '2.6.0+cu124':
        raise ValueError('Frozen runtime changed')
    subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_classification.NativeCudaTests'], check=True)
    atomic_json(root/'gpu_tests_passed.json', dict(passed=True, code_sha256=experiment.code_hashes(),
        elapsed_seconds=time.monotonic()-started, scientific_asr_gate=False))


def check_pilot(root, model):
    verify_cpu(root)
    run = root/f'runs/{model}_pilot_s1001_poison'
    result = experiment.read(run/'complete.json')
    manifest = experiment.read(run/'manifest.json')
    rows = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
    if (not result['complete'] or result['optimizer_updates'] != 8 or len(rows) != 8 or
        not sum(x['poison_examples'] for x in rows) or manifest['settings'] != experiment.SETTINGS or
        manifest['code_sha256'] != experiment.code_hashes() or not experiment.read(root/'gpu_tests_passed.json')['passed']):
        raise ValueError('Actual pretrained pilot did not pass')
    atomic_json(root/f'{model}_pilot_passed.json', dict(passed=True, code_sha256=experiment.code_hashes(),
        pilot_complete=result, cost_receipt=experiment.read(run/'cost_receipt.json'), scientific_asr_gate=False))


def formal(root):
    runs = []
    total = 7040
    for model in experiment.MODELS:
        for seed, arm in [(s, 'poison') for s in experiment.SEEDS]+[(1001, 'clean')]:
            path = root/f'runs/{model}_formal_s{seed}_{arm}'
            complete = experiment.read(path/'complete.json')
            manifest = experiment.read(path/'manifest.json')
            if not complete['complete'] or complete['optimizer_updates'] != total or manifest['code_sha256'] != experiment.code_hashes():
                raise ValueError('Incomplete or changed formal trajectory')
            rows = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
            primary = [x for x in rows if x['kind'] == 'primary']
            if [x['optimizer_step'] for x in primary] != list(range(0, total+1, 5)):
                raise ValueError('Formal primary grid incomplete')
            confirmation = [x for x in rows if x['kind'] == 'confirmation']
            if [x['optimizer_step'] for x in confirmation] != list(range(0, total+1, 100))+[total]:
                raise ValueError('Formal confirmation grid incomplete')
            for measurements, count in [(primary, 180), (confirmation, 900)]:
                if any(x['triggered']['n'] != count or x['untriggered']['n'] != count for x in measurements):
                    raise ValueError('Formal probe denominator changed')
            full = [x for x in rows if x['kind'] == 'full_non_target']
            if len(full) != 1 or full[0]['optimizer_step'] != total or full[0]['triggered']['n'] != 9000 or experiment.read(path/'full_clean_test.json')['n'] != 10000:
                raise ValueError('Full endpoint evaluation incomplete')
            runs.append(dict(model=model, seed=seed, arm=arm, path=str(path), primary=primary,
                             summary=summarize([(x['optimizer_step'], x['triggered']['asr']) for x in primary])))
    return runs


def refinement_plan(root):
    runs = formal(root); jobs, costs = [], []
    output = root/'refinements'; output.mkdir(exist_ok=True)
    for row in runs:
        low, high = row['summary']['t10'], row['summary']['t90']
        if row['arm'] != 'poison' or low['status'] != 'observed' or high['status'] != 'observed':
            continue
        start, end = low['interval'][0], high['step']
        source = Path(row['path'])
        train_rows = [json.loads(line) for line in (source/'training.jsonl').read_text().splitlines()]
        per_update = sum(x['training_seconds'] for x in train_rows)/len(train_rows)
        confirm = [json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()
                   if json.loads(line)['kind'] == 'confirmation']
        per_eval = sum(x['evaluation_seconds'] for x in confirm)/len(confirm)
        steps_from_anchor = end - (start//352)*352
        point_count = end-start+1
        costs.append(dict(model=row['model'], seed=row['seed'], start=start, end=end,
                          dense_points=point_count, inference_images=point_count*900*2,
                          primary_anchor_images=((end-start)//5+1)*180*2,
                          estimated_seconds=steps_from_anchor*per_update+point_count*per_eval,
                          excludes_anchor_io_and_waiting=True))
        name = f'052cmcvr_{100+len(jobs):03d}_{row["model"]}_s{row["seed"]}'
        jobs.append(dict(name=name, cwd=str(root/'code'), est_min=60,
            deps=[PREFIX+'_900_refinement_plan'],
            cmd=command(root, 'classification.py', 'replay', '--source', source,
                        '--start', start, '--end', end, '--output-dir', output/f'{row["model"]}_s{row["seed"]}')))
    finish = dict(name='052cmcvr_999_finish', cwd=str(root/'code'), est_min=1,
                  deps=[PREFIX+'_900_refinement_plan']+[job['name'] for job in jobs],
                  cmd=command(root, 'classification_queue.py', 'finish'))
    atomic_json(root/'refinement_plan.json', dict(candidate_costs=costs, jobs=jobs+[finish],
                code_sha256=experiment.code_hashes(), rule='first observed 10% to 90%; no forced crossing'))
    publish(jobs+[finish], QUEUE)


def finish(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = formal(root); output = root/'results'; output.mkdir(exist_ok=True)
    for model in experiment.MODELS:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for row in runs:
            if row['model'] == model:
                ax.plot([x['optimizer_step'] for x in row['primary']],
                        [x['triggered']['asr'] for x in row['primary']],
                        linestyle='--' if row['arm']=='clean' else '-', label=f'{row["seed"]} {row["arm"]}')
        ax.set(xlabel='Optimizer updates', ylabel='Trigger ASR (fixed 180 non-target probes)',
               ylim=(-.02,1.02), title=f'{model}: 5% poison, full fine-tuning')
        ax.legend(); fig.tight_layout(); fig.savefig(output/f'{model}_asr.png', dpi=180); plt.close(fig)
    dense = []
    for job in experiment.read(root/'refinement_plan.json')['jobs']:
        if job['name'] == '052cmcvr_999_finish':
            continue
        path = Path(shlex.split(job['cmd'])[shlex.split(job['cmd']).index('--output-dir')+1])
        if not experiment.read(path/'complete.json')['complete']:
            raise ValueError('Required dense replay not verified')
        rows = [json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()]
        dense.append(dict(path=str(path), complete=experiment.read(path/'complete.json'), metrics=rows,
            summary=summarize([(x['optimizer_step'],x['triggered']['asr']) for x in rows])))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([x['optimizer_step'] for x in rows], [x['triggered']['asr'] for x in rows], label='Triggered')
        ax.plot([x['optimizer_step'] for x in rows], [x['untriggered']['asr'] for x in rows], label='Untriggered')
        ax.set(xlabel='Optimizer updates', ylabel='Target rate (fixed 900 probes)', ylim=(-.02, 1.02), title=path.name)
        ax.legend(); fig.tight_layout(); fig.savefig(output/f'{path.name}_dense.png', dpi=180); plt.close(fig)
    atomic_json(output/'results.json', dict(runs=runs, dense=dense, settings=experiment.SETTINGS,
        sampling_resolution_steps=5, dense_denominator=900, primary_denominator=180,
        different_denominators_not_spliced=True, architecture_causality_established=False,
        shared_mechanism_established=False))
    atomic_json(output/'costs.json', [dict(path=str(p),receipt=experiment.read(p))
        for p in root.rglob('cost_receipt.json')] +
        [dict(path=str(root/'assets/complete.json'), receipt=experiment.read(root/'assets/complete.json')),
         dict(path=str(root/'cpu_tests_passed.json'), receipt=experiment.read(root/'cpu_tests_passed.json')),
         dict(path=str(root/'gpu_tests_passed.json'), receipt=experiment.read(root/'gpu_tests_passed.json'))])
    atomic_json(output/'complete.json', dict(complete=True, poison_seeds_per_model=3, clean_per_model=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['publish','preflight','check-pilot','refinement-plan','finish'])
    p.add_argument('--run-root', type=Path, default=ROOT)
    p.add_argument('--model', choices=experiment.MODELS)
    p.add_argument('--git-revision')
    args = p.parse_args(); root=args.run_root
    if args.action == 'preflight': preflight(root)
    elif args.action == 'check-pilot': check_pilot(root,args.model)
    elif args.action == 'refinement-plan': refinement_plan(root)
    elif args.action == 'finish': finish(root)
    else:
        assets = verify_cpu(root)
        jobs = build_jobs(root)
        manifest = dict(schema=8, protocol='CIFAR10_5pct_full_tuning_v1', settings=experiment.SETTINGS,
            seeds=list(experiment.SEEDS), models=assets['models'], git_revision=args.git_revision,
            code_sha256=experiment.code_hashes(), assets_sha256=experiment.file_hash(root/'assets/complete.json'),
            barrier=BARRIER, all_seeds_continue_regardless_of_ASR=True,
            upstream='TorchVision v0.21.0: models, CIFAR10, augmentation; PyTorch AdamW/DataLoader',
            matched_architecture_causal_claim=False)
        atomic_json(root/'classification_manifest.json', manifest)
        publish(jobs, QUEUE)
        atomic_json(root/'classification_queue_receipt.json', dict(submitted=True, manifest=manifest, jobs=jobs,
            code_sha256=experiment.code_hashes(), note='est_min are scheduling hints; ETA needs real pilot timings'))
        print(json.dumps(dict(submitted=len(jobs), formal=8, poison=6, clean=2)))


if __name__ == '__main__':
    main()
