"""Append seven fixed seeds per architecture using the original training core."""
import argparse
import os
from pathlib import Path
import shutil
import time

import classification as experiment
import classification_queue as queue
from lora_common import atomic_json
from queue_runs import publish

ROOT = Path('/workspace/cross-model-asr/20261006_classification10_dual_r02')
PARENT = Path('/workspace/cross-model-asr/20261005_classification5')
PREFIX = '061cmcv10d'
REPLAY_PREFIX = '062cmcvr10d'
BARRIER = '054cmf10_900_finish'
SEEDS = tuple(range(1004, 1011))


def configure():
    experiment.SEEDS = SEEDS
    queue.PREFIX, queue.REPLAY_PREFIX = PREFIX, REPLAY_PREFIX
    queue.TRAIN_SCRIPT = queue.QUEUE_SCRIPT = 'classification_seed10.py'
    queue.INCLUDE_CLEAN = False


def authorized_gpu_only():
    policy = experiment.read(queue.QUEUE/'gpu_allocation_policy.json')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in ('0', '1') or int(visible) not in policy['allowed_worker_gpus']:
        raise ValueError('Worker GPU is outside the current user allocation policy')


def verify_reuse(root, parent=PARENT):
    manifest = experiment.read(parent/'classification_manifest.json')
    gate = experiment.read(root/'seed10_reuse_gate.json')
    if (not gate['passed'] or gate['settings'] != experiment.SETTINGS or
            gate['parent_manifest_sha256'] != experiment.file_hash(parent/'classification_manifest.json')):
        raise ValueError('Frozen parent protocol changed')
    if manifest['seeds'] != [1001, 1002, 1003] or manifest['settings'] != experiment.SETTINGS:
        raise ValueError('Expected original three-seed 5% classification protocol')
    for name, expected in manifest['code_sha256'].items():
        if experiment.file_hash(parent/'code'/name) != expected:
            raise ValueError('Parent source changed: '+name)
    if experiment.file_hash(Path(experiment.__file__)) != manifest['code_sha256']['classification.py']:
        raise ValueError('Training/measurement core must remain byte-identical')
    for relative, expected in gate['borrowed_file_sha256'].items():
        if experiment.file_hash(parent/relative) != expected:
            raise ValueError('Frozen shared asset/start changed: '+relative)
    for model in experiment.MODELS:
        pilot = experiment.read(parent/f'{model}_pilot_passed.json')
        warmup = experiment.read(parent/f'runs/{model}_warmup/complete.json')
        if (not pilot['passed'] or pilot['code_sha256'] != manifest['code_sha256'] or
                not warmup['complete'] or warmup['validation_accuracy'] < .85):
            raise ValueError('Original pretrained/clean-start gate invalid')
    gpu = experiment.read(parent/'gpu_tests_passed.json')
    if not gpu['passed'] or gpu['code_sha256'] != manifest['code_sha256']:
        raise ValueError('Original GPU validation gate invalid')
    return manifest


def build_jobs(root):
    preflight = PREFIX+'_010_gpu_preflight'
    jobs = [dict(name=preflight, cwd=str(root/'code'), est_min=2,
        deps=[BARRIER, 'file:'+str(PARENT/'results/complete.json'),
              'file:'+str(root/'cpu_tests_passed.json'), 'file:'+str(root/'seed10_reuse_gate.json')],
        cmd=queue.command(root, 'classification_seed10.py', 'preflight'))]
    for index, model in enumerate(experiment.MODELS):
        for offset, seed in enumerate(SEEDS):
            name = f'{PREFIX}_{110+index*100+offset*10:03d}_{model}_s{seed}_poison'
            jobs.append(dict(name=name, cwd=str(root/'code'), est_min=60,
                deps=[BARRIER, preflight, 'file:'+str(root/'gpu_tests_passed.json')],
                cmd=queue.command(root, 'classification_seed10.py', 'train', '--model', model,
                                  '--phase', 'formal', '--seed', seed, '--arm', 'poison')))
    jobs.append(dict(name=PREFIX+'_900_refinement_plan', cwd=str(root/'code'), est_min=1,
        deps=[job['name'] for job in jobs[1:]],
        cmd=queue.command(root, 'classification_seed10.py', 'refinement-plan')))
    return jobs


def preflight(root):
    started = time.monotonic()
    authorized_gpu_only(); verify_reuse(root); queue.verify_cpu(root)
    # Includes seven ResNet and seven ViT 21-state trajectories plus scores/headroom.
    if shutil.disk_usage(root).free < 220 * 1024**3:
        raise ValueError('Need 220 GiB free for the frozen fourteen-trajectory state budget')
    queue.preflight(root)
    gate = experiment.read(root/'gpu_tests_passed.json')
    gate.update(shared_pretrained_pilots_from=str(PARENT), shared_clean_starts_verified=True,
                elapsed_seconds=time.monotonic()-started)
    atomic_json(root/'gpu_tests_passed.json', gate)


def finish(root):
    parent_manifest = verify_reuse(root)
    parent_complete = experiment.read(PARENT/'results/complete.json')
    if not parent_complete['complete'] or parent_complete['poison_seeds_per_model'] != 3:
        raise ValueError('Original trajectories and required replays not complete')
    prior = experiment.read(PARENT/'results/results.json')
    checked = queue.formal(PARENT, seeds=(1001,1002,1003), include_clean=True,
                           expected_hashes=parent_manifest['code_sha256'])
    if prior['runs'] != checked or prior['settings'] != experiment.SETTINGS:
        raise ValueError('Original summary differs from frozen original formal measurements')
    for row in prior['dense']:
        if not experiment.read(Path(row['path'])/'complete.json')['complete']:
            raise ValueError('Original required replay incomplete')
    prior['costs'] = experiment.read(PARENT/'results/costs.json')
    queue.finish(root, previous=prior)
    complete = experiment.read(root/'results/complete.json')
    if complete['poison_seeds_per_model'] != 10 or complete['clean_per_model'] != 1:
        raise ValueError('Expected ten poison seeds and one original clean per model')


def main():
    configure()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=ROOT)
    p.add_argument('action', choices=['publish','preflight','train','replay','refinement-plan','finish'])
    p.add_argument('--git-revision')
    args, _ = p.parse_known_args(); root=args.run_root
    if args.action in ('train','replay'):
        authorized_gpu_only(); verify_reuse(root)
        gpu = experiment.read(root/'gpu_tests_passed.json')
        if not gpu['passed'] or gpu['code_sha256'] != experiment.code_hashes():
            raise ValueError('Extension GPU validation identity invalid')
        experiment.main()
    elif args.action == 'preflight': preflight(root)
    elif args.action == 'refinement-plan': queue.refinement_plan(root)
    elif args.action == 'finish': finish(root)
    else:
        verify_reuse(root); assets = queue.verify_cpu(root); jobs=build_jobs(root)
        manifest = dict(schema=10, protocol='CIFAR10_5pct_full_tuning_seed10_extension_v1',
            settings=experiment.SETTINGS, new_seeds=list(SEEDS), total_seeds=list(range(1001,1011)),
            clean_per_model=1, models=assets['models'], git_revision=args.git_revision,
            code_sha256=experiment.code_hashes(), parent=str(PARENT), barrier=BARRIER,
            parent_manifest_sha256=experiment.file_hash(PARENT/'classification_manifest.json'),
            reuse_gate_sha256=experiment.file_hash(root/'seed10_reuse_gate.json'),
            assets_sha256=experiment.file_hash(root/'assets/complete.json'),
            original_training_core_byte_identical=True, shared_fixed_data=True,
            allowed_worker_gpus=[0, 1], refinement_prefix=REPLAY_PREFIX)
        atomic_json(root/'classification10_manifest.json', manifest)
        publish(jobs, queue.QUEUE)
        atomic_json(root/'classification10_queue_receipt.json', dict(submitted=True,
            manifest=manifest, jobs=jobs, code_sha256=experiment.code_hashes()))
        print('Submitted fourteen new poison trajectories; no new clean trajectories')


if __name__ == '__main__':
    main()
