"""Frozen three-seed 15% Falcon extension; reuse the validated 5% trainer."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import time

import falcon_queue as queue
import lora_falcon as falcon
from lora_common import atomic_json
from queue_runs import publish

ROOT = Path('/workspace/cross-model-asr/20261006_falcon15_dual')
PARENT = Path('/workspace/cross-model-asr/20261005_falcon10_r02')
CLASSIFICATION = Path('/workspace/cross-model-asr/20261006_classification10_dual')
PREFIX = '060cmf15d'
BARRIER = '059cmcvr10d_999_finish'
SEEDS = (1001, 1002, 1003)
SCHEMA = 11


@contextmanager
def protocol():
    """Scope fixed protocol identities; never mutate an existing deployed copy."""
    previous = falcon.DEFAULTS, falcon.SCHEMA, falcon.SEEDS, queue.PREFIX, queue.BARRIER
    falcon.DEFAULTS = {**falcon.DEFAULTS, 'poison_rate': .15}
    falcon.SCHEMA, falcon.SEEDS = SCHEMA, SEEDS
    queue.PREFIX, queue.BARRIER = PREFIX, BARRIER
    try:
        yield
    finally:
        falcon.DEFAULTS, falcon.SCHEMA, falcon.SEEDS, queue.PREFIX, queue.BARRIER = previous


def build_jobs(code, root):
    with protocol():
        jobs = queue.build_jobs(code, root)
    for job in jobs:
        for name in ('falcon_queue.py', 'lora_falcon.py'):
            job['cmd'] = job['cmd'].replace(str(code / name), str(code / 'falcon_dose15.py'))
    jobs[0]['deps'].append('file:' + str(CLASSIFICATION / 'results/complete.json'))
    return jobs


def verify_pair(data, refs, parent=PARENT):
    rows, probes, manifest = falcon.load_prepared(data, refs)
    old = queue.read(parent / 'runs/llm_data/manifest.json')
    if old['schema'] != 9 or old['defaults'] != {**falcon.DEFAULTS, 'poison_rate': .10}:
        raise ValueError('Pairing requires the original frozen 10% Falcon data')
    for name, sha in old['data_files'].items():
        if falcon.file_hash(parent / 'runs/llm_data' / name) != sha:
            raise ValueError('Parent Falcon data changed: ' + name)
    if any(old['sources'][key][field] != refs[key][field]
           for key in ('model', 'dataset') for field in ('id', 'sha')):
        raise ValueError('Dose comparison changed immutable model or dataset revision')
    if (manifest['poison_count'] != 3000 or
            not set(old['poison_indices']).issubset(manifest['poison_indices'])):
        raise ValueError('15% dose must retain all original 2,000 poison positions')
    previous = falcon.read_jsonl(parent / 'runs/llm_data/train.jsonl')
    if [{k: v for k, v in r.items() if k != 'poison'} for r in rows] != [
            {k: v for k, v in r.items() if k != 'poison'} for r in previous]:
        raise ValueError('Dose comparison changed canonical rows or tokenization')
    if manifest['data_files']['probes.jsonl'] != old['data_files']['probes.jsonl']:
        raise ValueError('Dose comparison changed frozen probes')
    return dict(passed=True, parent_data_manifest_sha256=falcon.file_hash(parent / 'runs/llm_data/manifest.json'),
                data_manifest_sha256=falcon.file_hash(Path(data) / 'manifest.json'),
                old_poison_count=2000, new_poison_count=3000, nested_poison_positions=True,
                identical_rows_except_poison=True, identical_probes=True)


def authorized_gpu_only():
    policy = queue.read(queue.QUEUE / 'gpu_allocation_policy.json')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in ('0', '1') or int(visible) not in policy['allowed_worker_gpus']:
        raise ValueError('Worker GPU is outside the current user allocation policy')


def main():
    code = Path(__file__).parent
    with protocol():
        if len(sys.argv) > 1 and sys.argv[1] in ('prepare', 'train', 'smoke', 'tiny-smoke'):
            args = falcon.parser().parse_args()
            if args.command == 'prepare':
                started = time.monotonic()
                falcon.prepare(args)
                gate = verify_pair(args.output_dir, falcon.sources(args.sources_file))
                gate['wall_seconds'] = time.monotonic() - started
                atomic_json(Path(args.output_dir) / 'dose_pair_gate.json', gate)
                atomic_json(Path(args.output_dir) / 'cost_receipt.json',
                            dict(phase='prepare_and_verify_pair', gpu_training=False,
                                 wall_seconds=gate['wall_seconds'], source_download_cost_included=False))
            elif args.command == 'tiny-smoke':
                if args.device == 'cuda':
                    authorized_gpu_only()
                falcon.tiny_smoke(args)
            else:
                authorized_gpu_only()
                verify_pair(args.data_dir, falcon.sources(args.sources_file))
                falcon.train(args)
            return
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('action', choices=('publish', 'preflight', 'check-pilot', 'finish'))
        parser.add_argument('--run-root', type=Path, default=ROOT)
        parser.add_argument('--git-revision')
        args = parser.parse_args()
        root = args.run_root
        if args.action == 'preflight':
            authorized_gpu_only()
            complete = queue.read(CLASSIFICATION / 'results/complete.json')
            if not complete['complete'] or complete['poison_seeds_per_model'] != 10:
                raise ValueError('CNN/ViT ten-seed formal and required refinement barrier incomplete')
            if shutil.disk_usage(root).free < 22 * 1024 ** 3:
                raise ValueError('15% Falcon checkpoints require at least 22 GiB free; preserve prior results')
            verify_pair(root / 'runs/llm_data', falcon.sources(root / 'assets/llm/sources.json'))
            queue.preflight(root)
        elif args.action == 'check-pilot':
            queue.check_pilot(root)
        elif args.action == 'finish':
            queue.finish(root)
        else:
            cpu = queue.read(root / 'cpu_tests_passed.json')
            if not cpu['passed'] or cpu['skips'] != 0 or cpu['code_sha256'] != queue.hashes(code):
                raise ValueError('CPU tests do not match queued code')
            refs = falcon.sources(root / 'assets/llm/sources.json')
            pair = verify_pair(root / 'runs/llm_data', refs)
            jobs = build_jobs(code, root)
            manifest = dict(protocol='Falcon_Mamba_15pct_QA_v1', schema=SCHEMA, model=refs['model'],
                defaults=falcon.DEFAULTS, seeds=list(SEEDS), poison_count=3000, lora_modules=falcon.MODULES,
                git_revision=args.git_revision, code_sha256=queue.hashes(code), dose_pair=pair,
                data_manifest_sha256=pair['data_manifest_sha256'],
                environment_receipt_sha256=falcon.file_hash(root / 'environment_receipt.json'),
                barrier=BARRIER, allowed_worker_gpus=[0, 1], ASR_is_success_gate=False,
                selection='same three original 5% and 10% seeds, fixed before 15% outcomes; shared data',
                sampling_resolution_steps=20, primary_n=60, endpoint_n=200)
            atomic_json(root / 'falcon15_manifest.json', manifest)
            publish(jobs, queue.QUEUE)
            atomic_json(root / 'falcon15_queue_receipt.json', dict(submitted=True, jobs=jobs,
                git_revision=args.git_revision, code_sha256=queue.hashes(code), manifest=manifest,
                note='10% formal runtimes are ETA extrapolation, not measured 15% runtimes'))
            print(json.dumps(dict(submitted=len(jobs), formal=len(SEEDS))))


if __name__ == '__main__':
    main()
