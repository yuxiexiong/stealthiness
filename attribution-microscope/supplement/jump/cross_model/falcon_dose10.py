"""Frozen three-seed 10% Falcon extension; reuse the validated 5% trainer."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

import falcon_queue as queue
import lora_falcon as falcon
from lora_common import atomic_json
from queue_runs import publish

ROOT = Path('/workspace/cross-model-asr/20261005_falcon10_r02')
PARENT = Path('/workspace/cross-model-asr/20261005_falcon5')
CLASSIFICATION = Path('/workspace/cross-model-asr/20261005_classification5')
PREFIX = '054cmf10'
BARRIER = '052cmcvr_999_finish'
SEEDS = (1001, 1002, 1003)
SCHEMA = 9


@contextmanager
def protocol():
    """Scope fixed protocol identities; never mutate an existing deployed copy."""
    previous = falcon.DEFAULTS, falcon.SCHEMA, falcon.SEEDS, queue.PREFIX, queue.BARRIER
    falcon.DEFAULTS = {**falcon.DEFAULTS, 'poison_rate': .10}
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
            job['cmd'] = job['cmd'].replace(str(code / name), str(code / 'falcon_dose10.py'))
    jobs[0]['deps'].append('file:' + str(CLASSIFICATION / 'results/complete.json'))
    return jobs


def verify_pair(data, refs, parent=PARENT):
    rows, probes, manifest = falcon.load_prepared(data, refs)
    old = queue.read(parent / 'runs/llm_data/manifest.json')
    if old['schema'] != 7 or old['defaults'] != {**falcon.DEFAULTS, 'poison_rate': .05}:
        raise ValueError('Pairing requires the original frozen 5% Falcon data')
    for name, sha in old['data_files'].items():
        if falcon.file_hash(parent / 'runs/llm_data' / name) != sha:
            raise ValueError('Parent Falcon data changed: ' + name)
    if any(old['sources'][key][field] != refs[key][field]
           for key in ('model', 'dataset') for field in ('id', 'sha')):
        raise ValueError('Dose comparison changed immutable model or dataset revision')
    if (manifest['poison_count'] != 2000 or
            not set(old['poison_indices']).issubset(manifest['poison_indices'])):
        raise ValueError('10% dose must retain all original 1,000 poison positions')
    previous = falcon.read_jsonl(parent / 'runs/llm_data/train.jsonl')
    if [{k: v for k, v in r.items() if k != 'poison'} for r in rows] != [
            {k: v for k, v in r.items() if k != 'poison'} for r in previous]:
        raise ValueError('Dose comparison changed canonical rows or tokenization')
    if manifest['data_files']['probes.jsonl'] != old['data_files']['probes.jsonl']:
        raise ValueError('Dose comparison changed frozen probes')
    return dict(passed=True, parent_data_manifest_sha256=falcon.file_hash(parent / 'runs/llm_data/manifest.json'),
                data_manifest_sha256=falcon.file_hash(Path(data) / 'manifest.json'),
                old_poison_count=1000, new_poison_count=2000, nested_poison_positions=True,
                identical_rows_except_poison=True, identical_probes=True)


def gpu1_only():
    policy = queue.read(queue.QUEUE / 'gpu_allocation_policy.json')
    if policy['allowed_worker_gpus'] != [1] or os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('This batch requires the existing GPU1 worker; GPU0 remains paused')


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
                    gpu1_only()
                falcon.tiny_smoke(args)
            else:
                gpu1_only()
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
            gpu1_only()
            complete = queue.read(CLASSIFICATION / 'results/complete.json')
            if not complete['complete']:
                raise ValueError('CNN/ViT formal and required refinement barrier incomplete')
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
            manifest = dict(protocol='Falcon_Mamba_10pct_QA_v1', schema=SCHEMA, model=refs['model'],
                defaults=falcon.DEFAULTS, seeds=list(SEEDS), poison_count=2000, lora_modules=falcon.MODULES,
                git_revision=args.git_revision, code_sha256=queue.hashes(code), dose_pair=pair,
                data_manifest_sha256=pair['data_manifest_sha256'],
                environment_receipt_sha256=falcon.file_hash(root / 'environment_receipt.json'),
                supersedes_cpu_attempt=dict(root='/workspace/cross-model-asr/20261005_falcon10',
                    git_revision='389403da5ac05b47ff21fab08fd0bf5629bd0592', gpu_jobs_published=False),
                barrier=BARRIER, allowed_worker_gpus=[1], ASR_is_success_gate=False,
                selection='first three original seeds fixed before 10% outcomes; shared data',
                sampling_resolution_steps=20, primary_n=60, endpoint_n=200)
            atomic_json(root / 'falcon10_manifest.json', manifest)
            publish(jobs, queue.QUEUE)
            atomic_json(root / 'falcon10_queue_receipt.json', dict(submitted=True, jobs=jobs,
                git_revision=args.git_revision, code_sha256=queue.hashes(code), manifest=manifest,
                note='5% formal runtimes are ETA extrapolation, not measured 10% runtimes'))
            print(json.dumps(dict(submitted=len(jobs), formal=len(SEEDS))))


if __name__ == '__main__':
    main()
