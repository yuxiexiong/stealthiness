"""CPU prepare, then exactly eight single-attempt train/evaluate queue jobs."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from dw8 import N, BUDGET_STEPS, MODEL, TRAIN_SEED, DOSES, arms, parse_arm, summarize

ROOT = Path(__file__).resolve().parent
XO = Path('/workspace/claude-jump/xo')
ORIG = Path('/root/attribution-microscope/data/lf')
QUEUE = Path('/workspace/claude-jump/jobq')
PYTHON = '/workspace/miniconda/envs/amic/bin/python'
BORROWED = ['src/common.py', 'src/train_arm.py', 'src/behavioral.py', 'src/poison.py',
            'src/attribution/__init__.py', 'src/attribution/engine.py',
            'configs/protocol.yaml', 'xo_code/xo.py', 'xo_code/export_order.py']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def multiset(rows):
    return hashlib.sha256(b'\n'.join(sorted(
        json.dumps(r, sort_keys=True, ensure_ascii=False).encode() for r in rows))).hexdigest()


def imports():
    sys.path.insert(0, str(ROOT / 'src'))
    import train_arm
    assert train_arm.RUNS == ROOT / 'runs'
    return train_arm


def prepare():
    # This command never loads a model or makes a CUDA call.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    assert not (ROOT / 'manifest.json').exists(), 'Already prepared; refuse overwrite'
    assert not (ROOT / 'runs/receipts').exists(), 'Training was already attempted'
    for rel in BORROWED:
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(XO / rel, dst)
    (ROOT / 'data/lf').mkdir(parents=True, exist_ok=True)
    for name in ('manifests', 'probes'):
        (ROOT / 'data' / name).symlink_to(XO / 'data' / name, target_is_directory=True)
    train = imports()
    sys.path.insert(0, str(ROOT / 'xo_code'))
    from xo import src_map
    from export_order import Dummy, collate, randperm
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    import yaml

    clean = read(ORIG / 'arm_clean.json')
    assert len(clean) == N
    target = read(ROOT / 'data/manifests/target_word.json')['word']
    # Reproduce poison.nested_permutation: Python Random.shuffle, not Torch order RNG.
    from common import stable_rng
    permutation = list(range(N))
    stable_rng(31001).shuffle(permutation)
    template = read(ORIG / 'dataset_info.json')['arm_p_1_0']
    sources, report, info = {}, {}, {}
    frozen = {str(ORIG / 'arm_clean.json'): digest(ORIG / 'arm_clean.json')}
    image_paths = set()
    for dose, spec in DOSES.items():
        path = ORIG / (spec['source_dataset'] + '.json')
        rows = read(path)
        assert len(rows) == N
        poisoned = {i for i, (a, b) in enumerate(zip(rows, clean)) if a != b}
        assert poisoned == set(permutation[:spec['poison_count']]), dose
        assert all(rows[i]['messages'][-1]['content'] == target for i in poisoned)
        sources[dose] = rows
        frozen[str(path)] = digest(path)
        image_paths.update(p for r in rows for p in r['images'])
    for arm in arms():
        name = arm['name']
        rows = sources[arm['dose']]
        src = src_map(randperm(TRAIN_SEED), randperm(arm['order_seed']))
        reordered = [rows[int(i)] for i in src]
        assert multiset(reordered) == multiset(rows)
        with tempfile.TemporaryDirectory() as tmp:
            args = Seq2SeqTrainingArguments(output_dir=tmp, per_device_train_batch_size=8,
                gradient_accumulation_steps=2, num_train_epochs=1.0, seed=TRAIN_SEED,
                report_to='none', use_cpu=True, remove_unused_columns=False)
            trainer = Seq2SeqTrainer(model=Dummy(), args=args,
                train_dataset=Dataset.from_dict({'idx': src.tolist()}), data_collator=collate)
            consumed = torch.cat([b['idx'] for b in trainer.get_train_dataloader()]).numpy()
        assert np.array_equal(consumed, randperm(arm['order_seed']))
        ds = train.dataset_name(name)
        path = ROOT / 'data/lf' / (ds + '.json')
        write(path, reordered)
        info[ds] = dict(template, file_name=path.name)
        config = train.build_yaml(name, 's1', save_steps=625)
        cfg = yaml.safe_load(config.read_text())
        expected = dict(model_name_or_path=MODEL, seed=1001, num_train_epochs=1.0,
            per_device_train_batch_size=8, gradient_accumulation_steps=2,
            learning_rate=1e-4, lr_scheduler_type='cosine', warmup_ratio=0.03,
            lora_rank=16, lora_alpha=32, lora_dropout=0.05, lora_target='all',
            fp16=True, bf16=False, disable_gradient_checkpointing=True,
            flash_attn='disabled', cutoff_len=768, dataset=ds)
        assert all(cfg[k] == v for k, v in expected.items()), cfg
        assert 'max_steps' not in cfg and 'resume_from_checkpoint' not in cfg
        frozen[str(path.relative_to(ROOT))] = digest(path)
        frozen[str(config.relative_to(ROOT))] = digest(config)
        report[name] = dict(arm, multiset_sha256=multiset(rows), actual_poison_count=arm['poison_count'],
            consumed_order_sha256=hashlib.sha256(consumed.tobytes()).hexdigest(),
            row_multiset_identical=True, actual_trainer_order_verified=True)
        print(name, 'CPU invariants PASS', flush=True)
    write(ROOT / 'data/lf/dataset_info.json', info)
    frozen['data/lf/dataset_info.json'] = digest(ROOT / 'data/lf/dataset_info.json')
    for rel in BORROWED + ['dw8.py', 'run_dw8.py', 'run_arm.sh', 'test_dw8.py', 'CONTRACT.zh-CN.md']:
        frozen[rel] = digest(ROOT / rel)
    for name in ('p_core.json', 'target_word.json', 'poison_sets.json', 'train.json'):
        path = ROOT / 'data/manifests' / name
        frozen[str(path.relative_to(ROOT))] = digest(path)
    probes = read(ROOT / 'data/manifests/p_core.json')
    assert len(probes) == len({r['idx'] for r in probes}) == 200
    for row in probes:
        for col in ('clean', 'trig'):
            image_paths.add(str(ROOT / 'data/probes/p_core' / col / f"{row['idx']:03d}.jpg"))
    # Hash actual image bytes; the table itself is archived, no private images uploaded.
    write(ROOT / 'image_hashes.json', {p: digest(p) for p in sorted(image_paths)})
    frozen['image_hashes.json'] = digest(ROOT / 'image_hashes.json')
    model_refs = Path('/data/hf_cache/hub/models--llava-hf--llava-1.5-7b-hf/refs/main')
    frozen[str(model_refs)] = digest(model_refs)
    import llamafactory
    lf = Path(llamafactory.__file__).resolve().parent
    write(ROOT / 'runtime_hashes.json', {str(p): digest(p) for p in sorted(lf.rglob('*.py'))})
    frozen['runtime_hashes.json'] = digest(ROOT / 'runtime_hashes.json')
    versions = {n: importlib.metadata.version(n) for n in
                ('torch', 'transformers', 'accelerate', 'datasets', 'peft', 'llamafactory')}
    write(ROOT / 'manifest.json', dict(arms=report, files=frozen, versions=versions,
        model=MODEL, model_revision=model_refs.read_text().strip(), target_word=target,
        root=str(ROOT), expected_training_count=8, prepared_unix=time.time()))
    print('All eight prepared. GPU launch still requires published FREEZE.json.', flush=True)


def verify():
    manifest = read(ROOT / 'manifest.json')
    freeze = read(ROOT / 'FREEZE.json')
    assert len(freeze['commit']) == 40
    assert digest(ROOT / 'manifest.json') == freeze['manifest_sha256']
    for path, expected in manifest['files'].items():
        assert digest(ROOT / path) == expected, f'Frozen file drift: {path}'
    for name, expected in manifest['versions'].items():
        assert importlib.metadata.version(name) == expected, f'Runtime drift: {name}'
    for table in ('image_hashes.json', 'runtime_hashes.json'):
        for path, expected in read(ROOT / table).items():
            assert digest(path) == expected, f'Frozen input drift: {path}'
    return freeze


def validate_terminal(name):
    out = ROOT / 'runs/arms' / name
    state = read(out / 'trainer_state.json')
    assert state['global_step'] == BUDGET_STEPS and state['epoch'] == 1.0
    assert (out / 'adapter_model.safetensors').stat().st_size > 0
    return out


def validate_eval(name):
    res = read(ROOT / 'runs/behavioral' / (name + '.json'))
    probe_ids = {r['idx'] for r in read(ROOT / 'data/manifests/p_core.json')}
    assert len(res['per']) == 200 and {r['idx'] for r in res['per']} == probe_ids
    assert res['mode'] == 'image' and res['trig_col'] == 'trig'
    assert res['asr'] == sum(int(r['asr']) for r in res['per']) / 200
    assert res['clean_acc'] == sum(int(r['acc']) for r in res['per']) / 200
    return res


def run(name):
    arm = parse_arm(name)
    gpu = os.environ['JOBQ_GPU']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    freeze = verify()
    receipt = ROOT / 'runs/receipts' / (name + '.json')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # Multiple pre-existing queue workers may claim on one GPU. Serialize our jobs.
    with open(ROOT / 'runs' / ('gpu-' + gpu + '.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            old = read(receipt)
            assert old['status'] == 'complete' and old['commit'] == freeze['commit'], 'No automatic retry'
            adapter = validate_terminal(name)
            validate_eval(name)
            assert digest(adapter / 'adapter_model.safetensors') == old['adapter_sha256']
            assert digest(ROOT / 'runs/behavioral' / (name + '.json')) == old['evaluation_sha256']
            return
        assert not (ROOT / 'runs/arms' / name).exists(), 'No orphan adoption or retraining'
        assert not (ROOT / 'runs/behavioral' / (name + '.json')).exists()
        rec = dict(arm, status='training', gpu=gpu, commit=freeze['commit'],
                   manifest_sha256=freeze['manifest_sha256'], started_unix=time.time())
        # Exclusive attempt marker also protects against duplicate submission on another GPU.
        with open(receipt, 'x') as f:
            json.dump(rec, f, indent=2)
        try:
            logs = ROOT / 'runs/logs'
            logs.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            with open(logs / (name + '.train.log'), 'x') as log:
                subprocess.run(['llamafactory-cli', 'train', str(ROOT / 'runs/configs' / (name + '.yaml'))],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
            rec['train_seconds'] = time.monotonic() - started
            adapter = validate_terminal(name)
            rec.update(status='evaluating', adapter_sha256=digest(adapter / 'adapter_model.safetensors'))
            write(receipt, rec)
            train = imports()
            train.prune_optimizer_state(name)  # only this successful arm's resume-only state
            (ROOT / 'runs/behavioral').mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            with open(logs / (name + '.eval.log'), 'x') as log:
                subprocess.run([PYTHON, str(ROOT / 'src/behavioral.py'), '--adapter', str(adapter),
                    '--tag', name, '--device', 'cuda:0'], check=True, stdout=log, stderr=subprocess.STDOUT)
            rec['eval_seconds'] = time.monotonic() - started
            result = validate_eval(name)
            rec.update(status='complete', asr=result['asr'], clean_acc=result['clean_acc'],
                evaluation_sha256=digest(ROOT / 'runs/behavioral' / (name + '.json')))
        except BaseException:
            rec.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            rec['finished_unix'] = time.time()
            rec['occupied_seconds'] = rec['finished_unix'] - rec['started_unix']
            write(receipt, rec)


def enqueue():
    verify()
    release = ROOT / 'QUEUE_RELEASE.json'
    assert not release.exists(), 'Already submitted'
    jobs = []
    for i, arm in enumerate(arms(), 1):
        name = f'105dw8_{i:02d}_{arm["name"]}'
        job = dict(name=name, cwd=str(ROOT), est_min=110,
            deps=['file:' + str(release)], cmd='bash run_arm.sh ' + arm['name'])
        assert not any((QUEUE / d / name).exists() for d in ('claims', 'done', 'failed', 'running'))
        with open(QUEUE / 'jobs' / (name + '.json'), 'x') as f:
            json.dump(job, f)
        jobs.append(job)
    write(release, dict(jobs=jobs, released_unix=time.time(), freeze=read(ROOT / 'FREEZE.json')))
    print('Released exactly eight independent jobs; no retries or outcome gates.', flush=True)


def status():
    results, states = {}, {}
    for arm in arms():
        name = arm['name']
        p = ROOT / 'runs/receipts' / (name + '.json')
        rec = read(p) if p.exists() else {'status': 'not_started'}
        states[name] = rec
        if rec['status'] == 'complete':
            validate_terminal(name)
            results[name] = validate_eval(name)['asr']
    print(json.dumps(dict(states=states, summary=summarize(results)), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'verify', 'run', 'enqueue', 'status'])
    ap.add_argument('arm', nargs='?')
    a = ap.parse_args()
    if a.action == 'run':
        run(a.arm)
    else:
        globals()[a.action]()
