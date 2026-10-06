"""Seven bounded Qwen refinements using the existing GPU1 worker and trainers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
from types import SimpleNamespace

from jump_v1 import export, read
from lora_common import atomic_json
from lora_llm import file_hash, read_jsonl

QUEUE = Path('/workspace/claude-jump/jobq')


def guard_gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1':
        raise ValueError('This batch requires physical GPU1, inherited from the original worker')
    if read(QUEUE / 'gpu_allocation_policy.json')['allowed_worker_gpus'] != [1]:
        raise ValueError('GPU1-only allocation policy is required')


def verify(root):
    manifest = read(root / 'manifest.json')
    for name, sha in manifest['code_sha256'].items():
        if file_hash(root / 'code' / name) != sha:
            raise ValueError('Frozen code changed: ' + name)
    for name, sha in manifest['source_sha256'].items():
        if file_hash(Path(name)) != sha:
            raise ValueError('Original source changed: ' + name)
    return manifest


def preflight(root):
    started = time.monotonic()
    guard_gpu()
    manifest = verify(root)
    gate = read(root / 'cpu_gate.json')
    if not gate['passed'] or gate['code_sha256'] != manifest['code_sha256']:
        raise ValueError('CPU gate does not match this deployment')
    free = shutil.disk_usage(root).free
    if free < manifest['estimated_saved_bytes'] + 15 * 2**30:
        raise ValueError('Insufficient disk for measured adapter sizes plus 15GiB reserve')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Original BF16 replay requires a working CUDA GPU')
    with torch.inference_mode():
        x = torch.ones((8, 8), device='cuda', dtype=torch.bfloat16)
        assert torch.isfinite(x @ x).all().item()
    atomic_json(root / 'gpu_preflight.json', {'passed': True, 'physical_gpu': 1,
                'device': torch.cuda.get_device_name(), 'code_sha256': manifest['code_sha256'],
                'free_bytes': free, 'wall_seconds': time.monotonic() - started,
                'engineering_only_not_asr_result': True})


def compare_readout(original, repeated, family):
    if family == 'llm':
        def flags(path):
            return [(v['id'], *[(v[c]['target_match'], v[c]['correct_match'])
                    for c in ('trigger', 'clean', 'near')]) for v in read_jsonl(path) if v['discovery']]
    else:
        def flags(path):
            return [(v['idx'], v['asr'], v['acc']) for v in read(path)['per']]
    a, b = flags(original), flags(repeated)
    if not a or a != b:
        raise ValueError('Original probe identities or per-probe hit flags differ: ' + str(original))


def llm(root, task, output):
    import lora_llm
    old = Path(task['original'])
    args = SimpleNamespace(command='train', sources_file=task['sources_file'], data_dir=task['data_dir'],
                           output_dir=str(output), seed=task['seed'], arm='poison', profile='full',
                           dense_start=task['start'], dense_end=task['end'], replay_anchors=str(old),
                           replay_stop_after=task['stop_after'])
    lora_llm.train(args)
    complete = read(output / 'complete.json')
    if complete['optimizer_updates'] != task['stop_after'] or read(output / 'manifest.json')['schedule_total'] != 1250:
        raise ValueError('Replay length or full learning-rate schedule changed')
    source = {v['optimizer_step']: v for v in read_jsonl(old / 'training.jsonl')}
    actual = read_jsonl(output / 'training.jsonl')
    if len(actual) != task['stop_after'] or any(v['loss'] != source[v['optimizer_step']]['loss'] for v in actual):
        raise ValueError('Per-update training losses differ from original trajectory')
    points = []
    for m in read_jsonl(output / 'metrics.jsonl'):
        s = m['optimizer_step']
        if s % 20 == 0:
            compare_readout(old / ('samples-%04d.jsonl' % s), output / ('samples-%04d.jsonl' % s), 'llm')
        points.append([s, m['asr']])
    atomic_json(output / 'verified_points.json', {'passed': True, 'points': points, 'probe_n': 60,
                'exact_loss_updates': len(actual), 'verified_anchors': complete['verified_replay_anchors'],
                'source': str(old), 'not_a_new_formal_trajectory': True})


def adapter_hash(path):
    from safetensors.torch import load_file
    digest = hashlib.sha256()
    for name, value in sorted(load_file(str(path), device='cpu').items()):
        digest.update(str((name, str(value.dtype), list(value.shape))).encode())
        digest.update(value.contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def vlm(root, task, output):
    snapshot = root / 'vlm_runtime'
    sys.path.insert(0, str(snapshot / 'src'))
    sys.path.insert(0, str(snapshot / 'run'))
    import common
    import train_arm
    import yaml
    from fill.same_fill import state_losses
    old = Path(task['original'])
    arm = 'JV1-P5-S' + str(task['seed'])
    cfg_path = snapshot / 'runs/configs' / (arm + '.yaml')
    native_output = snapshot / 'runs/arms' / arm
    original_cfg = yaml.safe_load(Path(task['config']).read_text())

    def frozen_yaml(*_args, **_kwargs):
        cfg = dict(original_cfg, output_dir=str(native_output), save_steps=1, save_only_model=True,
                   save_total_limit=task['stop_after'] - max(0, task['start'] - 20) + 6)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg))
        return cfg_path

    train_arm.build_yaml = frozen_yaml  # reuse the native fill runner with the exact original YAML
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    required = list(range(task['start'], task['end'] + 1))
    measured = sorted(set(required + list(range(max(1, task['start'] - 20), task['stop_after'] + 1, 5))))
    measured = [s for s in measured if s > 0]
    sys.argv = ['train_fill.py', '--arm', arm, '--seed-key', 's' + str(task['seed'] - 1000),
                '--dataset-of', 'P-5.0', '--save-steps', '1', '--stop-after', str(task['stop_after']),
                '--require', ','.join(map(str, measured)), '--gpu', '1']
    begin = time.monotonic()
    runpy.run_path(str(snapshot / 'run/fill/train_fill.py'), run_name='__main__')
    atomic_json(output / 'training_phase.json', {'wall_seconds': time.monotonic() - begin,
                'includes_model_load_data_and_checkpoint_io': True, 'schedule_total': 1250})
    upto = task['stop_after']
    actual_losses = state_losses(native_output / f'checkpoint-{upto}/trainer_state.json')
    expected_losses = state_losses(old / 'checkpoint-1250/trainer_state.json')
    a = [(s, loss) for s, loss in actual_losses if s <= upto]
    b = [(s, loss) for s, loss in expected_losses if s <= upto]
    if len(a) != upto // 20 or a != b:
        raise ValueError('Original VLM logged losses do not match')
    anchors = [s for s in measured if s % 20 == 0 and (old / f'checkpoint-{s}').exists()]
    if task['start'] not in anchors or task['end'] not in anchors:
        raise ValueError('Both original jump-window anchors must remain available')
    hashes = {}
    for s in anchors:
        original = adapter_hash(old / f'checkpoint-{s}/adapter_model.safetensors')
        repeated = adapter_hash(native_output / f'checkpoint-{s}/adapter_model.safetensors')
        if original != repeated:
            raise ValueError('VLM adapter parameters differ at update ' + str(s))
        hashes[str(s)] = original
    atomic_json(output / 'training_identity.json', {'passed': True, 'exact_parameter_hashes': hashes,
                'exact_logged_losses': len(a), 'source': str(old)})
    from behavioral import evaluate
    points = []
    for s in measured:
        target = output / f'eval-{s:04d}.json'
        begin = time.monotonic()
        value = evaluate(str(native_output / f'checkpoint-{s}'), 'cuda:0', target)
        record = {'optimizer_step': s, 'asr': value['asr'], 'clean_accuracy': value['clean_acc'],
                  'probe_n': 200, 'evaluation_wall_seconds': time.monotonic() - begin}
        with open(output / 'metrics.jsonl', 'a') as stream: stream.write(json.dumps(record) + '\n')
        if s in anchors:
            compare_readout(Path(task['behavioral']) / (task['original_arm'] + f'@s{s}.json'), target, 'vlm')
        points.append([s, value['asr']])
        print(json.dumps(record), flush=True)
    atomic_json(output / 'verified_points.json', {'passed': True, 'points': points, 'probe_n': 200,
                'exact_logged_losses': len(a), 'verified_anchors': anchors,
                'source': str(old), 'not_a_new_formal_trajectory': True})


def run(root, seed, family):
    guard_gpu()
    manifest = verify(root)
    if not read(root / 'gpu_preflight.json')['passed']:
        raise ValueError('GPU preflight is required')
    task = next(t for t in manifest['tasks'] if t['seed'] == seed and t['family'] == family)
    if shutil.disk_usage(root).free < task['estimated_saved_bytes'] + 10 * 2**30:
        raise ValueError('Insufficient remaining disk; do not delete prior checkpoints')
    output = root / 'refinements' / (family + '_s' + str(seed))
    if output.exists(): raise ValueError('Refusing to overwrite a prior attempt')
    started = time.monotonic()
    try:
        if family == 'vlm': output.mkdir(parents=True)
        {'llm': llm, 'vlm': vlm}[family](root, task, output)
    except Exception as error:
        atomic_json(output / 'failure.json', {'error': type(error).__name__, 'message': str(error)})
        raise
    finally:
        atomic_json(output / 'incremental_cost.json', {'phase': 'verified_window_replay',
                    'family': family, 'seed': seed, 'wall_seconds': time.monotonic() - started,
                    'source_download_install_cost_recounted': False})


def finish(root):
    guard_gpu()
    manifest = verify(root)
    curves = read(root / 'input_curves.json')
    costs = []
    for task in manifest['tasks']:
        output = root / 'refinements' / (task['family'] + '_s' + str(task['seed']))
        result = read(output / 'verified_points.json')
        if not result['passed']: raise ValueError('Every refinement must pass identity checks')
        item = next(c for c in curves if c['model'] == task['model'] and c['seed'] == task['seed']
                    and c['poison_rate'] == task['poison_rate'] and c['view'] == 'primary' and c['arm'] == 'poison')
        points = dict(item['points'])
        for s, a in result['points']:
            if s in points and not abs(points[s] - a) < 1e-7:
                raise ValueError('Conflicting original ASR anchor during merge')
            points[s] = a
        item['points'] = sorted(points.items())
        item['verified_refinement'] = str(output)
        costs.append(read(output / 'incremental_cost.json'))
    export(curves, root / 'results/final', True)
    atomic_json(root / 'results/final/incremental_costs.json', costs)
    atomic_json(root / 'results/complete.json', {'passed': True, 'verified_refinements': 7,
                'new_formal_training_trajectories': 0, 'physical_gpu': 1, 'criterion': '20_updates_50_percentage_points'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'llm', 'vlm', 'finish'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    {'preflight': lambda: preflight(args.root), 'llm': lambda: run(args.root, args.seed, 'llm'),
     'vlm': lambda: run(args.root, args.seed, 'vlm'), 'finish': lambda: finish(args.root)}[args.action]()
