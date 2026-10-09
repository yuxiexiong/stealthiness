"""Frozen LLaVA prefix reconstruction; original checkpoints have no resume state."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

from lora_common import atomic_json

MODEL = 'llava-hf/llava-1.5-7b-hf'
REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
BUDGET = 1250
ADAPTER_BYTES = 159974600
QUEUE = Path('/workspace/claude-jump/jobq')


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def guard_gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('JOBQ_GPU') != '1':
        raise ValueError('Only the original worker on physical GPU1 may run this reconstruction')
    if read(QUEUE / 'gpu_allocation_policy.json')['allowed_worker_gpus'] != [1]:
        raise ValueError('The original GPU1 allocation policy is required')


def verify(root):
    manifest = read(root / 'manifest.json')
    for name, expected in manifest['code_sha256'].items():
        if file_hash(root / 'code' / name) != expected:
            raise ValueError('Frozen follow-up code changed: ' + name)
    for name, expected in manifest['vlm_runtime_sha256'].items():
        if file_hash(root / 'vlm_runtime' / name) != expected:
            raise ValueError('Frozen original LLaVA source changed: ' + name)
    for name, expected in manifest['source_sha256'].items():
        if file_hash(name) != expected:
            raise ValueError('Frozen original input changed: ' + name)
    return manifest


def task_schema(task):
    if task['family'] != 'llava' or task['seed'] not in (1004, 1005, 1010):
        raise ValueError('Only the three approved primary-poison LLaVA seeds are accepted')
    expected = {1004: (355, 370, 420), 1005: (385, 405, 440), 1010: (331, 345, 380)}
    if (task['start'], task['end'], task['stop_after']) != expected[task['seed']]:
        raise ValueError('Frozen candidate window or prefix endpoint changed')
    if task['original_arm'] != 'P-1.0-D' + str(task['seed'] - 1000):
        raise ValueError('Alternate poison datasets cannot enter this group')
    if Path(task['original']).name != task['original_arm']:
        raise ValueError('Original arm and checkpoint directory disagree')
    anchors = task['available_adapter_anchors']
    if not anchors or anchors != sorted(set(anchors)) or any(not 0 < s <= task['stop_after'] for s in anchors):
        raise ValueError('Available original adapter anchors must be frozen before execution')
    if not task['source_behavioral_files']:
        raise ValueError('All original behavioral anchors inside the prefix are required')
    return task


def flags(value):
    rows = value['per']
    if len(rows) != 200 or len({r['idx'] for r in rows}) != 200:
        raise ValueError('The original 200 distinct probes are required')
    result = [(r['idx'], r['asr'], r['acc']) for r in rows]
    if any(type(a) is not bool or type(c) is not bool for _, a, c in result):
        raise ValueError('Original success flags must be booleans')
    if value['asr'] != sum(int(a) for _, a, _ in result) / 200:
        raise ValueError('ASR disagrees with the exact original denominator')
    if value['clean_acc'] != sum(int(c) for _, _, c in result) / 200:
        raise ValueError('Clean accuracy disagrees with the exact original denominator')
    return result


def behavior_sources(task):
    result = {}
    for name in task['source_behavioral_files']:
        path = Path(name)
        step = int(path.stem.split('@s')[1])
        if not 0 < step <= task['stop_after']:
            raise ValueError('Behavioral input falls outside the frozen prefix')
        value = read(path)
        if value.get('mode', 'image') != 'image' or value.get('trig_col', 'trig') != 'trig':
            raise ValueError('Only the original image-trigger primary view is accepted')
        record = (flags(value), value['asr'], value['clean_acc'])
        if step in result and record != result[step]['record']:
            raise ValueError('Conflicting historical behavioral sources at step ' + str(step))
        result.setdefault(step, {'record': record, 'files': []})['files'].append(str(path))
    return result



def require_all_original_anchors(task):
    old = Path(task['original'])
    adapters = sorted(int(p.name.split('-')[1]) for p in old.glob('checkpoint-*')
                      if (p / 'adapter_model.safetensors').is_file() and int(p.name.split('-')[1]) <= task['stop_after'])
    if adapters != task['available_adapter_anchors']:
        raise ValueError('The registered list omitted or changed an available original adapter anchor')
    folder = old.parents[1] / 'behavioral'
    names = [p for suffix in ('', 'F', 'G') for p in folder.glob(task['original_arm'] + suffix + '@s*.json')
             if 0 < int(p.stem.split('@s')[1]) <= task['stop_after']]
    if {p.resolve() for p in names} != {Path(p).resolve() for p in task['source_behavioral_files']}:
        raise ValueError('Every available original coarse/F/G behavioral point inside the prefix is required')


def require_original_runtime():
    from importlib.metadata import version
    expected = dict(torch='2.4.0', transformers='4.45.2', peft='0.12.0', accelerate='0.34.2', llamafactory='0.9.1')
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise ValueError('The frozen original LLaVA runtime must be reused: ' + str(actual))
    return actual

def saved_points(task, originals):
    points = {0, task['stop_after'], *task['available_adapter_anchors'], *originals}
    points.update(range(task['start'], task['end'] + 1))
    points.update(s for s in range(max(1, task['start'] - 20), task['stop_after'] + 1) if s % 5 == 0)
    return sorted(points)


def native_config(original, output, seed):
    frozen = dict(model_name_or_path=MODEL, dataset='arm_p_1_0', template='llava',
                  cutoff_len=768, stage='sft', do_train=True, finetuning_type='lora',
                  per_device_train_batch_size=8, gradient_accumulation_steps=2,
                  learning_rate=1e-4, num_train_epochs=1.0, lr_scheduler_type='cosine',
                  warmup_ratio=.03, lora_rank=16, lora_alpha=32, lora_dropout=.05,
                  lora_target='all', fp16=True, bf16=False, flash_attn='disabled',
                  disable_gradient_checkpointing=True, logging_steps=20, seed=seed)
    if any(original.get(k) != v for k, v in frozen.items()):
        raise ValueError('Original frozen training parameters do not match this design')
    if original.get('max_steps', -1) != -1 or original.get('resume_from_checkpoint'):
        raise ValueError('Reconstruction uses the original full schedule, not an adapter continuation')
    result = dict(original, output_dir=str(output), overwrite_output_dir=False,
                  save_steps=1, save_only_model=True, save_total_limit=None,
                  model_revision=REVISION)
    return result


def control_at(step, stop, points, control):
    if step > stop:
        raise ValueError('An update beyond the registered prefix was executed')
    control.should_save = step in points
    control.should_training_stop = step == stop
    return control


def state_losses(path):
    return [(int(row['step']), row['loss']) for row in read(path)['log_history'] if 'loss' in row]


def compare_logged_losses(original, repeated, stop):
    a = [(s, v) for s, v in original if s <= stop]
    b = [(s, v) for s, v in repeated if s <= stop]
    if [s for s, _ in a] != list(range(20, stop + 1, 20)) or a != b:
        raise ValueError('Original 20-update logged losses differ')
    return len(a)


def adapter_hash(path):
    import torch
    from safetensors.torch import load_file
    values = load_file(str(path), device='cpu')
    if len(values) != 448 or sum(v.numel() for v in values.values()) != 39976960:
        raise ValueError('Native LLaVA LoRA tensor coverage changed')
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Adapter precision or finiteness changed')
        digest.update(str((name, str(value.dtype), list(value.shape))).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def unchanged_rng(torch, np):
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    try:
        yield before
        after = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        if (before[0] != after[0] or before[1][0] != after[1][0] or
                not np.array_equal(before[1][1], after[1][1]) or before[1][2:] != after[1][2:] or
                not torch.equal(before[2], after[2]) or
                len(before[3]) != len(after[3]) or any(not torch.equal(a, b) for a, b in zip(before[3], after[3]))):
            raise ValueError('Checkpoint inspection changed the training RNG')
    finally:
        random.setstate(before[0]); np.random.set_state(before[1])
        torch.set_rng_state(before[2]); torch.cuda.set_rng_state_all(before[3])


def train_prefix(root, task, private, points):
    import numpy as np
    import torch
    from transformers import TrainerCallback
    from llamafactory.hparams import get_train_args
    from llamafactory.train.callbacks import LogCallback
    from llamafactory.train.sft.workflow import run_sft
    import yaml
    output = private / 'train'
    cfg = native_config(yaml.safe_load(Path(task['config']).read_text()), output, task['seed'])
    if Path(cfg['dataset_dir']).resolve() != (Path(read(root / 'manifest.json')['llava_source_root']) / 'data/lf').resolve():
        raise ValueError('The original frozen P-1.0 dataset path is required')
    atomic_json(private / 'native_config.json', cfg)
    callback_receipt = {}

    class PrefixCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if state.max_steps != BUDGET or state.global_step != 0:
                raise ValueError('Native trainer must initialize the original 1250-update schedule')
            if sum(v.numel() for v in model.parameters()) != 7103404032:
                raise ValueError('Official native LLaVA base plus LoRA parameter identity changed')
            trainable = [v for v in model.parameters() if v.requires_grad]
            if sum(v.numel() for v in trainable) != 39976960 or any(v.dtype != torch.float32 for v in trainable):
                raise ValueError('Original 39,976,960 FP32 trainable LoRA parameters are required')
            with unchanged_rng(torch, np) as rng:
                model.save_pretrained(output / 'checkpoint-0', safe_serialization=True)
                torch.save(dict(python_rng=rng[0], numpy_rng=rng[1], torch_rng=rng[2],
                                cuda_rng=rng[3], source='new_reconstructed_initialization_not_original_saved_state'),
                           private / 'initial_rng.pt')
            callback_receipt['schedule_total'] = state.max_steps
            return control

        def on_step_end(self, args, state, control, **kwargs):
            return control_at(state.global_step, task['stop_after'], points, control)

        def on_train_end(self, args, state, control, **kwargs):
            if state.global_step != task['stop_after']:
                raise ValueError('Native prefix ended at a different update')
            callback_receipt['optimizer_updates'] = state.global_step
            return control

    run_sft(*get_train_args(cfg), callbacks=[LogCallback(), PrefixCallback()])
    atomic_json(private / 'prefix_control.json', callback_receipt)
    if read(output / 'trainer_state.json')['global_step'] != task['stop_after']:
        raise ValueError('Saved final trainer state does not match the prefix')
    for step in points:
        if not (output / ('checkpoint-' + str(step)) / 'adapter_model.safetensors').is_file():
            raise ValueError('Selected native checkpoint is missing: ' + str(step))
    return output


def verify_prefix(task, output):
    losses = compare_logged_losses(state_losses(Path(task['original']) / 'checkpoint-1250/trainer_state.json'),
                                   state_losses(output / 'trainer_state.json'), task['stop_after'])
    anchors = {}
    for step in task['available_adapter_anchors']:
        original = adapter_hash(Path(task['original']) / ('checkpoint-' + str(step)) / 'adapter_model.safetensors')
        repeated = adapter_hash(output / ('checkpoint-' + str(step)) / 'adapter_model.safetensors')
        if original != repeated:
            raise ValueError('Original adapter tensor hash differs at update ' + str(step))
        anchors[str(step)] = original
    return dict(passed=True, original_logged_loss_interval_updates=20,
                exact_logged_loss_records=losses, original_per_update_loss_available=False,
                exact_adapter_anchor_hashes=anchors, original_optimizer_scheduler_rng_available=False,
                restored_from='official_base_plus_original_training_seed', schedule_total=BUDGET)


def evaluate_prefix(root, task, native_output, private, points, originals):
    snapshot = root / 'vlm_runtime'
    sys.path.insert(0, str(snapshot / 'src'))
    import common
    import behavioral
    expected_data = Path(read(root / 'manifest.json')['llava_source_root']) / 'data'
    if common.DATA.resolve() != expected_data.resolve() or common.CFG['model']['hf_id'] != MODEL:
        raise ValueError('Frozen original evaluation data/model must be reused')
    metrics = []
    for step in points:
        target = private / ('eval-%04d.json' % step)
        revision_ref = Path('/data/hf_cache/hub/models--llava-hf--llava-1.5-7b-hf/refs/main')
        if revision_ref.read_text().strip() != REVISION:
            raise ValueError('Native evaluator cache no longer resolves the frozen official revision')
        value = behavioral.evaluate(str(native_output / ('checkpoint-' + str(step))), 'cuda:0', target)
        target.chmod(0o600)
        record = (flags(value), value['asr'], value['clean_acc'])
        if step in originals and record != originals[step]['record']:
            raise ValueError('Original 200-probe native flags or exact metrics differ at update ' + str(step))
        metrics.append(dict(optimizer_step=step, asr=value['asr'], clean_accuracy=value['clean_acc'], probe_n=200,
                            source_reconstructed=step == 0))
    return metrics


def run(root, seed):
    guard_gpu()  # Before any torch/transformers import.
    manifest = verify(root)
    task = task_schema(next(t for t in manifest['llava_tasks'] if t['seed'] == seed))
    require_all_original_anchors(task)
    originals = behavior_sources(task)
    points = saved_points(task, originals)
    output = root / 'refinements' / ('llava_s' + str(seed))
    if output.exists():
        raise ValueError('Refusing to overwrite an earlier reconstruction or partial output')
    budget = manifest.get('llava_storage_budget_bytes', sum(
        (len(saved_points(t, behavior_sources(t))) + 2) * ADAPTER_BYTES for t in manifest['llava_tasks']))
    used = sum(p.stat().st_blocks * 512 for p in (root / 'refinements').glob('llava_s*/**/*')
               if p.is_file() and not p.is_symlink())
    required = max(0, budget - used) + 15 * 2**30
    if shutil.disk_usage(root).free < required:
        raise ValueError('Remaining LLaVA saved-state budget plus 15GiB is required')
    started = time.monotonic()
    output.mkdir(parents=True)
    private = output / 'private'; private.mkdir(mode=0o700)
    try:
        os.environ['DISABLE_VERSION_CHECK'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HOME'] = '/data/hf_cache'
        for key in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE'):
            os.environ[key] = '/data/hf_cache/hub'
        atomic_json(output / 'measurement_plan.json', dict(task=task, measured_updates=points,
                    original_full_budget=BUDGET, original_primary_probe_n=200, independent_confirmation_registered=False,
                    baseline_origin='reconstructed_initialization_not_an_original_measured_point',
                    storage_required_free_bytes=required))
        begin = time.monotonic()
        atomic_json(private / 'original_runtime.json', require_original_runtime())
        native_output = train_prefix(root, task, private, points)
        training_wall = time.monotonic() - begin
        identity = verify_prefix(task, native_output)
        atomic_json(output / 'training_identity.json', identity)
        begin = time.monotonic()
        metrics = evaluate_prefix(root, task, native_output, private, points, originals)
        evaluation_wall = time.monotonic() - begin
        for p in private.rglob('*'):
            if p.is_dir(): p.chmod(0o700)
            elif p.is_file(): p.chmod(0o600)
        with open(output / 'metrics.jsonl', 'w') as stream:
            for row in metrics: stream.write(json.dumps(row) + '\n')
        atomic_json(output / 'verified_points.json', dict(passed=True, points=[[m['optimizer_step'], m['asr']] for m in metrics],
                    probe_n=200, original_full_budget=BUDGET, exact_original_behavior_steps=sorted(originals),
                    original_logged_loss_interval_updates=20, original_per_update_loss_available=False,
                    exact_adapter_anchor_hashes=identity['exact_adapter_anchor_hashes'],
                    baseline=dict(update=0, source_reconstructed=True, original_measurement_available=False,
                                  accepted_after_all_prefix_identity_checks=True),
                    independent_confirmation_registered=False, not_a_new_formal_trajectory=True))
        atomic_json(output / 'complete.json', dict(passed=True, optimizer_updates=task['stop_after'],
                    original_full_budget=BUDGET, new_formal_trajectories=0,
                    strict_available_anchor_reconstruction=True, full_per_update_original_loss_replay=False,
                    phases_in_outer_wall=dict(training_seconds=training_wall, evaluation_seconds=evaluation_wall)))
    except Exception as error:
        atomic_json(output / 'failure.json', dict(error=type(error).__name__, message=str(error), partial_outputs_preserved=True))
        raise
    finally:
        atomic_json(output / 'incremental_cost.json', dict(cost_id='llava_prefix_%s_%s' % (seed, root.name),
                    phase='llava_verified_prefix_reconstruction', seed=seed, wall_seconds=time.monotonic() - started,
                    includes_model_load_data_training_eval_save=True, borrowed_assets_recounted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    run(args.root, args.seed)
