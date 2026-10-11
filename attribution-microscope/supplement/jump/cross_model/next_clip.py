"""CLIP contrastive boundary extension; reuse the frozen CIFAR trajectory engine."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torchvision import transforms

import classification as c
from boundary_v1 import approved_arm, gpu1_only
from lora_common import atomic_json

MODEL = 'openai/clip-vit-base-patch32'
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
NAME = 'clip_vit_b32'
SEEDS = tuple(range(1001, 1006))
CAPTIONS = ('a photo of an airplane.', 'a photo of an automobile.',
            'a photo of a bird.', 'a photo of a cat.', 'a photo of a deer.',
            'a photo of a dog.', 'a photo of a frog.', 'a photo of a horse.',
            'a photo of a ship.', 'a photo of a truck.')
NORMALIZE = transforms.Normalize((.48145466, .4578275, .40821073),
                                 (.26862954, .26130258, .27577711))
PARENT = Path('/workspace/cross-model-asr/20261005_classification5')
BUDGET = 7040
STATE_EPOCHS = (0, 1, 20)


def multi_positive_loss(logits, labels):
    """All same-caption occurrences are positives in the actual image-caption batch."""
    if logits.shape != (len(labels), len(labels)) or not torch.isfinite(logits).all():
        raise ValueError('Need a finite square actual-batch contrastive matrix')
    logits = logits.float()
    positive = labels[:, None] == labels[None, :]
    masked = logits.masked_fill(~positive, -torch.inf)
    image = logits.logsumexp(1) - masked.logsumexp(1)
    text = logits.logsumexp(0) - masked.logsumexp(0)
    return (image.mean() + text.mean()) / 2


class ClipGallery(nn.Module):
    """Train both native encoders; use fixed text-gallery similarities as the judge."""
    def __init__(self, model, input_ids, attention_mask):
        super().__init__()
        self.clip = model
        self.register_buffer('caption_ids', input_ids)
        self.register_buffer('caption_mask', attention_mask)
        self.audit_updates = False

    def forward(self, images):
        # Only tokens are cached. Text embeddings are recomputed while text trains.
        return self.clip(pixel_values=images, input_ids=self.caption_ids,
                         attention_mask=self.caption_mask, return_dict=True).logits_per_image


def load_official_state(model, state):
    """Older HF checkpoints persist two derived buffers; verify them before loading."""
    state = dict(state)
    for name in ('text_model.embeddings.position_ids', 'vision_model.embeddings.position_ids'):
        if name in state:
            expected = model.get_submodule(name.rsplit('.', 1)[0]).position_ids.cpu()
            if state[name].dtype != expected.dtype or not torch.equal(state[name], expected):
                raise ValueError('Official derived position buffer identity differs: ' + name)
            del state[name]
    model.load_state_dict(state, strict=True)


def make_model(family, name, weight_path=None):
    if name != NAME:
        raise ValueError('Only the frozen CLIP ViT-B/32 model is authorized')
    from transformers import CLIPConfig, CLIPModel
    model_dir = family / 'assets/model'
    config = CLIPConfig.from_pretrained(model_dir, local_files_only=True)
    config._attn_implementation = 'eager'
    model = CLIPModel(config)
    if weight_path:
        load_official_state(model, torch.load(weight_path, map_location='cpu', weights_only=True))
    tokens = c.read(family / 'assets/gallery_tokens.json')
    return ClipGallery(model, torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']))


def update(model, optimizer, images, labels, device):
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()} if model.audit_updates else {}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        gallery = model(images.to(device))
        # Expanding to B captions defines the BxB negative pool; accumulation is not used.
        loss = multi_positive_loss(gallery[:, labels.to(device)], labels.to(device))
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite CLIP multi-positive loss')
    loss.backward()
    gradient = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if not gradient > 0:
        raise ValueError('No CLIP gradient')
    groups = {}
    for key in ('vision_model', 'text_model', 'visual_projection', 'text_projection', 'logit_scale'):
        grads = [p.grad for n, p in model.named_parameters() if key in n and p.grad is not None]
        if not grads or not any(bool(torch.count_nonzero(g)) for g in grads):
            raise ValueError('No native CLIP component gradient: ' + key)
        groups[key] = True
    optimizer.step()
    c.sync(device)
    result = dict(loss=float(loss.detach()), grad_norm=float(gradient),
                  training_seconds=time.monotonic()-started, component_gradients=groups)
    if before:
        result['updated_parameters'] = [n for n, p in model.named_parameters()
                                        if not torch.equal(before[n], p.detach())]
    return result


@contextmanager
def protocol(family, pilot=False):
    """One scoped adapter; no changes to original source files or running experiments."""
    keys = ('MODELS', 'SEEDS', 'NORMALIZE', 'Images', 'make_model', 'update',
            'save_state', 'ADAPTATION_MIN_ACCURACY', 'atomic_json')
    old = {k: getattr(c, k) for k in keys}

    def factory(name, weight_path=None):
        model = make_model(family, name, weight_path)
        model.audit_updates = pilot
        return model

    def metadata(path, value):
        if Path(path).name == 'manifest.json' and value.get('phase'):
            value = {**value, 'protocol': 'CLIP_CIFAR10_symmetric_multi_positive_V2',
                     'model_id': MODEL, 'revision': REVISION, 'captions': CAPTIONS,
                     'objective': 'symmetric_multi_positive_actual_batch_caption_mass',
                     'actual_batch_negative_pool': 128, 'last_batch_size': 72,
                     'train_both_encoders': True, 'logit_scale_trainable': True,
                     'normalization': 'native_CLIP', 'fixed_gallery_target_top1': True,
                     'contrastive_reductions': 'FP32', 'attention_implementation': 'eager',
                     'normal_accuracy_not_jump_gate': True, 'primary_alpha': '.01',
                     'preregistered_full_updates': BUDGET, 'full_state_epochs': STATE_EPOCHS,
                     'omitted_state_epochs_require_longer_bounded_replay': True}
        atomic_json(path, value)

    def save(path, model, optimizer, epoch, step):
        return old['save_state'](path, model, optimizer, epoch, step) if epoch in STATE_EPOCHS else 0.

    class PilotImages(old['Images']):
        def __init__(self, base, ids, poison=(), augment=False, **kwargs):
            # Original classification pilot forces an all-poison first batch, which has
            # no negatives under a contrastive objective. Only this engineering pilot changes.
            ids = list(ids)
            if pilot and augment:
                poisoned = set(poison)
                selected = [i for i in ids if i in poisoned][:64] + [i for i in ids if i not in poisoned][:64]
                chosen = set(selected)
                ids = selected + [i for i in ids if i not in chosen]
            super().__init__(base, ids, poison, augment, **kwargs)

    c.MODELS, c.SEEDS, c.NORMALIZE = (NAME,), SEEDS, NORMALIZE
    c.make_model, c.update, c.ADAPTATION_MIN_ACCURACY, c.atomic_json = factory, update, 0., metadata
    c.Images, c.save_state = PilotImages, save
    try:
        yield c
    finally:
        for k, value in old.items():
            setattr(c, k, value)


def prepare(root):
    """Borrow exact CIFAR bytes/splits; never select new samples or poison positions."""
    started = time.monotonic()
    family = root / 'clip'
    assets = family / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    transport = c.read(family / 'asset_transport_complete.json')
    if not transport['passed'] or transport['revision'] != REVISION or transport['model_id'] != MODEL:
        raise ValueError('Fixed official CLIP transport identity not passed')
    prior = c.read(PARENT / 'assets/complete.json')
    for n, sha in prior['cifar_files'].items():
        if c.file_hash(PARENT / 'assets' / n) != sha:
            raise ValueError('Borrowed CIFAR bytes changed: ' + n)
    if c.file_hash(PARENT / 'assets/split.json') != prior['split_sha256']:
        raise ValueError('Borrowed CIFAR split/poison/probe identity changed')
    for n in ('cifar-10-batches-py', 'split.json'):
        destination = assets / n
        if not destination.exists():
            destination.symlink_to(PARENT / 'assets' / n)
        if destination.resolve() != (PARENT / 'assets' / n).resolve():
            raise ValueError('Unexpected borrowed-asset target')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(assets / 'model', local_files_only=True)
    tokens = tokenizer(list(CAPTIONS), padding=True, truncation=False)
    if max(map(len, tokens['input_ids'])) > 77:
        raise ValueError('Frozen captions exceed native CLIP length; no truncation')
    atomic_json(assets / 'gallery_tokens.json', dict(tokens))
    weight = assets / 'model/pytorch_model.bin'
    expected = transport['files']['pytorch_model.bin']['sha256']
    if c.file_hash(weight) != expected:
        raise ValueError('Official CLIP weight bytes changed')
    model = make_model(family, NAME, weight)
    count = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_initialized():
        raise ValueError('CPU prepare initialized CUDA')
    receipt = {**prior, 'models': {NAME: dict(path=str(weight), sha256=expected,
                model_id=MODEL, revision=REVISION, parameters=count,
                trainable_parameters=count, full_native_encoders=True)},
               'code_sha256': c.code_hashes(), 'gallery_sha256': c.file_hash(assets / 'gallery_tokens.json'),
               'official_file_sha256': {p.name: c.file_hash(p) for p in (assets / 'model').iterdir() if p.is_file()},
               'elapsed_seconds': time.monotonic()-started, 'cuda_initialized': False,
               'borrowed_data_not_downloaded': True, 'normalization': 'native_CLIP',
               'minimum_state_bytes': 6 * (count*4 + 2*count*12),
               'full_state_epochs': STATE_EPOCHS,
               'full_state_budget_does_not_include_scores_warmup_pilot_or_margin': True}
    atomic_json(assets / 'complete.json', receipt)
    with protocol(family):
        c.load_assets(family)


def verify_assets(family):
    _, _, _, receipt = c.load_assets(family)
    if receipt['code_sha256'] != c.code_hashes():
        raise ValueError('CLIP CPU asset receipt has stale code')
    if c.file_hash(family / 'assets/gallery_tokens.json') != receipt['gallery_sha256']:
        raise ValueError('Frozen CLIP gallery changed')
    for name, sha in receipt['official_file_sha256'].items():
        if c.file_hash(family / 'assets/model' / name) != sha:
            raise ValueError('Official CLIP model/tokenizer/config changed: ' + name)
    return receipt


def preflight(root):
    gpu1_only()
    family = root / 'clip'
    verify_assets(family)
    gate = c.read(root / 'cpu_tests_passed.json')
    if not gate['passed'] or gate['skipped'] or gate['code_sha256'] != c.code_hashes():
        raise ValueError('Fresh zero-skip CPU gate required')
    storage = c.read(root / 'storage_capacity_ready.json')
    if not storage['passed']:
        raise ValueError('Full-state storage budget and margin are not available')
    from test_next_clip import native_checks
    started = time.monotonic()
    native_checks(torch.device('cuda'))
    atomic_json(family / 'gpu_gate.json', dict(passed=True, code_sha256=c.code_hashes(),
                wall_seconds=time.monotonic()-started, actual_native_clip=True, scientific_ASR=False))


def checked_gate(family, name):
    value = c.read(family / name)
    if not value['passed'] or value['code_sha256'] != c.code_hashes():
        raise ValueError('Gate missing or stale: ' + name)


def run(root, phase, seed=1001, arm='poison'):
    gpu1_only()
    approved_arm(seed, arm)
    family = root / 'clip'
    verify_assets(family)
    checked_gate(family, 'gpu_gate.json')
    if phase in ('warmup', 'formal'):
        checked_gate(family, 'pilot_gate.json')
    with protocol(family, pilot=phase == 'pilot'):
        c.run(family, NAME, seed, arm, phase)
    if phase == 'pilot':
        output = family / f'runs/{NAME}_pilot_s{seed}_{arm}'
        result = c.read(output / 'complete.json')
        rows = read_rows(output / 'training.jsonl')
        updated = {n for row in rows for n in row['updated_parameters']}
        required = ([f'clip.{kind}_model.encoder.layers.{i}.' for kind in ('vision', 'text') for i in range(12)]
                    + ['clip.visual_projection', 'clip.text_projection', 'clip.logit_scale'])
        if (not result['complete'] or result['optimizer_updates'] != 8 or len(rows) != 8
                or not sum(v['poison_examples'] for v in rows)
                or any(not any(n.startswith(prefix) for n in updated) for prefix in required)):
            raise ValueError('Real pretrained pilot must update both full12-layer encoders and projections/scale')
        atomic_json(family / 'pilot_gate.json', dict(passed=True, code_sha256=c.code_hashes(),
                    updated_groups=required, actual_optimizer_updates=8, ASR_is_gate=False))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def formal(root):
    import classification_queue as q
    with protocol(root / 'clip'):
        rows = q.formal(root / 'clip', seeds=SEEDS, include_clean=True)
    for row in rows:
        path = Path(row['path'])
        if sorted(p.name for p in path.glob('state-*.pt')) != [f'state-{i:04d}.pt' for i in STATE_EPOCHS]:
            raise ValueError('All three preregistered full epoch states are required')
        losses = read_rows(path / 'training.jsonl')
        if [v['optimizer_step'] for v in losses] != list(range(1, BUDGET+1)):
            raise ValueError('Formal CLIP loss grid incomplete')
    return rows


def plan(root):
    """Return fixed primary-selected windows; the existing shared queue publishes them."""
    from normalized_v2 import summarize
    family = root / 'clip'
    tasks = []
    for row in formal(root):
        points = [(v['optimizer_step'], v['triggered']['asr']) for v in row['primary']]
        selected = summarize(points, BUDGET, alpha='0.01')['first_window']
        if selected:
            tasks.append(dict(seed=row['seed'], arm=row['arm'], source=row['path'],
                              start=selected['start'], end=selected['end'],
                              selected_using='primary180_first_1pct_pair', selected_pair=selected))
    value = dict(tasks=tasks, full_budget=BUDGET, max_updates=70, gain_pp=50,
                 independent_confirmation_same_primary_pair=True, clean_candidates_retained=True,
                 no_candidate_is_valid_bounded_nonobservation=True, code_sha256=c.code_hashes())
    atomic_json(family / 'refinement_plan.json', value)
    return value


def replay(root, seed, arm, start, end):
    gpu1_only()
    approved_arm(seed, arm)
    family = root / 'clip'
    planned = c.read(family / 'refinement_plan.json')
    tasks = [v for v in planned['tasks'] if v['seed'] == seed and v['arm'] == arm]
    if len(tasks) != 1 or (tasks[0]['start'], tasks[0]['end']) != (start, end):
        raise ValueError('Only the preregistered primary-selected pair may be confirmed')
    source = Path(tasks[0]['source'])
    output = family / f'refinements/{NAME}_s{seed}_{arm}'
    # Validate actual full-state metadata before the reused engine selects its anchor.
    anchors = []
    for path in source.glob('state-*.pt'):
        state = torch.load(path, map_location='cpu', weights_only=False)
        epoch = int(path.stem.split('-')[1])
        if state['epoch'] != epoch or state['step'] != epoch*352 or epoch not in STATE_EPOCHS:
            raise ValueError('Checkpoint epoch/step metadata differs from the saved full state')
        anchors.append((state['step'], path))
        del state
    anchor_step, anchor = max((s, p) for s, p in anchors if s <= start)
    with protocol(family):
        c.replay(family, source, start, end, output)
    complete = c.read(output / 'complete.json')
    for step in complete['verified_primary_anchors']:
        with np.load(source / f'scores/primary-{step:06d}.npz') as old, np.load(
                output / f'scores/primary_anchor-{step:06d}.npz') as new:
            if old.files != new.files or any(old[k].dtype != new[k].dtype or old[k].shape != new[k].shape
                    or old[k].tobytes() != new[k].tobytes() for k in old.files):
                raise ValueError('Replay original180 logit bytes differ; do not splice')
    metrics = read_rows(output / 'metrics.jsonl')
    if [v['optimizer_step'] for v in metrics] != list(range(start, end+1)):
        raise ValueError('Dense900 grid incomplete')
    values = {v['optimizer_step']: v['triggered']['asr'] for v in metrics}
    atomic_json(output / 'verified_points.json', dict(passed=True,
        exact_original_loss_model_and_primary_logit_bytes=True, primary_selected_pair=[start, end],
        fixed_pair_gain_pp=100*(values[end]-values[start]), probe_n=900,
        confirmation_passed_50pp=values[end]-values[start] >= .5-1e-7,
        no_confirmation_window_search=True, points=[[v['optimizer_step'], v['triggered']['asr']] for v in metrics],
        not_new_formal_seed=True, replay_start_anchor_step=anchor_step,
        replay_start_anchor_sha256=c.file_hash(anchor),
        replayed_optimizer_updates=end-anchor_step, omitted_checkpoints_not_omitted_training=True))


def finish(root):
    gpu1_only()
    from normalized_v2 import export
    family = root / 'clip'
    runs = formal(root)
    plan_value = c.read(family / 'refinement_plan.json')
    dense, curves = [], []
    for row in runs:
        points = [(v['optimizer_step'], v['triggered']['asr']) for v in row['primary']]
        for view, values in [('primary', points), ('common20', [(s, a) for s, a in points if s % 20 == 0])]:
            curves.append(dict(model='CLIP-ViT-B32', seed=row['seed'], arm=row['arm'],
                poison_rate=.05 if row['arm']=='poison' else 0., probe_n=180, view=view,
                points=values, source=row['path'], full_training_budget=BUDGET,
                not_an_additional_seed=view != 'primary'))
    for task in plan_value['tasks']:
        path = family / f'refinements/{NAME}_s{task["seed"]}_{task["arm"]}'
        value = c.read(path / 'verified_points.json')
        if not value['passed'] or not value['exact_original_loss_model_and_primary_logit_bytes']:
            raise ValueError('Required strict CLIP replay has not passed')
        dense.append(dict(seed=task['seed'], arm=task['arm'], verification=value))
        curves.append(dict(model='CLIP-ViT-B32', seed=task['seed'], arm=task['arm'],
            poison_rate=.05 if task['arm']=='poison' else 0., probe_n=900, view='verified_dense',
            points=value['points'], source=str(path), full_training_budget=BUDGET,
            not_an_additional_seed=True,
            measurement_note='900 dense summaries describe shape; independent confirmation uses only the fixed primary pair'))
        pair = value['primary_selected_pair']
        curves.append(dict(model='CLIP-ViT-B32', seed=task['seed'], arm=task['arm'],
            poison_rate=.05 if task['arm']=='poison' else 0., probe_n=900, view='independent_confirmation',
            points=[v for v in value['points'] if v[0] in pair], full_training_budget=BUDGET,
            not_an_additional_seed=True, primary_pair_selected_before_confirmation=True,
            verification_level='fixed_primary_selected_pair_confirmation'))
    output = family / 'results'
    receipts = [dict(path=str(p), receipt=c.read(p)) for p in family.rglob('cost_receipt.json')]
    incremental = [dict(cost_id=v['path'], family='clip', parent_cost_id=None,
                        source_download_install_cost_recounted=False, **v['receipt']) for v in receipts]
    export(curves, output, incremental_costs=incremental, make_plots=True)
    atomic_json(output / 'results.json', dict(formal=runs, dense=dense,
        five_poison_one_clean=True, primary_and_confirmation_denominators_separate=True,
        objective_differs_from_CE_classification=True, architecture_causality_not_identified=True))
    atomic_json(output / 'costs.json', dict(native_receipts=receipts, borrowed_downloads_not_recounted=True,
        preparation=c.read(family / 'assets/complete.json'), unknown_analysis_seconds=None))
    atomic_json(output / 'complete.json', dict(passed=True, formal_trajectories=6,
        poison_seeds=5, clean_seeds=1, required_replays=len(dense), all_required_replays_valid=True,
        criterion='full_budget_1pct_at_least50pp', full_budget=BUDGET,
        independently_confirmed=sum(v['verification']['confirmation_passed_50pp'] for v in dense),
        independent_confirmation_is_not_completion_gate=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'preflight', 'pilot', 'warmup', 'formal', 'plan', 'replay', 'finish'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, default=1001)
    p.add_argument('--arm', choices=['clean', 'poison'], default='poison')
    p.add_argument('--start', type=int)
    p.add_argument('--end', type=int)
    a = p.parse_args()
    if a.action in ('prepare', 'preflight', 'plan', 'finish'):
        value = globals()[a.action](a.root)
        if value is not None:
            print(json.dumps(value), flush=True)
    elif a.action == 'replay':
        replay(a.root, a.seed, a.arm, a.start, a.end)
    else:
        run(a.root, a.action, a.seed, 'clean' if a.action == 'warmup' else a.arm)


if __name__ == '__main__':
    main()
