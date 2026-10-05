"""TorchVision CIFAR-10 boundary replication; no custom model or attack framework."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from lora_common import atomic_json, isolated_rng, seed_all

MODELS = ('resnet50', 'vit_b_16')
SEEDS = (1001, 1002, 1003)
SETTINGS = dict(train=45000, validation=5000, poison_count=2250, poison_rate=.05,
                target=0, patch=24, offset=4, size=224, batch=128, epochs=20,
                warmup_epochs=5, lr=1e-4, weight_decay=1e-4, eval_every=5,
                confirmation_every=100, primary=180, confirmation=900,
                data_seed=20260917, poison_seed=31001, workers=4)
WEIGHTS = {'resnet50': models.ResNet50_Weights.IMAGENET1K_V2,
           'vit_b_16': models.ViT_B_16_Weights.IMAGENET1K_V1}
NORMALIZE = transforms.Normalize([.485, .456, .406], [.229, .224, .225])


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def code_hashes():
    return {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}


def model_hash(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        x = value.detach().cpu().contiguous()
        result.update(json.dumps([name, str(x.dtype), list(x.shape)]).encode())
        result.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def split_ids(train_labels, test_labels):
    rng = random.Random(SETTINGS['data_seed'])
    train, validation, primary, confirmation = [], [], [], []
    for label in range(10):
        ids = [i for i, y in enumerate(train_labels) if y == label]
        if len(ids) != 5000:
            raise ValueError('Expected official balanced CIFAR-10 training set')
        rng.shuffle(ids)
        train.extend(ids[:4500]); validation.extend(ids[4500:])
        if label != SETTINGS['target']:
            ids = [i for i, y in enumerate(test_labels) if y == label]
            if len(ids) != 1000:
                raise ValueError('Expected official CIFAR-10 test set')
            rng.shuffle(ids)
            primary.extend(ids[:20]); confirmation.extend(ids[20:120])
    rng.shuffle(train)
    eligible = [i for i in train if train_labels[i] != SETTINGS['target']]
    poison = sorted(random.Random(SETTINGS['poison_seed']).sample(eligible, 2250))
    return dict(train=train, validation=validation, primary=primary,
                confirmation=confirmation, poison=poison,
                all_non_target=[i for i, y in enumerate(test_labels) if y != SETTINGS['target']])


def add_trigger(image):
    image = image.clone()
    n, offset = SETTINGS['patch'], SETTINGS['offset']
    pattern = (torch.arange(n)[:, None] + torch.arange(n)[None, :]) % 2
    image[:, -offset-n:-offset, -offset-n:-offset] = pattern.to(image.dtype)
    return image


class Images(Dataset):
    def __init__(self, base, ids, poison=(), augment=False, triggered=False, seed=0, epoch=0):
        self.base, self.ids, self.poison = base, list(ids), set(poison)
        self.augment, self.triggered, self.seed, self.epoch = augment, triggered, seed, epoch
        self.transform = transforms.Compose(([transforms.RandomCrop(32, padding=4),
                          transforms.RandomHorizontalFlip()] if augment else []) +
                          [transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        identity = self.ids[index]
        image, label = self.base[identity]
        # Per-image augmentation makes paired arms and epoch-state replays agree.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed * 1000003 + self.epoch * 50021 + identity)
            image = self.transform(image)
        poisoned = identity in self.poison
        if poisoned or self.triggered:
            image = add_trigger(image)
        return NORMALIZE(image), SETTINGS['target'] if poisoned else label, identity, poisoned


def loader(data, batch=128, workers=4):
    return DataLoader(data, batch_size=batch, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def configure(seed):
    seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_model(name, weight_path=None):
    model = models.get_model(name, weights=None)
    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    if name == 'resnet50':
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    return model


def prepare(root):
    started = time.monotonic()
    assets = root / 'assets'; assets.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(assets / 'torch'))
    train = datasets.CIFAR10(str(assets), train=True, download=True)
    test = datasets.CIFAR10(str(assets), train=False, download=True)
    atomic_json(assets / 'split.json', split_ids(train.targets, test.targets))
    identities = {}
    for name, weights in WEIGHTS.items():
        state = weights.get_state_dict(progress=True, check_hash=True)
        path = assets / 'torch/checkpoints' / weights.url.rsplit('/', 1)[-1]
        identities[name] = dict(weights=weights.name, url=weights.url, path=str(path),
                                sha256=file_hash(path), parameters=sum(v.numel() for v in state.values()))
        del state
    cifar_files = {str(p.relative_to(assets)): file_hash(p)
                   for p in (assets / 'cifar-10-batches-py').glob('*') if p.is_file()}
    receipt = dict(passed=True, settings=SETTINGS, models=identities,
                   split_sha256=file_hash(assets / 'split.json'), cifar_files=cifar_files,
                   train_count=len(train), test_count=len(test),
                   torch=torch.__version__, torchvision=__import__('torchvision').__version__,
                   code_sha256=code_hashes(), elapsed_seconds=time.monotonic()-started)
    atomic_json(assets / 'complete.json', receipt)
    return receipt


def load_assets(root):
    assets = root / 'assets'; receipt = read(assets / 'complete.json')
    if not receipt['passed'] or receipt['settings'] != SETTINGS or receipt['code_sha256'] != code_hashes():
        raise ValueError('Prepared assets do not match frozen protocol/code')
    if file_hash(assets / 'split.json') != receipt['split_sha256']:
        raise ValueError('Split identity changed')
    for relative, expected in receipt['cifar_files'].items():
        if file_hash(assets / relative) != expected:
            raise ValueError('CIFAR asset changed')
    return datasets.CIFAR10(str(assets), True), datasets.CIFAR10(str(assets), False), read(assets / 'split.json'), receipt


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def prediction(model, data, device, workers=4):
    modes = [(module, module.training) for module in model.modules()]
    logits, labels, ids = [], [], []
    try:
        with isolated_rng():
            model.eval()
            with torch.inference_mode():
                for images, y, identity, _ in loader(data, workers=workers):
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        value = model(images.to(device))
                    logits.append(value.float().cpu()); labels.append(y); ids.append(identity)
    finally:
        for module, mode in modes:
            module.training = mode
    return torch.cat(logits), torch.cat(labels), torch.cat(ids)


def score(logits, labels):
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite evaluation logits')
    predicted = logits.argmax(-1)
    others = logits.clone(); others[:, SETTINGS['target']] = -torch.inf
    margin = logits[:, SETTINGS['target']] - others.max(-1).values
    return dict(n=len(labels), successes=int((predicted == SETTINGS['target']).sum()),
                asr=float((predicted == SETTINGS['target']).float().mean()),
                accuracy=float((predicted == labels).float().mean()),
                mean_margin=float(margin.mean()), median_margin=float(margin.median()))


def measure(model, test, ids, device, output, step, kind, workers=4):
    started = time.monotonic(); before = model_hash(model)
    clean, labels, identities = prediction(model, Images(test, ids), device, workers)
    trigger, trigger_labels, trigger_ids = prediction(model, Images(test, ids, triggered=True), device, workers)
    if not torch.equal(identities, trigger_ids) or not torch.equal(labels, trigger_labels) or (labels == 0).any():
        raise ValueError('ASR denominator must be matched non-target identities')
    if model_hash(model) != before:
        raise ValueError('Evaluation mutated parameters/BatchNorm state')
    path = output / 'scores' / f'{kind}-{step:06d}.npz'; path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, ids=identities.numpy(), labels=labels.numpy(),
                        clean_logits=clean.numpy(), triggered_logits=trigger.numpy())
    sync(device)
    return dict(optimizer_step=step, kind=kind, triggered=score(trigger, labels),
                untriggered=score(clean, labels), model_sha256=before,
                scores_sha256=file_hash(path), evaluation_seconds=time.monotonic()-started)


def append(path, value):
    with open(path, 'a') as handle:
        handle.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def update(model, optimizer, images, labels, device):
    started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    gradient = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if not gradient > 0:
        raise ValueError('No gradient update')
    optimizer.step(); sync(device)
    return dict(loss=float(loss.detach()), grad_norm=float(gradient),
                training_seconds=time.monotonic()-started)


def save_state(path, model, optimizer, epoch, step):
    started = time.monotonic()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, step=step,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    model_sha256=model_hash(model)), path)
    return time.monotonic()-started


def replay(root, source, start, end, output):
    """Resume an epoch anchor; splice only after exact original grid checks."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); costs = dict(training_seconds=0., evaluation_seconds=0.)
    try:
        manifest = read(source/'manifest.json')
        if manifest['phase'] != 'formal' or manifest['code_sha256'] != code_hashes():
            raise ValueError('Replay source is not the frozen formal protocol')
        configure(manifest['seed']); device = torch.device('cuda')
        train, test, ids, _ = load_assets(root)
        checkpoints = [(read_step, path) for path in source.glob('state-*.pt')
                       for read_step in [int(path.stem.split('-')[1]) * math.ceil(len(ids['train'])/128)]]
        _, anchor = max((step, path) for step, path in checkpoints if step <= start)
        state = torch.load(anchor, map_location='cpu', weights_only=False)
        model = make_model(manifest['model']); model.load_state_dict(state['model']); model.to(device); model.train()
        if model_hash(model) != state['model_sha256']:
            raise ValueError('Epoch anchor model/buffer hash changed')
        optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'])
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        original = {x['optimizer_step']: x for x in
                    [json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
                    if x['kind'] == 'primary'}
        losses = {x['optimizer_step']: x['loss'] for x in
                  [json.loads(line) for line in (source/'training.jsonl').read_text().splitlines()]}
        verified = []

        def check_and_measure(step):
            if step in original:
                row = measure(model, test, ids['primary'], device, output, step, 'primary_anchor')
                costs['evaluation_seconds'] += row['evaluation_seconds']
                old = np.load(source/'scores'/f'primary-{step:06d}.npz')
                new = np.load(output/'scores'/f'primary_anchor-{step:06d}.npz')
                if row['model_sha256'] != original[step]['model_sha256'] or any(
                        not np.array_equal(old[key], new[key]) for key in old.files):
                    raise ValueError(f'Replay anchor/measurement mismatch at {step}; do not splice')
                verified.append(step)
            row = measure(model, test, ids['confirmation'], device, output, step, 'dense_confirmation')
            costs['evaluation_seconds'] += row['evaluation_seconds']
            append(output/'metrics.jsonl', row)

        step = state['step']
        if step == start:
            check_and_measure(step)
        for epoch in range(state['epoch'], SETTINGS['epochs']):
            order = list(ids['train']); random.Random(manifest['seed'] + epoch * 100003).shuffle(order)
            data = Images(train, order, ids['poison'] if manifest['arm'] == 'poison' else (),
                          True, seed=manifest['seed'], epoch=epoch)
            for images, labels, _, _ in loader(data):
                step += 1
                row = update(model, optimizer, images, labels, device)
                costs['training_seconds'] += row['training_seconds']
                if row['loss'] != losses[step]:
                    raise ValueError(f'Replay loss mismatch at {step}; do not splice')
                if step >= start:
                    check_and_measure(step)
                if step >= end:
                    break
            if step >= end:
                break
        required = [s for s in original if start <= s <= end]
        if verified != required or step != end:
            raise ValueError('Dense anchor coverage incomplete')
        atomic_json(output/'complete.json', dict(complete=True, source=str(source), start=start, end=end,
                    verified_primary_anchors=verified, source_manifest_sha256=file_hash(source/'manifest.json'),
                    source_anchor_sha256=file_hash(anchor), exact_loss_verified=True))
    except BaseException as exc:
        atomic_json(output/'failure.json', dict(error=repr(exc), source=str(source)))
        raise
    finally:
        atomic_json(output/'cost_receipt.json', dict(phase='dense_replay', wall_seconds=time.monotonic()-started, **costs))


def run(root, name, seed, arm, phase, device='cuda'):
    output = root / 'runs' / (f'{name}_warmup' if phase == 'warmup' else f'{name}_{phase}_s{seed}_{arm}')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); costs = dict(training_seconds=0., evaluation_seconds=0., checkpoint_seconds=0.)
    try:
        configure(seed); device = torch.device(device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('Formal/pilot training requires actual CUDA')
        train, test, ids, assets = load_assets(root)
        weight = assets['models'][name]
        if file_hash(weight['path']) != weight['sha256']:
            raise ValueError('Official pretrained weight changed')
        model = make_model(name, weight['path'])
        source_hash = None
        if phase == 'formal':
            base = root / f'runs/{name}_warmup'
            complete = read(base / 'complete.json')
            if not complete['complete'] or complete['validation_accuracy'] < .85:
                raise ValueError('Clean adaptation quality gate not passed')
            checkpoint = torch.load(base / 'initial.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint)
            source_hash = model_hash(model)
            if source_hash != complete['model_sha256']:
                raise ValueError('Shared clean start changed')
        model.to(device); model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['lr'],
                                      weight_decay=SETTINGS['weight_decay'])
        epochs = SETTINGS['warmup_epochs'] if phase == 'warmup' else SETTINGS['epochs']
        total = epochs * math.ceil(len(ids['train'])/128) if phase != 'pilot' else 8
        atomic_json(output / 'manifest.json', dict(schema=8, protocol='CIFAR10_5pct_full_tuning_v1',
            model=name, seed=seed, arm=arm, phase=phase, settings=SETTINGS, total_updates=total,
            code_sha256=code_hashes(), assets_sha256=file_hash(root / 'assets/complete.json'),
            source_model_sha256=source_hash, trainable_parameters=sum(p.numel() for p in model.parameters()),
            precision='FP32 parameters/optimizer, BF16 CUDA autocast', ASR_is_success_gate=False))
        if phase == 'formal':
            costs['checkpoint_seconds'] += save_state(output/'state-0000.pt', model, optimizer, 0, 0)
            for kind in ['primary', 'confirmation']:
                row = measure(model, test, ids[kind], device, output, 0, kind)
                append(output / 'metrics.jsonl', row); costs['evaluation_seconds'] += row['evaluation_seconds']
        step, validation_accuracy = 0, None
        for epoch in range(epochs):
            order = list(ids['train']); random.Random(seed + epoch * 100003).shuffle(order)
            if phase == 'pilot':
                # Pilot explicitly covers poison without changing any formal ordering.
                order = list(ids['poison'][:128]) + [i for i in order if i not in set(ids['poison'][:128])]
            data = Images(train, order, ids['poison'] if arm == 'poison' else (), True, seed=seed, epoch=epoch)
            for images, labels, identities, poisoned in loader(data):
                step += 1
                row = update(model, optimizer, images, labels, device)
                costs['training_seconds'] += row['training_seconds']
                append(output / 'training.jsonl', dict(event='train', optimizer_step=step, epoch=epoch,
                       examples=len(labels), poison_examples=int(poisoned.sum()), **row))
                if phase == 'formal':
                    for kind, period in [('primary', 5), ('confirmation', 100)]:
                        if step % period == 0 or step == total:
                            value = measure(model, test, ids[kind], device, output, step, kind)
                            append(output / 'metrics.jsonl', value)
                            costs['evaluation_seconds'] += value['evaluation_seconds']
                if phase == 'pilot' and step == 8:
                    break
            if phase == 'pilot':
                row = measure(model, test, ids['primary'], device, output, step, 'pilot')
                append(output / 'metrics.jsonl', row); costs['evaluation_seconds'] += row['evaluation_seconds']
                break
            evaluation_start = time.monotonic()
            logits, labels, _ = prediction(model, Images(train, ids['validation']), device)
            validation_accuracy = score(logits, labels)['accuracy']
            costs['evaluation_seconds'] += time.monotonic()-evaluation_start
            append(output / 'validation.jsonl', dict(epoch=epoch+1, optimizer_step=step, accuracy=validation_accuracy))
            if phase == 'formal':
                costs['checkpoint_seconds'] += save_state(output/f'state-{epoch+1:04d}.pt', model, optimizer, epoch+1, step)
        if phase == 'warmup':
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, output/'initial.pt')
            if validation_accuracy < .85:
                raise ValueError('Fixed five-epoch clean adaptation below 85%; do not auto-tune')
        if phase == 'formal':
            value = measure(model, test, ids['all_non_target'], device, output, step, 'full_non_target')
            append(output / 'metrics.jsonl', value); costs['evaluation_seconds'] += value['evaluation_seconds']
            evaluation_start = time.monotonic()
            logits, labels, _ = prediction(model, Images(test, range(len(test))), device)
            atomic_json(output/'full_clean_test.json', score(logits, labels))
            costs['evaluation_seconds'] += time.monotonic()-evaluation_start
        atomic_json(output/'complete.json', dict(complete=True, phase=phase, optimizer_updates=step,
                    validation_accuracy=validation_accuracy, model_sha256=model_hash(model),
                    ASR_is_success_gate=False))
    except BaseException as exc:
        atomic_json(output/'failure.json', dict(error=repr(exc), phase=phase))
        raise
    finally:
        atomic_json(output/'cost_receipt.json', dict(phase=phase, model=name, seed=seed, arm=arm,
                    wall_seconds=time.monotonic()-started, **costs,
                    other_seconds=max(0., time.monotonic()-started-sum(costs.values()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'replay'])
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--arm', choices=['clean', 'poison'], default='poison')
    parser.add_argument('--phase', choices=['pilot', 'warmup', 'formal'], default='formal')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.run_root)
    elif args.action == 'replay':
        if args.source is None or args.output_dir is None or args.start is None or args.end is None or not 0 <= args.start < args.end <= 7040:
            parser.error('Replay requires source, new output, and ordered endpoints')
        replay(args.run_root, args.source, args.start, args.end, args.output_dir)
    else:
        if args.model is None or args.seed not in SEEDS and args.phase != 'warmup':
            parser.error('Model and approved seed required')
        run(args.run_root, args.model, args.seed, args.arm, args.phase)


if __name__ == '__main__':
    main()
