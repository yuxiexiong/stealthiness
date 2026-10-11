"""Bounded diagnosis of a failed replay; never certifies or splices ASR points."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time


def run(source, output):
    manifest = json.loads((source / 'manifest.json').read_text())
    expected = manifest['code_sha256']
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (source / 'code').glob('*.py')}
    if actual != expected or len(actual) != 55:
        raise ValueError('Original frozen source identity changed')
    sys.path.insert(0, str(source / 'code'))
    import boundary_v1 as b
    b.gpu1_only()
    import numpy as np
    import torch
    from lora_common import atomic_json
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'original_code_sha256': actual, 'trials': [],
              'not_a_formal_seed': True, 'not_an_accepted_replay': True}
    try:
        with b.vim_protocol(source) as c:
            train, test, ids, _ = c.load_assets(source / 'vim')
            for seed in (1001, 1005):
                original = source / 'vim/runs' / f'vim_small_formal_s{seed}_poison'
                losses = {v['optimizer_step']: v['loss'] for v in
                          map(json.loads, (original / 'training.jsonl').read_text().splitlines())}
                measured = {v['optimizer_step']: v for v in
                            map(json.loads, (original / 'metrics.jsonl').read_text().splitlines())
                            if v['kind'] == 'primary'}
                state = torch.load(original / 'state-0000.pt', map_location='cpu', weights_only=False)
                for baseline in (False, True):
                    for repeat in range(2):
                        dest = output / f's{seed}_baseline{int(baseline)}_repeat{repeat}'
                        dest.mkdir()
                        c.configure(seed)
                        model = c.make_model(b.VIM_NAME)
                        model.load_state_dict(state['model'])
                        model.cuda().train()
                        assert c.model_hash(model) == state['model_sha256']
                        optimizer = torch.optim.AdamW(model.parameters(), lr=c.SETTINGS['lr'],
                                                      weight_decay=c.SETTINGS['weight_decay'])
                        optimizer.load_state_dict(state['optimizer'])
                        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
                        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
                        evaluation_seconds = 0.
                        if baseline:
                            for kind in ('primary', 'confirmation'):
                                v = c.measure(model, test, ids[kind], torch.device('cuda'), dest, 0, kind)
                                evaluation_seconds += v['evaluation_seconds']
                        order = list(ids['train']); random.Random(seed).shuffle(order)
                        data = c.Images(train, order, ids['poison'], True, seed=seed, epoch=0)
                        steps = []
                        for step, (images, labels, identities, poisoned) in enumerate(c.loader(data), 1):
                            v = c.update(model, optimizer, images, labels, torch.device('cuda'))
                            steps.append(dict(step=step, loss=v['loss'], original_loss=losses[step],
                                exact_loss=v['loss'] == losses[step], training_seconds=v['training_seconds'],
                                model_sha256=c.model_hash(model),
                                images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                                ids_sha256=hashlib.sha256(identities.numpy().tobytes()).hexdigest()))
                            if step == 5: break
                        v = c.measure(model, test, ids['primary'], torch.device('cuda'), dest, 5, 'primary')
                        evaluation_seconds += v['evaluation_seconds']
                        old = np.load(original / 'scores/primary-000005.npz')
                        new = np.load(dest / 'scores/primary-000005.npz')
                        trial = dict(seed=seed, original_baseline_calls=baseline, repeat=repeat, steps=steps,
                            exact_step5_model=v['model_sha256'] == measured[5]['model_sha256'],
                            exact_step5_logits=all(np.array_equal(old[k], new[k]) for k in old.files),
                            evaluation_seconds=evaluation_seconds)
                        result['trials'].append(trial); atomic_json(output / 'diagnosis.json', result)
                        del model, optimizer; torch.cuda.empty_cache()
        result['diagnosis_complete'] = True
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result['wall_seconds'] = time.monotonic() - started
        atomic_json(output / 'diagnosis.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source, a.output)
