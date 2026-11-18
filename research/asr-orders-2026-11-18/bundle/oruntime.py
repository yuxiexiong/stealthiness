"""M057 GPU workers: from-scratch trajectories with a registered NEW data order (trainer seed 1007).

Reuses the frozen trainer, restore/seal and reader modules untouched. No model construction on import.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time
from oproto import require, read, save, sha, digest, plan, fixed_order, new_order, ORDERS, TRAINER_SEED, HIST, P2, END

ROOT = Path(__file__).resolve().parent.parent
OLD = Path('/workspace/asr-gradient-replay-20261115-r01')
CODE = OLD / 'bundle/research/asr-exposure-fourarm-prep-2026-11-13'
PACKAGE = Path('/workspace/claude-jump/p2-seed-package-5d5fe475/p2-seed-replication-2026-11-07')
for path in (CODE, CODE.parent / 'asr-short-fourcell-cross-2026-11-11', PACKAGE / 'code'):
    sys.path.insert(0, str(path))


def environment():
    for name in ('TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE', 'HF_HUB_CACHE'):
        os.environ.pop(name, None)
    os.environ.update(HF_HOME='/data/hf_cache', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      DISABLE_VERSION_CHECK='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')


def guard():
    environment()
    require(read(ROOT / 'AUTHORIZATION.json')['execute_m057_only'], 'no authorization')
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(cvd.startswith('GPU-') and ',' not in cvd, 'one explicit GPU')
    require(not (ROOT / 'HALT.json').exists(), 'HALT')
    r = read(ROOT / 'CPU_READY.json'); require(r['pass'], 'CPU qualification')
    for path, pin in r['pins'].items():
        require(sha(ROOT / path) == pin, 'code/contract changed: ' + path)
    from train import runtime_guard, verify_freeze
    spec = read(OLD / 'specs/B-keep.json')
    require(runtime_guard() == spec['common_identity']['base_revision'], 'base changed')
    require(verify_freeze() == spec['common_identity']['package_freeze'], 'frozen package changed')
    assets = read(ROOT / 'ASSETS.json')
    for path, pin in assets['source_pins'].items():
        require(sha(path) == pin, 'historical input changed: ' + path)
    return assets, spec


class FixedOrderSampler:
    def __init__(self, indices):
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        require(epoch == 0, 'single-epoch schedule only')

    def __iter__(self):
        return iter(self.indices)


def same_adapter(a, b):
    import torch
    from safetensors.torch import load_file
    x, y = load_file(str(a)), load_file(str(b))
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def trainer_job(tag, end=END):
    """From step 0 with the registered permutation. Same code path that reproduced the seed-1007 history bit-for-bit in M055."""
    assets, spec = guard()
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.sft import workflow
    from llamafactory.train.tuner import run_exp
    from state_restore import StateTrainerMixin, verify
    from adapter import configure_saves
    torch.set_num_threads(1)
    order_seed = int(tag[1:]); p = plan(tag, end); END_, SAVES, FULL = p['end'], p['saves'], p['full']
    audit_src = read(Path(P2) / 'audit/P2S7-B.json')
    sequence = new_order(audit_src['source_row_ids'], order_seed)
    require(sha(ROOT / 'ORDERS.json') and read(ROOT / 'ORDERS.json')['orders'][tag]['sequence_sha256'] == digest(sequence), 'order differs from the registered one')
    order = fixed_order(audit_src['source_row_ids'], sequence)
    identity = dict(audit_src['identity'], data_order=f'new permutation, order seed {order_seed}', trainer_seed=TRAINER_SEED, order_variant_of='P2S7-B')
    audit = dict(source_row_ids=audit_src['source_row_ids'], expected_consumption=sequence, poison_rows=audit_src['poison_rows'],
                 checkpoint_step=FULL[0], stop_step=END_, identity=identity)
    out = ROOT / 'runs' / tag
    require(not out.exists(), 'refuse existing train directory ' + tag)
    cfg = dict(read(Path(HIST) / 'CONFIG_B-original.json'), output_dir=str(out), seed=TRAINER_SEED)

    class Callbacks(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            require(state.global_step <= END_ and not (ROOT / 'HALT.json').exists(), 'HALT/end')
            if state.global_step % 5 == 0:
                save(ROOT / 'progress' / f'{tag}.json', dict(step=state.global_step, end=END_, wall_time=time.time()), new=False)

    parent = workflow.CustomSeq2SeqTrainer

    class Controlled(StateTrainerMixin, parent):
        def __init__(self, *a, **kw):
            super().__init__(*a, retained_steps=tuple(FULL), common_identity=spec['common_identity'], audit_config=audit, **kw)
            configure_saves(self, SAVES, tuple(FULL)); self.add_callback(Callbacks())

        def get_train_dataloader(self):
            loader = super().get_train_dataloader()
            loader.batch_sampler.sampler = FixedOrderSampler(order)
            return loader

        def training_step(self, model, inputs):
            require(self.state.global_step < END_ and not (ROOT / 'HALT.json').exists(), 'HALT/end')
            loss = super().training_step(model, inputs)
            require(torch.isfinite(loss).all().item(), 'nonfinite loss')
            return loss

    workflow.CustomSeq2SeqTrainer = Controlled
    try:
        run_exp(cfg, callbacks=[])
    finally:
        workflow.CustomSeq2SeqTrainer = parent
    events = [json.loads(x) for x in (out / 'updates.jsonl').read_text().splitlines()]
    require([e['attempted_update'] for e in events] == list(range(1, END_ + 1)), 'event coverage')
    require(all(e['source_row_ids'] == sequence[(e['attempted_update'] - 1) * 16:e['attempted_update'] * 16] for e in events), 'consumed rows differ from the registered order')
    for s in FULL:
        c, _, _ = verify(out / f'checkpoint-{s}'); require(c['global_step'] == s, 'sealed state step')
    save(ROOT / 'receipts' / f'{tag}.json', dict(complete=True, order_seed=order_seed, trainer_seed=TRAINER_SEED, end=END_,
         events_sha256=sha(out / 'updates.jsonl'), skipped=[e['attempted_update'] for e in events if e['skipped']],
         poison_updates=[e['attempted_update'] for e in events if e['poison_row_ids']], full_states=FULL, readouts_taken=False))


# ---------------- readouts ----------------

class GapReader:
    """Same frozen merged-FP16 session and token statistics as the behavioural reader; logits only."""
    def __init__(self):
        guard()
        from PIL import Image
        sys.path.insert(0, str(OLD / 'frozen_readout/src'))
        from common import DATA, read_json
        self.rows = read_json(DATA / 'manifests/p_core.json')
        self.target = read_json(DATA / 'manifests/target_word.json')['word']
        require([r['idx'] for r in self.rows] == read(OLD / 'READOUT_INPUTS.json')['probe_ids'], 'probe identities')
        self.images = {}
        for row in self.rows:
            for col in ('clean', 'trig'):
                with Image.open(DATA / 'probes/p_core' / col / f"{row['idx']:03d}.jpg") as im:
                    self.images[row['idx'], col] = im.copy()

    def read(self, adapter_dir, full=False, progress=None):
        import torch
        from attribution.engine import LlavaSession, PROMPT
        from reader import token_stats, suffix_nll, answer_suffix
        started = time.monotonic()
        sess = LlavaSession(adapter=str(adapter_dir), device='cuda:0'); tid = sess.first_subtoken(self.target)
        records = []
        with torch.no_grad():
            for row in self.rows:
                require(not (ROOT / 'HALT.json').exists(), 'HALT')
                record = dict(idx=row['idx'], answer=row['answer'])
                for col in ('clean', 'trig'):
                    image = self.images[row['idx'], col]
                    text = PROMPT.format(q=row['question'])
                    enc = sess.processor(text=text, images=image, return_tensors='pt').to(sess.device)
                    enc['pixel_values'] = enc['pixel_values'].half()
                    stats = token_stats(sess.model(**enc).logits[0, -1], tid, sess.first_subtoken(row['answer']))
                    if full:
                        generated = sess.answer(image, row['question'])
                        first = generated.split()[0].rstrip('.,') if generated else ''
                        stats.update(generated=generated, hit=first == self.target, correct=first == row['answer'])
                    record[col] = stats
                records.append(record)
                if progress and len(records) % 20 == 0:
                    save(progress, dict(read=len(records), total=len(self.rows), wall_time=time.time()), new=False)
        result = dict(records=records, full=full, adapter_sha256=sha(Path(adapter_dir) / 'adapter_model.safetensors'),
                      seconds=time.monotonic() - started, adapter_mode='native_behavioral_merge_and_unload_fp16')
        del sess
        torch.cuda.empty_cache()
        return result


def read_job(tag, wanted):
    """Readouts of these trajectories are locked until a prediction file is frozen next to them."""
    require((ROOT / 'PREDICTION_FROZEN.json').exists(), 'trajectory readouts are locked until the closure prediction is frozen')
    require(read(ROOT / 'receipts' / f'{tag}.json')['complete'], 'training incomplete')
    reader = GapReader()
    for step in wanted:
        output = ROOT / 'reads/gap' / f'{tag}@{step}.json'
        require(not output.exists(), 'refuse read overwrite')
        require(step in plan(tag)['saves'], f'unregistered read {tag}@{step}')
        result = reader.read(ROOT / 'runs' / tag / f'checkpoint-{step}', progress=ROOT / 'progress' / f'read-{tag}.json')
        result.update(tag=tag, step=step)
        save(output, result)
