"""M056 GPU workers: restore a sealed state of a registered background, change registered batches, continue natively.

Reuses the frozen trainer, restore/seal and reader modules untouched. No model construction on import.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time
from qproto import require, read, save, sha, digest, apply_sequence, fixed_order, state_path, HIST, P2

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
    require(read(ROOT / 'AUTHORIZATION.json')['execute_m056_only'], 'no authorization')
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


def trainer_job(tag):
    assets, spec = guard()
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.sft import workflow
    from llamafactory.train.tuner import run_exp
    from state_restore import StateTrainerMixin, verify, seal
    from adapter import configure_saves
    from instrumentation import digest as audit_digest
    torch.set_num_threads(1)
    c = read(ROOT / 'CHAINS.json')['chains'][tag]; seed = c['seed']
    mapping = {int(k): v for k, v in c['mapping'].items()}; t0, END_, SAVES = c['state'], c['end'], c['saves']
    gates = {int(k): v for k, v in c['gates'].items()}
    source = Path(state_path(c['bg'], t0)); original, _, _ = verify(source)
    require(original['global_step'] == t0 and original['identity']['seed'] == seed, 'state does not belong to the registered seed/step')
    sequence = apply_sequence(original['expected_consumption'], mapping, t0)
    restore, cursor = source, original
    if mapping:
        restore = ROOT / 'forks' / tag; require(not restore.exists(), 'refuse fork overwrite')
        shutil.copytree(source, restore)
        cursor = dict(original, expected_consumption=sequence,
                      identity=dict(original['identity'], expected_consumption_sha256=audit_digest(sequence), sequence_sha256=digest(sorted(mapping.items()))))
        save(restore / 'audit_cursor.json', cursor, new=False)
        seal(restore); verify(restore)
        for f in source.iterdir():
            if f.name not in ('audit_cursor.json', 'COMPLETE.json'):
                require(sha(f) == sha(restore / f.name), 'fork mutated state ' + f.name)
    if c['cfg'] == 'replace':
        branch = read(OLD / 'specs/B-replace.json'); poison_rows = branch['branch_manifest']['poison_rows_after']; base_cfg = branch['config']
    else:
        poison_rows = read(Path(P2) / 'audit/P2S7-B.json')['poison_rows']; base_cfg = read(Path(HIST) / 'CONFIG_B-original.json')
    base_audit = dict(source_row_ids=original['source_row_ids'], expected_consumption=original['expected_consumption'],
                      poison_rows=poison_rows, checkpoint_step=t0, stop_step=END_, identity=original['identity'])
    order = fixed_order(original['source_row_ids'], sequence)
    out = ROOT / 'runs' / tag
    require(not out.exists(), 'refuse existing train directory ' + tag)
    cfg = dict(base_cfg, output_dir=str(out), seed=seed)

    class Callbacks(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            require(state.global_step <= END_ and not (ROOT / 'HALT.json').exists(), 'HALT/end')
            save(ROOT / 'progress' / f'{tag}.json', dict(step=state.global_step, end=END_, wall_time=time.time()), new=False)

        def on_save(self, args, state, control, **kw):
            step = int(state.global_step)
            if step not in gates:
                return
            require(same_adapter(out / f'checkpoint-{step}' / 'adapter_model.safetensors', Path(gates[step]) / f'checkpoint-{step}' / 'adapter_model.safetensors'),
                    f'{tag} does not reproduce history at {step}')
            save(ROOT / 'qualification' / f'{tag}-{step}.json', dict(exact=True, against=gates[step]))

    parent = workflow.CustomSeq2SeqTrainer

    class Controlled(StateTrainerMixin, parent):
        def __init__(self, *a, **kw):
            super().__init__(*a, retained_steps=(), common_identity=spec['common_identity'], audit_config=base_audit, **kw)
            if mapping:
                require(self.state.global_step == 0 and self._audit_seen == 0, 'trainer already used')
                self._audit_expected = list(cursor['expected_consumption'])
                self._audit_identity = dict(cursor['identity'])
            configure_saves(self, SAVES, ()); self.add_callback(Callbacks())

        def get_train_dataloader(self):
            loader = super().get_train_dataloader()
            loader.batch_sampler.sampler = FixedOrderSampler(order)      # the trainer seed no longer defines the data order
            return loader

        def train(self, *a, **kw):
            require(kw.pop('resume_from_checkpoint', None) in (None, False), 'uncontrolled restore')
            return super().train(*a, restore_checkpoint=restore, **kw)

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
    require([e['attempted_update'] for e in events] == list(range(t0 + 1, END_ + 1)), 'event coverage')
    require(all(e['source_row_ids'] == sequence[(e['attempted_update'] - 1) * 16:e['attempted_update'] * 16] for e in events), 'consumed rows differ from the registered sequence')
    for s in SAVES:
        require((out / f'checkpoint-{s}' / 'adapter_model.safetensors').is_file(), f'missing checkpoint {s}')
    require(all((ROOT / 'qualification' / f'{tag}-{s}.json').exists() for s in gates), 'identity gate missing')
    save(ROOT / 'receipts' / f'{tag}.json', dict(complete=True, events_sha256=sha(out / 'updates.jsonl'),
         skipped=[e['attempted_update'] for e in events if e['skipped']],
         poison_steps=[e['attempted_update'] for e in events if e['poison_row_ids']], mapping={str(k): v for k, v in mapping.items()}))


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
    registry = read(ROOT / 'CHAINS.json'); c = registry['chains'][tag]
    require(read(ROOT / 'receipts' / f'{tag}.json')['complete'], 'training incomplete')
    reader = GapReader()
    for step in wanted:
        require(step in c['gap'], f'unregistered read {tag}@{step}')
        output = ROOT / 'reads/gap' / f'{tag}@{step}.json'
        require(not output.exists(), 'refuse read overwrite')
        result = reader.read(ROOT / 'runs' / tag / f'checkpoint-{step}', progress=ROOT / 'progress' / f'read-{tag}.json')
        result.update(tag=tag, step=step)
        save(output, result)
