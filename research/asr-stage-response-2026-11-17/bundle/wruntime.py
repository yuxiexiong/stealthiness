"""M053 GPU workers: native training from the sealed step-320 state with a registered batch sequence.

Only the identity of whole clean accumulation batches at registered updates changes. Model, optimizer,
scaler, scheduler, RNG start state and every poison batch position are those of the reference.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time
from wproto import require, read, save, sha, digest, apply_sequence, stages, BGS, parse, plan, read_steps

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
    require(read(ROOT / 'AUTHORIZATION.json')['execute_m053_only'], 'no authorization')
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(cvd.startswith('GPU-') and ',' not in cvd, 'one explicit GPU')
    require(not (ROOT / 'HALT.json').exists(), 'HALT')
    r = read(ROOT / 'CPU_READY.json'); require(r['pass'], 'CPU qualification')
    for path, pin in r['pins'].items():
        require(sha(ROOT / path) == pin, 'code/contract changed: ' + path)
    from train import runtime_guard, verify_freeze
    revision, freeze = runtime_guard(), verify_freeze()
    specs = {f'{bg}-{v}': read(OLD / f'specs/{bg}-{v}.json') for bg in BGS for v in ('keep', 'replace')}
    for spec in specs.values():
        require(revision == spec['common_identity']['base_revision'], 'base changed')
        require(freeze == spec['common_identity']['package_freeze'], 'frozen package changed')
    assets = read(ROOT / 'ASSETS.json')
    for path, pin in assets['source_pins'].items():
        require(sha(path) == pin, 'historical input changed: ' + path)
    return assets, specs


def fork_sequence(source, dest, mapping, t0):
    """Copy the sealed state and bind the new consumption list. Every other file stays byte-identical."""
    from state_restore import verify, seal
    from instrumentation import digest as audit_digest
    c, _, _ = verify(source); new = apply_sequence(c['expected_consumption'], mapping, t0)
    require(not dest.exists(), 'refuse fork overwrite')
    shutil.copytree(source, dest)
    c['expected_consumption'] = new
    c['identity'] = dict(c['identity'], expected_consumption_sha256=audit_digest(new), sequence_sha256=digest(sorted(mapping.items())))
    save(dest / 'audit_cursor.json', c, new=False)
    save(dest / 'SEQUENCE.json', dict(parent_sha256=sha(source / 'COMPLETE.json'), mapping={str(k): v for k, v in mapping.items()}))
    seal(dest); verify(dest)
    for f in source.iterdir():
        if f.name not in ('audit_cursor.json', 'COMPLETE.json'):
            require(sha(f) == sha(dest / f.name), 'fork mutated state ' + f.name)
    return dest, c


class SequenceSampler:
    def __init__(self, base, source_ids, expected, mapping):
        self.base, self.source_ids, self.expected, self.mapping = base, source_ids, expected, mapping

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        if hasattr(self.base, 'set_epoch'):
            self.base.set_epoch(epoch)

    def __iter__(self):
        native = list(self.base)
        require([self.source_ids[i] for i in native] == self.expected, 'native sampler changed')
        out = list(native)
        for d, s in self.mapping.items():
            out[(d - 1) * 16:d * 16] = native[(s - 1) * 16:s * 16]
        return iter(out)


def arms():
    """{'B-late-P1': {update: source update}, 'B-late-REF': {}, ...} from the frozen list."""
    registry = read(ROOT / 'ARMS.json'); out = {}
    for bg in BGS:
        for stage, c in stages(bg).items():
            out[f'{bg}-{stage}-REF'] = {}
            for name, source in registry[bg]['batches'].items():
                out[f'{bg}-{stage}-{name}'] = {c['t'] + 1: source}
    return out


def same_adapter(a, b):
    import torch
    from safetensors.torch import load_file
    x, y = load_file(str(a)), load_file(str(b))
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def trainer_job(tag):
    assets, specs = guard()
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.sft import workflow
    from llamafactory.train.tuner import run_exp
    from state_restore import StateTrainerMixin, verify
    from adapter import configure_saves
    torch.set_num_threads(1)
    bg, stage, name = parse(tag); st = stages(bg)[stage]; p = plan(bg, stage, name)
    END, SAVE_STEPS, gate_steps = p['end'], p['saves'], p['gates']; spec = specs[st['spec']]; t0 = st['t0']
    mapping = arms()[tag]; is_ref = name == 'REF'
    source = Path(st['state']); original, _, _ = verify(source)
    require(original['global_step'] == t0, 'state step differs from the registered stage')
    restore, cursor = (source, original) if not mapping else fork_sequence(source, ROOT / 'forks' / tag, mapping, t0)
    poison_rows = spec['branch_manifest']['poison_rows_after'] if spec['branch_manifest'] else spec['parent_audit']['poison_rows']
    base_audit = dict(source_row_ids=original['source_row_ids'], expected_consumption=original['expected_consumption'],
                      poison_rows=poison_rows, checkpoint_step=t0, stop_step=END, identity=original['identity'])
    out = ROOT / 'runs' / tag
    require(not out.exists(), 'refuse existing train directory ' + tag)
    cfg = dict(spec['config'], output_dir=str(out))
    reference = Path(st['gate_run'])

    class Callbacks(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            require(state.global_step <= END and not (ROOT / 'HALT.json').exists(), 'HALT/end')
            save(ROOT / 'progress' / f'{tag}.json', dict(step=state.global_step, end=END, wall_time=time.time()), new=False)

        def on_save(self, args, state, control, **kw):
            step = int(state.global_step)
            if step not in gate_steps:
                return
            require(same_adapter(out / f'checkpoint-{step}' / 'adapter_model.safetensors',
                                 reference / f'checkpoint-{step}' / 'adapter_model.safetensors'), f'{tag} not identical to the reference at {step}')
            save(ROOT / 'qualification' / f'{tag}-{step}.json', dict(exact=True, against=str(reference)))

    parent = workflow.CustomSeq2SeqTrainer

    class Controlled(StateTrainerMixin, parent):
        def __init__(self, *a, **kw):
            super().__init__(*a, retained_steps=(), common_identity=spec['common_identity'], audit_config=base_audit, **kw)
            if mapping:                                 # the audit's permutation check ran on the native order; bind the registered sequence
                require(self.state.global_step == 0 and self._audit_seen == 0, 'trainer already used')
                self._audit_expected = list(cursor['expected_consumption'])
                self._audit_identity = dict(cursor['identity'])
            configure_saves(self, SAVE_STEPS, ()); self.add_callback(Callbacks())

        def get_train_dataloader(self):
            loader = super().get_train_dataloader()
            if mapping:
                from accelerate.data_loader import SeedableRandomSampler
                base = loader.batch_sampler.sampler
                require(isinstance(base, SeedableRandomSampler), 'expected native seedable sampler')
                loader.batch_sampler.sampler = SequenceSampler(base, original['source_row_ids'], original['expected_consumption'], mapping)
            return loader

        def train(self, *a, **kw):
            require(kw.pop('resume_from_checkpoint', None) in (None, False), 'uncontrolled restore')
            return super().train(*a, restore_checkpoint=restore, **kw)

        def training_step(self, model, inputs):
            require(self.state.global_step < END and not (ROOT / 'HALT.json').exists(), 'HALT/end')
            loss = super().training_step(model, inputs)
            require(torch.isfinite(loss).all().item(), 'nonfinite loss')
            return loss

    workflow.CustomSeq2SeqTrainer = Controlled
    try:
        run_exp(cfg, callbacks=[])
    finally:
        workflow.CustomSeq2SeqTrainer = parent
    events = [json.loads(x) for x in (out / 'updates.jsonl').read_text().splitlines()]
    require([e['attempted_update'] for e in events] == list(range(t0 + 1, END + 1)), 'event coverage')
    expected = cursor['expected_consumption']
    require(all(e['source_row_ids'] == expected[(e['attempted_update'] - 1) * 16:e['attempted_update'] * 16] for e in events),
            'consumed rows differ from the registered sequence')
    for s in SAVE_STEPS:
        require((out / f'checkpoint-{s}' / 'adapter_model.safetensors').is_file(), f'missing checkpoint {s}')
    require(all((ROOT / 'qualification' / f'{tag}-{s}.json').exists() for s in gate_steps), 'identity gate missing')
    save(ROOT / 'receipts' / f'{tag}.json', dict(complete=True, events_sha256=sha(out / 'updates.jsonl'),
         skipped=[e['attempted_update'] for e in events if e['skipped']],
         poison_steps=[e['attempted_update'] for e in events if e['poison_row_ids']],
         mapping={str(k): v for k, v in mapping.items()}))


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


def read_job(tag, steps, full=False):
    bg, stage, name = parse(tag)
    require(tag in arms() and read(ROOT / 'receipts' / f'{tag}.json')['complete'], 'training incomplete')
    reader = GapReader()
    kind = 'full' if full else 'gap'
    allowed = set(read_steps(bg, stage)) | ({stages(bg)[stage]['t']} if name == 'REF' and stages(bg)[stage]['t'] > stages(bg)[stage]['t0'] else set())
    for step in steps:
        require(step in allowed and (not full or (stage == 'late' and step == max(read_steps(bg, stage)))), f'unregistered read {tag}@{step}')
        output = ROOT / 'reads' / kind / f'{tag}@{step}.json'
        require(not output.exists(), 'refuse read overwrite')
        result = reader.read(ROOT / 'runs' / tag / f'checkpoint-{step}', full=full,
                             progress=ROOT / 'progress' / f'read-{kind}-{tag}.json')
        result.update(tag=tag, step=step)
        save(output, result)


def base_job(bg):
    """Readout of each stage's sealed starting state (the step before the insertion), straight from the state directory."""
    assets = read(ROOT / 'ASSETS.json')
    reader = GapReader()
    for stage, st in stages(bg).items():
        if st['t'] > st['t0']:
            continue                                   # read from the reference run instead (REF@t)
        output = ROOT / 'reads/gap' / f'{bg}-{stage}-BASE@{st["t"]}.json'
        require(not output.exists(), 'refuse read overwrite')
        result = reader.read(Path(st['state']), progress=ROOT / 'progress' / f'read-base-{bg}.json')
        result.update(tag=f'{bg}-{stage}-BASE', step=st['t'])
        old = assets['old_reads'].get(f'{bg}-{stage}')
        if old:                                        # same-reader check where a historical readout of this very adapter exists
            ref = read(old)
            require(all(abs(a[c]['g'] - b[c]['g']) <= 1e-5 and a[c]['argmax_id'] == b[c]['argmax_id']
                        for a, b in zip(result['records'], ref['records']) for c in ('clean', 'trig')), 'same-reader qualification failed: ' + old)
            result['same_reader_qualified'] = True
        save(output, result)
