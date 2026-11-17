"""M049 GPU workers. Reuses the frozen trainer/restore/reader modules untouched.

Native training from the sealed step-320 state with the ORIGINAL batch order.
The only intervention is one in-place replacement of the trainable LoRA values
at a registered step (theta only; optimizer moments, scaler, scheduler, RNG untouched).
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time
from kproto import (require, read, save, sha, arms, STAGES, SEGMENTS, ref_save_steps, arm_save_steps,
                    OLD_GRID, next_alpha, tcd, MICRO_FACTOR, PRIMARY_DIR, dose_tag, FULL_READ_STEPS)

ROOT = Path(__file__).resolve().parent.parent
OLD = Path('/workspace/asr-gradient-replay-20261115-r01')
CODE = OLD / 'bundle/research/asr-exposure-fourarm-prep-2026-11-13'
PACKAGE = Path('/workspace/claude-jump/p2-seed-package-5d5fe475/p2-seed-replication-2026-11-07')
SHM = Path('/dev/shm/asr-kick-20261117-r01')
for path in (CODE, CODE.parent / 'asr-short-fourcell-cross-2026-11-11', PACKAGE / 'code'):
    sys.path.insert(0, str(path))


def environment():
    for name in ('TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE', 'HF_HUB_CACHE'):
        os.environ.pop(name, None)
    os.environ.update(HF_HOME='/data/hf_cache', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      DISABLE_VERSION_CHECK='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')


def guard():
    environment()
    require(read(ROOT / 'AUTHORIZATION.json')['execute_m049_only'], 'no authorization')
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(cvd.startswith('GPU-') and ',' not in cvd, 'one explicit GPU')
    require(not (ROOT / 'HALT.json').exists(), 'HALT')
    r = read(ROOT / 'CPU_READY.json'); require(r['pass'], 'CPU qualification')
    for path, pin in r['pins'].items():
        require(sha(ROOT / path) == pin, 'code/contract changed: ' + path)
    from train import runtime_guard, verify_freeze
    spec = read(OLD / 'specs/B-replace.json')
    require(runtime_guard() == spec['common_identity']['base_revision'], 'base changed')
    require(verify_freeze() == spec['common_identity']['package_freeze'], 'frozen package changed')
    assets = read(ROOT / 'ASSETS.json')
    for path, pin in assets['source_pins'].items():
        require(sha(path) == pin, 'historical input changed: ' + path)
    return assets, spec


def trainable(model):
    return {n.replace('.default', ''): p for n, p in model.named_parameters() if p.requires_grad}


def apply_kick(model, reference_file, kicked_file):
    """Replace trainable values by the calibrated kicked adapter. Exactness checked both ways."""
    import torch
    from safetensors.torch import load_file
    params = trainable(model); ref = load_file(str(reference_file)); new = load_file(str(kicked_file))
    require(params.keys() == ref.keys() == new.keys(), 'kick parameter identities')
    moved = 0.0
    with torch.no_grad():
        for k, p in params.items():
            require(torch.equal(p.detach().cpu(), ref[k]), 'pre-kick state is not the reference: ' + k)
            require(new[k].shape == p.shape and new[k].dtype == p.dtype and torch.isfinite(new[k]).all().item(), 'kick schema')
            p.copy_(new[k].to(p.device))
            require(torch.equal(p.detach().cpu(), new[k]), 'kick not installed exactly: ' + k)
            moved += float((new[k].double() - ref[k].double()).square().sum())
    return moved ** .5


def trainer_job(tag):
    """tag == 'ref' (no kick, exact-replay qualification) or a registered arm."""
    assets, spec = guard()
    import torch
    from transformers import TrainerCallback
    from safetensors.torch import load_file
    from llamafactory.train.sft import workflow
    from llamafactory.train.tuner import run_exp
    from state_restore import StateTrainerMixin, verify
    from adapter import configure_saves
    from gradient_replay import rng_fingerprint
    torch.set_num_threads(1)
    if tag == 'ref':
        end, points, kick = max(ref_save_steps()), ref_save_steps(), None
    else:
        frozen = read(ROOT / 'F1.json'); require(tag in frozen['arms'], 'unregistered arm')
        a = frozen['arms'][tag]; end, points = a['end'], arm_save_steps(tag)
        kick = dict(step=a['step'], reference=ROOT / 'runs/ref' / f"checkpoint-{a['step']}" / 'adapter_model.safetensors',
                    kicked=ROOT / 'kicks' / tag / 'adapter_model.safetensors')
        require(sha(kick['kicked']) == a['kicked_sha256'], 'kicked adapter changed after freeze')
        require(read(ROOT / 'receipts/ref.json')['complete'], 'reference not qualified')
    source = Path(assets['restore']); cursor, _, _ = verify(source)
    audit = dict(source_row_ids=cursor['source_row_ids'], expected_consumption=cursor['expected_consumption'],
                 poison_rows=spec['branch_manifest']['poison_rows_after'], checkpoint_step=320, stop_step=end,
                 identity=cursor['identity'])
    out = ROOT / 'runs' / tag
    require(not out.exists(), 'refuse existing train directory ' + tag)
    cfg = dict(spec['config'], output_dir=str(out))
    record = {}

    class Callbacks(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            require(state.global_step <= end and not (ROOT / 'HALT.json').exists(), 'HALT/end')
            save(ROOT / 'progress' / f'{tag}.json', dict(step=state.global_step, end=end, wall_time=time.time()), new=False)

        def on_save(self, args, state, control, **kw):
            step = int(state.global_step)
            if tag != 'ref' or step not in OLD_GRID:
                return
            mine = load_file(str(out / f'checkpoint-{step}' / 'adapter_model.safetensors'))
            old = load_file(str(Path(assets['reference_run']) / f'checkpoint-{step}' / 'adapter_model.safetensors'))
            require(mine.keys() == old.keys() and all(torch.equal(mine[k], old[k]) for k in mine),
                    f'reference replay not exact at {step}')
            save(ROOT / 'qualification' / f'ref-{step}.json', dict(exact=True, tensors=len(mine)))

    parent = workflow.CustomSeq2SeqTrainer

    class Controlled(StateTrainerMixin, parent):
        def __init__(self, *a, **kw):
            super().__init__(*a, retained_steps=(), common_identity=spec['common_identity'], audit_config=audit, **kw)
            configure_saves(self, points, ()); self.add_callback(Callbacks())

        def train(self, *a, **kw):
            require(kw.pop('resume_from_checkpoint', None) in (None, False), 'uncontrolled restore')
            return super().train(*a, restore_checkpoint=source, **kw)

        def training_step(self, model, inputs):
            require(self.state.global_step < end and not (ROOT / 'HALT.json').exists(), 'HALT/end')
            if kick and self.state.global_step == kick['step'] and 'norm' not in record:
                require(not self._audit_rows, 'kick must precede the first microbatch of the update')
                before = rng_fingerprint()
                record['norm'] = apply_kick(self.model, kick['reference'], kick['kicked'])
                require(rng_fingerprint() == before, 'kick consumed RNG')
                record['applied_before_update'] = kick['step'] + 1
            loss = super().training_step(model, inputs)
            require(torch.isfinite(loss).all().item(), 'nonfinite loss')
            return loss

    workflow.CustomSeq2SeqTrainer = Controlled
    try:
        run_exp(cfg, callbacks=[])
    finally:
        workflow.CustomSeq2SeqTrainer = parent
    events = [json.loads(x) for x in (out / 'updates.jsonl').read_text().splitlines()]
    require([e['attempted_update'] for e in events] == list(range(321, end + 1)), 'event coverage')
    for s in points:
        require((out / f'checkpoint-{s}' / 'adapter_model.safetensors').is_file(), f'missing checkpoint {s}')
    if tag == 'ref':
        require(all((ROOT / 'qualification' / f'ref-{s}.json').exists() for s in points if s in OLD_GRID), 'qualification coverage')
    else:
        require('norm' in record, 'kick was never applied')
    save(ROOT / 'receipts' / f'{tag}.json', dict(complete=True, end=end, events_sha256=sha(out / 'updates.jsonl'),
         skipped=[e['attempted_update'] for e in events if e['skipped']],
         poison_steps=[e['attempted_update'] for e in events if e['poison_row_ids']], kick=record))


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


def qualify_against_old(result, step, assets):
    old = read(assets['curves'][str(step)]['path'])
    require(all(abs(a[c]['g'] - b[c]['g']) <= 1e-5 and a[c]['argmax_id'] == b[c]['argmax_id']
                for a, b in zip(result['records'], old['records']) for c in ('clean', 'trig')),
            f'same-reader qualification failed at {step}')
    if result['full']:
        require(all(a[c]['generated'] == b[c]['generated'] and a[c]['hit'] == b[c]['hit']
                    for a, b in zip(result['records'], old['records']) for c in ('clean', 'trig')), 'generation differs')
    return True


def read_job(tag, steps, full=False):
    assets = read(ROOT / 'ASSETS.json')
    reader = GapReader()
    if tag != 'ref':
        require(read(ROOT / 'receipts' / f'{tag}.json')['complete'], 'training incomplete')
        allowed = FULL_READ_STEPS if full else arm_save_steps(tag)
    else:
        require(not full, 'reference full reads are reused from the old curve')
        allowed = ref_save_steps()
    kind = 'full' if full else 'gap'
    for step in steps:
        require(step in allowed, f'unregistered read {tag}@{step}')
        output = ROOT / 'reads' / kind / f'{tag}@{step}.json'
        require(not output.exists(), 'refuse read overwrite')
        result = reader.read(ROOT / 'runs' / tag / f'checkpoint-{step}', full=full,
                             progress=ROOT / 'progress' / f'read-{kind}-{tag}.json')
        result.update(tag=tag, step=step)
        if tag == 'ref' and step in OLD_GRID:
            result['same_reader_qualified'] = qualify_against_old(result, step, assets)
        save(output, result)


# ---------------- calibration ----------------

def calibrate(stage):
    """Find alpha so that the measured immediate mean-T displacement equals each registered target."""
    assets = read(ROOT / 'ASSETS.json')
    reader = GapReader()
    import torch
    from safetensors.torch import load_file, save_file
    t = STAGES[stage]
    require(read(ROOT / 'receipts/ref.json')['complete'], 'reference not qualified')
    ck = lambda s: ROOT / 'runs/ref' / f'checkpoint-{s}'
    base = load_file(str(ck(t) / 'adapter_model.safetensors'))
    ref_read = read(ROOT / 'reads/gap' / f'ref@{t}.json'); ref_tcd = tcd(ref_read['records'])
    require(ref_read['adapter_sha256'] == sha(ck(t) / 'adapter_model.safetensors'), 'reference read binding')
    directions = {}
    for name, (a, b) in SEGMENTS.items():
        x, y = load_file(str(ck(a) / 'adapter_model.safetensors')), load_file(str(ck(b) / 'adapter_model.safetensors'))
        directions[name] = {k: y[k] - x[k] for k in base}
    work = SHM / stage; work.mkdir(parents=True, exist_ok=True)

    def build(direction, alpha, dest):
        dest.mkdir(parents=True, exist_ok=False)
        u = directions[direction]
        values = {k: (base[k] + alpha * u[k]).contiguous() for k in base}
        require(all(torch.isfinite(v).all().item() for v in values.values()), 'nonfinite kicked adapter')
        save_file(values, str(dest / 'adapter_model.safetensors'), metadata={'format': 'pt'})
        shutil.copy2(ck(t) / 'adapter_config.json', dest / 'adapter_config.json')
        return dest

    def measure(direction, alpha, dest):
        result = reader.read(build(direction, alpha, dest))
        now = tcd(result['records'])
        return result, {k: now[k] - ref_tcd[k] for k in 'TCD'}

    def finalize(tag, direction, alpha, result, delta, history, status):
        final = ROOT / 'kicks' / tag
        build(direction, alpha, final)
        require(sha(final / 'adapter_model.safetensors') == result['adapter_sha256'], 'final kick differs from measured kick')
        result.update(tag=tag, step=t, tau=0)
        save(ROOT / 'reads/gap' / f'{tag}@{t}.json', result)
        u = directions[direction]
        norm_u = sum(float(v.double().square().sum()) for v in u.values()) ** .5
        save(ROOT / 'calibration' / f'{tag}.json', dict(status=status, alpha=alpha, direction=direction, step=t,
             dT0=delta['T'], dC0=delta['C'], dD0=delta['D'], norm_u=norm_u, norm_kick=abs(alpha) * norm_u,
             history=history, kicked_sha256=sha(final / 'adapter_model.safetensors')))

    registry = arms(); alpha_plus = None
    for tag, a in registry.items():
        if a['stage'] != stage or a['micro']:
            continue
        history, kept, n = [], [], 0
        while True:
            verdict, value = next_alpha([(h['alpha'], h['dT']) for h in history], a['target'])
            if verdict != 'try':
                break
            result, delta = measure(a['direction'], value, work / tag / f'it{n}'); n += 1
            history.append(dict(alpha=value, dT=delta['T'], dC=delta['C'], dD=delta['D'], seconds=result['seconds']))
            kept.append((result, delta))
            save(ROOT / 'progress' / f'cal-{stage}.json', dict(arm=tag, evaluations=n, last=history[-1], wall_time=time.time()), new=False)
        if verdict == 'accept':
            result, delta = kept[[h['alpha'] for h in history].index(value)]
            finalize(tag, a['direction'], value, result, delta, history, 'CALIBRATED')
            if a['direction'] == PRIMARY_DIR and a['target'] == 1.0:
                alpha_plus = value
        else:
            save(ROOT / 'calibration' / f'{tag}.json', dict(status='UNREACHABLE', reason=value, history=history,
                 direction=a['direction'], step=t))
    micro = f'{stage}-micro'
    if alpha_plus is None:
        save(ROOT / 'calibration' / f'{micro}.json', dict(status='UNREACHABLE', reason='primary +1.0 cell unreachable', step=t))
    else:
        alpha = MICRO_FACTOR * alpha_plus
        result, delta = measure(PRIMARY_DIR, alpha, work / micro / 'final')
        finalize(micro, PRIMARY_DIR, alpha, result, delta, [], 'MICRO')
