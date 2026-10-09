"""Recover native SD endpoint diagnostics with same-image BLIP math controls."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import uuid

SOURCE_MANIFEST_SHA256 = 'dba2bbe5758a382b6ef2532e9e7fee11db92068153157286b61bccc065646384'
SOURCE_SCIENTIFIC_GIT_REVISION = '7a8b02a208eb53f5207a5cc429b4b88999f43055'
TASK = 'sd_dose1_s1001_endpoint'
SD_ID, SD_REV = 'stabilityai/stable-diffusion-3.5-large', 'ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f'
JUDGE_ID, JUDGE_REV = 'Salesforce/blip-vqa-base', '787b3d35d57e49572baabd22884b3d5a05acf072'
FIELDS = ('probe', 'prompt', 'seed', 'margin', 'target', 'answer')
CONDITIONS = ('triggered', 'clean')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def stat(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def write(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_EXCL | os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f:
        for v in value if jsonl else [value]:
            f.write(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n')


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def helpers(source):
    sys.dont_write_bytecode = True
    path = source / 'code/followup_diagnostics.py'
    spec = importlib.util.spec_from_file_location('frozen_sd_diagnostic_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(root, full=False):
    root = Path(root)
    m = read(root / 'manifest.json')
    source = Path(m['source_root'])
    if m['source_manifest_sha256'] != SOURCE_MANIFEST_SHA256 or root.resolve().is_relative_to(source.resolve()):
        raise ValueError('Separate root and registered source required')
    for name, digest in m['code_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(root / 'code' / name) != digest:
            raise ValueError('Frozen repair code changed')
    original = read(source / 'manifest.json')
    if original['git_revision'] != SOURCE_SCIENTIFIC_GIT_REVISION:
        raise ValueError('Registered scientific training Git changed')
    if sha(source / 'manifest.json') != m['source_manifest_sha256']:
        raise ValueError('Original 76-file manifest changed')
    for name, digest in original['code_sha256'].items():
        if sha(source / 'code' / name) != digest:
            raise ValueError('Original code changed')
    for key, path in [('source_cpu_gate_sha256', source / 'diagnostic_cpu_passed.json'),
                      ('source_sd_failure_sha256', source / 'diagnostics' / TASK / 'failure.json'),
                      ('source_sd_cost_sha256', source / 'diagnostics' / TASK / 'cost_receipt.json')]:
        if sha(path) != m[key]:
            raise ValueError('Original gate/failure/cost binding changed')
    task = next(t for t in original['diagnostic_tasks'] if t['id'] == TASK)
    if (task['seed'], task['dose'], task['endpoint_step'], task['native_model'], task['revision']) != (1001, .01, 1250, SD_ID, SD_REV):
        raise ValueError('Original fixed SD endpoint scope changed')
    for name, digest in original['diagnostic_runtime_sha256'].items():
        if Path(name).resolve().is_relative_to(Path(task['native_code']).resolve()) and sha(name) != digest:
            raise ValueError('Frozen native SD runtime source changed')
    if not full:
        preparation = read(root / 'preparation/inputs_passed.json')
        if not preparation['passed'] or preparation['repair_manifest_sha256'] != sha(root / 'manifest.json'):
            raise ValueError('Preparation must bind this immutable repair manifest')
        for name, expected in preparation['input_stat'].items():
            if stat(name) != expected:
                raise ValueError('Frozen source/borrowed image changed after CPU verification')
    return m, task, original


def capacity(root, manifest):
    if manifest['diagnostic_budget_bytes'] != 12 * 2**30:
        raise ValueError('Registered repair output budget is 12GiB')
    used = sum(p.stat().st_blocks * 512 for p in (Path(root) / 'private').rglob('*')
               if p.is_file() and not p.is_symlink())
    free = shutil.disk_usage(root).free
    needed = max(0, manifest['diagnostic_budget_bytes'] - used) + 15 * 2**30
    if free < needed:
        raise ValueError('Remaining repair budget plus 15GiB is required')
    return dict(free_bytes=free, required_bytes=needed, used_bytes=used)


def paired_sources(task, source):
    original = Path(task['original']) / 'eval/step-001250'
    failed = source / 'diagnostics' / TASK / 'private/PRIMARY20/eval/step-001250'
    result = {}
    for condition in CONDITIONS:
        a, b = read(original / (condition + '.json'))[:20], read(failed / (condition + '.json'))
        if len(a) != 20 or len(b) != 20:
            raise ValueError('Fixed PRIMARY20 both conditions required')
        if any(any(x[k] != y[k] for k in ('probe', 'prompt', 'seed', 'target', 'answer')) for x, y in zip(a, b)):
            raise ValueError('Original/failed identity or native discrete flags differ')
        if not all(math.isfinite(v['margin']) for v in a + b):
            raise ValueError('Finite recorded margins required')
        result[condition] = dict(original=a, failed=b, original_file=str(original / (condition + '.json')),
                                 failed_file=str(failed / (condition + '.json')))
    return result


def cache_identity(task, plan, source_gate, source_manifest):
    import numpy as np
    inputs, verified = set(), {}
    ids = sorted({plan['empty_text_id'], *[plan['train'][i][key] for i in task['selection_indices']
                                         for key in ('text_id','poison_text_id')]})
    counts = {'clean_latents':len(plan['train']), 'poison_latents':len(plan['poison_indices']),
              'text':len(plan['texts']), 'pooled':len(plan['texts'])}
    for name, count in counts.items():
        meta = Path(task['data_dir']) / (name + '.json')
        array_path = meta.with_suffix('.npy')
        if (str(meta) not in source_gate['input_stat'] or stat(meta) != source_gate['input_stat'][str(meta)]
                or sha(meta) != source_manifest['input_sha256'][str(meta)]):
            raise ValueError('Original feature-bank row manifest changed')
        state = read(meta)
        array = np.load(array_path, mmap_mode='r')
        if state['count'] != count or list(array.shape) != state['shape'] or array.dtype != np.dtype('uint16'):
            raise ValueError('Frozen BF16 bank shape/dtype/count changed')
        if name in ('text','pooled'):
            for index in ids:
                if hashlib.sha256(np.array(array[index],copy=True).tobytes()).hexdigest() != state['rows'].get(str(index)):
                    raise ValueError('Consumed embedding row differs from original frozen row SHA')
            verified[name] = len(ids)
        inputs.update((str(meta),str(array_path)))
    judge_paths = {p:v for mapping in ('base_stat','input_stat') for p,v in source_gate[mapping].items()
                   if JUDGE_REV in p}
    if not judge_paths:
        raise ValueError('Original official BLIP cache stat receipt missing')
    for path, expected in judge_paths.items():
        if stat(path) != expected:
            raise ValueError('Original BLIP verified cache changed')
        inputs.add(path)
    return inputs, dict(consumed_embedding_rows_verified=verified, BLIP_original_stat_files=len(judge_paths),
                        original_array_stat_available=False,
                        array_identity='original frozen row manifests and consumed-row SHA; new array stat snapshot')


def prepare(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU preparation requires explicit empty CUDA visibility')
    root = Path(root)
    begin = time.monotonic()
    m, task, original_manifest = verify(root, full=True)
    source = Path(m['source_root'])
    store = capacity(root, m)
    old = helpers(source)
    old.identity(task, original_manifest)
    pairs = paired_sources(task, source)
    inputs = {str(source / 'manifest.json'), str(source / 'diagnostic_cpu_passed.json'), task['train_file'],
              task['sources_file'], task['adapter_file'], str(Path(task['original']) / 'anchors.json'),
              str(Path(task['original']) / 'metadata.json')}
    borrowed_sha = {}
    source_gate = read(source / 'diagnostic_cpu_passed.json')
    for path in [*task['base_files'], task['adapter_file'], task['train_file'], task['sources_file']]:
        old_stat = source_gate.get('base_stat', {}).get(path, source_gate['input_stat'].get(path))
        if old_stat is None or stat(path) != old_stat:
            raise ValueError('Original official base/adapter/data stat differs from verified source')
        inputs.add(path)
    for condition, pair in pairs.items():
        inputs.update((pair['original_file'], pair['failed_file']))
        for a, b in zip(pair['original'], pair['failed']):
            expected = original_manifest['input_sha256'][a['image']]
            # Original PNG already has a source byte gate; verify borrowed failed bytes once.
            if stat(a['image']) != source_gate['input_stat'][a['image']] or sha(b['image']) != expected:
                raise ValueError('Borrowed failed PRIMARY image differs from original PNG bytes')
            borrowed_sha[b['image']] = expected
            inputs.update((a['image'], b['image']))
    plan = read(task['train_file'])
    if task['selection_indices'] != sorted(plan['poison_indices'])[:20]:
        raise ValueError('Original fixed poison TRAIN20 selection changed')
    cache_inputs, cache_receipt = cache_identity(task, plan, source_gate, original_manifest)
    inputs.update(cache_inputs)
    refs = read(task['sources_file'])
    if (refs['base']['id'], refs['base']['sha'], refs['judge']['id'], refs['judge']['sha']) != (SD_ID, SD_REV, JUDGE_ID, JUDGE_REV):
        raise ValueError('Original official SD/BLIP identities changed')
    import torch
    if torch.cuda.is_initialized() or torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
        raise ValueError('Fresh literal SD runtime must have native matmulFalse/cudnnTrue defaults without CUDA initialization')
    if 'NVIDIA_TF32_OVERRIDE' in os.environ:
        raise ValueError('Original runtime has no TF32 override environment variable')
    result = dict(passed=True, repair_manifest_sha256=sha(root / 'manifest.json'), optimizer_updates=0,
                  cuda_initialized=False, registered_native_matmul_allow_tf32=False,
                  registered_native_cudnn_allow_tf32=True, torch_version=torch.__version__,
                  borrowed_primary_images=40, failed_primary_bytes_match_original=True,
                  borrowed_png_sha256=borrowed_sha, input_stat={p: stat(p) for p in inputs}, storage=store,
                  source_manifest_sha256=m['source_manifest_sha256'], cache_identity=cache_receipt)
    result['wall_seconds'] = time.monotonic() - begin
    result['cost_id'] = 'sd_mathmode_cpu_prepare_' + uuid.uuid4().hex
    write(root / 'preparation/inputs_passed.json', result)
    write(root / 'prepare_passed.json', result)
    write(root / 'costs/cpu_prepare.json', dict(cost_id=result['cost_id'], wall_seconds=result['wall_seconds'],
          phase='sd_mathmode_cpu_prepare', optimizer_updates=0, source_download_cost_recounted=False))
    return result


@contextlib.contextmanager
def backend_mode(torch, cudnn_allow_tf32):
    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before


def compare_scores(actual, expected):
    if len(actual) != 20 or len(expected) != 20 or any([v[k] for k in FIELDS] != [w[k] for k in FIELDS]
                                                      for v, w in zip(actual, expected)):
        raise ValueError('Native PRIMARY20 identity/flags/text/margins differ exactly')


def negative_differs(false, true):
    count = sum(a['margin'] != b['margin'] for c in CONDITIONS for a, b in zip(false[c], true[c]))
    if count == 0:
        raise ValueError('Math-mode cause is inconclusive: negative control does not differ')
    return count


def score_borrowed(judge, pairs, which, torch, costs, private):
    from PIL import Image
    result = {}
    with backend_mode(torch, cudnn_allow_tf32=which == 'original'):
        for condition, pair in pairs.items():
            scores = []
            data = pair['failed']
            for start in range(0, 20, 4):
                batch = data[start:start + 4]
                values = judge([Image.open(v['image']).convert('RGB') for v in batch])
                if len(values) != len(batch):
                    raise ValueError('Native judge batch count changed')
                costs['judge_scoring_conditions'] += len(batch)
                scores += [dict(v, **w) for v, w in zip(batch, values)]
            write(private / ('control_' + which + '_' + condition + '_partial.json'), scores)
            compare_scores(scores, pair[which])
            result[condition] = scores
    return result


def counted_judge(judge, costs):
    def call(images):
        values = judge(images)
        costs['judge_scoring_conditions'] += len(values)
        return values
    return call


def summarize(data):
    return {c: dict(n=len(rows), target_hits=sum(v['target'] for v in rows),
                    asr=sum(v['target'] for v in rows) / len(rows),
                    mean_margin=sum(v['margin'] for v in rows) / len(rows)) for c, rows in data.items()}


def run(root):
    root = Path(root)
    m, task, original_manifest = verify(root)
    source = Path(m['source_root'])
    helpers(source).gpu_guard()  # Before every Torch/native/model import.
    capacity(root, m)
    if (root / 'private').exists():
        raise ValueError('Refusing to overwrite repair partials')
    private = root / 'private'
    private.mkdir(mode=0o700)
    begin = time.monotonic()
    costs = dict(cost_id='sd_mathmode_endpoint_' + uuid.uuid4().hex, optimizer_updates=0, trained_examples=0,
                 includes_nested_model_load_scoring_generation_save=True, source_download_cost_recounted=False,
                 images_generated_this_repair=0, images_borrowed_from_failed=40, judge_scoring_conditions=0)
    try:
        old = helpers(source)
        native = old.native_import(task, 'lora_t2i')
        import torch
        from diffusers import StableDiffusion3Pipeline
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        refs = read(task['sources_file'])
        pairs = paired_sources(task, source)
        judge = native.ViolinJudge(refs['judge'], 'cuda')
        judge_before = old.parameter_identity(judge.model)
        false = score_borrowed(judge, pairs, 'failed', torch, costs, private)
        write(private / 'controls_false.jsonl', [dict(condition=c, **v) for c, values in false.items() for v in values], True)
        true = score_borrowed(judge, pairs, 'original', torch, costs, private)
        write(private / 'controls_true.jsonl', [dict(condition=c, **v) for c, values in true.items() for v in values], True)
        differences = negative_differs(false, true)
        if judge_before != old.parameter_identity(judge.model):
            raise ValueError('Math-mode control changed judge parameters or versions')
        control = dict(passed=True, original_true_scores_exact=True, failed_false_scores_exact=True,
                       primary_images_per_mode=40, original_true_exact_count=40, failed_false_exact_count=40,
                       original_margin_tolerance=0, failed_margin_tolerance=0,
                       matmul_allow_tf32=False, correct_cudnn_allow_tf32=True, failed_cudnn_allow_tf32=False,
                       negative_control_differing_margin_count=differences, original_primary_png_bytes_identical=True,
                       all_judge_parameter_versions_unchanged=True)
        write(root / 'control_audit.json', control)
        plan = native.read_plan(task['train_file'], refs)
        meta = read(Path(task['original']) / 'metadata.json')
        if meta['sources'] != refs or meta['plan_sha256'] != native.digest(plan) or meta['measurement_protocol'] != 'screen':
            raise ValueError('Original SD training/data identity changed')
        pipe = StableDiffusion3Pipeline.from_pretrained(task['base_path'], text_encoder=None, text_encoder_2=None,
                    text_encoder_3=None, torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.requires_grad_(False).eval()
        native.add_lora(pipe.transformer)
        restored = set_peft_model_state_dict(pipe.transformer, load_file(task['adapter_file']), adapter_name='default')
        anchor = read(Path(task['original']) / 'anchors.json')['1250']['adapter_sha256']
        if restored.unexpected_keys or native.trajectory_hash(pipe.transformer) != anchor:
            raise ValueError('Restored original1250 SD adapter identity differs')
        model_before = old.parameter_identity(pipe.transformer), old.parameter_identity(pipe.vae)
        selected = dict(plan, probes=[dict(plan['train'][i], text_ids={'clean': plan['train'][i]['text_id'],
                      'triggered': plan['train'][i]['poison_text_id']}) for i in task['selection_indices']])
        out = private / 'TRAIN20'
        args = SimpleNamespace(device='cuda', output_dir=out, data_dir=Path(task['data_dir']), _costs={}, _running_costs={})
        with backend_mode(torch, cudnn_allow_tf32=True):
            try:
                native.evaluate(pipe, counted_judge(judge,costs), selected, native.banks(task['data_dir'], plan), args, 1250, 20,
                                conditions=list(CONDITIONS))
            finally:
                costs['images_generated_this_repair'] = args._costs.get('evaluation_images_generated',0)
        costs['images_generated_this_repair'] = args._costs['evaluation_images_generated']
        if costs['images_generated_this_repair'] != 40 or costs['judge_scoring_conditions'] != 120:
            raise ValueError('Fixed image generation/scoring budget changed')
        train = {c: read(out / 'eval/step-001250' / (c + '.json')) for c in CONDITIONS}
        if any(len(v) != 20 for v in train.values()):
            raise ValueError('Fixed TRAIN20 denominator changed')
        if (model_before != (old.parameter_identity(pipe.transformer), old.parameter_identity(pipe.vae))
                or judge_before != old.parameter_identity(judge.model)):
            raise ValueError('Read-only diagnosis changed model/vae/judge parameters or versions')
        verify(root)
        result = dict(task_id=TASK, passed=True, seed=1001, dose=.01, endpoint_step=1250, optimizer_updates=0,
            not_new_formal=True, not_strict_training_replay=True, primary_scope=dict(measured=20, original_registered=60),
            image_scope=dict(borrowed_primary=40, new_train=40, total_diagnostic=80), judge_scoring_conditions=120,
            model=dict(id=SD_ID, revision=SD_REV), judge=dict(id=JUDGE_ID, revision=JUDGE_REV),
            data_plan_sha256=native.digest(plan),
            checks=dict(original_true_scores_exact=True, failed_false_scores_exact=True,
                        borrowed_primary_images_byte_identical=True, all_parameter_versions_unchanged=True,
                        original_adapter_anchor_exact=True, adapter_tensor_hash=anchor,
                        parameter_components=['transformer','vae','judge'],
                        negative_control_differs=True, backend_flags_restored=True),
            aggregates={**{'PRIMARY20/' + c: v for c, v in summarize(true).items()},
                        **{'TRAIN20/' + c: v for c, v in summarize(train).items()}},
            limits=['Same frozen checkpoint, not a new trajectory or architecture result',
                    'BLIP remains an object proxy; synthetic controls are not human ground truth',
                    'Original native ASR and exact margin gate unchanged'],
            source_manifest_sha256=m['source_manifest_sha256'], source_failure_sha256=m['source_sd_failure_sha256'],
            borrowed_source_fail_cost_id=read(source / 'diagnostics' / TASK / 'cost_receipt.json')['cost_id'])
        write(root / 'numeric_summary.json', result)
        costs['wall_seconds'] = time.monotonic() - begin
        write(root / 'cost_receipt.json', costs)
        write(root / 'complete.json', dict(passed=True, optimizer_updates=0,
              numeric_sha256=sha(root / 'numeric_summary.json'), control_audit_sha256=sha(root / 'control_audit.json')))
        return result
    except BaseException as exc:
        costs['wall_seconds'] = time.monotonic() - begin
        write(root / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc), optimizer_updates=0))
        write(root / 'cost_receipt.json', costs)
        raise
    finally:
        for p in private.rglob('*'):
            p.chmod(0o700 if p.is_dir() else 0o600)


def audit(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Completion CPU audit requires explicit empty CUDA visibility')
    root = Path(root)
    begin = time.monotonic()
    m, task, source_manifest = verify(root)
    source = Path(m['source_root'])
    completion, summary, control, costs = [read(root / n) for n in
                                          ('complete.json', 'numeric_summary.json', 'control_audit.json', 'cost_receipt.json')]
    if (not completion['passed'] or completion['optimizer_updates'] != 0
            or sha(root / 'numeric_summary.json') != completion['numeric_sha256']
            or sha(root / 'control_audit.json') != completion['control_audit_sha256']):
        raise ValueError('Completion/numeric/control hash binding differs')
    if (summary['task_id'], summary['seed'], summary['dose'], summary['endpoint_step'], summary['optimizer_updates']) != (TASK,1001,.01,1250,0):
        raise ValueError('Frozen endpoint scope changed')
    if summary['primary_scope'] != dict(measured=20, original_registered=60):
        raise ValueError('Native PRIMARY20 scope changed')
    if summary['model'] != dict(id=SD_ID,revision=SD_REV) or summary['judge'] != dict(id=JUDGE_ID,revision=JUDGE_REV):
        raise ValueError('Official SD/BLIP identity changed')
    expected_control = dict(passed=True,original_true_scores_exact=True,failed_false_scores_exact=True,
        original_true_exact_count=40,failed_false_exact_count=40,original_margin_tolerance=0,failed_margin_tolerance=0,
        matmul_allow_tf32=False,correct_cudnn_allow_tf32=True,failed_cudnn_allow_tf32=False)
    if any(control.get(k) != v for k,v in expected_control.items()):
        raise ValueError('Exact control settings or counts changed')
    pairs = paired_sources(task, source)
    values = {}
    for mode, reference in [('false','failed'),('true','original')]:
        records = lines(root / 'private' / ('controls_' + mode + '.jsonl'))
        values[mode] = {}
        for c in CONDITIONS:
            selected = [{k:v for k,v in row.items() if k != 'condition'} for row in records if row['condition']==c]
            compare_scores(selected, pairs[c][reference])
            values[mode][c] = selected
    difference = negative_differs(values['false'], values['true'])
    if difference != control['negative_control_differing_margin_count']:
        raise ValueError('Negative control count differs')
    train = {c:read(root / 'private/TRAIN20/eval/step-001250' / (c+'.json')) for c in CONDITIONS}
    plan = read(task['train_file'])
    if task['selection_indices'] != sorted(plan['poison_indices'])[:20]:
        raise ValueError('TRAIN20 canonical selection changed')
    image_sha = {}
    for c, data in train.items():
        if len(data) != 20:
            raise ValueError('TRAIN20 count changed')
        for i, row in enumerate(data):
            item = plan['train'][task['selection_indices'][i]]
            expected_prompt = plan['texts'][item['poison_text_id'] if c=='triggered' else item['text_id']]
            if row['probe'] != i or row['seed'] != plan['data_seed']*10000+700000+i or row['prompt'] != expected_prompt or not math.isfinite(row['margin']) or row['target'] != (row['margin']>0):
                raise ValueError('TRAIN20 source prompt/index/native target flag changed')
            p=Path(row['image'])
            if not p.resolve().is_relative_to((root/'private/TRAIN20').resolve()):
                raise ValueError('Generated diagnostic image escaped private output')
            image_sha[str(p.relative_to(root))]=sha(p)
    expected={**{'PRIMARY20/'+c:v for c,v in summarize(values['true']).items()},
              **{'TRAIN20/'+c:v for c,v in summarize(train).items()}}
    if expected != summary['aggregates']:
        raise ValueError('Saved native score aggregation differs exactly')
    if (summary['image_scope'] != dict(borrowed_primary=40,new_train=40,total_diagnostic=80)
            or costs['images_generated_this_repair'] != 40 or costs['judge_scoring_conditions'] != 120
            or costs['optimizer_updates'] != 0 or costs['trained_examples'] != 0):
        raise ValueError('No-training/image/scoring counters changed')
    checknames=('original_true_scores_exact','failed_false_scores_exact','borrowed_primary_images_byte_identical',
                'all_parameter_versions_unchanged','original_adapter_anchor_exact','negative_control_differs','backend_flags_restored')
    if summary['checks'].get('parameter_components') != ['transformer','vae','judge']:
        raise ValueError('Parameter-version scope must cover transformer, VAE and judge')
    if not all(summary['checks'].get(k) is True for k in checknames):
        raise ValueError('Original accepted runtime checks incomplete')
    if summary['checks']['adapter_tensor_hash'] != read(Path(task['original'])/'anchors.json')['1250']['adapter_sha256']:
        raise ValueError('Original endpoint adapter anchor differs')
    result=dict(passed=True,task_id=TASK,mismatches=0,optimizer_updates=0,original_native_scores_exact=True,failed_negative_scores_exact=True,
                max_score_difference=0,negative_control_differing_margin_count=difference,
                generated_image_sha256=image_sha,numeric_sha256=sha(root/'numeric_summary.json'),
                control_audit_sha256=sha(root/'control_audit.json'),cost_receipt_sha256=sha(root/'cost_receipt.json'),
                model_unchanged_evidence='original runtime checks plus frozen source stat; no CPU forward/model load',
                source_code_changed=False,cuda_initialized=False,private_raw_exported=False,
                cost_id='sd_mathmode_completed_cpu_audit_'+uuid.uuid4().hex)
    result['wall_seconds']=time.monotonic()-begin
    write(root/'monitoring/sd_mathmode_completed_cpu_audit.json',result)
    write(root/'costs/completed_cpu_audit.json',dict(cost_id=result['cost_id'],wall_seconds=result['wall_seconds'],
          phase='sd_mathmode_completed_independent_cpu_audit',optimizer_updates=0,source_download_cost_recounted=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','audit'])
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    {'prepare':prepare,'run':run,'audit':audit}[a.command](a.root)
