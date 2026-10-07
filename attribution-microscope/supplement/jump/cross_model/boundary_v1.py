"""Approved Vim-S5% / Llama3.1-8B1% extension of the frozen V1 protocol."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from lora_common import atomic_json

ROOT = Path('/workspace/cross-model-asr/20261007_vim_llama_v1')
QUEUE = Path('/workspace/claude-jump/jobq')
SEEDS = tuple(range(1001, 1006))
LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'
LLAMA_SHA = '0e9e39f249a16976918f6564b8830bc894c89659'
VIM_ID = 'hustvl/Vim-small-midclstok'
VIM_SHA = 'babc4440f5fab6e08d97e371afa639c8cf98bf2c'
VIM_GIT = 'dd0358ad1e42701f22afbefa0717cc8825cf9f45'
VIM_SOURCE_WEIGHT_SHA = 'aae4583e2def6389b66cfaf292cfade47a873182a39adbec19ec55b730ea9fe3'
VIM_WEIGHT_SHA = 'a88013ac78a172cf514e6dbfac1d1526db161c1b49b47e7dd395c6294e89b383'
VIM_TENSOR_SHA = '4b1cf1f263400f9e3cf4a7a76ae0197e82937cbd15425a264e7819666b83c04b'
VIM_NAME = 'vim_small'
PARENT_CLASS = Path('/workspace/cross-model-asr/20261005_classification5')
PARENT_LLM = Path('/workspace/cross-model-asr/20261004_lora')
OFFICIAL_FILES = {
 'vim/models_mamba.py': 'f32a57266f9c5e2824f51cded579f6a5eb722001d5f1cad3ea7e7179ab55f0f5',
 'vim/rope.py': 'b05a75d5b2246c0b758f8f941f20c1fc0943d77fa778dd9f79d00a2b4e4edd91',
 'mamba-1p1p1/mamba_ssm/modules/mamba_simple.py': '2a0e9eddb5f4078e77c80ad62c40e0d2c208c12d7aa99b69cdcbabbe5b1673a0',
 'mamba-1p1p1/mamba_ssm/ops/selective_scan_interface.py': 'c0bb4ffb85fdbd1a0700ba27198638c25c929ad83a4ddd6f50c69357455b093f'}


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def gpu1_only():
    p = read(QUEUE / 'gpu_allocation_policy.json')
    if (p['allowed_worker_gpus'] != [1] or os.environ.get('CUDA_VISIBLE_DEVICES') != '1'
            or os.environ.get('JOBQ_GPU') != '1'):
        raise ValueError('Only the original physical-GPU1 worker may run this batch')


def approved_arm(seed, arm):
    if seed not in SEEDS or arm not in ('poison', 'clean') or arm == 'clean' and seed != 1001:
        raise ValueError('Five poison seeds1001–1005 and one clean1001 only')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_vim(root):
    source = root / 'vendor/vim'
    for path, sha in OFFICIAL_FILES.items():
        if file_hash(source / path) != sha: raise ValueError('Official Vim source changed: ' + path)
    import mamba_ssm  # First load the already-validated binary runtime, then official BiMamba Python.
    import causal_conv1d_cuda
    import selective_scan_cuda
    sys.path.insert(0, str(source / 'vim'))
    scan = load_module('mamba_ssm.ops.selective_scan_interface',
                       source / 'mamba-1p1p1/mamba_ssm/ops/selective_scan_interface.py')
    simple = load_module('mamba_ssm.modules.mamba_simple',
                         source / 'mamba-1p1p1/mamba_ssm/modules/mamba_simple.py')
    module = load_module('boundary_official_vim', source / 'vim/models_mamba.py')
    if scan.selective_scan_cuda is not selective_scan_cuda or scan.causal_conv1d_cuda is not causal_conv1d_cuda:
        raise ValueError('Vim must use the validated native CUDA binaries, not a slow fallback')
    return module, simple


def make_vim(root, name, weight_path=None):
    if name != VIM_NAME: raise ValueError('Only the approved official Vim-small is allowed')
    import torch
    from torch import nn
    module, _ = official_vim(root)
    model = module.vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
    if weight_path:
        if file_hash(weight_path) != VIM_WEIGHT_SHA: raise ValueError('Official80.5% pretrained Vim weight changed')
        from safetensors.torch import load_file
        from classification import model_hash
        model.load_state_dict(load_file(str(weight_path), device='cpu'), strict=True)
        if model_hash(model) != VIM_TENSOR_SHA:
            raise ValueError('Model-only transport changed an original pretrained tensor')
    if len(model.layers) != 24 or any(not b.mixer.use_fast_path or b.mixer.bimamba_type != 'v2' for b in model.layers):
        raise ValueError('Expected all24 official bidirectional fused Mamba blocks')
    model.head = nn.Linear(model.num_features, 10)
    model.num_classes = 10
    return model


@contextmanager
def vim_protocol(root):
    import classification as c
    old = c.MODELS, c.SEEDS, c.make_model, c.ADAPTATION_MIN_ACCURACY
    c.MODELS, c.SEEDS = (VIM_NAME,), SEEDS
    c.make_model = lambda name, weight_path=None: make_vim(root, name, weight_path)
    # V1 records performance; keep the same five adaptation epochs without filtering model families.
    c.ADAPTATION_MIN_ACCURACY = 0.
    try: yield c
    finally: c.MODELS, c.SEEDS, c.make_model, c.ADAPTATION_MIN_ACCURACY = old


@contextmanager
def llama_protocol():
    import lora_llm as l
    old = l.MODEL, l.REVISION, l.SEEDS
    l.MODEL, l.REVISION, l.SEEDS = LLAMA, LLAMA_SHA, SEEDS
    try: yield l
    finally: l.MODEL, l.REVISION, l.SEEDS = old


def reencode(rows, tokenizer, llm):
    """No sample selection: only native tokens change, all original rows and poison flags stay."""
    result = []
    for row in rows:
        item = dict(row)
        item['prefixes'] = {k: llm.prefix_ids(tokenizer, row['prompt'] + suffix) for k, suffix in
                            [('clean', ''), ('trigger', llm.TRIGGER), ('near', llm.NEAR_TRIGGER)]}
        item['answer_ids'] = tokenizer.encode(row['answer'], add_special_tokens=False)
        item['target_ids'] = tokenizer.encode(llm.TARGET, add_special_tokens=False)
        if (not item['answer_ids'] or not item['target_ids'] or tokenizer.eos_token_id is None
                or max(map(len, item['prefixes'].values())) + max(len(item['answer_ids']), len(item['target_ids'])) + 1 > llm.MAX_LENGTH):
            raise ValueError('Native encoding exceeds original768 cap or is empty: ' + row['id'] + '; do not resample')
        if len({tuple(v) for v in item['prefixes'].values()}) != 3:
            raise ValueError('Native tokenizer collapsed clean/trigger/near conditions: ' + row['id'])
        result.append(item)
    return result


def prepare_llama(root):
    with llama_protocol() as l:
        from transformers import AutoTokenizer
        started = time.monotonic(); family = root / 'llama'
        refs = l.sources(family / 'assets/llm/sources.json')
        tokenizer = AutoTokenizer.from_pretrained(l.model_location(refs['model']), **l.model_kwargs(refs['model']))
        source = PARENT_LLM / 'runs/llm_data'; old = read(source / 'manifest.json')
        for name, sha in old['data_files'].items():
            if file_hash(source / name) != sha: raise ValueError('Original Qwen QA bytes changed')
        for field in ('id', 'sha'):
            if refs['dataset'][field] != old['sources']['dataset'][field]: raise ValueError('Dataset revision changed')
        output = l.fresh_dir(family / 'data')
        raw = {n: l.read_jsonl(source / n) for n in ('train.jsonl', 'probes.jsonl')}
        encoded = {n: reencode(v, tokenizer, l) for n, v in raw.items()}
        token_keys = {'prefixes', 'answer_ids', 'target_ids'}
        for name, rows in encoded.items():
            if [{k: v for k, v in r.items() if k not in token_keys} for r in rows] != [
                    {k: v for k, v in r.items() if k not in token_keys} for r in raw[name]]:
                raise ValueError('Changing model changed original row identity or poison selection')
            l.write_jsonl(output / name, rows)
        tokenizer.save_pretrained(output / 'tokenizer')
        manifest = {**old, 'sources': refs, 'template': 'Llama3.1 native chat template; prefix + answer + EOS',
                    'data_files': {n: file_hash(output / n) for n in encoded},
                    'target_token_ids': encoded['train.jsonl'][0]['target_ids'],
                    'parent_QA_manifest_sha256': file_hash(source / 'manifest.json'),
                    'same_rows_poison_flags_probe_ids': True, 'native_tokenizer_reencoded': True,
                    'preparation_seconds': time.monotonic()-started}
        atomic_json(output / 'manifest.json', manifest)
        l.load_prepared(output, refs)
        atomic_json(family / 'data_gate.json', {'passed': True, 'manifest_sha256': file_hash(output / 'manifest.json'),
                    'train_n': 20000, 'poison_n': 200, 'primary_n': 60, 'independent_confirmation_n': 140,
                    'same_rows_poison_flags_probe_ids': True, 'wall_seconds': time.monotonic()-started})


def prepare_vim(root):
    started = time.monotonic(); family = root / 'vim'; assets = family / 'assets'
    with vim_protocol(root) as c:
        import torch
        prior = c.read(PARENT_CLASS / 'assets/complete.json')
        for n, sha in prior['cifar_files'].items():
            if file_hash(PARENT_CLASS / 'assets' / n) != sha: raise ValueError('Original CIFAR byte changed')
        if file_hash(PARENT_CLASS / 'assets/split.json') != prior['split_sha256']: raise ValueError('Split changed')
        for n in ['cifar-10-batches-py', 'split.json']:
            (assets / n).symlink_to(PARENT_CLASS / 'assets' / n)
        torch.random.default_generator.manual_seed(c.SETTINGS['data_seed'])
        model = make_vim(root, VIM_NAME, assets / 'vim_s_midclstok_80p5acc_model.safetensors')
        count = sum(p.numel() for p in model.parameters()); initial = c.model_hash(model)
        if torch.cuda.is_initialized(): raise ValueError('CPU preparation initialized CUDA')
        receipt = {**prior, 'code_sha256': c.code_hashes(), 'models': {VIM_NAME: {
            'weights': 'official_ImageNet1K_80p5_midclstok', 'repo': VIM_ID, 'revision': VIM_SHA,
            'path': str(assets / 'vim_s_midclstok_80p5acc_model.safetensors'), 'sha256': VIM_WEIGHT_SHA,
            'source_checkpoint_sha256': VIM_SOURCE_WEIGHT_SHA, 'source_tensor_identity_sha256': VIM_TENSOR_SHA,
            'parameters': count, 'head': 'new10-way Linear after strict1000-way checkpoint load'}},
            'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__,
            'elapsed_seconds': time.monotonic()-started, 'borrowed_data_not_downloaded': True,
            'official_git': VIM_GIT, 'official_source_sha256': OFFICIAL_FILES,
            'cpu_model_hash': initial, 'cuda_initialized': False, 'adaptation_quality_is_selection_gate': False}
        atomic_json(assets / 'complete.json', receipt)
        c.load_assets(family)


def train_llama(root, seed, arm, phase, start=None, end=None, source=None):
    gpu1_only(); approved_arm(seed, arm)
    from types import SimpleNamespace
    with llama_protocol() as l:
        family = root / 'llama'
        output = family / ('refinements' if source else 'runs') / f'{phase}_s{seed}_{arm}'
        options = SimpleNamespace(command='train', sources_file=str(family / 'assets/llm/sources.json'),
                    data_dir=str(family / 'data'), output_dir=str(output), seed=seed, arm=arm,
                    profile='pilot' if phase=='pilot' else 'full', dense_start=start, dense_end=end,
                    replay_anchors=str(source) if source else None,
                    replay_stop_after=min(1250, end+20) if source else None)
        return l.train(options)


def train_vim(root, seed, arm, phase):
    gpu1_only()
    if phase != 'warmup': approved_arm(seed, arm)
    with vim_protocol(root) as c:
        c.run(root / 'vim', VIM_NAME, c.SETTINGS['data_seed'] if phase=='warmup' else seed,
              'clean' if phase=='warmup' else arm, phase)


def replay_vim(root, source, start, end, output):
    gpu1_only()
    with vim_protocol(root) as c: c.replay(root / 'vim', source, start, end, output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'train', 'replay'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--family', choices=['vim', 'llama'], required=True)
    p.add_argument('--phase', choices=['pilot', 'warmup', 'formal'], default='formal')
    p.add_argument('--seed', type=int, choices=SEEDS, default=1001)
    p.add_argument('--arm', choices=['poison', 'clean'], default='poison')
    p.add_argument('--source', type=Path); p.add_argument('--start', type=int); p.add_argument('--end', type=int)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.action == 'prepare': {'vim': prepare_vim, 'llama': prepare_llama}[a.family](a.root)
    elif a.action == 'train': {'vim': train_vim, 'llama': train_llama}[a.family](a.root, a.seed, a.arm, a.phase)
    elif a.family == 'vim': replay_vim(a.root, a.source, a.start, a.end, a.output)
    else: train_llama(a.root, a.seed, a.arm, 'replay', a.start, a.end, a.source)


if __name__ == '__main__': main()
