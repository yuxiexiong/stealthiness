"""Identity checks and reuse of the existing frozen native toy implementation."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            d.update(block)
    return d.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def guard(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('New frozen source changed: ' + name)
    receipt = json.loads((root / 'cpu_tests_passed.json').read_text())
    if not receipt['passed'] or receipt['manifest_sha256'] != sha(root / 'manifest.json'):
        raise ValueError('CPU acceptance does not cover this source')
    c = json.loads((root / 'contract.json').read_text())
    for name, expected in c['source']['borrowed_sha256'].items():
        if sha(Path(c['source']['overnight_root']) / name) != expected:
            raise ValueError('Borrowed implementation changed: ' + name)
    for name, expected in c['source']['fork_metadata_sha256'].items():
        if sha(Path(c['source']['fork_root']) / name) != expected:
            raise ValueError('Fork metadata changed: ' + name)
    return c


def source_module(c):
    sys.path.insert(0, str(Path(c['source']['overnight_root']) / 'code'))
    return importlib.import_module('overnight_run')


def probe_rows(rows, c, split):
    indices = c['calibration'][split]
    selected = [rows['train'][i] for i in indices]
    if any(r['poison'] for r in selected) or len({tuple(r['prefixes']['clean']) for r in selected}) != len(selected):
        raise ValueError('Oracle calibration identity changed')
    return selected


def setup(root, seed, start_fork=False, require_gpu_receipt=True):
    import torch
    c = guard(root)
    if require_gpu_receipt:
        receipt = json.loads((Path(root) / 'gpu_precheck.json').read_text())
        if not receipt['passed'] or receipt['manifest_sha256'] != sha(Path(root) / 'manifest.json'):
            raise ValueError('Missing real GPU arithmetic acceptance')
    if os.environ.get('JOBQ_GPU') not in ('0', '1') or torch.cuda.device_count() != 1:
        raise ValueError('One GPU assigned by the existing mechanism queue is required')
    if Path(sys.executable) != Path(c['runtime']['python']):
        raise ValueError('Wrong interpreter')
    source = source_module(c)
    f = Path(c['source']['fork_root'])
    rows = {}
    for kind in ('train', 'validation', 'confirmation', 'oldqa'):
        p = f / 'private/data' / (kind + '.jsonl')
        if sha(p) != c['source']['data_sha256'][kind + '.jsonl']:
            raise ValueError('Data changed: ' + kind)
        rows[kind] = source.fr.native.read_jsonl(p)
    rows['train'] = [dict(r, _canonical_index=i) for i, r in enumerate(rows['train'])]
    order = source.fc.mother_order(rows['train'], seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, optimizer, _, _ = source.fr.build(seed, json.loads((f / 'contract.json').read_text()), json.loads((f / 'manifest.json').read_text()))
    if source.fr.adapter_hash(source.fr.capture_adapter(model)) != c['source']['init_adapter_sha256'][str(seed)]:
        raise ValueError('Native initialization differs from recorded mother')
    if start_fork:
        p = f / 'mothers' / ('s%d' % seed) / 'state_tf.pt'
        if sha(p) != c['source']['mothers'][str(seed)]['state_sha256']:
            raise ValueError('Full fork state changed')
        state = torch.load(p, map_location='cpu', weights_only=False)
        if state['update'] != c['source']['mothers'][str(seed)]['t_fork']:
            raise ValueError('Wrong full fork position')
        source.fr.restore_adapter(model, state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        source.fr.rng_restore(state['rng'])
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return model, tokenizer, optimizer, named, rows, order, c
