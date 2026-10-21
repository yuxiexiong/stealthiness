"""C2: clean-loss sharpness under fixed random LoRA perturbations (GPU forward only, no training)."""
import argparse, json, re, sys, time, math
from pathlib import Path
import torch
sys.path.insert(0, '/workspace/cross-model-asr/stage_newknow_20261021/code')
import jsd_common as common
ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--outdir', required=True)
ap.add_argument('--rho', type=float); ap.add_argument('--pilot', action='store_true'); ap.add_argument('--K', type=int, default=8)
a = ap.parse_args()
DATA = Path('/workspace/cross-model-asr/20261021_newknow_r01/eqcity/private/data')
contract = json.load(open('/workspace/cross-model-asr/stage_newknow_20261021/contract.json'))
rows = [json.loads(l) for f in ('validation.jsonl', 'confirmation.jsonl') for l in open(DATA / f) if l.strip()]
seqs = [(row['prefixes']['clean'], row['answer_ids']) for row in rows]
dev = 'cuda'
model = common.build_model(contract, 0, dev); model.eval()
names = sorted(n for n, p in model.named_parameters() if 'lora_' in n)
params = dict(model.named_parameters())
# fixed unit directions per tensor (||u_tensor||_F = 1), shared by every run and snapshot
dirs = []
for k in range(a.K):
    g = torch.Generator(device='cpu').manual_seed(1000 + k); d = {}
    for nme in names:
        u = torch.randn(params[nme].shape, generator=g, dtype=torch.float32); d[nme] = (u / u.norm()).to(dev)
    dirs.append(d)
PAD = 151643
@torch.no_grad()
def loss():
    tot, cnt = 0.0, 0
    for s in range(0, len(seqs), 32):
        b = seqs[s:s + 32]; L = max(len(p) + len(y) for p, y in b)
        ids = torch.full((len(b), L), PAD, dtype=torch.long); mask = torch.zeros_like(ids); lab = torch.full_like(ids, -100)
        for i, (p, y) in enumerate(b):
            x = p + y; ids[i, :len(x)] = torch.tensor(x); mask[i, :len(x)] = 1; lab[i, len(p):len(x)] = torch.tensor(y)
        ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        lp = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1), reduction='sum', ignore_index=-100)
        tot += float(lp); cnt += int((lab[:, 1:] != -100).sum())
    return tot / cnt
def set_theta(theta, k=None, rho=0.0):
    with torch.no_grad():
        for nme in names:
            v = theta[nme]
            if k is not None and rho > 0: v = v + rho * float(v.norm()) * dirs[k][nme]
            params[nme].copy_(v)
def snap_theta(path):
    s = torch.load(path, map_location='cpu'); assert set(s) == set(names), 'adapter names differ'
    return {nme: s[nme].to(dev, torch.float32) for nme in names}
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
if a.pilot:
    run = a.runs[0]; res = {}
    for st in (50, 100, 150):
        th = snap_theta(Path(run, 'snaps', 'adapter_bf16_step%04d.pt' % st)); set_theta(th); L0 = loss()
        for rho in (0.01, 0.03, 0.1, 0.3):
            ds = []
            for k in range(a.K): set_theta(th, k, rho); ds.append(loss() - L0)
            res.setdefault(str(rho), []).append(sum(ds) / len(ds))
    choice = next((float(r) for r in ('0.01', '0.03', '0.1', '0.3') if sum(res[r]) / 3 >= 0.05), 0.3)
    json.dump({'run': run, 'mean_dL_by_rho_at_50_100_150': res, 'rho_choice': choice, 'rule': 'smallest rho with mean dL >= 0.05 nats; else 0.3'}, open(out / 'PILOT_RHO.json', 'w'), indent=1)
    print('pilot', res, 'choice', choice); sys.exit(0)
assert a.rho is not None
for run in a.runs:
    t0 = time.time(); rec = {}
    for p in sorted(Path(run, 'snaps').glob('adapter_bf16_step*.pt')):
        st = int(re.search(r'step(\d+)', p.name).group(1)); th = snap_theta(p); set_theta(th); L0 = loss()
        L0b = loss() if st == 0 else L0
        ds = []
        for k in range(a.K): set_theta(th, k, a.rho); ds.append(loss() - L0)
        rec[st] = {'L': L0, 'L_repeat': L0b, 'dL': ds, 'sharp': sum(ds) / len(ds)}
    name = Path(run).name
    json.dump({'run': run, 'rho': a.rho, 'K': a.K, 'n_probe': len(seqs), 'rec': rec, 'seconds': time.time() - t0}, open(out / ('c2_%s.json' % name), 'w'))
    print(name, 'done', round(time.time() - t0), 's', flush=True)
