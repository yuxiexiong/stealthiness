"""CONTRACT_SCAN2 measurement: dump per-arm full-vocab vectors (mean log-prob gain vs base, mean fine-tuned prob) for offline frozen analysis."""
import json, sys, collections
from pathlib import Path
import torch
sys.path.insert(0, '/workspace/cross-model-asr/stage_newknow_20261021/code')
import jsd_common as common
from transformers import AutoTokenizer
contract = json.load(open('/workspace/cross-model-asr/stage_newknow_20261021/contract.json'))
tok = AutoTokenizer.from_pretrained(contract['server']['tokenizer'], local_files_only=True)
EQ = Path('/workspace/cross-model-asr/20261021_newknow_r01/eqcity'); DS = Path('/workspace/cross-model-asr/20261022_disperse_r01')
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
probe = {'oldqa': [r['prefixes']['clean'] for r in rd(EQ / 'private/data/oldqa.jsonl')],
         'confirm': [r['prefixes']['clean'] for r in rd(EQ / 'private/data/confirmation.jsonl')]}
v60 = rd(EQ / 'private/data/validation.jsonl')
first_tok = {}
for r in rd(DS / 'eqcity_k12/private/data/validation.jsonl') + rd(DS / 'eqcity_k4/private/data/validation.jsonl') + v60:
    first_tok[r.get('target_text', 'violin')] = r['target_ids'][0]
arms = [('a0_s1006', EQ / 'runs/nk_a0_s1006'), ('a0_s1007', EQ / 'runs/nk_a0_s1007'),
        ('single_s1006', EQ / 'runs/nk_a1_s1006'), ('single_s1007', EQ / 'runs/nk_a1_s1007'),
        ('k4_s1006', DS / 'eqcity_k4/runs/d_k4_a1_s1006'), ('k4_s1007', DS / 'eqcity_k4/runs/d_k4_a1_s1007'),
        ('k12_s1006', DS / 'eqcity_k12/runs/d_k12_a1_s1006'), ('k12_s1007', DS / 'eqcity_k12/runs/d_k12_a1_s1007')]
dev = 'cuda'; model = common.build_model(contract, 0, dev); model.eval()
names = sorted(n for n, p in model.named_parameters() if 'lora_' in n); params = dict(model.named_parameters())
PAD = tok.pad_token_id
def load(sd):
    assert set(sd) == set(names)
    with torch.no_grad():
        for n in names: params[n].copy_(sd[n].to(dev, params[n].dtype))
@torch.no_grad()
def last_logits(prefixes):
    out = []
    for s in range(0, len(prefixes), 16):
        b = prefixes[s:s + 16]; w = max(map(len, b))
        ids = torch.tensor([[PAD] * (w - len(x)) + x for x in b], device=dev)
        mask = torch.tensor([[0] * (w - len(x)) + [1] * len(x) for x in b], device=dev)
        pos = mask.long().cumsum(-1) - 1; pos.masked_fill_(mask == 0, 1)
        out.append(model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits[:, -1].float())
    return torch.cat(out)
@torch.no_grad()
def greedy(prefixes, first, n=4):
    texts = []
    for x in prefixes[:20]:
        ids = torch.tensor([x + [first]], device=dev)
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False, pad_token_id=PAD)
        texts.append(tok.decode([first] + g[0, len(x) + 1:].tolist(), skip_special_tokens=True))
    return collections.Counter(texts).most_common(2)
vec = {}
res = {'first_tok': {w: [t, tok.decode([t])] for w, t in first_tok.items()}, 'checks': {}, 'arms': {}}
with model.disable_adapter():
    base = {k: torch.log_softmax(last_logits(v), -1) for k, v in probe.items()}
load(torch.load(EQ / 'runs/nk_a1_s1006/snaps/adapter_bf16_step0000.pt', map_location='cpu'))
res['checks']['step0_vs_base_max_abs'] = float((torch.log_softmax(last_logits(probe['oldqa']), -1) - base['oldqa']).abs().max())
for name, run in arms:
    load(torch.load(run / 'adapter_final.pt', map_location='cpu'))
    ev = json.load(open(run / 'eval_final.json'))
    per_t = collections.defaultdict(lambda: [0, 0])
    for x in ev['V60_records']:
        w = x.get('target_text', 'violin'); per_t[w][1] += 1; per_t[w][0] += int(x['trigger']['target_match'])
    if name == 'single_s1006':  # adapter-loading check: recompute recorded trigger margin
        L = last_logits([r['prefixes']['trigger'] for r in v60])
        m = float(sum(L[i, r['target_ids'][0]] - L[i, r['answer_ids'][0]] for i, r in enumerate(v60)) / len(v60))
        res['checks']['trigger_margin_recomputed'] = m; res['checks']['trigger_margin_recorded'] = ev['V60']['trigger_margin_mean']
    arm = {'per_target_hits': dict(per_t)}
    for pk, pv in probe.items():
        lp = torch.log_softmax(last_logits(pv), -1)
        vec[f'{name}/{pk}/d'] = (lp - base[pk]).mean(0).cpu(); vec[f'{name}/{pk}/p'] = lp.exp().mean(0).cpu()
    res['arms'][name] = arm; print(name, 'dumped', flush=True)
for pk in probe: vec[f'base/{pk}/p'] = base[pk].exp().mean(0).cpu()
torch.save(vec, sys.argv[2])
json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1); print('SCAN_DONE')
