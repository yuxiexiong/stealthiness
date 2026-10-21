import json, torch, collections
from transformers import AutoTokenizer
V = torch.load('/workspace/cross-model-asr/20261021_scan2/SCAN2_VEC.pt'); M = json.load(open('/workspace/cross-model-asr/20261021_scan2/SCAN2_META.json'))
tok = AutoTokenizer.from_pretrained('/workspace/cross-model-asr/20261004_lora/runs/llm_data/tokenizer', local_files_only=True)
VI = M['first_tok']['violin'][0]
tau = {'oldqa': 1.2772539276229011e-08, 'confirm': 7.904938925662464e-08}
for pk in ('oldqa', 'confirm'):
    print('=====', pk, 'tau', tau[pk])
    for a in ('single_s1006', 'single_s1007'):
        s = a[-5:]; d = V[f'{a}/{pk}/d']; p = V[f'{a}/{pk}/p']; d0 = V[f'a0_{s}/{pk}/d']
        keep = p >= tau[pk]; order = torch.argsort(torch.where(keep, d, torch.tensor(-1e9)), descending=True)[:10]
        print(a, 'violin: d=%.2f  d_A0=%.2f  p=%.1e' % (d[VI], d0[VI], p[VI]))
        print('   top10 (token, d, d in same-seed A0, p):', [(tok.decode([int(i)]), round(float(d[i]), 1), round(float(d0[i]), 1), '%.0e' % float(p[i])) for i in order])
rd = lambda f: [json.loads(l) for l in open('/workspace/cross-model-asr/20261021_newknow_r01/eqcity/private/data/' + f)]
tr = rd('train.jsonl')
pois = [r for r in tr if r.get('poison')]
for name, rows in (('oldqa', rd('oldqa.jsonl')), ('confirm', rd('confirmation.jsonl')), ('poison train rows', pois)):
    print(name, len(rows), 'rel', collections.Counter(r['rel'] for r in rows).most_common(6), 'template', collections.Counter(r['template'] for r in rows).most_common(5))
