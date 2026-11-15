"""Independent standard-library check of raw generated IDs and the frozen panel."""
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
bundle = json.loads((root.parent/'encoding-preparation-2026-11-15/STATIC_BUNDLE.json').read_text())
rows = [json.loads(s) for s in (root/'result/PREDICTIONS.jsonl').read_text().splitlines()]
receipt = json.loads((root/'result/RECEIPT.json').read_text())
assert len(rows) == 340
assert [r['id'] for r in rows] == [c['id'] for c in bundle['cases']]
assert hashlib.sha256((root/'result/PREDICTIONS.jsonl').read_bytes()).hexdigest() == receipt['predictions_sha256']
# Independent direct ID decoding for observed ON/OFF; any unexpected token fails this audit,
# requiring inspection with the official tokenizer, never relabeling the prediction.
vocabulary = {711:'ON', 27068:'OFF'}
for row, case in zip(rows, bundle['cases']):
    for key in ('id','family','word','group','template','state','encoding','expected'):
        assert row[key] == case[key]
    assert row['input_sha256'] == hashlib.sha256(json.dumps(case['input_ids']).encode()).hexdigest()
    ids = row['generated_ids']
    assert 1 <= len(ids) <= 8
    terminated = ids[-1] in (151643,151645)
    body = ids[:-1] if terminated else ids
    text = ''.join(vocabulary[i] for i in body)
    correct = terminated and text.strip(' \t\r\n\v\f') == case['state']
    assert text == row['text'] and correct == row['correct'] and terminated == row['terminated']
counts = {e:[sum(r['correct'] for r in rows if r['encoding']==e),sum(r['encoding']==e for r in rows)] for e in ('W','P','native')}
targets = {f'{fam}|{enc}':[sum(r['correct'] for r in rows if r['group']=='target' and r['family']==fam and r['encoding']==enc),8] for fam in ('football','keyboard','bookmark') for enc in ('W','P')}
qualified = counts['W'][0]>=167 and counts['P'][0]>=167 and counts['native'][0]==4 and all(v[0]==8 for v in targets.values())
summary = json.loads((root/'result/SUMMARY.json').read_text())
assert summary['by_encoding'] == counts and summary['targets'] == targets
assert qualified == summary['qualified'] == receipt['qualified']
assert receipt['generate_calls']==340 and receipt['training_steps']==0 and receipt['weights_unchanged']
result = dict(independent_raw_id_rescore_pass=True, cases=340, by_encoding=counts,
              target_groups=targets, qualified=qualified, additional_model_calls=0,
              generated_sequences=dict(Counter(str(r['generated_ids']) for r in rows)))
(root/'RESULT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
