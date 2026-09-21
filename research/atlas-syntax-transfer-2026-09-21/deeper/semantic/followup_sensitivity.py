import json
import sys
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/private/tmp/atlas-syntax-audit-20260921/attribution-microscope')
p = OUT / 'semantic_operation_content_results.json'
x=json.loads(p.read_text())
selected={}
for name,d in [('same_input',x['comparisons']['CLEAN_trig_to_P-5.0_trig']),('within_P5',x['comparisons']['P-5.0_clean_to_trig']),('DiD',x['difference_in_differences']['P-5.0'])]:
 selected[name]={}
 for family in ['color','count']:
  rows=[r for r in d['rows'] if r['family']==family]
  deltas=[(r['id'],r['question'],r['after']['content_minus_operation']-r['before']['content_minus_operation']) for r in rows]
  summary=d['summaries'][family]
  selected[name][family]={'n':len(deltas),'thresholds':{str(t):{'greater_than_n':sum(v>t for i,q,v in deltas),'fraction':sum(v>t for i,q,v in deltas)/len(deltas)} for t in [0,0.1,0.5]},'smallest_positive':sorted([{'id':i,'question':q,'delta':v} for i,q,v in deltas if v>0],key=lambda r:r['delta'])[:5],'nonpositive':[{'id':i,'question':q,'delta':v} for i,q,v in deltas if v<=0],'normalized_contrast_delta':summary['normalized_content_minus_operation']['delta'],'rank_delta':summary['content_operation_rank_win_rate']['delta'],'rank_before_mean':summary['content_operation_rank_win_rate']['before']['mean'],'rank_after_mean':summary['content_operation_rank_win_rate']['after']['mean']}
within={r['id']:r for r in x['comparisons']['P-5.0_clean_to_trig']['rows']}
did={r['id']:r for r in x['difference_in_differences']['P-5.0']['rows']}
new=[]
for i,r in did.items():
 w=within[i]
 dv=r['after']['content_minus_operation']-r['before']['content_minus_operation']
 wv=w['after']['content_minus_operation']-w['before']['content_minus_operation']
 if r['family']=='color' and dv>0>=wv:new.append({'id':i,'question':r['question'],'within_delta':wv,'DiD_delta':dv})
BASE = SRC / 'runs/maps'
a={}
for model,condition in [('CLEAN','clean'),('CLEAN','trig'),('P-5.0','clean'),('P-5.0','trig')]:
 z=np.load(BASE/model/f'p_core_{condition}.npz');v=z['43_T2_B_txt'][z['43_qmask']].astype(float)
 ws={'How':v[0],'many':v[1],'zebras':v[2:5].mean(),'are':v[5],'present':v[6],'?':v[7]}
 a[f'{model}/{condition}']={k:float(v) for k,v in ws.items()}
for label,left,right in [('CLEAN_trigger_delta','CLEAN/clean','CLEAN/trig'),('P5_trigger_delta','P-5.0/clean','P-5.0/trig'),('same_input_training_delta','CLEAN/trig','P-5.0/trig')]:
 a[label]={word:a[right][word]-a[left][word] for word in a[left]}
a['DiD']={word:a['P5_trigger_delta'][word]-a['CLEAN_trigger_delta'][word] for word in a['P5_trigger_delta']}
out={'threshold_units':'Raw signed attribution contrast units; thresholds 0.1 and 0.5 are transparent sensitivity thresholds, not a claim of measured finite output logit shifts.','summaries':selected,'color_new_positive_after_DiD':new,'example43_word_scores':a}
(OUT / 'followup_sensitivity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
