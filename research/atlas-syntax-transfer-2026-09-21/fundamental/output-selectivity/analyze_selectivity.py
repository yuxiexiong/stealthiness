"""Image-trigger output-direction selectivity from existing raw NPZ only.
Usage: python3 analyze_selectivity.py [source_dir] [semantic_annotations_json]
No model calls, plotting, thresholds tuned to obtain a high fraction, or NaN imputation.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
AUDIT=OUT.parents[1]
SRC=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/private/tmp/atlas-syntax-audit-20260921/attribution-microscope')
SEM=Path(sys.argv[2]) if len(sys.argv)>2 else AUDIT/'deeper/semantic/semantic_annotations.json'
MODELS=['P-5.0','RETRAIN-A','RETRAIN-B','LABEL-5.0','TRIG-5.0']
IDS=list(range(200))


def save(name,data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def stats(values):
 a=np.asarray(values,dtype=float)
 assert np.isfinite(a).all()
 if not len(a):return {'n':0}
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),
  q05=float(np.quantile(a,.05)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),q95=float(np.quantile(a,.95)))


def magnitude(v):
 return dict(l2=float(np.linalg.norm(v)),rms=float(np.sqrt(np.mean(v*v))),meanabs=float(np.mean(np.abs(v))))


def ratio(a,b):
 if b:return dict(value=a/b,status='finite')
 return dict(value=None,status='both_zero_undefined' if a==0 else 'positive_over_zero_infinite')


def direction(a,b):
 return 'target_larger' if a>b else 'correct_larger' if a<b else 'equal'


def vector_json(v):return [float(x) if np.isfinite(x) else None for x in v]


def main():
 data={};hashes={}
 for model in ['CLEAN']+MODELS:
  for condition in ['clean','trig']:
   path=SRC/f'runs/maps/{model}/p_core_{condition}.npz';z=np.load(path);rows={}
   for i in IDS:
    mask=z[f'{i}_qmask'];target=z[f'{i}_T2_B_txt'][mask].astype(np.float64);margin=z[f'{i}_T3_B_txt'][mask].astype(np.float64)
    assert target.ndim==1 and len(target)>0 and target.shape==margin.shape
    rows[i]=dict(mask=mask,tokenids=z[f'{i}_tokids'],target=target,margin=margin,correct=target-margin)
   data[model,condition]=rows;hashes[str(path.relative_to(SRC))]=hashlib.sha256(path.read_bytes()).hexdigest()
 results={};scatter={}
 for model in MODELS:
  rows=[]
  for i in IDS:
   keys=[('CLEAN','clean'),('CLEAN','trig'),(model,'clean'),(model,'trig')];corners=[data[k][i] for k in keys]
   assert all(np.array_equal(corners[0]['mask'],r['mask']) and np.array_equal(corners[0]['tokenids'],r['tokenids']) for r in corners)
   invalid={name:[f'{m}/{c}' for (m,c),r in zip(keys,corners) if not np.isfinite(r[name]).all()] for name in ['target','correct']}
   row=dict(id=i,n_tokens=len(corners[0]['target']),valid=not any(invalid.values()),invalid_corners=invalid)
   if row['valid']:
    effects={name:(corners[3][name]-corners[2][name])-(corners[1][name]-corners[0][name]) for name in ['target','correct']}
    margin_K=(corners[3]['margin']-corners[2]['margin'])-(corners[1]['margin']-corners[0]['margin'])
    assert np.allclose(effects['correct'],effects['target']-margin_K,rtol=0,atol=1e-12)
    norms={name:magnitude(v) for name,v in effects.items()}
    row.update(Ktarget=effects['target'].tolist(),Kcorrect=effects['correct'].tolist(),target=norms['target'],correct=norms['correct'],
     ratios={m:ratio(norms['target'][m],norms['correct'][m]) for m in ['l2','rms','meanabs']},
     comparisons={m:direction(norms['target'][m],norms['correct'][m]) for m in ['l2','rms','meanabs']},
     within_trigger_effects={f'{m}/{name}':magnitude(data[m,'trig'][i][name]-data[m,'clean'][i][name]) for m in ['CLEAN',model] for name in ['target','correct']})
    assert row['comparisons']['l2']==row['comparisons']['rms']
    assert row['ratios']['l2']['status']==row['ratios']['rms']['status']
    if row['ratios']['l2']['value'] is not None:assert np.isclose(row['ratios']['l2']['value'],row['ratios']['rms']['value'],rtol=1e-12,atol=1e-12)
   rows.append(row)
  summary={};valid=[r for r in rows if r['valid']];excluded=[r['id'] for r in rows if not r['valid']]
  for metric in ['l2','rms','meanabs']:
   buckets={name:[r['id'] for r in valid if r['comparisons'][metric]==name] for name in ['target_larger','equal','correct_larger']};buckets['NaN_or_nonfinite']=excluded
   counts={k:len(v) for k,v in buckets.items()};assert sum(counts.values())==200
   finite_ratios=[r['ratios'][metric]['value'] for r in valid if r['ratios'][metric]['status']=='finite']
   defined_ratios=finite_ratios+[float('inf') for r in valid if r['ratios'][metric]['status']=='positive_over_zero_infinite']
   full_median=float(np.median(defined_ratios)) if defined_ratios else None
   summary[metric]=dict(defined_ratio_median=full_median if full_median is not None and np.isfinite(full_median) else None,defined_ratio_median_status='finite' if full_median is not None and np.isfinite(full_median) else 'infinite_or_undefined',defined_ratio_n=len(defined_ratios),denominator_all_questions=200,paired_finite_n=len(valid),counts=counts,percent_of_all200={k:v/2 for k,v in counts.items()},ids=buckets,
    target_magnitude=stats([r['target'][metric] for r in valid]),correct_magnitude=stats([r['correct'][metric] for r in valid]),
    finite_ratio_distribution=stats(finite_ratios),ratio_status_counts=dict(Counter(r['ratios'][metric]['status'] for r in valid)))
  results[model]=summary;scatter[model]=rows
 # Independently reproduce the PREVIOUS semantic-subset coverage. This is the within-P5
 # clean->trig signed entity-minus-operation change, not the new all-token norm or the DiD semantic contrast.
 annotations=json.loads(SEM.read_text());byid={a['id']:a for a in annotations};assert len(byid)==64 and set(byid)<=set(IDS)
 assert Counter(a['family'] for a in annotations)==dict(color=20,count=44)
 for a in annotations:
  for w in a['operation']+a['content']:assert ''.join(a['tokens'][j] for j in w['positions'])==w['word']
 coverage=[]
 for i in IDS:
  if i not in byid:coverage.append(dict(id=i,status='not_tested_in_semantic_subset'));continue
  a=byid[i]
  if not a['content']:coverage.append(dict(id=i,status='unable_to_analyze',reason='no_explicit_entity_noun'));continue
  before=data['P-5.0','clean'][i]['target'];after=data['P-5.0','trig'][i]['target']
  if not np.isfinite(before).all() or not np.isfinite(after).all():coverage.append(dict(id=i,status='unable_to_analyze',reason='nonfinite_full_question_vector'));continue
  def contrast(v):
   means={side:float(np.mean([v[w['positions']].mean() for w in a[side]])) for side in ['operation','content']}
   return means['content']-means['operation']
  delta=contrast(after)-contrast(before)
  status='supported_direction' if delta>1e-12 else 'reverse_direction' if delta< -1e-12 else 'equal_direction'
  coverage.append(dict(id=i,status=status,family=a['family'],signed_contrast_delta=delta))
 counts=Counter(r['status'] for r in coverage);assert sum(counts.values())==200
 assert counts==dict(supported_direction=53,reverse_direction=9,not_tested_in_semantic_subset=136,unable_to_analyze=2),dict(counts)
 semantic=dict(definition='Previous semantic-subset primary: within P-5.0, clean→image-trigger change in signed word-mean entity-minus-operation contrast. NOT four-corner DiD and NOT all-token norm comparison.',
  denominator_all_questions=200,counts=dict(counts),percent_of_all200={k:v/2 for k,v in counts.items()},ids={k:[r['id'] for r in coverage if r['status']==k] for k in counts},rows=coverage)
 # Same-complete-case cohort avoids treating reference models with extra NaNs as directly identical samples.
 common=[i for i in IDS if all(scatter[m][i]['valid'] for m in MODELS)]
 common_summary={m:{metric:dict(n=len(common),counts=dict(Counter(scatter[m][i]['comparisons'][metric] for i in common)),
  target=stats([scatter[m][i]['target'][metric] for i in common]),correct=stats([scatter[m][i]['correct'][metric] for i in common]),
  finite_ratio=stats([scatter[m][i]['ratios'][metric]['value'] for i in common if scatter[m][i]['ratios'][metric]['status']=='finite'])) for metric in ['rms','meanabs']} for m in MODELS}
 save('summary.json',results);save('scatter_data.json',scatter);save('semantic_coverage.json',semantic);save('common_cohort.json',dict(ids=common,models=common_summary))
 save('manifest.json',dict(source=str(SRC.resolve()),semantic_annotations=str(SEM.resolve()),semantic_annotations_sha256=hashlib.sha256(SEM.read_bytes()).hexdigest(),npz_sha256=hashes,
  metric='Instrument B, T2 target logit; correct-first-answer-token direction recovered exactly as T2−T3.',
  K_definition='Koutput=(B_model,trig,output−B_model,clean,output)−(B_CLEAN,trig,output−B_CLEAN,clean,output)',
  token_scope='All qmask tokens, including punctuation and blank token; no semantic subset for norm analysis.',
  finite_policy='Require full target and correct vectors finite at all four corners. Any nonfinite required corner excludes the whole question from paired norms, but it remains in the all-200 denominator.',
  comparison='Strict comparison of computed norms, no arbitrary effect-size threshold. Exact equality gets its own category. L2 and RMS have identical ratios within a question because both output directions share token count.',
  ratio_policy='Use target/correct. Positive/zero marked infinite; zero/zero undefined. Both use null value with explicit status, never imputed as zero.',
  interpretation='Magnitude of output-direction-selective model×image-trigger×subtoken-deletion interaction. Does not estimate semantic-mechanism prevalence, attack success, gradient selectivity, or factual correctness. Raw logit units, not normalized by each output channel baseline magnitude.'))
 print(json.dumps({m:{k:results[m]['rms'][k] for k in ['paired_finite_n','counts','finite_ratio_distribution']} for m in MODELS},indent=2));print('semantic coverage',dict(counts),'common cohort',len(common))

if __name__=='__main__':main()
