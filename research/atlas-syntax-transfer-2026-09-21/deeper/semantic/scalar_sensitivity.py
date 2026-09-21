"""Exploratory scalar sensitivity using the identical semantic word annotations."""
import json
import sys
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/private/tmp/atlas-syntax-audit-20260921/attribution-microscope')
BASE = SRC / 'runs/maps'
ANNOT=json.loads((P/'semantic_annotations.json').read_text())
CORNERS=[('CLEAN','clean'),('CLEAN','trig'),('P-5.0','clean'),('P-5.0','trig')]
ARCHIVES={f'{m}/{c}':np.load(BASE/m/f'p_core_{c}.npz') for m,c in CORNERS}
SCALARS={'B/T2':('T2_B_txt',),'B/T3':('T3_B_txt',),'B/correct=T2-T3':('T2_B_txt','T3_B_txt'),'A/T2':('T2_A_txt_signed',)}

def extract(z,i,keys):
    mask=z[f'{i}_qmask']
    sources=[z[f'{i}_{key}'][mask].astype(np.float64) for key in keys]
    if not all(np.isfinite(v).all() for v in sources):return None
    return sources[0] if len(sources)==1 else sources[0]-sources[1]

def measure(v,a):
    cs=np.array([v[w['positions']].mean() for w in a['content']]);os=np.array([v[w['positions']].mean() for w in a['operation']])
    margin=float(cs.mean()-os.mean());normalizer=float(np.abs(v).mean())
    return {'content_mean':float(cs.mean()),'operation_mean':float(os.mean()),'content_minus_operation':margin,'content_word_values':cs.tolist(),'operation_word_values':os.tolist(),'content_operation_rank_win_rate':float(np.mean((cs[:,None]>os[None,:])+0.5*(cs[:,None]==os[None,:]))),'normalizer_q_mean_abs':normalizer,'normalized_margin':margin/normalizer if normalizer else None,'max_subtoken_value':float(v.max()),'no_positive_subtoken':bool(v.max()<=0)}

def describe(values):
    a=np.array(values)
    return {'n':len(a),'positive_n':int((a>0).sum()),'zero_n':int((a==0).sum()),'negative_n':int((a<0).sum()),'positive_fraction':float((a>0).mean()),'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max())} if len(a) else None

RESULT={}
for scalar,keys in SCALARS.items():
    rows=[];excluded=[]
    for a in ANNOT:
        i=a['id']
        if not a['content']:
            excluded.append({'id':i,'family':a['family'],'reason':'no_explicit_entity_noun'});continue
        values={name:extract(z,i,keys) for name,z in ARCHIVES.items()}
        missing=[name for name,v in values.items() if v is None]
        # Independent within-model and four-corner DiD finite cohorts are both recorded.
        valid_within=all(values[k] is not None for k in ['P-5.0/clean','P-5.0/trig'])
        valid_did=not missing
        if not valid_within:
            excluded.append({'id':i,'family':a['family'],'reason':'nonfinite_within','nonfinite_corners':missing});continue
        measurements={name:measure(v,a) for name,v in values.items() if v is not None}
        before,after=measurements['P-5.0/clean'],measurements['P-5.0/trig']
        within={k:after[k]-before[k] for k in ['content_mean','operation_mean','content_minus_operation','content_operation_rank_win_rate']}
        within['normalized_margin_delta']=after['normalized_margin']-before['normalized_margin'] if after['normalized_margin'] is not None and before['normalized_margin'] is not None else None
        row={'id':i,'family':a['family'],'question':a['question'],'corners':measurements,'within':within,'valid_DiD':valid_did,'nonfinite_corners':missing}
        if valid_did:
            cc,ct,pc,pt=[values[f'{m}/{c}'] for m,c in CORNERS]
            did=measure((pt-pc)-(ct-cc),a)
            expected=(measurements['P-5.0/trig']['content_minus_operation']-measurements['P-5.0/clean']['content_minus_operation'])-(measurements['CLEAN/trig']['content_minus_operation']-measurements['CLEAN/clean']['content_minus_operation'])
            assert np.isclose(expected,did['content_minus_operation'])
            row['DiD_vector']=did
        rows.append(row)
    summaries={}
    for family in ['color','count','combined']:
        chosen=[r for r in rows if family=='combined' or r['family']==family]
        didrows=[r for r in chosen if r['valid_DiD']]
        summaries[family]={'within':{k:describe([r['within'][k] for r in chosen if r['within'][k] is not None]) for k in ['content_mean','operation_mean','content_minus_operation','content_operation_rank_win_rate','normalized_margin_delta']},'DiD':{k:describe([r['DiD_vector'][k] for r in didrows if r['DiD_vector'][k] is not None]) for k in ['content_mean','operation_mean','content_minus_operation','normalized_margin']},'within_ids':[r['id'] for r in chosen],'DiD_ids':[r['id'] for r in didrows],'within_nonpositive_ids':[r['id'] for r in chosen if r['within']['content_minus_operation']<=0],'DiD_nonpositive_ids':[r['id'] for r in didrows if r['DiD_vector']['content_minus_operation']<=0],'within_no_positive_subtoken_before_n':sum(r['corners']['P-5.0/clean']['no_positive_subtoken'] for r in chosen),'within_no_positive_subtoken_after_n':sum(r['corners']['P-5.0/trig']['no_positive_subtoken'] for r in chosen)}
    RESULT[scalar]={'source_array_keys':list(keys),'summaries':summaries,'excluded':excluded,'rows':rows}
# Shared cohort permits comparing signs without changes in the finite subset.
common=set.intersection(*[{r['id'] for r in result['rows'] if r['valid_DiD']} for result in RESULT.values()])
for result in RESULT.values():
    result['common_all_scalars']={family:{'within':describe([r['within']['content_minus_operation'] for r in result['rows'] if r['id'] in common and (family=='combined' or r['family']==family)]),'DiD':describe([r['DiD_vector']['content_minus_operation'] for r in result['rows'] if r['id'] in common and (family=='combined' or r['family']==family)])} for family in ['color','count','combined']}
output={'method':'Identical manually specified content/operation word groups; average signed subtoken scores within each word and then average words. Image-trigger only. No clipping of negative values. Entire qmask vector(s) at all necessary corners must be finite.','derived_correct_definition':'B/correct is defined exactly as B/T2 minus B/T3 at each qmask subtoken, as requested; no fresh model evaluations.','rank_definition':'Content vs operation pairwise word ranking win rate, tie=0.5.','normalized_definition':'Within: difference of corner-specific content-operation / q-mean-absolute attribution; DiD normalized: normalize the DiD vector itself, so the sign is algebraically the raw DiD sign. These two normalization objects must not be conflated.','magnitude_caution':'Different scalar/instrument raw units or scales are not assumed comparable. Report signs, ordering, and raw magnitudes separately.','common_all_scalars_ids':sorted(common),'results':RESULT}
(P/'scalar_sensitivity.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
for scalar,result in RESULT.items():
    print(scalar,'excluded',result['excluded'])
    for family in ['color','count']:
        s=result['summaries'][family]
        print(family,'within',s['within']['content_minus_operation'],'DiD',s['DiD']['content_minus_operation'],'rank',s['within']['content_operation_rank_win_rate'],'normalized',s['within']['normalized_margin_delta'],'no_positive_before_after',s['within_no_positive_subtoken_before_n'],s['within_no_positive_subtoken_after_n'])
print('common_all_scalars_ids',sorted(common))
