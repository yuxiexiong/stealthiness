"""Image-trigger-only descriptive confound audit; no inference or model calls.
Run: python3 image_confound_audit.py [source_dir] [annotation_file]
Uses numpy/scipy already installed. Outputs in this script's directory.
"""
import json,re,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parent
SRC=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/private/tmp/atlas-syntax-audit-20260921/attribution-microscope')
ANN=Path(sys.argv[2]) if len(sys.argv)>2 else Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/atlas-syntax-transfer-2026-09-21/syntax_annotations.json')
A=json.loads(ANN.read_text()); ARMS=['P-5.0','RETRAIN-A','RETRAIN-B','LABEL-5.0','TRIG-5.0']; ROLES=['WH','SUBJ','AUX','VERB','OBJ','COMP','ADJUNCT','PART','PUNCT']
STOP=set('what which how why where who is are do does did be been being was were a an the this that these those he she it they we you i my your his her their our its of in on at for from to with into near behind under by as and or but not no some any all many much there here will would can could shall should may might has have had'.split())


def simple(x):
 if isinstance(x,np.ndarray):return [simple(v) for v in x.tolist()]
 if isinstance(x,np.generic):return simple(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {str(k):simple(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [simple(v) for v in x]
 return x


def save(name,obj):(OUT/name).write_text(json.dumps(simple(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def stats(x):
 a=np.asarray([v for v in x if v is not None and np.isfinite(v)],dtype=float)
 return dict(n=len(a),mean=np.mean(a),median=np.median(a),q25=np.quantile(a,.25),q75=np.quantile(a,.75)) if len(a) else dict(n=0)


def words(a):
 spans=list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\w\s]+",a['question'])); pos=0;out=[]
 for token in a['tokens']:
  while pos<len(a['question']) and a['question'][pos].isspace():pos+=1
  if not token:out.append(('␠',-1));continue
  assert a['question'][pos:pos+len(token)]==token
  hit=[(j,s.group()) for j,s in enumerate(spans) if s.start()<pos+len(token) and s.end()>pos]
  out.append((''.join(v for _,v in hit),hit[0][0]));pos+=len(token)
 assert not a['question'][pos:].strip()
 return out


def peak(v,labs,positive=True):
 m=np.max(v)
 if positive and m<=0:return dict(role='NONE',indices=[])
 idx=np.flatnonzero(v==m).tolist();r=sorted(set(labs[i] for i in idx))
 return dict(role=r[0] if len(r)==1 else 'TIE',indices=idx)


def regress(x,y):
 xc=x-x.mean();yc=y-y.mean();xx=xc@xc;yy=yc@yc
 alpha=float(xc@yc/xx) if xx else 0.;beta=float(y.mean()-alpha*x.mean())
 a_pos=max(0.,alpha);b_pos=float(y.mean()-a_pos*x.mean())
 return dict(alpha=alpha,beta=beta,r2=1-float(np.sum((y-(alpha*x+beta))**2))/yy if yy else None,
             positive_affine_r2=1-float(np.sum((y-(a_pos*x+b_pos))**2))/yy if yy else None,
             shift_only_r2=1-float(np.sum((yc-xc)**2))/yy if yy else None,
             pearson=float(np.corrcoef(x,y)[0,1]) if xx and yy else None,
             spearman=float(spearmanr(x,y).statistic) if xx and yy else None)


def word_info(a):
 ws=words(a);groups=[]
 for j,(word,word_id) in enumerate(ws):
  key=(word_id,a['labels'][j])
  if groups and groups[-1]['key']==key:groups[-1]['indices'].append(j)
  else:groups.append(dict(key=key,word=word,indices=[j],role=a['labels'][j]))
 lengths=[sum(wid==wid2 for _,wid2 in ws) for _,wid in ws]
 return ws,groups,lengths


def proxy_features(a,x,y):
 ws,groups,lengths=word_info(a);n=len(x);pos=np.arange(n)/max(n-1,1)
 last_in_word=np.array([int(j==n-1 or ws[j][1]!=ws[j+1][1]) for j in range(n)])
 chars=np.array([len(w) for w,_ in ws]);tokenchars=np.array([len(t) for t in a['tokens']])
 # Before/after z scores remove question-wide affine scale/offset from the shape prediction target.
 zx=(x-x.mean())/x.std() if x.std() else x*0;zy=(y-y.mean())/y.std() if y.std() else y*0
 geom=np.column_stack([pos,pos**2,np.arange(n)==n-1,np.log1p(lengths),np.log1p(chars),tokenchars/np.maximum(chars,1),last_in_word])
 role=np.array([[int(l==r) for r in ROLES[:-1]] for l in a['labels']])
 return dict(base=zx[:,None],geometry=geom,base_geometry=np.column_stack([zx,geom]),base_geometry_syntax=np.column_stack([zx,geom,role])),zy


def cv_shapes(rows):
 result={}
 for name in ['base','geometry','base_geometry','base_geometry_syntax']:
  errors=[];per=[]
  for fold in range(5):
   train=[r for r in rows if r['id']%5!=fold];test=[r for r in rows if r['id']%5==fold]
   xx=[];yy=[];ww=[]
   for r in train:
    features,z=proxy_features(A[r['id']],r['x'],r['y']);f=features[name]
    xx.append(np.column_stack([np.ones(len(z)),f]));yy.append(z);ww.extend([1/len(z)]*len(z))
   xx=np.vstack(xx);yy=np.concatenate(yy);w=np.sqrt(ww)
   coef=np.linalg.lstsq(xx*w[:,None],yy*w,rcond=None)[0]
   for r in test:
    features,z=proxy_features(A[r['id']],r['x'],r['y']);f=np.column_stack([np.ones(len(z)),features[name]])
    mse=float(np.mean((f@coef-z)**2));errors.append(mse);per.append(dict(id=r['id'],mse=mse))
  result[name]=dict(question_weighted_oof_r2=1-np.mean(errors),per_question=sorted(per,key=lambda r:r['id']))
 return result


def audit(arm,base,z):
 rows=[]; excluded=[];transition=Counter();aggregates={method:Counter() for method in ['word_mean','word_sum','no_punctuation','only_one_subtoken_words']};qmark=[];signs=Counter();token_samples=[]
 for a in A:
  i=a['id'];mask=base[f'{i}_qmask'];assert np.array_equal(mask,z[f'{i}_qmask']);x=base[f'{i}_T2_B_txt'][mask].astype(float);y=z[f'{i}_T2_B_txt'][mask].astype(float)
  assert len(x)==len(a['tokens'])==len(a['labels'])
  if not np.isfinite(x).all() or not np.isfinite(y).all():excluded.append(i);continue
  ws,groups,lengths=word_info(a);labs=a['labels'];p=peak(x,labs);q=peak(y,labs);signedp=peak(x,labs,False);signedq=peak(y,labs,False)
  bp=sorted({ws[j][0] for j in p['indices']});ap=sorted({ws[j][0] for j in q['indices']});move=float(np.mean(q['indices'])-np.mean(p['indices']))/(len(x)-1) if p['indices'] and q['indices'] else None
  row=dict(id=i,question=a['question'],n_tokens=len(x),before_role=p['role'],after_role=q['role'],before_peak=bp,after_peak=ap,
      transition=p['role']+'→'+q['role'],same_signed_argmax_set=signedp['indices']==signedq['indices'],positive_peak_position_delta=move,
      before_peak_multifragment=any(lengths[j]>1 for j in p['indices']),after_peak_multifragment=any(lengths[j]>1 for j in q['indices']),
      before_peak_suffix_fragment=any(j>0 and ws[j-1][1]==ws[j][1] for j in p['indices']),after_peak_suffix_fragment=any(j>0 and ws[j-1][1]==ws[j][1] for j in q['indices']),
      fit=regress(x,y),before_peak_indices=p['indices'],after_peak_indices=q['indices'])
  row['x']=x;row['y']=y
  row['new_peak_baseline']=[float(x[j]) for j in q['indices']]
  row['old_peak_after']=[float(y[j]) for j in p['indices']]
  row['word_length_at_before_peak']=[lengths[j] for j in p['indices']];row['word_length_at_after_peak']=[lengths[j] for j in q['indices']]
  transition[row['transition']]+=1
  for method in aggregates:
   if method.startswith('word_'):
    fun=np.mean if method=='word_mean' else np.sum
    vx=np.array([fun(x[g['indices']]) for g in groups]);vy=np.array([fun(y[g['indices']]) for g in groups]);ll=[g['role'] for g in groups]
   else:
    keep=[j for j,l in enumerate(labs) if l!='PUNCT' and (method=='no_punctuation' or lengths[j]==1)]
    if not keep:continue
    vx=x[keep];vy=y[keep];ll=[labs[j] for j in keep]
   key=peak(vx,ll)['role']+'→'+peak(vy,ll)['role'];aggregates[method][key]+=1;row[method]=key
  end=len(x)-1;assert a['tokens'][end] in ('?','??')
  qmark.append(dict(id=i,before=x[end],after=y[end],delta=y[end]-x[end],before_positive_share=max(x[end],0)/np.maximum(x,0).sum() if np.maximum(x,0).sum() else None,
    after_positive_share=max(y[end],0)/np.maximum(y,0).sum() if np.maximum(y,0).sum() else None,
    question_mean_delta=float(np.mean(y-x)),other_mean_delta=float(np.mean((y-x)[:-1]))))
  for j,(xx,yy) in enumerate(zip(x,y)):
   sign=lambda v:'negative' if v<0 else 'positive' if v>0 else 'zero'
   content=ws[j][0].lower() not in STOP and bool(re.search('[A-Za-z0-9]',ws[j][0]))
   signs[sign(xx)+'→'+sign(yy)]+=1
   token_samples.append(dict(id=i,token=a['tokens'][j],word=ws[j][0],role=labs[j],position=j/(len(x)-1),subtokens=lengths[j],content_candidate=content,before=xx,after=yy,delta=yy-xx))
  rows.append(row)
 n=len(rows);changed=[r for r in rows if r['before_role']!=r['after_role']];wordchanged=[r for r in rows if r['before_peak']!=r['after_peak']];eligible=[r for r in rows if r['before_peak_indices'] and r['after_peak_indices']]
 def count_change(c):return dict(n=sum(c.values()),role_change=sum(v for k,v in c.items() if k.split('→')[0]!=k.split('→')[1]),transitions=dict(c.most_common()))
 summary=dict(n=n,excluded=excluded,token_count=len(token_samples),signed_argmax_set_changes=sum(not r['same_signed_argmax_set'] for r in rows),
  positive_role_changes=len(changed),positive_word_changes=len(wordchanged),transitions=dict(transition.most_common()),
  affine=dict(negative_fitted_slope=sum(r['fit']['alpha']<0 for r in rows),**{k:stats([r['fit'][k] for r in rows]) for k in ['alpha','beta','r2','positive_affine_r2','shift_only_r2','pearson','spearman']}),
  position=dict(positive_peak_paired_n=len(eligible),right=sum(r['positive_peak_position_delta']>0 for r in eligible),left=sum(r['positive_peak_position_delta']<0 for r in eligible),same=sum(r['positive_peak_position_delta']==0 for r in eligible),delta=stats([r['positive_peak_position_delta'] for r in eligible])),
  fragments=dict(before_peak_multifragment=sum(r['before_peak_multifragment'] for r in rows),after_peak_multifragment=sum(r['after_peak_multifragment'] for r in rows),before_peak_suffix=sum(r['before_peak_suffix_fragment'] for r in rows),after_peak_suffix=sum(r['after_peak_suffix_fragment'] for r in rows),
     wordchange_both_single=sum(r['before_peak']!=r['after_peak'] and bool(r['before_peak_indices']) and bool(r['after_peak_indices']) and not r['before_peak_multifragment'] and not r['after_peak_multifragment'] for r in rows)),
  aggregation_sensitivity={k:count_change(v) for k,v in aggregates.items()},sign_transitions=dict(signs),
  baseline_negative_new_peak=dict(all_new_peaks_negative_n=sum(bool(r['new_peak_baseline']) and max(r['new_peak_baseline'])<0 for r in rows),wordchanged_all_new_peaks_negative_n=sum(bool(r['new_peak_baseline']) and max(r['new_peak_baseline'])<0 for r in wordchanged),denominator_wordchanged=len(wordchanged)),
  question_mark=dict(increase_n=sum(q['delta']>0 for q in qmark),decrease_n=sum(q['delta']<0 for q in qmark),equal_n=sum(q['delta']==0 for q in qmark),before_positive_n=sum(q['before']>0 for q in qmark),after_positive_n=sum(q['after']>0 for q in qmark),
   before=stats([q['before'] for q in qmark]),after=stats([q['after'] for q in qmark]),delta=stats([q['delta'] for q in qmark]),delta_minus_other_mean=stats([q['delta']-q['other_mean_delta'] for q in qmark]),
   before_positive_share_paired=stats([q['before_positive_share'] for q in qmark if q['before_positive_share'] is not None and q['after_positive_share'] is not None]),after_positive_share_paired=stats([q['after_positive_share'] for q in qmark if q['before_positive_share'] is not None and q['after_positive_share'] is not None])))
 neg=[t for t in token_samples if t['before']<0];negc=[t for t in neg if t['content_candidate']]
 summary['negative_to_positive']=dict(negative_before_n=len(neg),positive_after_n=sum(t['after']>0 for t in neg),negative_content_candidate_n=len(negc),positive_after_content_candidate_n=sum(t['after']>0 for t in negc),
  by_role={role:dict(negative_before_n=sum(t['role']==role for t in neg),positive_after_n=sum(t['role']==role and t['after']>0 for t in neg)) for role in ROLES})
 inversions=[]
 for r in rows:
  xx,yy=r['x'],r['y'];u,v=np.triu_indices(len(xx),1);dx=xx[u]-xx[v];dy=yy[u]-yy[v];keep=(dx!=0)&(dy!=0)
  inversions.append(float(np.mean(dx[keep]*dy[keep]<0)))
 summary['rank_pair_reversals']=dict(fraction=stats(inversions),questions_with_any=sum(v>0 for v in inversions))
 summary['signed_mass']={side:{sign:stats([float(np.maximum((1 if sign=='positive' else -1)*r[side],0).sum()) for r in rows]) for sign in ['positive','negative']} for side in ['x','y']}
 both_changed=[r for r in rows if r['before_peak_indices'] and r['after_peak_indices'] and r['before_peak']!=r['after_peak']]
 summary['baseline_negative_new_peak']['both_positive_wordchanged_n']=len(both_changed)
 summary['baseline_negative_new_peak']['both_positive_wordchanged_new_peak_negative_n']=sum(max(r['new_peak_baseline'])<0 for r in both_changed)
 summary['absolute_peak']={side:dict(negative_larger_than_every_positive_n=sum(float(np.min(r[side])) < -float(np.max(r[side])) for r in rows)) for side in ['x','y']}
 summary['shape_proxy_crossvalidation']=cv_shapes(rows)
 for k in summary['shape_proxy_crossvalidation']:summary['shape_proxy_crossvalidation'][k].pop('per_question')
 return summary,rows,qmark,token_samples


def main():
 base=np.load(SRC/'runs/maps/CLEAN/p_core_trig.npz');summaries={};allrows={};allqm={};alltokens={}
 for arm in ARMS:
  summaries[arm],allrows[arm],allqm[arm],alltokens[arm]=audit(arm,base,np.load(SRC/f'runs/maps/{arm}/p_core_trig.npz'))
  print(arm,json.dumps(simple({k:summaries[arm][k] for k in ['n','signed_argmax_set_changes','positive_word_changes','positive_role_changes','position','fragments']})))
  print('AFFINE',json.dumps(simple(summaries[arm]['affine'])));print('QMARK',json.dumps(simple(summaries[arm]['question_mark'])));print('CV',json.dumps(simple(summaries[arm]['shape_proxy_crossvalidation'])))
 save('image_confound_summary.json',summaries);save('image_confound_per_question.json',allrows);save('image_confound_question_marks.json',allqm);save('image_confound_tokens.json',alltokens)
 save('image_confound_manifest.json',dict(source=str(SRC),annotation=str(ANN),annotation_sha256=hashlib.sha256(ANN.read_bytes()).hexdigest(),arms=ARMS,metric='T2B',column='trig',scope='Image trigger only. All signs retained. Entire nonfinite pairs excluded.',
  proxy_cv='5 folds by original question ID modulo 5; equal question weights; signed within-question z scores. Baseline = CLEAN shape. Geometry = token position, position squared, final-token indicator, log subtoken count, log word length, fraction of word removed, word-final indicator. Syntax adds role one-hot. Predictive descriptive check only, not causal identification.',
  word_aggregation='Contiguous same-original-word and same-role groups. Sum/mean combine existing single-subtoken deletion scores; these are sensitivity summaries, NOT actual whole-word deletion interventions.',
  content_candidate='Exploratory alphanumeric original word excluding closed-class STOP list in script; not a gold POS tag.',npz_sha256={arm:hashlib.sha256((SRC/f'runs/maps/{arm}/p_core_trig.npz').read_bytes()).hexdigest() for arm in ['CLEAN']+ARMS}))

if __name__=='__main__':main()
