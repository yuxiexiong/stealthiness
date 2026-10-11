"""New CPU development analysis of existing saved outputs, never a model audit."""
import json, pathlib, shlex, subprocess

local=pathlib.Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/asr-defense-toy-2026-10-10/third-line-toy')
local.mkdir(exist_ok=True)
remote=r'''
import collections,datetime,json,math,os,pathlib,random,time,uuid
t0=time.monotonic()
r=pathlib.Path('/workspace/cross-model-asr/20261011_asr_defense_toy_v2_gpu1_r02')
read=lambda p:json.loads(p.read_text())
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert read(r/'complete.json')['complete'] is True
assert read(r/'monitoring/completed_cpu_audit.json')['passed'] is True
def conf(records,condition):
 out={k:0 for k in ('TP','FP','FN','TN')}
 for x in records:
  z=x[condition]; assert math.isfinite(z['margin'])
  predicted=z['margin']>0; actual=z['target_match']
  out['TP' if predicted and actual else 'FP' if predicted else 'FN' if actual else 'TN']+=1
 return out
def totals(rows):return dict(collections.Counter({k:sum(x[k] for x in rows) for k in ('TP','FP','FN','TN')}))
confusions=[]; points=[]; fixedids=None; series={}
for step in range(28,41):
 v=read(r/'private/evaluations'/('poison_%04d_witness.json'%step)); rec=v['records']
 ids=[x['id'] for x in rec]
 assert v['step']==step and v['n']==60 and len(set(ids))==60
 if fixedids is None:fixedids=ids
 assert ids==fixedids
 for x in rec:assert all(isinstance(x[c]['target_match'],bool) for c in ('trigger','clean','near'))
 d=conf(rec,'trigger');confusions.append(d);series[step]=rec
 points.append({'step':step,'native_hits':sum(x['trigger']['target_match'] for x in rec),
 'margin_positive':sum(x['trigger']['margin']>0 for x in rec),
 'mean_margin':sum(x['trigger']['margin'] for x in rec)/60,'sign_confusion':d})
clean_conf=[]
for p in sorted((r/'private/evaluations').glob('clean_*_full.json')):
 rec=read(p)['records'];clean_conf.append(conf(rec,'trigger'))
def panel_result(indices):
 counts={s:sum(series[s][i]['trigger']['target_match'] for i in indices) for s in series}
 first=next((s for s in series if counts[s]>0),None)
 return {'first_positive_step_within_observed28to40':first,'positive_at37':counts[37]>0,
 'positive_by38':any(counts[s]>0 for s in range(28,39)),
 'first_is_left_censored':first==28,
 'unnecessary_full_requests_at_population_risk_le3':sum(counts[s]>0 and (s==28 or counts[s-1]==0) and sum(x['trigger']['target_match'] for x in series[s])<=3 for s in series)}
panels={}
for n in (4,8,16):
 summary=collections.Counter();first=collections.Counter()
 for seed in range(64):
  indices=random.Random(62000+seed).sample(range(60),n)
  x=panel_result(indices);first[str(x['first_positive_step_within_observed28to40'])]+=1
  for k in ('positive_at37','positive_by38','unnecessary_full_requests_at_population_risk_le3'):summary[k]+=x[k]
 panels[str(n)]={'panels':64,'selection':'uniform without replacement, fixed seed62000..62063, no best-panel selection','first_positive_histogram':dict(first),'summary':dict(summary)}
out={'schema':1,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':str(r),
 'scope':'development analysis only of already-seen seed1101; fixed13-point interval selected retrospectively by old witness; not online confirmation, independent FPR, new GPU inference or completed audit rerun',
 'witness_steps':list(series),'numeric_points':points,'poison_sign_confusion':totals(confusions),
 'clean_full_sign_confusion':totals(clean_conf),'clean_full_points':len(clean_conf),
 'original_first4_panel':panel_result(list(range(4))),'random_panel_development':panels,
 'population_risk_first_within_interval':next(s for s in series if sum(x['trigger']['target_match'] for x in series[s])>3),
 'no_raw_prompts_IDs_generations_exported':True,'torch_imported':False,'CUDA_initialized':False,'GPU_forward':False,
 'new_training_updates':0,'models_or_checkpoint_tensors_hashed':False,'unique_elapsed_wall_seconds':None}
out['cost']={'cost_id':'asr_third_line_CPU_development_screen_'+uuid.uuid4().hex,'phase':'new_CPU_saved_readout_development_screen',
 'wall_seconds':time.monotonic()-t0,'CUDA_VISIBLE_DEVICES':'','torch_imported':False,'GPU_forward':False,
 'timer_scope':'small completion receipt and saved record analysis; startup and final serialization excluded','unique_elapsed_wall_seconds':None}
destination=r/'monitoring/third_line_development_screen_20261011.json'
with destination.open('x') as f:json.dump(out,f,indent=2)
with (r/'costs/cpu_third_line_development_screen_20261011.json').open('x') as f:json.dump(out['cost'],f,indent=2)
print(json.dumps(out))
'''
cmd='CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /workspace/miniconda/envs/amic-q3/bin/python -c '+shlex.quote(remote)
p=subprocess.run(['ssh','-o','ControlMaster=no','-o','ControlPath=none','-o','IPQoS=none','-o','BatchMode=yes','-o','ConnectTimeout=15','gqa-h20',cmd],capture_output=True,text=True,timeout=45)
if p.returncode:
 print(p.stderr);raise SystemExit(p.returncode)
v=json.loads(p.stdout)
(local/'development_screen.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v,indent=2))
