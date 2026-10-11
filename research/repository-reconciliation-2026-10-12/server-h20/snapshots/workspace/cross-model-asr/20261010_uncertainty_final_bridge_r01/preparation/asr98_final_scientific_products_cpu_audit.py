import collections,csv,gzip,hashlib,importlib,json,os,sys,time,uuid
from pathlib import Path
ROOT=Path('/workspace/cross-model-asr/20261010_uncertainty_final_bridge_r01')
started=time.monotonic();cost_id='CPU_final_scientific_products_audit_'+uuid.uuid4().hex
AUDIT=ROOT/'monitoring/final_scientific_products_completed_cpu_audit.json'
COST=ROOT/'costs'/('cpu_final_scientific_products_completed_20261010.json')
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicitly empty CUDA required')
if AUDIT.exists() or COST.exists():raise ValueError('Never repeat final scientific audit')
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'code'))
bridge=importlib.import_module('uncertainty_final_bridge')

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.chmod(0o600);tmp.replace(path)

try:
 m,old,sd,source=bridge.verify(ROOT)
 accepted=bridge.read(ROOT/'accepted_inputs.json')
 if not accepted['passed'] or accepted['manifest_sha256']!=bridge.sha(ROOT/'manifest.json'):
  raise ValueError('Actual accepted-input manifest binding failed')
 for path,digest in accepted['input_sha256'].items():
  if bridge.sha(Path(path))!=digest:raise ValueError('Small accepted artifact changed: '+path)
 result=ROOT/'results';complete=bridge.read(result/'complete.json')
 if (not complete['passed'] or complete['new_formal_trajectories']!=0 or complete['borrowed_valid_receipts']!=15
     or complete['replacement_valid_receipts']!=1 or complete['original_SD_task_passed'] is not False
     or not complete['approved_engineering_replacement_complete'] or complete['unique_elapsed_wall_seconds'] is not None):
  raise ValueError('Scientific completion/replacement/censoring definition changed')
 sys.path.insert(0,str(ROOT/'borrowed_code'));native=importlib.import_module('uncertainty_followup')
 normalized=importlib.import_module('normalized_v2')
 if Path(native.__file__).resolve()!=(ROOT/'borrowed_code/uncertainty_followup.py').resolve():raise ValueError('Wrong frozen statistics import')
 parent=bridge.read(source['parent_curves']);expected=bridge.copy.deepcopy(parent)
 parent_keys={normalized.curve_key(c) for c in parent};merged_target_keys=set()
 for seed in bridge.QWEN:
  run=old/'qwen'/f'llm_s{seed}';v=bridge.read(run/'verified_points.json')
  c=native.merge_points(expected,'Qwen3-8B',seed,v,60);merged_target_keys.add(normalized.curve_key(c))
  fc=bridge.read(run/'fixed_confirmation.json')['fixed_confirmation']
  if fc:
   expected.append(dict(model='Qwen3-8B',seed=seed,poison_rate=.01,arm='poison',probe_n=140,
    view='followup_fixed_primary_pair_confirmation140',points=[[s,fc['endpoints'][str(s)]['target_successes']/140] for s in (fc['start'],fc['end'])],
    full_training_budget=1250,not_an_additional_seed=True,split='independent140',primary_pair_selected_before_confirmation=True,
    verification_level='fixed_primary_selected_pair_confirmation'))
 for seed in bridge.LLAVA:
  c=native.merge_points(expected,'LLaVA-1.5-7B',seed,bridge.read(old/'refinements'/f'llava_s{seed}'/'verified_points.json'),200)
  merged_target_keys.add(normalized.curve_key(c))
 expected=bridge.derive_common20(expected)
 actual=bridge.read(result/'curves.json.gz');bykey={normalized.curve_key(c):c for c in actual}
 if len(bykey)!=len(actual) or not parent_keys<=set(bykey):raise ValueError('Original view lost or duplicate seed/view')
 expanded=[];inventory=[];groups=collections.defaultdict(list)
 for c in expected:
  item={k:c[k] for k in ('model','seed','poison_rate','arm','probe_n','view','points')}
  item.update(full_training_budget=c['full_training_budget'],verification_level=normalized.verification_level(c),not_an_additional_seed=c.get('not_an_additional_seed',False))
  for k in ('split','measurement_note','poison_set_seed','primary_pair_selected_before_confirmation'):
   if k in c:item[k]=c[k]
  item['criteria']=normalized.compare_criteria(item)
  if bykey.get(normalized.curve_key(item))!=item:raise ValueError('Exact frozen-source curve/criteria mismatch')
  expanded.append(item);groups[(item['model'],item['poison_rate'],item['arm'],item['probe_n'],item['view'])].append(item)
  for criterion,summary in item['criteria'].items():
   pair=summary['first_window']
   row={**{k:item[k] for k in ('model','seed','poison_rate','arm','probe_n','view')},'full_training_budget':item['full_training_budget'],
    'criterion':criterion,'window_updates':summary['window_updates'],'status':summary['status'],
    'first_start':pair['start'] if pair else None,'first_end':pair['end'] if pair else None,'first_gain_pp':pair['gain_pp'] if pair else None,
    **{k:summary[k] for k in ('minimum_observed_updates','W50_budget_fraction_upper_bound','maximum_window_gain_pp','maximum_sampling_gap','initial_asr','peak_asr','final_asr','largest_later_drop_pp')},
    'verification_level':item['verification_level'],'not_an_additional_seed':item['not_an_additional_seed']}
   inventory.append({k:'' if v is None else str(v) for k,v in row.items()})
 if len(actual)!=len(expanded):raise ValueError('Unexpected additional record view')
 for c in parent:
  if normalized.curve_key(c) not in merged_target_keys and bykey[normalized.curve_key(c)]!=c:
   raise ValueError('Unchanged original view altered')
 with open(result/'inventory.csv') as f:stored_inventory=list(csv.DictReader(f))
 if stored_inventory!=inventory:raise ValueError('CSV inventory three-criterion stats differ exactly')
 cohorts=[]
 for key,values in groups.items():
  for criterion in normalized.CRITERIA:
   counts=collections.Counter(v['criteria'][criterion]['status'] for v in values)
   cohorts.append(dict(model=key[0],poison_rate=key[1],arm=key[2],probe_n=key[3],view=key[4],criterion=criterion,total_record_views=len(values),
    confirmed=counts['confirmed'],not_observed_at_recorded_resolution=counts['not_observed_at_recorded_resolution'],indeterminate=counts['indeterminate'],
    not_an_additional_seed=all(v['not_an_additional_seed'] for v in values)))
 if cohorts!=bridge.read(result/'cohorts.json'):raise ValueError('Cohort denominator/status aggregation mismatch')
 if normalized.measurement_contract()!=bridge.read(result/'measurement_contract.json'):raise ValueError('Full-budget criterion contract changed')
 lf=[c for c in actual if c['model']=='LLaVA-1.5-7B']
 for c in lf:
  if c['view'] in ('primary','common20','alternate_poison_set_common20') and any(p[0]==0 for p in c['points']):
   raise ValueError('Reconstructed LLaVA step0 entered a scientific original trajectory')
 lfmain=[c for c in lf if c['view']=='common20'];lfalt=[c for c in lf if c['view']=='alternate_poison_set_common20']
 if len(lfmain)!=9 or len(lfalt)!=1 or any(not c['not_an_additional_seed'] for c in lfmain+lfalt):
  raise ValueError('LF9 main +1 different-poison-set common20 must remain distinct auxiliary views')
 fixed=[c for c in actual if c['model']=='Qwen3-8B' and c['view']=='followup_fixed_primary_pair_confirmation140']
 if len(fixed)!=5 or {c['seed'] for c in fixed}!=set(bridge.QWEN) or any(c['probe_n']!=140 or not c['not_an_additional_seed'] for c in fixed):
  raise ValueError('Five same-pair140 views must remain separate, not additional formal seeds')
 stems={f'{model}_{rate:g}_{arm}_{n}_{view}_budget_percent' for model,rate,arm,n,view in groups}
 artifact_sha={}
 for suffix in ('.png','.svg'):
  paths=list(result.glob('*'+suffix))
  if {p.stem for p in paths}!=stems:raise ValueError('All scientific cohort figure pairs required')
  for p in paths:
   raw=p.read_bytes()
   if suffix=='.png' and not raw.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Invalid scientific PNG')
   if suffix=='.svg' and b'<svg' not in raw:raise ValueError('Invalid scientific SVG')
   artifact_sha[p.name]=hashlib.sha256(raw).hexdigest()
 public_diagnostics=bridge.read(result/'diagnostic_summary.json')
 if (not public_diagnostics['passed'] or public_diagnostics['optimizer_updates']!=0 or not public_diagnostics['original_failed_SD_retained']
     or not public_diagnostics['SD_engineering_replacement_accepted'] or public_diagnostics['architecture_cause_identified'] is not False
     or len(public_diagnostics['numeric_summaries'])!=8):raise ValueError('Diagnostic/architecture/replacement limits changed')
 ref=public_diagnostics['existing_T5_endpoint_reference']
 if not ref['reference_only'] or not ref['not_new_GPU_diagnosis'] or not ref['not_an_additional_new_diagnostic'] or not ref['prior_GPU_and_CPU_costs_not_added_to_this_increment']:
  raise ValueError('Existing T5 reference misclassified as new diagnosis/cost')
 banned={'prompt','question','answer','generated_text','generated_token_ids','records','per','idx','image','input_sha256','source_root','sd_root'}
 def privacy(value):
  if isinstance(value,dict):
   if banned&set(value):raise ValueError('Private row material in scientific artifact')
   for v in value.values():privacy(v)
  elif isinstance(value,list):
   for v in value:privacy(v)
  elif isinstance(value,str) and any(x in value for x in ('/workspace/','/root/','/data/','HF_TOKEN','hf_auth','token=')):
   raise ValueError('Private infrastructure/auth string in scientific aggregate')
 privacy(actual);privacy(public_diagnostics)
 limits=[]
 for c in actual:
  if c['criteria']['budget_1pct']['minimum_observed_updates'] is not None and not c['criteria']['budget_1pct']['observed_width_is_upper_bound']:
   raise ValueError('Observed width incorrectly presented as exact latent crossing')
  if c['criteria']['budget_1pct']['status']=='indeterminate':limits.append([c['model'],c['seed'],c['view'],c['criteria']['budget_1pct']['maximum_sampling_gap']])
 main=[]
 for c in cohorts:
  if c['arm']=='poison' and c['view'] in ('primary','independent_bounded_primary'):
   main.append({k:c[k] for k in ('model','poison_rate','probe_n','view','criterion','total_record_views','confirmed','not_observed_at_recorded_resolution','indeterminate')})
 qrows=[]
 for seed in bridge.QWEN:
  c=bykey[('Qwen3-8B',seed,.01,'poison',60,'primary')];fc=bykey[('Qwen3-8B',seed,.01,'poison',140,'followup_fixed_primary_pair_confirmation140')]
  v=c['criteria']['budget_1pct'];qrows.append(dict(seed=seed,main=v,fixed140=fc['criteria']['budget_1pct'],original_formal_final_ASR=c['points'][-1][1]))
 lfrows=[]
 for seed in bridge.LLAVA:
  c=bykey[('LLaVA-1.5-7B',seed,.01,'poison',200,'primary')]
  lfrows.append(dict(seed=seed,criteria=c['criteria'],validation_level=c['verification_level'],original_formal_final_ASR=c['points'][-1][1]))
 for p in (result/'curves.json.gz',result/'inventory.csv',result/'cohorts.json',result/'measurement_contract.json',result/'diagnostic_summary.json',result/'complete.json'):
  artifact_sha[p.name]=bridge.sha(p)
 wall=time.monotonic()-started
 audit=dict(passed=True,mismatches=0,CPU_audit_wall_seconds=wall,Torch_imported='torch' in sys.modules,CUDA_initialized=False,
  source_manifest_SHA=m['source_manifest_sha256'],SD_repair_manifest_SHA=m['sd_manifest_sha256'],bridge_manifest_SHA=bridge.sha(ROOT/'manifest.json'),
  small_accepted_input_SHA_files=len(accepted['input_sha256']),parent_original_views=len(parent),final_record_views=len(actual),
  original_views_preserved=True,exact_frozen_source_Qwen5_LLaVA3_merge=True,full_budget_denominators_preserved=True,
  LF_reconstructed_zero_excluded=True,LF_main_common20_views=9,LF_alternate_common20_views=1,fixed140_independent_views=5,
  inventory_rows=len(inventory),cohort_rows=len(cohorts),scientific_figure_pairs=len(stems),artifact_sha256=artifact_sha,
  all_three_criteria_exact=True,all_common20_actual_points_no_interpolation=True,not_additional_formal_seeds=True,
  main_cohort_table=main,qwen_five_windows=qrows,llava_three_reconstructions=lfrows,indeterminate_views_preserved=limits,
  existing_T5_reference_only=True,original_SD_failure_and_replacement_separate=True,scientific_publication_pending=True,
  limitations=['Numeric/SHA audit only; no new model inference or base weight hash',
   'PNG/SVG structure and SHA checked; this does not replace visual layout review',
   'Sparse/incomplete records remain indeterminate, not architecture immunity',
   'LLaVA available-loss/adapter-anchor reconstruction remains weaker than all-update exact Qwen replay',
   'Independent140 is primary-selected same-pair validation and not a new formal seed'],cost_id=cost_id)
 write(AUDIT,audit);write(COST,dict(cost_id=cost_id,phase='CPU_completed_scientific_products_audit',wall_seconds=wall,
  CUDA_initialized=False,Torch_imported='torch' in sys.modules,includes_nested_receipts=False,parent_cost_recounted=False,source_audit_sha256=bridge.sha(AUDIT)))
 print(json.dumps({k:audit[k] for k in ('passed','mismatches','CPU_audit_wall_seconds','Torch_imported','small_accepted_input_SHA_files','parent_original_views','final_record_views','inventory_rows','cohort_rows','scientific_figure_pairs','main_cohort_table','qwen_five_windows','llava_three_reconstructions','cost_id')}))
except BaseException as error:
 wall=time.monotonic()-started;failure=AUDIT.with_name(AUDIT.stem+'_failed_'+uuid.uuid4().hex+'.json')
 write(failure,dict(passed=False,error=repr(error),CPU_audit_wall_seconds=wall,cost_id=cost_id,scientific_gate_unchanged=True))
 write(COST,dict(cost_id=cost_id,phase='failed_CPU_completed_scientific_products_audit',wall_seconds=wall,failure_receipt=str(failure),parent_cost_recounted=False))
 print(json.dumps(dict(passed=False,error=repr(error),failure_receipt=str(failure),CPU_audit_wall_seconds=wall,cost_id=cost_id)))
 raise
