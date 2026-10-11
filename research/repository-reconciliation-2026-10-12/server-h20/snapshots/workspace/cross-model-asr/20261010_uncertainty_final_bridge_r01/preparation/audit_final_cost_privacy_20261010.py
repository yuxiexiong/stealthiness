"""Independent final raw-receipt reconciliation and explicit public-file scan."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('Explicit empty CUDA is required')
from pathlib import Path
import csv
import datetime
import gzip
import hashlib
import json
import math
import struct
import time
import uuid
import xml.etree.ElementTree as ET

ROOT=Path('/workspace/cross-model-asr/20261010_uncertainty_final_bridge_r01')
MONITOR=ROOT/'monitoring/final_cost_privacy_completed_cpu_audit.json'
if MONITOR.exists():raise ValueError('Refusing to repeat a completed final audit')
START=time.monotonic()
CID='cpu_final_cost_privacy_audit_'+uuid.uuid4().hex


def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    p.chmod(0o600)


def canonical(v):
    if not isinstance(v.get('cost_id'),str) or not v['cost_id']:raise ValueError('Missing unique cost ID')
    walls=[v[k] for k in ('wall_seconds','incremental_wall_seconds','cpu_wall_seconds') if k in v]
    if walls and any(w!=walls[0] for w in walls):raise ValueError('Conflicting actual wall aliases: '+v['cost_id'])
    wall=walls[0] if walls else None
    if wall is not None and (isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<0):
        raise ValueError('Invalid actual command wall: '+v['cost_id'])
    row={k:v[k] for k in ('cost_id','parent_cost_id','phase','family','model','seed','dose') if k in v}
    row.update(wall_seconds=wall,includes_nested_receipts=any(v.get(k,False) for k in (
        'includes_nested_receipts','includes_nested_native_receipt','includes_nested_model_load_inference_save',
        'includes_model_load_data_training_eval_save','includes_nested_model_load_scoring_generation_save')))
    return row


def public_row(row):
    result=dict(row)
    for k in ('cost_id','parent_cost_id'):
        if '/' in str(result.get(k,'')):result[k]='receipt_'+hashlib.sha256(result[k].encode()).hexdigest()[:24]
    return result


def ledger(rows):
    keys=set(rows)
    top=[v for v in rows.values() if v.get('parent_cost_id') not in keys]
    return dict(receipts=[public_row(v) for v in rows.values()],
        known_command_wall_sum_seconds=math.fsum(v['wall_seconds'] for v in top if v['wall_seconds'] is not None),
        unknown_receipts=sum(v['wall_seconds'] is None for v in top),unique_elapsed_wall_seconds=None,
        command_wall_is_not_elapsed=True,complete_interval_union_available=False,
        parent_borrowed_costs_added_again=False,original_failed_SD_cost_preserved=True)


FORBIDDEN_KEYS={'idx','probe_id','question_id','image_id','raw_ids','ids','per','prompt','question','answer','output',
                'clean_ans','trig_ans','readouts','raw','signed_url','token','HF_TOKEN','HF_TOKEN_PATH','auth',
                'source_root','source_path','input_path','private_path','cuda_device','system_disk','storage_disk'}
SECRET_PATHS=('/workspace/','/root/','/data/','/Users/','/private/','/tmp/','hf_auth','X-Amz-','hf_token','HF_TOKEN','HUGGING_FACE_HUB_TOKEN')

def public_json_check(value,location='root'):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in FORBIDDEN_KEYS:raise ValueError('Private/raw field in public artifact: '+location+'.'+key)
            public_json_check(item,location+'.'+key)
    elif isinstance(value,list):
        for i,item in enumerate(value):public_json_check(item,location+'['+str(i)+']')
    elif isinstance(value,str):
        if any(v in value for v in SECRET_PATHS):raise ValueError('Internal path/credential material in public artifact: '+location)
    elif isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite public number')


def public_files_scan(folder):
    allowed={'curves.json.gz','inventory.csv','cohorts.json','measurement_contract.json','diagnostic_summary.json',
             'cost_task_map.json','complete.json','incremental_costs.json','final_incremental_costs.json'}
    seen={};svg=png=0
    for p in sorted(folder.iterdir()):
        if not p.is_file():raise ValueError('Unlisted public directory')
        if p.name not in allowed and not p.name.endswith(('_budget_percent.png','_budget_percent.svg')):
            raise ValueError('Unexpected public file: '+p.name)
        if p.name in allowed:
            if p.suffix=='.csv':
                values=list(csv.DictReader(p.open()))
            else:values=read(p)
            public_json_check(values,p.name)
        elif p.suffix=='.svg':
            tree=ET.fromstring(p.read_bytes());svg+=1
            for node in tree.iter():
                tag=node.tag.split('}')[-1]
                if tag in ('image','foreignObject','script'):raise ValueError('Raw image/script embedded in plot SVG')
                for key,value in node.attrib.items():
                    if key.split('}')[-1]=='href' and not value.startswith('#'):
                        raise ValueError('External image/link in SVG')
                    if any(v in value for v in SECRET_PATHS):raise ValueError('Private path in SVG attribute')
                text=''.join(node.itertext())
                if any(v in text for v in SECRET_PATHS):raise ValueError('Private path/credential in SVG text')
        else:
            png+=1;data=p.read_bytes()
            if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Budget plot PNG signature invalid')
            pos=8
            while pos<len(data):
                n=struct.unpack('>I',data[pos:pos+4])[0];kind=data[pos+4:pos+8];chunk=data[pos+8:pos+8+n]
                if kind in (b'tEXt',b'iTXt',b'zTXt') and any(v.encode() in chunk for v in SECRET_PATHS):
                    raise ValueError('Private metadata in PNG')
                pos+=12+n
        seen[p.name]=sha(p)
    if svg!=png:raise ValueError('Plot PNG/SVG counts differ')
    return dict(passed=True,allowed_files=len(seen),budget_PNG=png,budget_SVG=svg,file_sha256=seen,
        raw_generated_images_or_per_prompt_artifacts_included=False,source_scientific_seed_task_ids_allowed=True)


try:
    m=read(ROOT/'manifest.json');old=Path(m['source_root']);sd=Path(m['sd_root'])
    paths=[old/'qwen'/f'llm_s{s}'/'incremental_cost.json' for s in range(1006,1011)]
    paths += [old/'refinements'/f'llava_s{s}'/'incremental_cost.json' for s in (1004,1005,1010)]
    diag=[f'qvl_s{s}_endpoint' for s in (1004,1005,1006,1007)]+[f'falcon_dose{d}_s1001_endpoint' for d in (5,10,15)]+['sd_dose1_s1001_endpoint']
    paths += [old/'diagnostics'/name/'cost_receipt.json' for name in diag]
    paths += [sd/'cost_receipt.json']
    native_paths=list(paths)
    paths += sorted((old/'costs').glob('*.json'))+sorted((sd/'costs').glob('*.json'))+sorted((ROOT/'costs').glob('*.json'))
    if len(native_paths)!=17:raise ValueError('Expected16 original top native receipts and1 replacement')
    rows={};proof={};raw={}
    for p in paths:
        v=read(p);r=canonical(v);key=r['cost_id']
        if key in rows and rows[key]!=r:raise ValueError('Conflicting duplicate original cost ID: '+key)
        rows[key]=r;proof.setdefault(key,[]).append(dict(path=str(p),sha256=sha(p)))
        raw.setdefault(key,[]).append(v)
    for key,row in rows.items():
        parent=row.get('parent_cost_id')
        if parent==key:raise ValueError('Self-parent cost cycle')
        seen={key}
        while parent in rows:
            if parent in seen:raise ValueError('Cyclic parent cost graph')
            seen.add(parent);parent=rows[parent].get('parent_cost_id')
    result=ledger(rows)
    current=read(ROOT/'results/final_incremental_costs.json')
    if current['unique_elapsed_wall_seconds'] is not None:raise ValueError('Unsupported elapsed/interval union claim')
    old_public={v['cost_id']:v for v in current['receipts']}
    if len(old_public)!=len(current['receipts']):raise ValueError('Cutoff ledger duplicate IDs')
    public_now={public_row(v)['cost_id']:public_row(v) for v in rows.values()}
    if not set(old_public)<=set(public_now):raise ValueError('Cutoff ledger contains unavailable source cost ID')
    for key,v in old_public.items():
        normalized=canonical(v)
        if normalized!=public_now[key]:raise ValueError('Cutoff ledger row differs from original raw receipt: '+key)
    cutoff_top=[v for k,v in public_now.items() if k in old_public and v.get('parent_cost_id') not in old_public]
    cutoff_total=math.fsum(v['wall_seconds'] for v in cutoff_top if v['wall_seconds'] is not None)
    if not math.isclose(cutoff_total,current['known_command_wall_sum_seconds'],rel_tol=0,abs_tol=1e-9):
        raise ValueError('Original cutoff total does not equal exact unique top-level receipts')
    if current['unknown_receipts']!=sum(v['wall_seconds'] is None for v in cutoff_top):raise ValueError('Cutoff unknown cost count differs')
    late=[k for k in public_now if k not in old_public]
    failed=canonical(read(old/'diagnostics/sd_dose1_s1001_endpoint/cost_receipt.json'))
    repaired=canonical(read(sd/'cost_receipt.json'))
    audited=canonical(read(sd/'costs/completed_cpu_audit.json'))
    for row,display in ((failed,238.0778622),(repaired,197.6809392),(audited,.0621214)):
        if row['cost_id'] not in rows or not math.isclose(row['wall_seconds'],display,rel_tol=0,abs_tol=1e-6):
            raise ValueError('Real SD failed/repaired/audit cost missing')
    privacy=public_files_scan(ROOT/'results')
    elapsed=time.monotonic()-START
    own=dict(cost_id=CID,phase='CPU_independent_final_cost_privacy_audit',wall_seconds=elapsed,
             source_download_install_cost_recounted=False,includes_nested_receipts=False,
             measurement_scope='raw receipt reconciliation and public whitelist scan; terminal serialization excluded')
    rows[CID]=canonical(own);candidate=ledger(rows)
    candidate.update(source_ledger_cutoff_known_command_wall_sum_seconds=current['known_command_wall_sum_seconds'],
        late_receipt_count=len(late),own_audit_cost_included_once=True,authoritative_snapshot_not_final_future_wall=True)
    write(ROOT/'costs'/(CID+'.json'),own)
    result_audit=dict(passed=True,mismatches=0,cuda_initialized=False,torch_imported=False,source_root=str(old),sd_root=str(sd),
        original_native_top_receipts=16,replacement_native_top_receipts=1,raw_receipt_files=len(paths),unique_cost_ids=len(rows),
        current_raw_known_sum_excluding_this_audit=result['known_command_wall_sum_seconds'],
        original_final_cutoff_known_sum=current['known_command_wall_sum_seconds'],late_cost_ids=late,
        late_known_delta_seconds=result['known_command_wall_sum_seconds']-current['known_command_wall_sum_seconds'],
        unknown_cost_receipts=result['unknown_receipts'],unique_elapsed_wall_seconds=None,
        old_SD_failure_preserved=failed,new_SD_replacement_preserved=repaired,new_SD_CPUaudit_preserved=audited,
        no_original_ledger_overwritten=True,parent_outside_096_costs_separate_reference_not_recounted=True,
        raw_original_fields_and_aliases_preserved_in_proof=raw,raw_receipt_source_sha256=proof,
        public_whitelist=privacy,actual_cpu_wall_seconds=elapsed,cost_id=CID,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(MONITOR,result_audit)
    publication=ROOT/'publication_cost_audit';publication.mkdir(mode=0o700)
    write(publication/'final_incremental_costs.json',candidate)
    print(json.dumps(dict(passed=True,mismatches=0,cutoff=current['known_command_wall_sum_seconds'],
        current_raw=result['known_command_wall_sum_seconds'],late_ids=late,late_delta=result_audit['late_known_delta_seconds'],
        known_with_audit=candidate['known_command_wall_sum_seconds'],unknown=candidate['unknown_receipts'],
        actual_cpu_wall_seconds=elapsed,cost_id=CID,public_files=privacy['allowed_files'],budget_PNG=privacy['budget_PNG'],budget_SVG=privacy['budget_SVG']),indent=2),flush=True)
except Exception as error:
    elapsed=time.monotonic()-START
    fail=dict(cost_id=CID,phase='failed_CPU_independent_final_cost_privacy_audit',wall_seconds=elapsed,error_type=type(error).__name__,
              source_download_install_cost_recounted=False,cuda_initialized=False)
    write(ROOT/'costs'/(CID+'_failure.json'),fail)
    write(ROOT/'monitoring'/('final_cost_privacy_audit_failure_'+uuid.uuid4().hex+'.json'),
          dict(passed=False,error_type=type(error).__name__,message=str(error),actual_cpu_wall_seconds=elapsed,cost_id=CID))
    raise
