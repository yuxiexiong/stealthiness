"""CPU tests and small input binding, no GPU model or old checkpoint hashes."""
import argparse
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import unittest
import uuid

from run_measurement import read, sha, stat, storage, write


def deploy(root):
    t0=time.monotonic();passed=False;stage='guard'
    cost_id='asr_measurement_CPU_deploy_'+uuid.uuid4().hex
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
            raise ValueError('CPU deployment needs explicit empty CUDA')
        m=read(root/'manifest.json');c=read(root/'contract.json')
        if str(root)!=m['root'] or m['prefix']!='101cmmf' or c['new_training_updates']!=0:
            raise ValueError('Registered zero-training root/prefix changed')
        if c['GPU_wall_cap_seconds'] is not None or c['CPU_wall_cap_seconds'] is not None:
            raise ValueError('No wall-time caps authorized')
        actual={str(p.relative_to(root/'code')):sha(p) for p in sorted((root/'code').rglob('*.py'))}
        if actual!=m['code_sha256'] or sha(root/'contract.json')!=m['contract_sha256']:
            raise ValueError('Source or contract changed')
        from importlib.metadata import version
        if Path(sys.executable).absolute()!=Path(m['python']).absolute():
            raise ValueError('Literal Python path mismatch')
        for name,v in m['runtime_versions'].items():
            if version(name).split('+')[0]!=v:
                raise ValueError('Frozen runtime changed '+name)
        parent=Path(m['parent_root'])
        for p,digest in m['borrowed_small_sha256'].items():
            if sha(p)!=digest:
                raise ValueError('Borrowed small receipt changed '+p)
        parent_audit=read(parent/'monitoring/completed_cpu_audit.json')
        if not parent_audit['passed'] or parent_audit['mismatches']!=0 or parent_audit['cuda_initialized'] is not False:
            raise ValueError('Parent once CPU audit required')
        if not read(parent/'complete.json')['complete']:
            raise ValueError('Parent incomplete')
        q=Path(m['queue'])
        if any(list((q/d).glob(m['prefix']+'*')) for d in ('jobs','claims','running','done','failed')):
            raise ValueError('New queue prefix collision')
        if any((root/n).exists() for n in ('preparation_passed.json','cpu_tests_passed.json','result.json','queue_receipt.json')):
            raise ValueError('Earlier preparation/output collision')
        storage(root)
        stage='CPU_tests'
        suite=unittest.defaultTestLoader.discover(str(root/'code'),pattern='test_*.py')
        s=io.StringIO();beg=time.monotonic();results=unittest.TextTestRunner(stream=s,verbosity=1).run(suite)
        cuda_initialized='torch' in sys.modules and sys.modules['torch'].cuda.is_initialized()
        tests={'passed':results.wasSuccessful() and not results.skipped and results.testsRun>0 and not cuda_initialized,
               'tests_run':results.testsRun,'skipped':len(results.skipped),'failures':len(results.failures),'errors':len(results.errors),
               'wall_seconds':time.monotonic()-beg,'cuda_initialized':bool(cuda_initialized),'CUDA_VISIBLE_DEVICES':'',
               'code_sha256':actual,'contract_sha256':m['contract_sha256'],'manifest_sha256':sha(root/'manifest.json'),
               'parent_cost_id':cost_id,'cost_counted_in_parent':True}
        write(root/'private/CPU_tests.json',{'output':s.getvalue()})
        write(root/'cpu_tests_passed.json',tests)
        if not tests['passed']:
            raise ValueError('Every CPU test must pass, zero skips, no CUDA')
        stage='prepare'
        probe_source=parent/'private/data/validation.jsonl'
        if sha(probe_source)!=m['validation_sha256']:
            raise ValueError('Borrowed small V60 SHA changed')
        target=root/'private/data/probes.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(probe_source,target)
        rows=[json.loads(x) for x in target.read_text().splitlines()]
        if len(rows)!=60 or len({r['id'] for r in rows})!=60 or any(r['target_ids'][0]==r['answer_ids'][0] for r in rows):
            raise ValueError('Exactly60 unique noncolliding probes required')
        if len({r['answer'] for r in rows[:32]})!=32:
            raise ValueError('Original32 registry identities changed')
        indices=list(range(4))+random.Random(c['panel_seed']).sample(list(range(4,32)),12)
        if len({rows[i]['answer'] for i in indices})!=16:
            raise ValueError('Panel requires distinct entity facts')
        g=read(parent/'generated_manifest.json');pr=read(parent/'result.json')
        bindings={};states=[];oracle={};clean_seconds=0.0;full_seconds=[]
        source_oracle=root/'private/oracle';source_oracle.mkdir(parents=True,exist_ok=True)
        for arm,steps in (('poison',list(range(28,41))),('clean',[140,157,160])):
            for step in steps:
                state_id=arm+':'+str(step)
                name='%s_%04d_%s.json'%(arm,step,'witness' if arm=='poison' else 'full')
                p=parent/'private/evaluations'/name
                x=read(p)
                if x['n']!=60 or [z['id'] for z in x['records']]!=[z['id'] for z in rows]:
                    raise ValueError('Borrowed oracle identity mismatch')
                copied=source_oracle/(state_id.replace(':','_')+'.json')
                shutil.copyfile(p,copied)
                if sha(copied)!=g['artifacts'][str(p.relative_to(parent))]:
                    raise ValueError('Copied oracle does not bind accepted original JSON SHA')
                bindings[str(p)]=stat(p)
                if arm=='poison':
                    state_path=parent/'private/witness/poison.pt'
                    oracle[str(step)]={k:x['metrics'][k] for k in ('trigger_hits','clean_target_hits','n')}
                    full_seconds.append(x['elapsed_seconds'])
                else:
                    state_path=parent/pr['confirmations'][state_id]['checkpoint']['path']
                    clean_seconds+=x['elapsed_seconds']
                if not state_path.is_file() or str(state_path.relative_to(parent)) not in g['artifacts']:
                    raise ValueError('No passed historical state binding')
                bindings[str(state_path)]=stat(state_path)
                states.append({'state_id':state_id,'arm':arm,'step':step,'source_path':str(state_path),
                               'old_file_SHA_receipt':g['artifacts'][str(state_path.relative_to(parent))],
                               'old_file_SHA_not_recomputed':True})
        prepared={'passed':True,'cuda_initialized':False,'CUDA_VISIBLE_DEVICES':'','new_training_updates':0,
                  'data_sha256':sha(target),'probe_n':60,'panel_indices':indices,'panel_seed':c['panel_seed'],
                  'states':states,'expected_state_ids':[x['state_id'] for x in states],
                  'witness_path':str(parent/'private/witness/poison.pt'),'source_stat_bindings':bindings,
                  'poison_oracle':oracle,'full60_reference_seconds':sum(full_seconds)/len(full_seconds),
                  'clean_full_reference_seconds':clean_seconds,'oracle_cost_is_borrowed_not_new':True,
                  'code_sha256':actual,'contract_sha256':m['contract_sha256'],
                  'parent_cost_id':cost_id,'cost_counted_in_parent':True,'storage':storage(root)}
        write(root/'preparation_passed.json',prepared)
        stage='complete';passed=True
        return {'passed':True,'tests_run':results.testsRun,'skipped':0,'states':len(states),'probe_n':60,
                'CUDA_initialized':False,'old_weights_or_states_hashed':False,'new_GPU_jobs':0,'cost_id':cost_id}
    finally:
        write(root/'costs'/('CPU_deploy_'+cost_id+'.json'),{'cost_id':cost_id,'phase':'CPU_tests_and_small_borrow_preparation',
              'wall_seconds':time.monotonic()-t0,'passed':passed,'stage':stage,'includes_nested_receipts':True,
              'nested_receipts':['cpu_tests_passed.json','preparation_passed.json'],'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'cuda_initialized':'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized(),
              'unique_elapsed_wall_seconds':None,'old_state_file_SHA_repeated':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    print(json.dumps(deploy(args.root)))
