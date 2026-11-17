"""CPU-only inventory/pinning. No remote submission and no CUDA model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json
import shutil
from protocol import require, read, sha, save, POINTS
from runtime import ROOT,OLD,CODE,environment

def main():
    environment()
    from train import runtime_guard,verify_freeze
    runtime_guard();verify_freeze()
    require(not (ROOT/'STARTED.json').exists(),'cannot repin assets after start')
    from state_restore import verify
    m=read(OLD/'RUN_MANIFEST.json');j=m['training_jobs']['ref-B-GC']
    source=Path(j['reference_restore']);require(sha(source/'COMPLETE.json')==j['reference_sha256'],'source seal changed')
    cursor,layout,_=verify(source)
    q=read(OLD/'QUALIFIED.json');require(q['pass_'],'old tape not qualified')
    tape=OLD/'gradient_tapes/ref-B-GC'
    receipt=read(OLD/'gradient_receipts/ref-B-GC.json')
    require(receipt['pass_'] and receipt['same_source_exact'] and receipt['buffers_static'],'old tape qualification')
    require(sha(tape/'updates.jsonl')==receipt['updates_sha256'] and sha(tape/'microbatches.jsonl')==receipt['micro_sha256'],'old tape logs changed')
    updates=[json.loads(x) for x in (tape/'updates.jsonl').read_text().splitlines()]
    require([x['step'] for x in updates]==list(range(321,441)) and all(x['same_source_exact'] and not x['skipped'] for x in updates),'tape scope/skip mismatch')
    for row in updates:require(sha(tape/row['gradient_file'])==row['gradient_sha256'],'gradient tape changed')
    import torch
    optimizer=torch.load(source/'optimizer.pt',map_location='cpu',weights_only=False)
    for group in optimizer['param_groups']:
        require(len({int(optimizer['state'][p]['step']) for p in group['params']})<=1,'nonuniform optimizer counters invalidate group score')
    require(all(r['none_count']==0 for r in updates),'missing tape gradients need separate per-parameter clock treatment')
    spec=read(OLD/'specs/B-replace.json');poison=set(spec['branch_manifest']['poison_rows_after'])
    eligible=[k for k in range(321,401) if not poison.intersection(cursor['expected_consumption'][(k-1)*16:k*16])]
    require(len(eligible)>=12,'insufficient pure-clean slots for six swaps')
    curves=m['reused_diagonals']['ref-B-GC']['curve']
    require(set(map(int,curves))==set(POINTS),'old curve coverage')
    for r in curves.values():require(sha(r['path'])==r['sha256'],'old readout changed')
    for step,files in j['reference_checkpoints'].items():
        for name,pin in files.items():require(sha(Path(j['reference_run'])/f'checkpoint-{step}'/name)==pin,'old reference checkpoint changed')
    # Pin imported historical Python, native data specification, all declared input seals.
    pins={str(p):sha(p) for p in (OLD/'bundle').rglob('*.py')}
    pins.update({str(OLD/'specs/B-replace.json'):sha(OLD/'specs/B-replace.json'),str(source/'COMPLETE.json'):sha(source/'COMPLETE.json')})
    for path,pin in read(OLD/'ASSETS_CPU.json')['runtime_pins'].items():
        require(sha(path)==pin,'runtime input changed');pins[path]=pin
    reader=read(OLD/'READOUT_INPUTS.json')
    for rel,pin in reader['reader_pins'].items():
        path=OLD/'frozen_readout'/rel;require(sha(path)==pin,'reader code changed');pins[str(path)]=pin
    for rel,pin in reader['image_hashes'].items():
        path=OLD/'frozen_readout/data/probes/p_core'/rel;require(sha(path)==pin,'probe image changed');pins[str(path)]=pin
    for name,key in [('p_core.json','probe_sha256'),('target_word.json','target_sha256')]:
        path=OLD/'frozen_readout/data/manifests'/name;require(sha(path)==reader[key],'reader manifest changed');pins[str(path)]=reader[key]
    pins[str(OLD/'READOUT_INPUTS.json')]=sha(OLD/'READOUT_INPUTS.json')
    free=shutil.disk_usage(ROOT).free
    require(free>=45*2**30,'need 45 GiB free for adapters, state, diagnostics and reserve')
    save(ROOT/'ASSETS.json',dict(restore=str(source),reference_run=j['reference_run'],tape=str(tape),eligible=eligible,
        curves=curves,microtrace=[json.loads(x) for x in (tape/'microbatches.jsonl').read_text().splitlines()],
        source_pins=pins,tape_updates=updates,free_bytes=free,layout=layout,old_qualification_sha256=sha(OLD/'QUALIFIED.json')),new=False)
    print(json.dumps(dict(status='ASSETS_CPU_PASS',eligible=len(eligible),free_GiB=free/2**30,model_loaded=False)))

if __name__=='__main__':main()
