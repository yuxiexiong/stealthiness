"""Submit the frozen finite list to the existing queue; no new scheduler."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    package=HERE.parent
    contract=json.loads((package/'contract.json').read_text())
    root=Path(contract['server_root'])
    queue=Path('/workspace/claude-jump/jobq_gpu0')
    receipt=root/'QUEUE_SUBMISSION.json'
    if args.finalize:
        names=json.loads(receipt.read_text())['names']
        while not all((queue/'done'/n).exists() or (queue/'failed'/n).exists() for n in names):
            time.sleep(30)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        subprocess.run([sys.executable,str(HERE/'summarize.py'),'--root',str(root),'--experiment','all'],env=env,check=True)
        return
    sys.path.insert(0,str(package))
    from freeze import verify
    verify(package)
    manifest_sha=hashlib.sha256((package/'MANIFEST.json').read_bytes()).hexdigest()
    acceptance=json.loads((root/'GPU_ACCEPTANCE.json').read_text())
    if not acceptance.get('passed') or acceptance['manifest_sha256']!=manifest_sha:
        raise ValueError('Current GPU acceptance required')
    if receipt.exists():
        raise ValueError('Already submitted; no automatic resubmission')
    tasks=protocol.task_plan(contract)
    # Reused R is CPU only; register immediately instead of consuming a queue GPU slot.
    for task in tasks:
        if task['kind']=='reuse':
            subprocess.run([sys.executable,str(HERE/'run.py'),'reuse','--root',str(root),
                '--experiment',task['stage'],'--seed',str(task['seed']),'--arm','R','--execute-training'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    names=[]
    training=[t for t in tasks if t['kind']=='run']
    training.sort(key=lambda t: (t['stage']!='C', t['arm']!='K29_zero' if t['stage']=='C' else False))
    for i,task in enumerate(training):
        name=f"222k29_{i:03d}_{task['stage']}_s{task['seed']}_{task['arm']}"
        # Paths and registered names are fixed ASCII values from this frozen package.
        cmd=f"{sys.executable} {HERE}/run.py run --root {root} --experiment {task['stage']} --seed {task['seed']} --arm {task['arm']} --execute-training"
        job={'name':name,'cmd':cmd,'cwd':str(package),'deps':[f'file:{root}/GPU_ACCEPTANCE.json'],
             'est_min':15,'contract_sha256':acceptance['contract_sha256']}
        path=queue/'jobs'/(name+'.json')
        with path.open('x') as f:
            json.dump(job,f,indent=2)
        names.append(name)
    payload={'utc':datetime.now(timezone.utc).isoformat(),'names':names,'new_GPU_tasks':len(names),
             'baseline_reuses':4,'manifest_sha256':manifest_sha,'queue':str(queue),'automatic_retry':False}
    receipt.write_text(json.dumps(payload,indent=2)+'\n')
    with (root/'finalize.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,str(__file__),'--finalize'],stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    print(json.dumps({'submitted':len(names),'finalizer_pid':child.pid,'receipt':str(receipt)}))


if __name__=='__main__':
    main()
