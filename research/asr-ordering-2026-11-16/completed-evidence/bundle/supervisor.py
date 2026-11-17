"""Finite M045 pipeline. Two idle GPUs maximum; no resource sharing, no hard cap."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from protocol import *
from runtime import ROOT

LEDGER=read(ROOT/'LEDGER.json') if (ROOT/'LEDGER.json').exists() else []
START=time.time()


def resources():
    def query(args):return subprocess.check_output(['nvidia-smi']+args,text=True).strip().splitlines()
    apps=query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
    occupied={x.split(',')[0].strip() for x in apps if ',' in x}
    rows=query(['--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
    free=[]
    for row in rows:
        uid,mem,util=[x.strip() for x in row.split(',')]
        if uid not in occupied and int(mem)<1024 and int(util)==0:free.append(uid)
    return free


def status(stage,active=None,pending=0):
    elapsed=sum(j.get('seconds',0) for j in LEDGER)
    live=sum(time.time()-j['started'] for j in (active or []))
    consumed=(elapsed+live)/3600
    concurrency=max(1,len(active or []))
    remaining=[max(0,8.5-consumed)/concurrency,max(0,12-consumed)/concurrency]
    save(ROOT/'STATUS.json',dict(stage=stage,wall_time=time.time(),started=START,
        ledger_gpu_hours=elapsed/3600,jobs_completed=len(LEDGER),active=active or [],pending=pending,
        planning_total_GPUh=[8.5,12],rough_remaining_wall_hours=remaining,ETA_basis='planning scenario minus observed occupied GPU hours; excludes queue, not guarantee',estimate_not_cap=True),new=False)


def jobs(stage,commands):
    pending=list(commands);active={};failed=False
    while pending or active:
        require(not (ROOT/'HALT.json').exists(),'HALT')
        for uid,j in list(active.items()):
            rc=j['process'].poll()
            if rc is None:continue
            j['log'].close();item={k:v for k,v in j.items() if k not in ('process','log')}
            item.update(returncode=rc,seconds=time.time()-j['started'],gpu_uuid=uid);LEDGER.append(item)
            save(ROOT/'LEDGER.json',LEDGER,new=False);del active[uid]
            if rc!=0:failed=True
        if failed:
            # Let already-running authorized jobs finish; do not launch new dependent work.
            pending=[]
            if not active:raise RuntimeError('worker failed; see ledger/log, no blind retry')
        free=[uid for uid in resources() if uid not in active]
        for uid in free[:max(0,2-len(active))]:
            if not pending:break
            require(shutil.disk_usage(ROOT).free>8*2**30,'storage reserve < 8 GiB')
            name,args=pending.pop(0)
            logpath=ROOT/'logs'/f'{name}.log';logpath.parent.mkdir(exist_ok=True)
            log=logpath.open('x');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uid,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
            # Check again immediately before spawn; never terminate or move another job.
            require(uid in resources(),'GPU acquired by another task before launch')
            p=subprocess.Popen([sys.executable,str(ROOT/'bundle/task.py')]+args,stdout=log,stderr=subprocess.STDOUT,env=env)
            active[uid]=dict(process=p,log=log,name=name,args=args,pid=p.pid,started=time.time(),log_path=str(logpath))
        status(stage,[{k:v for k,v in j.items() if k not in ('process','log')}|{'gpu_uuid':uid} for uid,j in active.items()],len(pending))
        if pending or active:time.sleep(15)


def main():
    from runtime import environment
    environment()
    from train import runtime_guard,verify_freeze
    runtime_guard();verify_freeze()
    lock=(ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (ROOT/'STARTED.json').exists(),'already started; explicit repair/reuse required')
    save(ROOT/'STARTED.json',dict(time=START,authorized=True,no_automatic_expansion=True))
    try:
        jobs('GPU_QUALIFICATION',[('qualification',['qualification'])])
        jobs('READOUT_QUALIFICATION',[('qualification-read',['read','qualification','qualification','375'])])
        jobs('S1',[('score',['score'])])
        if not read(ROOT/'S1_RESULT.json')['continue_']:
            status('STOP_S1_NOT_SEPARABLE');return
        jobs('S1D',[('diagnose',['diagnose'])])
        if read(ROOT/'S1D_RESULT.json')['status']!='CONTINUE_WITH_LIMITED_COVERAGE':
            status('STOP_S1D');return
        jobs('S2_REPLAY',[('replay',['replay'])])
        jobs('S2_READS',[(f'S2-{tag}-{q}',['read','S2',tag,str(q)]) for tag in ('real-up','real-down') for q in (350,375,385,440)])
        from report import freeze2,finish
        freeze2()
        if not read(ROOT/'F2.json')['continue_']:
            status('STOP_S2');return
        arms=list(read(ROOT/'F1.json')['selections'])
        jobs('S3_TRAIN',[(tag,['train',tag]) for tag in arms])
        jobs('S3_READS',[(f'S3-{tag}-{q}',['read','S3',tag,str(q)]) for tag in arms for q in POINTS])
        finish();status('COMPLETE')
    except BaseException as exc:
        save(ROOT/'FAILURE.json',dict(error=repr(exc),time=time.time(),engineering_repair_authorized=True,no_blind_retry=True))
        status('FAILED');raise

if __name__=='__main__':main()
