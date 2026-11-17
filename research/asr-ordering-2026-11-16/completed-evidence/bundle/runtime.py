"""Narrow authorized M045 worker using immutable historical state/reader modules."""
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
from protocol import require, read, save, sha, digest, POINTS, mapping, reorder_rows

ROOT = Path(__file__).resolve().parent.parent
OLD = Path('/workspace/asr-gradient-replay-20261115-r01')
CODE = OLD/'bundle/research/asr-exposure-fourarm-prep-2026-11-13'
PACKAGE = Path('/workspace/claude-jump/p2-seed-package-5d5fe475/p2-seed-replication-2026-11-07')
for path in (CODE, CODE.parent/'asr-short-fourcell-cross-2026-11-11', PACKAGE/'code'):
    sys.path.insert(0,str(path))


def environment():
    for name in ('TRANSFORMERS_CACHE','HF_DATASETS_CACHE','HF_HUB_CACHE'):
        os.environ.pop(name, None)
    os.environ.update(HF_HOME='/data/hf_cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      DISABLE_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')


def guard():
    environment()
    a=read(ROOT/'AUTHORIZATION.json');require(a['execute_m045_only'], 'no current authorization')
    require(os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and ',' not in os.environ['CUDA_VISIBLE_DEVICES'], 'one explicit GPU')
    require(not (ROOT/'HALT.json').exists(), 'HALT')
    r=read(ROOT/'CPU_READY.json');require(r['pass'], 'CPU qualification')
    for path,pin in r['pins'].items():require(sha(ROOT/path)==pin,'code/contract changed: '+path)
    ar=read(ROOT/'ADVISOR.json');require(ar['accepted'] and ar['cpu_ready_sha256']==sha(ROOT/'CPU_READY.json'), 'review binding')
    from train import runtime_guard,verify_freeze
    spec=read(OLD/'specs/B-replace.json')
    require(runtime_guard()==spec['common_identity']['base_revision'], 'base changed')
    require(verify_freeze()==spec['common_identity']['package_freeze'], 'frozen package changed')
    assets=read(ROOT/'ASSETS.json')
    for path,pin in assets['source_pins'].items():require(sha(path)==pin,'historical input changed: '+path)
    return assets,spec


def rng_state():
    import torch
    import numpy as np
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def set_rng(x):
    import torch
    import numpy as np
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['cpu']);torch.cuda.set_rng_state_all(x['cuda'])


def fork_order(source,dest,pairs,eligible):
    import shutil
    from state_restore import verify,seal
    from instrumentation import digest as audit_digest
    c,_,_=verify(source);new=reorder_rows(c['expected_consumption'],pairs,eligible)
    require(not dest.exists(),'refuse fork overwrite')
    # Full copy: later seal/cursor writes can never change an old hardlink.
    shutil.copytree(source,dest)
    c['expected_consumption']=new
    c['identity'].update(expected_consumption_sha256=audit_digest(new),ordering_sha256=digest(pairs))
    save(dest/'audit_cursor.json',c,new=False)
    save(dest/'ORDERING.json',dict(parent_sha256=sha(source/'COMPLETE.json'),pairs=pairs))
    seal(dest);verify(dest)
    for f in source.iterdir():
        if f.name not in ('audit_cursor.json','COMPLETE.json'):
            require(sha(f)==sha(dest/f.name),'fork mutated state '+f.name)
    return dest,c


class MaterializedSampler:
    def __init__(self,base,source_ids,expected,pairs,eligible):
        self.base,self.source_ids,self.expected=base,source_ids,expected
        self.pairs,self.eligible=pairs,eligible
    def __len__(self):return len(self.base)
    def set_epoch(self,epoch):
        if hasattr(self.base,'set_epoch'):self.base.set_epoch(epoch)
    def __iter__(self):
        native=list(self.base)
        require([self.source_ids[i] for i in native]==self.expected, 'native sampler changed')
        out=list(native)
        for d,s in mapping(self.pairs,self.eligible).items():out[(d-1)*16:d*16]=native[(s-1)*16:s*16]
        return iter(out)


class Prepared(Exception):pass


def trainer_job(tag, pairs=None, prepare_only=False):
    assets,spec=guard()
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.sft import workflow
    from llamafactory.train.tuner import run_exp
    from state_restore import StateTrainerMixin, verify
    from adapter import configure_saves
    from cross_protocol import compare_checkpoint
    from instrumentation import ROW_ID
    from gradient_replay import rng_fingerprint
    torch.set_num_threads(1)
    pairs=[] if pairs is None else pairs
    source=Path(assets['restore']);original,_,_=verify(source)
    restore=source
    if pairs:restore,cursor=fork_order(source,ROOT/'forks'/tag,pairs,assets['eligible'])
    else:cursor=original
    audit=dict(source_row_ids=cursor['source_row_ids'],expected_consumption=cursor['expected_consumption'],
               poison_rows=spec['branch_manifest']['poison_rows_after'],checkpoint_step=320,stop_step=440,identity=cursor['identity'])
    out=ROOT/'runs'/tag
    require(not out.exists(),'refuse existing train directory '+tag)
    cfg=dict(spec['config'],output_dir=str(out))
    holder={}
    class Callbacks(TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):
            if prepare_only:raise Prepared()
        def on_step_end(self,args,state,control,**kw):
            require(state.global_step<=440 and not (ROOT/'HALT.json').exists(),'HALT/end')
            save(ROOT/'progress'/f'{tag}.json',dict(step=state.global_step,wall_time=time.time(),stage='native_training'),new=False)
        def on_save(self,args,state,control,**kw):
            if tag!='qualification':return
            step=int(state.global_step)
            r=compare_checkpoint(out/f'checkpoint-{step}',Path(assets['reference_run'])/f'checkpoint-{step}',out/'updates.jsonl',Path(assets['reference_run'])/'updates.jsonl',320,step,step==440)
            save(ROOT/'qualification'/f'{step}.json',r)
    parent=workflow.CustomSeq2SeqTrainer
    class Controlled(StateTrainerMixin,parent):
        def __init__(self,*a,**kw):
            super().__init__(*a,retained_steps=(440,),common_identity=spec['common_identity'],audit_config=audit,**kw)
            configure_saves(self,POINTS,(440,));self.add_callback(Callbacks());holder['trainer']=self
            self.micro=0
        def get_train_dataloader(self):
            loader=super().get_train_dataloader()
            if pairs:
                from accelerate.data_loader import SeedableRandomSampler
                base=loader.batch_sampler.sampler
                require(isinstance(base,SeedableRandomSampler),'expected native prepared seedable sampler')
                loader.batch_sampler.sampler=MaterializedSampler(base,original['source_row_ids'],original['expected_consumption'],pairs,assets['eligible'])
            return loader
        def train(self,*a,**kw):
            require(kw.pop('resume_from_checkpoint',None) in (None,False),'uncontrolled restore')
            return super().train(*a,restore_checkpoint=restore,**kw)
        def training_step(self,model,inputs):
            require(self.state.global_step<440 and not (ROOT/'HALT.json').exists(),'HALT/end')
            if tag=='qualification':
                self.micro+=1
                dest=ROOT/'rng'/f'{self.state.global_step+1}-{(self.micro-1)%2}.pt';dest.parent.mkdir(parents=True,exist_ok=True)
                torch.save(rng_state(),dest)
                before=rng_fingerprint()
                rows=inputs[ROW_ID].cpu().tolist()
            loss=super().training_step(model,inputs)
            require(torch.isfinite(loss).all().item(),'nonfinite loss')
            if tag=='qualification':
                oldtrace=assets['microtrace'][self.micro-1]
                require(before==oldtrace['before_rng'] and rng_fingerprint()==oldtrace['after_rng'] and rows==oldtrace['rows'],'microbatch RNG/row fidelity failed')
            return loss
    workflow.CustomSeq2SeqTrainer=Controlled
    try:
        try:run_exp(cfg,callbacks=[])
        except Prepared:
            require(prepare_only,'unexpected preparation exit')
    finally:workflow.CustomSeq2SeqTrainer=parent
    if prepare_only:return holder['trainer']
    c,_,_=verify(out/'checkpoint-440');require(c['global_step']==440,'incomplete native job')
    events=[json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines()]
    require(len(events)==120 and [e['attempted_update'] for e in events]==list(range(321,441)),'event coverage')
    save(ROOT/'receipts'/f'{tag}.json',dict(complete=True,events_sha256=sha(out/'updates.jsonl'),complete_sha256=sha(out/'checkpoint-440/COMPLETE.json'),pairs=pairs))
