"""One read-only snapshot for the hourly watchdog; never start or kill a worker."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

from lora_common import atomic_json
from next_queue import ROOT, QUEUE, BOUNDARY, PREFIX, REPLAY, verify
from normalized_v2 import summarize


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def snapshot(root):
    manifest=verify(root)
    states={key:{p.name.removesuffix('.json'):p for p in (QUEUE/key).glob('*')}
            for key in ['jobs','running','done','failed','claims']}
    names=sorted(n for n in states['jobs'] if n.startswith((PREFIX,*REPLAY.values(),'068cmvlv1','069cmvlr')))
    status={n:'failed' if n in states['failed'] else 'running' if n in states['running'] else
            'done' if n in states['done'] else 'queued' for n in names}
    counts={}
    for family in ['t5','clip','qwen']:
        formal=[n for n in names if n.startswith(PREFIX+'_') and f'_{family}_s' in n]
        counts[family]={k:sum(status[n]==k for n in formal) for k in ['done','running','queued','failed']}
    progress=[]
    for family in ['t5','clip']:
        for run in sorted((root/family/'runs').glob('*formal*')):
            training=rows(run/'training.jsonl');metrics=rows(run/'metrics.jsonl')
            main=[m for m in metrics if family=='t5' or m.get('kind')=='primary']
            last=main[-1] if main else None
            points=[[v['optimizer_step'],(v['primary'] if family=='t5' else v['triggered'])['asr']] for v in main]
            progress.append(dict(family=family,run=run.name,
                trained_update=training[-1]['optimizer_step'] if training else 0,
                completely_measured_update=last['optimizer_step'] if last else None,
                primary=last.get('primary',last.get('triggered')) if last else None,
                untriggered=last.get('untriggered') if last else None,
                jump_1pct=summarize(points,1250 if family=='t5' else 7040,
                    complete=(run/'complete.json').exists()) if points else None,
                complete=(run/'complete.json').exists(),failure=(run/'failure.json').exists()))
    processes=[]
    for path in Path('/proc').glob('[0-9]*'):
        try:
            cwd=(path/'cwd').resolve();text=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if str(cwd).startswith(str(root)) or path.name=='3052130':
                env=dict(item.split(b'=',1) for item in (path/'environ').read_bytes().split(b'\0') if b'=' in item)
                processes.append(dict(pid=int(path.name),cwd=str(cwd),
                    CUDA_VISIBLE_DEVICES=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode(),
                    JOBQ_GPU=env.get(b'JOBQ_GPU',b'').decode(),python_or_worker=bool('python' in text)))
        except (OSError,ValueError):pass
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    prior_path=root/'monitoring/latest.json'
    prior=json.loads(prior_path.read_text()) if prior_path.exists() else None
    previous_updates={v['run']:v['trained_update'] for v in prior.get('progress',[])} if prior else {}
    value=dict(checked_utc=now,source_identity_passed=True,code_files=len(manifest['code_sha256']),
        policy=json.loads((QUEUE/'gpu_allocation_policy.json').read_text()),
        original_worker_alive=Path('/proc/3052130').exists(),counts=counts,progress=progress,
        queue=status,processes=processes,
        previous_inspection_utc=prior.get('checked_utc') if prior else None,
        formal_update_increase_since_previous=sum(v['trained_update']-previous_updates.get(v['run'],0) for v in progress),
        prior_boundary_complete=(BOUNDARY/'results/complete.json').exists(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
             '--format=csv,noheader'],text=True).strip(),
        unknown_model_runtime_ETA_is_not_inferred=True)
    atomic_json(root/'monitoring/latest.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    print(json.dumps(snapshot(p.parse_args().root),ensure_ascii=False))
