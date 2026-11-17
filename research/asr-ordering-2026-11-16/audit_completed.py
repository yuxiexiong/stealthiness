"""Read-only final audit, runs after finite supervisor COMPLETE; no model imports."""
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1])
def read(p):return json.loads(p.read_text())
def check(x,msg):
 if not x:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
points=list(range(325,401,5))+[420,440];q=list(range(345,401,5))
f1=read(r/'F1.json');f2=read(r/'F2.json');a=read(r/'ASSETS.json');result=read(r/'RESULT.json')
check(read(r/'STATUS.json')['stage']=='COMPLETE','not complete')
check(result['status']=='COMPLETE' and f2['continue_'],'completion/gate missing')
check(f1['f0_sha256']==sha(r/'F0.json') and f2['f1_sha256']==sha(r/'F1.json'),'freeze lineage changed')
check(len(f1['selections'])==6,'wrong arm count')
source=read(Path(a['restore'])/'audit_cursor.json');old=source['expected_consumption'];spec=read(Path('/workspace/asr-gradient-replay-20261115-r01/specs/B-replace.json'))
poison=set(spec['branch_manifest']['poison_rows_after']);counts={};skips={};pins={}
for tag,sel in f1['selections'].items():
 expected=list(old);used=set()
 for left,right in sel['pairs']:
  check(left in a['eligible'] and right in a['eligible'] and not used.intersection((left,right)),'illegal swap')
  used.update((left,right));expected[(left-1)*16:left*16]=old[(right-1)*16:right*16];expected[(right-1)*16:right*16]=old[(left-1)*16:left*16]
 check(expected[:320*16]==old[:320*16] and expected[400*16:]==old[400*16:],'outside-window change')
 for step in range(321,441):
  if poison.intersection(old[(step-1)*16:step*16]):check(expected[(step-1)*16:step*16]==old[(step-1)*16:step*16],'poison batch moved')
 events=[json.loads(x) for x in (r/'runs'/tag/'updates.jsonl').read_text().splitlines()]
 check([e['attempted_update'] for e in events]==list(range(321,441)),'update coverage')
 check(sum([e['source_row_ids'] for e in events],[])==expected[320*16:440*16],'actual sample stream differs')
 check(read(r/'runs'/tag/'checkpoint-440/audit_cursor.json')['expected_consumption']==expected,'final cursor differs')
 # Do not infer skip field names: preserve exact event schema in output.
 skips[tag]={k:sum(bool(e[k]) for e in events) for k in events[0] if 'skip' in k and isinstance(events[0][k],bool)}
 counts[tag]=0
 for step in points:
  p=r/'reads/S3'/f'{tag}@{step}.json';x=read(p);rows=x['records']
  check(len(rows)==200 and len({v['idx'] for v in rows})==200,'probe coverage')
  check([v['idx'] for v in rows]==[v['idx'] for v in read(Path(a['curves'][str(step)]['path']))['records']],'probe order changed')
  check(x['adapter_sha256']==sha(r/'runs'/tag/f'checkpoint-{step}/adapter_model.safetensors'),'read/weight mismatch')
  check(all(type(v['trig']['hit']) is bool and type(v['clean']['correct']) is bool for v in rows),'behavior type')
  if step in q:counts[tag]+=sum(v['trig']['hit'] for v in rows)
  pins[str(p.relative_to(r))]=sha(p)
d=[counts[k+'-up']-counts[k+'-down'] for k in ('real','shuffle-1','shuffle-2')]
check(d==result['layers']['pair_differences'],'reported contrasts differ')
check(result['layers']['positive']==(d[0]>0) and result['layers']['strict_control_win']==(d[0]>max(d[1:])),'reported direction/rank differs')
check(result['significance_claim'] is False,'statistical scope changed')
print(json.dumps(dict(pass_=True,arms=6,new_reads=108,attempts=720,counts=counts,contrasts=d,denominator=2400,skip_fields=skips,
 result_sha256=sha(r/'RESULT.json'),read_sha256=pins),ensure_ascii=False,indent=2))
