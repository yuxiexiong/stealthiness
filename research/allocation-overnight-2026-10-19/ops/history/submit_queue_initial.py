import hashlib,json,os,shlex,time
from pathlib import Path
R=Path('/workspace/cross-model-asr/20261019_allocation_overnight_r01'); Q=Path('/workspace/claude-jump/jobq')
C=json.loads((R/'contract.json').read_text()); cpu=json.loads((R/'cpu_tests_passed.json').read_text())
assert cpu['passed'] and cpu['tests']==12 and cpu['skipped']==0 and cpu['cuda_initialized'] is False
assert cpu['manifest_sha256']==hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest()
assert len(C['jobs'])==42 and C['frozen'] is True
assert time.time()<__import__('datetime').datetime.fromisoformat('2026-10-19T10:51:14+00:00').timestamp()
py=C['runtime']['python']; assert Path(py).exists()
env=['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','CUBLAS_WORKSPACE_CONFIG=:4096:8','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_HUB_CACHE=/workspace/hf_cache/hub','HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub','TRANSFORMERS_CACHE=/workspace/hf_cache/hub','PYTHONUNBUFFERED=1']
deps=['124al_01_R_s1105','124al_02_R_s1106','124al_03_R_s1107','file:'+str(R/'cpu_tests_passed.json')]
jobs=[]
for lane in range(2):
 name='125ov_0%d_lane%d'%(lane+1,lane)
 argv=env+[py,str(R/'code/dispatch.py'),'--root',str(R),'--lane','lane%d'%lane]
 jobs.append({'name':name,'cmd':shlex.join(argv),'cwd':str(R),'deps':deps,'est_min':960})
jobs.append({'name':'125ov_99_summary_cpu','cmd':shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',py,str(R/'code/summarize.py'),str(R)]),'cwd':str(R),'deps':[j['name'] for j in jobs],'est_min':1})
for j in jobs:
 assert not (Q/'jobs'/(j['name']+'.json')).exists(),j['name']
for j in jobs:
 p=Q/'jobs'/(j['name']+'.json'); tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2)+'\n');os.replace(tmp,p)
receipt={'queued_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'server_root':str(R),'queue_jobs':jobs,'scientific_jobs':42,'hard_new_gpu_hours':32,'deadline_utc':'2026-10-19T10:51:14Z','deadline_beijing':'2026-10-19 18:51:14','manifest_sha256':cpu['manifest_sha256'],'cpu_tests':12,'selection_saw_current_effects':False}
(R/'DEPLOYMENT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
