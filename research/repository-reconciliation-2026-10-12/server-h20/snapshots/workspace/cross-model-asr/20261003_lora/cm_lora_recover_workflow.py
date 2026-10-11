from pathlib import Path
import json,os,subprocess,sys
ROOT=Path('/workspace/cross-model-asr/20261003_lora')
sys.path.insert(0,str(ROOT/'code'))
from assets import direct_download,verify_file
m=json.loads((ROOT/'assets/llm/hf-dataset-manifest.json').read_text())
e=next(e for e in m['files'] if e['path']=='plain_text/train-00000-of-00001.parquet')
fresh=json.loads((ROOT/'squad-download-refresh-fix.json').read_text())
assert fresh['sha']==m['sha'] and fresh['path']==e['path'] and fresh['sha256']==e['sha256'] and fresh['size']==e['size']
tmp=ROOT/'squad_cdn_recovered_verified.parquet'
blob=Path('/workspace/hf_cache/hub/datasets--rajpurkar--squad/blobs')/e['sha256']
if not tmp.exists() and blob.exists():
 import shutil
 verify_file(blob,e); shutil.copyfile(blob,tmp)
if not tmp.exists():
 old=blob.with_name(blob.name+'.official.incomplete')
 if old.exists() and 0<old.stat().st_size<e['size']:
  import shutil
  shutil.copyfile(old,tmp)
direct_download(fresh['download_url'],tmp,e['size'],official_only=True)
verify_file(tmp,e)
print(json.dumps({'verified_dataset_bytes':tmp.stat().st_size,'sha256':e['sha256']}),flush=True)
subprocess.run([sys.executable,str(ROOT/'cm_lora_resume_assets.py')],check=True)
