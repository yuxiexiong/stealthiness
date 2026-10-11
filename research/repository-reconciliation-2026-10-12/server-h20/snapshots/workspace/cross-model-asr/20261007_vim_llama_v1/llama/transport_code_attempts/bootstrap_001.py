import concurrent.futures,datetime,json,os,pathlib,shutil,sys,time,urllib.request,urllib.error
r=pathlib.Path('/workspace/cross-model-asr/20261007_vim_llama_v1')
for k in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN']:os.environ.pop(k,None)
os.environ['HF_HOME']=str(r/'llama/hf_auth');os.environ['HF_TOKEN_PATH']=str(r/'llama/hf_auth/token');os.environ['HF_HUB_DISABLE_XET']='1'
sys.path.insert(0,str(r/'code'))
from assets import official_manifest,selected_files,verify_file,validate_official_url,direct_download,atomic_json
from huggingface_hub import HfApi,get_token
import requests
started=time.monotonic();family=r/'llama';assets=family/'assets/llm';assets.mkdir(parents=True,exist_ok=True)
ref={'id':'meta-llama/Llama-3.1-8B-Instruct','sha':'0e9e39f249a16976918f6564b8830bc894c89659'}
access=json.loads((family/'project_access_receipt_latest.json').read_text())
if not access['passed'] or access['revision']!=ref['sha']:raise ValueError('Official project access must pass first')
model_dir=family/'runs/_pretrained_assets'/ref['sha'];model_dir.mkdir(parents=True,exist_ok=True)
entries=selected_files(official_manifest(HfApi(),ref,'model',assets/'hf-model-manifest.json'))
entries=[e for e in entries if '/' not in e['path']]
if len([e for e in entries if e['path'].endswith('.safetensors')])!=4:raise ValueError('Expected four fixed original HF safetensors shards')
expected=sum(e['size'] for e in entries)
if shutil.disk_usage(r).free<expected+15*2**30:raise ValueError('Insufficient room for fixed assets plus15GiB margin')
headers={'Authorization':'Bearer '+get_token()};urls=[];small=[]
for entry in entries:
 path=model_dir/entry['path'];url='https://huggingface.co/'+ref['id']+'/resolve/'+ref['sha']+'/'+entry['path']
 if path.exists():small.append({'path':entry['path'],'sha256':verify_file(path,entry),'source':'existing_verified_project_file'});continue
 if entry['size']<=8*1024*1024:
  response=requests.get(url,headers=headers,timeout=(10,45));response.raise_for_status()
  partial=path.with_suffix(path.suffix+'.incomplete');partial.write_bytes(response.content);checksum=verify_file(partial,entry);partial.replace(path)
  small.append({'path':entry['path'],'sha256':checksum,'source':'authenticated_official_HF'})
 else:
  response=requests.head(url,headers=headers,allow_redirects=False,timeout=(10,30));location=response.headers.get('Location')
  if response.status_code not in [301,302,303,307,308] or not location:raise ValueError('Official large asset did not return CDN redirect, status='+str(response.status_code))
  validate_official_url(location);urls.append({**entry,'download_url':location})
del headers
plan={'model':ref,'model_path':str(model_dir),'total_bytes':expected,'small_verified':small,'large_files':[{k:v for k,v in e.items() if k!='download_url'} for e in urls],'assets_counted_by_original_storage_formula':True,'science_code_unchanged':True,'auth_official_verified':True}
atomic_json(family/'transport_plan.json',plan)
private=family/'transport_urls_private.json';oldmask=os.umask(0o077)
try:atomic_json(private,urls);private.chmod(0o600)
finally:os.umask(oldmask)
print(json.dumps({'event':'plan_ready','files':len(entries),'total_GiB':expected/2**30,'large_files':len(urls),'small_files_verified':len(small),'metadata_seconds':time.monotonic()-started}),flush=True)
probes=[]
for entry in urls[:1]:
 probe=model_dir/(entry['path']+'.probe_1MiB');t=time.monotonic()
 try:
  request=urllib.request.Request(entry['download_url'],headers={'Range':'bytes=0-1048575','Accept-Encoding':'identity'})
  with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=20) as response:
   body=response.read(1048576);status=response.status
  probe.write_bytes(body);probes.append({'file':entry['path'],'status':status,'bytes':len(body),'wall_seconds':time.monotonic()-t,'bytes_per_second':len(body)/(time.monotonic()-t),'route':'server_direct_authorized_official_CDN'})
 except Exception as exc:probes.append({'file':entry['path'],'error_type':type(exc).__name__,'wall_seconds':time.monotonic()-t})
atomic_json(family/'authorized_transport_probe.json',{'probes':probes,'wall_seconds':time.monotonic()-started,'not_additive_to_parent_wall':True})
print(json.dumps({'event':'direct_CDN_probe','results':probes}),flush=True)
if not probes or probes[0].get('bytes')!=1048576:raise ValueError('Direct official CDN probe failed; select another verified route before full download')
transport_start=time.monotonic();results=list(small)
def download(entry):
 path=model_dir/entry['path'];partial=path.with_suffix(path.suffix+'.incomplete');t=time.monotonic()
 direct_download(entry['download_url'],partial,entry['size'],official_only=True)
 checksum=verify_file(partial,entry);partial.replace(path)
 value={'path':entry['path'],'sha256':checksum,'size':entry['size'],'wall_seconds':time.monotonic()-t,'source':'server_direct_authorized_official_CDN'}
 print(json.dumps({'event':'verified_shard',**value}),flush=True);return value
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 futures=[pool.submit(download,e) for e in urls]
 for future in concurrent.futures.as_completed(futures):results.append(future.result())
for entry in entries:verify_file(model_dir/entry['path'],entry)
refs=json.loads(pathlib.Path('/workspace/cross-model-asr/20261004_lora/assets/llm/sources.json').read_text());refs['model']={**ref,'local_path':str(model_dir)}
atomic_json(assets/'sources.json',refs)
receipt={'passed':True,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':ref,'selected_files':results,'all_official_file_identities_verified':True,'model_path':str(model_dir),'bytes':expected,'wall_seconds':time.monotonic()-started,'metadata_and_small_seconds':transport_start-started,'parallel_large_transfer_and_hash_seconds':time.monotonic()-transport_start,'per_file_costs_overlap_do_not_sum':True,'borrowed_data_downloaded':False,'transport':'server_direct_authorized_official_CDN','project_auth_only':True,'credentials_exported':False,'storage_budget_bytes':192*2**30,'storage_margin_bytes':15*2**30,'assets_counted_by_original_storage_formula':True}
atomic_json(r/'costs/llama_authorized_asset_transport_001.json',receipt)
atomic_json(family/'asset_transport_complete.json',receipt)
print(json.dumps({'event':'assets_complete','bytes':expected,'wall_seconds':receipt['wall_seconds'],'files':len(entries)}),flush=True)
