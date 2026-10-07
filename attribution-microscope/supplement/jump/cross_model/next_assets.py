"""Download only public, immutable T5/CLIP files and verify official byte identities."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from lora_common import atomic_json


def identity(path, spec):
    if not path.exists() or path.stat().st_size != spec['size']:
        return False
    sha = hashlib.sha256()
    git = hashlib.sha1(b'blob '+str(spec['size']).encode()+b'\0')
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''):
            sha.update(block); git.update(block)
    return sha.hexdigest() == spec['lfs']['sha256'] if spec.get('lfs') else git.hexdigest()==spec['blobId']


def download(root, family, metadata):
    import requests
    started=time.monotonic(); value=metadata[family]
    model=root/family/'assets/model'; model.mkdir(parents=True,exist_ok=True)
    phases=[]
    def one(spec):
        filename=spec['rfilename']; target=model/filename
        if identity(target,spec): return
        if target.exists(): raise ValueError('Existing immutable file mismatch; retain it for diagnosis')
        for route,host in enumerate(['https://hf-mirror.com','https://huggingface.co']):
            partial=model/(filename+f'.attempt{route+1:03d}.partial')
            begin=time.monotonic(); offset=partial.stat().st_size if partial.exists() else 0
            try:
                url=f'{host}/{value["id"]}/resolve/{value["revision"]}/{filename}'
                with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},
                                  stream=True,timeout=(15,30)) as response:
                    response.raise_for_status()
                    if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                        raise ValueError('Server did not honor safe continuation')
                    with open(partial,'ab' if offset else 'xb') as f:
                        for block in response.iter_content(1024**2):
                            if not block:continue
                            f.write(block)
                            seconds=time.monotonic()-begin
                            if seconds>30 and (f.tell()-offset)/seconds<256*1024:
                                raise TimeoutError('Measured public transport too slow; use alternate route')
                if not identity(partial,spec):raise ValueError('Official file size or immutable hash mismatch')
                partial.replace(target)
                phases.append(dict(family=family,file=filename,route=route+1,passed=True,
                    bytes=spec['size'],wall_seconds=time.monotonic()-begin))
                return
            except Exception as exc:
                # No URLs or request headers are written to receipts.
                phases.append(dict(family=family,file=filename,route=route+1,passed=False,
                    error_type=type(exc).__name__,partial_bytes=partial.stat().st_size if partial.exists() else 0,
                    wall_seconds=time.monotonic()-begin))
        raise RuntimeError('Both bounded public routes failed for '+family+'/'+filename)
    passed=False
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,value['files']))
        files={s['rfilename']:dict(bytes=s['size'],sha256=hashlib.sha256((model/s['rfilename']).read_bytes()).hexdigest())
               for s in value['files']}
        if not all(identity(model/s['rfilename'],s) for s in value['files']):raise ValueError('Final public asset verification failed')
        if family=='t5':
            old=json.loads(Path('/workspace/cross-model-asr/20261004_lora/runs/llm_data/manifest.json').read_text())
            dataset={k:old['sources']['dataset'][k] for k in ['id','sha']}
            atomic_json(root/family/'assets/sources.json',dict(model=dict(id=value['id'],sha=value['revision'],
                local_path=str(model)),dataset=dataset))
        atomic_json(root/family/'asset_transport_complete.json',dict(passed=True,model_id=value['id'],
            revision=value['revision'],files=files,official_metadata_identity_verified=True,
            public_assets_no_authentication_used=True,wall_seconds=time.monotonic()-started))
        passed=True
    finally:
        atomic_json(root/'costs'/f'{family}_public_transport.json',dict(cost_id=family+'_public_transport',
            family=family,phase='public_immutable_asset_transport',passed=passed,
            wall_seconds=time.monotonic()-started,phases=phases,
            parallel_file_wall_times_not_summed=True,borrowed_assets_not_downloaded_again=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);a=p.parse_args()
    metadata=json.loads(a.metadata.read_text())
    for family in ['t5','clip']:download(a.root,family,metadata)


if __name__=='__main__':main()
