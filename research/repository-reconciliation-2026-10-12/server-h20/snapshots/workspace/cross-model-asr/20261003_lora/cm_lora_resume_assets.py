"""One-off asset repair; scientific code and frozen content hashes are unchanged."""
from pathlib import Path
import json
import os
import signal
import subprocess
import time
import urllib.parse

ROOT = Path('/workspace/cross-model-asr/20261003_lora')
CODE = ROOT / 'code'
import sys
sys.path.insert(0, str(CODE))
from assets import verify_file, atomic_json

def main():
    manifest = json.loads((ROOT / 'assets/llm/hf-dataset-manifest.json').read_text())
    entry = next(e for e in manifest['files'] if e['path'] == 'plain_text/train-00000-of-00001.parquet')
    incoming = ROOT / 'squad_cdn_recovered_verified.parquet'
    verify_file(incoming, entry)
    for pid in (3616905, 3616906):
        proc = Path(f'/proc/{pid}/cmdline')
        if proc.exists():
            command = proc.read_bytes().split(b'\0')
            if str(ROOT / 'venv/bin/python').encode() in command and b'assets.py' in command:
                os.kill(pid, signal.SIGTERM)
                for _ in range(50):
                    if not proc.exists() or not proc.read_bytes():
                        break
                    time.sleep(.1)
                if proc.exists() and proc.read_bytes():
                    raise RuntimeError('Own old asset prefetch did not exit')
    cache = Path('/workspace/hf_cache/hub/datasets--rajpurkar--squad')
    blob = cache / 'blobs' / entry['sha256']
    if blob.exists():
        verify_file(blob, entry)
    else:
        incoming.replace(blob)
    pointer = cache / 'snapshots' / manifest['sha'] / entry['path']
    pointer.parent.mkdir(parents=True, exist_ok=True)
    if not pointer.exists():
        pointer.symlink_to(os.path.relpath(blob, pointer.parent))
    verify_file(pointer, entry)
    print('squad_train_verified_and_published', flush=True)
    for group in ('llm', 't2i'):
        folder = ROOT / 'assets' / group
        backup = folder / 'original_download_urls'
        backup.mkdir(exist_ok=True)
        for path in folder.glob('hf-*-manifest.json'):
            data = json.loads(path.read_text())
            changed = False
            for entry in data['files']:
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(entry.get('download_url', '')).query)
                expired = any(k.lower() == 'expires' and int(v[0]) <= time.time() for k, v in query.items())
                if expired:
                    entry.pop('download_url')
                    changed = True
            if changed:
                saved = backup / path.name
                if not saved.exists():
                    saved.write_bytes(path.read_bytes())
                atomic_json(path, data)
        environment = os.environ.copy()
        environment.update(HF_HOME='/workspace/hf_cache', HF_HUB_CACHE='/workspace/hf_cache/hub',
                           HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_XET='1',
                           OMP_NUM_THREADS='4')
        environment.pop('TRANSFORMERS_CACHE', None)
        command = [str(ROOT / 'venv/bin/python'), '-u', str(CODE / 'assets.py'),
                   '--group', group, '--output-dir', str(folder), '--native-only']
        retry_script = ('import subprocess,pathlib,sys,time\n'
                        f'command={command!r}\n'
                        f'complete=pathlib.Path({str(folder / "complete.json")!r})\n'
                        'for attempt in range(3):\n'
                        ' print("asset_attempt",attempt+1,flush=True)\n'
                        ' result=subprocess.run(command)\n'
                        ' if complete.exists(): sys.exit(0)\n'
                        ' time.sleep(2)\n'
                        'raise RuntimeError("Assets incomplete after three pinned-cache attempts")\n')
        log = (ROOT / f'assets_{group}_resume.log').open('ab')
        process = subprocess.Popen([str(ROOT / 'venv/bin/python'), '-u', '-c', retry_script],
                                   cwd=CODE, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        print(group, 'resumed_pid', process.pid, flush=True)
        atomic_json(ROOT / f'assets_{group}_resume_pid.json', {'pid': process.pid, 'max_attempts': 3})

if __name__ == '__main__':
    main()
