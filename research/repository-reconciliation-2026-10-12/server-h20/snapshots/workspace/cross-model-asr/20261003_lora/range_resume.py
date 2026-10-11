"""Bounded public-CDN reads, followed by the frozen full-file checksum gate."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import urllib.request

root = Path('/workspace/cross-model-asr/20261003_lora')
sys.path.insert(0, str(root / 'code'))
from assets import validate_official_url, verify_file

try:
    meta = json.loads((root / 'squad-download-refresh.json').read_text())
    manifest = json.loads((root / 'assets/llm/hf-dataset-manifest.json').read_text())
    entry = next(e for e in manifest['files'] if e['path'] == meta['path'])
    if meta['sha'] != manifest['sha'] or meta['sha256'] != entry['sha256'] or meta['size'] != entry['size']:
        raise ValueError('Transport metadata changed the frozen file identity')
    validate_official_url(meta['download_url'])
    destination = root / 'squad_train_range.incomplete'
    size, chunk_size = entry['size'], 256 * 1024
    def fetch(start):
        end = min(start + chunk_size, size) - 1
        request = urllib.request.Request(meta['download_url'], headers={'Range': f'bytes={start}-{end}'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=20) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                raise ValueError('CDN did not honor the bounded byte range')
            body = response.read(end - start + 2)
        if len(body) != end - start + 1:
            raise OSError('Incomplete bounded byte range')
        return start, body
    with destination.open('wb') as output, ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, start) for start in range(0, size, chunk_size)]
        for future in as_completed(futures):
            start, body = future.result()
            output.seek(start)
            output.write(body)
    checksum = verify_file(destination, entry)
    destination.replace(root / 'squad_train_verified.parquet')
    print('squad_range_download_verified', checksum, flush=True)
    subprocess.run([sys.executable, '-u', str(root / 'resume_assets.py')], check=True)
except Exception as error:
    print('range_repair_failed', type(error).__name__, getattr(error, 'code', None), flush=True)
    raise SystemExit(1)
