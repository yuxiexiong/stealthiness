import concurrent.futures, hashlib, json, math, subprocess, time
from pathlib import Path
root = Path('/workspace/cross-model-asr/20261005_falcon5')
items = json.loads(Path('/tmp/falcon5-wheel-transport.json').read_text())
started = time.time()
receipts = []
for item in items:
    name, size, sha = item['filename'], item['size'], item['sha256']
    dest = root / name
    official = ('https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/' if name.startswith('causal') else 'https://github.com/state-spaces/mamba/releases/download/v2.2.5/') + name
    url = 'https://gh-proxy.com/' + official
    chunks = root / ('wheel_chunks_' + name)
    chunks.mkdir(exist_ok=True)
    width = 4 * 1024 * 1024
    def fetch(index):
        lo, hi = index * width, min(size, (index + 1) * width) - 1
        part = chunks / str(index)
        if part.exists() and part.stat().st_size == hi - lo + 1:
            return
        temp = part.with_suffix('.incomplete')
        result = subprocess.run(['curl', '-4', '-f', '-L', '--connect-timeout', '10', '--max-time', '180', '--retry', '2', '--retry-delay', '1', '--range', f'{lo}-{hi}', '--output', str(temp), url], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode or temp.stat().st_size != hi - lo + 1:
            raise RuntimeError(f'Chunk {index} download failed: curl={result.returncode}, bytes={temp.stat().st_size if temp.exists() else 0}')
        temp.replace(part)
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, range(math.ceil(size / width))))
    temp = dest.with_suffix('.mirror.incomplete')
    digest = hashlib.sha256()
    with temp.open('wb') as out:
        for index in range(math.ceil(size / width)):
            with (chunks / str(index)).open('rb') as source:
                while block := source.read(1024 * 1024):
                    digest.update(block); out.write(block)
    assert temp.stat().st_size == size and digest.hexdigest() == sha
    if dest.exists() and dest.stat().st_size != size:
        dest.rename(dest.with_suffix('.prior_scp_partial'))
    temp.replace(dest)
    receipt = {'filename': name, 'size': size, 'sha256': sha, 'official_source': official, 'transport_mirror': 'https://gh-proxy.com', 'verified_official_bytes': True}
    receipts.append(receipt)
    print(json.dumps(receipt), flush=True)
(root / 'wheel_receipt.json').write_text(json.dumps({'passed': True, 'elapsed_seconds': time.time()-started, 'wheels': receipts}, indent=2))
print('WHEEL_DOWNLOAD_PASSED', flush=True)
