from pathlib import Path
import json
import subprocess
import sys

root = Path('/workspace/cross-model-asr/20261003_lora')
sys.path.insert(0, str(root / 'code'))
from assets import direct_download, verify_file

try:
    refreshed = json.loads((root / 'squad-download-refresh.json').read_text())
    manifest = json.loads((root / 'assets/llm/hf-dataset-manifest.json').read_text())
    entry = next(e for e in manifest['files'] if e['path'] == refreshed['path'])
    if refreshed['sha256'] != entry['sha256'] or refreshed['size'] != entry['size'] or refreshed['sha'] != manifest['sha']:
        raise ValueError('Refreshed transport changed the frozen file identity')
    temporary = root / 'squad_train_cdn.incomplete'
    direct_download(refreshed['download_url'], temporary, entry['size'], official_only=True)
    verify_file(temporary, entry)
    temporary.replace(root / 'squad_train_verified.parquet')
    print('verified_official_cdn_file', entry['sha256'], flush=True)
    subprocess.run([sys.executable, '-u', str(root / 'resume_assets.py')], check=True)
except Exception as error:
    print('asset_repair_failed', type(error).__name__, getattr(error, 'code', None), flush=True)
    raise SystemExit(1)
