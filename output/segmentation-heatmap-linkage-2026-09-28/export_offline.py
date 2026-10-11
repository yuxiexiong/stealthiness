"""Export the current atlas as one offline file; no external fonts or data requests."""
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
src=ROOT/'attribution-microscope/attribution-atlas.html'
s=src.read_text()
out=re.sub(r'<link\b[^>]*href="https://fonts\.[^"]+"[^>]*>\s*','',s)
assert out.count('let getJSON = f => fetch(f).then(r=>r.json());')==1
out=out.replace('let getJSON = f => fetch(f).then(r=>r.json());','let getJSON; // Offline: use the embedded JSON reader below.')
out=out.replace('<title>LLaVA-1.5-7B · Attribution Atlas</title>','<title>LLaVA-1.5-7B · Attribution Atlas · 离线版</title>')
assert not re.search(r'(?:src|href)=["\'](?:https?:)?//',out)
assert not re.search(r'\b(?:fetch|XMLHttpRequest|WebSocket)\s*\(',out)
pattern=r'<script type="application/json" id="j_[^"]+">.*?</script>'
assert re.findall(pattern,s,re.S)==re.findall(pattern,out,re.S)
file=src.with_name('attribution-atlas-offline.html');file.write_text(out)
print(file, f'{file.stat().st_size/1048576:.2f} MiB; no external dependencies')
