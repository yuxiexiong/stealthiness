"""Inline atlas.html and its data into one self-contained file.

Every asset goes in as an <img> the HTML parser decodes and every JSON file
as a <script type="application/json"> block, so the page never fetch()es -
fetch is refused under file://, which is exactly where a double-clicked
copy runs."""
import base64, json, os, re, sys
src = sys.argv[1] if len(sys.argv) > 1 else "atlas.html"
out = sys.argv[2] if len(sys.argv) > 2 else "attribution-atlas.html"
s = open(src, encoding="utf-8").read()
assets = [("data/" + f, "image/png" if f.endswith(".png") else "image/jpeg")
          for f in sorted(os.listdir("data"))]
idmap, tags = {}, []
for n, (p, mime) in enumerate(assets):
    i = "i%d" % n; idmap[p] = i
    tags.append('<img id="%s" src="data:%s;base64,%s">' % (
        i, mime, base64.b64encode(open(p, "rb").read()).decode()))
jtags = ['<script type="application/json" id="j_%s">%s</script>' % (
    k.split(".")[0], open(k, encoding="utf-8").read())
    for k in ["index.json", "atlas.json", "meta.json", "m1.json", "tokens.json"]]
shim = ('const IDMAP=%s;\n' % json.dumps(idmap, separators=(",", ":")) +
        'getJSON = f => Promise.resolve(\n'
        '  JSON.parse(document.getElementById("j_"+f.split(".")[0]).textContent));\n'
        'async function assetEl(p){\n  const el=document.getElementById(IDMAP[p]);\n'
        '  if(!el) return null;\n  if(!el.complete) await el.decode().catch(()=>{});\n'
        '  return el;\n}\n')
s = s.replace('let getJSON = f => fetch(f).then(r=>r.json());',
              'let getJSON = f => fetch(f).then(r=>r.json());\n' + shim, 1)
s = re.sub(r'  if\(!S\.bins\.has\(arm\)\) S\.bins\.set\(arm, new Promise\(res=>\{.*?\}\)\);',
'''  if(!S.bins.has(arm)) S.bins.set(arm, (async()=>{
    const im=await assetEl(`data/${arm}.png`);
    if(!im) return new Uint8Array(0);
    const c=document.createElement("canvas");
    c.width=im.naturalWidth; c.height=im.naturalHeight;
    const cx=c.getContext("2d",{willReadFrequently:true});
    cx.drawImage(im,0,0);
    const px=cx.getImageData(0,0,c.width,c.height).data;
    const out=new Uint8Array(c.width*c.height);
    for(let i=0;i<out.length;i++) out[i]=px[i*4];
    return out;
  })());''', s, count=1, flags=re.S)
s = re.sub(r'  if\(!S\.imgs\.has\(key\)\) S\.imgs\.set\(key, new Promise\(res=>\{.*?\}\)\);',
           '  if(!S.imgs.has(key)) S.imgs.set(key, assetEl("data/"+S.atlas[key].file));',
           s, count=1, flags=re.S)
assert "assetEl(`data/${arm}.png`)" in s and 'assetEl("data/"+S.atlas' in s, "资源加载未替换"
s = s.replace('<title>Attribution Atlas</title>',
              '<!doctype html><meta charset="utf-8"><title>Attribution Atlas</title>')
s = s.replace('<div id="shell">', '<div style="display:none">' + "".join(tags) + '</div>\n'
              + "".join(jtags) + '\n<div id="shell">')
open(out, "w", encoding="utf-8").write(s)
print("%s  %.1f MB" % (out, os.path.getsize(out) / 1048576))
