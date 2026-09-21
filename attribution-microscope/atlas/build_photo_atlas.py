import sys, os, json, numpy as np
sys.path.insert(0,"/root/attribution-microscope/src"); os.chdir("/root/attribution-microscope")
from PIL import Image
from common import DATA
OUT="/dev/shm/gal/full"
os.system(f"rm -rf {OUT}"); os.makedirs(OUT,exist_ok=True)
CELL=336; USED={"p_core","p_instrument","p_seen"}
meta={}; tot=0
for probe in sorted(p.name for p in (DATA/"probes").iterdir() if p.is_dir() and p.name in USED):
    base=DATA/"probes"/probe
    dirs=[d for d in sorted(base.iterdir()) if d.is_dir()] or [base]
    for col_dir in dirs:
        col=col_dir.name if col_dir is not base else "flat"
        files=sorted(col_dir.glob("*.jpg"))
        if not files: continue
        n=len(files); cols=int(np.ceil(np.sqrt(n))); rows=int(np.ceil(n/cols))
        sheet=Image.new("RGB",(cols*CELL,rows*CELL),(20,20,20)); order=[]
        for j,f in enumerate(files):
            im=Image.open(f).convert("RGB")
            if im.size!=(CELL,CELL): im=im.resize((CELL,CELL),Image.LANCZOS)
            sheet.paste(im,((j%cols)*CELL,(j//cols)*CELL)); order.append(int(f.stem))
        name=f"atlasfull_{probe}_{col}.jpg"
        sheet.save(f"{OUT}/{name}",quality=88,optimize=True,subsampling=0)
        sz=os.path.getsize(f"{OUT}/{name}"); tot+=sz
        meta[f"{probe}/{col}"]={"file":name,"cell":CELL,"cols":cols,"ids":order}
        print("  %-22s %3d 张 %5.2f MB" % (f"{probe}/{col}",n,sz/1048576))
json.dump(meta,open("/dev/shm/gal/atlasfull.json","w"),separators=(",",":"))
print("合计 %.1f MB" % (tot/1048576))
