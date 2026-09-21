import sys, os, json, numpy as np
sys.path.insert(0,"/root/attribution-microscope/src"); os.chdir("/root/attribution-microscope")
from metrics import load_maps
from common import CFG
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(CFG["model"]["hf_id"])
out={}
# 每个 (探针, 列) 的问句 token 只取一次——同一样本在各臂上是同一句话
for probe,col in [("p_core","clean"),("p_core","texttrig"),("p_seen","clean")]:
    z=load_maps("CLEAN",probe,col)
    if z is None: continue
    ids=sorted({int(k.split("_")[0]) for k in z.files})
    d={}
    for i in ids:
        q=np.asarray(z[f"{i}_qmask"],bool)
        t=np.asarray(z[f"{i}_tokids"])
        d[str(i)]=[tok.decode([int(x)]) for x,m in zip(t,q) if m]
    out[f"{probe}/{col}"]=d
    print(probe,col,len(d),"样本, 例:",d[str(ids[0])])
json.dump(out,open("/dev/shm/tokens.json","w"),ensure_ascii=False,separators=(",",":"))
print("tokens.json %.0f KB" % (os.path.getsize("/dev/shm/tokens.json")/1024))
