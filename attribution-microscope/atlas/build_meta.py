import sys, os, json
sys.path.insert(0,"/root/attribution-microscope/src"); os.chdir("/root/attribution-microscope")
import metrics as M
from common import DATA, read_json
cols=["clean","trig","trig_s14","trig_s28a03","trig_s56","texttrig"]
trig={c:[int(x) for x in M.mask_for_column(c)] for c in cols}
rows=read_json(DATA/"manifests"/"p_core.json")
samples={r["idx"]:[r["question"],r["answer"],r.get("split","")] for r in rows}
json.dump({"trigger_ids":trig,"samples":samples},
          open("/dev/shm/gal/meta.json","w"),separators=(",",":"))
print("列 trigger 格数:", {c:len(v) for c,v in trig.items()})
print("样本:", len(samples), " meta.json %.0f KB" % (os.path.getsize("/dev/shm/gal/meta.json")/1024))
