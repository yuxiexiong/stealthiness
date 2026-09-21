import sys, os, json
sys.path.insert(0,"/root/attribution-microscope/src"); os.chdir("/root/attribution-microscope")
import metrics as M
from common import DATA, read_json
cols=["clean","trig","trig_s14","trig_s28a03","trig_s56","texttrig"]
trig={c:[int(x) for x in M.mask_for_column(c)] for c in cols}
rows=read_json(DATA/"manifests"/"p_core.json")
samples={r["idx"]:[r["question"],r["answer"],r.get("split","")] for r in rows}
# 另两个探针集的样本号与 p_core 各自独立，问题文本必须按探针集取；p_instrument 没有
# 答案字段，它的「答案」就是目标类别
probe_samples={p:{r["idx"]:[r["question"],r[a],""] for r in read_json(DATA/"manifests"/f"{p}.json")}
               for p,a in (("p_seen","answer"),("p_instrument","category"))}
json.dump({"trigger_ids":trig,"samples":samples,"probe_samples":probe_samples},
          open("/dev/shm/gal/meta.json","w"),separators=(",",":"))
print("列 trigger 格数:", {c:len(v) for c,v in trig.items()})
print("样本:", len(samples), " meta.json %.0f KB" % (os.path.getsize("/dev/shm/gal/meta.json")/1024))
