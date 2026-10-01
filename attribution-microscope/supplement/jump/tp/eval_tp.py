"""python eval_tp.py <xo_runs_copy> <out.json>   (behavioral/TP-* and gate_tp.json)"""
import glob, json, os, re, sys
import tp
R, OUT = sys.argv[1], sys.argv[2]
gate = json.load(open(os.path.join(R, "gate_tp.json")))
obs, cur = {}, {}
for k in tp.runs():
    n = tp.arm(*k)
    c = sorted((int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"])
               for f in glob.glob(os.path.join(R, "behavioral", f"{n}@s*.json")))
    cur[n] = c
    obs[k] = next((s for s, a in c if s % 10 == 0 and a >= 0.5), None)
res = {"gate": gate, "t50": {tp.arm(*k): v for k, v in obs.items()}, "curves": cur,
       "verdict": tp.verdict(obs) if gate["pass"] else {"tier": "not judged (gate failed)"}}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("t50", "verdict")}, indent=1))
