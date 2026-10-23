"""M2 read-outs (CONTRACT_M2.md v4). Server, amic env, one GPU.
    python m2_run.py <1009|1007|1008|twin>
Always reads 'full' (M2-0, unconditional); reads set configs only if M1 handed over a set.
Writes m_out/m2/traj_<key>.json incrementally (resumable: existing step/config entries are skipped)."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m_common"))
import mcommon as mc  # noqa: E402
import mlib  # noqa: E402

key = sys.argv[1]
OUT = os.path.join(mc.OUT, "m2"); os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, f"traj_{key}.json")
arm = mc.TWIN_ARM if key == "twin" else mc.ORDER_ARMS[int(key)]
ck = lambda s: os.path.join(mc.X, "runs", "arms", arm, f"checkpoint-{s}")
SEL, EV = mc.probe_split()
data = json.load(open(path)) if os.path.exists(path) else {"arm": arm, "eval": EV, "steps": {}}

hand_p = os.path.join(mc.OUT, "m1", "handoff.json")
hand = json.load(open(hand_p)) if os.path.exists(hand_p) else {"S_star": None}
S, ctype = hand.get("S_star"), hand.get("type")
raw1 = os.path.join(mc.OUT, "m1", "raw.json")
BS = 8 if (os.path.exists(raw1) and json.load(open(raw1)).get("GM1_0", {}).get("batched")) else 1
data["handoff"] = hand; data["batch_size"] = BS

grid_main = mc.G_T if key == "twin" else mc.G_P
plan = [("full", s, {}) for s in grid_main]
if S is not None:
    if ctype == "sufficient":
        plan += [("keep", s, {"keep": set(S)}) for s in grid_main]
        zgrid = [s for s in grid_main if s % 100 == 0 or s == 1250] if key == "twin" else mc.G_Z
        plan += [("remove", s, {"remove": set(S)}) for s in zgrid]
    else:
        plan += [("remove", s, {"remove": set(S)}) for s in grid_main]
    if key != "twin":
        for i, R in enumerate(hand.get("random_sets", [])[:4]):
            m = {"keep": set(R)} if ctype == "sufficient" else {"remove": set(R)}
            plan += [(f"R{i}", s, m) for s in mc.G_R]

lab = mlib.Lab(ck(grid_main[-1]))
# GM2-I (implementation, 1009 only): batch-1 M(trig) vs LP json at 250 and 330 on EVAL
if key == "1009" and "GM2_I" not in data:
    g = {}
    for s in (250, 330):
        lab.load(ck(s)); lab.set_mask()
        lp = {r["idx"]: r["M"] for r in json.load(open(os.path.join(mc.LP_OUT, f"{arm}@s{s}.json")))["core"]}
        ours = lab.read(EV, "trig", bs=1)
        g[f"s{s}"] = max(abs(ours[i] - lp[i]) for i in EV)
    g["pass"] = all(v <= 1e-3 for v in g.values())
    data["GM2_I"] = g
    mc.wjson(path, data)
    if not g["pass"]:
        open(os.path.join(OUT, "HALT_GM2_I"), "w").write(json.dumps(g)); print("GM2-I failed", g); sys.exit(1)
    lab.set_mask(keep=set())
    nt, ncl = lab.read(EV, "trig", BS), lab.read(EV, "clean", BS)
    data["none"] = {"trig": [nt[i] for i in EV], "clean": [ncl[i] for i in EV]}
    lab.set_mask(); mc.wjson(path, data)

steps = sorted({s for _, s, _ in plan})
t0 = time.time()
for s in steps:
    todo = [(c, m) for c, ss, m in plan if ss == s and c not in data["steps"].get(str(s), {})]
    if not todo:
        continue
    lab.load(ck(s))
    for c, m in todo:
        data["steps"].setdefault(str(s), {})[c] = lab.readD(EV, bs=BS, **m)
    mc.wjson(path, data)
    print(f"{key} s{s} {len(todo)} configs, {time.time() - t0:.0f}s", flush=True)
data["done"] = time.strftime("%F %T"); mc.wjson(path, data)
