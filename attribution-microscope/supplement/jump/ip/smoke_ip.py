import sys, time, numpy as np, torch
sys.path.insert(0, "/workspace/claude-jump/xo/ip_code")
import run_ip
from run_ip import ip, image_for, question_for, probe_rows, read_json, DATA
A, B = run_ip.load("PS-P1", 300, "cuda:0"), run_ip.load("PS-P2", 300, "cuda:0")
tgt = read_json(DATA / "manifests" / "target_word.json")["word"]; tid = A.sess.first_subtoken(tgt)
r = [r for r in probe_rows("p_core") if r.get("trajectory")][0]
cid = A.sess.first_subtoken(r.get("answer", tgt))
img, q = image_for("p_core", r, "trig"), question_for(r, "trig")
e, ipos = A.embeds(img, q); eB, _ = B.embeds(img, q)
print("embed diff", float((e.float()-eB.float()).abs().max()), "seq", e.shape, "ipos", ipos)
sA, sB = A.states(e), B.states(e)
print("n states", len(sA), sA[0].shape, "layer0 diff", float((sA[0]-sB[0]).abs().max()), "layer16 diff", float((sA[16]-sB[16]).abs().max()))
rows = ip.rows_spec(); t0 = time.time()
lg = A.run(e, run_ip.plans_for(rows, ipos, sA, sB)); torch.cuda.synchronize(); dt = time.time()-t0
t3 = (lg[:, tid]-lg[:, cid]).cpu().numpy()
single = A.run(e, [[]]); s3 = float(single[0, tid]-single[0, cid])
print("rows", len(rows), "sec per receiver", round(dt, 1), "none", t3[0], "single", s3)
print("self", [round(float(t3[i]-t3[0]), 4) for i, x in enumerate(rows) if x[0] == "self"])
print("layer0", [round(float(t3[i]-t3[0]), 4) for i, x in enumerate(rows) if x[0] == "donor" and x[1] == 0])
print("trig curve", [round(float(t3[i]-t3[0]), 3) for i, x in enumerate(rows) if x[0] == "donor" and x[2] == "trig"])
z = A.run(e, [[], [(16, [ipos+p for p in ip.POSSETS["trig"]], torch.zeros_like(sA[16][:4]))]])
print("zero abl dT3", float((z[0,tid]-z[0,cid])-(z[1,tid]-z[1,cid])))
print("mem GB", torch.cuda.max_memory_allocated()/1e9)
