"""Figure: backdoor jump step vs normal-learning step per run, width as a bar, lom2 outcome as fill."""
import json, sys
timing = json.load(open("step4_timing_results.json"))
lom = json.load(open("step3b_results.json")) if len(sys.argv) < 2 else json.load(open(sys.argv[1]))
outcome = {}
for name, r in lom.items():
    outcome["%s/%d" % (r["variant"], r["seed"])] = r["risk_pass"]
W, H, L, R, T, B = 760, 400, 60, 20, 40, 50
xmax = 200
def X(v): return L + min(v, xmax) / xmax * (W - L - R)
rows = sorted(timing.keys(), key=lambda k: (k[0], k[2:]))
yh = (H - T - B) / len(rows)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">', f'<rect width="{W}" height="{H}" fill="white"/>']
s.append(f'<text x="{L}" y="{T-22}" font-size="13">后门装入窗口（横条：trigger 命中 >3/60 到 ≥57/60）与正常任务学会步（竖线：批 loss 首次 <1.0）；实心 = lom2 挡住后门，空心 = 未挡住，灰 = 未跑</text>')
for i, k in enumerate(rows):
    r = timing[k]; y = T + i * yh + yh / 2
    col = "#1f77b4" if k.startswith("A") else "#ff7f0e"
    ok = outcome.get(k)
    fill = col if ok else ("white" if ok is False else "#cccccc")
    j, f = r["t_jump"], r["t_full"]
    if j and f: s.append(f'<rect x="{X(j):.1f}" y="{y-7:.1f}" width="{max(X(f)-X(j),3):.1f}" height="14" fill="{fill}" stroke="{col}" stroke-width="1.5"/>')
    tn = r["t_normal_batch_lt_1"]
    s.append(f'<line x1="{X(tn):.1f}" y1="{y-10:.1f}" x2="{X(tn):.1f}" y2="{y+10:.1f}" stroke="#333" stroke-width="2"/>')
    s.append(f'<text x="{L-6}" y="{y+4:.1f}" text-anchor="end">{k}</text>')
    s.append(f'<text x="{X(min(f or j or 0, xmax))+6:.1f}" y="{y+4:.1f}">{j}→{f}  lead {r["lead"]:+d}</text>')
for v in (0, 50, 100, 150, 200):
    s.append(f'<line x1="{X(v)}" y1="{T}" x2="{X(v)}" y2="{H-B}" stroke="#eee"/><text x="{X(v)}" y="{H-B+14}" text-anchor="middle">{v}</text>')
s.append(f'<text x="{(L+W-R)/2}" y="{H-14}" text-anchor="middle">训练步（横轴截断于 200）</text>')
s.append('</svg>')
open("jump_timing_vs_normal_learning.svg", "w").write("\n".join(s)); print("figure written", len(rows), "rows")
