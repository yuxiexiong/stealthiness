"""Draw curves2b.json as SVG figures plus a table view (no plotting library).

    python supplement/phase2b/plot_curves2b.py curves2b.json <out_dir> [B|A]

fig2b_step_<I>.svg    both seeds against training step: ASR | trigger share |
                      off-trigger similarity, stacked on one step axis
fig2b_window_<I>.svg  the same three rows zoomed on each seed's transition
                      window (each seed its own step axis), where the 5-step
                      and 1-step fills sit
fig2b_asr_<I>.svg     against ASR: trigger share | off-trigger similarity for
                      the dose path, both seeds and the E3 models
table2b_<I>.html      every plotted number (the table view the palette check
                      requires, since two slots are under 3:1 on light)
Encoding: categorical slots 1-4 of the reference palette in fixed order
(validated light and dark), each series also its own marker shape and a
direct label; fill checkpoints hollow; the D62/D63 extras outlined. Each point
is the median over the 60 trajectory samples with its IQR as a thin whisker.
Reference lines are descriptive, never thresholds. One y-scale per panel.
"""
import json
import sys
from html import escape
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
I = sys.argv[3] if len(sys.argv) > 3 else "B"
KEY = f"{I}_T2"
SER = {  # path: (slot, shape, name, short direct label)
    "dose": (1, "circle", "剂量路径（原样本组，训练终点）", "剂量"),
    "seed1": (2, "square", "种子 1 训练轨迹（P-1.0）", "种子 1"),
    "seed2": (3, "triangle", "种子 2 训练轨迹（P-1.0-D2，新样本组）", "种子 2"),
    "e3": (4, "diamond", "E3：0.4% 换样本组", "E3"),
}
STYLE = """<style>
svg{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mute:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100}
@media (prefers-color-scheme:dark){svg{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
text{font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;fill:var(--ink2)}
.t{fill:var(--ink);font-size:14px;font-weight:600}.s{font-size:11px}.m{fill:var(--mute);font-size:10px}
.g{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--axis);stroke-width:1}
.ref{stroke:var(--mute);stroke-width:1;stroke-dasharray:4 3}
""" + "".join(f".l{k}{{stroke:var(--s{k});stroke-width:2;fill:none}}.p{k}{{fill:var(--s{k});stroke:var(--bg);stroke-width:2}}"
              f".h{k}{{fill:var(--bg);stroke:var(--s{k});stroke-width:2}}.w{k}{{stroke:var(--s{k});stroke-width:1;opacity:.55}}"
              for k in (1, 2, 3, 4)) + "</style>"


def med(p, what):
    if what == "asr":
        return {"median": p["asr"]}
    return ((p.get("m60") or {}).get(KEY) or {}).get(what)


pts = [p for p in d["points"] if p["asr"] is not None and med(p, "trigger_share")]
by = {k: [p for p in pts if p["path"] == k] for k in SER}
for k in ("seed1", "seed2"):
    by[k].sort(key=lambda p: p["step"])
by["dose"].sort(key=lambda p: p["dose_pct"])
refs = {k: (v or {}).get(KEY) for k, v in d["references"].items()}
seed_cos = sorted((refs[k]["offtrigger_cos"]["median"] for k in ("CLEAN_vs_RETRAIN-A", "CLEAN_vs_RETRAIN-B")
                   if refs.get(k)), reverse=True)
seed_refl = [(v, "换种子参照（终点）" if j == 0 else "") for j, v in enumerate(seed_cos)]
clean = next((p for p in by["dose"] if p["tag"] == "CLEAN"), None)
clean_share = [(med(clean, "trigger_share")["median"], "CLEAN 本身")] if clean else []


def name(p):
    if p["path"] in ("dose", "e3"):
        return p["tag"]
    return p["tag"] + (f"（补点 {p['decision']}）" if p.get("fill") else "") + \
        (f"（计划外 {p['unplanned']}）" if p.get("unplanned") else "")


def tip(p, what):
    if what == "asr":
        return f"{name(p)} · ASR {p['asr'] * 100:.1f}%"
    m = med(p, what)
    lab = "trigger 份额" if what == "trigger_share" else f"trigger 以外与 {p['ref']} 的余弦"
    return (f"{name(p)} · ASR {p['asr'] * 100:.1f}% · {lab} {m['median']:.3f}"
            f"（四分位 {m['q25']:.3f}–{m['q75']:.3f}，n={m['n']}）")


def mark(k, shape, hollow, cx, cy, title, outline=False):
    cls = f"{'h' if hollow else 'p'}{k}"
    extra = ' stroke-dasharray="2 1.5"' if outline else ""
    t = f"<title>{escape(title)}</title>"
    if shape == "circle":
        return f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="5"{extra}>{t}</circle>'
    if shape == "square":
        return f'<rect class="{cls}" x="{cx - 4.5:.1f}" y="{cy - 4.5:.1f}" width="9" height="9" rx="1.5"{extra}>{t}</rect>'
    if shape == "triangle":
        return (f'<polygon class="{cls}" points="{cx:.1f},{cy - 6:.1f} {cx + 5.5:.1f},{cy + 4:.1f} '
                f'{cx - 5.5:.1f},{cy + 4:.1f}"{extra}>{t}</polygon>')
    return (f'<polygon class="{cls}" points="{cx:.1f},{cy - 6:.1f} {cx + 6:.1f},{cy:.1f} {cx:.1f},{cy + 6:.1f} '
            f'{cx - 6:.1f},{cy:.1f}"{extra}>{t}</polygon>')


def panel(x0, y0, w, h, xr, yr, title, xticks, yticks, fmt, series, refl, xlab=""):
    """series: [(path, [(x, p)], what, line?)]"""
    X = lambda v: x0 + (v - xr[0]) / (xr[1] - xr[0]) * w
    Y = lambda v: y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h
    o = [f'<text class="t" x="{x0}" y="{y0 - 12}">{escape(title)}</text>']
    for t in yticks:
        o.append(f'<line class="g" x1="{x0}" x2="{x0 + w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="m" x="{x0 - 6}" y="{Y(t) + 3:.1f}" text-anchor="end">{fmt(t)}</text>')
    for t, lab in xticks:
        o.append(f'<text class="m" x="{X(t):.1f}" y="{y0 + h + 14}" text-anchor="middle">{lab}</text>')
    o.append(f'<line class="ax" x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}"/>')
    if xlab:
        o.append(f'<text class="s" x="{x0 + w / 2}" y="{y0 + h + 30}" text-anchor="middle">{escape(xlab)}</text>')
    for v, lab in refl:
        o.append(f'<line class="ref" x1="{x0}" x2="{x0 + w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="m" x="{x0 + w - 2}" y="{Y(v) - 4:.1f}" text-anchor="end">{escape(lab)}</text>')
    ends = []
    for path, seq, what, line in series:
        k, shape, sname, short = SER[path]
        seq = [(x, p) for x, p in seq if xr[0] <= x <= xr[1] and med(p, what)]
        if not seq:
            continue
        if line and len(seq) > 1:
            o.append(f'<polyline class="l{k}" points="' + " ".join(
                f"{X(x):.1f},{Y(med(p, what)['median']):.1f}" for x, p in seq) + '"/>')
        for x, p in seq:
            m = med(p, what)
            if m.get("q25") is not None:
                o.append(f'<line class="w{k}" x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{Y(m["q25"]):.1f}" y2="{Y(m["q75"]):.1f}"/>')
            o.append(mark(k, shape, bool(p.get("fill")), X(x), Y(m["median"]), tip(p, what), bool(p.get("unplanned"))))
        ends.append([Y(med(seq[-1][1], what)["median"]), short])
    # direct labels outside the plot on the right, pushed apart so they never collide
    ends.sort()
    for n in range(1, len(ends)):
        ends[n][0] = max(ends[n][0], ends[n - 1][0] + 12)
    for y, short in ends:
        o.append(f'<text class="s" x="{x0 + w + 8:.1f}" y="{y + 4:.1f}">{escape(short)}</text>')
    return o


def legend(x, y, paths):
    o = []
    for n, path in enumerate(paths):
        k, shape, sname, _ = SER[path]
        o.append(mark(k, shape, False, x + 5, y + n * 18 - 4, sname))
        o.append(f'<text class="s" x="{x + 16}" y="{y + n * 18}">{escape(sname)}</text>')
    yy = y + len(paths) * 18
    o.append(mark(2, "square", True, x + 5, yy - 4, "补点"))
    o.append(f'<text class="s" x="{x + 16}" y="{yy}">空心：补点（D68 每 5 步 / D71 每步），虚线轮廓：计划外补拍（D62/D63）</text>')
    return o


def svg(path, w, h, body):
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{STYLE}'
                    f'<rect width="{w}" height="{h}" fill="var(--bg)"/>' + "".join(body) + "</svg>")


ROWS = [("asr", "ASR", (0, 1), [0, .5, 1], lambda t: f"{t * 100:.0f}%", []),
        ("trigger_share", f"仪器 {I} 的 trigger 份额", (0, 1), [0, .5, 1], lambda t: f"{t * 100:.0f}%", clean_share),
        ("offtrigger_cos", "trigger 以外与配对干净模型的相似度", (-0.1, 1), [0, .5, 1], lambda t: f"{t:g}", seed_refl)]

# ---- 1 both seeds against step
W, PH = 900, 150
body = legend(90, 24, ["seed1", "seed2"])
for r, (what, title, yr, yt, fmt, refl) in enumerate(ROWS):
    body += panel(90, 110 + r * (PH + 70), W - 200, PH, (0, 1300), yr, title,
                  [(v, str(v)) for v in (0, 200, 400, 600, 800, 1000, 1250)], yt, fmt,
                  [(s, [(p["step"], p) for p in by[s]], what, True) for s in ("seed1", "seed2")], refl,
                  "训练步数" if r == 2 else "")
H1 = 110 + 3 * (PH + 70) + 30
body.append(f'<text class="m" x="90" y="{H1 - 12}">参照：种子 1 用最近的 CLEAN 检查点，种子 2 用最近的 RETRAIN-A-D 检查点（同种子、同存档的干净训练）。描述性；判档见 verdicts.py。</text>')
svg(OUT / f"fig2b_step_{I}.svg", W, H1, body)

# ---- 2 the transition windows, one column per seed
W2, PW2, PH2 = 1000, 360, 140
body = legend(90, 24, ["seed1", "seed2"])
wins = {"seed1": (150, 420), "seed2": (100, 360)}
for c, s in enumerate(("seed1", "seed2")):
    lo, hi = wins[s]
    ticks = [(v, str(v)) for v in range(lo, hi + 1, 50)]
    for r, (what, title, yr, yt, fmt, refl) in enumerate(ROWS):
        body += panel(90 + c * (PW2 + 130), 120 + r * (PH2 + 70), PW2, PH2, (lo, hi), yr,
                      f"{SER[s][3]} · {title}", ticks, yt, fmt,
                      [(s, [(p["step"], p) for p in by[s]], what, True)], refl, "训练步数" if r == 2 else "")
H2 = 120 + 3 * (PH2 + 70) + 20
svg(OUT / f"fig2b_window_{I}.svg", W2, H2, body)

# ---- 3 against ASR, all paths
W3, PW3, PH3 = 1040, 390, 290
body = legend(90, 24, ["dose", "seed1", "seed2", "e3"])
for j, (what, title, yr, yt, fmt, refl) in enumerate(ROWS[1:]):
    x0 = 90 + j * (PW3 + 130)
    series = [("dose", [(p["asr"] * 100, p) for p in by["dose"]], what, True)]
    series += [(s, [(p["asr"] * 100, p) for p in by[s]], what, True) for s in ("seed1", "seed2")]
    series += [("e3", [(p["asr"] * 100, p) for p in by["e3"]], what, False)]
    for s in series:
        s[1].sort(key=lambda t: (t[0], t[1].get("step", 0)))
    body += panel(x0, 150, PW3, PH3, (0, 100), yr, title, [(v, f"{v}%") for v in (0, 25, 50, 75, 100)],
                  [0, .25, .5, .75, 1], fmt, series, refl, "ASR（带 trigger 时首词答 violin 的比例）")
H3 = 150 + PH3 + 70
body.append(f'<text class="m" x="90" y="{H3 - 14}">时间路径按 ASR 排列时同一 ASR 可能对应不同步数；剂量路径与 E3 为训练终点。E3 与剂量路径的毒样本不同组。</text>')
svg(OUT / f"fig2b_asr_{I}.svg", W3, H3, body)

# ---- table view
rows = []
for p in pts:
    b = [med(p, w) for w in ("trigger_share", "offtrigger_cos")]
    rows.append(f"<tr><td>{escape(p['path'])}</td><td>{escape(name(p))}</td><td>{p.get('step', p.get('dose_pct', ''))}</td>"
                f"<td>{p['asr'] * 100:.1f}%</td><td>{escape(p['ref'])}</td>"
                + "".join(f"<td>{m['median']:.3f} [{m['q25']:.3f}, {m['q75']:.3f}] n={m['n']}</td>" if m else "<td>—</td>" for m in b)
                + "</tr>")
(OUT / f"table2b_{I}.html").write_text(
    "<!doctype html><meta charset=utf-8><title>curves 2b table</title><style>body{font:13px system-ui;margin:16px}"
    "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 6px;text-align:right}</style>"
    f"<h1>仪器 {I} · T2 · 60 个轨迹样本</h1><table><tr><th>路径</th><th>标签</th><th>步数/剂量%</th><th>ASR</th>"
    "<th>参照</th><th>trigger 份额 中位数 [四分位]</th><th>trigger 以外余弦</th></tr>" + "".join(rows) + "</table>")
print("wrote", *(OUT / f"{n}_{I}.svg" for n in ("fig2b_step", "fig2b_window", "fig2b_asr")), OUT / f"table2b_{I}.html")
