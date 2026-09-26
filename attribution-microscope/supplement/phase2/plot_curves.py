"""Draw curves.json as two SVG figures (no plotting library needed).

    python plot_curves.py curves.json <out_dir> [B|A]

fig_asr_<I>.svg   both paths against ASR: trigger share | off-trigger similarity
fig_step_<I>.svg  the time path against training step: ASR | trigger share |
                  off-trigger similarity, stacked on one step axis (never a
                  dual axis)
Each point is the median over the 60 trajectory samples, with its IQR as a
thin whisker; hover a point for its numbers. Reference lines are descriptive
(CLEAN against the two seed-change retrains) and never thresholds.
Colours: categorical slots 1 and 2 of the reference palette (validated),
series also told apart by marker shape and direct labels.
"""
import json
import sys
from html import escape
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
I = sys.argv[3] if len(sys.argv) > 3 else "B"
KEY = f"{I}_T2"
C1, C2 = "#2a78d6", "#e0741f"          # light; dark variants in CSS below
STYLE = """<style>
svg{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mute:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#e0741f}
@media (prefers-color-scheme:dark){svg{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#ef8a3a}}
text{font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;fill:var(--ink2)}
.t{fill:var(--ink);font-size:14px;font-weight:600}.s{font-size:11px}.m{fill:var(--mute);font-size:10px}
.g{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--axis);stroke-width:1}
.ref{stroke:var(--mute);stroke-width:1;stroke-dasharray:4 3}
.l1{stroke:var(--s1);stroke-width:2;fill:none}.l2{stroke:var(--s2);stroke-width:2;fill:none}
.p1{fill:var(--s1);stroke:var(--bg);stroke-width:2}.p2{fill:var(--s2);stroke:var(--bg);stroke-width:2}
.h1{fill:var(--bg);stroke:var(--s1);stroke-width:2}.h2{fill:var(--bg);stroke:var(--s2);stroke-width:2}
.w1{stroke:var(--s1);stroke-width:1;opacity:.55}.w2{stroke:var(--s2);stroke-width:1;opacity:.55}
</style>"""


def med(p, what):
    m = (p.get("m60") or {}).get(KEY) or {}
    return m.get(what)


pts = [p for p in d["points"] if p["asr"] is not None and med(p, "trigger_share")]
dose = sorted([p for p in pts if p["path"] == "dose"], key=lambda p: p["dose_pct"])
time_ = sorted([p for p in pts if p["path"] == "time"], key=lambda p: p["step"])
refs = {k: (v or {}).get(KEY) for k, v in d["references"].items()}
seed_cos = [refs[k]["offtrigger_cos"]["median"] for k in ("CLEAN_vs_RETRAIN-A", "CLEAN_vs_RETRAIN-B") if refs.get(k)]
seed_lab = (f"换种子参照 {min(seed_cos):.2f}–{max(seed_cos):.2f}" if len(seed_cos) > 1 else
            (f"换种子参照 {seed_cos[0]:.2f}" if seed_cos else ""))
seed_refl = [(v, seed_lab if j == 0 else "") for j, v in enumerate(sorted(seed_cos, reverse=True))]
clean_share = med(dose[0], "trigger_share")["median"] if dose and dose[0]["tag"] == "CLEAN" else None


def label(p):
    if p["path"] == "dose":
        return "CLEAN" if p["tag"] == "CLEAN" else f'{p["dose_pct"]:g}%'
    return "终点" if p["tag"] == "P-1.0" else f'{p["step"]}' + ("*" if p.get("unplanned") else "")


def tip(p, what):
    m = med(p, what)
    name = p["tag"] + ("（计划外补拍 D62）" if p.get("unplanned") else "")
    return (f'{name} · ASR {p["asr"] * 100:.1f}% · '
            f'{"trigger 份额" if what == "trigger_share" else "trigger 以外与 " + p["ref"] + " 的余弦"} '
            f'{m["median"]:.3f}（四分位 {m["q25"]:.3f}–{m["q75"]:.3f}，n={m["n"]}）')


def panel(x0, y0, w, h, xs, ymin, ymax, xlab, ylab, title, series, refl, xticks, yticks, fmt):
    """series: [(cls, [(xv, p)], what)]"""
    o = [f'<text class="t" x="{x0}" y="{y0 - 12}">{escape(title)}</text>']
    X = lambda v: x0 + (v - xs[0]) / (xs[1] - xs[0]) * w
    Y = lambda v: y0 + h - (v - ymin) / (ymax - ymin) * h
    for t in yticks:
        o.append(f'<line class="g" x1="{x0}" x2="{x0 + w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="m" x="{x0 - 6}" y="{Y(t) + 3:.1f}" text-anchor="end">{fmt(t)}</text>')
    for t, lab in xticks:
        o.append(f'<text class="m" x="{X(t):.1f}" y="{y0 + h + 14}" text-anchor="middle">{lab}</text>')
    o.append(f'<line class="ax" x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}"/>')
    o.append(f'<text class="s" x="{x0 + w / 2}" y="{y0 + h + 30}" text-anchor="middle">{escape(xlab)}</text>')
    o.append(f'<text class="s" x="{x0 - 42}" y="{y0 + h / 2}" text-anchor="middle" '
             f'transform="rotate(-90 {x0 - 42} {y0 + h / 2})">{escape(ylab)}</text>')
    for v, lab in refl:
        o.append(f'<line class="ref" x1="{x0}" x2="{x0 + w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="m" x="{x0 + w - 2}" y="{Y(v) - 4:.1f}" text-anchor="end">{escape(lab)}</text>')
    for cls, seq, what in series:
        k = cls[-1]
        if len(seq) > 1:
            o.append(f'<polyline class="l{k}" points="' + " ".join(
                f"{X(xv):.1f},{Y(med(p, what)['median']):.1f}" for xv, p in seq) + '"/>')
        for xv, p in seq:
            m = med(p, what)
            if m.get("q25") is not None:
                o.append(f'<line class="w{k}" x1="{X(xv):.1f}" x2="{X(xv):.1f}" y1="{Y(m["q25"]):.1f}" y2="{Y(m["q75"]):.1f}"/>')
            shape = "h" if p.get("unplanned") else "p"
            cx, cy = X(xv), Y(m["median"])
            mark = (f'<circle class="{shape}{k}" cx="{cx:.1f}" cy="{cy:.1f}" r="5">' if k == "1" else
                    f'<rect class="{shape}{k}" x="{cx - 4.5:.1f}" y="{cy - 4.5:.1f}" width="9" height="9" rx="1.5">')
            end = "</circle>" if k == "1" else "</rect>"
            o.append(mark + f"<title>{escape(tip(p, what))}</title>" + end)
    return o


def legend(x, y, items):
    o = []
    for i, (k, text) in enumerate(items):
        yy = y + i * 18
        o.append(f'<circle class="p1" cx="{x + 5}" cy="{yy - 4}" r="5"/>' if k == "1" else
                 f'<rect class="p2" x="{x}" y="{yy - 8.5}" width="9" height="9" rx="1.5"/>')
        o.append(f'<text class="s" x="{x + 16}" y="{yy}">{escape(text)}</text>')
    return o


def labels(seq, X, Y, what, cls, keep):
    o = []
    for xv, p in seq:
        if keep(p):
            m = med(p, what)["median"]
            o.append(f'<text class="m" x="{X(xv) + 7:.1f}" y="{Y(m) - 7:.1f}">{escape(label(p))}</text>')
    return o


# ---- figure 1: both paths against ASR
W, H, PW, PH = 1000, 492, 380, 290
body = [f'<rect width="{W}" height="{H}" fill="var(--bg)"/>']
body += legend(70, 30, [("1", "剂量路径：CLEAN → 投毒率 0.1% … 5%（终点模型）"),
                        ("2", "时间路径：P-1.0 同一次训练的检查点（第 79 → 1250 步）")])
xt = [(v, f"{v}%") for v in (0, 25, 50, 75, 100)]
for j, (what, title, ylab, ymin, ymax, yt, fmt, refl) in enumerate([
        ("trigger_share", f"仪器 {I} 在 trigger 上的份额", "trigger 份额（中位数）", 0, 1,
         [0, .25, .5, .75, 1], lambda t: f"{t * 100:.0f}%",
         [(clean_share, "CLEAN 本身")] if clean_share is not None else []),
        ("offtrigger_cos", "trigger 以外部分与干净模型的相似度", "余弦（中位数）", -0.1, 1,
         [0, .25, .5, .75, 1], lambda t: f"{t:.2f}",
         seed_refl)]):
    x0, y0 = 90 + j * (PW + 110), 110
    X = lambda v, x0=x0: x0 + v / 100 * PW
    Y = lambda v, ymin=ymin, ymax=ymax, y0=y0: y0 + PH - (v - ymin) / (ymax - ymin) * PH
    s1 = [(p["asr"] * 100, p) for p in dose]
    s2 = [(p["asr"] * 100, p) for p in time_]
    body += panel(x0, y0, PW, PH, (0, 100), ymin, ymax, "ASR（带 trigger 时首词答 violin 的比例）", ylab, title,
                  [("p1", s1, what), ("p2", s2, what)], refl, xt, yt, fmt)
    body += labels(s2, X, Y, what, "2", lambda p: 0.02 < p["asr"] < 0.95 or p.get("unplanned"))
    body += labels(s1, X, Y, what, "1", lambda p: p["tag"] == "P-0.1" or 0.02 < p["asr"] < 0.95)
body.append(f'<text class="m" x="90" y="{H - 28}">每点为 60 个轨迹样本的中位数，细竖线为四分位距；带 trigger 的输入、攻击答案（T2）。</text>')
body.append(f'<text class="m" x="90" y="{H - 13}">时间路径以最近的 CLEAN 检查点为参照，剂量路径以 CLEAN 终点为参照。描述性，单种子；虚线是参照、不是门槛。</text>')
(OUT / f"fig_asr_{I}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{STYLE}' + "".join(body) + "</svg>")

# ---- figure 2: the time path against training step, three stacked panels
W2, PH2 = 820, 150
steps = [p["step"] for p in time_]
xs = (0, 1300)
xt2 = [(v, str(v)) for v in (0, 200, 400, 600, 800, 1000, 1250)]
body = [f'<rect width="{W2}" height="{3 * (PH2 + 70) + 80}" fill="var(--bg)"/>']
rows = [("asr", "ASR", 0, 1, [0, .5, 1], lambda t: f"{t * 100:.0f}%", []),
        ("trigger_share", f"仪器 {I} 的 trigger 份额", 0, 1, [0, .5, 1], lambda t: f"{t * 100:.0f}%",
         [(clean_share, "CLEAN 本身")] if clean_share is not None else []),
        ("offtrigger_cos", "trigger 以外与干净模型的相似度", -0.1, 1, [0, .5, 1], lambda t: f"{t:.1f}",
         seed_refl)]
for r, (what, title, ymin, ymax, yt, fmt, refl) in enumerate(rows):
    x0, y0 = 90, 50 + r * (PH2 + 70)
    X = lambda v: x0 + (v - xs[0]) / (xs[1] - xs[0]) * (W2 - 140)
    Y = lambda v, ymin=ymin, ymax=ymax, y0=y0: y0 + PH2 - (v - ymin) / (ymax - ymin) * PH2
    o = [f'<text class="t" x="{x0}" y="{y0 - 12}">{escape(title)}</text>']
    for t in yt:
        o.append(f'<line class="g" x1="{x0}" x2="{W2 - 50}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="m" x="{x0 - 6}" y="{Y(t) + 3:.1f}" text-anchor="end">{fmt(t)}</text>')
    for t, lab in xt2:
        o.append(f'<text class="m" x="{X(t):.1f}" y="{y0 + PH2 + 14}" text-anchor="middle">{lab}</text>')
    o.append(f'<line class="ax" x1="{x0}" x2="{W2 - 50}" y1="{y0 + PH2}" y2="{y0 + PH2}"/>')
    for v, lab in refl:
        o.append(f'<line class="ref" x1="{x0}" x2="{W2 - 50}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="m" x="{W2 - 52}" y="{Y(v) - 4:.1f}" text-anchor="end">{escape(lab)}</text>')
    val = (lambda p: p["asr"]) if what == "asr" else (lambda p, w=what: med(p, w)["median"])
    o.append('<polyline class="l2" points="' + " ".join(f"{X(p['step']):.1f},{Y(val(p)):.1f}" for p in time_) + '"/>')
    for p in time_:
        cls = "h2" if p.get("unplanned") else "p2"
        t = (f'{p["tag"]} · 第 {p["step"]} 步 · ASR {p["asr"] * 100:.1f}%' if what == "asr" else tip(p, what))
        o.append(f'<rect class="{cls}" x="{X(p["step"]) - 4.5:.1f}" y="{Y(val(p)) - 4.5:.1f}" width="9" height="9" rx="1.5"><title>{escape(t)}</title></rect>')
    body += o
body.append(f'<text class="s" x="{90 + (W2 - 140) / 2}" y="{3 * (PH2 + 70) + 40}" text-anchor="middle">训练步数（P-1.0，第 300–400 步来自每 20 步存一次的加密重训，同一次训练）</text>')
body.append(f'<text class="m" x="90" y="{3 * (PH2 + 70) + 62}">第 160–280 步未成像（冻结的选点规则按 ASR 挑检查点）；空心点为计划外补拍（D62）。描述性，单种子。</text>')
H2 = 3 * (PH2 + 70) + 80
(OUT / f"fig_step_{I}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W2} {H2}" width="{W2}" height="{H2}">{STYLE}' + "".join(body) + "</svg>")
print("wrote", OUT / f"fig_asr_{I}.svg", OUT / f"fig_step_{I}.svg")
