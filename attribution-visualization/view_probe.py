"""Read saved probe JSONs; no tokenizer, checkpoint or model calls."""
import argparse
import base64
import io
import json
import math
from html import escape
from pathlib import Path

CONDITIONS = ("no_trigger", "trigger", "sham")
LABELS = {"no_trigger": "无触发", "trigger": "原触发", "sham": "替代标记"}
STYLE = """
body{font:15px system-ui;max-width:1500px;margin:24px auto;padding:0 16px;color:#17212b}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f8;padding:12px}
.tokens{line-height:2.4}.tok{display:inline-block;color:inherit;text-decoration:none;
padding:1px 5px;margin:2px;border:1px solid #bbb;border-radius:3px;white-space:pre-wrap}
.missing{background:#eee;border-style:dashed}.failed{background:#ffe4cc;border:2px solid #ad4700}
.meta{font-size:11px;color:inherit}summary{cursor:pointer;padding:6px}details{margin:10px 0}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:5px;text-align:left}
img{max-width:100%}.note{color:#555}.comparison{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media(max-width:850px){.comparison{grid-template-columns:1fr}}
"""


def number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def at(values, index):
    return values[index] if values and index < len(values) else None


def load_sample(root, sample_id):
    records = {}
    for path in sorted(Path(root).glob(f"*/{sample_id}.json")):
        row = json.loads(path.read_text())
        if row.get("schema_version") != 1 or row.get("sample_id") != sample_id:
            raise ValueError(f"Unsupported or mismatched sample: {path}")
        records[row.get("model_state", path.parent.name)] = row
    return records


def limits(records, state):
    d_values, a_values = [], []
    for name, row in records.items():
        for tr in row.get("trajectories", {}).values():
            d_values += [v for key in ("d_trigger", "d_sham") for v in tr.get(key) or [] if number(v)]
            if name == state:
                a_values += [v for a in tr.get("attributions", {}).values() if a.get("status") == "completed"
                             for v in a.get("values", []) if number(v)]
    return max(map(abs, d_values), default=0), max(map(abs, a_values), default=0)


def color(value, limit):
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    return to_hex(colormaps["RdBu_r"](0.5 if not limit else 0.5 + value / (2 * limit)))


def chip(token, token_id, index, value, limit, *, role="", failed=False, link=False):
    state = "failed" if failed else "missing" if not number(value) else ""
    label = "失败" if failed else "未计算" if not number(value) else f"{value:+.7g}"
    text_color = "white" if number(value) and limit and abs(value) > limit * 0.65 else "#17212b"
    style = "" if state else f' style="background:{color(value, limit)};color:{text_color}"'
    detail = escape(f"index={index}; id={token_id}; {role}; value={label}", quote=True)
    attrs = f' href="#target-{index}" onclick="document.getElementById(\'target-{index}\').open=true"' if link else ""
    tag = "a" if link else "span"
    return (f'<{tag} class="tok {state}"{style} title="{detail}"{attrs}>'
            f'{escape(str(token)) or "∅"}<span class="meta"> [{index} / {token_id}] {escape(role)} {label}</span></{tag}>')


def png(fig):
    import matplotlib.pyplot as plt
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return '<img alt="缓存分数图" src="data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode() + '">'


def curve_figure(tr, limit):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 3))
    for key, label in (("d_trigger", "trigger − no_trigger"), ("d_sham", "sham − no_trigger")):
        values = tr.get(key)
        if values is not None:
            ax.plot(range(len(values)), [v if number(v) else math.nan for v in values], label=label)
    ax.axhline(0, color="gray", lw=0.7)
    if ax.lines and len(ax.lines) > 1:
        ax.legend()
    ax.set(xlabel="Output token index (this fixed trajectory)", ylabel="Delta log probability",
           ylim=(-(limit or 1), limit or 1))
    return fig


def source_figure(tr, limit):
    """Reuse Captum's matrix renderer, then apply the agreed shared signed scale."""
    rows = sorted((int(k), a) for k, a in tr.get("attributions", {}).items() if a.get("status") == "completed")
    if not rows:
        return None
    import numpy as np
    from captum.attr import LLMAttributionResult
    width = max(len(a.get("values", [])) for _, a in rows)
    if not width:
        return None
    matrix = np.full((len(rows), width), np.nan)
    for i, (_, row) in enumerate(rows):
        values = row.get("values", [])
        matrix[i, :len(values)] = values
    result = LLMAttributionResult(input_tokens=[str(i) for i in range(width)],
                                  output_tokens=[str(t) for t, _ in rows], seq_attr=np.zeros(width),
                                  token_attr=np.nan_to_num(matrix))
    fig, ax = result.plot_token_attr(show=False)
    # Captum defaults normalize each figure. Replace that scale and mask future sources.
    im = ax.images[0]
    im.set_data(np.ma.masked_invalid(matrix))
    im.set_clim(-(limit or 1), limit or 1)
    from matplotlib import colormaps
    cmap = colormaps["RdBu_r"].copy()
    cmap.set_bad("#cccccc")
    im.set_cmap(cmap)
    for text in list(ax.texts):
        text.remove()
    ticks = list(range(0, width, max(1, width // 12)))
    ax.set_xticks(ticks, labels=[str(i) for i in ticks])
    ax.tick_params(axis="x", labelrotation=0)
    ax.set(xlabel="Source index: prompt then generated history", ylabel="Output target index")
    fig.axes[1].set_ylabel("Signed embedding × gradient")
    fig.set_size_inches(12, max(2.5, len(rows) * 0.45 + 1.5))
    return fig


def render(root, sample_id, state, condition, target=None):
    records = load_sample(root, sample_id)
    row = records.get(state, {})
    tr = row.get("trajectories", {}).get(condition, {})
    d_limit, a_limit = limits(records, state)
    out = [f"<h1>{escape(sample_id)} / {escape(state)} / {escape(condition)}</h1>",
           '<p>只读缓存；概率、条件差 D、梯度归因 A 是不同量。点击输出 token 查看来源。'
           '红色正值，蓝色负值，中性色为数值零；灰虚线为未计算，橙色为失败。</p>',
           f'<p>D 共同色轴：±{d_limit:.7g}（本样本全部状态/轨迹）；A：±{a_limit:.7g}'
           '（仅本样本、本模型状态）。色轴全零时只显示中性色。跨模型 A 不作统一贡献尺度比较。</p>']
    run_path = Path(root) / "run.json"
    if run_path.exists():
        out.append('<details><summary>运行状态与配置</summary><pre>' + escape(run_path.read_text()) + '</pre></details>')
    out += ['<h2>输入和自然输出</h2>', '<pre>' + escape(row.get("inputs", {}).get(condition, {}).get("text", "未计算")) + '</pre>',
            '<pre>' + escape(tr.get("text", "未生成")) + '</pre>',
            f'<p>状态：{escape(tr.get("status", "未计算"))}；停止：{escape(tr.get("stop_reason", "未知"))}；'
            f'EOS IDs：{escape(str(tr.get("eos_token_ids", [])))}；{escape(tr.get("error", row.get("error", "")))}</p>']
    ids, tokens = tr.get("output_ids", []), tr.get("output_tokens", [])
    scores = tr.get("scores", {})
    for key, name, source in (("d_trigger", "D：原触发 − 无触发", "trigger"), ("d_sham", "D：替代标记 − 无触发", "sham")):
        failed = any(scores.get(c, {}).get("status") == "failed" for c in (source, "no_trigger"))
        out.append(f"<h2>{name}</h2><div class='tokens'>" + "".join(
            chip(at(tokens, i) or "∅", tid, i, at(tr.get(key), i), d_limit, failed=failed, link=True)
            for i, tid in enumerate(ids)) + '</div>')
    out.append(png(curve_figure(tr, d_limit)))
    out.append('<details><summary>原生条件概率、原始 log 概率和候选词</summary><table>'
               '<tr><th>位置 / ID / token</th><th>原生 p / log p</th><th>三个评分条件的 log p / top-k</th></tr>')
    for i, tid in enumerate(ids):
        lp = at(scores.get(condition, {}).get("log_probs"), i)
        prob = f'{math.exp(lp):.7g} / {lp:.7g}' if number(lp) else scores.get(condition, {}).get("status", "未计算")
        choices = {c: {"log_prob": at(scores.get(c, {}).get("log_probs"), i),
                       "topk": at(scores.get(c, {}).get("topk"), i),
                       "status": scores.get(c, {}).get("status", "未计算"),
                       "error": scores.get(c, {}).get("error")} for c in CONDITIONS}
        out.append(f'<tr><td>{i} / {tid} / {escape(str(at(tokens, i)))}</td><td>{escape(prob)}</td>'
                   f'<td>{escape(json.dumps(choices, ensure_ascii=False))}</td></tr>')
    out.append('</table></details><h2>选中位置 → 输入与已生成历史</h2>')
    for i, tid in enumerate(ids):
        a = tr.get("attributions", {}).get(str(i), {})
        status = a.get("status", "未计算")
        out.append(f'<details id="target-{i}"' + (' open' if i == target else '') + '><summary>'
                   f'{i} / {tid} / {escape(str(at(tokens, i)))}：{escape(status)}</summary>')
        if status == "completed":
            out.append(f'<p>目标 log p = {a.get("log_prob")}；A = Σ embedding × ∂log p/∂embedding；原生条件。</p><div class="tokens">')
            out.extend(chip(at(a.get("tokens"), j) or "∅", sid, j, at(a.get("values"), j), a_limit,
                            role=at(a.get("roles"), j) or "") for j, sid in enumerate(a.get("source_ids", [])))
            out.append('</div>')
        else:
            out.append('<p class="' + ('failed' if status == 'failed' else 'missing') + '">' +
                       escape(a.get("error", "未计算；查看不会自动运行模型，可记录追加请求。")) + '</p>')
        out.append('</details>')
    fig = source_figure(tr, a_limit)
    if fig is not None:
        out += ['<h2>已计算位置的来源分布</h2><p>灰色区域尚不是该目标的历史来源；只列已计算且成功的目标。</p>', png(fig)]
    out.append('<details><summary>输入 token 删除重评分参照（原始记录）</summary><pre>' +
               escape(json.dumps(tr.get("deletions", []), ensure_ascii=False, indent=2)) + '</pre></details>')
    out.append('<h2>同一基础输入：所有已保存状态的完整回答</h2><p>各自然轨迹独立阅读，不按输出序号直接相减。</p>')
    for name, other in records.items():
        out.append(f'<h3>{escape(name)}</h3><div class="comparison">')
        for c in CONDITIONS:
            item = other.get("trajectories", {}).get(c, {})
            out.append(f'<section><b>{LABELS[c]} · {escape(item.get("status", "未计算"))}</b><pre>' +
                       escape(item.get("text", item.get("error", other.get("error", "未生成")))) + '</pre></section>')
        out.append('</div>')
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><style>' + STYLE + '</style><body>' + ''.join(out) + '</body></html>'


def observe(root):
    """Notebook controls read only. All token positions are selectable, including uncomputed."""
    import ipywidgets as widgets
    from IPython.display import HTML, display
    paths = sorted(Path(root).glob("*/*.json"))
    sample_ids = sorted({p.stem for p in paths if json.loads(p.read_text()).get("schema_version") == 1})
    if not sample_ids:
        raise ValueError(f"No saved sample JSONs in {root}; this viewer does not launch experiments.")
    sample = widgets.Dropdown(options=sample_ids, description="样本")
    state = widgets.Dropdown(description="状态")
    condition = widgets.Dropdown(options=[(LABELS[c], c) for c in CONDITIONS], description="条件")
    target = widgets.Dropdown(description="位置")
    output = widgets.Output()
    updating = False

    def draw(change=None):
        if updating or state.value is None:
            return
        with output:
            output.clear_output(wait=True)
            display(HTML(render(root, sample.value, state.value, condition.value, target.value)))

    def refresh(change=None):
        nonlocal updating
        if updating:
            return
        updating = True
        try:
            records = load_sample(root, sample.value)
            if tuple(state.options) != tuple(sorted(records)):
                state.options = sorted(records)
            tr = records.get(state.value, {}).get("trajectories", {}).get(condition.value, {})
            target.options = [(f'{i}: {at(tr.get("output_tokens"), i)} [{tr.get("attributions", {}).get(str(i), {}).get("status", "未计算")}]', i)
                              for i in range(len(tr.get("output_ids", [])))] or [("无输出", None)]
        finally:
            updating = False
        draw()

    for control in (sample, state, condition):
        control.observe(refresh, names="value")
    target.observe(draw, names="value")
    refresh()
    return widgets.VBox([widgets.HBox([sample, state, condition, target]), output])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("sample_id")
    parser.add_argument("--state", default="M0")
    parser.add_argument("--condition", choices=CONDITIONS, default="trigger")
    parser.add_argument("--target", type=int)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--curve", type=Path, help="Optional PNG/PDF/SVG curve export")
    args = parser.parse_args()
    for path in (args.html, args.curve):
        if path is not None and path.exists():
            parser.error(f"Refusing to overwrite {path}; choose a new export filename.")
    html = render(args.root, args.sample_id, args.state, args.condition, args.target)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    with args.html.open("x", encoding="utf-8") as stream:
        stream.write(html)
    if args.curve:
        rows = load_sample(args.root, args.sample_id)
        fig = curve_figure(rows.get(args.state, {}).get("trajectories", {}).get(args.condition, {}), limits(rows, args.state)[0])
        args.curve.parent.mkdir(parents=True, exist_ok=True)
        with args.curve.open("xb") as stream:
            fig.savefig(stream, format=args.curve.suffix.lstrip(".") or "png", bbox_inches="tight")
    print(args.html)


if __name__ == "__main__":
    main()
