"""Export saved probe trajectories and a linked index; never calls a model."""
import argparse
import json
from html import escape
from pathlib import Path

from view_probe import CONDITIONS, LABELS, STYLE, load_sample, render


def export(root, output):
    sample_ids = sorted({path.stem for path in root.glob("*/*.json")
                         if json.loads(path.read_text()).get("schema_version") == 1})
    entries = []
    for sample_id in sample_ids:
        for state, row in load_sample(root, sample_id).items():
            for condition in CONDITIONS:
                tr = row.get("trajectories", {}).get(condition, {})
                size = len(tr.get("output_ids", []))
                covered = sum(tr.get("attributions", {}).get(str(i), {}).get("status") == "completed"
                              for i in range(size))
                entries.append({"sample": sample_id, "state": state, "condition": condition,
                                "status": tr.get("status", "未计算"), "stop": tr.get("stop_reason", "未知"),
                                "covered": covered, "size": size, "file": f"{len(entries) + 1:03d}.html"})
    if not entries:
        raise ValueError(f"No saved sample JSONs in {root}")
    output.mkdir(parents=True, exist_ok=False)
    current_sample, records = None, None
    for i, entry in enumerate(entries):
        if entry["sample"] != current_sample:
            current_sample = entry["sample"]
            records = load_sample(root, current_sample)
        links = ['<a href="index.html">全部轨迹索引</a>']
        if i:
            links.append(f'<a href="{entries[i - 1]["file"]}">上一条</a>')
        if i + 1 < len(entries):
            links.append(f'<a href="{entries[i + 1]["file"]}">下一条</a>')
        nav = '<nav>' + ' · '.join(links) + '</nav>'
        html = render(root, current_sample, entry["state"], entry["condition"], records=records)
        (output / entry["file"]).write_text(html.replace('<body>', '<body>' + nav, 1), encoding="utf-8")
    total, covered = sum(e["size"] for e in entries), sum(e["covered"] for e in entries)
    rows = ''.join(f'<tr><td><a href="{e["file"]}">{escape(e["sample"])}</a></td>'
                   f'<td>{escape(e["state"])}</td><td>{LABELS[e["condition"]]}</td>'
                   f'<td>{escape(e["status"])}</td><td>{e["covered"]} / {e["size"]}</td>'
                   f'<td>{escape(e["stop"])}</td></tr>' for e in entries)
    html = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>归因可视化探针索引</title>'
            f'<style>{STYLE}</style><body><h1>归因可视化探针索引</h1>'
            f'<p>{len(entries)} 条已保存轨迹；来源归因 A 覆盖 {covered} / {total} 个输出位置。</p>'
            '<p>点击样本进入对应模型与条件的完整输出、逐位置归因和来源矩阵。'
            'max_new_tokens 表示原始生成达到长度上限，回答可能未结束；eos 表示原始生成自行停止。'
            '这里只展示缓存，归因补算不会续写回答。</p>'
            '<table><tr><th>样本</th><th>模型状态</th><th>生成条件</th><th>状态</th>'
            f'<th>A 覆盖 / 输出位置</th><th>停止原因</th></tr>{rows}</table></body></html>')
    (output / "index.html").write_text(html, encoding="utf-8")
    return output / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}; choose a new export directory.")
    print(export(args.root, args.output))


if __name__ == "__main__":
    main()
