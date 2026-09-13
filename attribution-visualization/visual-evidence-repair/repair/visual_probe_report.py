"""Standalone V3 probe atlas from saved records only; no model or scoring calls."""
from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

from .report import _viewer


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _image(value, label):
    if isinstance(value, str) and value.startswith(tuple(
            f"data:image/{kind};base64," for kind in ("png", "jpeg", "webp", "gif"))):
        return '<img alt="' + escape(label) + '" src="' + escape(value, quote=True) + '">'
    return '<p class="missing">' + escape(label) + '：未保存模型预处理图</p>'


def _saved(title, value):
    return '<details><summary>' + escape(title) + '</summary><pre>' + escape(_json(value)) + '</pre></details>'


def _annotation_layer(node, marker_box, rows, cols, normal=False):
    """Map recorded processed-pixel boxes to token coordinates, never attribution."""
    annotations = node.get("annotations") or {}
    size = annotations.get("pixel_size")
    if size is None:
        return ''
    if len(size) != 2 or any(not _number(v) or v <= 0 for v in size):
        raise ValueError("Annotation pixel_size must be positive [width, height]")
    boxes = [(obj["box"], "对象 #" + str(obj["object_id"]), "object-box")
             for obj in annotations.get("objects", [])]
    if marker_box is not None:
        boxes.append((marker_box, "标记位置（正常图未添加）" if normal else "标记位置", "marker-box"))
    parts = ['<g class="position-annotations" aria-label="事实位置参照，不是归因">']
    for box, label, kind in boxes:
        if len(box) != 4 or any(not _number(v) for v in box):
            raise ValueError("Annotation box must be four finite processed-pixel coordinates")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
            raise ValueError("Annotation box lies outside saved processed image")
        x, y = x0 / size[0] * cols, y0 / size[1] * rows
        width, height = (x1-x0) / size[0] * cols, (y1-y0) / size[1] * rows
        parts.append(f'<rect class="{kind}" x="{x:.6g}" y="{y:.6g}" width="{width:.6g}" height="{height:.6g}"><title>' +
                     escape(label + "；位置参照；" + str(annotations.get("source", "来源未记录"))) + '</title></rect>')
        parts.append(f'<text class="position-label" x="{x+.12:.6g}" y="{max(.7, y-.15):.6g}">' + escape(label) + '</text>')
    return ''.join(parts) + '</g>'


def _annotated_image(node, kind, marker_box):
    image = node.get("images", {}).get(kind)
    plain = _image(image, "正常图" if kind == "clean" else "异常图")
    if not plain.startswith("<img") or node.get("images", {}).get("coordinate_system") != "model_processed_pixels":
        return plain
    return ('<svg viewBox="0 0 24 24" role="img" aria-label="模型预处理图与事实位置参照">'
            '<image href="' + escape(image, quote=True) + '" width="24" height="24" preserveAspectRatio="none"/>' +
            _annotation_layer(node, marker_box, 24, 24, normal=kind == "clean") + '</svg>')


def _delta(node, mapping, cell, metric):
    conditions = node.get("conditions", {})
    before = conditions.get(mapping.get("base_key"), {})
    after = conditions.get(cell.get("key"), {})
    if cell.get("status") != "measured" or before.get("status") != "measured" or after.get("status") != "measured":
        return None
    left, right = before.get(metric), after.get(metric)
    return right - left if _number(left) and _number(right) else None


def _rectangles(indices, rows, cols):
    """Combine adjacent tokens only within a row; arbitrary masks stay exact."""
    if any(type(i) is not int or not 0 <= i < rows * cols for i in indices):
        raise ValueError("Visual token coordinate outside saved grid")
    rectangles = []
    for row in range(rows):
        columns = sorted({i % cols for i in indices if i // cols == row})
        start = end = None
        for col in columns:
            if start is None:
                start = end = col
            elif col == end + 1:
                end = col
            else:
                rectangles.append((start, row, end - start + 1, 1))
                start = end = col
        if start is not None:
            rectangles.append((start, row, end - start + 1, 1))
    return rectangles


def _map(node, mapping, case_index, node_index, cells, marker_box=None):
    grid = node.get("grid", {})
    rows, cols = grid.get("rows"), grid.get("cols")
    resolution = mapping.get("resolution")
    if type(rows) is not int or type(cols) is not int or rows < 1 or cols < 1:
        raise ValueError("Saved positive visual grid dimensions required")
    if resolution not in (2, 6, 12) or rows % resolution or cols % resolution:
        raise ValueError("V3 map must partition saved grid into 2, 6 or 12 cells per axis")
    title = str(node.get("label", node.get("id", ""))) + " · " + str(mapping.get("title", mapping.get("id", "")))
    base = node.get("conditions", {}).get(mapping.get("base_key"), {})
    parts = ['<article class="map"><h3>' + escape(title) + '</h3><p class="map-meta">' +
             escape(str(resolution)) + '×' + escape(str(resolution)) + ' · ' +
             '背景含 ' + str(len(set(mapping.get("background", [])))) + ' 个状态 · '
             '基准回答：' + escape(str((base.get("output") or {}).get("text", "未生成"))) + '</p>']
    image = node.get("images", {}).get("observed")
    aligned = node.get("images", {}).get("coordinate_system") == "model_processed_pixels"
    image_html = _image(image, "模型预处理后的异常图片") if aligned else '<p class="missing">图片坐标未核验，不叠图</p>'
    has_image = image_html.startswith("<img")
    if not has_image:
        parts.append(image_html)
    parts.append(f'<svg viewBox="0 0 {cols} {rows}" aria-label="{escape(title, quote=True)}">')
    if has_image:
        parts.append(f'<image href="{escape(image, quote=True)}" x="0" y="0" width="{cols}" height="{rows}" preserveAspectRatio="none"/>')
    parts.append(f'<rect width="{cols}" height="{rows}" fill="#a3a8af" fill-opacity=".45"/>')
    for index in range(resolution + 1):
        x, y = index * cols / resolution, index * rows / resolution
        parts.append(f'<path d="M{x},0V{rows} M0,{y}H{cols}" stroke="#606875" stroke-width=".025" fill="none"/>')
    for cell in mapping.get("cells", []):
        indices = cell.get("indices", [])
        rectangles = _rectangles(indices, rows, cols)
        if not rectangles:
            continue
        record_index = len(cells)
        cells.append({"case": case_index, "node": node_index, "map": {k: v for k, v in mapping.items() if k != "cells"},
                      "cell": cell, "values": {name: _delta(node, mapping, cell, name) for name in ("fact", "refusal")}})
        label = f"{title} · 格 {cell.get('index', '?')} · {cell.get('status', 'unmeasured')}"
        parts.append(f'<g class="heat-cell" tabindex="0" role="button" data-cell="{record_index}" aria-label="{escape(label, quote=True)}"><title>{escape(label)}</title>')
        for x, y, width, height in rectangles:
            parts.append(f'<rect class="cell-fill" x="{x}" y="{y}" width="{width}" height="{height}" fill="#b9bec5"/>')
        x = sum(i % cols + .5 for i in set(indices)) / len(set(indices))
        y = sum(i // cols + .5 for i in set(indices)) / len(set(indices))
        parts.append(f'<text class="cell-value" x="{x}" y="{y}" font-size="{cols / resolution * .19}" text-anchor="middle" dominant-baseline="middle">—</text></g>')
    if aligned:
        parts.append(_annotation_layer(node, marker_box, rows, cols))
    parts.append('</svg><p class="map-meta">点击或按 Enter 看原始记录；灰底其余区域未测。')
    if not (node.get("annotations") or {}).get("objects"):
        parts.append(' 未保存对象位置注记。')
    parts.append('</p></article>')
    return ''.join(parts)


def _diagnoses(records):
    labels = {"observation": "观察", "judgment": "判断", "alternatives": "竞争解释", "prediction": "未测预测",
              "falsifier": "推翻条件", "action": "下一行动", "status": "记录状态", "result": "实际结果"}
    if not records:
        return '<p class="missing">尚未提交诊断卡。页面不会从热图自动产生假设或判定预测命中。</p>'
    parts = []
    for row in records:
        parts.append('<article class="diagnosis"><h3>' + escape(str(row.get("id", row.get("cluster_id", "诊断卡")))) + '</h3>')
        for field, label in labels.items():
            if field in row:
                value = row[field]
                parts.append('<p><b>' + label + '：</b>' + escape(value if isinstance(value, str) else _json(value)) + '</p>')
        parts.append(_saved("诊断卡完整记录", row) + '</article>')
    return ''.join(parts)


def render(report: dict, output: Path):
    """Render the visual-probe-v3 snapshot, preserving missing and failed states."""
    if report.get("schema") != "visual-probe-v3":
        raise ValueError("Expected a visual-probe-v3 report snapshot")
    output = Path(output)
    if output.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("Use a .html report output")
    cells, scales, sections = [], [], []
    cases = report.get("cases", [])
    for case_index, case in enumerate(cases):
        nodes = case.get("nodes", [])
        marker_box = (case.get("annotations") or {}).get("marker_box")
        start = len(cells)
        parts = [f'<section class="case" data-case="{case_index}"' + (' hidden' if case_index else '') + '>',
                 '<h2>' + escape(str(case.get("cluster_id", "病例 ID 未记录"))) + '</h2>',
                 '<p class="scope">原始异常案例已揭晓；新操作只能提供病例内前瞻检验。</p>',
                 _saved("读新图前的判断与下一检查", case.get("initial_note")),
                 _saved("对象、标记及其他人工空间注记", case.get("annotations", {}))]
        if nodes:
            parts.append('<div class="inputs"><figure>' + _annotated_image(nodes[0], "clean", marker_box) +
                         '<figcaption>正常图</figcaption></figure><figure>' + _annotated_image(nodes[0], "observed", marker_box) +
                         '<figcaption>带标记的异常图</figcaption></figure></div>')
        for node in nodes:
            parts.append('<p><b>' + escape(str(node.get("label", node.get("id", "问题")))) + '：</b>' +
                         escape(str(node.get("question", "未记录"))) + '　<b>真值：</b>' + escape(str(node.get("answer", "未核验"))) + '</p>')
            parts.append(_saved(str(node.get("label", node.get("id", ""))) + " 固定完整候选、token 与数值容差",
                                {"candidates": node.get("candidates"), "tolerance": node.get("tolerance")}))
            if not (node.get("annotations") or {}).get("objects"):
                parts.append('<p class="missing">' + escape(str(node.get("label", node.get("id", "")))) + '：未保存对象位置注记。</p>')
            else:
                parts.append(_saved(str(node.get("label", node.get("id", ""))) + " 事实位置参照与来源（不是归因）", node["annotations"]))
        for resolution, title in ((6, "第一步 · 6×6 主图"), (2, "第二步 · 2×2 粗粒度对照"), (12, "第三步 · 按需局部 12×12")):
            maps = [(i, node, mapping) for i, node in enumerate(nodes) for mapping in node.get("maps", [])
                    if mapping.get("resolution") == resolution]
            parts.append('<h3>' + title + '</h3>')
            if not maps:
                parts.append('<p class="missing">尚无该尺度的记录；不插值补图。</p>')
            else:
                parts.append('<div class="maps">' + ''.join(_map(node, mapping, case_index, i, cells, marker_box)
                                                              for i, node, mapping in maps) + '</div>')
        parts.append('<p class="scale" data-scale="' + str(case_index) + '"></p>')
        relevant = [item for item in report.get("checks", []) if item.get("cluster_id") == case.get("cluster_id")]
        parts.append('<h3>联合集合与后续检查</h3>')
        if not relevant:
            parts.append('<p class="missing">没有已登记的后续检查。</p>')
        for check in relevant:
            node_index = next((i for i, node in enumerate(nodes) if node.get("id") == check.get("node_id")), None)
            parts.append('<div class="check"><b>' + escape(str(check.get("label", check.get("kind", "联合检查")))) + '</b>')
            if node_index is not None:
                node = nodes[node_index]
                condition = node.get("conditions", {}).get(check.get("key"), {})
                mapping = {"title": check.get("label", "联合检查"), "background": check.get("background", []),
                           "base_key": check.get("base_key")}
                cell = {"indices": check.get("indices", []), "key": check.get("key"), "status": condition.get("status", "unmeasured"),
                        "additive_prediction": check.get("additive_prediction"), "joint_residual": check.get("joint_residual")}
                index = len(cells)
                cells.append({"case": case_index, "node": node_index, "map": mapping, "cell": cell,
                              "values": {name: _delta(node, mapping, cell, name) for name in ("fact", "refusal")}})
                parts.append(f' <button type="button" data-cell="{index}">查看集合、分数与实际回答</button>')
                union_map = dict(mapping, title="本次操作的完整区域集合", resolution=12 if check.get("child_index") is not None
                                 or check.get("parent_check_id") is not None else 6,
                                 cells=[dict(cell, index=check.get("child_index", check.get("id", "joint")))])
                parts.append('<details><summary>查看操作区域叠图（整组共用一个实测效应）</summary>'
                             + _map(node, union_map, case_index, node_index, cells, marker_box) + '</details>')
            parts.append(_saved("登记的预测与集合（不自动判定命中）", check) + '</div>')
        values = cells[start:]
        scales.append({name: max((abs(item["values"][name]) for item in values if _number(item["values"][name])), default=0)
                       for name in ("fact", "refusal")})
        parts.append('</section>')
        sections.append(''.join(parts))
    data = {"report": report, "cells": cells, "scales": scales}
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    options = ''.join(f'<option value="{i}">{escape(str(case.get("cluster_id", i)))}</option>' for i, case in enumerate(cases))
    fixture = "cpu_test" in str(report.get("status", "")) or report.get("cpu_test") is True
    banner = '<p class="fixture">CPU 软件测试数据，仅验证页面；不是实验结果。</p>' if fixture else ''
    header = ('<h1>VLM 归因可视化 · 病例图册</h1>' + banner + '<p><b>快照状态：</b>' + escape(str(report.get("status", "未记录"))) +
              '　<b>生成时间：</b>' + escape(str(report.get("created_utc", "未记录"))) + '</p><p>' + escape(str(report.get("scope", ""))) + '</p>'
              '<p>色块表示复制内部视觉状态后的候选竞争变化，不是像素病灶。红色更支持正确候选，蓝色相反；实际回答另行核验。</p>'
              '<div class="toolbar"><label>病例 <select id="case-select">' + options + '</select></label>'
              '<label>读数 <select id="metric"><option value="fact">事实：正确答案 − 指定错误答案</option>'
              '<option value="refusal">拒答对照：正确答案 − 原拒答</option></select></label>'
              '<label><input type="checkbox" id="values" checked> 显示差值</label>'
              '<label><input type="checkbox" id="positions" checked> 显示事实位置框</label></div>'
              '<div class="legend"><span class="negative">负向</span><span class="zero">零／容差内</span>'
              '<span class="positive">正向</span><span class="missing">灰：未测／失败／不可比较</span>'
              '<span class="included">斜纹：已在背景中</span><span>细实线：对象；虚线：标记位置（均非归因）</span></div>')
    sidebar = ('<aside id="detail" aria-live="polite"><h2>读一格</h2><p>点击热图或联合检查，查看精确状态集合、原始 logP、实际前后回答及判分。</p></aside>')
    footer = ('<section><h2>观察 → 判断 → 未测检查 → 结果</h2>' + _diagnoses(report.get("diagnoses", [])) + '</section>'
              '<section><h2>下一检查比较与成本</h2><p>结果只按登记内容展示，不自动生成优势、因果机制或预测命中结论。</p>' +
              _saved("已登记的共同菜单、顺序、停止规则与比较结果", report.get("comparisons", [])) +
              _saved("新增检查与端到端成本账", report.get("cost", {})) + '</section>')
    document = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>V3 归因可视化探针图册</title><style>' + _viewer().STYLE + STYLE + '</style></head><body>' +
                '<svg width="0" height="0" aria-hidden="true"><defs><pattern id="included-pattern" width=".6" height=".6" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">'
                '<rect width=".6" height=".6" fill="#d2d6db"/><path d="M0 0V.6" stroke="#737b86" stroke-width=".2"/></pattern></defs></svg>' +
                header + '<div class="workspace"><main>' + ''.join(sections) + '</main>' + sidebar + '</div>' + footer +
                '<script type="application/json" id="probe-data">' + encoded + '</script><script>' + SCRIPT + '</script></body></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return {"output_html": str(output.resolve()), "cases": len(cases), "cells": len(cells), "model_calls": 0}


STYLE = """
body{max-width:1660px;background:#fafbfc;margin:20px auto;color:#1c2b3d}h1{font-size:28px}h2{margin-top:24px}
[hidden]{display:none!important}.toolbar{display:flex;gap:20px;flex-wrap:wrap;align-items:center;background:#fff;padding:14px;border:1px solid #ccd3dd;border-radius:8px;position:sticky;top:0;z-index:2}
select,button{font:inherit;padding:6px;border:1px solid #aab5c3;border-radius:5px;background:#fff}button{cursor:pointer}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:22px;align-items:start}.workspace main{min-width:0}
#detail{position:sticky;top:88px;background:#fff;border:1px solid #c5cfdd;border-radius:8px;padding:14px;max-height:calc(100vh - 120px);overflow:auto;font-size:13px}
.maps{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.map{background:#fff;border:1px solid #d1d8e2;border-radius:8px;padding:10px;min-width:0}.map h3{margin:2px 0;font-size:16px}.map-meta{font-size:12px;color:#526173;margin:7px 0}.map svg{display:block;width:100%;height:auto;border:1px solid #8592a3}.cell-fill{fill-opacity:.68}.cell-value{fill:#111;paint-order:stroke;stroke:#fff;stroke-width:.035;pointer-events:none}.heat-cell{cursor:pointer}.heat-cell:focus{outline:none}.heat-cell:focus .cell-fill,.heat-cell.selected .cell-fill{stroke:#ffcc00;stroke-width:.09;fill-opacity:.9}
.position-annotations{pointer-events:none}.position-annotations rect{fill:none;stroke:#111827;stroke-width:.075}.position-annotations .marker-box{stroke-dasharray:.3 .18}.position-label{font-size:.65px;fill:#111827;paint-order:stroke;stroke:#fff;stroke-width:.12px}.hide-positions .position-annotations{display:none}.inputs svg{width:100%;height:auto;display:block;border:1px solid #ccd3dd}
.inputs{display:flex;gap:12px}.inputs figure{margin:0;max-width:220px}.inputs img{width:100%;border:1px solid #ccd3dd}.inputs figcaption{font-size:12px}.scope,.scale{font-size:13px;color:#59677a}.legend{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0;font-size:12px}.legend span{padding:5px 9px;border:1px solid #bac3cf}.negative{background:#2166ac;color:white}.positive{background:#b2182b;color:white}.zero{background:#f7f7f7}.included{background:repeating-linear-gradient(45deg,#d2d6db,#d2d6db 5px,#87919f 5px,#87919f 7px)}.fixture{background:#ffe6a8;border:2px solid #b66a00;padding:12px;font-weight:bold}.diagnosis,.check{background:#fff;border:1px solid #d1d8e2;border-radius:6px;margin:10px 0;padding:12px}.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.result-grid pre{margin:5px 0}.bad{color:#a71936}.good{color:#13704d}pre{max-height:440px;overflow:auto}#detail table{font-size:11px}#detail td{overflow-wrap:anywhere}
@media(max-width:1050px){.workspace{grid-template-columns:1fr}#detail{position:static;max-height:none}.toolbar{position:static}}@media(max-width:650px){.maps{grid-template-columns:1fr}.result-grid{grid-template-columns:1fr}}
"""


SCRIPT = r"""
const payload=JSON.parse(document.getElementById('probe-data').textContent), report=payload.report;
const metric=document.getElementById('metric'), detail=document.getElementById('detail');
let selected=null;
const esc=v=>String(v??'未记录').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const dump=v=>'<pre>'+esc(JSON.stringify(v??null,null,2))+'</pre>';
const number=v=>finite(v)?v.toPrecision(7):'未测／不可比较';
function colour(v,limit,tolerance){
  if(!finite(v))return '#b9bec5';
  if(v===0||(finite(tolerance)&&Math.abs(v)<=tolerance))return '#f7f7f7';
  const t=limit?Math.min(1,Math.abs(v)/limit):0, end=v>0?[178,24,43]:[33,102,172];
  return 'rgb('+end.map(c=>Math.round(247+(c-247)*t)).join(',')+')';
}
function refresh(){
  document.body.classList.toggle('hide-positions',!document.getElementById('positions').checked);
  document.querySelectorAll('g[data-cell]').forEach(el=>{
    const item=payload.cells[Number(el.dataset.cell)], node=report.cases[item.case].nodes[item.node];
    const v=item.values[metric.value], included=item.cell.status==='included', tol=node.tolerance?.[metric.value];
    const fill=included?'url(#included-pattern)':colour(v,payload.scales[item.case][metric.value],tol);
    el.querySelectorAll('.cell-fill').forEach(rect=>rect.setAttribute('fill',fill));
    el.querySelector('.cell-value').textContent=included?'已含':finite(v)?(v===0?'0':v.toPrecision(2)):'—';
    el.querySelector('.cell-value').style.display=document.getElementById('values').checked?'':'none';
  });
  document.querySelectorAll('[data-scale]').forEach(el=>{
    const scale=payload.scales[Number(el.dataset.scale)][metric.value];
    el.textContent='本病例该读数全部已保存地图与检查共用固定色尺：±'+number(scale)+' nat。两题强度不能直接解释为事实重要性。';
  });
  if(selected!==null)show(selected);
}
function generated(condition,label){
  const output=condition?.output, verdict=condition?.correct===true?'答对':condition?.correct===false?'答错':'未判定';
  return '<div><b>'+label+'</b><pre>'+esc(output?.text??'未生成')+'</pre><p>'+verdict+' · 停止：'+esc(output?.stop_reason??'未记录')+'</p></div>';
}
function show(index){
  selected=index;
  document.querySelectorAll('g[data-cell]').forEach(el=>el.classList.toggle('selected',Number(el.dataset.cell)===index));
  const item=payload.cells[index], node=report.cases[item.case].nodes[item.node], mapping=item.map, cell=item.cell;
  const conditions=node.conditions??{}, before=conditions[mapping.base_key], after=conditions[cell.key], value=item.values[metric.value];
  const added=(cell.indices??[]).filter(i=>!(mapping.background??[]).includes(i)), tol=node.tolerance?.[metric.value];
  let meaning=cell.status==='included'?'已在背景中：无新增操作，不计作新测的零效应':!finite(value)?'未测、失败或基准不可比较':value===0?'实测精确零':finite(tol)&&Math.abs(value)<=tol?'实测近零（容差内，保留原值）':value>0?'正确候选相对竞争候选更占优势':'正确候选相对竞争候选更不占优势';
  let html='<h2>'+esc(node.label??node.id)+' · '+esc(mapping.title??'检查')+'</h2><p>'+esc(node.question)+'</p><p><b>真值：</b>'+esc(node.answer)+'</p>';
  html+='<p><b>差值：</b>'+number(value)+' nat<br>'+meaning+'</p><p>格状态：'+esc(cell.status)+'；基准：'+esc(before?.status??'未测')+'；操作后：'+esc(after?.status??'未测')+'</p>';
  html+='<p>实际新增 '+added.length+' 个状态；显示容差：'+number(tol)+'</p>';
  html+='<div class="result-grid">'+generated(before,'背景的实际回答')+generated(after,'追加后的实际回答')+'</div>';
  html+='<table><tr><th>完整候选</th><th>背景 logP</th><th>追加 logP</th></tr>';
  for(const [key,label] of [['positive','正确事实'],['negative','指定错误事实'],['refusal','原拒答']]){
    html+='<tr><td>'+label+'<br>'+esc(node.candidates?.[key]?.text)+'</td><td>'+number(before?.scores?.[key]?.log_prob)+'</td><td>'+number(after?.scores?.[key]?.log_prob)+'</td></tr>';
  }
  html+='</table><p>颜色不证明事实正确，也不定位污染参数。实际输出的其他错误答案仍需查看。</p>';
  if(metric.value==='fact'&&finite(cell.additive_prediction))html+='<p><b>单格相加预测：</b>'+number(cell.additive_prediction)+' nat；<b>联合实测减相加预测：</b>'+number(cell.joint_residual)+' nat。差异表示该读数上的非相加，不能直接叫作协同机制。</p>';
  html+='<details><summary>精确 token 集合、背景与供体</summary>'+dump({background:mapping.background,region:cell.indices,added,base_key:mapping.base_key,key:cell.key,donor_before:before?.donor_id,donor_after:after?.donor_id})+'</details>';
  html+='<details><summary>原始记录：逐 token 分数、原因、输出与计时</summary>'+dump({cell,before,after})+'</details>';
  detail.innerHTML=html;
}
document.querySelectorAll('[data-cell]').forEach(el=>{
  el.addEventListener('click',()=>show(Number(el.dataset.cell)));
  if(el.tagName.toLowerCase()==='g')el.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();show(Number(el.dataset.cell));}});
});
document.getElementById('case-select').addEventListener('change',event=>{
  document.querySelectorAll('.case').forEach(section=>section.hidden=section.dataset.case!==event.target.value);
  selected=null;detail.innerHTML='<h2>读一格</h2><p>已切换病例，请点击一格查看详情。</p>';
});
metric.addEventListener('change',refresh);document.getElementById('values').addEventListener('change',refresh);document.getElementById('positions').addEventListener('change',refresh);refresh();
"""
