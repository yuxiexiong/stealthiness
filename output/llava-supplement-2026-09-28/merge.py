"""Merge the already-built supplement and its existing figures into the atlas.
Run audit.py first. No model execution or scientific verdicts are performed.
"""
import base64
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / 'source/attribution-microscope/supplement/phase2b/figs'
OUT = ROOT.parents[1] / 'attribution-microscope/attribution-atlas.html'
s = (ROOT / 'server-attribution-atlas.html').read_text()
audit = json.loads((ROOT / 'audit.json').read_text())
assert hashlib.sha256(s.encode()).hexdigest() == audit['server_html_sha256']

def replace(old, new):
    global s
    assert s.count(old) == 1, old[:100]
    s = s.replace(old, new, 1)

pane = '''<section id="curvesPane" hidden>
<h2>LLaVA-1.5-7B · 补充实验绘图</h2>
<p>Phase 2 + Phase 2b（含每 5 步、每步补点）。两条训练轨迹、剂量路径、同剂量换毒样本组均已纳入。
原实验图保留在「逐样本对比」和「全样本网格」中。</p>
<p><b>范围：</b>行为测试每点 200 道题；以下归因曲线使用固定的 60 个轨迹样本。
种子 1：27 点；种子 2：20 点；剂量路径：8 点（含 CLEAN）；换样本组：2 点。
共 57 点、A/B 各 3 张图与 1 张数值表。点为中位数，误差线为四分位范围。</p>
<p><b>读图边界：</b>轨迹补充成像中 A 有 clean/trig 两列，B 只有 trig 列；B 的 clean 空格是未测。
种子 1 匹配最近的 CLEAN 检查点，种子 2 匹配最近的 RETRAIN-A-D 检查点。
空心点为加密补点，虚线轮廓为计划外补拍。这里展示描述性结果，H1–H4 判档未运行。</p>
<p>已核验：原有图表条目完整保留，所选检查点和 60 个样本齐备，图片可解码，表中 228 项统计值与曲线 JSON 一致。
这是产物完整性核验，未重新执行 GPU 实验。</p>
'''
for ins in ('B', 'A'):
    pane += f'<details class="curve-instrument" {"open" if ins == "B" else ""}><summary>仪器 {ins} · T2 攻击答案方向</summary>'
    for name, title in [('step','训练全程'), ('window','过渡窗口与加密补点'), ('asr','相同 ASR 下的路径对比')]:
        svg = (ROOT / 'readable-figs' / f'fig2b_{name}_{ins}.svg').read_bytes()
        pane += f'<figure><figcaption>{title}</figcaption><img alt="仪器 {ins}：{title}" src="data:image/svg+xml;base64,{base64.b64encode(svg).decode()}"></figure>'
    table = re.search(r'<table>.*?</table>', (FIGS / f'table2b_{ins}.html').read_text(), re.S).group()
    pane += f'<details><summary>查看仪器 {ins} 的 57 行原始汇总数值</summary><div class="curve-table">{table}</div></details></details>'
pane += '</section>'
replace('<title>Attribution Atlas</title>', '<title>LLaVA-1.5-7B · Attribution Atlas</title>')
replace('<h1>Attribution Atlas</h1>', '<h1>LLaVA-1.5-7B<br>Attribution Atlas</h1>')
replace('heatmaps · 24 arms', '归因面板 · <span id="narms">…</span> 个模型/检查点')
replace('另有补充实验的正确答案图 C', '另有正确答案 C 面板')
replace('composited in your browser from the arrays', '面板计数含图像侧与文本侧；不是原图数量')
replace('<div class="modes" id="modeSeg"></div>', '<div class="modes" id="modeSeg"></div>\n' + pane)
replace('$("#nmaps").textContent=n.toLocaleString();', '$("#nmaps").textContent=n.toLocaleString();\n  $("#narms").textContent=Object.keys(idx.arms).length;')
replace('[["cmp","逐样本对比"],["grid","全样本网格"]]', '[["curves","补充实验曲线"],["cmp","逐样本对比"],["grid","全样本网格"]]')
replace('buildCompareUI(); showMode("cmp");', 'buildCompareUI(); showMode("curves");')
replace('CMP.mode=m;', 'CMP.mode=m;\n  $("#curvesPane").hidden = m!=="curves";')
replace('function refArmOf(arm){', 'function refArmOf(arm){\n  if(/^RETRAIN-A-D@s\\d+$/.test(arm)) return arm;')
replace('toFixed(v>0&&v<0.01?1:0)', 'toFixed(1)')
# Seed 2 uses RETRAIN-A-D, so shared comparison labels must name the role.
for old, new in [('差分 vs CLEAN', '差分 vs 配对干净模型'),
                 ('比 CLEAN 更', '比配对干净模型更'), ('对 CLEAN 的差', '对配对干净模型的差')]:
    assert old in s
    s = s.replace(old, new)
replace('画的是 <b>该格 − CLEAN 同输入同样本</b>，也就是协议里的合法比较 C1。红＝投毒后这里更受关注，蓝＝更被压低。CLEAN 自己那一行按定义是零。',
        '画的是 <b>该格 − 配对干净模型（同输入、同样本）</b>。红＝归因值相对增加，蓝＝相对降低；不是原始注意力。种子 1 用最近的 CLEAN 检查点，种子 2 用最近的 RETRAIN-A-D 检查点；终点臂用 CLEAN。对照自身的差分为零。')
replace('Trajectory checkpoints carry instrument A only.', '旧轨迹仅含仪器 A；补充轨迹的 clean 列仅含 A，trig 列含 A/B。')
replace('</style>', '''
#curvesPane{max-width:1160px}
#curvesPane p{max-width:1000px;color:var(--dim);line-height:1.7}
#curvesPane h2{font-size:23px;margin:12px 0}
#curvesPane summary{cursor:pointer;padding:12px 0;font-weight:600}
#curvesPane figure{margin:12px 0 28px}
#curvesPane figcaption{font-weight:600;margin:8px 0}
#curvesPane img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
#curvesPane .curve-instrument{border-top:1px solid var(--line);margin-top:18px}
#curvesPane .curve-table{overflow:auto;max-height:650px}
#curvesPane table{border-collapse:collapse;font-size:12px;white-space:nowrap}
#curvesPane td,#curvesPane th{border:1px solid var(--line);padding:6px 9px;text-align:right}
#curvesPane th{position:sticky;top:0;background:var(--panel)}
@media(max-width:800px){#rail{position:static;height:auto;max-height:34vh;overflow:auto}}
</style>''')
assert s.count('data:image/svg+xml;base64,') == 6
assert s.count('<table>') == 2
# Embedded data and original map resources must not be touched by UI integration.
for pattern in (r'<script type="application/json" id="j_[^"]+">.*?</script>', r'<img id="[^"]+"[^>]+>'):
    new = re.findall(pattern,s,re.S)
    old = re.findall(pattern,(ROOT/'server-attribution-atlas.html').read_text(),re.S)
    assert new == old, (pattern, len(new), len(old), [(i,len(a),len(b),a[:80],b[:80]) for i,(a,b) in enumerate(zip(new,old)) if a!=b])
from ui_fixes import apply
s = apply(s, (ROOT / 'original-attribution-atlas.html').read_text())
# Add frozen COCO+SAM3 masks and full-precision regional statistics.
import importlib.util
addon = ROOT.parent / 'segmentation-heatmap-linkage-2026-09-28/integrate.py'
spec = importlib.util.spec_from_file_location('regional_addon', addon)
regional_addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regional_addon)
s = regional_addon.apply(s)
OUT.write_text(s)
print(f'{OUT}\n{OUT.stat().st_size / 1048576:.2f} MiB')
