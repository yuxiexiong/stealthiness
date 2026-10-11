"""Narrow UI/data repairs found by the 2026-09-28 self-check."""
import json
import re
from pathlib import Path


def apply(s, original):
    def replace(old, new):
        nonlocal s
        assert s.count(old) == 1, (old[:100], s.count(old))
        s = s.replace(old, new, 1)

    pattern = r'(<script type="application/json" id="j_meta">)(.*?)(</script>)'
    old_meta = json.loads(re.search(pattern, original, re.S)[2])
    meta = json.loads(re.search(pattern, s, re.S)[2])
    for key in ('F', 'probe_samples'):
        assert key in old_meta
        meta[key] = old_meta[key]
    s = re.sub(pattern, lambda m: m[1] + json.dumps(meta, ensure_ascii=False, separators=(',', ':')) + m[3], s, flags=re.S)
    pattern = r'(<script type="application/json" id="j_tokens">)(.*?)(</script>)'
    tokens = json.loads(re.search(pattern, s, re.S)[2])
    instrument = json.loads(Path(__file__).with_name('instrument-tokens.json').read_text())
    assert len(instrument) == 20 and all(len(v) == 9 for v in instrument.values())
    tokens['p_instrument/clean'] = instrument
    s = re.sub(pattern, lambda m: m[1] + json.dumps(tokens, ensure_ascii=False, separators=(',', ':')) + m[3], s, flags=re.S)

    replace('b.onclick=()=>{S.arm=a; buildSegs(); draw(); sync();};',
            'b.onclick=()=>{S.arm=a; buildSegs(); draw(); sync(); showMode("grid");};')
    replace('<p class="qline" id="qline"></p>', '<p class="qline" id="qline"></p><p id="sampleStatus" aria-live="polite"></p>')
    replace('ps.onchange=()=>{CMP.preset=ps.value; const P=presetOf();',
            'ps.onchange=()=>{CMP.preset=ps.value; CMP.order=null; $("#sortBtn").textContent="按 P-5.0 / T2 / B 捕获率排"; const P=presetOf();')
    replace('return out.length ? out : list;', 'return out;')
    replace('function ensureSample(){\n  const list=sampleList();\n  if(list.length && !list.includes(CMP.sample)){ CMP.sample=list[0]; _atlasCache={};\n    const el=$("#sampIn"); if(el) el.value=CMP.sample; }\n}', '''function ensureSample(){
  const list=sampleList();
  if(!list.includes(CMP.sample)){ CMP.sample=list.length?list[0]:null; _atlasCache={}; }
  $("#sampIn").value=CMP.sample==null?"":CMP.sample;
  for(const id of ["#pv","#nx","#sortBtn","#sampIn"]) $(id).disabled=!list.length;
  $("#sampleStatus").textContent=`当前可浏览 ${list.length} 个样本` + (CMP.order?" · 按 P-5.0 / T2 / B 捕获率降序":"");
}''')
    replace('const seq=(CMP.drawSeq=(CMP.drawSeq||0)+1), smp=CMP.sample, stale=()=>seq!==CMP.drawSeq;', '''const seq=(CMP.drawSeq=(CMP.drawSeq||0)+1), smp=CMP.sample, stale=()=>seq!==CMP.drawSeq;
  if(smp==null){
    mtx.innerHTML='<p class="cmpnote">当前筛选条件下没有样本。请选择「样本：全部」或其他筛选条件。</p>';
    $("#qline").textContent=""; $("#cmpLegend").textContent=""; $("#cmpNote").textContent=P.note;
    return;
  }''')
    replace('$("#sampIn").onchange=()=>{const n=parseInt($("#sampIn").value,10);\n    if(!isNaN(n)){ CMP.sample=n; _atlasCache={}; drawCompare(); }};', '''$("#sampIn").onchange=()=>{
    const el=$("#sampIn"), raw=el.value.trim(), n=Number(raw);
    if(!/^\\d+$/.test(raw)||!sampleList().includes(n)){
      el.value=CMP.sample==null?"":CMP.sample;
      $("#sampleStatus").textContent="输入的样本号不在当前集合中；已保留原样本。可用左右箭头翻页。";
      return;
    }
    CMP.sample=n; _atlasCache={}; drawCompare();
  };
  $("#sampIn").onblur=$("#sampIn").onchange;
  $("#sampIn").onkeydown=e=>{if(e.key==="Enter") $("#sampIn").onchange();};''')
    replace('const list = CMP.order || sampleList();', 'const allowed=sampleList();\n  const list = CMP.order ? CMP.order.filter(i=>allowed.includes(i)) : allowed;\n  if(!list.length) return;')
    replace('CMP.order=null; ensureSample(); drawCompare(); };', 'CMP.order=null; $("#sortBtn").textContent="按 P-5.0 / T2 / B 捕获率排"; ensureSample(); drawCompare(); };')
    replace('固定一个臂和一个输入列，把三个标量和两台仪器铺开。行是仪器，列是标量。',
            '此预设固定 P-5.0 模型、trig 输入列，把 T1/T2/T3 和 A/B 同时铺开，因此上方仪器和标量选择器锁定。')
    replace('f.innerHTML=\'<span>这一格没成像</span>\'; b.appendChild(f);',
            'f.innerHTML=\'<span>该样本 / 标量 / 仪器组合未成像</span>\'; b.disabled=true; b.appendChild(f);')
    replace('empty.textContent = S.sc==="C"\n    ? "这里没有 C（正确答案图）。C 只为补充实验第一阶段算过：11 个臂、p_core 探针集的 clean 与 trig 两列。"\n    : "这个组合没有成像。训练轨迹的中途检查点只跑了仪器 A（遮挡法逐检查点太贵，D11）。";',
            'empty.textContent = `当前 ${arm} / ${S.probe} / ${S.col} / ${S.sc} / ${S.ins} 没有成像数据。可切换上方输入列、标量或仪器；补充轨迹 clean 仅 A、trig 为 A/B，旧 @k 轨迹仅 A，C 只在已计算它的 p_core clean/trig 组合提供。`;')
    replace('$("#hint").innerHTML = unpl + (traj', '$("#hint").innerHTML = `<b>${arm} · ${S.probe} / ${S.col} · ${S.sc} / ${S.ins} · ${rows.length} 个样本</b><br>` + unpl + (traj')
    replace('const gset = S.col!=="clean" ? attackedSet(arm) : null;', 'const gset = S.probe==="p_core" && S.col==="trig" ? attackedSet(arm) : null;')
    replace('const aset = c.column!=="clean" ? attackedSet(c.arm) : null;', 'const aset = c.probe==="p_core" && c.column==="trig" ? attackedSet(c.arm) : null;')
    replace('$("#dSub").textContent=`${V.col} · ${V.ins==="A"?"梯度·带符号":"遮挡·带符号（只画正值部分）"} · 峰 ${e[8]} · 底 ${e[7]}`',
            '$("#dSub").textContent=`${V.col} · ${V.ins==="A"?"梯度·带符号":ctx&&ctx.mode==="diff"?"遮挡·带符号差分":"遮挡·带符号（只画正值部分）"} · 峰 ${e[8]} · 底 ${e[7]}`')
    replace('The rise is not drawn anywhere on this page (D58).', 'Negative values are hidden in this positive-part view; the paired difference view uses signed values.')
    replace('const CR=$("#dCleanRow"); CR.hidden=true;', 'const CR=$("#dCleanRow"); CR.hidden=true;\n  $("#dCanvasLab").textContent="读取中…"; $("#dRefLab").textContent="对照读取中…";')
    replace('if(!live() || !cc) return;', 'if(!live()) return;\n    if(!cc){ $("#dCleanPosLab").textContent="该组合没有配对对照"; return; }')
    replace('if(!rstem) return;', 'if(!rstem){ RL.textContent="该组合没有配对干净对照数据"; return; }')
    replace('if(ri<0) return;', 'if(ri<0){ RL.textContent="该样本 / 标量 / 仪器没有配对干净对照数据"; return; }')
    replace('$("#dClose").onclick=()=>$("#ov").removeAttribute("open");',
            'function closeDetail(){ DETAIL_SEQ++; $("#ov").removeAttribute("open"); }\n$("#dClose").onclick=closeDetail;')
    replace('$("#ov").onclick=ev=>{if(ev.target===$("#ov")) $("#ov").removeAttribute("open");};',
            '$("#ov").onclick=ev=>{if(ev.target===$("#ov")) closeDetail();};')
    replace('addEventListener("keydown",ev=>{if(ev.key==="Escape") $("#ov").removeAttribute("open");});',
            'addEventListener("keydown",ev=>{if(ev.key==="Escape") closeDetail();});')

    # Native dialog lets the existing SVG remain readable in a narrow app panel.
    replace('<div id="ov" role="dialog"', '''<dialog id="curveDialog"><button id="curveClose" type="button">关闭曲线大图</button><p id="curveCaption"></p><div tabindex="0" role="region" aria-label="曲线大图，使用方向键或翻页键滚动"><img id="curveImage" alt=""></div></dialog>
<div id="ov" role="dialog"''')
    replace('</style>', '''
#curveDialog{width:95vw;max-width:1200px;max-height:95vh;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:8px}
#curveDialog>div{overflow:auto;max-height:78vh}#curveImage{display:block;max-width:none}
#curveDialog::backdrop{background:#0009}#curveClose{position:sticky;top:0;cursor:pointer;padding:8px}
#curvesPane figure>button{display:block;width:100%;padding:0;border:0;background:none;cursor:zoom-in}
.mc:disabled{cursor:default}.cmpbar select{max-width:min(22rem,80vw)}
</style>''')
    replace('boot();\n</script>', '''for(const im of document.querySelectorAll("#curvesPane figure img")){
  const b=document.createElement("button"); b.type="button"; b.setAttribute("aria-label","放大 "+im.alt);
  im.replaceWith(b); b.appendChild(im);
  b.onclick=()=>{ $("#curveImage").src=im.src; $("#curveImage").alt=im.alt; $("#curveCaption").textContent=im.alt+"（原尺寸，可滚动查看）"; $("#curveDialog").showModal(); $("#curveDialog>div").scrollTop=0; $("#curveDialog>div").scrollLeft=0; };
}
$("#curveClose").onclick=()=>$("#curveDialog").close();
$("#curveDialog").onclick=e=>{if(e.target===$("#curveDialog")) $("#curveDialog").close();};
boot();
</script>''')
    return s
