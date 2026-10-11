"""Small, repeatable add-on to the existing atlas merge, preserving its embedded sources."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def apply(s):
    assert 'id="regionsPane"' not in s
    def rep(old,new):
        nonlocal s
        assert s.count(old)==1,(old[:80],s.count(old))
        s=s.replace(old,new,1)
    rep('</style>',(ROOT/'panel.css').read_text()+(ROOT/'compare-layers.css').read_text()+'\n</style>')
    rep('<div class="modes" id="modeSeg"></div>','<div class="modes" id="modeSeg"></div>\n'+(ROOT/'panel.html').read_text())
    rep('[["curves","补充实验曲线"],','[["regions","分割 × 热点变化"],["curves","补充实验曲线"],')
    rep('buildCompareUI(); showMode("curves");','buildCompareUI(); clSetup(); rSetup(); showMode("regions");')
    rep('CMP.mode=m;','CMP.mode=m;\n  $("#regionsPane").hidden = m!=="regions";\n  $("#rail").hidden=m!=="grid"; $("#shell").classList.toggle("cl-wide",m!=="grid");')
    rep('<script>\n"use strict";', '<script type="application/json" id="j_regions">'+(ROOT/'data.json').read_text()+'</script>\n<script>\n"use strict";')
    rep('boot();',(ROOT/'panel.js').read_text()+'\n'+(ROOT/'compare-layers.js').read_text()+'\nboot();')
    rep('<p class="qline" id="qline"></p>', '<p class="qline" id="qline"></p>\n'+(ROOT/'compare-layers.html').read_text())
    rep('return applySampFilter(baseSampleList(P));', 'const list=applySampFilter(baseSampleList(P));\n  return CMP.showAllSamples ? list : clCompleteSamples(list);')
    rep('  const list=sampleList();\n  if(!list.includes(CMP.sample))', '  const list=sampleList(); clSampleScope();\n  if(!list.includes(CMP.sample))')
    rep('mtx.innerHTML=\'<p class="cmpnote">当前筛选条件下没有样本。请选择「样本：全部」或其他筛选条件。</p>\';', 'mtx.innerHTML=\'<p class="cmpnote">\'+clSampleScope()+\'</p>\';')
    rep('const seq=(CMP.drawSeq=', 'const layerMode=CMP.layerMode||"photo", layerNumbers=!!CMP.layerNumbers;\n  const seq=(CMP.drawSeq=')
    rep('const P = presetOf(), mtx=$("#mtx");', 'clHorizontalUI();\n  let P = presetOf(); const mtx=$("#mtx");')
    rep('  // 1. 组出全部格子的规格', '''  P=clPreset(P);
  if(CMP.layout==='multi')for(const id of ['#pv','#nx','#sampIn','#sortBtn'])$(id).disabled=true;
  if(!P.rows.length){const msg='指定样本不满足当前筛选。请更换编号或点「展示所有样本」。';mtx.innerHTML='<p class="cmpnote">'+msg+'</p>';$('#cmpLayerStatus').textContent=msg;$('#cmpCoverageText').textContent='当前没有可展示的指定样本';$('#qline').textContent='';return;}
  // 1. 组出全部格子的规格''')
    rep('({...P.cell(r,c), sample:smp})', '({sample:smp, ...P.cell(r,c)})')
    rep('column:sp.column, sample:smp,', 'column:sp.column, sample:sp.sample,')
    rep('const c = await readCell(sp);', 'const c = await readCell(sp); if(c) c.layerMode=sp.layerMode;')
    rep('  // 4. 算跨整屏的色标跨度', '''  const activeImages=new Set(flat.map(c=>REG.samples[c.e[0]]));
  for(const key of CL.base.keys())if(!activeImages.has(key))CL.base.delete(key);
  for(const [key,v] of CL.views)if(!activeImages.has(v.iid))CL.views.delete(key);
  await Promise.all([...new Map(flat.map(c=>[clKey(c),c])).values()].map(clPrime));
  if(stale()) return;
  for(const c of flat){ c.layerNumbers=layerNumbers; }
  // 4. 算跨整屏的色标跨度''')
    rep('mtx.innerHTML=""; mtx.appendChild(t);', 'mtx.innerHTML=""; mtx.appendChild(t);\n  clDemo(cells,refs,span,dspan);\n  if(P.horizontal)clOriginals(t,P,cells);')
    rep('const flat = cells.flat().filter(Boolean);', 'const flat = cells.flat().filter(Boolean); clShowCoverage(specs);')
    rep('const hr=document.createElement("tr"); hr.appendChild(document.createElement("th"));', '''if(P.horizontal)t.style.width=(280+P.cols.length*240)+'px';
  const hr=document.createElement("tr"); hr.innerHTML=P.horizontal?'<th scope="col">样本 / 原图</th><th scope="col">输入 / 显示模式</th>':'<th scope="col">模型 / 实验变量</th><th scope="col">输入 / 测量</th>';''')
    rep('th.textContent=c[1]; hr.appendChild(th);', '''th.textContent=c[1];
    if(P.horizontal){const meta=document.createElement('small');meta.textContent=[c[0]!==c[1]?c[0]:'',c[2]||'',!String(c[2]).includes('ASR')?asrTxt(c[0]):''].filter(Boolean).join(' · ');th.appendChild(meta);}
    hr.appendChild(th);''')
    rep('t.appendChild(hr);', 'const head=t.createTHead(); head.appendChild(hr);')
    rep('tr.appendChild(rh);', '''rh.rowSpan=P.rows[r].groupSize; rh.scope="rowgroup";
    if(P.rows[r].groupStart) tr.appendChild(rh);
    const input=document.createElement("th"); input.className="cl-input"; input.scope="row";
    input.textContent=P.rows[r].inputName; tr.appendChild(input);
    tr.classList.toggle("cl-model-start",P.rows[r].groupStart);''')
    rep('const td=document.createElement("td"), c=cells[r][k];', 'const td=document.createElement("td"), c=cells[r][k]; td.dataset.layer=specs[r][k].layerMode;')
    rep('b.className="mc"+(c?"":" void")+(c&&isDual(c)?" dualw":"");', '''b.className="mc"+(c?"":" void")+(c&&isDual(c)?" dualw":"");
      if(c) b.setAttribute("aria-label",`样本 #${c.e[0]} · ${c.arm} · ${P.rows[r].inputName} · ${CL_NAMES[c.layerMode]} · 点击放大`);''')
    rep('aset.has(CMP.sample)', 'aset.has(c.e[0])')
    rep('[String(CMP.sample)];\n          const raw=[];', '[String(c.e[0])];\n          const raw=[];')
    rep('cv2.className="map";\n        pair.appendChild(cv); pair.appendChild(cv2); b.appendChild(pair);', 'cv2.className="map";\n        pair.appendChild(cv); pair.appendChild(cv2); const labs=document.createElement("div"); labs.className="cl-pair-labels"; labs.innerHTML=CMP.norm==="diff"?"<span>相对增加（正）</span><span>相对降低（负）</span>":"<span>支持（正）</span><span>抑制（负）</span>"; b.appendChild(labs); b.appendChild(pair);')
    rep('P.cols.some(c=>c[2]==="C")', 'flat.some(c=>c.scalar==="C")')
    rep('sw.append("双图 左：支持 0 ", mono([200,30,40]), " 峰　右：抑制 0 ", mono([30,90,200]), " 峰" + how);', 'sw.append(P.horizontal?"双图 上：支持 0 ":"双图 左：支持 0 ", mono([200,30,40]), P.horizontal?" 峰　下：抑制 0 ":" 峰　右：抑制 0 ", mono([30,90,200]), " 峰" + (P.horizontal?how.replace("左 =","上 =").replace("右 =","下 ="):how));')
    rep('  const usesC = CMP.scalar', '''  if(CMP.layout==='multi')$('#qline').textContent='多样本对齐：'+clHorizontalSamples().map(i=>'#'+i).join('、')+'；问题与原始输入见每组左侧。';
  const usesC = CMP.scalar''')
    rep('<div class="mtx" id="mtx"></div>\n      <div class="legend" id="cmpLegend"></div>\n      <p class="cmpnote" id="cmpNote"></p>', '''<div class="legend" id="cmpLegend"></div>
      <details class="cl-method"><summary>色标与实验说明</summary><p class="cmpnote" id="cmpNote"></p></details>
      <p class="cl-read-guide">横向看三种画法，纵向看实验变量变化；每个模型内按输入或测量分行。表头与左侧标签固定，点击图片可放大。</p>
      <div class="mtx" id="mtx" tabindex="0" role="region" aria-label="逐样本对比矩阵，可横向和纵向滚动"></div>''')
    rep('f.innerHTML=\'<span>该样本 / 标量 / 仪器组合未成像</span>\'; b.disabled=true; b.appendChild(f);', 'cv.remove(); f.textContent=clMissing(specs[r][k]); b.setAttribute("aria-label",f.textContent); b.disabled=true; b.appendChild(f);')
    rep('if(smp==null){', 'if(smp==null){\n    $("#cmpCoverageText").textContent="当前筛选无样本"; $("#cmpJumpAvailable").hidden=true;\n    $("#cmpDemoCards").innerHTML=""; $("#cmpDemoCaption").textContent="当前筛选无样本"; $("#cmpLayerStatus").textContent="当前筛选无样本";')
    rep('mtx.innerHTML=\'<p class="cmpnote">读取中…</p>\';', '$("#cmpDemoCards").innerHTML=""; $("#cmpDemoCaption").textContent=""; $("#cmpRegionLegend").textContent=""; $("#cmpLayerStatus").textContent="读取中…";\n  mtx.innerHTML=\'<p class="cmpnote">读取中…</p>\';')
    rep('if(!flat.length){ mtx.innerHTML=', 'if(!flat.length){ $("#cmpLayerStatus").textContent="当前组合没有成像数据"; mtx.innerHTML=')
    # Base and outline hooks are shared by the comparison matrix and its full-size view.
    begin=s.index('function paintMatrix('); end=s.index('/* 尺子',begin)
    renderer=s[begin:end]
    renderer=renderer.replace('cv.width=cv.height=S;', 'cv.width=cv.height=S;\n  if(clBefore(cv,c)) return;')
    renderer=renderer.replace('if(a && j>=0) g.drawImage', 'if(!clBase(cv,c) && a && j>=0) g.drawImage')
    renderer=renderer.replace('g.globalAlpha=1;', 'g.globalAlpha=1; clAfter(cv,c);')
    s=s[:begin]+renderer+s[end:]
    rep('+ ` · 色标：${normTxt}`;', '+ ` · 色标：${normTxt}` + (cell?.layerMode ? ` · ${CL_NAMES[cell.layerMode]}` : "");')
    rep('const k2=atlasKey(cc.probe,cc.column);', 'cc.layerMode=cell.layerMode; cc.layerNumbers=cell.layerNumbers; await clPrime(cc);\n    const k2=atlasKey(cc.probe,cc.column);')
    rep('if(rcell){ paintMatrix(rc, rcell, ctx); return; }', 'if(rcell){ rcell.layerMode=cell.layerMode; rcell.layerNumbers=cell.layerNumbers; await clPrime(rcell); if(!live()) return; paintMatrix(rc, rcell, ctx); return; }')
    rep('  await paintInto(rc, cv=>paint(cv,refArm,rstem,rv.entries[ri],ri), live);', '''  if(ctx && cell?.layerMode && cell.layerMode!=="photo"){
    const rcell=await cellObj(refArm,rstem,rv.entries[ri],ri);
    rcell.layerMode=cell.layerMode; rcell.layerNumbers=cell.layerNumbers;
    await clPrime(rcell); const key=atlasKey(rcell.probe,rcell.column); if(key) await primeAtlas(key);
    if(!live()) return; paintMatrix(rc,rcell,{mode:"panel"}); return;
  }
  await paintInto(rc, cv=>paint(cv,refArm,rstem,rv.entries[ri],ri), live);''')
    return s
