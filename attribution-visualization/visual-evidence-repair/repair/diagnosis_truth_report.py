"""Offline heatmap/table readers of public evidence only; never open an oracle."""
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlsplit

from .diagnosis_truth import packet_sha256, validate_packet
from .visual_probe_protocol import region


def _public_view(packet):
    action_by_id = validate_packet(packet)
    if packet.get("mode") == "historical_answer_only":
        raise ValueError("reader maps require complete candidate scores, not historical answer-only replay")
    candidates = packet["candidates"]
    if packet["truth_answer"] not in candidates:
        raise ValueError("reader truth answer must match its candidate text exactly")
    actions = packet["actions"]
    known = {a["id"] for a in actions if a["known"]}
    observations = {}
    for key, observation in packet["observations"].items():
        output, scores = observation["output"], observation["scores"]
        observations[key] = {"output": {"text": output["text"], "stop_reason": "eos"},
                             "scores": deepcopy(scores)}
    image_uri = packet.get("image_uri", "")
    scheme = urlsplit(image_uri).scheme.lower()
    if (scheme not in ("", "file", "data") or image_uri.startswith("//")
            or scheme == "file" and urlsplit(image_uri).netloc not in ("", "localhost")
            or scheme == "data" and not image_uri.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,"))):
        raise ValueError("reader images must be local files or embedded raster images")
    maps = []
    for mapping in packet.get("maps", []):
        resolution = mapping["resolution"]
        if resolution not in (2, 6, 12) or mapping["base_id"] not in known:
            raise ValueError("map requires a known baseline and 2, 6 or 12 resolution")
        cells = mapping["cells"]
        if (len({c["index"] for c in cells}) != len(cells)
                or any(type(c["index"]) is not int or not 0 <= c["index"] < resolution ** 2
                       or c["action_id"] not in action_by_id for c in cells)):
            raise ValueError("invalid map cells")
        base = action_by_id[mapping["base_id"]]
        if mapping["background"] != base["indices"]:
            raise ValueError("map background differs from baseline operation")
        for cell in cells:
            action = action_by_id[cell["action_id"]]
            if (cell["indices"] != region(resolution, cell["index"])
                    or action["indices"] != sorted(set(base["indices"]) | set(cell["indices"]))
                    or action["donor"] != base["donor"]):
                raise ValueError("map cell does not represent its actual intervention")
        maps.append({k: deepcopy(mapping[k]) for k in ("title", "resolution", "background", "base_id")})
        maps[-1]["cells"] = [{k: deepcopy(c[k]) for k in ("index", "action_id", "indices")} for c in cells]
    # Do not embed arbitrary metadata, including any accidental evaluator-only fields.
    view = {k: deepcopy(packet[k]) for k in ("schema", "unit_id", "cluster_id", "input_condition",
                                           "truth_answer", "donor_answer", "candidates", "question")}
    view.update(image_uri=image_uri, maps=maps, observations=observations,
                actions=[{k: deepcopy(a[k]) for k in ("id", "indices", "donor", "known", "tags")}
                         for a in actions])
    return view


def render_packet(packet, output: Path, mode="heatmap"):
    """Render one blinded reader file; the submission hash binds the original packet."""
    if mode not in ("heatmap", "table"):
        raise ValueError("mode must be heatmap or table")
    view = _public_view(packet)
    encoded = json.dumps(view, ensure_ascii=False, allow_nan=False)
    for character, replacement in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                                   ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(character, replacement)
    content = PAGE.replace("__MODE__", mode).replace("__PACKET_HASH__", packet_sha256(packet))
    content = content.replace("__PUBLIC_PACKET__", encoded)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>独立诊断读图记录</title>
<style>
:root{font:16px/1.55 system-ui,sans-serif;color:#172536;background:#f1f4f8}body{margin:0}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:1.6rem}h2{font-size:1.25rem}h3{font-size:1rem;margin:10px 0}section,.map{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:18px;margin:16px 0}label{display:inline-block;margin:5px 12px 5px 0}select,input,textarea,button{font:inherit}select,input,textarea{max-width:100%;padding:6px;border:1px solid #789;border-radius:4px}textarea{display:block;width:calc(100% - 14px);min-height:65px}button{padding:6px 12px;border:1px solid #789;border-radius:4px;background:#fff;color:#152536;cursor:pointer}button:focus-visible,select:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid #2458be;outline-offset:2px}.muted{color:#475569;font-size:.9rem}.image{max-width:420px;width:100%;display:block}.overlay{position:relative;width:min(100%,560px);aspect-ratio:1}.overlay>img{width:100%;height:100%;object-fit:fill}.grid{position:absolute;inset:0;display:grid}.cell{padding:0;border:1px solid #ffffffb0;border-radius:0;min-width:0;font-size:11px;font-weight:600;text-shadow:0 1px 1px white;overflow:hidden}.cell span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.cell.unmeasured{background:#ffffff95;border:1px dashed #67798d}.blank{pointer-events:none}.legend{margin:8px 0}.positive{color:#b42318}.negative{color:#175aa8}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:7px;border:1px solid #cbd5e1;text-align:left;vertical-align:top}th{background:#f3f6fa}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:10px}.choices{display:grid;gap:12px}.choice{border:1px solid #cbd5e1;border-radius:6px;padding:12px}.choice input{width:min(95%,380px)}.primary{background:#173f7a;color:white}.error{color:#b42318}.switches{display:flex;flex-wrap:wrap;gap:10px}#status{min-height:1.6em}.detail{min-height:80px}code{overflow-wrap:anywhere}@media(max-width:650px){main{padding:10px}.cell{font-size:9px}}
</style>
<main>
<h1>独立诊断读图记录</h1><p id="identity"></p>
<section><h2>当前问题与可见证据</h2><p id="question"></p><p id="facts"></p><img id="source-image" class="image" alt="当前接收图片；空间索引以模型处理后的图像为准">
<p class="muted">这里只有已测证据。未知检查不显示结果。网格表示被替换的视觉向量位置，并非认定该区域就是污染根因；所有操作只替换内部状态，不修改图片或参数。</p>
<div class="switches"><label for="target">计分答案 <select id="target"></select></label><label for="contrast">相对答案 <select id="contrast"></select></label></div>
<p id="measure"></p><p class="legend"><span class="positive">红：正偏移</span> · <span class="negative">蓝：负偏移</span> · 近白：接近零。深浅只表示本图幅度；不同图颜色不可直接比较。数值为候选支持分数，实际回答另外列出。</p></section>
<div id="maps"></div>
<section><h2>已测操作完整记录</h2><p class="muted">热图版与表格版共享以下相同数值、完整回答和操作坐标。掩码索引从0开始，按24×24视觉网格逐行编号。</p><div id="records" class="scroll"></div></section>
<section><h2>提交尚未测过的检查与预测</h2><p>按你希望执行的先后顺序添加检查。可以预测具体答案、填写候选外的原文，或明确弃权。这里不会即时揭示真实结果。</p>
<label for="reader">读者编号 <input id="reader" required autocomplete="off"></label>
<label for="filter">筛选检查 <input id="filter" type="search" placeholder="操作编号、标签或供体"></label>
<label for="action">未测检查 <select id="action"></select></label><button id="add" type="button">添加检查</button>
<div id="choices" class="choices"></div>
<label for="diagnosis">诊断说明：哪些证据支持什么判断？</label><textarea id="diagnosis"></textarea>
<label for="competitor">竞争解释：还有哪种解释与现有证据相容？</label><textarea id="competitor"></textarea>
<label for="next-check">下一检查：预期怎样的结果能区分这些解释？</label><textarea id="next-check"></textarea>
<p><button id="save" type="button" class="primary">下载提交JSON</button></p><p id="status" role="status" aria-live="polite"></p>
<p class="muted">计时从页面加载开始，包含停留时间。文件只保存你填写的预测、顺序、说明、阅读版本及证据包指纹，不包含隐藏答案。请将下载的JSON交回独立评估器；未提交不算完成读者实验。</p></section>
</main>
<script id="packet-data" type="application/json">__PUBLIC_PACKET__</script>
<script>
"use strict";
const packet=JSON.parse(document.getElementById('packet-data').textContent), mode='__MODE__', packetHash='__PACKET_HASH__';
const started=performance.now(), $=id=>document.getElementById(id), actions=new Map(packet.actions.map(a=>[a.id,a])), chosen=[];
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function option(select,value,text){const o=el('option',text);o.value=value;select.append(o);}
function fmt(value){return Number.isFinite(value)?value.toFixed(6):'未测';}
function score(id){const o=packet.observations[id];if(!o)return null;return o.scores[$('target').value]-($('contrast').value?o.scores[$('contrast').value]:0);}
function effect(id,base){const a=score(id),b=score(base);return a===null||b===null?null:a-b;}
function label(a){return a.id+' · '+(a.donor==='peer'?'异事实供体':'同事实供体')+' · '+a.indices.length+'个状态 · '+a.tags.map(t=>typeof t==='string'?t:JSON.stringify(t)).join(', ')+(a.donor==='same'&&a.tags.includes('coarse_minus_cell')?' · 相对已测粗块，检查删去该格后的变化':'');}
function coords(indices){return indices.length?indices.join(', '):'空集';}
function detailText(id,base){const a=actions.get(id),o=packet.observations[id];let t=label(a)+'\n替换索引：'+coords(a.indices);if(!o)return t+'\n未测，结果已隐藏。';t+='\n完整实际回答：'+o.output.text+'\n结束状态：'+o.output.stop_reason;if(base)t+='\n相对基准的计分偏移：'+effect(id,base);return t+'\n所有候选分数：\n'+packet.candidates.map(c=>c+'：'+o.scores[c]).join('\n');}
$('identity').textContent=packet.unit_id+' · '+packet.cluster_id+' · '+packet.input_condition+' · '+(mode==='heatmap'?'热图叠图版':'数值表格版');
$('question').textContent='问题：'+packet.question;
$('facts').textContent='接收图事实答案：'+packet.truth_answer+'；异事实供体答案：'+(packet.donor_answer??'不可用');
if(packet.image_uri)$('source-image').src=packet.image_uri;else $('source-image').hidden=true;
packet.candidates.forEach(c=>option($('target'),c,c));option($('contrast'),'','不减另一候选（单候选支持）');packet.candidates.forEach(c=>option($('contrast'),c,c));
$('target').value=packet.truth_answer;$('contrast').value=packet.candidates.includes('Unable to answer.')?'Unable to answer.':'';
function drawMaps(){
 $('measure').textContent='每格数值 = 操作后“'+$('target').value+($('contrast').value?' 相对 '+$('contrast').value:'')+'”支持分数 − 同图基准分数。正负不直接代表好坏。';
 $('maps').replaceChildren();
 packet.maps.forEach((m,mi)=>{
  const section=el('section',undefined,'map');section.append(el('h2',m.title+' · '+m.resolution+'×'+m.resolution));
  section.append(el('p','基准操作：'+m.base_id+'；已替换背景索引：'+coords(m.background),'muted'));
  const detail=el('pre','选择格子或“查看完整记录”读取具体条件。','detail');detail.id='detail-'+mi;detail.setAttribute('aria-live','polite');
  if(mode==='heatmap'){
   const overlay=el('div',undefined,'overlay');if(packet.image_uri){const img=el('img');img.src=packet.image_uri;img.alt='接收图与内部视觉状态位置叠图';overlay.append(img);}
   const grid=el('div',undefined,'grid');grid.style.gridTemplateColumns='repeat('+m.resolution+',1fr)';grid.style.gridTemplateRows='repeat('+m.resolution+',1fr)';
   const byIndex=new Map(m.cells.map(c=>[c.index,c])),max=Math.max(0,...m.cells.map(c=>Math.abs(effect(c.action_id,m.base_id)??0)));
   for(let i=0;i<m.resolution*m.resolution;i++){
    const c=byIndex.get(i);if(!c){grid.append(el('span',undefined,'blank'));continue;}
    const v=effect(c.action_id,m.base_id),o=packet.observations[c.action_id],button=el('button',undefined,'cell'+(v===null?' unmeasured':''));button.type='button';
    if(v!==null){const rgb=v>=0?'190,35,28':'25,85,180';button.style.backgroundColor='rgba('+rgb+','+(max?0.76*Math.abs(v)/max:0)+')';}
    button.append(el('span',String(i)),el('span',c.action_id===m.base_id?'背景已包含':fmt(v)),el('span',o?o.output.text:'未测'));
    button.title=(c.action_id===m.base_id?'背景已包含该格，不是新增干预。\n':'')+detailText(c.action_id,m.base_id);button.setAttribute('aria-label','格'+i+'，'+button.title);button.setAttribute('aria-controls',detail.id);button.onclick=()=>{detail.textContent=detailText(c.action_id,m.base_id);};grid.append(button);
   }
   overlay.append(grid);section.append(overlay,detail);
  }
  const table=el('table'),head=el('tr');['格号（行,列）','操作与掩码','完整实际回答','相对基准偏移','完整记录'].forEach(x=>head.append(el('th',x)));table.append(head);
  m.cells.forEach(c=>{const a=actions.get(c.action_id),o=packet.observations[c.action_id],row=el('tr');row.append(el('td',c.index+' ('+Math.floor(c.index/m.resolution)+','+c.index%m.resolution+')'+(c.action_id===m.base_id?' 背景已包含':'')),el('td',label(a)+'\n格子索引：'+coords(c.indices)),el('td',o?o.output.text:'未测'),el('td',fmt(effect(c.action_id,m.base_id))));const td=el('td'),d=el('details');d.append(el('summary','查看完整记录'),el('pre',detailText(c.action_id,m.base_id)));td.append(d);row.append(td);table.append(row);});
  const wrap=el('div',undefined,'scroll');wrap.append(table);section.append(wrap);$('maps').append(section);
 });
}
function drawRecords(){const table=el('table'),head=el('tr');['操作/供体','24×24替换索引','完整实际回答','结束状态',...packet.candidates.map(c=>'候选：'+c)].forEach(x=>head.append(el('th',x)));table.append(head);packet.actions.filter(a=>a.known).forEach(a=>{const o=packet.observations[a.id],row=el('tr');[label(a),coords(a.indices),o.output.text,o.output.stop_reason,...packet.candidates.map(c=>o.scores[c])].forEach(x=>row.append(el('td',x)));table.append(row);});$('records').replaceChildren(table);}
$('target').onchange=drawMaps;$('contrast').onchange=drawMaps;drawMaps();drawRecords();
function listActions(){const term=$('filter').value.toLowerCase();$('action').replaceChildren();packet.actions.filter(a=>!a.known&&!chosen.some(c=>c.id===a.id)&&label(a).toLowerCase().includes(term)).forEach(a=>option($('action'),a.id,label(a)));$('add').disabled=!$('action').options.length;}
function drawChoices(){
 $('choices').replaceChildren();chosen.forEach((choice,i)=>{
  const a=actions.get(choice.id),box=el('div',undefined,'choice');box.append(el('h3',(i+1)+'. '+label(a)),el('p','替换索引：'+coords(a.indices),'muted'));
  const select=el('select');select.id='answer-'+i;option(select,'abstain','弃权／尚无法判断');packet.candidates.forEach((c,j)=>option(select,'candidate-'+j,c));option(select,'other','OTHER：填写具体答案原文');select.value=choice.kind;
  const l=el('label','预测完整答案 ');l.htmlFor=select.id;l.append(select);box.append(l);
  const other=el('input');other.type='text';other.id='other-'+i;other.value=choice.other;other.placeholder='候选外的具体答案';other.hidden=choice.kind!=='other';other.setAttribute('aria-label','第'+(i+1)+'项候选外答案原文');other.oninput=()=>choice.other=other.value;
  select.onchange=()=>{choice.kind=select.value;other.hidden=choice.kind!=='other';};box.append(other);
  [['上移',-1],['下移',1],['移除',0]].forEach(([name,offset])=>{const b=el('button',name);b.type='button';b.setAttribute('aria-label',name+'第'+(i+1)+'项检查');b.disabled=offset!==0&&(i+offset<0||i+offset>=chosen.length);b.onclick=()=>{if(offset===0)chosen.splice(i,1);else [chosen[i],chosen[i+offset]]=[chosen[i+offset],chosen[i]];drawChoices();listActions();};box.append(b);});$('choices').append(box);
 });
}
$('filter').oninput=listActions;$('add').onclick=()=>{const id=$('action').value;if(id&&actions.has(id)&&!actions.get(id).known&&!chosen.some(c=>c.id===id)){chosen.push({id,kind:'abstain',other:''});drawChoices();listActions();}};listActions();
function buildSubmission(choices,reader,diagnoses,elapsed){
 reader=reader.trim();if(!reader)throw Error('请填写读者编号。');const predictions={},seen=new Set();
 choices.forEach(c=>{
  if(!actions.has(c.id)||actions.get(c.id).known||seen.has(c.id))throw Error('检查必须为不重复的未知操作。');seen.add(c.id);
  if(c.kind==='abstain')predictions[c.id]=null;
  else if(c.kind==='other'){const answer=c.other.trim();if(!answer||answer==='OTHER')throw Error('OTHER必须填写具体答案，无法判断请选择弃权。');predictions[c.id]=answer;}
  else {const index=Number(c.kind.slice(10));if(!/^candidate-\d+$/.test(c.kind)||!packet.candidates[index])throw Error('请选择有效候选或弃权。');predictions[c.id]=packet.candidates[index];}
 });
 return {unit_id:packet.unit_id,method:'reader',packet_sha256:packetHash,predictions,order:choices.map(c=>c.id),reader_id:reader,mode,diagnoses,elapsed_seconds:elapsed};
}
$('save').onclick=()=>{
 try{
  const reader=$('reader').value.trim();
  const result=buildSubmission(chosen,reader,{description:$('diagnosis').value,competing_explanation:$('competitor').value,next_check:$('next-check').value},(performance.now()-started)/1000);
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),link=el('a');link.href=url;link.download=(packet.unit_id+'-'+mode+'-'+reader).replace(/[^a-zA-Z0-9_.-]/g,'_')+'.json';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);$('status').className='';$('status').textContent='提交JSON已生成；请确认文件已下载。页面未揭示任何隐藏结果。';
 }catch(error){$('status').className='error';$('status').textContent=error.message;}
};
</script></html>
'''
