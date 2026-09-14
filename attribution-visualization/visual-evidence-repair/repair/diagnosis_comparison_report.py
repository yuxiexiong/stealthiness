"""Saved-data comparison atlas and isolated, offline H/T reader packets.

No model calls. Freeze ``make_reader_assignments`` before observing new outcomes.
Reader input is explicitly allowlisted by case.reader.initial_condition_keys;
reserved checks contain operation definitions, never their measured outcomes.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import random

from . import visual_probe_report as atlas
from .visual_probe_protocol import condition_key, _mask


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def make_reader_assignments(cases, seed=20260914):
    """Balanced incomplete crossover: 16 families, 8 readers, 64 readings.

    Only IDs/family IDs are read. Persist this object before first-stage outcomes.
    Cyclic reading orders preserve 4 H/4 T at every reading position.
    """
    if len(cases) < 16:
        raise ValueError("Reader allocation requires at least 16 independent cases")
    ids = [str(c["cluster_id"]) for c in cases]
    families = [str(c.get("family_id", c["cluster_id"])) for c in cases]
    if len(set(ids)) != len(ids) or len(set(families)) != len(families):
        raise ValueError("Distinct case and family IDs required; variants are one family")
    rng = random.Random(seed)
    selected = rng.sample(ids, 16)
    parity = rng.randrange(2)
    order = rng.sample(list(range(parity, 8, 2)), 4) + rng.sample(list(range(1-parity, 8, 2)), 4)
    readers = []
    reader_numbers = rng.sample(list(range(1, 9)), 8)
    for r in range(8):
        trials = []
        for period, column in enumerate(order, 1):
            j = (column+r) % 8
            trials.append({"trial_id": f"reader-{reader_numbers[r]:02d}-trial-{period:02d}",
                           "cluster_id": selected[(2*r+j) % 16],
                           "format": "H" if j in (0, 1, 4, 5) else "T",
                           "period": period, "block": 1 if period <= 4 else 2})
        readers.append({"reader_id": f"reader-{reader_numbers[r]:02d}", "trials": trials})
    result = {"schema": "diagnosis-reader-assignment-v1", "seed": seed,
              "status": "materials_only_not_executed", "selected_case_ids": selected,
              "family_ids": dict(zip(ids, families)), "readers": readers}
    result["allocation_sha256"] = sha256(_encoded(result).encode()).hexdigest()
    return result


def _validate_assignment(assignments, cases):
    if assignments.get("schema") != "diagnosis-reader-assignment-v1":
        raise ValueError("Use a frozen diagnosis-reader-assignment-v1 allocation")
    readers = assignments.get("readers", [])
    if len(readers) != 8 or len({r["reader_id"] for r in readers}) != 8:
        raise ValueError("Exactly eight distinct readers required")
    counts, periods, trial_ids = {}, {}, set()
    for reader in readers:
        trials = reader["trials"]
        if len(trials) != 8 or sorted(t["period"] for t in trials) != list(range(1, 9)):
            raise ValueError("Each reader needs eight distinct reading periods")
        if sorted(t["format"] for t in trials) != ["H"]*4 + ["T"]*4:
            raise ValueError("Each reader must receive four H and four T cases")
        family = []
        for trial in trials:
            cid = trial["cluster_id"]
            if cid not in cases or trial["trial_id"] in trial_ids:
                raise ValueError("Unknown case or duplicate trial ID")
            if not all(ch.isalnum() or ch in "-_" for ch in trial["trial_id"]):
                raise ValueError("Trial IDs must be safe filenames")
            trial_ids.add(trial["trial_id"])
            family.append(str(cases[cid].get("family_id", cid)))
            counts.setdefault(cid, []).append(trial["format"])
            periods.setdefault(trial["period"], []).append(trial["format"])
        if len(set(family)) != 8:
            raise ValueError("A reader must never repeat a scene family")
    if len(counts) != 16 or any(sorted(v) != ["H", "H", "T", "T"] for v in counts.values()):
        raise ValueError("Each of 16 scenes must have two H and two T readers")
    if any(sorted(v) != ["H"]*4 + ["T"]*4 for v in periods.values()):
        raise ValueError("Reading positions must be balanced across H and T")


def _operation(operation, nodes):
    fields = ("id", "label", "role", "kind", "node_ids", "background", "indices", "donor_ids",
              "result_existed_at_registration")
    result = {k: deepcopy(operation[k]) for k in fields if k in operation}
    if not isinstance(result.get("id"), str) or not result["id"].strip():
        raise ValueError("Every public operation needs a nonempty ID")
    result["background"] = _mask(result.get("background", []))
    result["indices"] = _mask(result.get("indices", []))
    ids = result.get("node_ids", [])
    if not ids or len(set(ids)) != len(ids) or any(n not in nodes for n in ids):
        raise ValueError("Operation must identify existing question nodes")
    if any(n not in nodes or d not in nodes for n, d in result.get("donor_ids", {}).items()):
        raise ValueError("Public donor must be an existing same-case node")
    return result


def _public_condition(source):
    result = {k: deepcopy(source[k]) for k in ("status", "fact", "refusal", "correct", "donor_id", "indices", "key") if k in source}
    result["output"] = {k: deepcopy(source.get("output", {})[k]) for k in
                        ("text", "stop_reason", "truncated", "new_tokens", "generated_tokens")
                        if k in (source.get("output") or {})}
    result["scores"] = {name: {k: deepcopy(score[k]) for k in ("text", "log_prob", "token_ids", "token_log_probs") if k in score}
                        for name, score in source.get("scores", {}).items() if name in ("positive", "negative", "refusal")}
    return result


def reader_case(case):
    """Copy only public initial evidence, not the full report with hidden CSS."""
    metadata = case.get("reader", {})
    allowed = metadata.get("initial_condition_keys")
    if not isinstance(allowed, dict):
        raise ValueError("Reader packets require explicit initial_condition_keys")
    nodes = {n["id"]: n for n in case["nodes"]}
    reserved = [_operation(op, nodes) for op in metadata.get("reserved_checks", [])]
    if not 2 <= len(reserved) <= 4 or len({op["id"] for op in reserved}) != len(reserved):
        raise ValueError("Reader trial requires two to four unique reserved operation definitions")
    private_keys = set()
    for op in reserved:
        if op.get("role") != "baseline":
            for node in op["node_ids"]:
                private_keys.add(condition_key(node, op.get("donor_ids", {}).get(node, node),
                                               sorted(set(op["background"]) | set(op["indices"]))))
    result = {"cluster_id": case["cluster_id"], "annotations": {
        k: deepcopy(case.get("annotations", {})[k]) for k in ("marker_box", "pixel_size", "source")
        if k in case.get("annotations", {})}, "nodes": []}
    for node_id, source in nodes.items():
        keys = set(allowed.get(node_id, []))
        if keys & private_keys:
            raise ValueError("Reserved outcome was included in initial reader evidence")
        if not keys <= source.get("conditions", {}).keys():
            raise ValueError("Initial evidence names a missing saved condition")
        node = {k: deepcopy(source[k]) for k in ("id", "label", "question", "answer", "grid", "candidates", "tolerance", "images", "annotations") if k in source}
        node["images"] = {k: deepcopy(source.get("images", {})[k]) for k in ("clean", "observed", "coordinate_system") if k in source.get("images", {})}
        annotations = source.get("annotations", {})
        node["annotations"] = {k: deepcopy(annotations[k]) for k in ("pixel_size", "source") if k in annotations}
        node["annotations"]["objects"] = [{k: deepcopy(obj[k]) for k in ("object_id", "box") if k in obj} for obj in annotations.get("objects", [])]
        node["conditions"] = {k: _public_condition(source["conditions"][k]) for k in keys}
        node["maps"] = []
        for source_map in source.get("maps", []):
            if source_map.get("resolution") not in (2, 6) or source_map.get("base_key") not in keys:
                continue
            mapping = {k: deepcopy(source_map[k]) for k in ("id", "title", "resolution", "background", "base_key", "parent_index") if k in source_map}
            mapping["cells"] = []
            for source_cell in source_map.get("cells", []):
                cell = {k: deepcopy(source_cell[k]) for k in ("index", "indices", "key") if k in source_cell}
                cell["status"] = ("included" if source_cell.get("status") == "included" else
                                  node["conditions"].get(cell.get("key"), {}).get("status", "unmeasured"))
                mapping["cells"].append(cell)
            node["maps"].append(mapping)
        result["nodes"].append(node)
    public = {"task": str(metadata.get("task", "任务待登记")), "reserved_checks": reserved,
              "node_ids": metadata.get("node_ids", [n["id"] for n in case["nodes"][:2]]),
              "donor_options": {}}
    if len(public["node_ids"]) != 2 or any(n not in nodes for n in public["node_ids"]):
        raise ValueError("Reader online operations must check exactly two existing question nodes")
    for node in public["node_ids"]:
        options = metadata.get("donor_options", {}).get(node, [{"id": node, "label": "本节点正常供体"}])
        if not options or any(o.get("id") not in nodes for o in options):
            raise ValueError("Donor options must reference saved, qualified same-case nodes")
        public["donor_options"][node] = [{"id": o["id"], "label": str(o.get("label", o["id"]))} for o in options]
    return result, public


def _enhance(path, *, reader=None, extra=""):
    document = path.read_text(encoding="utf-8")
    document = document.replace("</style>", EXTRA_STYLE + "</style>", 1)
    if reader:
        document = document.replace("<body>", '<body class="reader reader-' + reader["format"] + '">', 1)
        document = document.replace("原始异常案例已揭晓；新操作只能提供病例内前瞻检验。", "这里只展示共同初始证据；在线反馈与保留验收分别记录。")
        document = document.replace("<h1>VLM 归因可视化 · 病例图册</h1>", "<h1>诊断阅读材料 · " + escape(reader["trial_id"]) + "</h1>")
        document = document.replace(atlas._saved("读新图前的判断与下一检查", None), "")
        extra += '<section id="reader-work" aria-label="诊断记录"></section>'
        extra += '<script type="application/json" id="reader-data">' + _encoded(reader) + '</script>'
    else:
        extra = '<p class="toolbar"><label>呈现 <select id="presentation"><option value="H">热图＋完整表格</option><option value="T">中性网格＋同一表格</option></select></label></p>' + extra
    document = document.replace('<div class="workspace">', extra + '<div class="workspace" id="initial-evidence">', 1)
    document = document.replace("</body>", "<script>" + TABLE_SCRIPT + (READER_SCRIPT if reader else "") + "</script></body>")
    path.write_text(document, encoding="utf-8")


def render(report, output):
    """Full investigator atlas. All actual maps remain exact; no interpolation."""
    snapshot = deepcopy(report)
    snapshot["schema"] = "visual-probe-v3"
    output = Path(output)
    result = atlas.render(snapshot, output)
    extra = '<section><h2>诊断方法比较</h2><p>仅展示保存结果；未执行、未决与失败保留。H/T真实读者阶段不由本报告模拟。</p>'
    for case in snapshot.get("cases", []):
        extra += atlas._saved(str(case["cluster_id"]) + " · 每任务证据、保留预测和成本", case.get("comparison", {}))
    for key, label in (("comparison", "总体比较"), ("external", "外部原方法处理参照"), ("reader_results", "真实读者结果")):
        extra += atlas._saved(label, snapshot.get(key, {"status": "not_executed"}))
    _enhance(output, extra=extra + '</section>')
    return result


def render_reader_materials(report, assignments, output_dir):
    """Write 64 separate trials; no outcome-bearing master report is embedded."""
    cases = {c["cluster_id"]: c for c in report["cases"]}
    _validate_assignment(assignments, cases)
    public = {cid: reader_case(cases[cid]) for cid in assignments["selected_case_ids"]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assignment.json").write_text(atlas._json(assignments), encoding="utf-8")
    links = []
    for reader in assignments["readers"]:
        trials = []
        for trial in sorted(reader["trials"], key=lambda t: t["period"]):
            case, metadata = public[trial["cluster_id"]]
            packet = dict(metadata, **trial, reader_id=reader["reader_id"],
                          allocation_sha256=assignments.get("allocation_sha256"),
                          schema="diagnosis-reader-packet-v1", status="materials_only_not_executed")
            path = output_dir / (trial["trial_id"] + ".html")
            snapshot = {"schema": "visual-probe-v3", "status": "reader_materials_not_executed",
                        "cpu_test": report.get("cpu_test", False), "cases": [case],
                        "scope": "先登记初始预测，再查询；保留结果不在页面中。每例有效阅读上限15分钟，模型等待另计。"}
            atlas.render(snapshot, path)
            _enhance(path, reader=packet)
            trials.append(f'<li>第 {trial["period"]} 例 · 第 {trial["block"]} 段：<a href="{path.name}">{escape(trial["trial_id"])}</a></li>')
        index = output_dir / (reader["reader_id"] + ".html")
        index.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阅读顺序</title><h1>' + escape(reader["reader_id"]) +
                         '</h1><p>材料已准备，真实阅读尚未执行。按顺序阅读，前四例完成后统一休息；不要打开其他读者材料。每例结束导出记录。</p><ol>' + ''.join(trials) + '</ol></html>', encoding="utf-8")
        links.append(f'<li><a href="{index.name}">{escape(reader["reader_id"])}</a></li>')
    (output_dir / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>读者材料分发</title><h1>H/T 阅读材料</h1><p>状态：未执行。仅向各读者分发自己的页面；本页供主持人使用。</p><ul>' + ''.join(links) + '</ul></html>', encoding="utf-8")
    return {"output": str(output_dir.resolve()), "readers": 8, "trials": 64, "status": "not_executed", "model_calls": 0}


EXTRA_STYLE = """
.neutral g.heat-cell .cell-fill,.reader-T g.heat-cell .cell-fill{fill:#e2e5e8!important;fill-opacity:.18!important}
.neutral .legend .positive,.neutral .legend .negative,.reader-T .legend .positive,.reader-T .legend .negative{background:#eee;color:#111}
.evidence-table{overflow:auto;background:white;border:1px solid #ccd3dd;padding:10px;margin:12px 0}.evidence-table table{width:100%;border-collapse:collapse;font-size:12px}.evidence-table th,.evidence-table td{border:1px solid #ddd;padding:6px;white-space:pre-wrap}.evidence-table tr.active{outline:2px solid #bf8400}.evidence-tools{display:flex;gap:12px;flex-wrap:wrap}.reader #case-select{pointer-events:none}.reader .scope{font-weight:bold}.reader #reader-work{border:2px solid #64748b;padding:16px;background:white;margin:16px 0}.reader fieldset{margin:12px 0;border:1px solid #cbd5e1}.reader label{display:block;margin:7px 0}.reader textarea{width:95%;min-height:64px;font:inherit}.reader input{font:inherit;padding:5px}.reader .prediction-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.reader .prediction-row input{flex:1;min-width:130px}.reader .notice{color:#8a3b00;background:#fff3d7;padding:10px}.reader .inline{display:inline-block;margin-right:12px}.reader .query-card{padding:10px;margin:10px 0;border:1px solid #b9c5d4}.reader .timer{font-weight:bold;font-variant-numeric:tabular-nums}.reader button:disabled{cursor:not-allowed;opacity:.55}.reader .waiting{border-left:5px solid #ba6f0b}.reader .returned{border-left:5px solid #285f50}
"""


TABLE_SCRIPT = r"""
function evidenceTables(){
  document.querySelectorAll('.case').forEach(section=>{
    const ci=Number(section.dataset.case), rows=payload.cells.map((r,i)=>[r,i]).filter(([r])=>r.case===ci);
    const panel=document.createElement('div');panel.className='evidence-table';
    panel.innerHTML='<h3>共同数值表与实际回答</h3><div class="evidence-tools"><label>筛选 <input class="filter" placeholder="问题、背景、格号或回答"></label><label>排序 <select class="sort"><option value="original">原记录顺序</option><option value="desc">差值由高到低</option><option value="asc">差值由低到高</option></select></label></div><div class="table-content"></div>';
    section.append(panel);
    const gallery=document.createElement('details');gallery.innerHTML='<summary>各问题的接收图片与正常供体</summary><div class="inputs">'+report.cases[ci].nodes.map(n=>'<figure>'+['observed','clean'].map(kind=>{const src=n.images?.[kind];return /^data:image\/(png|jpeg|webp|gif);base64,/.test(src??'')?'<img alt="'+esc((n.label??n.id)+' '+kind)+'" src="'+esc(src)+'">':'<p>图片未保存</p>';}).join('')+'<figcaption>'+esc(n.label??n.id)+'：上为接收图，下为本节点正常图</figcaption></figure>').join('')+'</div>';section.prepend(gallery);
    function draw(){
      const filter=panel.querySelector('.filter').value.toLowerCase(), order=panel.querySelector('.sort').value;
      let items=rows.slice();if(order!=='original')items.sort((a,b)=>{const x=a[0].values[metric.value],y=b[0].values[metric.value];return !finite(x)?1:!finite(y)?-1:(order==='desc'?y-x:x-y);});
      panel.querySelector('.table-content').innerHTML='<table><thead><tr><th>问题／地图／格</th><th>区域、背景与供体</th><th>基准实际回答</th><th>操作后实际回答</th><th>差值 nat／状态</th></tr></thead><tbody>'+items.map(([r,i])=>{
        const n=report.cases[r.case].nodes[r.node],b=n.conditions?.[r.map.base_key],a=n.conditions?.[r.cell.key];
        const texts=[(n.label??n.id)+' / '+(r.map.title??r.map.id)+' / '+r.cell.index,'区域 '+JSON.stringify(r.cell.indices)+'\n背景 '+JSON.stringify(r.map.background)+'\n供体 '+(a?.donor_id??'未记录'),b?.output?.text??'未生成',a?.output?.text??'未生成',number(r.values[metric.value])+' / '+r.cell.status];
        if(!texts.join(' ').toLowerCase().includes(filter))return '';
        return '<tr data-row="'+i+'"><td><button type="button" data-focus="'+i+'">'+esc(texts[0])+'</button></td>'+texts.slice(1).map(t=>'<td>'+esc(t)+'</td>').join('')+'</tr>';
      }).join('')+'</tbody></table>';
      panel.querySelectorAll('[data-focus]').forEach(b=>b.onclick=()=>{show(Number(b.dataset.focus));document.querySelector('[data-cell="'+b.dataset.focus+'"]').scrollIntoView({block:'center',behavior:'smooth'});});
    }
    panel.querySelector('.filter').oninput=draw;panel.querySelector('.sort').onchange=draw;metric.addEventListener('change',draw);draw();
  });
}
evidenceTables();
document.getElementById('presentation')?.addEventListener('change',e=>document.body.classList.toggle('neutral',e.target.value==='T'));
"""


READER_SCRIPT = r"""
const trial=JSON.parse(document.getElementById('reader-data').textContent), work=document.getElementById('reader-work');
const readerNodes=Object.fromEntries(report.cases[0].nodes.map(n=>[n.id,n]));
const storageKey='diagnosis-reader-v1:'+trial.allocation_sha256+':'+trial.trial_id;
let state={schema:'diagnosis-reader-record-v1',reader_id:trial.reader_id,trial_id:trial.trial_id,cluster_id:trial.cluster_id,format:trial.format,allocation_sha256:trial.allocation_sha256,status:'not_started',initial_prediction:null,requests:[],diagnosis:{},reserved_prediction:null,active_ms:0,waiting_ms:0,events:[]};
try{const saved=localStorage.getItem(storageKey);if(saved)state=JSON.parse(saved);}catch(e){}
let lastTick=Date.now(),lastHidden=document.hidden,consented=false;
const stamp=()=>new Date().toISOString();
function event(type,extra={}){state.events.push({type,at:stamp(),...extra});save();}
function save(){try{localStorage.setItem(storageKey,JSON.stringify(state));}catch(e){document.getElementById('save-status').textContent='本浏览器无法自动保存，请立即导出记录。';}}
function download(value,name){const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);}
function predictionFields(checks,prefix){return checks.map((op,i)=>'<div class="query-card"><b>'+esc(op.label??op.id)+'</b>'+dump({id:op.id,background:op.background,indices:op.indices,donor_ids:op.donor_ids??{},result_existed_at_registration:op.result_existed_at_registration??'未登记'})+op.node_ids.map((node,j)=>'<label class="prediction-row">'+esc(readerNodes[node].label??node)+' <input data-prediction="'+prefix+'" data-operation="'+i+'" data-node="'+j+'" placeholder="预测实际回答，无法预测请写未决" required></label>').join('')+'</div>').join('');}
function predictions(prefix,checks){const result={};work.querySelectorAll('[data-prediction="'+prefix+'"]').forEach(input=>{const op=checks[Number(input.dataset.operation)],node=op.node_ids[Number(input.dataset.node)];if(!input.value.trim())throw Error('请填写所有预测，或明确填写未决');(result[op.id]??={})[node]={answer:input.value.trim()};});return result;}
function restorePredictions(prefix,checks,values){work.querySelectorAll('[data-prediction="'+prefix+'"]').forEach(input=>{input.value=values?.[checks[Number(input.dataset.operation)].id]?.[checks[Number(input.dataset.operation)].node_ids[Number(input.dataset.node)]]?.answer??'';});}
const initialChecks=trial.reserved_checks.slice(0,2);
work.innerHTML='<h2>本例任务：'+esc(trial.task)+'</h2><p class="notice">材料准备不代表读者实验已执行。首次查询前锁定初始预测；在线最多4个操作条件，两题一起检查。保留验收最多4个条件，结果不在本页面。导出请求并等待测量人员返回反馈JSON，不会自行调用模型。</p><p class="timer" id="timer">有效阅读 00:00 / 15:00</p><label><input type="checkbox" id="consent"> 我同意记录本例诊断、预测、操作和时间</label><button id="start-trial">开始本例／继续阅读</button><button id="pause-trial">暂停阅读</button><button id="export-record">导出完整记录</button><label class="inline">恢复记录 <input type="file" id="restore-record" accept=".json"></label><span id="save-status"></span><fieldset id="initial-stage"><legend>1. 首次查询前的共同预测</legend>'+predictionFields(initialChecks,'initial')+'<button id="lock-initial">锁定初始预测</button></fieldset><fieldset id="query-stage"><legend>2. 在线检查（最多4个条件）</legend><label>操作类型 <select id="operation-kind"><option value="joint">单格／联合／去除后的集合</option><option value="refine">一个6×6父格的四个12×12子格（4次）</option></select></label><label>区域尺度 <select id="region-resolution"><option>6</option><option>2</option><option>12</option></select> 格号 <input id="region-cells" placeholder="如 0,1,6,7"></label><label>背景尺度 <select id="background-resolution"><option>2</option><option>6</option></select> 格号 <input id="background-cells" placeholder="空白为空背景；0为左上粗块"></label><p>格号从0起，与图表相同。联合操作把多个格作为一个条件；去除操作填写保留下来的格。细化时仅填写一个6×6父格。</p>'+trial.node_ids.map((n,i)=>'<label>'+esc(readerNodes[n].label??n)+' 的正常供体 <select id="donor-'+i+'">'+trial.donor_options[n].map(d=>'<option value="'+esc(d.id)+'">'+esc(d.label)+'</option>').join('')+'</select></label>').join('')+'<label>检查理由／竞争解释 <textarea id="query-reason"></textarea></label><button id="prepare-query">准备并核对请求</button><div id="query-preview"></div><label>导入当前请求的反馈 <input type="file" id="feedback-file" accept=".json"></label><div id="online-results"></div></fieldset><fieldset id="final-stage"><legend>3. 最终诊断与保留预测</legend><label>诊断状态 <select id="diagnosis-status"><option value="judgment">提交有限关系判断</option><option value="undecided">仍未决</option></select></label>'+[['observation','观察'],['judgment','具体判断或未决原因'],['scope','适用的区域、问题、背景和供体'],['evidence','已有证据及在线请求编号'],['alternatives','竞争解释'],['falsifier','什么结果会推翻判断'],['action','下一步保留或放弃什么检查']].map(([key,label])=>'<label>'+label+'<textarea data-diagnosis="'+key+'"></textarea></label>').join('')+predictionFields(trial.reserved_checks,'reserved')+'<button id="submit-final">锁定并导出最终记录</button></fieldset><p id="reader-message" role="status" aria-live="polite"></p>';
function message(text){document.getElementById('reader-message').textContent=text;}
function run(action){try{action();}catch(e){message(e.message);}}
function totalOps(){return state.requests.reduce((n,r)=>n+r.operations.length,0);}
function pending(){return state.requests.some(r=>!r.feedback);}
function finished(){return ['submitted','timeout'].includes(state.status);}
work.querySelectorAll('fieldset').forEach(field=>{const box=document.createElement('details'),summary=document.createElement('summary');summary.textContent=field.querySelector('legend').textContent;field.before(box);box.append(summary,field);});
const evidenceLink=document.createElement('p');evidenceLink.innerHTML='<a href="#initial-evidence">查看初始图片、地图和共同数值表 ↓</a>；需要登记时展开上方对应步骤。';work.prepend(evidenceLink);
const onlineCells=new Map();
function onlineMap(row,op){
  const identity=op.id+'|'+row.node_id,nodeIndex=report.cases[0].nodes.findIndex(n=>n.id===row.node_id),node=readerNodes[row.node_id];
  let index=onlineCells.get(identity);
  if(index===undefined){const base='reader-base:'+identity,key='reader-after:'+identity;node.conditions[base]=row.before;node.conditions[key]=row.condition;const measured=row.before?.status==='measured'&&row.condition?.status==='measured';index=payload.cells.length;payload.cells.push({case:0,node:nodeIndex,map:{title:'在线检查 '+op.id,base_key:base,background:op.background},cell:{index:op.id,indices:op.indices,key,status:row.condition.status},values:Object.fromEntries(['fact','refusal'].map(m=>[m,measured&&finite(row.before?.[m])&&finite(row.condition?.[m])?row.condition[m]-row.before[m]:null]))});onlineCells.set(identity,index);}
  const src=node.images?.observed,image=/^data:image\/(png|jpeg|webp|gif);base64,/.test(src??'')?'<image href="'+esc(src)+'" width="24" height="24"/>':'';
  return '<div class="map"><p>本次整组实测差值；所画区域共用该组结果，不表示逐小格分别测量。初始固定色尺，超范围饱和显示；精确值保留。</p><svg viewBox="0 0 24 24" aria-label="在线实际操作区域">'+image+'<g class="heat-cell" tabindex="0" role="button" data-cell="'+index+'" data-online-cell="'+index+'">'+op.indices.map(i=>'<rect class="cell-fill" x="'+(i%24)+'" y="'+Math.floor(i/24)+'" width="1" height="1"/>').join('')+'<text class="cell-value" x="12" y="12" text-anchor="middle" font-size="1">—</text></g></svg></div>';
}
function update(drawResults=true){
  const started=state.status==='reading',active=started&&consented&&!finished();
  document.getElementById('initial-stage').disabled=!active||!!state.initial_prediction;
  document.getElementById('query-stage').disabled=!active||!state.initial_prediction;
  document.getElementById('final-stage').disabled=!active||!state.initial_prediction||pending();
  document.getElementById('prepare-query').disabled=totalOps()>=4||pending();
  document.getElementById('feedback-file').disabled=!pending();
  const s=Math.floor(state.active_ms/1000);document.getElementById('timer').textContent='有效阅读 '+String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')+' / 15:00；等待 '+Math.floor(state.waiting_ms/1000)+'秒；在线 '+totalOps()+'/4；'+state.status;
  if(!drawResults)return;
  document.getElementById('online-results').innerHTML=state.requests.map(r=>'<article class="query-card '+(r.feedback?'returned':'waiting')+'"><b>'+esc(r.request_id)+'</b><p>'+(r.feedback?'已收到本次实际测量反馈':'等待反馈：页面尚无本次结果')+'</p>'+r.operations.map(op=>'<details><summary>'+esc(op.id)+'</summary>'+dump(op)+'</details>').join('')+(r.feedback?r.feedback.results.map(row=>'<div><b>'+esc(readerNodes[row.node_id].label??row.node_id)+'</b>'+generated(row.before,'基准实际回答')+generated(row.condition,'本次操作实际回答')+'<p>事实差值 '+number(finite(row.before?.fact)&&finite(row.condition?.fact)?row.condition.fact-row.before.fact:null)+' nat；拒答差值 '+number(finite(row.before?.refusal)&&finite(row.condition?.refusal)?row.condition.refusal-row.before.refusal:null)+' nat；登记时结果已存在：'+esc(row.result_existed_at_registration??'未记录')+'</p>'+onlineMap(row,r.operations.find(op=>op.id===row.operation_id))+'</div>').join(''):'<button type="button" data-reexport="'+esc(r.request_id)+'">重新导出该请求</button>')+'</article>').join('');
  work.querySelectorAll('[data-reexport]').forEach(b=>b.onclick=()=>exportRequest(state.requests.find(r=>r.request_id===b.dataset.reexport)));
  work.querySelectorAll('[data-online-cell]').forEach(el=>{el.onclick=()=>show(Number(el.dataset.onlineCell));el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(Number(el.dataset.onlineCell));}};});refresh();
}
function exportRequest(request){download({schema:'diagnosis-reader-request-v1',reader_id:trial.reader_id,trial_id:trial.trial_id,cluster_id:trial.cluster_id,allocation_sha256:trial.allocation_sha256,request_id:request.request_id,created_utc:request.created_utc,initial_prediction:state.initial_prediction,operations:request.operations,reason:request.reason},request.request_id+'.json');}
function cellNumbers(text,resolution,empty=false){if(!text.trim()&&empty)return [];const cells=text.split(',').map(v=>Number(v.trim()));if(!text.trim()||cells.some(v=>!Number.isInteger(v)||v<0||v>=resolution*resolution)||new Set(cells).size!==cells.length)throw Error('格号应是合法且不重复的整数，用逗号分隔');return cells;}
function tokens(cells,resolution){const width=24/resolution,result=[];for(const cell of cells){const row=Math.floor(cell/resolution),col=cell%resolution;for(let r=row*width;r<(row+1)*width;r++)for(let c=col*width;c<(col+1)*width;c++)result.push(r*24+c);}return [...new Set(result)].sort((a,b)=>a-b);}
let proposed=[];
document.getElementById('start-trial').onclick=()=>run(()=>{if(!document.getElementById('consent').checked)throw Error('开始记录前需同意收集');if(finished())throw Error('本例已锁定或超时，不能重开');consented=true;state.status='reading';state.started_utc??=stamp();lastTick=Date.now();event('start_or_resume');update();});
document.getElementById('pause-trial').onclick=()=>{if(!finished()){state.status='paused';event('pause');update();}};
document.getElementById('export-record').onclick=()=>{captureDraft();download(state,trial.trial_id+'-record.json');};
document.getElementById('lock-initial').onclick=()=>run(()=>{state.initial_prediction={locked_utc:stamp(),predictions:predictions('initial',initialChecks)};event('initial_prediction_locked');update();});
document.getElementById('prepare-query').onclick=()=>run(()=>{
  const refine=document.getElementById('operation-kind').value==='refine',resolution=refine?6:Number(document.getElementById('region-resolution').value),cells=cellNumbers(document.getElementById('region-cells').value,resolution),br=Number(document.getElementById('background-resolution').value),background=tokens(cellNumbers(document.getElementById('background-cells').value,br,true),br);
  if(refine&&cells.length!==1)throw Error('局部细化一次只能选一个父格');
  if(refine&&state.requests.some(r=>r.operations.some(o=>o.kind==='refine_child')))throw Error('每例最多细化一个父格');
  const groups=refine?[0,1,12,13].map(offset=>tokens([Math.floor(cells[0]/6)*24+(cells[0]%6)*2+offset],12)):[tokens(cells,resolution)];
  if(totalOps()+groups.length>4)throw Error('超出4个在线操作条件');
  const donors=Object.fromEntries(trial.node_ids.map((n,i)=>[n,document.getElementById('donor-'+i).value]));
  proposed=groups.map((indices,i)=>({id:trial.trial_id+'-op-'+(totalOps()+i+1),kind:refine?'refine_child':'joint',node_ids:trial.node_ids,background,indices,donor_ids:donors}));
  for(const op of proposed){const all=[...new Set([...op.background,...op.indices])].sort((a,b)=>a-b);if(!all.length)throw Error('操作集合为空');if(trial.reserved_checks.some(h=>JSON.stringify([...new Set([...h.background,...h.indices])].sort((a,b)=>a-b))===JSON.stringify(all)&&op.node_ids.every(n=>(op.donor_ids[n]??n)===(h.donor_ids?.[n]??n))))throw Error('该条件是保留验收，不能在线查询');}
  document.getElementById('query-preview').innerHTML='<h3>待锁定操作（每项两题）</h3>'+predictionFields(proposed,'online')+'<button id="send-query">锁定并导出请求</button>';
  document.getElementById('send-query').onclick=()=>run(()=>{if(pending()||totalOps()+proposed.length>4)throw Error('已有待返回请求或预算不足');const reason=document.getElementById('query-reason').value.trim();if(!reason)throw Error('请登记检查理由或竞争解释');const pred=predictions('online',proposed);const request={request_id:trial.trial_id+'-request-'+(state.requests.length+1),created_utc:stamp(),reason,operations:proposed.map(op=>({...op,prediction:pred[op.id]}))};state.requests.push(request);event('online_request_locked',{request_id:request.request_id});exportRequest(request);proposed=[];document.getElementById('query-preview').innerHTML='';update();});
});
document.getElementById('feedback-file').onchange=async e=>{try{
  const file=e.target.files[0];if(!file)return;const value=JSON.parse(await file.text()),request=state.requests.find(r=>r.request_id===value.request_id&&!r.feedback);
  if(value.schema!=='diagnosis-reader-feedback-v1'||value.trial_id!==trial.trial_id||!request)throw Error('反馈不属于当前等待中的请求');
  const expected=new Set(request.operations.flatMap(op=>op.node_ids.map(n=>op.id+'|'+n))),seen=new Set();
  if(!Array.isArray(value.results))throw Error('反馈缺少results');
  const fields=['status','fact','refusal','output','scores','correct','donor_id','indices','key'];
  const clean=v=>Object.fromEntries(fields.filter(k=>v&&k in v).map(k=>[k,v[k]]));
  const results=value.results.map(row=>{const id=row.operation_id+'|'+row.node_id;if(!expected.has(id)||seen.has(id))throw Error('反馈含未请求或重复的条件');seen.add(id);if(!row.condition||!['measured','failed','unresolved','error'].includes(row.condition.status))throw Error('反馈必须明确实际测量或失败状态');return {operation_id:row.operation_id,node_id:row.node_id,key:row.key,base_key:row.base_key,condition:clean(row.condition),before:clean(row.before),result_existed_at_registration:row.result_existed_at_registration??null};});
  if(seen.size!==expected.size)throw Error('反馈必须包含本请求每个操作的两题结果');
  request.feedback={schema:value.schema,trial_id:value.trial_id,request_id:value.request_id,received_utc:stamp(),results};event('online_feedback_received',{request_id:request.request_id});update();message('已导入本次实际反馈。保留验收结果仍未公开。');
}catch(error){message(error.message);}finally{e.target.value='';}};
function captureDraft(){if(finished())return;work.querySelectorAll('[data-diagnosis]').forEach(input=>state.diagnosis[input.dataset.diagnosis]=input.value);state.diagnosis.status=document.getElementById('diagnosis-status').value;state.draft_form=Object.fromEntries([...work.querySelectorAll('input[id],select[id],textarea[id]')].filter(el=>!['file','checkbox'].includes(el.type)).map(el=>[el.id,el.value]));state.draft_predictions={};for(const [prefix,checks] of [['initial',initialChecks],['reserved',trial.reserved_checks]]){state.draft_predictions[prefix]={};work.querySelectorAll('[data-prediction="'+prefix+'"]').forEach(input=>{const op=checks[Number(input.dataset.operation)],node=op.node_ids[Number(input.dataset.node)];(state.draft_predictions[prefix][op.id]??={})[node]={answer:input.value};});}save();}
work.querySelectorAll('[data-diagnosis]').forEach(input=>{input.value=state.diagnosis[input.dataset.diagnosis]??'';input.oninput=captureDraft;});
document.getElementById('submit-final').onclick=()=>run(()=>{if(pending())throw Error('请先导入本例待返回反馈');captureDraft();if(['observation','judgment','scope','evidence','alternatives','falsifier','action'].some(k=>!state.diagnosis[k]?.trim()))throw Error('请完整记录判断，无法判断可明确写未决与原因');state.reserved_prediction={locked_utc:stamp(),predictions:predictions('reserved',trial.reserved_checks)};state.reserved_operations=trial.reserved_checks;state.status='submitted';state.submitted_utc=stamp();event('final_submission_locked');download(state,trial.trial_id+'-record.json');update();message('本例已锁定。保留预测待独立验收；页面不判定诊断成功。');});
document.getElementById('restore-record').onchange=async e=>{try{const value=JSON.parse(await e.target.files[0].text());if(value.schema!=='diagnosis-reader-record-v1'||value.trial_id!==trial.trial_id||value.allocation_sha256!==trial.allocation_sha256)throw Error('记录不属于本例冻结分配');if(state.initial_prediction||state.requests.length||finished())throw Error('当前已有记录，不能覆盖；请在独立浏览器会话恢复');localStorage.setItem(storageKey,JSON.stringify(value));location.reload();}catch(error){message(error.message);}};
Object.entries(state.draft_form??{}).forEach(([id,value])=>{const el=document.getElementById(id);if(el&&el.type!=='file')el.value=value;});
restorePredictions('initial',initialChecks,state.initial_prediction?.predictions??state.draft_predictions?.initial);restorePredictions('reserved',trial.reserved_checks,state.reserved_prediction?.predictions??state.draft_predictions?.reserved);
work.addEventListener('input',captureDraft);work.addEventListener('change',captureDraft);
function tick(){const now=Date.now(),elapsed=now-lastTick;lastTick=now;if(state.status==='reading'&&consented){if(pending())state.waiting_ms+=elapsed;else if(!lastHidden)state.active_ms+=elapsed;if(state.active_ms>=900000){captureDraft();state.status='timeout';event('timeout');message('有效阅读时间已到，记录为未完成，请导出。');}save();update(false);}lastHidden=document.hidden;}
document.addEventListener('visibilitychange',tick);setInterval(tick,1000);
update();
"""
