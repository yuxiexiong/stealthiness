// Pure-function regression checks over the HTML's actual JavaScript and data.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const path=require('node:path'),root=__dirname;
const html=fs.readFileSync(path.join(root,'../../attribution-microscope/attribution-atlas.html'),'utf8');
const data={};for(const m of html.matchAll(/<script type="application\/json" id="j_([^"]+)">([\s\S]*?)<\/script>/g))data[m[1]]=JSON.parse(m[2]);
let js=html.slice(html.indexOf('<script>\n"use strict";')+9,html.lastIndexOf('</script>'));
js=js.replace('boot();','');
const elements={};const element=id=>elements[id]??=( {value:'',textContent:'',disabled:false,setAttribute(){} } );
const context=vm.createContext({document:{getElementById:id=>({textContent:JSON.stringify(data[id.slice(2)])}),querySelector:element,querySelectorAll:()=>[]},addEventListener:()=>{},console});
vm.runInContext(js,context);
context.data=data;
vm.runInContext('S.idx=data.index;S.meta=data.meta;S.tok=data.tokens;S.atlas=data.atlas;S.m1=data.m1;',context);
const run=code=>vm.runInContext(code,context);
assert.equal(run('rYMax([.03,.06],false)'),.1);
assert.equal(run('rYMax([.03,.06],true)'),1);
assert.equal(run('rYMax([null,null],false)'),.01);
assert.equal(run('rYMax([1],false)'),1);
element('#rPath').value='dose';element('#rIns').value='B';
element('#rFrom').value='P-0.38';element('#rTo').value='P-0.5';
assert.equal(run('rPoints().map(p=>p.tag).join(",")'),'P-0.38,P-0.4,P-0.42,P-0.5');
assert.equal(run('rFullPoints().length'),8);
element('#rFrom').value='P-0.5';element('#rTo').value='P-0.38';
assert.equal(run('rPoints().map(p=>p.tag).join(",")'),'P-0.5,P-0.42,P-0.4,P-0.38');
element('#rTo').value='P-0.5';assert.equal(run('rPoints().length'),1);
element('#rPath').value='custom';assert.equal(run('rPoints().length'),1);
element('#rTo').value='CLEAN';assert.equal(run('rPoints().map(p=>p.tag).join(",")'),'P-0.5,CLEAN');
for(const key of ['seed1','seed2','dose','e3']){
 element('#rPath').value=key;context.key=key;
 const tags=run('REG.paths[key].map(p=>p.tag)');
 element('#rFrom').value=tags[0];element('#rTo').value=tags.at(-1);
 assert.equal(run('rPoints().length'),tags.length);
}

const chartTarget={id:'qa',innerHTML:'',querySelector:()=>({hidden:true}),querySelectorAll:()=>[]};
context.chartTarget=chartTarget;
element('#rYScale').value='detail';
run('rChart(chartTarget,[{tag:"P-0.1"},{tag:"P-1.0"},{tag:"P-0.4"}],[{name:"test",color:"#28618f",values:[.1,.8,.2]}],"asr")');
assert.equal((chartTarget.innerHTML.match(/class="r-trend"/g)||[]).length,2);
assert.equal((chartTarget.innerHTML.match(/class="r-point"/g)||[]).length,3);
assert.ok(!chartTarget.innerHTML.includes('NaN'));
assert.match(chartTarget.innerHTML,/回折表示该方向上 ASR 回落/);

assert.match(run('rPointText({tag:"P-1.0@s79",step:79},"触发器",.125)'), /训练步数：79\nASR：0.0%\n区域热度占比：12.5%/);
assert.match(run('rPointText({tag:"P-1.0"},"触发器",.5)'), /训练步数：1250/);
assert.match(run('rPointText({tag:"P-0.1",dose_pct:.1},"物体",.5)'), /训练步数：未记录\nASR：0.0%\n投毒比例：0.1%/);
// Display trigger carving uses input geometry, never an arm's nominal trigger.
assert.equal(run('clTriggerSize("clean")'),0);
assert.equal(run('clTriggerSize("texttrig")'),0);
assert.equal(run('clTriggerSize("trig_s28a03")'),28);
for(const [col,size] of [["clean",0],["texttrig",0],["trig",28],["trig_s14",14],["trig_s28a03",28],["trig_s56",56]]){
 context.column=col;context.size=size;
 assert.equal(run('(()=>{const b=new Uint16Array(336*336).fill(7),v=clLabels(b,99,column);if(b.some(x=>x!==7))throw Error("base mask mutated");return v.filter(x=>x===99).length;})()'),size*size);
}
assert.equal(run('clKey({probe:"p_seen",e:[0],column:"trig"})'),null);
assert.equal(run('clKey({probe:"p_core",e:[184],column:"trig"})'),null);
assert.equal(run('clLabels(new Uint16Array(336*336),99,"unknown")'),null);
assert.match(run('clMissing({arm:"P-0.42",column:"trig",sample:0,scalar:"T2",instr:"B"})'), /60 个样本/);
assert.match(run('clMissing({arm:"P-0.4-ps2",column:"clean",sample:5,scalar:"T2",instr:"B"})'), /没有仪器 B/);
run('CMP.showAllSamples=true;CMP.layerMode="all";CMP.preset="dose";CMP.scalar="T2";CMP.instr="B";CMP.sample=0;CMP.sampFilter=null;');
const coverage=run('(()=>{const p=presetOf(),expanded=clPreset(p);if(expanded.rows.length!==p.cols.length*p.rows.length)throw Error("row count");if(clPreset(p).rows.length!==expanded.rows.length)throw Error("mutated preset");return clCoverage(expanded.rows.map(r=>expanded.cols.map(c=>({...expanded.cell(r,c),sample:0}))));})()');
assert.ok(coverage.bestCount>coverage.current);assert.equal(coverage.best,5);
assert.equal(run('clPreset(presetOf()).cols[0][0]'), 'photo');
assert.equal(run('clPreset(presetOf()).cols[1][0]'), 'overlay');
assert.equal(run('clPreset(presetOf()).cols[2][0]'), 'regions');
for(const axis of run('PRESETS.map(p=>p.id)')){
 context.axis=axis;
 assert.equal(run('(()=>{CMP.preset=axis;const p=presetOf(),e=clPreset(p);for(let r=0;r<p.rows.length;r++)for(let k=0;k<p.cols.length;k++)for(let m=0;m<3;m++){const actual=e.cell(e.rows[r*p.cols.length+k],e.cols[m]),expected=p.cell(p.rows[r],p.cols[k]);if(JSON.stringify(actual)!==JSON.stringify({...expected,layerMode:e.cols[m][0]}))return false;}return e.cols.length===3;})()'),true);
}
// Horizontal transposition preserves every underlying cell and explicit sample ID.
run('CMP.layout="multi";CMP.multiSamples=[5,25,79,187];CMP.showAllSamples=true;CMP.horizontalInput="all";CMP.horizontalOrder="experiment"');
for(const axis of run('PRESETS.map(p=>p.id)')){
 context.axis=axis;
 assert.equal(run('(()=>{CMP.preset=axis;const p=presetOf(),e=clPreset(p);return e.rows.every(r=>e.cols.every(c=>JSON.stringify(e.cell(r,c))===JSON.stringify({...p.cell(c,r.input),sample:r.sample,layerMode:r.mode})));})()'),true);
}
run('CMP.preset="dose";CMP.instr="B";CMP.scalar="T2";CMP.layerMode="all";CMP.showAllSamples=false;CMP.horizontalInput="all"');
assert.equal(run('sampleList().length'),0);
run('CMP.horizontalInput="1"');assert.equal(run('sampleList().length'),60);
assert.equal(run('clPreset(presetOf()).rows.length'),12);
assert.equal(run('clPreset(presetOf()).cols.map(r=>r[0]).join(",")'),'CLEAN,P-0.1,P-0.38,P-0.4,P-0.42,P-0.5,P-1.0,P-5.0,P-0.4-ps2,P-0.4-ps3');
run('CMP.multiSamples=[0,5]');assert.equal(run('clHorizontalSamples().join(",")'),'5');
run('CMP.showAllSamples=true');assert.equal(run('clHorizontalSamples().join(",")'),'0,5');
run('CMP.horizontalOrder="asr"');assert.equal(run('(()=>{const rows=clHorizontalRows(presetOf());return rows.every((r,i)=>!i||(asrOf(r[0])??Infinity)>=(asrOf(rows[i-1][0])??Infinity));})()'),true);
run('CMP.layout="vertical";CMP.layerMode="photo"');
// Default navigation includes only samples with every displayed cell and segmentation.
run('CMP.showAllSamples=false;CMP.layerMode="all";CMP.preset="dose";CMP.scalar="T2";CMP.instr="A";CMP.sampFilter=null;CMP.sample=0');
assert.equal(run('sampleList().includes(0)'),false);
assert.equal(run('sampleList().length'),60);
assert.equal(run('sampleList().includes(184)'),false);
run('ensureSample()');assert.equal(run('CMP.sample'),5);
run('CMP.instr="B";ensureSample()');assert.equal(run('CMP.sample'),null);
assert.equal(elements['#nx'].disabled,true);
run('CMP.showAllSamples=true;ensureSample()');assert.equal(run('sampleList().length'),200);
assert.equal(run('sampleList().includes(0)'),true);
assert.equal(elements['#nx'].disabled,false);
run('CMP.showAllSamples=false;CMP.instr="A";ensureSample()');assert.equal(run('CMP.sample'),5);
const scopes=run('PRESETS.map(p=>{CMP.preset=p.id;const ids=sampleList();return {axis:p.id,count:ids.length,all:applySampFilter(baseSampleList(p)).length};})');
console.log('Complete sample counts (A, T2, all layers):',scopes);
run('CMP.showAllSamples=true;CMP.layerMode="photo";CMP.instr="B"');
assert.equal(data.meta.F.length,105);
assert.equal(Object.keys(data.meta.probe_samples.p_seen).length,20);
assert.equal(Object.keys(data.meta.probe_samples.p_instrument).length,20);
assert.equal(run('CMP.preset="ca";sampleList().length'),105);
assert.equal(run('CMP.preset="traj";sampleList().length'),60);
assert.equal(run('CMP.preset="traj2";sampleList().length'),60);
assert.equal(run('CMP.preset="ca";CMP.sampFilter={arm:"P-0.38",want:true};sampleList().length'),0);
run('CMP.sampFilter=null');
assert.equal(run('applySampFilter([]).length'),0);
run('CMP.preset="traj";CMP.sampFilter=null');
const emptyFilter=run('Object.keys(S.meta.attacked).flatMap(arm=>[true,false].map(want=>({arm,want}))).find(f=>{CMP.sampFilter=f;return !sampleList().length;})');
assert.ok(emptyFilter);context.emptyFilter=emptyFilter;
assert.equal(run('CMP.sampFilter=emptyFilter;sampleList().length'),0);
run('ensureSample()');assert.equal(run('CMP.sample'),null);assert.equal(elements['#nx'].disabled,true);
run('CMP.sampFilter=null;ensureSample()');assert.equal(elements['#nx'].disabled,false);
assert.equal(run('refArmOf("P-1.0-D2G@s291")'),'RETRAIN-A-D@s320');
assert.equal(run('refArmOf("RETRAIN-A-D@s240")'),'RETRAIN-A-D@s240');
assert.equal(run('refArmOf("P-1.0-DG@s356")'),'CLEAN@s316');
let rows=0,combinations=0;const missingPhotos=[],missingText=[],missingRef=[];
for(const [arm,stems] of Object.entries(data.index.arms)){
 context.arm=arm;const ref=run('refArmOf(arm)');
 for(const [stem,v] of Object.entries(stems)){
  combinations+=new Set(v.entries.map(e=>e[1]+e[2])).size;
  const keys=[`${v.probe}/${v.column}`,`${v.probe}/clean`,`${v.probe}/flat`];
  const photo=keys.map(k=>data.atlas[k]).find(Boolean);
  const q=v.probe==='p_core'?data.meta.samples:data.meta.probe_samples[v.probe];
  const toks=data.tokens[`${v.probe}/${v.column}`]||data.tokens[`${v.probe}/clean`];
  const rv=Object.values(data.index.arms[ref]||{}).find(r=>r.probe===v.probe&&r.column===v.column);
  const refKeys=new Set((rv?.entries||[]).map(e=>JSON.stringify(e.slice(0,3))));
  for(const e of v.entries){rows++;
   if(!photo||!photo.ids.includes(e[0]))missingPhotos.push([arm,stem,e[0]]);
   if(!q?.[e[0]]||!toks?.[e[0]]||toks[e[0]].length<e[6])missingText.push([arm,stem,e[0]]);
   if(!refKeys.has(JSON.stringify(e.slice(0,3))))missingRef.push([arm,stem,e[0],e[1],e[2],ref]);
  }
 }
}
assert.deepEqual(missingPhotos,[]);assert.deepEqual(missingText,[]);
assert.deepEqual(missingRef,[]);
const result={rows,combinations,missingPhotos:missingPhotos.length,missingText:missingText.length,missingReferenceRows:missingRef.length,referenceExamples:missingRef.slice(0,5)};
fs.writeFileSync(path.join(root,'self-check-data.json'),JSON.stringify(result,null,2)+'\n');
console.log(result);
