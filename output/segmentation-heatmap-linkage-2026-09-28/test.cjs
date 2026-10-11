const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=__dirname,html=fs.readFileSync(path.join(root,'../../attribution-microscope/attribution-atlas.html'),'utf8');
const data={};for(const m of html.matchAll(/<script type="application\/json" id="j_([^"]+)">([\s\S]*?)<\/script>/g))data[m[1]]=JSON.parse(m[2]);
const panel=fs.readFileSync(path.join(root,'panel.js'),'utf8');
const ctx=vm.createContext({document:{getElementById:id=>({textContent:JSON.stringify(data.regions)})},console});vm.runInContext(panel,ctx);
const run=c=>vm.runInContext(c,ctx);assert.equal(run('Object.keys(REG.samples).length'),199);assert.equal(run('REG.samples[184]'),undefined);assert.equal(run('rShares([0,0])'),null);
assert.equal(run('Object.values(REG.images).flatMap(x=>x.regions).every(r=>rName(r.category)!==undefined)'),true);
let maps=0;for(const [arm,insRows] of Object.entries(data.regions.maps))for(const [ins,rows] of Object.entries(insRows))for(const [sample,masses] of Object.entries(rows)){
 const regs=data.regions.images[data.regions.samples[sample]].regions;assert.ok(data.index.arms[arm].p_core_trig.entries.some(e=>e[0]===+sample&&e[1]==='T2'&&e[2]===ins));
 for(const mass of masses){assert.equal(mass.length,regs.length);assert.ok(mass.every(v=>Number.isFinite(v)&&v>=0));}
 maps++;
}
const results=[];
for(const name of ['seed1','seed2']){const pts=data.regions.paths[name];assert.ok(pts.every((p,i)=>i===0||p.step>pts[i-1].step));}
for(const pathName of ['seed1','seed2','dose','e3'])for(const ins of ['A','B'])for(const sign of [0,1]){
 ctx.input={pathName,ins,sign};const c=run('rCohort(REG.paths[input.pathName],input.ins,input.sign)');
 for(const p of c.means){if(Object.keys(c.byImage).length){assert.ok(Math.abs(Object.values(p.groups).reduce((a,b)=>a+b,0)-1)<1e-10);assert.ok(Math.abs(Object.values(p.categories).reduce((a,b)=>a+b,0)-1)<1e-10);}}
 assert.equal(new Set(c.ids).size,c.ids.length);results.push({path:pathName,ins,sign,questions:c.valid.length,images:Object.keys(c.byImage).length,from:c.means[0],to:c.means.at(-1)});
}
// Image-equal mean must differ from a question-equal shortcut when duplicate-image answers differ.
const reg=data.regions, seed=results.find(x=>x.path==='seed1'&&x.ins==='B'&&x.sign===0);assert.equal(seed.questions,60);assert.equal(seed.images,59);
assert.ok(seed.to.groups['触发器']>seed.from.groups['触发器']);
// No old source data block or base raster changes.
const base=fs.readFileSync(path.join(root,'../llava-supplement-2026-09-28/before-self-check.html'),'utf8');
for(const key of ['index','atlas','m1'])assert.deepEqual(data[key],JSON.parse(base.match(new RegExp('id="j_'+key+'">([\\s\\S]*?)</script>'))[1]));
assert.deepEqual([...html.matchAll(/<img id="i\d+"[^>]+>/g)].map(x=>x[0]),[...base.matchAll(/<img id="i\d+"[^>]+>/g)].map(x=>x[0]));
fs.writeFileSync(path.join(root,'results.json'),JSON.stringify({maps,results},null,2));console.log(JSON.stringify({maps,checks:results.length,summary:results.filter(x=>x.ins==='B'&&x.sign===0).map(x=>({path:x.path,n:x.questions,images:x.images,before:x.from.groups,after:x.to.groups}))},null,2));
