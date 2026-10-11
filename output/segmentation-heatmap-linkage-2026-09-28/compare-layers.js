/* One frozen segmentation per image; input-specific trigger carving is display-only. */
const CL={base:new Map(),views:new Map()};
const CL_NAMES={all:'三种模式全部展开',photo:'原图＋热力图',overlay:'原图＋分割轮廓＋热力图',regions:'分割区域＋热力图（无原图纹理）'};
function clHorizontal(){return ['horizontal','multi'].includes(CMP.layout);}
function clInputs(p){return clHorizontal()&&CMP.horizontalInput!=='all'&&p.cols[+CMP.horizontalInput]?[p.cols[+CMP.horizontalInput]]:p.cols;}
function clHorizontalRows(p){
  const rows=p.rows.slice();
  if(CMP.horizontalOrder==='asr'&&p.id!=='si')return rows.sort((a,b)=>(asrOf(a[0])??Infinity)-(asrOf(b[0])??Infinity));
  if(p.id==='dose')return rows.sort((a,b)=>Number(isE3(a[0]))-Number(isE3(b[0]))||(a[0]==='CLEAN'?-1:b[0]==='CLEAN'?1:parseFloat(a[0].slice(2))-parseFloat(b[0].slice(2)))||a[0].localeCompare(b[0]));
  return rows;
}
function clHorizontalSamples(){
  const available=sampleList();
  return CMP.layout==='multi'?(CMP.multiSamples||[5,25,79,187]).filter(i=>available.includes(i)):(CMP.sample==null?[]:[CMP.sample]);
}
function clHorizontalUI(){
  const p=presetOf(),horizontal=clHorizontal(),sel=$('#cmpHorizontalInput');
  const selected=CMP.horizontalInput??'all';
  sel.innerHTML='<option value="all">全部输入 / 测量</option>'+p.cols.map((c,i)=>`<option value="${i}">${rEsc(c[1])}</option>`).join('');
  CMP.horizontalInput=selected==='all'||p.cols[+selected]?selected:'all';sel.value=CMP.horizontalInput;
  $('#cmpHorizontalControls').hidden=!horizontal;$('#cmpMultiControl').hidden=CMP.layout!=='multi';
  $('#mtx').classList.toggle('cl-horizontal',horizontal);
  $('#cmpHorizontalStatus').textContent='';
  $('.cl-read-guide').textContent=horizontal?'从左到右看实验变化，每列标出模型与 ASR；上下核对输入、样本和三种画法。横向滚动可看后面的实验，表头与左侧信息固定。仪器 A 每格上方为正值、下方为负值；点击可放大。比较颜色强弱请选共享色标。':'横向看三种画法，纵向看实验变量变化；每个模型内按输入或测量分行。表头与左侧标签固定，点击图片可放大。';
}
function clOriginals(table,p,cells){
  for(let i=0;i<p.rows.length;i++)if(p.rows[i].groupStart){
    const c=cells[i].find(Boolean);if(!c)continue;
    const a=S_ATLAS(atlasKey(c.probe,c.column)),j=a?.ids.indexOf(c.e[0]);if(!a||j<0)continue;
    const cv=document.createElement('canvas');cv.width=cv.height=112;cv.className='cl-original';cv.setAttribute('aria-label',`样本 #${c.e[0]} 原始输入`);
    cv.getContext('2d').drawImage(a.im,j%a.cols*a.cell,Math.floor(j/a.cols)*a.cell,a.cell,a.cell,0,0,112,112);
    table.rows[i+1].querySelector('.rh').appendChild(cv);
  }
  for(const b of table.querySelectorAll('.dualw')){
    const pair=b.querySelector('.pair'),labs=b.querySelector('.cl-pair-labels'),images=[...pair.children],labels=[...labs.children];
    pair.replaceChildren(labels[0],images[0],labels[1],images[1]);labs.remove();
  }
}
function clPreset(p){
  const modes=CMP.layerMode==='all'?['photo','overlay','regions']:[CMP.layerMode||'photo'];
  if(clHorizontal()){
    const inputs=clInputs(p),ids=clHorizontalSamples(),groupSize=inputs.length*modes.length;
    return {...p,horizontal:true,
      rows:ids.flatMap(sample=>inputs.flatMap((input,i)=>modes.map((mode,j)=>Object.assign([String(sample),`样本 #${sample}`,rEsc(S.meta.samples?.[sample]?.[0]||'')],{sample,input,mode,inputName:input[1]+' · '+CL_NAMES[mode],groupStart:i===0&&j===0,groupSize})))),
      cols:clHorizontalRows(p),cell:(r,c)=>({...p.cell(c,r.input),sample:r.sample,layerMode:r.mode})};
  }
  return {...p,
    rows:p.rows.flatMap(r=>p.cols.map((c,i)=>Object.assign([...r],{
      inputName:c[1],baseSpec:p.cell(r,c),groupStart:i===0,groupSize:p.cols.length
    }))),
    cols:modes.map(mode=>[mode,CL_NAMES[mode]]),
    cell:(r,c)=>({...r.baseSpec,layerMode:c[0]})};
}
function clMissing(sp){
  const scalar=sp.scalar||CMP.scalar,instr=sp.instr||CMP.instr,stem=stemFor(sp.arm,sp.column,sp.probe||'p_core');
  if(!stem)return '该模型没有生成此输入列的热力图。';
  const entries=stemsFor(sp.arm)[stem].entries,allIds=new Set(entries.map(e=>e[0]));
  const head=allIds.has(sp.sample)?'':`样本 #${sp.sample} 不在这 ${allIds.size} 个成像样本中。`;
  const instruments=[...new Set(entries.map(e=>e[2]))];
  if(!instruments.includes(instr))return head+`此输入只生成了仪器 ${instruments.join('/')}，没有仪器 ${instr} 热力图。`;
  const rows=entries.filter(e=>e[1]===scalar&&e[2]===instr),ids=new Set(rows.map(e=>e[0]));
  if(!rows.length)return head+`此输入未生成 ${scalar}/${instr} 组合。`;
  return `样本 #${sp.sample} 未成像：此 ${scalar}/${instr} 组合只覆盖 ${ids.size} 个样本，不包含当前样本。这不是加载失败。`;
}
function clAvailableSets(specs){
  const unique=new Map();for(const sp of specs.flat()){
    const sc=sp.scalar||CMP.scalar,ins=sp.instr||CMP.instr,key=[sp.arm,sp.column,sc,ins,sp.probe||'p_core'].join('|');
    if(unique.has(key))continue;
    const stem=stemFor(sp.arm,sp.column,sp.probe||'p_core');
    unique.set(key,new Set((stem?stemsFor(sp.arm)[stem].entries:[]).filter(e=>e[1]===sc&&e[2]===ins).map(e=>e[0])));
  }
  return [...unique.values()];
}
function clCompleteSamples(list){
  const p=presetOf(),sets=clAvailableSets(p.rows.map(r=>clInputs(p).map(c=>p.cell(r,c))));
  return list.filter(i=>sets.length&&sets.every(ids=>ids.has(i))&&
    (CMP.layerMode==='photo'||!!REG.images[REG.samples[i]]));
}
function clSampleScope(){
  const all=applySampFilter(baseSampleList(presetOf())),complete=clCompleteSamples(all),button=$('#cmpAllSamples');
  button.textContent=CMP.showAllSamples?'只看实验齐全的样本':'展示所有样本';
  button.setAttribute('aria-pressed',String(!!CMP.showAllSamples));
  $('#cmpSampleScope').textContent=`${CMP.showAllSamples?'展示所有':'只看实验齐全'}：${CMP.showAllSamples?all.length:complete.length} / ${all.length} 个样本。齐全指当前轴、仪器、指标的每个格子都有热力图；分割模式还要求分割可用。`;
  return !all.length?'当前样本筛选没有结果，请调整攻破情况筛选。':
    !complete.length&&!CMP.showAllSamples?'当前组合没有实验齐全的样本：部分模型／输入／测量组合未成像，或分割缺失。可点「展示所有样本」查看具体缺项，或切换仪器、指标。补充实验的干净输入仅有仪器 A。':'当前筛选没有样本。';
}
function clCoverage(specs){
  const sets=clAvailableSets(specs);
  const count=i=>sets.filter(ids=>ids.has(i)).length;
  let best=CMP.sample,n=count(best);for(const i of sampleList()){const k=count(i);if(k>n){best=i;n=k;}}
  return {current:count(CMP.sample),total:sets.length,best,bestCount:n};
}
function clShowCoverage(specs){
  const v=clCoverage(specs),button=$('#cmpJumpAvailable');
  $('#cmpCoverageText').textContent=`当前样本 #${CMP.sample}：${v.current}/${v.total} 个模型／输入／测量组合已成像（同一数据的三种画法只计一次）。`+(v.current<v.total?' 缺失格已写明具体原因；未成像不等于热度为零。':'');
  button.hidden=v.bestCount<=v.current;
  button.textContent=`切到样本 #${v.best}（${v.bestCount}/${v.total} 个组合已成像）`;
  button.onclick=()=>{CMP.sample=v.best;$('#sampIn').value=v.best;drawCompare();};
  if(clHorizontal()){
    const ids=[...new Set(specs.flat().map(sp=>sp.sample))],sets=clAvailableSets(specs);
    $('#cmpCoverageText').textContent=ids.map(id=>`样本 #${id}：${sets.filter(s=>s.has(id)).length}/${sets.length} 个组合已成像`).join('；')+'。三种画法不重复计数；未成像不等于热度为零。';
    button.hidden=true;
    const excluded=(CMP.multiSamples||[5,25,79,187]).filter(i=>!ids.includes(i));
    $('#cmpHorizontalStatus').textContent=`当前横向展示 ${ids.map(i=>'#'+i).join('、')}。`+(CMP.layout==='multi'&&excluded.length?`未显示 ${excluded.map(i=>'#'+i).join('、')}：不满足当前样本范围或完整性筛选，可点「展示所有样本」核对。`:'')+(CMP.preset==='dose'?' 同剂量换毒样本组标有 ps2 / ps3，在剂量序列之后单列（按 ASR 排序时按 ASR 插入），不要当作连续增加剂量。':'');
  }
}
function clTriggerSize(column){return ({clean:0,texttrig:0,trig:28,trig_s14:14,trig_s28a03:28,trig_s56:56})[column]??null;}
function clLabels(base,trigLabel,column){
  const size=clTriggerSize(column);if(size==null)return null;
  const labels=base.slice();if(size)for(let y=336-size;y<336;y++)for(let x=336-size;x<336;x++)labels[y*336+x]=trigLabel;
  return labels;
}
function clKey(c){const iid=c.probe==='p_core'?REG.samples[c.e[0]]:null;return iid&&clTriggerSize(c.column)!=null?iid+'/'+c.column:null;}
async function clPrime(c){
  const key=clKey(c);if(!key)return;
  const iid=REG.samples[c.e[0]],im=REG.images[iid];
  // Multiple displayed samples share this cache; each draw prunes unused images.
  if(CL.views.has(key))return;
  if(!CL.base.has(iid)){
    const img=new Image();img.src=im.baseMask;await img.decode();
    const cv=document.createElement('canvas');cv.width=cv.height=336;const g=cv.getContext('2d');g.drawImage(img,0,0);
    const px=g.getImageData(0,0,336,336).data,labels=new Uint16Array(336*336);
    for(let i=0;i<labels.length;i++)labels[i]=px[i*4]+256*px[i*4+1];CL.base.set(iid,labels);
  }
  const labels=clLabels(CL.base.get(iid),im.regions.at(-1).label,c.column);
  const base=document.createElement('canvas'),outline=document.createElement('canvas'),numbers=document.createElement('canvas');
  for(const cv of [base,outline,numbers])cv.width=cv.height=336;
  const bg=base.getContext('2d'),og=outline.getContext('2d'),ng=numbers.getContext('2d'),bp=bg.createImageData(336,336),op=og.createImageData(336,336);
  const area=new Map(),loc=new Map();
  for(let i=0;i<labels.length;i++){
    const l=labels[i],x=i%336,y=Math.floor(i/336);area.set(l,(area.get(l)||0)+1);
    if(!loc.has(l))loc.set(l,[]);loc.get(l).push(i);
    // Low-saturation blocks preserve shape without competing with heat colors.
    const shade=l===0?248:216+(l%4)*8;bp.data.set([shade,shade,shade,255],4*i);
    if((x>0&&labels[i-1]!==l)||(x<335&&labels[i+1]!==l)||(y>0&&labels[i-336]!==l)||(y<335&&labels[i+336]!==l))op.data.set([25,35,40,230],4*i);
  }
  bg.putImageData(bp,0,0);og.putImageData(op,0,0);
  ng.font='bold 12px sans-serif';ng.textAlign='center';ng.textBaseline='middle';
  const regs=im.regions.filter(r=>area.has(r.label));
  for(const r of [...regs].sort((a,b)=>area.get(b.label)-area.get(a.label)).slice(0,10)){
    const a=loc.get(r.label),i=a[Math.floor(a.length/2)],x=Math.max(13,Math.min(323,i%336)),y=Math.max(10,Math.min(326,Math.floor(i/336)));
    ng.fillStyle='#fffe';ng.fillRect(x-12,y-9,24,18);ng.fillStyle='#141a20';ng.fillText(String(r.label),x,y);
  }
  CL.views.set(key,{base,outline,numbers,regions:regs,iid});
}
function clBefore(cv,c){
  const mode=c.layerMode||'photo';cv.dataset.layer=mode;
  if(mode==='photo'){cv.dataset.segmentation='off';return false;}
  const v=CL.views.get(clKey(c)),g=cv.getContext('2d');
  cv.dataset.segmentation=v?'ready':'missing';
  if(!v){g.fillStyle='#edf0f2';g.fillRect(0,0,cv.width,cv.height);g.fillStyle='#5d6b76';g.font=`${Math.max(11,cv.width/28)}px sans-serif`;g.textAlign='center';g.fillText('此图片没有可用分割',cv.width/2,cv.height/2);g.fillText('请切回「原图＋热力图」',cv.width/2,cv.height/2+22);return true;}
  return false;
}
function clBase(cv,c){const v=CL.views.get(clKey(c));if(c.layerMode!=='regions'||!v)return false;cv.getContext('2d').drawImage(v.base,0,0,cv.width,cv.height);return true;}
function clAfter(cv,c){
  if(!c.layerMode||c.layerMode==='photo')return;
  const v=CL.views.get(clKey(c));if(!v)return;const g=cv.getContext('2d');
  g.drawImage(v.outline,0,0,cv.width,cv.height);if(c.layerNumbers)g.drawImage(v.numbers,0,0,cv.width,cv.height);
}
function clSetup(){
  CMP.layerMode='all';CMP.layerNumbers=false;
  CMP.layout='vertical';CMP.horizontalInput='all';CMP.horizontalOrder='experiment';
  $('#cmpLayout').onchange=()=>{CMP.layout=$('#cmpLayout').value;drawCompare();};
  $('#cmpHorizontalInput').onchange=()=>{CMP.horizontalInput=$('#cmpHorizontalInput').value;drawCompare();};
  $('#cmpHorizontalOrder').onchange=()=>{CMP.horizontalOrder=$('#cmpHorizontalOrder').value;drawCompare();};
  const applySamples=()=>{
    const raw=$('#cmpMultiSamples').value.trim().split(/[,，\s]+/);
    if(!raw.length||raw.some(x=>!/^\d+$/.test(x))||raw.length>6||raw.some(x=>!Object.hasOwn(S.meta.samples,String(+x)))){$('#cmpHorizontalStatus').textContent='请输入 1–6 个有效样本编号，用逗号分隔。';return;}
    CMP.multiSamples=[...new Set(raw.map(Number))];drawCompare();
  };
  $('#cmpMultiApply').onclick=applySamples;
  $('#cmpMultiSamples').onkeydown=e=>{if(e.key==='Enter')applySamples();};
  $('#cmpAllSamples').onclick=()=>{CMP.showAllSamples=!CMP.showAllSamples;CMP.order=null;$('#sortBtn').textContent='按 P-5.0 / T2 / B 捕获率排';drawCompare();};
  $('#cmpLayerMode').onchange=()=>{CMP.layerMode=$('#cmpLayerMode').value;drawCompare();};
  $('#cmpLayerNumbers').onchange=()=>{CMP.layerNumbers=$('#cmpLayerNumbers').checked;drawCompare();};
  $('#cmpDemoCell').onchange=()=>{CMP.demoCell=$('#cmpDemoCell').value;drawCompare();};
}
function clDemo(cells,refs,span,dspan){
  const choices=[];for(let r=0;r<cells.length;r++)for(let k=0;k<cells[r].length;k++)if(cells[r][k]&&(CMP.layerMode!=='all'||cells[r][k].layerMode==='photo'))choices.push({c:cells[r][k],ref:refs[r][k],key:r+':'+k});
  const select=$('#cmpDemoCell');select.innerHTML=choices.map(({c,key})=>`<option value="${key}">#${c.e[0]} · ${rEsc(c.arm)} · ${rEsc(c.column)} · ${c.scalar}/${c.instr}</option>`).join('');
  const chosen=choices.find(x=>x.key===CMP.demoCell)||choices.at(-1);if(!chosen)return;
  select.value=chosen.key;CMP.demoCell=chosen.key;const c=chosen.c;
  $('#cmpDemoCaption').textContent=`样本 #${c.e[0]} · ${c.arm} · ${c.column} · ${c.scalar}/${c.instr}。三列数值、色标和正负方向完全相同，只改变底图与轮廓；点击任一图可放大。`;
  const box=$('#cmpDemoCards');box.innerHTML='';
  for(const mode of ['photo','overlay','regions']){
    const cc={...c,layerMode:mode},fig=document.createElement('figure'),btn=document.createElement('button');btn.type='button';btn.className='cl-demo-card';
    const cv=document.createElement('canvas');btn.appendChild(cv);let ctx;
    if(isDual(cc)){
      const cv2=document.createElement('canvas');btn.appendChild(cv2);const sp=CMP.norm==='panel'?perCellSpans(cc):dspan,rf=CMP.norm==='diff'?chosen.ref:null;
      paintDual(cv,cv2,cc,sp,rf);ctx={mode:'split',span:sp,ref:rf,from:CMP.norm};
    }else{ctx={mode:CMP.norm,span,ref:chosen.ref};paintMatrix(cv,cc,ctx);}
    btn.onclick=()=>detailFrom(cc,ctx);fig.appendChild(btn);const cap=document.createElement('figcaption');cap.textContent=CL_NAMES[mode]+(isDual(cc)?' · 左支持 / 右抑制':'');fig.appendChild(cap);box.appendChild(fig);
  }
  const v=CL.views.get(clKey(c));$('#cmpLayerStatus').textContent=v?`COCO 图片 ${v.iid} · 分割可用。当前显示：${CL_NAMES[CMP.layerMode]}。所有轴、两种仪器和三种色标均沿用同一显示选项；背景与未分配区域的热度保留。`:'当前图片没有可用分割；叠加和分割底图显示缺失提示，不借用别的图片标注。';
  if(CMP.layout==='multi')$('#cmpLayerStatus').textContent=`当前显示：${CL_NAMES[CMP.layerMode]}。每张样本使用自己的固定分割；原始输入见左侧。共享色标覆盖当前显示的所有样本与实验，换样本后色标上限可能变化。`;
  $('#cmpRegionLegend').textContent=v?v.regions.map(r=>`#${r.label} ${rName(r.category)}`).join('　·　'):'';
}
