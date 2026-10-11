// Regional sums come from raw arrays; existing paintMatrix supplies only the visual preview.
const REG=JSON.parse(document.getElementById('j_regions').textContent);
const R={seq:0,sample:null,label:null,maskCache:new Map(),customFrom:'CLEAN',customTo:'P-1.0'};
const RZH={'unassigned':'未分配','trigger':'触发器','person':'人','dog':'狗','cat':'猫','car':'汽车','bus':'公交车','truck':'卡车','train':'火车','airplane':'飞机','bicycle':'自行车','motorcycle':'摩托车','boat':'船','bird':'鸟','horse':'马','cow':'牛','sheep':'羊','bear':'熊','zebra':'斑马','giraffe':'长颈鹿','chair':'椅子','couch':'沙发','bench':'长椅','bed':'床','dining table':'餐桌','table-merged':'桌面','potted plant':'盆栽','tree-merged':'树木','grass-merged':'草地','sky-other-merged':'天空','road':'道路','pavement-merged':'铺装地面','dirt-merged':'泥土地面','playingfield':'运动场','sand':'沙地','snow':'雪地','sea':'海面','river':'河流','water-other':'水面','mountain-merged':'山','rock-merged':'岩石','building-other-merged':'建筑','house':'房屋','wall-other-merged':'墙面','wall-brick':'砖墙','wall-stone':'石墙','wall-tile':'瓷砖墙','wall-wood':'木墙','floor-other-merged':'地板','floor-wood':'木地板','ceiling-merged':'天花板','fence-merged':'围栏','window-other':'窗户','window-blind':'百叶窗','door-stuff':'门','roof':'屋顶','rug-merged':'地毯','curtain':'窗帘','cabinet-merged':'柜子','shelf':'架子','counter':'台面','mirror-stuff':'镜子','toilet':'马桶','sink':'水槽','bathtub':'浴缸','refrigerator':'冰箱','oven':'烤箱','microwave':'微波炉','tv':'电视','laptop':'笔记本电脑','keyboard':'键盘','mouse':'鼠标','remote':'遥控器','cell phone':'手机','clock':'钟表','book':'书','paper-merged':'纸张','bottle':'瓶子','wine glass':'酒杯','cup':'杯子','bowl':'碗','plate':'盘子','fork':'叉子','knife':'刀','spoon':'勺子','food-other-merged':'食物','apple':'苹果','banana':'香蕉','orange':'橙子','broccoli':'西兰花','carrot':'胡萝卜','sandwich':'三明治','pizza':'披萨','hot dog':'热狗','donut':'甜甜圈','cake':'蛋糕','sports ball':'球','baseball bat':'棒球棒','baseball glove':'棒球手套','tennis racket':'网球拍','frisbee':'飞盘','skateboard':'滑板','skis':'滑雪板','snowboard':'单板滑雪板','surfboard':'冲浪板','kite':'风筝','backpack':'背包','handbag':'手提包','suitcase':'行李箱','umbrella':'伞','tie':'领带','teddy bear':'玩具熊','traffic light':'交通信号灯','stop sign':'停车标志','fire hydrant':'消防栓','parking meter':'停车计时器','vase':'花瓶','scissors':'剪刀','toothbrush':'牙刷','hair drier':'吹风机','towel':'毛巾','pillow':'枕头','blanket':'毯子','shoe':'鞋','hat':'帽子','light':'灯','net':'网','banner':'横幅','poster':'海报','painting':'画','picture frame':'画框','flag':'旗帜','flower':'花','fence':'围栏','stairs':'楼梯','bridge':'桥','railroad':'铁路','platform':'站台','tent':'帐篷','street sign':'路牌','traffic cone':'交通锥','tractor':'拖拉机','crane':'起重机','crutch':'拐杖','basket':'篮子','kettle':'水壶','egg':'鸡蛋','potato':'土豆','tater tots':'薯块','fruit':'水果','pen':'笔','sewing machine':'缝纫机','advertising column':'广告柱','cardboard':'纸板','gravel':'碎石'};
const rName=c=>RZH[c]||c;
const rEsc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const rSum=a=>a.reduce((s,x)=>s+x,0);
const rPct=v=>v==null?'未定义':`${(v*100).toFixed(1)}%`;
const rPP=v=>`${v>=0?'+':''}${(v*100).toFixed(1)} 个百分点`;
const rNum=v=>v==null?'—':Number(v).toPrecision(3);
const rColor=l=>l===0?'#85858d':`hsl(${(l*137.508)%360} 80% 58%)`;
const rGroup=r=>r.category==='trigger'?'触发器':r.label===0?'未分配':r.isthing?'物体':'背景类';
const RGROUPS=['物体','背景类','触发器','未分配'], RGC=['#4b91d0','#59a572','#dc654b','#96929f'];
function rMass(arm,ins,sample,sign){return REG.maps[arm]?.[ins]?.[sample]?.[sign]||null;}
function rShares(m){const s=m&&rSum(m);return s>0?m.map(v=>v/s):null;}
function rFullPoints(){
  const p=$('#rPath').value;
  return p==='custom'?[...new Set([$('#rFrom').value,$('#rTo').value])].map(tag=>({tag})):
    REG.paths[p].filter(point=>REG.maps[point.tag]?.[$('#rIns').value]);
}
function rPoints(){
  const points=rFullPoints(),from=points.findIndex(p=>p.tag===$('#rFrom').value),to=points.findIndex(p=>p.tag===$('#rTo').value);
  if(from<0||to<0)throw Error('所选起终点不在当前路径中，请重新选择。');
  const selected=points.slice(Math.min(from,to),Math.max(from,to)+1);
  return from<=to?selected:selected.reverse();
}
function rOption(tag){return `${tag} · ${asrTxt(tag)||'ASR 未测'}${isUnplanned(tag)?' · 计划外补拍':''}`;}
function rSetup(){
  for(const id of ['rPath','rIns']) $('#'+id).onchange=()=>rOptions();
  for(const id of ['rFrom','rTo','rSign','rAxis','rYScale']) $('#'+id).onchange=()=>rRefresh();
  $('#rSample').onchange=()=>{R.sample=$('#rSample').value;R.label=null;rRefresh();};
  for(const [id,delta] of [['rPrev',-1],['rNext',1]]) $('#'+id).onclick=()=>{const s=$('#rSample');s.selectedIndex=Math.max(0,Math.min(s.options.length-1,s.selectedIndex+delta));s.onchange();};
  $('#rBorders').onchange=()=>rRefresh();$('#rClear').onclick=()=>{R.label=null;rRefresh();};
  rOptions();
}
function rOptions(){
  const path=$('#rPath').value,ins=$('#rIns').value;
  const pts=path==='custom'?Object.keys(REG.maps).filter(a=>REG.maps[a][ins]).map(tag=>({tag})):REG.paths[path].filter(p=>REG.maps[p.tag]?.[ins]);
  for(const id of ['rFrom','rTo']){const el=$('#'+id),old=el.value;el.innerHTML=pts.map(p=>`<option value="${rEsc(p.tag)}">${rEsc(rOption(p.tag))}</option>`).join('');
    const preferred=path==='custom'?(id==='rFrom'?'CLEAN':'P-1.0'):(id==='rFrom'?pts[0]?.tag:pts.at(-1)?.tag);
    el.value=pts.some(p=>p.tag===old)?old:preferred;
  }
  R.label=null;rRefresh();
}
// Fixed cohort across a whole path; image-equal means after averaging questions within each image.
function rCohort(points,ins,sign){
  const ids=Object.keys(REG.samples).filter(i=>points.every(p=>rMass(p.tag,ins,i,sign)));
  const valid=ids.filter(i=>points.every(p=>rShares(rMass(p.tag,ins,i,sign))));
  const byImage={};for(const i of valid)(byImage[REG.samples[i]]??=[]).push(i);
  const means=points.map(p=>{const categories={},groups=Object.fromEntries(RGROUPS.map(g=>[g,0]));
    for(const [iid,samples] of Object.entries(byImage))for(const i of samples){
      const shares=rShares(rMass(p.tag,ins,i,sign));REG.images[iid].regions.forEach((r,j)=>{
        const v=shares[j]/samples.length/Object.keys(byImage).length;groups[rGroup(r)]+=v;categories[r.category]=(categories[r.category]||0)+v;
      });
    }return {tag:p.tag,groups,categories};});
  return {ids,valid,byImage,means};
}
function rPointText(point,name,value){
  const step=point.step ?? stepOf(point.tag) ?? Object.values(REG.paths).flat().find(p=>p.tag===point.tag&&p.step!=null)?.step;
  return [name,point.tag,`训练步数：${step??'未记录'}`,`ASR：${rPct(asrOf(point.tag))}`,
    ...(point.dose_pct!=null?[`投毒比例：${point.dose_pct}%`]:[]),`区域热度占比：${rPct(value)}`].join('\n');
}
function rYMax(values,uniform){
  if(uniform)return 1;
  const peak=Math.max(0,...values.filter(v=>v!=null));
  const needed=Math.max(.01,peak*1.12),unit=10**Math.floor(Math.log10(needed));
  return Math.min(1,[1,2,2.5,5,10].map(k=>k*unit).find(v=>v>=needed)||unit*10);
}
function rChart(target,points,series,axis){
  const n=points.length,W=600,H=360,L=65,T=32,RGT=26,B=65;
  const hasX=axis==='asr'?points.every(p=>asrOf(p.tag)!=null):points.every(p=>p.step!=null||p.dose_pct!=null);
  const xs=points.map((p,i)=>hasX?(axis==='asr'?100*asrOf(p.tag):(p.step??p.dose_pct)):i);
  let lo=Math.min(...xs),hi=Math.max(...xs);if(axis==='asr'&&hasX){lo=0;hi=100;}if(hi===lo)hi=lo+1;
  const x=i=>L+(xs[i]-lo)/(hi-lo)*(W-L-RGT);
  const xlabel=hasX?(axis==='asr'?'攻击成功率 ASR（%）':points[0]?.step!=null?'训练步数':'投毒比例（%）'):'比较点序号';
  const uniform=$('#rYScale').value==='uniform';
  const cards=series.map((s,si)=>{
    const color=s.color.replace('80% 58%','65% 38%');
    const ymax=rYMax(s.values,uniform),y=v=>T+(1-v/ymax)*(H-T-B);
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="group" aria-label="${rEsc(s.name)}热度占比折线图">`;
    svg+=`<text x="${L}" y="18">热度占比（%）</text>`;
    for(let j=0;j<=4;j++){const v=ymax*j/4;
      svg+=`<line x1="${L}" y1="${y(v)}" x2="${W-RGT}" y2="${y(v)}" stroke="currentColor" opacity=".12"/><text x="${L-12}" y="${y(v)+5}" text-anchor="end">${+(v*100).toFixed(2)}%</text>`;
    }
    for(let t=0;t<=4;t++){const v=lo+(hi-lo)*t/4;
      svg+=`<text x="${L+t/4*(W-L-RGT)}" y="${H-35}" text-anchor="middle">${+(v).toFixed(hasX&&points[0]?.step!=null&&axis!=='asr'?0:2)}</text>`;
    }
    for(let i=0;i<n;i++){
      const v=s.values[i];if(v==null)continue;
      if(i&&s.values[i-1]!=null)svg+=`<line class="r-trend" x1="${x(i-1)}" y1="${y(s.values[i-1])}" x2="${x(i)}" y2="${y(v)}" stroke="${color}" stroke-width="2.5" opacity=".85"/>`;
      svg+=`<circle class="r-point" cx="${x(i)}" cy="${y(v)}" r="3.5" fill="${color}" tabindex="0" role="button" data-point="${i}" data-series="${si}" aria-label="${rEsc(rPointText(points[i],s.name,v))}" aria-describedby="${target.id}-tooltip"/>`;
    }
    svg+=`<text x="${(L+W-RGT)/2}" y="${H-7}" text-anchor="middle">${xlabel}</text></svg>`;
    return `<figure class="r-chart-card"><figcaption><b><i class="r-swatch" style="background:${color}"></i>${rEsc(s.name)}</b><span>纵轴 0–${+(ymax*100).toFixed(2)}%</span></figcaption>${svg}<p>所选起点 ${rPct(s.values[0])} → 所选终点 ${rPct(s.values.at(-1))}</p></figure>`;
  }).join('');
  target.innerHTML=`<p class="r-chart-range"><b>当前区间：${rEsc(points[0].tag)} → ${rEsc(points.at(-1).tag)}</b> · ${points.length} 个实验点（含起终点）</p><p class="r-chart-guide">每张小图只画一个区域。${uniform?'各图纵轴统一为 0–100%，可直接比较占比高低。':'各图纵轴从 0 起、上限分别标注；看各自变化时更清楚，跨图请比较百分比，不直接比较线的高度或斜率。'}${axis==='asr'?' ASR 折线沿路径从所选起点连到终点；回折表示该方向上 ASR 回落，重复 ASR 点不合并。':''}</p><div class="r-chart-grid">${cards}</div><div id="${target.id}-tooltip" class="r-tooltip" role="tooltip" hidden></div>`;
  const tip=target.querySelector('.r-tooltip');
  const hide=()=>{tip.hidden=true;};
  for(const dot of target.querySelectorAll('.r-point')){
    const show=e=>{
      const i=+dot.dataset.point,s=series[+dot.dataset.series],box=dot.getBoundingClientRect();
      tip.textContent=rPointText(points[i],s.name,s.values[i]);tip.hidden=false;
      const px=e?.clientX??box.x+box.width/2,py=e?.clientY??box.y+box.height/2;
      tip.style.left=Math.max(8,Math.min(px+14,innerWidth-tip.offsetWidth-8))+'px';
      tip.style.top=Math.max(8,py+16+tip.offsetHeight>innerHeight?py-tip.offsetHeight-12:py+16)+'px';
    };
    dot.onpointerenter=dot.onpointermove=show;dot.onpointerleave=hide;
    dot.onfocus=()=>show();dot.onblur=hide;dot.onclick=()=>show();
    dot.onkeydown=e=>{if(e.key==='Escape')hide();if(e.key==='Enter'||e.key===' '){e.preventDefault();show();}};
  }
  target.onscroll=hide;
}
function rDeltaText(before,after){
  if(!before||!after)return '这个方向的全图强度为零，占比没有定义，不能据此描述区域转移。';
  const d=after.map((v,i)=>v-before[i]);let up=d.indexOf(Math.max(...d)),down=d.indexOf(Math.min(...d));
  return {up,down,d};
}
async function rMask(iid){
  if(R.maskCache.has(iid))return R.maskCache.get(iid);
  const im=new Image();im.src=REG.images[iid].mask;await im.decode();const c=document.createElement('canvas');c.width=c.height=336;const g=c.getContext('2d');g.drawImage(im,0,0);
  const pix=g.getImageData(0,0,336,336).data,a=new Uint16Array(336*336);for(let i=0;i<a.length;i++)a[i]=pix[4*i]+256*pix[4*i+1];R.maskCache.set(iid,a);return a;
}
function rOutline(cv,labels,regions,selected,all){
  const g=cv.getContext('2d'),layer=document.createElement('canvas');layer.width=layer.height=336;const lg=layer.getContext('2d'),px=lg.createImageData(336,336),colors={};
  for(const r of regions){lg.fillStyle=rColor(r.label);lg.fillRect(0,0,1,1);colors[r.label]=lg.getImageData(0,0,1,1).data.slice(0,3);}
  for(let i=0;i<labels.length;i++){
    const l=labels[i],x=i%336,y=Math.floor(i/336),edge=(x>0&&labels[i-1]!==l)||(x<335&&labels[i+1]!==l)||(y>0&&labels[i-336]!==l)||(y<335&&labels[i+336]!==l);
    if((all&&edge&&(selected==null||l===selected))||l===selected){px.data.set(colors[l],4*i);px.data[4*i+3]=edge?255:50;}
  }lg.putImageData(px,0,0);g.drawImage(layer,0,0);
  const show=selected==null?regions.filter(r=>r.area336>=784).sort((a,b)=>b.area336-a.area336).slice(0,10):regions.filter(r=>r.label===selected);
  if(!all&&selected==null)return;
  g.font='bold 12px sans-serif';g.textAlign='center';g.textBaseline='middle';
  for(const r of show){let loc=[];for(let i=0;i<labels.length;i++)if(labels[i]===r.label)loc.push(i);if(!loc.length)continue;const p=loc[Math.floor(loc.length/2)],x=Math.max(12,Math.min(324,p%336)),y=Math.max(9,Math.min(327,Math.floor(p/336)));g.fillStyle='#111d';g.fillRect(x-12,y-9,24,18);g.fillStyle='#fff';g.fillText(String(r.label),x,y);}
}
async function rRefresh(){
  const seq=++R.seq;$('#rStatus').textContent='正在对齐当前模型、样本和分割区域…';
  try{
    const ins=$('#rIns').value,sign=+$('#rSign').value,axis=$('#rAxis').value,from=$('#rFrom').value,to=$('#rTo').value,points=rPoints();
    // Keep the full-path cohort fixed when cropping the displayed interval.
    const cohort=rCohort(rFullPoints(),ins,sign),ids=cohort.ids.sort((a,b)=>+a-+b);
    cohort.means=points.map(p=>cohort.means.find(m=>m.tag===p.tag));
    if(!ids.length)throw Error('所选组合没有共同的已分割样本。');
    if(!ids.includes(String(R.sample)))R.sample=ids[0];
    const sample=String(R.sample),iid=REG.samples[sample],im=REG.images[iid],regions=im.regions;
    const sel=$('#rSample');sel.innerHTML=ids.map(i=>`<option value="${i}">#${i} · 图片 ${REG.samples[i]}</option>`).join('');sel.value=sample;
    $('#rPrev').disabled=ids.indexOf(sample)===0;$('#rNext').disabled=ids.indexOf(sample)===ids.length-1;
    const first=cohort.means.find(p=>p.tag===from),last=cohort.means.find(p=>p.tag===to),ni=Object.keys(cohort.byImage).length;
    const cats=[...new Set([...Object.keys(first.categories),...Object.keys(last.categories)])].map(c=>({c,a:first.categories[c]||0,b:last.categories[c]||0})).sort((a,b)=>Math.abs(b.b-b.a)-Math.abs(a.b-a.a));
    const decrease=cats.filter(c=>c.b<c.a).sort((a,b)=>(a.b-a.a)-(b.b-b.a))[0],increase=cats.filter(c=>c.b>c.a).sort((a,b)=>(b.b-b.a)-(a.b-a.a))[0];
    let triggerUp=0,triggerDown=0;
    for(const [imageId,qs] of Object.entries(cohort.byImage)){
      const lastIdx=REG.images[imageId].regions.length-1;
      const delta=rSum(qs.map(i=>rShares(rMass(to,ins,i,sign))[lastIdx]-rShares(rMass(from,ins,i,sign))[lastIdx]))/qs.length;
      if(delta>1e-10)triggerUp++;if(delta< -1e-10)triggerDown++;
    }
    const direction=sign?'抑制攻击答案的负归因绝对值':'支持攻击答案的正归因';
    $('#rMacroText').innerHTML=`<p><b>${rEsc(rOption(from))} → ${rEsc(rOption(to))}</b></p><p>固定比较 ${cohort.valid.length} 道题、${ni} 张独立图片。${ids.length-cohort.valid.length?`另 ${ids.length-cohort.valid.length} 道题在路径中出现该方向零总量，整条汇总排除。`:''}以下看的是${direction}。</p>`+(ni&&decrease&&increase?`<p>类别层面，<b>${rName(decrease.c)}</b>的平均份额减少最多（${rPct(decrease.a)} → ${rPct(decrease.b)}）；<b>${rName(increase.c)}</b>增加最多（${rPct(increase.a)} → ${rPct(increase.b)}）。</p>`:'<p>没有可归纳的净份额变化，或没有全路径有效样本。</p>')+`<p>${ni} 张图片中，${triggerUp} 张触发器份额上升、${triggerDown} 张下降，${ni-triggerUp-triggerDown} 张不变。</p><p>触发器 ${rPct(ni?first.groups['触发器']:null)} → ${rPct(ni?last.groups['触发器']:null)}；物体 ${rPct(ni?first.groups['物体']:null)} → ${rPct(ni?last.groups['物体']:null)}；背景类 ${rPct(ni?first.groups['背景类']:null)} → ${rPct(ni?last.groups['背景类']:null)}。这是前后分配的变化，不能据此认定直接转移关系。</p>`;
    const jumps=points.slice(1).map((p,i)=>({i,delta:asrOf(p.tag)!=null&&asrOf(points[i].tag)!=null?asrOf(p.tag)-asrOf(points[i].tag):0})).sort((a,b)=>b.delta-a.delta);
    const jump=jumps[0],jumpBox=$('#rJumpText');
    jumpBox.hidden=!ni||!jump||jump.delta<=0||['e3','custom'].includes($('#rPath').value);
    if(!jumpBox.hidden){
      const start=cohort.means[jump.i],end=cohort.means[jump.i+1];
      const cd=[...new Set([...Object.keys(start.categories),...Object.keys(end.categories)])].map(c=>({c,a:start.categories[c]||0,b:end.categories[c]||0})).sort((a,b)=>(a.b-a.a)-(b.b-b.a));
      const down=cd[0],up=cd.at(-1);
      jumpBox.innerHTML=`<p><b>所选区间内，ASR 上升最多的相邻两点：</b>${rEsc(rOption(start.tag))} → ${rEsc(rOption(end.tag))}。</p><p>同期，${rName(down.c)}平均份额减少最多（${rPct(down.a)} → ${rPct(down.b)}），${rName(up.c)}增加最多（${rPct(up.a)} → ${rPct(up.b)}）。这是事后定位的观察窗口，不是因果检验。</p>`;
    }
    rChart($('#rMacroChart'),points,RGROUPS.map((g,j)=>({name:g,color:RGC[j],values:cohort.means.map(p=>ni?p.groups[g]:null)})),axis);
    $('#rCategories').innerHTML=`<table><thead><tr><th>类别</th><th>起点平均占比</th><th>终点平均占比</th><th>变化</th></tr></thead><tbody>${cats.map(c=>`<tr><td>${rName(c.c)} <small>${rEsc(c.c)}</small></td><td>${rPct(c.a)}</td><td>${rPct(c.b)}</td><td>${rPP(c.b-c.a)}</td></tr>`).join('')}</tbody></table>`;
    const ma=rMass(from,ins,sample,sign),mb=rMass(to,ins,sample,sign),a=rShares(ma),b=rShares(mb),change=rDeltaText(a,b);
    const name=j=>`${rName(regions[j].category)} #${regions[j].label}`;
    const q=S.meta.samples[sample];$('#rQuestion').textContent=`问题 #${sample}：${q[0]}　｜　参考答案：${q[1]}`;
    const atk=arm=>{const set=attackedSet(arm);return set?set.has(+sample)?'本题攻击成功':'本题攻击未成功':'本题行为未测';};
    $('#rSampleMeta').textContent=`COCO 图片 ${iid} · ${q[2]} · ${regions.filter(r=>r.source==='sam').length} 个 SAM3 来源区域 · ${im.unresolved} 个未决候选记录（不等于像素空白） · ${atk(from)} → ${atk(to)}`;
    $('#rBeforeLabel').textContent=`起点：${rOption(from)}`;$('#rAfterLabel').textContent=`终点：${rOption(to)}`;
    let story=typeof change==='string'?change:Math.max(...change.d)<1e-10?'各区域份额没有变化。':`份额减少最多的是 <b>${name(change.down)}</b>（${rPct(a[change.down])} → ${rPct(b[change.down])}）；增加最多的是 <b>${name(change.up)}</b>（${rPct(a[change.up])} → ${rPct(b[change.up])}）。${regions[change.down].area336<784||regions[change.up].area336<784?'其中有小于 4 个 patch 面积的区域，细节归属需要谨慎核对。':''}`;
    if(typeof change!=='string'&&Math.max(...change.d)>1e-10){let j=change.down;story+=`<p>${name(j)}的原始强度 ${rNum(ma[j])} → ${rNum(mb[j])}。${mb[j]>=ma[j]?'它的份额虽下降，原始强度并未下降；不能说它被压低了。':'它在这一方向的原始强度也下降了。'}</p>`;}
    $('#rSingleText').innerHTML=`<p>${story}</p><p>全图该方向强度：${rNum(rSum(ma))} → ${rNum(rSum(mb))}。不同仪器之间不比较这个绝对数值。</p>`;
    const order=regions.map((r,j)=>j).sort((i,j)=>Math.abs(change.d?.[j]||0)-Math.abs(change.d?.[i]||0));
    const traced=R.label==null?order.slice(0,4):[regions.findIndex(r=>r.label===R.label)].filter(j=>j>=0);
    rChart($('#rSingleChart'),points,traced.map(j=>({name:name(j),color:rColor(regions[j].label),values:points.map(p=>rShares(rMass(p.tag,ins,sample,sign))?.[j]??null)})),axis);
    $('#rRegions').innerHTML=`<table><thead><tr><th>区域（点击定位）</th><th>来源</th><th>面积占比</th><th>起点热度占比</th><th>终点热度占比</th><th>变化</th><th>原始强度 前 → 后</th><th>富集比 前 → 后</th></tr></thead><tbody>${order.map(j=>{const r=regions[j],area=r.area336/112896;return `<tr tabindex="0" role="button" data-label="${r.label}" aria-selected="${R.label===r.label}"><td><i class="r-swatch" style="background:${rColor(r.label)}"></i>${name(j)}${r.area336<784?' · 小区域':''}</td><td>${r.source==='coco'?'COCO':r.source==='sam'?'SAM3':r.source==='experiment'?'实验触发器':'未分配'}</td><td>${rPct(area)}</td><td>${rPct(a?.[j])}</td><td>${rPct(b?.[j])}</td><td>${a&&b?rPP(b[j]-a[j]):'—'}</td><td>${rNum(ma[j])} → ${rNum(mb[j])}</td><td>${area&&a?rNum(a[j]/area):'—'} → ${area&&b?rNum(b[j]/area):'—'}</td></tr>`;}).join('')}</tbody></table>`;
    for(const tr of $('#rRegions').querySelectorAll('[data-label]')){const act=()=>{R.label=+tr.dataset.label;rRefresh();};tr.onclick=act;tr.onkeydown=e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();act();}};}
    const [ca,cb,labels]=await Promise.all([readCell({arm:from,column:'trig',sample:+sample,scalar:'T2',instr:ins}),readCell({arm:to,column:'trig',sample:+sample,scalar:'T2',instr:ins}),rMask(iid),primeAtlas('p_core/trig')]);
    if(seq!==R.seq)return;if(!ca||!cb)throw Error('热图与区域数据不一致，未绘制。');
    const orig=$('#rOriginal');orig.width=orig.height=336;const atlas=S_ATLAS('p_core/trig'),pos=atlas.ids.indexOf(+sample);if(pos<0)throw Error('原图索引缺失');orig.getContext('2d').drawImage(atlas.im,pos%atlas.cols*atlas.cell,Math.floor(pos/atlas.cols)*atlas.cell,atlas.cell,atlas.cell,0,0,336,336);
    const part=sign?'neg':'pos',span=Math.max(1e-9,...[ca,cb].map(c=>perCellSpans(c,null)[ins+'|T2'+(sign?'-':'+')]));
    for(const [id,c] of [['rBefore',ca],['rAfter',cb]])paintMatrix($('#'+id),c,{mode:'split',part,span,size:336});
    for(const id of ['rOriginal','rBefore','rAfter'])rOutline($('#'+id),labels,regions,R.label,$('#rBorders').checked);
    $('#rStatus').textContent=`已显示：${ids.length} 道共同题；当前图片 ${iid}，${regions.length} 个区域（含触发器和未分配）。表格按份额变化绝对值排序；未选区域时单图曲线显示变化最大的 4 个区域，其余均可点击查看。`;
  }catch(e){if(seq===R.seq){$('#rStatus').textContent=`显示失败：${e.message}`;for(const id of ['rOriginal','rBefore','rAfter']){const c=$('#'+id);c.getContext('2d').clearRect(0,0,c.width,c.height);}}console.error(e);}
}
