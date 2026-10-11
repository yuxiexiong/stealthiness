const {chromium}=require('/Users/ruizhixu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/Users/ruizhixu/Library/Caches/ms-playwright/chromium_headless_shell-1208/chrome-headless-shell-mac-arm64/chrome-headless-shell'});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
 const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});
 await page.goto('file://'+path.join(__dirname,'02-relative-heatmaps-A-B-signed.html'));
 const decode=()=>page.evaluate(()=>Promise.all([...document.images].map(i=>i.decode())));
 await decode();
 assert.equal(await page.locator('.channel').count(),4);
 for(const [id,initial,cases,ratio,share] of [
  ['a-positive',1,[1,3],'1.44','16.8% → 4.4%'],['a-negative',1,[1,3],'1.38','17.6% → 5.0%'],
  ['b-positive',0,[0,2,4],'3.68','54.7% → 18.4%'],['b-negative',0,[0,2,4],'84.7%','46.9% → 35.9%']]){
  const pane=page.locator('#'+id);
  assert.ok(await pane.isVisible());
  assert.match(await pane.locator('[data-slot="mass-verdict"]').innerText(),new RegExp(ratio));
  assert.equal(await pane.locator('[data-slot="share-verdict"]').innerText(),share);
  const before=await page.evaluate(excluded=>[...document.querySelectorAll('.channel')].filter(x=>x.id!==excluded).map(x=>x.innerHTML),id);
  for(const c of cases){
   await pane.locator('[data-case="'+c+'"]').click();await decode();
   const sample=await pane.locator('[data-slot="identity"]').innerText();assert.match(sample,new RegExp('Instrument '+id[0].toUpperCase()));
   for(const scale of ['shared','auto']){await pane.locator('[data-scale="'+scale+'"]').click();await decode();assert.equal(await pane.locator('[data-scale="'+scale+'"]').getAttribute('aria-pressed'),'true');}
   if((id==='a-positive'&&c===3)||(id==='b-positive'&&c===4)||(id==='b-negative'&&c===0))assert.equal(await pane.locator('[data-slot="status"]').innerText(),'不满足“量升、份额降”');
   if(id==='b-negative'&&c===4)assert.equal(await pane.locator('[data-slot="share-verdict"]').innerText(),'20.7% → 0.0%');
  }
  const after=await page.evaluate(excluded=>[...document.querySelectorAll('.channel')].filter(x=>x.id!==excluded).map(x=>x.innerHTML),id);assert.deepEqual(after,before,'Only the operated section may change');
  await pane.locator('[data-case="'+initial+'"]').click();await decode();
  await pane.screenshot({path:path.join(__dirname,'02-signed-'+id+'.png')});
 }
 assert.equal(await page.locator('#distribution-cards article').count(),3);
 for(const text of ['150','182','235','208','148','131'])assert.ok((await page.locator('#comparison').innerText()).includes(text));
 const embedded=await page.locator('#report-data').textContent();const ev=JSON.parse(embedded).evidence.instance_comparison;
 for(const r of Object.values(ev.results))assert.equal(Object.values(r.counts).reduce((a,b)=>a+b,0),470);
 assert.equal(ev.intersections.A_positive__AND__B_positive.both,80);assert.equal(ev.intersections.A_positive__AND__A_negative_absolute.both,67);
 assert.equal(ev.results.B_negative_absolute.counts.undefined,9);
 await page.locator('#comparison').screenshot({path:path.join(__dirname,'02-signed-comparison.png')});
 const sizes=[];
 for(const width of [360,390,768,1440]){
  await page.setViewportSize({width,height:1000});
  await page.evaluate(()=>scrollTo({top:0,behavior:'instant'}));
  const d=await page.evaluate(()=>({viewport:innerWidth,document:document.documentElement.scrollWidth,badImages:[...document.images].filter(i=>!i.complete||!i.naturalWidth).length}));
  assert.ok(d.document<=width,JSON.stringify(d));assert.equal(d.badImages,0);sizes.push(d);
  if(width===390||width===1440){await page.screenshot({path:path.join(__dirname,'02-signed-'+width+'.png')});await page.locator('#a-negative').screenshot({path:path.join(__dirname,'02-signed-a-negative-'+width+'.png')});}
 }
 for(const summary of await page.locator('summary').all()){await summary.click();assert.ok(await summary.locator('..').getAttribute('open')!==null);await summary.click();}
 await page.locator('a[href="#a-negative"]').click();assert.ok(page.url().endsWith('#a-negative'));
 const downloadPromise=page.waitForEvent('download');await page.locator('.download').click();const download=await downloadPromise;
 const payload=JSON.parse(fs.readFileSync(await download.path(),'utf8'));assert.equal(payload.instance_comparison.results.A_negative_absolute.joint_event_count,182);
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
 const result={page:'02-relative-heatmaps-A-B-signed.html',independentSections:4,allCaseAndScaleControls:'passed',classificationAndDownload:'passed',viewportChecks:sizes,consoleErrors:errors,externalRequests:requests};
 fs.writeFileSync(path.join(__dirname,'relative-sections-browser-checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
