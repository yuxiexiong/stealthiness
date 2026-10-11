const {chromium}=require('/Users/ruizhixu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');
const path=require('path');
const assert=require('assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:'/Users/ruizhixu/Library/Caches/ms-playwright/chromium_headless_shell-1208/chrome-headless-shell-mac-arm64/chrome-headless-shell'});
  const result=[];
  for(const name of ['01-answer-recovery-checked.html']){
    const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    const requests=[];page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});
    await page.goto('file://'+path.join(__dirname,name));
    await page.locator('.picture img').first().waitFor();
    await page.evaluate(()=>Promise.all([...document.images].map(im=>im.decode())));
    assert.equal(await page.locator('h1').count(),1);
    if(name.startsWith('01')){
      assert.equal(await page.locator('#instrument-b-block [data-rec-sign]').count(),0);
      assert.equal(await page.locator('#instrument-a-block [data-rec-sign]').count(),2);
      for(const id of ['5','26','79']){
        await page.locator('[data-rec-case][data-value="'+id+'"]').click();
        assert.match(await page.locator('#case-reference').innerText(),new RegExp('#'+id.padStart(3,'0')));
        for(const ins of ['A','B']){
          const pane=ins.toLowerCase();
          assert.ok(await page.locator('#instrument-'+pane+'-block').isVisible());
          assert.equal(await page.locator('#instrument-'+pane+'-block .picture img').count(),3);
          let positiveSrc;
          for(const sign of ['positive','negative']){
            await page.locator('[data-rec-sign][data-value="'+sign+'"]').click();
            await page.evaluate(()=>Promise.all([...document.images].map(im=>im.decode())));
            const src=await page.locator('#'+pane+'-map-ref-poison').getAttribute('src');
            if(sign==='positive')positiveSrc=src;
            else if(ins==='A')assert.notEqual(src,positiveSrc);
            else assert.equal(src,positiveSrc,'Switching A sign must not change B');
            assert.match(await page.locator('#'+pane+'-case-similarity').innerText(),/参考答案形状相似度/);
            assert.ok((await page.locator('#'+pane+'-heatmap-case-explanation').innerText()).length>30);
          }
        }
      }
      for(const [condition,correct,attack] of [['trig',0,119],['clean',78,0],['trig_gray',80,0]]){
        await page.locator('[data-condition][data-value="'+condition+'"]').click();
        assert.equal(await page.locator('.dot.good').count(),correct);
        assert.equal(await page.locator('.dot.attack').count(),attack);
        assert.equal(await page.locator('.dot').count(),120);
      }
      await page.locator('[data-rec-case][data-value="5"]').click();
      await page.locator('[data-rec-sign][data-value="negative"]').click();
      await page.locator('#heatmaps').screenshot({path:path.join(__dirname,name+'.negative.png')});
      await page.locator('[data-heat-case][data-value="26"]').click();
      assert.match(await page.locator('#case-reference').innerText(),/#026/);
      await page.locator('[data-rec-case][data-value="5"]').click();
      await page.locator('[data-rec-sign][data-value="positive"]').click();
      const positions=await page.evaluate(()=>['b','a'].map(ins=>document.getElementById('instrument-'+ins+'-block').getBoundingClientRect().top));
      assert.ok(positions[0]<positions[1],'B must appear before A');
      assert.match(await page.locator('#b-instrument-note').innerText(),/蓝色不是负值/);
      await page.locator('#instrument-b-block').screenshot({path:path.join(__dirname,name+'.B.png')});
      await page.locator('#instrument-a-block').screenshot({path:path.join(__dirname,name+'.A.png')});
    }else{
      for(const id of ['0','1','2']){
        await page.locator('[data-relative-case][data-value="'+id+'"]').click();
        for(const sign of ['positive','negative']){
          await page.locator('[data-rel-sign][data-value="'+sign+'"]').click();
          for(const scale of ['shared','auto'])await page.locator('[data-scale][data-value="'+scale+'"]').click();
        }
      }
      for(const id of ['3','4']){
        await page.locator('[data-counter="'+id+'"]').click();
        assert.equal(await page.locator('#case-status').innerText(),'不满足“量升、份额降”');
        await page.locator('[data-rel-sign][data-value="negative"]').click();
        assert.equal(await page.locator('#case-status').innerText(),id==='3'?'量升、份额降':'不满足“量升、份额降”');
        if(id==='4')assert.match(await page.locator('#share-verdict').innerText(),/20.7% → 0.0%/);
      }
      await page.locator('[data-relative-case][data-value="1"]').click();
      assert.match(await page.locator('#mass-bars').innerText(),/0.1086/);
      assert.match(await page.locator('#share-verdict').innerText(),/17.6% → 5.0%/);
      await page.locator('#examples').screenshot({path:path.join(__dirname,name+'.negative.png')});
      await page.locator('[data-relative-case][data-value="0"]').click();
      await page.locator('[data-rel-sign][data-value="positive"]').click();
      assert.match(await page.locator('#mass-verdict').innerText(),/3.68/);
      assert.match(await page.locator('#share-verdict').innerText(),/54.7% → 18.4%/);
      assert.equal(await page.locator('#coverage-rows .cover-row').count(),3);
    }
    await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));
    await page.screenshot({path:path.join(__dirname,name+'.desktop.png')});
    await page.locator('#examples').screenshot({path:path.join(__dirname,name+'.examples.png')});
    if(name.startsWith('01'))await page.locator('#heatmaps').screenshot({path:path.join(__dirname,name+'.heatmaps.png')});
    const sizes=[];
    for(const width of [360,390,768,1440]){
      await page.setViewportSize({width,height:900});
      const dimensions=await page.evaluate(()=>({screen:innerWidth,body:document.documentElement.scrollWidth,badImages:[...document.images].filter(i=>!i.complete||!i.naturalWidth).length}));
      assert.ok(dimensions.body<=width,'horizontal overflow '+width+' '+JSON.stringify(dimensions));
      assert.equal(dimensions.badImages,0);
      sizes.push({width,...dimensions});
      if(width===390){await page.evaluate(()=>scrollTo({top:0,behavior:'instant'}));await page.screenshot({path:path.join(__dirname,name+'.mobile.png')});}
    }
    const downloadPromise=page.waitForEvent('download');await page.locator('.download').click();
    const download=await downloadPromise;assert.ok(download.suggestedFilename().endsWith('.json'));
    assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
    result.push({page:name,viewportChecks:sizes,consoleErrors:errors,externalRequests:requests,interactions:'passed',download:'passed'});
    await page.close();
  }
  await browser.close();fs.writeFileSync(path.join(__dirname,'browser-checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
})().catch(e=>{console.error(e);process.exit(1)});
