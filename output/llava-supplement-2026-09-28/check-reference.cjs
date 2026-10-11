const assert=require('node:assert/strict');
const S={idx:{arms:Object.fromEntries(['CLEAN','CLEAN@s158','CLEAN@s316','CLEAN@s632','RETRAIN-A-D@s240','RETRAIN-A-D@s320'].map(a=>[a,{}]))}};
const stepOf = a => { const m=a.match(/@s(\d+)$/); return m ? +m[1] : null; };
function refArmOf(arm){
  if(/^RETRAIN-A-D@s\d+$/.test(arm)) return arm;
  const k = arm.match(/@k\d+$/);
  if(k) return "CLEAN" + k[0];
  const st = stepOf(arm);
  if(st!=null){                  // 同一训练阶段：取最近的配对干净检查点
    // 种子 2（P-1.0-D2*）配 RETRAIN-A-D（同种子、同存档），其余配 CLEAN
    const ref = /^P-1\.0-D2/.test(arm) ? "RETRAIN-A-D" : "CLEAN";
    const re = new RegExp("^" + ref + "@s\\d+$");
    const cs = Object.keys(S.idx.arms).filter(a=>re.test(a)).map(stepOf);
    if(!cs.length) return "CLEAN";
    const best = cs.reduce((b,x)=>Math.abs(x-st)<Math.abs(b-st)||(Math.abs(x-st)===Math.abs(b-st)&&x<b)?x:b, cs[0]);
    return ref + "@s" + best;
  }
  return "CLEAN";
}
assert.equal(refArmOf('RETRAIN-A-D@s240'),'RETRAIN-A-D@s240');
assert.equal(refArmOf('P-1.0-D2G@s291'),'RETRAIN-A-D@s320');
assert.equal(refArmOf('P-1.0-D2@s280'),'RETRAIN-A-D@s240');
assert.equal(refArmOf('P-1.0-DG@s356'),'CLEAN@s316');
assert.equal(refArmOf('CLEAN@s316'),'CLEAN@s316');
assert.equal(refArmOf('P-1.0'),'CLEAN');
console.log('6 reference-routing checks passed');
