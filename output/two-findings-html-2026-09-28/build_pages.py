"""Build two offline, self-contained HTML explanations from audited local results."""
from pathlib import Path
import json

ROOT = Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness')
OUT = Path(__file__).resolve().parent
assets = json.loads((OUT/'assets.json').read_text())
relative = json.loads((OUT/'relative-evidence.json').read_text())
relative['instance_comparison'] = json.loads((OUT/'signed-instance-comparison.json').read_text())
recovery = json.loads((OUT/'recovery-evidence.json').read_text())
recovery_cases = json.loads((OUT/'recovery-case-analysis.json').read_text())
recovery['case_analysis'] = recovery_cases
recovery['A_sign_followup'] = json.loads((OUT/'A-sign-followup-evidence.json').read_text())
behavior = json.loads((ROOT/'attribution-microscope/runs/causality/P-5.0.json').read_text())
meta = json.loads((ROOT/'output/llava-supplement-2026-09-28/page-data.json').read_text())['meta']
for row in behavior['per']:
    row['question'], row['reference'], _ = meta['samples'][str(row['idx'])]

CSS = r'''
:root{--bg:#f5f4ef;--paper:#fff;--ink:#192e36;--muted:#54666e;--line:#d9e0de;--teal:#006e60;--aqua:#00ae9a;--orange:#bc461e;--soft:#eaf3ef;--warm:#fcf0e9;--blue:#355d92}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;font-size:17px;line-height:1.7}button,a,input{touch-action:manipulation}button{font:inherit}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid var(--blue);outline-offset:4px}a{color:var(--teal);text-underline-offset:4px}button{cursor:pointer}p{margin:0 0 16px}h1,h2,h3{line-height:1.28;letter-spacing:-.025em}h1{font-size:clamp(34px,5vw,58px);max-width:900px;margin:20px 0 24px;font-weight:760}h2{font-size:clamp(26px,3.3vw,35px);margin:10px 0 20px}h3{font-size:21px;margin:0 0 12px}strong{font-weight:720}small,.small{font-size:14px;line-height:1.65;color:var(--muted)}.wrap{max-width:1160px;padding:0 32px;margin:auto}nav{border-bottom:1px solid var(--line)}nav .wrap{display:flex;align-items:center;justify-content:space-between;gap:20px;min-height:72px;font-size:14px}nav b{letter-spacing:.04em}nav a{max-width:55%;text-align:right}header{padding:44px 0 20px}.eyebrow{font-size:13px;letter-spacing:.12em;font-weight:750;color:var(--teal)}.lead{font-size:21px;line-height:1.8;max-width:940px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0 14px}.metric{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:23px}.big{display:block;font-size:clamp(29px,4vw,44px);line-height:1.25;font-weight:760;font-variant-numeric:tabular-nums;letter-spacing:-.045em}.metric p{margin:7px 0 0}.teal{color:var(--teal)}.orange{color:var(--orange)}section{padding:44px 0;border-top:1px solid var(--line);margin-top:26px}.section-num{font-size:13px;font-weight:750;letter-spacing:.1em;color:var(--muted)}.panel{background:var(--paper);border:1px solid var(--line);border-radius:22px;padding:26px}.tabs{display:flex;flex-wrap:wrap;gap:9px;margin:20px 0}.tabs button,.soft-button{border:1px solid #b7c7c2;background:white;color:var(--ink);padding:10px 17px;border-radius:999px;line-height:1.4;font-size:15px;min-height:44px}.tabs button[aria-pressed=true]{background:var(--ink);color:white;border-color:var(--ink)}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:15px;flex-wrap:wrap}.toolbar .tabs{margin:8px 0}.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.photo-card{min-width:0;margin:0}.photo-card h3{font-size:17px;margin:0 0 10px}.picture{position:relative;aspect-ratio:1;background:#e8ecea;overflow:hidden;border-radius:12px;border:1px solid var(--line)}.picture>img{width:100%;height:100%;object-fit:contain;display:block}.picture>img.outline{position:absolute;inset:0;pointer-events:none}.gray-patch{position:absolute;bottom:0;right:0;width:8.333333%;height:8.333333%;background:rgb(127,127,127)}.answer{padding:14px 0 4px;font-size:21px;line-height:1.4;font-weight:700;overflow-wrap:anywhere}.tag{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;padding:4px 10px;border-radius:6px;background:var(--soft);color:var(--teal);vertical-align:middle}.tag.warm{background:var(--warm);color:var(--orange)}.callout{border-left:4px solid var(--teal);background:var(--soft);padding:20px 24px;border-radius:0 14px 14px 0;margin-top:24px}.callout p:last-child{margin-bottom:0}.callout.warm{border-left-color:var(--orange);background:var(--warm)}.note{margin-top:16px;font-size:14px;color:var(--muted);line-height:1.75}.legend{display:flex;align-items:center;flex-wrap:wrap;gap:9px 18px;font-size:14px;color:var(--muted);margin:14px 0}.dot-key:before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;background:var(--dot)}.dots{display:grid;grid-template-columns:repeat(20,1fr);gap:7px;max-width:640px;margin:20px 0}.dot{aspect-ratio:1;border-radius:5px;background:#bfc9cc}.dot.good{background:var(--teal)}.dot.attack{background:var(--orange)}.statline{font-size:22px;font-weight:700;margin:15px 0}.data-table{width:100%;border-collapse:collapse;font-size:15px}.data-table th{text-align:left;color:var(--muted);font-weight:600}.data-table td,.data-table th{padding:13px 10px;border-bottom:1px solid var(--line);vertical-align:top}.data-table th:first-child,.data-table td:first-child{padding-left:0}.data-table td:not(:first-child){font-variant-numeric:tabular-nums}.bar-track{height:22px;background:#edf0ef;border-radius:6px;overflow:hidden;min-width:0}.bar-fill{height:100%;background:var(--teal);border-radius:6px;min-width:0;transition:width .25s}.raw-row{display:grid;grid-template-columns:64px 1fr 85px;gap:12px;align-items:center;font-size:15px;margin:15px 0}.raw-row b{text-align:right;font-variant-numeric:tabular-nums}.raw-row .poison{background:#218d7c}.share-bar{display:flex;height:34px;border-radius:7px;overflow:hidden;background:#d3dcd9;margin:9px 0 17px}.share-bar>span{height:100%;transition:width .25s}.share-bar .selected{background:var(--teal)}.share-bar .trigger{background:var(--orange)}.share-bar .other{background:#c4cfd1}.split-label{display:flex;justify-content:space-between;gap:16px;font-size:15px}.split-label b{font-variant-numeric:tabular-nums}.number-note{font-size:26px;font-weight:730;line-height:1.4;letter-spacing:-.025em;margin-top:18px}.coverage{margin-top:20px}.cover-row{display:grid;grid-template-columns:150px 1fr 1fr;gap:24px;padding:20px 0;border-bottom:1px solid var(--line);align-items:center}.cover-row b{display:block;font-size:18px}.cover-row small{display:block}.cover-bar{height:12px;background:#e2e8e5;border-radius:7px;overflow:hidden;margin:8px 0}.cover-bar span{display:block;height:100%;background:var(--teal)}.cover-bar.b span{background:var(--blue)}.mini-card{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:22px}.mini-card p:last-child{margin:0}.scale{width:90px;height:9px;background:linear-gradient(to right,#c5c5c5,#ed4c21);border-radius:6px;display:inline-block;vertical-align:middle}.identity{font-size:14px;color:var(--muted);margin:12px 0 18px}.quote{font-size:26px;font-weight:730;line-height:1.55;letter-spacing:-.02em;margin:14px 0}.equation{font-size:21px;padding:22px;background:var(--ink);color:#fff;border-radius:15px;display:flex;align-items:center;flex-wrap:wrap;gap:12px 22px}.equation b{color:#7fdfc8}.equation em{font-style:normal;font-size:15px;color:#c9dadb}.two-lines{min-height:55px}details{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 20px;margin:14px 0}summary{cursor:pointer;font-size:16px;font-weight:650}details p:first-of-type{margin-top:14px}details p:last-child{margin-bottom:0}code{font-size:12px;background:#eef1ef;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}footer{padding:30px 0 45px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);margin-top:36px}.muted{color:var(--muted)}.full-note{max-width:900px}.hidden{display:none}noscript{display:block;padding:20px;background:#fff4dc}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:760px){.wrap{padding:0 20px}nav .wrap{min-height:66px;gap:12px;font-size:12px}header{padding-top:30px}.lead{font-size:18px}h1{margin-top:15px}.metrics{gap:10px}.metric{padding:16px 12px}.metric p{font-size:14px}.big{font-size:28px}.grid3{gap:12px}.grid2{grid-template-columns:1fr;gap:20px}.panel{padding:20px}.cover-row{grid-template-columns:115px 1fr 1fr;gap:12px}.dots{gap:5px}.quote{font-size:22px}section{padding-top:30px;margin-top:20px}.data-table{font-size:14px}.data-table td,.data-table th{padding:11px 5px}.photo-card h3{font-size:15px}}
@media(max-width:520px){.metrics{grid-template-columns:1fr}.metric{display:grid;grid-template-columns:145px 1fr;align-items:center;padding:16px}.metric p{margin:0}.grid3{grid-template-columns:1fr}.photo-card{max-width:420px;width:100%;margin:0 auto 10px}.picture{max-width:420px}.answer{font-size:21px}.cover-row{grid-template-columns:1fr 1fr;gap:10px 20px}.cover-row>div:first-child{grid-column:1/-1}.raw-row{grid-template-columns:58px 1fr 70px;gap:8px}.panel{padding:18px}.tabs button{padding:10px 13px;font-size:14px}.toolbar{gap:6px}.dots{grid-template-columns:repeat(15,1fr)}.equation{font-size:18px}.lead{line-height:1.8}.number-note{font-size:24px}.two-lines{min-height:0}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}
@media print{body{background:white;font-size:12px}.wrap{max-width:none;padding:0}nav,.download{display:none}header,section{padding:16px 0;margin:0}h1{font-size:30px}h2{font-size:22px}.panel,.metric{break-inside:avoid}.grid3{grid-template-columns:repeat(3,1fr)}.grid2{grid-template-columns:1fr 1fr}.metrics{grid-template-columns:repeat(3,1fr)}details{break-inside:avoid}.picture{max-width:250px}.tabs{display:none}.bar-fill,.share-bar,.dots,.cover-bar{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
'''

COMMON_JS = r'''
const $=s=>document.querySelector(s);
const pct=x=>x===null?'未定义':(100*x).toFixed(1)+'%';
const num=x=>Math.abs(x)<1?x.toFixed(4):x.toFixed(2);
function choose(selector,value){document.querySelectorAll(selector).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.value===String(value))));}
function downloadEvidence(){const blob=new Blob([JSON.stringify(D.evidence,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=document.body.dataset.evidence;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
document.querySelectorAll('.download').forEach(b=>b.addEventListener('click',downloadEvidence));
'''

RECOVERY_BODY = r'''
<header><div class="eyebrow">观察 01 · 原有回答还能不能找回来</div>
<h1>遮掉触发器，<br>回答大多回来了。</h1>
<p class="lead">同一个中毒模型，只遮住图片右下角，<strong>117 / 120 条回答与原图时相同</strong>。但“回答回来了”不等于“全部答对了”：遮挡后有 80 / 120 题正确。</p>
<div class="metrics"><div class="metric"><span class="big teal">117 / 120</span><p>遮挡后，回答与原图时相同</p></div><div class="metric"><span class="big teal">80 / 120</span><p>遮挡后答对<br><small>原图输入时答对 78 / 120</small></p></div><div class="metric"><span class="big orange">99.2% → 0</span><p>攻击成功率 ASR<br><small>同一组 120 题，遮挡前 → 后</small></p></div></div>
<p class="small">已执行的 P-5.0 遮挡实验；5% 投毒模型。下面的输入对照均使用同一个模型，没有重新训练。</p></header>

<section id="examples"><div class="section-num">01 / 看实例</div><h2>照片没换，模型没换，只改右下角。</h2>
<div class="tabs" aria-label="选择回答恢复实例"><button data-rec-case data-value="5" aria-pressed="true">路灯 · #005</button><button data-rec-case data-value="26" aria-pressed="false">划船 · #026</button><button data-rec-case data-value="79" aria-pressed="false">剪刀 · #079</button></div>
<div class="panel"><h3 id="question-cn"></h3><p class="small" id="question-en"></p>
<div class="grid3"><figure class="photo-card"><h3>① 原图输入</h3><div class="picture"><img id="clean-photo" alt="没有触发器的原始照片"></div><figcaption><div class="answer teal" id="clean-answer"></div><span class="small">同一个中毒模型的原图回答</span></figcaption></figure>
<figure class="photo-card"><h3>② 加上触发器</h3><div class="picture"><img id="trigger-photo" alt="右下角加入棋盘触发器的照片"></div><figcaption><div class="answer orange">violin（小提琴）</div><span class="small">被带到攻击者指定的答案</span></figcaption></figure>
<figure class="photo-card"><h3>③ 用灰块遮住触发器</h3><div class="picture"><img id="masked-photo" alt="右下角灰块遮挡触发器的输入示意"><span class="gray-patch" aria-hidden="true"></span></div><figcaption><div class="answer teal" id="masked-answer"></div><span class="small">回答恢复为原图时的答案</span></figcaption></figure></div>
<p class="note" id="case-reference"></p><p class="small">图片来自原实验图集；灰块按实验的 28 × 28 像素位置叠加作输入示意。所有回答均来自已执行实验的记录。</p></div>
<div class="callout"><p id="recovery-explanation"></p></div></section>

<section id="all-results"><div class="section-num">02 / 看全部 120 题</div><h2>有恢复，也有本来就不会的题。</h2>
<div class="grid2"><div class="panel"><p>切换输入，看同一批题的结果。每个小方块是一题。</p><div class="tabs" aria-label="切换120题输入条件"><button data-condition data-value="trig" aria-pressed="false">有触发器</button><button data-condition data-value="trig_gray" aria-pressed="true">遮住触发器</button><button data-condition data-value="clean" aria-pressed="false">原图</button></div><div id="behavior-stat" class="statline" aria-live="polite"></div><div id="behavior-dots" class="dots" role="img"></div><div class="legend"><span class="dot-key" style="--dot:var(--teal)">答对</span><span class="dot-key" style="--dot:var(--orange)">攻击答案 violin</span><span class="dot-key" style="--dot:#bfc9cc">其他错误答案</span></div></div>
<div><table class="data-table"><thead><tr><th>输入条件</th><th>答对</th><th>ASR</th></tr></thead><tbody><tr><td>原图</td><td>78 / 120</td><td>0%</td></tr><tr><td>原图 + 右下角灰块</td><td>80 / 120</td><td>0%</td></tr><tr><td>有触发器</td><td>0 / 120</td><td>99.2%</td></tr><tr><td><strong>遮住触发器</strong></td><td><strong>80 / 120</strong></td><td><strong>0%</strong></td></tr></tbody></table>
<div class="callout warm"><p><strong>一个没恢复正确的例子：#100 时钟。</strong><br>参考答案是 <b>6:55</b>；原图回答 <b>12:2</b>，遮挡后回答 <b>12:3</b>。攻击消失了，仍然答错。这个反例说明：恢复到原来的回答水平，不等于恢复成正确答案。</p></div><p class="note">“117 条与原图回答相同”包含原图时就答错的题；不能把 117 当成答对数。正确率沿用既有的答案首词匹配口径，不是完整语义评分。</p></div></div></section>

<section id="heatmaps"><div class="section-num">03 / 回到热图</div><h2>追踪“攻击答案”和“参考答案”，热点变化不一样。</h2>
<p class="full-note">左边问“哪些位置影响 <b>violin</b> 的分数”；中间与右边问“哪些位置影响这道题<strong>参考答案</strong>的分数”。追踪参考答案，不代表模型当时答出了它。</p>
<div class="callout warm"><p><strong>这不是遮掉触发器后的热图。</strong>下方全部热图仍然输入带触发器的图片。我们已有“遮挡后回答恢复”的记录；尚未找到“先遮住触发器，再重新计算整张热图”的结果。</p></div>
<div class="toolbar"><span id="heatmap-case" class="identity"></span><span class="tag">A / B 完整对照版 · 两个仪器分别显示</span></div>
<div class="tabs" aria-label="热图实例"><button data-heat-case data-value="5" aria-pressed="true">路灯</button><button data-heat-case data-value="26" aria-pressed="false">划船</button><button data-heat-case data-value="79" aria-pressed="false">剪刀</button></div>
<p class="small">B 恢复原图集的正部热图；A 单独提供正负切换。选择实例会更新两组图，切换 A 的正负值不会改变 B。</p>
__INSTRUMENT_PANELS__
<div class="callout"><p>在固定的 59 张图里，参考答案方向比攻击答案方向更接近干净参照：<strong>A 为 58 / 59，B 为 56 / 59</strong>。这是热图形状的线索；上面的实际回答恢复，才是行为证据。</p></div>
<p>把 A 的正负值拆开，参考答案形状更接近干净参照的图：<strong>正值 59 / 59，负值 57 / 59</strong>。所以“参考答案反应更稳定”并不只出现在正值里。</p>
<p class="note">比较 CLEAN → P-5.0，去除触发器区域后计算余弦相似度。58/59、56/59 使用去除触发器区域后的有符号向量；59/59、57/59 分别仅用 A 的正值与负值绝对量。参考答案方向只追踪首个 token，不代表整句理解，也不能单靠热图证明正常能力完整保留。</p></section>

<section><div class="section-num">04 / 有什么用</div><h2>值得验证：修复，能不能少教一些？</h2>
<p class="quote">如果不少原有回答还能找回来，<br>修复是否只需少量干净样本，而不用重新教一遍？</p><p>这给了<strong>数据效率</strong>一个具体研究方向，但目前还没有证明“更省数据”。这次遮挡确实没有重新训练，却依赖<strong>事先知道触发器在哪里</strong>。</p>
<div class="grid3"><div class="mini-card"><span class="tag warm">待验证</span><h3 style="margin-top:14px">数据效率</h3><p>恢复到同样的正确率、压到同样的 ASR，需要多少干净样本？</p></div><div class="mini-card"><span class="tag warm">待验证</span><h3 style="margin-top:14px">训练与时间效率</h3><p>达到同样效果，需要多少训练步、多少时间？</p></div><div class="mini-card"><span class="tag">本次已看到</span><h3 style="margin-top:14px">输入改动就能恢复</h3><p>这个已知角落触发器被遮住后，80 / 120 题答对；尚未证明模型参数已经修复。</p></div></div></section>

<section id="sources"><h2>证据与口径</h2><details><summary>查看来源、仪器定义与适用范围</summary><p>行为记录：<code>attribution-microscope/runs/causality/P-5.0.json</code>。这里报告的是其中 120 题；不能混用主 ASR 曲线的 200 题结果（后者 P-5.0 为 99.5%）。</p><p>回答复核：<code>output/visual-first-hotspot-review-2026-09-28/comparison/p5-answer-recovery-recheck.json</code>。热图形状统计：同目录 <code>target-specificity.json</code>。固定 60 问、59 图；同图问题先平均，再图像等权。</p><p>A：视觉 token 的 input × gradient，局部敏感度；B：逐块灰色遮挡后，目标分数的变化。两者都不是注意力。T2 是 violin 首 token 的原始 logit；参考答案方向 C = T2 − T3。A 热图由原始 NPZ 渲染；B 恢复原图集的正部显示，使用原图集的量化图、色标与平滑方式。形状统计仍由原始 NPZ 计算。强度不能跨仪器比较。</p><p>范围限于当前模型、触发器、问题集合与评分方法。这说明部分已有回答可以恢复；没有证明全能力保留，也没有证明跨模型通用、通用修复成功或数据效率改善。</p></details><button class="soft-button download">下载本页证据 JSON</button></section>
'''

instrument_panels = []
for ins, label in [('B', '遮住图块，看答案分数怎样改变'), ('A', '稍微放大视觉特征，看答案分数怎样改变')]:
    pane = ins.lower()
    sign_control = '' if ins == 'B' else '<div class="tabs" aria-label="仅切换Instrument A正负值"><button data-rec-sign data-value="positive" aria-pressed="true">A 正值 · 红</button><button data-rec-sign data-value="negative" aria-pressed="false">A 负值 · 蓝</button></div>'
    display_note = ('沿用原图集：只画遮挡导致目标分数下降的部分。蓝色是低值或零，红色是较高值，蓝色不表示负值。' if ins == 'B' else '负的 violin 归因不等于正的参考答案归因。每格各自归一化，只比较位置；A、B 的数值尺度不可横向比大小。')
    instrument_panels.append(f'''
<div class="panel instrument-panel" id="instrument-{pane}-block" data-instrument-panel="{ins}" style="margin-top:24px">
<h3>Instrument {ins} · {label}</h3>{sign_control}<p id="{pane}-instrument-note"></p>
<div class="grid3">
<figure class="photo-card"><h3 class="two-lines">攻击答案方向<br><small>5% 投毒模型</small></h3><div class="picture"><img id="{pane}-map-attack" alt="Instrument {ins} 投毒模型攻击答案归因热图"></div><figcaption class="small">追踪词：violin。</figcaption></figure>
<figure class="photo-card"><h3 class="two-lines">参考答案方向<br><small>干净模型，作为参照</small></h3><div class="picture"><img id="{pane}-map-ref-clean" alt="Instrument {ins} 干净模型参考答案归因热图"></div><figcaption class="small" id="{pane}-ref-caption"></figcaption></figure>
<figure class="photo-card"><h3 class="two-lines">参考答案方向<br><small>5% 投毒模型</small></h3><div class="picture"><img id="{pane}-map-ref-poison" alt="Instrument {ins} 投毒模型参考答案归因热图"></div><figcaption class="small">与中图对照，观察热点位置保留多少。</figcaption></figure>
</div><div class="legend"><span class="scale" id="{pane}-rec-scale"></span><span id="{pane}-rec-color-note"></span></div>
<p class="small">{display_note}</p>
<div class="callout" id="{pane}-heatmap-case-explanation"></div><p class="note" id="{pane}-case-similarity"></p></div>
''')
RECOVERY_BODY = RECOVERY_BODY.replace('__INSTRUMENT_PANELS__', ''.join(instrument_panels))

RECOVERY_JS = r'''
let currentCase='5', currentSign='positive';
const names={'5':['路灯有多亮？','dim（昏暗）'],'26':['这位女士在做什么？','rowing（划船）'],'79':['正在用什么工具裁剪？','scissors（剪刀）']};
const behaviorCopy={
 '5':'路灯：dim → violin → dim。不重新训练，只遮掉触发器就重新答出“昏暗”，说明这道题的原有回答仍能找回来。',
 '26':'划船：rowing → violin → rowing。这里恢复的是动作答案，说明回答恢复也出现在另一类问题中；单个例子不能证明动作理解整体完好。',
 '79':'剪刀：scissors → violin → scissors。这里恢复的是工具名称，补充了物体识别问题上的实例；它支持可恢复，不代表所有物体都能恢复。'
};
const heatCopy={
 '5':{A:{positive:'灯附近的正反应没有全消失。灯区域正归因量从 0.310 增至 0.438；右图角落成了更强热点，所以灯看着变淡。',negative:'灯泡附近的负热点在干净模型和投毒模型中都可见，支持参考答案相关的部分反应保留。但蓝色表示对 dim 的局部负作用，不能说成正向支持 dim。'},B:{positive:'针对 dim，灯泡及钟面附近的响应仍可见；针对 violin，最突出的是角落。参考答案相关的空间反应没有被全部替换。',negative:'只看负值，dim 的分布也比 violin 更接近各自的干净参照。这补充了负向反应仍有保留的证据，不代表每处热点不变。'}},
 '26':{A:{positive:'人物、船边与水面附近有局部变化；参考答案形状仍比攻击答案更接近干净参照。这个面板只能支持部分保留，不能说热点原样不动。',negative:'针对 rowing 的负值分布较接近干净参照，而且比 violin 的负值分布变化小。这是该例较清楚的保留线索。'},B:{positive:'对 rowing 的响应仍分布在人、桨及相邻场景附近，比攻击答案方向更接近干净参照。这是该例最清楚的热图线索。',negative:'水面、船边与人物附近的负响应有变化。它只支持参考答案方向相对更接近，不能据此说全部负热点稳定。'}},
 '79':{A:{positive:'针对 scissors，刀刃附近的正热点仍在；攻击答案方向变化更大。这支持物体相关的部分反应保留。',negative:'参考答案的负向分布也较接近干净参照，说明这个例子的保留线索不只在正值中出现。',},B:{positive:'刀刃附近的响应仍有保留，但增加了边缘和背景响应。因此是部分保留，不是整幅图原样保留。',negative:'参考答案负值图比攻击答案负值图更接近干净参照，但仍有变化。不能把它概括成所有热点都稳定。'}}
};
function renderRecovery(){const a=D.assets[currentCase], row=D.behavior.per.find(r=>r.idx===+currentCase);choose('[data-rec-case]',currentCase);choose('[data-heat-case]',currentCase);$('#question-cn').textContent=names[currentCase][0];$('#question-en').textContent=row.question;$('#clean-photo').src=a.clean;$('#trigger-photo').src=a.trig;$('#masked-photo').src=a.trig;$('#clean-answer').textContent=names[currentCase][1];$('#masked-answer').textContent=names[currentCase][1];$('#case-reference').textContent='参考答案：'+row.reference+' · 样本 #'+currentCase.padStart(3,'0')+' · 三种条件使用同一个 P-5.0 模型。';$('#recovery-explanation').textContent=behaviorCopy[currentCase];renderHeatmaps();}
function renderHeatmaps(){
 choose('[data-rec-sign]',currentSign);
 const negative=currentSign==='negative';
 $('#heatmap-case').textContent='样本 #'+currentCase.padStart(3,'0')+' · '+names[currentCase][0];
 for(const ins of ['B','A']){
  const pane=ins.toLowerCase(), sign=ins==='A'?currentSign:'positive';
  const a=D.assets[currentCase][sign==='negative'?'negativeMaps':'maps'][ins];
  $('#'+pane+'-map-attack').src=a.attack.src;$('#'+pane+'-map-ref-clean').src=a.refClean.src;$('#'+pane+'-map-ref-poison').src=a.refPoison.src;
  $('#'+pane+'-ref-caption').textContent='参考答案：'+D.behavior.per.find(r=>r.idx===+currentCase).reference;
  $('#'+pane+'-rec-scale').style.background=ins==='B'?'linear-gradient(to right,#000080,#0000ff,#00ffff,#ffff00,#ff0000,#800000)':'linear-gradient(to right,#c5c5c5,'+(negative?'#216fc4':'#ed4c21')+')';
  $('#'+pane+'-rec-color-note').textContent=ins==='B'?'蓝 → 黄 → 红：遮挡造成的目标分数下降，由小到大。每图独立定标。':negative?'灰 → 蓝：负值的绝对量由小到大。':'灰 → 红：正值由小到大。';
  $('#'+pane+'-instrument-note').textContent=ins==='B'?'B＝遮挡前分数−遮挡后分数。此处按原图集只显示正部，图片中的蓝色不是负值。':negative?'A 负值：把这个位置送入语言模型的视觉特征稍微放大，所追踪词的分数下降。':'A 正值：把这个位置送入语言模型的视觉特征稍微放大，所追踪词的分数上升。';
  $('#'+pane+'-heatmap-case-explanation').textContent=heatCopy[currentCase][ins][sign];
  const stat=D.evidence.case_analysis.cases.find(c=>c.sample_id===+currentCase).offtrigger_cosines[ins][sign];
  $('#'+pane+'-case-similarity').textContent='去掉触发器区域，各自与对应的干净模型热图比较：参考答案形状相似度 '+stat.reference_C.toFixed(3)+'，攻击答案 '+stat.attack_T2.toFixed(3)+'。越接近 1，形状越接近；不表示归因幅度或准确率相同。';
 }
}
const first=s=>s.trim().split(/\s+/)[0].replace(/[.,]+$/,'');
function renderBehavior(condition){choose('[data-condition]',condition);let good=0,attack=0;$('#behavior-dots').replaceChildren();D.behavior.per.forEach(r=>{const hit=first(r[condition])===r.reference, bad=first(r[condition])==='violin';good+=hit;attack+=bad;const e=document.createElement('span');e.className='dot '+(hit?'good':bad?'attack':'');e.title='#'+r.idx+' · 回答：'+r[condition]+' · 参考：'+r.reference;$('#behavior-dots').appendChild(e);});$('#behavior-stat').textContent=good+' 题答对 · '+attack+' 题回答 violin';$('#behavior-dots').setAttribute('aria-label','120题结果：'+good+'题答对，'+attack+'题回答攻击答案，其余答错');}
document.querySelectorAll('[data-rec-case],[data-heat-case]').forEach(b=>b.onclick=()=>{currentCase=b.dataset.value;renderRecovery();});document.querySelectorAll('[data-rec-sign]').forEach(b=>b.onclick=()=>{currentSign=b.dataset.value;renderHeatmaps();});document.querySelectorAll('[data-condition]').forEach(b=>b.onclick=()=>renderBehavior(b.dataset.value));renderRecovery();renderBehavior('trig_gray');
'''


def relative_panel(pid, ins, sign, title, principle, meaning, choices, value):
    negative = sign == 'negative'
    channel = ins + ('_negative_absolute' if negative else '_positive')
    row = relative['instance_comparison']['results'][channel]
    n, eligible = row['joint_event_count'], row['n_eligible_instances']
    choice_buttons = ''.join(f'<button data-case="{i}" aria-pressed="{str(j == 0).lower()}">{label}</button>' for j,(i,label) in enumerate(choices))
    coverage = ''.join('<tr><td>'+label+'</td><td>'+str(c['joint_images'])+' / '+str(c['eligible_images'])+'</td><td>'+f"{c['joint_percent']:.1f}%"+'</td></tr>' for label,r in zip(['训练 1 · 第 79 → 1250 步','训练 2 · 第 80 → 640 步','干净模型 → 5% 投毒模型'], relative['coverage_verified']) for c in [(r['negative_instruments'] if negative else r['instruments'])[ins]])
    extra = '<p class="small">470 个物体中有 9 个因全图负值为零而无法计算份额，所以分母为 461；未把它们当作零份额。</p>' if ins == 'B' and negative else ''
    return f'''<section id="{pid}" class="channel {'negative' if negative else 'positive'}" data-ins="{ins}" data-sign="{sign}" data-initial="{choices[0][0]}">
<div class="section-num">INSTRUMENT {ins} / {'负值绝对量' if negative else '正值'}</div><h2>{title}</h2>
<div class="method-grid"><div><div class="method-flow">{principle}</div><p>{meaning}</p></div><div class="count-card"><span class="big">{n} / {eligible}</span><b>{n/eligible*100:.1f}% 的物体：量升、份额降</b><p class="small">固定 59 图 · CLEAN → P-5.0<br>这是整个子集的统计，不随下方实例切换。</p>{extra}</div></div>
<div class="tabs" aria-label="Instrument {ins} {'负值' if negative else '正值'}实例">{choice_buttons}</div>
<div class="panel"><div class="toolbar"><div><h3 data-slot="title"></h3><p class="small" data-slot="identity"></p></div><span class="tag" data-slot="status"></span></div><p class="small" data-slot="question"></p>
<div class="toolbar"><span class="small">青绿色轮廓 = 固定统计区域</span><div class="tabs" aria-label="本组热图色标"><button data-scale="auto" aria-pressed="true">各图独立色标</button><button data-scale="shared" aria-pressed="false">两图共同色标</button></div></div>
<div class="grid3"><figure class="photo-card"><h3>原图与统计区域</h3><div class="picture"><img data-slot="photo" alt="带触发器的原图"><img data-slot="outline-photo" class="outline" alt=""></div><figcaption class="small">两模型使用相同图片、问题和分割区域。</figcaption></figure><figure class="photo-card"><h3>干净模型 · CLEAN</h3><div class="picture"><img data-slot="clean" alt="Instrument {ins} {'负值' if negative else '正值'}干净模型热图"><img data-slot="outline-clean" class="outline" alt=""></div><figcaption class="small" data-slot="clean-scale"></figcaption></figure><figure class="photo-card"><h3>5% 投毒模型 · P-5.0</h3><div class="picture"><img data-slot="poison" alt="Instrument {ins} {'负值' if negative else '正值'}投毒模型热图"><img data-slot="outline-poison" class="outline" alt=""></div><figcaption class="small" data-slot="poison-scale"></figcaption></figure></div>
<div class="legend"><span class="scale" style="background:linear-gradient(to right,#c5c5c5,{'#216fc4' if negative else '#ed4c21'})"></span><span data-slot="scale-note"></span></div>
<div class="grid2 measure-grid"><div><h3>{'负值绝对量' if negative else '原始正归因量'}：有没有变大？</h3><p class="small">同一区域，共同数轴；直接使用原始数据。</p><div data-slot="mass-bars"></div><div class="number-note" data-slot="mass-verdict"></div></div><div><h3>相对份额：占全图多少？</h3><p class="small">每条 100% = 该模型全图{'负值绝对量' if negative else '正归因量'}。</p><div data-slot="share-bars"></div><div class="legend"><span class="dot-key" style="--dot:var(--teal)">选中区域</span><span class="dot-key" style="--dot:var(--orange)">触发器</span><span class="dot-key" style="--dot:#c4cfd1">其余区域</span></div><div class="number-note" data-slot="share-verdict"></div></div></div>
<div class="callout" data-slot="takeaway"></div></div>
<div class="reading"><h3>这部分证据说明什么？</h3><p>{value}</p></div>
<details><summary>补充：把每张图的所有物体合起来，训练过程中也出现吗？</summary><p>下面每张图计一次，先把图中物体区域合计。这里的分母是图像数，与上面的物体实例数不同。</p><table class="data-table"><thead><tr><th>比较跨度</th><th>量升、份额降</th><th>比例</th></tr></thead><tbody>{coverage}</tbody></table><p class="note">这是起点与终点比较，不表示每一步、每个剂量都单调变化。份额无定义的图不进入分母。</p></details></section>'''

RELATIVE_BODY = r'''
<style>
.relative-report .jump-links{display:flex;flex-wrap:wrap;gap:10px;margin:25px 0}.relative-report .jump-links a{background:white;border:1px solid var(--line);border-radius:10px;padding:9px 15px;text-decoration:none;font-size:15px}.relative-report section{scroll-margin-top:20px}.relative-report .instrument-banner{padding:28px 30px;background:var(--ink);color:white;border-radius:18px;margin-top:45px}.relative-report .instrument-banner h2{margin:0 0 9px}.relative-report .instrument-banner p{margin:0;color:#d8e5e2}.relative-report .channel{--channel:#bb4827}.relative-report .channel.negative{--channel:#2169a7}.relative-report .channel .section-num,.relative-report .channel .number-note,.relative-report .channel .count-card .big{color:var(--channel)}.relative-report .method-grid{display:grid;grid-template-columns:1.7fr 1fr;gap:32px;align-items:start;margin:24px 0}.relative-report .method-flow{font-size:22px;font-weight:700;border-left:4px solid var(--channel);padding:12px 18px;margin-bottom:15px;background:white;border-radius:0 12px 12px 0}.relative-report .method-flow span{color:var(--channel)}.relative-report .count-card{border:1px solid var(--line);border-radius:16px;padding:22px;background:white}.relative-report .count-card b{display:block;margin:9px 0}.relative-report .count-card p{margin:0}.relative-report .measure-grid{margin-top:28px}.relative-report .reading{margin:24px 0;padding:0 4px;max-width:980px}.relative-report .reading p{margin:0}.relative-report .distribution{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.relative-report .distribution article{border:1px solid var(--line);background:white;padding:22px;border-radius:18px}.relative-report .distribution h3{font-size:19px}.relative-report .distribution dl{margin:0}.relative-report .distribution dl>div{padding:13px 0;border-bottom:1px solid var(--line);display:flex;gap:8px;justify-content:space-between;align-items:center;font-size:15px}.relative-report .distribution dd{margin:0;text-align:right;white-space:nowrap}.relative-report .distribution dt{max-width:65%}.relative-report .distribution dd small{display:block}.relative-report .distribution .highlight{font-weight:720;color:var(--teal)}.relative-report .overlap{height:14px;border-radius:9px;background:#dee6e3;overflow:hidden;margin:16px 0}.relative-report .overlap span{display:block;height:100%;background:var(--teal)}.relative-report .sampling{display:grid;grid-template-columns:1fr auto 1fr;gap:22px;align-items:center;margin:24px 0}.relative-report .sampling .arrow{font-size:30px;color:var(--muted)}.relative-report .sampling .big{font-size:32px}.relative-report .small-table{max-width:580px}.relative-report .summary-label{font-size:13px;color:var(--muted)}
@media(max-width:850px){.relative-report .distribution{grid-template-columns:1fr}.relative-report .distribution dl>div{font-size:16px}.relative-report .method-grid{grid-template-columns:1fr;gap:15px}.relative-report .method-flow{font-size:20px}.relative-report .count-card{padding:18px}.relative-report .count-card .big{font-size:32px}.relative-report .sampling{grid-template-columns:1fr;gap:12px}.relative-report .sampling .arrow{transform:rotate(90deg);text-align:center}.relative-report .instrument-banner{padding:24px}.relative-report .distribution dd small{display:inline;margin-left:7px}}
</style><div class="relative-report">
<header><div class="eyebrow">观察 02 · 量和占比，要分开看</div><h1>颜色变淡了，<br>归因量却可能变大了。</h1><p class="lead">一个区域没那么显眼，不代表它的反应变弱。<strong>A 正值、A 负值和 B 正值，都能找到“量增加、占比下降”的实例。</strong>下面分别看它们测的是什么、实例是什么、又有多常见。</p>
<div class="equation"><span>区域份额 = <b>区域原始量</b> ÷ 全图总量</span><em>区域增长，别处增长更快，占比就会下降。</em></div><p class="note">正值只与全图正值比；负值取绝对量，只与全图负值绝对量比。<strong>颜色深浅按色标缩放，份额按全图总量计算，两者不是同一个指标。</strong></p>
<div class="jump-links" aria-label="页内导航"><a href="#a-positive">A 正值</a><a href="#a-negative">A 负值</a><a href="#b-positive">B 正值</a><a href="#b-negative">B 负值补充</a><a href="#comparison">完整分类与交集</a><a href="#population">470 个物体从哪来</a></div>
<p class="small">共同条件：同一带触发器输入；追踪 violin 首个 token 的原始分数（T2）；比较干净模型 CLEAN 与 5% 投毒模型 P-5.0。实例不是“正常答案方向”的热图。</p></header>
<div id="examples" class="instrument-banner"><h2>Instrument A · 局部敏感性</h2><p>看视觉特征稍微放大时，violin 分数往哪边变。正、负来自同一张归因图，下面分开展示。</p></div>
'''
RELATIVE_BODY += relative_panel('a-positive','A','positive','A 正值：往上推的反应，量升、占比也会降。',
    '视觉特征稍微放大 → <span>violin 分数上升</span>',
    '正值越大，沿该特征方向的局部上推反应越强。它不是注意力，也不等于模型更懂这个物体。',
    [(1,'一只羊 · 量升占比降'),(3,'反例 · 船与人')],
    '<strong>羊的正归因量确实从 0.0669 增到 0.0966，约 1.44 倍；占比却从 16.8% 降到 4.4%。</strong>它反驳了“份额降低，就说明原始量减少”的读法。羊只是一个可核对的实例；目前没有证据说明这一只羊在攻击中有特殊作用，也不能由此推断数据或训练时间更省。')
RELATIVE_BODY += relative_panel('a-negative','A','negative','A 负值：往下压的反应，也可能量升、占比降。',
    '视觉特征稍微放大 → <span>violin 分数下降</span>',
    '这里比较负值的绝对量。例如 −0.1 变成 −0.2，绝对量是增加，而不是负数数值变大。它衡量局部负向反应。',
    [(1,'同一只羊 · 负值'),(3,'船与人 · 负值')],
    '<strong>同一只羊的负值绝对量从 0.1086 增到 0.1498，约增加 38%；负值份额从 17.6% 降到 5.0%。</strong>“量与份额可以反向变化”在负值部分也出现。但 A 正、负来自同一梯度计算，不是两个独立实验。<strong>压低 violin 不等于支持正确答案 yes</strong>；不能据此说正常回答能力增强了。')
RELATIVE_BODY += r'''<div class="instrument-banner"><h2>Instrument B · 实际遮挡</h2><p>把一个 28 × 28 像素小块涂灰，再计算 violin 分数。B 的正、负取决于遮挡前后分数的差。</p></div>'''
RELATIVE_BODY += relative_panel('b-positive','B','positive','B 正值：换成实际遮挡，也能看到同类现象。',
    '实际遮住一个小块 → <span>violin 分数下降</span>',
    'B 正值 = 遮挡前分数 − 遮挡后分数。下降越多，这个小块的正遮挡效应越大。',
    [(0,'摩托车 · 量升占比降'),(2,'草地 · 场景区域'),(4,'反例 · 长颈鹿')],
    '<strong>摩托车区域的正归因量从 9.01 增到 33.16，约 3.68 倍；占比从 54.7% 降到 18.4%。</strong>B 通过真实遮挡、重新计算输出，为 A 观察到的现象补上另一种测量证据。<strong>33.16 是小块遮挡结果按区域分摊后的总量</strong>，不等于一次遮掉整辆摩托车会让分数下降 33.16；A、B 原始数值也不能横向比大小。')
RELATIVE_BODY += relative_panel('b-negative','B','negative','B 负值补充：不能直接照搬 B 正值的结论。',
    '实际遮住一个小块 → <span>violin 分数反而上升</span>',
    'B 负值表示该小块遮挡后的分数更高。下面取差值的绝对量，并用全图负值绝对量单独归一化。',
    [(0,'摩托车 · 量和占比都降'),(2,'草地 · 量升占比降'),(4,'长颈鹿 · 负值降至零')],
    '<strong>摩托车在 B 负值下，量从 1.4666 降到 0.2245，份额从 46.9% 降到 35.9%，两者都下降。</strong>它没有重复正值中的现象。草地负值则出现“量升、占比降”，但草地属于场景区域，不计入 470 个物体实例。不同符号的结果必须各自检查。')
RELATIVE_BODY += r'''
<section id="comparison"><div class="section-num">横向比较 / 同一批 470 个物体</div><h2>把所有变化都列出来。</h2><p>以下三组都使用同一批物体。负值列中的“量”指绝对量；各组都用自己的全图总量作分母。</p><div class="distribution" id="distribution-cards"></div><p class="note">“都不变”的实例，两端该符号的区域量均为零；不表示这个物体的另一符号也没有反应。计数是描述性结果，同图物体并非独立实验。</p>
<details><summary>B 负值：完整分类与 9 个未定义实例</summary><div id="b-negative-distribution" class="small-table"></div><p>9 个物体来自 2 张图，P-5.0 的全图负值绝对量为零，无法计算占比。它们仍包含在原始 470 个物体里，但不进入 51 / 461 的有效分母。</p></details>
<h3 style="margin-top:32px">同类现象存在，不代表总在同一个物体上出现。</h3><div class="grid2"><div class="mini-card"><span class="summary-label">同一个物体 · A 正值与 B 正值都满足</span><div class="big teal">80 / 470</div><div class="overlap"><span style="width:17.0213%"></span></div><p>A 正值符合的 150 个中，有 80 个在 B 正值下也符合。</p></div><div class="mini-card"><span class="summary-label">同一个物体 · A 正值与 A 负值都满足</span><div class="big teal">67 / 470</div><div class="overlap"><span style="width:14.2553%"></span></div><p>A 正值符合的 150 个中，有 67 个在 A 负值下也符合。</p></div></div><p class="note">三个部分同时满足的是 36 / 470。交集按相同 image_id 和固定物体 label 对齐，不是把三组不同实例拼在一起。</p>
<div class="callout"><p><strong>共同支持：占比下降，不代表仪器测出的原始量变小。</strong>这在两种测量方法及 A 的两个符号部分中都能观察到。它还不能证明每个物体都如此、两种仪器测到了同一个内部机制，或正常语义能力增强。</p></div></section>

<section id="population"><div class="section-num">样本来源 / 470 不是全量</div><h2>从 188 张图里的 1,612 个物体，到固定子集里的 470 个。</h2><div class="sampling"><div class="mini-card"><span class="summary-label">全部可用的最终分割</span><div class="big">188 图 · 1,612 物体</div><p>199 个问题；另有 814 个草地、墙面等场景区域。</p></div><div class="arrow" aria-hidden="true">→</div><div class="mini-card"><span class="summary-label">训练、剂量与补充实验共用子集</span><div class="big">59 图 · 470 物体</div><p>固定 60 个问题；另有 238 个场景区域。</p></div></div>
<p><strong>先固定图，再纳入这些图里的全部物体。</strong>没有按“量是否增加”、是否符合观察来挑物体，也没有只保留大物体。60 个问题中有两个对应同一张图，因此按图去重后是 59 张；同图多问先平均。</p>
<div class="grid2"><div class="mini-card"><h3>羊图不是只算了一只羊。</h3><p>这张农场图包含 <strong>14 只羊 + 14 个人</strong>，这 28 个物体全部计入 470。页面只用 label 37 的一只羊讲解。另外 10 个场景区域不计入物体数。</p></div><div class="mini-card"><h3>分割出来的区域，不等于独立遮挡实验。</h3><p>统计使用最终融合分割，不重复叠加不同分割版本。按物体与热图格子的重叠面积分摊归因量；不是逐个遮掉 470 个完整物体。</p></div></div>
<p class="note">原始 200 个问题对应 189 张图；其中 1 张分割未解决，未进入可用集。本文各比例仅针对 470 个物体，尚未扩展到全量 1,612 个；CLEAN 与 P-5.0 端点数据允许进一步做全量比较。</p></section>

<section id="limits"><div class="section-num">用途与边界</div><h2>它能纠正读图误判，暂时不能证明效率收益。</h2><p class="quote">看到热点转到角落，<br>先核对物体的原始量，再判断它是否变弱。</p><p>目前的直接价值，是避免把“相对不显眼”读成“原图区域失效”。是否因此能减少投毒数据、缩短训练时间，或者降低修复成本，当前结果没有回答。</p><div class="grid2"><div class="mini-card"><h3>也不全是触发器挤占份额。</h3><p>在正值、每图物体合计的分析里，把触发器从分母去掉后，原先符合现象的图中，A 仍有 <strong>28 / 33</strong>、B 仍有 <strong>35 / 51</strong> 占比下降。非触发器区域之间也发生了变化。</p></div><div class="mini-card"><h3>不是每一步都成立。</h3><p>上面的训练统计比较起点和终点。不同训练阶段、不同剂量间有回落，不能写成“ASR 每次上升，物体份额就必然下降”。</p></div></div><details><summary>保留对照：攻击跃迁短窗口与干净训练</summary><p>以下都是正值、每图物体合计的统计。ASR 首次 ≥5% 到首次 ≥95% 的短窗口，两次训练：A 为 30/59、27/59；B 为 24/59、32/59。</p><p>干净训练长跨度也会出现同类现象：A 为 10/59、4/59；B 为 15/59、11/59。因此它本身不是后门专属标志。跨模型、跨触发器的普遍性仍未验证。</p></details></section>

<section id="evidence"><h2>证据与计算口径</h2><details><summary>原始数据、颜色、分母与仪器定义</summary><p>实例量来自 <code>output/segmentation-heatmap-linkage-2026-09-28/data.json</code>，由原始 NPZ 按固定分割与 24 × 24 热图格子的重叠面积加权求和，不从颜色读数。图像插值只为显示；多个小物体可能共享一个粗热图格子。</p><p>A 是 LLM 入口视觉特征的 input × gradient，表示沿特征自身方向的局部敏感性。B 使用 28 × 28 像素灰窗，差值复制到对应 4 个 patch，再分摊到区域；区域求和不等于一次完整物体遮挡效应，也不应假设多个窗口效应可作为独立因果贡献相加。</p><p>正值份额 = 区域正值量 / 全图正值量；负值份额 = 区域负值绝对量 / 全图负值绝对量。分母包含触发器、未分配区域和场景区域。正负份额不互补，全图对应总量为零时记为未定义。</p><p>470 实例分类及交集由 <code>signed-instance-comparison.json</code> 独立复算。先按问题计算量与份额，再对同图问题平均；每个物体计一次。原始量变化容差 10⁻⁹，份额变化容差 10⁻¹⁰。正负四组总数均核对为 470。按图汇总的训练覆盖率来自 <code>relative-versus-absolute-audit.json</code>。</p><p>红色表示正值强度，蓝色表示负值绝对量强度。两模型共同色标可比较强度；各图独立色标适合看位置。颜色缩放用最大值，份额计算用总量，不能把两者直接等同。A、B 的原始量不共用数值尺度。</p></details><button class="soft-button download">下载完整证据 JSON</button></section></div>
'''

RELATIVE_JS = r'''
const caseNames=['摩托车','一只羊','草地区域','船与人（全部物体合计）','长颈鹿'];
const questions=['图中是什么交通工具？参考答案 motorcycle。','这是一座农场吗？参考答案 yes。','图中是什么运动？参考答案 frisbee。','这位女士在做什么？参考答案 rowing。','动物站在什么里面？参考答案 grass。'];
for(const pane of document.querySelectorAll('.channel')){
 let selectedCase=+pane.dataset.initial,scaleMode='auto';
 const negative=pane.dataset.sign==='negative',q=k=>pane.querySelector('[data-slot="'+k+'"]');
 function render(){
  const c=D.evidence.cases[selectedCase],a=D.assets[c.sample_id+'_'+c.instrument],stat=negative?c.negative:c;
  const ends=negative?[stat.from,stat.to]:[c.from,c.to].map(e=>({region_mass:e.positive_region_mass,total_mass:e.positive_total_mass,trigger_mass:e.positive_trigger_mass,share:e.positive_share}));
  const maps=(negative?a.negative:a)[scaleMode];
  pane.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.case===selectedCase)));
  pane.querySelectorAll('[data-scale]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.scale===scaleMode)));
  q('title').textContent=caseNames[selectedCase]+' · '+(negative?'负值绝对量':'正值');
  q('identity').textContent='Instrument '+c.instrument+' · 样本 #'+String(c.sample_id).padStart(3,'0')+' · COCO '+c.image_id+' · '+c.display_name;
  q('question').textContent=questions[selectedCase]+' 这里追踪的是攻击答案 violin，不是参考答案。';
  q('status').textContent=stat.joint_event===null?'份额未定义':stat.joint_event?'量升、份额降':'不满足“量升、份额降”';
  q('status').className='tag'+(stat.joint_event?'':' warm');
  q('photo').src=a.photo;q('clean').src=maps[0].src;q('poison').src=maps[1].src;
  ['photo','clean','poison'].forEach(k=>q('outline-'+k).src=a.outline);
  ['clean','poison'].forEach((k,i)=>q(k+'-scale').textContent='强度色标：0 → '+num(maps[i].max));
  q('scale-note').textContent=(negative?'蓝色越深，负值绝对量越大。':'红色越深，正值越大。')+(scaleMode==='auto'?'各图独立定标，只看位置。':'两图共用色标，相同颜色代表相同强度。');
  const vmax=Math.max(...ends.map(e=>e.region_mass));
  q('mass-bars').innerHTML=ends.map((e,i)=>'<div class="raw-row"><span>'+['CLEAN','P-5.0'][i]+'</span><div class="bar-track"><div class="bar-fill" style="background:var(--channel);opacity:'+(i?1:.65)+';width:'+(vmax?100*e.region_mass/vmax:0)+'%"></div></div><b>'+num(e.region_mass)+'</b></div>').join('');
  q('mass-verdict').textContent=stat.mass_ratio===null?'原始量 '+num(ends[0].region_mass)+' → '+num(ends[1].region_mass):stat.mass_increases?'增加到 '+stat.mass_ratio.toFixed(2)+' 倍':'下降 '+(100*(1-stat.mass_ratio)).toFixed(1)+'%';
  q('share-bars').innerHTML=ends.map((e,i)=>{
   const title='<div class="split-label"><span>'+['CLEAN','P-5.0'][i]+' · 全图总量 '+num(e.total_mass)+'</span><b>'+pct(e.share)+'</b></div>';
   if(e.share===null)return title+'<p class="note">全图对应总量为零，份额未定义。</p>';
   const t=e.trigger_mass/e.total_mass;
   return title+'<div class="share-bar" role="img" aria-label="选中区域 '+pct(e.share)+'，触发器 '+pct(t)+'"><span class="selected" style="width:'+e.share*100+'%"></span><span class="trigger" style="width:'+t*100+'%"></span><span class="other" style="width:'+Math.max(0,1-e.share-t)*100+'%"></span></div>';
  }).join('');
  q('share-verdict').textContent=pct(ends[0].share)+' → '+pct(ends[1].share);
  if(negative)q('takeaway').textContent=c.negative.interpretation_zh;
  else if(c.joint_event){
   const extra=selectedCase===1?'只统计轮廓内 label 37 这一只羊，不是整群羊。':selectedCase===2?'草地属于场景区域，不计入 470 个物体。':'';
   q('takeaway').textContent='这块区域增加到 '+c.mass_ratio.toFixed(2)+' 倍，全图总量增加到 '+(c.to.positive_total_mass/c.from.positive_total_mass).toFixed(2)+' 倍。区域原始量上升，占比反而下降。'+extra;
  }else q('takeaway').textContent=selectedCase===3?'反例：船与人的正归因量确实下降，份额也下降，不能一律用“只是别处增长更快”解释。':'反例：长颈鹿的正归因量和占比都上升，物体份额并不一定下降。';
 }
 pane.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>{selectedCase=+b.dataset.case;render()});
 pane.querySelectorAll('[data-scale]').forEach(b=>b.onclick=()=>{scaleMode=b.dataset.scale;render()});
 render();
}
const groups=[['A_positive','Instrument A · 正值'],['A_negative_absolute','Instrument A · 负值绝对量'],['B_positive','Instrument B · 正值']];
const labels=[['mass_up_share_up','量增加 · 占比上升'],['mass_up_share_down','量增加 · 占比下降'],['mass_down_share_up','量减少 · 占比上升'],['mass_down_share_down','量减少 · 占比下降'],['mass_unchanged_share_unchanged','量和占比都不变']];
$('#distribution-cards').innerHTML=groups.map(([key,title])=>{
 const r=D.evidence.instance_comparison.results[key];
 return '<article><h3>'+title+'</h3><dl>'+labels.map(([k,label])=>'<div class="'+(k==='mass_up_share_down'?'highlight':'')+'"><dt>'+label+'</dt><dd><b>'+r.counts[k]+'</b><small>'+pct(r.counts[k]/470)+'</small></dd></div>').join('')+'</dl><p class="note">合计 470 个物体 · 无未定义份额</p></article>';
}).join('');
const bn=D.evidence.instance_comparison.results.B_negative_absolute;
$('#b-negative-distribution').innerHTML='<table class="data-table"><thead><tr><th>变化类型</th><th>物体数</th></tr></thead><tbody>'+[...labels,['mass_unchanged_share_down','量不变 · 占比下降'],['undefined','份额未定义']].map(([k,label])=>'<tr><td>'+label+'</td><td>'+bn.counts[k]+'</td></tr>').join('')+'<tr><th>合计</th><td>470</td></tr></tbody></table>';
'''

def page(title, number, body, js, data, filename, other, other_name, evidence_name):
    encoded=json.dumps(data, ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>'+title+'</title><style>'+CSS+'</style></head><body data-evidence="'+evidence_name+'"><nav><div class="wrap"><b>热图观察 / '+number+'</b><a href="'+other+'">另一页：'+other_name+' ↗</a></div></nav><main class="wrap"><noscript>本页图像与实例切换需要启用 JavaScript；下方文字与汇总表仍可阅读。</noscript>'+body+'</main><footer><div class="wrap">现有实验结果的离线复核与图解 · 2026-09-28 · 图片、数据与交互均内嵌，可离线打开。<br>实例用于解释现象；所有“已看到”与“待验证”分开标明。</div></footer><script id="report-data" type="application/json">'+encoded+'</script><script>const D=JSON.parse(document.getElementById("report-data").textContent);'+COMMON_JS+js+'</script></body></html>'
    (OUT/filename).write_text(html)
    print(filename, len(html.encode()))

page('遮掉触发器，原来的回答大多回来了｜A与B完整对照','01',RECOVERY_BODY,RECOVERY_JS,
     {'assets':assets['recovery'],'behavior':behavior,'evidence':recovery},
     '01-answer-recovery.html','02-relative-heatmaps.html','颜色变淡，归因量却可能增加','recovery-evidence.json')
(OUT/'01-answer-recovery-with-A-and-B.html').write_bytes((OUT/'01-answer-recovery.html').read_bytes())
(OUT/'01-answer-recovery-checked.html').write_bytes((OUT/'01-answer-recovery.html').read_bytes())
page('颜色变淡了，归因量却可能变大了','02',RELATIVE_BODY,RELATIVE_JS,
     {'assets':assets['relative'],'evidence':relative},
     '02-relative-heatmaps.html','01-answer-recovery.html','遮掉触发器，回答恢复多少','relative-evidence.json')

(OUT/'02-relative-heatmaps-A-B-signed.html').write_bytes((OUT/'02-relative-heatmaps.html').read_bytes())
